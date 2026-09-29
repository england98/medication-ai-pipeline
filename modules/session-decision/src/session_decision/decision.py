"""Accumulate received results only, then apply the documented E02 completion rule."""

import hashlib
import json
import re
from pathlib import Path

from medication_contracts import (
    CompletionRecord,
    ConsistencyCheck,
    ExcludedResult,
    ReceivedResult,
    ResourceRef,
    SupportingEvidence,
    VerificationRequest,
    VerificationResult,
)
from medication_contracts.records import append_json, key_path, new_id, utc_now, write_json
from pydantic import BaseModel, ConfigDict, Field
from vlm_verification.response import derive_verification


class DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    rule_version: str = "session-completion-v1"
    support_window_ms: int = Field(default=30000, ge=0)
    # Conservative, auditable explicit descriptions, never detector class labels.
    non_medication_patterns: list[str] = Field(default_factory=lambda: [
        r"\b(?:not (?:a |an )?(?:pill|medication|medicine)|candy|food|snack|cup|bottle)\b",
        r"사탕|음식|간식|물병|컵|약이 아(?:닌|님|니다)",
    ])


class SessionDecision:
    def __init__(self, session, directory: Path, config: DecisionConfig | None = None):
        self.session = session.model_copy(deep=True)
        self.directory = Path(directory)
        self.config = config or DecisionConfig()
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.config.non_medication_patterns]
        self.results = {}
        self.requests = {}
        self.invalid = {}
        self.completion = None
        config_path = self.directory / "decision-config.json"
        write_json(config_path, self.config)
        self.config_ref = ResourceRef(resource_id="decision-config-" + hashlib.sha256(str(config_path.resolve()).encode()).hexdigest(),
                                      locator=str(config_path.resolve()), media_type="application/json")
        journal = self.directory / "received.jsonl"
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                rid = json.loads(line)["request_id"]
                self.results[rid] = VerificationResult.model_validate_json(
                    key_path(self.directory / "received", rid).read_text(encoding="utf-8"))
                self.requests[rid] = VerificationRequest.model_validate_json(
                    key_path(self.directory / "requests", rid).read_text(encoding="utf-8"))
                admission = key_path(self.directory / "admission", rid)
                if not admission.exists():
                    raise ValueError("Cannot restore result without admission validation record")
                reason = json.loads(admission.read_text(encoding="utf-8"))["reason"]
                if reason:
                    self.invalid[rid] = reason
        completion_path = self.directory / "completion.json"
        if completion_path.exists():
            self.completion = CompletionRecord.model_validate_json(completion_path.read_text(encoding="utf-8"))
            if self.completion.context != self.session.context:
                raise ValueError("Persisted completion belongs to a different session")

    def receive(self, result: VerificationResult, request: VerificationRequest):
        result = VerificationResult.model_validate_json(result.model_dump_json())
        request = VerificationRequest.model_validate_json(request.model_dump_json())
        rid = result.request_id
        if rid in self.results:
            if self.results[rid] != result or self.requests[rid] != request:
                raise ValueError("Conflicting duplicate result/request")
            return self.completion or self._evaluate()
        if result.context.session_id != self.session.context.session_id:
            raise ValueError("Result belongs to another session")
        expected_clip = request.media.clip.clip_id if request.media.clip else None
        reason = None
        if result.context != self.session.context:
            reason = "TARGET_MISMATCH"
        elif (rid != request.request_id or result.context != request.context
              or result.candidate != request.event.candidate
              or result.action_range != request.event.action_range or result.clip_id != expected_clip):
            reason = "RESULT_REQUEST_LINK_MISMATCH"
        elif any(value.context != request.context or value.candidate != request.event.candidate
                 for value in (request.event, request.detection, request.media)):
            reason = "REQUEST_LINK_MISMATCH"
        elif result.processing_status == "OK":
            try:
                derived = derive_verification(result.result, result.candidate.candidate_type)
                if derived != result.verification:
                    reason = "VERIFICATION_RULE_MISMATCH"
            except (ValueError, KeyError):
                reason = "VERIFICATION_RULE_MISMATCH"
        self.results[rid], self.requests[rid] = result, request
        if reason:
            self.invalid[rid] = reason
        write_json(key_path(self.directory / "received", rid), result)
        write_json(key_path(self.directory / "requests", rid), request)
        write_json(key_path(self.directory / "admission", rid), {"request_id": rid, "reason": reason})
        append_json(self.directory / "received.jsonl", ReceivedResult(
            schema_version="1.0", request_id=rid, received_at=utc_now()))
        if self.completion is not None:
            append_json(self.directory / "late-results.jsonl", {"request_id": rid})
            return self.completion
        return self._evaluate()

    def _same_action(self, a, b):
        if a.candidate.candidate_id == b.candidate.candidate_id:
            return True
        objects_a = set(self.requests[a.request_id].event.object_track_ids)
        objects_b = set(self.requests[b.request_id].event.object_track_ids)
        overlaps = max(a.action_range.start_ms, b.action_range.start_ms) <= min(
            a.action_range.end_ms, b.action_range.end_ms)
        return bool(objects_a & objects_b) and overlaps

    def _conflict(self, core, other):
        if other.request_id in self.invalid or other.processing_status != "OK":
            return False
        if not self._same_action(core, other):
            return False
        checks = other.result.checks
        if other.candidate.candidate_type == "E02" and any(
            checks.get(name) == "NO" for name in (
                "object_is_medication", "object_transfer_into_mouth_visible")
        ):
            return True
        return any(p.search(other.result.object_description) for p in self.patterns)

    def _evaluate(self):
        ordered = sorted(self.results.values(), key=lambda r: (
            r.action_range.start_ms, r.action_range.end_ms, r.request_id))
        confirmed = [r for r in ordered if r.request_id not in self.invalid
                     and r.processing_status == "OK" and r.verification == "CONFIRMED"]
        cores = []
        conflicts = {}
        seen_candidates = set()
        for core in confirmed:
            if core.candidate.candidate_type != "E02":
                continue
            opposing = [r.request_id for r in ordered if r.request_id != core.request_id
                        and self._conflict(core, r)]
            if opposing:
                conflicts[core.request_id] = opposing
            elif core.candidate.candidate_id not in seen_candidates:
                cores.append(core)
                seen_candidates.add(core.candidate.candidate_id)
        append_json(self.directory / "evaluations.jsonl", {
            "schema_version": "1.0", "evaluated_at": utc_now(),
            "received_request_ids": list(self.results), "conflicts": conflicts,
            "eligible_core_request_ids": [c.request_id for c in cores]})
        if not cores:
            return None
        core_ids = [c.request_id for c in cores]
        supporting = []
        for result in confirmed:
            if result.candidate.candidate_type == "E02":
                continue
            objects = set(self.requests[result.request_id].event.object_track_ids)
            related = [c.request_id for c in cores
                       if objects & set(self.requests[c.request_id].event.object_track_ids)
                       and max(0, result.action_range.start_ms - c.action_range.end_ms,
                               c.action_range.start_ms - result.action_range.end_ms)
                       <= self.config.support_window_ms]
            if related:
                supporting.append(SupportingEvidence(
                    request_id=result.request_id, related_core_request_ids=related,
                    relation_reason="Shared observed object track within configured action window"))
        used = set(core_ids) | {s.request_id for s in supporting}
        excluded = []
        for result in ordered:
            if result.request_id not in used:
                reason = self.invalid.get(result.request_id) or (
                    "UNRESOLVED_CONFLICT" if result.request_id in conflicts else
                    f"{result.processing_status}/{result.verification or 'NO_VERIFICATION'}; "
                    "not selected as linked completion evidence")
                excluded.append(ExcludedResult(request_id=result.request_id, reason=reason))
        self.completion = CompletionRecord(
            schema_version="1.0", completion_id=new_id("completion"), context=self.session.context,
            session_result="COMPLETE", decided_at=utc_now(), received_request_ids=list(self.results),
            core_request_ids=core_ids, supporting_evidence=supporting, excluded_results=excluded,
            consistency=ConsistencyCheck(target_match=True, unresolved_conflict=False,
                checked_request_ids=list(self.results),
                reason="E02 identity/transfer checks confirmed; no received same-object/action conflict"),
            decision_rule_version=self.config.rule_version, decision_config=self.config_ref)
        write_json(self.directory / "completion.json", self.completion)
        return self.completion
