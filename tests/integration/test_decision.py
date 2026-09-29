
import pytest
from conftest import ROOT
from medication_contracts import (
    CHECKS_BY_EVENT,
    ModelResponse,
    ProcessingError,
    ResourceRef,
    TimeRange,
    VerificationRequest,
    VerificationResult,
)
from medication_contracts.records import utc_now
from session_decision import SessionDecision
from session_video import SessionManager
from vlm_verification.response import derive_verification


def evidence(session, rid, kind="E02", values=("YES", "YES"), *, objects=None,
             action=(0, 100), status="OK", description="a medication tablet", cid=None):
    request = VerificationRequest.model_validate_json((ROOT / "modules/vlm-event-detection/examples/request.e02.json").read_text())
    request.request_id = rid
    for value in (request, request.event, request.detection, request.media):
        value.context = session.context.model_copy(deep=True)
    key = request.event.candidate.model_copy(update={"candidate_id": cid or rid, "candidate_type": kind})
    for value in (request.event, request.detection, request.media):
        value.candidate = key.model_copy(deep=True)
    request.event.object_track_ids = ["object-one"] if objects is None else objects
    request.event.action_range = TimeRange(start_ms=action[0], end_ms=action[1])
    checks = dict(zip(CHECKS_BY_EVENT[kind], values))
    response = ModelResponse(checks=checks, object_description=description,
        observed_action="Visible action", evidence=[{"check": check, "input_indices": [0], "description": "Visible frame"} for check in checks])
    now = utc_now()
    result = VerificationResult(schema_version="1.0", request_id=rid, context=session.context,
        candidate=key, action_range=request.event.action_range, clip_id=request.media.clip.clip_id,
        input_record=ResourceRef(resource_id="input", locator="input.json", media_type="application/json") if status == "OK" else None,
        processing_status=status, result=response if status == "OK" else None,
        verification=derive_verification(response, kind) if status == "OK" else None,
        verification_rule_version="vlm-events-rules-v1", raw_output=response.model_dump_json() if status == "OK" else None,
        error=None if status == "OK" else ProcessingError(stage="INFERENCE" if status == "ERROR" else "INPUT",
            code="INFERENCE_FAILED" if status == "ERROR" else "MISSING_VIDEO", message="test failure"),
        response_received_at=now if status == "OK" else None, finished_at=now)
    return result, request


@pytest.mark.parametrize("kind,values,status", [
    ("E01", ("YES", "YES"), "OK"), ("E03", ("YES", "YES"), "OK"),
    ("E02", ("UNKNOWN", "YES"), "OK"), ("E02", ("NO", "YES"), "OK"),
    ("E02", ("YES", "YES"), "ERROR"), ("E02", ("YES", "YES"), "NOT_RUN"),
])
def test_support_uncertainty_and_failures_never_complete(tmp_path, session, kind, values, status):
    decision = SessionDecision(session, tmp_path)
    assert decision.receive(*evidence(session, "request", kind, values, status=status)) is None
    assert len(decision.results) == 1


def test_core_alone_completes_once_and_session_closes(tmp_path, session):
    decision = SessionDecision(session, tmp_path / "decision")
    pair = evidence(session, "core", objects=[])
    completion = decision.receive(*pair)
    assert completion.core_request_ids == ["core"]
    assert completion.supporting_evidence == []
    assert decision.receive(*pair) == completion
    assert len((tmp_path / "decision/received.jsonl").read_text().splitlines()) == 1
    manager = SessionManager(tmp_path / "session")
    manager.session = session
    assert manager.close(completion).session_status == "CLOSED"
    assert manager.close(completion).completion_id == completion.completion_id
    assert manager.view().session_result == "COMPLETE"


def test_arrival_order_does_not_replace_action_order_or_object_identity(tmp_path, session):
    decision = SessionDecision(session, tmp_path)
    assert decision.receive(*evidence(session, "later-water", "E03", action=(3000, 4000), objects=["cup"])) is None
    assert decision.receive(*evidence(session, "earlier-prep", "E01", action=(0, 100), objects=["object-one"])) is None
    completion = decision.receive(*evidence(session, "core", action=(500, 700)))
    assert completion.received_request_ids == ["later-water", "earlier-prep", "core"]
    assert [s.request_id for s in completion.supporting_evidence] == ["earlier-prep"]
    assert [e.request_id for e in completion.excluded_results] == ["later-water"]


@pytest.mark.parametrize("same_object,same_action,blocked", [(True, True, True), (True, False, False), (False, True, False)])
def test_conflict_scoped_to_same_object_and_action(tmp_path, session, same_object, same_action, blocked):
    decision = SessionDecision(session, tmp_path)
    decision.receive(*evidence(session, "negative", values=("NO", "YES"),
        objects=["object-one"] if same_object else ["other"], action=(0, 100) if same_action else (200, 300)))
    completion = decision.receive(*evidence(session, "core"))
    assert (completion is None) == blocked


def test_related_support_description_conflict_and_unrelated_new_action(tmp_path, session):
    decision = SessionDecision(session, tmp_path)
    decision.receive(*evidence(session, "support", "E01", description="a candy"))
    assert decision.receive(*evidence(session, "core")) is None
    assert decision.receive(*evidence(session, "new-core", action=(5000, 5100), objects=["new-pill"])) is not None


def test_wrong_target_excluded_and_changed_duplicate_rejected(tmp_path, session):
    decision = SessionDecision(session, tmp_path)
    result, request = evidence(session, "wrong")
    result.context = result.context.model_copy(update={"target_track_id": "other"})
    assert decision.receive(result, request) is None
    completion = decision.receive(*evidence(session, "core"))
    assert completion.excluded_results[0].reason == "TARGET_MISMATCH"
    changed = result.model_copy(update={"clip_id": "different"})
    with pytest.raises(ValueError, match="duplicate"):
        decision.receive(changed, request)


def test_completion_partition_rejects_missing_received(tmp_path, session):
    completion = SessionDecision(session, tmp_path).receive(*evidence(session, "core"))
    payload = completion.model_dump(mode="json")
    payload["received_request_ids"].append("missing")
    with pytest.raises(ValueError, match="partition"):
        type(completion).model_validate(payload)


def test_restart_preserves_receipts_and_completion(tmp_path, session):
    decision = SessionDecision(session, tmp_path)
    negative = evidence(session, "uncertain", values=("UNKNOWN", "UNKNOWN"))
    assert decision.receive(*negative) is None
    restored = SessionDecision(session, tmp_path)
    assert restored.receive(*negative) is None
    pair = evidence(session, "core")
    completion = restored.receive(*pair)
    replay = SessionDecision(session, tmp_path)
    assert replay.receive(*pair) == completion
    assert len((tmp_path / "received.jsonl").read_text().splitlines()) == 2
    assert replay.completion.completion_id == completion.completion_id


def test_recover_completion_written_before_session_state(tmp_path, session):
    completion = SessionDecision(session, tmp_path / "decision").receive(*evidence(session, "core"))
    manager = SessionManager(tmp_path / "session")
    manager.session = session.model_copy(deep=True)
    manager.close(completion)
    closure_before = (tmp_path / "session/closure.json").read_bytes()
    # Simulate a crash between writing closure and persisting the closed context.
    restored = SessionManager(tmp_path / "session")
    restored.session = session.model_copy(deep=True)
    restored.completion = completion
    assert restored.close(completion).session_status == "CLOSED"
    assert (tmp_path / "session/closure.json").read_bytes() == closure_before
