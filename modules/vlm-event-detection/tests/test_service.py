"""Service integration boundaries tested without downloading a model."""

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from vlm_verification.config import ExecutionConfig
from vlm_verification.contracts import (
    ModelInputRecord,
    RequestState,
    SessionContext,
    VerificationRequest,
)
from vlm_verification.media import TargetUnresolvedError
from vlm_verification.service import RecordConflict, VerificationService, VerificationWorker


class FakeBackend:
    def __init__(self, raw=None, prepare_error=None, generate_error=None):
        self.calls = 0
        self.raw = raw if raw is not None else json.dumps({
            "checks": {"object_is_medication": "UNKNOWN",
                       "object_transfer_into_mouth_visible": "YES"},
            "object_description": "Occluded small object",
            "observed_action": "Object transferred to mouth",
            "evidence": [
                {"check": "object_is_medication", "input_indices": [0],
                 "description": "Object hidden by fingers"},
                {"check": "object_transfer_into_mouth_visible", "input_indices": [1],
                 "description": "Object enters mouth"},
            ],
        })
        self.prepare_error = prepare_error
        self.generate_error = generate_error

    def prepare(self, request, config):
        if self.prepare_error:
            raise self.prepare_error
        return SimpleNamespace(record=ModelInputRecord(
            schema_version="1.0", request_id=request.request_id,
            rendered_prompt="Test-only rendered prompt; no real model involved",
            frames=[dict(input_index=i, clip_frame_index=i, roi_id=None,
                         source_bbox=[0.0, 0.0, 1.0, 1.0], resized_size=[64, 64],
                         padding=[0, 0, 0, 0], input_size=[64, 64]) for i in range(2)]),
            payload=None)

    def generate(self, prepared, config):
        self.calls += 1
        if self.generate_error:
            raise self.generate_error
        return self.raw


@pytest.fixture
def case(tmp_path):
    config = ExecutionConfig(model_revision="a" * 40)
    (tmp_path / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    (tmp_path / "clip.mp4").write_bytes(b"fake backend does not decode video")
    context = dict(session_id="s1", user_id="u1", scheduled_occurrence_id="o1",
                   stream_id="stream1", target_track_id="target1")
    candidate = dict(candidate_id="c1", candidate_type="E02", candidate_revision=1)
    timestamp = "2026-01-01T00:00:00.000Z"
    frames = [dict(stream_id="stream1", frame_index=i, source_ms=1000+i*100,
                   session_ms=i*100, width=64, height=64) for i in range(2)]
    unavailable = dict(status="UNAVAILABLE", items=[], reason="Test fixture")
    session = SessionContext.model_validate(dict(
        schema_version="1.0", context=context, session_revision=1, auth_id="a1",
        scheduled_at=timestamp, started_at=timestamp, source_time_origin_ms=1000,
        session_status="ACTIVE", tracking=None, completion_id=None, closed_at=None))
    request = VerificationRequest.model_validate(dict(
        schema_version="1.0", request_id="r1", context=context,
        event=dict(context=context, candidate=candidate, created_at=timestamp,
                   updated_at=timestamp, action_range=dict(start_ms=0, end_ms=100),
                   object_track_ids=[], regions=[dict(frame=frames[0], region_type="TARGET",
                       bbox=[0.0, 0.0, 1.0, 1.0], object_track_id=None)]),
        detection=dict(context=context, candidate=candidate,
                       detector_config=dict(resource_id="d1", locator="detector.json",
                                            media_type="application/json"),
                       samples=[dict(frame=frames[0], objects=unavailable,
                                     landmarks=unavailable, motions=unavailable)]),
        media=dict(context=context, candidate=candidate,
                   requested_range=dict(start_ms=0, end_ms=100), unavailable_reason=None,
                   clip=dict(clip_id="clip1", video=dict(resource_id="v1", locator="clip.mp4",
                             media_type="video/mp4"), actual_range=dict(start_ms=0, end_ms=100),
                             frames=[dict(clip_frame_index=i, clip_ms=i*100, source=f)
                                     for i, f in enumerate(frames)], rois=[])),
        created_at=timestamp, model_id=config.model_id, model_revision=config.model_revision,
        prompt_version=config.prompt_version, execution_config=dict(resource_id="config1",
                       locator="config.json", media_type="application/json")))
    return request, session, config, tmp_path


def execute(case, backend=None):
    request, session, config, directory = case
    backend = backend or FakeBackend()
    service = VerificationService(backend, config, directory / "outputs", directory)
    return service.verify(request, session), service, backend


def test_uncertainty_is_ok_and_input_provenance_is_saved(case):
    result, service, backend = execute(case)
    assert result.processing_status == "OK"
    assert result.verification == "UNCERTAIN"
    record = ModelInputRecord.model_validate_json(Path(result.input_record.locator).read_text())
    assert record.request_id == result.request_id
    assert [f.clip_frame_index for f in record.frames] == [0, 1]
    states = [RequestState.model_validate_json(line)
              for line in (Path(result.input_record.locator).parent / "states.jsonl").read_text().splitlines()]
    assert [s.request_status for s in states] == ["QUEUED", "RUNNING", "FINISHED"]
    assert states[-1].processing_status == "OK"
    # A duplicate survives a new service instance; inference is not invoked again.
    replay = VerificationService(backend, case[2], service.output_dir, case[3])
    assert replay.verify(case[0], case[1]) == result
    assert backend.calls == 1


@pytest.mark.parametrize("mutation,code", [
    (lambda r: setattr(r.media.context, "user_id", "other"), "LINK_MISMATCH"),
    (lambda r: setattr(r.detection.candidate, "candidate_revision", 2), "LINK_MISMATCH"),
    (lambda r: setattr(r.media.clip.frames[0].source, "source_ms", 999), "LINK_MISMATCH"),
    (lambda r: setattr(r.media.clip.frames[1], "clip_frame_index", 8), "LINK_MISMATCH"),
    (lambda r: setattr(r.media.clip.actual_range, "end_ms", 99), "LINK_MISMATCH"),
    (lambda r: setattr(r.event, "regions", []), "TARGET_UNRESOLVED"),
    (lambda r: setattr(r.media.clip.video, "locator", "missing.mp4"), "MISSING_VIDEO"),
])
def test_invalid_link_is_not_run(case, mutation, code):
    mutation(case[0])
    result, _, backend = execute(case)
    assert result.processing_status == "NOT_RUN"
    assert result.error.code == code
    assert result.verification is result.result is result.input_record is None
    assert result.raw_output is result.response_received_at is None
    assert backend.calls == 0


def test_missing_media_is_explicit_not_run(case):
    case[0].media = case[0].media.model_copy(update={
        "clip": None, "unavailable_reason": "Video acquisition finished without media"})
    result, _, backend = execute(case)
    assert result.processing_status == "NOT_RUN"
    assert result.clip_id is None
    assert backend.calls == 0


@pytest.mark.parametrize("backend,status,code,has_input", [
    (FakeBackend(prepare_error=RuntimeError("decode failed")), "ERROR", "PREPROCESS_FAILED", False),
    (FakeBackend(generate_error=RuntimeError("out of memory")), "ERROR", "INFERENCE_FAILED", True),
    (FakeBackend(raw="not JSON"), "ERROR", "INVALID_RESPONSE", True),
    (FakeBackend(raw="{}"), "ERROR", "MISSING_FIELD", True),
    (FakeBackend(raw=""), "ERROR", "INVALID_RESPONSE", True),
])
def test_technical_failures_preserve_only_available_artifacts(case, backend, status, code, has_input):
    result, _, _ = execute(case, backend)
    assert result.processing_status == status
    assert result.error.code == code
    assert (result.input_record is not None) == has_input
    assert result.verification is result.result is None
    if code in {"INVALID_RESPONSE", "MISSING_FIELD"}:
        assert result.raw_output == backend.raw
        assert result.response_received_at is not None


def test_same_request_id_different_content_cannot_overwrite(case):
    first, service, backend = execute(case)
    case[0].event.candidate.candidate_revision = 2
    with pytest.raises(RecordConflict):
        service.verify(case[0], case[1])
    saved = next(service.output_dir.glob("*/result.json"))
    assert json.loads(saved.read_text())["candidate"]["candidate_revision"] == 1
    assert backend.calls == 1


def test_separate_candidate_is_not_answer_cache(case):
    _, service, backend = execute(case)
    request = case[0].model_copy(deep=True)
    request.request_id = "r2"
    for value in (request.event, request.detection, request.media):
        value.candidate.candidate_id = "c2"
    result = service.verify(request, case[1])
    assert result.candidate.candidate_id == "c2"
    assert backend.calls == 2


def test_completed_request_replayed_after_session_update_keeps_original_result(case):
    first, service, backend = execute(case)
    case[1].session_revision = 2
    assert service.verify(case[0], case[1]) == first
    assert backend.calls == 1


def test_local_file_uri_is_supported(case):
    case[0].media.clip.video.locator = (case[3] / "clip.mp4").as_uri()
    case[0].execution_config.locator = (case[3] / "config.json").as_uri()
    result, _, _ = execute(case)
    assert result.processing_status == "OK"


def test_target_cannot_be_sampled_is_not_run(case):
    result, _, backend = execute(case, FakeBackend(prepare_error=TargetUnresolvedError("No target")))
    assert result.processing_status == "NOT_RUN"
    assert result.error.code == "TARGET_UNRESOLVED"
    assert backend.calls == 0


def test_backend_cannot_crop_away_target_and_claim_full_context(case):
    class CroppedBackend(FakeBackend):
        def prepare(self, request, config):
            prepared = super().prepare(request, config)
            for frame in prepared.record.frames:
                frame.source_bbox = [0.0, 0.0, 0.2, 0.2]
            return prepared

    result, _, backend = execute(case, CroppedBackend())
    assert result.processing_status == "ERROR"
    assert result.error.code == "PREPROCESS_FAILED"
    assert backend.calls == 0


def test_bounded_worker_does_not_block_capture(case):
    entered, release = Event(), Event()

    class BlockingBackend(FakeBackend):
        def generate(self, prepared, config):
            entered.set()
            assert release.wait(5)
            return super().generate(prepared, config)

    service = VerificationService(BlockingBackend(), case[2], case[3] / "out", case[3])
    worker = VerificationWorker(service, max_pending=1)
    try:
        future = worker.submit(case[0], case[1])
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="queue is full"):
            worker.submit(case[0], case[1])
        release.set()
        assert future.result(timeout=5).processing_status == "OK"
    finally:
        release.set()
        worker.shutdown()
