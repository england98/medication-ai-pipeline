import json

import pytest
from pydantic import ValidationError

from vlm_verification.contracts import (
    CandidateKey,
    FrameRef,
    ModelInputFrame,
    ModelInputRecord,
    ObservationSet,
    ProcessingError,
    RequestState,
    ResourceRef,
    TimeRange,
    VerificationRequest,
    VerificationResult,
)

NOW = "2026-09-29T00:00:00.000Z"


def context():
    return dict(session_id="session", user_id="user", scheduled_occurrence_id="occurrence",
                stream_id="stream", target_track_id="target")


def candidate():
    return dict(candidate_id="candidate", candidate_type="E02", candidate_revision=1)


def resource():
    return dict(resource_id="config", locator="config.json", media_type="application/json")


def frame():
    return dict(stream_id="stream", frame_index=0, source_ms=100, session_ms=0,
                width=640, height=480)


def input_frame():
    return dict(input_index=0, clip_frame_index=0, roi_id=None, source_bbox=[0, 0, 1, 1],
                resized_size=[320, 240], padding=[1, 2, 3, 4], input_size=[324, 246])


def request_payload():
    observation = dict(status="OBSERVED", items=[], reason=None)
    return dict(
        schema_version="1.0", request_id="request", context=context(),
        event=dict(context=context(), candidate=candidate(), created_at=NOW, updated_at=NOW,
                   action_range=dict(start_ms=0, end_ms=100), object_track_ids=[], regions=[]),
        detection=dict(context=context(), candidate=candidate(), detector_config=resource(),
                       samples=[dict(frame=frame(), objects=observation, landmarks=observation,
                                     motions=observation)]),
        media=dict(context=context(), candidate=candidate(), requested_range=dict(start_ms=0, end_ms=100),
                   clip=None, unavailable_reason="No video acquired"),
        created_at=NOW, model_id="Qwen/Qwen3-VL-2B-Instruct", model_revision="fixed-commit",
        prompt_version="1.0", execution_config=resource(),
    )


def terminal_payload():
    return dict(
        schema_version="1.0", request_id="request", context=context(), candidate=candidate(),
        action_range=dict(start_ms=0, end_ms=100), clip_id=None, input_record=None,
        processing_status="NOT_RUN", result=None, verification=None,
        verification_rule_version="1.0", raw_output=None,
        error=dict(stage="INPUT", code="MISSING_VIDEO", message="No video acquired"),
        response_received_at=None, finished_at=NOW,
    )


def test_nullable_field_is_still_required():
    payload = frame()
    del payload["session_ms"]
    with pytest.raises(ValidationError):
        FrameRef.model_validate(payload)


@pytest.mark.parametrize("bad", ["0", 0.0, True, -1])
def test_integer_fields_do_not_coerce(bad):
    payload = frame()
    payload["frame_index"] = bad
    with pytest.raises(ValidationError):
        FrameRef.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("bad", [0, True, "1"])
def test_revision_is_strict_positive_integer(bad):
    payload = candidate()
    payload["candidate_revision"] = bad
    with pytest.raises(ValidationError):
        CandidateKey.model_validate(payload)


@pytest.mark.parametrize("bbox", [[0, 0, 0, 1], [0, 1, 1, 0], [-0.1, 0, 1, 1],
                                  [0, 0, 1.1, 1], [False, 0, 1, 1], ["0", 0, 1, 1],
                                  [0, 0, float("nan"), 1], [0, 0, float("inf"), 1]])
def test_boxes_are_valid_normalized_finite_coordinates(bbox):
    payload = input_frame()
    payload["source_bbox"] = bbox
    with pytest.raises(ValidationError):
        ModelInputFrame.model_validate(payload)


def test_input_padding_and_order_are_exact():
    payload = input_frame()
    ModelInputFrame.model_validate(payload)
    payload["input_size"] = [320, 240]
    with pytest.raises(ValidationError):
        ModelInputFrame.model_validate(payload)
    payload = input_frame()
    payload["input_index"] = 1
    with pytest.raises(ValidationError):
        ModelInputRecord(schema_version="1.0", request_id="request", rendered_prompt="prompt", frames=[payload])


@pytest.mark.parametrize("instant", ["2026-09-29T00:00:00Z", "2026-09-29T00:00:00.000+00:00",
                                     "2026-02-30T00:00:00.000Z", "2026-09-29T25:00:00.000Z"])
def test_instants_require_valid_utc_millisecond_format(instant):
    payload = request_payload()
    payload["created_at"] = instant
    with pytest.raises(ValidationError):
        VerificationRequest.model_validate(payload)


def test_cross_object_link_mismatch_is_retained_for_service_not_run():
    payload = request_payload()
    payload["media"]["context"]["target_track_id"] = "other-person"
    payload["detection"]["candidate"]["candidate_revision"] = 2
    parsed = VerificationRequest.model_validate_json(json.dumps(payload))
    assert parsed.media.context.target_track_id == "other-person"
    assert parsed.detection.candidate.candidate_revision == 2


def test_open_candidate_is_not_a_verification_request():
    payload = request_payload()
    payload["event"]["action_range"]["end_ms"] = None
    with pytest.raises(ValidationError):
        VerificationRequest.model_validate(payload)


@pytest.mark.parametrize("change", ["missing", "unknown", "nested"])
def test_schema_version_is_explicit_on_top_level_only(change):
    payload = request_payload()
    if change == "missing":
        del payload["schema_version"]
    elif change == "unknown":
        payload["schema_version"] = "2.0"
    else:
        payload["event"]["schema_version"] = "1.0"
    with pytest.raises(ValidationError):
        VerificationRequest.model_validate(payload)


def test_unknown_contract_fields_are_rejected():
    with pytest.raises(ValidationError):
        ResourceRef(**resource(), secret="unexpected")


def test_unavailable_observations_differ_from_empty_observed_items():
    assert ObservationSet[str](status="OBSERVED", items=[], reason=None).items == []
    assert ObservationSet[str](status="UNAVAILABLE", items=[], reason="Occluded").reason == "Occluded"
    with pytest.raises(ValidationError):
        ObservationSet[str](status="UNAVAILABLE", items=[], reason=None)
    with pytest.raises(ValidationError):
        ObservationSet[str](status="UNAVAILABLE", items=["value"], reason="Occluded")


def test_terminal_not_run_is_not_a_verification():
    payload = terminal_payload()
    parsed = VerificationResult.model_validate(payload)
    assert parsed.verification is None
    payload["verification"] = "UNCERTAIN"
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


def test_error_preserves_empty_raw_response_and_receipt_time():
    payload = terminal_payload()
    payload.update(processing_status="ERROR", raw_output="", response_received_at=NOW,
                   error=dict(stage="PARSE", code="INVALID_RESPONSE", message="Empty response"))
    assert VerificationResult.model_validate(payload).raw_output == ""
    payload["response_received_at"] = None
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


def test_not_run_cannot_contain_model_input():
    payload = terminal_payload()
    payload["input_record"] = resource()
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


def test_error_stage_and_code_must_match():
    with pytest.raises(ValidationError):
        ProcessingError(stage="INPUT", code="INVALID_RESPONSE", message="Mismatch")


def test_request_state_terminal_status_and_time_order():
    RequestState(schema_version="1.0", request_id="r", request_status="FINISHED",
                 started_at=None, finished_at=NOW, processing_status="NOT_RUN")
    with pytest.raises(ValidationError):
        RequestState(schema_version="1.0", request_id="r", request_status="RUNNING",
                     started_at=None, finished_at=None, processing_status=None)
    with pytest.raises(ValidationError):
        RequestState(schema_version="1.0", request_id="r", request_status="FINISHED",
                     started_at="2026-09-29T00:00:00.001Z", finished_at=NOW, processing_status="ERROR")


def test_time_range_does_not_reverse():
    with pytest.raises(ValidationError):
        TimeRange(start_ms=2, end_ms=1)
