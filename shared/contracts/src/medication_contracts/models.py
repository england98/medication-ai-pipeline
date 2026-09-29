"""Strict JSON contracts from architecture sections 2-4 and 2-7.

Cross-message identity and source-time linkage are checked by the service so an
identifiable request can retain a NOT_RUN result for invalid input links.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

NonEmptyStr = Annotated[StrictStr, StringConstraints(min_length=1, pattern=r"\S")]
ID = NonEmptyStr
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
FiniteNumber = Annotated[StrictFloat, Field(allow_inf_nan=False)]
UnitNumber = Annotated[FiniteNumber, Field(ge=0, le=1)]
EventType = Literal["E01", "E02", "E03"]
CheckValue = Literal["YES", "NO", "UNKNOWN"]
VerificationStatus = Literal["CONFIRMED", "REFUTED", "UNCERTAIN"]
ProcessingStatus = Literal["OK", "ERROR", "NOT_RUN"]

CHECKS_BY_EVENT: dict[str, tuple[str, str]] = {
    "E01": ("handling_visible", "contents_removed_visible"),
    "E02": ("object_is_medication", "object_transfer_into_mouth_visible"),
    "E03": ("container_reaches_mouth", "drinking_motion_visible"),
}


def _valid_instant(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    return value


Instant = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"),
    AfterValidator(_valid_instant),
]


def _valid_bbox(value: list[float]) -> list[float]:
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("bbox minimum coordinates must be less than maximum coordinates")
    return value


Point = Annotated[list[UnitNumber], Field(min_length=2, max_length=2)]
BBox = Annotated[
    list[UnitNumber], Field(min_length=4, max_length=4), AfterValidator(_valid_bbox)
]
PixelSize = Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]
Padding = Annotated[list[NonNegativeInt], Field(min_length=4, max_length=4)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class ExchangeRecord(Contract):
    schema_version: Literal["1.0"]


class SessionKey(Contract):
    session_id: ID
    user_id: ID
    scheduled_occurrence_id: ID
    stream_id: ID
    target_track_id: ID


class CandidateKey(Contract):
    candidate_id: ID
    candidate_type: EventType
    candidate_revision: PositiveInt


class ResourceRef(Contract):
    resource_id: ID
    locator: NonEmptyStr
    media_type: NonEmptyStr


class TimeRange(Contract):
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt | None

    @model_validator(mode="after")
    def ordered(self) -> TimeRange:
        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms must be at least start_ms")
        return self


class FrameRef(Contract):
    stream_id: ID
    frame_index: NonNegativeInt
    source_ms: NonNegativeInt
    session_ms: NonNegativeInt | None
    width: PositiveInt
    height: PositiveInt


class FrameInput(Contract):
    frame: FrameRef
    image: ResourceRef


class TrackingUpdate(Contract):
    context: SessionKey
    frame: FrameRef
    tracking_status: Literal["TRACKED", "UNRESOLVED"]
    target_bbox: BBox | None

    @model_validator(mode="after")
    def bbox_matches_status(self) -> TrackingUpdate:
        if (self.tracking_status == "TRACKED") != (self.target_bbox is not None):
            raise ValueError("target_bbox is required exactly when tracking_status is TRACKED")
        return self


class SessionContext(ExchangeRecord):
    context: SessionKey
    session_revision: PositiveInt
    auth_id: ID
    scheduled_at: Instant
    started_at: Instant
    source_time_origin_ms: NonNegativeInt
    session_status: Literal["ACTIVE", "CLOSED"]
    tracking: TrackingUpdate | None
    completion_id: ID | None
    closed_at: Instant | None

    @model_validator(mode="after")
    def closure_matches_status(self) -> SessionContext:
        if self.session_status == "ACTIVE":
            if self.completion_id is not None or self.closed_at is not None:
                raise ValueError("ACTIVE sessions cannot have closure fields")
        elif self.completion_id is None or self.closed_at is None:
            raise ValueError("CLOSED sessions require completion_id and closed_at")
        if self.closed_at is not None and self.closed_at < self.started_at:
            raise ValueError("closed_at precedes started_at")
        return self


class CandidateRegion(Contract):
    frame: FrameRef
    region_type: Literal["TARGET", "HAND", "MOUTH", "OBJECT", "CONTEXT"]
    bbox: BBox
    object_track_id: ID | None


class EventCandidate(Contract):
    context: SessionKey
    candidate: CandidateKey
    created_at: Instant
    updated_at: Instant
    action_range: TimeRange
    object_track_ids: list[ID]
    regions: list[CandidateRegion]

    @model_validator(mode="after")
    def timestamps_ordered(self) -> EventCandidate:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")
        return self


T = TypeVar("T")


class ObservationSet(Contract, Generic[T]):
    status: Literal["OBSERVED", "UNAVAILABLE"]
    items: list[T]
    reason: NonEmptyStr | None

    @model_validator(mode="after")
    def availability_matches_items(self) -> ObservationSet[T]:
        if self.status == "UNAVAILABLE":
            if self.items or self.reason is None:
                raise ValueError("UNAVAILABLE observations require empty items and a reason")
        elif self.reason is not None:
            raise ValueError("OBSERVED observations require reason=null")
        return self


class DetectedObject(Contract):
    object_track_id: ID | None
    class_label: NonEmptyStr
    score: UnitNumber
    bbox: BBox
    associated_parts: list[NonEmptyStr]


class Landmark(Contract):
    part: NonEmptyStr
    index: NonNegativeInt
    point: Point
    score: UnitNumber | None


class MotionFeature(Contract):
    name: NonEmptyStr
    value: FiniteNumber
    unit: NonEmptyStr
    entity_refs: list[Annotated[StrictStr, StringConstraints(pattern=r"^(object|part):\S+$")]]
    range: TimeRange

    @model_validator(mode="after")
    def ended_range(self) -> MotionFeature:
        if self.range.end_ms is None:
            raise ValueError("motion range must have an end_ms")
        return self


class DetectionSample(Contract):
    frame: FrameRef
    objects: ObservationSet[DetectedObject]
    landmarks: ObservationSet[Landmark]
    motions: ObservationSet[MotionFeature]


class DetectionInfo(Contract):
    context: SessionKey
    candidate: CandidateKey
    detector_config: ResourceRef
    samples: Annotated[list[DetectionSample], Field(min_length=1)]


class ClipFrame(Contract):
    clip_frame_index: NonNegativeInt
    clip_ms: NonNegativeInt
    source: FrameRef


class RoiFrame(Contract):
    roi_id: ID
    clip_frame_index: NonNegativeInt
    bbox: BBox
    image: ResourceRef
    width: PositiveInt
    height: PositiveInt


class Clip(Contract):
    clip_id: ID
    video: ResourceRef
    actual_range: TimeRange
    frames: Annotated[list[ClipFrame], Field(min_length=1)]
    rois: list[RoiFrame]

    @model_validator(mode="after")
    def ended_range(self) -> Clip:
        if self.actual_range.end_ms is None:
            raise ValueError("clip actual_range must have an end_ms")
        return self


class CandidateMedia(Contract):
    context: SessionKey
    candidate: CandidateKey
    requested_range: TimeRange
    clip: Clip | None
    unavailable_reason: NonEmptyStr | None

    @model_validator(mode="after")
    def availability_matches_clip(self) -> CandidateMedia:
        if self.requested_range.end_ms is None:
            raise ValueError("requested_range must have an end_ms")
        if (self.clip is None) != (self.unavailable_reason is not None):
            raise ValueError("unavailable_reason is required exactly when clip is null")
        return self


class VerificationRequest(ExchangeRecord):
    request_id: ID
    context: SessionKey
    event: EventCandidate
    detection: DetectionInfo
    media: CandidateMedia
    created_at: Instant
    model_id: NonEmptyStr
    model_revision: NonEmptyStr
    prompt_version: NonEmptyStr
    execution_config: ResourceRef

    @model_validator(mode="after")
    def candidate_has_ended(self) -> VerificationRequest:
        if self.event.action_range.end_ms is None:
            raise ValueError("verification requests require an ended candidate")
        return self


class ModelInputFrame(Contract):
    input_index: NonNegativeInt
    clip_frame_index: NonNegativeInt
    roi_id: ID | None
    source_bbox: BBox
    resized_size: PixelSize
    padding: Padding
    input_size: PixelSize

    @model_validator(mode="after")
    def dimensions_match(self) -> ModelInputFrame:
        width, height = self.resized_size
        left, top, right, bottom = self.padding
        if self.input_size != [width + left + right, height + top + bottom]:
            raise ValueError("input_size must equal resized_size plus padding")
        return self


class ModelInputRecord(ExchangeRecord):
    request_id: ID
    rendered_prompt: NonEmptyStr
    frames: Annotated[list[ModelInputFrame], Field(min_length=1)]

    @model_validator(mode="after")
    def indices_are_contiguous(self) -> ModelInputRecord:
        if [frame.input_index for frame in self.frames] != list(range(len(self.frames))):
            raise ValueError("input indices must be consecutive from zero in actual input order")
        return self


class Evidence(Contract):
    check: NonEmptyStr
    input_indices: Annotated[list[NonNegativeInt], Field(min_length=1)]
    description: NonEmptyStr


class ModelResponse(Contract):
    checks: dict[NonEmptyStr, CheckValue]
    object_description: NonEmptyStr
    observed_action: NonEmptyStr
    evidence: list[Evidence]


ErrorStage = Literal["INPUT", "PREPROCESS", "INFERENCE", "PARSE", "VALIDATION"]
ErrorCode = Literal[
    "MISSING_VIDEO", "TARGET_UNRESOLVED", "LINK_MISMATCH", "PREPROCESS_FAILED",
    "INFERENCE_FAILED", "INVALID_RESPONSE", "MISSING_FIELD", "INVALID_VALUE",
    "CONTRADICTORY_RESPONSE", "EVIDENCE_MISMATCH",
]
ERROR_CODES_BY_STAGE: dict[str, set[str]] = {
    "INPUT": {"MISSING_VIDEO", "TARGET_UNRESOLVED", "LINK_MISMATCH"},
    "PREPROCESS": {"PREPROCESS_FAILED"},
    "INFERENCE": {"INFERENCE_FAILED"},
    "PARSE": {"INVALID_RESPONSE"},
    "VALIDATION": {"MISSING_FIELD", "INVALID_VALUE", "CONTRADICTORY_RESPONSE", "EVIDENCE_MISMATCH"},
}


class ProcessingError(Contract):
    stage: ErrorStage
    code: ErrorCode
    message: NonEmptyStr

    @model_validator(mode="after")
    def code_matches_stage(self) -> ProcessingError:
        if self.code not in ERROR_CODES_BY_STAGE[self.stage]:
            raise ValueError("processing error code does not belong to its stage")
        return self


class RequestState(ExchangeRecord):
    request_id: ID
    request_status: Literal["QUEUED", "RUNNING", "FINISHED"]
    started_at: Instant | None
    finished_at: Instant | None
    processing_status: ProcessingStatus | None

    @model_validator(mode="after")
    def state_fields_match(self) -> RequestState:
        if self.request_status == "QUEUED" and self.started_at is not None:
            raise ValueError("QUEUED requests have not started")
        if self.request_status == "RUNNING" and self.started_at is None:
            raise ValueError("RUNNING requests require started_at")
        if self.request_status == "FINISHED":
            if self.finished_at is None or self.processing_status is None:
                raise ValueError("FINISHED requests require finished_at and processing_status")
            if self.started_at is None and self.processing_status != "NOT_RUN":
                raise ValueError("only NOT_RUN requests may finish without starting")
        elif self.finished_at is not None or self.processing_status is not None:
            raise ValueError("unfinished requests cannot have terminal fields")
        if self.started_at is not None and self.finished_at is not None:
            if self.finished_at < self.started_at:
                raise ValueError("finished_at precedes started_at")
        return self


class VerificationResult(ExchangeRecord):
    request_id: ID
    context: SessionKey
    candidate: CandidateKey
    action_range: TimeRange
    clip_id: ID | None
    input_record: ResourceRef | None
    processing_status: ProcessingStatus
    result: ModelResponse | None
    verification: VerificationStatus | None
    verification_rule_version: NonEmptyStr
    raw_output: StrictStr | None
    error: ProcessingError | None
    response_received_at: Instant | None
    finished_at: Instant

    @model_validator(mode="after")
    def terminal_fields_match(self) -> VerificationResult:
        if self.action_range.end_ms is None:
            raise ValueError("result action_range must have an end_ms")
        if self.processing_status == "OK":
            required = (self.clip_id, self.input_record, self.result, self.verification,
                        self.raw_output, self.response_received_at)
            if any(value is None for value in required) or self.error is not None:
                raise ValueError("OK requires input, response, verification, and no error")
        else:
            if self.result is not None or self.verification is not None or self.error is None:
                raise ValueError("failed requests require an error and no verification result")
            if self.processing_status == "NOT_RUN":
                if any(value is not None for value in (
                    self.input_record, self.raw_output, self.response_received_at
                )):
                    raise ValueError("NOT_RUN cannot contain model input or response records")
                if self.error.stage != "INPUT":
                    raise ValueError("NOT_RUN requires an INPUT error")
            elif self.error.stage == "INPUT":
                raise ValueError("INPUT errors must use NOT_RUN")
        if (self.raw_output is None) != (self.response_received_at is None):
            raise ValueError("raw_output and response_received_at must be present together")
        if self.response_received_at is not None and self.response_received_at > self.finished_at:
            raise ValueError("response_received_at follows finished_at")
        return self


class ScheduledOccurrence(ExchangeRecord):
    scheduled_occurrence_id: ID
    user_id: ID
    scheduled_at: Instant


class AuthenticationResult(ExchangeRecord):
    auth_id: ID
    scheduled_occurrence_id: ID
    auth_status: Literal["SUCCEEDED", "FAILED", "ERROR"]
    user_id: ID | None
    stream_id: ID | None
    target_track_id: ID | None
    frame: FrameRef | None
    processed_at: Instant
    reason: NonEmptyStr | None

    @model_validator(mode="after")
    def validate_auth(self):
        linked = (self.user_id, self.stream_id, self.target_track_id, self.frame)
        if self.auth_status == "SUCCEEDED":
            if any(x is None for x in linked) or self.reason is not None:
                raise ValueError("Successful authentication requires all identity fields")
            if self.frame.stream_id != self.stream_id or self.frame.session_ms is not None:
                raise ValueError("Authentication frame must be pre-session and match stream")
        elif any(x is not None for x in linked) or self.reason is None:
            raise ValueError("Failed authentication requires null identity and a reason")
        return self


class ReceivedResult(ExchangeRecord):
    request_id: ID
    received_at: Instant


class SupportingEvidence(Contract):
    request_id: ID
    related_core_request_ids: Annotated[list[ID], Field(min_length=1)]
    relation_reason: NonEmptyStr


class ExcludedResult(Contract):
    request_id: ID
    reason: NonEmptyStr


class ConsistencyCheck(Contract):
    target_match: bool
    unresolved_conflict: bool
    checked_request_ids: list[ID]
    reason: NonEmptyStr


class CompletionRecord(ExchangeRecord):
    completion_id: ID
    context: SessionKey
    session_result: Literal["COMPLETE"]
    decided_at: Instant
    received_request_ids: list[ID]
    core_request_ids: Annotated[list[ID], Field(min_length=1)]
    supporting_evidence: list[SupportingEvidence]
    excluded_results: list[ExcludedResult]
    consistency: ConsistencyCheck
    decision_rule_version: NonEmptyStr
    decision_config: ResourceRef

    @model_validator(mode="after")
    def valid_partition(self):
        received = set(self.received_request_ids)
        core = set(self.core_request_ids)
        support = [x.request_id for x in self.supporting_evidence]
        partition = self.core_request_ids + support + [x.request_id for x in self.excluded_results]
        if len(received) != len(self.received_request_ids):
            raise ValueError("Duplicate received request")
        if len(partition) != len(set(partition)) or set(partition) != received:
            raise ValueError("Core, supporting and excluded must partition received results")
        checked = set(self.consistency.checked_request_ids)
        if not core.union(support) <= checked <= received:
            raise ValueError("Invalid consistency references")
        if any(not set(x.related_core_request_ids) <= core for x in self.supporting_evidence):
            raise ValueError("Supporting evidence references unknown core result")
        if not self.consistency.target_match or self.consistency.unresolved_conflict:
            raise ValueError("Completion requires matching targets and consistent evidence")
        return self


class CompletionNotice(ExchangeRecord):
    session_id: ID
    completion_id: ID
    record: ResourceRef


class SessionClosure(ExchangeRecord):
    session_id: ID
    completion_id: ID
    session_status: Literal["CLOSED"]
    closed_at: Instant


class SessionResultView(ExchangeRecord):
    session_id: ID
    user_id: ID
    scheduled_occurrence_id: ID
    session_status: Literal["ACTIVE", "CLOSED"]
    session_result: Literal["COMPLETE"] | None
    started_at: Instant
    decided_at: Instant | None
    closed_at: Instant | None
    completion_id: ID | None

    @model_validator(mode="after")
    def valid_state(self):
        if (self.session_result is None) != (self.completion_id is None):
            raise ValueError("Completion ID and result must agree")
        if (self.decided_at is None) != (self.completion_id is None):
            raise ValueError("Completion timestamp and ID must agree")
        if self.session_status == "CLOSED":
            if self.session_result is None or self.closed_at is None:
                raise ValueError("Closed session requires completion and closure")
            if self.closed_at < self.decided_at:
                raise ValueError("Closure precedes decision")
        elif self.closed_at is not None:
            raise ValueError("Active session cannot have closed_at")
        return self
