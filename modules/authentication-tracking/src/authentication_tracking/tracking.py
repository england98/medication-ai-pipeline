"""Explicit authentication boundary and conservative, single-target spatial tracking.

This is an offline operator binding, not biometric identity verification. On loss
or ambiguous association it latches UNRESOLVED; a new authenticated session is
required, rather than silently adopting a person who enters the same location.
"""

from medication_contracts import AuthenticationResult, FrameRef, ScheduledOccurrence, TrackingUpdate
from medication_contracts.geometry import iou
from medication_contracts.records import new_id, utc_now
from pydantic import BaseModel, ConfigDict, Field


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    min_iou: float = Field(default=0.3, gt=0, le=1)
    ambiguity_margin: float = Field(default=0.15, ge=0, le=1)
    overlap_limit: float = Field(default=0.25, ge=0, le=1)
    max_frame_gap_ms: int = Field(default=1500, ge=1)


def select_person(people, selection=None, min_iou=0.3):
    if selection is None:
        if len(people) != 1:
            raise ValueError("Operator binding requires exactly one person or an explicit target_bbox")
        return list(people[0])
    matches = [box for box in people if iou(box, selection) >= min_iou]
    if len(matches) != 1:
        raise ValueError("target_bbox does not unambiguously select one detected person")
    return list(matches[0])


def operator_authenticate(occurrence: ScheduledOccurrence, frame: FrameRef, *,
                          acknowledged: bool) -> AuthenticationResult:
    if not acknowledged:
        raise ValueError("Set operator_confirmed=true only after identifying the video subject")
    return AuthenticationResult(
        schema_version="1.0", auth_id=new_id("auth"),
        scheduled_occurrence_id=occurrence.scheduled_occurrence_id,
        auth_status="SUCCEEDED", user_id=occurrence.user_id, stream_id=frame.stream_id,
        target_track_id=new_id("person"), frame=frame.model_copy(update={"session_ms": None}),
        processed_at=utc_now(), reason=None,
    )


class TargetTracker:
    def __init__(self, initial_bbox, config: TrackingConfig | None = None):
        self.bbox = list(initial_bbox)
        self.config = config or TrackingConfig()
        self.lost = False
        self.last_source_ms = None

    def update(self, context, frame, people) -> TrackingUpdate:
        if self.last_source_ms is not None:
            gap = frame.source_ms - self.last_source_ms
            if gap <= 0:
                raise ValueError("Tracking frames must increase strictly")
            if gap > self.config.max_frame_gap_ms:
                self.lost = True
        self.last_source_ms = frame.source_ms
        ranked = sorted(((iou(self.bbox, b), i, b) for i, b in enumerate(people)), reverse=True)
        if not self.lost:
            if not ranked or ranked[0][0] < self.config.min_iou:
                self.lost = True
            elif len(ranked) > 1 and (
                ranked[0][0] - ranked[1][0] <= self.config.ambiguity_margin
                or any(iou(ranked[0][2], b) >= self.config.overlap_limit
                       for _, _, b in ranked[1:])
            ):
                self.lost = True
            else:
                self.bbox = list(ranked[0][2])
        return TrackingUpdate(context=context, frame=frame,
                              tracking_status="UNRESOLVED" if self.lost else "TRACKED",
                              target_bbox=None if self.lost else self.bbox)
