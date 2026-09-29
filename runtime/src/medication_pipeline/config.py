"""Strict experiment configuration; paths resolve relative to the config file."""

import json
from pathlib import Path
from typing import Literal

from authentication_tracking import TrackingConfig
from event_detection import DetectionConfig
from event_detection.perception import ModelConfig
from medication_contracts import BBox, ScheduledOccurrence
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from session_decision import DecisionConfig
from session_video.video import VideoConfig
from vlm_verification.config import ExecutionConfig

from .artifacts import validate_test_name


class AuthenticationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["operator", "external"] = "operator"
    operator_confirmed: bool = False
    result_path: str | None = None
    target_bbox: BBox | None = None

    @model_validator(mode="after")
    def boundary(self):
        if self.mode == "external" and not self.result_path:
            raise ValueError("External authentication requires result_path")
        return self


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    source: str
    input_format: str | None = None
    capture_options: dict[str, str] = Field(default_factory=dict)
    record_input: bool | None = None
    start_frame: int = Field(default=0, ge=0)
    output_dir: str = "../outputs"
    test_name: str | None = None
    occurrence: ScheduledOccurrence
    authentication: AuthenticationConfig = Field(default_factory=AuthenticationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    models: ModelConfig
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    vlm: ExecutionConfig
    max_pending: int = Field(default=2, ge=1, le=32)
    max_queued: int = Field(default=10000, ge=1)

    @field_validator("test_name")
    @classmethod
    def valid_test_name(cls, value):
        return None if value is None else validate_test_name(value)

    @model_validator(mode="after")
    def buffer_covers_discovery(self):
        minimum = (max(self.video.pre_ms.values(), default=0) + self.detection.approach_lookback_ms
                   + self.detection.min_active_ms + self.detection.analysis_interval_ms)
        if self.video.buffer_ms < minimum:
            raise ValueError(f"video.buffer_ms must be >= {minimum} to preserve candidate pre-roll")
        if self.tracking.max_frame_gap_ms < self.detection.analysis_interval_ms:
            raise ValueError("Tracking gap limit must cover analysis interval")
        return self


def load_config(path: Path):
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    def resolve(value):
        candidate = Path(value)
        return str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())

    data["output_dir"] = resolve(data.get("output_dir", "../outputs"))
    if not data.get("input_format"):
        data["source"] = resolve(data["source"])
    for name in ("object_weights", "person_weights", "face_landmarker", "hand_landmarker", "pose_landmarker"):
        if data.get("models", {}).get(name):
            data["models"][name] = resolve(data["models"][name])
    auth = data.get("authentication", {})
    if auth.get("result_path"):
        auth["result_path"] = resolve(auth["result_path"])
    model_id = data.get("vlm", {}).get("model_id", "")
    if model_id.startswith((".", "/")) or Path(model_id).is_absolute():
        data["vlm"]["model_id"] = resolve(model_id)
    llama = data.get("vlm", {}).get("llama_cpp")
    if llama:
        for name in ("server_path", "mmproj_path"):
            llama[name] = resolve(llama[name])
    return PipelineConfig.model_validate_json(json.dumps(data))
