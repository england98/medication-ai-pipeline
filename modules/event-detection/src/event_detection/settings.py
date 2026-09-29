from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    rule_version: str = "geometry-events-v1"
    analysis_interval_ms: int = Field(default=100, ge=1)
    approach_lookback_ms: int = Field(default=600, ge=0)
    min_active_ms: int = Field(default=100, ge=0)
    release_ms: int = Field(default=500, ge=0)
    mouth_enter: float = Field(default=0.16, gt=0)
    mouth_exit: float = Field(default=0.24, gt=0)
    approach_distance: float = Field(default=0.4, gt=0)
    approach_speed: float = Field(default=0.025, ge=0)
    hand_object_distance: float = Field(default=0.10, gt=0)
    container_mouth_distance: float = Field(default=0.14, gt=0)
    object_track_iou: float = Field(default=0.15, gt=0, le=1)
    object_track_max_gap_ms: int = Field(default=500, ge=1)
    medication_labels: list[str] = Field(default_factory=lambda: ["pill", "tablet", "capsule", "medicine", "medication"])
    packaging_labels: list[str] = Field(default_factory=lambda: ["blister", "medicine packet", "pill bottle", "package"])
    container_labels: list[str] = Field(default_factory=lambda: ["cup", "bottle", "glass", "mug"])

    @model_validator(mode="after")
    def hysteresis(self):
        if self.mouth_exit < self.mouth_enter:
            raise ValueError("mouth_exit must be >= mouth_enter")
        return self
