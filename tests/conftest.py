import json
from pathlib import Path

import pytest
from medication_contracts import FrameRef, SessionContext
from medication_pipeline.config import PipelineConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session():
    return SessionContext.model_validate_json((ROOT / "modules/vlm-event-detection/examples/session.json").read_text())


@pytest.fixture
def config(tmp_path):
    return PipelineConfig.model_validate({
        "source": str(tmp_path / "input.mp4"), "output_dir": str(tmp_path / "runs"),
        "occurrence": {"schema_version": "1.0", "scheduled_occurrence_id": "occurrence-test",
                       "user_id": "user-test", "scheduled_at": "2026-09-27T00:00:00.000Z"},
        "authentication": {"operator_confirmed": True},
        "models": {"object_weights": "missing.pt", "face_landmarker": "face.task",
                   "hand_landmarker": "hand.task", "pose_landmarker": "pose.task"},
        "detection": {"analysis_interval_ms": 100, "min_active_ms": 100,
                      "release_ms": 200, "approach_lookback_ms": 100},
        "video": {"buffer_ms": 1000, "pre_ms": {"E01": 100, "E02": 100, "E03": 100},
                  "post_ms": {"E01": 100, "E02": 100, "E03": 100}},
        "vlm": json.loads((ROOT / "modules/vlm-event-detection/examples/config.cpu.json").read_text()),
        "max_pending": 1,
    })


def frame(context, index, ms, *, width=64, height=48, origin=0):
    return FrameRef(stream_id=context.stream_id, frame_index=index, source_ms=ms + origin,
                    session_ms=ms, width=width, height=height)
