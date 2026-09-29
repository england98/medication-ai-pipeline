import json
import threading
from fractions import Fraction
from pathlib import Path

import av
import pytest
from medication_contracts import (
    CHECKS_BY_EVENT,
    DetectedObject,
    Landmark,
    ModelInputFrame,
    ModelInputRecord,
    ObservationSet,
)
from medication_pipeline.cli import doctor, main
from medication_pipeline.pipeline import run_pipeline
from PIL import Image
from session_video.video import read_video
from vlm_verification.backend import PreparedInput
from vlm_verification.media import load_clip_images


class ScriptedPerception:
    """Deterministic observations for orchestration tests, never a production backend."""
    def __init__(self):
        self.count = 0
        self.closed = False
        self.scanning_finished = threading.Event()

    def detect(self, image, frame):
        self.count += 1
        if frame.frame_index >= 16:
            self.scanning_finished.set()
        return [[0.05, 0.05, 0.95, 0.95]], ObservationSet[DetectedObject](status="OBSERVED", items=[], reason=None)

    def landmarks(self, image, tracking):
        near = 2 <= tracking.frame.frame_index <= 5 or 12 <= tracking.frame.frame_index <= 15
        return ObservationSet[Landmark](status="OBSERVED", reason=None, items=[
            Landmark(part="mouth", index=13, point=[0.5, 0.2], score=None),
            Landmark(part="hand_left", index=8, point=[0.5, 0.22 if near else 0.8], score=0.9)])

    def close(self):
        self.closed = True


class TestBackend:
    __test__ = False

    def __init__(self, answer="YES", block_until=None):
        self.answer, self.block_until = answer, block_until
        self.calls = 0

    def prepare(self, request, config):
        target = next(i for i, f in enumerate(request.media.clip.frames) if any(
            r.region_type == "TARGET" and r.frame == f.source for r in request.event.regions))
        images = load_clip_images(request.media.clip, max_frames=config.max_frames, required_indices=(target,))
        record = ModelInputRecord(schema_version="1.0", request_id=request.request_id,
            rendered_prompt="Integration-test backend; no real model predictions", frames=[
                ModelInputFrame(input_index=i, clip_frame_index=item.frame.clip_frame_index,
                    roi_id=None, source_bbox=item.source_bbox, resized_size=list(item.image.size),
                    padding=[0, 0, 0, 0], input_size=list(item.image.size)) for i, item in enumerate(images)])
        return PreparedInput(record, request.event.candidate.candidate_type)

    def generate(self, prepared, config):
        self.calls += 1
        if self.block_until:
            assert self.block_until.wait(timeout=10), "Capture stalled behind inference"
        checks = dict.fromkeys(CHECKS_BY_EVENT[prepared.payload], self.answer)
        return json.dumps({"checks": checks, "object_description": "A medication tablet",
            "observed_action": "Test observation", "evidence": [
                {"check": key, "input_indices": [0], "description": "Test frame"} for key in checks]})


def make_video(path, count=22):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for i in range(count):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (i * 10, 30, 90)))
            frame.pts, frame.time_base = i * 100, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_video_through_verification_and_completion(config):
    config.test_name = "video-e02-completion"
    config.record_input = True
    make_video(config.source)
    perception = ScriptedPerception()
    summary = run_pipeline(config, perception=perception, backend=TestBackend())
    assert summary["processing_errors"] == 0
    assert summary["session"]["session_result"] == "COMPLETE"
    assert summary["session"]["session_status"] == "CLOSED"
    run = Path(summary["run_dir"])
    assert run.name.startswith("video-e02-completion_")
    assert json.loads((run / "pipeline-config.json").read_text())["test_name"] == config.test_name
    assert json.loads((run / "test-run.json").read_text())["run_dir"] == str(run)
    assert (run / "session/completion.json").is_file()
    assert (run / "session/closure.json").is_file()
    assert list((run / "verification").glob("*/model_input.json"))
    assert perception.closed
    assert not list((run / "media/spool").glob("*"))
    assert summary["recording"]["status"] == "FINALIZED"
    assert len(list(read_video(summary["recording"]["video_path"], "replay"))) == summary["recording"]["frame_count"]


def test_pipeline_selects_gguf_and_preserves_completion_contract(config, monkeypatch):
    from vlm_verification.config import ExecutionConfig, LlamaCppConfig
    from vlm_verification.llama_cpp import LlamaCppBackend

    settings = ExecutionConfig(backend="llama_cpp", model_id="test.gguf", model_revision="a" * 40,
        local_files_only=True, dtype="auto", llama_cpp=LlamaCppConfig(
            server_path="server.exe", mmproj_path="vision.gguf"))
    config = config.model_copy(update={"vlm": settings})
    make_video(config.source)
    submitted = []

    def start(self, _):
        self.media_marker = "<__test_image__>"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def http(self, path, body=None, timeout=10):
        if path == "/apply-template":
            return {"prompt": "native: " + "\n".join(m["content"] for m in body["messages"])}
        if path == "/tokenize":
            return {"tokens": [1] * 40}
        submitted.append(body)
        return {"content": json.dumps({
            "checks": dict.fromkeys(CHECKS_BY_EVENT["E02"], "YES"),
            "object_description": "Synthetic test object", "observed_action": "Synthetic transfer",
            "evidence": [{"check": key, "input_indices": [0], "description": "Test evidence"}
                         for key in CHECKS_BY_EVENT["E02"]]}), "stop_type": "eos"}

    monkeypatch.setattr(LlamaCppBackend, "_start", start)
    monkeypatch.setattr(LlamaCppBackend, "_request", http)
    summary = run_pipeline(config, perception=ScriptedPerception())
    assert summary["processing_errors"] == 0
    assert summary["session"]["session_result"] == "COMPLETE"
    assert submitted and all(body["prompt"]["multimodal_data"] for body in submitted)
    run = Path(summary["run_dir"])
    assert (run / "session/closure.json").is_file()
    for path in (run / "verification").glob("*/runtime.json"):
        assert json.loads(path.read_text())["backend"] == "LlamaCppBackend"


def test_slow_vlm_does_not_stop_detection_and_separate_candidates_survive(config):
    make_video(config.source)
    perception = ScriptedPerception()
    backend = TestBackend("UNKNOWN", perception.scanning_finished)
    summary = run_pipeline(config, perception=perception, backend=backend)
    assert summary["processing_errors"] == 0
    assert summary["frames_analyzed"] == 22
    assert summary["requests"] == summary["results"] == backend.calls == 2
    assert summary["session"]["session_result"] is None
    assert summary["session"]["session_status"] == "ACTIVE"


def test_detect_only_never_creates_fake_results(config):
    make_video(config.source)
    summary = run_pipeline(config, perception=ScriptedPerception(), detect_only=True)
    assert summary["requests"] == 2
    assert summary["results"] == 0
    assert summary["session"]["session_result"] is None
    assert not (Path(summary["run_dir"]) / "verification").exists()
    assert Path(summary["run_dir"]).name.startswith("video-detection_")
    assert summary["recording"] is None


def test_model_errors_do_not_become_refutation(config):
    make_video(config.source)

    class FailingBackend(TestBackend):
        def generate(self, prepared, settings):
            raise RuntimeError("out of memory")

    summary = run_pipeline(config, perception=ScriptedPerception(), backend=FailingBackend())
    assert summary["processing_errors"] == 2
    assert summary["session"]["session_result"] is None
    results = [json.loads(line) for line in (Path(summary["run_dir"]) / "results.jsonl").read_text().splitlines()]
    assert all(r["verification"] is None and r["processing_status"] == "ERROR" for r in results)


def test_doctor_reports_missing_assets_and_cli_validates(config, tmp_path):
    assert doctor(config)["ready"] is False
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    assert main(["validate", "--config", str(path)]) == 0
    assert main(["doctor", "--config", str(path)]) == 2


def test_auth_not_acknowledged_never_starts_session(config):
    make_video(config.source)
    config.authentication.operator_confirmed = False
    with pytest.raises(ValueError, match="operator_confirmed"):
        run_pipeline(config, perception=ScriptedPerception(), detect_only=True)
    assert not list(Path(config.output_dir).glob("*/session/session.json"))


def test_deferred_verify_and_replay_preserve_results(config, monkeypatch):
    from medication_pipeline.cli import verify_run
    config.record_input = True
    make_video(config.source)
    initial = run_pipeline(config, perception=ScriptedPerception(), detect_only=True)
    backend = TestBackend()
    monkeypatch.setattr("medication_pipeline.pipeline.create_backend", lambda *_: backend)
    directory = Path(initial["run_dir"])
    first = verify_run(directory)
    assert first["session"]["session_status"] == "CLOSED"
    assert backend.calls == 2
    receipt_count = len((directory / "decision/received.jsonl").read_text().splitlines())
    second = verify_run(directory)
    assert second["session"]["completion_id"] == first["session"]["completion_id"]
    assert backend.calls == 2
    assert len((directory / "results.jsonl").read_text().splitlines()) == 2
    assert len((directory / "decision/received.jsonl").read_text().splitlines()) == receipt_count
    assert first["recording"] == second["recording"] == initial["recording"]


def test_verify_refuses_active_pipeline(tmp_path):
    from medication_pipeline.cli import verify_run
    (tmp_path / ".pipeline-running").write_text("test")
    with pytest.raises(RuntimeError, match="still running"):
        verify_run(tmp_path)
