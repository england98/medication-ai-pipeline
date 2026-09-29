import json
from pathlib import Path

import pytest
from conftest import frame
from medication_pipeline.pipeline import VerificationQueue, run_pipeline
from PIL import Image
from session_video.recording import InputRecorder
from session_video.video import SourceFrame, read_video
from test_pipeline import ScriptedPerception, make_video


def test_recording_preserves_vfr_source_mapping_and_pads_odd_sizes(tmp_path, session):
    recorder = InputRecorder(tmp_path / "recording")
    times = [0, 37, 104, 220, 365]
    for index, ms in enumerate(times):
        ref = frame(session.context, index, ms, width=65, height=49, origin=500)
        recorder.add(SourceFrame(ref, Image.new("RGB", (65, 49), "red")))
    metadata = recorder.close("END_OF_INPUT")
    assert metadata["status"] == "FINALIZED"
    assert metadata["frame_count"] == 5
    assert metadata["source_time_origin_ms"] == 500
    assert metadata["source_size"] == [65, 49]
    assert metadata["encoded_size"] == [66, 50]
    assert metadata["padding"] == [0, 0, 1, 1]
    decoded = list(read_video(recorder.path, "replay"))
    assert [item.ref.source_ms for item in decoded] == times
    assert all(item.image.size == (66, 50) for item in decoded)
    rows = [json.loads(line) for line in recorder.index_path.read_text().splitlines()]
    assert [row["recording_ms"] for row in rows] == times
    assert [row["source"]["source_ms"] for row in rows] == [500 + ms for ms in times]
    assert recorder.close("OTHER_REASON") == metadata


def test_recording_rejects_discontinuity_but_keeps_readable_prefix(tmp_path, session):
    recorder = InputRecorder(tmp_path / "recording")
    item = SourceFrame(frame(session.context, 0, 0), Image.new("RGB", (64, 48)))
    recorder.add(item)
    with pytest.raises(ValueError, match="discontinuity"):
        recorder.add(item)
    metadata = recorder.close("ERROR")
    assert metadata["status"] == "ERROR"
    assert len(list(read_video(recorder.path, "replay"))) == 1


def test_empty_recording_does_not_claim_a_video_exists(tmp_path):
    recorder = InputRecorder(tmp_path / "recording")
    metadata = recorder.close("ERROR")
    assert metadata["status"] == "EMPTY"
    assert metadata["video_path"] is None
    assert metadata["frame_count"] == 0
    assert not recorder.path.exists()


@pytest.mark.parametrize("enabled", [None, False])
def test_capture_records_all_received_frames_before_analysis_start(config, enabled):
    make_video(config.source)
    source = read_video(config.source, "simulated-camera")
    config.input_format = "dshow"
    config.source = "video=simulated-camera"
    config.start_frame = 2
    config.record_input = enabled
    result = run_pipeline(config, frames=source, perception=ScriptedPerception(), detect_only=True)
    directory = Path(result["run_dir"])
    if enabled is False:
        assert result["recording"] is None
        assert not (directory / "recording").exists()
    else:
        metadata = result["recording"]
        assert result["frames_buffered"] == 20
        assert metadata["frame_count"] == 22
        assert metadata["status"] == "FINALIZED"
        assert metadata["stop_reason"] == "END_OF_INPUT"
        assert len(list(read_video(metadata["video_path"], "replay"))) == 22


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interrupt_or_error_finalizes_video_before_worker_shutdown(config, monkeypatch, failure):
    make_video(config.source)
    source = read_video(config.source, "simulated-camera")
    config.input_format = "dshow"
    config.source = "video=simulated-camera"

    class FailingPerception(ScriptedPerception):
        def detect(self, image, ref):
            if ref.frame_index == 4:
                raise failure("simulated capture test interruption")
            return super().detect(image, ref)

    original_close = VerificationQueue.close
    checked = []

    def close_after_recording(queue):
        metadata = json.loads((queue.directory / "recording/metadata.json").read_text())
        assert metadata["status"] == "FINALIZED"
        assert len(list(read_video(metadata["video_path"], "replay"))) == 5
        checked.append(True)
        original_close(queue)

    monkeypatch.setattr(VerificationQueue, "close", close_after_recording)
    perception = FailingPerception()
    with pytest.raises(failure):
        run_pipeline(config, frames=source, perception=perception, detect_only=True)
    directory = next(Path(config.output_dir).iterdir())
    error = json.loads((directory / "run-error.json").read_text())
    metadata = json.loads((directory / "recording/metadata.json").read_text())
    expected = "INTERRUPTED" if failure is KeyboardInterrupt else "ERROR"
    assert error["run_status"] == metadata["stop_reason"] == expected
    assert checked and perception.closed
    assert not (directory / ".pipeline-running").exists()
    assert not (directory / "summary.json").exists()


def test_first_frame_is_recorded_even_if_target_selection_fails(config):
    make_video(config.source)
    config.record_input = True

    class NoPerson(ScriptedPerception):
        def detect(self, image, ref):
            _, objects = super().detect(image, ref)
            return [], objects

    with pytest.raises(ValueError):
        run_pipeline(config, perception=NoPerson(), detect_only=True)
    directory = next(Path(config.output_dir).iterdir())
    metadata = json.loads((directory / "recording/metadata.json").read_text())
    assert metadata["stop_reason"] == "ERROR"
    assert metadata["frame_count"] == 1
    assert len(list(read_video(metadata["video_path"], "replay"))) == 1
