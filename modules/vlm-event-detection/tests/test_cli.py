import json
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

from vlm_verification.cli import main
from vlm_verification.contracts import SessionContext, VerificationRequest
from vlm_verification.media import load_clip_images

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def input_args(command):
    return [command, "--request", str(EXAMPLES / "request.e02.json"),
            "--session", str(EXAMPLES / "session.json"),
            "--config", str(EXAMPLES / "config.cpu.json")]


def test_example_validate_does_not_initialize_backend(monkeypatch, capsys):
    from vlm_verification import backend

    def forbidden(*args, **kwargs):
        raise AssertionError("validate must not initialize the inference backend")

    monkeypatch.setattr(backend, "QwenBackend", forbidden)
    assert main(input_args("validate")) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_validate_rejects_different_referenced_config(tmp_path, capsys):
    config = json.loads((EXAMPLES / "config.cpu.json").read_text("utf-8"))
    config["max_frames"] += 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    args = input_args("validate")
    args[-1] = str(path)
    assert main(args) == 1
    assert "Referenced execution_config differs" in capsys.readouterr().err


def test_missing_video_run_records_not_run_without_model(tmp_path, capsys):
    assert main(input_args("run") + ["--output-dir", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["processing_status"] == "NOT_RUN"
    assert result["error"]["code"] == "MISSING_VIDEO"
    assert result["verification"] is None
    assert len(list(tmp_path.glob("*/result.json"))) == 1


def test_prepare_preserves_variable_pts_and_produces_decodable_contract(tmp_path, capsys):
    video = tmp_path / "sample.mp4"
    with av.open(str(video), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=1000)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for timestamp in [10, 110, 310, 450, 900]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), "gray"))
            frame.pts = timestamp
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    destination = tmp_path / "experiment"
    args = ["prepare", "--video", str(video), "--session", str(EXAMPLES / "session.json"),
            "--config", str(EXAMPLES / "config.cpu.json"), "--event", "E02",
            "--action-start-ms", "110", "--action-end-ms", "450",
            "--target-bbox", "0.1", "0.1", "0.9", "0.9", "--output-dir", str(destination)]
    assert main(args) == 0
    capsys.readouterr()
    request = VerificationRequest.model_validate_json((destination / "request.json").read_text("utf-8"))
    session = SessionContext.model_validate_json((destination / "session.json").read_text("utf-8"))
    assert [frame.clip_ms for frame in request.media.clip.frames] == [10, 110, 310, 450, 900]
    assert session.tracking.frame.source_ms == 10
    assert session.session_revision == 2
    assert request.detection.samples[0].objects.status == "UNAVAILABLE"
    assert len(load_clip_images(request.media.clip, max_frames=3)) == 3
    assert main(["validate", "--request", str(destination / "request.json"),
                 "--session", str(destination / "session.json"),
                 "--config", str(destination / "config.json")]) == 0
    assert main(args) == 1
    assert "must be new" in capsys.readouterr().err


def test_malformed_json_is_cli_input_error(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    args = input_args("validate")
    args[2] = str(path)
    assert main(args) == 1
    assert "ValidationError" in capsys.readouterr().err
