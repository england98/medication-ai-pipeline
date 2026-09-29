import json
from datetime import datetime, timezone

import pytest
from medication_pipeline import artifacts, cli
from medication_pipeline.config import PipelineConfig


def test_same_second_runs_preserve_previous_data_and_record_kst(tmp_path, monkeypatch):
    instant = datetime(2026, 9, 29, 16, 4, 5, 678000, tzinfo=timezone.utc)

    class Clock:
        @staticmethod
        def now(tz):
            return instant.astimezone(tz)

    monkeypatch.setattr(artifacts, "datetime", Clock)
    first = artifacts.create_test_directory(tmp_path, "webcam-q4-negative")
    (first / "result.json").write_text('"original"', encoding="utf-8")
    second = artifacts.create_test_directory(tmp_path, "webcam-q4-negative")
    assert first.name == "webcam-q4-negative_20260930_010405"
    assert second.name == "webcam-q4-negative-02_20260930_010405"
    assert (first / "result.json").read_text(encoding="utf-8") == '"original"'
    assert not (second / "result.json").exists()
    record = json.loads((second / "test-run.json").read_text(encoding="utf-8"))
    assert record["started_at"] == "2026-09-30T01:04:05.678+09:00"
    assert record["test_name"] == "webcam-q4-negative"
    assert record["run_dir"] == str(second)


@pytest.mark.parametrize("name", ["", "../escape", "x/y", "x\\y", "bad:name", "bad name", "a" * 65])
def test_invalid_names_cannot_escape_result_root(config, tmp_path, name):
    with pytest.raises(ValueError, match="test_name"):
        artifacts.create_test_directory(tmp_path / "new-root", name)
    assert not (tmp_path / "new-root").exists()
    with pytest.raises(ValueError, match="test_name"):
        PipelineConfig.model_validate({**config.model_dump(), "test_name": name})


def test_cli_test_name_overrides_config_without_editing_file(config, tmp_path, monkeypatch):
    config.test_name = "video-default"
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    original = path.read_bytes()
    received = []

    def run(settings, **kwargs):
        received.append(settings.test_name)
        return {"run_status": "FINISHED"}

    monkeypatch.setattr(cli, "doctor", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr(cli, "run_pipeline", run)
    assert cli.main(["run", "--config", str(path), "--test-name", "웹캠-빈손-오탐"]) == 0
    assert received == ["웹캠-빈손-오탐"]
    assert path.read_bytes() == original
