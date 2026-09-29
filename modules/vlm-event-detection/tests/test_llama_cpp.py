"""GGUF transport, provenance and failure boundaries, without a model download."""

import base64
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image
from test_service import FakeBackend
from test_service import case as case

from vlm_verification.backend import PreparedInput, QwenBackend, create_backend
from vlm_verification.config import ExecutionConfig, LlamaCppConfig
from vlm_verification.llama_cpp import LlamaCppBackend, image_size
from vlm_verification.media import LoadedImage, TargetUnresolvedError
from vlm_verification.service import VerificationService, VerificationWorker


@pytest.fixture
def gguf_case(case, monkeypatch):
    request, session, _, directory = case
    config = ExecutionConfig(model_id="model.gguf", model_revision="a" * 40,
        backend="llama_cpp", dtype="auto", device="cpu", local_files_only=True,
        llama_cpp=LlamaCppConfig(server_path="server.exe", mmproj_path="vision.gguf"))
    request = request.model_copy(update={"model_id": config.model_id})
    (directory / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    backend = LlamaCppBackend(directory, directory / "runtime")
    backend.artifact_dir.mkdir()
    backend.media_marker = "<__native_random_marker__>"
    monkeypatch.setattr(backend, "_start", lambda _: None)
    images = [LoadedImage(Image.new("RGB", (65, 49)), f, None, [0.0, 0.0, 1.0, 1.0])
              for f in request.media.clip.frames]
    monkeypatch.setattr("vlm_verification.llama_cpp.load_clip_images", lambda *a, **k: images)
    calls = []

    def http(path, body=None, timeout=10):
        calls.append((path, body))
        if path == "/apply-template":
            return {"prompt": "native template: " + "\n".join(m["content"] for m in body["messages"])}
        if path == "/tokenize":
            return {"tokens": list(range(100))}
        return {"content": FakeBackend().raw, "stop_type": "eos", "truncated": False}

    monkeypatch.setattr(backend, "_request", http)
    return request, session, config, directory, backend, calls


def test_gguf_native_prompt_and_pngs_match_saved_record(gguf_case):
    request, session, config, directory, backend, calls = gguf_case
    result = VerificationService(backend, config, directory / "results", directory).verify(request, session)
    assert result.processing_status == "OK"
    record = json.loads(next((directory / "results").glob("*/model_input.json")).read_text())
    sent = next(body for path, body in calls if path == "/completion")
    assert sent["prompt"]["prompt_string"] == record["rendered_prompt"]
    assert record["rendered_prompt"].count(backend.media_marker) == 2
    for entry, encoded in zip(record["frames"], sent["prompt"]["multimodal_data"]):
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            assert list(image.size) == entry["input_size"] == entry["resized_size"]
        assert entry["padding"] == [0, 0, 0, 0]
    assert sent["temperature"] == 0 and not sent["cache_prompt"]
    assert set(sent["json_schema"]["properties"]["checks"]["required"]) == {
        "object_is_medication", "object_transfer_into_mouth_visible"}


@pytest.mark.parametrize("size", [(1920, 1080), (49, 65), (1, 5000), (4096, 4096), (1, 1)])
def test_gguf_dimensions_stay_inside_native_patch_limits(size):
    w, h = image_size(*size, 3136, 262144)
    assert w % 32 == h % 32 == 0
    assert 4096 <= w * h <= 262144


def test_gguf_missing_target_rejected_before_server_start(gguf_case):
    request, _, config, _, backend, calls = gguf_case
    request = request.model_copy(update={"event": request.event.model_copy(update={"regions": []})})
    with pytest.raises(TargetUnresolvedError):
        backend.prepare(request, config)
    assert not calls


def test_gguf_context_overflow_never_sends_generation(gguf_case, monkeypatch):
    request, _, config, _, backend, calls = gguf_case
    original = backend._request
    monkeypatch.setattr(backend, "_request", lambda path, *a, **k:
        {"tokens": [1] * 20000} if path == "/tokenize" else original(path, *a, **k))
    with pytest.raises(ValueError, match="Input needs"):
        backend.prepare(request, config)
    assert all(path != "/completion" for path, _ in calls)


@pytest.mark.parametrize("reply", [
    {"content": "{}", "truncated": True},
    {"content": "{}", "stop_type": "limit"},
    {"content": ""},
])
def test_gguf_incomplete_output_never_becomes_valid_verdict(gguf_case, monkeypatch, reply):
    request, session, config, directory, backend, _ = gguf_case
    original = backend._request
    monkeypatch.setattr(backend, "_request", lambda path, *a, **k:
        reply if path == "/completion" else original(path, *a, **k))
    result = VerificationService(backend, config, directory / "results", directory).verify(request, session)
    assert result.processing_status == "ERROR"
    assert result.verification is None
    assert result.error.code == "INFERENCE_FAILED"


def test_gguf_server_restart_invalidates_prepared_input(gguf_case):
    request, _, config, _, backend, _ = gguf_case
    prepared = backend.prepare(request, config)
    altered = PreparedInput(prepared.record, replace(prepared.payload, media_marker="other-server"))
    with pytest.raises(ValueError, match="different server"):
        backend.generate(altered, config)


def test_backend_selection_and_worker_cleanup(gguf_case, monkeypatch):
    _, _, config, directory, backend, _ = gguf_case
    assert isinstance(create_backend(config, directory), LlamaCppBackend)
    assert isinstance(create_backend(ExecutionConfig(model_revision="a" * 40)), QwenBackend)
    closed = []
    monkeypatch.setattr(backend, "close", lambda: closed.append(True))
    worker = VerificationWorker(VerificationService(backend, config, directory))
    worker.shutdown()
    assert closed == [True]


def test_gguf_requires_explicit_native_precision_and_runtime():
    with pytest.raises(ValueError, match="runtime"):
        ExecutionConfig(backend="llama_cpp", model_revision="a" * 40)
    with pytest.raises(ValueError, match="dtype=auto"):
        ExecutionConfig(backend="llama_cpp", model_id="local.gguf", model_revision="a" * 40,
            local_files_only=True, llama_cpp=LlamaCppConfig(server_path="server", mmproj_path="vision"))


def test_legacy_transformers_records_replay_without_rewriting(case):
    request, session, config, directory = case
    backend = FakeBackend()
    service = VerificationService(backend, config, directory / "results", directory)
    first = service.verify(request, session)
    saved = next((directory / "results").glob("*/execution_config.json"))
    old_config = json.loads(saved.read_text())
    old_config.pop("backend")
    old_config.pop("llama_cpp")
    saved.write_text(json.dumps(old_config), encoding="utf-8")
    old_bytes = saved.read_bytes()
    assert service.verify(request, session) == first
    assert backend.calls == 1 and saved.read_bytes() == old_bytes


def test_gguf_hash_mismatch_fails_before_loading(case):
    _, _, _, directory = case
    server = directory / "server.exe"
    model = directory / "model.gguf"
    server.write_bytes(b"not executed")
    model.write_bytes(b"wrong weights")
    config = ExecutionConfig(backend="llama_cpp", model_id=str(model), model_revision="a" * 40,
        local_files_only=True, dtype="auto", llama_cpp=LlamaCppConfig(
            server_path=str(server), mmproj_path="missing.gguf"))
    backend = LlamaCppBackend(directory)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        backend._start(config)
    assert backend.process is None


def test_gguf_failed_start_cleans_owned_process_and_log(case, monkeypatch):
    _, _, _, directory = case
    server = directory / "server.exe"
    server.write_bytes(b"not executed")
    config = ExecutionConfig(backend="llama_cpp", model_id="model.gguf", model_revision="a" * 40,
        local_files_only=True, dtype="auto", llama_cpp=LlamaCppConfig(
            server_path=str(server), mmproj_path="vision.gguf"))
    monkeypatch.setattr("vlm_verification.llama_cpp.file_hash", lambda path:
        config.llama_cpp.model_sha256 if path == "model.gguf" else config.llama_cpp.mmproj_sha256)
    monkeypatch.setattr("vlm_verification.llama_cpp.subprocess.run", lambda *a, **k:
        SimpleNamespace(stdout=b"version: build 10991", stderr=b""))
    monkeypatch.setattr("vlm_verification.llama_cpp.subprocess.Popen", lambda *a, **k:
        SimpleNamespace(poll=lambda: 1))
    backend = LlamaCppBackend(directory)
    with pytest.raises(RuntimeError, match="startup failed"):
        backend._start(config)
    assert backend.process is None and backend.log is None
