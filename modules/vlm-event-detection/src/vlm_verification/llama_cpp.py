"""Owned, loopback-only llama.cpp b10991 worker for pinned Qwen3-VL GGUFs.

Images are resized explicitly onto Qwen3-VL's 32-pixel merged patch grid,
inside the same min/max token bounds passed to mtmd. Native preprocessing
therefore retains these dimensions. The native chat template is rendered by
the server and that exact string is submitted with ordered PNG payloads.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import io
import json
import math
import os
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .backend import PreparedInput
from .contracts import CHECKS_BY_EVENT, ModelInputFrame, ModelInputRecord
from .media import (
    MediaUnavailableError,
    TargetUnresolvedError,
    load_clip_images,
    sample_frame_indices,
)
from .prompts import PROMPT_VERSION, build_messages


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def image_size(width, height, min_pixels, max_pixels):
    """Preserve aspect ratio up to grid rounding; never crop or pad evidence."""
    minimum, maximum = math.ceil(min_pixels / 1024), max_pixels // 1024
    if minimum > maximum:
        raise ValueError("Pixel limits cannot represent a merged vision patch")
    scale = min(1.0, math.sqrt(maximum * 1024 / (width * height)))
    w = max(1, round(width * scale / 32))
    h = max(1, round(height * scale / 32))
    while w * h > maximum:
        if w >= h and w > 1:
            w -= 1
        else:
            h -= 1
    while w * h < minimum:
        options = [(w + 1, h), (w, h + 1)]
        options = [pair for pair in options if pair[0] * pair[1] <= maximum]
        if not options:
            # A narrow token interval may not contain a close aspect ratio.
            w, h = (minimum, 1) if width >= height else (1, minimum)
            break
        w, h = min(options, key=lambda pair: abs(math.log((pair[0] / pair[1]) / (width / height))))
    return w * 32, h * 32


def response_schema(event_type, image_count):
    checks = list(CHECKS_BY_EVENT[event_type])
    return {"type": "object", "additionalProperties": False,
        "required": ["checks", "object_description", "observed_action", "evidence"],
        "properties": {
            "checks": {"type": "object", "additionalProperties": False, "required": checks,
                       "properties": {key: {"type": "string", "enum": ["YES", "NO", "UNKNOWN"]} for key in checks}},
            "object_description": {"type": "string", "minLength": 1},
            "observed_action": {"type": "string", "minLength": 1},
            "evidence": {"type": "array", "minItems": 2, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["check", "input_indices", "description"],
                "properties": {"check": {"type": "string", "enum": checks},
                    "input_indices": {"type": "array", "minItems": 1,
                                      "items": {"type": "integer", "minimum": 0, "maximum": image_count - 1}},
                    "description": {"type": "string", "minLength": 1}}}}}}


@dataclass(frozen=True)
class LlamaPayload:
    prompt: str
    images: list[str]
    schema: dict
    config_snapshot: dict
    request_id: str
    media_marker: str


class LlamaCppBackend:
    def __init__(self, media_root=Path("."), artifact_dir=None):
        self.media_root = Path(media_root).resolve()
        self.artifact_dir = Path(artifact_dir or self.media_root / "llama-runtime").resolve()
        self.process = None
        self.log = None
        self.key = None
        self.base_url = None
        self.config_snapshot = None
        self.media_marker = None
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        atexit.register(self.close)

    def _request(self, path, body=None, timeout=10):
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        try:
            with self.http.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc

    def _start(self, config):
        snapshot = config.model_dump(mode="json")
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama.cpp stopped; inspect {self.artifact_dir / 'server.log'}")
            if self.config_snapshot != snapshot:
                raise ValueError("Cannot change configuration of a loaded GGUF worker")
            return
        options = config.llama_cpp
        if config.backend != "llama_cpp" or options is None:
            raise ValueError("Expected llama_cpp configuration")
        executable = Path(options.server_path).resolve(strict=True)
        for path, expected in ((config.model_id, options.model_sha256), (options.mmproj_path, options.mmproj_sha256)):
            if file_hash(path) != expected:
                raise ValueError(f"Pinned GGUF SHA256 mismatch: {path}")
        version = subprocess.run([str(executable), "--version"], capture_output=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
        version_text = (version.stdout + version.stderr).decode("utf-8", errors="replace")
        if options.runtime_version not in version_text and f"build {options.runtime_version[1:]}" not in version_text:
            raise ValueError(f"Expected llama.cpp {options.runtime_version}, received: {version_text}")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.key = secrets.token_hex(32)
        self.base_url = f"http://127.0.0.1:{port}"
        command = [str(executable), "--model", str(Path(config.model_id).resolve()),
            "--mmproj", str(Path(options.mmproj_path).resolve()), "--host", "127.0.0.1",
            "--port", str(port), "--ctx-size", str(options.context_size), "--parallel", "1",
            "--threads", str(options.threads), "--threads-batch", str(options.threads),
            "--batch-size", "256", "--ubatch-size", "128", "--flash-attn", "off",
            "--n-gpu-layers", "0" if config.device == "cpu" else str(options.gpu_layers),
            "--fit", "off", "--no-context-shift", "--no-webui", "--no-warmup",
            "--jinja", "--reasoning", "off", "--alias", "medication-qwen-q4",
            "--image-min-tokens", str(math.ceil(config.min_pixels / 1024)),
            "--image-max-tokens", str(config.max_pixels // 1024)]
        if config.device == "cpu":
            command += ["--device", "none"]
        if not options.mmproj_offload or config.device == "cpu":
            command += ["--no-mmproj-offload"]
        # Avoid inherited llama flags changing this recorded experiment.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("LLAMA_", "MTMD_"))}
        env["LLAMA_API_KEY"] = self.key
        self.log = (self.artifact_dir / "server.log").open("ab")
        try:
            self.process = subprocess.Popen(command, cwd=executable.parent, env=env,
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + options.startup_timeout_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama.cpp startup failed; inspect {self.artifact_dir / 'server.log'}")
                try:
                    props = self._request("/props", timeout=2)
                    if props.get("model_path"):
                        if not props.get("modalities", {}).get("vision"):
                            raise ValueError("Loaded server has no multimodal capability")
                        self.media_marker = props.get("media_marker")
                        if not isinstance(self.media_marker, str) or not self.media_marker:
                            raise ValueError("Native server did not declare its media marker")
                        (self.artifact_dir / "props.json").write_text(
                            json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8")
                        break
                except (OSError, RuntimeError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("Timed out loading llama.cpp model")
            self.config_snapshot = snapshot
            (self.artifact_dir / "runtime.json").write_text(json.dumps({
                "version": version_text, "server_sha256": file_hash(executable),
                "model_sha256": options.model_sha256, "mmproj_sha256": options.mmproj_sha256,
                "command": command, "config": snapshot,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def prepare(self, request, config):
        if (request.model_id, request.model_revision, request.prompt_version) != (
                config.model_id, config.model_revision, config.prompt_version):
            raise ValueError("Request model/revision/prompt differs from execution configuration")
        if request.prompt_version != PROMPT_VERSION:
            raise ValueError("Unsupported prompt version")
        clip = request.media.clip
        if clip is None:
            raise MediaUnavailableError(request.media.unavailable_reason or "No clip")
        target_indices = [f.clip_frame_index for f in clip.frames if any(
            r.region_type == "TARGET" and r.frame == f.source for r in request.event.regions)]
        if not target_indices:
            raise TargetUnresolvedError("No TARGET region maps exactly to a clip frame")
        sampled = sample_frame_indices(len(clip.frames), config.max_frames)
        required = () if set(sampled).intersection(target_indices) else (target_indices[0],)
        images = load_clip_images(clip, max_frames=config.max_frames, include_rois=config.include_rois,
            timestamp_tolerance_ms=config.timestamp_tolerance_ms, media_root=self.media_root,
            required_indices=required)
        from PIL import Image
        encoded, frames = [], []
        for index, item in enumerate(images):
            size = image_size(*item.image.size, config.min_pixels, config.max_pixels)
            pixels = item.image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
            stream = io.BytesIO()
            pixels.save(stream, format="PNG")
            encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
            frames.append(ModelInputFrame(input_index=index, clip_frame_index=item.frame.clip_frame_index,
                roi_id=item.roi_id, source_bbox=item.source_bbox, resized_size=list(size),
                padding=[0, 0, 0, 0], input_size=list(size)))
        self._start(config)
        marker = self.media_marker
        messages = []
        for message in build_messages(request, images):
            pieces = []
            for item in message["content"]:
                if item["type"] == "text":
                    if marker in item["text"]:
                        raise ValueError("Reserved media marker appears in metadata")
                    pieces.append(item["text"])
                else:
                    pieces.append(marker)
            messages.append({"role": message["role"], "content": "\n".join(pieces)})
        prompt = self._request("/apply-template", {"messages": messages})["prompt"]
        if prompt.count(marker) != len(encoded):
            raise ValueError("Native template changed the number of input images")
        # Count exact native text tokens plus the merged vision patches. Refuse
        # overflow instead of letting the server discard old evidence.
        token_count = len(self._request("/tokenize", {"content": prompt.replace(marker, ""),
                                                     "add_special": True})["tokens"])
        token_count += sum(f.input_size[0] * f.input_size[1] // 1024 + 4 for f in frames)
        if token_count + config.max_new_tokens > config.llama_cpp.context_size:
            raise ValueError(f"Input needs about {token_count} tokens plus {config.max_new_tokens} output; "
                             "increase llama_cpp.context_size or explicitly reduce max_frames/ROI budget")
        record = ModelInputRecord(schema_version="1.0", request_id=request.request_id,
                                  rendered_prompt=prompt, frames=frames)
        return PreparedInput(record, LlamaPayload(prompt, encoded,
            response_schema(request.event.candidate.candidate_type, len(frames)),
            config.model_dump(mode="json"), request.request_id, marker))

    def generate(self, prepared, config):
        payload = prepared.payload
        if not isinstance(payload, LlamaPayload) or payload.config_snapshot != config.model_dump(mode="json"):
            raise ValueError("Prepared GGUF input/configuration mismatch")
        self._start(config)
        if payload.media_marker != self.media_marker:
            raise ValueError("Prepared input belongs to a different server instance")
        try:
            result = self._request("/completion", {
                "prompt": {"prompt_string": payload.prompt, "multimodal_data": payload.images},
                "n_predict": config.max_new_tokens, "temperature": 0, "seed": 0,
                "repeat_penalty": 1.0, "cache_prompt": False, "stream": False,
                "json_schema": payload.schema,
            }, timeout=config.llama_cpp.request_timeout_s)
        except BaseException:
            self.close()
            raise
        filename = hashlib.sha256(payload.request_id.encode()).hexdigest() + ".json"
        (self.artifact_dir / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if result.get("truncated"):
            raise RuntimeError("llama.cpp truncated the prompt; evidence cannot be verified")
        if result.get("stop_type") == "limit" or result.get("stopped_limit"):
            raise RuntimeError("llama.cpp output hit the token limit; increase max_new_tokens")
        text = result.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("llama.cpp returned no visible response")
        return text

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.config_snapshot = None
