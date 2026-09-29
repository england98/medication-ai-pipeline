"""Lazy local Qwen3-VL inference using the checkpoint's native processor/template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import ExecutionConfig
from .contracts import ModelInputFrame, ModelInputRecord, VerificationRequest
from .media import (
    MediaUnavailableError,
    TargetUnresolvedError,
    load_clip_images,
    sample_frame_indices,
)
from .prompts import PROMPT_VERSION, build_messages


@dataclass(frozen=True)
class PreparedInput:
    record: ModelInputRecord
    payload: Any


class VerificationBackend(Protocol):
    def prepare(self, request: VerificationRequest, config: ExecutionConfig) -> PreparedInput: ...

    def generate(self, prepared: PreparedInput, config: ExecutionConfig) -> str: ...


def create_backend(config: ExecutionConfig, media_root=Path("."), artifact_dir=None):
    if config.backend == "llama_cpp":
        from .llama_cpp import LlamaCppBackend
        return LlamaCppBackend(media_root, artifact_dir)
    return QwenBackend(media_root)


@dataclass(frozen=True)
class _QwenPayload:
    inputs: Any
    processor: Any
    config_snapshot: dict[str, Any]


class QwenBackend:
    """One local model instance, with heavy dependencies imported only on demand.

    The pipeline passes individual, timestamp-labelled images to the native image
    processor. Full-scene sampling is explicit and reproducible; no video decoder
    inside transformers may silently change the evidence sequence.
    """

    def __init__(self, media_root: Path | str = Path(".")) -> None:
        self.media_root = Path(media_root)
        self._processor: Any = None
        self._processor_key: tuple[Any, ...] | None = None
        self._model: Any = None
        self._model_key: tuple[Any, ...] | None = None

    def _load_processor(self, config: ExecutionConfig) -> Any:
        key = (
            config.model_id,
            config.model_revision,
            config.min_pixels,
            config.max_pixels,
            config.local_files_only,
        )
        if self._processor_key != key:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise RuntimeError("Install the inference extra to use the Qwen backend") from exc
            self._processor = AutoProcessor.from_pretrained(
                config.model_id,
                revision=config.model_revision,
                local_files_only=config.local_files_only,
                trust_remote_code=False,
                use_fast=False,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            )
            self._processor_key = key
        return self._processor

    def prepare(self, request: VerificationRequest, config: ExecutionConfig) -> PreparedInput:
        if (request.model_id, request.model_revision, request.prompt_version) != (
            config.model_id,
            config.model_revision,
            config.prompt_version,
        ):
            raise ValueError("Request model/revision/prompt differs from execution configuration")
        if request.prompt_version != PROMPT_VERSION:
            raise ValueError(f"Unsupported prompt version: {request.prompt_version}")
        if request.media.clip is None:
            raise MediaUnavailableError(request.media.unavailable_reason or "No clip is available")
        clip = request.media.clip
        target_indices = [
            frame.clip_frame_index
            for frame in clip.frames
            if any(
                region.region_type == "TARGET" and region.frame == frame.source
                for region in request.event.regions
            )
        ]
        if not target_indices:
            raise TargetUnresolvedError("No TARGET region maps exactly to a clip frame")
        sampled = sample_frame_indices(len(clip.frames), config.max_frames)
        required = () if set(sampled).intersection(target_indices) else (target_indices[0],)
        images = load_clip_images(
            clip,
            max_frames=config.max_frames,
            include_rois=config.include_rois,
            timestamp_tolerance_ms=config.timestamp_tolerance_ms,
            media_root=self.media_root,
            required_indices=required,
        )
        processor = self._load_processor(config)
        messages = build_messages(request, images)
        rendered_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[rendered_prompt],
            images=[item.image for item in images],
            return_tensors="pt",
            padding=False,
            do_resize=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        grid = inputs.get("image_grid_thw")
        if grid is None or len(grid) != len(images):
            raise RuntimeError("Native processor did not return one image grid per supplied image")
        patch_size = getattr(processor.image_processor, "patch_size", None)
        if not isinstance(patch_size, int) or patch_size < 1:
            raise RuntimeError("Native image processor has no valid patch_size")
        frames: list[ModelInputFrame] = []
        for index, (item, row) in enumerate(zip(images, grid)):
            temporal, height_patches, width_patches = [int(value) for value in row]
            if temporal != 1 or min(height_patches, width_patches) < 1:
                raise RuntimeError("Unexpected image_grid_thw; cannot prove input mapping")
            size = [width_patches * patch_size, height_patches * patch_size]
            frames.append(
                ModelInputFrame(
                    input_index=index,
                    clip_frame_index=item.frame.clip_frame_index,
                    roi_id=item.roi_id,
                    source_bbox=item.source_bbox,
                    resized_size=size,
                    padding=[0, 0, 0, 0],
                    input_size=size,
                )
            )
        # Preserve the text actually tokenized, including native vision markers.
        actual_prompt = processor.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        record = ModelInputRecord(
            schema_version="1.0",
            request_id=request.request_id,
            rendered_prompt=actual_prompt,
            frames=frames,
        )
        return PreparedInput(
            record, _QwenPayload(inputs, processor, config.model_dump(mode="json"))
        )

    def generate(self, prepared: PreparedInput, config: ExecutionConfig) -> str:
        payload = prepared.payload
        if not isinstance(payload, _QwenPayload):
            raise TypeError("Prepared input was not produced by QwenBackend")
        if payload.config_snapshot != config.model_dump(mode="json"):
            raise ValueError("Execution configuration changed after input preparation")
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the inference extra to use the Qwen backend") from exc
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch reports no usable CUDA device")
        if (
            device.startswith("cuda")
            and config.dtype == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("The selected CUDA device does not support bfloat16")
        key = (
            config.model_id,
            config.model_revision,
            device,
            config.dtype,
            config.local_files_only,
        )
        if self._model_key != key:
            self._model = None
            self._model_key = None
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_id,
                revision=config.model_revision,
                local_files_only=config.local_files_only,
                trust_remote_code=False,
                dtype="auto" if config.dtype == "auto" else getattr(torch, config.dtype),
                attn_implementation="sdpa",
            )
            self._model = model.to(device).eval()
            self._model_key = key
        inputs = payload.inputs.to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        return payload.processor.batch_decode(
            output[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
