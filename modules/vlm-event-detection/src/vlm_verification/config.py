"""Explicit, serializable settings for a reproducible local experiment."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROMPT_VERSION = "vlm-events-v1"
RULE_VERSION = "vlm-events-rules-v1"


class LlamaCppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    server_path: str
    mmproj_path: str
    model_sha256: str = "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
    mmproj_sha256: str = "c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5"
    runtime_version: Literal["b10991"] = "b10991"
    context_size: int = Field(default=16384, ge=2048, le=131072)
    gpu_layers: int = Field(default=99, ge=0, le=999)
    mmproj_offload: bool = False
    threads: int = Field(default=4, ge=1, le=128)
    startup_timeout_s: int = Field(default=180, ge=1)
    request_timeout_s: int = Field(default=600, ge=1)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["transformers", "llama_cpp"] = "transformers"
    llama_cpp: LlamaCppConfig | None = None
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    prompt_version: Literal["vlm-events-v1"] = PROMPT_VERSION
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = "float32"
    max_frames: int = Field(default=8, ge=2, le=128)
    include_rois: bool = False
    min_pixels: int = Field(default=3136, ge=1024)
    max_pixels: int = Field(default=262144, ge=1024)
    max_new_tokens: int = Field(default=768, ge=128, le=8192)
    do_sample: Literal[False] = False
    local_files_only: bool = False
    timestamp_tolerance_ms: int = Field(default=2, ge=0, le=20)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.backend == "llama_cpp":
            if self.llama_cpp is None:
                raise ValueError("llama_cpp backend requires runtime and mmproj settings")
            if not self.model_id.lower().endswith(".gguf") or not self.local_files_only:
                raise ValueError("llama_cpp requires a local GGUF model with local_files_only=true")
            if self.dtype != "auto":
                raise ValueError("Use dtype=auto for GGUF; tensor precision is stored in the model")
            if self.max_new_tokens >= self.llama_cpp.context_size:
                raise ValueError("GGUF context must leave room for the prompt")
            if (self.min_pixels + 1023) // 1024 > self.max_pixels // 1024:
                raise ValueError("GGUF pixel limits must accommodate a 32x32 vision grid")
        elif self.llama_cpp is not None:
            raise ValueError("llama_cpp options require backend=llama_cpp")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        if self.device == "cpu" and self.dtype == "float16":
            raise ValueError("Use float32 on CPU; float16 requires a supported accelerator")
        return self
