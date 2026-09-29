"""Check local Qwen loading and one-frame generation, without a session decision."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.local.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/video-qwen-single-frame"),
                        help="Result path prefix; appends _YYYYMMDD_HHMMSS in KST")
    args = parser.parse_args()

    import av
    import torch
    from medication_pipeline.artifacts import create_test_directory
    from medication_pipeline.config import load_config
    from transformers import Qwen3VLForConditionalGeneration
    from vlm_verification.backend import QwenBackend

    config = load_config(args.config)
    settings = config.vlm
    if settings.backend != "transformers":
        raise ValueError("Use a Transformers configuration here; GGUF uses smoke_qwen_gguf.py")
    if not settings.local_files_only:
        raise ValueError("This smoke check requires local_files_only=true")
    args.output = create_test_directory(args.output.parent, args.output.name)
    print(f"Test output: {args.output}", file=sys.stderr, flush=True)
    with av.open(config.source) as container:
        frame = next(container.decode(video=0))
        image = frame.to_image()
        source_ms = round(float(frame.pts * frame.time_base) * 1000)
    # Bound this installation check; full candidate input limits remain unchanged.
    image.thumbnail((224, 224))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.monotonic()
    processor = QwenBackend()._load_processor(settings)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe the visible scene in one short sentence."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=False,
                       min_pixels=settings.min_pixels, max_pixels=settings.max_pixels)
    print("Native processor ready; loading local model", flush=True)
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings.model_id, revision=settings.model_revision,
        local_files_only=True, trust_remote_code=False,
        dtype="auto" if settings.dtype == "auto" else getattr(torch, settings.dtype),
        attn_implementation="sdpa",
    ).to(device).eval()
    loaded = time.monotonic()
    print(f"Model loaded in {loaded - started:.1f}s; generating up to 32 tokens", flush=True)
    with torch.inference_mode():
        result = model.generate(**inputs.to(model.device), max_new_tokens=32, do_sample=False)
    response = processor.batch_decode(result[:, inputs["input_ids"].shape[-1]:],
                                     skip_special_tokens=True)[0]
    if not response.strip():
        raise RuntimeError("Qwen returned no visible text")
    summary = {
        "status": "PASSED", "run_dir": str(args.output), "model_id": settings.model_id,
        "model_revision": settings.model_revision, "device": str(model.device),
        "dtype": str(model.dtype), "source": config.source, "source_ms": source_ms,
        "input_image_size": list(image.size), "max_new_tokens": 32,
        "response": response, "load_seconds": round(loaded - started, 2),
        "generation_seconds": round(time.monotonic() - loaded, 2),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "accelerate", "huggingface-hub", "tokenizers")},
        "authentication": "NOT_TESTED", "medication_verification": "NOT_TESTED",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
