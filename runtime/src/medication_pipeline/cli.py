import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

from medication_contracts import CompletionRecord, SessionContext
from medication_contracts.records import write_json
from session_video import SessionManager

from .config import load_config
from .pipeline import VerificationQueue, run_pipeline


def doctor(config, *, detect_only=False):
    libraries = ["pydantic", "av", "PIL", "numpy", "ultralytics", "mediapipe", "torch"]
    if not detect_only and config.vlm.backend == "transformers":
        libraries += ["transformers", "accelerate"]
    checks = {}
    for name in libraries:
        distribution = {"PIL": "Pillow"}.get(name, name)
        try:
            importlib.metadata.version(distribution)
            checks[f"python:{name}"] = importlib.util.find_spec(name) is not None
        except importlib.metadata.PackageNotFoundError:
            checks[f"python:{name}"] = False
    checks.update({f"model:{name}": Path(path).is_file() for name, path in config.models.assets().items()})
    should_record = config.record_input if config.record_input is not None else config.input_format is not None
    if should_record:
        try:
            from av.codec import Codec
            Codec("libx264", "w")
            checks["recording_encoder"] = True
        except Exception:
            checks["recording_encoder"] = False
    if config.input_format is None:
        checks["source"] = Path(config.source).is_file()
    if config.authentication.mode == "operator":
        checks["operator_confirmed"] = config.authentication.operator_confirmed
    else:
        checks["authentication_result"] = Path(config.authentication.result_path).is_file()
    if not detect_only:
        model_path = Path(config.vlm.model_id)
        if config.vlm.backend == "llama_cpp":
            checks["vlm_checkpoint"] = model_path.is_file()
            checks["vlm_mmproj"] = Path(config.vlm.llama_cpp.mmproj_path).is_file()
            checks["llama_server"] = Path(config.vlm.llama_cpp.server_path).is_file()
        elif model_path.is_dir():
            checks["vlm_checkpoint"] = (model_path / "config.json").is_file() and bool(
                list(model_path.glob("*.safetensors")) or list(model_path.glob("pytorch_model*.bin")))
        elif config.vlm.local_files_only:
            try:
                from huggingface_hub import snapshot_download
                path = Path(snapshot_download(config.vlm.model_id, revision=config.vlm.model_revision, local_files_only=True))
                checks["vlm_checkpoint"] = bool(list(path.glob("*.safetensors")))
            except Exception:
                checks["vlm_checkpoint"] = False
    return {"ready": all(checks.values()), "checks": checks,
            "note": "Model compatibility and inference quality require a real run; operator mode is offline identity binding."}


def verify_run(directory: Path):
    directory = directory.resolve()
    if (directory / ".pipeline-running").exists():
        raise RuntimeError("Pipeline is still running; wait for it to finish before verification")
    lock = directory / ".verification-running"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    queue = None
    try:
        os.close(fd)
        config = load_config(directory / "pipeline-config.json")
        manager = SessionManager(directory / "session")
        manager.session = SessionContext.model_validate_json((directory / "session/session.json").read_text(encoding="utf-8"))
        completion_path = directory / "session/completion.json"
        if completion_path.exists():
            manager.completion = CompletionRecord.model_validate_json(completion_path.read_text(encoding="utf-8"))
        queue = VerificationQueue(directory, config, manager)
        # Preserve previous delivery order when recovering, followed by not-yet-received requests.
        paths = {VerificationQueue.load(p)[0].request_id: p for p in (directory / "queue").glob("*.json")}
        journal = directory / "results.jsonl"
        old_ids = []
        if journal.exists():
            old_ids = list(dict.fromkeys(json.loads(line)["request_id"] for line in journal.read_text(encoding="utf-8").splitlines()))
        for rid in old_ids:
            if rid in paths:
                queue.queue.append(paths.pop(rid))
        remaining = sorted(paths.values(), key=lambda p: (VerificationQueue.load(p)[0].event.action_range.start_ms, p.name))
        queue.queue.extend(remaining)
        queue.drain()
        recording_path = directory / "recording/metadata.json"
        summary = {"schema_version": "1.0", "run_dir": str(directory), "run_status": "FINISHED",
            "mode": "VERIFY", "requests": len(old_ids) + len(remaining), "results": len(queue.received),
            "processing_errors": sum(r.processing_status != "OK" for r in queue.received),
            "recording": json.loads(recording_path.read_text(encoding="utf-8")) if recording_path.exists() else None,
            "session": manager.view().model_dump(mode="json"),
            "message": "영상 기반 복약 완료" if manager.completion else "복약 확인 중"}
        write_json(directory / "summary.json", summary, immutable=False)
        return summary
    finally:
        if queue:
            queue.close()
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLO/MediaPipe → candidate video → Qwen3-VL → session completion")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "doctor", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--detect-only", action="store_true")
        if name == "run":
            command.add_argument("--test-name", help="Purpose of this test; folder gets a KST timestamp suffix")
    verify = sub.add_parser("verify", help="Run queued verification from a detect-only or interrupted run")
    verify.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_run(args.run_dir)
        else:
            config = load_config(args.config)
            if args.command == "run" and args.test_name is not None:
                config = type(config).model_validate({**config.model_dump(), "test_name": args.test_name})
            if args.command == "validate":
                result = {"valid": True, "source": config.source, "output_dir": config.output_dir}
            elif args.command == "doctor":
                result = doctor(config, detect_only=args.detect_only)
            else:
                checks = doctor(config, detect_only=args.detect_only)
                if not checks["ready"]:
                    print(json.dumps(checks, ensure_ascii=False, indent=2))
                    return 2
                result = run_pipeline(config, detect_only=args.detect_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("ready") is False else (3 if result.get("processing_errors", 0) else 0)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
