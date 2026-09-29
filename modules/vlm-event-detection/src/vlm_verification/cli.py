"""Local commands for candidate-level verification; model imports stay lazy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from pydantic import ValidationError

from .config import ExecutionConfig
from .contracts import SessionContext, VerificationRequest


def _read_model(path: Path, model):
    # PowerShell-created UTF-8 files can carry a BOM.
    return model.model_validate_json(path.read_text(encoding="utf-8-sig"))


def _print_json(value, *, error: bool = False) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), file=sys.stderr if error else sys.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01/E02/E03 후보의 로컬 VLM 검증")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("validate", "모델을 로드하지 않고 입력 계약과 연결을 검사합니다."),
        ("run", "Qwen3-VL을 실행하고 요청별 처리 기록을 저장합니다."),
    ):
        sub = subparsers.add_parser(command, help=description)
        sub.add_argument("--request", required=True, type=Path)
        sub.add_argument("--session", required=True, type=Path)
        sub.add_argument("--config", required=True, type=Path)
        if command == "run":
            sub.add_argument("--output-dir", type=Path, default=Path("outputs"))

    prepare = subparsers.add_parser("prepare", help="짧은 실험 영상에서 수동 후보 입력을 만듭니다.")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--session", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--event", choices=("E01", "E02", "E03"), required=True)
    prepare.add_argument("--action-start-ms", type=int, required=True)
    prepare.add_argument("--action-end-ms", type=int, required=True)
    prepare.add_argument(
        "--target-bbox", nargs=4, type=float, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="전체 클립에서 동일 대상을 포함하는 정규화 좌표. 운영자가 확인한 고정 영역입니다.",
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _prepare(args: argparse.Namespace) -> dict:
    """Make an explicit manual candidate, using decoded presentation timestamps."""
    try:
        import av
    except ImportError as exc:
        raise ValueError("prepare requires PyAV: uv sync --locked --extra dev --python 3.10") from exc

    session = _read_model(args.session, SessionContext)
    config = _read_model(args.config, ExecutionConfig)
    if session.session_status != "ACTIVE":
        raise ValueError("prepare requires an ACTIVE session")
    if not 0 <= args.action_start_ms <= args.action_end_ms:
        raise ValueError("action range must satisfy 0 <= start <= end")
    left, top, right, bottom = args.target_bbox
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("target bbox must satisfy 0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1")

    video = args.video.resolve(strict=True)
    if not video.is_file():
        raise ValueError("video must be a local file")
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("prepare output directory must be new; existing inputs are never overwritten")

    context = session.context.model_dump(mode="json")
    frames = []
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError("video has no video stream")
        formats = set(container.format.name.split(","))
        if "mp4" not in formats:
            raise ValueError("prepare currently supports MP4/MOV clips; use request JSON for other formats")
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("video frame is missing a presentation timestamp")
            timestamp = round(frame.pts * frame.time_base * 1000)
            session_ms = timestamp - session.source_time_origin_ms
            if session_ms < 0:
                raise ValueError("video begins before the session source_time_origin_ms")
            if frames and timestamp <= frames[-1]["source"]["source_ms"]:
                raise ValueError("frame timestamps must be strictly increasing at millisecond precision")
            frames.append({
                "clip_frame_index": index,
                "clip_ms": timestamp,
                "source": {
                    "stream_id": context["stream_id"], "frame_index": index,
                    "source_ms": timestamp, "session_ms": session_ms,
                    "width": frame.width, "height": frame.height,
                },
            })
    if len(frames) < 2:
        raise ValueError("video must contain at least two timestamped frames")
    actual_range = {
        "start_ms": frames[0]["source"]["session_ms"],
        "end_ms": frames[-1]["source"]["session_ms"],
    }
    if not actual_range["start_ms"] <= args.action_start_ms <= args.action_end_ms <= actual_range["end_ms"]:
        raise ValueError(f"action range must be within the actual video range {actual_range}")

    # This is an operator-supplied target declaration, not a detector/authenticator result.
    session_json = session.model_dump(mode="json")
    if session_json["tracking"] is None:
        session_json["session_revision"] += 1
        session_json["tracking"] = {
            "context": context, "frame": frames[0]["source"], "tracking_status": "TRACKED",
            "target_bbox": args.target_bbox,
        }
    session = SessionContext.model_validate_json(json.dumps(session_json))
    candidate = {
        "candidate_id": "manual-" + uuid4().hex,
        "candidate_type": args.event,
        "candidate_revision": 1,
    }
    at = _now()
    resource_prefix = uuid4().hex
    unavailable = {"status": "UNAVAILABLE", "items": [], "reason": "Manual candidate; detector not run."}
    request_json = {
        "schema_version": "1.0", "request_id": "request-" + uuid4().hex,
        "context": context,
        "event": {
            "context": context, "candidate": candidate,
            "created_at": at, "updated_at": at,
            "action_range": {"start_ms": args.action_start_ms, "end_ms": args.action_end_ms},
            "object_track_ids": [],
            "regions": [
                {"frame": frame["source"], "region_type": "TARGET", "bbox": args.target_bbox,
                 "object_track_id": None} for frame in frames
            ],
        },
        "detection": {
            "context": context, "candidate": candidate,
            "detector_config": {
                "resource_id": resource_prefix + "-detector", "locator": "detector-config.json",
                "media_type": "application/json",
            },
            "samples": [{
                "frame": frames[0]["source"], "objects": unavailable,
                "landmarks": unavailable, "motions": unavailable,
            }],
        },
        "media": {
            "context": context, "candidate": candidate,
            "requested_range": actual_range,
            "clip": {
                "clip_id": "clip-" + uuid4().hex,
                "video": {"resource_id": resource_prefix + "-video", "locator": str(video),
                          "media_type": "video/quicktime" if video.suffix.lower() == ".mov" else "video/mp4"},
                "actual_range": actual_range, "frames": frames, "rois": [],
            },
            "unavailable_reason": None,
        },
        "created_at": at, "model_id": config.model_id, "model_revision": config.model_revision,
        "prompt_version": config.prompt_version,
        "execution_config": {"resource_id": resource_prefix + "-config", "locator": str(output / "config.json"),
                             "media_type": "application/json"},
    }
    request = VerificationRequest.model_validate_json(json.dumps(request_json))
    from .service import validate_links

    validate_links(request, session, config)
    bundle = {
        "request.json": request.model_dump(mode="json"),
        "session.json": session.model_dump(mode="json"),
        "config.json": config.model_dump(mode="json"),
        "detector-config.json": {
            "schema_version": "1.0", "mode": "manual-candidate", "version": "manual-v1",
            "description": "Operator supplied event interval and static target bbox; detector not run.",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, content in bundle.items():
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"request": str(output / "request.json"), "session": str(output / "session.json"),
            "config": str(output / "config.json"), "frame_count": len(frames),
            "actual_range": actual_range, "mode": "manual-candidate"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            _print_json(_prepare(args))
            return 0

        request = _read_model(args.request, VerificationRequest)
        session = _read_model(args.session, SessionContext)
        config = _read_model(args.config, ExecutionConfig)
        from .service import VerificationService, validate_links

        if args.command == "validate":
            validate_links(request, session, config)
            config_path = Path(request.execution_config.locator)
            if not config_path.is_absolute():
                config_path = args.request.resolve().parent / config_path
            if request.execution_config.media_type != "application/json":
                raise ValueError("execution_config must refer to JSON")
            if _read_model(config_path, ExecutionConfig) != config:
                raise ValueError("Referenced execution_config differs from --config settings")
            _print_json({"valid": True, "request_id": request.request_id,
                         "note": "Schema/link validation only; video decoding and model execution were not performed."})
            return 0

        from .backend import create_backend

        backend = create_backend(config, args.request.resolve().parent, args.output_dir / "llama-runtime")
        service = VerificationService(
            backend=backend,
            config=config, output_dir=args.output_dir, media_root=args.request.resolve().parent,
        )
        try:
            result = service.verify(request, session)
        finally:
            if hasattr(backend, "close"):
                backend.close()
        _print_json(result.model_dump(mode="json"))
        return 0 if result.processing_status == "OK" else 2
    except (OSError, ValueError, ValidationError) as exc:
        _print_json({"error": type(exc).__name__, "code": getattr(exc, "code", None),
                     "message": str(exc)}, error=True)
        return 1
    except Exception as exc:
        _print_json({"error": type(exc).__name__, "message": str(exc)}, error=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
