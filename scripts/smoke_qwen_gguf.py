"""Verify real detector candidates with Q4 GGUF, without claiming user authentication."""

import argparse
import json
import sys
import time
from pathlib import Path

from medication_contracts import (
    DetectionInfo,
    EventCandidate,
    SessionContext,
    TrackingUpdate,
    VerificationRequest,
)
from medication_contracts.records import new_id, utc_now, write_json
from medication_pipeline.artifacts import create_test_directory
from medication_pipeline.config import load_config
from medication_pipeline.pipeline import resource
from session_decision import SessionDecision
from session_video.video import SourceFrame, VideoManager, read_video
from vlm_verification.backend import create_backend
from vlm_verification.service import VerificationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.local.json"))
    parser.add_argument("--detections", type=Path, default=Path("outputs/real-detection-smoke-robot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/video-qwen-q4-candidate-verification"),
                        help="Result path prefix; appends _YYYYMMDD_HHMMSS in KST")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.vlm.backend != "llama_cpp":
        raise ValueError("Select backend=llama_cpp")
    events = [EventCandidate.model_validate(item) for item in json.loads(
        (args.detections / "candidates.json").read_text(encoding="utf-8"))]
    observations = [json.loads(line) for line in (
        args.detections / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    if not events or not observations:
        raise ValueError("No actual detector candidates to verify")
    detection_summary = json.loads((args.detections / "summary.json").read_text(encoding="utf-8"))
    if Path(detection_summary["source"]).resolve() != Path(config.source).resolve():
        raise ValueError("Detector evidence and configured source differ")
    context = events[0].context
    if context.user_id != "unverified-video-subject":
        raise ValueError("This script accepts only diagnostic detector evidence")
    output = create_test_directory(args.output.parent, args.output.name)
    print(f"Test output: {output}", file=sys.stderr, flush=True)
    write_json(output / "execution-config.json", config.vlm)
    manager = VideoManager(output / "media", config.video)
    for event in events:
        manager.update(event)
    first = observations[0]["tracking"]["frame"]
    origin = first["source_ms"] - first["session_ms"]
    media = []
    for item in read_video(config.source, context.stream_id):
        if item.ref.source_ms < origin:
            continue
        ref = item.ref.model_copy(update={"session_ms": item.ref.source_ms - origin})
        manager.add(SourceFrame(ref, item.image))
        media.extend(manager.ready())
    media.extend(manager.ready(eof=True))
    manager.cleanup()
    media = {item.candidate.candidate_id: item for item in media}
    at = utc_now()
    session = SessionContext(schema_version="1.0", context=context, session_revision=1,
        auth_id="diagnostic-fixture-not-authentication", scheduled_at=at, started_at=at,
        source_time_origin_ms=origin, session_status="ACTIVE", completion_id=None, closed_at=None,
        tracking=TrackingUpdate.model_validate(observations[-1]["tracking"]))
    write_json(output / "session.json", session)
    backend = create_backend(config.vlm, output, output / "llama-runtime")
    service = VerificationService(backend, config.vlm, output / "verification", output)
    decision = SessionDecision(session, output / "decision", config.decision)
    results = []
    started = time.monotonic()
    try:
        for event in events:
            samples = [{"frame": row["tracking"]["frame"], "objects": row["objects"],
                "landmarks": row["landmarks"], "motions": {"status": "UNAVAILABLE", "items": [],
                    "reason": "Stored adapter observations do not contain motion samples"}}
                for row in observations if event.action_range.start_ms <=
                row["tracking"]["frame"]["session_ms"] <= event.action_range.end_ms]
            request = VerificationRequest(schema_version="1.0", request_id=new_id("diagnostic-request"),
                context=context, event=event, detection=DetectionInfo(context=context,
                    candidate=event.candidate, detector_config=resource(args.detections / "detector-config.json"),
                    samples=samples), media=media[event.candidate.candidate_id], created_at=utc_now(),
                model_id=config.vlm.model_id, model_revision=config.vlm.model_revision,
                prompt_version=config.vlm.prompt_version,
                execution_config=resource(output / "execution-config.json"))
            print(f"Verifying {event.candidate.candidate_type}: {event.action_range}", flush=True)
            result = service.verify(request, session)
            decision.receive(result, request)
            results.append(result)
            print(f"{result.processing_status}: {result.verification or result.error}", flush=True)
    finally:
        backend.close()
    summary = {"run_dir": str(output), "mode": "REAL_DETECTION_REPLAY_Q4", "authentication": "NOT_TESTED",
        "quality_evaluation": "NOT_PERFORMED", "requests": len(results),
        "processing_errors": sum(r.processing_status != "OK" for r in results),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "results": [r.model_dump(mode="json") for r in results]}
    write_json(output / "summary.json", summary)
    print(f"Results saved: {output / 'summary.json'}", flush=True)
    return 0 if summary["processing_errors"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
