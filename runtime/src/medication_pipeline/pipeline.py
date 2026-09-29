"""Continuous detection, disk-backed request queue, and a separate VLM worker."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import ExitStack
from pathlib import Path

from authentication_tracking import TargetTracker, operator_authenticate
from authentication_tracking.tracking import select_person
from event_detection import EventDetector
from event_detection.perception import YoloMediaPipe
from medication_contracts import (
    AuthenticationResult,
    ResourceRef,
    SessionContext,
    VerificationRequest,
)
from medication_contracts.records import append_json, key_path, new_id, utc_now, write_json
from session_decision import SessionDecision
from session_video import SessionManager
from session_video.recording import InputRecorder
from session_video.video import SourceFrame, VideoManager, read_video
from vlm_verification.backend import create_backend
from vlm_verification.service import VerificationService, VerificationWorker

from .artifacts import create_test_directory


def resource(path, kind="config"):
    return ResourceRef(resource_id=kind + "-" + hashlib.sha256(str(path.resolve()).encode()).hexdigest(), locator=str(path.resolve()), media_type="application/json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(config):
    versions = {}
    for name in ("pydantic", "numpy", "Pillow", "av", "ultralytics", "mediapipe", "torch", "transformers", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "libraries": versions, "model_sha256": {name: sha256(path)
            for name, path in config.models.assets().items() if Path(path).is_file()},
            "source_sha256": sha256(config.source) if Path(config.source).is_file() else None,
            "placement": {"detection": "main-thread", "vlm": "single-worker-thread"}}


class VerificationQueue:
    def __init__(self, directory, config, manager, *, backend=None, detect_only=False):
        self.directory, self.config, self.manager = directory, config, manager
        self.detect_only = detect_only
        self.decision = SessionDecision(manager.session, directory / "decision", config.decision)
        self.queue = deque()
        self.pending = {}
        self.received = []
        journal = directory / "results.jsonl"
        self.logged_ids = {json.loads(line)["request_id"] for line in journal.read_text(encoding="utf-8").splitlines()} if journal.exists() else set()
        self.worker = None if detect_only else VerificationWorker(VerificationService(
            backend or create_backend(config.vlm, directory, directory / "llama-runtime"), config.vlm, directory / "verification", directory), config.max_pending)

    def enqueue(self, request, session):
        path = key_path(self.directory / "queue", request.request_id)
        write_json(path, {"request": request.model_dump(mode="json"), "session": session.model_dump(mode="json")})
        append_json(self.directory / "queue-events.jsonl", {"request_id": request.request_id,
                    "state": "READY", "at": utc_now(), "record": str(path.resolve())})
        self.queue.append(path)
        if len(self.queue) > self.config.max_queued:
            raise RuntimeError("max_queued exceeded; persisted requests retained for verify command")

    @staticmethod
    def load(path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return (VerificationRequest.model_validate_json(json.dumps(data["request"])),
                SessionContext.model_validate_json(json.dumps(data["session"])))

    def poll(self):
        if self.detect_only:
            return
        for future, path in list(self.pending.items()):
            if not future.done():
                continue
            result = future.result()
            request, _ = self.load(path)
            completion = self.decision.receive(result, request)
            self.received.append(result)
            if result.request_id not in self.logged_ids:
                append_json(self.directory / "results.jsonl", result)
                self.logged_ids.add(result.request_id)
            append_json(self.directory / "queue-events.jsonl", {"request_id": result.request_id,
                        "state": "FINISHED", "at": utc_now()})
            if completion is not None:
                self.manager.close(completion)
            del self.pending[future]
        while self.queue and len(self.pending) < self.config.max_pending:
            path = self.queue[0]
            request, session = self.load(path)
            try:
                future = self.worker.submit(request, session)
            except RuntimeError as exc:
                # A Future can be done just before its slot-release callback runs.
                if "queue is full" in str(exc):
                    break
                raise
            self.queue.popleft()
            self.pending[future] = path

    def drain(self):
        if self.detect_only:
            return
        while self.queue or self.pending:
            self.poll()
            if self.pending:
                wait(self.pending, timeout=0.1, return_when=FIRST_COMPLETED)

    def close(self):
        if self.worker:
            self.worker.shutdown()


def run_pipeline(config, *, detect_only=False, perception=None, backend=None, frames=None):
    source_kind = "webcam" if config.input_format else "video"
    mode = "detection" if detect_only else "pipeline"
    run_dir = create_test_directory(Path(config.output_dir), config.test_name or f"{source_kind}-{mode}")
    print(f"Test output: {run_dir}", file=sys.stderr, flush=True)
    write_json(run_dir / "pipeline-config.json", config)
    write_json(run_dir / "environment.json", environment(config))
    execution_path = run_dir / "execution-config.json"
    write_json(execution_path, config.vlm)
    detector_path = run_dir / "detector-config.json"
    write_json(detector_path, {"models": config.models.model_dump(), "rules": config.detection.model_dump(),
        "coordinate_system": "normalized original frame", "distance_unit": "target pixel height",
        "parts": ["hand_left", "hand_right", "mouth", "pose"],
        "motion_features": {"hand_object_distance": "target_height", "object_mouth_distance": "target_height",
            "hand_mouth_distance": "target_height", "hand_mouth_approach_speed": "target_height/s"}})
    external = None
    if config.authentication.mode == "external":
        external = AuthenticationResult.model_validate_json(Path(config.authentication.result_path).read_text(encoding="utf-8-sig"))
        if external.auth_status != "SUCCEEDED":
            raise ValueError("External authentication did not succeed")
    stream_id = external.stream_id if external else new_id("stream")
    source = iter(frames) if frames is not None else read_video(config.source, stream_id,
        input_format=config.input_format, options=config.capture_options)
    manager = SessionManager(run_dir / "session")
    video = VideoManager(run_dir / "media", config.video)
    queue = None
    detector = None
    tracking = None
    closed = {}
    snapshots = {}
    processed_frames = 0
    analysis_frames = 0
    last_analysis_ms = None
    reason = "END_OF_INPUT"
    recorder = None
    recording = None
    failed = False

    def close_recording():
        try:
            recorder.close(reason)
        except Exception as exc:
            # Preserve the original capture/detection error or KeyboardInterrupt.
            if not failed:
                raise
            print(f"Recording finalization failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def updates(items, snapshot):
        for update in items:
            event = update.event
            video.update(event)
            append_json(run_dir / "candidate-history.jsonl", {"candidate": event.candidate.model_dump(),
                "action_range": event.action_range.model_dump(), "updated_at": event.updated_at})
            cid = event.candidate.candidate_id
            if snapshot.tracking is not None and snapshot.tracking.tracking_status == "TRACKED":
                snapshots[cid] = snapshot.model_copy(deep=True)
            if event.action_range.end_ms is not None:
                closed[cid] = update

    def media_ready(eof=False):
        for media in video.ready(eof=eof):
            cid = media.candidate.candidate_id
            update = closed.pop(cid)
            request = VerificationRequest(schema_version="1.0", request_id=new_id("request"),
                context=update.event.context, event=update.event, detection=update.detection, media=media,
                created_at=utc_now(), model_id=config.vlm.model_id, model_revision=config.vlm.model_revision,
                prompt_version=config.vlm.prompt_version, execution_config=resource(execution_path))
            queue.enqueue(request, snapshots.pop(cid))

    running = run_dir / ".pipeline-running"
    running.write_text(str(os.getpid()), encoding="ascii")
    try:
        should_record = config.record_input if config.record_input is not None else config.input_format is not None
        if should_record:
            recorder = InputRecorder(run_dir / "recording")
        (run_dir / "ultralytics").mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(run_dir / "ultralytics"))
        perception = perception or YoloMediaPipe(config.models, config.detection)
        for item in source:
            if recorder:
                recorder.add(item)
            if item.ref.frame_index < config.start_frame:
                continue
            if queue:
                queue.poll()
                if manager.session.session_status == "CLOSED":
                    reason = "SESSION_COMPLETE"
                    break
            if manager.session is None:
                people, objects = perception.detect(item.image, item.ref)
                initial_box = select_person(people, config.authentication.target_bbox, config.tracking.min_iou)
                auth = external or operator_authenticate(config.occurrence, item.ref,
                    acknowledged=config.authentication.operator_confirmed)
                if auth.frame != item.ref:
                    raise ValueError("Authentication frame does not match source/start_frame")
                session = manager.start(config.occurrence, auth)
                write_json(run_dir / "session-initial.json", session)
                tracking = TargetTracker(initial_box, config.tracking)
                detector = EventDetector(session.context, resource(detector_path), config.detection)
                queue = VerificationQueue(run_dir, config, manager, backend=backend, detect_only=detect_only)
            else:
                objects = None
            ref = item.ref.model_copy(update={"session_ms": item.ref.source_ms - manager.session.source_time_origin_ms})
            item = SourceFrame(ref, item.image)
            video.add(item)
            processed_frames += 1
            if last_analysis_ms is None or ref.source_ms - last_analysis_ms >= config.detection.analysis_interval_ms:
                if objects is None:
                    people, objects = perception.detect(item.image, ref)
                state = tracking.update(manager.session.context, ref, people)
                session = manager.update(state)
                landmarks = perception.landmarks(item.image, state)
                updates(detector.process(state, objects, landmarks), session)
                append_json(run_dir / "observations.jsonl", {
                    "schema_version": "1.0", "tracking": state.model_dump(),
                    "objects": objects.model_dump(), "landmarks": landmarks.model_dump()})
                last_analysis_ms = ref.source_ms
                analysis_frames += 1
            media_ready()
        if recorder:
            recording = recorder.close(reason)
        if manager.session is None:
            raise ValueError("No input frame at configured start_frame")
        updates(detector.flush(), manager.session)
        media_ready(eof=True)
        queue.drain()
        video.cleanup()
        summary = {"schema_version": "1.0", "run_dir": str(run_dir), "run_status": "FINISHED",
            "stop_reason": reason, "mode": "DETECT_ONLY" if detect_only else "FULL",
            "frames_buffered": processed_frames, "frames_analyzed": analysis_frames,
            "recording": recording,
            "requests": len(list((run_dir / "queue").glob("*.json"))),
            "results": len(queue.received), "processing_errors": sum(r.processing_status != "OK" for r in queue.received),
            "session": manager.view().model_dump(mode="json"),
            "message": "영상 기반 복약 완료" if manager.completion else "복약 확인 중"}
        write_json(run_dir / "summary.json", summary, immutable=False)
        return summary
    except BaseException as exc:
        failed = True
        reason = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "ERROR"
        write_json(run_dir / "run-error.json", {"schema_version": "1.0", "run_dir": str(run_dir),
            "run_status": reason,
            "recording_metadata": str(recorder.metadata_path) if recorder else None,
            "error": f"{type(exc).__name__}: {exc}", "at": utc_now()}, immutable=False)
        raise
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(running.unlink, missing_ok=True)
            if queue:
                cleanup.callback(queue.close)
            if perception:
                cleanup.callback(perception.close)
            if hasattr(source, "close"):
                cleanup.callback(source.close)
            if recorder and not recorder.closed:
                # Finalize the recording before waiting on slow VLM shutdown.
                cleanup.callback(close_recording)
