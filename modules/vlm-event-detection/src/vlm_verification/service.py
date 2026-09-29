"""Candidate-scoped orchestration, provenance and immutable result persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any
from uuid import uuid4

from .config import RULE_VERSION, ExecutionConfig
from .contracts import (
    ModelInputRecord,
    ProcessingError,
    ResourceRef,
    SessionContext,
    VerificationRequest,
    VerificationResult,
)
from .media import MediaUnavailableError, TargetUnresolvedError, resolve_media_path
from .response import ResponseError, derive_verification, parse_response


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class InputRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RecordConflict(ValueError):
    """An immutable ID was reused with different contents."""


class RequestInProgress(RuntimeError):
    """Another process owns this request's output directory."""


def validate_links(
    request: VerificationRequest, session: SessionContext, config: ExecutionConfig
) -> None:
    """Validate inputs already declared ready for execution; never poll missing inputs."""
    def reject(message: str) -> None:
        raise InputRejected("LINK_MISMATCH", message)

    if request.context != session.context:
        reject("Request and session context differ")
    for name in ("event", "detection", "media"):
        value = getattr(request, name)
        if value.context != request.context:
            reject(f"{name}.context differs from request.context")
        if value.candidate != request.event.candidate:
            reject(f"{name}.candidate differs from the frozen candidate revision")
    if request.event.action_range.end_ms is None:
        reject("Only a closed action range can be verified")
    if (request.model_id, request.model_revision, request.prompt_version) != (
        config.model_id, config.model_revision, config.prompt_version
    ):
        reject("Request model/prompt settings differ from execution_config")
    if request.created_at > utc_now():
        reject("Request created_at is in the future")
    if session.tracking is not None:
        if session.tracking.context != session.context:
            reject("Tracking context differs from session context")
        if session.tracking.tracking_status == "UNRESOLVED":
            raise InputRejected("TARGET_UNRESOLVED", "Session target tracking is unresolved")
    clip = request.media.clip
    if clip is None:
        raise InputRejected("MISSING_VIDEO", request.media.unavailable_reason)
    requested = request.media.requested_range
    action = request.event.action_range
    if requested.start_ms > action.start_ms or requested.end_ms < action.end_ms:
        reject("Requested media range must contain the closed action range")
    if not (requested.start_ms <= clip.actual_range.start_ms
            <= clip.actual_range.end_ms <= requested.end_ms):
        reject("Actual clip range falls outside the requested range")
    if [frame.clip_frame_index for frame in clip.frames] != list(range(len(clip.frames))):
        reject("Clip frame indices must be contiguous from zero")
    if (clip.frames[0].source.session_ms, clip.frames[-1].source.session_ms) != (
        clip.actual_range.start_ms, clip.actual_range.end_ms
    ):
        reject("Actual clip range must match the first and last frame")
    for previous, current in zip(clip.frames, clip.frames[1:]):
        if not (previous.clip_ms < current.clip_ms
                and previous.source.frame_index < current.source.frame_index
                and previous.source.source_ms < current.source.source_ms):
            reject("Clip/source frame indices and timestamps must increase strictly")
    roi_ids = set()
    for roi in clip.rois:
        if roi.roi_id in roi_ids or roi.clip_frame_index >= len(clip.frames):
            reject("ROI IDs must be unique and reference existing clip frames")
        roi_ids.add(roi.roi_id)
    if not request.event.regions or not any(
        region.region_type == "TARGET" for region in request.event.regions
    ):
        raise InputRejected("TARGET_UNRESOLVED", "No target region identifies the person in the clip")

    frames = [frame.source for frame in clip.frames]
    frames += [sample.frame for sample in request.detection.samples]
    frames += [region.frame for region in request.event.regions]
    if session.tracking is not None:
        frames += [session.tracking.frame]
    for frame in frames:
        if frame.stream_id != request.context.stream_id:
            reject("Frame stream_id differs from request stream")
        if frame.session_ms != frame.source_ms - session.source_time_origin_ms:
            reject("Frame session_ms does not match source time origin")
    source_frames = {frame.source.frame_index: frame.source for frame in clip.frames}
    target_frames = [region for region in request.event.regions if region.region_type == "TARGET"]
    if not any(region.frame.frame_index in source_frames for region in target_frames):
        raise InputRejected("TARGET_UNRESOLVED", "Target regions have no frame in the input clip")
    for region in request.event.regions:
        if region.frame.frame_index in source_frames and (
            region.frame != source_frames[region.frame.frame_index]
        ):
            reject("Candidate region frame metadata conflicts with the clip")
        if region.object_track_id is not None and (
            region.object_track_id not in request.event.object_track_ids
        ):
            reject("Region refers to an object outside the candidate")
    for sample in request.detection.samples:
        if sample.frame.frame_index in source_frames and (
            sample.frame != source_frames[sample.frame.frame_index]
        ):
            reject("Detection frame metadata conflicts with the clip")
        for motion in sample.motions.items:
            for ref in motion.entity_refs:
                if ref.startswith("object:") and ref[7:] not in request.event.object_track_ids:
                    reject("Motion refers to an object outside the candidate")


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _immutable_write(path: Path, value: Any) -> None:
    content = _json(value)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        # Configs saved before optional backends existed mean Transformers.
        # Normalize defaults for comparison only; never rewrite old records.
        if isinstance(value, ExecutionConfig):
            stored = ExecutionConfig.model_validate(stored).model_dump(mode="json")
        if stored != json.loads(content):
            raise RecordConflict(f"Immutable record differs: {path}")
        return
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _validate_input_record(record: ModelInputRecord, request: VerificationRequest) -> None:
    if record.request_id != request.request_id:
        raise ValueError("ModelInputRecord belongs to a different request")
    clip = request.media.clip
    rois = {roi.roi_id: roi for roi in clip.rois}
    target_sources = {region.frame.frame_index for region in request.event.regions
                      if region.region_type == "TARGET"}
    if not any(frame.clip_frame_index < len(clip.frames)
               and clip.frames[frame.clip_frame_index].source.frame_index in target_sources
               and frame.roi_id is None and frame.source_bbox == [0, 0, 1, 1]
               for frame in record.frames):
        raise ValueError("Actual full-frame inputs have no mapped target region")
    for index, frame in enumerate(record.frames):
        if frame.input_index != index or frame.clip_frame_index >= len(clip.frames):
            raise ValueError("Invalid model input index or clip frame mapping")
        expected_bbox = [0, 0, 1, 1]
        if frame.roi_id is not None:
            roi = rois.get(frame.roi_id)
            if roi is None or roi.clip_frame_index != frame.clip_frame_index:
                raise ValueError("Model input references an unknown ROI")
            expected_bbox = roi.bbox
        x0, y0, x1, y1 = frame.source_bbox
        if not (expected_bbox[0] <= x0 < x1 <= expected_bbox[2]
                and expected_bbox[1] <= y0 < y1 <= expected_bbox[3]):
            raise ValueError("Model input crop falls outside the referenced source region")


class VerificationService:
    """Synchronous worker boundary. Detection can use VerificationWorker.submit()."""

    def __init__(self, backend: Any, config: ExecutionConfig, output_dir: Path,
                 media_root: Path = Path(".")):
        self.backend = backend
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.media_root = Path(media_root).resolve()

    def verify(self, request: VerificationRequest, session: SessionContext) -> VerificationResult:
        # Reparse snapshots so caller mutations cannot change running requests.
        request = VerificationRequest.model_validate_json(request.model_dump_json())
        session = SessionContext.model_validate_json(session.model_dump_json())
        key = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()
        directory = self.output_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / ".running"
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RequestInProgress(f"Request is running or was interrupted: {directory}") from exc
        try:
            os.close(lock_fd)
            _immutable_write(directory / "request.json", request)
            _immutable_write(directory / "execution_config.json", self.config)
            result_file = directory / "result.json"
            if result_file.exists():
                return VerificationResult.model_validate_json(result_file.read_text(encoding="utf-8"))
            _immutable_write(directory / "session.json", session)
            # A partial previous attempt must not silently rerun inference with the same ID.
            state_file = directory / "states.jsonl"
            if state_file.exists():
                raise RequestInProgress("Incomplete persisted attempt; submit a new request_id")
            return self._execute(request, session, directory)
        finally:
            lock.unlink(missing_ok=True)

    def _execute(self, request: VerificationRequest, session: SessionContext,
                 directory: Path) -> VerificationResult:
        started_at = None

        def state(status: str, processing: str | None = None, finished: str | None = None):
            entry = {"schema_version": "1.0", "request_id": request.request_id,
                     "request_status": status, "started_at": started_at,
                     "finished_at": finished, "processing_status": processing}
            with (directory / "states.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")

        state("QUEUED")
        started_at = utc_now()
        state("RUNNING")
        input_ref = None
        raw_output = None
        received_at = None
        response = None
        verification = None
        error = None
        status = "OK"
        stage = "INPUT"
        try:
            validate_links(request, session, self.config)
            if request.execution_config.media_type != "application/json":
                raise InputRejected("LINK_MISMATCH", "execution_config must refer to JSON")
            try:
                config_path = resolve_media_path(request.execution_config.locator, self.media_root)
                frozen_config = ExecutionConfig.model_validate_json(config_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError, MediaUnavailableError) as exc:
                raise InputRejected("LINK_MISMATCH", f"Cannot load execution_config: {exc}") from exc
            if frozen_config != self.config:
                raise InputRejected("LINK_MISMATCH", "Referenced execution_config differs from settings")
            video_path = resolve_media_path(request.media.clip.video.locator, self.media_root)
            if not video_path.is_file():
                raise InputRejected("MISSING_VIDEO", f"Video is not a local file: {video_path}")
            stage = "PREPROCESS"
            prepared = self.backend.prepare(request, self.config)
            record = ModelInputRecord.model_validate_json(prepared.record.model_dump_json())
            _validate_input_record(record, request)
            input_file = directory / "model_input.json"
            _immutable_write(input_file, record)
            input_ref = ResourceRef(resource_id=f"input-{directory.name}",
                                    locator=str(input_file), media_type="application/json")
            versions = {}
            for name in ("pydantic", "transformers", "torch", "Pillow", "av", "accelerate"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            _immutable_write(directory / "runtime.json", {
                "python": platform.python_version(), "platform": platform.platform(),
                "libraries": versions, "backend": type(self.backend).__name__,
            })
            stage = "INFERENCE"
            raw_output = self.backend.generate(prepared, self.config)
            received_at = utc_now()
            if not isinstance(raw_output, str):
                raw_output = None
                received_at = None
                raise TypeError("Backend must return the unmodified response string")
            stage = "PARSE"
            response = parse_response(raw_output, request.event.candidate.candidate_type,
                                      len(record.frames))
            verification = derive_verification(response, request.event.candidate.candidate_type)
        except InputRejected as exc:
            status = "NOT_RUN"
            error = ProcessingError(stage="INPUT", code=exc.code, message=str(exc))
        except ResponseError as exc:
            status = "ERROR"
            error = ProcessingError(stage=exc.stage, code=exc.code, message=str(exc))
        except TargetUnresolvedError as exc:
            status = "NOT_RUN"
            error = ProcessingError(stage="INPUT", code="TARGET_UNRESOLVED", message=str(exc))
        except MediaUnavailableError as exc:
            status = "NOT_RUN"
            error = ProcessingError(stage="INPUT", code="MISSING_VIDEO", message=str(exc))
        except Exception as exc:
            # Backend failures never become REFUTED or a valid UNKNOWN answer.
            status = "ERROR"
            code = {"PREPROCESS": "PREPROCESS_FAILED", "INFERENCE": "INFERENCE_FAILED",
                    "PARSE": "INVALID_RESPONSE"}.get(stage, "PREPROCESS_FAILED")
            error = ProcessingError(stage=stage if stage != "INPUT" else "PREPROCESS",
                                    code=code, message=f"{type(exc).__name__}: {exc}")
        finished = utc_now()
        result = VerificationResult(
            schema_version="1.0", request_id=request.request_id, context=request.context,
            candidate=request.event.candidate, action_range=request.event.action_range,
            clip_id=request.media.clip.clip_id if request.media.clip else None,
            input_record=input_ref, processing_status=status,
            result=response if status == "OK" else None,
            verification=verification if status == "OK" else None,
            verification_rule_version=RULE_VERSION, raw_output=raw_output,
            error=error, response_received_at=received_at, finished_at=finished,
        )
        _immutable_write(directory / "result.json", result)
        state("FINISHED", status, finished)
        return result


class VerificationWorker:
    """One model worker with bounded admission; submit returns without blocking capture."""

    def __init__(self, service: VerificationService, max_pending: int = 32):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.service = service
        self._slots = BoundedSemaphore(max_pending)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")

    def submit(self, request: VerificationRequest, session: SessionContext) -> Future:
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("Verification queue is full; retain candidate media and retry later")
        try:
            future = self._executor.submit(self.service.verify,
                                           request.model_copy(deep=True), session.model_copy(deep=True))
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
        if wait and hasattr(self.service.backend, "close"):
            self.service.backend.close()
