"""Disk-backed recent-frame buffer and pinned candidate evidence with real PTS."""

import math
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from medication_contracts import (
    CandidateMedia,
    Clip,
    ClipFrame,
    FrameRef,
    ResourceRef,
    RoiFrame,
    TimeRange,
)
from medication_contracts.records import new_id, write_json
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    buffer_ms: int = Field(default=5000, ge=1)
    pre_ms: dict[str, int] = Field(default_factory=lambda: {"E01": 1000, "E02": 1500, "E03": 1000})
    post_ms: dict[str, int] = Field(default_factory=lambda: {"E01": 1000, "E02": 1000, "E03": 1000})
    max_spool_bytes: int = Field(default=4_000_000_000, ge=1)
    roi_padding: float = Field(default=0.03, ge=0, le=0.5)
    max_roi_frames: int = Field(default=8, ge=0, le=128)

    @model_validator(mode="after")
    def event_windows(self):
        for windows in (self.pre_ms, self.post_ms):
            if set(windows) != {"E01", "E02", "E03"} or any(type(v) is not int or v < 0 for v in windows.values()):
                raise ValueError("Specify nonnegative millisecond windows for E01/E02/E03")
        return self


@dataclass
class SourceFrame:
    ref: FrameRef
    image: object


def read_video(source, stream_id, *, input_format=None, options=None):
    """Read a file or an explicit FFmpeg capture source; never invent file timestamps."""
    import av
    previous = None
    with av.open(str(source), format=input_format, options=options or {}) as container:
        if not container.streams.video:
            raise ValueError("Input contains no video stream")
        for index, decoded in enumerate(container.decode(video=0)):
            if decoded.pts is None or decoded.time_base is None:
                raise ValueError(f"Frame {index} has no source PTS")
            timestamp = round(decoded.pts * decoded.time_base * 1000)
            if timestamp < 0 or previous is not None and timestamp <= previous:
                raise ValueError(f"Non-increasing/negative source PTS at frame {index}: {timestamp}")
            previous = timestamp
            yield SourceFrame(FrameRef(stream_id=stream_id, frame_index=index,
                source_ms=timestamp, session_ms=None, width=decoded.width, height=decoded.height), decoded.to_image())


class VideoManager:
    def __init__(self, directory: Path, config: VideoConfig | None = None):
        self.directory = Path(directory).resolve()
        self.config = config or VideoConfig()
        for ranges in (self.config.pre_ms, self.config.post_ms):
            if set(ranges) != {"E01", "E02", "E03"} or any(type(v) is not int or v < 0 for v in ranges.values()):
                raise ValueError("Specify nonnegative millisecond windows for E01/E02/E03")
        self.spool = self.directory / "spool"
        self.spool.mkdir(parents=True, exist_ok=True)
        self.frames = OrderedDict()
        self.candidates = {}
        self.finished = {}
        self.total_bytes = 0
        self.latest = None

    def add(self, frame: SourceFrame):
        ref = frame.ref
        if ref.session_ms is None:
            raise ValueError("Buffer frames must belong to a session")
        if self.latest is not None and (ref.stream_id != self.latest.stream_id
                or ref.frame_index <= self.latest.frame_index or ref.source_ms <= self.latest.source_ms
                or ref.session_ms <= self.latest.session_ms
                or (ref.width, ref.height) != (self.latest.width, self.latest.height)):
            raise ValueError("Frame stream, size or timestamp discontinuity")
        if frame.image.size != (ref.width, ref.height):
            raise ValueError("Frame pixels do not match metadata")
        self.latest = ref
        path = self.spool / f"{ref.frame_index:012d}.jpg"
        frame.image.convert("RGB").save(path, quality=95, subsampling=0)
        size = path.stat().st_size
        self.frames[ref.frame_index] = (ref.model_copy(deep=True), path, size)
        self.total_bytes += size
        self._prune()
        if self.total_bytes > self.config.max_spool_bytes:
            raise RuntimeError("Pinned video exceeds max_spool_bytes; evidence retained, run stopped")

    def update(self, event):
        key = event.candidate.candidate_id
        if key in self.finished:
            previous, media = self.finished[key]
            if event != previous:
                raise ValueError("Cannot change a frozen candidate")
            return
        previous = self.candidates.get(key)
        if previous is not None:
            if event.context != previous.context or event.candidate.candidate_type != previous.candidate.candidate_type:
                raise ValueError("Candidate identity changed")
            if event.candidate.candidate_revision <= previous.candidate.candidate_revision:
                if event.candidate.candidate_revision == previous.candidate.candidate_revision and event != previous:
                    raise ValueError("Conflicting candidate revision")
                return
            if previous.action_range.end_ms is not None:
                raise ValueError("Closed candidate cannot be revised")
        self.candidates[key] = event.model_copy(deep=True)

    def _range(self, event):
        kind = event.candidate.candidate_type
        return TimeRange(start_ms=max(0, event.action_range.start_ms - self.config.pre_ms[kind]),
            end_ms=(event.action_range.end_ms + self.config.post_ms[kind]
                    if event.action_range.end_ms is not None else None))

    def _prune(self):
        if self.latest is None:
            return
        cutoff = self.latest.session_ms - self.config.buffer_ms
        ranges = [self._range(event) for event in self.candidates.values()]
        for index, (ref, path, size) in list(self.frames.items()):
            pinned = any(r.start_ms <= ref.session_ms and (r.end_ms is None or ref.session_ms <= r.end_ms) for r in ranges)
            if ref.session_ms < cutoff and not pinned:
                path.unlink()
                self.total_bytes -= size
                del self.frames[index]

    def ready(self, *, eof=False):
        output = []
        for key, event in list(self.candidates.items()):
            requested = self._range(event)
            if requested.end_ms is None:
                continue
            if not eof and (self.latest is None or self.latest.session_ms < requested.end_ms):
                continue
            media = self._build(event, requested)
            self.finished[key] = event, media
            del self.candidates[key]
            output.append(media)
        self._prune()
        return output

    def _build(self, event, requested):
        import av
        from PIL import Image
        selected = [(ref, path) for ref, path, _ in self.frames.values()
                    if requested.start_ms <= ref.session_ms <= requested.end_ms]
        if not selected:
            return CandidateMedia(context=event.context, candidate=event.candidate,
                requested_range=requested, clip=None, unavailable_reason="No frames retained in requested interval")
        clip_id = new_id("clip")
        directory = self.directory / "clips" / clip_id
        directory.mkdir(parents=True)
        video_path = directory / "video.mp4"
        first = selected[0][0]
        table = []
        # Explicit millisecond PTS; variable frame intervals survive encoding.
        with av.open(str(video_path), "w") as out:
            stream = out.add_stream("libx264", rate=30)
            stream.width, stream.height = first.width, first.height
            stream.pix_fmt = "yuv444p"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.options = {"crf": "18", "preset": "fast", "bf": "0"}
            for index, (ref, path) in enumerate(selected):
                pts = ref.source_ms - first.source_ms
                with Image.open(path) as image:
                    frame = av.VideoFrame.from_image(image.convert("RGB"))
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    out.mux(packet)
                table.append(ClipFrame(clip_frame_index=index, clip_ms=pts, source=ref))
            for packet in stream.encode():
                out.mux(packet)
        rois = []
        region_map = {}
        for region in event.regions:
            if region.region_type != "TARGET":
                region_map.setdefault(region.frame.frame_index, []).append(region.bbox)
        eligible = [index for index, (ref, _) in enumerate(selected) if ref.frame_index in region_map]
        count = min(len(eligible), self.config.max_roi_frames)
        indices = sorted({eligible[round(i * (len(eligible) - 1) / max(1, count - 1))] for i in range(count)})
        for index in indices:
            ref, path = selected[index]
            boxes = region_map[ref.frame_index]
            pad = self.config.roi_padding
            left = max(0, math.floor((min(b[0] for b in boxes) - pad) * ref.width))
            top = max(0, math.floor((min(b[1] for b in boxes) - pad) * ref.height))
            right = min(ref.width, math.ceil((max(b[2] for b in boxes) + pad) * ref.width))
            bottom = min(ref.height, math.ceil((max(b[3] for b in boxes) + pad) * ref.height))
            roi_id = new_id("roi")
            roi_path = directory / f"{roi_id}.jpg"
            with Image.open(path) as image:
                image.crop((left, top, right, bottom)).save(roi_path, quality=95, subsampling=0)
            rois.append(RoiFrame(roi_id=roi_id, clip_frame_index=index,
                bbox=[left / ref.width, top / ref.height, right / ref.width, bottom / ref.height],
                image=ResourceRef(resource_id=roi_id, locator=str(roi_path), media_type="image/jpeg"),
                width=right - left, height=bottom - top))
        media = CandidateMedia(context=event.context, candidate=event.candidate,
            requested_range=requested, unavailable_reason=None,
            clip=Clip(clip_id=clip_id, video=ResourceRef(resource_id=clip_id,
                locator=str(video_path), media_type="video/mp4"),
                actual_range=TimeRange(start_ms=first.session_ms, end_ms=selected[-1][0].session_ms),
                frames=table, rois=rois))
        write_json(directory / "media.json", media)
        return media

    def cleanup(self):
        """Remove only owned temporary frames after all clips have been materialized."""
        if self.candidates:
            raise RuntimeError("Cannot release frames while candidates still reference them")
        for _, path, _ in self.frames.values():
            path.unlink(missing_ok=True)
        self.frames.clear()
        self.total_bytes = 0
