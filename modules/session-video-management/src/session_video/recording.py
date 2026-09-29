"""Incremental recording of received frames, independent of candidate detection."""

from fractions import Fraction
from pathlib import Path

from medication_contracts.records import append_json, utc_now, write_json


class InputRecorder:
    """Encode every supplied frame with its original relative millisecond PTS."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "input.mp4"
        self.index_path = self.directory / "frames.jsonl"
        self.metadata_path = self.directory / "metadata.json"
        self.container = None
        self.stream = None
        self.first = None
        self.last = None
        self.frame_count = 0
        self.closed = False
        self.started_at = utc_now()
        self.finished_at = None
        self.stop_reason = None
        self.error = None
        self.encoded_size = None
        write_json(self.metadata_path, self.metadata())

    def metadata(self):
        return {
            "schema_version": "1.0",
            "status": "ERROR" if self.error else (
                "FINALIZED" if self.closed and self.frame_count else "EMPTY" if self.closed else "RECORDING"),
            "stop_reason": self.stop_reason,
            "video_path": str(self.path) if self.frame_count else None,
            "frame_index_path": str(self.index_path),
            "frame_count": self.frame_count,
            "source_time_origin_ms": self.first.source_ms if self.first else None,
            "first_source_frame": self.first.model_dump(mode="json") if self.first else None,
            "last_source_frame": self.last.model_dump(mode="json") if self.last else None,
            "source_size": [self.first.width, self.first.height] if self.first else None,
            "encoded_size": self.encoded_size,
            "padding": [0, 0, self.first.width % 2, self.first.height % 2] if self.first else None,
            "codec": "h264", "pixel_format": "yuv420p", "audio": False,
            "timestamp_mode": "source_ms_relative_to_first_received_frame",
            "started_at": self.started_at, "finished_at": self.finished_at,
            "error": self.error,
        }

    def add(self, item):
        import av
        from PIL import Image

        if self.closed:
            raise RuntimeError("Recording is already closed")
        ref = item.ref
        received_at = utc_now()
        try:
            if item.image.size != (ref.width, ref.height):
                raise ValueError("Recording pixels do not match frame metadata")
            if self.last is not None and (
                ref.stream_id != self.last.stream_id or ref.frame_index <= self.last.frame_index
                or ref.source_ms <= self.last.source_ms
                or (ref.width, ref.height) != (self.last.width, self.last.height)
            ):
                raise ValueError("Recording frame stream, size or timestamp discontinuity")
            if self.container is None:
                self.container = av.open(str(self.path), "w", format="mp4")
                self.stream = self.container.add_stream("libx264", rate=30)
                # Pad odd sizes on the right/bottom; do not stretch source pixels.
                self.encoded_size = [ref.width + ref.width % 2, ref.height + ref.height % 2]
                self.stream.width, self.stream.height = self.encoded_size
                self.stream.pix_fmt = "yuv420p"
                self.stream.time_base = self.stream.codec_context.time_base = Fraction(1, 1000)
                self.stream.codec_context.thread_count = 2
                self.stream.options = {"crf": "18", "preset": "veryfast", "tune": "zerolatency", "bf": "0"}
                self.first = ref.model_copy(deep=True)
            image = item.image.convert("RGB")
            if image.size != tuple(self.encoded_size):
                padded = Image.new("RGB", tuple(self.encoded_size))
                padded.paste(image, (0, 0))
                image = padded
            frame = av.VideoFrame.from_image(image)
            recording_ms = ref.source_ms - self.first.source_ms
            frame.pts = recording_ms
            frame.time_base = Fraction(1, 1000)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            append_json(self.index_path, {
                "recording_frame_index": self.frame_count,
                "recording_ms": recording_ms,
                "received_at": received_at,
                "source": ref.model_dump(mode="json"),
            })
            self.frame_count += 1
            self.last = ref.model_copy(deep=True)
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def close(self, stop_reason):
        if self.closed:
            return self.metadata()
        self.stop_reason = stop_reason
        try:
            try:
                if self.stream is not None:
                    for packet in self.stream.encode():
                        self.container.mux(packet)
            finally:
                if self.container is not None:
                    self.container.close()
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.closed = True
            self.finished_at = utc_now()
            write_json(self.metadata_path, self.metadata(), immutable=False)
        return self.metadata()
