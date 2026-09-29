"""Local media loading with explicit PTS-to-contract correspondence.

The clip frame table is authoritative. We never derive timestamps from an FPS
value, silently rebase timestamps, or substitute a nearby decoded frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from .contracts import Clip, ClipFrame


class MediaUnavailableError(RuntimeError):
    """Required local media cannot be obtained; inference was not run."""


class MediaMappingError(RuntimeError):
    """Available media disagrees with its declared frame/ROI mapping."""


class TargetUnresolvedError(MediaUnavailableError):
    """The sampled images cannot retain target identity within the image budget."""


@dataclass(frozen=True)
class LoadedImage:
    image: Any  # PIL.Image.Image; optional inference dependency is loaded lazily.
    frame: ClipFrame
    roi_id: str | None
    source_bbox: list[float]


def resolve_media_path(locator: str, media_root: Path | str = Path(".")) -> Path:
    """Resolve local paths and file URIs without downloading remote resources."""
    # A Windows drive letter is a path, not a URI scheme.
    if len(locator) > 1 and locator[1] == ":":
        path = Path(locator)
    else:
        parsed = urlparse(locator)
        if parsed.scheme and parsed.scheme != "file":
            raise MediaUnavailableError("Only local media paths and file: URIs are supported")
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise MediaUnavailableError("Remote file URI hosts are not supported")
            path = Path(url2pathname(parsed.path))
        else:
            path = Path(locator)
    if not path.is_absolute():
        path = Path(media_root) / path
    return path.resolve()


def sample_frame_indices(
    frame_count: int, max_frames: int, required_indices: tuple[int, ...] = ()
) -> list[int]:
    """Uniform index sampling, retaining both ends whenever two images fit."""
    if frame_count < 1 or max_frames < 1:
        raise ValueError("frame_count and max_frames must be positive")
    count = min(frame_count, max_frames)
    if any(index < 0 or index >= frame_count for index in required_indices):
        raise ValueError("A required frame index is outside the clip")
    selected = (
        {frame_count // 2}
        if count == 1
        else {round(i * (frame_count - 1) / (count - 1)) for i in range(count)}
    )
    protected = set(required_indices)
    if count > 1:
        protected.update((0, frame_count - 1))
    if len(protected) > count:
        raise TargetUnresolvedError("Frame budget cannot retain target identity and clip endpoints")
    for required in sorted(protected - selected):
        replaceable = selected - protected
        closest = min(replaceable, key=lambda index: (abs(index - required), index))
        selected.remove(closest)
        selected.add(required)
    return sorted(selected)


def _require_file(path: Path) -> None:
    try:
        available = path.is_file()
    except OSError as exc:
        raise MediaUnavailableError(f"Cannot access media: {path}") from exc
    if not available:
        raise MediaUnavailableError(f"Media file is unavailable: {path}")


def load_clip_images(
    clip: Clip,
    *,
    max_frames: int,
    include_rois: bool = False,
    timestamp_tolerance_ms: int = 2,
    media_root: Path | str = Path("."),
    required_indices: tuple[int, ...] = (),
) -> list[LoadedImage]:
    """Decode the clip, verify every frame, and retain a bounded image sequence.

    Supplied ROI images supplement full frames; they never replace scene context.
    All ROI images for sampled frames are included in deterministic input order.
    """
    if timestamp_tolerance_ms < 0:
        raise ValueError("timestamp_tolerance_ms must be nonnegative")
    path = resolve_media_path(clip.video.locator, media_root)
    _require_file(path)
    if clip.video.media_type not in {"video/mp4", "video/quicktime"}:
        raise MediaMappingError("The local backend currently accepts MP4/MOV clips only")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install the inference extra to load video (PyAV is missing)") from exc

    selected = sample_frame_indices(len(clip.frames), max_frames, required_indices)
    selected_set = set(selected)
    full_images: dict[int, LoadedImage] = {}
    count = 0
    try:
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise MediaMappingError("The clip must contain exactly one video stream")
            for index, decoded in enumerate(container.decode(video=0)):
                if index >= len(clip.frames):
                    raise MediaMappingError("Decoded clip contains more frames than Clip.frames")
                declared = clip.frames[index]
                if declared.clip_frame_index != index:
                    raise MediaMappingError(
                        "Clip.frames indices must start at zero and be contiguous"
                    )
                if decoded.pts is None or decoded.time_base is None:
                    raise MediaMappingError(f"Frame {index} has no presentation timestamp")
                actual_ms = decoded.pts * decoded.time_base * 1000
                if abs(actual_ms - declared.clip_ms) > timestamp_tolerance_ms:
                    raise MediaMappingError(
                        f"Frame {index} PTS is {float(actual_ms):.3f} ms, "
                        f"but Clip.frames declares {declared.clip_ms} ms"
                    )
                if (decoded.width, decoded.height) != (
                    declared.source.width,
                    declared.source.height,
                ):
                    raise MediaMappingError(f"Frame {index} dimensions disagree with FrameRef")
                if index in selected_set:
                    full_images[index] = LoadedImage(
                        image=decoded.to_image().convert("RGB"),
                        frame=declared,
                        roi_id=None,
                        source_bbox=[0.0, 0.0, 1.0, 1.0],
                    )
                count += 1
    except (FileNotFoundError, PermissionError) as exc:
        raise MediaUnavailableError(f"Cannot read media: {path}") from exc
    except MediaMappingError:
        raise
    except Exception as exc:
        raise MediaMappingError(f"Could not decode clip: {type(exc).__name__}: {exc}") from exc
    if count != len(clip.frames):
        raise MediaMappingError(
            f"Decoded {count} frames, but Clip.frames declares {len(clip.frames)}"
        )

    rois_by_frame: dict[int, list[Any]] = {}
    if include_rois:
        for roi in clip.rois:
            if roi.clip_frame_index in selected_set:
                rois_by_frame.setdefault(roi.clip_frame_index, []).append(roi)
    loaded: list[LoadedImage] = []
    for index in selected:
        loaded.append(full_images[index])
        for roi in rois_by_frame.get(index, []):
            roi_path = resolve_media_path(roi.image.locator, media_root)
            _require_file(roi_path)
            try:
                from PIL import Image

                with Image.open(roi_path) as image:
                    if image.size != (roi.width, roi.height):
                        raise MediaMappingError(
                            f"ROI {roi.roi_id} dimensions do not match RoiFrame"
                        )
                    rgb = image.convert("RGB")
            except (FileNotFoundError, PermissionError) as exc:
                raise MediaUnavailableError(f"Cannot read ROI: {roi_path}") from exc
            except MediaMappingError:
                raise
            except Exception as exc:
                raise MediaMappingError(f"Cannot decode ROI {roi.roi_id}: {exc}") from exc
            loaded.append(
                LoadedImage(
                    image=rgb,
                    frame=clip.frames[index],
                    roi_id=roi.roi_id,
                    source_bbox=list(roi.bbox),
                )
            )
    return loaded
