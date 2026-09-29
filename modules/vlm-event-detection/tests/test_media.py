from fractions import Fraction

import av
import pytest
from PIL import Image

from vlm_verification.contracts import Clip, ClipFrame, FrameRef, ResourceRef, RoiFrame, TimeRange
from vlm_verification.media import (
    MediaMappingError,
    MediaUnavailableError,
    TargetUnresolvedError,
    load_clip_images,
    resolve_media_path,
    sample_frame_indices,
)


@pytest.fixture
def variable_rate_clip(tmp_path):
    path = tmp_path / "variable.mp4"
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=1000)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index, timestamp in enumerate([10, 110, 310, 450, 900]):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (index * 40, 30, 50)))
            frame.pts = timestamp
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    frames = []
    with av.open(str(path)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            timestamp = round(frame.pts * frame.time_base * 1000)
            frames.append(
                ClipFrame(
                    clip_frame_index=index,
                    clip_ms=timestamp,
                    source=FrameRef(
                        stream_id="stream",
                        frame_index=index + 10,
                        source_ms=timestamp + 1000,
                        session_ms=timestamp,
                        width=64,
                        height=64,
                    ),
                )
            )
    return Clip(
        clip_id="clip",
        video=ResourceRef(resource_id="video", locator=str(path), media_type="video/mp4"),
        actual_range=TimeRange(start_ms=frames[0].clip_ms, end_ms=frames[-1].clip_ms),
        frames=frames,
        rois=[],
    )


def test_variable_rate_pts_and_order_are_preserved(variable_rate_clip):
    clip = variable_rate_clip
    intervals = [b.clip_ms - a.clip_ms for a, b in zip(clip.frames, clip.frames[1:])]
    assert len(set(intervals)) > 1
    loaded = load_clip_images(clip, max_frames=3, timestamp_tolerance_ms=0)
    assert [image.frame.clip_frame_index for image in loaded] == [0, 2, 4]
    assert [image.frame.source.frame_index for image in loaded] == [10, 12, 14]
    assert all(image.image.size == (64, 64) for image in loaded)
    assert all(image.source_bbox == [0, 0, 1, 1] for image in loaded)


def test_decoder_does_not_accept_invented_pts(variable_rate_clip):
    variable_rate_clip.frames[1].clip_ms += 10
    with pytest.raises(MediaMappingError, match="PTS"):
        load_clip_images(variable_rate_clip, max_frames=2)


def test_decoder_checks_unsampled_dimensions(variable_rate_clip):
    variable_rate_clip.frames[1].source.width = 32
    with pytest.raises(MediaMappingError, match="dimensions"):
        load_clip_images(variable_rate_clip, max_frames=2)


def test_decoder_checks_total_frame_count(variable_rate_clip):
    variable_rate_clip.frames.pop()
    with pytest.raises(MediaMappingError, match="more frames"):
        load_clip_images(variable_rate_clip, max_frames=2)


def test_rois_supplement_full_context_and_record_original_bbox(variable_rate_clip, tmp_path):
    path = tmp_path / "roi.png"
    Image.new("RGB", (20, 30)).save(path)
    variable_rate_clip.rois.append(
        RoiFrame(
            roi_id="roi-0",
            clip_frame_index=0,
            bbox=[0.1, 0.2, 0.4, 0.8],
            image=ResourceRef(resource_id="roi", locator=str(path), media_type="image/png"),
            width=20,
            height=30,
        )
    )
    loaded = load_clip_images(variable_rate_clip, max_frames=2, include_rois=True)
    assert [item.roi_id for item in loaded] == [None, "roi-0", None]
    assert loaded[1].source_bbox == [0.1, 0.2, 0.4, 0.8]
    assert loaded[1].image.size == (20, 30)
    assert loaded[1].frame == loaded[0].frame


def test_missing_media_is_not_run(variable_rate_clip):
    variable_rate_clip.video.locator += ".missing"
    with pytest.raises(MediaUnavailableError):
        load_clip_images(variable_rate_clip, max_frames=2)


def test_local_uri_and_remote_media_policy(tmp_path):
    local = tmp_path / "clip with spaces.mp4"
    assert resolve_media_path(local.as_uri()) == local.resolve()
    assert resolve_media_path("relative.mp4", tmp_path) == tmp_path / "relative.mp4"
    with pytest.raises(MediaUnavailableError):
        resolve_media_path("https://example.invalid/clip.mp4")


def test_required_target_frame_retains_endpoints():
    assert sample_frame_indices(12, 4, (2,)) == [0, 2, 7, 11]
    with pytest.raises(TargetUnresolvedError):
        sample_frame_indices(12, 2, (2,))
    assert sample_frame_indices(2, 8) == [0, 1]
    assert sample_frame_indices(1, 8) == [0]
