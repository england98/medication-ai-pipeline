from pathlib import Path

import pytest
from conftest import frame
from medication_contracts import CandidateKey, CandidateRegion, EventCandidate, TimeRange
from PIL import Image
from session_video.video import SourceFrame, VideoConfig, VideoManager, read_video
from vlm_verification.media import load_clip_images


def event(session, frames, end=None):
    return EventCandidate(context=session.context, candidate=CandidateKey(candidate_id="candidate", candidate_type="E02", candidate_revision=1),
        created_at="2026-09-27T00:00:00.000Z", updated_at="2026-09-27T00:00:00.000Z",
        action_range=TimeRange(start_ms=frames[0].session_ms, end_ms=end), object_track_ids=[],
        regions=[CandidateRegion(frame=f, region_type=kind, bbox=[0.1, 0.1, 0.9, 0.9], object_track_id=None)
                 for f in frames for kind in ("TARGET", "HAND")])


def test_vfr_pts_clipped_context_and_roi_mapping(tmp_path, session):
    manager = VideoManager(tmp_path, VideoConfig(buffer_ms=100,
        pre_ms={k: 100 for k in ("E01", "E02", "E03")}, post_ms={k: 150 for k in ("E01", "E02", "E03")}))
    refs = [frame(session.context, i, ms, width=65, height=49, origin=500) for i, ms in enumerate([0, 37, 104, 220, 365])]
    manager.add(SourceFrame(refs[0], Image.new("RGB", (65, 49), "red")))
    candidate = event(session, refs)
    manager.update(candidate)
    for ref in refs[1:]:
        manager.add(SourceFrame(ref, Image.new("RGB", (65, 49), "blue")))
    manager.update(candidate.model_copy(update={"candidate": candidate.candidate.model_copy(update={"candidate_revision": 2}),
        "action_range": TimeRange(start_ms=0, end_ms=365)}))
    assert not manager.ready()
    media = manager.ready(eof=True)[0]
    assert media.requested_range.end_ms == 515
    assert media.clip.actual_range.end_ms == 365
    assert [f.clip_ms for f in media.clip.frames] == [0, 37, 104, 220, 365]
    assert [f.source.source_ms for f in media.clip.frames] == [500, 537, 604, 720, 865]
    assert len(load_clip_images(media.clip, max_frames=5, include_rois=True)) == 10
    assert [f.ref.source_ms for f in read_video(media.clip.video.locator, "other-stream")] == [0, 37, 104, 220, 365]
    manager.cleanup()
    assert not list((tmp_path / "spool").glob("*.jpg"))
    assert Path(media.clip.video.locator).exists()


def test_pin_prevents_ring_eviction_and_revision_conflict(tmp_path, session):
    manager = VideoManager(tmp_path, VideoConfig(buffer_ms=50))
    ref = frame(session.context, 0, 0)
    manager.add(SourceFrame(ref, Image.new("RGB", (64, 48))))
    candidate = event(session, [ref])
    manager.update(candidate)
    manager.add(SourceFrame(frame(session.context, 1, 1000), Image.new("RGB", (64, 48))))
    assert len(manager.frames) == 2
    with pytest.raises(RuntimeError, match="reference"):
        manager.cleanup()
    with pytest.raises(ValueError, match="Conflicting"):
        manager.update(candidate.model_copy(update={"object_track_ids": ["different"]}))


def test_disk_limit_preserves_frame(tmp_path, session):
    manager = VideoManager(tmp_path, VideoConfig(max_spool_bytes=1))
    with pytest.raises(RuntimeError, match="max_spool_bytes"):
        manager.add(SourceFrame(frame(session.context, 0, 0), Image.new("RGB", (64, 48))))
    assert list((tmp_path / "spool").glob("*.jpg"))
