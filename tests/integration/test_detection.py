import pytest
from conftest import frame
from event_detection import DetectionConfig, EventDetector
from medication_contracts import (
    DetectedObject,
    Landmark,
    ObservationSet,
    ResourceRef,
    TrackingUpdate,
)


def sample(session, index, ms, *, near=True, tracked=True, cup=False, medication=False):
    tracking = TrackingUpdate(context=session.context, frame=frame(session.context, index, ms),
        tracking_status="TRACKED" if tracked else "UNRESOLVED",
        target_bbox=[0.05, 0.05, 0.95, 0.95] if tracked else None)
    landmarks = ObservationSet[Landmark](status="OBSERVED", reason=None, items=[
        Landmark(part="mouth", index=13, point=[0.5, 0.2], score=None),
        Landmark(part="hand_left", index=8, point=[0.5, 0.22 if near else 0.8], score=0.9)])
    objects = []
    if cup or medication:
        objects.append(DetectedObject(object_track_id="object-one", class_label="cup" if cup else "pill",
            score=0.9, bbox=[0.46, 0.18, 0.54, 0.25] if near else [0.46, 0.76, 0.54, 0.85], associated_parts=[]))
    return tracking, ObservationSet[DetectedObject](status="OBSERVED", items=objects, reason=None), landmarks


def detector(session):
    return EventDetector(session.context, ResourceRef(resource_id="detector", locator="config.json", media_type="application/json"),
        DetectionConfig(analysis_interval_ms=100, min_active_ms=100, release_ms=200, approach_lookback_ms=100))


def test_e02_does_not_require_objects_or_preparatory_event(session):
    d = detector(session)
    assert d.process(*sample(session, 0, 0)) == []
    update = d.process(*sample(session, 1, 100))[0]
    assert update.event.candidate.candidate_type == "E02"
    assert update.event.object_track_ids == []
    assert update.event.action_range.end_ms is None
    final = d.flush()[0]
    assert final.event.candidate.candidate_id == update.event.candidate.candidate_id
    assert final.event.candidate.candidate_revision > update.event.candidate.candidate_revision
    assert final.event.action_range.end_ms == 100


def test_repeated_action_new_identity_and_simultaneous_e03_preserved(session):
    d = detector(session)
    all_updates = []
    for index in range(12):
        all_updates += d.process(*sample(session, index, index * 100, near=index < 3 or index >= 8, cup=True))
    all_updates += d.flush()
    ended = [u.event for u in all_updates if u.event.action_range.end_ms is not None]
    assert [e.candidate.candidate_type for e in ended].count("E02") == 2
    assert [e.candidate.candidate_type for e in ended].count("E03") == 2
    assert len({e.candidate.candidate_id for e in ended}) == 4


def test_e01_and_source_motion_refs_are_consistent(session):
    d = detector(session)
    d.process(*sample(session, 0, 0, medication=True))
    updates = d.process(*sample(session, 1, 100, medication=True))
    assert {u.event.candidate.candidate_type for u in updates} == {"E01", "E02"}
    for update in updates:
        for sample_ in update.detection.samples:
            for motion in sample_.motions.items:
                assert all(ref[7:] in update.event.object_track_ids for ref in motion.entity_refs if ref.startswith("object:"))


def test_unresolved_tracking_closes_existing_without_creating_new(session):
    d = detector(session)
    d.process(*sample(session, 0, 0))
    d.process(*sample(session, 1, 100))
    ended = d.process(*sample(session, 2, 200, tracked=False))
    assert len(ended) == 1
    assert ended[0].event.action_range.end_ms == 100
    assert not d.process(*sample(session, 3, 300, tracked=False))


def test_duplicate_frame_rejected(session):
    d = detector(session)
    d.process(*sample(session, 0, 0))
    with pytest.raises(ValueError, match="increase"):
        d.process(*sample(session, 0, 0))


@pytest.mark.parametrize("cup,kind", [(True, "E03"), (False, "E02")])
def test_object_mouth_cues_survive_missing_hands(session, cup, kind):
    d = detector(session)
    updates = []
    for index in range(2):
        tracking, objects, landmarks = sample(session, index, index * 100, cup=cup, medication=not cup)
        landmarks.items = [point for point in landmarks.items if point.part == "mouth"]
        updates += d.process(tracking, objects, landmarks)
    assert kind in {u.event.candidate.candidate_type for u in updates}
