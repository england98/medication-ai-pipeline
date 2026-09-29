"""Timestamp-based candidate state machines. No feature is an event confirmation."""

from collections import deque
from dataclasses import dataclass, field

from medication_contracts import (
    CandidateKey,
    CandidateRegion,
    DetectionInfo,
    DetectionSample,
    EventCandidate,
    MotionFeature,
    ObservationSet,
    TimeRange,
)
from medication_contracts.geometry import around, distance, point_box_distance
from medication_contracts.records import new_id, utc_now

from .settings import DetectionConfig


def observed(items):
    return {"status": "OBSERVED", "items": items, "reason": None}


@dataclass
class CandidateUpdate:
    event: EventCandidate
    detection: DetectionInfo


@dataclass
class _Action:
    identifier: str
    kind: str
    channel: str
    start: int
    first_cue: int
    last_cue: int
    created: str
    revision: int = 0
    samples: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    objects: set = field(default_factory=set)


class EventDetector:
    def __init__(self, context, config_ref, config: DetectionConfig | None = None):
        self.context = context
        self.config_ref = config_ref
        self.config = config or DetectionConfig()
        self.actions = {}
        self.history = deque()
        self.previous_distances = {}
        self.last_ms = None

    def process(self, tracking, objects, landmarks) -> list[CandidateUpdate]:
        if tracking.context != self.context or tracking.frame.session_ms is None:
            raise ValueError("Detector requires matching session and timestamped tracking")
        frame = tracking.frame
        now = frame.session_ms
        if self.last_ms is not None and now <= self.last_ms:
            raise ValueError("Detection timestamps must increase strictly")
        self.last_ms = now
        if tracking.tracking_status != "TRACKED":
            self.previous_distances.clear()
            self.history.clear()
            return self.flush()
        bbox = tracking.target_bbox
        scale = max(1.0, (bbox[3] - bbox[1]) * frame.height)
        parts = {}
        for landmark in landmarks.items:
            parts.setdefault(landmark.part, []).append(landmark.point)
        hands = {name: points for name, points in parts.items() if name.startswith("hand_")}
        mouth_points = parts.get("mouth", [])
        mouth = ([sum(p[i] for p in mouth_points) / len(mouth_points) for i in (0, 1)]
                 if mouth_points else None)
        regions = [CandidateRegion(frame=frame, region_type="TARGET", bbox=bbox, object_track_id=None)]
        for name, points in hands.items():
            regions.append(CandidateRegion(frame=frame, region_type="HAND", bbox=around(points), object_track_id=None))
        if mouth_points:
            regions.append(CandidateRegion(frame=frame, region_type="MOUTH", bbox=around(mouth_points), object_track_id=None))
        associated = []
        cues = {}
        motions = []

        def motion(name, value, refs, begin=now, unit="target_height"):
            motions.append(MotionFeature(name=name, value=float(value), unit=unit,
                entity_refs=refs, range=TimeRange(start_ms=begin, end_ms=now)))

        for obj in objects.items:
            near = []
            for hand, points in hands.items():
                d = min(point_box_distance(p, obj.bbox, frame.width, frame.height, scale) for p in points)
                if d <= self.config.hand_object_distance:
                    near.append(hand)
            obj = obj.model_copy(update={"associated_parts": near})
            associated.append(obj)
            if obj.object_track_id:
                oid = obj.object_track_id
                label = obj.class_label.lower()
                oral_distance = (point_box_distance(mouth, obj.bbox, frame.width, frame.height, scale)
                                 if mouth is not None else None)
                oral_container = (label in self.config.container_labels and oral_distance is not None
                                  and oral_distance <= self.config.container_mouth_distance)
                oral_medication = (label in self.config.medication_labels + self.config.packaging_labels
                                   and oral_distance is not None and oral_distance <= self.config.mouth_enter)
                if near or oral_container or oral_medication:
                    regions.append(CandidateRegion(frame=frame, region_type="OBJECT", bbox=obj.bbox, object_track_id=oid))
                for hand in near:
                    d = min(point_box_distance(p, obj.bbox, frame.width, frame.height, scale) for p in hands[hand])
                    motion("hand_object_distance", d, [f"part:{hand}", f"object:{oid}"])
                if near and label in self.config.medication_labels + self.config.packaging_labels:
                    cues[("E01", oid)] = {oid}
                if oral_distance is not None:
                    motion("object_mouth_distance", oral_distance, ["part:mouth", f"object:{oid}"])
                if oral_container:
                    cues[("E03", oid)] = {oid}
                if oral_medication and not near:
                    # Object/packaging-to-mouth is also a cue when the hand is occluded.
                    existing = next((key for key, action in self.actions.items()
                                     if key[0] == "E02" and oid in action.objects), None)
                    cues[existing or ("E02", "object:" + oid)] = {oid}
        if mouth is None:
            self.previous_distances.clear()
        for hand, points in hands.items():
            if mouth is None:
                continue
            d = min(distance(p, mouth, frame.width, frame.height, scale) for p in points)
            previous = self.previous_distances.get(hand)
            speed = 0.0
            if previous and now > previous[0]:
                speed = (previous[1] - d) * 1000 / (now - previous[0])
                motion("hand_mouth_approach_speed", speed, [f"part:{hand}", "part:mouth"],
                       previous[0], "target_height/s")
            motion("hand_mouth_distance", d, [f"part:{hand}", "part:mouth"])
            self.previous_distances[hand] = (now, d)
            threshold = self.config.mouth_exit if ("E02", hand) in self.actions else self.config.mouth_enter
            if d <= threshold or (d <= self.config.approach_distance and speed >= self.config.approach_speed):
                object_ids = {o.object_track_id for o in associated
                              if hand in o.associated_parts and o.object_track_id}
                existing = next((key for key, action in self.actions.items()
                                 if key[0] == "E02" and object_ids & action.objects), None)
                existing = existing or next((key for key, ids in cues.items()
                                             if key[0] == "E02" and ids & object_ids), None)
                cues[existing or ("E02", hand)] = object_ids
        # Missing landmarks must not create a velocity across a period of occlusion.
        for hand in set(self.previous_distances) - set(hands):
            self.previous_distances.pop(hand)
        sample = DetectionSample(frame=frame,
            objects=objects.model_copy(update={"items": associated}), landmarks=landmarks,
            motions=ObservationSet[MotionFeature](**observed(motions)))
        while self.history and self.history[0][0].frame.session_ms < now - self.config.approach_lookback_ms:
            self.history.popleft()
        self.history.append((sample, regions))
        updates = []
        for key, object_ids in cues.items():
            if key not in self.actions:
                action = _Action(new_id("candidate"), key[0], key[1],
                    self.history[0][0].frame.session_ms, now, now, utc_now())
                # Preserve the observed approach, not a fabricated earlier frame.
                for old_sample, old_regions in list(self.history)[:-1]:
                    action.samples.append(old_sample)
                    action.regions.extend(old_regions)
                self.actions[key] = action
            action = self.actions[key]
            action.last_cue = now
            action.objects.update(object_ids)
        for key, action in list(self.actions.items()):
            action.samples.append(sample)
            action.regions.extend(regions)
            ended = key not in cues and now - action.last_cue >= self.config.release_ms
            if action.revision or action.last_cue - action.first_cue >= self.config.min_active_ms:
                updates.append(self._snapshot(action, ended, now))
            if ended:
                del self.actions[key]
        return updates

    def _snapshot(self, action, ended, end_ms):
        action.revision += 1
        key = CandidateKey(candidate_id=action.identifier, candidate_type=action.kind,
                           candidate_revision=action.revision)
        regions = [r for r in action.regions if r.object_track_id is None or r.object_track_id in action.objects]
        samples = []
        for sample in action.samples:
            motions = [m for m in sample.motions.items if all(
                not ref.startswith("object:") or ref[7:] in action.objects for ref in m.entity_refs)]
            samples.append(sample.model_copy(update={"motions": sample.motions.model_copy(update={"items": motions})}))
        event = EventCandidate(context=self.context, candidate=key, created_at=action.created,
            updated_at=utc_now(), action_range=TimeRange(start_ms=action.start,
            end_ms=end_ms if ended else None), object_track_ids=sorted(action.objects), regions=regions)
        detection = DetectionInfo(context=self.context, candidate=key,
                                  detector_config=self.config_ref, samples=samples)
        return CandidateUpdate(event, detection)

    def flush(self):
        updates = [self._snapshot(a, True, a.samples[-1].frame.session_ms)
                   for a in self.actions.values() if a.revision or
                   a.last_cue - a.first_cue >= self.config.min_active_ms]
        self.actions.clear()
        return updates
