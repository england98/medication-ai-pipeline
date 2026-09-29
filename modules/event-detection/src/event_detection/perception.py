"""Real YOLO and MediaPipe Tasks adapters with source-coordinate projection."""

from pathlib import Path

from medication_contracts import DetectedObject, Landmark, ObservationSet
from medication_contracts.geometry import distance, iou
from medication_contracts.records import new_id
from pydantic import BaseModel, ConfigDict, Field

from .settings import DetectionConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    object_weights: str
    person_weights: str | None = None
    face_landmarker: str
    hand_landmarker: str
    pose_landmarker: str
    device: str = "cpu"
    image_size: int = Field(default=640, ge=32)
    confidence: float = Field(default=0.25, gt=0, le=1)
    person_label: str = "person"
    class_map: dict[str, str] = Field(default_factory=dict)
    landmark_confidence: float = Field(default=0.5, gt=0, le=1)
    hand_wrist_distance: float = Field(default=0.15, gt=0)

    def assets(self):
        return {name: value for name, value in self.model_dump().items()
                if name in ("object_weights", "person_weights", "face_landmarker", "hand_landmarker", "pose_landmarker")
                and value}


class ObjectTracker:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.tracks = {}

    def update(self, objects, source_ms):
        self.tracks = {key: pair for key, pair in self.tracks.items()
                       if source_ms - pair[1] <= self.config.object_track_max_gap_ms}
        options = {}
        for index, obj in enumerate(objects):
            options[index] = [key for key, (old, _) in self.tracks.items()
                              if old.class_label == obj.class_label
                              and iou(old.bbox, obj.bbox) >= self.config.object_track_iou]
        counts = {}
        for keys in options.values():
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        output = []
        for index, obj in enumerate(objects):
            keys = options[index]
            # Never merge identities when two matches are plausible.
            key = keys[0] if len(keys) == 1 and counts[keys[0]] == 1 else new_id("object")
            item = obj.model_copy(update={"object_track_id": key})
            self.tracks[key] = item, source_ms
            output.append(item)
        return output


class YoloMediaPipe:
    def __init__(self, config: ModelConfig, rules: DetectionConfig):
        for name, path in config.assets().items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing {name}: {path}; supply a local model asset")
        import mediapipe as mp

        from ultralytics import YOLO

        self.mp = mp
        self.config = config
        self.objects_model = YOLO(config.object_weights)
        self.people_model = (YOLO(config.person_weights)
                             if config.person_weights and config.person_weights != config.object_weights
                             else self.objects_model)
        if config.person_label not in self.people_model.names.values():
            raise ValueError("Person detector has no configured person_label; supply person_weights")
        self.object_tracker = ObjectTracker(rules)
        self.tasks = []
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        try:
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base(model_asset_buffer=Path(config.face_landmarker).read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=config.landmark_confidence))
            self.tasks.append(self.face)
            self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=base(model_asset_buffer=Path(config.hand_landmarker).read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=config.landmark_confidence))
            self.tasks.append(self.hands)
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=base(model_asset_buffer=Path(config.pose_landmarker).read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=config.landmark_confidence))
            self.tasks.append(self.pose)
        except Exception:
            self.close()
            raise

    def _predict(self, model, image):
        import numpy as np
        result = model.predict(source=np.asarray(image.convert("RGB"))[:, :, ::-1].copy(),
            conf=self.config.confidence, imgsz=self.config.image_size,
            device=self.config.device, verbose=False)[0]
        objects = []
        for box in result.boxes:
            label = str(result.names[int(box.cls.item())])
            bbox = [max(0.0, min(1.0, float(v))) for v in box.xyxyn[0].tolist()]
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            objects.append(DetectedObject(object_track_id=None, class_label=label,
                score=float(box.conf.item()), bbox=bbox, associated_parts=[]))
        return objects

    def detect(self, image, frame):
        raw = self._predict(self.objects_model, image)
        people = raw if self.people_model is self.objects_model else self._predict(self.people_model, image)
        people = [o.bbox for o in people if o.class_label == self.config.person_label]
        objects = [o.model_copy(update={"class_label": self.config.class_map.get(o.class_label, o.class_label)})
                   for o in raw if o.class_label != self.config.person_label]
        tracked = self.object_tracker.update(objects, frame.source_ms)
        return people, ObservationSet[DetectedObject](status="OBSERVED", items=tracked, reason=None)

    def landmarks(self, image, tracking):
        if tracking.tracking_status != "TRACKED":
            return ObservationSet[Landmark](status="UNAVAILABLE", items=[], reason="TARGET_UNRESOLVED")
        import numpy as np
        frame = tracking.frame
        box = tracking.target_bbox
        # Integer crop edges are used in both the pixels and inverse projection.
        left, top = int(box[0] * frame.width), int(box[1] * frame.height)
        right, bottom = min(frame.width, int(box[2] * frame.width) + 1), min(frame.height, int(box[3] * frame.height) + 1)
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.asarray(crop).copy())
        timestamp = frame.source_ms
        pose = self.pose.detect_for_video(mp_image, timestamp)
        face = self.face.detect_for_video(mp_image, timestamp)
        hands = self.hands.detect_for_video(mp_image, timestamp)

        def project(point):
            return [max(0.0, min(1.0, (left + point.x * (right - left)) / frame.width)),
                    max(0.0, min(1.0, (top + point.y * (bottom - top)) / frame.height))]

        output = []
        wrists = []
        if pose.pose_landmarks:
            for index, point in enumerate(pose.pose_landmarks[0]):
                score = min(float(point.visibility or 0), float(point.presence or 0))
                if score < self.config.landmark_confidence or not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                    continue
                projected = project(point)
                output.append(Landmark(part="pose", index=index, point=projected, score=score))
                if index in (15, 16):
                    wrists.append(projected)
        if face.face_landmarks:
            for index in (13, 14, 61, 291):
                point = face.face_landmarks[0][index]
                if 0 <= point.x <= 1 and 0 <= point.y <= 1:
                    output.append(Landmark(part="mouth", index=index, point=project(point), score=None))
        for points, handedness in zip(hands.hand_landmarks, hands.handedness):
            wrist = project(points[0])
            scale = max(1, bottom - top)
            if not wrists or min(distance(wrist, p, frame.width, frame.height, scale) for p in wrists) > self.config.hand_wrist_distance:
                continue
            part = "hand_" + handedness[0].category_name.lower()
            for index, point in enumerate(points):
                if 0 <= point.x <= 1 and 0 <= point.y <= 1:
                    output.append(Landmark(part=part, index=index, point=project(point), score=float(handedness[0].score)))
        return ObservationSet[Landmark](status="OBSERVED", items=output, reason=None)

    def close(self):
        for task in self.tasks:
            task.close()
        self.tasks.clear()
