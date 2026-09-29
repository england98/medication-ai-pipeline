"""Run real YOLO/MediaPipe on a local video without authentication/completion claims."""
import argparse
import json
import os
import sys
from pathlib import Path

from authentication_tracking import TargetTracker
from event_detection import EventDetector
from event_detection.perception import YoloMediaPipe
from medication_contracts import ResourceRef, SessionKey
from medication_contracts.records import append_json, new_id, write_json
from medication_pipeline.artifacts import create_test_directory
from medication_pipeline.config import load_config
from session_video.video import read_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pipeline.example.json'))
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--output', type=Path, default=Path('outputs/video-yolo-mediapipe-detection'),
                        help='Result path prefix; appends _YYYYMMDD_HHMMSS in KST')
    args = parser.parse_args()
    config = load_config(args.config)
    args.output = create_test_directory(args.output.parent, args.output.name)
    print(f"Test output: {args.output}", file=sys.stderr, flush=True)
    (args.output / 'ultralytics').mkdir(exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(args.output.resolve() / 'ultralytics'))
    config_path = args.output.resolve() / 'detector-config.json'
    write_json(config_path, {'models': config.models.model_dump(), 'rules': config.detection.model_dump()})
    context = SessionKey(session_id=new_id('diagnostic'), user_id='unverified-video-subject',
        scheduled_occurrence_id='no-scheduled-occurrence', stream_id=new_id('stream'), target_track_id=new_id('person'))
    detector = EventDetector(context, ResourceRef(resource_id='diagnostic-config', locator=str(config_path), media_type='application/json'), config.detection)
    tracker, origin, last = None, None, None
    count = 0
    counts = {'TRACKED': 0, 'UNRESOLVED': 0, 'hand': 0, 'mouth': 0, 'pose': 0, 'objects': 0}
    events = []
    perception = YoloMediaPipe(config.models, config.detection)
    source = read_video(args.video, context.stream_id)
    try:
        for item in source:
            if last is not None and item.ref.source_ms - last < config.detection.analysis_interval_ms:
                continue
            people, objects = perception.detect(item.image, item.ref)
            last = item.ref.source_ms
            if tracker is None:
                if len(people) != 1:
                    continue
                tracker = TargetTracker(people[0], config.tracking)
                origin = item.ref.source_ms
            ref = item.ref.model_copy(update={'session_ms': item.ref.source_ms - origin})
            tracking = tracker.update(context, ref, people)
            landmarks = perception.landmarks(item.image, tracking)
            updates = detector.process(tracking, objects, landmarks)
            events.extend(u.event for u in updates if u.event.action_range.end_ms is not None)
            counts[tracking.tracking_status] += 1
            counts['objects'] += len(objects.items)
            for part in ('hand', 'mouth', 'pose'):
                counts[part] += int(any(landmark.part.startswith(part) for landmark in landmarks.items))
            append_json(args.output / 'observations.jsonl', {'tracking': tracking.model_dump(),
                'objects': objects.model_dump(), 'landmarks': landmarks.model_dump()})
            count += 1
            if count >= args.max_frames:
                break
        events.extend(u.event for u in detector.flush())
        summary = {'run_dir': str(args.output), 'source': str(args.video.resolve()), 'analyzed_frames': count,
            'counts': counts, 'candidates': {kind: sum(e.candidate.candidate_type == kind for e in events) for kind in ('E01', 'E02', 'E03')},
            'authentication': 'NOT_TESTED', 'vlm': 'NOT_RUN', 'quality_evaluation': 'NOT_PERFORMED'}
        write_json(args.output / 'summary.json', summary)
        write_json(args.output / 'candidates.json', [e.model_dump(mode='json') for e in events])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not count:
            raise RuntimeError('No unambiguous person for adapter smoke testing')
    finally:
        perception.close()
        source.close()


if __name__ == '__main__':
    main()
