"""Download small official baseline models for adapter smoke tests (not medication-trained YOLO)."""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ASSETS = {
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


EXPECTED_SHA256 = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in ASSETS.items():
        destination = args.output / name
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            os.replace(temporary, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Model hash mismatch: {destination}; expected pinned official asset")
        manifest[name] = {"url": url, "sha256": digest, "bytes": destination.stat().st_size}
        print(f"{name}: {destination.stat().st_size} bytes, sha256={digest}", flush=True)
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
