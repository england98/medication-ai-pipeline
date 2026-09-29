"""Download and verify the pinned Transformers checkpoint used by the pipeline."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def digest_file(path, algorithm, *, git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        Path(__file__).resolve().parents[1] / "models" / "Qwen3-VL-2B-Instruct"))
    args = parser.parse_args()
    output = args.output.resolve()
    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise RuntimeError("Repository revision does not match the pinned commit")
    files = [item for item in info.siblings if item.rfilename.endswith(
        (".json", ".safetensors", ".txt", ".md"))]
    print(f"Downloading {REPO_ID}@{REVISION} to {output}", flush=True)
    print(f"Expected total: {sum(item.size or 0 for item in files):,} bytes", flush=True)
    snapshot_download(REPO_ID, revision=REVISION, local_dir=output,
                      allow_patterns=[item.rfilename for item in files], token=False,
                      max_workers=2)
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "files": {}}
    for item in files:
        path = output / item.rfilename
        if path.stat().st_size != item.size:
            raise RuntimeError(f"Size mismatch: {path}")
        sha256 = digest_file(path, "sha256")
        if item.lfs:
            if sha256 != item.lfs.sha256:
                raise RuntimeError(f"SHA256 mismatch: {path}")
        elif digest_file(path, "sha1", git_blob=True) != item.blob_id:
            raise RuntimeError(f"Git blob hash mismatch: {path}")
        manifest["files"][item.rfilename] = {"bytes": item.size, "sha256": sha256}
        print(f"Verified {item.rfilename}: {item.size:,} bytes", flush=True)
    (output / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint ready: {output}", flush=True)


if __name__ == "__main__":
    main()
