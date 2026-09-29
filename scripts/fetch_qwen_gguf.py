"""Prepare the pinned official Q4_K_M model, F16 vision encoder and Windows runtime."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
REVISION = "52d6c8ffea26cc873ac5ad116f8631268d7eb503"
ASSETS = {
    "Qwen3VL-2B-Instruct-Q4_K_M.gguf": (1107409952, "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"),
    "mmproj-Qwen3VL-2B-Instruct-F16.gguf": (819394848, "c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5"),
}
RUNTIME = {
    "llama-b10991-bin-win-cuda-12.4-x64.zip": (254197034, "de86232a73a0fd5b96454309da2993f2dc6bf5e16ecc506192f551d49d833cdf"),
    "cudart-llama-bin-win-cuda-12.4-x64.zip": (391443627, "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain(path, source, url, size, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        if source and source.is_file():
            print(f"Copying local asset: {source.name}", flush=True)
            shutil.copyfile(source, temporary)
        else:
            print(f"Downloading: {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, 8 * 1024 * 1024)
        if temporary.stat().st_size != size or sha256(temporary) != digest:
            raise ValueError(f"Size/hash mismatch: {temporary}")
        temporary.replace(path)
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Size/hash mismatch: {path}")
    print(f"Verified {path.name}: {size:,} bytes", flush=True)
    return {"url": url, "bytes": size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-assets", type=Path, help="Optional medication-local-assets directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "models/Qwen3-VL-2B-Instruct-GGUF"
    manifest = {"repo_id": REPO, "revision": REVISION, "quantization": "Q4_K_M + F16 mmproj", "files": {}}
    for name, (size, digest) in ASSETS.items():
        source = args.reuse_assets / "models/qwen3-vl-2b" / name if args.reuse_assets else None
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        manifest["files"][name] = obtain(output / name, source, url, size, digest)
    (output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    runtime = root / "runtime/llama.cpp/b10991"
    files = {}
    for name, (size, digest) in RUNTIME.items():
        source = args.reuse_assets / "runtime/b10991" / name if args.reuse_assets else None
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/b10991/{name}"
        obtain(runtime / name, source, url, size, digest)
        with zipfile.ZipFile(runtime / name) as archive:
            for member in archive.infolist():
                target = (runtime / member.filename).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise ValueError("Unsafe archive path")
                archive.extract(member, runtime)
                if target.is_file():
                    files[member.filename] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    (runtime / "runtime-manifest.json").write_text(json.dumps(
        {"release": "b10991", "archives": {name: {"bytes": s, "sha256": h} for name, (s, h) in RUNTIME.items()}, "files": files},
        indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {output}\nRuntime: {runtime}", flush=True)


if __name__ == "__main__":
    main()
