"""Download the YOLOX ONNX weights from the official release, checking a pinned SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
MODELS = {
    # name: (file, sha256). YOLOX is Apache-2.0; both models take a 416x416 input.
    "nano": ("yolox_nano.onnx", "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "tiny": ("yolox_tiny.onnx", "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest_dir: str | Path = "models") -> Path:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from: {', '.join(MODELS)}")
    filename, digest = MODELS[name]
    dest = Path(dest_dir) / filename
    if dest.exists() and sha256(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(f"{RELEASE}/{filename}", timeout=60) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {filename}: expected {digest}, got {got}")
    tmp.replace(dest)
    return dest
