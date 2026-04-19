"""Freeze-check: the benchmark manifest's SHA-256 matches the files on disk."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_matches_files() -> None:
    manifest_path = Path("eval/MANIFEST.json")
    assert manifest_path.exists(), "run eval/build_benchmark.py first"
    manifest = json.loads(manifest_path.read_text())
    for name, meta in manifest["files"].items():
        path = Path("eval") / name
        assert path.exists(), f"missing {path}"
        assert _sha(path) == meta["sha256"], f"checksum mismatch for {path}"


if __name__ == "__main__":
    test_manifest_matches_files()
    print("freeze ok")
