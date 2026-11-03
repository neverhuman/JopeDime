from pathlib import Path
import os
import sys
import json
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("H285_CUSTODY_ROOT", str(HERE.parent)))
CLOSURE = ROOT / "code-closure"
PINS = json.loads((HERE / "dependency-pins.json").read_text())
for name, digest in PINS["files"].items():
    assert hashlib.sha256((CLOSURE / name).read_bytes()).hexdigest() == digest, name
sys.path.insert(0, str(CLOSURE))
import topk  # noqa: E402
import h265_core as core  # noqa: E402
import cluster_protocol as clusters  # noqa: E402

__all__ = ["ROOT", "CLOSURE", "PINS", "topk", "core", "clusters"]
