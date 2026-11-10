"""Load the hash-pinned H268 scientific closure preserved with the prereg S3 archive."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("H274_CUSTODY_ROOT", str(HERE.parent)))
CLOSURE = ROOT / "code-closure"
PINS = json.loads((HERE / "dependency-pins.json").read_text())
for name, digest in PINS["files"].items():
    assert hashlib.sha256((CLOSURE / name).read_bytes()).hexdigest() == digest, name
sys.path.insert(0, str(CLOSURE))

spec = importlib.util.spec_from_file_location("h268_frozen", CLOSURE / "h268_frozen.py")
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)

h = prev.h
common = prev.common
