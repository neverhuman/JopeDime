"""One in-output launch path for both smoke and scientific execution."""

import hashlib
import json
from pathlib import Path
import subprocess


def read_launch(output, manifest, smoke):
    output = Path(output).resolve()
    if {p.name for p in output.iterdir()} != {"launch.json"}:
        raise RuntimeError("fresh output containing only launch.json required")
    path = output / "launch.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("regular in-output launch.json required")
    launch = json.loads(path.read_text())
    expected = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if launch.get("manifest_sha256") != expected:
        raise RuntimeError("launch manifest mismatch")
    if launch.get("mode") != ("smoke" if smoke else "run"):
        raise RuntimeError("launch mode mismatch")
    if not smoke:
        if not launch.get("merged_source_verified"):
            raise RuntimeError("merged source required")
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(manifest).parent, text=True
        ).strip()
        if launch.get("source_commit") != head:
            raise RuntimeError("launch source HEAD mismatch")
        if launch.get("review_disposition") not in ("PASS", "TIMEOUT_SELF_REVIEW_PASS"):
            raise RuntimeError("review disposition required")
    return launch
