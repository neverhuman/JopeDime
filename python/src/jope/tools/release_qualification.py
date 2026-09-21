"""Transient-unit cleanup; never releases another task's control or kernel locks."""

import argparse
import time
from pathlib import Path

from qualification_slots import release_stale
from resume_expired_cycle import atomic, read


def release(root, output):
    control = root / "runs/qualification-control.json"
    if control.exists():
        request = read(control)
        owned = {str(path.relative_to(root)) for path in (output, output / "gpu-handoff")}
        if request.get("active") and request.get("output") in owned:
            atomic(
                control,
                dict(request, active=False, released_at=time.time(), release_reason="owning unit exited"),
            )
    manifest = output / "request.json"
    if manifest.exists():
        release_stale(root, manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    release(a.root.resolve(), a.output.resolve())
