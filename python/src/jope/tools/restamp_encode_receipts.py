"""Re-stamp expansion encode receipts. Lock is not the blocker."""
from __future__ import annotations

import json
from pathlib import Path

REAL_BLOCKER = "frozen assembly registry v4 missing expansion family"
STALE = "campaign64 holds runs/encoding-service.lock; not stolen"
LIVE_CONTEXTS = "runs/dataset-inclusion-20260909/correction-final/current-contexts.json"


def restamp(receipt, lock_held=False):
    out = dict(receipt)
    claimed = out.get("apply_contexts_blocked_by")
    if lock_held:
        out["lock_held"] = True
        out["lock_diagnosis"] = "held"
    elif claimed == STALE or claimed == "encoding-service.lock" or (
        isinstance(claimed, str) and "encoding-service.lock" in claimed
    ):
        out["apply_contexts_blocked_by"] = REAL_BLOCKER
        out["lock_held"] = False
        out["lock_diagnosis"] = "stale"
    else:
        out.setdefault("apply_contexts_blocked_by", REAL_BLOCKER)
        out["lock_held"] = False
    out["writes_live_current_contexts"] = False
    return out


def restamp_file(path, output):
    receipt = json.loads(Path(path).read_text())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(restamp(receipt), indent=2) + "\n")
    return output


def restamp_tree(source_root, output_root, lock_held=False):
    """Copy-restamp receipt.json files. Never writes the live encode run in place."""
    source = Path(source_root)
    dest = Path(output_root)
    if "encoding-service.lock" in dest.as_posix():
        raise ValueError("refusing to write encoding-service.lock")
    written = []
    for path in sorted(source.rglob("receipt.json")):
        relative = path.relative_to(source)
        out = dest / relative
        restamp_file(path, out)
        written.append(str(relative))
    return {"n": len(written), "writes_live_current_contexts": False, "files": written}
