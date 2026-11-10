#!/usr/bin/env python3
"""H417 APEX cohort manifest access (stdlib only; no GP stack, no table is opened).

Manifest: $H317_DATA/apex/APEX_COHORTS.json (default data root /home/ubuntu/jope-data/h335-cl), built create-once by
cohorts_h417.sh and pinned here by sha256. Split rule: d = sha256("APEX-split-v1" NUL family_key);
k = int(d[0:8], 16) mod 5 -> 0 CONFIRM, 1 DEV, 2 FIT-SEL, 3-4 FIT-ICL.

Tiers (fixed from table shape only, before any outcome): core = rows >= 800 and 3 <= width <= 400, the size regime of
DL-Train and dev91. Smaller tables stay in the manifest for training use; H417 does not deploy them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

MANIFEST_SHA256 = "11a57d80a67fcefbbdc83986062e50608eed78026cec7508ecbc5040999bc4eb"
COHORTS = ("CONFIRM", "DEV", "FIT-SEL", "FIT-ICL")
CORE_MIN_ROWS, CORE_MIN_WIDTH, CORE_MAX_WIDTH = 800, 3, 400

_ROWS: Dict[str, dict] = {}


def data_root() -> Path:
    return Path(os.environ.get("H317_DATA", "/home/ubuntu/jope-data/h335-cl"))


def manifest_path() -> Path:
    return data_root() / "apex" / "APEX_COHORTS.json"


def rows() -> Dict[str, dict]:
    """cid -> manifest row (context_sha256, dataset_id, family_key, family_id, train_only_jtt1_sha256, rows, width,
    cohort, ...)."""
    if not _ROWS:
        raw = manifest_path().read_bytes()
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise RuntimeError("APEX_COHORTS.json differs from the pinned manifest")
        doc = json.loads(raw)
        if doc["schema"] != "jope.h417.apex_cohorts/v1" or doc["n"] != 1153:
            raise RuntimeError("not the H417 APEX cohort manifest")
        for r in doc["contexts"]:
            if r["cohort"] not in COHORTS:
                raise RuntimeError(f"unknown cohort {r['cohort']}")
            _ROWS[r["context_sha256"]] = r
    return _ROWS


def is_core(r: dict) -> bool:
    return r["rows"] >= CORE_MIN_ROWS and CORE_MIN_WIDTH <= r["width"] <= CORE_MAX_WIDTH


def cohort(name: str, tier: str = "core") -> List[str]:
    """Sorted context ids of one cohort. tier: core | all."""
    if name not in COHORTS or tier not in ("core", "all"):
        raise SystemExit(f"unknown cohort / tier {name} / {tier}")
    return sorted(c for c, r in rows().items() if r["cohort"] == name and (tier == "all" or is_core(r)))


def panel(name: str) -> List[str]:
    """DEV / FIT-SEL: the core tier of that cohort. BASE: both (the H417 baseline deploy; read out per cohort). SMK2: the 2
    smallest core FIT-SEL tables (smoke; never read out). CONFIRM is refused: H417 neither deploys nor reads it."""
    if name in ("DEV", "FIT-SEL"):
        return cohort(name)
    if name == "BASE":
        return sorted(cohort("DEV") + cohort("FIT-SEL"))
    if name == "SMK2":
        r = rows()
        return sorted(sorted(cohort("FIT-SEL"), key=lambda c: (r[c]["rows"] * (r[c]["width"] + 1), c))[:2])
    raise SystemExit(f"panel {name} is not available in H417")


def family_of(cids: Sequence[str]) -> Dict[str, str]:
    r = rows()
    return {c: r[c]["family_id"] for c in cids}


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "panel":
        print("\n".join(panel(sys.argv[2])))
        raise SystemExit(0)
    print(json.dumps({"manifest_sha256": MANIFEST_SHA256,
                      "cohorts": {c: {"all": len(cohort(c, "all")), "core": len(cohort(c))} for c in COHORTS},
                      "panels": {p: len(panel(p)) for p in ("DEV", "FIT-SEL", "BASE", "SMK2")}}))
