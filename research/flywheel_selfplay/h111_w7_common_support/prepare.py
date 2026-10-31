"""Metadata-only W7 matching preparation; no outcome ledger or execution mode."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matching

H104_SHA256 = {
    "runner.py": "c5c46036c3bf45f12a9912fe02f9b3c6ba8d0cb23f9ca27ba168a1704d8d4217",
    "config.json": "828838fe32777ee39d813029ac981d66748f04c7ea6f18073a1b939ab2012ec7",
    "fold_manifest.json": "129002bcd6cfc4b2f3a0b9e0562967cd4510dcbf88364c2d11d0bbe8801c2c73",
}


def build_preparation():
    root = Path(__file__).resolve().parent.parent / "h104_w7_acquisition_replay_prep"
    for name, sha in H104_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("H104 source/metadata pin changed: " + name)
    spec = importlib.util.spec_from_file_location("h104_pinned", root / "runner.py")
    h104 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h104)
    cfg = h104.load_pinned(root / "config.json", H104_SHA256["config.json"])
    folds = h104.load_pinned(
        root / "fold_manifest.json", H104_SHA256["fold_manifest.json"]
    )
    plan = h104.build_plan(cfg, folds)
    out = matching.prepare_cells(plan)
    out["h104_plan_file_sha256"] = hashlib.sha256(h104.encoded(plan)).hexdigest()
    out["source_pins"] = H104_SHA256
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", required=True, action="store_true")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = build_preparation()
    with args.output.open("xb") as f:
        f.write((json.dumps(out, sort_keys=True, indent=2) + "\n").encode())
    print(
        json.dumps(
            dict(
                cells=len(out["cells"]),
                pending_slots=len(out["common_learner_receipt_panel"]),
                execution_admitted=False,
            )
        )
    )


if __name__ == "__main__":
    main()
