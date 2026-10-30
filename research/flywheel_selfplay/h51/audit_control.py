#!/usr/bin/env python3
"""Normalize H51 CONTROL evidence and verify seed-level adapter diversity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import adapter_weight_identity, atomic_json, sha256  # noqa: E402

ROOT = Path.home() / "jope-data/h51/runs"
BASE = Path.home() / "jope-data/h50/ckpt/round-785.pt"
SEEDS = (9, 10, 11)


def preserve_once(path: Path, suffix: str) -> None:
    backup = path.with_name(path.stem + suffix + path.suffix)
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())


def main() -> None:
    rows = []
    for seed in SEEDS:
        seed_dir = ROOT / "CONTROL" / f"seed{seed}"
        summary_path = seed_dir / "candidate-summary.json"
        selection_path = seed_dir / "selected-candidate.json"
        result_path = seed_dir / "train-result.json"
        if not all(path.is_file() for path in (summary_path, selection_path, result_path)):
            raise RuntimeError(f"incomplete H51-CONTROL seed{seed}")
        summary = json.loads(summary_path.read_text())
        selection = json.loads(selection_path.read_text())
        result = json.loads(result_path.read_text())
        checkpoint = seed_dir / selection["checkpoint"]
        adapter_sha, baseline_sha, update_l2 = adapter_weight_identity(checkpoint, BASE)
        if adapter_sha == baseline_sha or update_l2 <= 0:
            raise RuntimeError(f"H51-CONTROL seed{seed} selected r785 unchanged")
        preserve_once(summary_path, ".container-sha")
        preserve_once(selection_path, ".h48-label")
        preserve_once(result_path, ".h48-label")
        result["legacy_h48_arm_label"] = result.get("legacy_h48_arm_label", result.get("arm"))
        result["arm"] = "CONTROL"
        result["selected_adapter_sha256"] = adapter_sha
        result["selected_adapter_update_l2_vs_r785"] = update_l2
        atomic_json(result_path, result)
        selection["legacy_h48_arm_label"] = selection.get("legacy_h48_arm_label", selection.get("arm"))
        selection["arm"] = "CONTROL"
        selection["adapter_sha256"] = adapter_sha
        selection["adapter_update_l2_vs_r785"] = update_l2
        atomic_json(selection_path, selection)
        summary["adapter_file_sha256"] = sha256(checkpoint)
        summary["adapter_sha256"] = adapter_sha
        summary["r785_adapter_sha256"] = baseline_sha
        summary["adapter_update_l2_vs_r785"] = update_l2
        summary["adapter_param_count"] = 29_211
        summary["arm"] = "CONTROL"
        summary["legacy_h48_arm_label"] = result["legacy_h48_arm_label"]
        atomic_json(summary_path, summary)
        rows.append(summary)

    shas = [row["adapter_sha256"] for row in rows]
    if len(set(shas)) != len(SEEDS):
        raise RuntimeError(f"CONTROL seeds are not distinct adapters: {shas}")
    utilities = [row["inner_cv_mean_gain"] for row in rows]
    identical = len(set(utilities)) == 1
    devices = sorted({row["device"] for row in rows})
    device = devices[0] if len(devices) == 1 else "mixed; keep device strata separate"
    audit = {
        "schema": "jope.h51.seed-distinctness/v1",
        "arm": "CONTROL",
        "seeds": list(SEEDS),
        "adapter_sha256": shas,
        "distinct_adapter_shas": True,
        "adapter_update_l2_vs_r785": [row["adapter_update_l2_vs_r785"] for row in rows],
        "inner_cv_mean_gains": utilities,
        "identical_inner_cv_utilities": identical,
        "utility_interpretation": "step-like utility; not seed agreement" if identical else "inner utilities differ",
        "terminal_dl_val": "not read unless an admitted candidate is separately selected",
        "device": device,
    }
    atomic_json(ROOT / "control-seed-distinctness.json", audit)
    print(json.dumps(audit, allow_nan=False))


if __name__ == "__main__":
    main()
