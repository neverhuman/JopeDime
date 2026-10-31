#!/usr/bin/env python3
"""H86 W3 outcome-row status census and field-allowlisted status-mask export.

Classifies every admitted H50 DL-Train pool row by a fixed signature rule, measures W3 target
coverage, and exports only a per-row status code aligned to pool order. Never a W3 study run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.flywheel_selfplay.h75_w4_cached_baseline_shard import run_shard as h75_w4  # noqa: E402

H65_FOLDS = (
    REPO_ROOT
    / "research/flywheel_selfplay/h65_w3_continuous_top_resolved/fold_manifest.json"
)
H65_FOLDS_SHA = "cabcfc6f182ef3aeb77a10890d81de0c655f60b5bb3db2a56b70c44a58567067"

# Status codes, in precedence order; only PASS and FAIL are measured outcomes.
PASS, FAIL, SENTINEL, NONFINITE, ANOMALOUS = 0, 1, 2, 3, 4
NAMES = ("measured_pass", "measured_fail", "sentinel_shaped", "nonfinite", "anomalous")
FAIL_OFFSET = 100.0  # legacy encoding: a gate failure's fitness carries a +100 offset
TOP = 16
FLOOR = 0.05
CLIP = 6.0
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def classify(fitness: np.ndarray, logratio: np.ndarray, gate: np.ndarray) -> np.ndarray:
    status = np.where(gate, PASS, FAIL).astype(np.uint8)
    finite = np.isfinite(fitness) & np.isfinite(logratio)
    # A passing row cannot carry the failure offset; a failing row must carry it.
    status[gate & (fitness >= FAIL_OFFSET)] = ANOMALOUS
    status[~gate & (fitness < FAIL_OFFSET / 2)] = ANOMALOUS
    status[(fitness == FAIL_OFFSET) & (logratio == 0.0)] = SENTINEL
    status[~finite] = NONFINITE
    return status


def top_resolved(passing_fitness: np.ndarray) -> dict:
    """W3 arm c target construction on one context's measured passers (lower fitness is better)."""
    n = len(passing_fitness)
    if n == 0:
        return {"path": "no_passer", "passers": 0}
    ordered = np.sort(passing_fitness)
    reference = ordered[min(TOP, n) - 1]
    spread = float(reference - ordered[0])
    regret = (ordered - ordered[0]) / max(spread, FLOOR)
    return {
        "path": "top16" if n >= TOP else "last_passing_order_statistic",
        "passers": n,
        "floor_binds": spread < FLOOR,
        "clipped": int(np.sum(regret > CLIP)),
    }


def quantiles(values: np.ndarray) -> dict | None:
    if len(values) == 0:
        return None
    return {
        f"q{q:g}": float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))
    }


def order_sha(program_ids: np.ndarray) -> str:
    return hashlib.sha256(b"".join(program_ids.tolist())).hexdigest()


def run(input_root: Path, folds_path: Path, out_dir: Path, s3_prefix: str) -> dict:
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    cpu_started = time.process_time()
    expected = {
        input_root / "source_manifest.json": h75_w4.PACKET_FILE_SHA,
        input_root / "selected.json": h75_w4.SELECTED_SHA,
        folds_path: H65_FOLDS_SHA,
    }
    for path, sha in expected.items():
        if h75_w4.sha256(path) != sha:
            raise ValueError(f"pinned input hash mismatch: {path}")
    packet = json.loads((input_root / "source_manifest.json").read_text())
    if (
        packet["source_manifest_sha256"] != h75_w4.PACKET_ID
        or packet["outer_rows_delivered"] != 0
    ):
        raise ValueError("H50 packet identity or closure mismatch")
    folds = json.loads(folds_path.read_text())
    if (
        folds["training_role"] != "DL-Train"
        or folds["selected_file_sha256"] != h75_w4.SELECTED_SHA
    ):
        raise ValueError(
            "H65 fold manifest is not bound to the admitted training selection"
        )
    fold_of = {c: f["fold_index"] for f in folds["folds"] for c in f["held_contexts"]}
    rows = json.loads((input_root / "selected.json").read_text())["train"]
    if (
        len(rows) != 1219
        or {r["dl_split"] for r in rows} != {"DL-Train"}
        or set(fold_of) != {r["context_sha256"] for r in rows}
    ):
        raise ValueError(
            "selection is not the 1,219-context DL-Train roster of the fold manifest"
        )

    totals, by_fold = Counter(), defaultdict(Counter)
    paths, floor_binds, clipped, passers_total = Counter(), 0, 0, 0
    contexts_with = Counter()
    residual, fail_logratio, fail_excess = [], [], []
    cids, offsets, masks, pool_shas, order_shas = [], [0], [], [], []
    for row in sorted(rows, key=lambda r: r["context_sha256"]):
        context = row["context_sha256"]
        pool_path = input_root / "pools" / f"{context}.npz"
        if h75_w4.sha256(pool_path) != row["pool_sha256"]:
            raise ValueError(f"training pool hash mismatch for {context}")
        with np.load(pool_path, allow_pickle=False) as pool:
            ids = pool["program_sha256"]
            fitness = np.asarray(pool["fitness"], dtype=np.float64)
            logratio = np.asarray(pool["logratio"], dtype=np.float64)
            gate = np.asarray(pool["gate"], dtype=np.bool_)
        if (
            not len(ids)
            == len(fitness)
            == len(logratio)
            == len(gate)
            == row["n_candidates"]
        ):
            raise ValueError(f"pool arrays do not align for {context}")
        if int(gate.sum()) != row["n_gate_pass"]:
            raise ValueError(
                f"gate count differs from the admitted selection for {context}"
            )
        status = classify(fitness, logratio, gate)
        counts = Counter(
            {NAMES[k]: int(v) for k, v in zip(*np.unique(status, return_counts=True))}
        )
        totals.update(counts)
        by_fold[fold_of[context]].update(counts)
        for name in NAMES[2:]:
            contexts_with[name] += counts[name] > 0
        measured_pass = status == PASS
        target = top_resolved(fitness[measured_pass])
        paths[target["path"]] += 1
        passers_total += target["passers"]
        floor_binds += target.get("floor_binds", False)
        clipped += target.get("clipped", 0)
        residual.append(fitness[measured_pass] - 0.9 * logratio[measured_pass])
        measured_fail = status == FAIL
        fail_logratio.append(logratio[measured_fail])
        fail_excess.append(fitness[measured_fail] - FAIL_OFFSET)
        cids.append(context)
        offsets.append(offsets[-1] + len(status))
        masks.append(status)
        pool_shas.append(row["pool_sha256"])
        order_shas.append(order_sha(ids))

    out_dir.mkdir(parents=True, exist_ok=False)
    mask_path = out_dir / "status_mask.npz"
    with mask_path.open("xb") as handle:
        np.savez(
            handle,
            cids=np.asarray(cids),
            offsets=np.asarray(offsets, dtype=np.int64),
            status=np.concatenate(masks),
            pool_sha256=np.asarray(pool_shas),
            program_order_sha256=np.asarray(order_shas),
        )
    rows_total = sum(totals.values())
    has_target = paths["top16"] + paths["last_passing_order_statistic"]
    metrics = {
        "schema": "jopedime.h86-w3-outcome-status-census-metrics.v1",
        "status": "CENSUS_ONLY",
        "contexts": len(cids),
        "families": len({r["family_id"] for r in rows}),
        "rows": rows_total,
        "status_counts": {name: totals[name] for name in NAMES},
        "status_fractions": {name: totals[name] / rows_total for name in NAMES},
        "contexts_with_any": {name: contexts_with[name] for name in NAMES[2:]},
        "status_counts_by_fold": {
            str(k): {name: by_fold[k][name] for name in NAMES} for k in sorted(by_fold)
        },
        "w3_arm_c_top_resolved": {
            "rule": f"regret=(F-F(1))/max(F({TOP})-F(1),{FLOOR}), clip [0,{CLIP:g}]; fewer than {TOP} passers uses the last passing order statistic",
            "context_paths": {
                p: paths[p]
                for p in ("top16", "last_passing_order_statistic", "no_passer")
            },
            "floor_binds_contexts": floor_binds,
            "floor_binds_fraction_of_contexts_with_target": floor_binds / has_target
            if has_target
            else None,
            "clipped_passing_rows": clipped,
            "clipped_fraction_of_passing_rows": clipped / passers_total
            if passers_total
            else None,
        },
        "w3_arm_d_decomposed": {
            "measured_logratio_rows": totals["measured_pass"] + totals["measured_fail"],
            "measured_logratio_fraction": (
                totals["measured_pass"] + totals["measured_fail"]
            )
            / rows_total,
            "passing_residual_F_minus_0p9_logratio": quantiles(
                np.concatenate(residual)
            ),
            "failing_logratio": quantiles(np.concatenate(fail_logratio)),
            "failing_fitness_minus_100": quantiles(np.concatenate(fail_excess)),
        },
        "scope": "Full-pool census of admitted DL-Train arrays. Training slates are 192-candidate subsets, so slate-level target statistics differ. Status is inferred from value signatures; the packet records no per-row completion status.",
        "w3_study_run": False,
        "scientific_verdict": None,
    }
    prefix = s3_prefix.rstrip("/")
    manifest = {
        "schema": "jopedime.h86-w3-outcome-status-census-run.v1",
        "run_id": out_dir.name,
        # JOPE_SOURCE_SHA (as in deploy/ci/lambda.sh) lets a `git archive` export record its commit.
        "source_commit": os.environ.get("JOPE_SOURCE_SHA")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip(),
        "source_file_sha256": h75_w4.sha256(Path(__file__)),
        "s3_output_prefix": prefix,
        "s3_outputs": [
            f"{prefix}/{name}"
            for name in ("metrics.json", "manifest.json", "status_mask.npz")
        ],
        "status_mask_sha256": h75_w4.sha256(mask_path),
        "status_codes": dict(enumerate(NAMES)),
        "status_rule": {
            "precedence": "nonfinite > sentinel_shaped > anomalous > gate",
            "sentinel_shaped": "fitness == 100 and logratio == 0 exactly",
            "anomalous": "gate pass with fitness >= 100, or gate fail with fitness < 50",
            "measured": "codes 0 and 1 only",
            "provenance": "inferred from value signatures; not recorded completion status",
        },
        "export_fields": [
            "cids",
            "offsets",
            "status",
            "pool_sha256",
            "program_order_sha256",
        ],
        "export_excludes": "fitness, logratio, gate, program features and every DL-Val, outer, sealed or holdout field",
        "input_packet_id": h75_w4.PACKET_ID,
        "inputs": [
            {
                "local_path": str(input_root / "source_manifest.json"),
                "s3_key": None,
                "sha256": h75_w4.PACKET_FILE_SHA,
            },
            {
                "local_path": str(input_root / "selected.json"),
                "s3_key": None,
                "sha256": h75_w4.SELECTED_SHA,
            },
            {"local_path": str(folds_path), "s3_key": None, "sha256": H65_FOLDS_SHA},
            {
                "pools": len(pool_shas),
                "pool_sha256_list_sha256": hashlib.sha256(
                    "".join(pool_shas).encode()
                ).hexdigest(),
            },
        ],
        "host": socket.gethostname(),
        "cpu_seconds": time.process_time() - cpu_started,
        "wall_seconds": time.monotonic() - clock,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "Elo": None,
            "val_reg_inner": None,
            "U4": None,
            "descriptive_metrics_file": "metrics.json",
        },
        "verdict": "NO_STUDY_CLAIM; W3_INPUT_ADMISSION_CENSUS",
    }
    (out_dir / "metrics.json").write_bytes(h75_w4.json_bytes(metrics))
    (out_dir / "manifest.json").write_bytes(h75_w4.json_bytes(manifest))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path.home() / "jope-data/h50/input"
    )
    parser.add_argument("--folds", type=Path, default=H65_FOLDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--s3-prefix", required=True)
    args = parser.parse_args()
    metrics = run(args.input, args.folds, args.out, args.s3_prefix)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ("status_counts", "contexts_with_any", "w3_arm_c_top_resolved")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
