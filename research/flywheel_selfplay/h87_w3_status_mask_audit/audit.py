#!/usr/bin/env python3
"""Independently check the H86 W3 status-mask export against admitted DL-Train pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKET_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
PACKET_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
FOLDS_SHA = "cabcfc6f182ef3aeb77a10890d81de0c655f60b5bb3db2a56b70c44a58567067"
H86_RUN = "luna-tel-h86-20261031T0709Z-01"
H86_PREFIX = f"s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W3/{H86_RUN}"
H86_ARTIFACTS = {
    "manifest": {
        "key": f"{H86_PREFIX}/manifest.json",
        "version": "3d8Ty0jjfnXuZV3bkYiNIGx7Vk3XDJ9G",
        "sha256": "d9e3ba44209b6e1425313fff282055b99d88c192638f91a05f0762fb93a74f84",
    },
    "metrics": {
        "key": f"{H86_PREFIX}/metrics.json",
        "version": "GRzC6Wn0M_3SmMdIQ882Yub1aUGpewIU",
        "sha256": "500a9876d4a682e8e0e2de9f56670d7ac57a803d2de0ff5de1bef2756663168b",
    },
    "mask": {
        "key": f"{H86_PREFIX}/status_mask.npz",
        "version": "wmp4LaFHHHLpuRQLpV3_byxY13smxE2z",
        "sha256": "37a9271f92c1ab8f2e554c97e8a9cb414e50cb1362e263a41b1c93164868f4f6",
    },
}
FIELDS = {"cids", "offsets", "status", "pool_sha256", "program_order_sha256"}
NAMES = ("measured_pass", "measured_fail", "sentinel_shaped", "nonfinite", "anomalous")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def classify(fitness: np.ndarray, logratio: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Reimplement H86's declared value-signature rule for parity checking."""
    status = np.where(gate, 0, 1).astype(np.uint8)
    status[gate & (fitness >= 100.0)] = 4
    status[(~gate) & (fitness < 50.0)] = 4
    status[(fitness == 100.0) & (logratio == 0.0)] = 2
    status[~(np.isfinite(fitness) & np.isfinite(logratio))] = 3
    return status


def order_sha(ids: np.ndarray) -> str:
    return hashlib.sha256(b"".join(ids.tolist())).hexdigest()


def check_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"H86 {label} mismatch: {actual!r} != {expected!r}")


def audit(input_root: Path, folds_path: Path, artifacts: Path, out: Path, s3_prefix: str) -> dict:
    started = datetime.now(timezone.utc)
    cpu_started = time.process_time()
    wall_started = time.monotonic()
    expected_inputs = {
        input_root / "source_manifest.json": PACKET_FILE_SHA,
        input_root / "selected.json": SELECTED_SHA,
        folds_path: FOLDS_SHA,
        artifacts / "h86-manifest.json": H86_ARTIFACTS["manifest"]["sha256"],
        artifacts / "h86-metrics.json": H86_ARTIFACTS["metrics"]["sha256"],
        artifacts / "status_mask.npz": H86_ARTIFACTS["mask"]["sha256"],
    }
    for path, expected in expected_inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"pinned file hash mismatch: {path}")

    packet = json.loads((input_root / "source_manifest.json").read_text())
    selected = json.loads((input_root / "selected.json").read_text())
    folds = json.loads(folds_path.read_text())
    h86_manifest = json.loads((artifacts / "h86-manifest.json").read_text())
    h86_metrics = json.loads((artifacts / "h86-metrics.json").read_text())
    if packet.get("source_manifest_sha256") != PACKET_ID or packet.get("outer_rows_delivered") != 0:
        raise ValueError("H50 packet identity or outer-row closure mismatch")
    if folds.get("training_role") != "DL-Train" or folds.get("selected_file_sha256") != SELECTED_SHA:
        raise ValueError("H65 fold manifest identity mismatch")
    if h86_manifest.get("run_id") != H86_RUN or h86_manifest.get("status_mask_sha256") != H86_ARTIFACTS["mask"]["sha256"]:
        raise ValueError("H86 run manifest does not bind the pinned mask")
    if set(h86_manifest.get("export_fields", [])) != FIELDS:
        raise ValueError("H86 export field allowlist differs")
    if h86_manifest.get("export_excludes") != "fitness, logratio, gate, program features and every DL-Val, outer, sealed or holdout field":
        raise ValueError("H86 exclusion declaration differs")
    if h86_manifest.get("metrics", {}).get("descriptive_metrics_file") != "metrics.json":
        raise ValueError("H86 manifest does not identify its metrics artifact")
    if h86_metrics.get("schema") != "jopedime.h86-w3-outcome-status-census-metrics.v1" or h86_metrics.get("w3_study_run") is not False:
        raise ValueError("H86 metrics are not the pinned census-only schema")

    rows = selected.get("train", [])
    if len(rows) != 1219 or any(row.get("dl_split") != "DL-Train" for row in rows):
        raise ValueError("H50 train selection is not exactly the admitted DL-Train roster")
    fold_of = {context: item["fold_index"] for item in folds["folds"] for context in item["held_contexts"]}
    if set(fold_of) != {row["context_sha256"] for row in rows}:
        raise ValueError("H65 folds and H50 DL-Train context IDs differ")
    rows = sorted(rows, key=lambda row: row["context_sha256"])

    with np.load(artifacts / "status_mask.npz", allow_pickle=False) as archive:
        if set(archive.files) != FIELDS:
            raise ValueError(f"unexpected mask archive members: {sorted(archive.files)}")
        cids = archive["cids"].astype(str)
        offsets = archive["offsets"]
        mask = archive["status"]
        pool_shas = archive["pool_sha256"].astype(str)
        order_shas = archive["program_order_sha256"].astype(str)

    n = len(rows)
    if cids.tolist() != [row["context_sha256"] for row in rows]:
        raise ValueError("mask context ordering does not match sorted admitted training contexts")
    if offsets.dtype.kind not in "iu" or offsets.tolist()[:1] != [0] or len(offsets) != n + 1:
        raise ValueError("mask offsets have invalid dtype, length, or origin")
    if np.any(np.diff(offsets) < 0) or int(offsets[-1]) != len(mask):
        raise ValueError("mask offsets are not monotonic or do not span status values")
    if mask.dtype != np.uint8 or np.any(mask > 4):
        raise ValueError("mask status codes are not the declared uint8 values 0..4")
    if len(pool_shas) != n or len(order_shas) != n:
        raise ValueError("per-context hash arrays do not match context count")

    names = NAMES
    totals: Counter[str] = Counter()
    by_fold: dict[int, Counter[str]] = defaultdict(Counter)
    context_with: Counter[str] = Counter()
    floor_contexts = clipped_rows = passing_rows = measured_logratio_rows = 0
    residuals: list[np.ndarray] = []
    fail_logratios: list[np.ndarray] = []
    fail_excesses: list[np.ndarray] = []
    pool_sha_list: list[str] = []
    mismatches = 0
    for index, row in enumerate(rows):
        cid = row["context_sha256"]
        pool_path = input_root / "pools" / f"{cid}.npz"
        expected_pool_sha = row["pool_sha256"]
        if sha256(pool_path) != expected_pool_sha or pool_shas[index] != expected_pool_sha:
            raise ValueError(f"pool hash mismatch at DL-Train context {cid}")
        with np.load(pool_path, allow_pickle=False) as pool:
            ids = pool["program_sha256"]
            fitness = np.asarray(pool["fitness"], dtype=np.float64)
            logratio = np.asarray(pool["logratio"], dtype=np.float64)
            gate = np.asarray(pool["gate"], dtype=np.bool_)
        size = len(ids)
        if not (size == len(fitness) == len(logratio) == len(gate) == row["n_candidates"]):
            raise ValueError(f"DL-Train arrays do not align for {cid}")
        if int(gate.sum()) != row["n_gate_pass"] or int(offsets[index + 1] - offsets[index]) != size:
            raise ValueError(f"gate or mask extent mismatch for {cid}")
        if order_shas[index] != order_sha(ids):
            raise ValueError(f"ordered program identity hash mismatch for {cid}")
        computed = classify(fitness, logratio, gate)
        start, stop = int(offsets[index]), int(offsets[index + 1])
        observed = mask[start:stop]
        mismatches += int(np.count_nonzero(computed != observed))
        if not np.array_equal(computed, observed):
            raise ValueError(f"status values mismatch against pinned H86 rule for {cid}")
        fold_counts = Counter(int(code) for code in computed)
        for code, name in enumerate(names):
            count = fold_counts[code]
            totals[name] += count
            by_fold[fold_of[cid]][name] += count
        for code, name in ((2, names[2]), (3, names[3]), (4, names[4])):
            context_with[name] += bool(fold_counts[code])
        passed = fitness[computed == 0]
        if len(passed):
            ordered = np.sort(passed)
            spread = float(ordered[min(16, len(ordered)) - 1] - ordered[0])
            floor_contexts += spread < 0.05
            regret = (ordered - ordered[0]) / max(spread, 0.05)
            clipped_rows += int(np.count_nonzero(regret > 6.0))
        passing_rows += len(passed)
        failed = computed == 1
        measured_logratio_rows += int(np.count_nonzero(computed <= 1))
        residuals.append(fitness[computed == 0] - 0.9 * logratio[computed == 0])
        fail_logratios.append(logratio[failed])
        fail_excesses.append(fitness[failed] - 100.0)
        pool_sha_list.append(expected_pool_sha)

    rows_total = int(len(mask))
    expected_counts = {name: int(totals[name]) for name in names}
    if sum(expected_counts.values()) != rows_total or rows_total != h86_metrics.get("rows"):
        raise ValueError("H86 total row count differs from independent pool replay")
    if expected_counts != h86_metrics.get("status_counts"):
        raise ValueError("H86 status totals differ from independent pool replay")
    expected_by_fold = {str(fold): {name: int(by_fold[fold][name]) for name in names} for fold in sorted(by_fold)}
    if expected_by_fold != h86_metrics.get("status_counts_by_fold"):
        raise ValueError("H86 per-fold status counts differ from independent pool replay")
    context_paths = h86_metrics["w3_arm_c_top_resolved"]["context_paths"]
    if context_paths != {"top16": 1219, "last_passing_order_statistic": 0, "no_passer": 0}:
        raise ValueError("H86 arm-c path counts differ from independent pool replay")
    check_close(float(floor_contexts), float(h86_metrics["w3_arm_c_top_resolved"]["floor_binds_contexts"]), "floor-bind contexts")
    check_close(float(clipped_rows), float(h86_metrics["w3_arm_c_top_resolved"]["clipped_passing_rows"]), "clipped rows")
    check_close(float(passing_rows), float(h86_metrics["status_counts"]["measured_pass"]), "passing rows")
    if hashlib.sha256("".join(pool_sha_list).encode()).hexdigest() != h86_manifest["inputs"][3]["pool_sha256_list_sha256"]:
        raise ValueError("H86 pool-hash-list digest differs")
    if mismatches != 0:
        raise ValueError("mask has status mismatches")

    output_prefix = s3_prefix.rstrip("/")
    out.mkdir(parents=True, exist_ok=False)
    metrics = {
        "schema": "jopedime.h87-w3-status-mask-audit-metrics.v1",
        "status": "INDEPENDENT_MASK_INTEGRITY_AUDIT",
        "h86_run_id": H86_RUN,
        "contexts": n,
        "families": len({row["family_id"] for row in rows}),
        "dl_train_rows": rows_total,
        "mask_status_mismatches": mismatches,
        "mask_archive_fields_exact": True,
        "mask_offsets_valid": True,
        "mask_pool_hashes_match": True,
        "mask_program_order_hashes_match": True,
        "status_counts": expected_counts,
        "status_counts_by_fold": expected_by_fold,
        "sentinel_contexts": int(context_with["sentinel_shaped"]),
        "arm_c_floor_bind_contexts": int(floor_contexts),
        "arm_c_clipped_passing_rows": int(clipped_rows),
        "arm_d_measured_logratio_rows": int(measured_logratio_rows),
        "mask_sha256": H86_ARTIFACTS["mask"]["sha256"],
        "source_manifest_sha256": PACKET_FILE_SHA,
        "selected_sha256": SELECTED_SHA,
        "fold_manifest_sha256": FOLDS_SHA,
        "Elo": None,
        "U4": None,
        "val_reg_inner": None,
        "w3_study_run": False,
        "scientific_verdict": None,
    }
    manifest = {
        "schema": "jopedime.h87-w3-status-mask-audit-run.v1",
        "run_id": output_prefix.rsplit("/", 1)[-1],
        "source_commit": os.environ.get("JOPE_SOURCE_SHA") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_file_sha256": sha256(Path(__file__)),
        "s3_output_prefix": output_prefix,
        "s3_outputs": [f"{output_prefix}/{name}" for name in ("metrics.json", "manifest.json")],
        "h86_artifacts": H86_ARTIFACTS,
        "inputs": {
            "source_manifest_sha256": PACKET_FILE_SHA,
            "selected_sha256": SELECTED_SHA,
            "fold_manifest_sha256": FOLDS_SHA,
            "train_context_count": n,
            "pool_sha256_list_sha256": hashlib.sha256("".join(pool_sha_list).encode()).hexdigest(),
            "opened_payload_roles": ["DL-Train"],
            "opened_payload_count": n,
            "excluded_roles_opened": [],
        },
        "host": socket.gethostname(),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_seconds": time.process_time() - cpu_started,
        "wall_seconds": time.monotonic() - wall_started,
        "metrics": {"Elo": None, "val_reg_inner": None, "U4": None, "descriptive_metrics_file": "metrics.json"},
        "verdict": "INTEGRITY_ONLY; NO_PROVENANCE_OR_W3_ADMISSION_CLAIM",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--folds", type=Path, default=ROOT / "research/flywheel_selfplay/h65_w3_continuous_top_resolved/fold_manifest.json")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--s3-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.folds, args.artifacts, args.out, args.s3_prefix), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
