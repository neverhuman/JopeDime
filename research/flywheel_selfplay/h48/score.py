#!/usr/bin/env python3
"""Perform the terminal DL-Val score once for each selected H48 arm/seed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run as h48  # noqa: E402
import run_fused_closed_loop as base  # noqa: E402


def load_validation_packet(input_dir: Path):
    manifest = json.loads((input_dir / "source_manifest.json").read_text())
    if (
        manifest.get("source_manifest_sha256")
        != "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("terminal source-manifest red-team audit failed")
    selected = json.loads((input_dir / "selected.json").read_text())
    dev = selected.get("dev", [])
    if len(dev) != 91 or len({row["family_id"] for row in dev}) != 19:
        raise RuntimeError("terminal DL-Val roster mismatch")
    if any(row.get("dl_split") != "DL-Val" or row.get("task") != "regression" for row in dev):
        raise RuntimeError("terminal score received a non-DL-Val or non-regression row")
    pools = {}
    tables = {}
    for row in dev:
        cid = row["context_sha256"]
        pool_path = input_dir / "pools" / f"{cid}.npz"
        table_path = input_dir / "tables" / f"{cid}.jtt1"
        if base.sha(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"DL-Val pool hash mismatch: {cid}")
        if base.sha(table_path) != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"DL-Val train-only table hash mismatch: {cid}")
        with np.load(pool_path, allow_pickle=False) as source:
            programs = source["program"].astype(np.float32)
            pids = [value.decode() if isinstance(value, bytes) else str(value) for value in source["program_sha256"].tolist()]
            pools[cid] = {
                "rec": row,
                "pids": pids,
                "program": programs,
                "proj16": programs.astype(np.float64) @ base.evaluator.get_w1_sign_matrix() / np.sqrt(412),
                "struct_sig": base.compute_semantic_struct_sig_24d(programs),
                "fitness": source["fitness"].astype(np.float64),
                "logratio": source["logratio"].astype(np.float64),
                "gate": source["gate"].astype(bool),
                "prior_cost": np.zeros(len(programs), dtype=np.float64),
                "scale": 0.4208,
            }
        raw = table_path.read_bytes()
        nt, nq, width = base.struct.unpack_from("<III", raw, 4)
        if nq or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError("DL-Val requires train-only two-fold inputs")
        values = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
        tables[cid] = (values[:, :-1], values[:, -1])
    return dev, pools, tables


def bootstrap_median_seed_family_interval(seed_rows, field, seed, n=10000):
    families = sorted(seed_rows[0][field])
    if len(families) != 19:
        raise RuntimeError("family bootstrap requires all 19 DL-Val families")
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    for index in range(n):
        draw = rng.choice(families, size=len(families), replace=True)
        seed_metrics = [
            float(np.mean([row[field][family] for family in draw]))
            for row in seed_rows
        ]
        values[index] = float(np.median(seed_metrics))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    root = Path.home() / "jope-data/h48"
    input_dir = root / "input"
    checkpoint = root / "ckpt/round-785.pt"
    reference = json.loads((root / "reference/r785-val-baseline.json").read_text())
    if reference.get("source_checkpoint_sha256") != h48.CHECKPOINT_SHA:
        raise RuntimeError("incumbent DL-Val reference checkpoint mismatch")
    device_selection = json.loads((root / "runs/device-selection.json").read_text())
    device_name = device_selection["device"]
    device = torch.device(device_name)
    for arm in ("0", "PF", "ES"):
        for seed in range(3):
            seed_dir = root / "runs" / f"H48-{arm}" / f"seed{seed}"
            if not (seed_dir / "train-result.json").exists() or not (
                seed_dir / "selected-candidate.json"
            ).exists():
                raise RuntimeError("refusing terminal scoring before all H48 training selections exist")
            if (seed_dir / "final-val.json").exists():
                raise RuntimeError(f"refusing to repeat terminal DL-Val score for H48-{arm} seed{seed}")
    lock = open(Path.home() / "jope_plan/qualification.lock", "a")
    while True:
        if base.psutil.virtual_memory().available < 20 * 1024**3:
            lock.close()
            raise RuntimeError("available RAM below 20 GiB before terminal DL-Val scoring")
        if device.type == "cpu" and base.psutil.cpu_percent(interval=1) > 85:
            print("host CPU above 85%; waiting before terminal DL-Val scoring", flush=True)
            time.sleep(10)
            continue
        try:
            base.fcntl.flock(lock, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
        except BlockingIOError:
            print("qualification.lock busy; waiting before terminal scoring", flush=True)
            time.sleep(15)
            continue
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < 20 * 1024**3:
                base.fcntl.flock(lock, base.fcntl.LOCK_UN)
                time.sleep(30)
                continue
        break
    try:
        torch.set_num_threads(4)
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.90)
        dev_rows, dev_pools, dev_tables = load_validation_packet(input_dir)
        if [row["context_sha256"] for row in dev_rows] != reference["context_ids"]:
            raise RuntimeError("terminal DL-Val roster differs from pinned incumbent reference")
        spec = importlib.util.spec_from_file_location("h48_score_fused", input_dir / "cpu_exact_fused.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("approved fused model module could not be loaded")
        fused = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused)
        guard = base.guard
        net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        for name, parameter in net.adapter_10tok.named_parameters():
            if not name.startswith("anchor."):
                parameter.requires_grad_(True)
        guard.lock_dope_weights_and_eval_mode(net)
        checkpoint_state = torch.load(checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint_state["state_dict"], strict=True)
        if h48.h46.checkpoint_guard(guard, net) != h48.DOPE_SHA:
            raise RuntimeError("terminal scorer DOPE identity mismatch")
        if h48.digest(checkpoint) != h48.CHECKPOINT_SHA:
            raise RuntimeError("terminal scorer checkpoint SHA mismatch")
        tokens, masks = h48.h46.make_tokens(net, dev_tables, device)
        predict = h48.make_predictor(net, dev_pools, tokens, masks, device)
        dev_ids = [row["context_sha256"] for row in dev_rows]
        if set(dev_ids) != set(reference["per_context_utility"]):
            raise RuntimeError("incumbent reference context identity mismatch")

        scored = {}
        for arm in ("0", "PF", "ES"):
            seed_rows = []
            for seed in range(3):
                seed_dir = root / "runs" / f"H48-{arm}" / f"seed{seed}"
                if (seed_dir / "final-val.json").exists():
                    raise RuntimeError(f"refusing to score DL-Val twice for {arm} seed{seed}")
                selection = json.loads((seed_dir / "selected-candidate.json").read_text())
                train_result = json.loads((seed_dir / "train-result.json").read_text())
                artifact = seed_dir / selection["checkpoint"]
                state = torch.load(artifact, map_location=device, weights_only=True)
                if state.get("dope_sha256") != h48.DOPE_SHA or state.get("base_checkpoint_sha256") != h48.CHECKPOINT_SHA:
                    raise RuntimeError(f"selected adapter identity mismatch: {artifact}")
                h48.load_adapter_snapshot(net, state["adapter_trainable_state"])
                net.eval()
                started = time.monotonic()
                candidate = h48.evaluate_utility(dev_ids, dev_pools, predict)
                h48.h46.checkpoint_guard(guard, net, expected=h48.DOPE_SHA)
                if set(candidate["per_context"]) != set(reference["per_context_utility"]):
                    raise RuntimeError("terminal candidate context identity mismatch")
                family_delta = {
                    family: candidate["per_family"][family] - reference["per_family_utility"][family]
                    for family in reference["per_family_utility"]
                }
                context_delta = {
                    cid: candidate["per_context"][cid] - reference["per_context_utility"][cid]
                    for cid in dev_ids
                }
                row = {
                    "arm": arm,
                    "seed": seed,
                    "device": device_name,
                    "selected_fold": selection["fold"],
                    "selected_inner_cv_delta": selection["selected_delta"],
                    "val_context_utility": float(np.mean(list(candidate["per_context"].values()))),
                    "val_family_utility": candidate["mean"],
                    "final_delta": float(np.mean(list(context_delta.values()))),
                    "family_mean_delta": float(np.mean(list(family_delta.values()))),
                    "family_delta": family_delta,
                    "context_delta": context_delta,
                    "candidate_per_family_utility": candidate["per_family"],
                    "candidate_per_context_utility": candidate["per_context"],
                    "inner_cv_train_utility": train_result["inner_cv_selected_family_utility"],
                    "inner_cv_train_val_gap": train_result["inner_cv_selected_family_utility"] - candidate["mean"],
                    "train_wall_seconds": train_result["wall_seconds"],
                    "terminal_scoring_wall_seconds": time.monotonic() - started,
                    "checkpoint_sha256": h48.CHECKPOINT_SHA,
                    "dope_sha256": h48.DOPE_SHA,
                    "selected_adapter_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "trainable_params": h48.TRAINABLE_PARAMS,
                    "terminal_dl_val_score_count": 1,
                    "validation_payloads_opened": True,
                }
                out = seed_dir / "final-val.json"
                out.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                seed_rows.append(row)
                print(json.dumps({"arm": arm, "seed": seed, "final_delta": row["final_delta"], "val": row["val_context_utility"]}), flush=True)

            delta_ci = bootstrap_median_seed_family_interval(seed_rows, "family_delta", seed=48000 + ord(arm))
            utility_ci = bootstrap_median_seed_family_interval(seed_rows, "candidate_per_family_utility", seed=48100 + ord(arm))
            median_delta = float(np.median([row["final_delta"] for row in seed_rows]))
            median_utility = float(np.median([row["val_context_utility"] for row in seed_rows]))
            min_loco = min(
                float(np.mean([value for family, value in row["family_delta"].items() if family != left_out]))
                for row in seed_rows
                for left_out in row["family_delta"]
            )
            signatures = {tuple(sorted(row["context_delta"].items())) for row in seed_rows}
            scored[f"H48-{arm}"] = {
                "seeds": seed_rows,
                "median_final_delta": median_delta,
                "family_bootstrap_95_ci_of_median_seed_family_delta": delta_ci,
                "median_val_utility": median_utility,
                "family_bootstrap_95_ci_of_median_seed_family_utility": utility_ci,
                "median_inner_cv_train_val_gap": float(np.median([row["inner_cv_train_val_gap"] for row in seed_rows])),
                "minimum_leave_one_family_out_delta": min_loco,
                "effective_seed_replicates": len(signatures),
                "regression_elo": None,
                "strict_win": median_delta > 0 and delta_ci[0] > 0,
            }

        elo_path = ROOT / "python/src/jope/north_star/elo_tracker.py"
        elo_spec = importlib.util.spec_from_file_location("h48_elo_tracker", elo_path)
        if elo_spec is None or elo_spec.loader is None:
            raise RuntimeError("regression Elo tracker could not be loaded")
        elo = importlib.util.module_from_spec(elo_spec)
        sys.modules[elo_spec.name] = elo
        elo_spec.loader.exec_module(elo)
        for arm, record in scored.items():
            seed_rows = record["seeds"]
            median_by_cid = {
                cid: float(np.median([row["candidate_per_context_utility"][cid] for row in seed_rows]))
                for cid in dev_ids
            }
            elo_records = [
                elo.DatasetEvalRecord(
                    dataset_id=cid,
                    family_id=reference["family_ids"][cid],
                    task_type="regression",
                    scores_by_contender={"r785": reference["per_context_utility"][cid], arm: median_by_cid[cid]},
                )
                for cid in dev_ids
            ]
            record["regression_elo"] = elo.compute_family_grouped_bradley_terry_elo(
                elo_records, anchor_contender="r785", anchor_elo=1500.0
            )

        summary = {
            "schema": "jope.h48.terminal-analysis/v1",
            "device": device_name,
            "baseline": "r785 (reused H47 terminal baseline; no new baseline DL-Val score)",
            "selection": "inner family-grouped DL-Train CV; one terminal DL-Val score per selected arm/seed",
            "validation_contexts": 91,
            "validation_families": 19,
            "strict_win_rule": "median final delta > 0 and family-bootstrap 95% lower bound > 0; ties are failures",
            "source_manifest_sha256": "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3",
            "excluded_dl_holdout_count": 93,
            "excluded_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "dope_sha256": h48.DOPE_SHA,
            "checkpoint_sha256": h48.CHECKPOINT_SHA,
            "trainable_params": h48.TRAINABLE_PARAMS,
            "results": scored,
        }
        (root / "runs" / "h48-final-analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
