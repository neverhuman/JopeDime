#!/usr/bin/env python3
"""One terminal DL-Val score per H49 trained arm/seed, after training is complete."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SOURCE_MANIFEST_SHA = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H48_DIR = HERE.parent / "h48"
sys.path.insert(0, str(H48_DIR))
import run as h48  # noqa: E402
import run_fused_closed_loop as base  # noqa: E402


def load_validation_packet(input_dir: Path):
    manifest = json.loads((input_dir / "source_manifest.json").read_text())
    if (
        manifest.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("terminal source-manifest audit failed")
    selected = json.loads((input_dir / "selected.json").read_text())
    dev = selected.get("dev", [])
    if len(dev) != 91 or len({row["family_id"] for row in dev}) != 19:
        raise RuntimeError("terminal DL-Val roster mismatch")
    if any(row.get("dl_split") != "DL-Val" or row.get("task") != "regression" for row in dev):
        raise RuntimeError("non-DL-Val rows in terminal packet")
    pools, tables = {}, {}
    for row in dev:
        cid = row["context_sha256"]
        pool_path = input_dir / "pools" / f"{cid}.npz"
        table_path = input_dir / "tables" / f"{cid}.jtt1"
        if base.sha(pool_path) != row["pool_sha256"] or base.sha(table_path) != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"validation input hash mismatch: {cid}")
        with np.load(pool_path, allow_pickle=False) as source:
            programs = source["program"].astype(np.float32)
            pids = [v.decode() if isinstance(v, bytes) else str(v) for v in source["program_sha256"].tolist()]
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
            raise RuntimeError("terminal val requires train-only two-fold tables")
        values = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
        tables[cid] = (values[:, :-1], values[:, -1])
    return dev, pools, tables


def bootstrap_interval(seed_rows, field, seed, n=10_000):
    families = sorted(seed_rows[0][field])
    if len(families) != 19:
        raise RuntimeError("family bootstrap requires all 19 DL-Val families")
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = rng.choice(families, size=len(families), replace=True)
        draws[i] = np.median([
            np.mean([row[field][fam] for fam in sample]) for row in seed_rows
        ])
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def arm_seeds():
    return [("0", range(3)), ("SAM", range(3)), ("SWA", range(3)), ("ES2", range(3, 6))]


def main():
    root = Path.home() / "jope-data/h49"
    input_dir = root / "input"
    checkpoint = root / "ckpt/round-785.pt"
    reference = json.loads((root / "reference/r785-val-baseline.json").read_text())
    if reference.get("source_checkpoint_sha256") != h48.CHECKPOINT_SHA:
        raise RuntimeError("r785 reference/checkpoint identity mismatch")
    device_selection = json.loads((root / "runs/device-selection.json").read_text())
    device_name = device_selection.get("device")
    if device_name != "cuda:0":
        raise RuntimeError("H49 terminal reads are preregistered for cuda:0 only")
    device = torch.device(device_name)
    for logical_arm, seeds in arm_seeds():
        for seed in seeds:
            seed_dir = root / "runs" / f"H49-{logical_arm}" / f"seed{seed}"
            if not (seed_dir / "train-result.json").exists() or not (seed_dir / "selected-candidate.json").exists():
                raise RuntimeError(f"refusing terminal scoring before H49-{logical_arm} seed{seed} selection")
    lock = open(Path.home() / "jope_plan/qualification.lock", "a")
    while True:
        if base.psutil.virtual_memory().available < 20 * 1024**3:
            lock.close()
            raise RuntimeError("available RAM below 20 GiB before terminal DL-Val scoring")
        if base.psutil.cpu_percent(interval=1) > 85:
            print("host CPU above 85%; waiting before terminal scoring", flush=True)
            time.sleep(10)
            continue
        try:
            base.fcntl.flock(lock, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
        except BlockingIOError:
            print("qualification.lock busy; waiting before terminal scoring", flush=True)
            time.sleep(15)
            continue
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < 20 * 1024**3:
            base.fcntl.flock(lock, base.fcntl.LOCK_UN)
            time.sleep(30)
            continue
        break
    try:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.90)
        dev_rows, dev_pools, dev_tables = load_validation_packet(input_dir)
        if [row["context_sha256"] for row in dev_rows] != reference["context_ids"]:
            raise RuntimeError("DL-Val roster differs from pinned r785 baseline")
        if {row["family_id"] for row in dev_rows} != set(reference["per_family_utility"]):
            raise RuntimeError("DL-Val family set differs from pinned r785 baseline")
        fused_spec = importlib.util.spec_from_file_location("h49_score_fused", input_dir / "cpu_exact_fused.py")
        if fused_spec is None or fused_spec.loader is None:
            raise RuntimeError("cannot load approved fused model")
        fused = importlib.util.module_from_spec(fused_spec)
        fused_spec.loader.exec_module(fused)
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
        if h48.h46.checkpoint_guard(guard, net) != h48.DOPE_SHA or h48.digest(checkpoint) != h48.CHECKPOINT_SHA:
            raise RuntimeError("H49 scorer checkpoint/DOPE identity mismatch")
        train_selected, train_pools, train_tables = h48.load_train_packet(input_dir)
        train_ids = [row["context_sha256"] for row in train_selected["train"]]
        train_families = {row["family_id"] for row in train_selected["train"]}
        val_families = {row["family_id"] for row in dev_rows}
        context_overlap = len(set(train_ids) & {row["context_sha256"] for row in dev_rows})
        family_overlap = len(train_families & val_families)
        if context_overlap or family_overlap:
            raise RuntimeError("train/validation overlap in H49 scoring packet")
        train_tokens, train_masks = h48.h46.make_tokens(net, train_tables, device)
        train_predict = h48.make_predictor(net, train_pools, train_tokens, train_masks, device)
        tokens, masks = h48.h46.make_tokens(net, dev_tables, device)
        predict = h48.make_predictor(net, dev_pools, tokens, masks, device)
        dev_ids = [row["context_sha256"] for row in dev_rows]
        if set(dev_ids) != set(reference["per_context_utility"]):
            raise RuntimeError("baseline context identities differ from DL-Val packet")

        scored = {}
        for logical_arm, seeds in arm_seeds():
            rows = []
            for seed in seeds:
                seed_dir = root / "runs" / f"H49-{logical_arm}" / f"seed{seed}"
                final_path = seed_dir / "final-val.json"
                if final_path.exists():
                    prior = json.loads(final_path.read_text())
                    if prior.get("terminal_dl_val_score_count") != 1:
                        raise RuntimeError(f"invalid persisted terminal count for {logical_arm} seed{seed}")
                    rows.append(prior)
                    continue
                selection = json.loads((seed_dir / "selected-candidate.json").read_text())
                train_result = json.loads((seed_dir / "train-result.json").read_text())
                artifact = seed_dir / selection["checkpoint"]
                adapter = torch.load(artifact, map_location=device, weights_only=True)
                if adapter.get("dope_sha256") != h48.DOPE_SHA or adapter.get("base_checkpoint_sha256") != h48.CHECKPOINT_SHA:
                    raise RuntimeError(f"selected H49 adapter identity mismatch: {artifact}")
                h48.load_adapter_snapshot(net, adapter["adapter_trainable_state"])
                net.eval()
                started = time.monotonic()
                full_train = h48.evaluate_utility(train_ids, train_pools, train_predict)
                h48.h46.checkpoint_guard(guard, net, expected=h48.DOPE_SHA)
                candidate = h48.evaluate_utility(dev_ids, dev_pools, predict)
                fam_delta = {fam: candidate["per_family"][fam] - reference["per_family_utility"][fam] for fam in reference["per_family_utility"]}
                ctx_delta = {cid: candidate["per_context"][cid] - reference["per_context_utility"][cid] for cid in dev_ids}
                row = {
                    "arm": logical_arm, "seed": seed, "device": device_name,
                    "selected_fold": selection["fold"], "selected_inner_cv_delta": selection["selected_delta"],
                    "val_context_utility": float(np.mean(list(candidate["per_context"].values()))),
                    "val_family_utility": candidate["mean"],
                    "final_delta": float(np.mean(list(ctx_delta.values()))),
                    "family_mean_delta": float(np.mean(list(fam_delta.values()))),
                    "family_delta": fam_delta, "context_delta": ctx_delta,
                    "candidate_per_family_utility": candidate["per_family"],
                    "candidate_per_context_utility": candidate["per_context"],
                    "inner_cv_train_utility": train_result["inner_cv_selected_family_utility"],
                    "optimistic_selected_inner_cv_train_val_gap": train_result["inner_cv_selected_family_utility"] - candidate["mean"],
                    "full_train_context_utility": full_train["context_mean"],
                    "full_train_family_utility": full_train["mean"],
                    "flywheel_train_val_gap_context_mean": full_train["context_mean"] - float(np.mean(list(candidate["per_context"].values()))),
                    "train_wall_seconds": train_result["wall_seconds"],
                    "terminal_scoring_wall_seconds": time.monotonic() - started,
                    "checkpoint_sha256": h48.CHECKPOINT_SHA, "dope_sha256": h48.DOPE_SHA,
                    "selected_adapter_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "trainable_params": h48.TRAINABLE_PARAMS,
                    "terminal_dl_val_score_count": 1, "validation_payloads_opened": True,
                }
                temporary = final_path.with_suffix(".json.new")
                temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                temporary.replace(final_path)
                rows.append(row)
                print(json.dumps({"arm": logical_arm, "seed": seed, "final_delta": row["final_delta"]}), flush=True)
            fam_ci = bootstrap_interval(rows, "family_delta", 49000 + sum(map(ord, logical_arm)))
            median_fam = float(np.median([row["family_mean_delta"] for row in rows]))
            record = {
                "seeds": rows,
                "median_family_mean_final_delta": median_fam,
                "family_bootstrap_95_ci_of_median_seed_family_delta": fam_ci,
                "median_context_weighted_final_delta": float(np.median([row["final_delta"] for row in rows])),
                "median_val_utility": float(np.median([row["val_context_utility"] for row in rows])),
                "minimum_leave_one_family_out_delta": min(
                    float(np.mean([v for fam, v in row["family_delta"].items() if fam != excluded]))
                    for row in rows for excluded in row["family_delta"]
                ),
                "effective_seed_replicates": len({tuple(sorted(row["context_delta"].items())) for row in rows}),
                "regression_elo": None,
                "strict_win": False,
            }
            scored[f"H49-{logical_arm}"] = record

        baseline_rows = scored["H49-0"]["seeds"]
        baseline_family = {
            fam: float(np.median([row["candidate_per_family_utility"][fam] for row in baseline_rows]))
            for fam in reference["per_family_utility"]
        }
        baseline_context = {
            cid: float(np.median([row["candidate_per_context_utility"][cid] for row in baseline_rows]))
            for cid in dev_ids
        }
        for name in ("H49-SAM", "H49-SWA", "H49-ES2"):
            paired_rows = []
            for candidate_row in scored[name]["seeds"]:
                delta = {fam: candidate_row["candidate_per_family_utility"][fam] - baseline_family[fam] for fam in baseline_family}
                paired_rows.append({"seed": candidate_row["seed"], "family_delta": delta,
                    "family_mean_delta": float(np.mean(list(delta.values()))),
                    "context_mean_delta": candidate_row["val_context_utility"] - float(np.mean(list(baseline_context.values())))})
            scored[name]["paired_vs_H49_0"] = {
                "median_family_mean_delta": float(np.median([row["family_mean_delta"] for row in paired_rows])),
                "family_bootstrap_95_ci_of_median_seed_family_delta": bootstrap_interval(paired_rows, "family_delta", 49100 + sum(map(ord, name))),
                "median_context_mean_delta": float(np.median([row["context_mean_delta"] for row in paired_rows])),
                "paired_families": 19,
                "seed_pairing": "candidate seeds compared with per-family median of the three H49-0 seeds",
            }

        elo_path = ROOT / "python/src/jope/north_star/elo_tracker.py"
        elo_spec = importlib.util.spec_from_file_location("h49_elo_tracker", elo_path)
        if elo_spec is None or elo_spec.loader is None:
            raise RuntimeError("regression Elo tracker could not be loaded")
        elo = importlib.util.module_from_spec(elo_spec)
        sys.modules[elo_spec.name] = elo
        elo_spec.loader.exec_module(elo)
        for name, record in scored.items():
            rows = record["seeds"]
            median_by_cid = {cid: float(np.median([row["candidate_per_context_utility"][cid] for row in rows])) for cid in dev_ids}
            records = [elo.DatasetEvalRecord(dataset_id=cid, family_id=reference["family_ids"][cid], task_type="regression",
                scores_by_contender={"r785": reference["per_context_utility"][cid], name: median_by_cid[cid]}) for cid in dev_ids]
            record["regression_elo"] = elo.compute_family_grouped_bradley_terry_elo(records, anchor_contender="r785", anchor_elo=1500.0)
            record["strict_win"] = record["median_family_mean_final_delta"] > 0 and record["family_bootstrap_95_ci_of_median_seed_family_delta"][0] > 0 and record["regression_elo"]["ratings"][name] > 1500.0

        summary = {
            "schema": "jope.h49.terminal-analysis/v1", "device": device_name,
            "baseline": "r785 reused from H48; no baseline rescoring",
            "selection": "five fixed whole-family DL-Train folds; one terminal DL-Val read per trained arm/seed",
            "validation_contexts": 91, "validation_families": 19,
            "train_contexts": len(train_ids), "train_families": len(train_families),
            "context_overlap": context_overlap, "family_overlap": family_overlap,
            "terminal_reads_per_selected_seed": 1,
            "strict_win_rule": "median family-mean delta > 0 AND 10k family-bootstrap lower bound > 0 AND Elo > 1500",
            "selection_bias": "inner-CV maxima are selected estimates, not generalization gains; selected fold models fit 4/5 of train families",
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "excluded_dl_holdout_count": 93, "excluded_payloads_opened": 0, "outer_rows_delivered": 0,
            "dope_sha256": h48.DOPE_SHA, "checkpoint_sha256": h48.CHECKPOINT_SHA,
            "trainable_params": h48.TRAINABLE_PARAMS, "results": scored,
        }
        h48.atomic_json(root / "runs/h49-final-analysis.json", summary)
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
