#!/usr/bin/env python3
"""H255 Part B: Preregistered Live Rust GP 2+2 Closed-Loop Flywheel (`evaluate_gp_batch_rust`).

Executes a 2-round live closed-loop GP + Neural Critic experiment on 32 stratified DL-Train contexts
(32 distinct families, train-only 2-fold JTT1 inner CV via `training_cv_envelope`, `run_outer_test=False`),
comparing:
1. `LIVE_GP_LEGACY_STATIC_4`: Legacy `run_fused_closed_loop.py` proposal ranking (arbitrary pool slots `0..3` as revealed history, one-shot top-4).
2. `LIVE_GP_CHAMPION_COND_STATIC_4`: Champion-conditioned one-shot top-4 (true top-4 passing pool champions as revealed history).
3. `LIVE_GP_CAUSAL_2PLUS2_FLY`: Live 2+2 sequential Rust GP evaluation (Batch 1: 2 Rust CV evaluations -> observe live 2-fold JTT1 fitness & gate -> condition `adapter_10tok` trajectory kernel/cross-attn + structural diversity/exploitation on Batch 1 outcomes -> Batch 2: 2 Rust CV evaluations; total = 4 Rust CV evaluations/context/round).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
H252_DIR = HERE.parent / "h252_closed_loop_flywheel"
SELFPLAY_DIR = HERE.parent
MICRO = Path("/home/ubuntu/JopeDimeMicro")
OUT_DIR = Path("/home/ubuntu/jope-data/h255/part_b")

sys.path.insert(0, str(MICRO / "python"))
sys.path.insert(0, str(SELFPLAY_DIR))
sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(HERE))

import rust_evaluator as evaluator  # noqa: E402
from automatic_promotion import training_cv_envelope  # noqa: E402
import common_h252 as ch  # noqa: E402
import part_a_causal_2plus2_neural as pa  # noqa: E402

SAMPLE_SEED = 255140
ROUND_SEEDS = (255201, 255202)
BOOT_SEED = 255150
BOOT_DRAWS = 10_000
N_COHORT = 32
N_PROPOSALS = 16
BUDGET_EVAL = 4


def select_stratified_32_dl_train_cohort(
    train_rows: List[Dict[str, Any]],
    input_dir: Path,
) -> List[Dict[str, Any]]:
    """Deterministically select 32 DL-Train contexts across 32 distinct families with table size <= 120 KB for fast exact 2-fold Rust CV."""
    by_fam: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
    for r in train_rows:
        if r.get("dl_split") != "DL-Train" or r.get("has_meta_test") is True:
            raise RuntimeError("Invalid non-DL-Train context in cohort selection")
        c = r["context_sha256"]
        t_path = input_dir / "tables" / f"{c}.jtt1"
        sz = t_path.stat().st_size
        if sz <= 120_000:
            by_fam.setdefault(r["family_id"], []).append((sz, c, r))
    eligible_fams = sorted(by_fam.keys())
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen_fams = sorted(rng.choice(eligible_fams, size=min(N_COHORT, len(eligible_fams)), replace=False).tolist())
    selected = []
    for fam in chosen_fams:
        items = sorted(by_fam[fam], key=lambda x: (x[0], x[1]))
        selected.append(items[0][2])
    return selected


def predict_proposal_costs(
    net,
    device: torch.device,
    cf: Dict[str, np.ndarray],
    hist_prog: np.ndarray,
    hist_proj: np.ndarray,
    hist_f: np.ndarray,
    hist_g: np.ndarray,
    prop_prog: np.ndarray,
    prop_proj: np.ndarray,
) -> np.ndarray:
    """Run `net.adapter_10tok.predict_costs` conditioned on `hist_*` (length K_rev) to score `prop_*` (length P)."""
    k_rev = len(hist_f)
    p_len = len(prop_prog)
    m = k_rev + p_len
    prog_all = np.concatenate([hist_prog, prop_prog], axis=0).astype(np.float32)
    proj_all = np.concatenate([hist_proj, prop_proj], axis=0).astype(np.float32)
    with torch.no_grad():
        out = net.adapter_10tok.predict_costs(
            ctx_feat=torch.as_tensor(cf["ctx_feat_584"], device=device).reshape(1, 584),
            prog_412=torch.as_tensor(prog_all, device=device).reshape(1, m, 412),
            proj_16=torch.as_tensor(proj_all, device=device).reshape(1, m, 16),
            base_prior_cost=torch.zeros((1, m), device=device),
            scale=torch.full((1,), 0.4208, device=device),
            sos_prior=torch.zeros((1, m, 2), device=device),
            mm_tokens_10x32=torch.as_tensor(cf["tokens_10x32"], device=device).reshape(1, 10, 32),
            mm_mask_10=torch.as_tensor(cf["mask_10"], device=device).reshape(1, 10),
            revealed_indices=torch.arange(k_rev, dtype=torch.long, device=device).reshape(1, k_rev),
            revealed_raw_f=torch.as_tensor(hist_f, dtype=torch.float32, device=device).reshape(1, k_rev),
            revealed_gate=torch.as_tensor(hist_g, dtype=torch.float32, device=device).reshape(1, k_rev),
        )["total_cost"][0, k_rev:].cpu().numpy().astype(np.float64)
    return out


def main() -> None:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_hashes = pa.verify_frozen_h252_files()

    if ch.sha256_file(evaluator.WORKER_BIN) != evaluator.WORKER_SHA256:
        raise RuntimeError("Rust evaluator binary hash drift")

    train_rows, val_rows = ch.load_cohorts()
    cohort_rows = select_stratified_32_dl_train_cohort(train_rows, ch.INPUT_DIR)
    cohort_cids = [r["context_sha256"] for r in cohort_rows]
    pools_base = ch.load_pools(cohort_cids + [r["context_sha256"] for r in val_rows])
    ctx_feats = ch.load_context_features()

    # Verify Native91 anchor first
    native_check = ch.verify_r882_native91(val_rows, pools_base)
    if not native_check["passed"]:
        raise RuntimeError("Native91 check failed")

    device = torch.device("cpu")
    torch.set_num_threads(2)
    net, guard, before = pa.load_fused_network(device)

    # Load initial champions bank restricted strictly to DL-Train
    train_cid_set = {r["context_sha256"] for r in train_rows}
    raw_bank = [
        r for r in json.loads((ch.INPUT_DIR / "champions.json").read_text())
        if r["context_sha256"] in train_cid_set
    ]
    for parent in raw_bank:
        if "feat412" not in parent:
            c = parent["context_sha256"]
            lm_sum = ctx_feats[c]["lm48_z"][:12].tolist() if c in ctx_feats else [0.0] * 12
            feat, proj = evaluator.featurize_dsl_412(parent["dsl"])
            parent.update(
                feat412=feat.tolist(),
                proj16=proj.tolist(),
                landmark_summary=lm_sum,
            )

    # Prepare train-only 2-fold JTT1 envelopes once for the 32 contexts
    env_dir = OUT_DIR / "train_inputs"
    env_dir.mkdir(parents=True, exist_ok=True)
    gz_paths: Dict[str, Path] = {}
    for c in cohort_cids:
        gz = env_dir / f"{c}.jtt1.gz"
        if not gz.exists():
            raw_bytes = (ch.INPUT_DIR / "tables" / f"{c}.jtt1").read_bytes()
            gz.write_bytes(gzip.compress(training_cv_envelope(raw_bytes), mtime=0))
        gz_paths[c] = gz

    arm_names = (
        "LIVE_GP_LEGACY_STATIC_4",
        "LIVE_GP_CHAMPION_COND_STATIC_4",
        "LIVE_GP_CAUSAL_2PLUS2_FLY",
    )

    # Maintain independent pool incumbent state and parent bank per arm across the 2 closed-loop rounds
    incumbent_fitness: Dict[str, Dict[str, float]] = {a: {} for a in arm_names}
    initial_incumbent_fitness: Dict[str, float] = {}
    known_pids_arm: Dict[str, Dict[str, set]] = {a: {} for a in arm_names}
    parents_by_ctx_arm: Dict[str, Dict[str, List[Dict[str, Any]]]] = {a: {} for a in arm_names}

    for c in cohort_cids:
        p = pools_base[c]
        passing = p["fitness"][p["gate"]]
        inc_f = float(np.min(passing)) if len(passing) else 1.0
        initial_incumbent_fitness[c] = inc_f
        base_parents = [ copy.deepcopy(r) for r in raw_bank if r["context_sha256"] == c ]
        for a in arm_names:
            incumbent_fitness[a][c] = inc_f
            known_pids_arm[a][c] = set(p["pids"])
            parents_by_ctx_arm[a][c] = copy.deepcopy(base_parents)

    eval_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rust_eval_calls = 0
    rust_eval_candidates = 0

    def evaluate_candidates_cached(cid: str, cand_list: List[Dict[str, Any]], rnd_idx: int) -> List[Dict[str, Any]]:
        nonlocal rust_eval_calls, rust_eval_candidates
        missing = [x for x in cand_list if (cid, x["program_sha256"]) not in eval_cache]
        if missing:
            evaluator.EVAL_SCRATCH_ROOT = OUT_DIR / f"rust_eval_r{rnd_idx}"
            res = evaluator.evaluate_gp_batch_rust(
                str(gz_paths[cid]), missing, run_outer_test=False, max_workers=2, timeout_s=20
            )
            rust_eval_calls += 1
            rust_eval_candidates += len(missing)
            for x, r_out in zip(missing, res):
                eval_cache[(cid, x["program_sha256"])] = r_out
        return [eval_cache[(cid, x["program_sha256"])] for x in cand_list]

    round_logs: List[Dict[str, Any]] = []
    promotions_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arm_names}
    eval_stats_by_arm: Dict[str, Dict[str, int]] = {
        a: {"evaluated": 0, "passing": 0, "failed": 0, "promotions": 0, "batch2_promotions": 0}
        for a in arm_names
    }

    for rnd_idx, r_seed in enumerate(ROUND_SEEDS, start=1):
        rnd_t0 = time.monotonic()
        rnd_promos = {a: 0 for a in arm_names}
        for i_ctx, r in enumerate(cohort_rows):
            c = r["context_sha256"]
            fam = r["family_id"]
            p = pools_base[c]
            cf = ctx_feats[c]
            lm = cf["lm48_z"].tolist()

            # True top-4 passing pool champions for this context
            f_eff = np.where(p["gate"], p["fitness"], 2.0)
            champ_order = np.argsort(f_eff, kind="mergesort")[:4]
            champ_prog = p["program"][champ_order]
            champ_proj = p["proj16"][champ_order]
            champ_f = p["fitness"][champ_order]
            champ_g = p["gate"][champ_order]

            for a in arm_names:
                parents = parents_by_ctx_arm[a][c]
                offset = (rnd_idx + i_ctx) % len(parents) if parents else 0
                rot_parents = parents[offset:] + parents[:offset]
                raw_props = evaluator.propose_flywheel_candidates(
                    rot_parents,
                    lm,
                    raw_bank,
                    fam,
                    n_proposals=N_PROPOSALS,
                    seed=r_seed + i_ctx,
                )
                # Deduplicate against arm's known pool PIDs and within proposal list
                seen_local = set()
                props = []
                for cand in raw_props:
                    pid = cand["program_sha256"]
                    if pid not in known_pids_arm[a][c] and pid not in seen_local:
                        seen_local.add(pid)
                        props.append(cand)
                if not props:
                    continue

                prop_feats = [evaluator.featurize_dsl_412(x["dsl"]) for x in props]
                prop_prog = np.stack([x[0] for x in prop_feats]).astype(np.float32)
                prop_proj = np.stack([x[1] for x in prop_feats]).astype(np.float32)
                prop_sem24 = ch.compute_sem24_numpy(prop_prog)

                if a == "LIVE_GP_LEGACY_STATIC_4":
                    # Legacy run_fused_closed_loop.py: prepends p["program"][:4] (arbitrary slots 0..3)
                    cost_pred = predict_proposal_costs(
                        net,
                        device,
                        cf,
                        p["program"][:4],
                        p["proj16"][:4],
                        p["fitness"][:4],
                        p["gate"][:4],
                        prop_prog,
                        prop_proj,
                    )
                    sel_idx = np.argsort(cost_pred, kind="mergesort")[:BUDGET_EVAL].tolist()
                    chosen = [props[j] for j in sel_idx]
                    results = evaluate_candidates_cached(c, chosen, rnd_idx)
                    batch2_flags = [False] * len(results)

                elif a == "LIVE_GP_CHAMPION_COND_STATIC_4":
                    # Champion-conditioned one-shot top-4: prepends true top-4 passing pool champions
                    cost_pred = predict_proposal_costs(
                        net,
                        device,
                        cf,
                        champ_prog,
                        champ_proj,
                        champ_f,
                        champ_g,
                        prop_prog,
                        prop_proj,
                    )
                    sel_idx = np.argsort(cost_pred, kind="mergesort")[:BUDGET_EVAL].tolist()
                    chosen = [props[j] for j in sel_idx]
                    results = evaluate_candidates_cached(c, chosen, rnd_idx)
                    batch2_flags = [False] * len(results)

                else:
                    # LIVE_GP_CAUSAL_2PLUS2_FLY:
                    # Batch 1 (2 live Rust CV evaluations): top-2 under champion-conditioned critic + structural complexity prior
                    cost_s1 = predict_proposal_costs(
                        net,
                        device,
                        cf,
                        champ_prog,
                        champ_proj,
                        champ_f,
                        champ_g,
                        prop_prog,
                        prop_proj,
                    )
                    # Mild complexity regularization favoring compact valid trees in Batch 1
                    dsl_lens = np.array([len(x["dsl"]) / 160.0 for x in props], dtype=np.float64)
                    score_s1 = -cost_s1 - 0.03 * dsl_lens
                    idx_b1 = np.argsort(-score_s1, kind="mergesort")[:2].tolist()
                    chosen_b1 = [props[j] for j in idx_b1]
                    res_b1 = evaluate_candidates_cached(c, chosen_b1, rnd_idx)

                    if len(props) <= 2:
                        chosen = chosen_b1
                        results = res_b1
                        batch2_flags = [False] * len(results)
                    else:
                        # Batch 2 (2 live Rust CV evaluations): condition on Batch 1's true live Rust CV outcomes!
                        b1_f = np.array(
                            [float(r_b["fitness"]) if (r_b["status"] == "completed" and r_b.get("cv_gate_passed")) else 2.0 for r_b in res_b1],
                            dtype=np.float32,
                        )
                        b1_g = np.array(
                            [bool(r_b["status"] == "completed" and r_b.get("cv_gate_passed")) for r_b in res_b1],
                            dtype=bool,
                        )
                        # Combine top-2 pool champions + 2 newly evaluated Batch-1 ASTs as the 4-token trajectory history
                        hist_prog_b2 = np.concatenate([champ_prog[:2], prop_prog[idx_b1]], axis=0)
                        hist_proj_b2 = np.concatenate([champ_proj[:2], prop_proj[idx_b1]], axis=0)
                        hist_f_b2 = np.concatenate([champ_f[:2], b1_f], axis=0)
                        hist_g_b2 = np.concatenate([champ_g[:2], b1_g], axis=0)

                        cost_s2 = predict_proposal_costs(
                            net,
                            device,
                            cf,
                            hist_prog_b2,
                            hist_proj_b2,
                            hist_f_b2,
                            hist_g_b2,
                            prop_prog,
                            prop_proj,
                        )
                        # Structural distance in 24D sem24 + 16D proj16 from Batch 1
                        b1_sem = prop_sem24[idx_b1]
                        dist_sem = np.min(np.linalg.norm(prop_sem24[:, None, :] - b1_sem[None, :, :], axis=-1), axis=-1)
                        inc_now = incumbent_fitness[a][c]
                        best_b1_f = float(np.min(b1_f))

                        if best_b1_f < inc_now + 0.05:
                            # Batch 1 found a strong/near-promoting subtree: exploit neighborhood of best Batch-1 AST + diversity
                            best_local = idx_b1[int(np.argmin(b1_f))]
                            dist_win = np.linalg.norm(prop_sem24 - prop_sem24[best_local : best_local + 1], axis=-1)
                            sim_win = np.exp(-2.0 * dist_win)
                            score_s2 = -cost_s2 + 0.15 * sim_win + 0.08 * dist_sem
                        else:
                            # Batch 1 stalled or failed CV gate: diversify away from Batch 1's structural equivalence class
                            score_s2 = -cost_s2 + 0.25 * dist_sem - 0.02 * dsl_lens

                        for j in idx_b1:
                            score_s2[j] = -1e18
                        idx_b2 = np.argsort(-score_s2, kind="mergesort")[: min(2, len(props) - 2)].tolist()
                        chosen_b2 = [props[j] for j in idx_b2]
                        res_b2 = evaluate_candidates_cached(c, chosen_b2, rnd_idx)

                        chosen = chosen_b1 + chosen_b2
                        results = res_b1 + res_b2
                        batch2_flags = [False] * len(res_b1) + [True] * len(res_b2)

                # Record evaluations and check strict promotions against arm's incumbent
                for cand_obj, r_out, is_b2 in zip(chosen, results, batch2_flags):
                    pid = cand_obj["program_sha256"]
                    known_pids_arm[a][c].add(pid)
                    eval_stats_by_arm[a]["evaluated"] += 1
                    passed = bool(r_out.get("status") == "completed" and r_out.get("cv_gate_passed") is True)
                    if passed:
                        eval_stats_by_arm[a]["passing"] += 1
                        score = float(r_out["fitness"])
                        inc_before = incumbent_fitness[a][c]
                        if score < inc_before - 1e-6:
                            incumbent_fitness[a][c] = score
                            eval_stats_by_arm[a]["promotions"] += 1
                            if is_b2:
                                eval_stats_by_arm[a]["batch2_promotions"] += 1
                            rnd_promos[a] += 1
                            feat_c, proj_c = evaluator.featurize_dsl_412(cand_obj["dsl"])
                            # Prepend newly promoted champion to arm's parent bank for Round 2!
                            parents_by_ctx_arm[a][c].insert(
                                0,
                                {
                                    "context_sha256": c,
                                    "family_id": fam,
                                    "dl_split": "DL-Train",
                                    "dsl": cand_obj["dsl"],
                                    "program_sha256": pid,
                                    "feat412": feat_c.tolist(),
                                    "proj16": proj_c.tolist(),
                                    "landmark_summary": lm[:12],
                                },
                            )
                            promotions_by_arm[a].append(
                                {
                                    "round": rnd_idx,
                                    "context_sha256": c,
                                    "family_id": fam,
                                    "program_sha256": pid,
                                    "dsl": cand_obj["dsl"],
                                    "old_utility": -inc_before,
                                    "new_utility": -score,
                                    "utility_gain": (-score) - (-inc_before),
                                    "from_batch2_reveal": is_b2,
                                }
                            )
                    else:
                        eval_stats_by_arm[a]["failed"] += 1

        round_logs.append(
            {
                "round": rnd_idx,
                "wall_sec": time.monotonic() - rnd_t0,
                "promotions_this_round": rnd_promos,
                "mean_incumbent_utility": {
                    a: float(np.mean([-incumbent_fitness[a][c] for c in cohort_cids]))
                    for a in arm_names
                },
            }
        )

    guard.verify_dope_frozen_unchanged(net, before)
    shutil.rmtree(OUT_DIR / "rust_eval_r1", ignore_errors=True)
    shutil.rmtree(OUT_DIR / "rust_eval_r2", ignore_errors=True)

    # Compute paired family bootstrap 95% CIs on final incumbent utility gain over initial incumbent
    fam_utilities: Dict[str, np.ndarray] = {}
    for a in arm_names:
        fam_utilities[a] = np.array([-incumbent_fitness[a][c] for c in cohort_cids], dtype=np.float64)
    init_utilities = np.array([-initial_incumbent_fitness[c] for c in cohort_cids], dtype=np.float64)

    vs_initial = {
        a: ch.paired_family_bootstrap(
            fam_utilities[a], init_utilities, n_draws=BOOT_DRAWS, seed=ch.stream_seed(BOOT_SEED, "vs_init", a)
        )
        for a in arm_names
    }
    vs_legacy = {
        a: ch.paired_family_bootstrap(
            fam_utilities[a],
            fam_utilities["LIVE_GP_LEGACY_STATIC_4"],
            n_draws=BOOT_DRAWS,
            seed=ch.stream_seed(BOOT_SEED, "vs_legacy", a),
        )
        for a in arm_names
        if a != "LIVE_GP_LEGACY_STATIC_4"
    }

    out_payload = {
        "wave": "H255_PART_B",
        "title": "Live Rust GP 2+2 Closed-Loop Flywheel on 32 Stratified DL-Train Contexts",
        "elapsed_sec": time.monotonic() - t0,
        "frozen_h252_hashes": frozen_hashes,
        "evaluator_binary_sha256": evaluator.WORKER_SHA256,
        "dope_sha256": before.composite_sha256,
        "dope_unchanged": True,
        "outer_test_evaluated": False,
        "sealed_meta_test_read": False,
        "n_contexts": len(cohort_cids),
        "n_families": len({r["family_id"] for r in cohort_rows}),
        "n_rounds": len(ROUND_SEEDS),
        "proposals_per_context_per_round": N_PROPOSALS,
        "rust_evaluations_per_context_per_round": BUDGET_EVAL,
        "unique_rust_batch_calls": rust_eval_calls,
        "unique_rust_candidates_evaluated": rust_eval_candidates,
        "initial_mean_incumbent_utility": float(np.mean(init_utilities)),
        "final_mean_incumbent_utility": {a: float(np.mean(fam_utilities[a])) for a in arm_names},
        "eval_stats_by_arm": eval_stats_by_arm,
        "round_logs": round_logs,
        "vs_initial_incumbent": vs_initial,
        "vs_legacy_static_4": vs_legacy,
        "promotions_by_arm": promotions_by_arm,
    }
    ch.save_json(OUT_DIR / "part_b_metrics.json", out_payload)
    print(json.dumps({
        "status": "PART_B_OK",
        "elapsed_sec": round(out_payload["elapsed_sec"], 2),
        "eval_stats_by_arm": eval_stats_by_arm,
        "initial_mean_incumbent_utility": out_payload["initial_mean_incumbent_utility"],
        "final_mean_incumbent_utility": out_payload["final_mean_incumbent_utility"],
    }, indent=2))


if __name__ == "__main__":
    main()
