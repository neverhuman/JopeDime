#!/usr/bin/env python3
"""H256 Thrust 2 (Gates T2 & T3): Batched Multi-Subsample Deep Learning Seeding (`1x100% + 8x80% + 2xcol-jitter`)
-> Generation 0 -> 3-Generation Live Rust GP Evolution (`Gen 0..3`) + Leak-Free 2-Fold OOF Pareto Stacking.

Implements the complete Medium / High compute evolutionary search loop on top of H256:
  1. Batched Subsample Encoding (`B = 11` views per dataset):
     - View 0: `100%` of training rows encoded through `FusedLandmarkWorkspace` + locked DOPE +
       Stage-1 Full-S3 7-Split Regression Encoder + Stage-2 `PreCondenseResidualBridge`.
     - Views 1..8: `8` random fold-stratified `80%` subsamples of training rows.
     - Views 9..10: `2` semi-synthetic column-augmented `80%` subsamples.
  2. Batched Deep Learning Inference to Propose Generation 0 (`Gen 0`, `8` GP trees):
     - Evaluates all 11 dataset views simultaneously (`[11, M_bank]`) against the family-disjoint
       GP AST bank (`1,219` DL-Train champions + `90` novel qualified winners + per-view Landmark-V2
       synthesized pipelines).
     - Selects `8` structurally diverse Generation-0 GP trees combining View-0 (`100%`) Top-1,
       `8x80%` Subsample Borda Consensus, per-subsample specialist champions, and Landmark-V2
       regime-matched trees.
  3. 3-Generation Live Rust GP Evolution (`Gen 0` -> `Gen 1` -> `Gen 2` -> `Gen 3`):
     - Runs exact 2-fold `JTT1` inner CV (`bootstrap.s1main --fit-s1` via `training_cv_envelope`)
       on `Gen 0`.
     - Across Generations `1, 2, 3`:
       * Generates `20` candidate child GP trees per generation from the top fertile parents using
         `EditZero` typed AST mutations (`Parameter`, `ReplaceHead`, `InsertStage`, `DeleteStage`,
         `Crossover`) + subsample-view Landmark-V2 synthesis.
       * Pre-screens all `20` children in a single batched neural pass using `H256`
         `PreCondenseResidualBridge.forward_causal_trajectory_residual` conditioned on the true
         observed 2-fold CV fitnesses & gates from `Gen 0 .. g-1` + UCB semantic diversity (`sem24`).
       * Evaluates the Top-5 neural-screened children per generation in live Rust (`bootstrap.s1main --fit-s1`),
         retaining exact Fold-0 and Fold-1 out-of-fold predictions.
  4. Post-Evolution Inner-CV Selection & Leak-Free 2-Fold OOF Pareto Stacking:
     - Tracks progressive utility & Elo lift at `Gen 0`, `Gen 1`, `Gen 2`, `Gen 3`,
       `H256_MEDIUM_BATCH_EVO3` (unified 11-view batch pool + 3-gen GP champion), and
       `H256_HIGH_BATCH_EVO3_STACK3` (leak-free 2-fold OOF simplex stack of Top-3 diverse GP trees).
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SELFPLAY_DIR = HERE.parent
H252_DIR = SELFPLAY_DIR / "h252_closed_loop_flywheel"
H255_DIR = SELFPLAY_DIR / "h255_causal_2plus2_live_gp"
H256_DIR = SELFPLAY_DIR / "h256_pre_condense_residual_flywheel"
MICRO = Path("/home/ubuntu/JopeDimeMicro")
H256_OUT_DIR = Path("/home/ubuntu/jope-data/h256")
OUT_DIR = Path("/home/ubuntu/jope-data/h257")
CACHE_DIR = OUT_DIR / "cache"
EVO_SCRATCH_DIR = OUT_DIR / "rust_evo_scratch"

sys.path.insert(0, str(MICRO / "python"))
sys.path.insert(0, str(SELFPLAY_DIR))
sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(H256_DIR))
sys.path.insert(0, str(HERE))

import rust_evaluator as evaluator  # noqa: E402
from automatic_promotion import training_cv_envelope  # noqa: E402
import common_h252 as ch  # noqa: E402
import pretrain_full_s3_7split_bridge as p_s3  # noqa: E402
import run_h256_flywheel_ablation as r256  # noqa: E402
import train_and_export_h256_flagship_production as prod  # noqa: E402
import semi_synthetic_columns_and_views as ssa  # noqa: E402
from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402


def load_enriched_gp_champion_bank(
    train_rows: List[Dict[str, Any]],
    raw_cache: Dict[str, Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Loads all explicit GP DSL programs strictly from DL-Train (`champions.json` + `winners.json`),
    attaching precomputed `feat412`, `proj16`, `sem24`, and 12D `landmark_summary`.
    """
    bank_cache_path = CACHE_DIR / "gp_champion_bank_enriched.pt"
    if bank_cache_path.exists():
        return torch.load(bank_cache_path, map_location="cpu", weights_only=False)

    train_fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train_rows}
    raw_champs = json.loads((ch.INPUT_DIR / "champions.json").read_text())
    raw_winners = json.loads((ch.INPUT_DIR / "winners.json").read_text()).get(
        "all_90_qualified_novel_winner_records", []
    )

    seen_key = set()
    bank: List[Dict[str, Any]] = []

    for rec in list(raw_champs) + list(raw_winners):
        cid = rec.get("context_sha256", "")
        if cid not in train_fam_by_cid:
            continue
        dsl = rec.get("dsl", "").strip()
        if not dsl.startswith("(pipe X"):
            continue
        psha = hashlib.sha256(dsl.encode()).hexdigest()
        key = (cid, psha)
        if key in seen_key:
            continue
        seen_key.add(key)
        f412, p16 = evaluator.featurize_dsl_412(dsl)
        s24 = ch.compute_sem24_numpy(f412[None, :])[0]
        lm_sum = (
            raw_cache[cid]["lm48_z"][:12].astype(np.float32).tolist()
            if cid in raw_cache
            else [0.0] * 12
        )
        bank.append(
            {
                "context_sha256": cid,
                "family_id": train_fam_by_cid[cid],
                "dl_split": "DL-Train",
                "dsl": dsl,
                "program_sha256": psha,
                "feat412": f412.astype(np.float32),
                "proj16": p16.astype(np.float32),
                "sem24": s24.astype(np.float32),
                "landmark_summary": lm_sum,
            }
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(bank, bank_cache_path)
    return bank


def _eval_single_gp_with_oof_worker(args: Tuple[int, str, str, str, List[str], float]) -> Dict[str, Any]:
    slot_id, env_jtt1_path, dsl, operator, parents, timeout_s = args
    evaluator.GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.8)
    EVO_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    slot_dir = EVO_SCRATCH_DIR / f"slot-{os.getpid()}-{threading.get_ident()}-{slot_id}"
    slot_dir.mkdir(parents=True, exist_ok=True)

    prog_sha = hashlib.sha256(dsl.encode()).hexdigest()
    cand_obj = {
        "dsl": dsl,
        "program_sha256": prog_sha,
        "birth_seq": 0,
        "parents": list(parents),
        "operator": operator,
    }
    cand_path = slot_dir / "candidate.json"
    out_path = slot_dir / "eval_s1.json"
    cand_path.write_text(json.dumps(cand_obj, sort_keys=True))
    out_path.unlink(missing_ok=True)

    env = {
        "PATH": "/usr/bin:/bin",
        "RAYON_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "TMPDIR": str(slot_dir),
        "HOME": str(slot_dir),
    }
    t0 = time.monotonic()
    try:
        rc = subprocess.run(
            [
                "nice",
                "-n",
                "10",
                str(evaluator.WORKER_BIN),
                "--fit-s1",
                str(env_jtt1_path),
                str(cand_path),
                str(out_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            env=env,
        ).returncode
    except subprocess.TimeoutExpired:
        rc = -1

    wall_s = round(time.monotonic() - t0, 4)
    f412, proj16 = evaluator.featurize_dsl_412(dsl)
    sem24 = ch.compute_sem24_numpy(f412[None, :])[0]

    if rc != 0 or not out_path.exists():
        return {
            "dsl": dsl,
            "program_sha256": prog_sha,
            "operator": operator,
            "parents": list(parents),
            "status": "failed",
            "fitness": 100.0,
            "utility": -2.0,
            "logratio": 0.0,
            "cv_geomean_rmse_ratio": 1e9,
            "cv_gate_passed": False,
            "wall_s": wall_s,
            "feat412": f412,
            "proj16": proj16.astype(np.float32),
            "sem24": sem24,
            "fold0_preds": None,
            "fold1_preds": None,
            "fold0_base_rmse": None,
            "fold1_base_rmse": None,
        }

    try:
        doc = json.loads(out_path.read_text())
        inner_cv = doc.get("inner_cv") or []
        s1_res = evaluator.compute_s1_metrics_from_inner_cv(dsl, inner_cv)
        if s1_res["status"] == "completed" and len(inner_cv) == 2:
            f0_preds = np.asarray(inner_cv[0].get("predictions") or [], dtype=np.float64)
            f1_preds = np.asarray(inner_cv[1].get("predictions") or [], dtype=np.float64)
            b0 = float((inner_cv[0].get("baseline_metrics") or {}).get("rmse", 1.0))
            b1 = float((inner_cv[1].get("baseline_metrics") or {}).get("rmse", 1.0))
        else:
            f0_preds, f1_preds, b0, b1 = None, None, None, None
    except Exception:
        s1_res = {
            "status": "failed",
            "fitness": 100.0,
            "logratio": 0.0,
            "cv_geomean_rmse_ratio": 1e9,
            "cv_gate_passed": False,
        }
        f0_preds, f1_preds, b0, b1 = None, None, None, None

    gate = bool(s1_res.get("cv_gate_passed", False))
    fit_val = float(s1_res.get("fitness", 100.0))
    util_val = float(-fit_val) if gate else -2.0

    s1_res.update(
        {
            "dsl": dsl,
            "program_sha256": prog_sha,
            "operator": operator,
            "parents": list(parents),
            "utility": util_val,
            "wall_s": wall_s,
            "feat412": f412,
            "proj16": proj16.astype(np.float32),
            "sem24": sem24,
            "fold0_preds": f0_preds,
            "fold1_preds": f1_preds,
            "fold0_base_rmse": b0,
            "fold1_base_rmse": b1,
        }
    )
    return s1_res


def evaluate_gp_batch_with_oof_rust(
    env_jtt1_path: Path,
    candidates: List[Dict[str, Any]],
    max_workers: int = 6,
    timeout_s: float = 18.0,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    eff_workers = evaluator.GLOBAL_CPU_GUARD.adaptive_worker_count(max_workers)
    tasks = [
        (
            i,
            str(env_jtt1_path),
            c["dsl"],
            c.get("operator", "edit_zero"),
            c.get("parents", []),
            timeout_s,
        )
        for i, c in enumerate(candidates)
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(eff_workers, len(tasks))) as ex:
        for res in ex.map(_eval_single_gp_with_oof_worker, tasks):
            results.append(res)
    return results


def parse_jtt1_fold_targets(jtt1_path: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Extracts exact `(y_fold0, y_fold1, base_rmse_fold0, base_rmse_fold1)` from a 2-fold JTT1 file."""
    raw = jtt1_path.read_bytes()
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17)
    mat = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(
        nt, width + 1
    )
    y = mat[:, -1]
    y0 = y[folds == 0].copy()
    y1 = y[folds == 1].copy()
    b0 = float(np.sqrt(np.mean((y0 - float(np.mean(y1))) ** 2)))
    b1 = float(np.sqrt(np.mean((y1 - float(np.mean(y0))) ** 2)))
    return y0, y1, b0, b1


def compute_leak_free_2fold_oof_stack(
    passing_evals: List[Dict[str, Any]],
    y_fold0: np.ndarray,
    y_fold1: np.ndarray,
    base_rmse0: float,
    base_rmse1: float,
    max_models: int = 3,
) -> Optional[Dict[str, Any]]:
    """Constructs a strictly leak-free 2-fold out-of-fold simplex stack over the Top-`max_models`
    structurally diverse passing GP trees:
      - To predict Fold 0: fits simplex weights `w^(0)` (`w >= 0, sum(w) == 1`) exclusively on
        Fold-1 OOF predictions (`fold1_preds` vs `y_fold1`).
      - To predict Fold 1: fits simplex weights `w^(1)` (`w >= 0, sum(w) == 1`) exclusively on
        Fold-0 OOF predictions (`fold0_preds` vs `y_fold0`).
    Computes exact `jope.s1-two-fold/v1` RMSE ratios, `logratio`, `fitness`, and `utility`.
    """
    valid = [
        e
        for e in passing_evals
        if e.get("status") == "completed"
        and e.get("cv_gate_passed") is True
        and e.get("fold0_preds") is not None
        and e.get("fold1_preds") is not None
        and len(e["fold0_preds"]) == len(y_fold0)
        and len(e["fold1_preds"]) == len(y_fold1)
    ]
    if len(valid) < 2:
        return None

    valid_sorted = sorted(valid, key=lambda x: (float(x["fitness"]), x["program_sha256"]))
    # Select up to `max_models` diverse top trees
    chosen = [valid_sorted[0]]
    for cand in valid_sorted[1:]:
        if len(chosen) >= max_models:
            break
        # Check prediction correlation / diversity so we don't stack identical trees
        p0_c = cand["fold0_preds"]
        is_dup = any(
            np.max(np.abs(p0_c - prev["fold0_preds"])) < 1e-6 * max(base_rmse0, 1e-6)
            for prev in chosen
        )
        if not is_dup:
            chosen.append(cand)
    if len(chosen) < 2:
        return None

    k_m = len(chosen)
    P0 = np.column_stack([c["fold0_preds"] for c in chosen])  # [N0, K]
    P1 = np.column_stack([c["fold1_preds"] for c in chosen])  # [N1, K]

    # Grid search over simplex weights w in Delta^{K-1} (step 0.05) + shrinkage toward inverse-fitness prior
    f_vals = np.array([float(c["fitness"]) for c in chosen], dtype=np.float64)
    prior_w = np.exp((np.min(f_vals) - f_vals) / 0.15)
    prior_w /= np.sum(prior_w)

    simplex_grid = []
    if k_m == 2:
        for i in range(21):
            w0 = i * 0.05
            simplex_grid.append(np.array([w0, 1.0 - w0], dtype=np.float64))
    else:
        for i in range(21):
            for j in range(21 - i):
                w0 = i * 0.05
                w1 = j * 0.05
                w2 = max(0.0, 1.0 - w0 - w1)
                simplex_grid.append(np.array([w0, w1, w2], dtype=np.float64))
    simplex_grid.append(prior_w)

    # Fit w^(0) strictly on Fold 1 (`P1`, `y_fold1`) with L2 shrinkage toward `prior_w`
    best_w_for_fold0 = prior_w
    best_loss_on_fold1 = 1e30
    for w in simplex_grid:
        pred1 = P1 @ w
        mse1 = float(np.mean((pred1 - y_fold1) ** 2)) / max(base_rmse1 ** 2, 1e-12)
        reg = 0.02 * float(np.sum((w - prior_w) ** 2))
        if mse1 + reg < best_loss_on_fold1:
            best_loss_on_fold1 = mse1 + reg
            best_w_for_fold0 = w

    # Fit w^(1) strictly on Fold 0 (`P0`, `y_fold0`) with L2 shrinkage toward `prior_w`
    best_w_for_fold1 = prior_w
    best_loss_on_fold0 = 1e30
    for w in simplex_grid:
        pred0 = P0 @ w
        mse0 = float(np.mean((pred0 - y_fold0) ** 2)) / max(base_rmse0 ** 2, 1e-12)
        reg = 0.02 * float(np.sum((w - prior_w) ** 2))
        if mse0 + reg < best_loss_on_fold0:
            best_loss_on_fold0 = mse0 + reg
            best_w_for_fold1 = w

    oof_pred0 = P0 @ best_w_for_fold0
    oof_pred1 = P1 @ best_w_for_fold1
    rmse0 = float(np.sqrt(np.mean((oof_pred0 - y_fold0) ** 2)))
    rmse1 = float(np.sqrt(np.mean((oof_pred1 - y_fold1) ** 2)))
    r0 = max(rmse0 / max(base_rmse0, 1e-12), 1e-12)
    r1 = max(rmse1 / max(base_rmse1, 1e-12), 1e-12)

    lr0, lr1 = math.log(r0), math.log(r1)
    mean_lr = 0.5 * (lr0 + lr1)
    tail_lr = max(lr0, lr1)
    sd_lr = 0.5 * abs(lr0 - lr1)
    geomean_ratio = math.exp(mean_lr)
    wins = (1 if r0 <= 1.0 else 0) + (1 if r1 <= 1.0 else 0)
    gate = (wins == 2) and (geomean_ratio <= 0.90)

    # Weighted average AST complexity of active ensemble members
    w_mean = 0.5 * (best_w_for_fold0 + best_w_for_fold1)
    comp_weighted = sum(
        float(w_mean[idx]) * float(chosen[idx]["dsl"].count("(")) for idx in range(k_m)
    )
    if gate:
        fitness = 0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp_weighted
        utility = -fitness
    else:
        fitness = 100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0)
        utility = -2.0

    return {
        "status": "completed",
        "k_models": k_m,
        "member_shas": [c["program_sha256"] for c in chosen],
        "member_dsls": [c["dsl"] for c in chosen],
        "w_fold0": best_w_for_fold0.tolist(),
        "w_fold1": best_w_for_fold1.tolist(),
        "rmse_fold0": rmse0,
        "rmse_fold1": rmse1,
        "cv_geomean_rmse_ratio": geomean_ratio,
        "logratio": mean_lr,
        "cv_gate_passed": bool(gate),
        "fitness": float(fitness),
        "utility": float(utility),
    }


def propose_gen0_from_11view_batch(
    cid: str,
    family_id: str,
    views_ctx: Dict[str, np.ndarray],
    gp_bank: List[Dict[str, Any]],
    f_model: PreCondenseResidualBridge,
    g_model: PreCondenseResidualBridge,
    sc_std: FoldScaler,
    sc_s3: FoldScaler,
    device: torch.device,
    n_gen0: int = 8,
) -> List[Dict[str, Any]]:
    """Uses the 11-view subsample batch (`1x100% + 8x80% + 2xcol-jitter`) to propose `n_gen0`
    Generation-0 GP trees in a single batched neural forward pass!
    """
    eligible = [b for b in gp_bank if b["family_id"] != family_id]
    lm_11 = views_ctx["lm48_z"]  # [11, 48]

    # Step 1: Across all 11 views, retrieve the top-24 Landmark-V2 nearest-neighbor GP champions
    bank_lm12 = np.asarray([b["landmark_summary"] for b in eligible], dtype=np.float64)  # [M_el, 12]
    target_lm12 = lm_11[:, :12].astype(np.float64)  # [11, 12]
    dists_11xM = np.sum((target_lm12[:, None, :] - bank_lm12[None, :, :]) ** 2, axis=-1)  # [11, M_el]

    cand_indices_set = set()
    for b_v in range(len(lm_11)):
        top_b = np.argsort(dists_11xM[b_v])[:6].tolist()
        cand_indices_set.update(top_b)
    mean_dists = np.mean(dists_11xM, axis=0)
    cand_indices_set.update(np.argsort(mean_dists)[:16].tolist())

    shortlist = [eligible[i] for i in sorted(cand_indices_set)]

    # Also add Landmark-V2 grammar proposals generated from each subsample view's `lm48_z[b]`
    seen_dsl = {c["dsl"] for c in shortlist}
    for b_v in range(min(6, len(lm_11))):
        props_b = evaluator.propose_flywheel_candidates(
            parent_programs=shortlist[:4],
            landmark_v2_48=lm_11[b_v],
            champion_bank=eligible[:64],
            target_family_id=family_id,
            n_proposals=4,
            seed=prod.stream_seed("gen0_view_prop", cid, b_v),
        )
        for p in props_b:
            if p["dsl"] not in seen_dsl:
                seen_dsl.add(p["dsl"])
                f412, p16 = evaluator.featurize_dsl_412(p["dsl"])
                s24 = ch.compute_sem24_numpy(f412[None, :])[0]
                shortlist.append(
                    {
                        "dsl": p["dsl"],
                        "program_sha256": p["program_sha256"],
                        "operator": f"batch_view{b_v}_{p['operator']}",
                        "parents": p.get("parents", []),
                        "feat412": f412.astype(np.float32),
                        "proj16": p16.astype(np.float32),
                        "sem24": s24.astype(np.float32),
                    }
                )

    # Step 2: Run a SINGLE batched forward pass [11, M_short] through the H256 Stage-2 Neural Bridges
    m_short = len(shortlist)
    n_views = len(lm_11)
    prog_mat = np.stack([c["feat412"] for c in shortlist], axis=0).astype(np.float32)  # [M, 412]
    sem_mat = np.stack([c["sem24"] for c in shortlist], axis=0).astype(np.float32)  # [M, 24]
    proj_mat = np.stack([c["proj16"] for c in shortlist], axis=0).astype(np.float32)  # [M, 16]

    # Construct 32D cand_q representation for `PreCondenseResidualBridge`
    cand_q_32 = np.zeros((m_short, 32), dtype=np.float32)
    cand_q_32[:, :16] = proj_mat
    cand_q_32[:, 16:32] = sem_mat[:, :16]

    cand_q_batch = torch.as_tensor(cand_q_32, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, m_short, 32)
    sem24_batch = torch.as_tensor(sem_mat, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, m_short, 24)
    zeros_bm = torch.zeros((n_views, m_short), dtype=torch.float32, device=device)

    ct_std = ssa.transform_context_batch(
        sc_std,
        views_ctx["dope629"],
        views_ctx["hidden256"],
        views_ctx["col_lm_64x17"],
        views_ctx["col_mask_64"],
        device=device,
    )
    ct_s3 = ssa.transform_context_batch(
        sc_s3,
        views_ctx["dope629"],
        views_ctx["hidden256"],
        views_ctx["col_lm_64x17"],
        views_ctx["col_mask_64"],
        device=device,
    )
    with torch.no_grad():
        dp_f_11xM, _, _ = f_model.forward_prior_and_context(
            ct_std, cand_q_batch, sem24_batch, zeros_bm, zeros_bm
        )
        dp_g_11xM, _, _ = g_model.forward_prior_and_context(
            ct_s3, cand_q_batch, sem24_batch, zeros_bm, zeros_bm
        )
    neural_cost_11xM = 0.50 * dp_f_11xM.cpu().numpy() + 0.50 * dp_g_11xM.cpu().numpy()

    # Combine neural bridge cost with Landmark-V2 affinity per view
    lm_affinity_11xM = np.zeros((n_views, m_short), dtype=np.float64)
    for j, c in enumerate(shortlist):
        if "landmark_summary" in c:
            c_lm = np.asarray(c["landmark_summary"], dtype=np.float64)
            lm_affinity_11xM[:, j] = np.sqrt(np.sum((target_lm12 - c_lm[None, :]) ** 2, axis=-1))
        else:
            lm_affinity_11xM[:, j] = np.median(dists_11xM) ** 0.5

    total_score_11xM = -prod.standardize_scores_per_query(neural_cost_11xM) - 0.35 * prod.standardize_scores_per_query(lm_affinity_11xM)

    # Select 8 diverse Generation-0 GP trees across the 5 orthogonal batch roles:
    #   Role 1 (Slot 0): Full-Data Anchor (View 0 = 100% training data)
    #   Role 2 (Slot 1): Risk-Averse Subsample Consensus (Mean - 0.35 * Epistemic Std across Views 0..8)
    #   Role 3 (Slots 2..4): Mode-Seeking Subsample Basin Specialists across distinct estimator families
    #   Role 4 (Slot 5): Semi-Synthetic Column-Robust Specialist (Views 9..10 with synthetic noise/collinear/interaction cols)
    #   Role 5 (Slots 6..7): Epistemic UCB + Structural (sem24) Diversity Explorers (Mean + 0.50 * Std + 0.30 * min_d_sem24)
    chosen_indices: List[int] = []

    def _try_add(idx: int) -> bool:
        if idx not in chosen_indices and len(chosen_indices) < n_gen0:
            chosen_indices.append(int(idx))
            return True
        return False

    # Role 1: Full-Data Anchor (View 0)
    _try_add(int(np.argmax(total_score_11xM[0])))

    # Role 2: Risk-Averse Subsample Consensus across Views 0..8
    sub_views = total_score_11xM[: min(9, n_views)]
    w_sub = np.full(len(sub_views), 0.65 / max(len(sub_views) - 1, 1), dtype=np.float64)
    w_sub[0] = 0.35
    mu_sub = np.tensordot(sub_views, w_sub, axes=([0], [0]))
    sigma_sub = np.std(total_score_11xM[1 : min(9, n_views)], axis=0) if n_views > 2 else np.zeros(m_short)
    risk_averse_consensus = mu_sub - 0.35 * sigma_sub
    for idx in np.argsort(-risk_averse_consensus):
        if _try_add(int(idx)):
            break

    # Role 3: Mode-Seeking Subsample Basin Specialists across distinct terminal head families
    def _head_family(dsl_str: str) -> str:
        for fam_tok in ("histgbm", "ridge_gcv", "krr", "gpr", "elasticnet", "pls", "knn"):
            if fam_tok in dsl_str:
                return fam_tok
        return "other"

    represented_fams = {_head_family(shortlist[i]["dsl"]) for i in chosen_indices}
    for b_v in range(1, min(9, n_views)):
        order_bv = np.argsort(-total_score_11xM[b_v])
        added_specialist = False
        for idx in order_bv[:6]:
            hf = _head_family(shortlist[int(idx)]["dsl"])
            if hf not in represented_fams:
                if _try_add(int(idx)):
                    represented_fams.add(hf)
                    added_specialist = True
                    break
        if not added_specialist and len(chosen_indices) < 5:
            _try_add(int(order_bv[0]))
        if len(chosen_indices) >= 5:
            break

    # Role 4: Semi-Synthetic Column-Robust Specialist (Views 9..10)
    if n_views >= 11:
        semisyn_mean = 0.5 * (total_score_11xM[9] + total_score_11xM[10])
        col_drop_penalty = np.maximum(0.0, total_score_11xM[0] - semisyn_mean)
        col_robust_score = semisyn_mean - 0.50 * col_drop_penalty
        for idx in np.argsort(-col_robust_score):
            if _try_add(int(idx)):
                break

    # Role 5: Epistemic UCB + Structural (sem24) Diversity Explorers
    w11 = np.full(n_views, 0.65 / max(n_views - 1, 1), dtype=np.float64)
    w11[0] = 0.35
    consensus_score = np.tensordot(total_score_11xM, w11, axes=([0], [0]))
    while len(chosen_indices) < min(n_gen0, m_short):
        best_next = -1
        best_acq = -1e18
        chosen_sem = sem_mat[chosen_indices]
        for j in range(m_short):
            if j in chosen_indices:
                continue
            min_d = float(np.min(np.linalg.norm(sem_mat[j : j + 1] - chosen_sem, axis=-1)))
            acq = float(consensus_score[j]) + 0.50 * float(sigma_sub[j]) + 0.30 * min_d
            if acq > best_acq:
                best_acq = acq
                best_next = j
        if best_next < 0:
            break
        _try_add(best_next)

    return [
        {
            "dsl": shortlist[i]["dsl"],
            "program_sha256": shortlist[i]["program_sha256"],
            "operator": shortlist[i].get("operator", "batch11_seed"),
            "parents": shortlist[i].get("parents", []),
        }
        for i in chosen_indices
    ]


def screen_child_proposals_with_causal_trajectory(
    proposals: List[Dict[str, Any]],
    evaluated_pop: List[Dict[str, Any]],
    views_ctx: Dict[str, np.ndarray],
    f_model: PreCondenseResidualBridge,
    g_model: PreCondenseResidualBridge,
    sc_std: FoldScaler,
    sc_s3: FoldScaler,
    device: torch.device,
    k_select: int = 5,
) -> List[Dict[str, Any]]:
    """Uses the H256 Stage-2 Neural Bridge (`forward_prior_and_context` +
    `forward_causal_trajectory_residual`) across the 11-view batch conditioned on the top-4
    observed 2-fold CV outcomes from `evaluated_pop` to select `k_select` diverse child GP trees.
    """
    if len(proposals) <= k_select:
        return proposals

    # Select top-4 evaluated trees as the 4 revealed trajectory tokens
    valid_hist = [e for e in evaluated_pop if e.get("status") == "completed"]
    if len(valid_hist) < 4:
        valid_hist = evaluated_pop[:4]
    valid_hist_sorted = sorted(
        valid_hist,
        key=lambda x: (
            0 if x.get("cv_gate_passed") else 1,
            float(x.get("fitness", 100.0)),
            x.get("program_sha256", ""),
        ),
    )[:4]
    while len(valid_hist_sorted) < 4:
        valid_hist_sorted.append(valid_hist_sorted[-1])

    hist_prog = np.stack([h["feat412"] for h in valid_hist_sorted], axis=0).astype(np.float32)
    hist_proj = np.stack([h["proj16"] for h in valid_hist_sorted], axis=0).astype(np.float32)
    hist_sem = np.stack([h["sem24"] for h in valid_hist_sorted], axis=0).astype(np.float32)
    hist_f = np.array(
        [
            float(h["fitness"]) if h.get("cv_gate_passed") else 2.0
            for h in valid_hist_sorted
        ],
        dtype=np.float32,
    )
    hist_lr = np.clip(
        np.array([float(h.get("logratio", 0.0)) for h in valid_hist_sorted], dtype=np.float32),
        -1.5,
        1.5,
    )
    hist_g = np.array([bool(h.get("cv_gate_passed", False)) for h in valid_hist_sorted], dtype=np.float32)
    y4_comb = 0.70 * (hist_f - float(np.mean(hist_f))) + 0.30 * (hist_lr - float(np.mean(hist_lr)))

    prop_feats = [evaluator.featurize_dsl_412(p["dsl"]) for p in proposals]
    prop_prog = np.stack([x[0] for x in prop_feats], axis=0).astype(np.float32)
    prop_proj = np.stack([x[1] for x in prop_feats], axis=0).astype(np.float32)
    prop_sem = ch.compute_sem24_numpy(prop_prog)

    all_sem = np.concatenate([hist_sem, prop_sem], axis=0)  # [4 + P, 24]
    all_proj = np.concatenate([hist_proj, prop_proj], axis=0)  # [4 + P, 16]
    m_all = len(all_sem)
    cand_q_32 = np.zeros((m_all, 32), dtype=np.float32)
    cand_q_32[:, :16] = all_proj
    cand_q_32[:, 16:32] = all_sem[:, :16]

    n_views = len(views_ctx["lm48_z"])
    cand_q_b = torch.as_tensor(cand_q_32, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, m_all, 32)
    sem24_b = torch.as_tensor(all_sem, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, m_all, 24)
    zeros_bm = torch.zeros((n_views, m_all), dtype=torch.float32, device=device)

    ct_std = ssa.transform_context_batch(
        sc_std,
        views_ctx["dope629"],
        views_ctx["hidden256"],
        views_ctx["col_lm_64x17"],
        views_ctx["col_mask_64"],
        device=device,
    )
    ct_s3 = ssa.transform_context_batch(
        sc_s3,
        views_ctx["dope629"],
        views_ctx["hidden256"],
        views_ctx["col_lm_64x17"],
        views_ctx["col_mask_64"],
        device=device,
    )

    rev_idx_b = torch.arange(4, dtype=torch.long, device=device).unsqueeze(0).expand(n_views, 4)
    rev_y_comb_b = torch.as_tensor(y4_comb, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, 4)
    rev_y_f_b = torch.as_tensor(hist_f, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, 4)
    rev_g_b = torch.as_tensor(hist_g, dtype=torch.float32, device=device).unsqueeze(0).expand(n_views, 4)

    with torch.no_grad():
        dp_f, attn_f, bilin_f = f_model.forward_prior_and_context(
            ct_std, cand_q_b, sem24_b, zeros_bm, zeros_bm
        )
        dt_f = f_model.forward_causal_trajectory_residual(
            attn_f, bilin_f, cand_q_b, sem24_b, 0.03 * dp_f, rev_idx_b, rev_y_comb_b, rev_y_f_b, rev_g_b
        )
        dp_g, attn_g, bilin_g = g_model.forward_prior_and_context(
            ct_s3, cand_q_b, sem24_b, zeros_bm, zeros_bm
        )
        dt_g = g_model.forward_causal_trajectory_residual(
            attn_g, bilin_g, cand_q_b, sem24_b, 0.03 * dp_g, rev_idx_b, rev_y_comb_b, rev_y_f_b, rev_g_b
        )

    pred_cost_11xP = (
        0.50 * (dp_f[:, 4:] + dt_f[:, 4:]).cpu().numpy()
        + 0.50 * (dp_g[:, 4:] + dt_g[:, 4:]).cpu().numpy()
    )
    w11 = np.full(n_views, 0.65 / max(n_views - 1, 1), dtype=np.float64)
    w11[0] = 0.35
    mean_neural_score = np.tensordot(
        -prod.standardize_scores_per_query(pred_cost_11xP), w11, axes=([0], [0])
    )

    # Also compute Nadaraya-Watson semantic kernel bonus toward the best passing parent in `sem24`
    best_sem = hist_sem[0:1]
    cos_to_best = np.clip((prop_sem @ best_sem.T).squeeze(-1), -1.0, 1.0)
    dsl_comp = np.array([p["dsl"].count("(") / 12.0 for p in proposals], dtype=np.float64)
    combined_score = mean_neural_score + 0.25 * cos_to_best - 0.08 * dsl_comp

    # Select top (k_select - 2) by exploitation + 2 by UCB diversity
    n_exploit = max(1, k_select - 2)
    chosen_idx: List[int] = np.argsort(-combined_score)[:n_exploit].tolist()

    all_eval_sem = np.stack([e["sem24"] for e in evaluated_pop if "sem24" in e], axis=0)
    while len(chosen_idx) < min(k_select, len(proposals)):
        ref_sem = np.concatenate([all_eval_sem, prop_sem[chosen_idx]], axis=0)
        best_j = -1
        best_ucb = -1e18
        for j in range(len(proposals)):
            if j in chosen_idx:
                continue
            d_min = float(np.min(np.linalg.norm(prop_sem[j : j + 1] - ref_sem, axis=-1)))
            ucb = float(combined_score[j]) + 0.35 * d_min
            if ucb > best_ucb:
                best_ucb = ucb
                best_j = j
        if best_j < 0:
            break
        chosen_idx.append(best_j)

    return [proposals[j] for j in chosen_idx]


def run_3gen_live_gp_evolution_for_context(
    cid: str,
    family_id: str,
    env_jtt1_path: Path,
    raw_jtt1_path: Path,
    views_ctx: Dict[str, np.ndarray],
    gp_bank: List[Dict[str, Any]],
    f_model: PreCondenseResidualBridge,
    g_model: PreCondenseResidualBridge,
    sc_std: FoldScaler,
    sc_s3: FoldScaler,
    device: torch.device,
    eval_cache: Dict[Tuple[str, str], Dict[str, Any]],
    n_gen0: int = 8,
    n_generations: int = 3,
    children_per_gen: int = 5,
    proposals_per_gen: int = 20,
) -> Dict[str, Any]:
    """Executes the full 11-view batched neural seeding (`Gen 0`) + 3-generation live Rust GP
    evolution (`Gen 1` -> `Gen 2` -> `Gen 3`) + leak-free 2-fold OOF Pareto stacking for `cid`.
    """
    t0 = time.monotonic()
    y_fold0, y_fold1, base_rmse0, base_rmse1 = parse_jtt1_fold_targets(raw_jtt1_path)

    def _eval_cached(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        missing = [c for c in cands if (cid, c["program_sha256"]) not in eval_cache]
        if missing:
            res_list = evaluate_gp_batch_with_oof_rust(env_jtt1_path, missing, max_workers=6, timeout_s=18.0)
            for c_in, r_out in zip(missing, res_list):
                eval_cache[(cid, c_in["program_sha256"])] = r_out
        return [eval_cache[(cid, c["program_sha256"])] for c in cands]

    # Generation 0: Batched 11-view neural seeding -> 8 GP trees -> live 2-fold Rust CV
    gen0_cands = propose_gen0_from_11view_batch(
        cid=cid,
        family_id=family_id,
        views_ctx=views_ctx,
        gp_bank=gp_bank,
        f_model=f_model,
        g_model=g_model,
        sc_std=sc_std,
        sc_s3=sc_s3,
        device=device,
        n_gen0=n_gen0,
    )
    gen0_evals = _eval_cached(gen0_cands)

    population: List[Dict[str, Any]] = list(gen0_evals)
    seen_shas = {e["program_sha256"] for e in population}
    gen_records: Dict[int, List[Dict[str, Any]]] = {0: gen0_evals}

    eligible_bank = [b for b in gp_bank if b["family_id"] != family_id][:96]
    lm_11 = views_ctx["lm48_z"]

    # Generations 1, 2, 3
    for gen_idx in range(1, n_generations + 1):
        # Select top-4 fertile parents from current population
        fertile = sorted(
            population,
            key=lambda x: (
                0 if x.get("cv_gate_passed") else 1,
                float(x.get("fitness", 100.0)),
                x["program_sha256"],
            ),
        )[:4]

        # Generate candidate children across subsample views' Landmark-V2 vectors
        raw_props: List[Dict[str, Any]] = []
        for sub_v in range(3):
            lm_v = lm_11[(gen_idx * 3 + sub_v) % len(lm_11)]
            p_sub = evaluator.propose_flywheel_candidates(
                parent_programs=fertile,
                landmark_v2_48=lm_v,
                champion_bank=eligible_bank,
                target_family_id=family_id,
                n_proposals=proposals_per_gen // 2,
                seed=prod.stream_seed("evo_gen", cid, gen_idx, sub_v),
            )
            for p in p_sub:
                if p["program_sha256"] not in seen_shas:
                    seen_shas.add(p["program_sha256"])
                    raw_props.append(p)
                    if len(raw_props) >= proposals_per_gen:
                        break
            if len(raw_props) >= proposals_per_gen:
                break

        if not raw_props:
            gen_records[gen_idx] = []
            continue

        screened_children = screen_child_proposals_with_causal_trajectory(
            proposals=raw_props,
            evaluated_pop=population,
            views_ctx=views_ctx,
            f_model=f_model,
            g_model=g_model,
            sc_std=sc_std,
            sc_s3=sc_s3,
            device=device,
            k_select=children_per_gen,
        )
        child_evals = _eval_cached(screened_children)
        gen_records[gen_idx] = child_evals
        population.extend(child_evals)

    def _best_up_to_gen(max_g: int) -> Dict[str, Any]:
        sub_pop = [e for g in range(max_g + 1) for e in gen_records.get(g, [])]
        passing = [e for e in sub_pop if e.get("cv_gate_passed") is True]
        if passing:
            return min(passing, key=lambda x: (float(x["fitness"]), x["program_sha256"]))
        return min(sub_pop, key=lambda x: (float(x.get("fitness", 100.0)), x.get("program_sha256", "")))

    best_g0 = _best_up_to_gen(0)
    best_g1 = _best_up_to_gen(1)
    best_g2 = _best_up_to_gen(2)
    best_g3 = _best_up_to_gen(3)

    stack_res = compute_leak_free_2fold_oof_stack(
        passing_evals=population,
        y_fold0=y_fold0,
        y_fold1=y_fold1,
        base_rmse0=base_rmse0,
        base_rmse1=base_rmse1,
        max_models=3,
    )

    return {
        "cid": cid,
        "family_id": family_id,
        "n_evaluated_total": len(population),
        "n_passing_total": sum(1 for e in population if e.get("cv_gate_passed") is True),
        "gen0_best": {
            "dsl": best_g0["dsl"],
            "program_sha256": best_g0["program_sha256"],
            "fitness": float(best_g0["fitness"]),
            "utility": float(best_g0["utility"]),
            "logratio": float(best_g0["logratio"]),
            "cv_gate_passed": bool(best_g0["cv_gate_passed"]),
        },
        "gen1_best": {
            "dsl": best_g1["dsl"],
            "program_sha256": best_g1["program_sha256"],
            "fitness": float(best_g1["fitness"]),
            "utility": float(best_g1["utility"]),
            "logratio": float(best_g1["logratio"]),
            "cv_gate_passed": bool(best_g1["cv_gate_passed"]),
        },
        "gen2_best": {
            "dsl": best_g2["dsl"],
            "program_sha256": best_g2["program_sha256"],
            "fitness": float(best_g2["fitness"]),
            "utility": float(best_g2["utility"]),
            "logratio": float(best_g2["logratio"]),
            "cv_gate_passed": bool(best_g2["cv_gate_passed"]),
        },
        "gen3_best": {
            "dsl": best_g3["dsl"],
            "program_sha256": best_g3["program_sha256"],
            "fitness": float(best_g3["fitness"]),
            "utility": float(best_g3["utility"]),
            "logratio": float(best_g3["logratio"]),
            "cv_gate_passed": bool(best_g3["cv_gate_passed"]),
        },
        "stack3": stack_res,
        "wall_s": round(time.monotonic() - t0, 3),
    }


def run_gates_t2_and_t3() -> Dict[str, Any]:
    """Executes Gates T2 & T3 across all 91 DL-Val91 unseen validation contexts (19 families)
    and the 32-context stratified DL-Train live-GP cohort (32 families):
      - Batched 11-View Subsample Neural Seeding (`1x100% + 8x80% + 2xcol-jitter`) -> Gen 0 (8 trees)
      - 3-Generation Live Rust GP Evolution (`Gen 1` -> `Gen 2` -> `Gen 3`, 5 neural-screened children/gen)
      - Leak-Free 2-Fold OOF Pareto Stacking (`H256_HIGH_BATCH_EVO3_STACK3`)
    """
    t0 = time.time()
    device = torch.device("cpu")
    torch.set_num_threads(2)

    net, guard, dope_snap_before, _ = prod.p255.load_fused_network(device)
    train_rows, val_rows = ch.load_cohorts()
    raw_cache = r256.load_pre_condense_cache(prod.CACHE_PATH)
    pools_val = ch.load_pools(val_rows, verify_hashes=False)

    # Load GP champion bank strictly from DL-Train
    gp_bank = load_enriched_gp_champion_bank(train_rows, raw_cache)

    # Load 11-view batch features for DL-Val91
    val_11views = ssa.extract_or_load_11view_batch_for_contexts(
        val_rows, "dl_val91_11view_batch_features.npz", raw_cache, net, guard, dope_snap_before, device
    )

    # Load production DL-Train-only Stage-2 F & G bridge soup models (Phase-2 curriculum if available, else Phase-1 H256)
    p2_bridge_path = OUT_DIR / "checkpoints" / "h257_phase2_curriculum_bridges.pt"
    b_train1219 = torch.load(
        H256_OUT_DIR / "checkpoints" / "h256_production_1219_dl_train_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )
    f_model = PreCondenseResidualBridge(arm="F_FULL629_HIDDEN256_COL64X17", d_token=32, n_heads=4).to(device)
    g_model = PreCondenseResidualBridge(arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4).to(device)
    if p2_bridge_path.exists():
        p2_br = torch.load(p2_bridge_path, map_location="cpu", weights_only=False)
        f_model.load_state_dict(p2_br["stage2_bridge_arm_f_state_dict"], strict=True)
        g_model.load_state_dict(p2_br["stage2_bridge_arm_g_state_dict"], strict=True)
        if "expanded_gp_bank" in p2_br:
            gp_bank = p2_br["expanded_gp_bank"]
    else:
        f_model.load_state_dict(b_train1219["stage2_bridge_arm_f_soup_state_dict"], strict=True)
        g_model.load_state_dict(b_train1219["stage2_bridge_arm_g_soup_state_dict"], strict=True)
    f_model.eval()
    g_model.eval()

    train_cids = [r["context_sha256"] for r in train_rows]
    sc_std = FoldScaler.fit(train_cids, raw_cache)
    s3_ckpt = torch.load(H256_OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt", map_location="cpu", weights_only=False)
    sc_s3 = sc_std.blend_with_s3_scaler(s3_ckpt["scaler_policy_b"], alpha_s3=0.25)

    # Prepare train-only 2-fold JTT1 envelopes for all 91 DL-Val91 contexts
    env_dir = CACHE_DIR / "val91_jtt1_envelopes"
    env_dir.mkdir(parents=True, exist_ok=True)
    for r in val_rows:
        cid = r["context_sha256"]
        env_p = env_dir / f"{cid}.jtt1"
        if not env_p.exists():
            raw_bytes = (ch.INPUT_DIR / "tables" / f"{cid}.jtt1").read_bytes()
            env_p.write_bytes(training_cv_envelope(raw_bytes))

    evo_cache_path = CACHE_DIR / "live_gp_3gen_val91_cache.pt"
    if evo_cache_path.exists():
        saved = torch.load(evo_cache_path, map_location="cpu", weights_only=False)
        eval_cache = saved.get("eval_cache", {})
        val_evo_by_cid = saved.get("val_evo_by_cid", {})
    else:
        eval_cache = {}
        val_evo_by_cid = {}

    for idx, r in enumerate(val_rows):
        cid = r["context_sha256"]
        if cid in val_evo_by_cid:
            continue
        fam = r["family_id"]
        res_c = run_3gen_live_gp_evolution_for_context(
            cid=cid,
            family_id=fam,
            env_jtt1_path=env_dir / f"{cid}.jtt1",
            raw_jtt1_path=ch.INPUT_DIR / "tables" / f"{cid}.jtt1",
            views_ctx=val_11views[cid],
            gp_bank=gp_bank,
            f_model=f_model,
            g_model=g_model,
            sc_std=sc_std,
            sc_s3=sc_s3,
            device=device,
            eval_cache=eval_cache,
            n_gen0=8,
            n_generations=3,
            children_per_gen=5,
            proposals_per_gen=20,
        )
        val_evo_by_cid[cid] = res_c
        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_rows):
            torch.save({"eval_cache": eval_cache, "val_evo_by_cid": val_evo_by_cid}, evo_cache_path)
            print(
                f"[T2/T3] Evolved {idx + 1}/{len(val_rows)} DL-Val91 contexts "
                f"(last={cid[:10]}, g0_u={res_c['gen0_best']['utility']:.4f} -> g3_u={res_c['gen3_best']['utility']:.4f}, "
                f"elapsed={time.time() - t0:.1f}s)",
                flush=True,
            )

    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    return val_evo_by_cid


if __name__ == "__main__":
    run_gates_t2_and_t3()
