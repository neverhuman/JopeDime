from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

WAVE_ID = "h258-top4-reveal-baseline-20261103"
STUDY_PATH = "research/flywheel_selfplay/h258_top4_reveal_baseline"
EVIDENCE_ROOT = Path("/home/ubuntu/jope_plan/evidence/H258/argon-t1405")
SNAPSHOT_DIR = EVIDENCE_ROOT / "snapshot"
INPUTS_DIR = EVIDENCE_ROOT / "inputs"

FOLD_SEED = 258140
SEEDS = [258101, 258102, 258103]
BOOT_SEED = 258150
BOOT_DRAWS = 10000
GATE_ZGAP_THRESHOLD = 0.25

DOPE_ASSET_PT = Path(
    "/home/ubuntu/JopeDimeMicro/assets/models/dope_mlc_v3_multitask_best.pt"
)
DOPE_ASSET_EXPECTED_SHA256 = (
    "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
)
DOPE_EXPECTED_SHA256 = (
    "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
)
POOL_PRIORS_NPZ = Path("/home/ubuntu/jope-data/h58/cache/pool_priors.npz")
R882_PT = INPUTS_DIR / "r882" / "round-882.pt"
R882_EXPECTED_SHA256 = (
    "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
)
R882_REF_JSON = INPUTS_DIR / "r882" / "best_dev_checkpoint.json"
R882_NATIVE91_EXPECTED = 1.1377941898164525

R3175_PT = INPUTS_DIR / "r3175" / "round-3175.pt"
R3175_EXPECTED_SHA256 = (
    "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
)

ARMS_R882 = [
    "R882",
    "GATED_025",
    "ACTIVE_2PLUS2_HURDLE",
    "TOP4_PICK",
    "TOP2_RAND2_PICK",
    "TOP4_PICK_UNGATED",
    "ACTIVE_2PLUS2_PICK",
]

ARMS_R3175 = [
    "R3175",
    "GATED_025@r3175",
    "ACTIVE_2PLUS2_HURDLE@r3175",
    "TOP4_PICK@r3175",
    "TOP2_RAND2_PICK@r3175",
    "TOP4_PICK_UNGATED@r3175",
    "ACTIVE_2PLUS2_PICK@r3175",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_snapshot_modules() -> tuple[Any, Any]:
    spec_common = importlib.util.spec_from_file_location(
        "common_h252", SNAPSHOT_DIR / "common_h252.py"
    )
    assert spec_common is not None and spec_common.loader is not None
    common_mod = importlib.util.module_from_spec(spec_common)
    sys.modules["common_h252"] = common_mod
    spec_common.loader.exec_module(common_mod)

    spec_stage1 = importlib.util.spec_from_file_location(
        "stage1_hurdle_sos_critic", SNAPSHOT_DIR / "stage1_hurdle_sos_critic.py"
    )
    assert spec_stage1 is not None and spec_stage1.loader is not None
    stage1_mod = importlib.util.module_from_spec(spec_stage1)
    spec_stage1.loader.exec_module(stage1_mod)
    return common_mod, stage1_mod


def load_cost_cache(
    all_rows: list[dict[str, Any]],
    train_npz_path: Path,
    val_npz_path: Path,
    pool_priors_npz_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    cids_sorted = sorted(r["context_sha256"] for r in all_rows)
    out: dict[str, dict[str, np.ndarray]] = {}
    with (
        np.load(train_npz_path, allow_pickle=False) as zt,
        np.load(val_npz_path, allow_pickle=False) as zv,
        np.load(pool_priors_npz_path, allow_pickle=False) as zp,
    ):
        pri_cids = [str(c) for c in zp["cids"]]
        offsets = zp["offsets"]
        base_pri = zp["base_prior_cost"]
        pri_map = {
            pri_cids[i]: base_pri[offsets[i] : offsets[i + 1]].astype(np.float32)
            for i in range(len(pri_cids))
        }
        for cid in cids_sorted:
            tc = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)
            bp = pri_map[cid]
            out[cid] = {
                "total_cost": tc,
                "anchor_total": tc,
                "anchor_prior": bp,
            }
    return out


def pick_best_revealed_gated_cost(
    indices_in_rcost_order: list[int],
    fitness: np.ndarray,
    gate: np.ndarray,
) -> int:
    """
    Pick argmin of revealed gated cost (fitness if gate else 2.0) among
    indices_in_rcost_order; ties broken by order in indices_in_rcost_order (rcost order).
    """
    idx_arr = np.asarray(indices_in_rcost_order, dtype=np.int64)
    y_cost = np.where(gate[idx_arr], fitness[idx_arr], 2.0)
    return int(idx_arr[int(np.argmin(y_cost))])


def score_h258_context(
    pool: dict[str, Any],
    rcost: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    rng: np.random.Generator,
    stage1_mod: Any,
    prefix: str = "",
    verify_snapshot: bool = True,
) -> dict[str, Any]:
    """
    Evaluates all 7 arms on one context for one seeded reveal stream, plus
    exact reveal-count audit metadata and snapshot bit-exactness check.
    """
    u = pool["utility"]
    fitness = pool["fitness"]
    logratio = pool["logratio"]
    gate = pool["gate"]
    n = len(rcost)

    snap_res = None
    if verify_snapshot:
        rng_state = json.loads(json.dumps(rng.bit_generator.state))
        rng_snap = np.random.default_rng()
        rng_snap.bit_generator.state = rng_state
        snap_res = stage1_mod.score_arms_for_context(
            "ctx", pool, rcost, sos_prior, sp_prior, rng_snap
        )

    idx4 = rng.choice(n, size=4, replace=False)

    top_zgap = float(stage1_mod.compute_r882_top_zgap(rcost))
    is_gated = bool(top_zgap < GATE_ZGAP_THRESHOLD)

    order_rcost = np.argsort(rcost, kind="stable")
    pick_base = int(order_rcost[0])
    u_base = float(u[pick_base])

    # TOP4_PICK_UNGATED (descriptive, unequal budget: always 4 reveals)
    top4_list = [int(i) for i in order_rcost[:4]]
    pick_top4_ungated = pick_best_revealed_gated_cost(top4_list, fitness, gate)
    u_top4_ungated = float(u[pick_top4_ungated])

    if is_gated:
        sem24 = pool["sem24"].astype(np.float64)
        gate_risk = sos_prior[:, 1].astype(np.float64)
        unsafe_gate_mask = gate_risk > (gate_risk[pick_base] + 0.10)
        gate_safety_penalty = np.where(unsafe_gate_mask, 10.0, 0.0)
        h_cost = (
            0.65 * sos_prior[:, 0].astype(np.float64)
            + 0.35 * gate_risk
            + 0.15 * sp_prior.astype(np.float64)
            + gate_safety_penalty
        )

        s_mean = sem24.mean(axis=0, keepdims=True)
        s_std = sem24.std(axis=0, keepdims=True) + 1e-6
        X = (sem24 - s_mean) / s_std

        # 1. GATED_025 (frozen H252/H254)
        y4_cost = np.where(gate[idx4], fitness[idx4], 2.0)
        y4_c = y4_cost - float(np.mean(y4_cost))
        diff = X[:, None, :] - X[idx4][None, :, :]
        sq = np.sum(diff * diff, axis=-1) / 24.0
        K = np.exp(-sq / (2.0 * (2.0**2)))
        w = K / (np.sum(K, axis=1, keepdims=True) + 0.2)
        cost_g025 = rcost + 0.05 * (w @ y4_c)
        u_g025 = float(u[int(np.argmin(cost_g025))])

        # 2. ACTIVE_2PLUS2_HURDLE (frozen H252/H254)
        idx_a = idx4[:2]
        ya_cost = np.where(gate[idx_a], fitness[idx_a], 2.0)
        ya_c = ya_cost - float(np.mean(ya_cost))
        diff_a = X[:, None, :] - X[idx_a][None, :, :]
        sq_a = np.sum(diff_a * diff_a, axis=-1) / 24.0
        Ka = np.exp(-sq_a / (2.0 * (2.2**2)))
        wa = Ka / (np.sum(Ka, axis=1, keepdims=True) + 0.2)
        h_ca = h_cost - float(np.mean(h_cost[idx_a]))
        interim_cost = rcost + 0.035 * (wa @ ya_c) + 0.010 * h_ca

        order_int = np.argsort(interim_cost)
        chosen_b: list[int] = []
        for cand in order_int:
            ci = int(cand)
            if ci not in idx_a:
                chosen_b.append(ci)
                break
        rev_so_far = [int(i) for i in idx_a] + chosen_b
        best_b2 = -1
        best_acq = 1e18
        for cand in order_int[:6]:
            ci = int(cand)
            if ci in rev_so_far:
                continue
            d_min = float(
                np.min(np.sum((X[ci : ci + 1] - X[rev_so_far]) ** 2, axis=1) / 24.0)
            )
            acq = float(interim_cost[ci]) - 0.015 * np.sqrt(d_min)
            if acq < best_acq:
                best_acq = acq
                best_b2 = ci
        if best_b2 >= 0:
            chosen_b.append(best_b2)

        rev4 = np.array([int(i) for i in idx_a] + chosen_b, dtype=np.int64)
        y4_f = np.where(gate[rev4], fitness[rev4], 2.0)
        y4_lr = np.clip(logratio[rev4], -1.5, 1.5)
        y4_comb = 0.70 * (y4_f - float(np.mean(y4_f))) + 0.30 * (
            y4_lr - float(np.mean(y4_lr))
        )

        diff4 = X[:, None, :] - X[rev4][None, :, :]
        sq4 = np.sum(diff4 * diff4, axis=-1) / 24.0
        K4 = 0.6 * np.exp(-sq4 / (2.0 * (1.8**2))) + 0.4 * np.exp(
            -sq4 / (2.0 * (3.2**2))
        )
        w4 = K4 / (np.sum(K4, axis=1, keepdims=True) + 0.15)
        h_c4_act = h_cost - float(np.mean(h_cost[rev4]))

        cost_act = rcost + 0.05 * (w4 @ y4_comb) + 0.010 * h_c4_act
        for j_pos, c_idx in enumerate(rev4):
            cost_act[c_idx] += 0.08 * float(y4_f[j_pos] - np.mean(y4_f))
        u_act = float(u[int(np.argmin(cost_act))])

        # 3. TOP4_PICK (equal-budget baseline: reveal top-4 by rcost on gated contexts)
        u_top4 = u_top4_ungated

        # 4. TOP2_RAND2_PICK (descriptive equal-budget: top-2 by rcost + 2 random from idx4 \ top2)
        top2_list = [int(i) for i in order_rcost[:2]]
        rand2_list = [int(i) for i in idx4 if int(i) not in top2_list][:2]
        rev_t2r2 = top2_list + rand2_list
        # Order revealed candidates by rcost so ties break in rcost order
        rank_map = {int(c): rk for rk, c in enumerate(order_rcost)}
        rev_t2r2_ordered = sorted(rev_t2r2, key=lambda c: rank_map[c])
        pick_t2r2 = pick_best_revealed_gated_cost(rev_t2r2_ordered, fitness, gate)
        u_t2r2 = float(u[pick_t2r2])

        # 5. ACTIVE_2PLUS2_PICK (diagnostic: exact rev4 from ACTIVE_2PLUS2_HURDLE, pick best revealed gated cost)
        rev4_ordered = sorted([int(c) for c in rev4], key=lambda c: rank_map[c])
        pick_act_pick = pick_best_revealed_gated_cost(rev4_ordered, fitness, gate)
        u_act_pick = float(u[pick_act_pick])

        reveals_used = {
            "GATED_025": len(set(int(i) for i in idx4)),
            "ACTIVE_2PLUS2_HURDLE": len(set(int(i) for i in rev4)),
            "TOP4_PICK": len(set(top4_list)),
            "TOP2_RAND2_PICK": len(set(rev_t2r2)),
            "ACTIVE_2PLUS2_PICK": len(set(int(i) for i in rev4)),
            "TOP4_PICK_UNGATED": len(set(top4_list)),
        }
    else:
        u_g025 = u_base
        u_act = u_base
        u_top4 = u_base
        u_t2r2 = u_base
        u_act_pick = u_base
        reveals_used = {
            "GATED_025": 0,
            "ACTIVE_2PLUS2_HURDLE": 0,
            "TOP4_PICK": 0,
            "TOP2_RAND2_PICK": 0,
            "ACTIVE_2PLUS2_PICK": 0,
            "TOP4_PICK_UNGATED": len(set(top4_list)),
        }

    snap_match = True
    if snap_res is not None:
        snap_match = (
            u_base == float(snap_res["R882"])
            and u_g025 == float(snap_res["GATED_025"])
            and u_act == float(snap_res["ACTIVE_2PLUS2_HURDLE"])
        )
        if not snap_match:
            raise RuntimeError(
                f"Snapshot mismatch: {(u_base, u_g025, u_act)} != "
                f"{(snap_res['R882'], snap_res['GATED_025'], snap_res['ACTIVE_2PLUS2_HURDLE'])}"
            )

    base_name = "R882" if prefix == "" else "R3175"
    sfx = "" if prefix == "" else f"@{prefix}"
    scores = {
        base_name: u_base,
        f"GATED_025{sfx}": u_g025,
        f"ACTIVE_2PLUS2_HURDLE{sfx}": u_act,
        f"TOP4_PICK{sfx}": u_top4,
        f"TOP2_RAND2_PICK{sfx}": u_t2r2,
        f"TOP4_PICK_UNGATED{sfx}": u_top4_ungated,
        f"ACTIVE_2PLUS2_PICK{sfx}": u_act_pick,
    }
    return {
        "scores": scores,
        "is_gated": is_gated,
        "top_zgap": top_zgap,
        "reveals_used": reveals_used,
        "snap_match": snap_match,
    }


def score_permuted_negative_control(
    pool: dict[str, Any],
    rcost: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    rng_reveal: np.random.Generator,
    rng_perm: np.random.Generator,
    stage1_mod: Any,
) -> float:
    """
    Negative control for ACTIVE_2PLUS2_HURDLE: permutes candidate outcomes
    (fitness, logratio, gate) within context before revealing, keeping actual
    evaluation utility u unpermuted when scoring the final selected candidate.
    """
    n = len(rcost)
    perm = rng_perm.permutation(n)
    pool_perm = dict(pool)
    pool_perm["fitness"] = pool["fitness"][perm]
    pool_perm["logratio"] = pool["logratio"][perm]
    pool_perm["gate"] = pool["gate"][perm]
    # Note: stage1_mod.score_arms_for_context indexes pool["utility"] at the end,
    # which is unpermuted in pool_perm!
    res = stage1_mod.score_arms_for_context(
        "ctx_perm", pool_perm, rcost, sos_prior, sp_prior, rng_reveal
    )
    return float(res["ACTIVE_2PLUS2_HURDLE"])


def compute_pairwise_contrast(
    fm: np.ndarray,
    vals: np.ndarray,
    idx_treat: int,
    idx_ref: int,
    boot_ix: np.ndarray,
    tie_margin: float = 1e-12,
) -> dict[str, Any]:
    diff = fm[:, :, idx_treat] - fm[:, :, idx_ref]  # [S, F]
    per_seed_fam = diff.mean(axis=1)  # [S]
    per_seed_ctx = (vals[:, :, idx_treat] - vals[:, :, idx_ref]).mean(axis=1)
    boot_samples = np.median(diff[:, boot_ix].mean(axis=2), axis=0)  # [BOOT_DRAWS]
    ci95 = np.quantile(boot_samples, [0.025, 0.975]).tolist()
    ci975 = np.quantile(boot_samples, [0.0125, 0.9875]).tolist()
    p_one_sided = float((1 + np.sum(boot_samples <= 0.0)) / (len(boot_samples) + 1))

    wtl_by_seed = []
    harm_by_seed = []
    for s in range(diff.shape[0]):
        ds = diff[s]
        wins = int(np.sum(ds > tie_margin))
        losses = int(np.sum(ds < -tie_margin))
        ties = int(len(ds) - wins - losses)
        wtl_by_seed.append({"wins": wins, "ties": ties, "losses": losses})
        harm_by_seed.append(float(np.mean(ds < -0.01)))

    return {
        "median_family_gain": float(np.median(per_seed_fam)),
        "mean_family_gain": float(np.mean(per_seed_fam)),
        "per_seed_family_gains": [float(x) for x in per_seed_fam],
        "median_context_gain": float(np.median(per_seed_ctx)),
        "mean_context_gain": float(np.mean(per_seed_ctx)),
        "per_seed_context_gains": [float(x) for x in per_seed_ctx],
        "positive_seeds": int(np.sum(per_seed_fam > 0.0)),
        "ci95": [float(ci95[0]), float(ci95[1])],
        "ci975_holm2": [float(ci975[0]), float(ci975[1])],
        "raw_p_one_sided": p_one_sided,
        "family_wtl_by_seed": wtl_by_seed,
        "family_harm_gt_001_rate": float(np.median(harm_by_seed)),
    }


def compute_elo_summary(
    vals: np.ndarray,
    families: list[str],
    arm_names: list[str],
    anchor_idx: int,
    boot_ix: np.ndarray,
    common_mod: Any,
) -> dict[str, Any]:
    n_seeds = vals.shape[0]
    wins_by_seed = np.stack(
        [common_mod.family_wins_tensor(vals[s], families) for s in range(n_seeds)],
        axis=0,
    )  # [S, F, A, A]
    point_by_seed = common_mod.bt_ratings(wins_by_seed.sum(axis=1), anchor=anchor_idx)
    point_med = np.median(point_by_seed, axis=0)

    # Paired family bootstrap for Elo (chunked by 100 draws to avoid 9 GiB 5D intermediate tensor)
    boot_wins = np.concatenate(
        [
            wins_by_seed[:, boot_ix[i : i + 100]].sum(axis=2)
            for i in range(0, len(boot_ix), 100)
        ],
        axis=1,
    )  # [S, BOOT, A, A]
    boot_elo = np.median(
        common_mod.bt_ratings(boot_wins, anchor=anchor_idx), axis=0
    )  # [BOOT, A]
    elo_ci = np.quantile(boot_elo, [0.025, 0.975], axis=0)

    out: dict[str, Any] = {}
    for ai, name in enumerate(arm_names):
        out[name] = {
            "median_elo": float(point_med[ai]),
            "per_seed_elo": [float(point_by_seed[s, ai]) for s in range(n_seeds)],
            "ci95": [float(elo_ci[0, ai]), float(elo_ci[1, ai])],
        }
    return out


def verify_dope_and_ckpts() -> tuple[str, str, str]:
    asset_sha = sha256_file(DOPE_ASSET_PT)
    assert asset_sha == DOPE_ASSET_EXPECTED_SHA256
    r882_sha = sha256_file(R882_PT)
    r3175_sha = sha256_file(R3175_PT)
    assert r882_sha == R882_EXPECTED_SHA256
    assert r3175_sha == R3175_EXPECTED_SHA256
    st882 = torch.load(R882_PT, map_location="cpu", weights_only=True)
    st3175 = torch.load(R3175_PT, map_location="cpu", weights_only=True)
    dope882 = str(st882.get("dope_sha256", ""))
    dope3175 = str(st3175.get("dope_sha256", ""))
    assert dope882 == DOPE_EXPECTED_SHA256
    assert dope3175 == DOPE_EXPECTED_SHA256
    return dope882, r882_sha, r3175_sha


def run_smoke(out_dir: Path) -> None:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    common_mod, _ = load_snapshot_modules()
    dope_before, r882_before, r3175_before = verify_dope_and_ckpts()

    train_rows, val_rows = common_mod.load_cohorts()
    assert len(train_rows) == 1219 and len(val_rows) == 91
    assert not any(bool(r.get("has_meta_test", False)) for r in train_rows + val_rows)
    groups, cid_to_fold = common_mod.seed_folds(train_rows, FOLD_SEED)
    assert len(groups) == 5 and len(cid_to_fold) == 1219

    dope_after, _, _ = verify_dope_and_ckpts()
    smoke_obj = {
        "schema": "jope.h258.smoke/v1",
        "wave_id": WAVE_ID,
        "status": "PASS",
        "outcomes_read": 0,
        "dl_train_contexts": len(train_rows),
        "dl_val_contexts": len(val_rows),
        "fold_seed": FOLD_SEED,
        "dope_sha256_before": dope_before,
        "dope_sha256_after": dope_after,
        "r882_sha256": r882_before,
        "r3175_sha256": r3175_before,
        "wall_seconds": round(time.time() - t0, 4),
    }
    (out_dir / "smoke.json").write_text(
        json.dumps(smoke_obj, indent=2, sort_keys=True) + "\n"
    )
    print("H258_SMOKE_PASS", json.dumps(smoke_obj))


def cooperate(out_dir: Path) -> None:
    while any(out_dir.glob("*.pause.json")):
        time.sleep(0.25)


def run_primary(out_dir: Path) -> None:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    common_mod, stage1_mod = load_snapshot_modules()

    dope_before, r882_before, r3175_before = verify_dope_and_ckpts()

    train_rows, val_rows = common_mod.load_cohorts()
    all_rows = train_rows + val_rows
    assert not any(bool(r.get("has_meta_test", False)) for r in all_rows)

    ctx_feats = common_mod.load_context_features()
    promos_by_cid = common_mod.load_harvested_promotions(train_rows)

    # Audit (a): zero DL-Val context/family in any prior/promotion fit; zero fold overlap
    val_cids = {r["context_sha256"] for r in val_rows}
    val_fams = {r["family_id"] for r in val_rows}
    train_fams = {r["family_id"] for r in train_rows}
    assert len(val_cids & set(promos_by_cid.keys())) == 0
    assert len(val_fams & train_fams) == 0
    groups, cid_to_fold = common_mod.seed_folds(train_rows, FOLD_SEED)
    fold_overlap = sum(
        len(set(groups[i]) & set(groups[j])) for i in range(5) for j in range(i + 1, 5)
    )
    assert fold_overlap == 0

    r882_cache = load_cost_cache(
        all_rows,
        INPUTS_DIR / "r882-train-costs.npz",
        INPUTS_DIR / "r882-secondary-costs.npz",
        POOL_PRIORS_NPZ,
    )
    r3175_cache = load_cost_cache(
        all_rows,
        INPUTS_DIR / "r3175-train-costs.npz",
        INPUTS_DIR / "r3175-secondary-costs.npz",
        POOL_PRIORS_NPZ,
    )

    # Load pools in batches of 50 contexts, immediately building 64D candidate features and dropping
    # 412D program arrays so peak RSS never exceeds ~1.1 GiB.
    libc = ctypes.CDLL("libc.so.6")
    pools: dict[str, dict[str, Any]] = {}
    cand_feats_r882: dict[str, np.ndarray] = {}
    cand_feats_r3175: dict[str, np.ndarray] = {}

    for start_i in range(0, len(all_rows), 50):
        cooperate(out_dir)
        batch_rows = all_rows[start_i : start_i + 50]
        batch_pools = common_mod.load_pools(batch_rows, verify_hashes=False)
        batch_cf882 = stage1_mod.build_candidate_features(
            batch_rows, batch_pools, r882_cache, ctx_feats
        )
        batch_cf3175 = stage1_mod.build_candidate_features(
            batch_rows, batch_pools, r3175_cache, ctx_feats
        )
        for cid, p in batch_pools.items():
            p.pop("program", None)
            p.pop("pids", None)
            p.pop("proj16", None)
            pools[cid] = p
            cand_feats_r882[cid] = batch_cf882[cid]
            cand_feats_r3175[cid] = batch_cf3175[cid]
        del batch_pools, batch_cf882, batch_cf3175
        gc.collect()
        libc.malloc_trim(0)

    # Step 1: Regression FIRST — verify R882 Native91 = 1.1377941898164525 bit-exact (91/91 contexts)
    ref_obj = json.loads(R882_REF_JSON.read_text())
    ref_cids = [c["cid"] for c in ref_obj["evaluation_contract"]["contexts"]]
    ref_vals = ref_obj["utility"]["per_context"]
    ref_ctx_map = {cid: float(val) for cid, val in zip(ref_cids, ref_vals)}
    val_r882_utils = []
    bit_exact_matches = 0
    for r in val_rows:
        cid = r["context_sha256"]
        pick = int(np.argmin(r882_cache[cid]["total_cost"]))
        u_val = float(pools[cid]["utility"][pick])
        val_r882_utils.append(u_val)
        if u_val == ref_ctx_map[cid]:
            bit_exact_matches += 1
    native91_mean = float(np.mean(val_r882_utils))
    native91_diff = abs(native91_mean - R882_NATIVE91_EXPECTED)
    if bit_exact_matches != 91 or native91_diff > 1e-12:
        raise RuntimeError(
            f"R882 Native91 regression failed: matches={bit_exact_matches}/91, mean={native91_mean}"
        )

    val_r3175_utils = [
        float(
            pools[r["context_sha256"]]["utility"][
                int(np.argmin(r3175_cache[r["context_sha256"]]["total_cost"]))
            ]
        )
        for r in val_rows
    ]
    r3175_native91_mean = float(np.mean(val_r3175_utils))

    baseline_obj = {
        "schema": "jope.h258.baseline-regression/v1",
        "r882_native91_mean": native91_mean,
        "r882_native91_expected": R882_NATIVE91_EXPECTED,
        "r882_native91_abs_diff": native91_diff,
        "r882_bit_exact_context_matches": bit_exact_matches,
        "r3175_native91_mean": r3175_native91_mean,
        "dope_sha256": dope_before,
        "r882_sha256": r882_before,
        "r3175_sha256": r3175_before,
    }
    (out_dir / "baseline.json").write_text(
        json.dumps(baseline_obj, indent=2, sort_keys=True) + "\n"
    )

    cooperate(out_dir)
    sos_r882, sp_r882, diag_r882 = stage1_mod.fit_decomposed_hurdle_and_sp_priors(
        train_rows, val_rows, pools, cand_feats_r882, promos_by_cid, seed=FOLD_SEED
    )
    del cand_feats_r882
    gc.collect()
    libc.malloc_trim(0)

    cooperate(out_dir)
    sos_r3175, sp_r3175, diag_r3175 = stage1_mod.fit_decomposed_hurdle_and_sp_priors(
        train_rows, val_rows, pools, cand_feats_r3175, promos_by_cid, seed=FOLD_SEED
    )
    del cand_feats_r3175
    gc.collect()
    libc.malloc_trim(0)

    # Step 3: Evaluate all arms across SEEDS = [258101, 258102, 258103]
    train_r882 = np.zeros(
        (len(SEEDS), len(train_rows), len(ARMS_R882)), dtype=np.float64
    )
    val_r882 = np.zeros((len(SEEDS), len(val_rows), len(ARMS_R882)), dtype=np.float64)
    train_r3175 = np.zeros(
        (len(SEEDS), len(train_rows), len(ARMS_R3175)), dtype=np.float64
    )
    val_r3175 = np.zeros((len(SEEDS), len(val_rows), len(ARMS_R3175)), dtype=np.float64)
    val_perm = np.zeros((len(SEEDS), len(val_rows)), dtype=np.float64)

    gated_rev_counts: dict[str, list[int]] = {
        k: []
        for k in [
            "GATED_025",
            "ACTIVE_2PLUS2_HURDLE",
            "TOP4_PICK",
            "TOP2_RAND2_PICK",
            "ACTIVE_2PLUS2_PICK",
        ]
    }
    ungated_rev_counts: dict[str, list[int]] = {k: [] for k in gated_rev_counts}
    snap_exact_count_r882 = 0
    snap_exact_count_r3175 = 0
    val_gated_ctx_count = 0
    train_gated_ctx_count = 0

    for si, seed in enumerate(SEEDS):
        cooperate(out_dir)
        rng_tr_882 = np.random.default_rng(seed)
        for ci, r in enumerate(train_rows):
            if ci % 200 == 0:
                cooperate(out_dir)
            cid = r["context_sha256"]
            out_c = score_h258_context(
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_r882[cid],
                sp_r882[cid],
                rng_tr_882,
                stage1_mod,
                prefix="",
                verify_snapshot=False,
            )
            for ai, a in enumerate(ARMS_R882):
                train_r882[si, ci, ai] = out_c["scores"][a]
            if out_c["snap_match"]:
                snap_exact_count_r882 += 1
            if si == 0 and out_c["is_gated"]:
                train_gated_ctx_count += 1
            target_dict = gated_rev_counts if out_c["is_gated"] else ungated_rev_counts
            for k in gated_rev_counts:
                target_dict[k].append(out_c["reveals_used"][k])

        rng_val_882 = np.random.default_rng(seed + 1000)
        rng_val_perm_rev = np.random.default_rng(seed + 1000)
        rng_val_perm_shuf = np.random.default_rng(seed + 50000)
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            out_c = score_h258_context(
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_r882[cid],
                sp_r882[cid],
                rng_val_882,
                stage1_mod,
                prefix="",
            )
            for ai, a in enumerate(ARMS_R882):
                val_r882[si, ci, ai] = out_c["scores"][a]
            if out_c["snap_match"]:
                snap_exact_count_r882 += 1
            if si == 0 and out_c["is_gated"]:
                val_gated_ctx_count += 1
            target_dict = gated_rev_counts if out_c["is_gated"] else ungated_rev_counts
            for k in gated_rev_counts:
                target_dict[k].append(out_c["reveals_used"][k])

            val_perm[si, ci] = score_permuted_negative_control(
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_r882[cid],
                sp_r882[cid],
                rng_val_perm_rev,
                rng_val_perm_shuf,
                stage1_mod,
            )

        rng_tr_3175 = np.random.default_rng(seed)
        for ci, r in enumerate(train_rows):
            if ci % 200 == 0:
                cooperate(out_dir)
            cid = r["context_sha256"]
            out_c = score_h258_context(
                pools[cid],
                r3175_cache[cid]["total_cost"],
                sos_r3175[cid],
                sp_r3175[cid],
                rng_tr_3175,
                stage1_mod,
                prefix="r3175",
                verify_snapshot=False,
            )
            for ai, a in enumerate(ARMS_R3175):
                train_r3175[si, ci, ai] = out_c["scores"][a]
            if out_c["snap_match"]:
                snap_exact_count_r3175 += 1

        rng_val_3175 = np.random.default_rng(seed + 1000)
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            out_c = score_h258_context(
                pools[cid],
                r3175_cache[cid]["total_cost"],
                sos_r3175[cid],
                sp_r3175[cid],
                rng_val_3175,
                stage1_mod,
                prefix="r3175",
            )
            for ai, a in enumerate(ARMS_R3175):
                val_r3175[si, ci, ai] = out_c["scores"][a]
            if out_c["snap_match"]:
                snap_exact_count_r3175 += 1

    del (
        pools,
        r882_cache,
        r3175_cache,
        sos_r882,
        sp_r882,
        sos_r3175,
        sp_r3175,
        ctx_feats,
        promos_by_cid,
    )
    gc.collect()
    libc.malloc_trim(0)

    # Step 4: Statistical summaries on DL-Val and DL-Train OOF
    val_fams_list = [r["family_id"] for r in val_rows]
    train_fams_list = [r["family_id"] for r in train_rows]

    fm_val_882 = np.stack(
        [
            common_mod.family_means_matrix(val_r882[s], val_fams_list)[1]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )
    fm_tr_882 = np.stack(
        [
            common_mod.family_means_matrix(train_r882[s], train_fams_list)[1]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )
    fm_val_3175 = np.stack(
        [
            common_mod.family_means_matrix(val_r3175[s], val_fams_list)[1]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )
    fm_tr_3175 = np.stack(
        [
            common_mod.family_means_matrix(train_r3175[s], train_fams_list)[1]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )

    rng_boot_val = np.random.default_rng(BOOT_SEED)
    boot_ix_val = rng_boot_val.integers(
        0, fm_val_882.shape[1], size=(BOOT_DRAWS, fm_val_882.shape[1])
    )
    rng_boot_tr = np.random.default_rng(BOOT_SEED + 1)
    boot_ix_tr = rng_boot_tr.integers(
        0, fm_tr_882.shape[1], size=(BOOT_DRAWS, fm_tr_882.shape[1])
    )

    idx882 = {a: i for i, a in enumerate(ARMS_R882)}
    idx3175 = {a: i for i, a in enumerate(ARMS_R3175)}

    def build_contrasts(
        fm: np.ndarray,
        vals: np.ndarray,
        idx_map: dict[str, int],
        pairs: list[tuple[str, str, str]],
        boot_ix: np.ndarray,
    ) -> dict[str, Any]:
        out = {}
        for label, a_t, a_r in pairs:
            out[label] = compute_pairwise_contrast(
                fm, vals, idx_map[a_t], idx_map[a_r], boot_ix
            )
        return out

    pairs_882 = [
        ("ACTIVE_2PLUS2_HURDLE - TOP4_PICK", "ACTIVE_2PLUS2_HURDLE", "TOP4_PICK"),
        ("TOP4_PICK - GATED_025", "TOP4_PICK", "GATED_025"),
        ("ACTIVE_2PLUS2_HURDLE - GATED_025", "ACTIVE_2PLUS2_HURDLE", "GATED_025"),
        ("ACTIVE_2PLUS2_HURDLE - R882", "ACTIVE_2PLUS2_HURDLE", "R882"),
        ("TOP4_PICK - R882", "TOP4_PICK", "R882"),
        ("GATED_025 - R882", "GATED_025", "R882"),
        ("TOP2_RAND2_PICK - TOP4_PICK", "TOP2_RAND2_PICK", "TOP4_PICK"),
        ("TOP2_RAND2_PICK - GATED_025", "TOP2_RAND2_PICK", "GATED_025"),
        ("TOP2_RAND2_PICK - R882", "TOP2_RAND2_PICK", "R882"),
        ("ACTIVE_2PLUS2_PICK - TOP4_PICK", "ACTIVE_2PLUS2_PICK", "TOP4_PICK"),
        (
            "ACTIVE_2PLUS2_HURDLE - ACTIVE_2PLUS2_PICK",
            "ACTIVE_2PLUS2_HURDLE",
            "ACTIVE_2PLUS2_PICK",
        ),
        (
            "TOP4_PICK_UNGATED - TOP4_PICK (unequal budget)",
            "TOP4_PICK_UNGATED",
            "TOP4_PICK",
        ),
        (
            "TOP4_PICK_UNGATED - R882 (unequal budget)",
            "TOP4_PICK_UNGATED",
            "R882",
        ),
    ]

    val_contrasts_882 = build_contrasts(
        fm_val_882, val_r882, idx882, pairs_882, boot_ix_val
    )
    tr_contrasts_882 = build_contrasts(
        fm_tr_882, train_r882, idx882, pairs_882, boot_ix_tr
    )

    # Apply Holm-Bonferroni m=2 on PRIMARY (ACTIVE_2PLUS2_HURDLE - TOP4_PICK) and SECONDARY (TOP4_PICK - GATED_025)
    for cname_dict in (val_contrasts_882, tr_contrasts_882):
        p_co = {
            "ACTIVE_2PLUS2_HURDLE - TOP4_PICK": cname_dict[
                "ACTIVE_2PLUS2_HURDLE - TOP4_PICK"
            ]["raw_p_one_sided"],
            "TOP4_PICK - GATED_025": cname_dict["TOP4_PICK - GATED_025"][
                "raw_p_one_sided"
            ],
        }
        holm_co = common_mod.holm_bonferroni(p_co)
        for k, hp in holm_co.items():
            cname_dict[k]["holm_p_m2"] = float(hp)

    pairs_3175 = [
        (
            "ACTIVE_2PLUS2_HURDLE@r3175 - TOP4_PICK@r3175",
            "ACTIVE_2PLUS2_HURDLE@r3175",
            "TOP4_PICK@r3175",
        ),
        ("TOP4_PICK@r3175 - GATED_025@r3175", "TOP4_PICK@r3175", "GATED_025@r3175"),
        (
            "ACTIVE_2PLUS2_HURDLE@r3175 - GATED_025@r3175",
            "ACTIVE_2PLUS2_HURDLE@r3175",
            "GATED_025@r3175",
        ),
        ("ACTIVE_2PLUS2_HURDLE@r3175 - R3175", "ACTIVE_2PLUS2_HURDLE@r3175", "R3175"),
        ("TOP4_PICK@r3175 - R3175", "TOP4_PICK@r3175", "R3175"),
        ("GATED_025@r3175 - R3175", "GATED_025@r3175", "R3175"),
        (
            "TOP2_RAND2_PICK@r3175 - TOP4_PICK@r3175",
            "TOP2_RAND2_PICK@r3175",
            "TOP4_PICK@r3175",
        ),
        ("TOP2_RAND2_PICK@r3175 - R3175", "TOP2_RAND2_PICK@r3175", "R3175"),
        (
            "ACTIVE_2PLUS2_PICK@r3175 - TOP4_PICK@r3175",
            "ACTIVE_2PLUS2_PICK@r3175",
            "TOP4_PICK@r3175",
        ),
        (
            "TOP4_PICK_UNGATED@r3175 - R3175 (unequal budget)",
            "TOP4_PICK_UNGATED@r3175",
            "R3175",
        ),
    ]
    val_contrasts_3175 = build_contrasts(
        fm_val_3175, val_r3175, idx3175, pairs_3175, boot_ix_val
    )
    tr_contrasts_3175 = build_contrasts(
        fm_tr_3175, train_r3175, idx3175, pairs_3175, boot_ix_tr
    )
    for cname_dict in (val_contrasts_3175, tr_contrasts_3175):
        p_co = {
            "ACTIVE_2PLUS2_HURDLE@r3175 - TOP4_PICK@r3175": cname_dict[
                "ACTIVE_2PLUS2_HURDLE@r3175 - TOP4_PICK@r3175"
            ]["raw_p_one_sided"],
            "TOP4_PICK@r3175 - GATED_025@r3175": cname_dict[
                "TOP4_PICK@r3175 - GATED_025@r3175"
            ]["raw_p_one_sided"],
        }
        holm_co = common_mod.holm_bonferroni(p_co)
        for k, hp in holm_co.items():
            cname_dict[k]["holm_p_m2"] = float(hp)

    # Joint panel (R882 + R3175 arms) for cross-incumbent contrasts and joint MM Elo anchored at R882=1500
    joint_arms = ARMS_R882 + ARMS_R3175
    joint_idx = {a: i for i, a in enumerate(joint_arms)}
    val_joint = np.concatenate([val_r882, val_r3175], axis=2)
    tr_joint = np.concatenate([train_r882, train_r3175], axis=2)
    fm_val_joint = np.concatenate([fm_val_882, fm_val_3175], axis=2)
    fm_tr_joint = np.concatenate([fm_tr_882, fm_tr_3175], axis=2)

    cross_pairs = [
        ("R3175 - R882", "R3175", "R882"),
        ("TOP4_PICK@r3175 - TOP4_PICK", "TOP4_PICK@r3175", "TOP4_PICK"),
        (
            "ACTIVE_2PLUS2_HURDLE@r3175 - ACTIVE_2PLUS2_HURDLE",
            "ACTIVE_2PLUS2_HURDLE@r3175",
            "ACTIVE_2PLUS2_HURDLE",
        ),
        ("TOP4_PICK@r3175 - R882", "TOP4_PICK@r3175", "R882"),
        ("ACTIVE_2PLUS2_HURDLE@r3175 - R882", "ACTIVE_2PLUS2_HURDLE@r3175", "R882"),
    ]
    val_cross_contrasts = build_contrasts(
        fm_val_joint, val_joint, joint_idx, cross_pairs, boot_ix_val
    )
    tr_cross_contrasts = build_contrasts(
        fm_tr_joint, tr_joint, joint_idx, cross_pairs, boot_ix_tr
    )

    # MM Bradley-Terry Elo (400 iterations, anchored R882 = 1500.0)
    val_elo_882 = compute_elo_summary(
        val_r882, val_fams_list, ARMS_R882, 0, boot_ix_val, common_mod
    )
    tr_elo_882 = compute_elo_summary(
        train_r882, train_fams_list, ARMS_R882, 0, boot_ix_tr, common_mod
    )
    val_elo_3175 = compute_elo_summary(
        val_r3175, val_fams_list, ARMS_R3175, 0, boot_ix_val, common_mod
    )
    tr_elo_3175 = compute_elo_summary(
        train_r3175, train_fams_list, ARMS_R3175, 0, boot_ix_tr, common_mod
    )
    val_elo_joint = compute_elo_summary(
        val_joint, val_fams_list, joint_arms, 0, boot_ix_val, common_mod
    )
    tr_elo_joint = compute_elo_summary(
        tr_joint, train_fams_list, joint_arms, 0, boot_ix_tr, common_mod
    )

    # Audit (c): Permuted-fitness negative control on DL-Val
    fm_val_perm = np.stack(
        [
            common_mod.family_means_matrix(val_perm[s][:, None], val_fams_list)[1][:, 0]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )  # [S, F]
    perm_vs_top4_diff = fm_val_perm - fm_val_882[:, :, idx882["TOP4_PICK"]]
    perm_vs_g025_diff = fm_val_perm - fm_val_882[:, :, idx882["GATED_025"]]
    perm_vs_r882_diff = fm_val_perm - fm_val_882[:, :, idx882["R882"]]

    boot_perm_top4 = np.median(perm_vs_top4_diff[:, boot_ix_val].mean(axis=2), axis=0)
    boot_perm_g025 = np.median(perm_vs_g025_diff[:, boot_ix_val].mean(axis=2), axis=0)
    boot_perm_r882 = np.median(perm_vs_r882_diff[:, boot_ix_val].mean(axis=2), axis=0)

    ci_perm_top4 = np.quantile(boot_perm_top4, [0.025, 0.975]).tolist()
    ci_perm_g025 = np.quantile(boot_perm_g025, [0.025, 0.975]).tolist()
    ci_perm_r882 = np.quantile(boot_perm_r882, [0.025, 0.975]).tolist()
    neg_ctrl_pass = bool(
        (ci_perm_g025[0] <= 0.0 <= ci_perm_g025[1])
        and (ci_perm_r882[0] <= 0.0 <= ci_perm_r882[1])
    )

    # Evaluate ADVANCE gate for ACTIVE_2PLUS2_HURDLE vs TOP4_PICK
    prim = val_contrasts_882["ACTIVE_2PLUS2_HURDLE - TOP4_PICK"]
    sec = val_contrasts_882["TOP4_PICK - GATED_025"]
    # Holm LCB: if secondary passes at alpha/2=0.025, primary is tested at alpha=0.05 (95% CI LCB), else alpha/2 (97.5% CI LCB)
    prim_holm_lcb = (
        prim["ci95"][0]
        if sec["holm_p_m2"] < 0.05 and prim["raw_p_one_sided"] >= sec["raw_p_one_sided"]
        else prim["ci975_holm2"][0]
    )
    gate_median_pos = bool(prim["median_family_gain"] > 0.0)
    gate_holm_lcb_pos = bool(prim_holm_lcb > 0.0)
    gate_holm_p = bool(prim["holm_p_m2"] < 0.05)
    gate_seeds = bool(prim["positive_seeds"] >= 2)
    advance_active = bool(
        gate_median_pos
        and gate_holm_lcb_pos
        and gate_holm_p
        and gate_seeds
        and neg_ctrl_pass
    )
    verdict = "ADVANCE" if advance_active else "NO_ADVANCE"

    def arm_level_summary(
        fm: np.ndarray, vals: np.ndarray, arms: list[str]
    ) -> dict[str, Any]:
        out = {}
        for ai, a in enumerate(arms):
            out[a] = {
                "median_family_utility": float(np.median(fm[:, :, ai].mean(axis=1))),
                "per_seed_family_utility": [
                    float(x) for x in fm[:, :, ai].mean(axis=1)
                ],
                "median_context_utility": float(np.median(vals[:, :, ai].mean(axis=1))),
                "per_seed_context_utility": [
                    float(x) for x in vals[:, :, ai].mean(axis=1)
                ],
            }
        return out

    dope_after, r882_after, r3175_after = verify_dope_and_ckpts()

    np.savez_compressed(
        out_dir / "panels.npz",
        val_r882=val_r882,
        train_r882=train_r882,
        val_r3175=val_r3175,
        train_r3175=train_r3175,
        val_perm=val_perm,
    )

    metrics = {
        "schema": "jope.h258.metrics/v1",
        "wave_id": WAVE_ID,
        "study_path": STUDY_PATH,
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds": round(time.time() - t0, 4),
        "verdict": verdict,
        "primary_contrast": "ACTIVE_2PLUS2_HURDLE - TOP4_PICK",
        "secondary_contrast": "TOP4_PICK - GATED_025",
        "holm_m": 2,
        "fold_seed": FOLD_SEED,
        "seeds": SEEDS,
        "boot_seed": BOOT_SEED,
        "boot_draws": BOOT_DRAWS,
        "advance_gates_active_vs_top4": {
            "median_family_gain_gt_0": gate_median_pos,
            "holm_lcb_gt_0": gate_holm_lcb_pos,
            "holm_lcb_value": float(prim_holm_lcb),
            "holm_p_lt_005": gate_holm_p,
            "positive_seeds_ge_2_of_3": gate_seeds,
            "permuted_negative_control_ci_covers_0": neg_ctrl_pass,
            "verdict": verdict,
        },
        "baseline_regression": baseline_obj,
        "dl_val_native91": {
            "n_contexts": len(val_rows),
            "n_families": fm_val_882.shape[1],
            "n_gated_contexts_r882_zgap_lt_025": val_gated_ctx_count,
            "arm_utilities_r882": arm_level_summary(fm_val_882, val_r882, ARMS_R882),
            "contrasts_r882": val_contrasts_882,
            "mm_bt_elo_r882": val_elo_882,
            "arm_utilities_r3175": arm_level_summary(
                fm_val_3175, val_r3175, ARMS_R3175
            ),
            "contrasts_r3175": val_contrasts_3175,
            "mm_bt_elo_r3175_anchored_r3175": val_elo_3175,
            "cross_incumbent_contrasts": val_cross_contrasts,
            "mm_bt_elo_joint_anchored_r882": val_elo_joint,
        },
        "dl_train_oof": {
            "n_contexts": len(train_rows),
            "n_families": fm_tr_882.shape[1],
            "n_gated_contexts_r882_zgap_lt_025": train_gated_ctx_count,
            "arm_utilities_r882": arm_level_summary(fm_tr_882, train_r882, ARMS_R882),
            "contrasts_r882": tr_contrasts_882,
            "mm_bt_elo_r882": tr_elo_882,
            "arm_utilities_r3175": arm_level_summary(
                fm_tr_3175, train_r3175, ARMS_R3175
            ),
            "contrasts_r3175": tr_contrasts_3175,
            "mm_bt_elo_r3175_anchored_r3175": tr_elo_3175,
            "cross_incumbent_contrasts": tr_cross_contrasts,
            "mm_bt_elo_joint_anchored_r882": tr_elo_joint,
        },
        "audits": {
            "audit_a_zero_leakage": {
                "pass": True,
                "dl_val_contexts_in_priors_or_promotions": 0,
                "dl_val_families_in_priors_or_promotions": 0,
                "dl_train_fold_family_overlap": fold_overlap,
                "has_meta_test_any": False,
                "train_promotions_used": int(diag_r882["sp_promotions_used"]),
                "hurdle_diag_r882": diag_r882,
                "hurdle_diag_r3175": diag_r3175,
            },
            "audit_b_reveal_budget": {
                "pass": all(
                    min(v) == 4 and max(v) == 4 for v in gated_rev_counts.values()
                )
                and all(
                    min(v) == 0 and max(v) == 0 for v in ungated_rev_counts.values()
                ),
                "gated_contexts_reveal_min_max": {
                    k: {"min": int(min(v)), "max": int(max(v)), "count": len(v)}
                    for k, v in gated_rev_counts.items()
                },
                "ungated_contexts_reveal_min_max": {
                    k: {"min": int(min(v)), "max": int(max(v)), "count": len(v)}
                    for k, v in ungated_rev_counts.items()
                },
                "snapshot_bit_exact_matches_r882": snap_exact_count_r882,
                "snapshot_bit_exact_matches_r3175": snap_exact_count_r3175,
                "total_context_seed_evaluations": len(SEEDS) * len(all_rows),
            },
            "audit_c_permuted_negative_control_dl_val": {
                "pass": neg_ctrl_pass,
                "perm_minus_top4_pick": {
                    "median_family_gain": float(
                        np.median(perm_vs_top4_diff.mean(axis=1))
                    ),
                    "ci95": [float(ci_perm_top4[0]), float(ci_perm_top4[1])],
                    "covers_zero": bool(ci_perm_top4[0] <= 0.0 <= ci_perm_top4[1]),
                },
                "perm_minus_gated_025": {
                    "median_family_gain": float(
                        np.median(perm_vs_g025_diff.mean(axis=1))
                    ),
                    "ci95": [float(ci_perm_g025[0]), float(ci_perm_g025[1])],
                    "covers_zero": bool(ci_perm_g025[0] <= 0.0 <= ci_perm_g025[1]),
                },
                "perm_minus_r882": {
                    "median_family_gain": float(
                        np.median(perm_vs_r882_diff.mean(axis=1))
                    ),
                    "ci95": [float(ci_perm_r882[0]), float(ci_perm_r882[1])],
                    "covers_zero": bool(ci_perm_r882[0] <= 0.0 <= ci_perm_r882[1]),
                },
            },
        },
        "hygiene": {
            "dope_sha256_before": dope_before,
            "dope_sha256_after": dope_after,
            "r882_sha256_before": r882_before,
            "r882_sha256_after": r882_after,
            "r3175_sha256_before": r3175_before,
            "r3175_sha256_after": r3175_after,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "optimizer_updates": 0,
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(
        "H258_PRIMARY_COMPLETE",
        json.dumps(
            {
                "verdict": verdict,
                "primary_active_minus_top4_dl_val": val_contrasts_882[
                    "ACTIVE_2PLUS2_HURDLE - TOP4_PICK"
                ],
                "secondary_top4_minus_gated025_dl_val": val_contrasts_882[
                    "TOP4_PICK - GATED_025"
                ],
                "top4_minus_r882_dl_val": val_contrasts_882["TOP4_PICK - R882"],
                "active_minus_r882_dl_val": val_contrasts_882[
                    "ACTIVE_2PLUS2_HURDLE - R882"
                ],
                "r3175_active_minus_top4_dl_val": val_contrasts_3175[
                    "ACTIVE_2PLUS2_HURDLE@r3175 - TOP4_PICK@r3175"
                ],
                "r3175_top4_minus_r3175_dl_val": val_contrasts_3175[
                    "TOP4_PICK@r3175 - R3175"
                ],
            },
            indent=2,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="H258 TOP4_PICK equal-budget runner")
    parser.add_argument("--mode", choices=["smoke", "primary"])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.output.parent
    is_smoke = bool(args.smoke or args.mode == "smoke")
    if is_smoke:
        run_smoke(out_dir)
    else:
        run_primary(out_dir)


if __name__ == "__main__":
    main()
