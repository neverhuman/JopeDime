#!/usr/bin/env python3
"""Common deterministic data loader, split partitioner, endpoints (native, U4, 2+2 active reveal),
and statistical inference (10,000-draw paired family bootstrap, Holm-Bonferroni, 400-iter MM Elo)
for H252 (W-FLYWHEEL-NEXT: Closed-Loop Hurdle-RL, PCGrad-EMA Gradient Smoothing, and 2x2 Self-Play Flywheel).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

INPUT_DIR = Path("/home/ubuntu/jope-data/h50/input")
CACHE_DIR = Path("/home/ubuntu/jope-data/h58/cache")
R882_CKPT = Path("/home/ubuntu/jope-data/h51/ckpt/round-882.pt")
REF_BEST_DEV = Path("/home/ubuntu/jope-data/h195/inputs/best_dev_checkpoint.json")
HARVEST_PROMOS = Path("/home/ubuntu/jope-data/h252/harvest/promotions.jsonl")

DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
R882_NATIVE91_MEAN = 1.1377941898164525
PLATEAU_R3175_MEAN = 1.1389572403748425

SEEDS = (252101, 252102, 252103)
FOLD_SEEDS = {252101: 252141, 252102: 252142, 252103: 252143}
SAMPLE_SEED = 252140
BOOT_SEED = 252150
BOOT_DRAWS = 10_000
ELO_DRAWS = 1_000
TIE_MARGIN = 1e-4
HARM_DELTA = -0.01
LCB_MARGIN = 0.001
PLATEAU_MARGIN = 0.002


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_seed(*parts: Any) -> int:
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def save_json(path: Path | str, obj: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(p)
    return sha256_file(p)


def load_cohorts(input_dir: Path = INPUT_DIR) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pkt = json.loads((input_dir / "selected.json").read_text())
    train, val = pkt["train"], pkt["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError(f"Expected 1219 DL-Train and 91 DL-Val contexts, got {len(train)} and {len(val)}")
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("Expected 205 DL-Train families")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("Family overlap between DL-Train and DL-Val")
    for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
        for r in rows:
            if (
                r.get("dl_split") != split
                or r.get("task") != "regression"
                or r.get("has_meta_test") is True
                or r.get("source_split") not in ("train", "inner")
            ):
                raise RuntimeError(f"Protected or invalid context row: {r.get('context_sha256')}")
    return train, val


def load_context_features(cache_dir: Path = CACHE_DIR) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(cache_dir / "context_features.npz", allow_pickle=False) as z:
        cids = [str(c) for c in z["cids"]]
        tok = z["tokens_10x32"].astype(np.float32)
        mask = z["mask_10"].astype(bool)
        lm48 = z["lm48_z"].astype(np.float32)
        dope = z["dope629"].astype(np.float32)
        ctx584 = z["ctx_feat_584"].astype(np.float32)
        for i, cid in enumerate(cids):
            out[cid] = {
                "tokens_10x32": tok[i],
                "mask_10": mask[i],
                "lm48_z": lm48[i],
                "dope629": dope[i],
                "ctx_feat_584": ctx584[i],
            }
    return out


def compute_sem24_numpy(prog_412: np.ndarray) -> np.ndarray:
    """Fast vectorized 24D structural semantic signature matching compute_semantic_struct_sig_24d."""
    x = np.asarray(prog_412, dtype=np.float32)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, 412)
    padded = np.pad(flat, ((0, 0), (0, 20)), mode="constant").reshape(-1, 18, 24)
    sig = padded.mean(axis=1) + 0.5 * padded.max(axis=1)
    norms = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    return (sig / norms).reshape(*orig_shape, 24).astype(np.float32)


def load_pools(
    rows: List[Dict[str, Any]],
    input_dir: Path = INPUT_DIR,
    verify_hashes: bool = True,
) -> Dict[str, Dict[str, Any]]:
    pools: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cid = r["context_sha256"]
        p = input_dir / "pools" / f"{cid}.npz"
        if verify_hashes and sha256_file(p) != r["pool_sha256"]:
            raise RuntimeError(f"Pool hash mismatch for {cid}")
        with np.load(p, allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
            fitness = z["fitness"].astype(np.float64)
            logratio = z["logratio"].astype(np.float64)
            gate = z["gate"].astype(bool)
            pids = [s.decode("ascii") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in z["program_sha256"]]
            utility = np.where(gate, -fitness, -2.0).astype(np.float64)
            # Continuous fold disagreement s = (fitness - 0.9*logratio - 0.0005*c)/0.2 when gate==True
            # and continuous logratio everywhere non-sentinel (|logratio| < 4.9).
            valid_cont = np.isfinite(logratio) & (np.abs(logratio) < 4.9) & (np.abs(fitness) < 9.5)
            pools[cid] = {
                "program": prog,
                "sem24": compute_sem24_numpy(prog),
                "fitness": fitness,
                "logratio": logratio,
                "gate": gate,
                "utility": utility,
                "valid_cont": valid_cont,
                "pids": pids,
                "family_id": r["family_id"],
            }
    return pools


def load_harvested_promotions(
    train_rows: List[Dict[str, Any]],
    promo_path: Path = HARVEST_PROMOS,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load verified train-only GP promotions from flywheel-full-20261030, strictly checking zero Val/Holdout leakage."""
    train_cids = {r["context_sha256"]: r["family_id"] for r in train_rows}
    by_cid: Dict[str, List[Dict[str, Any]]] = {}
    if not promo_path.exists():
        return by_cid
    with promo_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["context_sha256"]
            if cid not in train_cids:
                raise RuntimeError(f"Promotion context {cid} outside DL-Train closure!")
            if not row.get("cv_gate_passed", False):
                continue
            feat412 = np.asarray(row["feat412"], dtype=np.float32)
            proj16 = np.asarray(row["proj16"], dtype=np.float32)
            by_cid.setdefault(cid, []).append(
                {
                    "context_sha256": cid,
                    "family_id": train_cids[cid],
                    "program_sha256": row["program_sha256"],
                    "dsl": row.get("dsl", ""),
                    "feat412": feat412,
                    "sem24": compute_sem24_numpy(feat412[None])[0],
                    "proj16": proj16,
                    "fitness": float(row["fitness"]),
                    "logratio": float(row["logratio"]),
                    "utility": float(-float(row["fitness"])),
                    "old_utility": float(row.get("old_utility", -float(row["fitness"]))),
                    "new_utility": float(row.get("new_utility", -float(row["fitness"]))),
                    "round": int(row.get("round", 0)),
                }
            )
    return by_cid


def seed_folds(train_rows: List[Dict[str, Any]], seed: int) -> Tuple[List[List[str]], Dict[str, int]]:
    families = sorted({r["family_id"] for r in train_rows})
    if len(families) != 205:
        raise ValueError("205 DL-Train families required")
    fseed = FOLD_SEEDS.get(seed, seed)
    groups = [
        sorted(x.tolist())
        for x in np.array_split(np.random.default_rng(fseed).permutation(families), 5)
    ]
    fam_to_fold = {f: k for k, g in enumerate(groups) for f in g}
    cid_to_fold = {r["context_sha256"]: fam_to_fold[r["family_id"]] for r in train_rows}
    return groups, cid_to_fold


def cell_folds(outer_k: int) -> Tuple[int, int, Tuple[int, ...]]:
    """Outer fold k: A = k (assessment), B = (k+1)%5 (selector/gate), FIT = remaining 3 folds."""
    a, b = outer_k, (outer_k + 1) % 5
    fit = tuple(j for j in range(5) if j not in (a, b))
    return a, b, fit


def eval_native(cost: np.ndarray, utility: np.ndarray, gate: np.ndarray | None = None) -> Tuple[float, int, bool]:
    """Flywheel native protocol: first argmin over entire pool (0..P-1)."""
    c = np.asarray(cost, dtype=np.float64)
    u = np.asarray(utility, dtype=np.float64)
    choice = int(np.argmin(c))
    inv = bool(not gate[choice]) if gate is not None else bool(u[choice] <= -1.999)
    return float(u[choice]), choice, inv


def eval_u4(cost: np.ndarray, utility: np.ndarray) -> float:
    """Budgeted search utility U4: max utility among 4 revealed anchors (0..3) and top-4 predicted unrevealed (4..P-1)."""
    c = np.asarray(cost, dtype=np.float64)
    u = np.asarray(utility, dtype=np.float64)
    top4_unrev = 4 + np.argsort(c[4:], kind="stable")[:4]
    return float(max(np.max(u[:4]), np.max(u[top4_unrev])))


def eval_active_2plus2(
    prior_cost: np.ndarray,
    sem24: np.ndarray,
    utility: np.ndarray,
    fitness: np.ndarray,
    gate: np.ndarray,
    kernel_scale: float = 4.0,
    blend_w: float = 0.35,
) -> Dict[str, float]:
    """W4 Sequential 2+2 Active Reveal at matched 4-observation budget:
    Step 1: Observe first 2 initial anchors (indices 0, 1).
    Step 2: Update prior_cost using Nadaraya-Watson kernel over revealed (0, 1) and pick 2 informative probes:
            1 exploitation pick (argmin updated cost on 2..P-1) + 1 UCB diversity pick (low cost + high semantic distance).
    Step 3: Observe those 2 chosen probes (total 4 observations = {0, 1, p1, p2}).
    Step 4: Re-update cost over all unrevealed candidates using all 4 revealed observations and return:
            - native_2plus2: utility of overall argmin (or best of 4 revealed + top-1 unrevealed)
            - u4_2plus2: best utility of the 4 revealed + top-4 unrevealed picks.
    """
    P = len(prior_cost)
    p_cost = np.asarray(prior_cost, dtype=np.float64)
    # Standardize prior_cost to comparable scale
    p_std = max(float(np.std(p_cost)), 1e-4)
    p_norm = (p_cost - float(np.mean(p_cost))) / p_std

    def kernel_update(rev_idx: List[int]) -> np.ndarray:
        rev_f = np.where(gate[rev_idx], fitness[rev_idx], np.minimum(fitness[rev_idx], 2.5))
        rev_target_z = rev_f / 0.4208
        rev_resid = rev_target_z - p_norm[rev_idx]
        # Cosine similarity in sem24
        cos_sim = np.clip(sem24 @ sem24[rev_idx].T, -1.0, 1.0)
        d2 = (1.0 - cos_sim) * kernel_scale
        wk = 0.75 / (0.15 + d2) + 0.25 / (1.50 + d2)
        sem_adj = (wk @ rev_resid) / (0.35 + wk.sum(axis=-1))
        return p_norm + blend_w * sem_adj, d2.min(axis=-1)

    # After 2 reveals (0, 1):
    cost_stage1, min_d2 = kernel_update([0, 1])
    unrev1 = np.arange(2, P)
    # Pick 1: best predicted among 2..P-1
    p1 = int(unrev1[int(np.argmin(cost_stage1[unrev1]))])
    # Pick 2: UCB diversity among remaining (cost_stage1 - 0.25 * min_dist_to_{0,1,p1})
    rem1 = unrev1[unrev1 != p1]
    d2_p1 = (1.0 - np.clip(sem24[rem1] @ sem24[p1], -1.0, 1.0)) * kernel_scale
    min_d2_rem = np.minimum(min_d2[rem1], d2_p1)
    ucb_score = cost_stage1[rem1] - 0.20 * np.sqrt(np.maximum(min_d2_rem, 0.0))
    p2 = int(rem1[int(np.argmin(ucb_score))])

    rev4 = [0, 1, p1, p2]
    cost_stage2, _ = kernel_update(rev4)
    # Anchor revealed slots at their actual observed standardized costs so argmin can pick a revealed winner if best
    obs_cost = cost_stage2.copy()
    for idx in rev4:
        obs_cost[idx] = (float(fitness[idx]) if gate[idx] else 2.0) / 0.4208

    mask_unrev = np.ones(P, dtype=bool)
    mask_unrev[rev4] = False
    unrev4 = np.flatnonzero(mask_unrev)
    top4_after = unrev4[np.argsort(cost_stage2[unrev4], kind="stable")[:4]]
    top1_overall = int(np.argmin(obs_cost))
    return {
        "native_2plus2": float(utility[top1_overall]),
        "u4_2plus2": float(max(np.max(utility[rev4]), np.max(utility[top4_after]))),
        "rev4_max": float(np.max(utility[rev4])),
    }


def zgap(r882_cost: np.ndarray, cand_cost: np.ndarray) -> float:
    """Outcome-free margin statistic: how strongly r882 disfavours cand_cost's argmin."""
    r = np.asarray(r882_cost, dtype=np.float64)
    c = np.asarray(cand_cost, dtype=np.float64)
    a, b = int(np.argmin(r)), int(np.argmin(c))
    if a == b:
        return 0.0
    sd = float(np.std(r))
    return math.inf if sd <= 1e-12 else float((r[b] - r[a]) / sd)


def family_means_matrix(values_2d: np.ndarray, families: List[str]) -> Tuple[List[str], np.ndarray]:
    """values_2d: [N_contexts, N_arms] -> returns (sorted_families, [N_families, N_arms])."""
    vals = np.asarray(values_2d, dtype=np.float64)
    fams_arr = np.asarray(families)
    order = sorted(set(families))
    fm = np.stack([vals[fams_arr == f].mean(axis=0) for f in order], axis=0)
    return order, fm


def bt_ratings(wins: np.ndarray, anchor: int = 0, prior: float = 0.01) -> np.ndarray:
    """Canonical 400-iteration MM Bradley-Terry Elo rating anchored at 1500 for `anchor`."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    s = np.ones(wins.shape[:-1], dtype=np.float64)
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(s[..., :, None] + s[..., None, :], 1e-12)
        ).sum(axis=-1)
        s = np.clip(totals / denom, 1e-6, 1e6)
        s /= np.maximum(s[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(s, 1e-9))


def family_wins_tensor(values_2d: np.ndarray, families: List[str], tie_margin: float = TIE_MARGIN) -> np.ndarray:
    """Pairwise win matrix aggregated to equal-weighted families: returns [N_families, N_arms, N_arms]."""
    diff = values_2d[:, :, None] - values_2d[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5)) * (
        1.0 - np.eye(values_2d.shape[1])
    )
    fams_arr = np.asarray(families)
    order = sorted(set(families))
    return np.stack([outcomes[fams_arr == f].mean(axis=0) for f in order], axis=0)


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    prev = 0.0
    m = len(pvals)
    for rank, k in enumerate(sorted(pvals, key=lambda x: (pvals[x], x))):
        prev = max(prev, min(1.0, pvals[k] * (m - rank)))
        out[k] = prev
    return out


def summarize_panel(
    rows: List[Dict[str, Any]],
    panel_3d: np.ndarray,
    arm_names: List[str],
    ref_idx: int = 0,
    boot_seed: int = BOOT_SEED,
    advance_margin: float = 0.002,
) -> Dict[str, Any]:
    """Comprehensive statistical summary for [N_seeds, N_contexts, N_arms] panel."""
    vals = np.asarray(panel_3d, dtype=np.float64)
    n_seeds, n_ctx, n_arms = vals.shape
    families = [r["family_id"] for r in rows]
    fam_order, fm0 = family_means_matrix(vals[0], families)
    n_fams = len(fam_order)
    fm = np.stack([family_means_matrix(vals[s], families)[1] for s in range(n_seeds)], axis=0)  # [S, F, A]

    rng = np.random.default_rng(boot_seed)
    boot_ix = rng.integers(0, n_fams, size=(BOOT_DRAWS, n_fams))

    ref_name = arm_names[ref_idx]
    contrasts: Dict[str, Any] = {}
    raw_pvals: Dict[str, float] = {}

    for ai, arm in enumerate(arm_names):
        if ai == ref_idx:
            continue
        diff = fm[:, :, ai] - fm[:, :, ref_idx]  # [S, F]
        per_seed_gain = diff.mean(axis=1)  # [S]
        per_seed_ctx_gain = (vals[:, :, ai] - vals[:, :, ref_idx]).mean(axis=1)
        boot_samples = np.median(diff[:, boot_ix].mean(axis=2), axis=0)  # [BOOT_DRAWS]
        ci95 = np.quantile(boot_samples, [0.025, 0.975]).tolist()
        lcb90 = float(np.quantile(boot_samples, 0.05))
        p_one_sided = float((1 + np.sum(boot_samples <= 0.0)) / (BOOT_DRAWS + 1))
        p_vs_margin = float((1 + np.sum(boot_samples <= LCB_MARGIN)) / (BOOT_DRAWS + 1))
        harm_01_by_seed = [float((diff[s] < HARM_DELTA).mean()) for s in range(n_seeds)]
        harm_any_by_seed = [int((diff[s] < -1e-4).sum()) for s in range(n_seeds)]
        wtl_by_seed = [
            [
                int((diff[s] > TIE_MARGIN).sum()),
                int((np.abs(diff[s]) <= TIE_MARGIN).sum()),
                int((diff[s] < -TIE_MARGIN).sum()),
            ]
            for s in range(n_seeds)
        ]
        key = f"{arm}_minus_{ref_name}"
        raw_pvals[arm] = p_one_sided
        contrasts[key] = {
            "per_seed_family_gain": per_seed_gain.tolist(),
            "median_family_gain": float(np.median(per_seed_gain)),
            "mean_family_gain": float(np.mean(per_seed_gain)),
            "per_seed_context_gain": per_seed_ctx_gain.tolist(),
            "median_context_gain": float(np.median(per_seed_ctx_gain)),
            "ci95": ci95,
            "lcb90": lcb90,
            "positive_seeds": int((per_seed_gain > 0).sum()),
            "p_one_sided": p_one_sided,
            "p_vs_001_margin": p_vs_margin,
            "harm_frac_gt_001_by_seed": harm_01_by_seed,
            "median_harm_frac_gt_001": float(np.median(harm_01_by_seed)),
            "harm_families_count_by_seed": harm_any_by_seed,
            "family_wtl_by_seed": wtl_by_seed,
        }

    holm_adj = holm_bonferroni(raw_pvals)
    for arm, hp in holm_adj.items():
        key = f"{arm}_minus_{ref_name}"
        c = contrasts[key]
        c["holm_p"] = hp
        c["advance"] = bool(
            c["median_family_gain"] > advance_margin
            and c["ci95"][0] > 0.0
            and hp < 0.05
            and c["positive_seeds"] >= 2
            and c["median_harm_frac_gt_001"] <= 0.10
        )

    # Joint Elo across all arms
    fw = np.stack([family_wins_tensor(vals[s], families) for s in range(n_seeds)], axis=0)  # [S, F, A, A]
    elo_by_seed = bt_ratings(fw.sum(axis=1), anchor=ref_idx)  # [S, A]
    elo_rng = np.random.default_rng(boot_seed + 1)
    elo_ix = elo_rng.integers(0, n_fams, size=(ELO_DRAWS, n_fams))
    elo_boot = np.median(bt_ratings(fw[:, elo_ix].sum(axis=2), anchor=ref_idx), axis=0)  # [ELO_DRAWS, A]
    elo_ci95 = np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist()

    return {
        "contexts": n_ctx,
        "families": n_fams,
        "arms": arm_names,
        "family_mean_by_seed": fm.mean(axis=1).tolist(),
        "context_mean_by_seed": vals.mean(axis=1).tolist(),
        "median_family_utility": np.median(fm.mean(axis=1), axis=0).tolist(),
        "median_context_utility": np.median(vals.mean(axis=1), axis=0).tolist(),
        "elo_by_seed": elo_by_seed.tolist(),
        "median_elo": np.median(elo_by_seed, axis=0).tolist(),
        "elo_ci95": elo_ci95,
        "contrasts": contrasts,
    }
