"""Wave H254 runner: independent, preregistered, frozen-constant confirmation of H252 Stage-1
ACTIVE_2PLUS2_HURDLE vs GATED_025 (equal 4-reveal budget) and R882 on DL-Val (Native91, 91 contexts / 19 families)
and DL-Train OOF (1,219 contexts / 205 families) with fresh fold seed 254140 and seeds 254101-254103."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GIT_REPO = Path("/home/ubuntu/JopeDime")
STUDY_REL = "research/flywheel_selfplay/h254_active2plus2_confirm"
EVIDENCE_DIR = Path("/home/ubuntu/jope_plan/evidence/H254/argon-t1305")
SNAPSHOT_DIR = EVIDENCE_DIR / "snapshot"
INPUTS_DIR = EVIDENCE_DIR / "inputs"

WAVE = "H254"
WAVE_ID = "h254-active2plus2-confirm-20261103"
FOLD_SEED = 254140
SEEDS = (254101, 254102, 254103)
BOOT_SEED = 254150
BOOT_DRAWS = 10_000
ELO_DRAWS = 1_000
HOLM_M = 2
ARMS = ("R882", "GATED_025", "ACTIVE_2PLUS2_HURDLE")

DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
R882_NATIVE91_EXACT = 1.1377941898164525
SNAPSHOT_COMMON_SHA256 = (
    "abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92"
)
SNAPSHOT_STAGE1_SHA256 = (
    "fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797"
)

# Byte-exact frozen constants of ACTIVE_2PLUS2_HURDLE from snapshot stage1_hurdle_sos_critic.py
ACTIVE_2PLUS2_CONSTANTS: dict[str, float] = {
    "zgap_thresh": 0.25,
    "gate_safety_margin": 0.10,
    "hurdle_w_cont": 0.65,
    "hurdle_w_gate": 0.35,
    "hurdle_w_sp": 0.15,
    "stage1_sigma": 2.2,
    "stage1_ridge": 0.20,
    "stage1_w_y": 0.035,
    "stage1_w_h": 0.010,
    "acq_ucb_c": 0.015,
    "comb_f_weight": 0.70,
    "comb_lr_weight": 0.30,
    "stage2_sigma1": 1.8,
    "stage2_sigma2": 3.2,
    "stage2_ridge": 0.15,
    "stage2_w_y": 0.05,
    "stage2_w_h": 0.010,
    "direct_shrink_w": 0.08,
}

PAUSE: Path | None = None


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_snapshot_modules() -> tuple[Any, Any]:
    """Load the byte-exact snapshot modules from SNAPSHOT_DIR after verifying their SHA-256."""
    common_path = SNAPSHOT_DIR / "common_h252.py"
    stage1_path = SNAPSHOT_DIR / "stage1_hurdle_sos_critic.py"
    if sha(common_path) != SNAPSHOT_COMMON_SHA256:
        raise RuntimeError(f"common_h252.py snapshot hash mismatch: {sha(common_path)}")
    if sha(stage1_path) != SNAPSHOT_STAGE1_SHA256:
        raise RuntimeError(
            f"stage1_hurdle_sos_critic.py snapshot hash mismatch: {sha(stage1_path)}"
        )

    spec_c = importlib.util.spec_from_file_location("common_h252", common_path)
    assert spec_c is not None and spec_c.loader is not None
    mod_c = importlib.util.module_from_spec(spec_c)
    sys.modules["common_h252"] = mod_c
    spec_c.loader.exec_module(mod_c)

    spec_s = importlib.util.spec_from_file_location(
        "stage1_hurdle_sos_critic", stage1_path
    )
    assert spec_s is not None and spec_s.loader is not None
    mod_s = importlib.util.module_from_spec(spec_s)
    sys.modules["stage1_hurdle_sos_critic"] = mod_s
    spec_s.loader.exec_module(mod_s)
    return mod_c, mod_s


ch252, s1 = load_snapshot_modules()


def cooperate() -> None:
    """H239 A1: block between steps while a cooperative pause is requested."""
    if PAUSE is not None:
        while PAUSE.exists():
            time.sleep(0.25)


def save_json(path: Path | str, obj: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)
    return sha(p)


def validate_launch(
    launch: dict[str, Any],
    source_commit: str | None,
    manifest_sha256: str,
    prereg_sha256: str,
    fold_manifest_sha256: str,
    smoke: bool = False,
) -> None:
    if launch.get("wave") != WAVE or launch.get("wave_id") != WAVE_ID:
        raise RuntimeError("launch wave mismatch")
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest_sha256 mismatch")
    if launch.get("prereg_sha256") != prereg_sha256:
        raise RuntimeError("launch prereg_sha256 mismatch")
    if launch.get("fold_manifest_sha256") != fold_manifest_sha256:
        raise RuntimeError("launch fold_manifest_sha256 mismatch")
    if not smoke:
        if not source_commit or len(source_commit) != 40:
            raise RuntimeError("primary launch requires 40-hex source_commit")
        if not launch.get("prereg_s3_manifest_version_id"):
            raise RuntimeError("primary launch requires prereg_s3_manifest_version_id")


def verify_pins_and_snapshot(
    pins: dict[str, Any], launch: dict[str, Any] | None = None, smoke: bool = False
) -> None:
    for kind in ("inputs", "assets", "snapshot"):
        for path, digest in pins[kind].items():
            if sha(path) != digest:
                raise RuntimeError(f"pin mismatch ({kind}): {path}")
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError(f"source pin mismatch: {path}")
    if sha(SNAPSHOT_DIR / "common_h252.py") != SNAPSHOT_COMMON_SHA256:
        raise RuntimeError("common_h252.py diverged from snapshot")
    if sha(SNAPSHOT_DIR / "stage1_hurdle_sos_critic.py") != SNAPSHOT_STAGE1_SHA256:
        raise RuntimeError("stage1_hurdle_sos_critic.py diverged from snapshot")
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"package mismatch: {name}")
    if not smoke and launch is not None:
        git_root = GIT_REPO if (GIT_REPO / ".git").exists() else REPO
        commit = launch["source_commit"]
        subprocess.check_call(
            ["git", "-C", str(git_root), "cat-file", "-e", f"{commit}^{{commit}}"]
        )
        for path in pins["source_sha256"]:
            blob = subprocess.check_output(
                ["git", "-C", str(git_root), "show", f"{commit}:{STUDY_REL}/{path}"]
            )
            if blob != (HERE / path).read_bytes():
                raise RuntimeError(f"git blob mismatch at {commit}: {path}")


def verify_dope_and_r882(ckpt_path: Path) -> dict[str, Any]:
    ckpt_sha = sha(ckpt_path)
    if ckpt_sha != R882_SHA256:
        raise RuntimeError(f"r882 checkpoint sha256 mismatch: {ckpt_sha}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != DOPE_SHA256:
        raise RuntimeError("r882 checkpoint round or dope_sha256 mismatch")
    return {
        "r882_ckpt_path": str(ckpt_path),
        "r882_sha256": ckpt_sha,
        "round": int(state["round"]),
        "dope_sha256": str(state["dope_sha256"]),
    }


def load_r882_cache_readonly(
    all_rows: list[dict[str, Any]],
    train_costs_npz: Path,
    val_costs_npz: Path,
    pool_priors_npz: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Read-only loader for R882 total_cost and base_prior_cost (never writes to /home/ubuntu/jope-data/h252)."""
    cids_sorted = sorted(r["context_sha256"] for r in all_rows)
    out: dict[str, dict[str, np.ndarray]] = {}
    with (
        np.load(train_costs_npz, allow_pickle=False) as zt,
        np.load(val_costs_npz, allow_pickle=False) as zv,
        np.load(pool_priors_npz, allow_pickle=False) as zp,
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


def audit_zero_val_leakage(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    promos_by_cid: dict[str, list[dict[str, Any]]],
    fold_seed: int = FOLD_SEED,
) -> dict[str, Any]:
    """Audit (a): Assert zero DL-Val family or context in any prior or promotion fit."""
    train_cids = {r["context_sha256"] for r in train_rows}
    train_fams = {r["family_id"] for r in train_rows}
    val_cids = {r["context_sha256"] for r in val_rows}
    val_fams = {r["family_id"] for r in val_rows}

    if not train_cids.isdisjoint(val_cids):
        raise RuntimeError("Audit (a) failed: DL-Train and DL-Val context overlap")
    if not train_fams.isdisjoint(val_fams):
        raise RuntimeError("Audit (a) failed: DL-Train and DL-Val family overlap")

    promo_cids = set(promos_by_cid.keys())
    promo_fams = {pr["family_id"] for prs in promos_by_cid.values() for pr in prs}
    val_cids_in_promo = len(promo_cids & val_cids)
    val_fams_in_promo = len(promo_fams & val_fams)
    if (
        val_cids_in_promo != 0
        or val_fams_in_promo != 0
        or not promo_cids.issubset(train_cids)
    ):
        raise RuntimeError("Audit (a) failed: promotion records outside DL-Train")

    groups, cid_to_fold = ch252.seed_folds(train_rows, fold_seed)
    fold_cid_overlap = []
    fold_fam_overlap = []
    val_cids_in_prior = len(set(cid_to_fold.keys()) & val_cids)
    val_fams_in_prior = sum(len(set(g) & val_fams) for g in groups)
    for f in range(5):
        oof_fams = set(groups[f])
        fit_fams = {fam for j, g in enumerate(groups) if j != f for fam in g}
        oof_cids = {c for c, fold in cid_to_fold.items() if fold == f}
        fit_cids = {c for c, fold in cid_to_fold.items() if fold != f}
        fold_cid_overlap.append(len(oof_cids & fit_cids) + len(fit_cids & val_cids))
        fold_fam_overlap.append(len(oof_fams & fit_fams) + len(fit_fams & val_fams))

    meta_test_count = sum(
        1 for r in train_rows + val_rows if r.get("has_meta_test") is True
    )
    if (
        val_cids_in_prior != 0
        or val_fams_in_prior != 0
        or any(fold_cid_overlap)
        or any(fold_fam_overlap)
        or meta_test_count != 0
    ):
        raise RuntimeError("Audit (a) failed: fold or holdout leakage detected")

    return {
        "status": "PASS",
        "fold_seed": int(fold_seed),
        "dl_train_contexts": len(train_rows),
        "dl_train_families": len(train_fams),
        "dl_val_contexts": len(val_rows),
        "dl_val_families": len(val_fams),
        "val_contexts_in_prior_fit": int(val_cids_in_prior),
        "val_families_in_prior_fit": int(val_fams_in_prior),
        "val_contexts_in_promotion_fit": int(val_cids_in_promo),
        "val_families_in_promotion_fit": int(val_fams_in_promo),
        "per_fold_train_oof_cid_overlap": fold_cid_overlap,
        "per_fold_train_oof_family_overlap": fold_fam_overlap,
        "meta_test_contexts_loaded": int(meta_test_count),
        "promotions_total_train_only": int(sum(len(v) for v in promos_by_cid.values())),
    }


def score_arms_instrumented(
    cid: str,
    pool: dict[str, Any],
    rcost: np.ndarray,
    sos_prior: np.ndarray,
    sp_prior: np.ndarray,
    rng: np.random.Generator,
    constants: dict[str, float] | None = None,
    revealed_fitness_override: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Byte-exact implementation of R882, GATED_025, and ACTIVE_2PLUS2_HURDLE from snapshot
    stage1_hurdle_sos_critic.py with explicit reveal-budget auditing and constant parameterization."""
    cfg = (
        ACTIVE_2PLUS2_CONSTANTS
        if constants is None
        else {**ACTIVE_2PLUS2_CONSTANTS, **constants}
    )
    u = pool["utility"]
    fitness = (
        pool["fitness"]
        if revealed_fitness_override is None
        else revealed_fitness_override
    )
    logratio = pool["logratio"]
    gate = pool["gate"]
    sem24 = pool["sem24"].astype(np.float64)
    n = len(rcost)

    idx4 = rng.choice(n, size=4, replace=False)
    top_zgap = s1.compute_r882_top_zgap(rcost)

    # 1. R882
    pick_r882 = int(np.argmin(rcost))
    u_r882 = float(u[pick_r882])

    # Hurdle Gate Feasibility Safety Filter + combined cost prior
    gate_risk = sos_prior[:, 1].astype(np.float64)
    unsafe_gate_mask = gate_risk > (gate_risk[pick_r882] + cfg["gate_safety_margin"])
    gate_safety_penalty = np.where(unsafe_gate_mask, 10.0, 0.0)
    h_cost = (
        cfg["hurdle_w_cont"] * sos_prior[:, 0].astype(np.float64)
        + cfg["hurdle_w_gate"] * gate_risk
        + cfg["hurdle_w_sp"] * sp_prior.astype(np.float64)
        + gate_safety_penalty
    )

    s_mean = sem24.mean(axis=0, keepdims=True)
    s_std = sem24.std(axis=0, keepdims=True) + 1e-6
    X = (sem24 - s_mean) / s_std

    # 2. GATED_025 (frozen H217 passive 4-reveal baseline on idx4)
    if top_zgap < 0.25:
        y4_cost = np.where(gate[idx4], fitness[idx4], 2.0)
        y4_c = y4_cost - float(np.mean(y4_cost))
        diff = X[:, None, :] - X[idx4][None, :, :]
        sq = np.sum(diff * diff, axis=-1) / 24.0
        K = np.exp(-sq / (2.0 * (2.0**2)))
        w = K / (np.sum(K, axis=1, keepdims=True) + 0.2)
        cost_g025 = rcost + 0.05 * (w @ y4_c)
        pick_g025 = int(np.argmin(cost_g025))
        u_g025 = float(u[pick_g025])
    else:
        pick_g025 = pick_r882
        u_g025 = u_r882

    # 3. ACTIVE_2PLUS2_HURDLE (Sequential 2+2 Active Reveal under exact 4-reveal budget)
    idx_a = idx4[:2]
    ya_cost = np.where(gate[idx_a], fitness[idx_a], 2.0)
    ya_c = ya_cost - float(np.mean(ya_cost))
    diff_a = X[:, None, :] - X[idx_a][None, :, :]
    sq_a = np.sum(diff_a * diff_a, axis=-1) / 24.0
    Ka = np.exp(-sq_a / (2.0 * (cfg["stage1_sigma"] ** 2)))
    wa = Ka / (np.sum(Ka, axis=1, keepdims=True) + cfg["stage1_ridge"])
    h_ca = h_cost - float(np.mean(h_cost[idx_a]))
    interim_cost = rcost + cfg["stage1_w_y"] * (wa @ ya_c) + cfg["stage1_w_h"] * h_ca

    order = np.argsort(interim_cost)
    chosen_b: list[int] = []
    for cand in order:
        ci = int(cand)
        if ci not in idx_a:
            chosen_b.append(ci)
            break
    rev_so_far = list(idx_a) + chosen_b
    best_b2 = -1
    best_acq = 1e18
    for cand in order[:6]:
        ci = int(cand)
        if ci in rev_so_far:
            continue
        d_min = float(
            np.min(np.sum((X[ci : ci + 1] - X[rev_so_far]) ** 2, axis=1) / 24.0)
        )
        acq = float(interim_cost[ci]) - cfg["acq_ucb_c"] * np.sqrt(d_min)
        if acq < best_acq:
            best_acq = acq
            best_b2 = ci
    if best_b2 >= 0:
        chosen_b.append(best_b2)

    rev4 = np.array(list(idx_a) + chosen_b, dtype=np.int64)
    if top_zgap < cfg["zgap_thresh"]:
        y4_f = np.where(gate[rev4], fitness[rev4], 2.0)
        y4_lr = np.clip(logratio[rev4], -1.5, 1.5)
        y4_comb = cfg["comb_f_weight"] * (y4_f - float(np.mean(y4_f))) + cfg[
            "comb_lr_weight"
        ] * (y4_lr - float(np.mean(y4_lr)))

        diff4 = X[:, None, :] - X[rev4][None, :, :]
        sq4 = np.sum(diff4 * diff4, axis=-1) / 24.0
        K4 = 0.6 * np.exp(-sq4 / (2.0 * (cfg["stage2_sigma1"] ** 2))) + 0.4 * np.exp(
            -sq4 / (2.0 * (cfg["stage2_sigma2"] ** 2))
        )
        w4 = K4 / (np.sum(K4, axis=1, keepdims=True) + cfg["stage2_ridge"])
        h_c4_act = h_cost - float(np.mean(h_cost[rev4]))

        cost_act = (
            rcost + cfg["stage2_w_y"] * (w4 @ y4_comb) + cfg["stage2_w_h"] * h_c4_act
        )
        for j_pos, c_idx in enumerate(rev4):
            cost_act[c_idx] += cfg["direct_shrink_w"] * float(
                y4_f[j_pos] - np.mean(y4_f)
            )
        pick_act = int(np.argmin(cost_act))
        u_act = float(u[pick_act])
    else:
        pick_act = pick_r882
        u_act = u_r882

    reveal_info = {
        "cid": cid,
        "gated_025_reveal_count": int(len(idx4)),
        "gated_025_unique_reveals": int(len(set(idx4.tolist()))),
        "active_2plus2_reveal_count": int(len(rev4)),
        "active_2plus2_unique_reveals": int(len(set(rev4.tolist()))),
        "stage_a_count": int(len(idx_a)),
        "stage_b_count": int(len(chosen_b)),
        "stage_a_b_disjoint": bool(set(idx_a.tolist()).isdisjoint(set(chosen_b))),
        "gate_open_025": bool(top_zgap < 0.25),
        "top_zgap": float(top_zgap),
        "pick_r882": pick_r882,
        "pick_gated_025": pick_g025,
        "pick_active_2plus2": pick_act,
    }
    return {
        "R882": u_r882,
        "GATED_025": u_g025,
        "ACTIVE_2PLUS2_HURDLE": u_act,
    }, reveal_info


def paired_family_contrast(
    rows: list[dict[str, Any]],
    panel_treat: np.ndarray,
    panel_ref: np.ndarray,
    boot_seed: int = BOOT_SEED,
    boot_draws: int = BOOT_DRAWS,
    holm_m: int = HOLM_M,
) -> dict[str, Any]:
    """10,000-draw paired family bootstrap contrast (panel shape [S, C])."""
    families = [r["family_id"] for r in rows]
    fam_order = sorted(set(families))
    n_seeds = panel_treat.shape[0]
    n_fams = len(fam_order)
    fams_arr = np.asarray(families)

    fm_treat = np.stack(
        [
            [panel_treat[s, fams_arr == f].mean() for f in fam_order]
            for s in range(n_seeds)
        ],
        axis=0,
    )
    fm_ref = np.stack(
        [
            [panel_ref[s, fams_arr == f].mean() for f in fam_order]
            for s in range(n_seeds)
        ],
        axis=0,
    )
    diff = fm_treat - fm_ref  # [S, F]
    per_seed_fam_gain = diff.mean(axis=1)  # [S]
    per_seed_ctx_gain = (panel_treat - panel_ref).mean(axis=1)  # [S]

    rng = np.random.default_rng(boot_seed)
    boot_ix = rng.integers(0, n_fams, size=(boot_draws, n_fams))
    boot_samples = np.median(diff[:, boot_ix].mean(axis=2), axis=0)  # [boot_draws]

    ci95 = np.quantile(boot_samples, [0.025, 0.975]).tolist()
    alpha_holm = 0.05 / float(holm_m)
    ci_holm = np.quantile(
        boot_samples, [alpha_holm / 2.0, 1.0 - alpha_holm / 2.0]
    ).tolist()
    lcb_holm_one_sided = float(np.quantile(boot_samples, alpha_holm))
    p_one_sided = float((1 + np.sum(boot_samples <= 0.0)) / (boot_draws + 1))
    wtl_by_seed = [
        [
            int((diff[s] > ch252.TIE_MARGIN).sum()),
            int((np.abs(diff[s]) <= ch252.TIE_MARGIN).sum()),
            int((diff[s] < -ch252.TIE_MARGIN).sum()),
        ]
        for s in range(n_seeds)
    ]
    harm_01_by_seed = [
        float((diff[s] < ch252.HARM_DELTA).mean()) for s in range(n_seeds)
    ]
    harm_any_by_seed = [int((diff[s] < -1e-4).sum()) for s in range(n_seeds)]

    return {
        "contexts": len(rows),
        "families": n_fams,
        "per_seed_family_gain": per_seed_fam_gain.tolist(),
        "median_family_gain": float(np.median(per_seed_fam_gain)),
        "mean_family_gain": float(np.mean(per_seed_fam_gain)),
        "per_seed_context_gain": per_seed_ctx_gain.tolist(),
        "median_context_gain": float(np.median(per_seed_ctx_gain)),
        "ci95": [float(ci95[0]), float(ci95[1])],
        "ci_holm_97_5": [float(ci_holm[0]), float(ci_holm[1])],
        "lcb_holm_one_sided": lcb_holm_one_sided,
        "ci95_covers_zero": bool(ci95[0] <= 0.0 <= ci95[1]),
        "positive_seeds": int((per_seed_fam_gain > 0.0).sum()),
        "p_one_sided": p_one_sided,
        "harm_frac_gt_001_by_seed": harm_01_by_seed,
        "median_harm_frac_gt_001": float(np.median(harm_01_by_seed)),
        "harm_families_count_by_seed": harm_any_by_seed,
        "family_wtl_by_seed": wtl_by_seed,
    }


def evaluate_decision_rule(
    primary_contrast: dict[str, Any],
    secondary_contrast: dict[str, Any],
    neg_control_contrast: dict[str, Any],
) -> dict[str, Any]:
    """Preregistered H254 Decision Rule (Holm m=2):
    ADVANCE only if primary (ACTIVE_2PLUS2_HURDLE - GATED_025 on DL-Val) median > 0
    AND Holm-adjusted CI > 0 (and Holm p < 0.05) AND >= 2/3 seeds positive
    AND negative control null (95% CI covers 0). Ties = NO_ADVANCE."""
    holm_p = ch252.holm_bonferroni(
        {
            "primary_active_minus_gated025": primary_contrast["p_one_sided"],
            "secondary_active_minus_r882": secondary_contrast["p_one_sided"],
        }
    )
    primary_contrast["holm_p"] = float(holm_p["primary_active_minus_gated025"])
    secondary_contrast["holm_p"] = float(holm_p["secondary_active_minus_r882"])

    g_median_pos = bool(primary_contrast["median_family_gain"] > 0.0)
    g_holm_ci_pos = bool(
        primary_contrast["ci95"][0] > 0.0
        and primary_contrast["ci_holm_97_5"][0] > 0.0
        and primary_contrast["holm_p"] < 0.05
    )
    g_seeds_pos = bool(primary_contrast["positive_seeds"] >= 2)
    g_neg_ctrl_null = bool(neg_control_contrast["ci95_covers_zero"])

    advance = bool(g_median_pos and g_holm_ci_pos and g_seeds_pos and g_neg_ctrl_null)
    return {
        "verdict": "ADVANCE" if advance else "NO_ADVANCE",
        "gates": {
            "primary_median_gt_0": g_median_pos,
            "primary_holm_adjusted_ci_gt_0_and_holm_p_lt_005": g_holm_ci_pos,
            "primary_positive_seeds_ge_2_of_3": g_seeds_pos,
            "negative_control_ci95_covers_0": g_neg_ctrl_null,
        },
        "holm_m": HOLM_M,
        "holm_p_values": holm_p,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()

    global PAUSE
    PAUSE = args.output.with_suffix(".pause.json")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")

    t0 = time.monotonic()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    pins = json.loads((HERE / "manifest.json").read_text())
    manifest_sha256 = sha(HERE / "manifest.json")
    prereg_sha256 = sha(HERE / "PREREG.md")
    fold_manifest_sha256 = sha(HERE / "fold_manifest.json")
    launch = json.loads(args.launch.read_text())

    verify_pins_and_snapshot(pins, launch=launch, smoke=args.smoke)
    validate_launch(
        launch,
        launch.get("source_commit"),
        manifest_sha256,
        prereg_sha256,
        fold_manifest_sha256,
        smoke=args.smoke,
    )

    ckpt_path = Path(pins["checkpoint"])
    dope_before = verify_dope_and_r882(ckpt_path)

    train_rows, val_rows = ch252.load_cohorts()
    fold_manifest = json.loads((HERE / "fold_manifest.json").read_text())
    groups, _ = ch252.seed_folds(train_rows, FOLD_SEED)
    if fold_manifest["fold_seed"] != FOLD_SEED or fold_manifest["families_by_seed"] != {
        str(s): groups for s in SEEDS
    }:
        raise RuntimeError("fold_manifest.json mismatch with seed_folds(254140)")

    if args.smoke:
        dope_after = verify_dope_and_r882(ckpt_path)
        save_json(
            out / "smoke.json",
            {
                "status": "PASS",
                "wave": WAVE,
                "wave_id": WAVE_ID,
                "scored_contexts": 0,
                "fold_seed": FOLD_SEED,
                "seeds": list(SEEDS),
                "arms": list(ARMS),
                "dope_before": dope_before,
                "dope_after": dope_after,
                "snapshot_common_sha256": sha(SNAPSHOT_DIR / "common_h252.py"),
                "snapshot_stage1_sha256": sha(
                    SNAPSHOT_DIR / "stage1_hurdle_sos_critic.py"
                ),
                "elapsed_sec": float(time.monotonic() - t0),
            },
        )
        return

    cooperate()
    all_rows = train_rows + val_rows
    ctx_feats = ch252.load_context_features()
    pools = ch252.load_pools(all_rows, verify_hashes=True)
    cooperate()

    promo_path = Path(pins["promotions_path"])
    promos_by_cid = ch252.load_harvested_promotions(train_rows, promo_path=promo_path)
    r882_cache = load_r882_cache_readonly(
        all_rows,
        Path(pins["r882_train_costs"]),
        Path(pins["r882_val_costs"]),
        Path(pins["pool_priors"]),
    )

    # Regression check FIRST: Native91 must reproduce 1.1377941898164525 bit-exact (91/91)
    ref_best = json.loads(Path(pins["reference"]).read_text())
    val_r882_choices = [
        int(np.argmin(r882_cache[r["context_sha256"]]["total_cost"])) for r in val_rows
    ]
    val_r882_utils = [
        float(pools[r["context_sha256"]]["utility"][c])
        for r, c in zip(val_rows, val_r882_choices, strict=True)
    ]
    val_r882_invalid = [
        not bool(pools[r["context_sha256"]]["gate"][c])
        for r, c in zip(val_rows, val_r882_choices, strict=True)
    ]
    native91_mean = float(np.mean(val_r882_utils))
    ref_utils = [float(x) for x in ref_best["utility"]["per_context"]]
    matches_91 = int(np.sum(np.asarray(val_r882_utils) == np.asarray(ref_utils)))
    if matches_91 != 91 or abs(native91_mean - R882_NATIVE91_EXACT) > 1e-15:
        raise RuntimeError(
            f"Native91 baseline check failed: matches={matches_91}/91, mean={native91_mean:.16f}"
        )
    save_json(
        out / "baseline.json",
        {
            "status": "PASS",
            "matches": matches_91,
            "mean": native91_mean,
            "expected_mean": R882_NATIVE91_EXACT,
            "bit_exact_91_of_91": bool(matches_91 == 91),
            "context_ids": [r["context_sha256"] for r in val_rows],
            "values": val_r882_utils,
            "choices": val_r882_choices,
            "invalid": val_r882_invalid,
        },
    )

    # Audit (a): zero DL-Val context/family in any prior or promotion fit
    audit_a = audit_zero_val_leakage(
        train_rows, val_rows, promos_by_cid, fold_seed=FOLD_SEED
    )

    # Build 64D outcome-free candidate features and fit 5-fold Hurdle + SP priors on DL-Train only (fold_seed=254140)
    cooperate()
    cand_feats = s1.build_candidate_features(all_rows, pools, r882_cache, ctx_feats)
    sos_priors, sp_priors, hurdle_diag = s1.fit_decomposed_hurdle_and_sp_priors(
        train_rows, val_rows, pools, cand_feats, promos_by_cid, seed=FOLD_SEED
    )
    cooperate()

    # Evaluate frozen arms across seeds 254101-254103 and verify byte-exact identity with snapshot function
    train_panel = np.zeros((len(SEEDS), len(train_rows), len(ARMS)), dtype=np.float64)
    val_panel = np.zeros((len(SEEDS), len(val_rows), len(ARMS)), dtype=np.float64)

    val_reveal_records: list[dict[str, Any]] = []
    train_reveal_records: list[dict[str, Any]] = []
    snapshot_exact_matches = 0

    for si, seed in enumerate(SEEDS):
        cooperate()
        rng_tr_inst = np.random.default_rng(seed)
        rng_tr_snap = np.random.default_rng(seed)
        for ci, r in enumerate(train_rows):
            cid = r["context_sha256"]
            res_inst, rev_info = score_arms_instrumented(
                cid,
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_priors[cid],
                sp_priors[cid],
                rng_tr_inst,
            )
            res_snap = s1.score_arms_for_context(
                cid,
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_priors[cid],
                sp_priors[cid],
                rng_tr_snap,
            )
            for ai, a in enumerate(ARMS):
                if res_inst[a] != res_snap[a]:
                    raise RuntimeError(
                        f"Snapshot mismatch on DL-Train {cid} seed {seed} arm {a}"
                    )
                train_panel[si, ci, ai] = res_inst[a]
            snapshot_exact_matches += 1
            train_reveal_records.append(rev_info)

        rng_val_inst = np.random.default_rng(seed + 1000)
        rng_val_snap = np.random.default_rng(seed + 1000)
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            res_inst, rev_info = score_arms_instrumented(
                cid,
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_priors[cid],
                sp_priors[cid],
                rng_val_inst,
            )
            res_snap = s1.score_arms_for_context(
                cid,
                pools[cid],
                r882_cache[cid]["total_cost"],
                sos_priors[cid],
                sp_priors[cid],
                rng_val_snap,
            )
            for ai, a in enumerate(ARMS):
                if res_inst[a] != res_snap[a]:
                    raise RuntimeError(
                        f"Snapshot mismatch on DL-Val {cid} seed {seed} arm {a}"
                    )
                val_panel[si, ci, ai] = res_inst[a]
            snapshot_exact_matches += 1
            val_reveal_records.append(rev_info)

    # Audit (b): per-context reveal count == 4 for ACTIVE_2PLUS2_HURDLE and GATED_025
    all_rev_records = val_reveal_records + train_reveal_records
    if not all(
        rec["gated_025_reveal_count"] == 4
        and rec["gated_025_unique_reveals"] == 4
        and rec["active_2plus2_reveal_count"] == 4
        and rec["active_2plus2_unique_reveals"] == 4
        and rec["stage_a_count"] == 2
        and rec["stage_b_count"] == 2
        and rec["stage_a_b_disjoint"]
        for rec in all_rev_records
    ):
        raise RuntimeError("Audit (b) failed: per-context reveal count != 4")

    audit_b = {
        "status": "PASS",
        "dl_val_context_seed_evals": len(val_reveal_records),
        "dl_train_context_seed_evals": len(train_reveal_records),
        "gated_025_reveal_count_min": int(
            min(r["gated_025_reveal_count"] for r in all_rev_records)
        ),
        "gated_025_reveal_count_max": int(
            max(r["gated_025_reveal_count"] for r in all_rev_records)
        ),
        "gated_025_unique_reveals_all_eq_4": True,
        "active_2plus2_reveal_count_min": int(
            min(r["active_2plus2_reveal_count"] for r in all_rev_records)
        ),
        "active_2plus2_reveal_count_max": int(
            max(r["active_2plus2_reveal_count"] for r in all_rev_records)
        ),
        "active_2plus2_unique_reveals_all_eq_4": True,
        "stage_a_plus_stage_b_all_2plus2_disjoint": True,
        "dl_val_zgap_lt_025_open_contexts_per_seed": int(
            sum(1 for r in val_reveal_records[: len(val_rows)] if r["gate_open_025"])
        ),
        "dl_train_zgap_lt_025_open_contexts_per_seed": int(
            sum(
                1 for r in train_reveal_records[: len(train_rows)] if r["gate_open_025"]
            )
        ),
        "snapshot_byte_exact_context_seed_matches": int(snapshot_exact_matches),
    }

    # Audit (c): NEGATIVE CONTROL — ACTIVE_2PLUS2_HURDLE with revealed fitness permuted within context
    cooperate()
    val_nc_act = np.zeros((len(SEEDS), len(val_rows)), dtype=np.float64)
    val_nc_g025 = np.zeros((len(SEEDS), len(val_rows)), dtype=np.float64)
    train_nc_act = np.zeros((len(SEEDS), len(train_rows)), dtype=np.float64)
    train_nc_g025 = np.zeros((len(SEEDS), len(train_rows)), dtype=np.float64)

    for si, seed in enumerate(SEEDS):
        rng_val_nc = np.random.default_rng(seed + 1000)
        for ci, r in enumerate(val_rows):
            cid = r["context_sha256"]
            p = pools[cid]
            perm_rng = np.random.default_rng(
                ch252.stream_seed("H254_NEGCTRL", seed, cid)
            )
            perm = perm_rng.permutation(len(p["fitness"]))
            res_nc, _ = score_arms_instrumented(
                cid,
                p,
                r882_cache[cid]["total_cost"],
                sos_priors[cid],
                sp_priors[cid],
                rng_val_nc,
                revealed_fitness_override=p["fitness"][perm],
            )
            val_nc_act[si, ci] = res_nc["ACTIVE_2PLUS2_HURDLE"]
            val_nc_g025[si, ci] = res_nc["GATED_025"]

        rng_tr_nc = np.random.default_rng(seed)
        for ci, r in enumerate(train_rows):
            cid = r["context_sha256"]
            p = pools[cid]
            perm_rng = np.random.default_rng(
                ch252.stream_seed("H254_NEGCTRL", seed, cid)
            )
            perm = perm_rng.permutation(len(p["fitness"]))
            res_nc, _ = score_arms_instrumented(
                cid,
                p,
                r882_cache[cid]["total_cost"],
                sos_priors[cid],
                sp_priors[cid],
                rng_tr_nc,
                revealed_fitness_override=p["fitness"][perm],
            )
            train_nc_act[si, ci] = res_nc["ACTIVE_2PLUS2_HURDLE"]
            train_nc_g025[si, ci] = res_nc["GATED_025"]

    nc_val_vs_g025_unperm = paired_family_contrast(
        val_rows, val_nc_act, val_panel[:, :, 1]
    )
    nc_val_vs_g025_perm = paired_family_contrast(val_rows, val_nc_act, val_nc_g025)
    nc_val_vs_r882 = paired_family_contrast(val_rows, val_nc_act, val_panel[:, :, 0])
    nc_train_vs_g025_unperm = paired_family_contrast(
        train_rows, train_nc_act, train_panel[:, :, 1]
    )
    nc_train_vs_r882 = paired_family_contrast(
        train_rows, train_nc_act, train_panel[:, :, 0]
    )

    audit_c = {
        "status": "PASS" if nc_val_vs_g025_unperm["ci95_covers_zero"] else "FAIL",
        "mechanism": "Within-context permutation of revealed pool fitness via deterministic stream_seed('H254_NEGCTRL', seed, cid); final candidate evaluated on true pool utility.",
        "dl_val_active2plus2_perm_minus_gated025": nc_val_vs_g025_unperm,
        "dl_val_active2plus2_perm_minus_gated025_perm": nc_val_vs_g025_perm,
        "dl_val_active2plus2_perm_minus_r882": nc_val_vs_r882,
        "dl_train_active2plus2_perm_minus_gated025": nc_train_vs_g025_unperm,
        "dl_train_active2plus2_perm_minus_r882": nc_train_vs_r882,
        "ci95_covers_zero": bool(nc_val_vs_g025_unperm["ci95_covers_zero"]),
    }

    # Audit (d): Descriptive +-50% (0.5x and 1.5x) perturbation on each ACTIVE_2PLUS2_HURDLE constant
    cooperate()
    sensitivity_rows: list[dict[str, Any]] = []
    for cname, base_val in ACTIVE_2PLUS2_CONSTANTS.items():
        for mult_label, mult in (("-50%", 0.5), ("+50%", 1.5)):
            cooperate()
            pert_cfg = {**ACTIVE_2PLUS2_CONSTANTS, cname: float(base_val * mult)}
            val_pert = np.zeros((len(SEEDS), len(val_rows)), dtype=np.float64)
            train_pert = np.zeros((len(SEEDS), len(train_rows)), dtype=np.float64)
            for si, seed in enumerate(SEEDS):
                rng_v = np.random.default_rng(seed + 1000)
                for ci, r in enumerate(val_rows):
                    cid = r["context_sha256"]
                    res_p, _ = score_arms_instrumented(
                        cid,
                        pools[cid],
                        r882_cache[cid]["total_cost"],
                        sos_priors[cid],
                        sp_priors[cid],
                        rng_v,
                        constants=pert_cfg,
                    )
                    val_pert[si, ci] = res_p["ACTIVE_2PLUS2_HURDLE"]
                rng_t = np.random.default_rng(seed)
                for ci, r in enumerate(train_rows):
                    cid = r["context_sha256"]
                    res_p, _ = score_arms_instrumented(
                        cid,
                        pools[cid],
                        r882_cache[cid]["total_cost"],
                        sos_priors[cid],
                        sp_priors[cid],
                        rng_t,
                        constants=pert_cfg,
                    )
                    train_pert[si, ci] = res_p["ACTIVE_2PLUS2_HURDLE"]

            c_val_g = paired_family_contrast(
                val_rows, val_pert, val_panel[:, :, 1], boot_draws=2000
            )
            c_val_r = paired_family_contrast(
                val_rows, val_pert, val_panel[:, :, 0], boot_draws=2000
            )
            c_tr_g = paired_family_contrast(
                train_rows, train_pert, train_panel[:, :, 1], boot_draws=2000
            )
            c_tr_r = paired_family_contrast(
                train_rows, train_pert, train_panel[:, :, 0], boot_draws=2000
            )
            sensitivity_rows.append(
                {
                    "constant": cname,
                    "base_value": float(base_val),
                    "perturbation": mult_label,
                    "perturbed_value": float(base_val * mult),
                    "dl_val_median_gain_vs_gated025": c_val_g["median_family_gain"],
                    "dl_val_ci95_vs_gated025": c_val_g["ci95"],
                    "dl_val_positive_seeds_vs_gated025": c_val_g["positive_seeds"],
                    "dl_val_median_gain_vs_r882": c_val_r["median_family_gain"],
                    "dl_val_ci95_vs_r882": c_val_r["ci95"],
                    "dl_train_oof_median_gain_vs_gated025": c_tr_g[
                        "median_family_gain"
                    ],
                    "dl_train_oof_median_gain_vs_r882": c_tr_r["median_family_gain"],
                }
            )

    audit_d = {
        "role": "descriptive_only_never_used_for_selection",
        "constants_count": len(ACTIVE_2PLUS2_CONSTANTS),
        "perturbation_factors": [0.5, 1.5],
        "total_perturbation_cells": len(sensitivity_rows),
        "positive_median_vs_gated025_on_dl_val_cells": int(
            sum(
                1 for r in sensitivity_rows if r["dl_val_median_gain_vs_gated025"] > 0.0
            )
        ),
        "positive_median_vs_r882_on_dl_val_cells": int(
            sum(1 for r in sensitivity_rows if r["dl_val_median_gain_vs_r882"] > 0.0)
        ),
        "cells": sensitivity_rows,
    }

    # Primary & Secondary Endpoints (Holm m=2, 10,000-draw paired family bootstrap + 400-iter MM Elo)
    val_primary = paired_family_contrast(
        val_rows, val_panel[:, :, 2], val_panel[:, :, 1]
    )
    val_secondary = paired_family_contrast(
        val_rows, val_panel[:, :, 2], val_panel[:, :, 0]
    )
    val_gated_vs_r882 = paired_family_contrast(
        val_rows, val_panel[:, :, 1], val_panel[:, :, 0]
    )

    train_primary = paired_family_contrast(
        train_rows, train_panel[:, :, 2], train_panel[:, :, 1]
    )
    train_secondary = paired_family_contrast(
        train_rows, train_panel[:, :, 2], train_panel[:, :, 0]
    )
    train_gated_vs_r882 = paired_family_contrast(
        train_rows, train_panel[:, :, 1], train_panel[:, :, 0]
    )
    train_holm = ch252.holm_bonferroni(
        {
            "primary_active_minus_gated025": train_primary["p_one_sided"],
            "secondary_active_minus_r882": train_secondary["p_one_sided"],
        }
    )
    train_primary["holm_p"] = float(train_holm["primary_active_minus_gated025"])
    train_secondary["holm_p"] = float(train_holm["secondary_active_minus_r882"])

    val_panel_summary = ch252.summarize_panel(
        val_rows, val_panel, list(ARMS), ref_idx=0, boot_seed=BOOT_SEED
    )
    train_panel_summary = ch252.summarize_panel(
        train_rows, train_panel, list(ARMS), ref_idx=0, boot_seed=BOOT_SEED
    )

    decision = evaluate_decision_rule(val_primary, val_secondary, nc_val_vs_g025_unperm)
    dope_after = verify_dope_and_r882(ckpt_path)

    np.savez_compressed(
        out / "panels.npz",
        seeds=np.asarray(SEEDS, dtype=np.int64),
        arms=np.asarray(ARMS),
        val_cids=np.asarray([r["context_sha256"] for r in val_rows]),
        train_cids=np.asarray([r["context_sha256"] for r in train_rows]),
        val_panel=val_panel,
        train_panel=train_panel,
        val_nc_act=val_nc_act,
        train_nc_act=train_nc_act,
    )

    metrics_obj = {
        "schema": "jope.hwave-metrics/v1",
        "wave": WAVE,
        "wave_id": WAVE_ID,
        "status": "COMPLETE",
        "verdict": decision["verdict"],
        "decision": decision,
        "fold_seed": FOLD_SEED,
        "seeds": list(SEEDS),
        "boot_seed": BOOT_SEED,
        "boot_draws": BOOT_DRAWS,
        "holm_m": HOLM_M,
        "arms": list(ARMS),
        "frozen_constants": ACTIVE_2PLUS2_CONSTANTS,
        "r882_native91_baseline": {
            "mean": native91_mean,
            "expected": R882_NATIVE91_EXACT,
            "matches": matches_91,
            "total": 91,
        },
        "dope_freeze": {
            "before": dope_before,
            "after": dope_after,
            "unchanged": bool(dope_before == dope_after),
        },
        "hurdle_oof_diagnostics_seed_254140": hurdle_diag,
        "dl_val": {
            "contexts": len(val_rows),
            "families": 19,
            "median_family_utility_by_arm": {
                a: float(val_panel_summary["median_family_utility"][i])
                for i, a in enumerate(ARMS)
            },
            "median_context_utility_by_arm": {
                a: float(val_panel_summary["median_context_utility"][i])
                for i, a in enumerate(ARMS)
            },
            "elo_400iter_mm_vs_r882": {
                a: {
                    "median_elo": float(val_panel_summary["median_elo"][i]),
                    "elo_by_seed": [
                        float(val_panel_summary["elo_by_seed"][s][i])
                        for s in range(len(SEEDS))
                    ],
                    "ci95": [float(x) for x in val_panel_summary["elo_ci95"][i]],
                }
                for i, a in enumerate(ARMS)
            },
            "primary_active_2plus2_hurdle_minus_gated_025": val_primary,
            "secondary_active_2plus2_hurdle_minus_r882": val_secondary,
            "diagnostic_gated_025_minus_r882": val_gated_vs_r882,
        },
        "dl_train_oof": {
            "contexts": len(train_rows),
            "families": 205,
            "median_family_utility_by_arm": {
                a: float(train_panel_summary["median_family_utility"][i])
                for i, a in enumerate(ARMS)
            },
            "median_context_utility_by_arm": {
                a: float(train_panel_summary["median_context_utility"][i])
                for i, a in enumerate(ARMS)
            },
            "elo_400iter_mm_vs_r882": {
                a: {
                    "median_elo": float(train_panel_summary["median_elo"][i]),
                    "elo_by_seed": [
                        float(train_panel_summary["elo_by_seed"][s][i])
                        for s in range(len(SEEDS))
                    ],
                    "ci95": [float(x) for x in train_panel_summary["elo_ci95"][i]],
                }
                for i, a in enumerate(ARMS)
            },
            "primary_active_2plus2_hurdle_minus_gated_025": train_primary,
            "secondary_active_2plus2_hurdle_minus_r882": train_secondary,
            "diagnostic_gated_025_minus_r882": train_gated_vs_r882,
        },
        "audits": {
            "audit_a_zero_dl_val_leakage": audit_a,
            "audit_b_equal_4_reveal_budget": audit_b,
            "audit_c_negative_control_permuted_fitness": audit_c,
            "audit_d_sensitivity_plus_minus_50pct": audit_d,
        },
        "elapsed_sec": float(time.monotonic() - t0),
    }
    save_json(out / "metrics.json", metrics_obj)
    print(
        f"[H254] COMPLETE in {metrics_obj['elapsed_sec']:.2f}s | Verdict={decision['verdict']} | "
        f"DL-Val Primary (ACTIVE_2PLUS2_HURDLE - GATED_025) = {val_primary['median_family_gain']:+.6f} "
        f"CI95={val_primary['ci95']} CI97.5={val_primary['ci_holm_97_5']} Holm_p={val_primary['holm_p']:.4g} | "
        f"Secondary (ACTIVE_2PLUS2_HURDLE - R882) = {val_secondary['median_family_gain']:+.6f} "
        f"CI95={val_secondary['ci95']} Holm_p={val_secondary['holm_p']:.4g} | "
        f"NegCtrl CI95={nc_val_vs_g025_unperm['ci95']} covers_0={nc_val_vs_g025_unperm['ci95_covers_zero']}"
    )


if __name__ == "__main__":
    main()
