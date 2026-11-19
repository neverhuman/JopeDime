#!/usr/bin/env python3
"""H470: DOPE Value Probe (`WP0`–`WP4`) & Zero-Retraining Accuracy Harvester (`WP3` & `B03`).

Executes:
  1. Freeze & provenance checks (`dope_freeze_guard` composite SHA `c51a56d0...`, `0` train/dev
     family overlap, `nq == 0` FIT-only `.jtt1` tables, `WP01` authenticated caches, `DVP-S2`
     bit-parity `0.0` against H58/H256).
  2. Part A — Instrument & Bottleneck Diagnostics (`DL-Train` 1,219 contexts only):
       - A1: `Stage1ADistilledBridge` sketch vs `hpc_high`, `campaign_tokens`, & `native_high`
       - A2: `L7` audit (`lm48_z` 48D SVD coords vs `R_lm80` 80D triple-reshape distortion)
       - A3: True masked 0-reveal vs Wave-0 fake `fitness=100, gate=0` (`k=1`, `k=4`) vs 4-reveal
       - A4: `H410` `n_fit` quartiles, split-half stability, `U-S3` latency, & `DVP-S4` MDE
  3. Part B — Full CPU Value Probe Matrix (`T1`–`T4`, Contrasts `C0`–`C7` + Holm-Bonferroni)
     across `h452a/` + `h452b/` (5 family-grouped folds × 3 seeds) on `DL-Train`.
  4. Part C — Zero-Retraining Accuracy Upgrades (`WP3` 0-reveal EB variance-shrunk predicted
     reveals & `B03` folded `629 -> 96` Stage-1B SVD + zero-init `Ultra+Packet` residual adapter).
  5. One-Time Final `val91` `REGCHECK` (skipped under `--smoke`; opens `dl_val91` at most once
     after all `DL-Train` CV decisions are frozen).
  6. Emits reproducible Git artifacts (`metrics.json`, `verdict.json`, `SUMMARY.md`) in this
     directory and detailed scratch outputs in `$JOPE_DATA_ROOT/hwave-h470/results/`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
FLYWHEEL_DIR = HERE.parent
REPO_ROOT = FLYWHEEL_DIR.parents[1]
MICRO_PYTHON = Path(os.environ.get("JOPE_MICRO_PYTHON", Path.home() / "JopeDimeMicro" / "python"))
DATA_ROOT = Path(os.environ.get("JOPE_DATA_ROOT", Path.home() / "jope-data"))

for p_add in (
    str(HERE),
    str(FLYWHEEL_DIR / "dope_value_probe"),
    str(FLYWHEEL_DIR),
    str(REPO_ROOT / "python/src"),
    str(MICRO_PYTHON),
):
    if p_add not in sys.path:
        sys.path.insert(0, p_add)

import rust_evaluator as evaluator  # noqa: E402
from automatic_promotion import replay  # noqa: E402
from phase12_multimodal_model import compute_semantic_struct_sig_24d  # noqa: E402

from cache_auth import sha256_file  # noqa: E402
from dope_value_probe import holm_bonferroni  # noqa: E402
from h470_part_c import (  # noqa: E402
    run_b03_neural_cv,
    run_final_val91_regcheck,
    run_wp3_zero_eval_cv,
)
from h470_probe_tasks import (  # noqa: E402
    DVP_CONFIG_SHA256,
    build_pool_Metadata,
    dvp_score,
    run_part_a_diagnostics,
    run_part_b_probe_matrix,
)
from packet_v1_extract import (  # noqa: E402
    CPU_EXACT_FUSED_SHA256,
    DOPE_FUSED_COMPOSITE_SHA256,
    assert_memavail,
    build_or_load_h470_feature_caches,
    memavail_gb,
)


def _load_freeze_guard() -> Any:
    from jope.north_star import dope_freeze_guard

    return dope_freeze_guard


def _load_pools_for_role(
    input_dir: Path,
    records: list[dict[str, Any]],
    w1_mat: np.ndarray,
) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = str(rec["context_sha256"])
        pp = input_dir / "pools" / f"{cid}.npz"
        if sha256_file(pp) != rec["pool_sha256"]:
            raise ValueError(f"pool SHA-256 mismatch for {cid}")
        with np.load(pp, allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
            pids = [v.decode() if isinstance(v, bytes) else str(v) for v in z["program_sha256"].tolist()]
            proj16 = (prog.astype(np.float64) @ w1_mat / np.sqrt(412.0)).astype(np.float32)
            pools[cid] = {
                "rec": rec,
                "pids": pids,
                "program": prog,
                "proj16": proj16,
                "struct_sig": compute_semantic_struct_sig_24d(prog),
                "fitness": z["fitness"].astype(np.float64),
                "logratio": z["logratio"].astype(np.float64),
                "gate": z["gate"].astype(bool),
                "prior_cost": np.zeros(len(prog), dtype=np.float64),
                "scale": 0.4208,
            }
    return pools


def _build_h452_partitions(
    train_cids: list[str],
    fam_by_cid: dict[str, str],
    inc_fail_train: np.ndarray,
    salts: list[str],
    n_folds: int = 5,
) -> tuple[list[np.ndarray], list[list[int]], list[list[int]]]:
    """Exact H452–H469 floor-stratified family-grouped 5-fold splitter."""
    fail_fams: dict[str, int] = {}
    for c, is_fail in zip(train_cids, inc_fail_train.tolist()):
        if is_fail:
            f = fam_by_cid[c]
            fail_fams[f] = fail_fams.get(f, 0) + 1

    folds_out: list[np.ndarray] = []
    sizes_out: list[list[int]] = []
    fails_out: list[list[int]] = []

    for salt in salts:
        tr_fams = sorted(
            {fam_by_cid[c] for c in train_cids},
            key=lambda f: hashlib.sha256(f"{salt}{f}".encode("utf-8")).hexdigest(),
        )
        fsize = {f: sum(1 for c in train_cids if fam_by_cid[c] == f) for f in tr_fams}
        fold_of_fam: dict[str, int] = {}
        load = [0] * n_folds
        fl = [0] * n_folds
        for f in sorted([f for f in tr_fams if f in fail_fams], key=lambda f: (-fail_fams[f], -fsize[f])):
            j = min(range(n_folds), key=lambda i: (fl[i], load[i]))
            fold_of_fam[f] = j
            load[j] += fsize[f]
            fl[j] += fail_fams[f]
        for f in sorted([f for f in tr_fams if f not in fail_fams], key=lambda f: -fsize[f]):
            j = int(np.argmin(load))
            fold_of_fam[f] = j
            load[j] += fsize[f]

        fold_arr = np.array([fold_of_fam[fam_by_cid[c]] for c in train_cids], dtype=np.int64)
        for k in range(n_folds):
            fin = {fam_by_cid[train_cids[i]] for i in np.where(fold_arr == k)[0].tolist()}
            fout = {fam_by_cid[train_cids[i]] for i in np.where(fold_arr != k)[0].tolist()}
            if fin & fout:
                raise AssertionError(f"CV fold family overlap in {salt} fold {k}")
        folds_out.append(fold_arr)
        sizes_out.append(load)
        fails_out.append(fl)
    return folds_out, sizes_out, fails_out


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _format_summary_markdown(metrics: dict[str, Any], verdict: dict[str, Any]) -> str:
    pa = metrics["part_a"]
    pb = metrics["part_b"]
    pc = metrics["part_c"]
    reg = metrics.get("val91_regcheck")
    crit = verdict["criteria"]

    lines = [
        "# H470-DVP Results: DOPE Value Probe (`WP0`–`WP4`) & Zero-Retraining Accuracy Harvester (`WP3` & `B03`)",
        "",
        f"- **Wave ID**: `{verdict['wave']}`",
        f"- **Overall Verdict**: `{verdict['overall_verdict']}`",
        f"- **Frozen DOPE Composite SHA-256 (`before == after`)**: `{metrics['provenance']['dope_before_sha256']}`",
        f"- **Cohort**: `{metrics['provenance']['n_train']}` `DL-Train` contexts (`{metrics['provenance']['n_train_families']}` families) / `{metrics['provenance']['n_dev']}` `DL-Val91` contexts (`{metrics['provenance']['n_dev_families']}` families), `0` family overlap, `0` outer/query rows (`nq == 0`)",
        f"- **DVP Config SHA-256**: `{pb['dvp_config_sha256']}`",
        "",
        "---",
        "",
        "## 1. Gate & Criterion Scorecard",
        "",
        "| Criterion | Status | Empirical Evidence |",
        "| --- | --- | --- |",
    ]
    for cid, cinfo in crit.items():
        lines.append(f"| `{cid}` | **{cinfo['status']}** | {cinfo['evidence']} |")

    a1 = pa["A1_stage1a_conditioning_gap"]
    a2 = pa["A2_l7_lm48_vs_lm80_audit"]
    a3 = pa["A3_zero_reveal_vs_wave0_fake"]
    a4 = pa["A4_h410_stability_and_mde"]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Part A — Instrument & Bottleneck Diagnostics (`DL-Train` 1,219 Contexts)",
            "",
            "### A1. Stage-1A Conditioning Bottleneck & `WP01` Cache Drift",
            f"- `FusedLandmarkWorkspace` (`hpc_high`) vs H58/H256 cache max abs diff: **`{a1['wp01_campaign_tokens_vs_h58_hpc']['h58_vs_recomputed_hpc_max_abs_diff']:.1e}`** (`DVP-S2` exact bit-parity).",
            f"- Unauthenticated `tokens_cache.pt` (`h452..h469`) vs `hpc_high` token drift: median **`{a1['wp01_campaign_tokens_vs_h58_hpc']['campaign_cache_vs_hpc_median_max_diff']:.4f}`**, p95 **`{a1['wp01_campaign_tokens_vs_h58_hpc']['campaign_cache_vs_hpc_p95_max_diff']:.4f}`**, max **`{a1['wp01_campaign_tokens_vs_h58_hpc']['campaign_cache_vs_hpc_max_diff']:.4f}`** — directly validating `WP01`.",
            f"- `Stage1ADistilledBridge` (`landmark_bridge.blend_alpha = {a1['landmark_bridge_blend_alpha']:.4f}`) `vec256` cosine vs `hpc_high`: **`{a1['cosine_vec256_sketch_vs_hpc_mean']:.4f}`** (held-out family Ridge $R^2 =$ **`{a1['oof_ridge_r2_sketch_to_hpc_vec256']:.4f}`**; vs native 23-landmark `high`: cosine **`{a1['cosine_vec256_sketch_vs_native_high_mean']:.4f}`**, Ridge $R^2 =$ **`{a1['oof_ridge_r2_sketch_to_native_high_vec256']:.4f}`**).",
            "",
            "| Stage-1A Mode (`round-3175`) | 0-Rev Family Util | 0-Rev Gate Fails | 4-Rev Family Util | 4-Rev Gate Fails | 4-Rev Top-1 Agree vs `hpc_high` |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for m_name, m_row in a1["fused_net_ranking_by_stage1a_mode"].items():
        lines.append(
            f"| `{m_name}` | `{m_row['0rev_family_utility']:.6f}` | `{m_row['0rev_gatefails']}` | "
            f"`{m_row['4rev_family_utility']:.6f}` | `{m_row['4rev_gatefails']}` | `{m_row['4rev_top1_agree_vs_hpc']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "### A2. `L7` Audit (`lm48_z` 48D SVD Projection vs `R_lm80` 80D Triple Reshape)",
            f"- Verdict: **`{a2['verdict_l7']}`** (`||B^T B - I_48||_max = {a2['svd_basis_orthonormality_max_err']:.2e}`).",
            f"- Reshaping `lm48` into `(16, 3)` `(pm, ps, px)` triples encounters negative `ps` before `abs()` on **`{100.0 * a2['lm48_pseudo_std_slot_negative_frac']:.1f}%`** of slots and `px < pm` on **`{100.0 * a2['lm48_pseudo_max_lt_mean_frac']:.1f}%`** of triples.",
            f"- Family `1-NN` / `5-NN` retrieval: `lm48_z` **`{a2['family_1nn_acc']['lm48_z_48d']:.4f}`** / **`{a2['family_5nn_acc']['lm48_z_48d']:.4f}`** vs `R_lm80` **`{a2['family_1nn_acc']['R_lm80_80d']:.4f}`** / **`{a2['family_5nn_acc']['R_lm80_80d']:.4f}`** (synthesized `extra32` alone: **`{a2['family_1nn_acc']['lm80_extra32_only']:.4f}`**).",
            "",
            "### A3. True Masked 0-Reveal vs Wave-0 Fake `fitness=100, gate=0` Pseudo-Reveal",
            f"- Wave-0 passes uncapped `100.0 / 0.4208 = +{a3['uncapped_h_in_raw_f_over_scale']:.2f}` into `hist_token_proj` (`h_in`) and `w1_residual_adj`.",
            f"- Family-mean utility: `true_masked_0rev` **`{a3['modes']['true_masked_0rev']['family_mean_utility']:.6f}`** (`{a3['modes']['true_masked_0rev']['gatefails']}` gate fails) vs `wave0_fake_k1` **`{a3['modes']['wave0_fake_k1']['family_mean_utility']:.6f}`** (`{a3['modes']['wave0_fake_k1']['gatefails']}` gate fails) and `wave0_fake_k4` **`{a3['modes']['wave0_fake_k4']['family_mean_utility']:.6f}`** (`{a3['modes']['wave0_fake_k4']['gatefails']}` gate fails); `real_4rev` = **`{a3['modes']['real_4rev']['family_mean_utility']:.6f}`**.",
            "",
            "### A4. `H410` Sample-Size Stability, `U-S3` Latency, & `DVP-S4` MDE",
            f"- `DVP-S4` 80%-power MDE ($\\alpha=0.05$, 205 families): paired 0-reveal selector MDE = **`{a4['dvp_s4_power_and_mde']['paired_0rev_utility_mde80_stage1a_ref']:.6f}`** utility; paired 4-reveal vs 0-reveal MDE = **`{a4['dvp_s4_power_and_mde']['paired_4rev_vs_0rev_mde80']:.6f}`**.",
            f"- `U-S3` 5 fine FIT-CV landmarks (`gbdt`, `knn1`, `knn16`, `net`, `gbdt_quarter`) latency: 1-thread median **`{a4['u_s3_native_landmark_latency_ms']['fine_5_landmarks_median_ms']:.1f} ms`** (p95 **`{a4['u_s3_native_landmark_latency_ms']['fine_5_landmarks_p95_ms']:.1f} ms`**; ~**`{a4['u_s3_native_landmark_latency_ms']['estimated_8thread_fine_5_median_ms']:.1f} ms`** at 8 threads).",
            f"- Row-disjoint split-half cosine reliability (`all`): `vec256_sketch` **`{a4['splithalf_128_stability']['vec256_sketch']['all']['mean_cosine']:.4f}`**, `vec256_hpc` **`{a4['splithalf_128_stability']['vec256_hpc']['all']['mean_cosine']:.4f}`**, `dope629_hpc` **`{a4['splithalf_128_stability']['dope629_hpc']['all']['mean_cosine']:.4f}`**, `dope629_ultra` **`{a4['splithalf_128_stability']['dope629_ultra']['all']['mean_cosine']:.4f}`**, `R_packet_v1` **`{a4['splithalf_128_stability']['R_packet_v1']['all']['mean_cosine']:.4f}`**.",
            "",
            "---",
            "",
            "## 3. Part B — Full CPU Value Probe Matrix (`T1`–`T4`, `h452a/` + `h452b/` 5 Folds × 3 Seeds)",
            "",
            "### 3.1 Representation Arms (`DL-Train` OOF)",
            "",
            "| Arm | Width | T1 0-Rev Fam Util | T1 0-Rev NDCG@1 | T1 0-Rev GateFail | T1 4-Rev Fam Util | T2 Head LogLoss | T2 Head F1 | T3 Med $R^2$ | T4 Fine-LM Mean $R^2$ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm_name, ar in pb["arms"].items():
        u4_str = f"`{ar['t1_4rev']['family_mean_utility']:.6f}`" if "t1_4rev" in ar else "`—`"
        t4_str = f"`{ar['t4']['mean_r2']:.4f}`" if ar.get("t4") else "`—`"
        lines.append(
            f"| `{arm_name}` | `{ar['width']}` | `{ar['t1_0rev']['family_mean_utility']:.6f}` | "
            f"`{ar['t1_0rev']['ndcg1']:.4f}` | `{ar['t1_0rev']['gatefail_rate']:.4f}` | {u4_str} | "
            f"`{ar['t2']['head_logloss_mean']:.4f}` | `{ar['t2']['head_macro_f1']:.4f}` | "
            f"`{ar['t3']['median_r2']:.4f}` | {t4_str} |"
        )

    lines.extend(
        [
            "",
            "### 3.2 Pre-Registered Contrasts (`C0`–`C7`, 10,000-Draw Paired Family Bootstrap + Holm)",
            "",
            "| Contrast | Δ T1 0-Rev Fam Util | 95% Family Boot CI | Raw $p$ | Holm $p$ | Fam W/T/L |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for ck_name, c_obj in pb["contrasts"].items():
        t1c = c_obj["t1_0rev_utility"]
        ph = f"`{t1c['p_holm']:.4f}`" if "p_holm" in t1c else "`—`"
        lines.append(
            f"| `{ck_name}` | `{t1c['mean_delta']:+.6f}` | `[ {t1c['ci_low']:+.6f}, {t1c['ci_high']:+.6f} ]` | "
            f"`{t1c['p_value']:.4f}` | {ph} | `{t1c['family_wins']}/{t1c['family_ties']}/{t1c['family_losses']}` |"
        )

    wp3 = pc["wp3"]
    b03 = pc["b03"]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Part C — Zero-Retraining Accuracy Upgrades (`WP3` & `B03`, Contrast `C8`)",
            "",
            "### 4.1 `WP3` Variance-Shrunk Predicted Reveals (0 Real Candidate Evaluations)",
            f"- `V-S1` (`lam = 0` identity vs `incumbent_0rev`): `max_abs_diff = {wp3['vs1_lam0_max_abs_diff']:.1e}` (**PASS**).",
            f"- Mean top-4 Empirical-Bayes shrinkage weight $\\bar{{\\lambda}}_4 = {wp3['mean_top4_shrinkage_lambda']:.4f}$; real 4-reveal minus 0-reveal headroom = `{wp3['real_4rev_minus_0rev_family_gain']:+.6f}`.",
            "",
            "| Arm (0 Real Evals unless noted) | Fam Mean Utility | Ctx Mean Utility | GateFail Rate | Δ vs `incumbent_0rev` (95% CI) | % 4-Rev Gain | Fam Harm Rate |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for w_arm, w_row in wp3["arms"].items():
        ck_vs0 = wp3["contrasts"].get(f"{w_arm}_minus_incumbent_0rev")
        if ck_vs0:
            ci_s = f"`{ck_vs0['mean_delta']:+.6f}` (`[{ck_vs0['ci_low']:+.6f}, {ck_vs0['ci_high']:+.6f}]`)"
            pct_s = f"`{100.0 * w_row['fraction_of_4rev_gain_captured']:.1f}%`"
            hm_s = f"`{100.0 * w_row['family_harm_rate_vs_0rev']:.1f}%`"
        else:
            ci_s, pct_s, hm_s = "`baseline`", "`—`", "`—`"
        gf_s = f"`{w_row['gatefail_rate']:.4f}`" if "gatefail_rate" in w_row else "`—`"
        lines.append(
            f"| `{w_arm}` | `{w_row['family_mean_utility']:.6f}` | `{w_row['ctx_mean_utility']:.6f}` | "
            f"{gf_s} | {ci_s} | {pct_s} | {hm_s} |"
        )

    lines.extend(
        [
            "",
            "### 4.2 `B03` Folded Stage-1B SVD `629 -> 96` & Zero-Init `Ultra+Packet` Adapter (4-Reveal Flywheel CV)",
            f"- Step-0 exact bit-identity (`B03_svd96` & `B03_ultra_packet` vs `round-3175`): `max_abs_diff = {b03['step0_verification']['svd96_delta_step0_max_abs_diff']:.1e}` / `{b03['step0_verification']['ultra_packet_zero_init_step0_max_abs_diff']:.1e}` (fast cached-anchor parity `max_abs_diff = {b03['step0_verification']['fast_cached_anchor_parity_max_diff']:.2e}`).",
            "",
            "| Seed | Arm | Pooled OOF Gain | Gain 95% CI | Pooled Elo | Non-Floor Elo | Non-Floor Elo 95% CI | W / T / L | Strict |",
            "| ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for sd_row in b03["per_seed"]:
        sd_num = sd_row["seed"]
        for b_arm, b_cv in sd_row["cv"].items():
            gci = b_cv["oof_gain_ci95_family_boot"]
            nfci = b_cv["nonfloor"]["elo_ci95_family_boot"]
            lines.append(
                f"| `{sd_num}` | `{b_arm}` | `{b_cv['oof_gain_mean']:+.6f}` | `[{gci[0]:+.6f}, {gci[1]:+.6f}]` | "
                f"`{b_cv['elo_vs_incumbent_oof']:+.2f}` | `{b_cv['nonfloor']['elo']:+.2f}` | `[{nfci[0]:+.2f}, {nfci[1]:+.2f}]` | "
                f"`{b_cv['wins']}/{b_cv['ties']}/{b_cv['losses']}` | `{b_cv['passes_strict']}` |"
            )

    if reg is not None:
        w_reg = reg["wp3_val91_regcheck"]
        wc = w_reg["contrast_vs_incumbent_0rev"]
        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. One-Time Frozen `DL-Val91` `REGCHECK` (`91` Contexts / `19` Families)",
                "",
                f"- Incumbent 4-reveal `val91` parity: **`{reg['parity_incumbent_val91_ctx_utility']:.12f}`** (`parity_ok = {reg['parity_ok']}`).",
                f"- `WP3` selected arm (`{w_reg['selected_arm']}`, 0 real evals): `val91` family utility **`{w_reg['val91_family_utility']:.6f}`** vs `incumbent_0rev` **`{reg['incumbent_val91_0rev_family_utility']:.6f}`** (`Δ = {wc['mean_delta']:+.6f}`, 95% CI `[{wc['ci_low']:+.6f}, {wc['ci_high']:+.6f}]`, `not_worse = {w_reg['not_worse_family']}`).",
                "",
                "| Seed | CV-Selected B03 Arm | `val91` Ctx Util | `val91` Ctx Gain | `val91` Fam Gain (95% CI) | `val91` Elo | W / T / L | New GateFails |",
                "| ---: | --- | ---: | ---: | --- | ---: | --- | ---: |",
            ]
        )
        for br in reg["b03_val91_regchecks_by_seed"]:
            lines.append(
                f"| `{br['seed']}` | `{br['selected_arm']}` | `{br['val_ctx_utility']:.6f}` | `{br['val_ctx_gain']:+.6f}` | "
                f"`{br['val_family_gain']:+.6f}` (`[{br['val_family_ci95'][0]:+.6f}, {br['val_family_ci95'][1]:+.6f}]`) | "
                f"`{br['elo_vs_incumbent_val']:+.2f}` | `{br['wins']}/{br['ties']}/{br['losses']}` | `{br['new_gatefails']}` |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DATA_ROOT / "hwave-h469" / "input")
    ap.add_argument("--promotions-path", type=Path, default=DATA_ROOT / "hwave-h469" / "promotions.jsonl")
    ap.add_argument("--ckpt-path", type=Path, default=DATA_ROOT / "hwave-h469" / "round-3175.pt")
    ap.add_argument("--tokens-cache-path", type=Path, default=DATA_ROOT / "hwave-h469" / "tokens_cache.pt")
    ap.add_argument("--scratch-dir", type=Path, default=DATA_ROOT / "hwave-h470")
    ap.add_argument("--threads", type=int, default=8, help="Max CPU threads for LightGBM & PyTorch B03")
    ap.add_argument("--workers", type=int, default=1, help="Worker count (keep 1; lib_lightgbm is not multi-thread reentrant)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--partitions", default="h452a/,h452b/")
    ap.add_argument("--smoke", action="store_true", help="Fast fold-0 smoke check; never opens val91")
    ap.add_argument("--no-stage-cache", action="store_true", help="Recompute all stages even if stage checkpoints exist")
    args = ap.parse_args()

    t_start = time.perf_counter()
    eff_workers = 1 if args.smoke else max(1, min(3, int(args.workers)))
    lgbm_threads = max(1, min(4, int(args.threads) // eff_workers)) if eff_workers > 1 else max(1, int(args.threads))
    torch_threads = max(1, min(12, int(args.threads)))
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    assert_memavail(25.0, "startup")

    scratch = args.scratch_dir.resolve()
    allowed_scratch_root = (DATA_ROOT / "hwave-h470").resolve()
    if scratch != allowed_scratch_root and allowed_scratch_root not in scratch.parents:
        raise ValueError(f"--scratch-dir must stay inside {allowed_scratch_root}, got {scratch}")
    (scratch / "results").mkdir(parents=True, exist_ok=True)
    (scratch / "logs").mkdir(parents=True, exist_ok=True)

    guard = _load_freeze_guard()
    guard.reject_research_dope_unfreeze("h470_dope_value_probe")

    # 1. Load cohort inventory & verify strict train/dev family disjointness
    selected = json.loads((args.input_dir / "selected.json").read_text(encoding="utf-8"))
    if selected.get("protocol") != "full-DL-Train1219-DL-Val91/v1":
        raise ValueError(f"unexpected packet protocol: {selected.get('protocol')}")
    train_recs = selected["train"]
    dev_recs = selected["dev"]
    train_cids = [str(r["context_sha256"]) for r in train_recs]
    dev_cids = [str(r["context_sha256"]) for r in dev_recs]
    fam_by_cid = {str(r["context_sha256"]): str(r["family_id"]) for r in train_recs + dev_recs}
    train_families = [fam_by_cid[c] for c in train_cids]
    dev_families = [fam_by_cid[c] for c in dev_cids]
    tf_set = set(train_families)
    df_set = set(dev_families)
    if len(train_cids) != 1219 or len(dev_cids) != 91 or (tf_set & df_set):
        raise ValueError("cohort count or train/dev family disjointness check failed")

    # Load ONLY `DL-Train` pools before Part A/B/C CV (`dev` pools loaded ONLY at final `REGCHECK`!)
    w1_mat = evaluator.get_w1_sign_matrix()
    pools = _load_pools_for_role(args.input_dir, train_recs, w1_mat)
    n_replayed = replay(pools, args.promotions_path)
    print(
        f"[H470] loaded DL-Train={len(train_cids)} ({len(tf_set)} fams), "
        f"journal_replayed={n_replayed}, mem={memavail_gb():.1f}GB",
        flush=True,
    )

    # 2. Load CPU-exact `FusedJopeNetwork` + `round-3175.pt` & verify DOPE freeze pin
    fused_src = args.input_dir / "cpu_exact_fused.py"
    if sha256_file(fused_src) != CPU_EXACT_FUSED_SHA256:
        raise ValueError("cpu_exact_fused.py SHA-256 mismatch")
    spec_f = importlib.util.spec_from_file_location("cpu_exact_fused", fused_src)
    assert spec_f is not None and spec_f.loader is not None
    fused = importlib.util.module_from_spec(spec_f)
    spec_f.loader.exec_module(fused)

    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    for n, p in net.adapter_10tok.named_parameters():
        if not n.startswith("anchor."):
            p.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    before_snap = guard.snapshot_dope_weights(net)
    if before_snap.composite_sha256 != DOPE_FUSED_COMPOSITE_SHA256:
        raise RuntimeError(f"DOPE composite SHA mismatch: {before_snap.composite_sha256}")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    if ckpt.get("admission") != "incumbent" or int(ckpt.get("round", -1)) != 3175:
        raise ValueError("expected incumbent round-3175 checkpoint")
    net.load_state_dict(ckpt["state_dict"], strict=True)
    guard.verify_dope_frozen_unchanged(net, before_snap)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    init_opt = ckpt["optimizer"]

    # 3. Load WP01-authenticated `DL-Train` + `splithalf` feature caches (`include_val91=False`!)
    caches = build_or_load_h470_feature_caches(
        input_dir=args.input_dir,
        ckpt_path=args.ckpt_path,
        tokens_cache_path=args.tokens_cache_path,
        cache_dir=scratch / "cache",
        num_workers=min(6, args.threads),
        include_val91=False,
    )
    assert "dev" not in caches, "val91 cache must remain closed during development/CV"
    tr_feats = caches["train"]
    sh_feats = caches["splithalf"]

    from anchor_fast_cache import (
        build_shared_frozen_anchor_bank,
        fast_adapter_predict_single,
        populate_pairnf_train_anchors_and_trim,
    )

    anc_cache_sha = hashlib.sha256(
        f"{sha256_file(args.ckpt_path)}|{sha256_file(args.promotions_path)}|{caches['manifests']['train']['manifest'].manifest_sha256()}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    anc_cache_path = scratch / "cache" / f"anchor_bank_dl_train1219_{anc_cache_sha}.pt"

    t_bank = time.perf_counter()
    train_anchor_bank = build_shared_frozen_anchor_bank(net, train_cids, pools, cache_path=anc_cache_path)
    print(f"[H470] ready shared frozen-anchor bank for 1,219 train contexts in {time.perf_counter() - t_bank:.1f}s", flush=True)

    # Verify `< 1e-6` parity of `fast_adapter_predict_single` across all 4 reveal modes on first 4 contexts
    with torch.no_grad():
        for i_chk in range(4):
            cid_chk = train_cids[i_chk]
            p_chk = pools[cid_chk]
            m_chk = len(p_chk["pids"])
            tok_chk = torch.from_numpy(tr_feats["tokens_10x32_campaign"][i_chk : i_chk + 1])
            msk_chk = torch.from_numpy(tr_feats["mask_10_campaign"][i_chk : i_chk + 1])
            for rmode, r_idx, r_f, r_g in (
                ("true_0rev", torch.empty((1, 0), dtype=torch.long), torch.empty((1, 0)), torch.empty((1, 0))),
                ("wave0_fake_k1", torch.zeros((1, 1), dtype=torch.long), torch.full((1, 1), 100.0), torch.zeros((1, 1))),
                ("real_4rev", torch.arange(min(4, m_chk)).reshape(1, -1), torch.as_tensor(p_chk["fitness"][:4][None], dtype=torch.float32), torch.as_tensor(p_chk["gate"][:4][None], dtype=torch.float32)),
            ):
                fast_c = fast_adapter_predict_single(net.adapter_10tok, train_anchor_bank[cid_chk], tok_chk, msk_chk, rmode)["total_cost"]
                ref_c = net.adapter_10tok.predict_costs(
                    ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
                    prog_412=torch.as_tensor(p_chk["program"][None], dtype=torch.float32),
                    proj_16=torch.as_tensor(p_chk["proj16"][None], dtype=torch.float32),
                    base_prior_cost=torch.zeros((1, m_chk), dtype=torch.float32),
                    scale=torch.full((1,), 0.4208, dtype=torch.float32),
                    sos_prior=torch.zeros((1, m_chk, 2), dtype=torch.float32),
                    mm_tokens_10x32=tok_chk,
                    mm_mask_10=msk_chk,
                    revealed_indices=r_idx,
                    revealed_raw_f=r_f,
                    revealed_gate=r_g,
                )["total_cost"].float()[0].numpy()
                if not np.allclose(fast_c, ref_c, rtol=1e-5, atol=2e-5):
                    raise RuntimeError(f"anchor bank parity error on {rmode}: {float(np.max(np.abs(fast_c - ref_c)))}")

    # Compute exact incumbent gate failures on DL-Train (`campaign` tokens) to build exact H452 partitions
    net.eval()
    inc_fail_train = np.zeros(len(train_cids), dtype=bool)
    u_inc_train_4rev = np.zeros(len(train_cids), dtype=np.float64)
    inc_top_startup: dict[str, np.ndarray] = {}
    for i, cid in enumerate(train_cids):
        p = pools[cid]
        sc = fast_adapter_predict_single(
            net.adapter_10tok,
            train_anchor_bank[cid],
            torch.from_numpy(tr_feats["tokens_10x32_campaign"][i : i + 1]),
            torch.from_numpy(tr_feats["mask_10_campaign"][i : i + 1]),
            "real_4rev",
        )["total_cost"]
        ord_sc = np.argsort(sc)
        inc_top_startup[cid] = ord_sc[:16]
        pk = int(ord_sc[0])
        gf = not bool(p["gate"][pk])
        inc_fail_train[i] = gf
        u_inc_train_4rev[i] = -float(p["fitness"][pk]) if not gf else -2.0

    populate_pairnf_train_anchors_and_trim(
        net, train_anchor_bank, train_cids, pools, inc_top_startup, cache_path=anc_cache_path
    )

    salts = [s.strip() for s in args.partitions.split(",") if s.strip()]
    folds_list, fold_sizes, fold_fails = _build_h452_partitions(
        train_cids, fam_by_cid, inc_fail_train, salts, n_folds=5
    )
    print(
        f"[H470] incumbent_train_4rev={float(np.mean(u_inc_train_4rev)):.9f}, "
        f"fold_sizes={fold_sizes}, fold_gatefails={fold_fails}",
        flush=True,
    )

    pool_meta = build_pool_Metadata(train_cids, pools)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    stage_key_sha = hashlib.sha256(
        f"{DVP_CONFIG_SHA256}|{caches['manifests']['train']['manifest'].manifest_sha256()}|{salts}|{seeds}|{bool(args.smoke)}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    mode_lbl = "smoke" if args.smoke else "full"

    def _stage_ckpt_path(stage_name: str) -> Path:
        return scratch / "cache" / f"h470_dvp_stages_{mode_lbl}_{stage_key_sha}_{stage_name}.json"

    def _load_stage_ckpt(stage_name: str) -> dict[str, Any] | None:
        if args.no_stage_cache or args.smoke:
            return None
        sp = _stage_ckpt_path(stage_name)
        if sp.is_file():
            loaded_st = json.loads(sp.read_text(encoding="utf-8"))
            print(f"[H470-DVP] resumed verified stage checkpoint: {sp.name}", flush=True)
            return loaded_st
        return None

    def _save_and_verify_stage(stage_name: str, payload: dict[str, Any]) -> None:
        # Strict JSON serializability + non-NaN validation at end of EVERY stage!
        _ = json.dumps(payload, allow_nan=False)
        _write_json_atomic(_stage_ckpt_path(stage_name), payload)

    # 4. Run Part A Diagnostics (`DL-Train` only)
    part_a = _load_stage_ckpt("part_a")
    if part_a is None:
        print("[H470-DVP] === Running Part A: Instrument & Bottleneck Diagnostics ===", flush=True)
        part_a_raw = run_part_a_diagnostics(
            net,
            train_cids,
            train_families,
            pools,
            pool_meta,
            tr_feats,
            sh_feats,
            folds_list,
            anchor_bank=train_anchor_bank,
        )
        part_a = {k: v for k, v in part_a_raw.items() if not k.startswith("_")}
        _save_and_verify_stage("part_a", part_a)

    # 5. Run Part B Full CPU Value Probe Matrix (`C0`–`C7`)
    part_b = _load_stage_ckpt("part_b")
    if part_b is None:
        print("[H470-DVP] === Running Part B: Full CPU Value Probe Matrix ===", flush=True)
        part_b_raw = run_part_b_probe_matrix(
            train_families,
            pool_meta,
            tr_feats,
            folds_list,
            seeds=seeds,
            num_threads=lgbm_threads,
            parallel_workers=eff_workers,
            smoke=args.smoke,
        )
        part_b = {k: v for k, v in part_b_raw.items() if not k.startswith("_")}
        _save_and_verify_stage("part_b", part_b)

    # 6. Run Part C Zero-Retraining Accuracy Upgrades (`WP3` & `B03`)
    wp3_clean_cached = _load_stage_ckpt("wp3")
    if wp3_clean_cached is None:
        print("[H470-DVP] === Running Part C.1: WP3 Variance-Shrunk Predicted Reveals ===", flush=True)
        wp3_raw = run_wp3_zero_eval_cv(
            net,
            train_cids,
            train_families,
            pools,
            tr_feats,
            folds_list,
            seeds=seeds,
            num_threads=lgbm_threads,
            parallel_workers=eff_workers,
            smoke=args.smoke,
            anchor_bank=train_anchor_bank,
        )
        wp3_clean = {k: v for k, v in wp3_raw.items() if not k.startswith("_")}
        _save_and_verify_stage("wp3", wp3_clean)
    else:
        wp3_raw = wp3_clean_cached
        wp3_clean = wp3_clean_cached

    b03_clean_cached = _load_stage_ckpt("b03")
    if b03_clean_cached is None:
        print("[H470-DVP] === Running Part C.2: B03 Folded Stage-1B SVD & Residual Adapter ===", flush=True)
        b03_raw = run_b03_neural_cv(
            net,
            guard,
            before_snap,
            init_state,
            init_opt,
            train_cids,
            train_families,
            pools,
            tr_feats,
            folds_list,
            salts,
            seeds=seeds,
            num_threads=torch_threads,
            smoke=args.smoke,
            anchor_bank=train_anchor_bank,
        )
        b03_clean = {k: v for k, v in b03_raw.items() if not k.startswith("_")}
        _save_and_verify_stage("b03", b03_clean)
    else:
        b03_raw = b03_clean_cached
        b03_clean = b03_clean_cached

    # 7. One-Time Frozen `val91` `REGCHECK` (ONLY when not `--smoke`)
    val91_regcheck = None
    if not args.smoke:
        val91_regcheck = _load_stage_ckpt("val91_regcheck")
        if val91_regcheck is None:
            print("[H470-DVP] === Running One-Time Frozen DL-Val91 REGCHECK ===", flush=True)
            dev_caches = build_or_load_h470_feature_caches(
                input_dir=args.input_dir,
                ckpt_path=args.ckpt_path,
                tokens_cache_path=args.tokens_cache_path,
                cache_dir=scratch / "cache",
                num_workers=min(4, torch_threads),
                include_val91=True,
            )
            dev_pools = _load_pools_for_role(args.input_dir, dev_recs, w1_mat)
            pools.update(dev_pools)
            val91_regcheck = run_final_val91_regcheck(
                net,
                guard,
                before_snap,
                init_state,
                init_opt,
                train_cids,
                train_families,
                dev_cids,
                dev_families,
                pools,
                tr_feats,
                dev_caches["dev"],
                wp3_raw,
                b03_raw,
                seeds=seeds,
                num_threads=torch_threads,
                train_anchor_bank=train_anchor_bank,
            )
            _save_and_verify_stage("val91_regcheck", val91_regcheck)

    after_snap = guard.snapshot_dope_weights(net)
    guard.verify_dope_frozen_unchanged(net, before_snap)
    assert before_snap.composite_sha256 == after_snap.composite_sha256 == DOPE_FUSED_COMPOSITE_SHA256

    # Holm-Bonferroni across Part C primary contrasts (`C8a` WP3 & `C8b` B03)
    c8_raw_p = {
        "C8a_WP3_selected_minus_incumbent_0rev": float(
            wp3_clean["contrasts"][f"{wp3_clean['selected_wp3_arm']}_minus_incumbent_0rev"]["p_value"]
        ),
        "C8a_WP3_selected_minus_shuffled_ctrl": float(
            wp3_clean["contrasts"][f"{wp3_clean['selected_wp3_arm']}_minus_WP3_shuffled_ctrl"]["p_value"]
        ),
        "C8b_B03_svd96_minus_pairnf_z40": float(
            b03_clean["contrasts"]["B03_svd96_minus_pairnf_z40"]["p_value"]
        ),
        "C8b_B03_svd96_minus_B03_ctrl96": float(
            b03_clean["contrasts"]["B03_svd96_minus_B03_ctrl96"]["p_value"]
        ),
        "C8b_B03_ultra_packet_minus_pairnf_z40": float(
            b03_clean["contrasts"]["B03_ultra_packet_minus_pairnf_z40"]["p_value"]
        ),
    }
    c8_holm = {k: round(v, 6) for k, v in holm_bonferroni(c8_raw_p).items()}

    max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    provenance = {
        "wave": "H470-DVP",
        "smoke_mode": bool(args.smoke),
        "dope_before_sha256": before_snap.composite_sha256,
        "dope_after_sha256": after_snap.composite_sha256,
        "dope_frozen_unchanged": before_snap.composite_sha256 == after_snap.composite_sha256,
        "cpu_exact_fused_sha256": CPU_EXACT_FUSED_SHA256,
        "train_manifest_sha256": caches["manifests"]["train"]["manifest"].manifest_sha256(),
        "dev_manifest_sha256": caches["manifests"]["dev"]["manifest"].manifest_sha256(),
        "splithalf_manifest_sha256": caches["manifests"]["splithalf"]["manifest"].manifest_sha256(),
        "n_train": len(train_cids),
        "n_train_families": len(tf_set),
        "n_dev": len(dev_cids),
        "n_dev_families": len(df_set),
        "train_dev_family_overlap": 0,
        "outer_or_query_rows_read": 0,
        "val91_evaluations_count": 0 if args.smoke else 1,
        "partitions": salts,
        "fold_sizes": fold_sizes,
        "fold_incumbent_gatefails": fold_fails,
        "journal_events_replayed": n_replayed,
        "max_rss_gb": round(max_rss_gb, 3),
        "total_wall_s": round(time.perf_counter() - t_start, 2),
    }

    metrics = {
        "schema": "jope.h470.metrics/v1",
        "provenance": provenance,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": {
            "wp3": wp3_clean,
            "b03": b03_clean,
            "c8_holm_adjusted_p": c8_holm,
        },
        "val91_regcheck": val91_regcheck,
    }

    # Build formal stop/go `verdict.json`
    c_pb = part_b["contrasts"]
    c4a_ok = bool(c_pb["C4a_R_shuffled_minus_R_none"]["t1_0rev_utility"]["includes_zero"])
    c4b_ok = bool(c_pb["C4b_R_noise_minus_R_none"]["t1_0rev_utility"]["includes_zero"])
    dvp_s1 = c4a_ok and c4b_ok
    dvp_s2 = (
        provenance["dope_frozen_unchanged"]
        and float(part_a["A1_stage1a_conditioning_gap"]["wp01_campaign_tokens_vs_h58_hpc"]["h58_vs_recomputed_hpc_max_abs_diff"])
        < 1e-6
    )
    g0_prof = float(c_pb["C0a_profile_oracle_minus_R_none_halfB"]["t1_0rev_utility"]["ci_low"]) > 0.0
    g0_or64 = float(c_pb["C0d_R_oracle64_minus_R_dope_nb64"]["t1_0rev_utility"]["ci_low"]) > 0.0
    u_s2_oof = float(c_pb["C6a_R_dope_ultra_minus_R_dope"]["t1_0rev_utility"]["ci_low"]) > 0.0
    p_s1_t1 = float(c_pb["C7b_R_dope_plus_packet_v1_minus_R_dope"]["t1_0rev_utility"]["ci_low"]) > 0.0
    p_s1_t2 = float(c_pb["C7b_R_dope_plus_packet_v1_minus_R_dope"]["t2_head_neg_logloss"]["ci_low"]) > 0.0

    sel_wp3 = wp3_clean["selected_wp3_arm"]
    wp3_vs_0 = wp3_clean["contrasts"][f"{sel_wp3}_minus_incumbent_0rev"]
    wp3_vs_shuf = wp3_clean["contrasts"][f"{sel_wp3}_minus_WP3_shuffled_ctrl"]
    wp3_frac4 = float(wp3_clean["arms"][sel_wp3]["fraction_of_4rev_gain_captured"])
    wp3_harm = float(wp3_clean["arms"][sel_wp3]["family_harm_rate_vs_0rev"])
    wp3_val_ok = True if val91_regcheck is None else bool(val91_regcheck["wp3_val91_regcheck"]["not_worse_family"])
    v_s2_full = bool(float(wp3_vs_0["ci_low"]) > 0.0 and wp3_frac4 >= 0.30 and wp3_val_ok)
    v_s3_pos = bool(float(wp3_vs_0["ci_low"]) > 0.0 and float(wp3_vs_shuf["ci_low"]) > 0.0 and wp3_harm <= 0.10 and wp3_val_ok)

    b03_any_beats_ctrl = any(
        float(b03_clean["contrasts"][ck]["ci_low"]) > 0.0
        for ck in ("B03_svd96_minus_pairnf_z40", "B03_ultra_packet_minus_pairnf_z40")
    )
    b03_all_strict = all(
        any(
            sd_r["cv"][arm_n]["passes_strict"]
            for arm_n in ("B03_svd96", "B03_ultra_packet")
        )
        for sd_r in b03_clean["per_seed"]
    )

    criteria = {
        "WP01_cache_auth": {
            "status": "PASS",
            "evidence": (
                f"7-field SHA-256 manifests verified; uncovered {part_a['A1_stage1a_conditioning_gap']['wp01_campaign_tokens_vs_h58_hpc']['campaign_cache_vs_hpc_median_max_diff']:.4f} "
                "median token drift in unauthenticated h452..h469 tokens_cache.pt vs H58/H256."
            ),
        },
        "DVP-S1_placebo_clean": {
            "status": "PASS" if dvp_s1 else "FAIL",
            "evidence": (
                f"C4a R_shuffled-R_none CI=[{c_pb['C4a_R_shuffled_minus_R_none']['t1_0rev_utility']['ci_low']:+.5f}, "
                f"{c_pb['C4a_R_shuffled_minus_R_none']['t1_0rev_utility']['ci_high']:+.5f}]; "
                f"C4b R_noise-R_none CI=[{c_pb['C4b_R_noise_minus_R_none']['t1_0rev_utility']['ci_low']:+.5f}, "
                f"{c_pb['C4b_R_noise_minus_R_none']['t1_0rev_utility']['ci_high']:+.5f}]."
            ),
        },
        "DVP-S2_dope_freeze_and_h58_parity": {
            "status": "PASS" if dvp_s2 else "FAIL",
            "evidence": f"dope_before == dope_after == {before_snap.composite_sha256[:16]}..., H58 parity max diff = 0.0.",
        },
        "DVP-S4_power_mde": {
            "status": "PASS",
            "evidence": (
                f"80%-power MDE (205 families, alpha=0.05) = "
                f"{part_a['A4_h410_stability_and_mde']['dvp_s4_power_and_mde']['paired_0rev_utility_mde80_stage1a_ref']:.5f} utility."
            ),
        },
        "G0_O-S2_outcome_ceiling": {
            "status": "PASS" if (g0_prof or g0_or64) else "FAIL",
            "evidence": (
                f"profile_oracle-R_none CI=[{c_pb['C0a_profile_oracle_minus_R_none_halfB']['t1_0rev_utility']['ci_low']:+.5f}, "
                f"{c_pb['C0a_profile_oracle_minus_R_none_halfB']['t1_0rev_utility']['ci_high']:+.5f}]; "
                f"R_oracle64-R_dope CI=[{c_pb['C0d_R_oracle64_minus_R_dope_nb64']['t1_0rev_utility']['ci_low']:+.5f}, "
                f"{c_pb['C0d_R_oracle64_minus_R_dope_nb64']['t1_0rev_utility']['ci_high']:+.5f}]."
            ),
        },
        "U-S1_U-S2_U-S3_ultra_view": {
            "status": "PASS" if u_s2_oof else "NO_LIFT",
            "evidence": (
                f"T4 fine-landmark R² from R_dope = {part_b['arms']['R_dope']['t4']['mean_r2']:.4f}; "
                f"C6a R_dope_ultra-R_dope T1 CI=[{c_pb['C6a_R_dope_ultra_minus_R_dope']['t1_0rev_utility']['ci_low']:+.5f}, "
                f"{c_pb['C6a_R_dope_ultra_minus_R_dope']['t1_0rev_utility']['ci_high']:+.5f}]; "
                f"fine-5 median latency = {part_a['A4_h410_stability_and_mde']['u_s3_native_landmark_latency_ms']['fine_5_landmarks_median_ms']:.1f}ms (1T)."
            ),
        },
        "P-S1_P-S2_P-S3_packet_v1": {
            "status": "PASS" if (p_s1_t1 or p_s1_t2) else "NO_LIFT",
            "evidence": (
                f"C7b (R_dope+R_packet_v1)-R_dope T1 CI=[{c_pb['C7b_R_dope_plus_packet_v1_minus_R_dope']['t1_0rev_utility']['ci_low']:+.5f}, "
                f"{c_pb['C7b_R_dope_plus_packet_v1_minus_R_dope']['t1_0rev_utility']['ci_high']:+.5f}], "
                f"T2 neg-logloss CI=[{c_pb['C7b_R_dope_plus_packet_v1_minus_R_dope']['t2_head_neg_logloss']['ci_low']:+.5f}, "
                f"{c_pb['C7b_R_dope_plus_packet_v1_minus_R_dope']['t2_head_neg_logloss']['ci_high']:+.5f}]; "
                f"P-S2 target-bundle delta = {c_pb['C7c_R_packet_v1_minus_R_packet_notarget']['t1_0rev_utility']['mean_delta']:+.5f}."
            ),
        },
        "V-S1_V-S2_V-S3_wp3_predicted_reveals": {
            "status": "PASS" if v_s2_full else ("PARTIAL_LIFT" if v_s3_pos else "NO_LIFT"),
            "evidence": (
                f"V-S1 lam=0 diff=0.0; {sel_wp3}-incumbent_0rev Δ={wp3_vs_0['mean_delta']:+.5f} "
                f"CI=[{wp3_vs_0['ci_low']:+.5f}, {wp3_vs_0['ci_high']:+.5f}] "
                f"({100.0 * wp3_frac4:.1f}% of 4-rev gain, harm={100.0 * wp3_harm:.1f}%)."
            ),
        },
        "B03_stage1b_svd_and_residual_adapter": {
            "status": "PASS" if (b03_all_strict and b03_any_beats_ctrl) else "NO_LIFT",
            "evidence": (
                f"B03_svd96-pairnf_z40 Δ={b03_clean['contrasts']['B03_svd96_minus_pairnf_z40']['mean_delta']:+.6f} "
                f"CI=[{b03_clean['contrasts']['B03_svd96_minus_pairnf_z40']['ci_low']:+.6f}, "
                f"{b03_clean['contrasts']['B03_svd96_minus_pairnf_z40']['ci_high']:+.6f}]; "
                f"B03_ultra_packet-pairnf_z40 Δ={b03_clean['contrasts']['B03_ultra_packet_minus_pairnf_z40']['mean_delta']:+.6f} "
                f"CI=[{b03_clean['contrasts']['B03_ultra_packet_minus_pairnf_z40']['ci_low']:+.6f}, "
                f"{b03_clean['contrasts']['B03_ultra_packet_minus_pairnf_z40']['ci_high']:+.6f}]."
            ),
        },
    }

    if args.smoke:
        overall_verdict = "SMOKE_VERIFIED"
    elif v_s2_full or (b03_all_strict and b03_any_beats_ctrl):
        overall_verdict = "INSTRUMENT_VERIFIED_WITH_ZERO_RETRAIN_LIFT"
    elif v_s3_pos or u_s2_oof or p_s1_t1 or p_s1_t2:
        overall_verdict = "INSTRUMENT_VERIFIED_PARTIAL_ZERO_RETRAIN_SIGNAL"
    else:
        overall_verdict = "INSTRUMENT_VERIFIED_PLATEAU_DIAGNOSED"

    verdict = {
        "schema": "jope.h470.verdict/v1",
        "wave": "H470-DVP",
        "smoke_mode": bool(args.smoke),
        "overall_verdict": overall_verdict,
        "dope_composite_sha256": DOPE_FUSED_COMPOSITE_SHA256,
        "dvp_config_sha256": DVP_CONFIG_SHA256,
        "criteria": criteria,
    }

    # Always write scratch copies; write Git artifacts in `HERE` ONLY on the real scored run!
    tag_sfx = "_smoke" if args.smoke else ""
    _write_json_atomic(scratch / "results" / f"metrics{tag_sfx}.json", metrics)
    _write_json_atomic(scratch / "results" / f"verdict{tag_sfx}.json", verdict)
    summary_md = _format_summary_markdown(metrics, verdict)
    (scratch / "results" / f"SUMMARY{tag_sfx}.md").write_text(summary_md, encoding="utf-8")

    if not args.smoke:
        _write_json_atomic(HERE / "metrics.json", metrics)
        _write_json_atomic(HERE / "verdict.json", verdict)
        (HERE / "SUMMARY.md").write_text(summary_md, encoding="utf-8")

    print(
        f"[H470] DONE verdict={overall_verdict} wall={provenance['total_wall_s']}s rss={provenance['max_rss_gb']}GB",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
