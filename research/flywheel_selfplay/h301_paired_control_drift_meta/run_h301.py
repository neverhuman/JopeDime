"""H301 (h301_paired_control_drift_meta): execute analysis-only meta-study on xbabe2."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_h301 as a301  # noqa: E402

CACHE_DIR = Path("/home/ubuntu/jope_plan/ns/receipts/h301_s3_json_cache")
T1820_DIR = Path("/home/ubuntu/jope_plan/evidence/fw-plateau-t1820-20261102")
T1845_DIR = Path("/home/ubuntu/jope_plan/evidence/fw-plateau-t1845-20261102")
T2115_DIR = Path("/home/ubuntu/jope_plan/evidence/fw-plateau-t2115-20261102")
MAX_RSS_BYTES = 2 * 1024**3


def install_analysis_only_guard() -> None:
    """Enforce no neural framework imports, no checkpoint/table opens, no forbidden cohort reads."""

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event == "import" and args and isinstance(args[0], str):
            mod = args[0].split(".")[0]
            if mod in ("torch", "tensorflow", "jax"):
                raise RuntimeError(f"Forbidden neural framework import in H301 analysis-only wave: {mod}")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0]).lower()
            if any(
                k in p
                for k in (
                    "fresh_cohort",
                    "fresh_pools",
                    "/sealed",
                    "/holdout",
                    "/dl-test",
                    "/dl_test",
                    "meta_test",
                    "meta-test",
                )
            ) or p.endswith((".pt", ".jtt1", ".jtt1.gz")):
                raise RuntimeError(f"Forbidden file open in H301 analysis-only wave: {p}")

    sys.addaudithook(audit_hook)


def check_rss_limit() -> int:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss_kb) * 1024
    if rss_bytes > MAX_RSS_BYTES:
        raise RuntimeError(f"Peak RSS {rss_bytes} exceeds 2 GiB cap ({MAX_RSS_BYTES})")
    return rss_bytes


def run_meta_analysis() -> dict[str, Any]:
    t0 = time.time()
    install_analysis_only_guard()

    # 1. Verify Step 1 Regression Receipt
    reg_path = HERE / "regression_receipt.json"
    assert reg_path.exists(), "Step 1 regression_receipt.json must exist before running meta-analysis"
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"
    assert reg["h297_exact_s3_get"]["bit_for_bit_match_all_arms"] is True
    assert reg["h298_exact_s3_get"]["bit_for_bit_match_all_arms"] is True

    # 2. Load selected.json dev91 metadata and r3175 baseline
    sel_path = CACHE_DIR / "selected.json"
    assert hashlib.sha256(sel_path.read_bytes()).hexdigest() == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    sel = json.loads(sel_path.read_text())
    dev_families = [r["family_id"] for r in sel["dev"]]
    assert len(dev_families) == 91 and len(set(dev_families)) == 19

    r3175_base = json.loads((CACHE_DIR / "TRIAL_BASELINE.json").read_text())
    assert r3175_base["mean"] == a301.R3175_DEV91_MEAN
    r3175_u = np.asarray(r3175_base["per_context"], dtype=np.float64)
    r3175_progs = list(r3175_base["selected_programs"])

    counts = a301.make_family_bootstrap_counts(19, a301.BOOTSTRAP_DRAWS, a301.BOOTSTRAP_SEED)

    # 3. Load H289 history.npz for t1820 (rounds 3176..3814) and post-r3175 flywheel val history (rounds 3176..4675)
    with np.load(CACHE_DIR / "h289/history.npz", allow_pickle=False) as z:
        hist_rounds = z["rounds"]
        hist_utils = z["utilities"]
        hist_progs = z["programs"]
    assert int(hist_rounds[0]) == 3175
    assert np.allclose(hist_utils[0], r3175_u, rtol=0.0, atol=1e-15)

    # Cross-link t1820 (rounds 3176..3814, 639 2-update baseline-loss proposals from r3175)
    t1820_mask = (hist_rounds >= 3176) & (hist_rounds <= 3814)
    t1820_rounds = hist_rounds[t1820_mask]
    t1820_utils = hist_utils[t1820_mask]
    t1820_progs = hist_progs[t1820_mask]
    assert len(t1820_rounds) == 639

    t1820_means = np.mean(t1820_utils, axis=1)
    t1820_chgs = np.sum(t1820_progs != hist_progs[0][None, :], axis=1)
    # Identify median round of t1820 (exact modal/median utility 1.1354387884248622, first occurrence round 3177)
    t1820_med_val = float(np.median(t1820_means))
    t1820_med_idx = int(np.flatnonzero(t1820_means == t1820_med_val)[0])
    t1820_med_round = int(t1820_rounds[t1820_med_idx])

    t1820_med_traj = a301.evaluate_trajectory_vs_r3175(
        wave="t1820",
        label=f"flywheel_2step_median_r{t1820_med_round}_of_639",
        seed=t1820_med_round,
        n_updates=2,
        lr=None,
        cand_per_context=t1820_utils[t1820_med_idx],
        cand_programs=t1820_progs[t1820_med_idx].tolist(),
        r3175_per_context=r3175_u,
        r3175_programs=r3175_progs,
        dev_families=dev_families,
        counts=counts,
    )
    t1820_mean_traj = a301.evaluate_trajectory_vs_r3175(
        wave="t1820",
        label="flywheel_2step_mean_of_639_rounds_3176_3814",
        seed="3176..3814",
        n_updates=2,
        lr=None,
        cand_per_context=np.mean(t1820_utils, axis=0),
        cand_programs=None,
        r3175_per_context=r3175_u,
        r3175_programs=r3175_progs,
        dev_families=dev_families,
        counts=counts,
    )
    t1820_mean_traj["changed_contexts"] = int(round(float(np.median(t1820_chgs))))

    # 4. Collect t1845 control_v2 (seeds 184501, 184502, 184503; n_updates=2)
    t1845_trajs = []
    for s in (184501, 184502, 184503):
        rec = json.loads((CACHE_DIR / f"control_v2-{s}.json").read_text())
        t1845_trajs.append(
            a301.evaluate_trajectory_vs_r3175(
                wave="t1845",
                label=f"control_v2-{s}",
                seed=s,
                n_updates=2,
                lr=float(rec["lr"]),
                cand_per_context=rec["per_context"],
                cand_programs=rec["selected_programs"],
                r3175_per_context=r3175_u,
                r3175_programs=r3175_progs,
                dev_families=dev_families,
                counts=counts,
            )
        )

    # Verify H297 and H298 2-update control seed 184501 regression replays match t1845-184501 bit-for-bit
    h297_reg184501 = json.loads((CACHE_DIR / "h297/regression-control.json").read_text())
    assert h297_reg184501["mean"] == t1845_trajs[0]["dev91_mean"] == 1.1354387884248622
    assert h297_reg184501["selected_programs"] == json.loads((CACHE_DIR / "control_v2-184501.json").read_text())["selected_programs"]

    # 5. Collect H297 CONTROL (seeds 29701, 29702, 29703; n_updates=8)
    h297_plan = json.loads((CACHE_DIR / "h297/plan.json").read_text())
    h297_trajs = []
    for s in (29701, 29702, 29703):
        rec = json.loads((CACHE_DIR / f"h297/CONTROL-{s}-result.json").read_text())
        h297_trajs.append(
            a301.evaluate_trajectory_vs_r3175(
                wave="H297",
                label=f"CONTROL-{s}",
                seed=s,
                n_updates=8,
                lr=float(h297_plan["lr"]),
                cand_per_context=rec["per_context"],
                cand_programs=rec["selected_programs"],
                r3175_per_context=r3175_u,
                r3175_programs=r3175_progs,
                dev_families=dev_families,
                counts=counts,
            )
        )

    # 6. Collect H298 PAIRED_CONTROL (seeds 29801, 29802, 29803; n_updates=8)
    h298_trajs = []
    for s in (29801, 29802, 29803):
        rec = json.loads((CACHE_DIR / f"h298/PAIRED_CONTROL-{s}.json").read_text())
        h298_trajs.append(
            a301.evaluate_trajectory_vs_r3175(
                wave="H298",
                label=f"PAIRED_CONTROL-{s}",
                seed=s,
                n_updates=8,
                lr=float(rec["lr"]),
                cand_per_context=rec["per_context"],
                cand_programs=rec["selected_programs"],
                r3175_per_context=r3175_u,
                r3175_programs=r3175_progs,
                dev_families=dev_families,
                counts=counts,
            )
        )

    # 7. Collect t2115 baseline-loss sweep trajectories from r3175 (steps 16, 64, 256 x LRs 2.5e-5, 1e-4, 3e-4)
    t2115_by_steps: dict[int, list[dict[str, Any]]] = {16: [], 64: [], 256: []}
    for steps in (16, 64, 256):
        for lr_str in ("2.5e-05", "0.0001", "0.0003"):
            rec = json.loads((T2115_DIR / f"result-lr{lr_str}-step{steps}.json").read_text())
            t2115_by_steps[steps].append(
                a301.evaluate_trajectory_vs_r3175(
                    wave=f"t2115_step{steps}",
                    label=f"lr{lr_str}-step{steps}",
                    seed=f"lr={lr_str}",
                    n_updates=steps,
                    lr=float(rec["lr"]),
                    cand_per_context=rec["per_context"],
                    cand_programs=rec["selected_programs"],
                    r3175_per_context=r3175_u,
                    r3175_programs=r3175_progs,
                    dev_families=dev_families,
                    counts=counts,
                )
            )

    # Summarize wave/horizon groups
    wave_groups_short = [
        a301.summarize_wave_group("t1820", 2, [t1820_med_traj]),
        a301.summarize_wave_group("t1845", 2, t1845_trajs),
        a301.summarize_wave_group("H297", 8, h297_trajs),
        a301.summarize_wave_group("H298", 8, h298_trajs),
    ]
    wave_groups_all = wave_groups_short + [
        a301.summarize_wave_group("t2115_step16", 16, t2115_by_steps[16]),
        a301.summarize_wave_group("t2115_step64", 64, t2115_by_steps[64]),
        a301.summarize_wave_group("t2115_step256", 256, t2115_by_steps[256]),
    ]

    # Compute median-over-waves control lift + 19-family 10k bootstrap CI95
    short_lifts = np.array([w["median_lift_vs_r3175"] for w in wave_groups_short], dtype=np.float64)
    short_boot = np.median(np.stack([w["_boot_median_lift"] for w in wave_groups_short], axis=0), axis=0)
    short_ci95 = np.quantile(short_boot, [0.025, 0.975]).tolist()
    short_elos = np.array([w["median_elo_vs_r3175"] for w in wave_groups_short], dtype=np.float64)
    short_elo_boot = np.median(np.stack([w["_boot_median_elo"] for w in wave_groups_short], axis=0), axis=0)
    short_elo_ci95 = np.quantile(short_elo_boot, [0.025, 0.975]).tolist()

    all_lifts = np.array([w["median_lift_vs_r3175"] for w in wave_groups_all], dtype=np.float64)
    all_boot = np.median(np.stack([w["_boot_median_lift"] for w in wave_groups_all], axis=0), axis=0)
    all_ci95 = np.quantile(all_boot, [0.025, 0.975]).tolist()
    all_elos = np.array([w["median_elo_vs_r3175"] for w in wave_groups_all], dtype=np.float64)
    all_elo_boot = np.median(np.stack([w["_boot_median_elo"] for w in wave_groups_all], axis=0), axis=0)
    all_elo_ci95 = np.quantile(all_elo_boot, [0.025, 0.975]).tolist()

    # (a) Slope of lift vs n_updates (and vs log2(n_updates))
    all_indiv_trajs = [t1820_med_traj] + t1845_trajs + h297_trajs + h298_trajs + t2115_by_steps[16] + t2115_by_steps[64] + t2115_by_steps[256]
    indiv_lifts = np.array([t["lift_vs_r3175"] for t in all_indiv_trajs], dtype=np.float64)
    indiv_boot = np.median(np.stack([t["_boot_lift"] for t in all_indiv_trajs], axis=0), axis=0)
    indiv_ci95 = np.quantile(indiv_boot, [0.025, 0.975]).tolist()
    indiv_elos = np.array([t["elo_vs_r3175"]["point"] for t in all_indiv_trajs], dtype=np.float64)
    indiv_elo_boot = np.median(np.stack([t["_boot_elo"] for t in all_indiv_trajs], axis=0), axis=0)
    indiv_elo_ci95 = np.quantile(indiv_elo_boot, [0.025, 0.975]).tolist()

    x_wave_all = np.array([w["n_updates"] for w in wave_groups_all], dtype=np.float64)
    y_wave_all = np.array([w["median_lift_vs_r3175"] for w in wave_groups_all], dtype=np.float64)
    yb_wave_all = np.stack([w["_boot_median_lift"] for w in wave_groups_all], axis=0)

    x_wave_short = np.array([w["n_updates"] for w in wave_groups_short], dtype=np.float64)
    y_wave_short = np.array([w["median_lift_vs_r3175"] for w in wave_groups_short], dtype=np.float64)
    yb_wave_short = np.stack([w["_boot_median_lift"] for w in wave_groups_short], axis=0)

    x_indiv_all = np.array([t["n_updates"] for t in all_indiv_trajs], dtype=np.float64)
    y_indiv_all = np.array([t["lift_vs_r3175"] for t in all_indiv_trajs], dtype=np.float64)
    yb_indiv_all = np.stack([t["_boot_lift"] for t in all_indiv_trajs], axis=0)

    slope_wave_linear_all = a301.fit_drift_slope_bootstrap(x_wave_all, y_wave_all, yb_wave_all)
    slope_wave_log2_all = a301.fit_drift_slope_bootstrap(np.log2(x_wave_all), y_wave_all, yb_wave_all)
    slope_wave_linear_short = a301.fit_drift_slope_bootstrap(x_wave_short, y_wave_short, yb_wave_short)
    slope_indiv_linear_all = a301.fit_drift_slope_bootstrap(x_indiv_all, y_indiv_all, yb_indiv_all)

    # (b) Winner's-Curse Analysis on Flywheel Val History
    post_mask = hist_rounds > 3175
    post_rounds = hist_rounds[post_mask]
    post_utils = hist_utils[post_mask]

    adm_utils = []
    for line in (CACHE_DIR / "adapter_admissions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("evaluation_identity") == a301.EVAL_IDENTITY and d.get("candidate_utility") is not None:
            u_val = float(d["candidate_utility"])
            if np.isfinite(u_val):
                adm_utils.append(u_val)
    adm_utils_arr = np.asarray(adm_utils, dtype=np.float64)

    wc = a301.analyze_winners_curse(
        r3175_per_context=r3175_u,
        post_r3175_utilities=post_utils,
        post_r3175_rounds=post_rounds,
        all_admission_utilities=adm_utils_arr,
        dev_families=dev_families,
        counts=counts,
    )

    # (c) Strict Decision Gate (PREREG Section 4c)
    # Criterion 1: Control lift < 0 with CI95 excluding 0 on the median over waves (primary: all pre-declared wave/horizon groups)
    crit1_all = bool(float(np.median(all_lifts)) < 0.0 and all_ci95[1] < 0.0)
    crit1_indiv = bool(float(np.median(indiv_lifts)) < 0.0 and indiv_ci95[1] < 0.0)
    crit1_short = bool(float(np.median(short_lifts)) < 0.0 and short_ci95[1] < 0.0)
    # Criterion 2: r3175 exceeds the 95th percentile of round-to-round val distribution by < 2 x its family-bootstrap SE
    crit2_marginal = bool(wc["post_r3175_2update_distribution"]["exceeds_q95_by_lt_2_marginal_se"])
    crit2_paired = bool(wc["post_r3175_2update_distribution"]["exceeds_q95_by_lt_2_paired_se"])

    if crit1_all and crit2_marginal and crit2_paired:
        verdict = "SELECTION_NOISE_PLATEAU"
    elif crit1_all and not crit2_marginal:
        verdict = "OPTIMIZER_PLATEAU"
    else:
        verdict = "INCONCLUSIVE"

    def clean_dict(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    peak_rss = check_rss_limit()

    payload = {
        "wave_id": "H301",
        "slug": "h301_paired_control_drift_meta",
        "owner": "rd_argon",
        "status": "COMPLETE",
        "verdict": verdict,
        "prereg_commit": "1dfbc2736e580288f7dbef0e13b1bdb93a986b1e",
        "elapsed_sec": round(time.time() - t0, 3),
        "mode": "ANALYSIS_ONLY",
        "neural_updates": 0,
        "neural_forwards": 0,
        "dope_sha256": a301.DOPE_SHA256,
        "dope_untouched": True,
        "evaluation_identity": a301.EVAL_IDENTITY,
        "r3175_baseline_mean": a301.R3175_DEV91_MEAN,
        "regression_step1": reg,
        "unclosed_waves_at_execution": {
            "H296": "PREREGISTERED_NOT_EXECUTED on origin/v1/next at execution time",
        },
        "median_over_waves_control_summary": {
            "all_waves_2_to_256_updates": {
                "waves": ["t1820", "t1845", "H297", "H298", "t2115_step16", "t2115_step64", "t2115_step256"],
                "median_lift_vs_r3175": float(np.median(all_lifts)),
                "ci95_family_bootstrap": [float(all_ci95[0]), float(all_ci95[1])],
                "prob_negative_bootstrap": float(np.mean(all_boot < 0.0)),
                "ci95_excludes_zero_strictly_negative": crit1_all,
                "median_elo_vs_r3175": float(np.median(all_elos)),
                "elo_ci95_family_bootstrap": [float(all_elo_ci95[0]), float(all_elo_ci95[1])],
            },
            "all_19_trajectories_pooled_median": {
                "n_trajectories": len(all_indiv_trajs),
                "median_lift_vs_r3175": float(np.median(indiv_lifts)),
                "ci95_family_bootstrap": [float(indiv_ci95[0]), float(indiv_ci95[1])],
                "prob_negative_bootstrap": float(np.mean(indiv_boot < 0.0)),
                "ci95_excludes_zero_strictly_negative": crit1_indiv,
                "median_elo_vs_r3175": float(np.median(indiv_elos)),
                "elo_ci95_family_bootstrap": [float(indiv_elo_ci95[0]), float(indiv_elo_ci95[1])],
            },
            "short_horizon_waves_2_to_8_updates": {
                "waves": ["t1820", "t1845", "H297", "H298"],
                "median_lift_vs_r3175": float(np.median(short_lifts)),
                "ci95_family_bootstrap": [float(short_ci95[0]), float(short_ci95[1])],
                "prob_negative_bootstrap": float(np.mean(short_boot < 0.0)),
                "ci95_excludes_zero_strictly_negative": crit1_short,
                "median_elo_vs_r3175": float(np.median(short_elos)),
                "elo_ci95_family_bootstrap": [float(short_elo_ci95[0]), float(short_elo_ci95[1])],
            },
        },
        "drift_slope_vs_n_updates": {
            "wave_medians_linear_per_update_2_to_256": slope_wave_linear_all,
            "wave_medians_per_doubling_log2_updates_2_to_256": slope_wave_log2_all,
            "wave_medians_linear_per_update_2_to_8": slope_wave_linear_short,
            "all_individual_trajectories_linear_per_update_2_to_256": slope_indiv_linear_all,
        },
        "winners_curse_analysis": wc,
        "decision_gate": {
            "criterion_1_median_control_lift_lt_0_ci95_excludes_0_all_waves": crit1_all,
            "criterion_1_median_control_lift_lt_0_ci95_excludes_0_all_19_trajectories": crit1_indiv,
            "criterion_1_sensitivity_short_waves_2_to_8_updates": crit1_short,
            "criterion_2_r3175_minus_q95_lt_2_marginal_family_se": crit2_marginal,
            "criterion_2_r3175_minus_q95_lt_2_paired_family_se": crit2_paired,
            "verdict": verdict,
        },
        "wave_horizon_summaries": [clean_dict(w) for w in wave_groups_all],
        "t1820_639_round_summary": {
            "rounds_audited": 639,
            "round_range": [3176, 3814],
            "n_updates": 2,
            "median_dev91_mean": t1820_med_val,
            "mean_dev91_mean": float(np.mean(t1820_means)),
            "min_dev91_mean": float(np.min(t1820_means)),
            "max_dev91_mean": float(np.max(t1820_means)),
            "median_changed_contexts": int(np.median(t1820_chgs)),
            "mean_trajectory_summary": clean_dict(t1820_mean_traj),
        },
        "trajectories": [clean_dict(t) for t in all_indiv_trajs],
        "resources": {
            "host": "xbabe2",
            "nice": 19,
            "cores": [0, 1],
            "peak_rss_bytes": peak_rss,
            "max_rss_limit_bytes": MAX_RSS_BYTES,
        },
    }

    out_path = HERE / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    assert out_path.stat().st_size < 50 * 1024, f"metrics.json size {out_path.stat().st_size} >= 50KB"
    return payload


if __name__ == "__main__":
    res = run_meta_analysis()
    print(
        json.dumps(
            {
                "verdict": res["verdict"],
                "median_control_lift_short": res["median_over_waves_control_summary"]["short_horizon_waves_2_to_8_updates"],
                "median_control_lift_all": res["median_over_waves_control_summary"]["all_waves_2_to_256_updates"],
                "drift_slope": res["drift_slope_vs_n_updates"],
                "winners_curse": res["winners_curse_analysis"],
                "decision_gate": res["decision_gate"],
            },
            indent=2,
        )
    )
