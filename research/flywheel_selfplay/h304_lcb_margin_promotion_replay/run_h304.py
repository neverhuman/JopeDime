"""H304 (h304_lcb_margin_promotion_replay): execute analysis-only promotion-rule replay on xbabe2 CPU."""

from __future__ import annotations

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
import analysis_h304 as a304  # noqa: E402

SCRATCH_DIR = Path("/home/ubuntu/jope_plan/ns/receipts/h304_scratch")
MAX_RSS_BYTES = 4 * 1024**3
MIN_MEM_AVAILABLE_KB = 20 * 1024**2


def install_analysis_only_guard() -> None:
    """Enforce no neural framework imports, no checkpoint/table opens, no forbidden cohort reads."""

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event == "import" and args and isinstance(args[0], str):
            mod = args[0].split(".")[0]
            if mod in ("torch", "tensorflow", "jax"):
                raise RuntimeError(f"Forbidden neural framework import in H304 analysis-only wave: {mod}")
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
                    "flywheel-gp-fused",
                    "flywheel-full-20261030",
                )
            ) or p.endswith((".pt", ".jtt1", ".jtt1.gz")):
                raise RuntimeError(f"Forbidden file open in H304 analysis-only wave: {p}")

    sys.addaudithook(audit_hook)


def check_resources() -> dict[str, int]:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss_kb) * 1024
    if rss_bytes > MAX_RSS_BYTES:
        raise RuntimeError(f"Peak RSS {rss_bytes} exceeds 4 GiB cap ({MAX_RSS_BYTES})")
    mem_avail_kb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_avail_kb = int(line.split()[1])
            break
    if mem_avail_kb < MIN_MEM_AVAILABLE_KB:
        raise RuntimeError(f"MemAvailable {mem_avail_kb} kB < 20 GiB minimum ({MIN_MEM_AVAILABLE_KB} kB)")
    return {"peak_rss_bytes": rss_bytes, "mem_available_kb": mem_avail_kb}


def compact_three_splits(res: dict[str, Any]) -> dict[str, Any]:
    per_seed_compact = {}
    for s_key, s_val in res["per_seed"].items():
        rules_c = {}
        for rname, rdict in s_val["rules"].items():
            rules_c[rname] = {
                "n_promotions": rdict["n_promotions"],
                "promotion_precision": rdict["promotion_precision"],
                "final_incumbent_round": rdict["final_incumbent_round"],
                "init_select_utility": rdict["init_select_utility"],
                "final_select_utility": rdict["final_select_utility"],
                "init_eval_family_equal_utility": rdict["init_eval_family_equal_utility"],
                "final_eval_family_equal_utility": rdict["final_eval_family_equal_utility"],
                "time_avg_eval_family_equal_utility": rdict["time_avg_eval_family_equal_utility"],
                "final_eval_gain_vs_r0": rdict["final_eval_gain_vs_r0"],
                "final_eval_gain_vs_r0_ci95": rdict["final_eval_gain_vs_r0_ci95"],
                "final_eval_gain_vs_r0_lcb95_one_sided": rdict["final_eval_gain_vs_r0_lcb95_one_sided"],
                "time_avg_eval_gain_vs_r0": rdict["time_avg_eval_gain_vs_r0"],
                "net_shrinkage_vs_init": rdict["net_shrinkage_vs_init"],
                "elo_vs_r3175_eval_half": rdict["elo_vs_r3175_eval_half"],
                "elo_vs_r3175_eval_half_ci95": rdict["elo_vs_r3175_eval_half_ci95"],
            }
        per_seed_compact[s_key] = {
            "seed": s_val["seed"],
            "n_select_families": s_val["n_select_families"],
            "n_eval_families": s_val["n_eval_families"],
            "n_select_contexts": s_val["n_select_contexts"],
            "n_eval_contexts": s_val["n_eval_contexts"],
            "rules": rules_c,
        }
    return {
        "select_mode": res["select_mode"],
        "split_seeds": res["split_seeds"],
        "decision": res["decision"],
        "across_seeds_summary": res["across_seeds_summary"],
        "per_seed": per_seed_compact,
    }


def run_h304() -> dict[str, Any]:
    t0 = time.time()
    install_analysis_only_guard()
    check_resources()

    # 1. Verify Step 1 Regression Receipt
    reg_path = HERE / "regression_receipt.json"
    assert reg_path.exists(), "Step 1 regression_receipt.json must exist before running H304"
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"

    hist_path = SCRATCH_DIR / "prepared/history.npz"
    adm_path = SCRATCH_DIR / "adapter_admissions.jsonl"
    assert hashlib.sha256(hist_path.read_bytes()).hexdigest() == a304.HISTORY_NPZ_SHA256
    assert hashlib.sha256(adm_path.read_bytes()).hexdigest() == a304.ADMISSIONS_JSONL_SHA256

    with np.load(hist_path, allow_pickle=False) as z:
        rounds = z["rounds"]
        utils = z["utilities"]
        families = [str(x) for x in z["families"].tolist()]

    assert int(rounds[0]) == 3175
    r3175_vec = utils[0].copy()
    r3175_mean = float(np.mean(r3175_vec))
    post_mask = rounds > 3175
    post_rounds = rounds[post_mask]
    post_utils = utils[post_mask]
    post_means = np.mean(post_utils, axis=1)

    assert len(post_rounds) == 1499
    assert abs(r3175_mean - 1.1389572403748425) <= 1e-15
    assert abs(float(np.mean(post_means)) - 1.1320995298853955) <= 1e-15
    assert abs(float(np.quantile(post_means, 0.95)) - 1.1354387884248622) <= 1e-15

    full_cache_path = SCRATCH_DIR / "metrics_full_h304.json"
    if full_cache_path.exists() and "--recompute" not in sys.argv:
        full_payload = json.loads(full_cache_path.read_text())
        primary_3seeds = full_payload["primary_r3175_init_family_equal_select"]
        primary_200splits = full_payload["primary_repeated_200_splits"]
        sens_ctx_3seeds = full_payload["sensitivity_r3175_init_context_mean_select"]
        sens_ctx_200splits = full_payload["sensitivity_r3175_init_context_mean_200_splits"]
        sens_post_3seeds = full_payload["sensitivity_r3176_post_peak_init_family_equal_select"]
        sens_post_200splits = full_payload["sensitivity_r3176_post_peak_init_family_equal_200_splits"]
        sens_post_ctx_3seeds = full_payload["sensitivity_r3176_post_peak_init_context_mean_select"]
        sens_post_ctx_200splits = full_payload["sensitivity_r3176_post_peak_init_context_mean_200_splits"]
    else:
        # 2. Primary Replay (r3175-initialized, rounds 3175..4675, select_mode="family_equal")
        primary_3seeds = a304.evaluate_three_splits(
            rounds=rounds,
            utils=utils,
            context_families=families,
            r3175_vec=r3175_vec,
            split_seeds=a304.SPLIT_SEEDS,
            select_mode="family_equal",
            boot_seed=a304.BOOT_SEED,
            boot_draws=a304.BOOT_DRAWS,
        )
        primary_200splits = a304.evaluate_repeated_200_splits(
            rounds=rounds,
            utils=utils,
            context_families=families,
            n_splits=a304.REPEATED_SPLITS_N,
            rng_seed=a304.BOOT_SEED,
            select_mode="family_equal",
        )

        # 3. Sensitivity A (r3175-initialized, rounds 3175..4675, select_mode="context_mean")
        sens_ctx_3seeds = a304.evaluate_three_splits(
            rounds=rounds,
            utils=utils,
            context_families=families,
            r3175_vec=r3175_vec,
            split_seeds=a304.SPLIT_SEEDS,
            select_mode="context_mean",
            boot_seed=a304.BOOT_SEED,
            boot_draws=a304.BOOT_DRAWS,
        )
        sens_ctx_200splits = a304.evaluate_repeated_200_splits(
            rounds=rounds,
            utils=utils,
            context_families=families,
            n_splits=a304.REPEATED_SPLITS_N,
            rng_seed=a304.BOOT_SEED,
            select_mode="context_mean",
        )

        # 4. Sensitivity B (Post-peak stream initialized at r3176, rounds 3176..4675, select_mode="family_equal")
        sens_post_3seeds = a304.evaluate_three_splits(
            rounds=post_rounds,
            utils=post_utils,
            context_families=families,
            r3175_vec=r3175_vec,
            split_seeds=a304.SPLIT_SEEDS,
            select_mode="family_equal",
            boot_seed=a304.BOOT_SEED,
            boot_draws=a304.BOOT_DRAWS,
        )
        sens_post_200splits = a304.evaluate_repeated_200_splits(
            rounds=post_rounds,
            utils=post_utils,
            context_families=families,
            n_splits=a304.REPEATED_SPLITS_N,
            rng_seed=a304.BOOT_SEED,
            select_mode="family_equal",
        )

        # 5. Sensitivity C (Post-peak stream initialized at r3176, rounds 3176..4675, select_mode="context_mean")
        sens_post_ctx_3seeds = a304.evaluate_three_splits(
            rounds=post_rounds,
            utils=post_utils,
            context_families=families,
            r3175_vec=r3175_vec,
            split_seeds=a304.SPLIT_SEEDS,
            select_mode="context_mean",
            boot_seed=a304.BOOT_SEED,
            boot_draws=a304.BOOT_DRAWS,
        )
        sens_post_ctx_200splits = a304.evaluate_repeated_200_splits(
            rounds=post_rounds,
            utils=post_utils,
            context_families=families,
            n_splits=a304.REPEATED_SPLITS_N,
            rng_seed=a304.BOOT_SEED,
            select_mode="context_mean",
        )

        res_info = check_resources()
        verdict = primary_3seeds["decision"]["verdict"]

        full_payload = {
            "wave_id": "H304",
            "slug": "h304_lcb_margin_promotion_replay",
            "owner": "rd_argon",
            "mode": "ANALYSIS_ONLY",
            "status": "COMPLETE",
            "verdict": verdict,
            "prereg_commit": a304.PREREG_COMMIT,
            "prepared_closure_version_id": a304.PREPARED_CLOSURE_VERSION_ID,
            "prepared_closure_sha256": a304.PREPARED_CLOSURE_SHA256,
            "history_npz_sha256": a304.HISTORY_NPZ_SHA256,
            "adapter_admissions_jsonl_sha256": a304.ADMISSIONS_JSONL_SHA256,
            "evaluation_identity": a304.EVAL_IDENTITY,
            "dope_sha256": a304.DOPE_SHA256,
            "dope_untouched": True,
            "optimizer_updates": 0,
            "neural_forwards": 0,
            "gpu_used": False,
            "fresh_pools_read": 0,
            "elapsed_sec": round(time.time() - t0, 2),
            "split_seeds": list(a304.SPLIT_SEEDS),
            "bootstrap_seed": a304.BOOT_SEED,
            "bootstrap_draws": a304.BOOT_DRAWS,
            "regression_step1": reg,
            "primary_r3175_init_family_equal_select": primary_3seeds,
            "primary_repeated_200_splits": primary_200splits,
            "sensitivity_r3175_init_context_mean_select": sens_ctx_3seeds,
            "sensitivity_r3175_init_context_mean_200_splits": sens_ctx_200splits,
            "sensitivity_r3176_post_peak_init_family_equal_select": sens_post_3seeds,
            "sensitivity_r3176_post_peak_init_family_equal_200_splits": sens_post_200splits,
            "sensitivity_r3176_post_peak_init_context_mean_select": sens_post_ctx_3seeds,
            "sensitivity_r3176_post_peak_init_context_mean_200_splits": sens_post_ctx_200splits,
            "resources": {
                "host": "xbabe2",
                "nice": 19,
                "cores": [0, 1, 2, 3],
                "peak_rss_bytes": res_info["peak_rss_bytes"],
                "max_rss_limit_bytes": MAX_RSS_BYTES,
                "mem_available_kb": res_info["mem_available_kb"],
            },
        }
        full_cache_path.write_text(json.dumps(full_payload, indent=2) + "\n")

    compact_payload = {
        k: v
        for k, v in full_payload.items()
        if not k.startswith("primary_") and not k.startswith("sensitivity_")
    }
    def compact_sens(s3: dict[str, Any], s200: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": s3["decision"],
            "across_seeds_summary": {
                r: {
                    "promotions_by_seed": rs["promotions_by_seed"],
                    "overall_promotion_precision": rs["overall_promotion_precision"],
                    "median_final_eval_gain_vs_r0": rs["median_final_eval_gain_vs_r0"],
                    "pooled_final_eval_gain_vs_r0_ci95": rs["pooled_final_eval_gain_vs_r0_ci95"],
                    "per_seed_final_lcb95_one_sided": rs["per_seed_final_lcb95_one_sided"],
                    "median_time_avg_eval_gain_vs_r0": rs["median_time_avg_eval_gain_vs_r0"],
                    "median_net_shrinkage_vs_init": rs["median_net_shrinkage_vs_init"],
                    "median_elo_vs_r3175": rs["median_elo_vs_r3175"],
                }
                for r, rs in s3["across_seeds_summary"].items()
            },
            "repeated_200_splits_rules": {
                r: {
                    "n_splits_with_promotions": v["n_splits_with_promotions"],
                    "total_promotions": v["total_promotions"],
                    "overall_promotion_precision": v["overall_promotion_precision"],
                    "mean_final_eval_gain_vs_r0": v["mean_final_eval_gain_vs_r0"],
                    "median_final_eval_gain_vs_r0": v["median_final_eval_gain_vs_r0"],
                    "splits_final_gain_gt_r0": v["splits_final_gain_gt_r0"],
                    "splits_final_gain_lt_r0": v["splits_final_gain_lt_r0"],
                    "mean_net_shrinkage_vs_init": v["mean_net_shrinkage_vs_init"],
                }
                for r, v in s200["rules"].items()
            },
        }

    compact_payload["primary_r3175_init_family_equal_select"] = compact_three_splits(primary_3seeds)
    compact_payload["primary_repeated_200_splits"] = primary_200splits
    compact_payload["sensitivity_r3175_init_context_mean_select"] = compact_sens(
        sens_ctx_3seeds, sens_ctx_200splits
    )
    compact_payload["sensitivity_r3176_post_peak_init_family_equal_select"] = compact_sens(
        sens_post_3seeds, sens_post_200splits
    )
    compact_payload["sensitivity_r3176_post_peak_init_context_mean_select"] = compact_sens(
        sens_post_ctx_3seeds, sens_post_ctx_200splits
    )

    out_path = HERE / "metrics.json"
    out_path.write_text(json.dumps(compact_payload, indent=2, allow_nan=False) + "\n")
    assert out_path.stat().st_size < 50 * 1024, f"metrics.json size {out_path.stat().st_size} >= 50KB"
    return compact_payload


if __name__ == "__main__":
    res = run_h304()
    print(
        json.dumps(
            {
                "verdict": res["verdict"],
                "elapsed_sec": res["elapsed_sec"],
                "primary_decision": res["primary_r3175_init_family_equal_select"]["decision"],
                "primary_across_seeds_summary": res["primary_r3175_init_family_equal_select"][
                    "across_seeds_summary"
                ],
                "primary_repeated_200_splits": res["primary_repeated_200_splits"]["rules"],
                "sensitivity_r3175_context_mean_decision": res[
                    "sensitivity_r3175_init_context_mean_select"
                ]["decision"],
                "sensitivity_r3176_post_peak_family_equal_decision": res[
                    "sensitivity_r3176_post_peak_init_family_equal_select"
                ]["decision"],
                "sensitivity_r3176_post_peak_family_equal_summary": res[
                    "sensitivity_r3176_post_peak_init_family_equal_select"
                ]["across_seeds_summary"],
            },
            indent=2,
        )
    )
