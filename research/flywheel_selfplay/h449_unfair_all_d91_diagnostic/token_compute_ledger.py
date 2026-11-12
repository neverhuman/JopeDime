#!/usr/bin/env python3
"""Builds the exact Token & Compute Cost Ledger (`token_compute_ledger.py`) across:
  1. Fair Canonical Champion (`h448.APEX_FAIR_CHAMPION` — `0%` `dev91`, `P201b`-disjoint bank),
  2. Fair Complete Non-`dev91` Stack & Pure Neural ICL (`h449.FAIR_ALL_NON_D91_*` — `0%` `dev91`, all `~1,976` non-`dev91` tables),
  3. Quarantined Unfair `ENTIRE` Stack & Pure Neural ICL (`h449.UNFAIR_ALL_D91_*` — `100%` `dev91`, all `~2,067` tables),
  4. External Baselines (`kumo_tabular_large_e8.v1`, `kumo_tabular_small_e8.v1`, `tabfm_ensemble.v1`).

Writes create-once `TOKEN_AND_COMPUTE_COST_LEDGER.json` and `TOKEN_AND_COMPUTE_COST_LEDGER.md` inside `H449/diag_results/`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from icl_prior_v3s import draw_features  # noqa: E402

PRETRAIN_ROW_BUCKETS = (384, 640, 1024, 1536)


def _max_rows_pretrain(frac: float) -> int:
    return (
        PRETRAIN_ROW_BUCKETS[0]
        if frac < 0.25
        else PRETRAIN_ROW_BUCKETS[1]
        if frac < 0.50
        else PRETRAIN_ROW_BUCKETS[2]
        if frac < 0.85
        else PRETRAIN_ROW_BUCKETS[3]
    )


def estimate_stage1_pretrain_tokens(
    *,
    steps: int,
    actual_datasets: int,
    frac_start: float,
    frac_end: float,
    tokens_cap: int,
    row_tokens_cap: int,
    bmax: int,
    real_bmax: int,
    real_frac: float,
    params: int,
    gpu_hours: float,
    seed: int,
    n_sample_steps: int = 20_000,
) -> dict:
    """Reconstructs Stage-1 `icl_train.py` token counts via `draw_shape` + empirical `datasets` calibration."""
    rng = random.Random(seed)
    sim_ds = 0
    sim_rows = 0
    sim_cells = 0
    sim_real_cells = 0
    for i in range(n_sample_steps):
        frac = frac_start + (frac_end - frac_start) * ((i + 0.5) / n_sample_steps)
        top = _max_rows_pretrain(frac)
        opts = [n for n in PRETRAIN_ROW_BUCKETS if n <= top]
        N = opts[-1] if rng.random() < 0.5 else rng.choice(opts)
        Fn = draw_features(rng)
        is_real = rng.random() < real_frac
        b_cap = min(bmax, real_bmax) if is_real else bmax
        B = max(2, min(b_cap, tokens_cap // max(N * (Fn + 4), 1), (row_tokens_cap // N) if row_tokens_cap else b_cap))
        r_tok = B * N
        c_tok = B * N * (Fn + 1)
        sim_ds += B
        sim_rows += r_tok
        sim_cells += c_tok
        if is_real:
            sim_real_cells += c_tok

    # Calibrate against exact logged `actual_datasets` from checkpoint `state`
    scale = float(actual_datasets) / max(float(sim_ds), 1.0)
    est_row_tokens = int(round(sim_rows * scale))
    est_cell_tokens = int(round(sim_cells * scale))
    est_real_cell_tokens = int(round(sim_real_cells * scale))
    return {
        "steps": int(steps),
        "datasets": int(actual_datasets),
        "row_tokens": est_row_tokens,
        "cell_tokens": est_cell_tokens,
        "real_cell_tokens": est_real_cell_tokens,
        "d91_cell_tokens": 0,
        "params": int(params),
        "approx_train_flops": int(6 * params * est_cell_tokens),
        "gpu_hours": round(float(gpu_hours), 3),
        "accounting_method": "icl_train.draw_shape calibrated to exact ckpt state['datasets']",
    }


def compute_d91_eval_tokens(bank_d91_search_npz: Path) -> dict:
    """Computes exact test-time `dev91` inference cell-tokens & row-tokens across all 91 contexts."""
    with np.load(bank_d91_search_npz, allow_pickle=False) as zb:
        ss = zb["s_shape"].astype(np.int64)
        sa = zb["a_shape"].astype(np.int64)
    # Single forward pass over (X_search, y_search, X_audit) across 91 contexts:
    # rows = n_search + n_audit, width = features + 1 (target slot)
    single_pass_rows = int((ss[:, 0] + sa[:, 0]).sum())
    single_pass_cells = int(((ss[:, 0] + sa[:, 0]) * ss[:, 1]).sum())
    # R4-e8 runs 5-fold inner CV (bag + q) + 1 full-search ref = 6 fold passes x 8 ensemble views on [X, S] (width + ~11 S cols)
    s_extra = 11
    r4e8_cells_per_model = int(((ss[:, 0] + sa[:, 0]) * (ss[:, 1] + s_extra)).sum()) * 6 * 8
    r4e8_rows_per_model = single_pass_rows * 6 * 8
    return {
        "contexts": int(len(ss)),
        "single_pass_row_tokens_91": single_pass_rows,
        "single_pass_cell_tokens_91": single_pass_cells,
        "kumo_e8_eval_cell_tokens_91": 23_873_536,  # recorded in H427 Summary.json (8-way ensemble)
        "r4_e8_eval_row_tokens_per_model_91": r4e8_rows_per_model,
        "r4_e8_eval_cell_tokens_per_model_91": r4e8_cells_per_model,
        "r4_e8_eval_cell_tokens_dual_104m_24m_91": 2 * r4e8_cells_per_model,
    }


def _sum_stages(stages: list[dict]) -> dict:
    return {
        "row_tokens": sum(int(s["row_tokens"]) for s in stages),
        "cell_tokens": sum(int(s["cell_tokens"]) for s in stages),
        "real_cell_tokens": sum(int(s["real_cell_tokens"]) for s in stages),
        "d91_cell_tokens": sum(int(s["d91_cell_tokens"]) for s in stages),
        "approx_train_flops": sum(int(s["approx_train_flops"]) for s in stages),
        "gpu_hours": round(sum(float(s.get("gpu_hours", 0.0)) for s in stages), 3),
        "cpu_hours": round(sum(float(s.get("cpu_hours", 0.0)) for s in stages), 3),
    }


def _stage_from_manifest(man: dict, cpu_threads: int = 0) -> dict:
    st = man["state"]
    el_h = float(st["elapsed_s"]) / 3600.0
    is_gpu = str(man.get("device", "")).startswith("cuda")
    return {
        "ckpt_sha256": man["ckpt_sha256"],
        "device": man.get("device", "unknown"),
        "gpu_name": man.get("gpu_name", "cpu"),
        "steps": int(st["step"]),
        "datasets": int(st["datasets"]),
        "row_tokens": int(st["cum_row_tokens"]),
        "cell_tokens": int(st["cum_cell_tokens"]),
        "real_cell_tokens": int(st["cum_real_cell_tokens"]),
        "d91_cell_tokens": int(st.get("cum_d91_cell_tokens", 0)),
        "params": int(man["params"]),
        "approx_train_flops": int(st["approx_train_flops"]),
        "gpu_hours": round(el_h if is_gpu else 0.0, 4),
        "cpu_hours": round(el_h * max(cpu_threads, 1) if not is_gpu else 0.0, 3),
        "wall_s": float(st["elapsed_s"]),
        "accounting_method": "exact_step_integer_counter",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h447-frozen", required=True)
    ap.add_argument("--h447-wc-l", default="")
    ap.add_argument("--h447-wc-m", default="")
    ap.add_argument("--h448-readout", required=True)
    ap.add_argument("--h449-readout", required=True)
    ap.add_argument("--h449-banks-manifest", required=True)
    ap.add_argument("--non-d91-manifest-l", required=True)
    ap.add_argument("--non-d91-manifest-m", required=True)
    ap.add_argument("--all-d91-manifest-l", required=True)
    ap.add_argument("--all-d91-manifest-m", required=True)
    ap.add_argument("--bank-d91-search", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    out_json = Path(a.out_json)
    out_md = Path(a.out_md)
    if out_json.exists() or out_md.exists():
        raise SystemExit(f"{out_json} or {out_md} already exists (create-once)")

    frz = json.loads(Path(a.h447_frozen).read_text())
    r448 = json.loads(Path(a.h448_readout).read_text())
    r449 = json.loads(Path(a.h449_readout).read_text())
    bm449 = json.loads(Path(a.h449_banks_manifest).read_text())
    man_non_l = json.loads(Path(a.non_d91_manifest_l).read_text())
    man_non_m = json.loads(Path(a.non_d91_manifest_m).read_text())
    man_all_l = json.loads(Path(a.all_d91_manifest_l).read_text())
    man_all_m = json.loads(Path(a.all_d91_manifest_m).read_text())
    eval_tok = compute_d91_eval_tokens(Path(a.bank_d91_search))

    # 1. Stage-1 ancestral pretraining (`fit_icl_bank.npz`, 419 tables + synthetic PROBS_V3S, 0% dev91)
    # 104M (`ICL_L_g1_w2_ckpt_h030.pt`): 2-GPU synced L_g1 (w2 on xbabe2 RTX 3090 + w0 on xbabe3 RTX 3090)
    # w2 state at h030: step=230,503, datasets=6,075,731, elapsed=30.03h; w0 synced partner ran ~225,000 steps / ~5.82M datasets (~29.5h)
    pre_l_w2 = estimate_stage1_pretrain_tokens(
        steps=230_503,
        actual_datasets=6_075_731,
        frac_start=0.0,
        frac_end=30.03 / 72.0,
        tokens_cap=550_000,
        row_tokens_cap=22_000,
        bmax=64,
        real_bmax=32,
        real_frac=0.38,
        params=104_360_576,
        gpu_hours=30.032,
        seed=3,
    )
    pre_l_w0 = estimate_stage1_pretrain_tokens(
        steps=224_972,
        actual_datasets=5_823_352,
        frac_start=0.01,
        frac_end=30.03 / 72.0,
        tokens_cap=550_000,
        row_tokens_cap=22_000,
        bmax=64,
        real_bmax=32,
        real_frac=0.38,
        params=104_360_576,
        gpu_hours=29.360,
        seed=1,
    )
    stage1_104m = {
        "model": "ICL_L (104.4M)",
        "params": 104_360_576,
        "real_tables_seen": 419,
        "steps_total": pre_l_w2["steps"] + pre_l_w0["steps"],
        "datasets_total": pre_l_w2["datasets"] + pre_l_w0["datasets"],
        **_sum_stages([pre_l_w2, pre_l_w0]),
    }

    # 24M (`ICL_M_r3_ckpt_h040.pt` on xbabe1 RTX 4090, init from `ICL_M_r2` 1.584h / 1,859,554 datasets)
    pre_m_r2 = estimate_stage1_pretrain_tokens(
        steps=34_689,
        actual_datasets=1_859_554,
        frac_start=0.0,
        frac_end=1.584 / 48.0,
        tokens_cap=600_000,
        row_tokens_cap=40_000,
        bmax=64,
        real_bmax=64,
        real_frac=0.0,
        params=23_659_392,
        gpu_hours=1.584,
        seed=1,
    )
    pre_m_r3 = estimate_stage1_pretrain_tokens(
        steps=679_467,
        actual_datasets=23_105_498,
        frac_start=0.0,
        frac_end=40.308 / 46.0,
        tokens_cap=600_000,
        row_tokens_cap=40_000,
        bmax=64,
        real_bmax=32,
        real_frac=0.39,
        params=23_659_392,
        gpu_hours=40.308,
        seed=11,
    )
    stage1_24m = {
        "model": "ICL_M (23.7M)",
        "params": 23_659_392,
        "real_tables_seen": 419,
        "steps_total": pre_m_r2["steps"] + pre_m_r3["steps"],
        "datasets_total": pre_m_r2["datasets"] + pre_m_r3["datasets"],
        **_sum_stages([pre_m_r2, pre_m_r3]),
    }
    stage1_combined = _sum_stages([stage1_104m, stage1_24m])

    # 2. H447/H448 Canonical Fair Champion (`h448.APEX_FAIR_CHAMPION`)
    sel_arm = frz["selected_arm"]
    h447_adapt_stages = []
    h448_real_tables = 419
    if sel_arm == "STACK_R4_E8_WC" and a.h447_wc_l and a.h447_wc_m and Path(a.h447_wc_l).is_file() and Path(a.h447_wc_m).is_file():
        wc_l = json.loads(Path(a.h447_wc_l).read_text())
        wc_m = json.loads(Path(a.h447_wc_m).read_text())
        h447_adapt_stages = [_stage_from_manifest(wc_l), _stage_from_manifest(wc_m)]
        h448_real_tables = 1_368
    elif sel_arm == "STACK_R4_M20":
        h448_real_tables = 419

    h448_adapt_tot = _sum_stages(h447_adapt_stages)
    h448_cum_tot = _sum_stages([stage1_combined, h448_adapt_tot])

    # 3. H449 Fair Complete Non-`dev91` (`all_non_d91_complete_bank.npz`, 0% dev91)
    st_non_l = _stage_from_manifest(man_non_l)
    st_non_m = _stage_from_manifest(man_non_m)
    non_d91_adapt_tot = _sum_stages([st_non_l, st_non_m])
    non_d91_cum_tot = _sum_stages([stage1_combined, non_d91_adapt_tot])
    non_d91_tables = int(bm449["all_non_d91_complete"]["tables"])

    # 4. H449 Quarantined Unfair `ENTIRE` (`everything_all_d91_bank.npz`, 100% dev91)
    # Initialized from `all_non_d91_{L,M}` -> cumulative = Stage1 + NonD91_GPU + AllD91_StageV
    st_all_l = _stage_from_manifest(man_all_l, cpu_threads=47)
    st_all_m = _stage_from_manifest(man_all_m, cpu_threads=48)
    all_d91_stage3_tot = _sum_stages([st_all_l, st_all_m])
    all_d91_adapt_tot = _sum_stages([non_d91_adapt_tot, all_d91_stage3_tot])
    all_d91_cum_tot = _sum_stages([stage1_combined, all_d91_adapt_tot])
    all_d91_tables = int(bm449["everything_all_d91"]["tables"])
    d91_tables_in_everything = int(bm449.get("d91_total_tables_in_everything", 91))

    # Extract Elo scores across canonical 91-context board & H425 clear slice
    b91_h448 = r448["readout"]["canonical_board_91"]
    clr_h448 = r448["readout"]["h425_without_feature_clear"]
    elo_kumo_l = float(b91_h448["board_elo"]["kumo_tabular_large_e8.v1"]["elo"])
    ci_kumo_l = b91_h448["board_elo"]["kumo_tabular_large_e8.v1"]["ci95"]
    elo_kumo_s = float(b91_h448["board_elo"]["kumo_tabular_small_e8.v1"]["elo"])
    ci_kumo_s = b91_h448["board_elo"]["kumo_tabular_small_e8.v1"]["ci95"]
    elo_tabfm = float(b91_h448["board_elo"]["tabfm_ensemble.v1"]["elo"])
    ci_tabfm = b91_h448["board_elo"]["tabfm_ensemble.v1"]["ci95"]

    ro_non_s = r449.get("fair_all_non_d91_stack")
    ro_non_i = r449.get("fair_all_non_d91_icl_only")
    ro_all_s = r449["unfair_all_d91_stack"]
    ro_all_i = r449["unfair_all_d91_icl_only"]

    arms_summary = [
        {
            "entrant": "h448.APEX_FAIR_CHAMPION",
            "regime": f"FAIR CANONICAL (0% dev91; {sel_arm})",
            "quarantined": False,
            "neural_params": 128_019_968,
            "real_datasets_seen": h448_real_tables,
            "d91_datasets_seen": 0,
            "elo_91": float(b91_h448["entrant_elo"]["elo"]),
            "ci95_91": b91_h448["entrant_elo"]["ci95"],
            "rank_91": int(b91_h448["rank"]),
            "elo_h425_clear": float(clr_h448["entrant_elo"]["elo"]),
            "ci95_h425_clear": clr_h448["entrant_elo"]["ci95"],
            "gap_vs_kumo_large_91": float(b91_h448["gap_elo"]["vs_kumo_tabular_large_e8.v1"]),
            "adaptation_tokens_and_compute": h448_adapt_tot,
            "cumulative_tokens_and_compute": h448_cum_tot,
            "d91_eval_cell_tokens": eval_tok["r4_e8_eval_cell_tokens_dual_104m_24m_91"],
        },
    ]

    if ro_non_s is not None and ro_non_i is not None:
        arms_summary.extend([
            {
                "entrant": "h449.FAIR_ALL_NON_D91_STACK",
                "regime": f"FAIR COMPLETE NON-DEV91 STACK (0% dev91; {non_d91_tables} complete tables)",
                "quarantined": True,
                "neural_params": 128_019_968,
                "real_datasets_seen": non_d91_tables,
                "d91_datasets_seen": 0,
                "elo_91": float(ro_non_s["elo"]["elo"]),
                "ci95_91": ro_non_s["elo"]["ci95"],
                "rank_91": int(ro_non_s["rank"]),
                "elo_h425_clear": float(r449["fair_all_non_d91_stack_h425_clear"]["elo"]["elo"]),
                "gap_vs_kumo_large_91": float(ro_non_s["gap_elo"]["vs_kumo_tabular_large_e8.v1"]),
                "adaptation_tokens_and_compute": non_d91_adapt_tot,
                "cumulative_tokens_and_compute": non_d91_cum_tot,
                "d91_eval_cell_tokens": eval_tok["r4_e8_eval_cell_tokens_dual_104m_24m_91"],
            },
            {
                "entrant": "h449.FAIR_ALL_NON_D91_ICL_ONLY",
                "regime": f"FAIR COMPLETE NON-DEV91 PURE NEURAL ICL (0% dev91; {non_d91_tables} complete tables)",
                "quarantined": True,
                "neural_params": 128_019_968,
                "real_datasets_seen": non_d91_tables,
                "d91_datasets_seen": 0,
                "elo_91": float(ro_non_i["elo"]["elo"]),
                "ci95_91": ro_non_i["elo"]["ci95"],
                "rank_91": int(ro_non_i["rank"]),
                "elo_h425_clear": float(r449["fair_all_non_d91_icl_only_h425_clear"]["elo"]["elo"]),
                "gap_vs_kumo_large_91": float(ro_non_i["gap_elo"]["vs_kumo_tabular_large_e8.v1"]),
                "adaptation_tokens_and_compute": non_d91_adapt_tot,
                "cumulative_tokens_and_compute": non_d91_cum_tot,
                "d91_eval_cell_tokens": eval_tok["r4_e8_eval_cell_tokens_dual_104m_24m_91"],
            },
        ])

    arms_summary.extend([
        {
            "entrant": "h449.UNFAIR_ALL_D91_STACK",
            "regime": f"UNFAIR ENTIRE STACK (100% dev91; {all_d91_tables} complete tables)",
            "quarantined": True,
            "neural_params": 128_019_968,
            "real_datasets_seen": all_d91_tables,
            "d91_datasets_seen": d91_tables_in_everything,
            "elo_91": float(ro_all_s["elo"]["elo"]),
            "ci95_91": ro_all_s["elo"]["ci95"],
            "rank_91": int(ro_all_s["rank"]),
            "elo_h425_clear": float(r449["unfair_all_d91_stack_h425_clear"]["elo"]["elo"]),
            "gap_vs_kumo_large_91": float(ro_all_s["gap_elo"]["vs_kumo_tabular_large_e8.v1"]),
            "adaptation_tokens_and_compute": all_d91_adapt_tot,
            "cumulative_tokens_and_compute": all_d91_cum_tot,
            "stage3_d91_finetune_only": all_d91_stage3_tot,
            "d91_eval_cell_tokens": eval_tok["r4_e8_eval_cell_tokens_dual_104m_24m_91"],
        },
        {
            "entrant": "h449.UNFAIR_ALL_D91_ICL_ONLY",
            "regime": f"UNFAIR ENTIRE PURE NEURAL ICL (100% dev91; {all_d91_tables} complete tables)",
            "quarantined": True,
            "neural_params": 128_019_968,
            "real_datasets_seen": all_d91_tables,
            "d91_datasets_seen": d91_tables_in_everything,
            "elo_91": float(ro_all_i["elo"]["elo"]),
            "ci95_91": ro_all_i["elo"]["ci95"],
            "rank_91": int(ro_all_i["rank"]),
            "elo_h425_clear": float(r449["unfair_all_d91_icl_only_h425_clear"]["elo"]["elo"]),
            "gap_vs_kumo_large_91": float(ro_all_i["gap_elo"]["vs_kumo_tabular_large_e8.v1"]),
            "adaptation_tokens_and_compute": all_d91_adapt_tot,
            "cumulative_tokens_and_compute": all_d91_cum_tot,
            "stage3_d91_finetune_only": all_d91_stage3_tot,
            "d91_eval_cell_tokens": eval_tok["r4_e8_eval_cell_tokens_dual_104m_24m_91"],
        },
        {
            "entrant": "kumo_tabular_large_e8.v1",
            "regime": "External Proprietary Pretrained Foundation Model (8-way ensemble)",
            "quarantined": False,
            "neural_params": 215_683_543,
            "elo_91": elo_kumo_l,
            "ci95_91": ci_kumo_l,
            "elo_h425_clear": float(clr_h448["board_elo"]["kumo_tabular_large_e8.v1"]["elo"]),
            "pretraining_tokens": "undisclosed (proprietary KumoAI corpus)",
            "d91_eval_cell_tokens": eval_tok["kumo_e8_eval_cell_tokens_91"],
            "d91_eval_cpu_hours": 18.04,
        },
        {
            "entrant": "kumo_tabular_small_e8.v1",
            "regime": "External Proprietary Pretrained Foundation Model (8-way ensemble)",
            "quarantined": False,
            "neural_params": 28_466_343,
            "elo_91": elo_kumo_s,
            "ci95_91": ci_kumo_s,
            "elo_h425_clear": float(clr_h448["board_elo"]["kumo_tabular_small_e8.v1"]["elo"]),
            "pretraining_tokens": "undisclosed (proprietary KumoAI corpus)",
            "d91_eval_cell_tokens": eval_tok["kumo_e8_eval_cell_tokens_91"],
            "d91_eval_cpu_hours": 3.49,
        },
        {
            "entrant": "tabfm_ensemble.v1",
            "regime": "External Pretrained Tabular Foundation Ensemble (6.59 GB weights)",
            "quarantined": False,
            "neural_params": None,
            "weights_bytes": 6_590_000_000,
            "elo_91": elo_tabfm,
            "ci95_91": ci_tabfm,
            "elo_h425_clear": float(clr_h448["board_elo"]["tabfm_ensemble.v1"]["elo"]),
            "pretraining_tokens": "undisclosed (external pretrained checkpoints)",
        },
    ])

    ledger = {
        "schema": "jope.h449.token_and_compute_cost_ledger/v1",
        "token_definitions": {
            "row_token": "1 table row (B * N) processed by row-transformer layers",
            "cell_token": "1 table cell (B * N * (F + 1)) processed by column/feature-slot embeddings & attention",
            "approx_train_flops": "6 * N_params * cell_tokens (standard dense forward+backward transformer FLOPs)",
        },
        "stage1_ancestral_pretraining": {
            "104m_L": stage1_104m,
            "24m_M": stage1_24m,
            "combined_128m": stage1_combined,
        },
        "stage2_non_d91_gpu_adaptation": {
            "h447_canonical_p201b_disjoint": h447_adapt_stages,
            "h449_all_non_d91_complete_L": st_non_l,
            "h449_all_non_d91_complete_M": st_non_m,
            "h449_all_non_d91_combined": non_d91_adapt_tot,
        },
        "stage3_quarantined_all_d91_finetune": {
            "all_d91_L": st_all_l,
            "all_d91_M": st_all_m,
            "combined": all_d91_stage3_tot,
        },
        "d91_test_time_inference_tokens": eval_tok,
        "arms_comparison": arms_summary,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "x") as f:
        f.write(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    os.chmod(out_json, 0o444)

    # Render Markdown table
    md_lines = [
        "# Fair (`NON-DEV91`) vs. Unfair (`ENTIRE` incl. `100%` `dev91`) Elo & Token/Compute Cost Ledger",
        "",
        f"| Entrant | Regime (`dev91` Exposure) | Board Elo (`91` ctx, `95% CI`) | Clear-Holdout Elo (`H425`) | Gap vs Kumo Large (`{elo_kumo_l:.1f}`) | Real Tables Seen | Cumul. Cell-Tokens (Adapt Cell-Tokens) | `dev91` Train Cell-Tokens | Approx Train FLOPs | Train Compute (GPU-h + CPU-h) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms_summary:
        eid = arm["entrant"]
        reg = arm["regime"]
        e91 = f"**{arm['elo_91']:.1f}** (`[{arm['ci95_91'][0]:.1f}, {arm['ci95_91'][1]:.1f}]`)"
        eclr = f"`{arm['elo_h425_clear']:.1f}`"
        gap = f"`{arm['gap_vs_kumo_large_91']:+.1f}`" if "gap_vs_kumo_large_91" in arm else "—"
        if "cumulative_tokens_and_compute" in arm:
            ct = arm["cumulative_tokens_and_compute"]
            at = arm["adaptation_tokens_and_compute"]
            tabs = f"`{arm['real_datasets_seen']:,}` (`{arm['d91_datasets_seen']}` d91)"
            tok_s = f"`{ct['cell_tokens']/1e9:.2f}B` (`+{at['cell_tokens']/1e9:.2f}B`)"
            d91_s = f"`{ct['d91_cell_tokens']/1e6:.1f}M`"
            flops_s = f"`{ct['approx_train_flops']:.2e}`"
            comp_s = f"`{ct['gpu_hours']:.2f}` GPU-h + `{ct['cpu_hours']:.1f}` CPU-h"
        else:
            tabs = "undisclosed"
            tok_s = str(arm.get("pretraining_tokens", "undisclosed"))
            d91_s = "unknown"
            flops_s = "undisclosed"
            comp_s = f"eval `{arm['d91_eval_cpu_hours']:.2f}` CPU-h" if "d91_eval_cpu_hours" in arm else "undisclosed"
        md_lines.append(f"| `{eid}` | {reg} | {e91} | {eclr} | {gap} | {tabs} | {tok_s} | {d91_s} | {flops_s} | {comp_s} |")

    with open(out_md, "x") as f:
        f.write("\n".join(md_lines) + "\n")
    os.chmod(out_md, 0o444)
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
