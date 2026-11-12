#!/usr/bin/env python3
"""Unified Token, FLOP & Compute Cost Ledger for Wave H450 (`token_compute_ledger_h450.py`).

Clearly separates:
  - Part A: Out-of-Sample Generalisation (`0%` `dev91` in neural weights: `h448.APEX_FAIR_CHAMPION`,
    optional `h449.FAIR_ALL_NON_D91_STACK`, `h450.FAIR_UPGRADED_ZERO_D91_CONTROL`, and external reference models).
  - Part B: Quarantined In-Corpus Random-Split `dev91` Fit Diagnostics (`dev91` included in combined `train U test`
    random-split training corpus; measures fit under random splits, NOT generalisation: `h450.AGGRESSIVE_RANDOM_ALL_CHAMPION`,
    `h450.NEURAL_PLUS_TAB_NO_GP`, `h450.ICL_BOOST_ONLY`, plus optional `h449.UNFAIR_ALL_D91_STACK`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

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


def _est_stage1(steps: int, actual_datasets: int, frac_start: float, frac_end: float, tokens_cap: int, row_tokens_cap: int, bmax: int, real_bmax: int, real_frac: float, params: int, gpu_hours: float, seed: int) -> dict:
    rng = random.Random(seed)
    sim_ds = sim_rows = sim_cells = sim_real = 0
    n_s = 20_000
    for i in range(n_s):
        frac = frac_start + (frac_end - frac_start) * ((i + 0.5) / n_s)
        top = _max_rows_pretrain(frac)
        opts = [n for n in PRETRAIN_ROW_BUCKETS if n <= top]
        N = opts[-1] if rng.random() < 0.5 else rng.choice(opts)
        Fn = draw_features(rng)
        is_real = rng.random() < real_frac
        b_cap = min(bmax, real_bmax) if is_real else bmax
        B = max(2, min(b_cap, tokens_cap // max(N * (Fn + 4), 1), (row_tokens_cap // N) if row_tokens_cap else b_cap))
        c_tok = B * N * (Fn + 1)
        sim_ds += B
        sim_rows += B * N
        sim_cells += c_tok
        if is_real:
            sim_real += c_tok
    sc = float(actual_datasets) / max(float(sim_ds), 1.0)
    ec = int(round(sim_cells * sc))
    return {
        "steps": int(steps),
        "datasets": int(actual_datasets),
        "unique_random_splits": int(actual_datasets),
        "row_tokens": int(round(sim_rows * sc)),
        "cell_tokens": ec,
        "real_cell_tokens": int(round(sim_real * sc)),
        "d91_cell_tokens": 0,
        "approx_train_flops": int(6 * params * ec),
        "gpu_hours": round(float(gpu_hours), 3),
        "cpu_hours": 0.0,
    }


def _sum(stages: list[dict]) -> dict:
    return {
        "unique_random_splits": sum(int(s.get("unique_random_splits", s.get("datasets", 0))) for s in stages),
        "row_tokens": sum(int(s["row_tokens"]) for s in stages),
        "cell_tokens": sum(int(s["cell_tokens"]) for s in stages),
        "real_cell_tokens": sum(int(s["real_cell_tokens"]) for s in stages),
        "d91_cell_tokens": sum(int(s.get("d91_cell_tokens", 0)) for s in stages),
        "approx_train_flops": sum(int(s["approx_train_flops"]) for s in stages),
        "gpu_hours": round(sum(float(s.get("gpu_hours", 0.0)) for s in stages), 3),
        "cpu_hours": round(sum(float(s.get("cpu_hours", 0.0)) for s in stages), 3),
    }


def _from_man(man: dict, cpu_threads: int = 0) -> dict:
    st = man["state"]
    wall_s = float(st.get("elapsed_s", st.get("elapsed", 0.0)))
    el_h = wall_s / 3600.0
    is_gpu = str(man.get("device", "")).startswith("cuda")
    return {
        "ckpt_sha256": man["ckpt_sha256"],
        "device": man.get("device", "cuda"),
        "gpu_name": man.get("gpu_name", "GPU"),
        "steps": int(st["step"]),
        "datasets": int(st["datasets"]),
        "unique_random_splits": int(st.get("cum_unique_random_splits", st["datasets"])),
        "row_tokens": int(st["cum_row_tokens"]),
        "cell_tokens": int(st["cum_cell_tokens"]),
        "real_cell_tokens": int(st["cum_real_cell_tokens"]),
        "d91_cell_tokens": int(st.get("cum_d91_cell_tokens", 0)),
        "params": int(man["params"]),
        "approx_train_flops": int(st["approx_train_flops"]),
        "gpu_hours": round(el_h if is_gpu else 0.0, 4),
        "cpu_hours": round(el_h * max(cpu_threads, 1) if not is_gpu else 0.0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h448-readout", required=True)
    ap.add_argument("--h450-readout", required=True)
    ap.add_argument("--h450-banks-manifest", required=True)
    ap.add_argument("--non-d91-manifest-l", required=True)
    ap.add_argument("--non-d91-manifest-m", required=True)
    ap.add_argument("--h450-manifest-la", required=True)
    ap.add_argument("--h450-manifest-ma", required=True)
    ap.add_argument("--h450-manifest-lb", required=True)
    ap.add_argument("--h450-manifest-mb", required=True)
    ap.add_argument("--h447-wc-l", default="/home/ubuntu/jope-data/h335-cl/h447/pub-r3/wc_L/wc_manifest.json")
    ap.add_argument("--h447-wc-m", default="/home/ubuntu/jope-data/h335-cl/h447/pub-r3/wc_M/wc_manifest.json")
    ap.add_argument("--h449-readout", default="/home/ubuntu/jope_plan/evidence/H449/diag_results/H449_UNFAIR_DEV91.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    out_json, out_md = Path(a.out_json), Path(a.out_md)
    if out_json.exists() or out_md.exists():
        raise SystemExit(f"{out_json} or {out_md} exists (create-once)")

    r448 = json.loads(Path(a.h448_readout).read_text())
    r450 = json.loads(Path(a.h450_readout).read_text())
    bm450 = json.loads(Path(a.h450_banks_manifest).read_text())

    pre_l_w2 = _est_stage1(230_503, 6_075_731, 0.0, 30.03 / 72.0, 550_000, 22_000, 64, 32, 0.38, 104_360_576, 30.032, 3)
    pre_l_w0 = _est_stage1(224_972, 5_823_352, 0.01, 30.03 / 72.0, 550_000, 22_000, 64, 32, 0.38, 104_360_576, 29.360, 1)
    pre_m_r2 = _est_stage1(34_689, 1_859_554, 0.0, 1.584 / 48.0, 600_000, 40_000, 64, 64, 0.0, 23_659_392, 1.584, 1)
    pre_m_r3 = _est_stage1(679_467, 23_105_498, 0.0, 40.308 / 46.0, 600_000, 40_000, 64, 32, 0.39, 23_659_392, 40.308, 11)
    stage1_tot = _sum([pre_l_w2, pre_l_w0, pre_m_r2, pre_m_r3])

    wc_stages = []
    if Path(a.h447_wc_l).is_file() and Path(a.h447_wc_m).is_file():
        wc_stages = [_from_man(json.loads(Path(a.h447_wc_l).read_text())), _from_man(json.loads(Path(a.h447_wc_m).read_text()))]
    h448_adapt = _sum(wc_stages)
    h448_cum = _sum([stage1_tot, h448_adapt])

    st_non_l = _from_man(json.loads(Path(a.non_d91_manifest_l).read_text()))
    st_non_m = _from_man(json.loads(Path(a.non_d91_manifest_m).read_text()))
    non_d91_adapt = _sum([st_non_l, st_non_m])
    non_d91_cum = _sum([stage1_tot, non_d91_adapt])

    st_la = _from_man(json.loads(Path(a.h450_manifest_la).read_text()))
    st_ma = _from_man(json.loads(Path(a.h450_manifest_ma).read_text()))
    st_lb = _from_man(json.loads(Path(a.h450_manifest_lb).read_text()))
    st_mb = _from_man(json.loads(Path(a.h450_manifest_mb).read_text()))
    h450_stage3 = _sum([st_la, st_ma, st_lb, st_mb])
    h450_adapt = _sum([non_d91_adapt, h450_stage3])
    h450_cum = _sum([stage1_tot, h450_adapt])

    b91 = r448["readout"]["canonical_board_91"]
    clr = r448["readout"]["h425_without_feature_clear"]

    part_a_fair = [
        {
            "entrant": "h448.APEX_FAIR_CHAMPION",
            "category": "Canonical Fair Board Entry (0% dev91)",
            "real_tables_seen": 1368,
            "d91_tables_seen": 0,
            "elo_91": float(b91["entrant_elo"]["elo"]),
            "ci95_91": b91["entrant_elo"]["ci95"],
            "elo_h425_clear": float(clr["entrant_elo"]["elo"]),
            "adaptation": h448_adapt,
            "cumulative": h448_cum,
        },
    ]

    z91 = r450.get("fair_zero_d91_upgraded_control_91")
    zclr = r450.get("fair_zero_d91_upgraded_control_h425_clear")
    if z91 is not None:
        part_a_fair.append({
            "entrant": "h450.FAIR_UPGRADED_ZERO_D91_CONTROL",
            "category": "Post-Hoc on dev91 (0% neural dev91 weights; upgrades designed post-H448/H449 — NOT fair)",
            "real_tables_seen": 3392,
            "d91_tables_seen": 0,
            "elo_91": float(z91["elo"]["elo"]),
            "ci95_91": z91["elo"]["ci95"],
            "elo_h425_clear": float(zclr["elo"]["elo"]) if zclr else None,
            "adaptation": non_d91_adapt,
            "cumulative": non_d91_cum,
        })

    r449_doc = json.loads(Path(a.h449_readout).read_text()) if Path(a.h449_readout).is_file() else None
    if r449_doc and r449_doc.get("fair_all_non_d91_stack"):
        fn = r449_doc["fair_all_non_d91_stack"]
        part_a_fair.append({
            "entrant": "h449.FAIR_ALL_NON_D91_STACK",
            "category": "Post-H448 Second Look (0% dev91, legacy R4-e8 + Guard+2L — diagnostic only)",
            "real_tables_seen": 3392,
            "d91_tables_seen": 0,
            "elo_91": float(fn["elo"]["elo"]),
            "ci95_91": fn["elo"]["ci95"],
            "elo_h425_clear": float(r449_doc["fair_all_non_d91_stack_h425_clear"]["elo"]["elo"]),
            "adaptation": non_d91_adapt,
            "cumulative": non_d91_cum,
        })

    part_a_fair.extend([
        {
            "entrant": "kumo_tabular_large_e8.v1",
            "category": "Stored Reference Foundation Model (215.7M params, 8-way ensemble)",
            "elo_91": float(b91["board_elo"]["kumo_tabular_large_e8.v1"]["elo"]),
            "ci95_91": b91["board_elo"]["kumo_tabular_large_e8.v1"]["ci95"],
            "elo_h425_clear": float(clr["board_elo"]["kumo_tabular_large_e8.v1"]["elo"]),
        },
        {
            "entrant": "tabfm_ensemble.v1",
            "category": "Stored Reference Foundation Ensemble (6.59 GB weights)",
            "elo_91": float(b91["board_elo"]["tabfm_ensemble.v1"]["elo"]),
            "ci95_91": b91["board_elo"]["tabfm_ensemble.v1"]["ci95"],
            "elo_h425_clear": float(clr["board_elo"]["tabfm_ensemble.v1"]["elo"]),
        },
    ])

    tot_tables = int(bm450["everything_combined_f64"]["tables"])
    c91 = r450["in_corpus_random_split_champion_91"]
    cclr = r450["in_corpus_random_split_champion_h425_clear"]
    ng91 = r450["in_corpus_neural_plus_tab_no_gp_91"]
    ic91 = r450["in_corpus_icl_boost_only_91"]

    part_b_in_corpus = [
        {
            "entrant": "h450.AGGRESSIVE_RANDOM_ALL_CHAMPION",
            "category": "In-Corpus Random-Split Fit (train U test merged & CSPRNG-shuffled, 4x GPU ckpts + Tab + Log-RMSE Stack)",
            "real_tables_seen": tot_tables,
            "d91_tables_seen": 133,
            "bt_elo_91": float(c91["elo"]["elo"]),
            "ci95_91": c91["elo"]["ci95"],
            "virtual_pos_vs_33": int(c91["rank_if_placed_against_frozen_33"]),
            "bt_elo_h425_clear": float(cclr["elo"]["elo"]),
            "stage3_h450_gpu_finetune": h450_stage3,
            "adaptation": h450_adapt,
            "cumulative": h450_cum,
        },
        {
            "entrant": "h450.NEURAL_PLUS_TAB_NO_GP",
            "category": "In-Corpus Random-Split Fit (H450-Neural + H450-Tab ONLY, 0 GP/M7/K3/F4)",
            "real_tables_seen": tot_tables,
            "d91_tables_seen": 133,
            "bt_elo_91": float(ng91["elo"]["elo"]),
            "ci95_91": ng91["elo"]["ci95"],
            "virtual_pos_vs_33": int(ng91["rank_if_placed_against_frozen_33"]),
            "stage3_h450_gpu_finetune": h450_stage3,
            "adaptation": h450_adapt,
            "cumulative": h450_cum,
        },
        {
            "entrant": "h450.ICL_BOOST_ONLY",
            "category": "In-Corpus Random-Split Fit (Pure Neural H450 2-Stage ICL ONLY, 0 Trees/GP)",
            "real_tables_seen": tot_tables,
            "d91_tables_seen": 133,
            "bt_elo_91": float(ic91["elo"]["elo"]),
            "ci95_91": ic91["elo"]["ci95"],
            "virtual_pos_vs_33": int(ic91["rank_if_placed_against_frozen_33"]),
            "stage3_h450_gpu_finetune": h450_stage3,
            "adaptation": h450_adapt,
            "cumulative": h450_cum,
        },
    ]
    if r449_doc and r449_doc.get("unfair_all_d91_stack"):
        u49 = r449_doc["unfair_all_d91_stack"]
        part_b_in_corpus.append({
            "entrant": "h449.UNFAIR_ALL_D91_STACK",
            "category": "In-Corpus Fit (0.25h CPU float32 legacy R4-e8 + Guard+2L)",
            "real_tables_seen": 3525,
            "d91_tables_seen": 133,
            "bt_elo_91": float(u49["elo"]["elo"]),
            "ci95_91": u49["elo"]["ci95"],
            "virtual_pos_vs_33": int(u49["rank"]),
        })

    doc = {
        "schema": "jope.h450.token_and_compute_cost_ledger/v1",
        "canonical_promotion_forbidden": True,
        "stage1_ancestral_pretraining": stage1_tot,
        "stage2_non_d91_gpu": non_d91_adapt,
        "stage3_h450_random_split_4ckpt_gpu": {
            "L_seedA": st_la,
            "M_seedA": st_ma,
            "L_seedB": st_lb,
            "M_seedB": st_mb,
            "combined": h450_stage3,
        },
        "part_a_out_of_sample_zero_d91": part_a_fair,
        "part_b_quarantined_in_corpus_random_split_fit": part_b_in_corpus,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "x") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.chmod(out_json, 0o444)

    md = [
        "# Wave H450 Unified Token, FLOP & Compute Cost Ledger",
        "",
        "## Part A — Out-of-Sample Generalisation (`0%` `dev91` in Neural Weights)",
        "",
        "| Entrant | Regime | `all_91` Elo (`95% CI`) | `H425` Clear Elo | Real Tables | Cumul. Cell-Tokens (`+Adapt`) | Train Compute |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in part_a_fair:
        e91 = f"**{r['elo_91']:.1f}** (`[{r['ci95_91'][0]:.1f}, {r['ci95_91'][1]:.1f}]`)"
        ec = f"`{r['elo_h425_clear']:.1f}`" if r.get("elo_h425_clear") is not None else "—"
        if "cumulative" in r:
            ct, at = r["cumulative"], r["adaptation"]
            md.append(
                f"| `{r['entrant']}` | {r['category']} | {e91} | {ec} | `{r['real_tables_seen']:,}` (`0` d91) | "
                f"`{ct['cell_tokens']/1e9:.2f}B` (`+{at['cell_tokens']/1e9:.2f}B`) | `{ct['gpu_hours']:.2f}` GPU-h |"
            )
        else:
            md.append(f"| `{r['entrant']}` | {r['category']} | {e91} | {ec} | undisclosed | undisclosed | external |")

    md.extend([
        "",
        "## Part B — Quarantined In-Corpus Random-Split `dev91` Fit Diagnostics (`canonical_promotion_forbidden = true`)",
        "> *Scientific Note:* Because these arms include `dev91` (`train U test` merged & CSPRNG-shuffled) inside their online random-split training corpus, their `dev91` BT-Elo numbers measure **in-corpus fit under random splits**, NOT out-of-sample generalisation, and are excluded from the canonical `dev91-audit-v1` board.",
        "",
        "| Quarantined Diagnostic Arm | Regime | `all_91` BT-Elo (`95% CI`) | Virtual Pos (vs 33) | Unique Random Splits (Stage 3) | Stage-3 Cell-Tokens (`d91` Cell-Tokens) | Stage-3 GPU-h |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in part_b_in_corpus:
        e91 = f"**{r['bt_elo_91']:.1f}** (`[{r['ci95_91'][0]:.1f}, {r['ci95_91'][1]:.1f}]`)"
        vp = f"`#{r['virtual_pos_vs_33']}/34`"
        if "stage3_h450_gpu_finetune" in r:
            s3 = r["stage3_h450_gpu_finetune"]
            md.append(
                f"| `{r['entrant']}` | {r['category']} | {e91} | {vp} | `{s3['unique_random_splits']:,}` | "
                f"`{s3['cell_tokens']/1e9:.2f}B` (`{s3['d91_cell_tokens']/1e6:.1f}M` d91) | `{s3['gpu_hours']:.2f}` GPU-h |"
            )
        else:
            md.append(f"| `{r['entrant']}` | {r['category']} | {e91} | {vp} | — | CPU 0.25h | `0.00` GPU-h |")

    with open(out_md, "x") as f:
        f.write("\n".join(md) + "\n")
    os.chmod(out_md, 0o444)
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
