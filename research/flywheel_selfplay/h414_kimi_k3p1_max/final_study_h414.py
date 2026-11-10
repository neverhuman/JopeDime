#!/usr/bin/env python3
"""H414 refresh of the H412 FINAL study (final_study_h412.py with the field H413 + H414 added): our strongest stored
model vs every frontier LLM regressor (now 6, with kimik3p1_max) and every other entrant on dev91-audit-v1 (31
entrants). Reads stored files only; it never fits, runs or rescores a model. JSON output only (the dashboard source);
the human-readable tables are the H413 / H414 wave READMEs and the board LADDER.md files.

  final_study_h414.py [--board DIR] [--tools DIR] [--repo DIR] [--out DIR]

Inputs:
  * the board's BOARD.json + scores/*.jsonl (default: the git copy research/elo/reference-board/boards/dev91-audit-v1),
    loaded by tools/ladder.py `load` (envelope-sha and duplicate checks) and scored with the unchanged H335 functions
    (h335_stats: elo_panel, headline, primary_test, family_contrast, family_mean);
  * AutoML Elo v1 / v2 RESULTS.json, read from git at pinned commits and checked by sha256. That is a TRAIN-family panel
    (64 families x 3 folds x 3 seeds); it is shown for rank agreement only and is never merged into the board.
  * section D (addendum): the stored dev91-s1-two-fold-v1 board (--board2, same loader; its Elo must reproduce the
    stored ladder.json) and H257's published two-fold table (origin/main, sha-pinned; its 13 reference rows must match
    the board's family means). A different split: listed beside the AUDIT numbers, never merged into them.
Outputs: final_study.json and FINAL_STUDY.md in --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RB = HERE.parents[1] / "elo" / "reference-board"
OURS = "h364.XSTACK_2L"
SECOND = "h335.H335_AGG"
ANCHOR = "h335.H257_REPLICA"
FRONTIER = {  # board id -> (display name, AutoML Elo method name)
    "h335.Opus5p5_Extra": ("Opus5p5_Extra", "Opus5p5_Extra"),
    "h335.Gemini4_argon_high": ("Gemini4_argon_high", "Gemini4_argon_high"),
    "h335.GPT6p1_Sol_Max": ("GPT6p1_Sol_Max", "GPT6p1_Sol_Max"),
    "h335.GPT6_Astra_Max": ("GPT6_Astra_Max", "GPT6_Astra_Max"),
    "h412.grok4p7_xhigh": ("grok4p7_xhigh", "grok4p7_xhigh"),
    "h414.kimik3p1_max": ("kimik3p1_max", "kimik3p1_max"),
}
FOUNDATION = ["kumo_tabular_large_e8.v1", "tabfm_ensemble.v1", "tabfm.v1", "tabpfn_v3p5.v1", "tabpfn_v3.v1",
              "kumo_tabular_small_e8.v1", "tabpfn_v3p5_e1.v1", "mitra_v2_zeroshot.v1", "tabicl_v2.v1", "tabdpt_1p3.v1"]
AUTOML = ["autogluon_300s.v1"]
CLASSIC = ["catboost.v1", "xgboost.v1", "random_forest.v1", "lightgbm.v1", "ridge.v1"]
LABEL = {"kumo_tabular_large_e8.v1": "Kumo-Tabular large", "tabfm_ensemble.v1": "TabFM ensemble", "tabfm.v1": "TabFM",
         "tabpfn_v3p5.v1": "TabPFN 3.5", "tabpfn_v3.v1": "TabPFN v3", "autogluon_300s.v1": "AutoGluon (300 s)",
         "catboost.v1": "CatBoost", "xgboost.v1": "XGBoost", "random_forest.v1": "Random forest",
         "lightgbm.v1": "LightGBM", "ridge.v1": "Ridge", OURS: "XSTACK_2L (ours)", SECOND: "H335_AGG (ours)",
         "kumo_tabular_small_e8.v1": "Kumo-Tabular small", "tabpfn_v3p5_e1.v1": "TabPFN 3.5 (1 estimator)",
         "mitra_v2_zeroshot.v1": "Mitra v2 (zero-shot)", "tabicl_v2.v1": "TabICL v2", "tabdpt_1p3.v1": "TabDPT 1.3"}
# The compiled source of each dev91 frontier row (H335 build tree; H412 LOCK code_sha256) and the AutoML-panel source.
SOURCE = {
    "Opus5p5_Extra": ("2e2352f8ac708707", "2e2352f8ac708707"),
    "Gemini4_argon_high": ("d5cd85fcf5e2bea8", "d5cd85fcf5e2bea8"),
    "GPT6p1_Sol_Max": ("f7597dd23e123a95", "fe5bb21ed62edb3a"),
    "GPT6_Astra_Max": ("b51ad3291c6e4333", "b51ad3291c6e4333"),
    "grok4p7_xhigh": ("8e648cef93186aca", "8e648cef93186aca"),
    "kimik3p1_max": ("0c211694ec66605b", "0c211694ec66605b"),
}
AUTOML_ELO = {  # version -> (commit, path, sha256)
    "v1": ("269a1bc24dafc8efd1a08d70f096335239c4c763", "assets/automl-elo/v1/RESULTS.json",
           "f1f7f3c304110bb17a2862d3405d2c55ff23ff6c1269c27189d0f16e7078e70a"),
    "v2": ("1177c4f9c1f51202e9c9ab861eeeb63d8732c33f", "assets/automl-elo/v2/RESULTS.json",
           "4f14193ab3058ce9d2168ddeca5de10e43847993706b63c74d8b2bb51e029d31"),
    "v3": ("706f2beca28fe6742d1236c98a6d690e495888e1", "assets/automl-elo/v3/RESULTS.json",
           "d6466e2c70b29f209253862ab8555da6d14a41ade08c8721f77e8c3dd95bfd60"),
}
BOOT_SEED, N_BOOT = 414160, 10_000
# Section D (addendum, owner 2026-11-07: "include all [other already-scored tabular models] in your table"): the other
# dev91 split. Its stored board, plus H257's published 25-entrant table (sha-pinned on origin/main), which also holds our
# older two-fold models. Different split and different u: never merged into the AUDIT numbers above.
TWOFOLD_ANCHOR = "h257.Opus5p5_Extra"
H257_TABLE = ("571873a38c961d960ca56b79cb086027ebf4c11f",
              "research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/frontier_elo_comparison.json",
              "e9e59a5c626f0747bf476d1dabcae76e46b4fbc5f18d2a6652d336a7f58439a6")
H257_SECTION = ("dl_val91_all_models_2fold_inner_cv_tournament", "s1_utility_gated_elo")
TWOFOLD_ONLY: dict = {}  # H413 gave all five former two-fold-only models an AUDIT row (now in SHARED)
SHARED = {  # two-fold board id -> (display name, the AUDIT-board row of the same model)
    "h257.Kumo_Tabular_L": ("Kumo-Tabular large", "kumo_tabular_large_e8.v1"),
    "h257.TabFM_Ensemble": ("TabFM ensemble", "tabfm_ensemble.v1"),
    "h257.TabFM": ("TabFM", "tabfm.v1"),
    "h257.PriorLabs_TabPFN_3p5": ("TabPFN 3.5 (8 estimators)", "tabpfn_v3p5.v1"),
    "h257.Opus5p5_Extra": ("Opus5p5_Extra", "h335.Opus5p5_Extra"),
    "h257.Gemini4_argon_high": ("Gemini4_argon_high", "h335.Gemini4_argon_high"),
    "h257.GPT6p1_Sol_Max": ("GPT6p1_Sol_Max", "h335.GPT6p1_Sol_Max"),
    "h257.GPT6_Astra_Max": ("GPT6_Astra_Max", "h335.GPT6_Astra_Max"),
    "h257.Kumo_Tabular_S": ("Kumo-Tabular small", "kumo_tabular_small_e8.v1"),
    "h257.PriorLabs_TabPFN_3p5_Single": ("TabPFN 3.5 (1 estimator)", "tabpfn_v3p5_e1.v1"),
    "h257.Mitra_v2": ("Mitra v2", "mitra_v2_zeroshot.v1"),
    "h257.TabDPT_1p3": ("TabDPT 1.3", "tabdpt_1p3.v1"),
    "h257.TabICL_v2": ("TabICL v2", "tabicl_v2.v1"),
    "h414.kimik3p1_max": ("kimik3p1_max", "h414.kimik3p1_max"),
}
TWOFOLD_LLM = {"h257.Opus5p5_Extra", "h257.Gemini4_argon_high", "h257.GPT6p1_Sol_Max", "h257.GPT6_Astra_Max",
               "h414.kimik3p1_max"}


def cls(e: str) -> str:
    if e in FRONTIER:
        return "frontier LLM"
    if e in FOUNDATION:
        return "foundation model"
    if e in AUTOML:
        return "AutoML"
    if e in CLASSIC:
        return "classic learner"
    return "JopeDime"


def name(e: str) -> str:
    return FRONTIER[e][0] if e in FRONTIER else LABEL.get(e, e)


def git_json(repo: Path, commit: str, path: str, sha: str) -> dict:
    raw = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise SystemExit(f"{commit[:10]}:{path} sha256 {got} != pinned {sha}")
    return json.loads(raw)


def gain_ci(H, a: dict, b: dict, fam_of: dict, rng: np.random.Generator) -> list:
    """95% family-bootstrap percentile interval of the unclipped family-equal mean gain."""
    _, D = H.family_contrast(a, b, fam_of, clip=None)
    ix = rng.integers(0, D.size, size=(N_BOOT, D.size))
    lo, hi = np.quantile(D[ix].mean(axis=1), [0.025, 0.975])
    return [float(lo), float(hi)]


def h2h(H, values: dict, fam_of: dict, focus: str, others: list, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for e in others:
        hd = H.headline(values[focus], values[e], fam_of)
        pt = H.primary_test(values[focus], values[e], fam_of)
        fams, D = H.family_contrast(values[focus], values[e], fam_of, clip=None)
        out[e] = {"headline": hd, "primary_clipped": pt, "gain_ci95_family_bootstrap": gain_ci(H, values[focus],
                  values[e], fam_of, rng), "family_gain": dict(zip(fams, map(float, D)))}
    return out


def bracket(order: list, e: str, shared: set) -> tuple:
    """The nearest shared entrant ranked above e and the nearest ranked below it (None at an end)."""
    i = order.index(e)
    above = next((x for x in reversed(order[:i]) if x in shared), None)
    below = next((x for x in order[i + 1:] if x in shared), None)
    return above, below


def twofold(H, L, board2: Path, repo: Path, fam_of: dict, elo: dict) -> dict:
    """Section D: the stored dev91-s1-two-fold-v1 board and H257's published two-fold table (stored files only)."""
    b2, fam2, v2, _, inc2, _ = L.load(board2)
    if inc2 or len(v2) != 14 or b2["board"] != "dev91-s1-two-fold-v1":
        raise SystemExit(f"expected the complete 14-entrant dev91-s1-two-fold-v1 board; incomplete={sorted(inc2)}")
    if fam2 != fam_of:
        raise SystemExit("the two dev91 boards do not hold the same 91 contexts / 19 families")
    for e in [*TWOFOLD_ONLY, *SHARED]:
        if e not in v2:
            raise SystemExit(f"missing two-fold entrant {e}")
    el2 = H.elo_panel(v2, fam2, TWOFOLD_ANCHOR)["elo"]
    stored = json.loads((board2 / "ladder.json").read_text())["elo"]["elo"]
    if any(abs(el2[e]["elo"] - stored[e]["elo"]) > 1e-6 for e in v2):
        raise SystemExit("two-fold Elo does not reproduce the board's stored ladder.json")
    fmu2 = {e: H.family_mean(v, fam2) for e, v in v2.items()}
    order2 = sorted(v2, key=lambda e: -el2[e]["elo"])
    rows = []
    for i, e in enumerate(order2, 1):
        nm = TWOFOLD_ONLY.get(e) or SHARED[e][0]
        audit = SHARED.get(e, (None, None))[1]
        rows.append({"rank": i, "id": e, "name": nm, "class": "frontier LLM" if e in TWOFOLD_LLM else "foundation model",
                     "elo": el2[e]["elo"], "ci95": el2[e]["ci95"], "family_mean_u": fmu2[e],
                     "audit_board_id": audit, "audit_elo": elo[audit]["elo"] if audit else None})
    # H257's published 25-entrant two-fold table: its 13 reference rows must match the stored board's family means
    tab = git_json(repo, *H257_TABLE)
    for k in H257_SECTION:
        tab = tab[k]
    for e in [x for x in v2 if not x.startswith("h414.")]:  # kimi is not in H257 s table
        k = e.split(".", 1)[1]
        if abs(tab[k]["family_mean"] - fmu2[e]) > 5e-6:
            raise SystemExit(f"H257 table {k} family mean {tab[k]['family_mean']} != board {fmu2[e]}")
    ref_keys = {e.split(".", 1)[1]: e for e in v2 if not e.startswith("h414.")}
    trows = []
    for i, (k, r) in enumerate(sorted(tab.items(), key=lambda kv: -kv[1]["joint_elo"]), 1):
        e = ref_keys.get(k)
        c = ("frontier LLM" if e in TWOFOLD_LLM else "foundation model") if e else "JopeDime (older)"
        nm = (TWOFOLD_ONLY.get(e) or SHARED[e][0]) if e else k
        trows.append({"rank": i, "key": k, "name": nm, "class": c, "joint_elo": r["joint_elo"],
                      "ci95": r["joint_elo_ci95"], "family_mean_u": r["family_mean"],
                      "two_fold_only": e in TWOFOLD_ONLY})
    shared_two = set(SHARED)
    place = {e: dict(zip(("above", "below"), bracket(order2, e, shared_two))) for e in TWOFOLD_ONLY}
    order_a = sorted(elo, key=lambda e: -elo[e]["elo"])
    ours_place = dict(zip(("above", "below"), bracket(order_a, OURS, {a for _, a in SHARED.values()})))
    sh2 = [e for e in order2 if e in SHARED]
    pos_a = {e: order_a.index(SHARED[e][1]) for e in sh2}
    swapped = [[x, y] for i, x in enumerate(sh2) for y in sh2[i + 1:] if pos_a[x] > pos_a[y]]
    return {"shared_order_two_fold": sh2, "shared_pairs_ordered_differently": swapped,"board": b2["board"], "board_sha256": b2["board_sha256"], "anchor": TWOFOLD_ANCHOR, "ladder": rows,
            "h257_table": {"commit": H257_TABLE[0], "path": H257_TABLE[1], "sha256": H257_TABLE[2],
                           "section": ".".join(H257_SECTION), "anchor": "R882", "rows": trows,
                           "reference_family_means_match_board": True},
            "placement_two_fold": place, "ours_placement_audit": ours_place,
            "scores_sha256": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((board2 / "scores").glob("*.jsonl"))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=str(RB / "boards" / "dev91-audit-v1"))
    ap.add_argument("--board2", default=str(RB / "boards" / "dev91-s1-two-fold-v1"))
    ap.add_argument("--tools", default=str(RB / "tools"))
    ap.add_argument("--repo", default=str(HERE.parents[2]))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import ladder as L  # noqa: E402  (imports h335_stats unchanged)
    H = L.H
    board, fam_of, values, meta, incomplete, _ = L.load(Path(a.board))
    if incomplete or len(values) != 31 or board["board"] != "dev91-audit-v1":
        raise SystemExit(f"expected the complete 31-entrant dev91-audit-v1 board; incomplete={sorted(incomplete)}")
    for e in [OURS, SECOND, ANCHOR, *FRONTIER, *FOUNDATION, *AUTOML, *CLASSIC]:
        if e not in values:
            raise SystemExit(f"missing entrant {e}")
    scores_sha = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((Path(a.board) / "scores").glob("*.jsonl"))}

    # A. the full ladder (the board's own statistic and field)
    elo = H.elo_panel(values, fam_of, ANCHOR)["elo"]
    fmu = {e: H.family_mean(v, fam_of) for e, v in values.items()}
    order = sorted(values, key=lambda e: -elo[e]["elo"])
    ladder = [{"rank": i, "id": e, "name": name(e), "class": cls(e), "elo": elo[e]["elo"], "ci95": elo[e]["ci95"],
               "family_mean_u": fmu[e], "geo_rmse_vs_mean_baseline": math.exp(-fmu[e])}
              for i, e in enumerate(order, 1)]

    # B. our strongest model vs everyone; H335_AGG and grok as second focus rows
    others = [e for e in order if e != OURS]
    vs_ours = h2h(H, values, fam_of, OURS, others, BOOT_SEED)
    vs_second = h2h(H, values, fam_of, SECOND, list(FRONTIER), BOOT_SEED + 1)
    vs_grok = h2h(H, values, fam_of, "h412.grok4p7_xhigh", [e for e in order if e != "h412.grok4p7_xhigh"],
                  BOOT_SEED + 2)
    vs_kimi = h2h(H, values, fam_of, "h414.kimik3p1_max", [e for e in order if e != "h414.kimik3p1_max"],
                  BOOT_SEED + 3)

    # C. frontier-only: dev91 field of the 5 LLMs (re-centred to a five-player mean of 1500, the AutoML v2
    #    convention) next to AutoML Elo v1 / v2. Different panels and different questions: compare ORDER only.
    fr_vals = {e: values[e] for e in FRONTIER}
    fr = H.elo_panel(fr_vals, fam_of, "h335.Opus5p5_Extra")["elo"]
    shift = 1500.0 - float(np.mean([fr[e]["elo"] for e in FRONTIER]))
    automl = {v: git_json(Path(a.repo), *spec) for v, spec in AUTOML_ELO.items()}
    rank = {}
    rank["dev91"] = {FRONTIER[e][0]: i for i, e in enumerate([x for x in order if x in FRONTIER], 1)}
    for v, r in automl.items():
        el = {m: (x["rating"] if isinstance(x, dict) else float(x)) for m, x in r["elo"].items()}
        rank[f"automl_{v}"] = {m: i for i, m in enumerate(sorted(el, key=lambda m: -el[m]), 1)}
        automl[v] = {"elo": el, "ci95": {m: (x.get("ci95") if isinstance(x, dict) else None)
                                         for m, x in r["elo"].items()}}
    frontier_rows = []
    for e, (nm, am) in FRONTIER.items():
        frontier_rows.append({
            "id": e, "name": nm, "dev91_elo": elo[e]["elo"], "dev91_ci95": elo[e]["ci95"],
            "dev91_elo_frontier_only": fr[e]["elo"] + shift,
            "dev91_ci95_frontier_only": [x + shift for x in fr[e]["ci95"]], "family_mean_u": fmu[e],
            "automl_v1_elo": automl["v1"]["elo"].get(am), "automl_v2_elo": automl["v2"]["elo"].get(am),
            "automl_v2_ci95": automl["v2"]["ci95"].get(am), "automl_v3_elo": automl["v3"]["elo"].get(am),
            "automl_v3_ci95": automl["v3"]["ci95"].get(am), "rank_automl_v3": rank["automl_v3"].get(am),
            "rank_dev91": rank["dev91"][nm],
            "rank_automl_v1": rank["automl_v1"].get(am), "rank_automl_v2": rank["automl_v2"].get(am),
            "dev91_source_sha256_16": SOURCE[nm][0], "automl_source_sha256_16": SOURCE[nm][1],
            "same_source_on_both_panels": SOURCE[nm][0] == SOURCE[nm][1]})
    same_order_v2 = [r["name"] for r in sorted(frontier_rows, key=lambda r: r["rank_dev91"]) if r["rank_automl_v2"]] == \
        sorted(rank["automl_v2"], key=lambda m: rank["automl_v2"][m])
    same_order_v3 = [r["name"] for r in sorted(frontier_rows, key=lambda r: r["rank_dev91"])] == \
        sorted(rank["automl_v3"], key=lambda m: rank["automl_v3"][m])
    same_order_v1 = [r["name"] for r in sorted(frontier_rows, key=lambda r: r["rank_dev91"]) if r["rank_automl_v1"]] == \
        sorted(rank["automl_v1"], key=lambda m: rank["automl_v1"][m])

    tf = twofold(H, L, Path(a.board2), Path(a.repo), fam_of, elo)

    best_fr = max(FRONTIER, key=lambda e: elo[e]["elo"])
    best_fd = max(FOUNDATION, key=lambda e: elo[e]["elo"])
    out = {
        "schema": "jope.h414.final-study/v1", "wave": "H414", "refreshes": "H412 final_study.json (v1/next 61ee043df7 / 5e0705eb9b)", "board": board["board"],
        "board_sha256": board["board_sha256"], "n_contexts": board["n_contexts"], "n_families": board["n_families"],
        "anchor": ANCHOR, "ours": OURS, "scores_sha256": scores_sha,
        "h335_stats_sha256": hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest(),
        "ladder_py_sha256": hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),
        "statistics": "h335_stats.elo_panel (family-grouped MM Bradley-Terry, 10,000-draw family bootstrap, seed "
                      "335160); headline + primary_test (clip 0.5, one-sided exact sign-flip, LOFO min); gain CI = "
                      f"{N_BOOT}-draw family bootstrap percentile, seed {BOOT_SEED}",
        "ladder": ladder, "ours_vs": vs_ours, "h335_agg_vs_frontier": vs_second, "grok_vs": vs_grok, "kimi_vs": vs_kimi,
        "frontier": {"rows": frontier_rows, "same_order_dev91_vs_automl_v3": same_order_v3,
                     "same_order_dev91_vs_automl_v2": same_order_v2,
                     "same_order_dev91_vs_automl_v1": same_order_v1,
                     "automl_sources": {v: {"commit": s[0], "path": s[1], "sha256": s[2]} for v, s in AUTOML_ELO.items()},
                     "frontier_only_centre": "six-player mean = 1500 (anchor fit on Opus5p5_Extra, then shifted)"},
        "two_fold": tf,
        "summary": {"ours_rank": order.index(OURS) + 1, "ours_elo": elo[OURS]["elo"], "best_frontier": best_fr,
                    "best_foundation": best_fd,
                    "ours_beats_all_frontier": all(vs_ours[e]["primary_clipped"]["pass"] for e in FRONTIER),
                    "ours_beats_any_foundation": any(vs_ours[e]["primary_clipped"]["pass"] for e in FOUNDATION),
                    "foundation_beat_ours": [e for e in FOUNDATION if vs_ours[e]["primary_clipped"]["inferior"]],
                    "grok_rank": order.index("h412.grok4p7_xhigh") + 1,
                    "kimi_rank": order.index("h414.kimik3p1_max") + 1, "n_entrants": len(order)},
    }
    od = Path(a.out)
    (od / "final_study.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], indent=1, sort_keys=True))


def pct(g: float) -> str:
    return f"{100 * (1 - math.exp(-g)):+.1f}%"


def row(e: str, r: dict) -> str:
    h, p, ci = r["headline"], r["primary_clipped"], r["gain_ci95_family_bootstrap"]
    verdict = "**better**" if p["pass"] else ("worse" if p["inferior"] else "not separated")
    return (f"| {name(e)} | {cls(e)} | {h['family_mean_gain']:+.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{pct(h['family_mean_gain'])} | {h['families_won']}/{h['families_tied']}/{h['families_lost']} | "
            f"{p['signflip_p']:.2g} | {p['lofo_min']:+.4f} | {verdict} |")


def tf_name(e: str | None) -> str:
    if e is None:
        return "—"
    return TWOFOLD_ONLY.get(e) or SHARED[e][0]


def render_d(o: dict) -> list:
    t, vs = o["two_fold"], o["ours_vs"]
    ht = t["h257_table"]
    md = ["", "## D. Tabular models scored only on the other dev91 split", "",
          "- **Five stored tabular models have no AUDIT score:** Kumo-Tabular small, TabPFN 3.5 (1 estimator), "
          "Mitra v2, TabDPT 1.3 and TabICL v2. They were scored once, on the same 91 contexts, under H257's two-fold "
          "split (H257 phase 2, 2026-11-04). Nothing here was rerun.",
          "- **They can't join table A.** The two splits score different rows with a different u (two-fold: "
          "−(0.9 · mean log RMSE ratio + 0.2 · SD of the two folds), −2.0 when the gate fails). The Elo scales differ, "
          "and the board rules never mix them (board README §4, rule 3).",
          "- **No model of ours is on both splits.** XSTACK_2L has no two-fold score, and `H257_REPLICA` is only the "
          "GP part of H257 (H335 `DISCLOSURES.md`), so it can't serve as a bridge.", "",
          f"### D.1 The stored two-fold board (13 entrants, anchor `{t['anchor']}` = 1500)", "",
          f"Board `{t['board']}`, sha256 `{t['board_sha256'][:16]}…`; this Elo reproduces its stored `ladder.json`.", "",
          "| Rank | Model | Class | Two-fold Elo | 95% CI | Family-mean u (two-fold) | Same model in table A |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in t["ladder"]:
        b = "**" if r["audit_board_id"] is None else ""
        a = "only on this split" if r["audit_board_id"] is None else f"AUDIT-Elo {r['audit_elo']:.1f}"
        md.append(f"| {r['rank']} | {b}{r['name']}{b} (`{r['id']}`) | {r['class']} | {r['elo']:.1f} | "
                  f"{r['ci95'][0]:.0f}–{r['ci95'][1]:.0f} | {r['family_mean_u']:.4f} | {a} |")
    md += ["", f"### D.2 H257's published two-fold table ({len(ht['rows'])} entrants, anchor {ht['anchor']} = 1500)", "",
           "- **This is the only stored table that puts the five next to models of ours.** Those are H257's older GP "
           "models, which were tuned on this split, so their rows are optimistic. XSTACK_2L is not in it.",
           "- **Its field is larger, so its Elo numbers differ from D.1.** Its 13 reference rows have the same "
           "family-mean u as the stored board (checked to 5e-6).",
           f"- Source: `{ht['commit'][:10]}:{ht['path']}` (origin/main), section `{ht['section']}`, sha256 "
           f"`{ht['sha256'][:16]}…`.", "",
           "| Rank | Model | Class | Elo | 95% CI | Family-mean u (two-fold) |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ht["rows"]:
        b = "**" if r["two_fold_only"] else ""
        md.append(f"| {r['rank']} | {b}{r['name']}{b} | {r['class']} | {r['joint_elo']:.1f} | "
                  f"{r['ci95'][0]:.0f}–{r['ci95'][1]:.0f} | {r['family_mean_u']:.4f} |")
    op = t["ours_placement_audit"]
    md += ["", "### D.3 Where the five sit relative to ours (rank chain, not a measurement)", "",
           f"- **On the AUDIT split, ours sits between {name(op['above'])} (above) and {name(op['below'])} (below)**, "
           "counting only models scored on both splits.",
           f"- **The two splits order the {len(t['shared_order_two_fold'])} shared models almost the same way.** "
           f"{len(t['shared_pairs_ordered_differently'])} of "
           f"{len(t['shared_order_two_fold']) * (len(t['shared_order_two_fold']) - 1) // 2} pairs flip: "
           + "; ".join(f"{tf_name(x)} / {tf_name(y)}" for x, y in t["shared_pairs_ordered_differently"])
           + ". So a rank chain is a guide, not a result.",
           "- Each of the five is placed the same way on the two-fold split.",
           "  - **Probably above ours:** the shared model just below it is significantly better than ours on AUDIT.",
           "  - **Probably below ours:** the shared model just above it is significantly worse than ours on AUDIT.",
           "  - **Not known:** neither holds.", "",
           "| Model | Two-fold: shared model above | Two-fold: shared model below | vs ours |", "| --- | --- | --- | --- |"]
    for e, p in t["placement_two_fold"].items():
        ab = SHARED[p["above"]][1] if p["above"] else None
        be = SHARED[p["below"]][1] if p["below"] else None
        if be and vs[be]["primary_clipped"]["inferior"]:
            v = f"probably above ours ({name(be)} is significantly better than ours)"
        elif ab and vs[ab]["primary_clipped"]["pass"]:
            v = f"probably below ours ({name(ab)} is significantly worse than ours)"
        else:
            v = "not known"
        md.append(f"| {TWOFOLD_ONLY[e]} | {tf_name(p['above'])} | {tf_name(p['below'])} | {v} |")
    md += ["", "- **To measure it instead:** score XSTACK_2L once on `dev91-s1-two-fold-v1` (board README §5). That "
           "runs our model on that board's split and reruns no reference model. It is a new dev91 read and needs its "
           "own LOCK.", "",
           "### D.4 Other stored score sets (different datasets or splits; listed for completeness)", "",
           "These come from [`research/elo/README.md`](../../elo/README.md); none is comparable with tables A or D.", "",
           "- **`p201b-train-audit-v1`:** the TRAIN board (201 contexts), for development only.",
           "- **dev91 S3 single-split** (precomputed, 2026-10): Kumo L / S, TabICL v2, TabDPT 1.3, Mitra v2 and "
           "TabPFN 3.5 on one split with no fold penalty.",
           "- **panel_v1** (240 regression + 240 binary datasets): Kumo 1874.5, AutoGluon 1601.3, CatBoost 1523.7, "
           "`jope-fused-net-v3` 1383.4.",
           "- **SOTArena Golden45** (45 public datasets, 305 methods): TabFM Ensemble 1934.9, TabPFN v3 1909.5, "
           "`jope-dime-regressor` 1687.0 (#31).",
           "- **AutoML Elo v1 / v2:** section C."]
    return md


def render(o: dict, order: list, elo: dict, fmu: dict) -> str:
    s, fr = o["summary"], o["frontier"]
    vs = o["ours_vs"]
    opus = vs[s["best_frontier"]]["headline"]
    gains = [vs[e]["headline"]["family_mean_gain"] for e in FRONTIER]
    md = ["# H412 FINAL study: our strongest model vs every frontier LLM on dev91-audit-v1", "",
          "**Board:** `dev91-audit-v1`, 91 dev contexts / 19 families, board sha256 "
          f"`{o['board_sha256'][:16]}…`. Every entrant fits on the SEARCH rows and predicts the AUDIT rows. "
          "u = −ln(RMSE / RMSE of the SEARCH-mean predictor). "
          f"AUDIT-Elo anchor `{o['anchor']}` = 1500.",
          "**Reads stored scores only.** No model was refit, rerun or rescored. The one new board row is "
          "`h412.grok4p7_xhigh`, scored once under the H412 LOCK. Our model is the stored `h364.XSTACK_2L` "
          "(owner's choice, 2026-11-06).", "",
          "## Result", "",
          f"- **Our strongest model, XSTACK_2L, ranks {s['ours_rank']} of 25 (AUDIT-Elo {s['ours_elo']:.1f}).**",
          "- **It beats every one of the 5 frontier LLM regressors.** Each contrast passes the H335 primary test "
          "(p < 0.025, LOFO > 0).",
          f"  - Against the highest-rated of them, {name(s['best_frontier'])}, the RMSE cut is "
          f"{pct(opus['family_mean_gain'])} and ours wins {opus['families_won']} of 19 families.",
          f"  - Across the 5, the RMSE cut runs from {pct(min(g for g in gains))} "
          f"({name(min(FRONTIER, key=lambda e: vs[e]['headline']['family_mean_gain']))}) to "
          f"{pct(max(g for g in gains))} ({name(max(FRONTIER, key=lambda e: vs[e]['headline']['family_mean_gain']))}).",
          "- **It does not beat any of the 5 foundation models.** Kumo, TabFM, the TabFM ensemble, TabPFN 3.5 and "
          "TabPFN v3 all rank above it.",
          f"  - Kumo-Tabular large leads the board: its RMSE is {pct(-vs['kumo_tabular_large_e8.v1']['headline']['family_mean_gain'])[1:]} "
          "lower than ours.",
          f"  - Foundation models significantly better than ours (mirrored primary test): "
          f"{', '.join(name(e) for e in s['foundation_beat_ours']) or 'none'}.",
          f"- **The new frontier entrant grok4p7_xhigh ranks {s['grok_rank']} of 25, last on the board.**",
          f"  - The frontier-only order on dev91 is the same as AutoML Elo v2: "
          f"{'yes' if fr['same_order_dev91_vs_automl_v2'] else 'no'}. AutoML Elo is a separate TRAIN panel.",
          "- **Five more stored tabular models** (Kumo-Tabular small, TabPFN 3.5 with 1 estimator, Mitra v2, TabDPT 1.3, "
          "TabICL v2) were scored only on the other dev91 split, so they have no AUDIT-Elo. Section D lists them with "
          "their stored two-fold Elo and where they sit relative to ours.", "",
          "## A. Full ladder (25 entrants, all 91/91)", "",
          "| Rank | Entrant | Class | AUDIT-Elo | 95% CI | Family-mean u | Geo-mean RMSE / mean baseline |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in o["ladder"]:
        b = "**" if r["id"] in (o["ours"],) else ""
        md.append(f"| {r['rank']} | {b}{r['name']}{b} (`{r['id']}`) | {r['class']} | {b}{r['elo']:.1f}{b} | "
                  f"{r['ci95'][0]:.0f}–{r['ci95'][1]:.0f} | {r['family_mean_u']:.4f} | "
                  f"{r['geo_rmse_vs_mean_baseline']:.3f} |")
    md += ["", "## B. XSTACK_2L head to head", "",
           "- **Gain** is the family-equal mean of u(XSTACK_2L) − u(other), with its 95% family-bootstrap interval.",
           "- **RMSE** is the cut 1 − exp(−gain). Positive means ours is better.",
           "- **p** and **LOFO** come from the H335 primary test (clip ±0.5, one-sided exact sign-flip over 19 "
           "families).",
           "- **Verdict:** \"better\" means the test passes. \"worse\" means the mirrored test passes.", "",
           "### vs the frontier LLM regressors", "",
           "| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [row(e, vs[e]) for e in order if e in FRONTIER]
    md += ["", "### vs the foundation models, AutoML and the classic learners", "",
           "| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [row(e, vs[e]) for e in order if e in FOUNDATION + AUTOML + CLASSIC]
    md += ["", "### vs the other JopeDime entrants", "",
           "| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [row(e, vs[e]) for e in order if cls(e) == "JopeDime" and e != o["ours"]]
    md += ["", "### Second row: H335_AGG (ours, the XSTACK_2L input) vs the frontier LLMs", "",
           "| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [row(e, o["h335_agg_vs_frontier"][e]) for e in order if e in FRONTIER]
    md += ["", "## C. The frontier LLMs: dev91 next to AutoML Elo", "",
           "- **The two panels answer different questions.**",
           "  - dev91 AUDIT: 91 dev contexts, one SEARCH → AUDIT fit.",
           "  - AutoML Elo: 64 TRAIN families × 3 folds × 3 seeds.",
           "- Their numbers are on different scales; compare the order only (PROTOCOL §4).",
           "- Our GP is never placed on the AutoML panel, because it was selected on TRAIN.",
           "- The \"frontier-only\" column re-fits the board's Elo on the 5 LLMs alone and centres their mean on "
           "1500, which is AutoML v2's convention.", "",
           "| Model | dev91 AUDIT-Elo (25-field) | dev91 frontier-only | AutoML v1 | AutoML v2 [95% CI] | "
           "Rank dev91 / v1 / v2 | Same source on both panels |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(fr["rows"], key=lambda r: r["rank_dev91"]):
        v1 = "—" if r["automl_v1_elo"] is None else f"{r['automl_v1_elo']:.1f}"
        ci = r["automl_v2_ci95"] or [float("nan")] * 2
        same = "yes" if r["same_source_on_both_panels"] else (
            f"**no**: dev91 `{r['dev91_source_sha256_16']}…`, AutoML `{r['automl_source_sha256_16']}…`")
        md.append(f"| {r['name']} | {r['dev91_elo_25']:.1f} | {r['dev91_elo_frontier_only']:.1f} | {v1} | "
                  f"{r['automl_v2_elo']:.1f} [{ci[0]:.0f}–{ci[1]:.0f}] | {r['rank_dev91']} / "
                  f"{r['rank_automl_v1'] or '—'} / {r['rank_automl_v2']} | {same} |")
    md += ["", f"- **The order is the same on both panels:** dev91 vs AutoML v2 "
           f"{'yes' if fr['same_order_dev91_vs_automl_v2'] else 'no'}; dev91 vs AutoML v1 (4 entrants) "
           f"{'yes' if fr['same_order_dev91_vs_automl_v1'] else 'no'}.",
           "- **Sol caveat:** the dev91 Sol row (H335 build tree) was compiled from source `f7597dd2…`. That is the "
           "2026-11-02 refactor of `crates/jope-automl/src/GPT6p1_Sol_Max.rs`, split into 3 modules (commits "
           "`d4c48d20f9` … `793667e62a`).",
           "  - The AutoML Sol score belongs to the earlier blob `fe5bb21e…`, as the Grok lane records in "
           "`assets/comparisons/protocol-elo/v1/tokens.json` (commit `3309223b45`, PR #2154, open).",
           "  - Nobody has checked that the two sources behave the same, so Sol's agreement across panels is not a "
           "same-code check. The other four rows use the same source on both panels.",
           "- AutoML inputs, read from git and sha-checked:"]
    for v, src in fr["automl_sources"].items():
        md.append(f"  - {v}: `{src['commit'][:10]}:{src['path']}`, sha256 `{src['sha256'][:16]}…`.")
    md += render_d(o)
    md += ["", "## Limits", "",
           "- **dev91 is a reuse cohort.** H257 was tuned on it.",
           "  - The 5 H381 / H410 contexts were adaptively exposed: `ac07466e`, `8eeca816`, `016648dc`, "
           "`6ecbb7d5` and `b1a5c1a1`.",
           "  - XSTACK_2L was selected on TRAIN (H364) and scored once (H369). It was not tuned on those "
           "contexts, and nothing in this study was chosen on dev91.",
           "- **Elo depends on the field.** Adding grok moved XSTACK_2L from 1726.8 to 1724.6; its family-mean u is "
           "unchanged.",
           "  - Effect sizes come from the head-to-head table, not from Elo differences.",
           "- **One run per entrant (seed 42).** The confidence intervals are family bootstraps; there is no seed "
           "variance.",
           "- **grok4p7_xhigh provenance:** its authorship record says `fresh_session: false`. This is recorded and "
           "does not affect scoring.",
           "- **Not done:** XSTACK_2L has no two-fold score, so section D ranks it against the five two-fold-only "
           "models by rank chain only. Scoring it once on `dev91-s1-two-fold-v1` (board README §5) would measure it; "
           "that is a new dev91 read and needs its own LOCK.",
           "- **Not done:** no new candidate of ours (owner's choice).",
           "  - The H410 TRAIN lead (pooled runs + strong-learner library, +8.1% RMSE on 120 TRAIN families, "
           "p 0.0006) is the next candidate. It needs its own LOCK before any dev91 read.", "",
           "Generated by `final_study_h412.py` from the stored board files; `final_study.json` has every number."]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
