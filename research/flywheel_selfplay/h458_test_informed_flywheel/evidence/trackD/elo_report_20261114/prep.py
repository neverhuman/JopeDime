"""Derives every table of the Elo report from the collected bundle. Output: report_tables.json."""
import collections
import calendar
import time as _time
import hashlib
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
datetime_ts = lambda u: calendar.timegm(_time.strptime(u, "%Y-%m-%dT%H:%M:%SZ"))  # noqa: E731
d = json.load(open(HERE + "elo_report_bundle.json"))
arms, field, board = d["arms"], d["field"], d["board"]

# ----------------------------------------------------------------------------- addendum: r4 epoch 48 (the run's final epoch)
# Scored at 2026-11-15 00:08 UTC, after the bundle was collected. The reader below is checked against the bundle's own epoch-47
# records before it is trusted for epoch 48.
def h450_readout_arms(path: str, label: str) -> tuple:
    j = json.load(open(HERE + path))
    out = []
    for k, v in j.items():
        if not (isinstance(v, dict) and "entrant" in v and "elo" in v):
            continue
        out.append({"file": label, "wave": "H450", "arm": k, "entrant": v["entrant"], "elo": v["elo"]["elo"], "ci95": v["elo"]["ci95"],
                    "family_mean_u": v["family_mean_u"], "rank33": v["rank_if_placed_against_frozen_33"], "n_entrants": v["n_entrants_in_calc"],
                    "gap_elo": v["gap_elo"], "wtl": {r: p_["wtl"] for r, p_ in v["pairs"].items()}, "gain": {r: p_["gain"] for r, p_ in v["pairs"].items()},
                    "kumo_l_in_fit": next(t["elo"] for t in v["top_10_order"] if t["entrant"] == "kumo_tabular_large_e8.v1"),
                    "mtime": int(os.path.getmtime(HERE + path)), "flags": {"canonical_promotion_forbidden": j.get("canonical_promotion_forbidden"), "schema": j.get("schema")}})
    return j, out


_, chk47 = h450_readout_arms("r4_g47_H450_DEV91.json", "H450/diag_results/r4_g47_r4_L_A_r4_L_B/H450_DEV91.json")
old47 = {a["arm"]: a for a in arms if a["file"] == "H450/diag_results/r4_g47_r4_L_A_r4_L_B/H450_DEV91.json"}
assert len(chk47) == len(old47) == 7
for a in chk47:
    for k in ("entrant", "elo", "ci95", "family_mean_u", "rank33", "n_entrants", "gap_elo", "wtl", "gain", "kumo_l_in_fit"):
        assert a[k] == old47[a["arm"]][k], (a["arm"], k)
G48, new48 = h450_readout_arms("r4_g48_H450_DEV91.json", "H450/diag_results/r4_g48_r4_L_A_r4_L_B/H450_DEV91.json")
assert not any("r4_g48" in a["file"] for a in arms)
arms = arms + new48
N_BUNDLE_ARMS, N_BUNDLE_FILES = len(d["arms"]), len({a["file"] for a in d["arms"]})
# ----------------------------------------------------------------------------- addendum, revision 14: every dev91 placement written after the bundle
# collect_rev14.py (run on xbabe2) read the five H456 Stage 3 matrices with a reader checked against the bundle's 456 H455 records, and the seven
# H462 price receipts with their look-ledger lines. H456 arms join the ranked tables (in-corpus scorings like H455's). H462's placements are
# dev91 development reads of an adaptive sequence: only locked entrants enter the ladder; every placement is listed, unranked, in its own table.
ADD = json.load(open(HERE + "arms_addendum_rev14.json"))
assert ADD["bundle_generated_utc"] == d["generated_utc"] and ADD["matrix_reader_checked_on_h455_arms"] == 456
h456 = [a for a in ADD["arms"] if a["wave"] == "H456"]
h462 = [a for a in ADD["arms"] if a["wave"] == "H462"]
assert len(h456) + len(h462) == len(ADD["arms"]) and not any(a["wave"] in ("H456", "H462") for a in arms)
for a in h456:
    a.setdefault("flags", {})
arms = arms + h456
# revision 17: the 19 H450 fleet readouts of 2026-11-13 the first bundle left out (Track S reconciler, finding R16-2); collect_rev17.py, same reader as r4 epoch 48
ADD17 = json.load(open(HERE + "arms_addendum_rev17.json"))
assert ADD17["bundle_generated_utc"] == d["generated_utc"] and ADD17["reader_checked_on_h450_arms"] == 236 and all(a["wave"] == "H450F" for a in ADD17["arms"])
N_FLEET_ARMS = len(ADD17["arms"])
# the fleet init control (mode v4_init_control: the all_non_d91_L init, no all-data episode) is listed but kept out of the fleet system means, as the r4 rows
# drop their init epoch (Track S reconciler, R17-3)
for a_ in ADD17["arms"]:
    if "fleet_x1_v4_init_ctrl" in a_["file"]:
        a_["wave"] = "H450F-init"
arms = arms + ADD17["arms"]
tsv = [ln.rstrip("\n").split("\t") for ln in open(HERE + "r4_traj_final.tsv") if ln.strip()]
tsv_rows = [{"tag": t[0], "champion": float(t[1]), "icl_only": float(t[2]), "legacy": float(t[3]), "no_gp": float(t[4]),
             "control_delta": float(t[5]) if len(t) > 5 and t[5] else None} for t in tsv]
old_traj = {r["tag"]: r for r in d["r4_traj"]}
for r in tsv_rows:
    if r["tag"] in old_traj:
        assert all(r[k] == old_traj[r["tag"]][k] for k in ("champion", "icl_only", "legacy", "no_gp")), r["tag"]
assert len(tsv_rows) == len(d["r4_traj"]) + 1 and tsv_rows[-1]["tag"].startswith("r4_g48")
d["r4_traj"] = tsv_rows
d["r4_final"] = {"matched_random_split_d91_delta": G48["matched_random_split_d91_delta"]}
n48 = {a["arm"]: a["elo"] for a in new48}
assert abs(n48["in_corpus_random_split_champion_91"] - tsv_rows[-1]["champion"]) < 1e-9 and abs(n48["in_corpus_legacy_guard_stack_91"] - tsv_rows[-1]["legacy"]) < 1e-9
TIE = 1e-4

# ----------------------------------------------------------------------------- the frozen field
CLASS = {}
NAMES = {
    "kumo_tabular_large_e8.v1": "Kumo Tabular Large (8 estimators)",
    "kumo_tabular_small_e8.v1": "Kumo Tabular Small (8 estimators)",
    "tabfm_ensemble.v1": "TabFM 1.0 ensemble (Google)",
    "tabfm.v1": "TabFM 1.0 (Google)",
    "tabpfn_v3p5.v1": "TabPFN 3.5 (Prior Labs)",
    "tabpfn_v3p5_e1.v1": "TabPFN 3.5, 1 estimator (Prior Labs)",
    "tabpfn_v3.v1": "TabPFN v3 (Prior Labs)",
    "mitra_v2_zeroshot.v1": "Mitra v2, zero-shot (AutoGluon)",
    "tabicl_v2.v1": "TabICL v2 (soda-inria)",
    "tabdpt_1p3.v1": "TabDPT 1.3 (Layer6)",
    "autogluon_300s.v1": "AutoGluon Tabular (default, 300 s)",
    "catboost.v1": "CatBoost (1000 x depth 6)",
    "xgboost.v1": "XGBoost (500 x depth 6)",
    "lightgbm.v1": "LightGBM (500 x 31 leaves)",
    "random_forest.v1": "Random forest (300 trees)",
    "ridge.v1": "Ridge (RidgeCV)",
    "h335.Opus5p5_Extra": "Opus 5.5 Extra (LLM regressor)",
    "h335.Gemini4_argon_high": "Gemini 4 argon high (LLM regressor)",
    "h335.GPT6p1_Sol_Max": "GPT-6.1 Sol Max (LLM regressor)",
    "h335.GPT6_Astra_Max": "GPT-6 Astra Max (LLM regressor)",
    "h412.grok4p7_xhigh": "Grok 4.7 xhigh (LLM regressor)",
    "h414.kimik3p1_max": "Kimi K3.1 max (LLM regressor)",
    "h448.APEX_FAIR_CHAMPION": "H448 APEX fair champion (pre-registered)",
    "h416.SXB": "H416 SXB",
    "h364.XSTACK_2L": "H364 two-level stack",
    "h335.H335_AGG": "H335 aggregate",
    "h335.OLD_MATCHED_CV3": "H335 old matched (CV-3)",
    "h335.OLD_MATCHED": "H335 old matched",
    "h335.H317_BASE": "H317 base",
    "h335.H257_REPLICA": "H257 replica (anchor = 1500)",
    "h335.BANK_NN8_PORTFOLIO": "H335 bank NN-8 portfolio",
    "h335.BANK_NN8_PICK": "H335 bank NN-8 pick",
    "h335.BANK_NN1": "H335 bank NN-1",
}
for k in field:
    if k.startswith(("kumo", "tabfm", "tabpfn", "mitra", "tabicl", "tabdpt")):
        CLASS[k] = "Tabular foundation model"
    elif k.split(".")[0] in ("autogluon_300s", "catboost", "xgboost", "lightgbm", "random_forest", "ridge"):
        CLASS[k] = "AutoML, trees, linear"
    elif "LLM regressor" in NAMES.get(k, ""):
        CLASS[k] = "LLM regressor"
    else:
        CLASS[k] = "Ours, earlier wave"

field_rows = []
for k, v in field.items():
    kind = "external" if CLASS[k] != "Ours, earlier wave" else ("ours_fair" if k == "h448.APEX_FAIR_CHAMPION" else "ours_earlier")
    field_rows.append({
        "id": k, "name": NAMES.get(k, k), "cls": CLASS[k], "kind": kind, "elo": v["elo"], "lo": v["ci95"][0], "hi": v["ci95"][1],
        "family_mean_u": v["family_mean_u"], "wall_s": v["median_wall_s"], "label": v["label"],
        "closure": "fair, pre-registered" if k == "h448.APEX_FAIR_CHAMPION" else ("reference" if kind == "external" else "earlier (board reuse cohort)"),
    })
field_rows.sort(key=lambda r: -r["elo"])
for i, r in enumerate(field_rows, 1):
    r["field_rank"] = i

# ----------------------------------------------------------------------------- our systems (best scored instance of each distinct recipe)
def ckpt_label(path: str) -> str:
    b = os.path.basename(path).replace(".json", "")
    if "diag_results" in path and "fleet" in path:
        return os.path.basename(os.path.dirname(path)).replace("_fleet_L_x1_fleet_L_x1", " fleet L-x1").replace("_", " ")
    if "diag_results" in path:
        m = re.search(r"r4_g(\d+)", path)
        return f"r4 epoch {int(m.group(1))}" if m else b
    b = re.sub(r"_MATRIX$", "", b)
    b = re.sub(r"^H45\d_", "", b)
    return b.replace("_", " ").lower()

SYSTEMS = [
    # (key, wave filter, arm name, display name, closure, component group)
    ("s_stack_icl", "H455", "stack_no_ugp_post_h454", "H455 stack: GP/tree pool + flywheel ICL columns", "in-corpus", "Full stack"),
    ("s_stack_ugp1000", "H455", "stack_ugp_g1000_cons_post_h454", "H455 stack + Ultra-GP, 1000 generations", "in-corpus", "Full stack"),
    ("s_r4_legacy", "H450", "in_corpus_legacy_guard_stack_91", "H450 r4 legacy guard stack (best of 34 epochs)", "in-corpus", "Full stack"),
    ("s_stack_ugp10", "H453", "stack_ugp_g10_cv7", "H453 stack + Ultra-GP, 10 generations", "in-corpus", "Full stack"),
    ("s_h452_guard", "H452", "phase0a_stack_cv7_guard", "H452 stack with CV-7 guard", "in-corpus", "Full stack"),
    ("s_h456_stack", "H456", "stack_no_ugp_post_h454", "H456 Stage 3 stack: GP/tree pool + flywheel ICL columns", "in-corpus", "Full stack"),
    ("s_fleet_legacy", "H450F", "in_corpus_legacy_guard_stack_91", "H450 fleet L-x1 legacy guard stack", "in-corpus", "Full stack"),
    ("s_fleet_champion", "H450F", "in_corpus_random_split_champion_91", "H450 fleet L-x1 champion", "in-corpus", "Full stack"),
    ("s_legacy_ctrl", "H452", "ctrl_h450_g16_legacy", "Legacy stack control (r4 epoch-16 ICL columns)", "in-corpus", "Full stack"),
    ("s_pure_neural", "H455", "neural_104m_all_post_h454", "H455 neural columns (incl. the GP-residual hybrid)", "in-corpus", "Neural only"),
    ("s_hyb_gpresid", "H455", "hyb_104m_gpresid_cv7_post_h454", "H455 ICL with GP-residual column", "in-corpus", "Neural only"),
    ("s_h449_unfair_stack", "H449", "unfair_all_d91_stack", "H449 stack, ICL trained on all of dev91", "in-corpus", "Full stack"),
    ("s_neural_e8", "H452", "neural_104m_e8_only", "H452 neural 104M e8 columns only", "in-corpus", "Neural only"),
    ("s_hyb_only", "H455", "hyb_104m_only_cv7_post_h454", "H455 ICL hybrid columns only", "in-corpus", "Neural only"),
    ("s_r4_champion", "H450", "in_corpus_random_split_champion_91", "H450 r4 champion (best of 34 epochs)", "in-corpus", "Full stack"),
    ("s_r4_nogp", "H450", "in_corpus_neural_plus_tab_no_gp_91", "H450 r4 neural + tabular, no GP", "in-corpus", "Neural only"),
    ("s_pure_cv7", "H455", "pure_104m_cv7_post_h454", "H455 ICL 104M alone (CV-7 columns)", "in-corpus", "ICL alone"),
    ("s_r4_icl", "H450", "in_corpus_icl_boost_only_91", "H450 r4 ICL only (best of 34 epochs)", "in-corpus", "ICL alone"),
    ("s_h449_unfair_icl", "H449", "unfair_all_d91_icl_only", "H449 ICL only, trained on all of dev91", "in-corpus", "ICL alone"),
    ("s_pure_only", "H454", "pure_104m_only_cv7_post_h454", "H454 ICL 104M alone (single column family)", "in-corpus", "ICL alone"),
    ("f_fair_oos", "H453", "fair_oos_0pct_d91_no_ugp", "H453 fair stack: ICL trained with 0% dev91", "fair", "Full stack"),
    ("f_h449_clean", "H452", "ctrl_h449_clean_g00_legacy", "Fair legacy stack control (H449 clean epoch-0 ICL)", "fair", "Full stack"),
    ("f_fair_ugp100", "H453", "fair_oos_0pct_d91_ugp_g100_raw_cv7", "H453 fair stack + Ultra-GP, 100 generations", "fair", "Full stack"),
    ("f_pure_ugp50", "H453", "pure_ugp_g50_cv7", "Ultra-GP alone, 50 generations (no ICL, no trees)", "fair", "GP alone"),
    ("f_h449_fair_stack", "H449", "fair_all_non_d91_stack", "H449 fair stack (ICL trained on all non-dev91)", "fair", "Full stack"),
    ("f_r4_fair_ctrl", "H450", "fair_zero_d91_upgraded_control_91", "H450 r4 fair control (0% dev91)", "fair", "Full stack"),
    ("f_h449_fair_icl", "H449", "fair_all_non_d91_icl_only", "H449 fair ICL only (trained on all non-dev91)", "fair", "ICL alone"),
]
REFS = ["kumo_tabular_large_e8.v1", "tabfm_ensemble.v1", "kumo_tabular_small_e8.v1", "tabfm.v1", "tabpfn_v3p5.v1", "tabpfn_v3p5_e1.v1", "h448.APEX_FAIR_CHAMPION", "h416.SXB"]
systems = []
for key, wave, arm, name, closure, group in SYSTEMS:
    cand = [a for a in arms if a["arm"] == arm and a["wave"] == wave]
    if not cand:
        cand = [a for a in arms if a["arm"] == arm]
    if not cand:
        raise SystemExit(f"no rows for {arm}")
    b = max(cand, key=lambda a: a["elo"])
    es = sorted({round(a["elo"], 2) for a in cand})
    systems.append({
        "key": key, "id": b["entrant"], "arm": arm, "name": name, "closure": closure, "group": group, "wave": b["wave"],
        "elo": b["elo"], "lo": b["ci95"][0], "hi": b["ci95"][1], "rank34": b["rank33"], "family_mean_u": b["family_mean_u"],
        "n_scored": len({(a["file"]) for a in cand}), "elo_min": es[0], "elo_max": es[-1], "checkpoint": ckpt_label(b["file"]), "file": b["file"],
        "gap": {r: (b["gap_elo"] or {}).get("vs_" + r) for r in REFS}, "wtl": {r: (b["wtl"] or {}).get(r) for r in REFS},
        "kumo_l_in_fit": b["kumo_l_in_fit"], "kind": "ours_fair" if closure == "fair" else "ours_incorpus",
    })
# rank for the H449 rows is not stored: derive from the frozen field
for s in systems:
    if s["rank34"] is None:
        s["rank34"] = 1 + sum(1 for r in field_rows if r["elo"] > s["elo"])

# A system scored more than once (r4 epochs, the 12 H455 checkpoints, repeated matrices) is ranked by the MEAN over its scored
# instances: the pre-specified value its owners asked for. The maximum over repeated noisy evaluations is selection-biased and is
# kept only as `elo_best` for the wave sections.
R4_ROWS = {"s_r4_legacy": "H450 r4 legacy guard stack", "s_r4_champion": "H450 r4 champion", "s_r4_nogp": "H450 r4 neural + tabular, no GP",
           "s_r4_icl": "H450 r4 ICL only"}
for s in systems:
    cand = [a for a in arms if a["arm"] == s["arm"] and a["wave"] == s["wave"]]
    if s["key"] in R4_ROWS:
        cand = [a for a in cand if "r4_g00_init" not in a["file"]]
    by_file = {a["file"]: a for a in cand}
    cand = list(by_file.values())
    n = len(cand)
    s["n_inst"] = n
    s["elo_best"], s["best_checkpoint"] = s["elo"], s["checkpoint"]
    if s["key"] == "f_r4_fair_ctrl":
        s["name"] = "H449 init (0% dev91) under the H450 stack"
        s["checkpoint"] = "g00 init"
        s["n_inst"] = 1
        continue
    if n > 1:
        s["elo"] = statistics.mean(a["elo"] for a in cand)
        s["lo"] = statistics.mean(a["ci95"][0] for a in cand)
        s["hi"] = statistics.mean(a["ci95"][1] for a in cand)
        for r in REFS:
            g = [a["gap_elo"].get("vs_" + r) for a in cand if (a["gap_elo"] or {}).get("vs_" + r) is not None]
            s["gap"][r] = statistics.mean(g) if g else None
        ranks = [a["rank33"] for a in cand if a["rank33"]]
        s["rank34"] = statistics.mode(ranks) if ranks else 1 + sum(1 for r in field_rows if r["elo"] > s["elo"])
        unit = "epochs" if s["key"] in R4_ROWS else ("checkpoints" if s["wave"] in ("H455", "H456") else "scorings")
        s["name"] = f"{R4_ROWS.get(s['key'], s['name'])} (mean of {n} {unit})"
        s["checkpoint"] = f"mean of {n} {unit}"
        if s["key"] in R4_ROWS:
            s["n_epochs"] = n
# H462: the locked entrants of the adaptive dev91 sequence (look ledger passes 3, 6 and 7). Pass 4's number is the same rule as pass 6 before a
# column-name fix (superseded); pass 5 locked a withdrawn entrant, an in-corpus one and a near-duplicate of pass 3, listed in section 17 only.
def h462_arm(fname: str, arm: str) -> dict:
    c = [a for a in h462 if a["file"] == "H462/" + fname and a["arm"] == arm]
    assert len(c) == 1, (fname, arm, len(c))
    return c[0]
H462_LOCKED = (("f_h462_p3", "M6_FAIR_R4E8_PRICE.json", "fair_h448_r4e8_m_multiseed.pred_avg_4_s12_only", "H462 fair stack, 4 predictions averaged (GP seeds x neural seeds)", 3),
               ("f_h462_p6", "M6_CVREP8_PRICE_V2.json", "cvrep4_pred_avg_16_s12", "H462 fair stack, 16 predictions averaged (+ repeated cross-validation)", 6),
               ("f_h462_p7", "M6_TABCV7_CVREP_PRICE.json", "tabcv7_cvrep8_prec48_all", "H462 fair stack, 48 predictions averaged (+ per-table classical learners)", 7),
               ("s_h462_p5_mv", "M6_MULTISEED_PRICE.json", "m_valley_zero_g16_multiseed.prec_avg_6_all", "H462 all-data M-valley stack, 6 predictions averaged (GP seeds x neural seeds)", 5),
               ("s_h462_p5_hyb", "M6_MULTISEED_PRICE.json", "fair_cv7_hyb_m_multiseed.prec_avg_6_all", "H462 hybrid stack, 6 predictions averaged (GP anchors carried dev91)", 5))
for key_, fname_, arm_, name_, pass_ in H462_LOCKED:
    a_ = h462_arm(fname_, arm_)
    assert a_["locked"] and a_["pass_index"] == pass_
    fair_ = a_["label"] == "0% dev91"
    assert fair_ == key_.startswith("f_"), (key_, a_["label"])
    lab_ = "0% dev91" if fair_ else ("in-corpus (1.8% dev91 in ICL pre-training)" if key_ == "s_h462_p5_mv" else "in-corpus (GP anchors carried dev91)")
    systems.append({"key": key_, "id": a_["entrant"], "arm": arm_, "name": name_, "closure": "fair" if fair_ else "in-corpus", "group": "Full stack", "wave": "H462", "elo": a_["elo"], "lo": a_["ci95"][0],
                    "hi": a_["ci95"][1], "rank34": a_["rank33"], "family_mean_u": a_["family_mean_u"], "n_scored": 1, "elo_min": round(a_["elo"], 2), "elo_max": round(a_["elo"], 2),
                    "checkpoint": f"look ledger pass {pass_}", "file": a_["file"], "gap": {r: a_["gap_elo"].get("vs_" + r) for r in REFS}, "wtl": {r: None for r in REFS},
                    "kumo_l_in_fit": a_["kumo_l_in_fit"], "kind": "ours_fair" if fair_ else "ours_incorpus", "n_inst": 1, "elo_best": a_["elo"], "best_checkpoint": f"look ledger pass {pass_}",
                    "ladder_label": f"{lab_}, locked (pass {pass_})"})
systems.sort(key=lambda s: -s["elo"])

ladder = [dict(r, source="frozen field") for r in field_rows] + [
    {"id": s["id"], "name": s["name"], "cls": "Ours, " + s["closure"], "kind": s["kind"], "elo": s["elo"], "lo": s["lo"], "hi": s["hi"],
     "family_mean_u": s["family_mean_u"],
     # Ultra-GP columns were evolved under the engine's default prior, which saw 289 dev91 random-split winners (confirmed by the runs' owner, 2026-11-15)
     "closure": s.get("ladder_label") or (("fair closure, post hoc" + ("; GP prior in-corpus" if re.search(r"(?<!no_)ugp", s["arm"].lower()) else "")) if s["closure"] == "fair" else s["closure"]), "source": s["checkpoint"],
     "rank34": s["rank34"], "key": s["key"], "wave": s["wave"]}
    for s in systems
]
L2E = json.load(open(HERE + "look2_SCORE.json"))
assert L2E["rank_arm"] == 1 and L2E["official"] in (True, False)
for key_, e_, gap_, name_, kind_, clos_, rank_ in (
        ("look2_arm", L2E["elo_arm"], L2E["gap_vs_kumo_l_arm"], "H458 all-data stack + 19 dev91 family specialists", "ours_recall",
         "in-corpus recall: every scored audit row was in its family's training pool with its label; locked (look 2)", L2E["rank_arm"]),
        ("look2_nr", L2E["elo_arm_nr"], L2E["gap_vs_kumo_l_arm_nr"], "H458 look-2 stack without the specialists (same look)", "ours_incorpus",
         "in-corpus, locked control (look 2)", L2E["rank_arm_nr"])):
    ladder.append({"id": "h458.look2." + key_, "name": name_, "cls": "Ours, in-corpus", "kind": kind_, "elo": e_["elo"], "lo": e_["ci95"][0], "hi": e_["ci95"][1],
                   "family_mean_u": None, "closure": clos_, "source": "look 2, locked", "rank34": rank_, "key": key_, "wave": "H458", "fit": "look2_34", "gap_kumo_l": gap_})
ladder.sort(key=lambda r: -r["elo"])
for i, r in enumerate(ladder, 1):
    r["pos"] = i

# ----------------------------------------------------------------------------- our 50 highest-scoring arms (deduplicated)
seen = {}
for a in sorted(arms, key=lambda a: a["mtime"]):
    seen.setdefault((a["arm"], round(a["elo"], 6), round(a["family_mean_u"] or 0.0, 6)), a)  # one row per distinct prediction (Track S reconciler, R16-1)
uniq = sorted(seen.values(), key=lambda a: -a["elo"])
def closure_of(a) -> str:
    n = a["arm"]
    if n.startswith(("fair_", "ctrl_h449_clean")) or n.startswith("pure_ugp"):
        return "fair"
    return "in-corpus"
# one row per distinct PREDICTION: arms with identical Elo and identical family mean utility are the same prediction
# under several names (Track S review F2) and are merged; the row lists how many names it has
groups = {}
for a in uniq:
    groups.setdefault((round(a["elo"], 6), round(a["family_mean_u"] or 0.0, 6)), []).append(a)
merged = sorted(groups.values(), key=lambda g: -g[0]["elo"])
# the table of single scorings also takes the H462 placements (with their lock status) and the two look-2 entrants
ext_groups = dict(groups)
for a in sorted(h462, key=lambda a: (not a["locked"], a["pass_index"])):
    ext_groups.setdefault((round(a["elo"], 6), round(a["family_mean_u"] or 0.0, 6)), []).append(a)
for key_, e_, gap_, rank_, arm_ in (("look2_arm", L2E["elo_arm"], L2E["gap_vs_kumo_l_arm"], L2E["rank_arm"], "stack_plus_19_family_specialists"),
                                    ("look2_nr", L2E["elo_arm_nr"], L2E["gap_vs_kumo_l_arm_nr"], L2E["rank_arm_nr"], "stack_without_specialists")):
    ext_groups[(round(e_["elo"], 6), key_)] = [{"arm": arm_, "entrant": "h458.look2." + key_, "file": "look2", "wave": "H458", "elo": e_["elo"], "ci95": e_["ci95"], "rank33": rank_,
                                                "gap_elo": {"vs_kumo_tabular_large_e8.v1": gap_}, "wtl": None, "family_mean_u": None, "look2": key_}]
merged_ext = sorted(ext_groups.values(), key=lambda g: -g[0]["elo"])
def status_of(a) -> str:
    if a.get("look2") == "look2_arm":
        return "look 2, locked; in-corpus recall"
    if a.get("look2"):
        return "look 2, locked control"
    if a["wave"] == "H462":
        return f"pass {a['pass_index']}, " + ("locked" if a["locked"] else "unlocked read")
    return ckpt_label(a["file"])
top50 = []
for i, g in enumerate(merged_ext[:50], 1):
    a = g[0]
    names = sorted({x["arm"] for x in g})
    top50.append({
        "n": i, "arm": a["arm"], "n_names": len(names), "names": names, "entrant": a["entrant"], "checkpoint": status_of(a), "wave": a["wave"], "elo": a["elo"],
        "lo": a["ci95"][0], "hi": a["ci95"][1], "rank34": a["rank33"], "gap_kumo_l": (a["gap_elo"] or {}).get("vs_kumo_tabular_large_e8.v1"),
        "gap_tabpfn35": (a["gap_elo"] or {}).get("vs_tabpfn_v3p5.v1"), "family_mean_u": a["family_mean_u"],
        "closure": ("fair" if a.get("label") == "0% dev91" else "in-corpus") if a["wave"] == "H462" or a.get("look2") else closure_of(a),
        "wtl_kumo_l": (a["wtl"] or {}).get("kumo_tabular_large_e8.v1"), "single_epoch": "diag_results" in a["file"],
    })
n_distinct_pred = len(merged)
n_distinct_elo = len({round(a["elo"], 6) for a in uniq})
all_unique = [{"arm": a["arm"], "elo": a["elo"], "wave": a["wave"], "closure": closure_of(a)} for a in uniq]

def group_of(n: str) -> str:
    if n.startswith(("pure_ugp",)):
        return "GP alone"
    if n.startswith(("pure_104m", "in_corpus_icl_boost", "unfair_all_d91_icl", "fair_all_non_d91_icl", "h452_neural_e8_hybrid_only")):
        return "ICL alone"
    if n.startswith(("neural", "hyb_", "in_corpus_neural_plus_tab")):
        return "Neural columns (incl. GP-residual hybrid)"
    if "champion" in n and n.startswith("in_corpus_random"):
        return "r4 champion stack"
    if n.startswith("fair_zero_d91"):
        return "r4 champion stack"
    return "Full stack (GP/tree pool + ICL)"
for u in all_unique:
    u["group"] = group_of(u["arm"])

# ----------------------------------------------------------------------------- Ultra-GP generations
ugp = []
for fam, pat, label in (("stack", r"^stack_ugp_g(\d+)_(cv7|2xcv7|cons)$", "Stack + Ultra-GP (in-corpus)"),
                        ("pure", r"^pure_ugp_g(\d+)_(cv7|2xcv7|cons)$", "Ultra-GP alone"),
                        ("fair", r"^fair_oos_0pct_d91_ugp_g(\d+)_(raw_cv7|2xcv7|cons|cv7)$", "Fair stack + Ultra-GP")):
    best = {}
    for a in arms:
        if a["wave"] not in ("H453", "H452"):
            continue
        m = re.match(pat, a["arm"])
        if not m:
            continue
        g, var = int(m.group(1)), m.group(2)
        k = (g, var)
        if k not in best or a["mtime"] > best[k]["mtime"]:
            best[k] = a
    for (g, var), a in sorted(best.items()):
        ugp.append({"series": label, "family": fam, "gen": g, "variant": var, "elo": a["elo"], "lo": a["ci95"][0], "hi": a["ci95"][1]})
base_stack = max((a for a in arms if a["arm"] == "stack_no_ugp_base"), key=lambda a: a["mtime"])
base_fair = max((a for a in arms if a["arm"] == "fair_oos_0pct_d91_no_ugp"), key=lambda a: a["mtime"])
ugp_base = {"stack_no_ugp": base_stack["elo"], "fair_no_ugp": base_fair["elo"]}

# ----------------------------------------------------------------------------- r4 trajectory
traj = []
for r in d["r4_traj"]:
    m = re.search(r"r4_g(\d+)", r["tag"])
    traj.append({"epoch": int(m.group(1)) if m else 0, **{k: r[k] for k in ("champion", "icl_only", "legacy", "no_gp", "control_delta")}})
traj.sort(key=lambda r: r["epoch"])

# ----------------------------------------------------------------------------- H455 checkpoints (stack, pure neural, ICL alone)
ck = collections.defaultdict(dict)
for a in arms:
    if a["wave"] != "H455":
        continue
    lab = ckpt_label(a["file"])
    for arm, col in (("stack_no_ugp_post_h454", "stack"), ("stack_ugp_g1000_cons_post_h454", "stack_ugp"), ("neural_104m_all_post_h454", "neural"),
                     ("hyb_104m_only_cv7_post_h454", "hyb"), ("pure_104m_cv7_post_h454", "icl"), ("pure_104m_only_cv7_post_h454", "icl_single")):
        if a["arm"] == arm:
            ck[lab][col] = a["elo"]
            ck[lab]["mtime"] = a["mtime"]
checkpoints = sorted(({"label": k, **v} for k, v in ck.items()), key=lambda r: -r.get("stack", 0))

# ----------------------------------------------------------------------------- fair against in-corpus pairs
def one(arm, wave=None):
    c = [a for a in arms if a["arm"] == arm and (wave is None or a["wave"] == wave)]
    return max(c, key=lambda a: a["mtime"]) if c else None
pairs = []
for label, fair_arm, in_arm, note in (
    ("H449 stack", "fair_all_non_d91_stack", "unfair_all_d91_stack", "ICL trained on all non-dev91 tables against all tables"),
    ("H449 ICL only", "fair_all_non_d91_icl_only", "unfair_all_d91_icl_only", "same pair, ICL column alone"),
    ("H453 stack", "fair_oos_0pct_d91_no_ugp", "stack_no_ugp_base", "0% dev91 in ICL training against the in-corpus stack"),
    ("H452 legacy control", "ctrl_h449_clean_g00_legacy", "ctrl_h450_g16_legacy", "clean epoch-0 ICL columns against r4 epoch-16 columns"),
):
    f, i = one(fair_arm), one(in_arm)
    pairs.append({"label": label, "fair": f["elo"], "in_corpus": i["elo"], "delta": i["elo"] - f["elo"], "note": note})
r4c = [a for a in arms if a["arm"] == "in_corpus_random_split_champion_91" and a["wave"] == "H450" and "r4_g00_init" not in a["file"]]
r4f = one("fair_zero_d91_upgraded_control_91")
pairs.append({"label": f"H450 r4 champion (mean of {len(r4c)} epochs)", "fair": r4f["elo"], "in_corpus": statistics.mean(a["elo"] for a in r4c),
              "delta": statistics.mean(a["elo"] for a in r4c) - r4f["elo"], "note": "random-split champion against its zero-dev91 control"})
for p_, k_ in zip(pairs, ("share only", "share only", "share + 16 epochs", "share + 16 epochs", "share + 2 to 48 epochs")):
    p_["contrast"] = k_

# ----------------------------------------------------------------------------- families and contexts (stored per-context utilities of the 33 field entrants)
ctxs = board["contexts"]
fam_of = {c["cid"]: c["family_id"] for c in ctxs}
fams = sorted({c["family_id"] for c in ctxs})
fam_ctx = {f: [c["cid"] for c in ctxs if c["family_id"] == f] for f in fams}
fam_meta = {f: {"n_ctx": len(fam_ctx[f]), "n_rows": int(statistics.median(c["n_rows"] for c in ctxs if c["family_id"] == f)),
                "width": int(statistics.median(c["width"] for c in ctxs if c["family_id"] == f)), "short": f[4:12]} for f in fams}
fam_u = {k: {f: statistics.mean(v["u"][c] for c in fam_ctx[f]) for f in fams} for k, v in field.items()}
fam_rank = {k: {} for k in field}
fam_best = {}
for f in fams:
    order = sorted(field, key=lambda k: -fam_u[k][f])
    for i, k in enumerate(order, 1):
        fam_rank[k][f] = i
    ext = [k for k in order if CLASS[k] != "Ours, earlier wave"]
    fam_best[f] = {"best": order[0], "best_u": fam_u[order[0]][f], "best_external": ext[0], "best_external_u": fam_u[ext[0]][f]}
wins = collections.Counter(v["best"] for v in fam_best.values())

def versus(a: str, b: str) -> dict:
    cw = ct = cl = 0
    fw = ft = fl = 0
    gains = []
    for f in fams:
        g = []
        for c in fam_ctx[f]:
            dlt = field[a]["u"][c] - field[b]["u"][c]
            g.append(dlt)
            if dlt > TIE:
                cw += 1
            elif dlt < -TIE:
                cl += 1
            else:
                ct += 1
        m = statistics.mean(g)
        gains.append(m)
        if m > TIE:
            fw += 1
        elif m < -TIE:
            fl += 1
        else:
            ft += 1
    return {"ctx_wtl": [cw, ct, cl], "fam_wtl": [fw, ft, fl], "family_mean_gain": statistics.mean(gains)}
OUR_FAIR = "h448.APEX_FAIR_CHAMPION"
vs_fair = [{"id": r["id"], "name": r["name"], "cls": r["cls"], "elo": r["elo"], **versus(OUR_FAIR, r["id"])} for r in field_rows if r["id"] != OUR_FAIR]
vs_kumo = [{"id": r["id"], "name": r["name"], "cls": r["cls"], "elo": r["elo"], **versus("kumo_tabular_large_e8.v1", r["id"])} for r in field_rows if r["id"] != "kumo_tabular_large_e8.v1"]
ctx_delta = sorted(({"cid": c["cid"][:10], "family": fam_meta[c["family_id"]]["short"],
                     "d_kumo_l": field[OUR_FAIR]["u"][c["cid"]] - field["kumo_tabular_large_e8.v1"]["u"][c["cid"]],
                     "d_tabpfn35": field[OUR_FAIR]["u"][c["cid"]] - field["tabpfn_v3p5.v1"]["u"][c["cid"]],
                     "u_ours": field[OUR_FAIR]["u"][c["cid"]], "u_kumo_l": field["kumo_tabular_large_e8.v1"]["u"][c["cid"]]} for c in ctxs),
                   key=lambda r: r["d_kumo_l"])
r4fam = d["r4_g47"]["per_family_mean_u_91"]
family_table = []
for f in fams:
    row = {"family": fam_meta[f]["short"], **fam_meta[f], "best": NAMES.get(fam_best[f]["best"], fam_best[f]["best"]), "best_id": fam_best[f]["best"], "best_u": fam_best[f]["best_u"],
           "kumo_l_u": fam_u["kumo_tabular_large_e8.v1"][f], "tabfm_ens_u": fam_u["tabfm_ensemble.v1"][f], "tabpfn35_u": fam_u["tabpfn_v3p5.v1"][f],
           "fair_u": fam_u[OUR_FAIR][f], "fair_rank": fam_rank[OUR_FAIR][f], "kumo_l_rank": fam_rank["kumo_tabular_large_e8.v1"][f],
           "r4_champion_u": r4fam[f]["h450_champion_in_corpus"], "r4_fair_ctrl_u": r4fam[f]["h450_zero_d91_ctrl"], "r4_icl_u": r4fam[f]["h450_icl_boost_only_in_corpus"]}
    row["fair_gap_to_best"] = row["fair_u"] - fam_best[f]["best_external_u"]
    row["r4_gap_to_best"] = row["r4_champion_u"] - fam_best[f]["best_external_u"]
    family_table.append(row)
HEAT = ["kumo_tabular_large_e8.v1", "tabfm_ensemble.v1", "kumo_tabular_small_e8.v1", "tabfm.v1", "tabpfn_v3p5.v1", "tabpfn_v3p5_e1.v1", OUR_FAIR, "tabpfn_v3.v1",
        "h416.SXB", "mitra_v2_zeroshot.v1", "tabicl_v2.v1", "tabdpt_1p3.v1", "autogluon_300s.v1", "catboost.v1", "xgboost.v1", "lightgbm.v1"]
heat = {"entrants": [{"id": k, "name": NAMES[k]} for k in HEAT], "families": [fam_meta[f]["short"] for f in fams],
        "rank": [[fam_rank[k][f] for f in fams] for k in HEAT], "u": [[fam_u[k][f] for f in fams] for k in HEAT]}

# ----------------------------------------------------------------------------- our three headline systems, context by context
# Per-context utilities of the best arm (released by the H455 owner on 2026-11-15, bit-for-bit equal to the published matrix) and of
# the best fair-closure arm (Track S inputs). The fair champion's are on the board.
def load_ctx(path: str, entry: str) -> dict:
    u = {}
    for line in open(HERE + path):
        r = json.loads(line)
        if r["entry"] == entry:
            u[r["cid"]] = r["utility"]
    assert set(u) == set(fam_of), (entry, len(u))
    return u


def versus_u(ua: dict, ub: dict) -> dict:
    """ctx: contexts won, tied, lost. fam: families where the family mean utility is higher (the scorers' stored W-T-L).
    game: the ladder's own unit, the sum over families of the share of contexts won (ties count half)."""
    cw = ct = cl = fw = ft = fl = 0
    game = 0.0
    for f in fams:
        g, sc = [], []
        for c in fam_ctx[f]:
            dlt = ua[c] - ub[c]
            g.append(dlt)
            if dlt > TIE:
                cw += 1
                sc.append(1.0)
            elif dlt < -TIE:
                cl += 1
                sc.append(0.0)
            else:
                ct += 1
                sc.append(0.5)
        m = statistics.mean(g)
        if m > TIE:
            fw += 1
        elif m < -TIE:
            fl += 1
        else:
            ft += 1
        game += statistics.mean(sc)
    return {"ctx_wtl": [cw, ct, cl], "fam_wtl": [fw, ft, fl], "game": game}


sys_by_key = {s_["key"]: s_ for s_ in systems}
HEAD = [
    {"key": "best", "name": "Best arm at its best checkpoint", "tag": "in-corpus", "elo": sys_by_key["s_stack_icl"]["elo_best"],
     "u": load_ctx("H455_TOP_ARMS_PER_CONTEXT_U.jsonl", "H455.M_X2_valley001_002_avg.stack_no_ugp_post_h454")},
    {"key": "fairc", "name": "Best fair-closure arm", "tag": "fair closure, post hoc", "elo": sys_by_key["f_fair_oos"]["elo"],
     "u": load_ctx("B_h451_h457.jsonl", "H453.gen1000.fair_oos_0pct_d91_no_ugp")},
    {"key": "fair", "name": "Fair champion H448", "tag": "fair, pre-registered", "elo": field[OUR_FAIR]["elo"], "u": field[OUR_FAIR]["u"]},
]
# H458 look 2 (2026-11-15): the legacy stack plus one recall specialist per family. In-corpus RECALL: the specialists were trained on
# these audit rows with their labels. It is shown beside the ladder, never as an entrant of the reference board.
if os.path.isfile(HERE + "look2_per_ctx_u.jsonl"):
    L2 = json.load(open(HERE + "look2_SCORE.json"))
    u2 = {}
    for line in open(HERE + "look2_per_ctx_u.jsonl"):
        r = json.loads(line)
        u2[r["cid"]] = r["utility"]
    assert set(u2) == set(fam_of), len(u2)
    HEAD.append({"key": "recall", "name": "Legacy stack + family recall specialists (H458 look 2)",
                 "tag": "in-corpus recall: these audit rows were trained on, with labels", "elo": L2["elo_arm"]["elo"], "u": u2})
    HEAD.append({"key": "norecall", "name": "The same stack without the recall columns (H458 look 2)", "tag": "in-corpus",
                 "elo": L2["elo_arm_nr"]["elo"], "u": {c: L2["per_ctx"][c]["u_arm_nr"] for c in u2}})
# the released per-context utilities must reproduce the W-T-L the scorers stored with each arm
for h_, k_ in ((HEAD[0], "s_stack_icl"), (HEAD[1], "f_fair_oos")):
    for r_ in REFS:
        assert versus_u(h_["u"], field[r_]["u"])["fam_wtl"] == list(sys_by_key[k_]["wtl"][r_]), (k_, r_)
KL = "kumo_tabular_large_e8.v1"
externals = [r for r in field_rows if r["cls"] != "Ours, earlier wave"]
h2h = []
for r in externals:
    row = {"id": r["id"], "name": r["name"], "cls": r["cls"], "elo": r["elo"]}
    for h_ in HEAD:
        row[h_["key"]] = versus_u(h_["u"], field[r["id"]]["u"])
    h2h.append(row)
head = []
for h_ in HEAD:
    others = [k for k in field if not (h_["key"] == "fair" and k == OUR_FAIR)]
    ext_ids = [r["id"] for r in externals]
    fu = {f: statistics.mean(h_["u"][c] for c in fam_ctx[f]) for f in fams}
    frank = {f: 1 + sum(1 for k in others if fam_u[k][f] > fu[f]) for f in fams}
    above_field = sum(1 for c in fam_of if h_["u"][c] > max(field[k]["u"][c] for k in others) + TIE)
    above_ext = sum(1 for c in fam_of if h_["u"][c] > max(field[k]["u"][c] for k in ext_ids) + TIE)
    dk = {f: [h_["u"][c] - field[KL]["u"][c] for c in fam_ctx[f]] for f in fams}
    deficit = {f: -sum(min(0.0, v) for v in dk[f]) for f in fams}
    tot = sum(deficit.values())
    top5 = sorted(fams, key=lambda f: -deficit[f])[:5]
    close = [f for f in fams if all(abs(v) <= 0.15 for v in dk[f])]
    head.append({
        "key": h_["key"], "name": h_["name"], "tag": h_["tag"], "elo": h_["elo"], "family_mean_u": statistics.mean(fu.values()),
        "fam_u": {fam_meta[f]["short"]: fu[f] for f in fams}, "fam_rank": {fam_meta[f]["short"]: frank[f] for f in fams},
        "n_fam_first": sum(1 for f in fams if frank[f] == 1), "n_fam_top3": sum(1 for f in fams if frank[f] <= 3),
        "ctx_above_field": above_field, "ctx_above_externals": above_ext,
        "kumo_win_cids": sorted(c[:10] for c in fam_of if h_["u"][c] - field[KL]["u"][c] > TIE),
        "kumo": {"ctx_wins": sum(1 for f in fams for v in dk[f] if v > TIE), "fam_swept_by_kumo": sum(1 for f in fams if all(v < -TIE for v in dk[f])),
                 "fam_won_all": [fam_meta[f]["short"] for f in fams if all(v > TIE for v in dk[f])],
                 "top5": [fam_meta[f]["short"] for f in top5], "top5_share": sum(deficit[f] for f in top5) / tot,
                 "n_close": len(close), "n_close_swept": sum(1 for f in close if all(v < -TIE for v in dk[f])),
                 "median_abs_gap": statistics.median(abs(v) for f in fams for v in dk[f])},
    })
head_by = {h_["key"]: h_ for h_ in head}
ctx_delta = sorted(({"cid": c["cid"][:10], "family": fam_meta[c["family_id"]]["short"],
                     "d_kumo_l": field[OUR_FAIR]["u"][c["cid"]] - field[KL]["u"][c["cid"]],
                     "d_kumo_l_best": HEAD[0]["u"][c["cid"]] - field[KL]["u"][c["cid"]],
                     "d_kumo_l_recall": (HEAD[3]["u"][c["cid"]] - field[KL]["u"][c["cid"]]) if len(HEAD) > 3 and HEAD[3]["key"] == "recall" else None,
                     "d_kumo_l_norecall": (HEAD[4]["u"][c["cid"]] - field[KL]["u"][c["cid"]]) if len(HEAD) > 4 and HEAD[4]["key"] == "norecall" else None,
                     "d_kumo_l_fairc": HEAD[1]["u"][c["cid"]] - field[KL]["u"][c["cid"]],
                     "d_tabpfn35": field[OUR_FAIR]["u"][c["cid"]] - field["tabpfn_v3p5.v1"]["u"][c["cid"]],
                     "d_tabpfn35_best": HEAD[0]["u"][c["cid"]] - field["tabpfn_v3p5.v1"]["u"][c["cid"]],
                     "u_ours": field[OUR_FAIR]["u"][c["cid"]], "u_best": HEAD[0]["u"][c["cid"]], "u_kumo_l": field[KL]["u"][c["cid"]]} for c in ctxs),
                   key=lambda r: r["d_kumo_l"])
for row in family_table:
    row["best_arm_u"] = head_by["best"]["fam_u"][row["family"]]
    row["best_arm_rank"] = head_by["best"]["fam_rank"][row["family"]]
    row["fairc_u"] = head_by["fairc"]["fam_u"][row["family"]]
    row["fairc_rank"] = head_by["fairc"]["fam_rank"][row["family"]]
    f_full = next(f for f in fams if fam_meta[f]["short"] == row["family"])
    row["best_arm_gap_to_best"] = row["best_arm_u"] - fam_best[f_full]["best_external_u"]
    row["best_arm_ctx_vs_kumo"] = sum(1 for c in fam_ctx[f_full] if HEAD[0]["u"][c] - field[KL]["u"][c] > TIE)
    row["fair_ctx_vs_kumo"] = sum(1 for c in fam_ctx[f_full] if field[OUR_FAIR]["u"][c] - field[KL]["u"][c] > TIE)
# heat map: our two newer headline arms are added at the rank they would take if inserted into the field
ins = [{"id": "ours.best", "name": "Our best arm, best checkpoint (in-corpus)", "rank": [head_by["best"]["fam_rank"][fam_meta[f]["short"]] for f in fams]},
       {"id": "ours.fairc", "name": "Our best fair-closure arm (post hoc)", "rank": [head_by["fairc"]["fam_rank"][fam_meta[f]["short"]] for f in fams]}]
heat["entrants"] = heat["entrants"][:4] + [{"id": i_["id"], "name": i_["name"]} for i_ in ins] + heat["entrants"][4:]
heat["rank"] = heat["rank"][:4] + [i_["rank"] for i_ in ins] + heat["rank"][4:]
heat["u"] = heat["u"][:4] + [[head_by[k_]["fam_u"][fam_meta[f]["short"]] for f in fams] for k_ in ("best", "fairc")] + heat["u"][4:]

# ----------------------------------------------------------------------------- external models only
# The frozen field holds 11 of our own earlier systems. A league of the 22 external entrants plus ONE system of ours is a balanced
# round robin (every pair plays 19 family games), so its Bradley-Terry order is the order of total games won and needs no refit.
# (Point raised and first computed by Track S, 2026-11-15 00:24Z; recomputed here.)
ext_ids = [r["id"] for r in externals]
league = {}
for h_ in HEAD:
    us = {k: field[k]["u"] for k in ext_ids}
    us["OURS"] = h_["u"]
    tot = {a_: sum(versus_u(us[a_], us[b_])["game"] for b_ in us if b_ != a_) for a_ in us}
    order = sorted(tot, key=lambda k: -tot[k])
    league[h_["key"]] = {"rank": order.index("OURS") + 1, "n": len(order), "games": tot["OURS"], "of": 19 * (len(order) - 1),
                         "table": [{"id": k, "name": h_["name"] if k == "OURS" else NAMES[k], "games": tot[k], "ours": k == "OURS"} for k in order]}
TP = "tabpfn_v3p5.v1"
groups3 = (("Our own 11 earlier systems on the board", [k for k in field if CLASS[k] == "Ours, earlier wave"]),
           ("The 12 classical and LLM entrants", [k for k in ext_ids if CLASS[k] != "Tabular foundation model"]),
           ("The 9 other tabular foundation models", [k for k in ext_ids if CLASS[k] == "Tabular foundation model" and k != TP]))
decomp = []
for lab_, ks in groups3:
    wb = sum(versus_u(HEAD[0]["u"], field[k]["u"])["game"] for k in ks)
    wt = sum(versus_u(field[TP]["u"], field[k]["u"])["game"] for k in ks)
    decomp.append({"opponents": lab_, "n": len(ks), "of": 19 * len(ks), "ours": wb, "tabpfn35": wt, "diff": wb - wt})
hh = versus_u(HEAD[0]["u"], field[TP]["u"])["game"]
decomp.append({"opponents": "Each other, head to head", "n": 1, "of": 19, "ours": hh, "tabpfn35": 19 - hh, "diff": 2 * hh - 19})
decomp.append({"opponents": "All 33 opponents", "n": 33, "of": 19 * 33, "ours": sum(r["ours"] for r in decomp), "tabpfn35": sum(r["tabpfn35"] for r in decomp),
               "diff": sum(r["diff"] for r in decomp)})
assert sum(len(ks) for _, ks in groups3) == 32

# ----------------------------------------------------------------------------- H458 look 2 (in-corpus recall) and the fresh-row probe
look2 = None
if os.path.isfile(HERE + "look2_SCORE.json"):
    pc2 = L2["per_ctx"]
    hrec = next(h_ for h_ in head if h_["key"] == "recall")
    NOT_TARGET = ("280b1a13", "4286560c", "08c91457")
    fam_rows = []
    for f in fams:
        cs = fam_ctx[f]
        sh = fam_meta[f]["short"]
        cmax = {c: max(field[k]["u"][c] for k in field) for c in cs}
        above = sum(1 for c in cs if u2[c] > cmax[c] + TIE)
        holder = collections.Counter(max(field, key=lambda k: field[k]["u"][c]) for c in cs if u2[c] <= cmax[c] + TIE)
        fam_rows.append({
            "family": sh, "n": len(cs), "u_arm": statistics.mean(u2[c] for c in cs), "u_base": statistics.mean(pc2[c]["u_base"] for c in cs),
            "field_max": statistics.mean(cmax.values()), "best": NAMES.get(fam_best[f]["best"], fam_best[f]["best"]), "best_u": fam_best[f]["best_u"],
            "above_field": above, "swept": above == len(cs), "target": sh not in NOT_TARGET, "rank": hrec["fam_rank"][sh],
            "w_recall": statistics.mean(pc2[c]["w_recall_arm"] for c in cs), "kumo_w": sum(1 for c in cs if u2[c] > field[KL]["u"][c] + TIE),
            "held_by": [{"name": NAMES.get(k, k), "n": n} for k, n in holder.most_common()],
        })
    pr = {k: json.load(open(HERE + f"probe_in/{k}.json")) for k in ("FRESH_CANON", "FRESH_MODES", "FRESH_60B6")}
    pm = pr["FRESH_CANON"]["paired"]["canon_mean3 - init_mean3"]
    p6 = pr["FRESH_60B6"]["paired"]
    look2 = {
        "arm": L2["arm"], "label": "in-corpus recall: these audit rows were trained on, with labels", "receipt_sha256": hashlib.sha256(open(HERE + "look2_SCORE.json", "rb").read()).hexdigest(),
        "elo": L2["elo_arm"]["elo"], "ci": L2["elo_arm"]["ci95"], "rank": L2["rank_arm"], "gap_kumo": L2["gap_vs_kumo_l_arm"],
        "elo_nr": L2["elo_arm_nr"]["elo"], "ci_nr": L2["elo_arm_nr"]["ci95"], "rank_nr": L2["rank_arm_nr"], "gap_kumo_nr": L2["gap_vs_kumo_l_arm_nr"],
        "pair": {k: L2["pair_arm_vs_base"][k] for k in ("wtl", "gain", "signflip_p", "lofo_min", "family_median")},
        "families": fam_rows, "n_swept": sum(r["swept"] for r in fam_rows), "n_swept_target": sum(r["swept"] and r["target"] for r in fam_rows),
        "n_target": sum(r["target"] for r in fam_rows), "gate_needed": 13, "ctx_above_field": hrec["ctx_above_field"], "ctx_above_externals": hrec["ctx_above_externals"],
        "ctx_above_field_nr": next(h_ for h_ in head if h_["key"] == "norecall")["ctx_above_field"],
        "n_fam_first": hrec["n_fam_first"], "w_recall_mean": statistics.mean(v["w_recall_arm"] for v in pc2.values()),
        "ctx_below_base": sum(1 for v in pc2.values() if v["u_arm"] < v["u_base"] - TIE), "family_mean_u": hrec["family_mean_u"],
        "family_mean_u_base": statistics.mean(r["u_base"] for r in fam_rows),
        "probe": {
            "n_families": pm["n_families"], "n_contexts": pm["n_contexts"], "du": pm["mean_family_du"], "ctx_wtl": pm["contexts_wtl"], "fam_wtl": pm["families_wtl"],
            "p": pm["signflip_p_two_sided"], "u_canon": pr["FRESH_CANON"]["summary"]["canon_mean3"]["family_mean_u"], "u_init": pr["FRESH_CANON"]["summary"]["init_mean3"]["family_mean_u"],
            "family_du": sorted(({"family": k[4:12], "du": v} for k, v in pm["family_du"].items()), key=lambda r: -r["du"]),
            "modes": {m: {"canon": pr["FRESH_MODES"]["summary"][f"canon_{m}"]["family_mean_u"], "init": pr["FRESH_MODES"]["summary"][f"init_{m}"]["family_mean_u"]} for m in ("rr", "yj", "rg")},
            "b60": {"canon": p6["canon_mean3 - init_mean3"]["mean_family_du"], "rand": p6["rand_mean3 - init_mean3"]["mean_family_du"],
                    "init_u": pr["FRESH_60B6"]["summary"]["init_mean3"]["family_mean_u"], "canon_ctx": p6["canon_mean3 - init_mean3"]["contexts_wtl"],
                    "rand_ctx": p6["rand_mean3 - init_mean3"]["contexts_wtl"]},
        },
    }

# ----------------------------------------------------------------------------- revision 10: Stage 3, seed averaging, the ten-context reads, the price list
R10 = HERE + "rev10_in/"
def league_of(u: dict) -> dict:
    us = {k: field[k]["u"] for k in ext_ids}
    us["OURS"] = u
    tot = {a_: sum(versus_u(us[a_], us[b_])["game"] for b_ in us if b_ != a_) for a_ in us}
    order = sorted(tot, key=lambda k: -tot[k])
    return {"rank": order.index("OURS") + 1, "n": len(order), "games": tot["OURS"], "of": 19 * (len(order) - 1)}
s3 = json.load(open(HERE + "stage3_in/H456_STAGE3_VALLEY_PROGRESSION_SUMMARY.json"))
S3 = (("l_x1_h455_valley001_004_avg", "H456.L_X1_V2_valley001_004_avg", "L (104M), mean of valleys 1 to 4"),
      ("l_x1_h455_valley004_soup50", "H456.L_X1_V2_valley004_soup50", "L (104M), valley 4 weight average"),
      ("m_x2_h455_valley001_002_003_avg", "H456.M_X2_valley001_002_003_avg", "M (24M), mean of valleys 1 to 3"),
      ("m_x2_h455_valley003_soup50", "H456.M_X2_valley003_soup50", "M (24M), valley 3 weight average"),
      ("lh455_v1v4_plus_mh455_v1v2v3_50_50", "H456.LH455V14_MH455V123_50_50", "L and M columns blended 50/50"))
stage3 = []
for w_, e_, lab_ in S3:
    a_ = s3["waves"][w_]["arms"]
    u_ = load_ctx("stage3_in/H456_STAGE3_PER_CONTEXT_U.jsonl", e_ + ".stack_no_ugp_post_h454")
    stage3.append({"label": lab_, "stack": a_["stack_no_ugp_post"]["elo"], "stack_ci": a_["stack_no_ugp_post"]["ci95"], "stack_rank": a_["stack_no_ugp_post"]["rank_of_34"],
                   "stack_ugp": a_["stack_ugp_g1000_cons_post"]["elo"], "neural": a_["neural_104m_all_post"]["elo"], "icl": a_["pure_104m_cv7_post"]["elo"],
                   "league": league_of(u_), "kumo": versus_u(u_, field[KL]["u"])})
m6 = json.load(open(R10 + "M6_SEED_BAGGING_PRICE.json"))["dev91_quarantined_post_h448"]
M6 = (("h448_fair_champion", "Fair champion H448 (0% dev91 in training)", "fair; the average is priced post hoc"),
      ("cw12_no_icl", "GP and tabular columns, no neural column", "fair closure, post hoc"),
      ("gp_only", "GP columns alone", "fair closure, post hoc"),
      ("m_valley_zero_g16_stack_no_ugp", "M-valley stack without the r4 epoch-16 column", "in-corpus (1.8% dev91 in the ICL pre-training)"),
      ("stage3_lm_zero_g16_stack_no_ugp", "Stage 3 L+M stack without the r4 epoch-16 column", "in-corpus (1.8% dev91 in the ICL pre-training)"),
      ("m_valley_1865_stack_no_ugp_in_corpus", "M-valley stack (the 1865.36 scoring)", "in-corpus"),
      ("stage3_lm_1869_stack_no_ugp_in_corpus", "Stage 3 L+M stack", "in-corpus"))
seedavg = [{"label": lab_, "tag": tag_, "before": m6[k_]["single_placement_mean_u_conv"]["elo"], "after": m6[k_]["single_placement_pred_avg_2"]["elo"],
            "gain": m6[k_]["single_placement_elo_gain"], "rank_after": m6[k_]["single_placement_pred_avg_2"]["rank"], "du": m6[k_]["family_mean_u_gain"],
            "fam_wtl": m6[k_]["pair_stats_vs_mean_conv"]["wtl"], "ctx_wtl": m6[k_]["ctx_wtl_tie1e4"], "p": m6[k_]["pair_stats_vs_mean_conv"]["signflip_p"]} for k_, lab_, tag_ in M6]
def pr10(name: str) -> dict:
    return json.load(open(R10 + name + ".json"))
def pair10(j: dict, key: str, flip: bool = False) -> dict:
    p_ = j["paired"][key]
    w_, t_, l_ = p_["contexts_wtl"]
    return {"w": l_ if flip else w_, "l": w_ if flip else l_, "du": -p_["mean_family_du"] if flip else p_["mean_family_du"], "p": p_["signflip_p_two_sided"], "scored": j["time_utc"]}
pH, pD, pM, pG, pR = (pr10(n_) for n_ in ("FRESH_HINTS_FROZEN_INIT", "FRESH_CTXSEARCH_DOSE_E500_E1000_E2000", "FRESH_CTXSEARCH_RAND_OWN_VS_LOFO_MATCHED_STEPS",
                                         "FRESH_CTXSEARCH_RAND_OWN_VS_LOFO_GENERALIST", "FRESH_CTXSEARCH_RAND_E2000_VS_INIT"))
hints_above = sum(1 for r_ in pH["per_context"].values()
                  if r_["cols"]["inith_mean3"]["u"] > max(r_["cols"][k_]["u"] for k_ in ("hridge_mean3", "hknn_mean3", "hgbm_mean3")))
probe10 = {
    "n": pR["n_scored"], "init_u": pR["summary"]["init_mean3"]["family_mean_u"], "own_u": pR["summary"]["ctxr_mean3"]["family_mean_u"],
    "tree_u": pH["summary"]["hgbm_mean3"]["family_mean_u"], "hint_u": pH["summary"]["inith_mean3"]["family_mean_u"], "hints_above_best": hints_above,
    "vs_init": [
        {"arm": "Fine-tune on the table's own rows, random feature slots", **pair10(pR, "ctxr_mean3 - init_mean3"), "rule": "7 or more of 10", "ok": True},
        {"arm": "The same with canonical slots, 500 exposures per row", **pair10(pD, "c500_mean3 - init_mean3"), "rule": "descriptive", "ok": None},
        {"arm": "A small boosted tree alone (no neural model)", **pair10(pH, "hgbm_mean3 - init_mean3"), "rule": "descriptive", "ok": None},
        {"arm": "Canonical slots, 1,000 exposures", **pair10(pD, "c1000_mean3 - init_mean3"), "rule": "descriptive", "ok": None},
        {"arm": "Frozen model given three hint columns", **pair10(pH, "inith_mean3 - init_mean3"), "rule": "7 or more of 10", "ok": True},
        {"arm": "Canonical slots, 2,000 exposures", **pair10(pD, "ctx_mean3 - init_mean3"), "rule": "6 or more of 10", "ok": False},
        {"arm": "Generalist trained on the other nine tables, matched steps", **pair10(pM, "init_mean3 - lofm_mean3", True), "rule": "7 or more of 10", "ok": False},
        {"arm": "Generalist trained on the other nine tables", **pair10(pG, "init_mean3 - lofo_mean3", True), "rule": "7 or more of 10", "ok": False},
    ],
    "own_vs_lofo": pair10(pG, "ctxr_mean3 - lofo_mean3"), "own_vs_lofm": pair10(pM, "ctxr_mean3 - lofm_mean3"), "foreign_vs_lofo": pair10(pG, "forr_mean3 - lofo_mean3"),
}
wl1 = {r_["scenario"].strip(): r_ for r_ in json.load(open(R10 + "WHATIF_LADDER_v1.json"))["rows"]}
wl2 = {r_["scenario"].strip(): r_ for r_ in json.load(open(R10 + "WHATIF_LADDER_v2.json"))["rows"]}
PRICE = (("The fair champion as it is, re-inserted", wl1, "fair champion as it is (re-inserted)"),
         ("Never below the best of five classic models (ridge, CatBoost, LightGBM, XGBoost, random forest) on a context", wl2, "FLOOR oracle: never below any of five classic models"),
         ("Best in the field on the one single-context family only", wl2, "fair champion, best-in-field on the singleton family 866f66b1 only"),
         ("Just past Kumo Large on all 28 precision contexts", wl2, "EXACT: fair champion just past Kumo L on all 28 precision contexts"),
         ("+0.05 utility on the 63 ordinary contexts", wl2, "EDGE: fair champion +0.05 u on the 63 ordinary contexts"),
         ("+0.10 utility on the 63 ordinary contexts", wl2, "EDGE: fair champion +0.10 u on the 63 ordinary contexts"),
         ("+0.05 utility on all 91 contexts", wl1, "fair champion +0.05 u on all 91 contexts"),
         ("+0.10 utility on all 91 contexts", wl1, "fair champion +0.10 u on all 91 contexts"),
         ("+0.15 utility on all 91 contexts", wl1, "fair champion +0.15 u on all 91 contexts"),
         ("TabPFN 3.5's pattern of wins at our mean utility", wl2, "TabPFN 3.5's row shifted to the fair champion's family-mean utility"),
         ("Best of the fair champion and six open tabular foundation models, per context", wl1, "best of {fair champion, 6 open tabular FMs} per context"),
         ("The same minus 0.02 utility", wl1, "the same minus 0.02 u (a stack rarely equals its best member)"))
price = [{"label": lab_, "elo": src_[k_]["elo"], "rank": src_[k_]["rank_of_34"], "games": src_[k_].get("games_vs_22_externals_of_418")} for lab_, src_, k_ in PRICE]
fv = json.load(open(R10 + "FLOOR_VIOLATIONS_v1.json"))["rows"]
rev10 = {"stage3": stage3, "stage3_best": {"elo": s3["best_overall_elo"], "ci": [s3["best_overall_lcb95"], s3["best_overall_ucb95"]]}, "seedavg": seedavg,
         "probe": probe10, "price": price, "floor_below": sum(1 for r_ in fv if r_["gap"] > TIE), "floor_n": len(fv),
         "inputs_sha256": {n_: hashlib.sha256(open(R10 + n_, "rb").read()).hexdigest() for n_ in sorted(os.listdir(R10))}}

# ----------------------------------------------------------------------------- revision 11: S9 closed, the fair series, line 45, Track S's reads, the C2 replay
R11 = HERE + "rev11_in/"
def s9_log(name: str) -> dict:
    rules, elo, span = {}, {}, {}
    for ln in open(R11 + name + ".log"):
        m_ = re.match(r"^([a-z0-9_]+) (\{.*\})\s*$", ln)
        if m_ and '"R1"' in ln:
            rules[m_.group(1)] = json.loads(m_.group(2))
        m_ = re.match(r"^Elo (\S+) ([0-9.]+) rank", ln)
        if m_:
            elo[m_.group(1)] = float(m_.group(2))
        m_ = re.match(r"^(start|exit \d+ end) (\S+)", ln)
        if m_:
            span[m_.group(1).split()[0]] = m_.group(2)
    return {"rules": rules, "elo": elo, "span": span}
S9B = (("p201b_medium_a5_prod_pure", "A5 boosted residual GP, 40 generations, pure anchor", "agy", "cl_mtc", "a0_medium_pure"),
       ("p201b_medium_a5_prod_hyb", "A5, 40 generations, hybrid anchor", "agy", "cl_mtc", "a0_medium_hyb"),
       ("p201b_high_a5_prod_pure", "A5, 160 generations, pure anchor", "agy", "cl_h458", "a0_high_pure"),
       ("p201b_high_a5_prod_hyb", "A5, 160 generations, hybrid anchor", "agy", "cl_h458", "a0_high_hyb"),
       ("p201b_medium_main", "A1 to A4 chains and A6a committee, 40 generations", "cl_h458", "agy", ""),
       ("p201b_high_main", "A1 to A4 chains and A6a committee, 160 generations", "cl_h458", "cl_mtc", ""))
s9 = []
for f_, lab_, builder_, scorer_, cpre_ in S9B:
    g_ = s9_log(f_)
    elig = {k: v for k, v in g_["rules"].items() if v.get("selection_eligible")}
    ca, cb, cx = (cpre_ + "_s0", cpre_ + "_s1", cpre_ + "_2x") if cpre_ else ("a0a", "a0b", "a0x")
    bag = next((v["T_over_ctrl_a"] for k, v in g_["rules"].items() if k == "a6b" or k.endswith("_runbag")), None)
    s9.append({"label": lab_, "builder": builder_, "scorer": scorer_, "n_eligible": len(elig), "n_r1": sum(1 for v in elig.values() if v["R1"]),
               "t_min": min(v["T_over_ctrl_a"] for v in elig.values()), "t_max": max(v["T_over_ctrl_a"] for v in elig.values()), "runbag_t": bag,
               "elo_v0": g_["elo"]["V0"], "elo_a": g_["elo"][ca], "elo_b": g_["elo"][cb], "elo_x": g_["elo"][cx],
               "elo_best_eligible": max(g_["elo"][k] for k in elig), "elo_worst_eligible": min(g_["elo"][k] for k in elig), "scored": g_["span"].get("exit", "")})
cyc = s9_log("p201b_medium_a5_cycle_pure")
tp = json.load(open(R11 + "M6_TABCV7_CVREP_PRICE.json"))
sp = tp["single_placements"]
series = [{"label": "Fair champion H448, one run", "elo": sp["n0_mean_conv"]["elo"], "rank": sp["n0_mean_conv"]["rank"], "note": "the only entrant registered before any dev91 read of its own"},
          {"label": "Two GP seed runs averaged", "elo": sp["pred_avg_2_n0"]["elo"], "rank": sp["pred_avg_2_n0"]["rank"], "note": "passes 1 and 2, no selection-board lock"},
          {"label": "Two seeds x two neural seeds", "elo": sp["gm2l_rep0_eq4_s12"]["elo"], "rank": sp["gm2l_rep0_eq4_s12"]["rank"], "note": "locked on the selection board"},
          {"label": "Plus four repeats of cross-validation (16 predictions)", "elo": sp["gm2l_cvrep4_eq16_s12"]["elo"], "rank": sp["gm2l_cvrep4_eq16_s12"]["rank"],
           "note": "locked; the same rule priced 1835.20 before a column-name fix"},
          {"label": "Plus per-table classical learners in a second stacker (48 predictions)", "elo": tp["locked_single_placement"]["elo"], "rank": tp["locked_single_placement"]["rank"],
           "note": "locked entrant of pass 7"}]
l45 = json.load(open(R11 + "FRESH_STACK_XFIT_OWN_FINETUNE.json"))
p45 = l45["paired"]["stk_ft - stk_base"]; n45 = l45["paired"]["stk_ftnaive - stk_base"]
f8_ = lambda k: k[4:12]
line45 = {"w": p45["contexts_wtl"][0], "l": p45["contexts_wtl"][2], "du": p45["mean_family_du"], "p": p45["signflip_p_two_sided"],
          "naive_w": n45["contexts_wtl"][0], "naive_l": n45["contexts_wtl"][2], "naive_du": n45["mean_family_du"],
          "by_family": sorted(({"family": f8_(k), "du": v} for k, v in p45["family_du"].items()), key=lambda r: -r["du"]), "scored": l45["time_utc"]}
cg = json.load(open(R11 + "WHATIF_CAPGAP_v1.json"))
fr = json.load(open(R11 + "S10_FIELDING_P201B121.json")); tc = json.load(open(R11 + "S10C_TAILCAL_P201B121.json")); nl = json.load(open(R11 + "S10D_NOLOOK_D91.json"))
rev11 = {"s9": s9, "s9_cycle_runbag_t": next(v["T_over_ctrl_a"] for k, v in cyc["rules"].items() if k.endswith("_runbag")), "series": series,
         "series_ci": tp["locked_single_placement"]["ci95"], "line45": line45, "capgap": cg["rows"], "capgap_ctx": cg["probe_contexts"],
         "fielding": {"v0_elo": fr["arms"]["V0"]["elo"], "n_ctx": len(fr["contexts"]), "top_share": fr["artefact_summary"]["V0"]["median_top_share"],
                      "n_top_half": fr["artefact_summary"]["V0"]["n_top_share_over_half"]},
         "tailcal_elo": {k: v["elo"] for k, v in tc["elo"].items()}, "nolook_configs": len(nl["counts"]), "nolook_max_leaves": max(v["n_leaves"] for v in nl["counts"].values()),
         "inputs_sha256": {n_: hashlib.sha256(open(R11 + n_, "rb").read()).hexdigest() for n_ in sorted(os.listdir(R11))}}

# ----------------------------------------------------------------------------- revision 14: the H462 placements, listed and labelled, and the inventory
def h462_status(a: dict) -> str:
    p_, lab_ = a["pass_index"], a["label"]
    if p_ == 7 and a["arm"] == "gm2l_cvrep4_eq16_s12":
        return "0% dev91; control: re-price of the pass-6 locked rule (third look)"
    if p_ == 7 and a["arm"] == "gm2l_rep0_eq4_s12":
        return "0% dev91; control: near-replica of the pass-3 locked rule"
    if p_ in (1, 2):
        return f"{lab_}; unlocked post-hoc dev91 read"
    if a["locked"]:
        if lab_ != "0% dev91":
            return f"{lab_}; locked entrant of pass {p_}"
        if p_ == 4:
            return "0% dev91; locked rule, superseded (column-name bug; re-priced in pass 6)"
        tail_ = "; re-price after the pass-4 column-name fix (second look at the same rule)" if p_ == 6 else ""
        return f"0% dev91 exposure; dev91 development read, pass {p_} of an adaptive sequence; locked entrant{tail_}"
    return f"{lab_}; post-lock descriptive (unranked)"
PASS_FILE = {v["pass_index"]: k for k, v in ADD["sources"].items() if "pass_index" in v}
h462_rows = sorted(({"pass": a["pass_index"], "file": os.path.basename(a["file"]).replace(".json", ""), "arm": a["arm"], "elo": a["elo"], "lo": a["ci95"][0], "hi": a["ci95"][1],
                     "rank33": a["rank33"], "family_mean_u": a["family_mean_u"], "locked": a["locked"], "label": a["label"], "status": h462_status(a)} for a in h462),
                   key=lambda r: (r["pass"], -r["elo"]))
# comparability with the bundle: every receipt that re-places the pre-registered fair champion alone must reproduce the frozen board's own rating of it
repro = {}
for a in h462:
    if a["arm"] in ("n0_mean_conv", "h448_fair_champion.one_run", "fair_h448_r4e8_m_multiseed.n0_mean_conv"):
        repro[a["pass_index"]] = a["elo"] - field[OUR_FAIR]["elo"]
inventory = [{"source": k, "arms": v["arms"], "pass": v.get("pass_index"), "ledger_label": v.get("ledger_label"), "locked_entrant": v.get("locked_entrant"),
              "ledger_count": v.get("n_entrants_scored_ledger"), "sha256": v["sha256"], "fair_champion_repro_delta": repro.get(v.get("pass_index"))} for k, v in ADD["sources"].items()]
h456_stack = sorted(({"checkpoint": ckpt_label(a["file"]), "elo": a["elo"], "rank": a["rank33"]} for a in h456 if a["arm"] == "stack_no_ugp_post_h454"), key=lambda r: -r["elo"])
rev14 = {"h462_rows": h462_rows, "inventory": inventory, "n_h456": len(h456), "n_h462": len(h462), "n_bundle_arms": N_BUNDLE_ARMS, "n_bundle_files": N_BUNDLE_FILES,
         "n_h462_locked": sum(1 for a in h462 if a["locked"]), "h456_stack": h456_stack, "h456_best": max(h456, key=lambda a: a["elo"])["elo"],
         "h462_max_incorpus": max((a for a in h462 if a["label"].startswith("in-corpus")), key=lambda a: a["elo"])["elo"],
         "h462_max_unlocked_fair": max((a for a in h462 if a["label"] == "0% dev91" and not a["locked"]), key=lambda a: a["elo"])["elo"],
         "top50_all_rank5": all(t["rank34"] == 5 for t in top50), "top50_kumo_wtl": sorted({tuple(t["wtl_kumo_l"]) for t in top50 if t["wtl_kumo_l"]}),
         "top50_waves": dict(collections.Counter(t["wave"] for t in top50)), "addendum_sha256": hashlib.sha256(open(HERE + "arms_addendum_rev14.json", "rb").read()).hexdigest()}

# ----------------------------------------------------------------------------- revision 17: the KPI (datasets through training) and the fleet addendum
kpi_ = json.load(open(HERE + "kpi_datasets_rev22.json"))
live_ = [r for r in kpi_["all_runs"] if kpi_ and (datetime_ts(kpi_["utc"]) - r["mtime"]) < 900]
rev17 = {"n_fleet_arms": len(ADD17["arms"]), "n_fleet_files": len(ADD17["sources"]), "addendum_sha256": hashlib.sha256(open(HERE + "arms_addendum_rev17.json", "rb").read()).hexdigest(),
         "fleet_best": max(ADD17["arms"], key=lambda a: a["elo"])["elo"],
         "kpi": {"utc": kpi_["utc"], "total": kpi_["total_datasets_fed"], "runs": kpi_["runs"], "target": kpi_["target"], "per_host": kpi_["per_host"], "rate_per_s": kpi_["rate_per_s"],
                 "top_runs": [{"run": r["run"], "host": r["host"], "n": r["n"], "d91_share": r["d91_share"], "real": r.get("real")} for r in kpi_["all_runs"][:12]],
                 "n_with_d91_share": sum(1 for r in kpi_["all_runs"] if r["d91_share"] is not None),
                 "policy": kpi_["policy"], "n_lineage": len(kpi_["unique_lineage_runs"]), "hosts_stale": kpi_.get("hosts_stale", {}),
                 "real": kpi_["real_table_episodes"], "real_known_in": kpi_["real_known_in_datasets"], "unique_logged": kpi_["unique_logged_sum"],
                 "unique_logged_in": kpi_["unique_logged_in_datasets"], "dups_logged": kpi_["dups_logged_sum"],
                 "live": [{"run": r["run"], "host": r["host"], "n": r["n"], "d91_share": r["d91_share"]} for r in live_]}}

out = {
    "look2": look2, "rev10": rev10, "rev11": rev11, "rev14": rev14, "rev17": rev17,
    "h2h": h2h, "head": head, "league": league, "tabpfn_decomp": decomp,
    "generated_utc": d["generated_utc"], "board": {k: board[k] for k in ("name", "sha256", "description", "split")}, "n_field": len(field_rows),
    "field_rows": field_rows, "systems": systems, "ladder": ladder, "top50": top50, "all_unique": all_unique, "n_arm_rows": len(arms), "n_unique": len(uniq),
    "n_distinct_pred": n_distinct_pred, "n_distinct_elo": n_distinct_elo,
    "n_files": len({a["file"] for a in arms}), "ugp": ugp, "ugp_base": ugp_base, "traj": traj, "checkpoints": checkpoints, "pairs": pairs, "refs": REFS,
    "ref_names": {r: NAMES[r] for r in REFS}, "family_table": family_table, "family_wins": [{"id": k, "name": NAMES.get(k, k), "families_best": n} for k, n in wins.most_common()],
    "vs_fair": vs_fair, "vs_kumo": vs_kumo, "ctx_delta": ctx_delta, "heat": heat, "sources": d["sources"],
    "r4_matched": d["r4_final"]["matched_random_split_d91_delta"],
}
json.dump(out, open(HERE + "report_tables.json", "w"), indent=1)
print("ladder rows", len(ladder), "systems", len(systems), "unique arm scores", len(uniq))
for r in ladder[:14]:
    print(f"  {r['pos']:2d} {r['elo']:8.2f} [{r['lo']:.0f},{r['hi']:.0f}] {r['kind']:13s} {r['name']}")
print("pairs:", [(p["label"], round(p["fair"], 1), round(p["in_corpus"], 1), round(p["delta"], 1)) for p in pairs])
print("family wins:", out["family_wins"])
print("fair champion vs Kumo L:", versus(OUR_FAIR, "kumo_tabular_large_e8.v1"))
print("fair champion vs TabPFN 3.5:", versus(OUR_FAIR, "tabpfn_v3p5.v1"))
print("ugp:", [(u["family"], u["gen"], u["variant"], round(u["elo"], 1)) for u in ugp])
print("checkpoints:", [(c["label"], round(c.get("stack", 0), 1), round(c.get("neural", 0), 1), round(c.get("icl", 0), 1)) for c in checkpoints])
print("traj epochs", [t["epoch"] for t in traj])
