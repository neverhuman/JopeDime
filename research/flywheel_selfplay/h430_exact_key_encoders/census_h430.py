#!/usr/bin/env python3
"""H430 stage 0: structure census of a panel from the H430 column records, and (TRAIN board P201b only) the share of
the stored foundation-model gap each structure type carries. Reads stored values only; fits nothing.

  census_h430.py flags --cols DIR --out FILE
  census_h430.py gap   --flags FILE --apply APPLY.json --board-scores DIR --out FILE

flags (SEARCH rows only; from cols_h430.py's <cid>.json records):
  K  key-determined   the best key reached DIRECT_MIN_U in at least 4 of the 5 folds, and the out-of-fold u of the
                      key_direct column is at least 2.0
  C  factor table     at least half of the columns have at most 20 levels, and the out-of-fold u of factor_glm is at
                      least 0.3 above that of gbm_plain (and the table is not K)
  none                otherwise
gap (P201b): per flag, the number of tables, the share of the positive gap sum of max(0, u_Kumo - u_SXB), the mean
  signed gap, and the foundation model's win count; then the same against the K3 arm of the stack step, the fraction
  of the positive gap closed on K tables, and the mean absolute change of the stack on unflagged tables.
  u_SXB is the stored value B of APPLY.json; Kumo L and TabPFN 3.5 come from the board's score files (TRAIN board
  reference rows, FMGAP Part A). Verdict CENSUS_CONCENTRATED when K tables carry at least 20% of the positive
  Kumo - SXB gap, else CENSUS_DIFFUSE; fewer than 10 K tables ends the wave (KEYS_RARE).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

K_MIN_FOLDS, K_MIN_U, C_MIN_SHARE, C_MIN_GAIN = 4, 2.0, 0.5, 0.3
FM = {"kumo": "kumo_tabular_large_e8.v1", "tabpfn35": "tabpfn_v3p5.v1"}


def cmd_flags(a) -> None:
    out = {}
    for p in sorted(Path(a.cols).glob("*.json")):
        if p.name.startswith("cols_status_"):
            continue
        r = json.loads(p.read_text())
        if not r.get("ok"):
            out[r["cid"]] = {"flag": "unfinished"}
            continue
        ref = r["refit"]
        share = ref["n_factor"] / max(ref["n_cols"], 1)
        u = r["oof_u"]
        if r["direct_folds"] >= K_MIN_FOLDS and u["key_direct"] >= K_MIN_U:
            flag = "K"
        elif share >= C_MIN_SHARE and u["factor_glm"] - u["gbm_plain"] >= C_MIN_GAIN:
            flag = "C"
        else:
            flag = "none"
        out[r["cid"]] = {"flag": flag, "direct_folds": r["direct_folds"], "oof_u": u, "factor_share": share,
                         "best_key": ref["best"], "width": r["width"], "n_search": r["n_search"]}
    counts = {}
    for v in out.values():
        counts[v["flag"]] = counts.get(v["flag"], 0) + 1
    res = {"schema": "jope.h430.census-flags/v1", "n": len(out), "counts": counts, "contexts": out,
           "rule": {"K_MIN_FOLDS": K_MIN_FOLDS, "K_MIN_U": K_MIN_U, "C_MIN_SHARE": C_MIN_SHARE, "C_MIN_GAIN": C_MIN_GAIN}}
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)
    print(json.dumps({"n": res["n"], "counts": counts}))


def _board(dirp: str, model: str) -> dict:
    out = {}
    for line in (Path(dirp) / f"{model}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "completed":
            out[r["cid"]] = float(r["utility"])
    return out


def _by_flag(flags: dict, cids: list, gap: dict) -> dict:
    tot = sum(max(0.0, gap[c]) for c in cids)
    res = {}
    for fl in ("K", "C", "none", "unfinished"):
        g = [gap[c] for c in cids if flags[c]["flag"] == fl]
        if g:
            res[fl] = {"tables": len(g), "share_of_positive_gap": sum(max(0.0, x) for x in g) / tot if tot > 0 else None,
                       "mean_signed_gap": sum(g) / len(g), "fm_wins": sum(x > 1e-4 for x in g),
                       "positive_gap_sum": sum(max(0.0, x) for x in g)}
    return res


def cmd_gap(a) -> None:
    flags = json.loads(Path(a.flags).read_text())["contexts"]
    per = json.loads(Path(a.apply).read_text())["per_context"]
    fm = {k: _board(a.board_scores, m) for k, m in FM.items()}
    cids = sorted(c for c in per if c in flags and all(c in v for v in fm.values()))
    res = {"schema": "jope.h430.census-gap/v1", "n": len(cids), "dropped": len(per) - len(cids)}
    for k in FM:
        res[k] = {"vs_SXB": _by_flag(flags, cids, {c: fm[k][c] - per[c]["B"] for c in cids}),
                  "vs_K3": _by_flag(flags, cids, {c: fm[k][c] - per[c]["K3"] for c in cids})}
        for fl in ("K", "C"):
            b, k3 = res[k]["vs_SXB"].get(fl), res[k]["vs_K3"].get(fl)
            if b and b["positive_gap_sum"] > 0:
                res[k][f"closed_on_{fl}"] = 1.0 - k3["positive_gap_sum"] / b["positive_gap_sum"]
    un = [abs(per[c]["K3"] - per[c]["B"]) for c in cids if flags[c]["flag"] == "none"]
    res["unflagged_mean_abs_change"] = sum(un) / len(un) if un else None
    nk = sum(flags[c]["flag"] == "K" for c in cids)
    share = (res["kumo"]["vs_SXB"].get("K") or {}).get("share_of_positive_gap") or 0.0
    res["k_tables"], res["k_share_of_kumo_gap"] = nk, share
    res["verdict"] = "KEYS_RARE" if nk < 10 else ("CENSUS_CONCENTRATED" if share >= 0.20 else "CENSUS_DIFFUSE")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("n", "k_tables", "k_share_of_kumo_gap", "unflagged_mean_abs_change", "verdict")}))
    for k in FM:
        print(k, json.dumps(res[k]["vs_SXB"]), "closed:", {x: res[k].get(f"closed_on_{x}") for x in ("K", "C")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("flags"); p.add_argument("--cols", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_flags)
    p = sub.add_parser("gap"); p.add_argument("--flags", required=True); p.add_argument("--apply", required=True)
    p.add_argument("--board-scores", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_gap)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
