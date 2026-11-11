#!/usr/bin/env python3
"""H433 census: which tables of a panel hold a two-column formula feature, and (TRAIN board P201b only) the share of
the stored foundation-model gap those tables carry. Reads stored values only; fits nothing.

  census_h433.py flags --cols DIR --out FILE
  census_h433.py gap   --flags FILE --apply APPLY.json --board-scores DIR --out FILE

flags (SEARCH rows only; from cols_h433.py's <cid>.json records):
  F  formula table   a derived feature survived the screen in at least 4 of the 5 folds, its best smoother gain over
                     both parents is at least 0.3 u and its best smoother score is at least 1.0 u
  none               otherwise
gap (P201b): per flag, the number of tables, the share of the positive gap sum of max(0, u_FM - u_SXB), the mean
  signed gap and the foundation model's win count; the same against the F4 arm of the stack step; the fraction of the
  positive gap closed on F tables; the mean absolute change of the stack on unflagged tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

F_MIN_FOLDS, F_MIN_GAIN, F_MIN_SCORE = 4, 0.3, 1.0
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
        f = r["survivor_folds"] >= F_MIN_FOLDS and r["best_gain"] >= F_MIN_GAIN and r["best_score"] >= F_MIN_SCORE
        names = [s["name"] for s in r["refit"]["survivors"][:3]]
        out[r["cid"]] = {"flag": "F" if f else "none", "survivor_folds": r["survivor_folds"], "best_gain": r["best_gain"],
                         "best_score": r["best_score"], "refit_survivors": names, "oof_u": r["oof_u"],
                         "width": r["width"], "n_search": r["n_search"]}
    counts = {}
    for v in out.values():
        counts[v["flag"]] = counts.get(v["flag"], 0) + 1
    res = {"schema": "jope.h433.census-flags/v1", "n": len(out), "counts": counts, "contexts": out,
           "rule": {"F_MIN_FOLDS": F_MIN_FOLDS, "F_MIN_GAIN": F_MIN_GAIN, "F_MIN_SCORE": F_MIN_SCORE}}
    with open(a.out, "x") as fh:
        json.dump(res, fh, sort_keys=True)
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
    for fl in ("F", "none", "unfinished"):
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
    res = {"schema": "jope.h433.census-gap/v1", "n": len(cids), "dropped": len(per) - len(cids)}
    for k in FM:
        res[k] = {"vs_SXB": _by_flag(flags, cids, {c: fm[k][c] - per[c]["B"] for c in cids}),
                  "vs_F4": _by_flag(flags, cids, {c: fm[k][c] - per[c]["F4"] for c in cids})}
        b, f4 = res[k]["vs_SXB"].get("F"), res[k]["vs_F4"].get("F")
        if b and b["positive_gap_sum"] > 0:
            res[k]["closed_on_F"] = 1.0 - f4["positive_gap_sum"] / b["positive_gap_sum"]
    un = [abs(per[c]["F4"] - per[c]["B"]) for c in cids if flags[c]["flag"] == "none"]
    res["unflagged_mean_abs_change"] = sum(un) / len(un) if un else None
    res["f_tables"] = sum(flags[c]["flag"] == "F" for c in cids)
    res["f_share_of_kumo_gap"] = (res["kumo"]["vs_SXB"].get("F") or {}).get("share_of_positive_gap") or 0.0
    with open(a.out, "x") as fh:
        json.dump(res, fh, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("n", "f_tables", "f_share_of_kumo_gap", "unflagged_mean_abs_change")}))
    for k in FM:
        print(k, json.dumps(res[k]["vs_SXB"]), "closed on F:", res[k].get("closed_on_F"))


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
