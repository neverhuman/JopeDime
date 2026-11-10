#!/usr/bin/env python3
"""Build a board's ladder from STORED scores only (never fits or runs a model).

  ladder.py BOARD_DIR [--anchor ID] [--focus ID] [--only ID,ID,...] [--write]

Reads BOARD_DIR/BOARD.json and every BOARD_DIR/scores/<id>.jsonl (entrant id = file stem). Refuses a record whose
board or envelope sha256 differs from BOARD.json, and a duplicate (entrant, cid). An entrant missing any context is
listed as incomplete and left out of the Elo fit.

Statistics are the H335 functions, imported unchanged from research/flywheel_selfplay/h335_aggressive_locked_elo/
h335_stats.py (sha256 ee3b4543..., the code the H335 verdict used):
  * AUDIT-Elo: elo_panel (family-grouped MM Bradley-Terry, tie 1e-4, prior 0.01, anchor = 1500, 10,000-draw family
    bootstrap, seed 335160);
  * focus vs each other entrant: headline (unclipped family-equal mean gain in u, % RMSE cut = 1 - exp(-gain),
    families won/tied/lost) and primary_test (clip +/-0.5, one-sided sign-flip: exact up to 24 families, else 2^20
    Monte Carlo; LOFO min; family median).
--write saves ladder.json and LADDER.md next to BOARD.json.
"""
import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
H335_DIR = Path(os.environ.get("REFBOARD_H335_DIR",
                               HERE.parents[2] / "flywheel_selfplay" / "h335_aggressive_locked_elo"))
if not (H335_DIR / "h335_stats.py").exists():
    H335_DIR = Path("/home/ubuntu/h335-stage/code")  # frozen read-only copy of the same file (xbabe2)
sys.path.insert(0, str(H335_DIR))
import h335_stats as H  # noqa: E402


def load(board_dir: Path, only=None):
    board = json.loads((board_dir / "BOARD.json").read_text())
    env = {c["cid"]: c["envelope_sha256"] for c in board["contexts"]}
    fam_of = {c["cid"]: c["family_id"] for c in board["contexts"]}
    values, meta, incomplete = {}, {}, {}
    n_preds_checked = [0]
    for p in sorted((board_dir / "scores").glob("*.jsonl")):
        eid = p.stem
        if only and eid not in only:
            continue
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        seen = collections.Counter(r["cid"] for r in recs)
        dup = [c for c, n in seen.items() if n > 1]
        if dup:
            raise SystemExit(f"{eid}: {len(dup)} duplicate contexts")
        for r in recs:
            if r["cid"] not in env:
                raise SystemExit(f"{eid}: context {r['cid'][:12]} is not on board {board['board']}")
            if r.get("board", board["board"]) != board["board"]:
                raise SystemExit(f"{eid}: record for board {r.get('board')}")
            if "envelope_sha256" in r and r["envelope_sha256"] != env[r["cid"]]:
                raise SystemExit(f"{eid}: envelope sha mismatch at {r['cid'][:12]}")
            pf = board_dir / "preds" / eid / f"{r['cid']}.f64"
            if "preds_sha256" in r and pf.parent.is_dir():  # board roots hold preds/; the git copy does not
                if hashlib.sha256(pf.read_bytes()).hexdigest() != r["preds_sha256"]:
                    raise SystemExit(f"{eid}: prediction file sha mismatch at {r['cid'][:12]}")
                n_preds_checked[0] += 1
        got = {r["cid"]: float(r["utility"]) for r in recs}
        st = dict(collections.Counter(r.get("status", "completed") for r in recs))
        walls = [float(r["wall_s"]) for r in recs if "wall_s" in r]
        m = {"status": st, "n": len(recs), "median_wall_s": statistics.median(walls) if walls else None,
             "label": (recs[0].get("spec") or {}).get("label") if recs else None}
        if set(got) != set(env):
            incomplete[eid] = {"have": len(got), "need": len(env), **m}
            continue
        values[eid], meta[eid] = got, m
    return board, fam_of, values, meta, incomplete, n_preds_checked[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("board_dir")
    ap.add_argument("--anchor", default="h335.H257_REPLICA")
    ap.add_argument("--focus", default="h335.H335_AGG")
    ap.add_argument("--only", default="")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    bd = Path(a.board_dir)
    only = set(a.only.split(",")) if a.only else None
    board, fam_of, values, meta, incomplete, n_preds = load(bd, only)
    if a.anchor not in values:
        raise SystemExit(f"anchor {a.anchor} has no complete scores on this board")
    elo = H.elo_panel(values, fam_of, a.anchor)
    fam_mean = {e: H.family_mean(v, fam_of) for e, v in values.items()}
    vs = {}
    if a.focus in values:
        for e in sorted(values):
            if e != a.focus:
                h, p = H.headline(values[a.focus], values[e], fam_of), H.primary_test(values[a.focus], values[e], fam_of)
                vs[e] = {"headline": h, "primary_clipped": p}
    out = {"schema": "jope.reference-board.ladder/v1", "board": board["board"], "board_sha256": board["board_sha256"],
           "n_contexts": board["n_contexts"], "n_families": board["n_families"], "anchor": a.anchor,
           "focus": a.focus, "elo": elo, "family_mean_u": fam_mean, "entrants": meta, "incomplete": incomplete,
           "focus_vs": vs, "prediction_files_verified": n_preds}
    order = sorted(values, key=lambda e: -elo["elo"][e]["elo"])
    lab = "AUDIT-Elo" if "audit" in board["board"] else "Elo"  # two-fold boards are not AUDIT scores
    md = [f"# Ladder: `{board['board']}`", "",
          f"Generated by `tools/ladder.py` from stored scores only. {board['n_contexts']} contexts / "
          f"{board['n_families']} families; board sha256 `{board['board_sha256'][:16]}…`.",
          f"{lab} anchor: `{a.anchor}` = 1500 (family bootstrap 95% CI). Elo depends on the field: compare entrants "
          "within this table, and use the head-to-head table for effect sizes.", "",
          f"| Rank | Entrant | {lab} | 95% CI | Family-mean u | Median wall (s) | Status |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, e in enumerate(order, 1):
        r = elo["elo"][e]
        st = ", ".join(f"{k} {v}" for k, v in sorted(meta[e]["status"].items()))
        w = meta[e]["median_wall_s"]
        md.append(f"| {i} | `{e}` | {r['elo']:.1f} | {r['ci95'][0]:.0f}–{r['ci95'][1]:.0f} | {fam_mean[e]:.4f} | "
                  f"{'' if w is None else f'{w:.1f}'} | {st} |")
    if incomplete:
        md += ["", "Incomplete (not rated): " + ", ".join(f"`{e}` {v['have']}/{v['need']}" for e, v in incomplete.items())]
    if vs:
        md += ["", f"## `{a.focus}` head to head", "",
               "Gain = family-equal mean of u(focus) − u(other); RMSE cut = 1 − exp(−gain). Clipped T / p / LOFO: the "
               "H335 primary procedure (clip ±0.5, one-sided sign-flip). Positive = focus better.", "",
               "| vs | Gain | RMSE cut | Families W/T/L | Clipped T | Sign-flip p | LOFO min | Passes |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for e in sorted(vs, key=lambda e: -vs[e]["headline"]["family_mean_gain"]):
            h, p = vs[e]["headline"], vs[e]["primary_clipped"]
            md.append(f"| `{e}` | {h['family_mean_gain']:+.4f} | {100 * h['pct_rmse_reduction']:+.1f}% | "
                      f"{h['families_won']}/{h['families_tied']}/{h['families_lost']} | {p['T_clipped_family_mean']:+.4f} | "
                      f"{p['signflip_p']:.4g} | {p['lofo_min']:+.4f} | {'yes' if p['pass'] else 'no'} |")
    text = "\n".join(md) + "\n"
    if a.write:
        (bd / "ladder.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        (bd / "LADDER.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
