#!/usr/bin/env python3
"""H381 E1 / E2-in-loop analysis (PREREG §2): every output on S60 vs the STORED H364 X0 AGG.g50.CV3@12.

Reads only TRAIN run outputs: stage records (stages/C_AGG/<cid>__<seed>.jsonl) and engine memos (gproot_AGG/memo).
  u            AUDIT utility (record value); X0 from runs/H364_C_xbabe0 (never rerun)
  search CPU   summed worker wall-seconds of the unique SEARCH evaluations an output used:
               LF outputs: low-fidelity memo records whose sha has gen <= gc (or is a seed-A/E extra);
               FR outputs: the same + the full-row memo records of that checkpoint's FR shas;
               AGG outputs (lf_rows 0): the memo SEARCH records with gen <= gc (+ seed extras).
               X0: every SEARCH record of its memo (it ran g50 only). AUDIT CPU and unit wall are reported apart.
Tests: h335_stats.primary_test / headline (clip 0.5), plus the PREREG family-bootstrap 95% LCB (10,000, seed 381150).

  analyze_e1_h381.py --x0 RUN --arm NAME=RUN [--arm ...] --panels panels.json [--panel S60|R60|S60+R60] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

H335_SRC = Path("/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo")
sys.path.insert(0, str(H335_SRC))
import h335_stats as H  # noqa: E402  (numpy only)

X0_KEY = "AGG.g50.CV3@12"
SEED = 364101


def load_records(run: Path, cids) -> Dict[str, dict]:
    out = {}
    for c in cids:
        p = run / "stages" / "C_AGG" / f"{c}__{SEED}.jsonl"
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        res = [r for r in recs if r.get("type") == "result"]
        if len(res) != 1:
            raise SystemExit(f"{p}: {len(res)} result records")
        out[c] = res[0]
    return out


def memo_walls(run: Path, cid: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """mode -> kind -> {sha: wall_s} (first record per sha)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for p in sorted((run / "gproot_AGG" / "memo").glob(f"{cid}.*.jsonl")):
        mode = p.name[len(cid) + 1:-len(".jsonl")]
        d = out.setdefault(mode, {})
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d.setdefault(r["kind"], {}).setdefault(r["sha"], float(r.get("wall_s") or 0.0))
    return out


def boot_lcb(arm: Dict[str, float], ctrl: Dict[str, float], fam_of, n: int = 10000, seed: int = 381150) -> float:
    _, D = H.family_contrast(arm, ctrl, fam_of, clip=H.CLIP)
    rng = np.random.default_rng(seed)
    means = D[rng.integers(0, len(D), size=(n, len(D)))].mean(axis=1)
    return float(np.quantile(means, 0.025))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x0", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--panel", default="S60", help="panel name(s) in panels.json, joined by + to pool (PREREG §8)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pj = json.loads(Path(a.panels).read_text())
    pan = {"cids": [c for n in a.panel.split("+") for c in pj[n]["cids"]],
           "families": [f for n in a.panel.split("+") for f in pj[n]["families"]]}
    if len(set(pan["families"])) != len(pan["cids"]):
        raise SystemExit("panel families must be distinct")
    cids, fam_of = pan["cids"], dict(zip(pan["cids"], pan["families"]))
    x0 = load_records(Path(a.x0), cids)
    x0u = {c: float(x0[c]["outputs"][X0_KEY]["value"]) for c in cids}
    x0cpu, x0w = {}, {}
    for c in cids:
        w = x0w[c] = memo_walls(Path(a.x0), c)["split"]
        x0cpu[c] = {"search": sum(w.get("search", {}).values()), "audit": sum(w.get("audit", {}).values())}
    X0_SEARCH = sum(v["search"] for v in x0cpu.values())
    rep = {"schema": "jope.h381.e1-analysis/v1", "panel": a.panel, "n": len(cids), "x0_key": X0_KEY,
           "x0": {"family_mean_u": H.family_mean(x0u, fam_of), "search_cpu_s": X0_SEARCH,
                  "audit_cpu_s": sum(v["audit"] for v in x0cpu.values()),
                  "unit_wall_s": sum(float(x0[c]["wall_s"]) for c in cids)}, "outputs": {}}
    for spec in a.arm:
        name, run = spec.split("=", 1)
        recs = load_records(Path(run), cids)
        keys = sorted(recs[cids[0]]["outputs"])
        memo = {c: memo_walls(Path(run), c) for c in cids}
        for k in keys:
            fam, gc_s, _cfg = k.split(".", 2)
            gc = int(gc_s[1:])
            vals, cpu = {}, 0.0
            for c in cids:
                r = recs[c]
                o = r["outputs"][k]
                vals[c] = float(o["value"])
                gen_of = r["gen_of"]
                mode = f"lf{r['lf_rows']}" if r["lf_active"] else "split"
                srch = memo[c].get(mode, {}).get("search", {})
                cpu += sum(w for s, w in srch.items() if gen_of.get(s, 0) <= gc)
                if fam == "FR":
                    full = memo[c].get("split", {}).get("search", {})
                    cpu += sum(full[s] for s in r["fr"][str(gc)]["shas"] if s in full)
            pt = H.primary_test(vals, x0u, fam_of)
            hl = H.headline(vals, x0u, fam_of)
            rep["outputs"][f"{name}:{k}"] = {
                "family_mean_u": H.family_mean(vals, fam_of), "primary_test": pt, "headline": hl,
                "boot_lcb95_clipped": boot_lcb(vals, x0u, fam_of), "search_cpu_s": cpu,
                "search_cpu_ratio_vs_x0": cpu / X0_SEARCH, "invalid": sum(1 for c in cids if
                                                                      recs[c]["outputs"][k].get("invalid")),
                "per_context_delta": {c: vals[c] - x0u[c] for c in cids}}
        audit = sum(sum(memo[c].get(m, {}).get("audit", {}).values()) for c in cids for m in memo[c])
        # host-speed check (descriptive, added after part A): the same full-row SEARCH program on the same context timed
        # in this run and in X0; a ratio below 1 means this run's host was faster, which flatters search_cpu_ratio
        same_x0 = same_run = 0.0
        n_same = 0
        for c in cids:
            mine, ref = memo[c].get("split", {}).get("search", {}), x0w[c].get("search", {})
            for s in mine.keys() & ref.keys():
                same_x0, same_run, n_same = same_x0 + ref[s], same_run + mine[s], n_same + 1
        rep.setdefault("runs", {})[name] = {"audit_cpu_s_all_outputs": audit,
                                            "unit_wall_s": sum(float(recs[c]["wall_s"]) for c in cids),
                                            "lf_rows": recs[cids[0]]["lf_rows"],
                                            "host_speed": {"n_same_programs": n_same, "x0_wall_s": same_x0,
                                                           "run_wall_s": same_run,
                                                           "ratio": same_run / same_x0 if same_x0 else None}}
    rules = {}
    o = rep["outputs"]
    if "A:FR.g100.CV3@12" in o:
        r = o["A:FR.g100.CV3@12"]
        rules["E1_win"] = {"output": "A:FR.g100.CV3@12", "primary_pass": r["primary_test"]["pass"],
                           "cpu_ratio": r["search_cpu_ratio_vs_x0"],
                           "pass": bool(r["primary_test"]["pass"] and r["search_cpu_ratio_vs_x0"] <= 1.0)}
    if "A:FR.g50.CV3@12" in o:
        r = o["A:FR.g50.CV3@12"]
        rules["E1_speed"] = {"output": "A:FR.g50.CV3@12", "boot_lcb95": r["boot_lcb95_clipped"],
                             "cpu_ratio": r["search_cpu_ratio_vs_x0"],
                             "pass": bool(r["boot_lcb95_clipped"] > -0.005 and r["search_cpu_ratio_vs_x0"] <= 0.40)}
    rep["rules"] = rules
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True))
    rows = []
    for k, r in sorted(o.items()):
        hl, pt = r["headline"], r["primary_test"]
        rows.append(f"{k:28s} gain {hl['family_mean_gain']:+.4f} rmse {100 * hl['pct_rmse_reduction']:+.2f}% "
                    f"W/T/L {hl['families_won']}/{hl['families_tied']}/{hl['families_lost']} "
                    f"p {pt['signflip_p']:.3f} lcb {r['boot_lcb95_clipped']:+.4f} cpu x{r['search_cpu_ratio_vs_x0']:.2f}")
    print("\n".join(rows))
    print(json.dumps(rules))
    if not all(math.isfinite(r["family_mean_u"]) for r in o.values()):
        raise SystemExit("non-finite output")


if __name__ == "__main__":
    main()
