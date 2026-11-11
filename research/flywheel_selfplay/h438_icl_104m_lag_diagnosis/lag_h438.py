#!/usr/bin/env python3
"""H438: why does the 104M in-context model lag the 24M model at matched hours? Stored logs and monitor reads only.

  lag_h438.py run --logs DIR --mon DIR --out FILE
  lag_h438.py selftest

Inputs: read-only copies of the four training logs (24M ICL_M_r3; 104M workers w0, w1, w2 of ICL_L_g1) and the
stored FIT-SEL monitor reads. No model is loaded, no pass is made, nothing is written outside --out.
Clocks: h = training hours of the log; step = optimizer steps of that process; tau = steps since the process
switched to the current recipe (residual-target tasks, prior v2). Window W = 104M hours 12 to 24.5.
Numbers (PREREG section 3):
  N1  gap 104M - 24M of the in-training real-table residual median at matched tau, mean over the evaluations in W
  N2  loss in the first minute after loading the group average minus loss in the last minute before it (W)
  N3  learning rate in W against each run's peak, and the schedule positions
  N4  gap of the synthetic held-out residual u at matched tau; realised task shares
Verdict (PREREG section 4), from worker 2, replicated on workers 0 and 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

W_LO, W_HI, TOL, BUMP_MIN = 12.0, 24.5, 0.02, 0.01
CHANGE = {"M_r3": 138685, "L_w0": 51167, "L_w1": 45598, "L_w2": 40692}   # step at which each process took the current recipe
PARAMS = {"M": 23659392, "L": 104360576}


def load(p: Path) -> dict:
    steps, syncs, evals, order = [], [], [], []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "sync" in r:
            syncs.append(r); order.append(("sync", len(syncs) - 1))
        elif "eval" in r:
            evals.append(r)
        elif "resume" in r:
            continue
        elif "loss" in r and "lr" in r:
            steps.append(r); order.append(("step", len(steps) - 1))
    return {"steps": steps, "syncs": syncs, "evals": evals, "order": order}


def interp(x, xs, ys):
    """Linear interpolation; nan outside the range of xs."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    out = np.interp(x, xs, ys)
    return np.where((np.asarray(x) < xs[0]) | (np.asarray(x) > xs[-1]), np.nan, out)


def series(evals, key, clock, change=0):
    pts = [((e["step"] - change) if clock == "tau" else e[clock], e[key]) for e in evals if e.get(key) is not None]
    pts.sort()
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def gap(L, M, key, clock, cl, cm) -> dict:
    """104M minus 24M at matched clock, over the 104M evaluations in the window."""
    ev = [e for e in L["evals"] if W_LO <= e["h"] <= W_HI and e.get(key) is not None]
    if not ev:
        return {"n": 0}
    x = np.array([(e["step"] - cl) if clock == "tau" else e[clock] for e in ev], float)
    xm, ym = series(M["evals"], key, clock, cm)
    g = np.array([e[key] for e in ev], float) - interp(x, xm, ym)
    k = np.isfinite(g)
    if not k.any():
        return {"n": 0, "evaluations_in_window": len(ev)}
    g = g[k]
    return {"n": int(len(g)), "evaluations_in_window": len(ev), "mean": float(g.mean()), "sd": float(g.std(ddof=1)) if len(g) > 1 else None,
            "se_naive": float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None, "min": float(g.min()), "max": float(g.max()),
            "first_third_mean": float(g[:max(1, len(g) // 3)].mean()), "last_third_mean": float(g[-max(1, len(g) // 3):].mean())}


def bump(run: dict, pseudo: bool = False) -> dict:
    """Loss by minute after a sync, minus the last minute before it. pseudo: blocks of 5 log lines for a run without syncs."""
    S, order = run["steps"], run["order"]
    rounds = []
    if pseudo:
        idx = [i for i, s in enumerate(S) if W_LO <= s["h"] <= W_HI]
        for j in range(5, len(idx) - 5, 5):
            rounds.append((S[idx[j - 1]]["loss"], [S[i]["loss"] for i in idx[j:j + 5]]))
    else:
        pos = [i for i, (k, _) in enumerate(order) if k == "sync"]
        for a, b in zip(pos, pos[1:] + [len(order)]):
            sy = run["syncs"][order[a][1]]
            if not (W_LO <= sy["h"] <= W_HI) or not sy.get("loaded", True):
                continue
            pre = [order[i][1] for i in range(max(0, a - 3), a) if order[i][0] == "step"]
            post = [order[i][1] for i in range(a + 1, b) if order[i][0] == "step"]
            if pre and len(post) >= 3:
                rounds.append((S[pre[-1]]["loss"], [S[i]["loss"] for i in post[:5]]))
    if len(rounds) < 5:
        return {"rounds": len(rounds)}
    out = {"rounds": len(rounds), "pre_mean": float(np.mean([r[0] for r in rounds])), "by_minute_after": []}
    for p in range(5):
        d = np.array([r[1][p] - r[0] for r in rounds if len(r[1]) > p])
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        out["by_minute_after"].append({"minute": p + 1, "n": int(len(d)), "mean": float(d.mean()), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]})
    return out


def facts(run: dict, name: str, cfg: dict, workers: int) -> dict:
    S = [s for s in run["steps"] if W_LO <= s["h"] <= W_HI]
    a, b = S[0], S[-1]
    dh = b["h"] - a["h"]
    sy = [s for s in run["syncs"] if W_LO <= s["h"] <= W_HI]
    wait = float(sum(s.get("wait_s") or 0.0 for s in sy))
    at24 = min(run["steps"], key=lambda s: abs(s["h"] - 24.0))
    size = "M" if name.startswith("M") else "L"
    o = {"steps_per_hour": (b["step"] - a["step"]) / dh, "tasks_per_step": (b["datasets"] - a["datasets"]) / max(1, b["step"] - a["step"]),
         "tasks_per_hour": (b["datasets"] - a["datasets"]) / dh, "step_at_h24": at24["step"], "tasks_at_h24": at24["datasets"],
         "tokens_per_step_config": cfg["tokens"], "row_tokens_config": cfg["row_tokens"],
         "tokens_per_parameter_at_h24_one_process": at24["step"] * cfg["tokens"] / PARAMS[size],
         "tokens_per_parameter_at_h24_all_workers": workers * at24["step"] * cfg["tokens"] / PARAMS[size],
         "lr_peak_config": cfg["lr"], "lr_min_in_window": float(min(s["lr"] for s in S)), "lr_max_in_window": float(max(s["lr"] for s in S)),
         "warmup_ends_h": 0.02 * cfg["hours"], "decay_starts_h": 0.85 * cfg["hours"], "schedule_hours": cfg["hours"],
         "loss_mean_in_window": float(np.mean([s["loss"] for s in S])),
         "shares_in_window": {k: float(np.mean([s[k] for s in S if s.get(k) is not None])) for k in ("real_share", "inj_share", "res_share", "rows_top") if any(s.get(k) is not None for s in S)},
         "syncs_in_window": len(sy)}
    if sy:
        d = np.array([s["dist"] for s in sy if s.get("dist") is not None], float)
        hh = np.array([s["h"] for s in sy if s.get("dist") is not None], float)
        o.update({"wait_s_mean": wait / len(sy), "wait_share_of_wall_time": wait / (dh * 3600.0 + wait),
                  "workers_in_average": {str(k): int(sum(1 for s in sy if s.get("n") == k)) for k in sorted({s.get("n") for s in sy if s.get("n") is not None})},
                  "dist_mean": float(d.mean()) if len(d) else None, "dist_first_quarter": float(d[:max(1, len(d) // 4)].mean()) if len(d) else None,
                  "dist_last_quarter": float(d[-max(1, len(d) // 4):].mean()) if len(d) else None,
                  "dist_slope_per_hour": float(np.polyfit(hh, d, 1)[0]) if len(d) > 2 else None})
    return o


def slope(run: dict, key: str, lo_step: float, hi_step: float, change: int) -> dict:
    x, y = series(run["evals"], key, "tau", change)
    k = (x >= lo_step) & (x <= hi_step)
    if k.sum() < 3:
        return {"n": int(k.sum())}
    c = np.polyfit(x[k], y[k], 1)
    return {"n": int(k.sum()), "per_10k_steps": float(c[0] * 1e4), "from": float(y[k][0]), "to": float(y[k][-1])}


def cmd_run(a) -> None:
    logs = Path(a.logs)
    runs = {n: load(logs / f"{n}.log.jsonl") for n in ("M_r3", "L_w0", "L_w1", "L_w2")}
    cfgs = {n: json.loads((logs / f"{n}.config.json").read_text()) for n in runs}
    M = runs["M_r3"]
    res = {"schema": "jope.h438.lag/v1", "window_hours_104M": [W_LO, W_HI], "tolerance_u": TOL, "change_steps": CHANGE,
           "records": {n: {"step_lines": len(r["steps"]), "syncs": len(r["syncs"]), "evals": len(r["evals"]), "last_h": r["steps"][-1]["h"]} for n, r in runs.items()},
           "facts": {n: facts(runs[n], n, cfgs[n], 1 if n == "M_r3" else 3) for n in runs}, "workers": {}}
    keys = {"real_res_u_median": "real residual median", "real_res_win_vs_ridge": "real residual win rate", "real_u_median": "real plain median",
            "real_win_vs_ridge": "real plain win rate", "u_res": "synthetic residual u", "u": "synthetic plain u", "loss": "evaluation loss"}
    for w in ("L_w2", "L_w0", "L_w1"):
        L = runs[w]
        o = {"gaps": {k: {"at_matched_tau": gap(L, M, k, "tau", CHANGE[w], CHANGE["M_r3"]), "at_matched_total_steps": gap(L, M, k, "step", 0, 0),
                          "at_matched_hours": gap(L, M, k, "h", 0, 0)} for k in keys},
             "sync_bump": bump(L)}
        tau = [e["step"] - CHANGE[w] for e in L["evals"] if 18.0 <= e["h"] <= W_HI]
        if tau:
            o["flatness_h18_to_h24"] = {"104M": slope(L, "real_res_u_median", min(tau), max(tau), CHANGE[w]),
                                        "24M_same_tau_range": slope(M, "real_res_u_median", min(tau), max(tau), CHANGE["M_r3"]),
                                        "tau_range": [float(min(tau)), float(max(tau))]}
        res["workers"][w] = o
    res["control_pseudo_rounds_24M"] = bump(M, pseudo=True)
    mon = {}
    for p in sorted(Path(a.mon).glob("*.json")):
        nm = p.stem
        if nm.startswith(("M_r3_h0", "L_w2_h0")) and "pre" not in nm and not nm.startswith(("stack", "oof")):
            j = json.loads(p.read_text())
            st, sm = j.get("state", {}), j.get("summary", {})
            mon[nm] = {"step": st.get("step"), "h": (round(st["elapsed"] / 3600.0, 2) if st.get("elapsed") is not None else None),
                       "ridgeres_median": (sm.get("ridgeres") or {}).get("median"), "full_median": (sm.get("full") or {}).get("median")}
    res["monitor_reads"] = mon
    g = res["workers"]["L_w2"]["gaps"]["real_res_u_median"]
    G, Gh = g["at_matched_tau"].get("mean"), g["at_matched_hours"].get("mean")
    b1 = res["workers"]["L_w2"]["sync_bump"].get("by_minute_after", [{}])[0]
    syn = res["workers"]["L_w2"]["gaps"]["u_res"]["at_matched_tau"].get("mean")
    rep = [res["workers"][w]["gaps"]["real_res_u_median"]["at_matched_tau"].get("mean") for w in ("L_w0", "L_w1")]
    bump_sig = bool(b1 and b1.get("mean", 0) >= BUMP_MIN and b1.get("ci95", [0, 0])[0] > 0)
    mix_sig = bool(syn is not None and G is not None and syn >= TOL and G <= -TOL)
    if G is None:
        verdict = "NOT_SEPARATED"
    elif G >= -TOL:
        verdict = "STEPS_FAVORED" if all(r is not None and r >= -TOL for r in rep) else "NOT_SEPARATED"
    elif bump_sig and mix_sig:
        verdict = "NOT_SEPARATED"
    elif bump_sig:
        verdict = "AVERAGING_FAVORED"
    elif mix_sig:
        verdict = "MIX_FAVORED"
    else:
        verdict = "OPTIMISATION_UNSEPARATED"
    res["decision"] = {"N1_gap_at_matched_tau_w2": G, "N1_replicates_w0_w1": rep, "gap_at_matched_hours_w2": Gh, "N2_bump_first_minute": b1,
                       "N2_bump_meets_rule": bump_sig, "N4_synthetic_gap_at_matched_tau": syn, "N4_mix_signature": mix_sig}
    res["verdict"] = verdict
    f = lambda d: "n/a" if not d.get("n") else f"{d['mean']:+.3f} (n {d['n']}, sd {d['sd'] if d['sd'] is None else round(d['sd'], 3)}, {d['min']:+.3f}..{d['max']:+.3f})"
    print("== H438: records " + "; ".join(f"{n} steps {r['step_lines']} syncs {r['syncs']} evals {r['evals']} last h {r['last_h']}" for n, r in res["records"].items()))
    for n, x in res["facts"].items():
        print(f"   {n}: steps/h {x['steps_per_hour']:.0f} tasks/step {x['tasks_per_step']:.2f} tasks/h {x['tasks_per_hour']:.0f} step@24h {x['step_at_h24']} "
              f"tokens/param@24h {x['tokens_per_parameter_at_h24_one_process']:.0f} (all workers {x['tokens_per_parameter_at_h24_all_workers']:.0f}) "
              f"lr {x['lr_min_in_window']:.2e}..{x['lr_max_in_window']:.2e} peak {x['lr_peak_config']:.2e} warmup<{x['warmup_ends_h']:.2f}h decay>{x['decay_starts_h']:.1f}h "
              f"loss {x['loss_mean_in_window']:.3f} shares {({k: round(v, 3) for k, v in x['shares_in_window'].items()})}"
              + (f" wait {x['wait_s_mean']:.0f}s/round = {100 * x['wait_share_of_wall_time']:.1f}% of wall; dist {x['dist_mean']:.4f} "
                 f"({x['dist_first_quarter']:.4f} -> {x['dist_last_quarter']:.4f}); n {x['workers_in_average']}" if x.get("syncs_in_window") else ""))
    for w, o in res["workers"].items():
        print(f"   {w}:")
        for k, lab in keys.items():
            gg = o["gaps"][k]
            print(f"      {lab:26s} matched tau {f(gg['at_matched_tau'])} | matched total steps {f(gg['at_matched_total_steps'])} | matched hours {f(gg['at_matched_hours'])}")
        sb = o["sync_bump"]
        if sb.get("by_minute_after"):
            print(f"      loss after a sync minus before ({sb['rounds']} rounds): " + " ".join(f"m{x['minute']} {x['mean']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]" for x in sb["by_minute_after"]))
        if o.get("flatness_h18_to_h24"):
            print(f"      residual median slope per 10k steps, h18-24: {o['flatness_h18_to_h24']}")
    cb = res["control_pseudo_rounds_24M"]
    if cb.get("by_minute_after"):
        print(f"   24M control (blocks of 5 log lines, {cb['rounds']}): " + " ".join(f"m{x['minute']} {x['mean']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]" for x in cb["by_minute_after"]))
    print(f"   monitor reads: { {k: (v['step'], v['h'], v['ridgeres_median'], v['full_median']) for k, v in mon.items()} }")
    print(f"   decision {res['decision']}")
    print(f"   verdict {verdict}")
    with open(a.out, "x") as fo:
        json.dump(res, fo, sort_keys=True)


def cmd_selftest(a) -> None:
    r = np.random.default_rng(5)
    mk = lambda n, rate, off: {"evals": [{"step": int(i * rate * 1000), "h": float(i), "real_res_u_median": 0.5 + 0.5 * (1 - np.exp(-i * rate / 6)) + off + 0.005 * r.standard_normal()} for i in range(1, n)],
                               "steps": [], "syncs": [], "order": []}
    L, M = mk(26, 7.7, 0.0), mk(26, 17.6, 0.0)
    print("selftest gap at matched tau", gap(L, M, "real_res_u_median", "tau", 0, 0), "at matched hours", gap(L, M, "real_res_u_median", "h", 0, 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for k in ("--logs", "--mon", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
