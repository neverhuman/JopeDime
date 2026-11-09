#!/usr/bin/env python3
"""H416 TRAIN driver (PREREG §5-§8).

  deploy  --mode dev|confirm --panel P201e|P201b|SMK2 --run-dir R --tag T --cores L [--jobs J] [--inner-jobs 3]
          [--timeout 300] [--design DESIGN.json]
          deploy_h416.h416_unit over the panel. Sources GS, GE, GL (per-fold out-of-fold screeners H0.1-A,
          H381-E2-S3fz, H415-L). The run dir's memo is the per-context concatenation of the source runs' memos (made
          by run_h416.sh deploy-prep). Search seed: 415101 on P201e (H415's arms), 335101 on P201b (the stored H335
          Phase T run and the H416 arms). SMK2 = the first 2 P201e contexts (smoke; dev stages only).
  select  --run-dir R --tag T --h415-dir D --panel-file F --out DESIGN.json
          P201e design grid and the PREREG §6 selection rule (numpy + h335_stats + meta_h416 only).
  analyze --run-dir R --tag T --gs-t D --gp GE=D --gp GL=D --board-dir D --panel-file F --design F --p201e F --out F
          P201b readout (PREREG §8): board Elo field, paired Elo tests, Holm, recommendation rule.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

SEEDS = {"P201e": 415101, "SMK2": 415101, "P201b": 335101, "ID2B": 335101}
SMOKE_PANELS = ("SMK2", "ID2B")  # first 2 contexts of P201e (dev path) / of P201b (confirm path)
NEURAL = {"GS": "H0.1-A", "GE": "H381-E2-S3fz", "GL": "H415-L"}
STAGE_GP = {"GE": "E416_AGG", "GL": "E416_AGG"}
KEY_GP = "AGG.g50.CV3@12"
BOOT_SEED, N_BOOT = 416150, 10_000
LADDER_SEED = 335160  # the board ladder's own seed (tools/ladder.py), for the reproduction check only
ANCHOR = "h335.H257_REPLICA"
X0B = "h335.H335_AGG"
REF_KFB, REF_BAG3 = "P3+LIB3.KFB", "P3+LIB3.BAG_CAR"
NEW = {"h416.SXB": "P3+LIB3.SXB", "h415.KFB": "P3+LIB3.KFB", "h410s.BAG3": "P3+LIB3.BAG_CAR",
       "h410s.BAG2": "P2+LIB2.BAG_CAR"}
COST_ORDER = ("h410s.BAG2", "h410s.BAG3", "h415.KFB", "h416.SXB")  # least to most compute (PREREG §8.3)
HOLM_ALPHA = 0.05
CHECK_TOL = 1e-9


# ----------------------------------------------------------------------------------------------------------------------
# deploy
# ----------------------------------------------------------------------------------------------------------------------
def cmd_deploy(a) -> None:
    import deploy_h416 as DH  # FIRST import of the GP stack: h335_core pins + guard

    import gp_h415 as GP15

    K, C, L, D = DH.K, DH.C, DH.L, DH.D
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    if a.panel in ("P201e", "SMK2"):
        cids = GP15.panel("P201e")
        if a.panel == "SMK2":  # smoke only: run_h416.sh writes it to H416_SMOKE_* run dirs, never scored
            cids = cids[:2]
        if a.mode == "confirm" and a.panel != "SMK2":
            raise SystemExit("confirm runs on P201b (SMK2 for the smoke)")
    else:
        import gp_h416 as GP16

        cids = GP16.panel(a.panel)  # P201b, or ID2B (smoke: its first 2 contexts)
        if a.mode != "confirm":
            raise SystemExit("P201b is the confirmation panel: mode confirm only")
    design = None
    if a.mode == "confirm":
        design = json.loads(Path(a.design).read_text())
        if design.get("schema") != "jope.h416.design/v1" or design.get("design") not in DH.M.DESIGNS:
            raise SystemExit("not an H416 DESIGN.json")
    seed = SEEDS[a.panel]
    run_dir = Path(a.run_dir)
    memo = run_dir / f"gproot_{K.GROUP}" / "memo"
    missing = [c for c in cids if not (memo / f"{c}.split.jsonl").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no concatenated memo (run deploy-prep first)")
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(C.DATA_ROOT) / "neural" / "runs" / NEURAL[s] for s in ("GS", "GE", "GL")}
    rt, scr, man = DH.DP.build_rt_t(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (seed,), nd)
    cfg = {"tag": a.tag, "mode": a.mode, "panel": a.panel, "cids": cids, "sources": list(nd),
           "neural": {s: str(p) for s, p in nd.items()}, "seed": seed, "timeout_s": a.timeout,
           "design": design, "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                                      "max_members": DH.M.MAX_MEMBERS},
           "code_sha256": K.code_manifest(DH.code_files() + ("gp_h415.py", "train_h415.py", "train_h416.py",
                                                             "gp_h416.py")),
           "screener_manifest": man}
    p = run_dir / f"config_{a.tag}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if any(old.get(k) != v for k, v in cfg.items()):
            raise SystemExit("run config mismatch")
    else:
        C.write_json(p, cfg)
    DH.GPH.install(0)
    npz_dir = run_dir / f"h416_{a.tag}_npz"
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    recs = L.run_parallel(lambda c, _s: DH.h416_unit(rt, a.tag, c, scr, seed, a.mode, npz_dir, design,
                                                     a.inner_jobs), [(c, 0) for c in order], a.jobs)
    st = {"tag": a.tag, "mode": a.mode, "units": len(recs), "wall_s": time.time() - t0, "engine": rt.engine.stats(),
          "audit_ledger": rt.ledger.check(), "akf_ledger": DH.DP15.KF_LEDGER.check(), "guard": C.guard_report(),
          "bank_counters": dict(D.BANK_COUNTERS), "physical_by_kind": DH.DP.physical_by_kind(rt),
          "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_{a.tag}_{int(t0)}.json", st)
    rt.engine.close()
    print(json.dumps({"tag": a.tag, "mode": a.mode, "units": len(recs), "wall_s": round(st["wall_s"], 1),
                      "physical_by_kind": st["physical_by_kind"], "akf_ledger": st["akf_ledger"]}))


# ----------------------------------------------------------------------------------------------------------------------
# shared readout helpers (numpy + h335_stats + meta_h416; no GP stack)
# ----------------------------------------------------------------------------------------------------------------------
def _stats_mod():
    sys.path.append("/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo")
    import numpy as np
    import h335_stats as H

    return np, H


def _last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def out_value(rec15: dict, k: str, x0: float):
    """(value, invalid, bag_fallback) of output k of an H415-stage record: BAG_<b> missing -> <b> (H410 rule); an
    invalid or missing output scores the baseline value x0 (0 gain) and is counted (H415 rule)."""
    o = rec15["outputs"]
    alts = [k] + ([k.replace("BAG_", "")] if "BAG_" in k else [])
    for i, kk in enumerate(alts):
        if kk in o:
            if o[kk].get("invalid") or o[kk]["raw"].get("status") != "completed":
                return x0, True, i > 0
            return float(o[kk]["value"]), False, i > 0
    return x0, True, False


def load_npz(np, rec: dict, run_dir: Path):
    p = run_dir / f"h416_{rec['tag']}_npz" / f"{rec['cid']}.npz"
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != rec["npz"]["sha256"]:
        raise SystemExit(f"{p}: npz sha differs from its record")
    z = np.load(p, allow_pickle=False)
    na = len(z["ya"])
    bag = [None if not np.all(np.isfinite(z["AKF"][:, i])) else z["AKF"][:, i] for i in range(z["AKF"].shape[1])]
    refit = [z["PA"][:, i] if bool(z["pa_ok"][i]) else None for i in range(z["PA"].shape[1])]
    refs = {str(k): (None if bool(n) else z["REF"][:, j]) for j, (k, n) in enumerate(zip(z["ref_keys"],
                                                                                       z["ref_none"]))}
    return {"Q": z["Q"], "y": z["y"], "ya": z["ya"], "ba": float(z["ba"]), "ym": float(z["ym"]), "na": na,
            "bag": bag, "refit": refit, "refs": refs}


def design_values(np, M, recs16: dict, run_dir: Path) -> dict:
    """{design: {cid: u}} from the npz matrices (offline; SEARCH-only weights, stored AUDIT columns)."""
    out = {d: {} for d in M.DESIGNS}
    info = {d: [] for d in M.DESIGNS}
    for c, r in recs16.items():
        z = load_npz(np, r, run_dir)
        for d in M.DESIGNS:
            pred, inf = M.design_pred(d, z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])
            out[d][c] = M.utility(pred, z["ya"], z["ba"])[0]
            info[d].append(inf)
    return out, info


def ref_check(np, M, recs16: dict, recs15: dict, run_dir: Path) -> float:
    """max |u(captured reference AUDIT prediction) - the H415 record's value| over every valid reference output (the
    CONS designs combine these predictions; PREREG §9.2)."""
    worst = 0.0
    for c, r in recs16.items():
        if r.get("error"):
            continue
        z = load_npz(np, r, run_dir)
        o = recs15[c]["outputs"]
        for k, p in z["refs"].items():
            if p is None:
                continue
            kk = next((x for x in [k] + ([k.replace("BAG_", "")] if "BAG_" in k else []) if x in o), None)
            if kk is None or o[kk].get("invalid") or o[kk]["raw"].get("status") != "completed":
                continue
            worst = max(worst, abs(M.utility(p, z["ya"], z["ba"])[0] - float(o[kk]["value"])))
    return float(worst)


def pair_stats(np, H, a: dict, b: dict, fam_of: dict) -> dict:
    """Family win fraction of a over b (tie 1e-4 = 0.5) and the clipped / unclipped contrasts."""
    cids = sorted(a)
    V = np.array([[a[c], b[c]] for c in cids])
    W = H.family_wins_tensor(V, [fam_of[c] for c in cids])
    pt = H.primary_test(a, b, fam_of)
    hl = H.headline(a, b, fam_of)
    return {"win_frac": float(W[:, 0, 1].mean()), "clipped_mean": pt["T_clipped_family_mean"],
            "signflip_p": pt["signflip_p"], "mirror_signflip_p": pt["mirror_signflip_p"], "lofo_min": pt["lofo_min"],
            "family_median": pt["family_median"], "pass": pt["pass"], "inferior": pt["inferior"],
            "gain": hl["family_mean_gain"], "pct_rmse": hl["pct_rmse_reduction"],
            "wtl": [hl["families_won"], hl["families_tied"], hl["families_lost"]]}


def elo_boot(np, H, values: dict, fam_of: dict, anchor: str, seed: int, n_boot: int):
    """h335_stats.elo_panel's computation, returning the bootstrap draws as well (identical point / CI)."""
    arms = list(values)
    cids = sorted(set.intersection(*[set(v) for v in values.values()]))
    V = np.array([[values[x][c] for x in arms] for c in cids])
    W = H.family_wins_tensor(V, [fam_of[c] for c in cids])
    ai = arms.index(anchor)
    point = H.bt_ratings(W.sum(axis=0), anchor=ai)
    rng = np.random.default_rng(seed)
    F = W.shape[0]
    ix = rng.integers(0, F, size=(n_boot, F))
    boot = np.concatenate([H.bt_ratings(W[ix[i:i + 500]].sum(axis=1), anchor=ai) for i in range(0, n_boot, 500)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    elo = {x: {"elo": float(point[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, x in enumerate(arms)}
    return arms, point, boot, elo


def delta_elo(np, arms, point, boot, x: str, y: str) -> dict:
    i, j = arms.index(x), arms.index(y)
    d = boot[:, i] - boot[:, j]
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"a": x, "b": y, "delta": float(point[i] - point[j]), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean()))),
            "p_le0": float((d <= 0).mean())}


def holm(ps: dict, alpha: float = HOLM_ALPHA) -> dict:
    order = sorted(ps, key=lambda k: (ps[k], k))
    m, rej, go = len(order), {}, True
    for i, k in enumerate(order):
        go = go and ps[k] <= alpha / (m - i)
        rej[k] = go
    return rej


# ----------------------------------------------------------------------------------------------------------------------
# select (P201e, PREREG §6)
# ----------------------------------------------------------------------------------------------------------------------
def cmd_select(a) -> None:
    np, H = _stats_mod()
    import meta_h416 as M

    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    pf = json.loads(Path(a.panel_file).read_text())
    if not ((cfg["panel"] == "P201e" and sorted(pf["cids"]) == sorted(cids)) or
            (cfg["panel"] == "SMK2" and set(cids) <= set(pf["cids"]))):
        raise SystemExit("select reads the full P201e dev run (SMK2: smoke)")
    fam_of = {c: pf["family_of"][c] for c in cids}
    recs16 = {c: _last_result(run_dir / "stages" / f"H416_{a.tag}" / f"{c}__dev.jsonl") for c in cids}
    recs15 = {c: _last_result(run_dir / "stages" / f"H415_{recs16[c].get('recomputed_as') or a.tag}"
                              / f"{c}__pool.jsonl") for c in cids}
    orig = {c: _last_result(Path(a.h415_dir) / f"{c}__pool.jsonl") for c in cids}
    # harness checks (PREREG §9.3)
    keys = sorted(set().union(*[set(r["outputs"]) for r in orig.values()]))
    rep_d = {}
    for k in keys:
        ds = [abs(recs15[c]["outputs"][k]["value"] - orig[c]["outputs"][k]["value"]) for c in cids
              if k in recs15[c]["outputs"] and k in orig[c]["outputs"]]
        miss = sum(1 for c in cids if (k in recs15[c]["outputs"]) != (k in orig[c]["outputs"]))
        rep_d[k] = {"max_abs_delta": float(max(ds)) if ds else None, "presence_mismatch": miss}
    kfb_d = max(abs(recs16[c]["kcar100_bag_u"] - orig[c]["outputs"][REF_KFB]["value"]) for c in cids)
    no_usable = [c for c in cids if recs16[c].get("error")]
    checks = {"h415_outputs_reproduced": rep_d, "kcar100_bag_vs_stored_kfb_max_abs_delta": float(kfb_d),
              "contexts_without_usable": len(no_usable),
              "akf_failed": int(sum(r.get("n_akf_failed", 0) for r in recs16.values())),
              "pa_failed": int(sum(r.get("n_pa_failed", 0) for r in recs16.values())),
              "ref_pred_max_abs_delta": ref_check(np, M, recs16, recs15, run_dir)}
    bad = [k for k in (REF_KFB, REF_BAG3, "P3+LIB3.CAR", "P3+LIB3.W48", "P3+LIB3.HFR", "P2+LIB2.BAG_CAR",
                       "P2+LIB2.CAR", "GS.stock") if k in rep_d and (rep_d[k]["max_abs_delta"] or 0) > CHECK_TOL]
    if kfb_d > CHECK_TOL or bad or no_usable or checks["ref_pred_max_abs_delta"] > CHECK_TOL:
        print(json.dumps(checks, indent=1))
        raise SystemExit(f"harness check failed (kfb delta {kfb_d}, outputs {bad}, no usable {len(no_usable)}, "
                         f"ref preds {checks['ref_pred_max_abs_delta']})")
    x0 = {c: float(orig[c]["outputs"]["GS.stock"]["value"]) for c in cids}
    refs = {k: {c: out_value(recs15[c], k, x0[c])[0] for c in cids} for k in (REF_KFB, REF_BAG3)}
    vals, info = design_values(np, M, recs16, run_dir)
    scores = {}
    for d in M.DESIGNS:
        sk = pair_stats(np, H, vals[d], refs[REF_KFB], fam_of)
        sb = pair_stats(np, H, vals[d], refs[REF_BAG3], fam_of)
        mem = [i.get("n_members") for i in info[d] if i.get("n_members") is not None]
        scores[d] = {"S": min(sk["win_frac"], sb["win_frac"]), "eligible": sk["clipped_mean"] >= 0
                     and sb["clipped_mean"] >= 0, "vs_KFB": sk, "vs_BAG3": sb,
                     "members_mean": float(np.mean(mem)) if mem else None,
                     "vs_X0e": pair_stats(np, H, vals[d], x0, fam_of)}
    pool = [d for d in M.DESIGNS if d != "KCAR100.bag"]
    elig = [d for d in pool if scores[d]["eligible"]]
    cand = elig or pool

    def key(d):
        s = scores[d]
        return (-s["S"], -min(s["vs_KFB"]["clipped_mean"], s["vs_BAG3"]["clipped_mean"]),
                s["members_mean"] if s["members_mean"] is not None else 0.0, M.DESIGNS.index(d))

    chosen = sorted(cand, key=key)[0]
    el = H.elo_panel({"X0e": x0, **{k: refs[k] for k in refs}, **{d: vals[d] for d in M.DESIGNS}}, fam_of, "X0e",
                     seed=BOOT_SEED, n_boot=N_BOOT)
    man = hashlib.sha256("".join(f"{c} {recs16[c]['npz']['sha256']}\n" for c in cids).encode()).hexdigest()
    out = {"schema": "jope.h416.design/v1", "design": chosen, "no_eligible_design": not elig,
           "rule": "PREREG §6: among the designs other than KCAR100.bag (= KFB) with clipped family-mean contrast >= 0 "
                   "against both KFB and P3+LIB3.BAG_CAR on P201e, the largest S = min(family win fraction vs KFB, vs "
                   "P3+LIB3.BAG_CAR); ties: larger min clipped contrast, then fewer mean members, then grid order",
           "dev_panel": cfg["panel"], "n_contexts": len(cids), "n_families": len(set(fam_of.values())),
           "checks": checks, "npz_manifest_sha256": man, "scores": scores, "elo_P201e": el,
           "meta_constants": {"bag_seed": M.BAG_SEED, "n_bags": M.N_BAGS, "w_min": M.W_MIN,
                              "max_members": M.MAX_MEMBERS}}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    Path(a.refs_out).write_text(json.dumps({"schema": "jope.h416.p201e-refs/v1", "KFB": refs[REF_KFB],
                                            "BAG3": refs[REF_BAG3], "family_of": fam_of}, sort_keys=True) + "\n")
    print(f"chosen {chosen}  (eligible designs {len(elig)}/{len(pool)})  checks kfb_delta {kfb_d:.2e}")
    for d in sorted(M.DESIGNS, key=key):
        s = scores[d]
        print(f"{d:16s} S {s['S']:.4f} elig {int(s['eligible'])} vsKFB win {s['vs_KFB']['win_frac']:.3f} "
              f"clip {s['vs_KFB']['clipped_mean']:+.4f} | vsBAG3 win {s['vs_BAG3']['win_frac']:.3f} "
              f"clip {s['vs_BAG3']['clipped_mean']:+.4f} | vsX0e gain {s['vs_X0e']['gain']:+.4f} "
              f"mem {s['members_mean'] if s['members_mean'] is None else round(s['members_mean'], 1)}")
    for k, v in sorted(el["elo"].items(), key=lambda kv: -kv[1]["elo"]):
        print(f"elo {k:16s} {v['elo']:7.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]")


# ----------------------------------------------------------------------------------------------------------------------
# analyze (P201b, PREREG §8)
# ----------------------------------------------------------------------------------------------------------------------
def load_board(board_dir: Path):
    """The staged P201b board (BOARD.json + score files, tools/ladder.py's load rules): entrants with all contexts."""
    b = json.loads((board_dir / "board_p201b_BOARD.json").read_text())
    env = {c["cid"]: c["envelope_sha256"] for c in b["contexts"]}
    fam_of = {c["cid"]: c["family_id"] for c in b["contexts"]}
    values, incomplete = {}, {}
    for p in sorted(board_dir.glob("board_p201b__*.jsonl")):
        eid = p.name[len("board_p201b__"):-len(".jsonl")]
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len({r["cid"] for r in recs}) != len(recs):
            raise SystemExit(f"{eid}: duplicate contexts")
        for r in recs:
            if r["cid"] not in env or r.get("board", b["board"]) != b["board"] or \
                    ("envelope_sha256" in r and r["envelope_sha256"] != env[r["cid"]]):
                raise SystemExit(f"{eid}: record not on the board / envelope mismatch")
        got = {r["cid"]: float(r["utility"]) for r in recs}
        if set(got) != set(env):
            incomplete[eid] = len(got)
            continue
        values[eid] = got
    return b, env, fam_of, values, incomplete


def cmd_analyze(a) -> None:
    np, H = _stats_mod()
    import meta_h416 as M

    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    design = json.loads(Path(a.design).read_text())
    if cfg["design"] != design:
        raise SystemExit("the deploy ran a different design")
    board, env, bfam, field, incomplete = load_board(Path(a.board_dir))
    checks = {}
    # (1) board reproduction: the committed ladder from the stored entrants alone (always the full board)
    lad = json.loads((Path(a.board_dir) / "board_p201b_ladder.json").read_text())
    _, _, _, el0 = elo_boot(np, H, field, bfam, ANCHOR, LADDER_SEED, N_BOOT)
    checks["board_ladder_reproduced_max_abs"] = float(max(
        max(abs(el0[e]["elo"] - v["elo"]), abs(el0[e]["ci95"][0] - v["ci95"][0]), abs(el0[e]["ci95"][1] - v["ci95"][1]))
        for e, v in lad["elo"]["elo"].items()))
    checks["board_entrants"] = sorted(field)
    checks["board_incomplete"] = incomplete
    pf = json.loads(Path(a.panel_file).read_text())
    smoke = cfg["panel"] == "ID2B"  # smoke: the board restricted to the deploy's contexts
    if smoke:
        if not set(cids) <= set(env):
            raise SystemExit("smoke contexts are not on the board")
        env, bfam = {c: env[c] for c in cids}, {c: bfam[c] for c in cids}
        field = {e: {c: v[c] for c in cids} for e, v in field.items()}
    elif cfg["panel"] != "P201b" or sorted(pf["cids"]) != sorted(cids):
        raise SystemExit("analyze reads the full P201b confirm run (ID2B: smoke)")
    if sorted(cids) != sorted(env) or not set(cids) <= set(pf["cids"]) or \
            any(pf["family_of"][c] != bfam[c] for c in cids):
        raise SystemExit("deploy / panel / board contexts or families differ")
    fam_of = bfam
    recs16 = {c: _last_result(run_dir / "stages" / f"H416_{a.tag}" / f"{c}__confirm.jsonl") for c in cids}
    recs15 = {c: _last_result(run_dir / "stages" / f"H415_{recs16[c].get('recomputed_as') or a.tag}"
                              / f"{c}__pool.jsonl") for c in cids}
    # (2) replay: GS stock == the stored Phase T output == the board's h335.H335_AGG; GE / GL stock == their records
    t = {c: _last_result(Path(a.gs_t) / f"{c}__{SEEDS['P201b']}.jsonl")["outputs"][KEY_GP] for c in cids}
    gp = dict(x.split("=", 1) for x in a.gp)
    stored = {n: {c: _last_result(Path(d) / "stages" / STAGE_GP[n] / f"{c}__{SEEDS['P201b']}.jsonl")["outputs"][KEY_GP]
                  for c in cids} for n, d in gp.items()}
    rep = {"GS": {"max_abs_delta_vs_T": float(max(abs(recs15[c]["outputs"]["GS.stock"]["value"] - t[c]["value"])
                                                  for c in cids)),
                  "max_abs_delta_vs_board": float(max(abs(recs15[c]["outputs"]["GS.stock"]["value"] - field[X0B][c])
                                                      for c in cids)),
                  "members_same": sum(1 for c in cids if (recs15[c]["outputs"]["GS.stock"].get("spec") or {}).get(
                      "members") == (t[c].get("spec") or {}).get("members"))}}
    for n in gp:
        rep[n] = {"max_abs_delta": float(max(abs(recs15[c]["outputs"][f"{n}.stock"]["value"] - stored[n][c]["value"])
                                             for c in cids)),
                  "members_same": sum(1 for c in cids if (recs15[c]["outputs"][f"{n}.stock"].get("spec") or {}).get(
                      "members") == (stored[n][c].get("spec") or {}).get("members"))}
    checks["replay"] = rep
    # (3) AUDIT envelope = the board's envelope; (4) SXB re-derived offline from the npz columns
    # the board envelope is the AUDIT Jtt1's to_bytes(); the worker copy (env_sha) adds 10 zero query rows
    checks["audit_envelope_matches_board"] = sum(1 for c in cids if recs16[c].get("audit_board_sha256") == env[c])
    ok16 = {c: r for c, r in recs16.items() if not r.get("error")}  # error = no usable H program (SXB invalid)
    checks["contexts_without_usable"] = len(cids) - len(ok16)
    vals, _ = design_values(np, M, ok16, run_dir)
    sx = {c: float(r["outputs"]["P3+LIB3.SXB"]["value"]) for c, r in ok16.items()}
    checks["sxb_offline_max_abs_delta"] = float(max(abs(vals[design["design"]][c] - sx[c]) for c in ok16))
    checks["kcar100_bag_vs_kfb_max_abs_delta"] = float(max(abs(r["kcar100_bag_u"] - r["kfb_u"]) for r in ok16.values()))
    checks["ref_pred_max_abs_delta"] = ref_check(np, M, ok16, recs15, run_dir)
    # gate (PREREG §9.2): no entrant value is computed or printed unless every harness check passes
    n = len(cids)
    gate = {"board_ladder": checks["board_ladder_reproduced_max_abs"] <= 1e-6,
            "gs_replay": rep["GS"]["max_abs_delta_vs_T"] <= CHECK_TOL
            and rep["GS"]["max_abs_delta_vs_board"] <= CHECK_TOL
            and rep["GS"]["members_same"] == n,
            **{f"{g}_replay": rep[g]["max_abs_delta"] <= CHECK_TOL and rep[g]["members_same"] == n for g in gp},
            "audit_envelope": checks["audit_envelope_matches_board"] == n,
            "sxb_offline": checks["sxb_offline_max_abs_delta"] <= CHECK_TOL,
            "kcar100_bag_is_kfb": checks["kcar100_bag_vs_kfb_max_abs_delta"] <= CHECK_TOL,
            "ref_preds": checks["ref_pred_max_abs_delta"] <= CHECK_TOL}
    checks["gate"] = gate
    if not all(gate.values()):
        Path(a.out).write_text(json.dumps({"schema": "jope.h416.train-analysis/v1", "panel": cfg["panel"],
                                           "gate_failed": sorted(k for k, v in gate.items() if not v),
                                           "checks": checks}, indent=1, sort_keys=True) + "\n")
        raise SystemExit(f"harness gate failed: {sorted(k for k, v in gate.items() if not v)} (no entrant read)")
    x0 = field[X0B]
    new, inv, bagfb = {}, {}, {}
    for e, k in NEW.items():
        if k == "P3+LIB3.SXB":
            new[e] = {c: sx.get(c, x0[c]) for c in cids}
            inv[e], bagfb[e] = len(cids) - len(sx), 0
            continue
        ov = {c: out_value(recs15[c], k, x0[c]) for c in cids}
        new[e] = {c: v[0] for c, v in ov.items()}
        inv[e] = sum(1 for v in ov.values() if v[1])
        bagfb[e] = sum(1 for v in ov.values() if v[2])
    # (5) the board panel: stored complete entrants + the 4 new entrants
    arms, point, boot, elo = elo_boot(np, H, {**field, **new}, fam_of, ANCHOR, BOOT_SEED, N_BOOT)
    p1 = delta_elo(np, arms, point, boot, "h416.SXB", "h415.KFB")
    p1["pass"] = bool(p1["ci95"][0] > 0)
    pairs = [(x, y) for i, x in enumerate(COST_ORDER[::-1]) for y in COST_ORDER[::-1][i + 1:]]
    dts = {f"{x} - {y}": delta_elo(np, arms, point, boot, x, y) for x, y in pairs}
    hr = holm({k: v["p_two_sided"] for k, v in dts.items()})
    for k, v in dts.items():
        v["holm_rejected"] = hr[k]
        v["direction"] = ("a_better" if v["delta"] > 0 else "b_better") if hr[k] else "not_distinguished"
        x, y = v["a"], v["b"]
        v["magnitude"] = pair_stats(np, H, new[x], new[y], fam_of)
    vs_x0 = {e: pair_stats(np, H, new[e], x0, fam_of) for e in NEW}
    top = max(NEW, key=lambda e: (elo[e]["elo"], -COST_ORDER.index(e)))
    distinct = {e: next(v for v in dts.values() if {v["a"], v["b"]} == {top, e})["holm_rejected"]
                for e in NEW if e != top}
    kb = dts["h415.KFB - h410s.BAG3"]
    verdict = "SXB_CONFIRMED" if p1["pass"] else "SXB_NOT_CONFIRMED"
    kfb_vs_bag3 = {"a_better": "KFB_BETTER", "b_better": "BAG3_BETTER"}.get(kb["direction"], "NOT_DISTINGUISHED")
    # (6) descriptive: the grid replicated on P201b; KFB vs BAG3 pooled over P201e + P201b
    grid = {}
    for d in M.DESIGNS:
        vd = {c: vals[d].get(c, x0[c]) for c in cids}  # no usable H program: the X0b value (0 gain)
        grid[d] = {"vs_KFB": pair_stats(np, H, vd, new["h415.KFB"], fam_of),
                   "vs_BAG3": pair_stats(np, H, vd, new["h410s.BAG3"], fam_of),
                   "vs_X0b": pair_stats(np, H, vd, x0, fam_of)}
    pooled = None
    if a.p201e:
        e = json.loads(Path(a.p201e).read_text())
        pooled_k = {**{f"e:{c}": v for c, v in e["KFB"].items()}, **{f"b:{c}": v for c, v in new["h415.KFB"].items()}}
        pooled_b = {**{f"e:{c}": v for c, v in e["BAG3"].items()},
                    **{f"b:{c}": v for c, v in new["h410s.BAG3"].items()}}
        pfam = {**{f"e:{c}": f for c, f in e["family_of"].items()}, **{f"b:{c}": fam_of[c] for c in cids}}
        pooled = pair_stats(np, H, pooled_k, pooled_b, pfam)
        pooled["n_contexts"] = len(pooled_k)
        pooled["n_families"] = len(set(pfam.values()))
    el_int = H.elo_panel({X0B: x0, **new, **{f"{n}.stock": {c: float(stored[n][c]["value"]) for c in cids}
                                             for n in gp}}, fam_of, X0B, seed=BOOT_SEED, n_boot=N_BOOT)
    ws = {}
    for r in list(recs15.values()) + list(recs16.values()):
        for k, v in r.get("worker_s", {}).items():
            ws[k] = ws.get(k, 0.0) + float(v)
    rep_out = {"schema": "jope.h416.train-analysis/v1", "panel": cfg["panel"], "smoke": smoke, "tag": a.tag,
               "n_contexts": len(cids),
               "n_families": len(set(fam_of.values())), "design": design["design"], "checks": checks,
               "invalid": inv, "bag_fallback_contexts": bagfb, "board_elo": elo, "primary": p1,
               "pairwise": dts, "vs_X0b": vs_x0, "verdict": verdict, "kfb_vs_bag3": kfb_vs_bag3,
               "recommendation": {"candidate": top, "board_elo": elo[top], "distinguished_from": distinct,
                                  "beats_X0b_primary": bool(vs_x0[top]["pass"]),
                                  "advice": "SCORE" if vs_x0[top]["pass"] else "HOLD"},
               "grid_replication": grid, "pooled_p201e_p201b_kfb_vs_bag3": pooled, "internal_elo": el_int,
               "worker_s_total": ws}
    Path(a.out).write_text(json.dumps(rep_out, indent=1, sort_keys=True) + "\n")
    print(f"checks: {json.dumps({k: v for k, v in checks.items() if k not in ('board_entrants',)})}")
    print(f"verdict {verdict}; KFB vs BAG3 {kfb_vs_bag3}; recommendation {top} (distinguished from {distinct}); "
          f"beats X0b {vs_x0[top]['pass']} -> {'SCORE' if vs_x0[top]['pass'] else 'HOLD'}")
    print(f"P1 SXB - KFB: dElo {p1['delta']:+.1f} [{p1['ci95'][0]:+.1f}, {p1['ci95'][1]:+.1f}] pass {p1['pass']}")
    for k, v in dts.items():
        m = v["magnitude"]
        print(f"{k:28s} dElo {v['delta']:+7.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] p2 {v['p_two_sided']:.4f} "
              f"holm {int(v['holm_rejected'])} | win {m['win_frac']:.3f} gain {m['gain']:+.4f} clip "
              f"{m['clipped_mean']:+.4f} W/T/L {'/'.join(map(str, m['wtl']))}")
    for e, s in vs_x0.items():
        print(f"{e:12s} vs X0b gain {s['gain']:+.4f} rmse {100 * s['pct_rmse']:+.2f}% clip {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo {s['lofo_min']:+.4f} pass {s['pass']} inv {inv[e]}")
    for k, v in sorted(elo.items(), key=lambda kv: -kv[1]["elo"]):
        print(f"board elo {k:28s} {v['elo']:7.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("--mode", required=True, choices=("dev", "confirm"))
    p.add_argument("--panel", required=True, choices=("P201e", "P201b") + SMOKE_PANELS)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--inner-jobs", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--design")
    p = sub.add_parser("select")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--h415-dir", required=True, help="the stored H415 P201e deploy stage dir (H415_P201E)")
    p.add_argument("--panel-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--refs-out", required=True, help="P201e KFB / BAG3 per-context values (for analyze --p201e)")
    p = sub.add_parser("analyze")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gs-t", required=True, help="stored Phase T stage dir (T_AGG) holding every P201b context")
    p.add_argument("--gp", action="append", required=True)
    p.add_argument("--board-dir", required=True)
    p.add_argument("--panel-file", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--p201e", help="P201e KFB / BAG3 values + families (pooled descriptive)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"deploy": cmd_deploy, "select": cmd_select, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
