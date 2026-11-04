"""H310 score + evaluation (runs only after the 3 DL-Train locks and the prereg push).

PRIMARY: each seed's frozen ENS_VOTE2 chooser picks once among r3175's GPU-exact top-8 dev91 pool; slot 0 = r3175's own
pick (regression (a)). Gain = utility(pick) - utility(r3175 slot 0), flywheel utility scale (gate ? -fitness : -2.0).
SECONDARY: the same picks vs r882 slot 0 (r882 Native91 per-context, regression (b)).
DESCRIPTIVE: the chooser on r882's dev91 top-8 pool vs r3175 slot 0; random-valid-slot null on the r3175 pool
(RNG 310150); clipped-at--1 sensitivity of the primary.
Statistics: analysis_h302.summarize_three_seeds via evaluate_h303.judge (whole-family bootstrap RNG 302150, 10,000
draws, MM Bradley-Terry Elo), concentration (drop top 1/5/10, drop-top-5 LCB95).

Decision (pre-registered; ties never win):
  DEV91_TRANSFERRED iff median over seeds of the family-equal gain vs r3175 > +0.001 AND pooled one-sided LCB95 > 0 AND
  per-seed one-sided LCB95 > 0 on >= 2/3 seeds AND median over seeds of the drop-top-5 gain > 0 AND harmed share
  (loss > 0.01) <= 35% each seed AND invalid excess vs r3175 slot 0 <= 1 each seed.
  DEV91_UNDERPOWERED iff not transferred AND pooled 95% CI half-width > 0.03 AND median gain > 0.
  Else DEV91_NOT_TRANSFERRED. No verdict authorizes serving or touches the flywheel."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = HERE.parent
ELO_LABEL = "reference = r3175 slot0 (dev91)"
NULL_RNG = 310150
CACHE = Path("/home/ubuntu/jope-data/h58/cache")


def install_score_guard():
    bad = ("/sealed", "/holdout", "meta_test", "meta-test", "dl-test", "dl_test", "fresh_")

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            s = os.fsdecode(args[0]).lower()
            if any(b in s for b in bad) or s.endswith(".pt"):
                raise RuntimeError("H310 score phase: forbidden open " + s)

    sys.addaudithook(hook)


def invalid(packet, j):
    return (not bool(packet["gate_k"][j])) or float(packet["utility_k"][j]) <= -2.0 + 1e-9


def decide(judge):
    s = judge["summary"]
    lcb = s["per_seed_lcb_one_sided"]["0.05"]
    d5 = [c["gain_drop_top5"] for c in judge["concentration"]]
    lo, hi = s["pooled_ci95"]
    crit = dict(
        c1_median_gain_gt_0_001=bool(s["median_gain_across_seeds"] > 0.001),
        c2_pooled_lcb95_gt_0=bool(s["pooled_lcb_one_sided_95"] > 0),
        c3_per_seed_lcb95_gt_0_on_2of3=bool(sum(x > 0 for x in lcb) >= 2),
        c4_median_drop_top5_gain_gt_0=bool(float(np.median(d5)) > 0),
        c5_harmed_le_35pct_each=bool(all(h <= 0.35 for h in s["harmed_share_by_seed"])),
        c6_invalid_excess_le_1_each=bool(all(x["excess"] <= 1 for x in judge["invalid_detail"])),
    )
    half = (hi - lo) / 2.0
    if all(crit.values()):
        verdict = "DEV91_TRANSFERRED"
    elif half > 0.03 and s["median_gain_across_seeds"] > 0:
        verdict = "DEV91_UNDERPOWERED"
    else:
        verdict = "DEV91_NOT_TRANSFERRED"
    return crit, verdict, dict(pooled_ci95_half_width=half, median_drop_top5_gain=float(np.median(d5)))


def family_d(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return np.stack([np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams]) for s in range(3)])


def power_note(u, families):
    D = family_d(u, families)
    se = [float(d.std(ddof=1) / np.sqrt(len(d))) for d in D]
    return dict(
        contexts=int(u.shape[1]),
        families=int(D.shape[1]),
        per_seed_family_gain_se=se,
        mde_one_sided_alpha05_power80=float(2.486 * np.median(se)),
        note="MDE = (z_.95 + z_.80) x median per-seed family-equal SE; post hoc from the observed family dispersion",
    )


def main(DATA, lock, reg, prereg_commit):
    install_score_guard()
    sys.path.insert(0, str(FS / "h302_ensvote2_trainseed_nested_cv"))
    sys.path.insert(1, str(FS / "h303_ensvote2_fully_nested_hparams"))
    import analysis_h302 as a302
    import evaluate_h303 as ev
    import run_h302 as r302
    import run_h303 as r303

    import run_h310 as R

    core, topk, veto, bagged, ens = r302.load_pinned_h287_modules(DATA / "score_mods")
    core.INPUT_POOLS_DIR = r302.POOLS
    sel = json.loads((r302.POOLS / "selected.json").read_bytes())
    assert hashlib.sha256((r302.POOLS / "selected.json").read_bytes()).hexdigest() == r302.SELECTED_JSON_SHA256
    dev = sel["dev"]
    cids = [r["context_sha256"] for r in dev]
    fams = [r["family_id"] for r in dev]
    assert len(cids) == 91
    ids = set(cids)
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        pc, off, base, proj = [str(c) for c in zp["cids"]], zp["offsets"], zp["base_prior_cost"], zp["proj16"]
        for i, c in enumerate(pc):
            if c in ids:
                aprior[c] = base[int(off[i]) : int(off[i + 1])].astype(np.float32).copy()
                proj16[c] = proj[int(off[i]) : int(off[i + 1])].astype(np.float32).copy()
    with np.load(CACHE / "context_features.npz", allow_pickle=False) as zc:
        lm, dp = zc["lm48_z"], zc["dope629"]
        for i, c in enumerate(str(x) for x in zc["cids"]):
            if c in ids:
                ctxf[c] = {"lm48_z": lm[i].astype(np.float32).copy(), "dope629": dp[i].astype(np.float32).copy()}
    with np.load(R.R3175_GPU_COSTS) as z:
        c3175 = {c: z[c].astype(np.float32) for c in cids}
    with np.load(R.R882_DEV_COSTS) as z:
        c882 = {c: z[c].astype(np.float32) for c in cids}
    P3 = topk.load_packets(dev, c3175, aprior, proj16, ctxf, ens.K)
    P8 = topk.load_packets(dev, c882, aprior, proj16, ctxf, ens.K)

    # Regressions (a) r3175 and (b) r882 -- after the lock, before any challenger pick.
    stored = json.loads(R.R3175_STORED.read_text())["dev_utility"]
    r3175_u = np.array([float(P3[c]["utility_k"][0]) for c in cids])
    eq_a = int(sum(float(a) == float(b) for a, b in zip(r3175_u, stored["per_context"])))
    ref = json.loads(R.R882_REF.read_text())
    ref_map = dict(zip([c["cid"] for c in ref["evaluation_contract"]["contexts"]], ref["utility"]["per_context"]))
    r882_u = np.array([float(P8[c]["utility_k"][0]) for c in cids])
    eq_b = int(sum(float(P8[c]["utility_k"][0]) == float(ref_map[c]) for c in cids))
    regression = dict(
        a_r3175=dict(per_context_equal=eq_a, mean=float(r3175_u.mean()), stored_mean=float(stored["mean"]),
                     expected=R.R3175_MEAN, ok=bool(eq_a == 91 and abs(r3175_u.mean() - R.R3175_MEAN) <= 1e-12)),
        b_r882=dict(per_context_equal=eq_b, mean=float(r882_u.mean()), expected=R.R882_MEAN,
                    ok=bool(eq_b == 91 and abs(r882_u.mean() - R.R882_MEAN) <= 1e-12)),
        c_train=dict(status=reg["status"], fold0_picks_sha256=reg["fold0_seed30001_picks_sha256_fresh"]),
    )  # fmt: skip
    out = DATA / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not (regression["a_r3175"]["ok"] and regression["b_r882"]["ok"]):
        (out / "REGRESSION_FAIL.json").write_text(json.dumps(regression, indent=1))
        raise RuntimeError("REGRESSION_FAIL: " + json.dumps(regression))

    # One-shot scoring of the 3 frozen choosers.
    X3 = np.concatenate([P3[c]["X"] for c in cids])
    X8 = np.concatenate([P8[c]["X"] for c in cids])
    picks3 = np.zeros((3, 91), dtype=np.int32)
    picks8 = np.zeros((3, 91), dtype=np.int32)
    chosen = {}
    for s, rs in enumerate((30001, 30002, 30003)):
        lp = DATA / "lock" / "fold_ALL" / f"root_{rs}" / "fit_lock.json"
        lk = json.loads(lp.read_text())
        ch = lk["chosen"]
        chosen[str(rs)] = ch
        for X, dst in ((X3, picks3), (X8, picks8)):
            outs = r303.score_members(lp.parent / "refit", lk["members"], X, 91, ens)
            ml = [r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
            dst[s] = r303.ens_picks(ml, ch["a"], ens)

    def mats(pool, picks, base_u, base_inv):
        u = np.zeros((3, 91, 2))
        inv = np.zeros((3, 91, 2), dtype=bool)
        for s in range(3):
            for i, c in enumerate(cids):
                j = int(picks[s, i])
                u[s, i] = [base_u[i], float(pool[c]["utility_k"][j])]
                inv[s, i] = [base_inv[i], invalid(pool[c], j)]
        return u, inv

    inv3 = [invalid(P3[c], 0) for c in cids]
    inv8 = [invalid(P8[c], 0) for c in cids]
    u_p, i_p = mats(P3, picks3, r3175_u, inv3)
    primary = ev.judge(a302, u_p, i_p, fams, picks3)
    crit, verdict, aux = decide(primary)
    u_s, i_s = mats(P3, picks3, r882_u, inv8)
    u_d, i_d = mats(P8, picks8, r3175_u, inv3)
    u_c = np.maximum(u_p, -1.0)
    rng = np.random.default_rng(NULL_RNG)
    nul = np.zeros((3, 91), dtype=np.int32)
    for s in range(3):
        for i, c in enumerate(cids):
            v = [j for j in range(1, len(P3[c]["gate_k"])) if bool(P3[c]["gate_k"][j])]
            nul[s, i] = int(rng.choice(v)) if v else 0
    u_n, i_n = mats(P3, nul, r3175_u, inv3)

    def pub(j):
        return dict(j, summary={k: v for k, v in j["summary"].items() if k != "harmed_families_by_seed"})

    full = dict(
        wave="H310",
        verdict=verdict,
        criteria=crit,
        criteria_aux=aux,
        elo_label=ELO_LABEL,
        prereg_commit=prereg_commit,
        regression=regression,
        lock=dict(locks=lock["locks"], dev91_rows_read_before_lock=lock["dev91_rows_read_before_lock"],
                  counters=lock["counters"]),
        chosen_configs=chosen,
        picks_sha256={str(rs): hashlib.sha256(picks3[s].tobytes()).hexdigest() for s, rs in enumerate((30001, 30002, 30003))},
        switches_r3175_pool=[int((p != 0).sum()) for p in picks3],
        seed_pick_agreement=dict(p01=float(np.mean(picks3[0] == picks3[1])), p02=float(np.mean(picks3[0] == picks3[2])),
                                 p12=float(np.mean(picks3[1] == picks3[2]))),
        power=power_note(u_p, fams),
        primary_vs_r3175_slot0=pub(primary),
        secondary_vs_r882_slot0=pub(ev.judge(a302, u_s, i_s, fams, picks3)),
        descriptive_r882_pool_vs_r3175=pub(ev.judge(a302, u_d, i_d, fams, picks8)),
        sensitivity_clipped_at_minus1=pub(ev.judge(a302, u_c, i_p, fams, picks3)),
        null_random_valid_slot=pub(ev.judge(a302, u_n, i_n, fams, nul)),
        r3175_dev91_mean=float(r3175_u.mean()),
        r882_dev91_mean=float(r882_u.mean()),
        oracle_top8_r3175_pool_mean=float(np.mean([P3[c]["oracle_top8_u"] for c in cids])),
    )  # fmt: skip
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(out / "picks.npz", r3175_pool=picks3, r882_pool=picks8, null=nul)
    s = primary["summary"]
    print(json.dumps(dict(verdict=verdict, criteria=crit, aux=aux, regression={k: v["ok"] if "ok" in v else v for k, v in regression.items()},
                          gains=s["per_seed_family_mean_gain"], median=s["median_gain_across_seeds"], pooled=s["pooled_ci95"],
                          lcb=s["per_seed_lcb_one_sided"]["0.05"], chosen=chosen)), flush=True)  # fmt: skip
