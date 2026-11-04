"""H315 (hw_t1030, rd_opus): margin-gated ENS_VOTE2 vs unchanged H303 NESTED, paired, TRAIN-only nested CV.

Per (partition P0/P1/P2, outer fold, root seed 30001-30003) task, inside outer_train only (H303 code via H308's
TRAIN-only loader; outer-test packets guarded until the fit_lock exists):
  inner CV  exactly H303's: for TAU in {0.05,0.10,0.15} and 4 inner family folds, fit the 3 members on inner_train,
            score inner_test (H303) AND inner_train (in-sample, for margin quantiles).
  BASE      H303's 18-config choice, unchanged (highest inner OOF family-equal mean; ties 1e-4 -> fewest switches,
            higher k, higher a, lower TAU). Its outer-test picks must equal the archived H303 (P0) / H308 (P1, P2) picks.
  MARGIN    18 configs x gate m in {0, q50, q75, q90} = 72. Predicted margin of a context = mean over the 3 members of
            (raw ranker score of the ensemble pick - raw score of slot 0); 0 when the ensemble keeps slot 0. Gate:
            keep the switch only if margin >= threshold; m=0 = no gate; q* = quantile of the margins of the switched
            inner_train contexts of that inner fold (in-sample members of that inner fold). Inner objective: family-
            equal mean over the inner-OOF family gains after dropping each inner fold's top-5 family gains. Ties
            (1e-4) -> larger m, then H303's order. Outer: refit on outer_train with the chosen TAU; the threshold is
            the chosen quantile of the switched outer_train margins of the refit members; fit_lock fsynced; outer_test
            scored once (score phase)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = Path(os.environ.get("H315_FS", str(HERE.parent)))
DATA = Path(os.environ.get("H315_DATA", "/home/ubuntu/jope-data/h315-opus"))
GATES = ("0", "q50", "q75", "q90")
GATE_Q = {"q50": 0.50, "q75": 0.75, "q90": 0.90}
GATE_RANK = {"0": 0, "q50": 1, "q75": 2, "q90": 3}
SALTS = {"P1": b"H308-outer-v1\x00", "P2": b"H308-outer-v2\x00"}
PARTS = ("P0", "P1", "P2")
SEEDS = (30001, 30002, 30003)
DROP_INNER = 5
R = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outer_fold(part, family, a302):
    if part == "P0":
        return a302.outer_fold_of_family(family)
    return hashlib.sha256(SALTS[part] + family.encode()).digest()[-1] % 4


def margins(outs, picks):
    """Ensemble predicted margin per context: mean over members of raw[j] - raw[0] for the ensemble pick j (0 if j==0)."""
    raw = np.stack([np.asarray(o["raw_scores"], dtype=np.float64) for o in outs])  # (3, n, K)
    p = np.asarray(picks, dtype=np.int64)
    n = len(p)
    m = raw[:, np.arange(n), p].mean(axis=0) - raw[:, :, 0].mean(axis=0)
    return np.where(p != 0, m, 0.0)


def thresholds(train_picks, train_margins):
    sw = np.asarray(train_margins)[np.asarray(train_picks) != 0]
    out = {"0": None}
    for g, q in GATE_Q.items():
        out[g] = float(np.quantile(sw, q)) if len(sw) else None
    return out


def gate(picks, marg, thr):
    if thr is None:
        return [int(j) for j in picks]
    return [int(j) if (int(j) != 0 and float(mg) >= thr) else 0 for j, mg in zip(picks, marg)]


def drop_top_objective(fold_gains):
    """fold_gains: list over inner folds of arrays of family gains; drop each fold's top DROP_INNER, mean of the rest."""
    kept = []
    for g in fold_gains:
        g = np.sort(np.asarray(g, dtype=np.float64))[::-1]
        kept.extend(g[DROP_INNER:].tolist())
    return float(np.mean(kept)) if kept else 0.0


def family_gains(picks_by_cid, cids, P8, fam_by_cid):
    fams = {}
    for c in cids:
        j = int(picks_by_cid[c])
        fams.setdefault(fam_by_cid[c], []).append(float(P8[c]["utility_k"][j]) - float(P8[c]["utility_k"][0]))
    return np.array([np.mean(v) for v in fams.values()])


def margin_choice_key(x):
    return (-GATE_RANK[x["m"]], x["inner_switches"], -x["k"], -x["a"], x["tau"])


def task(args):
    part, k, rseed = args
    r303 = R["r303"]
    G = r303.G
    import trainonly_loader as tl

    G["fam_to_fold"] = {f: outer_fold(part, f, G["a302"]) for f in G["_fams"]}
    veto, core, bagged, ens, fam_by_cid, train_rows = (
        G[x] for x in ("veto", "core", "bagged", "ens", "fam_by_cid", "train_rows")
    )
    t0 = time.time()
    tdir = DATA / part / f"fold_{k}" / f"root_{rseed}"
    lock_path = tdir / "fit_lock.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())["summary"]
    full = G["P8"]
    test_cids = {r["context_sha256"] for r in train_rows if G["fam_to_fold"][r["family_id"]] == k}
    P8 = tl.GuardedPackets(full, test_cids)
    G["P8"] = P8
    try:
        outer_train = [r for r in train_rows if G["fam_to_fold"][r["family_id"]] != k]
        ot_cids = [r["context_sha256"] for r in outer_train]
        inner_of = {r["family_id"]: r303.inner_fold(k, r["family_id"]) for r in outer_train}
        oof = {(t, kk, a): {} for t in r303.TAU_GRID for kk in r303.K_GRID for a in r303.A_GRID}
        oofm = {(t, kk, a, g): {} for t in r303.TAU_GRID for kk in r303.K_GRID for a in r303.A_GRID for g in GATES}
        thr_log = {}
        for tau in r303.TAU_GRID:
            for i in range(4):
                itrain = [r for r in outer_train if inner_of[r["family_id"]] != i]
                itest = [r for r in outer_train if inner_of[r["family_id"]] == i]
                idir = tdir / f"inner_tau{tau}_fold{i}"
                members = r303.fit_members(idir, k, rseed, itrain, P8, fam_by_cid, tau)
                tc = [r["context_sha256"] for r in itest]
                rc = [r["context_sha256"] for r in itrain]
                outs = r303.score_members(idir, members, np.concatenate([P8[c]["X"] for c in tc]), len(tc), ens)
                outs_tr = r303.score_members(idir, members, np.concatenate([P8[c]["X"] for c in rc]), len(rc), ens)
                for kk in r303.K_GRID:
                    ml = [r303.member_picks(o, tau, kk, veto, core, bagged) for o in outs]
                    ml_tr = [r303.member_picks(o, tau, kk, veto, core, bagged) for o in outs_tr]
                    for a in r303.A_GRID:
                        p_te = r303.ens_picks(ml, a, ens)
                        p_tr = r303.ens_picks(ml_tr, a, ens)
                        for c, j in zip(tc, p_te):
                            oof[(tau, kk, a)][c] = int(j)
                        thr = thresholds(p_tr, margins(outs_tr, p_tr))
                        thr_log[f"{tau}|{kk}|{a}|{i}"] = thr
                        m_te = margins(outs, p_te)
                        for g in GATES:
                            for c, j in zip(tc, gate(p_te, m_te, thr[g])):
                                oofm[(tau, kk, a, g)][c] = int(j)
        # BASE: H303's rule, unchanged
        table = []
        for (tau, kk, a), d in oof.items():
            picks = [d[c] for c in ot_cids]
            table.append(dict(tau=tau, k=kk, a=a, inner_gain=r303.family_gain(picks, ot_cids, P8, fam_by_cid),
                              inner_switches=int(sum(j != 0 for j in picks))))  # fmt: skip
        best = max(x["inner_gain"] for x in table)
        tied = [x for x in table if x["inner_gain"] >= best - r303.TIE]
        base = sorted(tied, key=lambda x: (x["inner_switches"], -x["k"], -x["a"], x["tau"]))[0]
        # MARGIN: drop-top-5-per-inner-fold objective
        folds = {i: [r["context_sha256"] for r in outer_train if inner_of[r["family_id"]] == i] for i in range(4)}
        mtable = []
        for (tau, kk, a, g), d in oofm.items():
            fg = [family_gains(d, folds[i], P8, fam_by_cid) for i in range(4)]
            mtable.append(dict(tau=tau, k=kk, a=a, m=g, inner_obj=drop_top_objective(fg),
                               inner_full=float(np.mean(np.concatenate(fg))),
                               inner_switches=int(sum(d[c] != 0 for c in ot_cids))))  # fmt: skip
        mbest = max(x["inner_obj"] for x in mtable)
        mtied = [x for x in mtable if x["inner_obj"] >= mbest - r303.TIE]
        marg = sorted(mtied, key=margin_choice_key)[0]
        refits = {}
        for tau in sorted({base["tau"], marg["tau"]}):
            refits[str(tau)] = r303.fit_members(tdir / f"refit_tau{tau}", k, rseed, outer_train, P8, fam_by_cid, tau)
        # MARGIN outer threshold: chosen quantile of the switched outer_train margins of the refit members (in-sample)
        mem = refits[str(marg["tau"])]
        outs_ot = r303.score_members(
            tdir / f"refit_tau{marg['tau']}", mem, np.concatenate([P8[c]["X"] for c in ot_cids]), len(ot_cids), ens
        )
        ml_ot = [r303.member_picks(o, marg["tau"], marg["k"], veto, core, bagged) for o in outs_ot]
        p_ot = r303.ens_picks(ml_ot, marg["a"], ens)
        outer_thr = thresholds(p_ot, margins(outs_ot, p_ot))[marg["m"]]
    finally:
        G["P8"] = full
    assert P8.denied == 0
    summary = dict(partition=part, outer_fold=k, root_seed=rseed, base=base, margin=marg, outer_threshold=outer_thr,
                   n_base_tied=len(tied), n_margin_tied=len(mtied), elapsed_sec=round(time.time() - t0, 1))  # fmt: skip
    lock = dict(schema="jope.h315.fit-lock/v1", summary=summary, inner_table_base=table, inner_table_margin=mtable,
                inner_thresholds=thr_log, refit_members=refits, outer_test_packets_accessed_before_lock=0,
                outer_test_packets_built_in_memory_before_lock=len(test_cids), guard_denials=P8.denied)  # fmt: skip
    tmp = lock_path.with_suffix(".new")
    with tmp.open("w") as f:
        json.dump(lock, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, lock_path)
    print(json.dumps(dict(task=[part, k, rseed], base=[base["tau"], base["k"], base["a"]],
                          margin=[marg["tau"], marg["k"], marg["a"], marg["m"]], thr=outer_thr, sec=summary["elapsed_sec"])), flush=True)  # fmt: skip
    return summary


def prepare():
    sys.path.insert(0, str(FS / "h303_ensvote2_fully_nested_hparams"))
    sys.path.insert(1, str(FS / "h308_ensvote2_trainonly_repartition"))
    pins = json.loads((HERE / "manifest.json").read_text())["sources"]
    for rel, digest in pins.items():
        if sha(FS / rel) != digest:
            raise RuntimeError("source pin mismatch: " + rel)
    import run_h303 as r303
    import trainonly_loader as tl

    sel = json.loads(Path("/home/ubuntu/jope-data/h50/input/selected.json").read_text())
    r303.np = tl.TrainOnlyNumpy({r["context_sha256"] for r in sel["train"]})
    r303.SCRATCH = DATA / "setup"
    r303.setup()
    r303.np = np
    r303.G["_fams"] = sorted({r["family_id"] for r in r303.G["train_rows"]})
    assert tl.COUNTERS["dlval_feature_rows_read"] == 0
    R.update(r303=r303, loader_counters=dict(tl.COUNTERS))
    return r303


def phase_fit(workers):
    prepare()
    tasks = [(p, k, s) for p in PARTS for k in range(4) for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(task, tasks))
    (DATA / "fit_done.json").write_text(
        json.dumps(dict(tasks=res, sec=round(time.time() - t0, 1), loader=R["loader_counters"]), indent=1)
    )


def score_part(r303, part):
    G = r303.G
    veto, core, bagged, ens, P8 = (G[x] for x in ("veto", "core", "bagged", "ens", "P8"))
    G["fam_to_fold"] = {f: outer_fold(part, f, G["a302"]) for f in G["_fams"]}
    cids = [r["context_sha256"] for r in G["train_rows"]]
    n = len(cids)
    out = {"BASE": np.full((3, n), -1, np.int32), "MARGIN": np.full((3, n), -1, np.int32)}
    locks = {}
    for k in range(4):
        test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
        X = np.concatenate([P8[cids[i]]["X"] for i in test])
        for s, rseed in enumerate(SEEDS):
            tdir = DATA / part / f"fold_{k}" / f"root_{rseed}"
            lp = tdir / "fit_lock.json"
            lock = json.loads(lp.read_text())
            locks[f"{k}:{rseed}"] = sha(lp)
            sm = lock["summary"]
            for arm, ch in (("BASE", sm["base"]), ("MARGIN", sm["margin"])):
                outs = r303.score_members(
                    tdir / f"refit_tau{ch['tau']}", lock["refit_members"][str(ch["tau"])], X, len(test), ens
                )
                ml = [r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
                p = r303.ens_picks(ml, ch["a"], ens)
                if arm == "MARGIN":
                    p = gate(p, margins(outs, p), sm["outer_threshold"] if ch["m"] != "0" else None)
                out[arm][s, test] = p
    assert all((v >= 0).all() for v in out.values())
    return out, locks


def phase_score():
    import evaluate_h315

    r303 = prepare()
    evaluate_h315.main(r303, DATA, {p: score_part(r303, p) for p in PARTS})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("fit", "score"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H315 requires CPU xbabe2 at nice 19")
    phase_fit(a.workers) if a.phase == "fit" else phase_score()
