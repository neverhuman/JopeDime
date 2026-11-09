"""H320 (hw_t1300, rd_opus): trimmed inner objective for the ENS_VOTE2 config choice vs unchanged H303 NESTED.

Per (partition P0/P1/P2, outer fold, root seed 30001-30003) task, inside outer_train only (H303 code via H308's
TRAIN-only loader; outer-test packets guarded until the fsynced fit_lock exists):
  inner CV  exactly H303's: for TAU in {0.05,0.10,0.15} and 4 inner family folds, the 3 members (H303 member seeds) are
            fitted on inner_train and score inner_test; k in {5,6,7}, a in {2,3} at aggregation (18 configs).
  BASE      H303's choice, unchanged (in-run copy; its outer picks must equal the archived H303/H308 picks).
  TRIM      same 18 configs and inner predictions; objective per config:
              for each inner fold i: g_c = utility(pick_c) - utility(slot0) for the inner_test contexts c of fold i;
                q = 95th percentile (numpy.quantile, linear) of |g| over those contexts; g'_c = clip(g_c, -q, q);
                family means of g' over the fold's families; drop the 10 largest family means; mean of the rest;
              objective = mean over the 4 inner folds.
            argmax; ties within 1e-4 -> fewer inner switches, then higher k, higher a, lower TAU.
  Outer     refit on outer_train with each chosen TAU (BASE and TRIM), fit_lock fsynced; outer_test scored once."""

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
FS = Path(os.environ.get("H320_FS", str(HERE.parent)))
DATA = Path(os.environ.get("H320_DATA", "/home/ubuntu/jope-data/h320-opus"))
SALTS = {"P1": b"H308-outer-v1\x00", "P2": b"H308-outer-v2\x00"}
PARTS = ("P0", "P1", "P2")
SEEDS = (30001, 30002, 30003)
CLIP_Q = 0.95
DROP_TOP = 10
R = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outer_fold(part, family, a302):
    if part == "P0":
        return a302.outer_fold_of_family(family)
    return hashlib.sha256(SALTS[part] + family.encode()).digest()[-1] % 4


def trimmed_fold_objective(gains, families):
    """One inner fold: clip context gains at the q95 of |gain|, family-equal means, drop the top 10, mean the rest."""
    g = np.asarray(gains, dtype=np.float64)
    q = float(np.quantile(np.abs(g), CLIP_Q)) if len(g) else 0.0
    g = np.clip(g, -q, q)
    fam = {}
    for x, f in zip(g, families):
        fam.setdefault(f, []).append(x)
    means = np.sort(np.array([np.mean(v) for v in fam.values()]))[::-1]
    rest = means[DROP_TOP:]
    return float(rest.mean()) if len(rest) else 0.0


def trim_key(x):
    return (x["inner_switches"], -x["k"], -x["a"], x["tau"])


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
        for tau in r303.TAU_GRID:
            for i in range(4):
                itrain = [r for r in outer_train if inner_of[r["family_id"]] != i]
                itest = [r for r in outer_train if inner_of[r["family_id"]] == i]
                idir = tdir / f"inner_tau{tau}_fold{i}"
                members = r303.fit_members(idir, k, rseed, itrain, P8, fam_by_cid, tau)
                tc = [r["context_sha256"] for r in itest]
                outs = r303.score_members(idir, members, np.concatenate([P8[c]["X"] for c in tc]), len(tc), ens)
                for kk in r303.K_GRID:
                    ml = [r303.member_picks(o, tau, kk, veto, core, bagged) for o in outs]
                    for a in r303.A_GRID:
                        for c, j in zip(tc, r303.ens_picks(ml, a, ens)):
                            oof[(tau, kk, a)][c] = int(j)
        table, ttable = [], []
        folds = {i: [r["context_sha256"] for r in outer_train if inner_of[r["family_id"]] == i] for i in range(4)}
        for (tau, kk, a), d in oof.items():
            picks = [d[c] for c in ot_cids]
            sw = int(sum(j != 0 for j in picks))
            table.append(
                dict(tau=tau, k=kk, a=a, inner_gain=r303.family_gain(picks, ot_cids, P8, fam_by_cid), inner_switches=sw)
            )
            per_fold = []
            for i in range(4):
                cs = folds[i]
                g = [float(P8[c]["utility_k"][d[c]]) - float(P8[c]["utility_k"][0]) for c in cs]
                per_fold.append(trimmed_fold_objective(g, [fam_by_cid[c] for c in cs]))
            ttable.append(
                dict(
                    tau=tau,
                    k=kk,
                    a=a,
                    inner_obj=float(np.mean(per_fold)),
                    inner_obj_by_fold=per_fold,
                    inner_switches=sw,
                )
            )
        best = max(x["inner_gain"] for x in table)
        tied = [x for x in table if x["inner_gain"] >= best - r303.TIE]
        base = sorted(tied, key=lambda x: (x["inner_switches"], -x["k"], -x["a"], x["tau"]))[0]
        tbest = max(x["inner_obj"] for x in ttable)
        ttied = [x for x in ttable if x["inner_obj"] >= tbest - r303.TIE]
        trim = sorted(ttied, key=trim_key)[0]
        refits = {}
        for tau in sorted({base["tau"], trim["tau"]}):
            refits[str(tau)] = r303.fit_members(tdir / f"refit_tau{tau}", k, rseed, outer_train, P8, fam_by_cid, tau)
    finally:
        G["P8"] = full
    assert P8.denied == 0
    same = base["tau"] == trim["tau"] and base["k"] == trim["k"] and base["a"] == trim["a"]
    summary = dict(partition=part, outer_fold=k, root_seed=rseed, base=base, trim={k_: v for k_, v in trim.items() if k_ != "inner_obj_by_fold"},
                   trim_same_as_base=same, n_base_tied=len(tied), n_trim_tied=len(ttied), elapsed_sec=round(time.time() - t0, 1))  # fmt: skip
    lock = dict(schema="jope.h320.fit-lock/v1", summary=summary, inner_table_base=table, inner_table_trim=ttable, refit_members=refits,
                outer_test_packets_accessed_before_lock=0, outer_test_packets_built_in_memory_before_lock=len(test_cids),
                guard_denials=P8.denied)  # fmt: skip
    tmp = lock_path.with_suffix(".new")
    with tmp.open("w") as f:
        json.dump(lock, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, lock_path)
    print(json.dumps(dict(task=[part, k, rseed], base=[base["tau"], base["k"], base["a"]], trim=[trim["tau"], trim["k"], trim["a"]],
                          same=same, sec=summary["elapsed_sec"])), flush=True)  # fmt: skip
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


def phase_regress():
    import evaluate_h320

    evaluate_h320.regress(prepare(), DATA)


def phase_fit(workers):
    prepare()
    assert json.loads((DATA / "regression_receipt.json").read_text())["status"] == "PASS"
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
    out = {"BASE": np.full((3, len(cids)), -1, np.int32), "TRIM": np.full((3, len(cids)), -1, np.int32)}
    locks, same = {}, []
    for k in range(4):
        test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
        X = np.concatenate([P8[cids[i]]["X"] for i in test])
        for s, rseed in enumerate(SEEDS):
            tdir = DATA / part / f"fold_{k}" / f"root_{rseed}"
            lp = tdir / "fit_lock.json"
            lock = json.loads(lp.read_text())
            locks[f"{k}:{rseed}"] = sha(lp)
            sm = lock["summary"]
            same.append(sm["trim_same_as_base"])
            for arm, ch in (("BASE", sm["base"]), ("TRIM", sm["trim"])):
                outs = r303.score_members(
                    tdir / f"refit_tau{ch['tau']}", lock["refit_members"][str(ch["tau"])], X, len(test), ens
                )
                ml = [r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
                out[arm][s, test] = r303.ens_picks(ml, ch["a"], ens)
    assert all((v >= 0).all() for v in out.values())
    return out, locks, same


def phase_score():
    import evaluate_h320

    r303 = prepare()
    evaluate_h320.main(r303, DATA, {p: score_part(r303, p) for p in PARTS})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("regress", "fit", "score"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H320 requires CPU xbabe2 at nice 19")
    {"regress": phase_regress, "score": phase_score}.get(a.phase, lambda: phase_fit(a.workers))()
