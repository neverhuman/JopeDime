"""H323 (hw_t1500, rd_opus): snapshot (SWA / EMA) averaging of ENS_VOTE2 member predictions vs unchanged H303 NESTED.

Per (partition P0/P1/P2, outer fold, root seed 30001-30003) task, inside outer_train only (H303 code via H308's
TRAIN-only loader; outer-test packets guarded until the fsynced fit_lock exists). The inner members are refitted with
H303's code and seeds (archived locks lack inner-member metadata) and scored three ways from the SAME fitted boosters:
  BASE  H303's member score: ranker and the 8 bag rankers predicted at the selected n_est (num_iteration = n_est).
  SWA   member score = uniform mean of the predictions at num_iteration = max(1, round(p * n_est)),
        p in (0.6, 0.7, 0.8, 0.9, 1.0); applied to the ranker raw scores and to each bag ranker's scores.
  EMA   same snapshots, weights 0.7**(4-j) for j = 0..4 (oldest -> newest), normalized.
  The veto gate (p_invalid) is the unchanged full model in every arm. Masking, top-4 margin selection, bag votes,
  BAG_SWITCH, the ENS_VOTE2 agreement, the 18-config grid, inner folds and H303's selection/tie rule are identical in
  every arm; each arm makes its own inner config choice, is refitted on outer_train with its chosen TAU, and its outer
  test is scored once with its own member score."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = Path(os.environ.get("H323_FS", str(HERE.parent)))
DATA = Path(os.environ.get("H323_DATA", "/home/ubuntu/jope-data/h323-opus"))
SALTS = {"P1": b"H308-outer-v1\x00", "P2": b"H308-outer-v2\x00"}
PARTS = ("P0", "P1", "P2")
SEEDS = (30001, 30002, 30003)
SNAP_P = (0.6, 0.7, 0.8, 0.9, 1.0)
W_SWA = tuple([0.2] * 5)
_ema = [0.7 ** (4 - j) for j in range(5)]
W_EMA = tuple(w / sum(_ema) for w in _ema)
ARMS = {"BASE": None, "SWA": W_SWA, "EMA": W_EMA}
R = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outer_fold(part, family, a302):
    if part == "P0":
        return a302.outer_fold_of_family(family)
    return hashlib.sha256(SALTS[part] + family.encode()).digest()[-1] % 4


def snap_iters(n_est):
    return [max(1, int(round(p * n_est))) for p in SNAP_P]


def snap_predict(booster, X, n_est, weights):
    return sum(
        w * np.asarray(booster.predict(X, num_iteration=it), dtype=np.float64)
        for w, it in zip(weights, snap_iters(n_est))
    )


def score_member_snap(fold_dir, cell, X, n_ctx, ens, weights):
    """H287 ens.score_single_member_from_disk with snapshot-averaged ranker and bag scores (same files, sha-checked)."""
    import lightgbm as lgb

    G = R["r303"].G
    veto, core, bagged = G["veto"], G["core"], G["bagged"]
    fold_dir = Path(fold_dir)
    X2 = np.asarray(X, dtype=np.float32).reshape(n_ctx * ens.K, 568)
    for part in ("ranker", "veto"):
        if sha(fold_dir / cell[part]["file"]) != cell[part]["sha256"]:
            raise RuntimeError("member sha256 mismatch")
    n_est = int(cell["selected_n_estimators"])
    margin = math.inf if cell["strict_margin"] == "inf" else float(cell["strict_margin"])
    tau = float(cell.get("selected_tau", veto.TAU))
    q_fixed = int(cell.get("q_fixed", bagged.Q_FIXED))
    gate = lgb.Booster(model_file=str(fold_dir / cell["veto"]["file"]))
    ranker = lgb.Booster(model_file=str(fold_dir / cell["ranker"]["file"]))
    p_inv = np.asarray(gate.predict(X2), dtype=np.float64).reshape(n_ctx, ens.K)
    raw = snap_predict(ranker, X2, n_est, weights).reshape(n_ctx, ens.K)
    masked = veto.mask(raw, p_inv, tau=tau)
    j_ref = [core.select_from_top4_scores(masked[i], margin) for i in range(n_ctx)]
    bags = []
    for b in cell["bags"]:
        if sha(fold_dir / b["file"]) != b["sha256"]:
            raise RuntimeError("bag sha256 mismatch")
        bm = lgb.Booster(model_file=str(fold_dir / b["file"]))
        bags.append(snap_predict(bm, X2, n_est, weights).reshape(n_ctx, ens.K))
    bag_scores = np.stack(bags)
    votes = bagged.vote_counts(bag_scores, j_ref, margin)
    return dict(p_inv=p_inv, raw_scores=raw, bag_scores=bag_scores, j_ref=j_ref, votes=votes,
                picks_bag=bagged.switch_by_votes(j_ref, votes, q_fixed), _strict_margin=margin)  # fmt: skip


def score_arm(arm, fold_dir, members, X, n, ens):
    if ARMS[arm] is None:
        return R["r303"].score_members(fold_dir, members, X, n, ens)
    return [score_member_snap(fold_dir, c, X, n, ens, ARMS[arm]) for c in members]


def choose(r303, oof, ot_cids, P8, fam_by_cid):
    table = []
    for (tau, kk, a), d in oof.items():
        picks = [d[c] for c in ot_cids]
        table.append(
            dict(
                tau=tau,
                k=kk,
                a=a,
                inner_gain=r303.family_gain(picks, ot_cids, P8, fam_by_cid),
                inner_switches=int(sum(j != 0 for j in picks)),
            )
        )
    best = max(x["inner_gain"] for x in table)
    tied = [x for x in table if x["inner_gain"] >= best - r303.TIE]
    return sorted(tied, key=lambda x: (x["inner_switches"], -x["k"], -x["a"], x["tau"]))[0], table, len(tied)


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
        oof = {arm: {(t, kk, a): {} for t in r303.TAU_GRID for kk in r303.K_GRID for a in r303.A_GRID} for arm in ARMS}
        for tau in r303.TAU_GRID:
            for i in range(4):
                itrain = [r for r in outer_train if inner_of[r["family_id"]] != i]
                itest = [r for r in outer_train if inner_of[r["family_id"]] == i]
                idir = tdir / f"inner_tau{tau}_fold{i}"
                members = r303.fit_members(idir, k, rseed, itrain, P8, fam_by_cid, tau)
                tc = [r["context_sha256"] for r in itest]
                X = np.concatenate([P8[c]["X"] for c in tc])
                for arm in ARMS:
                    outs = score_arm(arm, idir, members, X, len(tc), ens)
                    for kk in r303.K_GRID:
                        ml = [r303.member_picks(o, tau, kk, veto, core, bagged) for o in outs]
                        for a in r303.A_GRID:
                            for c, j in zip(tc, r303.ens_picks(ml, a, ens)):
                                oof[arm][(tau, kk, a)][c] = int(j)
        chosen, tables, ntied = {}, {}, {}
        for arm in ARMS:
            chosen[arm], tables[arm], ntied[arm] = choose(r303, oof[arm], ot_cids, P8, fam_by_cid)
        refits = {}
        for tau in sorted({chosen[a]["tau"] for a in ARMS}):
            refits[str(tau)] = r303.fit_members(tdir / f"refit_tau{tau}", k, rseed, outer_train, P8, fam_by_cid, tau)
    finally:
        G["P8"] = full
    assert P8.denied == 0
    summary = dict(partition=part, outer_fold=k, root_seed=rseed, chosen=chosen, n_tied=ntied,
                   same_as_base={a: all(chosen[a][x] == chosen["BASE"][x] for x in ("tau", "k", "a")) for a in ("SWA", "EMA")},
                   elapsed_sec=round(time.time() - t0, 1))  # fmt: skip
    lock = dict(schema="jope.h323.fit-lock/v1", summary=summary, inner_tables=tables, refit_members=refits,
                outer_test_packets_accessed_before_lock=0, outer_test_packets_built_in_memory_before_lock=len(test_cids),
                guard_denials=P8.denied)  # fmt: skip
    tmp = lock_path.with_suffix(".new")
    with tmp.open("w") as f:
        json.dump(lock, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, lock_path)
    print(
        json.dumps(
            dict(
                task=[part, k, rseed],
                chosen={a: [c["tau"], c["k"], c["a"]] for a, c in chosen.items()},
                sec=summary["elapsed_sec"],
            )
        ),
        flush=True,
    )
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
    import evaluate_h323

    evaluate_h323.regress(prepare(), DATA)


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
    out = {arm: np.full((3, len(cids)), -1, np.int32) for arm in ARMS}
    locks, same = {}, {"SWA": [], "EMA": []}
    for k in range(4):
        test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
        X = np.concatenate([P8[cids[i]]["X"] for i in test])
        for s, rseed in enumerate(SEEDS):
            tdir = DATA / part / f"fold_{k}" / f"root_{rseed}"
            lp = tdir / "fit_lock.json"
            lock = json.loads(lp.read_text())
            locks[f"{k}:{rseed}"] = sha(lp)
            sm = lock["summary"]
            for a in same:
                same[a].append(sm["same_as_base"][a])
            for arm in ARMS:
                ch = sm["chosen"][arm]
                d = tdir / f"refit_tau{ch['tau']}"
                outs = score_arm(arm, d, lock["refit_members"][str(ch["tau"])], X, len(test), ens)
                ml = [r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
                out[arm][s, test] = r303.ens_picks(ml, ch["a"], ens)
    assert all((v >= 0).all() for v in out.values())
    return out, locks, same


def phase_score():
    import evaluate_h323

    r303 = prepare()
    evaluate_h323.main(r303, DATA, {p: score_part(r303, p) for p in PARTS})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("regress", "fit", "score"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H323 requires CPU xbabe2 at nice 19")
    {"regress": phase_regress, "score": phase_score}.get(a.phase, lambda: phase_fit(a.workers))()
