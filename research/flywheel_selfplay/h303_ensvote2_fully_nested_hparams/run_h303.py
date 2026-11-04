"""H303 (hw_t0535, rd_opus): ENS_VOTE2 with FULLY NESTED hyperparameters, plus a concentration audit.

H302 imported bagged.Q_FIXED (= the member vote k), veto.TAU and the 2-of-3 agreement as constants tuned on all of DL-Train
(H287-H294). H303 chooses them INSIDE each H302 outer_train only:
  grid  TAU in {0.05, 0.10, 0.15}  x  member vote k (= Q_FIXED) in {5, 6, 7} of 8  x  agreement a in {2, 3} of 3  (18 configs);
  inner CV  4 family-grouped folds of the outer_train families (fold = SHA256("H303-inner-v1" NUL str(outer_fold) NUL family)
        last byte mod 4); for each TAU the 3 members of the root seed are FIT on each inner_train with TAU in force (H287's
        veto.mask default and veto.TAU are set to that TAU, so TAU also governs each member's own inner lr/n_est/margin
        selection); k and a act at aggregation only;
  choice  the config with the highest inner out-of-fold family-equal mean gain over the outer_train families; configs
        within 1e-4 of the best are tied and resolved to the MORE conservative one (fewest inner switches, then higher k,
        then higher a, then lower TAU);
  refit  the 3 members on the full outer_train with the chosen TAU (with TAU 0.10 this is exactly H302's member fit);
        a fit_lock is fsynced; only then is outer_test scored, once, with the chosen (TAU, k, a).
Outer-test rows never enter any choice. Everything else is H302's code, pins and seeds (imported unchanged)."""

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
H302 = HERE.parent / "h302_ensvote2_trainseed_nested_cv"
SCRATCH = Path(os.environ.get("H303_SCRATCH", "/home/ubuntu/jope-data/h303/scratch"))
TAU_GRID = (0.05, 0.10, 0.15)
K_GRID = (5, 6, 7)
A_GRID = (2, 3)
TIE = 1e-4
INNER_SALT = b"H303-inner-v1\x00"
H302_PINS = {
    "run_h302.py": None,  # filled from manifest.json at run time
    "analysis_h302.py": None,
}
G = {}  # per-process globals (set in the parent before fork)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_fold(outer_fold, family):
    return hashlib.sha256(INNER_SALT + str(outer_fold).encode() + b"\x00" + family.encode()).digest()[-1] % 4


def set_tau(veto, tau):
    """Put TAU in force for H287's veto module: the module constant and the mask() default."""
    veto.TAU = float(tau)
    d = list(veto.mask.__defaults__)
    d[-1] = float(tau)
    veto.mask.__defaults__ = tuple(d)


def member_picks(out, tau, k, veto, core, bagged):
    """One frozen member's BAG_SWITCH picks for (tau, k), recomputed from its raw outputs."""
    margin = out["_strict_margin"]
    masked = veto.mask(np.asarray(out["raw_scores"]), np.asarray(out["p_inv"]), tau=tau)
    n = masked.shape[0]
    j_ref = [core.select_from_top4_scores(masked[i], margin) for i in range(n)]
    votes = bagged.vote_counts(np.asarray(out["bag_scores"]), j_ref, margin)
    return bagged.switch_by_votes(j_ref, votes, k)


def ens_picks(member_lists, a, ens):
    return [ens.vote_consensus_pick([m[i] for m in member_lists], a) for i in range(len(member_lists[0]))]


def score_members(fold_dir, members, X, n, ens):
    outs = []
    for c in members:
        o = ens.score_single_member_from_disk(fold_dir, c, X, n, ens.K)
        o["_strict_margin"] = math.inf if c["strict_margin"] == "inf" else float(c["strict_margin"])
        outs.append(o)
    return outs


def family_gain(picks, cids, P8, fam_by_cid):
    fams = {}
    for c, j in zip(cids, picks):
        fams.setdefault(fam_by_cid[c], []).append(float(P8[c]["utility_k"][int(j)]) - float(P8[c]["utility_k"][0]))
    return float(np.mean([np.mean(v) for v in fams.values()]))


def fit_members(fold_dir, outer_fold, rseed, rows, P8, fam_by_cid, tau):
    r302, veto, core, bagged, ens = G["r302"], G["veto"], G["core"], G["bagged"], G["ens"]
    set_tau(veto, tau)
    cids = [r["context_sha256"] for r in rows]
    packets = {c: P8[c] for c in cids}
    fbc = {c: fam_by_cid[c] for c in cids}
    check = sorted(cids)[:64]
    Xc = np.concatenate([packets[c]["X"] for c in check])
    fold_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for m in range(3):
        mseed = int(G["a302"].EXPECTED_MEMBER_SEEDS[rseed][m])
        members.append(
            r302.fit_and_freeze_member(
                fold_dir, outer_fold, rseed, m, mseed, rows, packets, fbc, Xc, core, veto, bagged, ens
            )
        )
    set_tau(veto, 0.10)
    return members


def task(args):
    """One (outer fold, root seed): inner CV over the grid, choice, refit, lock. Returns the lock (no outer-test read)."""
    k, rseed = args
    veto, core, bagged, ens, P8, fam_by_cid, train_rows = (
        G[x] for x in ("veto", "core", "bagged", "ens", "P8", "fam_by_cid", "train_rows")
    )
    t0 = time.time()
    tdir = SCRATCH / f"fold_{k}" / f"root_{rseed}"
    lock_path = tdir / "fit_lock.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())
    outer_train = [r for r in train_rows if G["fam_to_fold"][r["family_id"]] != k]
    ot_cids = [r["context_sha256"] for r in outer_train]
    inner_of = {r["family_id"]: inner_fold(k, r["family_id"]) for r in outer_train}
    oof = {(t, kk, a): {} for t in TAU_GRID for kk in K_GRID for a in A_GRID}
    for tau in TAU_GRID:
        for i in range(4):
            itrain = [r for r in outer_train if inner_of[r["family_id"]] != i]
            itest = [r for r in outer_train if inner_of[r["family_id"]] == i]
            idir = tdir / f"inner_tau{tau}_fold{i}"
            members = fit_members(idir, k, rseed, itrain, P8, fam_by_cid, tau)
            tc = [r["context_sha256"] for r in itest]
            X = np.concatenate([P8[c]["X"] for c in tc])
            outs = score_members(idir, members, X, len(tc), ens)
            for kk in K_GRID:
                ml = [member_picks(o, tau, kk, veto, core, bagged) for o in outs]
                for a in A_GRID:
                    for c, j in zip(tc, ens_picks(ml, a, ens)):
                        oof[(tau, kk, a)][c] = int(j)
    table = []
    for (tau, kk, a), d in oof.items():
        picks = [d[c] for c in ot_cids]
        table.append(
            dict(
                tau=tau,
                k=kk,
                a=a,
                inner_gain=family_gain(picks, ot_cids, P8, fam_by_cid),
                inner_switches=int(sum(j != 0 for j in picks)),
            )
        )
    best = max(x["inner_gain"] for x in table)
    tied = [x for x in table if x["inner_gain"] >= best - TIE]
    chosen = sorted(tied, key=lambda x: (x["inner_switches"], -x["k"], -x["a"], x["tau"]))[0]
    members = fit_members(tdir / "refit", k, rseed, outer_train, P8, fam_by_cid, chosen["tau"])
    lock = dict(schema="jope.h303.fit-lock/v1", outer_fold=k, root_seed=rseed, grid=dict(tau=TAU_GRID, k=K_GRID, a=A_GRID), inner_table=table,
                chosen=chosen, n_tied=len(tied), members=members, outer_test_rows_read_before_lock=0, elapsed_sec=round(time.time() - t0, 1))  # fmt: skip
    tmp = lock_path.with_suffix(".new")
    with tmp.open("w") as f:
        json.dump(lock, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, lock_path)
    print(json.dumps(dict(task=[k, rseed], chosen=chosen, n_tied=len(tied), sec=lock["elapsed_sec"])), flush=True)
    return lock


def setup():
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H303 requires CPU xbabe2 at nice 19")
    pins = json.loads((HERE / "manifest.json").read_text())
    for name, digest in pins["h302_sources"].items():
        if sha(H302 / name) != digest:
            raise RuntimeError("H302 source pin mismatch: " + name)
    sys.path.insert(0, str(H302))
    import analysis_h302 as a302
    import run_h302 as r302

    r302.install_h302_audit_guard()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ["H265_PAUSE_FILE"] = str(SCRATCH / "h303.pause.json")
    core, topk, veto, bagged, ens = r302.load_pinned_h287_modules(SCRATCH / "pinned_h287_mods")
    core.INPUT_POOLS_DIR = r302.POOLS
    sel_bytes = (r302.POOLS / "selected.json").read_bytes()
    assert hashlib.sha256(sel_bytes).hexdigest() == r302.SELECTED_JSON_SHA256
    train_rows = json.loads(sel_bytes)["train"]
    assert len(train_rows) == 1219 and len({r["family_id"] for r in train_rows}) == 205
    cids = [r["context_sha256"] for r in train_rows]
    with np.load(r302.H276_INPUTS / "r882-train-costs.npz") as zt:
        costs = {c: zt[c].astype(np.float32) for c in cids}
    ids, aprior, proj16, ctxf = set(cids), {}, {}, {}
    with np.load(r302.CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        pc, offsets = [str(c) for c in zp["cids"]], zp["offsets"]
        for i, c in enumerate(pc):
            if c in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior[c] = zp["base_prior_cost"][lo:hi].astype(np.float32).copy()
                proj16[c] = zp["proj16"][lo:hi].astype(np.float32).copy()
    with np.load(r302.CACHE / "context_features.npz", allow_pickle=False) as zc:
        for i, c in enumerate(str(x) for x in zc["cids"]):
            if c in ids:
                ctxf[c] = {
                    "lm48_z": zc["lm48_z"][i].astype(np.float32).copy(),
                    "dope629": zc["dope629"][i].astype(np.float32).copy(),
                }
    P8 = topk.load_packets(train_rows, costs, aprior, proj16, ctxf, ens.K)
    G.update(r302=r302, a302=a302, core=core, veto=veto, bagged=bagged, ens=ens, P8=P8, train_rows=train_rows,
             fam_by_cid={r["context_sha256"]: r["family_id"] for r in train_rows},
             fam_to_fold={f: a302.outer_fold_of_family(f) for f in {r["family_id"] for r in train_rows}})  # fmt: skip


def fit_phase(workers):
    """Inner CV + choice + refit + lock for all 12 (outer fold, root seed) tasks. Reads no outer-test row."""
    tasks = [(k, s) for k in range(4) for s in G["a302"].ROOT_TRAINING_SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        locks = list(ex.map(task, tasks))
    return locks


def score_phase(locks):
    """After every lock exists (and the prereg is on origin/v1/next): score each outer_test once with the chosen config."""
    veto, core, bagged, ens, P8, train_rows = (G[x] for x in ("veto", "core", "bagged", "ens", "P8", "train_rows"))
    a302 = G["a302"]
    cids = [r["context_sha256"] for r in train_rows]
    idx = {c: i for i, c in enumerate(cids)}
    picks = np.full((3, len(cids)), -1, dtype=np.int32)
    lock_sha = {}
    for lock in locks:
        k, rseed = lock["outer_fold"], lock["root_seed"]
        lp = SCRATCH / f"fold_{k}" / f"root_{rseed}" / "fit_lock.json"
        lock_sha[f"{k}:{rseed}"] = sha(lp)
        test = [c for c in cids if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
        X = np.concatenate([P8[c]["X"] for c in test])
        outs = score_members(lp.parent / "refit", lock["members"], X, len(test), ens)
        ch = lock["chosen"]
        ml = [member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
        s_idx = a302.ROOT_TRAINING_SEEDS.index(rseed)
        for c, j in zip(test, ens_picks(ml, ch["a"], ens)):
            picks[s_idx, idx[c]] = int(j)
    assert not np.any(picks < 0)
    return picks, lock_sha


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("fit", "score"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    setup()
    if a.phase == "fit":
        fit_phase(a.workers)
    else:
        import evaluate_h303

        evaluate_h303.main(
            G, score_phase, [json.loads(p.read_text()) for p in sorted(SCRATCH.glob("fold_*/root_*/fit_lock.json"))]
        )
