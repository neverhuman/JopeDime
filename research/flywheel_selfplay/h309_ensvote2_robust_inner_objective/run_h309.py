"""H309 phase-separated regression, robust inner choice, refit-lock and score."""

from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import robust_objective as rob
import trainonly_loader as tl

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/home/ubuntu/jope-data/h309-astra")
H303 = ROOT / "vendor/h303_ensvote2_fully_nested_hparams"
R = None
ORIGINAL_FIT = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return sha(p)


def capture_fit(*args, **kwargs):
    members = ORIGINAL_FIT(*args, **kwargs)
    save(Path(args[0]) / "members.json", members)
    return members


def setup(phase):
    global R, ORIGINAL_FIT
    pins = json.loads((ROOT / "source/dependency_pins.json").read_text())
    assert sha(ROOT / "source/trainonly_loader.py") == pins["loader_sha256"]
    for name, digest in pins["vendor"].items():
        assert sha(ROOT / "vendor" / name) == digest, name
    sys.path.insert(0, str(H303))
    import run_h303 as r

    R = r
    r.SCRATCH = DATA / phase
    raw = Path("/home/ubuntu/jope-data/h50/input/selected.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    )
    selected = json.loads(raw)
    train_cids = {x["context_sha256"] for x in selected["train"]}
    r.np = tl.TrainOnlyNumpy(train_cids)
    r.setup()
    r.np = np
    assert tl.COUNTERS["dlval_feature_rows_read"] == 0
    ORIGINAL_FIT = r.fit_members
    r.fit_members = capture_fit
    assert len(r.G["train_rows"]) == 1219 and len(r.G["P8"]) == 1219
    save(
        DATA / f"{phase}_loader_receipt.json",
        dict(
            phase=phase,
            loader_counters=tl.COUNTERS,
            train_contexts=1219,
            train_families=205,
            outer_test_rows_built_before_locks=True,
            by_outer_fold={
                str(k): sum(
                    r.G["fam_to_fold"][row["family_id"]] == k
                    for row in r.G["train_rows"]
                )
                for k in range(4)
            },
            note="All TRAIN features and utility rows built before locks; fitting packet guard separately enforces no outer-test access",
            source_sha256={f.name: sha(f) for f in (ROOT / "source").glob("*.py")},
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    return r


def guarded(task_fn, k, seed):
    g = R.G
    full = g["P8"]
    forbidden = {
        row["context_sha256"]
        for row in g["train_rows"]
        if g["fam_to_fold"][row["family_id"]] == k
    }
    guard = tl.GuardedPackets(full, forbidden)
    g["P8"] = guard
    try:
        result = task_fn(k, seed)
    finally:
        g["P8"] = full
    assert guard.denied == 0
    save(
        R.SCRATCH / f"fold_{k}/root_{seed}/guard_receipt.json",
        dict(
            outer_test_rows_built_before_lock=len(forbidden),
            outer_test_packet_accesses_before_lock=0,
            guard_denials=guard.denied,
        ),
    )
    return result


def score_fold(lock):
    r = R
    g = r.G
    k = lock["outer_fold"]
    seed = lock["root_seed"]
    rows = g["train_rows"]
    cids = [row["context_sha256"] for row in rows]
    test = [i for i, row in enumerate(rows) if g["fam_to_fold"][row["family_id"]] == k]
    X = np.concatenate([g["P8"][cids[i]]["X"] for i in test])
    outs = r.score_members(
        r.SCRATCH / f"fold_{k}/root_{seed}/refit",
        lock["members"],
        X,
        len(test),
        g["ens"],
    )
    ch = lock["chosen"]
    members = [
        r.member_picks(o, ch["tau"], ch["k"], g["veto"], g["core"], g["bagged"])
        for o in outs
    ]
    return test, np.asarray(r.ens_picks(members, ch["a"], g["ens"]), dtype=np.int32)


def regression_task(k, seed):
    original = R.family_gain
    profiles = []

    def capture(picks, cids, packets, family_by_cid):
        profile = rob.family_profile(picks, cids, packets, family_by_cid)
        profiles.append(profile)
        return original(picks, cids, packets, family_by_cid)

    R.family_gain = capture
    try:
        lock = R.task((k, seed))
    finally:
        R.family_gain = original
    assert len(profiles) == 18
    for row, profile in zip(lock["inner_table"], profiles):
        profile.update(row)
    save(
        DATA / "regression_inner_profiles.json",
        dict(
            mean_table_sha256=rob.canonical_sha(lock["inner_table"]), profiles=profiles
        ),
    )
    return lock


def strip(lock):
    return dict(
        outer_fold=lock["outer_fold"],
        root_seed=lock["root_seed"],
        chosen=lock["chosen"],
        inner_table=lock["inner_table"],
        n_tied=lock["n_tied"],
        members=[
            dict(
                ranker=m["ranker"]["sha256"],
                veto=m["veto"]["sha256"],
                bags=[b["sha256"] for b in m["bags"]],
                lr=m["selected_lr"],
                n_est=m["selected_n_estimators"],
                margin=m["strict_margin"],
                tau=m["selected_tau"],
            )
            for m in lock["members"]
        ],
    )


def regress():
    setup("regression")
    lock = guarded(regression_task, 0, 30001)
    ref = json.loads(
        (DATA / "h303_reference/scratch/fold_0/root_30001/fit_lock.json").read_text()
    )
    test, picks = score_fold(lock)
    with np.load(DATA / "h303_reference/out/picks.npz") as z:
        old = np.asarray(z["ENS_VOTE2_NESTED"][0][test], dtype=np.int32)
    fresh_sha = hashlib.sha256(picks.tobytes()).hexdigest()
    old_sha = hashlib.sha256(old.tobytes()).hexdigest()
    ok = (
        strip(lock) == strip(ref)
        and fresh_sha
        == old_sha
        == "18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa"
        and round(lock["chosen"]["inner_gain"], 7) == 0.0961251
        and [lock["chosen"][x] for x in ["tau", "k", "a"]] == [0.05, 5, 2]
    )
    receipt = dict(
        status="PASS" if ok else "REGRESSION_FAIL",
        task=[0, 30001],
        chosen=lock["chosen"],
        whole_lock_semantics_equal=strip(lock) == strip(ref),
        fresh_picks_sha256=fresh_sha,
        reference_picks_sha256=old_sha,
        reference_lock_sha256=sha(
            DATA / "h303_reference/scratch/fold_0/root_30001/fit_lock.json"
        ),
        inner_table_sha256=rob.canonical_sha(lock["inner_table"]),
        reference_inner_table_sha256=rob.canonical_sha(ref["inner_table"]),
        loader_counters=tl.COUNTERS,
        source_sha256={f.name: sha(f) for f in (ROOT / "source").glob("*.py")},
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    save(DATA / "regression_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    if not ok:
        raise RuntimeError("REGRESSION_FAIL")


def robust_task(k, seed):
    r = R
    g = r.G
    t0 = time.time()
    tdir = r.SCRATCH / f"fold_{k}/root_{seed}"
    tdir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in g["train_rows"] if g["fam_to_fold"][row["family_id"]] != k]
    cids = [row["context_sha256"] for row in rows]
    reuse = k == 0 and seed == 30001
    if reuse:
        cap = json.loads((DATA / "regression_inner_profiles.json").read_text())
        ref = json.loads(
            (
                DATA / f"h303_reference/scratch/fold_{k}/root_{seed}/fit_lock.json"
            ).read_text()
        )
        assert cap["mean_table_sha256"] == rob.canonical_sha(ref["inner_table"])
        profiles = cap["profiles"]
        assert all(p["cids"] == cids for p in profiles)
    else:
        inner = {row["family_id"]: r.inner_fold(k, row["family_id"]) for row in rows}
        oof = {
            (tau, kk, aa): {}
            for tau in r.TAU_GRID
            for kk in r.K_GRID
            for aa in r.A_GRID
        }
        for tau in r.TAU_GRID:
            for i in range(4):
                train = [row for row in rows if inner[row["family_id"]] != i]
                test = [row for row in rows if inner[row["family_id"]] == i]
                idir = tdir / f"inner_tau{tau}_fold{i}"
                members = r.fit_members(
                    idir, k, seed, train, g["P8"], g["fam_by_cid"], tau
                )
                tc = [row["context_sha256"] for row in test]
                X = np.concatenate([g["P8"][c]["X"] for c in tc])
                outs = r.score_members(idir, members, X, len(tc), g["ens"])
                for kk in r.K_GRID:
                    ml = [
                        r.member_picks(o, tau, kk, g["veto"], g["core"], g["bagged"])
                        for o in outs
                    ]
                    for aa in r.A_GRID:
                        for c, j in zip(tc, r.ens_picks(ml, aa, g["ens"])):
                            oof[(tau, kk, aa)][c] = int(j)
        profiles = []
        for (tau, kk, aa), d in oof.items():
            picks = [d[c] for c in cids]
            p = rob.family_profile(picks, cids, g["P8"], g["fam_by_cid"])
            p.update(
                tau=tau,
                k=kk,
                a=aa,
                inner_gain=r.family_gain(picks, cids, g["P8"], g["fam_by_cid"]),
                inner_switches=int(sum(j != 0 for j in picks)),
            )
            profiles.append(p)
    mean_table = [
        {key: p[key] for key in ("tau", "k", "a", "inner_gain", "inner_switches")}
        for p in profiles
    ]
    ref = json.loads(
        (
            DATA / f"h303_reference/scratch/fold_{k}/root_{seed}/fit_lock.json"
        ).read_text()
    )
    table = [
        {
            key: p[key]
            for key in (
                "tau",
                "k",
                "a",
                "inner_gain",
                "inner_switches",
                "trimmed_gain",
                "family_median",
            )
        }
        for p in profiles
    ]
    chosen, n_trim, n_med = rob.choose(table)
    members = r.fit_members(
        tdir / "refit", k, seed, rows, g["P8"], g["fam_by_cid"], chosen["tau"]
    )
    profile_sha = save(tdir / "inner_profiles.json", profiles)
    lock = dict(
        schema="jope.h309.fit-lock/v1",
        outer_fold=k,
        root_seed=seed,
        chosen=chosen,
        inner_table=table,
        mean_table=mean_table,
        mean_table_sha256=rob.canonical_sha(mean_table),
        h303_mean_table_sha256=rob.canonical_sha(ref["inner_table"]),
        mean_table_matches_h303=rob.canonical_sha(mean_table)
        == rob.canonical_sha(ref["inner_table"]),
        n_trimmed_ties=n_trim,
        n_median_ties=n_med,
        members=members,
        profiles_sha256=profile_sha,
        reused_fresh_regression_profiles=reuse,
        outer_test_rows_built_before_lock=sum(
            g["fam_to_fold"][row["family_id"]] == k for row in g["train_rows"]
        ),
        outer_test_packet_accesses_before_lock=0,
        elapsed_sec=time.time() - t0,
    )
    save(tdir / "fit_lock.json", lock)
    print(
        json.dumps(
            dict(
                task=[k, seed],
                chosen=chosen,
                mean_table_match=lock["mean_table_matches_h303"],
                elapsed=lock["elapsed_sec"],
            )
        ),
        flush=True,
    )
    return lock


def work(pair):
    return guarded(robust_task, *pair)


def fit(workers):
    assert (
        json.loads((ROOT / "source/train_input_receipt.json").read_text())["status"]
        == "PASS"
    )
    assert (
        json.loads((DATA / "regression_receipt.json").read_text())["status"] == "PASS"
    )
    setup("robust")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        locks = list(
            pool.map(work, [(k, s) for k in range(4) for s in (30001, 30002, 30003)])
        )
    inventory = {
        f"{x['outer_fold']}:{x['root_seed']}": sha(
            R.SCRATCH / f"fold_{x['outer_fold']}/root_{x['root_seed']}/fit_lock.json"
        )
        for x in locks
    }
    assert len(inventory) == 12
    save(
        DATA / "pre_score_lock_inventory.json",
        dict(
            locks=inventory,
            outer_test_scores_computed=False,
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    print("ALL12_LOCKS_FSYNCED_BEFORE_ROBUST_OUTER_SCORE", flush=True)


def score():
    assert (
        json.loads((DATA / "regression_receipt.json").read_text())["status"] == "PASS"
    )
    inventory = json.loads((DATA / "pre_score_lock_inventory.json").read_text())
    assert len(inventory["locks"]) == 12
    # Fail before scoring if any lock has changed since the fsynced inventory.
    for key, digest in inventory["locks"].items():
        k, seed = key.split(":")
        assert sha(DATA / f"robust/fold_{k}/root_{seed}/fit_lock.json") == digest
    save(
        DATA / "score_started.json",
        dict(
            inventory_sha256=sha(DATA / "pre_score_lock_inventory.json"),
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    setup("score")
    R.SCRATCH = DATA / "robust"
    picks = np.full((3, 1219), -1, np.int32)
    locks = []
    for k in range(4):
        for s, seed in enumerate((30001, 30002, 30003)):
            lock = json.loads(
                (R.SCRATCH / f"fold_{k}/root_{seed}/fit_lock.json").read_text()
            )
            test, pp = score_fold(lock)
            picks[s, test] = pp
            locks.append(lock)
    assert (picks >= 0).all()
    np.savez_compressed(DATA / "robust_picks.npz", picks=picks)
    import evaluate_h309

    evaluate_h309.main(R, locks, picks, DATA, ROOT / "source")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["regression", "fit", "score"])
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    assert 1 <= args.workers <= 6
    assert os.uname().nodename == "xbabe2" and os.getpriority(os.PRIO_PROCESS, 0) == 19
    if args.phase == "regression":
        regress()
    elif args.phase == "fit":
        fit(args.workers)
    else:
        score()
