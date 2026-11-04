"""H319 (hw_t1205, rd_opus): bagged ENS_VOTE2 members vs unchanged H303 NESTED (BASE), paired, TRAIN-only.

BASE   = H303 NESTED as archived by H315 (S3 H315/picks.npz, exact version) for H308 partitions P0/P1/P2 x root seeds
         30001-30003 x 4 outer folds; per-task BASE choices (TAU, k, a) from H315 metrics.json (origin/v1/next).
BAG    = per (partition, outer fold, root seed): BASE's chosen (TAU, k, a); no new inner CV. B = 9 members, each the
         unchanged H302/H287 BAG_SWITCH member recipe (r302.fit_and_freeze_member with TAU in force) fitted on one
         whole-FAMILY bootstrap resample of outer_train. Bag b of a task:
           rng = numpy.random.default_rng(319190 + 1000 * partition_idx + root_seed + 10 * outer_fold)  (P0=0,P1=1,P2=2)
           for b in 0..8 (in order): families = rng.choice(sorted outer_train families, size=n_families, replace=True);
                                     member_seed = int(rng.integers(1, 2**31 - 1))
           each drawn family copy is relabelled as a distinct family "<family>#<copy>" with context ids "<cid>#<copy>"
           (cluster-bootstrap relabel; the same feature/label packets).
         Vote: each bag member's BAG_SWITCH pick at (TAU, k); switch from slot 0 iff >= ceil(9*a/3) bags pick the same
         nonzero candidate (a=2 -> 6/9, a=3 -> 9/9); pick = the most-voted nonzero candidate, ties -> slot 0.
Outer-test packets are guarded (any access raises) until the fsynced fit_lock exists; outer_test scored once."""

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
FS = Path(os.environ.get("H319_FS", str(HERE.parent)))
DATA = Path(os.environ.get("H319_DATA", "/home/ubuntu/jope-data/h319-opus"))
PARTS = ("P0", "P1", "P2")
SEEDS = (30001, 30002, 30003)
SALTS = {"P1": b"H308-outer-v1\x00", "P2": b"H308-outer-v2\x00"}
B = 9
BAG_BASE = 319190
R = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outer_fold(part, family, a302):
    if part == "P0":
        return a302.outer_fold_of_family(family)
    return hashlib.sha256(SALTS[part] + family.encode()).digest()[-1] % 4


def bag_rng(part, k, rseed):
    return np.random.default_rng(BAG_BASE + 1000 * PARTS.index(part) + rseed + 10 * k)


def bag_draws(families, part, k, rseed):
    """The 9 (bootstrap family list, member seed) draws of a task, in the preregistered order."""
    rng = bag_rng(part, k, rseed)
    fams = sorted(families)
    out = []
    for _ in range(B):
        drawn = rng.choice(fams, size=len(fams), replace=True)
        mseed = int(rng.integers(1, 2**31 - 1))
        out.append(([str(f) for f in drawn], mseed))
    return out


def relabel(drawn, rows_by_family):
    """Cluster bootstrap: the c-th copy of a family becomes family '<f>#<c>' with contexts '<cid>#<c>'."""
    seen, rows = {}, []
    for f in drawn:
        c = seen.get(f, 0)
        seen[f] = c + 1
        for r in rows_by_family[f]:
            rows.append(
                {
                    **r,
                    "context_sha256": f"{r['context_sha256']}#{c}",
                    "family_id": f"{f}#{c}",
                    "_src": r["context_sha256"],
                }
            )
    return rows


def vote(member_picks, a, n=B):
    """Switch iff >= ceil(n*a/3) members pick the same nonzero candidate; most-voted candidate; ties -> slot 0."""
    need = math.ceil(n * a / 3)
    out = []
    for i in range(len(member_picks[0])):
        counts = {}
        for m in member_picks:
            j = int(m[i])
            if j != 0:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            out.append(0)
            continue
        best = max(counts.values())
        top = [j for j, c in counts.items() if c == best]
        out.append(top[0] if len(top) == 1 and best >= need else 0)
    return out


def base_choices():
    m = json.loads((FS / "h315_margin_gated_ens_vote2_robust" / "metrics.json").read_text())
    return {(t["p"], t["f"], t["s"]): dict(zip(("tau", "k", "a"), t["base"])) for t in m["choices"]["tasks"]}


def task(args):
    part, k, rseed = args
    r303 = R["r303"]
    G = r303.G
    import trainonly_loader as tl

    G["fam_to_fold"] = {f: outer_fold(part, f, G["a302"]) for f in G["_fams"]}
    r302, veto, core, bagged, ens, train_rows = (G[x] for x in ("r302", "veto", "core", "bagged", "ens", "train_rows"))
    ch = R["base"][(part, k, rseed)]
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
        by_fam = {}
        for r in outer_train:
            by_fam.setdefault(r["family_id"], []).append(r)
        draws = bag_draws(by_fam, part, k, rseed)
        r303.set_tau(veto, ch["tau"])
        members = []
        for b, (drawn, mseed) in enumerate(draws):
            rows = relabel(drawn, by_fam)
            packets = {r["context_sha256"]: P8[r["_src"]] for r in rows}
            fbc = {r["context_sha256"]: r["family_id"] for r in rows}
            check = sorted(packets)[:64]
            Xc = np.concatenate([packets[c]["X"] for c in check])
            tdir.mkdir(parents=True, exist_ok=True)
            m = r302.fit_and_freeze_member(tdir, k, rseed, b, mseed, rows, packets, fbc, Xc, core, veto, bagged, ens)
            m["bootstrap_families_sha256"] = hashlib.sha256("\n".join(drawn).encode()).hexdigest()
            members.append(m)
        r303.set_tau(veto, 0.10)
    finally:
        G["P8"] = full
    assert P8.denied == 0
    summary = dict(partition=part, outer_fold=k, root_seed=rseed, base_choice=ch, bag_seeds=[s for _, s in draws],
                   distinct_families_per_bag=[len(set(d)) for d, _ in draws], elapsed_sec=round(time.time() - t0, 1))  # fmt: skip
    lock = dict(schema="jope.h319.fit-lock/v1", summary=summary, members=members, outer_test_packets_accessed_before_lock=0,
                outer_test_packets_built_in_memory_before_lock=len(test_cids), guard_denials=P8.denied)  # fmt: skip
    tmp = lock_path.with_suffix(".new")
    with tmp.open("w") as f:
        json.dump(lock, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, lock_path)
    print(json.dumps(dict(task=[part, k, rseed], choice=ch, sec=summary["elapsed_sec"])), flush=True)
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
    R.update(r303=r303, loader_counters=dict(tl.COUNTERS), base=base_choices())
    return r303


def phase_regress():
    import evaluate_h319

    r303 = prepare()
    evaluate_h319.regress(r303, DATA)


def phase_fit(workers):
    prepare()
    reg = json.loads((DATA / "regression_receipt.json").read_text())
    assert reg["status"] == "PASS", "regression must PASS before any BAG fit"
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
    picks = np.full((3, len(cids)), -1, np.int32)
    locks = {}
    for k in range(4):
        test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
        X = np.concatenate([P8[cids[i]]["X"] for i in test])
        for s, rseed in enumerate(SEEDS):
            tdir = DATA / part / f"fold_{k}" / f"root_{rseed}"
            lp = tdir / "fit_lock.json"
            lock = json.loads(lp.read_text())
            locks[f"{k}:{rseed}"] = sha(lp)
            ch = lock["summary"]["base_choice"]
            outs = r303.score_members(tdir, lock["members"], X, len(test), ens)
            ml = [r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
            picks[s, test] = vote(ml, ch["a"])
    assert (picks >= 0).all()
    return picks, locks


def phase_score():
    import evaluate_h319

    r303 = prepare()
    evaluate_h319.main(r303, DATA, {p: score_part(r303, p) for p in PARTS})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("regress", "fit", "score"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H319 requires CPU xbabe2 at nice 19")
    {"regress": phase_regress, "score": phase_score}.get(a.phase, lambda: phase_fit(a.workers))()
