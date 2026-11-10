"""H308 (hw_t0735, rd_opus): H303's NESTED ENS_VOTE2, frozen, through a clean-room TRAIN-only loader, on 2 new outer
partitions.

Imported unchanged with sha256 pins (manifest.json): run_h303.py / evaluate_h303.py, which import H302 (run_h302.py,
analysis_h302.py) and the H287 modules at 213b30ad52. H303's 18-config grid (TAU {0.05,0.10,0.15} x k {5,6,7} x
a {2,3}), its inner 4-fold family CV (SHA256("H303-inner-v1" NUL str(fold) NUL family) mod 4), its 1e-4 conservative
tie rule, root seeds 30001-30003 and member seeds are unchanged. Only the OUTER partition changes:
  P1: SHA256("H308-outer-v1" NUL family) last byte mod 4;  P2: SHA256("H308-outer-v2" NUL family) last byte mod 4.
P0 = H302/H303's SHA256("H302-outer-v1" NUL family) mod 4 (reported from H303, no refit).

Phases:
  regress  P0 outer fold 0 / seed 30001 with the TRAIN-only loader must reproduce H303's archived lock (choice
           (0.05,5,2), inner gain 0.0961251, 18-row inner table, member model hashes) and fold-0 picks sha 18fdb253...
  fit      24 tasks (P1, P2) x 4 folds x 3 seeds, each with outer-test packets guarded (any access raises) until its
           fit_lock exists; an honest receipt counts outer-test packets present in memory (built) vs accessed.
  score    after the prereg is on origin/v1/next: each outer_test scored once, then evaluate_h308."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import trainonly_loader as tl

HERE = Path(__file__).resolve().parent
H303 = HERE.parent / "h303_ensvote2_fully_nested_hparams"
DATA = Path(os.environ.get("H308_DATA", "/home/ubuntu/jope-data/h308"))
SALTS = {"P1": b"H308-outer-v1\x00", "P2": b"H308-outer-v2\x00"}
REF_LOCK = DATA / "h303_ref" / "scratch" / "fold_0" / "root_30001" / "fit_lock.json"
REF_PICKS = DATA / "h303_ref" / "out" / "picks.npz"
R = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outer_fold(salt: bytes, family: str) -> int:
    return hashlib.sha256(salt + family.encode()).digest()[-1] % 4


def load_h303():
    pins = json.loads((HERE / "manifest.json").read_text())["h303_sources"]
    for name, digest in pins.items():
        if sha(H303 / name) != digest:
            raise RuntimeError("H303 source pin mismatch: " + name)
    sys.path.insert(0, str(H303))
    import evaluate_h303 as ev
    import run_h303 as r303

    return r303, ev


def setup_trainonly(r303):
    """run_h303.setup() with the two mixed cache files opened through the TRAIN-only prefix loader."""
    sel = json.loads((Path("/home/ubuntu/jope-data/h50/input") / "selected.json").read_text())
    r303.np = tl.TrainOnlyNumpy({r["context_sha256"] for r in sel["train"]})
    r303.setup()
    r303.np = np
    assert tl.COUNTERS["dlval_feature_rows_read"] == 0
    R["loader_counters"] = dict(tl.COUNTERS)


def strip_lock(lock):
    keep = {k: lock[k] for k in ("outer_fold", "root_seed", "inner_table", "chosen", "n_tied")}
    keep["members"] = [
        dict(
            ranker=m["ranker"]["sha256"],
            veto=m["veto"]["sha256"],
            bags=[b["sha256"] for b in m["bags"]],
            lr=m["selected_lr"],
            n_est=m["selected_n_estimators"],
            margin=m["strict_margin"],
            tau=m.get("selected_tau"),
        )
        for m in lock["members"]
    ]
    return keep


def set_partition(r303, part):
    G = r303.G
    if part == "P0":
        G["fam_to_fold"] = {f: G["a302"].outer_fold_of_family(f) for f in G["_fams"]}
    else:
        G["fam_to_fold"] = {f: outer_fold(SALTS[part], f) for f in G["_fams"]}
    r303.SCRATCH = DATA / part


def guarded_task(args):
    """One (partition, outer fold, seed) H303 task with outer-test packets guarded until its fit_lock exists."""
    part, k, rseed = args
    r303 = R["r303"]
    G = r303.G
    set_partition(r303, part)
    test_cids = {r["context_sha256"] for r in G["train_rows"] if G["fam_to_fold"][r["family_id"]] == k}
    full = G["P8"]
    guard = tl.GuardedPackets(full, test_cids)
    G["P8"] = guard
    try:
        r303.task((k, rseed))
    finally:
        G["P8"] = full
    rec = dict(
        partition=part,
        outer_fold=k,
        root_seed=rseed,
        outer_test_contexts=len(test_cids),
        outer_test_packets_built_in_memory_before_lock=len(test_cids),
        outer_test_packets_accessed_before_lock=0,
        guard_denials=guard.denied,
        note="all 1,219 TRAIN packets (features + utilities) are built once in the parent before forking; the guard "
        "raises on any outer-test access inside the task, so 'accessed' is enforced, 'built' is the honest count",
    )
    assert guard.denied == 0
    p = r303.SCRATCH / f"fold_{k}" / f"root_{rseed}" / "guard_receipt.json"
    p.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def score_fold(r303, lock):
    """Outer-test picks of one locked task (chosen TAU, k, a), as run_h303.score_phase computes them."""
    G = r303.G
    k, rseed = lock["outer_fold"], lock["root_seed"]
    cids = [r["context_sha256"] for r in G["train_rows"]]
    test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
    X = np.concatenate([G["P8"][cids[i]]["X"] for i in test])
    d = r303.SCRATCH / f"fold_{k}" / f"root_{rseed}" / "refit"
    outs = r303.score_members(d, lock["members"], X, len(test), G["ens"])
    ch = lock["chosen"]
    ml = [r303.member_picks(o, ch["tau"], ch["k"], G["veto"], G["core"], G["bagged"]) for o in outs]
    return test, np.asarray(r303.ens_picks(ml, ch["a"], G["ens"]), dtype=np.int32)


def regress(r303):
    guard = guarded_task(("P0", 0, 30001))
    lock = json.loads((DATA / "P0" / "fold_0" / "root_30001" / "fit_lock.json").read_text())
    ref = json.loads(REF_LOCK.read_text())
    lock_ok = strip_lock(lock) == strip_lock(ref)
    test, fresh = score_fold(r303, lock)
    with np.load(REF_PICKS) as z:
        archived = np.asarray(z["ENS_VOTE2_NESTED"][0][test], dtype=np.int32)
    h_f, h_a = hashlib.sha256(fresh.tobytes()).hexdigest(), hashlib.sha256(archived.tobytes()).hexdigest()
    rec = dict(
        wave="H308",
        step="regression_first_trainonly_loader",
        task=["P0", 0, 30001],
        ref_lock_sha256=sha(REF_LOCK),
        fresh_chosen=lock["chosen"],
        ref_chosen=ref["chosen"],
        lock_choice_inner_table_and_member_hashes_equal=lock_ok,
        fold0_seed30001_picks_sha256_fresh=h_f,
        fold0_seed30001_picks_sha256_archived=h_a,
        n_fold0_contexts=len(test),
        loader_counters=R["loader_counters"],
        guard=guard,
    )
    rec["status"] = (
        "PASS"
        if lock_ok
        and h_f == h_a == "18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa"
        and lock["chosen"]["tau"] == 0.05
        and lock["chosen"]["k"] == 5
        and lock["chosen"]["a"] == 2
        and round(lock["chosen"]["inner_gain"], 7) == 0.0961251
        else "REGRESSION_FAIL"
    )
    (DATA / "regression_receipt.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    return rec["status"] == "PASS"


def fit_phase(r303, workers, parts):
    tasks = [(p, k, s) for p in parts for k in range(4) for s in r303.G["a302"].ROOT_TRAINING_SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(guarded_task, tasks))
    print(json.dumps(dict(fit_done=len(recs), sec=round(time.time() - t0, 1))), flush=True)


def score_phase(r303, part):
    """After all 12 locks of a partition exist: score each outer_test once."""
    set_partition(r303, part)
    G = r303.G
    n = len(G["train_rows"])
    picks = np.full((3, n), -1, dtype=np.int32)
    lock_sha, chosen, guards = {}, [], []
    for k in range(4):
        for s, rseed in enumerate(G["a302"].ROOT_TRAINING_SEEDS):
            lp = r303.SCRATCH / f"fold_{k}" / f"root_{rseed}" / "fit_lock.json"
            lock = json.loads(lp.read_text())
            lock_sha[f"{k}:{rseed}"] = sha(lp)
            guards.append(json.loads((lp.parent / "guard_receipt.json").read_text()))
            chosen.append(dict(outer_fold=k, root_seed=rseed, **lock["chosen"], n_tied=lock["n_tied"]))
            test, p = score_fold(r303, lock)
            picks[s, test] = p
    assert not np.any(picks < 0)
    return picks, lock_sha, chosen, guards


def prepare(r303):
    setup_trainonly(r303)
    G = r303.G
    G["_fams"] = sorted({r["family_id"] for r in G["train_rows"]})
    R["r303"] = r303


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("regress", "fit", "score"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--parts", default="P1,P2")
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H308 requires CPU xbabe2 at nice 19")
    r303, ev = load_h303()
    prepare(r303)
    if a.phase == "regress":
        sys.exit(0 if regress(r303) else 3)
    rec = json.loads((DATA / "regression_receipt.json").read_text())
    assert rec["status"] == "PASS", "regression must PASS before any H308 fit"
    parts = a.parts.split(",")
    if a.phase == "fit":
        fit_phase(r303, a.workers, parts)
    else:
        import evaluate_h308

        evaluate_h308.main(r303, ev, {p: score_phase(r303, p) for p in parts}, R["loader_counters"])
