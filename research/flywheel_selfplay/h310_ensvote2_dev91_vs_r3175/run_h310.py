"""H310 (hw_t0827, rd_opus): one-shot, preregistered dev91 transfer test of the FROZEN H303 NESTED ENS_VOTE2 procedure
against the flywheel incumbent r3175.

Phases (separate processes; each refuses to run out of order):
  regress-train  H308 regression: H303 P0 fold 0 / seed 30001 via H308's TRAIN-only loader must reproduce the archived
                 H303 lock and fold-0 picks sha 18fdb2538cea... (no dev91 access; dev guard active).
  lock           for each root seed 30001/30002/30003: H303 NESTED unchanged (18-config grid, inner 4-fold family CV on
                 ALL 205 DL-Train families with inner folds SHA256("H303-inner-v1" NUL "ALL" NUL family) mod 4, 1e-4
                 conservative ties, pinned members) -> one frozen chooser per seed; fit_lock.json per seed, fsynced.
                 A dev guard (audit hook) refuses any open of a dev91 pool file, dev91 costs or dev91 reference table;
                 the TRAIN-only loader decodes no DL-Val cache row; counters prove dev91_rows_read_before_lock == 0.
  score          only if the prereg commit is on origin/v1/next and all 3 locks exist: (a) r3175 regression on its
                 GPU-exact top-8 pool (slot 0 == stored round-3175 dev_utility, 91/91, mean 1.1389572403748425);
                 (b) r882 Native91 regression (slot 0 of r882's dev91 top-8 == best_dev_checkpoint per_context, 91/91,
                 mean 1.1377941898164525); then each seed's frozen chooser picks once on the r3175 pool (PRIMARY,
                 vs r3175 slot 0) and once on the r882 pool (descriptive); evaluate_h310 decides."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = HERE.parent
H303 = FS / "h303_ensvote2_fully_nested_hparams"
H308 = FS / "h308_ensvote2_trainonly_repartition"
DATA = Path(os.environ.get("H310_DATA", "/home/ubuntu/jope-data/h310"))
REF = DATA / "ref"
R3175_GPU_COSTS = REF / "r3175-gpu-costs.npz"  # S3 H281 attempt-001, VersionId EgHvHGbSUy6RSXD60ZKiItTdX8HGMysg
R3175_STORED = Path("/home/ubuntu/jope-data/h278/inputs/r3175/round-3175.json")
R882_DEV_COSTS = Path("/home/ubuntu/jope-data/h276/inputs/r882-secondary-costs.npz")
R882_REF = Path("/home/ubuntu/jope-data/h276/inputs/best_dev_checkpoint.json")
H303_REF = DATA / "h303_ref"
R3175_MEAN = 1.1389572403748425
R882_MEAN = 1.1377941898164525
LOCK_KEY = "ALL"
DEV_GUARD = {"denied": 0}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pins():
    return json.loads((HERE / "manifest.json").read_text())


def check_pins(inputs=False):
    """Source pins always; the 4 dev91 input files are hashed only in the score phase (after the lock)."""
    m = pins()
    for d, key in ((H303, "h303_sources"), (H308, "h308_sources")):
        for name, digest in m[key].items():
            if sha(d / name) != digest:
                raise RuntimeError(f"source pin mismatch: {d.name}/{name}")
    for p, digest in (m["inputs_sha256"] if inputs else {}).items():
        if sha(p) != digest:
            raise RuntimeError("input pin mismatch: " + p)


def dev_cids():
    sel = json.loads(Path("/home/ubuntu/jope-data/h50/input/selected.json").read_text())
    return [r["context_sha256"] for r in sel["dev"]], sel


def install_dev_guard(dev):
    """Lock-phase audit hook: any open of a dev91 pool, dev91 costs or dev91 reference table raises (and is counted)."""
    needles = tuple(c.lower() for c in dev) + (
        "secondary-costs",
        "r3175-gpu-costs",
        "round-3175",
        "best_dev_checkpoint",
        "/sealed",
        "/holdout",
        "meta_test",
        "meta-test",
        "dl-test",
        "dl_test",
        "fresh_",
    )

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            s = os.fsdecode(args[0]).lower()
            if any(n in s for n in needles) or s.endswith(".pt"):
                DEV_GUARD["denied"] += 1
                raise RuntimeError("H310 lock phase: dev91/forbidden open refused: " + s)

    sys.addaudithook(hook)


def load_train_side():
    """H303's setup() through H308's TRAIN-only loader (no DL-Val cache row decoded)."""
    sys.path.insert(0, str(H303))
    sys.path.insert(1, str(H308))
    import run_h303 as r303
    import trainonly_loader as tl

    dev, sel = dev_cids()
    r303.np = tl.TrainOnlyNumpy({r["context_sha256"] for r in sel["train"]})
    r303.SCRATCH = DATA / "train_side"
    r303.setup()
    r303.np = np
    G = r303.G
    G["_fams"] = sorted({r["family_id"] for r in G["train_rows"]})
    leaked = set(G["P8"]) & set(dev)
    counters = dict(
        tl.COUNTERS,
        dev91_packets_built_before_lock=len(leaked),
        dev91_guard_denials=DEV_GUARD["denied"],
    )
    if leaked or counters["dlval_feature_rows_read"]:
        raise RuntimeError("dev91 rows present before lock")
    return r303, counters


def lock_task(rseed):
    r303 = LOCK_CTX["r303"]
    r303.SCRATCH = DATA / "lock"
    r303.G["fam_to_fold"] = {f: "train" for f in r303.G["_fams"]}  # every family is lock-train (!= LOCK_KEY)
    lock = r303.task((LOCK_KEY, rseed))
    return dict(root_seed=rseed, chosen=lock["chosen"], n_tied=lock["n_tied"], elapsed_sec=lock["elapsed_sec"])


LOCK_CTX = {}


def phase_lock(workers):
    check_pins()
    dev, _ = dev_cids()
    install_dev_guard(dev)
    r303, counters = load_train_side()
    LOCK_CTX["r303"] = r303
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lock_task, (30001, 30002, 30003)))
    counters["dev91_guard_denials"] = DEV_GUARD["denied"]
    locks = {}
    for s in (30001, 30002, 30003):
        lp = DATA / "lock" / f"fold_{LOCK_KEY}" / f"root_{s}" / "fit_lock.json"
        lk = json.loads(lp.read_text())
        assert lk["inner_table"] and len(lk["inner_table"]) == 18
        locks[str(s)] = dict(fit_lock_sha256=sha(lp), chosen=lk["chosen"], n_tied=lk["n_tied"],
                             member_ranker_sha256=[m["ranker"]["sha256"] for m in lk["members"]])  # fmt: skip
    rec = dict(wave="H310", phase="lock", tasks=res, locks=locks, dev91_rows_read_before_lock=0, counters=counters)
    rec["dev91_rows_read_before_lock"] = int(
        counters["dlval_feature_rows_read"] + counters["dev91_packets_built_before_lock"]
    )
    assert rec["dev91_rows_read_before_lock"] == 0 and counters["dev91_guard_denials"] == 0
    (DATA / "lock_receipt.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)


def phase_regress_train():
    check_pins()
    dev, _ = dev_cids()
    install_dev_guard(dev)
    r303, counters = load_train_side()
    sys.path.insert(0, str(H308))
    import run_h308 as r308

    r308.DATA = DATA / "regress_train"
    r308.REF_LOCK = H303_REF / "scratch" / "fold_0" / "root_30001" / "fit_lock.json"
    r308.REF_PICKS = H303_REF / "out" / "picks.npz"
    r308.R.update(r303=r303, loader_counters=counters)
    ok = r308.regress(r303)
    rec = json.loads((r308.DATA / "regression_receipt.json").read_text())
    rec["dev91_guard_denials"] = DEV_GUARD["denied"]
    (DATA / "regression_train_receipt.json").write_text(json.dumps(rec, indent=1) + "\n")
    sys.exit(0 if ok and DEV_GUARD["denied"] == 0 else 3)


def phase_score(prereg_commit):
    import evaluate_h310

    check_pins()
    ok = subprocess.run(
        ["git", "-C", str(FS), "merge-base", "--is-ancestor", prereg_commit, "origin/v1/next"]
    ).returncode
    if ok != 0:
        raise RuntimeError("prereg commit is not on origin/v1/next: no dev91 score")
    check_pins(inputs=True)
    lock = json.loads((DATA / "lock_receipt.json").read_text())
    reg = json.loads((DATA / "regression_train_receipt.json").read_text())
    assert reg["status"] == "PASS" and lock["dev91_rows_read_before_lock"] == 0
    for s, v in lock["locks"].items():
        lp = DATA / "lock" / f"fold_{LOCK_KEY}" / f"root_{s}" / "fit_lock.json"
        if sha(lp) != v["fit_lock_sha256"]:
            raise RuntimeError("fit_lock changed after the lock receipt")
    evaluate_h310.main(DATA, lock, reg, prereg_commit)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("regress-train", "lock", "score"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--prereg", default="")
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H310 requires CPU xbabe2 at nice 19")
    if a.phase == "regress-train":
        phase_regress_train()
    elif a.phase == "lock":
        phase_lock(a.workers)
    else:
        phase_score(a.prereg)
