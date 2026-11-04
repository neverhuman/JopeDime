"""H306 (hw_t0630, rd_opus): H303's fully nested ENS_VOTE2 with an EXTENDED grid.

Everything is H303's code (run_h303.py / evaluate_h303.py imported with sha256 pins), which is H302's code and pins
(H287 213b30ad52, selected.json 981960ad..., outer folds SHA256("H302-outer-v1" NUL family) mod 4, root seeds
30001-30003 and member seeds, bootstrap 302150 x 10,000). Only the grid changes:
  TAU in {0.025, 0.05, 0.10}  x  member vote k in {4, 5, 6} of 8  x  agreement a in {2} of 3   (9 configs)
Inner 4-fold family CV inside each outer_train (H303's inner fold rule), ties within 1e-4 -> most conservative
(fewest inner switches, then higher k, higher a, lower TAU), refit on outer_train, fit_lock, outer_test scored once.

Phases:
  regress  fresh fit of the single H303 task (outer fold 0, root 30001) with H303's grid; its fit_lock (choice, inner
           table, member model hashes) and its fold-0 outer-test picks must equal the archived H303 ones; else
           REGRESSION_FAIL.
  fit      the 12 XGRID tasks (no outer-test read).
  score    after the prereg is on origin/v1/next: score each outer_test once and evaluate."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H303 = HERE.parent / "h303_ensvote2_fully_nested_hparams"
DATA = Path(os.environ.get("H306_DATA", "/home/ubuntu/jope-data/h306"))
XGRID = dict(tau=(0.025, 0.05, 0.10), k=(4, 5, 6), a=(2,))
REF_LOCK = DATA / "h303_ref" / "scratch" / "fold_0" / "root_30001" / "fit_lock.json"
REF_PICKS = DATA / "h303_ref" / "out" / "picks.npz"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_h303():
    pins = json.loads((HERE / "manifest.json").read_text())["h303_sources"]
    for name, digest in pins.items():
        if sha(H303 / name) != digest:
            raise RuntimeError("H303 source pin mismatch: " + name)
    sys.path.insert(0, str(H303))
    import evaluate_h303 as ev
    import run_h303 as r303

    return r303, ev


def strip_lock(lock):
    """The parts of a fit lock that must be bit-identical in a fresh refit (paths and timing excluded)."""
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


def regress(r303):
    r303.SCRATCH = DATA / "regress_scratch"
    r303.setup()
    G = r303.G
    lock = r303.task((0, 30001))
    ref = json.loads(REF_LOCK.read_text())
    lock_ok = strip_lock(lock) == strip_lock(ref)
    cids = [r["context_sha256"] for r in G["train_rows"]]
    test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == 0]
    X = np.concatenate([G["P8"][cids[i]]["X"] for i in test])
    outs = r303.score_members(r303.SCRATCH / "fold_0" / "root_30001" / "refit", lock["members"], X, len(test), G["ens"])
    ch = lock["chosen"]
    ml = [r303.member_picks(o, ch["tau"], ch["k"], G["veto"], G["core"], G["bagged"]) for o in outs]
    fresh = np.asarray(r303.ens_picks(ml, ch["a"], G["ens"]), dtype=np.int32)
    with np.load(REF_PICKS) as z:
        archived = np.asarray(z["ENS_VOTE2_NESTED"][0][test], dtype=np.int32)
    h_f, h_a = hashlib.sha256(fresh.tobytes()).hexdigest(), hashlib.sha256(archived.tobytes()).hexdigest()
    rec = dict(
        wave="H306",
        step="regression_first",
        task=[0, 30001],
        ref_lock_sha256=sha(REF_LOCK),
        fresh_chosen=ch,
        ref_chosen=ref["chosen"],
        lock_choice_inner_table_and_member_hashes_equal=lock_ok,
        fold0_seed30001_picks_sha256_fresh=h_f,
        fold0_seed30001_picks_sha256_archived=h_a,
        n_fold0_contexts=len(test),
        fold0_changed=int((fresh != 0).sum()),
    )
    rec["status"] = "PASS" if (lock_ok and h_f == h_a) else "REGRESSION_FAIL"
    (DATA / "regression_receipt.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    return rec["status"] == "PASS"


def set_xgrid(r303):
    r303.TAU_GRID, r303.K_GRID, r303.A_GRID = XGRID["tau"], XGRID["k"], XGRID["a"]
    r303.SCRATCH = DATA / "scratch"


# H305 audit (h): the H302/H303 loader decodes whole cache arrays, whose final 91 rows are DL-Val (dev91) features,
# before filtering to TRAIN. For the H306 fit and score phases, the two mixed cache files are opened as memory maps of
# their (ZIP_STORED) .npy members, so only the TRAIN rows that setup() indexes are decoded. Values are byte-identical.
SELECTIVE = {"pool_priors.npz", "context_features.npz"}


def stored_member(path, name):
    """np.memmap of an uncompressed .npy member of an .npz (no array bytes decoded until indexed)."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{path}:{name} is compressed; selective read impossible")
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            local = f.read(30)
            assert local[:4] == b"PK\x03\x04"
            n, m = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
            start = info.header_offset + 30 + n + m
            f.seek(start)
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            data_off = f.tell()
    assert not fortran and not dtype.hasobject
    return np.memmap(path, dtype=dtype, mode="r", offset=data_off, shape=shape)


class _SelectiveNpz:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, key):
        return stored_member(self.path, key + ".npy")


class _SelectiveNumpy:
    """Stands in for `np` inside run_h303: np.load of the two mixed cache files returns memory maps."""

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, file, *args, **kw):
        if Path(file).name in SELECTIVE:
            return _SelectiveNpz(Path(file))
        return np.load(file, *args, **kw)


def use_selective_loader(r303):
    r303.np = _SelectiveNumpy()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("regress", "fit", "score"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H306 requires CPU xbabe2 at nice 19")
    r303, ev = load_h303()
    if a.phase == "regress":
        sys.exit(0 if regress(r303) else 3)
    rec = json.loads((DATA / "regression_receipt.json").read_text())
    assert rec["status"] == "PASS", "regression must PASS before any H306 fit"
    set_xgrid(r303)
    use_selective_loader(r303)
    r303.setup()
    if a.phase == "fit":
        r303.fit_phase(a.workers)
    else:
        import evaluate_h306

        locks = [json.loads(p.read_text()) for p in sorted(r303.SCRATCH.glob("fold_*/root_*/fit_lock.json"))]
        evaluate_h306.main(r303, ev, locks)
