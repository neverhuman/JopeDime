#!/usr/bin/env python3
"""H430 end-to-end smoke of cols_h430.py on a synthetic two-context bank (no real table).

  smoke_h430.py --dir DIR      DIR must not exist; the bank and the columns are written there
One context has an exact key, the other has no structure. Checks the column files' shapes, that every value is
finite, that the key context's key_direct column is close to exact out of fold, and that a restart writes nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import keys_core as K  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(430)
    S, AU, labs, ys, cids = [], [], [], [], []
    for t in range(2):
        n, na, f = 600, 250, 9
        X = rng.normal(size=(n + na, f))
        if t == 0:
            g = rng.integers(0, 60, n + na)
            X[:, 3] = g * 7.0
            y = rng.normal(size=60)[g]
        else:
            y = X[:, 0] + 0.5 * rng.normal(size=n + na)
        S.append(np.column_stack([X[:n], y[:n]]).astype(np.float32)); AU.append(X[n:].astype(np.float32))
        labs.append((rng.permutation(n) % 5).astype(np.int8)); ys.append(y[:n].astype(np.float64))
        cids.append(f"{t:064x}")
    bank = d / "bank.npz"
    np.savez(bank, search=np.concatenate([s.ravel() for s in S]), s_shape=np.array([s.shape for s in S], np.int64),
             audit=np.concatenate([x.ravel() for x in AU]), a_shape=np.array([x.shape for x in AU], np.int64),
             lab=np.concatenate(labs), y64=np.concatenate(ys), cids=np.array(cids), family=np.array(["f0", "f1"]),
             cohort=np.array(["SMOKE", "SMOKE"]), ya_sha256=np.array(["-", "-"]), y_sha256=np.array(["-", "-"]))
    cmd = [sys.executable, "-B", str(HERE / "cols_h430.py"), "--bank", str(bank), "--out", str(d / "cols")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    for t, cid in enumerate(cids):
        z = np.load(d / "cols" / f"{cid}.npz")
        assert bool(z["ok"])
        for name in K.NAMES:
            assert z[f"q_{name}"].shape == (600,) and z[f"bag_{name}"].shape == (250,) and z[f"ref_{name}"].shape == (250,)
            assert all(np.all(np.isfinite(z[f"{p}_{name}"])) for p in ("q", "bag", "ref"))
        r = json.loads((d / "cols" / f"{cid}.json").read_text())
        if t == 0:
            assert r["direct_folds"] == 5 and r["oof_u"]["key_direct"] > 4.0, r["oof_u"]
        else:
            assert r["direct_folds"] == 0, r["direct_folds"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    assert json.loads(out.strip().splitlines()[-1])["written"] == 0
    print("SMOKE PASS: 2 contexts, 5 columns each, shapes and finiteness ok, key context exact, restart writes nothing")


if __name__ == "__main__":
    main()
