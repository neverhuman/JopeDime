#!/usr/bin/env python3
"""End-to-end synthetic smoke check of cols_matern.py + _safe_svd + stack_matern imports (no real table)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard_matern_core as GM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=False)

    # Check _safe_svd on rank-deficient / collinear / near-singular matrices
    rng = np.random.default_rng(434)
    A = rng.normal(size=(120, 40))
    A[:, 1] = A[:, 0]
    A[:, 2] = A[:, 0] * 1e-15
    U, s, Vt = GM._safe_svd(A)
    assert U.shape == (120, 40) and s.shape == (40,) and Vt.shape == (40, 40)
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(s)) and np.all(np.isfinite(Vt))

    S, AU, labs, ys, cids = [], [], [], [], []
    for t in range(2):
        n, na, f = 220, 90, 8
        X = rng.normal(size=(n + na, f))
        if t == 0:
            g = rng.integers(0, 18, n + na)
            X[:, 2] = g * 5.0
            raw = rng.normal(size=18)[g] + 0.5 * np.sin(X[:, 0]) + 0.1 * rng.normal(size=n + na)
            y = np.exp(0.6 * raw)
        else:
            y = 1.2 * X[:, 0] - 0.8 * X[:, 1] ** 2 + 0.2 * rng.normal(size=n + na)
        S.append(np.column_stack([X[:n], y[:n]]).astype(np.float32))
        AU.append(X[n:].astype(np.float32))
        labs.append((rng.permutation(n) % 5).astype(np.int8))
        ys.append(y[:n].astype(np.float64))
        cids.append(f"{t:064x}")
    bank = d / "bank.npz"
    np.savez(
        bank,
        search=np.concatenate([s.ravel() for s in S]),
        s_shape=np.array([s.shape for s in S], np.int64),
        audit=np.concatenate([x.ravel() for x in AU]),
        a_shape=np.array([x.shape for x in AU], np.int64),
        lab=np.concatenate(labs),
        y64=np.concatenate(ys),
        cids=np.array(cids),
        family=np.array(["f0", "f1"]),
        cohort=np.array(["SMOKE", "SMOKE"]),
        ya_sha256=np.array(["-", "-"]),
        y_sha256=np.array(["-", "-"]),
    )
    cmd = [sys.executable, "-B", str(HERE / "cols_matern.py"), "--bank", str(bank), "--out", str(d / "cols")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    for cid in cids:
        z = np.load(d / "cols" / f"{cid}.npz")
        assert bool(z["ok"])
        for name in GM.NAMES:
            assert z[f"q_{name}"].shape == (220,)
            assert z[f"bag_{name}"].shape == (90,)
            assert z[f"ref_{name}"].shape == (90,)
            assert all(np.all(np.isfinite(z[f"{p}_{name}"])) for p in ("q", "bag", "ref"))
        r = json.loads((d / "cols" / f"{cid}.json").read_text())
        assert r["ok"] is True
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    assert json.loads(out.strip().splitlines()[-1])["written"] == 0
    print("SMOKE_MATERN_OK 2 contexts x 9 columns finite, restart idempotent")


if __name__ == "__main__":
    main()
