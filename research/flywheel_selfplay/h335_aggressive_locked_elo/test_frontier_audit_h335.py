"""Synthetic-data tests for the H335 frontier AUDIT evaluator and its runner.

All data is synthetic (linear + Gaussian noise). Every file is written under
``/home/ubuntu/jope-data/h335-cl/frontier_test/`` (override: ``H335_FRONTIER_TEST_DIR``);
each pytest session uses its own ``runs/<stamp>`` subdirectory and nothing is removed.

Binary: ``H335_FRONTIER_BINARY`` (default ``/home/ubuntu/jope-data/h335-cl/build/frontier_audit_h335``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontier_audit_h335 as fa  # noqa: E402

BINARY = Path(os.environ.get(
    "H335_FRONTIER_BINARY", "/home/ubuntu/jope-data/h335-cl/build/frontier_audit_h335"))
TEST_ROOT = Path(os.environ.get(
    "H335_FRONTIER_TEST_DIR", "/home/ubuntu/jope-data/h335-cl/frontier_test"))
ENTRANTS = fa.ENTRANTS
SYNTH_CIDS = ("synth_lin_a", "synth_lin_b", "synth_lin_c")


# --------------------------------------------------------------------------- test helper

def write_jtt1(path, X, y, folds, nq=0, flag=2, query_rows=None):
    """Write a JTT1 envelope: 17-byte header (b"JTT1", u32 nt, u32 nq, u32 width, u8 flag),
    u8 fold[nt], then f64 row-major (nt + nq) x (width + 1) with the target in the last column."""
    nt = len(X)
    width = len(X[0])
    assert len(y) == nt and len(folds) == nt
    buf = bytearray(b"JTT1")
    buf += struct.pack("<III", nt, nq, width)
    buf += bytes([flag])
    buf += bytes(folds)
    for row, t in zip(X, y):
        buf += struct.pack(f"<{width + 1}d", *row, t)
    for q in range(nq):
        row = query_rows[q] if query_rows else [0.0] * (width + 1)
        buf += struct.pack(f"<{width + 1}d", *row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path


def synth_linear(seed, n=200, width=5, noise=0.3, audit_frac=0.3):
    rng = random.Random(seed)
    beta = [rng.uniform(-2.0, 2.0) for _ in range(width)]
    X = [[rng.gauss(0.0, 1.0) for _ in range(width)] for _ in range(n)]
    y = [1.5 + sum(b * x for b, x in zip(beta, row)) + rng.gauss(0.0, noise) for row in X]
    order = list(range(n))
    rng.shuffle(order)
    n_audit = int(round(audit_frac * n))
    folds = [0] * n
    for i in order[:n_audit]:
        folds[i] = 1
    return X, y, folds


def expected_utility(y, folds, preds):
    """Recompute the H335 AUDIT utility exactly as the harness does (same summation order)."""
    y_s = [t for t, f in zip(y, folds) if f == 0]
    y_a = [t for t, f in zip(y, folds) if f == 1]
    s = 0.0
    for t in y_s:
        s += t
    mean = s / len(y_s)
    base_sse = 0.0
    for t in y_a:
        base_sse += (t - mean) * (t - mean)
    base = math.sqrt(base_sse / len(y_a))
    sse = 0.0
    for p, t in zip(preds, y_a):
        sse += (p - t) * (p - t)
    rmse = math.sqrt(sse / len(y_a))
    ratio = max(rmse / max(base, 1e-12), 1e-12)
    return rmse, base, ratio, -math.log(ratio)


def run_binary(env_path, entrant, out_path, dump=None):
    cmd = [str(BINARY), str(env_path), entrant, str(out_path)]
    if dump is not None:
        cmd += ["--dump-preds", str(dump)]
    env = dict(os.environ, OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    return subprocess.run(cmd, capture_output=True, env=env, timeout=600)


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def run_dir():
    assert BINARY.is_file(), f"binary missing: {BINARY}"
    d = TEST_ROOT / "runs" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    d.mkdir(parents=True, exist_ok=False)
    return d


@pytest.fixture(scope="session")
def synth(run_dir):
    envs = run_dir / "envs"
    data = {}
    for k, cid in enumerate(SYNTH_CIDS):
        X, y, folds = synth_linear(seed=335 + k, width=5 + k)
        write_jtt1(envs / f"{cid}.jtt1", X, y, folds)
        data[cid] = (X, y, folds)
    return envs, data


@pytest.fixture(scope="session")
def binary_sha():
    return fa.sha256_file(BINARY)


# --------------------------------------------------------------------------- tests

def test_a_every_entrant_completes_via_runner(run_dir, synth, binary_sha):
    envs, _ = synth
    out = run_dir / "runner_a.jsonl"
    summary = fa.run(envs, SYNTH_CIDS, out, BINARY, cpu_limit_s=300, wall_limit_s=900,
                     jobs=4, binary_sha256=binary_sha)
    assert summary["ran"] == len(SYNTH_CIDS) * len(ENTRANTS)
    recs = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert {(r["cid"], r["entrant"]) for r in recs} == {
        (c, e) for c in SYNTH_CIDS for e in ENTRANTS}
    for r in recs:
        assert r["status"] == "completed", r
        assert r["seed"] == 42 and r["error"] is None
        assert r["n_search"] == 140 and r["n_audit"] == 60
        assert math.isfinite(r["utility"]) and r["utility"] > 0.5, r  # signal is strong
        assert r["binary_sha256"] == binary_sha
    # resumable: a second run skips every pair and appends nothing
    size = out.stat().st_size
    again = fa.run(envs, SYNTH_CIDS, out, BINARY, jobs=4, binary_sha256=binary_sha)
    assert again["ran"] == 0 and again["skipped"] == len(SYNTH_CIDS) * len(ENTRANTS)
    assert out.stat().st_size == size


def test_b_two_runs_byte_identical(run_dir, synth):
    envs, _ = synth
    for cid in SYNTH_CIDS:
        for e in ENTRANTS:
            outs, dumps = [], []
            for rep in (1, 2):
                o = run_dir / "b" / f"{cid}.{e}.r{rep}.json"
                d = run_dir / "b" / f"{cid}.{e}.r{rep}.f64"
                o.parent.mkdir(parents=True, exist_ok=True)
                p = run_binary(envs / f"{cid}.jtt1", e, o, dump=d)
                assert p.returncode == 0, p.stderr
                # wall_ms is the only timing-dependent field; compare all other bytes.
                outs.append(re.sub(rb'"wall_ms":[^,]*,', b'"wall_ms":X,', o.read_bytes()))
                dumps.append(d.read_bytes())
            assert outs[0] == outs[1], (cid, e)
            assert dumps[0] == dumps[1], (cid, e)


def test_c_utility_recomputed_from_dumped_predictions(run_dir, synth):
    envs, data = synth
    for cid in SYNTH_CIDS:
        _, y, folds = data[cid]
        for e in ENTRANTS:
            o = run_dir / "c" / f"{cid}.{e}.json"
            d = run_dir / "c" / f"{cid}.{e}.f64"
            o.parent.mkdir(parents=True, exist_ok=True)
            p = run_binary(envs / f"{cid}.jtt1", e, o, dump=d)
            assert p.returncode == 0, p.stderr
            rec = json.loads(o.read_text())
            raw = d.read_bytes()
            preds = list(struct.unpack(f"<{len(raw) // 8}d", raw))
            assert len(preds) == rec["n_audit"] == sum(folds)
            assert hashlib.sha256(raw).hexdigest() == rec["audit_preds_sha256"]
            rmse, base, ratio, u = expected_utility(y, folds, preds)
            assert abs(rec["utility"] - u) <= 1e-12, (cid, e, rec["utility"], u)
            assert abs(rec["rmse_audit"] - rmse) <= 1e-12
            assert abs(rec["base_rmse_audit"] - base) <= 1e-12
            assert abs(rec["ratio"] - ratio) <= 1e-12
            assert rec["cid"] == cid and rec["entrant"] == e
            assert rec["harness_version"] == "h335-frontier-audit-v1"


def test_d_nq_positive_is_refused(run_dir, binary_sha):
    X, y, folds = synth_linear(seed=9, n=50)
    env = write_jtt1(run_dir / "envs_nq" / "synth_nq.jtt1", X, y, folds, nq=3)
    out = run_dir / "d" / "synth_nq.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    p = run_binary(env, "Opus5p5_Extra", out)
    assert p.returncode == 3 and b"nq = 3" in p.stderr
    assert not out.exists()
    # fold byte outside {0, 1} and a non-2 flag are refused as well
    bad_fold = write_jtt1(run_dir / "envs_nq" / "synth_fold2.jtt1", X, y, [2] + folds[1:])
    assert run_binary(bad_fold, "Opus5p5_Extra", out).returncode == 3
    bad_flag = write_jtt1(run_dir / "envs_nq" / "synth_flag.jtt1", X, y, folds, flag=1)
    assert run_binary(bad_flag, "Opus5p5_Extra", out).returncode == 3
    # the runner records nothing for a refused envelope and raises
    jl = run_dir / "d" / "runner_d.jsonl"
    with pytest.raises(fa.EnvelopeRefused) as ei:
        fa.run(run_dir / "envs_nq", ["synth_nq"], jl, BINARY, binary_sha256=binary_sha)
    assert len(ei.value.summary["refused"]) == len(ENTRANTS)
    assert jl.read_bytes() == b""


def test_e_degenerate_cases_fail_cleanly(run_dir):
    X, _, folds = synth_linear(seed=11, n=120)
    cases = {
        "synth_const": ([3.25] * len(X), folds),       # constant target -> base RMSE 0
        "synth_noaudit": ([float(i) for i in range(len(X))], [0] * len(X)),  # empty AUDIT
        "synth_nosearch": ([float(i) for i in range(len(X))], [1] * len(X)),  # empty SEARCH
    }
    for cid, (y, fo) in cases.items():
        env = write_jtt1(run_dir / "envs_degen" / f"{cid}.jtt1", X, y, fo)
        for e in ENTRANTS:
            o = run_dir / "e" / f"{cid}.{e}.json"
            o.parent.mkdir(parents=True, exist_ok=True)
            p = run_binary(env, e, o)
            assert p.returncode == 0, (cid, e, p.stderr)
            rec = json.loads(o.read_text())
            assert rec["status"] == "failed" and rec["utility"] == 0.0, rec
            assert rec["error"] and rec["rmse_audit"] is None and rec["ratio"] is None


def test_f_runner_timeout_paths(run_dir, binary_sha):
    # A larger synthetic envelope that needs well over 1 s CPU for Opus5p5_Extra.
    X, y, folds = synth_linear(seed=21, n=20000, width=24)
    envs = run_dir / "envs_big"
    write_jtt1(envs / "synth_big.jtt1", X, y, folds)
    jl = run_dir / "f" / "runner_cpu.jsonl"
    s = fa.run(envs, ["synth_big"], jl, BINARY, cpu_limit_s=1, wall_limit_s=600, jobs=1,
               binary_sha256=binary_sha, entrants=["Opus5p5_Extra"])
    rec = json.loads(jl.read_text().splitlines()[-1])
    assert s["status_counts"] == {"timeout": 1}
    assert rec["status"] == "timeout" and rec["utility"] == 0.0
    assert "cpu limit" in rec["error"] and rec["cpu_s"] >= 0.5
    assert rec["n_search"] == 14000 and rec["n_audit"] == 6000
    # wall-clock path
    jl2 = run_dir / "f" / "runner_wall.jsonl"
    fa.run(envs, ["synth_big"], jl2, BINARY, cpu_limit_s=300, wall_limit_s=0.3, jobs=1,
           binary_sha256=binary_sha, entrants=["Opus5p5_Extra"])
    rec2 = json.loads(jl2.read_text().splitlines()[-1])
    assert rec2["status"] == "timeout" and rec2["utility"] == 0.0
    assert "wall limit" in rec2["error"]


def test_g_runner_rejects_unpinned_binary(run_dir, synth):
    envs, _ = synth
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fa.run(envs, SYNTH_CIDS[:1], run_dir / "g.jsonl", BINARY, binary_sha256="0" * 64)
    assert not (run_dir / "g.jsonl").exists()
