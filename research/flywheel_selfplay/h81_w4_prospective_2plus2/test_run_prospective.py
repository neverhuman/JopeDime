"""H81 structural tests: tiny synthetic fixtures only (no data payloads, models or native calls)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_prospective as h  # noqa: E402


def fixture(n=16, seed=0):
    rng = np.random.default_rng(seed)
    ids = [hashlib.sha256(f"p{i}".encode()).hexdigest() for i in range(n)]
    return rng.normal(size=n), rng.normal(size=(n, 16)), ids


def test_rules_match_merged_h79_exactly():
    path = h.HERE.parent / "h79_w4_prior_2plus2_cached" / "run_shard.py"
    if not path.exists():
        pytest.skip("H79 not present in this tree")
    src = path.read_text()
    ns = {}
    for name in ("ranked", "lengthscale", "posterior_mean"):  # execute H79's own definitions, not a copy
        start = src.index(f"def {name}(")
        end = src.index("\ndef ", start + 1)
        exec(compile("import math\nimport numpy as np\nNOISE_RATIO = 1.0\n" + src[start:end], str(path), "exec"), ns)
    mu, x, ids = fixture()
    assert ns["ranked"](mu, ids) == h.ranked(mu, ids)
    assert ns["lengthscale"](x) == h.lengthscale(x)
    shown, res = [3, 7], [0.4, -1.1]
    assert np.array_equal(ns["posterior_mean"](mu, x, shown, res, 1.7), h.posterior_mean(mu, x, shown, res, 1.7))


def test_two_plus_two_first_pair_is_prior_top_two_and_zero_residual_reduces_to_top4():
    mu, x, ids = fixture()
    top4 = h.first_stage("gbm_top4", mu, ids, 1)
    first = h.first_stage("gbm_2plus2", mu, ids, 1)
    assert first == top4[:2]
    second = h.second_stage("gbm_2plus2", mu, x, ids, first, [float(mu[i]) for i in first], h.lengthscale(x))
    assert first + second == top4  # no surprise -> same four as the nonadaptive prior


def test_swap_placebo_reverses_residuals_and_random_is_seeded():
    mu, x, ids = fixture(seed=3)
    first = h.first_stage("gbm_2plus2", mu, ids, 1)
    fu = [float(mu[first[0]]) + 2.0, float(mu[first[1]]) - 2.0]
    a = h.second_stage("gbm_2plus2", mu, x, ids, first, fu, 1.0)
    b = h.second_stage("gbm_2plus2_swap", mu, x, ids, first, fu, 1.0)
    assert len(set(a)) == len(set(b)) == 2 and not set(a) & set(first) and not set(b) & set(first)
    r1, r2 = h.first_stage("random4", mu, ids, 11), h.first_stage("random4", mu, ids, 11)
    assert r1 == r2 and len(set(r1)) == 4 and h.first_stage("random4", mu, ids, 12) != r1


def test_ranking_breaks_ties_by_program_sha():
    ids = ["b", "a", "c"]
    assert h.ranked(np.array([1.0, 1.0, 0.0]), ids) == [1, 0, 2]


def test_utility_keeps_execution_errors_out():
    assert h.utility("completed", True, -1.5) == 1.5
    assert h.utility("completed", False, 100.3) == -2.0
    assert h.utility("failed", None, None) is None and h.utility("timeout", None, None) is None


def row(c, fam, seed, arm, v, status="completed"):
    return dict(context=c, family=fam, seed=seed, arm=arm, status=status, retained_best=v if status == "completed" else None,
                revealed=["x"] * 4, native_wall_s=1.0, native_cpu_s=1.0, selection_cpu_s=0.01)


def test_analyze_gate_and_incomplete_verdict():
    rows = []
    for i in range(60):
        c, f = f"c{i}", f"f{i % 50}"
        for s in h.SEEDS:
            base = 1.0 + 0.01 * i
            for arm, off in (("random4", -0.3), ("gbm_top4", 0.0), ("gbm_2plus2", 0.05), ("gbm_2plus2_swap", 0.0)):
                rows.append(row(c, f, s, arm, base + off))
    res = h.analyze(rows)
    assert res["primary"]["family_mean"] == pytest.approx(0.05) and res["primary"]["common_gate_passed"]
    assert res["verdict"] == "ADVANCE" and res["families"] == 50
    rows[0] = row("c0", "f0", 701, "random4", None, status="incomplete")
    assert h.analyze(rows)["verdict"] == "INCOMPLETE"
    flat = [dict(r, retained_best=1.0) if r["status"] == "completed" else r for r in rows[1:]]
    assert h.analyze(flat)["primary"]["common_gate_passed"] is False  # zero effect cannot pass +0.001


def test_ledger_reserves_before_launch_and_caps(tmp_path):
    led = h.Ledger(tmp_path / "l.json", cap=3, deadline="2099-01-01T00:00:00Z")
    led.reserve(3)
    with pytest.raises(SystemExit):
        led.reserve(1)
    led.settle(dict(cpu_s=1.0, wall_s=2.0, status="timeout"))
    assert led.state["fits"] == 3 and led.state["settled"] == 1 and led.state["timeouts"] == 1
    with pytest.raises(SystemExit):
        h.Ledger(tmp_path / "l.json", cap=4, deadline="2099-01-01T00:00:00Z").reserve(1)  # cap changed on resume
    with pytest.raises(SystemExit):
        h.Ledger(tmp_path / "p.json", cap=3, deadline="2000-01-01T00:00:00Z").reserve(1)


def test_fit_store_dedupes_resumes_without_recharge_and_refuses_foreign_run(tmp_path, monkeypatch):
    calls = []

    def fake(context, dsl, scratch, timeout_s=600):
        calls.append(dsl)
        return dict(rc=0, wall_s=0.1, cpu_s=0.1, input_sha256="i", status="completed", fitness=-1.0, logratio=-1.0, gate=True)
    monkeypatch.setattr(h, "native_fit", fake)
    led = h.Ledger(tmp_path / "l.json", cap=10, deadline="2099-01-01T00:00:00Z")
    st = h.FitStore(tmp_path / "f.jsonl", "run1")
    out = st.run([("c", "(pipe X (ridge))"), ("c", "(pipe X (ridge))"), ("c", "(pipe X (knn))")], tmp_path, led, 2)
    assert len(calls) == 2 and len(out) == 2 and led.state["fits"] == 2
    h.FitStore(tmp_path / "f.jsonl", "run1").run([("c", "(pipe X (ridge))")], tmp_path, led, 2)
    assert len(calls) == 2 and led.state["fits"] == 2
    with pytest.raises(SystemExit):
        h.FitStore(tmp_path / "f.jsonl", "run2")


def test_envelope_pads_queries_and_rejects_query_rows():
    import struct
    raw = b"JTT1" + struct.pack("<III", 4, 0, 1) + bytes([2]) + bytes([0, 1, 0, 1]) + np.zeros(8, np.float64).tobytes()
    out = h.envelope(raw)
    assert struct.unpack_from("<I", out, 8)[0] == 10 and out[:len(raw)][17:] == raw[17:]
    bad = bytearray(raw); struct.pack_into("<I", bad, 8, 1)
    with pytest.raises(ValueError):
        h.envelope(bytes(bad))
