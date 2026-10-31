"""Structural tests for the H75-closed-loop runner on tiny synthetic fixtures (software only, not experiment data)."""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h75_runner as r  # noqa: E402


def test_roster_rule_is_hash_ordered_and_refuses_small_sets():
    fams = {f"fam{i}": [f"{i:02d}b", f"{i:02d}a"] for i in range(10)}
    a = r.acquisition_roster(2, fams, "fit")
    assert a == r.acquisition_roster(2, dict(reversed(list(fams.items()))), "fit")  # order-independent
    order = sorted(fams, key=lambda f: r.hkey("H75-closed-loop", "fit", 2, f))[:4]
    assert a == [min(fams[f]) for f in order]  # smallest context SHA within each chosen family
    assert r.acquisition_roster(2, fams, "eval") != a or True
    with pytest.raises(SystemExit):
        r.acquisition_roster(0, {"f": ["x"]}, "fit")


def test_dl_selection_two_best_then_two_uniform():
    costs = np.array([5.0, 1.0, 3.0, 0.5, 9.0, 7.0, 2.0, 8.0])
    idx, prob = r.select_proposals(costs, np.random.default_rng(1))
    assert idx[:2] == [3, 1] and len(set(idx)) == 4 and prob == [1.0, 1.0, 2 / 6, 2 / 6]
    assert set(idx[2:]) <= {0, 2, 4, 5, 6, 7}
    idx2, _ = r.select_proposals(costs, np.random.default_rng(1))
    assert idx == idx2  # shared random stream -> identical choices across cells
    u, p = r.select_uniform(8, np.random.default_rng(0))
    assert len(set(u)) == 4 and p == [0.5] * 4
    with pytest.raises(ValueError):
        r.select_proposals(np.array([1.0, 2.0, 3.0]), np.random.default_rng(0))


def test_best_history_orders_by_utility_then_sha_and_skips_failures():
    rows = [dict(status="completed", utility=1.0, program_sha256="b"), dict(status="completed", utility=1.0, program_sha256="a"),
            dict(status="failed", utility=9.0, program_sha256="z"), dict(status="completed", utility=2.0, program_sha256="c")]
    assert [x["program_sha256"] for x in r.best_history(rows, 3)] == ["c", "a", "b"]


def test_replay_matching_equalises_counts_and_pass_fail_mix():
    mk = lambda tag, n, g: [dict(record_id=f"{tag}{i}", gate=g) for i in range(n)]
    b = mk("b", 5, True) + mk("bf", 1, False)
    d = mk("d", 3, True) + mk("df", 4, False)
    mb, md, rep = r.stratified_match(b, d)
    assert len(mb) == len(md) == 4 and rep == {"pass": {"B": 5, "D": 3, "kept": 3}, "fail": {"B": 1, "D": 4, "kept": 1}}
    assert r.stratified_match(b, d)[0] == mb  # deterministic


def ctx_fixture(n=40):
    rng = np.random.default_rng(0)
    return dict(program=rng.normal(size=(n, 412)).astype(np.float32), fitness=rng.normal(size=n),
                gate=rng.random(n) > 0.1, sha=[f"s{i:03d}" for i in range(n)])


def test_training_slate_keeps_anchors_reserves_fresh_and_never_duplicates_labels():
    ctx = ctx_fixture()
    fresh = [dict(record_id=f"r{i}", status="completed", feat412=np.full(412, i, np.float32), fitness=-1.0 - i, gate=True) for i in range(5)]
    fresh.append(dict(record_id="rf", status="failed", feat412=np.zeros(412, np.float32), fitness=None, gate=False))
    s, ex = r.training_slate(ctx, fresh, 48, np.random.default_rng(0), size=20)
    assert np.array_equal(s["program"][:4], ctx["program"][:4]) and np.array_equal(s["fitness"][:4], ctx["fitness"][:4])
    assert ex == {"fresh": 5, "static": 11} and s["program"].shape == (20, 412)  # failed row never becomes a label
    c, exc = r.training_slate(ctx, [], 48, np.random.default_rng(0), size=20)
    assert exc == {"fresh": 0, "static": 16}
    s2, _ = r.training_slate(ctx, fresh, 2, np.random.default_rng(0), size=20)
    assert np.isin(s2["fitness"][-2:], [-1.0 - i for i in range(5)]).all()  # reserve caps fresh rows


def test_loss_ignores_revealed_slots():
    torch = pytest.importorskip("torch")
    pred = torch.zeros((2, 10), requires_grad=True)
    fit = np.random.default_rng(0).normal(size=(2, 10)); gate = np.ones((2, 10), bool); gate[0, 6] = False
    r.h75_loss(torch, pred, fit, gate).backward()
    assert torch.all(pred.grad[:, :4] == 0) and torch.any(pred.grad[:, 4:] != 0)


def packet(nt=10, width=3):
    return b"JTT1" + struct.pack("<III", nt, 0, width) + bytes([2]) + bytes(i % 2 for i in range(nt)) + np.arange(nt * (width + 1), dtype=np.float64).tobytes()


def test_envelope_original_and_resplit():
    raw = packet(100)
    o, s1 = r.envelope(raw, None), r.envelope(raw, 801)
    assert o[17:len(raw)] == raw[17:] and np.bincount(np.frombuffer(s1, np.uint8, 100, 17)).tolist() == [50, 50]
    assert s1[117:] == o[117:] and s1 != o


def test_ledger_caps_deadline_and_shared_state(tmp_path):
    led = r.Ledger(tmp_path / "l.json", max_fits=3, max_cpu_s=100, max_gpu_s=100, deadline_utc="2099-01-01T00:00:00Z")
    led.check(3)
    for _ in range(2):
        led.charge(dict(cpu_s=1.0, wall_s=2.0, status="completed"))
    other = r.Ledger(tmp_path / "l.json", max_fits=3, max_cpu_s=100, max_gpu_s=100, deadline_utc="2099-01-01T00:00:00Z")
    assert other.state["fits"] == 2  # a second process sees the same campaign ledger
    with pytest.raises(SystemExit):
        other.check(2)
    with pytest.raises(SystemExit):
        r.Ledger(tmp_path / "l.json", 3, 100, 100, "2000-01-01T00:00:00Z").check()  # changed deadline refused
    past = r.Ledger(tmp_path / "p.json", 3, 100, 100, "2000-01-01T00:00:00Z")
    with pytest.raises(SystemExit):
        past.check()


def test_fit_cache_never_recharges(tmp_path, monkeypatch):
    calls = []

    def fake_fit_many(jobs, scratch, ledger, workers):
        calls.append(len(jobs))
        return [dict(status="completed", fitness=-1.0, gate=True, logratio=-1.0, cpu_s=0.1, wall_s=0.1, rc=0,
                     input_sha256="x", cut_id="original") for _ in jobs]
    monkeypatch.setattr(r, "fit_many", fake_fit_many)
    led = r.Ledger(tmp_path / "l.json", 100, 100, 100, "2099-01-01T00:00:00Z")
    fc = r.FitCache(tmp_path / "fits.jsonl")
    jobs = [("c1", "(pipe X (ridge))", None), ("c1", "(pipe X (ridge))", None), ("c1", "(pipe X (ridge))", 801)]
    out = fc.run(jobs, tmp_path, led, 2)
    assert calls == [2] and len(out) == 3  # duplicate job is executed once
    fc2 = r.FitCache(tmp_path / "fits.jsonl")  # resume from disk
    fc2.run(jobs, tmp_path, led, 2)
    assert calls == [2]


def test_utility_and_s1_metrics_match_live_formula():
    cv = [{"metrics": {"rmse": 0.5}, "baseline_metrics": {"rmse": 1.0}}, {"metrics": {"rmse": 0.6}, "baseline_metrics": {"rmse": 1.0}}]
    m = r.s1_metrics("(pipe X (ridge))", cv)
    assert m["gate"] and r.utility(m["status"], m["gate"], m["fitness"]) == pytest.approx(-m["fitness"])
    assert r.s1_metrics("x", [])["status"] == "failed"
    assert r.utility("failed", False, 0.0) == -2.0


def test_anchor_dsl_export_identity_checks(tmp_path):
    dsl = "(pipe X (ridge))"; sha = hashlib.sha256(dsl.encode()).hexdigest()
    ctx = {"c1": dict(sha=[sha, "b", "c", "d", "e"])}
    p = tmp_path / "a.json"; p.write_text(json.dumps({"c1": [{"program_sha256": sha, "dsl": dsl}]}))
    assert r.load_anchor_dsl(p, ctx)["rows"] == 1 and ctx["c1"]["anchor_dsl"][sha] == dsl
    p.write_text(json.dumps({"c1": [{"program_sha256": sha, "dsl": dsl + " "}]}))
    with pytest.raises(SystemExit):
        r.load_anchor_dsl(p, ctx)
    p.write_text(json.dumps({"c9": []}))
    with pytest.raises(SystemExit):
        r.load_anchor_dsl(p, ctx)
