"""Targeted tests for the H72 W8 control runner (no native fits, no data payloads)."""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w8_control as w  # noqa: E402


def packet(nt=10, width=3):
    folds = bytes([i % 2 for i in range(nt)])
    rows = np.arange(nt * (width + 1), dtype=np.float64).tobytes()
    return b"JTT1" + struct.pack("<III", nt, 0, width) + bytes([2]) + folds + rows


def test_envelope_original_keeps_training_bytes_and_pads_queries():
    raw = packet()
    out = w.envelope(raw, None)
    nt, nq, width = struct.unpack_from("<III", out, 4)
    assert (nt, nq, width) == (10, 10, 3)
    assert out[17:len(raw)] == raw[17:]
    assert out[len(raw):] == bytes(10 * (width + 1) * 8)


def test_envelope_resplit_changes_only_fold_bytes_and_keeps_balance():
    raw = packet(nt=200)
    a, b = w.envelope(raw, 701), w.envelope(raw, 701)
    assert a == b  # deterministic
    folds = np.frombuffer(a, np.uint8, 200, 17)
    assert np.bincount(folds).tolist() == [100, 100]
    assert a[17 + 200:] == w.envelope(raw, None)[17 + 200:]  # rows and padding untouched
    assert a[17:217] != raw[17:217]
    assert w.envelope(raw, 702)[17:217] != a[17:217]


def test_envelope_rejects_query_rows():
    raw = bytearray(packet())
    struct.pack_into("<I", raw, 8, 1)
    with pytest.raises(ValueError):
        w.envelope(bytes(raw), None)


def test_s1_metrics_matches_live_formula():
    cv = [{"metrics": {"rmse": 0.5}, "baseline_metrics": {"rmse": 1.0}},
          {"metrics": {"rmse": 0.6}, "baseline_metrics": {"rmse": 1.0}}]
    r = w.s1_metrics("(pipe X (ridge))", cv)
    lr0, lr1 = math.log(0.5), math.log(0.6)
    expect = 0.8 * (lr0 + lr1) / 2 + 0.1 * max(lr0, lr1) + 0.1 * abs(lr0 - lr1) / 2 + 0.0005 * 2
    assert r["cv_gate_passed"] and abs(r["fitness"] - expect) < 1e-12
    assert w.utility(dict(r, status="completed")) == pytest.approx(-expect)
    assert w.utility({"status": "completed", "cv_gate_passed": False, "fitness": 100.1}) == -2.0


def test_simulate_budget_prefix_and_decisions():
    u_inc, inc_rs = 1.0, [1.0, 1.0, 1.0]
    stream = [0.5, 1.2, 0.9, 1.3, 0.8, 1.4] + [0.0] * 34  # wins at 1, 3, 5
    verify = {1: [1.1, 1.1, 1.1], 3: [0.9, 0.9, 0.9], 5: [1.2, 0.8, 1.3]}
    s = w.simulate(stream, u_inc, verify, inc_rs, budget=40)
    assert s["p0_promote"] == [1, 3, 5]  # P0 promotes every apparent win
    assert s["p1_apparent"] == [1, 3, 5]
    assert s["p1_promote"] == [1, 5]  # mean resplit delta > 0 only for 1 and 5
    assert s["p1_spent"] == s["p1_evaluated"] + 3 * 3 <= 40
    assert s["p1_evaluated"] == 40 - 9  # P1 evaluates fewer proposals at equal cost
    assert s["verify_needed"] == []


def test_simulate_lists_unverified_wins_and_respects_tight_budget():
    s = w.simulate([1.5, 1.5, 1.5], 1.0, {}, [1, 1, 1], budget=6)
    assert s["verify_needed"] == [0]  # second win cannot afford 1 + 3 within 6
    assert s["p1_spent"] <= 6 and s["p0_promote"] == [0, 1, 2]


def test_confirmed_is_strict_mean_difference():
    assert w.confirmed([1.1, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert not w.confirmed([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


def test_frozen_incumbent_is_best_original_cut_program(tmp_path):
    plan = {"pairs": [
        {"context": "c1", "family": "f1", "incumbent_dsl": "I", "incumbent_sha": "si", "challenger_dsl": "A", "challenger_sha": "sa"},
        {"context": "c1", "family": "f1", "incumbent_dsl": "I", "incumbent_sha": "si", "challenger_dsl": "B", "challenger_sha": "sb"}]}
    rows = []
    for i, ch_u in enumerate((1.0, 1.4)):
        for role, u in (("incumbent", 1.2), ("challenger", ch_u)):
            for seed in (None, 701, 702, 703):
                rows.append({"pair_index": i, "role": role, "seed": seed, "status": "completed",
                             "cv_gate_passed": True, "fitness": -u})
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "res.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    inc = w.frozen_incumbents(tmp_path / "res.jsonl", tmp_path / "plan.json")
    assert inc["c1"]["sha"] == "sb" and inc["c1"]["u_orig"] == pytest.approx(1.4)
    assert inc["c1"]["u_701_703"] == pytest.approx([1.4, 1.4, 1.4])
