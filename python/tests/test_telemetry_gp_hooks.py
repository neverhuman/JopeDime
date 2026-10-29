"""GP generation recorder: marshalling of flywheel evaluator results (rows G1-G12, E2)."""

from __future__ import annotations

import hashlib
import json
import math

from jope.telemetry import gp_hooks as h


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def summarize_json(self, generation_json: str) -> str:
        g = json.loads(generation_json)
        self.calls.append(g)
        return json.dumps({"schema": h.GEN_KIND, "pop_size": len(g["candidates"])})


def result(dsl: str, **kw):
    r = {
        "dsl": dsl,
        "operator": "edit_parameter",
        "status": "completed",
        "fitness": -1.0,
        "cv_gate_passed": True,
        "cv_mean_log_ratio": -0.5,
        "cv_tail_log_ratio": -0.2,
        "cv_log_ratio_sd": 0.1,
        "test_rmse_ratio": 0.5,
        "complexity_open_parens": 4,
        "proj16": [1.0 if i % 2 else -1.0 for i in range(16)],
    }
    r.update(kw)
    return r


def test_marshals_result_dicts_and_resets_after_finish() -> None:
    s = FakeSession()
    emitted = []
    bank_dsl = "(pipe X (b) (h))"
    bank_sha = hashlib.sha256(bank_dsl.encode()).hexdigest()
    rec = h.GenerationRecorder(s, "ctx", 2, emit=lambda k, v: emitted.append((k, v)), bank_shas=[bank_sha])
    rec.proposals(10, 3)
    rec.add(result("(pipe X (a) (h))"), parent_dsl="(pipe X (h))", improved=True)
    rec.add(result(bank_dsl, status="failed", fitness=100.0, cv_gate_passed=False, test_rmse_ratio=None))
    rec.add(result("(pipe X (c) (h))", status="failed", rc=-1))
    out = rec.finish(5)

    g = s.calls[0]
    assert (g["context"], g["round"], g["generation"]) == ("ctx", 2, 5)
    assert (g["proposals"], g["seen_rejects"]) == (10, 3)
    c0, c1, c2 = g["candidates"]
    assert c0["provenance"] == "novel" and c1["provenance"] == "bank"
    assert c0["proj16_bits"] == sum(1 << i for i in range(1, 16, 2))
    assert c0["behavior"][:3] == [-0.5, -0.2, 0.1] and math.isclose(c0["behavior"][3], math.log(0.5))
    assert c1["behavior"][3] is None
    assert (c1["status"], c2["status"]) == ("failed", "timeout")
    assert c0["improved"] is True and c1["improved"] is None
    assert out["pop_size"] == 3 and emitted == [(h.GEN_KIND, out)]

    rec.finish(6)
    assert s.calls[1]["candidates"] == [] and s.calls[1]["proposals"] == 0


def test_status_and_bits_edge_cases() -> None:
    assert h.status_of({"status": "failed", "rc": 2}) == "nonzero_rc"
    assert h.status_of({"status": "failed", "rc": 0}) == "parse_fail"
    assert h.status_of({}) == "failed"
    assert h.proj16_bits(None) is None and h.proj16_bits([1.0] * 15) is None
    assert h.error_profile({"cv_mean_log_ratio": float("nan")})[0] is None
