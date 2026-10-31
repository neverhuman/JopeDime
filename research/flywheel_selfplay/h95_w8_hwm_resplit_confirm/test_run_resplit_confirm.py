"""H95 structural tests on synthetic fixtures (no data payloads or native calls)."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_resplit_confirm as h  # noqa: E402

DSL_I, DSL_A, DSL_B = "(add x0 x1)", "(mul x0 x1)", "(sub x0 x1)"
S = lambda d: hashlib.sha256(d.encode()).hexdigest()
CHAMP = {"c1": dict(dsl=DSL_I, program_sha256=S(DSL_I))}
HWM = [dict(context_sha256="c1", family_id="f1", program_sha256=S(d), dsl=d, found_by="PRIOR_top4",
            fitness_original_cut=-1.1, pool_best_fitness=-1.0) for d in (DSL_A, DSL_B)]


def fake_results(gain_by_sha: dict, fail=None, logratio_same=False):
    res = {}
    for role, sha, base in (("inc", S(DSL_I), 1.0), *[("chal", s, 1.0 + g) for s, g in gain_by_sha.items()]):
        for s in h.CUTS:
            u = 1.1 if (role == "chal" and s is None) else base
            r = dict(status="completed", cv_gate_passed=True, fitness=-u, logratio=-0.5 if logratio_same or role == "inc" else -0.6)
            if fail and fail == (sha, s):
                r = dict(status="failed")
            res[h.job_id(role, "c1", sha, s)] = r
    return res


def test_plan_covers_seven_cuts_per_program_and_dedups_the_incumbent():
    jobs = h.plan(HWM, CHAMP)
    assert len(jobs) == 3 * len(h.CUTS) and len({j["job_id"] for j in jobs}) == len(jobs)
    bad = [dict(HWM[0], dsl="(other)")]
    with pytest.raises(SystemExit):
        h.plan(bad, CHAMP)


def test_execution_errors_are_incomplete_never_a_loss():
    assert h.u_of(dict(status="failed")) is None and h.u_of(None) is None
    assert h.u_of(dict(status="completed", cv_gate_passed=False, fitness=0.3)) == -2.0
    res = fake_results({S(DSL_A): 0.01, S(DSL_B): 0.01}, fail=(S(DSL_B), 802))
    rows = {r["program_sha256"]: r for r in h.classify(HWM, CHAMP, res, {"c1": dict(best=1.0, champion=1.0)})}
    assert rows[S(DSL_B)]["status"] == "incomplete" and not rows[S(DSL_B)]["confirmed"]
    assert rows[S(DSL_A)]["status"] == "completed" and rows[S(DSL_A)]["material"]


def test_confirmation_material_and_complexity_flags():
    res = fake_results({S(DSL_A): 0.0005, S(DSL_B): -0.02}, logratio_same=True)
    rows = {r["program_sha256"]: r for r in h.classify(HWM, CHAMP, res, {"c1": dict(best=1.0, champion=1.0)})}
    a, b = rows[S(DSL_A)], rows[S(DSL_B)]
    assert a["confirmed"] and not a["material"] and a["complexity_only"] and a["identity_challenger"] and a["identity_incumbent"]
    assert not b["confirmed"] and not b["confirmed_701_703"]
    s = h.summarize(list(rows.values()))
    assert (s["confirmed"], s["material"], s["complexity_only"], s["confirmed_and_not_complexity_only"]) == (1, 0, 2, 0)
    assert s["verdict"].startswith("0/2 MATERIAL")


def test_loader_refuses_changed_bytes(tmp_path):
    f = tmp_path / "w8_control.py"; f.write_text("# not the merged helper\n")
    with pytest.raises(SystemExit):
        h.load(f, h.H72_SHA, "x")
