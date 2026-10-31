"""Synthetic matching controls; no outcome payloads or model execution."""

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "h111_matching", Path(__file__).with_name("matching.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cell():
    return dict(
        held_fold=0,
        seed=701,
        schedule_sha256=sha("synthetic schedule"),
        families=[
            dict(
                family_id=f"f{i}",
                context_sha256=sha(f"c{i}"),
                physical_table_sha256=sha(f"t{i}"),
            )
            for i in range(20)
        ],
    )


def records():
    return [
        dict(
            held_fold=0,
            seed=701,
            arm=a,
            family_id=f"f{i}",
            context_sha256=sha(f"c{i}"),
            program_sha256=sha(f"{a}/{i}/{j}"),
            status="completed_pass" if j < 3 else "completed_gate_fail",
            completion_receipt_sha256=sha(f"event/{a}/{i}/{j}"),
        )
        for a in m.ARMS
        for i in range(20)
        for j in range(4)
    ]


def test_full_matching_preserves_status_and_different_programs():
    r = m.match_cell(cell(), records())
    assert r["status"] == "CANDIDATE_MATCH" and r["common_support_families"] == 20
    assert all(
        v["matched"] == 80 and v["excluded"] == v["discarded_completed"] == 0
        for v in r["arm_counts"].values()
    )
    assert {s["matched_per_arm"] for s in r["strata"]} == {1, 3}
    assert len({tuple(r["strata"][0]["selected"][a]) for a in m.ARMS}) == 3
    assert (
        not r["execution_admitted"]
        and not r["advance_admitted"]
        and not r["source_provenance_verified"]
    )
    assert all(x is None for x in r["metrics"].values())


def test_input_permutation_is_byte_identical():
    rs = records()
    assert m.encoded(m.match_cell(cell(), rs)) == m.encoded(
        m.match_cell(cell(), rs[::-1])
    )


def test_minimum_counts_and_discarded_census_are_per_family_and_status():
    rs = records()
    for r in rs:
        if r["arm"] == m.ARMS[1] and r["status"] == "completed_gate_fail":
            r.update(status="unknown", completion_receipt_sha256=None)
    out = m.match_cell(cell(), rs)
    assert all(a["matched"] == 60 for a in out["arm_counts"].values())
    assert out["arm_counts"][m.ARMS[0]]["discarded_completed"] == 20
    assert out["arm_counts"][m.ARMS[1]]["excluded_by_status"]["unknown"] == 20
    assert all(
        s["matched_per_arm"] == 0
        for s in out["strata"]
        if s["status"] == "completed_gate_fail"
    )


@pytest.mark.parametrize("state", ["unknown", "timeout", "error"])
def test_noncompleted_rows_never_supply_targets(state):
    rs = records()
    for r in rs:
        r.update(status=state, completion_receipt_sha256=None)
    out = m.match_cell(cell(), rs)
    assert out["status"] == "UNTESTED" and out["common_support_families"] == 0
    assert all(
        a["matched"] == 0 and a["excluded"] == 80 for a in out["arm_counts"].values()
    )


def test_seven_common_families_cannot_shrink_batch_contract():
    rs = records()
    for r in rs:
        if int(r["family_id"][1:]) >= 7:
            r.update(status="unknown", completion_receipt_sha256=None)
    out = m.match_cell(cell(), rs)
    assert out["common_support_families"] == 7 and out["status"] == "UNTESTED"


@pytest.mark.parametrize("n", [0, 1, 239])
def test_missing_attempts_remain_pending_even_with_some_common_support(n):
    out = m.match_cell(cell(), records()[:n])
    assert out["status"] == "PENDING" and not out["selection_final"]
    assert out["missing_attempt_records"] == 240 - n


@pytest.mark.parametrize(
    "field,value",
    [
        ("held_fold", 1),
        ("seed", 702),
        ("seed", True),
        ("arm", "extra"),
        ("family_id", "foreign"),
        ("context_sha256", "f" * 64),
        ("program_sha256", "bad"),
        ("status", "signature_pass"),
        ("completion_receipt_sha256", None),
        ("utility", 5.0),
        ("fitness", -1.0),
        ("status", None),
    ],
)
def test_malformed_foreign_or_outcome_fields_are_refused(field, value):
    rs = records()
    rs[0][field] = value
    with pytest.raises(ValueError):
        m.match_cell(cell(), rs)


def test_duplicate_attempt_refused_not_silently_deduplicated():
    rs = records()
    with pytest.raises(ValueError, match="duplicate"):
        m.match_cell(cell(), rs + [rs[0]])


def test_fifth_distinct_attempt_refused():
    rs = records()
    extra = dict(rs[0], program_sha256=sha("extra"))
    with pytest.raises(ValueError, match="more than four"):
        m.match_cell(cell(), rs + [extra])


@pytest.mark.parametrize(
    "mutation", ["family", "context", "table", "short", "seed", "bool_fold"]
)
def test_invalid_schedule_refused(mutation):
    c = cell()
    if mutation in ("family", "context", "table"):
        k = dict(
            family="family_id", context="context_sha256", table="physical_table_sha256"
        )[mutation]
        c["families"][1][k] = c["families"][0][k]
    elif mutation == "short":
        c["families"].pop()
    elif mutation == "seed":
        c["seed"] = 999
    else:
        c["held_fold"] = True
    with pytest.raises(ValueError):
        m.match_cell(c, records())


def plan():
    rows = []
    for f in range(5):
        for s in (701, 702, 703):
            rows.extend(
                dict(r, held_fold=f, seed=s, round=i)
                for i, r in enumerate(cell()["families"])
            )
    return dict(
        schema="jope.h104.w7-dry-run/v1", execution_admitted=False, schedule=rows
    )


def test_complete_plan_yields_exactly_45_pending_slots():
    out = m.prepare_cells(plan())
    assert len(out["cells"]) == 15 and len(out["common_learner_receipt_panel"]) == 45
    assert len({r["slot_id"] for r in out["common_learner_receipt_panel"]}) == 45
    assert all(
        r["status"] == "pending" and r["matching_receipt_sha256"] is None
        for r in out["common_learner_receipt_panel"]
    )
    assert (
        not out["execution_admitted"]
        and out["actual_fits"] == out["actual_optimizer_updates"] == 0
    )


def test_partial_or_duplicate_plan_refused():
    p = plan()
    with pytest.raises(ValueError):
        m.prepare_cells(dict(p, schedule=p["schedule"][:-1]))
    p["schedule"][1] = copy.deepcopy(p["schedule"][0])
    with pytest.raises(ValueError):
        m.prepare_cells(p)
