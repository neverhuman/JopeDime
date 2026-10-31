"""Synthetic contract controls, not scientific observations or data admission."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("h104_runner", ROOT / "runner.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def config():
    return json.loads((ROOT / "config.json").read_text())


def folds():
    rows = [
        {
            "context_sha256": r.sha(f"context{i}".encode()),
            "family_id": f"family{i % 205:03}",
            "physical_table_sha256": r.sha(f"table{i % 205}".encode()),
            "fold": (i % 205) % 5,
        }
        for i in range(1219)
    ]
    return {
        "schema": "jope.w6.family-folds/v1",
        "fold_seed": 4900,
        "fold_count": 5,
        "context_count": 1219,
        "family_count": 205,
        "families_by_fold": [
            [f"family{i:03}" for i in range(205) if i % 5 == k] for k in range(5)
        ],
        "assignments": rows,
    }


def offers():
    return [
        {
            "program_sha256": r.sha(f"program{i}".encode()),
            "prior_cost": float(i),
            "disagreement": float(i),
            "descriptor": "linear/1..8" if i < 3 else "tree/9..16",
        }
        for i in range(8)
    ]


def test_full_schedule_keeps_budget_and_fold_separation():
    m = folds()
    before = copy.deepcopy(m)
    p = r.build_plan(config(), m)
    assert m == before
    assert len(p["schedule"]) == 300
    assert p["planned_new_fits_in_shard"] == 3600
    assert p["planned_context_rounds_per_arm_seed_in_shard"] == 100
    lookup = {x["context_sha256"]: x for x in m["assignments"]}
    for job in p["schedule"]:
        assert lookup[job["context_sha256"]]["fold"] != job["held_fold"]
        assert set(job["arm_selection_seeds"]) == set(r.ARMS)
    for held in range(5):
        for seed in config()["seeds"]:
            selected = [
                x for x in p["schedule"] if x["held_fold"] == held and x["seed"] == seed
            ]
            assert len({x["family_id"] for x in selected}) == len(selected) == 20
    assert p["execution_admitted"] is p["advance_admitted"] is False
    assert p["actual_fits"] == p["actual_optimizer_updates"] == 0
    assert p["actual_campaign_consumption"] is p["remaining_authorized_budget"] is None


def test_fold_shards_recombine_exactly_without_budget_multiplication():
    c, m = config(), folds()
    full = r.build_plan(c, m)
    shards = [r.build_plan(c, m, f) for f in [[0], [1], [2], [3, 4]]]
    assert sum(x["planned_new_fits_in_shard"] for x in shards) == 3600
    assert sum((x["schedule"] for x in shards), []) == full["schedule"]
    assert (
        sum((x["common_learner_receipt_panel"] for x in shards), [])
        == full["common_learner_receipt_panel"]
    )


def test_pending_matching_receipts_never_become_measured_zero():
    p = r.build_plan(config(), folds())
    slots = p["common_learner_receipt_panel"]
    assert len(slots) == len({s["slot_id"] for s in slots}) == 45
    for s in slots:
        assert s["status"] == "pending"
        assert all(
            s[k] is None
            for k in [
                "receipt_sha256",
                "matching_receipt_sha256",
                "matched_rows",
                "discarded_rows",
                "excluded_rows",
                "common_support_rows",
            ]
        )
    assert "UNTESTED" in p["common_support_rule"]
    assert all(x is None for x in p["metrics"].values())


@pytest.mark.parametrize(
    "key,value",
    [
        ("dope_sha256", "0" * 64),
        ("acquisition_rounds_per_fold", 100),
        ("native_fit_ceiling", 9600),
        ("gpu_seconds_ceiling", 14400),
        ("v3_import_enabled", True),
        ("execution_admitted", True),
        ("seeds", [701, 701, 703]),
    ],
)
def test_contract_drift_refuses(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError):
        r.build_plan(c, folds())


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "family_leak",
        "table_leak",
        "outcome_field",
        "missing_row",
        "declared_count",
        "false_partition",
    ],
)
def test_fold_closure_refuses(change):
    m = folds()
    rows = m["assignments"]
    if change == "duplicate":
        rows[1] = rows[0].copy()
    elif change == "family_leak":
        rows[1]["family_id"] = rows[0]["family_id"]
    elif change == "table_leak":
        rows[1]["physical_table_sha256"] = rows[0]["physical_table_sha256"]
    elif change == "outcome_field":
        rows[0]["fitness"] = -1
    elif change == "missing_row":
        rows.pop()
    elif change == "declared_count":
        m["family_count"] = 204
    else:
        m["families_by_fold"][0].pop()
    with pytest.raises(ValueError):
        r.build_plan(config(), m)


@pytest.mark.parametrize("shard", [[], [0, 0], [5], [True]])
def test_invalid_shard_refuses(shard):
    with pytest.raises(ValueError):
        r.build_plan(config(), folds(), shard)


@pytest.mark.parametrize("arm", r.ARMS)
def test_acquisition_is_distinct_reproducible_and_has_uniform_support(arm):
    rows = offers()
    result = r.select_four(rows, arm, 701)
    assert result == r.select_four(list(reversed(rows)), arm, 701)
    assert len({x["program_sha256"] for x in result}) == 4
    remaining = {x["program_sha256"] for x in rows}
    for slot in result:
        assert set(slot["eligible_ids"]) == remaining
        if slot["rule"] == "uniform_remaining":
            assert slot["conditional_draw_probability"] == 1 / len(remaining)
        remaining.remove(slot["program_sha256"])
    assert result[-1]["rule"] == "uniform_remaining"


def test_qd_novel_descriptor_then_disagreement_and_unchanged_inputs():
    rows = offers()
    rows[2]["disagreement"] = 1000
    rows[4]["disagreement"] = 20
    before = copy.deepcopy(rows)
    result = r.select_four(rows, r.ARMS[2], 701)
    assert result[0]["program_sha256"] == rows[0]["program_sha256"]
    assert result[1]["program_sha256"] == rows[1]["program_sha256"]
    assert result[2]["program_sha256"] == rows[4]["program_sha256"]
    assert rows == before


@pytest.mark.parametrize(
    "change", ["outcome", "nan", "duplicate", "short", "negative_disagreement"]
)
def test_acquisition_rejects_labels_or_invalid_features(change):
    rows = offers()
    if change == "outcome":
        rows[0]["fitness"] = 0
    elif change == "nan":
        rows[0]["prior_cost"] = float("nan")
    elif change == "duplicate":
        rows[1] = rows[0].copy()
    elif change == "short":
        rows = rows[:3]
    else:
        rows[0]["disagreement"] = -1
    with pytest.raises(ValueError):
        r.select_four(rows, r.ARMS[2], 701)


def test_cli_requires_dry_run_pins_and_create_only_output(tmp_path):
    c, m = config(), folds()
    mp = tmp_path / "folds.json"
    mp.write_bytes(r.encoded(m))
    c["fold_manifest_sha256"] = r.sha(mp.read_bytes())
    cp = tmp_path / "config.json"
    cp.write_bytes(r.encoded(c))
    out = tmp_path / "plan.json"
    cmd = [
        sys.executable,
        str(ROOT / "runner.py"),
        "--config",
        str(cp),
        "--config-sha256",
        r.sha(cp.read_bytes()),
        "--fold-manifest",
        str(mp),
        "--output",
        str(out),
    ]
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert not out.exists()
    assert subprocess.run(cmd + ["--dry-run"], capture_output=True).returncode == 0
    before = out.read_bytes()
    assert subprocess.run(cmd + ["--dry-run"], capture_output=True).returncode != 0
    assert out.read_bytes() == before
    mp.write_bytes(mp.read_bytes() + b" ")
    cmd[-1] = str(tmp_path / "new.json")
    assert subprocess.run(cmd + ["--dry-run"], capture_output=True).returncode != 0
    assert not (tmp_path / "new.json").exists()


def test_duplicate_json_keys_refuse(tmp_path):
    p = tmp_path / "duplicate.json"
    p.write_bytes(b'{"fold":0,"fold":1}')
    with pytest.raises(ValueError, match="duplicate"):
        r.load_pinned(p, r.sha(p.read_bytes()))
