"""Synthetic regression controls, never scientific or source-admission evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "h101_check", Path(__file__).with_name("check.py")
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
H = {
    k: check.digest(k.encode())
    for k in (
        "context",
        "child",
        "parent",
        "pool",
        "receipt",
        "alias",
        "ledger",
        "bytes",
    )
}


def ref():
    return {
        "s3_uri": "s3://synthetic-control/metadata.json",
        "version_id": "synthetic-version",
        "sha256": H["bytes"],
    }


def packet():
    return {
        "family_alias_registry_sha256": H["alias"],
        "sealed_family_manifest": {
            "scope": "global_sealed_meta_test",
            "complete": True,
            "source": ref(),
            "independent_authority_receipt": ref(),
            "family_ids": ["sealed-family"],
            "family_alias_registry_sha256": H["alias"],
        },
        "pool_membership": {
            H["context"]: {
                "pool_sha256": H["pool"],
                "receipt_sha256": H["receipt"],
                "program_sha256": [H["child"], H["parent"]],
            }
        },
        "children": [
            {
                "context_sha256": H["context"],
                "child_sha256": H["child"],
                "family_id": "fit-family",
                "parent_sha256": H["parent"],
                "operator": "mutation",
                "pool_sha256": H["pool"],
                "pool_membership_receipt_sha256": H["receipt"],
                "lineage_source": ref(),
                "original_row_source": dict(
                    ref(),
                    source_row_id="row-3",
                    context_sha256=H["context"],
                    child_sha256=H["child"],
                    original_training_row_ledger_sha256=H["ledger"],
                    verified_byte_receipt_sha256=H["receipt"],
                ),
            }
        ],
    }


def protocol():
    return {
        "folds": 5,
        "seeds": [701, 702, 703],
        "budgets": {
            "reservation": "existing_W9_envelope_not_additive",
            "deadline_utc": "2026-10-31T14:00:00Z",
            "native_calls_max": 9600,
        },
    }


def test_complete_relations_never_authorize_execution():
    result = check.check_w5(packet())
    assert result["verdict"] == "PASS_CHECKED_RELATIONS"
    assert result["execution_admitted"] is False


@pytest.mark.parametrize(
    "change", ["absent", "partial", "alias", "authority", "duplicate"]
)
def test_global_exclusion_requires_complete_bound_manifest(change):
    data = packet()
    sealed = data["sealed_family_manifest"]
    if change == "absent":
        data["sealed_family_manifest"] = None
    elif change == "partial":
        sealed["complete"] = False
    elif change == "alias":
        sealed["family_alias_registry_sha256"] = H["child"]
    elif change == "authority":
        sealed.pop("independent_authority_receipt")
    else:
        sealed["family_ids"] *= 2
    result = check.check_w5(data)
    assert result["verdict"] == "FAIL"
    assert result["checks"]["global_sealed_family_exclusion"]["overlap_count"] is None


def test_actual_sealed_overlap_fails():
    data = packet()
    data["children"][0]["family_id"] = "sealed-family"
    assert (
        check.check_w5(data)["checks"]["global_sealed_family_exclusion"][
            "overlap_count"
        ]
        == 1
    )
    assert check.check_w5(data)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "change",
    [
        "absent_parent",
        "foreign_parent",
        "self_parent",
        "wrong_pool",
        "wrong_receipt",
        "no_operator",
        "child_absent",
        "bad_membership",
    ],
)
def test_lineage_requires_same_pool_membership(change):
    data = packet()
    row = data["children"][0]
    pool = data["pool_membership"][H["context"]]
    if change == "absent_parent":
        row.pop("parent_sha256")
    elif change == "foreign_parent":
        row["parent_sha256"] = H["alias"]
    elif change == "self_parent":
        row["parent_sha256"] = row["child_sha256"]
    elif change == "wrong_pool":
        row["pool_sha256"] = H["alias"]
    elif change == "wrong_receipt":
        row["pool_membership_receipt_sha256"] = H["alias"]
    elif change == "no_operator":
        row.pop("operator")
    elif change == "child_absent":
        pool["program_sha256"] = [H["parent"]]
    else:
        pool["program_sha256"] = None
    assert check.check_w5(data)["checks"]["parent_child_lineage"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "wrong_context",
        "wrong_child",
        "etag",
        "no_version",
        "no_ledger",
        "no_row",
    ],
)
def test_original_row_requires_immutable_bound_source(change):
    data = packet()
    origin = data["children"][0]["original_row_source"]
    if change == "missing":
        data["children"][0]["original_row_source"] = None
    elif change == "wrong_context":
        origin["context_sha256"] = H["alias"]
    elif change == "wrong_child":
        origin["child_sha256"] = H["alias"]
    elif change == "etag":
        origin["sha256"] = "etag-is-not-a-sha"
    elif change == "no_version":
        origin["version_id"] = "null"
    elif change == "no_ledger":
        origin.pop("original_training_row_ledger_sha256")
    else:
        origin["source_row_id"] = ""
    assert check.check_w5(data)["checks"]["original_row_source"]["status"] == "FAIL"


@pytest.mark.parametrize("rows", [[], [None], [{}], [{"context_sha256": []}]])
def test_malformed_census_refuses(rows):
    data = packet()
    data["children"] = rows
    with pytest.raises(ValueError):
        check.check_w5(data)


def test_duplicate_children_refuse():
    data = packet()
    data["children"] *= 2
    with pytest.raises(ValueError):
        check.check_w5(data)


def test_byte_pin_and_duplicate_json_keys_refuse(tmp_path):
    p = tmp_path / "input.json"
    p.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="SHA-256"):
        check.load_pinned(p, H["bytes"])
    with pytest.raises(ValueError, match="duplicate"):
        check.load_pinned(p, check.digest(p.read_bytes()))


def test_w9_slots_exclude_held_fold_preserve_budget_and_unknowns():
    original = protocol()
    snapshot = copy.deepcopy(original)
    plan = check.prepare_w9(original, H["bytes"])
    assert original == snapshot
    assert plan["original_budget"] == original["budgets"]
    assert len(plan["slots"]) == len({s["slot_id"] for s in plan["slots"]}) == 150
    for slot in plan["slots"]:
        assert set(slot["fit_folds"]) == set(range(5)) - {slot["held_fold"]}
        assert slot["receipt_sha256"] is None
    assert plan["actual_consumption"] is plan["remaining_authorized_budget"] is None
    assert plan["execution_admitted"] is plan["protocol_amended"] is False
    assert plan["completed_replay_receipts"] == 0
    assert "never terminal R" in plan["uniform_memory_source"]


@pytest.mark.parametrize("change", ["folds", "seeds", "reservation"])
def test_w9_contract_change_refuses(change):
    p = protocol()
    if change == "reservation":
        p["budgets"]["reservation"] = "new-budget"
    else:
        p[change] = 1
    with pytest.raises(ValueError):
        check.prepare_w9(p, H["bytes"])


def test_cli_refusal_produces_plan_and_never_overwrites(tmp_path):
    data = packet()
    data["sealed_family_manifest"] = None
    paths = [tmp_path / name for name in ("packet.json", "protocol.json")]
    for p, value in zip(paths, [data, protocol()]):
        p.write_text(json.dumps(value))
    output = tmp_path / "out"
    args = [
        sys.executable,
        str(Path(check.__file__)),
        "--input",
        str(paths[0]),
        "--sha256",
        check.digest(paths[0].read_bytes()),
        "--w9-protocol",
        str(paths[1]),
        "--w9-protocol-sha256",
        check.digest(paths[1].read_bytes()),
        "--output",
        str(output),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 1
    assert (
        json.loads((output / "w9_cpu_prep.json").read_text())["pending_replay_receipts"]
        == 150
    )
    before = (output / "results.json").read_bytes()
    again = subprocess.run(args, capture_output=True, text=True)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
    assert (output / "results.json").read_bytes() == before
