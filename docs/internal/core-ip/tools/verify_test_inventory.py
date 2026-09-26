#!/usr/bin/env python3
"""Check the maintained regression test inventory against the CI coverage map (W0).

Every test-bearing coverage item has exactly one role. Regression items name their
subsystems, every required regression subsystem has an executing regression item, and
every preserved baseline failure has one triage record. This is a static map: it
neither collects nor executes tests and does not upgrade any failure into a pass.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
INVENTORY = Path(__file__).resolve().parents[1] / "findings" / "test-inventory.json"
SCHEMA = "jope.regression-test-inventory/v1"
ROLES = {"regression", "regression-real-corpus", "deferred-classification",
         "historical", "research", "infrastructure"}
SUBSYSTEM_ROLES = {"regression", "regression-real-corpus"}
REQUIRED = {"preparation", "encoders", "policy", "gp", "fitting", "teaching",
            "training", "storage", "export"}
TRIAGE = {"fit-authority-fence", "product-defect", "stale-expectation",
          "missing-real-corpus-input", "deferred-classification", "lane-timeout",
          "untriaged"}
STATUS = {"open", "fixed-locally", "moved", "ci-verified"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def test_items(coverage):
    """Coverage items that carry collected tests: test-enabled targets and Python families."""
    keys = set()
    for key, item in coverage["items"].items():
        discovery = item["discovery"]
        if discovery["kind"] == "python-family" or (
                discovery["kind"] == "target" and discovery.get("test")):
            keys.add(key)
    return keys


def executing(coverage, key):
    lanes = coverage["lanes"]
    return any(lanes[lane]["stage"] == "execution" and lane != "rust-check"
               for lane in coverage["items"][key]["lanes"])


def verify(inventory_path=INVENTORY, root=ROOT):
    coverage = json.loads((root / "deploy/ci/coverage.json").read_text())
    baseline = json.loads((root / "docs/internal/core-ip/findings/baseline.json").read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    require(inventory.get("schema") == SCHEMA, "inventory schema")
    subsystems = inventory.get("subsystems", {})
    require(REQUIRED <= set(subsystems), "required subsystem definitions")
    packages = {f"W{i}" for i in range(13)}
    for name, value in subsystems.items():
        require(value.get("work_packages") and set(value["work_packages"]) <= packages
                and value.get("description"), f"subsystem {name} definition")
    items = inventory.get("items", {})
    expected = test_items(coverage)
    require(not expected - set(items), f"unclassified: {sorted(expected - set(items))[:5]}")
    require(not set(items) - expected, f"stale: {sorted(set(items) - expected)[:5]}")
    covered = Counter()
    for key, item in items.items():
        role = item.get("role")
        require(role in ROLES, f"{key}: role")
        if role in SUBSYSTEM_ROLES:
            names = item.get("subsystems")
            require(isinstance(names, list) and names and len(set(names)) == len(names)
                    and set(names) <= set(subsystems), f"{key}: subsystems")
            if role == "regression" and executing(coverage, key):
                covered.update(names)
        else:
            require(not item.get("subsystems") and item.get("reason"), f"{key}: reason")
    uncovered = REQUIRED - set(covered)
    require(not uncovered, f"no executing regression item for {sorted(uncovered)}")

    failures = {(family["lane"], failure["class"], failure["test"])
                for family in baseline["families"] for failure in family["failures"]}
    triage = inventory.get("baseline_triage", [])
    seen = set()
    for record in triage:
        identity = (record.get("lane"), record.get("class"), record.get("test"))
        require(identity in failures, f"triage for unknown baseline failure {identity}")
        require(identity not in seen, f"duplicate triage {identity}")
        seen.add(identity)
        require(record.get("item") in items, f"{identity}: inventory item")
        require(record.get("triage") in TRIAGE, f"{identity}: triage")
        require(record.get("status") in STATUS, f"{identity}: status")
        require(bool(record.get("note")), f"{identity}: note")
        if record["status"] != "open":
            require(record.get("evidence"), f"{identity}: {record['status']} without evidence")
        if record["status"] == "moved":
            require(record.get("moved_to"), f"{identity}: moved without destination")
        if record["triage"] in {"missing-real-corpus-input", "deferred-classification",
                                "fit-authority-fence", "untriaged"}:
            require(record["status"] in {"open", "moved"},
                    f"{identity}: {record['triage']} cannot be marked fixed")
    require(seen == failures, f"untriaged baseline failures: {sorted(failures - seen)[:3]}")
    return {
        "items": len(items),
        "roles": dict(sorted(Counter(i["role"] for i in items.values()).items())),
        "executing_regression_subsystems": dict(sorted(covered.items())),
        "baseline_failures": len(failures),
        "triage_status": dict(sorted(Counter(r["status"] for r in triage).items())),
        "scope": "static classification only; no collection, execution or qualification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    args = parser.parse_args()
    print(json.dumps(verify(args.inventory), sort_keys=True))
