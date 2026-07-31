#!/usr/bin/env python3
"""Validate Advanced V3 graph/ledger V2 and its immutable V1 migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests" / "advanced-v3"
V1_GRAPH = MANIFESTS / "task-graph.v1.json"
V1_LEDGER = MANIFESTS / "work-ledger.v1.jsonl"
V2_GRAPH = MANIFESTS / "task-graph.v2.json"
V2_LEDGER = MANIFESTS / "work-ledger.v2.jsonl"
TASK_RECEIPTS = MANIFESTS / "task-receipts"
V2_GRAPH_SCHEMA = "jope-dime.advanced-v3-task-graph.v2"
V2_LEDGER_SCHEMA = "jope-dime.advanced-v3-work-ledger.v2"
TASK_RECEIPT_SCHEMA = "jope-dime.advanced-v3-task-receipt.v1"
ZERO_HASH = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
REQUIRED_TASK_FIELDS = {
    "id",
    "owner",
    "depends_on",
    "state",
    "title",
    "milestone",
    "exact_paths",
    "interfaces",
    "deliverables",
    "acceptance",
}
REQUIRED_PORTABLE_TASKS = {
    "P-00",
    "P-1A",
    "P-1B",
    "P-1C",
    "P-1D",
    "P-2",
    "P-3",
    "P-4",
    "P-5",
    "P-6",
}


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(entry: dict[str, object]) -> str:
    body = dict(entry)
    body.pop("entry_sha256", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nonempty_strings(value: object, field: str, *, allow_empty: bool = False) -> None:
    require(isinstance(value, list), f"{field} must be an array")
    if not allow_empty:
        require(bool(value), f"{field} must not be empty")
    require(
        all(isinstance(item, str) and item for item in value),
        f"{field} must contain only non-empty strings",
    )


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path.relative_to(ROOT)}: {exc}") from exc
    require(isinstance(value, dict), f"{path.relative_to(ROOT)} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read {path.relative_to(ROOT)}: {exc}") from exc
    require(bool(lines), f"{path.relative_to(ROOT)} must not be empty")
    entries: list[dict[str, object]] = []
    for number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"{path.relative_to(ROOT)}:{number} is blank")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{path.relative_to(ROOT)}:{number} is invalid JSON: {exc}"
            ) from exc
        require(isinstance(entry, dict), f"ledger line {number} must be an object")
        entries.append(entry)
    return entries


def validate_graph() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, str],
]:
    v1 = load_json(V1_GRAPH)
    v2 = load_json(V2_GRAPH)
    require(v2.get("schema") == V2_GRAPH_SCHEMA, "unexpected V2 graph schema")

    states = v2.get("states")
    nonempty_strings(states, "states")
    assert isinstance(states, list)
    state_set = set(states)
    require(len(state_set) == len(states), "graph states must be unique")

    tasks = v2.get("tasks")
    require(isinstance(tasks, list) and tasks, "V2 graph tasks must be a non-empty array")
    task_by_id: dict[str, dict[str, object]] = {}
    for index, raw_task in enumerate(tasks):
        require(isinstance(raw_task, dict), f"task {index} must be an object")
        task = raw_task
        missing = REQUIRED_TASK_FIELDS - task.keys()
        require(not missing, f"task {task.get('id', index)} missing {sorted(missing)}")
        task_id = task["id"]
        require(isinstance(task_id, str) and task_id, f"task {index} has invalid id")
        require(task_id not in task_by_id, f"duplicate task id {task_id}")
        require(isinstance(task["owner"], str) and task["owner"], f"{task_id}: invalid owner")
        require(task["state"] in state_set, f"{task_id}: unknown state {task['state']}")
        require(isinstance(task["title"], str) and task["title"], f"{task_id}: invalid title")
        require(
            isinstance(task["milestone"], str) and task["milestone"],
            f"{task_id}: invalid milestone",
        )
        nonempty_strings(task["depends_on"], f"{task_id}.depends_on", allow_empty=True)
        nonempty_strings(task["exact_paths"], f"{task_id}.exact_paths")
        require(
            all(not any(char in path for char in "*?[]") for path in task["exact_paths"]),
            f"{task_id}: exact_paths must not contain globs",
        )
        interfaces = task["interfaces"]
        require(isinstance(interfaces, dict), f"{task_id}.interfaces must be an object")
        require(set(interfaces) == {"consumes", "produces"}, f"{task_id}: invalid interfaces")
        nonempty_strings(interfaces["consumes"], f"{task_id}.interfaces.consumes", allow_empty=True)
        nonempty_strings(interfaces["produces"], f"{task_id}.interfaces.produces")
        nonempty_strings(task["deliverables"], f"{task_id}.deliverables")
        nonempty_strings(task["acceptance"], f"{task_id}.acceptance")
        task_by_id[task_id] = task

    current_states = v2.get("current_states")
    require(isinstance(current_states, dict), "current_states must be an object")
    effective_states: dict[str, str] = {}
    for task_id, task in task_by_id.items():
        base_state = task["state"]
        assert isinstance(base_state, str)
        effective_states[task_id] = base_state
    for task_id, state in current_states.items():
        require(isinstance(task_id, str) and task_id in task_by_id, f"unknown current-state task {task_id}")
        require(isinstance(state, str) and state in state_set, f"{task_id}: invalid current state {state}")
        effective_states[task_id] = state

    for task_id, task in task_by_id.items():
        dependencies = task["depends_on"]
        require(len(dependencies) == len(set(dependencies)), f"{task_id}: duplicate dependency")
        for dependency in dependencies:
            require(dependency in task_by_id, f"{task_id}: unknown dependency {dependency}")
            require(dependency != task_id, f"{task_id}: self dependency")
        if effective_states[task_id] in {"ready", "in_progress", "review", "verified"}:
            require(
                all(effective_states[dependency] == "verified" for dependency in dependencies),
                f"{task_id}: active/verified task has an unverified dependency",
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        require(task_id not in visiting, f"dependency cycle includes {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in task_by_id[task_id]["depends_on"]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_by_id:
        visit(task_id)

    missing_portable = REQUIRED_PORTABLE_TASKS - task_by_id.keys()
    require(not missing_portable, f"missing portable/search-v2 tasks {sorted(missing_portable)}")
    for task_id in sorted(REQUIRED_PORTABLE_TASKS):
        task = task_by_id[task_id]
        references = task.get("cross_references")
        nonempty_strings(references, f"{task_id}.cross_references")
        assert isinstance(references, list)
        require(
            all(reference in task_by_id for reference in references),
            f"{task_id}: unknown cross-reference",
        )
        require(
            isinstance(task.get("six_wave_reference"), str)
            and bool(task["six_wave_reference"]),
            f"{task_id}: missing six-wave reference",
        )
    require(task_by_id["P-6"].get("optional") is True, "P-6 must be explicitly optional")
    require(
        all(task_by_id[task_id].get("optional") is not True for task_id in REQUIRED_PORTABLE_TASKS - {"P-6"}),
        "only P-6 may be optional in the portable/search-v2 spine",
    )

    v1_tasks = v1.get("tasks")
    require(isinstance(v1_tasks, list), "V1 graph tasks must be an array")
    require(
        v2.get("migration", {}).get("imported_task_count") == len(v1_tasks),
        "migration imported_task_count does not match V1",
    )
    for raw_task in v1_tasks:
        require(isinstance(raw_task, dict), "V1 task must be an object")
        task_id = raw_task["id"]
        require(task_id in task_by_id, f"V1 task {task_id} missing from V2")
        for field in ("owner", "depends_on", "state"):
            require(
                task_by_id[task_id][field] == raw_task[field],
                f"V1 task {task_id} changed {field} in V2",
            )

    migration = v2.get("migration")
    require(isinstance(migration, dict), "V2 graph migration must be an object")
    require(migration.get("source_graph_sha256") == sha256(V1_GRAPH), "V1 graph digest drift")
    require(migration.get("source_ledger_sha256") == sha256(V1_LEDGER), "V1 ledger digest drift")
    v1_entries = load_jsonl(V1_LEDGER)
    require(
        migration.get("source_ledger_last_sequence") == v1_entries[-1].get("sequence") == 51,
        "migration must seal V1 ledger sequence 51",
    )
    return v1, v2, task_by_id, effective_states


def validate_receipt(entry: dict[str, object]) -> None:
    receipt_ref = entry.get("receipt")
    require(isinstance(receipt_ref, dict), "task transition must reference a receipt")
    relative_path = receipt_ref.get("path")
    require(isinstance(relative_path, str) and relative_path, "invalid receipt path")
    receipt_path = ROOT / relative_path
    require(receipt_path.is_file(), f"missing receipt {relative_path}")
    require(receipt_ref.get("sha256") == sha256(receipt_path), f"receipt digest drift: {relative_path}")
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == TASK_RECEIPT_SCHEMA, f"unexpected receipt schema: {relative_path}")
    require(receipt.get("task_id") == entry.get("task_id"), f"receipt task mismatch: {relative_path}")
    require(receipt.get("ledger_sequence") == entry.get("sequence"), f"receipt ledger sequence mismatch: {relative_path}")
    require(receipt.get("from") == entry.get("from"), f"receipt from-state mismatch: {relative_path}")
    require(receipt.get("to") == entry.get("to"), f"receipt to-state mismatch: {relative_path}")
    require(receipt.get("actor") == entry.get("actor"), f"receipt actor mismatch: {relative_path}")
    receipt_sequence = receipt.get("receipt_sequence")
    require(
        isinstance(receipt_sequence, int) and receipt_sequence > 0,
        f"invalid receipt sequence: {relative_path}",
    )
    expected_path = f"manifests/advanced-v3/task-receipts/{entry['task_id']}/{receipt_sequence:04d}.v1.json"
    require(relative_path == expected_path, f"receipt path/sequence mismatch: {relative_path}")
    nonempty_strings(receipt.get("changed_paths"), f"{relative_path}.changed_paths", allow_empty=True)
    nonempty_strings(receipt.get("checks"), f"{relative_path}.checks")
    nonempty_strings(receipt.get("evidence"), f"{relative_path}.evidence")


def validate_ledger(
    v2_graph: dict[str, object],
    task_by_id: dict[str, dict[str, object]],
    effective_states: dict[str, str],
    imported_task_ids: set[str],
) -> None:
    entries = load_jsonl(V2_LEDGER)
    previous = ZERO_HASH
    for expected_sequence, entry in enumerate(entries, 1):
        require(entry.get("schema") == V2_LEDGER_SCHEMA, f"ledger {expected_sequence}: schema")
        require(entry.get("sequence") == expected_sequence, f"ledger {expected_sequence}: sequence")
        timestamp = entry.get("timestamp")
        require(isinstance(timestamp, str) and UTC_RE.fullmatch(timestamp), f"ledger {expected_sequence}: timestamp")
        require(entry.get("prev_entry_sha256") == previous, f"ledger {expected_sequence}: broken previous hash")
        recorded = entry.get("entry_sha256")
        require(isinstance(recorded, str) and HASH_RE.fullmatch(recorded), f"ledger {expected_sequence}: entry hash")
        require(recorded == canonical_hash(entry), f"ledger {expected_sequence}: content hash mismatch")
        previous = recorded

    genesis = entries[0]
    require(genesis.get("event") == "migration_genesis", "first V2 ledger event must be migration_genesis")
    source = genesis.get("source")
    require(isinstance(source, dict), "genesis source must be an object")
    require(source.get("last_sequence") == 51, "genesis must reference V1 sequence 51")
    require(source.get("graph_sha256") == sha256(V1_GRAPH), "genesis V1 graph digest drift")
    require(source.get("ledger_sha256") == sha256(V1_LEDGER), "genesis V1 ledger digest drift")
    graph_ref = genesis.get("v2_graph")
    require(isinstance(graph_ref, dict), "genesis v2_graph must be an object")
    initial_graph_sha = graph_ref.get("sha256")
    require(
        isinstance(initial_graph_sha, str) and HASH_RE.fullmatch(initial_graph_sha),
        "genesis V2 graph digest is invalid",
    )
    latest_graph_sha = initial_graph_sha
    projected_states = {
        task_id: task["state"] for task_id, task in task_by_id.items()
    }
    allowed_transitions = {
        "queued": {"ready", "blocked", "rejected"},
        "ready": {"in_progress", "blocked", "rejected"},
        "in_progress": {"review", "blocked", "rejected"},
        "review": {"verified", "blocked", "rejected"},
        "verified": set(),
        "blocked": set(),
        "rejected": set(),
    }
    transitioned_tasks: set[str] = set()
    for entry in entries[1:]:
        event = entry.get("event")
        if event == "graph_revision":
            require(
                entry.get("previous_graph_sha256") == latest_graph_sha,
                f"ledger {entry['sequence']}: graph revision does not follow the prior graph",
            )
            revision = entry.get("v2_graph")
            require(isinstance(revision, dict), f"ledger {entry['sequence']}: missing graph revision")
            revision_sha = revision.get("sha256")
            require(
                isinstance(revision_sha, str) and HASH_RE.fullmatch(revision_sha),
                f"ledger {entry['sequence']}: invalid graph revision hash",
            )
            latest_graph_sha = revision_sha
        elif event == "task_transition":
            validate_receipt(entry)
            task_id = entry.get("task_id")
            require(isinstance(task_id, str) and task_id in task_by_id, f"ledger {entry['sequence']}: unknown task")
            if task_id not in transitioned_tasks and task_id not in imported_task_ids:
                initial = entry.get("from")
                require(isinstance(initial, str) and initial in allowed_transitions, f"ledger {entry['sequence']}: invalid additive-task initial state")
                projected_states[task_id] = initial
            current = projected_states[task_id]
            require(entry.get("from") == current, f"ledger {entry['sequence']}: transition from-state drift")
            target = entry.get("to")
            require(isinstance(target, str) and target in allowed_transitions[current], f"ledger {entry['sequence']}: invalid transition {current}->{target}")
            if target in {"ready", "in_progress", "review", "verified"}:
                require(
                    all(projected_states[dependency] == "verified" for dependency in task_by_id[task_id]["depends_on"]),
                    f"ledger {entry['sequence']}: transition has unverified dependency",
                )
            require(entry.get("graph_sha256") == latest_graph_sha, f"ledger {entry['sequence']}: transition graph drift")
            projected_states[task_id] = target
            transitioned_tasks.add(task_id)
        elif event == "task_correction":
            validate_receipt(entry)
            task_id = entry.get("task_id")
            require(isinstance(task_id, str) and task_id in task_by_id, f"ledger {entry['sequence']}: unknown corrected task")
            current = projected_states[task_id]
            require(entry.get("from") == entry.get("to") == current, f"ledger {entry['sequence']}: correction changed task state")
            require(entry.get("graph_sha256") == latest_graph_sha, f"ledger {entry['sequence']}: correction graph drift")
        else:
            raise ValidationError(f"ledger {entry['sequence']}: unknown event {event}")
    require(latest_graph_sha == sha256(V2_GRAPH), "latest V2 graph digest drift")
    require(projected_states == effective_states, "current_states do not match ledger-derived task states")
    current_states = v2_graph.get("current_states")
    assert isinstance(current_states, dict)
    require(
        current_states == {task_id: projected_states[task_id] for task_id in transitioned_tasks},
        "current_states must project exactly the tasks transitioned in V2",
    )
    require(
        v2_graph.get("schema") == V2_GRAPH_SCHEMA,
        "validated graph schema changed during validation",
    )


def main() -> int:
    try:
        v1_graph, v2_graph, task_by_id, effective_states = validate_graph()
        v1_tasks = v1_graph.get("tasks")
        assert isinstance(v1_tasks, list)
        imported_task_ids = {task["id"] for task in v1_tasks if isinstance(task, dict)}
        validate_ledger(v2_graph, task_by_id, effective_states, imported_task_ids)
    except ValidationError as exc:
        print(f"advanced-v3 V2 validation failed: {exc}", file=sys.stderr)
        return 1
    print("advanced-v3 V2 validation passed: graph, DAG, V1 migration, and ledger chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
