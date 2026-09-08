#!/usr/bin/env python3
"""Validate the append-only Unified Program V3 graph and ledger genesis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests" / "advanced-v3"
V2_GRAPH = MANIFESTS / "task-graph.v2.json"
V2_LEDGER = MANIFESTS / "work-ledger.v2.jsonl"
V3_GRAPH = MANIFESTS / "task-graph.v3.json"
V3_LEDGER = MANIFESTS / "work-ledger.v3.jsonl"
ROADMAP = ROOT / "ROADMAP.md"

GRAPH_SCHEMA = "jope-dime.unified-program-task-graph.v3"
LEDGER_SCHEMA = "jope-dime.unified-program-work-ledger.v3"
ZERO_HASH = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
CANONICAL_MILESTONES = [
    "U0-AUTHORITY",
    "U1-FOUNDATION",
    "U2-RECOVERY",
    "U3-CUTOVER",
    "U4-SCIENCE",
    "U5-COMPILE",
    "U6-CPU",
    "U7-RELEASE-A",
    "U8-CANARY",
    "U9A-GPU",
    "U9B-HYPERION",
    "U10-OPTIONAL-TASKS",
]


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path.relative_to(ROOT)}: {exc}") from exc
    require(isinstance(value, dict), f"{path.relative_to(ROOT)} must be an object")
    return value


def load_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read {path.relative_to(ROOT)}: {exc}") from exc
    require(bool(lines), f"{path.relative_to(ROOT)} is empty")
    entries = []
    for number, line in enumerate(lines, 1):
        require(bool(line), f"{path.relative_to(ROOT)}:{number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid ledger JSON at line {number}: {exc}"
            ) from exc
        require(isinstance(value, dict), f"ledger line {number} is not an object")
        entries.append(value)
    return entries


def canonical_hash(entry: dict) -> str:
    body = dict(entry)
    body.pop("entry_sha256", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_graph(graph: dict, v2: dict) -> None:
    require(graph.get("schema") == GRAPH_SCHEMA, "unexpected V3 graph schema")
    require(
        graph.get("authority") == "successor", "V3 is not marked successor authority"
    )
    states = graph.get("states")
    require(
        isinstance(states, list)
        and len(states) == len(set(states))
        and all(isinstance(state, str) and state for state in states),
        "V3 states are invalid",
    )

    milestones = graph.get("milestones")
    require(isinstance(milestones, list), "V3 milestones are absent")
    require(
        [row.get("id") for row in milestones if isinstance(row, dict)]
        == CANONICAL_MILESTONES,
        "V3 milestones are absent, duplicated, or noncanonical",
    )
    milestone_by_id = {row["id"]: row for row in milestones}
    for row in milestones:
        require(
            set(row)
            == {"id", "phase", "state", "optional", "depends_on", "title", "exit_gate"},
            f"{row.get('id')} fields drift",
        )
        require(row["state"] in states, f"{row['id']} has an invalid state")
        require(
            isinstance(row["optional"], bool), f"{row['id']} optional flag is invalid"
        )
        require(
            isinstance(row["depends_on"], list), f"{row['id']} dependencies are invalid"
        )
        require(
            all(dep in milestone_by_id for dep in row["depends_on"]),
            f"{row['id']} has an unknown dependency",
        )
        require(
            all(
                isinstance(row[field], str) and row[field]
                for field in ("phase", "title", "exit_gate")
            ),
            f"{row['id']} descriptive fields are invalid",
        )
    require(
        {row["id"] for row in milestones if row["optional"]}
        == {"U9A-GPU", "U9B-HYPERION", "U10-OPTIONAL-TASKS"},
        "optional release lanes drift",
    )

    migration = graph.get("migration")
    require(isinstance(migration, dict), "V3 migration is absent")
    frozen = [
        (migration.get("frozen_v2_graph"), V2_GRAPH),
        (migration.get("frozen_v2_ledger"), V2_LEDGER),
        (migration.get("frozen_roadmap"), ROADMAP),
    ]
    for reference, path in frozen:
        require(
            isinstance(reference, dict), f"frozen reference for {path.name} is absent"
        )
        require(
            reference.get("path") == str(path.relative_to(ROOT)),
            f"frozen path drift for {path.name}",
        )
        require(
            reference.get("sha256") == sha256(path),
            f"frozen digest drift for {path.name}",
        )

    v2_entries = load_jsonl(V2_LEDGER)
    v2_ledger = migration["frozen_v2_ledger"]
    require(
        v2_ledger.get("last_sequence") == len(v2_entries) == 54,
        "V2 terminal sequence drift",
    )
    require(
        v2_ledger.get("last_entry_sha256") == v2_entries[-1].get("entry_sha256"),
        "V2 terminal entry drift",
    )
    require(
        COMMIT_RE.fullmatch(str(migration.get("source_commit"))) is not None,
        "source commit is invalid",
    )
    require(
        COMMIT_RE.fullmatch(str(migration.get("verification_handoff_commit")))
        is not None,
        "verification handoff commit is invalid",
    )

    mapping = migration.get("old_to_canonical_milestone")
    require(isinstance(mapping, dict), "old-task mapping is absent")
    old_ids = {row.get("id") for row in v2.get("tasks", []) if isinstance(row, dict)}
    require(
        len(old_ids) == migration.get("old_task_count") == 89,
        "V2 task-count binding drift",
    )
    require(set(mapping) == old_ids, "old-task mapping is not complete and exact")
    require(
        set(mapping.values()) <= set(CANONICAL_MILESTONES),
        "old-task mapping names an unknown milestone",
    )
    for old in ("M1-EVIDENCE", "P-2"):
        require(
            mapping.get(old) == "U4-SCIENCE",
            f"{old} was not collapsed into scientific evidence",
        )
    for old in ("M2-TYPED", "P-4"):
        require(
            mapping.get(old) == "U5-COMPILE",
            f"{old} was not collapsed into typed compilation",
        )
    for old in ("M3-EXEC", "P-3"):
        require(
            mapping.get(old) == "U6-CPU", f"{old} was not collapsed into CPU execution"
        )
    require(
        mapping.get("M6-HYPERION") == mapping.get("P-6") == "U9B-HYPERION",
        "Hyperion lanes were not collapsed",
    )
    require(
        mapping.get("M7-ACCEL")
        == mapping.get("P-5")
        == mapping.get("A-15")
        == "U9A-GPU",
        "GPU lanes were not collapsed",
    )

    handoff = migration.get("workspace_handoff")
    require(isinstance(handoff, list), "workspace handoff is absent")
    handoff_paths = {row.get("path") for row in handoff if isinstance(row, dict)}
    require(
        handoff_paths
        == {
            "scripts/__pycache__/test_exact_packed_corpus.cpython-312.pyc",
            "sotarena",
            ".claude/",
            "src/bin/jope_selfplay_wave.rs",
        },
        "workspace handoff does not exactly bind the pre-migration dirty/live paths",
    )
    wave = next(
        row for row in handoff if row["path"] == "src/bin/jope_selfplay_wave.rs"
    )
    require(
        wave.get("sha256") == sha256(ROOT / wave["path"]),
        "optional-floor wave handoff drift",
    )

    policy = graph.get("frozen_capability_policy")
    require(isinstance(policy, dict), "frozen capability policy is absent")
    scq80 = policy.get("scq80")
    require(
        isinstance(scq80, dict)
        and scq80.get("classification") == "test_adaptive"
        and scq80.get("benchmark_valid") is False
        and scq80.get("release_evidence") is False
        and scq80.get("terminal_action") == "archive_after_integrity_audit",
        "SCQ80 classification is not permanently fail-closed",
    )
    for capability in ("cpp17", "es2020", "gpu", "hyperion"):
        require(
            policy.get(capability, {}).get("first_release_required") is False,
            f"{capability} became a Release A requirement",
        )


def validate_ledger(graph: dict) -> None:
    entries = load_jsonl(V3_LEDGER)
    previous = ZERO_HASH
    for sequence, entry in enumerate(entries, 1):
        require(
            entry.get("schema") == LEDGER_SCHEMA, f"ledger {sequence}: schema drift"
        )
        require(entry.get("sequence") == sequence, f"ledger {sequence}: sequence drift")
        require(
            entry.get("prev_entry_sha256") == previous,
            f"ledger {sequence}: broken chain",
        )
        require(
            UTC_RE.fullmatch(str(entry.get("timestamp"))) is not None,
            f"ledger {sequence}: invalid timestamp",
        )
        recorded = entry.get("entry_sha256")
        require(
            HASH_RE.fullmatch(str(recorded)) is not None,
            f"ledger {sequence}: invalid entry digest",
        )
        require(
            recorded == canonical_hash(entry),
            f"ledger {sequence}: content digest drift",
        )
        previous = recorded

    genesis = entries[0]
    require(
        genesis.get("event") == "migration_genesis",
        "V3 ledger does not start with migration genesis",
    )
    source = genesis.get("source")
    migration = graph["migration"]
    require(isinstance(source, dict), "V3 genesis source is absent")
    require(
        source.get("v2_graph_sha256") == migration["frozen_v2_graph"]["sha256"],
        "genesis V2 graph drift",
    )
    require(
        source.get("v2_ledger_sha256") == migration["frozen_v2_ledger"]["sha256"],
        "genesis V2 ledger drift",
    )
    require(
        source.get("v2_last_sequence")
        == migration["frozen_v2_ledger"]["last_sequence"],
        "genesis V2 sequence drift",
    )
    require(
        source.get("v2_last_entry_sha256")
        == migration["frozen_v2_ledger"]["last_entry_sha256"],
        "genesis V2 entry drift",
    )
    require(
        source.get("roadmap_sha256") == migration["frozen_roadmap"]["sha256"],
        "genesis roadmap drift",
    )
    require(
        source.get("source_commit") == migration["source_commit"],
        "genesis source commit drift",
    )
    require(
        source.get("verification_handoff_commit")
        == migration["verification_handoff_commit"],
        "genesis handoff commit drift",
    )
    graph_ref = genesis.get("v3_graph")
    require(isinstance(graph_ref, dict), "genesis V3 graph reference is absent")
    require(
        graph_ref.get("path") == str(V3_GRAPH.relative_to(ROOT)),
        "genesis V3 graph path drift",
    )
    require(
        graph_ref.get("sha256") == sha256(V3_GRAPH), "genesis V3 graph digest drift"
    )


def main() -> int:
    try:
        v2 = load_json(V2_GRAPH)
        graph = load_json(V3_GRAPH)
        validate_graph(graph, v2)
        validate_ledger(graph)
    except ValidationError as exc:
        print(f"unified V3 validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        "unified V3 validation passed: frozen V2, exact migration, graph, and ledger chain"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
