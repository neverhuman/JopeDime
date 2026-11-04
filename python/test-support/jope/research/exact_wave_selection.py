from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from test_exact_campaign import HEAD_PREFIXES, dataset_order, top_level_stages
from test_exact_common import (
    atomic_json,
    compose_seed,
    evaluation_protocol_sha256,
    prime_source_root,
    selfplay_authority,
    sha256_file,
    sha256_text,
    validate_catalog_manifest,
)
from test_exact_pair_registry import PairRegistry
from test_exact_freshness import epoch_for_checkpoint

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
CATALOG_POINTER = ROOT / "catalog" / "current.json"
# The Rust wave builder deserializes the pending cohort into `CohortPlan`
# (jope_selfplay_wave.rs:230), which requires `sealed_cohort_sha256` -- the
# digest of the frozen sealed cohort the wave is bound to. Omitting it makes
# `build` fail with "missing field `sealed_cohort_sha256`" on every poll, which
# starves the queue: no wave is ever built, so no jobs are ever submitted.
SEALED_COHORT = ROOT / "catalog" / "private" / "sealed-current-v1.json"
# The Rust wave builder is the authority for this lane (jope_selfplay_supply
# invokes only jope_selfplay_wave) and its COHORT_SCHEMA is "JopeMicroCohort.v2"
# (jope_selfplay_wave.rs:31). This writer emitted "...v1" into the SAME
# waves/pending/ directory, so any cohort it wrote was rejected with
# "unsupported cohort schema" and starved the queue until someone deleted the
# file by hand. The payloads are otherwise byte-compatible -- identical top-level
# keys, identical {task,dataset_hash,micro_lane} entries, the documented 16/32/16
# lanes -- so the version string was the only divergence. Keep this in lockstep
# with COHORT_SCHEMA; they are one contract.
MICRO_COHORT_SCHEMA = "JopeMicroCohort.v2"
CORPUS = ROOT / "corpus" / "strict-test-exact.sqlite"
BACKLOG = ROOT / "ops" / "backlog.sqlite"
PAIR_LEDGER = ROOT / "ops" / "strict-wave-pairs.sqlite"
CHAMPIONS = ROOT / "indexes" / "champions.json"
CURRENT_HOF = ROOT / "indexes" / "hall-of-fame.json"
ACCEPTED = ROOT / "training" / "accepted.json"
PROPOSALS = ROOT / "proposals" / "dl-proposals.jsonl"
BASE_CHECKPOINT = Path(
    "/home/ubuntu/jope-flywheel/training/round1/checkpoints/"
    "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt"
)
SHARD_SCHEMA = "JopeFlywheelCorpusEvalShard.v1"
WAVE = re.compile(r"^strict-wave-(\d{6})$")

TRANSFORMS = (
    "(standardize)",
    "(robust_scale)",
    "(rank append=true)",
    "(rank append=false)",
    "(sigmoid append=true)",
    "(sigmoid append=false)",
    "(row_stats)",
    "(blom append=true)",
    "(cheb degree=2)",
    "(cheb degree=3)",
    "(cheb degree=4)",
    "(cheb degree=5)",
    "(cheb append=true degree=2)",
    "(cheb append=true degree=3)",
    "(pair_prod append=true topk=2)",
    "(pair_prod append=true topk=4)",
    "(pair_prod append=true topk=8)",
    "(pair_prod append=true topk=12)",
    "(quantile_bins bins=4)",
    "(quantile_bins bins=8)",
    "(quantile_bins bins=16)",
    "(winsor_clip q=0.01)",
    "(winsor_clip q=0.02)",
    "(winsor_clip q=0.05)",
    "(winsor_clip q=0.1)",
    "(hinge knots=4)",
    "(hinge knots=8)",
)


def checkpoint() -> tuple[Path, str]:
    if ACCEPTED.is_file():
        value = json.loads(ACCEPTED.read_text())
        path = Path(value["checkpoint"])
        digest = value["checkpoint_sha256"]
    else:
        path = BASE_CHECKPOINT
        digest = sha256_file(path)
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("accepted checkpoint content verification failed")
    return path, digest


def next_wave_id() -> str:
    indexes = []
    for parent in (ROOT / "waves", ROOT / "manifests"):
        if not parent.is_dir():
            continue
        for path in parent.iterdir():
            match = WAVE.fullmatch(path.name)
            if match:
                indexes.append(int(match.group(1)))
    return f"strict-wave-{max(indexes, default=0) + 1:06d}"


def task_valid(genome: str, task: str) -> bool:
    stages = top_level_stages(genome)
    if not stages:
        return False
    head = stages[-1][1:].split(None, 1)[0].rstrip(")")
    if head not in HEAD_PREFIXES:
        return False
    return head == "logit_ridge" if task == "binary" else head != "logit_ridge"


def normalized_body(stages: list[str], task: str) -> list[str]:
    body = stages[:-1]
    if task == "binary":
        return [
            stage
            for stage in body
            if stage[1:].split(None, 1)[0].rstrip(")") != "target"
        ]
    targets = [
        stage
        for stage in body
        if stage[1:].split(None, 1)[0].rstrip(")") == "target"
    ]
    rest = [stage for stage in body if stage not in targets]
    return [*targets[:1], *rest]


def mutate(
    parents: list[dict],
    *,
    task: str,
    dataset_hash: str,
    wave_id: str,
    attempt: int,
) -> dict | None:
    selector = int(
        sha256_text(f"{wave_id}:{task}:{dataset_hash}:{attempt}")[:16], 16
    )
    parent = parents[selector % len(parents)]
    other = parents[(selector // 101 + attempt) % len(parents)]
    stages = top_level_stages(parent["canonical_genome"])
    other_stages = top_level_stages(other["canonical_genome"])
    if not stages or not other_stages:
        return None
    head = stages[-1]
    body = normalized_body(stages, task)
    other_body = normalized_body(other_stages, task)
    first = TRANSFORMS[(selector // 7) % len(TRANSFORMS)]
    second = TRANSFORMS[(selector // 1009) % len(TRANSFORMS)]
    minimum = 1 if body and body[0].startswith("(target ") else 0
    position = minimum + (selector % (len(body) - minimum + 1))
    mode = (selector // 31) % 5
    if mode == 0:
        child_body = [*body[:position], first, *body[position:]]
    elif mode == 1:
        child_body = [*body[:position], first, second, *body[position:]]
    elif mode == 2 and len(body) > minimum:
        remove = minimum + (selector % (len(body) - minimum))
        kept = [stage for index, stage in enumerate(body) if index != remove]
        position = min(position, len(kept))
        child_body = [*kept[:position], first, *kept[position:]]
    elif mode == 3 and other_body:
        left = body[: max(minimum, len(body) // 2)]
        right_start = 1 if other_body[0].startswith("(target ") else 0
        child_body = [*left, *other_body[right_start : right_start + 4], first]
    else:
        child_body = [*body, first]
    genome = "(pipe X " + " ".join([*child_body, head]) + ")"
    return {
        "canonical_genome": genome,
        "semantic_sha256": sha256_text(genome),
        "source": "strict_wave_policy_mutation_crossover",
        "parentage": [
            parent["semantic_sha256"],
            other["semantic_sha256"],
        ],
        "parent_score_receipt": parent.get("score_receipt"),
        "mutation_attempt": attempt,
        "strict_wave": wave_id,
    }


def load_known_pairs() -> dict[tuple[str, str], set[str]]:
    known: dict[tuple[str, str], set[str]] = defaultdict(set)
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    for task, dataset, semantic in con.execute(
        "SELECT task,dataset_hash,semantic_sha256 FROM observations"
    ):
        known[(task, dataset)].add(semantic)
    for task, dataset, semantic in con.execute(
        """
        SELECT task,dataset_hash,semantic_sha256 FROM failures
        WHERE failure IN ('parse','fit','nonfinite','task-invalid','shape')
        """
    ):
        known[(task, dataset)].add(semantic)
    con.close()
    manifest_paths: set[Path] = set()
    if BACKLOG.is_file():
        con = sqlite3.connect(f"file:{BACKLOG}?mode=ro", uri=True)
        manifest_paths.update(
            Path(row[0]) for row in con.execute("SELECT manifest FROM claims")
        )
        con.close()
    for directory in (ROOT / "manifests").glob("strict-wave-*"):
        if directory.is_dir():
            manifest_paths.update(directory.glob("*.json"))
    ledger = sqlite3.connect(PAIR_LEDGER)
    ledger.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS manifests(
          path TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS pairs(
          task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
          semantic_sha256 TEXT NOT NULL,
          PRIMARY KEY(task,dataset_hash,semantic_sha256)
        ) WITHOUT ROWID;
        """
    )
    indexed = {row[0] for row in ledger.execute("SELECT path FROM manifests")}
    pending = sorted(path for path in manifest_paths if str(path) not in indexed)
    for index, path in enumerate(pending, 1):
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        ledger.executemany(
            "INSERT OR IGNORE INTO pairs VALUES(?,?,?)",
            [
                (
                    value["task"],
                    value["dataset_hash"],
                    candidate["semantic_sha256"],
                )
                for candidate in value["candidates"]
            ],
        )
        ledger.execute("INSERT OR IGNORE INTO manifests VALUES(?)", (str(path),))
        if index % 500 == 0:
            ledger.commit()
    ledger.commit()
    for task, dataset, semantic in ledger.execute(
        "SELECT task,dataset_hash,semantic_sha256 FROM pairs"
    ):
        known[(task, dataset)].add(semantic)
    ledger.close()
    return known


def load_parents() -> tuple[dict[tuple[str, str], list[dict]], dict[str, list[dict]]]:
    local: dict[tuple[str, str], list[dict]] = defaultdict(list)
    global_by_semantic: dict[str, dict[str, dict]] = {
        "regression": {},
        "binary": {},
    }
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT task,dataset_hash,semantic_sha256,canonical_genome,
               candidate_source,score,receipt_sha256
        FROM observations
        WHERE candidate_source='accepted_dl_checkpoint'
        ORDER BY task,dataset_hash,score DESC,semantic_sha256
        """
    )
    for row in rows:
        value = {
            "canonical_genome": row["canonical_genome"],
            "semantic_sha256": row["semantic_sha256"],
            "source": row["candidate_source"],
            "score_receipt": {
                "score": float(row["score"]),
                "receipt_sha256": row["receipt_sha256"],
            },
        }
        key = (row["task"], row["dataset_hash"])
        if len(local[key]) < 64:
            local[key].append(value)
        current = global_by_semantic[row["task"]].get(row["semantic_sha256"])
        if current is None or value["score_receipt"]["score"] > current["score_receipt"]["score"]:
            global_by_semantic[row["task"]][row["semantic_sha256"]] = value
    con.close()
    champions = json.loads(CHAMPIONS.read_text())
    for row in champions:
        best = row["best_new_strict"]
        local[(row["task"], row["dataset_hash"])].insert(
            0,
            {
                "canonical_genome": best["canonical_genome"],
                "semantic_sha256": best["semantic_sha256"],
                "source": "current_strict_champion",
                "score_receipt": {
                    "score": float(best["score"]),
                    "receipt_sha256": best["receipt_sha256"],
                },
            },
        )
    global_parents = {
        task: sorted(
            values.values(),
            key=lambda row: (
                -row["score_receipt"]["score"],
                row["semantic_sha256"],
            ),
        )[:512]
        for task, values in global_by_semantic.items()
    }
    return local, global_parents


def load_checkpoint_proposals(
    path: Path, checkpoint_sha256: str
) -> dict[tuple[str, str], list[dict]]:
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("model_checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError("wave proposal checkpoint provenance drift")
            rows[(row["task"], row["dataset_hash"])].append(row)
    return rows


def select_micro_cohort(
    entries: list[dict],
    observations: dict[tuple[str, str], int],
    *,
    wave_id: str,
) -> list[dict]:
    """Select deterministic 64-dataset task cohorts (16/32/16)."""

    # Sealed families are the held-out zero-shot cohort and MUST never enter a
    # wave. The Rust planner filters them (jope_selfplay_wave.rs: `.filter(|row|
    # !sealed_families.contains(&row.source_family))`) and the Rust builder fails
    # closed on any that slip through ("cohort exposes a sealed source family").
    # This selector did not filter, so every cohort it wrote was unbuildable and
    # the queue starved behind it, repeatedly, all of 2026-07-28. It gets worse as
    # the catalog grows: 1,054 of 19,951 live rows (5.28%) are in a sealed family,
    # so a 128-dataset draw hits one with probability ~99.9%.
    sealed_families = {
        row["source_family"]
        for row in json.loads(SEALED_COHORT.read_text()).get("entries", [])
        if row.get("source_family")
    }

    # Same explicit-activation contract as the Rust selector: optional tasks
    # join only when the operator names them in JOPE_MICRO_COHORT_OPTIONAL_TASKS
    # AND their pool can fill a whole 64-dataset lane. Pool size alone must
    # never change wave shape.
    activated = tuple(
        task.strip()
        for task in os.environ.get("JOPE_MICRO_COHORT_OPTIONAL_TASKS", "").split(",")
        if task.strip()
    )
    selector_tasks = list(LEGACY_COHORT_TASKS) + [
        task for task in ("multiclass", "timeseries") if task in activated
    ]
    selected_all = []
    for task in selector_tasks:
        pool = [
            dict(row)
            for row in entries
            if row.get("task") == task
            and row.get("source_family") not in sealed_families
        ]
        if len(pool) < 64:
            if task not in LEGACY_COHORT_TASKS:
                continue
            raise RuntimeError(f"{task}: a micro-wave requires at least 64 datasets")

        def tie(row: dict, lane: str) -> bytes:
            return bytes.fromhex(
                sha256_text(
                    f"{wave_id}|{task}|{lane}|{row['dataset_hash']}"
                )
            )

        def observed(row: dict) -> int:
            return int(
                row.get("observation_count")
                or observations.get((task, row["dataset_hash"]), 0)
            )

        anchor_pool = [
            row
            for row in pool
            if row.get("stable_anchor")
            or row.get("golden42")
            or row.get("current_catalog", True)
        ]
        anchors = sorted(anchor_pool, key=lambda row: tie(row, "anchor"))[:16]
        used = {row["dataset_hash"] for row in anchors}
        least = sorted(
            (row for row in pool if row["dataset_hash"] not in used),
            key=lambda row: (observed(row), tie(row, "least")),
        )[:32]
        used.update(row["dataset_hash"] for row in least)
        hard = sorted(
            (row for row in pool if row["dataset_hash"] not in used),
            key=lambda row: (
                -float(
                    row.get("uncertainty")
                    or row.get("hardness")
                    or row.get("incumbent_regret")
                    or 0.0
                ),
                tie(row, "hard"),
            ),
        )[:16]
        cohort = [
            *({**row, "micro_lane": "stable_anchor"} for row in anchors),
            *({**row, "micro_lane": "least_observed"} for row in least),
            *({**row, "micro_lane": "hard_uncertain"} for row in hard),
        ]
        if len(cohort) != 64 or len(
            {row["dataset_hash"] for row in cohort}
        ) != 64:
            raise RuntimeError(f"{task}: deterministic micro-cohort is incomplete")
        selected_all.extend(cohort)
    return selected_all


# Per-task lane composition is frozen at 16/32/16; WHICH tasks a cohort carries
# is declared by the cohort itself ("tasks", defaulting to the two-task legacy
# set exactly like Rust's CohortPlan::tasks serde default). A fixed six-key map
# here rejected every valid multiclass lane -- epoch-06 lane 01 flagged it, and
# wave 546 proved it.
MICRO_LANE_SHAPE = {"stable_anchor": 16, "least_observed": 32, "hard_uncertain": 16}
LEGACY_COHORT_TASKS = ("regression", "binary")



__all__ = ['ROOT', 'CATALOG_POINTER', 'SEALED_COHORT', 'MICRO_COHORT_SCHEMA', 'CORPUS', 'BACKLOG', 'PAIR_LEDGER', 'CHAMPIONS', 'CURRENT_HOF', 'ACCEPTED', 'PROPOSALS', 'BASE_CHECKPOINT', 'SHARD_SCHEMA', 'WAVE', 'TRANSFORMS', 'MICRO_LANE_SHAPE', 'LEGACY_COHORT_TASKS', 'checkpoint', 'next_wave_id', 'task_valid', 'normalized_body', 'mutate', 'load_known_pairs', 'load_parents', 'load_checkpoint_proposals', 'select_micro_cohort']
