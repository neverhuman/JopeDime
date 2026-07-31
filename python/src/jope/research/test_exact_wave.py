#!/usr/bin/env python3
"""Build the next immutable strict-test wave from current policy-era parents."""

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

    selected_all = []
    for task in ("regression", "binary"):
        pool = [
            dict(row)
            for row in entries
            if row.get("task") == task
            and row.get("source_family") not in sealed_families
        ]
        if len(pool) < 64:
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


MICRO_COHORT_LANE_SIZES = {
    ("regression", "stable_anchor"): 16,
    ("regression", "least_observed"): 32,
    ("regression", "hard_uncertain"): 16,
    ("binary", "stable_anchor"): 16,
    ("binary", "least_observed"): 32,
    ("binary", "hard_uncertain"): 16,
}


def _validate_pending_cohort(
    existing: dict,
    *,
    wave_id: str,
    checkpoint_sha256: str,
    catalog_sha256: str,
    catalog_entries: list[dict],
) -> None:
    """Validate an already-written pending micro-cohort.

    A pending cohort declares ``"immutable": True``: once written it IS the
    cohort. It must NOT be re-derived and compared for equality, because
    ``select_micro_cohort`` is a function of live mutable state -- its
    ``least_observed`` lane ranks datasets by ``observed(row)``, and those
    counts grow as ingestion advances. Both the ORDER and the MEMBERSHIP of
    that lane therefore change minutes after the file is written, so an
    equality check against a fresh derivation can never succeed again and the
    phase deadlocks permanently.

    What must actually hold is the cohort's BINDING (it was built for this
    wave, this checkpoint and this catalog) and its STRUCTURE (the documented
    16/32/16 lanes over 64 distinct datasets per task, all drawn from the bound
    catalog). Those are checked here; a violation is real drift and still
    raises.
    """
    if existing.get("schema") != MICRO_COHORT_SCHEMA:
        raise RuntimeError(
            f"pending micro-cohort identity drift: schema {existing.get('schema')!r}"
        )
    for field, expected in (
        ("wave", wave_id),
        ("checkpoint_sha256", checkpoint_sha256),
        ("catalog_sha256", catalog_sha256),
    ):
        if existing.get(field) != expected:
            raise RuntimeError(
                f"pending micro-cohort identity drift: {field} "
                f"{existing.get(field)!r} != {expected!r}"
            )
    entries = existing.get("entries") or []
    lanes = Counter(
        (entry.get("task"), entry.get("micro_lane")) for entry in entries
    )
    if lanes != Counter(MICRO_COHORT_LANE_SIZES):
        raise RuntimeError(
            f"pending micro-cohort identity drift: lane sizes {dict(lanes)}"
        )
    identities = {(entry["task"], entry["dataset_hash"]) for entry in entries}
    if len(identities) != len(entries):
        raise RuntimeError(
            "pending micro-cohort identity drift: duplicate dataset entries"
        )
    catalogued = {
        (row.get("task"), row.get("dataset_hash")) for row in catalog_entries
    }
    missing = sorted(identities - catalogued)
    if missing:
        raise RuntimeError(
            "pending micro-cohort identity drift: "
            f"{len(missing)} datasets absent from the bound catalog, "
            f"first {missing[0]}"
        )


RUST_WAVE_PLANNER = Path("/home/ubuntu/jope-dime/target/release/jope_selfplay_wave")
PROPOSALS_POINTER = ROOT / "proposals" / "current.json"


def _rust_plan_micro_cohort(wave_id: str, checkpoint_path: Path | None = None) -> bool:
    """Derive the cohort with the Rust planner. Returns True when it wrote one.

    ONE cohort selector, not two. Two producers write
    `waves/pending/<wave>.cohort.json` -- this module, driven by the promotion
    controller's `seed_next_strict_wave` phase, and `jope_selfplay_supply`, which
    shells out to `jope_selfplay_wave plan`. Both ADOPT an existing pending
    cohort rather than re-deriving it, so whichever writes first decides the
    wave. When the two selectors disagree, the wave's compute allocation is
    decided by a race.

    They did disagree. Measured 2026-07-29: the Rust selector carries the
    breadth-over-depth admission policy (hard 128-observation cap, retirement of
    converged datasets, score-spread admission) and planned strict-wave-000173
    with 8 of 128 slots on datasets already at or over the cap. The Python
    selector below has no cap and planned strict-wave-000174 with 47 -- both
    `stable_anchor` lanes back to 16/16 capped at median 1,215 and 1,861
    observations each. It also counts observations from the PAIR REGISTRY rather
    than the corpus, which is a second, independent divergence.

    Delegating instead of copying the policy is deliberate: a second copy is the
    drift that produced this bug in the first place.

    Failure is never fatal. Any non-zero exit, timeout, missing binary or missing
    output leaves the caller to fall back to the in-module selector, so this can
    never block the controller.
    """
    if not RUST_WAVE_PLANNER.is_file():
        return False
    checkpoint_pointer = checkpoint_path
    if checkpoint_pointer is None:
        for candidate in (
            ROOT / "training" / "learning.json",
            ROOT / "training" / "accepted.json",
            PROPOSALS_POINTER,
        ):
            if candidate.is_file():
                checkpoint_pointer = candidate
                break
    if checkpoint_pointer is None:
        return False
    command = [
        str(RUST_WAVE_PLANNER),
        "plan",
        "--catalog-pointer", str(CATALOG_POINTER),
        "--checkpoint-pointer", str(checkpoint_pointer),
        "--corpus", str(CORPUS),
        "--sealed-cohort", str(SEALED_COHORT),
        "--output-root", str(ROOT / "waves"),
        "--wave-id", wave_id,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=900, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0:
        sys.stderr.write(
            "rust wave planner declined %s: %s\n"
            % (wave_id, (completed.stderr or "").strip()[:400])
        )
        return False
    return True


def plan_micro_cohort() -> dict:
    wave_id = next_wave_id()
    pointer = json.loads(CATALOG_POINTER.read_text())
    catalog = json.loads(Path(pointer["catalog"]).read_text())
    _, checkpoint_hash = checkpoint()
    target = ROOT / "waves" / "pending" / f"{wave_id}.cohort.json"
    if target.is_file():
        # Adopt the immutable pending cohort; do not re-derive it. Re-deriving
        # would both deadlock (see _validate_pending_cohort) and re-read the
        # corpus on every poll of a pending phase.
        payload = json.loads(target.read_text())
        if not payload.get("sealed_cohort_sha256"):
            # Self-heal a cohort written before this field was emitted. Backfill
            # is safe: the sealed cohort is frozen, so the digest is the value
            # the writer would have recorded, and no existing field is touched.
            # Without this the Rust builder rejects the file forever and the
            # queue starves behind a wave that can never be built.
            payload["sealed_cohort_sha256"] = sha256_file(SEALED_COHORT)
            atomic_json(target, payload)
        _validate_pending_cohort(
            payload,
            wave_id=wave_id,
            checkpoint_sha256=checkpoint_hash,
            catalog_sha256=pointer["catalog_sha256"],
            catalog_entries=catalog["entries"],
        )
    elif _rust_plan_micro_cohort(wave_id, checkpoint_path=None) and target.is_file():
        # Preferred path: the Rust planner wrote the cohort. Adopt it exactly as
        # the branch above adopts any pending cohort.
        payload = json.loads(target.read_text())
        _validate_pending_cohort(
            payload,
            wave_id=wave_id,
            checkpoint_sha256=checkpoint_hash,
            catalog_sha256=pointer["catalog_sha256"],
            catalog_entries=catalog["entries"],
        )
    else:
        # Fallback selector. This runs ONLY when the Rust planner could not be
        # invoked, and it does NOT implement the breadth-over-depth admission
        # policy (see _rust_plan_micro_cohort). Keeping it means a broken or
        # missing Rust binary can never stop `seed_next_strict_wave`; the cost is
        # that such a wave is planned uncapped, which is loud in the per-wave
        # accounting because its cohort carries ~8x the prior-observation mass.
        known = load_known_pairs()
        observations = {
            identity: len(semantics) for identity, semantics in known.items()
        }
        cohort = select_micro_cohort(
            catalog["entries"], observations, wave_id=wave_id
        )
        payload = {
            "schema": MICRO_COHORT_SCHEMA,
            "immutable": True,
            "wave": wave_id,
            "checkpoint_epoch": epoch_for_checkpoint(checkpoint_hash),
            "checkpoint_sha256": checkpoint_hash,
            "catalog_sha256": pointer["catalog_sha256"],
            "sealed_cohort_sha256": sha256_file(SEALED_COHORT),
            "entries": [
                {
                    "task": row["task"],
                    "dataset_hash": row["dataset_hash"],
                    "micro_lane": row["micro_lane"],
                }
                for row in cohort
            ],
        }
        atomic_json(target, payload)
    return {
        **payload,
        "cohort": str(target),
        "cohort_sha256": sha256_file(target),
    }


def wave_seed(
    *,
    task: str,
    dataset: str,
    wave_id: str,
    local: list[dict],
    dl: list[dict],
    common: list[dict],
) -> list[dict]:
    if len(dl) < 64:
        raise RuntimeError(f"{task}:{dataset} has only {len(dl)} checkpoint proposals")
    parents = [*local, *dl, *common]
    for row in parents:
        row.setdefault("semantic_sha256", sha256_text(row["canonical_genome"]))
    offspring = []
    for attempt in range(20_000):
        child = mutate(
            parents,
            task=task,
            dataset_hash=dataset,
            wave_id=wave_id,
            attempt=attempt,
        )
        if child is not None and task_valid(child["canonical_genome"], task):
            offspring.append(child)
        if len(offspring) >= 512:
            break
    fills = [*offspring[64:], *dl[64:], *common[96:], *local[32:]]
    return compose_seed(
        {
            "local_elite": local,
            "dl_proposal": dl,
            "common_winner": common,
            "offspring": offspring,
        },
        fills,
    )


def build(
    candidates_per_dataset: int = 256,
    max_shard: int = 100,
    *,
    proposals_path: Path = PROPOSALS,
    cohort_path: Path | None = None,
) -> dict:
    if not 1 <= max_shard <= 100:
        raise RuntimeError("max shard size must be in 1..100")
    if candidates_per_dataset != 256:
        raise RuntimeError("strict waves require exactly 256 candidates per dataset")
    if not CORPUS.is_file() or not CURRENT_HOF.is_file() or not CHAMPIONS.is_file():
        raise RuntimeError("strict corpus and champion indexes must exist")
    wave_id = next_wave_id()
    wave_final = ROOT / "waves" / wave_id
    manifests_final = ROOT / "manifests" / wave_id
    if manifests_final.exists():
        raise RuntimeError(f"immutable wave already exists: {wave_id}")
    # -------------------------------------------------------------------
    # Claim the identity ATOMICALLY, up front, instead of only checking it.
    #
    # Two lanes allocate wave ids independently: this builder and the Rust
    # micro-wave supply lane. Checking `wave_final.exists()` here and renaming
    # onto it ~35 minutes later is a TOCTOU window the width of the whole build,
    # and it closed for real on 2026-07-28: `seed_next_strict_wave` attempt 899
    # ran for 36 minutes and died with
    #   OSError: [Errno 39] Directory not empty:
    #     '.../waves/.strict-wave-000172.tmp-3799533' -> '.../waves/strict-wave-000172'
    # because the supply lane published its own strict-wave-000172 (schema
    # JopeRealtimeSelfplayWave.v1, 1,408 manifests, a DIFFERENT cohort) at
    # 23:43:00 while this build was in its pair-registry phase.
    #
    # `mkdir(exist_ok=False)` is a single atomic syscall, and the final
    # `os.replace` below still works because rename(2) may replace an EMPTY
    # directory. The other lane is diverted because both Rust allocators key on
    # the directory NAME only (`jope_selfplay_wave.rs::next_wave_id` :1039-1066
    # via `parse_wave_id`, `jope_selfplay_supply.rs` via `valid_wave_id`).
    #
    # An orphaned claim (this process SIGKILLed mid-build, which happens here --
    # `waves/.strict-wave-000005.tmp-867990` has been orphaned since 2026-07-26)
    # leaves an empty wave directory. That is INERT to every consumer: both
    # `jope_selfplay_supply.rs::validate_ready_wave` (:686-688) and
    # `jope_selfplay_queue.rs` (:1800-1802) skip a wave directory with no
    # `summary.json`. It costs one wave id out of 999,999.
    # -------------------------------------------------------------------
    wave_final.parent.mkdir(parents=True, exist_ok=True)
    try:
        wave_final.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"immutable wave already exists: {wave_id}") from error
    wave_temporary = wave_final.with_name(f".{wave_id}.tmp-{os.getpid()}")
    manifests_temporary = manifests_final.with_name(
        f".{wave_id}.tmp-{os.getpid()}"
    )
    wave_temporary.mkdir(parents=True, exist_ok=False)
    manifests_temporary.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CURRENT_HOF, wave_temporary / "hall-of-fame.json")
    hof_hash = sha256_file(wave_temporary / "hall-of-fame.json")
    _, checkpoint_hash = checkpoint()
    proposals = load_checkpoint_proposals(proposals_path, checkpoint_hash)
    authority = validate_catalog_manifest(CATALOG_POINTER, verify_contexts=False)
    pointer = json.loads(CATALOG_POINTER.read_text())
    catalog = json.loads(Path(pointer["catalog"]).read_text())
    if authority["row_count"] != len(catalog["entries"]):
        raise RuntimeError("catalog authority row count drift")
    known = load_known_pairs()
    local_parents, global_parents = load_parents()
    hof_pointer = json.loads((ROOT / "hof" / "current.json").read_text())
    seed_hof = json.loads(Path(hof_pointer["hall_of_fame"]).read_text())
    counts: dict[str, dict[str, int]] = {"regression": {}, "binary": {}}
    for (task, dataset), semantics in known.items():
        counts[task][dataset] = len(semantics)
    if cohort_path is not None:
        planned = json.loads(cohort_path.read_text())
        if (
            planned.get("schema") != MICRO_COHORT_SCHEMA
            or planned.get("wave") != wave_id
            or planned.get("checkpoint_sha256") != checkpoint_hash
            or planned.get("catalog_sha256") != pointer["catalog_sha256"]
            or len(planned.get("entries") or []) != 128
        ):
            raise RuntimeError("planned micro-cohort provenance drift")
        planned_rows = {
            (row["task"], row["dataset_hash"]): row
            for row in planned["entries"]
        }
        cohort = [
            {**row, **planned_rows[(row["task"], row["dataset_hash"])]}
            for row in catalog["entries"]
            if (row["task"], row["dataset_hash"]) in planned_rows
        ]
    else:
        cohort = select_micro_cohort(catalog["entries"], {
            (task, dataset): count
            for task, task_counts in counts.items()
            for dataset, count in task_counts.items()
        }, wave_id=wave_id)
    cohort_ids = {
        (row["task"], row["dataset_hash"]) for row in cohort
    }
    ordered = dataset_order(
        [
            row
            for row in catalog["entries"]
            if (row["task"], row["dataset_hash"]) in cohort_ids
        ],
        counts,
    )
    lanes = {
        (row["task"], row["dataset_hash"]): row["micro_lane"]
        for row in cohort
    }
    sys.path.insert(0, str(prime_source_root()))
    from jope.prime_label_to_corpus import GenomeEngineResolver

    resolver = GenomeEngineResolver()
    registry = PairRegistry()
    manifest_count = pair_count = deficient = 0
    reused_count = 0
    for ordinal, entry in enumerate(ordered):
        task, dataset = entry["task"], entry["dataset_hash"]
        local = [
            row
            for row in local_parents.get((task, dataset), [])
            if task_valid(row["canonical_genome"], task)
            and resolver.resolve(row["canonical_genome"], task_type=task) is not None
        ]
        common = [
            {
                **row,
                "source": "task_common_winner",
                "semantic_sha256": row["semantic_sha256"],
            }
            for row in seed_hof["common_winners"][task]
            if task_valid(row["canonical_genome"], task)
            and resolver.resolve(row["canonical_genome"], task_type=task) is not None
        ]
        if not local:
            local = [
                row
                for row in global_parents[task]
                if task_valid(row["canonical_genome"], task)
            ][:32]
        if not local:
            raise RuntimeError(f"{task}:{dataset} has no task-valid policy parent")
        selected = wave_seed(
            task=task,
            dataset=dataset,
            wave_id=wave_id,
            local=local,
            dl=[dict(row) for row in proposals.get((task, dataset), [])],
            common=common,
        )
        frozen_hash = sha256_text(
            json.dumps(
                {
                    "wave": wave_id,
                    "task": task,
                    "dataset_hash": dataset,
                    "semantics": [row["semantic_sha256"] for row in selected],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for row in selected:
            row.update(
                frozen_seed_sha256=frozen_hash,
                model_checkpoint_sha256=checkpoint_hash,
                hall_of_fame_sha256=hof_hash,
            )
        protocol = evaluation_protocol_sha256(
            task=task,
            metric="r2" if task == "regression" else "auc",
            score_tier="test_exact",
            train_sha256=entry["train_sha256"],
            test_sha256=entry["test_sha256"],
        )
        generated_at_ms = int(time.time() * 1000)
        owner = f"wave:{wave_id}"
        fresh = []
        reservation_outcomes = registry.reserve_batch(
            [
                {
                    "task": task,
                    "dataset_hash": dataset,
                    "semantic_sha256": row["semantic_sha256"],
                    "protocol_sha256": protocol,
                    "canonical_genome": row["canonical_genome"],
                }
                for row in selected
            ],
            owner=owner,
        )
        for row, created in zip(selected, reservation_outcomes, strict=True):
            if created:
                fresh.append(row)
            else:
                existing = registry.get(
                    task=task,
                    dataset_hash=dataset,
                    semantic_sha256=row["semantic_sha256"],
                    protocol_sha256=protocol,
                )
                if existing is not None and existing["reservation_owner"] == owner:
                    fresh.append(row)
                else:
                    reused_count += 1
        if not fresh:
            deficient += 1
        dataset_bindings = []
        for offset in range(0, len(fresh), max_shard):
            candidates = fresh[offset : offset + max_shard]
            manifest = {
                "schema": SHARD_SCHEMA,
                "strict_wave": wave_id,
                "micro_lane": lanes[(task, dataset)],
                "task": task,
                "dataset_hash": dataset,
                "checkpoint_epoch": epoch_for_checkpoint(checkpoint_hash),
                "checkpoint_sha256": checkpoint_hash,
                "catalog_sha256": pointer["catalog_sha256"],
                "protocol_sha256": protocol,
                "generated_at_ms": generated_at_ms,
                "train_csv": entry["train_csv"],
                "test_csv": entry["test_csv"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "label_authority": selfplay_authority(
                    task=task,
                    wave=wave_id,
                    frozen_seed_sha256=frozen_hash,
                    model_checkpoint_sha256=checkpoint_hash,
                    hall_of_fame_sha256=hof_hash,
                ),
                "candidates": candidates,
            }
            name = (
                f"{manifest_count:06d}-{task[0]}-{dataset}-"
                f"{offset // max_shard:02d}.json"
            )
            atomic_json(manifests_temporary / name, manifest)
            manifest_path = manifests_temporary / name
            manifest_hash = sha256_file(manifest_path)
            dataset_bindings.append(
                {
                    "manifest_path": str(manifests_final / name),
                    "manifest_sha256": manifest_hash,
                    "pairs": [
                        {
                            "task": task,
                            "dataset_hash": dataset,
                            "semantic_sha256": row["semantic_sha256"],
                            "protocol_sha256": protocol,
                        }
                        for row in candidates
                    ],
                }
            )
            manifest_count += 1
            pair_count += len(candidates)
        registry.bind_manifests(
            owner=owner,
            manifests=dataset_bindings,
        )
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(ordered):
            print(
                json.dumps(
                    {
                        "wave": wave_id,
                        "datasets_complete": ordinal + 1,
                        "datasets_total": len(ordered),
                        "fresh_pairs": pair_count,
                        "manifests": manifest_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "schema": "JopeTestOnlyStrictWave.v1",
        "wave_kind": "deterministic_128_dataset_micro_wave",
        "wave": wave_id,
        "checkpoint_epoch": epoch_for_checkpoint(checkpoint_hash),
        "catalog_sha256": pointer["catalog_sha256"],
        "hall_of_fame_sha256": hof_hash,
        "model_checkpoint_sha256": checkpoint_hash,
        "datasets": len(ordered),
        "task_datasets": {
            task: sum(row["task"] == task for row in ordered)
            for task in ("regression", "binary")
        },
        "micro_cohort": [
            {
                "task": row["task"],
                "dataset_hash": row["dataset_hash"],
                "lane": row["micro_lane"],
            }
            for row in cohort
        ],
        "candidates_per_dataset": candidates_per_dataset,
        "fresh_pairs": pair_count,
        "known_protocol_pairs_reused": reused_count,
        "seed_pairs": len(ordered) * 256,
        "manifests": manifest_count,
        "deficient_datasets": deficient,
        "max_candidates_per_manifest": max_shard,
    }
    atomic_json(wave_temporary / "summary.json", summary)
    wave_final.parent.mkdir(parents=True, exist_ok=True)
    manifests_final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(wave_temporary, wave_final)
    os.replace(manifests_temporary, manifests_final)
    registry.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-per-dataset", type=int, default=256)
    parser.add_argument("--max-shard", type=int, default=100)
    parser.add_argument("--proposals", type=Path, default=PROPOSALS)
    parser.add_argument("--cohort", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.candidates_per_dataset,
                args.max_shard,
                proposals_path=args.proposals,
                cohort_path=args.cohort,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
