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

from exact_wave_selection import ROOT, CATALOG_POINTER, SEALED_COHORT, MICRO_COHORT_SCHEMA, CORPUS, BACKLOG, PAIR_LEDGER, CHAMPIONS, CURRENT_HOF, ACCEPTED, PROPOSALS, BASE_CHECKPOINT, SHARD_SCHEMA, WAVE, TRANSFORMS, MICRO_LANE_SHAPE, LEGACY_COHORT_TASKS, checkpoint, next_wave_id, task_valid, normalized_body, mutate, load_known_pairs, load_parents, load_checkpoint_proposals, select_micro_cohort

def micro_cohort_lane_sizes(tasks):
    return {
        (task, lane): count
        for task in tasks
        for lane, count in MICRO_LANE_SHAPE.items()
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
    cohort_tasks = tuple(existing.get("tasks") or LEGACY_COHORT_TASKS)
    lanes = Counter(
        (entry.get("task"), entry.get("micro_lane")) for entry in entries
    )
    if lanes != Counter(micro_cohort_lane_sizes(cohort_tasks)):
        raise RuntimeError(
            f"pending micro-cohort identity drift: lane sizes {dict(lanes)} "
            f"for declared tasks {list(cohort_tasks)}"
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
            "tasks": sorted(
                {row["task"] for row in cohort},
                key=("regression", "binary", "timeseries", "multiclass").index,
            ),
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



__all__ = ['RUST_WAVE_PLANNER', 'PROPOSALS_POINTER', 'micro_cohort_lane_sizes', '_validate_pending_cohort', '_rust_plan_micro_cohort', 'plan_micro_cohort']
