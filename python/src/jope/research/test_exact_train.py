#!/usr/bin/env python3
"""Build strict-only policy rows and run the fused 60-pass xbabe2 trainer."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import sys
import random
import re
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import jope_spec_policy
from test_exact_common import atomic_json, prime_source_root, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
CORPUS_DB = ROOT / "corpus" / "strict-test-exact.sqlite"
CATALOG_POINTER = ROOT / "catalog" / "current.json"
WARM = Path(
    "/home/ubuntu/jope-flywheel/training/round1/checkpoints/"
    "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt"
)
WARM_SHA = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592"
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
STRIDE_FLOATS = 1536 + 512
TRAINER_POLICY = jope_spec_policy.policy("trainer")


def render_resource_block() -> str:
    """Bind this renderer to the central trainer resource policy."""

    return jope_spec_policy.render_resource_block("trainer")


def render_placement_block() -> str:
    """Bind this renderer to the central trainer capability probes."""

    return jope_spec_policy.render_placement_block("trainer")

# ---------------------------------------------------------------------------
# EQUAL PER-DATASET REPRESENTATION
#
# DATASET_ROW_CAP (128) was an upper BOUND, not a quota. A dataset with 2,612
# strict observations contributed 128 rows and one with 6 contributed 6, so the
# deepest datasets outweighed the shallowest ~21x in every epoch — and the deep
# datasets are exactly the ones the model can memorise. Bounds do not equalise;
# quotas do.
#
# EQUAL_QUOTA is a QUOTA: every dataset contributes EXACTLY this many rows to
# every epoch (all of them, if it owns fewer). Because a static corpus file
# cannot express "a different 32 each epoch", the corpus holds a per-dataset
# POOL and `train()` takes the equal-quota draw from that pool once per epoch
# with a per-epoch seed. Over 24 epochs a deep dataset still exposes its whole
# pool; no single epoch sees more of it than of any other dataset.
#
# N=32 chosen from the measured per-dataset observation distribution over the
# 5,531 datasets that own strict observations (regression train median 806, max
# 2,265; binary train median 220, max 2,612):
#   - Measured on the live corpus after the encodability filter, 88.8-90.7% of
#     datasets clear 32 in every one of the four task/split cells (binary train
#     1239/1376, regression train 3284/3695, binary val 127/143, regression val
#     381/420). The next tier up (64) drops that to 80-85%.
#   - The regression sub-32 tail is the transient first-touch tier (datasets
#     sitting at exactly 25 or 31 observations because they have been evaluated
#     once); they cross 32 on their next wave, so 32 does not pin the quota to
#     an artefact the way 24 would.
#   - 32 divides both per-task batch quotas exactly (round(256*0.75)=192 -> 6
#     datasets/batch, 256-192=64 -> 2 datasets/batch), so every batch carries a
#     constant number of equally-sized reward-pair groups.
EQUAL_QUOTA = 32
VALIDATION_QUOTA = 32
# Rows FETCHED per dataset before the encodability filter. ~17% of observations
# resolve to no grammar engine and are dropped, so the quota must be taken from
# post-filter survivors or it is not a quota at all. Measured: a 128-wide
# candidate window already puts every dataset that CAN clear 32 encodable rows
# over the line (a 64-wide window is enough; wider windows add nothing), so this
# keeps the fetch and resolve cost identical to the old DATASET_ROW_CAP=128.
CANDIDATE_ROW_CAP = 128
# recent_plus_replay only: slots reserved for the newest observations before the
# rest of the pool is filled by an even spread over the dataset's history.
RECENCY_SLOTS = CANDIDATE_ROW_CAP // 2
# Sub-quota datasets: repeat their rows to reach EQUAL_QUOTA. OFF by default.
# There is no per-row weight channel that reaches the reward heads (see
# `equal_quota_draw`), so replication is the only mechanism that would equalise
# their total contribution — but it equalises it with DUPLICATES, which teaches
# the imitation head to memorise the very datasets that have the least evidence,
# and contributes nothing to the reward heads (a duplicated pair has a zero
# score gap, and self_play_bt_loss weights pairs by |gap|).
QUOTA_TAIL_REPLICATE = False
# Bumped whenever the sampler configuration changes: the snapshot directory is
# keyed on it, otherwise a rebuild at an unchanged watermark silently returns the
# corpus built by the PREVIOUS sampler.
SAMPLER_TAG = f"eq{EQUAL_QUOTA}v{VALIDATION_QUOTA}c{CANDIDATE_ROW_CAP}p2"

# Recipe E is intentionally additive. Recipes A-D keep their existing batching
# and objectives byte-for-byte. Fractions are enforced over every five
# 256-row batches because 15% and 10% are not both integral in one such batch.
RECIPE_E_BATCH_MIX = {
    "uniform_broad_replay": 0.50,
    "exact_current_parent_pairs": 0.25,
    "completed_descendant_trajectories": 0.15,
    "validity_instability_hard_negatives": 0.10,
}
RECIPE_E_LOSS_WEIGHTS = {
    "sequence_imitation": 0.20,
    "immediate_pairwise_listwise_reward": 0.25,
    "descendant_value": 0.30,
    "directional_edit_operator_policy": 0.15,
    "validity_stability_calibration": 0.10,
}
# The (task, metric) pairs the trainer accepts into the frozen corpus. This is
# the SINGLE authority: the freeze CTE derives its WHERE clause from it, the
# packed-corpus task guard derives its superset check from it, and the CLI
# derives --task choices from it. Before this existed the CTE, the guard, and
# the CLI each hard-coded regression/binary separately, so every multiclass and
# timeseries observation was silently excluded from every checkpoint -- waves
# evaluated them, the corpus stored them, and the model never saw one.
ENABLED_TASK_METRICS = {
    "regression": "r2",
    "binary": "auc",
    "multiclass": "macro_f1",
    "timeseries": "negative_mase",
}
# Tasks that must ALWAYS be present in a packed corpus. The optional tasks join
# as their observations arrive; their absence is a data condition, not an error.
MANDATORY_TASKS = ("regression", "binary")
# Optional tasks join training through JopeOptionalTaskReadiness.v2. Their
# task-specific dataset floors are deliberately not interchangeable.
#
# Evaluation is deliberately NOT gated on it: waves keep running the task, so
# the evidence that clears the floor can actually accumulate. Gating wave
# generation instead would be a deadlock dressed as a safeguard.
#
# The number guards the shared trunk. Task share tracks dataset share (rows are
# capped per dataset by CANDIDATE_ROW_CAP, not per task), so admitting a task
# with 29 datasets does not give it 0.1% of the gradient harmlessly -- it gives
# those 29 datasets unlimited repetition across epochs, which is how a trunk
# memorises a handful of datasets and calls it a task.
OPTIONAL_TASK_DATASET_FLOORS = {"multiclass": 500, "timeseries": 42}
OPTIONAL_TASK_READINESS = ROOT / "ops" / "optional-task-readiness.v2.json"


def load_optional_task_readiness(path: Path = OPTIONAL_TASK_READINESS) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get("schema") != "JopeOptionalTaskReadiness.v2":
        raise RuntimeError("optional-task readiness schema drift")
    generated = int(value.get("generated_at_unix_ms") or 0)
    budget = int(value.get("freshness_budget_seconds") or 0)
    now = int(time.time() * 1000)
    if generated <= 0 or budget <= 0 or now < generated or now - generated > budget * 1000:
        raise RuntimeError("optional-task readiness is stale or future-invalid")
    inputs = value.get("input_digests")
    if (
        not isinstance(inputs, list)
        or not inputs
        or inputs != sorted(set(inputs))
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in inputs)
    ):
        raise RuntimeError("optional-task readiness input digests are invalid")
    rows = value.get("tasks")
    if (
        not isinstance(rows, list)
        or [row.get("task") for row in rows] != ["multiclass", "timeseries"]
    ):
        raise RuntimeError("optional-task readiness rows are incomplete or unordered")
    return value


def optional_task_training_allowed(task: str, row: dict, datasets: int) -> bool:
    floor = OPTIONAL_TASK_DATASET_FLOORS[task]
    common = (
        row.get("state") == "shadow_ready"
        and row.get("training_allowed") is True
        and int(row.get("distinct_dataset_identities") or 0) >= floor
        and datasets >= floor
        and int(row.get("valid_v2_cache_bindings") or 0) >= datasets
        and int(row.get("valid_context_transform_bindings") or 0) >= datasets
        and row.get("evaluator_retained") is True
        and int(row.get("task_valid_head_count") or 0) > 0
        and row.get("fresh_unbiased_shadow_evidence") is True
        and not row.get("blockers")
    )
    if task == "multiclass":
        return (
            common
            and int(row.get("minimum_required") or 0) == 500
            and row.get("primary_metric") == "macro_f1"
            and row.get("native_error_metric") == "one_minus_macro_f1"
        )
    return (
        common
        and datasets == 42
        and int(row.get("distinct_dataset_identities") or 0) == 42
        and int(row.get("minimum_required") or 0) == 42
        and int(row.get("minimum_rolling_origins_per_dataset") or 0) >= 3
        and row.get("causal_reduction_verified") is True
        and row.get("no_future_leakage") is True
        and row.get("primary_metric") == "negative_mase"
        and row.get("native_error_metric") == "mase"
    )


def eight_slot_task_weights(readiness: dict | None) -> dict[str, tuple[int, int]]:
    rows = {
        row["task"]: row for row in (readiness or {}).get("tasks", [])
        if isinstance(row, dict) and row.get("task") in OPTIONAL_TASK_DATASET_FLOORS
    }
    multiclass = rows.get("multiclass", {}).get("training_allowed") is True
    timeseries = rows.get("timeseries", {}).get("training_allowed") is True
    missing = int(not multiclass) + int(not timeseries)
    return {
        "regression": (6 + missing, 16),
        "binary": (6 + missing, 16),
        "multiclass": (2 if multiclass else 0, 16),
        "timeseries": (2 if timeseries else 0, 16),
    }


def require_optional_task_admission(
    task: str, readiness_path: Path | None
) -> dict | None:
    if task in MANDATORY_TASKS:
        return None
    path = readiness_path or OPTIONAL_TASK_READINESS
    readiness = load_optional_task_readiness(path)
    if readiness is None:
        raise ValueError(f"{task} training requires JopeOptionalTaskReadiness.v2")
    row = next(row for row in readiness["tasks"] if row["task"] == task)
    if row.get("training_allowed") is not True:
        raise ValueError(f"{task} is not admitted for optional-task training")
    return readiness


def trainable_task_metrics(con, readiness: dict | None = None) -> dict:
    """Tasks whose observations may enter a checkpoint, task -> metric.

    Mandatory tasks always qualify. Optional tasks require both corpus counts
    and a fresh V2 readiness authority; missing readiness is typed absence.
    """

    trainable = {
        task: metric
        for task, metric in ENABLED_TASK_METRICS.items()
        if task in MANDATORY_TASKS
    }
    if readiness is None:
        readiness = load_optional_task_readiness()
    readiness_rows = {
        row["task"]: row for row in (readiness or {}).get("tasks", [])
        if isinstance(row, dict) and row.get("task") in OPTIONAL_TASK_DATASET_FLOORS
    }
    for task, metric in sorted(ENABLED_TASK_METRICS.items()):
        if task in MANDATORY_TASKS:
            continue
        (datasets,) = con.execute(
            "SELECT count(DISTINCT dataset_hash) FROM observations "
            "WHERE task=? AND metric=?",
            (task, metric),
        ).fetchone()
        row = readiness_rows.get(task)
        if row is not None and optional_task_training_allowed(task, row, datasets):
            trainable[task] = metric
        else:
            print(
                f"[trainer] {task} withheld from training: {datasets} distinct "
                f"datasets; JopeOptionalTaskReadiness.v2 is absent or not "
                f"shadow-ready at task floor {OPTIONAL_TASK_DATASET_FLOORS[task]}; "
                f"evaluation continues",
                flush=True,
            )
    return trainable


RECIPE_E_MODEL_OUTPUTS = (
    "immediate_utility",
    "strict_win_probability",
    "descendant_value_h1",
    "descendant_value_h2",
    "descendant_value_h4",
    "descendant_value_h8",
    "mutation_operator_distribution",
    "parent_conditioned_structured_edit_distribution",
    "validity_probability",
    "instability_probability",
    "reward_quantile_lower",
    "reward_quantile_median",
    "reward_quantile_upper",
)


def recipe_e_training_trigger(
    new_eligible_labels: int,
    new_completed_descendant_receipts: int,
) -> bool:
    """Recipe E trains after either fixed information-mass threshold."""

    if min(new_eligible_labels, new_completed_descendant_receipts) < 0:
        raise ValueError("recipe E trigger counts cannot be negative")
    return (
        new_eligible_labels >= 50_000
        or new_completed_descendant_receipts >= 5_000
    )


def recipe_e_batch_lane_counts(batch_size: int, batch_index: int) -> dict[str, int]:
    """Return an exact deterministic 50/25/15/10 cumulative allocation."""

    if batch_size <= 0 or batch_index < 0 or batch_size % 8:
        raise ValueError("recipe E requires a positive batch size divisible by eight")
    # The pair lane must contain whole DL/GP pairs in every individual batch.
    pair_rows = batch_size // 4
    if pair_rows % 2:
        raise ValueError("recipe E pair lane must contain an even row count")
    previous_total = batch_index * batch_size
    current_total = (batch_index + 1) * batch_size
    broad_rows = batch_size // 2
    descendant_rows = (
        (current_total * 15) // 100 - (previous_total * 15) // 100
    )
    hard_rows = batch_size - broad_rows - pair_rows - descendant_rows
    return {
        "uniform_broad_replay": broad_rows,
        "exact_current_parent_pairs": pair_rows,
        "completed_descendant_trajectories": descendant_rows,
        "validity_instability_hard_negatives": hard_rows,
    }


def recipe_e_evidence_lane(metadata: dict, accepted_parent_sha256: str) -> str:
    """Classify one row into one non-overlapping recipe-E evidence lane."""

    pair_id = metadata.get("causal_pair_id")
    pair_lane = metadata.get("causal_lane")
    if (
        pair_id
        and pair_lane in {"native_dl_direct", "matched_gp_comparator"}
        and metadata.get("causal_accepted_parent_checkpoint_sha256")
        == accepted_parent_sha256
    ):
        return "exact_current_parent_pairs"
    if metadata.get("descendant_credit_receipt_sha256"):
        if metadata.get("descendant_censored", False):
            return "censored_descendant"
        if metadata.get("descendant_horizon") not in {1, 2, 4, 8}:
            raise ValueError("recipe E descendant row has an invalid horizon")
        return "completed_descendant_trajectories"
    if any(
        bool(metadata.get(name))
        for name in (
            "validity_hard_negative",
            "instability_hard_negative",
            "adversarial_hard_case",
            "candidate_invalid",
        )
    ):
        return "validity_instability_hard_negatives"
    return "uniform_broad_replay"


class GpuUtilizationMonitor:
    """Collect low-overhead utilization samples over the real training window."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.values: list[float] = []
        self.started = 0.0
        self.finished = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.started = time.monotonic()

        def sample() -> None:
            while not self._stop.is_set():
                observed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if observed.returncode == 0:
                    for line in observed.stdout.splitlines():
                        try:
                            self.values.append(float(line.strip()))
                        except ValueError:
                            pass
                self._stop.wait(15)

        self._thread = threading.Thread(
            target=sample, name="gpu-utilization-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20)
        self.finished = time.monotonic()

    def summary(self) -> dict:
        elapsed = max(0.0, self.finished - self.started)
        median = statistics.median(self.values) if self.values else None
        return {
            "samples": len(self.values),
            "sample_interval_seconds": 15,
            "elapsed_seconds": elapsed,
            "median_gpu_utilization_percent": median,
            "fifteen_minute_window_complete": elapsed >= 900 and len(self.values) >= 60,
            "meets_sixty_percent_gate": (
                elapsed >= 900
                and len(self.values) >= 60
                and median is not None
                and median >= 60.0
            ),
        }


def four_worker_policy_batch_iterator(train_v3, original):
    """Ordered four-worker CPU collation with pinned-memory CUDA prefetch."""

    def iterator(batch_rows, vocab, *, device, artifact_cache):
        if not str(device).startswith("cuda"):
            yield from original(
                batch_rows,
                vocab,
                device=device,
                artifact_cache=artifact_cache,
            )
            return
        torch = train_v3.torch
        row_iter = iter(batch_rows)
        futures = deque()

        def collate(rows):
            return train_v3._pin_policy_batch(
                train_v3.rows_to_policy_v3_batch(
                    rows,
                    vocab,
                    device="cpu",
                    artifact_cache=artifact_cache,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="packed-loader"
        ) as workers:
            for _ in range(8):
                try:
                    futures.append(workers.submit(collate, next(row_iter)))
                except StopIteration:
                    break
            copy_stream = torch.cuda.Stream(device=device)
            current_stream = torch.cuda.current_stream(device=device)
            while futures:
                host_batch = futures.popleft().result()
                try:
                    futures.append(workers.submit(collate, next(row_iter)))
                except StopIteration:
                    pass
                with torch.cuda.stream(copy_stream):
                    device_batch = train_v3._move_policy_batch(
                        host_batch, device, non_blocking=True
                    )
                    ready = torch.cuda.Event()
                    ready.record(copy_stream)
                current_stream.wait_event(ready)
                train_v3._record_policy_batch_stream(device_batch, current_stream)
                yield device_batch
                del host_batch

    return iterator

# The ONLY observation columns build_corpus reads. `SELECT *` inside the frozen
# CTE made the sorter carry all 31 columns of every strict row — including
# receipt_path, parentage_json, predictions_path and six unused sha256 hexes —
# which is what turned the freeze into 46 GB of temp writes. Projecting here is
# output-identical: every name below is dereferenced in phase 3, and no other
# column is.
OBSERVATION_COLUMNS = (
    "task",
    "dataset_hash",
    "semantic_sha256",
    "canonical_genome",
    "metric",
    "score",
    "runtime_seconds",
    "receipt_sha256",
    "model_checkpoint_sha256",
    "hall_of_fame_sha256",
    "created_at_ms",
    "strict_wave",
)
# Appended once per completed freeze. The dataset-count guardrail reads it, and
# it is the only place a freeze-to-freeze coverage regression is visible without
# re-reading 10 GB of JSONL.
FREEZE_HISTORY = "training/corpus-freeze-history.v1.jsonl"
# Lifted into ops/health-alarms.v1.json by jope_selfplay_alarms.
FREEZE_GUARD = "ops/corpus-freeze-guard.v1.json"
# ops/generalization-plan.md: "assert in the corpus builder that train and
# validation per-dataset medians are within 2x of each other, and fail the
# freeze if not". Asserted on the EPOCH-DRAW medians (rows the model actually
# sees in one epoch), not the pool medians: the pool is 128-wide for train and
# 32-wide for validation BY DESIGN, so a pool-median assert would fire on a
# healthy corpus and say nothing about the failure it exists to catch (train
# median 6 vs validation median 485, an 80x inversion that manufactured ~80% of
# the apparent overfit gap).
SPLIT_MEDIAN_RATIO_MAX = 2.0


def even_spread(items: list, cap: int) -> list:
    """Keep exactly ``min(len(items), cap)`` items, evenly spaced, deterministically.

    Used instead of the old ``ORDER BY created_at_ms LIMIT 128`` head slice. A
    dataset's oldest observations are all proposals from one checkpoint on one
    wave, so the head slice returned a single correlated cluster: measured over
    259 sampled datasets, the oldest-128 window yields 68 encodable rows on
    average while an even spread of the same width yields 89, and the spread
    clears the 32-row quota for 140/159 datasets against the head slice's 132.
    """
    count = len(items)
    if count <= cap:
        return list(items)
    return [items[(index * count) // cap] for index in range(cap)]


def task_reward(task: str, score: float) -> float:
    """Map a task's native metric onto the shared [-1, 1] reward scale.

    ``r2``, ``auc`` and ``macro_f1`` already live on a scale where clamping to
    [-1, 1] is meaningful: 1.0 is perfect and 0.0 is the trivial baseline.
    ``negative_mase`` does not. It is ``-MASE``, so parity with the seasonal-naive
    baseline is **-1.0**, not 0.0, and every model merely as good as naive or worse
    lands at or below the clamp floor. The bare ``max(-1, min(1, score))`` therefore
    collapsed that whole half of the timeseries distribution onto the single value
    -1.0 and squeezed all genuine skill into (-1, 0] -- the ranker would have seen
    almost no ordering to learn from, while the pipeline looked healthy.

    Shifting timeseries by +1 restores r2's semantics (1.0 perfect, 0.0 parity with
    the baseline, negative worse than it), after which one clamp is correct for all
    four tasks.
    """
    value = float(score)
    if task == "timeseries":
        value += 1.0
    return max(-1.0, min(1.0, value))


def equal_quota_draw(pool: list, quota: int, rng: random.Random) -> list:
    """Draw EXACTLY ``quota`` rows from ``pool`` (all of them if pool is smaller).

    No loss-weight compensation is applied to sub-quota datasets, and this is a
    deliberate, checked decision rather than an oversight:

      * ``TrainingRow.metadata["seq_weight"]`` exists and IS honoured — but only
        by the imitation loss. model_v3.loss_components folds it into
        ``row_weights`` for ``masked_sequence_nll`` alone (model_v3.py:403-406).
      * The reward heads have NO per-row weight at all. ``self_play_bt_loss`` and
        ``rank_pct_listwise_loss`` take (reward, target, ids) and weight PAIRS by
        the score gap; a dataset's influence on them is a function of how many of
        its rows share a batch, i.e. of COUNT, not of weight.
      * ``seq_weight`` is in any case unusable from here: train_v3.py:1409-1416
        OVERWRITES it to 1.0/0.05/0.0 for every row whose provenance is not one
        of ("corpus", "quant_report", "seed"), and ours is
        "internal_test_exact_selfplay".

    So COUNT is the only channel that reaches both the imitation head and the
    reward heads, which is exactly what an equal quota equalises. The residual
    under-contribution of the sub-quota tail is left visible in the manifest
    (``quota_stats.below_quota_*``) instead of being papered over with a weight
    that would silently apply to one of the four losses.
    """
    if len(pool) <= quota:
        if QUOTA_TAIL_REPLICATE and pool:
            return [pool[index % len(pool)] for index in range(quota)]
        return list(pool)
    order = list(range(len(pool)))
    rng.shuffle(order)
    return [pool[index] for index in sorted(order[:quota])]


def complete_causal_members(connection: sqlite3.Connection) -> tuple[dict, dict]:
    """Return receipt-exact pair metadata without consulting deduplicated source tags."""

    exists = connection.execute(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='causal_evaluation_receipts_v2')"
    ).fetchone()[0]
    if not exists:
        return {}, {
            "complete_pairs": 0,
            "task_counts": {},
            "exclusions": {},
            "source": "CausalEvaluationReceipt.v2 table absent",
        }
    completed_waves: dict[int, str] = {}
    waves_root = ROOT / "waves"
    if waves_root.is_dir():
        for directory in waves_root.glob("strict-wave-*"):
            try:
                wave = int(directory.name.removeprefix("strict-wave-"))
                summary = json.loads((directory / "summary.json").read_text())
            except (ValueError, OSError, json.JSONDecodeError):
                continue
            eligibility = (summary.get("wave_summary") or {}).get(
                "evidence_eligibility"
            ) or summary.get("evidence_eligibility") or {}
            if (
                summary.get("immutable") is True
                and eligibility.get("promotion") is True
                and eligibility.get("recovery_accounting_only") is not True
            ):
                completed_waves[wave] = str(summary.get("checkpoint_sha256") or "")

    rows = connection.execute(
        """
        SELECT receipt_sha256,candidate_sha256,task,context_identity,
               accepted_parent_checkpoint_sha256,wave,lane,pair_index,pair_id,
               score,success,proposal_valid,ancestry,cross_context_backfill,
               valid_lineage
          FROM causal_evaluation_receipts_v2
         WHERE canary=0 AND wave>=191
         ORDER BY pair_id,lane,candidate_sha256,receipt_sha256
        """
    )
    pairs: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        pairs[str(row["pair_id"])].append(row)

    members: dict[tuple[str, str], dict] = {}
    exclusions = Counter()
    task_counts = Counter()
    for pair_id, receipts in pairs.items():
        identities = {
            (
                row["task"],
                row["context_identity"],
                row["accepted_parent_checkpoint_sha256"],
                int(row["wave"]),
                int(row["pair_index"]),
            )
            for row in receipts
        }
        if len(identities) != 1:
            exclusions["ambiguous_identity"] += 1
            continue
        task, context, parent, wave, pair_index = next(iter(identities))
        if completed_waves.get(wave) != parent:
            exclusions["incomplete_or_wrong_parent_wave"] += 1
            continue
        lanes: dict[str, dict[str, list[sqlite3.Row]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in receipts:
            lane = str(row["lane"])
            if lane not in {"native_dl_direct", "matched_gp_comparator"}:
                continue
            repairable = (
                bool(row["success"])
                and bool(row["proposal_valid"])
                and not bool(row["cross_context_backfill"])
                and bool(row["valid_lineage"])
                and (
                    (lane == "native_dl_direct" and row["ancestry"] == "accepted_dl")
                    or (
                        lane == "matched_gp_comparator"
                        and row["ancestry"] == "none"
                    )
                )
                and row["score"] is not None
            )
            if repairable:
                lanes[lane][str(row["candidate_sha256"])].append(row)
        if set(lanes) != {"native_dl_direct", "matched_gp_comparator"} or any(
            len(candidates) != 1 for candidates in lanes.values()
        ):
            exclusions["partial_or_ambiguous_members"] += 1
            continue
        direct = next(iter(lanes["native_dl_direct"].values()))
        comparator = next(iter(lanes["matched_gp_comparator"].values()))
        direct_score = float(direct[0]["score"])
        comparator_score = float(comparator[0]["score"])
        common = {
            "causal_pair_id": pair_id,
            "causal_pair_index": pair_index,
            "causal_accepted_parent_checkpoint_sha256": parent,
            "causal_wave": wave,
            "causal_task": task,
            "causal_context_identity": context,
            "causal_score_gap": direct_score - comparator_score,
        }
        for lane, lane_rows in (
            ("native_dl_direct", direct),
            ("matched_gp_comparator", comparator),
        ):
            for row in lane_rows:
                members[(str(row["receipt_sha256"]), str(row["candidate_sha256"]))] = {
                    **common,
                    "causal_lane": lane,
                }
        task_counts[str(task)] += 1
    return members, {
        "complete_pairs": sum(task_counts.values()),
        "task_counts": dict(task_counts),
        "exclusions": dict(exclusions),
        "source": "CausalEvaluationReceipt.v2 plus immutable completed wave summaries",
    }


def quota_distribution(counts: list[int], quota: int) -> dict:
    """Per-dataset row-count distribution — the acceptance evidence for the quota."""
    if not counts:
        return {"datasets": 0, "rows": 0}
    ordered = sorted(counts)
    return {
        "datasets": len(ordered),
        "rows": sum(ordered),
        "quota": quota,
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "std": round(statistics.pstdev(ordered), 4),
        "at_quota_datasets": sum(1 for value in ordered if value == quota),
        "below_quota_datasets": sum(1 for value in ordered if value < quota),
        "below_quota_rows": sum(value for value in ordered if value < quota),
        "above_quota_datasets": sum(1 for value in ordered if value > quota),
    }


def freeze_history_rows(root: Path = ROOT) -> list[dict]:
    """Every completed freeze, oldest first. Missing/corrupt lines are skipped."""
    path = root / FREEZE_HISTORY
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_freeze_history(record: dict, root: Path = ROOT) -> None:
    path = root / FREEZE_HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_freeze_guard(record: dict, root: Path = ROOT) -> dict:
    """Coverage guardrail: alarm when the corpus dataset count FALLS.

    ops/generalization-plan.md asked for this as an ALARM rather than a hard
    failure, and deliberately so: evidence only accumulates, so a fall is always
    a sampler/catalog defect rather than a property of the data — but refusing to
    freeze on it would leave the trainer pinned to the OLD corpus, which is the
    exact starvation this workstream exists to end. Freeze, then shout.
    """
    prior = freeze_history_rows(root)
    best = max((int(row.get("datasets_total") or 0) for row in prior), default=0)
    best_at = next(
        (
            int(row.get("watermark") or 0)
            for row in reversed(prior)
            if int(row.get("datasets_total") or 0) == best
        ),
        0,
    )
    datasets = int(record["datasets_total"])
    regressed = best > 0 and datasets < best
    guard = {
        "schema": "JopeCorpusFreezeGuard.v1",
        "generated_at_ms": int(time.time() * 1000),
        "alarm": "corpus_dataset_count_fell",
        "regressed": regressed,
        "severity": "critical" if regressed else "ok",
        "observed": (
            f"{datasets} datasets at watermark {record['watermark']} "
            f"({record['corpus_mode']}/{record['sampler']})"
        ),
        "threshold": (
            f">={best} datasets (high-water mark set at watermark {best_at})"
            if best
            else "no prior freeze recorded"
        ),
        "watermark": record["watermark"],
        "datasets_total": datasets,
        "previous_best_datasets": best,
        "previous_best_watermark": best_at,
        "per_split_datasets": record.get("per_split_datasets", {}),
        "epoch_draw_medians": record.get("epoch_draw_medians", {}),
    }
    atomic_json(root / FREEZE_GUARD, guard)
    return guard


def split_median_divergence(epoch_draw_stats: dict) -> dict:
    """train-vs-validation per-dataset epoch-draw median ratio, per task and overall.

    Returns {scope: {"train": m, "validation": m, "ratio": r}}. `ratio` is the
    larger median over the smaller, so it is >= 1 and direction-free; a scope
    with no rows on one side is reported with ratio None and is not asserted on
    (a task can legitimately be absent from a split only if the freeze already
    failed the both-splits-populated check above).
    """
    medians = {
        key: float(value["median"])
        for key, value in epoch_draw_stats.items()
        if value.get("datasets")
    }
    scopes: dict[str, dict] = {}
    tasks = sorted({key.split("/", 1)[0] for key in medians})
    for task in tasks:
        train = medians.get(f"{task}/train")
        validation = medians.get(f"{task}/validation")
        scopes[task] = {
            "train": train,
            "validation": validation,
            "ratio": (
                max(train, validation) / min(train, validation)
                if train and validation
                else None
            ),
        }
    return scopes


def create_catalog_binding(
    *,
    pointer_path: Path = CATALOG_POINTER,
    root: Path = ROOT,
) -> Path:
    pointer = json.loads(pointer_path.read_text())
    catalog_path = Path(pointer["catalog"]).resolve()
    data_path = Path(pointer["data"]).resolve()
    if sha256_file(catalog_path) != pointer["catalog_sha256"]:
        raise RuntimeError("catalog pointer content hash drift")
    if sha256_file(data_path) != pointer["data_sha256"]:
        raise RuntimeError("catalog data content hash drift")
    bindings = root / "training" / "catalog-bindings"
    snapshots = bindings / "pointer-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    canonical_pointer = json.dumps(
        pointer, sort_keys=True, separators=(",", ":")
    ).encode()
    pointer_identity = hashlib.sha256(canonical_pointer).hexdigest()
    pointer_snapshot = snapshots / f"{pointer_identity}.json"
    if pointer_snapshot.is_file():
        if json.loads(pointer_snapshot.read_text()) != pointer:
            raise RuntimeError("immutable catalog pointer snapshot drift")
    else:
        atomic_json(pointer_snapshot, pointer)
    binding = {
        "schema": "CatalogBinding.v1",
        "catalog_path": str(catalog_path),
        "catalog_sha256": pointer["catalog_sha256"],
        "data_path": str(data_path),
        "data_sha256": pointer["data_sha256"],
        "pointer_snapshot_path": str(pointer_snapshot),
        "pointer_snapshot_sha256": sha256_file(pointer_snapshot),
        "created_at_ms": int(pointer_path.stat().st_mtime_ns // 1_000_000),
    }
    binding_identity = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding_path = bindings / f"{binding_identity}.json"
    if binding_path.is_file():
        if json.loads(binding_path.read_text()) != binding:
            raise RuntimeError("immutable catalog binding drift")
    else:
        atomic_json(binding_path, binding)
    return binding_path


def resolve_catalog_binding(
    manifest: dict,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        candidates = [explicit]
    elif manifest.get("catalog_binding"):
        candidates = [Path(manifest["catalog_binding"])]
    else:
        candidates = sorted(
            (ROOT / "training" / "catalog-bindings").glob("*.json")
        )
    for path in candidates:
        if not path.is_file():
            continue
        binding = json.loads(path.read_text())
        if (
            binding.get("schema") == "CatalogBinding.v1"
            and binding.get("catalog_sha256") == manifest["catalog_sha256"]
            and sha256_file(Path(binding["catalog_path"]))
            == binding["catalog_sha256"]
            and sha256_file(Path(binding["data_path"])) == binding["data_sha256"]
        ):
            return path
    live = create_catalog_binding()
    binding = json.loads(live.read_text())
    if binding["catalog_sha256"] != manifest["catalog_sha256"]:
        raise RuntimeError(
            "no historical CatalogBinding.v1 matches the frozen corpus"
        )
    return live


def ensure_packed(
    manifest_path: Path,
    *,
    catalog_binding_path: Path | None = None,
) -> dict:
    from test_exact_packed_corpus import materialize

    manifest = json.loads(manifest_path.read_text())
    binding = resolve_catalog_binding(manifest, catalog_binding_path)
    return materialize(
        manifest_path,
        ROOT / "training" / "packed",
        catalog_binding_path=binding,
    )


def publish_corpus_pointer(
    manifest_path: Path,
    packed_manifest: Path,
    watermark: int,
    *,
    root: Path = ROOT,
) -> dict:
    """Write training/current-corpus.json, but NEVER backwards.

    `jope_selfplay_trainer_watchdog.rs:594` hard-fails its cycle when this
    pointer regresses ("training corpus pointer regressed from X to Y"), so a
    single backwards write stops the DL trainer from ever being resubmitted.
    Regression is now structurally possible because two independent producers
    advance this file: the promotion tournament (at ITS pinned watermark) and the
    standalone freezer (at the live COUNT(*)). Monotonicity is enforced here, in
    the one function both of them go through, rather than in either of them.
    """
    path = root / "training" / "current-corpus.json"
    pointer = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packed_manifest": str(packed_manifest),
        "packed_manifest_sha256": sha256_file(packed_manifest),
        "watermark": int(watermark),
    }
    if path.is_file():
        try:
            live = int(json.loads(path.read_text()).get("watermark") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            live = 0
        if live > int(watermark):
            return {
                **pointer,
                "published": False,
                "live_watermark": live,
                "reason": "pointer is monotone; live watermark is ahead",
            }
    atomic_json(path, pointer)
    return {**pointer, "published": True}


def publish_corpus_freeze_state(
    *,
    watermark: int,
    stage: str,
    manifest_path: Path,
    catalog_binding_path: Path,
    packed_manifest_path: Path | None = None,
) -> dict:
    stages = {"rows_frozen": 0, "packed": 1, "published": 2}
    if stage not in stages:
        raise ValueError("invalid corpus freeze stage")
    path = (
        ROOT
        / "ops"
        / "corpus-freeze-state-v2"
        / f"watermark-{watermark:09d}.json"
    )
    prior = json.loads(path.read_text()) if path.is_file() else None
    if prior is not None and stages[prior["stage"]] > stages[stage]:
        return prior
    value = {
        "schema": "CorpusFreezeState.v2",
        "watermark": int(watermark),
        "catalog_binding_path": str(catalog_binding_path),
        "catalog_binding_sha256": sha256_file(catalog_binding_path),
        "selected_rows_path": str(manifest_path),
        "selected_rows_sha256": sha256_file(manifest_path),
        "packed_corpus_path": (
            str(packed_manifest_path) if packed_manifest_path is not None else None
        ),
        "packed_corpus_sha256": (
            sha256_file(packed_manifest_path)
            if packed_manifest_path is not None
            else None
        ),
        "stage": stage,
        "revision": int((prior or {}).get("revision", -1)) + 1,
    }
    atomic_json(path, value)
    return value


def accepted_checkpoint(task: str | None = None) -> tuple[Path, str]:
    if task not in {None, "binary", "regression"}:
        raise ValueError("accepted checkpoint task must be binary or regression")
    pointer = ROOT / "training" / "accepted-by-task.json"
    if not pointer.is_file():
        raise RuntimeError("authoritative accepted-by-task pointer is absent")
    value = json.loads(pointer.read_text())
    if value.get("schema") != "JopeSelfplayAcceptedByTask.v1":
        raise RuntimeError("authoritative accepted-by-task pointer schema drift")
    tasks = value.get("tasks") or {}
    if task is None:
        binary = tasks.get("binary") or {}
        regression = tasks.get("regression") or {}
        if (
            binary.get("checkpoint_sha256") != regression.get("checkpoint_sha256")
            or binary.get("checkpoint") != regression.get("checkpoint")
        ):
            raise RuntimeError(
                "task checkpoints diverged; a task argument is required"
            )
        row = binary
        task_label = "shared"
    else:
        row = tasks.get(task) or {}
        task_label = task
    path = Path(row["checkpoint"])
    digest = row["checkpoint_sha256"]
    if path.is_file() and sha256_file(path) == digest:
        return path, digest
    raise RuntimeError(
        f"{task_label} accepted checkpoint pointer failed content verification"
    )


def build_corpus(
    min_rows: int = 1,
    *,
    freeze_watermark: int | None = None,
    corpus_mode: str = "full_consolidation",
    catalog_binding_path: Path | None = None,
) -> dict:
    if not CORPUS_DB.is_file():
        raise RuntimeError("strict corpus database does not exist yet")
    if corpus_mode not in {"recent_plus_replay", "full_consolidation"}:
        raise ValueError("unsupported continuation corpus mode")
    catalog_binding_path = (
        catalog_binding_path or create_catalog_binding()
    ).resolve()
    binding = json.loads(catalog_binding_path.read_text())
    if binding.get("schema") != "CatalogBinding.v1":
        raise RuntimeError("unsupported catalog binding")
    binding_sha256 = sha256_file(catalog_binding_path)
    con = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    # The frozen CTE sorts ~2-3M wide rows by a 5-column key and the per-dataset
    # ROW_NUMBER() window re-sorts that output. On SQLite's default ~2 MB page
    # cache every one of those sorts spills to disk: a build was observed doing
    # 122 GB of reads and 46 GB of temp writes against an 8 GB database and still
    # not finishing after 17 minutes. Give the sorter real memory instead.
    con.execute("PRAGMA cache_size = -4000000")  # ~4 GB, negative = KiB
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA mmap_size = 8589934592")  # 8 GB mmap of the DB file
    pointer = {
        "catalog": binding["catalog_path"],
        "catalog_sha256": binding["catalog_sha256"],
        "data": binding["data_path"],
        "data_sha256": binding["data_sha256"],
    }
    catalog = json.loads(Path(binding["catalog_path"]).read_text())
    entries = {
        (entry["task"], entry["dataset_hash"]): entry
        for entry in catalog["entries"]
    }
    total = int(con.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
    if total < min_rows:
        raise RuntimeError(f"strict corpus has {total} rows; needs {min_rows}")
    watermark = total if freeze_watermark is None else int(freeze_watermark)
    if watermark < min_rows or watermark > total:
        raise RuntimeError(
            f"cannot freeze watermark {watermark} from {total} strict labels"
        )
    causal_members, causal_pair_stats = complete_causal_members(con)
    # SAMPLER_TAG is part of the identity: the snapshot used to be keyed on the
    # watermark alone, so changing the sampler and rebuilding at an unchanged
    # watermark returned the OLD corpus from the early-return below without a
    # word of warning.
    legacy_snapshot = (
        ROOT
        / "training"
        / f"snapshot-{watermark:09d}-{corpus_mode}-{SAMPLER_TAG}"
    )
    snapshot = legacy_snapshot.with_name(
        f"{legacy_snapshot.name}-c{binding_sha256[:12]}"
    )
    if legacy_snapshot.exists():
        legacy_manifest = legacy_snapshot / "corpus-manifest.json"
        if (
            legacy_manifest.is_file()
            and json.loads(legacy_manifest.read_text()).get("catalog_sha256")
            == binding["catalog_sha256"]
        ):
            snapshot = legacy_snapshot
    # The snapshot identity is fully determined by (watermark, mode, sampler),
    # all of which are known here — so the cached-snapshot check MUST happen
    # before the candidate query, not after it. It used to sit below a
    # `.fetchall()` of the whole windowed query, so every idempotent re-freeze
    # paid the full multi-GB scan and sort and then threw the result away.
    if snapshot.exists():
        manifest_path = snapshot / "corpus-manifest.json"
        if not manifest_path.is_file():
            con.close()
            raise RuntimeError(f"incomplete corpus snapshot requires quarantine: {snapshot}")
        manifest = json.loads(manifest_path.read_text())
        con.close()
        publish_corpus_freeze_state(
            watermark=watermark,
            stage="rows_frozen",
            manifest_path=manifest_path,
            catalog_binding_path=catalog_binding_path,
        )
        packed = ensure_packed(
            manifest_path, catalog_binding_path=catalog_binding_path
        )
        packed_manifest_path = (
            ROOT
            / "training"
            / "packed"
            / packed["binding_sha256"]
            / "manifest.json"
        )
        publish_corpus_freeze_state(
            watermark=watermark,
            stage="packed",
            manifest_path=manifest_path,
            catalog_binding_path=catalog_binding_path,
            packed_manifest_path=packed_manifest_path,
        )
        publication = publish_corpus_pointer(
            manifest_path,
            packed_manifest_path,
            watermark,
        )
        if publication["published"]:
            publish_corpus_freeze_state(
                watermark=watermark,
                stage="published",
                manifest_path=manifest_path,
                catalog_binding_path=catalog_binding_path,
                packed_manifest_path=packed_manifest_path,
            )
        return manifest
    projection = ",".join(OBSERVATION_COLUMNS)
    # The ORDER BY ... LIMIT exists ONLY to decide which rows are inside the
    # watermark. When the watermark covers every row in the table it cannot bind
    # (the strict predicate selects a subset of `total` rows and the limit is
    # `total`), so the whole 4.6M-row five-column sort is dead work. Dropping it
    # is output-identical, not an approximation: membership of `frozen` is
    # unchanged, and `dataset_rank` is assigned by its own window ORDER BY
    # (created_at_ms, semantic_sha256) which does not depend on the CTE's order.
    covers_every_row = watermark >= total
    # Four-task freeze: the WHERE clause is derived from ENABLED_TASK_METRICS so
    # the CTE can never silently disagree with the guard or the CLI about which
    # observations are learnable. Each task is pinned to its OWN metric -- an
    # observation whose metric drifted from its task's contract stays excluded.
    # ...and an optional task additionally has to clear the dataset floor before
    # its observations are learnable. This fails CLOSED: a task under the floor
    # is excluded from training while its evaluation continues, so the evidence
    # that clears the floor keeps accumulating. Mandatory tasks are never gated.
    trainable_tasks = trainable_task_metrics(con)
    task_predicate = " OR ".join(
        f"(task='{task}' AND metric='{metric}')"
        for task, metric in sorted(trainable_tasks.items())
    )
    if covers_every_row:
        frozen_cte = f"""
            WITH frozen AS (
              SELECT {projection} FROM observations
              WHERE {task_predicate}
            )
        """
        frozen_params: tuple = ()
    else:
        frozen_cte = f"""
            WITH frozen AS (
              SELECT {projection} FROM observations
              WHERE {task_predicate}
              ORDER BY created_at_ms,receipt_sha256,task,dataset_hash,semantic_sha256
              LIMIT ?
            )
        """
        frozen_params = (watermark,)
    # ONE candidate query for BOTH corpus modes and BOTH splits. The old code had
    # three separate queries and the split-specific cap only existed on the
    # recent_plus_replay path, so full_consolidation quietly handed validation
    # 128 rows per dataset where the other mode handed it 32.
    #
    # ((dataset_rank - 1) * cap) % dataset_total < cap keeps EXACTLY
    # min(dataset_total, cap) rows, evenly spaced across the dataset's history:
    # the residues (r-1)*cap mod total are the multiples of gcd(cap, total), each
    # hit gcd times, and there are cap/gcd of them below cap. It is the SQL twin
    # of even_spread(). The second clause reserves the newest RECENCY_SLOTS rows
    # for recent_plus_replay (and is unreachable when recency=0, since
    # dataset_rank <= dataset_total always).
    recency = RECENCY_SLOTS if corpus_mode == "recent_plus_replay" else 0
    cursor = con.execute(
        frozen_cte
        + """
          , ranked AS (
            SELECT frozen.*,
              ROW_NUMBER() OVER (
                PARTITION BY task,dataset_hash
                ORDER BY created_at_ms,semantic_sha256
              ) AS dataset_rank,
              COUNT(*) OVER (PARTITION BY task,dataset_hash) AS dataset_total
            FROM frozen
          )
          SELECT * FROM ranked
          WHERE ((dataset_rank - 1) * ?) % dataset_total < ?
             OR dataset_rank > dataset_total - ?
          ORDER BY task,dataset_hash,dataset_rank
        """,
        (*frozen_params, CANDIDATE_ROW_CAP, CANDIDATE_ROW_CAP, recency),
    )

    sys.path.insert(0, str(prime_source_root()))
    from jope.prime_label_to_corpus import GenomeEngineResolver
    from jope.schema import TrainingRow

    resolver = GenomeEngineResolver()
    kept, skipped = Counter(), Counter()
    per_dataset_rows: Counter = Counter()
    task_counts: Counter = Counter()
    candidate_label_count = 0

    # --- phase 1: catalog + encodability filter, BEFORE any quota -------------
    # This ordering is the whole point. ~17% of observations resolve to no
    # grammar engine, and the old code capped first and filtered second, so the
    # "cap" it enforced was on candidates while the number that actually reached
    # the model was whatever survived — 18 rows for one dataset and 128 for
    # another out of the same nominal cap of 128.
    #
    # STREAMED, not `.fetchall()`. At the 1.8M watermark the materialised result
    # was ~366k rows; at COUNT(*) it is ~6x that, and the discarded ~17%
    # unencodable and not-in-catalog rows never need to exist in memory at all.
    survivors: dict[tuple[str, str], list] = defaultdict(list)
    for row in cursor:
        candidate_label_count += 1
        task_counts[row["task"]] += 1
        identity = (row["task"], row["dataset_hash"])
        if entries.get(identity) is None:
            skipped["not_in_frozen_catalog"] += 1
            continue
        engine = resolver.resolve(row["canonical_genome"], task_type=row["task"])
        if engine is None:
            skipped["unencodable"] += 1
            continue
        survivors[identity].append((row, engine))
    # Superset-over-enabled, not equality: the mandatory backbone (regression,
    # binary) must be present, unknown tasks must never slip in, and the optional
    # tasks (multiclass, timeseries) join as their first observations arrive.
    # The old equality check would have REJECTED the corpus the moment the first
    # multiclass observation landed -- turning new-task success into a trainer
    # outage.
    present = set(task_counts)
    if not present.issuperset(MANDATORY_TASKS):
        con.close()
        raise RuntimeError(
            f"shared trainer needs the mandatory tasks {sorted(MANDATORY_TASKS)}, "
            f"found {dict(task_counts)}"
        )
    if not present.issubset(ENABLED_TASK_METRICS):
        con.close()
        raise RuntimeError(
            f"corpus contains tasks outside the enabled contract: {dict(task_counts)}"
        )

    blob = Path(pointer["data"]).read_bytes()
    temporary = snapshot.with_name(f".{snapshot.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    handles = {
        "train": (temporary / "corpus.train.jsonl").open("w"),
        "validation": (temporary / "corpus.val.jsonl").open("w"),
    }
    try:
        # --- phase 2: build the per-dataset pool ------------------------------
        # train      -> a pool the per-epoch equal-quota draw samples from
        # validation -> exactly VALIDATION_QUOTA rows, fixed for every epoch, so
        #               the selection metric moves for learning reasons only
        pools: dict[tuple[str, str], list] = {}
        for identity, encodable in survivors.items():
            split = entries[identity]["split"]
            if split == "validation":
                pools[identity] = even_spread(encodable, VALIDATION_QUOTA)
                continue
            if corpus_mode == "recent_plus_replay":
                recent = encodable[-RECENCY_SLOTS:]
                replay = even_spread(
                    encodable[: max(0, len(encodable) - RECENCY_SLOTS)],
                    CANDIDATE_ROW_CAP - len(recent),
                )
                pools[identity] = [*replay, *recent]
            else:
                pools[identity] = even_spread(encodable, CANDIDATE_ROW_CAP)

        # --- phase 3: materialise ---------------------------------------------
        for identity in sorted(pools):
            entry = entries[identity]
            split = entry["split"]
            quota = VALIDATION_QUOTA if split == "validation" else EQUAL_QUOTA
            pool = pools[identity]
            # The dataset embedding is per DATASET, not per row — unpack once.
            offset = int(entry["catalog_index"]) * STRIDE_FLOATS * 4
            values = struct.unpack(
                f"<{STRIDE_FLOATS}f",
                blob[offset : offset + STRIDE_FLOATS * 4],
            )
            embedding = list(values[:1536])
            target_embedding = list(values[1536:])
            for row, engine in pool:
                reward = task_reward(row["task"], row["score"])
                strict_wave = (
                    row["strict_wave"]
                    if "strict_wave" in row.keys() and row["strict_wave"]
                    else "strict-wave-000000"
                )
                metadata = {
                    "provenance": "internal_test_exact_selfplay",
                    "task_type": row["task"],
                    "split": split,
                    "source_family": entry["source_family"],
                    "dataset_hash": row["dataset_hash"],
                    "semantic_plan_hash": row["semantic_sha256"],
                    "score_tier": "test_exact",
                    "test_adaptive": True,
                    "benchmark_valid": False,
                    "selection_bias": "test-exact-reused",
                    "generation": strict_wave,
                    "score_robust": reward,
                    "oof_r2_raw": reward,
                    "reward_pair_identity": (
                        f"test-exact:{row['task']}:{row['dataset_hash']}:{row['metric']}"
                    ),
                    # Left at 1.0 on purpose. train_v3.py:1409-1416 overwrites
                    # seq_weight for every "generated" provenance (ours included)
                    # before the first batch is built, and it reaches only the
                    # imitation loss anyway — see equal_quota_draw's docstring.
                    "seq_weight": 1.0,
                    # Read by the per-epoch sampler in train().
                    "epoch_quota": quota,
                    "quota_pool_size": len(pool),
                    "quota_deficit": max(0, quota - len(pool)),
                    "observation_rank": int(row["dataset_rank"]),
                    "dataset_observation_total": int(row["dataset_total"]),
                    "promotion_eligible": True,
                    "promotion_scope": "internal-selected-test-flywheel-only",
                    "benchmark_promotion_eligible": False,
                    "label_created_at_ms": int(row["created_at_ms"]),
                    "model_checkpoint_sha256": row["model_checkpoint_sha256"],
                    "hall_of_fame_sha256": row["hall_of_fame_sha256"],
                    "receipt_sha256": row["receipt_sha256"],
                }
                causal = causal_members.get(
                    (row["receipt_sha256"], row["semantic_sha256"])
                )
                if causal is not None:
                    metadata.update(causal)
                training_row = TrainingRow(
                    dataset_id=row["dataset_hash"],
                    source="internal_test_exact_selfplay",
                    embedding=embedding,
                    target_embedding=target_embedding,
                    genome=row["canonical_genome"],
                    canonical_genome=row["canonical_genome"],
                    engine=engine,
                    valid=True,
                    metric=row["metric"],
                    score=reward,
                    cost_ms=float(row["runtime_seconds"]) * 1000.0,
                    metadata=metadata,
                )
                handles[split].write(json.dumps(training_row.to_json()) + "\n")
                kept[f"{row['task']}/{split}"] += 1
            per_dataset_rows[(identity[0], split, identity[1])] = len(pool)
    finally:
        for handle in handles.values():
            handle.close()
        con.close()

    # Per-task/split acceptance evidence, recorded in the manifest so a corpus
    # can be audited without re-reading 6 GB of JSONL.
    pool_counts: dict[str, list[int]] = defaultdict(list)
    for (task, split, _), count in per_dataset_rows.items():
        pool_counts[f"{task}/{split}"].append(count)
    quota_stats = {
        key: quota_distribution(
            counts, VALIDATION_QUOTA if key.endswith("/validation") else EQUAL_QUOTA
        )
        for key, counts in sorted(pool_counts.items())
    }

    def epoch_draw_count(pool_size: int, quota: int) -> int:
        if QUOTA_TAIL_REPLICATE and pool_size:
            return quota
        return min(pool_size, quota)

    epoch_draw_stats = {}
    for key, counts in sorted(pool_counts.items()):
        quota = VALIDATION_QUOTA if key.endswith("/validation") else EQUAL_QUOTA
        epoch_draw_stats[key] = quota_distribution(
            [epoch_draw_count(count, quota) for count in counts], quota
        )
    def quarantine(reason: str) -> None:
        rejected = ROOT / "training" / "rejected" / temporary.name.removeprefix(".")
        rejected.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, rejected)
        raise RuntimeError(f"{reason}; partial snapshot moved to {rejected}")

    if not kept["regression/validation"] or not kept["binary/validation"]:
        quarantine(f"both task validation splits must be populated: {dict(kept)}")

    # GUARDRAIL 2 (ops/generalization-plan.md): train and validation per-dataset
    # medians must be within 2x. Asserted on the EPOCH-DRAW medians, and it fails
    # the freeze rather than warning: the sampler defect it catches (train median
    # 6 vs validation median 485) produced an 8-nat "overfit gap" that was read as
    # a model problem for a full day.
    median_scopes = split_median_divergence(epoch_draw_stats)
    diverged = {
        scope: values
        for scope, values in median_scopes.items()
        if values["ratio"] is not None and values["ratio"] > SPLIT_MEDIAN_RATIO_MAX
    }
    if diverged:
        quarantine(
            "train/validation per-dataset epoch-draw medians diverge by more than "
            f"{SPLIT_MEDIAN_RATIO_MAX}x: {diverged}"
        )
    manifest = {
        "schema": "JopeTestOnlyTrainingCorpus.v1",
        "source_db": str(CORPUS_DB),
        "source_db_sha256": sha256_file(CORPUS_DB),
        "source_label_watermark": watermark,
        "corpus_mode": corpus_mode,
        "selected_label_count": sum(kept.values()),
        "candidate_label_count": candidate_label_count,
        "source_label_total": total,
        "watermark_covers_every_row": covers_every_row,
        "sampler": {
            "id": SAMPLER_TAG,
            "equal_quota": EQUAL_QUOTA,
            "validation_quota": VALIDATION_QUOTA,
            "candidate_row_cap": CANDIDATE_ROW_CAP,
            "recency_slots": recency,
            "tail_replicate": QUOTA_TAIL_REPLICATE,
            "pool_selection": "even_spread_over_dataset_history",
            "quota_applied": "post_encodability_filter",
            "epoch_resample": "train split only, per-epoch seed, in train()",
        },
        # Rows per dataset held in the corpus file (the per-epoch draw samples
        # EQUAL_QUOTA of these for train; validation pool == quota, so its pool
        # stats and its epoch-draw stats are identical by construction).
        "quota_stats": quota_stats,
        # Rows per dataset the model actually sees in ONE epoch.
        "epoch_draw_stats": epoch_draw_stats,
        "split_median_divergence": median_scopes,
        "strict_filter": {
            "regression": {"score_tier": "test_exact", "metric": "r2"},
            "binary": {"score_tier": "test_exact", "metric": "auc"},
            "test_adaptive": True,
            "benchmark_valid": False,
            "selection_bias": "test-exact-reused",
            "promotion_scope": "internal-selected-test-flywheel-only",
        },
        "counts": dict(kept),
        "skipped": dict(skipped),
        "train": str(snapshot / "corpus.train.jsonl"),
        "train_sha256": sha256_file(temporary / "corpus.train.jsonl"),
        "validation": str(snapshot / "corpus.val.jsonl"),
        "validation_sha256": sha256_file(temporary / "corpus.val.jsonl"),
        "catalog_sha256": pointer["catalog_sha256"],
        "catalog_binding": str(catalog_binding_path),
        "catalog_binding_sha256": binding_sha256,
        "source_family_disjoint": catalog["source_family_disjoint"],
        "causal_pair_receipt_stats": causal_pair_stats,
    }
    # GUARDRAIL 1 (ops/generalization-plan.md): alarm when the corpus dataset
    # count FALLS between freezes. Judged BEFORE the manifest is written, because
    # `test_exact_packed_corpus.verify` binds the packed corpus to the manifest's
    # sha256 — a manifest rewritten after packing invalidates the binding and the
    # trainer then refuses the corpus outright.
    per_split_datasets = {
        key: int(value["datasets"]) for key, value in quota_stats.items()
    }
    history_record = {
        "schema": "JopeCorpusFreezeHistory.v1",
        "at_ms": int(time.time() * 1000),
        "watermark": watermark,
        "source_label_total": total,
        "corpus_mode": corpus_mode,
        "sampler": SAMPLER_TAG,
        "manifest": str(snapshot / "corpus-manifest.json"),
        "datasets_total": sum(per_split_datasets.values()),
        "per_split_datasets": per_split_datasets,
        "per_split_rows": {
            key: int(value["rows"]) for key, value in quota_stats.items()
        },
        "epoch_draw_rows": sum(
            int(value["rows"]) for value in epoch_draw_stats.values()
        ),
        "epoch_draw_medians": {
            key: float(value["median"]) for key, value in epoch_draw_stats.items()
        },
        "split_median_divergence": median_scopes,
        "candidate_label_count": candidate_label_count,
        "selected_label_count": sum(kept.values()),
    }
    guard = publish_freeze_guard(history_record)
    manifest["freeze_guard"] = {
        "regressed": guard["regressed"],
        "previous_best_datasets": guard["previous_best_datasets"],
        "datasets_total": guard["datasets_total"],
    }
    atomic_json(temporary / "corpus-manifest.json", manifest)
    os.replace(temporary, snapshot)
    # Only now is the freeze durable, so only now does it enter the history the
    # next freeze compares itself against.
    append_freeze_history(history_record)
    manifest_path = snapshot / "corpus-manifest.json"
    publish_corpus_freeze_state(
        watermark=watermark,
        stage="rows_frozen",
        manifest_path=manifest_path,
        catalog_binding_path=catalog_binding_path,
    )
    packed = ensure_packed(
        manifest_path,
        catalog_binding_path=catalog_binding_path,
    )
    packed_manifest = (
        ROOT / "training" / "packed" / packed["binding_sha256"] / "manifest.json"
    )
    publish_corpus_freeze_state(
        watermark=watermark,
        stage="packed",
        manifest_path=manifest_path,
        catalog_binding_path=catalog_binding_path,
        packed_manifest_path=packed_manifest,
    )
    publication = publish_corpus_pointer(manifest_path, packed_manifest, watermark)
    if publication["published"]:
        publish_corpus_freeze_state(
            watermark=watermark,
            stage="published",
            manifest_path=manifest_path,
            catalog_binding_path=catalog_binding_path,
            packed_manifest_path=packed_manifest,
        )
    return manifest


def train(
    corpus_manifest: Path,
    output: Path,
    *,
    device: str,
    warm: Path,
    recipe: str = "a",
    seed: int = 20260726,
    task: str,
    mode: str,
    packed_manifest: Path | None = None,
    packed_source_train: Path | None = None,
    packed_source_validation: Path | None = None,
    packed_catalog_data: Path | None = None,
    accepted_parent_sha256: str | None = None,
    optional_task_readiness: Path | None = None,
) -> dict:
    if task not in ENABLED_TASK_METRICS:
        raise ValueError("trainer task is outside the four-task contract")
    if mode not in {"shadow", "production"}:
        raise ValueError("trainer mode must be shadow or production")
    if task not in MANDATORY_TASKS and mode != "shadow":
        raise ValueError("optional-task training is shadow-only")
    readiness = require_optional_task_admission(task, optional_task_readiness)
    if accepted_parent_sha256 is not None and (
        len(accepted_parent_sha256) != 64
        or any(character not in "0123456789abcdef" for character in accepted_parent_sha256)
    ):
        raise ValueError("accepted parent must be a lowercase SHA-256")
    warm_hash = sha256_file(warm)
    if warm == WARM and warm_hash != WARM_SHA:
        raise RuntimeError("accepted warm-start checkpoint hash drift")
    manifest = json.loads(corpus_manifest.read_text())
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(prime_source_root()))
    import jope.policy.train_v3 as train_v3

    training_rows = None
    packed_artifacts = None
    causal_pair_count = 0
    descendant_receipt_count = 0
    hard_negative_count = 0
    if packed_manifest is not None:
        from test_exact_packed_corpus import load_training_rows

        training_rows, packed_artifacts = load_training_rows(
            packed_manifest,
            task=task,
            source_manifest_path=corpus_manifest,
            source_train=packed_source_train,
            source_validation=packed_source_validation,
            catalog_data=packed_catalog_data,
        )
        if recipe == "c":
            validation = [
                row
                for row in training_rows
                if (row.metadata or {}).get("split") == "validation"
            ]
            train_rows = [
                row
                for row in training_rows
                if (row.metadata or {}).get("split") == "train"
            ]
            newest = sorted(
                train_rows,
                key=lambda row: (
                    -int((row.metadata or {}).get("label_created_at_ms") or 0),
                    str((row.metadata or {}).get("receipt_sha256") or ""),
                ),
            )[:50_000]
            newest_ids = {id(row) for row in newest}
            historical = [row for row in train_rows if id(row) not in newest_ids]
            random.Random(seed).shuffle(historical)
            training_rows = [*newest, *historical[: len(newest)], *validation]
        if recipe == "d":
            causal_pair_count = len(
                {
                    (row.metadata or {}).get("causal_pair_id")
                    for row in training_rows
                    if (row.metadata or {}).get(
                        "causal_accepted_parent_checkpoint_sha256"
                    )
                    == accepted_parent_sha256
                    and (row.metadata or {}).get("causal_pair_id")
                }
            )
            if causal_pair_count < 32:
                raise ValueError(
                    "causal-pair recipe d requires at least 32 complete "
                    "accepted-parent pairs"
                )
        if recipe == "e":
            causal_pair_count = len(
                {
                    (row.metadata or {}).get("causal_pair_id")
                    for row in training_rows
                    if (row.metadata or {}).get(
                        "causal_accepted_parent_checkpoint_sha256"
                    )
                    == accepted_parent_sha256
                    and (row.metadata or {}).get("causal_pair_id")
                }
            )
            descendant_receipt_count = len(
                {
                    (row.metadata or {}).get(
                        "descendant_credit_receipt_sha256"
                    )
                    for row in training_rows
                    if (row.metadata or {}).get(
                        "descendant_credit_receipt_sha256"
                    )
                    and not (row.metadata or {}).get(
                        "descendant_censored", False
                    )
                }
            )
            hard_negative_count = sum(
                recipe_e_evidence_lane(
                    row.metadata or {}, accepted_parent_sha256 or ""
                )
                == "validity_instability_hard_negatives"
                for row in training_rows
            )
            if causal_pair_count < 32:
                raise ValueError(
                    "recipe E requires at least 32 complete current-parent pairs"
                )
            if descendant_receipt_count < 1:
                raise ValueError(
                    "recipe E requires completed descendant-credit receipts"
                )
            if hard_negative_count < 1:
                raise ValueError(
                    "recipe E requires validity or instability hard negatives"
                )
    if recipe not in {"a", "b", "c", "d", "e"}:
        raise ValueError("checkpoint recipe must be a, b, c, d, or e")
    if recipe == "d" and (
        mode != "shadow"
        or packed_manifest is None
        or accepted_parent_sha256 is None
    ):
        raise ValueError(
            "causal-pair recipe d is shadow-only and requires packed rows "
            "plus an accepted-parent pin"
        )
    if recipe == "e" and (
        mode != "shadow"
        or packed_manifest is None
        or accepted_parent_sha256 is None
    ):
        raise ValueError(
            "descendant-credit recipe E is shadow-only and requires packed rows "
            "plus an accepted-parent pin"
        )

    regression_fraction = 0.75

    # ---- per-epoch EQUAL-QUOTA draw -------------------------------------
    # The corpus file holds a POOL per dataset; this takes exactly
    # metadata["epoch_quota"] rows out of each pool, re-drawn every epoch from a
    # seed derived from (run seed, epoch). Equal per-epoch quota, different rows
    # each epoch: no epoch over-represents a deep dataset, and across 24 epochs
    # the deep pools are still fully exercised.
    #
    # The draw MUST be identical on every DDP rank — train_v3 shards the batch
    # list by rank and deadlocks on an uneven split — hence a private, fully
    # determined Random and a sorted dataset iteration order.
    epochs_drawn = {"count": 0}

    def equal_quota_epoch_rows(rows, epoch):
        pools = defaultdict(list)
        for row in rows:
            metadata = row.metadata or {}
            pools[
                (str(metadata.get("task_type")), str(metadata.get("dataset_hash")))
            ].append(row)
        rng = random.Random((int(seed) & 0xFFFF_FFFF) * 1_000_003 + epoch * 2_654_435_761)
        drawn_by_family = defaultdict(lambda: defaultdict(list))
        for key in sorted(pools):
            pool = pools[key]
            quota = int((pool[0].metadata or {}).get("epoch_quota") or EQUAL_QUOTA)
            selected = equal_quota_draw(pool, quota, rng)
            family = str((pool[0].metadata or {}).get("source_family") or "")
            if not family:
                raise ValueError("training row lacks source_family for balanced sampling")
            drawn_by_family[family][key].extend(selected)
        if not drawn_by_family:
            return []
        # Every family contributes the same total weight. Inside a family,
        # datasets are visited round-robin in a seeded order, so a large source
        # family cannot dominate and a deep dataset cannot dominate its family.
        family_quota = min(
            sum(len(dataset_rows) for dataset_rows in datasets.values())
            for datasets in drawn_by_family.values()
        )
        balanced = []
        for family in sorted(drawn_by_family):
            datasets = drawn_by_family[family]
            keys = sorted(datasets)
            rng.shuffle(keys)
            positions = {key: 0 for key in keys}
            family_rows = []
            while len(family_rows) < family_quota:
                progressed = False
                for key in keys:
                    position = positions[key]
                    if position < len(datasets[key]):
                        family_rows.append(datasets[key][position])
                        positions[key] += 1
                        progressed = True
                        if len(family_rows) == family_quota:
                            break
                if not progressed:
                    break
            if len(family_rows) != family_quota:
                raise ValueError("family-balanced sampler could not satisfy its quota")
            balanced.extend(family_rows)
        return balanced

    def grouped_batches(rows, batch_size, rng):
        from jope.train import _dataset_grouped_batches

        rows = list(rows)
        # train_v3 routes BOTH its epoch batching and its validation batching
        # through this hook. Resample the train split only: the validation corpus
        # already holds exactly VALIDATION_QUOTA rows per dataset and has to stay
        # byte-identical across epochs, or the checkpoint-selection metric moves
        # for sampling reasons instead of learning reasons.
        # ("development" is what jope.split_integrity renames "validation" to; it
        # does not write the rename back onto the row, but accept it anyway.)
        splits = {str((row.metadata or {}).get("split")) for row in rows}
        is_validation = bool(rows) and splits <= {"validation", "development"}
        if recipe == "e" and rows and not is_validation:
            epochs_drawn["count"] += 1
            if any(
                str((row.metadata or {}).get("task_type")) != task
                for row in rows
            ):
                raise ValueError("recipe E batch crossed task identity")
            pair_members = defaultdict(dict)
            lane_rows = defaultdict(list)
            for row in rows:
                metadata = row.metadata or {}
                lane = recipe_e_evidence_lane(
                    metadata, accepted_parent_sha256 or ""
                )
                if lane == "censored_descendant":
                    continue
                if lane == "exact_current_parent_pairs":
                    pair_id = metadata["causal_pair_id"]
                    pair_lane = metadata["causal_lane"]
                    if pair_lane in pair_members[pair_id]:
                        raise ValueError(
                            "recipe E has an ambiguous current-parent pair lane"
                        )
                    pair_members[pair_id][pair_lane] = row
                else:
                    lane_rows[lane].append(row)
            pairs = [
                (
                    members["native_dl_direct"],
                    members["matched_gp_comparator"],
                )
                for _, members in sorted(pair_members.items())
                if set(members)
                == {"native_dl_direct", "matched_gp_comparator"}
            ]
            required_lanes = {
                "uniform_broad_replay",
                "completed_descendant_trajectories",
                "validity_instability_hard_negatives",
            }
            missing = sorted(
                lane for lane in required_lanes if not lane_rows[lane]
            )
            if missing or not pairs:
                raise ValueError(
                    f"recipe E evidence lanes are incomplete: {missing}"
                )
            epoch = epochs_drawn["count"]
            for lane in required_lanes:
                lane_rows[lane] = equal_quota_epoch_rows(
                    lane_rows[lane], epoch
                )
                rng.shuffle(lane_rows[lane])
            rng.shuffle(pairs)
            nominal = recipe_e_batch_lane_counts(batch_size, 0)
            batch_count = max(
                1,
                (len(lane_rows["uniform_broad_replay"])
                 + nominal["uniform_broad_replay"] - 1)
                // nominal["uniform_broad_replay"],
                (len(pairs) * 2
                 + nominal["exact_current_parent_pairs"] - 1)
                // nominal["exact_current_parent_pairs"],
                (len(lane_rows["completed_descendant_trajectories"])
                 + nominal["completed_descendant_trajectories"] - 1)
                // nominal["completed_descendant_trajectories"],
                (len(lane_rows["validity_instability_hard_negatives"])
                 + nominal["validity_instability_hard_negatives"] - 1)
                // nominal["validity_instability_hard_negatives"],
            )

            def cycle(pool, start, size):
                return [
                    pool[(start + index) % len(pool)]
                    for index in range(size)
                ]

            output = []
            lane_offsets = defaultdict(int)
            pair_offset = 0
            for batch_index in range(batch_count):
                counts = recipe_e_batch_lane_counts(
                    batch_size, batch_index
                )
                batch = []
                broad_lane = "uniform_broad_replay"
                batch.extend(
                    cycle(
                        lane_rows[broad_lane],
                        lane_offsets[broad_lane],
                        counts[broad_lane],
                    )
                )
                lane_offsets[broad_lane] += counts[broad_lane]
                pairs_per_batch = (
                    counts["exact_current_parent_pairs"] // 2
                )
                selected_pairs = cycle(
                    pairs, pair_offset, pairs_per_batch
                )
                pair_offset += pairs_per_batch
                batch.extend(
                    row
                    for pair in selected_pairs
                    for row in pair
                )
                for lane in (
                    "completed_descendant_trajectories",
                    "validity_instability_hard_negatives",
                ):
                    batch.extend(
                        cycle(
                            lane_rows[lane],
                            lane_offsets[lane],
                            counts[lane],
                        )
                    )
                    lane_offsets[lane] += counts[lane]
                if len(batch) != batch_size:
                    raise ValueError("recipe E batch composition drift")
                output.append(batch)
            return output
        if recipe == "d" and rows and not is_validation:
            epochs_drawn["count"] += 1
            broad = equal_quota_epoch_rows(rows, epochs_drawn["count"])
            pair_members = defaultdict(dict)
            pair_flags = {}
            for row in rows:
                metadata = row.metadata or {}
                pair_id = metadata.get("causal_pair_id")
                lane = metadata.get("causal_lane")
                if (
                    pair_id
                    and lane in {"native_dl_direct", "matched_gp_comparator"}
                    and metadata.get(
                        "causal_accepted_parent_checkpoint_sha256"
                    )
                    == accepted_parent_sha256
                ):
                    if lane in pair_members[pair_id]:
                        raise ValueError("causal recipe has an ambiguous pair lane")
                    pair_members[pair_id][lane] = row
                    pair_flags[pair_id] = bool(
                        metadata.get("causal_pair_hard")
                        or metadata.get("causal_pair_novel")
                    )
            pairs = [
                (
                    members["native_dl_direct"],
                    members["matched_gp_comparator"],
                )
                for pair_id, members in sorted(pair_members.items())
                if set(members)
                == {"native_dl_direct", "matched_gp_comparator"}
            ]
            if not pairs:
                raise ValueError(
                    "causal recipe has no complete pair for its accepted parent"
                )
            weighted_pairs = []
            for pair_id, members in sorted(pair_members.items()):
                if set(members) != {
                    "native_dl_direct",
                    "matched_gp_comparator",
                }:
                    continue
                pair = (
                    members["native_dl_direct"],
                    members["matched_gp_comparator"],
                )
                weighted_pairs.append(pair)
                if pair_flags[pair_id]:
                    weighted_pairs.append(pair)
            rng.shuffle(weighted_pairs)
            broad_quota = (batch_size * 3) // 4
            paired_rows = batch_size - broad_quota
            if paired_rows < 2 or paired_rows % 2:
                raise ValueError("causal recipe batch cannot preserve exact pairs")
            broad_rows = [
                row
                for batch in _dataset_grouped_batches(
                    broad, broad_quota, rng
                )
                for row in batch
            ]
            batch_count = max(
                1, (len(broad_rows) + broad_quota - 1) // broad_quota
            )

            def cycle(pool, start, size):
                return [
                    pool[(start + index) % len(pool)] for index in range(size)
                ]

            output = []
            pairs_per_batch = paired_rows // 2
            for index in range(batch_count):
                batch = cycle(
                    broad_rows, index * broad_quota, broad_quota
                )
                selected_pairs = cycle(
                    weighted_pairs,
                    index * pairs_per_batch,
                    pairs_per_batch,
                )
                batch.extend(
                    row for direct_comparator in selected_pairs for row in direct_comparator
                )
                output.append(batch)
            return output
        if rows and not is_validation:
            epochs_drawn["count"] += 1
            rows = equal_quota_epoch_rows(rows, epochs_drawn["count"])

        regression = [
            row
            for row in rows
            if str((row.metadata or {}).get("task_type")) == "regression"
        ]
        binary = [
            row
            for row in rows
            if str((row.metadata or {}).get("task_type")) == "binary"
        ]
        if not regression or not binary:
            return _dataset_grouped_batches(rows, batch_size, rng)
        regression_quota = round(batch_size * regression_fraction)
        binary_quota = batch_size - regression_quota
        if min(regression_quota, binary_quota) < 1:
            raise ValueError("task mixture must retain both task types")

        def ordered(pool, quota):
            return [
                row
                for batch in _dataset_grouped_batches(pool, quota, rng)
                for row in batch
            ]

        regression_rows = ordered(regression, regression_quota)
        binary_rows = ordered(binary, binary_quota)
        count = max(
            (len(regression_rows) + regression_quota - 1) // regression_quota,
            (len(binary_rows) + binary_quota - 1) // binary_quota,
        )

        def cycle(pool, start, size):
            return [pool[(start + index) % len(pool)] for index in range(size)]

        return [
            cycle(regression_rows, index * regression_quota, regression_quota)
            + cycle(binary_rows, index * binary_quota, binary_quota)
            for index in range(count)
        ]

    original_grouped_batches = train_v3._grouped_batches
    original_precompute = train_v3.precompute_row_artifacts
    original_policy_batch_iterator = train_v3._iter_policy_batches
    train_v3._grouped_batches = grouped_batches
    if packed_artifacts is not None:
        train_v3._iter_policy_batches = four_worker_policy_batch_iterator(
            train_v3, original_policy_batch_iterator
        )
    if packed_artifacts is not None:
        train_v3.precompute_row_artifacts = (
            lambda rows, vocab, persistent_cache=None: {
                id(row): packed_artifacts[id(row)]
                for row in rows
                if id(row) in packed_artifacts
            }
        )

    checkpoint = output / "best.model.pt"
    resume = output / "resume.current.pt"
    resume_existing = resume if resume.is_file() else None
    gpu_monitor = GpuUtilizationMonitor(
        packed_artifacts is not None and str(device).startswith("cuda")
    )
    gpu_monitor.start()
    try:
        summary = train_v3.train_policy_v3_from_jsonl(
            manifest["train"],
            checkpoint,
            # 24 not 60: with the per-dataset caps each epoch now sees ~3.3x more
            # rows, and the selected checkpoint has been epoch 25 (the first
            # phase-B epoch) in every run to date — the remaining 35 epochs only
            # ever moved the selection metric 0.6512 -> 0.6493.
            epochs=24,
            batch_size=256,
            device=device,
            metrics_jsonl=output / "train_metrics.jsonl",
            progress_jsonl=output / "train_progress.json",
            seed=seed,
            # 5e-4 is a FROM-SCRATCH rate. Every run since generation 3 warm-starts
            # from an accepted checkpoint, so the first optimizer steps land on
            # weights that are already near a good basin and 5e-4 spends the early
            # epochs undoing them (a-g7: train loss 3.71 at epoch 1 from a warm
            # start that validates at 0.735 pairwise). 2e-4 for continuation.
            #
            # No warmup or cosine/step decay was added, deliberately:
            #   * The LambdaLR lambdas are hardcoded in train_v3.py:1784-1791 with
            #     no injection point, so any warmup means editing jope-prime —
            #     which is pinned by sha256 as `trainer_source` provenance in this
            #     and every other spec, and is shared with the live comparison run.
            #   * It would be a sixth simultaneous change in a run whose whole
            #     purpose is to measure the five already-validated ones.
            #   * The schedule is not in fact flat: the encoder group drops to
            #     0.1x at the phase A->B boundary (train_v3.py:1945), and
            #     anti_overfit adds plateau-triggered LR halving on top.
            #   * Warmup exists to blunt warm-start shock from a too-large rate;
            #     halving the rate addresses the same failure mode directly.
            lr=2e-4,
            # ONE-WAY DOOR: this moves the optimizer from 4 param groups to 5 (the
            # task/routing/stability heads split out of the reward branch to carry
            # weight decay, train_v3.py:1712-1748) and train_v3.py:1303-1310 refuses
            # to resume an anti-overfit checkpoint without it. It can therefore only
            # ever START a run, never join one -- which is what this is. Turning it
            # on also sets embedding jitter 0.05 / token dropout 0.10 on the model
            # (train_v3.py:1548-1550), EMA 0.999, weight decay 0.05 on the decayed
            # groups (0.01 otherwise), and plateau LR halving at patience 12.
            anti_overfit_experiment=True,
            # Select on CONDITIONALITY, not raw pairwise accuracy. The old metric
            # is fully satisfiable by the genome-only ablation control (which has
            # matched or beaten the conditional model every epoch of every run),
            # so it could never distinguish a policy that uses the dataset
            # embedding from one that ignores it.
            checkpoint_metric="conditional_score",
            extra_rows=None if training_rows is not None else [manifest["validation"]],
            init_from=None if resume_existing else warm,
            resume_from=resume_existing,
            prime_vocab=True,
            routing_classes=14,
            d_model=512,
            n_layers=6,
            n_heads=4,
            ffn_dim=2048,
            amp=True,
            autotune_batch_size=True,
            artifact_cache_path=(
                None if packed_artifacts is not None else output / "artifact_cache.pt"
            ),
            resume_checkpoint=resume,
            # 3, not 12. Macro validation only runs in phase B (train_v3.py:2199 gates on
            # phase == "B"), so with phase_a_epochs=10 of 24 the patience budget is spent
            # entirely inside a 14-epoch window — a patience of 12 can essentially never
            # fire, and no run has ever early-stopped.
            #
            # Measured on run c-r000024 (val_conditional_score, the selection metric):
            #   ep11 0.76710 (best) | 12 bad(1) | 13 0.76725 best, reset | 14 bad(1)
            #   15 bad(2) | 16 bad(3) -> patience 3 stops here
            # and the metric over the remaining 8 epochs went 0.76673 -> 0.76793, i.e. it
            # never recovered past the epoch-13 best by more than measurement noise (the
            # whole 24-epoch spread of val_conditional_score is 0.76519..0.76966 = 0.0045).
            # Meanwhile epoch_sec grows 190s -> 336s across a run, so those 8 epochs are
            # the most expensive ones: 2,533s of 4,547s wall clock, 56% of the generation,
            # bought for less than the noise floor.
            #
            # Bounded on both sides: 3 still tolerates two consecutive down-passes, which
            # is what the epoch 12->13 recovery above actually needed.
            early_stopping_patience=3,
            training_rows=training_rows,
            resume_extra={
                "source_label_watermark": manifest["source_label_watermark"],
                "source_corpus_manifest_sha256": sha256_file(corpus_manifest),
                "accepted_ancestor_sha256": warm_hash,
                "recipe": recipe,
                "seed": seed,
                "recipe_e_batch_mix": (
                    RECIPE_E_BATCH_MIX if recipe == "e" else None
                ),
                "recipe_e_loss_weights": (
                    RECIPE_E_LOSS_WEIGHTS if recipe == "e" else None
                ),
                "recipe_e_model_outputs": (
                    list(RECIPE_E_MODEL_OUTPUTS) if recipe == "e" else None
                ),
                "task_batch_mixture": {
                    "regression": 1.0 if task == "regression" else 0.0,
                    "binary": 1.0 if task == "binary" else 0.0,
                },
                "task": task,
                "trainer_mode": mode,
                "packed_manifest_sha256": (
                    sha256_file(packed_manifest) if packed_manifest else None
                ),
            },
        )
    finally:
        gpu_monitor.stop()
        train_v3._grouped_batches = original_grouped_batches
        train_v3.precompute_row_artifacts = original_precompute
        train_v3._iter_policy_batches = original_policy_batch_iterator
    if checkpoint.is_file():
        checkpoint_hash = sha256_file(checkpoint)
        immutable = output / f"{checkpoint_hash}.model.pt"
        shutil.copy2(checkpoint, immutable)
        summary["checkpoint_sha256"] = checkpoint_hash
        summary["checkpoint_path"] = str(immutable)
    metrics_path = output / "train_metrics.jsonl"
    metrics_rows = (
        [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        if metrics_path.is_file()
        else []
    )
    phase_a = [
        row for row in metrics_rows
        if row.get("phase") == "A"
        and row.get("val_pairwise_win_accuracy") is not None
    ]
    phase_b = [
        row for row in metrics_rows
        if row.get("phase") == "B"
        and row.get("val_pairwise_win_accuracy") is not None
    ]
    if not phase_b:
        raise RuntimeError("trainer produced no eligible phase-B validation checkpoint")
    selected_epoch = summary.get("selected_checkpoint_epoch")
    selected = next(
        (row for row in phase_b if row.get("epoch") == selected_epoch),
        max(phase_b, key=lambda row: float(row["val_pairwise_win_accuracy"])),
    )
    summary["phase_a_diagnostic_peak"] = (
        None
        if not phase_a
        else max(float(row["val_pairwise_win_accuracy"]) for row in phase_a)
    )
    summary["selected_phase_b_accuracy"] = float(
        selected["val_pairwise_win_accuracy"]
    )
    summary["selected_phase_b_epoch"] = int(selected["epoch"])
    summary["selected_phase_b_task_accuracy"] = {
        task: selected.get(f"val_{task}_pairwise_win_accuracy")
        for task in ("regression", "binary")
    }
    summary["accepted_baseline_sha256"] = warm_hash
    summary["corpus_manifest_sha256"] = sha256_file(corpus_manifest)
    summary["accepted_ancestor_sha256"] = warm_hash
    summary["recipe"] = recipe
    summary["training_recipe"] = (
        {
            "batch_mix": RECIPE_E_BATCH_MIX,
            "loss_weights": RECIPE_E_LOSS_WEIGHTS,
            "model_outputs": list(RECIPE_E_MODEL_OUTPUTS),
            "architecture": {
                "d_model": 512,
                "layers": 6,
                "architecture_expansion": False,
            },
            "equal_rows_per_dataset_per_epoch": EQUAL_QUOTA,
            "complete_parent_pairs": causal_pair_count,
            "completed_descendant_credit_receipts": descendant_receipt_count,
            "hard_negative_rows": hard_negative_count,
            "censored_descendants_have_zero_training_weight": True,
            "learned_risk_ranking_enabled": False,
            "accepted_parent_checkpoint_sha256": accepted_parent_sha256,
        }
        if recipe == "e"
        else {
            "broad_replay_fraction": 0.75,
            "exact_causal_pair_fraction": 0.25,
            "hard_novel_oversampling": True,
            "fixed_anchor_holdout": True,
            "complete_parent_pairs": causal_pair_count,
            "accepted_parent_checkpoint_sha256": accepted_parent_sha256,
        }
        if recipe == "d"
        else {
            "broad_replay_fraction": 1.0,
            "exact_causal_pair_fraction": 0.0,
            "hard_novel_oversampling": False,
            "fixed_anchor_holdout": True,
            "complete_parent_pairs": 0,
            "accepted_parent_checkpoint_sha256": accepted_parent_sha256,
        }
    )
    epoch_throughput = [
        float(row["train_rows"]) / float(row["epoch_sec"])
        for row in metrics_rows
        if isinstance(row.get("train_rows"), (int, float))
        and isinstance(row.get("epoch_sec"), (int, float))
        and float(row["epoch_sec"]) > 0
    ]
    benchmark_throughput = max(
        (
            float(row.get("samples_per_second") or 0.0)
            for row in summary.get("batch_benchmark") or []
            if row.get("finite") is not False
        ),
        default=0.0,
    )
    sustained_throughput = (
        statistics.median(epoch_throughput) if epoch_throughput else 0.0
    )
    summary["packed_loader"] = {
        "deterministic_workers": 4 if packed_artifacts is not None else 0,
        "pinned_memory_prefetch": packed_artifacts is not None,
        "loader_window_batches": 8 if packed_artifacts is not None else 0,
        "sustained_training_rows_per_second": sustained_throughput,
        "batch_microbenchmark_samples_per_second": benchmark_throughput,
        "throughput_fraction_of_microbenchmark": (
            sustained_throughput / benchmark_throughput
            if benchmark_throughput > 0
            else None
        ),
        "meets_seventy_percent_gate": (
            benchmark_throughput > 0
            and sustained_throughput >= 0.70 * benchmark_throughput
        ),
    }
    summary["gpu_utilization"] = gpu_monitor.summary()
    summary["seed"] = seed
    summary["task"] = task
    summary["trainer_mode"] = mode
    summary["task_sampling_weights"] = eight_slot_task_weights(readiness)
    summary["checkpoint_contract"] = (
        "ShadowCheckpoint.v1" if mode == "shadow" else "ProductionCheckpointCandidate.v1"
    )
    summary["promotion_eligible"] = False
    atomic_json(output / "train-result.json", summary)
    return summary


def render_spec(
    corpus_manifest: Path,
    *,
    recipe: str = "a",
    seed: int = 20260726,
    task: str,
    mode: str = "shadow",
    packed_manifest: Path | None = None,
    optional_task_readiness: Path | None = None,
) -> dict:
    manifest = json.loads(corpus_manifest.read_text())
    watermark = int(manifest["source_label_watermark"])
    if recipe not in {"a", "b", "c", "d", "e"}:
        raise ValueError("checkpoint recipe must be a, b, c, d, or e")
    if task not in ENABLED_TASK_METRICS:
        raise ValueError("trainer task is outside the four-task contract")
    if mode not in {"shadow", "production"}:
        raise ValueError("trainer mode must be shadow or production")
    if task not in MANDATORY_TASKS and mode != "shadow":
        raise ValueError("optional-task training is shadow-only")
    readiness = require_optional_task_admission(task, optional_task_readiness)
    if recipe == "d" and (mode != "shadow" or packed_manifest is None):
        raise ValueError("causal-pair recipe d is shadow-only and packed-only")
    if recipe == "e" and (mode != "shadow" or packed_manifest is None):
        raise ValueError(
            "descendant-credit recipe E is shadow-only and packed-only"
        )
    job_id = f"jope-selfplay-{mode}-{task}-{watermark:09d}-{recipe}"
    spec = ROOT / "specs" / f"{job_id}.toml"
    script = Path(__file__)
    common = script.with_name("test_exact_common.py")
    prime_source = prime_source_root()
    train_source = prime_source / "jope" / "policy" / "train_v3.py"
    output_name = f"watermark-{watermark:09d}"
    # Optional tasks have no accepted production pointer by design; they warm
    # from the still-frozen shared binary/regression parent in shadow only.
    warm, warm_hash = accepted_checkpoint(task if task in MANDATORY_TASKS else None)
    inputs = [
        ("script", script, f"scripts/{script.name}"),
        ("common", common, f"scripts/{common.name}"),
        (
            "packed_source",
            script.with_name("test_exact_packed_corpus.py"),
            "scripts/test_exact_packed_corpus.py",
        ),
        ("manifest", corpus_manifest, "corpus/corpus-manifest.json"),
        ("checkpoint", warm, f"checkpoint/{warm.name}"),
        ("trainer_source", train_source, "provenance/train_v3.py"),
    ]
    packed_arg = ""
    readiness_arg = ""
    if readiness is not None:
        readiness_path = optional_task_readiness or OPTIONAL_TASK_READINESS
        inputs.append(
            ("optional_task_readiness", readiness_path, "evidence/optional-task-readiness.v2.json")
        )
        readiness_arg = (
            ', "--optional-task-readiness", "{input:optional_task_readiness}"'
        )
    if packed_manifest is not None:
        packed = json.loads(packed_manifest.read_text())
        pack_root = packed_manifest.parent
        inputs.append(("packed_manifest", packed_manifest, "packed/manifest.json"))
        inputs.append(
            (
                "packed_catalog",
                Path(packed["catalog_data"]),
                "packed/catalog-data.f32le",
            )
        )
        for index, name in enumerate(sorted(packed["files"])):
            inputs.append(
                (f"packed_{index}", pack_root / name, f"packed/{name}")
            )
        packed_arg = (
            ', "--packed-manifest", "{input:packed_manifest}", '
            '"--packed-catalog-data", "{input:packed_catalog}"'
        )
    else:
        inputs.extend(
            [
                ("train", Path(manifest["train"]), "corpus/corpus.train.jsonl"),
                (
                    "validation",
                    Path(manifest["validation"]),
                    "corpus/corpus.val.jsonl",
                ),
            ]
        )
    lines = [
        "[job]",
        f'id = "{job_id}"',
        'project = "jope-selfplay-train"',
        'tier = "normal"',
        'command = "/usr/bin/python3"',
        (
            'args = ["{input:script}", "train", "--corpus-manifest", "{input:manifest}", '
            f'"--output", "{{output:training}}/{output_name}", "--device", "cuda", '
            f'"--warm", "{{input:checkpoint}}", "--recipe", "{recipe}", "--seed", "{seed}", '
            f'"--task", "{task}", "--mode", "{mode}", '
            f'"--accepted-parent-sha256", "{warm_hash}"'
            f"{packed_arg}{readiness_arg}]"
        ),
        "",
    ]
    lines.extend(render_resource_block().splitlines())
    lines.append("")
    lines.extend(render_placement_block().splitlines())
    lines.extend(
        [
            "",
            *jope_spec_policy.render_env_block(
                "trainer", f"{prime_source}:{script.parent}"
            ).splitlines(),
            "",
            "[preempt]",
            f"preemptible = {str(bool(TRAINER_POLICY['preemptible'])).lower()}",
            'checkpoint = "none"',
            'resume = "rerun"',
            "",
            *jope_spec_policy.render_limits_block("trainer").splitlines(),
            "",
        ]
    )
    for name, source, target in inputs:
        lines.extend(
            [
                "[[input]]",
                f'name = "{name}"',
                f"source = {json.dumps(str(source))}",
                f"target = {json.dumps(target)}",
                f'sha256 = "{sha256_file(source)}"',
                "",
            ]
        )
    lines.extend(
        [
            "[[output]]",
            'name = "training"',
            'source = "training"',
            "directory = true",
            "",
        ]
    )
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("\n".join(lines))
    return {
        "job_id": job_id,
        "spec": str(spec),
        "watermark": watermark,
        "recipe": recipe,
        "seed": seed,
        "accepted_ancestor_sha256": warm_hash,
        "task": task,
        "mode": mode,
    }


def _establish_cuda_context_early(device: str) -> None:
    """Create the CUDA context BEFORE the packed corpus load.

    scq reclaims a GPU from a job that shows "no CUDA context or utilization"
    for IDLE_RECLAIM_SECS (120 s) while other GPU work is queued. That policy is
    correct -- a GPU job not touching CUDA is holding a GPU it is not using.

    But this trainer loads ~1.94 M packed rows before it ever touches the device,
    and that load takes ~5 minutes (measured 2026-08-04: 38.6 % complete at the
    120 s mark). So the reclaim fired every time, requeued the job, and the next
    attempt started the same load from zero -- an unbreakable loop in which the
    trainer could never reach a single training step. Two attempts of
    shadow-regression-008145630-s000013 died this way at 38.6 % and 23.2 %.

    Touching CUDA here makes the job's GPU claim honest from the first second
    rather than five minutes in. The allocation is one element, is freed
    immediately, and consumes no RNG, so it cannot perturb training or the
    bit-exactness contracts. Failure is non-fatal: if CUDA is unavailable the
    real device setup downstream reports it properly.
    """
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.init()
        marker = torch.zeros(1, device=device)
        torch.cuda.synchronize(device=device)
        del marker
        print(
            f"cuda-context-established device={device} "
            f"name={torch.cuda.get_device_name(device)}",
            flush=True,
        )
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"cuda-context-early-init-skipped {error}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-corpus")
    build.add_argument("--min-rows", type=int, default=1)
    build.add_argument("--freeze-watermark", type=int)
    build.add_argument("--catalog-binding", type=Path)
    build.add_argument(
        "--corpus-mode",
        choices=("recent_plus_replay", "full_consolidation"),
        default="full_consolidation",
    )
    train_command = commands.add_parser("train")
    train_command.add_argument("--corpus-manifest", type=Path, required=True)
    train_command.add_argument("--output", type=Path, required=True)
    train_command.add_argument("--device", default="cuda")
    train_command.add_argument("--warm", type=Path, default=WARM)
    train_command.add_argument(
        "--recipe", choices=("a", "b", "c", "d", "e"), default="a"
    )
    train_command.add_argument("--seed", type=int, default=20260726)
    train_command.add_argument(
        "--task", choices=tuple(sorted(ENABLED_TASK_METRICS)), required=True
    )
    train_command.add_argument("--mode", choices=("shadow", "production"), required=True)
    train_command.add_argument("--packed-manifest", type=Path)
    train_command.add_argument("--packed-source-train", type=Path)
    train_command.add_argument("--packed-source-validation", type=Path)
    train_command.add_argument("--packed-catalog-data", type=Path)
    train_command.add_argument("--accepted-parent-sha256")
    train_command.add_argument("--optional-task-readiness", type=Path)
    render = commands.add_parser("render-spec")
    render.add_argument("--corpus-manifest", type=Path, required=True)
    render.add_argument(
        "--recipe", choices=("a", "b", "c", "d", "e"), default="a"
    )
    render.add_argument("--seed", type=int, default=20260726)
    render.add_argument("--task", choices=tuple(sorted(ENABLED_TASK_METRICS)), required=True)
    render.add_argument("--mode", choices=("shadow", "production"), default="shadow")
    render.add_argument("--packed-manifest", type=Path)
    render.add_argument("--optional-task-readiness", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        _establish_cuda_context_early(args.device)
    if args.command == "build-corpus":
        print(
            json.dumps(
                build_corpus(
                    args.min_rows,
                    freeze_watermark=args.freeze_watermark,
                    corpus_mode=args.corpus_mode,
                    catalog_binding_path=args.catalog_binding,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "train":
        print(
            json.dumps(
                train(
                    args.corpus_manifest,
                    args.output,
                    device=args.device,
                    warm=args.warm,
                    recipe=args.recipe,
                    seed=args.seed,
                    task=args.task,
                    mode=args.mode,
                    packed_manifest=args.packed_manifest,
                    packed_source_train=args.packed_source_train,
                    packed_source_validation=args.packed_source_validation,
                    packed_catalog_data=args.packed_catalog_data,
                    accepted_parent_sha256=args.accepted_parent_sha256,
                    optional_task_readiness=args.optional_task_readiness,
                ),
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                render_spec(
                    args.corpus_manifest,
                    recipe=args.recipe,
                    seed=args.seed,
                    task=args.task,
                    mode=args.mode,
                    packed_manifest=args.packed_manifest,
                    optional_task_readiness=args.optional_task_readiness,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
