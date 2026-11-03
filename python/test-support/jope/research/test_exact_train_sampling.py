from __future__ import annotations

from test_exact_train import *

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


def complete_causal_members(
    connection: sqlite3.Connection, *, root: Path = ROOT
) -> tuple[dict, dict]:
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
    waves_root = root / "waves"
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
