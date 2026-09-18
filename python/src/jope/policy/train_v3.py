"""Training entry for policy_v3 with the A->B curriculum and hard val-split policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from ..engines import EngineRouter
from ..grammars import all_grammars, grammar_for
from ..grammars.spec import GrammarSpec
from ..grammars.vocab import (
    grammars_from_vocab_v3,
    merge_learned_bins,
    observed_param_tokens,
)
from ..io import read_jsonl
from ..model import _pairwise_accuracy, _pearson, _top1_regret
from ..pipeline_text import format_pipeline_actions
from ..schema import TrainingRow
from .losses import (
    PHASE_A_PROFILE,
    PHASE_B_PROFILE,
    POLICY_V3_DIAGNOSTIC_COMPONENTS,
    POLICY_V3_LOSS_COMPONENTS,
    clamp_log_vars,
    kg_combine,
)
from .model_v3 import (
    DATASET_BLOCK_RMS,
    MODEL_VERSION,
    TARGET_BLOCK_RMS,
    JopePolicyModelV3,
    precompute_row_artifacts,
    rows_to_policy_v3_batch,
)
from .tokens import (
    ActionVocab,
    build_action_vocab,
    build_prime_action_vocab,
    decode_tokens,
)

PRIME_ROUTING_CLASSES = 14
TRAINER_RESUME_V1 = "PrimePolicyTrainerResume.v1"
ROW_ARTIFACT_CACHE_V1 = "PolicyRowArtifactCache.v1"
_ROW_ARTIFACT_CACHE_MEMORY: dict[str, dict[str, Any]] = {}
CUDA_BATCH_CANDIDATES = (256, 512, 1024)
CUDA_PREFETCH_DEPTH = 2
CUDA_MINIMUM_VRAM_HEADROOM = 0.20
ANTI_OVERFIT_EXPERIMENT_SCHEMA = "PrimeAntiOverfitExperiment.v1"
ANTI_OVERFIT_EMA_DECAY = 0.999
ANTI_OVERFIT_WEIGHT_DECAY = 0.05
ANTI_OVERFIT_EMBEDDING_JITTER = 0.05
ANTI_OVERFIT_TOKEN_DROPOUT = 0.10
ANTI_OVERFIT_DATASET_ROW_CAP = 128
ANTI_OVERFIT_LEGACY_WEIGHT = 0.35
ANTI_OVERFIT_PLATEAU_PATIENCE = 12
TRAINING_SCHEDULE_SCHEMA = "PrimeTrainerSchedule.v1"
TRAINING_SCHEDULE_SELECTION_SCHEMA = "PrimeTrainerScheduleSelection.v1"
LIFELONG_MAX_PASSES = 3
FAMILY_HELD_EARLY_STOPPING_PATIENCE = 1
PREREGISTERED_TRAINING_SCHEDULES = (
    "heads-only",
    "trunk-decoder",
    "trunk",
)
_TRAINING_SCHEDULE_ROLE_ROOTS = {
    "trunk": (
        "encoder",
        # v3.4: the reward branch's own context encoder. It is trunk, not head — it is a
        # representation, and _training_schedule_parameter_roles raises unless every
        # parameter matches exactly one role, so this registration is load-bearing.
        "reward_encoder",
        "reward_token_emb",
        "reward_norm",
        "z_gamma",
        "z_proj",
    ),
    "decoder": (
        "ctx_proj",
        "token_emb",
        "pos_emb",
        "decoder",
        "out_norm",
    ),
    "heads": (
        "reward_heads",
        "binary_reward_heads",
        "reward_head_ablation",
        "runtime_head",
        "stability_head",
        "binary_stability_head",
        "routing_head",
        "binary_routing_head",
        "loss_log_vars",
    ),
}
_TRAINING_SCHEDULE_LEARNING_RATES = {
    "heads-only": {"trunk": None, "decoder": None, "heads": 1e-5},
    "trunk-decoder": {"trunk": 3e-6, "decoder": 3e-6, "heads": 1e-5},
    "trunk": {"trunk": 1e-5, "decoder": None, "heads": 3e-5},
}
_TRAINING_SCHEDULE_NO_DECAY_ROOTS = (
    # v3.4 reward context encoder: same no-decay policy as the rest of the reward branch
    # (decay under a weak pairwise signal shrinks the reward toward a constant).
    "reward_encoder",
    "reward_token_emb",
    "reward_norm",
    "z_gamma",
    "z_proj",
    "reward_heads",
    "binary_reward_heads",
    "stability_head",
    "binary_stability_head",
    "routing_head",
    "binary_routing_head",
    "loss_log_vars",
)


def preregistered_training_schedule_config(name: str) -> dict[str, Any]:
    """Return one fixed staged schedule identity; production passes no schedule."""

    if name not in PREREGISTERED_TRAINING_SCHEDULES:
        raise ValueError(f"unknown preregistered training schedule {name!r}")
    payload = {
        "schema": TRAINING_SCHEDULE_SCHEMA,
        "name": name,
        "roles": [
            {
                "role": role,
                "module_roots": list(_TRAINING_SCHEDULE_ROLE_ROOTS[role]),
                "learning_rate": _TRAINING_SCHEDULE_LEARNING_RATES[name][role],
                "trainable": _TRAINING_SCHEDULE_LEARNING_RATES[name][role] is not None,
            }
            for role in ("trunk", "decoder", "heads")
        ],
        "weight_decay_policy": "preserve-production-v1",
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _training_schedule_parameter_roles(
    model: JopePolicyModelV3,
) -> dict[str, tuple[str, ...]]:
    """Classify every parameter exactly once under the frozen schedule contract."""

    roles: dict[str, list[str]] = {role: [] for role in _TRAINING_SCHEDULE_ROLE_ROOTS}
    for parameter_name, _ in model.named_parameters():
        matches = [
            role
            for role, roots in _TRAINING_SCHEDULE_ROLE_ROOTS.items()
            if any(
                parameter_name == root or parameter_name.startswith(f"{root}.")
                for root in roots
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"training schedule must classify {parameter_name!r} exactly once; "
                f"matched {matches}"
            )
        roles[matches[0]].append(parameter_name)
    return {role: tuple(names) for role, names in roles.items()}


def _configure_preregistered_training_schedule(
    model: JopePolicyModelV3,
    name: str,
    *,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    """Freeze roles and build exact-LR optimizer groups for one staged schedule."""

    config = preregistered_training_schedule_config(name)
    roles = _training_schedule_parameter_roles(model)
    named_parameters = dict(model.named_parameters())
    for parameter in named_parameters.values():
        parameter.requires_grad_(False)

    optimizer_groups: list[dict[str, Any]] = []
    for role_config in config["roles"]:
        role = role_config["role"]
        learning_rate = role_config["learning_rate"]
        if learning_rate is None:
            continue
        decayed = []
        no_decay = []
        for parameter_name in roles[role]:
            parameter = named_parameters[parameter_name]
            parameter.requires_grad_(True)
            destination = (
                no_decay
                if any(
                    parameter_name == root or parameter_name.startswith(f"{root}.")
                    for root in _TRAINING_SCHEDULE_NO_DECAY_ROOTS
                )
                else decayed
            )
            destination.append(parameter)
        for decay_class, parameters, group_weight_decay in (
            ("decayed", decayed, weight_decay),
            ("no_decay", no_decay, 0.0),
        ):
            if parameters:
                optimizer_groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": group_weight_decay,
                        "schedule_role": role,
                        "decay_class": decay_class,
                    }
                )
    return optimizer_groups, roles


def _validate_training_schedule_resume(
    resume_payload: dict[str, Any],
    expected_schedule: dict[str, Any] | None,
) -> None:
    saved_schedule = resume_payload.get("training_schedule")
    if saved_schedule != expected_schedule:
        raise ValueError(
            "trainer resume schedule identity does not match the requested staged "
            "schedule; optimizer state reuse is forbidden across schedule drift"
        )


def _finite_schedule_metric(
    payload: dict[str, Any], key: str, *, minimum: float, maximum: float | None
) -> float:
    value = payload.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"schedule result {key!r} must be finite")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"schedule result {key!r} is outside its valid range")
    return result


def select_preregistered_training_schedule(
    candidates: Iterable[dict[str, Any]],
    *,
    accepted_baseline: dict[str, Any],
) -> dict[str, Any]:
    """Select from all three family-held trials without sequence/NLL regression."""

    if not isinstance(accepted_baseline, dict):
        raise ValueError("accepted schedule baseline must be an object")
    baseline = {
        "sequence_validity": _finite_schedule_metric(
            accepted_baseline, "sequence_validity", minimum=0.0, maximum=1.0
        ),
        "seq_nll": _finite_schedule_metric(
            accepted_baseline, "seq_nll", minimum=0.0, maximum=None
        ),
    }
    by_schedule: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("schedule candidates must be objects")
        schedule = candidate.get("schedule")
        if schedule not in PREREGISTERED_TRAINING_SCHEDULES:
            raise ValueError(f"unknown schedule result {schedule!r}")
        if schedule in by_schedule:
            raise ValueError(f"duplicate schedule result {schedule!r}")
        schedule_identity = preregistered_training_schedule_config(schedule)[
            "identity_sha256"
        ]
        if candidate.get("schedule_identity_sha256") != schedule_identity:
            raise ValueError(
                f"schedule result {schedule!r} is not bound to its immutable identity"
            )
        normalized = {
            "schedule": schedule,
            "schedule_identity_sha256": schedule_identity,
            "family_held_pairwise_accuracy": _finite_schedule_metric(
                candidate,
                "family_held_pairwise_accuracy",
                minimum=0.0,
                maximum=1.0,
            ),
            "family_held_reward_correlation": _finite_schedule_metric(
                candidate,
                "family_held_reward_correlation",
                minimum=-1.0,
                maximum=1.0,
            ),
            "sequence_validity": _finite_schedule_metric(
                candidate, "sequence_validity", minimum=0.0, maximum=1.0
            ),
            "seq_nll": _finite_schedule_metric(
                candidate, "seq_nll", minimum=0.0, maximum=None
            ),
        }
        rejection_reasons = []
        if normalized["sequence_validity"] < baseline["sequence_validity"]:
            rejection_reasons.append("sequence_validity_regression")
        if normalized["seq_nll"] > baseline["seq_nll"]:
            rejection_reasons.append("seq_nll_regression")
        normalized["eligible"] = not rejection_reasons
        normalized["rejection_reasons"] = rejection_reasons
        by_schedule[schedule] = normalized

    missing = set(PREREGISTERED_TRAINING_SCHEDULES) - set(by_schedule)
    if missing:
        raise ValueError(
            "schedule selection requires all preregistered results; missing "
            f"{sorted(missing)}"
        )
    normalized_candidates = [
        by_schedule[schedule] for schedule in PREREGISTERED_TRAINING_SCHEDULES
    ]
    ranked = sorted(
        (candidate for candidate in normalized_candidates if candidate["eligible"]),
        key=lambda candidate: (
            -candidate["family_held_pairwise_accuracy"],
            -candidate["family_held_reward_correlation"],
            candidate["schedule"],
        ),
    )
    return {
        "schema": TRAINING_SCHEDULE_SELECTION_SCHEMA,
        "accepted_baseline": baseline,
        "candidates": normalized_candidates,
        "eligible_ranking": [candidate["schedule"] for candidate in ranked],
        "selected_schedule": ranked[0]["schedule"] if ranked else None,
        "selected_schedule_identity_sha256": (
            ranked[0]["schedule_identity_sha256"] if ranked else None
        ),
    }


def _advance_family_held_early_stopping(
    metric: float | None,
    *,
    best: float | None,
    bad_passes: int,
    patience: int = FAMILY_HELD_EARLY_STOPPING_PATIENCE,
) -> tuple[float | None, int, bool]:
    """Advance the family-held pairwise early-stop state for one pass."""
    if best is not None and not math.isfinite(best):
        raise ValueError("family-held early-stop best must be finite")
    if bad_passes < 0:
        raise ValueError("family-held early-stop bad-pass count must be nonnegative")
    if patience < 1:
        raise ValueError("family-held early-stop patience must be positive")
    if metric is not None and math.isfinite(metric) and (best is None or metric > best):
        return float(metric), 0, False
    next_bad_passes = bad_passes + 1
    return best, next_bad_passes, next_bad_passes >= patience


def _checkpoint_payload_pairwise_accuracy(
    payload: dict[str, Any] | None,
) -> float | None:
    if not isinstance(payload, dict):
        return None
    selected = payload.get("best_payload")
    if isinstance(selected, dict):
        metrics = selected.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get("val_pairwise_win_accuracy")
            if type(value) in (int, float) and math.isfinite(float(value)):
                return float(value)
        value = selected.get("metric_value")
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
    metrics = payload.get("checkpoint_state_metrics")
    if isinstance(metrics, dict):
        value = metrics.get("val_pairwise_win_accuracy")
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
    if payload.get("checkpoint_state_metric_name") == "val_pairwise_win_accuracy":
        value = payload.get("checkpoint_state_metric_value")
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
    if payload.get("checkpoint_metric") == "val_pairwise_win_accuracy":
        value = payload.get("best_checkpoint_metric_value")
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
    return None


def anti_overfit_experiment_config() -> dict[str, Any]:
    """Return the fixed, preregistered experiment settings."""

    return {
        "schema": ANTI_OVERFIT_EXPERIMENT_SCHEMA,
        "ema_decay": ANTI_OVERFIT_EMA_DECAY,
        "weight_decay": ANTI_OVERFIT_WEIGHT_DECAY,
        "embedding_jitter": ANTI_OVERFIT_EMBEDDING_JITTER,
        "token_dropout": ANTI_OVERFIT_TOKEN_DROPOUT,
        "dataset_row_cap": ANTI_OVERFIT_DATASET_ROW_CAP,
        "legacy_weight": ANTI_OVERFIT_LEGACY_WEIGHT,
        "plateau_lr_halving_patience": ANTI_OVERFIT_PLATEAU_PATIENCE,
    }


def _apply_legacy_replay_weight(
    rows: Iterable[TrainingRow], *, watermark_cutoff: int
) -> int:
    """Downweight train rows known to predate the current fresh-label cycle."""

    if watermark_cutoff <= 0:
        return 0
    weighted = 0
    for row in rows:
        metadata = row.metadata if row.metadata is not None else {}
        if metadata.get("split", "train") != "train":
            continue
        try:
            watermark = int(metadata.get("label_watermark") or 0)
        except (TypeError, ValueError):
            continue
        if not 0 < watermark <= watermark_cutoff:
            continue
        current = metadata.get("seq_weight", 1.0)
        if not isinstance(current, (int, float)) or not math.isfinite(float(current)):
            current = 1.0
        metadata["seq_weight"] = float(current) * ANTI_OVERFIT_LEGACY_WEIGHT
        metadata["legacy_replay_weight"] = ANTI_OVERFIT_LEGACY_WEIGHT
        row.metadata = metadata
        weighted += 1
    return weighted


def _update_ema_state(
    ema_state: dict[str, torch.Tensor], model_state: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for key, value in model_state.items():
            shadow = ema_state[key]
            if value.dtype.is_floating_point:
                shadow.mul_(ANTI_OVERFIT_EMA_DECAY).add_(
                    value, alpha=1.0 - ANTI_OVERFIT_EMA_DECAY
                )
            else:
                shadow.copy_(value)


def _advance_plateau_lr(
    metric: float | None,
    *,
    best: float | None,
    bad_epochs: int,
    lr_scale: float,
) -> tuple[float | None, int, float, bool]:
    """Advance the fixed 12-epoch plateau counter and halve LR when due."""

    if metric is None or not math.isfinite(metric):
        return best, bad_epochs, lr_scale, False
    if best is None or metric > best:
        return float(metric), 0, lr_scale, False
    bad_epochs += 1
    if bad_epochs < ANTI_OVERFIT_PLATEAU_PATIENCE:
        return best, bad_epochs, lr_scale, False
    return best, 0, lr_scale * 0.5, True


def _score_label(row: TrainingRow) -> float | None:
    metadata = row.metadata or {}
    for value in (metadata.get("score_robust"), metadata.get("oof_r2_raw"), row.score):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _recompute_reward_targets(rows: list[TrainingRow]) -> None:
    """Stamp ranking targets only among rows sharing one evidence authority."""

    grouped: dict[str, list[tuple[int, float]]] = {}
    for index, row in enumerate(rows):
        score = _score_label(row)
        if score is None:
            continue
        metadata = row.metadata or {}
        identity = (
            metadata.get("reward_pair_identity")
            or metadata.get("dataset_hash")
            or row.dataset_id
        )
        grouped.setdefault(str(identity), []).append((index, score))

    for entries in grouped.values():
        best = max(score for _, score in entries)
        order = sorted(entries, key=lambda item: item[1])
        denominator = max(1, len(order) - 1)
        for position, (index, score) in enumerate(order):
            metadata = rows[index].metadata if rows[index].metadata is not None else {}
            metadata["oof_delta_to_incumbent"] = score - best
            metadata["rank_pct"] = position / denominator
            rows[index].metadata = metadata


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(path, json.dumps(payload).encode("utf-8"))


# Four-task batch mix (D5): regression 4 : binary 2 : timeseries 1 : multiclass 1
# per 8 rows — expressed as batch_size divisors, in canonical task order.
TASK_BATCH_QUOTA_DIVISOR = {
    "regression": 2,
    "binary": 4,
    "timeseries": 8,
    "multiclass": 8,
}


def _grouped_batches(
    rows: list[TrainingRow], batch_size: int, rng: random.Random
) -> list[list[TrainingRow]]:
    from ..train import _dataset_grouped_batches

    by_task: dict[str, list[TrainingRow]] = {}
    for row in rows:
        # Missing task_type means a legacy (pre-fused) corpus row: regression.
        # An EXPLICIT unknown task must raise — the reward_stats precedent: the
        # old reg/bin filter pair silently dropped such rows from every batch.
        task = str((row.metadata or {}).get("task_type") or "regression")
        if task not in TASK_BATCH_QUOTA_DIVISOR:
            raise ValueError(
                f"unknown task_type {task!r} in training rows; "
                f"known tasks: {sorted(TASK_BATCH_QUOTA_DIVISOR)}"
            )
        by_task.setdefault(task, []).append(row)
    present = [task for task in TASK_BATCH_QUOTA_DIVISOR if task in by_task]
    if len(present) <= 1:
        return _dataset_grouped_batches(rows, batch_size, rng)

    def coverage_order(task_rows: list[TrainingRow], quota: int) -> list[TrainingRow]:
        grouped = _dataset_grouped_batches(task_rows, quota, rng)
        return [row for group in grouped for row in group]

    def take_cycle(
        pool: list[TrainingRow], start: int, count: int
    ) -> list[TrainingRow]:
        return [pool[(start + offset) % len(pool)] for offset in range(count)]

    if set(present) == {"regression", "binary"}:
        # EXPLICIT SPECIAL CASE (D5): a {regression, binary}-only corpus keeps the
        # legacy 75/25 composition BIT-IDENTICALLY — same quotas, same RNG stream
        # consumption order (regression grouped first, then binary), same cycling.
        # tests/test_grouped_batches.py pins this against a frozen copy of the
        # pre-four-task implementation.
        regression = by_task["regression"]
        binary = by_task["binary"]
        if batch_size < 4 or batch_size % 4:
            raise ValueError(
                "fused training requires a positive batch_size divisible by four for a 75/25 task mix"
            )
        binary_quota = max(1, batch_size // 4)
        regression_quota = batch_size - binary_quota
        regression_order = coverage_order(regression, regression_quota)
        binary_order = coverage_order(binary, binary_quota)
        batch_count = max(
            math.ceil(len(regression_order) / regression_quota),
            math.ceil(len(binary_order) / binary_quota),
        )
        return [
            take_cycle(regression_order, index * regression_quota, regression_quota)
            + take_cycle(binary_order, index * binary_quota, binary_quota)
            for index in range(batch_count)
        ]

    # Four-task path (at least one of timeseries/multiclass present).
    if batch_size < 8 or batch_size % 8:
        raise ValueError(
            "fused four-task training requires a positive batch_size divisible by "
            "eight for the 4:2:1:1 regression:binary:timeseries:multiclass mix"
        )
    quotas = {
        task: max(1, batch_size // TASK_BATCH_QUOTA_DIVISOR[task]) for task in present
    }
    # Slots budgeted for absent tasks go to the first present task in canonical
    # order (regression when it exists), keeping every batch exactly batch_size.
    quotas[present[0]] += batch_size - sum(quotas.values())
    orders = {task: coverage_order(by_task[task], quotas[task]) for task in present}
    batch_count = max(
        math.ceil(len(orders[task]) / quotas[task]) for task in present
    )
    return [
        [
            row
            for task in present
            for row in take_cycle(orders[task], index * quotas[task], quotas[task])
        ]
        for index in range(batch_count)
    ]


def _map_policy_batch_tensors(batch, transform: Callable[[torch.Tensor], torch.Tensor]):
    values = {}
    for item in fields(batch):
        value = getattr(batch, item.name)
        values[item.name] = transform(value) if torch.is_tensor(value) else value
    return type(batch)(**values)


def _pin_policy_batch(batch):
    """Pin every CPU tensor so CUDA copies can run asynchronously."""

    def pin(value: torch.Tensor) -> torch.Tensor:
        if value.device.type != "cpu" or value.is_pinned():
            return value
        return value.pin_memory()

    return _map_policy_batch_tensors(batch, pin)


def _move_policy_batch(batch, device: str | torch.device, *, non_blocking: bool):
    return _map_policy_batch_tensors(
        batch,
        lambda value: value.to(device=device, non_blocking=non_blocking),
    )


def _record_policy_batch_stream(batch, stream: torch.cuda.Stream) -> None:
    for item in fields(batch):
        value = getattr(batch, item.name)
        if torch.is_tensor(value) and value.device.type == "cuda":
            value.record_stream(stream)


def _iter_policy_batches(
    batch_rows: list[list[TrainingRow]],
    vocab: ActionVocab,
    *,
    device: str | torch.device,
    artifact_cache: dict[int, tuple[list[int], list[list[int]]]],
) -> Iterator:
    """Collate on CPU and keep two pinned batches queued on a CUDA copy stream."""

    if not str(device).startswith("cuda"):
        for rows in batch_rows:
            yield rows_to_policy_v3_batch(
                rows, vocab, device=device, artifact_cache=artifact_cache
            )
        return

    copy_stream = torch.cuda.Stream(device=device)
    pending: deque[tuple[Any, torch.cuda.Event, Any]] = deque()
    row_iter = iter(batch_rows)

    def enqueue(rows: list[TrainingRow]) -> None:
        host_batch = _pin_policy_batch(
            rows_to_policy_v3_batch(
                rows, vocab, device="cpu", artifact_cache=artifact_cache
            )
        )
        with torch.cuda.stream(copy_stream):
            device_batch = _move_policy_batch(host_batch, device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(copy_stream)
        pending.append((device_batch, ready, host_batch))

    for _ in range(CUDA_PREFETCH_DEPTH):
        try:
            enqueue(next(row_iter))
        except StopIteration:
            break

    current_stream = torch.cuda.current_stream(device=device)
    while pending:
        device_batch, ready, host_batch = pending.popleft()
        current_stream.wait_event(ready)
        _record_policy_batch_stream(device_batch, current_stream)
        # Keep the pinned source alive until the consumer finishes this step.
        yield device_batch
        del host_batch
        try:
            enqueue(next(row_iter))
        except StopIteration:
            pass


def _rows_for_cuda_benchmark(
    rows: list[TrainingRow], batch_size: int
) -> list[TrainingRow]:
    regression = [
        row
        for row in rows
        if str((row.metadata or {}).get("task_type") or "regression") == "regression"
    ]
    binary = [
        row
        for row in rows
        if str((row.metadata or {}).get("task_type") or "regression") == "binary"
    ]

    def cycle(pool: list[TrainingRow], count: int) -> list[TrainingRow]:
        return [pool[index % len(pool)] for index in range(count)]

    if regression and binary:
        binary_count = batch_size // 4
        return cycle(regression, batch_size - binary_count) + cycle(
            binary, binary_count
        )
    return cycle(rows, batch_size)


def _missing_adam_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Conservatively reserve room for AdamW's two lazy moment tensors."""
    total = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "exp_avg" not in state:
                total += 2 * parameter.numel() * parameter.element_size()
    return total


def _select_cuda_batch_size(
    results: list[dict[str, Any]],
    *,
    minimum_headroom: float = CUDA_MINIMUM_VRAM_HEADROOM,
) -> int:
    eligible = [
        result
        for result in results
        if result.get("finite") is True
        and result.get("eligible_on_all_ranks", True) is True
        and math.isfinite(float(result.get("samples_per_second", float("nan"))))
        and float(result["samples_per_second"]) > 0.0
        and float(result.get("vram_headroom_fraction", -1.0)) >= minimum_headroom
    ]
    if not eligible:
        raise RuntimeError("no finite CUDA batch candidate retained 20% VRAM headroom")
    fastest = max(
        eligible,
        key=lambda result: (
            float(result["samples_per_second"]),
            int(result["batch_size"]),
        ),
    )
    return int(fastest["batch_size"])


def _synchronize_cuda_batch_benchmarks(results: list[dict[str, Any]], dist_ctx) -> None:
    if not dist_ctx.enabled:
        return
    import torch.distributed as dist

    for result in results:
        locally_eligible = (
            result.get("finite") is True
            and float(result.get("vram_headroom_fraction", -1.0))
            >= CUDA_MINIMUM_VRAM_HEADROOM
        )
        eligible = torch.tensor(
            1 if locally_eligible else 0,
            dtype=torch.int32,
            device=dist_ctx.device,
        )
        speed = torch.tensor(
            float(result.get("samples_per_second", 0.0)) if locally_eligible else 0.0,
            dtype=torch.float64,
            device=dist_ctx.device,
        )
        headroom = torch.tensor(
            max(0.0, float(result.get("vram_headroom_fraction", 0.0))),
            dtype=torch.float64,
            device=dist_ctx.device,
        )
        dist.all_reduce(eligible, op=dist.ReduceOp.MIN)
        dist.all_reduce(speed, op=dist.ReduceOp.MIN)
        dist.all_reduce(headroom, op=dist.ReduceOp.MIN)
        result["eligible_on_all_ranks"] = bool(eligible.item())
        result["samples_per_second"] = float(speed.item())
        result["vram_headroom_fraction"] = float(headroom.item())


def _benchmark_cuda_batch_sizes(
    model,
    optimizer: torch.optim.Optimizer,
    rows: list[TrainingRow],
    vocab: ActionVocab,
    *,
    device: str | torch.device,
    phase: str,
    profile,
    amp: bool,
    artifact_cache: dict[int, tuple[list[int], list[list[int]]]],
    dist_ctx,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Measure training throughput and peak memory without changing model state."""
    results: list[dict[str, Any]] = []
    rng_state = _capture_rng_state(rng)
    was_training = model.training
    model.train()
    optimizer_state_reserve = _missing_adam_state_bytes(optimizer)
    try:
        for candidate in CUDA_BATCH_CANDIDATES:
            result: dict[str, Any] = {
                "batch_size": candidate,
                "finite": False,
                "samples_per_second": 0.0,
                "vram_headroom_fraction": 0.0,
            }
            device_batch = None
            prefetched_batch = None
            host_batch = None
            warm_components = None
            warm_loss = None
            components = None
            loss = None
            try:
                host_batch = _pin_policy_batch(
                    rows_to_policy_v3_batch(
                        _rows_for_cuda_benchmark(rows, candidate),
                        vocab,
                        device="cpu",
                        artifact_cache=artifact_cache,
                    )
                )
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                free_before, total_memory = torch.cuda.mem_get_info(device)
                baseline_reserved = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
                device_batch = _move_policy_batch(host_batch, device, non_blocking=True)
                # Hold the second staged batch while timing the first so the
                # headroom gate accounts for the production prefetch footprint.
                prefetched_batch = _move_policy_batch(
                    host_batch, device, non_blocking=True
                )
                torch.cuda.synchronize(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    warm_components = model.loss_components(device_batch, phase=phase)
                    warm_loss = kg_combine(
                        warm_components, model.loss_log_vars, profile
                    )
                warm_loss.backward()
                torch.cuda.synchronize(device)

                optimizer.zero_grad(set_to_none=True)
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                with torch.autocast(device_type="cuda", enabled=amp):
                    components = model.loss_components(device_batch, phase=phase)
                    loss = kg_combine(components, model.loss_log_vars, profile)
                loss.backward()
                finished.record()
                finished.synchronize()
                elapsed_seconds = max(
                    1e-9, float(started.elapsed_time(finished)) / 1_000.0
                )
                peak_reserved = torch.cuda.max_memory_reserved(device)
                incremental_peak = max(0, peak_reserved - baseline_reserved)
                estimated_free_at_peak = max(
                    0,
                    free_before - incremental_peak - optimizer_state_reserve,
                )
                finite = bool(
                    torch.isfinite(warm_loss).item() and torch.isfinite(loss).item()
                )
                result.update(
                    {
                        "finite": finite,
                        "samples_per_second": (
                            candidate / elapsed_seconds if finite else 0.0
                        ),
                        "vram_headroom_fraction": (
                            estimated_free_at_peak / max(1, total_memory)
                        ),
                        "peak_reserved_bytes": int(peak_reserved),
                        "optimizer_state_reserve_bytes": int(optimizer_state_reserve),
                    }
                )
            except torch.cuda.OutOfMemoryError:
                result["error"] = "cuda_oom"
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                result["error"] = "cuda_oom"
            finally:
                optimizer.zero_grad(set_to_none=True)
                del loss
                del components
                del warm_loss
                del warm_components
                del device_batch
                del prefetched_batch
                del host_batch
                torch.cuda.empty_cache()
            results.append(result)
    finally:
        _restore_rng_state(rng_state, rng)
        model.train(was_training)
        optimizer.zero_grad(set_to_none=True)
    _synchronize_cuda_batch_benchmarks(results, dist_ctx)
    return results


_PREFIX_EXPANDABLE_STATE_KEYS = frozenset(
    {"token_emb.weight", "reward_token_emb.weight"}
)


def _vocab_for_training_rows(
    specs: list[GrammarSpec], rows: list[TrainingRow], *, force_prime: bool = False
) -> tuple[list[GrammarSpec], ActionVocab]:
    """Use Prime's append-only vocabulary only for corpora that contain Prime rows."""
    if not force_prime and not any(row.engine == "prime" for row in rows):
        return specs, build_action_vocab(specs)
    if not any(spec.engine == "prime" for spec in specs):
        specs = [*specs, grammar_for("prime")]
    return specs, build_prime_action_vocab()


# Parameter roots whose warm start may NEVER be silently skipped. Heads are allowed to
# re-initialise (v3.1 changed their input width and the comment above says so); a context
# encoder, the decoder trunk or an embedding table that quietly re-randomises is the
# "lobotomy" failure class that cost this project many generations when a 130-tensor
# checkpoint warm-started a 152-tensor model with no error anywhere.
_WARM_START_CRITICAL_ROOTS = (
    "encoder",
    "reward_encoder",
    "ctx_proj",
    "token_emb",
    "pos_emb",
    "decoder",
    "out_norm",
    "reward_token_emb",
    "reward_norm",
    "z_gamma",
    "z_proj",
)


def migrate_context_state(
    model: JopePolicyModelV3,
    previous_state: Mapping[str, torch.Tensor],
    *,
    source_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Rewrite a v3.3 context encoder into the v3.4 two-encoder / normalised-input shape.

    Three transformations, all of them EXACTLY output-preserving:

    1. Target-block pruning. v3.3's ``encoder.0.weight`` is (d_model, dataset_dim +
       target_dim). The trailing ``target_dim - target_keep_dim`` input dims are
       structurally zero in every row of the context catalog (verified: 376 of 512, the
       contiguous tail [136, 511], on all 19,951 catalog rows and at the sidecar source),
       so those weight columns multiply exactly 0 and dropping them cannot change any
       output. Columns that exist in the target shape but not in the source are ZERO-filled
       — the exact continuation of "this input dim has never contributed".

    2. Per-block rescaling. v3.4 feeds ``block / rms(block)`` instead of ``block``, so each
       surviving column is multiplied by the block's rms. ``W_new @ (x / rms) == W_old @ x``
       identically. What changes is not the function but the GRADIENT geometry: an Adam step
       is ~constant in weight space, so before this the target block's function-space
       learning rate was ~14x smaller than the dataset block's purely because its inputs
       were smaller.

    3. Un-sharing. ``reward_encoder.*`` is seeded as an exact clone of the migrated
       ``encoder.*``, so the reward branch's context is bit-identical to its v3.3 parent at
       step 0 and the two encoders diverge only through the gradients they actually receive.

    Returns ``(migrated_state, notes)``. ``previous_state`` is never mutated.
    """

    notes: list[str] = []
    state = dict(previous_state)
    key = "encoder.0.weight"
    source = state.get(key)
    if source is None or source.ndim != 2:
        return state, notes

    dataset_dim = int(model.dataset_dim)
    target_source_width = int(source.shape[1]) - dataset_dim
    target_target_width = int(model.target_keep_dim)
    if target_source_width <= 0:
        raise ValueError(
            f"checkpoint {key} has input width {source.shape[1]} which is not wider than "
            f"dataset_dim {dataset_dim}: the context layout is incompatible"
        )

    # The scale the SOURCE checkpoint's weights were written against. A v3.3 checkpoint was
    # trained on raw blocks (rms factor 1.0); a v3.4 checkpoint records its own constants in
    # the checkpoint payload (not in the state dict), so they arrive via source_payload.
    payload = source_payload or {}
    source_block_norm = bool(payload.get("input_block_norm", False))
    source_dataset_rms = (
        float(payload.get("dataset_block_rms", DATASET_BLOCK_RMS))
        if source_block_norm
        else 1.0
    )
    source_target_rms = (
        float(payload.get("target_block_rms", TARGET_BLOCK_RMS))
        if source_block_norm
        else 1.0
    )
    dataset_gain = (
        model.dataset_block_rms if model.input_block_norm else 1.0
    ) / source_dataset_rms
    target_gain = (
        model.target_block_rms if model.input_block_norm else 1.0
    ) / source_target_rms

    if (
        target_source_width == target_target_width
        and dataset_gain == 1.0
        and target_gain == 1.0
    ):
        migrated_encoder_first = source
    else:
        migrated_encoder_first = torch.zeros(
            (source.shape[0], dataset_dim + target_target_width),
            dtype=source.dtype,
        )
        migrated_encoder_first[:, :dataset_dim] = source[:, :dataset_dim] * dataset_gain
        shared = min(target_source_width, target_target_width)
        migrated_encoder_first[:, dataset_dim : dataset_dim + shared] = (
            source[:, dataset_dim : dataset_dim + shared] * target_gain
        )
        if target_target_width > target_source_width:
            notes.append(
                f"target block widened {target_source_width}->{target_target_width}; "
                f"{target_target_width - target_source_width} new columns zero-filled"
            )
        elif target_target_width < target_source_width:
            dropped = source[:, dataset_dim + target_target_width :]
            notes.append(
                f"target block pruned {target_source_width}->{target_target_width}; "
                f"dropped columns carried |w|max={float(dropped.abs().max()):.6g} "
                "against structurally-zero inputs"
            )
        if dataset_gain != 1.0 or target_gain != 1.0:
            notes.append(
                f"per-block rescale applied (dataset x{dataset_gain:.6g}, "
                f"target x{target_gain:.6g}) — output-preserving"
            )
        state[key] = migrated_encoder_first

    if model.reward_encoder is not None and not any(
        k.startswith("reward_encoder.") for k in state
    ):
        cloned = 0
        for name in list(state):
            if name == "encoder" or name.startswith("encoder."):
                state[f"reward_encoder.{name[len('encoder.'):]}"] = state[name].clone()
                cloned += 1
        notes.append(
            f"reward_encoder seeded as an exact clone of the migrated encoder ({cloned} tensors)"
        )
    return state, notes


def _warm_start_model(
    model: JopePolicyModelV3,
    previous_state: dict[str, torch.Tensor],
    *,
    strict_context: bool = True,
    source_payload: Mapping[str, Any] | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Load matching tensors and exact legacy prefixes of append-only embeddings."""
    previous_state, migration_notes = migrate_context_state(
        model, previous_state, source_payload=source_payload
    )
    for note in migration_notes:
        print(f"context migration: {note}")
    current = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    prefix_copied: list[str] = []
    for key, previous in previous_state.items():
        target = current.get(key)
        if target is None:
            continue
        if previous.shape == target.shape:
            loadable[key] = previous
            continue
        if (
            key in _PREFIX_EXPANDABLE_STATE_KEYS
            and previous.ndim == target.ndim
            and previous.shape[1:] == target.shape[1:]
            and 0 < previous.shape[0] <= target.shape[0]
        ):
            expanded = target.detach().clone()
            expanded[: previous.shape[0]].copy_(
                previous.to(device=expanded.device, dtype=expanded.dtype)
            )
            loadable[key] = expanded
            prefix_copied.append(key)
    if strict_context:
        missing = sorted(
            key
            for key in current
            if key.split(".")[0] in _WARM_START_CRITICAL_ROOTS and key not in loadable
        )
        if missing:
            raise ValueError(
                "warm start would silently re-randomise representation tensors "
                f"{missing}; refusing (this is the 'lobotomy' failure class). Source "
                f"checkpoint carried {len(previous_state)} tensors, target model needs "
                f"{len(current)}."
            )
    model.load_state_dict(loadable, strict=False)
    return len(loadable), tuple(sorted(prefix_copied))


def _target_keep_dim_for_corpus(
    rows: Iterable[TrainingRow], *, checkpoint: Mapping[str, Any] | None = None
) -> int | None:
    """Live width of the target-summary block, measured from the corpus itself.

    ``target_summary`` arrives 512-wide but the target sidecar
    (``train_target.vs``, fixed at 1x512 by selfplay_expansion.rs:2110) populates only its
    leading dims and zero-pads the rest: measured 136 live / 376 structurally zero on every
    one of the 19,951 rows of the live context catalog, on the 8,485-row catalog the frozen
    packed corpus binds, and at the sidecar source. Those 376 dims own 376 x 512 = 192,512
    parameters of ``encoder.0.weight`` that receive exactly zero gradient forever.

    Measuring rather than hard-coding matters because this system is meant to grow encodings
    forever: a wider sidecar widens this automatically, and ``migrate_context_state``
    zero-fills the new columns. It never SHRINKS below a warm-start checkpoint's recorded
    width, because a learned column must not be discarded on the strength of one corpus.
    """

    seen: set[str] = set()
    live = 0
    for row in rows:
        key = str(row.dataset_id)
        if key in seen:
            continue
        seen.add(key)
        target = row.target_embedding
        if not target:
            continue
        for index in range(len(target) - 1, live - 1, -1):
            if target[index]:
                live = index + 1
                break
    if live <= 0:
        return None
    recorded = None if checkpoint is None else checkpoint.get("target_keep_dim")
    if recorded is not None:
        live = max(live, int(recorded))
    return live


def validate_prime_authority_checkpoint(state: dict[str, Any]) -> None:
    """Reject smoke/legacy checkpoints that cannot seed Prime authority."""
    from .tokens import build_generation31_action_vocab

    frozen = build_generation31_action_vocab().tokens
    saved = list((state.get("action_vocab") or {}).get("tokens") or [])
    if len(saved) < len(frozen) or saved[: len(frozen)] != frozen:
        raise ValueError(
            "Prime authority requires the exact generation-31 vocabulary prefix; "
            f"checkpoint has {len(saved)} tokens and is smoke-only"
        )
    if int(state.get("routing_classes") or 0) != PRIME_ROUTING_CLASSES:
        raise ValueError(
            f"Prime authority requires {PRIME_ROUTING_CLASSES} routing classes; "
            f"checkpoint has {state.get('routing_classes')!r} and is smoke-only"
        )
    embedding_rows = int(state.get("embedding_rows") or 0)
    if embedding_rows < len(saved):
        raise ValueError(
            "checkpoint embedding table is smaller than its action vocabulary"
        )


def _embedding_rows_for_warm_start(
    vocab: ActionVocab, checkpoint: dict[str, Any] | None
) -> int:
    rows = vocab.size + 256
    if checkpoint is None:
        return rows
    previous_state = checkpoint.get("model", checkpoint)
    previous_embedding = previous_state.get("token_emb.weight")
    return max(
        rows,
        int(checkpoint.get("embedding_rows") or 0),
        int(previous_embedding.shape[0]) if previous_embedding is not None else 0,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace a mutable trainer snapshot after a complete torch save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _persistent_row_artifacts(
    path: Path, vocab: ActionVocab
) -> tuple[dict[str, tuple[list[int], list[list[int]]]], str]:
    vocab_hash = hashlib.sha256(
        json.dumps(vocab.tokens, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    key = str(path.resolve())
    payload = _ROW_ARTIFACT_CACHE_MEMORY.get(key)
    if payload is None and path.is_file():
        candidate = torch.load(path, map_location="cpu", weights_only=False)
        if (
            candidate.get("schema") == ROW_ARTIFACT_CACHE_V1
            and candidate.get("vocabulary_sha256") == vocab_hash
            and isinstance(candidate.get("entries"), dict)
        ):
            payload = candidate
    if payload is None:
        payload = {
            "schema": ROW_ARTIFACT_CACHE_V1,
            "vocabulary_sha256": vocab_hash,
            "entries": {},
        }
    _ROW_ARTIFACT_CACHE_MEMORY[key] = payload
    return payload["entries"], vocab_hash


def _capture_rng_state(rng: random.Random) -> dict[str, Any]:
    import numpy as np

    return {
        "python_global": random.getstate(),
        "python_local": rng.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: dict[str, Any], rng: random.Random) -> None:
    import numpy as np

    random.setstate(state["python_global"])
    rng.setstate(state["python_local"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _val_loss(
    model,
    val_rows,
    vocab,
    batch_size,
    device,
    profile,
    artifact_cache,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
):
    """Full loss (same combine + profile as training) on the held-out validation datasets.
    Reported per epoch alongside train_loss so overfitting (train↓ / val↑) is visible.
    Uses the frozen model in eval mode; no grad."""
    m = model.module if hasattr(model, "module") else model
    was_training = m.training
    m.eval()
    total = 0.0
    comp_totals: dict[str, float] = defaultdict(float)
    steps = 0
    import random as _r

    # Use the module-level batcher, not a direct `from ..train import` — callers
    # monkey-patch `_grouped_batches` to apply the train task mixture, and the
    # direct import bypassed it, leaving validation batches single-task and
    # averaging only 1.63 distinct datasets. Group size drives the listwise and
    # BCE floors, so that asymmetry alone was several nats of apparent "overfit".
    batches = _grouped_batches(list(val_rows), batch_size, _r.Random(0))
    total_batches = len(batches)
    try:
        with torch.no_grad():
            for batch_index, batch_rows in enumerate(batches, start=1):
                try:
                    batch = rows_to_policy_v3_batch(
                        batch_rows, vocab, device=device, artifact_cache=artifact_cache
                    )
                except ValueError:
                    if progress is not None:
                        update: dict[str, Any] = {
                            "validation_batch": batch_index,
                            "validation_batches": total_batches,
                            "validation_pct": round(
                                100.0 * batch_index / max(1, total_batches), 1
                            ),
                            "val_partial_batches": steps,
                        }
                        if steps:
                            update["val_partial"] = total / steps
                        progress(update)
                    continue
                components = m.loss_components(batch, phase="B")
                loss = kg_combine(components, m.loss_log_vars, profile)
                total += float(loss.detach().cpu())
                for name, value in components.items():
                    comp_totals[name] += float(value.detach().cpu())
                steps += 1
                if progress is not None:
                    progress(
                        {
                            "validation_batch": batch_index,
                            "validation_batches": total_batches,
                            "validation_pct": round(
                                100.0 * batch_index / max(1, total_batches), 1
                            ),
                            "val_partial": total / steps,
                            "val_partial_batches": steps,
                        }
                    )
    finally:
        if was_training:
            m.train()
    if steps == 0:
        return None, {}
    averaged = {k: v / steps for k, v in sorted(comp_totals.items())}
    averaged.update(_excess_over_floor(averaged))
    return total / steps, dict(sorted(averaged.items()))


def _excess_over_floor(components: Mapping[str, float]) -> dict[str, float]:
    """``*_excess`` readings: the component minus the floor it cannot go below.

    ``reward_listwise_excess`` = CE - H(target) = mean per-group KL(target || prediction).
    It is >= 0 and reaches 0 only on a perfect listwise fit, so unlike the raw CE it is
    comparable across corpora whose group-size mix differs.

    ``stability_bce_excess`` = weighted BCE - the same-weight best CONSTANT predictor at the
    batch's own base rate. POSITIVE means the instability head is worse than predicting the
    base rate. On run c-r000024 the raw val stability_bce ran 1.3079 -> 1.6479 against a
    constant-predictor floor of 1.1959, i.e. the excess was positive from epoch 1 — a fact
    the raw number never surfaced.
    """
    out: dict[str, float] = {}
    for value_key, floor_key, name in (
        ("reward_listwise", "reward_listwise_entropy", "reward_listwise_excess"),
        ("stability_bce", "stability_bce_baseline", "stability_bce_excess"),
    ):
        value = components.get(value_key)
        floor = components.get(floor_key)
        if value is not None and floor:
            out[name] = float(value) - float(floor)
    return out


def resolve_grammars(corpus: Path, rows: list[TrainingRow]) -> list[GrammarSpec]:
    """Prefer the frozen v3/v4 vocab snapshot next to the corpus; else merge learned bins.

    DRIFT GUARD: a frozen snapshot whose grammar STRUCTURE (op set, param kinds, tasks,
    generate flags, expr bins) disagrees with the source grammar silently trains the model
    on tokens the emitter can't compile (this happened: stale ``sigmoid_col0`` macros).
    Training hard-fails on drift with a migration hint rather than proceeding corrupted.
    """
    vocab_path = corpus.parent / "active_vocab.json"
    if vocab_path.exists():
        try:
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict) and data.get("engines"):
            from ..data.migrate_vocab_v4 import check_vocab_drift

            ok, source_hash, frozen_hash = check_vocab_drift(data)
            if not ok:
                raise ValueError(
                    f"frozen vocab at {vocab_path} has drifted from the source grammar "
                    f"(source={source_hash} frozen={frozen_hash}). Run: "
                    f"PYTHONPATH=src python3 -m jope.cli migrate-vocab-v4 --vocab {vocab_path}"
                )
            grammars = grammars_from_vocab_v3(data)
            return [grammars[e] for e in sorted(grammars)]
    specs = []
    for spec in all_grammars():
        observed = observed_param_tokens(rows, spec)
        specs.append(merge_learned_bins(spec, observed))
    return specs


def train_policy_v3_from_jsonl(
    corpus: str | Path,
    checkpoint: str | Path | None = None,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 5e-4,
    device: str = "cpu",
    metrics_jsonl: str | Path | None = None,
    seed: int = 0,
    phase_a_fraction: float = 0.4,
    checkpoint_metric: str = "val_pairwise_win_accuracy",
    allow_train_val_fallback: bool = False,
    greedy_eval_datasets: int = 16,
    router: EngineRouter | None = None,
    d_model: int = 256,
    n_layers: int = 2,
    ffn_dim: int = 1024,
    n_heads: int = 4,
    amp: bool = False,
    autotune_batch_size: bool = False,
    progress_jsonl: str | Path | None = None,
    extra_rows: list[str | Path] | None = None,
    init_from: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    prime_vocab: bool = False,
    routing_classes: int | None = None,
    resume_extra: dict[str, Any] | None = None,
    initial_epoch: int = 0,
    artifact_cache_path: str | Path | None = None,
    training_rows: Iterable[TrainingRow] | None = None,
    checkpoint_state: str = "best",
    reset_checkpoint_selection_on_resume: bool = False,
    anti_overfit_experiment: bool = False,
    legacy_watermark_cutoff: int = 0,
    training_schedule: str | None = None,
    family_held_early_stopping: bool = False,
    branch_from_accepted_resume: bool = False,
    early_stopping_patience: int | None = None,
) -> dict[str, Any]:
    """Train policy v3 from JSONL, or directly from already-built ``TrainingRow`` objects.

    ``training_rows`` is an optional production fast path; omitting it preserves the
    historical JSONL API. ``checkpoint_state`` defaults to the historical best-state
    export. A fresh arena cycle may reset checkpoint selection while retaining the
    accepted ancestor's model and optimizer state.
    """
    from .ddp import (
        aggregate_scalars,
        all_ranks_true,
        all_reduce_gradients,
        barrier,
        configure_cuda_fast_math,
        init_distributed,
        is_main,
        shard_for_rank,
        shutdown_distributed,
        unwrap,
        wrap_ddp,
    )

    dist_ctx = init_distributed(device)
    configure_cuda_fast_math()
    if init_from and resume_from:
        raise ValueError("init_from and resume_from are mutually exclusive")
    if branch_from_accepted_resume and resume_from is None:
        raise ValueError("an accepted-resume branch requires resume_from")
    if branch_from_accepted_resume and training_schedule is None:
        raise ValueError("an accepted-resume branch requires a training schedule")
    if routing_classes is not None and routing_classes < 1:
        raise ValueError("routing_classes must be positive when specified")
    if checkpoint_state not in ("best", "current"):
        raise ValueError("checkpoint_state must be 'best' or 'current'")
    if legacy_watermark_cutoff < 0:
        raise ValueError("legacy_watermark_cutoff must be nonnegative")
    if training_schedule is not None and anti_overfit_experiment:
        raise ValueError(
            "staged training schedules and the anti-overfit experiment must run "
            "as separate experiments"
        )
    if training_schedule is not None and not 1 <= epochs <= LIFELONG_MAX_PASSES:
        raise ValueError(
            "staged training schedules are limited to one through three passes"
        )
    if family_held_early_stopping and not 1 <= epochs <= LIFELONG_MAX_PASSES:
        raise ValueError("family-held lifelong cycles are limited to three passes")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    training_schedule_identity = (
        preregistered_training_schedule_config(training_schedule)
        if training_schedule is not None
        else None
    )

    metrics_path = Path(metrics_jsonl) if metrics_jsonl else None
    if metrics_path is not None and is_main(dist_ctx):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = (
        Path(progress_jsonl)
        if progress_jsonl
        else (
            metrics_path.parent / "train_progress.json"
            if metrics_path is not None
            else None
        )
    )
    if progress_path is not None and is_main(dist_ctx):
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    progress_started = time.time()

    def emit_progress(phase: str, **fields: Any) -> None:
        if progress_path is None or not is_main(dist_ctx):
            return
        now = time.time()
        _atomic_json(
            progress_path,
            {
                "phase": phase,
                "batch": 0,
                "total_batches": 0,
                "pct": 0.0,
                "active_label_watermark": int(
                    (resume_extra or {}).get("source_label_watermark", 0) or 0
                ),
                "elapsed_sec": now - progress_started,
                "updated_at": now,
                "progress_timestamp_ms": int(now * 1_000),
                **fields,
            },
        )

    resume_payload: dict[str, Any] | None = None
    if resume_from:
        emit_progress("model_restore", input_mode="resume")
        resume_payload = torch.load(
            str(resume_from), map_location="cpu", weights_only=False
        )
        if resume_payload.get("schema") != TRAINER_RESUME_V1:
            raise ValueError(
                f"unsupported trainer resume schema {resume_payload.get('schema')!r}"
            )
        if not branch_from_accepted_resume:
            _validate_training_schedule_resume(
                resume_payload, training_schedule_identity
            )
        saved_experiment = resume_payload.get("anti_overfit_experiment")
        if saved_experiment is not None and not branch_from_accepted_resume:
            expected = anti_overfit_experiment_config()
            if not anti_overfit_experiment:
                raise ValueError(
                    "anti-overfit resume requires anti_overfit_experiment=True"
                )
            if saved_experiment != expected:
                raise ValueError("anti-overfit resume configuration has drifted")

    corpus = Path(corpus)
    input_mode = "in_memory" if training_rows is not None else "jsonl"
    emit_progress("row_load", input_mode=input_mode)
    if training_rows is None:
        rows = [TrainingRow.from_json(r) for r in read_jsonl(corpus)]
    else:
        rows = list(training_rows)
        if any(not isinstance(row, TrainingRow) for row in rows):
            raise TypeError("training_rows must contain only TrainingRow objects")
    # Self-play generations: former-self rows live in sibling files, merged at load.
    for extra in extra_rows or []:
        extra = Path(extra)
        if extra.exists():
            rows.extend(TrainingRow.from_json(r) for r in read_jsonl(extra))
    if not rows:
        raise ValueError("corpus contains no rows")
    # Split before grammar discovery, reward-target recomputation, routing labels,
    # embedding-shape selection, losses, validation, and checkpoint selection.
    # Report-only evidence must not influence even an indirect training input.
    from ..split_integrity import partition_training_rows

    input_rows = len(rows)
    split_partition = partition_training_rows(rows)
    train_rows = split_partition["train"]
    val_rows = split_partition["development"]
    report_only_rows = split_partition["report_only"]
    model_row_ids = {id(row) for row in (*train_rows, *val_rows)}
    rows = [row for row in rows if id(row) in model_row_ids]
    # Embedding dims must be homogeneous (torch.tensor batches are rectangular). A
    # concurrent producer (GP factory/miner) restarted against an older
    # standardization can inject mismatched rows — drop them loudly, don't crash.
    from collections import Counter as _Counter

    dim_counts = _Counter(len(r.embedding) for r in rows)
    if len(dim_counts) > 1:
        majority_dim = dim_counts.most_common(1)[0][0]
        dropped = sum(c for d, c in dim_counts.items() if d != majority_dim)
        rows = [r for r in rows if len(r.embedding) == majority_dim]
        train_rows = [r for r in train_rows if len(r.embedding) == majority_dim]
        val_rows = [r for r in val_rows if len(r.embedding) == majority_dim]
        print(
            f"WARNING: dropped {dropped} rows with non-majority embedding dims {dict(dim_counts)} (kept {majority_dim})"
        )
    val_fallback = False
    if not val_rows:
        if not allow_train_val_fallback:
            raise ValueError(
                "policy_v3 requires a validation split; refusing train-subset fallback "
                "(pass allow_train_val_fallback=True only for CI smoke tests)"
            )
        val_fallback = True
        val_rows = train_rows[: min(len(train_rows), batch_size)]
    if not train_rows:
        raise ValueError("policy_v3 corpus contains no train-split rows")

    emit_progress(
        "row_prepare",
        input_mode=input_mode,
        rows=len(rows),
        input_rows=input_rows,
        train_rows=len(train_rows),
        validation_rows=len(val_rows),
        report_only_rows=len(report_only_rows),
    )
    specs = resolve_grammars(corpus, rows)
    specs, vocab = _vocab_for_training_rows(specs, rows, force_prime=prime_vocab)
    if resume_payload is not None:
        saved_tokens = (resume_payload.get("action_vocab") or {}).get("tokens")
        if saved_tokens != vocab.tokens:
            raise ValueError(
                "resume action vocabulary does not match the current Prime bundle"
            )
    resolved_device = (
        dist_ctx.device
        if dist_ctx.enabled
        else (
            device
            if (not device.startswith("cuda") or torch.cuda.is_available())
            else "cpu"
        )
    )

    # BUG-B fix (all 7 external reviews): the reward/listwise losses read ONLY
    # metadata.oof_delta_to_incumbent, which self-play/miner/GP rows never set —
    # so every fresh evaluated row trained imitation but NEVER the selector.
    # Recompute canonical within-dataset labels for ALL provenances here.
    _recompute_reward_targets(rows)
    finite_by_source: dict[str, list[int]] = {}
    for r in rows:
        md = r.metadata if r.metadata is not None else {}
        prov = str(md.get("provenance", "corpus")).split("_g")[0] or "corpus"
        v = md.get("oof_delta_to_incumbent")
        ok = isinstance(v, (int, float)) and math.isfinite(float(v))
        finite_by_source.setdefault(prov, []).append(1 if ok else 0)
        # provenance imitation weights (V7 anti self-confirmation): only true winners
        # deserve full imitation; generic generated rows barely; unscored ones not at all.
        generated = prov not in ("corpus", "quant_report", "seed")
        if generated:
            rp = md.get("rank_pct")
            is_winner = ok and (
                float(v) == 0.0 or (isinstance(rp, (int, float)) and rp >= 0.9)
            )
            md["seq_weight"] = 1.0 if is_winner else (0.05 if ok else 0.0)
            r.metadata = md
    legacy_weighted_rows = (
        _apply_legacy_replay_weight(
            train_rows, watermark_cutoff=legacy_watermark_cutoff
        )
        if anti_overfit_experiment
        else 0
    )
    if is_main(dist_ctx):
        rates = {
            k: round(sum(v) / max(1, len(v)), 3)
            for k, v in sorted(finite_by_source.items())
        }
        print(f"reward_finite_rate_by_source: {rates}")
        for k, v in rates.items():
            if k != "corpus" and v < 0.95 and len(finite_by_source[k]) > 100:
                print(
                    f"WARNING: source '{k}' reward coverage {v} < 0.95 — selector is starved of it"
                )

    # dataset_dim follows the corpus: v3.1 rows carry 512, v3.2 rows 1536.
    row_dataset_dim = len(rows[0].embedding) if rows else 512

    # v3.2 routing supervision: stamp each row with its dataset's winner-head-family
    # class so the encoder is FORCED to preserve routing signal (see routing_ce).
    def _head_of(r):
        a = (r.metadata or {}).get("actions") or {}
        h = a.get("head", {}).get("op") if isinstance(a.get("head"), dict) else None
        if h:
            return str(h)
        # rows from self-play/miner/GP lack `actions`: parse the head op from the
        # canonical genome (last parenthesized group) so routing labels track the
        # CURRENT champions, not just the original corpus winners.
        g = r.canonical_genome or r.genome or ""
        m = re.findall(r"\(([a-z0-9_]+)", g)
        return m[-1] if len(m) > 1 else None

    winner: dict[str, tuple[float, str]] = {}
    for r in rows:
        md = r.metadata or {}
        s = _score_label(r)
        head = _head_of(r)
        did = md.get("dataset_hash") or r.dataset_id
        if s is None or head is None or did is None:
            continue
        if did not in winner or s > winner[did][0]:
            winner[did] = (float(s), str(head))
    from collections import Counter

    fam_counts = Counter(h for _, h in winner.values())
    saved_families = (
        list(resume_payload.get("routing_families") or [])
        if resume_payload is not None
        else []
    )
    if saved_families:
        routing_families = saved_families
    else:
        routing_families = sorted(k for k, c in fam_counts.items() if c >= 20)
        if routing_classes is not None:
            routing_families = routing_families[: max(0, routing_classes - 1)]
    n_routing = routing_classes or (len(routing_families) + 1)
    if resume_payload is not None:
        saved_routing = int(resume_payload["routing_classes"])
        if routing_classes is not None and routing_classes != saved_routing:
            raise ValueError(
                f"resume routing class mismatch: {saved_routing} != {routing_classes}"
            )
        n_routing = saved_routing
    if len(routing_families) >= n_routing:
        raise ValueError("routing family mapping leaves no fixed 'other' class")
    fam_index = {f: i for i, f in enumerate(routing_families)}
    other_idx = n_routing - 1  # rare families collapse to the final stable class
    for r in rows:
        md = r.metadata if r.metadata is not None else {}
        did = md.get("dataset_hash") or r.dataset_id
        w = winner.get(did)
        md["routing_class"] = fam_index.get(w[1], other_idx) if w else -1
        r.metadata = md
    if is_main(dist_ctx):
        print(
            f"routing supervision: {len(winner)} datasets, {n_routing} classes {routing_families}"
        )

    initial_payload = resume_payload
    if initial_payload is None and init_from:
        initial_payload = torch.load(
            str(init_from), map_location="cpu", weights_only=False
        )
    if prime_vocab and initial_payload is not None:
        validate_prime_authority_checkpoint(initial_payload)
    if resume_payload is None or branch_from_accepted_resume:
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    embedding_rows = _embedding_rows_for_warm_start(vocab, initial_payload)
    corpus_has_binary = any(
        str((row.metadata or {}).get("task_type") or "regression") == "binary"
        for row in rows
    )
    fused_task_heads = (
        bool(resume_payload.get("fused_task_heads", False))
        if resume_payload is not None
        else corpus_has_binary
    )
    if corpus_has_binary and not fused_task_heads:
        raise ValueError("binary rows require a fused-task-head shadow checkpoint")
    emit_progress(
        "model_restore",
        input_mode=input_mode,
        target_epoch=(
            int(resume_payload.get("global_epoch", 0)) + 1
            if resume_payload is not None
            else int(initial_epoch) + 1
        ),
        device=str(resolved_device),
    )
    target_keep_dim = _target_keep_dim_for_corpus(
        rows,
        checkpoint=resume_payload if resume_payload is not None else initial_payload,
    )
    if is_main(dist_ctx):
        print(
            f"context target block: keeping {target_keep_dim} dims "
            f"(structurally-zero tail dropped)"
        )
    base_model = JopePolicyModelV3(
        vocab,
        d_model=d_model,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        n_heads=n_heads,
        dataset_dim=row_dataset_dim,
        routing_classes=n_routing,
        embedding_rows=embedding_rows,
        fused_task_heads=fused_task_heads,
        target_keep_dim=target_keep_dim,
    ).to(resolved_device)
    if anti_overfit_experiment:
        base_model.training_embedding_jitter = ANTI_OVERFIT_EMBEDDING_JITTER
        base_model.training_token_dropout = ANTI_OVERFIT_TOKEN_DROPOUT
    # Warm start (self-play): initialize from the best former checkpoint instead of a
    # fresh random init — fresh-init-per-generation made generation quality a lottery
    # (gen-3 regressed −97 Elo). Other shape changes stay filtered; append-only
    # token tables retain their exact legacy prefix.
    if resume_payload is not None:
        base_model.load_state_dict(resume_payload["model"], strict=True)
    elif initial_payload is not None:
        prev_state = initial_payload.get("model", initial_payload)
        loaded_count, prefix_copied = _warm_start_model(
            base_model, prev_state, source_payload=initial_payload
        )
        if is_main(dist_ctx):
            detail = (
                f"; prefix-copied {','.join(prefix_copied)}" if prefix_copied else ""
            )
            print(
                f"warm-started {loaded_count}/{len(base_model.state_dict())} tensors "
                f"from {init_from}{detail}"
            )
    model = wrap_ddp(base_model, dist_ctx)

    # Encode every train+val row ONCE (token ids + legal masks) and reuse across epochs —
    # removes the dominant per-epoch CPU cost that otherwise idles the GPU.
    emit_progress(
        "row_preprocess",
        input_mode=input_mode,
        target_epoch=(
            int(resume_payload.get("global_epoch", 0)) + 1
            if resume_payload is not None
            else int(initial_epoch) + 1
        ),
        rows=len(train_rows) + len(val_rows),
        device=str(resolved_device),
    )
    preprocess_started = time.perf_counter()
    persistent_artifacts = None
    persistent_before = 0
    persistent_vocab_hash = None
    if artifact_cache_path is not None:
        persistent_artifacts, persistent_vocab_hash = _persistent_row_artifacts(
            Path(artifact_cache_path), vocab
        )
        persistent_before = len(persistent_artifacts)
    artifact_cache = precompute_row_artifacts(
        train_rows + val_rows, vocab, persistent_cache=persistent_artifacts
    )
    persistent_after = (
        len(persistent_artifacts) if persistent_artifacts is not None else 0
    )
    preprocess_cache_misses = max(0, persistent_after - persistent_before)
    preprocess_cache_hits = (
        max(0, len(train_rows) + len(val_rows) - preprocess_cache_misses)
        if persistent_artifacts is not None
        else 0
    )
    if (
        artifact_cache_path is not None
        and persistent_artifacts is not None
        and len(persistent_artifacts) != persistent_before
        and is_main(dist_ctx)
    ):
        _atomic_torch_save(
            {
                "schema": ROW_ARTIFACT_CACHE_V1,
                "vocabulary_sha256": persistent_vocab_hash,
                "entries": persistent_artifacts,
            },
            Path(artifact_cache_path),
        )
    preprocess_sec = time.perf_counter() - preprocess_started
    dev_is_cuda = str(resolved_device).startswith("cuda")
    accepted_baseline_started = time.perf_counter()
    accepted_ancestor_pairwise_accuracy: float | None = None
    accepted_ancestor_metrics: dict[str, float] | None = None
    if initial_payload is not None and family_held_early_stopping:
        current_validation_ids = sorted(
            str(value)
            for value in (resume_extra or {}).get("validation_dataset_ids", [])
        )
        trainer_extra = initial_payload.get("trainer_extra")
        saved_validation_ids = (
            sorted(str(value) for value in trainer_extra.get("validation_dataset_ids", []))
            if isinstance(trainer_extra, dict)
            else []
        )
        if (
            not branch_from_accepted_resume
            and training_schedule is None
            and current_validation_ids
            and saved_validation_ids == current_validation_ids
        ):
            accepted_ancestor_pairwise_accuracy = (
                _checkpoint_payload_pairwise_accuracy(initial_payload)
            )
        if (
            training_schedule is not None
            or accepted_ancestor_pairwise_accuracy is None
        ) and is_main(dist_ctx):
            with torch.no_grad():
                accepted_metrics = policy_v3_validation_metrics(
                    unwrap(model),
                    val_rows,
                    vocab,
                    batch_size=batch_size,
                    device=resolved_device,
                    greedy_eval_datasets=0,
                    router=router,
                )
            accepted_ancestor_pairwise_accuracy = accepted_metrics.get(
                "pairwise_win_accuracy"
            )
            accepted_ancestor_metrics = {
                key: float(accepted_metrics[key])
                for key in (
                    "pairwise_win_accuracy",
                    "reward_cv_correlation",
                    "sequence_validity",
                    "seq_nll",
                )
                if type(accepted_metrics.get(key)) in (int, float)
                and math.isfinite(float(accepted_metrics[key]))
            }
        if dist_ctx.enabled:
            import torch.distributed as dist

            accepted_state = torch.tensor(
                [
                    float("nan")
                    if accepted_ancestor_pairwise_accuracy is None
                    else accepted_ancestor_pairwise_accuracy
                ],
                dtype=torch.float64,
                device=resolved_device,
            )
            dist.broadcast(accepted_state, src=0)
            accepted_value = float(accepted_state.item())
            accepted_ancestor_pairwise_accuracy = (
                None if math.isnan(accepted_value) else accepted_value
            )
    accepted_baseline_sec = time.perf_counter() - accepted_baseline_started
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev_is_cuda)

    m = unwrap(model)
    log_var_params = list(m.loss_log_vars.parameters())
    log_var_ids = {id(p) for p in log_var_params}
    encoder_params = [p for p in m.encoder.parameters()]
    encoder_ids = {id(p) for p in encoder_params}
    reward_modules = [
        m.reward_token_emb,
        m.reward_norm,
        m.reward_heads,
        m.z_gamma,
        m.z_proj,
    ]
    # v3.4: the reward context encoder joins the REWARD group, not the encoder group. That
    # is deliberate — the encoder group's LR is cut to 0.1x at the phase A->B boundary
    # (below, opt.param_groups[1]) to stop the sequence path from re-shaping z late in a
    # run. The reward branch has the opposite need: its ranking signal is the weakest and
    # slowest of the objectives and must keep learning through phase B.
    if getattr(m, "reward_encoder", None) is not None:
        reward_modules.append(m.reward_encoder)
    if m.binary_reward_heads is not None:
        reward_modules.append(m.binary_reward_heads)
    head_modules = [m.stability_head]
    if m.binary_stability_head is not None:
        head_modules.append(m.binary_stability_head)
    if m.routing_head is not None:
        head_modules.append(m.routing_head)
    if m.binary_routing_head is not None:
        head_modules.append(m.binary_routing_head)
    if not anti_overfit_experiment:
        reward_modules.extend(head_modules)
        head_modules = []
    head_params = [p for module in head_modules for p in module.parameters()]
    head_ids = {id(p) for p in head_params}
    reward_params = [p for module in reward_modules for p in module.parameters()]
    reward_ids = {id(p) for p in reward_params}
    other_params = [
        p
        for p in model.parameters()
        if id(p) not in log_var_ids
        and id(p) not in encoder_ids
        and id(p) not in reward_ids
        and id(p) not in head_ids
    ]
    decayed_weight = ANTI_OVERFIT_WEIGHT_DECAY if anti_overfit_experiment else 0.01
    if training_schedule is not None:
        optimizer_groups, _ = _configure_preregistered_training_schedule(
            m, training_schedule, weight_decay=decayed_weight
        )
    else:
        optimizer_groups = [
            {"params": other_params, "lr": lr, "weight_decay": decayed_weight},
            {"params": encoder_params, "lr": lr, "weight_decay": decayed_weight},
            # No weight decay on the reward branch: under a weak pairwise signal decay
            # steadily shrinks it toward a constant output (a collapse driver).
            {"params": reward_params, "lr": lr, "weight_decay": 0.0},
            {"params": log_var_params, "lr": lr, "weight_decay": 0.0},
        ]
        if anti_overfit_experiment:
            optimizer_groups.append(
                {
                    "params": head_params,
                    "lr": lr,
                    "weight_decay": ANTI_OVERFIT_WEIGHT_DECAY,
                }
            )
    # TF32 + a fused optimizer are pure throughput on Ampere-class hardware
    # (the 3090 this trains on) and change no semantics we depend on: reductions
    # stay fp32, and the ranking objective is far above TF32's mantissa noise.
    # AMP is already on, so the matmuls this affects are the fp32 fallbacks.
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    opt = torch.optim.AdamW(
        optimizer_groups,
        lr=lr,
        weight_decay=decayed_weight,
        # Fused AdamW collapses the per-parameter-group elementwise kernels into
        # one launch; with 152 tensors across 4-5 groups that is a measurable
        # slice of step time. Falls back automatically off CUDA.
        fused=bool(resolved_device.startswith("cuda") and torch.cuda.is_available()),
    )
    rng = random.Random(seed)
    phase_a_epochs = (
        int(resume_payload["phase_a_epochs"])
        if resume_payload is not None
        else (
            0
            if phase_a_fraction <= 0.0
            else max(1, int(round(phase_a_fraction * epochs)))
        )
    )
    start_epoch = (
        int(resume_payload.get("global_epoch", 0))
        if resume_payload
        else int(initial_epoch)
    )
    if training_schedule is not None:
        scheduler_lambdas = [(lambda _: 1.0) for _ in optimizer_groups]
    else:
        scheduler_lambdas = [
            (lambda _: 1.0),
            (lambda step: 0.1 if step >= phase_a_epochs else 1.0),
            (lambda _: 1.0),
            (lambda _: 1.0),
        ]
        if anti_overfit_experiment:
            scheduler_lambdas.append(lambda _: 1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=scheduler_lambdas
    )
    if resume_payload is not None and not branch_from_accepted_resume:
        optimizer_state_compatible = resume_payload.get(
            "optimizer_state_compatible_with_model", True
        )
        if type(optimizer_state_compatible) is not bool:
            raise ValueError(
                "resume optimizer/model compatibility marker must be a boolean"
            )
        saved_optimizer = resume_payload["optimizer"]
        optimizer_layout_matches = len(saved_optimizer.get("param_groups", [])) == len(
            opt.param_groups
        )
        if optimizer_state_compatible and optimizer_layout_matches:
            opt.load_state_dict(saved_optimizer)
            scheduler.load_state_dict(resume_payload["scheduler"])
        elif is_main(dist_ctx) and not optimizer_state_compatible:
            print(
                "resume model is an arena-selectable historical/EMA state: "
                "starting fresh optimizer, scheduler, and scaler state"
            )
        elif is_main(dist_ctx):
            print(
                "anti-overfit optimizer layout changed: starting fresh optimizer moments"
            )
        # A checkpoint written while AMP was disabled stores an EMPTY GradScaler
        # state dict; loading that into an enabled scaler raises "source state
        # dict is empty". When enabling AMP on resume, skip the empty state and
        # let the scaler recalibrate from its default scale (safe — it self-tunes
        # within a few steps). Populated scaler state is still restored normally.
        saved_scaler = resume_payload.get("scaler") or {}
        if optimizer_state_compatible and saved_scaler.get("scale") is not None:
            scaler.load_state_dict(saved_scaler)
        _restore_rng_state(resume_payload["rng"], rng)
    requested_batch_size = batch_size
    batch_benchmark: list[dict[str, Any]] = []
    batch_selection = "configured"
    if autotune_batch_size and dev_is_cuda:
        saved_batch_size = (
            resume_payload.get("batch_size") if resume_payload is not None else None
        )
        if (
            resume_payload is not None
            and resume_payload.get("batch_autotune") is True
            and type(saved_batch_size) is int
            and saved_batch_size in CUDA_BATCH_CANDIDATES
        ):
            batch_size = saved_batch_size
            batch_benchmark = list(resume_payload.get("batch_benchmark") or [])
            batch_selection = "resume"
        else:
            benchmark_phase = "A" if start_epoch + 1 <= phase_a_epochs else "B"
            benchmark_profile = (
                PHASE_A_PROFILE if benchmark_phase == "A" else PHASE_B_PROFILE
            )
            batch_benchmark = _benchmark_cuda_batch_sizes(
                unwrap(model),
                opt,
                train_rows,
                vocab,
                device=resolved_device,
                phase=benchmark_phase,
                profile=benchmark_profile,
                amp=scaler.is_enabled(),
                artifact_cache=artifact_cache,
                dist_ctx=dist_ctx,
                rng=rng,
            )
            batch_size = _select_cuda_batch_size(batch_benchmark)
            batch_selection = "benchmark"
        if is_main(dist_ctx):
            print(
                f"CUDA batch selection: {batch_size} ({batch_selection}; "
                f"candidates={batch_benchmark})"
            )
    history: list[float] = (
        list(resume_payload.get("loss_history", [])) if resume_payload else []
    )
    retain_prior_selection = resume_payload is not None and not reset_checkpoint_selection_on_resume
    best_value: float | None = (
        resume_payload.get("best_value") if retain_prior_selection else None
    )
    best_payload: dict[str, Any] | None = (
        resume_payload.get("best_payload") if retain_prior_selection else None
    )
    imitation_payload: dict[str, Any] | None = (
        resume_payload.get("imitation_payload") if retain_prior_selection else None
    )
    ema_state: dict[str, torch.Tensor] | None = None
    if anti_overfit_experiment:
        saved_ema = (
            resume_payload.get("anti_overfit_ema")
            if resume_payload is not None
            else None
        )
        ema_state = (
            {
                key: value.detach().clone().to(resolved_device)
                for key, value in saved_ema.items()
            }
            if isinstance(saved_ema, dict) and saved_ema
            else {
                key: value.detach().clone()
                for key, value in unwrap(model).state_dict().items()
            }
        )
    retain_plateau_state = anti_overfit_experiment and retain_prior_selection
    plateau_best: float | None = (
        resume_payload.get("anti_overfit_plateau_best")
        if retain_plateau_state
        else None
    )
    plateau_bad_epochs = (
        int(resume_payload.get("anti_overfit_plateau_bad_epochs", 0) or 0)
        if retain_plateau_state
        else 0
    )
    anti_overfit_lr_scale = (
        float(resume_payload.get("anti_overfit_lr_scale", 1.0) or 1.0)
        if retain_plateau_state
        else 1.0
    )
    started = time.time()
    current_phase = ""
    grammar_specs_json = [spec.to_json() for spec in specs]
    latest_resume_state: dict[str, Any] | None = None
    family_held_best: float | None = None
    family_held_bad_passes = 0
    family_held_early_stopped = False
    macro_validation_best: float | None = (
        resume_payload.get("macro_validation_best")
        if resume_payload is not None and early_stopping_patience is not None
        else None
    )
    macro_validation_bad_passes = (
        int(resume_payload.get("macro_validation_bad_passes", 0) or 0)
        if resume_payload is not None and early_stopping_patience is not None
        else 0
    )
    macro_validation_early_stopped = False
    completed_local_epochs = 0

    for local_epoch in range(1, max(1, epochs) + 1):
        epoch = start_epoch + local_epoch
        completed_local_epochs = local_epoch
        nonfinite_batches = 0
        phase = "A" if epoch <= phase_a_epochs else "B"
        if phase != current_phase:
            current_phase = phase
            unwrap(model).set_phase(phase)
            if training_schedule is None:
                opt.param_groups[1]["lr"] = (
                    lr * (0.1 if phase == "B" else 1.0) * anti_overfit_lr_scale
                )
        profile = PHASE_A_PROFILE if phase == "A" else PHASE_B_PROFILE
        model.train()
        epoch_started = time.time()
        # accumulate on-device — no per-step .cpu() sync (that stalled the pipeline)
        total_t = torch.zeros((), device=resolved_device)
        component_totals_t: dict[str, torch.Tensor] = {
            name: torch.zeros((), device=resolved_device)
            for name in POLICY_V3_LOSS_COMPONENTS + POLICY_V3_DIAGNOSTIC_COMPONENTS
        }
        steps = 0
        shuffled = train_rows[:]
        rng.shuffle(shuffled)
        all_batches = _grouped_batches(shuffled, batch_size, rng)
        # DDP: each rank trains a disjoint stride of the grouped batches (grouping preserved).
        # CRITICAL: every rank must run the SAME number of steps — with manual gradient
        # all-reduce, an uneven split lets one rank exit early and its next collective
        # (the epoch-total all_reduce) pairs against the other rank's gradient all_reduce
        # → NCCL deadlock (observed: frozen at batch N-1/N, both GPUs spinning at 100%).
        # The rng is seed-identical across ranks, so all_batches is identical everywhere;
        # truncate every shard to the minimum shard length (loses ≤ world-1 batches/epoch).
        my_batches = shard_for_rank(all_batches, dist_ctx)
        if dist_ctx.enabled and dist_ctx.world_size > 1:
            min_len = min(
                len(all_batches[r :: dist_ctx.world_size])
                for r in range(dist_ctx.world_size)
            )
            my_batches = my_batches[:min_len]
        n_batches = max(1, len(my_batches))
        emit_progress(
            "cuda_train" if dev_is_cuda else "cpu_train",
            epoch=epoch,
            epochs=epochs,
            training_phase=phase,
            batch=0,
            total_batches=n_batches,
            pct=0.0,
            input_mode=input_mode,
            device=str(resolved_device),
        )
        last_progress = 0.0
        prepared_batches = _iter_policy_batches(
            my_batches,
            vocab,
            device=resolved_device,
            artifact_cache=artifact_cache,
        )
        for bi, batch in enumerate(prepared_batches):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                components = unwrap(model).loss_components(batch, phase=phase)
                loss = kg_combine(components, unwrap(model).loss_log_vars, profile)
            # NaN guard: never backprop a nonfinite loss — one bad batch otherwise
            # corrupts the weights and every later batch/val reads NaN (seen gen 7).
            finite_loss = bool(torch.isfinite(loss).detach().item())
            if not all_ranks_true(finite_loss, dist_ctx):
                opt.zero_grad()
                nonfinite_batches += 1
                del loss
                del components
                del batch
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                all_reduce_gradients(
                    unwrap(model), dist_ctx
                )  # average grads across ranks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                all_reduce_gradients(
                    unwrap(model), dist_ctx
                )  # average grads across ranks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            clamp_log_vars(unwrap(model).loss_log_vars)
            if ema_state is not None:
                _update_ema_state(ema_state, unwrap(model).state_dict())
            total_t += loss.detach()
            for name, value in components.items():
                if name in component_totals_t:
                    component_totals_t[name] += value.detach()
            steps += 1
            # Part D: sub-100ms in-epoch % progress (rank 0), throttled to ~50ms
            if progress_path is not None and is_main(dist_ctx):
                now = time.time()
                if now - last_progress >= 0.05 or bi + 1 == n_batches:
                    last_progress = now
                    emit_progress(
                        "cuda_train" if dev_is_cuda else "cpu_train",
                        epoch=epoch,
                        epochs=epochs,
                        training_phase=phase,
                        batch=bi + 1,
                        total_batches=n_batches,
                        pct=round(100.0 * (bi + 1) / n_batches, 1),
                        running_loss=float(total_t.detach().cpu()) / max(1, steps),
                        input_mode=input_mode,
                    )
            del loss
            del components
            del batch
        # single sync per epoch, then all_reduce across ranks
        local = {"__loss__": float(total_t.detach().cpu()), "__steps__": float(steps)}
        for name, t in component_totals_t.items():
            local[name] = float(t.detach().cpu())
        agg = aggregate_scalars(local, dist_ctx)
        steps_all = max(1.0, agg["__steps__"])
        train_loss = agg["__loss__"] / steps_all
        component_avgs = {
            k: agg[k] / steps_all
            for k in POLICY_V3_LOSS_COMPONENTS + POLICY_V3_DIAGNOSTIC_COMPONENTS
        }
        # Excess-over-floor reporting. Both of these headline numbers are dominated by an
        # irreducible constant, so the raw value cannot say whether the model learned
        # anything: reward_listwise is ListNet CE bounded below by its own target entropy
        # (~2.35 of a measured 2.26-2.49), and stability_bce is a pos_weighted BCE whose
        # best CONSTANT predictor scores 1.1959, not ln 2. The excesses are what to watch.
        component_avgs.update(_excess_over_floor(component_avgs))
        history.append(train_loss)
        # Fail fast on a poisoned epoch: weights are unrecoverable once NaN — raising
        # lets the supervisor relaunch and warm-start from the last healthy checkpoint
        # instead of burning the remaining epochs (and risking a garbage checkpoint).
        if not math.isfinite(train_loss):
            raise RuntimeError(
                f"epoch {epoch}: train_loss is nonfinite ({nonfinite_batches} nonfinite "
                f"batches skipped) — aborting so the supervisor restarts from a healthy checkpoint"
            )
        if nonfinite_batches and is_main(dist_ctx):
            print(
                f"WARNING epoch {epoch}: skipped {nonfinite_batches} nonfinite-loss batches"
            )

        # Part C: full val loss on held-out real datasets (rank 0), same profile as train.
        validation_progress_state: dict[str, Any] = {}

        def report_validation_progress(update: dict[str, Any]) -> None:
            validation_progress_state.update(update)
            emit_progress(
                "validation",
                epoch=epoch,
                epochs=epochs,
                training_phase=phase,
                batch=n_batches,
                total_batches=n_batches,
                pct=100.0,
                running_loss=train_loss,
                input_mode=input_mode,
                **update,
            )

        emit_progress(
            "validation",
            epoch=epoch,
            epochs=epochs,
            training_phase=phase,
            batch=n_batches,
            total_batches=n_batches,
            pct=100.0,
            running_loss=train_loss,
            input_mode=input_mode,
            validation_batch=0,
            validation_batches=0,
            validation_pct=0.0,
        )
        raw_state_backup: dict[str, torch.Tensor] | None = None
        if ema_state is not None:
            raw_state_backup = {
                key: value.detach().clone()
                for key, value in unwrap(model).state_dict().items()
            }
            unwrap(model).load_state_dict(ema_state)
        val_loss, val_loss_components = (
            _val_loss(
                model,
                val_rows,
                vocab,
                # Validation runs at the CONFIGURED batch size, never the VRAM
                # autotuned one. Listwise/BCE losses scale with per-batch group
                # size, so letting the autotuner pick 1024 over the requested 256
                # inflated val_loss by ~0.9 nats without changing model quality —
                # and made val_loss incomparable across machines and runs.
                requested_batch_size,
                resolved_device,
                profile,
                artifact_cache,
                progress=report_validation_progress,
            )
            if is_main(dist_ctx)
            else (None, {})
        )

        emit_progress(
            "validation_metrics",
            epoch=epoch,
            epochs=epochs,
            training_phase=phase,
            batch=n_batches,
            total_batches=n_batches,
            pct=100.0,
            running_loss=train_loss,
            val_loss=val_loss,
            input_mode=input_mode,
            **{
                **validation_progress_state,
                "val_partial": val_loss,
                "val_partial_complete": val_loss is not None,
            },
        )
        metrics = (
            policy_v3_validation_metrics(
                unwrap(model),
                val_rows,
                vocab,
                batch_size=batch_size,
                device=resolved_device,
                greedy_eval_datasets=greedy_eval_datasets
                if epoch % 5 == 0 or epoch == epochs
                else 0,
                router=router,
            )
            if is_main(dist_ctx)
            else {}
        )
        stop_after_pass = False
        if family_held_early_stopping:
            if is_main(dist_ctx):
                (
                    family_held_best,
                    family_held_bad_passes,
                    stop_after_pass,
                ) = _advance_family_held_early_stopping(
                    metrics.get("pairwise_win_accuracy"),
                    best=family_held_best,
                    bad_passes=family_held_bad_passes,
                )
            if dist_ctx.enabled:
                import torch.distributed as dist

                early_stop_state = torch.tensor(
                    [
                        float("nan") if family_held_best is None else family_held_best,
                        float(family_held_bad_passes),
                        1.0 if stop_after_pass else 0.0,
                    ],
                    dtype=torch.float64,
                    device=resolved_device,
                )
                dist.broadcast(early_stop_state, src=0)
                early_stop_values = early_stop_state.tolist()
                family_held_best = (
                    None
                    if math.isnan(early_stop_values[0])
                    else float(early_stop_values[0])
                )
                family_held_bad_passes = int(early_stop_values[1])
                stop_after_pass = bool(early_stop_values[2])
            family_held_early_stopped = stop_after_pass
        if early_stopping_patience is not None and phase == "B":
            if is_main(dist_ctx):
                macro_value = metrics.get(checkpoint_metric.removeprefix("val_"))
                if macro_value is not None and (
                    macro_validation_best is None or macro_value > macro_validation_best
                ):
                    macro_validation_best = float(macro_value)
                    macro_validation_bad_passes = 0
                else:
                    macro_validation_bad_passes += 1
                macro_validation_early_stopped = (
                    macro_validation_bad_passes >= early_stopping_patience
                )
            if dist_ctx.enabled:
                import torch.distributed as dist

                macro_stop_state = torch.tensor(
                    [
                        float("nan")
                        if macro_validation_best is None
                        else macro_validation_best,
                        float(macro_validation_bad_passes),
                        1.0 if macro_validation_early_stopped else 0.0,
                    ],
                    dtype=torch.float64,
                    device=resolved_device,
                )
                dist.broadcast(macro_stop_state, src=0)
                macro_values = macro_stop_state.tolist()
                macro_validation_best = (
                    None if math.isnan(macro_values[0]) else float(macro_values[0])
                )
                macro_validation_bad_passes = int(macro_values[1])
                macro_validation_early_stopped = bool(macro_values[2])
            stop_after_pass = stop_after_pass or macro_validation_early_stopped
        overfit_gap = (val_loss - train_loss) if val_loss is not None else None
        row = {
            "epoch": epoch,
            "epochs": epochs,
            "phase": phase,
            "train_loss": train_loss,
            "loss": train_loss,
            "val_loss": val_loss,
            "overfit_gap": overfit_gap,
            "val_loss_components": val_loss_components,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "val_fallback": val_fallback,
            "batch_size": batch_size,
            "requested_batch_size": requested_batch_size,
            "batch_selection": batch_selection,
            "batch_benchmark": batch_benchmark,
            "cuda_prefetch_depth": CUDA_PREFETCH_DEPTH if dev_is_cuda else 0,
            "amp": scaler.is_enabled(),
            "world_size": dist_ctx.world_size,
            "lr": lr,
            "device": str(resolved_device),
            "model_version": MODEL_VERSION,
            "epoch_sec": time.time() - epoch_started,
            "elapsed_sec": time.time() - started,
            "loss_weights": {
                name: float(torch.exp(-p.detach()).cpu())
                for name, p in unwrap(model).loss_log_vars.items()
            },
            "train_loss_components": {k: v for k, v in sorted(component_avgs.items())},
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        if family_held_early_stopping:
            row.update(
                {
                    "family_held_early_stopping": True,
                    "family_held_best_pairwise_accuracy": family_held_best,
                    "family_held_bad_passes": family_held_bad_passes,
                    "family_held_early_stopped": stop_after_pass,
                }
            )
        if early_stopping_patience is not None:
            row.update(
                {
                    "early_stopping_patience": early_stopping_patience,
                    "macro_validation_best": macro_validation_best,
                    "macro_validation_bad_passes": macro_validation_bad_passes,
                    "macro_validation_early_stopped": macro_validation_early_stopped,
                }
            )
        if anti_overfit_experiment:
            row["anti_overfit_experiment"] = anti_overfit_experiment_config()
            row["anti_overfit_lr_scale"] = anti_overfit_lr_scale
            row["legacy_weighted_rows"] = legacy_weighted_rows
        emit_progress(
            "epoch_complete",
            epoch=epoch,
            epochs=epochs,
            training_phase=phase,
            batch=n_batches,
            total_batches=n_batches,
            pct=100.0,
            running_loss=train_loss,
            val_loss=val_loss,
            overfit_gap=overfit_gap,
            epoch_done=True,
            input_mode=input_mode,
            family_held_early_stopping=family_held_early_stopping,
            family_held_early_stopped=stop_after_pass,
        )
        if metrics_path is not None and is_main(dist_ctx):
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        if is_main(dist_ctx):
            if phase == "A":
                nll = metrics.get("seq_nll")
                if nll is not None and (
                    imitation_payload is None or nll < imitation_payload["metric_value"]
                ):
                    imitation_payload = {
                        "metric_value": nll,
                        "epoch": epoch,
                        "label_watermark": int(
                            (resume_extra or {}).get("source_label_watermark", 0) or 0
                        ),
                        "metrics": {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_seq_nll": nll,
                        },
                        "state": {
                            k: v.detach().cpu().clone()
                            for k, v in unwrap(model).state_dict().items()
                        },
                    }
            else:
                value = metrics.get(checkpoint_metric.removeprefix("val_"))
                if value is not None and (best_value is None or value > best_value):
                    best_value = value
                    best_payload = {
                        "metric_value": value,
                        "epoch": epoch,
                        "label_watermark": int(
                            (resume_extra or {}).get("source_label_watermark", 0) or 0
                        ),
                        "metrics": {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_pairwise_win_accuracy": metrics.get(
                                "pairwise_win_accuracy"
                            ),
                            "val_reward_cv_correlation": metrics.get(
                                "reward_cv_correlation"
                            ),
                            "val_sequence_validity": metrics.get(
                                "sequence_validity"
                            ),
                            "val_seq_nll": metrics.get("seq_nll"),
                        },
                        "state": {
                            k: v.detach().cpu().clone()
                            for k, v in unwrap(model).state_dict().items()
                        },
                    }
        if raw_state_backup is not None:
            unwrap(model).load_state_dict(raw_state_backup)
        lr_halved = False
        if anti_overfit_experiment:
            plateau_metric = (
                metrics.get(checkpoint_metric.removeprefix("val_"))
                if is_main(dist_ctx) and phase == "B"
                else None
            )
            if is_main(dist_ctx):
                (
                    plateau_best,
                    plateau_bad_epochs,
                    anti_overfit_lr_scale,
                    lr_halved,
                ) = _advance_plateau_lr(
                    plateau_metric,
                    best=plateau_best,
                    bad_epochs=plateau_bad_epochs,
                    lr_scale=anti_overfit_lr_scale,
                )
            if dist_ctx.enabled:
                import torch.distributed as dist

                plateau_state = torch.tensor(
                    [
                        float("nan") if plateau_best is None else plateau_best,
                        float(plateau_bad_epochs),
                        anti_overfit_lr_scale,
                        1.0 if lr_halved else 0.0,
                    ],
                    dtype=torch.float64,
                    device=resolved_device,
                )
                dist.broadcast(plateau_state, src=0)
                plateau_values = plateau_state.tolist()
                plateau_best = (
                    None
                    if math.isnan(plateau_values[0])
                    else float(plateau_values[0])
                )
                plateau_bad_epochs = int(plateau_values[1])
                anti_overfit_lr_scale = float(plateau_values[2])
                lr_halved = bool(plateau_values[3])
        scheduler.step()
        if anti_overfit_experiment:
            for group in opt.param_groups:
                group["lr"] *= anti_overfit_lr_scale
        historical_selected_payload = best_payload or imitation_payload
        checkpoint_state_epoch = (
            epoch
            if checkpoint_state == "current"
            else (historical_selected_payload or {}).get("epoch")
        )
        checkpoint_state_metric = (
            metrics.get(checkpoint_metric.removeprefix("val_"))
            if checkpoint_state == "current"
            else (historical_selected_payload or {}).get("metric_value")
        )
        checkpoint_state_label_watermark = (
            int((resume_extra or {}).get("source_label_watermark", 0) or 0)
            if checkpoint_state == "current"
            else (historical_selected_payload or {}).get("label_watermark")
        )
        emit_progress(
            "checkpoint_write",
            epoch=epoch,
            epochs=epochs,
            training_phase=phase,
            batch=n_batches,
            total_batches=n_batches,
            pct=100.0,
            running_loss=train_loss,
            val_loss=val_loss,
            overfit_gap=overfit_gap,
            continuation_epoch=epoch,
            checkpoint_state_kind=checkpoint_state,
            selected_checkpoint_epoch=checkpoint_state_epoch,
            selected_checkpoint_metric=checkpoint_state_metric,
            selected_checkpoint_label_watermark=checkpoint_state_label_watermark,
            selected_state_is_current=checkpoint_state == "current",
            historical_best_epoch=(best_payload or {}).get("epoch"),
            historical_best_metric=(best_payload or {}).get("metric_value"),
            **(
                {
                    "anti_overfit_lr_halved": lr_halved,
                    "anti_overfit_lr_scale": anti_overfit_lr_scale,
                }
                if anti_overfit_experiment
                else {}
            ),
        )
        if resume_checkpoint is not None and is_main(dist_ctx):
            resume_state = {
                "schema": TRAINER_RESUME_V1,
                "global_epoch": epoch,
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in unwrap(model).state_dict().items()
                },
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _capture_rng_state(rng),
                "loss_history": history,
                "best_value": best_value,
                "best_payload": best_payload,
                "imitation_payload": imitation_payload,
                "phase_a_epochs": phase_a_epochs,
                "routing_classes": n_routing,
                "routing_families": routing_families,
                "action_vocab": {"tokens": vocab.tokens, "version": vocab.version},
                "grammar_specs": grammar_specs_json,
                "embedding_rows": unwrap(model).embedding_rows,
                "batch_size": batch_size,
                "batch_autotune": bool(autotune_batch_size and dev_is_cuda),
                "batch_benchmark": batch_benchmark,
                "last_checkpoint_state": checkpoint_state,
                "resume_model_state_kind": "current",
                "optimizer_state_compatible_with_model": True,
                "architecture": {
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "ffn_dim": ffn_dim,
                    "dataset_dim": row_dataset_dim,
                },
                "trainer_extra": dict(resume_extra or {}),
                "macro_validation_best": macro_validation_best,
                "macro_validation_bad_passes": macro_validation_bad_passes,
            }
            if family_held_early_stopping:
                resume_state.update(
                    {
                        "family_held_early_stopping": True,
                        "family_held_best_pairwise_accuracy": family_held_best,
                        "family_held_bad_passes": family_held_bad_passes,
                        "family_held_early_stopped": stop_after_pass,
                    }
                )
            if training_schedule_identity is not None:
                resume_state["training_schedule"] = training_schedule_identity
            if anti_overfit_experiment:
                resume_state.update(
                    {
                        "anti_overfit_experiment": anti_overfit_experiment_config(),
                        "anti_overfit_ema": {
                            key: value.detach().cpu().clone()
                            for key, value in (ema_state or {}).items()
                        },
                        "anti_overfit_plateau_best": plateau_best,
                        "anti_overfit_plateau_bad_epochs": plateau_bad_epochs,
                        "anti_overfit_lr_scale": anti_overfit_lr_scale,
                    }
                )
            latest_resume_state = resume_state
            _atomic_torch_save(resume_state, Path(resume_checkpoint))
        barrier(dist_ctx)
        if stop_after_pass:
            break

    core = unwrap(model)
    continuation_epoch = start_epoch + completed_local_epochs
    historical_selected_payload = best_payload or imitation_payload
    checkpoint_selected_payload = (
        None if checkpoint_state == "current" else historical_selected_payload
    )
    checkpoint_state_epoch = (
        continuation_epoch
        if checkpoint_state == "current"
        else (checkpoint_selected_payload or {}).get("epoch")
    )
    checkpoint_state_metric_name = (
        checkpoint_metric
        if best_payload is not None or checkpoint_state == "current"
        else "val_seq_nll"
    )
    checkpoint_state_metric_value = (
        metrics.get(checkpoint_metric.removeprefix("val_"))
        if checkpoint_state == "current"
        else (checkpoint_selected_payload or {}).get("metric_value")
    )
    checkpoint_state_label_watermark = (
        int((resume_extra or {}).get("source_label_watermark", 0) or 0)
        if checkpoint_state == "current"
        else (checkpoint_selected_payload or {}).get("label_watermark")
    )
    checkpoint_state_metrics = (
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        if checkpoint_state == "current"
        else (checkpoint_selected_payload or {}).get("metrics")
    )
    if checkpoint_selected_payload is not None:
        published_model_state_kind = (
            "ema_best"
            if anti_overfit_experiment
            else (
                "current_best"
                if checkpoint_selected_payload.get("epoch") == continuation_epoch
                else "historical_best"
            )
        )
    elif ema_state is not None:
        published_model_state_kind = "ema_current"
    else:
        published_model_state_kind = "current"
    optimizer_state_compatible_with_model = published_model_state_kind in {
        "current",
        "current_best",
    }
    if checkpoint is not None and is_main(dist_ctx):
        out = Path(checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_selected_payload is not None:
            core.load_state_dict(checkpoint_selected_payload["state"])
        elif ema_state is not None:
            core.load_state_dict(ema_state)
        _atomic_torch_save(
            core.checkpoint_payload(
                grammar_specs=grammar_specs_json,
                loss_history=history,
                epochs=continuation_epoch,
                routing_families=routing_families,
                checkpoint_metric=checkpoint_metric,
                best_checkpoint_metric_value=(best_payload or {}).get("metric_value"),
                best_checkpoint_epoch=(best_payload or {}).get("epoch"),
                historical_best_label_watermark=(best_payload or {}).get(
                    "label_watermark"
                ),
                checkpoint_state_kind=checkpoint_state,
                checkpoint_state_epoch=checkpoint_state_epoch,
                checkpoint_state_metric_name=checkpoint_state_metric_name,
                checkpoint_state_metric_value=checkpoint_state_metric_value,
                checkpoint_state_label_watermark=checkpoint_state_label_watermark,
                checkpoint_state_metrics=checkpoint_state_metrics,
                historical_best_epoch=(best_payload or {}).get("epoch"),
                historical_best_metric_value=(best_payload or {}).get("metric_value"),
                continuation_epoch=continuation_epoch,
                val_fallback=val_fallback,
                batch_size=batch_size,
                batch_autotune=bool(autotune_batch_size and dev_is_cuda),
                batch_benchmark=batch_benchmark,
                amp=scaler.is_enabled(),
                cuda_prefetch_depth=CUDA_PREFETCH_DEPTH if dev_is_cuda else 0,
                **(
                    {
                        "family_held_early_stopping": True,
                        "family_held_best_pairwise_accuracy": family_held_best,
                        "family_held_bad_passes": family_held_bad_passes,
                        "family_held_early_stopped": family_held_early_stopped,
                        "passes_completed": completed_local_epochs,
                    }
                    if family_held_early_stopping
                    else {}
                ),
                **(
                    {"training_schedule": training_schedule_identity}
                    if training_schedule_identity is not None
                    else {}
                ),
                **(
                    {
                        "anti_overfit_experiment": anti_overfit_experiment_config(),
                        "anti_overfit_lr_scale": anti_overfit_lr_scale,
                        "legacy_weighted_rows": legacy_weighted_rows,
                    }
                    if anti_overfit_experiment
                    else {}
                ),
            ),
            out,
        )
        if imitation_payload is not None:
            imitation_out = out.with_suffix(".imitation.pt")
            state_backup = {
                k: v.detach().cpu().clone() for k, v in core.state_dict().items()
            }
            core.load_state_dict(imitation_payload["state"])
            _atomic_torch_save(
                core.checkpoint_payload(
                    grammar_specs=grammar_specs_json,
                    curriculum_phase="A",
                    best_checkpoint_metric_value=imitation_payload["metric_value"],
                    best_checkpoint_epoch=imitation_payload["epoch"],
                    val_fallback=val_fallback,
                    **(
                        {"training_schedule": training_schedule_identity}
                        if training_schedule_identity is not None
                        else {}
                    ),
                    **(
                        {
                            "anti_overfit_experiment": anti_overfit_experiment_config()
                        }
                        if anti_overfit_experiment
                        else {}
                    ),
                ),
                imitation_out,
            )
            core.load_state_dict(state_backup)
        if resume_checkpoint is not None:
            if latest_resume_state is None:
                raise RuntimeError("trainer completed without a resumable state")
            # Arena acceptance binds the published model bytes.  A historical-best
            # or EMA model cannot safely carry optimizer moments from the final raw
            # training state, so publish the exact same model in the resume artifact
            # and mark its optimizer/scheduler/scaler state as non-reusable.  The next
            # accepted-lineage cycle then starts fresh optimizer state from precisely
            # the model that won the arena.
            latest_resume_state["model"] = {
                key: value.detach().cpu().clone()
                for key, value in core.state_dict().items()
            }
            latest_resume_state["last_checkpoint_state"] = checkpoint_state
            latest_resume_state["resume_model_state_kind"] = (
                published_model_state_kind
            )
            latest_resume_state["optimizer_state_compatible_with_model"] = (
                optimizer_state_compatible_with_model
            )
            _atomic_torch_save(latest_resume_state, Path(resume_checkpoint))
    barrier(dist_ctx)
    shutdown_distributed(dist_ctx)
    summary = {
        "rows": len(rows),
        "input_rows": input_rows,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "report_only_rows": len(report_only_rows),
        "report_only_split_counts": split_partition["report_only_counts"],
        "split_compatibility": split_partition["compatibility"],
        "split_manifest_sha256": split_partition["split_manifest_sha256"],
        "epochs": epochs,
        "passes_completed": completed_local_epochs,
        "global_epoch": continuation_epoch,
        "loss": history[-1] if history else None,
        "loss_history": history,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "best_metric": best_value,
        "historical_best_epoch": (best_payload or {}).get("epoch"),
        "historical_best_metric": (best_payload or {}).get("metric_value"),
        "checkpoint_state_kind": checkpoint_state,
        "selected_checkpoint_epoch": checkpoint_state_epoch,
        "selected_checkpoint_label_watermark": checkpoint_state_label_watermark,
        "selected_checkpoint_metric_name": checkpoint_state_metric_name,
        "selected_checkpoint_metric": checkpoint_state_metric_value,
        "selected_checkpoint_metrics": checkpoint_state_metrics,
        "selected_state_is_current": checkpoint_state == "current",
        "resume_model_state_kind": published_model_state_kind,
        "optimizer_state_compatible_with_model": (
            optimizer_state_compatible_with_model
        ),
        "family_held_early_stopping": family_held_early_stopping,
        "family_held_best_pairwise_accuracy": family_held_best,
        "family_held_bad_passes": family_held_bad_passes,
        "family_held_early_stopped": family_held_early_stopped,
        "early_stopping_patience": early_stopping_patience,
        "macro_validation_best": macro_validation_best,
        "macro_validation_bad_passes": macro_validation_bad_passes,
        "macro_validation_early_stopped": macro_validation_early_stopped,
        "accepted_ancestor_pairwise_accuracy": accepted_ancestor_pairwise_accuracy,
        "accepted_ancestor_metrics": accepted_ancestor_metrics,
        "accepted_ancestor_baseline_sec": accepted_baseline_sec,
        "input_mode": input_mode,
        "val_fallback": val_fallback,
        "vocab_size": vocab.size,
        "embedding_rows": embedding_rows,
        "routing_classes": n_routing,
        "batch_size": batch_size,
        "requested_batch_size": requested_batch_size,
        "batch_selection": batch_selection,
        "batch_benchmark": batch_benchmark,
        "cuda_prefetch_depth": CUDA_PREFETCH_DEPTH if dev_is_cuda else 0,
        "amp": scaler.is_enabled(),
        "preprocess_sec": preprocess_sec,
        "preprocess_cache_hits": preprocess_cache_hits,
        "preprocess_cache_misses": preprocess_cache_misses,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }
    if anti_overfit_experiment:
        summary.update(
            {
                "anti_overfit_experiment": anti_overfit_experiment_config(),
                "anti_overfit_lr_scale": anti_overfit_lr_scale,
                "anti_overfit_plateau_bad_epochs": plateau_bad_epochs,
                "legacy_weighted_rows": legacy_weighted_rows,
            }
        )
    if training_schedule_identity is not None:
        summary["training_schedule"] = training_schedule_identity
    return summary


@torch.no_grad()
def policy_v3_validation_metrics(
    model: JopePolicyModelV3,
    rows: list[TrainingRow],
    vocab,
    *,
    batch_size: int = 256,
    device: str = "cpu",
    greedy_eval_datasets: int = 0,
    router: EngineRouter | None = None,
) -> dict[str, float]:
    if not rows:
        return {}
    model.eval()
    nll_total = 0.0
    nll_steps = 0
    head_hits: dict[str, int] = defaultdict(int)
    head_totals: dict[str, int] = defaultdict(int)
    values: list[float] = []
    preds: list[float] = []
    stds: list[float] = []
    ablation_preds: list[float] = []
    ids: list[str] = []
    tasks: list[str] = []
    from .losses import masked_sequence_nll

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = rows_to_policy_v3_batch(batch_rows, vocab, device=device)
        z = model.context(batch.embedding, batch.target_summary)
        # v3.4: reward_stats reads the REWARD encoder. Passing the sequence z here would
        # score every validation genome through the wrong representation and make every
        # ranking metric (and therefore checkpoint selection) meaningless.
        z_reward = model.reward_context(batch.embedding, batch.target_summary)
        logits = model.decoder_logits(z, batch.token_ids, batch.pad_mask)
        targets = batch.token_ids[:, 1:]
        target_pad = batch.pad_mask[:, 1:]
        nll = masked_sequence_nll(
            logits, targets, batch.legal, target_pad, label_smoothing=0.0
        )
        nll_total += float(nll)
        nll_steps += 1
        # per-engine head accuracy at the head-op position
        masked_logits = logits.masked_fill(~batch.legal, float("-inf"))
        pred_ids = masked_logits.argmax(dim=-1)
        for i, row in enumerate(batch_rows):
            seq = batch.token_ids[i]
            for t in range(1, int(batch.pad_mask[i].sum())):
                op = vocab.ops.get(int(seq[t]))
                if op is not None and op.role == "head":
                    head_totals[row.engine] += 1
                    head_hits[row.engine] += int(int(pred_ids[i, t - 1]) == int(seq[t]))
                    break
        mask = torch.isfinite(batch.robust_delta)
        if mask.any():
            # Route each row to its own task's reward heads. Omitting task_types
            # makes _task_lane_mask default EVERY row to "regression", so on a
            # fused-head checkpoint the binary rows were scored by the regression
            # heads and the binary branch was never exercised — which is why
            # val_binary_pairwise_win sat at chance and conditional_advantage
            # read negative regardless of what the model had actually learned.
            # (Unknown task_type values raise inside _task_lane_mask.)
            row_tasks = [
                str((row.metadata or {}).get("task_type") or "regression")
                for row in batch_rows
            ]
            mean, std = model.reward_stats(
                z_reward, batch.token_ids, batch.pad_mask, row_tasks
            )
            values.extend(batch.robust_delta[mask].cpu().tolist())
            preds.extend(mean[mask].cpu().tolist())
            stds.extend(std[mask].cpu().tolist())
            ids.extend(
                [d for d, keep in zip(batch.reward_pair_ids, mask.cpu().tolist()) if keep]
            )
            tasks.extend(
                [
                    str((row.metadata or {}).get("task_type") or "regression")
                    for row, keep in zip(batch_rows, mask.cpu().tolist())
                    if keep
                ]
            )
            if hasattr(model, "reward_ablation"):
                abl = model.reward_ablation(batch.token_ids, batch.pad_mask)
                ablation_preds.extend(abl[mask].cpu().tolist())

    out: dict[str, float] = {
        "seq_nll": nll_total / max(1, nll_steps),
        "pairwise_win_accuracy": _pairwise_accuracy(preds, values, ids),
        "top1_historical_regret": _top1_regret(preds, values, ids),
        "reward_cv_correlation": _pearson(preds, values),
        "uncertainty_error_spearman": _spearman(
            stds, [abs(p - v) for p, v in zip(preds, values)]
        ),
    }
    # Per-task lanes (four-task program): metrics are emitted only for tasks with
    # at least one validation row — a task absent from the val split contributes
    # no keys rather than NaN placeholders.
    per_task_ablation = (
        ablation_preds if ablation_preds and len(ablation_preds) == len(values) else None
    )
    for task in ("regression", "binary", "timeseries", "multiclass"):
        indexes = [index for index, value in enumerate(tasks) if value == task]
        if not indexes:
            continue
        task_preds = [preds[index] for index in indexes]
        task_values = [values[index] for index in indexes]
        task_ids = [ids[index] for index in indexes]
        out[f"{task}_pairwise_win_accuracy"] = _pairwise_accuracy(
            task_preds, task_values, task_ids
        )
        out[f"{task}_reward_cv_correlation"] = _pearson(task_preds, task_values)
        out[f"{task}_top1_historical_regret"] = _top1_regret(
            task_preds, task_values, task_ids
        )
        if per_task_ablation is not None:
            task_ablation_win = _pairwise_accuracy(
                [per_task_ablation[index] for index in indexes], task_values, task_ids
            )
            out[f"{task}_ablation_pairwise_win"] = task_ablation_win
            out[f"{task}_conditional_advantage"] = (
                out[f"{task}_pairwise_win_accuracy"] - task_ablation_win
            )
    # v3.1 anti-collapse guard: conditional heads must beat the genome-only control.
    if ablation_preds and len(ablation_preds) == len(values):
        abl_win = _pairwise_accuracy(ablation_preds, values, ids)
        out["ablation_pairwise_win"] = abl_win
        out["conditional_advantage"] = out["pairwise_win_accuracy"] - abl_win
        # Selection metric. Raw pairwise accuracy is fully satisfiable by the
        # genome-only control — every run so far has matched or trailed it while
        # scoring ~0.65 — so checkpoint selection on it cannot tell a policy that
        # uses the dataset embedding from one that ignores it. Weighting the
        # advantage makes the trainer keep the checkpoint that is actually
        # conditional, not merely accurate.
        out["conditional_score"] = (
            out["pairwise_win_accuracy"] + 2.0 * out["conditional_advantage"]
        )
    for engine, total in head_totals.items():
        out[f"head_accuracy_{engine}"] = head_hits[engine] / max(1, total)

    # greedy one-shot validity + free-sample behavior on a subsample of datasets
    if greedy_eval_datasets > 0 and router is not None:
        from .decode import greedy_sequence, sample_sequences

        by_dataset: dict[str, TrainingRow] = {}
        for row in rows:
            by_dataset.setdefault(str(row.dataset_id), row)
        subset = list(by_dataset.values())[:greedy_eval_datasets]
        probes: list[tuple[str, str]] = []
        engine_choices: list[str] = []
        expr_count = 0
        sample_count = 0
        stage_counts: list[int] = []
        for row in subset:
            emb = torch.tensor(row.embedding, dtype=torch.float32)
            tgt = (
                torch.tensor(row.target_embedding, dtype=torch.float32)
                if row.target_embedding
                else None
            )
            for engine in vocab.engines:
                seq = greedy_sequence(model, emb, tgt, engine=engine, device=device)
                _, actions = decode_tokens(seq, vocab)
                probes.append((engine, format_pipeline_actions(actions)))
            # free samples (engine unforced): measures engine routing + expr usage (RF6)
            for seq in sample_sequences(
                model, emb, tgt, n=4, temperature=0.8, seed=17, device=device
            ):
                engine_choice, actions = decode_tokens(seq, vocab)
                engine_choices.append(engine_choice)
                sample_count += 1
                stage_counts.append(len(actions.get("transforms") or []))
                if any(
                    "expr" in (st.get("params") or {})
                    for st in actions.get("transforms") or []
                ):
                    expr_count += 1
        results = router.validate_many(probes)
        out["one_shot_greedy_valid_rate"] = sum(
            1 for r in results if r.get("valid")
        ) / max(1, len(results))
        if sample_count:
            out["expr_genome_fraction"] = expr_count / sample_count
            out["mean_transform_count"] = sum(stage_counts) / sample_count
            from collections import Counter as _Counter

            probs = [c / sample_count for c in _Counter(engine_choices).values()]
            out["engine_choice_entropy"] = -sum(p * math.log(p) for p in probs if p > 0)
            out["engine_choice_share"] = {
                e: engine_choices.count(e) / sample_count for e in vocab.engines
            }
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return 0.0

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    return _pearson(ranks(a), ranks(b))
