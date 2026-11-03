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
# Keep the script importable as test_exact_train when executed by path; the
# split modules import shared configuration and helpers from this namespace.
sys.modules.setdefault("test_exact_train", sys.modules[__name__])

from test_exact_train_config import (
    CANDIDATE_ROW_CAP, CATALOG_POINTER, CORPUS_DB, EQUAL_QUOTA, ENABLED_TASK_METRICS,
    MANDATORY_TASKS,
    OPTIONAL_TASK_DATASET_FLOORS, OPTIONAL_TASK_READINESS, QUOTA_TAIL_REPLICATE,
    RECENCY_SLOTS, RECIPE_E_BATCH_MIX, RECIPE_E_LOSS_WEIGHTS,
    ROOT, SAMPLER_TAG, SCQ,
    STRIDE_FLOATS, TRAINER_POLICY, VALIDATION_QUOTA, WARM, WARM_SHA,
    atomic_json, prime_source_root, sha256_file,
    render_placement_block, render_resource_block,
)







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


from test_exact_train_policy import (
    GpuUtilizationMonitor, RECIPE_E_MODEL_OUTPUTS, eight_slot_task_weights, four_worker_policy_batch_iterator,
    load_optional_task_readiness, optional_task_training_allowed,
    recipe_e_batch_lane_counts, recipe_e_evidence_lane, recipe_e_training_trigger,
    require_optional_task_admission,
    trainable_task_metrics,
)
from test_exact_train_sampling import (
    OBSERVATION_COLUMNS, SPLIT_MEDIAN_RATIO_MAX, append_freeze_history, freeze_history_rows,
    complete_causal_members as _complete_causal_members, equal_quota_draw, even_spread, publish_freeze_guard,
    quota_distribution, task_reward,
)
from test_exact_train_catalog import (
    FREEZE_GUARD, FREEZE_HISTORY, create_catalog_binding, ensure_packed,
    publish_corpus_freeze_state, publish_corpus_pointer, resolve_catalog_binding,
    split_median_divergence,
)
from test_exact_train_batching import make_grouped_batches
from test_exact_train_corpus import prepare_corpus, _build_corpus_impl
from test_exact_train_training import _train_impl


def complete_causal_members(connection: sqlite3.Connection) -> tuple[dict, dict]:
    return _complete_causal_members(connection, root=ROOT)

def build_corpus(
    min_rows: int = 1,
    *,
    freeze_watermark: int | None = None,
    corpus_mode: str = "full_consolidation",
    catalog_binding_path: Path | None = None,
) -> dict:
    return _build_corpus_impl(
        min_rows,
        freeze_watermark=freeze_watermark,
        corpus_mode=corpus_mode,
        catalog_binding_path=catalog_binding_path,
    )


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
    return _train_impl(
        corpus_manifest, output, device=device, warm=warm, recipe=recipe,
        seed=seed, task=task, mode=mode, packed_manifest=packed_manifest,
        packed_source_train=packed_source_train,
        packed_source_validation=packed_source_validation,
        packed_catalog_data=packed_catalog_data,
        accepted_parent_sha256=accepted_parent_sha256,
        optional_task_readiness=optional_task_readiness,
        _prime_source_root=prime_source_root,
    )






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
