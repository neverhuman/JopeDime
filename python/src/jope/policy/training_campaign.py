"""Bounded experimental Python policy training; never native serving promotion."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

TASKS = ("regression", "binary", "multiclass")
LINEAGE = "asm.experimental-python-policy-v3/v1"
PLAN = "jope.training-plan/v1"
NUMERICS = "torch-policy-v3:float32-storage:cuda-tf32=true:amp=false:threads=1:experimental/v1"
CLASS_CONTEXT = "asm.experimental-class-partition-pool/v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path, deadline_unix_ms=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if deadline_unix_ms is not None:
                require(
                    int(time.time() * 1000) < deadline_unix_ms,
                    "category validation deadline exhausted",
                )
            h.update(block)
    if deadline_unix_ms is not None:
        require(
            int(time.time() * 1000) < deadline_unix_ms,
            "category validation deadline exhausted",
        )
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def artifact(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "artifact must be a regular file")
    return {"path": str(path), "sha256": sha(path)}


def checked(ref, deadline_unix_ms=None):
    require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256"},
        "artifact reference required",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is not None,
        "invalid artifact digest",
    )
    path = Path(ref["path"])
    require(
        path.is_absolute() and path.is_file() and sha(path, deadline_unix_ms) == ref["sha256"],
        "artifact changed",
    )
    return path


def write(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(value) + b"\n"
    if immutable:
        if path.exists():
            require(path.read_bytes() == data, f"immutable artifact differs: {path}")
            return
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.pending")
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_state(state):
    state = Path(state).resolve()
    require(
        not state.is_relative_to(Path("/tmp")),
        "training state requires durable custody outside /tmp",
    )
    return state


def runtime_identity():
    package = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*.py"))}
    packages = {}
    required = {"torch", "numpy", "safetensors"}
    # Hash installed bytes, not just a version or wheel RECORD that might be stale.
    # Pin every installed dependency, including CUDA bindings/path discovery and
    # preprocessing libraries. Package-name prefix allowlists miss new runtime edges.
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        files = {}
        for item in sorted(distribution.files or [], key=str):
            path = Path(distribution.locate_file(item))
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            require(path.is_file(), f"installed runtime file absent: {name}/{item}")
            files[str(item)] = sha(path)
        require(files, f"installed distribution has no file inventory: {name}")
        require(name not in packages, f"ambiguous installed runtime distribution: {name}")
        packages[name] = {
            "version": distribution.version,
            "files_sha256": digest(files),
            "files": len(files),
        }
    require(required.issubset(packages), "required numerical distribution unavailable")
    return {
        "sources": sources,
        "python": artifact(sys.executable),
        # Preserve the venv invocation path separately from physical interpreter bytes.
        "python_invocation": str(Path(sys.executable).absolute()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "venv_configuration": artifact(Path(sys.prefix) / "pyvenv.cfg")
        if (Path(sys.prefix) / "pyvenv.cfg").is_file()
        else None,
        "python_version": sys.version,
        "packages": packages,
    }


def experiment_for_plan(plan):
    """Field order exactly follows jope_pipeline::experiment::Experiment serialization."""
    checkpoints = {
        f"{task}:{key}": value["sha256"]
        for task, category in plan["category_inputs"].items()
        for key, value in category["inputs"].items()
    }
    return {
        "schema": "jope.experiment/v1",
        "panel_sha256": plan["plan_sha256"],
        "methods_sha256": digest(plan["runtime"]),
        "source_sha": plan["source_sha"],
        "checkpoints": dict(sorted(checkpoints.items())),
        "seed": plan["seed"],
        "resources": {
            "workers": 1,
            "cpus": 20,
            "memory_bytes": 64 * 1024**3,
            "swap_bytes": 0,
            "gpu_tasks": int(plan["device"] == "cuda:0"),
            "threads_per_worker": 1,
            "total_seconds": 3600,
            "reporting_seconds": 5,
            "per_fit_seconds": 1200,
        },
        "expected_artifacts": [f"{task}/head.json" for task in plan["tasks"]],
        "protocol": "asm-training-v1:category=1200:total=3600:continuation-required",
        "numerical_policy": NUMERICS,
    }


def experiment_identity(experiment):
    return hashlib.sha256(
        json.dumps(experiment, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def validate_context(row, record, encoder_sha, data_kind):
    require(
        record["task"] == row["metadata"]["task_type"] and record["dataset_id"] == row["dataset_id"],
        "context task/dataset differs from training row",
    )
    require(record["encoder_sha256"] == encoder_sha, "context encoder identity changed")
    require(
        record["partition"] == "fit" and re.fullmatch(r"[0-9a-f]{64}", record["fitting_rows_sha256"]),
        "context lacks train-only row identity",
    )
    values = row["embedding"] + row["target_embedding"]
    require(
        len(row["embedding"]) in (512, 1536) and len(row["target_embedding"]) == 512,
        "context dimensions must be exact; padding/truncation is forbidden",
    )
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in values),
        "nonfinite context",
    )
    if record.get("context_encoding") == "f32le/v1":
        require(data_kind == "admitted", "f32LE identity is for real admitted contexts")
        for key, expected in (
            ("embedding", "embedding_f32le_sha256"),
            ("target_embedding", "target_embedding_f32le_sha256"),
        ):
            try:
                packed = struct.pack(f"<{len(row[key])}f", *row[key])
            except OverflowError as error:
                raise ValueError("context exceeds finite f32 range") from error
            require(
                list(struct.unpack(f"<{len(row[key])}f", packed)) == row[key],
                "context is not exact finite f32",
            )
            require(
                hashlib.sha256(packed).hexdigest() == record[expected],
                "row context f32 bytes changed",
            )
    else:
        require(record.get("context_encoding") is None, "unknown context encoding")
        require(
            record["embedding_sha256"] == digest(row["embedding"])
            and record["target_embedding_sha256"] == digest(row["target_embedding"]),
            "row context changed",
        )
    if data_kind == "synthetic_fixture":
        require(
            record["lineage"] == "asm.synthetic-training-context/v1",
            "synthetic context mislabeled",
        )
        return
    context = read(checked(record["context_artifact"]))
    producer = read(checked(record["producer_receipt"]))
    require(
        producer["output_sha256"] == record["context_artifact"]["sha256"]
        and producer["fitting_rows_sha256"] == record["fitting_rows_sha256"]
        and producer["encoder_sha256"] == encoder_sha,
        "encoder producer receipt differs",
    )
    for ref in producer["inputs"]:
        checked(ref)
    require(producer["inputs"], "encoder input closure is absent")
    require(context["values"] == values, "context artifact differs from corpus vectors")
    expected_lineage = (
        "asm.experimental-regression-context/v1" if record["task"] == "regression" else CLASS_CONTEXT
    )
    require(
        context["lineage"] == expected_lineage
        and context["policy_compatible"] is False
        and record["lineage"] == expected_lineage
        and len(row["embedding"]) == 1536,
        "real training requires explicit experimental2048 context lineage",
    )
    require(
        producer["training"] == context["training"],
        "context producer training closure differs",
    )
    packed = struct.pack("<2048f", *values)
    require(
        hashlib.sha256(packed).hexdigest() == context["context_sha256"],
        "context vector digest differs",
    )
    if record["task"] != "regression":
        require(
            type(context["classes"]) is int
            and (context["classes"] == 2 if record["task"] == "binary" else context["classes"] >= 3),
            "classification partition differs from task",
        )
    require(
        context["training"]["selected_training_rows_sha256"] == record["fitting_rows_sha256"],
        "context selected row closure differs",
    )


class CategoryUnavailable(ValueError):
    """An intact admitted manifest has no usable rows for this category."""


def admitted_rows(inputs, data_kind, selected_task=None):
    projected = None
    if data_kind == "admitted":
        from .training_input_closure import admit

        projected = admit(inputs)
    corpus = checked(inputs["corpus"])
    contexts = read(checked(inputs["contexts"]))
    checked(inputs["encoder"])
    vocabulary = read(checked(inputs["vocabulary"]))
    require(vocabulary.get("engines"), "frozen grammar vocabulary is required")
    if "checkpoint" in inputs:
        checked(inputs["checkpoint"])
    require(
        contexts["schema"] == "jope.training-context-index/v1",
        "unsupported context index",
    )
    eligibility = {}
    used_eligibility = set()
    if data_kind == "admitted":
        manifest = read(checked(inputs["admission"]))
        require(
            manifest["schema"] == "jope.qualified-teaching-freeze/v1",
            "measured teaching manifest required",
        )
        eligibility = {
            (
                entry["comparison"]["task"],
                entry["comparison"]["dataset_hash"],
                entry["program_sha256"],
            ): entry
            for entry in manifest["entries"]
        }
    rows = []
    families = {}
    datasets = {}
    dataset_contexts = {}
    used_contexts = set()
    for line in corpus.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata", {})
        task, split = metadata.get("task_type"), metadata.get("split")
        require(
            task in TASKS and split in ("train", "development"),
            "unknown task or report-only labels in corpus",
        )
        require(metadata.get("source_family"), "source family identity required")
        for key in (
            "authority_split",
            "dataset_split",
            "evidence_split",
            "source_split",
        ):
            require(
                metadata.get(key, split) in ("train", "development"),
                "report-only authority in corpus",
            )
        family_key = metadata["source_family"]
        require(
            families.setdefault(family_key, split) == split,
            "source family crosses train/development",
        )
        require(
            datasets.setdefault(row["dataset_id"], split) == split,
            "dataset crosses train/development",
        )
        record = contexts["contexts"].get(metadata.get("context_id"))
        require(record is not None, "context receipt absent")
        used_contexts.add(metadata["context_id"])
        validate_context(row, record, inputs["encoder"]["sha256"], data_kind)
        context_key = digest([row["embedding"], row["target_embedding"], record["fitting_rows_sha256"]])
        require(
            dataset_contexts.setdefault((row["dataset_id"], record["fitting_rows_sha256"]), context_key)
            == context_key,
            "one dataset has incompatible row contexts",
        )
        if data_kind == "admitted":
            recipe_bound = "teaching_recipe" in metadata
            program = (
                metadata.get("executable_program_sha256")
                if recipe_bound
                else hashlib.sha256(row["genome"].encode()).hexdigest()
            )
            key = (task, row["dataset_id"], program)
            require(key in eligibility, "row lacks measured teaching admission")
            entry = eligibility[key]
            used_eligibility.add(key)
            require(
                entry["comparison"]["vocabulary_sha256"] == inputs["vocabulary"]["sha256"],
                "teaching vocabulary differs from frozen comparison",
            )
            if recipe_bound:
                from .training_corpus_contract import validate_teaching_recipe

                validate_teaching_recipe(row, entry, vocabulary)
            require(
                row.get("canonical_genome") in (None, row["genome"]),
                "unadmitted canonical genome override",
            )
            require(
                record.get("comparison_sha256") == entry["freeze"]["comparison_sha256"],
                "context comparison differs from measured teaching evidence",
            )
            metadata["reward_pair_identity"] = entry["freeze"]["comparison_sha256"]
            require(
                entry["label_role"] == "development" and math.isfinite(entry["candidate_loss"]),
                "teaching row is not finite development evidence",
            )
            # The trainer's ranking labels are greater-is-better; preserve the original loss separately.
            metadata["measured_oof_loss"] = entry["candidate_loss"]
            metadata["score_robust"] = -entry["candidate_loss"]
        else:
            require(
                row["source"] == "asm-synthetic-training-fixture",
                "synthetic source mislabeled",
            )
        row["metadata"] = metadata
        rows.append(row)
    require(
        used_contexts == set(contexts["contexts"]), "context inventory contains missing or unused entries"
    )
    if projected is not None:
        require(
            used_contexts == set(projected["contexts"]) and families == projected["families"],
            "corpus differs from projected family/context roles",
        )
        require(used_eligibility == set(eligibility), "unused teaching entries in projected admission")
    if selected_task is not None:
        rows = [row for row in rows if row["metadata"]["task_type"] == selected_task]
    if not rows:
        raise CategoryUnavailable("no admitted training rows")
    for task in {row["metadata"]["task_type"] for row in rows}:
        require(
            len({len(row["embedding"]) for row in rows if row["metadata"]["task_type"] == task}) == 1,
            "category has heterogeneous context dimensions; dropping rows is forbidden",
        )
    return rows


def synthetic_inputs(state, tasks):
    """Explicit fixtures; no corpus quality, real encoding, or serving claim."""
    from ..grammars import all_grammars
    from ..grammars.vocab import build_active_vocab_v3

    root = state / "synthetic-inputs"
    root.mkdir(parents=True, exist_ok=True)
    write(
        root / "encoder.json",
        {"schema": "asm.synthetic-encoder-fixture/v1"},
        immutable=True,
    )
    encoder = artifact(root / "encoder.json")
    rows, contexts = [], {}
    for task in tasks:
        head = {
            "regression": "ridge",
            "binary": "logit_ridge",
            "multiclass": "multiclass_ridge",
        }[task]
        for split in ("train", "development"):
            for index in range(4):
                identity = f"synthetic-{task}-{split}-{index // 2}"
                context_id = f"{identity}-{index}"
                embedding = [float((index // 2) % 3)] * 1536
                # Synthetic classifier contexts keep every target dimension live.
                # No original label value enters this deterministic fixture encoding.
                target = (
                    [1.0, 0.5] + [0.0] * 510
                    if task == "regression"
                    else [0.125 if column % 2 == 0 else 0.0625 for column in range(512)]
                )
                rows.append(
                    {
                        "dataset_id": identity,
                        "source": "asm-synthetic-training-fixture",
                        "embedding": embedding,
                        "target_embedding": target,
                        "genome": f"(pipe X ({head}))",
                        "score": float(index),
                        "metric": "synthetic-ranking-signal",
                        "engine": "featmath",
                        "metadata": {
                            "task_type": task,
                            "split": split,
                            "source_family": identity,
                            "context_id": context_id,
                            "provenance": "corpus",
                        },
                    }
                )
                contexts[context_id] = {
                    "task": task,
                    "dataset_id": identity,
                    "encoder_sha256": encoder["sha256"],
                    "partition": "fit",
                    "fitting_rows_sha256": digest([0, 1]),
                    "embedding_sha256": digest(embedding),
                    "target_embedding_sha256": digest(target),
                    "lineage": "asm.synthetic-training-context/v1",
                }
    corpus = root / "corpus.jsonl"
    data = b"".join(encoded(row) + b"\n" for row in rows)
    if corpus.exists():
        require(corpus.read_bytes() == data, "synthetic corpus changed")
    else:
        with corpus.open("xb") as stream:
            stream.write(data)
    write(
        root / "contexts.json",
        {"schema": "jope.training-context-index/v1", "contexts": contexts},
        immutable=True,
    )
    write(
        root / "active_vocab.json",
        build_active_vocab_v3([], all_grammars()),
        immutable=True,
    )
    return {
        "corpus": artifact(corpus),
        "contexts": artifact(root / "contexts.json"),
        "encoder": encoder,
        "vocabulary": artifact(root / "active_vocab.json"),
    }


def make_plan(
    state,
    *,
    manifest=None,
    task="all-admitted",
    synthetic=False,
    device="cuda:0",
    source_sha,
    seed=7,
    passes=2,
):
    require(re.fullmatch(r"[0-9a-f]{40}", source_sha), "full source commit required")
    require(
        device in ("cuda:0", "cpu") and (device != "cpu" or synthetic),
        "CPU is explicit synthetic qualification only",
    )
    require(
        type(passes) is int and 2 <= passes <= 1000,
        "at least two passes required for continuation proof",
    )
    require(task in (*TASKS, "all-admitted"), "unsupported task scope")
    require(
        synthetic != (manifest is not None),
        "select either admitted inputs or explicit synthetic fixtures",
    )
    state = durable_state(state)
    requested = list(TASKS) if task == "all-admitted" else [task]
    inputs = synthetic_inputs(state, requested) if synthetic else read(manifest)
    if not synthetic:
        require(
            inputs.pop("schema", None) == "jope.experimental-training-inputs/v1",
            "unsupported input manifest",
        )
    require(
        type(seed) is int and 0 <= seed < 2**32,
        "seed must fit the NumPy unsigned32 seed contract",
    )
    category_inputs, counts, fallbacks, unavailable = {}, {}, {}, {}
    for category in requested:
        kind = "synthetic_fixture" if synthetic else "admitted"
        selected_inputs = inputs
        try:
            rows = admitted_rows(selected_inputs, kind, category)
            count = {
                split: sum(row["metadata"]["split"] == split for row in rows)
                for split in ("train", "development")
            }
            if not all(count.values()):
                raise CategoryUnavailable("category lacks disjoint train and development rows")
        except CategoryUnavailable as error:
            if not synthetic and category == "regression" and task == "all-admitted":
                unavailable[category] = str(error)
                continue
            if synthetic or category == "regression":
                raise
            # User-approved fallback is a separately labelled category. No regression row is renamed.
            fallbacks[category] = str(error)
            kind = "synthetic_fixture"
            selected_inputs = synthetic_inputs(state / ("fallback-" + category), [category])
            rows = admitted_rows(selected_inputs, kind, category)
            count = {
                split: sum(row["metadata"]["split"] == split for row in rows)
                for split in ("train", "development")
            }
        category_inputs[category] = {"data_kind": kind, "inputs": selected_inputs}
        counts[category] = count
    selected = list(category_inputs)
    require(
        selected,
        "no category admitted and no authorized classification fallback available",
    )
    kinds = {item["data_kind"] for item in category_inputs.values()}
    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed_admitted_and_synthetic_fixture"
    plan = {
        "schema": PLAN,
        "lineage": LINEAGE,
        "data_kind": kind,
        "qualification_only": device == "cpu",
        "scope": task,
        "tasks": selected,
        "unavailable_tasks": [kind for kind in requested if kind not in selected],
        "device": device,
        "category_inputs": category_inputs,
        "fallbacks": fallbacks,
        "unavailable_reasons": unavailable,
        "requested_manifest": artifact(manifest) if manifest is not None else None,
        "counts": counts,
        "source_sha": source_sha,
        "seed": seed,
        "passes_per_category": passes,
        "runtime": runtime_identity(),
    }
    plan["plan_sha256"] = digest(plan)
    write(state / "plan.json", plan, immutable=True)
    write(state / "experiment.json", experiment_for_plan(plan), immutable=True)
    return plan


def verify_plan(state):
    state = durable_state(state)
    plan = read(state / "plan.json")
    body = dict(plan)
    identity = body.pop("plan_sha256")
    require(
        plan["schema"] == PLAN and plan["lineage"] == LINEAGE and digest(body) == identity,
        "training plan changed",
    )
    require(
        plan["runtime"] == runtime_identity(),
        "trainer/interpreter/dependencies changed since plan",
    )
    require(
        read(Path(state) / "experiment.json") == experiment_for_plan(plan),
        "training experiment changed",
    )
    if plan["requested_manifest"] is not None:
        checked(plan["requested_manifest"])
    rows = []
    for task, category in plan["category_inputs"].items():
        rows.extend(admitted_rows(category["inputs"], category["data_kind"], task))
    return plan, rows


def original_window(state, plan, supervisor_unit, *, admitting=True):
    state = durable_state(state)
    experiment = experiment_for_plan(plan)
    if plan["device"] == "cuda:0" and admitting:
        require(
            isinstance(supervisor_unit, str)
            and re.fullmatch(
                r"jope-asm-"
                + experiment_identity(experiment)
                + r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.service",
                supervisor_unit,
            )
            is not None,
            "CUDA training requires the native supervised jope train wrapper",
        )
        require(
            any(
                supervisor_unit in line.split(":", 2)[-1].split("/")
                for line in Path("/proc/self/cgroup").read_text().splitlines()
            ),
            "training process is outside supervisor cgroup",
        )
    elif admitting and plan["device"] == "cpu" and not (state / "window.json").exists():
        # This branch is explicitly CPU synthetic qualification and cannot claim GPU accounting.
        now = int(time.time() * 1000)
        write(
            state / "window.json",
            {
                "experiment_sha256": experiment_identity(experiment),
                "started_unix_ms": now,
                "deadline_unix_ms": now + 3600000,
            },
            immutable=True,
        )
    window = read(state / "window.json")
    require(
        window["experiment_sha256"] == experiment_identity(experiment)
        and window["deadline_unix_ms"] == window["started_unix_ms"] + 3600000,
        "original training window changed",
    )
    require(
        int(time.time() * 1000) >= window["started_unix_ms"],
        "clock precedes original admission",
    )
    if admitting:
        write(state / "training-window.json", window, immutable=True)
    else:
        require(
            read(state / "training-window.json") == window,
            "original training window differs",
        )
    return window


def category_window(state, task, window, now=None, *, admitting=True):
    now = int(time.time() * 1000) if now is None else now
    path = Path(state) / task / "window.json"
    if path.exists():
        category = read(path)
        require(
            category["global_window_sha256"] == digest(window) and category["task"] == task,
            "category resume changed its original global window",
        )
        require(
            window["started_unix_ms"] <= category["started_unix_ms"] <= now,
            "clock precedes category admission",
        )
        require(
            category["deadline_unix_ms"]
            == min(category["started_unix_ms"] + 1200000, window["deadline_unix_ms"] - 5000),
            "category deadline reset",
        )
        return category
    require(admitting, "category has no original admission window")
    require(
        window["started_unix_ms"] <= now < window["deadline_unix_ms"] - 5000,
        "global GPU window exhausted",
    )
    category = {
        "task": task,
        "global_window_sha256": digest(window),
        "started_unix_ms": now,
        "deadline_unix_ms": min(now + 1200000, window["deadline_unix_ms"] - 5000),
    }
    write(path, category, immutable=True)
    return category


def checked_device(device):
    import torch

    if device == "cuda:0":
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "training requires exactly one visible CUDA GPU; no CPU fallback",
        )
        require(
            "RTX 3090" in torch.cuda.get_device_name(0),
            "training requires the pinned RTX3090",
        )
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        require(gpu_uuid not in ("", "None"), "CUDA GPU identity unavailable")
    else:
        require(device == "cpu", "unsupported device")
    torch.set_num_threads(1)
    return {
        "requested": device,
        "resolved": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda:0" else None,
        "gpu_uuid": gpu_uuid if device == "cuda:0" else None,
    }


def verify_checkpoint(ref, plan, task, previous=None):
    import torch

    from .model_v3 import load_policy_v3
    from .train_v3 import verify_training_proof

    payload = torch.load(checked(ref), map_location="cpu", weights_only=False)
    binding = payload.get("asm_training")
    require(
        binding
        == {
            "plan_sha256": plan["plan_sha256"],
            "lineage": LINEAGE,
            "task": task,
            "data_kind": plan["category_inputs"][task]["data_kind"],
        },
        "checkpoint context/corpus/campaign binding changed",
    )
    proof = payload["training_proof"]
    verify_training_proof(load_policy_v3(payload), proof)
    require(
        proof["published_changed_from_initial"] and not proof["published_inherited"],
        "campaign proof requires the published parameters to change",
    )
    require(payload["training_contract"]["tasks"] == [task], "checkpoint task changed")
    if previous is not None:
        require(
            proof["optimizer_state_resumed"]
            and proof["initial_parameter_sha256"] == previous["parameter_sha256"]
            and proof["optimizer_updates_total"]
            == previous["optimizer_updates_total"] + proof["optimizer_updates_this_run"],
            "optimizer/model continuation was not proved",
        )
    return proof


def native_invocation():
    path = os.environ.get("JOPE_NATIVE_INVOCATION")
    return artifact(path) if path else None


def stopped_attempt(state, attempt, started):
    """A missing controller outcome needs an authenticated native drain observation."""
    if (attempt / "rejection.json").exists():
        return None
    path = attempt / "outcome.json"
    if path.exists():
        outcome = read(path)
        require(
            outcome.get("status") in {"completed", "cancelled", "failed", "budget_exhausted"},
            "prior attempt has no recoverable stopped outcome",
        )
        return artifact(path)
    invocation = started.get("native_invocation")
    require(invocation is not None, "prior worker has no stopped outcome or native invocation")
    request_path = checked(invocation)
    request = read(request_path)
    require(
        request.get("schema") == "jope.training-child/v1"
        and request["state"] == str(Path(state).resolve())
        and request["native_plan_sha256"] == sha(Path(state) / "native-plan.json"),
        "recovery native invocation differs",
    )
    require(
        request_path == Path(state) / "native-invocations" / request["nonce"] / "request.json",
        "recovery invocation is outside controller custody",
    )
    drain = artifact(request_path.parent / "drained.json")
    observed = read(checked(drain))
    require(
        observed.get("schema") == "jope.training-drained/v1"
        and observed.get("populated") is False
        and observed.get("request_sha256") == invocation["sha256"]
        and observed["ended_unix_ms"] >= started["started_unix_ms"],
        "native stopped proof differs",
    )
    recovery = {
        "schema": "jope.training-recovery-stop/v1",
        "status": "interrupted",
        "native_invocation": invocation,
        "native_drain": drain,
        "stopped_observed_unix_ms": observed["ended_unix_ms"],
        "wall_seconds": None,
        "cpu_seconds": None,
        "gpu_seconds": None,
        "accounting_reason": "controller outcome missing; native drain proves stopped, elapsed usage unknown",
    }
    path = attempt / "recovery-stop.json"
    write(path, recovery, immutable=True)
    return artifact(path)


def interrupted_resume(state, task, attempt, binding, expected_epoch):
    """Recover only a stopped attempt of this exact pass and original window."""
    import torch

    for prior in sorted((Path(state) / task).glob("attempt-*"), reverse=True):
        if prior >= Path(attempt) or not (prior / "started.json").is_file():
            continue
        if (prior / "rejection.json").exists():
            continue
        started = read(prior / "started.json")
        require(
            started["plan_sha256"] == binding["plan_sha256"] and started["task"] == task,
            "prior attempt belongs to another plan/task",
        )
        for path in (prior / "resume.pt", prior / "resume.updates.pt"):
            if not path.exists():
                continue
            ref = artifact(path)
            payload = torch.load(checked(ref), map_location="cpu", weights_only=False)
            if payload.get("run_binding") != binding:
                continue  # An earlier, already committed pass has a different parent.
            progress = payload.get("strict_progress")
            if progress is None:
                continue
            require(progress["sampler"]["epoch"] == expected_epoch, "interrupted pass epoch changed")
            stopped = stopped_attempt(state, prior, started)
            if stopped is None:
                continue
            require(
                started["window_sha256"] == binding["window_sha256"]
                and started["category_window_sha256"] == binding["category_window_sha256"],
                "interrupted pass original window differs",
            )
            return {
                "checkpoint": ref,
                "started": artifact(prior / "started.json"),
                "outcome": stopped,
            }
    return None


def worker(state, task, attempt):
    import torch

    from .train_v3 import _atomic_torch_save, train_policy_v3_from_jsonl

    state, attempt = Path(state), Path(attempt)
    plan, rows = verify_plan(state)
    window = original_window(state, plan, os.environ.get("JOPE_SUPERVISOR_UNIT"))
    category = category_window(state, task, window)
    require(
        int(time.time() * 1000) < category["deadline_unix_ms"],
        "original category window exhausted",
    )
    device = checked_device(plan["device"])
    write(state / "device.json", device, immutable=True)
    category_inputs = plan["category_inputs"][task]
    inputs = category_inputs["inputs"]
    rows = [row for row in rows if row["metadata"]["task_type"] == task]
    require(task in plan["tasks"], "unplanned training task")
    parent_path = state / task / "head.json"
    parent = read_head(parent_path) if parent_path.exists() else None
    require(
        parent is None or parent["device"] == device,
        "resume CUDA device identity changed",
    )
    previous = verify_checkpoint(parent["resume"], plan, task) if parent else None
    run_binding = {
        "plan_sha256": plan["plan_sha256"],
        "task": task,
        "parent_receipt_sha256": parent["receipt"]["sha256"] if parent else None,
        "window_sha256": digest(window),
        "category_window_sha256": digest(category),
    }
    recovered = interrupted_resume(
        state, task, attempt, run_binding, (parent["global_epoch"] if parent else 0) + 1
    )
    resume_ref = recovered["checkpoint"] if recovered else (parent["resume"] if parent else None)
    corpus = attempt / "corpus.jsonl"
    with corpus.open("xb") as stream:
        stream.write(b"".join(encoded(row) + b"\n" for row in rows))
    shutil.copyfile(checked(inputs["vocabulary"]), attempt / "active_vocab.json")
    checkpoint, resume = attempt / "policy.pt", attempt / "resume.pt"
    tiny = category_inputs["data_kind"] == "synthetic_fixture"
    result = train_policy_v3_from_jsonl(
        corpus,
        checkpoint,
        epochs=1,
        batch_size=2 if tiny else 256,
        device=plan["device"],
        lr=5e-4,
        seed=plan["seed"],
        phase_a_fraction=0.0,
        checkpoint_state="current",
        greedy_eval_datasets=0,
        amp=False,
        autotune_batch_size=False,
        d_model=8 if tiny else 256,
        n_layers=1 if tiny else 2,
        n_heads=2 if tiny else 4,
        ffn_dim=16 if tiny else 1024,
        training_scope=f"category:{task}",
        strict_resume=True,
        resume_from=checked(resume_ref) if resume_ref else None,
        init_from=checked(inputs["checkpoint"]) if not resume_ref and "checkpoint" in inputs else None,
        checkpoint_metadata=run_binding,
        resume_checkpoint=resume,
        metrics_jsonl=attempt / "epochs.jsonl",
        progress_jsonl=attempt / "progress.json",
        resume_extra={
            "plan_sha256": plan["plan_sha256"],
            "context_sha256": inputs["contexts"]["sha256"],
        },
    )
    binding = {
        "plan_sha256": plan["plan_sha256"],
        "lineage": LINEAGE,
        "task": task,
        "data_kind": category_inputs["data_kind"],
    }
    for path in (checkpoint, resume):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["asm_training"] = binding
        _atomic_torch_save(payload, path)
    proof = verify_checkpoint(artifact(checkpoint), plan, task, previous)
    require(
        verify_checkpoint(artifact(resume), plan, task, previous) == proof,
        "resume/export reload differs",
    )
    if parent:
        before = torch.load(checked(parent["resume"]), map_location="cpu", weights_only=False)["optimizer"][
            "state"
        ]
        after = torch.load(resume, map_location="cpu", weights_only=False)["optimizer"]["state"]
        steps = [(key, float(value["step"])) for key, value in before.items() if "step" in value]
        require(
            steps
            and all(key in after and float(after[key]["step"]) >= value for key, value in steps)
            and any(float(after[key]["step"]) > value for key, value in steps),
            "optimizer step counters did not continue",
        )
    receipt = {
        "schema": "jope.training-pass/v1",
        "plan_sha256": plan["plan_sha256"],
        "task": task,
        "device": device,
        "checkpoint": artifact(checkpoint),
        "resume": artifact(resume),
        "training_proof": proof,
        "parent_receipt_sha256": parent["receipt"]["sha256"] if parent else None,
        "global_epoch": result["global_epoch"],
        "completed_unix_ms": int(time.time() * 1000),
        "recovered_from": recovered,
    }
    write(attempt / "complete.json", receipt, immutable=True)
    return receipt


def stop_group(child):
    # Cancellation is immediate at the category deadline; the enclosing service
    # independently kills all descendants if the controller itself dies.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


@contextlib.contextmanager
def cancellation_signals():
    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"training cancelled by signal {signum}")

    previous = {signum: signal.signal(signum, cancelled) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def read_head(path):
    head = read(path)
    receipt = read(checked(head["receipt"]))
    require(
        {key: value for key, value in head.items() if key not in ("receipt", "committed_passes")} == receipt,
        "head differs from its immutable receipt",
    )
    require(
        type(head["committed_passes"]) is int
        and head["committed_passes"] == head["global_epoch"]
        and head["committed_passes"] > 0,
        "head pass count changed",
    )
    return head


def launch_worker(arguments, env, attempt, deadline_unix_ms):
    started = time.monotonic()
    started_unix_ms = int(time.time() * 1000)
    child, status, reason, interrupted = None, "failed", None, False
    with (
        (attempt / "stdout.log").open("xb") as stdout,
        (attempt / "stderr.log").open("xb") as stderr,
    ):
        try:
            child = subprocess.Popen(arguments, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            remaining = (deadline_unix_ms - int(time.time() * 1000)) / 1000
            if remaining <= 0:
                status = "budget_exhausted"
            else:
                child.wait(timeout=remaining)
                status = "completed" if child.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "budget_exhausted"
        except OSError as error:
            status, reason = "unavailable", str(error)
        except KeyboardInterrupt:
            status, interrupted = "cancelled", True
        finally:
            if child is not None:
                stop_group(child)
    ended_unix_ms = int(time.time() * 1000)
    if status == "completed" and ended_unix_ms > deadline_unix_ms:
        status, reason = (
            "budget_exhausted",
            "observed process exit exceeded original category cutoff",
        )
    write(
        attempt / "outcome.json",
        {
            "status": status,
            "reason": reason,
            "exit_code": child.returncode if child is not None else None,
            "started_unix_ms": started_unix_ms,
            "ended_unix_ms": ended_unix_ms,
            "deadline_unix_ms": deadline_unix_ms,
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": None,
            "gpu_seconds": None,
            "accounting_reason": "wall reservation bounds GPU allocation; CPU/GPU active seconds unmeasured",
        },
        immutable=True,
    )
    if interrupted:
        raise KeyboardInterrupt("training cancelled; original deadlines retained")
    return status


def validate_execution_timing(started, outcome, complete, category, validated_unix_ms=None):
    times = [
        category["started_unix_ms"],
        started["started_unix_ms"],
        outcome["started_unix_ms"],
        complete["completed_unix_ms"],
        outcome["ended_unix_ms"],
        outcome["deadline_unix_ms"],
        category["deadline_unix_ms"],
    ]
    require(
        all(type(value) is int and value >= 0 for value in times),
        "invalid execution timestamp",
    )
    require(
        times == sorted(times),
        "observed execution exceeded or changed original category cutoff",
    )
    wall = outcome["wall_seconds"]
    require(
        type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
        "unknown/nonfinite execution wall time",
    )
    require(
        outcome["status"] == "completed" and outcome["exit_code"] == 0,
        "committed pass did not complete",
    )
    if validated_unix_ms is not None:
        require(
            type(validated_unix_ms) is int
            and outcome["ended_unix_ms"] <= validated_unix_ms <= category["deadline_unix_ms"],
            "controller validation exceeded original category cutoff",
        )


def requested_coverage(plan, failures):
    requested = list(TASKS) if plan["scope"] == "all-admitted" else [plan["scope"]]
    failed = {value["task"]: value for value in failures}
    categories = [
        {
            "task": task,
            "data_kind": plan["category_inputs"][task]["data_kind"],
            "status": failed.get(task, {}).get("status", "completed"),
            "fallback_reason": plan["fallbacks"].get(task),
        }
        for task in plan["tasks"]
    ]
    categories.extend(
        {
            "task": task,
            "data_kind": None,
            "status": "unavailable",
            "reason": plan["unavailable_reasons"].get(task),
        }
        for task in plan["unavailable_tasks"]
    )
    execution_success = not failures
    complete_requested = execution_success and not plan["unavailable_tasks"] and not plan["fallbacks"]
    return {
        "scope": plan["scope"],
        "requested_tasks": requested,
        "categories": categories,
        "fallbacks": plan["fallbacks"],
        "unavailable_tasks": plan["unavailable_tasks"],
        "unavailable_reasons": plan["unavailable_reasons"],
        "execution_success": execution_success,
        "requested_coverage_complete": complete_requested,
        "success": complete_requested,
    }


def run(state, *, resume=False, supervisor_unit=None):
    state = durable_state(state)
    state.mkdir(parents=True, exist_ok=True)
    with cancellation_signals(), (state / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, _ = verify_plan(state)
        require(
            resume == (state / "training-window.json").exists(),
            "use run for a fresh plan or resume for its original window",
        )
        window = original_window(state, plan, supervisor_unit)
        failures = []
        for task in plan["tasks"]:
            try:
                category = category_window(state, task, window)
            except ValueError as error:
                failures.append({"task": task, "status": "budget_exhausted", "reason": str(error)})
                continue
            root = state / task
            head_path = root / "head.json"
            while True:
                head = read_head(head_path) if head_path.exists() else None
                if head and head["committed_passes"] >= plan["passes_per_category"]:
                    break
                remaining = (category["deadline_unix_ms"] - int(time.time() * 1000)) / 1000
                # Reserve five seconds of the SAME category window for bounded
                # receipt/checkpoint hashing and durable commit, never an extra clock.
                if remaining <= 5:
                    failures.append({"task": task, "status": "budget_exhausted"})
                    break
                number = len(list(root.glob("attempt-*"))) + 1
                attempt = root / f"attempt-{number:06d}"
                attempt.mkdir()
                write(
                    attempt / "started.json",
                    {
                        "plan_sha256": plan["plan_sha256"],
                        "task": task,
                        "window_sha256": digest(window),
                        "category_window_sha256": digest(category),
                        "started_unix_ms": int(time.time() * 1000),
                        "native_invocation": native_invocation(),
                    },
                    immutable=True,
                )
                env = os.environ.copy()
                env["JOPE_SUPERVISOR_UNIT"] = supervisor_unit or ""
                env["PYTHONHASHSEED"] = str(plan["seed"])
                env.update(
                    {
                        name: "1"
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "WORLD_SIZE",
                        )
                    }
                )
                status = launch_worker(
                    [
                        sys.executable,
                        "-m",
                        "jope.policy.training_campaign",
                        "_worker",
                        "--state",
                        str(state),
                        "--task",
                        task,
                        "--attempt",
                        str(attempt),
                    ],
                    env,
                    attempt,
                    category["deadline_unix_ms"] - 5000,
                )
                if status != "completed":
                    failures.append({"task": task, "status": status, "attempt": str(attempt)})
                    break
                try:
                    receipt = read(attempt / "complete.json")
                    previous = head["training_proof"] if head else None
                    require(
                        receipt["parent_receipt_sha256"] == (head["receipt"]["sha256"] if head else None)
                        and receipt["global_epoch"] == (head["global_epoch"] if head else 0) + 1,
                        "training pass parent/epoch changed",
                    )
                    started = read(attempt / "started.json")
                    outcome = read(attempt / "outcome.json")
                    validate_execution_timing(started, outcome, receipt, category)
                    require(
                        receipt["plan_sha256"] == plan["plan_sha256"]
                        and receipt["task"] == task
                        and receipt["device"] == read(state / "device.json"),
                        "worker completion identity changed",
                    )
                    # worker() already reloaded BOTH published/resume states and
                    # proved optimizer continuation inside its supervised deadline.
                    # Authenticate those exact bytes here; offline verify replays
                    # the actual loads independently without extending training.
                    checked(receipt["checkpoint"], category["deadline_unix_ms"])
                    checked(receipt["resume"], category["deadline_unix_ms"])
                    proof = receipt["training_proof"]
                    require(
                        proof["published_changed_from_initial"]
                        and not proof["published_inherited"]
                        and proof["optimizer_updates_this_run"] > 0,
                        "worker has no changed-parameter update proof",
                    )
                    if previous is not None:
                        require(
                            proof["optimizer_state_resumed"]
                            and proof["initial_parameter_sha256"] == previous["parameter_sha256"]
                            and proof["optimizer_updates_total"]
                            == previous["optimizer_updates_total"] + proof["optimizer_updates_this_run"],
                            "worker continuation proof differs",
                        )
                    validated = int(time.time() * 1000)
                    validate_execution_timing(started, outcome, receipt, category, validated)
                    committed = {
                        **receipt,
                        "validated_unix_ms": validated,
                        "execution": {
                            name: artifact(attempt / (name + ".json"))
                            for name in ("started", "outcome", "complete")
                        },
                    }
                    write(attempt / "committed.json", committed, immutable=True)
                    write(
                        head_path,
                        {
                            **committed,
                            "committed_passes": 1 + (head["committed_passes"] if head else 0),
                            "receipt": artifact(attempt / "committed.json"),
                        },
                    )
                except (ValueError, OSError, RuntimeError, KeyError) as error:
                    rejection = {
                        "task": task,
                        "status": "proof_rejected",
                        "reason": str(error),
                        "attempt": str(attempt),
                    }
                    write(attempt / "rejection.json", rejection, immutable=True)
                    failures.append(rejection)
                    break
        report = {
            "schema": "jope.training-run/v1",
            "plan_sha256": plan["plan_sha256"],
            "lineage": LINEAGE,
            "data_kind": plan["data_kind"],
            "qualification_only": plan["qualification_only"],
            "original_window": window,
            "failures": failures,
            "reported_unix_ms": int(time.time() * 1000),
            **requested_coverage(plan, failures),
        }
        write(state / "report.json", report)
        return report


def verify(state):
    state = Path(state)
    plan, _ = verify_plan(state)
    original_window(state, plan, None, admitting=False)
    results = []
    for task in plan["tasks"]:
        head = read_head(state / task / "head.json")
        require(head["committed_passes"] >= 2, "no optimizer continuation proof yet")
        category = category_window(state, task, read(state / "window.json"), admitting=False)
        by_sha = {sha(path): path for path in (state / task).glob("attempt-*/committed.json")}
        chain, cursor = [], head["receipt"]["sha256"]
        while cursor is not None:
            require(
                cursor in by_sha and cursor not in [value[0] for value in chain],
                "receipt chain missing/cyclic",
            )
            path = by_sha[cursor]
            receipt = read(path)
            if receipt.get("recovered_from") is not None:
                require(
                    set(receipt["recovered_from"]) == {"checkpoint", "started", "outcome"},
                    "interrupted recovery evidence incomplete",
                )
                for ref in receipt["recovered_from"].values():
                    checked(ref)
                recovery = read(checked(receipt["recovered_from"]["outcome"]))
                if recovery.get("schema") == "jope.training-recovery-stop/v1":
                    checked(recovery["native_invocation"])
                    checked(recovery["native_drain"])
            execution = {key: read(checked(ref)) for key, ref in receipt["execution"].items()}
            require(
                set(execution) == {"started", "outcome", "complete"},
                "execution receipts missing",
            )
            require(
                {
                    key: value
                    for key, value in receipt.items()
                    if key not in ("execution", "validated_unix_ms")
                }
                == execution["complete"],
                "controller receipt differs from worker completion",
            )
            started = execution["started"]
            require(
                started["plan_sha256"] == plan["plan_sha256"]
                and started["task"] == task
                and started["window_sha256"] == digest(read(state / "window.json"))
                and started["category_window_sha256"] == digest(category)
                and category["started_unix_ms"] <= started["started_unix_ms"] <= receipt["completed_unix_ms"],
                "execution start differs from original admission",
            )
            validate_execution_timing(
                started,
                execution["outcome"],
                receipt,
                category,
                receipt["validated_unix_ms"],
            )
            require(
                receipt["plan_sha256"] == plan["plan_sha256"]
                and receipt["device"] == read(state / "device.json")
                and receipt["task"] == task
                and receipt["completed_unix_ms"] <= category["deadline_unix_ms"],
                "receipt identity/deadline changed",
            )
            chain.append((cursor, receipt))
            cursor = receipt["parent_receipt_sha256"]
        require(
            len(chain) == head["committed_passes"],
            "committed pass count differs from chain",
        )
        previous = None
        for epoch, (_, receipt) in enumerate(reversed(chain), 1):
            require(receipt["global_epoch"] == epoch, "global epoch changed")
            proof = verify_checkpoint(receipt["checkpoint"], plan, task, previous)
            verify_checkpoint(receipt["resume"], plan, task, previous)
            previous = proof
        require(
            previous == head["training_proof"],
            "head differs from recorded continuation chain",
        )
        results.append(
            {
                "task": task,
                "data_kind": plan["category_inputs"][task]["data_kind"],
                "parameter_sha256": previous["parameter_sha256"],
                "optimizer_updates": previous["optimizer_updates_total"],
                "completed_passes": head["committed_passes"],
                "planned_passes": plan["passes_per_category"],
            }
        )
    return {
        "schema": "jope.training-verification/v1",
        "lineage": LINEAGE,
        "plan_sha256": plan["plan_sha256"],
        "data_kind": plan["data_kind"],
        "qualification_only": plan["qualification_only"],
        **requested_coverage(
            plan,
            [
                {"task": result["task"], "status": "incomplete"}
                for result in results
                if result["completed_passes"] < result["planned_passes"]
            ],
        ),
        "completed_plan": all(result["completed_passes"] >= result["planned_passes"] for result in results),
        "tasks": results,
    }


def export(state, output):
    state, output = Path(state), Path(output)
    proof = verify(state)
    require(not output.exists(), "export destination must be new")
    output.mkdir(parents=True)
    plan = read(state / "plan.json")
    for task in plan["tasks"]:
        head = read_head(state / task / "head.json")
        target = output / task
        target.mkdir()
        for name in ("checkpoint", "resume"):
            shutil.copyfile(
                checked(head[name]),
                target / ("policy.pt" if name == "checkpoint" else "resume.pt"),
            )
        inputs = plan["category_inputs"][task]["inputs"]
        for key in ("contexts", "encoder", "vocabulary"):
            shutil.copyfile(checked(inputs[key]), target / (key + Path(inputs[key]["path"]).suffix))
    write(output / "verification.json", proof, immutable=True)
    write(output / "plan.json", plan, immutable=True)
    write(
        output / "manifest.json",
        {
            "schema": "jope.experimental-training-export/v1",
            "lineage": LINEAGE,
            "serving_eligible": False,
            "artifacts": [
                {"path": str(path.relative_to(output)), "sha256": sha(path)}
                for path in sorted(output.rglob("*"))
                if path.is_file()
            ],
        },
        immutable=True,
    )
    return {"output": str(output), "verification": proof}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jope train")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--state", type=Path, required=True)
    plan.add_argument("--inputs", type=Path)
    plan.add_argument("--task", choices=(*TASKS, "all-admitted"), default="all-admitted")
    plan.add_argument("--synthetic-fixture", action="store_true")
    plan.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    plan.add_argument("--source-sha", required=True)
    plan.add_argument("--seed", type=int, default=7)
    plan.add_argument("--passes", type=int, default=2)
    comparison = sub.add_parser("compare-plan")
    comparison.add_argument("--state", type=Path, required=True)
    comparison.add_argument("--task", choices=TASKS, required=True)
    comparison.add_argument("--fresh-inputs", type=Path, required=True)
    comparison.add_argument("--fresh-preparation", type=Path)
    comparison.add_argument("--updates", type=int, required=True)
    for name in ("run", "resume", "compare", "export", "verify", "_worker"):
        command = sub.add_parser(name)
        command.add_argument("--state", type=Path, required=True)
        if name in ("run", "resume", "compare"):
            command.add_argument("--supervisor-unit")
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        if name == "_worker":
            command.add_argument("--task", choices=TASKS, required=True)
            command.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = make_plan(
                args.state,
                manifest=args.inputs,
                task=args.task,
                synthetic=args.synthetic_fixture,
                device=args.device,
                source_sha=args.source_sha,
                seed=args.seed,
                passes=args.passes,
            )
        elif args.command == "compare-plan":
            from .training_comparison import prepare

            result = prepare(args.state, args.task, args.fresh_inputs, args.updates, args.fresh_preparation)
        elif args.command == "compare":
            from .training_comparison import run as compare

            result = compare(args.state, args.supervisor_unit)
        elif args.command in ("run", "resume"):
            result = run(
                args.state,
                resume=args.command == "resume",
                supervisor_unit=args.supervisor_unit,
            )
        elif args.command == "_worker":
            result = worker(args.state, args.task, args.attempt)
        elif args.command == "verify":
            result = verify(args.state)
        else:
            result = export(args.state, args.output)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result.get("success", True) else 1
    except KeyboardInterrupt as error:
        print(
            json.dumps(
                {
                    "schema": "jope.training-error/v1",
                    "error": str(error),
                    "status": "cancelled",
                }
            ),
            file=sys.stderr,
        )
        return 130
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(
            json.dumps({"schema": "jope.training-error/v1", "error": str(error)}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
