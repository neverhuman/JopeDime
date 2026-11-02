"""Bounded input validation for private experimental proposals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from pathlib import Path

from . import training_campaign as campaign

REQUEST = "jope.experimental-proposal-request/v1"
OUTPUT = "jope.experimental-proposal-output/v1"
LINEAGE = "asm.experimental-python-policy-v3/proposals-v1"
NUMERICS = "torch-cpu-f32:threads=1:deterministic:task-conditioned-mean-reward/v1"
SELECTION = "mean-reward-desc-executable-sha256-first64/v1"
CONTEXTS = {
    "regression": "asm.experimental-regression-context/v1",
    "binary": campaign.CLASS_CONTEXT,
    "multiclass": campaign.CLASS_CONTEXT,
}
MAX_METADATA = 32 * 1024 * 1024
MAX_CHECKPOINT = 1024 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def shape(value, keys, name):
    require(isinstance(value, dict) and set(value) == set(keys), f"invalid {name} fields")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def artifact_bytes(ref, maximum=MAX_METADATA):
    shape(ref, ("path", "sha256"), "artifact")
    path = Path(ref["path"])
    require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
        "artifact path is not physical",
    )
    require(path.stat().st_size <= maximum, "artifact exceeds bounded input")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    require(
        len(data) <= maximum and sha_bytes(data) == ref["sha256"],
        "artifact bytes changed",
    )
    return data


def private_write(path, value):
    path = Path(path)
    require(
        path.is_absolute() and not path.resolve().is_relative_to(Path("/tmp")),
        "private output requires durable absolute path",
    )
    data = campaign.encoded(value) + b"\n"
    require(len(data) <= MAX_METADATA, "proposal output exceeds bounded size")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def validate_request(request):
    shape(
        request,
        (
            "schema",
            "task",
            "export_manifest",
            "checkpoint",
            "vocabulary",
            "context",
            "generic_bank",
            "canonicalizer",
            "canonicalizer_source_sha40",
            "runtime_sha256",
            "training",
            "data_schema",
            "neural_seed",
            "sampling",
        ),
        "experimental request",
    )
    require(
        request["schema"] == REQUEST and request["task"] in CONTEXTS,
        "experimental task/schema mismatch",
    )
    training = request["training"]
    shape(
        training,
        (
            "prepared_sha256",
            "training_sha256",
            "selected_training_rows_sha256",
            "preprocessing_sha256",
            "cut",
        ),
        "training closure",
    )
    for name in (
        "prepared_sha256",
        "training_sha256",
        "selected_training_rows_sha256",
        "preprocessing_sha256",
    ):
        value = training[name]
        require(
            isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "invalid training hash",
        )
    require(type(training["cut"]) is int and 0 <= training["cut"] < 5, "invalid cut")
    require(
        type(request["neural_seed"]) is int and 0 <= request["neural_seed"] < 2**64,
        "invalid neural seed",
    )
    runtime = request["runtime_sha256"]
    require(
        isinstance(runtime, str) and len(runtime) == 64 and all(c in "0123456789abcdef" for c in runtime),
        "invalid runtime hash",
    )
    source = request["canonicalizer_source_sha40"]
    require(
        isinstance(source, str) and len(source) == 40 and all(c in "0123456789abcdef" for c in source),
        "invalid canonicalizer source",
    )
    sampling = request["sampling"]
    shape(sampling, ("count", "temperature", "top_p", "max_len"), "sampling")
    require(
        type(sampling["count"]) is int and 64 <= sampling["count"] <= 512,
        "sample count outside bounded range",
    )
    require(
        type(sampling["max_len"]) is int and 8 <= sampling["max_len"] <= 120,
        "sequence length outside bounded range",
    )
    require(
        type(sampling["temperature"]) in (int, float)
        and math.isfinite(sampling["temperature"])
        and 0 < sampling["temperature"] <= 2,
        "invalid temperature",
    )
    require(
        type(sampling["top_p"]) in (int, float)
        and math.isfinite(sampling["top_p"])
        and 0 < sampling["top_p"] <= 1,
        "invalid top_p",
    )


def load_context(request):
    context = json.loads(artifact_bytes(request["context"]))
    shape(
        context,
        (
            "lineage",
            "training",
            "values",
            "context_sha256",
            "classes",
            "policy_compatible",
        ),
        "experimental context",
    )
    task = request["task"]
    require(
        context["lineage"] in (CONTEXTS[task], "asm.synthetic-training-context/v1")
        and context["policy_compatible"] is False,
        "context is not separate experimental lineage",
    )
    require(context["training"] == request["training"], "context training closure changed")
    classes = context["classes"]
    require(
        type(classes) is int
        and (classes == 0 if task == "regression" else classes == 2 if task == "binary" else classes >= 3),
        "context class count differs from task",
    )
    values = context["values"]
    require(
        isinstance(values, list) and len(values) == 2048,
        "context must be exact1536+512; no padding or truncation",
    )
    require(
        all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 32 for v in values),
        "context nonfinite or unclipped",
    )
    require(
        sha_bytes(struct.pack("<2048f", *values)) == context["context_sha256"],
        "context float32 bytes changed",
    )
    return context


def export_inputs(request):
    root = Path(request["export_manifest"]["path"]).parent
    manifest = json.loads(artifact_bytes(request["export_manifest"]))
    require(
        manifest["schema"] == "jope.experimental-training-export/v1"
        and manifest["lineage"] == campaign.LINEAGE
        and manifest["serving_eligible"] is False,
        "not an experimental training export",
    )
    entries = manifest["artifacts"]
    require(
        len({entry["path"] for entry in entries}) == len(entries),
        "duplicate export artifact",
    )
    by_path = {}
    for entry in entries:
        shape(entry, ("path", "sha256"), "export artifact")
        relative = Path(entry["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "export path escape",
        )
        by_path[entry["path"]] = {
            "path": str(root / relative),
            "sha256": entry["sha256"],
        }
    task = request["task"]
    require(
        by_path.get(f"{task}/policy.pt") == request["checkpoint"],
        "checkpoint differs from export",
    )
    vocabulary = request["vocabulary"]
    require(
        any(
            value == vocabulary and name.startswith(f"{task}/vocabulary.") for name, value in by_path.items()
        ),
        "vocabulary differs from export",
    )
    plan = json.loads(artifact_bytes(by_path["plan.json"]))
    verification_bytes = artifact_bytes(by_path["verification.json"])
    verification = json.loads(verification_bytes)
    require(
        plan["schema"] == campaign.PLAN
        and plan["lineage"] == campaign.LINEAGE
        and task in plan["tasks"]
        and plan["source_sha"] == request["canonicalizer_source_sha40"],
        "export plan task/lineage changed",
    )
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(
        campaign.digest(unsigned_plan) == plan["plan_sha256"],
        "export plan checksum changed",
    )
    require(
        verification["schema"] == "jope.training-verification/v1"
        and verification["lineage"] == campaign.LINEAGE
        and verification["plan_sha256"] == plan["plan_sha256"]
        and verification["completed_plan"] is True,
        "export has no completed training proof",
    )
    require(
        plan["category_inputs"][task]["inputs"]["vocabulary"]["sha256"] == vocabulary["sha256"],
        "training vocabulary changed",
    )
    records = [row for row in verification["tasks"] if row["task"] == task]
    require(
        len(records) == 1 and records[0]["completed_passes"] >= records[0]["planned_passes"] >= 2,
        "reload/continuation proof incomplete",
    )
    return plan, records[0], sha_bytes(verification_bytes)


