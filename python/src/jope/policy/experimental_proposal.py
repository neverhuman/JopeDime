"""Private experimental Python proposals; never an accepted Rust policy receipt.

The parent owns context admission, generic-bank reconstruction and actual supervisor
receipts. This worker only emits claims, which become evidence after that admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import time
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


ARCHITECTURE_FIELDS = (
    "model_version",
    "d_model",
    "n_layers",
    "n_heads",
    "ffn_dim",
    "dataset_dim",
    "target_dim",
    "reward_ensemble_k",
    "embedding_rows",
    "routing_classes",
    "fused_task_heads",
    "target_keep_dim",
    "split_reward_context",
    "input_block_norm",
    "dataset_block_rms",
    "target_block_rms",
    "context_schema",
    "task_head_schema",
)


def architecture(payload):
    # The exact checkpoint SHA pins these fields; refuse the general loader's
    # historical defaults or shape-compatible reinterpretation on this route.
    require(all(key in payload for key in ARCHITECTURE_FIELDS), "incomplete experimental architecture")
    for key in (
        "d_model",
        "n_layers",
        "n_heads",
        "ffn_dim",
        "dataset_dim",
        "target_dim",
        "reward_ensemble_k",
        "embedding_rows",
        "target_keep_dim",
    ):
        require(type(payload[key]) is int and payload[key] > 0, "invalid experimental architecture")
    require(
        type(payload["routing_classes"]) is int and payload["routing_classes"] >= 0,
        "invalid routing architecture",
    )
    for key in ("fused_task_heads", "split_reward_context", "input_block_norm"):
        require(type(payload[key]) is bool, "invalid experimental architecture switch")
    for key in ("dataset_block_rms", "target_block_rms"):
        require(
            type(payload[key]) in (int, float) and math.isfinite(payload[key]) and payload[key] > 0,
            "invalid experimental normalization architecture",
        )
    return {key: payload[key] for key in ARCHITECTURE_FIELDS}


def strict_model(request, context, plan, record):
    import io

    import torch

    from ..grammars.vocab import grammars_from_vocab_v3
    from .model_v3 import MODEL_VERSION, TASK_LANE, load_policy_v3
    from .train_v3 import verify_training_proof

    # policy.pt contains tensors and primitive metadata; optimizer/RNG resume.pt is refused.
    payload = torch.load(
        io.BytesIO(artifact_bytes(request["checkpoint"], MAX_CHECKPOINT)),
        map_location="cpu",
        weights_only=True,
    )
    task = request["task"]
    require(
        payload.get("artifact_role") == "experimental_policy_export"
        and payload["model_version"] == MODEL_VERSION,
        "not the experimental export architecture",
    )
    require(
        payload.get("asm_training")
        == {
            "plan_sha256": plan["plan_sha256"],
            "lineage": campaign.LINEAGE,
            "task": task,
            "data_kind": plan["category_inputs"][task]["data_kind"],
        },
        "checkpoint training binding changed",
    )
    require(
        payload["training_contract"]["tasks"] == [task],
        "checkpoint was not trained for this task",
    )
    require(
        payload["dataset_dim"] == 1536 and payload["target_dim"] == 512,
        "checkpoint context dimensions differ",
    )
    keep = payload["target_keep_dim"]
    require(
        type(keep) is int and 1 <= keep <= 512 and all(v == 0 for v in context["values"][1536 + keep :]),
        "context has signal in dropped target dimensions",
    )
    if task != "regression":
        require(
            payload["fused_task_heads"] is True
            and payload["task_head_schema"]["lanes"].get(task) == TASK_LANE[task],
            "classification task head unavailable",
        )
    vocabulary = json.loads(artifact_bytes(request["vocabulary"]))
    if vocabulary.get("schema") == "parallel.action_vocab.v1":
        from .training_corpus_contract import load_frozen_action_vocab

        views, table = load_frozen_action_vocab(vocabulary)
        require(
            payload.get("frozen_action_vocab") == vocabulary and payload["action_vocab"] == table.to_json(),
            "checkpoint full frozen action table differs",
        )
        expected_specs = [spec.to_json() for spec in views]
    else:
        require(payload.get("frozen_action_vocab") is None, "snapshot cannot replace a full action table")
        frozen = grammars_from_vocab_v3(vocabulary)
        expected_specs = [frozen[name].to_json() for name in sorted(frozen)]
    require(expected_specs == payload["grammar_specs"], "checkpoint grammar differs from frozen vocabulary")
    require(
        all(spec["engine"] != "prime" for spec in payload["grammar_specs"]),
        "Prime checkpoints need their separate loader",
    )
    expected_architecture = architecture(payload)
    model = load_policy_v3(payload)
    require(
        architecture(model.checkpoint_payload()) == expected_architecture,
        "loaded architecture differs from exact experimental checkpoint",
    )
    require(
        model.vocab.to_json() == payload["action_vocab"],
        "vocabulary append or reordering is forbidden",
    )
    require(
        task in model.vocab.task_ids and model.vocab.heads_for("featmath", task, generate_only=True),
        "task has no executable proposal grammar",
    )
    current = model.state_dict()
    require(
        set(current) == set(payload["model"]),
        "checkpoint has missing or extra state; no initialized head fallback",
    )
    for name, value in current.items():
        saved = payload["model"][name]
        require(
            value.shape == saved.shape and value.dtype == saved.dtype and torch.equal(value, saved),
            f"checkpoint state not loaded exactly: {name}",
        )
        require(
            not value.is_floating_point() or bool(torch.isfinite(value).all()),
            "checkpoint state nonfinite",
        )
    proof = payload["training_proof"]
    verify_training_proof(model, proof)
    require(
        proof["published_changed_from_initial"] is True
        and proof["published_inherited"] is False
        and proof["optimizer_state_resumed"] is True,
        "checkpoint lacks changed/reloaded/continued parameters",
    )
    require(
        proof["parameter_sha256"] == record["parameter_sha256"]
        and proof["optimizer_updates_total"] == record["optimizer_updates"],
        "export proof differs from actual checkpoint",
    )
    return model.eval(), proof


def load_bank(request):
    bank = json.loads(artifact_bytes(request["generic_bank"]))
    shape(bank, ("schema", "task", "policy", "seed", "candidates"), "generic bank")
    require(
        bank["schema"] == "jope.experimental-generic-bank/v1" and bank["task"] == request["task"],
        "generic bank task/schema changed",
    )
    require(
        isinstance(bank["policy"], str)
        and bank["policy"]
        and type(bank["seed"]) is int
        and 0 <= bank["seed"] < 2**64,
        "generic bank policy/seed absent",
    )
    require(
        80 <= len(bank["candidates"]) <= 4096 - request["sampling"]["count"],
        "generic bank size outside bounded range",
    )
    for row in bank["candidates"]:
        shape(row, ("seed", "genome"), "generic bank candidate")
        require(
            type(row["seed"]) is int
            and 0 <= row["seed"] < 2**64
            and isinstance(row["genome"], str)
            and 0 < len(row["genome"].encode()) <= 16384,
            "invalid generic bank candidate",
        )
    return bank


def canonicalize(request, programs, output_directory, remaining):
    artifact_bytes(request["canonicalizer"], 1024 * 1024 * 1024)
    payload = {
        "schema": "jope.experimental-canonicalization-request/v1",
        "task": request["task"],
        "data_schema": request["data_schema"],
        "programs": programs,
    }
    input_path, output_path = (
        output_directory / "canonicalization-input.json",
        output_directory / "canonicalization-output.json",
    )
    private_write(input_path, payload)
    argv = [
        request["canonicalizer"]["path"],
        "benchmark",
        "synthesized-validate",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(output_directory),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with (
        (output_directory / "canonicalizer.stdout").open("xb") as stdout,
        (output_directory / "canonicalizer.stderr").open("xb") as stderr,
    ):
        subprocess.run(
            argv,
            env=environment,
            cwd=output_directory,
            stdout=stdout,
            stderr=stderr,
            timeout=remaining(),
            check=True,
        )
    data = artifact_bytes(campaign.artifact(output_path))
    response = json.loads(data)
    require(
        response["schema"] == "jope.experimental-canonicalization-response/v1"
        and response["input_sha256"] == campaign.sha(input_path),
        "canonicalizer input binding changed",
    )
    require(len(response["results"]) == len(programs), "canonicalizer omitted results")
    for index, row in enumerate(response["results"]):
        require(
            row["index"] == index and row["original_bytes_sha256"] == sha_bytes(programs[index].encode()),
            "canonicalizer reordered/changed input",
        )
        recipe = row.get("recipe")
        if recipe is not None:
            require(
                recipe["original_proposal"] == programs[index]
                and recipe["original_bytes_sha256"] == row["original_bytes_sha256"],
                "canonicalizer original bytes changed",
            )
            for value, checksum in (
                ("proposal_canonical", "proposal_canonical_sha256"),
                ("executable_canonical", "executable_sha256"),
            ):
                require(
                    sha_bytes(recipe[value].encode()) == recipe[checksum],
                    "canonicalizer recipe checksum changed",
                )
    return response["results"], sha_bytes(data)


def assemble(neural, bank, results):
    """Rank actual neural candidates, fill from the pinned typed bank, retain every origin."""
    ranked = sorted(
        neural,
        key=lambda row: (
            -row["reward_mean"],
            results[row["program_index"]]["recipe"]["executable_sha256"],
        ),
    )
    selected, seen = [], set()
    for row in ranked:
        recipe = results[row["program_index"]]["recipe"]
        key = recipe["executable_sha256"]
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                **row,
                "recipe": recipe,
                "origin": "neural_policy",
                "bank_index": None,
                "candidate_seed": None,
            }
        )
        if len(selected) == 64:
            break
    neural_count = len(selected)
    if neural_count:
        maximum = selected[0]["reward_mean"]
        normalizer = maximum + math.log(sum(math.exp(row["reward_mean"] - maximum) for row in selected))
        for row in selected:
            row["rank_log_weight"] = row["reward_mean"] - normalizer
    offset = len(results) - len(bank["candidates"])
    for index, candidate in enumerate(bank["candidates"]):
        recipe = results[offset + index].get("recipe")
        if recipe is None or recipe["executable_sha256"] in seen:
            continue
        seen.add(recipe["executable_sha256"])
        selected.append(
            {
                "recipe": recipe,
                "origin": "neural_rejection_fallback" if len(selected) < 64 else "random_exploration",
                "rank_log_weight": None,
                "reward_mean": None,
                "reward_std": None,
                "bank_index": index,
                "candidate_seed": candidate["seed"],
                "program_index": offset + index,
                "tokens": None,
            }
        )
        if len(selected) == 80:
            break
    require(
        len(selected) == 80,
        "typed generic bank cannot fill80 unique slots; no substituted baseline",
    )
    for slot, row in enumerate(selected):
        row["slot"] = slot
    return selected, neural_count


def run(request, output_path, *, request_sha256):
    import torch

    from ..pipeline_text import format_pipeline_actions
    from .automaton import DecoderState
    from .decode import sample_sequences
    from .tokens import decode_tokens

    started = time.monotonic()

    def remaining():
        seconds = 30 - (time.monotonic() - started)
        require(seconds > 0, "experimental proposal local ceiling exhausted")
        return seconds

    validate_request(request)
    require(
        isinstance(request_sha256, str)
        and len(request_sha256) == 64
        and all(c in "0123456789abcdef" for c in request_sha256),
        "invalid request artifact hash",
    )
    require(
        campaign.digest(campaign.runtime_identity()) == request["runtime_sha256"],
        "proposal runtime changed",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    context, bank = load_context(request), load_bank(request)
    plan, record, export_proof_sha = export_inputs(request)
    model, proof = strict_model(request, context, plan, record)
    remaining()
    embedding = torch.tensor(context["values"][:1536], dtype=torch.float32, device="cpu")
    target = torch.tensor(context["values"][1536:], dtype=torch.float32, device="cpu")
    sampling = request["sampling"]
    sequences = sample_sequences(
        model,
        embedding,
        target,
        n=sampling["count"],
        engine="featmath",
        task=request["task"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        seed=request["neural_seed"],
        max_len=sampling["max_len"],
        device="cpu",
    )
    programs, decoded, rejected = [], [], []
    for sample_index, sequence in enumerate(sequences):
        try:
            state = DecoderState(model.vocab)
            for token in sequence[1:]:
                state.advance(token)
            require(state.is_done(), "incomplete sampled sequence")
            engine, actions = decode_tokens(sequence, model.vocab)
            require(
                engine == "featmath" and actions["task"] == request["task"],
                "decoded task changed",
            )
            programs.append(format_pipeline_actions(actions))
            decoded.append(
                {
                    "sample_index": sample_index,
                    "program_index": len(programs) - 1,
                    "tokens": sequence,
                }
            )
        except (ValueError, IndexError, KeyError) as error:
            rejected.append({"sample_index": sample_index, "reason": str(error)})
    programs.extend(row["genome"] for row in bank["candidates"])
    results, canonicalization_sha = canonicalize(request, programs, Path(output_path).parent, remaining)
    valid = []
    for row in decoded:
        if results[row["program_index"]].get("recipe") is None:
            rejected.append(
                {
                    "sample_index": row["sample_index"],
                    "reason": results[row["program_index"]]["error"],
                }
            )
        else:
            valid.append(row)
    if valid:
        with torch.no_grad():
            mean, std = model.score_token_sequences(
                embedding,
                target,
                [row["tokens"] for row in valid],
                device="cpu",
                task=request["task"],
            )
        for row, value, uncertainty in zip(valid, mean.tolist(), std.tolist(), strict=True):
            require(
                math.isfinite(value) and math.isfinite(uncertainty),
                "neural ranking nonfinite",
            )
            row.update(reward_mean=value, reward_std=uncertainty)
    remaining()
    candidates, neural_count = assemble(valid, bank, results)
    response = {
        "schema": OUTPUT,
        "lineage": LINEAGE,
        "serving_eligible": False,
        "request_sha256": request_sha256,
        "request_identity_sha256": campaign.digest(request),
        "backend": "torch-cpu",
        "task": request["task"],
        "training": request["training"],
        "checkpoint_sha256": request["checkpoint"]["sha256"],
        "export_manifest_sha256": request["export_manifest"]["sha256"],
        "export_verification_sha256": export_proof_sha,
        "training_proof_sha256": campaign.digest(proof),
        "parameter_sha256": proof["parameter_sha256"],
        "checkpoint_training_data_kind": plan["category_inputs"][request["task"]]["data_kind"],
        "vocabulary_sha256": request["vocabulary"]["sha256"],
        "context_artifact_sha256": request["context"]["sha256"],
        "context_sha256": context["context_sha256"],
        "context_lineage": context["lineage"],
        "runtime_sha256": request["runtime_sha256"],
        "numerical_policy": NUMERICS,
        "selection_policy": SELECTION,
        "sampling": sampling,
        "neural_seed": request["neural_seed"],
        "generic_bank_sha256": request["generic_bank"]["sha256"],
        "generic_policy": bank["policy"],
        "generic_seed": bank["seed"],
        "canonicalizer": request["canonicalizer"],
        "canonicalizer_source_sha40": request["canonicalizer_source_sha40"],
        "canonicalization_sha256": canonicalization_sha,
        "neural_candidates": neural_count,
        "rejections": rejected,
        "candidates": candidates,
    }
    require(
        all(row["rank_log_weight"] is None or math.isfinite(row["rank_log_weight"]) for row in candidates),
        "rank weight nonfinite",
    )
    private_write(output_path, response)
    return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    request = json.loads(artifact_bytes({"path": str(args.request), "sha256": args.request_sha256}))
    run(request, args.output, request_sha256=args.request_sha256)


if __name__ == "__main__":
    main()
