"""Private experimental Python proposals; never an accepted Rust policy receipt.

The parent owns context admission, generic-bank reconstruction and actual supervisor
receipts. This worker only emits claims, which become evidence after that admission.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from . import training_campaign as campaign
from .experimental_proposal_inputs import (
    CONTEXTS as CONTEXTS,
)
from .experimental_proposal_inputs import (
    LINEAGE as LINEAGE,
)
from .experimental_proposal_inputs import (
    MAX_CHECKPOINT as MAX_CHECKPOINT,
)
from .experimental_proposal_inputs import (
    MAX_METADATA as MAX_METADATA,
)
from .experimental_proposal_inputs import (
    NUMERICS as NUMERICS,
)
from .experimental_proposal_inputs import (
    OUTPUT as OUTPUT,
)
from .experimental_proposal_inputs import (
    REQUEST as REQUEST,
)
from .experimental_proposal_inputs import (
    SELECTION as SELECTION,
)
from .experimental_proposal_inputs import (
    artifact_bytes as artifact_bytes,
)
from .experimental_proposal_inputs import (
    export_inputs as export_inputs,
)
from .experimental_proposal_inputs import (
    load_context as load_context,
)
from .experimental_proposal_inputs import (
    private_write as private_write,
)
from .experimental_proposal_inputs import (
    require as require,
)
from .experimental_proposal_inputs import (
    sha_bytes as sha_bytes,
)
from .experimental_proposal_inputs import (
    shape as shape,
)
from .experimental_proposal_inputs import (
    validate_request as validate_request,
)
from .experimental_proposal_model import (
    ARCHITECTURE_FIELDS as ARCHITECTURE_FIELDS,
)
from .experimental_proposal_model import (
    architecture as architecture,
)


def strict_model(request, context, plan, record):
    import io

    import torch

    from ..grammars.vocab import grammars_from_vocab_v3 as grammars_from_vocab_v3
    from .model_v3 import (
        MODEL_VERSION as MODEL_VERSION,
    )
    from .model_v3 import (
        TASK_LANE as TASK_LANE,
    )
    from .model_v3 import (
        load_policy_v3 as load_policy_v3,
    )
    from .train_v3 import verify_training_proof as verify_training_proof

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
        from .training_corpus_contract import load_frozen_action_vocab as load_frozen_action_vocab

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

    from ..pipeline_text import format_pipeline_actions as format_pipeline_actions
    from .automaton import DecoderState as DecoderState
    from .decode import sample_sequences as sample_sequences
    from .tokens import decode_tokens as decode_tokens

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
