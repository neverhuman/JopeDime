#!/usr/bin/env python3
"""Maintainer-private illustration of the existing Python resume API.

Default is metadata-only. Numerical execution is diagnostic and requires an
external, separately admitted CPU/GPU/memory/deadline supervisor. This script
is not the pending jope train front door and proves no corpus admission.
Only trust resume files from the verified private training input closure:
the historical API deserializes Python/PyTorch objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or sha256(resolved) != expected:
        raise ValueError(f"input digest changed: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--settings-sha256", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--resume-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-diagnostic", action="store_true")
    args = parser.parse_args()
    settings_file = checked(args.settings, args.settings_sha256)
    settings = json.loads(settings_file.read_text())
    if settings.get("example_schema") != "manual.python-resume-settings/v1":
        raise ValueError("unknown illustrative settings format")
    required = {"batch_size", "lr", "seed", "phase_a_fraction", "checkpoint_metric",
                "greedy_eval_datasets", "d_model", "n_layers", "ffn_dim", "n_heads",
                "prime_vocab", "routing_classes"}
    kwargs = settings["kwargs"]
    if set(kwargs) != required:
        raise ValueError("supply every original compatibility setting; unknown fields refused")
    corpus = checked(args.corpus, args.corpus_sha256)
    resume = checked(args.resume, args.resume_sha256)
    # This checks a supplied closure inventory, not whether it is complete or eligible.
    closure = settings["audited_input_files"]
    if not isinstance(closure, list) or not closure:
        raise ValueError("supply a separately audited full input closure")
    for artifact in closure:
        checked(Path(artifact["path"]), artifact["sha256"])
    output = args.output.resolve()
    call = {
        "corpus": str(corpus), "checkpoint": str(output / "policy.pt"),
        "resume_from": str(resume), "resume_checkpoint": str(output / "resume.pt"),
        "metrics_jsonl": str(output / "metrics.jsonl"),
        "progress_jsonl": str(output / "progress.json"),
        "epochs": 1, "checkpoint_state": "current", "allow_train_val_fallback": False,
        "amp": False, "autotune_batch_size": False, "device": "cuda:0", **kwargs,
    }
    if not args.execute_diagnostic:
        print(json.dumps({"status": "metadata_only_unexecuted", "call": call,
                          "eligibility": "not_established_by_this_helper"}, indent=2))
        return
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("example permits one GPU process only")
    # Import only in the explicit numerical branch; --help/default never import Torch.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one usable visible CUDA GPU required; CPU fallback refused")
    if "3090" not in torch.cuda.get_device_name(0):
        raise ValueError("example expects the separately allocated RTX 3090")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "PrimePolicyTrainerResume.v1"
            or payload.get("optimizer_state_compatible_with_model") is not True
            or payload.get("resume_model_state_kind") != "current"
            or payload.get("last_checkpoint_state") != "current"
            or payload.get("training_schedule") is not None
            or payload.get("anti_overfit_experiment") is not None):
        raise ValueError("example only supports compatible current-state ordinary resumes")
    architecture = payload.get("architecture", {})
    if any(architecture.get(key) != kwargs[key]
           for key in ("d_model", "n_layers", "n_heads", "ffn_dim")):
        raise ValueError("original architecture differs from supplied settings")
    del payload
    # Exclusive directory creation refuses accidental overwrite or reusing a run.
    output.mkdir(parents=True, exist_ok=False)
    from jope.policy.train_v3 import train_policy_v3_from_jsonl
    result = train_policy_v3_from_jsonl(**call)
    with (output / "api-result.json").open("x") as handle:
        json.dump({"scope": "diagnostic_api_call_not_product_acceptance",
                   "result": result}, handle, indent=2)


if __name__ == "__main__":
    main()
