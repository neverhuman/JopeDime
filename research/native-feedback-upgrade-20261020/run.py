"""Three controlled feedback waves on immutable S3 development outcomes.

This is an offline replay experiment, not a production trainer or promotion job.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import tarfile
import time

import numpy as np
import torch
from safetensors.torch import save_file

from feedback import (canonical, digest, FeedbackLedger, ValueAdapter, choose,
                      ensemble_predict, parameter_hash, train_adapter,
                      selection_metric, paired_interval)


def publish(path, value):
    with path.open("xb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint(path, models, optimizers, *, ledger, feature_receipt, config, wave):
    """Safe tensor child; named optimizer inventory retained without pickle."""
    tensors = {}
    groups = []
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        for key, value in model.state_dict().items():
            tensors[f"member{i}.model.{key}"] = value.detach().contiguous()
        names = {id(p): name for name, p in model.named_parameters()}
        group_record = []
        for group in optimizer.param_groups:
            record = {k: v for k, v in group.items() if k != "params"}
            record["params"] = [names[id(p)] for p in group["params"]]
            group_record.append(record)
        groups.append(group_record)
        for parameter, state in optimizer.state.items():
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError("unsupported non-tensor optimizer state")
                tensors[f"member{i}.optimizer.{names[id(parameter)]}.{key}"] = value.detach().contiguous()
    tensors["torch_cpu_rng"] = torch.get_rng_state()
    save_file(tensors, str(path.with_suffix(".safetensors")))
    metadata = {"schema": "jope.native-value-adapter-checkpoint/v1", "wave": wave,
                "architecture": {"input": config["feature_width"], "hidden": [192, 96], "residual": True},
                "parameters": [parameter_hash(m) for m in models], "optimizer": "AdamW",
                "optimizer_groups": groups, "scheduler": None, "training_seed_plan": config["ensemble_seeds"],
                "feature_receipt_sha256": digest(feature_receipt), "native_checkpoint_parameters": config["checkpoint_parameters"],
                "target_snapshot_sha256": ledger.identity(), "acquired_targets": len(ledger.targets),
                "state_sha256": hashlib.sha256(path.with_suffix(".safetensors").read_bytes()).hexdigest(),
                "resume": "Explicit adapter experiment checkpoint. No native planner/outbox recovery or production promotion."}
    publish(path.with_suffix(".json"), metadata)
    return metadata


def main(root):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = json.loads((root / "preregistration.json").read_text())
    source_root = Path(__file__).parent
    for name, expected in config["source_sha256"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("preregistered source drift: " + name)
    feature_receipt = json.loads((root / "feature-receipt.json").read_text())
    for name in ["features", "controls"]:
        path = root / (name + (".npz" if name == "features" else ".json"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != feature_receipt[name + "_sha256"]:
            raise ValueError("frozen feature/control bytes changed")
    config["feature_width"] = feature_receipt["feature_width"]
    raw = subprocess.check_output(["zstd", "-dc", str(root / "inputs.tar.zst")])
    if hashlib.sha256((root / "inputs.tar.zst").read_bytes()).hexdigest() != config["input_pins"]["inputs.tar.zst"]:
        raise ValueError("pool input changed")
    archive = tarfile.open(fileobj=io.BytesIO(raw))
    rows = [json.loads(line) for line in archive.extractfile("inputs/pools.jsonl")]
    rowmap = {r["context_id"]: r for r in rows}
    split = config["split"]
    train_ids = sorted(cid for cid, identity in split.items() if identity["partition"] == "train")
    assessment_ids = sorted(cid for cid, identity in split.items() if identity["partition"] == "assessment")
    arrays = np.load(root / "features.npz")
    # Labels for assessment are deliberately not passed to training/acquisition.
    train_features = {cid: torch.tensor(arrays[cid]) for cid in train_ids}
    assessment_features = {cid: torch.tensor(arrays[cid]) for cid in assessment_ids}
    train_rows = [rowmap[cid] for cid in train_ids]
    out = root / "experiment"
    out.mkdir(exist_ok=False)

    def guard():
        if time.monotonic() - started > config["limits"]["wall_seconds"]:
            raise RuntimeError("fixed experiment wall budget exhausted")
        # macOS ru_maxrss is bytes. Linux reports KiB.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname != "Darwin":
            rss *= 1024
        if rss > config["limits"]["memory_bytes"]:
            raise RuntimeError("experiment memory budget exceeded")

    initial = FeedbackLedger(split, train_rows)
    for cid in train_ids:
        rng = np.random.default_rng(int(digest(["initial", cid])[:16], 16))
        available = list(range(len(rowmap[cid]["members"])))
        for _ in range(config["initial_uniform_outcomes_per_train_context"]):
            member, propensity = choose(available, None, None, mode="uniform", rng=rng)
            initial.acquire(cid, member, wave=0, propensity=propensity)
            available.remove(member)
    models, optimizers = [], []
    initial_receipts = []
    for seed in config["ensemble_seeds"]:
        guard()
        torch.manual_seed(seed)
        model = ValueAdapter(config["feature_width"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        initial_receipts.append(train_adapter(model, optimizer, train_features, initial, steps=config["initial_updates"], seed=seed))
        models.append(model)
        optimizers.append(optimizer)
    initial_checkpoint = checkpoint(out / "initial", models, optimizers, ledger=initial,
                                    feature_receipt=feature_receipt, config=config, wave=0)
    with gzip.open(out / "initial-consumption.json.gz", "wb") as handle:
        handle.write(canonical(initial_receipts))
    publish(out / "initial-targets.json", [vars(t) for _, t in sorted(initial.targets.items())])
    print(json.dumps({"state": "INITIAL_ADAPTER_TRAINED", "updates_per_member": config["initial_updates"], "targets": len(initial.targets), "seconds": time.monotonic()-started}), flush=True)
    arms = {name: {"models": copy.deepcopy(models), "optimizers": None,
                   "ledger": copy.deepcopy(initial), "waves": []}
            for name in ["continuation", "uniform_feedback", "targeted_feedback"]}
    for arm in arms.values():
        arm["optimizers"] = []
        for model, original in zip(arm["models"], optimizers):
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            optimizer.load_state_dict(copy.deepcopy(original.state_dict()))
            arm["optimizers"].append(optimizer)
    # Every arm shares identical initial parameters AND optimizer state.
    for name, arm in arms.items():
        if [parameter_hash(m) for m in arm["models"]] != initial_checkpoint["parameters"]:
            raise ValueError("arm initialization mismatch")
    for wave in range(1, config["waves"] + 1):
        for name, arm in arms.items():
            guard()
            ledger = arm["ledger"]
            previous = ledger.identity()
            if name != "continuation":
                mode = "uniform" if name == "uniform_feedback" else "targeted"
                for cid in train_ids:
                    predictions, uncertainty = ensemble_predict(arm["models"], train_features[cid])
                    available = [i for i in range(len(rowmap[cid]["members"])) if (cid, i) not in ledger.targets]
                    rng = np.random.default_rng(int(digest(["acquire", wave, cid])[:16], 16))
                    for _ in range(min(config["new_outcomes_per_train_context_per_wave"], len(available))):
                        member, propensity = choose(available, predictions, uncertainty, mode=mode, rng=rng)
                        ledger.acquire(cid, member, wave=wave, propensity=propensity)
                        available.remove(member)
            receipts = []
            for seed, model, optimizer in zip(config["ensemble_seeds"], arm["models"], arm["optimizers"]):
                guard()
                receipts.append(train_adapter(model, optimizer, train_features, ledger,
                                              steps=config["updates_per_wave"], seed=seed + wave*1000))
            record = checkpoint(out / f"{name}-wave{wave}", arm["models"], arm["optimizers"], ledger=ledger,
                                feature_receipt=feature_receipt, config=config, wave=wave)
            record.update(parent_target_snapshot_sha256=previous,
                          targets_changed=ledger.identity() != previous,
                          updates_this_wave=config["updates_per_wave"] * len(models))
            arm["waves"].append(record)
            with gzip.open(out / f"{name}-wave{wave}-consumption.json.gz", "wb") as handle:
                handle.write(canonical(receipts))
            publish(out / f"{name}-wave{wave}-targets.json", [vars(t) for _, t in sorted(ledger.targets.items())])
            publish(out / f"{name}-wave{wave}-events.json", ledger.events)
            print(json.dumps({"state": "FEEDBACK_WAVE_COMPLETE", "arm": name, "wave": wave,
                              "targets": len(ledger.targets), "actual_adapter_updates": config["updates_per_wave"] * len(models),
                              "target_snapshot_sha256": ledger.identity(), "seconds": time.monotonic()-started}), flush=True)
    # No assessment outcomes have been used to choose a checkpoint, architecture,
    # objective coefficient or acquisition rule. Evaluate fixed final checkpoints.
    controls = json.loads((root / "controls.json").read_text())
    predictions = {cid: dict(controls[cid]) for cid in assessment_ids}
    for cid in assessment_ids:
        predictions[cid]["frozen_initial_adapter"] = ensemble_predict(models, assessment_features[cid])[0].tolist()
        for name, arm in arms.items():
            predictions[cid][name] = ensemble_predict(arm["models"], assessment_features[cid])[0].tolist()
        predictions[cid]["uniform_hash"] = [int(digest(["random-baseline", cid, i])[:12], 16) / 16**12 for i in range(len(rowmap[cid]["members"]))]
    metrics = {}
    per_context = {}
    groups = {cid: split[cid]["group"] for cid in assessment_ids}
    for budget in [1, 4, 8]:
        per_context[str(budget)] = {}
        for name in predictions[assessment_ids[0]]:
            scores = {cid: selection_metric(rowmap[cid], predictions[cid][name], budget) for cid in assessment_ids}
            per_context[str(budget)][name] = scores
            parent_values = {}
            for cid, values in scores.items():
                parent_values.setdefault(groups[cid], []).append(values)
            metrics[f"{name}/budget{budget}"] = {key: float(np.mean([np.mean([v[key] for v in values]) for values in parent_values.values()]))
                for key in ["quality", "normalized_quality", "regret", "beats_incumbent", "best_found"]}
    comparisons = {}
    planned = [("uniform_feedback", "continuation"), ("targeted_feedback", "uniform_feedback")]
    planned += [(arm, ref) for arm in ["uniform_feedback", "targeted_feedback"]
                for ref in ["native17773_trie", "native429_trie", "native17773_qhead", "native429_qhead", "frozen_initial_adapter"]]
    for candidate, reference in planned:
        values = per_context["4"]
        comparisons[f"{candidate}-{reference}"] = paired_interval(
            {cid: row["normalized_quality"] for cid, row in values[candidate].items()},
            {cid: row["normalized_quality"] for cid, row in values[reference].items()}, groups)
    primary = comparisons["uniform_feedback-continuation"]
    result = {"schema": "jope.native-feedback-upgrade-comparison/v1", "source_base": config["base"],
              "branch": config["branch"], "preregistration_sha256": hashlib.sha256((root / "preregistration.json").read_bytes()).hexdigest(),
              "assessment_contexts": len(assessment_ids), "assessment_parent_components": len(set(groups.values())),
              "metrics": metrics, "paired_99pct_intervals_budget4": comparisons,
              "primary_replay_effect_positive": primary["lower"] > 0,
              "all_native_controls_beaten_by_uniform_at_99pct": all(comparisons[f"uniform_feedback-{r}"]["lower"] > 0 for r in ["native17773_trie", "native429_trie", "native17773_qhead", "native429_qhead"]),
              "wave_receipts": {name: arm["waves"] for name, arm in arms.items()},
              "acquisition": {name: {"unique_measured_train_pairs": len(arm["ledger"].targets), "charged_logical_replay_slots": len(arm["ledger"].events), "fresh_gp_fits": 0} for name, arm in arms.items()},
              "actual_adapter_optimizer_updates": len(models)*config["initial_updates"] + len(arms)*config["waves"]*len(models)*config["updates_per_wave"],
              "elapsed_seconds": time.monotonic()-started,
              "limitations": [config["exposure"], config["conclusion_scope"],
                  "Frozen finite candidate pools exclude historical incumbent programs; candidates and cached outcomes predate this experiment.",
                  "Adapter training uses new ranking targets; native backbone weights and their teacher losses remain unchanged.",
                  "99% intervals describe retained parent variation in this historical replay. Secondary comparisons are exploratory, without familywise correction.",
                  "Three ensemble members form one policy, not three independent experimental replications.",
                  "No fresh numerical GP fits, raw S3 dataset refits, online pipeline integration, deployment or sealed confirmation."],
              "qualified_self_improvement": False, "production_promoted": False}
    publish(out / "comparison.json", result)
    with gzip.open(out / "per-context.json.gz", "wb") as handle:
        handle.write(canonical(per_context))
    publish(out / "assessment-predictions.json", predictions)
    print(json.dumps({"state": "COMPLETE", "primary": primary,
                      "native_controls_beaten": result["all_native_controls_beaten_by_uniform_at_99pct"],
                      "updates": result["actual_adapter_optimizer_updates"], "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
