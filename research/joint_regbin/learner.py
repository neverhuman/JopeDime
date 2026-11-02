"""Pinned request/response learner for the JRB-01 cached-pool study.

One implementation serves all three arms. The reviewed recipe is AdamW with
lr=1e-3, betas=(.9,.999), eps=1e-8 and weight_decay=.01, no dropout or scheduler.
Only the existing 29,211 adapter parameters and two binary-head parameters
learn; regression uses the existing identity head/Smooth L1, binary uses BCE.

Each payload is an allow_pickle=False NPZ: pool_json (a scalar JSON candidate
pool from candidate_builder), program, proj16, target, tokens_10x32, mask_10.
Evaluation additionally supplies per-candidate native inner-CV rmse (regression)
or auc/log_loss (binary), plus ridge_rmse or ridge_auc. These are dataset
prediction metrics, NEVER RMSE/AUC computed between adapter costs and gate
labels. Select minimum predicted cost for regression, maximum gate logit for
binary; ties take the first frozen candidate. Report cached native metrics.
Payload construction, role admission and review are the provider's gates.

The production CLI always requires the inherited real qualification lock. CPU
contract tests call the same update/checkpoint functions without reserving GPU
compute; synthetic tests cannot issue a qualified learning receipt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

# The driver executes this file from its private output directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from research.joint_regbin import driver, protocol, runtime, worker

RECIPE = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
PAYLOAD_SCHEMA = "jope.joint-regbin.learner-payload/v1"
# Filled from the reviewed base; a dependency change requires a new learner pin.
SOURCE_PINS = {
    "runtime.py": "efd51ff4218bc9e7f81db552a3735aaa17723437039ab27db001ff12cdb471c8",
    "worker.py": "154a8a57177e0f3552601dba10d91c4b0bb7639ebf48edbb53bad24651682e97",
    "candidate_builder.py": "25920e24fb22101e644bd1672f0b376093301821e3d5337a105aba9b7df67d02",
    "protocol.py": "b3bdb37fe2ef00f433b886c5864850f5b6d4da97118d3b55d549ea002b040612",
    "admission.py": "57c8c0850fb8517e2cad5ea54414054f9b1eca27a80a11c6168663e4ef47e3dd",
    "driver.py": "7473ce8252f9de879a62fdf8d31cc124be8af01ed797a5ac7a3edeb72baf291d",
}


def verify_sources():
    root = Path(__file__).parent
    for name, expected in SOURCE_PINS.items():
        runtime.verify_file_pin(root / name, expected)


def verify_lock(fd, path=driver.REAL_LOCK):
    """Require the inherited *held* open-file description, not just its inode."""
    actual, expected = os.fstat(fd), Path(path).stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("inherited descriptor is not the real qualification lock")
    with Path(path).open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError("qualification lock is not held")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError("inherited descriptor does not own the lock") from exc


def resource_check(device):
    if os.getloadavg()[0] > 0.85 * len(os.sched_getaffinity(0)):
        raise RuntimeError("CPU load exceeds 85% host budget")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        if free < 0.1 * total:
            raise RuntimeError("GPU memory exceeds 90% budget")


def configure(device):
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    resource_check(device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.90, device)


def validate_request(request):
    worker.validate_request(request)
    protocol.validate_config(request["config"])
    for key in ("plan_sha256", "learner_sha256", "initial_checkpoint_sha256"):
        protocol.check_sha(request[key])
    if request["learner_sha256"] != protocol.file_sha(Path(__file__)):
        raise ValueError("learner file hash mismatch")
    if request["initial_checkpoint_sha256"] != runtime.R882_CHECKPOINT_SHA256:
        raise ValueError("initial checkpoint must be pinned R882")
    if request["dope_sha256"] != protocol.DOPE:
        raise ValueError("frozen DOPE pin mismatch")
    if request["evaluation_protocol"] != protocol.EVALUATION_PROTOCOL:
        raise ValueError("evaluation protocol mismatch")
    if request["arm"] not in protocol.ARMS or type(request["seed"]) is not int:
        raise ValueError("invalid arm or seed")
    if not isinstance(request["job_id"], str) or not request["job_id"]:
        raise ValueError("job identity required")
    rows = request["contexts"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate context identity")
    if request["phase"] == "train":
        task = {"reg-only": "regression", "bin-only": "binary"}.get(request["arm"])
        if task and any(row["task"] != task for row in rows):
            raise ValueError("single-task arm contains another task")
        job = dict(arm=request["arm"], seed=request["seed"], train=rows)
        if request["batches"] != list(protocol.batches(job, request["config"])):
            raise ValueError("batch schedule differs from frozen arm/seed budget")


def load_payloads(rows, packet):
    """Resolve the entire supplied closure before opening its first payload."""
    paths = [(row, protocol.safe_path(packet, row["payload"])) for row in rows]
    pools, tokens, masks, metrics = {}, {}, {}, {}
    for row, path in paths:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
            raise ValueError("worker payload hash mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            meta = json.loads(str(data["pool_json"].item()))
            if meta.get("learner_payload_schema") != PAYLOAD_SCHEMA:
                raise ValueError("learner payload schema mismatch")
            pool = meta["pool"]
            identity = pool["context"]
            if identity.get("context_id") != row["id"] or any(
                identity.get(key) != row[key]
                for key in (
                    "dataset",
                    "family",
                    "task",
                    "role",
                    "source_split",
                    "origin",
                    "has_meta_test",
                )
            ):
                raise ValueError("payload context/task/role identity mismatch")
            pool = dict(
                pool,
                **{key: data[key].copy() for key in ("program", "proj16", "target")},
            )
            cid = row["id"]
            pools[cid] = pool
            tokens[cid] = torch.as_tensor(data["tokens_10x32"].copy())
            masks[cid] = torch.as_tensor(data["mask_10"].copy())
            metrics[cid] = meta.get("evaluation")
    return pools, tokens, masks, metrics


def trainable_state(pinned):
    return {
        **{
            name: value.detach().cpu().clone()
            for name, value in pinned.net.named_parameters()
            if value.requires_grad
        },
        **{
            "heads." + name: value.detach().cpu().clone()
            for name, value in pinned.heads.named_parameters()
        },
    }


def parameter_sha(pinned):
    return pinned.h144_runtime.tensor_hash(trainable_state(pinned))


def save_checkpoint(path, pinned, request, round_number):
    state = dict(
        schema="jope.joint-regbin.private-checkpoint/v1",
        plan_sha256=request["plan_sha256"],
        job_id=request["job_id"],
        learner_sha256=request["learner_sha256"],
        initial_checkpoint_sha256=request["initial_checkpoint_sha256"],
        round=round_number,
        parameters_sha256=parameter_sha(pinned),
        dope_sha256=protocol.DOPE,
        recipe=RECIPE,
        state_dict=trainable_state(pinned),
    )
    with Path(path).open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return protocol.file_sha(path)


def load_checkpoint(path, pinned, request, trace):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != trace["checkpoint_sha256"]:
        raise ValueError("private checkpoint file hash mismatch")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        state.get("schema") != "jope.joint-regbin.private-checkpoint/v1"
        or any(
            state.get(key) != request[key]
            for key in (
                "plan_sha256",
                "job_id",
                "learner_sha256",
                "initial_checkpoint_sha256",
            )
        )
        or state.get("round") != trace["round"]
        or state.get("dope_sha256") != protocol.DOPE
        or state.get("recipe") != RECIPE
        or set(state["state_dict"]) != set(trainable_state(pinned))
    ):
        raise ValueError("private checkpoint source/parameter inventory mismatch")
    with torch.no_grad():
        for name, value in pinned.net.named_parameters():
            if value.requires_grad:
                value.copy_(state["state_dict"][name])
        pinned.heads.load_state_dict(
            {
                key.removeprefix("heads."): value
                for key, value in state["state_dict"].items()
                if key.startswith("heads.")
            },
            strict=True,
        )
    pinned.verify_frozen_state()
    if (
        parameter_sha(pinned) != state["parameters_sha256"]
        or parameter_sha(pinned) != trace["parameters_after"]
    ):
        raise ValueError("private checkpoint parameter digest mismatch")


def train(request, pinned, core, directory, device):
    """Execute the supplied schedule; every round has a private checkpoint."""
    directory.mkdir(exist_ok=False)
    before = parameter_sha(pinned)
    by_id = {row["id"]: row for row in request["contexts"]}
    traces = []
    config = request["config"]
    for round_number in range(1, config["rounds"] + 1):
        initial = parameter_sha(pinned)
        start = (round_number - 1) * config["steps"]
        orders = request["batches"][start : start + config["steps"]]
        updates = []
        tick = time.monotonic()
        for batch in orders:
            resource_check(device)
            update = core.train_batch([by_id[cid] for cid in batch])
            gradients = [
                value.grad
                for value in (*pinned.adapter.parameters(), *pinned.heads.parameters())
                if value.grad is not None
            ]
            update["finite_gradients"] = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            update["gradient_l2"] = math.sqrt(
                sum(float(gradient.double().square().sum()) for gradient in gradients)
            )
            if not update["finite_gradients"] or any(
                not bool(torch.isfinite(value).all())
                for value in trainable_state(pinned).values()
            ):
                raise ValueError("nonfinite gradients or updated parameters")
            updates.append(update)
        final = parameter_sha(pinned)
        if initial == final:
            raise ValueError("round has no changed trainable parameters")
        path = directory / f"round-{round_number:03d}.pt"
        checkpoint_sha = save_checkpoint(path, pinned, request, round_number)
        traces.append(
            dict(
                round=round_number,
                optimizer_steps=sum(update["optimizer_steps"] for update in updates),
                batch_sha256=protocol.digest(orders),
                parameters_before=initial,
                parameters_after=final,
                dope_sha256=protocol.DOPE,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_path=str(path),
                updates=updates,
                wall_seconds=time.monotonic() - tick,
            )
        )
    return dict(
        schema=protocol.SCHEMA,
        plan_sha256=request["plan_sha256"],
        job_id=request["job_id"],
        learner_sha256=request["learner_sha256"],
        initial_checkpoint_sha256=request["initial_checkpoint_sha256"],
        actual_budget=dict(
            optimizer_steps=sum(trace["optimizer_steps"] for trace in traces),
            context_presentations=sum(
                update["contexts"] for trace in traces for update in trace["updates"]
            ),
            conditioning_token_slots=sum(
                update["contexts"] * config["tokens"]
                for trace in traces
                for update in trace["updates"]
            ),
        ),
        parameters_before=before,
        parameters_after=parameter_sha(pinned),
        dope_before=protocol.DOPE,
        dope_after=protocol.DOPE,
        round_receipts=traces,
        scores=[],
        optimizer_recipe=RECIPE,
        source_pins=SOURCE_PINS,
    )


def score(row, prediction, metrics, trace):
    """Choose a candidate, then report its pinned native dataset metric."""
    if not isinstance(metrics, dict) or not bool(torch.isfinite(prediction).all()):
        raise ValueError(
            "finite predictions and native cached evaluation metrics required"
        )
    task = row["task"]
    index = int(
        prediction.reshape(-1).argmin()
        if task == "regression"
        else prediction.reshape(-1).argmax()
    )
    metric = "rmse" if task == "regression" else "auc"
    values = metrics.get(metric)
    if not isinstance(values, list) or len(values) != prediction.numel():
        raise ValueError("native evaluation metrics do not cover the frozen pool")
    value, baseline = values[index], metrics.get("ridge_" + metric)
    if (
        type(value) not in (int, float)
        or type(baseline) not in (int, float)
        or not all(math.isfinite(x) for x in (value, baseline))
    ):
        raise ValueError("selected native metric is unavailable/nonfinite")
    if task == "regression":
        if value <= 0 or baseline <= 0:
            raise ValueError("protocol requires positive native RMSE/ridge RMSE")
        curve = math.log(value / baseline)
    else:
        losses = metrics.get("log_loss")
        if (
            not 0 <= value <= 1
            or not 0 <= baseline <= 1
            or not isinstance(losses, list)
            or len(losses) != len(values)
        ):
            raise ValueError("native binary evaluation metrics outside contract")
        curve = losses[index]
        if type(curve) not in (int, float) or not math.isfinite(curve) or curve < 0:
            raise ValueError("selected native binary log-loss unavailable")
    return dict(
        round=trace["round"],
        id=row["id"],
        status="ok",
        metric=metric,
        value=value,
        baseline=baseline,
        curve=curve,
        selected_candidate=index,
        model_sha256=trace["checkpoint_sha256"],
        evaluation_sha256=row["payload_sha256"],
        evaluation_protocol=protocol.EVALUATION_PROTOCOL,
    )


def evaluate(request, pinned, core, metrics, output, device):
    receipt = request["training_receipt"]
    for key in ("plan_sha256", "job_id", "learner_sha256", "initial_checkpoint_sha256"):
        if receipt.get(key) != request[key]:
            raise ValueError("training receipt source/job mismatch")
    config = request["config"]
    traces = receipt["round_receipts"]
    if len(traces) != config["rounds"] or receipt[
        "actual_budget"
    ] != protocol.expected_counts(config):
        raise ValueError("training receipt budget mismatch")
    if parameter_sha(pinned) != receipt["parameters_before"]:
        raise ValueError("evaluation did not start from identical R882/head parameters")
    scores = []
    with torch.inference_mode():
        for trace in traces:
            if trace["round"] % config["eval_every"]:
                continue
            resource_check(device)
            path = Path(trace["checkpoint_path"]).resolve()
            if not path.is_relative_to(output.resolve()) or protocol._is_protected(
                path
            ):
                raise ValueError("checkpoint escapes private job output")
            load_checkpoint(path, pinned, request, trace)
            for row in request["contexts"]:
                prediction, _target = core.predict(row)
                scores.append(score(row, prediction, metrics[row["id"]], trace))
    return dict(receipt, scores=scores)


def execute(request, packet, checkpoint, response, device):
    verify_sources()
    validate_request(request)
    response = Path(response)
    driver.output_location(response.parent)
    if response.exists() or response.is_symlink():
        raise ValueError("response must be a new private file")
    torch.manual_seed(request["seed"])
    pinned = runtime.initialize_r882_worker(checkpoint)
    pinned.net.to(device)
    pinned.heads.to(device)
    pools, tokens, masks, metrics = load_payloads(request["contexts"], packet)
    packer = pinned.make_candidate_packer(pools, tokens, masks, device=device)
    parameters = [
        value
        for value in (*pinned.adapter.parameters(), *pinned.heads.parameters())
        if value.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, **RECIPE)
    core = pinned.build_core(optimizer, packer)
    if request["phase"] == "train":
        result = train(
            request,
            pinned,
            core,
            response.parent / (response.stem + "-checkpoints"),
            device,
        )
    else:
        result = evaluate(request, pinned, core, metrics, response.parent, device)
    pinned.verify_frozen_state()
    driver.write_new(response, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("request", "response", "packet", "initial-checkpoint"):
        parser.add_argument("--" + argument, required=True, type=Path)
    args = parser.parse_args()
    fd = int(os.environ["JOPE_QUALIFICATION_LOCK_FD"])
    verify_lock(fd)
    device = torch.device(os.environ.get("JOPE_JRB_DEVICE", "cpu"))
    configure(device)
    request_path = protocol.safe_path(args.request.parent, args.request.name)
    execute(
        json.loads(request_path.read_text()),
        args.packet,
        args.initial_checkpoint,
        args.response,
        device,
    )
    verify_lock(fd)


if __name__ == "__main__":
    main()
