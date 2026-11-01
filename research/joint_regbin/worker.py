"""Task heads and one-step learner core for the JRB-01 worker.

Only admitted context rows are executable. TEL's metadata-only role candidates
do not include payloads and cannot be passed directly to this core. The tensor
packer is injected by the flywheel runtime so this worker cannot silently
substitute a different feature representation. Synthetic fixtures use the same
callable boundary and never open real payloads.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F

from . import protocol

TASK_HEAD_SOURCE = Path.home() / "jope_plan/ns/binary_clf_head_and_metrics.py"
BINARY_SOURCE_SHA256 = (
    "6b533cdec8d58e029593ff53ead9336851bbdd0535dff21becbdec597783a11b"
)
BINARY_HEAD_SOURCE = "binary_clf_head_and_metrics.py"
WORKER_SCHEMA = "jope.joint-regbin.worker/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics_module(path: Path = TASK_HEAD_SOURCE):
    """Load the prescribed binary head only when its source pin matches."""
    if sha256_file(path) != BINARY_SOURCE_SHA256:
        raise ValueError("binary task-head source pin mismatch")
    spec = importlib.util.spec_from_file_location("jrb01_binary_head", path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load pinned binary task-head source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_context_rows(rows: list[dict]) -> None:
    """Validate TEL's role-manifest context shape before any payload access."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("worker requires role-admitted context rows")
    required = {
        "id",
        "dataset",
        "family",
        "task",
        "role",
        "source_split",
        "origin",
        "has_meta_test",
        "payload",
        "payload_sha256",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("worker context fields differ from role-manifest schema")
        if row["task"] not in protocol.TASKS:
            raise ValueError("worker accepts only explicit regression/binary tasks")
        if row["role"] not in ("train", "val", "panel"):
            raise ValueError("worker context role is invalid")
        if row["source_split"] not in ("train", "inner"):
            raise ValueError("worker context is outside train-row closure")
        if row["origin"] != "real" or row["has_meta_test"] is not False:
            raise ValueError("worker refuses synthetic, sealed, or holdout contexts")
        protocol.relative_payload(row["payload"])
        protocol.check_sha(row["payload_sha256"])


def validate_request(request: dict) -> None:
    """Check the driver's request and TEL role rows before payload packing."""
    if not isinstance(request, dict) or request.get("schema") != protocol.SCHEMA:
        raise ValueError("worker request schema mismatch")
    phase = request.get("phase")
    if phase not in ("train", "evaluate"):
        raise ValueError("worker phase must be train or evaluate")
    contexts = request.get("contexts")
    validate_context_rows(contexts)
    if phase == "train":
        if any(row["role"] != "train" for row in contexts):
            raise ValueError("training request contains a non-training role")
        batches = request.get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError("training request lacks its fixed batch schedule")
        known = {row["id"] for row in contexts}
        if any(
            not isinstance(batch, list)
            or not batch
            or any(context_id not in known for context_id in batch)
            for batch in batches
        ):
            raise ValueError("training schedule references unknown contexts")
    elif "training_receipt" not in request:
        raise ValueError("evaluation request lacks the private training receipt")


class TwoTaskHeads(nn.Module):
    """Regression identity head plus a scalar linear binary logit head.

    ``shared_output`` is the per-candidate scalar produced by the existing
    flywheel regression adapter. Binary initialization is the identity mapping;
    the binary head is then learned with BCE-with-logits alongside the shared
    adapter. Platt calibration is applied only for reported binary metrics.
    """

    def __init__(self, shared_width: int = 1):
        super().__init__()
        if type(shared_width) is not int or shared_width < 1:
            raise ValueError("positive shared adapter output width required")
        self.binary = nn.Linear(shared_width, 1)
        with torch.no_grad():
            self.binary.weight.zero_()
            self.binary.bias.zero_()
            if shared_width == 1:
                self.binary.weight.fill_(1.0)

    def forward(self, task: str, shared_output: torch.Tensor) -> torch.Tensor:
        if task == "regression":
            if shared_output.shape[-1] != 1:
                raise ValueError(
                    "existing regression head must emit one cost per candidate"
                )
            return shared_output.squeeze(-1)
        if task == "binary":
            return self.binary(shared_output).squeeze(-1)
        raise ValueError(
            "unknown task; task labels are never inferred from target values"
        )


def task_loss(
    task: str, prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("prediction and task target shapes must match and be nonempty")
    if not bool(torch.isfinite(target).all()) or not bool(
        torch.isfinite(prediction).all()
    ):
        raise ValueError("nonfinite task predictions or targets")
    if task == "regression":
        return F.smooth_l1_loss(prediction.float(), target.float())
    if task == "binary":
        if not bool(((target == 0) | (target == 1)).all()):
            raise ValueError("binary targets must be explicit zero/one labels")
        return F.binary_cross_entropy_with_logits(prediction.float(), target.float())
    raise ValueError("unknown task")


def train_step(
    adapter: nn.Module,
    heads: TwoTaskHeads,
    optimizer: torch.optim.Optimizer,
    task: str,
    packed_inputs: dict,
    target: torch.Tensor,
) -> dict:
    """Apply one task-labelled update through the shared flywheel adapter."""
    if task not in protocol.TASKS:
        raise ValueError("task label must be regression or binary")
    output = adapter.predict_costs(**packed_inputs)
    shared = output.get("total_cost")
    if not isinstance(shared, torch.Tensor):
        raise ValueError("flywheel adapter must return tensor total_cost")
    if shared.shape != target.shape:
        raise ValueError("shared adapter outputs must align with per-candidate targets")
    prediction = heads(task, shared.unsqueeze(-1))
    loss = task_loss(task, prediction, target)
    before = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )
    if before != 29211:
        raise ValueError("exactly 29,211 shared adapter parameters must be trainable")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in adapter.parameters()
        if parameter.requires_grad
    ):
        raise ValueError("task loss did not produce a finite shared-adapter gradient")
    before_state = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    optimizer.step()
    changed = any(
        not torch.equal(before_state[name], parameter.detach())
        for name, parameter in adapter.named_parameters()
        if name in before_state
    )
    if not changed:
        raise ValueError("optimizer step did not change shared-adapter parameters")
    return {
        "task": task,
        "loss": float(loss.detach()),
        "adapter_parameters": before,
        "parameters_changed": changed,
    }


def adapter_fingerprint(adapter: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(adapter.named_parameters()):
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def hash_batch_ids(context_ids: Sequence[str]) -> str:
    raw = json.dumps(list(context_ids), separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def train_context_batch(
    adapter: nn.Module,
    heads: TwoTaskHeads,
    optimizer: torch.optim.Optimizer,
    contexts: Sequence[tuple[str, str, dict, torch.Tensor]],
) -> dict:
    """Combine a scheduled context batch into exactly one optimizer update.

    Each tuple is ``(context_id, explicit_task, packed_flywheel_inputs,
    candidate_targets)``.
    The caller's packer remains responsible for the exact frozen flywheel
    candidate-pool representation and for restricting targets to TRAIN rows.
    """
    if not contexts:
        raise ValueError("empty scheduled context batch")
    optimizer.zero_grad(set_to_none=True)
    by_task = {task: 0 for task in protocol.TASKS}
    losses = []
    context_ids = []
    before = adapter_fingerprint(adapter)
    trainable = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    if sum(parameter.numel() for parameter in trainable) != 29211:
        raise ValueError("exactly 29,211 shared adapter parameters must be trainable")
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if any(
        id(parameter) not in optimizer_parameters
        for parameter in (*trainable, *heads.parameters())
    ):
        raise ValueError("optimizer must include the shared adapter and binary head")
    for context_id, task, packed_inputs, target in contexts:
        if task not in protocol.TASKS:
            raise ValueError("explicit task labels required in every context")
        if not isinstance(packed_inputs, dict) or not isinstance(target, torch.Tensor):
            raise ValueError("packed inputs and candidate target tensor required")
        output = adapter.predict_costs(**packed_inputs)
        shared = output.get("total_cost")
        if not isinstance(shared, torch.Tensor) or shared.shape != target.shape:
            raise ValueError(
                "adapter scalar output must align with per-candidate targets"
            )
        prediction = heads(task, shared.unsqueeze(-1))
        losses.append(task_loss(task, prediction, target))
        by_task[task] += 1
        context_ids.append(str(context_id))
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite scheduled-batch loss")
    loss.backward()
    if not any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    ):
        raise ValueError("scheduled task loss did not reach the shared adapter")
    optimizer.step()
    after = adapter_fingerprint(adapter)
    if before == after:
        raise ValueError("scheduled optimizer update did not change adapter parameters")
    batch_sha = hash_batch_ids(context_ids)
    return {
        "optimizer_steps": 1,
        "contexts": len(contexts),
        "task_contexts": by_task,
        "loss": float(loss.detach()),
        "batch_sha256": batch_sha,
        "parameters_before": before,
        "parameters_after": after,
    }


class WorkerCore:
    """Manifest-bound worker core with an injected flywheel pool packer."""

    def __init__(self, adapter, heads, optimizer, packer):
        if not callable(packer):
            raise ValueError("the flywheel candidate-pool packer is required")
        self.adapter = adapter
        self.heads = heads
        self.optimizer = optimizer
        self.packer = packer

    def train_batch(self, rows: list[dict]) -> dict:
        validate_context_rows(rows)
        if any(row["role"] != "train" for row in rows):
            raise ValueError("only train-role contexts may update the worker")
        contexts = []
        for row in rows:
            packed_inputs, target = self.packer(row)
            contexts.append((row["id"], row["task"], packed_inputs, target))
        return train_context_batch(self.adapter, self.heads, self.optimizer, contexts)

    def predict(self, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
        validate_context_rows([row])
        packed_inputs, target = self.packer(row)
        with torch.no_grad():
            shared = self.adapter.predict_costs(**packed_inputs)["total_cost"]
            prediction = self.heads(row["task"], shared.unsqueeze(-1))
        if prediction.shape != target.shape:
            raise ValueError("worker prediction and candidate target shapes differ")
        return prediction, target


def calibrated_binary_metrics(
    train_logits: torch.Tensor,
    train_target: torch.Tensor,
    eval_logits: torch.Tensor,
    eval_target: torch.Tensor,
) -> dict:
    """Report Platt-calibrated loss and AUC using the pinned task-spec helpers."""
    if train_logits.ndim != 1 or eval_logits.ndim != 1:
        raise ValueError("binary context logits must be one-dimensional")
    if (
        train_target.shape != train_logits.shape
        or eval_target.shape != eval_logits.shape
    ):
        raise ValueError("binary labels must align with candidate logits")
    module = binary_metrics_module()
    train_mask = torch.ones_like(train_target, dtype=torch.bool).reshape(1, -1)
    eval_mask = torch.ones_like(eval_target, dtype=torch.bool).reshape(1, -1)
    alpha, beta = module.fit_batched_platt_calibration_gpu(
        train_logits.reshape(1, -1), train_target.float().reshape(1, -1), train_mask
    )
    calibrated = torch.sigmoid(
        (alpha * eval_logits.reshape(1, -1) + beta).clamp(-15.0, 15.0)
    )
    auc = module.batched_roc_auc_gpu(
        calibrated, eval_target.float().reshape(1, -1), eval_mask
    )
    eps = 1e-6
    log_loss = F.binary_cross_entropy(
        calibrated.clamp(eps, 1 - eps), eval_target.float().reshape(1, -1)
    )
    return {"log_loss": float(log_loss), "roc_auc": float(auc[0])}


def regression_metrics(
    prediction: torch.Tensor, target: torch.Tensor, ridge_rmse: float
) -> dict:
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("regression metric tensors must be aligned and nonempty")
    if not math.isfinite(ridge_rmse) or ridge_rmse <= 0:
        raise ValueError("positive finite ridge RMSE required")
    rmse = torch.mean((prediction.float() - target.float()).square()).sqrt()
    value = float(rmse)
    return {"rmse": value, "log_rmse_ratio": math.log(value / ridge_rmse)}
