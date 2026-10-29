"""DOPE freeze guard for fused JopeDime + frozen DOPE training (TELEMETRY_SPEC rows D1-D5).

This is the product-tree port of the NS3 reference guard
(`/home/ubuntu/jope_plan/ns/dope_freeze_guard.py`). It hashes tensors the same way, so its
digests are comparable with NS3's. It also closes the two gaps the NS3 spot-check found:

* every DOPE submodule's ``training`` flag is checked, not only the root's, so a nested
  ``Dropout`` left in train mode fails;
* per-submodule parameter counts are reconciled against the global total, so parameters
  registered directly on the root model are counted too.

The guard runs at run start, at each epoch boundary and at checkpoints. Hashing copies the DOPE
tensors to the host, which synchronizes the device; that is accepted at epoch granularity and
never happens per step. ``step`` (row D4) only reads Python attributes.

A violation raises :class:`DopeFreezeViolationError` after the full observation is recorded, so
the emitted record lists every broken invariant, not just the first.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

DOPE_SUBMODULE = "dope_model"
ROOT_GROUP = "<root>"
RECORD_KIND = "jope.telemetry.dope-guard/v1"

# Normalizer buffers on the fused model that belong to DOPE (same list as NS3).
DOPE_NORMALIZER_BUFFERS = (
    "step2_lm_mean_1x256",
    "step2_lm_std_1x256",
    "step2_svd_basis_256x48",
    "lm80_mu",
    "lm80_sd",
    "mu_lm48",
    "sd_lm48",
    "mu_bb20",
    "sd_bb20",
    "mu_dm24",
    "sd_dm24",
)

Emit = Callable[[str, dict[str, Any]], Any]


class DopeFreezeViolationError(RuntimeError):
    """A DOPE weight, buffer, grad flag, optimizer membership or eval-mode invariant broke."""


def tensor_le_sha256(t: torch.Tensor) -> str:
    """NS3's tensor digest: ``"<numpy dtype str>:<shape>:"`` then the little-endian C-order bytes.

    NumPy has no bfloat16, which NS3 can't hash at all; those tensors are hashed through their
    int16 bit pattern under the tag ``bfloat16`` instead.
    """
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    if t.dtype == torch.bfloat16:
        arr = t.view(torch.int16).numpy()
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
        h.update(f"bfloat16:{list(arr.shape)}:".encode())
    else:
        arr = t.numpy()
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
        h.update(f"{arr.dtype.str}:{list(arr.shape)}:".encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


@dataclass(frozen=True)
class Snapshot:
    composite_sha256: str
    params_sha256: str
    buffers_sha256: str
    num_param_tensors: int
    num_param_elements: int
    num_buffer_tensors: int
    num_buffer_elements: int
    tensor_hashes: dict[str, str] = field(repr=False)


def _dope(fused_net: nn.Module) -> nn.Module:
    dope = getattr(fused_net, DOPE_SUBMODULE, None)
    if not isinstance(dope, nn.Module):
        raise DopeFreezeViolationError(f"fused model has no `{DOPE_SUBMODULE}` submodule")
    return dope


def snapshot(fused_net: nn.Module) -> Snapshot:
    """Digest every DOPE parameter and buffer plus the DOPE normalizer buffers (NS3 layout)."""
    dope = _dope(fused_net)
    hashes: dict[str, str] = {}
    h_params = hashlib.sha256()
    h_buffers = hashlib.sha256()
    n_pt = n_pe = n_bt = n_be = 0
    for name, p in sorted(dope.named_parameters(), key=lambda kv: kv[0]):
        full = f"{DOPE_SUBMODULE}.{name}"
        th = tensor_le_sha256(p)
        hashes[full] = th
        h_params.update(f"{full}={th}\n".encode())
        n_pt += 1
        n_pe += int(p.numel())
    if n_pt == 0:
        raise DopeFreezeViolationError(f"`{DOPE_SUBMODULE}` has no parameters")
    for name, b in sorted(dope.named_buffers(), key=lambda kv: kv[0]):
        full = f"{DOPE_SUBMODULE}.buffer.{name}"
        th = tensor_le_sha256(b)
        hashes[full] = th
        h_buffers.update(f"{full}={th}\n".encode())
        n_bt += 1
        n_be += int(b.numel())
    net_buffers = dict(fused_net.named_buffers())
    for name in DOPE_NORMALIZER_BUFFERS:
        if name in net_buffers:
            full = f"normalizer.{name}"
            th = tensor_le_sha256(net_buffers[name])
            hashes[full] = th
            h_buffers.update(f"{full}={th}\n".encode())
            n_bt += 1
            n_be += int(net_buffers[name].numel())
    p_sha = h_params.hexdigest()
    b_sha = h_buffers.hexdigest()
    composite = hashlib.sha256(f"params:{p_sha}|buffers:{b_sha}".encode()).hexdigest()
    return Snapshot(composite, p_sha, b_sha, n_pt, n_pe, n_bt, n_be, hashes)


def changed_tensors(before: Snapshot, after: Snapshot) -> list[str]:
    keys = set(before.tensor_hashes) | set(after.tensor_hashes)
    return sorted(k for k in keys if before.tensor_hashes.get(k) != after.tensor_hashes.get(k))


def param_counts(model: nn.Module) -> dict[str, Any]:
    """Row D2: trainable and frozen element counts per top-level submodule, reconciled.

    Parameters registered directly on the root are grouped under ``<root>``. ``reconciled`` says
    whether the groups sum to the totals over ``model.parameters()``; a shared parameter is
    counted once, in its first named group.
    """
    groups: dict[str, dict[str, int]] = {}
    for name, p in model.named_parameters():
        g = name.split(".", 1)[0] if "." in name else ROOT_GROUP
        c = groups.setdefault(g, {"trainable": 0, "frozen": 0})
        c["trainable" if p.requires_grad else "frozen"] += int(p.numel())
    total_trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total_frozen = sum(int(p.numel()) for p in model.parameters() if not p.requires_grad)
    reconciled = (
        sum(c["trainable"] for c in groups.values()) == total_trainable
        and sum(c["frozen"] for c in groups.values()) == total_frozen
    )
    return {
        "by_submodule": groups,
        "total_trainable": total_trainable,
        "total_frozen": total_frozen,
        "reconciled": reconciled,
    }


def dope_params_in_optimizer(fused_net: nn.Module, optimizer: torch.optim.Optimizer | None) -> int:
    """Row D3: DOPE parameter tensors present in any optimizer group."""
    if optimizer is None:
        return 0
    dope_ids = {id(p) for p in _dope(fused_net).parameters()}
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    return len(dope_ids & in_opt)


def dope_training_submodules(fused_net: nn.Module) -> list[str]:
    """Row D4: DOPE submodules (root included, as ``""``) whose ``training`` flag is set."""
    return [name for name, m in _dope(fused_net).named_modules() if m.training]


def dope_grads_present(fused_net: nn.Module) -> int:
    """Row D4: DOPE parameters holding a ``.grad``."""
    return sum(1 for p in _dope(fused_net).parameters() if p.grad is not None)


class DopeFreezeGuard:
    """Enforces and records the NORTH_STAR DOPE freeze around a fused training loop.

    Call ``run_start()`` once the optimizer exists, ``epoch_begin(e)`` / ``epoch_end(e)``
    around every epoch, ``checkpoint(step)`` before saving, and ``step(step)`` after each
    ``backward()`` (it does work only every ``sample_every`` steps). ``emit`` receives
    ``(RECORD_KIND, record)``; with no ``emit`` the guard still enforces.
    """

    def __init__(
        self,
        fused_net: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        emit: Emit | None = None,
        sample_every: int = 100,
    ) -> None:
        self.net = fused_net
        self.optimizer = optimizer
        self.emit = emit
        self.sample_every = max(1, int(sample_every))
        self.reference: Snapshot | None = None

    def _observe(self, boundary: str, at: int, compare: bool) -> dict[str, Any]:
        started = time.perf_counter()
        snap = snapshot(self.net)
        equal: bool | None = None
        changed: list[str] = []
        if compare and self.reference is not None:
            equal = snap.composite_sha256 == self.reference.composite_sha256
            if not equal:
                changed = changed_tensors(self.reference, snap)
        if self.reference is None:
            self.reference = snap
        counts = param_counts(self.net)
        dope_counts = counts["by_submodule"].get(DOPE_SUBMODULE, {"trainable": 0, "frozen": 0})
        jope_trainable = counts["total_trainable"] - dope_counts["trainable"]
        training = dope_training_submodules(self.net)
        rec: dict[str, Any] = {
            "boundary": boundary,
            "epoch": at,
            "hashes_equal": equal,
            "changed_tensors": changed,
            "composite_sha256": snap.composite_sha256,
            "params_sha256": snap.params_sha256,
            "buffers_sha256": snap.buffers_sha256,
            "reference_composite_sha256": self.reference.composite_sha256,
            "num_param_tensors": snap.num_param_tensors,
            "num_param_elements": snap.num_param_elements,
            "num_buffer_tensors": snap.num_buffer_tensors,
            "trainable_dope_params": dope_counts["trainable"],
            "trainable_jope_params": jope_trainable,
            "params_by_submodule": counts["by_submodule"],
            "params_reconciled": counts["reconciled"],
            "dope_params_in_optimizer": dope_params_in_optimizer(self.net, self.optimizer),
            "dope_grads_present": dope_grads_present(self.net),
            "dope_training_mode": bool(training),
            "dope_training_submodules": training,
        }
        rec["guard_ms"] = (time.perf_counter() - started) * 1e3
        self._enforce(rec)
        return rec

    def _enforce(self, rec: dict[str, Any]) -> None:
        reasons = []
        if rec["hashes_equal"] is False:
            reasons.append(f"DOPE hash changed: {', '.join(rec['changed_tensors'])}")
        if rec["trainable_dope_params"]:
            reasons.append(f"{rec['trainable_dope_params']} DOPE parameter elements have requires_grad=True")
        if rec["dope_params_in_optimizer"]:
            reasons.append(f"{rec['dope_params_in_optimizer']} DOPE parameters are in optimizer groups")
        if rec["dope_grads_present"]:
            reasons.append(f"{rec['dope_grads_present']} DOPE parameters hold a .grad")
        if rec["dope_training_mode"]:
            reasons.append(f"DOPE submodules in train mode: {rec['dope_training_submodules']}")
        if not rec["params_reconciled"]:
            reasons.append("per-submodule parameter counts don't sum to the model totals")
        if rec.get("trainable_jope_params", 1) == 0:
            reasons.append("no trainable JopeDime parameters")
        rec["violations"] = reasons
        if self.emit is not None:
            self.emit(RECORD_KIND, rec)
        if reasons:
            raise DopeFreezeViolationError("; ".join(reasons))

    def run_start(self) -> dict[str, Any]:
        self.reference = None
        return self._observe("run_start", 0, compare=False)

    def epoch_begin(self, epoch: int) -> dict[str, Any]:
        return self._observe("epoch_begin", epoch, compare=True)

    def epoch_end(self, epoch: int) -> dict[str, Any]:
        return self._observe("epoch_end", epoch, compare=True)

    def checkpoint(self, step: int) -> dict[str, Any]:
        return self._observe("checkpoint", step, compare=True)

    def step(self, step: int) -> None:
        """Row D4 between epoch hashes: attribute reads only, every ``sample_every`` steps."""
        if step % self.sample_every:
            return
        grads = dope_grads_present(self.net)
        training = dope_training_submodules(self.net)
        if grads or training:
            rec = {
                "boundary": "step",
                "epoch": step,
                "hashes_equal": None,
                "changed_tensors": [],
                "trainable_dope_params": 0,
                "dope_params_in_optimizer": 0,
                "dope_grads_present": grads,
                "dope_training_mode": bool(training),
                "dope_training_submodules": training,
                "params_reconciled": True,
            }
            self._enforce(rec)


def lock_dope(fused_net: nn.Module) -> None:
    """Freeze DOPE: ``requires_grad=False`` and eval mode on every DOPE submodule, kept in eval
    when ``fused_net.train(True)`` is called later. The guard verifies; this is the setup."""
    dope = _dope(fused_net)
    for p in dope.parameters():
        p.requires_grad_(False)
        p.grad = None
    dope.eval()
    original_train = fused_net.train

    def train(mode: bool = True) -> nn.Module:
        original_train(mode)
        dope.eval()
        return fused_net

    fused_net.train = train  # type: ignore[method-assign]
