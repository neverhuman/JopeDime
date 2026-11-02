"""Shared immutable types and low-level tensor helpers for the DOPE freeze guard."""

import hashlib
from dataclasses import dataclass

import torch

DOPE_NORMALIZER_BUFFERS: tuple[str, ...] = (
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


class DopeFreezeViolationError(RuntimeError):
    """Raised when any DOPE weight, buffer, grad flag, or eval mode invariant is violated."""


@dataclass(frozen=True)
class DopeFreezeSnapshot:
    composite_sha256: str
    params_sha256: str
    buffers_sha256: str
    num_param_tensors: int
    num_param_elements: int
    num_buffer_tensors: int
    num_buffer_elements: int
    tensor_hashes: dict[str, str]
    tensor_shapes: dict[str, list[int]]


def _tensor_le_sha256(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().newbyteorder("<")
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}:{list(arr.shape)}:".encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _storage_ptr(t: torch.Tensor) -> int | None:
    if not isinstance(t, torch.Tensor) or int(t.numel()) == 0:
        return None
    try:
        ptr = int(t.untyped_storage().data_ptr())
    except Exception:
        ptr = int(t.data_ptr())
    return ptr if ptr != 0 else None


def _assert_finite_tensor(t: torch.Tensor, label: str) -> None:
    if (t.is_floating_point() or t.is_complex()) and not bool(torch.isfinite(t).all()):
        raise DopeFreezeViolationError(
            f"DOPE freeze violation: `{label}` contains non-finite (NaN/Inf) values."
        )
