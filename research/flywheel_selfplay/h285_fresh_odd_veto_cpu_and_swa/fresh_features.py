"""Exact H58 feature functions extracted without importing training entry points."""

from __future__ import annotations
import numpy as np
import torch
from types import SimpleNamespace


def landmark_float32(features):
    """Avoid overflow before the frozen landmark engine's per-column scaling.

    Rescale only columns outside float32 range by their finite maximum magnitude.
    Raw JTT1 data and labels used by GP evaluation remain untouched.
    """
    values = np.asarray(features, dtype=np.float64)
    maxima = np.max(np.where(np.isfinite(values), np.abs(values), 0.0), axis=0)
    columns = maxima > np.finfo(np.float32).max
    if columns.any():
        values = values.copy()
        values[:, columns] /= maxima[columns]
    return np.ascontiguousarray(values, dtype=np.float32)


from hpc.landmark_workspace import FusedLandmarkWorkspace  # noqa: E402
from phase12_multimodal_model import compute_semantic_struct_sig_24d  # noqa: E402

base = SimpleNamespace(
    FusedLandmarkWorkspace=FusedLandmarkWorkspace,
    landmark_float32=landmark_float32,
    compute_semantic_struct_sig_24d=compute_semantic_struct_sig_24d,
)


def extract_single_table_features(
    net: torch.nn.Module,
    ws: base.FusedLandmarkWorkspace,
    x_np: np.ndarray,
    y_np: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run frozen Stage-1A landmark + DOPE629 and Stage-1B token projection for one table."""
    yd = y_np.astype(np.float64)
    yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
    vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
        base.landmark_float32(x_np), np.ascontiguousarray(yn)
    )
    colbuf = np.zeros((64, 17), np.float32)
    ncols = min(len(cols), 64)
    colbuf[:ncols, : min(cols.shape[1], 17)] = cols[:ncols, :17]

    with torch.no_grad():
        d629, lm48 = net._run_dope_mlc_v3_and_lm48(
            torch.as_tensor(vec, device=device).reshape(1, 256),
            torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
            use_landmarks=True,
        )
        tok, mask, lm48_z = net._project_stage1b_tokens_batched_gpu(lm48, d629)

    d629_np = d629[0].cpu().numpy().astype(np.float32)
    lm48_z_np = lm48_z[0].cpu().numpy().astype(np.float32)
    tok_np = tok[0].cpu().numpy().astype(np.float32)
    mask_np = mask[0].cpu().numpy().astype(bool)
    ctx_584 = np.zeros((584,), dtype=np.float32)
    ctx_584[88:200] = d629_np[:112]
    ctx_584[536:584] = lm48_z_np
    return tok_np, mask_np, d629_np, lm48_z_np, ctx_584


def compute_candidate_priors(
    net: torch.nn.Module,
    prog_412: np.ndarray,
    lm48_z_1d: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute real base_prior_cost [M] and sos_prior [M, 2] using frozen network buffers."""
    m = prog_412.shape[0]
    sem_24 = base.compute_semantic_struct_sig_24d(prog_412)
    with torch.no_grad():
        c_t = torch.as_tensor(prog_412, dtype=torch.float32, device=device).unsqueeze(0)
        s_t = torch.as_tensor(sem_24, dtype=torch.float32, device=device).unsqueeze(0)
        l_t = torch.as_tensor(lm48_z_1d, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        base_p = net._predict_calibrated_prior_batched_gpu(
            c_t, s_t, l_t, track_prefix="prior10"
        )[0]
        sos_idx = torch.full((1, m), -1, dtype=torch.long, device=device)
        sos_p = net._compute_sos_prior_batched_gpu(c_t, sos_idx)[0]
    return base_p.cpu().numpy().astype(np.float32), sos_p.cpu().numpy().astype(
        np.float32
    )
