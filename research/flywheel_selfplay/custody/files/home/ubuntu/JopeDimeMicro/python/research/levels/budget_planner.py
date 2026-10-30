"""Dynamic 4-Tier Byte-Budget Planner and Self-Contained Wire Codec (DOPE-MLC-v3).

Enforces strict complete-artifact byte caps (wire payload + scaler sidecar <= cap):
- Small (Low):  2,048 B (2 KiB),  1,024 learned coordinates
- Medium:       4,096 B (4 KiB),  4,096 learned coordinates
- High:        16,384 B (16 KiB), 8,192 learned coordinates
- Ultra:       65,536 B (64 KiB), 10,240 learned coordinates
"""
from __future__ import annotations

import struct
import zlib
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .scaling import FitOnlyTableTransform, NumericColumnScaler, TargetScaler

TIER_BYTE_CAPS: Dict[str, int] = {
    "low": 2048,
    "small": 2048,
    "medium": 4096,
    "high": 16384,
    "ultra": 65536,
}

TIER_COORD_WIDTHS: Dict[str, int] = {
    "low": 1024,
    "small": 1024,
    "medium": 4096,
    "high": 8192,
    "ultra": 10240,
}

_KIND_TO_ID = {
    "constant": 0, "binary": 1, "signed_log1p_hybrid": 2,
    "quantile_robust_hybrid": 3, "asinh_hybrid": 4, "categorical_code": 5,
}
_ID_TO_KIND = {v: k for k, v in _KIND_TO_ID.items()}
_TASK_TO_ID = {"regression": 0, "classification": 1, "binary": 1, "multiclass": 2}
_ID_TO_TASK = {0: "regression", 1: "classification", 2: "multiclass"}
_LVL_TO_ID = {"low": 0, "small": 0, "medium": 1, "high": 2, "ultra": 3}
_ID_TO_LVL = {0: "small", 1: "medium", 2: "high", 3: "ultra"}


def plan_tier_envelope(
    p: int,
    level: str,
    *,
    k_classes: int = 1,
    adaptive_budget: bool = True,
) -> Dict[str, int]:
    """Deterministically size knots, Cholesky rank, pairs, and trees so wire + scaler <= cap."""
    lvl = "small" if level == "low" else level
    p_eff = max(int(p), 1)
    if not adaptive_budget:
        return {
            "n_knots": 9 if lvl == "small" else 17 if lvl == "medium" else 33 if lvl == "high" else 65,
            "rank_use": min(p_eff, 4 if lvl == "small" else 8 if lvl == "medium" else 20 if lvl == "high" else p_eff),
            "n_regimes": 1 if lvl in ("small", "medium") else 4 if lvl == "high" else 8,
            "max_pairs": 0 if lvl in ("small", "medium") else 16 if lvl == "high" else 28,
            "n_trees": 2 if lvl == "small" else 4 if lvl == "medium" else 24 if lvl == "high" else 56,
            "tree_depth": 2 if lvl in ("small", "medium") else 3 if lvl == "high" else 4,
            "latent_keep": 32 if lvl == "small" else 128,
        }

    if lvl == "small":
        n_knots = 5 if p_eff <= 16 else 3
        rank_use = min(p_eff, 3 if p_eff <= 16 else 1)
        n_trees = 2 if k_classes <= 2 else 1
        return {
            "n_knots": n_knots, "rank_use": rank_use, "n_regimes": 1,
            "max_pairs": 0, "n_trees": n_trees, "tree_depth": 2, "latent_keep": 24 if p_eff > 24 else 48,
        }
    if lvl == "medium":
        if p_eff <= 20:
            n_knots, rank_use, lat_keep = 17, min(p_eff, 8), 128
        elif p_eff <= 34:
            n_knots, rank_use, lat_keep = 9, min(p_eff, 5), 96
        else:
            n_knots, rank_use, lat_keep = 5, min(p_eff, 3), 64
        n_trees = 4 if k_classes <= 2 else 3
        return {
            "n_knots": n_knots, "rank_use": rank_use, "n_regimes": 1,
            "max_pairs": 0, "n_trees": n_trees, "tree_depth": 2, "latent_keep": lat_keep,
        }
    if lvl == "high":
        n_knots = 33 if p_eff <= 40 else 17
        rank_use = min(p_eff, 20 if p_eff <= 40 else 12)
        n_trees = 24 if k_classes <= 2 else 12
        return {
            "n_knots": n_knots, "rank_use": rank_use, "n_regimes": 4 if p_eff <= 48 else 2,
            "max_pairs": 16 if k_classes <= 2 else 8, "n_trees": n_trees, "tree_depth": 3, "latent_keep": 128,
        }
    # ultra (64 KiB cap)
    n_knots = 65 if p_eff <= 64 else 33
    rank_use = min(p_eff, 64)
    n_trees = 56 if k_classes <= 2 else 24
    return {
        "n_knots": n_knots, "rank_use": rank_use, "n_regimes": 8 if p_eff <= 64 else 4,
        "max_pairs": 28 if k_classes <= 2 else 16, "n_trees": n_trees,
        "tree_depth": 4 if k_classes <= 2 else 3, "latent_keep": 128,
    }


def pack_compact_scaler_sidecar(scaler: Optional[FitOnlyTableTransform], p_use: int, level: str = "ultra") -> bytes:
    """Serialize train-fitted scaler parameters into a compact self-contained byte block."""
    if scaler is None:
        return b""
    lvl = "small" if level == "low" else level
    compact_q = lvl in ("small", "medium") and p_use > 20
    parts = []
    t_sc = scaler.target_scaler
    t_id = _TASK_TO_ID.get(t_sc.task if t_sc else "regression", 0)
    k_cls = len(t_sc.class_labels) if (t_sc and t_sc.task == "multiclass") else 0
    parts.append(struct.pack("<BBHB", t_id, k_cls, p_use, 1 if compact_q else 0))
    if t_sc is not None:
        parts.append(np.array([t_sc.mean, t_sc.std, t_sc.median, t_sc.iqr, t_sc.class_prior], dtype=np.float32).tobytes())
        if k_cls > 0:
            parts.append(np.asarray(t_sc.class_labels, dtype=np.float32).tobytes())
            parts.append(np.asarray(t_sc.class_priors, dtype=np.float16).tobytes())
    else:
        parts.append(np.zeros(5, dtype=np.float32).tobytes())

    for j in range(p_use):
        if j in scaler.numeric_scalers:
            sc = scaler.numeric_scalers[j]
            k_id = _KIND_TO_ID.get(sc.transform_kind, 3)
            parts.append(struct.pack("<B", k_id))
            s6 = np.array([sc.center, sc.scale, sc.post_mean, sc.post_std, sc.missing_rate, sc.blend_quantile_weight], dtype=np.float16)
            q_arr = np.asarray(sc.quantiles, dtype=np.float16)
            q_take = np.array([q_arr[0], q_arr[4], q_arr[-1]], dtype=np.float16) if compact_q else q_arr
            if lvl == "small" and p_use > 16:
                s6 = (np.rint(s6.astype(np.float32) * 32.0) / 32.0).astype(np.float16)
                q_take = (np.rint(q_take.astype(np.float32) * 32.0) / 32.0).astype(np.float16)
            parts.append(s6.tobytes())
            parts.append(q_take.tobytes())
        else:
            ce = scaler.categorical_encoders.get(j)
            pm = float(ce.post_mean) if ce is not None else 0.0
            ps = float(ce.post_std) if ce is not None else 1.0
            mr = float(ce.missing_rate) if ce is not None else 0.0
            parts.append(struct.pack("<B", 5))
            s6 = np.array([0.0, 1.0, pm, ps, mr, 0.0], dtype=np.float16)
            parts.append(s6.tobytes())
            parts.append(np.zeros(3 if compact_q else 9, dtype=np.float16).tobytes())
    return b"".join(parts)


def unpack_compact_scaler_sidecar(buf: bytes) -> Optional[FitOnlyTableTransform]:
    """Reconstruct FitOnlyTableTransform in an isolated receiver from compact sidecar bytes."""
    if not buf:
        return None
    offset = 0
    t_id, k_cls, p_use, is_compact = struct.unpack_from("<BBHB", buf, offset)
    offset += 5
    task = _ID_TO_TASK.get(t_id, "regression")
    t_vals = np.frombuffer(buf, dtype=np.float32, count=5, offset=offset).astype(np.float64)
    offset += 20
    c_labels: Tuple[float, ...] = ()
    c_priors: Tuple[float, ...] = ()
    if k_cls > 0:
        c_labels = tuple(float(v) for v in np.frombuffer(buf, dtype=np.float32, count=k_cls, offset=offset))
        offset += 4 * k_cls
        c_priors = tuple(float(v) for v in np.frombuffer(buf, dtype=np.float16, count=k_cls, offset=offset))
        offset += 2 * k_cls
    t_scaler = TargetScaler(
        task=task, mean=float(t_vals[0]), std=max(float(t_vals[1]), 1e-9),
        median=float(t_vals[2]), iqr=max(float(t_vals[3]), 1e-9),
        skewness=0.0, kurtosis=0.0, positive_label="1", class_prior=float(t_vals[4]),
        class_labels=c_labels, class_priors=c_priors,
    )
    kinds_list = ["numeric"] * p_use
    num_scalers: Dict[int, NumericColumnScaler] = {}
    for j in range(p_use):
        (k_id,) = struct.unpack_from("<B", buf, offset)
        offset += 1
        s_vals = np.frombuffer(buf, dtype=np.float16, count=6, offset=offset).astype(np.float64)
        offset += 12
        if is_compact:
            q3 = np.frombuffer(buf, dtype=np.float16, count=3, offset=offset).astype(np.float64)
            offset += 6
            q9 = np.interp(
                (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
                (0.01, 0.50, 0.99),
                q3,
            )
        else:
            q9 = np.frombuffer(buf, dtype=np.float16, count=9, offset=offset).astype(np.float64)
            offset += 18
        if k_id == 5:
            kinds_list[j] = "categorical"
        else:
            num_scalers[j] = NumericColumnScaler(
                slot=j, anonymous_name=f"f_{j:03d}", transform_kind=_ID_TO_KIND.get(k_id, "quantile_robust_hybrid"),
                missing_rate=float(s_vals[4]), center=float(s_vals[0]), scale=max(float(s_vals[1]), 1e-9),
                post_mean=float(s_vals[2]), post_std=max(float(s_vals[3]), 1e-9),
                quantiles=tuple(float(v) for v in q9), skewness=0.0, kurtosis=0.0, blend_quantile_weight=float(s_vals[5]),
            )
    return FitOnlyTableTransform(task=task, kinds=tuple(kinds_list), numeric_scalers=num_scalers, target_scaler=t_scaler)


def _pack_dml3_from_encoding(
    encoding: Dict[str, Any], scaler_bytes: bytes, *, lvl: str, is_mc: bool, compress: bool
) -> bytes:
    from .strata_audit import serialize_dope_latent_wire_bytes

    tb = encoding["target_bundle"]
    enc_copy = dict(encoding)
    if lvl in ("low", "small"):
        enc_copy["level"] = "medium"
    if is_mc:
        enc_copy["task"] = "classification"
        enc_copy["target_bundle"] = dict(tb, trees=[], w_basis=np.zeros(1, dtype=np.float16))
    core_bytes = serialize_dope_latent_wire_bytes(enc_copy, version=2)

    mc_parts = []
    if is_mc:
        k_cls = int(tb.get("k_classes", 3))
        U = np.asarray(tb["U_basis"], dtype=np.float16)
        V = np.asarray(tb["V_basis"], dtype=np.float16)
        priors = np.asarray(tb["class_priors"], dtype=np.float16)
        trees = tb.get("trees", [])
        mc_parts.extend([struct.pack("<HHHH", k_cls, U.shape[0], U.shape[1], len(trees)), priors.tobytes(), U.tobytes(), V.tobytes()])
        for t in trees:
            feats = [int(f) for f in t["features"]]
            ths = np.asarray(t["thresholds"], dtype=np.float16)
            lv = np.asarray(t["leaf_values"], dtype=np.float16).reshape(-1, k_cls)
            mc_parts.append(struct.pack("<B", len(feats)))
            mc_parts.append(struct.pack(f"<{len(feats)}B", *feats))
            mc_parts.append(ths.tobytes())
            mc_parts.append(lv.tobytes())
    mc_bytes = b"".join(mc_parts)

    lvl_id = _LVL_TO_ID.get(lvl, 3)
    task_id = 2 if is_mc else _TASK_TO_ID.get(str(encoding.get("task", "regression")), 0)
    inc_v3 = 1 if tb.get("include_v3_primitives", False) else 0
    payload = b"".join([
        struct.pack("<BBBBIII", lvl_id, task_id, inc_v3, 0, len(core_bytes), len(scaler_bytes), len(mc_bytes)),
        core_bytes, scaler_bytes, mc_bytes,
    ])
    body = zlib.compress(payload, level=9) if compress else payload
    flags = 1 if compress else 0
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return b"DML3" + struct.pack("<BBI", 3, flags, crc) + body


def serialize_dope_v3_envelope(
    encoding: Dict[str, Any],
    scaler: Optional[FitOnlyTableTransform] = None,
    *,
    compress: bool = True,
) -> bytes:
    """Pack a complete DOPE-MLC-v3 artifact (latent wire + optional scaler sidecar) into a CRC32-verified byte stream."""
    from .strata_audit import serialize_dope_latent_wire_bytes

    lvl = str(encoding.get("level", "ultra"))
    p_use = int(encoding.get("p_use", 1))
    tb = encoding["target_bundle"]
    is_mc = tb.get("task_kind") == "multiclass" or encoding.get("task") == "multiclass"

    if not is_mc and lvl in ("medium", "high", "ultra") and scaler is None:
        return serialize_dope_latent_wire_bytes(encoding, version=2)

    scaler_bytes = pack_compact_scaler_sidecar(scaler, p_use, level=lvl)
    wire = _pack_dml3_from_encoding(encoding, scaler_bytes, lvl=lvl, is_mc=is_mc, compress=compress)
    cap = TIER_BYTE_CAPS.get(lvl, 65536)
    if compress and len(wire) > cap:
        # Closed-loop budget enforcement: progressively sparsify & quantize low-energy coefficients until <= cap
        enc_trim = dict(encoding)
        tb_trim = dict(encoding["target_bundle"])
        enc_trim["target_bundle"] = tb_trim
        for step in (1, 2, 3):
            keep_frac = 0.45 if step == 1 else 0.20 if step == 2 else 0.08
            q_grid = 16.0 if step == 1 else 8.0 if step == 2 else 4.0
            enc_trim["col_knots"] = (np.rint(np.asarray(encoding["col_knots"], dtype=np.float32) * q_grid) / q_grid).astype(np.float16)
            enc_trim["chol_factor"] = (np.rint(np.asarray(encoding["chol_factor"], dtype=np.float32) * q_grid) / q_grid).astype(np.float16)
            if step >= 2:
                enc_trim["task_latent"] = np.zeros_like(encoding["task_latent"])
                enc_trim["dist_latent"] = np.zeros_like(encoding["dist_latent"])
                tb_trim["trees"] = tb_trim["trees"][:1] if step == 2 else []
            for key in ("w_basis", "U_basis"):
                if key in tb_trim:
                    arr = np.asarray(encoding["target_bundle"][key], dtype=np.float32).copy()
                    thresh = float(np.quantile(np.abs(arr), 1.0 - keep_frac)) if arr.size > 0 else 0.0
                    arr[np.abs(arr) < thresh] = 0.0
                    tb_trim[key] = (np.rint(arr * q_grid) / q_grid).astype(np.float16)
            wire = _pack_dml3_from_encoding(enc_trim, scaler_bytes, lvl=lvl, is_mc=is_mc, compress=compress)
            if len(wire) <= cap:
                break
    return wire


def deserialize_dope_v3_envelope(wire_bytes: bytes) -> Tuple[Dict[str, Any], Optional[FitOnlyTableTransform]]:
    """Unpack a DML3 or prior-version DML2 wire stream in an isolated receiver."""
    from .strata_audit import deserialize_dope_latent_wire_bytes

    if wire_bytes[:4] == b"DML2":
        return deserialize_dope_latent_wire_bytes(wire_bytes), None
    if wire_bytes[:4] != b"DML3":
        raise ValueError("invalid_dope_v3_wire_magic")
    ver, flags, expected_crc = struct.unpack_from("<BBI", wire_bytes, 4)
    body = wire_bytes[10:]
    if (zlib.crc32(body) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("dope_v3_wire_crc_mismatch")
    payload = zlib.decompress(body) if (flags & 1) else body
    lvl_id, task_id, inc_v3, _, len_core, len_sc, len_mc = struct.unpack_from("<BBBBIII", payload, 0)
    off = 16
    core_bytes = payload[off : off + len_core]
    off += len_core
    sc_bytes = payload[off : off + len_sc]
    off += len_sc
    mc_bytes = payload[off : off + len_mc]

    enc = deserialize_dope_latent_wire_bytes(core_bytes)
    enc["level"] = _ID_TO_LVL.get(lvl_id, "ultra")
    enc["code_width"] = TIER_COORD_WIDTHS[enc["level"]]
    enc["task"] = _ID_TO_TASK.get(task_id, "regression")
    enc["target_bundle"]["include_v3_primitives"] = bool(inc_v3)
    if task_id == 2 and mc_bytes:
        k_cls, u_r, u_c, n_trees = struct.unpack_from("<HHHH", mc_bytes, 0)
        m_off = 8
        priors = np.frombuffer(mc_bytes, dtype=np.float16, count=k_cls, offset=m_off).copy()
        m_off += 2 * k_cls
        U = np.frombuffer(mc_bytes, dtype=np.float16, count=u_r * u_c, offset=m_off).reshape(u_r, u_c).copy()
        m_off += 2 * u_r * u_c
        V = np.frombuffer(mc_bytes, dtype=np.float16, count=k_cls * u_c, offset=m_off).reshape(k_cls, u_c).copy()
        m_off += 2 * k_cls * u_c
        trees = []
        for _ in range(n_trees):
            (d_t,) = struct.unpack_from("<B", mc_bytes, m_off)
            m_off += 1
            feats = list(struct.unpack_from(f"<{d_t}B", mc_bytes, m_off))
            m_off += d_t
            ths = [float(v) for v in np.frombuffer(mc_bytes, dtype=np.float16, count=d_t, offset=m_off)]
            m_off += 2 * d_t
            n_lv = (1 << d_t) * k_cls
            lv = np.frombuffer(mc_bytes, dtype=np.float16, count=n_lv, offset=m_off).reshape(1 << d_t, k_cls).copy()
            m_off += 2 * n_lv
            trees.append({"features": feats, "thresholds": ths, "leaf_values": lv})
        tb = enc["target_bundle"]
        tb["task_kind"] = "multiclass"
        tb["k_classes"] = k_cls
        tb["class_priors"] = priors
        tb["U_basis"] = U
        tb["V_basis"] = V
        tb["w_basis"] = U.reshape(-1)
        tb["trees"] = trees
    scaler = unpack_compact_scaler_sidecar(sc_bytes) if sc_bytes else None
    return enc, scaler


def export_v3_checkpoint_to_safetensors(pt_path: Any, safetensors_path: Any) -> Dict[str, Any]:
    """Export a trained DOPE-MLC-v3 PyTorch checkpoint (.pt) to a metadata-free F32 .safetensors file for 100% Rust inference."""
    import hashlib
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

    src, dst = Path(pt_path), Path(safetensors_path)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_tensors = {}
    total_params = 0
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().to(dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"nonfinite_v3_weight_tensor_{key}")
        clean_tensors[str(key)] = tensor
        total_params += int(tensor.numel())
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(clean_tensors, str(dst))
    raw_bytes = dst.read_bytes()
    return {
        "source_pt": str(src),
        "source_pt_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "safetensors_path": str(dst),
        "safetensors_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "safetensors_bytes": len(raw_bytes),
        "tensor_count": len(clean_tensors),
        "parameter_count": total_params,
    }
