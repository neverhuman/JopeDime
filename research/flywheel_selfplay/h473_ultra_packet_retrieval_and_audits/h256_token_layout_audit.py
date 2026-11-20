#!/usr/bin/env python3
"""H473 Part A1 (DV-01): Authoritative 92-Token Layout & Slice-Bug Telemetry Audit.

Defines `jope.h256-token-layout/v1` for `PreCondenseResidualBridge` (Arms F and G),
enforces contiguous non-overlapping `[0..92]` coverage plus separate 8-D non-token
`scale_side_features`, and quantifies the exact norm distortion produced by the
historical 93-token slice indices (`[0:32]`, `[32:44]`, `[44]`, `[45:77]`, `[77:93]`).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

H256_TOKEN_LAYOUT_SCHEMA = "jope.h256-token-layout/v1"
PRE_CONDENSE_NPZ_SHA256 = "5821dc6b5931d6b415f9bea3308d94e54a7b60c1cd9ebbb94fcebb9155fa4980"

AUTHORITATIVE_92TOK_BLOCKS: tuple[tuple[str, int, int, int, bool], ...] = (
    ("desc256", 0, 8, 256, False),
    ("task128", 8, 12, 128, False),
    ("dist128", 12, 16, 128, False),
    ("heads53_lognorms11", 16, 18, 64, False),
    ("col_gates64", 18, 20, 64, False),
    ("hidden256", 20, 28, 256, False),
    ("columns64", 28, 92, 1152, True),
)


def resolve_jope_roots(anchor_file: Path) -> tuple[Path, Path, Path]:
    here = anchor_file.resolve().parent
    candidate_repo = here.parents[2] if len(here.parents) >= 3 else here
    if (candidate_repo / "python" / "src" / "jope").is_dir():
        repo_root = candidate_repo
    else:
        repo_root = Path(os.environ.get("JOPE_REPO_ROOT", "/home/ubuntu/JopeDime")).resolve()
    flywheel_root = repo_root / "research" / "flywheel_selfplay"
    micro_python = Path(os.environ.get("JOPE_MICRO_PYTHON", "/home/ubuntu/JopeDimeMicro/python")).resolve()
    return repo_root, flywheel_root, micro_python


def digest_file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_strict_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
    staged = path.with_name(f".{path.name}.stage")
    staged.write_text(serialized, encoding="utf-8")
    staged.replace(path)


def available_memory_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 64.0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            return float(parts[1]) / (1024.0 * 1024.0)
    return 64.0


def verify_mem_available_gib(min_gib: float = 15.0, stage: str = "h473") -> float:
    avail = available_memory_gib()
    if avail < float(min_gib):
        raise RuntimeError(f"MemAvailable {avail:.2f} GiB below {min_gib:.2f} GiB floor at {stage}")
    return avail


def family_unweighted_mean(values: Sequence[float] | np.ndarray, families: Sequence[str]) -> float:
    grouped: dict[str, list[float]] = {}
    for val, fam in zip(values, families, strict=True):
        grouped.setdefault(str(fam), []).append(float(val))
    if not grouped:
        raise ValueError("family_unweighted_mean requires non-empty inputs")
    return float(sum(sum(vs) / len(vs) for vs in grouped.values()) / len(grouped))


def derange_across_families(families: Sequence[str], seed: int) -> np.ndarray:
    fam_arr = np.asarray([str(f) for f in families], dtype=object)
    n_items = len(fam_arr)
    uniq, counts = np.unique(fam_arr, return_counts=True)
    if n_items < 2 or len(uniq) < 2:
        raise ValueError("cross-family derangement requires at least two distinct families")
    rng = np.random.default_rng(int(seed))
    max_cnt = int(np.max(counts))
    if 2 * max_cnt <= n_items:
        order: list[int] = []
        for idx_f in rng.permutation(len(uniq)).tolist():
            members = np.where(fam_arr == uniq[idx_f])[0]
            order.extend(rng.permutation(members).tolist())
        ord_np = np.asarray(order, dtype=np.int64)
        donor = np.empty(n_items, dtype=np.int64)
        donor[ord_np] = np.roll(ord_np, max_cnt)
    else:
        donor = np.empty(n_items, dtype=np.int64)
        for pos in range(n_items):
            choices = np.where(fam_arr != fam_arr[pos])[0]
            donor[pos] = int(choices[int(rng.integers(0, len(choices)))])
    if np.any(fam_arr[donor] == fam_arr):
        raise AssertionError("donor shares family with target context")
    return donor


def build_h256_token_layout_spec() -> dict[str, Any]:
    blocks_map = {name: [start, end] for name, start, end, _, _ in AUTHORITATIVE_92TOK_BLOCKS}
    block_entries = [
        {
            "name": name,
            "token_range": [start, end],
            "token_count": end - start,
            "source_feature_width": src_w,
            "masked_by_col_mask_64": is_masked,
        }
        for name, start, end, src_w, is_masked in AUTHORITATIVE_92TOK_BLOCKS
    ]
    spec: dict[str, Any] = {
        "schema": H256_TOKEN_LAYOUT_SCHEMA,
        "bridge_module": "PreCondenseResidualBridge",
        "applicable_arms": [
            "F_FULL629_HIDDEN256_COL64X17",
            "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
        ],
        "total_tokens": 92,
        "token_dim": 32,
        "blocks": blocks_map,
        "block_entries": block_entries,
        "scale_side_features": {
            "name": "scale_side_features",
            "width": 8,
            "source_block_log_norms_width": 11,
            "is_sequence_token": False,
        },
        "historical_93tok_slice_bug": {
            "claimed_total_tokens": 93,
            "actual_total_tokens": 92,
            "misaligned_slices": {
                "vec256_0_32": {"slice": [0, 32], "actual_contents": "full629[0:20] + hidden256[20:28] + columns64[28:32]"},
                "lm71_32_44": {"slice": [32, 44], "actual_contents": "columns64[4:16]"},
                "scale_44": {"slice": [44, 45], "actual_contents": "columns64[16:17] (scale_side_features is non-token 8D)"},
                "hidden256_45_77": {"slice": [45, 77], "actual_contents": "columns64[17:49]"},
                "col_lm_77_93": {"slice": [77, 93], "actual_contents": "columns64[49:64] silently truncated to 15 tokens [77:92]"},
            },
        },
    }
    verify_contiguous_token_layout(spec)
    return spec


def verify_contiguous_token_layout(spec: Mapping[str, Any]) -> None:
    if spec.get("schema") != H256_TOKEN_LAYOUT_SCHEMA:
        raise ValueError(f"unexpected token layout schema: {spec.get('schema')}")
    total_tokens = int(spec.get("total_tokens", -1))
    token_dim = int(spec.get("token_dim", -1))
    if total_tokens != 92 or token_dim != 32:
        raise ValueError(f"expected 92 tokens x 32 dim, got {total_tokens} x {token_dim}")
    side = spec.get("scale_side_features", {})
    if int(side.get("width", -1)) != 8 or bool(side.get("is_sequence_token", True)):
        raise ValueError("scale_side_features must be width-8 non-token output")
    blocks = spec.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("blocks mapping required")
    intervals = [(str(k), int(v[0]), int(v[1])) for k, v in blocks.items()]
    intervals.sort(key=lambda row: row[1])
    if intervals[0][1] != 0:
        raise ValueError(f"token layout must start at 0, got {intervals[0][1]}")
    cursor = 0
    covered = 0
    for name, start, end in intervals:
        if start != cursor or end <= start:
            raise ValueError(f"non-contiguous or overlapping block {name}: [{start}, {end}] at cursor {cursor}")
        covered += end - start
        cursor = end
    if cursor != total_tokens or covered != total_tokens:
        raise ValueError(f"token coverage {cursor} != {total_tokens}")


def compute_token_slice_telemetry(
    tokens_92x32: np.ndarray,
    col_mask_64: np.ndarray,
    scale_side_8: np.ndarray,
) -> dict[str, Any]:
    tok = np.asarray(tokens_92x32, dtype=np.float64)
    cmask = np.asarray(col_mask_64, dtype=bool)
    side8 = np.asarray(scale_side_8, dtype=np.float64)
    if tok.ndim != 3 or tok.shape[1:] != (92, 32):
        raise ValueError(f"expected [N, 92, 32] tokens, got {tok.shape}")
    if cmask.shape != (tok.shape[0], 64):
        raise ValueError(f"expected [N, 64] col_mask_64, got {cmask.shape}")
    if side8.shape != (tok.shape[0], 8):
        raise ValueError(f"expected [N, 8] scale_side_8, got {side8.shape}")

    tok_l2 = np.linalg.norm(tok, axis=-1)  # [N, 92]
    col_tok_l2 = tok_l2[:, 28:92]  # [N, 64]
    inactive_mask = ~cmask
    inactive_max_abs = float(np.max(np.abs(tok[:, 28:92, :][inactive_mask]))) if np.any(inactive_mask) else 0.0

    active_col_means: list[float] = []
    for row_l2, row_m in zip(col_tok_l2, cmask, strict=True):
        active_col_means.append(float(np.mean(row_l2[row_m])) if np.any(row_m) else 0.0)
    col_active_mean = float(np.mean(active_col_means))

    auth: dict[str, float] = {}
    for name, start, end, _, is_masked in AUTHORITATIVE_92TOK_BLOCKS:
        if is_masked:
            auth["columns64_active_masked"] = round(col_active_mean, 6)
            auth["columns64_unmasked_all64"] = round(float(np.mean(col_tok_l2)), 6)
        else:
            auth[name] = round(float(np.mean(tok_l2[:, start:end])), 6)
    auth["full629_combined_0_20"] = round(float(np.mean(tok_l2[:, 0:20])), 6)
    scale8_l2 = float(np.mean(np.linalg.norm(side8, axis=-1)))
    auth["scale_side_features_8d_l2"] = round(scale8_l2, 6)

    bug_slice_77_93 = tok_l2[:, 77:93]
    buggy = {
        "claimed_total_tokens": 93,
        "actual_77_93_slice_length": int(bug_slice_77_93.shape[1]),
        "silent_truncation_tokens_lost": int(16 - bug_slice_77_93.shape[1]),
        "vec256_0_32_mean_l2": round(float(np.mean(tok_l2[:, 0:32])), 6),
        "lm71_32_44_mean_l2": round(float(np.mean(tok_l2[:, 32:44])), 6),
        "scale_44_mean_l2": round(float(np.mean(tok_l2[:, 44])), 6),
        "hidden256_45_77_mean_l2": round(float(np.mean(tok_l2[:, 45:77])), 6),
        "col_lm_77_93_clipped_mean_l2": round(float(np.mean(bug_slice_77_93)), 6),
    }

    def _err_pair(bug_val: float, true_val: float) -> dict[str, float]:
        diff = bug_val - true_val
        rel = diff / max(abs(true_val), 1e-12)
        return {
            "authoritative": round(true_val, 6),
            "historical_buggy": round(bug_val, 6),
            "abs_error_buggy_minus_true": round(diff, 6),
            "rel_error_frac": round(rel, 6),
        }

    return {
        "n_contexts": int(tok.shape[0]),
        "active_columns_mean": round(float(np.mean(np.sum(cmask, axis=1))), 4),
        "inactive_column_token_max_abs": round(inactive_max_abs, 8),
        "authoritative_92tok_l2": auth,
        "historical_buggy_93tok_l2": buggy,
        "slice_bug_error_quantification": {
            "desc256_vs_buggy_vec256_0_32": _err_pair(buggy["vec256_0_32_mean_l2"], auth["desc256"]),
            "hidden256_20_28_vs_buggy_45_77": _err_pair(buggy["hidden256_45_77_mean_l2"], auth["hidden256"]),
            "columns64_active_vs_buggy_77_93": _err_pair(
                buggy["col_lm_77_93_clipped_mean_l2"], auth["columns64_active_masked"]
            ),
            "scale_side8_vs_buggy_tok44": _err_pair(buggy["scale_44_mean_l2"], scale8_l2),
        },
    }


def _compute_raw_standardized_block_norms(
    raw_by_cid: Mapping[str, Mapping[str, np.ndarray]],
    fit_cids: Sequence[str],
    eval_cids: Sequence[str],
) -> dict[str, float]:
    dope_fit = np.stack([raw_by_cid[c]["dope629"] for c in fit_cids], axis=0).astype(np.float64)
    hid_fit = np.stack([raw_by_cid[c]["hidden256"] for c in fit_cids], axis=0).astype(np.float64)
    d_mu, d_sd = dope_fit.mean(axis=0), np.maximum(dope_fit.std(axis=0), 1e-4)
    h_mu, h_sd = hid_fit.mean(axis=0), np.maximum(hid_fit.std(axis=0), 1e-4)

    active_cols_fit = [
        raw_by_cid[c]["col_lm_64x17"][raw_by_cid[c]["col_mask_64"]]
        for c in fit_cids
        if np.any(raw_by_cid[c]["col_mask_64"])
    ]
    c_cat = np.concatenate(active_cols_fit, axis=0).astype(np.float64)
    c_mu, c_sd = c_cat.mean(axis=0), np.maximum(c_cat.std(axis=0), 1e-4)

    dope_ev = np.clip(
        (np.stack([raw_by_cid[c]["dope629"] for c in eval_cids], axis=0).astype(np.float64) - d_mu) / d_sd,
        -15.0,
        15.0,
    )
    hid_ev = np.clip(
        (np.stack([raw_by_cid[c]["hidden256"] for c in eval_cids], axis=0).astype(np.float64) - h_mu) / h_sd,
        -15.0,
        15.0,
    )
    col_active_l2: list[float] = []
    for cid in eval_cids:
        cm = raw_by_cid[cid]["col_mask_64"]
        if np.any(cm):
            cz = np.clip((raw_by_cid[cid]["col_lm_64x17"][cm].astype(np.float64) - c_mu) / c_sd, -15.0, 15.0)
            col_active_l2.append(float(np.mean(np.linalg.norm(cz, axis=-1))))
        else:
            col_active_l2.append(0.0)

    return {
        "desc256_z_l2": round(float(np.mean(np.linalg.norm(dope_ev[:, 0:256], axis=-1))), 6),
        "task128_z_l2": round(float(np.mean(np.linalg.norm(dope_ev[:, 256:384], axis=-1))), 6),
        "dist128_z_l2": round(float(np.mean(np.linalg.norm(dope_ev[:, 384:512], axis=-1))), 6),
        "heads53_z_l2": round(float(np.mean(np.linalg.norm(dope_ev[:, 512:565], axis=-1))), 6),
        "col_gates64_z_l2": round(float(np.mean(np.linalg.norm(dope_ev[:, 565:629], axis=-1))), 6),
        "hidden256_z_l2": round(float(np.mean(np.linalg.norm(hid_ev, axis=-1))), 6),
        "col_lm_64x17_active_masked_z_l2": round(float(np.mean(col_active_l2)), 6),
    }


def _eval_bridge_checkpoint_tokens(
    bridge_cls: Any,
    scaler_cls: Any,
    arm_name: str,
    state_dict: Mapping[str, Any],
    scaler_dict: Mapping[str, Any],
    raw_by_cid: Mapping[str, Mapping[str, np.ndarray]],
    eval_cids: Sequence[str],
) -> dict[str, Any]:
    device = torch.device("cpu")
    scaler = scaler_cls(
        dope629_mean=np.asarray(scaler_dict["dope629_mean"], dtype=np.float32),
        dope629_std=np.asarray(scaler_dict["dope629_std"], dtype=np.float32),
        hidden256_mean=np.asarray(scaler_dict["hidden256_mean"], dtype=np.float32),
        hidden256_std=np.asarray(scaler_dict["hidden256_std"], dtype=np.float32),
        col18_mean=np.asarray(scaler_dict["col18_mean"], dtype=np.float32),
        col18_std=np.asarray(scaler_dict["col18_std"], dtype=np.float32),
    )
    bridge = bridge_cls(arm=arm_name, d_token=32, n_heads=4).to(device)
    bridge.load_state_dict(state_dict, strict=False)
    bridge.eval()

    tok_rows = np.empty((len(eval_cids), 92, 32), dtype=np.float32)
    mask_rows = np.empty((len(eval_cids), 64), dtype=bool)
    side_rows = np.empty((len(eval_cids), 8), dtype=np.float32)
    with torch.no_grad():
        for idx, cid in enumerate(eval_cids):
            entry = raw_by_cid[cid]
            ct = scaler.transform_context(entry, device)
            tok_t, _, side_t = bridge.encode_context_tokens(ct)
            tok_rows[idx] = tok_t[0].cpu().numpy()
            mask_rows[idx] = entry["col_mask_64"]
            side_rows[idx] = side_t[0].cpu().numpy()
    return compute_token_slice_telemetry(tok_rows, mask_rows, side_rows)


def run_dv01_h256_layout_audit(
    h256_dir: Path,
    train_cids: Sequence[str],
    *,
    layout_out_path: Path | None = None,
    val_cids: Sequence[str] | None = None,
) -> dict[str, Any]:
    verify_mem_available_gib(15.0, "dv01_layout_audit")
    spec = build_h256_token_layout_spec()
    if layout_out_path is not None:
        write_strict_json(layout_out_path, spec)

    npz_path = h256_dir / "cache" / "pre_condense_context_features.npz"
    actual_sha = digest_file_sha256(npz_path)
    if actual_sha != PRE_CONDENSE_NPZ_SHA256:
        raise ValueError(f"pre_condense_context_features.npz SHA mismatch: {actual_sha}")

    _, flywheel_root, _ = resolve_jope_roots(Path(__file__))
    h256_code_dir = str(flywheel_root / "h256_pre_condense_residual_flywheel")
    if h256_code_dir not in sys.path:
        sys.path.insert(0, h256_code_dir)
    from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402

    with np.load(npz_path, allow_pickle=False) as zf:
        cids_all = [str(c) for c in zf["cids"].tolist()]
        dope629 = zf["dope629"].astype(np.float32)
        hidden256 = zf["hidden256"].astype(np.float32)
        col_lm = zf["col_lm_64x17"].astype(np.float32)
        col_mask = zf["col_mask_64"].astype(bool)
        tok10 = zf["tokens_10x32"].astype(np.float32)
        msk10 = zf["mask_10"].astype(bool)

    raw_by_cid = {
        cid: {
            "dope629": dope629[i],
            "hidden256": hidden256[i],
            "col_lm_64x17": col_lm[i],
            "col_mask_64": col_mask[i],
            "tokens_10x32": tok10[i],
            "mask_10": msk10[i],
        }
        for i, cid in enumerate(cids_all)
    }

    ckpt_dir = h256_dir / "checkpoints"
    checkpoints_telem: dict[str, Any] = {}
    ckpt_specs = (
        (
            "stage1_s3_all15796_encoder",
            ckpt_dir / "stage1_full_s3_reg_all15796_encoder.pt",
            "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
            "encoder_state_dict",
            "scaler_all15796",
        ),
        (
            "stage2_arm_F_seed256101_fold0",
            ckpt_dir / "stage2_F_FULL629_HIDDEN256_COL64X17_seed256101_fold0.pt",
            "F_FULL629_HIDDEN256_COL64X17",
            "state_dict",
            "scaler",
        ),
        (
            "stage2_arm_G_seed256101_fold0",
            ckpt_dir / "stage2_G_FULL_S3_REG_7SPLIT_11121_PRETRAINED_seed256101_fold0.pt",
            "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
            "state_dict",
            "scaler",
        ),
    )
    for label, cp_path, arm_id, sd_key, sc_key in ckpt_specs:
        if cp_path.is_file():
            ck = torch.load(cp_path, map_location="cpu", weights_only=False)
            entry_out: dict[str, Any] = {
                "checkpoint_sha256": digest_file_sha256(cp_path),
                "dl_train1219": _eval_bridge_checkpoint_tokens(
                    PreCondenseResidualBridge, FoldScaler, arm_id, ck[sd_key], ck[sc_key], raw_by_cid, train_cids
                ),
            }
            if val_cids is not None:
                all_1310 = list(train_cids) + list(val_cids)
                entry_out["dl_val91"] = _eval_bridge_checkpoint_tokens(
                    PreCondenseResidualBridge, FoldScaler, arm_id, ck[sd_key], ck[sc_key], raw_by_cid, val_cids
                )
                entry_out["all_1310_historical_repro"] = _eval_bridge_checkpoint_tokens(
                    PreCondenseResidualBridge, FoldScaler, arm_id, ck[sd_key], ck[sc_key], raw_by_cid, all_1310
                )
            checkpoints_telem[label] = entry_out

    raw_norms: dict[str, Any] = {
        "dl_train1219": _compute_raw_standardized_block_norms(raw_by_cid, train_cids, train_cids),
    }
    if val_cids is not None:
        raw_norms["dl_val91_fit_on_train"] = _compute_raw_standardized_block_norms(raw_by_cid, train_cids, val_cids)

    return {
        "schema": H256_TOKEN_LAYOUT_SCHEMA,
        "pre_condense_npz_sha256": actual_sha,
        "contiguous_0_92_verified": True,
        "total_tokens": 92,
        "token_dim": 32,
        "scale_side_features_width": 8,
        "raw_standardized_feature_norms": raw_norms,
        "checkpoint_token_telemetry": checkpoints_telem,
    }
