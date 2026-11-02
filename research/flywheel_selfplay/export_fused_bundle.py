#!/usr/bin/env python3
"""RES-06: deterministic, unsigned h40b serving export from pinned RES-02 copies.

Usage: python export_fused_bundle.py --metadata manifest-template.json --output NEW_DIR
The template uses the existing fused-manifest schema, omitting bundle_digest and
both computed weight hashes. Qualification/redistribution declarations belong to
the caller; exporting never establishes parity, qualification, or release trust.
No original host asset, dataset, optimizer, network module, or GPU is used.

BundleDigest: SHA256(domain || frame(canonical unsigned manifest) || u64(file_count)
|| [frame(UTF8 relative name) || frame(exact bytes)] sorted by name), where frame
is u64 little-endian byte length followed by bytes. Domain is
b"jope.fused-bundle/v1\0". Only bundle_digest is omitted from the manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import jsonschema
import numpy as np
import torch

FROZEN_DOPE_COMPOSITE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas/jope.fused-bundle.v1.schema.json"
CUSTODY_PATH = Path(__file__).resolve().parent / "custody"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_PINNED_BYTES = 128 * 1024 * 1024
PAYLOADS = {"dope.safetensors", "jope.safetensors", "featurizer.json", "bank.jsonl.zst"}
MODEL_PRIOR_BANK_METADATA = {
    "role": "model_prior",
    "coordinate_space": "standardized_lm48_l2_unit",
}
CONSTANTS = dict(
    scale=0.4208, ctx_feat_dim=584, revealed=4, proposals=8, proposals_explore=16, admit=4, elite_pool=48
)
NORMALIZERS = (
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
# Fixed architecture inventory, independently transcribed from GOV-03 r782
# checkpoint 427b0d46...f3a4af0c and its pinned cpu_exact_fused.py constructor.
# It is NEVER inferred from the candidate being checked. Shapes include buffers.
DOPE_SHAPES = {
    "cov_net.net.0.bias": (128,),
    "cov_net.net.0.weight": (128, 192),
    "cov_net.net.2.bias": (128,),
    "cov_net.net.2.weight": (128,),
    "cov_net.net.3.bias": (64,),
    "cov_net.net.3.weight": (64, 128),
    "router.backbone.0.bias": (256,),
    "router.backbone.0.weight": (256, 140),
    "router.backbone.2.bias": (256,),
    "router.backbone.2.weight": (256,),
    "router.backbone.3.bias": (256,),
    "router.backbone.3.weight": (256, 256),
    "router.backbone.5.bias": (256,),
    "router.backbone.5.weight": (256,),
    "router.bb_recon_head.bias": (16,),
    "router.bb_recon_head.weight": (16, 256),
    "router.dist_latent_head.bias": (128,),
    "router.dist_latent_head.weight": (128, 256),
    "router.gate_head.bias": (10,),
    "router.gate_head.weight": (10, 256),
    "router.input_norm.bias": (140,),
    "router.input_norm.weight": (140,),
    "router.inter_attn.bilinear_u.bias": (128,),
    "router.inter_attn.bilinear_u.weight": (128, 256),
    "router.inter_attn.bilinear_v.bias": (128,),
    "router.inter_attn.bilinear_v.weight": (128, 256),
    "router.inter_attn.norm.bias": (256,),
    "router.inter_attn.norm.weight": (256,),
    "router.inter_attn.out_proj.bias": (256,),
    "router.inter_attn.out_proj.weight": (256, 384),
    "router.inter_attn.qkv.bias": (96,),
    "router.inter_attn.qkv.weight": (96, 32),
    "router.inter_attn.token_proj.bias": (32,),
    "router.inter_attn.token_proj.weight": (32, 32),
    "router.landmark_recon_head.bias": (23,),
    "router.landmark_recon_head.weight": (23, 256),
    "router.level_embed.weight": (3, 256),
    "router.small_embed": (256,),
    "router.task_latent_head.bias": (128,),
    "router.task_latent_head.weight": (128, 256),
    "router.v3_gate_head.bias": (4,),
    "router.v3_gate_head.weight": (4, 256),
    "router.v3_inter_norm.bias": (116,),
    "router.v3_inter_norm.weight": (116,),
    "target_net.col_gate_head.bias": (1,),
    "target_net.col_gate_head.weight": (1, 145),
    "target_net.fc1.bias": (128,),
    "target_net.fc1.weight": (128, 192),
    "target_net.fc2.bias": (128,),
    "target_net.fc2.weight": (128, 128),
    "target_net.film_beta.bias": (64,),
    "target_net.film_beta.weight": (64, 128),
    "target_net.film_gamma.bias": (64,),
    "target_net.film_gamma.weight": (64, 128),
    "target_net.out_head.bias": (1,),
    "target_net.out_head.weight": (1, 128),
}

ANCHOR_SHAPES = {
    "alpha_attn": (),
    "alpha_prior": (),
    "alpha_struct_kernel": (),
    "alpha_ucb": (),
    "cross_attn.in_proj_bias": (384,),
    "cross_attn.in_proj_weight": (384, 128),
    "cross_attn.out_proj.bias": (128,),
    "cross_attn.out_proj.weight": (128, 128),
    "ctx_block.fc_gate.bias": (256,),
    "ctx_block.fc_gate.weight": (256, 128),
    "ctx_block.fc_out.bias": (128,),
    "ctx_block.fc_out.weight": (128, 256),
    "ctx_block.fc_val.bias": (256,),
    "ctx_block.fc_val.weight": (256, 128),
    "ctx_block.norm.bias": (128,),
    "ctx_block.norm.weight": (128,),
    "ctx_in.0.bias": (584,),
    "ctx_in.0.weight": (584,),
    "ctx_in.1.bias": (128,),
    "ctx_in.1.weight": (128, 584),
    "evolvability_head.0.bias": (256,),
    "evolvability_head.0.weight": (256,),
    "evolvability_head.1.bias": (64,),
    "evolvability_head.1.weight": (64, 256),
    "evolvability_head.3.bias": (1,),
    "evolvability_head.3.weight": (1, 64),
    "gate_barrier_weight": (),
    "gate_head.0.bias": (258,),
    "gate_head.0.weight": (258,),
    "gate_head.1.bias": (64,),
    "gate_head.1.weight": (64, 258),
    "gate_head.3.bias": (1,),
    "gate_head.3.weight": (1, 64),
    "history_token_in.0.bias": (128,),
    "history_token_in.0.weight": (128, 132),
    "history_token_in.1.bias": (128,),
    "history_token_in.1.weight": (128,),
    "landmark_router.0.bias": (64,),
    "landmark_router.0.weight": (64,),
    "landmark_router.1.bias": (64,),
    "landmark_router.1.weight": (64, 64),
    "landmark_router.3.bias": (16,),
    "landmark_router.3.weight": (16, 64),
    "prior_head.0.bias": (256,),
    "prior_head.0.weight": (256,),
    "prior_head.1.bias": (128,),
    "prior_head.1.weight": (128, 256),
    "prior_head.3.bias": (1,),
    "prior_head.3.weight": (1, 128),
    "prog_block.fc_gate.bias": (256,),
    "prog_block.fc_gate.weight": (256, 128),
    "prog_block.fc_out.bias": (128,),
    "prog_block.fc_out.weight": (128, 256),
    "prog_block.fc_val.bias": (256,),
    "prog_block.fc_val.weight": (256, 128),
    "prog_block.norm.bias": (128,),
    "prog_block.norm.weight": (128,),
    "prog_in.0.bias": (558,),
    "prog_in.0.weight": (558,),
    "prog_in.1.bias": (128,),
    "prog_in.1.weight": (128, 558),
    "traj_out.0.bias": (260,),
    "traj_out.0.weight": (260,),
    "traj_out.1.bias": (64,),
    "traj_out.1.weight": (64, 260),
    "traj_out.3.bias": (1,),
    "traj_out.3.weight": (1, 64),
}

ADAPTER_SHAPES = {
    "cand_query_proj.0.bias": (32,),
    "cand_query_proj.0.weight": (32, 66),
    "cand_query_proj.1.bias": (32,),
    "cand_query_proj.1.weight": (32,),
    "ctx_bilinear_u.weight": (16, 32),
    "emb_kernel_scale_raw": (),
    "hist_token_proj.0.bias": (32,),
    "hist_token_proj.0.weight": (32, 60),
    "hist_token_proj.1.bias": (32,),
    "hist_token_proj.1.weight": (32,),
    "mm_cross_attn.in_proj_bias": (96,),
    "mm_cross_attn.in_proj_weight": (96, 32),
    "mm_cross_attn.out_proj.bias": (32,),
    "mm_cross_attn.out_proj.weight": (32, 32),
    "mm_prior_out.0.bias": (24,),
    "mm_prior_out.0.weight": (24, 50),
    "mm_prior_out.2.bias": (1,),
    "mm_prior_out.2.weight": (1, 24),
    "mm_prior_scale": (),
    "modality_gate_logits": (10,),
    "modality_norm.bias": (32,),
    "modality_norm.weight": (32,),
    "modality_type_emb": (10, 32),
    "null_hist_token": (1, 1, 32),
    "null_kernel_w0_raw": (),
    "parent_delta_net.0.bias": (48,),
    "parent_delta_net.0.weight": (48, 114),
    "parent_delta_net.1.bias": (48,),
    "parent_delta_net.1.weight": (48,),
    "parent_delta_net.3.bias": (1,),
    "parent_delta_net.3.weight": (1, 48),
    "prog_412_bottleneck.0.bias": (16,),
    "prog_412_bottleneck.0.weight": (16, 412),
    "prog_412_bottleneck.1.bias": (16,),
    "prog_412_bottleneck.1.weight": (16,),
    "prog_bilinear_v.weight": (16, 24),
    "prog_mean": (412,),
    "prog_std": (412,),
    "sem_kernel_scale_raw": (),
    "traj_attn_scale": (),
    "traj_cross_attn.in_proj_bias": (96,),
    "traj_cross_attn.in_proj_weight": (96, 32),
    "traj_cross_attn.out_proj.bias": (32,),
    "traj_cross_attn.out_proj.weight": (32, 32),
    "traj_delta_head.0.bias": (24,),
    "traj_delta_head.0.weight": (24, 67),
    "traj_delta_head.2.bias": (1,),
    "traj_delta_head.2.weight": (1, 24),
    "traj_kernel_blend": (),
}

BUFFERS_SHAPES = {
    "anchor_alphas_4": (1, 4, 1, 1),
    "anchor_eye_65": (65, 65),
    "bank_feat412": (885, 412),
    "bank_lm48_z_881": (881, 48),
    "bank_ns_mask_flt": (885,),
    "bank_op_desc": (885, 8),
    "bank_op_onehot": (885, 8),
    "bank_proj16": (885, 16),
    "bank_sem24": (885, 24),
    "bank_sos_2": (885, 2),
    "dope_to_lm48.bias": (48,),
    "dope_to_lm48.weight": (48, 629),
    "frac_13": (13,),
    "gcv_lambdas": (6,),
    "grid_idx_13": (13,),
    "grid_idx_257": (257,),
    "landmark_bridge.blend_alpha": (),
    "landmark_bridge.proj.bias": (256,),
    "landmark_bridge.proj.weight": (256, 256),
    "lm80_mu": (1, 80),
    "lm80_sd": (1, 80),
    "mask_10_base": (1, 10),
    "mu_bb20": (1, 20),
    "mu_dm24": (1, 24),
    "mu_lm48": (1, 48),
    "nolm_bridge.blend_alpha": (),
    "nolm_bridge.proj.bias": (256,),
    "nolm_bridge.proj.weight": (256, 256),
    "nolm_use_distilled_lm48": (),
    "prior10_default_median": (),
    "prior10_feat_mean_150": (1, 1, 150),
    "prior10_feat_std_150": (1, 1, 150),
    "prior10_w_150": (150,),
    "priorfp_default_median": (),
    "priorfp_feat_mean_150": (1, 1, 150),
    "priorfp_feat_std_150": (1, 1, 150),
    "priorfp_w_150": (150,),
    "prog_mean_412": (1, 1, 412),
    "prog_std_412": (1, 1, 412),
    "qhat_b_0": (),
    "qhat_mu_x_93": (1, 1, 93),
    "qhat_sd_x_93": (1, 1, 93),
    "qhat_w_93": (93,),
    "reg_eye_basis_max": (193, 193),
    "sd_bb20": (1, 20),
    "sd_dm24": (1, 24),
    "sd_lm48": (1, 48),
    "sos_jitter_1x1x256": (1, 1, 256),
    "sos_struct_mat_256x24": (256, 24),
    "sos_struct_val_256": (256,),
    "step2_lm_mean_1x256": (1, 256),
    "step2_lm_std_1x256": (1, 256),
    "step2_svd_basis_256x48": (256, 48),
    "svd_slot0_m0_active_mask": (1, 184),
    "svd_slot0_m0_effective_proj_mat": (184, 32),
    "svd_slot0_m0_mu": (1, 184),
    "svd_slot0_m0_sd_safe": (1, 184),
    "svd_slot1_m1_active_mask": (1, 48),
    "svd_slot1_m1_effective_proj_mat": (48, 32),
    "svd_slot1_m1_mu": (1, 48),
    "svd_slot1_m1_sd_safe": (1, 48),
    "svd_slot2_m2_active_mask": (1, 629),
    "svd_slot2_m2_effective_proj_mat": (629, 32),
    "svd_slot2_m2_mu": (1, 629),
    "svd_slot2_m2_sd_safe": (1, 629),
    "svd_slot3_dm16_active_mask": (1, 24),
    "svd_slot3_dm16_effective_proj_mat": (24, 32),
    "svd_slot3_dm16_mu": (1, 24),
    "svd_slot3_dm16_sd_safe": (1, 24),
    "svd_slot3_in16_active_mask": (1, 256),
    "svd_slot3_in16_effective_proj_mat": (256, 32),
    "svd_slot3_in16_mu": (1, 256),
    "svd_slot3_in16_sd_safe": (1, 256),
    "svd_slot4_d629_mid_active_mask": (1, 629),
    "svd_slot4_d629_mid_effective_proj_mat": (629, 32),
    "svd_slot4_d629_mid_mu": (1, 629),
    "svd_slot4_d629_mid_sd_safe": (1, 629),
    "svd_slot5_d629_tail_active_mask": (1, 629),
    "svd_slot5_d629_tail_effective_proj_mat": (629, 32),
    "svd_slot5_d629_tail_mu": (1, 629),
    "svd_slot5_d629_tail_sd_safe": (1, 629),
    "svd_slot6_d280_res_active_mask": (1, 256),
    "svd_slot6_d280_res_effective_proj_mat": (256, 32),
    "svd_slot6_d280_res_mu": (1, 256),
    "svd_slot6_d280_res_sd_safe": (1, 256),
    "svd_slot7_lm_fine_active_mask": (1, 184),
    "svd_slot7_lm_fine_effective_proj_mat": (184, 32),
    "svd_slot7_lm_fine_mu": (1, 184),
    "svd_slot7_lm_fine_sd_safe": (1, 184),
    "svd_slot8_v3_full92_active_mask": (1, 92),
    "svd_slot8_v3_full92_effective_proj_mat": (92, 32),
    "svd_slot8_v3_full92_mu": (1, 92),
    "svd_slot8_v3_full92_sd_safe": (1, 92),
    "svd_slot9_kron_active_mask": (1, 64),
    "svd_slot9_kron_effective_proj_mat": (64, 32),
    "svd_slot9_kron_mu": (1, 64),
    "svd_slot9_kron_sd_safe": (1, 64),
    "tau_grid_257": (257,),
    "w1_sign_matrix_412x16": (412, 16),
}

TENSOR_SHAPES = dict(BUFFERS_SHAPES)
for _prefix, _shapes in (
    ("dope_model.", DOPE_SHAPES),
    ("gamenet_anchor.", ANCHOR_SHAPES),
    ("adapter_10tok.anchor.", ANCHOR_SHAPES),
    ("adapter_10tok.", ADAPTER_SHAPES),
):
    TENSOR_SHAPES.update({_prefix + name: shape for name, shape in _shapes.items()})


class ExportError(ValueError):
    """Input cannot establish a complete frozen serving bundle."""


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(data):
    def invalid(value):
        raise ExportError(f"non-finite JSON literal: {value}")

    return json.loads(data, object_pairs_hook=_object, parse_constant=invalid)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _array(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return array.astype(array.dtype.newbyteorder("<"), copy=False)


def dope_composite(state):
    def tensor_hash(tensor):
        array = _array(tensor)
        return sha256(f"{array.dtype.str}:{list(array.shape)}:".encode() + array.tobytes())

    parameters = "".join(
        f"{name}={tensor_hash(state[name])}\n" for name in sorted(state) if name.startswith("dope_model.")
    )
    buffers = "".join(f"normalizer.{name}={tensor_hash(state[name])}\n" for name in NORMALIZERS)
    return sha256(f"params:{sha256(parameters.encode())}|buffers:{sha256(buffers.encode())}".encode())


def validate_state(state):
    if not isinstance(state, dict) or set(state) != set(TENSOR_SHAPES):
        missing = sorted(set(TENSOR_SHAPES) - set(state)) if isinstance(state, dict) else []
        extra = sorted(set(state) - set(TENSOR_SHAPES)) if isinstance(state, dict) else []
        raise ExportError(f"complete tensor manifest required: missing={missing}, extra={extra}")
    for name, shape in TENSOR_SHAPES.items():
        value = state[name]
        dtype = torch.int64 if name in ("grid_idx_13", "grid_idx_257") else torch.float32
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ExportError(f"tensor shape/dtype mismatch: {name}, expected {shape}/{dtype}")
        if value.device.type != "cpu" or not torch.isfinite(value).all().item():
            raise ExportError(f"tensor must be finite and CPU-resident: {name}")
    if dope_composite(state) != FROZEN_DOPE_COMPOSITE:
        raise ExportError("frozen DOPE parameter/buffer composite mismatch")
    for name in ANCHOR_SHAPES:
        if not torch.equal(state["gamenet_anchor." + name], state["adapter_10tok.anchor." + name]):
            raise ExportError(f"adapter frozen anchor aliases disagree: {name}")


def safetensors_bytes(state):
    """Canonical SafeTensors v1 framing; no timestamps, metadata, pickle or aliases."""
    header, chunks, offset = {}, [], 0
    for name in sorted(state):
        value = state[name]
        data = _array(value).tobytes()
        header[name] = {
            "dtype": "I64" if value.dtype == torch.int64 else "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        chunks.append(data)
    encoded = canonical_json(header)
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)


def compute_bundle_digest(manifest, files):
    unsigned = dict(manifest)
    unsigned.pop("bundle_digest", None)
    digest = hashlib.sha256(b"jope.fused-bundle/v1\0")

    def frame(data):
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)

    frame(canonical_json(unsigned))
    digest.update(struct.pack("<Q", len(files)))
    for name in sorted(files):
        if (
            name == "manifest.json"
            or not isinstance(name, str)
            or not name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or name.startswith("/")
            or "\\" in name
        ):
            raise ExportError("bundle payload name must be canonical and relative")
        frame(name.encode("utf-8"))
        frame(files[name])
    return "sha256:" + digest.hexdigest()


def _assert_no_symlink_path(path):
    """Reject symlinks in every existing component without resolving them away."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ExportError(f"missing custody path component: {current}") from error
        if stat.S_ISLNK(mode):
            raise ExportError(f"symlink in custody path: {current}")
    return path


def _external_rows(custody):
    """Validate EXTERNAL.json against the original custody manifest."""
    verifier_path = Path(__file__).resolve().parent / "custody/verify_custody.py"
    spec = importlib.util.spec_from_file_location("_jope_custody_verifier", verifier_path)
    if spec is None or spec.loader is None:
        raise ExportError("cannot load custody verifier")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    try:
        manifest_path = _assert_no_symlink_path(Path(custody) / "MANIFEST.json")
        external_path = _assert_no_symlink_path(Path(custody) / "EXTERNAL.json")
        manifest = json.loads(manifest_path.read_bytes())
        document = json.loads(external_path.read_bytes())
        expected, _ = verifier._manifest_entries(manifest)
        entries = verifier._external_entries(document, expected)
        return entries
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError, ValueError, subprocess.CalledProcessError) as error:
        raise ExportError(f"invalid external custody pins: {error}") from error


def _read_checked(stream, expected, size, label):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ExportError("invalid pinned SHA-256")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0 or size > MAX_PINNED_BYTES):
        raise ExportError("invalid or oversized pinned byte count")
    chunks = []
    total = 0
    while chunk := stream.read(min(1024 * 1024, MAX_PINNED_BYTES + 1 - total)):
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_PINNED_BYTES:
            raise ExportError(f"oversized pinned input: {label}")
    data = b"".join(chunks)
    if sha256(data) != expected or (size is not None and len(data) != size):
        raise ExportError(f"pinned input hash/length mismatch: {label}")
    return data


def _cache_bytes(root, expected, size):
    root = Path(root)
    if not root.is_absolute():
        raise ExportError("external cache root must be absolute")
    target = root / "sha256" / expected
    for component in reversed(target.parents):
        try:
            mode = component.lstat().st_mode
        except OSError as error:
            raise ExportError("missing external cache path") from error
        if stat.S_ISLNK(mode):
            raise ExportError("symlink in external cache path")
        if component == root and not stat.S_ISDIR(mode):
            raise ExportError("external cache root is not a directory")
    try:
        mode = target.lstat().st_mode
    except OSError as error:
        raise ExportError("missing external cache payload") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ExportError("external cache payload is not a regular file")
    if target.stat().st_size > MAX_PINNED_BYTES or (size is not None and target.stat().st_size != size):
        raise ExportError("external cache payload size differs from pin")
    with target.open("rb") as stream:
        return _read_checked(stream, expected, size, target.name)


def _git_bytes(row, expected, size):
    commit = row["recovery_commit"]
    git_path = row["git_path"]
    object_id = row["git_object_id"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", commit)
        or not re.fullmatch(r"[0-9a-f]{40}", object_id)
        or git_path != "research/flywheel_selfplay/custody/files/" + row["path"].lstrip("/")
    ):
        raise ExportError("invalid immutable external Git pin")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{commit}:{git_path}"],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ExportError("pinned Git recovery commit/path unavailable") from error
    if actual != object_id:
        raise ExportError("external Git recovery object identity differs")
    command = ["git", "-C", str(REPO_ROOT), "cat-file", "blob", row["git_object_id"]]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        data = _read_checked(process.stdout, expected, size, row["git_path"])
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise ExportError(f"cannot read pinned Git recovery blob: {error.strip()}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return data


def _pinned_bytes(custody, original, expected, size=None, external_root=None, external_rows=None):
    parts = PurePosixPath(original).parts
    if (
        not isinstance(original, str)
        or not original.startswith("/")
        or any(part in ("", ".", "..") for part in original.split("/")[1:])
    ):
        raise ExportError("unsafe custody source path")
    relative = Path(*parts[1:])
    if external_rows is None:
        external_rows = _external_rows(custody)
    external = external_rows.get(relative)
    if external is not None:
        if external["sha256"] != expected or external["bytes"] != size:
            raise ExportError("external pin differs from requested custody pin")
        if external_root is not None:
            return _cache_bytes(external_root, expected, size)
        return _git_bytes(external, expected, size)

    path = Path(custody) / "files" / relative
    _assert_no_symlink_path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ExportError(f"missing pinned input: {path.name}") from error
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size > MAX_PINNED_BYTES:
        raise ExportError(f"missing or oversized pinned input: {path.name}")
    with path.open("rb") as stream:
        return _read_checked(stream, expected, size, path.name)


def load_inputs(custody=CUSTODY_PATH, external_root=None):
    """Read committed custody copies only; restricted pickle load after byte pins."""
    custody = Path(custody)
    manifest_path = _assert_no_symlink_path(custody / "MANIFEST.json")
    checkpoint_path = _assert_no_symlink_path(custody / "CHECKPOINT.json")
    manifest_data = manifest_path.read_bytes()
    checkpoint_data = checkpoint_path.read_bytes()
    manifest, pin = read_json(manifest_data), read_json(checkpoint_data)
    external_rows = _external_rows(custody)
    checkpoint_hash = pin["checkpoint"]["sha256"]
    if (
        manifest["checkpoint_sha256"] != checkpoint_hash
        or pin["dope"]["loaded_weights_sha256"] != FROZEN_DOPE_COMPOSITE
    ):
        raise ExportError("custody checkpoint/DOPE pins disagree")
    sources = {
        "custody/MANIFEST.json": sha256(manifest_data),
        "custody/CHECKPOINT.json": sha256(checkpoint_data),
    }

    def asset(basename, collection="assets"):
        entries = [row for row in manifest[collection] if PurePosixPath(row["path"]).name == basename]
        if len(entries) != 1:
            raise ExportError(f"exactly one pinned source required: {basename}")
        row = entries[0]
        data = _pinned_bytes(custody, row["path"], row["sha256"], row["bytes"], external_root, external_rows)
        sources[basename] = sha256(data)
        return data

    checkpoint = asset(PurePosixPath(pin["checkpoint"]["path"]).name)
    if sha256(checkpoint) != checkpoint_hash or len(checkpoint) != pin["checkpoint"]["bytes"]:
        raise ExportError("checkpoint asset disagrees with selected pin")
    sources["checkpoint.pt"] = checkpoint_hash
    loaded = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
    if (
        loaded["dope_sha256"] != FROZEN_DOPE_COMPOSITE
        or loaded["round"] != pin["checkpoint"]["training_round"]
    ):
        raise ExportError("checkpoint lineage metadata disagrees with custody pin")
    # NumPy reconstruction is needed only by this byte-pinned historical asset.
    # Never fall back to weights_only=False or import its original network.
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    reconstruct = numpy_core.multiarray._reconstruct
    numpy_globals = [reconstruct, np.ndarray, np.dtype]
    if reconstruct.__module__ != "numpy.core.multiarray":
        numpy_globals.append((reconstruct, "numpy.core.multiarray._reconstruct"))
    numpy_globals.extend(type(np.dtype(kind)) for kind in ("float32", "float64", "int64", "int32", "bool"))
    projection = asset("frozen_stage1_stage2_inference_params_v3.pt")
    with torch.serialization.safe_globals(numpy_globals):
        frozen = torch.load(io.BytesIO(projection), map_location="cpu", weights_only=True)
    feature_names = read_json(asset("program-feature-names.json"))
    asset("cpu_exact_fused.py", "import_files")
    asset("rank_prepare.py", "import_files")
    return dict(
        state=loaded["state_dict"],
        champions=frozen["champ_templates_881"],
        feature_names=feature_names,
        source_hashes=sources,
    )


def bank_bytes(champions, state):
    if not isinstance(champions, list) or len(champions) != 881:
        raise ExportError("complete pinned champion bank requires 881 records")
    lines = []
    for index, champion in enumerate(champions):
        if set(champion) != {"cid", "family_id", "dsl", "program_sha256", "feat412", "proj16", "lm48_z"}:
            raise ExportError(f"champion must contain only the pinned row-free fields: {index}")
        family = champion["family_id"].removeprefix("fam-")
        dsl = champion["dsl"]
        if (
            not re.fullmatch("[0-9a-f]{64}", family)
            or not isinstance(dsl, str)
            or not dsl
            or sha256(dsl.encode()) != champion["program_sha256"]
        ):
            raise ExportError(f"champion family/program identity mismatch: {index}")
        row = dict(dsl=dsl, family_hash=family, **MODEL_PRIOR_BANK_METADATA)
        for field, count, tensor in (("feat412", 412, "bank_feat412"), ("proj16", 16, "bank_proj16")):
            values = np.asarray(champion[field])
            if (
                values.shape != (count,)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
                or not np.array_equal(values, _array(state[tensor][index + 4]))
            ):
                raise ExportError(f"champion {field} disagrees with checkpoint: {index}")
            row[field] = values.tolist()
        summary = _array(state["bank_lm48_z_881"][index])
        raw_summary = np.asarray(champion["lm48_z"])
        if (
            raw_summary.shape != (48,)
            or raw_summary.dtype != np.float32
            or not np.isfinite(raw_summary).all()
        ):
            raise ExportError(f"invalid champion landmark summary: {index}")
        normalized = raw_summary / max(float(np.linalg.norm(raw_summary)), 1e-6)
        if not np.isfinite(summary).all() or not np.allclose(normalized, summary, atol=1e-6, rtol=1e-6):
            raise ExportError(f"champion landmark summary disagrees with checkpoint: {index}")
        row["lm48_summary"] = summary.tolist()
        lines.append(canonical_json(row) + b"\n")
    # Preserve the pinned bank's meaningful row order, including repeated programs.
    result = subprocess.run(
        ["zstd", "-q", "-3", "-T1", "--check", "-c"],
        input=b"".join(lines),
        capture_output=True,
        check=True,
    )
    return result.stdout


def export_bundle(*, state, champions, feature_names, source_hashes, metadata, output):
    validate_state(state)
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != 412
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(set(feature_names)) != 412
    ):
        raise ExportError("412 unique ordered feature names required")
    computed = {"bundle_digest", "dope_weights_sha256", "jope_weights_sha256"}
    if not isinstance(metadata, dict) or computed & set(metadata):
        raise ExportError("metadata must be an unsigned manifest template without computed identities")
    manifest = dict(metadata)
    if (
        manifest.get("constants") != CONSTANTS
        or not manifest.get("surrogate_recipe")
        or not manifest.get("trained_objectives")
    ):
        raise ExportError("exact h40b constants and explicit training declarations required")
    sources = dict(manifest.get("reference_source_sha256", {}))
    for name, value in source_hashes.items():
        if name in sources and sources[name] != value:
            raise ExportError(f"reference source pin conflict: {name}")
        sources[name] = value
    manifest["reference_source_sha256"] = sources
    dope = {name: value for name, value in state.items() if name.startswith("dope_model.")}
    jope = {name: value for name, value in state.items() if not name.startswith("dope_model.")}
    files = {
        "dope.safetensors": safetensors_bytes(dope),
        "jope.safetensors": safetensors_bytes(jope),
        "featurizer.json": canonical_json(
            dict(
                feature_names=feature_names,
                recipe="rank_prepare.features/412-v1",
                proj16_domain="jope-context-cross-v1",
            )
        )
        + b"\n",
        "bank.jsonl.zst": bank_bytes(champions, state),
    }
    assert set(files) == PAYLOADS
    manifest.update(
        dope_weights_sha256=sha256(files["dope.safetensors"]),
        jope_weights_sha256=sha256(files["jope.safetensors"]),
    )
    manifest["bundle_digest"] = compute_bundle_digest(manifest, files)
    jsonschema.Draft202012Validator(read_json(SCHEMA_PATH.read_bytes())).validate(manifest)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ExportError("output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fused-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for name, data in files.items():
            (staging / name).write_bytes(data)
        (staging / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        # No writes or deletion of any existing destination, even an empty one.
        # Link-free output is created only after every input and manifest passes.
        if output.exists() or output.is_symlink():
            raise ExportError("output appeared during export")
        _publish_directory(staging, output)
    return manifest


def _publish_directory(staging, output):
    """Linux research exporter: atomically refuse even an empty racing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ExportError("atomic no-replace publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1):  # AT_FDCWD, RENAME_NOREPLACE
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custody", type=Path, default=CUSTODY_PATH)
    parser.add_argument("--external-root", type=Path, help="cache containing sha256/<pinned SHA-256> files")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    try:
        manifest = export_bundle(
            **load_inputs(args.custody, external_root=args.external_root), metadata=read_json(args.metadata.read_bytes()), output=args.output
        )
    except (
        ExportError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        jsonschema.ValidationError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"fused bundle export refused: {error}", file=sys.stderr)
        return 2
    print(manifest["bundle_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
