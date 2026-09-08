"""Behavior-preserving weight-space symmetries for the TabPFN-v3 regressor.

Every transform here moves the weights (often maximally in L2) while leaving the
network's function EXACTLY unchanged. This demonstrates that raw weight distance
measures coordinates, not behavior: the identical function has representations
~sqrt(2) (=141%) apart in relative L2, or unboundedly far with rescalings.

Architecture (class TabPFNV3, 504 tensors / 58.28M params), verified:
- ICL residual D=512 (24 blocks, 8 heads x 64, NO rope), MLP hidden 1024, GELU.
- Head: output_norm(RMSNorm 512) -> output_projection.0 Linear(512->1024) -> GELU
  -> output_projection.2 Linear(1024->5000). Bias-free attn/mlp, weight-only RMSNorm.
- softmax_scaling (QASSMax) applies a coordinatewise (Hadamard) rescale to Q, so
  Q/K admit permutations only (not continuous rotation); O/V is fully free.

All ops take a state_dict (name->tensor) and return a NEW state_dict; the model is
rebuilt + strict-loaded to verify functional equivalence on real episodes.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from .model_io import CONFIG_NAME, default_weights_dir
from .paths import import_regressor_train

D_MODEL = 512
N_HEADS = 8
HEAD_DIM = 64          # D_MODEL // N_HEADS
N_ICL = 24
ICL_HIDDEN = 1024
HEAD_HIDDEN = 1024     # output_projection hidden


# --------------------------------------------------------------------------
# model build + forward + distance metrics
# --------------------------------------------------------------------------
def build_model(state_dict: dict, device="cpu", full_kv=False):
    train_mod = import_regressor_train()
    cfg = json.loads((default_weights_dir() / CONFIG_NAME).read_text())
    model = train_mod.get_architecture(train_mod.TabPFNV3Config.from_dict(cfg))
    res = model.load_state_dict(state_dict, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    if full_kv:
        # Disable the inference-time multiquery K/V head reduction so the query
        # path uses all 8 K/V heads. Per-head K/V symmetries (head perm, O/V) are
        # exact for this full-attention function; the deployed sliced-KV path is
        # a separate speed approximation that slightly breaks them.
        for m in model.modules():
            if hasattr(m, "num_kv_heads_test"):
                m.num_kv_heads_test = None
    return model.to(device).eval()


@torch.no_grad()
def predict(model, cx, cy, qx, device="cpu"):
    """Bit-faithful single-view forward (eval_protocol._mean_single): returns the
    de-standardized scalar predictions for query rows qx."""
    cx = np.asarray(cx, np.float32); qx = np.asarray(qx, np.float32)
    cy = np.asarray(cy, np.float32)
    mu, sd = float(cy.mean()), float(cy.std()) + 1e-8
    borders = model.regression_borders.float()
    centres = (borders[:-1] + (borders[1:] - borders[:-1]) / 2).to(device)
    cxt = torch.from_numpy(cx).to(device)
    yt = torch.from_numpy((cy - mu) / sd).to(device).unsqueeze(1)
    x = torch.cat([cxt, torch.from_numpy(qx).to(device)], 0).unsqueeze(1)
    logits = model(x, yt).float()
    mean_z = (torch.softmax(logits, -1) @ centres).reshape(-1)
    return (mean_z * sd + mu).cpu().numpy(), logits.reshape(qx.shape[0], -1).cpu()


def dist_metrics(a: dict, b: dict, keys=None):
    """Global relative-L2 + cosine + bit-exact-identical fraction over `keys`."""
    keys = keys or [k for k in a if a[k].is_floating_point()]
    sqd = sqa = dot = sqb = 0.0; tot = same = 0
    for k in keys:
        x = a[k].float().reshape(-1); y = b[k].float().reshape(-1)
        d = y - x
        sqd += float((d * d).sum()); sqa += float((x * x).sum())
        sqb += float((y * y).sum()); dot += float((x * y).sum())
        tot += x.numel(); same += int((d == 0).sum())
    return dict(rel_l2=sqd**0.5 / max(sqa**0.5, 1e-12),
               cos=dot / max(sqa**0.5 * sqb**0.5, 1e-12),
               same_frac=same / tot, l2=sqd**0.5, n=tot)


def _rng(seed):
    return np.random.default_rng(seed)


def _perm_t(n, rng):
    return torch.from_numpy(rng.permutation(n).astype(np.int64))


# --------------------------------------------------------------------------
# 1. HEAD-MLP hidden permutation  (the demonstration)
# --------------------------------------------------------------------------
def permute_head_mlp(sd: dict, seed=0) -> dict:
    """Permute the head MLP's 1024 hidden units. GELU is elementwise so it
    commutes with the permutation -> output logits are algebraically identical.
    Shuffles output_projection.0 rows(+bias) and output_projection.2 input-cols."""
    out = {k: v.clone() for k, v in sd.items()}
    rho = _perm_t(HEAD_HIDDEN, _rng(seed))
    out["output_projection.0.weight"] = out["output_projection.0.weight"][rho, :]
    out["output_projection.0.bias"] = out["output_projection.0.bias"][rho]
    out["output_projection.2.weight"] = out["output_projection.2.weight"][:, rho]
    return out


# --------------------------------------------------------------------------
# 2. ICL MLP hidden permutation (per block)  — mlp.0 rows <-> mlp.2 cols
# --------------------------------------------------------------------------
def permute_icl_mlp(sd: dict, seed=0, layers=range(N_ICL)) -> dict:
    out = {k: v.clone() for k, v in sd.items()}
    rng = _rng(seed)
    for n in layers:
        rho = _perm_t(ICL_HIDDEN, rng)
        w0 = f"icl_blocks.{n}.mlp.0.weight"; w2 = f"icl_blocks.{n}.mlp.2.weight"
        out[w0] = out[w0][rho, :]
        out[w2] = out[w2][:, rho]
    return out


def _head_block_index(head_perm: np.ndarray) -> torch.Tensor:
    """512-index gather that reorders the 8 head-blocks (each 64 wide)."""
    return torch.from_numpy(
        np.concatenate([head_perm[h] * HEAD_DIM + np.arange(HEAD_DIM)
                        for h in range(N_HEADS)]).astype(np.int64))


# --------------------------------------------------------------------------
# 3. Whole-head permutation (per block) — reorder the 8 attention heads
# --------------------------------------------------------------------------
def permute_heads(sd: dict, seed=0, layers=range(N_ICL)) -> dict:
    out = {k: v.clone() for k, v in sd.items()}
    rng = _rng(seed)
    for n in layers:
        idx = _head_block_index(rng.permutation(N_HEADS))
        a = f"icl_blocks.{n}.icl_attention."
        for p in ("q_projection", "k_projection", "v_projection"):
            out[a + p + ".weight"] = out[a + p + ".weight"][idx, :]       # rows=head out
        out[a + "out_projection.weight"] = out[a + "out_projection.weight"][:, idx]  # cols
        bm = a + "softmax_scaling_layer.base_mlp.2."
        out[bm + "weight"] = out[bm + "weight"][idx, :]                   # 512 rows = heads
        out[bm + "bias"] = out[bm + "bias"][idx]
    return out


# --------------------------------------------------------------------------
# 5. O/V transform (per head, per block) — the CONTINUOUS / UNBOUNDED lever
#    V head-rows <- A @ rows ;  out_projection head-cols <- cols @ A^{-1}
# --------------------------------------------------------------------------
def _rand_orthogonal(n, rng):
    g = torch.from_numpy(rng.standard_normal((n, n))).float()
    q, r = torch.linalg.qr(g)
    return q * torch.sign(torch.diagonal(r))  # unique Haar-ish


def ov_transform(sd: dict, seed=0, layers=range(N_ICL), mode="orthogonal",
                 scale=8.0) -> dict:
    """mode='orthogonal': norm-preserving (~sqrt2). mode='scale': A=c*I diagonal,
    scales V up by c and out-cols down by 1/c -> UNBOUNDED distance, exact output."""
    out = {k: v.clone() for k, v in sd.items()}
    rng = _rng(seed)
    for n in layers:
        a = f"icl_blocks.{n}.icl_attention."
        vw = out[a + "v_projection.weight"]        # (512,512) rows=V out
        ow = out[a + "out_projection.weight"]      # (512,512) cols=head in
        for h in range(N_HEADS):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            if mode == "orthogonal":
                A = _rand_orthogonal(HEAD_DIM, rng)
                Ainv = A.t()
            else:  # diagonal scale c*I  (unbounded)
                c = float(scale) * (1.0 + 0.5 * rng.standard_normal())
                c = max(abs(c), 0.5)
                A = torch.eye(HEAD_DIM) * c
                Ainv = torch.eye(HEAD_DIM) / c
            vw[sl, :] = (A @ vw[sl, :].float()).to(vw.dtype)
            ow[:, sl] = (ow[:, sl].float() @ Ainv).to(ow.dtype)
    return out


# --------------------------------------------------------------------------
# 6. Global E=128 permutation  ->  induces I4 (x) sigma on the 512 stream
# --------------------------------------------------------------------------
E_DIM = 128
N_CLS = 4  # D = 4 CLS tokens x 128


def global_e_permute(sd: dict, seed=0) -> dict:
    """Permute the E=128 embedding channels by one sigma, everywhere they are
    read/written across x_embed, y-encoders, feature_distribution_embedder,
    column_aggregator, AND the induced I4(x)sigma on the 512 ICL residual +
    head. Leaves head-internal attn dims, softmax_scaling, and rope untouched."""
    out = {k: v.clone() for k, v in sd.items()}
    sigma_np = _rng(seed).permutation(E_DIM)
    sig = torch.from_numpy(sigma_np.astype(np.int64))
    idx512 = torch.from_numpy(
        np.concatenate([b * E_DIM + sigma_np for b in range(N_CLS)]).astype(np.int64))

    def rows(k, idx):
        out[k] = out[k][idx, :] if out[k].ndim == 2 else out[k][idx]

    def cols(k, idx):
        out[k] = out[k][:, idx]

    ks = set(out)
    # -- E-space (sigma over 128) --
    rows("x_embed.weight", sig); rows("x_embed.bias", sig)
    rows("col_y_encoder.weight", sig); rows("col_y_encoder.bias", sig)
    # feature_distribution_embedder layers
    import re
    fde = sorted({int(m.group(1)) for k in ks
                  if (m := re.match(r"feature_distribution_embedder\.layers\.(\d+)\.", k))})
    for L in fde:
        p = f"feature_distribution_embedder.layers.{L}."
        cols(p + "inducing_vectors", sig)                     # queries: E = columns
        for J in (1, 2):
            b = p + f"cross_attn_block{J}."
            for pr in ("q_projection", "k_projection", "v_projection"):
                cols(b + f"attn.{pr}.weight", sig)            # read E (input cols)
            rows(b + "attn.out_projection.weight", sig)       # write E (rows)
            cols(b + "mlp.0.weight", sig); rows(b + "mlp.2.weight", sig)
            for ln in ("layernorm_q", "layernorm_kv", "layernorm2"):
                rows(b + f"{ln}.weight", sig)
    # column_aggregator
    cols("column_aggregator.cls_tokens", sig)                 # (4,128): E = columns
    cag = sorted({int(m.group(1)) for k in ks
                  if (m := re.match(r"column_aggregator\.blocks\.(\d+)\.", k))})
    for K in cag:
        b = f"column_aggregator.blocks.{K}."
        for pr in ("q_projection", "k_projection", "v_projection"):
            cols(b + f"attention.{pr}.weight", sig)
        rows(b + "attention.out_projection.weight", sig)
        rows(b + "layernorm.weight", sig); rows(b + "layernorm_mlp.weight", sig)
        cols(b + "mlp.0.weight", sig); rows(b + "mlp.2.weight", sig)
    rows("column_aggregator.out_ln.weight", sig)              # induces sigma in every 512 block
    # -- induced 512 side (I4 x sigma) --
    rows("icl_y_encoder.weight", idx512); rows("icl_y_encoder.bias", idx512)
    for n in range(N_ICL):
        a = f"icl_blocks.{n}."
        rows(a + "layernorm.weight", idx512)
        rows(a + "layernorm_mlp.weight", idx512)
        for pr in ("q_projection", "k_projection", "v_projection"):
            cols(a + f"icl_attention.{pr}.weight", idx512)    # read residual (cols)
        rows(a + "icl_attention.out_projection.weight", idx512)  # write residual (rows)
        cols(a + "mlp.0.weight", idx512); rows(a + "mlp.2.weight", idx512)
    rows("output_norm.weight", idx512)
    cols("output_projection.0.weight", idx512)                # head reads residual
    return out


# --------------------------------------------------------------------------
# scramble_all — compose everything for maximum weight distance
# --------------------------------------------------------------------------
def scramble_all(sd: dict, seed=0, mode="orthogonal") -> dict:
    """Compose the head-MLP perm, all 24 ICL-MLP perms, all head perms, and all
    O/V transforms. mode='orthogonal' -> norm-preserving (~141% relL2, cos~0,
    output identical); mode='unbounded' -> O/V diagonal scaling (relL2 >> 1)."""
    out = global_e_permute(sd, seed)
    out = permute_head_mlp(out, seed + 1)
    out = permute_icl_mlp(out, seed + 2)
    out = permute_heads(out, seed + 3)
    out = ov_transform(out, seed + 4,
                       mode="scale" if mode == "unbounded" else "orthogonal")
    return out


# --------------------------------------------------------------------------
# verification harness
# --------------------------------------------------------------------------
def verify_equivalence(sd0: dict, sd1: dict, views, device="cpu", label="",
                       full_kv=False):
    m0, m1 = build_model(sd0, device, full_kv), build_model(sd1, device, full_kv)
    max_pred = max_logit = 0.0
    for v in views:
        p0, l0 = predict(m0, v.Xtr, v.ytr, v.Xte, device)
        p1, l1 = predict(m1, v.Xtr, v.ytr, v.Xte, device)
        rng = max(abs(p0).max(), 1e-9)
        max_pred = max(max_pred, float(np.abs(p1 - p0).max()))
        max_logit = max(max_logit, float((l1 - l0).abs().max()))
    return dict(label=label, max_abs_pred_delta=max_pred,
                max_abs_logit_delta=max_logit)
