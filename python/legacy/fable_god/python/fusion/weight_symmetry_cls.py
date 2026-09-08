"""Behavior-preserving weight-symmetry scrambles for the CLASSIFICATION fusion experts.

The regressor toolkit (elo_finetune/weight_symmetry.py) is hardcoded to the regressor
state_dict; this re-implements the symmetry group for the two classification experts:
  A = TabPFN-v3-cls   (torch .ckpt, key tree: icl_blocks/many_class_decoder/output_norm/...)
  B = TabICL-cls      (safetensors, packed-QKV tf blocks: col_embedder/row_interactor/icl_predictor)

Each op clones the state_dict and mutates a handful of keys, preserving the model's
FUNCTION (predictions) while moving weights maximally far (cosine→0, relL2→√2) from
the original. Ops are tagged CLEAN (exact on both the full-attention and the deployed
multiquery/sliced-KV path) or CONCERNING (exact only on full attention; A's per-head
K/V ops shift the sliced-KV test path — A deploys num_kv_heads_test=1).

Constraints (both experts): softmax-scaling (ssmax) applies a per-coordinate Hadamard
rescale to Q, and RoPE pins Q/K order → Q/K admit PERMUTATIONS only, never rotations.
O/V is a free GL change-of-basis (orthogonal, or unbounded diagonal scale).
"""
from __future__ import annotations
import torch


# ------------------------------------------------------------------ helpers
def _rng(seed):
    g = torch.Generator(); g.manual_seed(int(seed)); return g


def _perm(n, seed):
    return torch.randperm(n, generator=_rng(seed))


def _head_index(head_perm, n_heads, head_dim):
    """A permutation over `n_heads` head-blocks → a flat index over n_heads*head_dim
    that reorders whole 64-wide (or 16-wide) head blocks."""
    return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in head_perm])


def _rand_orthogonal(n, seed):
    a = torch.randn(n, n, generator=_rng(seed), dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q *= torch.sign(torch.diagonal(r)).unsqueeze(0)          # sign-fix for a proper Haar sample
    return q


def dist_metrics(a: dict, b: dict, keys=None):
    """Per-set relL2 / cosine / same-fraction between two state_dicts (float tensors)."""
    keys = keys or [k for k in a if a[k].is_floating_point()]
    num = den_a = den_b = dot = same = tot = 0.0
    for k in keys:
        x = a[k].flatten().double(); y = b[k].flatten().double()
        num += ((y - x) ** 2).sum().item(); den_a += (x * x).sum().item()
        den_b += (y * y).sum().item(); dot += (x * y).sum().item()
        same += (x == y).sum().item(); tot += x.numel()
    import math
    return dict(rel_l2=math.sqrt(num) / max(math.sqrt(den_a), 1e-12),
                cos=dot / max(math.sqrt(den_a * den_b), 1e-12),
                same_frac=same / max(tot, 1), n=int(tot))


def _clone(sd):
    return {k: t.clone() for k, t in sd.items()}


# =================================================================== EXPERT A (TabPFN-v3-cls)
A_N_ICL = 24
A_ICL_HID = 1024
A_ICL_HEADS = 8
A_ICL_HEADDIM = 64          # 8*64 = 512
A_DEC_HEADS = 6
A_DEC_HEADDIM = 64          # 6*64 = 384


def a_permute_icl_mlp(sd, seed=0):
    """CLEAN. Permute each icl block's 1024 MLP hidden units (fc1 rows, fc2 cols). GELU
    elementwise → exact; internal → hidden output unchanged (transparent to the trunk)."""
    out = _clone(sd)
    for n in range(A_N_ICL):
        p = _perm(A_ICL_HID, seed * 100 + n)
        out[f"icl_blocks.{n}.mlp.0.weight"] = out[f"icl_blocks.{n}.mlp.0.weight"][p, :]
        out[f"icl_blocks.{n}.mlp.2.weight"] = out[f"icl_blocks.{n}.mlp.2.weight"][:, p]
    return out


def a_permute_stage12_mlp(sd, seed=0):
    """CLEAN. Permute the 256-hidden MLPs in stage-1 (feature_distribution_embedder) and
    stage-2 (column_aggregator). Internal, exact."""
    out = _clone(sd); s = seed * 1000
    # stage 1: 3 ISAB layers × 2 cross-attn blocks
    for L in range(3):
        for J in (1, 2):
            pre = f"feature_distribution_embedder.layers.{L}.cross_attn_block{J}.mlp"
            if f"{pre}.0.weight" in out:
                p = _perm(out[f"{pre}.0.weight"].shape[0], s); s += 1
                out[f"{pre}.0.weight"] = out[f"{pre}.0.weight"][p, :]
                out[f"{pre}.2.weight"] = out[f"{pre}.2.weight"][:, p]
    # stage 2: column_aggregator blocks
    n = 0
    while f"column_aggregator.blocks.{n}.mlp.0.weight" in out:
        p = _perm(out[f"column_aggregator.blocks.{n}.mlp.0.weight"].shape[0], s); s += 1
        out[f"column_aggregator.blocks.{n}.mlp.0.weight"] = out[f"column_aggregator.blocks.{n}.mlp.0.weight"][p, :]
        out[f"column_aggregator.blocks.{n}.mlp.2.weight"] = out[f"column_aggregator.blocks.{n}.mlp.2.weight"][:, p]
        n += 1
    return out


def a_permute_decoder_heads(sd, seed=0):
    """CLEAN. Permute the 6 decoder heads of many_class_decoder (q/k projection rows +
    biases + softmax_scaling base_mlp.2 rows). The decoder AVERAGES over heads
    (test_output.mean(2)) → head order is irrelevant → exact."""
    out = _clone(sd)
    idx = _head_index(_perm(A_DEC_HEADS, seed), A_DEC_HEADS, A_DEC_HEADDIM)
    for proj in ("q_projection", "k_projection"):
        out[f"many_class_decoder.{proj}.weight"] = out[f"many_class_decoder.{proj}.weight"][idx, :]
        b = f"many_class_decoder.{proj}.bias"
        if b in out:
            out[b] = out[b][idx]
    ss = "many_class_decoder.softmax_scaling_layer.base_mlp.2"
    if f"{ss}.weight" in out:
        out[f"{ss}.weight"] = out[f"{ss}.weight"][idx, :]
        if f"{ss}.bias" in out:
            out[f"{ss}.bias"] = out[f"{ss}.bias"][idx]
    return out


def a_permute_icl_heads(sd, seed=0):
    """CONCERNING (deploys sliced-KV). Reorder the 8 attention heads of every icl block
    (q/k/v rows, out cols, softmax_scaling base_mlp.2 rows). Exact on full-KV; shifts the
    deployed test path that reads only KV-head-0."""
    out = _clone(sd)
    for n in range(A_N_ICL):
        idx = _head_index(_perm(A_ICL_HEADS, seed * 100 + n), A_ICL_HEADS, A_ICL_HEADDIM)
        a = f"icl_blocks.{n}.icl_attention"
        for proj in ("q_projection", "k_projection", "v_projection"):
            out[f"{a}.{proj}.weight"] = out[f"{a}.{proj}.weight"][idx, :]
        out[f"{a}.out_projection.weight"] = out[f"{a}.out_projection.weight"][:, idx]
        ss = f"{a}.softmax_scaling_layer.base_mlp.2"
        out[f"{ss}.weight"] = out[f"{ss}.weight"][idx, :]
        if f"{ss}.bias" in out:
            out[f"{ss}.bias"] = out[f"{ss}.bias"][idx]
    return out


def a_ov_transform_icl(sd, seed=0, mode="orthogonal", scale=8.0):
    """CONCERNING. Per-head O/V change-of-basis on every icl block: V rows[h] = A@V,
    out cols[h] = O@Ainv. V·Oᵀ invariant → exact on full-KV; shifts sliced-KV."""
    out = _clone(sd); hd = A_ICL_HEADDIM
    for n in range(A_N_ICL):
        a = f"icl_blocks.{n}.icl_attention"
        V = out[f"{a}.v_projection.weight"].double().clone()
        O = out[f"{a}.out_projection.weight"].double().clone()
        for h in range(A_ICL_HEADS):
            sl = slice(h * hd, (h + 1) * hd)
            if mode == "scale":
                c = float(scale * (1 + 0.5 * torch.randn(1, generator=_rng(seed * 1000 + n * 8 + h)).item()))
                c = c if abs(c) > 0.5 else 0.5
                A = torch.eye(hd, dtype=torch.float64) * c; Ainv = torch.eye(hd, dtype=torch.float64) / c
            else:
                A = _rand_orthogonal(hd, seed * 1000 + n * 8 + h); Ainv = A.T
            V[sl, :] = A @ V[sl, :]
            O[:, sl] = O[:, sl] @ Ainv
        out[f"{a}.v_projection.weight"] = V.to(out[f"{a}.v_projection.weight"].dtype)
        out[f"{a}.out_projection.weight"] = O.to(out[f"{a}.out_projection.weight"].dtype)
    return out


# =================================================================== EXPERT B (TabICL-cls)
def _b_blocks(sd, prefix):
    n = 0
    while any(k.startswith(f"{prefix}.blocks.{n}.") for k in sd):
        n += 1
    return n


def b_permute_mlp(sd, seed=0):
    """CLEAN. Permute the MLP hidden of every tf block (linear1 rows+bias, linear2 cols)."""
    out = _clone(sd); s = seed * 10000
    for prefix in ("icl_predictor.tf_icl", "row_interactor.tf_row"):
        for n in range(_b_blocks(out, prefix)):
            w1 = f"{prefix}.blocks.{n}.linear1.weight"
            if w1 not in out:
                continue
            p = _perm(out[w1].shape[0], s); s += 1
            out[w1] = out[w1][p, :]
            if f"{prefix}.blocks.{n}.linear1.bias" in out:
                out[f"{prefix}.blocks.{n}.linear1.bias"] = out[f"{prefix}.blocks.{n}.linear1.bias"][p]
            out[f"{prefix}.blocks.{n}.linear2.weight"] = out[f"{prefix}.blocks.{n}.linear2.weight"][:, p]
    # col_embedder has two MHA sublayers per block, each with linear1/linear2
    cp = "col_embedder.tf_col"
    for n in range(_b_blocks(out, cp)):
        for sub in ("multihead_attn1", "multihead_attn2"):
            w1 = f"{cp}.blocks.{n}.{sub}.linear1.weight"
            if w1 not in out:
                continue
            p = _perm(out[w1].shape[0], s); s += 1
            out[w1] = out[w1][p, :]
            if f"{cp}.blocks.{n}.{sub}.linear1.bias" in out:
                out[f"{cp}.blocks.{n}.{sub}.linear1.bias"] = out[f"{cp}.blocks.{n}.{sub}.linear1.bias"][p]
            out[f"{cp}.blocks.{n}.{sub}.linear2.weight"] = out[f"{cp}.blocks.{n}.{sub}.linear2.weight"][:, p]
    return out


def _b_perm_packed_heads(out, attn, n_heads, head_dim, seed):
    """Permute heads of a packed nn.MultiheadAttention: in_proj_weight is [Q;K;V] each
    (n_heads*head_dim) rows; out_proj cols reorder; ssmax base_mlp.2 rows reorder."""
    D = n_heads * head_dim
    idx = _head_index(_perm(n_heads, seed), n_heads, head_dim)
    ipw = f"{attn}.in_proj_weight"; ipb = f"{attn}.in_proj_bias"
    W = out[ipw].clone()
    for t in range(3):                                        # Q, K, V thirds
        W[t * D:(t + 1) * D, :] = W[t * D:(t + 1) * D, :][idx, :]
    out[ipw] = W
    if ipb in out:
        B = out[ipb].clone()
        for t in range(3):
            B[t * D:(t + 1) * D] = B[t * D:(t + 1) * D][idx]
        out[ipb] = B
    out[f"{attn}.out_proj.weight"] = out[f"{attn}.out_proj.weight"][:, idx]
    ss = f"{attn}.ssmax_layer.base_mlp.2"
    if f"{ss}.weight" in out:
        out[f"{ss}.weight"] = out[f"{ss}.weight"][idx, :]
        if f"{ss}.bias" in out:
            out[f"{ss}.bias"] = out[f"{ss}.bias"][idx]


def b_permute_heads(sd, seed=0):
    """CLEAN (B has no sliced path). Permute the 8 attention heads of every tf block."""
    out = _clone(sd); s = seed * 20000
    for prefix, hd in (("icl_predictor.tf_icl", 64), ("row_interactor.tf_row", 16)):
        for n in range(_b_blocks(out, prefix)):
            attn = f"{prefix}.blocks.{n}.attn"
            if f"{attn}.in_proj_weight" in out:
                _b_perm_packed_heads(out, attn, 8, hd, s); s += 1
    cp = "col_embedder.tf_col"
    for n in range(_b_blocks(out, cp)):
        for sub in ("multihead_attn1", "multihead_attn2"):
            attn = f"{cp}.blocks.{n}.{sub}.attn"
            if f"{attn}.in_proj_weight" in out:
                _b_perm_packed_heads(out, attn, 8, 16, s); s += 1
    return out


def b_ov_transform(sd, seed=0, mode="orthogonal", scale=8.0):
    """CLEAN. Per-head O/V change-of-basis on every tf block (V-third of in_proj ↔ out_proj)."""
    out = _clone(sd); s = seed * 30000
    specs = [("icl_predictor.tf_icl", 64, "attn")]
    specs += [("row_interactor.tf_row", 16, "attn")]
    def do(attn, hd, seed_):
        D = 8 * hd
        ipw = out[f"{attn}.in_proj_weight"].double().clone()
        O = out[f"{attn}.out_proj.weight"].double().clone()
        ipb = out.get(f"{attn}.in_proj_bias")
        ipb = ipb.double().clone() if ipb is not None else None       # V has a bias (unlike expert A)
        Vblock = ipw[2 * D:3 * D, :].clone()
        Vbias = ipb[2 * D:3 * D].clone() if ipb is not None else None
        for h in range(8):
            sl = slice(h * hd, (h + 1) * hd)
            if mode == "scale":
                c = float(scale * (1 + 0.5 * torch.randn(1, generator=_rng(seed_ + h)).item()))
                c = c if abs(c) > 0.5 else 0.5
                A = torch.eye(hd, dtype=torch.float64) * c; Ainv = torch.eye(hd, dtype=torch.float64) / c
            else:
                A = _rand_orthogonal(hd, seed_ + h); Ainv = A.T
            Vblock[sl, :] = A @ Vblock[sl, :]
            if Vbias is not None:
                Vbias[sl] = A @ Vbias[sl]                             # transform the V bias too
            O[:, sl] = O[:, sl] @ Ainv
        ipw[2 * D:3 * D, :] = Vblock
        out[f"{attn}.in_proj_weight"] = ipw.to(out[f"{attn}.in_proj_weight"].dtype)
        out[f"{attn}.out_proj.weight"] = O.to(out[f"{attn}.out_proj.weight"].dtype)
        if ipb is not None:
            ipb[2 * D:3 * D] = Vbias
            out[f"{attn}.in_proj_bias"] = ipb.to(out[f"{attn}.in_proj_bias"].dtype)
    for prefix, hd, sub in specs:
        for n in range(_b_blocks(out, prefix)):
            if f"{prefix}.blocks.{n}.{sub}.in_proj_weight" in out:
                do(f"{prefix}.blocks.{n}.{sub}", hd, s * 97 + n); s += 1
    cp = "col_embedder.tf_col"
    for n in range(_b_blocks(out, cp)):
        for sub in ("multihead_attn1", "multihead_attn2"):
            if f"{cp}.blocks.{n}.{sub}.attn.in_proj_weight" in out:
                do(f"{cp}.blocks.{n}.{sub}.attn", 16, s * 97 + n); s += 1
    return out


# =================================================================== GLOBAL channel permutation
# One σ∈S_128 relabels the embedding channels everywhere they are read/written, plus the
# induced I₄⊗σ on the 512-d icl residual stream. Preserves the expert's LOGITS exactly
# (every residual reader/writer moves consistently) but PERMUTES the 512-d hidden output
# (output_norm), which the trunk consumes → compensate the trunk adapter (or train it).
# CLEAN on both paths (the icl q/k/v touch here is the shared INPUT dim, not per-head K/V).
A_E = 128
A_NCLS = 4


def a_global_e_permute(sd, seed=0):
    out = _clone(sd)
    sig = _perm(A_E, seed)
    idx512 = torch.cat([b * A_E + sig for b in range(A_NCLS)])          # I₄⊗σ on the 512 stream

    def R(k, idx):                                                       # permute rows (output dim)
        if k in out:
            out[k] = out[k][idx]

    def C(k, idx):                                                       # permute cols (input dim)
        if k in out:
            out[k] = out[k][:, idx]

    # --- E-space (128) ---
    R("x_embed.weight", sig); R("x_embed.bias", sig)
    C("col_y_encoder.embedding.weight", sig)                            # nn.Embedding: output = last dim
    L = 0
    while f"feature_distribution_embedder.layers.{L}.inducing_vectors" in out:
        p = f"feature_distribution_embedder.layers.{L}."
        C(f"{p}inducing_vectors", sig)
        for J in (1, 2):
            a = f"{p}cross_attn_block{J}."
            for pr in ("q_projection", "k_projection", "v_projection"):
                C(f"{a}attn.{pr}.weight", sig)
            R(f"{a}attn.out_projection.weight", sig)
            C(f"{a}mlp.0.weight", sig); R(f"{a}mlp.2.weight", sig)
            for ln in ("layernorm_q", "layernorm_kv", "layernorm2"):
                R(f"{a}{ln}.weight", sig)
        L += 1
    C("column_aggregator.cls_tokens", sig)
    K = 0
    while f"column_aggregator.blocks.{K}.attention.q_projection.weight" in out:
        a = f"column_aggregator.blocks.{K}."
        for pr in ("q_projection", "k_projection", "v_projection"):
            C(f"{a}attention.{pr}.weight", sig)
        R(f"{a}attention.out_projection.weight", sig)
        C(f"{a}mlp.0.weight", sig); R(f"{a}mlp.2.weight", sig)
        R(f"{a}layernorm.weight", sig); R(f"{a}layernorm_mlp.weight", sig)
        K += 1
    R("column_aggregator.out_ln.weight", sig)
    # --- 512 icl stream (idx512) ---
    C("icl_y_encoder.embedding.weight", idx512)
    for n in range(A_N_ICL):
        b = f"icl_blocks.{n}."
        R(f"{b}layernorm.weight", idx512); R(f"{b}layernorm_mlp.weight", idx512)
        for pr in ("q_projection", "k_projection", "v_projection"):
            C(f"{b}icl_attention.{pr}.weight", idx512)
        R(f"{b}icl_attention.out_projection.weight", idx512)
        C(f"{b}mlp.0.weight", idx512); R(f"{b}mlp.2.weight", idx512)
    R("output_norm.weight", idx512)                                    # THE HIDDEN — permuted by idx512
    C("many_class_decoder.q_projection.weight", idx512)
    C("many_class_decoder.k_projection.weight", idx512)
    return out, idx512


def _b_nblocks(sd, prefix):
    n = 0
    while any(k.startswith(f"{prefix}.blocks.{n}.") for k in sd):
        n += 1
    return n


def b_global_permute(sd, seed=0):
    """CLEAN. One σ∈S_128 relabels B's col+row embedding channels; induced I₄⊗σ on the
    512-d icl stream (4 CLS tokens × 128). Preserves logits; permutes the 512-d hidden
    (row_interactor output) by idx512 → trunk compensation / train. Moves the decoder,
    norms, embeddings, inducing/cls tokens that the head/MLP/O-V ops leave untouched."""
    out = _clone(sd)
    sig = _perm(128, seed)
    idx512 = torch.cat([b * 128 + sig for b in range(4)])

    def R(k, idx):
        if k in out:
            out[k] = out[k][idx]

    def C(k, idx):
        if k in out:
            out[k] = out[k][:, idx]

    def attn(a, idx):                                    # packed in_proj: input=cols; out_proj: output=rows
        C(f"{a}.in_proj_weight", idx)
        R(f"{a}.out_proj.weight", idx)

    def ff(pre, idx):
        C(f"{pre}.linear1.weight", idx); R(f"{pre}.linear2.weight", idx)

    def norms(pre, idx):
        for nm in ("norm1", "norm2"):
            R(f"{pre}.{nm}.weight", idx); R(f"{pre}.{nm}.bias", idx)

    # ---- col_embedder (σ on 128) ----
    R("col_embedder.in_linear.weight", sig); R("col_embedder.in_linear.bias", sig)
    cp = "col_embedder.tf_col"
    for n in range(_b_nblocks(out, cp)):
        C(f"{cp}.blocks.{n}.ind_vectors", sig)
        for sub in ("multihead_attn1", "multihead_attn2"):
            pre = f"{cp}.blocks.{n}.{sub}"
            attn(f"{pre}.attn", sig); ff(pre, sig); norms(pre, sig)
    # ---- row_interactor (σ on 128) -> hidden permuted by idx512 ----
    C("row_interactor.cls_tokens", sig)
    R("row_interactor.out_ln.weight", sig); R("row_interactor.out_ln.bias", sig)
    rp = "row_interactor.tf_row"
    for n in range(_b_nblocks(out, rp)):
        pre = f"{rp}.blocks.{n}"
        attn(f"{pre}.attn", sig); ff(pre, sig); norms(pre, sig)
    # ---- icl_predictor (idx512) ----
    R("icl_predictor.y_encoder.weight", idx512)
    ip = "icl_predictor.tf_icl"
    for n in range(_b_nblocks(out, ip)):
        pre = f"{ip}.blocks.{n}"
        attn(f"{pre}.attn", idx512); ff(pre, idx512); norms(pre, idx512)
    R("icl_predictor.ln.weight", idx512); R("icl_predictor.ln.bias", idx512)
    C("icl_predictor.decoder.0.weight", idx512)
    return out, idx512


# =================================================================== softmax-scaling MLP hidden perms
# The ssmax/softmax_scaling query_mlp + base_mlp are standard 2-layer MLPs, so permuting
# their hidden units (fc0 rows, fc2 cols) is EXACT — moves the last unmoved weight matrices.
def _perm_ssmax(out, pre, seed):
    moved = False
    for i, mlp in enumerate(("query_mlp", "base_mlp")):
        w0 = f"{pre}.{mlp}.0.weight"
        if w0 not in out:
            continue
        p = _perm(out[w0].shape[0], seed * 10 + i)
        out[w0] = out[w0][p, :]
        if f"{pre}.{mlp}.0.bias" in out:
            out[f"{pre}.{mlp}.0.bias"] = out[f"{pre}.{mlp}.0.bias"][p]
        w2 = f"{pre}.{mlp}.2.weight"
        if w2 in out:
            out[w2] = out[w2][:, p]; moved = True
    return moved


def a_permute_softmax_mlps(sd, seed=0):
    """CLEAN. Permute every A softmax_scaling_layer's query_mlp/base_mlp hidden (icl blocks,
    decoder, stage-1/2). Standard MLPs -> exact."""
    out = _clone(sd)
    pres = sorted({k.rsplit(".query_mlp.", 1)[0].rsplit(".base_mlp.", 1)[0]
                   for k in out if "softmax_scaling_layer" in k and (".query_mlp." in k or ".base_mlp." in k)})
    for i, pre in enumerate(pres):
        _perm_ssmax(out, pre, seed * 1000 + i)
    return out


def b_permute_ssmax_mlps(sd, seed=0):
    """CLEAN. Permute every B ssmax_layer's query_mlp/base_mlp hidden. Exact."""
    out = _clone(sd)
    pres = sorted({k.rsplit(".query_mlp.", 1)[0].rsplit(".base_mlp.", 1)[0]
                   for k in out if "ssmax_layer" in k and (".query_mlp." in k or ".base_mlp." in k)})
    for i, pre in enumerate(pres):
        _perm_ssmax(out, pre, seed * 1000 + i)
    return out
