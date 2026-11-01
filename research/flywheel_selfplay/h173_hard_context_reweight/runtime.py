"""H173 runtime: input audit, frozen-state guard, None-safe grad clipping, and per-context regret-weighted loss."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path("/home/ubuntu/JopeDimeMicro")
INPUT = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache/context_features.npz")
HERE = Path(__file__).resolve().parent
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
GUARD_COMMIT = "a1b08cbfabe77410a7c323762146c85a50d7f16f"
GUARD_PATH = "python/src/jope/north_star/dope_freeze_guard.py"
GUARD_SHA = "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c"
FORBIDDEN = (
    "meta-test",
    "meta_test",
    "holdout",
    "sealed",
    "test_fold",
    "outer_test",
    "val_19",
    "s3_val",
    "data/flywheel_selfplay",
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Audit:
    def __init__(self, allowed, output_root):
        self.allowed = {str(Path(p).resolve()) for p in allowed}
        self.output_root = str(Path(output_root).resolve())
        self.opened = set()
        self.refused = []

    def check(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, os.PathLike)):
            return
        raw = str(args[0])
        if raw.startswith(("/proc/", "/sys/", "/dev/", "/etc/", "/usr/", "/lib")):
            return
        mode = str(args[1]) if len(args) > 1 else "r"
        p = str(Path(raw).resolve())
        low = p.lower()
        if any(tok in low for tok in FORBIDDEN):
            self.refused.append(p)
            raise RuntimeError(f"forbidden path: {p}")
        if p == self.output_root or p.startswith(self.output_root + "/"):
            return
        if any(ch in mode for ch in "wa+x"):
            if p.startswith("/home/ubuntu/"):
                self.refused.append(p)
                raise RuntimeError(f"write outside output directory: {p}")
            return
        if p.startswith("/home/ubuntu/jope-data/"):
            if p not in self.allowed:
                self.refused.append(p)
                raise RuntimeError(f"unpinned data read: {p}")
            self.opened.add(p)

    def receipt(self):
        return dict(
            opened_count=len(self.opened),
            protected_reads=0,
            refused=self.refused,
        )


def tensor_hash(items):
    h = hashlib.sha256()
    for name, t in items:
        a = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(str(a.dtype).encode())
        h.update(memoryview(a.numpy()))
    return h.hexdigest()


def load_fused():
    p = INPUT / "cpu_exact_fused.py"
    m = types.ModuleType("h173_fused")
    m.__file__ = str(p)
    sys.modules[m.__name__] = m
    exec(compile(p.read_bytes(), str(p), "exec"), m.__dict__)
    return m


def build(checkpoint):
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    fused = load_fused()
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != DOPE:
        raise RuntimeError("wrong checkpoint")
    net.load_state_dict(state["state_dict"], strict=True)
    src = subprocess.check_output(
        ["git", "show", GUARD_COMMIT + ":" + GUARD_PATH], cwd=HERE
    )
    if hashlib.sha256(src).hexdigest() != GUARD_SHA:
        raise RuntimeError("guard source mismatch")
    guard = types.ModuleType("h173_guard")
    sys.modules[guard.__name__] = guard
    exec(compile(src, GUARD_PATH, "exec"), guard.__dict__)
    for name, p in net.named_parameters():
        p.requires_grad_(
            name.startswith("adapter_10tok.")
            and not name.startswith("adapter_10tok.anchor.")
        )
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    if len(names) != 47 or sum(net.state_dict()[n].numel() for n in names) != 29211:
        raise RuntimeError("unexpected trainable adapter surface")
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE hash mismatch")
    frozen = tensor_hash(
        (n, t) for n, t in sorted(net.state_dict().items()) if n not in set(names)
    )
    return net, guard, names, frozen, fused


def check_frozen(net, guard, names, expected):
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE hash changed")
    names_set = set(names)
    got = tensor_hash(
        (n, t) for n, t in sorted(net.state_dict().items()) if n not in names_set
    )
    if got != expected:
        raise RuntimeError("non-adapter state changed")
    if net.training or net.adapter_10tok.training or net.dope_model.training:
        raise RuntimeError("eval mode violated")


def dropout_census(net):
    import torch

    return dict(
        adapter_dropout_modules=[
            dict(name=n, p=m.p)
            for n, m in net.adapter_10tok.named_modules()
            if not n.startswith("anchor")
            and isinstance(m, torch.nn.modules.dropout._DropoutNd)
        ],
        adapter_attention_dropout=[
            dict(name=n, p=m.dropout)
            for n, m in net.adapter_10tok.named_modules()
            if not n.startswith("anchor") and isinstance(m, torch.nn.MultiheadAttention)
        ],
        modality_dropout_attribute=net.adapter_10tok.modality_dropout_p,
        effective_encoder=net.adapter_10tok.encode_multimodal_context_and_candidates.__func__.__name__,
        adapter_training=bool(net.adapter_10tok.training),
        dope_training=bool(net.dope_model.training),
    )


def reset_adapter_to_r882(net, r882_sd, names):
    """Restores the 47 trainable adapter tensors (29,211 parameters) to exact r882 state."""
    import torch

    net.load_state_dict(r882_sd, strict=True)
    net.eval()
    for name, p in net.named_parameters():
        p.requires_grad_(
            name.startswith("adapter_10tok.")
            and not name.startswith("adapter_10tok.anchor.")
        )
    cur_names = [n for n, p in net.named_parameters() if p.requires_grad]
    if sorted(cur_names) != sorted(names) or len(cur_names) != 47:
        raise RuntimeError("adapter parameter tensor names changed")
    param_map = dict(net.named_parameters())
    params = [param_map[n] for n in names]
    if sum(p.numel() for p in params) != 29211:
        raise RuntimeError("unexpected trainable parameter count")
    init_state = {n: net.state_dict()[n].detach().cpu().clone() for n in names}
    init_vec = torch.cat([init_state[n].reshape(-1) for n in names])
    return params, init_state, init_vec


def forward_adapter_with_anchor(
    net,
    prog_t,
    sos_t,
    tok_t,
    mask_t,
    scale_t,
    rev_idx,
    rev_raw_f,
    rev_gate,
    anchor_total,
    anchor_eff_prior,
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    ctx_pooled, cand_q, sem24, mm_prior_adj = (
        ad.encode_multimodal_context_and_candidates(
            mm_tokens_10x32=tok_t,
            mm_mask_10=mask_t,
            prog_412=prog_t,
            sos_prior=sos_t,
            active_modality_override_mask=None,
        )
    )
    static = {
        "eff_prior": anchor_eff_prior + mm_prior_adj,
        "mm_prior_adj": mm_prior_adj,
        "sem24": sem24,
        "cand_q": cand_q,
        "cand_q_norm": F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
        "ctx_pooled": ctx_pooled,
    }
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale_t,
        revealed_indices=rev_idx,
        revealed_raw_f=rev_raw_f,
        revealed_gate=rev_gate,
    )
    return (anchor_total + delta).float()


def safe_grad_norm_and_clip(params, max_norm=1.0):
    """Clips active gradients and safely handles p.grad is None without raising AttributeError."""
    import torch

    active = [p for p in params if p.grad is not None]
    none_indices = [i for i, p in enumerate(params) if p.grad is None]
    if not active:
        raise RuntimeError("no active parameter gradients in backward pass")
    raw_norm = float(
        torch.nn.utils.clip_grad_norm_(active, max_norm, error_if_nonfinite=True)
    )
    return raw_norm, none_indices


def per_context_loss(pred, fitness, gate):
    """Exact H146/H156 per-context loss vector [B]: valid target=clip(fitness, -4.0, 2.0), invalid=2.0, SmoothL1 + 0.1 KL at T=0.2."""
    import torch
    import torch.nn.functional as F

    target = torch.where(
        gate.bool(), fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0)
    )
    return F.smooth_l1_loss(pred, target, reduction="none").mean(1) + 0.1 * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        F.softmax(-target / 0.2, dim=-1),
        reduction="none",
    ).sum(1)


def loss(pred, fitness, gate):
    """Unweighted batch mean loss (exact H146/H156 parity)."""
    return per_context_loss(pred, fitness, gate).mean()


def weighted_loss(pred, fitness, gate, context_weights=None):
    """Per-context regret-weighted batch loss: mean_i(w_{c_i} * L_i). Bit-identical to loss() when w == 1."""
    import torch

    ctx_l = per_context_loss(pred, fitness, gate)
    if context_weights is None:
        return ctx_l.mean()
    w = torch.as_tensor(
        context_weights, dtype=ctx_l.dtype, device=ctx_l.device
    ).reshape(-1)
    if w.shape != ctx_l.shape:
        raise ValueError(f"context_weights shape mismatch: {w.shape} != {ctx_l.shape}")
    if not torch.isfinite(w).all() or (w <= 0.0).any():
        raise ValueError("context_weights must be finite and strictly positive")
    return (ctx_l * w).mean()
