"""H168 runtime: input audit, frozen-state guard, H156 W10/W20 adapter width configuration, and None-safe grad clipping."""

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
    m = types.ModuleType("h168_fused")
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
    guard = types.ModuleType("h168_guard")
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


def configure_adapter_width(net, r882_sd, names, width_mult, seed):
    """H156 Amendment A1 width configuration (SiLU-preserving) for W10 (1.0x, 29,211 params) and W20 (2.0x, 46,595 params)."""
    import torch
    import torch.nn as nn

    ad = net.adapter_10tok
    ad.prog_412_bottleneck = nn.Sequential(
        nn.Linear(412, 16), nn.LayerNorm(16), nn.SiLU()
    )
    ad.cand_query_proj = nn.Sequential(nn.Linear(66, 32), nn.LayerNorm(32), nn.SiLU())
    ad.ctx_bilinear_u = nn.Linear(32, 16, bias=False)
    ad.prog_bilinear_v = nn.Linear(24, 16, bias=False)
    ad.mm_prior_out = nn.Sequential(nn.Linear(50, 24), nn.SiLU(), nn.Linear(24, 1))
    ad.traj_delta_head = nn.Sequential(nn.Linear(67, 24), nn.SiLU(), nn.Linear(24, 1))
    ad.parent_delta_net = nn.Sequential(
        nn.Linear(114, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 1)
    )
    net.load_state_dict(r882_sd, strict=True)
    if abs(width_mult - 1.0) > 1e-9:
        if abs(width_mult - 2.0) > 1e-9:
            raise ValueError(
                f"unsupported width_mult={width_mult}; only 1.0 (W10) and 2.0 (W20) registered"
            )
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + int(round(width_mult * 1000)))
        d_bot = int(round(16 * width_mult))
        d_bi = int(round(16 * width_mult))
        d_mm = int(round(24 * width_mult))
        d_traj = int(round(24 * width_mult))
        d_par = int(round(48 * width_mult))
        with torch.no_grad():
            w_bot0 = ad.prog_412_bottleneck[0].weight.detach().cpu()
            b_bot0 = ad.prog_412_bottleneck[0].bias.detach().cpu()
            w_bot1 = ad.prog_412_bottleneck[1].weight.detach().cpu()
            b_bot1 = ad.prog_412_bottleneck[1].bias.detach().cpu()
            w_cq0 = ad.cand_query_proj[0].weight.detach().cpu()
            b_cq0 = ad.cand_query_proj[0].bias.detach().cpu()
            w_cq1 = ad.cand_query_proj[1].weight.detach().cpu()
            b_cq1 = ad.cand_query_proj[1].bias.detach().cpu()
            w_bi_u = ad.ctx_bilinear_u.weight.detach().cpu()
            w_bi_v = ad.prog_bilinear_v.weight.detach().cpu()
            w_mm0 = ad.mm_prior_out[0].weight.detach().cpu()
            b_mm0 = ad.mm_prior_out[0].bias.detach().cpu()
            w_mm2 = ad.mm_prior_out[2].weight.detach().cpu()
            b_mm2 = ad.mm_prior_out[2].bias.detach().cpu()
            w_tr0 = ad.traj_delta_head[0].weight.detach().cpu()
            b_tr0 = ad.traj_delta_head[0].bias.detach().cpu()
            w_tr2 = ad.traj_delta_head[2].weight.detach().cpu()
            b_tr2 = ad.traj_delta_head[2].bias.detach().cpu()
            w_pa0 = ad.parent_delta_net[0].weight.detach().cpu()
            b_pa0 = ad.parent_delta_net[0].bias.detach().cpu()
            w_pa1 = ad.parent_delta_net[1].weight.detach().cpu()
            b_pa1 = ad.parent_delta_net[1].bias.detach().cpu()
            w_pa3 = ad.parent_delta_net[3].weight.detach().cpu()
            b_pa3 = ad.parent_delta_net[3].bias.detach().cpu()

            ad.prog_412_bottleneck = nn.Sequential(
                nn.Linear(412, d_bot), nn.LayerNorm(d_bot), nn.SiLU()
            )
            ad.cand_query_proj = nn.Sequential(
                nn.Linear(50 + d_bot, 32), nn.LayerNorm(32), nn.SiLU()
            )
            ad.cand_query_proj[1].weight.copy_(w_cq1)
            ad.cand_query_proj[1].bias.copy_(b_cq1)
            ad.ctx_bilinear_u = nn.Linear(32, d_bi, bias=False)
            ad.prog_bilinear_v = nn.Linear(24, d_bi, bias=False)
            ad.mm_prior_out = nn.Sequential(
                nn.Linear(34 + d_bi, d_mm), nn.SiLU(), nn.Linear(d_mm, 1)
            )
            ad.traj_delta_head = nn.Sequential(
                nn.Linear(67, d_traj), nn.SiLU(), nn.Linear(d_traj, 1)
            )
            ad.parent_delta_net = nn.Sequential(
                nn.Linear(114, d_par),
                nn.LayerNorm(d_par),
                nn.SiLU(),
                nn.Linear(d_par, 1),
            )

            eps_s = 1e-3
            ad.prog_412_bottleneck[0].weight.copy_(torch.cat([w_bot0, w_bot0], dim=0))
            ad.prog_412_bottleneck[0].bias.copy_(torch.cat([b_bot0, b_bot0], dim=0))
            ad.prog_412_bottleneck[1].weight.copy_(torch.cat([w_bot1, w_bot1], dim=0))
            ad.prog_412_bottleneck[1].bias.copy_(torch.cat([b_bot1, b_bot1], dim=0))

            w_cq_b = w_cq0[:, 48:64]
            e_cq = torch.randn(w_cq_b.shape, generator=g, dtype=w_cq0.dtype) * eps_s
            ad.cand_query_proj[0].weight.copy_(
                torch.cat(
                    [
                        w_cq0[:, :48],
                        0.5 * w_cq_b + e_cq,
                        0.5 * w_cq_b - e_cq,
                        w_cq0[:, 64:],
                    ],
                    dim=1,
                )
            )
            ad.cand_query_proj[0].bias.copy_(b_cq0)

            ad.ctx_bilinear_u.weight.copy_(torch.cat([w_bi_u, w_bi_u], dim=0))
            ad.prog_bilinear_v.weight.copy_(torch.cat([w_bi_v, w_bi_v], dim=0))

            w_mm_b = w_mm0[:, 32:48]
            e_bi = torch.randn(w_mm_b.shape, generator=g, dtype=w_mm0.dtype) * eps_s
            w_mm_in = torch.cat(
                [
                    w_mm0[:, :32],
                    0.5 * w_mm_b + e_bi,
                    0.5 * w_mm_b - e_bi,
                    w_mm0[:, 48:],
                ],
                dim=1,
            )
            ad.mm_prior_out[0].weight.copy_(torch.cat([w_mm_in, w_mm_in], dim=0))
            ad.mm_prior_out[0].bias.copy_(torch.cat([b_mm0, b_mm0], dim=0))
            e_mm2 = torch.randn(w_mm2.shape, generator=g, dtype=w_mm2.dtype) * eps_s
            ad.mm_prior_out[2].weight.copy_(
                torch.cat([0.5 * w_mm2 + e_mm2, 0.5 * w_mm2 - e_mm2], dim=1)
            )
            ad.mm_prior_out[2].bias.copy_(b_mm2)

            ad.traj_delta_head[0].weight.copy_(torch.cat([w_tr0, w_tr0], dim=0))
            ad.traj_delta_head[0].bias.copy_(torch.cat([b_tr0, b_tr0], dim=0))
            e_tr2 = torch.randn(w_tr2.shape, generator=g, dtype=w_tr2.dtype) * eps_s
            ad.traj_delta_head[2].weight.copy_(
                torch.cat([0.5 * w_tr2 + e_tr2, 0.5 * w_tr2 - e_tr2], dim=1)
            )
            ad.traj_delta_head[2].bias.copy_(b_tr2)

            ad.parent_delta_net[0].weight.copy_(torch.cat([w_pa0, w_pa0], dim=0))
            ad.parent_delta_net[0].bias.copy_(torch.cat([b_pa0, b_pa0], dim=0))
            ad.parent_delta_net[1].weight.copy_(torch.cat([w_pa1, w_pa1], dim=0))
            ad.parent_delta_net[1].bias.copy_(torch.cat([b_pa1, b_pa1], dim=0))
            e_pa3 = torch.randn(w_pa3.shape, generator=g, dtype=w_pa3.dtype) * eps_s
            ad.parent_delta_net[3].weight.copy_(
                torch.cat([0.5 * w_pa3 + e_pa3, 0.5 * w_pa3 - e_pa3], dim=1)
            )
            ad.parent_delta_net[3].bias.copy_(b_pa3)

    dev = next(net.adapter_10tok.anchor.parameters()).device
    ad.to(dev)
    net.eval()
    for name, p in net.named_parameters():
        p.requires_grad_(
            name.startswith("adapter_10tok.")
            and not name.startswith("adapter_10tok.anchor.")
        )
    cur_names = [n for n, p in net.named_parameters() if p.requires_grad]
    if sorted(cur_names) != sorted(names):
        raise RuntimeError("adapter parameter tensor names changed")
    param_map = dict(net.named_parameters())
    params = [param_map[n] for n in names]
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


def loss(pred, fitness, gate):
    """Exact H156/H146 (SHA 1fbb2c89) live-recipe loss: valid target=clip(fitness, -4.0, 2.0), invalid=2.0, SmoothL1 + 0.1 KL at T=0.2."""
    import torch
    import torch.nn.functional as F

    target = torch.where(
        gate.bool(), fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0)
    )
    return (
        F.smooth_l1_loss(pred, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()
