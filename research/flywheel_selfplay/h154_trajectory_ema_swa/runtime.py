"""H154 input auditing, raw-byte state checks, squash-compatible source binding, trajectory EMA/SWA accumulators, and train-only stat recalibration."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
GUARD_COMMIT = "a1b08cbfabe77410a7c323762146c85a50d7f16f"
GUARD_PATH = "python/src/jope/north_star/dope_freeze_guard.py"
GUARD_SHA = "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c"
INPUT = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache/context_features.npz")
H154_DATA = Path("/home/ubuntu/jope-data/h154")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

BOUND_FILES = (
    "research/flywheel_selfplay/h154_trajectory_ema_swa/PREREG.md",
    "research/flywheel_selfplay/h154_trajectory_ema_swa/runner.py",
    "research/flywheel_selfplay/h154_trajectory_ema_swa/runtime.py",
    "research/flywheel_selfplay/h154_trajectory_ema_swa/science.py",
    "research/flywheel_selfplay/h154_trajectory_ema_swa/monitor.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)

MIN_NICE = 19
DISK_USED_MAX = 0.95
DISK_FREE_MIN_BYTES = 40_000_000_000
SCRATCH_LIMIT_BYTES = 1_000_000_000
LOAD_MAX = 85.0
RSS_LIMIT_BYTES = 6_000_000_000


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def tree_bytes(path):
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def guards(path):
    """Hard-stop (never pause) on nice, disk, load or scratch; called at launch, per seed/arm, and at completion."""
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H154 requires nice >= 19")
    usage = shutil.disk_usage(path)
    used = 1.0 - usage.free / usage.total
    if used >= DISK_USED_MAX or usage.free < DISK_FREE_MIN_BYTES:
        raise RuntimeError(f"disk stop: used {used:.4f}, free {usage.free} B")
    load = os.getloadavg()[0]
    if load > LOAD_MAX:
        raise RuntimeError(f"load stop: {load:.1f} > 85.0")
    scratch = sum(tree_bytes(t) for t in {Path(path).resolve(), H154_DATA.resolve()})
    if scratch >= SCRATCH_LIMIT_BYTES:
        raise RuntimeError(f"scratch stop: {scratch} B >= 1 GB")
    return dict(
        disk_used_fraction=round(used, 4),
        disk_free_bytes=usage.free,
        load1=load,
        scratch_bytes=scratch,
    )


def bind_source(launch, repo=REPO, prereg_sha=None):
    """Squash-compatible source blob binding: verifies local executed bytes match both reviewed_head and source_commit."""
    reviewed = launch["reviewed_head"]
    source = launch["source_commit"]
    expected_prereg = prereg_sha or launch.get("prereg_sha256")
    if expected_prereg and sha(repo / BOUND_FILES[0]) != expected_prereg:
        raise RuntimeError("PREREG bytes differ from preregistration")
    bound = {}
    for rel in BOUND_FILES:
        local = (repo / rel).read_bytes()
        for commit in (reviewed, source):
            blob = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=repo)
            if blob != local:
                raise RuntimeError(f"{rel}: executed bytes differ from {commit}")
        bound[rel] = hashlib.sha256(local).hexdigest()
    return dict(
        reviewed_head=reviewed,
        source_commit=source,
        bound_sha256=bound,
    )


def tensor_hash(state):
    import torch

    h = hashlib.sha256()
    for n, t in sorted(state.items()):
        h.update((n + str(t.dtype) + str(tuple(t.shape))).encode())
        h.update(
            t.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


class Audit:
    def __init__(self, allowed, output):
        self.allowed = {str(Path(p).resolve()) for p in allowed}
        self.output = Path(output).resolve()
        self.opened = set()
        self.refused = []

    def check(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p)
        mode = args[1]
        flags = args[2] if len(args) > 2 else 0
        write = (isinstance(mode, str) and any(x in mode for x in "wax+")) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        protected = any(
            x in s.lower()
            for x in [
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
                "/data/flywheel_selfplay/",
            ]
        )
        data = p.is_relative_to(Path("/home/ubuntu/jope-data"))
        if (
            protected
            or (data and s not in self.allowed and not p.is_relative_to(self.output))
            or (
                write
                and (s in self.allowed or (data and not p.is_relative_to(self.output)))
            )
        ):
            self.refused.append(s)
            raise RuntimeError("input/output boundary refusal: " + s)
        if data or s.startswith("/home/ubuntu/JopeDimeMicro/"):
            self.opened.add(s)

    def receipt(self):
        return dict(opened=sorted(self.opened), refused=self.refused, protected_reads=0)


def load_fused():
    p = INPUT / "cpu_exact_fused.py"
    m = types.ModuleType("h154_fused")
    m.__file__ = str(p)
    sys.modules[m.__name__] = m
    exec(compile(p.read_bytes(), str(p), "exec"), m.__dict__)
    return m


def build(checkpoint):
    import torch

    fused = load_fused()
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != DOPE:
        raise RuntimeError("wrong checkpoint")
    net.load_state_dict(state["state_dict"], strict=True)
    git_cwd = (
        HERE
        if subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=HERE, capture_output=True
        ).returncode
        == 0
        else Path("/home/ubuntu/JopeDime")
    )
    src = subprocess.check_output(
        ["git", "show", GUARD_COMMIT + ":" + GUARD_PATH], cwd=git_cwd
    )
    if hashlib.sha256(src).hexdigest() != GUARD_SHA:
        raise RuntimeError("guard source mismatch")
    guard = types.ModuleType("h154_guard")
    sys.modules[guard.__name__] = guard
    exec(compile(src, GUARD_PATH, "exec"), guard.__dict__)
    for n, p in net.named_parameters():
        p.requires_grad_(
            n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")
        )
    names = tuple(n for n, p in net.named_parameters() if p.requires_grad)
    if (
        len(names) != 47
        or sum(p.numel() for p in net.parameters() if p.requires_grad) != 29211
    ):
        raise RuntimeError("adapter inventory")
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE mismatch")
    frozen = tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
    net._eval_batch_key = None
    anchor_cache = {}
    sem_cache = {}
    orig_anchor = net.adapter_10tok.anchor.predict_costs
    orig_sem24 = fused.compute_semantic_struct_sig_24d_torch

    def cached_anchor(**kw):
        key = getattr(net, "_eval_batch_key", None)
        if key is None:
            return orig_anchor(**kw)
        if key not in anchor_cache:
            out = orig_anchor(**kw)
            anchor_cache[key] = {
                "total_cost": out["total_cost"].detach().float(),
                "eff_prior": out["eff_prior"].detach().float(),
                "fail_logit": out["fail_logit"].detach().float(),
            }
        return anchor_cache[key]

    def cached_sem24(prog_412):
        key = getattr(net, "_eval_batch_key", None)
        if key is None:
            return orig_sem24(prog_412)
        if key not in sem_cache:
            sem_cache[key] = orig_sem24(prog_412).detach()
        return sem_cache[key]

    net.adapter_10tok.anchor.predict_costs = cached_anchor
    fused.compute_semantic_struct_sig_24d_torch = cached_sem24
    return net, guard, names, frozen, fused


class TrajectoryEMAState:
    """Float64 exponential moving average of the 47 trainable adapter tensors along the 40-step trajectory."""

    def __init__(self, initial_state_dict, names, decay):
        import torch

        if decay not in (0.99, 0.995, 0.999):
            raise ValueError("H154 EMA decay must be in {0.99, 0.995, 0.999}")
        self.names = tuple(names)
        self.decay = float(decay)
        self.steps = 0
        self.ema = {
            n: initial_state_dict[n].detach().cpu().to(torch.float64).clone()
            for n in self.names
        }

    def after_fast_step(self, current_state_dict):
        import torch

        with torch.no_grad():
            for n in self.names:
                cur = current_state_dict[n].detach().cpu().to(torch.float64)
                self.ema[n].mul_(self.decay).add_(cur, alpha=1.0 - self.decay)
        self.steps += 1

    def export_state_dict(self, base_state_dict):
        if self.steps != 40:
            raise RuntimeError(f"EMA requires 40 steps before export, got {self.steps}")
        out = {k: v.detach().clone() for k, v in base_state_dict.items()}
        for n in self.names:
            out[n] = self.ema[n].to(dtype=base_state_dict[n].dtype).clone()
        return out


class TrajectorySWAState:
    """Float64 uniform SWA over the last K in {5, 10, 20} round adapter snapshots along the 20-round trajectory."""

    def __init__(self, names):
        self.names = tuple(names)
        self.snapshots = []

    def after_step(self, step_idx, current_state_dict):
        import torch

        # Record end-of-round snapshot every 2 fast steps (20 rounds across 40 steps).
        if step_idx % 2 != 0:
            return False
        snap = {
            n: current_state_dict[n].detach().cpu().to(torch.float64).clone()
            for n in self.names
        }
        self.snapshots.append(snap)
        return True

    def export_state_dict(self, base_state_dict, k):
        import torch

        if k not in (5, 10, 20):
            raise ValueError("H154 SWA window K must be in {5, 10, 20}")
        if len(self.snapshots) != 20:
            raise RuntimeError(
                f"SWA requires 20 round snapshots before export, got {len(self.snapshots)}"
            )
        window = self.snapshots[-k:]
        out = {k_name: v.detach().clone() for k_name, v in base_state_dict.items()}
        inv_k = 1.0 / float(k)
        for n in self.names:
            acc = torch.zeros_like(window[0][n], dtype=torch.float64)
            for snap in window:
                acc.add_(snap[n], alpha=inv_k)
            out[n] = acc.to(dtype=base_state_dict[n].dtype).clone()
        return out


def compute_recal_rms(mm_adj_list, traj_adj_list):
    """Compute float64 RMS dispersion of centered mm_prior_adj and traj_attn_adj over in-fold train slates."""
    import numpy as np

    mm_flat = np.concatenate(
        [np.asarray(x, dtype=np.float64).reshape(-1) for x in mm_adj_list]
    )
    traj_flat = np.concatenate(
        [np.asarray(x, dtype=np.float64).reshape(-1) for x in traj_adj_list]
    )
    mm_rms = float(np.sqrt(np.mean(np.square(mm_flat))))
    traj_rms = float(np.sqrt(np.mean(np.square(traj_flat))))
    if (
        not np.isfinite(mm_rms)
        or not np.isfinite(traj_rms)
        or mm_rms <= 0
        or traj_rms <= 0
    ):
        raise RuntimeError("nonfinite or nonpositive calibration RMS")
    return dict(mm_rms=mm_rms, traj_rms=traj_rms)


def apply_stat_recalibration(
    state_dict,
    names,
    ref_rms,
    cand_rms,
    bounds=(0.85, 1.15),
):
    """Recalibrate trainable adapter output head scales on in-fold train slates to match reference variance."""
    import numpy as np

    lo, hi = float(bounds[0]), float(bounds[1])
    mm_ratio = float(ref_rms["mm_rms"]) / max(float(cand_rms["mm_rms"]), 1e-12)
    traj_ratio = float(ref_rms["traj_rms"]) / max(float(cand_rms["traj_rms"]), 1e-12)
    mm_scale = float(np.clip(mm_ratio, lo, hi))
    traj_scale = float(np.clip(traj_ratio, lo, hi))

    out = {k: v.detach().clone() for k, v in state_dict.items()}
    mm_keys = (
        "adapter_10tok.mm_prior_out.2.weight",
        "adapter_10tok.mm_prior_out.2.bias",
    )
    traj_keys = (
        "adapter_10tok.traj_delta_head.2.weight",
        "adapter_10tok.traj_delta_head.2.bias",
    )
    name_set = set(names)
    for k in mm_keys:
        if k in out and k in name_set:
            out[k] = (out[k].double() * mm_scale).to(dtype=out[k].dtype).clone()
    for k in traj_keys:
        if k in out and k in name_set:
            out[k] = (out[k].double() * traj_scale).to(dtype=out[k].dtype).clone()

    receipt = dict(
        ref_mm_rms=float(ref_rms["mm_rms"]),
        ref_traj_rms=float(ref_rms["traj_rms"]),
        cand_mm_rms_before=float(cand_rms["mm_rms"]),
        cand_traj_rms_before=float(cand_rms["traj_rms"]),
        raw_mm_ratio=mm_ratio,
        raw_traj_ratio=traj_ratio,
        applied_mm_scale=mm_scale,
        applied_traj_scale=traj_scale,
        bounds=[lo, hi],
    )
    return out, receipt


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
        adapter_training=net.adapter_10tok.training,
        dope_training=net.dope_model.training,
    )


def check_frozen(net, guard, names, frozen):
    if (
        tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
        != frozen
    ):
        raise RuntimeError("non-adapter bitwise drift")
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE drift")


def loss(pred, fitness, gate):
    import torch
    import torch.nn.functional as F

    target = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    return (
        F.smooth_l1_loss(pred, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()
