"""H147 input auditing, raw-byte frozen state checks, squash-compatible source binding, and masked/unmasked listwise loss."""

import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import types

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
GUARD_COMMIT = "a1b08cbfabe77410a7c323762146c85a50d7f16f"
GUARD_PATH = "python/src/jope/north_star/dope_freeze_guard.py"
GUARD_SHA = "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c"
ELO_COMMIT = "c9961262bf8c2e21180ac86d9e2a30f5b83ef5a1"
ELO_PATH = "python/src/jope/north_star/elo_tracker.py"
ELO_SHA = "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835"

INPUT = Path("/home/ubuntu/jope-data/h50/input")
H147_DATA = Path("/home/ubuntu/jope-data/h147")
TRAIN_SELECTED = H147_DATA / "inputs" / "train_only_selected_1219.json"
TRAIN_CACHE = H147_DATA / "inputs" / "train_only_context_features_1219.npz"
VAL_CACHE = H147_DATA / "inputs" / "val_only_context_features_91.npz"
S3_INVENTORY = Path("/home/ubuntu/jope_plan/ns/s3_manifest_inventory.json")
MIXED_H58_CACHE = Path("/home/ubuntu/jope-data/h58/cache/context_features.npz")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

BOUND_FILES = (
    "research/flywheel_selfplay/h147_reveal_alignment/PREREG.md",
    "research/flywheel_selfplay/h147_reveal_alignment/PREREG_AMENDMENT_A1.md",
    "research/flywheel_selfplay/h147_reveal_alignment/fold_manifest.json",
    "research/flywheel_selfplay/h147_reveal_alignment/runner.py",
    "research/flywheel_selfplay/h147_reveal_alignment/runtime.py",
    "research/flywheel_selfplay/h147_reveal_alignment/science.py",
    "research/flywheel_selfplay/h147_reveal_alignment/monitor.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
    "python/src/jope/north_star/elo_tracker.py",
)

MIN_NICE = 19
DISK_USED_MAX = 0.95
DISK_FREE_MIN_BYTES = 40_000_000_000
SCRATCH_LIMIT_BYTES = 3_000_000_000
LOAD_MAX = 85.0
RSS_LIMIT_BYTES = 6_000_000_000
CPU_SECONDS_CAP = 7200.0
WALL_SECONDS_CAP = 10800.0
ABSOLUTE_DEADLINE_UNIX = 1793541600.0  # 2026-11-01T14:00:00Z


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def atomic_claim(path, payload):
    """Create a one-use marker file atomically with O_CREAT | O_EXCL; refuse if already present."""
    p = Path(path)
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def tree_bytes(path):
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def cpu_seconds_used():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime)


def guards(path, started_monotonic=None):
    """Hard-stop on nice, disk, load, scratch, CPU cap, wall cap, or absolute deadline."""
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H147 requires nice >= 19")
    if time.time() > ABSOLUTE_DEADLINE_UNIX:
        raise RuntimeError("H147 absolute deadline 2026-11-01T14:00:00Z exceeded")
    if started_monotonic is not None and (
        time.monotonic() - started_monotonic > WALL_SECONDS_CAP
    ):
        raise RuntimeError("H147 wall_seconds cap (10800s) exceeded")
    cpu_s = cpu_seconds_used()
    if cpu_s > CPU_SECONDS_CAP:
        raise RuntimeError(f"H147 cpu_seconds cap (7200s) exceeded: {cpu_s:.1f}s")
    usage = shutil.disk_usage(path)
    used = 1.0 - usage.free / usage.total
    if used >= DISK_USED_MAX or usage.free < DISK_FREE_MIN_BYTES:
        raise RuntimeError(f"disk stop: used {used:.4f}, free {usage.free} B")
    load = os.getloadavg()[0]
    if load > LOAD_MAX:
        raise RuntimeError(f"load stop: {load:.1f} > 85.0")
    scratch = sum(tree_bytes(t) for t in {Path(path).resolve(), H147_DATA.resolve()})
    if scratch >= SCRATCH_LIMIT_BYTES:
        raise RuntimeError(f"scratch stop: {scratch} B >= 3 GB")
    return dict(
        disk_used_fraction=round(used, 4),
        disk_free_bytes=usage.free,
        load1=load,
        scratch_bytes=scratch,
        cpu_seconds=round(cpu_s, 3),
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
    """Enforces read/write boundaries: refuses sealed/holdout/meta-test, non-allowlisted jope-data reads, and writes outside output or /tmp."""

    def __init__(self, allowed, output, forbidden_reads=()):
        self.allowed = {str(Path(p).resolve()) for p in allowed}
        self.forbidden_reads = {str(Path(p).resolve()) for p in forbidden_reads}
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
                "outer-test",
                "outer_test",
                "meta-test",
                "meta_test",
                "qualification.lock",
                "/data/flywheel_selfplay/",
            ]
        )
        data = p.is_relative_to(Path("/home/ubuntu/jope-data"))
        jopedime = p.is_relative_to(Path("/home/ubuntu/JopeDime"))
        jope_plan = p.is_relative_to(Path("/home/ubuntu/jope_plan"))
        in_output = p.is_relative_to(self.output)
        in_tmp = p.is_relative_to(Path("/tmp"))
        write_refused = write and (
            s in self.allowed
            or jopedime
            or (jope_plan and not in_output)
            or (data and not in_output)
            or not (in_output or in_tmp)
        )
        if (
            protected
            or s in self.forbidden_reads
            or (data and s not in self.allowed and not in_output)
            or write_refused
        ):
            self.refused.append(s)
            raise RuntimeError("input/output boundary refusal: " + s)
        if data or s.startswith("/home/ubuntu/JopeDimeMicro/"):
            self.opened.add(s)

    def receipt(self):
        return dict(opened=sorted(self.opened), refused=self.refused, protected_reads=0)


def load_fused():
    p = INPUT / "cpu_exact_fused.py"
    m = types.ModuleType("h147_fused")
    m.__file__ = str(p)
    sys.modules[m.__name__] = m
    exec(compile(p.read_bytes(), str(p), "exec"), m.__dict__)
    return m


def load_elo_tracker():
    git_cwd = (
        HERE
        if subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=HERE, capture_output=True
        ).returncode
        == 0
        else Path("/home/ubuntu/JopeDime")
    )
    src = subprocess.check_output(
        ["git", "show", ELO_COMMIT + ":" + ELO_PATH], cwd=git_cwd
    )
    if hashlib.sha256(src).hexdigest() != ELO_SHA:
        raise RuntimeError("elo_tracker source SHA mismatch")
    mod = types.ModuleType("h147_elo_tracker")
    mod.__file__ = ELO_PATH
    sys.modules[mod.__name__] = mod
    exec(compile(src, ELO_PATH, "exec"), mod.__dict__)
    return mod


def assert_frozen_anchor(net, names, anchor_sha=None):
    """Assert that adapter_10tok.anchor is a frozen non-adapter module with zero trainable parameters or state drift."""
    anchor = getattr(net.adapter_10tok, "anchor", None)
    if anchor is None:
        raise RuntimeError("missing adapter_10tok.anchor module")
    if anchor.training:
        raise RuntimeError("anchor module must be in eval mode")
    if any(p.requires_grad for p in anchor.parameters()):
        raise RuntimeError("anchor parameters must be frozen (requires_grad=False)")
    name_set = set(names)
    if any(n.startswith("adapter_10tok.anchor.") for n in name_set) or any(
        f"adapter_10tok.anchor.{k}" in name_set for k in anchor.state_dict()
    ):
        raise RuntimeError(
            "anchor tensors must be excluded from trainable adapter names"
        )
    cur_sha = tensor_hash(anchor.state_dict())
    if anchor_sha is not None and cur_sha != anchor_sha:
        raise RuntimeError("anchor state_dict bitwise drift")
    return cur_sha


def eval_batch_cache_key(split_name, cids, indexes, pools):
    """Build evaluation/step cache key from (split_name, tuple(context_ids), candidate_set_sha256)."""
    import numpy as np

    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError("split_name must be a non-empty string")
    cids_tuple = tuple(str(c) for c in cids)
    if not cids_tuple or len(cids_tuple) != len(indexes):
        raise ValueError("cids and indexes must be non-empty and equal length")
    h = hashlib.sha256()
    h.update(split_name.encode("utf-8"))
    h.update(b"\x00")
    for cid, ix in zip(cids_tuple, indexes, strict=True):
        ix_arr = np.ascontiguousarray(np.asarray(ix, dtype=np.int64))
        p = pools[cid]
        prog = np.ascontiguousarray(p["program"][ix_arr])
        rev_k = min(4, len(ix_arr))
        rev_fit = np.ascontiguousarray(p["fitness"][ix_arr[:rev_k]])
        rev_gate = np.ascontiguousarray(p["gate"][ix_arr[:rev_k]])
        h.update(cid.encode("utf-8"))
        h.update(ix_arr.view(np.uint8).tobytes())
        h.update(prog.view(np.uint8).tobytes())
        h.update(rev_fit.view(np.uint8).tobytes())
        h.update(rev_gate.view(np.uint8).tobytes())
    return (split_name, cids_tuple, h.hexdigest())


def validate_eval_cache_key(key):
    """Require cache key to be (split_name, tuple(context_ids), 64-hex candidate_set_sha256)."""
    if (
        not isinstance(key, tuple)
        or len(key) != 3
        or not isinstance(key[0], str)
        or not key[0].strip()
        or not isinstance(key[1], tuple)
        or not key[1]
        or not all(isinstance(c, str) and c for c in key[1])
        or not isinstance(key[2], str)
        or len(key[2]) != 64
    ):
        raise RuntimeError(
            "eval cache key must be (split_name, tuple(context_ids), candidate_set_sha256)"
        )
    return key


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
    guard = types.ModuleType("h147_guard")
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
    anchor_sha = assert_frozen_anchor(net, names)
    net._anchor_sha256 = anchor_sha
    frozen = tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
    net._eval_batch_key = None
    anchor_cache = {}
    sem_cache = {}
    orig_anchor = net.adapter_10tok.anchor.predict_costs
    orig_sem24 = fused.compute_semantic_struct_sig_24d_torch
    net._anchor_cache = anchor_cache
    net._sem24_cache = sem_cache

    def cached_anchor(**kw):
        if net.adapter_10tok.anchor.training or any(
            p.requires_grad for p in net.adapter_10tok.anchor.parameters()
        ):
            raise RuntimeError("anchor must remain a frozen eval-mode module")
        key = getattr(net, "_eval_batch_key", None)
        if key is None:
            return orig_anchor(**kw)
        validate_eval_cache_key(key)
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
        validate_eval_cache_key(key)
        if key not in sem_cache:
            sem_cache[key] = orig_sem24(prog_412).detach()
        return sem_cache[key]

    net.adapter_10tok.anchor.predict_costs = cached_anchor
    fused.compute_semantic_struct_sig_24d_torch = cached_sem24
    return net, guard, names, frozen, fused


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
    assert_frozen_anchor(net, names, getattr(net, "_anchor_sha256", None))
    if (
        tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
        != frozen
    ):
        raise RuntimeError("non-adapter bitwise drift")
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE drift")


def target_from_fitness_gate(fitness, gate):
    import torch

    return torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))


def loss(pred, fitness, gate, mask_revealed=False):
    """H147 SmoothL1 (beta=1) + 0.1 * KL(target || pred, tau=0.2); when mask_revealed=True, uses only unrevealed slots 4..191."""
    import torch.nn.functional as F

    if pred.ndim != 2 or pred.shape != fitness.shape or pred.shape != gate.shape:
        raise ValueError("pred, fitness, and gate must share 2D shape [B, M]")
    if mask_revealed:
        if pred.shape[1] <= 4:
            raise ValueError("mask_revealed requires > 4 candidates per slate")
        pred_eff = pred[:, 4:]
        fit_eff = fitness[:, 4:]
        gate_eff = gate[:, 4:]
    else:
        pred_eff = pred
        fit_eff = fitness
        gate_eff = gate

    target = target_from_fitness_gate(fit_eff, gate_eff)
    return (
        F.smooth_l1_loss(pred_eff, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred_eff / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()


def revealed_loss_and_grad_diagnostics(pred, fitness, gate, mask_revealed=False):
    """Compute detached diagnostic fraction of SmoothL1 loss and d(loss)/d(pred) norm^2 on positions 0..3."""
    import torch
    import torch.nn.functional as F

    with torch.enable_grad():
        p_probe = pred.detach().clone().requires_grad_(True)
        obj = loss(
            p_probe, fitness.detach(), gate.detach(), mask_revealed=mask_revealed
        )
        (g_pred,) = torch.autograd.grad(obj, p_probe, create_graph=False)
    with torch.no_grad():
        g_tot = float(torch.sum(g_pred.double() ** 2).item())
        g_rev = float(torch.sum(g_pred[:, :4].double() ** 2).item())
        grad_mass_share = g_rev / g_tot if g_tot > 0.0 else 0.0
        if mask_revealed:
            loss_mass_share = 0.0
        else:
            target = target_from_fitness_gate(fitness.detach(), gate.detach())
            sl1 = F.smooth_l1_loss(pred.detach(), target, reduction="none")
            kl = 0.1 * F.kl_div(
                F.log_softmax(-pred.detach() / 0.2, dim=-1),
                F.softmax(-target / 0.2, dim=-1),
                reduction="none",
            )
            slot_cost = (sl1 / pred.shape[1]) + kl
            tot_c = float(slot_cost.double().sum().item())
            rev_c = float(slot_cost[:, :4].double().sum().item())
            loss_mass_share = rev_c / tot_c if tot_c > 0.0 else 0.0
    return dict(
        revealed_loss_share=float(loss_mass_share),
        revealed_pred_grad_norm2_share=float(grad_mass_share),
    )
