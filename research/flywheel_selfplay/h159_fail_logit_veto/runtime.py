"""H159 input auditing, raw-byte frozen r882 state checks, and user-visible disk/host resource guards (zero-update)."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import psutil

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
GUARD_COMMIT = "a1b08cbfabe77410a7c323762146c85a50d7f16f"
GUARD_PATH = "python/src/jope/north_star/dope_freeze_guard.py"
GUARD_SHA = "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c"
INPUT = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache/context_features.npz")
H159_DATA = Path("/home/ubuntu/jope-data/h159")
HERE = Path(__file__).resolve().parent

MIN_NICE = 19
DISK_USED_MAX = 0.95
DISK_FREE_MIN_BYTES = 40_000_000_000
SCRATCH_LIMIT_BYTES = 3_000_000_000
LOAD_MAX = 85.0
MEM_AVAIL_MIN_BYTES = 16 * 1024**3


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def disk_stats(path):
    fs = os.statvfs(path)
    if fs.f_blocks <= 0:
        raise RuntimeError("filesystem reports no blocks")
    used_blocks = fs.f_blocks - fs.f_bfree
    used_with_user_available = used_blocks + fs.f_bavail
    if used_blocks < 0 or used_with_user_available <= 0:
        raise RuntimeError("filesystem reports invalid block accounting")
    return dict(
        free=fs.f_bavail * fs.f_frsize,
        disk_used=used_blocks / used_with_user_available,
    )


def scratch_bytes(tree=H159_DATA):
    p = Path(tree)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def guards(path):
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H159 requires nice >= 19")
    ds = disk_stats(path)
    if ds["disk_used"] >= DISK_USED_MAX or ds["free"] < DISK_FREE_MIN_BYTES:
        raise RuntimeError(
            f"disk stop: used {ds['disk_used']:.4f}, free {ds['free']} B"
        )
    if scratch_bytes() >= SCRATCH_LIMIT_BYTES:
        raise RuntimeError("scratch limit exceeded")
    if os.getloadavg()[0] > LOAD_MAX:
        raise RuntimeError("host load limit exceeded")
    if psutil.virtual_memory().available < MEM_AVAIL_MIN_BYTES:
        raise RuntimeError("host MemAvailable below 16 GiB")


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
        micro = s.startswith("/home/ubuntu/JopeDimeMicro/")
        if (
            protected
            or (data and s not in self.allowed and not p.is_relative_to(self.output))
            or (
                write
                and (
                    s in self.allowed
                    or (data and not p.is_relative_to(self.output))
                    or micro
                )
            )
        ):
            self.refused.append(s)
            raise RuntimeError("input/output boundary refusal: " + s)
        if data or micro:
            self.opened.add(s)

    def receipt(self):
        return dict(opened=sorted(self.opened), refused=self.refused, protected_reads=0)


def load_fused():
    p = INPUT / "cpu_exact_fused.py"
    m = types.ModuleType("h159_fused")
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
    guard = types.ModuleType("h159_guard")
    sys.modules[guard.__name__] = guard
    exec(compile(src, GUARD_PATH, "exec"), guard.__dict__)
    names = tuple(
        n
        for n, _ in net.named_parameters()
        if n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")
    )
    if len(names) != 47 or sum(net.state_dict()[n].numel() for n in names) != 29211:
        raise RuntimeError("adapter inventory")
    for p in net.parameters():
        p.requires_grad_(False)
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE mismatch")
    frozen = tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
    adapter_hash = tensor_hash({n: net.state_dict()[n] for n in names})
    return net, guard, names, frozen, adapter_hash, fused


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


def check_frozen(net, guard, names, frozen, adapter_hash):
    if (
        tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
        != frozen
    ):
        raise RuntimeError("non-adapter bitwise drift")
    if tensor_hash({n: net.state_dict()[n] for n in names}) != adapter_hash:
        raise RuntimeError("adapter bitwise drift in zero-update H159")
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("DOPE drift")
