"""Average named adapter parameters only; preserve all r882 buffers/frozen tensors."""

import hashlib
from science import WEIGHTS, ROUNDS


def raw(tensor):
    import torch

    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        h.update(raw(tensor))
    return h.hexdigest()


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and raw(a) == raw(b)


def average(states, names, arm):
    import torch

    if arm not in WEIGHTS or set(states) != set(ROUNDS) or not names:
        raise ValueError("fixed arm, four checkpoints and named parameters required")
    base = states[882]
    if any(set(s) != set(base) for s in states.values()) or not set(names) <= set(base):
        raise ValueError("checkpoint tensor-key mismatch")
    result = {n: t.clone() for n, t in base.items()}
    for name in names:
        t = base[name]
        if not t.is_floating_point() or any(
            s[name].shape != t.shape
            or s[name].dtype != t.dtype
            or not torch.isfinite(s[name]).all()
            for s in states.values()
        ):
            raise ValueError("invalid adapter tensor")
        if arm == "R882":
            continue
        acc = torch.zeros_like(t, dtype=torch.float64)
        for round, weight in zip(ROUNDS, WEIGHTS[arm], strict=True):
            acc.add_(states[round][name].double(), alpha=weight)
        result[name] = acc.to(t.dtype)
    for name in set(base) - set(names):
        if not equal(result[name], base[name]):
            raise RuntimeError("frozen tensor/buffer modified")
    return result


def verify_dope(states, tensor_keys):
    if not tensor_keys:
        raise ValueError("empty DOPE identity")
    for state in states.values():
        for name in tensor_keys:
            if name not in state or not equal(state[name], states[882][name]):
                raise RuntimeError("DOPE tensor differs bitwise: " + name)
