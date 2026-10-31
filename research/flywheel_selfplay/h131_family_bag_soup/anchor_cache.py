"""Exact frozen-anchor memoization for full-pool scoring only."""

import hashlib
import json
import torch


def tensor_hash(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(b"\0")
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise RuntimeError("CPU tensors required")
            t = value.detach().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
        h.update(b"\0")
    return h.hexdigest()


class AnchorCache:
    def __init__(self):
        self.enabled = False
        self.identity = None
        self.entries = {}
        self.verified = set()
        self.hits = 0
        self.misses = 0

    def attach(self, anchor):
        state = tensor_hash(anchor.state_dict())
        if self.identity is not None and state != self.identity:
            raise RuntimeError("frozen anchor state differs")
        self.identity = state
        original = anchor.predict_costs

        def predict(**kwargs):
            if not self.enabled:
                return original(**kwargs)
            key = state + ":" + tensor_hash(kwargs)
            if key not in self.entries:
                out = original(**kwargs)
                self.entries[key] = {
                    n: out[n].detach().clone()
                    for n in ("total_cost", "eff_prior", "fail_logit")
                }
                self.misses += 1
            else:
                if key not in self.verified:
                    out = original(**kwargs)
                    if any(
                        not torch.equal(out[n], v) for n, v in self.entries[key].items()
                    ):
                        raise RuntimeError("anchor cache mismatch")
                    self.verified.add(key)
                self.hits += 1
            return self.entries[key]

        anchor.predict_costs = predict

    def receipt(self):
        return dict(
            anchor_sha256=self.identity,
            entries=len(self.entries),
            verified_keys=len(self.verified),
            hits=self.hits,
            misses=self.misses,
            bytes=sum(
                t.numel() * t.element_size()
                for d in self.entries.values()
                for t in d.values()
            ),
        )
