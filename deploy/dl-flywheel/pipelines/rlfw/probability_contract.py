"""Differentiable, EOS-inclusive actor likelihood on a frozen program trie.

This adapter calls PolicyLM.forward with the decoder's prefix/PAD convention.
It deliberately leaves the continuous worker's grammar-masked objective alone.
Torch is imported only when the adapter runs; the core RLFW package stays lazy.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActorLikelihood:
    total: object
    per_token: tuple
    support_sha256: str
    aggregation: str = "sum_including_eos"


def actor_logprob(net, ctx_z, token_rows, trie, *, temperature=1.0):
    """Return differentiable sequence sums, retaining raw tokens and row order.

    Dropout is disabled while computing the actor distribution. The caller may
    wrap this function in no_grad for inference; training keeps the graph.
    Shapes and active rows match BatchedTrie.decode_many at every prefix.
    """
    import torch

    rows = tuple(tuple(row) for row in token_rows)
    if not math.isfinite(temperature) or not 0.1 <= temperature <= 2:
        raise ValueError("decoder temperature must be in [0.1, 2]")
    if (
        not 1 <= len(rows) <= 1024
        or ctx_z.ndim != 2
        or ctx_z.shape != (len(rows), net.cfg.ctx_dim)
        or not torch.isfinite(ctx_z).all()
    ):
        raise ValueError("finite context matrix aligned with raw sequences required")
    if any(
        not row or row[0] != 1 or row[-1] != 2 or row not in trie.programs
        for row in rows
    ):
        raise ValueError("complete registered raw trie leaves including EOS required")
    parts = [[] for _ in rows]
    previous = net.training
    net.eval()
    try:
        for position in range(1, max(map(len, rows))):
            active = [i for i, row in enumerate(rows) if position < len(row)]
            tokens = torch.full(
                (len(active), position + 1),
                net.cfg.pad_id,
                dtype=torch.long,
                device=ctx_z.device,
            )
            for j, i in enumerate(active):
                tokens[j, :position] = torch.tensor(
                    rows[i][:position], device=ctx_z.device
                )
            lengths = torch.full(
                (len(active),), position + 1, dtype=torch.long, device=ctx_z.device
            )
            output = net(
                tokens, lengths, ctx_z, torch.tensor(active, device=ctx_z.device)
            )
            latest = output.logits[:, position - 1].double() / temperature
            for j, i in enumerate(active):
                children = sorted(trie.get(rows[i][:position]))
                chosen = rows[i][position]
                if not children or chosen not in children:
                    raise ValueError("raw action outside frozen trie children")
                supported = latest[j, children]
                if not torch.isfinite(supported).all():
                    raise ValueError("nonfinite supported actor logits")
                parts[i].append(latest[j, chosen] - torch.logsumexp(supported, dim=0))
    finally:
        net.train(previous)
    per_token = tuple(torch.stack(part) for part in parts)
    return ActorLikelihood(
        torch.stack([part.sum() for part in per_token]), per_token, trie.identity
    )


def weighted_archive_ce(likelihood, weights):
    """Weights are fixed measured archive targets, never length normalizers."""
    import torch

    weights = torch.as_tensor(
        weights, dtype=torch.float64, device=likelihood.total.device
    ).detach()
    if (
        weights.shape != likelihood.total.shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1e-12, rtol=0)
    ):
        raise ValueError("finite nonnegative unit-mass archive weights required")
    return -(weights * likelihood.total).sum()


def require_same_support_reference(actor_support, reference_support, parameter_sha):
    """A successor reference is the frozen start actor UNDER successor support."""
    if actor_support != reference_support:
        raise ValueError("cross-support forward KL is undefined for added mass")
    if parameter_sha != START_PARAMETER_SHA256:
        raise ValueError("reference must be the frozen ea078456 start checkpoint")


START_PARAMETER_SHA256 = (
    "ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb"
)
