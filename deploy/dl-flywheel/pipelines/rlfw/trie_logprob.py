"""Summed, EOS-inclusive likelihood under the actual trie children support."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrieLogProb:
    total: float
    per_token: tuple[float, ...]
    support: str = "trie_children"
    aggregation: str = "sum_including_eos"

    def __post_init__(self):
        if self.support != "trie_children" or self.aggregation != "sum_including_eos":
            raise ValueError("only summed EOS-inclusive trie likelihood is accepted")
        if not all(math.isfinite(x) and x <= 0 for x in self.per_token):
            raise ValueError("finite nonpositive token log-probabilities required")
        if (
            not math.isfinite(self.total)
            or abs(math.fsum(self.per_token) - self.total) > 1e-12
        ):
            raise ValueError("total must sum token log-probabilities")


def child_logprob(logits, choices, chosen, temperature):
    """Python floats are float64; subtract the maximum before logsumexp."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    choices = tuple(sorted(choices))
    if not choices or chosen not in choices:
        raise ValueError("chosen token is outside trie children")
    values = [float(logits[token]) / temperature for token in choices]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("finite supported logits required")
    if len(choices) == 1:
        return 0.0
    peak = max(values)
    return (values[choices.index(chosen)] - peak) - math.log(
        math.fsum(math.exp(value - peak) for value in values)
    )


def trie_program_logprob(
    policy, ctx, tokens, trie, temperature=1.0, *, support="trie_children"
):
    """Policy supplies next_logits(ctx, prefix); trie supplies programs and get.

    Complete tokens include BOS and EOS. A grammar-normalized probability or
    mean-token NLL is not this API's input. Policy binding disables dropout.
    """
    if support != "trie_children":
        raise ValueError("grammar-normalized input is refused")
    if not hasattr(policy, "next_logits"):
        raise TypeError("raw logits policy required; mean-token NLL is not a policy")
    tokens = tuple(tokens)
    if tokens not in trie.programs:
        raise ValueError("complete registered trie leaf including EOS required")
    previous = getattr(policy, "training", None)
    if hasattr(policy, "eval"):
        policy.eval()
    try:
        logits = (
            policy.program_logits(ctx, tokens)
            if hasattr(policy, "program_logits")
            else None
        )
        per_token = tuple(
            child_logprob(
                logits[position - 1]
                if logits is not None
                else policy.next_logits(ctx, tokens[:position]),
                trie.get(tokens[:position]),
                chosen,
                temperature,
            )
            for position, chosen in enumerate(tokens[1:], 1)
        )
    finally:
        if previous is not None and hasattr(policy, "train"):
            policy.train(previous)
    return TrieLogProb(math.fsum(per_token), per_token)
