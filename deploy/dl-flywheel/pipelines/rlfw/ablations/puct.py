"""Compressed-trie PUCT over complete programs.

Measured q is revealed only after a leaf is chosen. It is not an input to
the selection score. c = 1.25 and seed = 7 are pre-registered.
"""

from __future__ import annotations

import math

PUCT_C = 1.25
PUCT_SEED = 7


class _Node:
    def __init__(self):
        self.children = {}
        self.program_index = None
        self.prior = 0.0
        self.visits = 0


def _require_distribution(priors):
    if any((not math.isfinite(p)) or p <= 0 for p in priors):
        raise ValueError("priors must be finite and positive")
    if abs(math.fsum(priors) - 1.0) > 1e-9:
        raise ValueError("priors must sum to 1")


def _build(programs, priors):
    root = _Node()
    root.prior = 1.0
    for index, tokens in enumerate(programs):
        tokens = tuple(tokens)
        if len(tokens) < 2:
            raise ValueError("program needs BOS and EOS")
        node = root
        for token in tokens[1:]:
            if type(token) is not int:
                raise ValueError("token ids must be ints")
            child = node.children.get(token)
            if child is None:
                if node.program_index is not None:
                    raise ValueError("program is a prefix of another program")
                child = _Node()
                node.children[token] = child
            node = child
            node.prior += priors[index]
        if node.children:
            raise ValueError("program is a prefix of another program")
        if node.program_index is not None:
            raise ValueError("duplicate program")
        node.program_index = index
    return root


def _remaining(node, done):
    if node.program_index is not None:
        return node.program_index not in done
    return any(_remaining(child, done) for child in node.children.values())


def puct_select(programs, priors, q_of, budget, c=PUCT_C, seed=PUCT_SEED):
    """Select ``budget`` distinct programs. ``q_of(index)`` runs at the leaf."""
    if c != PUCT_C or seed != PUCT_SEED:
        raise ValueError("PUCT c=1.25 and seed=7 are pre-registered")
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive int")
    if len(programs) != len(priors) or len(programs) < budget:
        raise ValueError("budget exceeds the labelled support")
    _require_distribution(priors)
    root = _build(programs, priors)
    selected = []
    revealed = []
    done = set()
    while len(selected) < budget:
        node = root
        path = [root]
        while node.program_index is None:
            choices = [
                token
                for token, child in node.children.items()
                if _remaining(child, done)
            ]
            if not choices:
                raise RuntimeError("PUCT ran out of labelled programs")
            ranked = []
            for token in choices:
                child = node.children[token]
                conditional = child.prior / node.prior
                score = (
                    PUCT_C
                    * conditional
                    * math.sqrt(max(node.visits, 1))
                    / (1 + child.visits)
                )
                ranked.append((score, conditional, -token, token))
            ranked.sort(reverse=True)
            node = node.children[ranked[0][3]]
            path.append(node)
        for seen in path:
            seen.visits += 1
        done.add(node.program_index)
        selected.append(node.program_index)
        revealed.append(q_of(node.program_index))
    return {
        "selected": selected,
        "q": revealed,
        "c": PUCT_C,
        "seed": PUCT_SEED,
        "budget": budget,
        "uses_measured_q_for_selection": False,
    }
