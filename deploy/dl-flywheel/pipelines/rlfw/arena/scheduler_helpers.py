import random

from .arms import Candidate


def context_order(spec):
    order = list(spec.contexts)
    random.Random(spec.order_seed).shuffle(order)
    if spec.schedule == "one-per-lineage-then-all":
        first, rest, seen = [], [], set()
        for c in order:
            (rest if c.lineage_id in seen else first).append(c)
            seen.add(c.lineage_id)
        order = first + rest
    return order


def candidate_value(c):
    return c.to_dict()


def load_candidate(value):
    return Candidate.from_dict(value)

