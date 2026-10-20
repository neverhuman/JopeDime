"""Outcome-free posterior-sampling policy and frozen parent bootstrap."""
import math
import random
from protocol import digest


def priorities(available, predictions):
    choices = sorted(set(available))
    if not choices or not predictions or choices[0] < 0:
        raise ValueError('nonempty available candidates and ensemble required')
    if any(len(row) <= choices[-1] or any(not math.isfinite(float(row[k])) for k in choices) for row in predictions):
        raise ValueError('finite prediction per available candidate required')
    return [max(choices, key=lambda k: (float(row[k]), -k)) for row in predictions]


def thompson_probabilities(available, predictions):
    choices = sorted(set(available))
    votes = priorities(choices, predictions)
    return {k: .25 / len(choices) + .75 * votes.count(k) / len(votes) for k in choices}


def thompson_pick(available, predictions, rng):
    choices = sorted(set(available))
    votes = priorities(choices, predictions)
    member = int(rng.choice(list(range(len(votes)))))
    picked = votes[member] if rng.random() < .75 else int(rng.choice(choices))
    probability = thompson_probabilities(choices, predictions)[picked]
    return picked, probability, member


def parent_bootstrap_weights(parents, member_seed):
    weights = {}
    for parent in sorted(set(parents)):
        rng = random.Random(int(digest([member_seed, parent])[:16], 16))
        product, count = 1., 0
        while product > math.exp(-1):
            if count >= 100:
                raise ValueError('bootstrap sampler exceeded finite bound')
            product *= rng.random()
            count += 1
        weights[parent] = count - 1
    return weights
