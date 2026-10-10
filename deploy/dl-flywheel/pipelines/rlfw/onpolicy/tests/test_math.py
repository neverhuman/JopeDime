"""Independent CDF derivative, multinomial enumeration, subsets and controls."""

import itertools
import math
import random
import unittest

from rlfw.onpolicy.objective import pkpo_weights, rloo_weights

Z = (-0.4, 0.8, -0.1)
Q = (-0.7, 1.2, 0.3)
H = 0.2


def softmax(z):
    e = [math.exp(x - max(z)) for x in z]
    return tuple(x / math.fsum(e) for x in e)


def analytic(z, q, h, b):
    """Independent CDF formula and derivative; no estimator weights used."""
    p = softmax(z)
    levels = sorted(set(max(h, x) for x in q))
    value, gradient, previous, previous_grad = 0.0, [0.0] * len(q), 0.0, [0.0] * len(q)
    for level in levels:
        mask = [max(h, x) <= level for x in q]
        cdf = math.fsum(x for x, included in zip(p, mask) if included)
        cdf_grad = [p[i] * (float(mask[i]) - cdf) for i in range(len(q))]
        mass, derivative = cdf**b, [b * cdf ** (b - 1) * x for x in cdf_grad]
        value += (level - h) * (mass - previous)
        for i in range(len(q)):
            gradient[i] += (level - h) * (derivative[i] - previous_grad[i])
        previous, previous_grad = mass, derivative
    return value, tuple(gradient)


def counts(g):
    for n0 in range(g + 1):
        for n1 in range(g - n0 + 1):
            yield (n0, n1, g - n0 - n1)


def exhaustive(
    p, q, h, b, g, *, behavior=None, correction="none", dense=False, std=False
):
    """Exactly exhaust every ordered group via permutation-invariant counts.

    G=16 has 43,046,721 ordered groups, represented by 153 count vectors.
    Multiplicity is the exact multinomial coefficient, not a sample or dedup.
    """
    mu = behavior or p
    terms = [[], [], []]
    for n in counts(g):
        actions = tuple(i for i in range(3) for _ in range(n[i]))
        mass = (
            math.factorial(g)
            / math.prod(math.factorial(x) for x in n)
            * math.prod(mu[i] ** n[i] for i in range(3))
        )
        values = tuple(q[i] for i in actions)
        weights = rloo_weights(values) if dense else pkpo_weights(values, h, b)
        if std:
            mean = math.fsum(values) / g
            scale = max(
                1e-8, math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (g - 1))
            )
            weights = tuple(x / scale for x in weights)
        joint = (
            math.prod(p[i] / mu[i] for i in actions) if correction == "joint" else 1.0
        )
        for k in range(3):
            estimate = math.fsum(
                w
                * (p[i] / mu[i] if correction == "item" else joint)
                * (float(i == k) - p[k])
                for i, w in zip(actions, weights)
            )
            terms[k].append(mass * estimate)
    return tuple(math.fsum(t) for t in terms)


def finite(z, q, h, b):
    epsilon = 1e-5
    out = []
    for i in range(len(z)):
        plus, minus = list(z), list(z)
        plus[i] += epsilon
        minus[i] -= epsilon
        out.append(
            (analytic(plus, q, h, b)[0] - analytic(minus, q, h, b)[0]) / (2 * epsilon)
        )
    return tuple(out)


class MathTests(unittest.TestCase):
    def assertVector(self, actual, expected, tolerance=1e-9):
        self.assertLessEqual(
            max(abs(x - y) for x, y in zip(actual, expected)), tolerance
        )

    def test_all_eight_B_G_combinations_match_analytic_and_finite_difference(self):
        p = softmax(Z)
        for b in (1, 2, 4, 8):
            for g in (b, 2 * b):
                with self.subTest(B=b, G=g):
                    exact = exhaustive(p, Q, H, b, g)
                    self.assertVector(exact, analytic(Z, Q, H, b)[1], 1e-12)
                    self.assertVector(exact, finite(Z, Q, H, b))

    def test_multinomial_compression_equals_literal_ordered_enumeration(self):
        p = softmax(Z)
        for b, g in ((1, 2), (2, 4), (4, 4)):
            terms = [[], [], []]
            for actions in itertools.product(range(3), repeat=g):
                mass = math.prod(p[i] for i in actions)
                weights = pkpo_weights([Q[i] for i in actions], H, b)
                for k in range(3):
                    terms[k].append(
                        mass
                        * math.fsum(
                            w * (float(i == k) - p[k]) for i, w in zip(actions, weights)
                        )
                    )
            self.assertVector(
                tuple(math.fsum(x) for x in terms), exhaustive(p, Q, H, b, g), 1e-12
            )

    def test_PKPO_equals_independent_subset_marginals_with_ties(self):
        u = (0.1, 1.2, 1.2, -0.5, 0.3, 0.3, 0.8, 0.0)
        for b in (1, 2, 4, 8):
            out = [0.0] * len(u)
            for subset in itertools.combinations(range(len(u)), b):
                for i in subset:
                    baseline = max((H, *(u[j] for j in subset if i != j)))
                    out[i] += max(0.0, u[i] - baseline) / math.comb(len(u), b)
            self.assertVector(pkpo_weights(u, H, b), out, 1e-12)

    def test_RLOO_unbiased_and_singleton_fixed_baseline(self):
        p = softmax(Z)
        mean = math.fsum(x * y for x, y in zip(p, Q))
        expected = tuple(p[i] * (Q[i] - mean) for i in range(3))
        for g in (1, 2, 4, 8, 16):
            self.assertVector(exhaustive(p, Q, H, 1, g, dense=True), expected, 1e-12)

    def test_std_normalized_variant_is_measurably_biased_and_refused(self):
        p = softmax(Z)
        unbiased = exhaustive(p, Q, H, 1, 4, dense=True)
        biased = exhaustive(p, Q, H, 1, 4, dense=True, std=True)
        self.assertGreater(max(abs(x - y) for x, y in zip(unbiased, biased)), 0.01)
        with self.assertRaisesRegex(ValueError, "biased"):
            rloo_weights(Q, std_normalize=True)

    def test_stale_dense_item_IS_and_HWM_joint_IS_are_unbiased(self):
        p, mu = softmax(Z), softmax((0.0, 0.4, 0.2))
        for b, g in ((1, 2), (2, 4), (4, 8), (8, 16)):
            self.assertVector(
                exhaustive(p, Q, H, b, g, behavior=mu, correction="joint"),
                analytic(Z, Q, H, b)[1],
                1e-12,
            )
        self.assertVector(
            exhaustive(p, Q, H, 1, 8, behavior=mu, correction="item", dense=True),
            exhaustive(p, Q, H, 1, 8, dense=True),
            1e-12,
        )
        wrong = exhaustive(p, Q, H, 4, 8, behavior=mu, correction="item")
        self.assertGreater(
            max(abs(x - y) for x, y in zip(wrong, analytic(Z, Q, H, 4)[1])), 0.01
        )

    def test_inverted_reward_control_decreases_true_objective(self):
        gradient = exhaustive(softmax(Z), Q, H, 2, 4)
        before = analytic(Z, Q, H, 2)[0]
        good = tuple(z + 0.1 * g for z, g in zip(Z, gradient))
        # Inverting reward negates the exact same score-function update.
        inverted = tuple(z - 0.1 * g for z, g in zip(Z, gradient))
        self.assertGreater(analytic(good, Q, H, 2)[0], before)
        self.assertLess(analytic(inverted, Q, H, 2)[0], before)

    def test_seeded_MC_agrees_with_exact_with_reported_sampling_uncertainty(self):
        p, rng, n = softmax(Z), random.Random(731), 8000
        for b in (1, 2, 4, 8):
            for g in (b, 2 * b):
                sums, squares = [0.0] * 3, [0.0] * 3
                for _ in range(n):
                    actions = rng.choices(range(3), weights=p, k=g)
                    w = pkpo_weights([Q[i] for i in actions], H, b)
                    for k in range(3):
                        x = math.fsum(
                            v * (float(i == k) - p[k]) for i, v in zip(actions, w)
                        )
                        sums[k] += x
                        squares[k] += x * x
                exact = analytic(Z, Q, H, b)[1]
                for k in range(3):
                    mean = sums[k] / n
                    se = math.sqrt(max(0.0, squares[k] / n - mean * mean) / (n - 1))
                    self.assertLessEqual(abs(mean - exact[k]), 6 * se + 1e-12)

    def test_numerical_primitives_refuse_nonfinite_and_invalid_B(self):
        for values in ((), (math.nan,), (math.inf,)):
            with self.assertRaises(ValueError):
                rloo_weights(values)
        for b in (0, 3, True):
            with self.assertRaises(ValueError):
                pkpo_weights((1.0, 2.0), H, b)


if __name__ == "__main__":
    unittest.main()
