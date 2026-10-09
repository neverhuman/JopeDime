"""Measured-tilt hand math, source refusal and train-role isolation."""

from dataclasses import replace
import hashlib
import math
from types import SimpleNamespace
import unittest

from rlfw.exit_targets import TargetConfig, build_targets, tilt_weights

SHA = "a" * 64


def config(form="set"):
    return TargetConfig(*([SHA] * 7), form=form)


def row(index, q):
    text = "p" + str(index)
    return {
        "context_id": "train-a",
        "lineage_id": "parent-a",
        "role": "train",
        "source": "measured",
        "outcome_class": "qualified_complete",
        "evaluator_sha256": SHA,
        "canonicalizer_sha256": SHA,
        "protocol": config().protocol,
        "role_map_sha256": SHA,
        "hwm_key_sha256": SHA,
        "receipt_sha256": str(index + 1) * 64,
        "q": q,
        "q_opt": max(-math.log(1000), min(math.log(1000), q)),
        "q_grid": round(q * 1_000_000),
        "program": {
            "dsl": text,
            "program_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokens": [1, index + 3, 2],
        },
        "iid_group_id": "draw-group",
        "iid_group_size": 2,
        "draw_index": index,
        "selection_rule": "iid-trie-sample",
        "sampling_checkpoint_sha256": SHA,
    }


def proof(value):
    return {
        **{
            k: value[k]
            for k in (
                "context_id",
                "lineage_id",
                "hwm_key_sha256",
                "receipt_sha256",
                "q",
                "q_opt",
                "q_grid",
            )
        },
        "program_sha256": value["program"]["program_sha256"],
        "qualified": True,
        "closure_verified": True,
        "parity_verified": True,
        "parity_tolerance": 1e-9,
    }


class TargetTests(unittest.TestCase):
    def build(self, rows, cfg=None, role=None, verify=proof, probabilities=None):
        trie = SimpleNamespace(
            identity=SHA,
            programs={tuple(r["program"]["tokens"]): r["program"] for r in rows},
        )
        return build_targets(
            rows,
            cfg or config(),
            trie=trie,
            role_check=role or (lambda c: {"role": "train", "role_map_sha256": SHA}),
            verify_receipt=verify,
            anchor_logprob=lambda c, t: math.log((probabilities or {}).get(t, 0.5)),
        )

    def test_tilt_derivative_is_measured_variance_and_mean_is_monotone(self):
        q = [-0.4, 0.1, 0.5]
        logp = [math.log(x) for x in (0.2, 0.5, 0.3)]

        def mean(eta):
            w, _ = tilt_weights(q, q, logp, replace(config(), temperature=1 / eta))
            return math.fsum(a * b for a, b in zip(w, q)), w

        prior = -math.inf
        for eta in (0.1, 0.5, 1.0, 2.0, 4.0):
            value, _ = mean(eta)
            self.assertGreaterEqual(value, prior)
            prior = value
        value, w = mean(2.0)
        variance = math.fsum(
            weight * (score - value) ** 2 for weight, score in zip(w, q)
        )
        derivative = (mean(2.0 + 1e-5)[0] - mean(2.0 - 1e-5)[0]) / 2e-5
        self.assertAlmostEqual(derivative, variance, delta=1e-6)

    def test_set_anchor_factor_and_iid_weights_differ_by_design(self):
        rows = [row(0, 0.0), row(1, 0.2)]
        probabilities = {(1, 3, 2): 0.9, (1, 4, 2): 0.1}
        set_target = self.build(rows, probabilities=probabilities)["manifest"][
            "contexts"
        ][0]
        iid_target = self.build(rows, config("raw_iid"), probabilities=probabilities)[
            "manifest"
        ]["contexts"][0]

        def weights(target):
            return {p["q"]: p["weight"] for p in target["programs"]}

        self.assertAlmostEqual(
            weights(set_target)[0.2], math.e * 0.1 / (0.9 + math.e * 0.1), places=12
        )
        self.assertAlmostEqual(
            weights(iid_target)[0.2], math.e / (1 + math.e), places=12
        )
        self.assertAlmostEqual(set_target["certified_anchor_mass"], 1.0)
        root = next(p for p in set_target["prefixes"] if p["prefix"] == (1,))
        self.assertAlmostEqual(root["reach_mass"], 1.0)
        self.assertAlmostEqual(sum(p for t, p in root["children"]), 1.0)
        self.assertTrue(
            all(
                p["children"] == ((2, 1.0),)
                for p in set_target["prefixes"]
                if len(p["prefix"]) == 2
            )
        )

    def test_raw_iid_generalized_alpha_matches_target_density(self):
        q = (0.0, 0.2)
        probabilities = (0.9, 0.1)
        logp = tuple(math.log(p) for p in probabilities)
        for alpha in (0.0, 0.5, 1.0, 2.0):
            with self.subTest(alpha=alpha):
                iid, baseline = tilt_weights(
                    q, q, logp, replace(config("raw_iid"), alpha=alpha)
                )
                numerator = [
                    p ** (alpha - 1) * math.exp(value / 0.2)
                    for p, value in zip(probabilities, q)
                ]
                expected = [x / math.fsum(numerator) for x in numerator]
                for actual, reference in zip(iid, expected):
                    self.assertAlmostEqual(actual, reference, places=12)
                self.assertEqual(baseline, 0.1)
                # Recover the same generalized target after integrating over
                # the sampling distribution on this complete two-leaf support.
                integrated = [p * weight for p, weight in zip(probabilities, iid)]
                total = math.fsum(integrated)
                set_weights, _ = tilt_weights(
                    q, q, logp, replace(config(), alpha=alpha)
                )
                for actual, reference in zip(integrated, set_weights):
                    self.assertAlmostEqual(actual / total, reference, places=12)

    def test_raw_iid_nondefault_alpha_preserves_duplicate_draw_and_prefix_mass(self):
        rows = [row(0, 0.0), row(0, 0.0), row(1, 0.2)]
        for index, value in enumerate(rows):
            value.update(draw_index=index, iid_group_size=3)
        probabilities = {(1, 3, 2): 0.9, (1, 4, 2): 0.1}
        for alpha in (0.0, 2.0):
            with self.subTest(alpha=alpha):
                target = self.build(
                    rows,
                    replace(config("raw_iid"), alpha=alpha),
                    probabilities=probabilities,
                )["manifest"]["contexts"][0]
                numerator = [
                    probabilities[tuple(r["program"]["tokens"])] ** (alpha - 1)
                    * math.exp(r["q"] / 0.2)
                    for r in rows
                ]
                expected = [x / math.fsum(numerator) for x in numerator]
                self.assertEqual(len(target["programs"]), 3)
                for program, reference in zip(target["programs"], expected):
                    self.assertAlmostEqual(program["weight"], reference, places=12)
                self.assertAlmostEqual(target["certified_anchor_mass"], 1.0)
                root = next(p for p in target["prefixes"] if p["prefix"] == (1,))
                children = dict(root["children"])
                self.assertAlmostEqual(
                    children[3], expected[0] + expected[1], places=12
                )
                self.assertAlmostEqual(children[4], expected[2], places=12)
                self.assertAlmostEqual(root["reach_mass"], 1.0)

    def test_set_deduplicates_but_iid_preserves_complete_draws(self):
        rows = [row(0, 0.0), row(0, 0.0), row(1, 0.2)]
        for index, r in enumerate(rows):
            r.update(draw_index=index, iid_group_size=3)
        set_t = self.build(rows)["manifest"]["contexts"][0]
        iid_t = self.build(rows, config("raw_iid"))["manifest"]["contexts"][0]
        self.assertEqual(len(set_t["programs"]), 2)
        self.assertEqual(len(iid_t["programs"]), 3)
        self.assertAlmostEqual(iid_t["certified_anchor_mass"], 1.0)
        bad = [{**r, "selection_rule": "ranked"} for r in rows]
        with self.assertRaises(ValueError):
            self.build(bad, config("raw_iid"))

    def test_dev_refused_before_qualification_filter_or_receipt_access(self):
        dev = {**row(0, 0.0), "role": "development", "outcome_class": "infra_unknown"}
        with self.assertRaises(ValueError):
            self.build(
                [dev],
                role=lambda c: {"role": "development", "role_map_sha256": SHA},
                verify=lambda r: self.fail("must reject dev before closure access"),
            )

    def test_predicted_q_cannot_replace_measured_q_even_with_opposite_order(self):
        measured = [row(0, 0.0), row(1, 0.2)]
        for r in measured:
            r["q_hat"] = -r["q"]
        with self.assertRaises(ValueError):
            self.build(measured)

    def test_closure_parity_and_raw_quality_integrity(self):
        rows = [row(0, 0.0), row(1, 0.2)]
        with self.assertRaises(ValueError):
            self.build(rows, verify=lambda r: {**proof(r), "parity_tolerance": 1e-8})
        with self.assertRaises(ValueError):
            self.build([{**rows[0], "q_grid": 123}, rows[1]])
        with self.assertRaises(ValueError):
            self.build([{**rows[0], "source": "predicted"}, rows[1]])
        with self.assertRaises(ValueError):
            self.build([{**rows[0], "outcome_class": "infra_unknown"}, rows[1]])
        with self.assertRaises(ValueError):
            self.build([{**rows[0], "hwm_key_sha256": "b" * 64}, rows[1]])

    def test_singleton_equal_scores_and_incomplete_iid_have_zero_actor_support(self):
        for rows, cfg in [
            ([row(0, 0.1)], config()),
            ([row(0, 0.1), row(1, 0.1)], config()),
        ]:
            target = self.build(rows, cfg)["manifest"]["contexts"][0]
            self.assertFalse(target["eligible"])
            self.assertEqual(target["programs"], ())
            self.assertEqual(target["prefixes"], ())
        unknown = {
            **row(1, 0.2),
            "outcome_class": "infra_unknown",
            "q": None,
            "q_opt": None,
            "q_grid": None,
        }
        target = self.build([row(0, 0.0), unknown], config("raw_iid"))["manifest"][
            "contexts"
        ][0]
        self.assertFalse(target["eligible"])

    def test_conflicting_duplicate_and_raw_iid_inventory_refused(self):
        with self.assertRaises(ValueError):
            self.build([row(0, 0.0), row(0, 0.1), row(1, 0.2)])
        with self.assertRaises(ValueError):
            self.build(
                [row(0, 0.0), {**row(1, 0.2), "draw_index": 0}], config("raw_iid")
            )

    def test_fresh_role_is_checked_again_before_final_manifest(self):
        calls = []

        def roles(c):
            calls.append(c)
            return {
                "role": "train" if len(calls) <= 2 else "development",
                "role_map_sha256": SHA,
            }

        with self.assertRaises(ValueError):
            self.build([row(0, 0.0), row(1, 0.2)], role=roles)

    def test_canonical_catalog_and_receipt_program_identity_are_bound(self):
        from copy import deepcopy

        rows = [row(0, 0.0), row(1, 1.0)]
        catalog = {
            tuple(r["program"]["tokens"]): {
                k: r["program"][k] for k in ("dsl", "program_sha256")
            }
            for r in rows
        }
        measured = {r["receipt_sha256"]: deepcopy(r) for r in rows}

        def build(items, verify):
            return build_targets(
                items,
                config(),
                trie=SimpleNamespace(identity=SHA, programs=catalog),
                role_check=lambda c: {"role": "train", "role_map_sha256": SHA},
                verify_receipt=verify,
                anchor_logprob=lambda c, t: math.log(0.5),
            )

        def verified(r):
            return proof(measured[r["receipt_sha256"]])

        target = build(rows, verified)["manifest"]["contexts"][0]
        self.assertTrue(target["eligible"])
        swapped = deepcopy(rows)
        for i, r in enumerate(swapped):
            r["program"]["tokens"] = rows[1 - i]["program"]["tokens"]
        with self.assertRaisesRegex(ValueError, "canonical trie leaf"):
            build(swapped, verified)

        def wrong_receipt(r):
            return {**verified(r), "program_sha256": "b" * 64}

        with self.assertRaisesRegex(ValueError, "receipt closure"):
            build(rows, wrong_receipt)
        outside = deepcopy(rows)
        outside[1]["program"]["tokens"] = [1, 8, 2]
        self.assertFalse(
            build(outside, verified)["manifest"]["contexts"][0]["eligible"]
        )


if __name__ == "__main__":
    unittest.main()
