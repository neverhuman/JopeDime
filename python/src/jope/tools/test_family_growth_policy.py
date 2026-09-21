import tempfile
import unittest
from pathlib import Path

import family_growth_policy as g


class GrowthFixture:
    def __init__(self, root, parent=None):
        self.root = root
        (root / "manifests").mkdir(exist_ok=True)
        (root / "credentials/provenance-review").mkdir(parents=True, exist_ok=True)
        (root / "credentials/provenance-review/test-reviewer.key").write_bytes(
            b"test-only-key-never-production!!!"
        )
        policy = dict(
            schema="parallel.family_growth_policy.v3", seed=g.REGISTRY_SEED, reviewers=["test-reviewer"]
        )
        self.policy = self.put("manifests/family-growth-policy.json", policy)
        if parent is None:
            parent = dict(
                schema="parallel.family_registry.v1",
                families={
                    "old-train": dict(source_keys=["old-train-alias"], partition="meta_train"),
                    "old-test": dict(source_keys=["old-test-alias"], partition="meta_test"),
                },
            )
        self.parent = self.put("parent.json", parent)
        source = self.put("source.json", dict(synthetic_unit_fixture=True))
        components = []
        for family, row in parent["families"].items():
            components.append(
                dict(component_id=family, aliases=sorted({family, *row["source_keys"]}), source_objects=[])
            )
        components.append(dict(component_id="new", aliases=["new", "new-alias"], source_objects=[source]))
        self.graph = dict(schema="parallel.component_graph.v1", components=components)
        self.graph_ref = self.put("graph.json", self.graph)
        prior = self.put(
            "prior-use.json", dict(component_id="new", occurrences=[], scope="synthetic-test-fixture")
        )
        self.decision = dict(
            schema="parallel.provenance_decision.v3",
            component_id="new",
            admission="approved",
            reviewer="test-reviewer",
            graph_sha256=self.graph_ref["sha256"],
            parent_registry_sha256=self.parent["sha256"],
            policy_sha256=self.policy["sha256"],
            task="regression",
            target="y",
            predictors=["x"],
            units="fixture-units",
            entity_time_contract=dict(strategy="iid"),
            source_objects=[source],
            prior_use_findings=dict(
                status="unused", parent_registry_sha256=self.parent["sha256"], evidence=[prior]
            ),
        )
        self.sign()

    def put(self, name, value):
        import hashlib
        import json

        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def sign(self):
        import hashlib
        import hmac

        ref = self.put("decision.json", self.decision)
        key = (self.root / "credentials/provenance-review/test-reviewer.key").read_bytes()
        ref["signature"] = hmac.new(
            key, ("parallel.family_growth.v3:" + ref["sha256"]).encode(), hashlib.sha256
        ).hexdigest()
        self.decisions = [ref]

    def assign(self, names):
        return g.assign_new_components(
            names,
            root=self.root,
            graph_ref=self.graph_ref,
            parent_registry_ref=self.parent,
            decision_receipts=self.decisions,
        )


class FamilyGrowthPolicyTests(unittest.TestCase):
    def test_hash_matches_families_rs_buckets(self):
        from binary_prepare import stable_partition

        hashed, bucket, partition = g.stable_assignment("example-family")
        self.assertEqual(len(hashed), 64)
        self.assertLess(bucket, 1_000_000)
        self.assertEqual(partition, stable_partition("example-family"))
        again = g.stable_assignment("example-family")
        self.assertEqual(again, (hashed, bucket, partition))

    def test_live_and_sealed_aliases_are_not_repartitioned(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = GrowthFixture(Path(folder))
            result = fixture.assign(["old-train-alias", "old-test-alias", "new-alias", "new"])
            self.assertEqual(result["excluded_live_components"], ["old-train"])
            self.assertEqual(result["sealed_test_components"], ["old-test"])
            self.assertEqual([r["component_id"] for r in result["rows"]], ["new"])
            self.assertEqual(result["rows"][0]["aliases"], ["new", "new-alias"])
            self.assertFalse(result["execution_eligible"])

    def test_non_regression_and_boolean_provenance_are_refused(self):
        with self.assertRaisesRegex(ValueError, "regression-only"):
            g.assign_new_components(["a"], task="binary")
        for value in (False, True):
            with self.assertRaisesRegex(ValueError, "provenance Booleans"):
                g.assign_new_components(["a"], provenance_resolved=value)
        with self.assertRaisesRegex(ValueError, "durable provenance"):
            g.assign_new_components(["a"])

    def test_synthetic_filter_uses_audit_fields_not_hex_names(self):
        self.assertTrue(g.is_synthetic({"source_real_data": False}, "aa" * 8))
        self.assertTrue(g.is_synthetic({"synthetic_origin": True}, "real-looking"))
        self.assertTrue(g.is_synthetic({}, "quant/synth_foo"))
        self.assertFalse(g.is_synthetic({"source_real_data": True}, "aa" * 8))

    def test_decisions_never_auto_admit(self):
        draft = g.decide_unresolved(
            [
                dict(dataset="real-1", source_path="/tmp/exists", meta={"source_real_data": True}),
                dict(dataset="binary/3c288e95bcdaf2fa", directory="binary/3c288e95bcdaf2fa"),
                dict(dataset="hexname", meta={"source_real_data": False}, directory="deadbeef"),
                dict(dataset="unresolved-bare", meta={}),
            ]
        )
        self.assertEqual(draft["auto_admitted"], 0)
        self.assertFalse(draft["writes_initial_decisions"])
        by_key = {row["key"]: row for row in draft["decisions"]}
        self.assertEqual(by_key["real-1"]["admission"], "unresolved")
        self.assertEqual(by_key["binary/3c288e95bcdaf2fa"]["admission"], "excluded")
        self.assertEqual(by_key["hexname"]["admission"], "excluded")
        self.assertEqual(by_key["unresolved-bare"]["admission"], "unresolved")

    def test_refuses_live_index_and_keeps_canary_off_live(self):
        with self.assertRaisesRegex(ValueError, "live"):
            g.refuse_live_index("x/" + g.LIVE_CONTEXTS)
        contract = g.v5_canary_contract()
        self.assertEqual(contract["selected_rule"], "frozen_80_20_hash")
        self.assertFalse(contract["apply_contexts_ran"])

    def test_decision_authentication_and_alias_invariance(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = GrowthFixture(Path(folder))
            left = fixture.assign(["new"])["rows"]
            right = fixture.assign(["new-alias", "new"])["rows"]
            self.assertEqual(left, right)
            fixture.decisions[0]["signature"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "unauthenticated"):
                fixture.assign(["new"])

    def test_duplicate_conflicting_stale_and_incomplete_decisions_deny(self):
        for mutation in ("duplicate", "component", "parent", "units", "prior", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                fixture = GrowthFixture(Path(folder))
                if mutation == "duplicate":
                    fixture.decisions *= 2
                else:
                    decision = fixture.decision
                    if mutation == "component":
                        decision["component_id"] = "missing"
                    elif mutation == "parent":
                        decision["parent_registry_sha256"] = "a" * 64
                    elif mutation == "units":
                        del decision["units"]
                    elif mutation == "prior":
                        decision["prior_use_findings"]["evidence"] = []
                    else:
                        decision["source_objects"] = []
                    fixture.sign()
                with self.assertRaises(ValueError):
                    fixture.assign(["new"])

    def test_graph_merge_quarantines_prior_partition_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = GrowthFixture(Path(folder))
            components = fixture.graph["components"]
            merged = dict(
                component_id="old-train",
                aliases=components[0]["aliases"] + components[1]["aliases"],
                source_objects=[],
            )
            fixture.graph["components"] = [merged, components[2]]
            fixture.graph_ref = fixture.put("graph.json", fixture.graph)
            fixture.decision["graph_sha256"] = fixture.graph_ref["sha256"]
            fixture.sign()
            result = fixture.assign(["old-train-alias"])
            self.assertEqual(result["quarantined_components"], ["old-train"])
            self.assertEqual(result["rows"], [])

    def test_forged_assignment_and_source_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = GrowthFixture(Path(folder))
            assignment = fixture.assign(["new"])
            path = fixture.root / "assignment.json"
            fixture.put("assignment.json", assignment)
            self.assertEqual(g.verify_assignment(path), assignment)
            assignment["rows"][0]["partition"] = "meta_test"
            fixture.put("assignment.json", assignment)
            with self.assertRaisesRegex(ValueError, "differs from original"):
                g.verify_assignment(path)
            (fixture.root / "source.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "content changed"):
                fixture.assign(["new"])

    def test_v2_hashes_component_id_and_never_opens_test(self):
        hashed, bucket, partition = g.stable_assignment_v2("canonical-cc")
        self.assertNotEqual(partition, "meta_test")
        self.assertIn(partition, ("meta_train", "development_validation"))
        again = g.stable_assignment_v2("canonical-cc")
        self.assertEqual(again, (hashed, bucket, partition))
        alias = g.stable_assignment_v2("alias-string")
        self.assertNotEqual(alias[0], hashed)


if __name__ == "__main__":
    unittest.main()
