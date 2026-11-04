"""Fit-chain quarantine contract tests for the judge lane."""

import unittest

from .test_judge_lane import (
    ARM, TracingTable, _disclosures, _family, build_document, fit_chain, fit_judge_split,
)

class SplitAndQuarantineTest(unittest.TestCase):
    def test_t06_judge_rows_stay_out_of_the_fit_chain(self):
        row_ids = [f"r{index:02d}" for index in range(10)]
        fit_ids, judge_ids = fit_judge_split(row_ids, 7)
        self.assertEqual(len(fit_ids), 8)
        self.assertEqual(len(judge_ids), 2)
        self.assertEqual(set(fit_ids).isdisjoint(judge_ids), True)
        self.assertEqual(fit_judge_split(row_ids, 7), (fit_ids, judge_ids))
        table = {
            row_id: {"features": [float(index), 0.0], "target": float(index)}
            for index, row_id in enumerate(row_ids)
        }
        traced = TracingTable(table, judge_ids)
        built = fit_chain(traced, fit_ids, judge_row_ids=judge_ids, proposals=["p"])
        self.assertEqual(built["fit_row_ids"], fit_ids)
        self.assertEqual(built["judge_row_ids_ignored"], judge_ids)
        self.assertTrue(set(built["context_row_ids"]).isdisjoint(judge_ids))
        self.assertTrue(set(built["query_row_ids"]).isdisjoint(judge_ids))

    def test_t07_chain_records_fit_row_ids_and_ignores_judge_targets(self):
        row_ids = [f"r{index:02d}" for index in range(10)]
        fit_ids, judge_ids = fit_judge_split(row_ids, 7)
        table = {
            row_id: {"features": [float(index), 0.0], "target": float(index)}
            for index, row_id in enumerate(row_ids)
        }
        base = fit_chain(table, fit_ids, judge_row_ids=judge_ids, proposals=["p"])
        self.assertEqual(base["weights"]["fit_row_ids"], fit_ids)
        poisoned = {key: dict(value) for key, value in table.items()}
        poisoned[judge_ids[0]] = {
            "features": [999.0, 999.0],
            "target": 999.0,
        }
        self.assertEqual(
            fit_chain(poisoned, fit_ids, judge_row_ids=judge_ids, proposals=["p"])["vector"],
            base["vector"],
        )
        shifted = {key: {"features": list(value["features"]), "target": value["target"]} for key, value in table.items()}
        shifted[fit_ids[0]]["target"] = -50.0
        self.assertNotEqual(
            fit_chain(shifted, fit_ids, judge_row_ids=judge_ids, proposals=["p"])["cache_key"],
            base["cache_key"],
        )

    def test_refused_roles_and_split_does_not_take_features(self):
        for role in ("development", "sealed", "rights_pending", "train"):
            raw = _family("fam", source_ids=["family-src"], disclosed=["kumo-src"])
            raw["role"] = role
            with self.assertRaises(ValueError):
                build_document([raw], {"fam": _disclosures("fam")}, arm=ARM, seed=1, kind="fixture")
        self.assertEqual(len(fit_judge_split(["a", "b"], 1)[0]), 1)


