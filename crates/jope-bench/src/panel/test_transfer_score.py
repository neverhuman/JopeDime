import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('transfer_score', Path(__file__).with_name('transfer_score.py'))
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class Fixture:
    """Synthetic plan/freeze whose gains are exact by construction (no reporter involved)."""

    def __init__(self, root, datasets, outcome):
        self.root = Path(root)
        self.specification = {"arms": [{"id": "parent"}, {"id": "challenger"}], "seeds": [7], "datasets": []}
        self.searches, self.predictions = [], []
        for dataset, family in datasets:
            labels = {f"{dataset}-r{i}": float(i) for i in range(20)}
            self.specification["datasets"].append(
                {"id": dataset, "family": family, "labels": self.asset(f"{dataset}-labels.json", labels)})
            rows = sorted(labels)
            for outer in range(5):
                for audit in [0, 1, 2, 3, 4, None]:
                    score_rows = rows[outer * 4:(outer + 1) * 4]
                    train = [r for r in rows if r not in score_rows]
                    search = {"dataset": dataset, "seed": 7, "outer": outer, "audit": audit,
                              "train": train, "score": score_rows, "inner_score": [train]}
                    self.searches.append(search)
                    mean = sum(labels[r] for r in train) / len(train)
                    for arm in ("parent", "challenger"):
                        values = outcome(dataset, audit, arm, [labels[r] for r in score_rows], mean)
                        self.predictions.append(self.asset(
                            f"p-{len(self.predictions)}.json",
                            {"search_sha256": score.search_digest(search), "arm": arm, "predictions": values}))

    def asset(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def plan(self):
        return {"specification": self.specification, "searches": self.searches, "failures": []}

    def freeze(self):
        return {"predictions": self.predictions, "failures": []}


def perfect_challenger(dataset, audit, arm, actual, mean):
    # Parent predicts the training mean (gain denominator), challenger is exact: every gain is 1.
    return list(actual) if arm == "challenger" else [mean] * len(actual)


class TransferScore(unittest.TestCase):
    def test_exact_gain_is_one_hundred_points(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1")], perfect_challenger)
            result = score.score(f.plan(), f.freeze())
            self.assertTrue(result["complete"])
            self.assertAlmostEqual(result["kpi_percentage_points"], 100.0, places=12)

    def test_min_is_taken_per_unit_and_families_weigh_equally(self):
        def outer_regression(dataset, audit, arm, actual, mean):
            # d2's challenger equals the parent on outer searches: its unit minimum is zero.
            if dataset == "d2" and audit is None:
                return [mean] * len(actual)
            return perfect_challenger(dataset, audit, arm, actual, mean)
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1"), ("d2", "f2"), ("d3", "f2")], outer_regression)
            result = score.score(f.plan(), f.freeze())
            # f1 = 1.0; f2 = mean(d2 = 0.0, d3 = 1.0) = 0.5; families equally: 0.75.
            self.assertAlmostEqual(result["kpi_percentage_points"], 75.0, places=12)
            self.assertAlmostEqual(result["families"]["f2"]["contribution"], 0.5, places=12)

    def test_missing_pair_and_zero_baseline_never_become_zero_gains(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1")], perfect_challenger)
            freeze = f.freeze()
            freeze["predictions"].pop()
            result = score.score(f.plan(), freeze)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["kpi_percentage_points"])
            self.assertTrue(any(r.startswith("missing") for r in result["reasons"]))
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1")], perfect_challenger)
            constant = {r: 3.0 for r in json.loads(Path(f.specification["datasets"][0]["labels"]["path"]).read_text())}
            f.specification["datasets"][0]["labels"] = f.asset("constant.json", constant)
            result = score.score(f.plan(), f.freeze())
            self.assertFalse(result["complete"])
            self.assertIsNone(result["kpi_percentage_points"])
            self.assertEqual(len(result["units"]), 5)

    def test_tampered_or_unplanned_predictions_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1")], perfect_challenger)
            freeze = f.freeze()
            Path(freeze["predictions"][0]["path"]).write_text('{"tampered": true}')
            with self.assertRaisesRegex(ValueError, "altered asset"):
                score.score(f.plan(), freeze)
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1")], perfect_challenger)
            plan = f.plan()
            plan["searches"] = plan["searches"][1:]
            with self.assertRaisesRegex(ValueError, "unplanned prediction"):
                score.score(plan, f.freeze())

    def test_comparison_detects_a_different_headline(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, [("d1", "f1")], perfect_challenger)
            result = score.score(f.plan(), f.freeze())
            report = {"ok": True, "kpi_percentage_points": 100.0,
                      "families": {"f1": {"audit": 1.0, "outer": 1.0, "contribution": 1.0}},
                      "units": [{"dataset": u["dataset"], "seed": u["seed"], "outer": u["outer"], "gains": u["gains"]}
                                for u in result["units"]],
                      "losses": [{"search_sha256": k[0], "arm": k[1], "rmse": v[0], "training_mean_rmse": v[1]}
                                 for k, v in result["losses"].items()]}
            self.assertEqual(score.compare(result, report), [])
            report["kpi_percentage_points"] = 100.5
            self.assertTrue(any(p.startswith("headline") for p in score.compare(result, report)))
            report["ok"] = False
            self.assertTrue(any(p.startswith("completeness") for p in score.compare(result, report)))


if __name__ == "__main__":
    unittest.main()
