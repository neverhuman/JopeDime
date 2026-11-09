import json, unittest
from pathlib import Path
from native_h352_contract import build, validate


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        r = json.loads(
            (Path(__file__).parent / "native_baseline_reference.json").read_text()
        )
        self.params = dict(r["constructor_params"], num_leaves=15, min_child_samples=20)
        self.native = dict(r["native_params"], num_leaves="15", min_data_in_leaf="20")
        self.contract = build(
            [
                dict(
                    kind="LGBMRanker",
                    constructor_params=self.params,
                    native_params=self.native,
                )
            ]
        )

    def test_only_MCS_native_delta(self):
        self.assertTrue(
            validate(
                dict(self.params, min_child_samples=40),
                dict(self.native, min_data_in_leaf="40"),
                "LGBMRanker",
                self.contract,
            )
        )

    def test_even_unexposed_native_default_drift_rejected(self):
        for k, v in [
            ("max_bin", "127"),
            ("bagging_seed", "123"),
            ("feature_fraction_seed", "123"),
        ]:
            with self.assertRaises(AssertionError):
                validate(
                    dict(self.params, min_child_samples=40),
                    dict(self.native, min_data_in_leaf="40", **{k: v}),
                    "LGBMRanker",
                    self.contract,
                )

    def test_outside_unchanged_grid_rejected(self):
        with self.assertRaises(AssertionError):
            validate(
                dict(self.params, min_child_samples=40, learning_rate=0.12345),
                dict(self.native, min_data_in_leaf="40", learning_rate=".12345"),
                "LGBMRanker",
                self.contract,
            )


if __name__ == "__main__":
    unittest.main()
