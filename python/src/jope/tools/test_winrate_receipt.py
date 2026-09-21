import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import winrate_receipt as w
from selfplay_milestones import BARS


class WinRateReceiptTests(unittest.TestCase):
    def test_diagnostic_pairs_keep_honest_final_rate(self):
        diagnostic = dict(
            later_vs_teacher=dict(neural=dict(pairs=140, final_better=2, final_equal=138, final_worse=0))
        )
        with tempfile.TemporaryDirectory() as folder:
            paired = Path(folder) / "pairs.csv"
            paired.write_text(
                "partition,family,cut,rng_seed,opponent,neural_rmse,opponent_rmse,outcome\n"
                "meta_train,a,0,1,generic,1,1,tie\n"
                "meta_train,a,0,2,generic,1,2,win\n"
            )
            dest = Path(folder) / "receipt.json"
            row = w.receipt(diagnostic=diagnostic, dest=dest, diagnostic_path=None, paired_path=paired)
            self.assertEqual(row["schema"], "parallel.winrate_receipt.v2")
            self.assertAlmostEqual(row["final_gp_rate"], 2 / 140)
            self.assertEqual(row["neural_seed_rate"], "unavailable")
            self.assertEqual(row["final_gp_margin_rate"], "unavailable")
            self.assertEqual(row["vs_random"], "unavailable")
            self.assertEqual(row["vs_generic"], dict(win=1, loss=0, tie=1))
            self.assertFalse(row["world_class"])
            self.assertFalse(row["existence_proof"])
            self.assertFalse(row["serving_written"])
            self.assertFalse(row["independent_evaluation"])
            self.assertLess(row["final_gp_rate"], BARS["final_gp_beats_teacher"])

    def test_quality_rows_use_win_rates_and_stay_below_bars(self):
        rows = [
            {
                "arm": "neural",
                "best_retained_teacher": 1.0,
                "best_initial_neural_seed": 1.14,
                "best_final_gp": 1.0,
            }
        ] * 20
        row = w.receipt(quality={"rows": rows})
        self.assertEqual(row["neural_seed_rate"], 0.0)
        self.assertEqual(row["final_gp_rate"], 0.0)
        self.assertEqual(row["final_gp_margin_rate"], 0.0)
        self.assertEqual(row["vs_random"], "unavailable")
        self.assertFalse(row["world_class"])

    def test_one_winning_sample_is_not_world_class(self):
        rows = [
            {
                "arm": "neural",
                "best_retained_teacher": 1.0,
                "best_initial_neural_seed": 0.9,
                "best_final_gp": 0.8,
            }
        ]
        row = w.receipt(quality={"rows": rows})
        self.assertGreater(row["neural_seed_rate"], 0)
        self.assertFalse(row["world_class"])
        self.assertFalse(row["serving_written"])

    def test_repeated_wins_cannot_supply_twenty_independent_pairs(self):
        sample = dict(
            arm="neural", best_retained_teacher=1.0, best_initial_neural_seed=0.9, best_final_gp=0.8
        )
        result = w.receipt(quality={"rows": [sample] * 20})
        self.assertEqual(result["win_rates"]["neural_seed_n"], 1)
        self.assertFalse(result["world_class"])
        self.assertFalse(result["existence_proof"])
        self.assertFalse(result["shuffled_retired"])

    def test_invalid_counts_and_rewritten_history_reject(self):
        bad = dict(later_vs_teacher=dict(neural=dict(pairs=1, final_better=20, final_equal=0, final_worse=0)))
        self.assertEqual(w.receipt(diagnostic=bad)["final_gp_rate"], "unavailable")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.json"
            path.write_text('{"schema":"retained-v1"}')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "immutable"):
                w.receipt(dest=path)
            self.assertEqual(path.read_bytes(), before)

    def test_duplicate_or_conflicting_csv_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"
            header = "partition,family,cut,rng_seed,opponent,neural_rmse,opponent_rmse,outcome\n"
            sample = "meta_train,a,0,1,generic,1,2,win\n"
            path.write_text(header + sample * 20)
            self.assertEqual(w.paired_summary(path)["rows"], 1)
            path.write_text(header + sample + "meta_train,a,0,1,generic,2,1,loss\n")
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                w.paired_summary(path)

    def test_public_receipt_rejects_report_source_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quality.json"
            path.write_text('{"rows":[]}')
            with self.assertRaisesRegex(ValueError, "source bytes"):
                w.receipt(quality={"rows": [{"arm": "neural"}]}, quality_path=path)

    def test_public_receipt_binds_the_json_and_csv_bytes_it_processed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            quality, paired = root / "quality.json", root / "pairs.csv"
            quality.write_text(
                json.dumps(
                    dict(
                        rows=[
                            dict(
                                arm="neural",
                                best_retained_teacher=1.0,
                                best_initial_neural_seed=0.9,
                                best_final_gp=0.8,
                            )
                        ]
                    )
                )
            )
            paired.write_text(
                "partition,family,cut,rng_seed,opponent,neural_rmse,opponent_rmse,outcome\n"
                "meta_train,a,0,1,generic,1,2,win\n"
            )
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (quality, paired)}
            actual_json, actual_bytes = w.read_with_digest, w.metadata_bytes

            def change_json(path, *args):
                result = actual_json(path, *args)
                Path(path).write_text('{"rows":[]}')
                return result

            def change_csv(path, *args):
                result = actual_bytes(path, *args)
                Path(path).write_text("changed after read")
                return result

            with (
                patch.object(w, "read_with_digest", change_json),
                patch.object(w, "metadata_bytes", change_csv),
            ):
                result = w.receipt(quality_path=quality, paired_path=paired)
            self.assertEqual(result["sources"]["quality_sha256"], before[quality])
            self.assertEqual(result["sources"]["paired_sha256"], before[paired])
            self.assertEqual(result["quality_neural_rows"], 1)
            self.assertEqual(result["vs_generic"], dict(win=1, loss=0, tie=0))

    def test_duplicate_csv_columns_reject_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.csv"
            path.write_text(
                "partition,family,cut,rng_seed,opponent,neural_rmse,neural_rmse,opponent_rmse,outcome\n"
                "meta_train,a,0,1,generic,10,1,2,win\n"
            )
            with self.assertRaisesRegex(ValueError, "unique CSV"):
                w.receipt(paired_path=path)


if __name__ == "__main__":
    unittest.main()
