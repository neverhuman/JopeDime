import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import screen_pilot_select as s
import scientific_metadata as metadata


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inventory(head="ridge", count=64):
    rows = []
    for family in range(8):
        for cut in (0, 1):
            bound = dict(family=f"family-{family}", dataset=f"dataset-{family}",
                         cut=cut, partition="meta_train")
            for key in metadata.CONTEXT_FIELDS:
                if key.endswith("_sha256"):
                    bound[key] = sha(f"{family}:{cut}:{key}")
            candidates = []
            for number in range(count):
                genome = f"(pipe X ({head} n={number}))"
                candidates.append(dict(genome=genome, program_sha256=sha(genome)))
            rows.append(dict(bound, candidates=candidates))
    return dict(schema="parallel.screen_metadata.v1", contexts=rows)


def freeze(root, value):
    (root / "metadata.json").write_text(json.dumps(value))
    result = s.select(root, "pilot", metadata="metadata.json")
    return metadata.read(root / "pilot/selection.json"), result["selection_sha256"]


def scores(selection, selection_sha, axis="same_row_f32"):
    rows = []
    for number, p in enumerate(selection["programs"]):
        proxy = p["genome"].replace("(histgbm2", "(histgbm", 1) if axis == "histgbm_proxy" else p["genome"]
        rows.append(dict(context=p["context"], program_sha256=p["program_sha256"],
                         proxy_genome=proxy, proxy_program_sha256=sha(proxy),
                         proxy_contract_sha256=sha(axis), score=float(number % 64), status="ok"))
    return dict(schema="parallel.screen_scores.v1", selection_sha256=selection_sha,
                axis=axis, rows=rows)


class ScreenPilotSelectTests(unittest.TestCase):
    def test_public_selection_freezes_exact64_before_any_reference_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            value = inventory()
            for row in value["contexts"]:
                for p in row["candidates"]:
                    p["original_reference"] = dict(path="missing-original.json", sha256=sha("missing"))
            original = s.reference
            def checked_reference(*args):
                self.assertTrue((root / "pilot/selection.json").exists())
                frozen = metadata.read(root / "pilot/selection.json")
                self.assertEqual(len(frozen["contexts"]), 16)
                self.assertEqual(len(frozen["programs"]), 1024)
                return original(*args)
            with patch.object(s, "reference", checked_reference):
                selection, _ = freeze(root, value)
            status = metadata.read(root / "pilot/reference-status.json")
            self.assertFalse(status["qualification_ready"])
            self.assertIsNone(status["retrospective_accuracy"])
            self.assertTrue(all(not r["verified"] for r in status["references"]))
            self.assertEqual(len(selection["programs"]), 16 * 64)

    def test_public_cli_rejects_outcome_inventory_without_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            value = inventory()
            value["contexts"][0]["candidates"][0]["rmse"] = 0.
            (root / "metadata.json").write_text(json.dumps(value))
            result = subprocess.run([sys.executable, "-B", s.__file__, "--root", str(root),
                "--output", "pilot", "--metadata", "metadata.json"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outcomes", result.stderr)
            self.assertFalse((root / "pilot").exists())

    def test_original_reference_refuses_payload_path_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "data/targets.json").write_text("[]")
            value = inventory()
            value["contexts"][0]["candidates"][0]["original_reference"] = dict(
                path="data/targets.json", sha256=sha("[]"))
            actual_open = Path.open
            def refuse_open(path, *args, **kwargs):
                if "data" in path.parts:
                    raise AssertionError("payload must be rejected before open")
                return actual_open(path, *args, **kwargs)
            with patch.object(Path, "open", refuse_open):
                selection, _ = freeze(root, value)
            result = metadata.read(root / "pilot/reference-status.json")
            reasons = [row["unavailable_reason"] for row in result["references"]]
            self.assertTrue(any("data/ payloads" in reason for reason in reasons))
            self.assertEqual(selection["metadata_sha256"], hashlib.sha256((root / "metadata.json").read_bytes()).hexdigest())

    def test_best16_and_audit5_are_from_same64(self):
        with tempfile.TemporaryDirectory() as folder:
            selection, binding = freeze(Path(folder), inventory(count=70))
            score_rows = scores(selection, binding)
            chosen, audits = s.rejection_audits(selection, binding, score_rows)
            self.assertEqual(len(chosen), 16 * 16)
            self.assertEqual(len(audits), 16 * 5)
            for bound in selection["contexts"]:
                pool = {p["program_sha256"] for p in selection["programs"] if p["context"] == bound}
                top = {p["program_sha256"] for p in chosen if p["context"] == bound}
                audit = {p["program_sha256"] for p in audits if p["context"] == bound}
                self.assertEqual(len(top), 16)
                self.assertEqual(len(audit), 5)
                self.assertTrue(top <= pool and audit <= pool - top)
            score_rows["rows"].reverse()
            self.assertEqual(s.rejection_audits(selection, binding, score_rows), (chosen, audits))

    def test_same_population_failures_stay_and_extra_missing_duplicate_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            selection, binding = freeze(Path(folder), inventory())
            valid = scores(selection, binding)
            valid["rows"][63].update(status="failed", score=None)
            chosen, _ = s.rejection_audits(selection, binding, valid)
            self.assertEqual(len(chosen), 256)
            for mutate in (lambda x: x["rows"].pop(),
                           lambda x: x["rows"].append(copy.deepcopy(x["rows"][0])),
                           lambda x: x["rows"][0].update(program_sha256=sha("outside64")),
                           lambda x: x["rows"][0].update(score=True),
                           lambda x: x["rows"][0]["context"].update(runtime_sha256=sha("other-runtime"))):
                changed = copy.deepcopy(valid)
                mutate(changed)
                with self.assertRaises(ValueError):
                    s.rejection_audits(selection, binding, changed)

    def test_histogram_pairs_keep_original_and_proxy_sha_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            selection, binding = freeze(Path(folder), inventory(head="histgbm2"))
            score_rows = scores(selection, binding, "histgbm_proxy")
            chosen, audits = s.rejection_audits(selection, binding, score_rows)
            self.assertTrue(all(p["program_sha256"] != p["proxy_program_sha256"] for p in chosen + audits))
            self.assertEqual(len({p["pair_identity"] for p in chosen + audits}), 16 * 21)
            changed = copy.deepcopy(score_rows)
            changed["rows"][0]["proxy_genome"] = "(pipe X (scale) (histgbm))"
            changed["rows"][0]["proxy_program_sha256"] = sha(changed["rows"][0]["proxy_genome"])
            with self.assertRaisesRegex(ValueError, "heads"):
                s.rejection_audits(selection, binding, changed)

    def test_public_screen_result_stays_unqualified_and_preserves_old_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selection, binding = freeze(root, inventory())
            (root / "scores.json").write_text(json.dumps(scores(selection, binding)))
            result = s.select(root, "pilot", metadata="metadata.json", screen="scores.json")
            self.assertFalse(result["qualification_ready"])
            self.assertIsNone(result["retrospective_accuracy"])
            self.assertIsNone(result["prospective_total_cost"])
            old = root / "legacy"
            old.mkdir()
            (old / "selection.json").write_text('{"schema":"parallel.screen_pilot_selection.v1"}')
            before = (old / "selection.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "immutable"):
                s.select(root, old, metadata="metadata.json")
            self.assertEqual((old / "selection.json").read_bytes(), before)

    def test_duplicate_conflicting_and_nontraining_metadata_reject(self):
        for mutate in (lambda x: x["contexts"].append(copy.deepcopy(x["contexts"][0])),
                       lambda x: x["contexts"][0].update(partition="meta_test"),
                       lambda x: x["contexts"][0]["candidates"][0].update(program_sha256=sha("wrong"))):
            value = inventory()
            mutate(value)
            with self.assertRaises(ValueError):
                s.freeze_contexts(value)

    def test_spearman_ties_and_constants(self):
        self.assertAlmostEqual(s.spearman([1, 1, 2], [1, 2, 3]), math.sqrt(3) / 2)
        self.assertIsNone(s.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(s.spearman([True, 2], [1, 2]))


import math

if __name__ == "__main__":
    unittest.main()
