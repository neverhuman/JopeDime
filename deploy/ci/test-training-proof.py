#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location("training_proof", Path(__file__).with_name("training-proof.py"))
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class TrainingProofTests(unittest.TestCase):
    def setUp(self):
        # A retained CI receipt directory or custody directory, never /tmp.
        self.temporary = tempfile.TemporaryDirectory(prefix="training-proof-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.junit = self.root / "junit.xml"
        self.collection = self.root / "collection.json"

    def fixture(self, mode):
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite")
        collection = {"rust-suites": {}}
        for binary, name in proof.REQUIRED[mode]:
            ET.SubElement(suite, "testcase", classname=binary, name=name, time="0.01")
            binary_suite = collection["rust-suites"].setdefault(binary, {"status": "listed", "testcases": {}})
            binary_suite["testcases"][name] = {"ignored": False, "filter-match": {"status": "matches"}}
        return root, collection

    def save(self, root, collection):
        ET.ElementTree(root).write(self.junit)
        self.collection.write_text(json.dumps(collection))

    def test_all_three_tasks_and_native_features_are_required(self):
        for mode, count in (("python", 40), ("rust", 29), ("policy", 10)):
            with self.subTest(mode=mode):
                self.save(*self.fixture(mode))
                result = proof.verify(mode, self.junit, self.collection)
                self.assertEqual(result["required_executed"], count)

    def test_skipped_failed_and_error_are_not_execution_proof(self):
        for tag in ("skipped", "failure", "error"):
            with self.subTest(tag=tag):
                root, collection = self.fixture("python")
                ET.SubElement(next(root.iter("testcase")), tag)
                self.save(root, collection)
                with self.assertRaisesRegex(ValueError, "did not pass"):
                    proof.verify("python", self.junit)

    def test_missing_duplicate_and_wrong_suite_refuse(self):
        for mutation in ("missing", "duplicate", "wrong_suite"):
            with self.subTest(mutation=mutation):
                root, collection = self.fixture("python")
                suite = root.find("testsuite")
                case = suite.find("testcase")
                if mutation == "missing":
                    suite.remove(case)
                elif mutation == "duplicate":
                    ET.SubElement(suite, "testcase", **case.attrib)
                else:
                    case.set("classname", "unrelated")
                self.save(root, collection)
                with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                    proof.verify("python", self.junit)

    def test_missing_filtered_or_ignored_feature_test_refuses_even_with_green_junit(self):
        for mutation in ("missing", "filtered", "ignored", "not_listed"):
            with self.subTest(mutation=mutation):
                root, collection = self.fixture("rust")
                binary, name = proof.REQUIRED["rust"][0]
                suite = collection["rust-suites"][binary]
                if mutation == "missing":
                    del suite["testcases"][name]
                elif mutation == "filtered":
                    suite["testcases"][name]["filter-match"]["status"] = "mismatch"
                elif mutation == "ignored":
                    suite["testcases"][name]["ignored"] = True
                else:
                    suite["status"] = "skipped"
                self.save(root, collection)
                with self.assertRaisesRegex(ValueError, "not selected"):
                    proof.verify("rust", self.junit, self.collection)

    def test_actual_trained_regression_must_pass_even_with_other_green_cases(self):
        # Keep this case identity independent of REQUIRED: an omitted requirement
        # must not make its negative acceptance fixture disappear too.
        binary = "tests.test_experimental_proposal"
        name = "test_actual_trained_classifier_reload_sampling_uses_task_head[regression]"
        for mutation in ("missing", "skipped", "failure", "error", "duplicate", "wrong_suite"):
            with self.subTest(mutation=mutation):
                root, collection = self.fixture("python")
                suite = root.find("testsuite")
                matches = [case for case in suite.findall("testcase")
                           if case.get("classname") == binary and case.get("name") == name]
                # Also exercise a gate that has accidentally dropped regression.
                case = matches[0] if matches else ET.SubElement(
                    suite, "testcase", classname=binary, name=name, time="0.01")
                self.save(root, collection)
                proof.verify("python", self.junit)
                if mutation == "missing":
                    suite.remove(case)
                    # An unrelated success keeps aggregate counts green.
                    ET.SubElement(suite, "testcase", classname=binary,
                                  name="unrelated_green_case", time="0.01")
                elif mutation == "duplicate":
                    ET.SubElement(suite, "testcase", **case.attrib)
                elif mutation == "wrong_suite":
                    case.set("classname", "tests.unrelated")
                else:
                    ET.SubElement(case, mutation)
                self.save(root, collection)
                with self.assertRaisesRegex(ValueError, r"sampling_uses_task_head\[regression\]"):
                    proof.verify("python", self.junit)

    def test_rust_requires_collection_and_python_does_not(self):
        self.save(*self.fixture("rust"))
        with self.assertRaisesRegex(ValueError, "actual nextest collection"):
            proof.verify("rust", self.junit)
        self.save(*self.fixture("python"))
        self.assertEqual(proof.verify("python", self.junit)["required_executed"], 40)

    def slices(self, owners):
        """owners[i]: the slice (1 or 2) whose partition selects REQUIRED['rust'][i]."""
        pairs = []
        for shard in (1, 2):
            root, collection = self.fixture("rust")
            suite = root.find("testsuite")
            for (binary, name), owner in zip(proof.REQUIRED["rust"], owners):
                if owner != shard:
                    collection["rust-suites"][binary]["testcases"][name]["filter-match"] = {"status": "mismatch", "reason": "partition"}
                    suite.remove(next(case for case in suite.iter("testcase") if case.get("classname") == binary and case.get("name") == name))
            junit, listing = self.root / f"junit-{shard}.xml", self.root / f"collection-{shard}.json"
            ET.ElementTree(root).write(junit)
            listing.write_text(json.dumps(collection))
            pairs.append((listing, junit))
        return pairs

    def test_partitioned_slices_prove_every_case_exactly_once(self):
        owners = [1 + i % 2 for i in range(len(proof.REQUIRED["rust"]))]
        pairs = self.slices(owners)
        first = proof.verify("rust", pairs[0][1], pairs[0][0], partitioned=True)
        self.assertEqual((first["required_executed"], first["outside_partition"]), (15, 14))
        self.assertEqual(proof.union("rust", pairs)["required_executed"], 29)
        # Unpartitioned, a slice still fails: a case outside it is not selected.
        with self.assertRaisesRegex(ValueError, "not selected"):
            proof.verify("rust", pairs[0][1], pairs[0][0])
        # A case its own slice also leaves to the partition runs nowhere: not proven.
        listing = json.loads(pairs[0][0].read_text())
        binary, name = proof.REQUIRED["rust"][0]
        listing["rust-suites"][binary]["testcases"][name]["filter-match"] = {"status": "mismatch", "reason": "partition"}
        pairs[0][0].write_text(json.dumps(listing))
        root = ET.parse(pairs[0][1]).getroot()
        suite = root.find("testsuite")
        suite.remove(next(case for case in suite.iter("testcase") if case.get("classname") == binary and case.get("name") == name))
        ET.ElementTree(root).write(pairs[0][1])
        with self.assertRaisesRegex(ValueError, "not executed exactly once"):
            proof.union("rust", pairs)
        # Only the partition may exclude a case; any other filter still fails the slice.
        for match in ({"status": "mismatch", "reason": "default-filter"}, {"status": "mismatch", "reason": "ignored"}):
            listing["rust-suites"][binary]["testcases"][name]["filter-match"] = match
            pairs[0][0].write_text(json.dumps(listing))
            with self.assertRaisesRegex(ValueError, "not selected"):
                proof.verify("rust", pairs[0][1], pairs[0][0], partitioned=True)

    def test_case_outside_the_slice_must_not_be_reported_there(self):
        owners = [1] * len(proof.REQUIRED["rust"])
        pairs = self.slices(owners)
        root = ET.parse(pairs[1][1]).getroot()
        binary, name = proof.REQUIRED["rust"][0]
        ET.SubElement(root.find("testsuite"), "testcase", classname=binary, name=name, time="0.01")
        ET.ElementTree(root).write(pairs[1][1])
        with self.assertRaisesRegex(ValueError, "outside this partition was reported"):
            proof.verify("rust", pairs[1][1], pairs[1][0], partitioned=True)


if __name__ == "__main__":
    unittest.main()
