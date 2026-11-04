import unittest, copy, itertools, sys, subprocess
import numpy as np
import neighbors as n
import statistics_h314 as s
from fetch_metadata import allowed


def record(**kw):
    d = dict(
        dataset_id="d",
        family_id="f",
        source_ids=["p:1"],
        lineage_roots=["root:1"],
        dedupe_fingerprints=["fp1"],
        columns=["a", "b", "c", "d"],
        n_rows=100,
    )
    d.update(kw)
    return d


class Tests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(
            n.names([" A B ", "Ａ_b", "a-b", "Age", "age"]), ["a_b", "age"]
        )

    def test_generic_filename_not_lineage_identity(self):
        from classify import extract

        row = dict(
            id="d",
            canonical_sha256="canon",
            provenance=dict(platform="socrata", dataset_id="one", url="https://a/one"),
            shape=dict(rows=100, features=1, columns=2),
            task="regression",
        )
        meta = dict(
            source_platform="socrata",
            source_dataset_id="one",
            source_file="export.csv?accessType=DOWNLOAD",
            source_url="https://a/one",
            output_headers=["x", "target"],
        )
        r = extract(row, meta, dict(sha256="meta", version_id="v"), [], [])
        self.assertFalse(any("source_file:" in x for x in r["lineage_roots"]))
        self.assertEqual(
            r["lineage_field_values"]["source_file"], "export.csv?accessType=DOWNLOAD"
        )

    def test_source_identity(self):
        a = record()
        b = record(
            lineage_roots=["root:2"],
            dedupe_fingerprints=["fp2"],
            columns=["x"],
            n_rows=1000,
        )
        self.assertEqual(n.compare(a, b)["status"], "NEIGHBORED")
        self.assertTrue(n.compare(a, b)["clauses"]["a"])

    def test_lineage_identity(self):
        a = record()
        b = record(
            source_ids=["p:2"], dedupe_fingerprints=["fp2"], columns=["x"], n_rows=1000
        )
        self.assertTrue(n.compare(a, b)["lineage_root_match"])

    def test_fingerprint(self):
        a = record()
        b = record(
            source_ids=["p:2"], lineage_roots=["root:2"], columns=["x"], n_rows=1000
        )
        self.assertTrue(n.compare(a, b)["clauses"]["b"])

    def test_jaccard_boundary(self):
        a = record()
        b = record(
            source_ids=["p:2"],
            lineage_roots=["root:2"],
            dedupe_fingerprints=["fp2"],
            columns=["a", "b", "c", "d", "e"],
            n_rows=110,
        )
        r = n.compare(a, b)
        self.assertEqual(r["jaccard"], 0.8)
        self.assertTrue(r["clauses"]["c"])
        b["n_rows"] = 111
        self.assertFalse(n.compare(a, b)["clauses"]["c"])

    def test_missing_is_unknown(self):
        a = record(dedupe_fingerprints=[])
        b = record(
            source_ids=["p:2"],
            lineage_roots=["root:2"],
            dedupe_fingerprints=[],
            columns=["x"],
            n_rows=1000,
        )
        self.assertEqual(n.compare(a, b)["status"], "UNKNOWN")

    def test_empty_ids_never_match(self):
        self.assertIsNone(n.same([], []))

    def test_unknown_conjunction_false(self):
        self.assertFalse(n.both(None, False))
        self.assertIsNone(n.either(None, False))

    def test_positive_overrides_missing(self):
        self.assertEqual(
            n.compare(
                record(source_ids=[], columns=[]), record(source_ids=[], columns=[])
            )["status"],
            "NEIGHBORED",
        )

    def test_one_member_witness_suffices(self):
        dev = [record(), record(source_ids=["p:2"])]
        train = [
            record(
                source_ids=["p:2"],
                lineage_roots=["r3"],
                dedupe_fingerprints=["f3"],
                columns=["x"],
                n_rows=1000,
            )
        ]
        self.assertEqual(n.classify(dev, train)["status"], "NEIGHBORED")

    def test_no_data_blob_allowed(self):
        self.assertTrue(
            allowed(dict(key="blobs/sha256/meta", kind="meta"), {"meta"}, {"d"})
        )
        self.assertFalse(
            allowed(dict(key="blobs/sha256/train", kind="meta"), {"meta"}, {"d"})
        )
        self.assertFalse(
            allowed(
                dict(
                    key="full-source-registry/v1/sources/sha256/a.csv",
                    kind="source_registry",
                ),
                {"meta"},
                {"d"},
            )
        )

    def test_exact_flip_brute(self):
        d = np.array([[1.0, -2.0, 0.0], [2.0, 1.0, -1.0]])
        p, pp = s.exact_flip(d)
        v = np.array(
            [
                (d * np.array(sign)).mean(axis=1)
                for sign in itertools.product([-1, 1], repeat=3)
            ]
        )
        np.testing.assert_equal(p, (v >= d.mean(axis=1) - 1e-14).mean(axis=0))
        self.assertEqual(
            pp,
            float((np.median(v, axis=1) >= np.median(d.mean(axis=1)) - 1e-14).mean()),
        )

    def test_empty_group(self):
        self.assertIsNone(s.summarize(np.zeros((6, 0)))["pooled"])

    def test_ties_never_wins(self):
        r = s.summarize(np.zeros((6, 3)))
        self.assertEqual(r["per_seed"][0]["wins"], 0)
        self.assertEqual(r["per_seed"][0]["exact_signflip_p"], 1)
        self.assertEqual(r["per_seed"][0]["elo"], 1500)

    def test_drop_too_few(self):
        r = s.summarize(np.ones((6, 1)))
        self.assertTrue(r["degenerate_single_family"])
        self.assertIsNone(r["per_seed"][0]["drop_top"]["1"])

    def test_independence_floor(self):
        groups = {
            "CLEAN": s.summarize(np.ones((6, 9)) * 0.02),
            "NEIGHBORED": s.summarize(np.ones((6, 10)) * 0.02),
        }
        self.assertEqual(s.decide(groups)["verdict"], "INCONCLUSIVE")
        groups["CLEAN"] = s.summarize(np.ones((6, 10)) * 0.02)
        self.assertEqual(s.decide(groups)["verdict"], "H310_LINEAGE_INDEPENDENT")

    def test_confounded_and_tie_threshold(self):
        groups = {
            "CLEAN": s.summarize(np.ones((6, 10)) * 0.001),
            "NEIGHBORED": s.summarize(np.ones((6, 1)) * 0.02),
        }
        groups["CLEAN"]["pooled"]["gain"] = (
            0.001  # Exact decision-input boundary, not floating mean accumulation.
        )
        self.assertEqual(s.decide(groups)["verdict"], "H310_LINEAGE_CONFOUNDED")

    def test_metadata_process_blocks_outcomes(self):
        code = "from fetch_metadata import install_guard;install_guard();\ntry:open('/home/ubuntu/jope-data/h314-astra/references/all_seeds.npz')\nexcept RuntimeError:pass\nelse:raise AssertionError('outcome access allowed')"
        import os

        env = dict(
            os.environ, PYTHONPATH=str(__import__("pathlib").Path(__file__).parent)
        )
        self.assertEqual(
            subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True
            ).returncode,
            0,
        )


class SerializationTests(unittest.TestCase):
    def test_numpy_fisher_counts_serialize(self):
        import evaluate, json

        r = evaluate.fisher(
            np.array([True] * 10 + [False] * 9), ["CLEAN"] * 19, [True] * 19
        )
        self.assertEqual(
            json.loads(json.dumps(r, allow_nan=False))["table"], [[0, 10], [0, 9]]
        )
        self.assertIsNone(r["odds_ratio"])
        self.assertEqual(r["p_greater"], 1.0)


if __name__ == "__main__":
    unittest.main()
