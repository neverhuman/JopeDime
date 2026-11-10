"""Data-free controls: exact family statistics, immutable admission, scoped reads."""

import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import common
import infer_h356 as inf
import statistics_h356 as st
import resource_probe as rp
from fetch_h356 import RemoteZip, get_file

FIX = common.D / "test-fixtures"
FIX.mkdir(exist_ok=True)


class Science(unittest.TestCase):
    def test_tie_never_confirms(self):
        s = st.summarize(np.zeros((9, 4)), np.zeros(4), ["a", "a", "b", "b"], draws=100)
        self.assertEqual(s["median_elo"], 1500)
        self.assertEqual(s["paired_lift_lcb95"], 0)
        self.assertEqual(st.decide(s)[1], "NL15_DEV91_NOT_CONFIRMED")

    def test_constant_strict_positive_confirms(self):
        s = st.summarize(np.full((9, 4), 0.01), np.zeros(4), list("abcd"), draws=100)
        self.assertGreater(s["paired_lift_lcb95"], 0)
        self.assertEqual(st.decide(s)[1], "NL15_DEV91_CONFIRMED")

    def test_positive_raw_gain_inside_tie_stays_unconfirmed(self):
        s = st.summarize(np.full((9, 4), 1e-6), np.zeros(4), list("abcd"), draws=100)
        self.assertGreater(s["paired_lift_lcb95"], 0)
        self.assertEqual(s["median_elo"], 1500)
        self.assertEqual(st.decide(s)[1], "NL15_DEV91_NOT_CONFIRMED")

    def test_zero_lcb_rejected_even_with_positive_elo(self):
        self.assertEqual(
            st.decide(dict(median_elo=1600, paired_lift_lcb95=0))[1],
            "NL15_DEV91_NOT_CONFIRMED",
        )

    def test_nonpositive_elo_rejected_even_with_positive_lcb(self):
        self.assertEqual(
            st.decide(dict(median_elo=1500, paired_lift_lcb95=0.1))[1],
            "NL15_DEV91_NOT_CONFIRMED",
        )

    def test_unequal_context_counts_are_family_equal(self):
        u = np.zeros((9, 9))
        u[:, -1] = 1
        s = st.summarize(u, np.zeros(9), ["a"] * 8 + ["b"], draws=100)
        self.assertEqual(s["median_family_equal_lift"], 0.5)

    def test_shared_bootstrap_is_deterministic(self):
        u = np.random.default_rng(99).normal(size=(9, 12))
        f = [str(i // 3) for i in range(12)]
        self.assertEqual(
            st.summarize(u, np.zeros(12), f, draws=200),
            st.summarize(u, np.zeros(12), f, draws=200),
        )

    def test_root_grouping_preserves_partition_root_order(self):
        s = st.summarize(
            np.arange(9)[:, None] + np.zeros((9, 4)),
            np.zeros(4),
            list("abcd"),
            draws=50,
        )
        self.assertEqual([r["median_lift"] for r in s["per_root"]], [3, 4, 5])

    def test_elo_orientation_is_challenger_positive(self):
        self.assertGreater(st.bt_two(2, 10), 1500)
        self.assertLess(st.bt_two(10, 2), 1500)


class Admission(unittest.TestCase):
    def report(self, bad=None):
        v = {str(c): 90 for c in list(range(15)) + list(range(64, 79))}
        v.update(bad or {})
        return {
            "sysstat": {
                "hosts": [
                    {
                        "statistics": [
                            {"cpu-load": [{"cpu": c, "idle": x} for c, x in v.items()]}
                        ]
                    }
                ]
            }
        }

    def test_first_eight_from_only_pool_A(self):
        cores, idle, _ = rp.selection(self.report())
        self.assertEqual(cores, list(range(8)))

    def test_sibling_below_85_is_excluded(self):
        self.assertEqual(rp.selection(self.report({"64": 84}))[0], list(range(1, 9)))

    def test_exact_85_is_eligible(self):
        self.assertEqual(rp.selection(self.report({"0": 85, "64": 85}))[0][0], 0)

    def test_cpu_threshold_is_inclusive_70(self):
        self.assertTrue(rp.admission_pass([0, 1, 2, 3], 70))
        self.assertFalse(rp.admission_pass([0, 1, 2, 3], 70.001))

    def test_distinct_minimum_and_foreign_cores(self):
        for cores in ([0, 1, 2], [0, 0, 0, 0], [0, 1, 2, 15]):
            self.assertFalse(rp.admission_pass(cores, 10))

    def test_original_family_hash_routing(self):
        import analysis_h302

        for f in ["family-one", "family-two", "family-three"]:
            self.assertEqual(
                common.fold("P0", f), analysis_h302.outer_fold_of_family(f)
            )
            self.assertEqual(
                common.fold("P1", f),
                hashlib.sha256(b"H308-outer-v1\0" + f.encode()).digest()[-1] % 4,
            )
            self.assertEqual(
                common.fold("P2", f),
                hashlib.sha256(b"H308-outer-v2\0" + f.encode()).digest()[-1] % 4,
            )


class Scope(unittest.TestCase):
    def test_foreign_context_and_outcome_access_refused(self):
        with self.assertRaises(AssertionError):
            common.require_cid("outside")
        cid = common.rows()[0]["context_sha256"]
        for member in ["fitness", "gate", "other"]:
            with self.assertRaises(AssertionError):
                common.require_cid(cid, member, False)
        common.require_cid(cid, "program", False)
        common.require_cid(cid, "fitness", True)

    def test_exclusive_marker_prevents_replay(self):
        with tempfile.TemporaryDirectory(dir=FIX) as d:
            p = Path(d) / "marker.json"
            common.durable(p, {"state": "STARTED"}, exclusive=True)
            with self.assertRaises(FileExistsError):
                common.durable(p, {"state": "STARTED"}, exclusive=True)

    def test_only_declared_feature_range_is_read(self):
        import trainonly_loader as tl

        with tempfile.TemporaryDirectory(dir=FIX) as d:
            root = Path(d)
            (root / "input").mkdir()
            p = root / "input/context_features.npz"
            np.savez(
                p, lm48_z=np.array([[-99, -99], [42, 43], [-99, -99]], dtype=np.float32)
            )
            offset, _, _ = tl._member_layout(p, "lm48_z.npy")
            with p.open("r+b") as out:
                out.seek(offset)
                out.write(b"\x01")
            with self.assertRaises(zipfile.BadZipFile):
                with np.load(p, allow_pickle=False) as z:
                    z["lm48_z"]
            cid = common.rows()[0]["context_sha256"]
            reader = object.__new__(inf.DevCache)
            reader.tl = tl
            reader.expected = {("context_features.npz", "lm48_z", cid): (1, 2)}
            reader.ranges = []
            with patch.object(inf, "D", root):
                value = reader.read(cid, "context_features.npz", "lm48_z")
            np.testing.assert_array_equal(value, np.array([[42, 43]], dtype=np.float32))
            self.assertEqual(reader.ranges[0]["bytes"], 8)

    def test_foreign_data_and_checkpoint_open_denied(self):
        code = "import common\ncommon.install_read_guard()\nfor p in ['/home/ubuntu/jope-data/h50/input/pools/never.npz','/home/ubuntu/no-model.pt']:\n try: open(p,'rb')\n except AssertionError: pass\n else: raise RuntimeError('guard bypass')"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=common.S, check=True)

    def test_all_fit_update_entrypoints_refuse(self):
        import lightgbm as lgb

        saved = (lgb.train, lgb.LGBMModel.fit, lgb.Booster.update, lgb.Booster.refit)
        try:
            c = inf.forbid_fitting()
            for fn in [
                lgb.train,
                lgb.LGBMModel.fit,
                lgb.Booster.update,
                lgb.Booster.refit,
            ]:
                with self.assertRaises(RuntimeError):
                    fn(None)
            self.assertEqual(c["attempts"], 4)
        finally:
            lgb.train, lgb.LGBMModel.fit, lgb.Booster.update, lgb.Booster.refit = saved


class Client:
    def __init__(self, raw, version="V"):
        self.raw = raw
        self.version = version

    def get_object(self, **kw):
        r = self.raw
        if "Range" in kw:
            lo, hi = map(int, kw["Range"].split("=")[1].split("-"))
            r = r[lo : hi + 1]
        return dict(VersionId=self.version, Body=io.BytesIO(r))


class Custody(unittest.TestCase):
    def test_wrong_exact_version_refused(self):
        spec = dict(
            bucket="b",
            key="k",
            version_id="V",
            sha256=hashlib.sha256(b"ok").hexdigest(),
        )
        with self.assertRaises(AssertionError):
            get_file(Client(b"ok", "OTHER"), spec)

    def test_wrong_object_sha_refused(self):
        spec = dict(
            bucket="b",
            key="k",
            version_id="V",
            sha256=hashlib.sha256(b"ok").hexdigest(),
        )
        with self.assertRaises(AssertionError):
            get_file(Client(b"corrupt"), spec)

    def test_correct_exact_version_sha_proven(self):
        spec = dict(
            bucket="b",
            key="k",
            version_id="V",
            sha256=hashlib.sha256(b"ok").hexdigest(),
        )
        self.assertTrue(
            get_file(Client(b"ok"), spec)["exact_version_get_sha256_verified"]
        )

    def test_remote_zip_extracts_only_selected_member(self):
        f = io.BytesIO()
        with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("allowed.txt", "ok")
            z.writestr("forbidden.txt", "never decode")
        raw = f.getvalue()
        remote = RemoteZip(
            Client(raw), dict(bucket="b", key="k", version_id="V", bytes=len(raw))
        )
        with zipfile.ZipFile(remote) as z:
            self.assertEqual(z.read("allowed.txt"), b"ok")
        self.assertTrue(remote.ranges)


if __name__ == "__main__":
    unittest.main()
