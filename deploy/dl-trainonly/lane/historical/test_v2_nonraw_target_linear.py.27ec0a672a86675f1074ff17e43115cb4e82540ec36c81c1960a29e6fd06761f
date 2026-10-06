"""LINEAR-DETERMINED on the clean non-raw routes (tools/v2_build_local.py L7b, tools/v2_build_l7a.py L7a): both go through
v2_build_local.compute_dataset, which runs dltrain.target_linear.check_jtt1 once per context on the ENCODER payload.

  Refuse   an exactly linear target refuses the dataset on L7b and on L7a (target_linear_in_features; run_dataset_v2
           writes nothing); the per-context results are in the refusal detail
  Review   a near-linear target is built and flagged (target_linear_near_review) in every context.json; ordinary data
           passes with verdict None
  Payload  the recorded result is check_jtt1 of that context's own encoder file (not the teacher), and outer-test rows
           never change it
  Verify   verify_tree flags a context whose recorded verdict is REFUSE or whose result is missing
"""
import gzip
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import target_linear as tl
from tests import test_v2_nonraw_split_v2 as t

tool, l7a = t.tool, t.l7a


def linear(noise=0.0):
    num = np.nan_to_num(t.NUM)                                            # no missing cells: imputation stays exact
    y = 2.0 * num[:, 0] + 0.01 * num[:, 1] - 0.3 * num[:, 2] + 5.0
    if noise:
        y = y + np.random.default_rng(5).normal(0, noise, len(y))
    A, meta = t.t1_table(num, t.CAT, y)
    return A, meta, t.raw_values(num, t.CAT, y)


def linear_t1():
    """The T1 table itself with its target column an exact linear function of three stored feature columns (the T1 table
    is float32-printed, so a target linear in the RAW values is only near-linear in it)."""
    A = np.array(t.A, copy=True)
    A[:, -1] = 0.4 * A[:, 0] + 0.2 * A[:, 1] - 0.3 * A[:, 2] + 0.1
    return A


def inputs(mode, A=None, meta=None, V=None):
    if mode == "L7b":
        return t.t1_inputs(t.A if A is None else A, t.META if meta is None else meta)
    return t.l7a_inputs(t.VALUES if V is None else V)


class Refuse(unittest.TestCase):
    def test_an_exactly_linear_target_is_refused_on_both_routes(self):
        A, meta, V = linear()
        for mode, inp in (("L7b", inputs("L7b", linear_t1())), ("L7a", inputs("L7a", V=V))):
            with self.subTest(mode=mode):
                with self.assertRaises(tool.SplitV2Error) as cm:
                    tool.compute_dataset(t.plan_entry(), inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2)
                self.assertEqual(cm.exception.reason, tl.REFUSE)
                self.assertTrue(all(float(v) < 1e-8 for v in json.loads(cm.exception.detail).values()))

    def test_run_dataset_v2_writes_nothing(self):
        A, meta, V = linear()
        out = t.scratch()
        try:
            e = dict(t.plan_entry(), context_extra=tool.group_annotation_fields(None))
            rec = tool.run_dataset_v2(e, out, "L7b", lambda e: inputs("L7b", linear_t1()), set(), {"code": "x"}, log=lambda s: None,
                                      version=tool.SPLIT_V2)
            self.assertEqual((rec["status"], rec["reason"]), ("failed", tl.REFUSE))
            self.assertEqual([d for d in out.iterdir() if d.name != "_datasets"], [])
        finally:
            shutil.rmtree(out, ignore_errors=True)


class Review(unittest.TestCase):
    def test_near_linear_is_flagged_and_ordinary_data_passes(self):
        A, meta, V = linear()                                             # float32-printed T1: near-linear (~1e-7) on L7b
        _, _, Vn = linear(noise=1e-4)
        for mode, inp in (("L7b", inputs("L7b", A, meta)), ("L7a", inputs("L7a", V=Vn))):
            with self.subTest(mode=mode):
                res = tool.compute_dataset(t.plan_entry(), inp, mode, set(), {"code": "x"}, version=tool.SPLIT_V2)
                self.assertEqual({c["target_linear"]["verdict"] for c in res["contexts"]}, {tl.REVIEW})
                self.assertEqual({crec["target_linear"]["verdict"] for _, _, crec in res["results"]}, {tl.REVIEW})
        for mode in ("L7b", "L7a"):
            res = t.computed(mode)
            self.assertEqual({crec["target_linear"]["verdict"] for _, _, crec in res["results"]}, {None})


class Payload(unittest.TestCase):
    def test_the_result_is_the_encoders_and_ignores_test_rows(self):
        for mode in ("L7b", "L7a"):
            res = t.computed(mode)
            for vid, files, crec in res["results"]:
                enc = gzip.decompress(files["encoder.jtt1.gz"])
                self.assertEqual(crec["target_linear"], tl.check_jtt1(enc), (mode, vid))
        A, meta, V = linear()
        res = tool.compute_dataset(t.plan_entry(), inputs("L7b", A, meta), "L7b", set(), {"code": "x"}, version=tool.SPLIT_V2)
        diff = 0
        for vid, files, crec in res["results"]:
            enc, tea = (gzip.decompress(files[n]) for n in ("encoder.jtt1.gz", "teacher.jtt1.gz"))
            self.assertEqual(crec["target_linear"], tl.check_jtt1(enc))
            diff += tl.check_jtt1(tea) != tl.check_jtt1(enc)
        self.assertGreater(diff, 0)                                       # the teacher payload would give another record


class Verify(unittest.TestCase):
    def test_verify_flags_refused_or_missing_results(self):
        out = t.scratch()
        try:
            e = dict(t.plan_entry(), context_extra=tool.group_annotation_fields(None), shard=0)
            sd = out / "shard-0"; sd.mkdir()
            rec = tool.run_dataset_v2(e, sd, "L7b", lambda e: inputs("L7b"), set(), {"code": "x"}, log=lambda s: None,
                                      version=tool.SPLIT_V2)
            self.assertEqual(rec["status"], "done", rec.get("detail"))
            pl = {"datasets": [e], "split_version": tool.SPLIT_V2, "values_mode": "L7b", "partition": "t1", "scope": "fixture"}
            (out / "existing-view-ids.json").write_text(json.dumps({"view_ids": []}))
            with mock.patch.object(tool, "scan_existing_view_ids", lambda exclude=(): ([], {})):
                ok = tool.verify_tree(out, pl, {"code": "x"})
                self.assertFalse([p for p in ok["problems"] if "target_linear" in p], ok["problems"])
                d = sd / rec["contexts"][0]["view_id"]
                cj = json.loads((d / "context.json").read_bytes())
                for bad in (dict(cj, target_linear=dict(cj["target_linear"], verdict=tl.REFUSE)),
                            {k: v for k, v in cj.items() if k != "target_linear"}):
                    (d / "context.json").write_text(json.dumps(bad))
                    got = tool.verify_tree(out, pl, {"code": "x"})
                    self.assertTrue([p for p in got["problems"] if "target_linear" in p], got["problems"])
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
