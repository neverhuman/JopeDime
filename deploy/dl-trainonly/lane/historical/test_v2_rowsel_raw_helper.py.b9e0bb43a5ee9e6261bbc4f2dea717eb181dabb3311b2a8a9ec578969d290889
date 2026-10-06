"""Mechanism proof for tools/v2_rowsel_raw_helper.py on a fixture (no real data). The "owner" is the builder's own
run_dataset writing into shard-<i>, exactly what a running shard does; the helper runs with the same code and plan.

  Determinism   a helper-built dataset equals the owner-built one: JTT1/A0 bytes, view ids and identities; context.json
                differs only in utc_us; the dataset record only in timing fields
  Skip          the owner's build loop skips a helper-published dataset (dataset_done is checked right before each start)
  Selection     tail first; never the owner's frontier; margin and min-gap; claimed and done datasets excluded
  Claims        O_EXCL: a dataset is claimed once
  Races         owner done first; owner mid-write (lost race, nothing written); empty target (replaced atomically);
                helper first then the owner (owner writes nothing); crash mid-publish (no .done; the owner completes)
  Guards        code identity with the owners' records; private tmp names; nothing in the owner tree is ever deleted
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import test_v2_rowsel_raw as base

SRC = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v2_rowsel_raw_helper", SRC / "tools" / "v2_rowsel_raw_helper.py")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
owner_tool = base.tool                           # the builder module as a running shard has it
V2 = owner_tool.SPLIT_V2
SMALL = np.array(base.VALUES[:1500], copy=True)  # |U| <= 2000: passthrough selection, every gate still runs
EST = (60.0, 50.0, 40.0, 30.0, 20.0, 10.0)
HASHES = [hashlib.sha256(f"helper-fixture-{k}".encode()).hexdigest()[:16] for k in range(len(EST))]


def fake_rematch(inp):
    return np.array(SMALL, copy=True), base.SPECS, {"values_sha256": "0" * 64, "read_report_equal": True}


def entry(h, est, shard=0):
    e = base.plan_entry(h=h)
    e.update(shard=shard, est_cpu_s=est, split_version=V2)
    return e


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="rowselraw-helper-", dir=os.environ.get("TMPDIR")))
        self.out = self.dir / "TREE"
        (self.out / "shard-0").mkdir(parents=True)
        self.plan = {"of": 3, "split_version": V2, "datasets": [entry(h, est) for h, est in zip(HASHES, EST)]}
        (self.out / "plan.json").write_text(json.dumps(self.plan))
        self.patches = [mock.patch.object(t, name, fn) for t in (owner_tool, helper.tool)
                        for name, fn in (("load_inputs", base.fake_inputs), ("rematch", fake_rematch))]
        for p in self.patches:
            p.start()
        self.codes = owner_tool.code_shas()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.addCleanup(lambda: setattr(helper.tool, "atomic", owner_tool.atomic.__wrapped__)
                        if hasattr(owner_tool.atomic, "__wrapped__") else None)
        self._orig_helper_atomic = helper.tool.atomic

    def tearDown(self):
        helper.tool.atomic = self._orig_helper_atomic
        shutil.rmtree(self.dir, ignore_errors=True)

    def owner_build(self, k, out=None):
        """What a running owner shard does for queue position k."""
        out = out or self.out
        (out / "shard-0").mkdir(parents=True, exist_ok=True)
        return owner_tool.run_dataset(dict(self.plan["datasets"][k]), out / "shard-0", self.codes, None, None, None, set(),
                                      base._Guard(), log=lambda s: None)

    def args(self, **kw):
        a = argparse.Namespace(route="l7a", split_version=V2, helper_id="h01", owners="0", of=3, out=str(self.out),
                               margin=1.0, min_gap=1, max_datasets=1, dry_run=None, no_guard=True)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def helper_run(self, **kw):
        return helper.run(self.args(**kw), src=object(), cache=None, inv={}, held=set(), log=lambda s: None)

    def files(self, root):
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    def ctx_dirs(self, rec, out=None):
        return [(out or self.out) / "shard-0" / c["view_id"] for c in rec["contexts"]]


def strip_time(ctx):
    ctx = dict(ctx); ctx.pop("utc_us", None)
    return ctx


TIMING = ("cpu_s", "wall_s", "utc_us", "maxrss_mb", "contexts_written", "contexts_already_done")


def strip_record(r):
    r = {k: v for k, v in r.items() if k not in TIMING}
    r["contexts"] = [{k: v for k, v in c.items() if k not in ("cpu_s", "wall_s")} for c in r["contexts"]]
    return r


class Mechanism(Fixture):
    def test_helper_output_equals_owner_output_and_the_owner_skips_it(self):
        self.owner_build(0)                                     # the owner has started: its first record exists
        doc = self.helper_run()
        self.assertEqual((doc["built"], doc["lost_races"]), (1, 0), doc["events"])
        h = doc["events"][0]["dataset_hash"]
        self.assertEqual(h, HASHES[-1])                          # tail first
        rec_h = json.loads((self.out / "shard-0" / "_datasets" / f"{h}.json").read_bytes())
        self.assertTrue(owner_tool.dataset_done(self.out / "shard-0", h, self.codes))
        # the owner's loop: dataset_done is checked immediately before run_dataset, per dataset
        src = inspect.getsource(owner_tool.build)
        self.assertLess(src.index("if dataset_done(out_shard, h, codes):"), src.index("rec = run_dataset("))
        # determinism: the same dataset built by the owner path into another tree
        other = self.dir / "OTHER"
        rec_o = self.owner_build(len(HASHES) - 1, out=other)
        self.assertEqual(strip_record(rec_h), strip_record(rec_o))
        for dh, do in zip(self.ctx_dirs(rec_h), self.ctx_dirs(rec_o, other)):
            self.assertEqual(dh.name, do.name)
            for f in ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32", ".done"):
                self.assertEqual((dh / f).read_bytes(), (do / f).read_bytes(), f)
            self.assertEqual(strip_time(json.loads((dh / "context.json").read_bytes())),
                             strip_time(json.loads((do / "context.json").read_bytes())))
            self.assertEqual(sorted(p.name for p in dh.iterdir()), sorted(p.name for p in do.iterdir()))

    def test_selection_is_tail_first_never_the_frontier_and_respects_the_margin(self):
        self.owner_build(0)                                      # frontier = position 1
        a = self.args(margin=4.0, min_gap=2)
        picks = helper.dry_run(a, 5)
        self.assertEqual([p["dataset_hash"] for p in picks], [HASHES[5]])   # 40+30+20=90 >= 4*10; next: 40+30=70 < 4*20
        self.assertEqual((picks[0]["frontier"], picks[0]["position"]), (1, 5))
        loose = helper.dry_run(self.args(margin=0.0, min_gap=1), 10)
        self.assertEqual([p["dataset_hash"] for p in loose], [HASHES[5], HASHES[4], HASHES[3]])  # stops 1 short of the frontier
        self.assertNotIn(HASHES[1], [p["dataset_hash"] for p in loose])
        self.assertEqual(self.files(self.out).keys() - {"plan.json"}, {k for k in self.files(self.out) if k.startswith("shard-0")})

    def test_claims_are_exclusive_and_excluded_from_selection(self):
        self.owner_build(0)
        self.assertTrue(helper.claim(self.out, HASHES[5], "a"))
        self.assertFalse(helper.claim(self.out, HASHES[5], "b"))
        picks = helper.dry_run(self.args(margin=0.0, min_gap=1), 10)
        self.assertNotIn(HASHES[5], [p["dataset_hash"] for p in picks])

    def test_the_helper_refuses_other_code(self):
        rec = self.owner_build(0)
        p = self.out / "shard-0" / "_datasets" / f"{HASHES[0]}.json"
        r = json.loads(p.read_bytes()); r["code_sha256"] = dict(r["code_sha256"], **{"dltrain/t1raw.py": "0" * 64})
        p.write_text(json.dumps(r))
        with self.assertRaises(SystemExit):
            self.helper_run()

    def test_private_tmp_names(self):
        target = self.dir / "x.json"
        helper.unique_atomic(target, b"abc")
        helper.unique_atomic(target, b"abc")
        self.assertEqual(target.read_bytes(), b"abc")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir() if p.name.startswith(".")), [])
        self.assertNotEqual(f".{target.name}.{os.getpid()}.helper.tmp", f".{target.name}.tmp")   # never the owners' tmp path


class Races(Fixture):
    def staged(self, k):
        """The helper's computation of position k in its staging dir (what it holds just before publish)."""
        sd = self.out / "_helpers" / "h01" / "shard-0"
        sd.mkdir(parents=True, exist_ok=True)
        return helper.tool.run_dataset(dict(self.plan["datasets"][k]), sd, self.codes, None, None, None, set(), base._Guard(),
                                       log=lambda s: None), sd

    def test_owner_done_first(self):
        rec_o = self.owner_build(5)
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in (self.out / "shard-0").rglob("*") if p.is_file()}
        rec_h, sd = self.staged(5)
        res = helper.publish(rec_h, sd, self.out / "shard-0")
        self.assertEqual((res["lost_race"], len(res["already_done"]), res["published"], res["record_written"]), (None, 5, [], False))
        after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in (self.out / "shard-0").rglob("*") if p.is_file()}
        self.assertEqual(before, after)                          # not one byte or mtime of the owner's tree changed

    def test_owner_mid_write_is_a_clean_lost_race(self):
        rec_h, sd = self.staged(5)
        t = self.out / "shard-0" / rec_h["contexts"][2]["view_id"]
        t.mkdir(parents=True)
        (t / ".encoder.jtt1.gz.tmp").write_bytes(b"partial")      # the owner is in its phase 1 for this context
        res = helper.publish(rec_h, sd, self.out / "shard-0")
        self.assertIn("owner is writing", res["lost_race"])
        self.assertEqual(res["published"], [])
        self.assertEqual((t / ".encoder.jtt1.gz.tmp").read_bytes(), b"partial")
        self.assertFalse((self.out / "shard-0" / "_datasets" / f"{HASHES[5]}.json").exists())
        self.assertFalse(list((self.out / "shard-0").glob("*/.done")))
        rec_o = self.owner_build(5)                               # the owner finishes it itself
        self.assertEqual(rec_o["status"], "done")

    def test_an_empty_target_is_replaced_atomically_and_the_owner_writes_identical_bytes(self):
        rec_h, sd = self.staged(5)
        t = self.out / "shard-0" / rec_h["contexts"][0]["view_id"]
        t.mkdir(parents=True)                                    # the owner's mkdir, no file yet
        res = helper.publish(rec_h, sd, self.out / "shard-0")
        self.assertEqual((res["lost_race"], len(res["published"]), res["record_written"]), (None, 5, True))
        enc = (t / "encoder.jtt1.gz").read_bytes()
        owner_tool.atomic(t / "encoder.jtt1.gz", enc)            # the owner's own later write into the same path
        self.assertEqual((t / "encoder.jtt1.gz").read_bytes(), enc)

    def test_helper_first_then_the_owner_writes_nothing(self):
        rec_h, sd = self.staged(5)
        helper.publish(rec_h, sd, self.out / "shard-0")
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for d in self.ctx_dirs(rec_h) for p in d.iterdir()}
        rec_o = self.owner_build(5)                               # an owner that had started before the helper finished
        self.assertEqual((rec_o["status"], rec_o["contexts_written"], rec_o["contexts_already_done"]), ("done", 0, 5))
        after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for d in self.ctx_dirs(rec_h) for p in d.iterdir()}
        self.assertEqual(before, after)

    def test_a_crash_mid_publish_leaves_no_done_and_the_owner_completes(self):
        rec_h, sd = self.staged(5)
        real = os.rename
        n = {"dirs": 0}

        def flaky(a, b):
            if Path(a).is_dir():
                n["dirs"] += 1
                if n["dirs"] == 3:
                    raise OSError("simulated crash")
            return real(a, b)
        with mock.patch.object(helper.os, "rename", flaky):
            res = helper.publish(rec_h, sd, self.out / "shard-0")
        self.assertEqual(len(res["published"]), 2)
        self.assertTrue(res["lost_race"])
        self.assertFalse(list((self.out / "shard-0").glob("*/.done")))
        self.assertFalse(owner_tool.dataset_done(self.out / "shard-0", HASHES[5], self.codes))
        rec_o = self.owner_build(5)
        self.assertEqual((rec_o["status"], rec_o["contexts_written"]), ("done", 5))
        self.assertTrue(owner_tool.dataset_done(self.out / "shard-0", HASHES[5], self.codes))

    def test_nothing_in_the_owner_tree_is_ever_deleted(self):
        self.owner_build(0)
        snap = set(self.files(self.out / "shard-0"))
        self.helper_run(max_datasets=3, margin=0.0)
        self.assertLessEqual(snap, set(self.files(self.out / "shard-0")))


if __name__ == "__main__":
    unittest.main()
