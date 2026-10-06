import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from dltrain import cpuguard


def admission(tmp, sampled, admit=True, rolling=10.0):
    p = Path(tmp) / "cpu-admission.json"
    p.write_text(json.dumps({"admit": admit, "sampled_unix_seconds": sampled,
                             "nodes": {"xbabe2": {"cpu_rolling_percent": rolling}}}))
    return p


class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)

    def run_check(self, now=1000.0, floor=0, **kw):
        return cpuguard.check(now=now, path=admission(self.tmp.name, **kw), floor=floor)

    def test_admitted(self):
        ok, reasons, obs = self.run_check(sampled=990.0)
        self.assertTrue(ok, reasons)
        self.assertEqual(obs["age_s"], 10.0)

    def test_stale_future_admit_rolling(self):
        self.assertEqual(self.run_check(sampled=954.0)[1], ["admission_stale"])
        self.assertEqual(self.run_check(sampled=1006.0)[1], ["admission_stale"])
        self.assertEqual(self.run_check(sampled=990.0, admit=False)[1], ["admit_false"])
        self.assertEqual(self.run_check(sampled=990.0, admit="true")[1], ["admit_false"])
        self.assertEqual(self.run_check(sampled=990.0, rolling=75.0)[1], ["cpu_rolling_high"])
        self.assertEqual(self.run_check(sampled=990.0, rolling=None)[1], ["cpu_rolling_high"])
        self.assertEqual(self.run_check(sampled=990.0, rolling=float("nan"))[1], ["cpu_rolling_high"])
        self.assertEqual(self.run_check(sampled=True)[1], ["admission_stale"])

    def test_unreadable_and_floor(self):
        ok, reasons, _ = cpuguard.check(now=1.0, path=Path(self.tmp.name) / "missing.json", floor=0)
        self.assertEqual(reasons, ["admission_unreadable"])
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("[1, 2]")
        self.assertEqual(cpuguard.check(now=1.0, path=bad, floor=0)[1], ["admission_unreadable"])
        ok, reasons, obs = self.run_check(sampled=990.0, floor=10**30)
        self.assertEqual(reasons, ["host_floor"])
        self.assertIn("root_free_bytes", obs)


class GuardLoop(unittest.TestCase):
    def make(self, answers, **kw):
        t = {"now": 0.0}
        seq = iter(answers)
        g = cpuguard.Guard(clock=lambda: t["now"], sleep=lambda s: t.__setitem__("now", t["now"] + s),
                           checker=lambda: next(seq), poll_s=15.0, **kw)
        return g, t

    def test_pause_resume_journal(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            j = Path(tmp) / "guard.jsonl"
            g, _ = self.make([(False, ["admit_false"], {}), (False, ["cpu_rolling_high"], {}), (True, [], {})], journal_path=j)
            self.assertEqual(g.wait(), 30.0)
            events = [json.loads(line)["event"] for line in j.read_text().splitlines()]
            self.assertEqual(events, ["pause", "resume"])
            s = g.summary()
            self.assertEqual((s["checks"], s["pauses"], s["paused_s"]), (3, 1, 30.0))
            self.assertEqual(s["reasons"], ["admit_false", "cpu_rolling_high"])

    def test_tick_rate_limited(self):
        g, t = self.make([(True, [], {})] * 3, check_every_s=10.0)
        g.tick()
        t["now"] = 5.0
        g.tick()
        self.assertEqual(g.checks, 1)
        t["now"] = 10.0
        g.tick()
        self.assertEqual(g.checks, 2)

    def test_timeout(self):
        g, _ = self.make([(False, ["admission_stale"], {})] * 10, max_pause_s=30.0)
        with self.assertRaises(cpuguard.GuardTimeout):
            g.wait()


class SliceLimits(unittest.TestCase):
    def fake(self, out):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr="")

    def test_ok_and_drift(self):
        good = "CPUQuotaPerSecUSec=6s\nMemoryMax=12884901888\nMemorySwapMax=0\n"
        self.assertTrue(cpuguard.slice_limits(self.fake(good))[1])
        self.assertEqual(cpuguard.require_slice_limits(self.fake(good))["MemoryMax"], "12884901888")
        with self.assertRaises(RuntimeError):
            cpuguard.require_slice_limits(self.fake("CPUQuotaPerSecUSec=infinity\nMemoryMax=infinity\nMemorySwapMax=infinity\n"))


if __name__ == "__main__":
    unittest.main()
