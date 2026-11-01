"""Data-free launch regression, SAM restoration, selection and boundary controls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import science as s
from runtime import Audit
from launch import read_launch
from monitor import classify
from runner import _perturbation_deltas, _optimizer_step, baseline_gate, subpanel


class H164Tests(unittest.TestCase):
    def test_launch_exact_audit_path_and_old_parent_refusal(self):
        # Real audit hook in a child: no hook survives into the other tests.
        code = """import sys, json, hashlib, tempfile
from pathlib import Path
from runtime import Audit
from launch import read_launch
root=Path(tempfile.mkdtemp(prefix="h164-test-",dir="/home/ubuntu/jope-data/h164/tests"))
out=root/"smoke";out.mkdir();m=root/"manifest.json";m.write_text("{}")
launch=dict(mode="smoke",manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest())
(out/"launch.json").write_text(json.dumps(launch));(root/"launch.json").write_text(json.dumps(launch))
a=Audit([m,out/"launch.json"],out);sys.addaudithook(a.check)
assert read_launch(out,m,True)==launch
try:(root/"launch.json").read_text()
except RuntimeError:pass
else:raise AssertionError("old parent path escaped audit")
assert len(a.refused)==1 and a.receipt()["protected_reads"]==0
"""
        root = Path("/home/ubuntu/jope-data/h164/tests")
        root.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).parent,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_launch_refuses_mismatch_cached_output_and_symlink(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
            root = Path(t)
            out = root / "out"
            out.mkdir()
            m = root / "m"
            m.write_text("{}")
            launch = dict(
                mode="smoke", manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest()
            )
            p = out / "launch.json"
            p.write_text(json.dumps(launch))
            self.assertEqual(read_launch(out, m, True), launch)
            with self.assertRaises(RuntimeError):
                read_launch(out, m, False)
            m.write_text("changed")
            with self.assertRaises(RuntimeError):
                read_launch(out, m, True)
            m.write_text("{}")
            (out / "metrics.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                read_launch(out, m, True)
            (out / "metrics.json").unlink()
            p.unlink()
            p.symlink_to(m)
            with self.assertRaises(RuntimeError):
                read_launch(out, m, True)

    def test_run_requires_merged_reviewed_matching_head(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as t:
            root = Path(t)
            out = root / "out"
            out.mkdir()
            m = root / "m"
            m.write_text("{}")
            p = out / "launch.json"
            v = dict(
                mode="run",
                manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),
                source_commit="head",
                merged_source_verified=True,
                review_disposition="PASS",
            )
            with patch("launch.subprocess.check_output", return_value="head\n"):
                p.write_text(json.dumps(v))
                self.assertEqual(read_launch(out, m, False), v)
                for key, value in [
                    ("merged_source_verified", False),
                    ("source_commit", "wrong"),
                    ("review_disposition", "pending"),
                ]:
                    p.write_text(json.dumps(dict(v, **{key: value})))
                    with self.assertRaises(RuntimeError):
                        read_launch(out, m, False)

    def test_audit_protected_unknown_and_readonly_input(self):
        a = Audit(
            ["/home/ubuntu/jope-data/h164/inputs/pinned"],
            "/home/ubuntu/jope-data/h164/custody/run",
        )
        for path, mode in [
            ("/home/ubuntu/jope-data/sealed/data", "r"),
            ("/home/ubuntu/jope-data/unknown", "r"),
            ("/home/ubuntu/jope-data/h164/inputs/pinned", "w"),
            ("/x/meta-test/p", "r"),
            ("/x/data/flywheel_selfplay/a", "r"),
        ]:
            with self.assertRaises(RuntimeError):
                a.check("open", (path, mode, 0))
        a.check("open", ("/home/ubuntu/jope-data/h164/inputs/pinned", "r", 0))

    def test_sam_radius_and_direction(self):
        p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        p.grad = torch.tensor([3.0, 4.0])
        for rho in [0.01, 0.05]:
            delta = _perturbation_deltas([p], rho)[0][1]
            torch.testing.assert_close(delta, torch.tensor([0.6, 0.8]) * rho)

    def test_asam_scaled_radius(self):
        p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
        p.grad = torch.tensor([4.0, 5.0])
        scale = p.detach().abs() + 0.01
        delta = _perturbation_deltas([p], 0.5, True)[0][1]
        torch.testing.assert_close(
            torch.linalg.vector_norm(delta / scale), torch.tensor(0.5)
        )
        expected = (
            0.5 * scale.square() * p.grad / torch.linalg.vector_norm(scale * p.grad)
        )
        torch.testing.assert_close(delta, expected)

    def test_sam_restores_exactly_on_second_pass_failure(self):
        for arm in ["SAM01", "SAM05", "ASAM05"]:
            p = torch.nn.Parameter(torch.tensor([0.1234567, -2.1234567]))
            initial = p.detach().clone()
            opt = torch.optim.AdamW([p], lr=0.01)
            calls = 0

            def objective():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("second-pass failure")
                return p.square().sum()

            with self.assertRaisesRegex(RuntimeError, "second-pass failure"):
                _optimizer_step(arm, opt, [p], objective, 1)
            self.assertTrue(torch.equal(p, initial))
            self.assertEqual(len(opt.state), 0)

    def test_actual_one_update_per_arm_and_pass_counts(self):
        for arm in s.ARMS:
            p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
            before = p.detach().clone()
            opt = torch.optim.AdamW([p], lr=0.01)
            calls = 0

            def objective():
                nonlocal calls
                calls += 1
                return p.square().sum()

            _, _, passes = _optimizer_step(arm, opt, [p], objective, 1)
            self.assertEqual(passes, 1 if arm == "CTRL" else 2)
            self.assertEqual(calls, passes)
            self.assertEqual(opt.state[p]["step"].item(), 1)
            self.assertFalse(torch.equal(p, before))

    def test_zero_nonfinite_gradients_refused(self):
        p = torch.nn.Parameter(torch.ones(2))
        for grad in [torch.zeros(2), torch.tensor([float("nan"), 1.0])]:
            p.grad = grad
            with self.assertRaises(RuntimeError):
                _perturbation_deltas([p], 0.05)

    def rows(self, n=25):
        return [
            dict(
                context_sha256=f"c{i}",
                family_id=f"f{i}",
                dl_split="DL-Train",
                has_meta_test=False,
            )
            for i in range(n)
        ]

    def test_folds_oof_coverage_and_no_fit_selector_assess_overlap(self):
        rows = self.rows()
        seen = []
        for outer in range(5):
            fit, select, assess = s.partition(rows, outer)
            a, b, c = ({r["family_id"] for r in x} for x in (fit, select, assess))
            self.assertFalse(a & b or a & c or b & c)
            seen.extend(r["context_sha256"] for r in assess)
        self.assertEqual(sorted(seen), sorted(r["context_sha256"] for r in rows))
        for bad in [
            rows + [rows[0]],
            [dict(r, dl_split="DL-Val") for r in rows],
            [dict(r, has_meta_test=True) for r in rows],
        ]:
            with self.assertRaises(ValueError):
                s.folds(bad)

    def test_paired_batch_plan_exact_inventory(self):
        ids = ["c1", "c2"]
        orders = {c: np.arange(200) for c in ids}
        p = s.batch_plan(ids, orders, s.SEEDS[0])
        q = s.batch_plan(ids, orders, s.SEEDS[0])
        self.assertEqual(len(p), 40)
        for (stream, cids, ixs), (stream2, cids2, ixs2) in zip(p, q, strict=True):
            self.assertEqual((stream, cids), (stream2, cids2))
            self.assertEqual(len(cids), 16)
            self.assertLessEqual(set(cids), set(ids))
            np.testing.assert_array_equal(ixs, ixs2)
            self.assertEqual(np.asarray(ixs).shape, (16, 192))

    def test_strict_margin_and_ties_reject(self):
        for gain in [0.0, 0.001, 0.001 + 5e-13]:
            panel = np.zeros((25, 5))
            panel[:, 1:] = gain
            v = s.choose(self.rows(), panel, 1)
            self.assertEqual(v["selected_step"], 0)
        panel = np.zeros((25, 5))
        panel[:, 1:] = 0.002
        v = s.choose(self.rows(), panel, 1)
        self.assertEqual(v["selected_step"], 10)
        panel[:, 2] += 0.5e-12
        self.assertEqual(s.choose(self.rows(), panel, 1)["best_step"], 10)

    def test_nonzero_descriptive_survives_step_zero_selection(self):
        panel = np.zeros((25, 5))
        panel[:, 1:] = [-0.1, -0.2, -0.3, -0.4]
        v = s.choose(self.rows(), panel, 1)
        self.assertEqual(v["best_step"], 0)
        self.assertEqual(v["best_nonzero_step"], 10)
        self.assertAlmostEqual(v["best_nonzero_gain"], -0.1)

    def test_selector_negative_lcb_rejects(self):
        panel = np.zeros((25, 5))
        panel[0, 1] = 1.0
        panel[1:, 1] = -0.01
        self.assertEqual(s.choose(self.rows(), panel, 1)["selected_step"], 0)

    def test_equal_family_weights_and_bt_anchor(self):
        _, fm = s.family_matrix(np.array([[0.0], [0.0], [3.0]]), ["a", "a", "b"])
        self.assertEqual(fm.mean(), 1.5)
        ratings = s.bt_ratings(np.ones((5, 5)) - np.eye(5), 0)
        np.testing.assert_allclose(ratings, 1500.0)

    def test_holm_four_arm_family(self):
        summary = {
            "contrasts": {
                arm: {
                    "vs_r882": dict(
                        p_one_sided=0.01, median_gain=0.002, ci95=[0.001, 0.003]
                    )
                }
                for arm in s.ARMS
            }
        }
        d = s.decide(summary)
        self.assertTrue(all(v["advance"] for v in d.values()))
        self.assertTrue(all(v["holm_adjusted_p"] == 0.04 for v in d.values()))
        summary["contrasts"]["CTRL"]["vs_r882"]["median_gain"] = 0.001
        self.assertFalse(s.decide(summary)["CTRL"]["advance"])
        del summary["contrasts"]["CTRL"]
        with self.assertRaises(ValueError):
            s.decide(summary)

    def test_complete_summary_and_null_reject(self):
        panel = np.ones((3, 25, 5))
        v = s.summarize(self.rows(), panel)
        np.testing.assert_allclose(v["median_elo"], 1500)
        self.assertFalse(any(x["advance"] for x in s.decide(v).values()))
        with self.assertRaises(ValueError):
            s.summarize(self.rows(), panel[:2])
        panel[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            s.summarize(self.rows(), panel)

    def test_guard_boundaries(self):
        base = dict(
            rss=1,
            threads=4,
            free=40_000_000_000,
            disk_used=0.94,
            scratch=1,
            load=85,
            available=16 * 1024**3,
            elapsed=43200,
        )
        self.assertEqual(classify(base), ("run", None))
        for key, value in [
            ("rss", 6_000_000_000),
            ("threads", 5),
            ("free", 39_999_999_999),
            ("disk_used", 0.95),
            ("scratch", 3_000_000_000),
            ("load", 85.01),
            ("available", 16 * 1024**3 - 1),
            ("elapsed", 43200.01),
        ]:
            self.assertEqual(classify(dict(base, **{key: value}))[0], "abort")

    def test_baseline_requires_every_value_not_mean_only(self):
        values = np.ones(91)
        ref = {"utility": {"per_context": values.tolist()}, "best_utility": 1.0}
        self.assertEqual(baseline_gate(values, ref)["status"], "PASS")
        values[0] += 0.1
        values[1] -= 0.1
        with self.assertRaises(RuntimeError):
            baseline_gate(values, ref)

    def test_subpanel_retains_context_order(self):
        p = dict(
            context_ids=["b", "a"], values=[2, 1], choices=[4, 3], invalid=[False, True]
        )
        self.assertEqual(subpanel(p, [{"context_sha256": "a"}])["values"], [1])


if __name__ == "__main__":
    unittest.main()
