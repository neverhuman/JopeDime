"""Data-free negative controls for KD direction, veto, closure and admission."""

import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from runtime import Audit, distillation_loss, live_per_context, loss, tensor_hash
from science import ARMS, KD_ARMS, SEEDS, batch_plan, contrast, decide, folds
from monitor import classify, sample

torch.set_num_threads(1)


def rows(n=19):
    return [
        dict(
            context_sha256=f"c{i}",
            family_id=f"f{i}",
            dl_split="DL-Train",
            source_split="train",
        )
        for i in range(n)
    ]


def healthy():
    return dict(
        rss=1000,
        threads=2,
        free=50_000_000_000,
        disk_used=0.9,
        scratch=1,
        load=20,
        available=20 * 1024**3,
    )


class H158Tests(unittest.TestCase):
    def setUp(self):
        self.pred = torch.tensor(
            [[0.2, -0.1, 0.4], [0.1, 0.5, -0.3]], requires_grad=True
        )
        self.f = torch.tensor([[0.3, 0.1, -0.8], [0.1, 0.7, -0.4]])
        self.g = torch.tensor([[True, False, True], [True, True, True]])
        self.teacher = torch.tensor(
            [[0.5, -0.2, 0.1], [-0.3, 0.4, 0.6]], requires_grad=True
        )
        self.bad = torch.tensor([True, False])

    def kd(self, alpha, veto, bad=None):
        return distillation_loss(
            self.pred,
            self.f,
            self.g,
            self.teacher,
            self.bad if bad is None else bad,
            alpha,
            veto,
        )

    def test_ctrl_exact_live(self):
        self.assertTrue(torch.equal(self.kd(0, False), loss(self.pred, self.f, self.g)))

    def test_all_veto_full_live_not_scaled(self):
        self.assertTrue(
            torch.equal(
                self.kd(0.7, True, torch.ones(2, dtype=torch.bool)),
                loss(self.pred, self.f, self.g),
            )
        )

    def test_zero_veto_equals_ablation(self):
        self.assertTrue(
            torch.equal(
                self.kd(0.7, True, torch.zeros(2, dtype=torch.bool)),
                self.kd(0.7, False),
            )
        )

    def test_partial_veto_and_teacher_to_student_direction(self):
        p = torch.softmax(-self.teacher.detach() / 0.2, -1)
        q = torch.softmax(-self.pred / 0.2, -1)
        kl = (p * (p.log() - q.log())).sum(-1)
        live = live_per_context(self.pred, self.f, self.g)
        expected = (live[0] + 0.3 * live[1] + 0.7 * kl[1]) / 2
        self.assertTrue(torch.allclose(self.kd(0.7, True), expected, atol=1e-6))

    def test_teacher_never_gets_grad(self):
        self.kd(0.7, False).backward()
        self.assertIsNone(self.teacher.grad)
        self.assertGreater(float(self.pred.grad.abs().sum()), 0)

    def test_identical_teacher_zero_pure_kl(self):
        x = distillation_loss(
            self.pred, self.f, self.g, self.pred.detach(), self.bad, 1, False
        )
        self.assertAlmostEqual(float(x.detach()), 0, places=6)

    def test_veto_grad_equals_live_on_bad_context(self):
        grad = torch.autograd.grad(self.kd(0.7, True), self.pred)[0]
        live_grad = torch.autograd.grad(loss(self.pred, self.f, self.g), self.pred)[0]
        self.assertTrue(torch.equal(grad[0], live_grad[0]))

    def test_teacher_shape_refused(self):
        with self.assertRaises(ValueError):
            distillation_loss(
                self.pred, self.f, self.g, self.teacher[:1], self.bad, 0.3, True
            )

    def test_teacher_nonfinite_refused(self):
        with torch.no_grad():
            self.teacher[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.kd(0.3, True)

    def test_alpha_refused(self):
        with self.assertRaises(ValueError):
            self.kd(1.1, True)

    def test_gate_failure_maps_to_cost_two(self):
        f = self.f.clone()
        f[0, 1] = -1000
        self.assertTrue(
            torch.equal(loss(self.pred, f, self.g), loss(self.pred, self.f, self.g))
        )

    def test_ties_never_advance(self):
        c = contrast(rows(), np.zeros((3, 19)), np.zeros(19))
        self.assertEqual(c["elo"], 1500)
        self.assertFalse(
            any(v["advance"] for v in decide({a: c for a in KD_ARMS}).values())
        )

    def test_positive_spread_advances(self):
        c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        self.assertTrue(
            all(v["advance"] for v in decide({a: c for a in KD_ARMS}).values())
        )

    def test_exact_margin_reject(self):
        c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        c["median_gain"] = 0.001
        self.assertFalse(
            any(v["advance"] for v in decide({a: c for a in KD_ARMS}).values())
        )

    def test_holm_prefix_cannot_be_bypassed(self):
        c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        cs = {
            a: dict(
                c,
                p_one_sided=p,
                holm_lcb_candidates=[-1, 1, 1] if i == 0 else [1, 1, 1],
            )
            for i, (a, p) in enumerate(zip(KD_ARMS, [0.001, 0.002, 0.003], strict=True))
        }
        self.assertFalse(any(v["advance"] for v in decide(cs).values()))

    def test_control_cannot_enter_holm(self):
        with self.assertRaises(ValueError):
            decide({a: {} for a in ARMS})

    def test_missing_hypothesis_refused(self):
        with self.assertRaises(ValueError):
            decide({})

    def test_median_not_seed_mean(self):
        c = contrast(
            rows(), np.array([[-0.01] * 19, [-0.01] * 19, [1.0] * 19]), np.zeros(19)
        )
        self.assertLess(c["median_gain"], 0)

    def test_equal_family_not_context(self):
        r = rows(4) + [dict(context_sha256=f"x{i}", family_id="f0") for i in range(30)]
        c = contrast(r, np.array([[1, -1, -1, -1] + [1] * 30] * 3), np.zeros(34))
        self.assertLess(c["median_gain"], 0)

    def test_missing_seed_or_nan_refused(self):
        for x in (np.zeros((2, 19)), np.full((3, 19), np.nan)):
            with self.assertRaises(ValueError):
                contrast(rows(), x, np.zeros(19))

    def test_folds_complete_disjoint(self):
        f = folds(rows(205))
        self.assertEqual([len(x) for x in f], [41] * 5)
        self.assertEqual(len(set(sum(f, []))), 205)
        self.assertEqual(f, folds(list(reversed(rows(205)))))

    def test_protected_and_duplicate_rows_refused(self):
        for r in (
            [dict(rows()[0], dl_split="DL-Val")],
            [dict(rows()[0], has_meta_test=True)],
            rows() + rows(),
        ):
            with self.assertRaises(ValueError):
                folds(r)

    def test_sampler_fit_only_paired_40(self):
        orders = {f"c{i}": np.arange(250) for i in range(8)}
        p = batch_plan(list(orders), orders, SEEDS[0])
        q = batch_plan(list(reversed(orders)), orders, SEEDS[0])
        self.assertEqual(len(p), 40)
        for (s, c, i), (ss, cc, ii) in zip(p, q, strict=True):
            self.assertEqual((s, c), (ss, cc))
            self.assertTrue(set(c) <= set(orders))
            for x, y in zip(i, ii, strict=True):
                np.testing.assert_array_equal(x, y)
                self.assertEqual(len(x), 192)

    def test_runtime_stub_allowed_but_protected_refused(self):
        a = Audit(
            ["/home/ubuntu/jope-data/h158/inputs/a.pt"],
            Path("/home/ubuntu/jope-data/h158/out"),
        )
        a.check(
            "open",
            (
                "/tmp/tmpabc/_remote_module_non_scriptable.py",
                "w",
                os.O_CREAT | os.O_WRONLY,
            ),
        )
        for p in [
            "/tmp/sealed/data.npz",
            "/home/ubuntu/jope-data/foreign/a.pt",
            "/home/ubuntu/JopeDime/data/flywheel_selfplay/a.pt",
        ]:
            with self.assertRaises(RuntimeError):
                a.check("open", (p, "r", 0))
        with self.assertRaises(RuntimeError):
            a.check(
                "open", ("/home/ubuntu/jope-data/h158/inputs/a.pt", "w", os.O_WRONLY)
            )

    def test_resource_limits_terminal(self):
        for key, value in [
            ("rss", 6_000_000_000),
            ("threads", 5),
            ("free", 39_999_999_999),
            ("disk_used", 0.95),
            ("scratch", 3_000_000_000),
            ("load", 85.01),
            ("available", 16 * 1024**3 - 1),
        ]:
            self.assertEqual(classify(dict(healthy(), **{key: value}))[0], "abort")
        self.assertEqual(
            classify(
                dict(
                    healthy(),
                    threads=4,
                    free=40_000_000_000,
                    load=85,
                    available=16 * 1024**3,
                )
            )[0],
            "run",
        )

    def test_reserved_blocks_counted(self):
        fs = type(
            "FS",
            (),
            dict(f_blocks=1000, f_bfree=70, f_bavail=40, f_frsize=1_000_000_000),
        )()
        with patch("monitor.os.statvfs", return_value=fs):
            s = sample(Path("/nonexistent-h158-test"))
        self.assertGreater(s["disk_used"], 0.95)
        self.assertEqual(classify(s)[0], "abort")

    def test_raw_tensor_hash_signed_zero(self):
        self.assertNotEqual(
            tensor_hash({"x": torch.tensor([0.0])}),
            tensor_hash({"x": torch.tensor([-0.0])}),
        )

    def test_wall_deadline_aborts(self):
        self.assertEqual(classify(dict(healthy(), elapsed=9000.01))[0], "abort")


if __name__ == "__main__":
    unittest.main()
