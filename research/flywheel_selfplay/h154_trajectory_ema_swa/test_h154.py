"""Data-free controls for H154 trajectory EMA/SWA accumulators, train-only stat recalibration, 7-arm Holm gates, hard-stop guards, and squash-compatible source binding."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from monitor import classify, environment
from runner import baseline_gate, evaluate, selected
from runtime import (
    BOUND_FILES,
    CACHE,
    HERE,
    INPUT,
    Audit,
    TrajectoryEMAState,
    TrajectorySWAState,
    apply_stat_recalibration,
    assert_frozen_anchor,
    bind_source,
    build,
    check_frozen,
    compute_recal_rms,
    eval_batch_cache_key,
    guards,
    loss,
    tensor_hash,
    validate_eval_cache_key,
)
from science import (
    ARMS,
    BOOT_SEED,
    CANDIDATE_ARMS,
    DOSE_INVALID_BOUNDS,
    EMA_DECAYS,
    FOLD_SEED,
    HOLM_M,
    HOLM_M_CTRL,
    RECAL_BOUNDS,
    RECAL_SLATES,
    SEEDS,
    SWA_WINDOWS,
    batch_plan,
    contrast,
    decide,
    dose_check,
    folds,
    study_verdict,
)


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


def valid_drifts_and_utilities():
    drifts = {
        "CTRL": [0.44, 0.44, 0.44],
        "EMA_099": [0.28, 0.28, 0.28],
        "EMA_0995": [0.22, 0.22, 0.22],
        "EMA_0999": [0.12, 0.12, 0.12],
        "SWA_K5": [0.39, 0.39, 0.39],
        "SWA_K10": [0.34, 0.34, 0.34],
        "SWA_K20": [0.25, 0.25, 0.25],
    }
    utilities = {
        arm: np.full((3, 91), 1.10 + 0.001 * idx, dtype=np.float64)
        for idx, arm in enumerate(ARMS)
    }
    return drifts, utilities


class H154Tests(unittest.TestCase):
    def test_registered_arms_and_constants(self):
        self.assertEqual(SEEDS, (154101, 154102, 154103))
        self.assertEqual((FOLD_SEED, BOOT_SEED), (154140, 154141))
        self.assertEqual(
            EMA_DECAYS, {"EMA_099": 0.99, "EMA_0995": 0.995, "EMA_0999": 0.999}
        )
        self.assertEqual(SWA_WINDOWS, {"SWA_K5": 5, "SWA_K10": 10, "SWA_K20": 20})
        self.assertEqual(
            CANDIDATE_ARMS,
            ("EMA_099", "EMA_0995", "EMA_0999", "SWA_K5", "SWA_K10", "SWA_K20"),
        )
        self.assertEqual(set(ARMS), {"CTRL", *CANDIDATE_ARMS})
        self.assertEqual((HOLM_M, HOLM_M_CTRL), (7, 6))
        self.assertEqual(RECAL_BOUNDS, (0.85, 1.15))
        self.assertEqual(RECAL_SLATES, 8)
        self.assertEqual(DOSE_INVALID_BOUNDS, (0.95, 1.05))

    def test_ties_reject(self):
        c = contrast(rows(), np.zeros((3, 19)), np.zeros(19))
        drifts, utilities = valid_drifts_and_utilities()
        d = decide(
            {a: c for a in ARMS},
            {a: c for a in CANDIDATE_ARMS},
            drifts,
            utilities,
        )
        self.assertTrue(all(not v["advance"] for v in d["vs_r882"].values()))
        self.assertTrue(all(not v["advance"] for v in d["vs_ctrl"].values()))
        self.assertEqual(c["elo"], 1500.0)
        self.assertEqual(
            study_verdict(d),
            (
                "NO_ADVANCE",
                "NO_ADVANCE",
                "NO_TRAJECTORY_AVERAGING_GAIN_OVER_CTRL",
                None,
            ),
        )

    def test_positive_spread_advances_only_on_three_seeds(self):
        c3 = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        drifts, utilities = valid_drifts_and_utilities()
        d3 = decide(
            {a: c3 for a in ARMS},
            {a: c3 for a in CANDIDATE_ARMS},
            drifts,
            utilities,
        )
        self.assertTrue(all(v["advance"] for v in d3["vs_r882"].values()))
        self.assertTrue(all(v["advance"] for v in d3["vs_ctrl"].values()))
        self.assertLess(c3["top3_positive_share"], 0.8)
        self.assertEqual(
            study_verdict(d3),
            ("ADVANCE", "ADVANCE", "TRAJECTORY_AVERAGING_BEATS_CTRL", "CTRL"),
        )

        c2 = contrast(
            rows(),
            np.array([[0.01] * 19, [0.01] * 19, [-0.001] * 19]),
            np.zeros(19),
        )
        d2 = decide(
            {a: c2 for a in ARMS},
            {a: c2 for a in CANDIDATE_ARMS},
            drifts,
            utilities,
        )
        self.assertTrue(all(not v["advance"] for v in d2["vs_r882"].values()))
        self.assertTrue(all(not v["advance"] for v in d2["vs_ctrl"].values()))

    def test_ctrl_governed_by_holm_seven_and_cannot_bypass(self):
        ctrl_c = dict(
            p_one_sided=0.01,
            median_gain=0.01,
            ci95=[0.001, 0.02],
            seeds_beating=3,
            top3_positive_share=0.2,
        )
        fail_c = dict(
            p_one_sided=1.0,
            median_gain=-0.01,
            ci95=[-0.02, -0.001],
            seeds_beating=0,
            top3_positive_share=1.0,
        )
        drifts, utilities = valid_drifts_and_utilities()
        r882_map = {a: (ctrl_c if a == "CTRL" else fail_c) for a in ARMS}
        ctrl_map = {a: fail_c for a in CANDIDATE_ARMS}
        d = decide(r882_map, ctrl_map, drifts, utilities)
        self.assertAlmostEqual(d["vs_r882"]["CTRL"]["holm_adjusted_p"], 0.07)
        self.assertFalse(d["vs_r882"]["CTRL"]["gates"]["holm"])
        self.assertFalse(d["vs_r882"]["CTRL"]["advance"])
        self.assertEqual(
            study_verdict(d),
            (
                "NO_ADVANCE",
                "NO_ADVANCE",
                "NO_TRAJECTORY_AVERAGING_GAIN_OVER_CTRL",
                None,
            ),
        )

    def test_holm_seven_vs_r882_and_six_vs_ctrl(self):
        drifts, utilities = valid_drifts_and_utilities()
        c_border = dict(
            p_one_sided=0.0075,
            median_gain=0.01,
            ci95=[0.001, 0.02],
            seeds_beating=3,
            top3_positive_share=0.3,
        )
        # 0.0075 * 7 = 0.0525 >= 0.05 -> fails 7-arm Holm vs r882, but 0.0075 * 6 = 0.045 < 0.05 -> passes 6-arm Holm vs CTRL
        d = decide(
            {a: c_border for a in ARMS},
            {a: c_border for a in CANDIDATE_ARMS},
            drifts,
            utilities,
        )
        self.assertFalse(any(v["advance"] for v in d["vs_r882"].values()))
        self.assertTrue(all(v["advance"] for v in d["vs_ctrl"].values()))
        self.assertEqual(
            study_verdict(d),
            (
                "NO_ADVANCE",
                "NO_ADVANCE",
                "TRAJECTORY_AVERAGING_BEATS_CTRL",
                None,
            ),
        )

    def test_dose_check_and_invalid_dose_blocks_advance(self):
        drifts, utilities = valid_drifts_and_utilities()
        # Make EMA_099 within 5% of CTRL drift AND identical utilities to CTRL -> DOSE_INVALID
        drifts["EMA_099"] = [0.44 * 0.95, 0.44 * 0.95, 0.44 * 0.95]
        utilities["EMA_099"] = utilities["CTRL"].copy()
        doses = dose_check(drifts, utilities)
        self.assertEqual(doses["EMA_099"]["status"], "DOSE_INVALID")
        self.assertEqual(doses["EMA_0995"]["status"], "DOSE_VALID")

        pos_c = contrast(rows(), np.full((3, 19), 0.01), np.zeros(19))
        neg_c = contrast(rows(), np.full((3, 19), -0.01), np.zeros(19))
        r882_map = {a: (pos_c if a == "EMA_099" else neg_c) for a in ARMS}
        ctrl_map = {a: (pos_c if a == "EMA_099" else neg_c) for a in CANDIDATE_ARMS}
        d = decide(r882_map, ctrl_map, drifts, utilities)
        self.assertTrue(d["vs_r882"]["EMA_099"]["efficacy_only_advance"])
        self.assertFalse(d["vs_r882"]["EMA_099"]["gates"]["dose"])
        self.assertFalse(d["vs_r882"]["EMA_099"]["advance"])
        self.assertFalse(d["vs_ctrl"]["EMA_099"]["advance"])
        self.assertEqual(
            study_verdict(d),
            (
                "NO_ADVANCE",
                "NO_ADVANCE",
                "NO_TRAJECTORY_AVERAGING_GAIN_OVER_CTRL",
                None,
            ),
        )

    def test_trajectory_ema_state(self):
        import torch

        init = {
            "adapter_10tok.w": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "frozen.w": torch.tensor([9.0, 10.0], dtype=torch.float32),
        }
        with self.assertRaises(ValueError):
            TrajectoryEMAState(init, ("adapter_10tok.w",), 0.9)
        ema = TrajectoryEMAState(init, ("adapter_10tok.w",), 0.99)
        with self.assertRaises(RuntimeError):
            ema.export_state_dict(init)

        expected = np.array([1.0, 2.0], dtype=np.float64)
        for step in range(1, 41):
            cur = {
                "adapter_10tok.w": torch.tensor(
                    [float(step), float(2 * step)], dtype=torch.float32
                ),
                "frozen.w": torch.tensor([99.0, 99.0], dtype=torch.float32),
            }
            ema.after_fast_step(cur)
            expected = 0.99 * expected + 0.01 * np.array(
                [float(step), float(2 * step)], dtype=np.float64
            )
        out = ema.export_state_dict(init)
        np.testing.assert_allclose(
            out["adapter_10tok.w"].numpy(),
            expected.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )
        # Non-trainable tensor must come from base_state_dict unchanged
        self.assertTrue(torch.equal(out["frozen.w"], init["frozen.w"]))

    def test_trajectory_swa_state(self):
        import torch

        init = {
            "adapter_10tok.w": torch.tensor([0.0], dtype=torch.float32),
            "frozen.w": torch.tensor([7.0], dtype=torch.float32),
        }
        swa = TrajectorySWAState(("adapter_10tok.w",))
        with self.assertRaises(ValueError):
            swa.export_state_dict(init, 3)
        recorded_steps = []
        for step in range(1, 41):
            cur = {
                "adapter_10tok.w": torch.tensor([float(step)], dtype=torch.float32),
                "frozen.w": torch.tensor([88.0], dtype=torch.float32),
            }
            if swa.after_step(step, cur):
                recorded_steps.append(step)
        self.assertEqual(recorded_steps, list(range(2, 41, 2)))
        self.assertEqual(len(swa.snapshots), 20)

        # Last 5 round snapshots are steps [32, 34, 36, 38, 40] -> mean = 36.0
        out5 = swa.export_state_dict(init, 5)
        self.assertAlmostEqual(float(out5["adapter_10tok.w"].item()), 36.0, places=6)
        self.assertTrue(torch.equal(out5["frozen.w"], init["frozen.w"]))

        # Last 10 round snapshots are steps [22, 24, ..., 40] -> mean = 31.0
        out10 = swa.export_state_dict(init, 10)
        self.assertAlmostEqual(float(out10["adapter_10tok.w"].item()), 31.0, places=6)

        # Last 20 round snapshots are steps [2, 4, ..., 40] -> mean = 21.0
        out20 = swa.export_state_dict(init, 20)
        self.assertAlmostEqual(float(out20["adapter_10tok.w"].item()), 21.0, places=6)

    def test_compute_recal_rms_and_apply_stat_recalibration(self):
        import torch

        rms = compute_recal_rms(
            [np.array([3.0, -3.0]), np.array([3.0, -3.0])],
            [np.array([2.0, 2.0]), np.array([-2.0, -2.0])],
        )
        self.assertAlmostEqual(rms["mm_rms"], 3.0)
        self.assertAlmostEqual(rms["traj_rms"], 2.0)

        with self.assertRaises(RuntimeError):
            compute_recal_rms([np.zeros(4)], [np.ones(4)])

        names = (
            "adapter_10tok.mm_prior_out.2.weight",
            "adapter_10tok.mm_prior_out.2.bias",
            "adapter_10tok.traj_delta_head.2.weight",
            "adapter_10tok.traj_delta_head.2.bias",
            "adapter_10tok.other.weight",
        )
        sd = {n: torch.tensor([2.0, -4.0], dtype=torch.float32) for n in names}
        sd["frozen.weight"] = torch.tensor([5.0, 6.0], dtype=torch.float32)

        # Unclipped ratio (1.10) for mm, clipped upper ratio (2.0 -> 1.15) for traj
        out, receipt = apply_stat_recalibration(
            sd,
            names,
            ref_rms=dict(mm_rms=1.10, traj_rms=2.0),
            cand_rms=dict(mm_rms=1.00, traj_rms=1.0),
            bounds=(0.85, 1.15),
        )
        self.assertAlmostEqual(receipt["applied_mm_scale"], 1.10)
        self.assertAlmostEqual(receipt["applied_traj_scale"], 1.15)
        np.testing.assert_allclose(
            out["adapter_10tok.mm_prior_out.2.weight"].numpy(),
            np.array([2.2, -4.4], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out["adapter_10tok.traj_delta_head.2.weight"].numpy(),
            np.array([2.3, -4.6], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        self.assertTrue(
            torch.equal(
                out["adapter_10tok.other.weight"], sd["adapter_10tok.other.weight"]
            )
        )
        self.assertTrue(torch.equal(out["frozen.weight"], sd["frozen.weight"]))

        # Lower clip boundary (0.5 -> 0.85)
        _, receipt_lo = apply_stat_recalibration(
            sd,
            names,
            ref_rms=dict(mm_rms=0.5, traj_rms=0.5),
            cand_rms=dict(mm_rms=1.0, traj_rms=1.0),
            bounds=(0.85, 1.15),
        )
        self.assertAlmostEqual(receipt_lo["applied_mm_scale"], 0.85)
        self.assertAlmostEqual(receipt_lo["applied_traj_scale"], 0.85)

    def test_median_not_mean(self):
        c = contrast(
            rows(), np.array([[-0.01] * 19, [-0.01] * 19, [1.0] * 19]), np.zeros(19)
        )
        self.assertLess(c["median_gain"], 0)
        self.assertLess(c["ci95"][1], 0)

    def test_equal_family_not_context(self):
        r = rows(4) + [dict(context_sha256=f"x{i}", family_id="f0") for i in range(30)]
        v = np.array([[1, -1, -1, -1] + [1] * 30] * 3, dtype=float)
        c = contrast(r, v, np.zeros(34))
        self.assertLess(c["median_gain"], 0)

    def test_concentration_refusal(self):
        v = np.zeros((3, 19))
        v[:, 0] = 1
        c = contrast(rows(), v, np.zeros(19))
        self.assertEqual(c["top3_positive_share"], 1.0)

    def test_missing_seed_refusal(self):
        with self.assertRaises(ValueError):
            contrast(rows(), np.zeros((2, 19)), np.zeros(19))

    def test_nonfinite_refusal(self):
        with self.assertRaises(ValueError):
            contrast(rows(), np.full((3, 19), np.nan), np.zeros(19))

    def test_missing_arm_refusal(self):
        drifts, utilities = valid_drifts_and_utilities()
        with self.assertRaises(ValueError):
            decide({}, {}, drifts, utilities)
        with self.assertRaises(ValueError):
            dose_check({"CTRL": [0.1, 0.1, 0.1]}, utilities)

    def test_fold_disjoint(self):
        r = rows(205)
        f = folds(r)
        self.assertEqual([len(x) for x in f], [41] * 5)
        self.assertEqual(len(set(sum(f, []))), 205)
        self.assertEqual(f, folds(r))

    def test_protected_fold_refusal(self):
        r = rows()
        r[0]["dl_split"] = "DL-Val"
        with self.assertRaises(ValueError):
            folds(r)

    def test_paired_batch_plan(self):
        orders = {"a": np.arange(250), "b": np.arange(250)}
        a = batch_plan(["a", "b"], orders, SEEDS[0])
        b = batch_plan(["b", "a"], orders, SEEDS[0])
        self.assertEqual(len(a), 40)
        for x, y in zip(a, b, strict=True):
            self.assertEqual(x[:2], y[:2])
            self.assertTrue(
                all(np.array_equal(i, j) for i, j in zip(x[2], y[2], strict=True))
            )

    def test_resource_boundaries_all_abort(self):
        s = dict(
            rss=1,
            threads=2,
            free=40_000_000_000,
            disk_used=0.94,
            scratch=0,
            load=85.0,
            available=16 * 1024**3,
        )
        self.assertEqual(classify(s), ("run", None))
        for k, v, expected_why in [
            ("rss", 6_000_000_000, "RSS"),
            ("threads", 5, "TOTAL_THREADS"),
            ("free", 39_999_999_999, "DISK"),
            ("disk_used", 0.95, "DISK"),
            ("scratch", 1_000_000_000, "SCRATCH"),
            ("load", 85.01, "LOAD"),
            ("available", 16 * 1024**3 - 1, "MEMORY"),
        ]:
            self.assertEqual(classify(dict(s, **{k: v})), ("abort", expected_why))

    def test_runtime_guards_hard_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("os.getpriority", return_value=19),
                mock.patch(
                    "shutil.disk_usage",
                    return_value=mock.Mock(free=50_000_000_000, total=100_000_000_000),
                ),
                mock.patch("os.getloadavg", return_value=(10.0, 10.0, 10.0)),
                mock.patch("runtime.tree_bytes", return_value=1000),
            ):
                res = guards(tmp)
                self.assertEqual(res["disk_free_bytes"], 50_000_000_000)

            # Nice < 19 -> hard stop
            with mock.patch("os.getpriority", return_value=10):
                with self.assertRaises(RuntimeError):
                    guards(tmp)

            # Disk free < 40GB -> hard stop
            with (
                mock.patch("os.getpriority", return_value=19),
                mock.patch(
                    "shutil.disk_usage",
                    return_value=mock.Mock(free=39_000_000_000, total=100_000_000_000),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    guards(tmp)

            # Load > 85.0 -> hard stop
            with (
                mock.patch("os.getpriority", return_value=19),
                mock.patch(
                    "shutil.disk_usage",
                    return_value=mock.Mock(free=50_000_000_000, total=100_000_000_000),
                ),
                mock.patch("os.getloadavg", return_value=(85.1, 10.0, 10.0)),
            ):
                with self.assertRaises(RuntimeError):
                    guards(tmp)

            # Scratch >= 1GB -> hard stop
            with (
                mock.patch("os.getpriority", return_value=19),
                mock.patch(
                    "shutil.disk_usage",
                    return_value=mock.Mock(free=50_000_000_000, total=100_000_000_000),
                ),
                mock.patch("os.getloadavg", return_value=(10.0, 10.0, 10.0)),
                mock.patch("runtime.tree_bytes", return_value=600_000_000),
            ):
                with self.assertRaises(RuntimeError):
                    guards(tmp)

    def test_squash_compatible_bind_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            for idx, rel in enumerate(BOUND_FILES):
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"bound-content-{idx}\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "reviewed head"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            reviewed = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
                .decode()
                .strip()
            )
            # Create a second commit with identical tree (simulating squash-merged main commit)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "squash commit"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            source = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
                .decode()
                .strip()
            )
            prereg_sha = hashlib.sha256(
                (repo / BOUND_FILES[0]).read_bytes()
            ).hexdigest()
            receipt = bind_source(
                dict(reviewed_head=reviewed, source_commit=source),
                repo=repo,
                prereg_sha=prereg_sha,
            )
            self.assertEqual(receipt["reviewed_head"], reviewed)
            self.assertEqual(receipt["source_commit"], source)

            # Mutating a bound file locally must fail bind_source
            (repo / BOUND_FILES[1]).write_text("tampered\n")
            with self.assertRaises(RuntimeError):
                bind_source(
                    dict(reviewed_head=reviewed, source_commit=source),
                    repo=repo,
                    prereg_sha=prereg_sha,
                )

    def test_environment_one_thread(self):
        self.assertEqual(environment()["OMP_NUM_THREADS"], "1")

    def test_read_audit(self):
        out_dir = Path("/home/ubuntu/jope_plan/evidence/H154/attempt_unit_test")
        a = Audit(
            [
                "/home/ubuntu/jope-data/h50/input/selected.json",
                "/home/ubuntu/jope-data/h154/inputs/round-882.pt",
            ],
            out_dir,
        )
        # Allowed read and allowed output/tmp writes
        a.check("open", ("/home/ubuntu/jope-data/h50/input/selected.json", "r", 0))
        a.check("open", (str(out_dir / "metrics.json"), "w", 0))
        a.check("open", ("/tmp/_remote_module_non_scriptable.py", "w", 0))

        # Refused reads and writes across /home/ubuntu/JopeDime, ~/jope_plan (outside output),
        # ~/jope-data (except output), and every sealed/holdout/meta-test path
        for path, mode in [
            ("/home/ubuntu/jope-data/holdout/data", "r"),
            ("/home/ubuntu/jope-data/sealed/panel.json", "r"),
            ("/home/ubuntu/jope-data/meta-test/eval.json", "r"),
            ("/home/ubuntu/jope-data/meta_test/eval.json", "r"),
            ("/home/ubuntu/jope-data/qualification.lock", "r"),
            ("/home/ubuntu/JopeDime/data/flywheel_selfplay/live", "r"),
            ("/home/ubuntu/jope-data/h50/input/pools/unknown.npz", "r"),
            ("/home/ubuntu/JopeDime/README.md", "w"),
            (
                "/home/ubuntu/JopeDime/.agent/worktrees/h154-agy/research/flywheel_selfplay/h154_trajectory_ema_swa/runner.py",
                "w",
            ),
            ("/home/ubuntu/JopeDime/data/flywheel_selfplay/live", "w"),
            ("/home/ubuntu/jope_plan/AGENT_CHAT.md", "w"),
            ("/home/ubuntu/jope_plan/evidence/H154/launch.json", "w"),
            ("/home/ubuntu/jope_plan/evidence/H154/other_attempt/metrics.json", "w"),
            ("/home/ubuntu/jope-data/h50/input/selected.json", "w"),
            ("/home/ubuntu/jope-data/h154/inputs/round-882.pt", "w"),
            ("/home/ubuntu/jope-data/h154/other/file.json", "w"),
            ("/tmp/sealed/leak.json", "w"),
            ("/tmp/holdout/leak.json", "w"),
            ("/tmp/meta-test/leak.json", "w"),
            ("/tmp/meta_test/leak.json", "w"),
            ("/tmp/qualification.lock", "w"),
            (str(out_dir / "sealed" / "leak.json"), "w"),
            (str(out_dir / "holdout" / "leak.json"), "w"),
        ]:
            with self.assertRaises(
                RuntimeError, msg=f"expected refusal for {path} ({mode})"
            ):
                a.check("open", (path, mode, 0))

    def test_eval_cache_key_structure_and_validation(self):
        fake_pools = {
            "cid_a": dict(
                program=np.ones((8, 412), dtype=np.float32),
                fitness=np.linspace(-1.0, 1.0, 8, dtype=np.float64),
                gate=np.ones(8, dtype=bool),
            ),
            "cid_b": dict(
                program=np.full((8, 412), 2.0, dtype=np.float32),
                fitness=np.linspace(-0.5, 0.5, 8, dtype=np.float64),
                gate=np.ones(8, dtype=bool),
            ),
        }
        k1 = eval_batch_cache_key(
            "DL-Val", ["cid_a", "cid_b"], [np.arange(8), np.arange(8)], fake_pools
        )
        self.assertEqual(validate_eval_cache_key(k1), k1)
        self.assertEqual(k1[0], "DL-Val")
        self.assertEqual(k1[1], ("cid_a", "cid_b"))
        self.assertEqual(len(k1[2]), 64)

        # Split change must change key
        k_split = eval_batch_cache_key(
            "Fold-0", ["cid_a", "cid_b"], [np.arange(8), np.arange(8)], fake_pools
        )
        self.assertNotEqual(k1, k_split)

        # Context ID order change must change key
        k_cids = eval_batch_cache_key(
            "DL-Val", ["cid_b", "cid_a"], [np.arange(8), np.arange(8)], fake_pools
        )
        self.assertNotEqual(k1, k_cids)

        # Candidate subset/index change must change key
        k_ix = eval_batch_cache_key(
            "DL-Val", ["cid_a", "cid_b"], [np.arange(4), np.arange(4)], fake_pools
        )
        self.assertNotEqual(k1[2], k_ix[2])

        # Mutating candidate program features must change candidate_set_sha256
        fake_pools["cid_a"]["program"][0, 0] = 99.0
        k_prog = eval_batch_cache_key(
            "DL-Val", ["cid_a", "cid_b"], [np.arange(8), np.arange(8)], fake_pools
        )
        self.assertNotEqual(k1[2], k_prog[2])

        # Bare batch index or bare context tuple must be rejected by validate_eval_cache_key
        for bad_key in [
            0,
            ("cid_a", "cid_b"),
            ("", ("cid_a",), "a" * 64),
            ("DL-Val", (), "a" * 64),
            ("DL-Val", ("cid_a",), "short"),
        ]:
            with self.assertRaises(RuntimeError):
                validate_eval_cache_key(bad_key)

    def test_anchor_frozen_and_cache_bitwise_parity_dl_val_and_fold0(self):
        import json
        import torch

        torch.set_num_threads(1)
        pins = json.loads((HERE / "manifest.json").read_text())
        net, guard, names, frozen, fused = build(Path(pins["checkpoint"]))

        # 1. Assert anchor is a frozen non-adapter module and tampering is caught
        anchor_sha = assert_frozen_anchor(net, names, net._anchor_sha256)
        self.assertEqual(anchor_sha, net._anchor_sha256)
        with self.assertRaises(RuntimeError):
            assert_frozen_anchor(net, (*names, "adapter_10tok.anchor.leak"))
        first_anchor_param = next(net.adapter_10tok.anchor.parameters())
        first_anchor_param.requires_grad_(True)
        try:
            with self.assertRaises(RuntimeError):
                assert_frozen_anchor(net, names, net._anchor_sha256)
        finally:
            first_anchor_param.requires_grad_(False)

        # Bare batch index on net._eval_batch_key must be refused by cached_anchor/cached_sem24
        net._eval_batch_key = 0
        try:
            with self.assertRaises(RuntimeError):
                fused.compute_semantic_struct_sig_24d_torch(torch.zeros((1, 4, 412)))
        finally:
            net._eval_batch_key = None

        # 2. Load DL-Val (all 91 contexts) and Fold 0 (all 254 contexts)
        train, val = selected()
        groups = folds(train)
        fold0 = [r for r in train if r["family_id"] in groups[0]]
        self.assertEqual(len(val), 91)
        self.assertEqual(len(fold0), 254)

        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        # Keep full 3000 candidates on first batch (8 contexts) of DL-Val and Fold 0,
        # and 64 candidates across all remaining 91 DL-Val and 254 Fold 0 contexts.
        full_cids = {r["context_sha256"] for r in val[:8] + fold0[:8]}
        for r in val + fold0:
            cid = r["context_sha256"]
            p = INPUT / "pools" / f"{cid}.npz"
            with np.load(p, allow_pickle=False) as z:
                cap = None if cid in full_cids else 64
                program = z["program"][:cap].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"][:cap].astype(np.float64),
                    gate=z["gate"][:cap].astype(bool),
                )
        with np.load(CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            tok = z["tokens_10x32"]
            mask = z["mask_10"]
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
                if c in pools
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool)
                for i, c in enumerate(ids)
                if c in pools
            }

        state_a = {k: v.detach().clone() for k, v in net.state_dict().items()}
        gen = torch.Generator(device="cpu").manual_seed(154999)
        state_b = {k: v.detach().clone() for k, v in state_a.items()}
        for n in names:
            noise = torch.randn(
                state_b[n].shape, generator=gen, dtype=torch.float32
            ).to(dtype=state_b[n].dtype)
            state_b[n] = (state_b[n] + 0.08 * noise).clone()

        results_by_state = {}
        for label, sd in [("state_a", state_a), ("state_b", state_b)]:
            net.load_state_dict(sd, strict=True)
            check_frozen(net, guard, names, frozen)
            val_uncached = evaluate(
                net,
                pools,
                tokens,
                masks,
                val,
                split_name="DL-Val",
                use_cache=False,
                return_scores=True,
            )
            val_cached = evaluate(
                net,
                pools,
                tokens,
                masks,
                val,
                split_name="DL-Val",
                use_cache=True,
                return_scores=True,
            )
            fold0_uncached = evaluate(
                net,
                pools,
                tokens,
                masks,
                fold0,
                split_name="Fold-0",
                use_cache=False,
                return_scores=True,
            )
            fold0_cached = evaluate(
                net,
                pools,
                tokens,
                masks,
                fold0,
                split_name="Fold-0",
                use_cache=True,
                return_scores=True,
            )
            for u, c in [(val_uncached, val_cached), (fold0_uncached, fold0_cached)]:
                self.assertEqual(u["values"], c["values"])
                self.assertEqual(u["choices"], c["choices"])
                self.assertEqual(u["invalid"], c["invalid"])
                self.assertEqual(u["context_ids"], c["context_ids"])
                for s_u, s_c in zip(u["scores"], c["scores"], strict=True):
                    self.assertTrue(np.array_equal(s_u, s_c))
            check_frozen(net, guard, names, frozen)
            results_by_state[label] = (val_cached, fold0_cached)

        # Confirm the two adapter states produce genuinely different score vectors on both splits
        self.assertFalse(
            all(
                np.array_equal(a, b)
                for a, b in zip(
                    results_by_state["state_a"][0]["scores"],
                    results_by_state["state_b"][0]["scores"],
                    strict=True,
                )
            )
        )
        self.assertFalse(
            all(
                np.array_equal(a, b)
                for a, b in zip(
                    results_by_state["state_a"][1]["scores"],
                    results_by_state["state_b"][1]["scores"],
                    strict=True,
                )
            )
        )
        # Confirm cache keys stored in net._anchor_cache include both "DL-Val" and "Fold-0" splits
        cached_splits = {k[0] for k in net._anchor_cache}
        self.assertEqual(cached_splits, {"DL-Val", "Fold-0"})

    def test_signed_zero_byte_guard(self):
        import torch

        self.assertNotEqual(
            tensor_hash({"x": torch.tensor([0.0])}),
            tensor_hash({"x": torch.tensor([-0.0])}),
        )

    def test_buffer_change_guard(self):
        import torch

        self.assertNotEqual(
            tensor_hash({"buffer": torch.tensor([1.0])}),
            tensor_hash({"buffer": torch.tensor([2.0])}),
        )

    def test_control_loss(self):
        import torch

        torch.set_num_threads(1)
        p = torch.tensor([[0.2, 0.4]], requires_grad=True)
        f = torch.tensor([[0.3, 100.0]])
        g = torch.tensor([[True, False]])
        value = loss(p, f, g)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_baseline_exact_refusal(self):
        ref = {"utility": {"per_context": [1.0] * 91}, "best_utility": 1.0}
        self.assertEqual(baseline_gate([1.0] * 91, ref)["matches"], 91)
        with self.assertRaises(RuntimeError):
            baseline_gate([1.00000001] + [1.0] * 90, ref)


if __name__ == "__main__":
    unittest.main()
