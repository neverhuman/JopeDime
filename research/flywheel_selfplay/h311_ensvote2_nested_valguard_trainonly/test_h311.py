import tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
import guard_policy as p
import run_h311 as r
import trainonly_loader as tl
import evaluate_h311 as ev


class BoundaryTests(unittest.TestCase):
    def test_labels_refused_before_file_open(self):
        labels = r.ScopedLabels(
            ["train", "test"], np.zeros((2, 8), int), [0], root="/nonexistent"
        )
        with self.assertRaisesRegex(RuntimeError, "outer_test_labels"):
            labels.read([1])
        self.assertEqual(labels.denied, 1)
        self.assertEqual(labels.reads, 0)

    def test_authorized_train_labels_and_invalid_utility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "pools").mkdir()
            np.savez(
                root / "pools/a.npz",
                fitness=np.array([0.3, 0.8]),
                gate=np.array([True, False]),
            )
            a = r.ScopedLabels(["a"], np.array([[0, 1]]), [0], root)
            u, g = a.read([0])
            np.testing.assert_array_equal(u, [[-0.3, -2.0]])
            self.assertEqual(a.reads, 1)

    def test_feature_loader_never_decodes_poisoned_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pool.npz"
            prog = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.savez(
                path, program=prog, fitness=np.zeros(3), gate=np.ones(3, dtype=bool)
            )
            for name in ["fitness.npy", "gate.npy"]:
                off, _, _ = tl._member_layout(path, name)
                with path.open("r+b") as f:
                    f.seek(off)
                    f.write(b"\xff")
            with np.load(path) as z:
                with self.assertRaises(zipfile.BadZipFile):
                    z["fitness"]
            np.testing.assert_array_equal(r.read_program_only(path), prog)

    def test_train_prefix_poisoned_tail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "context_features.npz"
            x = np.arange(20, dtype=np.float32).reshape(5, 4)
            np.savez(path, cids=np.array(["a", "b", "c", "v0", "v1"]), lm48_z=x)
            off, _, _ = tl._member_layout(path, "lm48_z.npy")
            with path.open("r+b") as f:
                f.seek(off + 3 * 16)
                f.write(np.full((2, 4), np.nan, np.float32).tobytes())
            with tl.TrainOnlyNumpy({"a", "b", "c"}).load(path) as z:
                np.testing.assert_array_equal(z["lm48_z"], x[:3])

    def test_train_prefix_must_be_contiguous(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "context_features.npz"
            np.savez(path, cids=np.array(["a", "v", "b"]), lm48_z=np.zeros((3, 2)))
            with self.assertRaisesRegex(RuntimeError, "prefix"):
                tl.TrainOnlyNumpy({"a", "b"}).load(path)

    def test_partition_salts_match_reference(self):
        import hashlib

        for part, salt in zip(r.PARTS, r.SALTS):
            self.assertEqual(
                r.fold_of(part, "family"),
                hashlib.sha256((salt + "\0family").encode()).digest()[-1] % 4,
            )


class PolicyTests(unittest.TestCase):
    def test_fallback_matches_frozen_h295_for_two_vote_primary(self):
        import sys

        sys.path.insert(0, str(r.P / "vendor/h295_label_free_validity_guard"))
        import valguard as vg

        rng = np.random.default_rng(31101)
        n = 100
        members = rng.integers(0, 8, size=(3, n))
        votes = rng.integers(0, 9, size=(3, n))
        raw = rng.normal(size=(3, n, 8))
        prob = rng.uniform(0, 0.1, size=(n, 8))
        primary = np.array(
            [
                vg.get_voted_candidates_for_context(
                    members[:, i], votes[:, i], raw[:, i]
                )[0]
                for i in range(n)
            ]
        )
        for tau in p.GRID:
            expected = vg.apply_valguard_on_indices(
                np.arange(n), 0, members[None], votes[None], raw[None], prob, tau
            )
            np.testing.assert_array_equal(
                p.fallback(primary, members, votes, raw, prob, tau), expected
            )

    def inputs(self):
        return (
            np.array([1]),
            np.array([[1], [1], [2]]),
            np.array([[7], [6], [8]]),
            np.zeros((3, 1, 8)),
            np.zeros((1, 8)),
        )

    def test_safe_primary_retained(self):
        args = self.inputs()
        self.assertEqual(p.fallback(*args, 0.03).tolist(), [1])

    def test_unsafe_primary_uses_safe_member_alternative(self):
        args = list(self.inputs())
        args[-1][0, 1] = 0.2
        self.assertEqual(p.fallback(*args, 0.03).tolist(), [2])

    def test_never_switches_to_riskier_slot0(self):
        args = list(self.inputs())
        args[-1][0, [0, 1, 2]] = [0.4, 0.2, 0.3]
        self.assertEqual(p.fallback(*args, 0.03).tolist(), [1])

    def test_three_vote_primary_not_silently_two(self):
        args = list(self.inputs())
        args[0][0] = 0
        self.assertEqual(p.fallback(*args, 0.03).tolist(), [0])

    def test_slot0_escape_respects_strict_floor(self):
        args = list(self.inputs())
        args[0][0] = 0
        args[-1][0, 0] = 0.055
        self.assertEqual(p.fallback(*args, 0.03).tolist(), [0])
        args[-1][0, 0] = 0.056
        self.assertEqual(p.fallback(*args, 0.03).tolist(), [1])

    def test_tau_objective_ties_choose_larger(self):
        a, m, v, raw, prob = self.inputs()
        u = np.zeros((1, 8))
        g = np.ones((1, 8), bool)
        tau, _ = p.choose_tau(a, m, v, raw, prob, u, g, np.array(["f"]))
        self.assertEqual(tau, 0.06)

    def test_family_equal_not_context_equal(self):
        u = np.zeros((3, 2))
        u[:, 1] = [1, 1, -1]
        gain, _, _ = p.evaluate(
            np.ones(3, int), u, np.ones_like(u, bool), np.array(["a", "a", "b"])
        )
        self.assertEqual(gain, 0)

    def test_all_advance_gates_and_null_ties(self):
        good = p.decision([10] * 9, [8] * 9, [9] * 9, [3] * 9, [3] * 9, -0.002, -0.005)
        self.assertEqual(good["verdict"], "VALGUARD_NESTED_ADVANCE")
        self.assertEqual(
            p.decision([10] * 9, [8] * 9, [8] * 9, [3] * 9, [3] * 9, 0, 0)["verdict"],
            "VALGUARD_NULL_EQUIVALENT",
        )
        self.assertEqual(
            p.decision([9] * 9, [8] * 9, [9] * 9, [3] * 9, [3] * 9, 0, 0)["verdict"],
            "VALGUARD_NOT_ADVANCED",
        )
        self.assertEqual(
            p.decision([10] * 9, [8] * 9, [9] * 9, [3] * 9, [3] * 9, -0.0021, 0)[
                "verdict"
            ],
            "VALGUARD_NOT_ADVANCED",
        )
        self.assertEqual(
            p.decision([10] * 9, [8] * 9, [9] * 9, [3] * 9, [3] * 9, 0, -0.0051)[
                "verdict"
            ],
            "VALGUARD_NOT_ADVANCED",
        )
        self.assertEqual(
            p.decision([10] * 9, [8] * 9, [9] * 9, [3] * 9, [4] * 9, 0, 0)["verdict"],
            "VALGUARD_NOT_ADVANCED",
        )

    def test_null_clause_requires_six_strict_cells(self):
        self.assertEqual(
            p.decision([10] * 9, [8] * 9, [9] * 6 + [8] * 3, [3] * 9, [3] * 9, 0, 0)[
                "verdict"
            ],
            "VALGUARD_NESTED_ADVANCE",
        )
        self.assertEqual(
            p.decision([10] * 9, [8] * 9, [9] * 5 + [8] * 4, [3] * 9, [3] * 9, 0, 0)[
                "verdict"
            ],
            "VALGUARD_NULL_EQUIVALENT",
        )

    def test_shared_identical_cells_do_not_shrink_bootstrap(self):
        d = np.arange(20, dtype=float) / 20
        idx = np.random.default_rng(2).integers(0, 20, size=(100, 20))
        stat, boot = ev.statistics(d, idx)
        pooled = ev.pool([stat] * 9, [boot] * 9)
        self.assertEqual(pooled, {k: stat[k] for k in boot})

    def test_trim_removes_five_families_not_five_contexts(self):
        d = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 50.0, 60.0, 70.0, 80.0, 90.0])
        self.assertEqual(ev.trimmed(d), -1.0)


if __name__ == "__main__":
    unittest.main()
