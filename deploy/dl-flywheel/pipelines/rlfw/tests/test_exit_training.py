"""Exact trie KL, gradient likelihood, default-zero and dev-refusal fixtures."""

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from rlfw.exit_objective import (
    exact_trie_kl,
    objective,
    policy_imitation,
    sampled_kl,
    trie_logprob,
)
from rlfw.exit_offline_train import (
    TrainConfig,
    append_event,
    train_updates,
    write_checkpoint,
)
from rlfw.arena.game_spec import digest
from rlfw.exit_targets import build_targets
from rlfw.tests.test_exit_targets import config as target_config, proof, row

SHA = "a" * 64


class ToyTrie:
    identity = SHA
    programs = {
        (1, 3, 2): row(0, 0.0)["program"],
        (1, 4, 5, 2): {**row(1, 0.0)["program"], "tokens": [1, 4, 5, 2]},
        (1, 4, 6, 2): {**row(2, 0.0)["program"], "tokens": [1, 4, 6, 2]},
    }

    def get(self, prefix):
        return {
            (1,): (3, 4),
            (1, 3): (2,),
            (1, 4): (5, 6),
            (1, 4, 5): (2,),
            (1, 4, 6): (2,),
        }.get(tuple(prefix), ())


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        torch.set_num_threads(1)

    def policy(self, first=0.4, second=0.7):
        torch = self.torch

        class Toy(torch.nn.Module):
            trie_sha256 = SHA

            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(
                    torch.tensor(
                        [
                            math.log(first),
                            math.log(1 - first),
                            math.log(second),
                            math.log(1 - second),
                        ],
                        dtype=torch.float64,
                    )
                )

            def next_logits(self, context, prefix):
                if self.training:
                    raise AssertionError("policy likelihood must disable dropout")
                zero = self.values.sum() * 0
                logits = torch.stack([zero for _ in range(7)])
                if prefix == (1,):
                    logits[3], logits[4] = self.values[0], self.values[1]
                elif prefix == (1, 4):
                    logits[5], logits[6] = self.values[2], self.values[3]
                return logits

            def connected_zero(self):
                return self.values.sum() * 0

        return Toy()

    def test_full_support_kl_matches_leaf_hand_math_and_gradient(self):
        p, a = self.policy(), self.policy(0.6, 0.5)
        value = exact_trie_kl(
            p,
            a,
            {"train": None},
            ToyTrie(),
            deadline=time.time() + 10,
            check_train=lambda c: None,
        )
        probabilities = (0.4, 0.6 * 0.7, 0.6 * 0.3)
        reference = (0.6, 0.4 * 0.5, 0.4 * 0.5)
        expected = sum(x * math.log(x / y) for x, y in zip(probabilities, reference))
        self.assertAlmostEqual(float(value), expected, places=12)
        value.backward()
        self.assertTrue(self.torch.isfinite(p.values.grad).all())
        self.assertIsNone(a.values.grad)
        self.assertTrue(p.training and a.training)

    def test_on_policy_kl_estimator_has_exact_gradient_at_matched_draw_frequencies(
        self,
    ):
        p, a = self.policy(0.5, 0.5), self.policy(0.6, 0.5)
        exact_policy = self.policy(0.5, 0.5)

        def scalar(logits, choices, chosen, temperature):
            values = [float(logits[t]) / temperature for t in choices]
            peak = max(values)
            return (
                values[choices.index(chosen)]
                - peak
                - math.log(math.fsum(math.exp(x - peak) for x in values))
            )

        draws = [(1, 3, 2), (1, 3, 2), (1, 4, 5, 2), (1, 4, 6, 2)]
        sampled = sampled_kl(p, a, [None] * 4, draws, ToyTrie(), reference_fn=scalar)
        exact = exact_trie_kl(
            exact_policy,
            a,
            {"train": None},
            ToyTrie(),
            deadline=time.time() + 10,
            check_train=lambda c: None,
        )
        self.assertAlmostEqual(float(sampled), float(exact), places=12)
        sampled.backward()
        exact.backward()
        self.assertTrue(
            self.torch.allclose(
                p.values.grad, exact_policy.values.grad, atol=1e-12, rtol=0
            )
        )

    def test_equal_anchor_and_singleton_support_kl_are_exact_zero(self):
        p, a = self.policy(), self.policy()
        value = exact_trie_kl(
            p,
            a,
            {"train": None},
            ToyTrie(),
            deadline=time.time() + 10,
            check_train=lambda c: None,
        )
        self.assertEqual(float(value), 0.0)
        singleton = SimpleNamespace(
            identity=SHA,
            programs={(1, 3, 2)},
            get=lambda u: {(1,): (3,), (1, 3): (2,)}.get(u, ()),
        )
        p.next_logits = lambda c, u: self.fail("singleton KL must not forward")
        value = exact_trie_kl(
            p,
            a,
            {"train": None},
            singleton,
            deadline=time.time() + 10,
            check_train=lambda c: None,
        )
        self.assertEqual(float(value), 0.0)

    def test_incomplete_probe_support_mismatch_and_dev_are_hard_errors(self):
        p, a = self.policy(), self.policy()
        with self.assertRaises(TimeoutError):
            exact_trie_kl(
                p,
                a,
                {"train": None},
                ToyTrie(),
                deadline=1.0,
                check_train=lambda c: None,
            )
        p.trie_sha256 = "b" * 64
        with self.assertRaises(ValueError):
            exact_trie_kl(
                p,
                a,
                {"train": None},
                ToyTrie(),
                deadline=time.time() + 10,
                check_train=lambda c: None,
            )
        p.trie_sha256 = SHA

        def refused(c):
            raise ValueError("dev forbidden")

        with self.assertRaises(ValueError):
            exact_trie_kl(
                p,
                a,
                {"development": None},
                ToyTrie(),
                deadline=time.time() + 10,
                check_train=refused,
            )

    def test_summed_eos_likelihood_gradient_checked_against_reference(self):
        p = self.policy()

        def reference(logits, choices, chosen, temperature):
            values = [float(logits[t]) / temperature for t in choices]
            peak = max(values)
            return (
                values[choices.index(chosen)]
                - peak
                - math.log(math.fsum(math.exp(x - peak) for x in values))
            )

        value, per = trie_logprob(p, None, (1, 4, 5, 2), ToyTrie(), reference=reference)
        self.assertAlmostEqual(float(value), math.log(0.6 * 0.7), places=12)
        self.assertEqual(float(per[-1]), 0.0)
        value.backward()
        self.assertGreater(abs(float(p.values.grad[0])), 0.0)
        with self.assertRaises(ValueError):
            trie_logprob(p, None, (1, 4, 5, 2), ToyTrie(), reference=lambda *a: 123.0)

    def test_default_zero_returns_original_result_and_bitwise_gradients(self):
        torch = self.torch
        one = self.policy()
        two = self.policy()

        def base(model, batch):
            return (
                model.values.sin().square().sum(),
                {"base": True},
                {"base": "parts"},
            )

        expected = base(one, None)
        captured = []

        def original(model, batch):
            captured.append(base(model, batch))
            return captured[-1]

        actual = objective(
            original,
            two,
            None,
            pi_loss=lambda: self.fail("zero flag called PI"),
            kl_loss=lambda: self.fail("zero flag called KL"),
        )
        self.assertIs(actual, captured[0])
        expected[0].backward()
        actual[0].backward()
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(one.values.grad, two.values.grad))

    def test_zero_support_actor_and_dev_refusal_before_forward(self):
        p = self.policy()
        target = {"context_id": "train", "role": "train", "eligible": False}
        p.next_logits = lambda c, u: self.fail("unsupported actor must not forward")
        value = policy_imitation(
            p, {"train": None}, [target], ToyTrie(), check_train=lambda c: None
        )
        self.assertEqual(float(value), 0.0)
        value.backward()
        self.assertTrue(
            self.torch.equal(p.values.grad, self.torch.zeros_like(p.values))
        )
        target["role"] = "development"
        with self.assertRaises(ValueError):
            policy_imitation(
                p, {"train": None}, [target], ToyTrie(), check_train=lambda c: None
            )

    def test_prefix_actor_equals_weighted_summed_likelihood_and_gradient(self):
        p = self.policy()
        rows = [row(0, 0.0), row(1, 0.2)]
        rows[1]["program"]["tokens"] = [1, 4, 5, 2]
        target = build_targets(
            rows,
            target_config(),
            trie=ToyTrie(),
            role_check=lambda c: {"role": "train", "role_map_sha256": SHA},
            verify_receipt=proof,
            anchor_logprob=lambda c, t: math.log(0.4 if t == (1, 3, 2) else 0.42),
        )["manifest"]["contexts"]

        def scalar(logits, choices, chosen, temperature):
            values = [float(logits[t]) / temperature for t in choices]
            peak = max(values)
            return (
                values[choices.index(chosen)]
                - peak
                - math.log(math.fsum(math.exp(x - peak) for x in values))
            )

        loss = policy_imitation(
            p,
            {"train-a": None},
            target,
            ToyTrie(),
            check_train=lambda c: None,
            reference=scalar,
        )
        weights = {
            tuple(v["program"]["tokens"]): v["weight"] for v in target[0]["programs"]
        }
        a, b = weights[(1, 3, 2)], weights[(1, 4, 5, 2)]
        self.assertAlmostEqual(
            float(loss), -a * math.log(0.4) - b * math.log(0.42), places=12
        )
        loss.backward()
        self.assertAlmostEqual(float(p.values.grad[0]), 0.4 - a, places=12)
        self.assertAlmostEqual(float(p.values.grad[2]), b * (0.7 - 1), places=12)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "ckpt/challenger_toy").mkdir(parents=True)

        @dataclass
        class Cfg:
            pad_id: int = 0

            def to_json(self):
                return {"pad_id": self.pad_id}

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = Cfg()
                self.values = torch.nn.Parameter(
                    torch.tensor([0.0, 0.0, 0.0, 0.3, 0.1, 0.2, 0.1])
                )

            def forward(self, tokens, lengths, ctx_z, seq_ctx):
                return SimpleNamespace(
                    logits=self.values.reshape(1, 1, -1).expand(
                        tokens.shape[0], tokens.shape[1] - 1, -1
                    )
                )

        class Pin:
            def __init__(self):
                self.net = Net()
                self.torch = torch
                self.trie = ToyTrie()
                self.normalizer_sha256 = SHA
                self.vocab = SimpleNamespace(sha256=SHA)
                self.device = "cpu"

            def sample_many(self, contexts, *, seeds, deadline):
                return [
                    {"program": {"program_sha256": "b" * 64}, "tokens": [1, 3, 2]}
                    for c in contexts
                ], {}

        self.p, self.a = Pin(), Pin()
        self.parameter_sha = lambda n: hashlib.sha256(
            n.values.detach().numpy().tobytes()
        ).hexdigest()
        anchor = self.parameter_sha(self.a.net)
        tc = replace(target_config(), anchor_sha256=anchor)
        rows = [row(0, 0.0), row(1, 0.2)]
        rows[1]["program"]["tokens"] = [1, 4, 5, 2]
        self.targets = build_targets(
            rows,
            tc,
            trie=ToyTrie(),
            role_check=lambda c: {"role": "train", "role_map_sha256": SHA},
            verify_receipt=proof,
            anchor_logprob=lambda c, t: math.log(0.4),
        )
        self.cfg = TrainConfig(
            str(self.root),
            str(self.root / "ckpt/challenger_toy"),
            time.time() + 60,
            SHA,
            self.targets["target_sha256"],
            anchor,
            SHA,
            SHA,
            (),
            SHA,
            max_updates=2,
        )

        class Batch:
            context_ids = ["train-a"]
            tokens = torch.tensor([[1, 3, 2]])
            lengths = torch.tensor([3])
            ctx_z = torch.tensor([[0.0]])
            seq_ctx = torch.tensor([0])

            def to(self, device):
                return self

        self.checkpoints = []
        self.forwards = []

        def base(out, b):
            self.forwards.append(b.context_ids)
            return out.logits[0, 0, 3].square(), {"fixture": True}, {}

        self.runtime = {
            "policy": self.p,
            "anchor": self.a,
            "parameter_sha256": self.parameter_sha,
            "contexts": {
                "train-a" if i == 0 else "train-" + str(i): [0.0] for i in range(64)
            },
            "probe_contexts": tuple(
                "train-a" if i == 0 else "train-" + str(i) for i in range(64)
            ),
            "role_check": lambda c: {"role": "train", "role_map_sha256": SHA},
            "next_batch": lambda step: Batch(),
            "base_objective": base,
            "checkpoint": lambda n, opt, path, step, cfg, state: (
                self.checkpoints.append((str(path), step, state))
            ),
        }

    def test_two_synthetic_updates_change_learner_only_and_journal_then_checkpoint(
        self,
    ):
        initial = self.parameter_sha(self.p.net)
        result = train_updates(self.cfg, self.runtime, self.targets)
        self.assertEqual(result["completed_updates"], 2)
        self.assertNotEqual(initial, self.parameter_sha(self.p.net))
        self.assertEqual(initial, self.parameter_sha(self.a.net))
        self.assertEqual(len(self.checkpoints), 1)
        self.assertTrue(Path(self.checkpoints[0][0]).is_relative_to(self.cfg.out_dir))
        events = [
            json.loads(line)
            for line in (Path(self.cfg.out_dir) / "updates.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [e["kind"] for e in events],
            [
                "probe",
                "step_reserved",
                "step_complete",
                "step_reserved",
                "step_complete",
                "probe",
            ],
        )
        self.assertEqual(events[-1]["distinct_sample_sha_rate"], 1 / 64)
        self.assertEqual(result["state"], "shadow_candidate")

    def test_checkpoint_stores_changed_numeric_state_and_exact_support_create_only(
        self,
    ):
        from safetensors.torch import load_file

        source = self.root / "sealed-support-fixture"
        source.mkdir()
        names = (
            "vocab.json",
            "grammar.json",
            "trie-programs.json",
            "trie-identity.json",
            "preprocessing.json",
            "snapshot-spec.json",
            "service-config.json",
        )
        for name in names:
            (source / name).write_text(json.dumps({"fixture": name}))
        optimizer = self.torch.optim.AdamW(self.p.net.parameters(), lr=1e-4)
        optimizer.zero_grad()
        self.p.net.values.sum().backward()
        optimizer.step()
        path = Path(self.cfg.out_dir) / "update_000001"
        manifest = write_checkpoint(
            self.cfg,
            self.p,
            optimizer,
            source,
            path,
            1,
            "shadow_candidate",
            self.parameter_sha,
        )
        tensors = load_file(str(path / "state.safetensors"))
        self.assertTrue(self.torch.equal(tensors["model.values"], self.p.net.values))
        self.assertIn("optimizer.values.exp_avg", tensors)
        self.assertIn("rng.torch_cpu", tensors)
        self.assertEqual(manifest["parameter_sha256"], self.parameter_sha(self.p.net))
        for name in names:
            self.assertEqual((source / name).read_bytes(), (path / name).read_bytes())
        for line in (path / "MANIFEST.sha256").read_text().splitlines():
            pin, name = line.split("  ")
            self.assertEqual(
                pin, hashlib.sha256((path / name).read_bytes()).hexdigest()
            )
            self.assertEqual((path / name).stat().st_mode & 0o222, 0)
        self.assertFalse((path / "optimizer-control.json").exists())
        with self.assertRaises(FileExistsError):
            write_checkpoint(
                self.cfg,
                self.p,
                optimizer,
                source,
                path,
                1,
                "shadow_candidate",
                self.parameter_sha,
            )

    def test_dev_target_is_refused_before_any_forward_or_step(self):
        self.targets["manifest"]["contexts"][0]["role"] = "development"
        pin = digest(self.targets["manifest"])
        self.targets["target_sha256"] = pin
        cfg = replace(self.cfg, target_sha256=pin)
        before = self.parameter_sha(self.p.net)
        with self.assertRaises(ValueError):
            train_updates(cfg, self.runtime, self.targets)
        self.assertEqual(self.forwards, [])
        self.assertEqual(before, self.parameter_sha(self.p.net))
        self.assertEqual(self.checkpoints, [])

    def test_stop_and_deadline_cannot_be_reset_by_trainer(self):
        (self.root / "state").mkdir()
        (self.root / "state/STOP.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            train_updates(self.cfg, self.runtime, self.targets)
        self.assertEqual(self.forwards, [])
        (
            self.root / "state/STOP.json"
        ).unlink()  # This freshly created fixture marker only.
        with mock.patch(
            "rlfw.exit_offline_train.time.time", return_value=self.cfg.deadline_unix + 1
        ):
            with self.assertRaises(RuntimeError):
                train_updates(self.cfg, self.runtime, self.targets)
        with self.assertRaises(ValueError):
            replace(self.cfg, deadline_unix=1791550801.0)

    def test_probe_64_requirement_and_output_escape_are_refused(self):
        self.runtime["probe_contexts"] = self.runtime["probe_contexts"][:63]
        with self.assertRaises(ValueError):
            train_updates(self.cfg, self.runtime, self.targets)
        for path in (
            self.root / "state/challenger_bad",
            self.root / "ckpt/champion",
            self.root / "ckpt/challenger_a/sub",
        ):
            with self.assertRaises(ValueError):
                replace(self.cfg, out_dir=str(path))

    def test_journal_symlink_cannot_overwrite_other_evidence(self):
        evidence = self.root / "other-evidence.json"
        evidence.write_text("preserve")
        (Path(self.cfg.out_dir) / "updates.jsonl").symlink_to(evidence)
        with self.assertRaises(OSError):
            append_event(self.cfg.out_dir, {"kind": "test"})
        self.assertEqual(evidence.read_text(), "preserve")

    def test_probe_at_threshold_disqualifies_checkpoint(self):
        with mock.patch(
            "rlfw.exit_offline_train.exact_trie_kl",
            return_value=self.torch.tensor(0.05, dtype=self.torch.float64),
        ):
            result = train_updates(self.cfg, self.runtime, self.targets)
        self.assertEqual(result["state"], "disqualified_kl")
        self.assertEqual(self.checkpoints[-1][-1], "disqualified_kl")

    def test_role_change_after_loss_prevents_optimizer_update(self):
        before = self.parameter_sha(self.p.net)
        changed = []
        base = self.runtime["base_objective"]

        def compute(out, b):
            changed.append(True)
            return base(out, b)

        self.runtime["base_objective"] = compute
        self.runtime["role_check"] = lambda c: {
            "role": "development" if changed else "train",
            "role_map_sha256": SHA,
        }
        with self.assertRaises(ValueError):
            train_updates(self.cfg, self.runtime, self.targets)
        self.assertEqual(before, self.parameter_sha(self.p.net))
        result = json.loads((Path(self.cfg.out_dir) / "receipt.json").read_text())
        self.assertEqual(result["completed_updates"], 0)
        self.assertIsNone(result["reserved_update_outcome_unknown"])

    def test_interrupted_step_is_unknown_not_reported_as_zero_actual_updates(self):
        def interrupted(opt):
            with self.torch.no_grad():
                self.p.net.values.add_(0.01)
            raise RuntimeError("fixture interrupted optimizer")

        with mock.patch.object(self.torch.optim.AdamW, "step", interrupted):
            with self.assertRaises(RuntimeError):
                train_updates(self.cfg, self.runtime, self.targets)
        result = json.loads((Path(self.cfg.out_dir) / "receipt.json").read_text())
        self.assertEqual(result["completed_updates"], 0)
        self.assertEqual(result["reserved_update_outcome_unknown"], 1)
        self.assertEqual(result["state"], "incomplete_or_refused")
        self.assertEqual(self.checkpoints, [])


if __name__ == "__main__":
    unittest.main()
