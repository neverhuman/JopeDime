from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_matched_v2 as q
import seed_experiment as s


@contextmanager
def admitted(*args, **kwargs):
    yield


def fixture(root):
    output = root / "runs/matched-experiment"
    output.mkdir(parents=True)
    evaluator = root / "evaluate.py"
    evaluator.write_text('''#!/usr/bin/env python3
import hashlib,json,sys,time
from pathlib import Path
arg=lambda name:sys.argv[sys.argv.index(name)+1]
d=Path(arg('--output')); marker=d/'started'; resumed=marker.exists(); marker.write_text('yes'); time.sleep(.5)
while (Path(arg('--root'))/'hold-first-attempt').exists(): time.sleep(.01)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
score=.8 if d.name=='larger_pool' else 1.0
obs=[dict(genome='program',program_sha256=hashlib.sha256(b'program').hexdigest(),evaluation_index=0,
evaluator_sha256=sha(Path(sys.argv[0])),prepared_sha256='p',cut=0,runtime_seconds=.08,target_scale=1.0,
training_sha256='tr',validation_sha256='va',failure=None,metrics={'rmse':score,'semantic_sha256':'signature'})]
(d/'observations.json').write_text(json.dumps(obs))
(d/'search.json').write_text(json.dumps(dict(seeds_sha256=sha(Path(arg('--seeds'))),vocabulary_sha256=sha(Path(arg('--vocabulary'))),
rng_seed=int(arg('--rng-seed')),completed_evaluations=1,final_tests_opened=False,best_validation_rmse=score,best_program='program',wall_seconds=.08,resumed=resumed)))
''')
    evaluator.chmod(0o755)
    s.write(root / "vocab.json", {})
    view = dict(dataset="d", family="f", cut=0, prepared="prepared", prepared_sha256="p", training_sha256="tr", validation_sha256="va", target_scale=1.0, populations={}, provenance={})
    for arm in q.ARMS:
        path = output / "frozen" / arm / "seeds.json"
        s.write(path, {})
        view["populations"][arm] = str(path.relative_to(root))
        view["provenance"][arm] = {"program": {"program_origins": [{"kind": "bank"}], "selection": "neural_reward_heads", "retained_teacher": False}}
    manifest = {"schema": "parallel.seed_experiment.v2", "output": str(output.relative_to(root)), "views": [view],
        "evaluator": str(evaluator.relative_to(root)), "evaluator_sha256": s.digest(evaluator), "vocabulary": "vocab.json", "vocabulary_sha256": s.digest(root / "vocab.json"),
        "blocks": [{"id": f"seed-{seed}", "view": 0, "seed": seed} for seed in q.SEEDS], "selected_checkpoint_source": "frozen", "qualification": "used families"}
    s.write(output / "manifest.json", manifest)
    return output, manifest


class MatchedTests(unittest.TestCase):
    def test_partial_launch_counts_only_started_workers_and_preserves_interruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, "spare_lease", admitted), patch.object(q, "admission_reason", return_value=None):
            root = Path(tmp)
            output, manifest = fixture(root)
            start = q.runtime.popen
            children = []
            def partial(*args, **kwargs):
                if children:
                    raise FileNotFoundError("injected second rank launch failure")
                child = start(*args, **kwargs)
                children.append(child)
                return child
            with patch.object(q.runtime, "popen", partial), self.assertRaises(FileNotFoundError):
                q.run(root, manifest, 15, 1)
            self.assertIsNotNone(children[0].poll())
            report = q.report(root, manifest)
            self.assertEqual(report["physical_search_attempts"], 1)
            self.assertEqual(report["admitted_search_attempts"], 3)
            self.assertEqual(report["attempts_with_uncertain_worker_start"], [])
            report = q.run(root, manifest, 20, 1)
            self.assertTrue(report["complete"])
            self.assertEqual(report["physical_search_attempts"], 13)
            self.assertEqual(report["admitted_search_attempts"], 15)
            self.assertEqual(report["interrupted_blocks_preserved"], 1)

    def test_three_seed_blocks_are_recomputed_and_resume_without_relaunch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, "spare_lease", admitted), patch.object(q, "admission_reason", return_value=None):
            root = Path(tmp)
            output, manifest = fixture(root)
            result = q.run(root, manifest, 15, 2)
            self.assertTrue(result["complete"])
            self.assertEqual(result["physical_search_attempts"], 9)
            again = q.run(root, manifest, 15, 2)
            self.assertEqual(again["physical_search_attempts"], 9)
            self.assertEqual(len(again["comparisons"]), 6)
            self.assertTrue(all(v["complete"] for v in again["comparisons"].values()))
            ledger = (output / "launches.jsonl").read_bytes()
            path = next(output.glob("blocks/*/attempt-*/*/search.json"))
            value = s.read(path)
            value["best_validation_rmse"] = .0001
            s.write(path, value)
            with self.assertRaisesRegex(ValueError, "winner differs"):
                q.run(root, manifest, 15, 2)
            self.assertEqual(ledger, (output / "launches.jsonl").read_bytes())

    def test_interrupted_block_is_resumed_as_diagnostic_then_rerun_for_wall_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, "spare_lease", admitted):
            root = Path(tmp)
            output, manifest = fixture(root)
            cancelled = False
            (root / "hold-first-attempt").write_text("hold until cleanup")
            def admission(_):
                nonlocal cancelled
                if cancelled:
                    (root / "hold-first-attempt").unlink(missing_ok=True)
                if not cancelled and list(output.glob("blocks/*/attempt-*/*/started")):
                    cancelled = True
                    return "mandatory GPU handoff"
                return None
            with patch.object(q, "admission_reason", side_effect=admission):
                result = q.run(root, manifest, 20, 1)
            self.assertTrue(cancelled)
            self.assertTrue(result["complete"])
            self.assertEqual(result["interrupted_blocks_preserved"], 1)
            self.assertEqual(result["wall_eligible_searches"], 9)
            self.assertEqual(result["physical_search_attempts"], 15)
            self.assertEqual(result["launches_with_unknown_elapsed_cost"], [])
            receipts = result["attempts"]
            self.assertEqual(sum(r["wall_eligible"] for r in receipts), 3)


if __name__ == "__main__":
    unittest.main()
