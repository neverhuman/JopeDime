import argparse
import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import seed_experiment as s


def bind_view(root, view):
    source = root / "source.csv"
    source.write_text("x,y\n0,0\n1,1\n")
    recipe = root / view["prepared"] / f"cut-{view['cut']}/preprocessing.json"
    s.write(recipe, {"target_scale": view["target_scale"]})
    cut = {
        "index": view["cut"],
        "train_indices": [0],
        "validation_indices": [1],
        "training_sha256": view["training_sha256"],
        "validation_sha256": view["validation_sha256"],
        "preprocessing_sha256": s.digest(recipe),
    }
    path = root / view["prepared"] / "prepared.json"
    s.write(
        path,
        {
            "dataset": view["dataset"],
            "task": "regression",
            "cuts": [cut],
            "storage": {"source": "source.csv", "source_sha256": s.digest(source)},
        },
    )
    view.update(
        prepared_sha256=s.digest(path),
        preprocessing_sha256=s.digest(recipe),
        membership_sha256=s.sha(s.compact([[0], [1]])),
    )


def fixture_index():
    contexts, groups = [], []
    for family in range(8):
        for cut in range(8):
            context = len(contexts)
            contexts.append(
                {
                    "dataset": f"d{family}",
                    "family": f"f{family}",
                    "cut": cut,
                    "partition": "meta_validation",
                    "prepared_sha256": f"p{family}",
                }
            )
            groups.append(
                {
                    "context": context,
                    "family": f"f{family}",
                    "partition": "meta_validation",
                    "records": list(range(context * 15, (context + 1) * 15)),
                }
            )
    return {"contexts": contexts, "groups": groups}


class SeedExperimentTests(unittest.TestCase):
    def test_second_controller_cannot_load_a_stale_queue_or_launch_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs/experiment"
            s.write(
                output / "manifest.json",
                {
                    "output": "runs/experiment",
                    "schema": "parallel.seed_experiment.v1",
                    "production_champion_writes": False,
                },
            )
            args = argparse.Namespace(root=root, manifest=Path("runs/experiment/manifest.json"))
            with (output / "comparison-controller.lock").open("a") as owner:
                s.fcntl.flock(owner, s.fcntl.LOCK_EX | s.fcntl.LOCK_NB)
                with patch.object(s, "_compare") as implementation:
                    with self.assertRaisesRegex(RuntimeError, "another comparison controller"):
                        s.compare(args)
                    implementation.assert_not_called()
            with patch.object(s, "_compare", return_value="resumed") as implementation:
                self.assertEqual(s.compare(args), "resumed")
                implementation.assert_called_once_with(args)

    def test_panel_reconstruction_binds_exact_sample_records(self):
        index = fixture_index()
        panel = s.validation_panel(index, 8, 20260907)
        state = {
            "sampler_id": s.SAMPLER,
            "seed": 20260907,
            "world": 8,
            "validation": {"panel_sha256": s.sha(s.compact(panel))},
        }
        actual, rows = s.panel_cuts(index, state)
        self.assertEqual(actual, panel)
        self.assertEqual(len(actual), 8)
        self.assertTrue(all(len(rank) == 256 for rank in actual))
        self.assertEqual({r["family"] for r in rows}, {f"f{x}" for x in range(8)})
        index["groups"].reverse()
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            s.panel_cuts(index, state)

    def test_decision_cuts_exclude_panel_and_anchor_ids_instead_of_assuming_2_3(self):
        index = fixture_index()
        excluded = {(f"d{family}", cut) for family in range(8) for cut in (0, 1, 2, 4)}
        views = s.choose_views(index, excluded)
        self.assertEqual(len(views), 16)
        self.assertEqual([r["cut"] for r in views], [3, 5] * 8)
        excluded.update(("d0", cut) for cut in range(8))
        with self.assertRaisesRegex(ValueError, "insufficient"):
            s.choose_views(index, excluded)

    def test_missing_and_failed_prefixes_cannot_be_presented_as_complete_success(self):
        view = {"prepared_sha256": "p", "cut": 3, "training_sha256": "tr", "validation_sha256": "va"}
        programs = [str(i) for i in range(8)]
        results = {
            s.evaluation_key(view, p, "e"): {"status": "complete", "rmse": 1 + i}
            for i, p in enumerate(programs[:7])
        }
        incomplete = s.prefix_outcome(programs, view, results, "e", 8)
        self.assertFalse(incomplete["complete"])
        self.assertIsNone(incomplete["best_rmse"])
        results[s.evaluation_key(view, "7", "e")] = {"status": "timeout", "rmse": None}
        complete = s.prefix_outcome(programs, view, results, "e", 8)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["attempts_complete"])
        self.assertEqual(complete["failures"], 1)
        self.assertEqual(complete["best_rmse"], 1)
        self.assertEqual(complete["observed_best_rmse"], 1)
        results[s.evaluation_key(view, "7", "e")]["status"] = "cancelled"
        self.assertFalse(s.prefix_outcome(programs, view, results, "e", 8)["complete"])
        programs += [str(i) for i in range(8, 16)]
        for program in programs:
            results.setdefault(s.evaluation_key(view, program, "e"), {"status": "complete", "rmse": 1})
        results[s.evaluation_key(view, "7", "e")]["status"] = "timeout"
        self.assertFalse(s.prefix_outcome(programs, view, results, "e", 16)["complete"])

    def test_family_balancing_requires_both_decision_cuts(self):
        views = []
        for family in ("a", "b"):
            for cut in range(2):
                arms = {
                    arm: {"best8": {"complete": True, "best_rmse": rmse}}
                    for arm, rmse in (("larger_pool", 0.9 if family == "a" else 1.1), ("generic", 1.0))
                }
                views.append({"family": family, "cut": cut, "arms": arms})
        result = s.family_comparison(views, "larger_pool", "generic", "best8", 2)
        self.assertAlmostEqual(result["family_balanced_ratio"], (0.9 * 1.1) ** 0.5)
        self.assertEqual((result["wins"], result["losses"]), (1, 1))
        views[0]["arms"]["larger_pool"]["best8"]["complete"] = False
        result = s.family_comparison(views, "larger_pool", "generic", "best8", 2)
        self.assertFalse(result["complete"])
        self.assertNotIn("a", result["family_ratios"])

    def test_six_rankings_seal_before_fit_and_resume_reuses_exact_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs/experiment"
            evaluator = root / "evaluate.py"
            evaluator.write_text(
                "#!/usr/bin/env python3\nimport json,sys,time\nfrom pathlib import Path\ntime.sleep(.1)\np=Path(sys.argv[sys.argv.index('--output')+1]); p.write_text(json.dumps({'rmse': 1.0}))\n"
            )
            evaluator.chmod(0o755)
            programs = [f"program-{i}" for i in range(64)]
            view = {
                "dataset": "d",
                "family": "f",
                "cut": 3,
                "prepared": "prepared",
                "prepared_sha256": "p",
                "training_sha256": "tr",
                "validation_sha256": "va",
                "target_scale": 2.0,
            }
            manifest = {
                "schema": "parallel.seed_experiment.v1",
                "output": "runs/experiment",
                "production_champion_writes": False,
                "artifacts": [{"path": "evaluate.py", "sha256": s.digest(evaluator)}],
                "evaluator": "evaluate.py",
                "evaluator_sha256": s.digest(evaluator),
                "views": [view],
            }
            if "views" in manifest:
                manifest["schema"] = "parallel.seed_experiment.v2"
                manifest["evaluation_policy"] = s.evaluation_policy(
                    manifest["evaluator_sha256"], s.executable_dependencies(root / manifest["evaluator"])
                )
                manifest["runtime_dependencies"] = s.executable_dependencies(root / manifest["evaluator"])
                manifest["meeting_gate"] = manifest["evaluation_policy"]["meeting_gate"]
                for view in manifest["views"]:
                    bind_view(root, view)
                    view["evaluation_policy_sha256"] = s.sha(s.compact(manifest["evaluation_policy"]))
            s.write(output / "manifest.json", manifest)
            rankings_path = output / "proposals/d/cut-3/rankings.json"
            s.write(
                rankings_path,
                {
                    "dataset": "d",
                    "family": "f",
                    "cut": 3,
                    "manifest_sha256": s.digest(output / "manifest.json"),
                    "training_context_only": True,
                    "retained_teachers_consulted": False,
                    "heldout_labels_read": False,
                    "gp_outcomes_read": False,
                    "rankings": {arm: programs for arm in s.ARMS},
                },
            )
            s.write(
                rankings_path.parent / "complete.json",
                {
                    "manifest_sha256": s.digest(output / "manifest.json"),
                    "rankings_sha256": s.digest(rankings_path),
                },
            )
            args = argparse.Namespace(
                root=root,
                manifest=output / "manifest.json",
                workers=2,
                prefix=8,
                seconds=20,
                downstream=False,
            )

            @contextlib.contextmanager
            def lease(*args, **kwargs):
                yield

            with (
                patch("experiment_scheduler.admission_reason", return_value=None),
                patch("experiment_scheduler.spare_lease", lease),
            ):
                result = s.compare(args)
                self.assertEqual(result["physical_fits"], 8)
                self.assertEqual(result["logical_evaluations"], 48)
                self.assertEqual(result["physical_attempts"], 8)
                again = s.compare(args)
                self.assertEqual(again["physical_attempts"], 8)
                self.assertEqual(len(list((output / "evaluations").glob("*/result.json"))), 8)
                # Crash after the evaluator's atomic metrics write, before the
                # controller receipt: completed work must be recovered in place.
                orphan = next((output / "evaluations").glob("*/result.json"))
                key = orphan.parent.name
                orphan.unlink()
                ledger = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
                (output / "attempts.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in ledger if r["key"] != key)
                )
                recovered = s.compare(args)
                self.assertEqual(recovered["physical_attempts"], 8)
                self.assertEqual(recovered["physical_fits"], 8)
                self.assertTrue(s.read(orphan)["recovered_completed_artifact"])
                changed = json.loads(rankings_path.read_text())
                changed["rankings"]["current"] = list(reversed(programs))
                s.write(rankings_path, changed)
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    s.compare(args)

    def test_gpu_handoff_records_cancellation_and_resumes_owned_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs/experiment"
            evaluator = root / "evaluate.py"
            evaluator.write_text(
                "#!/usr/bin/env python3\nimport json,sys,time\nfrom pathlib import Path\nwhile not (Path(sys.argv[0]).parent/'resume-admitted').exists(): time.sleep(.01)\np=Path(sys.argv[sys.argv.index('--output')+1]); p.write_text(json.dumps({'rmse': 1.0}))\n"
            )
            evaluator.chmod(0o755)
            programs = [f"program-{i}" for i in range(64)]
            view = {
                "dataset": "d",
                "family": "f",
                "cut": 3,
                "prepared": "prepared",
                "prepared_sha256": "p",
                "training_sha256": "tr",
                "validation_sha256": "va",
                "target_scale": 1.0,
            }
            manifest = {
                "schema": "parallel.seed_experiment.v1",
                "output": "runs/experiment",
                "production_champion_writes": False,
                "artifacts": [{"path": "evaluate.py", "sha256": s.digest(evaluator)}],
                "evaluator": "evaluate.py",
                "evaluator_sha256": s.digest(evaluator),
                "views": [view],
            }
            if "views" in manifest:
                manifest["schema"] = "parallel.seed_experiment.v2"
                manifest["evaluation_policy"] = s.evaluation_policy(
                    manifest["evaluator_sha256"], s.executable_dependencies(root / manifest["evaluator"])
                )
                manifest["runtime_dependencies"] = s.executable_dependencies(root / manifest["evaluator"])
                manifest["meeting_gate"] = manifest["evaluation_policy"]["meeting_gate"]
                for view in manifest["views"]:
                    bind_view(root, view)
                    view["evaluation_policy_sha256"] = s.sha(s.compact(manifest["evaluation_policy"]))
            s.write(output / "manifest.json", manifest)
            ranking = output / "proposals/d/cut-3/rankings.json"
            s.write(
                ranking,
                {
                    "dataset": "d",
                    "family": "f",
                    "cut": 3,
                    "manifest_sha256": s.digest(output / "manifest.json"),
                    "training_context_only": True,
                    "retained_teachers_consulted": False,
                    "heldout_labels_read": False,
                    "gp_outcomes_read": False,
                    "rankings": {arm: programs for arm in s.ARMS},
                },
            )
            s.write(
                ranking.parent / "complete.json",
                {"manifest_sha256": s.digest(output / "manifest.json"), "rankings_sha256": s.digest(ranking)},
            )

            @contextlib.contextmanager
            def lease(*args, **kwargs):
                yield

            calls = 0

            def admission(root):
                nonlocal calls
                calls += 1
                if calls >= 4:
                    (root / "resume-admitted").touch()
                return "mandatory GPU block" if calls == 3 else None

            args = argparse.Namespace(
                root=root,
                manifest=output / "manifest.json",
                workers=2,
                prefix=8,
                seconds=20,
                downstream=False,
            )
            with (
                patch("experiment_scheduler.admission_reason", admission),
                patch("experiment_scheduler.spare_lease", lease),
            ):
                result = s.compare(args)
            attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["status"] == "cancelled" for r in attempts), 2)
            self.assertEqual(result["physical_attempts"], 10)
            self.assertEqual(result["physical_fits"], 8)
            self.assertEqual(result["logical_evaluations"], 48)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "escapes"):
                s.inside(Path(tmp), "../production")

    def test_downstream_harvests_complete_searches_before_handoff_and_binds_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs/experiment"
            evaluator = root / "search.py"
            evaluator.write_text("""#!/usr/bin/env python3
import json,sys,time,hashlib
from pathlib import Path
arg=lambda k: Path(sys.argv[sys.argv.index(k)+1])
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dest=arg('--output'); dest.mkdir(parents=True,exist_ok=True)
counter=dest/'executed'; counter.write_text(counter.read_text()+'x' if counter.exists() else 'x')
identity={'evaluator_sha256':digest(Path(sys.argv[0])),'prepared_sha256':'p','cut':3}
score=0.8 if dest.name=='larger_pool' else 1.0
obs=[dict(identity, metrics={'rmse':score,'semantic_sha256':'signature'}, failure=None, runtime_seconds=.01), dict(identity, metrics=None, failure='candidate_or_view_timeout', runtime_seconds=.02)]
for i,r in enumerate(obs): r.update(genome=str(i),program_sha256=hashlib.sha256(str(i).encode()).hexdigest(),evaluation_index=i,target_scale=1.0)
(dest/'observations.json').write_text(json.dumps(obs))
(dest/'search.json').write_text(json.dumps({'seeds_sha256':digest(arg('--seeds')),'vocabulary_sha256':digest(arg('--vocabulary')),'final_tests_opened':False,'completed_evaluations':2,'wall_seconds':0.03,'best_validation_rmse':score,'best_program':'0'}))
time.sleep(.3)
""")
            evaluator.chmod(0o755)
            (root / "corpus").mkdir()
            (root / "corpus/vocabulary.json").write_text("{}")
            manifest = {
                "output": "runs/experiment",
                "corpus": "corpus",
                "evaluator": "search.py",
                "evaluator_sha256": s.digest(evaluator),
            }
            if "views" in manifest:
                manifest["schema"] = "parallel.seed_experiment.v2"
                manifest["evaluation_policy"] = s.evaluation_policy(
                    manifest["evaluator_sha256"], s.executable_dependencies(root / manifest["evaluator"])
                )
                manifest["runtime_dependencies"] = s.executable_dependencies(root / manifest["evaluator"])
                manifest["meeting_gate"] = manifest["evaluation_policy"]["meeting_gate"]
                for view in manifest["views"]:
                    bind_view(root, view)
                    view["evaluation_policy_sha256"] = s.sha(s.compact(manifest["evaluation_policy"]))
            s.write(output / "manifest.json", manifest)
            view = {"dataset": "d", "family": "f", "cut": 3, "prepared": "prepared", "prepared_sha256": "p"}
            ranking = output / "proposals/d/cut-3/rankings.json"
            s.write(ranking, {})
            assemblies = []
            for arm in ("current", "larger_pool", "generic"):
                seeds = output / f"downstream/d/cut-3/{arm}/seeds.json"
                s.write(seeds, {"candidates": [{"genome": str(i)} for i in range(80)]})
                receipt = {
                    "manifest_sha256": s.digest(output / "manifest.json"),
                    "rankings_sha256": s.digest(ranking),
                    "seeds_sha256": s.digest(seeds),
                }
                s.write(seeds.parent / "assembly.json", receipt)
                assemblies.append({"dataset": "d", "cut": 3, "arm": arm, "seeds_sha256": s.digest(seeds)})
            s.write(
                output / "assemblies-complete.json",
                {"manifest_sha256": s.digest(output / "manifest.json"), "assemblies": assemblies},
            )

            @contextlib.contextmanager
            def lease(*args, **kwargs):
                yield

            handed_off = False

            def admission(root):
                nonlocal handed_off
                if (
                    not handed_off
                    and len(list((output / "downstream-results").glob("*/cut-*/*/search.json"))) == 2
                ):
                    handed_off = True
                    return "mandatory GPU handoff"
                return None

            with (
                patch("experiment_scheduler.admission_reason", admission),
                patch("experiment_scheduler.spare_lease", lease),
            ):
                result = s.run_downstream(root, manifest, [(view, {})], time.monotonic() + 10, 2)
                self.assertTrue(result["complete"])
                self.assertTrue(result["no_aggregate_regression"])
                self.assertEqual(result["physical_evaluations"], 6)
                self.assertEqual(result["physical_fits"], 3)
                self.assertEqual(result["failed_evaluations"], 3)
                self.assertEqual(result["failure_codes"], {"candidate_or_view_timeout": 3})
                self.assertAlmostEqual(result["candidate_seconds_observed"], 0.09)
                self.assertTrue(handed_off)
                again = s.run_downstream(root, manifest, [(view, {})], time.monotonic() + 10, 2)
                self.assertTrue(again["complete"])
                self.assertEqual(
                    [p.read_text() for p in (output / "downstream-results").glob("*/cut-*/*/executed")],
                    ["x"] * 3,
                )
                path = output / "downstream-results/d/cut-3/current/search.json"
                changed = s.read(path)
                changed["seeds_sha256"] = "tampered"
                s.write(path, changed)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    s.run_downstream(root, manifest, [(view, {})], time.monotonic() + 10, 2)


if __name__ == "__main__":
    unittest.main()
