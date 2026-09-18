import argparse
import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import binary_verify
import experiment_runtime as runtime
import seed_experiment as s
import verify_experiment
from test_seed_experiment import bind_view


@contextmanager
def admitted(*args, **kwargs):
    yield


def fixture(root):
    output = root / "runs/experiment"
    evaluator = root / "evaluate.py"
    evaluator.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index('--output')+1]); p.write_text(json.dumps({'rmse':1.0}))\n")
    evaluator.chmod(0o755)
    programs = [f"program-{i}" for i in range(64)]
    policy = s.evaluation_policy(s.digest(evaluator), s.executable_dependencies(evaluator))
    view = dict(dataset="d", family="f", cut=3, prepared="prepared", prepared_sha256="p",
                training_sha256="tr", validation_sha256="va", target_scale=2.0,
                preprocessing_sha256="pp", membership_sha256="m", evaluation_policy_sha256=s.sha(s.compact(policy)))
    bind_view(root, view)
    manifest = dict(schema="parallel.seed_experiment.v2", output="runs/experiment", production_champion_writes=False,
                    evaluation_policy=policy, meeting_gate=policy["meeting_gate"], runtime_dependencies=s.executable_dependencies(evaluator),
                    artifacts=[dict(path="evaluate.py", sha256=s.digest(evaluator))], evaluator="evaluate.py",
                    evaluator_sha256=s.digest(evaluator), views=[view])
    s.write(output / "manifest.json", manifest)
    path = output / "proposals/d/cut-3/rankings.json"
    s.write(path, dict(dataset="d", family="f", cut=3, manifest_sha256=s.digest(output / "manifest.json"),
                       training_context_only=True, retained_teachers_consulted=False, heldout_labels_read=False,
                       gp_outcomes_read=False, rankings={arm: programs for arm in s.ARMS}))
    s.write(path.parent / "complete.json", dict(manifest_sha256=s.digest(output / "manifest.json"), rankings_sha256=s.digest(path)))
    return output, argparse.Namespace(root=root, manifest=output / "manifest.json", workers=2, prefix=8, seconds=10, downstream=False)


class HardeningTests(unittest.TestCase):
    def test_r1_symlink_cannot_redirect_any_candidate_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, args = fixture(root)
            outside = root / "outside"
            outside.mkdir()
            (output / "evaluations").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                s.compare(args)
            self.assertEqual(list(outside.iterdir()), [])
            with self.assertRaisesRegex(ValueError, "symlink"):
                runtime.write(output / "evaluations/escaped.json", {})

    def test_production_ancestors_and_derived_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s.write(root / "manifests/campaign.json", {"schema": "parallel.campaign.v1", "output": "runs/live"})
            for path in (root, root / "runs", root / "data", root / "runs/live", root / "runs/live/nested"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    runtime.output_root(root, path)

    def test_registered_gp_stores_and_future_sibling_cuts_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s.write(root / "manifests/generated/gp.json", {"schema": "parallel.gp_jobs.v1", "jobs": [{"output": "runs/gp-reviewed3/d/cut-96"}]})
            s.write(root / "runs/gp-priority-jobs.json", {"schema": "parallel.gp_jobs.v1", "jobs": [{"output": "runs/other-live-gp/d/cut-1"}]})
            for name in ("gp-reviewed3", "other-live-gp"):
                for suffix in ("", "/d/cut-96", "/new-dataset/cut-999"):
                    with self.assertRaisesRegex(ValueError, "overlaps production"):
                        runtime.output_root(root, root / "runs" / (name + suffix))
            self.assertEqual(runtime.output_root(root, "runs/new-experiment"), root / "runs/new-experiment")

    def test_r2_sigkill_does_not_leave_a_worker_running_with_released_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = """import fcntl,os,sys,time
from pathlib import Path
import experiment_runtime as r
root=Path(sys.argv[1]); fd=os.open(root/'lease',os.O_CREAT|os.O_RDWR,0o600)
fcntl.flock(fd,fcntl.LOCK_EX); r._lease_fds=(fd,)
p=r.popen([sys.executable,'-c','import time; time.sleep(60)'])
(root/'pid').write_text(str(p.pid))
time.sleep(60)
"""
            env = dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).parent))
            parent = subprocess.Popen([sys.executable, "-c", script, str(root)], env=env)
            try:
                deadline = time.monotonic() + 5
                while not (root / "pid").exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                worker = int((root / "pid").read_text())
                parent.kill()
                parent.wait()
                with (root / "lease").open("r+") as lock:
                    while time.monotonic() < deadline:
                        state = Path(f"/proc/{worker}/stat")
                        alive = state.exists() and state.read_text().split()[2] != "Z"
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            self.assertFalse(alive, "lease released while worker still runs")
                            break
                        except BlockingIOError:
                            time.sleep(.01)
                    else:
                        self.fail("worker retained lease after controller death")
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()

    def test_r3_each_cache_field_and_foreign_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiment_scheduler.admission_reason", return_value=None), patch("experiment_scheduler.spare_lease", admitted):
            output, args = fixture(Path(tmp))
            s.compare(args)
            path = next((output / "evaluations").glob("*/result.json"))
            original = s.read(path)
            for field, value in {"rmse": .000001, "key": "x", "program_sha256": "x", "prepared_sha256": "x",
                                 "evaluator_sha256": "x", "status": "mystery", "seconds": -1,
                                 "metric_sha256": "x", "evaluation_policy_sha256": "x"}.items():
                with self.subTest(field=field):
                    s.write(path, dict(original, **{field: value}))
                    with self.assertRaises(ValueError):
                        s.compare(args)
            s.write(path, original)
            (output / "evaluations/foreign").mkdir()
            with self.assertRaisesRegex(ValueError, "unexpected"):
                s.compare(args)

    def test_r4_torn_tail_and_receipt_before_ledger_commit_do_not_refit(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiment_scheduler.admission_reason", return_value=None), patch("experiment_scheduler.spare_lease", admitted):
            output, args = fixture(Path(tmp))
            before = s.compare(args)
            ledger = runtime.journal(output / "attempts.jsonl")
            runtime.atomic_bytes(output / "attempts.jsonl", b"".join(s.compact(r)+b"\n" for r in ledger[:-1]))
            with (output / "launches.jsonl").open("ab") as stream:
                stream.write(b'{"torn":')
            after = s.compare(args)
            self.assertEqual(after["physical_attempts"], before["physical_attempts"])
            self.assertEqual(len(runtime.journal(output / "attempts.jsonl")), 8)
            self.assertEqual(len(list(output.glob("launches.jsonl.torn-*"))), 1)
            with (output / "launches.jsonl").open("ab") as stream:
                stream.write(b'{broken}\n')
            with self.assertRaisesRegex(ValueError, "corrupt complete"):
                s.compare(args)

    def test_sigkill_at_shared_durable_write_boundaries_preserves_committed_state(self):
        script = """import os,signal,sys
from pathlib import Path
import experiment_runtime as r
path=Path(sys.argv[1]); operation,boundary=sys.argv[2:]
fsync,replace,open_file=os.fsync,os.replace,os.open
syncs=0
def crash(point):
    if point==boundary: os.kill(os.getpid(),signal.SIGKILL)
def opened(*args,**kwargs):
    fd=open_file(*args,**kwargs)
    if args[1]&os.O_CREAT: crash('created')
    return fd
def synced(fd):
    global syncs
    syncs+=1
    kind='file' if syncs==1 else 'directory'
    crash('before_'+kind+'_fsync');fsync(fd);crash('after_'+kind+'_fsync')
def replaced(*args,**kwargs):
    crash('before_replace');replace(*args,**kwargs);crash('after_replace')
os.open,os.fsync,os.replace=opened,synced,replaced
if operation=='atomic': r.write(path,{'generation':2})
else: r.append(path,{'generation':2})
"""
        env = dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).parent))
        for operation in ("atomic", "journal"):
            boundaries = ["created", "before_file_fsync", "after_file_fsync",
                          "before_directory_fsync", "after_directory_fsync"]
            if operation == "atomic":
                boundaries += ["before_replace", "after_replace"]
            for boundary in boundaries:
                with self.subTest(operation=operation, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "state.json"
                    if operation == "atomic":
                        runtime.write(path, {"generation": 1})
                    else:
                        runtime.append(path, {"generation": 1})
                    process = subprocess.run([sys.executable, "-c", script, str(path), operation, boundary], env=env)
                    self.assertEqual(process.returncode, -signal.SIGKILL, "crash boundary was not reached")
                    if operation == "atomic":
                        self.assertIn(s.read(path), ({"generation": 1}, {"generation": 2}))
                        runtime.write(path, {"generation": 3})
                        self.assertEqual(s.read(path), {"generation": 3})
                    else:
                        rows = runtime.journal(path, repair=True)
                        self.assertEqual(rows[0], {"generation": 1})
                        self.assertIn(rows, ([{"generation": 1}], [{"generation": 1}, {"generation": 2}]))
                        runtime.append(path, {"generation": 3})
                        self.assertEqual(runtime.journal(path), rows + [{"generation": 3}])

    def test_invalid_orphan_metrics_are_rejected_before_another_fit(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiment_scheduler.admission_reason", return_value=None), patch("experiment_scheduler.spare_lease", admitted):
            output, args = fixture(Path(tmp))
            s.compare(args)
            path = next((output / "evaluations").glob("*/result.json"))
            path.unlink()
            launches = (output / "launches.jsonl").read_bytes()
            for value in (True, -1, "1.0", None):
                s.write(path.parent / "metrics.json", {"rmse": value})
                self.assertFalse(s.valid_metrics(path.parent / "metrics.json"))
                with self.assertRaisesRegex(ValueError, "durable evaluator metric"):
                    s.compare(args)
                self.assertEqual(launches, (output / "launches.jsonl").read_bytes())

    def test_retried_cancellation_cannot_mask_a_new_durable_metric(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiment_scheduler.admission_reason", return_value=None), patch("experiment_scheduler.spare_lease", admitted):
            output, args = fixture(Path(tmp))
            s.compare(args)
            path = next((output / "evaluations").glob("*/result.json"))
            cancelled = dict(s.read(path), status="cancelled", failure="interrupted", rmse=None, metric_sha256=None)
            s.write(path, cancelled)
            (path.parent / "metrics.json").unlink()
            rows = [cancelled if r["key"] == cancelled["key"] else r for r in runtime.journal(output / "attempts.jsonl")]
            runtime.atomic_bytes(output / "attempts.jsonl", b"".join(s.compact(r)+b"\n" for r in rows))
            write = s.write
            def interrupt_receipt(destination, value):
                if destination == path:
                    raise RuntimeError("interrupted after metric commit")
                return write(destination, value)
            with patch.object(s, "write", interrupt_receipt), self.assertRaisesRegex(RuntimeError, "after metric commit"):
                s.compare(args)
            self.assertFalse(path.exists())
            self.assertTrue(s.valid_metrics(path.parent / "metrics.json"))
            admitted_launches = (output / "launches.jsonl").read_bytes()
            report = s.compare(args)
            self.assertEqual(admitted_launches, (output / "launches.jsonl").read_bytes())
            self.assertTrue(s.read(path)["recovered_completed_artifact"])
            self.assertEqual(report["physical_fits"], 8)

    def test_r5_empty_binary_inventory_cannot_claim_eighty_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs/experiment"
            s.write(output / "aliases.json", {"families": []})
            s.write(output / "jobs.json", {"jobs": []})
            s.write(output / "contexts/index.json", {"task": "binary", "qualification": True, "encodings": []})
            s.write(output / "contexts-initial/index.json", {"encodings": []})
            s.write(output / "report.json", {"admitted_contexts": 80, "completed_contexts": 80,
                "aliases_sha256": s.digest(output / "aliases.json"), "jobs_sha256": s.digest(output / "jobs.json"),
                "context_index": "runs/experiment/contexts/index.json", "context_index_sha256": s.digest(output / "contexts/index.json")})
            with self.assertRaises(ValueError):
                binary_verify.verify(root, output)

    def test_read_only_audit_reconciles_bound_results_and_reports_missing_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiment_scheduler.admission_reason", return_value=None), patch("experiment_scheduler.spare_lease", admitted):
            root = Path(tmp)
            output, args = fixture(root)
            s.compare(args)
            before = {str(p): s.digest(p) for p in output.rglob("*") if p.is_file()}
            report = verify_experiment.audit(root, args.manifest)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["results_awaiting_journal_reconciliation"], [])
            self.assertEqual(before, {str(p): s.digest(p) for p in output.rglob("*") if p.is_file()})
            ledger = runtime.journal(output / "attempts.jsonl")
            ledger[0]["attempt_id"] = "forged"
            runtime.atomic_bytes(output / "attempts.jsonl", b"".join(s.compact(r)+b"\n" for r in ledger))
            self.assertFalse(verify_experiment.audit(root, args.manifest)["passed"])
            (root / "evaluate.py").unlink()
            self.assertEqual(verify_experiment.audit(root, args.manifest)["missing_dependencies"], ["evaluate.py"])

    def test_r6_zero_rmse_stays_in_declared_ratio_denominator(self):
        views = [{"family": "f", "arms": {arm: {"best8": {"complete": True, "best_rmse": score}}
                  for arm, score in (("larger_pool", 0), ("generic", 1))}}]
        result = s.family_comparison(views, "larger_pool", "generic", "best8", 1)
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(result["family_balanced_ratio"], 1e-12)
        views[0]["arms"]["larger_pool"]["best8"]["best_rmse"] = None
        self.assertFalse(s.family_comparison(views, "larger_pool", "generic", "best8", 1)["complete"])

    def test_semantic_environment_rejected_before_admission(self):
        for key in ("JAIN_HYPERION_NO_NORMALIZE", "JAIN_HYPERION_FULL_CANVAS"):
            with patch.dict(os.environ, {key: "1"}), self.assertRaisesRegex(ValueError, "semantic Hyperion"):
                runtime.child_env()


if __name__ == "__main__":
    unittest.main()
