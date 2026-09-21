import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qualify_resident_workers import ObservedTask, verify_refused_attempt
from resume_expired_cycle import atomic, read, sha


class RefusalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.launch = self.output / "launch-original"
        fixture = self.root / "fixture.py"
        fixture.write_text(
            """import os,sys,time
from pathlib import Path
sys.path.insert(0,TOOLS)
from resident_gpu_pool import process_identity,native_request_id
from resume_expired_cycle import atomic,sha
root=Path(sys.argv[1]);output=Path(sys.argv[sys.argv.index('--output')+1]);pool=root/'pool'
while not (root/'continue').exists():time.sleep(.01)
launch=output/'launch-original';launch.mkdir()
request=dict(root=str(root),rank=0,env={},executable_sha256=sha(Path('/proc/self/exe')),
             task=dict(kind='Train',run=str(output),launch=str(launch)))
identity=native_request_id(request);task=pool/'tasks'/identity;task.mkdir(parents=True)
(pool/'dispatches').mkdir(exist_ok=True)
atomic(pool/'dispatches'/identity,dict(request_sha256=identity,client=process_identity(os.getpid())))
atomic(task/'request.json',request)
atomic(task/'accepted.json',dict(schema='parallel.resident_task_acceptance.v2',request_sha256=identity,
    worker=dict(process=process_identity(os.getpid()),executable_sha256=request['executable_sha256'])))
atomic(launch/'controller.json',dict(passed=False,rank_failure_injected=True))
atomic(output/'resident-reconciliation-required.json',dict(launch=str(launch),attempt=0,automatic_retry_permitted=False))
raise SystemExit(7)
""".replace("TOOLS", repr(str(Path(__file__).resolve().parent)))
        )
        command = [
            sys.executable,
            str(fixture),
            str(self.root),
            "--output",
            str(self.output),
            "--fail-rank",
            "0",
        ]
        log = self.root / "process.log"
        with log.open("wb") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=stream)

            def cleanup():
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)

            self.addCleanup(cleanup)
            self.observed = ObservedTask.capture(child, command, self.root / "process-start.json")
            (self.root / "continue").touch()
            child.wait(timeout=5)
        self.assertEqual(child.returncode, 7, log.read_text())
        self.observed.finish(self.root / "process-exit.json", log)
        self.refusal = read(self.output / "resident-reconciliation-required.json")

    def verify(self):
        return verify_refused_attempt(
            self.output,
            observed=self.observed,
            root=self.root,
            pool=self.pool,
            executable=sha(Path(sys.executable)),
        )

    def test_single_original_failure_passes_only_refusal_scope(self):
        result = self.verify()
        self.assertEqual(result["status"], "observed_ambiguous_acceptance_refusal")
        self.assertFalse(result["recovery_qualified"])
        self.assertFalse(result["rank_failure_injection_qualified"])

    def test_boolean_declaration_without_observed_process_never_qualifies(self):
        with self.assertRaisesRegex(RuntimeError, "original observed process"):
            verify_refused_attempt(self.output)

    def test_original_accepted_task_and_observed_exit_are_required(self):
        acceptance = next(self.pool.glob("tasks/*/accepted.json"))
        acceptance.unlink()
        with self.assertRaisesRegex(RuntimeError, "accepted resident task"):
            self.verify()

    def test_changed_process_outcome_is_rejected(self):
        outcome = read(self.root / "process-exit.json")
        outcome["observed_exit"] = 0
        atomic(self.root / "process-exit.json", outcome)
        with self.assertRaisesRegex(RuntimeError, "outcome identity"):
            self.verify()

    def test_new_launch_cannot_hide_automatic_reexecution(self):
        (self.output / "launch-retry").mkdir()
        with self.assertRaisesRegex(RuntimeError, "new launch"):
            self.verify()

    def test_unrelated_failure_does_not_qualify_injection(self):
        atomic(self.launch / "controller.json", dict(passed=False, rank_failure_injected=False))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.verify()

    def test_stale_completion_or_wrong_original_launch_is_rejected(self):
        atomic(self.output / "complete.json", dict(success=True))
        with self.assertRaisesRegex(RuntimeError, "published completion"):
            self.verify()
        (self.output / "complete.json").unlink()
        self.refusal["launch"] = str(self.output / "another")
        atomic(self.output / "resident-reconciliation-required.json", self.refusal)
        with self.assertRaisesRegex(RuntimeError, "original launch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
