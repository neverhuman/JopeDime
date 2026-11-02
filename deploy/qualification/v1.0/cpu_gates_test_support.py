"""CPU qualification controller proofs; no Rust build, GPU, release or dataset jobs."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "cpu_gates_under_test", HERE / "cpu_gates.py"
)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)

ONE_TEST = (
    "import os,unittest; "
    "assert os.environ['CUDA_VISIBLE_DEVICES']==''; "
    "assert 'AWS_ACCESS_KEY_ID' not in os.environ and 'GITHUB_TOKEN' not in os.environ; "
    "suite=unittest.TestSuite([unittest.FunctionTestCase(lambda:None)]); "
    "r=unittest.TextTestRunner().run(suite); raise SystemExit(not r.wasSuccessful())"
)


class ControllerBase(unittest.TestCase):
    def setUp(self):
        base = Path(
            os.environ.get(
                "JOPE_QUALIFICATION_TEST_TMP",
                ROOT / "target/qualification-controller-tests",
            )
        )
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.repo = self.path / "repo"
        self.repo.mkdir()
        self.controls = mock.patch.object(cpu, "CONTROLS", {})
        self.controls.start()
        self.addCleanup(self.controls.stop)
    def plan_metadata(self, status=True):
        root = self.path / "master"
        (root / "source").mkdir(parents=True)
        (root / "tasks").mkdir()
        (root / "status").mkdir()
        (root / "master_plan.md").write_text(
            "All 28 release gates + Q1\n"
            "| P021 | [TEL-01](tasks/T021-TEL-01-cpu.md) | CPU math | done merged #1 |\n"
        )
        (root / "source/spec-v1.md").write_text(
            "## 14. Release gates (28) and Q1\n"
            + "\n".join(
                f"| **{gate}** | Gate {gate} | Exact signed bytes |"
                for gate in cpu.GATE_IDS[:-1]
            )
            + "\n**Capability gate Q1 — learned guidance.** Q1 chooses the default.\n"
        )
        (root / "tasks/T021-TEL-01-cpu.md").write_text("| Gates closed | T1 |\n")
        if status:
            (root / "status/TEL-01.md").write_text(
                "# Historical PASS on a different source\n"
            )
        return root
    def source(self, commands):
        (self.repo / "deploy/ci").mkdir(parents=True)
        (self.repo / ".gitignore").write_text("__pycache__/\n")
        lanes = {}
        for name, command, backend in commands:
            lanes[name] = dict(
                command=command,
                result_backend=backend,
                requires_nonzero_tests=backend != "none",
            )
        for name in cpu.INPUTS:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (self.repo / "deploy/ci/coverage.json").write_text(
            json.dumps(dict(lanes=lanes))
        )
        (self.repo / "deploy/ci/full-main.py").write_text(
            "GROUPS = " + repr({"cpu": tuple(lanes)}) + "\n"
        )
        (self.repo / "deploy/ci/results.py").write_bytes(
            (ROOT / "deploy/ci/results.py").read_bytes()
        )
        subprocess.run(
            ["git", "init", "-q", "--initial-branch=main", str(self.repo)], check=True
        )
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        head = cpu.git(self.repo, "rev-parse", "HEAD")
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "update-ref",
                "refs/remotes/origin/main",
                head,
            ],
            check=True,
        )
        return head
    def run_checks(self, **kwargs):
        with mock.patch.object(
            cpu,
            "admission",
            return_value=dict(cpu_count=8, load1=0, disk_used_percent=1),
        ):
            # execute_gate's default is bound at definition, so inject admission at the call seam.
            original = cpu.execute_gate

            def gate(*args, **options):
                return original(
                    *args,
                    **options,
                    check=lambda _: dict(cpu_count=8, load1=0, disk_used_percent=1),
                )

            with mock.patch.object(cpu, "execute_gate", side_effect=gate):
                return cpu.run(
                    self.repo, self.path / "evidence", execute=True, **kwargs
                )
