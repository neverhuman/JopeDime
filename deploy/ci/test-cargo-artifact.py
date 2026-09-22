#!/usr/bin/env python3
"""Artifact protocol regressions; no Rust compilation or external dependencies."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HELPER = Path(__file__).with_name("cargo-artifact.py").resolve()
spec = importlib.util.spec_from_file_location("cargo_artifact", HELPER)
artifact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifact)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        # Tests create only disposable fixtures, not a Cargo target directory.
        self.temp = tempfile.TemporaryDirectory(prefix="artifact protocol ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "external target with spaces" / "program"
        self.binary.parent.mkdir()
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        self.meta = {"packages": [{"name": "demo", "id": "path+file:///demo#0.1.0"}],
                     "workspace_members": ["path+file:///demo#0.1.0"],
                     "target_directory": str(self.binary.parent)}
        self.message = {"reason": "compiler-artifact", "package_id": self.meta["workspace_members"][0],
                        "target": {"kind": ["bin"], "name": "demo"},
                        "profile": {"test": False}, "executable": str(self.binary), "fresh": True}
        self.finished = {"reason": "build-finished", "success": True}

    def select(self, messages=None, **kwargs):
        text = "\n".join(json.dumps(m) for m in
                         (messages if messages is not None else [self.message, self.finished]))
        return artifact.executable(text, self.meta, "demo", "bin", "demo", **kwargs)

    def test_cached_and_new_builds_with_spaces(self):
        for fresh in (True, False):
            self.message["fresh"] = fresh
            self.assertEqual(self.select(), self.binary)

    def test_rejects_missing_target_and_nonmatching_package_kind_or_name(self):
        for field, value in [("package_id", "dependency"),
                             ("target", {"kind": ["example"], "name": "demo"}),
                             ("target", {"kind": ["bin"], "name": "other"}),
                             ("profile", {"test": True}), ("executable", None)]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, "found 0"):
                    self.select([{**self.message, field: value}, self.finished])

    def test_rejects_ambiguous_artifacts(self):
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.select([self.message, {**self.message, "executable": str(self.root / "other")}, self.finished])

    def test_rejects_nonexecutables_and_missing_files(self):
        self.binary.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "not executable"):
            self.select()
        self.binary.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.select()

    def test_rejects_failed_or_unfinished_stream(self):
        for messages in ([self.message], [self.message, {**self.finished, "success": False}]):
            with self.assertRaisesRegex(ValueError, "successful completed build"):
                self.select(messages)

    def test_translates_container_mount(self):
        self.message["executable"] = "/work/external target with spaces/program"
        self.assertEqual(self.select(path_map=("/work", str(self.root))), self.binary)
        with self.assertRaises(ValueError):
            self.select(path_map=("/other", str(self.root)))

    def cli(self, *args, exit_status=0):
        fake = self.root / "cargo"
        fake.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
root = pathlib.Path(__file__).parent
with (root/'commands.jsonl').open('a') as f:
    f.write(json.dumps({'argv':sys.argv[1:], 'target':os.environ.get('CARGO_TARGET_DIR'),
                       'incremental':os.environ.get('CARGO_INCREMENTAL'),
                       'debug':os.environ.get('CARGO_PROFILE_DEV_DEBUG')})+'\\n')
if sys.argv[1] == 'metadata':
    print((root/'metadata.json').read_text())
else:
    print((root/'messages.jsonl').read_text())
    sys.exit(int(os.environ['FAKE_BUILD_EXIT']))
""")
        fake.chmod(0o755)
        (self.root / "metadata.json").write_text(json.dumps(self.meta))
        (self.root / "messages.jsonl").write_text("\n".join(map(json.dumps, [self.message, self.finished])))
        env = {**os.environ, "PATH": str(self.root)+os.pathsep+os.environ["PATH"],
               "FAKE_BUILD_EXIT": str(exit_status)}
        env.pop("CARGO_INCREMENTAL", None)
        env.pop("CARGO_PROFILE_DEV_DEBUG", None)
        return subprocess.run([sys.executable, str(HELPER), *args], env=env,
                              text=True, capture_output=True)

    def test_late_build_failure_preserves_exit_and_prints_no_executable(self):
        result = self.cli("build", "--package", "demo", "--bin", "demo", exit_status=101)
        self.assertEqual(result.returncode, 101, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_cli_returns_only_path_and_passes_profile_features_and_target(self):
        result = self.cli("build", "--package", "demo", "--bin", "demo", "--profile", "release",
                          "--features", "a,b", "--features", "c", "--target-dir", str(self.binary.parent))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(self.binary)+"\n")
        commands = [json.loads(line) for line in (self.root/"commands.jsonl").read_text().splitlines()]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command["target"], str(self.binary.parent))
            self.assertEqual(command["incremental"], "0")
            self.assertEqual(command["debug"], "line-tables-only")
        argv = commands[-1]["argv"]
        self.assertEqual(argv[argv.index("--profile")+1], "release")
        self.assertEqual(argv[-4:], ["--features", "a,b", "--features", "c"])

    def test_effective_target_directory_comes_from_metadata(self):
        result = self.cli("target-dir")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(self.binary.parent)+"\n")


if __name__ == "__main__":
    unittest.main()
