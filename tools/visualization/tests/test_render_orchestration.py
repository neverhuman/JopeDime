"""Rendering orchestration admission and source-preservation checks; no GPU/SSH."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "orchestration"
SPEC = importlib.util.spec_from_file_location("jain_render_test_config", SOURCE / "common.py")
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


class RenderOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep scratch outside both product Git and /tmp, matching the repo contract.
        cache = Path.home() / ".cache" / "jain-visual-tests"
        cache.mkdir(parents=True, exist_ok=True)
        cls.scratch = tempfile.TemporaryDirectory(prefix="orchestration-", dir=cache)
        cls.root = Path(cls.scratch.name)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def clean_env(self):
        return {k: v for k, v in os.environ.items() if not k.startswith("JAIN_")}

    def test_explicit_existing_external_root(self):
        with mock.patch.dict(os.environ, self.clean_env(), clear=True):
            with self.assertRaisesRegex(ValueError, "Set JAIN_ARTIFACT_ROOT"):
                CONFIG.validate_configuration()
        with mock.patch.dict(os.environ, {"JAIN_ARTIFACT_ROOT": str(self.root)}):
            self.assertEqual(CONFIG.validate_configuration(), self.root)
            self.assertEqual(CONFIG.artifact_family("jain-intro-v2"), self.root / "jain-intro-v2")
        with mock.patch.dict(os.environ, {"JAIN_ARTIFACT_ROOT": "relative/outputs"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                CONFIG.validate_configuration()

    def test_output_root_cannot_enter_product_checkout_or_worktrees(self):
        for root in CONFIG.product_roots():
            for candidate in (root, root / "uncreated-render-output"):
                with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, "outside"):
                    CONFIG.external_path(candidate)
        link = self.root / "checkout-link"
        link.symlink_to(CONFIG.REPO, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            CONFIG.external_path(link / "render-output")

    def test_rendering_disabled_and_uhd_requires_separate_admission(self):
        env = self.clean_env() | {"JAIN_ARTIFACT_ROOT": str(self.root)}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "JAIN_ENABLE_REMOTE_RENDER"):
                CONFIG.require_render_enabled()
            os.environ["JAIN_ENABLE_REMOTE_RENDER"] = "1"
            CONFIG.require_render_enabled()
            with self.assertRaisesRegex(ValueError, "UHD is stopped"):
                CONFIG.require_render_enabled(uhd=True)
            os.environ["JAIN_ENABLE_LOCAL_RENDER"] = "1"
            with self.assertRaisesRegex(ValueError, "JAIN_BLENDER"):
                CONFIG.require_render_enabled(local=True)

    def test_historical_intro_hero_also_requires_uhd_admission(self):
        env = self.clean_env() | {"JAIN_ARTIFACT_ROOT": str(self.root), "JAIN_ENABLE_REMOTE_RENDER": "1"}
        result = subprocess.run((sys.executable, str(SOURCE / "run_intro_render.py"), "hero"),
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UHD is stopped", result.stderr)

    def test_host_aliases_reject_shell_fragments_and_duplicate_workers(self):
        for value in ("host;touch-file,other", "host,host", "host", "a,b,c"):
            with mock.patch.dict(os.environ, {"JAIN_RENDER_HOSTS": value}):
                with self.assertRaises(ValueError):
                    CONFIG.configured_hosts()

    def test_all_cli_help_without_custody_or_rendering(self):
        commands = [(sys.executable, str(path), "--help") for path in SOURCE.glob("*.py")]
        commands += [("bash", str(path), "--help") for path in SOURCE.glob("*.sh")]
        for command in commands:
            with self.subTest(script=command[1]):
                result = subprocess.run(command, env=self.clean_env(), capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_imports_and_disabled_transports_do_not_execute_subprocesses(self):
        program = """
import importlib, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected process or network operation')
subprocess.run = forbidden
subprocess.Popen = forbidden
subprocess.check_output = forbidden
for path in pathlib.Path(sys.argv[1]).glob('*.py'):
    importlib.import_module(path.stem)
for name in ('run_intro_render', 'run_dendrite_render'):
    module = importlib.import_module(name)
    try:
        module.ssh('unused-host', ':')
    except ValueError as error:
        assert 'JAIN_ENABLE_REMOTE_RENDER' in str(error), str(error)
    else:
        raise AssertionError('transport was enabled without explicit admission')
"""
        result = subprocess.run((sys.executable, "-c", program, str(SOURCE)),
                                env=self.clean_env() | {"JAIN_ARTIFACT_ROOT": str(self.root)},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nested_output_symlinks_and_job_inputs_fail_before_transport(self):
        program = """
import pathlib, subprocess, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import common, run_intro_render as intro, run_veox_render as endcap, render_uhd as uhd
root = pathlib.Path(sys.argv[2]) / 'nested'
root.mkdir()
(root / 'shards').symlink_to(common.REPO, target_is_directory=True)
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected transport/write before output validation')
with patch.object(intro, 'ROOT', root), patch.object(intro, 'stage_and_launch', forbidden):
    try:
        intro.host_render('test-host', {}, {}, 'remote', 'job', (1,), 16, 100)
    except ValueError as error:
        assert 'outside' in str(error), str(error)
    else:
        raise AssertionError('nested output symlink accepted')
with patch.object(endcap, 'ROOT', root), patch.object(endcap, 'stage', forbidden):
    try:
        endcap.render_host('test-host', 'remote', 'job', (1,), {}, {}, 96)
    except ValueError as error:
        assert 'outside' in str(error), str(error)
    else:
        raise AssertionError('nested output symlink accepted')
assert set(uhd.probes()) == {'render-a', 'render-b'}
for args in [('demo', 'demo;touch-x', (89,), 256, 'case-a'), ('demo', 'safe', (89,), 256, 'bad;id'),
             ('intro', 'safe', (331,), 256, None), ('intro', 'safe', (True,), 256, None)]:
    try:
        uhd.Job(*args)
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe job accepted')
"""
        result = subprocess.run((sys.executable, "-c", program, str(SOURCE), str(self.root)),
                                env=self.clean_env() | {"JAIN_ARTIFACT_ROOT": str(self.root),
                                                        "JAIN_RENDER_HOSTS": "render-a,render-b"},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_relative_local_runtime_is_rejected(self):
        env = self.clean_env() | {"JAIN_ARTIFACT_ROOT": str(self.root), "JAIN_ENABLE_LOCAL_RENDER": "1",
                                 "JAIN_BLENDER": "blender"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "absolute executable path"):
                CONFIG.require_render_enabled(local=True)

    def test_preserved_source_manifest_and_script_syntax(self):
        inventory = json.loads((SOURCE / "external-source-manifest.json").read_text())
        self.assertEqual(len(inventory["files"]), 22)
        for entry in inventory["files"]:
            path = SOURCE / entry["tracked_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["tracked_sha256"])
            self.assertEqual(len(entry["source_sha256"]), 64)
        for path in SOURCE.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
        for path in SOURCE.glob("*.sh"):
            result = subprocess.run(("bash", "-n", str(path)), capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
