"""Pinned Git source test groups for test_plan_publisher."""

from pathlib import Path
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import plan_publisher as publisher

ROOT = Path(__file__).resolve().parents[4]

class PinnedModuleLoaderTests(unittest.TestCase):
    """Real private Git inputs; no source authorization or import mock in positives."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="q1-loader-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.state = self.root / "publisher"
        self.state.mkdir()
        self.env = dict(
            os.environ,
            GIT_AUTHOR_NAME="Loader fixture",
            GIT_COMMITTER_NAME="Loader fixture",
            GIT_AUTHOR_EMAIL="loader@example.invalid",
            GIT_COMMITTER_EMAIL="loader@example.invalid",
        )
        self.path = "deploy/qualification/q1/harness/q1_cut.py"
        self.data = b"VALUE = 'pinned Git source'\n"
        (self.repo / self.path).parent.mkdir(parents=True)
        (self.repo / self.path).write_bytes(self.data)
        self.git("init", "-q")
        self.source = self.commit()
        self.frozen = self.source
        self.git("update-ref", "refs/remotes/origin/main", self.source)
        self.pins = dict(publisher.MODULE_PINS)
        self.pins["q1_cut"] = (self.path, publisher.sha(self.data), False)
        for name, value in [
            ("REPO", self.repo),
            ("E", self.state),
            ("FROZEN_SOURCE", self.frozen),
            ("MODULE_PINS", self.pins),
        ]:
            item = patch.object(publisher, name, value)
            item.start()
            self.addCleanup(item.stop)
        present, previous = "q1_cut" in sys.modules, sys.modules.get("q1_cut")

        def restore():
            if present:
                sys.modules["q1_cut"] = previous
            else:
                sys.modules.pop("q1_cut", None)

        self.addCleanup(restore)

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args],
            env=self.env,
            stderr=subprocess.PIPE,
            text=True,
        ).strip()

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "private module fixture")
        return self.git("rev-parse", "HEAD")

    def filename(self):
        return (
            self.state
            / "frozen"
            / self.frozen
            / "metadata"
            / self.source
            / "validation-inputs/q1_cut.py"
        )

    def test_git_pin_wins_over_filesystem_source_and_bytecode(self):
        filename = self.filename()
        filename.parent.mkdir(parents=True)
        filename.write_text("raise AssertionError('filesystem source was executed')\n")
        bytecode = Path(importlib.util.cache_from_source(str(filename)))
        bytecode.parent.mkdir()
        bytecode.write_bytes(b"untrusted bytecode must never be read")
        value = publisher.module("q1_cut", self.source, filename)
        self.assertEqual(value.VALUE, "pinned Git source")
        self.assertEqual(Path(value.__file__), filename)
        self.assertEqual(
            bytecode.read_bytes(), b"untrusted bytecode must never be read"
        )
        loader = publisher._PinnedGitLoader(
            "q1_cut", filename, self.data, publisher.sha(self.data)
        )
        with self.assertRaises(OSError):
            loader.get_data(str(bytecode))

    def test_unreviewed_commit_refused_even_with_identical_code(self):
        (self.repo / "unreviewed.txt").write_text("unmerged branch\n")
        self.source = self.commit()
        with self.assertRaisesRegex(ValueError, "reviewed main ancestry"):
            publisher.module("q1_cut", self.source, self.filename())

    def test_changed_git_bytes_refused_before_import(self):
        (self.repo / self.path).write_text("VALUE = 'unreviewed drift'\n")
        self.source = self.commit()
        self.git("update-ref", "refs/remotes/origin/main", self.source)
        with patch.object(publisher._PinnedGitLoader, "exec_module") as imported:
            with self.assertRaisesRegex(ValueError, "Git bytes differ"):
                publisher.module("q1_cut", self.source, self.filename())
            imported.assert_not_called()

    def test_unknown_names_escaped_paths_and_symlinks_are_refused(self):
        with self.assertRaisesRegex(ValueError, "module name"):
            publisher.module("arbitrary", self.source, self.filename())
        escaped = self.filename().parent / os.pardir / self.filename().name
        self.assertIn(os.pardir, escaped.parts)
        for filename in [
            Path("q1_cut.py"),
            self.root / "outside.py",
            escaped,
            self.filename().with_name("other.py"),
        ]:
            with self.subTest(filename=str(filename)), self.assertRaises(ValueError):
                publisher.module("q1_cut", self.source, filename)
        filename = self.filename()
        filename.parent.parent.mkdir(parents=True)
        target = self.root / "foreign"
        target.mkdir()
        filename.parent.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "unsafe module filename"):
            publisher.module("q1_cut", self.source, filename)

    def test_full_source_sha_and_frozen_source_are_required(self):
        for source in [b"VALUE = 'arbitrary code'", "main", "a" * 39, "g" * 40, None]:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "full module source"),
            ):
                publisher.module("q1_cut", source, self.filename())
        (self.repo / "later.txt").write_text("later source\n")
        later = self.commit()
        with self.assertRaisesRegex(ValueError, "frozen compiled source"):
            publisher._authorize_module_source(later, True)

    def test_failed_import_restores_existing_and_absent_module_state(self):
        code = b"raise RuntimeError('deliberate pinned fixture failure')\n"
        (self.repo / self.path).write_bytes(code)
        self.source = self.commit()
        self.git("update-ref", "refs/remotes/origin/main", self.source)
        self.pins["q1_cut"] = (self.path, publisher.sha(code), False)
        previous = object()
        sys.modules["q1_cut"] = previous
        with self.assertRaisesRegex(RuntimeError, "deliberate pinned fixture"):
            publisher.module("q1_cut", self.source, self.filename())
        self.assertIs(sys.modules["q1_cut"], previous)
        sys.modules.pop("q1_cut")
        with self.assertRaisesRegex(RuntimeError, "deliberate pinned fixture"):
            publisher.module("q1_cut", self.source, self.filename())
        self.assertNotIn("q1_cut", sys.modules)


class ReviewedModulePins(unittest.TestCase):
    # The signed Q1 plan's metadata source; re-signing at a newer source needs a reviewed re-pin.
    SIGNED_METADATA_SOURCE = "fa07e1336c8fc7c04994faadf8f6efabded0a195"

    def test_pins_equal_reviewed_git_bytes(self):
        for name, (path, expected, frozen) in publisher.MODULE_PINS.items():
            source = publisher.FROZEN_SOURCE if frozen else self.SIGNED_METADATA_SOURCE
            with self.subTest(name=name):
                shown = subprocess.run(
                    ["git", "-C", str(ROOT), "show", f"{source}:{path}"],
                    capture_output=True,
                )
                if shown.returncode != 0:
                    self.skipTest(f"{source} is not in local Git history")
                self.assertEqual(hashlib.sha256(shown.stdout).hexdigest(), expected)
