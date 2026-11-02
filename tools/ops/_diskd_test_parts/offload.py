"""Hermetic tests for jope-diskd's fail-closed cleanup boundaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import gzip
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import jope_diskd



class DiskOffloadTests(unittest.TestCase):
    def test_cargo_target_cleanup_preserves_checkpoint_artifact(self) -> None:
        repo = self.home / "JopeDime"
        worktree = repo / ".agent" / "worktrees" / "merged-lane"
        target = worktree / "target"
        checkpoint = target / "debug" / "deps" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        for path in (checkpoint, checkpoint.parent, checkpoint.parent.parent, target):
            os.utime(path, (1, 1))
        candidate = {
            "worktree": worktree,
            "target": target,
            "target_exists": True,
            "stale_target": True,
            "branch": "merged-lane",
            "head": "abc123",
            "disposition": "merged",
            "clean": True,
            "ignored_safe": True,
        }
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", return_value=self.home / ".agent" / "worktrees"),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
            mock.patch.object(jope_diskd.shutil, "rmtree") as remove_tree,
        ):
            freed = jope_diskd.handle_cargo_targets(
                {"id": "cargo-targets", "path": "~/JopeDime/.agent/worktrees", "remove_merged_worktrees": True},
                repo,
                {"local_only": True},
                lambda _event: None,
                dry_run=False,
                now=10**12,
                needed_bytes=1,
            )
        self.assertEqual(freed, 0)
        self.assertTrue(checkpoint.is_file())
        remove_tree.assert_not_called()

    def test_stale_target_in_open_pr_worktree_is_removed_without_removing_source(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "open-lane"
        target = worktree / "target"
        deps = target / "debug" / "deps"
        deps.mkdir(parents=True)
        artifact = deps / "libbuild.rlib"
        artifact.write_bytes(b"build")
        for path in (target, target / "debug", deps, artifact):
            os.utime(path, (1, 1))
        candidate = {
            "worktree": worktree,
            "target": target,
            "target_exists": True,
            "stale_target": True,
            "branch": "open-lane",
            "head": "abc123",
            "disposition": None,
            "clean": True,
        }
        entry = {"id": "cargo-targets", "path": "~/JopeDime/.agent/worktrees", "ttl_hours": 24,
                 "remove_stale_targets_on_open_worktrees": True}
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
            mock.patch.object(jope_diskd, "worktree_head_is_unchanged", return_value=True),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=True),
            mock.patch.object(jope_diskd, "path_size", return_value=4096),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, {"local_only": True}, lambda _event: None, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 4096)
        self.assertFalse(target.exists())
        self.assertTrue(worktree.is_dir())

    def test_stale_target_in_dirty_worktree_is_retained(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "dirty-lane"
        target = worktree / "target"
        deps = target / "debug" / "deps"
        deps.mkdir(parents=True)
        artifact = deps / "libbuild.rlib"
        artifact.write_bytes(b"build")
        for path in (target, target / "debug", deps, artifact):
            os.utime(path, (1, 1))
        candidate = {
            "worktree": worktree,
            "target": target,
            "target_exists": True,
            "stale_target": True,
            "branch": "dirty-lane",
            "head": "abc123",
            "disposition": None,
            "clean": False,
        }
        entry = {
            "id": "cargo-targets",
            "path": "~/JopeDime/.agent/worktrees",
            "ttl_hours": 24,
            "remove_stale_targets_on_open_worktrees": True,
        }
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, {"local_only": True}, lambda _event: None, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(artifact.exists())

    def test_merged_worktree_cleanup_preserves_evidence_target(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "merged-lane"
        target = worktree / "target"
        report = target / "debug" / "reports" / "receipt.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}", encoding="utf-8")
        for path in (target, target / "debug", target / "debug" / "reports", report):
            os.utime(path, (1, 1))
        candidate = {
            "worktree": worktree,
            "target": target,
            "target_exists": True,
            "stale_target": True,
            "branch": "merged-lane",
            "head": "abc123",
            "disposition": "merged",
            "clean": True,
        }
        entry = {"id": "cargo-targets", "path": "~/JopeDime/.agent/worktrees", "ttl_hours": 24,
                 "remove_merged_worktrees": True}
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, {"local_only": True}, lambda _event: None, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(report.exists())
        self.assertTrue(worktree.exists())

    def test_s3_key_is_host_day_hash_scoped(self) -> None:
        root = self.home / "reports"
        root.mkdir()
        path = root / "summary.json"
        path.write_text("{}", encoding="utf-8")
        key = jope_diskd.s3_key(path, root, "xbabe2", jope_diskd.dt.date(2026, 10, 31), "abc123")
        self.assertEqual(key, "offload/xbabe2/20261031/abc123/summary.json")

    def test_offload_removes_only_after_upload_verification(self) -> None:
        state = self.home / ".local" / "state" / "jope-diskd"
        state.mkdir(parents=True)
        path = state / "old.log.gz"
        path.write_bytes(b"verified log")
        os.utime(path, (1, 1))
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "upload_and_verify", return_value=False),
        ):
            self.assertFalse(jope_diskd.offload_file(path, state, events.append, dry_run=False, now=10**12))
            self.assertTrue(path.exists())
            self.assertEqual(events[-1]["event"], "offload_verify_failed")
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "upload_and_verify", return_value=True),
        ):
            self.assertTrue(jope_diskd.offload_file(path, state, events.append, dry_run=False, now=10**12))
            self.assertFalse(path.exists())
            self.assertEqual(events[-1]["event"], "offload_local_removed")

    def test_jankurai_report_tree_offloads_old_files_and_removes_empty_dirs(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktrees.mkdir(parents=True)
        root = repo / "target" / "jankurai" / "reports"
        nested = root / "old" / "run"
        nested.mkdir(parents=True)
        report = nested / "summary.csv"
        report.write_text("score\n1\n", encoding="utf-8")
        for path in (repo / "target", repo / "target" / "jankurai", root, root / "old", nested, report):
            os.utime(path, (1, 1))
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "upload_and_verify", return_value=True),
        ):
            jope_diskd.offload_jankurai_reports(repo, worktrees, events.append, dry_run=False, now=10**12)
        self.assertFalse(report.exists())
        self.assertFalse(root.exists())
        self.assertEqual(events[-1]["event"], "offload_empty_dir_removed")

    def test_log_rotation_compresses_and_verifies_before_removal(self) -> None:
        state = self.home / ".local" / "state" / "jope-diskd"
        state.mkdir(parents=True)
        active = state / "jope-diskd.log"
        active.write_bytes(b"long enough to rotate")
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd, "LOG_LIMIT_BYTES", 4),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            jope_diskd.rotate_daemon_log(self.home, events.append, dry_run=False, now=jope_diskd.time.time())
        self.assertFalse(active.exists())
        archives = list(state.glob("jope-diskd-*.log.gz"))
        self.assertEqual(len(archives), 1)
        with gzip.open(archives[0], "rb") as stream:
            self.assertEqual(stream.read(), b"long enough to rotate")
        self.assertEqual(events[0]["event"], "log_rotated")

    def test_log_rotation_preserves_existing_archive_paths_and_link_targets(
        self,
    ) -> None:
        fixed = jope_diskd.dt.datetime(2020, 1, 1, tzinfo=jope_diskd.dt.timezone.utc)
        for kind in ("regular", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                home = self.home / kind
                state = home / ".local" / "state" / "jope-diskd"
                state.mkdir(parents=True)
                active = state / "jope-diskd.log"
                active.write_bytes(b"new daemon log that must remain available")
                archive = state / "jope-diskd-20200101T000000Z.log.gz"
                victim = home / "unrelated.bin"
                victim.write_bytes(b"unrelated content must survive")
                if kind == "regular":
                    with gzip.open(archive, "wb") as stream:
                        stream.write(b"older archived log must survive")
                elif kind == "symlink":
                    archive.symlink_to(victim)
                else:
                    os.link(victim, archive)
                archive_before = archive.read_bytes()
                victim_before = victim.read_bytes()
                active_before = active.read_bytes()
                events = []
                with (
                    mock.patch.object(jope_diskd, "LOG_LIMIT_BYTES", 1),
                    mock.patch.object(jope_diskd, "utc_now", return_value=fixed),
                    mock.patch.object(
                        jope_diskd, "open_handles_under", return_value=set()
                    ),
                ):
                    jope_diskd.rotate_daemon_log(
                        home, events.append, dry_run=False, now=fixed.timestamp()
                    )
                self.assertEqual(archive.read_bytes(), archive_before)
                self.assertEqual(victim.read_bytes(), victim_before)
                self.assertEqual(active.read_bytes(), active_before)
                self.assertEqual(archive.is_symlink(), kind == "symlink")
                self.assertEqual(events, [])

    def test_log_rotation_does_not_offload_archives_by_default(self) -> None:
        state = self.home / ".local" / "state" / "jope-diskd"
        state.mkdir(parents=True)
        archive = state / "jope-diskd-20200101T000000Z.log.gz"
        with gzip.open(archive, "wb") as stream:
            stream.write(b"retain locally")
        os.utime(archive, (1, 1))
        with mock.patch.object(jope_diskd, "offload_file") as offload:
            jope_diskd.rotate_daemon_log(self.home, lambda _event: None, dry_run=False, now=10**12)
        offload.assert_not_called()
        self.assertTrue(archive.exists())

    def test_alert_is_preview_only_then_idempotent_per_host_day(self) -> None:
        policy = {"watermarks": {"alert_percent": 95}}
        jope_diskd.alert_if_needed(self.home, 96, policy, True)
        alert = self.home / "nudges" / "DISK_ALERT.md"
        self.assertFalse(alert.exists())
        jope_diskd.alert_if_needed(self.home, 96, policy, False)
        first = alert.read_text(encoding="utf-8")
        jope_diskd.alert_if_needed(self.home, 98, policy, False)
        self.assertEqual(alert.read_text(encoding="utf-8"), first)
        self.assertEqual(first.count("disk usage"), 1)
