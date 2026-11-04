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



class DiskSourceCleanupTests(unittest.TestCase):
    def test_policy_rejects_hosts_outside_the_exact_allowlist(self) -> None:
        policy = {
            "version": 1,
            "local_only": True,
            "allow_log_rotation": True,
            "allow_log_offload": False,
            "allow_report_offload": False,
            "allowed_hosts": ["xbabe1", "xbabe2", "xbabe3", "xbabe0"],
            "repo_root": "~/JopeDime",
            "sccache_max_bytes": 50 * jope_diskd.GIB,
            "offload": {
                "bucket": jope_diskd.OFFLOAD_BUCKET,
                "prefix": jope_diskd.OFFLOAD_PREFIX,
                "retention_days": 14,
                "sha256_verify_before_remove": True,
            },
            "watermarks": {"soft_percent": 85, "aggressive_percent": 90, "alert_percent": 95, "low_percent": 80},
            "deny_prefixes": [],
            "paths": [],
        }
        path = self.home / "policy.yaml"
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(path)

    def test_user_service_avoids_unsupported_kernel_module_sandbox(self) -> None:
        unit = Path(__file__).with_name("systemd").joinpath("jope-diskd.service").read_text(encoding="utf-8")
        self.assertNotIn("ProtectKernelModules", unit)
        self.assertIn("NoNewPrivileges=yes", unit)
        self.assertIn("ProtectSystem=full", unit)
        self.assertIn("ProtectKernelTunables=yes", unit)
        self.assertIn("ProtectControlGroups=yes", unit)
        self.assertIn("PrivateTmp=yes", unit)

    def test_apply_requires_explicit_local_only_environment(self) -> None:
        with mock.patch.dict(os.environ, {"JOPE_DISKD_LOCAL_ONLY": "0"}):
            self.assertEqual(jope_diskd.main(["--apply"]), 2)

    def test_state_directory_symlink_is_rejected(self) -> None:
        outside = self.home / "outside"
        outside.mkdir()
        state = self.home / ".local" / "state" / "jope-diskd"
        state.parent.mkdir(parents=True)
        state.symlink_to(outside, target_is_directory=True)
        self.assertIsNone(jope_diskd.safe_state_dir(self.home, create=True))

    def test_cache_root_rejects_outside_home_and_deny_prefix(self) -> None:
        root = self.cache()
        with mock.patch.object(jope_diskd.Path, "home", return_value=self.home):
            self.assertTrue(jope_diskd.safe_cache_root(root, self.policy))
            self.assertFalse(jope_diskd.safe_cache_root(self.home.parent, self.policy))
            denied = self.home / "protected"
            denied.mkdir()
            self.policy["deny_prefixes"] = [str(denied)]
            self.assertFalse(jope_diskd.safe_cache_root(denied, self.policy))

    def test_cache_handler_rejects_retargeted_allowlist_id(self) -> None:
        outside_cache = self.home / "important-files"
        outside_cache.mkdir()
        old = outside_cache / "old.txt"
        old.write_text("keep", encoding="utf-8")
        os.utime(old, (1, 1))
        entry = {"id": "pip-cache", "path": str(outside_cache)}
        with mock.patch.object(jope_diskd.Path, "home", return_value=self.home):
            freed = jope_diskd.handle_cache(
                entry, self.policy, 1, 0, 100, False, lambda _event: None, now=10**12
            )
        self.assertEqual(freed, 0)
        self.assertTrue(old.exists())

    def test_hard_deny_covers_protected_prefix_families(self) -> None:
        self.assertTrue(jope_diskd.protected_path(Path("/home/ubuntu/admiralty-copy/data")))
        self.assertTrue(jope_diskd.protected_path(Path("/home/ubuntu/guestd/job")))
        self.assertTrue(jope_diskd.protected_path(Path("/home/ubuntu/sealed-family")))

    def test_cache_scan_skips_symlink_without_following_it(self) -> None:
        root = self.cache()
        old = root / "old.whl"
        old.write_bytes(b"cache")
        os.utime(old, (1, 1))
        (root / "link").symlink_to(self.home, target_is_directory=True)
        self.assertEqual(jope_diskd.cache_candidates(root, 0, now=10**12), [old])

    def test_cache_scan_preserves_metadata_and_cached_git_contents(self) -> None:
        root = self.cache()
        ordinary = root / "ordinary.cache"
        ordinary.write_bytes(b"cache")
        os.utime(ordinary, (1, 1))
        for name in ("CACHEDIR.TAG", ".gitignore", ".lock"):
            marker = root / name
            marker.write_bytes(b"metadata")
            os.utime(marker, (1, 1))
        git_dir = root / "repositories" / "repo.git" / ".git"
        git_dir.mkdir(parents=True)
        git_file = git_dir / "config"
        git_file.write_bytes(b"git metadata")
        os.utime(git_file, (1, 1))
        self.assertEqual(jope_diskd.cache_candidates(root, 0, now=10**12), [ordinary])

    def test_uv_cache_policy_is_observe_only(self) -> None:
        policy = json.loads((Path(__file__).with_name("disk-policy.yaml")).read_text(encoding="utf-8"))
        uv = next(entry for entry in policy["paths"] if entry["id"] == "uv-cache")
        self.assertEqual(uv["mode"], "observe")
        self.assertNotIn("kind", uv)
        self.assertTrue(policy["allow_log_rotation"])
        self.assertFalse(policy["allow_report_offload"])

    def test_run_does_not_rotate_or_offload_logs_when_policy_disables_it(self) -> None:
        policy = {
            "version": 1,
            "local_only": True,
            "allowed_hosts": ["xbabe1", "xbabe2", "xbabe3"],
            "allow_log_rotation": False,
            "allow_log_offload": False,
            "allow_report_offload": False,
            "repo_root": str(self.home / "JopeDime"),
            "watermarks": {"soft_percent": 85, "aggressive_percent": 90, "alert_percent": 95, "low_percent": 80},
            "paths": [],
        }
        usage = mock.Mock(used=95, free=5)
        with (
            mock.patch.object(jope_diskd, "read_policy", return_value=policy),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd.socket, "gethostname", return_value="xbabe2"),
            mock.patch.object(jope_diskd.shutil, "disk_usage", return_value=usage),
            mock.patch.object(jope_diskd, "run_lock", return_value=nullcontext()),
            mock.patch.object(jope_diskd, "alert_if_needed"),
            mock.patch.object(jope_diskd, "report_only_paths"),
            mock.patch.object(jope_diskd, "audit_logger", return_value=lambda _event: None),
            mock.patch.object(jope_diskd, "rotate_daemon_log") as rotate,
            mock.patch.object(jope_diskd, "offload_jankurai_reports") as offload,
        ):
            self.assertEqual(jope_diskd.main(["--dry-run"]), 0)
        rotate.assert_not_called()
        offload.assert_not_called()

    def test_run_rotates_own_log_without_enabling_report_offload(self) -> None:
        policy = {
            "version": 1,
            "local_only": True,
            "allowed_hosts": ["xbabe1", "xbabe2", "xbabe3"],
            "allow_log_rotation": True,
            "allow_log_offload": False,
            "allow_report_offload": False,
            "repo_root": str(self.home / "JopeDime"),
            "watermarks": {"soft_percent": 85, "aggressive_percent": 90, "alert_percent": 95, "low_percent": 80},
            "paths": [],
        }
        usage = mock.Mock(used=95, free=5)
        with (
            mock.patch.object(jope_diskd, "read_policy", return_value=policy),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd.socket, "gethostname", return_value="xbabe1"),
            mock.patch.object(jope_diskd.shutil, "disk_usage", return_value=usage),
            mock.patch.object(jope_diskd, "run_lock", return_value=nullcontext()),
            mock.patch.object(jope_diskd, "alert_if_needed"),
            mock.patch.object(jope_diskd, "report_only_paths"),
            mock.patch.object(jope_diskd, "audit_logger", return_value=lambda _event: None),
            mock.patch.object(jope_diskd, "rotate_daemon_log") as rotate,
            mock.patch.object(jope_diskd, "offload_jankurai_reports") as offload,
        ):
            self.assertEqual(jope_diskd.main(["--dry-run"]), 0)
        rotate.assert_called_once()
        self.assertFalse(rotate.call_args.kwargs["offload_archives"])
        offload.assert_not_called()

    def test_dry_run_reports_without_removing_or_auditing(self) -> None:
        root = self.cache()
        old = root / "old.whl"
        old.write_bytes(b"cache")
        os.utime(old, (1, 1))
        entry = {"id": "uv-cache", "path": str(root)}
        with (
            mock.patch.object(jope_diskd, "safe_cache_root", return_value=True),
            mock.patch.object(jope_diskd, "path_size", return_value=100),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            events: list[dict[str, object]] = []
            freed = jope_diskd.handle_cache(entry, self.policy, 1, 0, 100, True, events.append, now=10**12)
        self.assertEqual(freed, jope_diskd.allocated_bytes(old.stat()))
        self.assertTrue(old.exists())
        self.assertEqual(events, [])

    def test_apply_audits_each_deletion_before_and_after(self) -> None:
        root = self.cache()
        old = root / "old.whl"
        old.write_bytes(b"cache")
        os.utime(old, (1, 1))
        entry = {"id": "uv-cache", "path": str(root)}
        with (
            mock.patch.object(jope_diskd, "safe_cache_root", return_value=True),
            mock.patch.object(jope_diskd, "path_size", return_value=100),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            events: list[dict[str, object]] = []
            freed = jope_diskd.handle_cache(entry, self.policy, 1, 0, 100, False, events.append, now=10**12)
        self.assertGreater(freed, 0)
        self.assertFalse(old.exists())
        self.assertEqual([e["event"] for e in events], ["delete_intent", "deleted"])

    def test_cache_with_unknown_open_handle_is_retained(self) -> None:
        root = self.cache()
        old = root / "old.whl"
        old.write_bytes(b"cache")
        os.utime(old, (1, 1))
        entry = {"id": "uv-cache", "path": str(root)}
        with (
            mock.patch.object(jope_diskd, "safe_cache_root", return_value=True),
            mock.patch.object(jope_diskd, "path_size", return_value=100),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=None),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            freed = jope_diskd.handle_cache(entry, self.policy, 1, 0, 100, False, lambda _event: None, now=10**12)
        self.assertEqual(freed, 0)
        self.assertTrue(old.exists())

    def test_current_process_open_handle_is_detected_on_proc_hosts(self) -> None:
        if not Path("/proc/self/fd").is_dir():
            self.skipTest("Linux procfs check")
        root = self.cache()
        item = root / "in-use.whl"
        item.write_bytes(b"cache")
        with item.open("rb"):
            opened = jope_diskd.open_handles_under(root)
        self.assertIsNotNone(opened)
        self.assertIn(item.resolve(), opened)

    def test_lsof_fallback_is_scoped_to_the_janitor_uid(self) -> None:
        root = self.cache()
        completed = jope_diskd.subprocess.CompletedProcess([], 1, "", "")
        with (
            mock.patch.object(jope_diskd.shutil, "which", return_value="/usr/bin/lsof"),
            mock.patch.object(jope_diskd.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(jope_diskd.lsof_open_paths(root.resolve(), user_id=jope_diskd.current_uid()), set())
        command = run.call_args.args[0]
        self.assertIn("-a", command)
        self.assertIn("-u", command)
        self.assertIn(str(jope_diskd.current_uid()), command)

    def test_dry_run_does_not_remove_empty_directories(self) -> None:
        root = self.cache()
        empty = root / "empty"
        empty.mkdir()
        jope_diskd.remove_empty_dirs(root, True, lambda _event: None)
        self.assertTrue(empty.exists())

    def test_empty_directory_cleanup_does_not_enter_cached_git(self) -> None:
        root = self.cache()
        repository = root / "repositories" / "repo"
        git_dir = repository / ".git"
        git_dir.mkdir(parents=True)
        empty_subdir = git_dir / "objects"
        empty_subdir.mkdir()
        jope_diskd.remove_empty_dirs(root, False, lambda _event: None)
        self.assertTrue(git_dir.is_dir())
        self.assertTrue(empty_subdir.is_dir())

    def test_remote_tests_are_observe_only_without_cleanup_kind(self) -> None:
        policy = json.loads((Path(__file__).with_name("disk-policy.yaml")).read_text(encoding="utf-8"))
        remote = next(entry for entry in policy["paths"] if entry["id"] == "remote-tests")
        self.assertEqual(remote["mode"], "observe")
        self.assertTrue(remote["manual_only"])
        self.assertNotIn("kind", remote)
        self.assertFalse(hasattr(jope_diskd, "remote_scratch_candidates"))

    def test_disk_policy_uses_two_hour_target_ttl_and_worktree_idle_gate(self) -> None:
        policy = json.loads((Path(__file__).with_name("disk-policy.yaml")).read_text(encoding="utf-8"))
        cargo = next(entry for entry in policy["paths"] if entry["id"] == "cargo-targets")
        self.assertEqual(cargo["ttl_hours"], 2)
        self.assertEqual(cargo["worktree_idle_hours"], 2)

    def test_process_reference_scan_checks_cargo_target_and_worktree_cwd(self) -> None:
        proc_root = self.home / "proc"
        process = proc_root / "123"
        process.mkdir(parents=True)
        worktree = self.home / "JopeDime" / ".agent" / "worktrees" / "lane"
        target = worktree / "target"
        target.mkdir(parents=True)
        uid = jope_diskd.current_uid()
        (process / "status").write_text(f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n", encoding="utf-8")
        (process / "cwd").symlink_to(self.home)
        (process / "environ").write_bytes(f"CARGO_TARGET_DIR={target}\0".encode())
        self.assertTrue(jope_diskd.local_process_references(target, worktree, proc_root))
        (process / "environ").write_bytes(b"PATH=/usr/bin\0")
        self.assertFalse(jope_diskd.local_process_references(target, worktree, proc_root))
        (process / "cwd").unlink()
        (process / "cwd").symlink_to(worktree)
        self.assertTrue(jope_diskd.local_process_references(target, worktree, proc_root))

    def test_process_reference_scan_skips_other_users_before_reading_process_data(self) -> None:
        proc_root = self.home / "proc"
        process = proc_root / "456"
        process.mkdir(parents=True)
        worktree = self.home / "JopeDime" / ".agent" / "worktrees" / "lane"
        target = worktree / "target"
        target.mkdir(parents=True)
        uid = jope_diskd.current_uid()
        (process / "status").write_text(f"Uid:\t{uid}\t1234\t1234\t1234\n", encoding="utf-8")
        self.assertFalse(jope_diskd.local_process_references(target, worktree, proc_root))

    def test_worktree_idle_gate_requires_old_index_and_head(self) -> None:
        worktree = self.home / "JopeDime" / ".agent" / "worktrees" / "lane"
        git_dir = self.home / "gitdir"
        git_dir.mkdir()
        head = git_dir / "HEAD"
        index = git_dir / "index"
        head.touch()
        index.touch()
        with mock.patch.object(
            jope_diskd.subprocess,
            "run",
            return_value=jope_diskd.subprocess.CompletedProcess([], 0, f"{git_dir}\n", ""),
        ):
            self.assertFalse(jope_diskd.worktree_idle_for_cleanup(worktree, now=1000, idle_seconds=7200))
            os.utime(head, (1, 1))
            os.utime(index, (1, 1))
            self.assertTrue(jope_diskd.worktree_idle_for_cleanup(worktree, now=10_000, idle_seconds=7200))

    def test_target_evidence_families_are_not_recursively_deleted(self) -> None:
        target = self.home / "target"
        (target / "debug" / "deps").mkdir(parents=True)
        (target / "debug" / "deps" / "libbuild.rlib").write_bytes(b"build")
        for path in (target, target / "debug", target / "debug" / "deps", target / "debug" / "deps" / "libbuild.rlib"):
            os.utime(path, (1, 1))
        self.assertTrue(jope_diskd.safe_target_tree(target, now=10**12))
        report = target / "jankurai" / "reports" / "receipt.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}", encoding="utf-8")
        for path in (target / "jankurai", report.parent, report):
            os.utime(path, (1, 1))
        self.assertFalse(jope_diskd.safe_target_tree(target, now=10**12))

    def test_target_model_and_dataset_artifacts_are_not_rebuildable_caches(self) -> None:
        for name in (
            "model.pt", "model.pth", "checkpoint.ckpt", "weights.safetensors",
            "dataset.bin", "array.npy", "model.h5", "training.pkl", "model.onnx",
            "rows.tfrecord",
        ):
            with self.subTest(name=name):
                target = self.home / name / "target"
                artifact = target / "debug" / "deps" / name
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"synthetic protected artifact")
                for path in (artifact, artifact.parent, artifact.parent.parent, target):
                    os.utime(path, (1, 1))
                self.assertFalse(jope_diskd.safe_target_tree(target, now=10**12))
