"""Hermetic tests for jope-diskd's fail-closed cleanup boundaries."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import gzip
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import jope_diskd


class DiskPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="jope-diskd-test-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.policy = {"deny_prefixes": []}

    def cache(self) -> Path:
        root = self.home / ".cache" / "uv"
        root.mkdir(parents=True)
        return root

    def test_policy_is_json_yaml_and_watermarks_are_ordered(self) -> None:
        policy = {
            "version": 1,
            "local_only": True,
            "allow_log_rotation": True,
            "allow_log_offload": False,
            "allow_report_offload": False,
            "allowed_hosts": ["xbabe1", "xbabe2", "xbabe3"],
            "repo_root": "~/JopeDime",
            "sccache_max_bytes": 60 * jope_diskd.GIB,
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
        self.assertEqual(jope_diskd.read_policy(path), policy)
        self.assertTrue(policy["local_only"])
        self.assertEqual(policy["allowed_hosts"], ["xbabe1", "xbabe2", "xbabe3"])
        self.assertTrue(policy["allow_log_rotation"])
        self.assertFalse(policy["allow_log_offload"])
        self.assertFalse(policy["allow_report_offload"])
        policy["watermarks"]["soft_percent"] = 95
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(path)

    def test_policy_rejects_hosts_outside_the_exact_allowlist(self) -> None:
        policy = {
            "version": 1,
            "local_only": True,
            "allow_log_rotation": True,
            "allow_log_offload": False,
            "allow_report_offload": False,
            "allowed_hosts": ["xbabe1", "xbabe2", "xbabe3", "xbabe0"],
            "repo_root": "~/JopeDime",
            "sccache_max_bytes": 60 * jope_diskd.GIB,
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

    def test_process_reference_scan_checks_cargo_target_and_worktree_cwd(self) -> None:
        proc_root = self.home / "proc"
        process = proc_root / "123"
        process.mkdir(parents=True)
        worktree = self.home / "JopeDime" / ".agent" / "worktrees" / "lane"
        target = worktree / "target"
        target.mkdir(parents=True)
        (process / "cwd").symlink_to(self.home)
        (process / "environ").write_bytes(f"CARGO_TARGET_DIR={target}\0".encode())
        self.assertTrue(jope_diskd.local_process_references(target, worktree, proc_root))
        (process / "environ").write_bytes(b"PATH=/usr/bin\0")
        self.assertFalse(jope_diskd.local_process_references(target, worktree, proc_root))
        (process / "cwd").unlink()
        (process / "cwd").symlink_to(worktree)
        self.assertTrue(jope_diskd.local_process_references(target, worktree, proc_root))

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

    def test_cargo_target_candidates_require_exact_merged_pr_head(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "merged-lane"
        target = worktree / "target"
        (target / "debug" / "deps").mkdir(parents=True)
        (target / "debug" / "deps" / "libbuild.rlib").write_bytes(b"build")
        os.utime(target, (1, 1))
        calls = [
            jope_diskd.subprocess.CompletedProcess([], 0, f"worktree {worktree}\nbranch refs/heads/merged-lane\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, json.dumps([{
                "headRefName": "merged-lane", "headRefOid": "abc123", "mergedAt": "2026-10-30T00:00:00Z"
            }]), ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "merged-lane\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "abc123\n", ""),
        ]
        with mock.patch.object(jope_diskd.subprocess, "run", side_effect=calls):
            found = jope_diskd.merged_worktree_targets(repo, worktrees, 24 * 3600, now=10**12)
        self.assertEqual(found, [(worktree.resolve(), target, "merged-lane", "abc123")])

    def test_cargo_target_cleanup_skips_any_live_or_unknown_consumer(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "merged-lane"
        target = worktree / "target"
        (target / "debug" / "deps").mkdir(parents=True)
        (target / "debug" / "deps" / "libbuild.rlib").write_bytes(b"build")
        os.utime(target, (1, 1))
        entry = {"id": "cargo-targets", "path": "~/JopeDime/.agent/worktrees"}
        policy = {"local_only": True}
        calls = [
            jope_diskd.subprocess.CompletedProcess([], 0, f"worktree {worktree}\nbranch refs/heads/merged-lane\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, json.dumps([{
                "headRefName": "merged-lane", "headRefOid": "abc123", "mergedAt": "2026-10-30T00:00:00Z"
            }]), ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "merged-lane\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "abc123\n", ""),
        ]
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd.subprocess, "run", side_effect=calls),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=None),
            mock.patch.object(jope_diskd, "path_size", return_value=4096),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, policy, lambda _event: None, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(target.exists())

    def test_cargo_target_deletes_exact_merged_head_after_all_gates(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "merged-lane"
        target = worktree / "target"
        deps = target / "debug" / "deps"
        deps.mkdir(parents=True)
        artifact = deps / "libbuild.rlib"
        artifact.write_bytes(b"build")
        for path in (target, target / "debug", deps, artifact):
            os.utime(path, (1, 1))
        entry = {"id": "cargo-targets", "path": "~/JopeDime/.agent/worktrees"}
        policy = {"local_only": True}
        branch = "merged-lane"
        head = "abc123"
        calls = [
            jope_diskd.subprocess.CompletedProcess([], 0, f"worktree {worktree}\nbranch refs/heads/{branch}\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, json.dumps([{
                "headRefName": branch, "headRefOid": head, "mergedAt": "2026-10-30T00:00:00Z"
            }]), ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, f"{branch}\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, f"{head}\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, f"{branch}\n", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, f"{head}\n", ""),
        ]
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd.subprocess, "run", side_effect=calls),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
            mock.patch.object(jope_diskd, "path_size", return_value=4096),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, policy, events.append, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 4096)
        self.assertFalse(target.exists())
        self.assertEqual([event["event"] for event in events], ["delete_intent_tree", "deleted_tree"])

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


if __name__ == "__main__":
    unittest.main()
