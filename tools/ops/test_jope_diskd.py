"""Hermetic tests for jope-diskd's fail-closed cleanup boundaries."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
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
            "watermarks": {"soft_percent": 85, "aggressive_percent": 90, "alert_percent": 95, "low_percent": 80},
            "deny_prefixes": [],
            "paths": [],
        }
        path = self.home / "policy.yaml"
        path.write_text(json.dumps(policy), encoding="utf-8")
        self.assertEqual(jope_diskd.read_policy(path), policy)
        policy["watermarks"]["soft_percent"] = 95
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(path)

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

    def test_remote_scratch_requires_dead_pid_and_age(self) -> None:
        root = self.home / "remote_tests"
        root.mkdir()
        scratch = root / "src.123456"
        scratch.mkdir()
        os.utime(scratch, (1, 1))
        with (
            mock.patch.object(jope_diskd, "safe_cache_root", return_value=True),
            mock.patch.object(jope_diskd, "pid_is_live", return_value=False),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            found = jope_diskd.remote_scratch_candidates(root, self.policy, 24 * 3600, now=10**12)
        self.assertEqual(found, [scratch])
        with (
            mock.patch.object(jope_diskd, "pid_is_live", return_value=True),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            self.assertEqual(jope_diskd.remote_scratch_candidates(root, self.policy, 24 * 3600, now=10**12), [])

    def test_remote_scratch_rejects_retargeted_root(self) -> None:
        root = self.home / "other"
        root.mkdir()
        scratch = root / "src.123456"
        scratch.mkdir()
        os.utime(scratch, (1, 1))
        with (
            mock.patch.object(jope_diskd, "safe_cache_root", return_value=True),
            mock.patch.object(jope_diskd, "pid_is_live", return_value=False),
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
        ):
            self.assertEqual(jope_diskd.remote_scratch_candidates(root, self.policy, 24 * 3600, now=10**12), [])

    def test_process_identity_names_are_strict(self) -> None:
        self.assertIsNotNone(jope_diskd.SCRATCH_NAME.fullmatch("src.123"))
        self.assertIsNone(jope_diskd.SCRATCH_NAME.fullmatch("src.123.backup"))
        self.assertIsNone(jope_diskd.SCRATCH_NAME.fullmatch("src.current"))

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
