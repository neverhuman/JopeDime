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

    def test_disk_policy_sets_target_cap_and_exact_source_tar_prune(self) -> None:
        policy = json.loads(
            (Path(__file__).with_name("disk-policy.yaml")).read_text(encoding="utf-8")
        )
        cargo = next(
            entry for entry in policy["paths"] if entry["id"] == "cargo-targets"
        )
        source_tars = next(
            entry for entry in policy["paths"] if entry["id"] == "jankurai-source-tars"
        )
        remote_scratch = next(
            entry for entry in policy["paths"] if entry["id"] == "remote-source-scratch"
        )
        self.assertEqual(cargo["per_target_cap_bytes"], 20 * jope_diskd.GIB)
        self.assertEqual(source_tars["kind"], "jankurai-source-tar")
        self.assertEqual(source_tars["path"], "~/JopeDime")
        self.assertEqual(source_tars["max_file_bytes"], 512 * 1024**2)
        self.assertEqual(source_tars["ttl_hours"], 24)
        self.assertEqual(
            remote_scratch,
            {
                "id": "remote-source-scratch",
                "kind": "remote-source-scratch",
                "path": "~/lane_work",
                "ttl_seconds": 3600,
            },
        )
        config = self.home / "policy.yaml"
        config.write_text(json.dumps(policy), encoding="utf-8")
        self.assertEqual(jope_diskd.read_policy(config), policy)
        cargo["per_target_cap_bytes"] = 0
        config.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(config)
        cargo["per_target_cap_bytes"] = 20 * jope_diskd.GIB
        source_tars["path"] = "~/other-repo"
        config.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(config)
        source_tars["path"] = "~/JopeDime"
        remote_scratch["ttl_seconds"] = 1800
        config.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(config)

    def test_remote_source_scratch_dry_run_only_plans_old_unprotected_sources(
        self,
    ) -> None:
        root = self.home / "lane_work"
        lane = root / "svc08"
        safe = lane / "src-old"
        safe.mkdir(parents=True)
        (safe / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        archive = lane / "candidate-source.tar"
        archive.write_bytes(b"source snapshot")
        recent = lane / "src-recent"
        recent.mkdir()
        (recent / "Cargo.toml").write_text("recent", encoding="utf-8")
        protected = lane / "src-protected"
        (protected / "checkpoints").mkdir(parents=True)
        (protected / "checkpoints" / "model.pt").write_bytes(b"keep")
        receipt = lane / "receipts"
        receipt.mkdir()
        protected_archive = receipt / "bundle-source.tar"
        protected_archive.write_bytes(b"receipt")
        dataset = lane / "src-dataset"
        (dataset / "datasets").mkdir(parents=True)
        (dataset / "datasets" / "blob.bin").write_bytes(b"dataset")
        flywheel = lane / "src-flywheel"
        (flywheel / "flywheel_runs" / "run-1").mkdir(parents=True)
        run_archive = flywheel / "flywheel_runs" / "run-1" / "source.tar"
        run_archive.write_bytes(b"run")
        old_paths = (
            safe,
            safe / "Cargo.toml",
            archive,
            protected,
            protected / "checkpoints",
            protected / "checkpoints" / "model.pt",
            receipt,
            protected_archive,
            dataset,
            dataset / "datasets",
            dataset / "datasets" / "blob.bin",
            flywheel,
            flywheel / "flywheel_runs",
            flywheel / "flywheel_runs" / "run-1",
            run_archive,
        )
        for path in old_paths:
            os.utime(path, (1, 1))
        now = jope_diskd.time.time()
        entry = {
            "id": "remote-source-scratch",
            "path": "~/lane_work",
            "ttl_seconds": 3600,
        }
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(
                jope_diskd, "local_process_references", return_value=False
            ),
            mock.patch.object(jope_diskd, "path_size", return_value=100),
        ):
            freed = jope_diskd.handle_remote_source_scratch(
                entry, events.append, dry_run=True, now=now, needed_bytes=1000
            )
        self.assertEqual(freed, 115)
        self.assertEqual(events, [])
        self.assertTrue(safe.exists())
        self.assertTrue(archive.exists())
        self.assertTrue(recent.exists())
        self.assertTrue((protected / "checkpoints" / "model.pt").exists())
        self.assertTrue(protected_archive.exists())
        self.assertTrue((dataset / "datasets" / "blob.bin").exists())
        self.assertTrue(run_archive.exists())

    def test_remote_source_scratch_apply_rechecks_and_audits_only_old_candidates(
        self,
    ) -> None:
        root = self.home / "lane_work"
        lane = root / "svc08"
        safe = lane / "src-old"
        safe.mkdir(parents=True)
        (safe / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        archive = lane / "snapshot-source.tar"
        archive.write_bytes(b"old")
        for path in (safe, safe / "Cargo.toml", archive):
            os.utime(path, (1, 1))
        entry = {
            "id": "remote-source-scratch",
            "path": "~/lane_work",
            "ttl_seconds": 3600,
        }
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(
                jope_diskd, "local_process_references", return_value=False
            ),
            mock.patch.object(jope_diskd, "path_size", return_value=10),
        ):
            freed = jope_diskd.handle_remote_source_scratch(
                entry, events.append, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 10)
        self.assertFalse(safe.exists())
        self.assertTrue(archive.exists())
        self.assertEqual(
            [event["event"] for event in events], ["delete_intent_tree", "deleted_tree"]
        )

    def test_remote_source_scratch_retains_active_or_uncertain_candidates(self) -> None:
        root = self.home / "lane_work"
        lane = root / "svc08"
        safe = lane / "src-old"
        safe.mkdir(parents=True)
        (safe / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        for path in (safe, safe / "Cargo.toml"):
            os.utime(path, (1, 1))
        entry = {
            "id": "remote-source-scratch",
            "path": "~/lane_work",
            "ttl_seconds": 3600,
        }
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=None),
            mock.patch.object(
                jope_diskd, "local_process_references", return_value=False
            ),
        ):
            freed = jope_diskd.handle_remote_source_scratch(
                entry, lambda _event: None, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(safe.exists())

    def test_cleanup_inventory_uses_per_worktree_target_cap(self) -> None:
        repo = self.home / "JopeDime"
        worktrees_root = repo / ".agent" / "worktrees"
        worktree = worktrees_root / "open-lane"
        target = worktree / "target"
        target.mkdir(parents=True)
        os.utime(target, (1, 1))
        worktree_list = jope_diskd.subprocess.CompletedProcess(
            [], 0, f"worktree {worktree}\nHEAD abc123\nbranch refs/heads/open-lane\n", ""
        )
        not_merged = jope_diskd.subprocess.CompletedProcess([], 1, "", "")
        clean = jope_diskd.subprocess.CompletedProcess([], 0, "", "")

        def inventory(cap: int):
            with (
                mock.patch.object(jope_diskd.subprocess, "run", side_effect=[worktree_list, not_merged, clean]),
                mock.patch.object(jope_diskd, "path_size", return_value=100),
                mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=True),
            ):
                return jope_diskd.worktree_cleanup_inventory(
                    repo,
                    worktrees_root,
                    ttl_seconds=3600,
                    now=100_000,
                    include_closed_pr_worktrees=False,
                    idle_seconds=3600,
                    per_target_cap_bytes=cap,
                )

        self.assertEqual(inventory(101), [])
        over_cap = inventory(99)
        self.assertEqual(len(over_cap or []), 1)
        self.assertTrue(over_cap[0]["target_over_cap"])

    def source_tar_fixture(self) -> tuple[Path, Path, Path, dict[str, object], str]:
        repo = self.home / "JopeDime"
        worktrees_root = repo / ".agent" / "worktrees"
        worktree = worktrees_root / "lane"
        report_root = worktree / "target" / "jankurai" / "reports"
        source_tar = report_root / "run-1" / "input" / "source.tar"
        source_tar.parent.mkdir(parents=True)
        with source_tar.open("wb") as stream:
            stream.truncate(600 * 1024**2)
        os.utime(source_tar, (1, 1))
        worktrees_root.mkdir(parents=True, exist_ok=True)
        outside_checkout = repo.parent / "external-checkout"
        outside_checkout.mkdir(parents=True)
        listing = (
            f"worktree {repo}\nHEAD mainhead\n\n"
            f"worktree {worktree}\nHEAD lanehead\nbranch refs/heads/lane\n\n"
            f"worktree {outside_checkout}\nHEAD externalhead\nbranch refs/heads/external\n"
        )
        entry: dict[str, object] = {
            "id": "jankurai-source-tars",
            "kind": "jankurai-source-tar",
            "path": "~/JopeDime",
            "max_file_bytes": 512 * 1024**2,
            "ttl_hours": 24,
        }
        return repo, worktrees_root, source_tar, entry, listing

    def test_jankurai_source_tar_dry_run_preserves_file_and_does_not_log(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        report_root = source_tar.parents[2]
        summary = report_root / "run-1" / "summary.json"
        summary.write_text("{}\n", encoding="utf-8")
        small_source = report_root / "run-2" / "input" / "source.tar"
        small_source.parent.mkdir(parents=True)
        with small_source.open("wb") as stream:
            stream.truncate(int(entry["max_file_bytes"]))
        os.utime(small_source, (1, 1))
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=True, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, source_tar.stat().st_size)
        self.assertTrue(source_tar.exists())
        self.assertTrue(summary.exists())
        self.assertTrue(small_source.exists())
        self.assertEqual(events, [])

    def test_jankurai_source_tar_apply_unlinks_exact_file_and_logs_both_events(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=False, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, 600 * 1024**2)
        self.assertFalse(source_tar.exists())
        self.assertEqual([event["event"] for event in events], ["delete_intent_file", "deleted_file"])

    def test_jankurai_source_tar_pruner_skips_open_or_unreadable_inputs(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=None),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=False, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(source_tar.exists())
        self.assertEqual(events, [])

    def test_jankurai_source_tar_pruner_skips_open_file(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value={source_tar.resolve()}),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=False, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(source_tar.exists())
        self.assertEqual(events, [])

    def test_jankurai_source_tar_pruner_skips_recent_file(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        os.utime(source_tar, (99_999_000, 99_999_000))
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=True),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=False, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(source_tar.exists())
        self.assertEqual(events, [])

    def test_jankurai_source_tar_pruner_skips_active_checkout(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=True),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=False, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(source_tar.exists())
        self.assertEqual(events, [])

    def test_jankurai_source_tar_pruner_skips_hardlink(self) -> None:
        repo, worktrees_root, source_tar, entry, listing = self.source_tar_fixture()
        second_link = self.home / "second-link.tar"
        os.link(source_tar, second_link)
        events: list[dict[str, object]] = []
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                return_value=jope_diskd.subprocess.CompletedProcess([], 0, listing, ""),
            ),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
        ):
            freed = jope_diskd.handle_jankurai_source_tars(
                entry, repo, worktrees_root, events.append, dry_run=False, now=100_000_000, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(source_tar.exists())
        self.assertTrue(second_link.exists())
        self.assertEqual(events, [])

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

    def test_cargo_incremental_binary_cache_remains_rebuildable(self) -> None:
        target = self.home / "incremental" / "target"
        artifact = target / "debug" / "incremental" / "crate-hash" / "dep-graph.bin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"synthetic Cargo incremental cache")
        for path in (artifact, artifact.parent, artifact.parent.parent, artifact.parent.parent.parent, target):
            os.utime(path, (1, 1))
        self.assertTrue(jope_diskd.safe_target_tree(target, now=10**12))
        payload = artifact.parent / "training-data.bin"
        payload.write_bytes(b"synthetic non-Cargo binary payload")
        os.utime(payload, (1, 1))
        self.assertFalse(jope_diskd.safe_target_tree(target, now=10**12))

    def test_disk_policy_enables_closed_worktree_and_open_target_rules(self) -> None:
        policy_path = Path(__file__).with_name("disk-policy.yaml")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        entry = next(entry for entry in policy["paths"] if entry["id"] == "cargo-targets")
        self.assertEqual(entry["ttl_hours"], 2)
        self.assertEqual(entry["worktree_idle_hours"], 2)
        self.assertTrue(entry["remove_merged_worktrees"])
        self.assertTrue(entry["remove_closed_pr_worktrees"])
        self.assertTrue(entry["remove_stale_targets_on_open_worktrees"])
        temp_policy = self.home / "policy.yaml"
        temp_policy.write_text(json.dumps(policy), encoding="utf-8")
        self.assertEqual(jope_diskd.read_policy(temp_policy), policy)
        entry["remove_closed_pr_worktrees"] = "true"
        temp_policy.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(ValueError):
            jope_diskd.read_policy(temp_policy)

    def test_closed_pr_worktree_disposition_requires_exact_branch_head(self) -> None:
        closed = {("closed-lane", "deadbeef"): "CLOSED"}
        self.assertEqual(
            jope_diskd.cleanup_disposition(
                main_ancestor=False, branch="closed-lane", head="deadbeef", pr_states=closed,
            ),
            "closed",
        )
        self.assertIsNone(
            jope_diskd.cleanup_disposition(
                main_ancestor=False, branch="closed-lane", head="other", pr_states=closed,
            )
        )
        self.assertIsNone(
            jope_diskd.cleanup_disposition(
                main_ancestor=False, branch="open-lane", head="abc", pr_states={("open-lane", "abc"): "OPEN"},
            )
        )

    def test_worktree_removal_refuses_untracked_ignored_data(self) -> None:
        worktree = self.home / "lane"
        worktree.mkdir()
        (worktree / ".private-cache").mkdir()
        with mock.patch.object(
            jope_diskd.subprocess,
            "run",
            return_value=jope_diskd.subprocess.CompletedProcess([], 0, ".private-cache/\0", ""),
        ):
            self.assertFalse(jope_diskd.worktree_has_only_rebuildable_ignored_files(worktree))
        (worktree / "python" / ".venv").mkdir(parents=True)
        (worktree / "python" / "__pycache__").mkdir()
        (worktree / "AGENT_CHAT.md.lock").touch()
        with mock.patch.object(
            jope_diskd.subprocess,
            "run",
            return_value=jope_diskd.subprocess.CompletedProcess(
                [], 0, "target/\0python/.venv/\0python/__pycache__/\0AGENT_CHAT.md.lock\0", "",
            ),
        ):
            self.assertTrue(jope_diskd.worktree_has_only_rebuildable_ignored_files(worktree))

    def test_worktree_list_parser_preserves_detached_and_locked_state(self) -> None:
        parsed = jope_diskd.parse_worktree_list(
            "worktree /repo/main\nHEAD abc\nbranch refs/heads/main\n\n"
            "worktree /repo/lane\nHEAD def\ndetached\nlocked in-use\n"
        )
        self.assertEqual(parsed[0], {"path": "/repo/main", "head": "abc", "branch": "main"})
        self.assertEqual(parsed[1], {"path": "/repo/lane", "head": "def", "locked": True})

    def test_cleanup_inventory_selects_exact_closed_clean_head_and_skips_locked_lane(self) -> None:
        repo = self.home / "JopeDime"
        root = repo / ".agent" / "worktrees"
        lane = root / "closed-lane"
        locked_lane = root / "locked-lane"
        lane.mkdir(parents=True)
        locked_lane.mkdir()
        calls = [
            jope_diskd.subprocess.CompletedProcess(
                [], 0,
                f"worktree {lane}\nHEAD deadbeef\nbranch refs/heads/closed-lane\n\n"
                f"worktree {locked_lane}\nHEAD lockedhead\ndetached\nlocked service lane\n",
                "",
            ),
            jope_diskd.subprocess.CompletedProcess([], 1, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, json.dumps([{
                "headRefName": "closed-lane", "headRefOid": "deadbeef", "state": "CLOSED", "mergedAt": None,
            }]), ""),
        ]
        with (
            mock.patch.object(jope_diskd.subprocess, "run", side_effect=calls),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=True),
        ):
            found = jope_diskd.worktree_cleanup_inventory(repo, root, 24 * 3600, now=10**12)
        self.assertEqual(len(found or []), 1)
        self.assertEqual(found[0]["worktree"], lane.resolve())
        self.assertEqual(found[0]["disposition"], "closed")
        self.assertTrue(found[0]["clean"])
        self.assertIsNone(found[0]["ignored_safe"])
        self.assertFalse(found[0]["target_exists"])

    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_inventory_does_not_refresh_a_stale_stat_index_or_reset_the_idle_clock(self) -> None:
        repo = self.home / "JopeDime"
        root = repo / ".agent" / "worktrees"
        lane = root / "idle-lane"
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

        def git(*args: str, cwd: Path = repo) -> None:
            subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, env=env,
                           check=True, capture_output=True)

        repo.mkdir()
        git("init", "-q", "-b", "main")
        (repo / "tracked.txt").write_text("a\n", encoding="utf-8")
        git("add", "tracked.txt")
        git("commit", "-q", "-m", "init")
        git("update-ref", "refs/remotes/origin/main", "HEAD")
        git("worktree", "add", "-q", "--detach", str(lane), "main")
        git_dir = repo / ".git" / "worktrees" / "idle-lane"
        now = time.time()
        # A build or checkout leaves the tracked file's stat newer than the index entry.
        os.utime(lane / "tracked.txt", (now - 3600, now - 3600))
        for name in ("index", "HEAD"):
            os.utime(git_dir / name, (now - 3 * 3600, now - 3 * 3600))
        index_mtime = (git_dir / "index").stat().st_mtime
        found = jope_diskd.worktree_cleanup_inventory(repo, root, 2 * 3600, now=now, include_closed_pr_worktrees=False)
        self.assertEqual((git_dir / "index").stat().st_mtime, index_mtime)
        self.assertEqual([(c["worktree"], c["disposition"], c["idle"]) for c in found or []],
                         [(lane.resolve(), "merged", True)])

    def test_cleanup_inventory_rejects_closed_pr_with_different_local_head(self) -> None:
        repo = self.home / "JopeDime"
        root = repo / ".agent" / "worktrees"
        lane = root / "closed-lane"
        lane.mkdir(parents=True)
        calls = [
            jope_diskd.subprocess.CompletedProcess(
                [], 0, f"worktree {lane}\nHEAD localhead\nbranch refs/heads/closed-lane\n", "",
            ),
            jope_diskd.subprocess.CompletedProcess([], 1, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
            jope_diskd.subprocess.CompletedProcess([], 0, json.dumps([{
                "headRefName": "closed-lane", "headRefOid": "closedhead", "state": "CLOSED", "mergedAt": None,
            }]), ""),
        ]
        with (
            mock.patch.object(jope_diskd.subprocess, "run", side_effect=calls),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=True),
        ):
            found = jope_diskd.worktree_cleanup_inventory(repo, root, 24 * 3600, now=10**12)
        self.assertEqual(found, [])

    def test_cleanup_inventory_skips_recently_touched_merged_worktree(self) -> None:
        repo = self.home / "JopeDime"
        root = repo / ".agent" / "worktrees"
        lane = root / "recent-lane"
        lane.mkdir(parents=True)
        inventory = jope_diskd.subprocess.CompletedProcess(
            [], 0, f"worktree {lane}\nHEAD mergedhead\nbranch refs/heads/recent-lane\n", "",
        )
        with (
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                side_effect=[
                    inventory,
                    jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
                    jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
                ],
            ),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=False),
        ):
            found = jope_diskd.worktree_cleanup_inventory(repo, root, 2 * 3600, now=10**12)
        self.assertEqual(found, [])

    def test_cleanup_inventory_skips_detached_worktree_with_non_main_reflog_commit(self) -> None:
        repo = self.home / "JopeDime"
        root = repo / ".agent" / "worktrees"
        lane = root / "detached-lane"
        lane.mkdir(parents=True)
        with (
            mock.patch.object(
                jope_diskd.subprocess,
                "run",
                side_effect=[
                    jope_diskd.subprocess.CompletedProcess(
                        [], 0, f"worktree {lane}\nHEAD mainhead\ndetached\n", "",
                    ),
                    jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
                    jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
                    jope_diskd.subprocess.CompletedProcess([], 0, "mainhead\nlocalonly\n", ""),
                    jope_diskd.subprocess.CompletedProcess([], 0, "", ""),
                    jope_diskd.subprocess.CompletedProcess([], 1, "", ""),
                ],
            ),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=True),
        ):
            found = jope_diskd.worktree_cleanup_inventory(repo, root, 2 * 3600, now=10**12)
        self.assertEqual(len(found or []), 1)
        self.assertFalse(found[0]["history_safe"])

    def test_cargo_target_cleanup_skips_any_live_or_unknown_consumer(self) -> None:
        repo = self.home / "JopeDime"
        worktrees = repo / ".agent" / "worktrees"
        worktree = worktrees / "merged-lane"
        target = worktree / "target"
        (target / "debug" / "deps").mkdir(parents=True)
        (target / "debug" / "deps" / "libbuild.rlib").write_bytes(b"build")
        os.utime(target, (1, 1))
        entry = {
            "id": "cargo-targets",
            "path": "~/JopeDime/.agent/worktrees",
            "remove_merged_worktrees": True,
        }
        policy = {"local_only": True}
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
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=None),
            mock.patch.object(jope_diskd, "path_size", return_value=4096),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, policy, lambda _event: None, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 0)
        self.assertTrue(target.exists())

    def test_merged_worktree_is_not_removed_if_idle_window_closes_before_mutation(self) -> None:
        repo = self.home / "JopeDime"
        worktree = repo / ".agent" / "worktrees" / "merged-lane"
        worktree.mkdir(parents=True)
        candidate = {
            "worktree": worktree,
            "target": worktree / "target",
            "target_exists": False,
            "stale_target": False,
            "branch": "merged-lane",
            "head": "abc123",
            "disposition": "merged",
            "clean": True,
            "ignored_safe": True,
        }
        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", return_value=repo / ".agent" / "worktrees"),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=False),
            mock.patch.object(jope_diskd, "path_size", return_value=1024),
            mock.patch.object(jope_diskd.subprocess, "run") as run,
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
        self.assertTrue(worktree.exists())
        run.assert_not_called()

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
        entry = {
            "id": "cargo-targets",
            "path": "~/JopeDime/.agent/worktrees",
            "remove_merged_worktrees": True,
        }
        policy = {"local_only": True}
        branch = "merged-lane"
        head = "abc123"
        candidate = {
            "worktree": worktree,
            "target": target,
            "target_exists": True,
            "stale_target": True,
            "branch": branch,
            "head": head,
            "disposition": "merged",
            "clean": True,
            "ignored_safe": True,
        }
        events: list[dict[str, object]] = []

        def remove_worktree(*_args: object, **_kwargs: object) -> jope_diskd.subprocess.CompletedProcess[str]:
            worktree.rmdir()
            return jope_diskd.subprocess.CompletedProcess([], 0, "", "")

        with (
            mock.patch.object(jope_diskd.Path, "home", return_value=self.home),
            mock.patch.object(jope_diskd, "expand_path", side_effect=lambda value: self.home / value.removeprefix("~/")),
            mock.patch.object(jope_diskd, "worktree_cleanup_inventory", return_value=[candidate]),
            mock.patch.object(jope_diskd.subprocess, "run", side_effect=remove_worktree),
            mock.patch.object(jope_diskd, "open_handles_under", return_value=set()),
            mock.patch.object(jope_diskd, "local_process_references", return_value=False),
            mock.patch.object(jope_diskd, "worktree_head_is_unchanged", return_value=True),
            mock.patch.object(jope_diskd, "worktree_idle_for_cleanup", return_value=True),
            mock.patch.object(jope_diskd, "path_size", return_value=4096),
        ):
            freed = jope_diskd.handle_cargo_targets(
                entry, repo, policy, events.append, dry_run=False, now=10**12, needed_bytes=1
            )
        self.assertEqual(freed, 4096)
        self.assertFalse(target.exists())
        self.assertFalse(worktree.exists())
        self.assertEqual(
            [event["event"] for event in events],
            ["delete_intent_tree", "deleted_tree", "remove_worktree_intent", "removed_worktree"],
        )

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


if __name__ == "__main__":
    unittest.main()
