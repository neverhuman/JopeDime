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
