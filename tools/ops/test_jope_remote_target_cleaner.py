"""Tests for the xbabe3 rebuildable-only remote Cargo target cleaner."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jope_remote_target_cleaner as cleaner


class RemoteTargetCleanerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="jope-remote-target-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "remote_tests" / "target"
        self.marker_target()

    def marker_target(self) -> None:
        (self.target / "debug" / "deps").mkdir(parents=True, exist_ok=True)
        (self.target / "CACHEDIR.TAG").write_text("cargo cache\n", encoding="utf-8")
        (self.target / ".rustc_info.json").write_text("{}\n", encoding="utf-8")
        (self.target / "debug" / "deps" / "libsample.rlib").write_bytes(b"compiled")

    def docker_result(self, returncode: int = 0, stdout: str = "") -> mock.Mock:
        return mock.Mock(returncode=returncode, stdout=stdout)

    def test_docker_prune_targets_only_dangling_jope_targets_without_running_mounts(self) -> None:
        runner = mock.Mock(side_effect=[
            self.docker_result(stdout="jope-pr-ci-target\njope-main-full-target\njope-sdk\nother-target\n"),
            self.docker_result(stdout="runner-id\n"),
            self.docker_result(),
            self.docker_result(stdout="local /var/lib/docker/volumes/jope-pr-ci-target/_data"),
            self.docker_result(stdout="jope-pr-ci-target"),
        ])
        lock = self.root / "lock" / "cleaner.lock"
        audit = self.root / "audit.jsonl"
        with mock.patch.object(cleaner, "append_audit") as append_audit:
            removed = cleaner.prune_unmounted_jope_target_volumes(
                free_bytes=1,
                minimum_free_bytes=100,
                audit=audit,
                lock=lock,
                runner=runner,
            )
        self.assertEqual(removed, ["jope-pr-ci-target"])
        self.assertEqual(runner.call_count, 5)
        self.assertEqual(
            [call.args[0][1:3] for call in runner.call_args_list],
            [["volume", "ls"], ["ps", "--quiet"], ["ps", "--quiet"], ["volume", "inspect"], ["volume", "rm"]],
        )
        self.assertEqual(
            [call.args[0]["event"] for call in append_audit.call_args_list],
            ["docker_target_volume_delete_intent", "docker_target_volume_deleted"],
        )

    def test_docker_prune_rejects_nonlocal_or_noncanonical_volume(self) -> None:
        for inspection in (
            "nfs /var/lib/docker/volumes/jope-pr-ci-target/_data",
            "local /other/path",
        ):
            runner = mock.Mock(side_effect=[
                self.docker_result(stdout="jope-pr-ci-target\n"),
                self.docker_result(),
                self.docker_result(stdout=inspection),
            ])
            with mock.patch.object(cleaner, "append_audit") as append_audit:
                removed = cleaner.prune_unmounted_jope_target_volumes(
                    free_bytes=1,
                    minimum_free_bytes=100,
                    audit=self.root / "audit.jsonl",
                    lock=self.root / f"lock/{len(inspection)}.lock",
                    runner=runner,
                )
            self.assertEqual(removed, [])
            self.assertEqual(runner.call_count, 3)
            append_audit.assert_not_called()

    def test_docker_prune_refuses_race_and_stops_when_docker_unavailable(self) -> None:
        runner = mock.Mock(side_effect=[
            self.docker_result(stdout="jope-pr-ci-target\n"),
            self.docker_result(),
            self.docker_result(stdout="local /var/lib/docker/volumes/jope-pr-ci-target/_data"),
            self.docker_result(returncode=1),
        ])
        with mock.patch.object(cleaner, "append_audit") as append_audit:
            removed = cleaner.prune_unmounted_jope_target_volumes(
                free_bytes=1, minimum_free_bytes=100, audit=self.root / "audit.jsonl",
                lock=self.root / "lock/race.lock", runner=runner,
            )
        self.assertEqual(removed, [])
        self.assertEqual(append_audit.call_args_list[-1].args[0]["event"], "docker_target_volume_remove_refused")

        runner = mock.Mock(return_value=self.docker_result(returncode=1))
        self.assertEqual(
            cleaner.prune_unmounted_jope_target_volumes(
                free_bytes=1, minimum_free_bytes=100, audit=self.root / "audit.jsonl",
                lock=self.root / "lock/unavailable.lock", runner=runner,
            ),
            [],
        )
        runner.assert_called_once()

    def test_docker_prune_skips_when_disk_floor_is_met(self) -> None:
        runner = mock.Mock()
        self.assertEqual(
            cleaner.prune_unmounted_jope_target_volumes(
                free_bytes=100, minimum_free_bytes=100, audit=self.root / "audit.jsonl",
                lock=self.root / "lock/space-ok.lock", runner=runner,
            ),
            [],
        )
        runner.assert_not_called()

    def test_x3_volume_reclaim_is_remeasured_before_target_cleanup(self) -> None:
        with (
            mock.patch.object(cleaner, "TARGET", self.target),
            mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent),
            mock.patch.object(
                cleaner.shutil,
                "disk_usage",
                side_effect=[mock.Mock(free=1), mock.Mock(free=cleaner.MIN_FREE_BYTES_BY_HOST["xbabe3"])],
            ),
            mock.patch.object(cleaner, "prune_unmounted_jope_target_volumes", return_value=["jope-ci-target"]) as prune,
            mock.patch.object(cleaner, "process_references") as process_references,
        ):
            self.assertEqual(
                cleaner.run(
                    apply=True,
                    target=self.target,
                    scratch_root=self.target.parent,
                    audit=self.root / "audit.jsonl",
                    lock=self.root / "lock/cleaner.lock",
                    owner_uid=os.getuid(),
                    host="xbabe3",
                ),
                0,
            )
        prune.assert_called_once()
        process_references.assert_not_called()
        self.assertTrue((self.target / "debug" / "deps").exists())

    def test_cargo_tree_accepts_rebuildable_outputs(self) -> None:
        candidates = cleaner.cargo_output_candidates(self.target, os.getuid())
        self.assertIsNotNone(candidates)
        self.assertEqual([entry[0] for entry in candidates or []], [self.target / "debug" / "deps"])
        self.assertGreater(candidates[0][1], 0)
        self.assertEqual(candidates[0][2], 1)

    def test_cargo_tree_rejects_protected_or_linked_content(self) -> None:
        checkpoint = self.target / "debug" / "deps" / "checkpoint.pt"
        checkpoint.write_bytes(b"protected")
        self.assertEqual(cleaner.cargo_output_candidates(self.target, os.getuid()), [])
        checkpoint.unlink()
        external = self.root / "outside"
        external.write_text("keep", encoding="utf-8")
        (self.target / "debug" / "deps" / "linked.rlib").symlink_to(external)
        self.assertEqual(cleaner.cargo_output_candidates(self.target, os.getuid()), [])

    def test_rebuildable_scan_preserves_target_tmp_fixtures(self) -> None:
        fixture = self.target / "tmp" / "golden" / "train.csv"
        fixture.parent.mkdir(parents=True)
        fixture.write_text("fixture", encoding="utf-8")
        candidates = cleaner.cargo_output_candidates(self.target, os.getuid())
        self.assertEqual([entry[0] for entry in candidates or []], [self.target / "debug" / "deps"])
        self.assertTrue(fixture.exists())

    def make_proc(self, *, cwd: Path | None = None, env: bytes = b"", fd_target: Path | None = None) -> Path:
        proc = self.root / "proc"
        process = proc / "123"
        if process.exists():
            shutil.rmtree(process)
        (process / "fd").mkdir(parents=True)
        (process / "status").write_text(f"Name:\ttest\nUid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n", encoding="utf-8")
        actual_cwd = cwd or self.root
        (process / "cwd").symlink_to(actual_cwd, target_is_directory=True)
        (process / "environ").write_bytes(env)
        if fd_target is not None:
            (process / "fd" / "3").symlink_to(fd_target)
        return proc

    def test_process_scan_detects_cargo_target_environment_and_open_fds(self) -> None:
        proc = self.make_proc(env=f"CARGO_TARGET_DIR={self.target}\0".encode())
        self.assertTrue(cleaner.process_references(self.target, proc))
        proc = self.make_proc(fd_target=self.target / "debug" / "deps" / "libsample.rlib")
        self.assertTrue(cleaner.process_references(self.target, proc))

    def test_stale_source_cwd_without_target_reference_does_not_block(self) -> None:
        source = self.target.parent / "src.123456"
        source.mkdir()
        proc = self.make_proc(cwd=source)
        self.assertFalse(cleaner.process_references(self.target, proc))

    def test_process_scan_reports_unknown_when_proc_is_unavailable(self) -> None:
        self.assertIsNone(cleaner.process_references(self.target, self.root / "missing-proc"))

    def test_process_scan_tolerates_pid_exit_race(self) -> None:
        proc = self.make_proc()
        with mock.patch.object(cleaner.os, "readlink", side_effect=ProcessLookupError):
            self.assertFalse(cleaner.process_references(self.target, proc))

    def test_dry_run_keeps_target_and_records_no_deletion(self) -> None:
        audit = self.root / "audit.jsonl"
        lock = self.root / "lock" / "cleaner.lock"
        with (
            mock.patch.object(cleaner, "TARGET", self.target),
            mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent),
            mock.patch.object(cleaner.shutil, "disk_usage", return_value=mock.Mock(free=1)),
            mock.patch.object(cleaner, "process_references", return_value=False),
            mock.patch.object(cleaner, "prune_unmounted_jope_target_volumes", return_value=[]),
            mock.patch.object(cleaner.socket, "gethostname", return_value="xbabe3"),
        ):
            self.assertEqual(cleaner.run(apply=False, target=self.target, scratch_root=self.target.parent,
                                         audit=audit, lock=lock, owner_uid=os.getuid()), 0)
        self.assertTrue(self.target.exists())
        self.assertFalse(audit.exists())

    def test_active_cargo_reference_prevents_apply(self) -> None:
        audit = self.root / "audit.jsonl"
        lock = self.root / "lock" / "cleaner.lock"
        with (
            mock.patch.object(cleaner, "TARGET", self.target),
            mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent),
            mock.patch.object(cleaner.shutil, "disk_usage", return_value=mock.Mock(free=1)),
            mock.patch.object(cleaner, "process_references", return_value=True),
            mock.patch.object(cleaner, "prune_unmounted_jope_target_volumes", return_value=[]),
            mock.patch.object(cleaner.socket, "gethostname", return_value="xbabe3"),
        ):
            self.assertEqual(cleaner.run(apply=True, target=self.target, scratch_root=self.target.parent,
                                         audit=audit, lock=lock, owner_uid=os.getuid()), 0)
        self.assertTrue((self.target / "debug" / "deps").exists())
        self.assertFalse(audit.exists())

    def test_apply_removes_only_verified_target_and_audits_it(self) -> None:
        audit = self.root / "audit.jsonl"
        lock = self.root / "lock" / "cleaner.lock"
        with (
            mock.patch.object(cleaner, "TARGET", self.target),
            mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent),
            mock.patch.object(cleaner.shutil, "disk_usage", return_value=mock.Mock(free=1)),
            mock.patch.object(cleaner, "process_references", return_value=False),
            mock.patch.object(cleaner, "prune_unmounted_jope_target_volumes", return_value=[]),
            mock.patch.object(cleaner.socket, "gethostname", return_value="xbabe3"),
        ):
            self.assertEqual(cleaner.run(apply=True, target=self.target, scratch_root=self.target.parent,
                                         audit=audit, lock=lock, owner_uid=os.getuid()), 0)
        self.assertTrue(self.target.exists())
        self.assertFalse((self.target / "debug" / "deps").exists())
        self.assertTrue((self.target / "CACHEDIR.TAG").exists())
        records = audit.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["event"] for line in records], ["delete_intent", "deleted"])

    def test_wrong_host_and_high_free_space_preserve_target(self) -> None:
        with mock.patch.object(cleaner, "TARGET", self.target), mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent):
            self.assertEqual(cleaner.run(apply=True, target=self.target, scratch_root=self.target.parent,
                                         owner_uid=os.getuid(), host="xbabe2"), 0)
            with mock.patch.object(cleaner.shutil, "disk_usage", return_value=mock.Mock(free=cleaner.MIN_FREE_BYTES_BY_HOST["xbabe3"])):
                self.assertEqual(cleaner.run(apply=True, target=self.target, scratch_root=self.target.parent,
                                             owner_uid=os.getuid(), host="xbabe3"), 0)
        self.assertTrue(self.target.exists())

    def test_xbabe1_requires_120_gib_and_uses_host_specific_audit(self) -> None:
        lock = self.root / "lock" / "cleaner.lock"
        with (
            mock.patch.object(cleaner, "TARGET", self.target),
            mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent),
            mock.patch.object(cleaner.shutil, "disk_usage", return_value=mock.Mock(free=119 * 1024**3)),
            mock.patch.object(cleaner, "process_references", return_value=False),
            mock.patch.object(cleaner, "prune_unmounted_jope_target_volumes") as prune,
            mock.patch.object(cleaner, "append_audit") as append_audit,
        ):
            self.assertEqual(cleaner.run(apply=True, target=self.target, scratch_root=self.target.parent,
                                         lock=lock, owner_uid=os.getuid(), host="xbabe1"), 0)
        self.assertEqual(append_audit.call_args_list[0].args[1], Path("/var/log/jope-diskd-xbabe1/remote-target-cleaner.jsonl"))
        prune.assert_not_called()
        self.assertFalse((self.target / "debug" / "deps").exists())
        self.assertTrue(self.target.exists())

    def test_unapproved_host_is_fail_closed(self) -> None:
        with mock.patch.object(cleaner, "TARGET", self.target), mock.patch.object(cleaner, "SCRATCH_ROOT", self.target.parent):
            self.assertEqual(cleaner.run(apply=True, target=self.target, scratch_root=self.target.parent,
                                         owner_uid=os.getuid(), host="xbabe2"), 0)
        self.assertTrue(self.target.exists())


if __name__ == "__main__":
    unittest.main()
