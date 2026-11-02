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



class DiskWorktreeCleanupTests(unittest.TestCase):
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
