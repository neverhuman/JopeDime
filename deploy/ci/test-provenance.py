#!/usr/bin/env python3
"""Provenance bind: PR base vs exact-head for runner-change PRs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_bind():
    path = ROOT / "deploy/runner/provenance_bind.py"
    spec = importlib.util.spec_from_file_location("provenance_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = load_bind()

COMMON_TEMPLATE = """#!/bin/bash
JOPE_RUNNER_DEPLOYED=(
  /opt/jope-runner/common.sh
  /opt/jope-runner/run.sh
  /opt/jope-runner/disk-guard.sh
  /opt/jope-runner/prune.sh
  /opt/jope-runner/job-started.sh
  /opt/jope-runner/job-policy.py
  /etc/systemd/system/jope-runner@.service
  /etc/systemd/system/jope-runner@.service.d/runtime-max.conf
  /etc/systemd/system/jope-runner@.timer
  /etc/systemd/system/jope-runner-prune.service
  /etc/systemd/system/jope-runner-prune.timer
  /etc/systemd/system/jope-docker-proxy.service
  /etc/tmpfiles.d/jope-exclusive-cpu.conf
)
# marker={marker}
"""

DEPLOYED_REL = [
    "deploy/runner/common.sh",
    "deploy/runner/run.sh",
    "deploy/runner/disk-guard.sh",
    "deploy/runner/prune.sh",
    "deploy/runner/job-started.sh",
    "deploy/runner/job-policy.py",
    "deploy/runner/jope-runner@.service",
    "deploy/runner/jope-runner@.service.d/runtime-max.conf",
    "deploy/runner/jope-runner@.timer",
    "deploy/runner/jope-runner-prune.service",
    "deploy/runner/jope-runner-prune.timer",
    "deploy/runner/jope-docker-proxy.service",
    "deploy/runner/jope-exclusive-cpu.conf",
]

BASH_DIGEST = r"""
mapfile -t deployed < <(sed -n '/^JOPE_RUNNER_DEPLOYED=(/,/^)/p' deploy/runner/common.sh   | grep -oE '/(opt/jope-runner|etc/systemd/system|etc/tmpfiles.d)/[^ ]+')
files=()
for d in "${deployed[@]}"; do
  rel=${d#/opt/jope-runner/}
  rel=${rel#/etc/systemd/system/}
  rel=${rel#/etc/tmpfiles.d/}
  files+=("deploy/runner/$rel")
done
cat "${files[@]}" | sha256sum | cut -d' ' -f1
"""


def write_tree(repo: Path, marker: str) -> None:
    for rel in DEPLOYED_REL:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("common.sh"):
            path.write_text(COMMON_TEMPLATE.format(marker=marker))
        else:
            path.write_text(f"{rel}\n{marker}\n")
    (repo / "deploy/runner/entrypoint.sh").write_text(f"entrypoint\n{marker}\n")
    (repo / "deploy/runner/mac-capacity.json").write_text(json.dumps(
        {"runner_label": "jope-mac-ci"} if marker.startswith("head") else {}))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def digest_of(repo: Path, rev=None) -> str:
    return bind.scripts_digest(repo, rev)


class ManualPrBindTests(unittest.TestCase):
    def facts(self):
        repository = dict(id=1376159324, full_name="neverhuman/JopeDime")
        env = dict(GITHUB_REPOSITORY="neverhuman/JopeDime", GITHUB_REPOSITORY_ID="1376159324",
            GITHUB_SHA="b"*40, GITHUB_REF="refs/heads/codex/runner",
            GITHUB_WORKFLOW_REF="neverhuman/JopeDime/.github/workflows/ci.yml@refs/heads/codex/runner")
        pr = dict(number=420, state="open", merged=False,
            head=dict(sha="b"*40, ref="codex/runner", repo=repository.copy()),
            base=dict(sha="a"*40, ref="main", repo=repository.copy()))
        return env,pr

    def test_exact_open_manual_pr_uses_current_merged_base(self):
        env,pr=self.facts()
        with mock.patch.object(bind,"github_public_json",side_effect=[pr,dict(object=dict(sha="a"*40))]) as api:
            self.assertEqual("a"*40,bind.manual_pr_bind(env,"420"))
            self.assertEqual(["pulls/420","git/ref/heads/main"],[c.args[0] for c in api.call_args_list])

    def test_wrong_head_fork_workflow_state_base_and_stale_main_refuse(self):
        import copy
        mutations=[("state","closed"),("merged",True),("number",421),
            ("head.sha","c"*40),("head.ref","foreign"),("head.repo.id",1),
            ("head.repo.full_name","foreign/repo"),("base.ref","other"),
            ("base.sha","HEAD"),("base.repo.id",1),("base.repo.full_name","foreign/repo")]
        for key,value in mutations:
            with self.subTest(key=key):
                env,pr=self.facts();changed=copy.deepcopy(pr);dst=changed;parts=key.split(".")
                for part in parts[:-1]: dst=dst[part]
                dst[parts[-1]]=value
                with mock.patch.object(bind,"github_public_json",return_value=changed),self.assertRaises(ValueError):
                    bind.manual_pr_bind(env,"420")
        env,pr=self.facts()
        with mock.patch.object(bind,"github_public_json",side_effect=[pr,dict(object=dict(sha="c"*40))]),self.assertRaises(ValueError):
            bind.manual_pr_bind(env,"420")
        for key,value in [("GITHUB_SHA","c"*40),("GITHUB_REF","refs/heads/main"),
            ("GITHUB_WORKFLOW_REF","neverhuman/JopeDime/.github/workflows/full-main.yml@refs/heads/codex/runner"),
            ("GITHUB_REPOSITORY_ID","1"),("GITHUB_REPOSITORY","foreign/repo")]:
            with self.subTest(key=key):
                env,pr=self.facts();env[key]=value
                with mock.patch.object(bind,"github_public_json",return_value=pr),self.assertRaises(ValueError):
                    bind.manual_pr_bind(env,"420")
        for number in ["0","-1","420/path"," 420","0420","1"*10]:
            with self.subTest(number=number),mock.patch.object(bind,"github_public_json") as api,self.assertRaises(ValueError):
                bind.manual_pr_bind(self.facts()[0],number)
            api.assert_not_called()

    def test_dispatch_payload_routes_through_validation_and_default_stays_exact(self):
        scratch=ROOT/'target/tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            path=Path(name)/'event.json';env,_=self.facts()
            env.update(GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_EVENT_PATH=str(path),BASE_SHA='a'*40)
            path.write_text(json.dumps(dict(inputs=dict(provenance_pr='420'))))
            with mock.patch.object(bind,'manual_pr_bind',return_value='a'*40) as validate:
                self.assertEqual('a'*40,bind.bind_rev_from_env(env))
                validate.assert_called_once_with(env,'420')
            path.write_text(json.dumps(dict(inputs=dict(provenance_pr=''))))
            with mock.patch.object(bind,'github_public_json') as api:
                self.assertIsNone(bind.bind_rev_from_env(env))
                api.assert_not_called()

    def test_api_missing_token_and_failure_are_not_exact_head_fallback(self):
        with self.assertRaises(ValueError):bind.github_public_json("pulls/420",{})
        with mock.patch.object(bind,"github_public_json",side_effect=OSError("unavailable")),self.assertRaises(OSError):
            bind.manual_pr_bind(self.facts()[0],"420")


class ProvenanceBindTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        write_tree(self.repo, "base-v1")
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "test")
        git(self.repo, "config", "advice.defaultBranchName", "false")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-m", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.base_digest = digest_of(self.repo, self.base_sha)

        write_tree(self.repo, "head-v2-runner-change")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-m", "runner change")
        self.head_sha = git(self.repo, "rev-parse", "HEAD")
        self.head_digest = digest_of(self.repo, None)
        self.assertNotEqual(self.base_digest, self.head_digest)

        self.ep_base = hashlib.sha256(
            bind.file_bytes(self.repo, self.base_sha, "deploy/runner/entrypoint.sh")
        ).hexdigest()
        self.ep_head = hashlib.sha256(
            (self.repo / "deploy/runner/entrypoint.sh").read_bytes()
        ).hexdigest()
        self.assertNotEqual(self.ep_base, self.ep_head)

        self.ep_file = self.repo / "fake-image-entrypoint"
        self.image_id = "sha256:image-fixed"

    def tearDown(self):
        self.tmp.cleanup()

    def _env(self, event, host_digest, entrypoint_marker, **extra):
        if entrypoint_marker == "base":
            self.ep_file.write_bytes(
                bind.file_bytes(self.repo, self.base_sha, "deploy/runner/entrypoint.sh")
            )
        else:
            self.ep_file.write_bytes(
                (self.repo / "deploy/runner/entrypoint.sh").read_bytes()
            )
        env = {
            "GITHUB_EVENT_NAME": event,
            "JOPE_RUNNER_SCRIPTS_SHA256": host_digest,
            "JOPE_RUNNER_ENTRYPOINT_PATH": str(self.ep_file),
            "JOPE_RUNNER_IMAGE_ID": self.image_id,
            "JOPE_RUNNER_IMAGE_ID_RECORDED": self.image_id,
            "JOPE_RUNNER_HOST": "xbabe1",
            "GITHUB_WORKSPACE": str(self.repo),
        }
        env.update(extra)
        return env

    def test_candidate_capability_label_cannot_bootstrap_its_own_pool(self):
        self.assertEqual("jope-ci", bind.runner_pool(self.repo,
            {"GITHUB_EVENT_NAME":"pull_request", "BASE_SHA":self.base_sha}))
        self.assertEqual("jope-mac-ci", bind.runner_pool(self.repo,
            {"GITHUB_EVENT_NAME":"push"}))
        path=self.repo/"deploy/runner/mac-capacity.json"
        for label in ["foreign", "jope-ci,jope-main-guard", "jope-mac-ci\nINJECTED=1", None, 12]:
            path.write_text(json.dumps({"runner_label":label}))
            with self.subTest(label=label),self.assertRaises(ValueError):
                bind.runner_pool(self.repo,{"GITHUB_EVENT_NAME":"push"})

    def test_pull_request_passes_when_host_matches_base_not_head(self):
        env = self._env("pull_request", self.base_digest, "base", BASE_SHA=self.base_sha)
        result = bind.verify_scripts_and_image(self.repo, env)
        self.assertEqual(result["bind_rev"], self.base_sha)
        self.assertEqual(result["expected_scripts_sha256"], self.base_digest)

    def test_pull_request_fails_when_host_at_other_digest(self):
        other = hashlib.sha256(b"not-base-not-head").hexdigest()
        env = self._env("pull_request", other, "base", BASE_SHA=self.base_sha)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(self.repo, env)

    def test_pull_request_fails_when_host_only_matches_head(self):
        env = self._env("pull_request", self.head_digest, "head", BASE_SHA=self.base_sha)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(self.repo, env)

    def test_main_push_requires_exact_match_to_checkout(self):
        env = self._env("push", self.head_digest, "head")
        result = bind.verify_scripts_and_image(self.repo, env)
        self.assertEqual(result["bind_rev"], "WORKTREE")
        self.assertEqual(result["expected_scripts_sha256"], self.head_digest)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(
                self.repo, self._env("push", self.base_digest, "base")
            )

    def test_workflow_dispatch_requires_exact_match(self):
        env = self._env("workflow_dispatch", self.head_digest, "head")
        bind.verify_scripts_and_image(self.repo, env)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(
                self.repo, self._env("workflow_dispatch", self.base_digest, "base")
            )

    def test_pull_request_without_base_sha_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "BASE_SHA"):
            bind.bind_rev("pull_request", "")

    def test_hooks_follow_same_bind(self):
        host = Path(self.tmp.name) / "host-hooks"
        host.mkdir()
        for name in bind.HOOK_NAMES:
            (host / name).write_bytes(
                bind.file_bytes(self.repo, self.base_sha, f"deploy/runner/{name}")
            )
        env = {"GITHUB_EVENT_NAME": "pull_request", "BASE_SHA": self.base_sha}
        bind.verify_hooks(self.repo, host, env)
        for name in bind.HOOK_NAMES:
            (host / name).write_bytes((self.repo / "deploy/runner" / name).read_bytes())
        with self.assertRaisesRegex(ValueError, "admission hook differs"):
            bind.verify_hooks(self.repo, host, env)
        env_push = {"GITHUB_EVENT_NAME": "push"}
        bind.verify_hooks(self.repo, host, env_push)

    def test_event_path_supplies_base_sha(self):
        event = Path(self.tmp.name) / "event.json"
        event.write_text(json.dumps({"pull_request": {"base": {"sha": self.base_sha}}}))
        env = self._env(
            "pull_request", self.base_digest, "base", GITHUB_EVENT_PATH=str(event)
        )
        env.pop("BASE_SHA", None)
        result = bind.verify_scripts_and_image(self.repo, env)
        self.assertEqual(result["bind_rev"], self.base_sha)

    def _shallow_clone(self) -> Path:
        """A clean depth-1 clone of the PR head, as actions/checkout leaves a cold workspace."""
        git(self.repo, "config", "uploadpack.allowReachableSHA1InWant", "true")
        clone = Path(self.tmp.name) / "shallow-checkout"
        subprocess.check_call(
            ["git", "clone", "-q", "--depth=1", f"file://{self.repo}", str(clone)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.assertEqual(git(clone, "rev-parse", "HEAD"), self.head_sha)
        self.assertFalse(bind.has_commit(clone, self.base_sha))
        return clone

    def test_shallow_checkout_fetches_base_then_binds_to_it(self):
        clone = self._shallow_clone()
        # The cold-workspace failure the explicit fetch exists for.
        with self.assertRaises(subprocess.CalledProcessError):
            bind.scripts_digest(clone, self.base_sha)
        env = self._env("pull_request", self.base_digest, "base", BASE_SHA=self.base_sha)
        self.assertEqual(bind.ensure_bind_rev(clone, env), self.base_sha)
        self.assertTrue(bind.has_commit(clone, self.base_sha))
        result = bind.verify_scripts_and_image(clone, env)
        self.assertEqual(result["expected_scripts_sha256"], self.base_digest)
        head_env = self._env("pull_request", self.head_digest, "head", BASE_SHA=self.base_sha)
        with self.assertRaisesRegex(ValueError, "different files"):
            bind.verify_scripts_and_image(clone, head_env)

    def test_shallow_checkout_push_fetches_nothing(self):
        clone = self._shallow_clone()
        self.assertIsNone(bind.ensure_bind_rev(clone, {"GITHUB_EVENT_NAME": "push"}))
        self.assertFalse(bind.has_commit(clone, self.base_sha))

    def test_unfetchable_base_fails_closed(self):
        clone = self._shallow_clone()
        env = {"GITHUB_EVENT_NAME": "pull_request", "BASE_SHA": "0" * 40}
        with mock.patch.object(bind, "has_commit", return_value=False), \
                mock.patch.object(bind.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "all-zero push sentinel"):
                bind.ensure_bind_rev(clone, env)
            popen.assert_not_called()

    def test_fetch_token_goes_through_git_config_env_not_argv(self):
        env = bind.fetch_env(
            self.repo,
            {"JOPE_PROVENANCE_FETCH_TOKEN": "tok", "GITHUB_SERVER_URL": "https://github.com"},
        )
        n = int(env["GIT_CONFIG_COUNT"]) - 1
        self.assertEqual(env[f"GIT_CONFIG_KEY_{n}"], "http.https://github.com/.extraheader")
        self.assertEqual(
            env[f"GIT_CONFIG_VALUE_{n}"], "AUTHORIZATION: basic eC1hY2Nlc3MtdG9rZW46dG9r"
        )
        no_token = bind.fetch_env(self.repo, {})
        self.assertNotIn("http.https://github.com/.extraheader", no_token.values())

    def test_base_sha_must_be_40_lowercase_hex(self):
        sha = self.base_sha
        bad = [
            "HEAD",
            "main",
            "refs/heads/main",
            sha.upper(),
            sha[:7],
            sha[:39],
            sha + "0",
            sha[:39] + "g",
            " " + sha,
            sha + "\n",
            "--upload-pack=touch${IFS}pwned" + "0" * 10,
            "0" * 64,
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "40 lowercase hex"):
                    bind.bind_rev("pull_request", value)
        self.assertEqual(bind.bind_rev("pull_request", sha), sha)

    def test_malformed_base_sha_is_refused_before_any_fetch(self):
        clone = self._shallow_clone()
        env = {"GITHUB_EVENT_NAME": "pull_request", "BASE_SHA": self.base_sha[:12]}
        with mock.patch.object(bind.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "40 lowercase hex"):
                bind.ensure_bind_rev(clone, env)
            popen.assert_not_called()

    def test_malformed_event_payload_base_fails_closed(self):
        event = Path(self.tmp.name) / "event.json"
        event.write_text(json.dumps({"pull_request": {"base": {"sha": "main"}}}))
        env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_EVENT_PATH": str(event)}
        with self.assertRaisesRegex(ValueError, "40 lowercase hex"):
            bind.bind_rev_from_env(env)

    def test_base_fetch_times_out_and_fails_closed(self):
        clone = self._shallow_clone()
        hang = Path(self.tmp.name) / "hang-upload-pack"
        hang.write_text("#!/bin/sh\nexec sleep 60\n")
        hang.chmod(0o755)
        git(clone, "config", "remote.origin.uploadpack", str(hang))
        env = {
            "GITHUB_EVENT_NAME": "pull_request",
            "BASE_SHA": self.base_sha,
            "JOPE_PROVENANCE_FETCH_TIMEOUT": "1",
        }
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "timed out after 1s"):
            bind.ensure_bind_rev(clone, env)
        self.assertLess(time.monotonic() - started, 30)
        self.assertFalse(bind.has_commit(clone, self.base_sha))

    def test_fetch_timeout_default_and_bad_overrides(self):
        self.assertEqual(bind.fetch_timeout({}), float(bind.FETCH_TIMEOUT_SECONDS))
        self.assertEqual(bind.fetch_timeout({"JOPE_PROVENANCE_FETCH_TIMEOUT": "30"}), 30.0)
        for raw in ("0", "-5", "abc", "nan", "inf"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "JOPE_PROVENANCE_FETCH_TIMEOUT"):
                    bind.fetch_timeout({"JOPE_PROVENANCE_FETCH_TIMEOUT": raw})

    def test_fetch_timeout_upper_cap_fails_closed(self):
        cap = bind.FETCH_TIMEOUT_MAX_SECONDS
        self.assertEqual(
            bind.fetch_timeout({"JOPE_PROVENANCE_FETCH_TIMEOUT": str(cap)}), float(cap)
        )
        for raw in (str(cap + 1), "600.5", "1e9"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "exceeds the 600s cap"):
                    bind.fetch_timeout({"JOPE_PROVENANCE_FETCH_TIMEOUT": raw})

    def test_stale_shallow_lock_is_removed_then_fetch_succeeds(self):
        clone = self._shallow_clone()
        lock = clone / ".git" / "shallow.lock"
        lock.write_text("")
        env = {"GITHUB_EVENT_NAME": "pull_request", "BASE_SHA": self.base_sha}
        self.assertEqual(bind.ensure_bind_rev(clone, env), self.base_sha)
        self.assertFalse(lock.exists())
        self.assertTrue(bind.has_commit(clone, self.base_sha))

    def test_held_shallow_lock_fails_closed(self):
        clone = self._shallow_clone()
        lock = clone / ".git" / "shallow.lock"
        lock.write_text("")
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys,time; f=open(sys.argv[1]); print('held', flush=True); time.sleep(60)",
                str(lock),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "held")
            env = {"GITHUB_EVENT_NAME": "pull_request", "BASE_SHA": self.base_sha}
            with self.assertRaisesRegex(ValueError, f"held by {holder.pid} "):
                bind.ensure_bind_rev(clone, env)
            # Refused before fetching: the lock is untouched and the base is still absent.
            self.assertTrue(lock.exists())
            self.assertFalse(bind.has_commit(clone, self.base_sha))
        finally:
            holder.kill()
            holder.wait()

    def test_uninspectable_proc_fails_closed(self):
        clone = self._shallow_clone()
        (clone / ".git" / "shallow.lock").write_text("")
        missing = Path(self.tmp.name) / "no-proc"
        with self.assertRaisesRegex(ValueError, "cannot inspect processes"):
            bind.clear_stale_shallow_lock(clone, proc_root=missing)

    def test_no_lock_is_a_no_op(self):
        clone = self._shallow_clone()
        self.assertFalse(bind.clear_stale_shallow_lock(clone))

    def test_digest_rev_validation(self):
        sha = self.base_sha
        self.assertIsNone(bind.validate_digest_rev("WORKTREE"))
        for good in (sha, "main", "HEAD", "origin/main", "refs/heads/ops/x-1.2"):
            with self.subTest(good=good):
                self.assertEqual(bind.validate_digest_rev(good), good)
        bad = [
            "",
            "-n",
            "--output=/tmp/x",
            "HEAD~1",
            "HEAD^",
            "main:deploy",
            "main@{1}",
            "a..b",
            "a//b",
            "main.lock",
            "main/",
            "has space",
            "-" + sha,
            sha + "\n",
            "x" * 201,
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "digest REV"):
                    bind.validate_digest_rev(value)

    def test_digest_cli_rejects_bad_rev_and_extra_args(self):
        script = ROOT / "deploy/runner/provenance_bind.py"
        env = dict(os.environ, GITHUB_WORKSPACE=str(self.repo))
        ok = subprocess.run(
            [sys.executable, str(script), "digest", self.base_sha],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(ok.stdout.strip(), self.base_digest)
        for argv in (["digest", "--output=x"], ["digest", "HEAD~1"], ["digest", "main", "extra"]):
            with self.subTest(argv=argv):
                bad = subprocess.run(
                    [sys.executable, str(script), *argv],
                    env=env, capture_output=True, text=True,
                )
                self.assertEqual(bad.returncode, 1)
                self.assertEqual(bad.stdout, "")

    def test_parse_matches_repo_common_sh(self):
        common = (ROOT / "deploy/runner/common.sh").read_text()
        paths = bind.parse_deployed_relpaths(common)
        self.assertGreaterEqual(len(paths), 10)
        live = bind.scripts_digest(ROOT, None)
        bash = subprocess.check_output(
            ["bash", "-ec", BASH_DIGEST], cwd=ROOT, text=True
        ).strip()
        self.assertEqual(live, bash)


if __name__ == "__main__":
    unittest.main()
