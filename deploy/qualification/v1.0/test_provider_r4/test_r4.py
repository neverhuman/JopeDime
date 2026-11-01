"""Data-free R4 provider controls; synthetic receipts are not release evidence."""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "providers"))
r4 = importlib.import_module("r4")
common = importlib.import_module("common")
Context, ProviderError = common.Context, common.ProviderError


class R4Tests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("JOPE_QUALIFICATION_TEST_TMP", HERE / "target"))
        scratch.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="r4-control-", dir=scratch)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        source = self.repo / "crates/jope-appliance/tests/system/live_driver.rs"
        source.parent.mkdir(parents=True)
        source.write_text(
            'const DRIVER: &str = r####"pass"####;\n'
            'const OBSERVER: &str = r####"pass"####;\n'
        )
        self.artifact = self.root / "candidate"
        self.artifact.write_bytes(b"fixture provider candidate; not a release")
        self.ctx = Context(
            "R4",
            self.artifact,
            hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            "a" * 40,
            self.root / "result.json",
        )
        self.source = self.ctx.source_sha
        self.dirty = ""
        self.calls = []
        self.probe_calls = 0
        self.after_probe = lambda: None
        self.mutate = lambda files: None
        self.code = 0
        self.scope = "internal-restricted"
        self.profile = "full"
        self.verified = True
        self.inspect_source = self.source
        self.addCleanup(patch.stopall)
        patch.object(r4, "ROOT", self.repo).start()
        patch.object(common, "ROOT", self.repo).start()
        patch.object(r4, "RUNTIME_BASE", self.root / "runtime").start()
        patch.object(r4, "execute", self.probe).start()

    def child(self, args, **kwargs):
        self.calls.append(args)
        if args[:2] == ["git", "rev-parse"]:
            output = self.source
        elif args[:2] == ["git", "status"]:
            output = self.dirty
        elif "inspect" in args:
            output = json.dumps(
                {
                    "manifest": {
                        "profile": self.profile,
                        "scope": self.scope,
                        "source": {"sha40": self.inspect_source},
                    }
                }
            )
        elif "verify" in args:
            output = json.dumps(
                {"schema": "jope.appliance-verify/v1", "ok": self.verified}
            )
        else:
            self.fail(f"unexpected child: {args}")
        return subprocess.CompletedProcess(args, 0, output, "")

    def probe(self, args, env):
        self.probe_calls += 1
        self.calls.append(args)
        self.assertIn("--r4-only", args)
        self.assertNotIn("--all-live", args)
        self.assertEqual(args[args.index("--binary") + 1], str(self.artifact))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        evidence = Path(args[args.index("--out") + 1])
        events = [{"run_id": "run_fixture", "sequence": str(i)} for i in [1, 2]]
        report = {
            "source": self.ctx.source_sha,
            "binary_sha256": self.ctx.artifact_sha256,
            "version": {"source_sha40": self.ctx.source_sha},
            "cleanup_errors": [],
            "projection_oracle_controls": True,
            "r4_qualified": True,
            "mcp_sequences_contiguous": True,
            "mcp_equals_cli_events": True,
            "terminal_event_projection_observed": True,
            "terminal_champion_render_matches_mcp_snapshot": True,
            "state_at_close": "running",
            "terminal_first": {"exit": 0},
            "terminal_reattach": {"exit": 0},
            "r4_final": {
                "state": "completed",
                "run_id": "run_fixture",
                "event_watermark": "2",
                "best": {"algorithm_id": "alg_fixture", "cv_value": 0.125},
            },
        }
        files = {
            "report.json": report,
            "live-mcp-events.json": events,
            "live-cli-events.json": copy.deepcopy(events),
            "terminal-sse-streams.json": [{"events": events[:1]}, {"events": events}],
        }
        self.mutate(files)
        for name, value in files.items():
            (evidence / name).write_text(json.dumps(value))
        (evidence / "terminal-after-reattach.pty").write_text("alg_fixture 0.125")
        self.after_probe()
        return subprocess.CompletedProcess(
            args, self.code, "controlled probe output", ""
        )

    def invoke(self):
        return r4.qualify(self.ctx, run=self.child)

    def refused(self):
        with self.assertRaises(
            (ProviderError, OSError, ValueError, KeyError, TypeError)
        ):
            self.invoke()
        self.assertFalse(self.ctx.result.exists())

    def test_complete_live_observations_emit_three_candidate_bound_assertions(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads(self.ctx.result.read_text())
        self.assertEqual(
            result,
            {
                "schema": "jope.rel07.gate-execution/v1",
                "id": "R4",
                "status": "PASS",
                "source_sha": self.ctx.source_sha,
                "artifact_sha256": self.ctx.artifact_sha256,
                "counts": r4.COUNTS,
            },
        )
        self.assertEqual(self.probe_calls, 1)

    def test_context_refuses_missing_binding_wrong_gate_hash_and_short_source(self):
        env = {
            "JOPE_REL07_GATE_ID": "R4",
            "JOPE_REL07_ARTIFACT": str(self.artifact),
            "JOPE_REL07_ARTIFACT_SHA256": self.ctx.artifact_sha256,
            "JOPE_REL07_SOURCE_SHA": self.ctx.source_sha,
            "JOPE_REL07_RESULT": str(self.ctx.result),
        }
        for change in [
            {"JOPE_REL07_GATE_ID": "S2"},
            {"JOPE_REL07_ARTIFACT": ""},
            {"JOPE_REL07_ARTIFACT_SHA256": "0" * 64},
            {"JOPE_REL07_SOURCE_SHA": "a" * 7},
        ]:
            with (
                self.subTest(change=change),
                patch.dict(os.environ, env | change, clear=True),
            ):
                with self.assertRaises(ProviderError):
                    common.context("R4")

    def test_symlink_or_nonfile_candidate_never_executes_probe(self):
        from dataclasses import replace

        link = self.root / "linked-candidate"
        link.symlink_to(self.artifact)
        for path in [link, self.repo]:
            self.ctx = replace(self.ctx, artifact=path)
            self.refused()
        self.assertEqual(self.probe_calls, 0)

    def test_development_nonfull_unverified_or_wrong_source_package_refuses(self):
        for key, value in [
            ("scope", "development"),
            ("profile", "lite"),
            ("verified", False),
            ("inspect_source", "b" * 40),
        ]:
            before = getattr(self, key)
            setattr(self, key, value)
            with self.subTest(key=key):
                self.refused()
            setattr(self, key, before)
        self.assertEqual(self.probe_calls, 0)

    def test_wrong_or_dirty_checkout_refuses_before_probe(self):
        for source, dirty in [("b" * 40, ""), (self.source, " M tracked.rs")]:
            self.source, self.dirty = source, dirty
            self.refused()
        self.assertEqual(self.probe_calls, 0)

    def test_failed_probe_returns_actual_exit_without_passing_result(self):
        self.code = 42
        self.assertEqual(self.invoke(), 42)
        self.assertFalse(self.ctx.result.exists())
        self.assertEqual(
            self.ctx.result.with_name("result.json.r4")
            .joinpath("stdout.log")
            .read_text(),
            "controlled probe output",
        )

    def test_artifact_or_source_drift_after_probe_refuses(self):
        self.after_probe = lambda: self.artifact.write_bytes(b"changed")
        self.refused()

    def test_source_drift_after_probe_refuses(self):
        self.after_probe = lambda: setattr(self, "source", "b" * 40)
        self.refused()

    def test_all_required_report_checks_fail_closed(self):
        # Test report admission independently without reusing a successful result.
        self.assertEqual(self.invoke(), 0)
        evidence = self.ctx.result.with_name("result.json.r4")
        report_path = evidence / "report.json"
        original = json.loads(report_path.read_text())
        mutations = [
            {"source": "b" * 40},
            {"binary_sha256": "0" * 64},
            {"version": {"source_sha40": "b" * 40}},
            {"error": "child failed"},
            {"cleanup_errors": ["still running"]},
            {"state_at_close": "completed"},
            {"terminal_first": {"exit": 1}},
            {"terminal_reattach": {"exit": 1}},
        ] + [
            {key: False}
            for key in [
                "projection_oracle_controls",
                "r4_qualified",
                "mcp_sequences_contiguous",
                "mcp_equals_cli_events",
                "terminal_event_projection_observed",
                "terminal_champion_render_matches_mcp_snapshot",
            ]
        ]
        for mutation in mutations:
            report_path.write_text(json.dumps(original | mutation))
            with self.subTest(mutation=mutation), self.assertRaises(ProviderError):
                r4.assertions(self.ctx, evidence)

    def test_empty_gapped_duplicate_and_other_run_events_cannot_pass(self):
        self.assertEqual(self.invoke(), 0)
        evidence = self.ctx.result.with_name("result.json.r4")
        for events in [
            [],
            [{"run_id": "run_fixture", "sequence": "2"}],
            [{"run_id": "run_fixture", "sequence": "1"}] * 2,
            [{"run_id": "wrong", "sequence": str(i)} for i in [1, 2]],
        ]:
            for name in ["live-mcp-events.json", "live-cli-events.json"]:
                (evidence / name).write_text(json.dumps(events))
            with self.subTest(events=events), self.assertRaises(ProviderError):
                r4.assertions(self.ctx, evidence)

    def test_terminal_mcp_and_cli_must_match_actual_retained_events(self):
        self.mutate = lambda files: files["terminal-sse-streams.json"][1].update(
            events=[]
        )
        self.refused()

    def test_cli_mismatch_cannot_be_hidden_by_report_pass(self):
        self.mutate = lambda files: files.update({"live-cli-events.json": []})
        self.refused()

    def test_wrong_final_watermark_and_champion_screen_refuse(self):
        self.assertEqual(self.invoke(), 0)
        evidence = self.ctx.result.with_name("result.json.r4")
        report_path = evidence / "report.json"
        original = json.loads(report_path.read_text())
        changed = copy.deepcopy(original)
        changed["r4_final"]["event_watermark"] = "3"
        report_path.write_text(json.dumps(changed))
        with self.assertRaises(ProviderError):
            r4.assertions(self.ctx, evidence)
        report_path.write_text(json.dumps(original))
        (evidence / "terminal-after-reattach.pty").write_text("wrong champion")
        with self.assertRaises(ProviderError):
            r4.assertions(self.ctx, evidence)

    def test_missing_malformed_or_symlink_evidence_cannot_pass(self):
        self.assertEqual(self.invoke(), 0)
        evidence = self.ctx.result.with_name("result.json.r4")
        target = evidence / "report.json"
        raw = target.read_bytes()
        for replacement in [b"{bad", b"[]"]:
            target.write_bytes(replacement)
            with self.assertRaises((ValueError, AttributeError)):
                r4.assertions(self.ctx, evidence)
        target.unlink()
        with self.assertRaises(ProviderError):
            r4.assertions(self.ctx, evidence)
        other = self.root / "elsewhere.json"
        other.write_bytes(raw)
        target.symlink_to(other)
        with self.assertRaises(ProviderError):
            r4.assertions(self.ctx, evidence)

    def test_existing_result_and_probe_evidence_are_not_overwritten(self):
        evidence = self.ctx.result.with_name("result.json.r4")
        evidence.mkdir()
        sentinel = evidence / "prior"
        sentinel.write_text("retained")
        self.refused()
        self.assertEqual(sentinel.read_text(), "retained")
        self.ctx.result.write_text("prior result")
        with self.assertRaises(ProviderError):
            self.invoke()
        self.assertEqual(self.ctx.result.read_text(), "prior result")

    def test_evidence_cannot_resolve_into_source_checkout(self):
        from dataclasses import replace

        self.ctx = replace(
            self.ctx, result=self.root / "runtime/../checkout/result.json"
        )
        self.refused()
        self.assertEqual(self.probe_calls, 0)


class ProcessCleanup(unittest.TestCase):
    def test_timeout_stops_only_the_new_probe_process_group(self):
        child = Mock(pid=23456, returncode=-signal.SIGTERM)
        child.communicate.side_effect = [
            subprocess.TimeoutExpired(["probe"], 0.1),
            ("partial", "timeout"),
        ]
        process = Mock()
        process.__enter__ = Mock(return_value=child)
        process.__exit__ = Mock(return_value=False)
        with (
            patch.object(subprocess, "Popen", return_value=process) as start,
            patch.object(os, "killpg") as stop,
        ):
            result = r4.execute(["probe"], {}, timeout=0.1)
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.stdout, "partial")

    def test_timeout_escalates_only_its_probe_group(self):
        child = Mock(pid=23456)
        child.communicate.side_effect = [
            subprocess.TimeoutExpired(["probe"], 0.1),
            subprocess.TimeoutExpired(["probe"], 5),
            ("partial", ""),
        ]
        process = Mock()
        process.__enter__ = Mock(return_value=child)
        process.__exit__ = Mock(return_value=False)
        with (
            patch.object(subprocess, "Popen", return_value=process),
            patch.object(os, "killpg") as stop,
        ):
            self.assertEqual(r4.execute(["probe"], {}, timeout=0.1).returncode, 124)
        self.assertEqual(
            stop.call_args_list,
            [
                unittest.mock.call(child.pid, signal.SIGTERM),
                unittest.mock.call(child.pid, signal.SIGKILL),
            ],
        )

    def test_interruption_stops_owned_group_and_restores_handlers(self):
        previous = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        child = Mock(pid=23456)
        child.communicate.side_effect = [InterruptedError("cancelled"), ("partial", "")]
        process = Mock()
        process.__enter__ = Mock(return_value=child)
        process.__exit__ = Mock(return_value=False)
        with (
            patch.object(subprocess, "Popen", return_value=process),
            patch.object(os, "killpg") as stop,
        ):
            with self.assertRaises(InterruptedError):
                r4.execute(["probe"], {})
        stop.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual({sig: signal.getsignal(sig) for sig in previous}, previous)


if __name__ == "__main__":
    unittest.main()
