import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import experiment_runtime as runtime
import test_kernel_release as release_tests
from kernel_bundle import DOCUMENTS, pack, verify_bundle
from kernel_remote_publish import publish
from kernel_remote_state import compare, compare_health
from kernel_research_check import screening
from kernel_verify import Deferred, monitor


class BundleTests(unittest.TestCase):
    def setUp(self):
        release_tests.KernelReleaseTests.setUp(self)
        for name in (
            *DOCUMENTS,
            "Cargo.lock",
            "research/algorithm-kernel-regimes/fixture.json",
            "docs/audits/algorithm-kernel-upgrade-2026-09-08/evidence.json",
        ):
            p = self.base / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}\n")

    def tearDown(self):
        release_tests.KernelReleaseTests.tearDown(self)

    def test_roundtrip_and_extra_file_tampering(self):
        bundle = pack(self.base, self.base / "bundles")
        receipt = verify_bundle(bundle)
        self.assertFalse(receipt["activation_allowed"])
        self.assertIn("workspace/" + DOCUMENTS[-1], receipt["files"])
        self.assertEqual(pack(self.base, self.base / "bundles"), bundle)
        bundle.chmod(0o755)
        (bundle / "unexpected").write_text("must not be ignored")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_bundle(bundle)

    def test_changed_probe_creates_a_new_immutable_bundle(self):
        first = pack(self.base, self.base / "bundles")
        (self.base / "research/algorithm-kernel-regimes/fixture.json").write_text("changed")
        second = pack(self.base, self.base / "bundles")
        self.assertNotEqual(first, second)
        verify_bundle(first)
        verify_bundle(second)

    def test_verified_readonly_incoming_publishes_with_same_parent_rename(self):
        final = pack(self.base, self.base / "bundles")
        incoming = final.with_name(".incoming-test")
        final.rename(incoming)
        with self.assertRaises(ValueError):
            verify_bundle(incoming)
        expected = verify_bundle(incoming, require_name=False)
        publish(incoming, final, expected)
        publish(incoming, final, expected)  # Existing identical source is reused.
        verify_bundle(final)


class PreservationTests(unittest.TestCase):
    def test_normal_progress_is_allowed_but_restart_and_source_drift_are_not(self):
        before = {
            "stable": {"service": {"pid": 10, "invocation": "a"}, "kernel": "sha-a"},
            "telemetry": {"completed": 10},
        }
        after = copy.deepcopy(before)
        after["telemetry"]["completed"] = 11
        self.assertTrue(compare(before, after)["stable_identities_unchanged"])
        after["stable"]["service"]["invocation"] = "b"
        self.assertFalse(compare(before, after)["stable_identities_unchanged"])
        after = copy.deepcopy(before)
        after["stable"]["kernel"] = "sha-b"
        self.assertEqual(compare(before, after)["changed"], ["kernel"])

    def test_campaign_reclaim_kills_owned_subprocess_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            pidfile = Path(temporary) / "descendant"
            script = (
                "import subprocess,time; from pathlib import Path; "
                'p=subprocess.Popen(["' + sys.executable + '","-c","import time; time.sleep(120)"]); '
                "Path(" + repr(str(pidfile)) + ").write_text(str(p.pid)); time.sleep(120)"
            )
            child = runtime.popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL)
            try:
                limit = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < limit:
                    time.sleep(0.01)
                self.assertTrue(pidfile.exists())
                descendant = int(pidfile.read_text())
                with patch(
                    "kernel_verify.scheduler.admission_reason", return_value="mandatory training block"
                ):
                    with self.assertRaises(Deferred):
                        monitor(child, time.monotonic() + 30, Path(temporary))
                self.assertIsNotNone(child.poll())
                status = Path("/proc") / str(descendant) / "status"

                def still_running():
                    try:
                        return "\nState:\tZ" not in status.read_text()
                    except (FileNotFoundError, ProcessLookupError):
                        return False

                limit = time.monotonic() + 2
                while still_running() and time.monotonic() < limit:
                    time.sleep(0.01)
                self.assertFalse(still_running())
            finally:
                runtime.stop([child])

    def test_timeout_stops_owned_process(self):
        child = runtime.popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            with self.assertRaises(TimeoutError):
                monitor(child, time.monotonic() - 1)
            self.assertIsNotNone(child.poll())
        finally:
            runtime.stop([child])


class OperationalHealthTests(unittest.TestCase):
    def setUp(self):
        self.before = {
            "timestamp": 100,
            "stable": {"jope-parallel-gp": {"properties": {"InvocationID": "original"}}},
            "telemetry": {
                "gp-service.json": {
                    "timestamp": 100,
                    "phase": "running",
                    "active": 104,
                    "completed_views": 20,
                    "completed_evaluations": 200,
                    "failed_views": 0,
                    "queued": 20,
                },
                "cycle-progress.json": {"timestamp": 100},
                "hardware.json": {"timestamp": 100, "errors": []},
            },
        }
        import kernel_remote_state as k

        self.before.update(schema="kernel.live-state.v2", custody="live", capture_errors=[])
        for unit in k.UNITS:
            self.before["stable"].setdefault(unit, {}).update(
                definition="original definition",
                exe_sha256="a" * 64,
                properties=dict(
                    InvocationID="original",
                    MainPID="0" if unit in k.FENCED_UNITS[2:] else "123",
                    ActiveState="inactive" if unit in k.FENCED_UNITS[2:] else "active",
                ),
            )
        self.after = copy.deepcopy(self.before)
        self.after["timestamp"] = 104
        for record in self.after["telemetry"].values():
            record["timestamp"] = 103
        self.after["telemetry"]["gp-service.json"]["completed_views"] = 21

    def test_advancing_campaign_and_empty_queue_pass(self):
        result = compare_health(self.before, self.after)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["observations"]["completed_views_delta"], 1)
        self.after["telemetry"]["gp-service.json"].update(active=0, queued=0)
        self.assertEqual(compare_health(self.before, self.after)["status"], "passed")

    def test_failures_backwards_counters_and_stopped_dispatch_fail(self):
        for field, value in (
            ("failed_views", 1),
            ("completed_views", 19),
            ("completed_evaluations", 199),
            ("active", 0),
        ):
            with self.subTest(field=field):
                after = copy.deepcopy(self.after)
                after["telemetry"]["gp-service.json"][field] = value
                self.assertEqual(compare_health(self.before, after)["status"], "failed")
        self.after["telemetry"]["hardware.json"]["errors"] = ["sensor unavailable"]
        self.assertEqual(compare_health(self.before, self.after)["status"], "failed")

    def test_missing_stale_future_or_invalid_evidence_cannot_pass(self):
        for name in ("gp-service.json", "cycle-progress.json", "hardware.json"):
            for stamp in (None, 80, 120, float("nan"), True):
                with self.subTest(name=name, stamp=stamp):
                    after = copy.deepcopy(self.after)
                    after["telemetry"][name]["timestamp"] = stamp
                    self.assertEqual(compare_health(self.before, after)["status"], "inconclusive")
            after = copy.deepcopy(self.after)
            del after["telemetry"][name]
            self.assertEqual(compare_health(self.before, after)["status"], "inconclusive")
        for field, value in (("failed_views", True), ("active", -1), ("queued", "20"), ("phase", None)):
            after = copy.deepcopy(self.after)
            after["telemetry"]["gp-service.json"][field] = value
            self.assertEqual(compare_health(self.before, after)["status"], "inconclusive")
        self.after["timestamp"] = 99
        self.assertEqual(compare_health(self.before, self.after)["status"], "inconclusive")

    def test_unrelated_fresh_packer_cannot_mask_stale_cycle_progress(self):
        self.after["telemetry"]["cycle-progress.json"]["timestamp"] = 80
        self.assertEqual(compare_health(self.before, self.after)["status"], "inconclusive")
        for phase in ["packing", "complete"]:
            with self.subTest(phase=phase):
                self.after["telemetry"]["corpus-progress.json"] = {
                    "timestamp": 103,
                    "phase": phase,
                    "experiment": "an-unrelated-campaign",
                }
                self.assertEqual(compare_health(self.before, self.after)["status"], "inconclusive")

    def test_restart_does_not_turn_reset_counters_into_valid_evidence(self):
        self.after["stable"]["jope-parallel-gp"]["properties"]["InvocationID"] = "restarted"
        self.after["telemetry"]["gp-service.json"]["completed_views"] = 0
        result = compare_health(self.before, self.after)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["observations"], {})

    def test_presence_flags_cannot_certify_migration(self):
        fence = dict(legacy_dispatch_fenced=True, registry_dispatch_enabled=False)
        before = dict(
            schema="kernel.live-state.v1",
            custody="planned_migration",
            timestamp=100,
            stable={"fence": fence},
            telemetry={},
            fence=fence,
        )
        self.assertEqual(compare_health(before, before)["status"], "failed")


class ResearchCheckTests(unittest.TestCase):
    def test_missing_duplicate_and_changed_context_cannot_pass(self):
        repo = Path(__file__).resolve().parents[2]
        probes = repo / "research/algorithm-kernel-regimes"
        # The test is also packaged in an immutable parallel source release.
        if not probes.exists():
            self.skipTest("Development probe data belongs to the outer work bundle")
        manifest = probes / "screening-manifest-v1.json"
        reference = probes / "screening-probe-v1.json"
        original = json.loads(reference.read_text())
        self.assertEqual(screening(reference, manifest, reference)["results_compared"], 40)
        for mode in ("missing", "duplicate", "context", "nonfinite"):
            changed = copy.deepcopy(original)
            if mode == "missing":
                changed["cases"].pop()
            elif mode == "duplicate":
                changed["cases"][1] = changed["cases"][0]
            elif mode == "context":
                changed["manifest_identity"] = "wrong"
            else:
                changed["cases"][0]["models"][0]["mse"] = float("nan")
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "probe.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    screening(path, manifest, reference)


if __name__ == "__main__":
    unittest.main()
