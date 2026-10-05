"""CPU-only controls; synthetic objects never establish live GPU/S3 qualification."""

import concurrent.futures
import copy
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gpu_gate as gate
import outputs as out

NOW = dt.datetime(2026, 10, 5, 18, tzinfo=dt.timezone.utc)


class Scratch(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[2] / "target" / "dl-trainonly-tests"
        target.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)


class BudgetTests(Scratch):
    def setUp(self):
        super().setUp()
        self.path = self.root / "budget.sqlite"
        self.budget = gate.Budget(self.path, initialize=True)

    def job(self, identity="a", host="xbabe2", hours=1):
        return {
            "job_id": identity,
            "host": host,
            "gpu_seconds": hours * 3600,
            "output_ids": [out.digest(identity.encode())],
        }

    def reserve(self, job, **kw):
        return self.budget.reserve(job, now=NOW, **kw)

    def test_all_three_hosts_valid_but_execution_unadmitted(self):
        for host in sorted(gate.HOSTS):
            result = self.reserve(self.job(host, host))
            self.assertTrue(result["budget_reserved"])
            self.assertFalse(result["execution_admitted"])

    def test_wrong_hosts_refused(self):
        for host in (
            "xbabe0",
            "xbabe4",
            "localhost",
            "xbabe3.evil",
            "XBABE2",
            "xbabe2 ",
        ):
            with self.subTest(host=host), self.assertRaises(gate.Refusal):
                self.reserve(self.job(host=host))

    def test_plan_boundary_and_contingency(self):
        self.reserve(self.job(hours=16))
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("b", hours=1))
        result = self.reserve(self.job("b", hours=8), contingency=True)
        self.assertEqual(result["charged_gpu_seconds"], 24 * 3600)
        self.assertEqual(result["contingency_seconds"], 8 * 3600)
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("c"), contingency=True)

    def test_single_job_over_hard_cap(self):
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job(hours=25), contingency=True)

    def test_retry_restart_never_refunds(self):
        request = self.job(hours=16)
        self.reserve(request)
        self.budget.finish("a", 1, "cancelled")
        self.budget = gate.Budget(self.path)
        with self.assertRaisesRegex(gate.Refusal, "terminal job"):
            self.reserve(request)
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("b"))

    def test_terminal_replay_refused_for_every_outcome(self):
        for outcome in ("complete", "failed", "cancelled", "unknown"):
            request = self.job(outcome)
            self.reserve(request)
            self.budget.finish(outcome, 1, outcome)
            with self.assertRaisesRegex(gate.Refusal, "terminal job"):
                self.reserve(request)

    def test_duplicate_delivery_and_dispatch_are_separate(self):
        request = self.job()
        self.assertTrue(self.reserve(request)["new_reservation"])
        repeated = self.reserve(request)
        self.assertFalse(repeated["new_reservation"])
        self.assertEqual(repeated["charged_gpu_seconds"], 3600)
        result = self.budget.claim_dispatch(request, now=NOW)
        self.assertTrue(result["dispatch_claimed"])
        self.assertFalse(result["execution_admitted"])
        with self.assertRaisesRegex(gate.Refusal, "already claimed"):
            gate.Budget(self.path).claim_dispatch(request, now=NOW)

    def test_two_attempts_include_failed_and_unknown_restarts(self):
        original = self.job()
        self.reserve(original)
        self.budget.finish("a", 1, "failed")
        retry = original | {"job_id": "retry"}
        result = gate.Budget(self.path).reserve(retry, now=NOW)
        self.assertEqual(result["charged_gpu_seconds"], 7200)
        self.budget.finish("retry", 0, "unknown")
        with self.assertRaisesRegex(gate.Refusal, "two-attempt"):
            gate.Budget(self.path).reserve(original | {"job_id": "third"}, now=NOW)

    def test_concurrent_output_attempt_cap(self):
        original = self.job()
        self.reserve(original)

        def attempt(identity):
            try:
                gate.Budget(self.path).reserve(original | {"job_id": identity}, now=NOW)
                return True
            except gate.Refusal:
                return False

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(attempt, ("b", "c"))), [False, True])

    def test_batched_attempt_reservation_rolls_back(self):
        original = self.job()
        self.reserve(original)
        self.reserve(original | {"job_id": "second"})
        fresh = self.job("fresh")
        with self.assertRaises(gate.Refusal):
            self.reserve(
                fresh | {"output_ids": fresh["output_ids"] + original["output_ids"]}
            )
        with self.budget.connect() as db:
            self.assertEqual(
                db.execute(
                    "SELECT count(*) FROM attempts WHERE output_id=?",
                    (fresh["output_ids"][0],),
                ).fetchone()[0],
                0,
            )
        self.assertTrue(self.reserve(fresh)["new_reservation"])

    def test_output_identity_and_dispatch_scope(self):
        original = self.job()
        for ids in ([], ["short"], [original["output_ids"][0]] * 2, "not-list", [None]):
            with self.assertRaises(gate.Refusal):
                self.reserve(original | {"output_ids": ids})
        with self.assertRaises(gate.Refusal):
            self.budget.claim_dispatch(original, now=NOW)
        self.reserve(original)
        with self.assertRaises(gate.Refusal):
            self.reserve(original | {"output_ids": ["f" * 64]})
        for request, now in (
            (original | {"host": "xbabe3"}, NOW),
            (original, gate.DEADLINE),
        ):
            with self.assertRaises(gate.Refusal):
                self.budget.claim_dispatch(request, now=now)

    def test_changed_retry_and_terminal_refused(self):
        self.reserve(self.job())
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job(host="xbabe3"))
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job(hours=2))
        self.budget.finish("a", 3, "failed")
        self.budget.finish("a", 3, "failed")
        with self.assertRaises(gate.Refusal):
            self.budget.finish("a", 0, "complete")

    def test_overrun_retained_and_blocks_new_jobs(self):
        self.reserve(self.job())
        self.budget.finish("a", 25 * 3600, "failed")
        with self.assertRaises(gate.Refusal):
            gate.Budget(self.path).reserve(self.job("b"), contingency=True, now=NOW)

    def test_expired_crossing_and_naive_clock(self):
        for now in (
            gate.DEADLINE,
            gate.DEADLINE - dt.timedelta(seconds=1),
            NOW.replace(tzinfo=None),
        ):
            with self.subTest(now=now), self.assertRaises(gate.Refusal):
                self.budget.reserve(self.job(), now=now)

    def test_finish_at_deadline(self):
        result = self.budget.reserve(
            self.job(), now=gate.DEADLINE - dt.timedelta(hours=1)
        )
        self.assertTrue(result["budget_reserved"])

    def test_invalid_seconds_and_id(self):
        for seconds in (0, -1, True, float("nan"), float("inf"), 1.5, "3600"):
            with self.subTest(seconds=seconds), self.assertRaises(gate.Refusal):
                self.reserve(self.job() | {"gpu_seconds": seconds})
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job("../escape"))
        with self.assertRaises(gate.Refusal):
            self.reserve(self.job() | {"deadline": "2027-01-01"})

    def test_concurrent_reservations(self):
        # Separate SQLite connections race for 20 hours in a 16-hour plan.
        def attempt(identity):
            try:
                return gate.Budget(self.path).reserve(
                    self.job(identity, hours=10), now=NOW
                )["budget_reserved"]
            except gate.Refusal:
                return False

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(attempt, ("a", "b"))), [False, True])

    def test_missing_ledger_never_resets_charges(self):
        self.reserve(self.job(hours=16))
        saved = self.path.read_bytes()
        self.path.unlink()
        with self.assertRaisesRegex(gate.Refusal, "restore custody"):
            gate.Budget(self.path)
        self.assertFalse(self.path.exists())
        with self.assertRaises(FileExistsError):
            gate.Budget(self.path, initialize=True)
        self.assertFalse(self.path.exists())
        self.path.write_bytes(saved)
        with self.assertRaises(gate.Refusal):
            gate.Budget(self.path).reserve(self.job("b"), now=NOW)

    def test_first_initialization_is_explicit_and_exclusive(self):
        path = self.root / "another.sqlite"
        with self.assertRaisesRegex(gate.Refusal, "restore custody"):
            gate.Budget(path)
        self.assertFalse(path.exists())
        budget = gate.Budget(path, initialize=True)
        budget.reserve(self.job(), now=NOW)
        with self.assertRaises(FileExistsError):
            gate.Budget(path, initialize=True)
        self.assertEqual(
            gate.Budget(path).reserve(self.job(), now=NOW)["charged_gpu_seconds"], 3600
        )

    def test_missing_after_open_never_recreates_database(self):
        self.path.unlink()
        with self.assertRaises(sqlite3.OperationalError):
            self.reserve(self.job())
        self.assertFalse(self.path.exists())

    def test_missing_or_replaced_marker_refuses_resume(self):
        marker = self.path.with_name(self.path.name + ".initialization.json")
        saved = marker.read_bytes()
        marker.unlink()
        with self.assertRaisesRegex(gate.Refusal, "restore custody"):
            gate.Budget(self.path)
        marker.write_text("different plan\n")
        with self.assertRaisesRegex(gate.Refusal, "marker differs"):
            gate.Budget(self.path)
        marker.write_bytes(saved)
        self.assertTrue(
            gate.Budget(self.path).reserve(self.job(), now=NOW)["new_reservation"]
        )

    def test_changed_persisted_policy_refused(self):
        with self.budget.connect() as db:
            db.execute("UPDATE policy SET value='renewed'")
        with self.assertRaises(gate.Refusal):
            gate.Budget(self.path)

    def test_unreserved_finish_and_bad_outcome(self):
        with self.assertRaises(gate.Refusal):
            self.budget.finish("missing", 0, "failed")
        self.reserve(self.job())
        with self.assertRaises(gate.Refusal):
            self.budget.finish("a", 0, "passed")


def context(i=0):
    return {
        "dataset_hash": f"{i:016x}",
        "fold": 0,
        "view_id": f"context-{i}",
        "context_sha256": out.digest(str(i).encode()),
        "outer_train_indices_sha256": "a" * 64,
        "outer_test_indices_sha256": "b" * 64,
    }


def pins():
    return {
        "schema": "jope.dl-encoder-pins/v1",
        "encoders": {
            c: {
                "research_only": c == "tabpfn35",
                "assets": {"synthetic": {"sha256": "c" * 64, "bytes": 1}},
            }
            for c in out.COMPONENTS
        },
    }


def bindings(i=0):
    return {
        f"context-{i}": {
            k: "d" * 64
            for k in (
                "scaler_sha256",
                "code_sha256",
                "train_input_sha256",
                "recipe_sha256",
            )
        }
    }


class OutputTests(Scratch):
    def manifest(self, bound=True):
        return out.generate(
            [context()],
            pins(),
            source_sha256="e" * 64,
            bindings=bindings() if bound else None,
            expected_count=1,
        )

    def store(self, raw):
        reference = {"sha256": out.digest(raw), "bytes": len(raw)}
        (self.root / reference["sha256"]).write_bytes(raw)
        return reference

    def events(self, manifest):
        events = []
        for row in manifest["outputs"]:
            obj = self.store(b"synthetic-array:" + row["component"].encode()) | {
                "bucket": out.BUCKET,
                "key": row["object_key"],
                "version_id": "object-version",
            }
            receipt = {
                "schema": "jope.dl-output-receipt/v1",
                "manifest_sha256": out.digest(out.canonical(manifest)),
                "expectation": row,
                "object": obj,
                "if_none_match": "*",
            }
            ref = self.store(out.canonical(receipt)) | {
                "bucket": out.BUCKET,
                "key": row["receipt_key"],
                "version_id": "receipt-version",
            }
            events.append(
                {
                    "output_id": row["output_id"],
                    "outcome": "complete",
                    "object": obj,
                    "receipt": ref,
                }
            )
        return events

    def test_full_inventory_six_encoders_quarantines_tabpfn(self):
        manifest = out.generate(
            [context(i) for i in range(4873)], pins(), source_sha256="e" * 64
        )
        self.assertEqual(manifest["encoder_outputs"], 29238)
        self.assertEqual(
            sum(r["component"] == "dope" for r in manifest["outputs"]), 4873
        )
        self.assertFalse(manifest["execution_admitted"])
        self.assertTrue(
            all(
                not r["production_loader_allowed"]
                for r in manifest["outputs"]
                if r["component"] == "tabpfn35"
            )
        )

    def test_count_duplicate_fold_and_bad_sha(self):
        with self.assertRaises(ValueError):
            out.generate([context()], pins(), source_sha256="e" * 64)
        with self.assertRaises(ValueError):
            out.generate(
                [context(), context() | {"view_id": "alias"}],
                pins(),
                source_sha256="e" * 64,
                expected_count=2,
            )
        with self.assertRaises(ValueError):
            out.generate(
                [context() | {"context_sha256": "short"}],
                pins(),
                source_sha256="e" * 64,
                expected_count=1,
            )

    def test_create_only_refuses_existing_and_symlink(self):
        path = self.root / "manifest.json"
        out.create_only(path, b"original")
        with self.assertRaises(FileExistsError):
            out.create_only(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")
        link = self.root / "link"
        link.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            out.create_only(link, b"write")

    def test_authorized_complete_custody(self):
        manifest = self.manifest()
        result = out.check(manifest, self.events(manifest), self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["counts"]["complete"], 6)
        self.assertFalse(result["execution_admitted"])

    def test_research_prefix_is_owner_scoped_and_rebinding_refused(self):
        manifest = self.manifest()
        self.assertEqual(out.RESEARCH_PREFIX, out.PREFIX + "research-only/")
        row = next(r for r in manifest["outputs"] if r["component"] == "tabpfn35")
        for product in manifest["outputs"]:
            for field in ("object_key", "receipt_key", "key_template"):
                self.assertTrue(product[field].startswith(out.PREFIX))
                self.assertEqual(
                    product[field].startswith(out.PREFIX + "research-only/"),
                    product["component"] == "tabpfn35",
                )
        row["object_key"] = row["object_key"].replace(out.RESEARCH_PREFIX, out.PREFIX)
        with self.assertRaises(ValueError):
            out.check(manifest, [], self.root)

    def test_legacy_sibling_research_manifest_is_refused(self):
        manifest = self.manifest()
        manifest["research_prefix"] = "experiments/gp-policy-dl/v2-trainonly-research/"
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            out.check(manifest, [], self.root)

    def test_dope_pin_matches_owner_lane_binary(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "docs/internal/core-ip/dl-trainonly/encoder-pins.json"
        )
        release = json.loads(path.read_bytes())
        # Independently frozen lane-build identity, not the pin's own sidecar.
        expected = {
            "sha256": "bc3c446098cd3a080fc4ce2e2ff7b2f10bf9549fb08f053d4ca14af263b2d926",
            "bytes": 9337920,
        }
        self.assertEqual(release["encoders"]["dope"]["assets"]["binary"], expected)
        current = out.generate(
            [context()],
            release,
            source_sha256="e" * 64,
            bindings=bindings(),
            expected_count=1,
        )
        stale_pins = copy.deepcopy(release)
        stale_pins["encoders"]["dope"]["assets"]["binary"] = {
            "sha256": "c972c631c7c1053436ba31a088e223a9d394078bdb46148bdcc8f63e2487b5f8",
            "bytes": 9530432,
        }
        stale = out.generate(
            [context()],
            stale_pins,
            source_sha256="e" * 64,
            bindings=bindings(),
            expected_count=1,
        )
        ids = {r["component"]: r["output_id"] for r in current["outputs"]}
        stale_ids = {r["component"]: r["output_id"] for r in stale["outputs"]}
        self.assertNotEqual(ids["dope"], stale_ids["dope"])
        for component in out.COMPONENTS:
            if component != "dope":
                self.assertEqual(ids[component], stale_ids[component])
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate completion"):
            out.check(current, self.events(stale), self.root)

    def test_noncomplete_extra_fields_refused(self):
        manifest = self.manifest()
        with self.assertRaises(ValueError):
            out.check(
                manifest,
                [
                    {
                        "output_id": manifest["outputs"][0]["output_id"],
                        "outcome": "unknown",
                        "ignored_object": "extra",
                    }
                ],
                self.root,
            )

    def test_unbound_draft_cannot_complete(self):
        manifest = self.manifest(False)
        result = out.check(manifest, self.events(manifest), self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["invalid"]), 6)

    def test_failures_missing_and_unknown_stay_visible(self):
        manifest = self.manifest()
        events = self.events(manifest)
        for i, state in enumerate(("failed", "cancelled", "unknown")):
            events[i] = {"output_id": events[i]["output_id"], "outcome": state}
        result = out.check(manifest, events[:4], self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing"], 2)
        self.assertEqual(
            result["counts"], {"complete": 1, "failed": 1, "cancelled": 1, "unknown": 1}
        )

    def test_duplicates_and_unexpected_output_refused(self):
        manifest = self.manifest()
        events = self.events(manifest)
        with self.assertRaises(ValueError):
            out.check(manifest, events + events[:1], self.root)
        with self.assertRaises(ValueError):
            out.check(
                manifest, [{"output_id": "f" * 64, "outcome": "failed"}], self.root
            )

    def test_corrupt_truncated_missing_symlink_objects(self):
        for corruption in ("corrupt", "truncated", "missing", "symlink"):
            with self.subTest(corruption=corruption):
                manifest = self.manifest()
                events = self.events(manifest)
                ref = events[0]["object"]
                path = self.root / ref["sha256"]
                if corruption == "corrupt":
                    path.write_bytes(b"x" * ref["bytes"])
                elif corruption == "truncated":
                    path.write_bytes(b"x")
                elif corruption == "missing":
                    path.unlink()
                else:
                    target = self.root / "outside"
                    target.write_bytes(path.read_bytes())
                    path.unlink()
                    path.symlink_to(target)
                result = out.check(manifest, events, self.root)
                self.assertFalse(result["complete"])
                self.assertEqual(len(result["invalid"]), 1)
                if path.is_symlink():
                    path.unlink()

    def test_fifo_refused_without_blocking(self):
        # The original open(O_RDONLY) blocked before reaching fstat. Bound the
        # receiving-path replay so a regression fails without hanging the suite.
        ref = {"sha256": "f" * 64, "bytes": 1}
        os.mkfifo(self.root / ref["sha256"])
        script = """import json, sys, outputs
try:
    outputs.blob(sys.argv[1], json.loads(sys.argv[2]))
except ValueError as error:
    assert str(error) == 'blob must be a regular file'
    print('REFUSED_NONREGULAR')
else:
    raise AssertionError('FIFO accepted')
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root), json.dumps(ref)],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "REFUSED_NONREGULAR")

    def test_wrong_key_bucket_and_latest_version(self):
        for field, value in (
            ("key", "staging/old"),
            ("bucket", "other"),
            ("version_id", "latest"),
            ("version_id", "null"),
            ("version_id", ""),
        ):
            with self.subTest(field=field, value=value):
                manifest = self.manifest()
                events = self.events(manifest)
                events[0]["object"][field] = value
                self.assertEqual(
                    len(out.check(manifest, events, self.root)["invalid"]), 1
                )

    def test_receipt_swaps_wrong_inputs_and_unconditional_write(self):
        manifest = self.manifest()
        for change in ("swap", "context", "input", "conditional", "manifest"):
            with self.subTest(change=change):
                events = self.events(manifest)
                if change == "swap":
                    events[0]["receipt"] = events[1]["receipt"]
                else:
                    ref = events[0]["receipt"]
                    rec = json.loads((self.root / ref["sha256"]).read_bytes())
                    if change == "context":
                        rec["expectation"]["context"]["fold"] = 99
                    elif change == "input":
                        rec["expectation"]["input_binding"]["train_input_sha256"] = (
                            "f" * 64
                        )
                    elif change == "conditional":
                        rec["if_none_match"] = None
                    else:
                        rec["manifest_sha256"] = "f" * 64
                    events[0]["receipt"] = ref | self.store(out.canonical(rec))
                self.assertEqual(
                    len(out.check(manifest, events, self.root)["invalid"]), 1
                )

    def test_altered_counts_pin_row_and_precondition_refused(self):
        manifest = self.manifest()
        for change in ("count", "row", "pin", "precondition", "omit"):
            altered = copy.deepcopy(manifest)
            if change == "count":
                altered["encoder_outputs"] -= 1
            elif change == "row":
                altered["outputs"][0]["encoder_pin_sha256"] = "f" * 64
            elif change == "pin":
                altered["pins"]["encoders"]["target"]["assets"]["synthetic"][
                    "sha256"
                ] = "f" * 64
            elif change == "precondition":
                altered["publication_precondition"]["unconditional_fallback"] = True
            else:
                altered["outputs"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                out.check(altered, [], self.root)

    def test_input_and_model_changes_produce_new_identity(self):
        original = self.manifest()
        changed = bindings()
        changed["context-0"]["scaler_sha256"] = "f" * 64
        updated = out.generate(
            [context()],
            pins(),
            source_sha256="e" * 64,
            bindings=changed,
            expected_count=1,
        )
        self.assertNotEqual(
            original["outputs"][0]["output_id"], updated["outputs"][0]["output_id"]
        )
        changed_pins = pins()
        changed_pins["encoders"]["hyperion"]["assets"]["synthetic"]["sha256"] = "f" * 64
        updated = out.generate(
            [context()],
            changed_pins,
            source_sha256="e" * 64,
            bindings=bindings(),
            expected_count=1,
        )
        self.assertNotEqual(
            original["outputs"][0]["output_id"], updated["outputs"][0]["output_id"]
        )

        def by_component(manifest):
            return {r["component"]: r["output_id"] for r in manifest["outputs"]}

        self.assertNotEqual(
            by_component(original)["context2048"], by_component(updated)["context2048"]
        )
        changed_pins["encoders"]["tabpfn35"]["research_only"] = False
        with self.assertRaises(ValueError):
            out.generate(
                [context()], changed_pins, source_sha256="e" * 64, expected_count=1
            )

    def test_extra_or_partial_bindings_refused(self):
        for b in (bindings(1), {"context-0": {"scaler_sha256": "d" * 64}}):
            with self.assertRaises(ValueError):
                out.generate(
                    [context()],
                    pins(),
                    source_sha256="e" * 64,
                    bindings=b,
                    expected_count=1,
                )

    def test_pinned_input_replacement_refused(self):
        path = self.root / "input"
        path.write_bytes(b"original")
        sha = out.digest(path.read_bytes())
        self.assertEqual(out.pinned_file(path, sha), b"original")
        path.write_bytes(b"replacement")
        with self.assertRaises(ValueError):
            out.pinned_file(path, sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
