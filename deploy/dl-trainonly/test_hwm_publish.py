"""Synthetic receiving-path and failure-injection fixtures; no private input or AWS access."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hwm_publish as h


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.cid, self.cut, self.family = (
            "a" * 64,
            "trainonly-split-v2:" + "b" * 64,
            "fam-" + "c" * 64,
        )
        self.dsl = "(pipe X (ridge lambda_frac=0.01))"
        self.ph = h.digest(self.dsl.encode())
        self.input = {
            "context_sha256": self.cid,
            "context_json_sha256": "d" * 64,
            "cut_id": self.cut,
            "dataset_hash": "1234567890abcdef",
            "fold": 0,
            "ledger": "production",
            "use": "train_teacher",
            "view_id": "synthetic-f0",
            "set": "synthetic",
            "p3": {"family_id": self.family, "assignment": "train_fit"},
            "s1_input": {"sha256": "e" * 64},
            "shape": {"k": 2},
        }
        self.final = {
            "schema": "jope.dl-s1-prod-context-final/v1",
            **self.input,
            "s1_input_sha256": "e" * 64,
            "outcome": "selection_reached",
            "summary_bytes": 6,
            "summary_tar_gz_sha256": h.digest(b"opaque"),
            "hwm": {"cut_id": self.cut, "comparable_only_with_same_cut_id": True},
            "binaries": {k: "f" * 64 for k in h.BINARIES},
            "attempts": {
                "charged": 1248,
                "finished": 1248,
                "by_status": {"completed": 1248},
            },
            "g0": {
                "manifest": {
                    "context_sha256": self.cid,
                    "selection_protocol": "jope.s1-two-fold/v1",
                    "ready": True,
                    "attempt_accounting_verified": False,
                    "sampling_resume_qualified": False,
                }
            },
            "outer_test": {"status": "pending", "targets": ["PRIVATE_CANARY"]},
            "selection": {
                "leader_test_rmse_field": None,
                "engine_status_schema": "jope.island-engine-status/s1-v1",
                "cv_geomean_rmse_ratio": 0.8,
                "fitness": -0.1,
                "program_sha256": self.ph,
                "program": {
                    "dsl": self.dsl,
                    "program_sha256": self.ph,
                    "parents": [],
                    "birth_seq": 1,
                    "operator": "seed",
                },
            },
        }
        self.event = {
            "event": "final",
            **{
                k: self.final[k]
                for k in (
                    "context_sha256",
                    "cut_id",
                    "use",
                    "view_id",
                    "set",
                    "outcome",
                )
            },
            "attempts": {"by_status": {"completed": 1248}, "charged": 1248},
            "summary_sha256": h.digest(b"opaque"),
            "selected_program_sha256": self.ph,
            "selected_cv_ratio": 0.8,
            "selected_fitness": -0.1,
        }
        self.lane, self.role, self.denied = "production", "train_fit", False
        self.emit()

    def write(self, path, doc):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = h.common.canonical(doc)
        path.write_bytes(data)
        return h.digest(data)

    def emit(self):
        self.input["ledger"] = self.lane
        self.input["p3"]["assignment"] = self.role
        self.input["use"] = "selection_only" if self.role == "dev" else "train_teacher"
        for field in ("ledger", "p3", "use"):
            self.final[field] = copy.deepcopy(self.input[field])
        self.event["use"] = self.input["use"]
        il = self.source / "inputs.json"
        ih = self.write(il, {"admitted": [self.input]})
        dp = self.source / "deny.json"
        dh = self.write(
            dp,
            {"contexts": [{"view_id": self.input["view_id"]}] if self.denied else []},
        )
        hp = self.source / "holdout.json"
        hh = self.write(hp, {"assignment": {self.family: self.role}})
        rp = self.source / "registry.json"
        rh = self.write(
            rp,
            {
                "families": {
                    self.family: {
                        "members": [{"dataset_hash": self.input["dataset_hash"]}]
                    }
                }
            },
        )
        header = {
            "schema": "jope.dl-s1-prod-ledger-header/v1",
            "selection": "inner-CV only on inner-train (fixture)",
            "binaries": {k: "f" * 64 for k in h.BINARIES.values()},
            "input_list": {"path": str(il), "sha256": ih},
            "denylist": {"path": str(dp), "sha256": dh},
            "p3": {
                "holdout": str(hp),
                "holdout_sha256": hh,
                "registry": str(rp),
                "registry_sha256": rh,
            },
            "plan_from_pinned_s1_budget": {
                "protocol": "jope.s1-fixed-equal/v1",
                "per_context": 1248,
            },
        }
        self.header_sha = self.write(self.source / "ledger.header.json", header)
        self.final["header_sha256"] = self.header_sha
        fp = self.source / "contexts" / self.cid / "final.json"
        self.event["final_sha256"] = self.write(fp, self.final)
        (fp.parent / "summary.tar.gz").write_bytes(b"opaque")
        for lane in ("production", "research"):
            rows = [{"event": "open", "ledger": lane, "header_sha256": self.header_sha}]
            if lane == self.lane:
                rows += [
                    {
                        "event": "assign",
                        "context_sha256": self.cid,
                        "cut_id": self.cut,
                        "use": self.input["use"],
                    },
                    self.event,
                ]
            (self.source / f"ledger-{lane}.jsonl").write_bytes(
                b"".join(h.common.canonical(row) + b"\n" for row in rows)
            )

    def freeze(self):
        plan = self.root / "plan.json"
        return plan, h.freeze(self.source, self.header_sha, plan)["sha256"]

    def publish(self, bucket=None, work="publish", receipt="receipt.json"):
        plan, sha = self.freeze()
        clients = h.transport.fake_clients(self.root / (work + "-budget.jsonl"), bucket)
        with (
            mock.patch.object(h.transport, "fake_clients", return_value=clients),
            mock.patch.object(
                h.transport, "real_clients", side_effect=AssertionError("AWS forbidden")
            ),
        ):
            result = h.publish(plan, sha, self.root / work, self.root / receipt, True)
        return result, clients[2]

    def test_valid_manifest_exact_version_and_no_label_data(self):
        result, bucket = self.publish()
        self.assertEqual(result["counts"], {"selection_reached": 1})
        for versions in bucket.objects.values():
            self.assertEqual(len(versions), 1)
            for _, body in versions:
                self.assertNotIn(b"PRIVATE_CANARY", body)
                self.assertNotIn(b"outer_test", body)
                self.assertNotIn(b"leader_test_rmse_field", body)
        reads = [c for c in bucket.calls if c[0] == "GetObject"]
        self.assertTrue(reads and all(c[2] for c in reads))
        obj = h.inventory(self.source, self.header_sha)[1][0]["body"]
        self.assertEqual(obj["authority"], "progress_evidence_only")
        self.assertFalse(obj["g0_qualification"]["attempt_accounting_verified"])

    def test_research_and_dev_scopes_preserved(self):
        self.lane, self.role = "research", "dev"
        self.emit()
        obj = h.inventory(self.source, self.header_sha)[1][0]
        self.assertTrue(obj["key"].startswith(h.s3io.V2_RESEARCH_ONLY_KEY))
        self.assertEqual(obj["body"]["scope"], "selection_only")
        result, bucket = self.publish()
        for key in bucket.keys():  # noqa: SIM118 - FakeBucket API, not a dict
            if "/gp-hwm/" in key:
                self.assertTrue(key.startswith(h.s3io.V2_RESEARCH_ONLY_KEY))
        self.assertEqual(result["completed"], 1)

    def test_no_selection_is_a_terminal_record(self):
        self.final["outcome"] = self.event["outcome"] = "g0_not_ready"
        del self.final["selection"]
        for k in ("selected_program_sha256", "selected_cv_ratio", "selected_fitness"):
            self.event[k] = None
        self.emit()
        plan, objects = h.inventory(self.source, self.header_sha)
        self.assertEqual(plan["counts"], {"g0_not_ready": 1})
        self.assertIn("/gp-terminal/", objects[0]["key"])
        self.assertNotIn("selection", objects[0]["body"])

    def test_wrong_header_refused(self):
        with self.assertRaises(h.Refused):
            h.inventory(self.source, "0" * 64)

    def test_changed_dependency_refused(self):
        (self.source / "registry.json").write_bytes(b"{}")
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_final_hash_swap_refused_before_clients(self):
        plan, sha = self.freeze()
        (self.source / "contexts" / self.cid / "final.json").write_bytes(b"{}")
        with (
            mock.patch.object(h.transport, "fake_clients") as clients,
            self.assertRaises(h.Refused),
        ):
            h.publish(
                plan, sha, self.root / "publish", self.root / "receipt.json", True
            )
        clients.assert_not_called()
        self.assertFalse((self.root / "publish").exists())

    def test_wrong_cut_program_metrics_and_nonfinite_refused(self):
        for field, value in (
            ("cut_id", "trainonly-split-v2:" + "0" * 64),
            ("s1_input_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                saved = self.final[field]
                self.final[field] = value
                self.emit()
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)
                self.final[field] = saved
        for field, value in (
            ("leader_test_rmse_field", 0.01),
            ("program_sha256", "0" * 64),
            ("cv_geomean_rmse_ratio", 0.95),
            ("fitness", True),
        ):
            with self.subTest(field=field):
                old = self.final["selection"][field]
                self.final["selection"][field] = value
                self.emit()
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)
                self.final["selection"][field] = old
        with self.assertRaises(h.Refused):
            h.decode(b'{"fitness":NaN}')

    def test_rights_and_family_refusals(self):
        for role in ("heldout", "sealed_excluded", "unknown"):
            self.role = role
            self.emit()
            with self.assertRaises(h.Refused):
                h.inventory(self.source, self.header_sha)
        self.role, self.denied = "train_fit", True
        self.emit()
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_duplicate_and_torn_final_ledger_refused(self):
        p = self.source / "ledger-production.jsonl"
        initial = p.read_bytes()
        p.write_bytes(initial + h.common.canonical(self.event) + b"\n")
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)
        p.write_bytes(initial.rstrip(b"\n"))
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_missing_committed_artifact_hard_fails(self):
        (self.source / "contexts" / self.cid / "summary.tar.gz").unlink()
        with self.assertRaises(FileNotFoundError):
            h.inventory(self.source, self.header_sha)

    def test_ledger_append_does_not_change_frozen_set(self):
        plan, sha = self.freeze()
        with (self.source / "ledger-production.jsonl").open("ab") as stream:
            stream.write(b'{"event":"defer","context_sha256":"new"}\n')
        self.assertEqual(h.prepare(plan, sha)[0]["artifact_count"], 1)

    def test_forbidden_only_perturbation_leaves_payload_unchanged(self):
        before = h.inventory(self.source, self.header_sha)
        forbidden = self.source / "outer-test" / "private.json"
        self.write(forbidden, {"targets": [1]})
        self.write(forbidden, {"targets": [999]})
        after = h.inventory(self.source, self.header_sha)
        self.assertEqual(before, after)
        with self.assertRaises(h.Refused):
            h.read_regular(forbidden)

    def test_authorized_program_change_invalidates_frozen_plan(self):
        plan, sha = self.freeze()
        self.final["selection"]["program"]["dsl"] = "(pipe X (ridge lambda_frac=0.02))"
        ph = h.digest(self.final["selection"]["program"]["dsl"].encode())
        self.final["selection"]["program"]["program_sha256"] = self.final["selection"][
            "program_sha256"
        ] = ph
        self.event["selected_program_sha256"] = ph
        self.emit()
        with self.assertRaises(h.Refused):
            h.prepare(plan, sha)

    def test_duplicate_json_keys_refused(self):
        with self.assertRaises(h.Refused):
            h.decode(b'{"a":1,"a":2}')

    def test_symlink_open_rejected_without_creating_a_link(self):
        # Host owner prohibits new symlinks; inject ELOOP at the O_NOFOLLOW boundary.
        with (
            mock.patch.object(h.os, "open", side_effect=OSError(40, "link refused")),
            self.assertRaises(OSError),
        ):
            h.read_regular(self.source / "ledger.header.json")

    def test_interrupted_put_resume_and_duplicate_delivery(self):
        plan, sha = self.freeze()
        _, objects = h.prepare(plan, sha)
        clients = h.transport.fake_clients(self.root / "budget.jsonl")
        bucket = clients[2]
        key = objects[0]["key"]
        bucket.put_faults[key] = [("after", RuntimeError("interrupted"))]
        with mock.patch.object(h.transport, "fake_clients", return_value=clients):
            with self.assertRaises(RuntimeError):
                h.publish(
                    plan, sha, self.root / "publish", self.root / "failed.json", True
                )
            self.assertFalse(any("/manifests/" in k for k in list(bucket.keys())))
            failed = json.loads((self.root / "failed.json").read_bytes())
            self.assertEqual((failed["status"], failed["remaining"]), ("failed", 1))
            h.publish(
                plan, sha, self.root / "publish", self.root / "resumed.json", True
            )
            put_count = len(bucket.puts())
            h.publish(
                plan, sha, self.root / "publish", self.root / "duplicate.json", True
            )
            self.assertEqual(len(bucket.puts()), put_count)
            self.assertEqual(len(bucket.objects[(h.s3io.BUCKET, key)]), 1)

    def test_corrupted_exact_readback_blocks_manifest(self):
        plan, sha = self.freeze()
        _, objects = h.prepare(plan, sha)
        clients = h.transport.fake_clients(self.root / "budget.jsonl")
        clients[2].corrupt_reads[objects[0]["key"]] = b"corrupt"
        with (
            mock.patch.object(h.transport, "fake_clients", return_value=clients),
            self.assertRaises(h.s3io.ReadbackMismatch),
        ):
            h.publish(
                plan, sha, self.root / "publish", self.root / "receipt.json", True
            )
        self.assertFalse(any("/manifests/" in k for k in list(clients[2].keys())))

    def test_existing_different_object_conflicts(self):
        plan, sha = self.freeze()
        _, objects = h.prepare(plan, sha)
        clients = h.transport.fake_clients(self.root / "budget.jsonl")
        clients[2].objects[(h.s3io.BUCKET, objects[0]["key"])] = [
            ("already", b"different")
        ]
        with (
            mock.patch.object(h.transport, "fake_clients", return_value=clients),
            self.assertRaises(h.s3io.PoisonedKey),
        ):
            h.publish(
                plan, sha, self.root / "publish", self.root / "receipt.json", True
            )

    def test_dry_run_never_constructs_aws_or_touches_original_budget(self):
        plan, sha = self.freeze()
        budget = self.root / "original-budget.jsonl"
        # Empty valid ledger; ensure the original stays byte-identical.
        budget.write_bytes(b"")
        with mock.patch.object(
            h.transport, "real_clients", side_effect=AssertionError("AWS forbidden")
        ):
            h.publish(
                plan,
                sha,
                self.root / "publish",
                self.root / "receipt.json",
                True,
                budget,
            )
        self.assertEqual(budget.read_bytes(), b"")
        self.assertNotEqual(
            (self.root / "publish/dry-run-budget.jsonl").read_bytes(), b""
        )


if __name__ == "__main__":
    unittest.main()
