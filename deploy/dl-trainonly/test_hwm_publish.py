"""Synthetic receiving-path and failure-injection fixtures; no private input or AWS access."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hwm_publish as h

# Literal contract strings (not the adapter's constants) so the fixture also runs against an adapter without them.
C1_SCHEMA = "jope.dl-s1-prod-ledger-header/v1"
C2_SCHEMA = "jope.dl-s1-prod-c2-ledger-header/v1"
G3_SCHEMA = "jope.dl-family-g3-decisions/v1"
CAT_V1_SCHEMA = "jope.dl-family-catalog-v246-decisions/v1"
CAT_V2_SCHEMA = "jope.dl-family-catalog-v246-decisions/v2"


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
        # Campaign-2 mode (G3-appended contexts): one research-only ledger, the family absent from the pinned registry and
        # holdout, its P3 role taken from the sha-bound G3 decision file. g3_role is the DECIDED role; self.role is what the
        # input row claims.
        self.c2_mode, self.g3_role, self.g3_family, self.g3_on_c1 = False, None, None, False
        self.ledgers = None
        # Catalog mode (GP HWMs for every admitted catalog context, VP OWNER 7:20am MT): the dataset is unknown to the pinned
        # registry/holdout and the input list binds approved catalog-v246 decision file(s). cat_role is the DECIDED role.
        self.catalog_mode, self.cat_bind, self.cat_role, self.cat_family, self.cat_ds = False, True, None, None, None
        self.cat_v2, self.cat_reviewer, self.cat_approves, self.cat_holdout_role = None, "Lane A (laneA)", True, None
        pins = (getattr(h, "G3_DECISIONS_SHA256", None), getattr(h, "APPROVED_P3_DECISIONS", None))  # absent on older adapters
        self.addCleanup(lambda: setattr(h, "G3_DECISIONS_SHA256", pins[0]) or setattr(h, "APPROVED_P3_DECISIONS", pins[1]))
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
        listdoc = {"admitted": [self.input]}
        if self.catalog_mode and self.cat_bind:
            listdoc["catalog_p3"] = self.emit_catalog()
        ih = self.write(il, listdoc)
        dp = self.source / "deny.json"
        dh = self.write(
            dp,
            {"contexts": [{"view_id": self.input["view_id"]}] if self.denied else []},
        )
        hp = self.source / "holdout.json"
        rp = self.source / "registry.json"
        if self.c2_mode or self.catalog_mode:
            held = {self.family: self.cat_holdout_role} if self.cat_holdout_role else {}
            hh = self.write(hp, {"assignment": held})
            rh = self.write(rp, {"families": {}})
        else:
            hh = self.write(hp, {"assignment": {self.family: self.role}})
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
            "schema": C2_SCHEMA if self.c2_mode else C1_SCHEMA,
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
        if self.c2_mode or self.g3_on_c1:
            gp = self.source / "g3-decisions.json"
            gh = self.write(
                gp,
                {
                    "schema": G3_SCHEMA,
                    "per_identity": [
                        {
                            "dataset_hash": self.input["dataset_hash"],
                            "family_id": self.g3_family or self.family,
                            "role": self.g3_role or self.role,
                        }
                    ],
                },
            )
            header["g3_decisions"] = {"path": str(gp), "sha256": gh}
            h.G3_DECISIONS_SHA256 = gh  # the fixture's file stands in for the owner-named 796e2724
        lanes = self.ledgers or (["research"] if self.c2_mode else None)
        if lanes is not None:
            header["ledgers"] = {lane: str(self.source / f"ledger-{lane}.jsonl") for lane in lanes}
        self.header_sha = self.write(self.source / "ledger.header.json", header)
        self.final["header_sha256"] = self.header_sha
        fp = self.source / "contexts" / self.cid / "final.json"
        self.event["final_sha256"] = self.write(fp, self.final)
        (fp.parent / "summary.tar.gz").write_bytes(b"opaque")
        for stale in self.source.glob("ledger-*.jsonl"):
            stale.unlink()
        for lane in [l for l in ("production", "research") if lanes is None or l in lanes]:
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

    def emit_catalog(self):
        """Write the catalog decision file(s) and approval record(s); pin them as approved; return the list's catalog_p3."""
        ds = self.cat_ds or self.input["dataset_hash"]
        d1 = {"schema": CAT_V1_SCHEMA, "per_dataset": [
            {"dataset_hash": ds, "family_id": self.cat_family or self.family, "role": self.cat_role or self.role}]}
        p1 = self.source / "catalog-v1.json"
        self.cat_sha = self.write(p1, d1)
        a1 = self.source / "approval-v1.json"
        named = self.cat_sha if self.cat_approves else "0" * 64
        a1sha = self.write(a1, {"decision_sha256": named, "reviewer": self.cat_reviewer, "review_ref": "AGENT_CHAT fixture"})
        approved = {self.cat_sha: {"approval": str(a1), "approval_sha256": a1sha, "reviewer": "lane a"}}
        ref = {"decision": str(p1), "sha256": self.cat_sha}
        if self.cat_v2 is not None:
            d2 = {"schema": CAT_V2_SCHEMA, "per_dataset": [{"dataset_hash": ds, **self.cat_v2}]}
            p2, a2 = self.source / "catalog-v2.json", self.source / "approval-v2.json"
            self.cat_sha_v2 = self.write(p2, d2)
            a2sha = self.write(a2, {"candidate_sha256": self.cat_sha_v2, "verdict": "APPROVE", "reviewer": "Lane C, independent"})
            approved[self.cat_sha_v2] = {"approval": str(a2), "approval_sha256": a2sha, "reviewer": "lane c"}
            ref.update(decision_v2=str(p2), sha256_v2=self.cat_sha_v2)
        h.APPROVED_P3_DECISIONS = approved
        return ref

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

    def assert_output_alias_refused(self, work_dir, receipt):
        plan, sha = self.freeze()
        before = {
            str(p.relative_to(self.source)): p.read_bytes()
            for p in self.source.rglob("*")
            if p.is_file()
        }
        with (
            mock.patch.object(h.transport, "fake_clients") as fake,
            mock.patch.object(h.transport, "real_clients") as real,
            mock.patch.object(h.transport, "PublishLedger") as ledger,
        ):
            with self.assertRaises(h.Refused):
                h.publish(plan, sha, work_dir, receipt, True)
            fake.assert_not_called()
            real.assert_not_called()
            ledger.assert_not_called()
        after = {
            str(p.relative_to(self.source)): p.read_bytes()
            for p in self.source.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)
        self.assertFalse((self.root / "publish").exists())

    def test_work_directory_parent_alias_refused_before_state(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.assert_output_alias_refused(
            outside / ".." / "source", self.root / "receipt.json"
        )

    def test_receipt_parent_alias_refused_before_state(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.assert_output_alias_refused(
            self.root / "publish", outside / ".." / "source" / "receipt.json"
        )

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


    # ---- campaign-2 (G3-appended) receiving path: owner GP-2 (ENG:watcher 05:16Z), G3 decisions 796e2724 as the P3 role source
    def use_c2(self, role="train_fit", g3_role=None, g3_family=None):
        self.c2_mode, self.lane, self.role, self.g3_role, self.g3_family = True, "research", role, g3_role, g3_family
        self.emit()

    def test_c2_research_only_ledger_with_g3_roles_publishes(self):
        for role in ("train_fit", "train_nonpool", "dev"):
            with self.subTest(role=role):
                self.use_c2(role)
                self.assertFalse((self.source / "ledger-production.jsonl").exists())
                plan, objects = h.inventory(self.source, self.header_sha)
                self.assertEqual(set(plan["ledgers"]), {"research"})
                self.assertEqual(len(objects), 1)
                self.assertTrue(objects[0]["key"].startswith(h.s3io.V2_RESEARCH_ONLY_KEY))
                self.assertEqual(objects[0]["body"]["family"], {"family_id": self.family, "assignment": role})
                self.assertIn(str(self.source / "g3-decisions.json"), {r["path"] for r in plan["sources"]})
        result, bucket = self.publish()
        self.assertEqual(result["completed"], 1)
        manifests = [k for k in bucket.keys() if "/manifests/gp-hwm/" in k]  # noqa: SIM118 - FakeBucket API
        self.assertEqual(len(manifests), 1)
        self.assertTrue(manifests[0].startswith(h.s3io.V2_RESEARCH_ONLY_KEY))

    def test_c2_decided_heldout_sealed_roleless_refused(self):
        # MUST-FAIL: the decided G3 role, not the row's claim, admits the context; a row claiming train_fit for a family the
        # decision file holds out (or seals, or leaves roleless) is refused.
        for decided in ("heldout", "sealed_excluded", "ROLELESS_BRIDGE"):
            with self.subTest(decided=decided):
                self.use_c2("train_fit", g3_role=decided)
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)
        for role in ("heldout", "sealed_excluded", "roleless_bridge"):
            with self.subTest(role=role):
                self.use_c2(role)
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)

    def test_c2_row_role_disagreeing_with_decision_refused(self):
        self.use_c2("train_fit", g3_role="train_nonpool")
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_c2_decision_file_swap_refused(self):
        # MUST-FAIL: the decision file is bound by sha256 in the header; any byte change is refused before projection.
        self.use_c2("train_fit")
        h.inventory(self.source, self.header_sha)
        self.write(
            self.source / "g3-decisions.json",
            {"schema": G3_SCHEMA, "per_identity": [
                {"dataset_hash": self.input["dataset_hash"], "family_id": self.family, "role": "train_nonpool"}]},
        )
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_c2_family_missing_from_decisions_refused(self):
        self.use_c2("train_fit", g3_family="fam-" + "9" * 64)
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_g3_decisions_on_campaign1_header_refused(self):
        self.g3_on_c1 = True
        self.emit()
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_registry_and_holdout_keep_precedence_over_decisions(self):
        # A dataset/family the pinned registry and holdout already know keeps its registry family and holdout role; the
        # decision file only adds appended identities (campaign-2 input-list rule).
        self.g3_on_c1 = False
        self.c2_mode = False
        self.emit()
        reg = {"families": {self.family: {"members": [{"dataset_hash": self.input["dataset_hash"]}]}}}
        self.c2_mode, self.lane = True, "research"
        self.g3_role, self.g3_family = "heldout", "fam-" + "8" * 64
        self.emit()
        hold = {"assignment": {self.family: "train_fit"}}
        rh, hh = self.write(self.source / "registry.json", reg), self.write(self.source / "holdout.json", hold)
        hdr = json.loads((self.source / "ledger.header.json").read_bytes())
        hdr["p3"].update(registry_sha256=rh, holdout_sha256=hh)
        self.header_sha = self.write(self.source / "ledger.header.json", hdr)
        self.final["header_sha256"] = self.header_sha
        self.event["final_sha256"] = self.write(self.source / "contexts" / self.cid / "final.json", self.final)
        rows = [{"event": "open", "ledger": "research", "header_sha256": self.header_sha},
                {"event": "assign", "context_sha256": self.cid, "cut_id": self.cut, "use": self.input["use"]}, self.event]
        (self.source / "ledger-research.jsonl").write_bytes(b"".join(h.common.canonical(r) + b"\n" for r in rows))
        objects = h.inventory(self.source, self.header_sha)[1]
        self.assertEqual(objects[0]["body"]["family"], {"family_id": self.family, "assignment": "train_fit"})

    def test_unknown_or_missing_declared_lane_refused(self):
        self.use_c2("train_fit")
        hdr = json.loads((self.source / "ledger.header.json").read_bytes())
        hdr["ledgers"]["sealed"] = "x"
        sha = self.write(self.source / "ledger.header.json", hdr)
        with self.assertRaises(h.Refused):
            h.inventory(self.source, sha)
        self.ledgers = ["production", "research"]
        self.emit()
        (self.source / "ledger-production.jsonl").unlink()
        with self.assertRaises((h.Refused, FileNotFoundError)):
            h.inventory(self.source, self.header_sha)


    # ---- catalog P3 role authority (VP OWNER 7:20am MT, Jepson ruling): approved, sha-bound catalog-v246 decision files
    def use_catalog(self, role="train_fit", **knobs):
        self.catalog_mode, self.lane, self.role = True, "research", role
        for name, value in knobs.items():
            setattr(self, name, value)
        self.emit()

    def test_catalog_roles_publish_with_family_source(self):
        for role in ("train_fit", "train_nonpool", "dev"):
            with self.subTest(role=role):
                self.use_catalog(role)
                plan, objects = h.inventory(self.source, self.header_sha)
                body = objects[0]["body"]
                self.assertEqual(body["family"], {"family_id": self.family, "assignment": role})
                self.assertEqual(body["scope"], "selection_only" if role == "dev" else "train_teacher")
                self.assertEqual(
                    body["provenance"]["family_source"],
                    {"path": str(self.source / "catalog-v1.json"), "sha256": self.cat_sha},
                )
                self.assertTrue(objects[0]["key"].startswith(h.s3io.V2_RESEARCH_ONLY_KEY))
                paths = {r["path"] for r in plan["sources"]}
                self.assertTrue({str(self.source / "catalog-v1.json"), str(self.source / "approval-v1.json")} <= paths)
        result, bucket = self.publish()
        self.assertEqual(result["completed"], 1)

    def test_registry_records_keep_the_merged_payload(self):
        # Regression: campaign-1 registry/holdout records carry exactly the merged publisher's provenance (no family_source).
        objects = h.inventory(self.source, self.header_sha)[1]
        self.assertEqual(
            set(objects[0]["body"]["provenance"]),
            {"header_sha256", "final_sha256", "input_sha256", "context_json_sha256", "summary_sha256", "binaries"},
        )

    def test_catalog_unknown_role_refused(self):
        # MUST-FAIL (fail closed): a dataset the registry does not know, with no catalog binding or absent from the bound file.
        self.use_catalog("train_fit", cat_bind=False)
        with self.assertRaisesRegex(h.Refused, "invalid family identity"):
            h.inventory(self.source, self.header_sha)
        self.use_catalog("train_fit", cat_bind=True, cat_ds="f" * 16)
        with self.assertRaisesRegex(h.Refused, "absent from the catalog decisions"):
            h.inventory(self.source, self.header_sha)

    def test_catalog_sealed_heldout_and_disagreeing_roles_refused(self):
        for decided in ("sealed_excluded", "heldout", "train_nonpool"):
            with self.subTest(decided=decided):
                self.use_catalog("train_fit", cat_role=decided)
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)
        for claimed in ("sealed_excluded", "heldout"):
            with self.subTest(claimed=claimed):
                self.use_catalog(claimed, cat_role=None)
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)

    def test_catalog_decision_tamper_or_unapproved_refused(self):
        self.use_catalog("train_fit")
        h.inventory(self.source, self.header_sha)
        self.write(self.source / "catalog-v1.json", {"schema": CAT_V1_SCHEMA, "per_dataset": [
            {"dataset_hash": self.input["dataset_hash"], "family_id": self.family, "role": "train_nonpool"}]})
        with self.assertRaisesRegex(h.Refused, "source hash mismatch"):
            h.inventory(self.source, self.header_sha)
        self.use_catalog("train_fit")
        h.APPROVED_P3_DECISIONS = {}
        with self.assertRaisesRegex(h.Refused, "not an approved role source"):
            h.inventory(self.source, self.header_sha)

    def test_catalog_missing_or_non_approving_record_refused(self):
        self.use_catalog("train_fit")
        (self.source / "approval-v1.json").unlink()
        with self.assertRaisesRegex(h.Refused, "approval record missing"):
            h.inventory(self.source, self.header_sha)
        for knobs in ({"cat_approves": False}, {"cat_reviewer": "Lane B (laneB)"}):
            with self.subTest(**knobs):
                self.use_catalog("train_fit", **knobs)
                with self.assertRaisesRegex(h.Refused, "does not approve"):
                    h.inventory(self.source, self.header_sha)

    def test_catalog_two_bound_files_cited_source_and_disagreement(self):
        self.input["p3_source"] = "catalog-decisions <v2>"
        self.use_catalog("train_fit", cat_v2={"family_id": self.family, "role": "train_fit"})
        self.input["p3_source"] = "catalog-decisions " + self.cat_sha_v2[:12]
        self.emit()
        body = h.inventory(self.source, self.header_sha)[1][0]["body"]
        self.assertEqual(body["provenance"]["family_source"]["sha256"], self.cat_sha_v2)
        self.use_catalog("train_fit", cat_v2={"family_id": self.family, "role": "train_nonpool"})
        with self.assertRaisesRegex(h.Refused, "disagree"):
            h.inventory(self.source, self.header_sha)

    def test_catalog_holdout_keeps_precedence(self):
        # A family the pinned holdout already assigns keeps the holdout role (as the S1 input prep resolved it).
        self.use_catalog("train_nonpool", cat_role="train_fit", cat_holdout_role="train_nonpool")
        body = h.inventory(self.source, self.header_sha)[1][0]["body"]
        self.assertEqual(body["family"]["assignment"], "train_nonpool")
        self.use_catalog("train_fit", cat_role="train_fit", cat_holdout_role="train_nonpool")
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_c2_records_name_the_g3_source_and_only_the_owner_named_file(self):
        self.use_c2("train_fit")
        body = h.inventory(self.source, self.header_sha)[1][0]["body"]
        self.assertEqual(body["provenance"]["family_source"]["path"], str(self.source / "g3-decisions.json"))
        h.G3_DECISIONS_SHA256 = "0" * 64
        with self.assertRaisesRegex(h.Refused, "owner-named G3 decision file"):
            h.inventory(self.source, self.header_sha)

if __name__ == "__main__":
    unittest.main()
