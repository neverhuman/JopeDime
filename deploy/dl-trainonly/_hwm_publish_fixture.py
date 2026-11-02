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
CAT_V3_SCHEMA = "jope.dl-family-catalog-v246-decisions/v3-outliers"
ONE_HOLDOUT_SCHEMA = "jope.dltrain-one-holdout-role-map/v1"


class PublisherFixture(unittest.TestCase):
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
        self.cat_v2_schema = CAT_V2_SCHEMA
        # ONE-HOLDOUT mode (Jepson 2026-10-07 5:10pm MT): the input list binds a pinned role map naming P3 heldout contexts as
        # training data. oh_listed: the map names this context; oh_p3: the P3 the map records (default: the row's); oh_pin: the
        # map's sha is pinned in the adapter.
        self.oh_mode, self.oh_listed, self.oh_p3, self.oh_pin = False, True, None, True
        pins = (getattr(h, "G3_DECISIONS_SHA256", None), getattr(h, "APPROVED_P3_DECISIONS", None),
                getattr(h, "ONE_HOLDOUT_ROLE_MAPS", None))  # absent on older adapters
        self.addCleanup(lambda: setattr(h, "G3_DECISIONS_SHA256", pins[0]) or setattr(h, "APPROVED_P3_DECISIONS", pins[1])
                        or setattr(h, "ONE_HOLDOUT_ROLE_MAPS", pins[2]))
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
        if self.oh_mode:
            listdoc["one_holdout"] = {"role_map_sha256": self.emit_role_map()}
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
            d2 = {"schema": self.cat_v2_schema, "per_dataset": [{"dataset_hash": ds, **self.cat_v2}]}
            p2, a2 = self.source / "catalog-v2.json", self.source / "approval-v2.json"
            self.cat_sha_v2 = self.write(p2, d2)
            a2sha = self.write(a2, {"candidate_sha256": self.cat_sha_v2, "verdict": "APPROVE", "reviewer": "Lane C, independent"})
            approved[self.cat_sha_v2] = {"approval": str(a2), "approval_sha256": a2sha, "reviewer": "lane c"}
            ref.update(decision_v2=str(p2), sha256_v2=self.cat_sha_v2)
        h.APPROVED_P3_DECISIONS = approved
        return ref

    def emit_role_map(self):
        """Write the ONE-HOLDOUT role map fixture; pin it unless oh_pin is False; return its sha256."""
        listed = [{"context_sha256": self.cid, "view_id": self.input["view_id"],
                   "p3": copy.deepcopy(self.oh_p3 or self.input["p3"])}] if self.oh_listed else []
        rm = self.source / "role-map.json"
        self.oh_sha = self.write(rm, {"schema": ONE_HOLDOUT_SCHEMA, "newly_training": {"heldout_non_g3": listed,
                                                                                      "catalog_heldout_p3v1": []}})
        h.ONE_HOLDOUT_ROLE_MAPS = {self.oh_sha: str(rm)} if self.oh_pin else {}
        return self.oh_sha

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

