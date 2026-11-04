"""Campaign-2/G3 publisher cases, mixed into the legacy test module."""

from _hwm_publish_fixture import *  # noqa: F401,F403


class Campaign2Cases:
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


