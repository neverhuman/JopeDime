"""Catalog and one-holdout publisher cases, mixed into the legacy test module."""

from _hwm_publish_fixture import *  # noqa: F401,F403


class CatalogCases:
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

    def use_one_holdout(self, role, **knobs):
        self.oh_mode, self.role = True, role
        for name, value in knobs.items():
            setattr(self, name, value)
        self.emit()

    def test_one_holdout_heldout_publishes_as_training_with_role_source(self):
        self.use_one_holdout("heldout")
        plan, objects = h.inventory(self.source, self.header_sha)
        body = objects[0]["body"]
        self.assertEqual(body["family"], {"family_id": self.family, "assignment": "heldout"})
        self.assertEqual(body["scope"], "train_teacher")
        self.assertEqual(body["provenance"]["role_source"], {"path": str(self.source / "role-map.json"), "sha256": self.oh_sha})
        self.assertIn(str(self.source / "role-map.json"), {r["path"] for r in plan["sources"]})
        result, bucket = self.publish()
        self.assertEqual(result["completed"], 1)

    def test_one_holdout_bound_list_leaves_other_roles_unchanged(self):
        # CONTROL: a bound role map changes nothing for the already-admitted roles (no role_source on them).
        for role in ("train_fit", "train_nonpool", "dev"):
            with self.subTest(role=role):
                self.use_one_holdout(role)
                body = h.inventory(self.source, self.header_sha)[1][0]["body"]
                self.assertEqual(body["family"]["assignment"], role)
                self.assertNotIn("role_source", body["provenance"])

    def test_one_holdout_heldout_without_binding_refused(self):
        # MUST-FAIL: without a bound role map, heldout stays refused exactly as before ONE-HOLDOUT.
        self.role = "heldout"
        self.emit()
        with self.assertRaisesRegex(h.Refused, "unadmitted family/input"):
            h.inventory(self.source, self.header_sha)

    def test_one_holdout_unpinned_or_tampered_role_map_refused(self):
        self.use_one_holdout("heldout", oh_pin=False)
        with self.assertRaisesRegex(h.Refused, "not an approved role source"):
            h.inventory(self.source, self.header_sha)
        self.use_one_holdout("heldout", oh_pin=True)
        h.inventory(self.source, self.header_sha)
        self.write(self.source / "role-map.json", {"schema": ONE_HOLDOUT_SCHEMA, "newly_training": {"x": []}})
        with self.assertRaisesRegex(h.Refused, "source hash mismatch"):
            h.inventory(self.source, self.header_sha)

    def test_one_holdout_unlisted_or_wrong_p3_refused(self):
        self.use_one_holdout("heldout", oh_listed=False)
        with self.assertRaisesRegex(h.Refused, "unadmitted family/input"):
            h.inventory(self.source, self.header_sha)
        self.use_one_holdout("heldout", oh_listed=True, oh_p3={"family_id": "fam-" + "9" * 64, "assignment": "heldout"})
        with self.assertRaisesRegex(h.Refused, "unadmitted family/input"):
            h.inventory(self.source, self.header_sha)

    def test_one_holdout_never_admits_sealed_or_scope_escalation(self):
        # MUST-FAIL: the one holdout (sealed_excluded) stays refused even when a bound, pinned map lists the context.
        self.use_one_holdout("sealed_excluded", oh_p3=None)
        with self.assertRaisesRegex(h.Refused, "unadmitted family/input"):
            h.inventory(self.source, self.header_sha)
        self.use_one_holdout("heldout")
        self.input["use"] = self.final["use"] = "selection_only"
        self.event["use"] = "selection_only"
        self.write(self.source / "inputs.json", {"admitted": [self.input], "one_holdout": {"role_map_sha256": self.oh_sha}})
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_catalog_v3_outliers_pin_and_schema(self):
        pin = h.APPROVED_P3_DECISIONS["c846aa5dbb9c03ce815cebffe4284b42be2731d94fbe41e6bf64cd9e371545f0"]
        self.assertEqual((pin["approval_sha256"], pin["reviewer"]),
                         ("2288883b323f86370a28089a90ae035f2f18db14ab12a4437e9b45e9bad2c5ed", "lane c"))
        self.cat_v2_schema = CAT_V3_SCHEMA
        self.use_catalog("train_fit", cat_v2={"family_id": self.family, "role": "train_fit"})
        self.input["p3_source"] = "catalog-decisions " + self.cat_sha_v2[:12]
        self.emit()
        body = h.inventory(self.source, self.header_sha)[1][0]["body"]
        self.assertEqual(body["provenance"]["family_source"]["sha256"], self.cat_sha_v2)
        self.cat_v2_schema = "jope.dl-family-catalog-v246-decisions/v4"
        self.emit()
        with self.assertRaisesRegex(h.Refused, "unsupported catalog decisions schema"):
            h.inventory(self.source, self.header_sha)
