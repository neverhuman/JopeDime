"""The P2 -> P3 handoff the real sequence (S4 -> S6) relies on, with both modules read-only and the tests_p3 fake bucket
(the real s3io lane guard applies).
- The shared constants: schemas, the owner-message name, the canonical OWNER_DIRECTIVES.md, the seed rows.
- The agreed owner confirmation line: holdout's fixture message parses under the tool's grammar dl-d5-confirm/v1.
- families.publish on a small final-phase out dir produces a P2 packet receipt that holdout.check_gate accepts:
  schema, packet P2, PASS, and the registry s3ref at objects/family-registry/sha256/<sha>.json, identical in
  publish.registry_final and the s3_ledger. The decision file itself is not published (gap 7).
The registry here is the tests_p3 fixture registry (synthetic: P3I-6). A families-built g2 final registry goes through
publish, check_gate, load_registry and review_gate in test_d5_simulate."""
import json
import unittest
from pathlib import Path

from tests_dl_d5.fixtures_dl_d5 import CSV_PIN, OWNER_DECISIONS, D5Case, T, sha, synthetic_final_registry
from dltrain import common, families, family_screen, holdout


class TestP2ToP3Interface(D5Case):
    def test_shared_constants(self):
        self.assertEqual(families.DECISIONS_SCHEMA, holdout.DECISIONS_SCHEMA)
        self.assertEqual(families.REGISTRY_SCHEMA, holdout.REGISTRY_SCHEMA)
        self.assertEqual(families.EXPERIMENT_ID, holdout.EXPERIMENT_ID)
        self.assertIn(families.PACKET_RECEIPT_SCHEMA, holdout.P2_RECEIPT_SCHEMAS)
        self.assertEqual((holdout.SEED_ROW_KIND, holdout.EXPECTED_SEED_ROWS), ("seed_source_mapping", 10))
        self.assertEqual(set(holdout.SEED_ROW_OK), {"confirm_mapping", "add_members"})
        self.assertEqual(holdout.OWNER_DECISIONS_SHA256, OWNER_DECISIONS[1])
        self.assertEqual(families.DECISION_OPTIONS["seed_source_mapping"][0], "confirm_mapping")
        self.assertEqual(T.MESSAGE_NAME, families.OWNER_MESSAGE_NAME)
        self.assertNotIn(T.DRAFT_SCHEMA, (holdout.DECISIONS_SCHEMA, families.SIMULATION_SCHEMA))
        self.assertNotIn(T.WHATIF_SCHEMA, (holdout.DECISIONS_SCHEMA, families.SIMULATION_SCHEMA))
        self.assertNotIn(T.DRAFT_KIND, families.PROVENANCE_KINDS)
        self.assertNotIn(T.WHATIF_KIND, families.PROVENANCE_KINDS)

    def test_canonical_owner_record_locations_agree(self):
        """The unpatched module constants (D5Case patches the live ones): the tool and holdout name the same canonical
        OWNER_DIRECTIVES.md, holdout reads owner messages from the receipts root the tool writes them to."""
        self.assertEqual(Path(T.OWNER_DIRECTIVES_PATH).name, families.OWNER_DIRECTIVE_NAMES[0])
        canonical = 'OWNER_DIRECTIVES_PATH = Path("/home/ubuntu/JopeDime/.agent/plans/OWNER_DIRECTIVES.md")'
        self.assertIn(canonical, Path(holdout.__file__).read_text())
        self.assertIn(canonical, Path(T.__file__).read_text())
        self.assertIn('P2_RECEIPTS_DIR = RUN / "receipts" / "P2"', Path(holdout.__file__).read_text())
        self.assertEqual(T.RECEIPTS_ROOT, families.RECEIPTS_DIR)

    def test_holdouts_agreed_owner_line_parses_under_the_tool_grammar(self):
        from tests_p3 import fixtures_p3 as fx3
        line = fx3.owner_message(CSV_PIN).decode()
        self.assertEqual(T.parse_confirmation(line, CSV_PIN), {"review_csv_sha12": CSV_PIN[:12], "Q1": "a", "Q2": "a",
                                                               "Q3": "a", "Q4": "a", "Q5": "a"})

    def test_families_publish_feeds_holdout_check_gate(self):
        from tests_p3 import fixtures_p3 as fx3        # the shared fake bucket; brings boto3/s3io into the test process
        out = self.tmp / "families"
        reg_body = synthetic_final_registry("a" * 64)
        files = {families.OUT_PRE: b'{"fake":"pre-review"}', families.OUT_CSV: b"review_group_id,kind\n",
                 families.OUT_FINAL: reg_body}
        files.update({f"{families.SCREEN_DIR}/{n}": b"{}" for n in family_screen.SCREEN_FILES})
        for name, body in files.items():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_bytes(body)
        alias = b'{"fake":"alias-sources"}'
        (out / "alias-sources.json").write_bytes(alias)
        vpath = self.tmp / "verify-final.json"
        vpath.write_bytes(common.canonical({"schema": families.VERIFY_RECEIPT_SCHEMA, "verdict": "PASS", "phase": "final",
                                            "alias_sources_sha256": sha(alias),
                                            "outputs": {n: {"sha256": sha(b)} for n, b in files.items()}}))
        bucket = fx3.FakeBucket()
        rpath = self.tmp / "receipts" / "P2" / "receipt.json"
        pub = families.publish(out, receipt_path=rpath, verify_receipt_path=vpath, clients=fx3.fake_clients(bucket))
        self.assertEqual(len(bucket.puts()), 7)
        ref = pub["refs"]["registry_final"]
        self.assertEqual(ref["key"], holdout._key(holdout.REGISTRY_OBJECT_REL.format(sha=sha(reg_body))))
        self.assertEqual(ref["key"], f"experiments/gp-policy-dl/v1/objects/family-registry/sha256/{sha(reg_body)}.json")
        rec = json.loads(rpath.read_bytes())
        self.assertEqual((rec["schema"], rec["packet"], rec["verdict"], rec["phase"]),
                         (families.PACKET_RECEIPT_SCHEMA, "P2", "PASS", "final"))
        self.assertNotIn("review-decisions.json", rec["outputs"])
        self.assertFalse(any("review-decisions" in k for k in bucket.puts()))
        reg = holdout.load_registry(out / families.OUT_FINAL, sha(reg_body))
        gate = holdout.check_gate(reg, rpath, OWNER_DECISIONS[0], OWNER_DECISIONS[1], sha(rpath.read_bytes()))
        self.assertEqual(gate["registry_s3"], {k: ref[k] for k in holdout.REF_KEYS})
        self.assertEqual(gate["p2_receipt"]["sha256"], sha(rpath.read_bytes()))
        # P3I-2: the ref resolves at its exact VersionId in the bucket publish wrote to
        got = holdout.check_registry_object(fx3.fake_clients(bucket)["reader"], gate["registry_s3"], reg.sha256, reg.bytes)
        self.assertEqual(got["version_id"], ref["version_id"])
        # P3 pins the receipt bytes: a receipt that names another registry, or a changed receipt, STOPs the gate
        other = holdout.load_registry(self._write("other.json", synthetic_final_registry("b" * 64)),
                                      sha(synthetic_final_registry("b" * 64)))
        with self.assertRaises(holdout.HoldoutStop):
            holdout.check_gate(other, rpath, OWNER_DECISIONS[0], OWNER_DECISIONS[1], sha(rpath.read_bytes()))
        with self.assertRaises(holdout.HoldoutStop):
            holdout.check_gate(reg, rpath, OWNER_DECISIONS[0], OWNER_DECISIONS[1], "0" * 64)

    def _write(self, name: str, body: bytes) -> Path:
        p = self.tmp / name
        p.write_bytes(body)
        return p


if __name__ == "__main__":
    unittest.main()
