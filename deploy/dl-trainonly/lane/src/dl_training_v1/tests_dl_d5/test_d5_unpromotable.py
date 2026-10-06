"""D-MAP-6: DRAFT and what-if files are structurally unpromotable.

- families refuses them in every generation, staging and real, through validate_decisions. Its real-generation CLI
  refuses them by path before any build; the full-build refusal of a copy at <out>/review-decisions.json is in
  test_d5_simulate.
- holdout refuses them in both of its modes: the real review gate, and the non-CLI DRAFT dry-run mode.
- holdout commit STOPs on them before any S3 call.
- Even hand-promoted copies are refused: the schema rewritten, then the kind.
- The in-memory rehearsal variant is refused by every real generation and by holdout's real gate, and it is never
  written to a file.
- The tool's own promotion paths (check --real, check --registry, dryrun of a real file) refuse."""
import io
import contextlib
import json
import unittest
from pathlib import Path

from tests_dl_d5.fixtures_dl_d5 import (ALIAS, CONFIRM_TEXT, CSV, CSV_PIN, OWNER_DECISIONS, PRE, D5Case, NoGuard, T,
                                        check_args, dryrun_args, real_args, sha, sim_ok, stub_build,
                                        synthetic_final_registry)
from dltrain import common, families, holdout


class UnconfirmedCase(D5Case):
    def setUp(self):
        super().setUp()
        self.made = {"draft": self.draft(), "what_if": self.whatif()}
        self.model = stub_build()(str(ALIAS)).pre

    def validate(self, raw, real):
        return families.validate_decisions(raw, self.model.rows, review_csv_sha256=CSV_PIN,
                                           pre_sha256=sha(PRE.read_bytes()), alias_sha256=sha(ALIAS.read_bytes()),
                                           pre=self.model, real=real, receipts_dir=self.roots.receipts)

    def registry_for(self, psha, name):
        """A phase-final fixture registry that binds the given decision file sha (the strongest position for it)."""
        body = synthetic_final_registry(psha)
        reg = self.tmp / f"registry-{name}.json"
        reg.write_bytes(body)
        return reg, sha(body)


class TestFamiliesRefuses(UnconfirmedCase):
    def test_validate_decisions_refuses_in_every_generation(self):
        for mode, (p, _) in self.made.items():
            raw = p.read_bytes()
            for real in (False, True):
                with self.subTest(mode=mode, real=real), self.assertRaisesRegex(families.FamilyStop,
                                                                                "decisions schema is not"):
                    self.validate(raw, real)
                with self.subTest(mode=mode, real=real, as_simulation=True), \
                        self.assertRaisesRegex(families.FamilyStop, "simulation"):
                    families.validate_decisions(raw, self.model.rows, review_csv_sha256=CSV_PIN,
                                                pre_sha256=sha(PRE.read_bytes()), alias_sha256=sha(ALIAS.read_bytes()),
                                                pre=self.model, real=real, simulation=True)

    def test_hand_promoted_copies_are_still_refused(self):
        for mode, (p, _) in self.made.items():
            doc = json.loads(p.read_bytes())
            doc.update(schema=families.DECISIONS_SCHEMA, reviewer="owner")             # the schema rewritten
            for real in (False, True):
                with self.subTest(mode=mode, step="schema", real=real), \
                        self.assertRaisesRegex(families.FamilyStop, "known kind"):
                    self.validate(common.canonical(doc), real)
            doc["provenance"] = dict(doc["provenance"], kind="owner_verbatim_message", message_sha256="a1" * 32,
                                     message_utc="2026-10-05T15:30Z")                   # then the kind
            doc["rulings"]["split_registry_excluded_seals_nothing"] = True
            with self.subTest(mode=mode, step="kind", real=True), \
                    self.assertRaisesRegex(families.FamilyStop, "marks DRAFT or SIMULATION"):
                self.validate(common.canonical(doc), True)

    def test_the_real_generation_cli_refuses_the_file_before_any_build(self):
        from unittest import mock
        gen = self.bound_copy("g2")
        for mode, (p, psha) in self.made.items():
            argv = ["build", "--config", str(gen / "alias-sources.json"), "--out", str(gen), "--decisions", str(p),
                    "--decisions-sha256", psha]
            out, err = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), mock.patch.object(families, "FAMILIES_DIR", Path(self.roots.families)), \
                    mock.patch.object(families, "run_build", side_effect=AssertionError("no build may start")), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = families._main(argv, roots=(Path(self.roots.families), Path(self.roots.receipts)),
                                      guard=NoGuard(), expect=None)
                self.assertEqual(code, families.EXIT["STOP"], err.getvalue())
                self.assertIn("review-decisions.json", err.getvalue())
                self.assertFalse((gen / families.OUT_FINAL).exists())


class TestHoldoutRefuses(UnconfirmedCase):
    def test_review_gate_refuses_in_both_modes(self):
        for mode, (p, psha) in self.made.items():
            reg_path, rsha = self.registry_for(psha, mode)
            reg = holdout.load_registry(reg_path, rsha)
            for dry in (False, True):
                with self.subTest(mode=mode, draft_dry_run=dry), \
                        self.assertRaisesRegex(holdout.HoldoutStop, "decisions schema"):
                    holdout.review_gate(reg, p.read_bytes(), CSV.read_bytes(), draft_dry_run=dry)

    def test_commit_stops_before_any_s3_call(self):
        from tests_p3 import fixtures_p3 as fx3
        for mode, (p, psha) in self.made.items():
            reg_path, rsha = self.registry_for(psha, mode)
            for dry in (False, True):
                with self.subTest(mode=mode, draft_dry_run=dry):
                    bucket = fx3.FakeBucket()
                    hold = self.tmp / f"holdout-{mode}-{int(dry)}"
                    argv = ["commit", "--registry", str(reg_path), "--registry-sha256", rsha, "--decisions", str(p),
                            "--review-csv", str(CSV), "--p2-receipt", str(self.tmp / "no-p2-receipt.json"),
                            "--owner-decisions", str(OWNER_DECISIONS[0]), "--owner-decisions-sha256",
                            OWNER_DECISIONS[1], "--out", str(hold)]
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = holdout._main(argv, clients=fx3.fake_clients(bucket), draft_dry_run=dry)
                    line = json.loads(out.getvalue().strip().splitlines()[-1])
                    self.assertEqual((code, line["status"]), (2, "STOP"), out.getvalue() + err.getvalue())
                    self.assertIn("decisions schema", json.dumps(line["error"]))
                    self.assertEqual((bucket.calls, bucket.puts()), ([], []))
                    self.assertFalse((hold / "commitment.json").exists() or (hold / "seed.hex").exists())


class TestRehearsalVariant(UnconfirmedCase):
    def test_refused_by_every_real_generation_and_by_holdouts_real_gate(self):
        for mode, (p, psha) in self.made.items():
            doc = json.loads(p.read_bytes())
            reh = T.rehearsal_doc(doc, mode, psha)
            body = common.canonical(reh)
            self.assertEqual(families.check_provenance(reh["provenance"], real=False), {"draft": True, "record": None})
            with self.subTest(mode=mode, gate="families real"), \
                    self.assertRaisesRegex(families.FamilyStop, "marks DRAFT or SIMULATION"):
                families.check_provenance(reh["provenance"], real=True, receipts_dir=self.roots.receipts)
            with self.subTest(mode=mode, gate="validate real"), \
                    self.assertRaisesRegex(families.FamilyStop, "marks DRAFT or SIMULATION"):
                self.validate(body, True)
            self.assertTrue(self.validate(body, False)["effective"])                  # staging only
            reg_path, rsha = self.registry_for(sha(body), f"reh-{mode}")
            reg = holdout.load_registry(reg_path, rsha)
            with self.subTest(mode=mode, gate="holdout real"), \
                    self.assertRaisesRegex(holdout.HoldoutStop, "marks DRAFT or SIMULATION"):
                holdout.review_gate(reg, body, CSV.read_bytes())
            # never on disk: no file the tool wrote holds the rehearsal bytes
            self.assertNotIn(sha(body), {sha((self.tmp / f).read_bytes()) for f in self.files()})


class TestToolPromotionPaths(UnconfirmedCase):
    def test_check_and_dryrun_refuse(self):
        for mode, (p, psha) in self.made.items():
            with self.subTest(mode=mode):
                self.stop(check_args(p, psha, "--real"), "R9", contains="never a decision file")
                body = synthetic_final_registry(psha)
                reg = self.tmp / f"registry-{mode}.json"
                reg.write_bytes(body)
                self.stop(check_args(p, psha, "--registry", reg, "--registry-sha256", sha(body)), "R9",
                          contains="never reaches the holdout gate")
        # a real decision file never runs as a rehearsal
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        s = self.ok(real_args(bdir, self.roots.receipts, msg, msha), build_fn=sim_ok())
        self.stop(dryrun_args(s["out"]["path"], s["out"]["sha256"]), "R9", contains="rehearses only a DRAFT or what-if")

    def test_unconfirmed_files_are_written_only_to_the_staging_dir(self):
        for mode, (p, psha) in self.made.items():
            self.assertEqual(p.parent, Path(self.roots.stage).resolve())
            self.assertTrue(p.name.startswith(T.PREFIX[mode]))
            self.assertEqual(oct(p.stat().st_mode & 0o777), "0o444")
        files = self.files()
        self.assertFalse([f for f in files if f.startswith("families/")])
        self.assertEqual(sorted(f.split("/")[-1].split("p2-decisions-expand-")[0] for f in files
                                if f.startswith("receipts/")),
                         ["DRAFT-UNCONFIRMED-", "WHATIF-NOT-A-DECISION-"])


if __name__ == "__main__":
    unittest.main()
