"""End to end on the bound g2 inputs with the real families.run_build (about 45-70 s per build in the lane slice,
family_pin.CpuGuard consulted). Every write goes to injected roots under RUN/tmp/dl-d5-g2. A bound copy links its
`inputs` to RUN/families/g2/inputs, which is only ever read.

- DRAFT (the relay letters, rights_note undecided): the literal outcome 797 / N 640 / k 32 through the in-memory
  rehearsal variant; check; check --real refused.
- what-if with the recommended answers (Q5=a, rights_note unknown): 797 / N 652 / k 33.
- real: an expansion from the owner line with the CL-2 real-path simulation, which predicts the final registry sha256.
  Then:
  - check --real;
  - the families CLI in a real generation (patched FAMILIES_DIR), which builds exactly the predicted registry.json;
  - check --registry, where holdout.review_gate applies the owner-record and registry-marker rules;
  - families verify -> publish (fake bucket) -> holdout check_gate, check_registry_object, load_registry and
    review_gate on that families-built registry (P3I-6).
- The families CLI in a real generation refuses a DRAFT copied to <out>/review-decisions.json (D-MAP-6).

These tests run only while dltrain families/family_keys/family_pin/family_screen still equal the 'code' block of
registry.pre-review.json e21690dd (CODE_MATCHES). Under code drift a real build cannot rebuild the bound bytes; they are
then skipped with the drifted module names (an explained skip), and TestCodeBinding proves the R7 refusal."""
import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from tests_dl_d5.fixtures_dl_d5 import (CODE_MATCHES, CONFIRM_TEXT, CSV, CSV_PIN, DRIFT_REASON, LITERAL, LITERAL_DECISIONS,
                                        LITERAL_OUTCOME, OWNER_DECISIONS, PRE, RECOMMENDED, REC_DECISIONS, REC_OUTCOME,
                                        SHARED, D5Case, T, caching_build, check_args, directive_args, draft_args,
                                        real_args, sha, sim_ok, whatif_args)
from dltrain import common, families, family_pin, holdout


def real_build_kw():
    return {"build_fn": caching_build, "guard_factory": family_pin.CpuGuard}


class RealCase(D5Case):
    PIN_CODE = False          # the tool reads the real code shas; the real build embeds them

    def families_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(families, "FAMILIES_DIR", Path(self.roots.families)), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            code = families._main([str(a) for a in argv], roots=(Path(self.roots.families), Path(self.roots.receipts)),
                                  guard=family_pin.CpuGuard())
        return code, out.getvalue(), err.getvalue()


@unittest.skipUnless(CODE_MATCHES, DRIFT_REASON)
class TestDraftEndToEnd(RealCase):
    def test_draft_gives_the_literal_outcome_and_checks(self):
        s = self.ok(draft_args("--simulate", "--expect-outcome", LITERAL_OUTCOME), **real_build_kw())
        self.assertEqual(s["simulation"]["outcome"], LITERAL)
        self.assertEqual((s["decision_counts"], s["undecided_rows"], s["simulation"]["real"]),
                         (LITERAL_DECISIONS, 14, False))
        draft, dsha = Path(s["out"]["path"]), s["out"]["sha256"]
        self.assertEqual(draft.name, f"DRAFT-UNCONFIRMED-review-decisions-{dsha[:16]}.json")
        self.assertEqual((draft.parent, oct(draft.stat().st_mode & 0o777)), (Path(self.roots.stage).resolve(), "0o444"))
        rec = json.loads(Path(s["receipt"]["path"]).read_text())
        self.assertEqual((rec["schema"], rec["mode"], rec["verdict"]), ("jope.dl-p2-decisions-expand/v2", "draft", "PASS"))
        self.assertTrue(rec["notice"].startswith("DRAFT-UNCONFIRMED"))
        self.assertEqual({k: v for k, v in rec["code"].items() if k != "tool"}, json.loads(PRE.read_bytes())["code"])
        self.assertEqual((rec["code"]["tool"], rec["inputs"]["review_csv"]["sha256"]), (T.tool_sha256(), CSV_PIN))
        sim = rec["simulation"]
        self.assertTrue(sim["pre_review_equal"] and sim["review_csv_equal"])
        self.assertEqual((sim["decision_deltas"]["families"], sim["decision_deltas"]["undecided_rows"]), (-91, 14))
        self.assertEqual(sim["decision_deltas"]["provenance"], {"draft": True, "simulation": False, "record": None})
        self.assertEqual(sim["decisions_sha256"], rec["validation"]["rehearsal_sha256"])
        self.assertEqual(rec["provenance"], T.DRAFT_PROVENANCE)
        rehearsal_registry = SHARED["final"][sim["decisions_sha256"]]
        self.assertEqual(sha(rehearsal_registry), sim["registry_final"]["sha256"])
        self.assertTrue(json.loads(rehearsal_registry)["decision_deltas"]["provenance"]["draft"])
        # check: byte-equal re-expansion + validation of the rehearsal variant on the real in-memory pre-review model
        chk = check_args(draft, dsha)
        c = self.ok(chk, **real_build_kw())
        self.assertEqual((c["mode"], c["re_expansion_byte_equal"], c["pre_review_rebuilt_equal"]), ("draft", True, True))
        self.assertEqual(c["validate"], {"of": "rehearsal variant", "real": False, "effective_decisions": 399,
                                         "undecided_rows": 14})
        self.assertEqual(set(c["unpromotable"]), {"families_staging", "families_real", "holdout",
                                                  "holdout_draft_dry_run"})
        self.stop(chk + ["--real"], "R9", **real_build_kw())


@unittest.skipUnless(CODE_MATCHES, DRIFT_REASON)
class TestWhatIfRecommended(RealCase):
    def test_recommended_answers_give_n652_k33(self):
        s = self.ok(whatif_args("--simulate", "--expect-outcome", REC_OUTCOME), **real_build_kw())
        self.assertEqual(s["simulation"]["outcome"], RECOMMENDED)
        self.assertEqual((s["decision_counts"], s["undecided_rows"]), (REC_DECISIONS, 0))
        self.assertTrue(Path(s["out"]["path"]).name.startswith("WHATIF-NOT-A-DECISION-review-decisions-"))


@unittest.skipUnless(CODE_MATCHES, DRIFT_REASON)
class TestRealEndToEnd(RealCase):
    def test_real_expansion_cli_build_registry_gate_and_p3_interface(self):
        bdir = self.bound_copy(inputs=True)
        receipts = Path(self.roots.receipts)
        msg, msha = self.message(CONFIRM_TEXT)
        s = self.ok(real_args(bdir, receipts, msg, msha), **real_build_kw())
        self.assertEqual((s["mode"], s["simulation"]["outcome"], s["simulation"]["real"]), ("real", RECOMMENDED, True))
        predicted, dsha = s["simulation"]["registry_final"]["sha256"], s["out"]["sha256"]
        stored = receipts.resolve() / f"dl-d5-owner-message-{msha[:16]}.txt"
        final = SHARED["final"][dsha]
        self.assertEqual(sha(final), predicted)
        self.assertEqual(json.loads(final)["decision_deltas"]["provenance"],
                         {"draft": False, "simulation": False,
                          "record": {"path": str(stored), "sha256": msha, "form": "stored_owner_message"}})
        out_path = bdir / "review-decisions.json"
        chk = check_args(out_path, dsha, "--real", bdir=bdir)
        c = self.ok(chk, **real_build_kw())
        self.assertEqual((c["mode"], c["validate"]["real"], c["validate"]["effective_decisions"]), ("real", True, 413))
        # the families CLI in a real generation builds exactly the registry.json the expansion predicted (P3I-3 c)
        code, out, err = self.families_cli(["build", "--config", bdir / "alias-sources.json", "--out", bdir,
                                            "--decisions", out_path, "--decisions-sha256", dsha,
                                            "--receipt", receipts / "build-g2-final.json"])
        self.assertEqual(code, 0, err[-1500:])
        reg_path = bdir / families.OUT_FINAL
        self.assertEqual(sha(reg_path.read_bytes()), predicted)
        # check --registry: holdout.review_gate (P2 rule with the stored owner message directly in the receipts dir,
        # the registry marker) and the tool-side P3I-1 gate
        g = self.ok(chk + ["--registry", reg_path, "--registry-sha256", predicted], **real_build_kw())["review_gate"]
        self.assertEqual((g["review_decisions_sha256"], g["seed_rows"], g["review_rows"]), (dsha, 10, 413))
        self.assertEqual((g["provenance"]["draft"], g["registry_provenance"]["record"]["sha256"]), (False, msha))
        # P3I-6: verify -> publish (fake bucket) -> holdout gates on this families-built registry
        from tests_p3 import fixtures_p3 as fx3
        vpath = receipts / "verify-g2-final.json"
        with mock.patch.object(families, "FAMILIES_DIR", Path(self.roots.families)):
            rep = families.verify(str(bdir / "alias-sources.json"), bdir, out_path, guard=family_pin.CpuGuard(),
                                  expect=families.EXPECT, full_rehash=False, receipt_path=vpath, decisions_sha256=dsha,
                                  receipts_dir=receipts)
        self.assertEqual((rep["verdict"], rep["failed"], rep["phase"]), ("PASS", [], "final"), rep.get("failed"))
        bucket = fx3.FakeBucket()
        clients = fx3.fake_clients(bucket)
        rpath = receipts / "receipt.json"
        pub = families.publish(bdir, receipt_path=rpath, verify_receipt_path=vpath, clients=clients)
        self.assertEqual((len(bucket.puts()), pub["refs"]["registry_final"]["sha256"]), (7, predicted))
        reg = holdout.load_registry(reg_path, predicted)
        gate = holdout.check_gate(reg, rpath, OWNER_DECISIONS[0], OWNER_DECISIONS[1], sha(rpath.read_bytes()))
        self.assertEqual(gate["registry_s3"]["sha256"], predicted)
        got = holdout.check_registry_object(clients["reader"], gate["registry_s3"], reg.sha256, reg.bytes)
        self.assertEqual(got["version_id"], pub["refs"]["registry_final"]["version_id"])
        rg = holdout.review_gate(reg, out_path.read_bytes(), CSV.read_bytes())
        self.assertEqual((rg["provenance"]["draft"], rg["provenance"]["record"]["form"], rg["seed_rows"]),
                         (False, "stored_owner_message", 10))
        self.assertEqual(holdout.pool_members(reg.families).__len__(), RECOMMENDED["N_pool"])


@unittest.skipUnless(CODE_MATCHES, DRIFT_REASON)
class TestRealGenerationRefusesADraft(RealCase):
    def test_a_draft_copied_to_the_decision_file_name_stops_the_real_build(self):
        made = self.ok(draft_args())
        raw = Path(made["out"]["path"]).read_bytes()
        gen = self.bound_copy(inputs=True)
        target = gen / families.DECISIONS_NAME
        target.write_bytes(raw)
        os.chmod(target, 0o444)
        code, out, err = self.families_cli(["build", "--config", gen / "alias-sources.json", "--out", gen,
                                            "--decisions", target, "--decisions-sha256", sha(raw)])
        self.assertEqual(code, families.EXIT["STOP"], out[-500:])
        self.assertIn("decisions schema is not", err)
        self.assertFalse((gen / families.OUT_FINAL).exists())


@unittest.skipUnless(CODE_MATCHES, DRIFT_REASON)
class TestDirectiveProvenance(RealCase):
    def test_owner_directive_expansion_and_check_real(self):
        bdir = self.bound_copy(inputs=True)          # check rebuilds the pre-review from the bound copy's inputs
        d, dsha = self.directives("# Owner directives v7", "", "- " + CONFIRM_TEXT)
        s = self.ok(directive_args(bdir, self.roots.receipts, d, dsha), build_fn=sim_ok())
        chk = check_args(s["out"]["path"], s["out"]["sha256"], "--real", bdir=bdir)
        c = self.ok(chk, **real_build_kw())
        self.assertEqual((c["verdict"], c["mode"], c["validate"]["effective_decisions"]), ("PASS", "real", 413))
        d.write_text("# Owner directives v8\n")                  # the cited file changed after the expansion
        self.stop(chk, "R9", **real_build_kw())                # its pin no longer matches the recorded file_sha256


if __name__ == "__main__":
    unittest.main()
