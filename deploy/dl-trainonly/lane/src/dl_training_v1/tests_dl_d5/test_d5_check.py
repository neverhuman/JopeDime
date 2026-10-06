"""check against a stand-in pre-review model (no real build): the byte-equal re-expansion, validation of a real file
(--real: families.validate_decisions(real=True) with the stored owner message) and of an unconfirmed file through its
rehearsal variant, the unpromotable self-checks, the --real refusals, the holdout.review_gate dry run with the tool-side
P3I-1 registry gate, and the P3I-4 rehearsal-code gate. Also the R7 code binding without the pin patch. Real-build
variants are in test_d5_simulate."""
import json
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from tests_dl_d5.fixtures_dl_d5 import (A4, ALIAS, ALIAS_PIN, CONFIRM_TEXT, CSV, CSV_PIN, LITERAL_COUNTS, LITERAL_DECISIONS,
                                        LITERAL_OUTCOME, PINNED_CODE, POSTED_UTC, PRE, PRE_PIN, REC_DECISIONS, D5Case, T,
                                        check_args, directive_args, draft_args, dryrun_args, pinned_code, real_args, sha,
                                        sim_ok, stub_build, synthetic_final_registry, whatif_args)
from dltrain import common, families


def record_of(roots, msha):
    return {"path": str(Path(roots.receipts).resolve() / f"dl-d5-owner-message-{msha[:16]}.txt"), "sha256": msha,
            "form": "stored_owner_message"}


class TestCheckUnconfirmed(D5Case):
    def test_draft_check_reexpands_validates_and_refuses_promotion(self):
        calls = []
        draft, dsha = self.draft("--simulate", "--expect-outcome", LITERAL_OUTCOME,
                                 build_fn=sim_ok(counts=LITERAL_COUNTS, calls=calls))
        self.assertEqual(calls, ["final"])
        c = self.ok(check_args(draft, dsha), build_fn=stub_build(calls=calls))
        self.assertEqual(calls, ["final", "pre"])                  # check runs the pre-review build once
        self.assertEqual((c["verdict"], c["mode"], c["re_expansion_byte_equal"], c["pre_review_rebuilt_equal"]),
                         ("PASS", "draft", True, True))
        self.assertEqual(c["validate"], {"of": "rehearsal variant", "real": False, "effective_decisions": 399,
                                         "undecided_rows": 14})
        self.assertEqual(c["decision_counts"], LITERAL_DECISIONS)
        self.assertEqual(set(c["unpromotable"]), {"families_staging", "families_real", "holdout",
                                                  "holdout_draft_dry_run"})
        for name, reason in c["unpromotable"].items():
            self.assertIn("schema", reason, name)
        self.assertIn("decisions provenance (P2 rule", c["unpromotable"]["holdout"])
        self.assertEqual((c["real"], c["review_gate"], c["decisions"]["sha256"]), (None, None, dsha))
        # --real and --registry never accept an unconfirmed file
        self.stop(check_args(draft, dsha, "--real"), "R9", contains="refuses a DRAFT file")
        reg, body = self.tmp / "registry.json", synthetic_final_registry(dsha)
        reg.write_bytes(body)
        self.stop(check_args(draft, dsha, "--registry", reg, "--registry-sha256", sha(body)), "R9",
                  contains="never reaches the holdout gate")
        # the pre-review model must rebuild the bound bytes (code drift that keeps the shas cannot pass either)
        for name, pair in {"pre": (PRE.read_bytes() + b" ", CSV.read_bytes()),
                           "csv": (PRE.read_bytes(), CSV.read_bytes()[:-1])}.items():
            with self.subTest(differs=name):
                self.stop(check_args(draft, dsha), "R7", contains="rebuilds different",
                          build_fn=stub_build(pre_review=pair))

    def test_whatif_check(self):
        w, wsha = self.whatif()
        c = self.ok(check_args(w, wsha))
        self.assertEqual((c["mode"], c["validate"]["effective_decisions"], c["decision_counts"]),
                         ("what_if", 413, REC_DECISIONS))
        self.stop(check_args(w, wsha, "--real"), "R9", contains="refuses a what-if file")

    def test_check_refuses_files_that_are_not_the_mechanical_expansion(self):
        draft, dsha = self.draft()
        stage = Path(self.roots.stage)

        def variant(name, mutate):
            doc = json.loads(draft.read_bytes())
            mutate(doc)
            p = stage / f"DRAFT-UNCONFIRMED-{name}.json"
            p.write_bytes(common.canonical(doc))
            return check_args(p, sha(p.read_bytes()))

        def first_subset_to_independent(doc):
            d = next(d for d in doc["decisions"] if d["decision"] == "union_subset")
            d["decision"] = "independent"
            del d["sets"]
        self.stop(variant("tampered", first_subset_to_independent), "EXPANSION", contains="['decisions']")
        line = self.stop(variant("oldtool", lambda doc: doc["expansion"].update(tool_sha256="0" * 64)), "EXPANSION")
        self.assertIn("another version of this tool", line["reason"])
        self.stop(variant("shape", lambda doc: doc["expansion"].update(shape_sha256="1" * 64)), "R8")
        self.stop(variant("answers", lambda doc: doc["answers"].update(Q2="b")), "R9")       # a DRAFT is the relay's
        self.stop(variant("rights", lambda doc: doc["answers"].update(rights_note="unknown")), "R9")
        self.stop(variant("split", lambda doc: doc["answers"].update(split_registry_confirmed=True)), "R9")
        self.stop(variant("prov", lambda doc: doc["provenance"].update(note="edited")), "R9")
        self.stop(variant("prohibited", lambda doc: doc["answers"].update(rights_note="prohibited")), "R4")
        self.stop(variant("q3b", lambda doc: doc["answers"].update(Q3="b")), "R2")
        self.stop(variant("noanswers", lambda doc: doc.pop("answers")), "EXPANSION")
        self.stop(variant("ruling", lambda doc: doc["rulings"].update(split_registry_excluded_seals_nothing=True)),
                  "EXPANSION", contains="['rulings']")
        # promoting the schema by hand: the relay kind is no provenance a real file can carry
        self.stop(variant("promoted", lambda doc: doc.update(schema=families.DECISIONS_SCHEMA, reviewer="owner")), "R3")
        self.stop(variant("promoted2", lambda doc: (doc.update(schema=families.DECISIONS_SCHEMA, reviewer="owner"),
                                                    doc["answers"].update(split_registry_confirmed=True))), "R9",
                  contains="not one this tool writes")
        self.stop(variant("simulation", lambda doc: doc.update(schema=families.SIMULATION_SCHEMA)), "EXPANSION",
                  contains="neither a decision file")
        self.stop(check_args(draft, "0" * 64), "R7")                                         # the decisions pin


class TestCheckReal(D5Case):
    def expand(self, text=CONFIRM_TEXT, **kw):
        bdir = self.bound_copy()
        msg, msha = self.message(text)
        seen = []
        argv = real_args(bdir, self.roots.receipts, msg, msha, **kw)
        s = self.ok(argv, build_fn=sim_ok(seen=seen))
        return bdir, msha, argv, s, seen

    def test_message_expansion_then_check_real(self):
        bdir, msha, argv, s, seen = self.expand()
        self.assertEqual((s["mode"], s["decision_counts"], s["undecided_rows"], s["unpromotable"]),
                         ("real", REC_DECISIONS, 0, None))
        self.assertEqual({k: s["simulation"]["outcome"][k] for k in ("families", "N_pool", "k")},
                         {"families": 797, "N_pool": 652, "k": 33})
        out_path = bdir / "review-decisions.json"
        (raw, kw), = seen                                         # the simulation is the CL-2 real path
        self.assertEqual(raw, out_path.read_bytes())
        self.assertEqual(kw, {"real": True, "receipts_dir": str(Path(self.roots.receipts).resolve())})
        stored = Path(self.roots.receipts).resolve() / f"dl-d5-owner-message-{msha[:16]}.txt"
        self.assertEqual((Path(s["out"]["path"]), Path(s["owner_message"]["path"])), (out_path.resolve(), stored))
        self.assertEqual(stored.read_bytes(), CONFIRM_TEXT.encode())
        for p in (out_path, stored):
            self.assertEqual(oct(p.stat().st_mode & 0o777), "0o444")
        doc = json.loads(out_path.read_bytes())
        self.assertEqual(doc["provenance"], {"kind": "owner_verbatim_message",
                                             "session": argv[argv.index("--session") + 1],
                                             "message_utc": "2026-10-05T15:30:00Z", "message_sha256": msha,
                                             "message_text": CONFIRM_TEXT})
        self.assertEqual(doc["answers"], dict(A4, Q5="a", rights_note="unknown", split_registry_confirmed=True))
        self.assertEqual(doc["confirmation"]["question_posted_utc"], POSTED_UTC)
        self.assertEqual(doc["binds"], {"review_csv_sha256": CSV_PIN, "registry_pre_review_sha256": PRE_PIN,
                                        "alias_sources_sha256": ALIAS_PIN})
        rec = json.loads(Path(s["receipt"]["path"]).read_text())
        self.assertEqual((rec["schema"], rec["mode"], rec["notice"]), ("jope.dl-p2-decisions-expand/v2", "real", None))
        self.assertNotIn("message_text", rec["provenance"])
        self.assertEqual(rec["provenance_checks"]["parsed"], dict(A4, Q5="a", review_csv_sha12="aad99b1a9af6"))
        self.assertEqual(rec["real_generation_gate"], {"draft": False, "record_form": "pinned_record",
                                                       "record_sha256": msha})
        self.assertEqual(set(rec["code"]), {"families", "family_keys", "family_pin", "family_screen", "tool"})
        self.assertEqual((rec["mapping"], rec["shape"]["expect"]["builtin"], rec["tally"]["decisions"],
                          rec["validation"]), ("dl-d5-map/v2", "builtin:aad99b1a", 413,
                                               {"accepted": True, "effective_decisions": 413}))
        self.assertEqual((rec["answers"], rec["rights"], rec["split_registry_confirmed"]), (A4, "unknown", True))
        self.assertEqual(rec["owner_message"]["sha256"], msha)
        dsha = s["out"]["sha256"]
        chk = check_args(out_path, dsha, "--real", bdir=bdir)
        c = self.ok(chk)
        self.assertEqual((c["mode"], c["real"]["owner_message"]["path"], c["real"]["bound_dir"], c["validate"]["real"]),
                         ("real", str(stored), str(bdir.resolve()), True))
        self.assertEqual(c["provenance_checks"]["parsed"], dict(A4, Q5="a", review_csv_sha12="aad99b1a9af6"))
        # the same message and answers expand to the same bytes (a new receipt; the files already exist)
        again = put = list(argv)
        put[put.index("--receipt") + 1] = str(Path(self.roots.receipts) / "decisions-expand-1791197000000001.json")
        s2 = self.ok(again, build_fn=sim_ok())
        self.assertEqual((s2["out"]["sha256"], s2["out"]["existed"], s2["owner_message"]["existed"]), (dsha, True, True))
        # check --real: a copy outside the bound families dir; a missing stored owner message
        copy = self.tmp / "review-decisions.json"
        copy.write_bytes(out_path.read_bytes())
        self.stop(check_args(copy, dsha, "--real", bdir=bdir), "R10")
        self.ok(check_args(copy, dsha, bdir=bdir))                   # without --real the location is not checked
        stored.chmod(0o644)
        stored.unlink()
        self.stop(chk, "R9", contains="stored owner message")

    def test_registry_gates(self):
        """holdout.review_gate (owner record directly in the receipts dir, registry marker) plus the tool-side P3I-1
        gate on the registry's decision_deltas.provenance."""
        bdir, msha, argv, s, _ = self.expand()
        out_path, dsha = bdir / "review-decisions.json", s["out"]["sha256"]
        chk = check_args(out_path, dsha, "--real", bdir=bdir)
        good = {"draft": False, "simulation": False, "record": record_of(self.roots, msha)}
        reg, body = bdir / "registry.json", synthetic_final_registry(dsha, provenance=good)
        reg.write_bytes(body)
        g = self.ok(chk + ["--registry", reg, "--registry-sha256", sha(body)])["review_gate"]
        self.assertEqual((g["review_decisions_sha256"], g["seed_rows"], g["review_rows"]), (dsha, 10, 413))
        self.assertEqual(g["registry_provenance"]["record"]["sha256"], msha)
        self.assertEqual(g["provenance"]["draft"], False)
        self.stop(chk + ["--registry", reg], "R7", contains="go together")
        self.stop(chk + ["--registry", reg, "--registry-sha256", "0" * 64], "GATE")
        bad = {"none": None, "draft": dict(good, draft=True), "simulation": dict(good, simulation=True),
               "staging": dict(good, record=None), "other_record": dict(good, record=dict(good["record"], sha256="e" * 64)),
               "other_form": dict(good, record=dict(good["record"], form="pinned_record"))}
        for name, prov in bad.items():
            with self.subTest(registry_provenance=name):
                p = self.tmp / f"registry-{name}.json"
                b = synthetic_final_registry(dsha, provenance=prov) if prov is not None else synthetic_final_registry(dsha)
                p.write_bytes(b)
                self.stop(chk + ["--registry", p, "--registry-sha256", sha(b)], "GATE")
        for name, kw, reason in (("pre_review", {"pre_sha": "f" * 64}, "registry_pre_review_sha256"),
                                 ("alias", {"alias_sha": "d" * 64}, "alias_sources_sha256")):
            with self.subTest(registry_binds_other=name):
                p = self.tmp / f"registry-{name}.json"
                b = synthetic_final_registry(dsha, provenance=good, **kw)
                p.write_bytes(b)
                self.stop(chk + ["--registry", p, "--registry-sha256", sha(b)], "GATE", contains=reason)

    def test_dryrun_code_gate(self):
        """P3I-4: --dryrun-receipt needs a PASS rehearsal receipt that ran the current holdout and family code."""
        from dltrain import holdout
        bdir, msha, argv, s, _ = self.expand()
        out_path, dsha = bdir / "review-decisions.json", s["out"]["sha256"]
        good = {"draft": False, "simulation": False, "record": record_of(self.roots, msha)}
        reg, body = bdir / "registry.json", synthetic_final_registry(dsha, provenance=good)
        reg.write_bytes(body)
        chk = check_args(out_path, dsha, "--real", bdir=bdir) + ["--registry", reg, "--registry-sha256", sha(body)]
        code = dict(T.current_code_shas(), holdout_source_tree=holdout.source_tree_sha256())

        def receipt(name, **over):
            doc = {"schema": T.DRYRUN_SCHEMA, "verdict": "PASS", "code": code} | over
            p = self.tmp / f"dryrun-{name}.json"
            p.write_bytes(common.receipt_bytes(doc))
            return [str(p), "--dryrun-receipt-sha256", sha(p.read_bytes())]
        g = self.ok(chk + ["--dryrun-receipt"] + receipt("ok"))["review_gate"]
        self.assertEqual(g["dryrun_code"]["code_equal"], sorted(code))
        self.stop(chk + ["--dryrun-receipt"] + receipt("stop", verdict="STOP"), "GATE", contains="not a PASS")
        self.stop(chk + ["--dryrun-receipt"] + receipt("old", code=dict(code, holdout_source_tree="0" * 64)), "GATE",
                  contains="['holdout_source_tree']")
        self.stop(chk + ["--dryrun-receipt", receipt("ok")[0]], "R7", contains="go together")
        self.stop(check_args(out_path, dsha, "--real", bdir=bdir, ) + ["--dryrun-receipt"] + receipt("x"), "R10",
                  contains="only with --registry")

    def test_crafted_real_files_are_refused(self):
        bdir, msha, argv, s, _ = self.expand()
        out_path = Path(s["out"]["path"])

        def variant(name, mutate):
            doc = json.loads(out_path.read_bytes())
            mutate(doc)
            p = self.tmp / f"{name}.json"
            p.write_bytes(common.canonical(doc))
            return check_args(p, sha(p.read_bytes()), bdir=bdir)

        def session(value):
            return lambda doc: doc["provenance"].update(session=value)

        def relay_message(doc):
            doc["provenance"].update(message_sha256=T.RELAY_SHA256)
        self.stop(variant("simulation-session", session("SIMULATION-NOT-A-DECISION")), "R9", contains="SIMULATION")
        self.stop(variant("draft-session", session("DRAFT-UNCONFIRMED-RELAY-b785a123")), "R9", contains="DRAFT")
        self.stop(variant("relay", relay_message), "R9", contains="does not hash")
        self.stop(variant("text", lambda doc: doc["provenance"].update(message_text=CONFIRM_TEXT + " ")), "R9")
        self.stop(variant("split-false", lambda doc: doc["answers"].update(split_registry_confirmed=False)), "R3")
        self.stop(variant("answers", lambda doc: doc["answers"].update(Q4="b")), "R9", contains="not the --answers")
        self.stop(variant("rights", lambda doc: doc["answers"].update(rights_note="undecided")), "R9",
                  contains="not --rights undecided")
        self.stop(variant("kind", lambda doc: doc["provenance"].update(kind="agent_relay")), "R9")
        self.stop(variant("posted", lambda doc: doc["confirmation"].update(question_posted_utc="2026-10-05T16:00Z")),
                  "R9", contains="predates")
        self.stop(variant("letters", lambda doc: doc["confirmation"]["letters"].update(Q5="b")), "EXPANSION",
                  contains="['confirmation']")
        with mock.patch.object(T, "RELAY_SHA256", msha):          # the recorded message is the relayed paste
            self.stop(check_args(out_path, s["out"]["sha256"], "--real", bdir=bdir), "R9", contains="relayed")

    def test_directive_expansion_then_check_real(self):
        bdir = self.bound_copy()
        d, dsha = self.directives("# Owner directives v7", "", "- " + CONFIRM_TEXT)
        s = self.ok(directive_args(bdir, self.roots.receipts, d, dsha), build_fn=sim_ok())
        self.assertIsNone(s["owner_message"])
        doc = json.loads(Path(s["out"]["path"]).read_bytes())
        self.assertEqual(doc["provenance"], {"kind": "owner_directive", "file": str(d.resolve()), "version": "v7",
                                             "lines": "3", "file_sha256": dsha})
        self.assertIsNone(doc["confirmation"]["question_posted_utc"])
        chk = check_args(s["out"]["path"], s["out"]["sha256"], "--real", bdir=bdir)
        c = self.ok(chk)
        self.assertEqual((c["verdict"], c["mode"], c["validate"]["effective_decisions"], c["real"]["owner_message"]),
                         ("PASS", "real", 413, None))
        rec = {"path": str(d.resolve()), "sha256": dsha, "form": "owner_directive"}
        reg, body = bdir / "registry.json", synthetic_final_registry(
            s["out"]["sha256"], provenance={"draft": False, "simulation": False, "record": rec})
        reg.write_bytes(body)
        g = self.ok(chk + ["--registry", reg, "--registry-sha256", sha(body)])["review_gate"]
        self.assertEqual(g["registry_provenance"]["record"]["form"], "owner_directive")
        d.write_text("# Owner directives v8\n")                  # the cited file changed after the expansion
        self.stop(chk, "R9", contains="differs from its pin")


class TestCheckVariants(D5Case):
    def test_every_answer_variant_round_trips(self):
        """Each Q1/Q2/Q4 and rights variant (what-if), expanded in memory, is accepted through its rehearsal variant by
        validate_decisions on the stand-in model with the same effective decisions: 413, or 399 with rights undecided."""
        c_b, p_b, a_b = (T.read_pinned(CSV, CSV_PIN, "c"), T.read_pinned(PRE, PRE_PIN, "p"),
                         T.read_pinned(ALIAS, ALIAS_PIN, "a"))
        for answers, rights, effective in ((A4, "undecided", 399), (dict(A4, Q1="b"), "unknown", 413),
                                           (dict(A4, Q2="b"), "unknown", 413), (dict(A4, Q2="c"), "unknown", 413),
                                           (dict(A4, Q4="b"), "unknown", 413)):
            with self.subTest(answers=answers, rights=rights):
                e = T.build_expansion(mode="what_if", csv_b=c_b, pre_b=p_b, alias_b=a_b, answers=answers, rights=rights,
                                      provenance=dict(T.WHATIF_PROVENANCE),
                                      want_shape=T.BUILTIN_SHAPES["builtin:aad99b1a"])
                model = stub_build()(str(a_b.path)).pre
                v = families.validate_decisions(e.build_body, model.rows, review_csv_sha256=CSV_PIN, pre_sha256=PRE_PIN,
                                                alias_sha256=a_b.sha256, pre=model)
                self.assertEqual(len(v["effective"]), effective)
                self.assertEqual(Counter(v["effective"].values()), Counter(e.tally["decision_counts"]))
                with self.assertRaises(families.FamilyStop):
                    families.validate_decisions(e.body, model.rows, review_csv_sha256=CSV_PIN, pre_sha256=PRE_PIN,
                                                alias_sha256=a_b.sha256, pre=model)


class TestCodeBinding(D5Case):
    """Without the pin patch: the tool reads families._code_shas() itself."""
    PIN_CODE = False

    def test_the_tool_binds_the_current_code(self):
        drift = sorted(k for k in set(PINNED_CODE) | set(families._code_shas())
                       if PINNED_CODE.get(k) != families._code_shas().get(k))
        if not drift:
            self.ok(draft_args())
            return
        line = self.stop(draft_args(), "R7", contains="current code differs from registry.pre-review.json 'code'")
        self.assertIn(str(drift), line["reason"])
        self.assertIn("rebuild different pre-review bytes", line["reason"])

    def test_each_module_sha_is_bound(self):
        for module in ("families", "family_keys", "family_pin", "family_screen", "a_new_module"):
            with self.subTest(module=module), \
                    mock.patch.object(T, "current_code_shas", lambda m=module: dict(PINNED_CODE, **{m: "0" * 64})):
                self.stop(whatif_args(), "R7", contains=f"['{module}']")
        missing = {k: v for k, v in PINNED_CODE.items() if k != "family_pin"}
        with mock.patch.object(T, "current_code_shas", lambda: dict(missing)):
            self.stop(whatif_args(), "R7", contains="['family_pin']")

    def test_check_and_dryrun_refuse_under_drift(self):
        with pinned_code():
            draft, dsha = self.draft()
        with mock.patch.object(T, "current_code_shas", lambda: dict(PINNED_CODE, family_screen="0" * 64)):
            self.stop(check_args(draft, dsha), "R7", contains="['family_screen']")
            self.stop(dryrun_args(draft, dsha), "R7", contains="['family_screen']")


if __name__ == "__main__":
    unittest.main()
