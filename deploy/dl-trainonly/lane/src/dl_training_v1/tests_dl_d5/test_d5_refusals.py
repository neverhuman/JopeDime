"""Refusals R1..R12 (exit 2, nothing written, the refusal named on stderr). Real-mode cases run against copies of the
bound g2 inputs inside injected roots; R11 uses a stub build (the real simulation is in test_d5_simulate)."""
import json
import unittest
from pathlib import Path
from unittest import mock

from tests_dl_d5.fixtures_dl_d5 import (ALIAS, CONFIRM_TEXT, CSV, PRE, PRE_PIN, REC_COUNTS, REC_OUTCOME,
                                        RELAY_STYLE_TEXT, D5Case, T, csv_bytes, directive_args, draft_args, drop_flags,
                                        fid, put_flag, real_args, row, sha, sim_ok, stub_build, subset_basis,
                                        synthetic_rows, whatif_args, world_whatif_args, write_world)
from dltrain import common, families


class TestR1Answers(D5Case):
    def test_answer_syntax_and_ranges(self):
        for answers in ("", "Q1=a,Q2=a,Q3=a", "Q1=a,Q2=a,Q3=a,Q4=a,Q1=a", "Q1=a,Q2=a,Q3=a,Q4=a,Q5=a",
                        "Q1=c,Q2=a,Q3=a,Q4=a", "Q1=a,Q2=d,Q3=a,Q4=a", "Q1=a,Q2=a,Q3=a,Q4=c", "Q1=A,Q2=a,Q3=a,Q4=a",
                        "Q1=a, Q2=a,Q3=a,Q4=a", "Q1:a,Q2=a,Q3=a,Q4=a"):
            with self.subTest(answers=answers):
                self.stop(whatif_args(answers=answers), "R1")


class TestR2PackageGranularity(D5Case):
    def test_q3b_refused(self):
        line = self.stop(whatif_args(answers="Q1=a,Q2=a,Q3=b,Q4=a"), "R2")
        self.assertIn("rdatasets_pkg", line["reason"])


class TestR3SplitRegistry(D5Case):
    def test_real_mode_needs_the_split_registry_confirmation(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        argv = [a for a in real_args(bdir, self.roots.receipts, msg, msha) if a != "--split-registry-confirmed"]
        self.stop(argv, "R3", build_fn=sim_ok())


class TestR4Rights(D5Case):
    def test_rights_absent_or_not_a_dl_d5_answer(self):
        self.stop(drop_flags(whatif_args(), "--rights"), "R4")
        for bad in ("prohibited", "Unknown", ""):
            with self.subTest(rights=bad):
                self.stop(whatif_args(rights=bad), "R4")

    def test_owner_text_q5c_is_refused(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT.replace("Q5=a", "Q5=c"))
        self.stop(real_args(bdir, self.roots.receipts, msg, msha), "R4", build_fn=sim_ok())


class TestR5Kinds(D5Case):
    def world(self, mutate):
        rows = synthetic_rows()
        mutate(rows)
        return write_world(self.tmp / "w", rows)

    def test_uncovered_kinds(self):
        for kind in ("withdrawn_family_key", "hf_owner", "pmlb_stem", "socrata_4x4_cross_host", "hunt_semantic_signature",
                     "brand_new_kind"):
            with self.subTest(kind=kind):
                w = self.world(lambda rows: rows.append(row("x" + kind, kind, [fid("a"), fid("b")], "unresolved",
                                                            options="independent;union;union_subset;unresolved")))
                self.stop(world_whatif_args(w), "R5", contains=f"row kind {kind} is not covered by dl-d5-map/v2")

    def test_recommendation_outside_the_options(self):
        def csv_opts(rows):
            rows[2]["decision_options"] = "union;unresolved"            # publisher, recommendation independent
        self.stop(world_whatif_args(self.world(csv_opts)), "R5",
                  contains="recommendation independent is not among the row's decision options")

        def fam_opts(rows):
            rows[5]["recommendation"], rows[5]["decision_options"] = "maybe", "maybe;unknown"   # rights_note
        self.stop(world_whatif_args(self.world(fam_opts)), "R5",
                  contains="recommendation maybe is not among the row's decision options")

    def test_decision_outside_the_options(self):
        def only_unseal(rows):
            rows[4]["decision_options"] = "unseal"                      # sealed_adjacent; Q2=a decides keep_sealed
        self.stop(world_whatif_args(self.world(only_unseal)), "R5",
                  contains="decision keep_sealed is outside the row's decision options")


class TestR6UnionSubsetSets(D5Case):
    def test_basis_and_set_rules(self):
        a, b, c, d = (fid(x) for x in "abcd")
        tag = "agent-generated (deterministic rule, not a decision): pairwise nds/name/legacy evidence; sets="
        ab, ad, bc = sorted([a, b]), sorted([a, d]), sorted([b, c])
        compact = lambda v: json.dumps(v, separators=(",", ":"))  # noqa: E731
        bad = {
            "two_sets_markers": (subset_basis([ab]) + " sets=[]", "exactly one 'sets='"),
            "other_rule_text": ("agent-generated: pairwise evidence; sets=" + compact([ab]), "union_subset rule text"),
            "one_member": (tag + compact([[a]]), ">= 2 distinct families"),
            "repeated_member": (tag + compact([[a, a]]), ">= 2 distinct families"),
            "unsorted": (tag + compact([sorted([a, b], reverse=True)]), "not sorted"),
            "outside_row": (tag + compact([ad]), "outside the row's family_ids"),
            "overlap": (tag + compact([ab, bc]), "not pairwise disjoint"),
            "not_fam_id": (tag + compact([["fam-xyz", a]]), "fam-<64hex>"),
            "non_canonical": (tag + json.dumps([ab]), "not canonical JSON"),
            "empty": (tag + "[]", "non-empty list"),
            "not_json": (tag + "[[" + a, "union_subset rule text"),
            "not_strict_json": (tag + '[["' + a + '",NaN]]', "not strict JSON"),
        }
        for name, (basis, reason) in bad.items():
            with self.subTest(case=name):
                rows = synthetic_rows()
                rows[1]["recommendation_basis"] = basis
                w = write_world(self.tmp / name, rows)
                if w["shape"] is None:          # the shape itself cannot be computed: use a shape file of this csv
                    ok = write_world(self.tmp / (name + "-ok"), synthetic_rows())
                    doc = json.loads(ok["shape"][0].read_bytes())
                    doc["review_csv_sha256"] = w["csv"][1]
                    p = self.tmp / (name + "-shape.json")
                    p.write_bytes(common.canonical(doc))
                    w["shape"] = (p, sha(p.read_bytes()))
                self.stop(world_whatif_args(w), "R6", contains=reason)


class TestR7Binding(D5Case):
    def test_pins_and_files(self):
        self.stop(draft_args(csv_pin="0" * 64), "R7")
        self.stop(draft_args(pre_pin=PRE_PIN.upper()), "R7")
        self.stop(draft_args(alias_path=self.tmp / "missing.json"), "R7")

    def test_pre_review_registry_fields(self):
        rows = synthetic_rows()
        for name, kw in {"phase": {"phase": "final"}, "csv": {"bind_csv": "1" * 64}, "alias": {"bind_alias": "2" * 64},
                         "code": {"code": dict(T.current_code_shas(), families="3" * 64)}}.items():
            with self.subTest(field=name):
                w = write_world(self.tmp / name, rows, **kw)
                self.stop(world_whatif_args(w), "R7")

    def test_review_csv_ids_and_columns(self):
        rows = synthetic_rows()
        rows[2]["review_group_id"] = rows[1]["review_group_id"]
        w = write_world(self.tmp / "dup", rows)
        self.stop(world_whatif_args(w, shape=self.shape_for(w)), "R7")
        cb = csv_bytes(synthetic_rows()).replace(b"review_group_id,kind", b"group_id,kind", 1)
        w = write_world(self.tmp / "cols", synthetic_rows(), csv_override=cb)
        self.stop(world_whatif_args(w, shape=self.shape_for(w)), "R7")

    def shape_for(self, w):
        """A valid shape file bound to w's review.csv bytes (whose rows cannot be parsed themselves)."""
        ok = write_world(self.tmp / ("ok-" + w["csv"][1][:8]), synthetic_rows())
        doc = json.loads(ok["shape"][0].read_bytes())
        doc["review_csv_sha256"] = w["csv"][1]
        p = self.tmp / f"shape-{w['csv'][1][:8]}.json"
        p.write_bytes(common.canonical(doc))
        return p, sha(p.read_bytes())


class TestR8Shape(D5Case):
    def shape_file(self, doc) -> list:
        p = self.tmp / f"shape-{sha(common.canonical(doc))[:8]}.json"
        p.write_bytes(common.canonical(doc))
        return [str(p), "--expect-shape-sha256", sha(p.read_bytes())]

    def with_shape(self, argv, spec) -> list:
        i = argv.index("--expect-shape")
        return argv[:i + 1] + spec + argv[i + 2:]

    def test_shape_mismatch(self):
        base = T.BUILTIN_SHAPES["builtin:aad99b1a"]
        for name, doc in {"rows": dict(base, rows=414), "digest": dict(base, rows_digest="0" * 64),
                          "code": dict(base, code=dict(base["code"], families="1" * 64)),
                          "options": dict(base, decision_options=dict(base["decision_options"], rights_note=["unknown"]))
                          }.items():
            with self.subTest(changed=name):
                self.stop(self.with_shape(whatif_args(), self.shape_file(doc)), "R8", contains="shape differs")

    def test_shape_binds_its_review_csv(self):
        other = dict(T.BUILTIN_SHAPES["builtin:aad99b1a"], review_csv_sha256="c" * 64)
        self.stop(self.with_shape(whatif_args(), self.shape_file(other)), "R8", contains="binds review.csv cccccccccccc")
        w = write_world(self.tmp / "w", synthetic_rows())
        argv = world_whatif_args(w)
        self.stop(argv[:argv.index("--expect-shape") + 1] + ["builtin:aad99b1a"] + argv[argv.index("--expect-shape") + 4:],
                  "R8", contains="binds review.csv aad99b1a9af6")

    def test_shape_source_errors(self):
        argv = whatif_args()
        i = argv.index("--expect-shape")
        self.stop(argv[:i + 1] + ["builtin:c7377cf8"] + argv[i + 2:], "R8", contains="superseded")
        self.stop(argv[:i] + argv[i + 2:], "R8")
        p = self.tmp / "s.json"
        p.write_bytes(common.canonical(T.BUILTIN_SHAPES["builtin:aad99b1a"]))
        self.stop(argv[:i + 1] + [str(p), "--expect-shape-sha256", "f" * 64] + argv[i + 2:], "R8")
        v1 = {k: v for k, v in T.BUILTIN_SHAPES["builtin:aad99b1a"].items() if k != "review_csv_sha256"}
        self.stop(self.with_shape(argv, self.shape_file(dict(v1, schema="jope.dl-d5-review-shape/v1"))), "R8",
                  contains="not a jope.dl-d5-review-shape/v2")
        self.stop(self.with_shape(argv, self.shape_file(v1)), "R8", contains="binds no review.csv")

    def test_an_edited_builtin_literal_is_refused(self):
        edited = dict(T.BUILTIN_SHAPES["builtin:aad99b1a"], rows=999)
        with mock.patch.dict(T.BUILTIN_SHAPES, {"builtin:aad99b1a": edited}):
            self.stop(whatif_args(), "R8", contains="no longer hashes to its pin")


class TestR9Provenance(D5Case):
    def test_draft_takes_only_the_relay_letters_and_no_rights_choice(self):
        argv = draft_args()
        i = argv.index("--relay-sha256")
        self.stop(argv[:i + 1] + ["0" * 64] + argv[i + 2:], "R9")
        self.stop(argv[:i] + argv[i + 2:], "R9")
        self.stop(draft_args(answers="Q1=b,Q2=a,Q3=a,Q4=a"), "R9")
        self.stop(draft_args(answers="Q1=a,Q2=c,Q3=a,Q4=a"), "R9")
        line = self.stop(draft_args(rights="unknown"), "R9", contains="adds no choice the relay did not make")
        self.assertIn("--rights must be undecided", line["reason"])

    def real(self, text=CONFIRM_TEXT, *extra, mutate=None, **kw):
        bdir = self.bound_copy()
        self._n = getattr(self, "_n", 0) + 1
        msg, msha = self.message(text, name=f"owner-message-{self._n}.txt")
        argv = real_args(bdir, self.roots.receipts, msg, msha, *extra, **kw)
        return mutate(argv) if mutate else argv

    def test_message_rules(self):
        def put(flag, value):
            return lambda argv: put_flag(argv, flag, value)
        cases = {
            "sha_mismatch": (self.real(mutate=put("--message-sha256", "e" * 64)), "differs from its pin"),
            "draft_session": (self.real(mutate=put("--session", "DRAFT-UNCONFIRMED-RELAY-b785a123")),
                              "starts with DRAFT, SIMULATION or WHATIF"),
            "simulation_session": (self.real(mutate=put("--session", "simulation-not-a-decision")),
                                   "starts with DRAFT, SIMULATION or WHATIF"),
            "whatif_session": (self.real(mutate=put("--session", "WHATIF-x")), "starts with DRAFT, SIMULATION or WHATIF"),
            "draft_word_in_session": (self.real(mutate=put("--session", "screen 1 (draft relay)")), "CL-2"),
            "empty_session": (self.real(mutate=put("--session", " ")), "session is required"),
            "bad_utc": (self.real(mutate=put("--message-utc", "2026-10-05 11:00")), "ISO-8601 UTC"),
            "relay_style_text": (self.real(RELAY_STYLE_TEXT), "exactly one line"),
            "answers_differ": (self.real(CONFIRM_TEXT.replace("Q2=a", "Q2=b")), "not the --answers letters"),
            "q5_vs_rights": (self.real(CONFIRM_TEXT.replace("Q5=a", "Q5=b")), "Q5=b (rights_note undecided)"),
            "no_kind": (self.real(mutate=lambda a: drop_flags(a, "--provenance-kind")), "--provenance-kind"),
            "unknown_kind": (self.real(mutate=put("--provenance-kind", "agent_relay")), "--provenance-kind"),
            "directive_args_with_message": (self.real(CONFIRM_TEXT, "--directive-version", "v1"),
                                            "directive arguments go only with"),
            "non_utf8_message": (self.real_bytes(b"DL-D5: Q1=a \xff"), "not UTF-8"),
        }
        for name, (argv, reason) in cases.items():
            with self.subTest(case=name):
                self.stop(argv, "R9", contains=reason, build_fn=sim_ok())

    def real_bytes(self, raw: bytes):
        bdir = self.bound_copy()
        p = self.tmp / "owner-message-bytes.bin"
        p.write_bytes(raw)
        return real_args(bdir, self.roots.receipts, p, sha(raw))

    def test_relay_sha_refused_as_owner_message(self):
        argv = self.real()
        msha = argv[argv.index("--message-sha256") + 1]
        with mock.patch.object(T, "RELAY_SHA256", msha):
            self.stop(argv, "R9", contains="relayed DL-D5 paste", build_fn=sim_ok())

    def test_directive_rules(self):
        bdir = self.bound_copy()
        d, dsha = self.directives("header", "DL-D5 note", CONFIRM_TEXT)
        good = directive_args(bdir, self.roots.receipts, d, dsha, version="v1", lines="3")
        for name, (argv, reason) in {
                "lines_out_of_range": (put_flag(good, "--directive-lines", "9"), "outside the file"),
                "lines_without_the_line": (put_flag(good, "--directive-lines", "2"), "not the affirmative"),
                "two_lines": (put_flag(good, "--directive-lines", "2-3"), "exactly one line"),
                "missing_version": (drop_flags(good, "--directive-version"), "--directive-version"),
                "sha_mismatch": (put_flag(good, "--directive-sha256", "d" * 64), "differs from its pin"),
                "message_args": (good + ["--session", "s"], "message arguments"),
                "question_time": (good + ["--question-posted-utc", "2026-10-05T15:00Z"], "message arguments")}.items():
            with self.subTest(case=name):
                self.stop(argv, "R9", contains=reason, build_fn=sim_ok())
        self.ok(good, build_fn=sim_ok())


class TestR10Paths(D5Case):
    def test_mode_arguments(self):
        for mode_args in (draft_args, whatif_args):
            with self.subTest(mode=mode_args.__name__):
                self.stop(mode_args("--out", str(self.tmp / "x.json")), "R10")
                self.stop(mode_args("--split-registry-confirmed"), "R10")
                self.stop(mode_args("--receipt", str(self.tmp / "r.json")), "R10")
                self.stop(mode_args("--question-posted-utc", "2026-10-05T15:00Z"), "R10")
        self.stop(whatif_args("--relay-sha256", T.RELAY_SHA256), "R10", contains="DRAFT argument")
        self.stop(draft_args("--what-if"), "R10", contains="exclude each other")
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        self.stop(real_args(bdir, self.roots.receipts, msg, msha, "--relay-sha256", T.RELAY_SHA256), "R10",
                  build_fn=sim_ok())
        self.stop(drop_flags(real_args(bdir, self.roots.receipts, msg, msha), "--receipt"), "R10", build_fn=sim_ok())

    def test_real_output_and_receipt_locations(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        argv = real_args(bdir, self.roots.receipts, msg, msha)
        self.stop(put_flag(argv, "--out", bdir / "decisions.json"), "R10", build_fn=sim_ok())
        self.stop(put_flag(argv, "--out", self.tmp / "review-decisions.json"), "R10", build_fn=sim_ok())
        self.stop(put_flag(argv, "--receipt", self.tmp / "decisions-expand-1791197000000000.json"), "R10",
                  build_fn=sim_ok())
        self.stop(put_flag(argv, "--receipt", Path(self.roots.receipts) / "expand.json"), "R10", build_fn=sim_ok())
        other = self.bound_copy("../outside")                       # a bound dir outside the families root
        self.stop(real_args(other, self.roots.receipts, msg, msha), "R10", build_fn=sim_ok())

    def test_alias_sources_must_be_the_bound_dirs_own(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        argv = put_flag(real_args(bdir, self.roots.receipts, msg, msha), "--alias-sources", ALIAS)
        self.stop(argv, "R10", contains="builds only from its own alias-sources.json", build_fn=sim_ok())

    def test_split_bound_dir(self):
        bdir = self.bound_copy()
        other = Path(self.roots.families) / "other"
        other.mkdir(parents=True)
        (other / "registry.pre-review.json").write_bytes(PRE.read_bytes())
        msg, msha = self.message(CONFIRM_TEXT)
        argv = put_flag(real_args(bdir, self.roots.receipts, msg, msha), "--pre-review",
                        other / "registry.pre-review.json")
        self.stop(argv, "R10", build_fn=sim_ok())

    def test_existing_files(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        argv = real_args(bdir, self.roots.receipts, msg, msha)
        (bdir / "review-decisions.json").write_bytes(b"{}")
        self.stop(argv, "R10", build_fn=sim_ok())
        (bdir / "review-decisions.json").unlink()
        rp = Path(self.roots.receipts)
        rp.mkdir(parents=True, exist_ok=True)
        (rp / "decisions-expand-1791197000000000.json").write_bytes(b"{}")
        self.stop(argv, "R10", build_fn=sim_ok())
        (rp / "decisions-expand-1791197000000000.json").unlink()
        (rp / T.MESSAGE_NAME.format(sha16=msha[:16])).write_bytes(b"other bytes")
        self.stop(argv, "R10", build_fn=sim_ok())


class TestR11Simulation(D5Case):
    def test_arguments(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        argv = real_args(bdir, self.roots.receipts, msg, msha)
        self.stop([a for a in argv if a != "--simulate"], "R11", build_fn=sim_ok())
        self.stop(drop_flags(argv, "--expect-outcome"), "R11", build_fn=sim_ok())
        for bad in ("families=797,N_pool=652", "families=797,N_pool=652,k=33,bogus=1", "families=797,N_pool=x,k=33",
                    "families=797,families=797,N_pool=652,k=33"):
            with self.subTest(outcome=bad):
                self.stop(put_flag(argv, "--expect-outcome", bad), "R11", build_fn=sim_ok())
        self.stop(whatif_args("--expect-outcome", REC_OUTCOME), "R11")

    def test_simulated_build_must_pass_and_match(self):
        pre, cb = PRE.read_bytes(), CSV.read_bytes()
        cases = {"verdict_stop": stub_build(verdict="STOP", counts=REC_COUNTS, pre=pre, csv_b=cb),
                 "phase_pre_review": stub_build(phase="pre_review", counts=REC_COUNTS, pre=pre, csv_b=cb),
                 "phase_simulation": stub_build(phase="simulation", counts=REC_COUNTS, pre=pre, csv_b=cb),
                 "no_final_registry": stub_build(counts=REC_COUNTS, pre=pre, csv_b=cb, final=None),
                 "pre_review_differs": stub_build(counts=REC_COUNTS, pre=pre + b" ", csv_b=cb),
                 "review_csv_differs": stub_build(counts=REC_COUNTS, pre=pre, csv_b=cb[:-1]),
                 "counts_differ": stub_build(counts=dict(REC_COUNTS, N_pool=640), pre=pre, csv_b=cb)}
        for name, fn in cases.items():
            with self.subTest(case=name):
                self.stop(whatif_args("--simulate", "--expect-outcome", REC_OUTCOME), "R11", build_fn=fn)

        def raising(*a, **k):
            raise families.FamilyStop("2 circuit_breaker row(s) undecided: STOP")
        self.stop(whatif_args("--simulate", "--expect-outcome", REC_OUTCOME), "R11", build_fn=raising)

    def test_a_real_simulation_without_an_owner_record_stops(self):
        bdir = self.bound_copy()
        msg, msha = self.message(CONFIRM_TEXT)
        fn = stub_build(counts=REC_COUNTS, pre=PRE.read_bytes(), csv_b=CSV.read_bytes(),
                        deltas={"provenance": {"draft": False, "simulation": False, "record": None}})
        line = self.stop(real_args(bdir, self.roots.receipts, msg, msha), "R11", writes=True, build_fn=fn)
        self.assertIn("did not resolve a pinned owner record", line["reason"])
        self.assertIn("the owner message stays stored", line["reason"])        # an owner record, not a decision
        self.assertEqual(self.files(), sorted(["owner-message.txt", *self.bound_files(bdir),
                                               f"receipts/P2/dl-d5-owner-message-{msha[:16]}.txt"]))

    def bound_files(self, bdir):
        return [str((bdir / n).relative_to(self.tmp)) for n in ("alias-sources.json", "registry.pre-review.json",
                                                                 "review.csv")]

    def test_stub_pass_writes_exactly_the_file_and_its_receipt(self):
        seen = []
        code, out, err = self.cli(whatif_args("--simulate", "--expect-outcome", REC_OUTCOME), build_fn=sim_ok(seen=seen))
        self.assertEqual(code, 0, err)
        s = json.loads(out)
        files = self.files()
        self.assertEqual(len(files), 2, files)
        self.assertTrue(files[0].startswith("receipts/P2/staging/WHATIF-NOT-A-DECISION-p2-decisions-expand-"), files)
        self.assertEqual(files[1], f"stage/WHATIF-NOT-A-DECISION-review-decisions-{s['out']['sha256'][:16]}.json")
        self.assertEqual(oct((self.tmp / files[1]).stat().st_mode & 0o777), "0o444")
        # the simulation read the in-memory rehearsal variant (never written), as a staging build
        (raw, kw), = seen
        self.assertEqual(kw, {})
        reh = json.loads(raw)
        self.assertEqual((reh["schema"], reh["rehearsal_of"]["sha256"]), (families.DECISIONS_SCHEMA, s["out"]["sha256"]))
        self.assertNotIn(sha(raw), {sha((self.tmp / f).read_bytes()) for f in files})


class TestR12Unpromotable(D5Case):
    def test_a_families_acceptance_of_an_unconfirmed_file_stops(self):
        def accept_all(raw, rows, **kw):
            return {"effective": {r["review_group_id"]: "unresolved" for r in rows}}
        with mock.patch.object(families, "validate_decisions", accept_all):
            self.stop(whatif_args(), "R12", contains="accepted an unconfirmed file")


class TestUsage(D5Case):
    def test_parser_errors_are_stops(self):
        self.stop(["nonsense"], "USAGE")
        self.stop(["expand", "--answers", "Q1=a"], "USAGE")
        self.stop(["check", "--decisions", "x"], "USAGE")
        self.stop(whatif_args("--unknown-flag"), "USAGE")
        self.stop(["dryrun", "--full-rehash"], "USAGE")


if __name__ == "__main__":
    unittest.main()
