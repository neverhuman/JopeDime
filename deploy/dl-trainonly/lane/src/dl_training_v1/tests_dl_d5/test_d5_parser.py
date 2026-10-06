"""The strict owner-text grammar dl-d5-confirm/v1 (D-MAP-1, D-MAP-2, D-MAP-3, D-MAP-7) at the unit level and through
the real-mode CLI (stub simulation): exactly one affirmative confirmation line, bound to the g2 review.csv sha12, with
Q1..Q4 equal to --answers, Q5 equal to --rights in both directions, the split-registry sentence, and a message time not
before the question set. The four texts the D-mapping reviewer showed were accepted by version 1 are refused here."""
import unittest
from pathlib import Path

from tests_dl_d5.fixtures_dl_d5 import (A4, CONFIRM_TEXT, CSV_PIN, G1_CSV_PIN, MESSAGE_UTC, POSTED_UTC, RELAY_STYLE_TEXT,
                                        SESSION, D5Case, T, directive_args, drop_flags, put_flag, real_args, sha, sim_ok)

LINE = CONFIRM_TEXT
# D-MAP-1 evidence: each was ACCEPTED by version 1 (letters gathered from anywhere, 'split registry' as a substring)
REVIEW_PROBES = {
    "rejected_suggestion": "Do not apply your suggested line (Q1=a, Q2=a, Q3=a, Q4=a, Q5=a; split-registry excluded seals "
                           "nothing). Wait for g2.",
    "question": "Before I decide: what happens with Q1=a, Q2=a, Q3=a, Q4=a, Q5=a? Does the split-registry excluded list "
                "really seal nothing?",
    "negated_split": "DL-D5 confirmed for review.csv aad99b1a9af6: Q1=a, Q2=a, Q3=a, Q4=a, Q5=a; I do NOT confirm that the "
                     "split-registry excluded list seals nothing.",
    "other_decision_label": "DL-D13: Q1=a. DL-D5: Q2=a, Q3=a, Q4=a, Q5=a; split-registry excluded seals nothing.",
}
# D-MAP-2 evidence: rights wording that version 1 accepted with --rights unknown/undecided
RIGHTS_PROBES = ("DL-D5 confirmed for review.csv aad99b1a9af6: Q1=a, Q2=a, Q3=a, Q4=a; split-registry excluded seals "
                 "nothing; leave the rights_note rows undecided.",
                 "DL-D5 confirmed for review.csv aad99b1a9af6: Q1=a, Q2=a, Q3=a, Q4=a; split-registry excluded seals "
                 "nothing; rights_note=prohibited.")


def parse(text, csv_sha=CSV_PIN, **kw):
    return T.parse_confirmation(text, csv_sha, **kw)


class TestGrammarUnit(unittest.TestCase):
    def refused(self, text, code="R9", contains=None, **kw):
        with self.assertRaises(T.Refused) as cm:
            parse(text, **kw)
        self.assertEqual(cm.exception.code, code, cm.exception.reason)
        if contains:
            self.assertIn(contains, cm.exception.reason)
        return cm.exception

    def test_the_line_parses(self):
        self.assertEqual(parse(LINE), dict(A4, Q5="a", review_csv_sha12="aad99b1a9af6"))

    def test_tolerated_variation(self):
        for text in (LINE + "\n", "  " + LINE + "  \r\n", LINE[:-1],
                     "dl-d5 CONFIRMED for REVIEW.CSV AAD99B1A9AF6: q1=A, q2=a, q3=a, q4=a, q5=a; Split-Registry excluded "
                     "seals nothing.",
                     "DL-D5  confirmed for review.csv aad99b1a9af6 : Q1 = a ,Q2=a,  Q3=a,Q4=a ,Q5=a ;split-registry "
                     "excluded  seals nothing ."):
            with self.subTest(text=text):
                self.assertEqual(parse(text), dict(A4, Q5="a", review_csv_sha12="aad99b1a9af6"))
        self.assertEqual(parse("DL-D5 confirmed for review.csv aad99b1a9af6: Q1=b, Q2=c, Q3=a, Q4=b, Q5=b; split-registry "
                               "excluded seals nothing.")["Q2"], "c")

    def test_review_probes_are_refused(self):
        for name, text in REVIEW_PROBES.items():
            with self.subTest(probe=name):
                self.refused(text)
        for text in RIGHTS_PROBES:
            with self.subTest(text=text[-40:]):
                self.refused(text)

    def test_the_line_with_anything_else_is_refused(self):
        cases = {
            "second_line": (LINE + "\nThanks.", "exactly one line"),
            "leading_line": ("Here it is:\n" + LINE, "exactly one line"),
            "trailing_words": (LINE + " Go ahead.", "not the affirmative"),
            "leading_words": ("OK: " + LINE, "not the affirmative"),
            "question_mark": (LINE[:-1] + "?", "not the affirmative"),
            "quoted": ('"' + LINE + '"', "not the affirmative"),
            "not_confirmed": (LINE.replace("confirmed", "not confirmed"), "not the affirmative"),
            "unconfirmed": (LINE.replace("confirmed", "unconfirmed"), "not the affirmative"),
            "other_decision": (LINE.replace("DL-D5", "DL-D13"), "not the affirmative"),
            "no_q5": (LINE.replace(", Q5=a", ""), "not the affirmative"),
            "order": (LINE.replace("Q1=a, Q2=a", "Q2=a, Q1=a"), "not the affirmative"),
            "two_letters": (LINE.replace("Q1=a", "Q1=ab"), "not the affirmative"),
            "paren_letter": (LINE.replace("Q1=a", "Q1=(a)"), "not the affirmative"),
            "annotation": (LINE.replace("Q1=a", "Q1=a (264 link rows)"), "not the affirmative"),
            "no_split": (LINE.replace("; split-registry excluded seals nothing.", "."), "not the affirmative"),
            "split_negated": (LINE.replace("seals nothing", "seals something"), "not the affirmative"),
            "split_spaced": (LINE.replace("split-registry", "split registry"), "not the affirmative"),
            "en_dash": (LINE.replace("split-registry", "split–registry"), "not the affirmative"),
            "long_s": (LINE.replace("seals", "ſeals"), "not the affirmative"),
            "full_sha": (LINE.replace("aad99b1a9af6", CSV_PIN), "not the affirmative"),
            "short_sha": (LINE.replace("aad99b1a9af6", "aad99b1a"), "not the affirmative"),
            "no_sha": (LINE.replace(" for review.csv aad99b1a9af6", ""), "not the affirmative"),
            "empty": ("  \n ", "empty"),
            "unicode_line_sep": (LINE.replace("; ", "; "), "not the affirmative"),
        }
        for name, (text, reason) in cases.items():
            with self.subTest(case=name):
                self.refused(text, contains=reason)

    def test_the_sha12_binds_the_bound_review_csv(self):
        g1_line = LINE.replace("aad99b1a9af6", G1_CSV_PIN[:12])
        e = self.refused(g1_line, contains="not the bound review.csv aad99b1a9af6")
        self.assertIn("c7377cf8bc86", e.reason)
        self.refused(LINE, csv_sha=G1_CSV_PIN, contains="confirms review.csv aad99b1a9af6")

    def test_directive_bullet(self):
        self.assertEqual(parse("- " + LINE, directive=True)["Q5"], "a")
        self.assertEqual(parse("* " + LINE, directive=True)["Q5"], "a")
        self.refused("- " + LINE)                                    # a message carries no bullet
        self.refused("- - " + LINE, directive=True)
        self.refused("DL-D5: " + LINE, directive=True)

    def test_relay_style_text_never_parses(self):
        self.refused(RELAY_STYLE_TEXT, contains="exactly one line")
        self.refused(RELAY_STYLE_TEXT.replace("\n", " "), contains="not the affirmative")


class TestLetters(unittest.TestCase):
    def letters(self, q5="a", **over):
        return dict(A4, Q5=q5, review_csv_sha12="aad99b1a9af6", **over)

    def refused(self, parsed, answers, rights, code, contains):
        with self.assertRaises(T.Refused) as cm:
            T.check_letters(parsed, answers, rights)
        self.assertEqual(cm.exception.code, code)
        self.assertIn(contains, cm.exception.reason)

    def test_q1_q4_equal_answers(self):
        self.assertEqual(T.check_letters(self.letters(), dict(A4), "unknown")["rights_note"], "unknown")
        self.refused(self.letters(Q2="b"), dict(A4), "unknown", "R9", "not the --answers letters")
        self.refused(self.letters(), dict(A4, Q4="b"), "unknown", "R9", "not the --answers letters")

    def test_q5_and_rights_in_both_directions(self):
        self.assertEqual(T.check_letters(self.letters("b"), dict(A4), "undecided")["rights_note"], "undecided")
        self.refused(self.letters("a"), dict(A4), "undecided", "R9", "Q5=a (rights_note unknown), not --rights undecided")
        self.refused(self.letters("b"), dict(A4), "unknown", "R9", "Q5=b (rights_note undecided), not --rights unknown")
        self.refused(self.letters("c"), dict(A4), "unknown", "R4", "prohibited")
        self.refused(self.letters("d"), dict(A4), "unknown", "R9", "Q5 must be a")


class TestMessageProvenanceUnit(unittest.TestCase):
    def prov(self, text=LINE, *, utc=MESSAGE_UTC, posted=POSTED_UTC, session=SESSION, kind="owner_verbatim_message",
             rights="unknown"):
        raw = text.encode()
        return T.message_provenance(kind, session, utc, T.Bound(Path("m"), raw, sha(raw)), dict(A4), rights, CSV_PIN,
                                    posted)

    def refused(self, contains, **kw):
        with self.assertRaises(T.Refused) as cm:
            self.prov(**kw)
        self.assertEqual(cm.exception.code, "R9")
        self.assertIn(contains, cm.exception.reason)

    def test_accepted(self):
        prov, checks, conf = self.prov()
        self.assertEqual(prov, {"kind": "owner_verbatim_message", "session": SESSION, "message_utc": MESSAGE_UTC,
                                "message_sha256": sha(LINE.encode()), "message_text": LINE})
        self.assertEqual(conf, {"grammar": "dl-d5-confirm/v1", "review_csv_sha12": "aad99b1a9af6",
                                "letters": dict(A4, Q5="a"), "question_posted_utc": POSTED_UTC})
        self.assertEqual(checks["parsed"]["review_csv_sha12"], "aad99b1a9af6")
        self.assertEqual(self.prov(kind="owner_session_selections")[0]["kind"], "owner_session_selections")
        self.assertEqual(self.prov(utc=POSTED_UTC)[0]["message_utc"], POSTED_UTC)     # equal time is not earlier

    def test_time_rules(self):
        self.refused("predates the question set", utc="2026-10-05T14:59:59.999Z")
        self.refused("predates the question set", utc="2026-10-04T23:00Z")
        self.refused("ISO-8601", utc="2026-10-05 15:30")
        self.refused("--question-posted-utc", posted=None)
        self.refused("--question-posted-utc", posted="yesterday")

    def test_session_rules(self):
        for s in ("DRAFT-UNCONFIRMED-RELAY-b785a123", "simulation", "WHATIF-NOT-A-DECISION", " "):
            with self.subTest(session=s):
                self.refused("session", session=s)

    def test_utc_key_order(self):
        self.assertLess(T.utc_key("2026-10-05T15:00Z"), T.utc_key("2026-10-05T15:00:00.000001Z"))
        self.assertEqual(T.utc_key("2026-10-05T15:00Z"), T.utc_key("2026-10-05T15:00:00.000Z"))
        self.assertIsNone(T.utc_key("2026-10-05T15:00:00+00:00"))


class TestRealModeOwnerText(D5Case):
    """Through the CLI: the strict grammar decides before anything is written."""

    def real(self, text=LINE, *extra, **kw):
        bdir = self.bound_copy()
        self._n = getattr(self, "_n", 0) + 1
        msg, msha = self.message(text, name=f"owner-message-{self._n}.txt")
        return real_args(bdir, self.roots.receipts, msg, msha, *extra, **kw)

    def test_review_probes_through_the_cli(self):
        for name, text in REVIEW_PROBES.items():
            with self.subTest(probe=name):
                self.stop(self.real(text), "R9", build_fn=sim_ok())

    def test_bindings_through_the_cli(self):
        cases = {
            "g1_sha12": (self.real(LINE.replace("aad99b1a9af6", G1_CSV_PIN[:12])), "R9", "not the bound review.csv"),
            "q5_b_vs_unknown": (self.real(LINE.replace("Q5=a", "Q5=b")), "R9", "not --rights unknown"),
            "q5_a_vs_undecided": (self.real(rights="undecided"), "R9", "not --rights undecided"),
            "q5_c": (self.real(LINE.replace("Q5=a", "Q5=c")), "R4", "prohibited"),
            "answers_differ": (self.real(LINE.replace("Q2=a", "Q2=b")), "R9", "not the --answers letters"),
            "message_before_question": (self.real(message_utc="2026-10-05T14:22:45Z"), "R9", "predates"),
            "no_question_time": (drop_flags(self.real(), "--question-posted-utc"), "R9", "--question-posted-utc"),
            "relay_style": (self.real(RELAY_STYLE_TEXT), "R9", "exactly one line"),
        }
        for name, (argv, code, reason) in cases.items():
            with self.subTest(case=name):
                self.stop(argv, code, contains=reason, build_fn=sim_ok())

    def test_q5b_with_rights_undecided_is_accepted(self):
        s = self.ok(self.real(LINE.replace("Q5=a", "Q5=b"), rights="undecided",
                              outcome="families=797,N_pool=640,k=32"), build_fn=sim_ok(counts={"families": 797,
                                                                                               "N_pool": 640}))
        self.assertEqual((s["mode"], s["undecided_rows"]), ("real", 14))
        self.assertNotIn("unknown", s["decision_counts"])

    def test_directive_line_rules(self):
        bdir = self.bound_copy()
        d, dsha = self.directives("# Owner directives v7", "", "- " + LINE, "other line",
                                  "DL-D5 confirmed: Q1=a, Q2=a, Q3=a, Q4=a, Q5=a; split-registry excluded seals nothing.")
        s = self.ok(directive_args(bdir, self.roots.receipts, d, dsha, lines="3"), build_fn=sim_ok())
        self.assertEqual(s["mode"], "real")
        for lines, reason in (("2-3", "exactly one line"), ("4", "not the affirmative"), ("5", "not the affirmative"),
                              ("1", "not the affirmative"), ("9", "outside the file")):
            with self.subTest(lines=lines):
                bdir2 = self.bound_copy(f"g2-{lines}")
                self.stop(directive_args(bdir2, self.roots.receipts, d, dsha, lines=lines, utc=1791197000000001),
                          "R9", contains=reason, build_fn=sim_ok())
        # any file other than the owner's OWNER_DIRECTIVES.md is refused, even with the same name and bytes
        other = self.tmp / "agent" / "OWNER_DIRECTIVES.md"
        other.parent.mkdir()
        other.write_bytes(d.read_bytes())
        bdir3 = self.bound_copy("g2-other")
        self.stop(put_flag(directive_args(bdir3, self.roots.receipts, other, dsha, utc=1791197000000002),
                           "--directive-file", other), "R9", contains="accepts only the owner's directive file",
                  build_fn=sim_ok())


if __name__ == "__main__":
    unittest.main()
