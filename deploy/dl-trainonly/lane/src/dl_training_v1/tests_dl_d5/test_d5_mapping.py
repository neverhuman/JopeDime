"""dl-d5-map/v2 against the real g2 review.csv aad99b1a (no build): the v2 shape and its builtin pin, per-answer decision
counts, verbatim sets, row order, binds, rulings, the real and the unconfirmed document layouts, validate_decisions
acceptance, byte determinism, import and output hygiene."""
import ast
import csv
import io
import json
import os
import re
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from tests_dl_d5.fixtures_dl_d5 import (A4, ALIAS, ALIAS_PIN, BREAKER_ID, CONFIRM_TEXT, CSV, CSV_PIN, MESSAGE_UTC,
                                        POSTED_UTC, PRE, PRE_PIN, REC_DECISIONS, SESSION, SHAPE_PIN, SRC, D5Case, T,
                                        draft_args, pinned_code, sha, whatif_args)
from dltrain import common, families

LINK_EXPECT = {"rdatasets_pkg": {"independent": 37, "union_subset": 14}, "publisher": {"independent": 31, "union_subset": 6},
               "socrata_domain": {"independent": 20}, "github_owner": {"independent": 4, "union_subset": 1},
               "legacy_source_family": {"union_subset": 28}, "tidytuesday_year": {"independent": 2, "union_subset": 1},
               "name_candidate": {"union_subset": 14}, "classic_candidate": {"union_subset": 1, "unresolved": 10},
               "nds_value_weak": {"union_subset": 7}, "multi_member_family": {"confirm_union": 88}}


def inputs():
    return (T.read_pinned(CSV, CSV_PIN, "review.csv"), T.read_pinned(PRE, PRE_PIN, "pre-review"),
            T.read_pinned(ALIAS, ALIAS_PIN, "alias-sources"))


def expansion(answers=None, rights="unknown", mode="what_if", provenance=None, confirmation=None):
    """The in-memory expansion of the bound set (no build), with the R7 code reading pinned to the set's 'code'."""
    c, p, a = inputs()
    with pinned_code():
        return T.build_expansion(mode=mode, csv_b=c, pre_b=p, alias_b=a, answers=dict(answers or A4), rights=rights,
                                 provenance=provenance or dict(T.UNCONFIRMED_PROVENANCE.get(mode, {})),
                                 want_shape=T.BUILTIN_SHAPES["builtin:aad99b1a"], confirmation=confirmation)


def real_expansion(text=CONFIRM_TEXT, rights="unknown"):
    raw = text.encode()
    prov, checks, conf = T.message_provenance("owner_verbatim_message", SESSION, MESSAGE_UTC,
                                              T.Bound(Path("m"), raw, sha(raw)), dict(A4), rights, CSV_PIN, POSTED_UTC)
    return expansion(rights=rights, mode="real", provenance=prov, confirmation=conf), checks


def real_rows():
    return list(csv.DictReader(io.StringIO(CSV.read_bytes().decode("utf-8"))))


class TestShape(D5Case):
    def test_builtin_shape_is_the_bound_review_csv(self):
        rows = T.parse_rows(CSV.read_bytes())
        shape = T.compute_shape(rows, CSV_PIN)
        self.assertEqual(shape, T.BUILTIN_SHAPES["builtin:aad99b1a"])
        self.assertEqual(T.shape_sha256(shape), SHAPE_PIN)
        self.assertEqual(T.BUILTIN_SHAPE_SHA256, {"builtin:aad99b1a": SHAPE_PIN})
        self.assertEqual(shape["link_rows"], {"rows": 264, "independent": 94, "confirm_union": 88, "union_subset": 72,
                                              "unresolved": 10})
        kinds = Counter(r["kind"] for r in rows)
        self.assertEqual((len(rows), kinds["seed_source_mapping"], kinds["sealed_adjacent"], kinds["rights_note"],
                          kinds["coverage_values_only"] + kinds["coverage_exact_only"]), (413, 10, 37, 14, 84))
        self.assertEqual(shape["code"], json.loads(PRE.read_bytes())["code"])      # the frozen family code
        self.assertEqual(shape["decision_options"], {k: list(families.DECISION_OPTIONS[k]) for k in sorted(kinds)})
        self.assertEqual(set(shape), {"schema", *T.SHAPE_KEYS, "by_kind"})

    def test_equal_counts_with_other_content_are_another_shape(self):
        """D-MAP-3: moving one family between two union_subset sets keeps every count but changes rows_digest."""
        rows = T.parse_rows(CSV.read_bytes())
        base = T.compute_shape(rows, CSV_PIN)
        r = next(x for x in rows if x["recommendation"] == "union_subset" and len(T.parse_sets(x)) == 1
                 and len(T.family_ids(x)) > len(T.parse_sets(x)[0]))
        sets = T.parse_sets(r)
        outside = sorted(set(T.family_ids(r)) - set(sets[0]))[0]
        moved = [sorted(sets[0][1:] + [outside])]
        r["recommendation_basis"] = r["recommendation_basis"].split("sets=")[0] + "sets=" + common.canonical(moved).decode()
        other = T.compute_shape(rows, CSV_PIN)
        self.assertEqual({k: v for k, v in other.items() if k != "rows_digest"},
                         {k: v for k, v in base.items() if k != "rows_digest"})
        self.assertNotEqual(other["rows_digest"], base["rows_digest"])
        self.assertEqual(T.shape_diff(base, other), ["rows_digest"])

    def test_shape_cli_raw_and_create_only_out(self):
        code, out, err = self.cli(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN,
                                   "--expect-shape", "builtin:aad99b1a"])
        self.assertEqual(code, 0, err)
        s = json.loads(out)
        self.assertEqual((s["verdict"], s["builtin_equal"]["builtin:aad99b1a"], s["link_rows"]["rows"], s["shape_sha256"]),
                         ("PASS", True, 264, SHAPE_PIN))
        code, raw, err = self.cli(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN, "--raw"])
        self.assertEqual(code, 0, err)
        self.assertEqual(raw.encode(), common.canonical(T.BUILTIN_SHAPES["builtin:aad99b1a"]))
        target = Path(self.roots.stage_receipts) / f"dl-d5-shape-{SHAPE_PIN[:16]}.json"
        w = self.ok(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN, "--out", target])["written"]
        self.assertEqual((w["sha256"], w["existed"], oct(target.stat().st_mode & 0o777)), (SHAPE_PIN, False, "0o444"))
        self.assertTrue(self.ok(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN, "--out", target])
                        ["written"]["existed"])
        code, out, err = self.cli(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN, "--expect-shape", target,
                                   "--expect-shape-sha256", SHAPE_PIN])
        self.assertEqual(code, 0, err)
        self.stop(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN, "--out",
                   Path(self.roots.stage_receipts) / "shape.json"], "R10")
        self.stop(["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN, "--out",
                   self.tmp / f"dl-d5-shape-{SHAPE_PIN[:16]}.json"], "R10")


class TestMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = real_rows()
        cls.kind = {r["review_group_id"]: r["kind"] for r in cls.rows}
        cls.rec = {r["review_group_id"]: r["recommendation"] for r in cls.rows}
        cls.e, cls.checks = real_expansion()

    def by_kind(self, e):
        return e.tally["by_kind"]

    def test_q1a_link_rows_take_their_recommendation(self):
        bk = self.by_kind(self.e)
        for kind, want in LINK_EXPECT.items():
            self.assertEqual(bk[kind], want, kind)
        link = Counter(d["decision"] for d in self.e.doc["decisions"] if self.kind[d["review_group_id"]] in T.LINK_KINDS)
        self.assertEqual(dict(link), {"independent": 94, "confirm_union": 88, "union_subset": 72, "unresolved": 10})
        self.assertEqual(sum(link.values()), 264)
        self.assertEqual(bk["seed_source_mapping"], {"confirm_mapping": 10})

    def test_q2a_keeps_all_37_seals(self):
        self.assertEqual(self.by_kind(self.e)["sealed_adjacent"], {"keep_sealed": 37})
        self.assertEqual(self.e.doc["rulings"]["provisional_seals"], "keep")

    def test_q4a_residual_rows_unresolved(self):
        bk = self.by_kind(self.e)
        self.assertEqual(bk["circuit_breaker"], {"unresolved": 1})
        self.assertEqual(bk["lineage_b_related_class"], {"unresolved": 2})
        self.assertEqual(bk["lineage_b_audit_family"], {"unresolved": 1})
        self.assertEqual(bk["coverage_values_only"]["unresolved"] + bk["coverage_exact_only"]["unresolved"], 84)
        dec = {d["review_group_id"]: d["decision"] for d in self.e.doc["decisions"]}
        self.assertEqual(dec[BREAKER_ID], "unresolved")
        audit = [g for g, k in self.kind.items() if k == "lineage_b_audit_family"]
        self.assertEqual([self.rec[g] for g in audit], ["union_subset"])        # recommended, but not applied
        self.assertEqual([dec[g] for g in audit], ["unresolved"])

    def test_q5a_rights_unknown_and_totals(self):
        self.assertEqual(self.by_kind(self.e)["rights_note"], {"unknown": 14})
        self.assertEqual(self.e.tally["decision_counts"], REC_DECISIONS)
        self.assertEqual((self.e.tally["decisions"], self.e.tally["undecided_rows"]), (413, 0))
        self.assertEqual(self.e.doc["class_decisions"], [])

    def test_sets_copied_verbatim_in_row_order(self):
        rx = T.UNION_SUBSET_BASIS
        by_id = {r["review_group_id"]: r for r in self.rows}
        subset = [d for d in self.e.doc["decisions"] if d["decision"] == "union_subset"]
        self.assertEqual(len(subset), 72)
        for d in subset:
            text = rx.fullmatch(by_id[d["review_group_id"]]["recommendation_basis"]).group(1)
            self.assertEqual(d["sets"], json.loads(text))
            self.assertEqual(common.canonical(d["sets"]).decode(), text)
        self.assertEqual(sum(len(d["sets"]) for d in subset), 80)                # the audit row's set is not applied
        self.assertTrue(all(set(d) == {"review_group_id", "decision"} for d in self.e.doc["decisions"]
                            if d["decision"] != "union_subset"))
        order = [r["review_group_id"] for r in self.rows]
        got = [d["review_group_id"] for d in self.e.doc["decisions"]]
        self.assertEqual(got, [g for g in order if g in set(got)])

    def test_real_document_layout(self):
        doc = self.e.doc
        self.assertEqual(set(doc), {"schema", "reviewer", "provenance", "binds", "rulings", "decisions", "class_decisions",
                                    "answers", "confirmation", "expansion"})
        self.assertEqual((doc["schema"], doc["reviewer"]), (families.DECISIONS_SCHEMA, "owner"))
        self.assertEqual(doc["binds"], {"review_csv_sha256": CSV_PIN, "registry_pre_review_sha256": PRE_PIN,
                                        "alias_sources_sha256": ALIAS_PIN})
        self.assertEqual(doc["rulings"], {"rdatasets_granularity": "item", "provisional_seals": "keep",
                                          "related_class_policy": "lineage-B related-class families stay unresolved: "
                                                                  "train, never held out",
                                          "split_registry_excluded_seals_nothing": True})
        self.assertEqual(doc["answers"], dict(A4, Q5="a", rights_note="unknown", split_registry_confirmed=True))
        self.assertEqual(doc["confirmation"], {"grammar": "dl-d5-confirm/v1", "review_csv_sha12": "aad99b1a9af6",
                                               "letters": dict(A4, Q5="a"), "question_posted_utc": POSTED_UTC})
        self.assertEqual(doc["expansion"], {"tool": "tools/p2_decisions.py", "tool_sha256": T.tool_sha256(),
                                            "mapping": "dl-d5-map/v2", "shape_sha256": SHAPE_PIN})
        self.assertEqual(doc["provenance"]["message_text"], CONFIRM_TEXT)
        self.assertEqual(self.checks["parsed"], dict(A4, Q5="a", review_csv_sha12="aad99b1a9af6"))
        self.assertEqual(self.e.build_body, self.e.body)                         # a real file is what the build reads
        self.assertEqual(self.e.validation, {"accepted": True, "effective_decisions": 413})

    def test_unconfirmed_document_layout(self):
        for mode, rights in (("draft", "undecided"), ("what_if", "unknown")):
            with self.subTest(mode=mode):
                e = expansion(rights=rights, mode=mode)
                doc = e.doc
                self.assertEqual(doc["schema"], T.UNCONFIRMED_SCHEMA[mode])
                self.assertNotIn(doc["schema"], (families.DECISIONS_SCHEMA, families.SIMULATION_SCHEMA))
                self.assertNotEqual(doc["reviewer"], "owner")
                self.assertNotIn(doc["provenance"]["kind"], families.PROVENANCE_KINDS)
                self.assertTrue(families.provenance_marks_draft(doc["provenance"]))
                self.assertIsNone(doc["rulings"]["split_registry_excluded_seals_nothing"])
                self.assertEqual(doc["rulings_unconfirmed"], ["split_registry_excluded_seals_nothing"])
                self.assertEqual(doc["answers"], dict(A4, Q5=None if mode == "draft" else "a", rights_note=rights,
                                                      split_registry_confirmed=False))
                self.assertNotIn("confirmation", doc)
                self.assertTrue(doc["notice"].startswith(T.PREFIX[mode].rstrip("-")))
                # the rehearsal variant: the decision schema, DRAFT-marked, the same decisions; never written
                reh = json.loads(e.build_body)
                self.assertEqual((reh["schema"], reh["reviewer"]), (families.DECISIONS_SCHEMA, "owner"))
                self.assertEqual(reh["decisions"], doc["decisions"])
                self.assertTrue(reh["rulings"]["split_registry_excluded_seals_nothing"])
                self.assertTrue(families.provenance_marks_draft(reh["provenance"]))
                self.assertEqual(reh["rehearsal_of"]["sha256"], sha(e.body))
                self.assertEqual(e.validation["rehearsal_sha256"], sha(e.build_body))
                self.assertEqual(set(e.validation["unpromotable"]), {"families_staging", "families_real"})
                for reason in e.validation["unpromotable"].values():
                    self.assertIn("decisions schema is not", reason)

    def test_variants(self):
        base = dict(REC_DECISIONS)
        cases = {
            ("b", "a", "a", "a", "unknown"): {"confirm_mapping": 10, "keep_sealed": 37, "unknown": 14, "unresolved": 352},
            ("a", "b", "a", "a", "unknown"): dict(base, keep_sealed=33, unseal=4),
            ("a", "c", "a", "a", "unknown"): {**{k: v for k, v in base.items() if k != "keep_sealed"}, "unseal": 37},
            ("a", "a", "a", "b", "unknown"): dict(base, union=1, unresolved=97),
            ("a", "a", "a", "a", "undecided"): {k: v for k, v in base.items() if k != "unknown"},
        }
        for (q1, q2, q3, q4, rights), want in cases.items():
            with self.subTest(answers=(q1, q2, q3, q4), rights=rights):
                e = expansion({"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}, rights)
                self.assertEqual(e.tally["decision_counts"], want)
                self.assertEqual(e.tally["undecided_rows"], 14 if rights == "undecided" else 0)
                self.assertEqual(e.validation["accepted"], True)
                self.assertEqual(e.doc["rulings"]["provisional_seals"], "keep" if q2 == "a" else "override")
                if q1 == "b":
                    self.assertTrue(all(d["decision"] == "unresolved" for d in e.doc["decisions"]
                                        if self.kind[d["review_group_id"]] in T.LINK_KINDS))
                if q2 == "b":
                    self.assertTrue(all(d["decision"] == self.rec[d["review_group_id"]] for d in e.doc["decisions"]
                                        if self.kind[d["review_group_id"]] == "sealed_adjacent"))
                if rights == "undecided":
                    self.assertFalse(any(self.kind[d["review_group_id"]] == "rights_note" for d in e.doc["decisions"]))
                if q4 == "b":
                    dec = {d["review_group_id"]: d["decision"] for d in e.doc["decisions"]}
                    self.assertEqual(dec[BREAKER_ID], "union")

    def test_bytes_are_canonical_and_deterministic(self):
        again, _ = real_expansion()
        self.assertEqual(again.body, self.e.body)
        self.assertEqual(self.e.body, common.canonical(json.loads(self.e.body)))
        for mode, rights in (("draft", "undecided"), ("what_if", "unknown")):
            a, b = expansion(rights=rights, mode=mode), expansion(rights=rights, mode=mode)
            self.assertEqual((a.body, a.build_body), (b.body, b.build_body))
        self.assertNotRegex(self.e.body.decode(), r'"(utc|utc_us|time|timestamp|created|started_utc_us)"')


class TestImportHygiene(unittest.TestCase):
    STDLIB = {"__future__", "argparse", "contextlib", "csv", "io", "json", "math", "os", "re", "shutil", "sys", "time",
              "traceback", "collections", "dataclasses", "pathlib", "types"}

    def test_module_level_and_lazy_imports(self):
        tree = ast.parse((SRC / "tools" / "p2_decisions.py").read_text())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                top |= {f"{node.module}.{a.name}" if node.module == "dltrain" else node.module for a in node.names}
        self.assertEqual(top - self.STDLIB, {"dltrain.common", "dltrain.families", "dltrain.family_pin"})
        lazy = {}
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        lazy.setdefault(f"{node.module}.{a.name}", set()).add(fn.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        lazy.setdefault(a.name, set()).add(fn.name)
        self.assertEqual(lazy, {"dltrain.holdout": {"cmd_check", "cmd_dryrun", "holdout_refuses", "dryrun_code_gate"},
                                "tests_p3.fixtures_p3": {"cmd_dryrun"}, "unittest.mock": {"cmd_dryrun"}})
        names = set(top) | set(lazy)
        self.assertFalse({n for n in names if n.split(".")[0] in ("boto3", "botocore", "torch") or n == "dltrain.s3io"},
                         "the tool never imports s3io, boto3 or torch itself")
        src = (SRC / "tools" / "p2_decisions.py").read_text()
        for banned in ("jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry", "boto3", "torch"):
            self.assertNotRegex(src, rf"(?m)^\s*(import|from)\s+{banned}\b", banned)
        self.assertNotRegex(src, r"(?m)^\s*(from\s+dltrain\s+import\s+.*\bs3io\b|import\s+dltrain\.s3io\b)")

    def test_expand_and_shape_load_no_s3_modules(self):
        code = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('p2_decisions', sys.argv[1])\n"
            "t = importlib.util.module_from_spec(spec); sys.modules['p2_decisions'] = t; spec.loader.exec_module(t)\n"
            "c = t.read_pinned(sys.argv[2], sys.argv[3], 'c'); p = t.read_pinned(sys.argv[4], sys.argv[5], 'p')\n"
            "a = t.read_pinned(sys.argv[6], sys.argv[7], 'a')\n"
            "import json; code = json.loads(p.raw)['code']; t.current_code_shas = lambda: dict(code)\n"
            "t.compute_shape(t.parse_rows(c.raw), c.sha256)\n"
            "e = t.build_expansion(mode='draft', csv_b=c, pre_b=p, alias_b=a, answers=dict(t.RELAY_ANSWERS),"
            " rights='undecided', provenance=dict(t.DRAFT_PROVENANCE), want_shape=t.BUILTIN_SHAPES['builtin:aad99b1a'])\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('boto3', 'botocore', 'jope', 'gp_worker',"
            " 'local_gateway', 'reuse_cache', 'intake_registry', 'torch', 'tests_p3') or m in ('dltrain.s3io',"
            " 'dltrain.holdout'))\n"
            "print(len(e.body), bad)\n")
        r = subprocess.run([sys.executable, "-c", code, str(SRC / "tools" / "p2_decisions.py"), str(CSV), CSV_PIN,
                            str(PRE), PRE_PIN, str(ALIAS), ALIAS_PIN], capture_output=True, text=True, timeout=300,
                           cwd=str(SRC), env=dict(os.environ))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(r.stdout.strip().endswith("[]"), r.stdout)


class TestOutputHygiene(D5Case):
    """Nothing the tool prints or writes carries a trigger_value, members or evidence cell of review.csv."""

    @staticmethod
    def sensitive_cells():
        out = []
        for i, r in enumerate(real_rows()):
            for col in ("trigger_value", "members", "evidence"):
                for part in {r[col], *re.split(r" \| |; |\n", r[col])}:
                    part = part.strip()
                    if len(part) >= 12 and not re.fullmatch(r"[a-z_=0-9]+", part):
                        out.append((col, i, part))
        return out

    def test_cli_output_and_files_are_clean(self):
        texts = {}
        for argv in (draft_args(), whatif_args(), ["shape", "--review-csv", CSV, "--review-csv-sha256", CSV_PIN],
                     draft_args(answers="Q1=a,Q2=a,Q3=a,Q4=b")):          # the last one is a refusal line
            code, out, err = self.cli(argv)
            texts[f"run{len(texts)}"] = out + err
        for p in self.tmp.rglob("*"):
            if p.is_file():
                texts[str(p.relative_to(self.tmp))] = p.read_text()
        self.assertGreaterEqual(len(texts), 8)
        cells = self.sensitive_cells()
        self.assertGreater(len(cells), 100)
        hits = sorted({(name, col, i) for col, i, v in cells for name, text in texts.items() if v in text})
        self.assertEqual(hits, [], "review.csv cell text leaked (names withheld)")


if __name__ == "__main__":
    unittest.main()
