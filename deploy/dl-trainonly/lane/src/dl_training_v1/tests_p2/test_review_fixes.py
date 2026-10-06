"""Offline proofs for the P2 review fixes (findings CL-1..CL-6, SC-1..SC-9) and the network-free local re-pin.

Every test builds a synthetic World (tests_p2.test_families) under TMPDIR; nothing touches SQLite, S3 or the network.
"""
import contextlib
import io
import json
import os
import re
import stat
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import common
from dltrain import families as fam
from dltrain import family_pin as fp
from dltrain import family_screen as fs
from tests_p2.test_families import (CANARIES, ROWS, Case, H, csv_rows, decisions_doc, fam_of, put, sha, standard_world)

# tools/p2_decisions.py (DL-D5 expansion) parses union_subset sets from recommendation_basis with this rule (R6)
DL_D5_UNION_SUBSET_BASIS = re.compile(r"agent-generated \(deterministic rule, not a decision\): pairwise nds/name/legacy "
                                      r"evidence; sets=(\[.*\])")
TOOL_DRAFT_PROVENANCE = {   # tools/p2_decisions.py DRAFT_PROVENANCE (shape)
    "kind": "owner_verbatim_message", "session": "DRAFT-UNCONFIRMED-RELAY-b785a123",
    "message_utc": "2026-10-05T10:07Z relay arrival (not an owner timestamp)", "message_sha256": H("relay"),
    "draft": True, "confirmed_by_owner": False, "unconfirmed": ["Q1-Q4"], "note": "relayed paste"}
TOOL_RELATED_CLASS_TEXT = "lineage-B related-class families stay unresolved: train, never held out"


def rows_of(res, kind):
    return [r for r in res.pre.rows if r["kind"] == kind]


def chain_tables(n: int, seed: int = 900) -> list:
    """n tables T_k = [C_k, C_k+1, C_k+2, target_k]: consecutive tables share two whole columns row-aligned (a strong
    value link), tables two apart share one column (no link), and no column reaches more than 2 other tables."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 10 ** 6, size=ROWS) / 10 ** 6 for _ in range(n + 2)]
    return [np.column_stack([cols[k], cols[k + 1], cols[k + 2], rng.integers(0, 10 ** 6, size=ROWS) / 10 ** 6])
            for k in range(n)]


class FixCase(Case):
    def built(self, w):
        w.write()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        return res

    def final(self, w, res, doc, name=None, **kw):
        p = w.out / (name or f"dec-{sha(common.canonical(doc))[:8]}.json")
        put(p, common.canonical(doc))
        return w.build(p, **kw)

    def repin(self, w, out_name="g2", withhold=None, rec=None, attempt=""):
        rec = rec or (self.tmp / "receipts")
        argv = ["pin", "--from-local", str(w.out), "--from-receipt", str(w.pin_receipt), "--out", str(w.out / out_name),
                "--receipt", str(rec / f"repin-{out_name}{attempt}.json")]
        if withhold is not None:
            wp = self.tmp / f"withhold-{out_name}.json"
            body = put(wp, common.canonical({"schema": fp.WITHHOLD_SCHEMA, "identities": withhold, "reason": "test"}))
            argv += ["--withhold", str(wp), "--withhold-sha256", sha(body)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fam._main(argv, roots=(w.out, rec), guard=fp.NoGuard(), expect=None)
        return code, out.getvalue(), err.getvalue()


# ======================================================================================================== CL-1
class TestCL1Recommendations(FixCase):
    def test_own_trigger_rows_are_never_independent(self):
        w = standard_world(self.tmp / "w")
        w.frozen("LEG1", legacy="only-own-legacy")
        w.frozen("LEG2", legacy="only-own-legacy")
        w.frozen("TOKA", prov={"platform": "hostx", "dataset_id": "ta", "url": "https://a-host.org/files/zebrafish_growth.csv"})
        w.frozen("TOKB", prov={"platform": "hosty", "dataset_id": "tb", "url": "https://b-host.org/data/zebrafish_growth.csv"})
        res = self.built(w)
        leg = next(r for r in rows_of(res, "legacy_source_family") if r["trigger_value"] == "only-own-legacy")
        pair = sorted(res.pre.fid_of[w.item(n)] for n in ("LEG1", "LEG2"))
        self.assertEqual(leg["recommendation"], "union_subset")
        m = DL_D5_UNION_SUBSET_BASIS.fullmatch(leg["recommendation_basis"])
        self.assertIsNotNone(m)                                  # the DL-D5 expansion still parses the sets
        self.assertEqual(json.loads(m.group(1)), [pair])
        self.assertIn("legacy_source_family", {x["kind"] for x in leg["evidence"]["pairwise"]})
        name = next(r for r in rows_of(res, "name_candidate") if r["trigger_value"] == "zebrafish_growth")
        self.assertEqual(name["recommendation"], "union_subset")
        for r in res.pre.rows:
            if r["kind"] in fam.OWN_EVIDENCE_KINDS:
                self.assertNotEqual(r["recommendation"], "independent", r["kind"])
        classic = rows_of(res, "classic_candidate")
        self.assertTrue(classic)
        for r in classic:                                        # its own curated match only: owner judgment
            self.assertEqual(r["recommendation"], "unresolved")
            self.assertNotIn("trigger-only", r["recommendation_basis"])

    def test_sealed_adjacent_recommendations_are_consistent(self):
        w = standard_world(self.tmp / "w")
        res = self.built(w)
        sa = {r["_subject"][0]: r for r in rows_of(res, "sealed_adjacent")}
        hard = res.pre.fid_of[w.item("SEALED")]
        mate, audit, out = (res.pre.fid_of[w.item(n)] for n in ("MATE", "AUDITMATE", "OUTMATE"))
        legacy_rows = [r for r in rows_of(res, "legacy_source_family") if {hard, mate} <= set(r["family_ids"])]
        self.assertTrue(legacy_rows and legacy_rows[0]["recommendation"] == "union_subset")
        self.assertEqual(sa[mate]["recommendation"], "keep_sealed")       # another row unites it with a hard family
        self.assertIn("uniting it with a hard-sealed family", sa[mate]["recommendation_basis"])
        self.assertEqual(sa[audit]["recommendation"], "keep_sealed")      # spanning lineage-B audit family
        self.assertEqual(sa[out]["recommendation"], "unseal")             # legacy adjacency to a dataset outside the epoch


# ======================================================================================================== CL-2
def owner_message_provenance(text: str, **extra) -> dict:
    return {"kind": "owner_verbatim_message", "session": "screen 1.owner, typed by the owner",
            "message_utc": "2026-10-05T12:00:00Z", "message_sha256": sha(text.encode()), "message_text": text, **extra}


class TestCL2Provenance(FixCase):
    def setUp(self):
        super().setUp()
        self.w = standard_world(self.tmp / "w")
        self.res = self.built(self.w)
        self.rec = self.tmp / "receipts"
        self.rec.mkdir(parents=True, exist_ok=True)

    def doc(self, prov, **kw):
        d = decisions_doc(self.res, **kw)
        d["provenance"] = prov
        return d

    def test_draft_and_simulation_provenance(self):
        draft = self.doc(dict(TOOL_DRAFT_PROVENANCE))
        fin = self.final(self.w, self.res, draft)                      # staging (not a real generation): allowed
        self.assertEqual(fin.final.phase, "final")
        self.assertTrue(fin.final.deltas["provenance"]["draft"])
        with self.assertRaisesRegex(fam.FamilyStop, "DRAFT or SIMULATION"):
            self.final(self.w, self.res, draft, real=True, receipts_dir=self.rec)
        sim = self.doc({"kind": "owner_session_selections", "session": "SIMULATION-NOT-A-DECISION",
                        "message_utc": "SIMULATION", "message_sha256": "0" * 64})
        with self.assertRaisesRegex(fam.FamilyStop, "degenerate"):         # the old simulation files can never validate
            self.final(self.w, self.res, sim)
        simdoc = dict(sim, schema=fam.SIMULATION_SCHEMA)
        with self.assertRaisesRegex(fam.FamilyStop, "never a decision file"):
            self.final(self.w, self.res, simdoc)
        p = self.w.out / "sim.json"
        put(p, common.canonical(simdoc))
        res = self.w.build(p, simulation=True)
        self.assertIn(fam.OUT_SIMULATION, res.outputs)
        self.assertNotIn(fam.OUT_FINAL, res.outputs)
        self.assertEqual(json.loads(res.outputs[fam.OUT_SIMULATION])["phase"], "simulation")
        with self.assertRaisesRegex(fam.FamilyStop, "simulation"):
            self.w.build(p, simulation=True, real=True)
        with self.assertRaisesRegex(fam.FamilyStop, "schema"):            # a decision file never runs as a simulation
            self.w.build(self._write_doc(self.doc(owner_message_provenance("x"))), simulation=True)

    def _write_doc(self, doc):
        p = self.w.out / f"d-{sha(common.canonical(doc))[:8]}.json"
        put(p, common.canonical(doc))
        return p

    def test_api_never_downgrades_a_generation_under_families_to_staging(self):
        """build()/verify() called directly with real=False still treat an output directory under RUN/families as a
        real generation: a DRAFT decision file is refused and no registry.json is written."""
        p = self._write_doc(self.doc(dict(TOOL_DRAFT_PROVENANCE)))
        with mock.patch.object(fam, "FAMILIES_DIR", self.w.out):
            self.assertTrue(fam.is_real_generation(self.w.out))
            with self.assertRaisesRegex(fam.FamilyStop, "DRAFT or SIMULATION"):
                fam.build(self.w.config, self.w.out, p, guard=fp.NoGuard(), expect=None, real=False,
                          receipts_dir=self.rec)
            self.assertFalse((self.w.out / fam.OUT_FINAL).exists())
            with self.assertRaisesRegex(fam.FamilyStop, "DRAFT or SIMULATION"):
                fam.verify(self.w.config, self.w.out, p, guard=fp.NoGuard(), expect=None, real=False,
                           receipts_dir=self.rec)
        # outside RUN/families (staging) the same file still builds in memory
        self.assertEqual(self.w.build(p).final.phase, "final")

    def test_draft_markers_anywhere_are_refused_for_a_real_generation(self):
        text = "Q1=a Q2=a Q3=a Q4=a split-registry confirmed"
        put(self.rec / fam.OWNER_MESSAGE_NAME.format(sha16=sha(text.encode())[:16]), text.encode())
        ok = owner_message_provenance(text)
        self.assertEqual(self.final(self.w, self.res, self.doc(ok), real=True, receipts_dir=self.rec).final.phase, "final")
        for change in ({"session": "screen 1 (draft relay)"}, {"session": "owner-Simulation-3"}, {"unconfirmed": ["Q2"]},
                       {"note": "draft: awaiting the owner"}, {"confirmed_by_owner": False}):
            with self.subTest(change=sorted(change)):
                with self.assertRaisesRegex(fam.FamilyStop, "DRAFT or SIMULATION"):
                    self.final(self.w, self.res, self.doc(dict(ok, **change)), real=True, receipts_dir=self.rec)
        fine = dict(ok, session="screen 1.owner draftsman-host")      # a word that only contains the marker
        self.assertFalse(fam.provenance_marks_draft(fine))
        self.assertEqual(self.final(self.w, self.res, self.doc(fine), real=True, receipts_dir=self.rec).final.phase, "final")
        self.assertTrue(fam.provenance_marks_draft({"kind": "owner_directive", "named_by_owner": {"session": "DRAFT-x"}}))

    def test_message_format_rules(self):
        text = "Q1=a Q2=a Q3=a Q4=a Q5=a split-registry confirmed; rights unknown"
        bad_utc = self.doc(owner_message_provenance(text, message_utc="yesterday"))
        with self.assertRaisesRegex(fam.FamilyStop, "ISO-8601"):
            self.final(self.w, self.res, bad_utc)
        bad_text = self.doc(dict(owner_message_provenance(text), message_text=text + " (edited)"))
        with self.assertRaisesRegex(fam.FamilyStop, "message_text does not hash"):
            self.final(self.w, self.res, bad_text)
        empty = self.doc(dict(owner_message_provenance(text), message_sha256=fam.EMPTY_SHA256, message_text=None))
        with self.assertRaisesRegex(fam.FamilyStop, "degenerate"):
            self.final(self.w, self.res, empty)
        self.assertEqual(self.final(self.w, self.res, self.doc(owner_message_provenance(text))).final.phase, "final")

    def test_real_generation_needs_a_pinned_owner_record(self):
        text = "Q1=a Q2=a Q3=a Q4=a Q5=a split-registry confirmed"
        doc = self.doc(owner_message_provenance(text))
        with self.assertRaisesRegex(fam.FamilyStop, "pinned owner record"):
            self.final(self.w, self.res, doc, real=True, receipts_dir=self.rec)
        stored = put(self.rec / fam.OWNER_MESSAGE_NAME.format(sha16=sha(text.encode())[:16]), text.encode())
        fin = self.final(self.w, self.res, doc, real=True, receipts_dir=self.rec)
        self.assertEqual(fin.final.deltas["provenance"]["record"]["form"], "stored_owner_message")
        self.assertEqual(fin.final.deltas["provenance"]["record"]["sha256"], sha(stored))
        # an explicit record: an owner decisions receipt naming the message sha256 and timestamp
        text2 = "Q1=a Q2=a Q3=a Q4=a split-registry confirmed (second message)"
        prov = owner_message_provenance(text2)
        receipt = put(self.tmp / "owner-decisions-x.json", common.canonical(
            {"provenance": {"owner_message_utc": prov["message_utc"], "message_sha256": prov["message_sha256"]}}))
        ok = self.doc(dict(prov, record={"path": str(self.tmp / "owner-decisions-x.json"), "sha256": sha(receipt)}))
        self.assertEqual(self.final(self.w, self.res, ok, real=True, receipts_dir=self.rec).final.phase, "final")
        other = put(self.tmp / "other.json", b'{"provenance": {}}')
        bad = self.doc(dict(prov, record={"path": str(self.tmp / "other.json"), "sha256": sha(other)}))
        with self.assertRaisesRegex(fam.FamilyStop, "names neither"):
            self.final(self.w, self.res, bad, real=True, receipts_dir=self.rec)
        moved = self.doc(dict(prov, record={"path": str(self.tmp / "owner-decisions-x.json"), "sha256": "1" * 64}))
        with self.assertRaisesRegex(fam.FamilyStop, "degenerate|real sha256"):
            self.final(self.w, self.res, moved, real=True, receipts_dir=self.rec)

    def test_owner_directive_must_be_owner_directives_or_named_by_the_owner(self):
        d1 = put(self.tmp / "OWNER_DIRECTIVES.md", b"# directives\nDL-D5: Q1=a Q2=a Q3=a Q4=a\nsplit-registry ok\n")
        prov = {"kind": "owner_directive", "file": str(self.tmp / "OWNER_DIRECTIVES.md"), "version": "v1", "lines": "2-3",
                "file_sha256": sha(d1)}
        self.assertEqual(self.final(self.w, self.res, self.doc(prov), real=True, receipts_dir=self.rec).final.phase, "final")
        d2 = put(self.tmp / "dl-notes.md", b"DL-D5 answers\n")
        prov2 = dict(prov, file=str(self.tmp / "dl-notes.md"), file_sha256=sha(d2), lines="1")
        with self.assertRaisesRegex(fam.FamilyStop, "named_by_owner"):
            self.final(self.w, self.res, self.doc(prov2), real=True, receipts_dir=self.rec)
        text = "use dl-notes.md for DL-D5"
        put(self.rec / fam.OWNER_MESSAGE_NAME.format(sha16=sha(text.encode())[:16]), text.encode())
        named = {"kind": "owner_verbatim_message", "session": "screen 1", "message_utc": "2026-10-05T12:30Z",
                 "message_sha256": sha(text.encode())}
        fin = self.final(self.w, self.res, self.doc(dict(prov2, named_by_owner=named)), real=True, receipts_dir=self.rec)
        self.assertEqual(fin.final.deltas["provenance"]["record"]["form"], "owner_directive")
        with self.assertRaisesRegex(fam.FamilyStop, "lines"):
            self.final(self.w, self.res, self.doc(dict(prov, lines="2-9")), real=True, receipts_dir=self.rec)

    def test_dl_d5_draft_document_still_validates_in_memory(self):
        """keep validate_decisions compatible with the DL-D5 expansion: the tool's DRAFT document (TMP staging) and its
        rulings wording validate when not a real generation; the related-class wording maps to 'unresolved'."""
        doc = self.doc(dict(TOOL_DRAFT_PROVENANCE), rulings={"rdatasets_granularity": "item", "provisional_seals": "keep",
                                                             "related_class_policy": TOOL_RELATED_CLASS_TEXT,
                                                             "split_registry_excluded_seals_nothing": True})
        doc["answers"] = {"Q1": "a", "Q2": "a", "Q3": "a", "Q4": "a", "rights_note": "unknown"}
        doc["expansion"] = {"tool": "tools/p2_decisions.py", "mapping": "dl-d5-map/v1"}
        vrows = [{"review_group_id": r["review_group_id"], "kind": r["kind"], "family_ids": r["family_ids"],
                  "recommendation": r["recommendation"]} for r in self.res.pre.rows]
        v = fam.validate_decisions(common.canonical(doc), vrows, review_csv_sha256=sha(self.res.outputs[fam.OUT_CSV]),
                                   pre_sha256=sha(self.res.outputs[fam.OUT_PRE]), alias_sha256=self.res.base.inp.sha256,
                                   pre=self.res.pre)
        self.assertEqual(v["rulings"]["applied"]["related_class_policy"], "unresolved")
        self.assertTrue(v["provenance"]["draft"])
        bad = dict(doc, rulings=dict(doc["rulings"], related_class_policy="whatever the agent wrote"))
        with self.assertRaisesRegex(fam.FamilyStop, "related_class_policy"):
            fam.validate_decisions(common.canonical(bad), vrows, review_csv_sha256=sha(self.res.outputs[fam.OUT_CSV]),
                                   pre_sha256=sha(self.res.outputs[fam.OUT_PRE]), alias_sha256=self.res.base.inp.sha256,
                                   pre=self.res.pre)

    def test_cli_real_generation_decision_file_rules(self):
        w = self.w
        roots = (w.out, self.rec)
        text = "Q1=a Q2=a Q3=a Q4=a split-registry confirmed"
        put(self.rec / fam.OWNER_MESSAGE_NAME.format(sha16=sha(text.encode())[:16]), text.encode())

        def run(argv):
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(fam, "FAMILIES_DIR", w.out), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                code = fam._main(argv, roots=roots, guard=fp.NoGuard(), expect=None)
            return code, out.getvalue() + err.getvalue()

        base = ["build", "--config", str(w.config), "--out", str(w.out)]
        dec = self.doc(owner_message_provenance(text))
        body = common.canonical(dec)
        put(w.out / "dec.json", body)
        code, msg = run(base + ["--decisions", str(w.out / "dec.json"), "--decisions-sha256", sha(body)])
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("review-decisions.json", msg)
        p = w.out / fam.DECISIONS_NAME
        put(p, body)
        code, msg = run(base + ["--decisions", str(p), "--decisions-sha256", sha(body)])
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("read-only", msg)
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        code, msg = run(base + ["--decisions", str(p)])
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("mandatory", msg)
        code, msg = run(["build", "--config", str(self.tmp / "w" / "elsewhere.json"), "--out", str(w.out)])
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("own alias-sources.json", msg)
        code, msg = run(base + ["--decisions", str(p), "--decisions-sha256", sha(body)])
        self.assertEqual(code, 0, msg[-800:])
        self.assertTrue((w.out / fam.OUT_FINAL).is_file())
        # the same CLI with a DRAFT decision file STOPs in a real generation
        w2 = standard_world(self.tmp / "w2")
        res2 = self.built(w2)
        d2 = decisions_doc(res2)
        d2["provenance"] = dict(TOOL_DRAFT_PROVENANCE)
        b2 = put(w2.out / fam.DECISIONS_NAME, common.canonical(d2))
        os.chmod(w2.out / fam.DECISIONS_NAME, 0o444)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(fam, "FAMILIES_DIR", w2.out), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fam._main(["build", "--config", str(w2.config), "--out", str(w2.out), "--decisions",
                              str(w2.out / fam.DECISIONS_NAME), "--decisions-sha256", sha(b2)],
                             roots=(w2.out, self.rec), guard=fp.NoGuard(), expect=None)
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("DRAFT or SIMULATION", err.getvalue())


# ======================================================================================================== CL-3
class TestCL3Rulings(FixCase):
    def test_package_granularity_is_applied_or_refused(self):
        w = standard_world(self.tmp / "w")
        res = self.built(w)
        pkg = rows_of(res, "rdatasets_pkg")[0]
        rul = {"rdatasets_granularity": "package", "provisional_seals": "keep", "related_class_policy": "unresolved",
               "split_registry_excluded_seals_nothing": True}
        fin = self.final(w, res, decisions_doc(res, rulings=rul))
        self.assertEqual(fam_of(fin, w.item("PKG1"))["family_id"], fam_of(fin, w.item("PKG2"))["family_id"])
        self.assertEqual(fin.final.deltas["rulings"]["applied"]["package_unions"], 1)
        self.assertEqual(fin.final.effective[pkg["review_group_id"]], "union")
        with self.assertRaisesRegex(fam.FamilyStop, "'package' contradicts"):
            self.final(w, res, decisions_doc(res, rulings=rul, group={pkg["review_group_id"]: {"decision": "independent"}}))
        item = dict(rul, rdatasets_granularity="item")
        fin2 = self.final(w, res, decisions_doc(res, rulings=item, group={pkg["review_group_id"]: {"decision": "independent"}}))
        self.assertNotEqual(fam_of(fin2, w.item("PKG1"))["family_id"], fam_of(fin2, w.item("PKG2"))["family_id"])

    def test_related_class_policy(self):
        w = standard_world(self.tmp / "w")
        w.frozen("RELCLS", legacy="uniformity-trial")
        res = self.built(w)
        rc = rows_of(res, "lineage_b_related_class")
        self.assertEqual(len(rc), 1)
        group = {rc[0]["review_group_id"]: {"decision": "independent"}}
        with self.assertRaisesRegex(fam.FamilyStop, "related_class_policy 'unresolved' contradicts"):
            self.final(w, res, decisions_doc(res, group=group))
        rul = {"rdatasets_granularity": "item", "provisional_seals": "keep", "related_class_policy": "per_row",
               "split_registry_excluded_seals_nothing": True}
        fin = self.final(w, res, decisions_doc(res, group=group, rulings=rul))
        self.assertEqual(fin.final.effective[rc[0]["review_group_id"]], "independent")
        alias = dict(rul, related_class_policy=TOOL_RELATED_CLASS_TEXT)
        with self.assertRaisesRegex(fam.FamilyStop, "related_class_policy 'unresolved'"):
            self.final(w, res, decisions_doc(res, group=group, rulings=alias))


# ======================================================================================================== CL-4 / SC-5
def _idents(n):
    return {f"i{k:02d}": fam.Ident(identity=f"i{k:02d}", kind="registry_item", role="frozen", dataset_hash=f"{k:016x}")
            for k in range(n)}


class TestCL4Breaker(FixCase):
    def test_supernodes_count_merged_datasets_and_hold_only_cross_edges(self):
        idents = _idents(14)
        E = fam._edge
        hub = [E("i00", f"i{k:02d}", "normalized_url", "https://hub.example.org/x.csv") for k in range(1, 9)]
        intra = [E(f"i{a:02d}", f"i{b:02d}", "nds:name_jaccard", f"n{a}-{b}") for a in range(9) for b in range(a + 1, 9)]
        cross = [E("i03", "i09", "nds:value_strong", "s1"), E("i05", "i10", "nds:value_strong", "s2")]
        held, trips = fam.breaker(hub + intra + cross, idents, 10)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0]["datasets"], 11)                   # the 9-dataset hub plus two strong neighbours
        self.assertEqual(held, set(cross))                           # exact and redundant intra edges are never held
        self.assertEqual(trips[0]["supernode_datasets"], [9, 1, 1])
        self.assertEqual(fam.breaker(cross, idents, 10), (set(), []))   # the old subgraph-only view saw 4 datasets
        big = ([E("i00", f"i{k:02d}", "normalized_url", "u") for k in range(1, 13)]
               + [E(f"i{a:02d}", f"i{a + 1:02d}", "nds:name_jaccard", f"m{a}") for a in range(12)])
        self.assertEqual(fam.breaker(big, idents, 10), (set(), []))     # redundant name edges inside one key component
        held2, trips2 = fam.breaker(hub + cross, idents, 10, never_hold=frozenset({"i09"}))
        self.assertEqual(held2, {cross[1]})                          # a strong edge touching a sealed super-node applies
        self.assertEqual(trips2[0]["kept_sealed_edges"], [cross[0]])

    def test_breaker_row_evidence_and_union_subset_leak_stop(self):
        w = standard_world(self.tmp / "w")
        for k, m in enumerate(chain_tables(11)):
            w.frozen(f"CH{k:02d}", data=m)
        res = self.built(w)
        self.assertEqual(res.summary["verdict"], "STOP")
        br = rows_of(res, "circuit_breaker")
        self.assertEqual(len(br), 1)
        ev = br[0]["evidence"]["held_cross_family_edges"]
        self.assertEqual(len(ev), 10)
        self.assertTrue(all(e["kind"] == "nds:value_strong" and e["containment"] == 1.0 and e["fa"] != e["fb"] for e in ev))
        fids = [res.pre.fid_of[w.item(f"CH{k:02d}")] for k in range(11)]
        self.assertEqual(len(set(fids)), 11)
        g = br[0]["review_group_id"]
        sub = decisions_doc(res, group={g: {"decision": "union_subset", "sets": [sorted(fids[:2])]}})
        with self.assertRaisesRegex(fam.FamilyStop, "held content/value_strong") as cm:
            self.final(w, res, sub)
        self.assertTrue(cm.exception.detail["held_value_edges"])
        fin = self.final(w, res, decisions_doc(res, group={g: {"decision": "union"}}))
        self.assertEqual(len({fam_of(fin, w.item(f"CH{k:02d}"))["family_id"] for k in range(11)}), 1)
        fin2 = self.final(w, res, decisions_doc(res, group={g: {"decision": "unresolved"}}))
        self.assertEqual({fam_of(fin2, w.item(f"CH{k:02d}"))["independence_qualified"] for k in range(11)}, {"unresolved"})

    def test_held_edges_touching_a_sealed_family_seal_and_withhold(self):
        w = standard_world(self.tmp / "w", breaker=True)
        w.lb_extra = {"extra-val": {"family": "extra-val", "partition": "meta_validation", "source_keys": [],
                                    "datasets": [{"dataset": w.h("GH00"), "source_key": "/abs/gh00.csv", "evidence": []}]}}
        res = self.built(w)
        sealed = fam_of(res, w.item("GH00"))
        self.assertTrue(sealed["sealed"])
        self.assertEqual(sealed["sealed_basis"], "lineage_b_member")
        others = [fam_of(res, w.item(f"GH{k:02d}")) for k in range(1, 12)]
        held_adj = [F for F in others if any(k == "held_edge" for k, _, _ in F["_adjacency"])]
        self.assertTrue(held_adj)                                    # SC-5: a held edge to a hard-sealed family seals
        self.assertTrue(all(F["sealed"] and F["sealed_basis"] == "provisional_pending_review" for F in held_adj))
        for F in others:                                             # SC-3: the breaker row holds a sealed family
            self.assertTrue(F["sealed"] or F["withheld"])
        rows = csv_rows(res)
        br = [r for r in rows if r["kind"] == "circuit_breaker"][0]
        self.assertTrue(br["trigger_value"].startswith("sha256:"))
        self.assertNotIn("GH01 display", br["members"])
        sa = [r for r in rows_of(res, "sealed_adjacent") if r["_subject"][0] in {F["family_id"] for F in held_adj}]
        self.assertEqual(len(sa), len(held_adj))
        self.assertTrue(all(r["recommendation"] == "keep_sealed" for r in sa))


# ======================================================================================================== CL-5 / SC-4 / SC-8 / local re-pin
def leaky_world(root):
    w = standard_world(root)
    w.ds["MATE"]["meta_extra"] = {"source_url": "https://mate.example.org/CANARY-MATE-SOURCE.csv",
                                  "source_dataset_id": "CANARY-MATE-SRCID", "source_family": CANARIES["legacy_sealed"]}
    w.ds["PLAIN2"]["meta_extra"] = {"source_url": "https://plain2-meta.example.org/plain2.csv", "source_family": "plain-fam"}
    w.incoming("INCMATE", display="CANARY-INCMATE-DISPLAY", legacy=CANARIES["legacy_sealed"], folds=(0, 1))
    return w


class TestLocalRepin(FixCase):
    def test_repin_redacts_inputs_reproduces_the_build_and_verify_scans_inputs(self):
        w = leaky_world(self.tmp / "w")
        res1 = self.built(w)
        self.assertTrue(fam_of(res1, w.item("INCMATE"))["sealed"])
        rep1 = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None)
        self.assertIn("canary_scan_clean", rep1["failed"])           # CL-5: generation-1 inputs hold sealed strings
        self.assertTrue(any(k.startswith("inputs/meta/") for k in rep1["canary"]["hits"]))
        self.assertIn("inputs/registry-snapshot.json", rep1["canary"]["hits"])
        code, out, err = self.repin(w)
        self.assertEqual(code, 0, err[-1500:])
        g2 = w.out / "g2"
        cfg = json.loads((g2 / "alias-sources.json").read_bytes())
        self.assertEqual(cfg["repin"]["s3_requests"], 0)
        hidden = set(cfg["repin"]["redaction"]["identities"])
        self.assertIn(w.item("MATE"), hidden)
        self.assertIn(w.item("INCMATE"), hidden)
        self.assertNotIn(w.item("PLAIN2"), hidden)
        metas = {e["datasets"][0]: json.loads((g2 / e["local_path"]).read_bytes()) for e in cfg["inputs"] if e["role"] == "meta"}
        mate, plain2 = metas[w.h("MATE")], metas[w.h("PLAIN2")]
        for k in ("source_url", "source_dataset_id", "source_family"):
            self.assertEqual(set(mate[k]), {"value_sha256"})          # every META_SENSITIVE field of a sealed member
        self.assertEqual(set(plain2["source_url"]), {"value_sha256"})  # lossless fields of every meta
        self.assertEqual(plain2["source_family"], "plain-fam")        # derive() reads it: kept for non-sealed members
        snap = json.loads((g2 / "inputs" / "registry-snapshot.json").read_bytes())
        for it in snap["items"]:
            row = (it.get("payload") or {}).get("catalog_row")
            if it["epoch_role"] in ("frozen", "incoming") and isinstance(row, dict):
                self.assertFalse(set(row) & set(fp.ROW_SENSITIVE))
            if it["epoch_role"] == "sibling":
                self.assertEqual(row["display_name"], "PLAIN1 display")   # a non-sealed sibling row stays
        blob = "\n".join(p.read_text() for p in (g2 / "inputs").rglob("*") if p.is_file())
        for c in ("CANARY-MATE-SOURCE", "CANARY-MATE-SRCID", "CANARY-INCMATE-DISPLAY"):
            self.assertNotIn(c, blob)
        res2 = fam.run_build(g2 / "alias-sources.json", None, guard=fp.NoGuard(), expect=None)
        self.assertEqual(res2.outputs[fam.OUT_CSV], res1.outputs[fam.OUT_CSV])
        fam.write_outputs(g2, res2.outputs)
        rep2 = fam.verify(g2 / "alias-sources.json", g2, guard=fp.NoGuard(), expect=None, receipts_dir=self.tmp / "receipts")
        self.assertEqual(rep2["verdict"], "PASS", rep2["failed"])
        self.assertGreater(rep2["canary"]["scope"]["inputs"], 20)
        self.assertGreaterEqual(rep2["canary"]["scope"]["receipts"], 1)
        receipt = json.loads((self.tmp / "receipts" / "repin-g2.json").read_bytes())
        self.assertEqual((receipt["schema"], receipt["verdict"], receipt["s3_requests"]), (fp.REPIN_RECEIPT_SCHEMA, "PASS", 0))
        self.assertNotIn("CANARY", json.dumps(receipt))
        code, _, err = self.repin(w, attempt="-2")                    # create-only: the generation exists
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("already pinned", err)

    def test_repin_reverifies_every_pinned_file_and_the_seal_floor(self):
        w = leaky_world(self.tmp / "w")
        self.built(w)
        code, _, err = self.repin(w, "g2")
        self.assertEqual(code, 0, err[-800:])
        g2 = w.out / "g2"
        cfg = json.loads((g2 / "alias-sources.json").read_bytes())
        # a config whose seal floor or redaction list misses a sealed member STOPs the build
        floor = cfg["repin"]["seal_floor"]["identities"]
        for tamper, msg in (("redaction", "did not redact"), ("seal_floor", "seal floor")):
            c2 = json.loads(json.dumps(cfg))
            if tamper == "redaction":
                c2["repin"]["redaction"]["identities"] = [i for i in floor if i != w.item("MATE")]
            else:
                extra = sorted(set(c2["repin"]["seal_floor"]["identities"]) | {w.item("PLAIN1")})
                c2["repin"]["seal_floor"] = {"identities": extra, "sha256": sha(common.canonical(extra))}
            with self.subTest(tamper):
                with self.assertRaisesRegex(fam.FamilyStop, msg):
                    fam.run_build(g2 / "alias-sources.json", None, guard=fp.NoGuard(), expect=None,
                                  raw_config=common.canonical(c2))
        # a changed generation-1 file STOPs the next re-pin before anything is written
        meta = next(p for p in (w.out / "inputs" / "meta").glob("*.json"))
        os.chmod(meta, 0o644)
        meta.write_bytes(meta.read_bytes() + b" ")
        code, _, err = self.repin(w, "g3")
        self.assertEqual(code, fam.EXIT["STOP"])
        self.assertIn("local copy changed", err)
        self.assertFalse((w.out / "g3").exists())

    def test_pinned_spec_under_receipts_is_repinned_as_a_redacted_copy(self):
        """SC-8: the P2 survey sits in receipts/P2 and holds sealed names. The re-pin pins a redacted copy inside the new
        generation, so the original can move to restricted custody; verify FAILs while the original stays."""
        rec = self.tmp / "receipts"
        w = standard_world(self.tmp / "w")
        w.spec_path = rec / "survey" / "spec.json"
        w.spec_text = json.dumps({"result": {"spec": "families of " + CANARIES["lb_family_t"] + " stay sealed", "n": 3}})
        self.built(w)
        code, _, err = self.repin(w)
        self.assertEqual(code, 0, err[-800:])
        g2 = w.out / "g2"
        cfg = json.loads((g2 / "alias-sources.json").read_bytes())
        e = next(x for x in cfg["inputs"] if x["name"] == "p2_survey")
        self.assertNotIn("path", e)
        self.assertEqual(e["redacted_from"]["sha256"], sha(w.spec_text.encode()))
        copy = json.loads((g2 / e["local_path"]).read_bytes())
        self.assertEqual(copy["result"]["n"], 3)
        self.assertNotIn(CANARIES["lb_family_t"], json.dumps(copy))
        self.assertIn("[sealed:", copy["result"]["spec"])
        res2 = fam.run_build(g2 / "alias-sources.json", None, guard=fp.NoGuard(), expect=None)
        fam.write_outputs(g2, res2.outputs)
        rep = fam.verify(g2 / "alias-sources.json", g2, guard=fp.NoGuard(), expect=None, receipts_dir=rec)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertEqual(set(rep["canary"]["hits"]), {"receipts/survey/spec.json"})
        os.chmod(w.spec_path, 0o644)
        os.remove(w.spec_path)                       # the original leaves receipts (restricted custody): g2 still verifies
        rep = fam.verify(g2 / "alias-sources.json", g2, guard=fp.NoGuard(), expect=None, receipts_dir=rec)
        self.assertEqual(rep["verdict"], "PASS", rep["failed"])

    def test_cli_pin_mode_arguments(self):
        w = standard_world(self.tmp / "w")
        w.write()
        rec = self.tmp / "receipts"
        for argv, msg in ((["pin", "--out", str(w.out / "g2"), "--from-receipt", str(w.pin_receipt)], "need --from-local"),
                          (["pin", "--out", str(w.out / "g2"), "--from-local", str(w.out)], "needs --from-receipt"),
                          (["pin", "--out", str(w.out / "g2"), "--from-local", str(w.out), "--from-receipt",
                            str(w.pin_receipt), "--ledger", str(rec / "l.jsonl")], "no S3 request"),
                          (["pin", "--out", str(w.out / "g2"), "--from-local", str(self.tmp), "--from-receipt",
                            str(w.pin_receipt)], "outside")):
            with self.subTest(msg):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    code = fam._main(argv, roots=(w.out, rec), guard=fp.NoGuard(), expect=None)
                self.assertEqual(code, fam.EXIT["STOP"])
                self.assertIn(msg, err.getvalue())

    def test_verify_scans_the_receipts_directory(self):
        w = standard_world(self.tmp / "w")
        self.built(w)
        rec = self.tmp / "rec"
        put(rec / "clean.json", b'{"counts": 3}')
        rep = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None, receipts_dir=rec)
        self.assertEqual(rep["verdict"], "PASS", rep["failed"])
        put(rec / "survey" / "spec.json", json.dumps({"text": "... " + CANARIES["lb_family_t"] + " ..."}).encode())
        rep = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None, receipts_dir=rec)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertEqual(rep["canary"]["hits"], {"receipts/survey/spec.json": 1})
        self.assertNotIn(CANARIES["lb_family_t"], json.dumps(rep))
        put(rec / "blob.bin", b"\xff\xfe\x00binary")
        rep = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None, receipts_dir=rec)
        self.assertIn("scan_files_decodable", rep["failed"])


# ======================================================================================================== SC-1
class TestSC1SealedReferenceScreen(FixCase):
    def test_name_subset_and_plural_legacy_seal_provisionally(self):
        w = standard_world(self.tmp / "w")
        w.x_names, w.x_target = ["zq_alpha", "zq_beta"], "y"            # two non-generic names: no name link possible
        w.frozen("SUBSET1", names=["zq_alpha", "zq_beta", "other_c"], target="t_sub")
        w.frozen("PLURAL1", legacy=CANARIES["legacy_sealed"] + "s")
        res = self.built(w)
        cov = res.base.scr.tables["sealedref:" + w.hX]
        self.assertEqual(cov["n_names"], 2)
        sub = fam_of(res, w.item("SUBSET1"))
        self.assertTrue(sub["sealed"])
        self.assertEqual(sub["sealed_basis"], "provisional_pending_review")
        self.assertIn("nds:name_subset", {k for k, _, _ in sub["_adjacency"]})
        plu = fam_of(res, w.item("PLURAL1"))
        self.assertTrue(plu["sealed"])
        self.assertIn("legacy_source_family", {k for k, _, _ in plu["_adjacency"]})
        sa = {r["_subject"][0]: r for r in rows_of(res, "sealed_adjacent")}
        self.assertEqual(sa[sub["family_id"]]["recommendation"], "keep_sealed")
        links = [json.loads(x) for x in res.outputs["screen/links.jsonl"].decode().splitlines()]
        ns = [r for r in links if r["kind"] == "nds:name_subset"]
        self.assertEqual(len(ns), 1)
        self.assertNotIn("shared", ns[0]["evidence"])                 # name hashes only
        self.assertNotIn("zq_alpha", res.outputs["screen/links.jsonl"].decode())

    def test_screen_coverage_counts_non_stop_columns(self):
        rng = np.random.default_rng(5)
        shared = rng.integers(0, 10 ** 6, size=(ROWS, 2)) / 10 ** 6
        tables = []
        for k in range(7):                                            # 7 tables share 2 columns: both are stop columns
            own = rng.integers(0, 10 ** 6, size=(ROWS, 1)) / 10 ** 6
            tables.append(fs.make_table(f"t{k}", "epoch", np.column_stack([shared, own])))
        names = {f"t{k}": [f"a{k}_x", f"b{k}_x", f"c{k}_x"] for k in range(7)}
        scr = fs.near_dup_screen(tables, names)
        t0 = scr.tables["t0"]
        self.assertEqual((t0["n_informative"], t0["n_stop_columns"], t0["n_informative_nonstop"]), (3, 2, 1))
        self.assertEqual(t0["screen_coverage"], "names_only")          # one non-stop column cannot link by values
        self.assertEqual(len(scr.value_strong), 0)

    def test_name_subset_rules_and_stop_relaxed_sealed_pairs(self):
        rng = np.random.default_rng(6)
        mk = lambda: rng.integers(0, 10 ** 6, size=(ROWS, 3)) / 10 ** 6   # noqa: E731
        two = fs.make_table("sealedref:two", "sealed_ref", mk())
        one = fs.make_table("sealedref:one", "sealed_ref", mk())
        sup = fs.make_table("item:sup", "epoch", mk())
        same1 = fs.make_table("item:same1", "epoch", mk())
        more1 = fs.make_table("item:more1", "epoch", mk())
        names = {"sealedref:two": ["p_one", "p_two", "x1"], "sealedref:one": ["solo_name"],
                 "item:sup": ["p_one", "p_two", "p_three"], "item:same1": ["solo_name"], "item:more1": ["solo_name", "zz_q"]}
        scr = fs.near_dup_screen([two, one, sup, same1, more1], names)
        got = sorted((r["a"], r["b"], r["evidence"]["rule"]) for r in scr.name_subsets)
        self.assertEqual(got, [("item:same1", "sealedref:one", "identical"), ("item:sup", "sealedref:two", "superset")])
        # a sealed reference and an epoch table whose shared columns are stop columns: reported as a stop-relaxed weak link
        shared = rng.integers(0, 10 ** 6, size=(ROWS, 2)) / 10 ** 6
        tabs = [fs.make_table("sealedref:r", "sealed_ref", np.column_stack([shared, mk()[:, :1]]))]
        for k in range(6):
            tabs.append(fs.make_table(f"item:e{k}", "epoch", np.column_stack([shared, mk()[:, :1]])))
        scr2 = fs.near_dup_screen(tabs, {})
        relaxed = [r for r in scr2.value_weak if r["evidence"].get("stop_relaxed")]
        self.assertEqual(len(relaxed), 6)
        self.assertTrue(all("sealedref:r" in (r["a"], r["b"]) for r in relaxed))
        self.assertEqual(len(scr2.value_strong), 0)


# ======================================================================================================== SC-2
class TestSC2SealedJoinSet(FixCase):
    def test_catalog_and_record_keys_join_and_source_keys_seal(self):
        w = standard_world(self.tmp / "w")
        w.x_source_key, w.x_evidence, w.x_record_source_key = "/abs/path/xout.csv", [], "/abs/path/xout.csv"
        w.x_catalog_prov = {"platform": "rdatasets", "dataset_id": "CanaryXPkg/CanaryXItem",
                            "url": "https://data.xout-src.org/files/xout-table.csv"}
        w.frozen("XITEM", prov={"platform": "rdatasets", "dataset_id": "otherpkg/canaryxitem",
                                "url": "https://vincentarelbundock.github.io/Rdatasets/csv/CanaryXPkg/CanaryXItem.csv"})
        w.frozen("XURLMIRROR", prov={"platform": "mirror", "dataset_id": "xurl-mirror",
                                     "url": "https://data.xout-src.org/files/xout-table.csv"})
        res = self.built(w)
        lb = res.base.lb
        self.assertIn("rdatasets:canaryxpkg/canaryxitem", lb["SK"])     # from the catalog v1 row, not lineage-B
        self.assertEqual(lb["sk_sources"]["catalog_v1"], 1)
        xi, xu = fam_of(res, w.item("XITEM")), fam_of(res, w.item("XURLMIRROR"))
        self.assertTrue(xi["sealed"] and xu["sealed"])
        self.assertIn("lineage_b_rdatasets_alias", xi["sealed_bases"])
        self.assertEqual(xu["sealed_basis"], "lineage_b_source_key")
        self.assertIn(("normalized_url", "https://data.xout-src.org/files/xout-table.csv"), lb["S_out_keys"])
        rep = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None)
        self.assertEqual(rep["verdict"], "PASS", rep["failed"])
        self.assertTrue(rep["checks"]["base_sealed_recomputed_equal"])
        self.assertEqual(rep["independent_base_sealed"]["unexplained"], 0)
        # the independent path really is independent: without its catalog join XITEM is a disagreement
        with mock.patch.object(fam, "_row_keys_independent", lambda row: set()), \
                mock.patch.object(fp, "rdatasets_keys", lambda prov: set()):
            self.assertNotIn(w.h("XITEM"), fam.independent_base_sealed(res.base.inp))


# ======================================================================================================== SC-3
class TestSC3DisplayWithholding(FixCase):
    def test_union_sealing_a_displayed_member_stops_then_withhold_repin_passes(self):
        w = standard_world(self.tmp / "w")
        w.frozen("PKGSEAL", display="Pkgseal Display Name", prov={
            "platform": "rdatasets", "dataset_id": "canarypkg/pkgsealitem",
            "url": "https://vincentarelbundock.github.io/Rdatasets/csv/CanaryPkg/pkgsealitem.csv"})
        res = self.built(w)
        pkg = next(r for r in rows_of(res, "rdatasets_pkg") if res.pre.fid_of[w.item("PKGSEAL")] in r["family_ids"])
        self.assertFalse(fam_of(res, w.item("PKGSEAL"))["sealed"])
        self.assertIn("Pkgseal Display Name", pkg["members"])
        with self.assertRaisesRegex(fam.FamilyStop, "SC-3") as cm:
            self.final(w, res, decisions_doc(res, group={pkg["review_group_id"]: {"decision": "union"}}))
        ids = cm.exception.detail["withhold_identities"]
        self.assertEqual(ids, [w.item("PKGSEAL")])
        code, _, err = self.repin(w, "g2", withhold=ids)
        self.assertEqual(code, 0, err[-800:])
        g2 = w.out / "g2"
        res2 = fam.run_build(g2 / "alias-sources.json", None, guard=fp.NoGuard(), expect=None)
        F = fam_of(res2, w.item("PKGSEAL"))
        self.assertTrue(F["withheld"] and not F["sealed"])
        self.assertEqual(F["withheld_basis"], ["repin_withhold"])
        self.assertNotIn("Pkgseal Display Name", res2.outputs[fam.OUT_CSV].decode())
        self.assertNotIn("Pkgseal Display Name", res2.outputs[fam.OUT_PRE].decode())
        fam.write_outputs(g2, res2.outputs)
        pkg2 = next(r for r in rows_of(res2, "rdatasets_pkg") if res2.pre.fid_of[w.item("PKGSEAL")] in r["family_ids"])
        doc = decisions_doc(res2, group={pkg2["review_group_id"]: {"decision": "union"}})
        p = g2 / "dec.json"
        put(p, common.canonical(doc))
        fin = fam.run_build(g2 / "alias-sources.json", p, guard=fp.NoGuard(), expect=None)
        self.assertTrue(fam_of(fin, w.item("PKGSEAL"))["sealed"])
        fam.write_outputs(g2, fin.outputs)
        rep = fam.verify(g2 / "alias-sources.json", g2, p, guard=fp.NoGuard(), expect=None)
        self.assertEqual(rep["verdict"], "PASS", rep["failed"])

    def test_recommended_union_with_a_sealed_family_is_withheld_up_front(self):
        w = standard_world(self.tmp / "w")
        w.ds["ALIAS"]["legacy"] = "l1chain"
        w.frozen("PCHAIN", legacy="l1chain", prov={
            "platform": "rdatasets", "dataset_id": "canarypkg/pchainitem",
            "url": "https://vincentarelbundock.github.io/Rdatasets/csv/CanaryPkg/pchainitem.csv"})
        w.frozen("DCHAIN", display="Dchain Display Name", prov={
            "platform": "rdatasets", "dataset_id": "canarypkg/dchainitem", "url": "https://mirror.example.org/files/pchainitem.csv"})
        res = self.built(w)
        P, D = fam_of(res, w.item("PCHAIN")), fam_of(res, w.item("DCHAIN"))
        self.assertTrue(P["sealed"])                                  # provisional (legacy shared with ALIAS)
        self.assertFalse(D["sealed"])
        self.assertTrue(D["withheld"])
        self.assertIn("recommended_union_with_sealed_family", D["withheld_basis"])
        self.assertNotIn("Dchain Display Name", res.outputs[fam.OUT_CSV].decode())
        reg = json.loads(res.outputs[fam.OUT_PRE])
        self.assertTrue(reg["families"][D["family_id"]]["display_withheld"])
        self.assertIsNone(reg["families"][D["family_id"]]["display"])
        self.assertEqual(reg["counts"]["display_withheld"]["families"], 1)
        # accepting every recommended union never seals a member the pre-review showed in clear
        group = {r["review_group_id"]: {"decision": "union_subset", "sets": r["_sets"]}
                 for r in res.pre.rows if r["recommendation"] == "union_subset"}
        fin = self.final(w, res, decisions_doc(res, group=group))
        self.assertTrue(fam_of(fin, w.item("DCHAIN"))["sealed"])


# ======================================================================================================== SC-6 / SC-7
class TestSC6SC7(FixCase):
    def test_sealed_adjacent_evidence_is_non_identifying(self):
        w = standard_world(self.tmp / "w")
        res = self.built(w)
        mate = res.pre.fid_of[w.item("MATE")]
        hard = res.pre.fid_of[w.item("SEALED")]
        row = next(r for r in csv_rows(res) if r["kind"] == "sealed_adjacent" and r["family_ids"].split(";")[0] == mate)
        ev = json.loads(row["evidence"])
        adj = ev["adjacency"]
        self.assertTrue(adj)
        leg = [a for a in adj if a["kind"] == "legacy_source_family"]
        self.assertEqual(leg[0]["value_sha256"], hashlib_sha(CANARIES["legacy_sealed"]))
        self.assertIn(hard, leg[0]["hard_families"])
        self.assertNotIn(CANARIES["legacy_sealed"], row["evidence"])

    def test_four_character_canaries(self):
        cans = fam.Canaries(["wage", "abc", "Longer Sealed Display Name"])
        self.assertIn("wage", list(cans))
        self.assertNotIn("abc", list(cans))
        self.assertTrue(cans.withhold("Hourly Wage Data"))
        self.assertFalse(cans.withhold("wages by state"))
        self.assertEqual(fam.canary_scan({"a": "x wage y", "b": "wages"}, cans), {"a": 1, "b": 0})


def hashlib_sha(text: str) -> str:
    return sha(text.encode())


# ======================================================================================================== SC-9 / CL-6
class TestSC9AndCL6(FixCase):
    def test_sealed_references_must_cover_s_minus_e(self):
        w = standard_world(self.tmp / "w")
        w.content_unavailable = []
        w.write()
        with self.assertRaisesRegex(fam.FamilyStop, "do not cover"):
            w.build()
        w2 = standard_world(self.tmp / "w2")
        w2.write()
        with self.assertRaisesRegex(fam.FamilyStop, "beyond the expected list"):
            fam.run_build(w2.config, None, guard=fp.NoGuard(), expect={"sealed_content_unavailable": []})

    def test_sealed_seed_family_row_trigger_is_hashed(self):
        w = standard_world(self.tmp / "w")
        w.lb_extra = {"seed-val": {"family": "seed-val", "partition": "meta_validation", "source_keys": [],
                                   "datasets": [{"dataset": w.h("SEED"), "source_key": "/abs/seed.csv", "evidence": []}]}}
        res = self.built(w)
        seed_a = next(r for r in csv_rows(res) if r["kind"] == "seed_source_mapping"
                      and r["family_ids"] == res.pre.fid_of["beyondarena:srcA"])
        self.assertTrue(seed_a["trigger_value"].startswith("sha256:"))
        seed_b = next(r for r in csv_rows(res) if r["kind"] == "seed_source_mapping"
                      and r["family_ids"] == res.pre.fid_of["beyondarena:srcB"])
        self.assertEqual(seed_b["trigger_value"], "beyondarena:srcB")

    def test_split_names_its_edge_kinds(self):
        w = standard_world(self.tmp / "w")
        res = self.built(w)
        near = next(r for r in rows_of(res, "multi_member_family")
                    if w.item("NEAR1") in res.pre.families[r["family_ids"][0]]["_members"])
        self.assertEqual(near["evidence"]["edge_kinds"], {"nds:value_strong": 1})
        self.assertEqual(near["evidence"]["split_needs_explicit"], ["nds:value_strong"])
        self.assertTrue(near["evidence"]["nds_edges"][0]["J"])
        g = near["review_group_id"]
        with self.assertRaisesRegex(fam.FamilyStop, "edge_kinds"):
            self.final(w, res, decisions_doc(res, group={g: {"decision": "split"}}))
        with self.assertRaisesRegex(fam.FamilyStop, "edge_kinds go only"):
            self.final(w, res, decisions_doc(res, group={g: {"decision": "confirm_union", "edge_kinds": ["github_repo"]}}))
        with self.assertRaisesRegex(fam.FamilyStop, "edge_kinds go only"):
            self.final(w, res, decisions_doc(res, group={g: {"decision": "split", "edge_kinds": ["normalized_url"]}}))
        fin = self.final(w, res, decisions_doc(res, group={g: {"decision": "split", "edge_kinds": ["nds:value_strong"]}}))
        self.assertNotEqual(fam_of(fin, w.item("NEAR1"))["family_id"], fam_of(fin, w.item("NEAR2"))["family_id"])


if __name__ == "__main__":
    unittest.main()
