"""Offline proofs of the P5 red-team fixes (2026-10-05 review of corpus_build/corpus_schema, plus the owner's
cv_log_ratio_sd ruling of 11:10Z), at build and verify level over the synthetic tests_p5.fixtures_p5.World:

- CL-1 the cv_log_ratio_sd tolerance at build level: a 1-ULP native/outcome hit completes, is counted per context and
  listed in the run receipt (manifest, or the sealed heldout summary for a heldout context); verify re-derives it;
- CL-2 the read guard (verify of a COMPLETE RUN/corpus/epoch-000 without gate receipts) and the write guard order;
- CL-3 verify --counts refuses stops, unlisted exclusions and a wrong completed-native count, and the real run's rules;
- CL-4 the backend-field disagreement split; CL-5 Guard.tick in the statistics loops; CL-6 code identity in the
  pins and per invocation; CL-7 the P4a seed-list pin and the real run's 192 seeds;
- LA-2 the closed ctx-record schema and the train closure; LA-6 the sealed/prohibited family closure; LA-7 heldout
  aggregates only in the sealed heldout summary; LA-9 evaluation reuse at build level; LA-10 the publish kinds.

Every write goes under TMPDIR; nothing touches S3; RUN/corpus is never created (CORPUS_OUT is patched to a TMPDIR
path where a test needs the real-run rules). Run only inside the lane slice:
  python -m unittest tests_p5.test_corpus_redteam -v
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from pathlib import Path
from unittest import mock

from dltrain import common, cpuguard
from dltrain import corpus_build as B
from dltrain import corpus_schema as S
from dltrain import programs as P
from tests_p5 import fixtures_p5 as fx
from tests_p5.test_corpus_build import BuildCase, edit_json, edit_lines, first_detailed_completed, ok_guard, repin, sha

UNIFORM = (fx.CtxSpec("U0", "frozen", 0, "train_fit", "fam-u0"),
           fx.CtxSpec("U1", "incoming", 1, "train_fit", "fam-u1", item="UA"),
           fx.CtxSpec("U2", "frozen", 2, "dev", "fam-u2"),
           fx.CtxSpec("U3", "incoming", 3, "heldout", "fam-u3", item="UB"))     # 8 completed native rows each


def nudge(x: float, ulps: int) -> float:
    for _ in range(abs(ulps)):
        x = math.nextafter(x, math.inf if ulps > 0 else -math.inf)
    return x


class TickCounter(cpuguard.Guard):
    """A permissive Guard that counts tick() calls by the statistics function on the call stack."""

    def __init__(self):
        super().__init__(checker=lambda: (True, [], {}), poll_s=0.01)
        self.sites: dict[str, int] = {}

    def tick(self):
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_name in ("label_stats_from_shards", "runtime_report_from_shards", "distinct_sorted",
                                    "_verify_counts"):
                self.sites[f.f_code.co_name] = self.sites.get(f.f_code.co_name, 0) + 1
                break
            f = f.f_back
        return super().tick()


class RedTeam(BuildCase):

    # ------------------------------------------------------------------------------------------------- CL-1
    def test_cv_sd_tolerance_at_build_level(self):
        """A 1-ULP cv_log_ratio_sd disagreement completes and is receipted with its context, program and both
        values' bits; on a heldout context the receipt is the sealed heldout summary, never the top manifest."""
        w = self.world()
        hits = {}
        for name in ("F1", "F2"):
            k, i, p = first_detailed_completed(w.context(name))
            rows = [json.loads(x) for x in w.paths[name]["native"].read_bytes().splitlines()]
            before = rows[k]["cv_log_ratio_sd"]
            edit_lines(w.paths[name]["native"], lambda rs, k=k: rs[k].update(cv_log_ratio_sd=nudge(rs[k]["cv_log_ratio_sd"], 1)))
            repin(w, name)
            hits[name] = (k, i, p, before)
        out = self.built(w)
        man = json.loads((out / "manifest.json").read_bytes())
        tol = man["tolerances"]
        k, i, p, before = hits["F1"]
        self.assertEqual((tol["native_outcome_ties"], tol["contexts_with_ties"], tol["by_context"]),
                         (1, 1, {w.ids["F1"]: 1}))
        self.assertEqual(tol["items"], [{"context_sha256": w.ids["F1"], "partition": "train_fit",
                                         "field": "cv_log_ratio_sd", "program_sha256": p, "native_line": k + 1,
                                         "outcome_line": i + 1, "native_bits": S.f64_hex(nudge(before, 1)),
                                         "outcome_bits": S.f64_hex(before), "ulps": 1}])
        self.assertEqual(tol["canonical_1ulp_from_writer"], 1)             # canonical == the outcome writer's value
        self.assertEqual(man["deviations"]["cv_log_ratio_sd"]["option"], "scoped_tolerance")
        recs = self.ctx_records(out)
        self.assertEqual(recs[w.ids["F1"]]["counts"]["cv_sd_ulp_ties"], 1)
        rows = {r["program_sha256"]: r for r in self.all_rows(out, "records") if r["context_sha256"] == w.ids["F1"]}
        self.assertEqual(rows[p]["cv_log_ratio_sd"], before)
        self.assertNotIn(w.ids["F2"], json.dumps(tol))
        hsum = json.loads((out / "heldout" / "summary.json").read_bytes())
        k2, i2, p2, before2 = hits["F2"]
        self.assertEqual([(x["context_sha256"], x["program_sha256"], x["native_bits"]) for x in hsum["tolerances"]["items"]],
                         [(w.ids["F2"], p2, S.f64_hex(nudge(before2, 1)))])
        rep = B.verify(out, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        self.assertEqual(B.verify(out, role="heldout_report", guard=ok_guard())["verdict"], "PASS")
        # a doctored receipt no longer matches its recomputation from the ctx-records
        edit_json(out / "manifest.json", lambda d: d["tolerances"].update(native_outcome_ties=0))
        self.assertIn("the manifest's cv_log_ratio_sd tolerance receipt differs from the ctx-records",
                      B.verify(out, guard=ok_guard())["failures"])

    # ------------------------------------------------------------------------------------------------- CL-2
    def test_read_and_write_guards(self):
        u = self.world("u", UNIFORM)
        out = self.built(u, "corpus-u", expected_completed=8)
        real_root = self.tmp / "real"
        receipts = self.tmp / "receipts-dir"
        with mock.patch.object(B, "CORPUS_OUT", out), mock.patch.object(B, "RECEIPTS_DIR", receipts), \
                mock.patch.object(S, "EXPECT_COMPLETED_PER_FOLD", 8):
            # verify, the poison test and the receipt need no gate receipt once the real corpus is COMPLETE
            rep = B.verify(out, schema_allowlist=True, counts=True, poison_test_run=True, poison_work=self.tmp / "pw",
                           receipt=receipts / "verify.json", guard=ok_guard())
            self.assertEqual(rep["verdict"], "PASS", rep["failures"])
            self.assertTrue((receipts / "verify.json").is_file())
            self.assertEqual(rep["poison"]["verdict"], "PASS")
            # a write into it still needs both gate receipts
            kind, _ctx = self.stop_of(lambda: self.build(u, "corpus-u", expected_completed=None))
            self.assertEqual(kind, "gate")
        with mock.patch.object(B, "CORPUS_OUT", out), mock.patch.object(B, "RECEIPTS_DIR", receipts):
            # the real corpus must have been built with the sealed 1,280 (here: 8)
            rep = B.verify(out, guard=ok_guard())
            self.assertIn("the real corpus was built with expected_completed 8, not 1280", rep["failures"])
        # an incomplete real corpus cannot be read
        partial = real_root / "epoch-000"
        partial.mkdir(parents=True)
        (partial / "cursor.json").write_text(json.dumps({"status": "RUNNING"}))
        with mock.patch.object(B, "CORPUS_OUT", partial), self.assertRaises(S.CorpusStop) as cm:
            B.verify(partial, guard=ok_guard())
        self.assertEqual(cm.exception.kind, "gate")
        # a TMPDIR that contains RUN/corpus never counts; nothing under RUN/corpus but CORPUS_OUT is ever writable
        corpus_existed = B.CORPUS_ROOT.exists()
        with mock.patch.dict(os.environ, {"TMPDIR": str(B.RUN)}):
            self.assertIsNone(B._tmp_root())
            for bad in (B.RUN / "corpus" / "epoch-001", B.RUN / "tmp" / "anything", B.CORPUS_OUT, B.CORPUS_OUT / "x"):
                with self.subTest(out=str(bad)), self.assertRaises(S.CorpusStop):
                    B._check_out(bad)
            self.assertEqual(B._check_out(B.SAMPLE_ROOT / "p5-sample-x"), Path(os.path.realpath(B.SAMPLE_ROOT / "p5-sample-x")))
        self.assertEqual(B.CORPUS_ROOT.exists(), corpus_existed)
        # the poison harness's default work root and the receipts directory (with TMPDIR elsewhere)
        with mock.patch.dict(os.environ, {"TMPDIR": str(self.tmp / "narrow")}):
            self.assertEqual(B._check_out(B.SAMPLE_ROOT / "p5-poison-abc"),
                             Path(os.path.realpath(B.SAMPLE_ROOT / "p5-poison-abc")))
            for bad in (B.SAMPLE_ROOT / "p5-poison-", B.SAMPLE_ROOT / "p5-poisonx"):
                with self.subTest(out=str(bad)), self.assertRaises(S.CorpusStop):
                    B._check_out(bad)
            self.assertEqual(B._check_receipt(B.RECEIPTS_DIR / "x.json"), Path(os.path.realpath(B.RECEIPTS_DIR / "x.json")))
            for bad in (B.RECEIPTS_DIR, B.RUN / "other" / "x.json"):
                with self.subTest(receipt=str(bad)), self.assertRaises(S.CorpusStop):
                    B._check_receipt(bad)

    # ------------------------------------------------------------------------------------------------- CL-3
    def test_verify_counts_refuses_stops_and_unlisted_exclusions(self):
        w = self.world()
        for name in ("F1", "F2"):                       # a train_fit and a heldout context stop
            k, _i, _p = first_detailed_completed(w.context(name))
            edit_lines(w.paths[name]["native"], lambda rs, k=k: rs[k].update(cv_mean_log_ratio=rs[k]["cv_mean_log_ratio"] + 1e-3))
            repin(w, name)
        out = self.built(w, continue_on_stop=True)
        cur, man = self.cursor(out), json.loads((out / "manifest.json").read_bytes())
        self.assertEqual((cur["stop_count"], man["stop_count"]), (2, 2))
        self.assertEqual(man["exclusions"], {"sealed_set": 1, "stopped:native_outcome_disagreement": 1, "withdrawn": 1})
        hsum = json.loads((out / "heldout" / "summary.json").read_bytes())
        self.assertEqual(hsum["context_flags"], {w.ids["F2"]: "stopped:native_outcome_disagreement"})
        self.assertEqual(hsum["exclusions"], {"stopped:native_outcome_disagreement": 1})
        fails = B.verify(out, guard=ok_guard())["failures"]
        self.assertIn(f"train_fit: context {w.ids['F1'][:16]} stopped (native_outcome_disagreement), which is not a "
                      f"listed exclusion", fails)
        self.assertIn("2 contexts stopped (--continue-on-stop): a STOP is never a listed exclusion", fails)
        self.assertTrue(any("manifest exclusions outside the listed reasons" in f for f in fails), fails)
        held = B.verify(out, role="heldout_report", guard=ok_guard())["failures"]
        self.assertIn(f"heldout: context {w.ids['F2'][:16]} stopped (native_outcome_disagreement), which is not a "
                      f"listed exclusion", held)
        # a wrong completed-native count, and an inventory that disagrees
        w2 = self.world("w2")
        out2 = self.built(w2, "out2")
        self.assertEqual(B.verify(out2, guard=ok_guard())["verdict"], "PASS")
        edit_json(out2 / "manifest.json", lambda d: d["inputs"]["expected_completed"].update(
            {w2.ids["F1"]: d["inputs"]["expected_completed"][w2.ids["F1"]] + 1}))
        self.assertTrue(any(f.startswith(f"train_fit: context {w2.ids['F1'][:16]} holds") for f in
                            B.verify(out2, guard=ok_guard())["failures"]))
        out3 = self.built(w2, "out3")
        inv = json.loads(gzip.decompress(w2.inventory.read_bytes()))
        for r in inv:
            if r["context_sha256"] == w2.ids["I0"]:
                r["native"]["completed"] += 1
        other = self.tmp / "inventory-other.json.gz"
        other.write_bytes(gzip.compress(json.dumps(inv).encode(), mtime=0))
        edit_json(out3 / "manifest.json", lambda d: d["inputs"].update(
            inventory={"path": str(other), "sha256": sha(other.read_bytes()), "bytes": other.stat().st_size, "rows": 7}))
        self.assertTrue(any("!= the sealed inventory" in f and w2.ids["I0"][:16] in f
                            for f in B.verify(out3, guard=ok_guard())["failures"]))

    def test_real_run_rules(self):
        """The real run (CORPUS_OUT, patched to TMPDIR here) behind PASS gate receipts: --expected-completed is
        refused, the sealed inventory defaults to the epoch's pin and is required, and the seed list must hold the
        192 P2 seed hashes; --continue-on-stop is refused."""
        w = self.world()
        real = self.tmp / "real" / "epoch-000"
        p3, p4 = self.tmp / "p3.json", self.tmp / "p4.json"
        p3.write_text(json.dumps({"verdict": "PASS", "partition_map_sha256": sha(w.pmap.read_bytes())}))
        p4.write_text(json.dumps({"verdict": "PASS"}))
        gates = {"out": real, "gate_p3": p3, "gate_p4": p4}
        with mock.patch.object(B, "CORPUS_OUT", real):
            for label, kw, kind in (("an expected-completed override", {}, "gate"),
                                    ("no inventory (the epoch has no pin)", {"expected_completed": None,
                                                                             "inventory": None}, "gate"),
                                    ("the World's seed list", {"expected_completed": None}, "input"),
                                    ("continue-on-stop", {"expected_completed": None, "continue_on_stop": True}, "gate")):
                with self.subTest(label):
                    got, _ctx = self.stop_of(lambda: self.build(w, **gates, **kw))
                    self.assertEqual(got, kind)
            seeds192 = self.tmp / "seeds192.json"
            seeds192.write_text(json.dumps(sorted(w.seed_set | {sha(f"s{i}".encode()) for i in range(192 - len(w.seed_set))})))
            got, _ctx = self.stop_of(lambda: self.build(w, **gates, expected_completed=None, seeds=seeds192))
            self.assertEqual(got, "native_completed")            # the sealed 1,280: past every input rule
        self.assertFalse((real / "train").exists())

    # ---------------------------------------------------------------------------------------- CL-4 / CL-5
    def test_backend_disagreement_split_and_stats_ticks(self):
        w = self.world()
        guard = TickCounter()
        out = self.built(w, guard=guard)
        self.assertGreater(guard.sites.get("label_stats_from_shards", 0), 0)
        self.assertGreater(guard.sites.get("runtime_report_from_shards", 0), 0)
        self.assertGreater(guard.sites.get("distinct_sorted", 0), 0)
        vguard = TickCounter()
        self.assertEqual(B.verify(out, guard=vguard)["verdict"], "PASS")
        for site in ("label_stats_from_shards", "runtime_report_from_shards", "_verify_counts"):
            self.assertGreater(vguard.sites.get(site, 0), 0, site)
        known = unknown = 0
        for r in self.ctx_records(out).values():
            c = r["counts"]
            self.assertEqual(c["backend_field_disagreements"],
                             c["backend_field_disagreements_known"] + c["backend_field_disagreements_unknown"])
            known += c["backend_field_disagreements_known"]
            unknown += c["backend_field_disagreements_unknown"]
        self.assertGreater(known, 0)          # a local_lambda row whose attempt ran on an aws pool
        self.assertGreater(unknown, 0)        # rows without node, attempt or unanimous pool (derived unknown)
        man = json.loads((out / "manifest.json").read_bytes())
        self.assertEqual(man["reconciliation"]["backend_field_disagreements"],
                         man["reconciliation"]["backend_field_disagreements_known"]
                         + man["reconciliation"]["backend_field_disagreements_unknown"])

    # ------------------------------------------------------------------------------------------------- CL-6
    def test_code_identity(self):
        w = self.world()
        out = self.built(w)
        cur = self.cursor(out)
        self.assertEqual(cur["pins"]["code"], B.code_shas())
        self.assertEqual({json.dumps(inv["code"], sort_keys=True) for inv in cur["ledger"]["invocations"]},
                         {json.dumps(B.code_shas(), sort_keys=True)})
        changed = {**B.code_shas(), "corpus_build.py": "0" * 64}
        with mock.patch.object(B, "code_shas", return_value=changed):
            kind, _ctx = self.stop_of(lambda: self.build(w))
        self.assertEqual(kind, "cursor")                       # a code change refuses the resume
        edit_json(out / "cursor.json", lambda d: d["ledger"]["invocations"].append(
            {**d["ledger"]["invocations"][0], "code": changed}))
        self.assertIn("the build's invocations ran different code (a resume across a code change)",
                      B.verify(out, guard=ok_guard())["failures"])

    # ------------------------------------------------------------------------------------------------- LA-2
    def test_ctx_record_schema_and_train_closure(self):
        w = self.world()
        out = self.built(w)
        recs = self.ctx_records(out)
        rec = recs[w.ids["F1"]]
        self.assertEqual(B.check_ctx_record(rec), [])
        for label, bad in (("an extra top-level key", {**rec, "best_test_rmse": 0.1}),
                           ("an extra count", {**rec, "counts": {**rec["counts"], "rows_seen": 3}}),
                           ("an extra pin key", {**rec, "sources": {**rec["sources"], "native": {
                               **rec["sources"]["native"], "first_line": "{}"}}}),
                           ("a non-hex raw sha", {**rec, "jtt1": {**rec["jtt1"], "raw_sha256": "x"}}),
                           ("an unlisted exclusion", {**rec, "excluded": "too_slow"}),
                           ("a missing label", {k: v for k, v in rec.items() if k != "label"})):
            with self.subTest(label):
                self.assertTrue(B.check_ctx_record(bad))
        self.assertEqual(B.check_ctx_record({**rec, "best_test_rmse": 0.1})[0], "$: additionalProperties ['best_test_rmse']")
        real_project = B.project_context

        def leaky(*a, **kw):
            res = real_project(*a, **kw)
            res.ctx_record["extra_pin"] = 1
            return res
        with mock.patch.object(B, "project_context", leaky):
            kind, _ctx = self.stop_of(lambda: self.build(w, "out-leaky"))
        self.assertEqual(kind, "ctx_record")
        # the train closure binds the train shards, statistics, tokenizer and test-independent pins only
        ref = B.train_closure_ref(out)
        self.assertEqual(ref["path"], "train/manifest.json")
        doc = json.loads((out / ref["path"]).read_bytes())
        self.assertEqual((doc["schema"], doc["partition"], doc["role"], doc["run_plan_ref"]),
                         (B.PARTITION_SCHEMA, "train_fit", "fit", True))
        self.assertEqual(set(doc["tables"]), {"records", "contexts", "refused"})
        files = [f["path"] for t in doc["tables"].values() for f in t["files"]]
        self.assertTrue(files and all(f.startswith(("train/records/", "train/contexts/")) for f in files))
        text = json.dumps(doc)
        for needle in ("provenance", "heldout", "exposed", "lineage", "ctx-records", str(self.tmp), str(w.root)):
            self.assertNotIn(needle, text)
        self.assertEqual(doc["inputs"], {"partition_map_sha256": sha(w.pmap.read_bytes()),
                                         "seeds_sha256": sha(w.seeds.read_bytes())})
        for part in ("dev", "diag", "heldout"):
            self.assertIs(json.loads(B.manifest_path(out, part).read_bytes())["run_plan_ref"], False)
        fake = self.tmp / "not-a-closure"
        (fake / "train").mkdir(parents=True)
        (fake / "train" / "manifest.json").write_text(json.dumps({**doc, "run_plan_ref": False}))
        with self.assertRaises(ValueError):
            B.train_closure_ref(fake)

    # ------------------------------------------------------------------------------------------------- LA-6
    def test_closure_sealed_and_prohibited_groups(self):
        """The reviewer's probe: one dataset and family whose contexts hold these roles."""

        def run(roles):
            entries, pmap = {}, {}
            for j, role in enumerate(roles):
                cid = sha(f"closure:{roles}:{j}".encode())
                entries[cid] = {"status": "included", "dataset_hash": "d" * 16}
                pmap[cid] = {"family_id": "fam-one", "split_role": role}
            members = {p: [] for p in P.PARTITIONS}
            B.closure_checks(entries, pmap, members)

        for roles in (("train_fit", "sealed", "diag"), ("train_fit", "prohibited", "train_fit"),
                      ("dev", "sealed", "dev"), ("heldout", "prohibited"), ("sealed", "prohibited"),
                      ("train_fit", "dev", "train_fit")):
            with self.subTest(roles=roles), self.assertRaises(S.CorpusStop) as cm:
                run(roles)
            self.assertEqual(cm.exception.kind, "closure")
        for roles in (("sealed", "sealed"), ("prohibited",), ("train_fit", "diag", "train_fit"), ("dev", "dev")):
            with self.subTest(roles=roles):
                run(roles)

    # ------------------------------------------------------------------------------------------------- LA-7
    def test_heldout_aggregates_stay_sealed(self):
        specs = (fx.CtxSpec("H0", "frozen", 1, "train_fit", "fam-h0"),
                 fx.CtxSpec("H1", "incoming", 2, "dev", "fam-h1", item="HA"),
                 fx.CtxSpec("H2", "frozen", 2, "heldout", "fam-h2", leak="exact"),
                 fx.CtxSpec("H3", "incoming", 3, "heldout", "fam-h3", item="HB"))
        w = self.world("held", specs)
        out = self.built(w)
        man = json.loads((out / "manifest.json").read_bytes())
        self.assertEqual(man["exclusions"], {"sealed_set": 1, "withdrawn": 1})
        self.assertNotIn("heldout", man["totals"]["by_partition"])
        self.assertEqual(set(man["sealed"]["heldout"]), {"partition", "exposed", "lineage", "provenance", "label_stats",
                                                         "summary"})
        self.assertEqual(set(man["label_stats_sha256"]), {"train_fit", "dev", "diag", "seed_stats"})
        hsum = json.loads((out / "heldout" / "summary.json").read_bytes())
        self.assertEqual(hsum["exclusions"], {"target_leak": 1})
        self.assertEqual(hsum["context_flags"], {w.ids["H2"]: "target_leak"})
        self.assertEqual(hsum["totals"]["by_partition"]["heldout"]["contexts"], 1)
        cur = self.cursor(out)
        self.assertNotIn("context_flags", cur)
        derived = json.dumps({k: man[k] for k in ("exclusions", "totals", "reconciliation", "tolerances")})
        self.assertNotIn(w.ids["H2"], derived)
        self.assertNotIn(w.ids["H3"], derived)
        for p in ("train_fit", "dev"):
            for path in (B.manifest_path(out, p), B.part_dir(out, p) / "label-stats.json",
                         B.manifest_path(out, p, "ctx-records"), B.manifest_path(out, p, "lineage")):
                self.assertNotIn(w.ids["H2"], path.read_text(), str(path))
        rep = B.verify(out, poison_test_run=True, poison_work=self.tmp / "poison", guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        picked = {x["context_sha256"] for x in rep["poison"]["selected"]}
        self.assertIn(w.ids["H3"], picked)
        self.assertNotIn(w.ids["H2"], picked)                  # a heldout context with no rows is never copied
        self.assertEqual(rep["poison"]["nonempty_partitions"], ["heldout", "dev"])

    # ------------------------------------------------------------------------------------------------- LA-9
    def test_evaluation_reuse_at_build_level(self):
        w = self.world()
        sc = w.context("F1")
        entry = next(e for e in w.epoch_doc["contexts"] if e["context_sha256"] == sc.cid)
        rows = [json.loads(x) for x in w.paths["F1"]["outcome"].read_bytes().splitlines()]
        i = next(j for j, r in enumerate(rows) if r["phase"] == "bootstrap" and r["record_status"] == "completed"
                 and r.get("program"))
        reuse = {"schema": S.REUSE_SCHEMA, "source_context_sha256": "4" * 64, "new_training_executed": False,
                 "compatibility_identity": {"input_raw_sha256": entry["input_ref"]["raw_sha256"],
                                            "evaluator_sha256": S.EVALUATOR_SHA256, "candidate_timeout_ms": 90000,
                                            "train_rows": sc.spec.n_train, "encoded_features": sc.spec.p,
                                            "test_rows": sc.spec.n_test}}
        edit_lines(w.paths["F1"]["outcome"], lambda rs: rs[i].update(evaluation_reuse=reuse))
        repin(w, "F1")
        out = self.built(w)
        counts = self.ctx_records(out)[sc.cid]["counts"]
        self.assertEqual((counts["evaluation_reuse_rows"], counts["evaluation_reuse_foreign"]), (1, 1))
        man = json.loads((out / "manifest.json").read_bytes())
        self.assertEqual((man["reconciliation"]["evaluation_reuse_rows"], man["reconciliation"]["evaluation_reuse_foreign"]),
                         (1, 1))
        rep = B.verify(out, poison_test_run=True, poison_work=self.tmp / "poison", guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        # a source context mapped to another partition is a STOP (the map knows it)
        pmap = json.loads(w.pmap.read_bytes())
        pmap["4" * 64] = {"family_id": "fam-other", "split_role": "heldout"}
        other = self.tmp / "map-reuse.json"
        other.write_bytes(json.dumps(pmap).encode())
        kind, ctx = self.stop_of(lambda: self.build(w, "out-reuse", partition_map=other))
        self.assertEqual((kind, ctx), ("evaluation_reuse", sc.cid))

    # ------------------------------------------------------------------------------------------------- LA-10
    def test_publish_kinds(self):
        w = self.world()
        out = self.built(w)
        receipt = self.tmp / "verify.json"
        rep = B.verify(out, poison_test_run=True, poison_work=self.tmp / "poison", receipt=receipt, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        g18 = self.tmp / "g18.json"
        g18.write_text(json.dumps({"verdict": "PASS"}))
        with mock.patch("dltrain.s3io.make_client", side_effect=AssertionError("publish must not open S3")):
            with self.assertRaises(S.CorpusStop) as cm:
                B.publish(out=out, verify_receipt=receipt, g18_receipt=g18, s3_cap_bytes=10**12, guard=ok_guard())
        plan = cm.exception.plan["plan"]
        for o in plan:
            sealed = S.is_sealed_path(o["source"])
            with self.subTest(source=o["source"]):
                self.assertEqual(o["sealed"], sealed)
                if o["kind"] == "manifest":
                    self.assertTrue(o["key"].startswith(B.PUBLISH_SEALED_MANIFEST_PREFIX if sealed
                                                        else B.PUBLISH_MANIFEST_PREFIX))
                else:
                    self.assertTrue(o["key"].startswith(B.PUBLISH_PREFIXES[o["kind"]]))
                    if o["kind"] == "shard":
                        self.assertTrue(o["source"].startswith(("train/", "eval/")))
                    if sealed:
                        self.assertEqual(o["kind"], "sealed")
        self.assertEqual([o["source"] for o in plan if o.get("run_plan_ref")], ["train/manifest.json"])
        self.assertIn("heldout/summary.json", {o["source"] for o in plan if o["sealed"]})
        self.assertTrue(any(o["kind"] == "sealed" and o["source"].endswith(".json") for o in plan))   # sidecars
        self.assertEqual({o["kind"] for o in plan}, {"shard", "sealed", "lineage", "provenance", "manifest"})


if __name__ == "__main__":
    import unittest
    unittest.main()
