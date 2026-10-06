"""Offline proofs for dltrain.corpus_build (PLAN §5 P5 train-only corpus build; spec §3).

Every build runs over the synthetic tests_p5.fixtures_p5.World (frozen and incoming context trees in the layouts
dltrain.programs reads, the incoming cursor and transport index, exposed side files with planted sentinels, a JTT1
cache and a train_fit tokenizer) under tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']). Nothing touches S3:
the JTT1 fetch path runs against a fake client and a patched s3io.get_exact. The CPU admission guard is given a
permissive checker and the slice-limit check is stubbed (tests_p5.test_corpus_hygiene owns those call-order proofs).

Run only inside the lane slice: python -m unittest tests_p5.test_corpus_build -v
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import math
import shutil
import unittest
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dltrain import common, cpuguard, labels, sampler
from dltrain import corpus_build as B
from dltrain import corpus_schema as S
from dltrain import programs as P
from tests_p5 import fixtures_p5 as fx

TARGET = 20_000          # tiny shard target: records ~8-11 KB and exposed ~24-37 KB per World context


def ok_guard() -> cpuguard.Guard:
    return cpuguard.Guard(checker=lambda: (True, [], {}), poll_s=0.01)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def edit_lines(path: Path, fn) -> None:
    """Rewrite a JSONL file through fn(rows) (rows are re-dumped as the fixture writes them)."""
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    fn(rows)
    path.write_bytes(b"".join(fx.dump(r) for r in rows))


def edit_json(path: Path, fn) -> None:
    doc = json.loads(path.read_bytes())
    fn(doc)
    path.write_bytes(json.dumps(doc, sort_keys=True).encode())


def repin(w: fx.World, name: str) -> None:
    """After editing a context's sources: recompute the packer summary (frozen history-ref.json or the incoming
    cursor entry and fingerprint) and the epoch native_source, prepared_ref and input pins."""
    sc, paths = w.context(name), w.paths[name]
    native = paths["native"].read_bytes()
    transport = paths["transport"].read_bytes()
    if sc.spec.cohort == "frozen":
        outcome = paths["outcome"].read_bytes()
        edit_json(paths["summary"], lambda d: d.update(B.packer_stats(native, outcome, transport)))
    else:
        cur = json.loads(w.sources.incoming_cursor.read_bytes())
        idx = B.ItemIndex(paths["outcome"], {sc.cid: cur[sc.cid]["stats"]["outcome_bytes"]})
        outcome = b"".join(idx.lines(sc.cid))
        boot = paths["bootstrap_results"].read_bytes()
        cur[sc.cid]["stats"].update(B.packer_stats(native, outcome, transport))
        cur[sc.cid]["fingerprint"] = sha(native + outcome + transport + boot)
        w.sources.incoming_cursor.write_bytes(json.dumps(cur, sort_keys=True).encode())
    prepared = paths["prepared"].read_bytes()

    def fix(doc):
        for e in doc["contexts"]:
            if e["context_sha256"] == sc.cid:
                e["native_source"].update(bytes=len(native), sha256=sha(native))
                e["prepared_ref"].update(bytes=len(prepared), sha256=sha(prepared))
    edit_json(w.epoch, fix)
    w.epoch_sha256 = sha(w.epoch.read_bytes())


def set_input(w: fx.World, name: str, raw: bytes) -> None:
    """Replace a context's JTT1 input (cache file and epoch input_ref)."""
    body = fx.gz(raw)
    ref = fx.input_ref(raw, body)
    (w.jtt1_cache / f"{ref['raw_sha256']}.jtt1.gz").write_bytes(body)

    def fix(doc):
        for e in doc["contexts"]:
            if e["context_sha256"] == w.ids[name]:
                e["input_ref"] = ref
    edit_json(w.epoch, fix)
    w.epoch_sha256 = sha(w.epoch.read_bytes())


def first_detailed_completed(sc: fx.SynthContext) -> tuple[int, int, str]:
    """(native index, outcome index of its last evolution row, sha) of the first detailed completed native program."""
    last = {}
    for i, r in enumerate(sc.outcome_rows):
        if r.get("phase") == "evolution":
            last[r["program_sha256"]] = i
    for k, r in enumerate(sc.native_rows):
        p = r["program"]["program_sha256"]
        o = sc.outcome_rows[last[p]] if p in last else None
        if r["status"] == "completed" and o is not None and (o.get("evaluation_detail") or {}).get("inner_cv"):
            return k, last[p], p
    raise AssertionError("no detailed completed program")


def values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from values(v)
    else:
        yield obj


def parquet_files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.parquet"))}


def output_files(root: Path) -> dict[str, bytes]:
    """Every shard and every deterministic statistics or manifest file (not the cursor or the top manifest)."""
    out = parquet_files(root)
    for p in sorted(root.rglob("*.json")):
        rel = str(p.relative_to(root))
        if rel not in ("cursor.json", "manifest.json"):
            out[rel] = p.read_bytes()
    return out


class FakeEvents:
    def __init__(self):
        self.handlers = []

    def register(self, name, handler, unique_id=None):
        self.handlers.append((name, handler))


class FakeClient:
    """Stands in for an s3io read client: the patched get_exact fires its before-send handlers once per call."""

    def __init__(self):
        self.meta = SimpleNamespace(events=FakeEvents())


class BuildCase(unittest.TestCase):
    def setUp(self):
        self._tmp = fx.tmpdir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        patcher = mock.patch.object(cpuguard, "require_slice_limits", return_value={})
        self.slice_check = patcher.start()
        self.addCleanup(patcher.stop)

    def world(self, name: str = "w", specs=fx.DEFAULT_SPECS, **kw) -> fx.World:
        return fx.World(self.tmp / name, specs, **kw)

    def build(self, w: fx.World, out: str = "out", **kw) -> dict:
        args = dict(epoch=w.epoch, epoch_sha256=w.epoch_sha256, partition_map=w.pmap, seeds=w.seeds,
                    tokenizer=w.tok_dir, out=self.tmp / out, jtt1_cache=w.jtt1_cache, budget_bytes=0, offline=True,
                    incoming_cursor=w.sources.incoming_cursor,
                    incoming_transport_index=w.sources.incoming_transport_index, inventory=w.inventory,
                    expected_completed=w.expected_completed, guard=ok_guard())
        args.update(kw)
        return B.run(**args)

    def built(self, w: fx.World, out: str = "out", **kw) -> Path:
        res = self.build(w, out, **kw)
        self.assertEqual((res["status"], res["exit"]), ("COMPLETE", B.EXIT_OK), res)
        return self.tmp / out

    def stop_of(self, fn) -> tuple[str, str | None]:
        """(kind, context) of a STOP, whether raised before the cursor exists or returned by run()."""
        try:
            res = fn()
        except S.CorpusStop as err:
            return err.kind, err.context
        self.assertEqual(res["exit"], B.EXIT_STOP, res)
        return res["stop"]["kind"], res["stop"]["context"]

    def cursor(self, out: Path) -> dict:
        return json.loads((out / "cursor.json").read_bytes())

    def ctx_records(self, out: Path) -> dict[str, dict]:
        cur = self.cursor(out)
        return {r["context_sha256"]: r for p in P.PARTITIONS for r in B.read_ctx_records(out, cur, p)}

    def all_rows(self, out: Path, table: str) -> list[dict]:
        rows = []
        cur = self.cursor(out)
        for p in P.PARTITIONS:
            for sh in cur["writers"][p][table]["shards"]:
                rows += pq.read_table(out / sh["path"], use_threads=False).to_pylist()
        return rows


class Build(BuildCase):

    def test_end_to_end_and_verify(self):
        w = self.world()
        out = self.built(w)
        self.assertGreater(self.slice_check.call_count, 0)
        rep = B.verify(out, schema_allowlist=True, counts=True, rehash_sources=True, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        self.assertGreater(rep["checked"]["sources_rehashed"], 0)
        self.assertGreater(rep["checked"]["parquet_files"], 10)
        for rel in ("train/records", "train/contexts", "provenance/train_fit", "eval/dev/records", "eval/dev/refused",
                    "eval/diag/records", "heldout/records", "heldout/contexts", "heldout/refused", "heldout/ctx-records",
                    "heldout/exposed", "heldout/lineage", "exposed/train_fit", "exposed/dev", "exposed/diag",
                    "lineage/train_fit", "lineage/dev", "lineage/diag", "provenance/dev", "provenance/diag"):
            self.assertTrue((out / rel).is_dir() and any((out / rel).iterdir()), rel)
        for rel in ("exposed/heldout", "lineage/heldout", "train/exposed", "train/lineage", "eval/heldout",
                    "train/ctx-records", "eval/dev/ctx-records", "provenance/heldout"):
            self.assertFalse((out / rel).exists(), rel)
        for rel in ("train/manifest.json", "train/label-stats.json", "train/seed-stats.json", "eval/dev/manifest.json",
                    "eval/dev/label-stats.json", "eval/diag/label-stats.json", "heldout/manifest.json",
                    "heldout/label-stats.json", "heldout/summary.json", "provenance/train_fit/manifest.json",
                    "runtime-report.json", "manifest.json", "cursor.json", "build.lock"):
            self.assertTrue((out / rel).is_file(), rel)
        self.assertFalse((out / "eval/dev/seed-stats.json").exists())
        # nothing under train/ but the train shards, the statistics and the train closure
        self.assertEqual({p.relative_to(out).parts[1] for p in (out / "train").rglob("*") if p.is_file()} - {"records", "contexts"},
                         {"manifest.json", "label-stats.json", "seed-stats.json"})
        recs = self.ctx_records(out)
        self.assertEqual(set(recs), {sc.cid for sc in w.contexts.values()})
        for sc in w.contexts.values():
            r = recs[sc.cid]
            with self.subTest(context=sc.spec.name):
                self.assertEqual(r["split_role"], sc.spec.split_role)
                self.assertEqual(r["summary"]["unique_programs"], sc.stats()["unique_programs"])
                self.assertEqual(r["records"] + r["refused"], sc.stats()["unique_programs"])
                self.assertEqual(r["counts"]["native_completed"], w.expected_completed[sc.cid])
                self.assertEqual(r["checks"], {f"R{i}": True for i in range(1, 11)})
                self.assertEqual(r["dict_crosscheck"]["source"], "context_rows")
        man = json.loads((out / "manifest.json").read_bytes())
        by_part = man["totals"]["by_partition"]
        # the non-sealed manifest's totals, reconciliation and tolerances cover train_fit, dev and diag only; heldout's
        # are in the sealed heldout summary, referenced by sha256 (red-team LA-7)
        open_scs = [sc for sc in w.contexts.values() if sc.spec.split_role != "heldout"]
        self.assertEqual({p: d["contexts"] for p, d in by_part.items()},
                         dict(Counter(sc.spec.split_role for sc in open_scs)))
        total = sum(sc.stats()["unique_programs"] for sc in open_scs)
        self.assertEqual(sum(d["records"] + d["refused"] for d in by_part.values()), total)
        self.assertEqual(sum(d["unique_programs"] for d in man["totals"]["by_cohort"].values()), total)
        self.assertEqual(man["exclusions"], {"sealed_set": 1, "withdrawn": 1})
        self.assertEqual(man["tokenizer_closure"]["relation"], "equal")
        self.assertEqual(man["reconciliation"]["R6"], len(open_scs))
        self.assertIs(man["run_plan_ref"], False)
        closure = B.train_closure_ref(out)
        self.assertEqual({k: closure[k] for k in ("path", "sha256")}, man["train_closure"])
        self.assertEqual(man["sealed"]["heldout"]["summary"]["sha256"], sha((out / "heldout" / "summary.json").read_bytes()))
        held_sc = w.context("F2")
        hsum = json.loads((out / "heldout" / "summary.json").read_bytes())
        self.assertEqual(hsum["totals"]["by_partition"]["heldout"]["unique_programs"], held_sc.stats()["unique_programs"])
        self.assertEqual(hsum["reconciliation"]["R6"], 1)
        self.assertEqual(man["deviations"]["cv_log_ratio_sd"], S.CV_SD_POLICY)
        self.assertEqual(man["deviations"]["runtime_target"], S.RUNTIME_POLICY_NOTE)
        self.assertEqual((man["tolerances"]["native_outcome_ties"], man["tolerances"]["items"]), (0, []))
        self.assertGreater(man["tolerances"]["canonical_stored"], 0)
        self.assertNotIn("context_flags", self.cursor(out))
        self.assertEqual((self.cursor(out)["stop_count"], man["stop_count"]), (0, 0))
        self.assertEqual(len(self.all_rows(out, "records")),
                         sum(d["records"] for d in by_part.values()) + hsum["totals"]["by_partition"]["heldout"]["records"])
        # the expected-completed override is honoured: the production pin (1280) is a STOP on this World
        kind, ctx = self.stop_of(lambda: self.build(w, "out-1280", expected_completed=None))
        self.assertEqual(kind, "native_completed")
        # the CLI over a World whose contexts all hold 8 completed native rows
        specs = (fx.CtxSpec("U0", "frozen", 0, "train_fit", "fam-u0"),
                 fx.CtxSpec("U1", "incoming", 1, "train_fit", "fam-u1", item="UA"),
                 fx.CtxSpec("U2", "frozen", 2, "dev", "fam-u2"),
                 fx.CtxSpec("U3", "incoming", 3, "heldout", "fam-u3", item="UB"))
        u = self.world("u", specs)
        self.assertEqual(set(u.expected_completed.values()), {8})
        real_guard = cpuguard.Guard
        argv = ["run", "--epoch", str(u.epoch), "--epoch-sha256", u.epoch_sha256, "--partition-map", str(u.pmap),
                "--seeds", str(u.seeds), "--tokenizer", str(u.tok_dir), "--out", str(self.tmp / "cli"),
                "--jtt1-cache", str(u.jtt1_cache), "--budget-bytes", "0", "--offline", "--inventory", str(u.inventory),
                "--incoming-cursor", str(u.sources.incoming_cursor),
                "--incoming-transport-index", str(u.sources.incoming_transport_index), "--expected-completed", "8"]
        stdout = io.StringIO()
        with mock.patch.object(cpuguard, "Guard", lambda **kw: real_guard(checker=lambda: (True, [], {}), **kw)), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(B.main(argv), B.EXIT_OK)
            self.assertEqual(B.main(["verify", "--out", str(self.tmp / "cli"), "--schema-allowlist", "--counts"]),
                             B.EXIT_OK)
            self.assertEqual(B.main(argv[:-2] + ["--expected-completed", "9", "--out", str(self.tmp / "cli-9")]),
                             B.EXIT_STOP)
        lines = [json.loads(x) for x in stdout.getvalue().splitlines()]
        self.assertEqual((lines[0]["status"], lines[1]["verdict"]), ("COMPLETE", "PASS"))

    def test_records_feed_labels_and_sampler(self):
        w = self.world()
        out = self.built(w)
        ctxs, recs = [], []
        for p in ("train_fit", "dev", "diag"):
            c, r = B.load_split(out, p, S.READ_ROLE[p])
            ctxs += c
            recs += r
        self.assertTrue(recs)
        for r in recs:
            self.assertEqual(list(r), S.TRAIN_RECORD_SCHEMA.names)
        for c in ctxs:
            self.assertEqual(list(c), S.TRAIN_CONTEXT_SCHEMA.names)
        assignment = {sc.spec.family_id: ("train_fit" if sc.spec.split_role in ("train_fit", "diag") else sc.spec.split_role)
                      for sc in w.contexts.values()}
        enc = {c["context_sha256"]: np.zeros(4, dtype=np.float32) for c in ctxs}
        corpus = sampler.Corpus.build(ctxs, recs, enc, assignment)
        fit = sampler.fit(corpus, sampler.SamplerConfig(b_ctx=2))
        train_ids = {sc.cid for sc in w.contexts.values() if sc.spec.split_role == "train_fit"}
        self.assertEqual(set(fit.pi), train_ids)
        self.assertEqual(fit.excluded["diag_contexts"], 1)
        # labels.label_corpus over the read-back train rows reproduces train/label-stats.json
        tc, tr = B.load_split(out, "train_fit")
        table = labels.label_corpus(tc, tr)
        labs = [lab for ls in table.records.values() for lab in ls]
        stats = json.loads((out / "train" / "label-stats.json").read_bytes())
        self.assertEqual((stats["contexts"], stats["records"]), (len(tc), len(tr)))
        self.assertEqual(stats["context_status_counts"], dict(Counter(c.status for c in table.contexts.values())))
        self.assertEqual(stats["outcome_counts"], dict(Counter(lab.outcome for lab in labs)))
        self.assertEqual(stats["q_status_counts"], dict(Counter(lab.q_status for lab in labs)))
        self.assertEqual(stats["gate_pass_counts"], dict(Counter("null" if lab.gate_pass is None else
                                                                 str(lab.gate_pass).lower() for lab in labs)))
        qs = np.asarray([lab.q for lab in labs if lab.q is not None], dtype=np.float64)
        self.assertEqual(stats["q_quantiles"]["p50"], float(np.percentile(qs, 50, method="linear")))
        self.assertEqual(stats["q_quantiles"]["p100"], float(qs.max()))
        self.assertGreater(stats["completed_detailed"], 0)
        self.assertGreater(stats["completed_detailless"], 0)
        by = {(lab.context_sha256, lab.program_sha256): lab for lab in labs}
        for r in tr:
            self.assertIs(r["gate_pass"], by[(r["context_sha256"], r["program_sha256"])].gate_pass)
            self.assertEqual(r["outcome"], labels.normalize_outcome(r))
        for c in tc:
            cl = table.contexts[c["context_sha256"]]
            self.assertEqual((c["s_c"], c["eps_c"], c["label_status"]), (cl.s_c, cl.eps_c, cl.status))
            self.assertEqual(c["q_cap"], cl.q_cap)
            self.assertEqual(c["baseline_rmse_j"], None if cl.baseline_rmse_j is None else list(cl.baseline_rmse_j))
            self.assertEqual(c["s_c"], labels.robust_scale(c["y_train"]))
            self.assertEqual(c["n_train"], len(c["y_train"]))
        seeds = json.loads((out / "train" / "seed-stats.json").read_bytes())
        shared = [r for r in tr if r["shared_seed"]]
        self.assertEqual(seeds["records_shared_seed"], len(shared))
        self.assertEqual(seeds["seeds_present"], len({r["program_sha256"] for r in shared}))
        self.assertEqual(seeds["seeds"], len(w.seed_set))
        self.assertEqual(seeds["seed_list_sha256"], sha(w.seeds.read_bytes()))

    def test_split_routing_and_exclusions(self):
        w = self.world()
        out = self.built(w)
        role_of = {sc.cid: sc.spec.split_role for sc in w.contexts.values()}
        for p in P.PARTITIONS:
            c, r = B.load_split(out, p, S.READ_ROLE[p])
            with self.subTest(partition=p):
                self.assertEqual({x["context_sha256"] for x in c}, {k for k, v in role_of.items() if v == p})
                self.assertTrue(all(x["split_role"] == p for x in c + r))
                self.assertEqual({x["context_sha256"] for x in r} - {x["context_sha256"] for x in c}, set())
        tc, tr = B.load_split(out, "train_fit")
        train_fams = {sc.spec.family_id for sc in w.contexts.values() if sc.spec.split_role in ("train_fit", "diag")}
        self.assertEqual({r["family_id"] for r in tr} | {c["family_id"] for c in tc}, train_fams & {
            sc.spec.family_id for sc in w.contexts.values() if sc.spec.split_role == "train_fit"})
        self.assertNotIn(w.ids["S"], {r["context_sha256"] for r in self.all_rows(out, "lineage")})
        man = json.loads((out / "manifest.json").read_bytes())
        self.assertEqual(man["exclusions"], {"sealed_set": 1, "withdrawn": 1})
        # map role prohibited is counted as prohibited_excluded
        pmap = json.loads(w.pmap.read_bytes())

        def with_map(name, change):
            doc = json.loads(json.dumps(pmap))
            change(doc)
            path = self.tmp / f"{name}.json"
            path.write_bytes(json.dumps(doc).encode())
            return path

        prohibited = with_map("map-prohibited", lambda d: d.__setitem__(w.ids["S"], {"family_id": "fam-s",
                                                                                     "split_role": "prohibited"}))
        out2 = self.built(w, "out-prohibited", partition_map=prohibited)
        self.assertEqual(json.loads((out2 / "manifest.json").read_bytes())["exclusions"],
                         {"prohibited_excluded": 1, "withdrawn": 1})
        cases = [
            ("unknown map role", lambda d: d.__setitem__(w.ids["S"], {"family_id": "fam-s", "split_role": "quarantine"}),
             "unlisted_exclusion"),
            ("a dataset in two families", lambda d: d[w.ids["F3"]].__setitem__("family_id", "fam-z"), "closure"),
            ("a family mixing train and dev", lambda d: d[w.ids["I1"]].__setitem__("family_id", "fam-a"), "closure"),
            ("a frozen fold-0 diag context", lambda d: d[w.ids["F0"]].__setitem__("split_role", "diag"), "closure"),
            # red-team LA-6: sealing and rights are family properties, so a sealed or prohibited group is the family's
            # only group (fam-b holds train_fit F1 and diag F3; fam-e holds dev I1)
            ("a sealed context in a train family", lambda d: d[w.ids["S"]].__setitem__("family_id", "fam-b"), "closure"),
            ("a prohibited context in a dev family",
             lambda d: d.__setitem__(w.ids["S"], {"family_id": "fam-e", "split_role": "prohibited"}), "closure"),
            ("a sealed context in a heldout family", lambda d: d[w.ids["S"]].__setitem__("family_id", "fam-c"), "closure"),
        ]
        for i, (label, change, kind) in enumerate(cases):
            with self.subTest(label):
                path = with_map(f"map-{i}", change)
                got, _ctx = self.stop_of(lambda: self.build(w, f"out-bad-{i}", partition_map=path))
                self.assertEqual(got, kind)
                self.assertFalse((self.tmp / f"out-bad-{i}" / "train").exists())
        # an epoch exclusion reason outside P1's list is a STOP
        epoch = json.loads(w.epoch.read_bytes())
        for e in epoch["contexts"]:
            if e["context_sha256"] == w.ids["X"]:
                e["exclusion_reason"] = "mystery"
        bad_epoch = self.tmp / "epoch-bad.json"
        bad_epoch.write_bytes(json.dumps(epoch).encode())
        got, _ctx = self.stop_of(lambda: self.build(w, "out-bad-epoch", epoch=bad_epoch,
                                                    epoch_sha256=sha(bad_epoch.read_bytes())))
        self.assertEqual(got, "unlisted_exclusion")

    def test_no_exposed_values_reachable(self):
        w = self.world()
        out = self.built(w)
        sent = w.sentinel_values()
        self.assertTrue(sent["floats"] and sent["strings"] and sent["f32"] and sent["ints"])
        floats = sent["floats"] | sent["f32"]
        needles = [s.encode() for s in sent["strings"]]

        def assert_clean(vals, where):
            for v in vals:
                if v is None or isinstance(v, bool):
                    continue
                if isinstance(v, float):
                    self.assertNotIn(v, floats, where)
                elif isinstance(v, int):
                    self.assertNotIn(v, sent["ints"], where)
                elif isinstance(v, str):
                    self.assertFalse(any(s in v for s in sent["strings"]), where)

        cur = self.cursor(out)
        scanned = 0
        nonsealed_parquet = [p for p in out.rglob("*.parquet")
                             if p.relative_to(out).parts[0] in ("train", "eval", "lineage")]
        self.assertTrue(nonsealed_parquet)
        for p in nonsealed_parquet:
            assert_clean(values(pq.read_table(p, use_threads=False).to_pylist()), str(p))
            raw = p.read_bytes()
            self.assertFalse(any(n in raw for n in needles), str(p))
            scanned += 1
        jsons = B.nonsealed_json(out, cur)
        self.assertIn(out / "train" / "label-stats.json", jsons)
        self.assertIn(out / "train" / "seed-stats.json", jsons)
        self.assertIn(out / "runtime-report.json", jsons)
        for p in jsons:
            raw = p.read_bytes()
            assert_clean(values(json.loads(raw)), str(p))
            self.assertFalse(any(n in raw for n in needles), str(p))
            self.assertEqual(B.key_violations(json.loads(raw)), [], str(p))
            scanned += 1
        self.assertGreater(scanned, 15)
        # and the planted values do reach the sealed exposed tables
        found = Counter()
        for part in P.PARTITIONS:
            for f in B.open_exposed(out, part, role="exposed_report"):
                for v in values(pq.read_table(f, use_threads=False).to_pylist()):
                    if isinstance(v, float) and not isinstance(v, bool):
                        found["float"] += v in floats
                    elif isinstance(v, str):
                        found["string"] += v in sent["strings"]
                    elif isinstance(v, int) and not isinstance(v, bool):
                        found["int"] += v in sent["ints"]
                found[part] += 1
        self.assertGreater(found["float"], 50)
        self.assertGreater(found["string"], 20)
        self.assertGreater(found["int"], 10)
        self.assertEqual(found["heldout"], 1)

    def test_exposed_and_lineage_tables(self):
        w = self.world()
        edit_json(w.paths["F1"]["summary"], lambda d: d.update(signal_admitted_improvements=d["signal_admitted_improvements"] + 1))
        out = self.built(w)
        exposed = {}
        for part in P.PARTITIONS:
            for f in B.open_exposed(out, part, role="exposed_report"):
                for r in pq.read_table(f, use_threads=False).to_pylist():
                    exposed[(r["context_sha256"], r["program_sha256"])] = r
        records = {(r["context_sha256"], r["program_sha256"]): r for r in self.all_rows(out, "records")}
        lineage = {(r["context_sha256"], r["program_sha256"]): r for r in self.all_rows(out, "lineage")}
        self.assertEqual(len(exposed), sum(sc.stats()["unique_programs"] for sc in w.contexts.values()))
        self.assertEqual(set(lineage), set(exposed))
        for name in ("F0", "I0", "F2"):
            sc = w.context(name)
            st = sc.stats()
            flags = fx.high_water(sc.native_rows, sc.best0)
            last_evo = {}
            for i, r in enumerate(sc.outcome_rows):
                if r.get("phase") == "evolution":
                    last_evo[r["program_sha256"]] = i
            with self.subTest(context=name):
                for k, row in enumerate(sc.native_rows):
                    p = row["program"]["program_sha256"]
                    e = exposed[(sc.cid, p)]
                    self.assertEqual(e["native_packed_line"], 1 + st["unique_programs"] + st["outcome_events"] + k)
                    self.assertEqual(e["native_line_sha256"], sha(sc.native_lines[k]))
                    self.assertEqual((e["history_ref_bucket"], e["history_ref_sha256"]), (fx.BUCKET, sc.history_sha256))
                    self.assertIs(e["new_signal_admitted_high_water"], flags[k])
                    self.assertEqual(e["result_key"], row.get("result_key"))
                    if row["status"] == "completed":
                        self.assertEqual((e["test_rmse"], e["fitness"]), (row["test_rmse"], row["fitness"]))
                        self.assertEqual(len(e["semantic_signature_f32"]), 64)
                        self.assertEqual(len(e["sampled_absolute_error_f32"]), 192)
                    i = last_evo.get(p)
                    if i is not None:
                        self.assertEqual(e["outcome_packed_line"], 1 + st["unique_programs"] + i)
                        self.assertEqual(e["outcome_line_sha256"], sha(sc.outcome_lines[i]))
                        o = sc.outcome_rows[i]
                        if o.get("result_ref"):
                            self.assertEqual(e["result_ref_key"], o["result_ref"]["key"])
                    lin = lineage[(sc.cid, p)]
                    self.assertEqual((lin["operator"], lin["parents"], lin["birth_seq"], lin["island_policy"]),
                                     (row["program"]["operator"], row["program"]["parents"], row["program"]["birth_seq"],
                                      row["policy"]))
                    self.assertEqual(lin["native_line"], k + 1)
                    if i is not None:
                        o = sc.outcome_rows[i]
                        self.assertEqual((lin["generation"], lin["attempt_id"], lin["node_raw"], lin["backend_field"],
                                          lin["outcome_line"]),
                                         (o["generation"], o.get("attempt_id"), o.get("node"), o.get("backend"), i + 1))
                for p, exp in sc.expect.items():
                    self.assertEqual(lineage[(sc.cid, p)]["backend_source"], exp["backend_source"], exp["kind"])
                    if (sc.cid, p) in records:
                        self.assertEqual(records[(sc.cid, p)]["backend"], exp["backend"], exp["kind"])
                    self.assertEqual(lineage[(sc.cid, p)]["origin"], exp["origin"])
            self.assertTrue(any(flags), name)
        # F1's summary count disagrees with the recompute: its flags are null and the disagreement is counted
        f1 = [e for (c, _p), e in exposed.items() if c == w.ids["F1"]]
        self.assertTrue(f1)
        self.assertTrue(all(e["new_signal_admitted_high_water"] is None for e in f1))
        counts = json.loads(B.manifest_path(out, "train_fit", "exposed").read_bytes())["counts"]
        self.assertEqual(counts["hw:mismatch"], 1)
        self.assertEqual(counts["hw:verified"], 3)
        # the backend audit: a local_lambda row whose attempt ran on an aws pool
        aws_attempt = [lin for lin in lineage.values() if lin["backend_source"] == "transport_attempt"
                       and lin["backend_field"] == "local_lambda" and any(x.startswith("aws:") for x in lin["pools"])]
        self.assertTrue(aws_attempt)
        self.assertTrue(all(lin["backend_field_agrees"] is False for lin in aws_attempt))
        # sealed files and the test-influenced lineage table carry no column statistics; the others do
        cur = self.cursor(out)
        for part in P.PARTITIONS:
            for t in B.PARQUET_TABLES:
                for sh in cur["writers"][part][t]["shards"]:
                    md = pq.ParquetFile(out / sh["path"]).metadata
                    stats_set = {md.row_group(g).column(j).is_stats_set for g in range(md.num_row_groups)
                                 for j in range(md.num_columns)}
                    self.assertEqual(stats_set, {B.has_statistics(part, t)}, sh["path"])
                    self.assertEqual(B.has_statistics(part, t), part != "heldout" and t not in ("exposed", "lineage"))
        # the lineage table opens only under lineage_report and its manifest says what it is (red-team LA-8)
        for part in ("train_fit", "dev", "diag"):
            lman = json.loads(B.manifest_path(out, part, "lineage").read_bytes())
            self.assertEqual((lman["role"], lman["run_plan_ref"], lman["write_statistics"]), ("lineage_report", False, False))
            self.assertIn("outer-test-derived", lman["note"])
            with self.assertRaises(PermissionError):
                B.open_lineage(out, part)
            for role in ("fit", "training_validation", "diagnostic_only", "exposed_report"):
                with self.assertRaises(PermissionError):
                    B.open_lineage(out, part, role)
            files = B.open_lineage(out, part, "lineage_report")
            self.assertEqual([str(f.relative_to(out)) for f in files],
                             [sh["path"] for sh in cur["writers"][part]["lineage"]["shards"]])
        with self.assertRaises(PermissionError):
            B.open_lineage(out, "heldout", "lineage_report")

    def test_heldout_role_guard(self):
        w = self.world()
        out = self.built(w)
        with self.assertRaises(PermissionError):
            B.open_partition(out, "heldout")
        with self.assertRaises(PermissionError):
            B.load_split(out, "heldout", "fit")
        # red-team LA-4: every partition opens only under its own read role
        for part in P.PARTITIONS:
            for role in B.ROLES:
                with self.subTest(partition=part, role=role):
                    if role == S.READ_ROLE[part]:
                        self.assertEqual(set(B.open_partition(out, part, role)), {"records", "contexts", "refused"})
                    else:
                        with self.assertRaises(PermissionError):
                            B.open_partition(out, part, role)
        for part in ("dev", "diag"):
            with self.assertRaises(PermissionError):
                B.load_split(out, part, "fit")
        with self.assertRaises(ValueError):
            B.open_partition(out, "train_fit", "lineage_report")
        # provenance (ctx-records): provenance_audit for train_fit/dev/diag, heldout_report for heldout
        self.assertTrue(B.open_provenance(out, "train_fit", "provenance_audit"))
        self.assertTrue(B.open_provenance(out, "heldout", "heldout_report"))
        for part, role in (("train_fit", None), ("train_fit", "fit"), ("dev", "heldout_report"),
                           ("heldout", "provenance_audit")):
            with self.assertRaises(PermissionError):
                B.open_provenance(out, part, role)
        with self.assertRaises(PermissionError):
            B.open_exposed(out, "train_fit")
        with self.assertRaises(PermissionError):
            B.open_exposed(out, "heldout", role="heldout_report")
        self.assertTrue(B.open_exposed(out, "heldout", role="exposed_report"))
        real_read_table = pq.read_table
        sealed_reads = []

        def guarded(source, *a, **kw):
            if S.is_sealed_path(source):
                sealed_reads.append(str(source))
                raise AssertionError(f"sealed file decoded without a role: {source}")
            return real_read_table(source, *a, **kw)

        with mock.patch.object(pq, "read_table", guarded):
            rep = B.verify(out, schema_allowlist=True, counts=True, rehash_sources=True, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        self.assertEqual(sealed_reads, [])
        self.assertGreater(rep["checked"]["heldout_footer_files"], 0)
        held = w.context("F2")
        rep2 = B.verify(out, schema_allowlist=True, counts=True, role="heldout_report", guard=ok_guard())
        self.assertEqual(rep2["verdict"], "PASS", rep2["failures"])
        self.assertEqual(rep2["checked"]["records_rows"] - rep["checked"]["records_rows"],
                         held.stats()["unique_programs"] - 2)
        self.assertEqual(rep2["checked"]["label_stats"], 4)
        c, r = B.load_split(out, "heldout", "heldout_report")
        self.assertEqual([x["context_sha256"] for x in c], [held.cid])
        # a doctored heldout shard is caught at footer level without decoding
        cur = self.cursor(out)
        target = out / cur["writers"]["heldout"]["records"]["shards"][0]["path"]
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"\0")
        with mock.patch.object(pq, "read_table", guarded):
            self.assertEqual(B.verify(out, guard=ok_guard())["verdict"], "FAIL")
        self.assertEqual(sealed_reads, [])

    def test_target_leak_exclusion(self):
        specs = (fx.CtxSpec("L0", "frozen", 0, "train_fit", "fam-l0", leak="exact"),
                 fx.CtxSpec("L1", "frozen", 1, "train_fit", "fam-l1", leak="affine"),
                 fx.CtxSpec("L2", "incoming", 2, "train_fit", "fam-l2", item="LA", leak="test_only"),
                 fx.CtxSpec("L3", "incoming", 3, "dev", "fam-l3", item="LB", leak="train_only"),
                 fx.CtxSpec("L4", "frozen", 2, "train_fit", "fam-l4"))
        w = self.world("leak", specs)
        out = self.built(w)
        recs = self.ctx_records(out)
        leaked = {w.ids[n] for n in ("L0", "L1", "L3")}
        for cid in leaked:
            r = recs[cid]
            self.assertEqual((r["excluded"], r["records"], r["refused"]), ("target_leak", 0, 0))
            self.assertTrue(r["leak"]["target_leak"])
            self.assertNotIn("label", r)
        self.assertEqual(recs[w.ids["L0"]]["leak"]["exact"], [4])
        self.assertTrue(recs[w.ids["L0"]]["fold0_construction_contradiction"])
        self.assertFalse(recs[w.ids["L1"]]["fold0_construction_contradiction"])
        self.assertEqual([i for i, _r in recs[w.ids["L1"]]["leak"]["affine"]], [4])
        self.assertEqual(recs[w.ids["L1"]]["leak"]["exact"], [])
        for name in ("L2", "L4"):
            r = recs[w.ids[name]]
            self.assertIsNone(r["excluded"])
            self.assertFalse(r["leak"]["target_leak"])
            self.assertGreater(r["records"], 0)
            self.assertIn("label", r)
        for table in ("records", "contexts", "refused", "lineage"):
            self.assertFalse({r["context_sha256"] for r in self.all_rows(out, table)} & leaked, table)
        for part in P.PARTITIONS:
            for f in B.open_exposed(out, part, role="exposed_report"):
                self.assertFalse(set(pq.read_table(f, columns=["context_sha256"]).column(0).to_pylist()) & leaked)
        man = json.loads((out / "manifest.json").read_bytes())
        self.assertEqual(man["exclusions"]["target_leak"], 3)
        self.assertEqual(man["exclusions"]["fold0_construction_contradiction"], 1)
        self.assertEqual(man["tokenizer_closure"]["relation"], "subset_unverified_without_program_dict")
        stats = json.loads((out / "train" / "label-stats.json").read_bytes())
        self.assertEqual(stats["contexts"], 2)
        self.assertEqual(B.verify(out, guard=ok_guard())["verdict"], "PASS")

    def test_reconciliation_stops(self):
        def bump(name, key, by=1):
            return lambda w: edit_json(w.paths[name]["summary"], lambda d: d.update({key: d[key] + by}))

        def stray_outcome(w):
            def add(rows):
                tmpl = next(r for r in rows if r.get("phase") == "evolution")
                dsl = "(pipe X (norm_row) (ridge lambda_frac=0.4242))"
                rows.append({**tmpl, "program": fx.program_obj(dsl), "program_sha256": fx.sha(dsl)})
            edit_lines(w.paths["F1"]["outcome"], add)
            repin(w, "F1")

        def seed_pool(w):
            dsl = "(pipe X (robust_scale) (ridge lambda_frac=0.999))"
            with open(w.paths["F1"]["seed_pool"], "ab") as f:
                f.write(fx.dump({"program": fx.program_obj(dsl, operator="historical_seed"), "schema": "jope.recovery-seed/v1"}))

        def cv_disagree(w):
            k, _i, _p = first_detailed_completed(w.context("F1"))
            edit_lines(w.paths["F1"]["native"], lambda rows: rows[k].update(cv_mean_log_ratio=rows[k]["cv_mean_log_ratio"] + 1e-3))
            repin(w, "F1")

        def baseline(w):
            k, i, _p = first_detailed_completed(w.context("F1"))
            sd = {}

            def change(rows):
                block = rows[i]["evaluation_detail"]["inner_cv"][0]["baseline_metrics"]
                block["rmse"] = block["rmse"] * 1.5
                # both writers' cv_log_ratio_sd follow the changed detail, so the baseline disagreement is what STOPs
                rmse, base = S.inner_cv_pairs(rows[i]["evaluation_detail"])
                sd["v"] = rows[i]["cv_log_ratio_sd"] = S.canonical_cv_log_ratio_sd(rmse, base)
            edit_lines(w.paths["F1"]["outcome"], change)
            edit_lines(w.paths["F1"]["native"], lambda rows: rows[k].update(cv_log_ratio_sd=sd["v"]))
            repin(w, "F1")

        def stale_sd(w):
            _k, i, _p = first_detailed_completed(w.context("F1"))

            def change(rows):                   # a detail change the writers' cv_log_ratio_sd does not follow
                block = rows[i]["evaluation_detail"]["inner_cv"][1]["metrics"]
                block["rmse"] = block["rmse"] * 1.5
            edit_lines(w.paths["F1"]["outcome"], change)
            repin(w, "F1")

        def gate(w):
            k, i, _p = first_detailed_completed(w.context("F1"))
            edit_lines(w.paths["F1"]["native"], lambda rows: rows[k].update(cv_gate_passed=not rows[k]["cv_gate_passed"]))
            edit_lines(w.paths["F1"]["outcome"], lambda rows: rows[i].update(cv_gate_passed=not rows[i]["cv_gate_passed"]))
            repin(w, "F1")

        def prepared(key, value):
            def change(w):
                edit_json(w.paths["F1"]["prepared"], lambda d: d["context"].update({key: value}))
                repin(w, "F1")
            return change

        def nonfinite(w):
            sc = w.context("F1")
            X, y = fx.jtt1_matrix(sc.spec.n_train, sc.spec.n_test, sc.spec.p, tag="jtt1:F1")
            y[3] = math.nan
            set_input(w, "F1", fx.jtt1_raw(X, y, sc.spec.n_train))

        def raw_program(w):
            sc = w.context("F3")
            p = next(p for p, e in sc.expect.items() if e["origin"] == "transport_only")
            dsl = sc.dsl[p] + " "

            def change(rows):
                for r in rows:
                    if r.get("program_sha256") == p:
                        r["program"] = {**r["program"], "dsl": dsl, "program_sha256": fx.sha(dsl)}
                        r["program_sha256"] = fx.sha(dsl)
            edit_lines(w.paths["F3"]["transport"], change)
            repin(w, "F3")

        def max_seq(w):
            edit_json(w.tok_dir / "lengths.json", lambda d: d.update(max_seq=16))

        def fingerprint(w):
            def change(doc):
                fp = doc[w.ids["I0"]]["fingerprint"]
                doc[w.ids["I0"]]["fingerprint"] = fp[:-1] + ("0" if fp[-1] != "0" else "1")
            edit_json(w.sources.incoming_cursor, change)

        def reuse_other_input(w):
            _k, i, _p = first_detailed_completed(w.context("F1"))
            sc = w.context("F1")

            def change(rows):
                rows[i]["evaluation_reuse"] = {
                    "schema": S.REUSE_SCHEMA, "source_context_sha256": "4" * 64,
                    "compatibility_identity": {"input_raw_sha256": "0" * 64, "evaluator_sha256": S.EVALUATOR_SHA256,
                                               "candidate_timeout_ms": 90000, "train_rows": sc.spec.n_train,
                                               "encoded_features": sc.spec.p, "test_rows": sc.spec.n_test}}
            edit_lines(w.paths["F1"]["outcome"], change)
            repin(w, "F1")

        def sd_two_ulp(w):
            k, _i, _p = first_detailed_completed(w.context("F1"))
            edit_lines(w.paths["F1"]["native"], lambda rows: rows[k].update(
                cv_log_ratio_sd=math.nextafter(math.nextafter(rows[k]["cv_log_ratio_sd"], 1.0), 1.0)))
            repin(w, "F1")

        cases = [  # (label, tamper, run kwargs, expected kind, context name)
            ("native lines != native_events", bump("F1", "native_events"), {}, "count", "F1"),
            ("an outcome program neither native nor bootstrap", stray_outcome, {}, "outcome_not_native", "F1"),
            ("unique_programs off by one", bump("F1", "unique_programs"), {}, "count", "F1"),
            ("native completed != expected", None, {"expected_delta": "F1"}, "native_completed", "F1"),
            ("seed-pool set != bootstrap set", seed_pool, {}, "seed_pool", "F1"),
            ("native/outcome cv disagreement", cv_disagree, {}, "native_outcome_disagreement", "F1"),
            ("baseline disagreement (LabelError)", baseline, {}, "label", "F1"),
            ("a gate disagreement off the 1.05 boundary", gate, {}, "gate_drift", "F1"),
            ("candidate_timeout_ms != 90000", prepared("candidate_timeout_ms", 60000), {}, "timeout_pin", "F1"),
            ("evaluator sha256 != pin", prepared("evaluator_sha256", "0" * 64), {}, "evaluator_pin", "F1"),
            ("JTT1 n_train != prepared train_rows", prepared("train_rows", 49), {}, "jtt1_shape", "F1"),
            ("non-finite y_train", nonfinite, {}, "nonfinite_target", "F1"),
            ("a RAW program", raw_program, {}, "raw_program", "F3"),
            ("a train_fit program over max_seq", max_seq, {}, "max_seq", None),
            ("an incoming fingerprint mismatch", fingerprint, {}, "source", "I0"),
            ("an evaluation reuse of another input", reuse_other_input, {}, "evaluation_reuse", "F1"),
            ("cv_log_ratio_sd 2 ULP apart", sd_two_ulp, {}, "native_outcome_disagreement", "F1"),
            ("a writer cv_log_ratio_sd stale against its detail", stale_sd, {}, "cv_sd_recompute", "F1"),
        ]
        for i, (label, tamper, kw, kind, name) in enumerate(cases):
            with self.subTest(label):
                w = self.world(f"r{i}")
                if tamper is not None:
                    tamper(w)
                if kw.pop("expected_delta", None):
                    kw["expected_completed"] = {**w.expected_completed, w.ids["F1"]: w.expected_completed[w.ids["F1"]] + 1}
                got, ctx = self.stop_of(lambda: self.build(w, f"out-{i}", **kw))
                self.assertEqual(got, kind)
                if name is not None:
                    self.assertEqual(ctx, w.ids[name])
                out = self.tmp / f"out-{i}"
                cur = self.cursor(out)
                self.assertEqual(cur["status"], "STOPPED")
                self.assertEqual(cur["stop"]["kind"], kind)
                recorded = {sh["path"] for p in P.PARTITIONS for t in B.TABLES for sh in cur["writers"][p][t]["shards"]}
                for f in out.rglob("*.parquet"):
                    rel = str(f.relative_to(out))
                    self.assertIn(rel, recorded)          # no partial or unrecorded shard
                    if ctx is not None:
                        cids = set(pq.read_table(f, columns=["context_sha256"]).column(0).to_pylist())
                        self.assertNotIn(ctx, cids)
                self.assertFalse((out / "manifest.json").exists())

    def test_refused_over_max_seq(self):
        w = self.world()
        out = self.built(w)
        cur = self.cursor(out)
        self.assertEqual(cur["writers"]["train_fit"]["refused"]["shards"], [])
        self.assertFalse((out / "train" / "refused").exists())
        recs = self.ctx_records(out)
        for name, part, n in (("I1", "dev", 1), ("F2", "heldout", 2)):
            sc = w.context(name)
            longs = {p for p, e in sc.expect.items() if e.get("long")}
            self.assertEqual(len(longs), n)
            rows = []
            for sh in cur["writers"][part]["refused"]["shards"]:
                rows += pq.read_table(out / sh["path"], use_threads=False).to_pylist()
            with self.subTest(context=name):
                self.assertEqual({r["program_sha256"] for r in rows if r["context_sha256"] == sc.cid}, longs)
                for r in rows:
                    self.assertEqual((r["reason"], r["split_role"]), ("over_max_seq", part))
                    self.assertGreater(r["length"], w.max_seq)
                rec = recs[sc.cid]
                self.assertEqual(rec["refused"], n)
                self.assertEqual(rec["records"] + rec["refused"], sc.stats()["unique_programs"])
                self.assertEqual(rec["counts"]["refused"], n)
                self.assertEqual(sum(rec["counts"]["refused_by_outcome"].values()), n)
                man = json.loads(B.manifest_path(out, part).read_bytes())
                self.assertEqual(man["counts"]["refused"], n)
        dev_records = B.load_split(out, "dev", "training_validation")[1]
        self.assertFalse({r["program_sha256"] for r in dev_records} & {p for p, e in w.context("I1").expect.items()
                                                                        if e.get("long")})
        self.assertTrue(all(r["length"] <= w.max_seq for r in dev_records))

    def test_shards_deterministic_and_sized(self):
        w = self.world()
        a = self.built(w, "a", target_shard_bytes=TARGET)
        b = self.built(w, "b", target_shard_bytes=TARGET)
        fa, fb = output_files(a), output_files(b)
        self.assertEqual(set(fa), set(fb))
        self.assertEqual(fa, fb)
        self.assertTrue(any(k.endswith("label-stats.json") for k in fa))
        cur = self.cursor(a)
        for t in ("records", "exposed", "lineage"):
            self.assertGreaterEqual(len(cur["writers"]["train_fit"][t]["shards"]), 2, t)
        for p in P.PARTITIONS:
            for t in B.PARQUET_TABLES:
                shards = cur["writers"][p][t]["shards"]
                sealed = B.is_sealed(p, t)
                seen, prev = set(), None
                for j, sh in enumerate(shards):
                    path = a / sh["path"]
                    self.assertEqual(S.check_parquet(path, B.table_schema(t), not sealed,
                                                     sealed=not B.has_statistics(p, t), json_sample=1), [], sh["path"])
                    md = pq.ParquetFile(path).metadata
                    self.assertEqual(set(md.metadata.keys()), {b"ARROW:schema"})
                    self.assertTrue(all(md.row_group(g).num_rows <= 65536 for g in range(md.num_row_groups)))
                    if j < len(shards) - 1:
                        self.assertGreaterEqual(sh["arrow_nbytes"], TARGET // 2, sh["path"])
                        self.assertLessEqual(sh["arrow_nbytes"], 2 * TARGET, sh["path"])
                    cols = ["context_sha256", "program_sha256"] if "program_sha256" in B.table_schema(t).names \
                        else ["context_sha256"]
                    tbl = pq.read_table(path, columns=cols, use_threads=False)
                    keys = list(zip(*(tbl.column(c).to_pylist() for c in cols)))
                    self.assertEqual(keys, sorted(keys))
                    if prev is not None and keys:
                        self.assertLess(prev, keys[0])
                    if keys:
                        prev = keys[-1]
                    ctxs = {k[0] for k in keys}
                    self.assertFalse(ctxs & seen, f"a context split across shards in {p}/{t}")
                    seen |= ctxs
                    self.assertEqual((keys[0][0], keys[-1][0]) if keys else None,
                                     (sh["first_context"], sh["last_context"]) if keys else None)

    def test_budget_pause_resume(self):
        w = self.world()
        ref = self.built(w, "ref")
        refs = {e["context_sha256"]: e["input_ref"] for e in w.epoch_doc["contexts"] if e.get("input_ref")}
        store = {r["key"]: (w.jtt1_cache / f"{r['raw_sha256']}.jtt1.gz").read_bytes() for r in refs.values()}
        gets = Counter()

        def fake_get(client, key, version_id, bucket=None):
            gets[key] += 1
            for name, handler in client.meta.events.handlers:
                if name == "before-send.s3.GetObject":
                    handler()
            return store[key], {"VersionId": version_id}

        order = [c.context_sha256 for c in P._plan(w.epoch, w.pmap).contexts]
        k = 3
        budget = sum(refs[c]["bytes"] for c in order[:k])
        cache = self.tmp / "cache"
        factory = mock.Mock(side_effect=FakeClient)
        with mock.patch("dltrain.s3io.get_exact", fake_get):
            r1 = self.build(w, "run", jtt1_cache=cache, budget_bytes=budget, offline=False, client_factory=factory)
            self.assertEqual((r1["status"], r1["exit"]), ("PAUSED_BUDGET", B.EXIT_PAUSED), r1)
            led = self.cursor(self.tmp / "run")["ledger"]
            self.assertEqual((led["calls"], led["http_attempts"], led["bytes"]), (k, k, budget))
            self.assertEqual(len(list(cache.iterdir())), k)
            self.assertEqual(self.cursor(self.tmp / "run")["status"], "PAUSED_BUDGET")
            self.assertFalse((self.tmp / "run" / "train").exists())
            r2 = self.build(w, "run", jtt1_cache=cache, budget_bytes=sum(r["bytes"] for r in refs.values()),
                            offline=False, client_factory=factory)
            self.assertEqual(r2["status"], "COMPLETE", r2)
        self.assertEqual(set(gets.values()), {1})                 # no input is fetched twice
        self.assertEqual(len(gets), len(order))
        led = self.cursor(self.tmp / "run")["ledger"]
        self.assertEqual((led["calls"], led["http_attempts"]), (len(order), len(order)))
        self.assertEqual(len(led["invocations"]), 2)
        self.assertEqual(output_files(self.tmp / "run"), output_files(ref))
        # --max-gets: a call is admitted only while calls + 1 and http_attempts + 3 stay within it
        with mock.patch("dltrain.s3io.get_exact", fake_get):
            r3 = self.build(w, "max-gets", jtt1_cache=self.tmp / "cache3", budget_bytes=10**9, offline=False,
                            max_gets=4, client_factory=FakeClient)
        self.assertEqual(r3["status"], "PAUSED_BUDGET")
        self.assertEqual(self.cursor(self.tmp / "max-gets")["ledger"]["calls"], 2)
        # --offline: a cache miss is a STOP and no client is ever created
        never = mock.Mock(side_effect=AssertionError("no client in offline mode"))
        kind, _ctx = self.stop_of(lambda: self.build(w, "offline", jtt1_cache=self.tmp / "cache4", offline=True,
                                                     client_factory=never))
        self.assertEqual(kind, "jtt1_missing")
        never.assert_not_called()

    def test_crash_resume_and_tamper(self):
        w = self.world()
        ref = self.built(w, "ref", target_shard_bytes=TARGET)
        real_publish = P._publish_immutable

        def crashing(after: int):
            n = {"shards": 0}

            def publish(path, data):
                res = real_publish(path, data)
                if str(path).endswith(".parquet"):
                    n["shards"] += 1
                    if n["shards"] == after:
                        raise RuntimeError("simulated crash after a shard write, before the cursor records it")
                return res
            return mock.patch.object(P, "_publish_immutable", publish)

        with crashing(3), self.assertRaises(RuntimeError):
            self.build(w, "run", target_shard_bytes=TARGET)
        run = self.tmp / "run"
        recorded = {sh["path"] for p in P.PARTITIONS for t in B.TABLES
                    for sh in self.cursor(run)["writers"][p][t]["shards"]}
        orphans = {str(f.relative_to(run)) for f in run.rglob("*.parquet")} - recorded
        self.assertEqual(len(orphans), 1)
        res = self.build(w, "run", target_shard_bytes=TARGET)
        self.assertEqual(res["status"], "COMPLETE", res)
        adopted = [sh["path"] for p in P.PARTITIONS for t in B.TABLES
                   for sh in self.cursor(run)["writers"][p][t]["shards"] if sh["adopted"]]
        self.assertEqual(set(adopted), orphans)
        self.assertEqual(output_files(run), output_files(ref))
        # a tampered completed shard: the resume refuses (CONFLICT)
        with crashing(4), self.assertRaises(RuntimeError):
            self.build(w, "run2", target_shard_bytes=TARGET)
        cur2 = self.cursor(self.tmp / "run2")
        done = next(sh for p in P.PARTITIONS for t in B.TABLES for sh in cur2["writers"][p][t]["shards"])
        target = self.tmp / "run2" / done["path"]
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"\0")
        with self.assertRaises(common.ImmutableConflict):
            self.build(w, "run2", target_shard_bytes=TARGET)
        # an orphan with other bytes: the re-flush is a CONFLICT (exit 4), never an overwrite
        with crashing(2), self.assertRaises(RuntimeError):
            self.build(w, "run3", target_shard_bytes=TARGET)
        run3 = self.tmp / "run3"
        rec3 = {sh["path"] for p in P.PARTITIONS for t in B.TABLES for sh in self.cursor(run3)["writers"][p][t]["shards"]}
        orphan = next(f for f in run3.rglob("*.parquet") if str(f.relative_to(run3)) not in rec3)
        orphan.chmod(0o644)
        orphan.write_bytes(orphan.read_bytes()[:-1] + b"\1")
        res3 = self.build(w, "run3", target_shard_bytes=TARGET)
        self.assertEqual((res3["status"], res3["exit"]), ("STOPPED", B.EXIT_CONFLICT), res3)
        # any changed pin refuses the resume of a finished build
        seeds2 = self.tmp / "seeds2.json"
        seeds2.write_bytes(json.dumps(sorted(w.seed_set | {fx.sha("(pipe X (other))")})).encode())
        map2 = self.tmp / "map2.json"
        pmap = json.loads(w.pmap.read_bytes())
        pmap.pop(fx.sha("ctx:not-in-epoch"))
        map2.write_bytes(json.dumps(pmap).encode())
        epoch2 = self.tmp / "epoch2.json"
        epoch2.write_bytes(json.dumps({**json.loads(w.epoch.read_bytes()), "note": "changed"}).encode())
        tok2 = self.tmp / "tok2"
        shutil.copytree(w.tok_dir, tok2)
        edit_json(tok2 / "lengths.json", lambda d: d.update(max_seq=d["max_seq"] + 32))
        changes = {"settings": {"target_shard_bytes": 2 * TARGET}, "seeds": {"seeds": seeds2},
                   "partition map": {"partition_map": map2},
                   "epoch": {"epoch": epoch2, "epoch_sha256": sha(epoch2.read_bytes())},
                   "tokenizer": {"tokenizer": tok2}, "expected completed": {"expected_completed": {
                       **w.expected_completed, fx.sha("ctx:other"): 1}}}
        for label, kw in changes.items():
            with self.subTest(pin=label):
                kind, _ctx = self.stop_of(lambda: self.build(w, "run", **{"target_shard_bytes": TARGET, **kw}))
                self.assertEqual(kind, "cursor")
        self.assertEqual(self.build(w, "run", target_shard_bytes=TARGET)["status"], "COMPLETE")

    def test_runtime_report(self):
        w = self.world()
        out = self.built(w)
        rr = json.loads((out / "runtime-report.json").read_bytes())
        self.assertEqual(rr["partitions"], ["train_fit", "dev", "diag"])
        self.assertNotIn("heldout", {g["partition"] for g in rr["groups"]})
        expect_backend = {(sc.cid, p): e["backend"] for sc in w.contexts.values() for p, e in sc.expect.items()}
        groups: dict = defaultdict(lambda: {"counts": Counter(), "completed": [], "timeout": []})
        for p in ("train_fit", "dev", "diag"):
            ctxs, recs = B.load_split(out, p, S.READ_ROLE[p])
            cohort = {c["context_sha256"]: c["cohort"] for c in ctxs}
            for r in recs:
                self.assertEqual(r["backend"], expect_backend[(r["context_sha256"], r["program_sha256"])])
                g = groups[(r["backend"], p, cohort[r["context_sha256"]])]
                o = r["outcome"]
                cls = ("completed" if o == "completed" else "failed" if o.startswith("failed:") else
                       "timeout" if o == "timeout@90000ms" else "abandoned" if o.startswith("abandoned:") else "missing")
                g["counts"][cls] += 1
                g["counts"]["records"] += 1
                if cls in ("completed", "timeout") and r["wall_us"] is not None:
                    g[cls].append(r["wall_us"])
        self.assertEqual({(g["backend"], g["partition"], g["cohort"]) for g in rr["groups"]}, set(groups))
        self.assertEqual({g["backend"] for g in rr["groups"]}, {"local", "aws_lambda", "unknown"})
        for g in rr["groups"]:
            mine = groups[(g["backend"], g["partition"], g["cohort"])]
            c = mine["counts"]
            with self.subTest(group=(g["backend"], g["partition"], g["cohort"])):
                for k in ("records", "completed", "failed", "timeout", "abandoned", "missing"):
                    self.assertEqual(g[k], c[k], k)
                denom = c["completed"] + c["failed"] + c["timeout"]
                self.assertEqual(g["timeout_rate"], c["timeout"] / denom if denom else None)
                for cls in ("completed", "timeout"):
                    vals = sorted(mine[cls])
                    want = {"n": len(vals), "max": vals[-1] if vals else None}
                    for name, q in (("p10", "0.1"), ("p50", "0.5"), ("p90", "0.9"), ("p99", "0.99")):
                        want[name] = vals[max(1, math.ceil(Fraction(q) * len(vals))) - 1] if vals else None
                    self.assertEqual(g[f"wall_us_{cls}"], want)
        timeouts = [g for g in rr["groups"] if g["timeout"]]
        self.assertTrue(timeouts)
        self.assertTrue(all(g["wall_us_timeout"]["max"] >= 90_000_000 for g in timeouts))

    def test_incoming_index_matches_programs_demux(self):
        w = self.world()
        cursor = json.loads(w.sources.incoming_cursor.read_bytes())
        i0, i2 = w.context("I0"), w.context("I2")
        path = w.paths["I0"]["outcome"]
        self.assertEqual(path, w.paths["I2"]["outcome"])
        wanted = {sc.cid: cursor[sc.cid]["stats"]["outcome_bytes"] for sc in (i0, i2)}
        idx = B.ItemIndex(path, wanted)
        ctxs = {c.context_sha256: c for c in P.load_contexts(w.epoch, w.pmap)}
        for sc in (i0, i2):
            lines = list(idx.lines(sc.cid))
            self.assertEqual(lines, sc.outcome_lines)
            c = ctxs[sc.cid]
            s = P.history_summary(c, cursor)
            ev = P._incoming_evidence([(c, s)], P._complete_lines(path), path, w.sources)[sc.cid]
            self.assertNotIsInstance(ev, P.ReconcileError)
            rec = ev.prefixes["outcome"]
            self.assertEqual((rec["bytes"], rec["lines"], rec["sha256"]),
                             (sum(map(len, lines)), len(lines), sha(b"".join(lines))))
            self.assertEqual(sha(sc.native_bytes + b"".join(lines) + sc.transport_bytes + sc.boot_results_bytes),
                             cursor[sc.cid]["fingerprint"])
        self.assertGreater(idx.partial_tail, 0)                  # the live, unterminated last line
        self.assertGreater(idx.scanned_lines, idx.kept_lines)    # interleaved stranger contexts
        self.assertEqual(idx.kept_lines, len(i0.outcome_lines) + len(i2.outcome_lines))
        self.assertFalse(idx.release(i0.cid))
        self.assertTrue(idx.release(i2.cid))
        # a shorter recorded prefix that ends inside a line is a STOP; one ending on a boundary leaves a tail
        bad = B.ItemIndex(path, {i0.cid: wanted[i0.cid] - 1})
        with self.assertRaises(S.CorpusStop) as cm:
            list(bad.lines(i0.cid))
        self.assertEqual((cm.exception.kind, cm.exception.context), ("source", i0.cid))
        short = B.ItemIndex(path, {i0.cid: wanted[i0.cid] - len(i0.outcome_lines[-1])})
        self.assertEqual(list(short.lines(i0.cid)), i0.outcome_lines[:-1])
        self.assertEqual(short.tail[i0.cid], 1)
        # a changed cursor entry of a pending context after a pause is a STOP
        res = self.build(w, "paused", local_quota_bytes=1)
        self.assertEqual((res["status"], res["exit"]), ("PAUSED_QUOTA", B.EXIT_PAUSED), res)
        self.assertEqual(len(self.cursor(self.tmp / "paused")["incoming_cursor_reads"]), 1)
        i1 = w.context("I1")

        def change(doc):
            fp = doc[i1.cid]["fingerprint"]
            doc[i1.cid]["fingerprint"] = fp[:-1] + ("0" if fp[-1] != "0" else "1")
        edit_json(w.sources.incoming_cursor, change)
        kind, ctx = self.stop_of(lambda: self.build(w, "paused"))
        self.assertEqual((kind, ctx), ("source", i1.cid))

    def test_dictionary_crosscheck(self):
        w = self.world()
        dict_dir = self.tmp / "dict"
        P.build(w.epoch, w.pmap, w.seeds, dict_dir, sources=w.sources)
        with_dict = self.built(w, "with-dict", program_dict=dict_dir)
        without = self.built(w, "without")
        for out, how in ((with_dict, "program_dict"), (without, "context_rows")):
            for r in self.ctx_records(out).values():
                self.assertEqual((r["dict_crosscheck"]["source"], r["dict_crosscheck"]["equal"]), (how, True))
        self.assertEqual(parquet_files(with_dict), parquet_files(without))
        self.assertEqual(json.loads((with_dict / "manifest.json").read_bytes())["tokenizer_closure"]["relation"], "equal")
        # R7 reads origin and the exact record_status from P4a's sealed side table (programs.SealedIndex, role
        # r7_crosscheck); the published chunks carry neither. A doctored sealed row is STOP(dict_mismatch).
        journal = dict_dir / ".access-journal.jsonl"
        self.assertTrue(journal.exists())
        self.assertIn("r7_crosscheck", {json.loads(x)["role"] for x in journal.read_bytes().splitlines()})
        doctored = self.tmp / "dict-doctored"
        shutil.copytree(dict_dir, doctored)
        man = json.loads((doctored / "manifest.json").read_bytes())
        chunk = man["sealed"]["partitions"]["train_fit"]["chunks"][0]
        f = doctored / chunk["path"]
        t = pq.read_table(f, use_threads=False)
        statuses = t.column("record_status").to_pylist()
        statuses[statuses.index("completed")] = "failed"
        values_ = P.SEALED_DICT_COLUMNS["record_status"]
        arr = pa.DictionaryArray.from_arrays(pa.array([values_.index(s) for s in statuses], type=pa.int8()),
                                             pa.array(values_, type=pa.string()))
        t = t.set_column(t.schema.get_field_index("record_status"), P.SEALED_SCHEMA.field("record_status"), arr)
        f.chmod(0o644)
        pq.write_table(t, f, compression="zstd")
        chunk.update(sha256=common.sha256_file(f), bytes=f.stat().st_size)
        (doctored / "manifest.json").chmod(0o644)
        (doctored / "manifest.json").write_bytes(common.canonical(man) + b"\n")
        kind, ctx = self.stop_of(lambda: self.build(w, "doctored", program_dict=doctored))
        self.assertEqual(kind, "dict_mismatch")
        self.assertIn(ctx, set(chunk["contexts"]))
        # a dictionary of another epoch is refused up front
        other = self.tmp / "dict-other"
        shutil.copytree(dict_dir, other)
        (other / "manifest.json").chmod(0o644)
        edit_json(other / "manifest.json", lambda d: d["inputs"]["epoch"].update(sha256="0" * 64))
        kind, _ctx = self.stop_of(lambda: self.build(w, "other", program_dict=other))
        self.assertEqual(kind, "input")
        # red-team CL-7: so is a dictionary built from another seed list (its shared_seed column would disagree)
        seeds2 = self.tmp / "seeds-other.json"
        seeds2.write_bytes(json.dumps(sorted(w.seed_set | {fx.sha("(pipe X (other))")})).encode())
        dict2 = self.tmp / "dict-seeds2"
        P.build(w.epoch, w.pmap, seeds2, dict2, sources=w.sources)
        kind, _ctx = self.stop_of(lambda: self.build(w, "seeds2", program_dict=dict2))
        self.assertEqual(kind, "input")
        self.assertEqual(self.build(w, "seeds2-ok", program_dict=dict2, seeds=seeds2)["status"], "COMPLETE")
        # the tokenizer closure becomes a recorded subset when a train_fit context is excluded as target_leak
        specs = (fx.CtxSpec("K0", "frozen", 1, "train_fit", "fam-k0", leak="exact"),
                 fx.CtxSpec("K1", "frozen", 2, "train_fit", "fam-k1"),
                 fx.CtxSpec("K2", "incoming", 3, "dev", "fam-k2", item="KA"))
        k = self.world("leak", specs)
        kdict = self.tmp / "kdict"
        P.build(k.epoch, k.pmap, k.seeds, kdict, sources=k.sources)
        out = self.built(k, "leak-out", program_dict=kdict)
        closure = json.loads((out / "manifest.json").read_bytes())["tokenizer_closure"]
        self.assertEqual(closure["relation"], "subset")
        own = {p for p in k.context("K0").dsl} - {p for p in k.context("K1").dsl}
        self.assertEqual(closure["difference"], len(own))
        self.assertEqual(closure["vocab_program_set_sha256"], k.train_program_set_sha256)


class VerifyPoisonPublish(BuildCase):
    """verify --poison-test end to end (spec §3.10) on the World, its sensitivity controls, the verify receipt and the
    gated publish planner (which must never reach an S3 write path)."""

    def test_verify_runs_poison_test_with_controls(self):
        w = self.world()
        out = self.built(w)
        receipt = self.tmp / "receipts" / "verify.json"
        rep = B.verify(out, schema_allowlist=True, counts=True, poison_test_run=True, poison_contexts=2,
                       poison_work=self.tmp / "poison", receipt=receipt, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        poison = rep["poison"]
        self.assertEqual((poison["schema"], poison["verdict"]), (B.POISON_SCHEMA, "PASS"), poison.get("failures"))
        self.assertTrue(poison["train_identical"])
        self.assertEqual(poison["controls"], {"exposed_differ": True, "nontrain_differ": True, "jtt1_differ": True})
        self.assertEqual(poison["neutrality"], {"records_equal": True, "contexts_equal": True})
        picked = {(x["cohort"], x["split_role"]) for x in poison["selected"]}
        # red-team LA-1: every non-empty non-train partition is perturbed (diag F3 shares fam-b and its dataset with
        # train_fit F1)
        self.assertEqual(picked, {("frozen", "train_fit"), ("frozen", "heldout"), ("frozen", "diag"),
                                  ("incoming", "train_fit"), ("incoming", "dev")})
        self.assertEqual((poison["nonempty_partitions"], poison["perturbed_partitions"]),
                         (["heldout", "dev", "diag"], ["heldout", "dev", "diag"]))
        self.assertEqual(sorted(poison["train_files"]), ["train/contexts/part-00000.parquet", "train/label-stats.json",
                                                         "train/manifest.json", "train/records/part-00000.parquet",
                                                         "train/seed-stats.json"])
        self.assertTrue(all(v["A"] == v["B"] for v in poison["train_files"].values()))
        self.assertEqual(poison["train_closure"]["A"], poison["train_closure"]["B"])
        self.assertTrue(poison["runtime_train_identical"])
        doc = json.loads(receipt.read_bytes())
        self.assertEqual((doc["schema"], doc["verdict"]), (B.VERIFY_SCHEMA, "PASS"))
        with self.assertRaises(common.ImmutableConflict):          # receipts are create-only
            B.verify(out, receipt=receipt, guard=ok_guard(), schema_allowlist=False)
        for sens in ("train_rmse", "leak_dev"):                    # tests_p5.test_corpus_poison runs every control
            with self.subTest(sensitivity=sens):
                r = B.poison_test(out, self.tmp / f"poison-{sens}", sensitivity=sens, guard=ok_guard())
                self.assertEqual(r["verdict"], "FAIL")
                self.assertEqual({k: v["status"] for k, v in r["builds"].items()}, {"A": "COMPLETE", "B": "COMPLETE"})
                self.assertFalse(r["train_identical"])
                self.assertEqual(r["failures"], ["train outputs differ between A and B"])
        # publish is gated: it plans the create-only objects and STOPs; no S3 client is ever requested
        g18 = self.tmp / "g18.json"
        g18.write_bytes(json.dumps({"verdict": "PASS"}).encode())
        with mock.patch("dltrain.s3io.make_client", side_effect=AssertionError("publish must not open S3")) as mk:
            with self.assertRaises(S.CorpusStop) as cm:
                B.publish(out=out, verify_receipt=receipt, g18_receipt=g18, s3_cap_bytes=10**12, guard=ok_guard())
            self.assertEqual(cm.exception.kind, "gate")
            self.assertIn("not enabled", str(cm.exception))
            plan = cm.exception.plan
            self.assertEqual(plan["objects"], len(plan["plan"]))
            # red-team LA-10: sealed files never share the training shards' object kind or manifest family
            self.assertTrue(all(o["key"].startswith(tuple(B.PUBLISH_PREFIXES.values())
                                                    + (B.PUBLISH_MANIFEST_PREFIX, B.PUBLISH_SEALED_MANIFEST_PREFIX))
                                for o in plan["plan"]))
            for o in plan["plan"]:
                if S.is_sealed_path(o["source"]):
                    self.assertTrue(o["key"].startswith((B.PUBLISH_PREFIXES["sealed"], B.PUBLISH_SEALED_MANIFEST_PREFIX)),
                                    o)
                if o["key"].startswith(B.PUBLISH_PREFIXES["shard"]):
                    self.assertTrue(o["source"].startswith(("train/", "eval/")), o)
            self.assertTrue(plan["within_cap"])
            with self.assertRaises(S.CorpusStop) as cm:          # the plan must also fit the DL-D7 S3 cap
                B.publish(out=out, verify_receipt=receipt, g18_receipt=g18, s3_cap_bytes=1, guard=ok_guard())
            self.assertIn("exceeds", str(cm.exception))
            failed = self.tmp / "verify-fail.json"
            failed.write_bytes(json.dumps({**doc, "poison": {"verdict": "FAIL"}}).encode())
            with self.assertRaises(S.CorpusStop) as cm:
                B.publish(out=out, verify_receipt=failed, g18_receipt=g18, s3_cap_bytes=10**12, guard=ok_guard())
            self.assertIn("poison-test PASS", str(cm.exception))
            mk.assert_not_called()

    def test_sample_selection_map_and_dry_run(self):
        specs = [fx.CtxSpec(f"{c[0].upper()}{f}{j}", c, f, "train_fit", f"fam-{c}{f}{j}",
                            item=(f"it{j}" if c == "incoming" else None))
                 for c in ("frozen", "incoming") for f in range(5) for j in range(4)]
        w = self.world("w40", specs)
        picks = B.select_sample(w.epoch, per_cell=4, seed=20261005)
        self.assertEqual(picks, B.select_sample(json.loads(w.epoch.read_bytes()), 4, 20261005))
        self.assertEqual(len(picks), 40)
        self.assertEqual(Counter((e["cohort"], e["fold"]) for e in picks), {(c, f): 4 for c in P.COHORTS for f in range(5)})
        with self.assertRaises(S.CorpusStop) as cm:
            B.select_sample(w.epoch, per_cell=5)
        self.assertEqual(cm.exception.kind, "sample")
        pmap = B.sample_partition_map(picks)
        roles_by_ds = defaultdict(set)
        for e in picks:
            m = pmap[e["context_sha256"]]
            self.assertEqual(m["family_id"], "p5-sample-fam:" + e["dataset_hash"])
            roles_by_ds[e["dataset_hash"]].add("train" if m["split_role"] in ("train_fit", "diag") else m["split_role"])
            if m["split_role"] == "diag":
                self.assertEqual(e["fold"], 3)
        self.assertTrue(all(len(v) == 1 for v in roles_by_ds.values()))
        per_cohort = {c: Counter(next(iter(roles_by_ds[d])) for d in {e["dataset_hash"] for e in picks if e["cohort"] == c})
                      for c in P.COHORTS}
        for c in P.COHORTS:
            self.assertEqual((per_cohort[c]["sealed"], per_cohort[c]["heldout"], per_cohort[c]["dev"]), (1, 3, 3))
        self.assertEqual({m["split_role"] for m in pmap.values()}, {"train_fit", "dev", "diag", "heldout", "sealed"})
        for bad in ("../x", "a/b", "", "."):
            with self.assertRaises(S.CorpusStop):
                B.sample_out(bad)
        with self.assertRaises(S.CorpusStop):
            B.sample_out("ok", self.tmp / "p5-sample-ok")             # under TMPDIR but not RUN/tmp/p5-sample-*
        # a dry run over the 40-context World: fake read client, <= 64 HTTP attempts, verify + poison PASS
        label = f"unittest-{sha(str(self.tmp).encode())[:12]}"
        out = B.sample_out(label)
        self.addCleanup(shutil.rmtree, out, True)
        old_tempdir = B.tempfile.tempdir
        self.addCleanup(setattr, B.tempfile, "tempdir", old_tempdir)
        pool = self.tmp / "seed-pool.jsonl"
        pool.write_bytes(b"".join(fx.dump({"program": fx.program_obj(s, operator="historical_seed"), "rank": 1})
                                  for s in fx.SHARED_SEEDS))
        store = {e["input_ref"]["key"]: (w.jtt1_cache / f"{e['input_ref']['raw_sha256']}.jtt1.gz").read_bytes()
                 for e in w.epoch_doc["contexts"] if e.get("input_ref")}
        attempts = Counter()

        def fake_get(client, key, version_id, bucket=None):
            attempts[key] += 1
            for name, handler in client.meta.events.handlers:
                if name == "before-send.s3.GetObject":
                    handler()
            return store[key], {}

        with mock.patch("dltrain.s3io.get_exact", fake_get), mock.patch.object(S, "EXPECT_COMPLETED_PER_FOLD", 8):
            res = B.sample(label=label, jtt1_cache=self.tmp / "cache", epoch=w.epoch, epoch_sha256=w.epoch_sha256,
                           seed_pool=pool, seed_pool_sha256=sha(pool.read_bytes()), with_sampler_plan=True,
                           incoming_cursor=w.sources.incoming_cursor,
                           incoming_transport_index=w.sources.incoming_transport_index, client_factory=FakeClient,
                           guard=ok_guard(), argv=["sample", "--label", label])
            self.assertEqual(B.tempfile.tempdir, str(out / "tmp"))
            with self.assertRaises(S.CorpusStop):                      # a non-empty OUT needs --resume
                B.sample(label=label, jtt1_cache=self.tmp / "cache", epoch=w.epoch, guard=ok_guard())
        self.assertEqual((res["verdict"], res["exits"]), ("PASS", {"fetch": 0, "build": 0, "verify": 0}))
        rep = json.loads((out / "sample-report.json").read_bytes())
        self.assertEqual(rep["schema"], B.SAMPLE_REPORT_SCHEMA)
        self.assertEqual(rep["processed_contexts"], 40 - sum(1 for m in pmap.values() if m["split_role"] == "sealed"))
        self.assertEqual(rep["poison"]["verdict"], "PASS")
        self.assertEqual(set(attempts.values()), {1})
        self.assertEqual(len(attempts), rep["processed_contexts"])
        led = rep["measurements"]["ledger"]
        self.assertEqual((led["calls"], led["http_attempts"]), (len(attempts), len(attempts)))
        self.assertLessEqual(led["http_attempts"], 64)
        self.assertEqual(rep["measurements"]["sampler_plan"]["contexts"], sum(1 for m in pmap.values()
                                                                              if m["split_role"] == "train_fit"))
        self.assertIsNotNone(rep["measurements"]["rates"]["parse_mb_per_s"])
        self.assertEqual(rep["projection_pass"]["sums"]["unique_programs"],
                         sum(sc.stats()["unique_programs"] for sc in w.contexts.values()))
        self.assertEqual(json.loads((out / "receipt.json").read_bytes())["verdict"], "PASS")
        seeds = json.loads((out / "seeds.json").read_bytes())["seed_hashes"]
        self.assertEqual(seeds, sorted(fx.sha(s) for s in fx.SHARED_SEEDS))


if __name__ == "__main__":
    unittest.main()
