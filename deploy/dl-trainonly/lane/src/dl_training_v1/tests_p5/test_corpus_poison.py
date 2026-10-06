"""Offline proofs for the P5 poison test (PLAN §5 P5 Validation; spec §3.10, tests_p5/test_corpus_poison.py).

One synthetic World is built once (setUpClass); corpus_build.poison_test then runs over copies of the selected
contexts: per cohort the first train_fit one and, round-robin over heldout, dev and diag, enough non-train ones that
every non-empty non-train partition is perturbed (red-team LA-1; the World's frozen diag context F3 shares its family
and dataset with train_fit F1). Build A reads unperturbed copies; build B reads copies with every exposed field (the
outer phase's timing values kept, corpus_schema.RUNTIME_POLICY), every JTT1 outer-test byte, the side files' exposed
fields and the non-train records perturbed, with re-pinned summaries, fingerprints and input_refs. These tests check
the receipt and also re-read both copies independently, so a harness that silently failed to perturb something would
FAIL here, and every sensitivity control (a train-path change, the declared runtime coupling and a code-level leak of
each non-train partition into the train statistics) must turn the verdict FAIL. Nothing touches S3; every write goes
under TMPDIR. Run only inside the lane slice: python -m unittest tests_p5.test_corpus_poison -v
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq

from dltrain import cpuguard, jtt1
from dltrain import corpus_build as B
from dltrain import corpus_schema as S
from dltrain import programs as P
from tests_p5 import fixtures_p5 as fx

TRAIN_FILES = ["train/contexts/part-00000.parquet", "train/label-stats.json", "train/manifest.json",
               "train/records/part-00000.parquet", "train/seed-stats.json"]
INNER_RMSE = re.compile(r"inner_cv\[\d+\]\.metrics\.rmse")        # the per-fold program RMSE (not baseline/training)
KEPT = {".".join(p) for p in S.project_timing_keep_exposed()}     # the outer phase's timing values (kept in B)
TRAIN_DIFF = "train outputs differ between A and B"
RUNTIME_DIFF = "runtime-report train_fit groups differ between A and B"


def ok_guard() -> cpuguard.Guard:
    return cpuguard.Guard(checker=lambda: (True, [], {}), poll_s=0.01)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flat(obj, prefix=""):
    """(dotted key, value) for every leaf of a JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def exposed_leaves(row: dict) -> dict:
    """Every leaf under a key that corpus_schema.name_is_exposed names (the poison harness's perturbation set)."""
    out = {}

    def walk(v, path, inside):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else k, inside or S.name_is_exposed(str(k)))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]", inside)
        elif inside:
            out[path] = v
    walk(row, "", False)
    return out


class Poison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = fx.tmpdir()
        cls.tmp = Path(cls._tmp.name)
        cls._patch = mock.patch.object(cpuguard, "require_slice_limits", return_value={})
        cls._patch.start()
        try:
            cls.w = fx.World(cls.tmp / "w")
            cls.out = cls.tmp / "out"
            res = B.run(epoch=cls.w.epoch, epoch_sha256=cls.w.epoch_sha256, partition_map=cls.w.pmap, seeds=cls.w.seeds,
                        tokenizer=cls.w.tok_dir, out=cls.out, jtt1_cache=cls.w.jtt1_cache, budget_bytes=0, offline=True,
                        incoming_cursor=cls.w.sources.incoming_cursor,
                        incoming_transport_index=cls.w.sources.incoming_transport_index, inventory=cls.w.inventory,
                        expected_completed=cls.w.expected_completed, guard=ok_guard())
            if (res["status"], res["exit"]) != ("COMPLETE", B.EXIT_OK):
                raise AssertionError(f"main build: {res}")
            cls.receipt = B.poison_test(cls.out, cls.tmp / "poison", poison_contexts=2, guard=ok_guard())
            cls.sens = {s: B.poison_test(cls.out, cls.tmp / f"poison-{s}", sensitivity=s, guard=ok_guard())
                        for s in B.POISON_SENSITIVITY}
        except BaseException:
            cls._patch.stop()
            cls._tmp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()
        cls._tmp.cleanup()

    # ------------------------------------------------------------------------------------------------ helpers
    def variant(self, name: str) -> dict:
        root = Path(self.receipt["work"]) / name
        epoch = json.loads((root / "epoch.json").read_bytes())
        return {"root": root, "entries": {e["context_sha256"]: e for e in epoch["contexts"]},
                "cache": (self.w.jtt1_cache if name == "A" else root / "jtt1-cache")}

    def rows(self, v: dict, cid: str) -> dict:
        """The native, outcome and transport rows of one copied context (incoming outcome lines demuxed by context),
        plus its seed-pool, bootstrap-results and prepared.json."""
        e = v["entries"][cid]
        native = Path(e["native_source"]["path"])
        d = native.parent
        if e["cohort"] == "frozen":
            outcome = (d / "history-outcomes.jsonl").read_bytes().splitlines()
            transport = (d / "history-transports.jsonl").read_bytes().splitlines()
        else:
            events = d.parent.parent / "hwm-events.jsonl"
            outcome = [x for x in events.read_bytes().splitlines() if P._line_context(x + b"\n") == cid]
            transport = (v["root"] / "src" / "incoming" / "transport-index" / f"{cid}.jsonl").read_bytes().splitlines()
        out = {k: [json.loads(x) for x in lines] for k, lines in
               (("native", native.read_bytes().splitlines()), ("outcome", outcome), ("transport", transport),
                ("seed_pool", (d / "seed-pool.jsonl").read_bytes().splitlines()),
                ("boot", (d / "bootstrap-results.jsonl").read_bytes().splitlines()))}
        out["prepared"] = json.loads((d / "prepared.json").read_bytes())
        return out

    def selected(self) -> list[dict]:
        return self.receipt["selected"]

    # ----------------------------------------------------------------------------------------------------- tests
    def test_train_outputs_invariant(self):
        r = self.receipt
        self.assertEqual((r["schema"], r["verdict"]), (B.POISON_SCHEMA, "PASS"), r.get("failures"))
        self.assertEqual({k: v["status"] for k, v in r["builds"].items()}, {"A": "COMPLETE", "B": "COMPLETE"})
        picked = {(x["cohort"], x["split_role"]) for x in self.selected()}
        # red-team LA-1: heldout, dev and diag are all perturbed
        self.assertEqual(picked, {("frozen", "train_fit"), ("frozen", "heldout"), ("frozen", "diag"),
                                  ("incoming", "train_fit"), ("incoming", "dev")})
        self.assertEqual((r["nonempty_partitions"], r["perturbed_partitions"]),
                         (["heldout", "dev", "diag"], ["heldout", "dev", "diag"]))
        # the whole train/ tree (the train closure, shards and statistics) is byte-identical (red-team LA-2)
        self.assertTrue(r["train_identical"])
        self.assertEqual(sorted(r["train_files"]), TRAIN_FILES)
        a_out, b_out = Path(r["work"]) / "A" / "corpus", Path(r["work"]) / "B" / "corpus"
        for rel in TRAIN_FILES:
            self.assertEqual((a_out / rel).read_bytes(), (b_out / rel).read_bytes(), rel)
            self.assertEqual(r["train_files"][rel], {"A": sha((a_out / rel).read_bytes()), "B": sha((b_out / rel).read_bytes())})
        self.assertEqual(r["train_closure"], {"A": sha((a_out / "train/manifest.json").read_bytes()),
                                              "B": sha((b_out / "train/manifest.json").read_bytes())})
        self.assertTrue(r["runtime_train_identical"])
        # the test-dependent identity pins moved to provenance/: they differ while train/ does not
        prov_a = B._files_under(a_out, ["provenance/train_fit"])
        prov_b = B._files_under(b_out, ["provenance/train_fit"])
        self.assertTrue(prov_a)
        self.assertNotEqual(prov_a, prov_b)
        # every control differs
        self.assertEqual(r["controls"], {"exposed_differ": True, "nontrain_differ": True, "jtt1_differ": True})
        A, Bv = self.variant("A"), self.variant("B")
        train_ids = {x["context_sha256"] for x in self.selected() if x["split_role"] == "train_fit"}
        for x in self.selected():
            cid = x["context_sha256"]
            ra, rb = self.rows(A, cid), self.rows(Bv, cid)
            with self.subTest(context=cid[:12], role=x["split_role"]):
                # (1) every exposed field of every copied row is perturbed (official_test.*, test_rmse*, fitness,
                #     TabPFN refs, both signatures, flags, result_*), keys and nesting unchanged; the outer phase's
                #     timing values alone are kept (the runtime policy's possible world)
                n_exposed = kept = 0
                for kind in ("native", "outcome", "transport", "seed_pool", "boot"):
                    for a, b in zip(ra[kind], rb[kind]):
                        ea, eb = exposed_leaves(a), exposed_leaves(b)
                        self.assertEqual(set(ea), set(eb), kind)
                        for key, val in ea.items():
                            if val is None:
                                continue
                            if key in KEPT:
                                self.assertEqual(eb[key], val, f"{kind} {key} must be kept")
                                kept += 1
                            else:
                                n_exposed += 1
                                self.assertNotEqual(eb[key], val, f"{kind} {key} was not perturbed")
                self.assertGreater(n_exposed, 50)
                self.assertGreater(kept, 0)
                pa_, pb_ = exposed_leaves(ra["prepared"]), exposed_leaves(rb["prepared"])
                self.assertTrue(pa_ and all(pb_[k] != v for k, v in pa_.items() if v is not None), pa_)
                # (2) every JTT1 outer-test byte is XOR 0x5A; the train bytes change only for non-train contexts
                raw_a = jtt1.load_raw(A["entries"][cid]["input_ref"], A["cache"])
                raw_b = jtt1.load_raw(Bv["entries"][cid]["input_ref"], Bv["cache"])
                start, stop = fx.test_range(raw_a)
                self.assertEqual(stop, len(raw_a))
                self.assertEqual(raw_b[start:], bytes(c ^ 0x5A for c in raw_a[start:]))
                self.assertEqual(jtt1.parse_header(raw_a), jtt1.parse_header(raw_b))
                same_train = jtt1.train_rows_sha256(raw_a) == jtt1.train_rows_sha256(raw_b)
                self.assertEqual(same_train, cid in train_ids)
                self.assertNotEqual(A["entries"][cid]["input_ref"]["raw_sha256"], Bv["entries"][cid]["input_ref"]["raw_sha256"])
                if cid in train_ids:
                    # (3a) a train context's train-path view is untouched, wall_us included
                    for kind, spec in (("native", S.NATIVE_TRAIN_SPEC), ("outcome", S.OUTCOME_TRAIN_SPEC),
                                       ("transport", S.TRANSPORT_TRAIN_SPEC)):
                        self.assertEqual(len(ra[kind]), len(rb[kind]))
                        for a, b in zip(ra[kind], rb[kind]):
                            self.assertEqual(json.dumps(S.prune(a, spec), sort_keys=True),
                                             json.dumps(S.prune(b, spec), sort_keys=True), kind)
                    continue
                # (3b) a non-train context's records are perturbed: metrics x0.37, completed <-> failed flips,
                #      node and pool swaps, wall_us x2 and 2 extra programs
                self.assertEqual(len(rb["native"]), len(ra["native"]) + 2)
                st_a = {n["program"]["program_sha256"]: n["status"] for n in ra["native"]}
                st_b = {n["program"]["program_sha256"]: n["status"] for n in rb["native"]}
                flipped = [p for p in st_a if st_a[p] != st_b[p]]
                self.assertTrue(flipped)
                self.assertTrue(all({st_a[p], st_b[p]} == {"completed", "failed"} for p in flipped))
                self.assertEqual(Counter(st_a.values())["completed"], Counter(st_b.values())["completed"])
                extra = set(st_b) - set(st_a)
                self.assertEqual(len(extra), 2)
                scaled = walls = nodes = 0
                for a, b in zip(ra["outcome"], rb["outcome"]):
                    for (ka, va), (kb, vb) in zip(flat(a.get("evaluation_detail") or {}), flat(b.get("evaluation_detail") or {})):
                        self.assertEqual(ka, kb)
                        if INNER_RMSE.fullmatch(ka) and isinstance(va, float) and va:
                            self.assertAlmostEqual(vb, va * B.POISON_SCALE, delta=1e-12 * abs(va))
                            scaled += 1
                    if isinstance(a.get("wall_us"), int):
                        self.assertEqual(b["wall_us"], 2 * a["wall_us"])
                        walls += 1
                    if isinstance(a.get("node"), str) and a["node"].startswith("xbabe"):
                        self.assertTrue(b["node"].startswith("aws:"))
                        nodes += 1
                    if b.get("evaluation_detail", {}).get("inner_cv") and isinstance(b.get("cv_log_ratio_sd"), float):
                        rmse, base = S.inner_cv_pairs(b["evaluation_detail"])
                        self.assertEqual(b["cv_log_ratio_sd"], S.canonical_cv_log_ratio_sd(rmse, base))
                pools = sum(1 for a, b in zip(ra["transport"], rb["transport"])
                            if isinstance(a.get("pool"), str) and a["pool"][:4] != b["pool"][:4])
                self.assertGreater(scaled, 0)
                self.assertGreater(walls, 0)
                self.assertGreater(nodes, 0)
                self.assertGreater(pools, 0)
        # the controls differ in the built outputs too: exposed tables and non-train outputs
        for rel in ("exposed/train_fit", "heldout/exposed", "eval/dev/records", "eval/diag/records", "heldout/records"):
            fa = B._files_under(a_out, [rel])
            fb = B._files_under(b_out, [rel])
            self.assertTrue(fa, rel)
            self.assertNotEqual(fa, fb, rel)

    def test_sensitivity_controls(self):
        """Each control must make the same comparison FAIL: one train_fit record's rmse_j, its context's JTT1 train
        rows, a consistent shift of a train context's outer-phase wall time with its whole-evaluation wall_us (the
        declared runtime coupling, red-team LA-3), and a deliberate code-level leak of heldout, dev or diag labels
        into train/label-stats.json (red-team LA-1 negative controls)."""
        want = {"train_rmse": ([TRAIN_DIFF], {"train/records/part-00000.parquet"}),
                "train_jtt1": ([TRAIN_DIFF], {"train/contexts/part-00000.parquet"}),
                "official_wall": ([TRAIN_DIFF, RUNTIME_DIFF], {"train/records/part-00000.parquet"}),
                "leak_heldout": ([TRAIN_DIFF], {"train/label-stats.json", "train/manifest.json"}),
                "leak_dev": ([TRAIN_DIFF], {"train/label-stats.json", "train/manifest.json"}),
                "leak_diag": ([TRAIN_DIFF], {"train/label-stats.json", "train/manifest.json"})}
        self.assertEqual(set(want), set(B.POISON_SENSITIVITY))
        for sens, r in self.sens.items():
            with self.subTest(sensitivity=sens):
                self.assertEqual(r["verdict"], "FAIL")
                self.assertEqual({k: v["status"] for k, v in r["builds"].items()}, {"A": "COMPLETE", "B": "COMPLETE"})
                self.assertFalse(r["train_identical"])
                failures, changed_at_least = want[sens]
                self.assertEqual(r["failures"], failures)
                changed = {k for k, v in r["train_files"].items() if v["A"] != v["B"]}
                self.assertLessEqual(changed_at_least, changed)
                if sens.startswith("leak_"):
                    self.assertEqual(changed, changed_at_least)              # the shards themselves are clean
                    self.assertIn(sens[len("leak_"):], r["perturbed_partitions"])
                self.assertNotEqual(r["train_closure"]["A"], r["train_closure"]["B"])
                self.assertEqual(r["controls"], {"exposed_differ": True, "nontrain_differ": True, "jtt1_differ": True})
        # the leak hook is a context manager and leaves the module as it found it
        self.assertEqual(B.label_stats_from_shards.__name__, "label_stats_from_shards")

    def test_harness_neutrality(self):
        """Build A's train rows for the copied contexts equal the main build's rows for those contexts."""
        self.assertEqual(self.receipt["neutrality"], {"records_equal": True, "contexts_equal": True})
        train_ids = {x["context_sha256"] for x in self.selected() if x["split_role"] == "train_fit"}
        self.assertEqual(len(train_ids), 2)
        cur = json.loads((self.out / "cursor.json").read_bytes())
        a_out = Path(self.receipt["work"]) / "A" / "corpus"
        for table in ("records", "contexts"):
            main = [row for sh in cur["writers"]["train_fit"][table]["shards"]
                    for row in pq.read_table(self.out / sh["path"], use_threads=False).to_pylist()
                    if row["context_sha256"] in train_ids]
            mine = [row for f in sorted((a_out / "train" / table).glob("*.parquet"))
                    for row in pq.read_table(f, use_threads=False).to_pylist()]
            with self.subTest(table=table):
                self.assertTrue(main)
                self.assertEqual(mine, main)
        # A's source copies are byte-identical to the originals (paths aside); only B is perturbed
        A = self.variant("A")
        for cid in train_ids:
            e = A["entries"][cid]
            orig = next(x for x in self.w.epoch_doc["contexts"] if x["context_sha256"] == cid)
            self.assertEqual({k: e["native_source"][k] for k in ("bytes", "sha256")},
                             {k: orig["native_source"][k] for k in ("bytes", "sha256")})
            self.assertEqual(e["input_ref"], orig["input_ref"])
            self.assertEqual({k: e["prepared_ref"][k] for k in ("bytes", "sha256")},
                             {k: orig["prepared_ref"][k] for k in ("bytes", "sha256")})


if __name__ == "__main__":
    unittest.main()
