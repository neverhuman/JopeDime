"""Lane hygiene for P5 (spec tests_p5/test_corpus_hygiene.py): the suite-wide write guard, the out-directory gate,
the CPU guards' call order, the absence of any S3 write path, the confinement of exposed key strings and the pinned
contract of the dltrain.programs (P4a) helpers that P5 imports.

Every write goes under TMPDIR (tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])); nothing touches S3 (the read
client and get_exact are fakes). Run only inside the lane slice: python -m unittest tests_p5.test_corpus_hygiene -v
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import gzip
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dltrain import common, cpuguard, jtt1
from dltrain import corpus_build as B
from dltrain import corpus_schema as S
from dltrain import programs as P
from tests_p5 import fixtures_p5 as fx

FORBIDDEN_IMPORTS = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry"}
EXPOSED_TOKENS = ("official_test", "test_rmse", "fitness", "tabpfn35", "signature", "sampled_absolute_error",
                  "full_prediction", "population_selected", "semantic_duplicate", "result_key", "result_ref",
                  "signal_admitted", "best_signal_test")
ALLOWED_DEFS = re.compile(r"project_\w+_exposed|recompute_high_water")
ALLOWED_ASSIGNS = {"EXPOSED_RECORD_SCHEMA", "EXPOSED_NAMES", "FORBIDDEN_SUBSTRINGS"}
WRITE_CALLS = re.compile(r"(put_\w*|upload\w*|copy_object|delete_object\w*|create_multipart_upload|complete_multipart_upload"
                         r"|put_create_only|put_if_match)")
SMALL = (fx.CtxSpec("H0", "frozen", 1, "train_fit", "fam-h0"),
         fx.CtxSpec("H1", "incoming", 2, "train_fit", "fam-h1", item="HA"),
         fx.CtxSpec("H2", "frozen", 2, "dev", "fam-h2"))
# The private (and public) dltrain.programs symbols P5 relies on, with the signatures it was written against. A drift
# here means programs.py changed under P5: re-run this suite and re-review P5 (spec §3.2).
PROGRAMS_CONTRACT = {
    "_plan": "(epoch: 'Path | str', partition_map: 'Path | str | Mapping') -> '_Plan'",
    "_summary": ("(ctx: 'ContextRef', cursor: 'Mapping[str, Any] | None', cursor_pin: 'dict | None' = None) -> "
                 "'tuple[HistorySummary, dict]'"),
    "_ctx_dir": "(ctx: 'ContextRef') -> 'Path'",
    "_item_gp": "(ctx: 'ContextRef') -> 'Path'",
    "_transport_path": "(ctx: 'ContextRef', sources: 'Sources') -> 'Path'",
    "_Prefix": "(path: 'Path', nbytes: 'int', kind: 'str', ctx: 'str | None', extra: 'Sequence[Any]' = ())",
    "_complete_lines": "(path: 'Path') -> 'Iterator[bytes]'",
    "_line_context": "(line: 'bytes') -> 'str | None'",
    "_snapshot": "(path: 'Path') -> 'bytes'",
    "_seed_pool": "(ctx: 'ContextRef') -> 'tuple[set[str], dict]'",
    "_load_seeds": "(src: 'Path | str | Iterable[str]') -> 'tuple[frozenset[str], dict]'",
    "_check_program": "(dsl: 'Any', sha: 'Any', where: 'str', ctx: 'str | None' = None) -> 'None'",
    "_project_cursor_entry": "(e: 'Mapping') -> 'dict'",
    "_publish_immutable": "(path: 'Path', data: 'bytes') -> 'dict'",
    "_rehash": "(rec: 'dict', ctx_id: 'str') -> 'str | None'",
    "_is_hex64": "(v: 'Any') -> 'bool'",
    "load_incoming_cursor": "(path: 'Path | str') -> 'tuple[dict[str, dict], dict]'",
    "ReconcileError": "(message: 'str', context: 'str | None' = None, kind: 'str' = 'reconcile')",
    # R7 reads origin and the exact record_status from the sealed side table only (role r7_crosscheck), and the
    # tokenizer-closure subset check reads the train_fit partition through the guarded opener (role fit)
    "SealedIndex": "(dict_dir: 'Path | str', *, role: 'str' = 'r7_crosscheck')",
    "open_partition": "(out_dir: 'Path | str', partition: 'str', *, role: 'str' = 'fit') -> 'pads.Dataset'",
    "context_rows": None, "build": None, "verify": None, "iter_distinct": None,      # public; checked by name/kwargs
    "Sources": None, "DEFAULT_SOURCES": None, "ContextRef": None, "HistorySummary": None,
    "SUMMARY_KEYS": None, "ABANDON_REASONS": None, "BOOTSTRAP_STATUSES": None, "PARTITIONS": None, "COHORTS": None,
    "ORIGINS": None, "READ_BLOCK": None,
}
SPEC_PINNED = {"_plan", "_summary", "_Prefix", "_ctx_dir", "_item_gp", "_transport_path", "_complete_lines",
               "_line_context", "_snapshot", "_seed_pool", "_load_seeds", "_check_program", "_project_cursor_entry",
               "_publish_immutable", "load_incoming_cursor", "Sources", "ReconcileError", "SUMMARY_KEYS",
               "ABANDON_REASONS", "DEFAULT_SOURCES", "ContextRef", "HistorySummary"}


def ok_guard() -> cpuguard.Guard:
    return cpuguard.Guard(checker=lambda: (True, [], {}), poll_s=0.01)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeEvents:
    def __init__(self):
        self.handlers = []

    def register(self, name, handler, unique_id=None):
        self.handlers.append((name, handler))


class StrictFakeClient:
    """A read-client stand-in: only meta.events exists; asking for any other attribute (put_object, upload_file, ...)
    fails the test."""

    def __init__(self):
        self.meta = SimpleNamespace(events=FakeEvents())

    def __getattr__(self, name):
        raise AssertionError(f"S3 client attribute {name!r} was requested")


class RecordingGuard(cpuguard.Guard):
    """A permissive Guard that records wait() calls on a shared timeline and classifies each tick() by its caller:
    'line' inside corpus_build.scan_sources, 'context' inside the build or fetch loops."""

    def __init__(self, timeline: list):
        super().__init__(checker=lambda: (True, [], {}), poll_s=0.01)
        self.timeline = timeline
        self.tick_sites: Counter = Counter()

    def tick(self):
        names, f = set(), sys._getframe(1)
        while f is not None:
            names.add(f.f_code.co_name)
            f = f.f_back
        self.tick_sites["line" if "scan_sources" in names else
                        "context" if names & {"build_phase", "fetch_phase"} else "other"] += 1
        return super().tick()

    def wait(self):
        self.timeline.append(("wait",))
        return super().wait()


def module_tree(mod) -> ast.Module:
    return ast.parse(Path(mod.__file__).read_text())


def attr_uses(tree: ast.AST, alias: str) -> set[str]:
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == alias}


class Hygiene(unittest.TestCase):

    def setUp(self):
        self._tmp = fx.tmpdir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def world(self, name: str = "w", specs=SMALL) -> fx.World:
        return fx.World(self.tmp / name, specs)

    def run_kw(self, w: fx.World, out: Path, **kw) -> dict:
        args = dict(epoch=w.epoch, epoch_sha256=w.epoch_sha256, partition_map=w.pmap, seeds=w.seeds,
                    tokenizer=w.tok_dir, out=out, jtt1_cache=w.jtt1_cache, budget_bytes=0, offline=True,
                    incoming_cursor=w.sources.incoming_cursor,
                    incoming_transport_index=w.sources.incoming_transport_index, inventory=w.inventory,
                    expected_completed=w.expected_completed, guard=ok_guard())
        args.update(kw)
        return args

    # ------------------------------------------------------------------------------------------------ write guard
    def test_write_guard(self):
        # the hook is installed at import, armed for the whole suite, rooted at realpath(TMPDIR) (never /tmp)
        self.assertEqual(fx.ALLOWED_ROOT, os.path.realpath(os.environ["TMPDIR"]))
        self.assertFalse(fx.ALLOWED_ROOT == "/tmp" or fx.ALLOWED_ROOT.startswith("/tmp/"))
        self.assertTrue(fx._GUARD["armed"])
        self.assertTrue(str(self.tmp.resolve()).startswith(fx.ALLOWED_ROOT + os.sep))
        inside = os.path.join(fx.ALLOWED_ROOT, "x", "y.json")
        outside = "/home/ubuntu/JopeDime/never-written-by-p5-tests.json"
        w = os.O_WRONLY | os.O_CREAT
        self.assertEqual(fx.write_target("open", (outside, "wb", w)), outside)
        self.assertEqual(fx.write_target("open", (outside, None, os.O_RDWR | os.O_CREAT | os.O_EXCL)), outside)
        self.assertEqual(fx.write_target("open", (outside, "a", 0)), outside)
        self.assertIsNone(fx.write_target("open", (outside, "rb", os.O_RDONLY)))
        self.assertIsNone(fx.write_target("open", (3, "wb", w)))
        self.assertIsNone(fx.write_target("open", (os.devnull, None, os.O_RDWR)))      # a subprocess DEVNULL handle
        self.assertEqual(fx.write_target("os.rename", (inside, outside, -1, -1)), outside)
        self.assertEqual(fx.write_target("os.link", (inside, outside, -1, -1)), outside)
        self.assertEqual(fx.write_target("os.mkdir", (outside, 0o755, -1)), outside)
        self.assertEqual(fx.write_target("os.remove", (outside, -1)), outside)
        self.assertEqual(fx.write_target("shutil.copyfile", (inside, outside)), outside)
        self.assertIsNone(fx.write_target("os.listdir", (outside,)))
        self.assertFalse(fx.under_root(fx.ALLOWED_ROOT + "-sibling/x"))
        # the hook is live: narrow the root, write on both sides, and only the outside write is recorded
        allowed = self.tmp / "allowed"
        allowed.mkdir()
        n0 = len(fx._GUARD["violations"])
        with mock.patch.object(fx, "ALLOWED_ROOT", os.path.realpath(allowed)):
            (self.tmp / "sibling.txt").write_bytes(b"x")
            (allowed / "inside.txt").write_bytes(b"y")
        seen = fx._GUARD["violations"][n0:]
        del fx._GUARD["violations"][n0:]
        self.assertEqual([t for _, t in seen], [os.path.realpath(self.tmp / "sibling.txt")])
        # zero writes outside TMPDIR so far; tests_p5/__init__.load_tests re-checks after the last module
        self.assertEqual(fx.violations(), [])
        # _check_out: under TMPDIR (tests), RUN/tmp/p5-sample-* (sample) or RUN/corpus/epoch-000 behind gate receipts
        corpus_existed = B.CORPUS_OUT.exists()
        self.assertEqual(B._check_out(self.tmp / "out"), Path(os.path.realpath(self.tmp / "out")))
        for bad in (B.CORPUS_OUT, B.RUN / "corpus" / "epoch-001", B.RUN / "inputs" / "jtt1",
                    Path(B.__file__).resolve().parent, Path("/home/ubuntu/scratch/p5-elsewhere")):
            with self.subTest(out=str(bad)), self.assertRaises(S.CorpusStop) as cm:
                B._check_out(bad)
            self.assertEqual(cm.exception.kind, "gate")
        narrow = self.tmp / "narrow"
        narrow.mkdir()
        with mock.patch.dict(os.environ, {"TMPDIR": str(narrow)}):
            self.assertEqual(B._check_out(narrow / "x"), Path(os.path.realpath(narrow / "x")))
            self.assertEqual(B._check_out(B.SAMPLE_ROOT / "p5-sample-hygiene"),
                             Path(os.path.realpath(B.SAMPLE_ROOT / "p5-sample-hygiene")))
            for bad in (B.SAMPLE_ROOT / "p5-sample-", B.SAMPLE_ROOT / "not-a-sample", B.SAMPLE_ROOT, self.tmp / "x"):
                with self.subTest(out=str(bad)), self.assertRaises(S.CorpusStop):
                    B._check_out(bad)
        with mock.patch.dict(os.environ, {"TMPDIR": "/tmp"}), self.assertRaises(S.CorpusStop):
            B._check_out("/tmp/p5-x")                       # /tmp never counts as TMPDIR
        # RUN/corpus/epoch-000 only with PASS receipts for P3 (binding this partition map) and P4
        pmap = self.tmp / "map.json"
        pmap.write_bytes(b"{}")
        p3 = self.tmp / "p3.json"
        p3.write_text(json.dumps({"verdict": "PASS", "partition_map_sha256": sha(pmap.read_bytes())}))
        p3_other = self.tmp / "p3-other.json"
        p3_other.write_text(json.dumps({"verdict": "PASS", "partition_map_sha256": "0" * 64}))
        p4, p4_fail = self.tmp / "p4.json", self.tmp / "p4-fail.json"
        p4.write_text(json.dumps({"verdict": "PASS"}))
        p4_fail.write_text(json.dumps({"verdict": "FAIL"}))
        self.assertEqual(B._check_out(B.CORPUS_OUT, gate_p3=p3, gate_p4=p4, partition_map=pmap),
                         Path(os.path.realpath(B.CORPUS_OUT)))
        for g3, g4 in ((p3_other, p4), (p3, p4_fail), (None, p4), (p3, None)):
            with self.subTest(gate=(str(g3), str(g4))), self.assertRaises(S.CorpusStop):
                B._check_out(B.CORPUS_OUT, gate_p3=g3, gate_p4=g4, partition_map=pmap)
        self.assertEqual(B.CORPUS_OUT.exists(), corpus_existed)          # the gate check never creates it

    # --------------------------------------------------------------------------------------------- guard order
    def test_guards_called_first(self):
        w = self.world()
        timeline: list = []

        def mark(name):
            def side(*a, **kw):
                timeline.append((name, *a))
                return {}
            return side

        def first(pred) -> int:
            return next(i for i, e in enumerate(timeline) if pred(e))

        def roots_open(*roots):
            return lambda e: e[0] in ("open", "make_client", "get") and (
                e[0] != "open" or any(fx.under_root(e[1], os.path.realpath(r)) for r in roots))

        def assert_order(*roots):
            i_req = first(lambda e: e[0] == "require_slice_limits")
            i_thr = first(lambda e: e[0] == "set_threads")
            self.assertLess(i_req, i_thr)
            self.assertEqual(timeline[i_thr], ("set_threads", 4))
            self.assertLess(i_thr, first(roots_open(*roots)))

        store = {e["input_ref"]["key"]: (w.jtt1_cache / f"{e['input_ref']['raw_sha256']}.jtt1.gz").read_bytes()
                 for e in w.epoch_doc["contexts"] if e.get("input_ref")}

        def fake_get(client, key, version_id, bucket=None):
            timeline.append(("get", key))
            for name, handler in client.meta.events.handlers:
                if name == "before-send.s3.GetObject":
                    handler()
            return store[key], {"VersionId": version_id}

        def make_client(*a, **kw):
            timeline.append(("make_client", a, kw))
            return StrictFakeClient()

        real_publish = P._publish_immutable

        def publish(path, data):
            timeline.append(("publish", str(path)))
            return real_publish(path, data)

        out = self.tmp / "run"
        guard = RecordingGuard(timeline)
        with mock.patch.object(cpuguard, "require_slice_limits", side_effect=mark("require_slice_limits")), \
                mock.patch.object(cpuguard, "set_threads", side_effect=mark("set_threads")), \
                mock.patch("dltrain.s3io.make_client", side_effect=make_client), \
                mock.patch("dltrain.s3io.get_exact", fake_get), \
                mock.patch.object(P, "_publish_immutable", publish), mock.patch.object(B, "LINE_TICK", 7), \
                fx.trace_opens(timeline):
            res = B.run(**self.run_kw(w, out, jtt1_cache=self.tmp / "cache", budget_bytes=10**9, offline=False,
                                      guard=guard))
        self.assertEqual((res["status"], res["exit"]), ("COMPLETE", B.EXIT_OK), res)
        assert_order(w.root, out, self.tmp / "cache")
        # Guard.tick inside the line loops and the context loops; Guard.wait before each GET and each shard flush
        self.assertGreater(guard.tick_sites["line"], 0)
        self.assertGreater(guard.tick_sites["context"], 0)
        gets = [i for i, e in enumerate(timeline) if e[0] == "get"]
        self.assertEqual(len(gets), len(w.contexts))       # one exact GET per processed context
        def shard(path: str) -> bool:          # a Parquet shard or a ctx-record shard (provenance/<p>/, heldout/ctx-records/)
            p = Path(path)
            return p.suffix == ".parquet" or (p.name.startswith("part-") and (
                p.parent.name == "ctx-records" or p.parent.parent.name == "provenance"))
        flushes = [i for i, e in enumerate(timeline) if e[0] == "publish" and shard(e[1])]
        self.assertGreater(len(flushes), 5)
        for kind, idxs in (("GET", gets), ("shard flush", flushes)):
            prev = -1
            for i in idxs:
                self.assertIn(("wait",), timeline[prev + 1:i], f"no Guard.wait before a {kind}")
                prev = i
        # verify: the same order before it reads the built corpus
        timeline.clear()
        with mock.patch.object(cpuguard, "require_slice_limits", side_effect=mark("require_slice_limits")), \
                mock.patch.object(cpuguard, "set_threads", side_effect=mark("set_threads")), fx.trace_opens(timeline):
            rep = B.verify(out, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        assert_order(out)
        # sample: the same order, then tempfile.tempdir = OUT/tmp before the first source read (a bad epoch pin STOPs)
        label = "hygiene-" + sha(str(self.tmp).encode())[:12]
        sample_out = B.SAMPLE_ROOT / f"p5-sample-{label}"
        self.addCleanup(shutil.rmtree, sample_out, True)
        self.addCleanup(setattr, B.tempfile, "tempdir", B.tempfile.tempdir)
        timeline.clear()
        with mock.patch.object(cpuguard, "require_slice_limits", side_effect=mark("require_slice_limits")), \
                mock.patch.object(cpuguard, "set_threads", side_effect=mark("set_threads")), \
                fx.trace_opens(timeline), self.assertRaises(S.CorpusStop) as cm:
            B.sample(label=label, jtt1_cache=self.tmp / "cache", epoch=w.epoch, epoch_sha256="0" * 64, guard=ok_guard())
        self.assertEqual(cm.exception.kind, "input")
        assert_order(w.root, sample_out)
        self.assertEqual(B.tempfile.tempdir, str(sample_out / "tmp"))
        # a GuardTimeout is a resumable pause: exit 75 with the cursor saved, then the resume completes
        journal = self.tmp / "paused-journal.jsonl"
        refusing = cpuguard.Guard(journal_path=journal, checker=lambda: (False, ["admit_false"], {}), poll_s=0,
                                  max_pause_s=0, sleep=lambda s: None)
        with mock.patch.object(cpuguard, "require_slice_limits", return_value={}):
            res = B.run(**self.run_kw(w, self.tmp / "paused", guard=refusing))
            self.assertEqual((res["status"], res["exit"]), ("PAUSED_GUARD", B.EXIT_PAUSED), res)
            cur = json.loads((self.tmp / "paused" / "cursor.json").read_bytes())
            self.assertEqual((cur["status"], cur["pause"]["status"]), ("PAUSED_GUARD", "PAUSED_GUARD"))
            self.assertEqual(cur["ledger"]["invocations"][-1]["status"], "PAUSED_GUARD")
            self.assertEqual([json.loads(x)["event"] for x in journal.read_bytes().splitlines()], ["pause", "timeout"])
            res = B.run(**self.run_kw(w, self.tmp / "paused"))
            self.assertEqual((res["status"], res["exit"]), ("COMPLETE", B.EXIT_OK), res)
            # the CLI maps a GuardTimeout to exit 75 as well
            real_guard = cpuguard.Guard
            with mock.patch.object(cpuguard, "Guard", lambda **kw: real_guard(
                    checker=lambda: (False, ["cpu_rolling_high"], {}), poll_s=0, sleep=lambda s: None,
                    **{**kw, "max_pause_s": 0})), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(B.main(["verify", "--out", str(self.tmp / "paused")]), B.EXIT_PAUSED)
            self.assertEqual(json.loads(err.getvalue())["status"], "PAUSED_GUARD")

    # ------------------------------------------------------------------------------------- imports and S3 writes
    def test_imports_and_no_s3_writes(self):
        for mod in (S, B):
            tree = module_tree(mod)
            imported, top = set(), set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names} | {a.name for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported |= {(node.module or "").split(".")[0], node.module or ""} | {a.name for a in node.names}
            for node in tree.body:
                if isinstance(node, ast.Import):
                    top |= {a.name for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    top |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
            with self.subTest(module=mod.__name__):
                self.assertFalse(imported & FORBIDDEN_IMPORTS, imported & FORBIDDEN_IMPORTS)
                self.assertFalse({"boto3", "botocore", "dltrain.s3io", "s3io"} & top, top)
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
                names = [(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)) for n in calls]
                self.assertEqual([x for x in names if x and WRITE_CALLS.fullmatch(x)], [])
                for n in calls:
                    fname = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    if fname == "make_client":                     # read mode only: no arguments at all
                        self.assertEqual((n.args, n.keywords), ([], []), ast.unparse(n))
                    if fname == "get_exact":
                        self.fail(f"{mod.__name__} calls get_exact directly; jtt1.load_raw is the only S3 path")
        self.assertFalse(attr_uses(module_tree(S), "s3io"))
        self.assertEqual(attr_uses(module_tree(B), "s3io"), {"make_client"})
        loads = [n for n in ast.walk(module_tree(B)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "load_raw"]
        self.assertTrue(loads)
        self.assertEqual(sum(1 for n in loads if len(n.args) >= 3), 1)     # one call site passes a client: fetch_phase
        # the mock proof: a fetch from an empty cache asks s3io for one read client and GETs only through load_raw
        w = self.world("fetch", SMALL[:2])
        store = {e["input_ref"]["key"]: (w.jtt1_cache / f"{e['input_ref']['raw_sha256']}.jtt1.gz").read_bytes()
                 for e in w.epoch_doc["contexts"] if e.get("input_ref")}
        callers = []

        def fake_get(client, key, version_id, bucket=None):
            names, f = [], sys._getframe(1)
            while f is not None:
                names.append(f.f_code.co_name)
                f = f.f_back
            callers.append(names[:3])
            self.assertIsInstance(client, StrictFakeClient)
            for name, handler in client.meta.events.handlers:
                if name == "before-send.s3.GetObject":
                    handler()
            return store[key], {"VersionId": version_id}

        refuse = mock.Mock(side_effect=AssertionError("an S3 write helper was called"))
        with mock.patch.object(cpuguard, "require_slice_limits", return_value={}), \
                mock.patch("dltrain.s3io.make_client", side_effect=lambda *a, **k: StrictFakeClient()) as mk, \
                mock.patch("dltrain.s3io.get_exact", fake_get), mock.patch("dltrain.s3io.put_create_only", refuse), \
                mock.patch("dltrain.s3io.put_if_match", refuse):
            res = B.run(**self.run_kw(w, self.tmp / "fetch-out", jtt1_cache=self.tmp / "cache", budget_bytes=10**9,
                                      offline=False, phase="fetch"))
        self.assertEqual(res["status"], "FETCHED", res)
        self.assertEqual(mk.call_args_list, [mock.call()])
        self.assertEqual(len(callers), len(w.contexts))
        self.assertTrue(all(c[0] == "load_raw" and c[1] == "fetch_phase" for c in callers), callers)
        refuse.assert_not_called()
        led = json.loads((self.tmp / "fetch-out" / "cursor.json").read_bytes())["ledger"]
        self.assertEqual((led["calls"], led["http_attempts"]), (len(w.contexts), len(w.contexts)))

    # -------------------------------------------------------------------------------------- exposed confinement
    def test_exposed_names_confined(self):
        offenders = []
        for mod in (S, B):
            tree = module_tree(mod)
            for top in tree.body:
                if isinstance(top, (ast.FunctionDef, ast.ClassDef)) and ALLOWED_DEFS.fullmatch(top.name):
                    continue
                if isinstance(top, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ALLOWED_ASSIGNS
                                                       for t in top.targets):
                    continue
                for node in ast.walk(top):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        hits = [t for t in EXPOSED_TOKENS if t in node.value]
                        if hits:
                            offenders.append((mod.__name__, getattr(top, "name", type(top).__name__),
                                              getattr(node, "lineno", 0), hits))
        self.assertEqual(offenders, [])
        # corpus_build holds no exposed key string at all: it routes values through corpus_schema's projectors
        build_strings = [n.value for n in ast.walk(module_tree(B)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertFalse([s for s in build_strings if any(t in s for t in EXPOSED_TOKENS)])
        # the non-sealed tables and their JSON Schemas carry no exposed name; the sealed one carries them all
        for tid, schema in S.TABLES.items():
            names = list(schema.names) + list(json.loads(S.json_schema_bytes(tid))["properties"])
            exposed = [n for n in names if S.name_is_exposed(n)]
            if tid in S.SEALED_TABLES:
                self.assertTrue(exposed)
            else:
                self.assertEqual(exposed, [], tid)

    # ------------------------------------------------------------------------------------ programs.py contract
    def test_programs_contract(self):
        used = attr_uses(module_tree(B), "P") | attr_uses(module_tree(S), "programs")
        self.assertTrue(SPEC_PINNED <= set(PROGRAMS_CONTRACT))
        self.assertEqual(used - set(PROGRAMS_CONTRACT), set(), "P5 uses a programs symbol outside the pinned contract")
        for name, sig in PROGRAMS_CONTRACT.items():
            with self.subTest(symbol=name):
                self.assertTrue(hasattr(P, name))
                if sig is not None:
                    self.assertEqual(str(inspect.signature(getattr(P, name))), sig)
        for name, kwargs in (("context_rows", {"seeds", "item_events", "sources", "evidence"}),
                             ("build", {"chunk_contexts", "resume", "sources", "s3_fallback"}),
                             ("verify", {"reconcile_history_summaries", "rehash_sources", "sources"}),
                             ("iter_distinct", {"role", "batch"})):
            params = inspect.signature(getattr(P, name)).parameters
            self.assertTrue(kwargs <= set(params), (name, set(params)))
        # constants and record types
        self.assertEqual(P.SUMMARY_KEYS, ("unique_programs", "native_events", "native_bytes", "outcome_bytes",
                                          "outcome_events", "transport_bytes", "transport_events"))
        self.assertEqual(set(P.ABANDON_REASONS), {"transport_error", "capacity_deferred_before_claim", "runtime_error",
                                                  "blocked_existing_claim"})
        self.assertEqual(P.BOOTSTRAP_STATUSES, ("completed", "failed", "timeout", "missing"))
        self.assertEqual((P.PARTITIONS, P.COHORTS, P.ORIGINS),
                         (("train_fit", "dev", "diag", "heldout"), ("frozen", "incoming"),
                          ("evolution", "bootstrap", "transport_only")))
        self.assertGreater(P.READ_BLOCK, 0)
        self.assertEqual([f.name for f in dataclasses.fields(P.Sources)], ["incoming_cursor", "incoming_transport_index"])
        self.assertEqual([f.name for f in dataclasses.fields(P.HistorySummary)],
                         [*P.SUMMARY_KEYS, "history_sha256", "fingerprint", "bootstrap_source_sha256"])
        self.assertEqual([f.name for f in dataclasses.fields(P.ContextRef)],
                         ["context_sha256", "cohort", "view_id", "item_id", "dataset_hash", "fold", "native", "prepared",
                          "history_sha256", "family_id", "split_role"])
        err = P.ReconcileError("m", "c" * 64, "count")
        self.assertEqual((err.context, err.kind), ("c" * 64, "count"))
        self.assertTrue(P._is_hex64("a" * 64) and not P._is_hex64("A" * 64) and not P._is_hex64("a" * 63))
        # line helpers
        cid = "ab" * 32
        line = json.dumps({"schema": "x", "context_sha256": cid, "program": {"dsl": "y"}}).encode() + b"\n"
        self.assertEqual(P._line_context(line), cid)
        self.assertIsNone(P._line_context(b'{"schema":"x"}\n'))
        f = self.tmp / "live.jsonl"
        f.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
        self.assertEqual(list(P._complete_lines(f)), [b'{"a":1}\n', b'{"b":2}\n'])
        self.assertEqual(P._snapshot(f), b'{"a":1}\n{"b":2}\n')
        self.assertEqual(P._snapshot(self.tmp / "absent.jsonl"), b"")
        extra = hashlib.sha256()
        reader = P._Prefix(f, 16, "native", cid, (extra,))
        self.assertEqual(list(reader), [b'{"a":1}\n', b'{"b":2}\n'])
        self.assertEqual((reader.lines, reader.sha.hexdigest(), extra.hexdigest()),
                         (2, sha(b'{"a":1}\n{"b":2}\n'), sha(b'{"a":1}\n{"b":2}\n')))
        with self.assertRaises(P.ReconcileError) as cm:
            list(P._Prefix(f, 12, "native", cid))                  # a recorded prefix ending inside a line
        self.assertEqual(cm.exception.kind, "source")
        with self.assertRaises(P.ReconcileError) as cm:
            list(P._Prefix(f, 64, "native", cid))                  # fewer bytes than recorded
        self.assertEqual(cm.exception.kind, "source")
        # program checks, seeds, cursor projection and create-only publication
        dsl = "(pipe X (norm_row) (ridge_gcv))"
        P._check_program(dsl, sha(dsl.encode()), "w")
        with self.assertRaises(P.ReconcileError) as cm:
            P._check_program(dsl, "0" * 64, "w")
        self.assertEqual(cm.exception.kind, "sha")
        hashes = [sha(b"a"), sha(b"b")]
        self.assertEqual(P._load_seeds(hashes)[0], frozenset(hashes))
        sp = self.tmp / "seeds.json"
        sp.write_text(json.dumps({"seed_hashes": hashes}))
        self.assertEqual(P._load_seeds(sp)[0], frozenset(hashes))
        with self.assertRaises(P.ReconcileError):
            P._load_seeds(["nothex"])
        raw_entry = {"stats": {**{k: 1 for k in P.SUMMARY_KEYS}, "best_test_rmse": 0.5, "signal_admitted_improvements": 3},
                     "source_ref": {"sha256": "1" * 64, "key": "k"}, "fingerprint": "2" * 64,
                     "generation_zero_source_sha256": "3" * 64, "view": {"context_sha256": cid, "tabpfn35": {"rmse": 1}}}
        proj = P._project_cursor_entry(raw_entry)
        self.assertEqual(proj, {"stats": {k: 1 for k in P.SUMMARY_KEYS}, "source_ref_sha256": "1" * 64,
                                "fingerprint": "2" * 64, "generation_zero_source_sha256": "3" * 64,
                                "view_context_sha256": cid})
        self.assertEqual(P._project_cursor_entry(proj), proj)
        target = self.tmp / "pub" / "x.bin"
        self.assertFalse(P._publish_immutable(target, b"abc")["existed"])
        self.assertTrue(P._publish_immutable(target, b"abc")["existed"])
        self.assertEqual(target.stat().st_mode & 0o777, 0o444)
        with self.assertRaises(common.ImmutableConflict):
            P._publish_immutable(target, b"abd")
        # layout helpers, summaries and the seed pool over a World
        w = self.world("contract", SMALL[:2])
        plan = P._plan(w.epoch, w.pmap)
        self.assertEqual(list(plan.contexts),
                         sorted(plan.contexts, key=lambda c: (P.PARTITIONS.index(c.split_role), c.context_sha256)))
        self.assertEqual({c.context_sha256 for c in plan.contexts}, {sc.cid for sc in w.contexts.values()})
        self.assertEqual(plan.excluded["counts"], {"frozen:role:sealed": 1, "incoming:epoch:withdrawn": 1})
        self.assertEqual(plan.epoch_pin["sha256"], w.epoch_sha256)
        cursor, pin = P.load_incoming_cursor(w.sources.incoming_cursor)
        self.assertEqual(set(pin), {"path", "layout", "bytes", "sha256", "read_utc_us"})
        for c in plan.contexts:
            sc = next(x for x in w.contexts.values() if x.cid == c.context_sha256)
            with self.subTest(context=sc.spec.name):
                self.assertEqual(P._ctx_dir(c), w.paths[sc.spec.name]["dir"])
                self.assertEqual(P._transport_path(c, w.sources), w.paths[sc.spec.name]["transport"])
                s, spin = P._summary(c, cursor if c.cohort == "incoming" else None, pin)
                self.assertEqual({k: getattr(s, k) for k in P.SUMMARY_KEYS}, sc.stats())
                self.assertEqual(s.history_sha256, sc.history_sha256)
                if c.cohort == "incoming":
                    self.assertEqual(P._item_gp(c), w.paths[sc.spec.name]["outcome"].parent)
                    self.assertEqual((s.fingerprint, s.bootstrap_source_sha256),
                                     (sha(sc.native_bytes + sc.outcome_bytes + sc.transport_bytes + sc.boot_results_bytes),
                                      sha(sc.boot_results_bytes)))
                else:
                    self.assertEqual(spin["sha256"], common.sha256_file(w.paths[sc.spec.name]["summary"]))
                pool, _pool_pin = P._seed_pool(c)
                self.assertEqual(pool, {p for p, e in sc.expect.items() if e["origin"] == "bootstrap" or e.get("bootstrap")})
                self.assertEqual(P._rehash({"layout": "file_prefix", "path": str(w.paths[sc.spec.name]["native"]),
                                            "bytes": len(sc.native_bytes)}, c.context_sha256), sha(sc.native_bytes))


if __name__ == "__main__":
    unittest.main()
