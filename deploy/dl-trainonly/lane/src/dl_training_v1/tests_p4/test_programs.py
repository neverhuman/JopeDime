"""Offline proofs for dltrain.programs (PLAN §5 P4a program dictionary).

Three fixtures:
- a synthetic world (frozen and incoming context trees written under RUN/tmp, with poisonable test fields and
  conditioned prior champions in the frozen seed pools) for the reconcile, allowlist, tamper, growth, resume,
  fallback and role-guard cases and the leakage-review cases (sealed origin / conditioned_seed side table, read
  roles and the access journal, unlisted or substituted chunks, cursor projection, the bootstrap-results
  cross-check, the frozen fold-0 diag guard);
- six real contexts from the pinned epoch (3 frozen in folds 0-2, 3 incoming in folds 2-4) built with a fake
  partition map;
- ten real frozen contexts (2 per fold) for the conditioned-seed sealing on real data.
Real JSONL is only read by dltrain.programs' projections; nothing here reads or copies a test-derived field.

Run only inside the lane slice: python -m unittest tests_p4.test_programs -v
"""
import contextlib
import dataclasses
import hashlib
import io
import json
import random
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq

from dltrain import programs as P

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
TMP_ROOT = RUN / "tmp"
EPOCH = RUN / "epochs" / "epoch-000.json"
FORBIDDEN_KEY_PARTS = ("test_rmse", "fitness", "best_test", "signature", "rank", "source_evidence", "tabpfn", "rmse", "ratio",
                       "prediction", "selected", "improvement", "cv_", "error_f32")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dump(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def _prog(dsl: str, op: str = "typed_mutation_v4") -> dict:
    return {"birth_seq": 1, "dsl": dsl, "operator": op, "parents": [], "program_sha256": _sha(dsl)}


def _flip(hexstr: str) -> str:
    return hexstr[:-1] + ("0" if hexstr[-1] != "0" else "1")


BROAD = ['(pipe X (standardize) (ridge_gcv))', '(sym affine "(c0*-0.5)")',
         '(portfolio (member (pipe X (norm_row) (krr))) (member (pipe X (ridge_gcv))) (blend folds=5 guard=clip kind=nnls repeats=1))']
NATIVE_STATUS = ["completed", "completed", "completed", "failed", "completed", "timeout", ("abandoned", "transport_error"),
                 "completed", ("abandoned", "capacity_deferred_before_claim"), "completed", "completed", "completed"]
BOOT_STATUS = ["completed", "failed", "completed", "missing"]
# name, cohort, fold, role, family, natives, own seeds, transport-only programs, own seed is also native, item.
# Frozen own seeds are conditioned prior champions (jope.regression-research-conditioned-seed/v1): F0 has one
# 'missing' and one 'completed', F1's overlaps a native program.
SPECS = [("F0", "frozen", 0, "train_fit", "fam-a", 12, 2, 2, False, None),
         ("F1", "frozen", 1, "heldout", "fam-b", 9, 1, 0, True, None),
         ("I0", "incoming", 2, "train_fit", "fam-c", 10, 1, 1, False, "A"),
         ("I1", "incoming", 3, "dev", "fam-c", 8, 0, 0, False, "A"),
         ("I2", "incoming", 4, "diag", "fam-d", 7, 1, 0, False, "B")]


def _stats(native: bytes, outcomes: bytes, transports: bytes) -> dict:
    """pack_fold_history.build_history's counts, recomputed over synthetic sources."""
    progs = set()
    for line in outcomes.splitlines():
        r = json.loads(line)
        if r.get("program"):
            progs.add(r["program_sha256"])
    for line in native.splitlines():
        digest = json.loads(line).get("program", {}).get("program_sha256")
        if digest:
            progs.add(digest)
    for line in transports.splitlines():
        r = json.loads(line)
        if r.get("program"):
            progs.add(r["program_sha256"])
    return {"unique_programs": len(progs), "native_events": len(native.splitlines()), "native_bytes": len(native),
            "outcome_bytes": len(outcomes), "outcome_events": len(outcomes.splitlines()),
            "transport_bytes": len(transports), "transport_events": len(transports.splitlines())}


class World:
    """A synthetic epoch, partition map, seed list and context trees whose summaries follow the packer exactly."""

    def __init__(self, root: Path, poison: int = 0):
        self.root, self.poison = root, poison
        self.ids = {name: _sha(f"ctx:{name}") for name, *_ in SPECS}
        self.ids.update({"X": _sha("ctx:X"), "S": _sha("ctx:S"), "Z": _sha("ctx:Z")})
        self.sources = P.Sources(root / "incoming" / "fold-archive-cursor-v2.json", root / "incoming" / "transport-index")
        self.sources.incoming_transport_index.mkdir(parents=True)
        self.paths: dict[str, dict[str, Path]] = {}
        self.expect: dict[str, dict] = {}
        self._ctx_events: dict[str, list[bytes]] = {}
        entries, cursor, events = [], {}, {"A": [], "B": []}
        for spec in SPECS:
            entries.append(self._context(spec, events))
        for item, per_ctx in events.items():
            # the item journal interleaves its contexts line by line, with a stranger context mixed in
            stranger = [_dump({"phase": "evolution", "record_status": "completed", "context_sha256": self.ids["Z"],
                               "program": _prog(f"(pipe X (ridge lambda_frac=0.{k + 1}))"),
                               "program_sha256": _sha(f"(pipe X (ridge lambda_frac=0.{k + 1}))")}) for k in range(3)]
            merged = []
            for k in range(max(len(x) for x in per_ctx)):
                merged.extend(x[k] for x in per_ctx if k < len(x))
                if k % 4 == 0 and stranger:
                    merged.append(stranger.pop())
            self._gp(item).joinpath("hwm-events.jsonl").write_bytes(b"".join(merged + stranger))
        for e in entries:
            if e["cohort"] == "incoming":
                lines = self._ctx_events[e["context_sha256"]]
                outcomes = b"".join(lines)
                native = Path(e["native_source"]["path"]).read_bytes()
                transports = self.paths[self._name(e)]["transport"].read_bytes()
                boot = self.paths[self._name(e)]["boot"].read_bytes()
                st = _stats(native, outcomes, transports)
                cursor[e["context_sha256"]] = {
                    "schema": "jope.incoming-fold-telemetry-pointer/v2", "stats": {**st, "best_test_rmse": 0.25 + poison},
                    "source_ref": {"bucket": "b", "key": "k", "version_id": "v", "sha256": e["history_ref"]["sha256"], "bytes": 1},
                    "fingerprint": hashlib.sha256(native + outcomes + transports + boot).hexdigest(),
                    "generation_zero_source_sha256": hashlib.sha256(boot).hexdigest(),
                    "view": {"context_sha256": e["context_sha256"], "tabpfn35": {"rmse": 1.0 + poison}}}
                self.expect[self._name(e)]["summary"] = st
        cursor[self.ids["Z"]] = {"stats": {}, "view": {"context_sha256": self.ids["Z"]}}
        self.sources.incoming_cursor.write_bytes(json.dumps(cursor).encode())
        entries.append({"cohort": "incoming", "status": "excluded", "exclusion_reason": "withdrawn", "context_sha256": self.ids["X"]})
        entries.append({"cohort": "frozen", "status": "included", "exclusion_reason": None, "context_sha256": self.ids["S"]})
        self.epoch = root / "epoch.json"
        self.epoch.write_bytes(json.dumps({"schema": "jope.dl-corpus-epoch/v1", "experiment_id": "synthetic", "epoch": 0,
                                           "contexts": entries}).encode())
        pmap = {self.ids[name]: {"family_id": fam, "split_role": role} for name, _, _, role, fam, *_ in SPECS}
        pmap[self.ids["S"]] = {"family_id": "fam-s", "split_role": "sealed"}
        pmap[self.ids["X"]] = {"family_id": "fam-x", "split_role": "train_fit"}
        pmap[_sha("ctx:not-in-epoch")] = {"family_id": "fam-y", "split_role": "dev"}
        self.pmap = root / "partition-map.json"
        self.pmap.write_bytes(json.dumps(pmap).encode())
        self.seed_set = frozenset([_sha(d) for d in BROAD] + [_sha("(pipe X (never_seen))")])
        self.seeds = root / "seeds.json"
        self.seeds.write_bytes(json.dumps(sorted(self.seed_set)).encode())

    def _name(self, e: dict) -> str:
        return next(n for n, cid in self.ids.items() if cid == e["context_sha256"])

    def _gp(self, item: str) -> Path:
        return self.root / "incoming" / "items" / _sha(f"item:{item}") / "run" / "gp"

    def _context(self, spec, events) -> dict:
        name, cohort, fold, role, fam, n_native, n_own, n_tonly, own_is_native, item = spec
        cid, poison = self.ids[name], self.poison
        dataset = cid[:16]
        if cohort == "frozen":
            view = f"{dataset}__r0__f{fold}"
            d = self.root / "frozen" / "gp" / "contexts" / view
        else:
            view = f"{dataset}__r0__f{fold}__test_adaptive_{cid[:12]}"
            d = self._gp(item) / "contexts" / view
        d.mkdir(parents=True)
        natives = [f"(pipe X (winsor_clip q=0.0{k % 9 + 1}) (histgbm lr=0.{k + 1} seed={fold}{k}))" for k in range(n_native)]
        own = [natives[0]] if own_is_native else [f"(pipe X (robust_scale) (ridge lambda_frac=0.{fold}{k}))" for k in range(n_own)]
        seeds = BROAD + own
        tonly = [f"(pipe X (norm_row) (krr alpha=0.{fold}{k}))" for k in range(n_tonly)]
        native = b""
        for k, dsl in enumerate(natives):
            st = NATIVE_STATUS[k % len(NATIVE_STATUS)]
            if isinstance(st, tuple):
                native += _dump({"schema": "jope.island-abandonment/v1", "status": st[0], "reason": st[1], "context_sha256": cid,
                                 "program_sha256": _sha(dsl), "program": _prog(dsl), "marker": f"AA{name}"})
            else:
                native += _dump({"schema": "jope.island-scored-evaluation/v1", "status": st, "context_sha256": cid,
                                 "program": _prog(dsl), "test_rmse": 0.5 + poison + k / 1000, "fitness": 1.5 + poison,
                                 "semantic_signature_f32_b64": f"SIG{poison}{k}", "policy": "test_gated", "marker": f"AA{name}"})
        outcome_lines = []
        for k, dsl in enumerate(seeds):
            outcome_lines.append(_dump({"phase": "bootstrap", "record_status": BOOT_STATUS[k % 4], "context_sha256": cid, "view_id": view,
                                        "program": _prog(dsl, "historical_seed"), "program_sha256": _sha(dsl),
                                        "test_rmse": 0.75 + poison, "fitness": 2.0 + poison}))
        if name == "F0":         # a retried seed: the last outcome row wins
            outcome_lines.append(_dump({"phase": "bootstrap", "record_status": "completed", "context_sha256": cid, "view_id": view,
                                        "program": _prog(seeds[1], "historical_seed"), "program_sha256": _sha(seeds[1])}))
        for k, dsl in enumerate(natives):
            st = NATIVE_STATUS[k % len(NATIVE_STATUS)]
            outcome_lines.append(_dump({"phase": "evolution", "record_status": "missing" if isinstance(st, tuple) else st,
                                        "context_sha256": cid, "program": _prog(dsl), "program_sha256": _sha(dsl),
                                        "test_rmse": 0.5 + poison, "marker": f"BB{name}"}))
        transport = b""
        for k, dsl in enumerate(natives[:4] + tonly + tonly[:1]):
            row = {"context_sha256": cid, "program": _prog(dsl), "program_sha256": _sha(dsl), "wall_ns": 1000 + k, "marker": f"CC{name}"}
            if cohort == "frozen":
                row = {"_source_transport_journal": "hybrid-events.jsonl", "_telemetry_view_context_sha256": cid, **row}
            transport += _dump(row)
        transport += _dump({"context_sha256": cid, "attempt_id": "no-program", "wall_ns": 7})
        seed_pool = b"".join(_dump({"schema": "jope.seed/v1", "rank": 10 * poison + k, "source_evidence": [f"EV{poison}"],
                                    "program": _prog(dsl, "historical_seed")}) for k, dsl in enumerate(BROAD))
        conditioned = cohort == "frozen"
        seed_pool += b"".join(_dump({"schema": P.CONDITIONED_SEED_SCHEMA if conditioned else "jope.seed/v1",
                                     "rank": 10 * poison + k, "program": _prog(dsl, "historical_seed"),
                                     **({"source_context": _sha(f"fold0:{poison}")} if conditioned else {})})
                              for k, dsl in enumerate(own))
        last_boot = {}
        for line in outcome_lines:
            r = json.loads(line)
            if r.get("phase") == "bootstrap":
                last_boot[r["program_sha256"]] = r["record_status"]
        nfile = d / f"evaluations-{cid[:16]}.jsonl"
        nfile.write_bytes(native)
        (d / "seed-pool.jsonl").write_bytes(seed_pool)
        paths = {"native": nfile, "seed_pool": d / "seed-pool.jsonl"}
        if cohort == "frozen":
            outcomes = b"".join(outcome_lines)
            for fname, body in (("history-outcomes.jsonl", outcomes), ("history-transports.jsonl", transport)):
                (d / fname).write_bytes(body)
            # the dispatch journal: one row per bootstrap outcome row (a retry appends a row; the last one wins)
            (d / "bootstrap-results.jsonl").write_bytes(b"".join(
                _dump({"program_sha256": json.loads(x)["program_sha256"], "record_status": json.loads(x)["record_status"],
                       "status": "cached", "context_sha256": cid, "result_key": f"k{k}"})
                for k, x in enumerate(outcome_lines) if json.loads(x).get("phase") == "bootstrap"))
            paths["boot"] = d / "bootstrap-results.jsonl"
            st = _stats(native, outcomes, transport)
            (d / "history-ref.json").write_bytes(json.dumps({
                "schema": "jope.per-fold-history-ref/v1", "view_id": view, "context_sha256": cid, "outer_fold": fold,
                "history_ref": {"bucket": "b", "key": "k", "version_id": "v", "sha256": _sha(f"history:{name}"), "bytes": 1},
                "best_test_rmse": 0.125 + poison, "signal_admitted_improvements": poison, **st}).encode())
            paths.update(outcome=d / "history-outcomes.jsonl", transport=d / "history-transports.jsonl", summary=d / "history-ref.json")
            self.expect[name] = {"summary": st}
        else:
            boot = b"".join(_dump({"program_sha256": _sha(dsl), "record_status": BOOT_STATUS[k % 4], "status": "completed_transport",
                                   "context_sha256": cid}) for k, dsl in enumerate(seeds))
            (d / "bootstrap-results.jsonl").write_bytes(boot)
            tpath = self.sources.incoming_transport_index / f"{cid}.jsonl"
            tpath.write_bytes(transport)
            events[item].append(outcome_lines)
            self._ctx_events[cid] = outcome_lines
            paths.update(outcome=self._gp(item) / "hwm-events.jsonl", transport=tpath, boot=d / "bootstrap-results.jsonl")
            self.expect[name] = {}
        self.paths[name] = paths
        self.expect[name].update(native=len(natives), boot=len(seeds), overlap=int(own_is_native), tonly=len(tonly),
                                 rows=len(set(natives) | set(seeds) | set(tonly)), role=role, cohort=cohort,
                                 conditioned={_sha(x) for x in own} if conditioned else set(),
                                 boot_status={k: v for k, v in last_boot.items()})
        entry = {"cohort": cohort, "status": "included", "exclusion_reason": None, "context_sha256": cid, "view_id": view,
                 "dataset_hash": dataset, "fold": fold,
                 "history_ref": {"bucket": "b", "key": "k", "version_id": "v", "sha256": _sha(f"history:{name}"), "bytes": 1},
                 "native_source": {"path": str(nfile), "bytes": len(native), "sha256": hashlib.sha256(native).hexdigest()},
                 "prepared_ref": {"path": str(d / "prepared.json"), "sha256": "0" * 64, "bytes": 0}}
        if item:
            entry["item_id"] = _sha(f"item:{item}")
        return entry

    def replace_once(self, path: Path, old: bytes, new: bytes) -> None:
        body = path.read_bytes()
        assert len(old) == len(new) and old in body, (path, old)
        path.write_bytes(body.replace(old, new, 1))

    def bump_unique(self, name: str) -> None:
        doc = json.loads(self.paths[name]["summary"].read_bytes())
        doc["unique_programs"] += 1
        self.paths[name]["summary"].write_bytes(json.dumps(doc).encode())

    def append(self, path: Path, body: bytes) -> None:
        with open(path, "ab") as f:
            f.write(body)


def _chunks(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*.parquet"))}


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield ("key", f"{path}.{k}", k)
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield ("value", path, obj)


class _Asserts:
    def assert_no_test_fields(self, out: Path) -> None:
        """Every output is the column allowlist plus counts and pins: no test-named key and no float anywhere. The
        published chunks are exactly ROW_SCHEMA (no origin); the sealed twins exactly SEALED_SCHEMA."""
        files = list(out.rglob("*.parquet"))
        self.assertTrue(files)
        for f in files:
            pf = pq.ParquetFile(f)
            schema = P.SEALED_SCHEMA if f.relative_to(out).parts[0] == P.SEALED_DIR else P.ROW_SCHEMA
            self.assertEqual(pf.schema_arrow.names, schema.names)
            self.assertTrue(pf.schema_arrow.equals(schema, check_metadata=False))
            self.assertEqual(set((pf.metadata.metadata or {}).keys()), {b"ARROW:schema"})
            if schema is P.ROW_SCHEMA:
                self.assertNotIn("origin", pf.schema_arrow.names)
        for name in ("manifest.json", "cursor.json"):
            doc = json.loads((out / name).read_bytes())
            for kind, where, v in _walk(doc):
                if kind == "key":
                    low = v.lower()
                    self.assertFalse(any(part in low for part in FORBIDDEN_KEY_PARTS), f"{name}{where}")
                else:
                    self.assertNotIsInstance(v, float, f"{name}{where}")


class SyntheticTests(_Asserts, unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="p4a-synthetic-", dir=TMP_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def world(self, name: str, poison: int = 0) -> World:
        return World(self.tmp / name, poison)

    def build(self, w: World, out: str, **kw) -> dict:
        kw.setdefault("chunk_contexts", 2)
        return P.build(w.epoch, w.pmap, w.seeds, self.tmp / out, sources=w.sources, **kw)

    def test_build_reconciles_labels_and_verifies(self):
        w = self.world("w")
        m = self.build(w, "out")
        recs = m["contexts"]
        self.assertEqual(set(recs), {w.ids[n] for n, *_ in SPECS})
        for name, *_ in SPECS:
            r, e = recs[w.ids[name]], w.expect[name]
            with self.subTest(context=name):
                self.assertEqual(r["rows"], e["rows"])
                self.assertEqual(r["rows"], r["summary"]["unique_programs"])
                self.assertEqual(r["native_rows"], e["native"])
                self.assertEqual(r["native_rows"], r["summary"]["native_events"])
                self.assertEqual((r["bootstrap_set"], r["seed_pool"]), (e["boot"], e["boot"]))
                self.assertEqual(r["bootstrap_overlap_native"], e["overlap"])
                self.assertEqual(r["bootstrap_rows"], e["boot"] - e["overlap"])
                self.assertEqual(r["transport_only_rows"], e["tonly"])
                self.assertEqual(r["split_role"], e["role"])
                self.assertEqual(r["shared_seed_rows"], len(BROAD))
                self.assertEqual(r["prefixes"]["outcome"]["pin"], "bytes+lines" if e["cohort"] == "frozen" else "fingerprint")
        f0, f1 = recs[w.ids["F0"]], recs[w.ids["F1"]]
        self.assertEqual(f0["bootstrap_events"], f0["bootstrap_set"] + 1)      # the retried seed counted once
        self.assertEqual(f0["by_status"]["transport_only"], 2)
        self.assertEqual(f1["prefixes"]["transport"]["skipped"], "residual_zero")
        self.assertIn("fingerprint", recs[w.ids["I1"]]["prefixes"]["transport"]["pin"])
        self.assertEqual(m["totals"]["rows"], sum(w.expect[n]["rows"] for n, *_ in SPECS))
        self.assertEqual(m["totals"]["rows"], m["totals"]["unique_programs"])
        self.assertEqual(m["excluded"], {"counts": {"frozen:role:sealed": 1, "incoming:epoch:withdrawn": 1}, "map_entries_outside_epoch": 1})
        self.assertEqual([m["partitions"][p]["contexts"] for p in P.PARTITIONS], [2, 1, 1, 1])
        self.assertEqual(len(m["partitions"]["train_fit"]["chunks"]), 1)
        rep = P.verify(self.tmp / "out", rehash_sources=True, sources=w.sources)
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        self.assertEqual(rep["checked"]["rows"], m["totals"]["rows"])
        self.assertGreater(rep["checked"]["prefixes_rehashed"], 0)
        self.assertEqual(rep["checked"]["summaries_reconciled"], len(SPECS))
        self.assert_no_test_fields(self.tmp / "out")
        # a chunk changed after the build is caught by verify
        target = self.tmp / "out" / m["partitions"]["dev"]["chunks"][0]["path"]
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"\0")
        self.assertEqual(P.verify(self.tmp / "out", sources=w.sources)["verdict"], "FAIL")

    def test_poisoned_test_fields_leave_chunks_byte_identical(self):
        a, b = self.world("a", poison=0), self.world("b", poison=7)
        self.assertNotEqual(a.paths["F0"]["native"].read_bytes(), b.paths["F0"]["native"].read_bytes())
        self.build(a, "out-a")
        self.build(b, "out-b")
        ca, cb = _chunks(self.tmp / "out-a"), _chunks(self.tmp / "out-b")
        self.assertEqual(len(ca), 8)                      # 4 published chunks and their 4 sealed twins
        self.assertEqual(ca, cb)
        self.assert_no_test_fields(self.tmp / "out-b")

    def test_summary_projection_is_the_allowlist(self):
        fields = {"unique_programs", "native_events", "native_bytes", "outcome_bytes", "outcome_events", "transport_bytes",
                  "transport_events", "history_sha256", "fingerprint", "bootstrap_source_sha256"}
        a, b = self.world("a", poison=0), self.world("b", poison=3)
        ca, cb = ({c.view_id[:16]: c for c in P.load_contexts(w.epoch, w.pmap)} for w in (a, b))
        raw_a = json.loads(a.sources.incoming_cursor.read_bytes())
        raw_b = json.loads(b.sources.incoming_cursor.read_bytes())
        for key in ca:
            sa = P.history_summary(ca[key], raw_a)
            sb = P.history_summary(cb[key], raw_b)
            self.assertEqual(set(dataclasses.asdict(sa)), fields)
            self.assertEqual(sa.unique_programs, sb.unique_programs)
            self.assertEqual(sa.history_sha256, sb.history_sha256)
        projected, pin = P.load_incoming_cursor(a.sources.incoming_cursor)
        for entry in projected.values():
            self.assertEqual(set(entry), {"stats", "source_ref_sha256", "fingerprint", "generation_zero_source_sha256", "view_context_sha256"})
            self.assertEqual(tuple(entry["stats"]), P.SUMMARY_KEYS)
        self.assertEqual(pin["sha256"], hashlib.sha256(a.sources.incoming_cursor.read_bytes()).hexdigest())

    def test_tampered_sources_stop(self):
        cases = [  # (label, tampered context, expected STOP kind, tamper)
            ("native byte inside the pinned prefix", "F1", "source", lambda w: w.replace_once(w.paths["F1"]["native"], b"AAF1", b"AAF9")),
            ("frozen outcome dsl edited in place", "F0", "sha",
             lambda w: w.replace_once(w.paths["F0"]["outcome"], b"(robust_scale)", b"(robust_scalf)")),
            ("frozen outcome prefix truncated", "F0", "source",
             lambda w: w.paths["F0"]["outcome"].write_bytes(w.paths["F0"]["outcome"].read_bytes()[:-1])),
            ("frozen transport prefix truncated", "F0", "source",
             lambda w: w.paths["F0"]["transport"].write_bytes(w.paths["F0"]["transport"].read_bytes()[:-10])),
            ("incoming event edited in place", "I0", "source", lambda w: w.replace_once(w.paths["I0"]["outcome"], b"BBI0", b"BBI9")),
            ("incoming transport edited in place (residual 0)", "I1", "source",
             lambda w: w.replace_once(w.paths["I1"]["transport"], b"CCI1", b"CCI9")),
            ("incoming bootstrap-results edited", "I2", "source",
             lambda w: w.replace_once(w.paths["I2"]["boot"], b"completed_transport", b"completed_transpore")),
            ("seed pool gains a program", "I2", "seed_pool",
             lambda w: w.append(w.paths["I2"]["seed_pool"], _dump({"program": _prog("(pipe X (extra))")}))),
            ("summary unique_programs off by one", "F0", "count", lambda w: w.bump_unique("F0")),
        ]
        for i, (label, target, kind, tamper) in enumerate(cases):
            with self.subTest(label):
                w = self.world(f"t{i}")
                tamper(w)
                with self.assertRaises(P.ReconcileError) as cm:
                    self.build(w, f"out{i}")
                self.assertEqual((cm.exception.context, cm.exception.kind), (w.ids[target], kind))

    def test_grown_sources_read_to_recorded_prefix(self):
        a, b = self.world("a"), self.world("b")
        extra = _dump({"status": "completed", "context_sha256": b.ids["F0"], "program": _prog("(pipe X (late))")})
        b.append(b.paths["F0"]["native"], extra)
        b.append(b.paths["F0"]["outcome"], extra)
        b.append(b.paths["F0"]["transport"], extra)
        b.append(b.paths["I0"]["outcome"], _dump({"phase": "bootstrap", "record_status": "completed", "context_sha256": b.ids["I0"],
                                                  "program": _prog("(pipe X (late))"), "program_sha256": _sha("(pipe X (late))")}))
        b.append(b.paths["I0"]["outcome"], b'{"phase":"evolution","context_sha256":"' + b.ids["I1"].encode())   # a write in progress
        b.append(b.paths["I0"]["transport"], extra)
        self.build(a, "out-a")
        mb = self.build(b, "out-b")
        self.assertEqual(_chunks(self.tmp / "out-a"), _chunks(self.tmp / "out-b"))
        self.assertGreater(mb["contexts"][b.ids["F0"]]["prefixes"]["native"]["file_bytes"], mb["contexts"][b.ids["F0"]]["prefixes"]["native"]["bytes"])
        self.assertEqual(mb["contexts"][b.ids["I0"]]["prefixes"]["outcome"]["tail_lines"], 1)

    def test_resume_after_crash_is_byte_identical(self):
        w = self.world("w")
        ref = self.build(w, "ref", chunk_contexts=1)
        calls = {"n": 0}
        real_write = P._write_cursor

        def crash_after_second(out, cursor):
            real_write(out, cursor)
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash after the second chunk")

        with mock.patch.object(P, "_write_cursor", crash_after_second), self.assertRaises(RuntimeError):
            self.build(w, "run", chunk_contexts=1)
        self.assertFalse((self.tmp / "run" / "manifest.json").exists())
        seen = []
        real_rows = P.context_rows

        def counting(ctx, *a, **k):
            seen.append(ctx.context_sha256)
            return real_rows(ctx, *a, **k)

        with mock.patch.object(P, "context_rows", counting):
            m = self.build(w, "run", chunk_contexts=1)
        self.assertEqual(len(seen), len(SPECS) - 2)
        self.assertEqual(_chunks(self.tmp / "run"), _chunks(self.tmp / "ref"))
        self.assertEqual(m["partitions"], ref["partitions"])
        self.assertEqual(m["totals"], ref["totals"])
        self.assertEqual(len(m["inputs"]["incoming_cursor_reads"]), 2)
        again = self.build(w, "run", chunk_contexts=1)                       # a finished build republishes the same manifest
        self.assertEqual(again, m)
        with self.assertRaises(P.ReconcileError):
            self.build(w, "run", chunk_contexts=1, resume=False)
        with self.assertRaises(P.ReconcileError):
            self.build(w, "run", chunk_contexts=2)                            # other settings than the cursor's pins
        self.assertEqual(P.verify(self.tmp / "run", rehash_sources=True, sources=w.sources)["verdict"], "PASS")

    def test_role_guard_closure_and_iter_distinct(self):
        w = self.world("w")
        m = self.build(w, "out")
        out = self.tmp / "out"
        with self.assertRaises(PermissionError):
            P.open_partition(out, "heldout")
        with self.assertRaises(PermissionError):
            list(P.iter_distinct(out, "heldout"))
        broad = {bytes.fromhex(_sha(d)) for d in BROAD}
        excluded = {bytes.fromhex(w.ids[n]) for n in ("X", "S", "Z")}
        for p in P.PARTITIONS:
            role = P.PARTITION_ROLE[p]
            t = P.open_partition(out, p, role=role).to_table(use_threads=False)
            self.assertEqual(t.num_rows, m["partitions"][p]["rows"])
            self.assertEqual(set(t.column("split_role").to_pylist()), {p})
            ctxs = set(t.column("context_sha256").to_pylist())
            self.assertEqual(ctxs, {bytes.fromhex(w.ids[n]) for n, _, _, r, *_ in SPECS if r == p})
            self.assertFalse(ctxs & excluded)
            shared = {s for s, flag in zip(t.column("program_sha256").to_pylist(), t.column("shared_seed").to_pylist()) if flag}
            self.assertEqual(shared, broad)                                       # a shared seed is copied into every partition
            pairs = list(P.iter_distinct(out, p, role=role))
            self.assertEqual(pairs, sorted(set(pairs)))
            self.assertEqual(len(pairs), m["partitions"][p]["distinct_programs"])
            self.assertEqual(set(pairs), {(s.hex(), d) for s, d in zip(t.column("program_sha256").to_pylist(), t.column("dsl").to_pylist())})
            for sha, dsl in pairs:
                self.assertEqual(hashlib.sha256(dsl.encode()).hexdigest(), sha)

    def test_s3_fallback_only_after_mismatch(self):
        w = self.world("w")
        ref = self.build(w, "ref")
        f1 = next(c for c in P.load_contexts(w.epoch, w.pmap) if c.context_sha256 == w.ids["F1"])
        s1 = P.history_summary(f1)
        clean, _ = P.context_rows(f1, s1, seeds=w.seed_set, sources=w.sources)
        calls = []

        def stub(ctx, err):
            calls.append((ctx.context_sha256, err.kind))
            return clean, s1

        self.build(w, "clean", s3_fallback=stub)
        self.assertEqual(calls, [])
        w.replace_once(w.paths["F1"]["native"], b"AAF1", b"AAF9")
        m = self.build(w, "fell-back", s3_fallback=stub)
        self.assertEqual(calls, [(w.ids["F1"], "source")])
        self.assertEqual(m["contexts"][w.ids["F1"]]["source"], "s3_fallback")
        self.assertEqual(m["totals"]["sources"], {"local": len(SPECS) - 1, "s3_fallback": 1})
        self.assertEqual(_chunks(self.tmp / "fell-back"), _chunks(self.tmp / "ref"))
        self.assertEqual(ref["totals"]["rows"], m["totals"]["rows"])
        with self.assertRaises(P.ReconcileError):                              # a fallback must match its own summary
            self.build(w, "bad-fallback", s3_fallback=lambda ctx, err: (clean[:-1], s1))

    def test_cli_build_verify_and_stop(self):
        w = self.world("w")
        common_args = ["--incoming-cursor", str(w.sources.incoming_cursor), "--incoming-transport-index", str(w.sources.incoming_transport_index)]
        build = ["build", "--epoch", str(w.epoch), "--partition-map", str(w.pmap), "--seeds", str(w.seeds), "--chunk-contexts", "2", *common_args]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(P.main(build + ["--out", str(self.tmp / "cli")]), 0)
            self.assertEqual(P.main(["verify", "--out", str(self.tmp / "cli"), "--rehash-sources"]), 0)
            self.assertEqual(P.main(build + ["--out", str(self.tmp / "cli"), "--no-resume"]), 2)
            w.replace_once(w.paths["I2"]["boot"], b"completed_transport", b"completed_transpore")
            self.assertEqual(P.main(build + ["--out", str(self.tmp / "cli-tampered")]), 2)
        built = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual((built["verdict"], built["totals"]["rows"]), ("BUILT", sum(w.expect[n]["rows"] for n, *_ in SPECS)))
        self.assertEqual(json.loads(out.getvalue().splitlines()[1])["verdict"], "PASS")
        stops = [json.loads(line) for line in err.getvalue().splitlines()]
        self.assertEqual([(x["verdict"], x["kind"]) for x in stops], [("STOP", "cursor"), ("STOP", "source")])
        self.assertEqual(stops[1]["context"], w.ids["I2"])

    def test_partition_map_and_seed_validation(self):
        w = self.world("w")
        pmap = json.loads(w.pmap.read_bytes())
        missing = {k: v for k, v in pmap.items() if k != w.ids["I2"]}
        with self.assertRaises(P.ReconcileError) as cm:
            P.load_contexts(w.epoch, missing)
        self.assertEqual(cm.exception.kind, "closure")
        with self.assertRaises(P.ReconcileError):
            P.load_contexts(w.epoch, {**pmap, w.ids["I2"]: {"family_id": "fam-d", "split_role": "test"}})
        with self.assertRaises(P.ReconcileError):
            P.build(w.epoch, w.pmap, ["not-a-hash"], self.tmp / "out", sources=w.sources)
        ctxs = P.load_contexts(w.epoch, w.pmap)
        self.assertEqual([c.split_role for c in ctxs], ["train_fit", "train_fit", "dev", "diag", "heldout"])
        self.assertEqual(P.load_seed_hashes(w.seeds), w.seed_set)

    # ---------------------------------------------------------------- leakage-allowlist review (LA-1 .. LA-6)
    def test_la1_origin_and_conditioned_seeds_are_sealed(self):
        """No published column separates the conditioned prior champions (bootstrap and not a P2 seed): origin and
        conditioned_seed live only in the row-aligned sealed side table, the published record_status merges the
        bootstrap-only 'missing' into 'unevaluated', and no cell of published categorical values is purely
        conditioned. The manifest counts conditioned rows per partition."""
        w = self.world("w")
        m = self.build(w, "out")
        out = self.tmp / "out"
        self.assertEqual([c[0] for c in m["columns"]], P.ROW_SCHEMA.names)
        self.assertNotIn("origin", P.ROW_SCHEMA.names)
        self.assertNotIn("conditioned_seed", P.ROW_SCHEMA.names)
        self.assertEqual(set(P.DICT_COLUMNS["record_status"]), {"completed", "failed", "timeout", "unevaluated"})
        want_cond = {p: 0 for p in P.PARTITIONS}
        for name, *_ in SPECS:
            e = w.expect[name]
            want_cond[e["role"]] += len(e["conditioned"])
            self.assertEqual(m["contexts"][w.ids[name]]["conditioned_seed_rows"], len(e["conditioned"]))
        self.assertEqual({p: m["sealed"]["partitions"][p]["conditioned_rows"] for p in P.PARTITIONS}, want_cond)
        self.assertEqual(m["totals"]["conditioned_seed_rows"], 3)
        cond_by_ctx = {w.ids[n]: w.expect[n]["conditioned"] for n, *_ in SPECS}
        for p in P.PARTITIONS:
            pub = P.open_partition(out, p, role=P.PARTITION_ROLE[p]).to_table(use_threads=False)
            sealed = P.open_sealed(out, p, role="exposed_report").to_table(use_threads=False)
            self.assertEqual(sealed.schema.names, P.SEALED_SCHEMA.names)
            self.assertEqual(pub.column("program_sha256").to_pylist(), sealed.column("program_sha256").to_pylist())
            self.assertEqual(pub.column("context_sha256").to_pylist(), sealed.column("context_sha256").to_pylist())
            rows = list(zip(*(pub.column(c).to_pylist() for c in ("context_sha256", "program_sha256", "record_status",
                                                                  "shared_seed", "form", "cohort", "family_id"))))
            flags = sealed.column("conditioned_seed").to_pylist()
            for (cb, pb, *_), flag, origin, exact, st in zip(rows, flags, sealed.column("origin").to_pylist(),
                                                              sealed.column("record_status").to_pylist(),
                                                              pub.column("record_status").to_pylist()):
                self.assertEqual(flag, pb.hex() in cond_by_ctx[cb.hex()])
                self.assertEqual(st, P.published_status(exact))
                self.assertIn(exact, P.STATUSES_BY_ORIGIN[origin])
            cells = {}
            for (cb, _pb, *cat), flag in zip(rows, flags):
                cells.setdefault((cb, *cat), set()).add(flag)
            pure = [c for c, f in cells.items() if f == {True}]
            self.assertEqual(pure, [], p)                          # no published cell holds only champions
        f0 = P.open_sealed(out, "train_fit", role="exposed_report").to_table(use_threads=False)
        exact = dict(zip((x.hex() for x in f0.column("program_sha256").to_pylist()), f0.column("record_status").to_pylist()))
        missing_cond = [s for s in w.expect["F0"]["conditioned"] if exact.get(s) == "missing"]
        self.assertEqual(len(missing_cond), 1)                     # the champion the old 'missing' column exposed
        f1 = P.open_sealed(out, "heldout", role="exposed_report").to_table(use_threads=False)
        both = [o for o, c in zip(f1.column("origin").to_pylist(), f1.column("conditioned_seed").to_pylist()) if c]
        self.assertEqual(both, ["evolution"])                      # a champion that is also native keeps origin evolution
        self.assert_no_test_fields(out)

    def test_la2_roles_are_partition_specific_projected_and_journaled(self):
        w = self.world("w")
        self.build(w, "out")
        out = self.tmp / "out"
        journal = out / P.JOURNAL_NAME
        list(P.iter_distinct(out, "train_fit"))                    # fit on train_fit: not journaled
        self.assertFalse(journal.exists())
        for p, role in P.PARTITION_ROLE.items():
            for other in set(P.PARTITION_ROLE.values()) - {role}:
                with self.assertRaises(PermissionError, msg=(p, other)):
                    P.open_partition(out, p, role=other)
            self.assertEqual(P.open_partition(out, p, role=role).schema.names, P.ROW_SCHEMA.names)
            self.assertEqual(P.open_partition(out, p, role="roundtrip").schema.names, ["program_sha256", "dsl"])
            self.assertEqual(P.open_partition(out, p, role="preflight").schema.names, ["context_sha256", "dsl"])
            for sealed_role in P.SEALED_ROLES:
                with self.assertRaises(PermissionError):
                    P.open_partition(out, p, role=sealed_role)
            with self.assertRaises(PermissionError):
                P.open_sealed(out, p, role=role)
            with self.assertRaises(PermissionError):
                list(P.iter_distinct(out, p, role="preflight"))
            self.assertEqual(list(P.iter_distinct(out, p, role="roundtrip")), list(P.iter_distinct(out, p, role=role)))
            self.assertEqual(P.open_sealed(out, p, role="r7_crosscheck").schema.names,
                             ["context_sha256", "program_sha256", "origin", "record_status"])
        with self.assertRaises(ValueError):
            P.open_partition(out, "train_fit", role="nope")
        entries = [json.loads(x) for x in journal.read_bytes().splitlines()]
        self.assertTrue(entries)
        self.assertFalse([e for e in entries if e["partition"] == "train_fit" and e["role"] == "fit"])
        self.assertTrue(all(e["caller"].startswith("tests_p4.test_programs:") for e in entries), entries[0])
        self.assertTrue({"heldout_report", "roundtrip", "preflight", "r7_crosscheck"} <= {e["role"] for e in entries})
        self.assertEqual({e["table"] for e in entries}, {"published", "sealed"})
        # the R7 index through the guarded API equals context_rows' (program, origin, exact status)
        cursor = P.load_incoming_cursor(w.sources.incoming_cursor)[0]
        idx = P.SealedIndex(out)
        for c in P.load_contexts(w.epoch, w.pmap):
            rows, _ = P.context_rows(c, P.history_summary(c, cursor), seeds=w.seed_set, sources=w.sources)
            self.assertEqual(idx.rows(c.context_sha256), {(r.program_sha256, r.origin, r.record_status) for r in rows})
        with self.assertRaises(PermissionError):
            P.SealedIndex(out, role="fit")
        alt = self.tmp / "journal.jsonl"
        with mock.patch.dict("os.environ", {P.JOURNAL_ENV: str(alt)}):
            P.open_partition(out, "dev", role="roundtrip")
        self.assertEqual(json.loads(alt.read_bytes().splitlines()[0])["role"], "roundtrip")

    def test_la3_unlisted_or_substituted_chunks(self):
        w = self.world("w")
        m = self.build(w, "out")
        out = self.tmp / "out"
        self.assertEqual(P.verify(out, sources=w.sources)["verdict"], "PASS")
        heldout = out / m["partitions"]["heldout"]["chunks"][0]["path"]
        stray = out / "train_fit" / "part-00099.parquet"
        shutil.copyfile(heldout, stray)
        rep = P.verify(out, sources=w.sources)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertTrue(any("part-00099" in f and "does not list" in f for f in rep["failures"]), rep["failures"])
        (out / "train_fit" / ".part-00099.tmp").write_bytes(b"temp")      # dot-prefixed temporaries are ignored
        stray.unlink()
        self.assertEqual(P.verify(out, sources=w.sources)["verdict"], "PASS")
        (out / P.SEALED_DIR / "dev" / "part-00042.parquet").write_bytes(b"x")
        self.assertEqual(P.verify(out, sources=w.sources)["verdict"], "FAIL")
        (out / P.SEALED_DIR / "dev" / "part-00042.parquet").unlink()
        # open_partition checks bytes and sha256 against the manifest ...
        target = out / m["partitions"]["dev"]["chunks"][0]["path"]
        good = target.read_bytes()
        target.chmod(0o644)
        target.write_bytes(heldout.read_bytes())
        with self.assertRaises(P.ReconcileError) as cm:
            P.open_partition(out, "dev", role="training_validation")
        self.assertEqual(cm.exception.kind, "chunk")
        # ... and that a chunk holds only rows of its partition (a manifest re-pointed at the heldout chunk)
        doc = json.loads((out / "manifest.json").read_bytes())
        entry = doc["partitions"]["dev"]["chunks"][0]
        entry.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size)
        (out / "manifest.json").chmod(0o644)
        (out / "manifest.json").write_bytes(json.dumps(doc).encode())
        with self.assertRaises(P.ReconcileError) as cm:
            P.open_partition(out, "dev", role="roundtrip")
        self.assertEqual(cm.exception.kind, "closure")
        target.write_bytes(good)
        # without a cursor, a partition directory may hold only chunk names of the current layout
        w2 = self.world("w2")
        fresh = self.tmp / "fresh"
        (fresh / "dev").mkdir(parents=True)
        (fresh / "dev" / "part-00007.parquet").write_bytes(b"stale")
        with self.assertRaises(P.ReconcileError) as cm:
            self.build(w2, "fresh")
        self.assertEqual(cm.exception.kind, "cursor")
        (fresh / "dev" / "part-00007.parquet").unlink()
        self.build(w2, "fresh")
        self.assertEqual(P.verify(fresh, sources=w2.sources)["verdict"], "PASS")

    def test_la4_raw_cursor_entries_are_always_projected(self):
        w = self.world("w")
        ctx = next(c for c in P.load_contexts(w.epoch, w.pmap) if c.cohort == "incoming")
        raw = json.loads(w.sources.incoming_cursor.read_bytes())[ctx.context_sha256]
        proj = P._project_cursor_entry(raw)
        self.assertEqual(tuple(proj), P.CURSOR_PROJECTION_KEYS)
        self.assertIs(P._project_cursor_entry(proj), proj)
        # a plain dict is raw input, whatever its keys: re-projected (it has no source_ref and no view)
        self.assertEqual(P._project_cursor_entry(dict(proj)), {**proj, "source_ref_sha256": None, "view_context_sha256": None})
        # a producer-side flat key can neither switch the projection off nor replace the history pin
        hostile = {**raw, "source_ref_sha256": ctx.history_sha256, "source_ref": {**raw["source_ref"], "sha256": "b" * 64}}
        got = P._project_cursor_entry(hostile)
        self.assertEqual(tuple(got), P.CURSOR_PROJECTION_KEYS)
        self.assertNotIn("best_test_rmse", json.dumps(got))
        self.assertEqual(got["source_ref_sha256"], "b" * 64)
        with self.assertRaises(P.ReconcileError) as cm:
            P.history_summary(ctx, {ctx.context_sha256: hostile})
        self.assertEqual(cm.exception.kind, "summary")
        self.assertEqual(P.history_summary(ctx, {ctx.context_sha256: raw}).history_sha256, ctx.history_sha256)

    def test_la5_frozen_bootstrap_results_cross_check(self):
        # one same-length edit of a projected field in a frozen outcome prefix: counted, within the tolerated one
        w = self.world("one")
        w.replace_once(w.paths["F0"]["outcome"], b'"record_status":"missing"', b'"record_status":"timeout"')
        m = self.build(w, "out-one")
        r = m["contexts"][w.ids["F0"]]
        self.assertEqual(r["bootstrap_results"]["disagreements"], 1)
        self.assertEqual(m["totals"]["bootstrap_results"]["disagreeing_contexts"], [w.ids["F0"]])
        self.assertEqual(m["totals"]["bootstrap_results"]["tolerated"], 1)
        self.assertEqual(r["prefixes"]["bootstrap_results"]["pin"], "cross_check")
        self.assertEqual(P.verify(self.tmp / "out-one", rehash_sources=True, sources=w.sources)["verdict"], "PASS")
        # an edit of the snapshot after the build is caught by its recorded pin
        w.replace_once(w.paths["F0"]["boot"], b'"status":"cached"', b'"status":"cachex"')
        rep = P.verify(self.tmp / "out-one", rehash_sources=True, sources=w.sources)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertTrue(any("bootstrap_results" in f for f in rep["failures"]), rep["failures"])
        # a second disagreement in the build is a STOP
        w2 = self.world("two")
        w2.replace_once(w2.paths["F0"]["outcome"], b'"record_status":"missing"', b'"record_status":"timeout"')
        w2.replace_once(w2.paths["F1"]["outcome"], b'"record_status":"missing"', b'"record_status":"timeout"')
        with self.assertRaises(P.ReconcileError) as cm:
            self.build(w2, "out-two")
        self.assertEqual(cm.exception.kind, "bootstrap_results")
        # the program sets must agree
        w3 = self.world("set")
        body = w3.paths["F1"]["boot"].read_bytes().splitlines(keepends=True)
        w3.paths["F1"]["boot"].write_bytes(b"".join(body[:-1]))
        with self.assertRaises(P.ReconcileError) as cm:
            self.build(w3, "out-set")
        self.assertEqual((cm.exception.context, cm.exception.kind), (w3.ids["F1"], "bootstrap_results"))
        # a clean build checks every context (both cohorts) and finds no disagreement
        w4 = self.world("clean")
        m4 = self.build(w4, "out-clean")
        self.assertEqual(m4["totals"]["bootstrap_results"]["checked_contexts"], len(SPECS))
        self.assertEqual(m4["totals"]["bootstrap_results"]["disagreements"], 0)

    def test_la6_frozen_fold0_never_in_diag(self):
        w = self.world("w")
        pmap = json.loads(w.pmap.read_bytes())
        bad = {**pmap, w.ids["F0"]: {"family_id": "fam-a", "split_role": "diag"}}
        with self.assertRaises(P.ReconcileError) as cm:
            P.load_contexts(w.epoch, bad)
        self.assertEqual((cm.exception.context, cm.exception.kind), (w.ids["F0"], "closure"))
        ok = {**pmap, w.ids["F1"]: {"family_id": "fam-b", "split_role": "diag"}}      # frozen fold 1: allowed
        self.assertIn("diag", [c.split_role for c in P.load_contexts(w.epoch, ok) if c.context_sha256 == w.ids["F1"]])


def _real_selection():
    doc = json.loads(EPOCH.read_bytes())
    inc = sorted((c for c in doc["contexts"] if c["status"] == "included"), key=lambda c: c["context_sha256"])
    picked = [next(c for c in inc if c["cohort"] == cohort and c["fold"] == fold)
              for cohort, folds in (("frozen", (0, 1, 2)), ("incoming", (2, 3, 4))) for fold in folds]
    head = {k: doc[k] for k in ("schema", "experiment_id", "epoch")}
    return head, picked


class RealContextsTests(_Asserts, unittest.TestCase):
    """Six pinned real contexts, 3 frozen (folds 0-2) and 3 incoming (folds 2-4), under a fake partition map."""

    ROLES = ("train_fit", "dev", "heldout", "train_fit", "diag", "train_fit")

    @classmethod
    def setUpClass(cls):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="p4a-real-", dir=TMP_ROOT))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        head, cls.picked = _real_selection()
        cls.epoch = cls.tmp / "epoch-sub.json"
        cls.epoch.write_bytes(json.dumps({**head, "contexts": cls.picked}).encode())
        cls.pmap = cls.tmp / "partition-map.json"
        cls.pmap.write_bytes(json.dumps({c["context_sha256"]: {"family_id": "fake-" + c["dataset_hash"], "split_role": r}
                                         for c, r in zip(cls.picked, cls.ROLES)}).encode())
        cls.contexts = P.load_contexts(cls.epoch, cls.pmap)
        pools = [P._seed_pool(c)[0] for c in cls.contexts]
        cls.seed_set = frozenset(set.intersection(*pools))
        cls.seeds = cls.tmp / "seeds.json"
        cls.seeds.write_bytes(json.dumps(sorted(cls.seed_set)).encode())
        cls.out = cls.tmp / "program-dict"
        cls.manifest = P.build(cls.epoch, cls.pmap, cls.seeds, cls.out, chunk_contexts=2)
        cls.report = P.verify(cls.out, rehash_sources=True)

    def test_real_build_reconciles_and_verifies(self):
        m, rep = self.manifest, self.report
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        self.assertEqual(rep["checked"]["rows"], m["totals"]["rows"])
        self.assertEqual(rep["checked"]["summaries_reconciled"], 6)
        self.assertGreaterEqual(rep["checked"]["prefixes_rehashed"], 6 * 4)
        self.assertEqual(m["totals"]["rows"], m["totals"]["unique_programs"])
        self.assertEqual({c["fold"] for c in self.picked if c["cohort"] == "frozen"}, {0, 1, 2})
        self.assertEqual({c["fold"] for c in self.picked if c["cohort"] == "incoming"}, {2, 3, 4})
        for c in self.picked:
            r = m["contexts"][c["context_sha256"]]
            with self.subTest(context=c["context_sha256"][:16]):
                self.assertEqual(r["source"], "local")
                self.assertEqual(r["rows"], r["summary"]["unique_programs"])
                self.assertEqual(r["native_rows"], r["summary"]["native_events"])
                self.assertEqual(r["bootstrap_set"], r["seed_pool"])
                self.assertEqual(r["native_rows"] + r["bootstrap_rows"] + r["transport_only_rows"], r["rows"])
                self.assertEqual(r["prefixes"]["native"]["sha256"], c["native_source"]["sha256"])
                self.assertEqual(r["summary"]["history_sha256"], c["history_ref"]["sha256"])
                self.assertEqual(r["prefixes"]["outcome"]["pin"], "bytes+lines" if c["cohort"] == "frozen" else "fingerprint")
                self.assertEqual(r["shared_seed_rows"], len(self.seed_set))
        self.assertTrue(any(r["transport_only_rows"] for r in m["contexts"].values()))
        self.assertEqual({p: m["partitions"][p]["contexts"] for p in P.PARTITIONS}, {"train_fit": 3, "dev": 1, "diag": 1, "heldout": 1})
        self.assertEqual(len(m["partitions"]["train_fit"]["chunks"]), 2)
        self.assertEqual(len(list(P.iter_distinct(self.out, "dev", role="training_validation"))),
                         m["partitions"]["dev"]["distinct_programs"])
        self.assertEqual(m["totals"]["bootstrap_results"]["checked_contexts"], 6)
        self.assertLessEqual(m["totals"]["bootstrap_results"]["disagreements"], 1)
        frozen = [r for r in m["contexts"].values() if r["cohort"] == "frozen"]
        self.assertTrue(all(r["prefixes"]["bootstrap_results"]["pin"] == "cross_check" for r in frozen))
        self.assert_no_test_fields(self.out)
        print(json.dumps({"real_p4a_sample": {
            "totals": {k: m["totals"][k] for k in ("contexts", "rows", "unique_programs")},
            "partitions": {p: {k: m["partitions"][p][k] for k in ("contexts", "rows", "distinct_programs")} for p in P.PARTITIONS},
            "contexts": {cid[:16]: {k: r[k] for k in ("cohort", "fold", "split_role", "rows", "native_rows", "bootstrap_rows",
                                                     "transport_only_rows", "shared_seed_rows")} for cid, r in m["contexts"].items()},
            "verify": rep["checked"]}}, sort_keys=True), file=sys.stderr)

    def test_real_tampered_prefixes_stop(self):
        cursor = P.load_incoming_cursor(P.DEFAULT_SOURCES.incoming_cursor)[0]
        frozen = next(c for c in self.contexts if c.cohort == "frozen" and self.manifest["contexts"][c.context_sha256]["transport_only_rows"])
        incoming = next(c for c in self.contexts if c.cohort == "incoming")
        for ctx in (frozen, incoming):
            s = P.history_summary(ctx, cursor)
            rows, rec = P.context_rows(ctx, s, seeds=self.seed_set)            # untampered: passes
            self.assertEqual(len(rows), s.unique_programs)
            tampered = {   # label -> (context, summary, expected STOP kind)
                "native sha256 pin": (dataclasses.replace(ctx, native={**ctx.native, "sha256": _flip(ctx.native["sha256"])}), s, "source"),
                "outcome prefix one byte short": (ctx, dataclasses.replace(s, outcome_bytes=s.outcome_bytes - 1), "source"),
                "outcome prefix one byte long": (ctx, dataclasses.replace(s, outcome_bytes=s.outcome_bytes + 1), "source"),
                "transport prefix one byte short": (ctx, dataclasses.replace(s, transport_bytes=s.transport_bytes - 1), "source"),
                "unique_programs off by one": (ctx, dataclasses.replace(s, unique_programs=s.unique_programs + 1), "count"),
                "native_events off by one": (ctx, dataclasses.replace(s, native_events=s.native_events + 1), "count"),
            }
            if ctx.cohort == "incoming":
                tampered["archiver fingerprint"] = (ctx, dataclasses.replace(s, fingerprint=_flip(s.fingerprint)), "source")
                tampered["bootstrap-results sha256"] = (
                    ctx, dataclasses.replace(s, bootstrap_source_sha256=_flip(s.bootstrap_source_sha256)), "source")
            for label, (c2, s2, kind) in tampered.items():
                with self.subTest(cohort=ctx.cohort, tamper=label):
                    with self.assertRaises(P.ReconcileError) as cm:
                        P.context_rows(c2, s2, seeds=self.seed_set)
                    self.assertEqual((cm.exception.context, cm.exception.kind), (ctx.context_sha256, kind))


class RealConditionedSeedTests(_Asserts, unittest.TestCase):
    """LA-1 on real data: 10 frozen contexts (2 per fold, Random(20261005)) built under a fake partition map. Every
    frozen context carries 16 conditioned prior champions; the published chunks must not single them out."""

    @classmethod
    def setUpClass(cls):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="p4a-real-la1-", dir=TMP_ROOT))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        doc = json.loads(EPOCH.read_bytes())
        frozen = sorted((c for c in doc["contexts"] if c["status"] == "included" and c["cohort"] == "frozen"),
                        key=lambda c: c["context_sha256"])
        rng, by_fold = random.Random(20261005), {}
        for c in frozen:
            by_fold.setdefault(c["fold"], []).append(c)
        cls.picked = [x for f in sorted(by_fold) for x in rng.sample(by_fold[f], 2)]
        epoch = cls.tmp / "epoch-sub.json"
        epoch.write_bytes(json.dumps({**{k: doc[k] for k in ("schema", "experiment_id", "epoch")}, "contexts": cls.picked}).encode())
        roles = ("train_fit", "dev", "train_fit", "heldout", "train_fit", "diag", "train_fit", "dev", "train_fit", "diag")
        pmap = cls.tmp / "partition-map.json"
        pmap.write_bytes(json.dumps({c["context_sha256"]: {"family_id": "fake-" + c["dataset_hash"], "split_role": r}
                                     for c, r in zip(cls.picked, roles)}).encode())
        contexts = P.load_contexts(epoch, pmap)
        cls.shared = frozenset(set.intersection(*(P._seed_pool(c)[0] for c in contexts)))   # the broad seeds
        seeds = cls.tmp / "seeds.json"
        seeds.write_bytes(json.dumps(sorted(cls.shared)).encode())
        cls.out = cls.tmp / "program-dict"
        cls.manifest = P.build(epoch, pmap, seeds, cls.out, chunk_contexts=4)

    def test_real_conditioned_seeds_are_sealed(self):
        m, out = self.manifest, self.out
        self.assertEqual(P.verify(out, rehash_sources=True)["verdict"], "PASS")
        self.assertEqual(len(self.picked), 10)
        for c in self.picked:
            r = m["contexts"][c["context_sha256"]]
            self.assertEqual((r["conditioned_seeds"], r["conditioned_seed_rows"]), (16, 16), c["context_sha256"][:16])
            self.assertEqual(r["bootstrap_results"]["programs"], r["bootstrap_set"])
        self.assertEqual(sum(m["sealed"]["partitions"][p]["conditioned_rows"] for p in P.PARTITIONS), 160)
        stats = {"cells": 0, "pure_cells": 0}
        old = Counter()
        for p in P.PARTITIONS:
            pub = P.open_partition(out, p, role=P.PARTITION_ROLE[p]).to_table(use_threads=False)
            sealed = P.open_sealed(out, p, role="exposed_report").to_table(use_threads=False)
            self.assertNotIn("origin", pub.schema.names)
            self.assertTrue(set(pub.column("record_status").to_pylist()) <= set(P.PUBLISHED_STATUSES))
            cols = [pub.column(c).to_pylist() for c in ("context_sha256", "record_status", "shared_seed", "form", "cohort",
                                                       "family_id", "split_role")]
            cells = {}
            for row, flag, origin, exact in zip(zip(*cols), sealed.column("conditioned_seed").to_pylist(),
                                                 sealed.column("origin").to_pylist(), sealed.column("record_status").to_pylist()):
                cells.setdefault(row, set()).add(flag)
                old[("origin=bootstrap&!shared", origin == "bootstrap" and not row[2], flag)] += 1
                old[("status=missing&!shared", exact == "missing" and not row[2], flag)] += 1
            stats["cells"] += len(cells)
            stats["pure_cells"] += sum(1 for f in cells.values() if f == {True})
        self.assertEqual(stats["pure_cells"], 0)          # no published value combination isolates champions
        # what the pre-fix columns exposed: every champion through origin, and some through 'missing' alone
        self.assertEqual(old[("origin=bootstrap&!shared", True, True)], 160)
        self.assertGreater(old[("status=missing&!shared", True, True)], 0)
        self.assert_no_test_fields(out)
        print(json.dumps({"real_la1": {**stats, "conditioned_rows": 160,
                                       "pre_fix_predicates": {f"{a}:{b}/{c}": n for (a, b, c), n in sorted(old.items())}}},
                         sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
