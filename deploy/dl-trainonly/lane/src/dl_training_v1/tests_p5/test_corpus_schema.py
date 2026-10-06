"""Offline proofs for dltrain.corpus_schema (PLAN §5 P5; P5 IMPLEMENTATION SPEC §2).

Every input is synthetic (tests_p5/fixtures_p5.py: real-shaped rows with planted sentinels); nothing reads the real
corpus or S3, and every write goes under TMPDIR. Run only inside the lane slice:
  python -m unittest tests_p5.test_corpus_schema -v
"""
from __future__ import annotations

import ast
import base64
import dataclasses
import json
import math
import random
import re
import unittest
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dltrain import corpus_schema as cs
from dltrain import dsl_tokenizer, labels, programs
from tests_p5 import fixtures_p5 as fx

S, F64, F32, B = pa.string(), pa.float64(), pa.float32(), pa.bool_()
TRAIN_COLUMNS = [
    ("context_sha256", S, False), ("context_schema", S, False), ("dataset_hash", S, False), ("family_id", S, False),
    ("split_role", S, False), ("program_sha256", S, False), ("skeleton_sha256", S, False),
    ("tokens", pa.list_(pa.uint16()), False), ("length", pa.uint16(), False), ("outcome", S, False),
    ("rmse_j", pa.list_(F64), True), ("baseline_rmse_j", pa.list_(F64), True), ("cv_mean_log_ratio", F64, True),
    ("cv_log_ratio_sd", F64, True), ("cv_mean_ratio", F64, True), ("gate_pass", B, True), ("wall_us", pa.int64(), True),
    ("backend", S, False), ("shared_seed", B, False),
]
CONTEXT_COLUMNS = [
    ("context_sha256", S, False), ("context_schema", S, False), ("dataset_hash", S, False), ("family_id", S, False),
    ("split_role", S, False), ("cohort", S, False), ("fold", pa.int8(), False), ("frozen_fold0", B, False),
    ("n_train", pa.int32(), False), ("n_features", pa.int32(), False), ("inner_folds", pa.int8(), False),
    ("y_train", pa.list_(F64), False), ("train_rows_sha256", S, False), ("s_c", F64, False), ("eps_c", F64, False),
    ("label_status", S, False), ("baseline_rmse_j", pa.list_(F64), True), ("q_cap", F64, True),
    ("n_records", pa.int32(), False), ("n_refused", pa.int32(), False), ("eligibility_source", S, False),
]
LINEAGE_COLUMNS = [
    ("context_sha256", S, False), ("program_sha256", S, False), ("split_role", S, False), ("origin", S, False),
    ("operator", S, True), ("parents", pa.list_(S), True), ("birth_seq", pa.int64(), True),
    ("generation", pa.int32(), True), ("island_policy", S, True), ("island_id", pa.int32(), True),
    ("origin_island", pa.int32(), True), ("migration_epoch", pa.int32(), True), ("proposal_seq", pa.int64(), True),
    ("proposal_draw", pa.int64(), True), ("born_completed", pa.int32(), True), ("roots", pa.list_(S), True),
    ("native_line", pa.int32(), True), ("outcome_line", pa.int32(), True), ("bootstrap_events", pa.int16(), False),
    ("evolution_outcome_rows", pa.int16(), False), ("transport_attempts", pa.int32(), False),
    ("transport_status", S, True), ("attempt_id", S, True), ("node_raw", S, True), ("pools", pa.list_(S), True),
    ("backend_source", S, False), ("backend_field", S, True), ("backend_field_agrees", B, True),
]
REFUSED_COLUMNS = [("context_sha256", S, False), ("program_sha256", S, False), ("split_role", S, False),
                   ("length", pa.int32(), False), ("reason", S, False)]
# PLAN §5 P5 "Sealed exposed table" fields and the spec's history pointer.
PLAN_EXPOSED = {"fitness", "test_rmse", "test_rmse_ratio", "gp_rmse_over_tabpfn35", "gp_over_tabpfn35", "tabpfn35_rmse",
                "semantic_signature_f32", "sampled_absolute_error_f32", "full_prediction_sha256", "population_selected",
                "new_signal_admitted_high_water", "semantic_duplicate_of_active", "result_key",
                "result_ref_bucket", "result_ref_key", "result_ref_version_id", "result_ref_sha256"}
HISTORY_POINTER = {"history_ref_bucket", "history_ref_key", "history_ref_version_id", "history_ref_sha256",
                   "native_packed_line", "outcome_packed_line", "native_line_sha256", "outcome_line_sha256",
                   "exposed_sources"}
FORBIDDEN_TYPES = (pa.types.is_binary, pa.types.is_large_binary, pa.types.is_fixed_size_binary,
                   pa.types.is_dictionary, pa.types.is_struct, pa.types.is_map)
EXPOSED_TOKENS = ("official_test", "test_rmse", "fitness", "tabpfn35", "signature", "sampled_absolute_error",
                  "full_prediction", "population_selected", "semantic_duplicate", "result_key", "result_ref",
                  "signal_admitted", "best_signal_test")
NAN, INF = float("nan"), float("inf")
_CTX_CACHE: dict = {}


def plain(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: plain(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def canon(obj) -> str:
    """Canonical text that keeps NaN, +-inf and the sign of -0.0 distinct."""
    return json.dumps(plain(obj), sort_keys=True, allow_nan=True)


def synth(name: str = "T0", **kw) -> fx.SynthContext:
    key = (name, tuple(sorted(kw.items())))
    if key not in _CTX_CACHE:
        _CTX_CACHE[key] = fx.synth_context(fx.CtxSpec(name, **kw))
    return _CTX_CACHE[key]


def build(sc: fx.SynthContext, lines=None, *, vocab=None, max_seq=512, seeds=()):
    """Parse a context's lines and assemble its records (the train path end to end)."""
    ident = sc.ident
    nl, ol, tl = lines if lines is not None else (sc.native_lines, sc.outcome_lines, sc.transport_lines)
    native = [cs.parse_native(line, i + 1, ident)[0] for i, line in enumerate(nl)]
    outcome = [cs.parse_outcome(line, i + 1, ident)[0] for i, line in enumerate(ol)]
    transport = [cs.parse_transport(line, i + 1, ident)[0] for i, line in enumerate(tl)]
    accs, index, _ = cs.accumulate_programs(native, outcome, transport)
    out = cs.build_context_records(sc.ctx_meta(), accs, vocab or fx.context_vocab(sc), max_seq, seeds,
                                   attempt_index=index)
    return (native, outcome, transport), out


def all_types(t: pa.DataType):
    yield t
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        yield from all_types(t.value_type)
    elif pa.types.is_struct(t):
        for f in t:
            yield from all_types(f.type)


def nat(status, reason=None, sha="a" * 64, dsl="(pipe X (norm_row) (ridge_gcv))", line=1, cvm=None, cvsd=None, gate=None):
    return cs.NativeRow(line, sha, dsl, status, reason, cvm, cvsd, gate)


def out(phase, rs, ts=None, sha="a" * 64, dsl="(pipe X (norm_row) (ridge_gcv))", line=2, node=None, attempt=None,
        cvm=None, cvsd=None, ratio=None, gate=None, rmse=None, base=None, wall=None):
    return cs.OutcomeRow(line, phase, sha, dsl, rs, ts, attempt, node, wall, cvm, cvsd, ratio, gate,
                         None if rmse is None else tuple(rmse), None if base is None else tuple(base))


def trans(attempt, pool, sha="a" * 64, dsl=None, line=3):
    return cs.TransportRow(line, attempt, pool, sha, dsl)


class SchemaExact(unittest.TestCase):
    def _assert_exact(self, schema, columns):
        self.assertEqual([(f.name, f.nullable) for f in schema], [(n, nl) for n, _, nl in columns])
        for f, (name, typ, _) in zip(schema, columns):
            self.assertTrue(f.type.equals(typ), f"{name}: {f.type} != {typ}")
            self.assertEqual(str(f.type), str(typ), name)          # equals() ignores list child names; str() does not
            if pa.types.is_list(f.type):
                self.assertEqual(f.type.value_field.name, "item", name)
            for t in all_types(f.type):
                self.assertFalse(any(check(t) for check in FORBIDDEN_TYPES), f"{name}: forbidden type {t}")
        self.assertEqual(schema.metadata, None)

    def _assert_json_schema(self, tid, schema):
        js = cs.json_schema(tid)
        jsonschema.Draft202012Validator.check_schema(js)
        self.assertEqual(js["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIs(js["additionalProperties"], False)
        self.assertEqual(js["type"], "object")
        self.assertEqual(js["required"], schema.names)
        self.assertEqual(list(js["properties"]), schema.names)
        self.assertEqual(js["x-arrow-schema-sha256"], cs.schema_sha256(schema))
        for f in schema:
            prop = js["properties"][f.name]
            if f.name.endswith("sha256"):
                self.assertEqual(prop["pattern"], "^[0-9a-f]{64}$", f.name)
            if pa.types.is_list(f.type):
                self.assertIn("items", prop, f.name)
            nullable_type = isinstance(prop["type"], list) and "null" in prop["type"]
            self.assertEqual(nullable_type, f.nullable, f.name)
        path = cs.SCHEMAS_DIR / f"{tid.split('/')[0]}.schema.json"
        self.assertEqual(path, cs.schema_file(tid))
        self.assertEqual(path.read_bytes(), cs.json_schema_bytes(tid), f"{path} differs from the generated bytes")
        self.assertEqual(cs.json_schema_bytes(tid), cs.json_schema_bytes(tid))
        return js

    def test_train_record_columns(self):
        s = cs.TRAIN_RECORD_SCHEMA
        self._assert_exact(s, TRAIN_COLUMNS)
        self.assertEqual(len(s), 19)
        for name in s.names:
            self.assertFalse(name.endswith(("_json", "_b64")), name)
            self.assertNotIn(name, cs.EXPOSED_NAMES)
            self.assertFalse(any(sub in name for sub in cs.FORBIDDEN_SUBSTRINGS), name)
            self.assertFalse(cs.name_is_exposed(name), name)
        from tests import fixtures_p7
        self.assertEqual(set(fixtures_p7.records()[0]) | {"gate_pass"}, set(s.names))
        self.assertEqual(cs.schema_sha256(s), cs.common.sha256_bytes(s.to_string(show_schema_metadata=False).encode()))
        js = self._assert_json_schema(cs.TRAIN_RECORD_ID, s)
        props = js["properties"]
        self.assertEqual(props["tokens"]["items"], {"maximum": 65535, "minimum": 0, "type": "integer"})
        self.assertEqual((props["length"]["minimum"], props["length"]["maximum"]), (2, 512))
        self.assertEqual(props["outcome"]["pattern"], cs.OUTCOME_PATTERN)
        self.assertEqual(props["rmse_j"]["minItems"], 2)
        self.assertEqual(props["rmse_j"]["maxItems"], 2)
        self.assertEqual(props["backend"]["enum"], list(cs.BACKENDS))
        for good in (*cs.OUTCOME_CLASSES, "failed:compile_error", "failed:" + "x" * 40):
            self.assertRegex(good, cs.OUTCOME_PATTERN)
        for bad in ("failed:", "failed:Compile", "failed:" + "x" * 41, "timeout@60000ms", "abandoned:other", "done"):
            self.assertIsNone(re.match(cs.OUTCOME_PATTERN, bad), bad)
        # every assembled record validates; extra keys, bad outcomes and out-of-range ids do not
        _, built = build(synth())
        validator = jsonschema.Draft202012Validator(js)
        for rec in built.records:
            self.assertEqual(list(rec), s.names)
            validator.validate(rec)
        rec = dict(built.records[0])
        for bad in ({**rec, "provenance_json": "{}"}, {**rec, "outcome": "failed:"},
                    {**rec, "tokens": [1, 70000, 2]}, {k: v for k, v in rec.items() if k != "gate_pass"},
                    {**rec, "backend": "gpu"}, {**rec, "context_sha256": "XYZ"}):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(bad)

    def test_other_tables(self):
        self._assert_exact(cs.TRAIN_CONTEXT_SCHEMA, CONTEXT_COLUMNS)
        self._assert_exact(cs.LINEAGE_RECORD_SCHEMA, LINEAGE_COLUMNS)
        self._assert_exact(cs.REFUSED_RECORD_SCHEMA, REFUSED_COLUMNS)
        self.assertEqual(len(cs.TRAIN_CONTEXT_SCHEMA), 21)
        for schema in (cs.TRAIN_CONTEXT_SCHEMA, cs.LINEAGE_RECORD_SCHEMA, cs.REFUSED_RECORD_SCHEMA):
            for name in schema.names:
                self.assertFalse(cs.name_is_exposed(name), name)
                self.assertFalse(name.endswith(("_json", "_b64")), name)
        ex = cs.EXPOSED_RECORD_SCHEMA
        self.assertEqual(ex.names[:4], ["context_sha256", "program_sha256", "split_role", "origin"])
        self.assertEqual([f.nullable for f in ex][:4], [False] * 4)
        self.assertTrue(all(f.nullable for f in list(ex)[4:]))
        self.assertEqual(len(set(ex.names)), len(ex.names))
        self.assertLessEqual(PLAN_EXPOSED | HISTORY_POINTER, set(ex.names))
        for t in ("bucket", "key", "version_id", "sha256"):
            self.assertIn(f"result_ref_{t}", ex.names)
        for grp in ("metrics", "baseline_metrics", "training_metrics"):
            for m in ("rmse", "mae", "r2", "sse", "max_abs_error"):
                self.assertEqual(ex.field(f"official_test_{grp}_{m}").type, F64)
            self.assertEqual(ex.field(f"official_test_{grp}_n").type, pa.int64())
        self.assertEqual(ex.field("semantic_signature_f32").type, pa.list_(F32))
        self.assertEqual(ex.field("sampled_absolute_error_f32").type, pa.list_(F32))
        # every official_test key of a real-shaped evaluation detail maps to a column
        counts = Counter()
        block = fx.evaluation_detail(fx.Sentinels("map"), [0.1, 0.2], [0.3, 0.4])["official_test"]
        flat = cs.project_official_test_exposed(block, counts)
        self.assertEqual(counts, Counter())
        self.assertTrue(all(v is not None for v in flat.values()), [k for k, v in flat.items() if v is None])
        self.assertLessEqual(set(flat), set(ex.names))
        for name in ex.names[4:]:
            if name not in HISTORY_POINTER:
                self.assertTrue(cs.name_is_exposed(name), name)
        for tid, schema in cs.TABLES.items():
            if tid != cs.TRAIN_RECORD_ID:
                self._assert_json_schema(tid, schema)
        self.assertEqual(cs.check_json_schemas(), [])
        self.assertEqual(json.loads(cs.json_schema_bytes(cs.EXPOSED_RECORD_ID))["x-sealed"], True)
        self.assertEqual(sorted(cs.TABLES), sorted(cs.TABLE_SHORT.values()))


class SchemaRoundTrip(unittest.TestCase):
    def _roundtrip(self, table, path: Path, *, sealed=False):
        path.write_bytes(cs.parquet_bytes(table, sealed=sealed))
        pf = pq.ParquetFile(path)
        self.assertTrue(pf.schema_arrow.equals(table.schema))
        self.assertEqual(pf.schema_arrow.to_string(show_schema_metadata=False),
                         table.schema.to_string(show_schema_metadata=False))
        self.assertEqual(set(pf.metadata.metadata), {b"ARROW:schema"})
        for g in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(g)
            self.assertTrue(all(rg.column(j).compression == "ZSTD" for j in range(rg.num_columns)))
        return pf.read(use_threads=False).to_pylist()

    def test_parquet_roundtrip_edge_values(self):
        bos, eos = dsl_tokenizer.BOS, dsl_tokenizer.EOS
        tmpl = {"context_sha256": "c" * 64, "context_schema": "jope.gp-evaluation-context/v8", "dataset_hash": "d" * 16,
                "family_id": "fam-a", "split_role": "train_fit", "program_sha256": "0" * 64, "skeleton_sha256": "e" * 64,
                "tokens": [bos, 300, eos], "length": 3, "outcome": "completed", "rmse_j": None, "baseline_rmse_j": None,
                "cv_mean_log_ratio": None, "cv_log_ratio_sd": None, "cv_mean_ratio": None, "gate_pass": None,
                "wall_us": None, "backend": "local", "shared_seed": False}
        rows = [
            {**tmpl, "program_sha256": "1" * 64, "tokens": [bos, 0, 65535, eos], "length": 4,
             "outcome": "failed:nonfinite_metric", "rmse_j": [NAN, INF], "baseline_rmse_j": [1.0, -0.0],
             "cv_mean_log_ratio": -INF, "cv_log_ratio_sd": NAN, "cv_mean_ratio": -0.0, "wall_us": 0},
            {**tmpl, "program_sha256": "2" * 64, "gate_pass": True, "wall_us": 2**63 - 1, "backend": "aws_lambda",
             "cv_mean_log_ratio": 1e308, "cv_log_ratio_sd": 5e-324},
            {**tmpl, "program_sha256": "3" * 64, "outcome": "missing", "rmse_j": [0.25, None],
             "baseline_rmse_j": [None, None], "gate_pass": False, "shared_seed": True, "backend": "unknown"},
            {**tmpl, "program_sha256": "4" * 64, "rmse_j": [0.0, 0.0], "baseline_rmse_j": [0.2, 0.3],
             "outcome": "abandoned:capacity_deferred_before_claim", "tokens": [bos] + [65535] * 509 + [eos],
             "length": 511},
        ]
        ctx_rows = [{"context_sha256": "c" * 64, "context_schema": "jope.gp-evaluation-context/v5", "dataset_hash": "d" * 16,
                     "family_id": "f",
                     "split_role": "dev", "cohort": "frozen", "fold": 0, "frozen_fold0": True, "n_train": 3,
                     "n_features": 2, "inner_folds": 2, "y_train": [-0.0, 1e-300, -1e300],
                     "train_rows_sha256": "a" * 64, "s_c": 1.5, "eps_c": 1.5e-6, "label_status": "DEGENERATE_BASELINE",
                     "baseline_rmse_j": [0.0, 1e-9], "q_cap": None, "n_records": 0, "n_refused": 2**31 - 1,
                     "eligibility_source": "x" + cs.LEAK_PASS_SUFFIX},
                    {"context_sha256": "b" * 64, "context_schema": "jope.gp-evaluation-context/v8",
                     "dataset_hash": "d" * 16, "family_id": "f",
                     "split_role": "heldout", "cohort": "incoming", "fold": 127, "frozen_fold0": False,
                     "n_train": 1, "n_features": 1, "inner_folds": 2, "y_train": [2.5], "train_rows_sha256": "a" * 64,
                     "s_c": 0.0, "eps_c": 2.5e-12, "label_status": "UNKNOWN_BASELINE", "baseline_rmse_j": None,
                     "q_cap": -0.0, "n_records": 5, "n_refused": 0, "eligibility_source": "y"}]
        lineage_rows = [cs.lineage_row(context_sha256="c" * 64, program_sha256="1" * 64, split_role="dev",
                                       origin="transport_only", pools=["xbabe1:7431", "aws:us-east-1"],
                                       transport_attempts=2),
                        cs.lineage_row(context_sha256="c" * 64, program_sha256="2" * 64, split_role="dev",
                                       origin="evolution",
                                       native={"operator": "op", "parents": [], "birth_seq": 2**63 - 1,
                                               "island_policy": "test_gated", "island_id": -2**31, "roots": ["f" * 64]},
                                       outcome={"generation": 2**31 - 1, "backend_field": "local_lambda"},
                                       backend=cs.BackendInfo("aws_lambda", "transport_attempt", "aws:us-east-1"),
                                       native_line=1, outcome_line=2, bootstrap_events=2**15 - 1,
                                       evolution_outcome_rows=1)]
        refused_rows = [{"context_sha256": "c" * 64, "program_sha256": "9" * 64, "split_role": "heldout",
                         "length": 600, "reason": "over_max_seq"}]
        exposed_rows = [cs.project_record_exposed(context_sha256="c" * 64, program_sha256="1" * 64,
                                                  split_role="train_fit", origin="evolution"),
                        {**dict.fromkeys(cs.EXPOSED_RECORD_SCHEMA.names), "context_sha256": "c" * 64,
                         "program_sha256": "2" * 64, "split_role": "heldout", "origin": "bootstrap",
                         "semantic_signature_f32": [0.5, -0.0, float(np.float32(3.4e38))],
                         "sampled_absolute_error_f32": [], "test_rmse": NAN, "native_packed_line": 2**40}]
        self.assertEqual(lineage_rows[1]["backend_field_agrees"], False)
        with fx.tmpdir() as d:
            d = Path(d)
            back = self._roundtrip(cs.records_table(rows), d / "records.parquet")
            self.assertEqual(canon(back), canon(rows))
            self.assertEqual(math.copysign(1.0, back[0]["baseline_rmse_j"][1]), -1.0)
            self.assertEqual(math.copysign(1.0, back[0]["cv_mean_ratio"]), -1.0)
            self.assertTrue(math.isnan(back[0]["rmse_j"][0]))
            self.assertEqual(canon(cs.read_records(d / "records.parquet")), canon(rows))
            self.assertEqual(canon(cs.read_records([d / "records.parquet", d / "records.parquet"])), canon(rows + rows))
            self.assertEqual(cs.check_parquet(d / "records.parquet", cs.TRAIN_RECORD_SCHEMA, decode=True), [])
            self.assertEqual(cs.check_parquet(d / "records.parquet", cs.TRAIN_RECORD_SCHEMA, True, max_seq=511), [])
            self.assertEqual(cs.check_parquet(d / "records.parquet", cs.TRAIN_RECORD_SCHEMA, True, max_seq=510),
                             ["length 511 > max_seq 510"])
            back = self._roundtrip(cs.contexts_table(ctx_rows), d / "contexts.parquet")
            self.assertEqual(canon(back), canon(ctx_rows))
            self.assertEqual(canon(cs.read_contexts(d / "contexts.parquet")), canon(ctx_rows))
            self.assertEqual(cs.check_parquet(d / "contexts.parquet", cs.TRAIN_CONTEXT_SCHEMA, decode=True), [])
            for name, table, schema, rows_in in (("lineage", cs.lineage_table(lineage_rows), cs.LINEAGE_RECORD_SCHEMA,
                                                  lineage_rows),
                                                 ("refused", cs.refused_table(refused_rows), cs.REFUSED_RECORD_SCHEMA,
                                                  refused_rows)):
                back = self._roundtrip(table, d / f"{name}.parquet")
                self.assertEqual(canon(back), canon(rows_in))
                self.assertEqual(cs.check_parquet(d / f"{name}.parquet", schema, decode=True), [])
            sealed_dir = d / "exposed" / "train_fit"
            sealed_dir.mkdir(parents=True)
            back = self._roundtrip(cs.exposed_table(exposed_rows), sealed_dir / "part-00000.parquet", sealed=True)
            self.assertEqual(canon(back), canon(exposed_rows))
            self.assertEqual(cs.check_parquet(sealed_dir / "part-00000.parquet", cs.EXPOSED_RECORD_SCHEMA, True), [])
            md = pq.ParquetFile(sealed_dir / "part-00000.parquet").metadata
            self.assertFalse(any(md.row_group(0).column(j).is_stats_set for j in range(md.num_columns)))
            # negative controls of the validator
            (sealed_dir / "with-stats.parquet").write_bytes(cs.parquet_bytes(cs.exposed_table(exposed_rows)))
            fails = cs.check_parquet(sealed_dir / "with-stats.parquet", cs.EXPOSED_RECORD_SCHEMA)
            self.assertTrue(any("statistics" in f for f in fails), fails)
            self.assertTrue(cs.check_parquet(d / "records.parquet", cs.TRAIN_RECORD_SCHEMA, max_row_group=2))
            self.assertTrue(cs.check_parquet(d / "records.parquet", cs.TRAIN_CONTEXT_SCHEMA))
            t = cs.records_table(rows)
            pq.write_table(t, d / "snappy.parquet", compression="snappy")
            self.assertTrue(any("ZSTD" in f for f in cs.check_parquet(d / "snappy.parquet", cs.TRAIN_RECORD_SCHEMA)))
            pq.write_table(t.replace_schema_metadata({"k": "v"}), d / "meta.parquet", compression="zstd")
            self.assertTrue(any("metadata" in f for f in cs.check_parquet(d / "meta.parquet", cs.TRAIN_RECORD_SCHEMA)))
            # pyarrow's default compliant list naming stores "element": equals() passes, the exact check does not
            pq.write_table(t, d / "compliant.parquet", compression="zstd")
            self.assertTrue(pq.ParquetFile(d / "compliant.parquet").schema_arrow.equals(cs.TRAIN_RECORD_SCHEMA))
            fails = cs.check_parquet(d / "compliant.parquet", cs.TRAIN_RECORD_SCHEMA)
            self.assertEqual(len(fails), 1, fails)
            self.assertIn("child names", fails[0])
            self.assertEqual(canon(cs.read_records(d / "compliant.parquet")), canon(rows))
            loose = pa.schema([f.with_nullable(True) for f in cs.TRAIN_RECORD_SCHEMA])
            bad_rows = [dict(rows[1], tokens=[bos, None, eos], rmse_j=[0.1, 0.2, 0.3], backend="gpu")]
            pq.write_table(pa.Table.from_pylist(bad_rows, schema=loose), d / "bad.parquet", compression="zstd")
            with self.assertRaises(cs.CorpusStop):
                cs.read_records(d / "bad.parquet")
            self.assertTrue(cs.check_parquet(d / "bad.parquet", cs.TRAIN_RECORD_SCHEMA, decode=True))

    def test_read_back_feeds_labels_and_sampler(self):
        """read_records/read_contexts output goes straight into labels.label_corpus and sampler.Corpus.build/fit."""
        from dltrain import sampler
        rows, ctxs = [], []
        for sc in (synth(), synth("T1", cohort="incoming", fold=3, item="T")):
            _, built = build(sc)
            ti = cs.train_input(sc.jtt1_raw, sc.prepared, sc.ident)
            rows.extend(built.records)
            ctxs.append(cs.build_context_row(sc.ctx_meta(), ti, built.context_label, len(built.records), 0))
        with fx.tmpdir() as d:
            rp, cp = Path(d) / "records.parquet", Path(d) / "contexts.parquet"
            rp.write_bytes(cs.parquet_bytes(cs.records_table(rows)))
            cp.write_bytes(cs.parquet_bytes(cs.contexts_table(ctxs)))
            back_rows, back_ctx = cs.read_records(rp), cs.read_contexts(cp)
        self.assertEqual(canon(back_rows), canon(rows))
        self.assertEqual(canon(back_ctx), canon(ctxs))
        table = labels.label_corpus(back_ctx, back_rows)
        for c in back_ctx:
            got = table.contexts[c["context_sha256"]]
            self.assertEqual((got.status, got.s_c, got.eps_c, got.q_cap), (c["label_status"], c["s_c"], c["eps_c"],
                                                                            c["q_cap"]))
            self.assertEqual(list(got.baseline_rmse_j or []) or None, c["baseline_rmse_j"])
        gates = {(lab.context_sha256, lab.program_sha256): lab.gate_pass for labs in table.records.values() for lab in labs}
        self.assertEqual(gates, {(r["context_sha256"], r["program_sha256"]): r["gate_pass"] for r in back_rows})
        corpus = sampler.Corpus.build(back_ctx, back_rows, {c["context_sha256"]: np.zeros(4, np.float32) for c in back_ctx},
                                      {c["family_id"]: "train_fit" for c in back_ctx})
        fit = sampler.fit(corpus, sampler.SamplerConfig(b_ctx=2))
        self.assertEqual(fit.n_contexts(), 2)

    def test_table_builders_are_strict(self):
        _, built = build(synth())
        rec = dict(built.records[0])
        cases = {
            "extra key": {**rec, "provenance_json": "{}"},
            "missing key": {k: v for k, v in rec.items() if k != "wall_us"},
            "float in int": {**rec, "wall_us": 1.5},
            "bool in double": {**rec, "cv_mean_ratio": True},
            "str for a list": {**rec, "tokens": "abc"},
            "null token": {**rec, "tokens": [dsl_tokenizer.BOS, None, dsl_tokenizer.EOS], "length": 3},
            "float token": {**rec, "tokens": [dsl_tokenizer.BOS, 7.0, dsl_tokenizer.EOS], "length": 3},
            "rmse of 3": {**rec, "rmse_j": [0.1, 0.2, 0.3]},
            "null backend": {**rec, "backend": None},
            "uint16 overflow": {**rec, "tokens": [dsl_tokenizer.BOS, 70000, dsl_tokenizer.EOS], "length": 3},
            "backend enum": {**rec, "backend": "gpu"},
            "split enum": {**rec, "split_role": "sealed"},
            "outcome class": {**rec, "outcome": "failed:"},
            "length mismatch": {**rec, "length": rec["length"] + 1},
            "no EOS": {**rec, "tokens": rec["tokens"][:-1], "length": rec["length"] - 1},
            "bytes for str": {**rec, "family_id": b"fam"},
        }
        for name, bad in cases.items():
            with self.subTest(case=name), self.assertRaises(cs.CorpusStop) as cm:
                cs.records_table([rec, bad])
            self.assertEqual(cm.exception.kind, "table", name)
        self.assertEqual(cs.records_table(built.records).num_rows, len(built.records))


class Prune(unittest.TestCase):
    def test_allowlist_and_type_strictness(self):
        spec = {"a": True, "b": {"c": True}, "e": [{"f": True}], "z": True}
        row = {"a": 1, "b": {"c": "x", "d": 3}, "e": [{"f": 1.5, "g": [1]}, {"g": 2}], "h": {"deep": [1, 2]}}
        self.assertEqual(cs.prune(row, spec), {"a": 1, "b": {"c": "x"}, "e": [{"f": 1.5}, {}]})
        self.assertNotIn("z", cs.prune(row, spec))
        self.assertEqual(cs.prune({"a": None, "b": None, "e": None}, spec), {"a": None, "b": None, "e": None})
        self.assertEqual(cs.prune({"a": NAN}, spec)["a"].__class__, float)
        for bad in ({"a": [1]}, {"a": {"x": 1}}, {"b": 5}, {"b": [1]}, {"b": {"c": [1]}}, {"e": {"f": 1}},
                    {"e": "abc"}, {"e": [{"f": {"x": 1}}]}, {"e": [5]}):
            with self.subTest(row=bad), self.assertRaises(cs.CorpusStop) as cm:
                cs.prune(bad, spec)
            self.assertEqual(cm.exception.kind, "row")
        for bad in ([1], "x", None, 3):
            with self.assertRaises(cs.CorpusStop) as cm:
                cs.prune(bad, spec)
            self.assertEqual(cm.exception.kind, "row")
        sc = synth()
        for lines, spec_, parse in ((sc.native_lines, cs.NATIVE_TRAIN_SPEC, cs.parse_native),
                                    (sc.outcome_lines, cs.OUTCOME_TRAIN_SPEC, cs.parse_outcome),
                                    (sc.transport_lines, cs.TRANSPORT_TRAIN_SPEC, cs.parse_transport)):
            for line in lines:
                view = cs.prune(json.loads(line), spec_)
                for path, key in fx.walk_keys(view):
                    self.assertFalse(any(tok in key for tok in EXPOSED_TOKENS), path)
                parse(line, 1, sc.ident)
        ident = sc.ident
        native = json.loads(sc.native_lines[0])
        outcome = next(json.loads(line) for line in sc.outcome_lines if b'"inner_cv"' in line)
        for row, parse in (({**native, "program": [native["program"]]}, cs.parse_native),
                           ({**native, "cv_mean_log_ratio": [0.1]}, cs.parse_native),
                           ({**outcome, "evaluation_detail": "detail"}, cs.parse_outcome),
                           ({**outcome, "evaluation_detail": {**outcome["evaluation_detail"],
                                                              "inner_cv": {"index": 0}}}, cs.parse_outcome),
                           ({**outcome, "wall_us": "12"}, cs.parse_outcome),
                           ({**outcome, "cv_gate_passed": 1}, cs.parse_outcome),
                           ({**outcome, "node": 3}, cs.parse_outcome)):
            with self.assertRaises(cs.CorpusStop) as cm:
                parse(fx.dump(row), 7, ident)
            self.assertEqual(cm.exception.kind, "row", row.keys())
            self.assertEqual(cm.exception.context, sc.cid)
        with self.assertRaises(cs.CorpusStop) as cm:
            cs.parse_native(b"{not json\n", 1, ident)
        self.assertEqual(cm.exception.kind, "source")
        with self.assertRaises(cs.CorpusStop) as cm:
            cs.parse_native(b"[1, 2]\n", 1, ident)
        self.assertEqual(cm.exception.kind, "source")

    @staticmethod
    def _junk(rng: random.Random, depth: int = 0):
        r = rng.randrange(10 if depth < 2 else 8)
        if r == 0:
            return None
        if r == 1:
            return rng.random() < 0.5
        if r == 2:
            return rng.randrange(-10**9, 10**9)
        if r == 3:
            return rng.uniform(-1e6, 1e6)
        if r == 4:
            return rng.choice((NAN, INF, -INF, -0.0))
        if r in (5, 6, 7):
            return rng.choice(("SENTINEL-fuzz", "", "local_lambda", "aws:us-east-1", "xbabe3")) + str(rng.random())
        if r == 8:
            return [Prune._junk(rng, depth + 1) for _ in range(rng.randrange(4))]
        return {f"zz_{rng.randrange(1000)}": Prune._junk(rng, depth + 1) for _ in range(rng.randrange(4))}

    @classmethod
    def _mutate(cls, node, spec, rng: random.Random) -> int:
        """Randomize, delete or insert every key of `node` outside `spec`; recurse into allowlisted containers."""
        if not isinstance(node, dict):
            return 0
        changed = 0
        for key in list(node):
            sub = spec.get(key)
            if sub is None:
                r = rng.random()
                if r < 0.35:
                    del node[key]
                    changed += 1
                elif r < 0.85:
                    node[key] = cls._junk(rng)
                    changed += 1
            elif isinstance(sub, dict):
                changed += cls._mutate(node[key], sub, rng)
            elif isinstance(sub, list) and isinstance(node[key], list):
                for item in node[key]:
                    changed += cls._mutate(item, sub[0], rng)
        for _ in range(rng.randrange(3)):
            node[f"zz_fuzz_{rng.randrange(10**6)}"] = cls._junk(rng)
            changed += 1
        return changed

    def test_fuzz_non_allowlisted_keys(self):
        contexts = [synth("FZF", cohort="frozen", fold=2), synth("FZI", cohort="incoming", fold=1, item="FZ")]
        vocab = fx.context_vocab(*contexts)
        baseline = {}
        for sc in contexts:
            parsed, built = build(sc, vocab=vocab)
            baseline[sc.cid] = (canon(parsed), canon(built.records), canon(built.counts))
        rng = random.Random(20261005)
        total_changes = 0
        specs = (cs.NATIVE_TRAIN_SPEC, cs.OUTCOME_TRAIN_SPEC, cs.TRANSPORT_TRAIN_SPEC)
        for round_no in range(200):
            for sc in contexts:
                lines = []
                for rows, spec in zip((sc.native_rows, sc.outcome_rows, sc.transport_rows), specs):
                    mutated = []
                    for row in rows:
                        row = fx.deep(row)
                        total_changes += self._mutate(row, spec, rng)
                        mutated.append((json.dumps(row) + "\n").encode())
                    lines.append(mutated)
                parsed, built = build(sc, lines, vocab=vocab)
                got = (canon(parsed), canon(built.records), canon(built.counts))
                self.assertEqual(got, baseline[sc.cid], f"round {round_no} context {sc.spec.name}")
        self.assertGreater(total_changes, 200 * 60)
        # sensitivity control: an allowlisted train value does change the output (a gate-passing row keeps passing)
        sc = contexts[0]
        rows = [fx.deep(r) for r in sc.outcome_rows]
        target = next(r for r in rows if r.get("evaluation_detail", {}).get("inner_cv") and r.get("cv_gate_passed"))
        target["evaluation_detail"]["inner_cv"][0]["metrics"]["rmse"] *= 0.37
        # the writers' cv_log_ratio_sd follows the changed detail (else the canonical recompute is rightly a STOP)
        detail = target["evaluation_detail"]["inner_cv"]
        by_index = {e["index"]: e for e in detail}
        sd = cs.canonical_cv_log_ratio_sd([by_index[j]["metrics"]["rmse"] for j in (0, 1)],
                                          [by_index[j]["baseline_metrics"]["rmse"] for j in (0, 1)])
        target["cv_log_ratio_sd"] = sd
        natives = [fx.deep(r) for r in sc.native_rows]
        for r in natives:
            if r["program"]["program_sha256"] == target["program_sha256"]:
                r["cv_log_ratio_sd"] = sd
        _, built = build(sc, ([fx.dump(r) for r in natives], [fx.dump(r) for r in rows], sc.transport_lines),
                         vocab=vocab)
        self.assertNotEqual(canon(built.records), baseline[sc.cid][1])
        with self.assertRaises(cs.CorpusStop) as cm:             # the stale writer value of the changed detail
            target["cv_log_ratio_sd"] = bump(sd, 2)
            build(sc, ([fx.dump(r) for r in natives], [fx.dump(r) for r in rows], sc.transport_lines), vocab=vocab)
        self.assertIn(cm.exception.kind, ("cv_sd_recompute", "native_outcome_disagreement"))


class Outcome(unittest.TestCase):
    def test_classification_table(self):
        cases = [
            ("evolution", nat("completed"), out("evolution", "completed", "completed_transport"), None, "completed"),
            ("evolution", nat("failed"), out("evolution", "failed", "completed_transport"), None, "failed:unclassified"),
            ("evolution", nat("timeout"), out("evolution", "timeout", "completed_transport"), None, "timeout@90000ms"),
            ("evolution", nat("completed"), out("evolution", "completed", "cached"),
             out("bootstrap", "failed", "cached"), "completed"),
            ("bootstrap", None, None, out("bootstrap", "completed", "cached"), "completed"),
            ("bootstrap", None, None, out("bootstrap", "failed", "completed_transport"), "failed:unclassified"),
            ("bootstrap", None, None, out("bootstrap", "timeout", "completed_transport"), "timeout@90000ms"),
            ("bootstrap", None, None, out("bootstrap", "missing", "recovered_after_transport_error"), "missing"),
            ("bootstrap", None, None, out("bootstrap", "missing", None), "missing"),
            ("bootstrap", None, None, out("bootstrap", "missing", "cached"), "missing"),
            ("transport_only", None, None, None, "missing"),
        ]
        for reason in programs.ABANDON_REASONS:
            cases.append(("evolution", nat("abandoned", reason), out("evolution", "missing", reason), None,
                          f"abandoned:{reason}"))
            cases.append(("bootstrap", None, None, out("bootstrap", "missing", reason), f"abandoned:{reason}"))
        for origin, n, e, b, want in cases:
            with self.subTest(origin=origin, want=want, n=n and n.status, e=e and e.record_status):
                counts = Counter()
                self.assertEqual(cs.classify_outcome(origin, n, e, b, counts), want)
                self.assertEqual(counts["native_without_outcome"], 0)
                self.assertIn(want, cs.OUTCOME_CLASSES)
        counts = Counter()
        for st, want in (("completed", "completed"), ("failed", "failed:unclassified"),
                         ("timeout", "timeout@90000ms"), ("abandoned", "abandoned:runtime_error")):
            n = nat(st, "runtime_error" if st == "abandoned" else None)
            self.assertEqual(cs.classify_outcome("evolution", n, None, None, counts), want)
        self.assertEqual(counts["native_without_outcome"], 4)
        self.assertEqual(cs.classify_outcome("evolution", nat("completed"), None, None), "completed")
        # end to end on real-shaped rows: every program's class equals the fixture's construction table
        for sc in (synth(), synth("T1", cohort="incoming", fold=3, item="T")):
            _, built = build(sc)
            for sha, e in sc.expect.items():
                self.assertEqual(built.outcomes[sha], e["outcome"], (sc.spec.name, e["kind"]))
            self.assertTrue(all(not o.startswith("failed:") or o == "failed:unclassified"
                                for o in built.outcomes.values()))
            self.assertEqual(built.counts["evolution_outcome_duplicates"], 1)
            self.assertEqual(sum(built.counts["by_outcome"].values()) + built.counts["refused"],
                             sc.stats()["unique_programs"])

    def test_disagreements_stop(self):
        stops = [
            (("evolution", nat("completed"), out("evolution", "failed", "completed_transport"), None),
             "native_outcome_disagreement"),
            (("evolution", nat("failed"), out("evolution", "completed", "completed_transport"), None),
             "native_outcome_disagreement"),
            (("evolution", nat("abandoned", "transport_error"),
              out("evolution", "missing", "capacity_deferred_before_claim"), None), "native_outcome_disagreement"),
            (("evolution", nat("abandoned", "transport_error"), out("evolution", "completed", "transport_error"), None),
             "native_outcome_disagreement"),
            (("evolution", nat("timeout"), out("evolution", "missing", "transport_error"), None),
             "native_outcome_disagreement"),
            (("bootstrap", None, out("evolution", "completed", "cached"), out("bootstrap", "completed", "cached")),
             "outcome_not_native"),
            (("transport_only", None, out("evolution", "completed", "cached"), None), "outcome_not_native"),
            ((None, None, out("evolution", "completed", "cached"), None), "outcome_not_native"),
        ]
        for args, kind in stops:
            with self.subTest(args=args[0], kind=kind), self.assertRaises(cs.CorpusStop) as cm:
                cs.classify_outcome(*args)
            self.assertEqual(cm.exception.kind, kind)
        sc = synth()
        ident = sc.ident
        native_rows = [json.loads(line) for line in sc.native_lines]
        scored = next(r for r in native_rows if r["status"] == "completed")
        abandon = next(r for r in native_rows if r["status"] == "abandoned")
        outcome = json.loads(sc.outcome_lines[-1])
        transport = next(json.loads(line) for line in sc.transport_lines if b"_telemetry_view" in line)
        row_stops = [
            (cs.parse_native, {**scored, "status": "crashed"}, "row"),
            (cs.parse_native, {**abandon, "reason": "power_cut"}, "row"),
            (cs.parse_native, {**abandon, "program_sha256": "b" * 64}, "row"),
            (cs.parse_native, {**scored, "context_sha256": "f" * 64}, "identity"),
            (cs.parse_native, {**scored, "program": {**scored["program"], "program_sha256": "b" * 64}}, "sha"),
            (cs.parse_outcome, {**outcome, "phase": "warmup"}, "row"),
            (cs.parse_outcome, {**outcome, "record_status": "lost"}, "row"),
            (cs.parse_outcome, {**outcome, "view_id": "other__r0__f1"}, "identity"),
            (cs.parse_outcome, {**outcome, "outer_fold": sc.spec.fold + 1}, "identity"),
            (cs.parse_outcome, {**outcome, "dataset_hash": "0" * 16}, "identity"),
            (cs.parse_outcome, {**outcome, "program_sha256": "b" * 64}, "row"),
            (cs.parse_transport, {**transport, "_telemetry_view_context_sha256": "f" * 64}, "identity"),
            (cs.parse_transport, {**transport, "program_sha256": "b" * 64}, "row"),
        ]
        for parse, row, kind in row_stops:
            with self.subTest(parse=parse.__name__, kind=kind), self.assertRaises(cs.CorpusStop) as cm:
                parse(fx.dump(row), 3, ident)
            self.assertEqual(cm.exception.kind, kind)
        # an evolution outcome row for a bootstrap-only program, and native/outcome cv disagreement, in the assembly
        boot_line = next(line for line in sc.outcome_lines if b'"phase":"bootstrap"' in line
                         and b'"record_status":"completed"' in line and b'"historical_seed"' in line)
        boot = json.loads(boot_line)
        stray = {**boot, "phase": "evolution", "generation": 9}
        lines = (sc.native_lines, sc.outcome_lines + [fx.dump(stray)], sc.transport_lines)
        with self.assertRaises(cs.CorpusStop) as cm:
            build(sc, lines)
        self.assertEqual(cm.exception.kind, "outcome_not_native")
        originals = [json.loads(line) for line in sc.outcome_lines]
        idx = next(i for i, r in enumerate(originals) if r["phase"] == "evolution" and r.get("cv_mean_log_ratio"))
        for key, change in (("cv_mean_log_ratio", lambda v: v + 1e-12), ("cv_log_ratio_sd", lambda v: v * 2),
                            ("cv_gate_passed", lambda v: not v)):
            evo_rows = [fx.deep(r) for r in originals]
            evo_rows[idx][key] = change(evo_rows[idx][key])
            with self.subTest(key=key), self.assertRaises(cs.CorpusStop) as cm:
                build(sc, (sc.native_lines, [fx.dump(r) for r in evo_rows], sc.transport_lines))
            self.assertEqual(cm.exception.kind, "native_outcome_disagreement")

    def _edge_context(self):
        """A frozen context whose natives carry the §6.3 edge metrics (each native row agrees with its outcome)."""
        sc = synth("EDGE", n_native=0, n_boot_only=0, boot_overlap=False, n_tonly=0)
        x = fx.Sentinels("edge")
        ident = sc.ident
        base = list(sc.base)
        specs = [
            ("normal", fx.cv_block(base, 0.3)),
            ("normal2", fx.cv_block(base, -0.2)),
            ("nan", {"rmse_j": [NAN, base[1]], "cv_mean_rmse_ratio": NAN, "cv_mean_log_ratio": NAN,
                     "cv_log_ratio_sd": NAN, "cv_gate_passed": False}),
            ("inf", {"rmse_j": [INF, base[1]], "cv_mean_rmse_ratio": INF, "cv_mean_log_ratio": INF,
                     "cv_log_ratio_sd": INF, "cv_gate_passed": False}),
            ("negative", {"rmse_j": [-0.5, base[1]], "cv_mean_rmse_ratio": 0.1, "cv_mean_log_ratio": None,
                          "cv_log_ratio_sd": None, "cv_gate_passed": True}),
            ("exact_fit", {"rmse_j": [0.0, 0.0], "cv_mean_rmse_ratio": 0.0, "cv_mean_log_ratio": -INF,
                           "cv_log_ratio_sd": NAN, "cv_gate_passed": True}),
            ("detailless_ninf", {"cv_mean_rmse_ratio": 0.0, "cv_mean_log_ratio": -INF, "cv_log_ratio_sd": NAN,
                                 "cv_gate_passed": True}),
            ("detailless_nan", {"cv_mean_rmse_ratio": NAN, "cv_mean_log_ratio": NAN, "cv_log_ratio_sd": NAN,
                                "cv_gate_passed": None}),
        ]
        native, evo, names = [], [], {}
        for k, (name, m) in enumerate(specs):
            prog = fx.program_obj(f"(pipe X (std_scale) (ridge lambda_frac=0.{k + 1}))")
            names[prog["program_sha256"]] = name
            native.append(fx.dump(fx.native_row(ident, prog, "completed", k=k, x=x, metrics=m)))
            evo.append(fx.dump(fx.outcome_row(ident, prog, phase="evolution", record_status="completed",
                                              transport_status="completed_transport", generation=5 + k, x=x,
                                              node="xbabe1", backend="local_lambda", wall_us=1000 + k, metrics=m,
                                              base=base, detail="rmse_j" in m)))
        return sc, (native, evo, []), names

    def test_nonfinite_and_exact_fit(self):
        sc, lines, names = self._edge_context()
        vocab = fx.tiny_vocab(json.loads(line)["program"]["dsl"] for line in lines[0])
        _, built = build(sc, lines, vocab=vocab)
        by_name = {names[r["program_sha256"]]: r for r in built.records}
        lab = {names[r.program_sha256]: r for r in built.record_labels}
        for name in ("nan", "inf", "negative"):
            self.assertEqual(by_name[name]["outcome"], "failed:nonfinite_metric", name)
            self.assertEqual(by_name[name]["outcome"], labels.normalize_outcome({**by_name[name], "outcome": "completed"}))
            self.assertIsNone(by_name[name]["gate_pass"])
            self.assertEqual(lab[name].feas_class, labels.FEAS_FAILED)
        for name in ("detailless_ninf", "detailless_nan"):
            self.assertEqual(by_name[name]["outcome"], "completed", name)
            self.assertIsNone(by_name[name]["rmse_j"])
            self.assertEqual(lab[name].q_status, "unknown_detailless_nonfinite", name)
            self.assertIsNone(lab[name].q)
        self.assertEqual(by_name["exact_fit"]["outcome"], "completed")
        self.assertEqual(by_name["exact_fit"]["rmse_j"], [0.0, 0.0])
        self.assertEqual(lab["exact_fit"].q_status, "known")
        self.assertTrue(math.isfinite(lab["exact_fit"].q))
        self.assertEqual(lab["exact_fit"].q, built.context_label.q_cap)
        self.assertTrue(by_name["exact_fit"]["gate_pass"])
        self.assertEqual(built.context_label.status, "OK")
        self.assertEqual(built.counts["by_outcome"], {"completed": 5, "failed:nonfinite_metric": 3})
        self.assertEqual((built.counts["detailed_completed"], built.counts["detailless_completed"]), (3, 2))
        self.assertEqual(built.counts["gate_unverified"], 0)
        self.assertEqual(built.context_label.counts["nonfinite"], 3)
        for rec in built.records:
            self.assertEqual(rec["outcome"], labels.normalize_outcome(rec))


def completed_lines(sc: fx.SynthContext, items, *, tag: str):
    """Native and evolution lines of completed programs [(name, metrics, baseline pair or None)], node xbabe1."""
    x = fx.Sentinels(tag)
    native, evo, names = [], [], {}
    for k, (name, m, base) in enumerate(items):
        prog = fx.program_obj(f"(pipe X (std_scale) (ridge lambda_frac=0.{k + 1}{len(tag)}))")
        names[prog["program_sha256"]] = name
        native.append(fx.dump(fx.native_row(sc.ident, prog, "completed", k=k, x=x, metrics=m)))
        evo.append(fx.dump(fx.outcome_row(sc.ident, prog, phase="evolution", record_status="completed",
                                          transport_status="completed_transport", generation=5 + k, x=x, node="xbabe1",
                                          backend="local_lambda", wall_us=1000 + k, metrics=m,
                                          base=base or sc.base, detail="rmse_j" in m)))
    return (native, evo, []), names


class Records(unittest.TestCase):
    def test_assembly_refusals_raw_gate_and_labels(self):
        sc = synth("LONG", split_role="dev", n_long=2)
        vocab = fx.context_vocab(sc)
        (native, outcome, transport), _ = build(sc, vocab=vocab, max_seq=4096)
        accs, index, _ = cs.accumulate_programs(native, outcome, transport)
        short = max(len(dsl_tokenizer.tokenize(sc.dsl[p], vocab)) for p, e in sc.expect.items() if not e.get("long"))
        seeds = {next(iter(sorted(sc.dsl)))}
        cr = cs.build_context_records(sc.ctx_meta(), accs, vocab, short, seeds, attempt_index=index)
        longs = sorted(p for p, e in sc.expect.items() if e.get("long"))
        self.assertEqual([r["program_sha256"] for r in cr.refused], longs)
        self.assertTrue(all(r["reason"] == "over_max_seq" and r["split_role"] == "dev" and r["length"] > short
                            for r in cr.refused))
        self.assertEqual(len(cr.records) + len(cr.refused), sc.stats()["unique_programs"])
        self.assertEqual(sum(cr.counts["refused_by_outcome"].values()), 2)
        self.assertEqual(cs.refused_table(cr.refused).num_rows, 2)
        self.assertEqual([r["program_sha256"] for r in cr.records], sorted(r["program_sha256"] for r in cr.records))
        self.assertEqual([lab.program_sha256 for lab in cr.record_labels], [r["program_sha256"] for r in cr.records])
        for rec, lab in zip(cr.records, cr.record_labels):
            self.assertEqual(list(rec), cs.TRAIN_RECORD_SCHEMA.names)
            self.assertEqual(rec["gate_pass"], lab.gate_pass)
            self.assertEqual(rec["outcome"], lab.outcome)
            self.assertEqual(rec["shared_seed"], rec["program_sha256"] in seeds)
            self.assertEqual(rec["skeleton_sha256"], dsl_tokenizer.skeleton_sha256(sc.dsl[rec["program_sha256"]]))
            self.assertEqual(rec["tokens"], dsl_tokenizer.tokenize(sc.dsl[rec["program_sha256"]], vocab))
        relabel = labels.label_context({**sc.ctx_meta(), "split_role": "dev"}, cr.records)
        self.assertEqual(canon(relabel[0].to_json()), canon(cr.context_label.to_json()))
        ti = cs.train_input(sc.jtt1_raw, sc.prepared, sc.ident)
        row = cs.build_context_row(sc.ctx_meta(), ti, cr.context_label, len(cr.records), len(cr.refused))
        self.assertEqual(list(row), cs.TRAIN_CONTEXT_SCHEMA.names)
        self.assertEqual((row["s_c"], row["eps_c"], row["n_refused"], row["frozen_fold0"]),
                         (labels.robust_scale(ti.y), labels.epsilon(ti.y, ti.s_c), 2, False))
        self.assertTrue(row["eligibility_source"].endswith(cs.LEAK_PASS_SUFFIX))
        self.assertEqual(cs.contexts_table([row]).num_rows, 1)
        with self.assertRaises(cs.CorpusStop) as cm:
            cs.build_context_records({**sc.ctx_meta(), "split_role": "train_fit"}, accs, vocab, short, seeds,
                                     attempt_index=index)
        self.assertEqual(cm.exception.kind, "max_seq")
        for key in ("context_schema", "y_train", "family_id", "dataset_hash"):
            meta = {k: v for k, v in sc.ctx_meta().items() if k != key}
            with self.subTest(missing=key), self.assertRaises(cs.CorpusStop) as cm:
                cs.build_context_records(meta, accs, vocab, 4096, seeds)
            self.assertEqual((cm.exception.kind, cm.exception.context), ("input", sc.cid))
        # RAW: a native program whose text matches no form (built directly; the parsers already refuse non-forms)
        raw_dsl = "(pipe X (std_scale)) trailing"
        raw_acc = cs.ProgramAcc(fx.sha(raw_dsl), native=nat("failed", sha=fx.sha(raw_dsl), dsl=raw_dsl))
        with self.assertRaises(cs.CorpusStop) as cm:
            cs.build_context_records(sc.ctx_meta(), [raw_acc], vocab, 4096, ())
        self.assertEqual(cm.exception.kind, "raw_program")

    def test_gate_ties_drift_and_baseline_disagreement(self):
        sc = synth("GATE", n_native=0, n_boot_only=0, boot_overlap=False, n_tonly=0)
        b = list(sc.base)
        r = [1.05 * b[0], 1.05 * b[1]]
        label_gate = 0.5 * (r[0] / b[0] + r[1] / b[1]) <= 1.05
        # the writers' cv_log_ratio_sd is the canonical value of the row's own inner-CV detail (a writer value more
        # than 1 ULP from it is a STOP of its own, cv_sd_recompute)
        tie = {"rmse_j": r, "cv_mean_rmse_ratio": 1.05 + 5e-10, "cv_mean_log_ratio": -0.0487,
               "cv_log_ratio_sd": cs.canonical_cv_log_ratio_sd(r, b), "cv_gate_passed": not label_gate}
        far_r = [1.3 * b[0], 1.2 * b[1]]
        far = {"rmse_j": far_r, "cv_mean_rmse_ratio": 1.25, "cv_mean_log_ratio": -0.22,
               "cv_log_ratio_sd": cs.canonical_cv_log_ratio_sd(far_r, b), "cv_gate_passed": True}
        ok = fx.cv_block(b, 0.4)
        lines, _ = completed_lines(sc, [("ok", ok, None), ("tie", tie, None)], tag="tie")
        vocab = fx.tiny_vocab(json.loads(line)["program"]["dsl"] for line in lines[0])
        _, built = build(sc, lines, vocab=vocab)
        self.assertEqual(built.counts["gate_boundary_ties"], 1)
        lines, _ = completed_lines(sc, [("ok", ok, None), ("far", far, None)], tag="far")
        vocab = fx.tiny_vocab(json.loads(line)["program"]["dsl"] for line in lines[0])
        with self.assertRaises(cs.CorpusStop) as cm:
            build(sc, lines, vocab=vocab)
        self.assertEqual(cm.exception.kind, "gate_drift")
        lines, _ = completed_lines(sc, [("ok", ok, None), ("other", fx.cv_block([b[0] * 2, b[1]], 0.1),
                                                           [b[0] * 2, b[1]])], tag="base")
        vocab = fx.tiny_vocab(json.loads(line)["program"]["dsl"] for line in lines[0])
        with self.assertRaises(cs.CorpusStop) as cm:
            build(sc, lines, vocab=vocab)
        self.assertEqual(cm.exception.kind, "label")


class InnerCV(unittest.TestCase):
    @staticmethod
    def entry(i, r, b, kind="inner_cv"):
        return {"index": i, "kind": kind, "metrics": {"rmse": r, "mae": 9.9}, "baseline_metrics": {"rmse": b, "n": 800},
                "training_metrics": {"rmse": 0.01}, "wall_us": 7}

    def detail(self, entries, mode="partitioned_kfold", official=None):
        d = {"inner_cv_mode": mode, "inner_cv": entries, "schema": "jope.gp-signal-evaluation/v7"}
        if official is not None:
            d["official_test"] = official
        return d

    def test_pairs_by_index(self):
        e = self.entry
        self.assertEqual(cs.inner_cv_pairs(self.detail([e(1, 0.2, 0.3), e(0, 0.1, 0.4)])), ([0.1, 0.2], [0.4, 0.3]))
        self.assertEqual(cs.inner_cv_pairs(self.detail([e(0, 0.1, 0.4), e(1, 0.2, 0.3)])), ([0.1, 0.2], [0.4, 0.3]))
        self.assertEqual(cs.inner_cv_pairs(self.detail([e(1, 0.2, 0.3)])), ([None, 0.2], [None, 0.3]))
        self.assertEqual(cs.inner_cv_pairs(self.detail([e(0, 0.1, 0.4)])), ([0.1, None], [0.4, None]))
        self.assertEqual(cs.inner_cv_pairs(self.detail([e(0, None, 0.4), e(1, 2, 0.3)])), ([None, 2.0], [0.4, 0.3]))
        self.assertEqual(cs.inner_cv_pairs(self.detail([{"index": 0, "kind": "inner_cv", "metrics": None}])),
                         ([None, None], [None, None]))
        for nothing in (None, {}, {"inner_cv": None}, {"inner_cv": []}, {"inner_cv_mode": "other"},
                        {"inner_cv_mode": "other", "inner_cv": []}):
            self.assertEqual(cs.inner_cv_pairs(nothing), (None, None), nothing)
        bad = [
            self.detail([e(0, 0.1, 0.4), e(0, 0.2, 0.3)]),
            self.detail([e(2, 0.1, 0.4)]),
            self.detail([e(0, 0.1, 0.4, kind="official_test")]),
            self.detail([e(0, 0.1, 0.4)], mode="kfold"),
            self.detail([e(0, 0.1, 0.4), e(1, 0.2, 0.3), e(1, 0.2, 0.3)]),
            self.detail([{**e(0, 0.1, 0.4), "index": True}]),
            self.detail([{**e(0, 0.1, 0.4), "index": "0"}]),
            self.detail([e(0, "0.1", 0.4)]),
            self.detail([e(0, 0.1, True)]),
            self.detail([{**e(0, 0.1, 0.4), "metrics": [0.1]}]),
            self.detail([None]),
            {"inner_cv_mode": "partitioned_kfold", "inner_cv": {"index": 0}},
            ["not", "a", "detail"],
        ]
        for d in bad:
            with self.subTest(detail=str(d)[:80]), self.assertRaises(cs.CorpusStop) as cm:
                cs.inner_cv_pairs(d)
            self.assertEqual(cm.exception.kind, "inner_cv_shape")
        rng = random.Random(7)
        want = cs.inner_cv_pairs(self.detail([e(1, 0.2, 0.3), e(0, 0.1, 0.4)]))
        for _ in range(50):
            official = {k: Prune._junk(rng) for k in ("metrics", "baseline_metrics", "index", "kind", "rmse")}
            got = cs.inner_cv_pairs(self.detail([e(1, 0.2, 0.3), e(0, 0.1, 0.4)], official=official))
            self.assertEqual(got, want)
        # through parse_outcome: entries in swapped order are placed by index
        sc = synth()
        x = fx.Sentinels("icv")
        prog = fx.program_obj("(pipe X (std_scale) (ridge lambda_frac=0.5))")
        row = fx.outcome_row(sc.ident, prog, phase="evolution", record_status="completed",
                             transport_status="completed_transport", generation=5, x=x,
                             metrics=fx.cv_block(sc.base, 0.1), base=sc.base)
        row["evaluation_detail"] = fx.evaluation_detail(x, [0.11, 0.22], [0.33, 0.44], order=(1, 0))
        parsed, _ = cs.parse_outcome(fx.dump(row), 1, sc.ident)
        self.assertEqual((parsed.rmse_j, parsed.baseline_rmse_j), ((0.11, 0.22), (0.33, 0.44)))
        row["evaluation_detail"]["official_test"] = {"metrics": "garbage", "kind": [1, 2]}
        self.assertEqual(cs.parse_outcome(fx.dump(row), 1, sc.ident)[0], parsed)


class Backend(unittest.TestCase):
    def test_derivation(self):
        local = ("xbabe3", "xbabe2-extra", "xbabe1:7431", "xbabe12", "xbabe3-extra:7432")
        # red-team CL-4: the attempt node's `aws` prefix decides; the region-less aws:<function> node of the hybrid
        # gateway (real epoch-0 rows) and its bare `aws` pool are AWS Lambda too
        aws = (f"aws:ca-central-1:{fx.AWS_FN}", "aws:us-east-1", "aws:eu-west-2", "aws:ap-southeast-2:fn.v1_x-2",
               f"aws:{fx.AWS_FN}", "aws", "aws:us-east", "aws:US-east-1")
        unknown = (None, "", "local_lambda", "aws_lambda", "xbabe", "xbabe3 ", "aws:", "aws:us-east-1:", "aws::fn",
                   "AWS:us-east-1", "aws:a b", "aws-us-east-1", "gpu-node-7", 3)
        for s in local:
            self.assertEqual(cs.backend_class(s), "local", s)
        for s in aws:
            self.assertEqual(cs.backend_class(s), "aws_lambda", s)
        for s in unknown:
            self.assertEqual(cs.backend_class(s), "unknown", s)
        sha = "a" * 64

        def acc(native=None, evo=None, boot=None, attempts=(), tdsl=None):
            a = cs.ProgramAcc(sha, native=native, evo=evo, boot=boot, transport_dsl=tdsl)
            a.attempts.extend(attempts)
            return a

        n = nat("completed")
        cases = [
            (acc(n, out("evolution", "completed", node="xbabe3")), {}, ("local", "outcome_node")),
            (acc(n, out("evolution", "completed", node=f"aws:ca-central-1:{fx.AWS_FN}")), {},
             ("aws_lambda", "outcome_node")),
            (acc(n, out("evolution", "completed", attempt="t1")), {"t1": trans("t1", "aws:us-east-1")},
             ("aws_lambda", "transport_attempt")),
            (acc(n, out("evolution", "completed", attempt="t1"), attempts=[trans("t1", "xbabe1:7431")]), None,
             ("local", "transport_attempt")),
            (acc(n, out("evolution", "completed", attempt="t9")), {"t1": trans("t1", "aws:us-east-1")},
             ("unknown", "none")),
            (acc(n, out("evolution", "completed")), {}, ("unknown", "none")),
            (acc(n, None), {}, ("unknown", "none")),
            (acc(None, None, out("bootstrap", "completed", "cached")), {}, ("unknown", "none")),
            (acc(None, None, out("bootstrap", "completed", attempt="t2")), {"t2": trans("t2", "xbabe2-extra")},
             ("local", "transport_attempt")),
            (acc(tdsl="(pipe X (krr))", attempts=[trans("u1", "aws:us-east-1"), trans("u2", "aws:eu-west-2")]), {},
             ("aws_lambda", "transport_unanimous")),
            (acc(tdsl="(pipe X (krr))", attempts=[trans("u1", "xbabe3-extra"), trans("u2", "aws:us-west-2")]), {},
             ("unknown", "none")),
            (acc(tdsl="(pipe X (krr))", attempts=[trans("u1", "xbabe1:7431")]), {}, ("local", "transport_unanimous")),
            (acc(tdsl="(pipe X (krr))", attempts=[trans("u1", "xbabe1:7431"), trans("u2", None)]), {},
             ("unknown", "none")),
            (acc(n, out("evolution", "completed"), attempts=[trans("u1", "aws:us-east-1")]), {}, ("unknown", "none")),
        ]
        for a, index, want in cases:
            with self.subTest(want=want, origin=a.origin):
                info = cs.derive_backend(a, index)
                self.assertEqual((info.backend, info.source), want)
                self.assertIn(info.backend, cs.BACKENDS)
                self.assertIn(info.source, cs.BACKEND_SOURCES)
        weird = "gpu-node-7-very-long-host-name-here"
        info = cs.derive_backend(acc(n, out("evolution", "completed", node=weird)))
        self.assertEqual((info.backend, info.source, info.unknown_prefix), ("unknown", "outcome_node", weird[:24]))
        gateway = cs.derive_backend(acc(n, out("evolution", "completed", node=f"aws:{fx.AWS_FN}")))
        self.assertEqual((gateway.backend, gateway.source, gateway.raw, gateway.unknown_prefix),
                         ("aws_lambda", "outcome_node", f"aws:{fx.AWS_FN}", None))
        bare = cs.derive_backend(acc(tdsl="(pipe X (krr))", attempts=[trans("u1", "aws"), trans("u2", "aws:us-east-1")]))
        self.assertEqual((bare.backend, bare.source), ("aws_lambda", "transport_unanimous"))
        index, dups = cs.index_attempts([trans("t1", "xbabe1"), trans(None, "x"), trans("t1", "aws:us-east-1"),
                                         trans("t2", "xbabe2")])
        self.assertEqual((index["t1"].pool, dups, sorted(index)), ("aws:us-east-1", 1, ["t1", "t2"]))
        # the row's own backend string is audit-only: local_lambda with an aws pool is aws_lambda, a disagreement
        sc = synth()
        rows = [json.loads(line) for line in sc.outcome_lines]
        row = next(r for r in rows if r["phase"] == "evolution" and r.get("backend") == "local_lambda"
                   and "node" not in r and r["record_status"] == "completed" and r.get("attempt_id"))
        parsed, full = cs.parse_outcome(fx.dump(row), 1, sc.ident)
        tparsed = [cs.parse_transport(line, i + 1, sc.ident)[0] for i, line in enumerate(sc.transport_lines)]
        attempt_index, _ = cs.index_attempts(tparsed)
        a = cs.ProgramAcc(parsed.program_sha256, native=cs.parse_native(next(
            line for line in sc.native_lines if parsed.program_sha256.encode() in line), 1, sc.ident)[0], evo=parsed)
        info = cs.derive_backend(a, attempt_index)
        self.assertEqual((info.backend, info.source, info.raw), ("aws_lambda", "transport_attempt", "aws:us-east-2"))
        lineage = cs.lineage_row(context_sha256=sc.cid, program_sha256=parsed.program_sha256, split_role="train_fit",
                                 origin="evolution", outcome=cs.project_outcome_lineage(full), backend=info)
        self.assertEqual((lineage["backend_field"], lineage["backend_field_agrees"]), ("local_lambda", False))
        self.assertTrue(cs.backend_field_agrees("aws_lambda", "aws_lambda"))
        self.assertTrue(cs.backend_field_agrees("local_lambda", "local"))
        self.assertIsNone(cs.backend_field_agrees(None, "local"))
        # every program of real-shaped contexts: backend and source equal the fixture's construction table
        for scx in (sc, synth("T1", cohort="incoming", fold=3, item="T")):
            _, built = build(scx)
            for psha, e in scx.expect.items():
                b = built.backends[psha]
                self.assertEqual((b.backend, b.source), (e["backend"], e["backend_source"]), e["kind"])
            self.assertEqual(sum(built.counts["backend_source"].values()), len(scx.expect))
        # prepared.json execution_backend is never consulted: the projection drops it and records are unchanged
        proj = cs.project_prepared(sc.prepared)
        self.assertNotIn("execution_backend", proj)
        self.assertEqual(set(proj), set(cs.PREPARED_CONTEXT_KEYS))
        other = fx.deep(sc.prepared)
        other["context"]["execution_backend"] = "local"
        self.assertEqual(cs.project_prepared(other), proj)
        ti_a = cs.train_input(sc.jtt1_raw, sc.prepared, sc.ident)
        ti_b = cs.train_input(sc.jtt1_raw, other, sc.ident)
        self.assertEqual((ti_a.s_c, ti_a.eps_c, ti_a.prepared_checks), (ti_b.s_c, ti_b.eps_c, ti_b.prepared_checks))
        # an unrecognized node is recorded by its 24-character prefix in the context counts
        rows2 = [fx.deep(r) for r in sc.outcome_rows]
        target = next(r for r in rows2 if r.get("node") == "xbabe3")
        target["node"] = weird
        _, built = build(sc, (sc.native_lines, [fx.dump(r) for r in rows2], sc.transport_lines))
        self.assertEqual(built.counts["unknown_backend_prefixes"], {weird[:24]: 1})
        self.assertEqual(built.backends[target["program_sha256"]].backend, "unknown")


class Jtt1(unittest.TestCase):
    def test_train_targets_and_counterfactual(self):
        n_train, n_test, p = 40, 13, 4
        X, y = fx.jtt1_matrix(n_train, n_test, p, tag="jt")
        raw = fx.jtt1_raw(X, y, n_train)
        ti = cs.train_input(raw)
        self.assertEqual((ti.n_train, ti.n_test, ti.p, ti.inner_folds), (n_train, n_test, p, 2))
        np.testing.assert_array_equal(ti.y, y[:n_train])
        np.testing.assert_array_equal(ti.X, X[:n_train])
        self.assertEqual(ti.inner_labels.tolist(), [i % 2 for i in range(n_train)])
        self.assertEqual(ti.s_c, labels.robust_scale(y[:n_train]))
        self.assertEqual(ti.eps_c, labels.epsilon(y[:n_train], ti.s_c))
        self.assertEqual(ti.raw_sha256, fx.sha_bytes(raw))
        self.assertEqual(ti.to_json()["n_test"], n_test)
        leak = cs.leak_audit_train(ti.X, ti.y)
        flipped = fx.xor_test_rows(raw)
        a, b = fx.test_range(raw)
        self.assertEqual(b, len(raw))
        self.assertTrue(all(x != y_ for x, y_ in zip(raw[a:b], flipped[a:b])))
        self.assertEqual(flipped[:a], raw[:a])
        tf = cs.train_input(flipped)
        np.testing.assert_array_equal(tf.y, ti.y)
        np.testing.assert_array_equal(tf.X, ti.X)
        self.assertEqual((tf.s_c, tf.eps_c, tf.train_rows_sha256), (ti.s_c, ti.eps_c, ti.train_rows_sha256))
        self.assertNotEqual(tf.raw_sha256, ti.raw_sha256)
        self.assertEqual(cs.leak_audit_train(tf.X, tf.y), leak)
        # NaN in an outer-test target is never decoded; NaN in an outer-train target is a STOP
        y_test_nan = y.copy()
        y_test_nan[n_train + 3] = NAN
        self.assertEqual(cs.train_input(fx.jtt1_raw(X, y_test_nan, n_train)).s_c, ti.s_c)
        y_train_nan = y.copy()
        y_train_nan[5] = NAN
        for bad_y in (y_train_nan, np.where(np.arange(len(y)) == 7, INF, y)):
            with self.assertRaises(cs.CorpusStop) as cm:
                cs.train_input(fx.jtt1_raw(X, bad_y, n_train))
            self.assertEqual(cm.exception.kind, "nonfinite_target")
        # format refusals
        for bad_raw in (b"JTT2" + raw[4:], raw + b"\0", raw[:-1], raw[:10]):
            with self.assertRaises(cs.CorpusStop) as cm:
                cs.train_input(bad_raw)
            self.assertEqual(cm.exception.kind, "jtt1")
        with self.assertRaises(cs.CorpusStop) as cm:
            cs.train_input(fx.jtt1_raw(X, y, n_train, inner_folds=3))
        self.assertEqual(cm.exception.kind, "jtt1_shape")
        # prepared.json cross-checks
        ident = {"context_sha256": fx.sha("ctx:jt"), "view_id": "jt__r0__f2", "dataset_hash": "jt", "fold": 2}
        good = fx.prepared_doc(ident, n_train, n_test, p, raw=raw)
        checks = cs.train_input(raw, good, ident).prepared_checks
        self.assertTrue(checks and all(checks.values()))
        self.assertEqual(len(checks), 9)
        mismatches = [({"train_rows": n_train + 1}, "jtt1_shape"), ({"encoded_features": p + 1}, "jtt1_shape"),
                      ({"train_rows": float(n_train)}, "jtt1_shape"), ({"candidate_timeout_ms": 60000}, "timeout_pin"),
                      ({"evaluator_sha256": "0" * 64}, "evaluator_pin"),
                      ({"inner_cv_schema": "jope.inner-cv/kfold/v1"}, "inner_cv_pin"),
                      ({"context_sha256": "f" * 64}, "prepared_identity"), ({"outer_fold": 3}, "prepared_identity"),
                      ({"dataset_hash": "other"}, "prepared_identity")]
        for override, kind in mismatches:
            with self.subTest(override=override), self.assertRaises(cs.CorpusStop) as cm:
                cs.train_input(raw, fx.prepared_doc(ident, n_train, n_test, p, raw=raw, overrides=override), ident)
            self.assertEqual(cm.exception.kind, kind)
        body = json.dumps(good).encode()
        ref = {"sha256": fx.sha_bytes(body), "bytes": len(body)}
        proj = cs.project_prepared(body, ref)
        self.assertEqual(set(proj), set(cs.PREPARED_CONTEXT_KEYS))
        for never in ("execution_backend", "route", "test_rows", "intentional_test_adaptation", "benchmark_claim_allowed"):
            self.assertNotIn(never, proj)
        for bad_ref in ({"sha256": "0" * 64, "bytes": len(body)}, {"sha256": ref["sha256"], "bytes": len(body) + 1}):
            with self.assertRaises(cs.CorpusStop) as cm:
                cs.project_prepared(body, bad_ref)
            self.assertEqual(cm.exception.kind, "prepared_pin")
        self.assertEqual(cs.check_prepared(proj, {"n_train": n_train, "p": p, "inner_folds": 2}), cs.check_prepared(
            good, {"n_train": n_train, "p": p, "inner_folds": 2}))


def reference_audit(X: np.ndarray, y: np.ndarray):
    """audit_target_leakage.py:60-96, transcribed (the matrix given is all the caller decodes)."""
    target_sd = float(np.std(y))
    exact, affine = [], []
    for feature_index in range(X.shape[1]):
        values = X[:, feature_index]
        feature_sd = float(np.std(values))
        if np.array_equal(values, y):
            exact.append(feature_index)
        if feature_sd <= 0 or target_sd <= 0:
            continue
        slope = float(np.cov(values, y, ddof=0)[0, 1] / (feature_sd * feature_sd))
        intercept = float(np.mean(y) - slope * np.mean(values))
        relative_rmse = float(np.sqrt(np.mean((y - (slope * values + intercept)) ** 2)) / target_sd)
        if relative_rmse <= 1e-10:
            affine.append((feature_index, relative_rmse))
    return tuple(exact), tuple(affine), bool(exact or affine)


class Leak(unittest.TestCase):
    def _orthogonal(self, y: np.ndarray, seed: int) -> np.ndarray:
        """A vector orthogonal to 1 and y with the same population SD as y."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(len(y))
        yc = y - y.mean()
        z = z - z.mean()
        z = z - (z @ yc) / (yc @ yc) * yc
        return z / np.std(z) * np.std(y)

    def test_exact_affine_and_edges(self):
        n_train, n_test, p = 64, 20, 6
        X, y = fx.jtt1_matrix(n_train, n_test, p, tag="leak")
        Xt, yt = X[:n_train].copy(), y[:n_train].copy()
        clean = cs.leak_audit_train(Xt, yt)
        self.assertEqual((clean.exact, clean.affine, clean.target_leak, clean.nonfinite_columns), ((), (), False, 0))
        with np.errstate(all="ignore"):
            with self.subTest("exact"):
                X2 = Xt.copy()
                X2[:, 2] = yt
                r = cs.leak_audit_train(X2, yt)
                self.assertEqual((r.exact, r.target_leak), ((2,), True))
                self.assertEqual([i for i, _ in r.affine], [2])
            with self.subTest("affine 2y+3"):
                X2 = Xt.copy()
                X2[:, 4] = 2.0 * yt + 3.0
                r = cs.leak_audit_train(X2, yt)
                self.assertEqual((r.exact, [i for i, _ in r.affine], r.target_leak), ((), [4], True))
                self.assertLessEqual(r.affine[0][1], cs.LEAK_REL_RMSE)
            z = self._orthogonal(yt, 3)
            for delta, flags in ((1e-9, False), (1e-11, True)):
                with self.subTest(delta=delta):
                    X2 = Xt.copy()
                    X2[:, 1] = yt + delta * z
                    r = cs.leak_audit_train(X2, yt)
                    self.assertEqual(r.target_leak, flags)
                    self.assertEqual(r.exact, ())
                    ref = reference_audit(X2, yt)
                    self.assertEqual((r.exact, r.affine), ref[:2])
                    slope = np.cov(X2[:, 1], yt, ddof=0)[0, 1] / np.std(X2[:, 1]) ** 2
                    icpt = yt.mean() - slope * X2[:, 1].mean()
                    rel = np.sqrt(np.mean((yt - (slope * X2[:, 1] + icpt)) ** 2)) / np.std(yt)
                    self.assertAlmostEqual(rel / delta, 1.0, places=2)
            with self.subTest("constant y"):
                yc = np.full(n_train, 2.5)
                X2 = Xt.copy()
                X2[:, 0] = 2.5
                X2[:, 3] = 2.0 * yc + 3.0
                r = cs.leak_audit_train(X2, yc)
                self.assertEqual((r.exact, r.affine, r.target_leak), ((0,), (), True))
            with self.subTest("non-finite columns"):
                X2 = Xt.copy()
                X2[:, 1] = NAN
                X2[3, 2] = INF
                X2[:, 5] = yt
                X2[7, 5] = NAN
                r = cs.leak_audit_train(X2, yt)
                self.assertEqual((r.exact, r.affine, r.target_leak, r.nonfinite_columns), ((), (), False, 3))
            with self.subTest("zero variance"):
                X2 = Xt.copy()
                X2[:, 3] = 7.0
                r = cs.leak_audit_train(X2, yt)
                self.assertEqual((r.exact, r.affine, r.target_leak), ((), (), False))
            rng = np.random.default_rng(11)
            for trial in range(25):
                Xr = rng.standard_normal((50, 7))
                yr = rng.standard_normal(50)
                if trial % 3 == 0:
                    Xr[:, trial % 7] = yr
                if trial % 4 == 1:
                    Xr[:, (trial + 3) % 7] = -0.5 * yr + 11.0
                if trial % 5 == 2:
                    Xr[:, 0] = 1.0
                r = cs.leak_audit_train(Xr, yr)
                self.assertEqual((r.exact, r.affine, r.target_leak), reference_audit(Xr, yr), trial)
        # a leak only in outer-test rows is not a leak; a leak only in outer-train rows is
        for leak, want in (("test_only", False), ("train_only", True), ("exact", True), ("affine", True)):
            with self.subTest(leak=leak):
                X3, y3 = fx.jtt1_matrix(n_train, n_test, p, tag="leak", leak=leak)
                ti = cs.train_input(fx.jtt1_raw(X3, y3, n_train))
                r = cs.leak_audit_train(ti.X, ti.y)
                self.assertEqual(r.target_leak, want)
                self.assertEqual((r.exact, r.affine, r.target_leak), reference_audit(X3[:n_train], y3[:n_train]))
                if leak == "train_only":
                    self.assertFalse(reference_audit(X3, y3)[2])      # the full matrix would hide it
                if leak == "test_only":
                    tf = cs.train_input(fx.xor_test_rows(fx.jtt1_raw(X3, y3, n_train)))
                    self.assertEqual(cs.leak_audit_train(tf.X, tf.y), r)
        with self.assertRaises(ValueError):
            cs.leak_audit_train(Xt[:, 0], yt)
        self.assertEqual(clean.to_json(), {"exact": [], "affine": [], "nonfinite_columns": 0, "target_leak": False})


class PackedLine(unittest.TestCase):
    @staticmethod
    def build_history(native: bytes, outcomes: bytes, transports: bytes) -> list[dict]:
        """pack_fold_history.build_history's v1 line layout, transcribed (lines as decoded JSON objects)."""
        lines = [{"type": "manifest"}]
        programs_, records = {}, []
        for line in outcomes.splitlines():
            row = json.loads(line)
            if row.get("program"):
                programs_[row["program_sha256"]] = row["program"]
            records.append({"type": "evaluation_outcome", "source_event": row})
        for line in native.splitlines():
            row = json.loads(line)
            program = row.get("program", {})
            if program.get("program_sha256"):
                programs_[program["program_sha256"]] = program
            records.append({"type": "evolution_decision", "source_event": row})
        for line in transports.splitlines():
            row = json.loads(line)
            if row.get("program"):
                programs_[row["program_sha256"]] = row["program"]
            records.append({"type": "transport_attempt", "source_event": row})
        for digest, program in sorted(programs_.items()):
            lines.append({"type": "program", "program_sha256": digest, "program": program})
        lines.extend(records)
        lines.append({"type": "summary"})
        return lines

    def test_build_history_layout(self):
        for sc in (synth(), synth("T1", cohort="incoming", fold=3, item="T")):
            stats = sc.stats()
            layout = self.build_history(sc.native_bytes, sc.outcome_bytes, sc.transport_bytes)
            self.assertEqual(len(layout), 2 + stats["unique_programs"] + stats["outcome_events"]
                             + stats["native_events"] + stats["transport_events"])
            for kind, lines, rtype in (("outcome", sc.outcome_lines, "evaluation_outcome"),
                                       ("native", sc.native_lines, "evolution_decision"),
                                       ("transport", sc.transport_lines, "transport_attempt")):
                for i, line in enumerate(lines):
                    j = cs.packed_line(kind, i, stats)
                    self.assertEqual(layout[j]["type"], rtype)
                    self.assertEqual(canon(layout[j]["source_event"]), canon(json.loads(line)))
                with self.assertRaises(ValueError):
                    cs.packed_line(kind, len(lines), stats)
            self.assertEqual(layout[cs.packed_line("manifest", 0, stats)]["type"], "manifest")
            self.assertEqual(cs.packed_line("summary", 0, stats), len(layout) - 1)
            for i in range(stats["unique_programs"]):
                self.assertEqual(layout[cs.packed_line("program", i, stats)]["type"], "program")
            summary = programs.HistorySummary(**stats, history_sha256="0" * 64)
            self.assertEqual(cs.packed_line("native", 2, summary), cs.packed_line("native", 2, stats))
        for bad in (("native", -1), ("native", True), ("nope", 0), ("manifest", 1)):
            with self.assertRaises(ValueError):
                cs.packed_line(bad[0], bad[1], stats)


class FixtureWorld(unittest.TestCase):
    def test_world_is_a_valid_p4a_epoch(self):
        """The file-backed World reconciles under P4a (programs.build + verify) and its pins load through the real
        readers, so corpus_build tests can rely on it."""
        from dltrain import jtt1
        with fx.tmpdir() as d:
            w = fx.World(Path(d) / "world")
            m = programs.build(w.epoch, w.pmap, w.seeds, Path(d) / "dict", sources=w.sources, chunk_contexts=2)
            self.assertEqual(m["totals"]["rows"], sum(sc.stats()["unique_programs"] for sc in w.contexts.values()))
            self.assertEqual(m["excluded"]["counts"], {"frozen:role:sealed": 1, "incoming:epoch:withdrawn": 1})
            rep = programs.verify(Path(d) / "dict", rehash_sources=True, sources=w.sources)
            self.assertEqual(rep["verdict"], "PASS", rep["failures"])
            for name, sc in w.contexts.items():
                rec = m["contexts"][sc.cid]
                self.assertEqual(rec["split_role"], sc.spec.split_role)
                statuses = Counter(e["outcome"] for e in sc.expect.values())
                self.assertEqual(rec["by_origin"].get("transport_only", 0),
                                 sum(e["origin"] == "transport_only" for e in sc.expect.values()), name)
                self.assertEqual(rec["native_rows"], sc.spec.n_native + sc.spec.n_long, name)
                self.assertGreater(statuses["completed"], 0)
            vocab, max_seq, closure, pins = cs.load_tokenizer(w.tok_dir)
            self.assertEqual(closure["program_set_sha256"], w.train_program_set_sha256)
            self.assertEqual((vocab.sha256, max_seq), (w.vocab.sha256, w.max_seq))
            entries = {e["context_sha256"]: e for e in w.epoch_doc["contexts"]}
            for sc in w.contexts.values():
                e = entries[sc.cid]
                self.assertEqual(jtt1.load_raw(e["input_ref"], w.jtt1_cache), sc.jtt1_raw)
                prep = cs.load_prepared(e["prepared_ref"]["path"], e["prepared_ref"], sc.cid)
                ti = cs.train_input(sc.jtt1_raw, prep, programs._context_ref(e, w.pmap_doc[sc.cid]))
                self.assertTrue(all(ti.prepared_checks.values()))
                long_lengths = [len(dsl_tokenizer.tokenize(sc.dsl[p], vocab)) for p, x in sc.expect.items() if x.get("long")]
                self.assertTrue(all(n > max_seq for n in long_lengths))
                if sc.spec.split_role == "train_fit":
                    self.assertFalse(long_lengths)
                    self.assertTrue(all(len(dsl_tokenizer.tokenize(dsl, vocab)) <= max_seq for dsl in sc.dsl.values()))
            sentinels = w.sentinel_values()
            self.assertTrue(all(sentinels[k] for k in ("floats", "f32", "ints", "strings")))
            hw_doc = json.loads(w.frozen_qualified.read_bytes())
            f0 = w.context("F0")
            self.assertEqual(cs.project_best0_exposed(hw_doc, "frozen", f0.view_id), (f0.best0, True))
            ref_doc = json.loads(w.paths["F0"]["summary"].read_bytes())
            self.assertEqual(cs.project_admitted_exposed(ref_doc, "frozen"), f0.admitted)
            natives = [json.loads(line) for line in f0.native_lines]
            self.assertEqual(cs.recompute_high_water(natives, f0.best0, f0.admitted)[1], "verified")


class Exposed(unittest.TestCase):
    def test_projections_counts_and_high_water(self):
        sc = synth()
        natives = [json.loads(line) for line in sc.native_lines]
        outcomes = [json.loads(line) for line in sc.outcome_lines]
        done = next(r for r in natives if r["status"] == "completed")
        counts = Counter()
        n = cs.project_native_exposed(done, counts)
        self.assertEqual(set(n), {"fitness", "test_rmse", "test_rmse_ratio", "full_prediction_sha256",
                                  "semantic_signature_f32", "sampled_absolute_error_f32", "population_selected",
                                  "semantic_duplicate_of_active", "result_key"})
        self.assertEqual((len(n["semantic_signature_f32"]), len(n["sampled_absolute_error_f32"])), (64, 192))
        self.assertEqual(n["semantic_signature_f32"], np.frombuffer(base64.b64decode(
            done["semantic_signature_f32_b64"]), dtype="<f4").tolist())
        self.assertEqual(n["test_rmse"], done["test_rmse"])
        self.assertEqual(counts, Counter())
        bad = {**done, "semantic_signature_f32_b64": "!!not base64!!",
               "sampled_absolute_error_f32_b64": base64.b64encode(b"12345").decode(), "fitness": "x",
               "population_selected": 1}
        n2 = cs.project_native_exposed(bad, counts)
        self.assertEqual((n2["semantic_signature_f32"], n2["sampled_absolute_error_f32"], n2["fitness"],
                          n2["population_selected"]), (None, None, None, None))
        self.assertEqual((counts[cs.X_BAD_B64], counts[cs.X_BAD_TYPE]), (2, 2))
        detailed = next(r for r in outcomes if r["phase"] == "evolution"
                        and r.get("evaluation_detail", {}).get("official_test"))
        counts = Counter()
        o = cs.project_outcome_exposed(detailed, counts)
        self.assertEqual(counts, Counter())
        self.assertEqual(o["official_test_metrics_rmse"], detailed["evaluation_detail"]["official_test"]["metrics"]["rmse"])
        self.assertEqual(o["result_ref_sha256"], detailed["result_ref"]["sha256"])
        extra = fx.deep(detailed)
        extra["evaluation_detail"]["official_test"]["surprise"] = 1
        extra["evaluation_detail"]["official_test"]["metrics"]["median_ae"] = 2.0
        cs.project_outcome_exposed(extra, counts)
        self.assertEqual(counts[cs.X_UNMAPPED], 2)
        row = cs.project_record_exposed(context_sha256=sc.cid, program_sha256=detailed["program_sha256"],
                                        split_role="heldout", origin="evolution",
                                        native_full=next(r for r in natives if r["program"]["program_sha256"]
                                                         == detailed["program_sha256"]),
                                        outcome_full=detailed, history_ref={"bucket": "b", "key": "k",
                                                                            "version_id": "v", "sha256": "f" * 64},
                                        native_packed_line=10, outcome_packed_line=4, native_line_sha256="1" * 64,
                                        outcome_line_sha256="2" * 64, high_water=True)
        self.assertEqual(list(row), cs.EXPOSED_RECORD_SCHEMA.names)
        self.assertEqual(row["test_rmse"], next(r for r in natives if r["program"]["program_sha256"]
                                                == detailed["program_sha256"])["test_rmse"])
        self.assertEqual(row["gp_rmse_over_tabpfn35"], row["test_rmse"] / detailed["tabpfn35_rmse"])
        self.assertEqual(row["exposed_sources"], "native+outcome")
        self.assertEqual(cs.exposed_table([row]).num_rows, 1)
        only_outcome = cs.project_record_exposed(context_sha256=sc.cid, program_sha256=detailed["program_sha256"],
                                                 split_role="dev", origin="bootstrap",
                                                 outcome_full={**detailed, "tabpfn35_rmse": 0.0})
        self.assertEqual(only_outcome["test_rmse"], detailed["test_rmse"])
        self.assertIsNone(only_outcome["gp_rmse_over_tabpfn35"])
        self.assertEqual(cs.project_record_exposed(context_sha256=sc.cid, program_sha256="0" * 64, split_role="dev",
                                                   origin="transport_only")["exposed_sources"], "none")
        flags, status = cs.recompute_high_water(natives, sc.best0, sc.admitted)
        self.assertEqual((status, sum(flags)), ("verified", sc.admitted))
        self.assertEqual(flags, fx.high_water(natives, sc.best0))
        self.assertEqual(cs.recompute_high_water(natives, sc.best0, sc.admitted + 1), (None, "mismatch"))
        self.assertEqual(cs.recompute_high_water(natives, sc.best0, None), (None, "no_expected"))
        broken = [fx.deep(r) for r in natives]
        for r in broken:
            if r["status"] == "completed" and r.get("cv_gate_passed"):
                del r["test_rmse"]
        self.assertEqual(cs.recompute_high_water(broken, sc.best0, sc.admitted), (None, "unverifiable"))
        frozen_doc = {sc.view_id: {"bootstrap_summary": {"best_signal_test": {"test_rmse": 0.5}}}}
        self.assertEqual(cs.project_best0_exposed(frozen_doc, "frozen", sc.view_id), (0.5, True))
        self.assertEqual(cs.project_best0_exposed(frozen_doc, "frozen", "other"), (None, False))
        inc_doc = {"qualified": [{"view_id": "v1", "bootstrap_summary": {"best_signal_test": {"test_rmse": 2}}}]}
        self.assertEqual(cs.project_best0_exposed(inc_doc, "incoming", "v1"), (2.0, True))
        self.assertEqual(cs.project_best0_exposed(inc_doc, "incoming", "v2"), (None, False))
        self.assertEqual(cs.project_admitted_exposed({"signal_admitted_improvements": 4}, "frozen"), 4)
        self.assertEqual(cs.project_admitted_exposed({"stats": {"signal_admitted_improvements": 3}}, "incoming"), 3)
        self.assertIsNone(cs.project_admitted_exposed({"stats": {}}, "incoming"))
        nl = cs.project_native_lineage(done)
        evo = next(r for r in outcomes if r["program_sha256"] == done["program"]["program_sha256"]
                   and r["phase"] == "evolution")
        ol = cs.project_outcome_lineage(evo)
        lin = cs.lineage_row(context_sha256=sc.cid, program_sha256=done["program"]["program_sha256"],
                             split_role="train_fit", origin="evolution", native=nl, outcome=ol, native_line=1,
                             outcome_line=3, evolution_outcome_rows=1, transport_attempts=2,
                             pools=["xbabe3-extra", "xbabe3-extra"],
                             backend=cs.BackendInfo("local", "outcome_node", "xbabe3"))
        self.assertEqual(list(lin), cs.LINEAGE_RECORD_SCHEMA.names)
        self.assertEqual((lin["operator"], lin["island_policy"], lin["generation"], lin["backend_field_agrees"]),
                         (done["program"]["operator"], done["policy"], evo["generation"], True))
        self.assertEqual(cs.lineage_table([lin]).num_rows, 1)
        for key in list(lin) + list(nl) + list(ol):
            self.assertFalse(cs.name_is_exposed(key), key)

    def test_exposed_strings_confined(self):
        """Exposed key strings appear only in the project_*_exposed projectors, the high-water recompute and the
        denylist and sealed-schema assignments (and nowhere in a docstring of any other scope)."""
        tree = ast.parse(Path(cs.__file__).read_text())
        allowed_assign = {"EXPOSED_RECORD_SCHEMA", "EXPOSED_NAMES", "FORBIDDEN_SUBSTRINGS"}
        allowed_def = re.compile(r"project_\w+_exposed|recompute_high_water")
        offenders = []
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.ClassDef)) and allowed_def.fullmatch(top.name):
                continue
            if isinstance(top, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed_assign
                                                   for t in top.targets):
                continue
            for node in ast.walk(top):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    hits = [t for t in EXPOSED_TOKENS if t in node.value]
                    if hits:
                        offenders.append((getattr(top, "name", type(top).__name__), getattr(node, "lineno", 0), hits))
        self.assertEqual(offenders, [])
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for needed in ("project_native_exposed", "project_outcome_exposed", "project_native_lineage",
                       "project_outcome_lineage", "recompute_high_water"):
            self.assertIn(needed, names)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {(node.module or "").split(".")[0]} | {a.name for a in node.names}
        self.assertFalse(imported & {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry", "boto3",
                                     "botocore", "s3io"}, imported)


def bump(x: float, ulps: int) -> float:
    """x moved by |ulps| units in the last place, toward +inf for positive ulps."""
    for _ in range(abs(ulps)):
        x = math.nextafter(x, math.inf if ulps > 0 else -math.inf)
    return x


class CvSd(unittest.TestCase):
    """Owner red-team 2026-10-05T11:10Z (CL-1): one canonical cv_log_ratio_sd implementation for every value P5 stores
    or recomputes; the native/outcome gate exact on every field except cv_log_ratio_sd, which tolerates at most 1 ULP
    with every tolerated hit counted and listed; 2 ULP, or 1 ULP on any other field, still STOPs."""

    def test_canonical_formula(self):
        r, b = [0.11, 0.19], [0.2, 0.25]
        l0, l1 = math.log(r[0] / b[0]), math.log(r[1] / b[1])
        m = (l0 + l1) / 2
        want = math.sqrt(((l0 - m) * (l0 - m) + (l1 - m) * (l1 - m)) / 2)
        self.assertEqual(cs.canonical_cv_log_ratio_sd(r, b), want)
        self.assertEqual(cs.canonical_cv_log_ratio_sd(tuple(r), tuple(b)), want)
        self.assertEqual(cs.canonical_cv_log_ratio_sd([0.3, 0.3], [0.3, 0.3]), 0.0)
        self.assertAlmostEqual(cs.canonical_cv_log_ratio_sd(r, b), abs(l0 - l1) / 2, delta=1e-15)
        for bad_r, bad_b in (([0.0, 0.1], b), ([NAN, 0.1], b), ([INF, 0.1], b), ([-0.1, 0.1], b),
                             ([0.1, 0.2], [0.0, 0.2]), ([0.1, None], b), (None, b), ([0.1], [0.2]),
                             ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]), ([True, 0.2], b), (["0.1", 0.2], b),
                             ([1e-320, 0.2], [1e300, 0.2])):
            self.assertIsNone(cs.canonical_cv_log_ratio_sd(bad_r, bad_b), (bad_r, bad_b))
        # the fixture's writers use the native engine's two-pass formula: equal to the canonical value bit for bit
        for a in (0.3, -0.2, 0.05, 1.7, -0.9):
            block = fx.cv_block([0.21, 0.37], a)
            self.assertEqual(cs.canonical_cv_log_ratio_sd(block["rmse_j"], [0.21, 0.37]), block["cv_log_ratio_sd"])
        policy = cs.CV_SD_POLICY
        self.assertEqual((policy["field"], policy["option"], policy["max_ulps"], policy["exact_fields"]),
                         ("cv_log_ratio_sd", "scoped_tolerance", 1, ["cv_mean_log_ratio", "cv_gate_passed"]))
        self.assertTrue(policy["why"] and policy["request"] and policy["canonical"])

    def test_ulp_distance(self):
        x = 0.0009373677655208116
        self.assertEqual(cs.ulp_distance(x, x), 0)
        self.assertEqual(cs.ulp_distance(x, bump(x, 1)), 1)
        self.assertEqual(cs.ulp_distance(bump(x, 1), x), 1)
        self.assertEqual(cs.ulp_distance(x, bump(x, -2)), 2)
        self.assertEqual(cs.ulp_distance(0.0, -0.0), 0)
        self.assertEqual(cs.ulp_distance(0.0, 5e-324), 1)
        self.assertEqual(cs.ulp_distance(-0.0, 5e-324), 1)
        self.assertEqual(cs.ulp_distance(0.0009373677655208116, 0.0009373677655208117), 1)     # the sample's hit
        for a, b in ((x, -x), (x, NAN), (NAN, NAN), (x, INF), (INF, INF), (x, None), (None, None), (1, 1.0)):
            self.assertIsNone(cs.ulp_distance(a, b), (a, b))
        self.assertEqual(cs.f64_hex(1.0), "3ff0000000000000")
        self.assertEqual(cs.f64_hex(-0.0), "8000000000000000")

    def test_native_outcome_gate(self):
        sd, mean = 0.0009373677655208116, 0.001069911130110965
        native = nat("completed", cvm=mean, cvsd=sd, gate=True, line=93)

        def evo(**kw):
            values = {"cvm": mean, "cvsd": sd, "gate": True, "line": 189, **kw}
            return out("evolution", "completed", "completed_transport", **values)

        self.assertIsNone(cs._check_cv_agreement(native, evo(), "c" * 64))
        hit = cs._check_cv_agreement(native, evo(cvsd=bump(sd, 1)), "c" * 64)
        self.assertEqual(hit, {"field": "cv_log_ratio_sd", "program_sha256": "a" * 64, "native_line": 93,
                               "outcome_line": 189, "native_bits": cs.f64_hex(sd),
                               "outcome_bits": cs.f64_hex(bump(sd, 1)), "ulps": 1})
        self.assertEqual(cs._check_cv_agreement(native, evo(cvsd=bump(sd, -1)), "c" * 64)["ulps"], 1)
        stops = [("2 ULP on cv_log_ratio_sd", {"cvsd": bump(sd, 2)}),
                 ("-2 ULP on cv_log_ratio_sd", {"cvsd": bump(sd, -2)}),
                 ("1 ULP on cv_mean_log_ratio", {"cvm": bump(mean, 1)}),
                 ("1 ULP on cv_mean_log_ratio and on cv_log_ratio_sd", {"cvm": bump(mean, 1), "cvsd": bump(sd, 1)}),
                 ("cv_gate_passed flipped", {"gate": False}),
                 ("cv_gate_passed null", {"gate": None}),
                 ("cv_log_ratio_sd NaN", {"cvsd": NAN}),
                 ("cv_log_ratio_sd null", {"cvsd": None}),
                 ("cv_log_ratio_sd negated", {"cvsd": -sd})]
        for label, kw in stops:
            with self.subTest(label), self.assertRaises(cs.CorpusStop) as cm:
                cs._check_cv_agreement(native, evo(**kw), "c" * 64)
            self.assertEqual(cm.exception.kind, "native_outcome_disagreement")
        exact_fit = nat("completed", cvm=-INF, cvsd=NAN, gate=True)
        self.assertIsNone(cs._check_cv_agreement(exact_fit, evo(cvm=-INF, cvsd=NAN), "c" * 64))
        zero = nat("completed", cvm=mean, cvsd=0.0, gate=True)
        self.assertEqual(cs._check_cv_agreement(zero, evo(cvsd=5e-324), "c" * 64)["ulps"], 1)

    def test_assembly_counts_lists_and_stores_canonical(self):
        sc = synth()
        originals = [json.loads(line) for line in sc.outcome_lines]
        natives = [json.loads(line) for line in sc.native_lines]
        last = {}
        for i, r in enumerate(originals):
            if r["phase"] == "evolution":
                last[r["program_sha256"]] = i
        det = next(i for i in last.values() if originals[i]["record_status"] == "completed"
                   and (originals[i].get("evaluation_detail") or {}).get("inner_cv"))
        dl = next(i for i in last.values() if originals[i]["record_status"] == "completed"
                  and not originals[i].get("evaluation_detail") and originals[i].get("cv_log_ratio_sd") is not None)
        p_det, p_dl = originals[det]["program_sha256"], originals[dl]["program_sha256"]
        k_det = next(k for k, r in enumerate(natives) if r["program"]["program_sha256"] == p_det)
        sd_det, sd_dl = originals[det]["cv_log_ratio_sd"], originals[dl]["cv_log_ratio_sd"]

        def built_with(out_change=None, nat_change=None):
            rows, nrows = [fx.deep(r) for r in originals], [fx.deep(r) for r in natives]
            if out_change:
                out_change(rows)
            if nat_change:
                nat_change(nrows)
            return build(sc, ([fx.dump(r) for r in nrows], [fx.dump(r) for r in rows], sc.transport_lines))[1]

        plain = build(sc)[1]
        self.assertEqual((plain.counts["cv_sd_ulp_ties"], plain.counts["cv_sd_ulp_tie_items"],
                          plain.counts["cv_sd_canonical_ulp"], plain.counts["cv_sd_writer_nonfinite"]), (0, [], 0, 0))
        self.assertGreater(plain.counts["cv_sd_canonical"], 0)
        self.assertGreater(plain.counts["cv_sd_verbatim"], 0)
        for rec in plain.records:
            canon = cs.canonical_cv_log_ratio_sd(rec["rmse_j"], rec["baseline_rmse_j"])
            if canon is not None:
                self.assertEqual(rec["cv_log_ratio_sd"], canon)

        def tie(rows):
            rows[det]["cv_log_ratio_sd"] = bump(sd_det, 1)
            rows[dl]["cv_log_ratio_sd"] = bump(sd_dl, -1)
        got = built_with(tie)
        c = got.counts
        self.assertEqual(c["cv_sd_ulp_ties"], 2)
        items = {i["program_sha256"]: i for i in c["cv_sd_ulp_tie_items"]}
        self.assertEqual(set(items), {p_det, p_dl})
        self.assertEqual(items[p_det], {"field": "cv_log_ratio_sd", "program_sha256": p_det, "native_line": k_det + 1,
                                        "outcome_line": det + 1, "native_bits": cs.f64_hex(sd_det),
                                        "outcome_bits": cs.f64_hex(bump(sd_det, 1)), "ulps": 1})
        recs = {r["program_sha256"]: r for r in got.records}
        # detailed: the canonical value of the row's own detail (here the native writer's), 1 ULP from the outcome's
        self.assertEqual(recs[p_det]["cv_log_ratio_sd"], sd_det)
        self.assertEqual(c["cv_sd_canonical_ulp"], 1)
        # detail-less: nothing to recompute, so the metrics (outcome) row's own value is stored verbatim
        self.assertIsNone(recs[p_dl]["rmse_j"])
        self.assertEqual(recs[p_dl]["cv_log_ratio_sd"], bump(sd_dl, -1))
        self.assertEqual({r["program_sha256"]: r for r in got.records if r["program_sha256"] not in (p_det, p_dl)},
                         {r["program_sha256"]: r for r in plain.records if r["program_sha256"] not in (p_det, p_dl)})
        # 2 ULP, or 1 ULP on another field: STOP
        for label, change in (("2 ULP", lambda rows: rows[det].update(cv_log_ratio_sd=bump(sd_det, 2))),
                              ("1 ULP on the mean", lambda rows: rows[det].update(
                                  cv_mean_log_ratio=bump(rows[det]["cv_mean_log_ratio"], 1))),
                              ("2 ULP detail-less", lambda rows: rows[dl].update(cv_log_ratio_sd=bump(sd_dl, 2)))):
            with self.subTest(label), self.assertRaises(cs.CorpusStop) as cm:
                built_with(change)
            self.assertEqual(cm.exception.kind, "native_outcome_disagreement")
        # both writers agree but sit 2 ULP from the canonical value of the row's own detail: STOP cv_sd_recompute;
        # 1 ULP from it on both sides is tolerated, counted twice, and the canonical value is stored
        for ulps, want in ((2, "stop"), (1, "ok")):
            def both(rows, u=ulps):
                rows[det]["cv_log_ratio_sd"] = bump(sd_det, u)

            def nat_both(nrows, u=ulps):
                nrows[k_det]["cv_log_ratio_sd"] = bump(sd_det, u)
            if want == "stop":
                with self.assertRaises(cs.CorpusStop) as cm:
                    built_with(both, nat_both)
                self.assertEqual(cm.exception.kind, "cv_sd_recompute")
            else:
                got = built_with(both, nat_both)
                self.assertEqual((got.counts["cv_sd_ulp_ties"], got.counts["cv_sd_canonical_ulp"]), (0, 2))
                self.assertEqual({r["program_sha256"]: r for r in got.records}[p_det]["cv_log_ratio_sd"], sd_det)


class ValueAllowlist(unittest.TestCase):
    """Red-team LA-5: the non-sealed tables' string columns hold allowlisted values only (64-hex shas, pattern-checked
    identifiers, no exposed name anywhere), in the builders and in the Parquet validator's decode path."""

    BLOB = '{"official_test": {"metrics": {"rmse": 0.0977}}, "test_rmse": 0.0977}'

    def test_builders_refuse(self):
        _, built = build(synth())
        rec = dict(built.records[0])
        bad_records = {
            "json blob in family_id": {**rec, "family_id": self.BLOB},
            "json blob in context_schema": {**rec, "context_schema": self.BLOB},
            "json blob in dataset_hash": {**rec, "dataset_hash": self.BLOB},
            "json blob in context_sha256": {**rec, "context_sha256": self.BLOB},
            "non-hex program_sha256": {**rec, "program_sha256": "NOT-HEX"},
            "uppercase skeleton_sha256": {**rec, "skeleton_sha256": "A" * 64},
            "short dataset_hash": {**rec, "dataset_hash": "d" * 15},
            "context_schema of another kind": {**rec, "context_schema": "v8"},
            "an exposed name as a family": {**rec, "family_id": "test_rmse"},
            "a forbidden substring in a family": {**rec, "family_id": "fam-best_of"},
        }
        for label, bad in bad_records.items():
            with self.subTest(label), self.assertRaises(cs.CorpusStop) as cm:
                cs.records_table([rec, bad])
            self.assertEqual(cm.exception.kind, "table")
        self.assertEqual(cs.records_table([rec]).num_rows, 1)
        lin = cs.lineage_row(context_sha256="c" * 64, program_sha256="1" * 64, split_role="dev", origin="evolution",
                             native={"operator": "typed_mutation_v4", "parents": ["2" * 64], "island_policy": "test_gated",
                                     "roots": ["3" * 64]},
                             outcome={"node_raw": f"aws:{fx.AWS_FN}", "transport_status": "completed_transport",
                                      "attempt_id": "0" * 32, "backend_field": "local_lambda"},
                             pools=["aws", "xbabe1:7431"])
        self.assertEqual(cs.lineage_table([lin]).num_rows, 1)          # a policy name such as test_gated is allowed
        for label, change in (("operator with a forbidden substring", {"operator": "best_of"}),
                              ("JSON in operator", {"operator": self.BLOB}), ("non-hex parent", {"parents": ["p"]}),
                              ("node with a space", {"node_raw": "xbabe1 extra"}), ("pool blob", {"pools": [self.BLOB]}),
                              ("transport status blob", {"transport_status": self.BLOB})):
            with self.subTest(label), self.assertRaises(cs.CorpusStop) as cm:
                cs.lineage_table([{**lin, **change}])
            self.assertEqual(cm.exception.kind, "table")
        ctx = {"context_sha256": "c" * 64, "context_schema": "jope.gp-evaluation-context/v6", "dataset_hash": "d" * 16,
               "family_id": "fam-" + "e" * 64, "split_role": "dev", "cohort": "frozen", "fold": 1, "frozen_fold0": False,
               "n_train": 1, "n_features": 1, "inner_folds": 2, "y_train": [1.0], "train_rows_sha256": "a" * 64,
               "s_c": 1.0, "eps_c": 1e-6, "label_status": "OK", "baseline_rmse_j": None, "q_cap": None, "n_records": 0,
               "n_refused": 0, "eligibility_source": fx.INCOMING_TEXT + cs.LEAK_PASS_SUFFIX}
        self.assertEqual(cs.contexts_table([ctx]).num_rows, 1)
        with self.assertRaises(cs.CorpusStop):
            cs.contexts_table([{**ctx, "eligibility_source": self.BLOB}])
        # the sealed exposed table is never value-checked (it is sealed, and its strings are pointers)
        row = {**dict.fromkeys(cs.EXPOSED_RECORD_SCHEMA.names), "context_sha256": "c" * 64, "program_sha256": "2" * 64,
               "split_role": "heldout", "origin": "bootstrap", "result_key": self.BLOB}
        self.assertEqual(cs.exposed_table([row]).num_rows, 1)

    def test_decode_path_and_json_sample(self):
        """The reviewer's probe: rows written past the builder hold JSON blobs and a non-hex sha. check_parquet's decode
        path and its JSON Schema sample both catch them; a clean file passes both."""
        _, built = build(synth())
        good = [dict(r) for r in built.records]
        bad = [dict(r) for r in good]
        for k in ("context_sha256", "context_schema", "dataset_hash", "family_id", "skeleton_sha256"):
            bad[0][k] = self.BLOB
        bad[0]["program_sha256"] = "NOT-HEX"
        with fx.tmpdir() as d:
            good_path, bad_path = Path(d) / "good.parquet", Path(d) / "bad.parquet"
            good_path.write_bytes(cs.parquet_bytes(cs.records_table(good)))
            table = pa.Table.from_pylist(bad, schema=cs.TRAIN_RECORD_SCHEMA)        # bypasses the strict builder
            bad_path.write_bytes(cs.parquet_bytes(table))
            self.assertEqual(cs.check_parquet(good_path, cs.TRAIN_RECORD_SCHEMA, True, json_sample=1), [])
            self.assertEqual(cs.check_parquet(bad_path, cs.TRAIN_RECORD_SCHEMA), [])      # footer level: invisible
            fails = cs.check_parquet(bad_path, cs.TRAIN_RECORD_SCHEMA, True)
            for col in ("context_sha256", "context_schema", "dataset_hash", "family_id", "skeleton_sha256",
                        "program_sha256"):
                self.assertTrue(any(f.startswith(col + ":") for f in fails), (col, fails))
            self.assertTrue(any("exposed name" in f for f in fails), fails)
            sample = cs.json_schema_sample_failures(pq.read_table(bad_path, use_threads=False), cs.TRAIN_RECORD_ID, 1)
            self.assertTrue(sample and all(f.startswith("row 0: JSON Schema") for f in sample), sample)
            self.assertTrue(any("JSON Schema" in f for f in cs.check_parquet(bad_path, cs.TRAIN_RECORD_SCHEMA, True,
                                                                             json_sample=4096)))


class Reuse(unittest.TestCase):
    """Red-team LA-9: an outcome row that imports another context's evaluation must name this context's exact input,
    the pinned evaluator and timeout and the JTT1 shape, and its source must not lie in another partition."""

    def test_parse_and_check(self):
        sc = synth()
        rows = [json.loads(line) for line in sc.outcome_lines]
        i = next(j for j, r in enumerate(rows) if r["phase"] == "bootstrap" and r["record_status"] == "completed"
                 and r.get("program"))
        ti = cs.train_input(sc.jtt1_raw, sc.prepared, sc.ident)
        n_train, p = sc.spec.n_train, sc.spec.p
        ident = {"candidate_timeout_ms": 90000, "encoded_features": p, "evaluator_sha256": cs.EVALUATOR_SHA256,
                 "input_raw_sha256": ti.raw_sha256, "test_rows": sc.spec.n_test, "train_rows": n_train}
        source = "4" * 64
        reuse = {"adopted_utc_ns_text": "1790883219483607910", "compatibility_identity": ident,
                 "new_training_executed": False, "schema": cs.REUSE_SCHEMA, "source_archive_sha256": "7" * 64,
                 "source_context_sha256": source, "source_ref": {"bucket": "b", "key": "k", "version_id": "v"}}

        def parsed(r):
            return cs.parse_outcome(fx.dump({**rows[i], "evaluation_reuse": r}), i + 1, sc.ident)[0]

        def check(r, role_of=None, split_role="train_fit"):
            return cs.check_evaluation_reuse([parsed(r)], cid=sc.cid, raw_sha256=ti.raw_sha256, header=ti.header(),
                                             split_role=split_role, role_of=role_of or {})

        got = parsed(reuse)
        self.assertEqual(got.reuse, cs.ReuseInfo(cs.REUSE_SCHEMA, source, ti.raw_sha256, cs.EVALUATOR_SHA256, 90000,
                                                 n_train, p))
        self.assertEqual(check(reuse), {"rows": 1, "foreign": 1})
        self.assertEqual(check({**reuse, "source_context_sha256": sc.cid}), {"rows": 1, "foreign": 0})
        self.assertEqual(check(reuse, {source: "train_fit"}), {"rows": 1, "foreign": 1})
        self.assertIsNone(cs.parse_outcome(sc.outcome_lines[i], i + 1, sc.ident)[0].reuse)
        # only the identity is read: the outer-test row count and the source pointers never are
        other = {**reuse, "source_ref": {"x": 1}, "source_archive_sha256": "0" * 64,
                 "compatibility_identity": {**ident, "test_rows": 1}}
        self.assertEqual(parsed(other).reuse, got.reuse)
        stops = [("another input", {**reuse, "compatibility_identity": {**ident, "input_raw_sha256": "0" * 64}}),
                 ("another evaluator", {**reuse, "compatibility_identity": {**ident, "evaluator_sha256": "1" * 64}}),
                 ("another timeout", {**reuse, "compatibility_identity": {**ident, "candidate_timeout_ms": 60000}}),
                 ("another n_train", {**reuse, "compatibility_identity": {**ident, "train_rows": n_train + 1}}),
                 ("a float n_train", {**reuse, "compatibility_identity": {**ident, "train_rows": float(n_train)}}),
                 ("another p", {**reuse, "compatibility_identity": {**ident, "encoded_features": p + 1}}),
                 ("no identity", {k: v for k, v in reuse.items() if k != "compatibility_identity"}),
                 ("another schema", {**reuse, "schema": "jope.gp-equivalent-evaluation-reuse/v2"}),
                 ("a non-hex source", {**reuse, "source_context_sha256": "src"})]
        for label, r in stops:
            with self.subTest(label), self.assertRaises(cs.CorpusStop) as cm:
                check(r)
            self.assertEqual((cm.exception.kind, cm.exception.context), ("evaluation_reuse", sc.cid))
        for role in ("heldout", "dev", "diag", "sealed", "prohibited"):
            with self.subTest(source_role=role), self.assertRaises(cs.CorpusStop) as cm:
                check(reuse, {source: role})
            self.assertEqual(cm.exception.kind, "evaluation_reuse")
        with self.assertRaises(cs.CorpusStop) as cm:
            parsed("reused")
        self.assertEqual(cm.exception.kind, "row")


if __name__ == "__main__":
    unittest.main()
