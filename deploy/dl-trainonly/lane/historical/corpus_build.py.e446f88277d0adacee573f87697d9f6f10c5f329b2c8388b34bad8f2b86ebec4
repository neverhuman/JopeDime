"""PLAN §5 P5: the train-only corpus build of epoch 0 (streaming reads from pinned local sources, within a budget).

Sources are read exactly as dltrain.programs (P4a) reads and verifies them; one exact-VersionId JTT1 input per
context is read through dltrain.jtt1.load_raw (create-only cache) and decoded only through jtt1.train_view (by
corpus_schema.train_input). Every source line is json-parsed once by dltrain.corpus_schema; the train path sees only
its allowlist pruning, and the full rows reach only corpus_schema's project_*_exposed / lineage projectors and the
high-water recompute. No exposed key string appears in this module.

Layout under OUT (the P5 epoch directory):
  train/{records,contexts,refused}/part-NNNNN.parquet, train/{manifest.json,label-stats.json,seed-stats.json}
      split_role train_fit only. train/manifest.json is the TRAIN CLOSURE: it binds only the train shards, the label
      and seed statistics, the tokenizer pins, the schema shas and test-independent input pins, and it is the only
      artifact a run plan may use as refs.corpus (train_closure_ref). Nothing under train/ depends on an outer-test
      byte or an exposed value (the poison test asserts the whole train/ tree is byte-identical).
  eval/{dev,diag}/...                                                          same layout plus label-stats.json
  provenance/{train_fit,dev,diag}/part-NNNNN.json + manifest.json   ctx-records: source pins, fingerprints, JTT1 raw
      sha256, reconciliation counts and tolerated hits; never a run-plan input (role provenance_audit)
  heldout/{records,contexts,refused,ctx-records,exposed,lineage}/, heldout/{manifest,label-stats,summary}.json  sealed
      (summary.json: the heldout totals, exclusions, reconciliation, tolerances and context flags)
  exposed/<partition>/ (sealed, with per-shard JSON sidecars) and lineage/<partition>/ (search lineage selected on
      outer-test-derived scores: role lineage_report, no Parquet statistics), each with manifest.json
  runtime-report.json, manifest.json (the build's provenance receipt; never a run-plan input), cursor.json,
  build.lock, guard-journal.jsonl
Sealed files (heldout/**, exposed/**) and lineage files carry no Parquet statistics. Every shard is create-only; the
cursor and the statistics files are atomic. Shards and statistics hold no timestamps and no paths, so builds are
byte-deterministic. Non-sealed documents carry no heldout-derived total, flag or counter (only sha256 references).

Commands (each first calls cpuguard.require_slice_limits() and cpuguard.set_threads(4)):
  run     --epoch E --epoch-sha256 H --partition-map M --seeds S --tokenizer TOK --out OUT --jtt1-cache C
          --budget-bytes N [--max-gets G] [--local-quota-bytes Q] [--program-dict D] [--inventory INV]
          [--incoming-cursor P] [--incoming-transport-index P] [--frozen-qualified-views P | --no-exposed-high-water]
          [--phase all|fetch|build] [--target-shard-bytes B] [--offline] [--no-resume] [--continue-on-stop]
          [--gate-p3 R --gate-p4 R]
          The real run (OUT = RUN/corpus/epoch-000) needs the P3/P4 gate receipts, refuses --continue-on-stop and
          --expected-completed, takes the sealed inventory from the epoch pin by default and needs the 192 P2 seeds.
  verify  --out OUT [--schema-allowlist] [--counts] [--poison-test [--poison-contexts 2] [--poison-work DIR]]
          [--role heldout_report] [--rehash-sources] [--receipt PATH]
          Reads OUT (RUN/corpus/epoch-000 too, once COMPLETE, without gate receipts); --receipt may go under
          RUN/receipts; the poison work defaults to RUN/tmp/p5-poison-*.
  sample  --label L [--offline] ...   bounded real-sample measurement (measurement only; RUN/tmp/p5-sample-<L>/)
  publish ...                gated (verify PASS with poison, G18 PASS, DL-D7 S3 cap); plans only, never writes S3
Exit codes: 0 PASS/COMPLETE, 2 STOP, 3 verify FAIL, 4 CONFLICT, 75 PAUSED (budget, local quota or GuardTimeout).

Parquet reads always pass use_threads=False (pyarrow 24 aborts with exit 134 on threaded reads in this venv).
Never imported: jope, gp_worker, local_gateway, reuse_cache, intake_registry. No S3 write path exists here; the
only S3 path is jtt1.load_raw with a read client.
"""
from __future__ import annotations

import argparse
import array
import base64
import binascii
import contextlib
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
import random
import re
import resource
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import jsonschema
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dltrain import common, cpuguard, jtt1, labels
from dltrain import corpus_schema as S
from dltrain import dsl_tokenizer as T
from dltrain import programs as P

# --------------------------------------------------------------------------------------------- constants

CURSOR_SCHEMA = "jope.dl-corpus-cursor/v2"
MANIFEST_SCHEMA = "jope.dl-corpus/v2"
PARTITION_SCHEMA = "jope.dl-corpus-partition/v2"
LABEL_STATS_SCHEMA = "jope.dl-corpus-label-stats/v1"
SEED_STATS_SCHEMA = "jope.dl-corpus-seed-stats/v1"
RUNTIME_SCHEMA = "jope.dl-corpus-runtime/v1"
VERIFY_SCHEMA = "jope.dl-corpus-verify/v2"
POISON_SCHEMA = "jope.dl-corpus-poison/v2"
CTX_RECORDS_SCHEMA = "jope.dl-corpus-ctx-records/v1"
CTX_RECORD_SCHEMA = "jope.dl-corpus-ctx-record/v2"
SEALED_MANIFEST_SCHEMA = "jope.dl-corpus-sealed-table/v1"
SEALED_SIDECAR_SCHEMA = "jope.dl-corpus-sealed-ctx/v1"
LINEAGE_MANIFEST_SCHEMA = "jope.dl-corpus-lineage/v2"
PROVENANCE_MANIFEST_SCHEMA = "jope.dl-corpus-provenance/v1"
HELDOUT_SUMMARY_SCHEMA = "jope.dl-corpus-heldout-summary/v1"
SAMPLE_REPORT_SCHEMA = "jope.dl-p5-sample-report/v2"
SAMPLE_RECEIPT_SCHEMA = "jope.dl-p5-sample-receipt/v2"
PUBLISH_PLAN_SCHEMA = "jope.dl-corpus-publish-plan/v2"

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
CORPUS_ROOT = RUN / "corpus"
CORPUS_OUT = CORPUS_ROOT / "epoch-000"
RECEIPTS_DIR = RUN / "receipts"
SAMPLE_ROOT = RUN / "tmp"
SAMPLE_NAME_RE = re.compile(r"p5-sample-[A-Za-z0-9._-]+")
POISON_NAME_RE = re.compile(r"p5-poison-[A-Za-z0-9._-]+")
DEFAULT_SEED_POOL = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/recovery-seed-pool.jsonl")
DEFAULT_SEED_POOL_SHA256 = "a469fb8263f9fb40bedc409808e5548ba3f2eac9859e62cdfc065911d5bd74d3"
DEFAULT_EPOCH = RUN / "epochs" / "epoch-000.json"
DEFAULT_EPOCH_SHA256 = "bcfb5230bc0f5f32f6b42edbfa3a6bf76b89e0593b593b9042290e8c077a87ba"
PUBLISH_PREFIXES = {                                     # object kind of each published file (red-team LA-10)
    "shard": "objects/corpus-shard/sha256/",             # train/dev/diag records, contexts, refused
    "sealed": "objects/corpus-sealed/sha256/",           # heldout/** and exposed/** (shards and sidecars)
    "lineage": "objects/corpus-lineage/sha256/",         # lineage/<partition>/
    "provenance": "objects/corpus-provenance/sha256/",   # provenance/<partition>/ ctx-records
}
PUBLISH_MANIFEST_PREFIX = "campaigns/gp-policy-dl-v1-e0/manifests/"
PUBLISH_SEALED_MANIFEST_PREFIX = "campaigns/gp-policy-dl-v1-e0/manifests-sealed/"
EPOCH_CAP_BYTES = 15 * 2**30
P2_SEED_COUNT = 192

EXIT_OK, EXIT_STOP, EXIT_FAIL, EXIT_CONFLICT, EXIT_PAUSED = 0, 2, 3, 4, 75

PARTITIONS = P.PARTITIONS
OPEN_PARTITIONS = ("train_fit", "dev", "diag")           # every partition but the sealed heldout
COHORTS = P.COHORTS
TABLES = ("records", "contexts", "refused", "ctx-records", "exposed", "lineage")
PARQUET_TABLES = ("records", "contexts", "refused", "exposed", "lineage")
PARTITION_TABLES = ("records", "contexts", "refused")    # what a partition manifest (the train closure) binds
TARGET_SHARD_BYTES = 128 * 2**20
ROW_GROUP_ROWS = S.ROW_GROUP_ROWS
ZSTD_LEVEL = S.ZSTD_LEVEL
LINE_TICK = 4096
EPOCH_EXCLUSION_REASONS = ("withdrawn", "intake_ineligible")
ROLE_REASONS = {"sealed": "sealed_set", "prohibited": "prohibited_excluded"}
PACKET_EXCLUSIONS = ("target_leak", "fold0_construction_contradiction")
LISTED_EXCLUSIONS = (*EPOCH_EXCLUSION_REASONS, *ROLE_REASONS.values(), *PACKET_EXCLUSIONS)
LABEL_Q_PCTS = (("p0", 0), ("p1", 1), ("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99),
                ("p100", 100))
RUNTIME_QS = (("p10", "0.1"), ("p50", "0.5"), ("p90", "0.9"), ("p99", "0.99"))
RUNTIME_PARTITIONS = ("train_fit", "dev", "diag")
NONTRAIN_PREFERENCE = ("heldout", "dev", "diag")
ROLES = ("fit", "training_validation", "diagnostic_only", "heldout_report")
LINEAGE_ROLE = "lineage_report"
PROVENANCE_ROLE = "provenance_audit"
JSON_SAMPLE_EVERY = 4096
RECON_KEYS = ("native_without_outcome", "evolution_outcome_duplicates", "backend_field_disagreements",
              "backend_field_disagreements_known", "backend_field_disagreements_unknown", "gate_boundary_ties",
              "gate_unverified", "detailed_completed", "detailless_completed", "bootstrap_rows_without_program",
              "attempt_id_duplicates", "non_program_rows", "evaluation_reuse_rows", "evaluation_reuse_foreign",
              "cv_sd_ulp_ties", "cv_sd_canonical", "cv_sd_canonical_ulp", "cv_sd_verbatim", "cv_sd_writer_nonfinite")


class Paused(Exception):
    """A resumable pause: status PAUSED_BUDGET, PAUSED_QUOTA or PAUSED_GUARD (exit 75)."""

    def __init__(self, status: str, reason: str):
        super().__init__(f"{status}: {reason}")
        self.status, self.reason = status, reason


def _stop(kind: str, message: str, context: str | None = None) -> S.CorpusStop:
    return S.CorpusStop(message, context, kind)


def _kind(err: BaseException) -> str:
    return str(getattr(err, "kind", None) or type(err).__name__)


def _ctx_stop(err: S.CorpusStop, cid: str) -> S.CorpusStop:
    """The same STOP, naming the context when the raiser did not."""
    if getattr(err, "context", None):
        return err
    return S.CorpusStop(getattr(err, "detail", str(err)), cid, err.kind)


def _from_reconcile(err: P.ReconcileError) -> S.CorpusStop:
    return S.CorpusStop(str(err), err.context, err.kind)


def _canon(obj: Any) -> bytes:
    return common.canonical(obj) + b"\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _nearest_rank(sorted_vals: Sequence[int], q: str) -> int | None:
    n = len(sorted_vals)
    if n == 0:
        return None
    return sorted_vals[max(1, math.ceil(Fraction(q) * n)) - 1]


def key_violations(doc: Any, where: str = "") -> list[str]:
    """Every JSON key (recursively) that is an exposed name or contains a forbidden substring."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if S.name_is_exposed(str(k)):
                out.append(f"{where}.{k}")
            out.extend(key_violations(v, f"{where}.{k}"))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            out.extend(key_violations(v, f"{where}[{i}]"))
    return out


# ------------------------------------------------------------------------------------------------- layout

def part_dir(out: Path, partition: str) -> Path:
    return Path(out) / S.PART_DIR[partition]


def table_dir(out: Path, partition: str, table: str) -> Path:
    """exposed/lineage/ctx-records live outside the partition directories (heldout keeps all of its own, sealed):
    exposed/<p>, lineage/<p> and provenance/<p> (ctx-records), so nothing under train/ or eval/ depends on a test byte."""
    out = Path(out)
    if table in ("exposed", "lineage", "ctx-records"):
        if partition == "heldout":
            return out / "heldout" / table
        return out / ("provenance" if table == "ctx-records" else table) / partition
    return part_dir(out, partition) / table


def is_sealed(partition: str, table: str) -> bool:
    return partition == "heldout" or table == "exposed"


def has_statistics(partition: str, table: str) -> bool:
    """Parquet column statistics: never in sealed files, nor in the test-influenced lineage table (red-team LA-8)."""
    return not is_sealed(partition, table) and table != "lineage"


def table_schema(table: str) -> pa.Schema:
    return S.TABLES[S.TABLE_SHORT[table]]


def table_id(table: str) -> str:
    return CTX_RECORDS_SCHEMA if table == "ctx-records" else S.TABLE_SHORT[table]


def _table_builder(table: str) -> Callable[[list], pa.Table]:
    return {"records": S.records_table, "contexts": S.contexts_table, "refused": S.refused_table,
            "exposed": S.exposed_table, "lineage": S.lineage_table}[table]


def manifest_path(out: Path, partition: str, table: str | None = None) -> Path:
    if table in ("exposed", "lineage", "ctx-records"):
        return table_dir(out, partition, table) / "manifest.json"
    return part_dir(out, partition) / "manifest.json"


def heldout_summary_path(out: Path) -> Path:
    return Path(out) / "heldout" / "summary.json"


# ------------------------------------------------------------------------------------- ctx-record schema

def _jobj(props: dict, required: Sequence[str] = ()) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(required)}


_J_INT = {"type": "integer", "minimum": 0}
_J_STR = {"type": "string"}
_J_BOOL = {"type": "boolean"}
_J_NUM = {"type": "number"}
_J_HEX = {"type": "string", "pattern": "^" + S.HEX64_PATTERN + "$"}
_J_OHEX = {"type": ["string", "null"], "pattern": "^" + S.HEX64_PATTERN + "$"}
_J_COUNTS = {"type": "object", "additionalProperties": _J_INT}
_J_PIN = _jobj({"path": _J_STR, "layout": _J_STR, "bytes": _J_INT, "lines": _J_INT, "sha256": _J_HEX, "pin": _J_STR,
                "file_bytes": _J_INT, "entry_sha256": _J_OHEX}, ("layout",))
_J_TIE = _jobj({"field": {"const": S.CV_SD_FIELD}, "program_sha256": _J_HEX, "native_line": _J_INT,
                "outcome_line": _J_INT, "native_bits": {"type": "string", "pattern": "^[0-9a-f]{16}$"},
                "outcome_bits": {"type": "string", "pattern": "^[0-9a-f]{16}$"},
                "ulps": {"type": "integer", "minimum": 1, "maximum": S.CV_SD_MAX_ULPS}},
               ("field", "program_sha256", "native_line", "outcome_line", "native_bits", "outcome_bits", "ulps"))
_CTX_COUNT_INTS = ("native_rows", "native_completed", "bootstrap_set", "bootstrap_events", "bootstrap_overlap_native",
                   "transport_only", "attempt_id_duplicates", "non_program_rows", "unique_programs", "records",
                   "refused", "native_without_outcome", "evolution_outcome_duplicates", "bootstrap_rows_without_program",
                   "detailed_completed", "detailless_completed", "gate_boundary_ties", "gate_unverified",
                   "backend_field_disagreements", "backend_field_disagreements_known",
                   "backend_field_disagreements_unknown", "evaluation_reuse_rows", "evaluation_reuse_foreign",
                   "cv_sd_ulp_ties", "cv_sd_canonical", "cv_sd_canonical_ulp", "cv_sd_verbatim",
                   "cv_sd_writer_nonfinite")
# The ctx-record allowlist (red-team LA-2): every object is closed (additionalProperties false).
CTX_RECORD_JSON_SCHEMA = {
    "$schema": S.JSON_SCHEMA_DIALECT, "$id": CTX_RECORD_SCHEMA, "title": CTX_RECORD_SCHEMA,
    "description": "P5 per-context provenance record: source pins, reconciliation counts, the train-only leak audit "
                   "and label status; never a run-plan input.",
    **_jobj({
        "schema": {"const": CTX_RECORD_SCHEMA}, "context_sha256": _J_HEX, "cohort": {"enum": list(COHORTS)},
        "fold": _J_INT, "view_id": _J_STR, "item_id": {"type": ["string", "null"]},
        "dataset_hash": {"type": "string", "pattern": "^" + S.VALUE_PATTERNS["dataset_hash"] + "$"},
        "family_id": {"type": "string", "pattern": "^" + S.VALUE_PATTERNS["family_id"] + "$"},
        "split_role": {"enum": list(PARTITIONS)},
        "context_schema": {"type": "string", "pattern": "^" + S.VALUE_PATTERNS["context_schema"] + "$"},
        "summary": _jobj({**{k: _J_INT for k in P.SUMMARY_KEYS}, "history_sha256": _J_HEX, "fingerprint": _J_OHEX,
                          "bootstrap_source_sha256": _J_OHEX}, (*P.SUMMARY_KEYS, "history_sha256")),
        "jtt1": _jobj({"n_train": _J_INT, "n_test": _J_INT, "p": _J_INT, "inner_folds": _J_INT, "raw_sha256": _J_HEX,
                       "train_rows_sha256": _J_HEX, "cache_hit": _J_BOOL},
                      ("n_train", "n_test", "p", "inner_folds", "raw_sha256", "train_rows_sha256", "cache_hit")),
        "leak": _jobj({"exact": {"type": "array", "items": _J_INT},
                       "affine": {"type": "array", "items": {"type": "array", "prefixItems": [_J_INT, _J_NUM],
                                                             "items": False, "minItems": 2}},
                       "nonfinite_columns": _J_INT, "target_leak": _J_BOOL},
                      ("exact", "affine", "nonfinite_columns", "target_leak")),
        "prepared": _jobj({k: _J_BOOL for k in ("train_rows", "encoded_features", "inner_folds", "candidate_timeout_ms",
                                                "evaluator_sha256", "inner_cv_schema", "context_sha256",
                                                "dataset_hash", "outer_fold")}),
        "sources": _jobj({k: _J_PIN for k in ("native", "outcome", "transport", "bootstrap_results", "seed_pool",
                                              "summary")}, ("native", "outcome", "transport", "seed_pool", "summary")),
        "dict_crosscheck": _jobj({"source": {"enum": ["program_dict", "context_rows"]}, "rows": _J_INT,
                                  "equal": {"const": True}}, ("source", "rows", "equal")),
        "excluded": {"enum": [None, "target_leak"]},
        "stopped": {"oneOf": [{"type": "null"}, _jobj({"kind": _J_STR, "message": _J_STR}, ("kind", "message"))]},
        "fold0_construction_contradiction": _J_BOOL, "records": _J_INT, "refused": _J_INT,
        "counts": _jobj({**{k: _J_INT for k in _CTX_COUNT_INTS},
                         **{k: _J_COUNTS for k in ("by_outcome", "by_backend", "backend_source", "refused_by_outcome")},
                         "unknown_backend_prefixes": {"type": "array", "items": {
                             "type": "array", "prefixItems": [_J_STR, _J_INT], "items": False, "minItems": 2}},
                         "cv_sd_ulp_tie_items": {"type": "array", "items": _J_TIE}}),
        "label": _jobj({"context_sha256": _J_HEX, "family_id": _J_STR, "dataset_hash": _J_STR,
                        "split_role": {"enum": list(PARTITIONS)}, "status": {"enum": list(S.LABEL_STATUSES)},
                        "s_c": _J_NUM, "eps_c": _J_NUM,
                        "baseline_rmse_j": {"oneOf": [{"type": "null"}, {"type": "array", "items": _J_NUM,
                                                                         "minItems": 2, "maxItems": 2}]},
                        "q_cap": {"type": ["number", "null"]},
                        "counts": _jobj({k: _J_INT for k in labels.COUNT_KEYS}, labels.COUNT_KEYS)},
                       ("context_sha256", "family_id", "dataset_hash", "split_role", "status", "s_c", "eps_c",
                        "baseline_rmse_j", "q_cap", "counts")),
        "checks": {"type": "object", "additionalProperties": False, "patternProperties": {"^R(10|[1-9])$": _J_BOOL}},
    }, ("schema", "context_sha256", "cohort", "fold", "view_id", "item_id", "dataset_hash", "family_id", "split_role",
        "excluded", "records", "refused", "stopped")),
}
_CTX_FULL_KEYS = ("context_schema", "summary", "jtt1", "leak", "prepared", "sources", "dict_crosscheck",
                  "fold0_construction_contradiction", "counts", "checks")
_CTX_VALIDATOR: Any = None


def check_ctx_record(rec: Any) -> list[str]:
    """Failures of one ctx-record against CTX_RECORD_JSON_SCHEMA (JSON paths and rule names only, never values), plus
    the keys a processed (not stopped) record must carry and the label of a labelled one."""
    global _CTX_VALIDATOR
    if _CTX_VALIDATOR is None:
        _CTX_VALIDATOR = jsonschema.Draft202012Validator(CTX_RECORD_JSON_SCHEMA)
    fails = []
    for e in _CTX_VALIDATOR.iter_errors(rec):
        where = "/".join(map(str, e.absolute_path)) or "$"
        extra = ""
        if e.validator == "additionalProperties" and isinstance(e.instance, dict):
            extra = f" {sorted(set(e.instance) - set((e.schema or {}).get('properties') or {}))[:5]}"
        fails.append(f"{where}: {e.validator}{extra}")
    if not fails and isinstance(rec, dict) and not rec.get("stopped"):
        missing = [k for k in _CTX_FULL_KEYS if k not in rec]
        if not rec.get("excluded") and "label" not in rec:
            missing.append("label")
        if missing:
            fails.append(f"$: missing {missing}")
    return fails[:5]


def _rel(out: Path, p: Path) -> str:
    return str(Path(p).relative_to(out))


# -------------------------------------------------------------------------------------------- out guard

def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _tmp_root() -> str | None:
    """realpath(TMPDIR) when it is a usable test root: never /, /tmp or below, and never a directory that contains
    RUN/corpus (such a TMPDIR would otherwise let a write slip past the real run's gates)."""
    tmp = os.environ.get("TMPDIR")
    if not tmp:
        return None
    root = os.path.realpath(tmp)
    if root in ("/", "/tmp") or root.startswith("/tmp/"):
        return None
    if _under(os.path.realpath(CORPUS_ROOT), root):
        return None
    return root


def _named_child(real: str, regex: re.Pattern) -> bool:
    root = os.path.realpath(SAMPLE_ROOT)
    if not _under(real, root) or real == root:
        return False
    first = Path(real).relative_to(root).parts[0]
    return regex.fullmatch(first) is not None


def _is_sample_dir(real: str) -> bool:
    return _named_child(real, SAMPLE_NAME_RE)


def _is_poison_dir(real: str) -> bool:
    """RUN/tmp/p5-poison-*: the default work directory of the poison harness (scratch copies, never a corpus)."""
    return _named_child(real, POISON_NAME_RE)


def _receipt_ok(path: str | os.PathLike | None, need_map_sha: str | None = None) -> bool:
    if not path:
        return False
    try:
        doc = json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return False
    if not isinstance(doc, dict):
        return False
    if doc.get("verdict", doc.get("status")) != "PASS" and doc.get("pass") is not True:
        return False
    if need_map_sha is not None:
        bound = doc.get("partition_map_sha256") or (doc.get("partition_map") or {}).get("sha256")
        if bound != need_map_sha:
            return False
    return True


def _check_out(out: str | os.PathLike, *, gate_p3: str | None = None, gate_p4: str | None = None,
               partition_map: str | os.PathLike | None = None) -> Path:
    """The write guard (run, and the poison harness's scratch builds). The real output RUN/corpus/epoch-000 is
    checked first and only with PASS receipts for --gate-p3 (binding this partition map) and --gate-p4; nothing else
    under RUN/corpus is ever writable. Otherwise only under TMPDIR (tests; a TMPDIR containing RUN/corpus never
    counts), RUN/tmp/p5-sample-* (sample) or RUN/tmp/p5-poison-* (poison work). Anything else is STOP(gate)."""
    real = os.path.realpath(out)
    corpus = os.path.realpath(CORPUS_OUT)
    if real == corpus:
        map_sha = common.sha256_file(partition_map) if partition_map and Path(partition_map).is_file() else None
        if map_sha and _receipt_ok(gate_p3, map_sha) and _receipt_ok(gate_p4):
            return Path(real)
        raise _stop("gate", f"{real}: the real P5 run needs PASS receipts for --gate-p3 (binding this partition map) "
                            "and --gate-p4")
    if _under(real, corpus) or _under(real, os.path.realpath(CORPUS_ROOT)):
        raise _stop("gate", f"{real}: under {CORPUS_ROOT} only {CORPUS_OUT} itself is a P5 output (gated)")
    root = _tmp_root()
    if root is not None and _under(real, root):
        return Path(real)
    if _is_sample_dir(real) or _is_poison_dir(real):
        return Path(real)
    raise _stop("gate", f"{real}: P5 writes only under TMPDIR, {SAMPLE_ROOT}/p5-sample-*, {SAMPLE_ROOT}/p5-poison-* "
                        f"or {CORPUS_OUT} (gated)")


def _check_read(out: str | os.PathLike) -> Path:
    """The read guard of verify, the poison test and publish (red-team CL-2): RUN/corpus/epoch-000 once its build is
    COMPLETE (cursor status and manifest), with no gate receipt (those gate the writes only); otherwise the places
    _check_out accepts."""
    real = os.path.realpath(out)
    if real == os.path.realpath(CORPUS_OUT):
        try:
            cur = json.loads((Path(real) / "cursor.json").read_bytes())
        except (OSError, ValueError):
            cur = {}
        if cur.get("status") != "COMPLETE" or not (Path(real) / "manifest.json").is_file():
            raise _stop("gate", f"{real}: not a COMPLETE build (cursor status {cur.get('status')!r})")
        return Path(real)
    return _check_out(real)


def _check_receipt(path: str | os.PathLike) -> Path:
    """A verify receipt goes under RUN/receipts (create-only) or wherever _check_out accepts a write."""
    real = os.path.realpath(path)
    root = os.path.realpath(RECEIPTS_DIR)
    if _under(real, root) and real != root:
        return Path(real)
    return _check_out(real)


# --------------------------------------------------------------------------------- exposed-side inputs
# Values are read only through corpus_schema's project_*_exposed functions; this module routes them to the sealed
# tables and pins the files (sha256, bytes, read time) in manifest.exposed_inputs.

def read_qualified(path: Path, cohort: str) -> tuple[dict[str, tuple[float | None, bool]], dict]:
    """view_id -> (best0, found) of one exposed-side qualification file (frozen gp/qualified-views.json or an incoming
    item's qualification.json), read once; the document is dropped after the projection."""
    raw = Path(path).read_bytes()
    pin = {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "read_utc_us": common.utc_us()}
    doc = json.loads(raw)
    if cohort == "frozen":
        views = [v for v in doc] if isinstance(doc, dict) else []
    else:
        qual = doc.get("qualified") if isinstance(doc, dict) else None
        views = [q.get("view_id") for q in qual if isinstance(q, dict)] if isinstance(qual, list) else []
    return {v: S.project_best0_exposed(doc, cohort, v) for v in views if isinstance(v, str)}, pin


def read_frozen_admitted(path: Path, want_sha256: str | None) -> tuple[int | None, int]:
    """The frozen history summary's improvement count (exposed-side), from the same history-ref.json bytes the
    allowlisted summary was read from."""
    raw = Path(path).read_bytes()
    if want_sha256 is not None and _sha(raw) != want_sha256:
        raise _stop("source", f"{path} changed between the summary read and the exposed-side read")
    return S.project_admitted_exposed(json.loads(raw), "frozen"), len(raw)


def read_incoming_cursor(path: Path, exposed: bool) -> tuple[dict[str, dict], dict, dict[str, int | None]]:
    """Read the live fold-archive cursor once: allowlisted projections (programs._project_cursor_entry), the file pin
    {path, layout, bytes, sha256, read_utc_us} and, when exposed is on, each entry's exposed-side improvement count."""
    raw = Path(path).read_bytes()
    read_us = common.utc_us()
    try:
        doc = json.loads(raw)
    except ValueError:
        raise _stop("summary", f"{path}: invalid JSON") from None
    if not isinstance(doc, dict):
        raise _stop("summary", f"{path} is not a JSON object")
    projected = {cid: P._project_cursor_entry(e) for cid, e in doc.items() if isinstance(e, dict)}
    side = {cid: S.project_admitted_exposed(e, "incoming") for cid, e in doc.items() if isinstance(e, dict)} if exposed else {}
    pin = {"path": str(path), "layout": "live_snapshot", "bytes": len(raw), "sha256": _sha(raw), "read_utc_us": read_us}
    return projected, pin, side


def entry_sha256(entry: Mapping | None) -> str | None:
    return None if entry is None else _sha(common.canonical(entry))


# ------------------------------------------------------------------------------------- item offset index

class ItemIndex:
    """One incoming item's run/gp/hwm-events.jsonl, scanned once for its pending contexts (programs._complete_lines
    and _line_context semantics): per context, array('q') offsets and lengths of its lines until their byte sum equals
    its outcome_bytes. A line crossing that boundary is a STOP(source); later lines count as tail; a trailing partial
    line is a write in progress. Lines are read back with os.pread; about 16 B of index per kept line."""

    def __init__(self, path: Path, wanted: Mapping[str, int]):
        self.path = Path(path)
        self.need = dict(wanted)
        self.offsets = {c: array.array("q") for c in wanted}
        self.lengths = {c: array.array("q") for c in wanted}
        self.got = {c: 0 for c in wanted}
        self.tail = {c: 0 for c in wanted}
        self.errors: dict[str, str] = {}
        self.scanned_bytes = self.scanned_lines = self.kept_lines = self.partial_tail = 0
        self.done: set[str] = set()
        try:
            self._scan()
        except P.ReconcileError as err:
            for c in wanted:
                self.errors.setdefault(c, f"{err.kind}: {err}")

    def _feed(self, off: int, line: bytes) -> None:
        cid = P._line_context(line)
        if cid is None or cid not in self.need or cid in self.errors:
            return
        need, got = self.need[cid], self.got[cid]
        if got == need:
            self.tail[cid] += 1
            return
        if got + len(line) > need:
            self.errors[cid] = f"the recorded outcome prefix ({need} B) ends inside a line"
            return
        self.offsets[cid].append(off)
        self.lengths[cid].append(len(line))
        self.got[cid] = got + len(line)
        self.kept_lines += 1

    def _scan(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            rest, rest_start = b"", 0
            while True:
                block = f.read(P.READ_BLOCK)
                if not block:
                    break
                self.scanned_bytes += len(block)
                data = rest + block
                start = 0
                while True:
                    nl = data.find(b"\n", start)
                    if nl < 0:
                        break
                    self.scanned_lines += 1
                    self._feed(rest_start + start, data[start:nl + 1])
                    start = nl + 1
                rest, rest_start = data[start:], rest_start + start
            self.partial_tail = len(rest)

    def lines(self, cid: str) -> Iterator[bytes]:
        if cid in self.errors:
            raise _stop("source", self.errors[cid], cid)
        if self.got.get(cid) != self.need.get(cid):
            raise _stop("source", f"incoming outcome lines hold {self.got.get(cid)} B; the summary records "
                                  f"{self.need.get(cid)} B", cid)
        fd = os.open(str(self.path), os.O_RDONLY)
        try:
            for off, n in zip(self.offsets[cid], self.lengths[cid]):
                b = os.pread(fd, n, off)
                if len(b) != n or not b.endswith(b"\n"):
                    raise _stop("source", "hwm-events.jsonl changed under the offset index", cid)
                yield b
        finally:
            os.close(fd)

    def release(self, cid: str) -> bool:
        """Mark one context done; True when every wanted context is done (the caller drops the index)."""
        self.done.add(cid)
        self.offsets[cid] = array.array("q")
        self.lengths[cid] = array.array("q")
        return self.done >= set(self.need)


# --------------------------------------------------------------------------------- P4a dictionary rows

class DictIndex:
    """The P4a program dictionary's R7 rows of one context, {(program_sha256, origin, record_status)}, read through
    programs.SealedIndex (role r7_crosscheck: the sealed side table's R7 columns only, each chunk sha-checked and the
    open journaled). The published dictionary carries no origin column, so this is the only route to it."""

    def __init__(self, dict_dir: Path):
        try:
            self._index = P.SealedIndex(Path(dict_dir), role="r7_crosscheck")
        except (OSError, ValueError, KeyError) as err:
            raise _stop("input", f"{dict_dir}: not a P4a program dictionary with a sealed side table ({err})") from None
        self.dir, self.sha256, self.doc = self._index.dir, self._index.sha256, self._index.doc

    def rows(self, cid: str) -> set:
        try:
            return self._index.rows(cid)
        except P.ReconcileError as err:
            raise _stop("dict_mismatch", f"P4a sealed side table: {err}", cid) from None

    def train_program_hashes(self) -> pa.Array:
        """The distinct train_fit program_sha256 values (hex, byte-sorted) through programs.open_partition (role fit:
        bytes, sha256 and split_role of every chunk checked)."""
        try:
            t = P.open_partition(self.dir, "train_fit", role="fit").to_table(columns=["program_sha256"],
                                                                             use_threads=False)
        except (P.ReconcileError, OSError, ValueError) as err:
            raise _stop("tokenizer_closure", f"P4a train_fit partition: {err}") from None
        vals = pc.unique(t.column(0).combine_chunks())
        vals = pa.array([b.hex() for b in vals.to_pylist()], type=pa.string())
        return vals.take(pc.sort_indices(vals))


def p4a_status(origin: str, outcome: str) -> str:
    """The P4a record_status compatible with a P5 outcome class (R7): P4a missing is missing or abandoned:* for
    bootstrap origin, and transport_only is missing."""
    if origin == "transport_only":
        return "transport_only"
    if outcome in (labels.COMPLETED, labels.NONFINITE):
        return "completed"
    if outcome.startswith(labels.FAILED_PREFIX):
        return "failed"
    if outcome == labels.TIMEOUT_OUTCOME:
        return "timeout"
    if outcome.startswith("abandoned:"):
        return outcome if origin == "evolution" else "missing"
    return "missing"


# ------------------------------------------------------------------------------------------ inputs

@dataclass
class Inputs:
    """Every pinned input of one build; resume refuses any change of `pins`."""
    epoch_path: Path
    epoch_doc: dict
    entries: dict
    plan: Any
    members: dict
    seeds: frozenset
    seeds_pin: dict
    map_path: Path
    vocab: Any
    max_seq: int
    tok_closure: dict
    tok_dir: Path
    tok_pin: dict
    dict_index: DictIndex | None
    inventory: dict | None
    inventory_pin: dict | None
    sources: P.Sources
    frozen_qualified: Path | None
    exposed: bool
    expected_completed: Any
    target_shard_bytes: int
    pins: dict
    tokenizer_closure: bool
    excluded: dict
    item_of: dict
    role_of: dict = field(default_factory=dict)
    real_run: bool = False


def load_epoch(path: Path, expect_sha256: str | None) -> tuple[dict, dict, dict]:
    raw = Path(path).read_bytes()
    sha = _sha(raw)
    if expect_sha256 and sha != expect_sha256:
        raise _stop("input", f"epoch sha256 {sha} != --epoch-sha256 {expect_sha256}")
    doc = json.loads(raw)
    if not isinstance(doc, dict) or not isinstance(doc.get("contexts"), list):
        raise _stop("input", f"{path} has no contexts list")
    entries = {e["context_sha256"]: e for e in doc["contexts"] if isinstance(e, dict) and "context_sha256" in e}
    return doc, entries, {"path": str(path), "sha256": sha, "bytes": len(raw)}


def _load_map(path: Path) -> dict:
    raw = Path(path).read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError:
        raise _stop("input", f"{path}: invalid JSON") from None
    if not isinstance(doc, dict):
        raise _stop("input", "partition map must be a JSON object context_sha256 -> {family_id, split_role}")
    allowed = set(PARTITIONS) | set(ROLE_REASONS)
    for cid, v in doc.items():
        role = v.get("split_role") if isinstance(v, dict) else None
        if role not in allowed:
            raise _stop("unlisted_exclusion", f"partition-map role {role!r} is neither a partition {PARTITIONS} nor a "
                                              f"listed exclusion {tuple(ROLE_REASONS)}", cid if _is_hex64(cid) else None)
    return doc


def closure_checks(entries: Mapping[str, dict], pmap: Mapping[str, dict], members: Mapping[str, list]) -> None:
    """Training closure (PLAN §5 P5): one family per dataset; no family mixes train (train_fit or diag), dev and
    heldout contexts; a family (so also a dataset) with a sealed or prohibited context holds no other group, since
    sealing and rights are family properties (red-team LA-6); no diag context in frozen fold 0. Each violation is
    STOP(closure)."""
    fam_of_ds: dict[str, str] = {}
    groups_of_fam: dict[str, set] = defaultdict(set)
    groups_of_ds: dict[str, set] = defaultdict(set)
    for cid in sorted(entries):
        e = entries[cid]
        if e.get("status") != "included" or cid not in pmap:
            continue
        m = pmap[cid]
        ds, fam, role = e.get("dataset_hash"), m["family_id"], m["split_role"]
        if fam_of_ds.setdefault(ds, fam) != fam:
            raise _stop("closure", f"dataset {ds} is mapped to families {fam_of_ds[ds]} and {fam}", cid)
        group = "train" if role in ("train_fit", "diag") else role
        groups_of_fam[fam].add(group)
        groups_of_ds[ds].add(group)
    excluded = set(ROLE_REASONS)
    for what, groups in (("family", groups_of_fam), ("dataset", groups_of_ds)):
        for key in sorted(groups, key=str):
            g = groups[key]
            if len(g & {"train", "dev", "heldout"}) > 1:
                raise _stop("closure", f"{what} {key} mixes {sorted(g)} contexts")
            if g & excluded and len(g) > 1:
                raise _stop("closure", f"{what} {key} holds {sorted(g & excluded)} contexts together with "
                                       f"{sorted(g - excluded)}: a sealed or prohibited group must be its only group")
    for c in members.get("diag", []):
        if c.cohort == "frozen" and c.fold == 0:
            raise _stop("closure", "a frozen fold-0 context is in the split diagnostic", c.context_sha256)


def load_inventory(path: Path, epoch_doc: Mapping) -> tuple[dict, dict]:
    """The sealed inventory's projection {context_sha256: native.completed} (nothing else is read)."""
    raw = Path(path).read_bytes()
    sha = _sha(raw)
    want = (epoch_doc.get("inventory") or {}).get("sha256") if isinstance(epoch_doc.get("inventory"), dict) else None
    if want and sha != want:
        raise _stop("input", f"inventory sha256 {sha} != the epoch's inventory pin {want}")
    rows = json.loads(gzip.decompress(raw))
    out = {}
    for r in rows:
        nat = r.get("native") if isinstance(r, dict) else None
        if isinstance(nat, dict) and isinstance(nat.get("completed"), int):
            out[r["context_sha256"]] = nat["completed"]
    return out, {"path": str(path), "sha256": sha, "bytes": len(raw), "rows": len(out)}


def _expected_pin(expected: Any) -> Any:
    if isinstance(expected, Mapping):
        return {"per_context_sha256": _sha(common.canonical(dict(sorted(expected.items())))), "contexts": len(expected)}
    return int(expected)


def load_inputs(*, epoch, epoch_sha256, partition_map, seeds, tokenizer, program_dict=None, inventory=None,
                incoming_cursor=None, incoming_transport_index=None, frozen_qualified_views=None, exposed=True,
                expected_completed=None, target_shard_bytes=TARGET_SHARD_BYTES, tokenizer_closure=True,
                real_run=False) -> Inputs:
    """Every pinned input of one build. real_run (OUT = RUN/corpus/epoch-000; red-team CL-3/CL-7): the expected
    completed count is the sealed 1,280 (an override is a STOP), the sealed inventory defaults to the epoch's own
    inventory pin and is required, and the seed list holds the 192 P2 seed hashes."""
    epoch_path, map_path = Path(epoch), Path(partition_map)
    doc, entries, epoch_pin = load_epoch(epoch_path, epoch_sha256)
    if real_run:
        if expected_completed is not None:
            raise _stop("gate", "the real P5 run takes the sealed native.completed count (1,280); --expected-completed "
                                "is for tests and samples only")
        if inventory is None:
            inventory = ((doc.get("inventory") or {}) if isinstance(doc.get("inventory"), dict) else {}).get("path")
        if not inventory or not Path(inventory).is_file():
            raise _stop("gate", "the real P5 run needs the sealed inventory (--inventory, default the epoch's pin)")
    pmap = _load_map(map_path)
    for e in doc["contexts"]:
        if isinstance(e, dict) and e.get("status") != "included":
            reason = e.get("exclusion_reason") or e.get("status")
            if reason not in EPOCH_EXCLUSION_REASONS:
                raise _stop("unlisted_exclusion", f"epoch exclusion reason {reason!r} is not listed "
                                                  f"{EPOCH_EXCLUSION_REASONS}", e.get("context_sha256"))
    try:
        plan = P._plan(epoch_path, map_path)
        seed_set, seeds_pin = P._load_seeds(seeds)
    except P.ReconcileError as err:
        raise _from_reconcile(err) from None
    if plan.epoch_pin["sha256"] != epoch_pin["sha256"]:
        raise _stop("input", "the epoch file changed while it was read")
    members = {p: [c for c in plan.contexts if c.split_role == p] for p in PARTITIONS}
    closure_checks(entries, pmap, members)
    if real_run and len(seed_set) != P2_SEED_COUNT:
        raise _stop("input", f"the real P5 run needs the {P2_SEED_COUNT} P2 seed hashes; the seed list holds "
                             f"{len(seed_set)}")
    tok_dir = Path(tokenizer)
    vocab, max_seq, closure, tok_pins = S.load_tokenizer(tok_dir)
    tok_pin = {"dir": str(tok_dir), **tok_pins}
    dict_index = DictIndex(Path(program_dict)) if program_dict else None
    if dict_index is not None:
        din = dict_index.doc.get("inputs") or {}
        if ((din.get("epoch") or {}).get("sha256") != epoch_pin["sha256"]
                or (din.get("partition_map") or {}).get("sha256") != plan.map_pin["sha256"]):
            raise _stop("input", "the P4a program dictionary was built from another epoch or partition map")
        if (din.get("seeds") or {}).get("sha256") != seeds_pin["sha256"]:
            raise _stop("input", "the P4a program dictionary was built from another seed list (its shared_seed "
                                 "column would disagree with P5's)")
    inv, inv_pin = load_inventory(Path(inventory), doc) if inventory else (None, None)
    sources = P.Sources(Path(incoming_cursor) if incoming_cursor else P.DEFAULT_SOURCES.incoming_cursor,
                        Path(incoming_transport_index) if incoming_transport_index
                        else P.DEFAULT_SOURCES.incoming_transport_index)
    expected = S.EXPECT_COMPLETED_PER_FOLD if expected_completed is None else expected_completed
    if isinstance(expected, Mapping):
        expected = {str(k): int(v) for k, v in expected.items()}
    fq = Path(frozen_qualified_views) if frozen_qualified_views else None
    item_of: dict[str, Path] = {}
    for c in plan.contexts:
        if c.cohort == "incoming":
            try:
                item_of[c.context_sha256] = P._item_gp(c)
            except P.ReconcileError:
                pass
    excluded = Counter()
    for key, n in plan.excluded["counts"].items():
        _cohort, how, reason = key.split(":", 2)
        excluded[ROLE_REASONS.get(reason, reason) if how == "role" else reason] += n
    settings = {"target_shard_bytes": int(target_shard_bytes), "max_shard_bytes": 2 * int(target_shard_bytes),
                "row_group_rows": ROW_GROUP_ROWS, "zstd_level": ZSTD_LEVEL,
                "expected_completed": _expected_pin(expected), "exposed_flags_recompute": bool(exposed),
                "tokenizer_closure_check": bool(tokenizer_closure),
                "pyarrow": pa.__version__}
    pins = {"epoch_sha256": epoch_pin["sha256"], "partition_map_sha256": plan.map_pin["sha256"],
            "seeds_sha256": seeds_pin["sha256"],
            "tokenizer": {"vocab_sha256": tok_pins["vocab_sha256"], "max_seq": max_seq,
                          "lengths_sha256": tok_pins["lengths_json_sha256"]},
            "program_dict_manifest_sha256": dict_index.sha256 if dict_index else None,
            "inventory_sha256": inv_pin["sha256"] if inv_pin else None,
            "sources": {"incoming_cursor": str(sources.incoming_cursor),
                        "incoming_transport_index": str(sources.incoming_transport_index),
                        "frozen_qualified_views": str(fq) if fq else None},
            "settings": settings,
            "schemas": {t: S.schema_sha256(table_schema(t)) for t in PARQUET_TABLES},
            "members_sha256": {p: _sha("".join(c.context_sha256 + "\n" for c in members[p]).encode())
                               for p in PARTITIONS},
            # red-team CL-6: a code change between invocations refuses the resume (a new OUT is needed)
            "code": code_shas()}
    return Inputs(epoch_path=epoch_path, epoch_doc=doc, entries=entries, plan=plan, members=members, seeds=seed_set,
                  seeds_pin=seeds_pin, map_path=map_path, vocab=vocab, max_seq=max_seq, tok_closure=closure,
                  tok_dir=tok_dir, tok_pin=tok_pin, dict_index=dict_index, inventory=inv, inventory_pin=inv_pin,
                  sources=sources, frozen_qualified=fq, exposed=bool(exposed), expected_completed=expected,
                  target_shard_bytes=int(target_shard_bytes), pins=pins, tokenizer_closure=bool(tokenizer_closure),
                  excluded=dict(sorted(excluded.items())),
                  item_of=item_of, role_of={cid: v["split_role"] for cid, v in pmap.items() if isinstance(v, dict)},
                  real_run=bool(real_run))


# ------------------------------------------------------------------------------------- per-context projection

@dataclass
class ContextResult:
    cid: str
    rows: dict
    ctx_record: dict
    sealed_ctx: dict
    excluded: str | None = None


@dataclass
class Env:
    """What project_context needs besides the context itself (all pinned by the cursor). line_count is the running
    number of parsed source lines of this invocation; the Guard ticks at every LINE_TICK-th line."""
    inputs: Inputs
    jtt1_cache: Path
    inc_admitted: Mapping
    item_index: Callable
    qualified: Callable
    tick: Callable
    local: Counter
    timers: Counter
    line_count: int = 0


@dataclass
class Scan:
    """The verified scan of one context (§3.3 step 2): every line parsed once, in source order."""
    natives: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    transports: list = field(default_factory=list)
    native_full: dict = field(default_factory=dict)
    native_order: list = field(default_factory=list)
    last_full: dict = field(default_factory=dict)
    line_sha: dict = field(default_factory=dict)
    pins: dict = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)


def _expected(inputs: Inputs, cid: str) -> int | None:
    e = inputs.expected_completed
    return e.get(cid) if isinstance(e, Mapping) else e


def _tick(env: Env) -> None:
    """One parsed source line: Guard.tick() at every LINE_TICK-th line of the invocation (across sources and contexts)."""
    env.line_count += 1
    if env.line_count % LINE_TICK == 0:
        env.tick()


def scan_sources(c, s, env: Env) -> Scan:
    """R1, R3, R4 and the incoming fingerprint: the native prefix (bytes, sha256, lines), the outcome prefix (bytes,
    lines; incoming lines from the item index) and the transport prefix (bytes, lines), parsed line by line."""
    cid = c.context_sha256
    sc = Scan()
    fp = hashlib.sha256() if c.cohort == "incoming" else None
    extra = (fp,) if fp is not None else ()
    t0 = time.perf_counter_ns()
    if s.native_bytes != c.native["bytes"]:
        raise _stop("source", f"summary native_bytes {s.native_bytes} != epoch pin {c.native['bytes']}", cid)
    reader = P._Prefix(Path(c.native["path"]), c.native["bytes"], "native", cid, extra)
    try:
        for line in reader:
            n = len(sc.natives) + 1
            _tick(env)
            row, full = S.parse_native(line, n, c)
            if row.program_sha256 in sc.native_full:
                raise _stop("row", f"native line {n}: duplicate native program {row.program_sha256[:16]}", cid)
            sc.natives.append(row)
            sc.native_full[row.program_sha256] = full
            sc.native_order.append(full)
            sc.line_sha[("native", row.program_sha256)] = _sha(line)
            sc.stats["native_completed"] += row.status == "completed"
    except P.ReconcileError as err:
        raise _from_reconcile(err) from None
    env.local["native"] += reader.nbytes
    if reader.sha.hexdigest() != c.native["sha256"]:
        raise _stop("source", f"native sha256 {reader.sha.hexdigest()} != epoch pin {c.native['sha256']}", cid)
    if reader.lines != s.native_events:
        raise _stop("count", f"native lines {reader.lines} != native_events {s.native_events}", cid)
    sc.pins["native"] = {"path": str(reader.path), "layout": "file_prefix", "bytes": reader.nbytes,
                         "lines": reader.lines, "sha256": reader.sha.hexdigest(), "pin": "epoch_native_source"}
    if c.cohort == "frozen":
        out_reader = P._Prefix(P._ctx_dir(c) / "history-outcomes.jsonl", s.outcome_bytes, "outcome", cid)
        out_lines: Iterable[bytes] = out_reader
        out_sha = None
    else:
        idx = env.item_index(c)
        out_lines, out_reader, out_sha = idx.lines(cid), None, hashlib.sha256()
    out_bytes = 0
    try:
        for line in out_lines:
            n = len(sc.outcomes) + 1
            out_bytes += len(line)
            _tick(env)
            if out_sha is not None:
                out_sha.update(line)
                fp.update(line)
            row, full = S.parse_outcome(line, n, c)
            sc.outcomes.append(row)
            if row.phase == "evolution":
                key = ("evolution", row.program_sha256)
            elif row.dsl is not None:
                key = ("bootstrap", row.program_sha256)
            else:
                continue
            sc.last_full[key] = full
            sc.line_sha[key] = _sha(line)
    except P.ReconcileError as err:
        raise _from_reconcile(err) from None
    env.local["outcome"] += out_bytes
    if out_bytes != s.outcome_bytes or len(sc.outcomes) != s.outcome_events:
        raise _stop("count", f"outcome prefix {out_bytes} B in {len(sc.outcomes)} lines != summary "
                             f"{s.outcome_bytes} B in {s.outcome_events}", cid)
    sc.pins["outcome"] = ({"path": str(out_reader.path), "layout": "file_prefix", "bytes": out_reader.nbytes,
                           "lines": out_reader.lines, "sha256": out_reader.sha.hexdigest(), "pin": "bytes+lines"}
                          if out_reader is not None else
                          {"path": str(P._item_gp(c) / "hwm-events.jsonl"), "layout": "ctx_lines", "bytes": out_bytes,
                           "lines": len(sc.outcomes), "sha256": out_sha.hexdigest(), "pin": "fingerprint"})
    tpath = P._transport_path(c, env.inputs.sources)
    t_reader = P._Prefix(tpath, s.transport_bytes, "transport", cid, extra)
    try:
        for line in t_reader:
            n = len(sc.transports) + 1
            _tick(env)
            sc.transports.append(S.parse_transport(line, n, c)[0])
    except P.ReconcileError as err:
        raise _from_reconcile(err) from None
    env.local["transport"] += t_reader.nbytes
    if t_reader.lines != s.transport_events:
        raise _stop("count", f"transport lines {t_reader.lines} != transport_events {s.transport_events}", cid)
    sc.pins["transport"] = {"path": str(tpath), "layout": "file_prefix", "bytes": t_reader.nbytes,
                            "lines": t_reader.lines, "sha256": t_reader.sha.hexdigest(),
                            "pin": "fingerprint" if fp is not None else "bytes+lines"}
    if fp is not None:
        bpath = P._ctx_dir(c) / "bootstrap-results.jsonl"
        boot = P._snapshot(bpath)
        env.local["bootstrap_results"] += len(boot)
        if _sha(boot) != s.bootstrap_source_sha256:
            raise _stop("source", "bootstrap-results snapshot sha256 != generation_zero_source_sha256", cid)
        fp.update(boot)
        if fp.hexdigest() != s.fingerprint:
            raise _stop("source", "sha256(native ‖ outcome ‖ transport ‖ bootstrap-results) != archiver fingerprint", cid)
        sc.pins["bootstrap_results"] = {"path": str(bpath), "layout": "file_prefix", "bytes": len(boot),
                                        "sha256": _sha(boot), "pin": "generation_zero_source_sha256"}
    env.timers["parse_ns"] += time.perf_counter_ns() - t0
    env.timers["lines"] += len(sc.natives) + len(sc.outcomes) + len(sc.transports)
    return sc


def _dict_crosscheck(c, s, classes: Mapping[str, tuple[str, str]], env: Env) -> dict:
    """R7: {(program, origin, compatible status)} == the P4a dictionary rows of c (or programs.context_rows)."""
    cid = c.context_sha256
    mine = {(k, origin, p4a_status(origin, cls)) for k, (origin, cls) in classes.items()}
    if env.inputs.dict_index is not None:
        theirs, how = env.inputs.dict_index.rows(cid), "program_dict"
    else:
        lines = list(env.item_index(c).lines(cid)) if c.cohort == "incoming" else None
        try:
            rows, _rec = P.context_rows(c, s, seeds=env.inputs.seeds, sources=env.inputs.sources, item_events=lines)
        except P.ReconcileError as err:
            raise _from_reconcile(err) from None
        theirs, how = {(r.program_sha256, r.origin, r.record_status) for r in rows}, "context_rows"
    if mine != theirs:
        raise _stop("dict_mismatch", f"P5 programs differ from P4a ({how}): {len(mine - theirs)} only in P5, "
                                     f"{len(theirs - mine)} only in P4a", cid)
    return {"source": how, "rows": len(mine), "equal": True}


def _metrics_key(a) -> tuple | None:
    if a.origin == "evolution":
        return ("evolution", a.program_sha256) if a.evo is not None else None
    if a.origin == "bootstrap":
        return ("bootstrap", a.program_sha256)
    return None


def project_context(c, entry: Mapping, s, spin: dict, env: Env) -> ContextResult:
    """The pure, deterministic projection of one context (PLAN §5 P5 §3.3 steps 2-10)."""
    cid = c.context_sha256
    inp = env.inputs
    t_start = time.perf_counter_ns()
    sc = scan_sources(c, s, env)
    # step 3: one ProgramAcc per program
    try:
        accs, attempt_index, dups = S.accumulate_programs(sc.natives, sc.outcomes, sc.transports)
    except S.CorpusStop as err:
        raise _ctx_stop(err, cid) from None
    progs = {k: a for k, a in accs.items() if a.origin is not None or a.evo is not None}
    # step 4: reconciliation R2, R5, R7 (R1/R3/R4 and the fingerprint were checked by the scan)
    cls_counts: Counter = Counter()
    classes: dict[str, tuple[str, str]] = {}
    for k in sorted(progs):
        a = progs[k]
        try:
            classes[k] = (a.origin, S.classify_outcome(a.origin, a.native, a.evo, a.boot, cls_counts))
        except S.CorpusStop as err:
            raise S.CorpusStop(f"program {k[:16]}: {err.detail}", cid, err.kind) from None
    if len(progs) != s.unique_programs:
        raise _stop("count", f"|native ∪ bootstrap ∪ transport-only| = {len(progs)} != unique_programs "
                             f"{s.unique_programs}", cid)
    boot = {k for k, a in progs.items() if a.boot is not None}
    try:
        pool, pool_pin = P._seed_pool(c)
    except P.ReconcileError as err:
        raise _from_reconcile(err) from None
    env.local["seed_pool"] += pool_pin.get("bytes", 0)
    if pool != boot:
        raise _stop("seed_pool", f"bootstrap set ({len(boot)}) != seed-pool set ({len(pool)})", cid)
    completed = sc.stats["native_completed"]
    want = _expected(inp, cid)
    inv = inp.inventory
    if want is None or completed != want or (inv is not None and inv.get(cid) != completed):
        raise _stop("native_completed", f"native completed rows {completed} != expected {want}"
                                        + ("" if inv is None else f" / inventory {inv.get(cid)}"), cid)
    dict_check = _dict_crosscheck(c, s, classes, env)
    # step 5: JTT1, s_c, eps_c and the train-only leak audit (R10)
    t0 = time.perf_counter_ns()
    ref = entry.get("prepared_ref") or {}
    try:
        prep = S.load_prepared(ref.get("path") or "", ref, cid)
    except OSError:
        raise _stop("source", "prepared.json is unreadable", cid) from None
    env.local["prepared"] += int(ref.get("bytes") or 0)
    iref = entry.get("input_ref") or {}
    cache_hit = bool(iref.get("raw_sha256")) and jtt1.cache_path(env.jtt1_cache, iref).exists()
    try:
        raw = jtt1.load_raw(iref, env.jtt1_cache)
    except jtt1.JTTError as err:
        raise _stop("jtt1_missing" if "not cached" in str(err) else "source", f"JTT1 input: {err}", cid) from None
    env.local["jtt1_cache"] += int(iref.get("bytes") or 0)
    ti = S.train_input(raw, prep, c)
    leak = S.leak_audit_train(ti.X, ti.y)
    reuse = S.check_evaluation_reuse(sc.outcomes, cid=cid, raw_sha256=ti.raw_sha256, header=ti.header(),
                                     split_role=c.split_role, role_of=inp.role_of)
    env.timers["jtt1_ns"] += time.perf_counter_ns() - t0
    summary = {**{k: getattr(s, k) for k in P.SUMMARY_KEYS}, "history_sha256": s.history_sha256,
               "fingerprint": s.fingerprint, "bootstrap_source_sha256": s.bootstrap_source_sha256}
    record = {"schema": CTX_RECORD_SCHEMA, "context_sha256": cid, "cohort": c.cohort, "fold": c.fold,
              "view_id": c.view_id, "item_id": c.item_id, "dataset_hash": c.dataset_hash, "family_id": c.family_id,
              "split_role": c.split_role, "context_schema": entry.get("context_schema"), "summary": summary,
              "jtt1": {**ti.to_json(), "cache_hit": cache_hit}, "leak": leak.to_json(),
              "prepared": dict(ti.prepared_checks or {}),
              "sources": {**sc.pins, "seed_pool": pool_pin, "summary": spin}, "dict_crosscheck": dict_check,
              "excluded": None, "stopped": None, "fold0_construction_contradiction": False}
    base_counts = {"native_rows": len(sc.natives), "native_completed": completed, "bootstrap_set": len(boot),
                   "bootstrap_events": sum(a.counts["bootstrap_events"] for a in progs.values()),
                   "bootstrap_overlap_native": sum(1 for k in boot if progs[k].native is not None),
                   "transport_only": sum(1 for a in progs.values() if a.origin == "transport_only"),
                   "attempt_id_duplicates": dups, "non_program_rows": len(accs) - len(progs),
                   "unique_programs": s.unique_programs, "evaluation_reuse_rows": reuse["rows"],
                   "evaluation_reuse_foreign": reuse["foreign"]}
    if leak.target_leak:
        record.update(excluded="target_leak", records=0, refused=0, counts=base_counts,
                      fold0_construction_contradiction=c.cohort == "frozen" and c.fold == 0,
                      checks={f"R{i}": True for i in (1, 2, 3, 4, 5, 7, 10)})
        rows = {t: [] for t in TABLES}
        rows["ctx-records"] = [record]
        return ContextResult(cid, rows, record, {"context_sha256": cid, "excluded": "target_leak"}, "target_leak")
    # step 6: records, refused rows and the labels join (R6, R8, R9 inside build_context_records)
    t0 = time.perf_counter_ns()
    meta = {"context_sha256": cid, "context_schema": entry.get("context_schema"), "dataset_hash": c.dataset_hash,
            "family_id": c.family_id, "split_role": c.split_role, "y_train": ti.y.tolist(), "cohort": c.cohort,
            "fold": c.fold, "eligibility_source": entry.get("eligibility_source")}
    cr = S.build_context_records(meta, progs, inp.vocab, inp.max_seq, inp.seeds, attempt_index=attempt_index)
    if len(cr.records) + len(cr.refused) != s.unique_programs:
        raise _stop("count", f"records {len(cr.records)} + refused {len(cr.refused)} != unique_programs "
                             f"{s.unique_programs}", cid)
    ctx_row = S.build_context_row(meta, ti, cr.context_label, len(cr.records), len(cr.refused))
    env.timers["records_ns"] += time.perf_counter_ns() - t0
    # step 7: sealed exposed rows; the high-water flags are recomputed and verified against the summary count
    t0 = time.perf_counter_ns()
    xcounts: Counter = Counter()
    flags_by_sha, hw, best0_found = None, "off", None
    if inp.exposed:
        q = env.qualified(c)
        best0, best0_found = (q or {}).get(c.view_id, (None, False))
        if c.cohort == "incoming":
            admitted = env.inc_admitted.get(cid)
        else:
            admitted, nbytes = read_frozen_admitted(P._ctx_dir(c) / "history-ref.json", spin.get("sha256"))
            env.local["exposed_side"] += nbytes
        flags, hw = S.recompute_high_water(sc.native_order, best0, admitted)
        if flags is not None:
            flags_by_sha = {row.program_sha256: f for row, f in zip(sc.natives, flags)}
    exposed_rows, lineage_rows = [], []
    href = entry.get("history_ref") or {}
    for k in sorted(progs):
        a = progs[k]
        m, mkey = a.metrics, _metrics_key(a)
        mfull = sc.last_full.get(mkey) if mkey is not None else None
        exposed_rows.append(S.project_record_exposed(
            context_sha256=cid, program_sha256=k, split_role=c.split_role, origin=a.origin,
            native_full=sc.native_full.get(k), outcome_full=mfull, history_ref=href,
            native_packed_line=S.packed_line("native", a.native.line_no - 1, s) if a.native is not None else None,
            outcome_packed_line=S.packed_line("outcome", m.line_no - 1, s) if m is not None else None,
            native_line_sha256=sc.line_sha.get(("native", k)),
            outcome_line_sha256=sc.line_sha.get(mkey) if mkey is not None else None,
            high_water=None if flags_by_sha is None else flags_by_sha.get(k), counts=xcounts))
        # step 8: lineage rows
        lineage_rows.append(S.lineage_row(
            context_sha256=cid, program_sha256=k, split_role=c.split_role, origin=a.origin,
            native=S.project_native_lineage(sc.native_full[k]) if a.native is not None else None,
            outcome=S.project_outcome_lineage(mfull) if mfull is not None else None,
            native_line=a.native.line_no if a.native is not None else None,
            outcome_line=m.line_no if m is not None else None, bootstrap_events=a.counts["bootstrap_events"],
            evolution_outcome_rows=a.counts["evolution_outcome_rows"],
            transport_attempts=a.counts["transport_attempts"],
            pools=[t.pool for t in a.attempts if t.pool is not None], backend=cr.backends[k]))
    env.timers["exposed_lineage_ns"] += time.perf_counter_ns() - t0
    # steps 9-10: the context row and the ctx-record
    counts = dict(base_counts)
    counts.update({k: v for k, v in cr.counts.items() if k != "unknown_backend_prefixes"})
    counts["unknown_backend_prefixes"] = [[k, v] for k, v in sorted(cr.counts.get("unknown_backend_prefixes", {}).items())]
    # every row whose own backend string names another class than the derived one; split by whether the derived class
    # is known (a contradiction to look at) or unknown (no node, attempt or unanimous pool: nothing to compare)
    disagree = [r for r in lineage_rows if r["backend_field_agrees"] is False]
    counts["backend_field_disagreements"] = len(disagree)
    counts["backend_field_disagreements_unknown"] = sum(1 for r in disagree
                                                        if cr.backends[r["program_sha256"]].backend == "unknown")
    counts["backend_field_disagreements_known"] = len(disagree) - counts["backend_field_disagreements_unknown"]
    record.update(records=len(cr.records), refused=len(cr.refused), counts=counts,
                  label=cr.context_label.to_json(), checks={f"R{i}": True for i in range(1, 11)})
    sealed = {"context_sha256": cid, "hw": hw, "best0_found": best0_found, "rows": len(exposed_rows),
              **{k: int(v) for k, v in sorted(xcounts.items())}}
    env.timers["context_ns"] += time.perf_counter_ns() - t_start
    env.timers["contexts"] += 1
    env.timers["programs"] += len(progs)
    return ContextResult(cid, {"records": cr.records, "contexts": [ctx_row], "refused": cr.refused,
                               "ctx-records": [record], "exposed": exposed_rows, "lineage": lineage_rows},
                         record, sealed)


# ------------------------------------------------------------------------------------------- shard writer

class Writer:
    """One writer per (partition, table): rows sorted by (context, program), shards cut only at context boundaries.
    Before a context: flush when buffered > 0 and buffered + new > MAX; after it: flush when buffered >= TARGET
    (Arrow Table.nbytes; JSON bytes for ctx-records). A single context above MAX is a STOP. Shards are create-only;
    equal existing bytes are adopted (an orphan written before a crash), different bytes are a CONFLICT."""

    def __init__(self, build: "Build", partition: str, table: str):
        self.b, self.p, self.t = build, partition, table
        self.state = build.cursor["writers"][partition][table]
        self.dir = table_dir(build.out, partition, table)
        self.sealed = is_sealed(partition, table)
        self.no_stats = not has_statistics(partition, table)
        self.target = build.inputs.target_shard_bytes
        self.max = 2 * self.target
        self.buf: list = []
        self.buf_nbytes = 0

    def _encode(self, rows: list) -> tuple[Any, int, int]:
        if self.t == "ctx-records":
            return list(rows), sum(len(common.canonical(r)) for r in rows), len(rows)
        tbl = _table_builder(self.t)(rows)
        return tbl, int(tbl.nbytes), tbl.num_rows

    def add(self, index: int, cid: str, rows: list, side: Any = None) -> None:
        if index < self.state["next_index"]:
            return
        obj, nb, n = self._encode(rows)
        if nb > self.max:
            raise _stop("shard_size", f"{self.t}: one context holds {nb} B > the {self.max} B shard maximum", cid)
        if self.buf_nbytes > 0 and self.buf_nbytes + nb > self.max:
            self.flush()
        self.buf.append((index, cid, obj, nb, n, side))
        self.buf_nbytes += nb
        if self.buf_nbytes >= self.target:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        last_index = self.buf[-1][0]
        n_rows = sum(x[4] for x in self.buf)
        if n_rows == 0:                       # only empty contexts (e.g. no refused program): no file, just advance
            self.state["next_index"] = last_index + 1
            self.buf, self.buf_nbytes = [], 0
            self.b.save_cursor()
            return
        self.b.guard.wait()
        t0 = time.perf_counter_ns()
        name = f"part-{len(self.state['shards']):05d}"
        if self.t == "ctx-records":
            data = _canon({"schema": CTX_RECORDS_SCHEMA, "partition": self.p,
                           "records": [r for x in self.buf for r in x[2]]})
            path = self.dir / f"{name}.json"
            arrow_nbytes, uncompressed = self.buf_nbytes, len(data)
        else:
            table = pa.concat_tables([x[2] for x in self.buf if x[4]]).combine_chunks()
            data = S.parquet_bytes(table, sealed=self.no_stats)
            path = self.dir / f"{name}.parquet"
            arrow_nbytes = int(table.nbytes)
            md = pq.ParquetFile(pa.BufferReader(data)).metadata
            uncompressed = sum(md.row_group(g).total_byte_size for g in range(md.num_row_groups))
        side = None
        if self.t == "exposed":
            side = _canon({"schema": SEALED_SIDECAR_SCHEMA, "partition": self.p,
                           "contexts": [x[5] for x in self.buf if x[5] is not None]})
        self.b.check_quota(len(data) + (len(side) if side is not None else 0))
        res = P._publish_immutable(path, data)
        shard = {"path": _rel(self.b.out, path), "sha256": res["sha256"], "bytes": res["bytes"], "rows": n_rows,
                 "first_context": self.buf[0][1], "last_context": self.buf[-1][1], "n_contexts": len(self.buf),
                 "first_index": self.buf[0][0], "last_index": last_index, "arrow_nbytes": arrow_nbytes,
                 "parquet_uncompressed": uncompressed, "adopted": bool(res.get("existed"))}
        if side is not None:
            sres = P._publish_immutable(self.dir / f"{name}.json", side)
            shard["sidecar"] = {"path": _rel(self.b.out, self.dir / f"{name}.json"), "sha256": sres["sha256"],
                                "bytes": sres["bytes"]}
        self.state["shards"].append(shard)
        self.state["next_index"] = last_index + 1
        self.buf, self.buf_nbytes = [], 0
        self.b.timers["write_ns"] += time.perf_counter_ns() - t0
        self.b.after_flush(self, shard)

    def finish(self, n_members: int) -> None:
        self.flush()
        if self.state["next_index"] < n_members:
            self.state["next_index"] = n_members
            self.b.save_cursor()


# -------------------------------------------------------------------------------------------- the build

class Build:
    """One `run` invocation over a claimed OUT directory (flock on OUT/build.lock held by the caller)."""

    def __init__(self, inputs: Inputs, out: Path, *, jtt1_cache: Path, budget_bytes: int | None, max_gets: int | None,
                 local_quota_bytes: int | None, offline: bool, resume: bool, continue_on_stop: bool, guard,
                 phase: str, argv: Sequence[str] | None, client_factory: Callable | None = None):
        self.inputs, self.out = inputs, Path(out)
        self.jtt1_cache = Path(jtt1_cache)
        self.budget_bytes, self.max_gets, self.local_quota = budget_bytes, max_gets, local_quota_bytes
        self.offline, self.resume, self.continue_on_stop = offline, resume, continue_on_stop
        self.guard, self.phase = guard, phase
        self.argv = common.redact_argv([str(a) for a in (argv or [])])
        self.client_factory = client_factory
        self.client = None
        self.local: Counter = Counter()
        self.timers: Counter = Counter()
        self.item_indexes: dict[Path, ItemIndex] = {}
        self.qual_cache: dict[Path, tuple[dict, dict]] = {}
        self.inc_cursor: dict | None = None
        self.inc_admitted: dict = {}
        self.pending_by_item: dict[Path, list] = defaultdict(list)
        self.cursor = self._open_cursor()

    # ---------------------------------------------------------------- cursor
    def _fresh_cursor(self) -> dict:
        return {"schema": CURSOR_SCHEMA, "pins": self.inputs.pins, "status": "RUNNING",
                "ledger": {"calls": 0, "http_attempts": 0, "bytes": 0, "cache_hits": 0, "local_bytes": {},
                           "invocations": []},
                "fetch": {"next_index": 0, "total": len(self.inputs.plan.contexts)},
                "writers": {p: {t: {"next_index": 0, "shards": []} for t in TABLES} for p in PARTITIONS},
                "members": {p: len(self.inputs.members[p]) for p in PARTITIONS},
                "incoming_cursor_reads": [], "incoming_entries": {}, "exposed_inputs": [],
                # contexts recorded as stopped under --continue-on-stop (sample only), a total over every partition
                # counted from the written ctx-records at finalize; identities and kinds stay in the ctx-records
                "stop_count": 0,
                "stop": None, "pause": None, "finalized": None,
                "started_utc_us": common.utc_us(), "completed_utc_us": None}

    def _open_cursor(self) -> dict:
        path = self.out / "cursor.json"
        if path.exists():
            if not self.resume:
                raise _stop("cursor", f"{path} exists and resume is off; shards are immutable, so use a new --out")
            cur = json.loads(path.read_bytes())
            if cur.get("schema") != CURSOR_SCHEMA or cur.get("pins") != self.inputs.pins:
                raise _stop("cursor", f"{path} was written for other inputs or settings (resume refuses any pin change)")
            for p in PARTITIONS:
                for t in TABLES:
                    for sh in cur["writers"][p][t]["shards"]:
                        for rec in (sh, sh.get("sidecar")):
                            if rec is None:
                                continue
                            f = self.out / rec["path"]
                            if not f.exists() or f.stat().st_size != rec["bytes"] or common.sha256_file(f) != rec["sha256"]:
                                raise common.ImmutableConflict(f"recorded shard {rec['path']} is missing or differs "
                                                               "from the cursor")
            return cur
        if any((self.out / d).exists() for d in ("train", "eval", "heldout", "exposed", "lineage", "provenance")) \
                and not self.resume:
            raise _stop("cursor", f"{self.out} holds shard directories but no cursor")
        return self._fresh_cursor()

    def save_cursor(self) -> None:
        common.atomic_write(self.out / "cursor.json", _canon(self.cursor))

    def _invocation(self) -> dict:
        return self.cursor["ledger"]["invocations"][-1]

    def _flush_local(self) -> None:
        led = self.cursor["ledger"]
        led["local_bytes"] = dict(sorted((Counter(led.get("local_bytes") or {}) + self.local).items()))
        inv = self._invocation()
        inv["local_bytes"] = dict(sorted((Counter(inv.get("local_bytes") or {}) + self.local).items()))
        inv["timers"] = dict(sorted((Counter(inv.get("timers") or {}) + self.timers).items()))
        self.local, self.timers = Counter(), Counter()

    def after_flush(self, w: Writer, shard: dict) -> None:
        self._flush_local()
        self.save_cursor()

    def check_quota(self, nbytes: int) -> None:
        if self.local_quota is None:
            return
        written = sum(sh["bytes"] + (sh.get("sidecar") or {}).get("bytes", 0)
                      for p in PARTITIONS for t in TABLES for sh in self.cursor["writers"][p][t]["shards"])
        fetched = self.cursor["ledger"]["bytes"]
        if written + fetched + nbytes > self.local_quota:
            raise Paused("PAUSED_QUOTA", f"local quota {self.local_quota} B: {written} B of shards + {fetched} B "
                                         f"fetched + {nbytes} B")

    def tick(self) -> None:
        self.guard.tick()

    # ---------------------------------------------------------------- driver
    def execute(self) -> dict:
        cur = self.cursor
        cur["ledger"]["invocations"].append({"phase": self.phase, "argv": self.argv, "started_utc_us": common.utc_us(),
                                             "finished_utc_us": None, "status": None,
                                             "code": self.inputs.pins["code"]})
        cur["status"], cur["pause"] = "RUNNING", None
        self.save_cursor()
        status, code, err_doc = "COMPLETE", EXIT_OK, None
        try:
            if self.phase in ("all", "fetch"):
                self.fetch_phase()
            if self.phase in ("all", "build"):
                self.build_phase()
                self.finalize()
            else:
                status = "FETCHED"
        except Paused as p:
            status, code = p.status, EXIT_PAUSED
            cur["pause"] = {"status": p.status, "reason": p.reason, "utc_us": common.utc_us()}
        except cpuguard.GuardTimeout as g:
            status, code = "PAUSED_GUARD", EXIT_PAUSED
            cur["pause"] = {"status": status, "reason": str(g)[:300], "utc_us": common.utc_us()}
        except common.ImmutableConflict as e:
            status, code = "STOPPED", EXIT_CONFLICT
            err_doc = {"kind": "conflict", "context": None, "message": str(e)[:500]}
        except (S.CorpusStop, P.ReconcileError) as e:
            status, code = "STOPPED", EXIT_STOP
            err_doc = {"kind": _kind(e), "context": getattr(e, "context", None), "message": str(e)[:500]}
        finally:
            self._flush_local()
            inv = self._invocation()
            inv.update(finished_utc_us=common.utc_us(), status=status, guard=self.guard.summary(),
                       maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            cur["status"] = "RUNNING" if status == "FETCHED" else status
            if err_doc is not None and cur.get("stop") is None:
                cur["stop"] = err_doc
            if status == "COMPLETE" and cur.get("completed_utc_us") is None:
                cur["completed_utc_us"] = common.utc_us()
            self.save_cursor()
        return {"status": status, "exit": code, "out": str(self.out), "stop": err_doc, "pause": cur.get("pause"),
                "ledger": {k: cur["ledger"][k] for k in ("calls", "http_attempts", "bytes", "cache_hits")}}

    # ---------------------------------------------------------------- fetch
    def _count_attempt(self, **_kw) -> None:
        self.cursor["ledger"]["http_attempts"] += 1

    def _client(self):
        if self.client is None:
            if self.client_factory is not None:
                self.client = self.client_factory()
            else:
                from dltrain import s3io
                self.client = s3io.make_client()
            self.client.meta.events.register("before-send.s3.GetObject", self._count_attempt,
                                             unique_id="dltrain-corpus-attempts")
        return self.client

    def fetch_phase(self) -> None:
        """Ensure every processed context's JTT1 is cached: a cache hit is re-verified (gz sha256 and bytes); a miss
        is one exact-VersionId GET through jtt1.load_raw, admitted only within --budget-bytes (cumulative) and
        --max-gets (calls + 1 and HTTP attempts + 3); otherwise PAUSED_BUDGET. --offline makes a miss a STOP."""
        cur, led = self.cursor, self.cursor["ledger"]
        ctxs = self.inputs.plan.contexts
        unsaved = 0
        while cur["fetch"]["next_index"] < len(ctxs):
            i = cur["fetch"]["next_index"]
            c = ctxs[i]
            self.tick()
            ref = self.inputs.entries[c.context_sha256].get("input_ref") or {}
            if not ref.get("raw_sha256") or not ref.get("sha256"):
                raise _stop("input", "epoch context lacks a JTT1 input_ref", c.context_sha256)
            path = jtt1.cache_path(self.jtt1_cache, ref)
            if path.exists():
                gz = path.read_bytes()
                if _sha(gz) != ref["sha256"] or len(gz) != int(ref["bytes"]):
                    raise _stop("source", f"cached JTT1 {path.name} differs from input_ref", c.context_sha256)
                led["cache_hits"] += 1
                self.local["jtt1_cache"] += len(gz)
                cur["fetch"]["next_index"] = i + 1
                unsaved += 1
                if unsaved >= 256:            # cache hits are cheap to re-verify after a crash
                    self.save_cursor()
                    unsaved = 0
                continue
            else:
                if self.offline:
                    raise _stop("jtt1_missing", f"--offline and {path.name} is not cached", c.context_sha256)
                need = int(ref["bytes"])
                if self.budget_bytes is None or led["bytes"] + need > self.budget_bytes:
                    raise Paused("PAUSED_BUDGET", f"--budget-bytes {self.budget_bytes}: {led['bytes']} B fetched + "
                                                  f"{need} B")
                if self.max_gets is not None and (led["calls"] + 1 > self.max_gets
                                                  or led["http_attempts"] + 3 > self.max_gets):
                    raise Paused("PAUSED_BUDGET", f"--max-gets {self.max_gets}: {led['calls']} calls, "
                                                  f"{led['http_attempts']} HTTP attempts")
                self.guard.wait()
                client = self._client()
                led["calls"] += 1
                sub: dict = {}
                try:
                    jtt1.load_raw(ref, self.jtt1_cache, client, sub)
                except jtt1.JTTError as err:
                    raise _stop("source", f"JTT1 GET: {err}", c.context_sha256) from None
                led["bytes"] += int(sub.get("bytes", 0))
            cur["fetch"]["next_index"] = i + 1
            self.save_cursor()
            unsaved = 0
        if unsaved:
            self.save_cursor()

    # ---------------------------------------------------------------- build
    def _start(self, p: str) -> int:
        ws = self.cursor["writers"][p]
        return min(ws[t]["next_index"] for t in TABLES) if self.inputs.members[p] else 0

    def _pending(self) -> list:
        return [c for p in PARTITIONS for c in self.inputs.members[p][self._start(p):]]

    def _read_incoming_cursor(self, pending: Sequence) -> None:
        if not any(c.cohort == "incoming" for c in pending):
            return
        proj, pin, side = read_incoming_cursor(self.inputs.sources.incoming_cursor, self.inputs.exposed)
        self.local["cursor"] += pin["bytes"]
        self.cursor["incoming_cursor_reads"].append(pin)
        recorded = self.cursor["incoming_entries"]
        pending_ids = {c.context_sha256 for c in pending}
        for c in self.inputs.plan.contexts:
            if c.cohort != "incoming":
                continue
            sha = entry_sha256(proj.get(c.context_sha256))
            if c.context_sha256 in recorded:
                if c.context_sha256 in pending_ids and recorded[c.context_sha256] != sha:
                    raise _stop("source", "the incoming cursor entry of a pending context changed since the first read",
                                c.context_sha256)
            else:
                recorded[c.context_sha256] = sha
        self.inc_cursor, self.inc_admitted = proj, side
        self.save_cursor()

    def _summary(self, c) -> tuple[Any, dict]:
        try:
            s, spin = P._summary(c, self.inc_cursor)
        except P.ReconcileError as err:
            raise _from_reconcile(err) from None
        if c.cohort == "incoming":
            return s, {"layout": "cursor_entry", "entry_sha256": self.cursor["incoming_entries"].get(c.context_sha256)}
        self.local["summary"] += spin.get("bytes", 0)
        return s, {k: spin[k] for k in ("path", "layout", "bytes", "sha256")}

    def _item_index(self, c) -> ItemIndex:
        gp = self.inputs.item_of.get(c.context_sha256)
        if gp is None:
            try:
                P._item_gp(c)
            except P.ReconcileError as err:
                raise _from_reconcile(err) from None
            raise _stop("layout", "incoming context without an item directory", c.context_sha256)
        idx = self.item_indexes.get(gp)
        if idx is None or c.context_sha256 not in idx.need:
            wanted = {}
            for x in self.pending_by_item.get(gp, []) or [c]:
                try:
                    wanted[x.context_sha256] = P._summary(x, self.inc_cursor)[0].outcome_bytes
                except P.ReconcileError:
                    continue
            if c.context_sha256 not in wanted:
                wanted[c.context_sha256] = self._summary(c)[0].outcome_bytes
            idx = ItemIndex(gp / "hwm-events.jsonl", wanted)
            self.local["item_scan"] += idx.scanned_bytes
            inv = self._invocation()
            for k, v in (("item_scans", 1), ("item_scan_lines", idx.scanned_lines), ("item_kept_lines", idx.kept_lines),
                         ("item_tail_lines", sum(idx.tail.values()))):
                inv[k] = inv.get(k, 0) + v
            self.item_indexes[gp] = idx
        return idx

    def _release_item(self, c) -> None:
        gp = self.inputs.item_of.get(c.context_sha256)
        idx = self.item_indexes.get(gp) if gp is not None else None
        if idx is not None and c.context_sha256 in idx.need and idx.release(c.context_sha256):
            del self.item_indexes[gp]

    def _qualified(self, c) -> dict | None:
        if not self.inputs.exposed:
            return None
        if c.cohort == "frozen":
            path = self.inputs.frozen_qualified or (P._ctx_dir(c).parent.parent / "qualified-views.json")
        else:
            path = P._item_gp(c) / "qualification.json"
        got = self.qual_cache.get(path)
        if got is None:
            if not path.exists():
                got = ({}, {"path": str(path), "missing": True})
            else:
                got = read_qualified(path, c.cohort)
                self.local["exposed_side"] += got[1]["bytes"]
            if c.cohort == "incoming" and len(self.qual_cache) >= 64:
                for k in [k for k in self.qual_cache if k.name == "qualification.json"][:32]:
                    del self.qual_cache[k]
            self.qual_cache[path] = got
            if not any(x.get("path") == str(path) for x in self.cursor["exposed_inputs"]):
                self.cursor["exposed_inputs"].append(got[1])
        return got[0]

    def build_phase(self) -> None:
        pending = self._pending()
        self._read_incoming_cursor(pending)
        self.pending_by_item = defaultdict(list)
        for c in pending:
            gp = self.inputs.item_of.get(c.context_sha256)
            if gp is not None:
                self.pending_by_item[gp].append(c)
        env = Env(inputs=self.inputs, jtt1_cache=self.jtt1_cache, inc_admitted=self.inc_admitted,
                  item_index=self._item_index, qualified=self._qualified, tick=self.tick, local=self.local,
                  timers=self.timers)
        for p in PARTITIONS:
            members = self.inputs.members[p]
            writers = {t: Writer(self, p, t) for t in TABLES}
            for i in range(self._start(p), len(members)):
                c = members[i]
                self.tick()
                env.local, env.timers = self.local, self.timers
                res = self._process(c, env)
                for t in TABLES:
                    writers[t].add(i, c.context_sha256, res.rows[t], res.sealed_ctx if t == "exposed" else None)
            for t in TABLES:
                writers[t].finish(len(members))

    def _process(self, c, env: Env) -> ContextResult:
        cid = c.context_sha256
        entry = self.inputs.entries[cid]
        try:
            try:
                s, spin = self._summary(c)
                res = project_context(c, entry, s, spin, env)
            except P.ReconcileError as err:          # a programs helper's STOP that no caller converted
                raise _from_reconcile(err) from None
        except S.CorpusStop as err:
            err = _ctx_stop(err, cid)
            if not self.continue_on_stop:
                raise err from None
            rec = {"schema": CTX_RECORD_SCHEMA, "context_sha256": cid, "cohort": c.cohort, "fold": c.fold,
                   "view_id": c.view_id, "item_id": c.item_id, "dataset_hash": c.dataset_hash,
                   "family_id": c.family_id, "split_role": c.split_role, "excluded": None, "records": 0, "refused": 0,
                   "stopped": {"kind": _kind(err), "message": str(err)[:300]}}
            self._check_record(rec)
            rows = {t: [] for t in TABLES}
            rows["ctx-records"] = [rec]
            return ContextResult(cid, rows, rec, {"context_sha256": cid, "stopped": _kind(err)}, None)
        finally:
            self._release_item(c)
        self._check_record(res.ctx_record)
        return res

    @staticmethod
    def _check_record(rec: dict) -> None:
        """Every ctx-record must pass the closed CTX_RECORD_JSON_SCHEMA before it is written (red-team LA-2)."""
        fails = check_ctx_record(rec)
        if fails:
            raise _stop("ctx_record", f"ctx-record outside its allowlist schema: {fails}", rec.get("context_sha256"))

    # ---------------------------------------------------------------- finalize
    def finalize(self) -> dict:
        """Statistics recomputed from the written shards (so resume is exact), the closure checks, then the partition
        manifests (train/manifest.json is the train closure), the exposed, lineage and provenance manifests, the sealed
        heldout summary and the top manifest (the build's provenance receipt, never a run-plan input). Non-sealed
        documents get totals, exclusions, reconciliation and tolerances over train_fit, dev and diag only; heldout's
        go to heldout/summary.json, referenced by sha256 (red-team LA-7)."""
        out, inp, cur = self.out, self.inputs, self.cursor
        label_shas: dict[str, str] = {}
        ctx_records = {p: read_ctx_records(out, cur, p) for p in PARTITIONS}
        cur["stop_count"] = sum(1 for p in PARTITIONS for r in ctx_records[p] if r.get("stopped"))
        for p in PARTITIONS:
            stats = label_stats_from_shards(out, cur, p, inp.seeds if p == "train_fit" else None, guard=self.guard)
            data = _canon(stats["label_stats"])
            common.atomic_write(part_dir(out, p) / "label-stats.json", data)
            label_shas[p] = _sha(data)
            if p == "train_fit":
                sdata = _canon(seed_stats_doc(stats["seed_stats"], inp.seeds_pin["sha256"], len(inp.seeds)))
                common.atomic_write(part_dir(out, p) / "seed-stats.json", sdata)
                label_shas["seed_stats"] = _sha(sdata)
        runtime = runtime_report_from_shards(out, cur, guard=self.guard)
        rdata = _canon(runtime)
        common.atomic_write(out / "runtime-report.json", rdata)
        totals, exclusions, recon = reconcile_totals(ctx_records, inp.members, inp.excluded, OPEN_PARTITIONS)
        h_totals, h_exclusions, h_recon = reconcile_totals(ctx_records, inp.members, {}, ("heldout",))
        self._closure_from_shards()
        part_docs, lineage_docs, prov_docs = {}, {}, {}
        sealed_docs: dict = {"exposed": {}, "heldout": {}}

        def publish(path: Path, doc: Mapping) -> dict:
            data = _canon(doc)
            P._publish_immutable(path, data)
            return {"path": _rel(out, path), "sha256": _sha(data)}

        for p in PARTITIONS:
            ref = publish(manifest_path(out, p), self._partition_manifest(p, label_shas, ctx_records[p]))
            refs = {t: publish(manifest_path(out, p, t), self._table_manifest(p, t))
                    for t in ("exposed", "lineage", "ctx-records")}
            if p == "heldout":
                sealed_docs["heldout"] = {"partition": ref, "exposed": refs["exposed"], "lineage": refs["lineage"],
                                          "provenance": refs["ctx-records"],
                                          "label_stats": {"path": _rel(out, part_dir(out, p) / "label-stats.json"),
                                                          "sha256": label_shas[p]}}
            else:
                part_docs[p] = ref
                sealed_docs["exposed"][p] = refs["exposed"]
                lineage_docs[p] = refs["lineage"]
                prov_docs[p] = refs["ctx-records"]
        flags = {r["context_sha256"]: ("stopped:" + r["stopped"]["kind"]) if r.get("stopped") else r["excluded"]
                 for r in ctx_records["heldout"] if r.get("stopped") or r.get("excluded")}
        sealed_docs["heldout"]["summary"] = publish(heldout_summary_path(out), {
            "schema": HELDOUT_SUMMARY_SCHEMA, "partition": "heldout", "totals": h_totals, "exclusions": h_exclusions,
            "reconciliation": h_recon, "tolerances": cv_sd_tolerances(ctx_records, ("heldout",)),
            "context_flags": dict(sorted(flags.items()))})
        closure = (self._tokenizer_closure(ctx_records["train_fit"]) if inp.tokenizer_closure else
                   {"relation": "not_checked", "reason": "subset build (poison harness): the closure is a full-epoch property"})
        expected = inp.expected_completed
        label_open = {k: v for k, v in label_shas.items() if k != "heldout"}
        manifest = {"schema": MANIFEST_SCHEMA, "experiment_id": inp.epoch_doc.get("experiment_id"),
                    "epoch": inp.epoch_doc.get("epoch"), "run_plan_ref": False,
                    "train_closure": part_docs["train_fit"],
                    "inputs": {"epoch": {"path": str(inp.epoch_path), "sha256": inp.plan.epoch_pin["sha256"],
                                         "bytes": inp.plan.epoch_pin["bytes"]},
                               "partition_map": {"path": str(inp.map_path), "sha256": inp.plan.map_pin["sha256"]},
                               "seeds": {k: inp.seeds_pin[k] for k in ("path", "sha256", "count")},
                               "tokenizer": inp.tok_pin, "jtt1_cache": str(self.jtt1_cache),
                               "program_dict": ({"path": str(inp.dict_index.dir), "manifest_sha256": inp.dict_index.sha256}
                                                if inp.dict_index else None),
                               "inventory": inp.inventory_pin, "sources": inp.pins["sources"],
                               "expected_completed": dict(sorted(expected.items())) if isinstance(expected, Mapping)
                               else expected,
                               "incoming_cursor_reads": cur["incoming_cursor_reads"],
                               "exposed_inputs": cur["exposed_inputs"]},
                    "pins": inp.pins, "settings": inp.pins["settings"], "code": code_shas(),
                    "partitions": part_docs, "sealed": sealed_docs, "lineage": lineage_docs, "provenance": prov_docs,
                    "label_stats_sha256": label_open, "runtime_report_sha256": _sha(rdata),
                    "totals": totals, "exclusions": exclusions, "reconciliation": recon,
                    "tolerances": cv_sd_tolerances(ctx_records, OPEN_PARTITIONS),
                    "deviations": {"cv_log_ratio_sd": S.CV_SD_POLICY, "runtime_target": S.RUNTIME_POLICY_NOTE},
                    "stop_count": cur["stop_count"],
                    "tokenizer_closure": closure,
                    "ledger": {k: cur["ledger"][k] for k in ("calls", "http_attempts", "bytes", "cache_hits",
                                                             "local_bytes")},
                    "guard": self.guard.summary(), "invocations": len(cur["ledger"]["invocations"]),
                    "times": {"started_utc_us": cur["started_utc_us"], "finalized_utc_us": common.utc_us()}}
        data = _canon(manifest)
        common.atomic_write(out / "manifest.json", data)
        cur["finalized"] = {"manifest_sha256": _sha(data)}
        self._closure_names()
        return manifest

    def _closure_names(self) -> None:
        """Training closure: no exposed name in a non-sealed schema or in any key of a non-sealed JSON (manifests,
        statistics, ctx-records and the cursor) — a STOP at build time and a FAIL at verify."""
        for t in ("records", "contexts", "refused", "lineage"):
            bad = [n for n in table_schema(t).names if S.name_is_exposed(n)]
            if bad:
                raise _stop("closure", f"non-sealed schema {t} holds exposed names {bad}")
        self.save_cursor()
        for path in nonsealed_json(self.out, self.cursor):
            bad = key_violations(json.loads(path.read_bytes()))
            if bad:
                raise _stop("closure", f"{_rel(self.out, path)} holds exposed key names {bad[:5]}")

    def _closure_from_shards(self) -> None:
        """No train/ row or context with a split_role other than train_fit (re-read from the written shards)."""
        for t in ("records", "contexts", "refused"):
            for sh in self.cursor["writers"]["train_fit"][t]["shards"]:
                roles = set(pq.read_table(self.out / sh["path"], columns=["split_role"],
                                          use_threads=False).column(0).to_pylist())
                if roles - {"train_fit"}:
                    raise _stop("closure", f"{sh['path']} holds split_role {sorted(roles)}")

    def _partition_manifest(self, p: str, label_shas: Mapping, recs: list) -> dict:
        """The partition manifest. For train_fit it is the TRAIN CLOSURE (red-team LA-2): it binds only the train
        shards, the label and seed statistics, the tokenizer pins, the schema shas and the test-independent
        partition-map and seed-list pins; no ctx-record, sealed or heldout reference, no path outside the partition and
        no timestamp. It is the only artifact a run plan may use as refs.corpus (train_closure_ref)."""
        cur, inp = self.cursor, self.inputs
        tables = {}
        for t in PARTITION_TABLES:
            tables[t] = {"schema_id": table_id(t), "schema_sha256": S.schema_sha256(table_schema(t)),
                         "files": [{k: sh[k] for k in ("path", "sha256", "bytes", "rows", "arrow_nbytes",
                                                         "parquet_uncompressed")}
                                   for sh in cur["writers"][p][t]["shards"]]}
        counts = Counter(contexts=0)
        for r in recs:
            counts["contexts"] += 1
            if r.get("excluded"):
                counts["excluded:" + r["excluded"]] += 1
            elif r.get("stopped"):
                counts["stopped"] += 1
            else:
                counts["records"] += r["records"]
                counts["refused"] += r["refused"]
        doc = {"schema": PARTITION_SCHEMA, "partition": p, "role": S.READ_ROLE[p], "run_plan_ref": p == "train_fit",
               "tables": tables, "label_stats_sha256": label_shas[p], "counts": dict(sorted(counts.items())),
               "tokenizer": {"vocab_sha256": inp.vocab.sha256, "max_seq": inp.max_seq,
                             "lengths_sha256": inp.tok_pin["lengths_json_sha256"]},
               "inputs": {"partition_map_sha256": inp.plan.map_pin["sha256"], "seeds_sha256": inp.seeds_pin["sha256"]},
               "parquet": {"compression": "zstd", "zstd_level": ZSTD_LEVEL, "row_group_rows": ROW_GROUP_ROWS,
                           "use_dictionary": True, "write_statistics": p != "heldout", "pyarrow": pa.__version__,
                           "target_shard_bytes": inp.target_shard_bytes}}
        if p == "train_fit":
            doc["seed_stats_sha256"] = label_shas.get("seed_stats")
        return doc

    def _table_manifest(self, p: str, t: str) -> dict:
        """Manifest of the exposed (sealed), lineage (role lineage_report, no statistics) or ctx-records (provenance;
        role provenance_audit, heldout's under heldout_report) table of one partition. None is a run-plan input."""
        shards = self.cursor["writers"][p][t]["shards"]
        schema = {"exposed": SEALED_MANIFEST_SCHEMA, "lineage": LINEAGE_MANIFEST_SCHEMA,
                  "ctx-records": PROVENANCE_MANIFEST_SCHEMA}[t]
        if t == "exposed":
            role = S.EXPOSED_READ_ROLE
        elif t == "lineage":
            role = LINEAGE_ROLE
        else:
            role = "heldout_report" if p == "heldout" else PROVENANCE_ROLE
        doc = {"schema": schema, "partition": p, "table": t, "schema_id": table_id(t),
               "schema_sha256": S.schema_sha256(table_schema(t)) if t != "ctx-records"
               else _sha(common.canonical(CTX_RECORD_JSON_SCHEMA)),
               "role": role, "run_plan_ref": False,
               "files": [{k: sh[k] for k in ("path", "sha256", "bytes", "rows", "arrow_nbytes", "parquet_uncompressed")}
                         for sh in shards]}
        if t != "ctx-records":
            doc["write_statistics"] = has_statistics(p, t)
        if t == "lineage":
            doc["note"] = ("search lineage (operator, parents, generation, birth_seq, island_policy, island and proposal "
                           "fields) of a search that selected on outer-test-derived scores (test-gated islands, elite "
                           "tuning): test-influenced, never read by the primary-arm sampler, never a run-plan input")
        if t == "exposed":
            totals: Counter = Counter()
            for sh in shards:
                side = sh.get("sidecar")
                if not side:
                    continue
                for ctx in json.loads((self.out / side["path"]).read_bytes())["contexts"]:
                    for k, v in ctx.items():
                        if k == "context_sha256":
                            continue
                        if isinstance(v, bool) or isinstance(v, str) or v is None:
                            totals[f"{k}:{v}"] += 1
                        elif isinstance(v, int):
                            totals[k] += v
            doc["sidecars"] = [sh["sidecar"] for sh in shards if sh.get("sidecar")]
            doc["counts"] = dict(sorted(totals.items()))
        return doc

    def _tokenizer_closure(self, train_recs: Sequence[Mapping]) -> dict:
        """sha256 of the sorted distinct train program_sha256 == the vocab input_closure.program_set_sha256. With
        train_fit target_leak exclusions the train set must be a subset of the P4a train_fit set (recorded)."""
        mine = distinct_sorted([self.out / sh["path"] for sh in self.cursor["writers"]["train_fit"]["records"]["shards"]],
                               "program_sha256", guard=self.guard)
        digest = program_set_sha256(mine)
        want = (self.inputs.tok_closure or {}).get("program_set_sha256")
        doc = {"train_programs": len(mine), "program_set_sha256": digest, "vocab_program_set_sha256": want,
               "relation": "equal" if digest == want else None}
        if digest == want:
            return doc
        if not any(r.get("excluded") == "target_leak" or r.get("stopped") for r in train_recs):
            raise _stop("tokenizer_closure", f"train program set {digest[:16]} != vocab input_closure {str(want)[:16]}")
        if self.inputs.dict_index is None:
            doc["relation"] = "subset_unverified_without_program_dict"
            return doc
        theirs = self.inputs.dict_index.train_program_hashes()
        if program_set_sha256(theirs) != want:
            raise _stop("tokenizer_closure", "the P4a train_fit program set does not hash to the vocab input_closure")
        inside = pc.is_in(mine, value_set=theirs)
        outside = len(mine) - pc.sum(inside).as_py() if len(mine) else 0
        if outside:
            raise _stop("tokenizer_closure", f"{outside} train programs are outside the vocab closure")
        doc.update(relation="subset", difference=len(theirs) - len(mine))
        return doc


def distinct_sorted(paths: Sequence[Path], column: str, *, hex_binary: bool = False, guard=None) -> pa.Array:
    """The distinct values of one string column over Parquet files, byte-sorted (binary(32) hashes as hex when
    hex_binary), kept in Arrow memory (millions of program hashes at full scale). Guard.tick() per file."""
    parts = []
    for p in paths:
        if guard is not None:
            guard.tick()
        col = pq.read_table(p, columns=[column], use_threads=False).column(0).combine_chunks()
        if isinstance(col, pa.DictionaryArray):
            col = col.dictionary_decode()
        parts.append(pc.unique(col))
    if not parts:
        return pa.array([], type=pa.string())
    vals = pc.unique(pa.concat_arrays(parts))
    if hex_binary:
        vals = pa.array([b.hex() for b in vals.to_pylist()], type=pa.string())
    return vals.take(pc.sort_indices(vals))


def program_set_sha256(sorted_vals: pa.Array) -> str:
    """dsl_tokenizer's input_closure rule: sha256 of the sorted distinct program_sha256 hex values, one per line."""
    h = hashlib.sha256()
    for i in range(0, len(sorted_vals), 65536):
        h.update("".join(s + "\n" for s in sorted_vals.slice(i, 65536).to_pylist()).encode("ascii"))
    return h.hexdigest()


def code_shas() -> dict:
    mods = (("corpus_build.py", sys.modules[__name__]), ("corpus_schema.py", S), ("programs.py", P),
            ("labels.py", labels), ("jtt1.py", jtt1), ("dsl_tokenizer.py", T), ("common.py", common),
            ("cpuguard.py", cpuguard))
    return {name: common.sha256_file(Path(mod.__file__)) for name, mod in mods}


def reconcile_totals(ctx_records: Mapping[str, list], members: Mapping[str, list], plan_excluded: Mapping[str, int],
                     partitions: Sequence[str] = PARTITIONS) -> tuple[dict, dict, dict]:
    """R6 over the given partitions' contexts: per context records + refused == unique_programs, and the sums per
    partition and cohort; exclusions by reason (P1's and the map roles from plan_excluded, this packet's target_leak
    and fold-0 contradictions, and stopped:<kind> under --continue-on-stop); the reconciliation summary (RECON_KEYS
    summed over the ctx-records' counts, and the R-check tallies)."""
    totals: dict = {"by_partition": {}, "by_cohort": {}}
    exclusions = Counter(plan_excluded)
    recon: Counter = Counter()
    for p in partitions:
        recs = ctx_records[p]
        if len(recs) != len(members[p]):
            raise _stop("count", f"partition {p}: {len(recs)} ctx-records != {len(members[p])} contexts")
        for r in recs:
            if r.get("excluded"):
                exclusions[r["excluded"]] += 1
                exclusions["fold0_construction_contradiction"] += bool(r.get("fold0_construction_contradiction"))
                continue
            if r.get("stopped"):
                exclusions["stopped:" + r["stopped"]["kind"]] += 1
                continue
            if r["records"] + r["refused"] != r["summary"]["unique_programs"]:
                raise _stop("count", "records + refused != unique_programs", r["context_sha256"])
            for key, k in (("by_partition", p), ("by_cohort", r["cohort"])):
                d = totals[key].setdefault(k, {"contexts": 0, "records": 0, "refused": 0, "unique_programs": 0})
                d["contexts"] += 1
                d["records"] += r["records"]
                d["refused"] += r["refused"]
                d["unique_programs"] += r["summary"]["unique_programs"]
            for k, v in (r.get("checks") or {}).items():
                recon[k] += bool(v)
            for k in RECON_KEYS:
                recon[k] += int((r.get("counts") or {}).get(k, 0) or 0)
    for key in ("by_partition", "by_cohort"):
        for d in totals[key].values():
            if d["records"] + d["refused"] != d["unique_programs"]:
                raise _stop("count", f"R6: Σ records + refused != Σ unique_programs ({key})")
    totals["by_partition"] = dict(sorted(totals["by_partition"].items()))
    totals["by_cohort"] = dict(sorted(totals["by_cohort"].items()))
    exclusions = Counter({k: v for k, v in exclusions.items() if v})
    return totals, dict(sorted(exclusions.items())), dict(sorted(recon.items()))


def cv_sd_tolerances(ctx_records: Mapping[str, list], partitions: Sequence[str]) -> dict:
    """The run receipt of the cv_log_ratio_sd tolerance (S.CV_SD_POLICY) over the given partitions: the counts summed
    from the ctx-records, the tolerated native/outcome hits per context, and every hit with its context, program and
    both values' bits."""
    sums: Counter = Counter()
    by_ctx, items = {}, []
    for p in partitions:
        for r in ctx_records[p]:
            c = r.get("counts") or {}
            for k in ("cv_sd_ulp_ties", "cv_sd_canonical", "cv_sd_canonical_ulp", "cv_sd_verbatim",
                      "cv_sd_writer_nonfinite"):
                sums[k] += int(c.get(k) or 0)
            ties = c.get("cv_sd_ulp_tie_items") or []
            if ties:
                by_ctx[r["context_sha256"]] = len(ties)
                items += [{"context_sha256": r["context_sha256"], "partition": p, **t} for t in ties]
    return {"field": S.CV_SD_FIELD, "max_ulps": S.CV_SD_MAX_ULPS, "native_outcome_ties": sums["cv_sd_ulp_ties"],
            "canonical_stored": sums["cv_sd_canonical"], "canonical_1ulp_from_writer": sums["cv_sd_canonical_ulp"],
            "verbatim_stored": sums["cv_sd_verbatim"], "writer_nonfinite": sums["cv_sd_writer_nonfinite"],
            "contexts_with_ties": len(by_ctx), "by_context": dict(sorted(by_ctx.items())),
            "items": sorted(items, key=lambda x: (x["context_sha256"], x["program_sha256"]))}


# ------------------------------------------------------------------------------------- statistics

def read_ctx_records(out: Path, cursor: Mapping, partition: str) -> list[dict]:
    recs: list[dict] = []
    for sh in cursor["writers"][partition]["ctx-records"]["shards"]:
        recs.extend(json.loads((Path(out) / sh["path"]).read_bytes())["records"])
    return recs


LABEL_COLUMNS = ("context_sha256", "program_sha256", "skeleton_sha256", "outcome", "rmse_j", "baseline_rmse_j",
                 "cv_mean_log_ratio", "cv_mean_ratio", "wall_us", "backend", "shared_seed")


def iter_context_rows(paths: Sequence[Path], columns: Sequence[str]) -> Iterator[tuple[str, list[dict]]]:
    """Rows grouped by context in file order (shards are sorted and cut only at context boundaries)."""
    cur_ctx, rows = None, []
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=list(columns), use_threads=False, batch_size=65536):
            for r in batch.to_pylist():
                if r["context_sha256"] != cur_ctx:
                    if cur_ctx is not None:
                        yield cur_ctx, rows
                    cur_ctx, rows = r["context_sha256"], []
                rows.append(r)
    if cur_ctx is not None:
        yield cur_ctx, rows


def contexts_rows(out: Path, cursor: Mapping, partition: str) -> dict[str, dict]:
    ctxs = {}
    for sh in cursor["writers"][partition]["contexts"]["shards"]:
        for r in pq.read_table(Path(out) / sh["path"], use_threads=False).to_pylist():
            ctxs[r["context_sha256"]] = r
    return ctxs


def label_stats_from_shards(out: Path, cursor: Mapping, partition: str, seeds: frozenset | None, *,
                            guard=None) -> dict:
    """label-stats (jope.dl-corpus-label-stats/v1) over the partition's read-back rows: labels.label_context per
    context, equal to labels.label_corpus (each context is labelled from its own rows only); with seeds, the train
    seed statistics too. Guard.tick() per context."""
    out = Path(out)
    ctxs = contexts_rows(out, cursor, partition)
    paths = [out / sh["path"] for sh in cursor["writers"][partition]["records"]["shards"]]
    status, outcomes, qstat, gates = Counter(), Counter(), Counter(), Counter()
    feas: dict[str, Counter] = defaultdict(Counter)
    rt: dict[str, Counter] = defaultdict(Counter)
    q_vals: list[float] = []
    clip_lo = clip_hi = detailed = detailless = teach = n_rec = 0
    seen: set[str] = set()
    per_seed: dict[str, Counter] = defaultdict(Counter)
    seed_ctx: dict[str, set] = defaultdict(set)
    seed_out: Counter = Counter()
    shared = 0
    for cid, rows in iter_context_rows(paths, LABEL_COLUMNS):
        if guard is not None:
            guard.tick()
        ctx = ctxs.get(cid)
        if ctx is None:
            raise _stop("count", f"{partition}: records of a context without a context row", cid)
        seen.add(cid)
        clab, labs = labels.label_context(ctx, rows)
        status[clab.status] += 1
        good_skeletons = set()
        by_sha = {r["program_sha256"]: r for r in rows}
        for lab in labs:
            n_rec += 1
            outcomes[lab.outcome] += 1
            qstat[lab.q_status] += 1
            gates["null" if lab.gate_pass is None else str(lab.gate_pass).lower()] += 1
            if lab.outcome == labels.COMPLETED:
                if labels.is_detailed(by_sha[lab.program_sha256]):
                    detailed += 1
                else:
                    detailless += 1
            if lab.q is not None:
                q_vals.append(float(lab.q))
                clip_lo += lab.q < -labels.Q_CLIP
                clip_hi += lab.q > labels.Q_CLIP
            if lab.feas_class is not None:
                feas[lab.backend]["labelled"] += 1
                feas[lab.backend]["feas_valid"] += bool(lab.feas_valid)
            if lab.rt_target is not None:
                rt[lab.backend]["rt_target"] += 1
                rt[lab.backend]["rt_valid"] += bool(lab.rt_valid)
            if lab.outcome == labels.COMPLETED and lab.q_status == "known" and lab.gate_pass is True:
                good_skeletons.add(lab.skeleton_sha256)
            if seeds is not None and lab.shared_seed:
                shared += 1
                seed_out[lab.outcome] += 1
                seed_ctx[lab.program_sha256].add(cid)
                ps = per_seed[lab.program_sha256]
                ps["completed"] += lab.outcome == labels.COMPLETED
                ps["gate_pass"] += lab.gate_pass is True
                ps["q_known"] += lab.q_status == "known"
        if clab.status == "OK" and len(good_skeletons) >= 4:
            teach += 1
    for cid in sorted(set(ctxs) - seen):         # every program of the context was refused
        status[labels.label_context(ctxs[cid], [])[0].status] += 1
    qs = None
    if q_vals:
        arr = np.asarray(q_vals, dtype=np.float64)
        qs = {name: float(np.percentile(arr, pct, method="linear")) for name, pct in LABEL_Q_PCTS}
    doc = {"label_stats": {"schema": LABEL_STATS_SCHEMA, "partition": partition, "contexts": len(ctxs),
                           "context_status_counts": dict(sorted(status.items())), "records": n_rec,
                           "outcome_counts": dict(sorted(outcomes.items())),
                           "q_status_counts": dict(sorted(qstat.items())),
                           "gate_pass_counts": dict(sorted(gates.items())), "completed_detailed": detailed,
                           "completed_detailless": detailless, "q_quantiles": qs,
                           "q_opt_clipped": {"low": clip_lo, "high": clip_hi},
                           "feas_valid_by_backend": {b: dict(sorted(v.items())) for b, v in sorted(feas.items())},
                           "rt_valid_by_backend": {b: dict(sorted(v.items())) for b, v in sorted(rt.items())},
                           "teacher_eligible_contexts": teach}}
    if seeds is not None:
        doc["seed_stats"] = {"seeds_present": len(per_seed), "records_shared_seed": shared,
                             "by_outcome": dict(sorted(seed_out.items())),
                             "per_seed": {s: {"contexts": len(seed_ctx[s]), "completed": v["completed"],
                                              "gate_pass": v["gate_pass"], "q_known": v["q_known"]}
                                          for s, v in sorted(per_seed.items())}}
    return doc


def seed_stats_doc(stats: Mapping, seed_list_sha256: str, n_seeds: int) -> dict:
    return {"schema": SEED_STATS_SCHEMA, "seed_list_sha256": seed_list_sha256, "seeds": n_seeds, **stats}


def runtime_report_from_shards(out: Path, cursor: Mapping, *, guard=None) -> dict:
    """Per derived backend x partition (train_fit, dev, diag; never heldout) x cohort: outcome counts, the timeout
    rate timeout / (completed + failed + timeout) and nearest-rank wall_us quantiles of completed and timeout rows.
    Guard.tick() per context."""
    out = Path(out)
    groups: dict[tuple, dict] = {}
    for p in RUNTIME_PARTITIONS:
        cohort_of = {cid: r["cohort"] for cid, r in contexts_rows(out, cursor, p).items()}
        paths = [out / sh["path"] for sh in cursor["writers"][p]["records"]["shards"]]
        for cid, rows in iter_context_rows(paths, ("context_sha256", "outcome", "backend", "wall_us")):
            if guard is not None:
                guard.tick()
            for r in rows:
                g = groups.setdefault((r["backend"], p, cohort_of[cid]),
                                      {"counts": Counter(), "completed": [], "timeout": []})
                o = r["outcome"]
                cls = ("completed" if o == labels.COMPLETED else "failed" if o.startswith(labels.FAILED_PREFIX)
                       else "timeout" if o == labels.TIMEOUT_OUTCOME else "abandoned" if o.startswith("abandoned:")
                       else "missing")
                g["counts"]["records"] += 1
                g["counts"][cls] += 1
                if cls in ("completed", "timeout") and r["wall_us"] is not None:
                    g[cls].append(int(r["wall_us"]))
    rows_out = []
    for (backend, p, coh), g in sorted(groups.items()):
        c = g["counts"]
        denom = c["completed"] + c["failed"] + c["timeout"]
        doc = {"backend": backend, "partition": p, "cohort": coh,
               **{k: c[k] for k in ("records", "completed", "failed", "timeout", "abandoned", "missing")},
               "timeout_rate": (c["timeout"] / denom) if denom else None}
        for cls in ("completed", "timeout"):
            vals = sorted(g[cls])
            doc[f"wall_us_{cls}"] = {**{name: _nearest_rank(vals, frac) for name, frac in RUNTIME_QS},
                                     "max": vals[-1] if vals else None, "n": len(vals)}
        rows_out.append(doc)
    return {"schema": RUNTIME_SCHEMA, "partitions": list(RUNTIME_PARTITIONS),
            "quantile_rule": "nearest rank: the ceil(q*n)-th smallest wall_us", "groups": rows_out}


# ------------------------------------------------------------------------------------------ readers

def open_partition(out: str | os.PathLike, partition: str, role: str = "fit") -> dict:
    """The manifest-listed files of one partition {records, contexts, refused} -> [paths], only under that partition's
    own read role (corpus_schema.READ_ROLE, red-team LA-4): fit opens train_fit, training_validation dev,
    diagnostic_only diag and heldout_report heldout (P10 reporting only); any other pairing is a PermissionError.
    The ctx-records are provenance (open_provenance), never part of a partition."""
    if partition not in PARTITIONS or role not in ROLES:
        raise ValueError(f"partition must be one of {PARTITIONS} and role one of {ROLES}")
    if role != S.READ_ROLE[partition]:
        raise PermissionError(f"the {partition} partition opens only under role={S.READ_ROLE[partition]!r}, not "
                              f"{role!r}")
    out = Path(out)
    doc = json.loads(manifest_path(out, partition).read_bytes())
    return {t: [out / f["path"] for f in doc["tables"][t]["files"]] for t in PARTITION_TABLES}


def load_split(out: str | os.PathLike, partition: str, role: str = "fit") -> tuple[list[dict], list[dict]]:
    """(contexts, records) of one partition as dicts that sampler.Corpus.build and labels.label_corpus accept, under
    the partition's own read role (open_partition)."""
    files = open_partition(out, partition, role)
    return S.read_contexts(files["contexts"]), S.read_records(files["records"])


def train_closure_ref(out: str | os.PathLike) -> dict:
    """{path, sha256, bytes} of the train closure (train/manifest.json): the only artifact a run plan may use as
    refs.corpus (red-team LA-2)."""
    out = Path(out)
    path = manifest_path(out, "train_fit")
    doc = json.loads(path.read_bytes())
    if doc.get("schema") != PARTITION_SCHEMA or doc.get("run_plan_ref") is not True:
        raise ValueError(f"{path} is not a {PARTITION_SCHEMA} train closure")
    return {"path": _rel(out, path), "sha256": common.sha256_file(path), "bytes": path.stat().st_size}


def open_lineage(out: str | os.PathLike, partition: str, role: str | None = None) -> list[Path]:
    """The lineage table of one partition (search lineage selected on outer-test-derived scores), only under role
    'lineage_report' (red-team LA-8; heldout's additionally needs the heldout partition, so it is refused here)."""
    if role != LINEAGE_ROLE:
        raise PermissionError(f"lineage tables open only under role={LINEAGE_ROLE!r}")
    if partition not in OPEN_PARTITIONS:
        raise PermissionError(f"lineage opens for {OPEN_PARTITIONS} only (heldout's is sealed)")
    out = Path(out)
    doc = json.loads(manifest_path(out, partition, "lineage").read_bytes())
    return [out / f["path"] for f in doc["files"]]


def open_provenance(out: str | os.PathLike, partition: str, role: str | None = None) -> list[Path]:
    """The ctx-record files of one partition: role 'provenance_audit' for train_fit, dev and diag, 'heldout_report'
    for heldout. Provenance pins depend on test bytes; never a run-plan input."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    want = "heldout_report" if partition == "heldout" else PROVENANCE_ROLE
    if role != want:
        raise PermissionError(f"the {partition} provenance opens only under role={want!r}")
    out = Path(out)
    doc = json.loads(manifest_path(out, partition, "ctx-records").read_bytes())
    return [out / f["path"] for f in doc["files"]]


def open_exposed(out: str | os.PathLike, partition: str, role: str | None = None) -> list[Path]:
    """The sealed exposed table of one partition, only under role='exposed_report' (never in a run-plan closure)."""
    if role != S.EXPOSED_READ_ROLE:
        raise PermissionError("exposed tables open only under role='exposed_report'")
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    out = Path(out)
    doc = json.loads(manifest_path(out, partition, "exposed").read_bytes())
    return [out / f["path"] for f in doc["files"]]


# ------------------------------------------------------------------------------------------- run entry

def _guard_for(out: Path, guard, max_pause_s: float | None):
    return guard if guard is not None else cpuguard.Guard(journal_path=out / "guard-journal.jsonl",
                                                          max_pause_s=max_pause_s)


def run(*, epoch, partition_map, seeds, tokenizer, out, jtt1_cache, budget_bytes=None, epoch_sha256=None,
        max_gets=None, local_quota_bytes=None, program_dict=None, inventory=None, incoming_cursor=None,
        incoming_transport_index=None, frozen_qualified_views=None, exposed=True, phase="all",
        target_shard_bytes=TARGET_SHARD_BYTES, offline=False, resume=True, continue_on_stop=False, gate_p3=None,
        gate_p4=None, expected_completed=None, max_pause_s=1800.0, guard=None, client_factory=None, tokenizer_closure=True,
        argv=None) -> dict:
    """One `run` invocation; returns {status, exit, ...}. STOPs before the cursor exists propagate as CorpusStop."""
    cpuguard.require_slice_limits()
    cpuguard.set_threads(4)
    if phase not in ("all", "fetch", "build"):
        raise ValueError("phase must be all, fetch or build")
    outp = _check_out(out, gate_p3=gate_p3, gate_p4=gate_p4, partition_map=partition_map)
    real_run = outp == Path(os.path.realpath(CORPUS_OUT))
    if continue_on_stop and real_run:
        raise _stop("gate", "--continue-on-stop is for the sample only")
    inputs = load_inputs(epoch=epoch, epoch_sha256=epoch_sha256, partition_map=partition_map, seeds=seeds,
                         tokenizer=tokenizer, program_dict=program_dict, inventory=inventory,
                         incoming_cursor=incoming_cursor, incoming_transport_index=incoming_transport_index,
                         frozen_qualified_views=frozen_qualified_views, exposed=exposed,
                         expected_completed=expected_completed, target_shard_bytes=target_shard_bytes,
                         tokenizer_closure=tokenizer_closure, real_run=real_run)
    outp.mkdir(parents=True, exist_ok=True)
    guard = _guard_for(outp, guard, max_pause_s)
    with open(outp / "build.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _stop("cursor", f"another build holds {outp / 'build.lock'}") from None
        b = Build(inputs, outp, jtt1_cache=Path(jtt1_cache), budget_bytes=budget_bytes, max_gets=max_gets,
                  local_quota_bytes=local_quota_bytes, offline=offline, resume=resume,
                  continue_on_stop=continue_on_stop, guard=guard, phase=phase, argv=argv,
                  client_factory=client_factory)
        return b.execute()


# ------------------------------------------------------------------------------------------- verify

class _Fails:
    def __init__(self):
        self.n, self.items = 0, []

    def __call__(self, msg: str) -> None:
        self.n += 1
        if len(self.items) < 200:
            self.items.append(msg)


def _check_file(out: Path, rec: Mapping, fail: _Fails) -> bool:
    f = Path(out) / rec["path"]
    if not f.exists() or f.stat().st_size != rec["bytes"] or common.sha256_file(f) != rec["sha256"]:
        fail(f"{rec['path']}: missing or bytes/sha256 differ from the cursor")
        return False
    return True


def nonsealed_json(out: Path, cur: Mapping) -> list[Path]:
    paths = [out / "cursor.json", out / "manifest.json", out / "runtime-report.json",
             part_dir(out, "train_fit") / "seed-stats.json"]
    for p in OPEN_PARTITIONS:
        paths += [manifest_path(out, p), part_dir(out, p) / "label-stats.json", manifest_path(out, p, "lineage"),
                  manifest_path(out, p, "ctx-records")]
        paths += [out / sh["path"] for sh in cur["writers"][p]["ctx-records"]["shards"]]
    return [p for p in paths if p.exists()]


def verify(out: str | os.PathLike, *, schema_allowlist: bool = True, counts: bool = True, poison_test_run: bool = False,
           poison_contexts: int = 2, poison_work: str | os.PathLike | None = None, role: str | None = None,
           rehash_sources: bool = False, receipt: str | os.PathLike | None = None, guard=None) -> dict:
    """verify (PLAN §5 P5 §3.9). It reads RUN/corpus/epoch-000 once COMPLETE without gate receipts (_check_read).
    Sealed files are checked at footer level (sha, bytes, schema, row counts, no statistics) unless
    role='heldout_report', which decodes heldout records, contexts and ctx-records; exposed tables are never decoded
    here. --schema-allowlist adds the per-column value allowlist, a JSON Schema check of a deterministic row sample
    of every decoded table and the closed ctx-record schema. Returns the jope.dl-corpus-verify/v2 receipt."""
    cpuguard.require_slice_limits()
    cpuguard.set_threads(4)
    if role not in (None, "heldout_report"):
        raise ValueError("role must be None or 'heldout_report'")
    out = _check_read(out)
    receipt_path = _check_receipt(receipt) if receipt else None
    guard = _guard_for(out, guard, 1800.0)
    guard.wait()
    fail = _Fails()
    cur = json.loads((out / "cursor.json").read_bytes())
    man = json.loads((out / "manifest.json").read_bytes()) if (out / "manifest.json").exists() else None
    checked: Counter = Counter()
    if man is None or cur.get("status") != "COMPLETE":
        fail("the build is not COMPLETE (no manifest)")
    decode_heldout = role == "heldout_report"
    if man is not None and man.get("schema") != MANIFEST_SCHEMA:
        fail(f"manifest schema {man.get('schema')!r} != {MANIFEST_SCHEMA}")
    if man is not None and schema_allowlist:
        for p in PARTITIONS:
            for t in PARQUET_TABLES:
                sealed = is_sealed(p, t)
                for sh in cur["writers"][p][t]["shards"]:
                    guard.tick()
                    if not _check_file(out, sh, fail):
                        continue
                    decode = (not sealed) or (decode_heldout and p == "heldout" and t != "exposed")
                    for msg in S.check_parquet(out / sh["path"], table_schema(t), decode,
                                               sealed=not has_statistics(p, t),
                                               json_sample=JSON_SAMPLE_EVERY if decode else None):
                        fail(f"{sh['path']}: {msg}")
                    if pq.ParquetFile(out / sh["path"]).metadata.num_rows != sh["rows"]:
                        fail(f"{sh['path']}: footer rows differ from the cursor")
                    checked["parquet_files"] += 1
                    if sh.get("sidecar"):
                        _check_file(out, sh["sidecar"], fail)
        for t in ("records", "contexts", "refused", "lineage"):
            bad = [n for n in table_schema(t).names if S.name_is_exposed(n)]
            if bad:
                fail(f"non-sealed schema {t} holds exposed names {bad}")
        for path in nonsealed_json(out, cur):
            try:
                doc = json.loads(path.read_bytes())
            except (OSError, ValueError) as err:
                fail(f"{_rel(out, path)}: unreadable ({err})")
                continue
            bad = key_violations(doc)
            if bad:
                fail(f"{_rel(out, path)}: exposed key names {bad[:5]}")
            checked["json_files"] += 1
        for p in PARTITIONS:
            if p == "heldout" and not decode_heldout:
                continue
            for r in read_ctx_records(out, cur, p):
                guard.tick()
                for msg in check_ctx_record(r):
                    fail(f"{p}: ctx-record {str(r.get('context_sha256'))[:16]}: {msg}")
                checked["ctx_records"] += 1
    if man is not None and counts:
        _verify_counts(out, cur, man, fail, checked, decode_heldout, guard)
        _verify_code(cur, man, fail)
    rehashed = 0
    if man is not None and rehash_sources:
        for p in PARTITIONS:
            if p == "heldout" and not decode_heldout:
                continue
            for r in read_ctx_records(out, cur, p):
                for name, pin in (r.get("sources") or {}).items():
                    if not isinstance(pin, dict) or pin.get("layout") not in ("file_prefix", "ctx_lines"):
                        continue
                    guard.tick()
                    try:
                        got = P._rehash(pin, r["context_sha256"])
                    except OSError as err:
                        got = f"error:{err}"
                    rehashed += 1
                    if got != pin.get("sha256"):
                        fail(f"context {r['context_sha256'][:16]} source {name}: sha256 now {str(got)[:24]} != recorded")
    poison = None
    if man is not None and poison_test_run:
        try:
            poison = poison_test(out, poison_work, poison_contexts=poison_contexts, guard=guard)
        except S.CorpusStop as err:
            poison = {"schema": POISON_SCHEMA, "verdict": "FAIL", "error": {"kind": _kind(err), "message": str(err)[:300]}}
        if poison.get("verdict") != "PASS":
            fail(f"poison test {poison.get('verdict')}: {poison.get('failures') or poison.get('error')}")
    doc = {"schema": VERIFY_SCHEMA, "out": str(out),
           "manifest_sha256": common.sha256_file(out / "manifest.json") if man is not None else None,
           "verdict": "PASS" if not fail.n else "FAIL", "failure_count": fail.n, "failures": fail.items[:100],
           "checks": {"schema_allowlist": schema_allowlist, "counts": counts, "rehash_sources": rehash_sources,
                      "poison_test": poison_test_run, "role": role},
           "checked": {**dict(sorted(checked.items())), "sources_rehashed": rehashed},
           "poison": poison, "guard": guard.summary(), "utc_us": common.utc_us()}
    if receipt_path is not None:
        common.write_receipt(receipt_path, doc)
    return doc


def _verify_code(cur: Mapping, man: Mapping, fail: _Fails) -> None:
    """Every invocation ran the same code, the code the cursor pins (red-team CL-6)."""
    codes = [inv.get("code") for inv in cur["ledger"]["invocations"]]
    if not codes or any(c is None for c in codes):
        fail("an invocation lacks its code sha256s")
    elif any(c != codes[0] for c in codes[1:]):
        fail("the build's invocations ran different code (a resume across a code change)")
    elif (man.get("pins") or {}).get("code") != codes[0] or (cur.get("pins") or {}).get("code") != codes[0]:
        fail("pins.code differs from the code the invocations ran")


def _sealed_ref_ok(out: Path, ref: Any, fail: _Fails, what: str) -> bool:
    """A sealed document referenced by sha256 from the top manifest: bytes checked, never decoded."""
    if not isinstance(ref, Mapping) or not (Path(out) / str(ref.get("path"))).is_file():
        fail(f"{what}: missing")
        return False
    if common.sha256_file(Path(out) / ref["path"]) != ref.get("sha256"):
        fail(f"{what}: sha256 differs from the manifest reference")
        return False
    return True


def _verify_counts(out: Path, cur: Mapping, man: Mapping, fail: _Fails, checked: Counter, decode_heldout: bool,
                   guard) -> None:
    """R6 and totals from re-read shards; label-stats, seed-stats and runtime-report recomputed and byte-compared;
    every ctx-record's exclusion is listed (no stopped context), every processed context holds the expected completed
    native rows (1,280 for the real corpus) and its inventory count, the manifest exclusions are listed reasons and the
    tolerance receipt equals its recomputation (red-team CL-3); heldout without a role: sha, bytes and footer row
    counts only, and the sealed heldout summary's sha256."""
    totals: Counter = Counter()
    decoded: list[str] = []
    inputs = man.get("inputs") or {}
    expected = inputs.get("expected_completed")
    real = Path(os.path.realpath(out)) == Path(os.path.realpath(CORPUS_OUT))
    if real and expected != S.EXPECT_COMPLETED_PER_FOLD:
        fail(f"the real corpus was built with expected_completed {expected!r}, not {S.EXPECT_COMPLETED_PER_FOLD}")
    inventory = None
    inv_pin = inputs.get("inventory")
    if inv_pin:
        try:
            inventory = load_inventory(Path(inv_pin["path"]), {"inventory": {"sha256": inv_pin["sha256"]}})[0]
        except (OSError, ValueError, KeyError, S.CorpusStop) as err:
            fail(f"the pinned inventory cannot be re-read: {str(err)[:200]}")
    elif real:
        fail("the real corpus was built without the sealed inventory")
    unlisted = sorted(k for k in (man.get("exclusions") or {}) if k not in LISTED_EXCLUSIONS)
    if unlisted:
        fail(f"manifest exclusions outside the listed reasons {LISTED_EXCLUSIONS}: {unlisted}")
    if int(cur.get("stop_count") or 0):
        fail(f"{cur.get('stop_count')} contexts stopped (--continue-on-stop): a STOP is never a listed exclusion")
    sealed = man.get("sealed") or {}
    held_refs = sealed.get("heldout") or {}
    for name in ("partition", "label_stats", "summary", "provenance", "exposed", "lineage"):
        _sealed_ref_ok(out, held_refs.get(name), fail, f"sealed heldout {name}")
    ctx_by_part: dict[str, list] = {}
    for p in PARTITIONS:
        if p == "heldout" and not decode_heldout:
            for t in ("records", "contexts", "refused", "ctx-records", "lineage", "exposed"):
                for sh in cur["writers"][p][t]["shards"]:
                    if _check_file(out, sh, fail) and t != "ctx-records":
                        if pq.ParquetFile(out / sh["path"]).metadata.num_rows != sh["rows"]:
                            fail(f"{sh['path']}: footer rows differ from the cursor")
                    checked["heldout_footer_files"] += 1
            continue
        decoded.append(p)
        guard.tick()
        recs = read_ctx_records(out, cur, p)
        ctx_by_part[p] = recs
        if len(recs) != cur["members"][p]:
            fail(f"{p}: {len(recs)} ctx-records != {cur['members'][p]} contexts")
        for r in recs:
            cid = str(r.get("context_sha256"))
            if r.get("stopped"):
                fail(f"{p}: context {cid[:16]} stopped ({(r.get('stopped') or {}).get('kind')}), which is not a "
                     f"listed exclusion")
                continue
            if r.get("excluded"):
                if r["excluded"] not in PACKET_EXCLUSIONS:
                    fail(f"{p}: context {cid[:16]} excluded for an unlisted reason")
                continue
            got = (r.get("counts") or {}).get("native_completed")
            want_n = expected.get(cid) if isinstance(expected, Mapping) else expected
            if got != want_n:
                fail(f"{p}: context {cid[:16]} holds {got} completed native rows, not {want_n}")
            if inventory is not None and inventory.get(cid) != got:
                fail(f"{p}: context {cid[:16]} native completed {got} != the sealed inventory {inventory.get(cid)}")
        want = {r["context_sha256"]: r for r in recs}
        got_rec, got_ref = Counter(), Counter()
        out_by: dict[str, Counter] = defaultdict(Counter)
        prev = None
        for sh in cur["writers"][p]["records"]["shards"]:
            if not _check_file(out, sh, fail):
                continue
            t = pq.read_table(out / sh["path"], columns=["context_sha256", "program_sha256", "split_role", "outcome"],
                              use_threads=False)
            for c, prog, rl, o in zip(*(t.column(n).to_pylist() for n in t.column_names)):
                if prev is not None and (c, prog) <= prev:
                    fail(f"{sh['path']}: rows are not strictly ordered by (context, program)")
                prev = (c, prog)
                if rl != p:
                    fail(f"{sh['path']}: a {rl} row in partition {p}")
                got_rec[c] += 1
                out_by[c][o] += 1
            checked["records_rows"] += t.num_rows
        for sh in cur["writers"][p]["refused"]["shards"]:
            if _check_file(out, sh, fail):
                t = pq.read_table(out / sh["path"], columns=["context_sha256", "split_role"], use_threads=False)
                for c, rl in zip(t.column(0).to_pylist(), t.column(1).to_pylist()):
                    got_ref[c] += 1
                    if rl != p:
                        fail(f"{sh['path']}: a {rl} refused row in partition {p}")
        ctx_rows = contexts_rows(out, cur, p)
        for cid, r in want.items():
            if r.get("excluded") or r.get("stopped"):
                if got_rec[cid] or got_ref[cid] or cid in ctx_rows:
                    fail(f"{p}: excluded context {cid[:16]} has rows")
                continue
            if got_rec[cid] != r["records"] or got_ref[cid] != r["refused"]:
                fail(f"{p}: context {cid[:16]} rows {got_rec[cid]}+{got_ref[cid]} != ctx-record "
                     f"{r['records']}+{r['refused']}")
            if r["records"] + r["refused"] != r["summary"]["unique_programs"]:
                fail(f"{p}: context {cid[:16]} records + refused != unique_programs")
            row = ctx_rows.get(cid)
            if row is None or (row["n_records"], row["n_refused"], row["split_role"]) != (r["records"], r["refused"], p):
                fail(f"{p}: the context row of {cid[:16]} disagrees with its ctx-record")
            if dict(out_by[cid]) != {k: v for k, v in r["counts"]["by_outcome"].items() if v}:
                fail(f"{p}: context {cid[:16]} outcome counts differ from its ctx-record")
            totals["records"] += r["records"]
            totals["refused"] += r["refused"]
            if p == "heldout":
                totals["heldout_records"] += r["records"]
                totals["heldout_refused"] += r["refused"]
        if set(got_rec) - set(want) or set(ctx_rows) - set(want):
            fail(f"{p}: rows of contexts outside the partition")
        stats = label_stats_from_shards(out, cur, p, None, guard=guard)
        if (part_dir(out, p) / "label-stats.json").read_bytes() != _canon(stats["label_stats"]):
            fail(f"{p}: label-stats.json differs from its recomputation")
        checked["label_stats"] += 1
    try:
        seed_set = P._load_seeds(man["inputs"]["seeds"]["path"])[0]
        st = label_stats_from_shards(out, cur, "train_fit", seed_set, guard=guard)["seed_stats"]
        doc = seed_stats_doc(st, man["inputs"]["seeds"]["sha256"], len(seed_set))
        if (part_dir(out, "train_fit") / "seed-stats.json").read_bytes() != _canon(doc):
            fail("seed-stats.json differs from its recomputation")
        checked["seed_stats"] += 1
    except (OSError, P.ReconcileError) as err:
        fail(f"seed-stats.json cannot be recomputed: {err}")
    if (out / "runtime-report.json").read_bytes() != _canon(runtime_report_from_shards(out, cur, guard=guard)):
        fail("runtime-report.json differs from its recomputation")
    checked["runtime_report"] += 1
    for p in PARTITIONS:
        for sh in cur["writers"][p]["exposed"]["shards"]:
            if _check_file(out, sh, fail) and pq.ParquetFile(out / sh["path"]).metadata.num_rows != sh["rows"]:
                fail(f"{sh['path']}: footer rows differ from the cursor")
            checked["exposed_footer_files"] += 1
    by_part = (man.get("totals") or {}).get("by_partition") or {}
    if "heldout" in by_part:
        fail("the non-sealed manifest carries heldout totals")
    want = [by_part.get(p, {"records": 0, "refused": 0}) for p in decoded if p != "heldout"]
    open_rec = totals["records"] - totals["heldout_records"]
    open_ref = totals["refused"] - totals["heldout_refused"]
    if sum(d["records"] for d in want) != open_rec or sum(d["refused"] for d in want) != open_ref:
        fail("manifest totals differ from the re-read shards")
    if man.get("tolerances") != cv_sd_tolerances(ctx_by_part, [p for p in OPEN_PARTITIONS if p in ctx_by_part]):
        fail("the manifest's cv_log_ratio_sd tolerance receipt differs from the ctx-records")
    if decode_heldout and _sealed_ref_ok(out, held_refs.get("summary"), fail, "sealed heldout summary"):
        hsum = json.loads((out / held_refs["summary"]["path"]).read_bytes())
        held = (hsum.get("totals") or {}).get("by_partition", {}).get("heldout", {"records": 0, "refused": 0})
        if (held["records"], held["refused"]) != (totals["heldout_records"], totals["heldout_refused"]):
            fail("the heldout summary's totals differ from the re-read heldout shards")
        if hsum.get("tolerances") != cv_sd_tolerances(ctx_by_part, ("heldout",)):
            fail("the heldout summary's tolerance receipt differs from the heldout ctx-records")


# ------------------------------------------------------------------------------------------- poison test

POISON_SCALE = 0.37
POISON_XOR = 0x5A
POISON_WALL_DELTA = 7_000_000          # µs added to the outer phase and the whole evaluation (official_wall control)
# Sensitivity controls, each of which must make the verdict FAIL: a train record's inner rmse; a train context's JTT1
# train rows; a train context's outer-phase wall time with its whole-evaluation wall_us (the declared runtime
# coupling, corpus_schema.RUNTIME_POLICY_NOTE); and a code-level leak of heldout, dev or diag labels into the train
# label statistics (red-team LA-1 negative controls).
POISON_SENSITIVITY = ("train_rmse", "train_jtt1", "official_wall", "leak_heldout", "leak_dev", "leak_diag")


def _perturb_scalar(v: Any, salt: str) -> Any:
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v * 3 + 7
    if isinstance(v, float):
        return v * 1.37 + 0.11 if math.isfinite(v) else v
    if isinstance(v, str):
        if _is_hex64(v):
            return _sha((salt + v).encode())
        try:
            raw = base64.b64decode(v.encode("ascii"), validate=True) if len(v) >= 16 else b""
        except (binascii.Error, ValueError, UnicodeEncodeError):
            raw = b""
        if raw and len(raw) % 4 == 0:
            with np.errstate(all="ignore"):           # real vectors may hold inf/NaN or values near float32 max
                vals = (np.frombuffer(raw, dtype="<f4").astype(np.float64) * 1.37 + 0.11).astype("<f4")
            return base64.b64encode(vals.tobytes()).decode("ascii")
        return "P5-POISON-" + v
    return v


def _perturb_tree(v: Any, salt: str, path: tuple = (), keep: frozenset = frozenset()) -> Any:
    if path in keep:
        return v
    if isinstance(v, dict):
        return {k: _perturb_tree(x, salt, path + (k,), keep) for k, x in v.items()}
    if isinstance(v, list):
        return [_perturb_tree(x, salt, path, keep) for x in v]
    return _perturb_scalar(v, salt)


def poison_exposed_fields(row: Any, salt: str, keep: frozenset = frozenset(), path: tuple = ()) -> Any:
    """A copy of one source document with every exposed field perturbed: any key that corpus_schema.name_is_exposed
    names (the official outer-test block, outer-test RMSEs and ratios, the TabPFN references, both prediction
    vectors, the selection and duplicate flags, result pointers and the summary's best and improvement values),
    recursively, keys and nesting unchanged. Values at a key path in `keep` stay as they are (the outer phase's
    timing values under corpus_schema.RUNTIME_POLICY, so the perturbed world stays possible)."""
    if path in keep:
        return row
    if isinstance(row, dict):
        return {k: (_perturb_tree(v, salt, path + (k,), keep) if S.name_is_exposed(str(k))
                    else poison_exposed_fields(v, salt, keep, path + (k,)))
                for k, v in row.items()}
    if isinstance(row, list):
        return [poison_exposed_fields(x, salt, keep, path) for x in row]
    return row


def packer_stats(native: bytes, outcomes: bytes, transports: bytes) -> dict:
    """pack_fold_history.build_history's summary counts over raw prefixes (the poison harness re-pins with them)."""
    progs = set()
    for line in outcomes.splitlines():
        r = json.loads(line)
        if r.get("program"):
            progs.add(r["program_sha256"])
    for line in native.splitlines():
        d = (json.loads(line).get("program") or {}).get("program_sha256")
        if d:
            progs.add(d)
    for line in transports.splitlines():
        r = json.loads(line)
        if r.get("program"):
            progs.add(r["program_sha256"])
    return {"unique_programs": len(progs), "native_events": len(native.splitlines()), "native_bytes": len(native),
            "outcome_bytes": len(outcomes), "outcome_events": len(outcomes.splitlines()),
            "transport_bytes": len(transports), "transport_events": len(transports.splitlines())}


def _dumps(row: Mapping) -> bytes:
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


def _read_prefix(path: Path, n: int) -> bytes:
    with open(path, "rb") as f:
        data = f.read(n)
    if len(data) != n:
        raise _stop("source", f"{path} holds fewer than {n} bytes")
    return data


def _gate_of(row: Mapping) -> bool | None:
    """labels' gate for an outcome row (½ Σ r_j/b_j <= 1.05 with detail, else cv_mean_rmse_ratio <= 1.05)."""
    rmse, base = S.inner_cv_pairs(row.get("evaluation_detail"))
    if rmse is not None and base is not None and None not in rmse and None not in base and all(b > 0 for b in base):
        return 0.5 * (rmse[0] / base[0] + rmse[1] / base[1]) <= labels.GATE_RATIO
    ratio = row.get("cv_mean_rmse_ratio")
    return None if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) else ratio <= labels.GATE_RATIO


def _swap_node(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    if v.startswith("xbabe"):
        return "aws:us-east-2:p5-poison"
    if v.startswith("aws:"):
        return "xbabe1"
    return v


def _swap_pool(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    if v.startswith("xbabe"):
        return "aws:us-east-2"
    if v.startswith("aws:"):
        return "xbabe1"
    return v


def _scale_inner_rmse(row: dict) -> None:
    detail = row.get("evaluation_detail")
    for e in (detail.get("inner_cv") or []) if isinstance(detail, dict) else []:
        m = e.get("metrics") if isinstance(e, dict) else None
        if isinstance(m, dict) and isinstance(m.get("rmse"), (int, float)) and not isinstance(m.get("rmse"), bool):
            m["rmse"] = m["rmse"] * POISON_SCALE


def _recanon_sd(rows: list[dict]) -> None:
    """After a perturbation of inner-CV detail the writers' cv_log_ratio_sd follows it: the canonical value of the
    row's own detail (corpus_schema.canonical_cv_log_ratio_sd), so the counterfactual stays consistent with the
    canonical recompute check."""
    for r in rows:
        try:
            rmse, base = S.inner_cv_pairs(r.get("evaluation_detail"))
        except S.CorpusStop:
            continue
        canon = S.canonical_cv_log_ratio_sd(rmse, base)
        if canon is not None and isinstance(r.get("cv_log_ratio_sd"), float):
            r["cv_log_ratio_sd"] = canon


def poison_records(cid: str, natives: list[dict], outcomes: list[dict], transports: list[dict]) -> None:
    """Perturb one non-train context's records in place (PLAN §5 P5 poison test): rmse and cv values x0.37,
    completed <-> failed status flips in native and outcome rows (pairwise, so native completed counts hold), node and
    pool swaps between xbabe and aws, wall_us x2 and 2 extra (failed) programs. cv_gate_passed is re-derived with the
    labels arithmetic and cv_log_ratio_sd with the canonical formula (detailed rows), and the native rows follow
    their evolution rows, so the native/outcome agreement, the canonical recompute and the gate check still hold."""
    for r in natives:
        for k in ("cv_mean_log_ratio", "cv_log_ratio_sd"):
            if isinstance(r.get(k), float):
                r[k] = r[k] * POISON_SCALE
    for r in outcomes:
        for k in ("cv_mean_log_ratio", "cv_log_ratio_sd", "cv_mean_rmse_ratio"):
            if isinstance(r.get(k), float):
                r[k] = r[k] * POISON_SCALE
        _scale_inner_rmse(r)
        if isinstance(r.get("wall_us"), int) and not isinstance(r.get("wall_us"), bool):
            r["wall_us"] = r["wall_us"] * 2
        if "node" in r:
            r["node"] = _swap_node(r["node"])
    for r in transports:
        if "pool" in r:
            r["pool"] = _swap_pool(r["pool"])
    last_evo: dict[str, dict] = {}
    for r in outcomes:
        if r.get("phase") == "evolution":
            last_evo[r.get("program_sha256")] = r
    done = [r for r in natives if r.get("status") == "completed"]
    failed = [r for r in natives if r.get("status") == "failed"]
    for a, b in zip(done, failed):
        sa, sb = (a.get("program") or {}).get("program_sha256"), (b.get("program") or {}).get("program_sha256")
        if last_evo.get(sa) is None or last_evo.get(sb) is None:
            continue
        a["status"], b["status"] = "failed", "completed"
        last_evo[sa]["record_status"], last_evo[sb]["record_status"] = "failed", "completed"
    for r in outcomes:
        if r.get("record_status") == "completed":
            r["cv_gate_passed"] = _gate_of(r)
    _recanon_sd(outcomes)
    for r in natives:
        evo = last_evo.get((r.get("program") or {}).get("program_sha256"))
        if evo is None:
            continue
        if r.get("status") == "completed":
            r["cv_gate_passed"] = evo.get("cv_gate_passed")
            for k in ("cv_mean_log_ratio", "cv_log_ratio_sd"):
                r[k] = evo.get(k)
        elif r.get("cv_log_ratio_sd") is not None or evo.get("cv_log_ratio_sd") is not None:
            r["cv_log_ratio_sd"] = evo.get("cv_log_ratio_sd")
    template = next((r for r in outcomes if r.get("phase") == "evolution"), None) or (outcomes[0] if outcomes else {})
    tag = int(cid[:6], 16) % 900 + 100
    for k in range(2):
        dsl = f"(pipe X (winsor_clip q=0.0{k + 1}) (ridge lambda_frac=0.{tag}{k}7))"
        prog = {"birth_seq": 990000 + k, "dsl": dsl, "operator": "p5_poison_extra", "parents": [],
                "program_sha256": _sha(dsl.encode())}
        natives.append({"context_sha256": cid, "island_id": 0, "policy": "p5_poison", "program": prog,
                        "proposal_draw": 0, "proposal_seq": 0, "schema": "jope.island-scored-evaluation/v1",
                        "status": "failed"})
        outcomes.append({"schema": template.get("schema"), "view_id": template.get("view_id"),
                         "dataset_hash": template.get("dataset_hash"), "outer_fold": template.get("outer_fold"),
                         "context_sha256": cid, "phase": "evolution", "generation": 99,
                         "program_sha256": prog["program_sha256"], "record_status": "failed",
                         "transport_status": "completed_transport", "wall_us": 12345 + k, "program": prog})


def poison_train_rmse(natives: list[dict], outcomes: list[dict]) -> bool:
    """Sensitivity control: scale one train record's inner-CV rmse (its last detailed completed evolution row) and
    re-derive its gate and its canonical cv_log_ratio_sd on both sides; True when a row was changed."""
    last: dict[str, dict] = {}
    for r in outcomes:
        if r.get("phase") == "evolution":
            last[r.get("program_sha256")] = r
    for r in outcomes:
        if (r.get("phase") == "evolution" and r.get("record_status") == "completed" and last.get(r.get("program_sha256")) is r
                and S.inner_cv_pairs(r.get("evaluation_detail"))[0] is not None):
            _scale_inner_rmse(r)
            r["cv_gate_passed"] = _gate_of(r)
            _recanon_sd([r])
            for n in natives:
                if (n.get("program") or {}).get("program_sha256") == r.get("program_sha256"):
                    n["cv_gate_passed"] = r["cv_gate_passed"]
                    n["cv_log_ratio_sd"] = r.get("cv_log_ratio_sd")
            return True
    return False


def jtt1_variant(raw: bytes, *, xor_test: bool, train_affine: bool) -> bytes:
    """A JTT1 copy: XOR every outer-test byte with 0x5A and/or replace the outer-train rows by X*0.37, y*1.37+0.5
    (an affine change that keeps the leak status of every column)."""
    h = jtt1.parse_header(raw)
    start, stop = jtt1.train_range(h)
    buf = bytearray(raw)
    if train_affine:
        block = np.frombuffer(bytes(raw[start:stop]), dtype="<f8").reshape(h["n_train"], h["p"] + 1).astype(np.float64)
        block[:, :-1] *= POISON_SCALE
        block[:, -1] = 1.37 * block[:, -1] + 0.5
        buf[start:stop] = block.astype("<f8").tobytes()
    if xor_test and stop < len(buf):
        buf[stop:] = (np.frombuffer(bytes(buf[stop:]), dtype=np.uint8) ^ POISON_XOR).tobytes()
    return bytes(buf)


def _files_under(root: Path, rels: Sequence[str]) -> dict[str, bytes]:
    out = {}
    for rel in rels:
        base = root / rel
        if base.is_file():
            out[rel] = base.read_bytes()
        elif base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    out[str(p.relative_to(root))] = p.read_bytes()
    return out


def _repin_reuse(rows: list[dict], old_raw_sha256: str | None, new_raw_sha256: str) -> int:
    """In a perturbed copy whose JTT1 input changed, an evaluation-reuse identity that named the original input now
    names the copy's (the counterfactual stays consistent with corpus_schema.check_evaluation_reuse). Returns the
    number of rows re-pinned."""
    n = 0
    for r in rows:
        reuse = r.get("evaluation_reuse")
        ident = reuse.get("compatibility_identity") if isinstance(reuse, dict) else None
        if isinstance(ident, dict) and old_raw_sha256 is not None and ident.get("input_raw_sha256") == old_raw_sha256:
            ident["input_raw_sha256"] = new_raw_sha256
            n += 1
    return n


def poison_official_wall(outcomes: list[dict]) -> int:
    """Sensitivity control (official_wall): shift the outer phase's wall time and the whole-evaluation wall_us of every
    outcome row that carries both by POISON_WALL_DELTA, a consistent counterfactual. Under RUNTIME_POLICY
    'whole_evaluation' the train records' wall_us then change, so the comparison must FAIL (the declared coupling)."""
    return sum(1 for r in outcomes if S.project_official_wall_shift_exposed(r, POISON_WALL_DELTA))


@contextlib.contextmanager
def _leaky_train_stats(partition: str):
    """Negative control (red-team LA-1): a deliberate code-level leak that folds `partition`'s label statistics into
    train/label-stats.json while it is active; the poison comparison must then FAIL."""
    g = globals()
    real = g["label_stats_from_shards"]

    def leaky(out, cursor, p, seeds, **kw):
        doc = real(out, cursor, p, seeds, **kw)
        if p == "train_fit":
            doc["label_stats"] = {**doc["label_stats"], "leaked_partition": partition,
                                  "leaked_label_stats": real(out, cursor, partition, None, **kw)["label_stats"]}
        return doc

    g["label_stats_from_shards"] = leaky
    try:
        yield
    finally:
        g["label_stats_from_shards"] = real


class _PoisonHarness:
    """PLAN §5 P5 poison test over copies of `per_cohort` contexts per cohort of a built corpus (spec §3.10), every
    non-empty non-train partition (heldout, dev and diag) perturbed at least once (red-team LA-1)."""

    def __init__(self, out: Path, work: Path, per_cohort: int, sensitivity: str | None, guard):
        self.out, self.work, self.per_cohort, self.sensitivity, self.guard = out, work, per_cohort, sensitivity, guard
        self.man = json.loads((out / "manifest.json").read_bytes())
        self.cur = json.loads((out / "cursor.json").read_bytes())
        ins = self.man["inputs"]
        self.epoch_doc = json.loads(Path(ins["epoch"]["path"]).read_bytes())
        if _sha(Path(ins["epoch"]["path"]).read_bytes()) != ins["epoch"]["sha256"]:
            raise _stop("input", "the epoch file no longer matches the built corpus")
        self.entries = {e["context_sha256"]: e for e in self.epoch_doc["contexts"]}
        self.pmap = json.loads(Path(ins["partition_map"]["path"]).read_bytes())
        self.sources = P.Sources(Path(ins["sources"]["incoming_cursor"]), Path(ins["sources"]["incoming_transport_index"]))
        self.exposed = bool(self.man["settings"]["exposed_flags_recompute"])
        self.plan = P._plan(Path(ins["epoch"]["path"]), Path(ins["partition_map"]["path"]))
        self.inc_raw: dict | None = None

    # ------------------------------------------------------------ selection
    def blocked(self) -> set:
        """Contexts that hold no rows (excluded as target_leak, or stopped): train_fit, dev and diag from their
        ctx-records, heldout from the sealed heldout summary's flags (the harness copies heldout sources anyway)."""
        out = set()
        for p in OPEN_PARTITIONS:
            out |= {r["context_sha256"] for r in read_ctx_records(self.out, self.cur, p)
                    if r.get("excluded") or r.get("stopped")}
        summary = heldout_summary_path(self.out)
        if summary.is_file():
            out |= set(json.loads(summary.read_bytes()).get("context_flags") or {})
        return out

    def select(self) -> list:
        """Per cohort, in sha order and skipping contexts that hold no rows: the first (per_cohort + 1) // 2 train_fit
        contexts, then max(per_cohort // 2, the number of non-empty non-train partitions of the cohort) non-train
        contexts taken round-robin over heldout, dev and diag, so every non-empty partition is perturbed at least once
        (red-team LA-1)."""
        members = {p: [c for c in self.plan.contexts if c.split_role == p] for p in PARTITIONS}
        blocked = self.blocked()
        n_train = (self.per_cohort + 1) // 2
        picked = []
        for cohort in COHORTS:
            picked += [c for c in members["train_fit"]
                       if c.cohort == cohort and c.context_sha256 not in blocked][:n_train]
            avail = {p: [c for c in members[p] if c.cohort == cohort and c.context_sha256 not in blocked]
                     for p in NONTRAIN_PREFERENCE}
            n_non = max(self.per_cohort // 2, sum(1 for p in NONTRAIN_PREFERENCE if avail[p]))
            chosen, depth = [], 0
            while len(chosen) < n_non and any(depth < len(avail[p]) for p in NONTRAIN_PREFERENCE):
                for p in NONTRAIN_PREFERENCE:
                    if depth < len(avail[p]) and len(chosen) < n_non:
                        chosen.append(avail[p][depth])
                depth += 1
            picked += chosen
        return picked

    def coverage(self, selected: Sequence) -> tuple[list, list]:
        """(non-empty non-train partitions of the build, the non-train partitions the selection perturbs)."""
        blocked = self.blocked()
        nonempty = [p for p in NONTRAIN_PREFERENCE
                    if any(c.split_role == p and c.context_sha256 not in blocked for c in self.plan.contexts)]
        perturbed = [p for p in NONTRAIN_PREFERENCE if any(c.split_role == p for c in selected)]
        return nonempty, perturbed

    # ------------------------------------------------------------ sources of one context
    def _raw_cursor_entry(self, cid: str) -> dict:
        if self.inc_raw is None:
            self.inc_raw = json.loads(self.sources.incoming_cursor.read_bytes())
        e = self.inc_raw.get(cid)
        if not isinstance(e, dict):
            raise _stop("source", "context is absent from the incoming cursor", cid)
        if entry_sha256(P._project_cursor_entry(e)) != self.cur["incoming_entries"].get(cid):
            raise _stop("source", "the incoming cursor entry changed since the build", cid)
        return e

    def bundle(self, c) -> dict:
        cid, entry = c.context_sha256, self.entries[c.context_sha256]
        raw_entry = self._raw_cursor_entry(cid) if c.cohort == "incoming" else None
        try:
            s = P._summary(c, {cid: P._project_cursor_entry(raw_entry)} if raw_entry is not None else None)[0]
        except P.ReconcileError as err:
            raise _from_reconcile(err) from None
        d = P._ctx_dir(c)
        native = _read_prefix(Path(c.native["path"]), c.native["bytes"])
        if _sha(native) != c.native["sha256"]:
            raise _stop("source", "native prefix differs from its epoch pin", cid)
        b = {"c": c, "entry": entry, "summary": s, "native": native, "raw_entry": raw_entry,
             "seed_pool": P._snapshot(d / "seed-pool.jsonl"),
             "prepared": Path(entry["prepared_ref"]["path"]).read_bytes(),
             # both cohorts: P4a cross-checks the bootstrap-results program set of every context
             "boot": P._snapshot(d / "bootstrap-results.jsonl")}
        if c.cohort == "frozen":
            b["outcome"] = _read_prefix(d / "history-outcomes.jsonl", s.outcome_bytes) if s.outcome_bytes else b""
            b["transport"] = (_read_prefix(d / "history-transports.jsonl", s.transport_bytes)
                              if s.transport_bytes else b"")
            b["history_ref"] = json.loads((d / "history-ref.json").read_bytes())
        else:
            idx = ItemIndex(P._item_gp(c) / "hwm-events.jsonl", {cid: s.outcome_bytes})
            b["outcome"] = b"".join(idx.lines(cid))
            tpath = P._transport_path(c, self.sources)
            b["transport"] = _read_prefix(tpath, s.transport_bytes) if s.transport_bytes else b""
        b["qualified"] = self._qualified_entry(c)
        return b

    def _qualified_entry(self, c) -> Any:
        if not self.exposed:
            return None
        if c.cohort == "frozen":
            fq = self.man["inputs"]["sources"].get("frozen_qualified_views")
            path = Path(fq) if fq else P._ctx_dir(c).parent.parent / "qualified-views.json"
            if not path.exists():
                return None
            return json.loads(path.read_bytes()).get(c.view_id)
        path = P._item_gp(c) / "qualification.json"
        if not path.exists():
            return None
        doc = json.loads(path.read_bytes())
        return next((q for q in doc.get("qualified") or [] if isinstance(q, dict) and q.get("view_id") == c.view_id), None)

    # ------------------------------------------------------------ one variant
    def write_variant(self, name: str, bundles: Sequence[dict], train_ids: set[str]) -> dict:
        root = self.work / name
        src = root / "src"
        cache = Path(self.man["inputs"]["jtt1_cache"]) if name == "A" else root / "jtt1-cache"
        poison = name == "B"
        frozen_gp = src / "frozen" / "gp"
        inc_root = src / "incoming"
        tindex = inc_root / "transport-index"
        tindex.mkdir(parents=True, exist_ok=True)
        cursor_doc, frozen_q, items_q, items_lines = {}, {}, defaultdict(list), defaultdict(list)
        entries, refs = [], {}
        keep = S.project_timing_keep_exposed()
        target = min(train_ids) if train_ids else None
        for b in bundles:
            c, s = b["c"], b["summary"]
            cid = c.context_sha256
            entry = copy.deepcopy(b["entry"])
            native, outcome, transport = b["native"], b["outcome"], b["transport"]
            seed_pool, prepared, boot = b["seed_pool"], b["prepared"], b["boot"]
            ref = dict(entry["input_ref"])
            if poison:
                # the JTT1 copy first: every outer-test byte XOR 0x5A, and the train rows too for non-train contexts
                raw = jtt1.load_raw(ref, Path(self.man["inputs"]["jtt1_cache"]))
                affine = cid not in train_ids or (self.sensitivity == "train_jtt1" and cid == target)
                new_raw = jtt1_variant(raw, xor_test=True, train_affine=affine)
                body = gzip.compress(new_raw, compresslevel=3, mtime=0)
                old_raw_sha256 = ref.get("raw_sha256")
                ref.update(raw_sha256=_sha(new_raw), raw_bytes=len(new_raw), sha256=_sha(body), bytes=len(body))
                P._publish_immutable(cache / f"{ref['raw_sha256']}.jtt1.gz", body)
                nat_rows = [poison_exposed_fields(json.loads(x), cid, keep) for x in native.splitlines()]
                out_rows = [poison_exposed_fields(json.loads(x), cid, keep) for x in outcome.splitlines()]
                tr_rows = [poison_exposed_fields(json.loads(x), cid, keep) for x in transport.splitlines()]
                _repin_reuse(out_rows, old_raw_sha256, ref["raw_sha256"])
                if cid not in train_ids:
                    poison_records(cid, nat_rows, out_rows, tr_rows)
                elif self.sensitivity == "train_rmse" and cid == target:
                    poison_train_rmse(nat_rows, out_rows)
                elif self.sensitivity == "official_wall" and cid == target:
                    poison_official_wall(out_rows)
                native = b"".join(_dumps(r) for r in nat_rows)
                outcome = b"".join(_dumps(r) for r in out_rows)
                transport = b"".join(_dumps(r) for r in tr_rows)
                # exposed fields of the side files too: seed-pool source evidence, bootstrap-results result
                # pointers, prepared.json's outer-test keys (P5 and P4a read none of them; their pins are provenance)
                seed_pool = b"".join(_dumps(poison_exposed_fields(json.loads(x), cid)) for x in seed_pool.splitlines()
                                     if x.strip())
                boot = b"".join(_dumps(poison_exposed_fields(json.loads(x), cid)) for x in boot.splitlines()
                                if x.strip())
                prepared = json.dumps(poison_exposed_fields(json.loads(prepared), cid), sort_keys=True).encode()
            stats = packer_stats(native, outcome, transport)
            if c.cohort == "frozen":
                d = frozen_gp / "contexts" / c.view_id
            else:
                gp = inc_root / "items" / c.item_id / "run" / "gp"
                d = gp / "contexts" / c.view_id
            d.mkdir(parents=True, exist_ok=True)
            npath = d / f"evaluations-{cid[:16]}.jsonl"
            npath.write_bytes(native)
            (d / "seed-pool.jsonl").write_bytes(seed_pool)
            (d / "prepared.json").write_bytes(prepared)
            (d / "bootstrap-results.jsonl").write_bytes(boot)
            if c.cohort == "frozen":
                (d / "history-outcomes.jsonl").write_bytes(outcome)
                (d / "history-transports.jsonl").write_bytes(transport)
                href = {**b["history_ref"], **stats}
                if poison:
                    href = poison_exposed_fields(href, cid)
                    href.update(stats)
                (d / "history-ref.json").write_bytes(json.dumps(href, sort_keys=True).encode())
                if b["qualified"] is not None:
                    frozen_q[c.view_id] = poison_exposed_fields(b["qualified"], cid) if poison else b["qualified"]
            else:
                (tindex / f"{cid}.jsonl").write_bytes(transport)
                items_lines[c.item_id].append(outcome)
                raw_entry = copy.deepcopy(b["raw_entry"])
                if poison:
                    raw_entry = poison_exposed_fields(raw_entry, cid)
                raw_entry.setdefault("stats", {}).update(stats)
                raw_entry["fingerprint"] = _sha(native + outcome + transport + boot)
                raw_entry["generation_zero_source_sha256"] = _sha(boot)
                cursor_doc[cid] = raw_entry
                if b["qualified"] is not None:
                    items_q[c.item_id].append(poison_exposed_fields(b["qualified"], cid) if poison else b["qualified"])
            refs[cid] = ref
            entry.update(native_source={"path": str(npath), "bytes": len(native), "sha256": _sha(native)},
                         prepared_ref={"path": str(d / "prepared.json"), "sha256": _sha(prepared),
                                       "bytes": len(prepared)},
                         input_ref=ref)
            entries.append(entry)
        for item, chunks in items_lines.items():
            gp = inc_root / "items" / item / "run" / "gp"
            (gp / "hwm-events.jsonl").write_bytes(b"".join(chunks))
            if items_q.get(item):
                (gp / "qualification.json").write_bytes(json.dumps({"failures": [], "qualified": items_q[item]},
                                                                   sort_keys=True).encode())
        (inc_root / "fold-archive-cursor-v2.json").write_bytes(json.dumps(cursor_doc, sort_keys=True).encode())
        fq_path = None
        if frozen_q:
            frozen_gp.mkdir(parents=True, exist_ok=True)
            fq_path = frozen_gp / "qualified-views.json"
            fq_path.write_bytes(json.dumps(frozen_q, sort_keys=True).encode())
        epoch_doc = {k: v for k, v in self.epoch_doc.items() if k != "contexts"}
        epoch_doc["contexts"] = entries
        (root / "epoch.json").write_bytes(json.dumps(epoch_doc, sort_keys=True).encode())
        (root / "partition-map.json").write_bytes(json.dumps({e["context_sha256"]: self.pmap[e["context_sha256"]]
                                                              for e in entries}, sort_keys=True).encode())
        return {"root": root, "cache": cache, "refs": refs, "frozen_qualified": fq_path,
                "cursor": inc_root / "fold-archive-cursor-v2.json", "index": tindex}

    def build(self, v: dict) -> dict:
        ins, settings = self.man["inputs"], self.man["settings"]
        inv = ins.get("inventory")
        leak = self.sensitivity[len("leak_"):] if (self.sensitivity or "").startswith("leak_") else None
        with (_leaky_train_stats(leak) if leak else contextlib.nullcontext()):
            return run(epoch=v["root"] / "epoch.json", partition_map=v["root"] / "partition-map.json",
                       seeds=ins["seeds"]["path"], tokenizer=ins["tokenizer"]["dir"], out=v["root"] / "corpus",
                       jtt1_cache=v["cache"], budget_bytes=0, offline=True, resume=False,
                       incoming_cursor=v["cursor"], incoming_transport_index=v["index"],
                       frozen_qualified_views=v["frozen_qualified"], exposed=self.exposed,
                       expected_completed=ins.get("expected_completed"), inventory=inv["path"] if inv else None,
                       target_shard_bytes=settings["target_shard_bytes"], guard=self.guard, tokenizer_closure=False,
                       argv=["poison-test", v["root"].name])

    # ------------------------------------------------------------ comparisons
    def main_rows(self, table: str, ids: set[str]) -> list[dict]:
        rows = []
        for sh in self.cur["writers"]["train_fit"][table]["shards"]:
            if not any(sh["first_context"] <= c <= sh["last_context"] for c in ids):
                continue
            t = pq.read_table(self.out / sh["path"], use_threads=False)
            rows += [r for r in t.to_pylist() if r["context_sha256"] in ids]
        return rows

    def execute(self) -> dict:
        fails: list[str] = []
        selected = self.select()
        train_ids = {c.context_sha256 for c in selected if c.split_role == "train_fit"}
        nonempty, perturbed = self.coverage(selected)
        doc = {"schema": POISON_SCHEMA, "work": str(self.work), "per_cohort": self.per_cohort,
               "sensitivity": self.sensitivity, "runtime_policy": S.RUNTIME_POLICY,
               "selected": [{"context_sha256": c.context_sha256, "cohort": c.cohort, "split_role": c.split_role}
                            for c in selected],
               "nonempty_partitions": nonempty, "perturbed_partitions": perturbed}
        if not train_ids:
            doc.update(verdict="FAIL", failures=["no train_fit context to compare"])
            return doc
        missing = [p for p in nonempty if p not in perturbed]
        if missing:
            fails.append(f"non-train partitions never perturbed: {missing}")
        bundles = [self.bundle(c) for c in selected]
        builds, variants = {}, {}
        for name in ("A", "B"):
            self.guard.wait()
            variants[name] = self.write_variant(name, bundles, train_ids)
            builds[name] = res = self.build(variants[name])
            if res["exit"] != EXIT_OK:
                fails.append(f"build {name}: {res['status']} {res.get('stop')}")
        doc["builds"] = {k: {"status": v["status"], "exit": v["exit"], "stop": v.get("stop")} for k, v in builds.items()}
        if any(v["exit"] != EXIT_OK for v in builds.values()):
            doc.update(verdict="FAIL", failures=fails)
            return doc
        a_out, b_out = variants["A"]["root"] / "corpus", variants["B"]["root"] / "corpus"
        # the whole train/ tree: shards, label and seed statistics and the train closure (train/manifest.json)
        fa, fb = _files_under(a_out, ("train",)), _files_under(b_out, ("train",))
        doc["train_files"] = {k: {"A": _sha(fa[k]) if k in fa else None, "B": _sha(fb[k]) if k in fb else None}
                              for k in sorted(set(fa) | set(fb))}
        doc["train_identical"] = bool(fa) and fa == fb
        doc["train_closure"] = {k: train_closure_ref(o)["sha256"] for k, o in (("A", a_out), ("B", b_out))}
        if not doc["train_identical"]:
            fails.append("train outputs differ between A and B")
        ra, rb = ([g for g in json.loads((o / "runtime-report.json").read_bytes())["groups"]
                   if g["partition"] == "train_fit"] for o in (a_out, b_out))
        doc["runtime_train_identical"] = bool(ra) and ra == rb
        if not doc["runtime_train_identical"]:
            fails.append("runtime-report train_fit groups differ between A and B")
        exp_rel = ("exposed", "heldout/exposed")
        non_rel = ("eval", "heldout/records", "heldout/contexts", "heldout/refused")
        ea, eb = _files_under(a_out, exp_rel), _files_under(b_out, exp_rel)
        na, nb = _files_under(a_out, non_rel), _files_under(b_out, non_rel)
        ja, jb = variants["A"]["refs"], variants["B"]["refs"]
        controls = {"exposed_differ": bool(ea) and ea != eb, "nontrain_differ": bool(na) and na != nb,
                    "jtt1_differ": all(ja[c]["raw_sha256"] != jb[c]["raw_sha256"] for c in ja)}
        doc["controls"] = controls
        for k, ok in controls.items():
            if not ok:
                fails.append(f"control {k} is false (the harness is insensitive)")
        neutral = {}
        for table in ("records", "contexts"):
            mine = self.main_rows(table, train_ids)
            a_rows = []
            for f in sorted((a_out / "train" / table).glob("*.parquet")):
                a_rows += pq.read_table(f, use_threads=False).to_pylist()
            neutral[f"{table}_equal"] = bool(mine) and mine == a_rows
            if not neutral[f"{table}_equal"]:
                fails.append(f"harness neutrality: A's train {table} differ from the main build's rows")
        doc["neutrality"] = neutral
        doc.update(verdict="PASS" if not fails else "FAIL", failures=fails, utc_us=common.utc_us())
        return doc


def poison_test(out: str | os.PathLike, work: str | os.PathLike | None = None, *, poison_contexts: int = 2,
                sensitivity: str | None = None, guard=None) -> dict:
    """PLAN §5 P5 poison test (spec §3.10) on a built corpus: build A over unperturbed copies of the selected contexts
    (per cohort the first train_fit one(s) and, round-robin over heldout, dev and diag, enough non-train ones that every
    non-empty non-train partition is perturbed; _PoisonHarness.select) and build B over copies with every exposed
    field (the outer phase's timing values kept under corpus_schema.RUNTIME_POLICY), every JTT1 outer-test byte, the
    side files' exposed fields and the non-train records and JTT1 train rows perturbed. PASS needs the whole train/
    tree (the train closure, shards and statistics) byte-identical, the runtime report's train_fit groups equal, every
    non-empty non-train partition perturbed, every control to differ and A's train rows equal to the main build's.
    Returns the jope.dl-corpus-poison/v2 receipt. A `sensitivity` (POISON_SENSITIVITY) must make the verdict FAIL.
    The work directory defaults to RUN/tmp/p5-poison-*."""
    if sensitivity is not None and sensitivity not in POISON_SENSITIVITY:
        raise ValueError(f"sensitivity must be one of {POISON_SENSITIVITY}")
    if poison_contexts < 1:
        raise ValueError("poison_contexts must be >= 1")
    out = _check_read(out)
    if work:
        work_dir = _check_out(work)
    else:
        work_dir = _check_out(tempfile.mkdtemp(prefix="p5-poison-", dir=str(SAMPLE_ROOT)))
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(work_dir)))
    guard = _guard_for(out, guard, 1800.0)
    return _PoisonHarness(out, run_dir, poison_contexts, sensitivity, guard).execute()


# ------------------------------------------------------------------------------------------- sample

def select_sample(epoch: str | os.PathLike | Mapping, per_cell: int = 4, seed: int = 20261005) -> list[dict]:
    """The sample's contexts: included epoch contexts grouped by (cohort, fold), each cell sorted by context_sha256,
    rng = random.Random(seed), then for cohort in (frozen, incoming), for fold 0..4: rng.sample(cell, per_cell).
    A cell with fewer than per_cell contexts is STOP(sample)."""
    doc = epoch if isinstance(epoch, Mapping) else json.loads(Path(epoch).read_bytes())
    cells: dict[tuple, list] = defaultdict(list)
    for e in doc["contexts"]:
        if isinstance(e, dict) and e.get("status") == "included":
            cells[(e["cohort"], e["fold"])].append(e)
    rng = random.Random(seed)
    picks: list[dict] = []
    for cohort in COHORTS:
        for fold in range(5):
            cell = sorted(cells.get((cohort, fold), []), key=lambda e: e["context_sha256"])
            if len(cell) < per_cell:
                raise _stop("sample", f"cell ({cohort}, fold {fold}) holds {len(cell)} < {per_cell} contexts")
            picks += rng.sample(cell, per_cell)
    return picks


SAMPLE_QUOTAS = (("sealed", 1), ("heldout", 3), ("dev", 3))
SAMPLE_ROLE_DOMAIN = b"jope.dl-p5-sample-role/v1\0"


def sample_partition_map(selected: Sequence[Mapping], quotas: Sequence[tuple[str, int]] = SAMPLE_QUOTAS) -> dict:
    """The sample's fake partition map (P4a format): family_id = 'p5-sample-fam:' + dataset_hash; per cohort, datasets
    in sha256('jope.dl-p5-sample-role/v1\\0' + dataset_hash) order fill sealed 1, heldout 3, dev 3, then train (an
    already-assigned dataset keeps its role); a train-family context is diag at fold 3, else train_fit."""
    role_of: dict[str, str] = {}
    for cohort in COHORTS:
        datasets = sorted({e["dataset_hash"] for e in selected if e["cohort"] == cohort},
                          key=lambda d: (_sha(SAMPLE_ROLE_DOMAIN + d.encode()), d))
        filled: Counter = Counter()
        for d in datasets:
            if d in role_of:
                continue
            for role, q in quotas:
                if filled[role] < q:
                    role_of[d] = role
                    filled[role] += 1
                    break
            else:
                role_of[d] = "train"
    pmap = {}
    for e in selected:
        role = role_of[e["dataset_hash"]]
        split = ("diag" if e["fold"] == 3 else "train_fit") if role == "train" else role
        pmap[e["context_sha256"]] = {"family_id": "p5-sample-fam:" + e["dataset_hash"], "split_role": split}
    return pmap


def sample_out(label: str, out: str | os.PathLike | None = None) -> Path:
    """OUT of a sample run: RUN/tmp/p5-sample-<label>, refusing anything else (STOP gate)."""
    if not isinstance(label, str) or SAMPLE_NAME_RE.fullmatch("p5-sample-" + label) is None or label in (".", ".."):
        raise _stop("gate", f"sample label {label!r} must match [A-Za-z0-9._-]+")
    path = Path(out) if out is not None else SAMPLE_ROOT / f"p5-sample-{label}"
    real = os.path.realpath(path)
    if os.path.dirname(real) != os.path.realpath(SAMPLE_ROOT) or not _is_sample_dir(real):
        raise _stop("gate", f"{real}: sample outputs go only to {SAMPLE_ROOT}/p5-sample-<label>")
    return Path(real)


def projection_pass(epoch_doc: Mapping, sources: P.Sources) -> dict:
    """Read-only projection over every included context: Σ of the allowlisted SUMMARY_KEYS (frozen history-ref.json
    files; the incoming cursor read once) and Σ input_ref.bytes."""
    sums: Counter = Counter()
    inc_ids = []
    for e in epoch_doc["contexts"]:
        if not isinstance(e, dict) or e.get("status") != "included":
            continue
        sums["contexts"] += 1
        sums["jtt1_gz_bytes"] += int((e.get("input_ref") or {}).get("bytes") or 0)
        if e["cohort"] == "frozen":
            native = e.get("native_source") if isinstance(e.get("native_source"), dict) else {}
            path = Path(native.get("path") or "/nonexistent").parent / "history-ref.json"
            if not path.is_file():
                sums["frozen_summary_missing"] += 1
                continue
            doc = json.loads(path.read_bytes())
            for k in P.SUMMARY_KEYS:
                sums[f"frozen_{k}"] += int(doc.get(k) or 0)
        else:
            inc_ids.append(e["context_sha256"])
    projected, pin = P.load_incoming_cursor(sources.incoming_cursor)
    missing = 0
    for cid in inc_ids:
        e = projected.get(cid)
        if e is None:
            missing += 1
            continue
        for k in P.SUMMARY_KEYS:
            sums[f"incoming_{k}"] += int(e["stats"].get(k) or 0)
    for k in P.SUMMARY_KEYS:
        sums[k] = sums[f"frozen_{k}"] + sums[f"incoming_{k}"]
    return {"sums": dict(sorted(sums.items())), "incoming_missing": missing,
            "incoming_cursor": {k: pin[k] for k in ("path", "bytes", "sha256", "read_utc_us")}}


@contextlib.contextmanager
def _phase(phases: dict, name: str):
    t0, c0 = time.time(), time.process_time()
    try:
        yield
    finally:
        phases[name] = {"wall_s": round(time.time() - t0, 3), "cpu_s": round(time.process_time() - c0, 3),
                        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def _quantiles(vals: Sequence[int]) -> dict:
    s = sorted(vals)
    return {**{k: _nearest_rank(s, q) for k, q in (("p50", "0.5"), ("p90", "0.9"), ("p99", "0.99"))},
            "max": s[-1] if s else None, "n": len(s)}


def _column_bytes(paths: Sequence[Path]) -> dict:
    out: Counter = Counter()
    for p in paths:
        md = pq.ParquetFile(p).metadata
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for j in range(rg.num_columns):
                out[rg.column(j).path_in_schema] += rg.column(j).total_compressed_size
    return dict(sorted(out.items()))


def sample_measurements(out: Path, corpus: Path, dict_dir: Path, tok_dir: Path, *, with_sampler_plan: bool) -> dict:
    """The sample report's measurements, from the built corpus (non-sealed shards decoded; sealed ones at footer
    level), its ctx-records, the sample P4a dictionary and the sample tokenizer."""
    cur = json.loads((corpus / "cursor.json").read_bytes())
    man = json.loads((corpus / "manifest.json").read_bytes()) if (corpus / "manifest.json").exists() else {}
    m: dict = {"tables": {}, "labels": {}, "tokens": {}, "backend": {}, "leak": [], "reconciliation": {}}
    rows_total = 0
    for p in PARTITIONS:
        for t in PARQUET_TABLES:
            shards = cur["writers"][p][t]["shards"]
            if not shards:
                continue
            rows = sum(sh["rows"] for sh in shards)
            comp = sum(sh["bytes"] for sh in shards)
            m["tables"][f"{p}/{t}"] = {"files": len(shards), "rows": rows, "arrow_nbytes": sum(sh["arrow_nbytes"] for sh in shards),
                                       "parquet_uncompressed": sum(sh["parquet_uncompressed"] for sh in shards),
                                       "parquet_bytes": comp, "bytes_per_row": comp / rows if rows else None,
                                       "column_compressed_bytes": _column_bytes([corpus / sh["path"] for sh in shards])}
            if t in ("records", "refused"):
                rows_total += rows
    m["rows_total"] = rows_total
    for p in ("train_fit", "dev", "diag"):
        f = part_dir(corpus, p) / "label-stats.json"
        if f.exists():
            m["labels"][p] = json.loads(f.read_bytes())
        lengths = []
        for sh in cur["writers"][p]["records"]["shards"]:
            lengths += pq.read_table(corpus / sh["path"], columns=["length"], use_threads=False).column(0).to_pylist()
        m["tokens"][p] = {"length": _quantiles(lengths),
                          "refused": sum(sh["rows"] for sh in cur["writers"][p]["refused"]["shards"])}
        by_backend, by_source, disagree = Counter(), Counter(), Counter()
        for r in read_ctx_records(corpus, cur, p):
            if r.get("leak"):
                m["leak"].append({"context_sha256": r["context_sha256"], "partition": p,
                                  "exact": len(r["leak"]["exact"]), "affine": len(r["leak"]["affine"]),
                                  "target_leak": r["leak"]["target_leak"],
                                  "fold0_construction_contradiction": r.get("fold0_construction_contradiction")})
            counts = r.get("counts") or {}
            by_backend.update({f"{r['cohort']}:{k}": v for k, v in (counts.get("by_backend") or {}).items()})
            by_source.update({f"{r['cohort']}:{k}": v for k, v in (counts.get("backend_source") or {}).items()})
            for k in ("backend_field_disagreements", "backend_field_disagreements_known",
                      "backend_field_disagreements_unknown"):
                disagree[k] += int(counts.get(k) or 0)
        m["backend"][p] = {"by_backend": dict(sorted(by_backend.items())), "backend_source": dict(sorted(by_source.items())),
                           **{k: disagree[k] for k in ("backend_field_disagreements", "backend_field_disagreements_known",
                                                       "backend_field_disagreements_unknown")}}
    try:
        vocab, max_seq, _closure = T._load_tok(tok_dir)
        m["tokens"]["max_seq"] = max_seq
        for p in PARTITIONS:
            stats = T.TokenStats(vocab, check=False, max_seq=max_seq)
            # P4a's projection role for token statistics: (program_sha256, dsl) of any partition, journaled
            for _sha256, dsl in P.iter_distinct(dict_dir, p, role="roundtrip"):
                stats.add(dsl)
            sm = stats.summary()
            m["tokens"].setdefault(p, {})["oov"] = {k: sm[k] for k in ("programs", "fallback_programs", "fallback_rate",
                                                                      "fallback_spans", "over_max_seq")}
    except (OSError, ValueError, T.TokenizerStop) as err:
        m["tokens"]["error"] = str(err)[:300]
    sealed = {}
    for p in PARTITIONS:
        mp = manifest_path(corpus, p, "exposed")
        if mp.exists():
            sealed[p] = json.loads(mp.read_bytes()).get("counts")
    m["exposed_checks"] = sealed
    vec_cols = [f.name for f in S.EXPOSED_RECORD_SCHEMA if pa.types.is_list(f.type)]
    m["exposed_vector_bytes_per_row"] = {
        key: {c: sum(v for k, v in t["column_compressed_bytes"].items() if k.split(".")[0] == c) / t["rows"]
              for c in vec_cols}
        for key, t in m["tables"].items() if key.endswith("/exposed") and t["rows"]}
    m["rates"] = sample_rates(cur["ledger"])
    m["reconciliation"] = man.get("reconciliation")
    m["exclusions"] = man.get("exclusions")
    m["tolerances"] = man.get("tolerances")
    m["deviations"] = man.get("deviations")
    # every --continue-on-stop STOP and target_leak exclusion of train_fit, dev and diag by context; heldout's stay in
    # the sealed heldout summary (only the build's total stop_count covers them here)
    m["context_flags"] = {r["context_sha256"]: ("stopped:" + r["stopped"]["kind"]) if r.get("stopped") else r["excluded"]
                          for p in OPEN_PARTITIONS for r in read_ctx_records(corpus, cur, p)
                          if r.get("stopped") or r.get("excluded")}
    m["stop_count"] = int(cur.get("stop_count") or 0)
    m["stops"] = [{"context_sha256": r["context_sha256"], "partition": p, "cohort": r["cohort"], **r["stopped"]}
                  for p in OPEN_PARTITIONS for r in read_ctx_records(corpus, cur, p) if r.get("stopped")]
    m["runtime"] = json.loads((corpus / "runtime-report.json").read_bytes()) if (corpus / "runtime-report.json").exists() else None
    m["ledger"] = cur["ledger"]
    if with_sampler_plan:
        m["sampler_plan"] = sampler_plan(corpus)
    return m


def sample_rates(ledger: Mapping) -> dict:
    """Measured per-stage costs from the build invocations' timers: parse MB/s over the native, outcome and transport
    bytes, microseconds per program for parse, records (tokenize, skeleton, labels), exposed + lineage projection
    and shard writes, and seconds per context."""
    timers: Counter = Counter()
    for inv in ledger.get("invocations") or []:
        timers.update(inv.get("timers") or {})
    local = Counter(ledger.get("local_bytes") or {})
    progs, ctxs = timers["programs"], timers["contexts"]
    parsed = local["native"] + local["outcome"] + local["transport"]
    return {"contexts": ctxs, "programs": progs, "lines": timers["lines"], "parsed_bytes": parsed,
            "parse_mb_per_s": parsed / 1e6 / (timers["parse_ns"] / 1e9) if timers["parse_ns"] else None,
            "us_per_program": ({k[:-3]: timers[k] / 1e3 / progs for k in ("parse_ns", "records_ns", "exposed_lineage_ns",
                                                                         "write_ns", "context_ns")} if progs else None),
            "jtt1_ms_per_context": timers["jtt1_ns"] / 1e6 / ctxs if ctxs else None,
            "s_per_context": timers["context_ns"] / 1e9 / ctxs if ctxs else None,
            "local_bytes": dict(sorted(local.items()))}


def sampler_plan(corpus: Path) -> dict:
    """sampler.context_plan over the train_fit contexts with dummy zero encodings: shares of the group and
    feasibility streams by reason."""
    from dltrain import sampler as SM
    ctxs, recs = load_split(corpus, "train_fit", "fit")
    if not ctxs:
        return {"contexts": 0}
    corpus_obj = SM.Corpus.build(ctxs, recs, {c["context_sha256"]: np.zeros(1, dtype=np.float32) for c in ctxs},
                                 {c["family_id"]: "train_fit" for c in ctxs})
    cfg = SM.SamplerConfig()
    streams: Counter = Counter()
    for c in sorted(corpus_obj.contexts):
        plan = SM.context_plan(corpus_obj, c, cfg)
        streams[plan.stream if plan.reason is None else f"{plan.stream}:{plan.reason}"] += 1
    n = sum(streams.values())
    return {"contexts": n, "streams": dict(sorted(streams.items())),
            "shares": {k: v / n for k, v in sorted(streams.items())}}


def projection_report(meas: Mapping, phases: Mapping, proj: Mapping, sample_contexts: int) -> dict:
    """Full-epoch projections from the measured per-row bytes and per-record time (measurement only)."""
    sums = proj["sums"]
    rows_full = sums.get("unique_programs", 0)
    rows_sample = meas.get("rows_total") or 0
    bytes_full = {}
    for key, t in meas["tables"].items():
        if t["rows"] and key.split("/")[1] in ("records", "exposed", "lineage", "refused"):
            bytes_full[key] = t["parquet_bytes"] / max(rows_sample, 1) * rows_full
    ctx_bytes = sum(t["parquet_bytes"] for k, t in meas["tables"].items() if k.endswith("/contexts"))
    bytes_full["contexts"] = ctx_bytes / max(sample_contexts, 1) * sums.get("contexts", 0)
    total = sum(bytes_full.values())
    build_s = (phases.get("build", {}).get("wall_s") or 0.0)
    per_record = build_s / rows_sample if rows_sample else None
    fetch = phases.get("fetch", {}).get("wall_s") or 0.0
    est = None if per_record is None else per_record * rows_full + fetch / max(sample_contexts, 1) * sums.get("contexts", 0)
    trig = {"output_over_half_cap": total > 0.5 * EPOCH_CAP_BYTES,
            "single_process_over_2h": est is not None and est > 7200}
    return {"rows": rows_full, "bytes_by_table": dict(sorted(bytes_full.items())), "total_bytes": total,
            "epoch_cap_bytes": EPOCH_CAP_BYTES, "fraction_of_cap": total / EPOCH_CAP_BYTES,
            "seconds_per_record": per_record, "estimated_build_s": est,
            "local_jsonl_bytes": sum(sums.get(k, 0) for k in ("native_bytes", "outcome_bytes", "transport_bytes")),
            "jtt1_gz_bytes": sums.get("jtt1_gz_bytes"), "decision_triggers": trig}


def sample(*, label: str, jtt1_cache: str | os.PathLike, epoch: str | os.PathLike = DEFAULT_EPOCH,
           epoch_sha256: str | None = DEFAULT_EPOCH_SHA256, per_cell: int = 4, seed: int = 20261005, max_gets: int = 64,
           seed_pool: str | os.PathLike = DEFAULT_SEED_POOL, seed_pool_sha256: str = DEFAULT_SEED_POOL_SHA256,
           poison_contexts: int = 2, with_sampler_plan: bool = False, continue_on_stop: bool = False,
           resume: bool = False, out: str | os.PathLike | None = None, incoming_cursor=None,
           incoming_transport_index=None, client_factory: Callable | None = None, guard=None,
           argv: Sequence[str] | None = None, offline: bool = False) -> dict:
    """The bounded real-sample measurement (measurement only; never a P5 run): S0 guards, S1 selection, S2 inputs
    (sample epoch, fake map, SEED192 proxy), S3 P4a at sample scale, S4 sample vocabulary, S5 fetch (<= max_gets
    exact GETs, budget = Σ input_ref.bytes; with offline, cached inputs only and a miss is a STOP), S6 build, S7
    verify (allowlist, counts, poison), S8 projection pass, S9 sampler plan, S10 report and receipt. The verdict is
    PASS only when fetch, build and verify pass and no context stopped. Returns the report summary."""
    t_start = common.utc_us()
    cpuguard.require_slice_limits()
    cpuguard.set_threads(4)
    outp = sample_out(label, out)
    outp.mkdir(parents=True, exist_ok=True)
    if any(x.name != "guard-journal.jsonl" for x in outp.iterdir()) and not resume:
        raise _stop("gate", f"{outp} is not empty; pass --resume to continue a sample")
    (outp / "tmp").mkdir(exist_ok=True)
    tempfile.tempdir = str(outp / "tmp")
    guard = guard if guard is not None else cpuguard.Guard(journal_path=outp / "guard-journal.jsonl", max_pause_s=1800)
    phases: dict = {}
    exits: dict = {}
    sources = P.Sources(Path(incoming_cursor) if incoming_cursor else P.DEFAULT_SOURCES.incoming_cursor,
                        Path(incoming_transport_index) if incoming_transport_index
                        else P.DEFAULT_SOURCES.incoming_transport_index)
    with _phase(phases, "select"):
        raw = Path(epoch).read_bytes()
        if epoch_sha256 and _sha(raw) != epoch_sha256:
            raise _stop("input", f"epoch sha256 {_sha(raw)} != {epoch_sha256}")
        epoch_doc = json.loads(raw)
        selected = select_sample(epoch_doc, per_cell, seed)
    with _phase(phases, "inputs"):
        head = {k: v for k, v in epoch_doc.items() if k != "contexts"}
        sample_epoch = outp / "sample-epoch.json"
        sdoc = {**head, "contexts": selected, "sample_of": {"path": str(epoch), "sha256": _sha(raw)}}
        common.atomic_write(sample_epoch, json.dumps(sdoc, sort_keys=True).encode())
        pmap = sample_partition_map(selected)
        map_path = outp / "partition-map.json"
        common.atomic_write(map_path, json.dumps(pmap, sort_keys=True).encode())
        pool_raw = Path(seed_pool).read_bytes()
        if _sha(pool_raw) != seed_pool_sha256:
            raise _stop("input", f"seed pool sha256 {_sha(pool_raw)} != {seed_pool_sha256}")
        seed_hashes = sorted({json.loads(line)["program"]["program_sha256"] for line in pool_raw.splitlines()
                              if line.strip()})
        seeds_path = outp / "seeds.json"
        common.atomic_write(seeds_path, json.dumps({"seed_hashes": seed_hashes, "label": "pre-P2 SEED192 proxy from "
                                                    "recovery-seed-pool.jsonl (program.program_sha256 only)"},
                                                   sort_keys=True).encode())
    processed = [e for e in selected if pmap[e["context_sha256"]]["split_role"] in PARTITIONS]
    dict_dir, tok_dir, corpus = outp / "program-dict", outp / "tokenizer", outp / "corpus"
    with _phase(phases, "p4a"):
        try:
            P.build(sample_epoch, map_path, seeds_path, dict_dir, sources=sources)
        except P.ReconcileError as err:
            raise S.CorpusStop(f"sample P4a build: {err}", err.context, f"p4a:{err.kind}") from None
        p4a = P.verify(dict_dir, rehash_sources=True, sources=sources)
        if p4a["verdict"] != "PASS":
            raise _stop("p4a", f"sample P4a verify FAIL: {p4a['failures'][:3]}")
    with _phase(phases, "tokenizer"):
        try:
            tok = T.build_artifacts(dict_dir, sample_epoch, tok_dir, automaton=False)
        except (T.TokenizerStop, T.SourcePinError) as err:
            raise _stop("tokenizer", f"sample tokenizer build: {err}") from None
    inv = (epoch_doc.get("inventory") or {}).get("path")
    common_kw = dict(epoch=sample_epoch, epoch_sha256=_sha(sample_epoch.read_bytes()), partition_map=map_path,
                     seeds=seeds_path, tokenizer=tok_dir, out=corpus, jtt1_cache=jtt1_cache,
                     budget_bytes=sum(int(e["input_ref"]["bytes"]) for e in processed), max_gets=max_gets,
                     program_dict=dict_dir, inventory=inv if inv and Path(inv).exists() else None,
                     incoming_cursor=sources.incoming_cursor, incoming_transport_index=sources.incoming_transport_index,
                     continue_on_stop=continue_on_stop, guard=guard, client_factory=client_factory, offline=offline)
    with _phase(phases, "fetch"):
        fetch = run(phase="fetch", argv=["sample-fetch", label], **common_kw)
        exits["fetch"] = fetch["exit"]
    build = None
    if fetch["exit"] == EXIT_OK:
        with _phase(phases, "build"):
            build = run(phase="build", argv=["sample-build", label], **common_kw)
            exits["build"] = build["exit"]
    rep = None
    if build is not None and build["exit"] == EXIT_OK:
        with _phase(phases, "verify"):
            rep = verify(corpus, schema_allowlist=True, counts=True, poison_test_run=True,
                         poison_contexts=poison_contexts, poison_work=outp / "poison", guard=guard)
            exits["verify"] = EXIT_OK if rep["verdict"] == "PASS" else EXIT_FAIL
    with _phase(phases, "projection"):
        proj = projection_pass(epoch_doc, sources)
    meas = None
    if build is not None and build["exit"] == EXIT_OK:
        with _phase(phases, "measure"):
            meas = sample_measurements(outp, corpus, dict_dir, tok_dir, with_sampler_plan=with_sampler_plan)
    projection = projection_report(meas, phases, proj, len(processed)) if meas else None
    report = {"schema": SAMPLE_REPORT_SCHEMA, "label": label, "measurement_only": True,
              "selection": [{"context_sha256": e["context_sha256"], "cohort": e["cohort"], "fold": e["fold"],
                             "dataset_hash": e["dataset_hash"], **pmap[e["context_sha256"]]} for e in selected],
              "processed_contexts": len(processed), "seed_proxy": {"path": str(seed_pool), "sha256": seed_pool_sha256,
                                                                  "count": len(seed_hashes)},
              "p4a": {"verdict": p4a["verdict"], "checked": p4a["checked"]},
              "tokenizer": {k: tok[k] for k in ("vocab_sha256", "vocab_size", "max_seq")},
              "phases": phases, "exits": exits, "fetch": fetch, "build": build,
              "verify": None if rep is None else {k: rep[k] for k in ("verdict", "failure_count", "failures", "checked")},
              "poison": None if rep is None else rep.get("poison"), "measurements": meas, "projection_pass": proj,
              "projection": projection}
    common.atomic_write(outp / "sample-report.json", common.receipt_bytes(report))
    cur = json.loads((corpus / "cursor.json").read_bytes()) if (corpus / "cursor.json").exists() else None
    stop_count = int((cur or {}).get("stop_count") or 0)
    verdict = ("PASS" if exits and all(v == EXIT_OK for v in exits.values()) and len(exits) == 3 and stop_count == 0
               else "FAIL")
    tol = (meas or {}).get("tolerances") or {}
    receipt = {"schema": SAMPLE_RECEIPT_SCHEMA, "label": label, "argv": common.redact_argv(list(argv or [])),
               "exits": exits, "verdict": verdict, "stop_count": stop_count, "offline": bool(offline),
               "started_utc_us": t_start, "finished_utc_us": common.utc_us(),
               "ledger": cur["ledger"] if cur is not None else None,
               "cv_log_ratio_sd": {"policy": S.CV_SD_POLICY,
                                   **{k: tol.get(k) for k in ("native_outcome_ties", "canonical_stored",
                                                              "canonical_1ulp_from_writer", "verbatim_stored",
                                                              "writer_nonfinite", "contexts_with_ties", "by_context",
                                                              "items")}},
               "runtime_target": S.RUNTIME_POLICY_NOTE,
               "guard": guard.summary(), "report_sha256": common.sha256_file(outp / "sample-report.json")}
    common.atomic_write(outp / "receipt.json", common.receipt_bytes(receipt))
    return {"verdict": verdict, "out": str(outp), "exits": exits, "projection": projection, "stop_count": stop_count}


# ------------------------------------------------------------------------------------------- publish (gated)

def publish(*, out: str | os.PathLike, verify_receipt: str | os.PathLike, g18_receipt: str | os.PathLike,
            s3_cap_bytes: int, guard=None) -> dict:
    """Gated (spec §3.11): requires a verify PASS receipt that includes a poison PASS, a G18 PASS receipt and the
    DL-D7 S3 cap. It computes the create-only object plan and then STOPs: this module holds no S3 write path, and the
    transfer belongs to a separately granted step. The plan rides on the STOP as `err.plan`. Object kinds keep the
    sealed tables apart (red-team LA-10): objects/corpus-shard/ (train, dev and diag shards only),
    objects/corpus-sealed/ (heldout/** and exposed/**, sidecars included), objects/corpus-lineage/ and
    objects/corpus-provenance/; sealed manifests go to manifests-sealed/, the others to manifests/. Only the train
    closure (train/manifest.json) is marked run_plan_ref."""
    cpuguard.require_slice_limits()
    cpuguard.set_threads(4)
    out = _check_read(out)
    _guard_for(out, guard, 1800.0).wait()
    vr = json.loads(Path(verify_receipt).read_bytes())
    if vr.get("schema") != VERIFY_SCHEMA or vr.get("verdict") != "PASS" or \
            not (vr.get("checks") or {}).get("poison_test") or (vr.get("poison") or {}).get("verdict") != "PASS":
        raise _stop("gate", "publish needs a verify PASS receipt that includes a poison-test PASS")
    if not _receipt_ok(g18_receipt):
        raise _stop("gate", "publish needs a G18 PASS receipt")
    cur = json.loads((out / "cursor.json").read_bytes())
    objects = []

    def kind_of(p: str, t: str) -> str:
        if is_sealed(p, t):
            return "sealed"
        return {"lineage": "lineage", "ctx-records": "provenance"}.get(t, "shard")

    for p in PARTITIONS:
        for t in TABLES:
            kind = kind_of(p, t)
            for sh in cur["writers"][p][t]["shards"]:
                ext = ".parquet" if sh["path"].endswith(".parquet") else ".json"
                objects.append({"source": sh["path"], "kind": kind, "sealed": kind == "sealed",
                                "key": PUBLISH_PREFIXES[kind] + sh["sha256"] + ext, "bytes": sh["bytes"]})
                side = sh.get("sidecar")
                if side:
                    objects.append({"source": side["path"], "kind": kind, "sealed": kind == "sealed",
                                    "key": PUBLISH_PREFIXES[kind] + side["sha256"] + ".json", "bytes": side["bytes"]})
    closure = manifest_path(out, "train_fit")
    for f in sorted(out.rglob("manifest.json")):
        rel = _rel(out, f)
        if Path(rel).parts[0] in ("program-dict", "poison", "tmp"):
            continue
        sealed = S.is_sealed_path(rel)
        objects.append({"source": rel, "kind": "manifest", "sealed": sealed, "run_plan_ref": f == closure,
                        "key": (PUBLISH_SEALED_MANIFEST_PREFIX if sealed else PUBLISH_MANIFEST_PREFIX)
                        + common.sha256_file(f) + ".json", "bytes": f.stat().st_size})
    summary = heldout_summary_path(out)
    if summary.is_file():
        objects.append({"source": _rel(out, summary), "kind": "manifest", "sealed": True, "run_plan_ref": False,
                        "key": PUBLISH_SEALED_MANIFEST_PREFIX + common.sha256_file(summary) + ".json",
                        "bytes": summary.stat().st_size})
    total = sum(o["bytes"] for o in objects)
    plan = {"schema": PUBLISH_PLAN_SCHEMA, "objects": len(objects), "bytes": total, "s3_cap_bytes": int(s3_cap_bytes),
            "within_cap": total <= int(s3_cap_bytes), "plan": objects}
    if total > int(s3_cap_bytes):
        err = _stop("gate", f"publish plan of {total} B exceeds the DL-D7 S3 cap {s3_cap_bytes} B")
    else:
        err = _stop("gate", f"publish is not enabled in this lane build: {len(objects)} objects, {total} B planned "
                            "(create-only, exact-version readback) for a separately granted step")
    err.plan = plan
    raise err


# ------------------------------------------------------------------------------------------------- CLI

def _print(doc: Mapping, stream=None) -> None:
    print(json.dumps(doc, sort_keys=True, default=str), file=stream or sys.stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.corpus_build", description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build (or resume) the P5 corpus")
    r.add_argument("--epoch", required=True)
    r.add_argument("--epoch-sha256", required=True)
    r.add_argument("--partition-map", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--jtt1-cache", required=True)
    r.add_argument("--budget-bytes", type=int, required=True)
    r.add_argument("--max-gets", type=int)
    r.add_argument("--local-quota-bytes", type=int)
    r.add_argument("--program-dict")
    r.add_argument("--inventory")
    r.add_argument("--incoming-cursor")
    r.add_argument("--incoming-transport-index")
    ex = r.add_mutually_exclusive_group()
    ex.add_argument("--frozen-qualified-views")
    ex.add_argument("--no-exposed-high-water", action="store_true")
    r.add_argument("--phase", choices=("all", "fetch", "build"), default="all")
    r.add_argument("--target-shard-bytes", type=int, default=TARGET_SHARD_BYTES)
    r.add_argument("--offline", action="store_true")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--continue-on-stop", action="store_true", help="sample only: record the STOP and skip the context")
    r.add_argument("--gate-p3")
    r.add_argument("--gate-p4")
    r.add_argument("--expected-completed", type=int, help=argparse.SUPPRESS)
    r.add_argument("--max-pause-s", type=float, default=1800.0)
    v = sub.add_parser("verify", help="verify a built corpus")
    v.add_argument("--out", required=True)
    v.add_argument("--schema-allowlist", action="store_true")
    v.add_argument("--counts", action="store_true")
    v.add_argument("--poison-test", action="store_true")
    v.add_argument("--poison-contexts", type=int, default=2)
    v.add_argument("--poison-work")
    v.add_argument("--role", choices=("heldout_report",))
    v.add_argument("--rehash-sources", action="store_true")
    v.add_argument("--receipt")
    s = sub.add_parser("sample", help="bounded real-sample measurement (measurement only)")
    s.add_argument("--label", required=True)
    s.add_argument("--per-cell", type=int, default=4)
    s.add_argument("--seed", type=int, default=20261005)
    s.add_argument("--max-gets", type=int, default=64)
    s.add_argument("--jtt1-cache", required=True)
    s.add_argument("--epoch", default=str(DEFAULT_EPOCH))
    s.add_argument("--epoch-sha256", default=DEFAULT_EPOCH_SHA256)
    s.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL))
    s.add_argument("--seed-pool-sha256", default=DEFAULT_SEED_POOL_SHA256)
    s.add_argument("--poison-contexts", type=int, default=2)
    s.add_argument("--with-sampler-plan", action="store_true")
    s.add_argument("--continue-on-stop", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--offline", action="store_true", help="cached JTT1 inputs only (a cache miss is a STOP)")
    s.add_argument("--incoming-cursor")
    s.add_argument("--incoming-transport-index")
    pb = sub.add_parser("publish", help="gated: plan the create-only publish (never writes S3 here)")
    pb.add_argument("--out", required=True)
    pb.add_argument("--verify-receipt", required=True)
    pb.add_argument("--g18-receipt", required=True)
    pb.add_argument("--s3-cap-bytes", type=int, required=True)
    args = ap.parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if args.cmd == "run":
            res = run(epoch=args.epoch, epoch_sha256=args.epoch_sha256, partition_map=args.partition_map,
                      seeds=args.seeds, tokenizer=args.tokenizer, out=args.out, jtt1_cache=args.jtt1_cache,
                      budget_bytes=args.budget_bytes, max_gets=args.max_gets, local_quota_bytes=args.local_quota_bytes,
                      program_dict=args.program_dict, inventory=args.inventory, incoming_cursor=args.incoming_cursor,
                      incoming_transport_index=args.incoming_transport_index,
                      frozen_qualified_views=args.frozen_qualified_views, exposed=not args.no_exposed_high_water,
                      phase=args.phase, target_shard_bytes=args.target_shard_bytes, offline=args.offline,
                      resume=not args.no_resume, continue_on_stop=args.continue_on_stop, gate_p3=args.gate_p3,
                      gate_p4=args.gate_p4, expected_completed=args.expected_completed, max_pause_s=args.max_pause_s,
                      argv=raw_argv)
            _print(res, sys.stdout if res["exit"] == EXIT_OK else sys.stderr)
            return res["exit"]
        if args.cmd == "verify":
            any_check = args.schema_allowlist or args.counts or args.poison_test or args.rehash_sources
            rep = verify(args.out, schema_allowlist=args.schema_allowlist or not any_check,
                         counts=args.counts or not any_check, poison_test_run=args.poison_test,
                         poison_contexts=args.poison_contexts, poison_work=args.poison_work, role=args.role,
                         rehash_sources=args.rehash_sources, receipt=args.receipt)
            _print({k: rep[k] for k in ("verdict", "failure_count", "failures", "checked")}
                   | {"poison": (rep.get("poison") or {}).get("verdict")})
            return EXIT_OK if rep["verdict"] == "PASS" else EXIT_FAIL
        if args.cmd == "sample":
            res = sample(label=args.label, jtt1_cache=args.jtt1_cache, epoch=args.epoch, epoch_sha256=args.epoch_sha256,
                         per_cell=args.per_cell, seed=args.seed, max_gets=args.max_gets, seed_pool=args.seed_pool,
                         seed_pool_sha256=args.seed_pool_sha256, poison_contexts=args.poison_contexts,
                         with_sampler_plan=args.with_sampler_plan, continue_on_stop=args.continue_on_stop,
                         resume=args.resume, incoming_cursor=args.incoming_cursor,
                         incoming_transport_index=args.incoming_transport_index, argv=raw_argv, offline=args.offline)
            _print(res)
            exits = list(res["exits"].values())
            return EXIT_PAUSED if EXIT_PAUSED in exits else (EXIT_OK if res["verdict"] == "PASS" else EXIT_FAIL)
        publish(out=args.out, verify_receipt=args.verify_receipt, g18_receipt=args.g18_receipt,
                s3_cap_bytes=args.s3_cap_bytes)
        return EXIT_OK
    except (S.CorpusStop, P.ReconcileError) as err:
        doc = {"verdict": "STOP", "kind": _kind(err), "context": getattr(err, "context", None), "error": str(err)[:500]}
        if getattr(err, "plan", None) is not None:
            doc["plan"] = err.plan
        _print(doc, sys.stderr)
        return EXIT_STOP
    except common.ImmutableConflict as err:
        _print({"verdict": "CONFLICT", "error": str(err)[:500]}, sys.stderr)
        return EXIT_CONFLICT
    except cpuguard.GuardTimeout as err:
        _print({"verdict": "PAUSED", "status": "PAUSED_GUARD", "error": str(err)[:300]}, sys.stderr)
        return EXIT_PAUSED


if __name__ == "__main__":
    raise SystemExit(main())
