"""PLAN §5 P4a: the program dictionary, built in one pass over pinned local sources.

Definition (pack_fold_history.build_history, which both cohorts use): the unique programs of a context are
native evaluation rows ∪ outcome rows that carry `program` ∪ transport rows that carry `program`. That count is
the packed history summary's `unique_programs`, and every context must reconcile to it exactly.

Sources per context (local and read-only; no S3, no network):
- native: the epoch's `native_source` JSONL, read to its pinned byte count;
- outcome prefix, which supplies the bootstrap programs and their record_status:
    frozen:   <ctxdir>/history-outcomes.jsonl, first `outcome_bytes` bytes;
    incoming: the item's run/gp/hwm-events.jsonl lines whose top-level context_sha256 is this context, until their
              byte sum reaches `outcome_bytes` (the file is append-only and live);
- transport prefix, which supplies the transport-only programs:
    frozen:   <ctxdir>/history-transports.jsonl, first `transport_bytes` bytes;
    incoming: <transport index>/<ctx>.jsonl, first `transport_bytes` bytes;
- history summary, read through an allowlisted projection only:
    frozen:   <ctxdir>/history-ref.json, whose history_ref.sha256 must equal the epoch pin;
    incoming: fold-archive-cursor-v2.json[ctx], whose source_ref.sha256 (and nothing else) must equal the epoch pin.
              Every raw entry is re-projected to exactly five keys; the file is rewritten live, so it is read once
              per run and its sha256 and read time are recorded;
- <ctxdir>/seed-pool.jsonl, a cross-check (its program set must equal the bootstrap set) and the source of the
  conditioned-seed flag (rows of schema jope.regression-research-conditioned-seed/v1: the 16 prior champions that
  prepare_launch.py ranked by the same dataset's fold-0 outer test). Only `program` and the row's `schema` marker
  are read; rank, source_evidence and source_context values are never touched;
- <ctxdir>/bootstrap-results.jsonl (snapshot to its last newline), a cross-check in both cohorts: its program set
  must equal the bootstrap set, and its last row's record_status per program is compared with the outcome row's
  (the outcome row wins). Disagreements are counted per context; a build STOPs when they exceed the design's
  observed rate of one per 2,864 contexts (at least one is always tolerated).

How each prefix is verified:
- native: bytes and sha256 against the epoch pin, and line count against native_events;
- outcome and transport: the recorded byte count must end on a line boundary, and the line count must equal the
  recorded event count. Incoming prefixes are also sha-pinned: the archiver's fingerprint is
  sha256(native ‖ outcome ‖ transport ‖ bootstrap-results snapshot), and the snapshot's own sha256 is
  generation_zero_source_sha256. Frozen outcome and transport prefixes have no local sha pin: their sha256 is
  recorded in the manifest and `verify --rehash-sources` re-checks it, which catches edits made AFTER the build.
  An edit made BEFORE the build that changes a projected field of a frozen outcome row (a bootstrap
  record_status, say) is caught only by the bootstrap-results cross-check: it shows up as a counted
  disagreement, and a second one in a build is a STOP. An edit that keeps every projected field is invisible.

Reconciliation per context (any failure raises ReconcileError, a STOP):
- sha256(dsl) == program_sha256 on every row;
- no duplicate (context, program);
- native rows == native_events;
- bootstrap set == seed-pool set == bootstrap-results set;
- every outcome-row program is a native or bootstrap program, so transport-only labels are exact;
- |rows| == unique_programs.
An optional `s3_fallback` hook may replace a failing context with rows from its exact-version S3 history. That
read needs the DL-D7 read line and is not implemented here, so the default is STOP.

Never read:
- historical-seen.jsonl;
- the seed-pool fields rank, source_evidence and source_context;
- any test-derived field (test_rmse, fitness, best_test_rmse, signatures, ...).
Every record is projected to named keys at parse time, and no other key is accessed, logged or written.

PUBLISHED SCHEMA (ROW_SCHEMA, the exact allowlist of the partition chunks): context_sha256, program_sha256, dsl,
family_id, split_role, record_status, shared_seed, cohort, form, dsl_bytes. There is no origin column, and the
published record_status is origin-independent: completed | failed | timeout | unevaluated, where unevaluated
merges abandoned:* (evolution only), missing (bootstrap only) and transport_only. Otherwise origin, or 'missing'
together with shared_seed, would single out the conditioned prior champions (bootstrap and not a P2 seed) in
every partition, a test-selected set PLAN §6.1 keeps sealed.

SEALED SIDE TABLE (SEALED_SCHEMA, sealed/<partition>/part-NNNNN.parquet, one per published chunk and row-aligned
with it): context_sha256, program_sha256, origin, the exact record_status and conditioned_seed. It is never part
of a run-plan closure and opens only through open_sealed / SealedIndex under role r7_crosscheck (P5's R7
cross-check: the origin/status columns only) or exposed_report (reporting). The manifest counts conditioned rows
per partition.

Partition membership is injected, because the P2/P3 adapter has not landed. It is a JSON object mapping
context_sha256 -> {"family_id", "split_role"}, where split_role is one of PARTITIONS or sealed/prohibited (those
two are only counted). A frozen fold-0 context mapped to diag is a STOP (PLAN §6.2: folds 1-4 carry 16 champions
selected on fold 0's outer test). A JSON list of the P2 seed program hashes sets shared_seed. When the adapter
lands, registry.json and holdout.json sha256 belong in the manifest and verify should re-derive the map.

READ ROLES (open_partition / iter_distinct / open_sealed / SealedIndex). A role is a claim about the caller's use,
so each role opens only what that use needs:
- fit -> train_fit, training_validation -> dev, diagnostic_only -> diag, heldout_report -> heldout (P10 reporting
  only): every published column of that one partition (corpus_schema.READ_ROLE);
- roundtrip -> any partition, (program_sha256, dsl) only: P4 round trip, automaton gate, Tier A corpus mode, P5
  token statistics;
- preflight -> any partition, (context_sha256, dsl) only: the oracle's informational width preflight;
- r7_crosscheck / exposed_report -> the sealed side table (see above).
Every open verifies each chunk's bytes and sha256 against the manifest and that it holds only rows of its
partition. Every open other than fit on train_fit appends {role, partition, table, columns, caller} to the access
journal (<OUT>/.access-journal.jsonl, or $JOPE_DL_DICT_ACCESS_JOURNAL), failing closed.

Outputs under OUT:
- {train_fit,dev,diag,heldout}/part-NNNNN.parquet: immutable, zstd, 128k-row groups. Each chunk covers 256 contexts
  of one partition, ordered by context_sha256, with rows sorted by program_sha256 within a context;
- sealed/{train_fit,dev,diag,heldout}/part-NNNNN.parquet: the sealed side table, one chunk per published chunk;
- cursor.json: atomic and resumable. Without a cursor, existing partition directories must hold only chunk names
  of the current layout (they are then re-published byte-identically or the build stops);
- manifest.json (jope.dl-program-dict/v1): a pure function of the cursor and the chunks. verify FAILs on any file
  in a partition directory that the manifest does not list (dot-prefixed temporaries excepted).

Parquet reads always pass use_threads=False. In this venv (pyarrow 24), a threaded Parquet read makes the
interpreter abort at exit ("terminate called without an active exception", exit 134).

Commands:
  build  --epoch E --partition-map M --seeds S --out OUT [--chunk-contexts 256] [--no-resume]
  verify --out OUT [--no-reconcile-history-summaries] [--rehash-sources]
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from dltrain import common

MANIFEST_SCHEMA = "jope.dl-program-dict/v1"
CURSOR_SCHEMA = "jope.dl-program-dict-cursor/v1"
VERIFY_SCHEMA = "jope.dl-program-dict-verify/v1"

PARTITIONS = ("train_fit", "dev", "diag", "heldout")
EXCLUDED_ROLES = ("sealed", "prohibited")
# Read roles (module docstring): the partition-specific roles open every published column of one partition, the
# projection roles any partition but only their columns, the sealed roles the sealed side table.
PARTITION_ROLE = {"train_fit": "fit", "dev": "training_validation", "diag": "diagnostic_only", "heldout": "heldout_report"}
PROJECTED_ROLES = {"roundtrip": ("program_sha256", "dsl"), "preflight": ("context_sha256", "dsl")}
SEALED_ROLES = {"r7_crosscheck": ("context_sha256", "program_sha256", "origin", "record_status"), "exposed_report": None}
OPEN_ROLES = (*PARTITION_ROLE.values(), *PROJECTED_ROLES)
JOURNAL_ENV = "JOPE_DL_DICT_ACCESS_JOURNAL"
JOURNAL_NAME = ".access-journal.jsonl"
JOURNAL_SCHEMA = "jope.dl-program-dict-access/v1"
COHORTS = ("frozen", "incoming")
ORIGINS = ("evolution", "bootstrap", "transport_only")
FORMS = ("pipe", "sym", "portfolio")
ABANDON_REASONS = ("transport_error", "capacity_deferred_before_claim", "runtime_error", "blocked_existing_claim")
NATIVE_STATUSES = ("completed", "failed", "timeout")
BOOTSTRAP_STATUSES = ("completed", "failed", "timeout", "missing")
RECORD_STATUSES = (*NATIVE_STATUSES, *(f"abandoned:{r}" for r in ABANDON_REASONS), "missing", "transport_only")
STATUSES_BY_ORIGIN = {
    "evolution": frozenset((*NATIVE_STATUSES, *(f"abandoned:{r}" for r in ABANDON_REASONS))),
    "bootstrap": frozenset(BOOTSTRAP_STATUSES),
    "transport_only": frozenset(("transport_only",)),
}
# The published record_status: every value occurs under every origin that can be evaluated, so no published
# column separates bootstrap (or conditioned-seed) rows from evolution rows.
PUBLISHED_STATUSES = ("completed", "failed", "timeout", "unevaluated")
CONDITIONED_SEED_SCHEMA = "jope.regression-research-conditioned-seed/v1"
BOOT_RESULTS_CONTEXTS_PER_DISAGREEMENT = 2864      # the design's observed rate (one disagreement in 2,864)


def published_status(record_status: str) -> str:
    """The origin-independent published class of an exact record_status."""
    return record_status if record_status in NATIVE_STATUSES else "unevaluated"


SUMMARY_KEYS = ("unique_programs", "native_events", "native_bytes", "outcome_bytes", "outcome_events",
                "transport_bytes", "transport_events")

CHUNK_CONTEXTS = 256
ROW_GROUP_ROWS = 128 * 1024
ZSTD_LEVEL = 6
READ_BLOCK = 8 << 20
DSL_BYTES_MAX = 65535


def _dict_type() -> pa.DataType:
    return pa.dictionary(pa.int8(), pa.string())


ROW_SCHEMA = pa.schema([            # the published allowlist: no origin, origin-independent record_status
    pa.field("context_sha256", pa.binary(32), nullable=False),
    pa.field("program_sha256", pa.binary(32), nullable=False),
    pa.field("dsl", pa.string(), nullable=False),
    pa.field("family_id", pa.string(), nullable=False),
    pa.field("split_role", _dict_type(), nullable=False),
    pa.field("record_status", _dict_type(), nullable=False),
    pa.field("shared_seed", pa.bool_(), nullable=False),
    pa.field("cohort", _dict_type(), nullable=False),
    pa.field("form", _dict_type(), nullable=False),
    pa.field("dsl_bytes", pa.uint16(), nullable=False),
])
DICT_COLUMNS = {"split_role": PARTITIONS, "record_status": PUBLISHED_STATUSES, "cohort": COHORTS, "form": FORMS}
SEALED_SCHEMA = pa.schema([         # the sealed side table, row-aligned with the published chunk of its name
    pa.field("context_sha256", pa.binary(32), nullable=False),
    pa.field("program_sha256", pa.binary(32), nullable=False),
    pa.field("origin", _dict_type(), nullable=False),
    pa.field("record_status", _dict_type(), nullable=False),
    pa.field("conditioned_seed", pa.bool_(), nullable=False),
])
SEALED_DICT_COLUMNS = {"origin": ORIGINS, "record_status": RECORD_STATUSES}
SEALED_DIR = "sealed"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_CTX_RE = re.compile(rb'"context_sha256"\s*:\s*"([0-9a-f]{64})"')
_PSHA_RE = re.compile(rb'"program_sha256"\s*:\s*"([0-9a-f]{64})"')
_PROGRAM_OBJ_RE = re.compile(rb'"program"\s*:\s*\{\s*"')
_BOOTSTRAP_RE = re.compile(rb'"phase"\s*:\s*"bootstrap"')


@dataclass(frozen=True)
class Sources:
    """Incoming-cohort roots that the epoch does not pin by path (frozen sources sit beside native_source)."""
    incoming_cursor: Path = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/fold-archive-cursor-v2.json")
    incoming_transport_index: Path = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/fold-archive-transport-index-v2")


DEFAULT_SOURCES = Sources()


class ReconcileError(RuntimeError):
    """A P4a STOP (PLAN §5 P4a Failure): a count, set, sha, source-prefix, summary or closure mismatch."""

    def __init__(self, message: str, context: str | None = None, kind: str = "reconcile"):
        super().__init__(message if context is None else f"[{context[:16]}] {message}")
        self.context = context
        self.kind = kind


@dataclass(frozen=True)
class ContextRef:
    context_sha256: str
    cohort: str
    view_id: str
    item_id: str | None
    dataset_hash: str
    fold: int
    native: dict
    prepared: dict
    history_sha256: str
    family_id: str
    split_role: str


@dataclass(frozen=True)
class HistorySummary:
    """The allowlisted projection of a packed history summary; nothing else is ever taken from it."""
    unique_programs: int
    native_events: int
    native_bytes: int
    outcome_bytes: int
    outcome_events: int
    transport_bytes: int
    transport_events: int
    history_sha256: str
    fingerprint: str | None = None
    bootstrap_source_sha256: str | None = None


@dataclass(frozen=True)
class ProgramRow:
    context_sha256: str
    program_sha256: str
    dsl: str
    origin: str
    record_status: str            # exact; the published chunk carries published_status(record_status)
    conditioned_seed: bool = False


# ----------------------------------------------------------------------------------------------- small helpers

def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and _HEX64.fullmatch(v) is not None


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _loads(raw: bytes, where: str, ctx: str | None = None) -> Any:
    try:
        return json.loads(raw)
    except ValueError as err:
        raise ReconcileError(f"{where}: invalid JSON ({err})", ctx, "source") from None


def _record(raw: bytes, where: str, ctx: str | None = None) -> dict:
    """One JSONL record; anything but a JSON object is a STOP."""
    r = _loads(raw, where, ctx)
    if not isinstance(r, dict):
        raise ReconcileError(f"{where}: record is not a JSON object", ctx, "source")
    return r


def program_form(dsl: str) -> str:
    if dsl.startswith("(pipe "):
        return "pipe"
    if dsl.startswith("(sym "):
        return "sym"
    if dsl.startswith("(portfolio "):
        return "portfolio"
    raise ReconcileError(f"unrecognized program form {dsl[:32]!r}", kind="row")


def _check_program(dsl: Any, sha: Any, where: str, ctx: str | None = None) -> None:
    if not isinstance(dsl, str) or not _is_hex64(sha):
        raise ReconcileError(f"{where}: program lacks a dsl string or a 64-hex program_sha256", ctx, "row")
    try:
        raw = dsl.encode("utf-8")
    except UnicodeEncodeError:
        raise ReconcileError(f"{where}: dsl is not encodable as UTF-8", ctx, "row") from None
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ReconcileError(f"{where}: sha256(dsl) != program_sha256 {sha[:16]}", ctx, "sha")
    if len(raw) > DSL_BYTES_MAX:
        raise ReconcileError(f"{where}: dsl of {len(raw)} bytes exceeds the uint16 dsl_bytes column", ctx, "row")
    try:
        program_form(dsl)
    except ReconcileError as err:
        raise ReconcileError(f"{where}: {err}", ctx, "row") from None


def _ctx_dir(ctx: ContextRef) -> Path:
    native = Path(ctx.native["path"])
    if native.name != f"evaluations-{ctx.context_sha256[:16]}.jsonl" or native.parent.name != ctx.view_id:
        raise ReconcileError(f"native_source {native} is not <view_id>/evaluations-<ctx16>.jsonl", ctx.context_sha256, "layout")
    return native.parent


def _item_gp(ctx: ContextRef) -> Path:
    d = _ctx_dir(ctx)
    gp = d.parent.parent
    if d.parent.name != "contexts" or (ctx.item_id and gp.parent.parent.name != ctx.item_id):
        raise ReconcileError(f"incoming context dir {d} is not items/<item_id>/run/gp/contexts/<view_id>", ctx.context_sha256, "layout")
    return gp


def _transport_path(ctx: ContextRef, sources: Sources) -> Path:
    if ctx.cohort == "frozen":
        return _ctx_dir(ctx) / "history-transports.jsonl"
    return Path(sources.incoming_transport_index) / f"{ctx.context_sha256}.jsonl"


def _snapshot(path: Path) -> bytes:
    """pack_fold_history.snapshot_lines: the file up to its last newline, or b'' when absent."""
    if not path.exists():
        return b""
    data = path.read_bytes()
    return data[: data.rfind(b"\n") + 1]


def _complete_lines(path: Path) -> Iterator[bytes]:
    """Every complete line of a live append-only file; a trailing partial line is a write in progress."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        rest = b""
        while True:
            block = f.read(READ_BLOCK)
            if not block:
                return
            parts = (rest + block).split(b"\n")
            rest = parts.pop()
            for p in parts:
                yield p + b"\n"


class _Prefix:
    """Iterate the complete lines of a file's first `nbytes` bytes while hashing them (and any extra hashers)."""

    def __init__(self, path: Path, nbytes: int, kind: str, ctx: str | None, extra: Sequence[Any] = ()):
        self.path, self.nbytes, self.kind, self.ctx, self.extra = Path(path), nbytes, kind, ctx, tuple(extra)
        self.sha = hashlib.sha256()
        self.lines = 0
        self.file_bytes = 0

    def __iter__(self) -> Iterator[bytes]:
        if self.nbytes == 0:
            self.file_bytes = self.path.stat().st_size if self.path.exists() else 0
            return
        try:
            f = open(self.path, "rb")
        except FileNotFoundError:
            raise ReconcileError(f"{self.kind}: {self.path} is missing but {self.nbytes} bytes are recorded", self.ctx, "source") from None
        with f:
            self.file_bytes = os.fstat(f.fileno()).st_size
            if self.file_bytes < self.nbytes:
                raise ReconcileError(f"{self.kind}: {self.path} has {self.file_bytes} bytes, fewer than the recorded prefix "
                                     f"of {self.nbytes}", self.ctx, "source")
            remaining, rest = self.nbytes, b""
            while remaining:
                block = f.read(min(READ_BLOCK, remaining))
                if not block:
                    raise ReconcileError(f"{self.kind}: {self.path} shrank while it was read", self.ctx, "source")
                remaining -= len(block)
                self.sha.update(block)
                for h in self.extra:
                    h.update(block)
                parts = (rest + block).split(b"\n")
                rest = parts.pop()
                self.lines += len(parts)
                for p in parts:
                    yield p + b"\n"
            if rest:
                raise ReconcileError(f"{self.kind}: the recorded prefix of {self.path} ({self.nbytes} B) ends inside a line",
                                     self.ctx, "source")

    def record(self, pin: str) -> dict:
        return {"path": str(self.path), "layout": "file_prefix", "bytes": self.nbytes, "lines": self.lines,
                "sha256": self.sha.hexdigest(), "file_bytes": self.file_bytes, "pin": pin}


def _line_context(line: bytes) -> str | None:
    """Top-level context_sha256 of a hwm-events line, as fold_archive.once() groups it."""
    found = _CTX_RE.findall(line)
    if len(found) == 1:
        return found[0].decode()
    if not found and b"context_sha256" not in line:
        return None
    v = _record(line, "hwm-events line").get("context_sha256")
    return v if isinstance(v, str) else None


def _line_program_sha(line: bytes, where: str, ctx: str | None) -> str | None:
    """The top-level program_sha256 of a row that carries a non-empty `program` (packer semantics), else None."""
    if _PROGRAM_OBJ_RE.search(line) is not None:
        found = set(_PSHA_RE.findall(line))
        if len(found) == 1:
            return found.pop().decode()
    elif b'"program"' not in line:
        return None
    r = _record(line, where, ctx)
    if not r.get("program"):
        return None
    sha = r.get("program_sha256")
    if not _is_hex64(sha):
        raise ReconcileError(f"{where}: a row with a program lacks a top-level program_sha256", ctx, "source")
    return sha


# -------------------------------------------------------------------------------------------- inputs and plan

def _load_partition_map(src: Path | str | Mapping) -> tuple[dict[str, dict], dict]:
    if isinstance(src, Mapping):
        path, data = None, dict(src)
        raw = common.canonical(data)
    else:
        path = Path(src)
        raw = path.read_bytes()
        data = _loads(raw, str(path))
    if not isinstance(data, dict):
        raise ReconcileError("partition map must be a JSON object context_sha256 -> {family_id, split_role}", kind="input")
    out = {}
    for cid, v in data.items():
        fam = v.get("family_id") if isinstance(v, dict) else None
        role = v.get("split_role") if isinstance(v, dict) else None
        if not _is_hex64(cid) or not isinstance(fam, str) or not fam or role not in PARTITIONS + EXCLUDED_ROLES:
            raise ReconcileError(f"partition map entry {str(cid)[:16]} needs a 64-hex key, a family_id and a split_role in "
                                 f"{PARTITIONS + EXCLUDED_ROLES}", kind="input")
        out[cid] = {"family_id": fam, "split_role": role}
    return out, {"path": None if path is None else str(path), "bytes": len(raw), "sha256": common.sha256_bytes(raw), "entries": len(out)}


def _load_seeds(src: Path | str | Iterable[str]) -> tuple[frozenset[str], dict]:
    if isinstance(src, (str, os.PathLike)):
        path = Path(src)
        raw = path.read_bytes()
        data = _loads(raw, str(path))
    else:
        path, data = None, sorted(src)
        raw = common.canonical(data)
    if isinstance(data, dict):
        data = data.get("seed_hashes")
    if not isinstance(data, list) or not all(_is_hex64(h) for h in data):
        raise ReconcileError("seed hashes must be a JSON list of 64-hex program hashes (or {\"seed_hashes\": [...]})", kind="input")
    hashes = frozenset(data)
    return hashes, {"path": None if path is None else str(path), "bytes": len(raw), "sha256": common.sha256_bytes(raw),
                    "count": len(hashes), "hashes": sorted(hashes)}


def load_seed_hashes(path: Path | str) -> frozenset[str]:
    return _load_seeds(path)[0]


@dataclass(frozen=True)
class _Plan:
    contexts: tuple[ContextRef, ...]
    excluded: dict
    epoch_pin: dict
    map_pin: dict


def _context_ref(e: dict, m: dict) -> ContextRef:
    cid = e["context_sha256"]
    native, hist = e.get("native_source"), e.get("history_ref")
    ok = (isinstance(native, dict) and isinstance(native.get("path"), str) and _is_count(native.get("bytes"))
          and _is_hex64(native.get("sha256")) and isinstance(hist, dict) and _is_hex64(hist.get("sha256"))
          and isinstance(e.get("view_id"), str) and isinstance(e.get("dataset_hash"), str) and _is_count(e.get("fold"))
          and (e["cohort"] == "frozen" or isinstance(e.get("item_id"), str)))
    if not ok:
        raise ReconcileError("epoch context lacks native_source {path,bytes,sha256}, history_ref.sha256, view_id, "
                             "dataset_hash, fold or (incoming) item_id", cid, "input")
    return ContextRef(context_sha256=cid, cohort=e["cohort"], view_id=e["view_id"], item_id=e.get("item_id"),
                      dataset_hash=e["dataset_hash"], fold=e["fold"],
                      native={"path": native["path"], "bytes": native["bytes"], "sha256": native["sha256"]},
                      prepared=dict(e.get("prepared_ref") or {}), history_sha256=hist["sha256"],
                      family_id=m["family_id"], split_role=m["split_role"])


def _plan(epoch: Path | str, partition_map: Path | str | Mapping) -> _Plan:
    epath = Path(epoch)
    raw = epath.read_bytes()
    doc = _loads(raw, str(epath))
    entries = doc.get("contexts") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        raise ReconcileError(f"{epath} has no contexts list", kind="input")
    pmap, map_pin = _load_partition_map(partition_map)
    contexts, excluded, missing, seen = [], Counter(), [], set()
    for e in entries:
        cid = e.get("context_sha256") if isinstance(e, dict) else None
        if not _is_hex64(cid) or cid in seen or e.get("cohort") not in COHORTS:
            raise ReconcileError(f"epoch context {str(cid)[:16]} is malformed or repeated", kind="input")
        seen.add(cid)
        if e.get("status") != "included":
            excluded[f"{e['cohort']}:epoch:{e.get('exclusion_reason') or e.get('status')}"] += 1
            continue
        m = pmap.get(cid)
        if m is None:
            missing.append(cid)
        elif m["split_role"] in EXCLUDED_ROLES:
            excluded[f"{e['cohort']}:role:{m['split_role']}"] += 1
        else:
            if m["split_role"] == "diag" and e["cohort"] == "frozen" and e.get("fold") == 0:
                raise ReconcileError("a frozen fold-0 context is mapped to diag: PLAN §6.2 excludes them from the split "
                                     "diagnostic (folds 1-4 carry 16 champions selected on fold 0's outer test)", cid,
                                     "closure")
            contexts.append(_context_ref(e, m))
    if missing:
        raise ReconcileError(f"{len(missing)} included epoch contexts have no partition-map entry (first {missing[0][:16]})",
                             kind="closure")
    contexts.sort(key=lambda c: (PARTITIONS.index(c.split_role), c.context_sha256))
    epoch_pin = {"path": str(epath), "bytes": len(raw), "sha256": common.sha256_bytes(raw),
                 "experiment_id": doc.get("experiment_id"), "epoch": doc.get("epoch"), "contexts": len(entries)}
    return _Plan(tuple(contexts), {"counts": dict(sorted(excluded.items())), "map_entries_outside_epoch": len(set(pmap) - seen)},
                 epoch_pin, map_pin)


def load_contexts(epoch: Path | str, partition_map: Path | str | Mapping) -> list[ContextRef]:
    """Included epoch contexts with a partition role, ordered by (PARTITIONS order, context_sha256).

    Every included context needs a partition-map entry (closure). Epoch-excluded contexts and sealed or
    prohibited ones are only counted.
    """
    return list(_plan(epoch, partition_map).contexts)


# ------------------------------------------------------------------------------------------- history summaries

CURSOR_PROJECTION_KEYS = ("stats", "source_ref_sha256", "fingerprint", "generation_zero_source_sha256",
                          "view_context_sha256")


class _ProjectedCursorEntry(dict):
    """A cursor entry already reduced to the allowlist. JSON never produces this type, so whatever keys a raw entry
    carries (a flat source_ref_sha256 included) it is always re-projected: no producer-side schema change can switch
    the projection or the history pin off."""


def _project_cursor_entry(e: Mapping) -> dict:
    """Allowlisted projection of one fold-archive-cursor-v2 entry (exactly CURSOR_PROJECTION_KEYS, stats exactly
    SUMMARY_KEYS); the history pin comes only from source_ref.sha256. stats.best_test_rmse and view.tabpfn35 are
    never touched."""
    if type(e) is _ProjectedCursorEntry:
        return e
    stats = e.get("stats") if isinstance(e.get("stats"), dict) else {}
    src = e.get("source_ref") if isinstance(e.get("source_ref"), dict) else {}
    view = e.get("view") if isinstance(e.get("view"), dict) else {}
    return _ProjectedCursorEntry(stats={k: stats.get(k) for k in SUMMARY_KEYS}, source_ref_sha256=src.get("sha256"),
                                 fingerprint=e.get("fingerprint"),
                                 generation_zero_source_sha256=e.get("generation_zero_source_sha256"),
                                 view_context_sha256=view.get("context_sha256"))


def load_incoming_cursor(path: Path | str) -> tuple[dict[str, dict], dict]:
    """Read the live incoming cursor once: projected entries plus {path, bytes, sha256, read_utc_us}."""
    path = Path(path)
    raw = path.read_bytes()
    read_us = common.utc_us()
    doc = _loads(raw, str(path))
    if not isinstance(doc, dict):
        raise ReconcileError(f"{path} is not a JSON object", kind="summary")
    out = {cid: _project_cursor_entry(e) for cid, e in doc.items() if isinstance(e, dict)}
    return out, {"path": str(path), "layout": "live_snapshot", "bytes": len(raw), "sha256": common.sha256_bytes(raw), "read_utc_us": read_us}


def _make_summary(values: Mapping, history_sha: Any, ctx: ContextRef, where: str,
                  fingerprint: Any = None, boot: Any = None) -> HistorySummary:
    bad = [k for k in SUMMARY_KEYS if not _is_count(values.get(k))]
    if bad:
        raise ReconcileError(f"{where}: summary lacks non-negative integers {bad}", ctx.context_sha256, "summary")
    if history_sha != ctx.history_sha256:
        raise ReconcileError(f"{where}: history sha256 {str(history_sha)[:16]} != epoch pin {ctx.history_sha256[:16]}",
                             ctx.context_sha256, "summary")
    if ctx.cohort == "incoming" and not (_is_hex64(fingerprint) and _is_hex64(boot)):
        raise ReconcileError(f"{where}: incoming summary lacks fingerprint or generation_zero_source_sha256", ctx.context_sha256, "summary")
    return HistorySummary(**{k: values[k] for k in SUMMARY_KEYS}, history_sha256=history_sha,
                          fingerprint=fingerprint if ctx.cohort == "incoming" else None,
                          bootstrap_source_sha256=boot if ctx.cohort == "incoming" else None)


def _summary(ctx: ContextRef, cursor: Mapping[str, Any] | None, cursor_pin: dict | None = None) -> tuple[HistorySummary, dict]:
    if ctx.cohort == "frozen":
        path = _ctx_dir(ctx) / "history-ref.json"
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise ReconcileError(f"{path} is missing", ctx.context_sha256, "summary") from None
        doc = _loads(raw, str(path), ctx.context_sha256)
        if not isinstance(doc, dict):
            raise ReconcileError(f"{path} is not a JSON object", ctx.context_sha256, "summary")
        href = doc.get("history_ref") if isinstance(doc.get("history_ref"), dict) else {}
        if (doc.get("context_sha256"), doc.get("view_id")) != (ctx.context_sha256, ctx.view_id):
            raise ReconcileError(f"{path} names another context or view", ctx.context_sha256, "summary")
        s = _make_summary({k: doc.get(k) for k in SUMMARY_KEYS}, href.get("sha256"), ctx, str(path))
        return s, {"path": str(path), "layout": "file_prefix", "bytes": len(raw), "sha256": common.sha256_bytes(raw)}
    if cursor is None:
        raise ReconcileError("an incoming summary needs the fold-archive cursor", ctx.context_sha256, "summary")
    e = cursor.get(ctx.context_sha256)
    if e is None:
        raise ReconcileError("context is absent from the incoming fold-archive cursor", ctx.context_sha256, "summary")
    e = _project_cursor_entry(e)
    if tuple(e) != CURSOR_PROJECTION_KEYS or tuple(e["stats"]) != SUMMARY_KEYS:
        raise ReconcileError("cursor entry projection is not the exact allowlist", ctx.context_sha256, "summary")
    if e["view_context_sha256"] != ctx.context_sha256:
        raise ReconcileError("cursor entry names another context", ctx.context_sha256, "summary")
    s = _make_summary(e["stats"], e["source_ref_sha256"], ctx, "fold-archive cursor", e["fingerprint"], e["generation_zero_source_sha256"])
    return s, dict(cursor_pin or {"layout": "live_snapshot"})


def history_summary(ctx: ContextRef, cursor: Mapping[str, Any] | None = None) -> HistorySummary:
    """Allowlisted summary: frozen from <ctxdir>/history-ref.json, incoming from the (raw or projected) cursor mapping."""
    return _summary(ctx, cursor)[0]


# --------------------------------------------------------------------------------------------- source scanning

def _scan_native(ctx: ContextRef, s: HistorySummary | None, extra: Sequence[Any] = ()) -> tuple[dict[str, tuple[str, str]], dict]:
    """program_sha256 -> (dsl, record_status) for the pinned native prefix; STOP on any pin, row or count mismatch."""
    cid, pin = ctx.context_sha256, ctx.native
    if s is not None and s.native_bytes != pin["bytes"]:
        raise ReconcileError(f"summary native_bytes {s.native_bytes} != epoch pin {pin['bytes']}", cid, "summary")
    reader = _Prefix(Path(pin["path"]), pin["bytes"], "native", cid, extra)
    rows: dict[str, tuple[str, str]] = {}
    n = 0
    for line in reader:
        n += 1
        where = f"native line {n}"
        r = _record(line, where, cid)
        prog = r.get("program")
        if r.get("context_sha256") != cid or not isinstance(prog, dict):
            raise ReconcileError(f"{where}: missing program or another context_sha256", cid, "source")
        dsl, sha, status = prog.get("dsl"), prog.get("program_sha256"), r.get("status")
        if status == "abandoned":
            reason = r.get("reason")
            if reason not in ABANDON_REASONS:
                raise ReconcileError(f"{where}: abandon reason {reason!r} is outside {ABANDON_REASONS}", cid, "row")
            status = f"abandoned:{reason}"
        elif status not in NATIVE_STATUSES:
            raise ReconcileError(f"{where}: status {status!r} is outside {NATIVE_STATUSES + ('abandoned',)}", cid, "row")
        _check_program(dsl, sha, where, cid)
        if sha in rows:
            raise ReconcileError(f"{where}: duplicate native program {sha[:16]}", cid, "row")
        rows[sha] = (dsl, status)
    if reader.sha.hexdigest() != pin["sha256"]:
        raise ReconcileError(f"native sha256 {reader.sha.hexdigest()} != epoch pin {pin['sha256']}", cid, "source")
    if s is not None and reader.lines != s.native_events:
        raise ReconcileError(f"native rows {reader.lines} != native_events {s.native_events}", cid, "count")
    return rows, reader.record("epoch_native_source")


class _OutcomeScan:
    """Bootstrap rows (last row wins, as in the packer) and every outcome-row program of one context."""

    def __init__(self, ctx: str):
        self.ctx = ctx
        self.boot: dict[str, tuple[str, str]] = {}
        self.boot_events = 0
        self.programs: set[str] = set()
        self.n = 0

    def add(self, line: bytes) -> None:
        self.n += 1
        where = f"outcome line {self.n}"
        if _BOOTSTRAP_RE.search(line) is not None:
            r = _record(line, where, self.ctx)
            prog = r.get("program")
            if r.get("phase") == "bootstrap" and prog:
                sha = r.get("program_sha256")
                if not isinstance(prog, dict) or r.get("context_sha256") != self.ctx or prog.get("program_sha256", sha) != sha:
                    raise ReconcileError(f"{where}: malformed bootstrap row", self.ctx, "source")
                _check_program(prog.get("dsl"), sha, where, self.ctx)
                status = r.get("record_status")
                if status not in BOOTSTRAP_STATUSES:
                    raise ReconcileError(f"{where}: bootstrap record_status {status!r} is outside {BOOTSTRAP_STATUSES}", self.ctx, "row")
                self.boot[sha] = (prog["dsl"], status)
                self.boot_events += 1
                self.programs.add(sha)
                return
        sha = _line_program_sha(line, where, self.ctx)
        if sha is not None:
            self.programs.add(sha)


@dataclass
class _Evidence:
    """Everything except the native rows that a context needs to assemble its rows."""
    boot: dict[str, tuple[str, str]]
    boot_events: int
    transport_only: dict[str, str]
    residual: int
    outcome_programs: int
    prefixes: dict[str, dict]
    boot_results: dict | None = None


def _boot_results_check(ctx: ContextRef, boot: Mapping[str, tuple[str, str]], body: bytes) -> dict:
    """bootstrap-results.jsonl (snapshot bytes) against the outcome prefix's bootstrap rows. The program sets must be
    equal (else STOP); each program's last row record_status is compared with the outcome row's (the outcome row
    wins) and disagreements are counted. Only program_sha256, record_status and context_sha256 are read."""
    cid = ctx.context_sha256
    last: dict[str, Any] = {}
    n = 0
    for line in body.splitlines():
        if not line.strip():
            continue
        n += 1
        r = _record(line, f"bootstrap-results line {n}", cid)
        sha = r.get("program_sha256")
        if not _is_hex64(sha) or r.get("context_sha256", cid) != cid:
            raise ReconcileError(f"bootstrap-results line {n}: no 64-hex program_sha256 or another context", cid,
                                 "bootstrap_results")
        last[sha] = r.get("record_status")
    if set(last) != set(boot):
        raise ReconcileError(f"bootstrap-results programs ({len(last)}) != bootstrap set ({len(boot)}): "
                             f"{len(set(last) - set(boot))} only in bootstrap-results, {len(set(boot) - set(last))} "
                             f"only in the outcome prefix", cid, "bootstrap_results")
    disagree = sum(1 for sha, (_dsl, st) in boot.items() if last[sha] != st)
    return {"programs": len(last), "rows": n, "disagreements": disagree}


def _scan_transport(ctx: ContextRef, s: HistorySummary, known: set[str], path: Path, *, extract: bool,
                    extra: Sequence[Any] = ()) -> tuple[dict[str, str], _Prefix]:
    """Transport-only programs (json.loads only for rows whose regex program_sha256 is not already known)."""
    cid = ctx.context_sha256
    reader = _Prefix(path, s.transport_bytes, "transport", cid, extra)
    found: dict[str, str] = {}
    n = 0
    for line in reader:
        n += 1
        if not extract:
            continue
        where = f"transport line {n}"
        sha = _line_program_sha(line, where, cid)
        if sha is None or sha in known or sha in found:
            continue
        r = _record(line, where, cid)
        prog = r.get("program")
        if r.get("_telemetry_view_context_sha256", r.get("context_sha256")) != cid:
            raise ReconcileError(f"{where}: transport row belongs to another fold", cid, "source")
        if not isinstance(prog, dict) or prog.get("program_sha256", sha) != sha:
            raise ReconcileError(f"{where}: malformed transport program", cid, "source")
        _check_program(prog.get("dsl"), sha, where, cid)
        found[sha] = prog["dsl"]
    if reader.lines != s.transport_events:
        raise ReconcileError(f"transport lines {reader.lines} != transport_events {s.transport_events}", cid, "count")
    return found, reader


def _finish_evidence(ctx: ContextRef, s: HistorySummary, native: Mapping[str, Any], scan: _OutcomeScan, outcome_rec: dict,
                     sources: Sources, fp: Any = None) -> _Evidence:
    cid = ctx.context_sha256
    known = set(native) | set(scan.boot)
    stray = scan.programs - known
    if stray:
        raise ReconcileError(f"{len(stray)} outcome-row programs are neither native nor bootstrap", cid, "count")
    residual = s.unique_programs - len(known)
    if residual < 0:
        raise ReconcileError(f"native ∪ bootstrap holds {len(known)} programs > unique_programs {s.unique_programs}", cid, "count")
    prefixes = {"outcome": outcome_rec}
    tpath = _transport_path(ctx, sources)
    transport_only: dict[str, str] = {}
    if residual > 0 or fp is not None:
        transport_only, reader = _scan_transport(ctx, s, known, tpath, extract=residual > 0, extra=() if fp is None else (fp,))
        prefixes["transport"] = reader.record("fingerprint" if fp is not None else "bytes+lines")
    else:
        prefixes["transport"] = {"path": str(tpath), "layout": "skipped", "bytes": s.transport_bytes,
                                 "lines": s.transport_events, "skipped": "residual_zero"}
    bpath = _ctx_dir(ctx) / "bootstrap-results.jsonl"
    boot = _snapshot(bpath)
    bsha = common.sha256_bytes(boot)
    if fp is not None:
        if bsha != s.bootstrap_source_sha256:
            raise ReconcileError("bootstrap-results snapshot sha256 != generation_zero_source_sha256", cid, "source")
        fp.update(boot)
        if fp.hexdigest() != s.fingerprint:
            raise ReconcileError("sha256(native ‖ outcome ‖ transport ‖ bootstrap-results) != archiver fingerprint", cid, "source")
    # frozen: the snapshot is the cross-check's own pin (verify --rehash-sources re-checks it)
    prefixes["bootstrap_results"] = {"path": str(bpath), "layout": "file_prefix", "bytes": len(boot), "sha256": bsha,
                                     "pin": "generation_zero_source_sha256" if fp is not None else "cross_check"}
    boot_results = _boot_results_check(ctx, scan.boot, boot)
    if len(known) + len(transport_only) != s.unique_programs:
        raise ReconcileError(f"native ∪ bootstrap ∪ transport-only = {len(known) + len(transport_only)} != "
                             f"unique_programs {s.unique_programs}", cid, "count")
    return _Evidence(boot=scan.boot, boot_events=scan.boot_events, transport_only=transport_only, residual=residual,
                     outcome_programs=len(scan.programs), prefixes=prefixes, boot_results=boot_results)


def _frozen_outcome(ctx: ContextRef, s: HistorySummary) -> tuple[_OutcomeScan, dict]:
    reader = _Prefix(_ctx_dir(ctx) / "history-outcomes.jsonl", s.outcome_bytes, "outcome", ctx.context_sha256)
    scan = _OutcomeScan(ctx.context_sha256)
    for line in reader:
        scan.add(line)
    if reader.lines != s.outcome_events:
        raise ReconcileError(f"outcome lines {reader.lines} != outcome_events {s.outcome_events}", ctx.context_sha256, "count")
    return scan, reader.record("bytes+lines")


def _frozen_evidence(ctx: ContextRef, s: HistorySummary, native: Mapping[str, Any], sources: Sources) -> _Evidence:
    scan, rec = _frozen_outcome(ctx, s)
    return _finish_evidence(ctx, s, native, scan, rec, sources)


class _IncomingState:
    """One incoming context while its item's hwm-events.jsonl is demuxed; the fingerprint hasher already holds native."""

    def __init__(self, ctx: ContextRef, s: HistorySummary, native: dict, fp: Any, events: Path):
        self.ctx, self.s, self.native, self.fp, self.events = ctx, s, native, fp, events
        self.scan = _OutcomeScan(ctx.context_sha256)
        self.sha = hashlib.sha256()
        self.bytes = self.lines = self.tail_lines = 0
        self.full = s.outcome_bytes == 0
        self.error: ReconcileError | None = None

    def feed(self, line: bytes) -> None:
        if self.error is not None:
            return
        if self.full:
            self.tail_lines += 1
            return
        if self.bytes + len(line) > self.s.outcome_bytes:
            self.error = ReconcileError(f"the recorded outcome prefix ({self.s.outcome_bytes} B) ends inside a line",
                                        self.ctx.context_sha256, "source")
            return
        self.bytes += len(line)
        self.lines += 1
        self.sha.update(line)
        self.fp.update(line)
        try:
            self.scan.add(line)
        except ReconcileError as err:
            self.error = err
        self.full = self.bytes == self.s.outcome_bytes

    def finish(self, sources: Sources) -> _Evidence:
        cid = self.ctx.context_sha256
        if self.error is not None:
            raise self.error
        if self.bytes != self.s.outcome_bytes or self.lines != self.s.outcome_events:
            raise ReconcileError(f"incoming outcome lines hold {self.bytes} B in {self.lines} lines; summary records "
                                 f"{self.s.outcome_bytes} B in {self.s.outcome_events}", cid, "source")
        rec = {"path": str(self.events), "layout": "ctx_lines", "bytes": self.bytes, "lines": self.lines,
               "sha256": self.sha.hexdigest(), "tail_lines": self.tail_lines, "pin": "fingerprint"}
        return _finish_evidence(self.ctx, self.s, self.native, self.scan, rec, sources, fp=self.fp)


def _incoming_evidence(pairs: Sequence[tuple[ContextRef, HistorySummary]], lines: Iterable[bytes], events: Path,
                       sources: Sources) -> dict[str, _Evidence | ReconcileError]:
    """Demux one item's events once for several contexts. Each context's fingerprint covers native ‖ outcome ‖ transport ‖ boot."""
    out: dict[str, _Evidence | ReconcileError] = {}
    states: dict[str, _IncomingState] = {}
    for ctx, s in pairs:
        try:
            fp = hashlib.sha256()
            native, _ = _scan_native(ctx, s, extra=(fp,))
            states[ctx.context_sha256] = _IncomingState(ctx, s, native, fp, events)
        except ReconcileError as err:
            out[ctx.context_sha256] = err
    if states:
        try:
            for line in lines:
                st = states.get(_line_context(line))
                if st is not None:
                    st.feed(line)
        except ReconcileError as err:
            out.update({cid: err for cid in states})
            return out
    for cid, st in states.items():
        try:
            out[cid] = st.finish(sources)
        except ReconcileError as err:
            out[cid] = err
    return out


def _seed_pool_detail(ctx: ContextRef) -> tuple[set[str], set[str], dict]:
    """(program_sha256 of every seed-pool row, the conditioned-seed subset, the prefix record). A row is a conditioned
    seed when its schema is CONDITIONED_SEED_SCHEMA (or it carries a source_context key, whose value is never read).
    rank, source_evidence and source_context values are never touched."""
    path = _ctx_dir(ctx) / "seed-pool.jsonl"
    if not path.exists():
        return set(), set(), {"path": str(path), "layout": "missing", "bytes": 0, "lines": 0}
    body = _snapshot(path)
    out: set[str] = set()
    cond: set[str] = set()
    n = 0
    for line in body.splitlines():
        if not line.strip():
            continue
        n += 1
        r = _record(line, f"seed-pool line {n}", ctx.context_sha256)
        prog = r.get("program")
        if not isinstance(prog, dict):
            raise ReconcileError(f"seed-pool line {n} lacks program", ctx.context_sha256, "source")
        _check_program(prog.get("dsl"), prog.get("program_sha256"), f"seed-pool line {n}", ctx.context_sha256)
        out.add(prog["program_sha256"])
        if r.get("schema") == CONDITIONED_SEED_SCHEMA or "source_context" in r:
            cond.add(prog["program_sha256"])
    return out, cond, {"path": str(path), "layout": "file_prefix", "bytes": len(body), "lines": n,
                       "sha256": common.sha256_bytes(body), "pin": "cross_check"}


def _seed_pool(ctx: ContextRef) -> tuple[set[str], dict]:
    """program.program_sha256 of every seed-pool row (rank and source_evidence are never touched)."""
    pool, _cond, rec = _seed_pool_detail(ctx)
    return pool, rec


def _assemble(ctx: ContextRef, s: HistorySummary, native: Mapping[str, tuple[str, str]], native_rec: dict, ev: _Evidence,
              seeds: frozenset[str]) -> tuple[list[ProgramRow], dict]:
    cid = ctx.context_sha256
    pool, cond, pool_rec = _seed_pool_detail(ctx)
    boot = set(ev.boot)
    if boot != pool:
        raise ReconcileError(f"bootstrap set ({len(boot)}) != seed-pool set ({len(pool)}): {len(boot - pool)} only in outcomes, "
                             f"{len(pool - boot)} only in seed-pool", cid, "seed_pool")
    rows = [ProgramRow(cid, sha, dsl, "evolution", st, sha in cond) for sha, (dsl, st) in native.items()]
    overlap = 0
    for sha, (dsl, st) in ev.boot.items():
        if sha in native:
            overlap += 1
        else:
            rows.append(ProgramRow(cid, sha, dsl, "bootstrap", st, sha in cond))
    rows.extend(ProgramRow(cid, sha, dsl, "transport_only", "transport_only") for sha, dsl in ev.transport_only.items())
    rows.sort(key=lambda r: r.program_sha256)
    if len({r.program_sha256 for r in rows}) != len(rows):
        raise ReconcileError("duplicate (context, program) rows", cid, "row")
    if len(rows) != s.unique_programs:
        raise ReconcileError(f"rows {len(rows)} != unique_programs {s.unique_programs}", cid, "count")
    rec = _context_record(ctx, s, rows, seeds, source="local")
    rec.update({"native_rows": len(native), "bootstrap_set": len(boot), "bootstrap_events": ev.boot_events,
                "bootstrap_overlap_native": overlap, "residual_before_transport": ev.residual,
                "outcome_programs": ev.outcome_programs, "seed_pool": len(pool), "conditioned_seeds": len(cond),
                "bootstrap_results": ev.boot_results,
                "prefixes": {"native": native_rec, **ev.prefixes, "seed_pool": pool_rec}})
    return rows, rec


def _context_record(ctx: ContextRef, s: HistorySummary, rows: Sequence[ProgramRow], seeds: frozenset[str], *, source: str) -> dict:
    by_origin = Counter(r.origin for r in rows)
    return {"context_sha256": ctx.context_sha256, "cohort": ctx.cohort, "fold": ctx.fold, "view_id": ctx.view_id,
            "family_id": ctx.family_id, "split_role": ctx.split_role, "source": source, "summary": dataclasses.asdict(s),
            "rows": len(rows), "bootstrap_rows": by_origin["bootstrap"], "transport_only_rows": by_origin["transport_only"],
            "shared_seed_rows": sum(r.program_sha256 in seeds for r in rows),
            "conditioned_seed_rows": sum(r.conditioned_seed for r in rows),
            "by_origin": dict(sorted(by_origin.items())),
            "by_status": dict(sorted(Counter(r.record_status for r in rows).items())),
            "by_form": dict(sorted(Counter(program_form(r.dsl) for r in rows).items()))}


# --------------------------------------------------------------------------------------------- public scanners

def iter_native(ctx: ContextRef, s: HistorySummary | None = None) -> Iterator[ProgramRow]:
    """Native rows (origin=evolution), sha-sorted. Rows are yielded only after bytes, sha256 and counts verify."""
    rows, _ = _scan_native(ctx, s)
    for sha in sorted(rows):
        yield ProgramRow(ctx.context_sha256, sha, rows[sha][0], "evolution", rows[sha][1])


def _item_events(ctx: ContextRef) -> Path:
    return _item_gp(ctx) / "hwm-events.jsonl"


def iter_bootstrap(ctx: ContextRef, s: HistorySummary, item_events: Sequence[bytes] | None = None, *,
                   sources: Sources = DEFAULT_SOURCES) -> Iterator[ProgramRow]:
    """Bootstrap programs of the outcome prefix (origin=bootstrap, even when also native), sha-sorted.

    Frozen: bytes and lines of history-outcomes.jsonl. Incoming: `item_events` are lines of the item's hwm-events.jsonl
    (read from disk when None); the full fingerprint is verified before anything is yielded.
    """
    if ctx.cohort == "frozen":
        if item_events is not None:
            raise ValueError("item_events applies to incoming contexts only")
        boot = _frozen_outcome(ctx, s)[0].boot
    else:
        path = _item_events(ctx)
        lines = item_events if item_events is not None else _complete_lines(path)
        ev = _incoming_evidence([(ctx, s)], lines, path, sources)[ctx.context_sha256]
        if isinstance(ev, ReconcileError):
            raise ev
        boot = ev.boot
    for sha in sorted(boot):
        yield ProgramRow(ctx.context_sha256, sha, boot[sha][0], "bootstrap", boot[sha][1])


def iter_transport_only(ctx: ContextRef, s: HistorySummary, known: set[str], *,
                        sources: Sources = DEFAULT_SOURCES) -> Iterator[ProgramRow]:
    """Transport rows whose program is not in `known`, deduplicated and sha-sorted, after bytes and lines verify.

    The incoming transport prefix is sha-verified by the context fingerprint in context_rows() and build().
    """
    found, _ = _scan_transport(ctx, s, set(known), _transport_path(ctx, sources), extract=True)
    for sha in sorted(found):
        yield ProgramRow(ctx.context_sha256, sha, found[sha], "transport_only", "transport_only")


def context_rows(ctx: ContextRef, s: HistorySummary, *, seeds: Iterable[str], item_events: Sequence[bytes] | None = None,
                 sources: Sources = DEFAULT_SOURCES, evidence: _Evidence | None = None) -> tuple[list[ProgramRow], dict]:
    """All rows of one context, sha-sorted, plus its reconciliation record (counts and prefix pins only). Raises ReconcileError."""
    seeds = frozenset(seeds)
    native, native_rec = _scan_native(ctx, s)
    if evidence is None:
        if ctx.cohort == "frozen":
            evidence = _frozen_evidence(ctx, s, native, sources)
        else:
            path = _item_events(ctx)
            lines = item_events if item_events is not None else _complete_lines(path)
            got = _incoming_evidence([(ctx, s)], lines, path, sources)[ctx.context_sha256]
            if isinstance(got, ReconcileError):
                raise got
            evidence = got
    return _assemble(ctx, s, native, native_rec, evidence, seeds)


# ------------------------------------------------------------------------------------------------------ writing

def _dict_array(values: list, domain: Sequence[str]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(values, type=pa.int8()), pa.array(domain, type=pa.string()))


def _rows_table(per_ctx: Sequence[tuple[ContextRef, Sequence[ProgramRow]]], seeds: frozenset[str]) -> pa.Table:
    """The published chunk (ROW_SCHEMA): no origin, the origin-independent published_status."""
    idx = {name: {v: i for i, v in enumerate(vals)} for name, vals in DICT_COLUMNS.items()}
    cols: dict[str, list] = {f.name: [] for f in ROW_SCHEMA}
    for ctx, rows in per_ctx:
        cb = bytes.fromhex(ctx.context_sha256)
        role, cohort = idx["split_role"][ctx.split_role], idx["cohort"][ctx.cohort]
        for r in rows:
            if r.context_sha256 != ctx.context_sha256:
                raise ReconcileError("row of another context in chunk input", ctx.context_sha256, "row")
            cols["context_sha256"].append(cb)
            cols["program_sha256"].append(bytes.fromhex(r.program_sha256))
            cols["dsl"].append(r.dsl)
            cols["family_id"].append(ctx.family_id)
            cols["split_role"].append(role)
            cols["record_status"].append(idx["record_status"][published_status(r.record_status)])
            cols["shared_seed"].append(r.program_sha256 in seeds)
            cols["cohort"].append(cohort)
            cols["form"].append(idx["form"][program_form(r.dsl)])
            cols["dsl_bytes"].append(len(r.dsl.encode("utf-8")))
    arrays = [_dict_array(cols[f.name], DICT_COLUMNS[f.name]) if f.name in DICT_COLUMNS else pa.array(cols[f.name], type=f.type)
              for f in ROW_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=ROW_SCHEMA)


def _sealed_table(per_ctx: Sequence[tuple[ContextRef, Sequence[ProgramRow]]]) -> pa.Table:
    """The sealed side table (SEALED_SCHEMA), row-aligned with _rows_table over the same input."""
    idx = {name: {v: i for i, v in enumerate(vals)} for name, vals in SEALED_DICT_COLUMNS.items()}
    cols: dict[str, list] = {f.name: [] for f in SEALED_SCHEMA}
    for ctx, rows in per_ctx:
        cb = bytes.fromhex(ctx.context_sha256)
        for r in rows:
            if r.origin not in STATUSES_BY_ORIGIN or r.record_status not in STATUSES_BY_ORIGIN[r.origin]:
                raise ReconcileError(f"row origin/status {r.origin}/{r.record_status}", ctx.context_sha256, "row")
            cols["context_sha256"].append(cb)
            cols["program_sha256"].append(bytes.fromhex(r.program_sha256))
            cols["origin"].append(idx["origin"][r.origin])
            cols["record_status"].append(idx["record_status"][r.record_status])
            cols["conditioned_seed"].append(bool(r.conditioned_seed))
    arrays = [_dict_array(cols[f.name], SEALED_DICT_COLUMNS[f.name]) if f.name in SEALED_DICT_COLUMNS
              else pa.array(cols[f.name], type=f.type) for f in SEALED_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=SEALED_SCHEMA)


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pq.ParquetWriter(sink, table.schema, compression="zstd", compression_level=ZSTD_LEVEL,
                          use_dictionary=True, write_statistics=True) as w:
        w.write_table(table, row_group_size=ROW_GROUP_ROWS)
    return sink.getvalue().to_pybytes()


def _publish_immutable(path: Path, data: bytes) -> dict:
    """Create-only and crash-safe: temp file + fsync + link(2). Equal existing bytes are success, others raise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = common.sha256_bytes(data)
    if path.exists():
        if path.read_bytes() != data:
            raise common.ImmutableConflict(f"{path} exists with different bytes")
        return {"path": str(path), "sha256": digest, "bytes": len(data), "existed": True}
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o444)
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise common.ImmutableConflict(f"{path} appeared with different bytes") from None
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return {"path": str(path), "sha256": digest, "bytes": len(data), "existed": False}


def _write_cursor(out: Path, cursor: dict) -> None:
    common.atomic_write(out / "cursor.json", common.canonical(cursor) + b"\n")


def _chunk_name(i: int) -> str:
    return f"part-{i:05d}.parquet"


def _layout(contexts: Sequence[ContextRef], chunk_contexts: int) -> dict[str, list[list[ContextRef]]]:
    out = {}
    for p in PARTITIONS:
        members = [c for c in contexts if c.split_role == p]
        out[p] = [members[i:i + chunk_contexts] for i in range(0, len(members), chunk_contexts)]
    return out


def _schema_sha256(schema: pa.Schema = ROW_SCHEMA) -> str:
    return common.sha256_bytes(schema.to_string(show_schema_metadata=False).encode())


def _sealed_path(rel: str) -> str:
    """Published chunk path <partition>/part-NNNNN.parquet -> its sealed twin sealed/<partition>/part-NNNNN.parquet."""
    return f"{SEALED_DIR}/{rel}"


def _stray_files(out: Path, layout: dict[str, list[list[ContextRef]]]) -> list[str]:
    """Files in the published or sealed partition directories that are not chunk names of this layout (dot-prefixed
    temporaries excepted)."""
    bad = []
    for p in PARTITIONS:
        names = {_chunk_name(i) for i in range(len(layout[p]))}
        for d in (out / p, out / SEALED_DIR / p):
            if d.is_dir():
                bad.extend(str(x.relative_to(out)) for x in sorted(d.iterdir())
                           if not x.name.startswith(".") and (x.name not in names or not x.is_file()))
    sealed_root = out / SEALED_DIR
    if sealed_root.is_dir():
        bad.extend(str(x.relative_to(out)) for x in sorted(sealed_root.iterdir())
                   if not x.name.startswith(".") and x.name not in PARTITIONS)
    return bad


class _IncomingCache:
    """Read each item's hwm-events.jsonl once for all of its pending contexts; keep only compact per-context evidence."""

    def __init__(self, pending: Sequence[ContextRef], cursor: Mapping[str, Any] | None, sources: Sources):
        self.cursor, self.sources = cursor, sources
        self.by_item: dict[Path, list[ContextRef]] = defaultdict(list)
        for c in pending:
            try:
                self.by_item[_item_gp(c)].append(c)
            except ReconcileError:
                pass            # raised again when the context itself is processed
        self.ready: dict[str, _Evidence | ReconcileError] = {}
        self.used: set[str] = set()

    def get(self, ctx: ContextRef) -> _Evidence:
        cid = ctx.context_sha256
        if cid not in self.ready:
            gp = _item_gp(ctx)
            pairs = []
            for c in self.by_item.get(gp) or [ctx]:
                if c.context_sha256 in self.used or c.context_sha256 in self.ready:
                    continue
                try:
                    pairs.append((c, _summary(c, self.cursor)[0]))
                except ReconcileError as err:
                    self.ready[c.context_sha256] = err
            events = gp / "hwm-events.jsonl"
            self.ready.update(_incoming_evidence(pairs, _complete_lines(events), events, self.sources))
        self.used.add(cid)
        got = self.ready.pop(cid)
        if isinstance(got, ReconcileError):
            raise got
        return got


def _fallback(ctx: ContextRef, err: ReconcileError, hook: Callable, seeds: frozenset[str]) -> tuple[list[ProgramRow], dict]:
    """Rows from the caller's exact-version history hook, held to the same row invariants and to its own summary."""
    rows, s = hook(ctx, err)
    cid = ctx.context_sha256
    if not isinstance(s, HistorySummary) or s.history_sha256 != ctx.history_sha256:
        raise ReconcileError("s3_fallback summary is not the epoch-pinned history", cid, "fallback")
    rows = sorted(rows, key=lambda r: r.program_sha256)
    for r in rows:
        if r.context_sha256 != cid or r.origin not in ORIGINS or r.record_status not in STATUSES_BY_ORIGIN[r.origin]:
            raise ReconcileError("s3_fallback row has another context, origin or status", cid, "fallback")
        _check_program(r.dsl, r.program_sha256, "s3_fallback row", cid)
    if len({r.program_sha256 for r in rows}) != len(rows) or len(rows) != s.unique_programs:
        raise ReconcileError(f"s3_fallback rows {len(rows)} != unique_programs {s.unique_programs} (or duplicates)", cid, "fallback")
    rec = _context_record(ctx, s, rows, seeds, source="s3_fallback")
    rec.update({"local_error": f"{err.kind}: {err}"[:400], "prefixes": {}})
    return rows, rec


def build(epoch: Path | str, partition_map: Path | str | Mapping, seeds: Path | str | Iterable[str], out_dir: Path | str, *,
          chunk_contexts: int = CHUNK_CONTEXTS, resume: bool = True, sources: Sources = DEFAULT_SOURCES,
          s3_fallback: Callable[[ContextRef, ReconcileError], tuple[list[ProgramRow], HistorySummary]] | None = None) -> dict:
    """Build every partition's immutable chunks, resumably, then publish manifest.json and return it. STOPs raise ReconcileError.

    One writer per output directory: an exclusive flock on OUT/build.lock is held for the whole build.
    """
    if chunk_contexts < 1:
        raise ValueError("chunk_contexts must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "build.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ReconcileError(f"another build holds {out / 'build.lock'}", kind="cursor") from None
        return _build_locked(epoch, partition_map, seeds, out, chunk_contexts=chunk_contexts, resume=resume, sources=sources,
                             s3_fallback=s3_fallback)


def _boot_results_budget(n_contexts: int) -> int:
    """Disagreements a build tolerates: the design's observed rate (one per 2,864 contexts), at least one."""
    return max(1, -(-n_contexts // BOOT_RESULTS_CONTEXTS_PER_DISAGREEMENT))


def _disagreements(rec: Mapping) -> int:
    return int((rec.get("bootstrap_results") or {}).get("disagreements", 0))


def _build_locked(epoch, partition_map, seeds, out: Path, *, chunk_contexts: int, resume: bool, sources: Sources,
                  s3_fallback: Callable | None) -> dict:
    plan = _plan(epoch, partition_map)
    seed_set, seeds_pin = _load_seeds(seeds)
    pins = {"epoch_sha256": plan.epoch_pin["sha256"], "partition_map_sha256": plan.map_pin["sha256"],
            "seeds_sha256": seeds_pin["sha256"], "chunk_contexts": chunk_contexts, "row_schema_sha256": _schema_sha256(),
            "sealed_schema_sha256": _schema_sha256(SEALED_SCHEMA),
            "row_group_rows": ROW_GROUP_ROWS, "zstd_level": ZSTD_LEVEL, "pyarrow": pa.__version__,
            "sources": {"incoming_cursor": str(sources.incoming_cursor), "incoming_transport_index": str(sources.incoming_transport_index)}}
    layout = _layout(plan.contexts, chunk_contexts)
    cpath = out / "cursor.json"
    if cpath.exists():
        if not resume:
            raise ReconcileError(f"{cpath} exists and resume is off; chunks are immutable, so use a new --out", kind="cursor")
        cursor = _record(cpath.read_bytes(), str(cpath))
        if cursor.get("schema") != CURSOR_SCHEMA or cursor.get("pins") != pins:
            raise ReconcileError(f"{cpath} was written for other inputs or settings", kind="cursor")
    else:
        if not resume and any((out / d).exists() for d in (*PARTITIONS, SEALED_DIR)):
            raise ReconcileError(f"{out} holds partition directories but no cursor", kind="cursor")
        stray = _stray_files(out, layout)
        if stray:          # stale chunks of an earlier layout would stay on disk unlisted
            raise ReconcileError(f"{out} has no cursor but holds {len(stray)} files that are not chunks of this layout "
                                 f"(first {stray[0]})", kind="cursor")
        cursor = {"schema": CURSOR_SCHEMA, "pins": pins, "chunks": {p: {} for p in PARTITIONS}, "contexts": {},
                  "incoming_cursor_reads": [], "started_utc_us": common.utc_us(), "completed_utc_us": None}
    budget = _boot_results_budget(len(plan.contexts))
    disagreements = sum(_disagreements(r) for r in cursor["contexts"].values())
    pending = [c for p in PARTITIONS for i, chunk in enumerate(layout[p]) if _chunk_name(i) not in cursor["chunks"][p] for c in chunk]
    inc_cursor, inc_pin = None, None
    if any(c.cohort == "incoming" for c in pending):
        inc_cursor, inc_pin = load_incoming_cursor(sources.incoming_cursor)
        cursor["incoming_cursor_reads"].append(inc_pin)
    cache = _IncomingCache([c for c in pending if c.cohort == "incoming"], inc_cursor, sources)
    for p in PARTITIONS:
        for i, chunk in enumerate(layout[p]):
            rel = f"{p}/{_chunk_name(i)}"
            done = cursor["chunks"][p].get(_chunk_name(i))
            if done is not None:
                f, sf = out / rel, out / done["sealed"]["path"]
                if (done["contexts"] != [c.context_sha256 for c in chunk] or not f.exists() or common.sha256_file(f) != done["sha256"]
                        or not sf.exists() or common.sha256_file(sf) != done["sealed"]["sha256"]):
                    raise ReconcileError(f"recorded chunk {rel} (or its sealed twin) is missing or differs from the cursor",
                                         kind="cursor")
                continue
            per_ctx, recs = [], []
            for ctx in chunk:
                try:
                    s, spin = _summary(ctx, inc_cursor, inc_pin)
                    ev = cache.get(ctx) if ctx.cohort == "incoming" else None
                    rows, rec = context_rows(ctx, s, seeds=seed_set, sources=sources, evidence=ev)
                    rec["prefixes"]["summary"] = spin
                except ReconcileError as err:
                    if s3_fallback is None:
                        raise
                    rows, rec = _fallback(ctx, err, s3_fallback, seed_set)
                disagreements += _disagreements(rec)
                if disagreements > budget:
                    raise ReconcileError(f"bootstrap-results disagree with the outcome rows {disagreements} times in this "
                                         f"build, above the tolerated {budget} (one per {BOOT_RESULTS_CONTEXTS_PER_DISAGREEMENT}"
                                         f" contexts)", ctx.context_sha256, "bootstrap_results")
                rec["chunk"] = rel
                per_ctx.append((ctx, rows))
                recs.append(rec)
            table = _rows_table(per_ctx, seed_set)
            sealed = _sealed_table(per_ctx)
            sres = _publish_immutable(out / _sealed_path(rel), _parquet_bytes(sealed))
            res = _publish_immutable(out / rel, _parquet_bytes(table))
            cursor["chunks"][p][_chunk_name(i)] = {"path": rel, "sha256": res["sha256"], "bytes": res["bytes"],
                                                   "rows": table.num_rows, "contexts": [c.context_sha256 for c in chunk],
                                                   "sealed": {"path": _sealed_path(rel), "sha256": sres["sha256"],
                                                              "bytes": sres["bytes"], "rows": sealed.num_rows}}
            cursor["contexts"].update({r["context_sha256"]: r for r in recs})
            _write_cursor(out, cursor)
    if cursor.get("completed_utc_us") is None:
        cursor["completed_utc_us"] = common.utc_us()
        _write_cursor(out, cursor)
    manifest = _manifest(out, cursor, plan, layout, seeds_pin)
    _publish_immutable(out / "manifest.json", common.canonical(manifest) + b"\n")
    return manifest


# ----------------------------------------------------------------------------------------------- manifest/stats

PUBLISHED_STAT_COLUMNS = ("record_status", "form", "cohort", "shared_seed")
SEALED_STAT_COLUMNS = ("origin", "record_status", "conditioned_seed")


def _partition_stats(files: Sequence[Path], schema: pa.Schema = ROW_SCHEMA, cols: Sequence[str] = PUBLISHED_STAT_COLUMNS) -> dict:
    if not files:
        return {"rows": 0, "distinct_programs": 0, "by": {c: {} for c in cols}}
    t = pads.dataset([str(f) for f in files], format="parquet", schema=schema).to_table(
        columns=["program_sha256", *cols], use_threads=False)
    out = {"rows": t.num_rows, "distinct_programs": pc.count_distinct(t.column("program_sha256")).as_py(), "by": {}}
    for c in cols:
        g = t.group_by(c, use_threads=False).aggregate([("program_sha256", "count"), ("program_sha256", "count_distinct")])
        keys = [str(k).lower() if isinstance(k, bool) else k for k in g.column(c).to_pylist()]
        rows = g.column("program_sha256_count").to_pylist()
        distinct = g.column("program_sha256_count_distinct").to_pylist()
        out["by"][c] = {k: {"rows": n, "distinct_programs": d} for k, n, d in sorted(zip(keys, rows, distinct))}
    return out


def _manifest(out: Path, cursor: dict, plan: _Plan, layout: dict, seeds_pin: dict) -> dict:
    recs = cursor["contexts"]
    parts, sealed_parts = {}, {}
    for p in PARTITIONS:
        entries = [cursor["chunks"][p][_chunk_name(i)] for i in range(len(layout[p]))]
        chunks = [{k: v for k, v in e.items() if k != "sealed"} for e in entries]
        stats = _partition_stats([out / c["path"] for c in chunks])
        members = [recs[c.context_sha256] for chunk in layout[p] for c in chunk]
        if stats["rows"] != sum(r["rows"] for r in members) or stats["rows"] != sum(c["rows"] for c in chunks):
            raise ReconcileError(f"partition {p}: chunk rows {stats['rows']} disagree with the per-context records", kind="count")
        parts[p] = {"contexts": len(members), "chunks": chunks, **stats}
        schunks = [{**e["sealed"], "published": e["path"], "contexts": e["contexts"]} for e in entries]
        sstats = _partition_stats([out / c["path"] for c in schunks], SEALED_SCHEMA, SEALED_STAT_COLUMNS)
        if sstats["rows"] != stats["rows"]:
            raise ReconcileError(f"partition {p}: sealed rows {sstats['rows']} != published rows {stats['rows']}", kind="count")
        cond = sstats["by"]["conditioned_seed"].get("true", {"rows": 0, "distinct_programs": 0})
        sealed_parts[p] = {"chunks": schunks, **sstats, "conditioned_rows": cond["rows"],
                           "conditioned_distinct_programs": cond["distinct_programs"]}
    by_cohort = {}
    for coh in COHORTS:
        rs = [r for r in recs.values() if r["cohort"] == coh]
        by_cohort[coh] = {"contexts": len(rs), "rows": sum(r["rows"] for r in rs),
                          "unique_programs": sum(r["summary"]["unique_programs"] for r in rs),
                          "native_rows": sum(r["by_origin"].get("evolution", 0) for r in rs),
                          "bootstrap_rows": sum(r["bootstrap_rows"] for r in rs),
                          "transport_only_rows": sum(r["transport_only_rows"] for r in rs),
                          "transport_only_contexts": sum(1 for r in rs if r["transport_only_rows"]),
                          "conditioned_seed_rows": sum(r.get("conditioned_seed_rows", 0) for r in rs)}
    rows = sum(r["rows"] for r in recs.values())
    unique = sum(r["summary"]["unique_programs"] for r in recs.values())
    if rows != unique or len(recs) != len(plan.contexts):
        raise ReconcileError(f"Σ rows {rows} != Σ unique_programs {unique} (or context count differs)", kind="count")
    boot_results = {"checked_contexts": sum(1 for r in recs.values() if r.get("bootstrap_results")),
                    "disagreements": sum(_disagreements(r) for r in recs.values()),
                    "disagreeing_contexts": sorted(c for c, r in recs.items() if _disagreements(r)),
                    "tolerated": _boot_results_budget(len(plan.contexts))}
    return {"schema": MANIFEST_SCHEMA, "experiment_id": plan.epoch_pin.get("experiment_id"), "epoch": plan.epoch_pin.get("epoch"),
            "inputs": {"epoch": plan.epoch_pin, "partition_map": plan.map_pin, "seeds": seeds_pin,
                       "sources": cursor["pins"]["sources"], "incoming_cursor_reads": cursor["incoming_cursor_reads"]},
            "pins": cursor["pins"],
            "columns": [[f.name, str(f.type)] for f in ROW_SCHEMA],
            "parquet": {"compression": "zstd", "zstd_level": ZSTD_LEVEL, "row_group_rows": ROW_GROUP_ROWS,
                        "chunk_contexts": cursor["pins"]["chunk_contexts"], "pyarrow": pa.__version__},
            "read_roles": {"published": {**{r: [p] for p, r in PARTITION_ROLE.items()},
                                         **{r: {"partitions": list(PARTITIONS), "columns": list(c)} for r, c in PROJECTED_ROLES.items()}},
                           "sealed": {r: {"partitions": list(PARTITIONS), "columns": list(c) if c else [f.name for f in SEALED_SCHEMA]}
                                      for r, c in SEALED_ROLES.items()}},
            "partitions": parts,
            "sealed": {"columns": [[f.name, str(f.type)] for f in SEALED_SCHEMA], "partitions": sealed_parts,
                       "never_in_run_plan_closure": True},
            "contexts": recs,
            "totals": {"contexts": len(recs), "rows": rows, "unique_programs": unique, "rows_equal_unique_programs": True,
                       "by_cohort": by_cohort, "sources": dict(sorted(Counter(r["source"] for r in recs.values()).items())),
                       "conditioned_seed_rows": sum(r.get("conditioned_seed_rows", 0) for r in recs.values()),
                       "bootstrap_results": boot_results},
            "excluded": plan.excluded,
            "started_utc_us": cursor["started_utc_us"], "completed_utc_us": cursor["completed_utc_us"]}


def _load_manifest(out: Path) -> dict:
    m = json.loads((out / "manifest.json").read_bytes())
    if m.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{out}/manifest.json is not {MANIFEST_SCHEMA}")
    return m


def _caller() -> str:
    """The first three frames outside this module (journal attribution)."""
    f = sys._getframe(1)
    frames = []
    while f is not None and len(frames) < 3:
        mod = f.f_globals.get("__name__", "?")
        if mod != __name__:
            frames.append(f"{mod}:{f.f_code.co_name}:{f.f_lineno}")
        f = f.f_back
    return " < ".join(frames)


def _journal(out: Path, manifest_sha256: str, partition: str, role: str, table: str, columns: Sequence[str],
             chunks: int) -> None:
    """Append one access record (role, partition, table, columns, caller) to the access journal; fails closed."""
    path = Path(os.environ.get(JOURNAL_ENV) or (out / JOURNAL_NAME))
    entry = {"schema": JOURNAL_SCHEMA, "utc_us": common.utc_us(), "dict": str(out), "manifest_sha256": manifest_sha256,
             "partition": partition, "role": role, "table": table, "columns": list(columns), "chunks": chunks,
             "caller": _caller(), "pid": os.getpid()}
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, common.canonical(entry) + b"\n")
    finally:
        os.close(fd)


def _verified_chunks(out: Path, chunks: Sequence[Mapping], what: str) -> list[Path]:
    """The chunk paths of a manifest section after checking each one's bytes and sha256 against the manifest."""
    files = []
    for c in chunks:
        f = out / c["path"]
        if not f.is_file() or f.stat().st_size != c["bytes"] or common.sha256_file(f) != c["sha256"]:
            raise ReconcileError(f"{what} chunk {c['path']} is missing or differs from the manifest", kind="chunk")
        files.append(f)
    return files


def _check_split_role(files: Sequence[Path], partition: str) -> None:
    for f in files:
        col = pq.read_table(f, columns=["split_role"], use_threads=False).column("split_role")
        got = set(pc.unique(col.cast(pa.string())).to_pylist()) if len(col) else set()
        if got - {partition}:
            raise ReconcileError(f"{f.name} of partition {partition} holds rows of {sorted(got - {partition})}", kind="closure")


def _role_columns(partition: str, role: str) -> tuple[str, ...]:
    """The published columns `role` may read from `partition` (PermissionError when it may not open it)."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    if role in PROJECTED_ROLES:
        return PROJECTED_ROLES[role]
    if role in SEALED_ROLES:
        raise PermissionError(f"role {role!r} reads the sealed side table only (open_sealed / SealedIndex)")
    if role not in PARTITION_ROLE.values():
        raise ValueError(f"role must be one of {OPEN_ROLES} (or a sealed role for open_sealed)")
    if PARTITION_ROLE[partition] != role:
        raise PermissionError(f"the {partition} partition opens only under role={PARTITION_ROLE[partition]!r} or a "
                              f"projection role {tuple(PROJECTED_ROLES)}; {role!r} opens {[p for p, r in PARTITION_ROLE.items() if r == role][0]}")
    return tuple(ROW_SCHEMA.names)


def open_partition(out_dir: Path | str, partition: str, *, role: str = "fit") -> pads.Dataset:
    """The manifest-listed chunks of one partition, as a dataset of the columns `role` may read (module docstring:
    fit opens train_fit, training_validation dev, diagnostic_only diag, heldout_report heldout (P10 only); roundtrip
    and preflight open any partition, projected). Each chunk's bytes, sha256 and split_role are checked first, and
    every open other than fit on train_fit is journaled. Read it with use_threads=False (module docstring)."""
    columns = _role_columns(partition, role)
    out = Path(out_dir)
    raw = (out / "manifest.json").read_bytes()
    m = json.loads(raw)
    if m.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{out}/manifest.json is not {MANIFEST_SCHEMA}")
    files = _verified_chunks(out, m["partitions"][partition]["chunks"], partition)
    _check_split_role(files, partition)
    if not (partition == "train_fit" and role == "fit"):
        _journal(out, common.sha256_bytes(raw), partition, role, "published", columns, len(files))
    schema = pa.schema([ROW_SCHEMA.field(c) for c in columns])
    return pads.dataset([str(f) for f in files], format="parquet", schema=schema)


def open_sealed(out_dir: Path | str, partition: str, *, role: str) -> pads.Dataset:
    """The sealed side table of one partition (origin, exact record_status, conditioned_seed), for
    role='r7_crosscheck' (P5's R7 columns only) or 'exposed_report'. Never part of a run-plan closure; every open is
    journaled."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    if role not in SEALED_ROLES:
        raise PermissionError(f"the sealed side table opens only under {tuple(SEALED_ROLES)}")
    columns = SEALED_ROLES[role] or tuple(SEALED_SCHEMA.names)
    out = Path(out_dir)
    raw = (out / "manifest.json").read_bytes()
    m = json.loads(raw)
    if m.get("schema") != MANIFEST_SCHEMA or "sealed" not in m:
        raise ValueError(f"{out}/manifest.json is not a {MANIFEST_SCHEMA} manifest with a sealed side table")
    files = _verified_chunks(out, m["sealed"]["partitions"][partition]["chunks"], f"sealed {partition}")
    _journal(out, common.sha256_bytes(raw), partition, role, "sealed", columns, len(files))
    return pads.dataset([str(f) for f in files], format="parquet", schema=pa.schema([SEALED_SCHEMA.field(c) for c in columns]))


class SealedIndex:
    """P5's R7 cross-check through the guarded API: rows(cid) -> {(program_sha256 hex, origin, record_status)} of one
    context, read from its sealed chunk (bytes and sha256 checked, R7 columns only, journaled per chunk)."""

    def __init__(self, dict_dir: Path | str, *, role: str = "r7_crosscheck"):
        if role not in SEALED_ROLES:
            raise PermissionError(f"the sealed side table opens only under {tuple(SEALED_ROLES)}")
        self.dir, self.role = Path(dict_dir), role
        raw = (self.dir / "manifest.json").read_bytes()
        self.sha256 = common.sha256_bytes(raw)
        self.doc = json.loads(raw)
        if self.doc.get("schema") != MANIFEST_SCHEMA or "sealed" not in self.doc:
            raise ValueError(f"{self.dir}/manifest.json is not a {MANIFEST_SCHEMA} manifest with a sealed side table")
        self.where: dict[str, tuple[str, dict]] = {}
        for p, part in self.doc["sealed"]["partitions"].items():
            for ch in part["chunks"]:
                for c in ch["contexts"]:
                    self.where[c] = (p, ch)
        self._path: str | None = None
        self._rows: dict[str, set] = {}

    def rows(self, cid: str) -> set:
        hit = self.where.get(cid)
        if hit is None:
            raise ReconcileError("context is absent from the program dictionary's sealed side table", cid, "closure")
        p, ch = hit
        if ch["path"] != self._path:
            f = _verified_chunks(self.dir, [ch], f"sealed {p}")[0]
            _journal(self.dir, self.sha256, p, self.role, "sealed", SEALED_ROLES["r7_crosscheck"], 1)
            t = pq.read_table(f, columns=["context_sha256", "program_sha256", "origin", "record_status"], use_threads=False)
            grouped: dict[str, set] = defaultdict(set)
            for c, prog, o, st in zip(*(t.column(n).to_pylist() for n in t.column_names)):
                grouped[c.hex()].add((prog.hex(), o, st))
            self._path, self._rows = ch["path"], dict(grouped)
        return self._rows.get(cid, set())


def iter_distinct(out_dir: Path | str, partition: str, *, role: str = "fit", batch: int = 65536) -> Iterator[tuple[str, str]]:
    """(program_sha256 hex, dsl) for each distinct program of a partition, in sha order (in memory: ~1.2 GB for train_fit).
    role: the partition's own role (fit for train_fit) or the projection role 'roundtrip' (any partition)."""
    if "dsl" not in _role_columns(partition, role) or "program_sha256" not in _role_columns(partition, role):
        raise PermissionError(f"role {role!r} cannot read (program_sha256, dsl)")
    t = open_partition(out_dir, partition, role=role).to_table(columns=["program_sha256", "dsl"], use_threads=False)
    if t.num_rows == 0:
        return
    order = pc.sort_indices(t, sort_keys=[("program_sha256", "ascending")])
    prev = None
    for start in range(0, len(order), batch):
        part = t.take(order.slice(start, batch))
        for sha, dsl in zip(part.column(0).to_pylist(), part.column(1).to_pylist()):
            if sha != prev:
                prev = sha
                yield sha.hex(), dsl


# ----------------------------------------------------------------------------------------------------- verify

def _rehash(rec: dict, ctx_id: str) -> str | None:
    """sha256 of a recorded prefix as it is on disk now, or None when the record has no re-hashable layout."""
    layout, path, n = rec.get("layout"), Path(rec.get("path") or ""), rec.get("bytes", 0)
    h = hashlib.sha256()
    if layout == "file_prefix":
        if n == 0:
            return h.hexdigest()
        with open(path, "rb") as f:
            remaining = n
            while remaining:
                block = f.read(min(READ_BLOCK, remaining))
                if not block:
                    return "short:" + str(n - remaining)
                h.update(block)
                remaining -= len(block)
        return h.hexdigest()
    if layout == "ctx_lines":
        got = 0
        for line in _complete_lines(path):
            if got >= n:
                break
            if _line_context(line) == ctx_id:
                h.update(line)
                got += len(line)
        return h.hexdigest() if got == n else f"short:{got}"
    return None


def verify(out_dir: Path | str, *, reconcile_history_summaries: bool = True, rehash_sources: bool = False,
           sources: Sources | None = None) -> dict:
    """Re-read every chunk and re-check schema, rows, shas, counts, closure and totals against the manifest."""
    out = Path(out_dir)
    fails: list[str] = []
    n_fail = 0

    def fail(msg: str) -> None:
        nonlocal n_fail
        n_fail += 1
        if len(fails) < 200:
            fails.append(msg)

    m = _load_manifest(out)
    recs: dict[str, dict] = m["contexts"]
    seeds = frozenset(m["inputs"]["seeds"]["hashes"])
    if sources is None:
        src = m["inputs"]["sources"]
        sources = Sources(Path(src["incoming_cursor"]), Path(src["incoming_transport_index"]))
    plan = None
    ep, mp = m["inputs"]["epoch"], m["inputs"]["partition_map"]
    try:
        if common.sha256_file(ep["path"]) != ep["sha256"] or not mp.get("path") or common.sha256_file(mp["path"]) != mp["sha256"]:
            fail("epoch or partition-map file no longer matches its pinned sha256")
        else:
            plan = _plan(ep["path"], mp["path"])
    except (OSError, ReconcileError) as err:
        fail(f"cannot re-derive the plan: {err}")
    if plan is not None:
        want = {c.context_sha256: c.split_role for c in plan.contexts}
        if want != {cid: r["split_role"] for cid, r in recs.items()}:
            fail("manifest contexts or roles differ from the epoch and partition map (closure)")
        if plan.excluded != m["excluded"]:
            fail("excluded counts differ from the epoch and partition map")
    counted: dict[str, Counter] = defaultdict(Counter)
    seen_in: dict[str, str] = {}
    n_rows = n_chunks = n_sealed = 0

    def check_file(f: Path, entry: Mapping, schema: pa.Schema, what: str):
        """The chunk's table after its bytes, sha256, exact schema, metadata and row groups check out (else None)."""
        if not f.is_file() or f.stat().st_size != entry["bytes"] or common.sha256_file(f) != entry["sha256"]:
            fail(f"{entry['path']}: missing or bytes/sha256 differ from the manifest")
            return None
        pf = pq.ParquetFile(f)
        if not pf.schema_arrow.equals(schema, check_metadata=False):
            fail(f"{entry['path']}: schema is not the exact {what} allowlist")
            return None
        if set((pf.metadata.metadata or {}).keys()) != {b"ARROW:schema"}:
            fail(f"{entry['path']}: unexpected key-value metadata")
        for g in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(g)
            if rg.num_rows > ROW_GROUP_ROWS or any(rg.column(j).compression != "ZSTD" for j in range(rg.num_columns)):
                fail(f"{entry['path']}: row group {g} exceeds {ROW_GROUP_ROWS} rows or is not zstd")
        t = pf.read(use_threads=False)
        if t.num_rows != entry["rows"]:
            fail(f"{entry['path']}: {t.num_rows} rows != manifest {entry['rows']}")
        return t

    sealed_parts = (m.get("sealed") or {}).get("partitions")
    if not isinstance(sealed_parts, dict) or set(sealed_parts) != set(PARTITIONS):
        fail("manifest has no sealed side table for every partition")
        sealed_parts = {p: {"chunks": []} for p in PARTITIONS}
    # only manifest-listed files may sit in the partition directories (dot-prefixed temporaries excepted)
    listed = {c["path"] for p in PARTITIONS for c in m["partitions"][p]["chunks"]}
    listed |= {c["path"] for p in PARTITIONS for c in sealed_parts[p]["chunks"]}
    for d in [out / p for p in PARTITIONS] + [out / SEALED_DIR / p for p in PARTITIONS]:
        if d.exists():
            for x in sorted(d.rglob("*")):
                rel = str(x.relative_to(out))
                if x.is_file() and not x.name.startswith(".") and rel not in listed:
                    fail(f"{rel}: a file in a partition directory that the manifest does not list")
    if (out / SEALED_DIR).exists():
        for x in sorted((out / SEALED_DIR).iterdir()):
            if not x.name.startswith(".") and x.name not in PARTITIONS:
                fail(f"{SEALED_DIR}/{x.name}: not a partition of the sealed side table")
    for p in PARTITIONS:
        part = m["partitions"][p]
        sealed_by_published = {c.get("published"): c for c in sealed_parts[p]["chunks"]}
        if len(sealed_by_published) != len(part["chunks"]) or set(sealed_by_published) != {c["path"] for c in part["chunks"]}:
            fail(f"partition {p}: sealed chunks are not one per published chunk")
        prev = None
        files, sfiles = [], []
        for ch in part["chunks"]:
            n_chunks += 1
            t = check_file(out / ch["path"], ch, ROW_SCHEMA, "published")
            sch = sealed_by_published.get(ch["path"])
            st = None
            if sch is not None:
                n_sealed += 1
                st = check_file(out / sch["path"], sch, SEALED_SCHEMA, "sealed")
                if sch.get("contexts") != ch.get("contexts"):
                    fail(f"{sch['path']}: contexts differ from its published twin {ch['path']}")
            if t is None:
                continue
            files.append(out / ch["path"])
            if st is not None:
                sfiles.append(out / sch["path"])
            members = set(ch["contexts"])
            cols = [t.column(name).to_pylist() for name in ROW_SCHEMA.names]
            if st is not None and st.num_rows == t.num_rows:
                scols = [st.column(name).to_pylist() for name in SEALED_SCHEMA.names]
            else:
                if st is not None:
                    fail(f"{sch['path']}: {st.num_rows} rows != its published twin's {t.num_rows}")
                scols = [[None] * t.num_rows for _ in SEALED_SCHEMA.names]
            for (cb, pb, dsl, fam, role, status, shared, cohort, form, nbytes), (scb, spb, origin, exact, cond) in zip(
                    zip(*cols), zip(*scols)):
                n_rows += 1
                key = (cb, pb)
                cid, sha = cb.hex(), pb.hex()
                rec = recs.get(cid)
                bad = []
                if prev is not None and key <= prev:
                    bad.append("rows not strictly ordered by (context, program)")
                prev = key
                if rec is None or cid not in members or rec.get("chunk") != ch["path"] or role != p or rec["split_role"] != p:
                    bad.append("context outside this chunk or partition")
                if hashlib.sha256(dsl.encode("utf-8")).hexdigest() != sha:
                    bad.append("sha256(dsl) != program_sha256")
                try:
                    form_ok = program_form(dsl) == form
                except ReconcileError:
                    form_ok = False
                if not form_ok or nbytes != len(dsl.encode("utf-8")):
                    bad.append("form or dsl_bytes")
                if status not in PUBLISHED_STATUSES:
                    bad.append("published record_status")
                if rec is not None and (cohort != rec["cohort"] or fam != rec["family_id"]):
                    bad.append("cohort or family_id")
                if shared != (sha in seeds):
                    bad.append("shared_seed")
                if st is not None:
                    if (scb, spb) != key:
                        bad.append("sealed row is not aligned with the published row")
                    if origin not in STATUSES_BY_ORIGIN or exact not in STATUSES_BY_ORIGIN[origin]:
                        bad.append("sealed origin/status")
                    elif published_status(exact) != status:
                        bad.append("published record_status != published_status(sealed record_status)")
                    if cond and origin == "transport_only":
                        bad.append("conditioned_seed on a transport-only row")
                if bad:
                    fail(f"{ch['path']} row {cid[:16]}/{sha[:16]}: {', '.join(bad)}")
                if seen_in.setdefault(cid, p) != p:
                    fail(f"context {cid[:16]} appears in partitions {seen_in[cid]} and {p}")
                c = counted[cid]
                c["rows"] += 1
                c["origin:" + str(origin)] += 1
                c["status:" + str(exact)] += 1
                c["form:" + str(form)] += 1
                c["published:" + str(status)] += 1
                c["shared"] += bool(shared)
                c["conditioned"] += bool(cond)
        try:
            stats = _partition_stats(files)
            sstats = _partition_stats(sfiles, SEALED_SCHEMA, SEALED_STAT_COLUMNS)
        except (OSError, pa.ArrowException) as err:
            stats = sstats = {"error": str(err)[:200]}
        if {k: part.get(k) for k in stats} != stats:
            fail(f"partition {p}: rows, distinct programs or breakdowns differ from the manifest")
        if {k: sealed_parts[p].get(k) for k in sstats} != sstats:
            fail(f"partition {p}: sealed rows or breakdowns differ from the manifest")
        if part["contexts"] != sum(1 for r in recs.values() if r["split_role"] == p):
            fail(f"partition {p}: context count differs from the per-context records")
    for cid, rec in recs.items():
        c = counted.get(cid, Counter())
        want = Counter({"rows": rec["rows"], "shared": rec["shared_seed_rows"],
                        "conditioned": rec.get("conditioned_seed_rows", 0)})
        for k, v in rec["by_origin"].items():
            want["origin:" + k] += v
        for k, v in rec["by_status"].items():
            want["status:" + k] += v
            want["published:" + published_status(k)] += v
        for k, v in rec["by_form"].items():
            want["form:" + k] += v
        if +c != +want or rec["rows"] != rec["summary"]["unique_programs"]:
            fail(f"context {cid[:16]}: chunk rows do not reconcile with its record and unique_programs")
    total_rows = sum(r["rows"] for r in recs.values())
    if total_rows != n_rows or total_rows != m["totals"]["rows"] or total_rows != sum(r["summary"]["unique_programs"] for r in recs.values()):
        fail(f"totals: chunk rows {n_rows}, records {total_rows}, manifest {m['totals']['rows']}")
    reconciled = rehashed = 0
    if reconcile_history_summaries or rehash_sources:
        ctxs = {c.context_sha256: c for c in plan.contexts} if plan is not None else {}
        inc = None
        if reconcile_history_summaries and any(r["cohort"] == "incoming" for r in recs.values()):
            inc = load_incoming_cursor(sources.incoming_cursor)[0]
        for cid, rec in sorted(recs.items()):
            if reconcile_history_summaries and rec["source"] == "local":
                try:
                    live = dataclasses.asdict(_summary(ctxs[cid], inc)[0]) if cid in ctxs else None
                except ReconcileError as err:
                    live = str(err)
                if live != rec["summary"]:
                    fail(f"context {cid[:16]}: live history summary no longer equals the recorded one ({str(live)[:120]})")
                reconciled += 1
            if rehash_sources:
                for name, pr in rec.get("prefixes", {}).items():
                    try:
                        got = _rehash(pr, cid)
                    except OSError as err:
                        got = f"error:{err}"
                    if got is None:
                        continue
                    rehashed += 1
                    if got != pr.get("sha256"):
                        fail(f"context {cid[:16]} prefix {name}: sha256 now {got[:24]} != recorded")
    return {"schema": VERIFY_SCHEMA, "out": str(out), "manifest_sha256": common.sha256_file(out / "manifest.json"),
            "verdict": "PASS" if not n_fail else "FAIL", "failure_count": n_fail, "failures": fails[:100],
            "checked": {"chunks": n_chunks, "sealed_chunks": n_sealed, "rows": n_rows, "contexts": len(recs),
                        "summaries_reconciled": reconciled, "prefixes_rehashed": rehashed},
            "utc_us": common.utc_us()}


# ------------------------------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.programs", description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the program dictionary (resumable)")
    b.add_argument("--epoch", required=True)
    b.add_argument("--partition-map", required=True, help="JSON {context_sha256: {family_id, split_role}} (P2/P3 output)")
    b.add_argument("--seeds", required=True, help="JSON list of the P2 seed program hashes")
    b.add_argument("--out", required=True)
    b.add_argument("--chunk-contexts", type=int, default=CHUNK_CONTEXTS)
    b.add_argument("--no-resume", action="store_true")
    v = sub.add_parser("verify", help="re-read and re-check a built dictionary")
    v.add_argument("--out", required=True)
    v.add_argument("--reconcile-history-summaries", action=argparse.BooleanOptionalAction, default=True)
    v.add_argument("--rehash-sources", action="store_true")
    for p in (b, v):
        p.add_argument("--incoming-cursor", default=None)
        p.add_argument("--incoming-transport-index", default=None)
    args = ap.parse_args(argv)
    sources = None
    if args.incoming_cursor or args.incoming_transport_index or args.cmd == "build":
        sources = Sources(Path(args.incoming_cursor or DEFAULT_SOURCES.incoming_cursor),
                          Path(args.incoming_transport_index or DEFAULT_SOURCES.incoming_transport_index))
    if args.cmd == "build":
        try:
            m = build(args.epoch, args.partition_map, args.seeds, args.out, chunk_contexts=args.chunk_contexts,
                      resume=not args.no_resume, sources=sources)
        except (ReconcileError, common.ImmutableConflict) as err:
            print(json.dumps({"verdict": "STOP", "kind": getattr(err, "kind", "immutable"), "context": getattr(err, "context", None),
                              "error": str(err)[:500]}), file=sys.stderr)
            return 2
        print(json.dumps({"verdict": "BUILT", "manifest": str(Path(args.out) / "manifest.json"), "totals": m["totals"],
                          "partitions": {p: {k: m["partitions"][p][k] for k in ("contexts", "rows", "distinct_programs")} for p in PARTITIONS}}))
        return 0
    rep = verify(args.out, reconcile_history_summaries=args.reconcile_history_summaries, rehash_sources=args.rehash_sources, sources=sources)
    print(json.dumps(rep))
    return 0 if rep["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
