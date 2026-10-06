"""Trainer-mode live producer for ``jope.dl-training-live/v1`` (PLAN §6.7, §5 P7 test 8, §5 P8 Live producer).

What this module does, and the rule behind each part:

- ``native_process_identity``: the trainer's own ``/proc/self`` identity, computed exactly as Codex's
  ``unified_observer.py`` computes it for a PID (§6.7 DL-D11 item 1). It only ever goes into
  ``native_process_identity`` and update receipts; ``process_identity`` stays the audit identity.
- ``write_update_receipt``: create-only ``jope.native-optimizer-update/v1`` receipts, one file per reported
  step, at most 64 KiB, refusing ``before == after`` (§5 P8: the observer re-reads them later).
- ``build_document``: enforces the §6.7 rules the schema cannot express (history and losses withheld until
  ``optimizer_parameter_change_verified``; ``initial != current``; ``validation_loss`` is dev-only; unknown is
  ``null`` and listed in ``unknown[]``; never NaN/Infinity), validates against ``claude/live.schema.json``
  (read-only) and bounds the encoding to 256 KiB.
- ``LiveWriter``: atomic writes to a caller-given path only (never under ``claude/``), from a thread that
  publishes the latest snapshot at most every 3 s (§6.7 Freshness).
- ``bridge_program``: rebuilds Codex's pinned bridge program (``dl_feed.py`` text, a newline, then the REMOTE
  string extracted with ``ast`` from ``dl_bridge.py``; neither file is imported) and substitutes only the
  source path, for replay tests (§5 P7 test 8).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import jsonschema

from dltrain import common

CLAUDE_DIR = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude")
LIVE_SCHEMA_PATH = CLAUDE_DIR / "live.schema.json"
CONSUMER_DIR = Path(
    "/home/ubuntu/JopeDime/.agent/experiments/lambda-training/local_lambda/full_regression/research_2k/"
    "per_fold_hwm/incoming_intake_v1/final_dashboard_v1/health_v1"
)
CONSUMER_PINS = {
    "dl_bridge.py": "27b522c7732691d20fdfa351f395b09d0572a36d7620fd99aac0948999840285",
    "dl_feed.py": "b226dfadfa5d823fd2d13424a91cd62d062c9d969ce579ccfae0cc1933704e04",
    "unified.js": "d66e514c92f519b52f89d605b457232278a066fb7fdfec1990f1766dd516a220",
    "health.js#fresh": "9513e5b5690faf9ad183ef177c9fde571e734ed22dda12e2d73a1431014e9377",
    "unified.js#dlMetrics": "af8385714ade98d08df397a9f8a53593780478ca81c64a6741b309f9d9a2960f",
}
BRIDGE_PROGRAM_SHA256 = "6e58f65e493974e3580b821903c0bef1b0409af43c04e17ed8b6083e5ee6f03a"  # unsubstituted
SOURCE_LINE = "p=root/'claude/live.json'"
MAX_DOC_BYTES = 262144
MAX_HISTORY = 200
MAX_UPDATE_RECEIPT_BYTES = 65536
MAX_INTERVAL_S = 3.0
LAUNCHER_KEYS = (
    "owner", "process_identity", "kickoff_ack_verified", "kickoff_ack_utc_iso", "kickoff_ack_receipt_path",
    "kickoff_ack_receipt_sha256", "kickoff_receipt_path", "plan_path", "audit_path", "handoff_path",
    "source_evidence_path", "encoding_inventory_path", "model_requested", "model_observed", "effort_observed",
    "ultracode_observed", "model_verification_utc_us", "model_verification_source", "audit_packet_state",
    "holdout", "gates", "decisions_pending", "decisions_recorded", "implementation", "encoding_upstream",
    "budget", "family_registry_ref", "holdout_ref", "tokenizer_ref", "corpus_ref", "corpus_pointer_drift",
    "encoding_manifest_ref", "not_applicable",
)

LIVE_SCHEMA_ID = "jope.dl-training-live/v1"
UPDATE_RECEIPT_SCHEMA = "jope.native-optimizer-update/v1"
CONTRACT_ID = "jope.dl-training-live/v1#claude-superset-r5"
OBSERVER_EXPIRY_S = 45
DEV_VALIDATION_SPLIT = "dev-family-v1"
AUDIT_IDENTITY_KEYS = ("host", "pid", "starttick", "screen")
NATIVE_IDENTITY_KEYS = ("host", "pid", "starttick", "cwd", "executable_sha256", "argv_sha256")
RATE_KEYS = ("dataset_splits_per_second", "examples_per_second", "batches_per_second", "tokens_per_second")
METRIC_KEYS = (
    "token_validity", "grammar_validity", "ranking_pair_accuracy", "dev_novel_hit_rate", "dev_cache_hit_rate",
    "dev_inner_cv_regret", "dev_tail_fraction", "heldout_contexts_evaluated", "heldout_contexts_total",
    "heldout_milestone",
)
COUNTER_KEYS = ("ml_calls", "gpu_jobs_started", "evaluator_claims", "s3_writes")
CHECKPOINT_UPLOAD_STATUSES = ("LOCAL_ONLY", "UPLOADING", "PUBLISHED_VERIFIED", "FAILED")
READBACK_STATUSES = ("pass", "fail", "pending")

# phase -> status for an admitted run (§6.7 Statuses); an unadmitted run is AUDITING_PLAN_UNADMITTED throughout.
_ADMITTED_STATUS = {
    "preflight": "ADMITTED_PREFLIGHT",
    "waiting_intake": "WAITING_INTAKE",
    "stepping": "TRAINING",
    "checkpointing": "CHECKPOINTING",
    "validating": "VALIDATING",
    "yielded": "YIELDED_TO_INTAKE",
    "yielded_unpublished": "YIELDED_UNPUBLISHED",
    "completed": "COMPLETED",
    "stopped_budget": "STOPPED_BUDGET",
    "failed": "FAILED",
    "aborted_gate": "ABORTED_GATE",
}
PHASES = tuple(_ADMITTED_STATUS)

# Top-level fields owned by build_document; a launcher not_applicable[] entry for one of them is dropped,
# because in trainer mode a null value of these fields is listed in unknown[] instead.
DYNAMIC_FIELDS = frozenset((
    "run_id", "native_process_identity", "launch_receipt_ref", "optimizer_update_ref", "optimizer_update_receipt",
    "admission_ref", "training_proof", "run", "epoch", "optimizer_step", "train_loss", "validation_loss",
    "validation_split", "history", "samples_per_second", "batch_size", "gradient_norm", "learning_rate",
    "encoding_dimensions", "gpu", "host_guard", "checkpoint_ref", "checkpoint_utc_us",
    "checkpoint_destination_readback", "checkpoint_resume_qualified", "checkpoint", "storage_bytes",
    "storage_cap_bytes", "metrics", "training_admitted", "optimizer_parameter_change_verified",
    *RATE_KEYS, *COUNTER_KEYS,
))

_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_WITHHELD = "withheld until optimizer_parameter_change_verified is true (PLAN §6.7)"
_UNKNOWN_REASON = {
    "admission_ref": "no admission receipt is referenced by this run",
    "run_id": "no run",
    "native_process_identity": "trainer process identity not declared",
    "launch_receipt_ref": "no create-only launch receipt for this launch (written by the P8 launcher)",
    "optimizer_update_ref": "no optimizer update receipt written yet",
    "optimizer_update_receipt": "no optimizer update receipt written yet",
    "optimizer_step": "no optimizer update receipt written yet",
    "training_proof": "optimizer parameter change not verified yet",
    "run": "run identity not reported",
    "epoch": "no completed reporting epoch",
    "train_loss": "no optimizer step reported yet",
    "validation_loss": "no development-family validation pass yet",
    "batch_size": "no optimizer step in this process yet",
    "gradient_norm": "no optimizer step in this process yet",
    "learning_rate": "no optimizer step in this process yet",
    "encoding_dimensions": "not reported",
    "checkpoint_ref": "no checkpoint published to a versioned store",
    "checkpoint_utc_us": "no checkpoint written yet",
    "checkpoint_destination_readback": "no published checkpoint to read back",
    "checkpoint_resume_qualified": "a restore proof is recorded by its own receipt, never by the trainer",
    "checkpoint": "no checkpoint written yet",
    "storage_bytes": "store bytes not counted by this process (no store, or a resumed launch)",
    "storage_cap_bytes": "no storage ceiling in the run plan",
    "gpu": "CPU trainer: no GPU observation",
    "host_guard": "GPU-gate polling belongs to the P8 launcher",
}
_RATE_REASON = "not measured yet in this process"
_COUNTER_REASON = "lane counts come only from P-packet receipts; this producer cannot derive one (PLAN §6.7)"
_METRIC_REASON = "no development-family evaluation value for this metric"


class LiveRefused(RuntimeError):
    """A live document, receipt or path would violate the §6.7 contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_outside_claude(path, what: str = "path") -> None:
    """Refuse any path that lexically or after symlink resolution lies under ``CLAUDE_DIR`` (never written)."""
    p = Path(path)
    candidates = (Path(os.path.abspath(p)), p.resolve())
    roots = (CLAUDE_DIR, CLAUDE_DIR.resolve())
    for c in candidates:
        for root in roots:
            if c == root or c.is_relative_to(root):
                raise LiveRefused(f"{what} {p} resolves under {root}; claude/ is read-only for this lane")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_hex64(x) -> bool:
    return isinstance(x, str) and _HEX64.fullmatch(x) is not None


def native_process_identity(host: str | None = None) -> dict:
    """This process's identity, computed like Codex ``unified_observer.py`` does for a PID (§6.7 DL-D11 item 1).

    ``starttick`` is field 22 of ``/proc/self/stat`` (index 19 after the last ``)``); the executable hash is over
    the resolved ``/proc/self/exe`` and the argv hash over the raw ``/proc/self/cmdline`` bytes.
    """
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    exe = Path("/proc/self/exe").resolve()
    argv = Path("/proc/self/cmdline").read_bytes()
    return {
        "host": host or socket.gethostname(),
        "pid": os.getpid(),
        "starttick": stat[19],
        "cwd": str(Path("/proc/self/cwd").resolve()),
        "executable_sha256": _sha256(exe.read_bytes()),
        "argv_sha256": _sha256(argv),
    }


def load_schema(path=LIVE_SCHEMA_PATH) -> tuple[dict, str]:
    """Read (never write) the producer schema; return it with the sha256 of its bytes."""
    raw = Path(path).read_bytes()
    schema = json.loads(raw)
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema, _sha256(raw)


def validate(doc, schema: Mapping) -> list[str]:
    """Sorted JSON Schema 2020-12 error messages (each prefixed by its JSON path); empty means valid."""
    validator = jsonschema.Draft202012Validator(schema)
    out = []
    for err in validator.iter_errors(doc):
        where = "/".join(str(x) for x in err.absolute_path) or "<root>"
        out.append(f"{where}: {err.message}")
    return sorted(out)


def encode_document(doc) -> bytes:
    """Compact sorted-key UTF-8 JSON with ``allow_nan=False``; refuses anything above 256 KiB."""
    try:
        data = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except ValueError as exc:
        raise LiveRefused(f"non-finite number refused: {exc}") from exc
    if len(data) > MAX_DOC_BYTES:
        raise LiveRefused(f"document is {len(data)} bytes > {MAX_DOC_BYTES}")
    return data


def write_update_receipt(update_dir, *, run_id: str, optimizer_step: int, process_identity: dict,
                         before_sha256: str, after_sha256: str, utc_us: int | None = None) -> tuple[dict, dict]:
    """Write one create-only ``jope.native-optimizer-update/v1`` receipt (§6.7; §5 P8 Live producer).

    The file is ``update_dir/<run_id>/update-<step:010d>.json``; an existing file with different bytes raises
    ``common.ImmutableConflict`` (receipts are never overwritten). Returns ``(receipt, localref)``.
    """
    assert_outside_claude(update_dir, "update_dir")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise LiveRefused(f"run_id {run_id!r} is not a safe path component")
    if not _is_int(optimizer_step) or optimizer_step < 1:
        raise LiveRefused(f"optimizer_step must be an integer >= 1, got {optimizer_step!r}")
    if not _is_hex64(before_sha256) or not _is_hex64(after_sha256):
        raise LiveRefused("before/after parameter hashes must be lowercase sha256 hex")
    if before_sha256 == after_sha256:
        raise LiveRefused("refused: before_parameter_sha256 == after_parameter_sha256 (no parameter change)")
    if not isinstance(process_identity, dict) or any(k not in process_identity for k in NATIVE_IDENTITY_KEYS):
        raise LiveRefused("an update receipt carries the trainer's native process identity, never the audit identity")
    if utc_us is not None and not _is_int(utc_us):
        raise LiveRefused("utc_us must be an integer")
    receipt = {
        "schema": UPDATE_RECEIPT_SCHEMA,
        "utc_us": common.utc_us() if utc_us is None else utc_us,
        "run_id": run_id,
        "optimizer_step": optimizer_step,
        "process_identity": copy.deepcopy(process_identity),
        "before_parameter_sha256": before_sha256,
        "after_parameter_sha256": after_sha256,
    }
    data = encode_document(receipt)
    if len(data) > MAX_UPDATE_RECEIPT_BYTES:
        raise LiveRefused(f"update receipt is {len(data)} bytes > {MAX_UPDATE_RECEIPT_BYTES}")
    res = common.immutable_write(Path(update_dir) / run_id / f"update-{optimizer_step:010d}.json", data)
    return receipt, {"path": res["path"], "bytes": res["bytes"], "sha256": res["sha256"]}


def status_for(phase: str, admitted: bool, has_receipt: bool) -> str:
    """Map a trainer phase to the verbatim §6.7 status."""
    if phase not in _ADMITTED_STATUS:
        raise ValueError(f"unknown trainer phase {phase!r}; expected one of {PHASES}")
    if not admitted:
        return "AUDITING_PLAN_UNADMITTED"
    if phase == "stepping" and not has_receipt:
        return "ADMITTED_PREFLIGHT"
    return _ADMITTED_STATUS[phase]


def _default_rates() -> dict:
    return dict.fromkeys(RATE_KEYS)


@dataclass(frozen=True)
class LiveState:
    """One trainer snapshot. ``validation_split`` and ``metrics`` record provenance of dev values (§6.7)."""

    phase: str
    training_admitted: bool = False
    admission_ref: dict | None = None
    run_id: str | None = None
    native_process_identity: dict | None = None
    launch_receipt_ref: dict | None = None
    update_receipt: dict | None = None
    update_ref: dict | None = None
    optimizer_parameter_change_verified: bool = False
    initial_parameter_sha256: str | None = None
    current_parameter_sha256: str | None = None
    optimizer_updates_total: int | None = None
    verified_utc_us: int | None = None
    train_loss: float | None = None
    validation_loss: float | None = None
    history: tuple = ()
    epoch: int | None = None
    rates: dict = field(default_factory=_default_rates)
    batch_size: int | None = None
    gradient_norm: float | None = None
    learning_rate: float | None = None
    encoding_dimensions: dict | None = None
    run_info: dict | None = None
    checkpoint_ref: dict | None = None
    checkpoint_utc_us: int | None = None
    checkpoint_readback: str | None = None
    checkpoint: dict | None = None
    storage_bytes: int | None = None
    storage_cap_bytes: int | None = None
    validation_split: str | None = None
    metrics: dict | None = None


def trainer_base_from_audit(audit_doc: Mapping) -> dict:
    """Deep copy of the launcher keys of the audit document; the audit ``process_identity`` stays as is (§6.7)."""
    identity = audit_doc.get("process_identity")
    if not isinstance(identity, dict) or any(k not in identity for k in AUDIT_IDENTITY_KEYS):
        raise LiveRefused("audit document has no owner process_identity {host, pid, starttick, screen}")
    return {k: copy.deepcopy(audit_doc[k]) for k in LAUNCHER_KEYS if k in audit_doc}


def _check_finite(obj, where: str = "") -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise LiveRefused(f"non-finite number at {where or '<root>'}; unknown values are null, never NaN/Infinity")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{where}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite(v, f"{where}/{i}")


def _number_or_none(name: str, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveRefused(f"{name} must be a number or null")
    if not math.isfinite(float(value)):
        raise LiveRefused(f"{name} is not finite; unknown values are null")
    return value


def _int_or_none(name: str, value):
    if value is None:
        return None
    if not _is_int(value):
        raise LiveRefused(f"{name} must be an integer or null")
    return value


def decimate_history(rows) -> list[dict]:
    """Check rows (strictly increasing integer epochs, finite losses) and keep at most 200, first and last kept."""
    rows = [dict(r) for r in rows]
    previous = None
    for i, row in enumerate(rows):
        if not _is_int(row.get("epoch")) or row["epoch"] < 0:
            raise LiveRefused(f"history[{i}].epoch must be a non-negative integer")
        if previous is not None and row["epoch"] <= previous:
            raise LiveRefused("history epochs must be strictly increasing")
        previous = row["epoch"]
        for key in ("train_loss", "validation_loss"):
            if row.get(key) is None:
                raise LiveRefused(f"history[{i}].{key} is required")
            _number_or_none(f"history[{i}].{key}", row[key])
        for key in ("optimizer_step", "utc_us"):
            if key in row and not _is_int(row[key]):
                raise LiveRefused(f"history[{i}].{key} must be an integer")
    n = len(rows)
    if n <= MAX_HISTORY:
        return rows
    span = MAX_HISTORY - 1
    picks = [(i * (n - 1) + span // 2) // span for i in range(MAX_HISTORY)]  # 0 ... n-1, distinct since n-1 > span
    return [rows[j] for j in picks]


def _check_receipt(receipt: dict, ref: dict, state: LiveState) -> None:
    if not isinstance(receipt, dict) or receipt.get("schema") != UPDATE_RECEIPT_SCHEMA:
        raise LiveRefused("optimizer_update_receipt must be a jope.native-optimizer-update/v1 receipt")
    if receipt.get("run_id") != state.run_id:
        raise LiveRefused("update receipt run_id differs from the document run_id")
    if not _is_int(receipt.get("optimizer_step")) or receipt["optimizer_step"] < 1:
        raise LiveRefused("update receipt optimizer_step must be an integer >= 1")
    before, after = receipt.get("before_parameter_sha256"), receipt.get("after_parameter_sha256")
    if not _is_hex64(before) or not _is_hex64(after):
        raise LiveRefused("update receipt parameter hashes must be sha256 hex")
    if before == after:
        raise LiveRefused("update receipt before == after: no parameter change")
    if state.native_process_identity is not None and receipt.get("process_identity") != state.native_process_identity:
        raise LiveRefused("update receipt process_identity differs from native_process_identity")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise LiveRefused("optimizer_update_ref must be a localref {path, bytes, sha256}")
    assert_outside_claude(ref["path"], "optimizer_update_ref")
    data = encode_document(receipt)
    if ref.get("sha256") != _sha256(data) or ref.get("bytes") != len(data) or len(data) > MAX_UPDATE_RECEIPT_BYTES:
        raise LiveRefused("optimizer_update_ref sha256/bytes do not bind optimizer_update_receipt")


def _training_proof(state: LiveState) -> dict:
    initial, current = state.initial_parameter_sha256, state.current_parameter_sha256
    if not _is_hex64(initial) or not _is_hex64(current):
        raise LiveRefused("verified documents need initial and current parameter sha256")
    if initial == current:
        raise LiveRefused("verified documents need initial_parameter_sha256 != current_parameter_sha256")
    total = state.optimizer_updates_total
    if not _is_int(total) or total < 1:
        raise LiveRefused("verified documents need optimizer_updates_total >= 1")
    if not _is_int(state.verified_utc_us):
        raise LiveRefused("verified documents need verified_utc_us")
    return {"initial_parameter_sha256": initial, "current_parameter_sha256": current,
            "optimizer_updates_total": total, "verified_utc_us": state.verified_utc_us}


def _clean_checkpoint(checkpoint):
    if checkpoint is None:
        return None
    if not isinstance(checkpoint, dict):
        raise LiveRefused("checkpoint must be an object or null")
    out = {k: v for k, v in checkpoint.items() if v is not None}
    if "upload_status" in out and out["upload_status"] not in CHECKPOINT_UPLOAD_STATUSES:
        raise LiveRefused(f"checkpoint.upload_status must be one of {CHECKPOINT_UPLOAD_STATUSES}")
    return out


def build_document(base: Mapping, state: LiveState, *, schema: Mapping) -> dict:
    """Build one trainer-mode ``jope.dl-training-live/v1`` document and refuse any §6.7 violation (LiveRefused)."""
    if not isinstance(state, LiveState):
        raise TypeError("state must be a LiveState")
    extra = sorted(set(base) - set(LAUNCHER_KEYS))
    if extra:
        raise LiveRefused(f"base may carry launcher keys only; got {extra}")
    audit_identity = base.get("process_identity")
    if not isinstance(audit_identity, dict) or any(k not in audit_identity for k in AUDIT_IDENTITY_KEYS):
        raise LiveRefused("base process_identity must be the audit identity {host, pid, starttick, screen}")
    native = state.native_process_identity
    if native is not None and audit_identity.get("pid") == native.get("pid") \
            and str(audit_identity.get("starttick")) == str(native.get("starttick")):
        raise LiveRefused("process_identity must stay the audit identity, never the trainer's own")
    receipt, ref = state.update_receipt, state.update_ref
    if (receipt is None) != (ref is None):
        raise LiveRefused("optimizer_update_receipt and optimizer_update_ref come together")
    if receipt is not None:
        _check_receipt(receipt, ref, state)
        if not state.optimizer_parameter_change_verified:
            raise LiveRefused("an optimizer update receipt proves a parameter change: the run must be verified")
    if state.checkpoint_readback is not None and state.checkpoint_readback not in READBACK_STATUSES:
        raise LiveRefused(f"checkpoint_readback must be one of {READBACK_STATUSES}")

    doc = copy.deepcopy(dict(base))
    if "not_applicable" in doc:
        doc["not_applicable"] = [e for e in doc["not_applicable"]
                                 if re.split(r"[./]", str(e.get("field", "")), maxsplit=1)[0] not in DYNAMIC_FIELDS]
    unknown: list[dict] = []

    def put(name, value, reason=None):
        doc[name] = value
        if value is None:
            unknown.append({"field": name, "reason": reason or _UNKNOWN_REASON[name]})

    run_id = state.run_id
    doc["schema"] = LIVE_SCHEMA_ID
    doc["utc_us"] = common.utc_us()
    doc["status"] = status_for(state.phase, bool(state.training_admitted), receipt is not None)
    doc["process_identity"] = copy.deepcopy(audit_identity)
    doc["source_kind"] = "trainer_telemetry"
    doc["training_admitted"] = bool(state.training_admitted)
    put("admission_ref", copy.deepcopy(state.admission_ref))
    put("run_id", run_id)
    put("native_process_identity", copy.deepcopy(native))
    put("launch_receipt_ref", copy.deepcopy(state.launch_receipt_ref))
    put("optimizer_update_ref", copy.deepcopy(ref))
    put("optimizer_update_receipt", copy.deepcopy(receipt))
    put("optimizer_step", receipt["optimizer_step"] if receipt is not None else None)

    verified = bool(state.optimizer_parameter_change_verified)
    doc["optimizer_parameter_change_verified"] = verified
    if verified:
        doc["training_proof"] = _training_proof(state)
        epoch = _int_or_none("epoch", state.epoch)
        train_loss = _number_or_none("train_loss", state.train_loss)
        validation_loss = _number_or_none("validation_loss", state.validation_loss)
        history = decimate_history(state.history)
        reasons = {}
    else:
        put("training_proof", None)
        epoch = train_loss = validation_loss = None
        history = []
        reasons = {"epoch": _WITHHELD, "train_loss": _WITHHELD, "validation_loss": _WITHHELD}
    if (validation_loss is not None or history) and state.validation_split != DEV_VALIDATION_SPLIT:
        raise LiveRefused("validation_loss and history are development-family only: validation_split must be "
                          f"{DEV_VALIDATION_SPLIT!r}, got {state.validation_split!r}")
    put("epoch", epoch, reasons.get("epoch"))
    put("train_loss", train_loss, reasons.get("train_loss"))
    put("validation_loss", validation_loss, reasons.get("validation_loss"))
    doc["validation_split"] = DEV_VALIDATION_SPLIT if validation_loss is not None else None
    doc["history"] = history

    rates = state.rates or {}
    unexpected = sorted(set(rates) - set(RATE_KEYS))
    if unexpected:
        raise LiveRefused(f"unknown rate keys {unexpected}")
    for key in RATE_KEYS:
        put(key, _number_or_none(key, rates.get(key)), _RATE_REASON)
    put("samples_per_second", None, "this trainer reports examples_per_second (sequences per second)")
    put("batch_size", _int_or_none("batch_size", state.batch_size))
    put("gradient_norm", _number_or_none("gradient_norm", state.gradient_norm))
    put("learning_rate", _number_or_none("learning_rate", state.learning_rate))
    put("encoding_dimensions", copy.deepcopy(state.encoding_dimensions))
    put("run", copy.deepcopy(state.run_info))
    put("checkpoint_ref", copy.deepcopy(state.checkpoint_ref))
    put("checkpoint_utc_us", _int_or_none("checkpoint_utc_us", state.checkpoint_utc_us))
    put("checkpoint_destination_readback",
        {"status": state.checkpoint_readback} if state.checkpoint_readback is not None else None)
    put("checkpoint_resume_qualified", None)
    put("checkpoint", _clean_checkpoint(copy.deepcopy(state.checkpoint)))
    put("storage_bytes", _int_or_none("storage_bytes", state.storage_bytes))
    put("storage_cap_bytes", _int_or_none("storage_cap_bytes", state.storage_cap_bytes))
    for key in COUNTER_KEYS:
        put(key, None, _COUNTER_REASON)
    doc["aws_lambda_calls"] = 0
    put("gpu", None)
    put("host_guard", None)

    metrics = dict.fromkeys(METRIC_KEYS)
    if state.metrics:
        unexpected = sorted(set(state.metrics) - set(METRIC_KEYS))
        if unexpected:
            raise LiveRefused(f"unknown metrics keys {unexpected}")
        if any(k.startswith("heldout") and state.metrics.get(k) is not None for k in METRIC_KEYS):
            raise LiveRefused("heldout metrics appear only as unsealed milestone values, never from the trainer")
        if verified:
            metrics.update({k: _number_or_none(f"metrics.{k}", v) for k, v in state.metrics.items()})
    for key in METRIC_KEYS:
        if metrics[key] is None:
            unknown.append({"field": f"metrics.{key}", "reason": _METRIC_REASON if verified else _WITHHELD})
    doc["metrics"] = metrics
    doc["counter_scope"] = f"trainer run {run_id} only"
    doc["contract"] = {
        "id": CONTRACT_ID,
        "producer_mode": "TRAINER",
        "trainer_phase": state.phase,
        "observer_expiry_s": OBSERVER_EXPIRY_S,
        "consumer_pins": dict(CONSUMER_PINS),
    }
    for key in LAUNCHER_KEYS:
        if key in base and base[key] is None:
            unknown.append({"field": key, "reason": "null in the audit launcher document; carried unchanged"})
    doc["unknown"] = unknown

    _check_finite(doc)
    data = encode_document(doc)
    errors = validate(json.loads(data), schema)
    if errors:
        raise LiveRefused("live document fails claude/live.schema.json: " + "; ".join(errors[:12]))
    return doc


class LiveWriter:
    """Atomic writer of trainer-mode documents to one caller-given path, at least every ``interval_s`` <= 3 s.

    ``update(state)`` stores the latest snapshot; the daemon thread started by ``start()`` republishes it on a
    fixed schedule; ``stop()`` joins the thread and publishes the final snapshot. The first error raised in the
    thread stops it and is kept in ``error`` (§6.7 Freshness; the trainer turns it into FAILED).
    """

    def __init__(self, out_path, *, schema: Mapping, base: Mapping, interval_s: float = MAX_INTERVAL_S):
        if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) \
                or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError(f"interval_s must be a positive finite number, got {interval_s!r}")
        if interval_s > MAX_INTERVAL_S:
            raise ValueError(f"interval_s {interval_s} > {MAX_INTERVAL_S}: the 45 s observer expiry needs <= 3 s")
        out_path = Path(out_path)
        assert_outside_claude(out_path, "live output path")
        self.out_path = out_path
        self.schema = schema
        self.base = copy.deepcopy(dict(base))
        self.interval_s = float(interval_s)
        self.error: BaseException | None = None
        self.writes = 0
        self.last: dict | None = None
        self._state: LiveState | None = None
        self._state_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def publish(self, state: LiveState) -> dict:
        """Build, validate and atomically write one document now; returns ``{path, sha256, bytes, utc_us}``."""
        with self._publish_lock:
            doc = build_document(self.base, state, schema=self.schema)
            res = common.atomic_write(self.out_path, encode_document(doc))
            self.writes += 1
            self.last = {"path": res["path"], "sha256": res["sha256"], "bytes": res["bytes"], "utc_us": doc["utc_us"]}
            return dict(self.last)

    def update(self, state: LiveState) -> None:
        if not isinstance(state, LiveState):
            raise TypeError("state must be a LiveState")
        with self._state_lock:
            self._state = state

    def snapshot(self) -> LiveState | None:
        with self._state_lock:
            return self._state

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="dltrain-live-writer", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        next_t = time.monotonic()
        try:
            while not self._stop_event.is_set():
                state = self.snapshot()
                if state is not None:
                    self.publish(state)
                next_t += self.interval_s
                now = time.monotonic()
                if next_t < now:
                    next_t = now
                if self._stop_event.wait(next_t - now):
                    return
        except BaseException as exc:  # noqa: BLE001 - the first thread error is reported to the trainer
            if self.error is None:
                self.error = exc

    def stop(self) -> dict | None:
        """Join the thread and publish the final snapshot; a failure is recorded in ``error``, not raised."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        state = self.snapshot()
        if state is None:
            return None
        try:
            return self.publish(state)
        except Exception as exc:  # noqa: BLE001
            if self.error is None:
                self.error = exc
            return None


def _remote_constant(bridge_text: str) -> str:
    for node in ast.parse(bridge_text).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str) \
                and any(isinstance(t, ast.Name) and t.id == "REMOTE" for t in node.targets):
            return node.value.value
    raise LiveRefused("re-pin required: dl_bridge.py has no top-level string REMOTE assignment")


def bridge_program(source_path) -> tuple[str, dict]:
    """Rebuild Codex's bridge program from pinned files and point it at ``source_path`` (§5 P7 test 8).

    The files are read, never imported. Any pin drift raises ``LiveRefused('re-pin required ...')``; the source
    line must occur exactly once and is the only substitution.
    """
    source_path = Path(source_path)
    if not source_path.is_absolute():
        raise LiveRefused("bridge source path must be absolute")
    assert_outside_claude(source_path, "bridge source path")
    texts, pins = {}, {}
    for name in ("dl_bridge.py", "dl_feed.py"):
        raw = (CONSUMER_DIR / name).read_bytes()
        digest = _sha256(raw)
        if digest != CONSUMER_PINS[name]:
            raise LiveRefused(f"re-pin required: {name} sha256 {digest} != pinned {CONSUMER_PINS[name]}")
        texts[name], pins[name] = raw.decode("utf-8"), digest
    program = texts["dl_feed.py"] + "\n" + _remote_constant(texts["dl_bridge.py"])
    program_sha = _sha256(program.encode("utf-8"))
    if program_sha != BRIDGE_PROGRAM_SHA256:
        raise LiveRefused(f"re-pin required: bridge program sha256 {program_sha} != {BRIDGE_PROGRAM_SHA256}")
    count = program.count(SOURCE_LINE)
    if count != 1:
        raise LiveRefused(f"bridge source line occurs {count} times; exactly one substitution is allowed")
    substituted = program.replace(SOURCE_LINE, f"p=pathlib.Path({str(source_path)!r})", 1)
    return substituted, {
        "dl_bridge.py": pins["dl_bridge.py"],
        "dl_feed.py": pins["dl_feed.py"],
        "program_sha256": program_sha,
        "substituted_program_sha256": _sha256(substituted.encode("utf-8")),
    }
