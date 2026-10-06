"""Deterministic synthetic P5 fixtures (FIXTURE_VERSION 'jope.dl-p5-fixture/v1').

- Real-shaped rows. The key layout of native evaluation rows, history-outcome (hwm-event) rows and transport rows
  follows the read-only 2026-10-05 inventory of frozen 014b93f43dbaa1a4__r0__f0 and incoming item ac2886b2... Every
  value is synthetic, and every exposed field carries a unique planted sentinel (Sentinels), so a leak scan can
  find it.
- JTT1 inputs (jtt1_raw), prepared.json documents (prepared_doc) and tiny tokenizers (write_tokenizer).
- synth_context(spec): one context's native, outcome and transport lines, seed pool, bootstrap results, JTT1 input,
  prepared.json, packer summary and the expected outcome class and backend of every program. The fixture computes
  these expectations from its own construction table, never through dltrain.corpus_schema.
- World(root): a file-backed epoch in the layouts dltrain.programs reads. It holds frozen and incoming context
  trees, the incoming cursor and transport index, the exposed side inputs, the sealed-inventory projection, a
  partition map, seeds, a JTT1 cache and a tokenizer built from train_fit programs.

The only randomness is h(tag), the first 8 bytes of sha256('jope.dl-p5-fixture/v1:' + tag). Nothing is read from
the real corpus or S3. World writes only under the root it is given, which tests take from
tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']).

Write guard (the tests_p3 pattern): importing this module installs a sys.addaudithook hook, armed for the whole
process, that records every audited write (open for writing, mkdir, chmod, remove, rmdir, rename, link, symlink,
truncate, utime, copyfile) whose resolved target is outside ALLOWED_ROOT = realpath(TMPDIR). Every P5 test module
imports this module, so the hook is live before the first test runs; tests_p5.test_corpus_hygiene asserts it is live
and clean, and tests_p5/__init__.load_tests appends a final check that the whole suite recorded zero violations.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import gzip
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FIXTURE_VERSION = "jope.dl-p5-fixture/v1"
TMP_ROOT = os.environ.get("TMPDIR") or ""
if not TMP_ROOT or not os.path.isdir(TMP_ROOT):
    raise RuntimeError("TMPDIR must name an existing directory (RUN/tmp); P5 tests never fall back to /tmp")
if os.path.realpath(TMP_ROOT) == "/tmp" or os.path.realpath(TMP_ROOT).startswith("/tmp/"):
    raise RuntimeError("TMPDIR must not be under /tmp (it is emptied at boot)")
ALLOWED_ROOT = os.path.realpath(TMP_ROOT)


# ------------------------------------------------------------------------------------------------ write guard

_GUARD: dict[str, Any] = {"armed": True, "busy": False, "violations": [], "trace": None}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_DIR_FD_ARG = {"os.mkdir": 2, "os.chmod": 2, "os.remove": 1, "os.rmdir": 1, "os.utime": 3, "os.chown": 3}


def under_root(path: str, root: str | None = None) -> bool:
    root = ALLOWED_ROOT if root is None else root
    return path == root or path.startswith(root + os.sep)


def _resolve(path, dir_fd=-1):
    if isinstance(path, int):
        return None                       # fd-based: its path was audited when the descriptor was opened
    path = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
    if not os.path.isabs(path):
        base = os.readlink(f"/proc/self/fd/{dir_fd}") if isinstance(dir_fd, int) and dir_fd >= 0 else os.getcwd()
        path = os.path.join(base, path)
    return os.path.realpath(path)


def write_target(event: str, args) -> str | None:
    """The resolved path an audited event writes, or None for reads and for events that write nothing. Opening the
    null device read-write (subprocess does it for DEVNULL handles, e.g. ctypes.util.find_library's ldconfig call
    during `import torch` in cpuguard.set_threads) creates and changes nothing, so it is not a write."""
    try:
        if event == "open":
            path, mode, flags = args
            writes = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or \
                     (isinstance(flags, int) and flags >= 0 and bool(flags & _WRITE_FLAGS))
            target = _resolve(path) if writes else None
            return None if target == os.devnull else target
        if event in _DIR_FD_ARG:
            i = _DIR_FD_ARG[event]
            return _resolve(args[0], args[i] if len(args) > i else -1)
        if event == "os.rename":
            src, dst = _resolve(args[0], args[2]), _resolve(args[1], args[3])
            return next((p for p in (src, dst) if p is not None and not under_root(p)), dst)
        if event == "os.link":
            return _resolve(args[1], args[3])
        if event == "os.symlink":
            return _resolve(args[1], args[2])
        if event == "os.truncate":
            return _resolve(args[0])
        if event == "shutil.copyfile":
            return _resolve(args[1])
    except Exception:  # noqa: BLE001 (an unresolvable target is itself a violation)
        return f"<unresolved {event}>"
    return None


def _audit_hook(event, args):
    if not _GUARD["armed"] or _GUARD["busy"]:
        return
    _GUARD["busy"] = True
    try:
        trace = _GUARD["trace"]
        if trace is not None and event == "open":
            try:
                path = _resolve(args[0])
            except Exception:  # noqa: BLE001
                path = None
            if path is not None:
                trace.append(("open", path))
        target = write_target(event, args)
        if target is not None and not under_root(target):
            _GUARD["violations"].append((event, target))
    finally:
        _GUARD["busy"] = False


sys.addaudithook(_audit_hook)


def violations() -> list:
    """Every write outside ALLOWED_ROOT recorded since the suite started."""
    return list(_GUARD["violations"])


@contextlib.contextmanager
def trace_opens(timeline: list):
    """Append ('open', resolved path) to `timeline` for every audited open while the block runs."""
    old = _GUARD["trace"]
    _GUARD["trace"] = timeline
    try:
        yield timeline
    finally:
        _GUARD["trace"] = old

EVALUATOR_SHA256 = "7ae59dca572a034740b1428aba3a0c5b4c7f12ebec86ba2b8fc286e88546e826"
INNER_CV_SCHEMA = "jope.inner-cv/exact-feature-component-stratified-2fold/v1"
BUCKET = "veox-jopedime-learning-use2"
AWS_FN = "jope-gp-hwm-hybrid-x86-20261001"
NATIVE_SCORED = "jope.island-scored-evaluation/v1"
NATIVE_ABANDON = "jope.island-abandonment/v1"
HWM_EVENT = "jope.per-fold-hwm-event/v1"
REGIONAL = "jope.hwm-regional-transport/v1"
HYBRID = "jope.hwm-hybrid-transport/v1"
INCOMING_LOCAL = "jope.incoming-local-transport/v1"
FROZEN_FOLD0_TEXT = "frozen_fold0_by_construction(clean views substituted; build_campaign.py:236-247)"
FROZEN_TEXT = "frozen_fold1_4: P5 target-leak recompute pending"
INCOMING_TEXT = "incoming: intake-qualified at sealing (intake_poll.py:96-108); P5 target-leak recompute pending"
SHARED_SEEDS = ("(pipe X (standardize) (ridge_gcv))", '(sym affine "(c0*-0.5)")',
                "(portfolio (member (pipe X (norm_row) (krr))) (member (pipe X (ridge_gcv))) "
                "(blend folds=5 guard=clip kind=nnls repeats=1))")


def h(tag: str) -> int:
    return int.from_bytes(hashlib.sha256((FIXTURE_VERSION + ":" + tag).encode()).digest()[:8], "big")


def u(tag: str) -> float:
    return h(tag) / 2**64


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(row: Mapping) -> bytes:
    """One JSONL line as the producers write it (compact, insertion order, NaN/Infinity tokens allowed)."""
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


def tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="p5-", dir=os.environ["TMPDIR"])


# ------------------------------------------------------------------------------------------------- sentinels

class Sentinels:
    """Unique planted values for exposed fields, recorded so a leak scan can look for them."""

    def __init__(self, tag: str, poison: int = 0):
        self.tag, self.poison, self.n = tag, poison, 0
        self.floats: list[float] = []
        self.f32: list[float] = []
        self.ints: list[int] = []
        self.strings: list[str] = []

    def _k(self) -> int:
        self.n += 1
        return self.n

    def f(self) -> float:
        k = self._k()
        v = 271828.0 + (h(f"{self.tag}:f:{k}") % 2**36) / 2**16 + 1e6 * self.poison
        self.floats.append(v)
        return v

    def i(self) -> int:
        k = self._k()
        v = 9_100_000_000 + (h(f"{self.tag}:i:{k}") % 10**8) * 10 + self.poison
        self.ints.append(v)
        return v

    def s(self, label: str) -> str:
        k = self._k()
        v = f"SENTINEL-{label}-{self.tag}-{k}-p{self.poison}"
        self.strings.append(v)
        return v

    def hex(self) -> str:
        k = self._k()
        v = sha(f"SENTINEL:{self.tag}:{k}:{self.poison}")
        self.strings.append(v)
        return v

    def b64(self, n: int) -> str:
        k = self._k()
        vals = np.array([4096.0 + ((h(f"{self.tag}:v:{k}:{j}") % 4096) + 64 * self.poison) / 64.0 for j in range(n)],
                        dtype="<f4")
        self.f32.extend(float(v) for v in vals)
        return base64.b64encode(vals.tobytes()).decode("ascii")

    def metric_block(self) -> dict:
        return {"mae": self.f(), "max_abs_error": self.f(), "n": self.i(), "r2": self.f(), "rmse": self.f(),
                "sse": self.f()}


# ------------------------------------------------------------------------------------------- row builders

def program_obj(dsl: str, *, operator: str = "typed_mutation_v4", parents: Sequence[str] = (), birth_seq: int = 1) -> dict:
    return {"birth_seq": birth_seq, "dsl": dsl, "operator": operator, "parents": list(parents), "program_sha256": sha(dsl)}


def cv_block(base: Sequence[float], a: float, skew: float = 0.02) -> dict:
    """Two-fold inner-CV metrics against the baseline pair; the evaluator gate uses the labels arithmetic and
    cv_log_ratio_sd the native engine's two-pass population SD (mean, then the mean squared deviation, then sqrt)."""
    rmse = [base[0] * math.exp(-a), base[1] * math.exp(-a + skew)]
    ratios = [rmse[0] / base[0], rmse[1] / base[1]]
    logs = [math.log(ratios[0]), math.log(ratios[1])]
    mean = (logs[0] + logs[1]) / 2
    sd = math.sqrt(((logs[0] - mean) * (logs[0] - mean) + (logs[1] - mean) * (logs[1] - mean)) / 2)
    return {"rmse_j": rmse, "cv_mean_rmse_ratio": 0.5 * (ratios[0] + ratios[1]),
            "cv_mean_log_ratio": 0.5 * (logs[0] + logs[1]), "cv_log_ratio_sd": sd,
            "cv_gate_passed": 0.5 * (rmse[0] / base[0] + rmse[1] / base[1]) <= 1.05,
            "cv_geomean_rmse_ratio": math.exp(0.5 * (logs[0] + logs[1])), "cv_tail_log_ratio": max(logs),
            "cv_fold_wins": sum(r < 1.0 for r in ratios)}


def native_row(ident: Mapping, prog: Mapping, status: str, *, k: int, x: Sentinels, metrics: Mapping | None = None,
               reason: str | None = None, policy: str = "test_gated", population_selected: bool = False) -> dict:
    """A native evaluation line: island-scored-evaluation/v1, or island-abandonment/v1 when status is abandoned."""
    cid = ident["context_sha256"]
    if status == "abandoned":
        return {"context_sha256": cid, "controller_recorded_utc_ns_text": str(1790907916701175835 + k),
                "island_id": k % 8, "policy": "novelty_quality", "program": dict(prog),
                "program_sha256": prog["program_sha256"], "proposal_draw": 17000 + k, "proposal_seq": 400 + k,
                "reason": reason, "schema": NATIVE_ABANDON, "status": "abandoned"}
    if status in ("failed", "timeout"):
        return {"context_sha256": cid, "island_id": k % 8, "policy": policy, "program": dict(prog),
                "proposal_draw": 9000 + k, "proposal_seq": 200 + k, "result_key": x.s("result-key"),
                "schema": NATIVE_SCORED, "status": status}
    m = metrics or {}
    return {"born_completed": k // 2, "complexity_open_parens": 4, "context_sha256": cid,
            "cv_fold_wins": m.get("cv_fold_wins", 2), "cv_gate_passed": m.get("cv_gate_passed"),
            "cv_geomean_rmse_ratio": m.get("cv_geomean_rmse_ratio"), "cv_log_ratio_sd": m.get("cv_log_ratio_sd"),
            "cv_mean_log_ratio": m.get("cv_mean_log_ratio"), "cv_tail_log_ratio": m.get("cv_tail_log_ratio"),
            "fitness": x.f(), "full_prediction_sha256": x.hex(), "island_id": k % 8, "migration_epoch": k % 3,
            "origin_island": (k + 1) % 8, "policy": policy, "population_selected": population_selected,
            "program": dict(prog), "proposal_draw": 100 + k, "proposal_seq": 4 + k, "result_key": x.s("result-key"),
            "roots": [sha(f"root:{cid}:{k}:a"), sha(f"root:{cid}:{k}:b")],
            "sampled_absolute_error_f32_b64": x.b64(192), "schema": NATIVE_SCORED,
            "semantic_duplicate_of_active": k % 5 == 4, "semantic_signature_f32_b64": x.b64(64), "status": "completed",
            "test_rmse": x.f(), "test_rmse_ratio": x.f()}


def _fit_block(x: Sentinels, index: int, kind: str, rmse: float | None, base: float | None) -> dict:
    metrics = x.metric_block() if kind == "official_test" else {"mae": 0.1, "max_abs_error": 0.7, "n": 800,
                                                                   "r2": 0.3, "rmse": rmse, "sse": 24.5}
    baseline = x.metric_block() if kind == "official_test" else {"mae": 0.15, "max_abs_error": 0.78, "n": 800,
                                                                    "r2": -0.0002, "rmse": base, "sse": 36.4}
    training = x.metric_block() if kind == "official_test" else {"mae": 0.11, "max_abs_error": 0.65, "n": 800,
                                                                    "r2": 0.48, "rmse": 0.15, "sse": 18.7}
    if kind == "official_test":
        return {"baseline_metrics": baseline, "finished_utc_us": x.i(), "fit_finished_utc_us": x.i(), "index": x.i(),
                "kind": x.s("kind"), "metrics": metrics, "model_sha256": x.hex(), "schema": x.s("schema"),
                "score_rows": x.i(), "serialized_prediction_replay_exact": True, "started_utc_us": x.i(),
                "train_rows": x.i(), "training_metrics": training, "wall_us": x.i()}
    return {"baseline_metrics": baseline, "finished_utc_us": 1790907123629639 + index,
            "fit_finished_utc_us": 1790907123629374 + index, "index": index, "kind": kind, "metrics": metrics,
            "model_sha256": sha(f"model:{index}:{rmse}"), "schema": "jope.teacher-fit/v1", "score_rows": 800,
            "serialized_prediction_replay_exact": True, "started_utc_us": 1790907123380695 + index,
            "train_rows": 800, "training_metrics": training, "wall_us": 248943 + index}


def evaluation_detail(x: Sentinels, rmse: Sequence[float | None] | None, base: Sequence[float | None] | None, *,
                      order: Sequence[int] = (0, 1), official: bool = True) -> dict:
    d = {"schema": "jope.gp-signal-evaluation/v7",
         "memory": ["VmPeak:\t   13488 kB", "VmHWM:\t   10264 kB", "VmRSS:\t   10264 kB", "VmSwap:\t       0 kB"],
         "inner_cv_mode": "partitioned_kfold"}
    if rmse is not None:
        d["inner_cv"] = [_fit_block(x, j, "inner_cv", rmse[j], base[j]) for j in order]
    if official:
        d["official_test"] = _fit_block(x, 0, "official_test", None, None)
    return d


def outcome_row(ident: Mapping, prog: Mapping, *, phase: str, record_status: str, transport_status: str | None,
                generation: int, x: Sentinels, node: str | None = None, backend: str | None = None,
                attempt_id: str | None = None, wall_us: int | None = None, metrics: Mapping | None = None,
                base: Sequence[float] | None = None, detail: bool = True, with_program: bool = True) -> dict:
    """A history-outcome (per-fold-hwm-event/v1) line with the real key layout."""
    row: dict[str, Any] = {"schema": HWM_EVENT, "utc_ns_text": str(1790907139080467900 + generation),
                           "view_id": ident["view_id"], "dataset_hash": ident["dataset_hash"],
                           "outer_fold": ident["fold"], "context_sha256": ident["context_sha256"], "phase": phase,
                           "generation": generation}
    if record_status == "missing":
        row.update(program_sha256=prog["program_sha256"], record_status="missing", transport_status=transport_status)
        if with_program:
            row["program"] = dict(prog)
        row.update(result_ref=None, error="HTTPError:HTTP Error 502: Bad Gateway")
        if attempt_id is not None:
            row["attempt_id"] = attempt_id
        if backend is not None:
            row["backend"] = backend
        row.update(new_training_requested=False, evaluation_missing=True)
        return row
    row.update(canonical_result_version_bound=True, program_sha256=prog["program_sha256"], record_status=record_status,
               transport_status=transport_status)
    if node is not None:
        row["node"] = node
    if backend is not None:
        row["backend"] = backend
    row.update(new_training_requested=True,
               result_ref={"bucket": x.s("bucket"), "key": x.s("result-ref-key"), "version_id": x.s("version"),
                           "sha256": x.hex()},
               max_memory_mb=11, wall_us=wall_us, evaluation_reuse=None)
    if attempt_id is not None:
        row.update(attempt_id=attempt_id, worker_started_utc_us=1790907123380634, worker_finished_utc_us=1790907124768859)
    if record_status == "completed" and detail:
        m = metrics or {}
        row["evaluation_detail"] = evaluation_detail(x, m.get("rmse_j"), base)
    if with_program:
        row["program"] = dict(prog)
    row["tabpfn35_rmse"] = x.f()
    if record_status == "completed":
        m = metrics or {}
        row.update(fitness=x.f(), cv_gate_passed=m.get("cv_gate_passed"), cv_fold_wins=m.get("cv_fold_wins", 2),
                   cv_geomean_rmse_ratio=m.get("cv_geomean_rmse_ratio"), cv_mean_rmse_ratio=m.get("cv_mean_rmse_ratio"),
                   cv_mean_log_ratio=m.get("cv_mean_log_ratio"), cv_tail_log_ratio=m.get("cv_tail_log_ratio"),
                   cv_log_ratio_sd=m.get("cv_log_ratio_sd"), test_rmse=x.f(), test_rmse_ratio=x.f(),
                   complexity_open_parens=8, gp_over_tabpfn35=x.f())
    return row


def transport_row(ident: Mapping, prog: Mapping | None, *, attempt_id: str | None, pool: str | None, kind: str,
                  k: int = 0) -> dict:
    """A transport attempt: kind regional (aws pool), hybrid (local pool, frozen) or incoming (index row)."""
    cid = ident["context_sha256"]
    base: dict[str, Any] = {}
    if kind in ("regional", "hybrid"):
        base = {"_source_transport_journal": "regional-events-v6.jsonl", "_telemetry_view_context_sha256": cid}
    if attempt_id is not None:
        base["attempt_id"] = attempt_id
    if kind == "regional":
        base.update({"archive_bytes": 26586 + k, "archive_sha256": sha(f"archive:{cid}:{k}"),
                     "aws_invoke_wall_ns": 5144732759 + k, "aws_request_id": f"req-{k:08d}", "backend": "aws_lambda",
                     "billing": {"billed_ms": 4707.0, "list_price_usd": 3.9e-05, "max_memory_used_mb": 71,
                                 "memory_mb": 512},
                     "context_sha256": cid, "estimated_overhead_usd": 0.000139, "finished_utc_ns_text": "1790907128585162760",
                     "function": AWS_FN, "function_version": "1", "input_cache_hit": False,
                     "input_transport": "canonical_pinned_url", "invoke_started_utc_ns_text": "1790907123331551801"})
    else:
        base.update({"backend": "local_lambda", "context_sha256": cid, "finished_utc_ns_text": "1790907125024541374"})
        if kind == "incoming":
            base["http_status"] = 200
    if pool is not None:
        base["pool"] = pool
    if prog is not None:
        base["program"] = dict(prog)
        base["program_sha256"] = prog["program_sha256"]
    if kind == "regional":
        base.update({"region": (pool or "aws:none").split(":")[1] if pool and pool.startswith("aws:") else None,
                     "reservation_wait_ns": 22979995, "runtime_error": None, "schema": REGIONAL,
                     "settlement_wait_ns": 108841400})
    else:
        base["schema"] = HYBRID if kind == "hybrid" else INCOMING_LOCAL
    base.update({"started_utc_ns_text": "1790907123308568370", "wall_ns": 5276594640 + k})
    return base


# ------------------------------------------------------------------------------------------------- JTT1 inputs

def jtt1_matrix(n_train: int, n_test: int, p: int, *, tag: str, leak: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """X[n, p] and y[n] (outer-train rows first). leak: exact | affine | test_only | train_only on the last column."""
    n = n_train + n_test
    X = np.empty((n, p), dtype=np.float64)
    for f in range(p):
        phase = u(f"{tag}:x:{f}")
        for i in range(n):
            X[i, f] = math.sin(0.37 * (i + 1) * (f + 1) + phase) + 0.01 * i
    y = np.array([math.cos(0.21 * i) + 0.5 * X[i, 0] - 0.25 * X[i, min(1, p - 1)] ** 2 + 0.1 * u(f"{tag}:y:{i}")
                  for i in range(n)], dtype=np.float64)
    if leak == "exact":
        X[:, p - 1] = y
    elif leak == "affine":
        X[:, p - 1] = 2.0 * y + 3.0
    elif leak == "test_only":
        X[n_train:, p - 1] = y[n_train:]
    elif leak == "train_only":
        X[:n_train, p - 1] = y[:n_train]
    elif leak is not None:
        raise ValueError(f"unknown leak {leak!r}")
    return X, y


def jtt1_raw(X: np.ndarray, y: np.ndarray, n_train: int, *, inner_folds: int = 2, magic: bytes = b"JTT1",
             fold_labels: bytes | None = None) -> bytes:
    n, p = X.shape
    labels_u8 = fold_labels if fold_labels is not None else bytes(i % 2 for i in range(n_train))
    mat = np.concatenate([X, y[:, None]], axis=1).astype("<f8")
    return magic + struct.pack("<IIIB", n_train, n - n_train, p, inner_folds) + bytes(labels_u8) + mat.tobytes()


def test_range(raw: bytes) -> tuple[int, int]:
    """[start, end) of the outer-test bytes of a raw JTT1 input (the fixture's own layout arithmetic)."""
    n_train, n_test, p, _ = struct.unpack("<IIIB", raw[4:17])
    start = 17 + n_train + 8 * (p + 1) * n_train
    return start, start + 8 * (p + 1) * n_test


def xor_test_rows(raw: bytes, key: int = 0x5A) -> bytes:
    a, b = test_range(raw)
    return raw[:a] + bytes(c ^ key for c in raw[a:b]) + raw[b:]


def gz(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=3, mtime=0)


def input_ref(raw: bytes, body: bytes) -> dict:
    rs = sha_bytes(raw)
    return {"bucket": BUCKET, "bytes": len(body), "compression": "gzip-3-mtime0",
            "key": f"experiments/gp-teacher-v6/objects/inputs/sha256/{rs}.jtt1.gz", "raw_bytes": len(raw),
            "raw_sha256": rs, "sha256": sha_bytes(body), "version_id": "v-" + rs[:24]}


def prepared_doc(ident: Mapping, n_train: int, n_test: int, p: int, *, raw: bytes | None = None,
                 overrides: Mapping | None = None) -> dict:
    """A prepared.json with the real key layout; the never-read keys hold sentinels."""
    ctx = {"benchmark_claim_allowed": False, "candidate_timeout_ms": 90000, "context_sha256": ident["context_sha256"],
           "dataset_hash": ident["dataset_hash"], "encoded_features": p, "evaluator_sha256": EVALUATOR_SHA256,
           "execution_backend": "aws_lambda", "inner_cv_policy": "exact_feature_component_stratified",
           "inner_cv_schema": INNER_CV_SCHEMA, "input_raw_sha256": sha_bytes(raw) if raw else "0" * 64,
           "input_ref": {"bucket": BUCKET, "key": "k", "version_id": "v", "sha256": "0" * 64, "bytes": 1},
           "intentional_test_adaptation": True, "membership_logical_sha256": "0" * 64,
           "membership_ref": {"bucket": BUCKET, "key": "m", "version_id": "v", "sha256": "0" * 64, "bytes": 1},
           "outer_fold": ident["fold"], "outer_repeat": 0, "reference_contract": "SENTINEL-reference-contract",
           "route": "SENTINEL-route", "sample_identity_sha256": "0" * 64, "schema": "jope.gp-evaluation-context/v5",
           "task": "regression", "test_rows": n_test, "train_rows": n_train}
    ctx.update(overrides or {})
    return {"context": ctx, "context_ref": {"bucket": BUCKET, "key": "c", "version_id": "v", "sha256": "0" * 64, "bytes": 1},
            "schema": "jope.gp-prepared-context/v1",
            "source_manifest": {"bucket": BUCKET, "key": "s", "version_id": "v", "sha256": "0" * 64, "bytes": 1,
                                "reused": True},
            "view_id": ident["view_id"]}


# ------------------------------------------------------------------------------------------------ tokenizer

def tiny_vocab(dsls: Iterable[str]):
    from dltrain import dsl_tokenizer
    return dsl_tokenizer.build_vocab(sorted(set(dsls)))[0]


def program_set_sha256(shas: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for s in sorted(set(shas)):
        hasher.update(s.encode("ascii") + b"\n")
    return hasher.hexdigest()


def write_tokenizer(tok_dir: Path, train_dsls: Iterable[str], *, max_seq: int | None = None) -> tuple[Any, int]:
    """vocab.json and lengths.json built from distinct train_fit programs only (dsl_tokenizer.build_vocab);
    max_seq = ceil32(train_fit max length) unless given."""
    from dltrain import dsl_tokenizer
    dsls = sorted(set(train_dsls))
    vocab = dsl_tokenizer.build_vocab(dsls)[0]
    longest = max(len(dsl_tokenizer.tokenize(d, vocab)) for d in dsls)
    max_seq = dsl_tokenizer.ceil32(longest) if max_seq is None else max_seq
    closure = {"partition": "train_fit", "distinct_programs": len(dsls),
               "program_set_sha256": program_set_sha256(sha(d) for d in dsls),
               "program_set_rule": "sha256 of the sorted distinct program_sha256 hex values, one per line"}
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "vocab.json").write_bytes(vocab.to_json(extra={"input_closure": closure}))
    lengths = {"schema": dsl_tokenizer.LENGTHS_SCHEMA, "vocab_sha256": vocab.sha256, "input_closure": closure,
               "max_length": longest, "max_seq": max_seq}
    (tok_dir / "lengths.json").write_bytes((json.dumps(lengths, indent=1, sort_keys=True) + "\n").encode())
    return vocab, max_seq


# --------------------------------------------------------------------------------------- one synthetic context

# Native program kinds, cycled by index: (status, reason, outcome-row kind).
NATIVE_KINDS = (
    ("completed", None, "detailed_local"),          # node xbabe3; hybrid transport (a duplicate attempt row on k=0)
    ("completed", None, "detailless"),              # generation 0-4: no node, attempt or inner-CV detail
    ("failed", None, "failed_local"),
    ("completed", None, "detailed_aws_node"),       # node aws:<region>:<function>; regional transport
    ("timeout", None, "timeout_local"),
    ("abandoned", "transport_error", "missing_attempt"),             # evolution row missing, attempt, no transport
    ("completed", None, "detailed_attempt_aws"),    # no node; attempt joins an aws pool; row backend local_lambda
    ("abandoned", "capacity_deferred_before_claim", "missing"),
    ("completed", None, "detailed_retry"),          # a missing/transport_error row, then the completed row (last wins)
    ("completed", None, "detailed_local"),
    ("failed", None, "failed_local"),
    ("abandoned", "runtime_error", "missing"),
    ("completed", None, "detailed_attempt_local"),  # no node; attempt joins an xbabe pool
    ("completed", None, "detailed_local"),
)
# Bootstrap-only kinds: (record_status, transport_status, row kind). One retried seed (missing, then completed).
BOOT_KINDS = (
    ("completed", "cached", "detailless"),
    ("failed", "completed_transport", "node_local"),
    ("missing", "transport_error", None),
    ("completed", "completed_transport", "detailed_attempt_local"),
    ("missing", "recovered_after_transport_error", None),
    ("timeout", "completed_transport", "node_local"),
)
TONLY_POOLS = (("aws:us-east-1", "aws:eu-west-2"), ("xbabe3-extra", "aws:us-west-2"), ("xbabe1:7431",))
EXPECTED_BACKEND = {
    "detailed_local": ("local", "outcome_node"), "detailless": ("unknown", "none"),
    "failed_local": ("local", "outcome_node"), "detailed_aws_node": ("aws_lambda", "outcome_node"),
    "timeout_local": ("local", "outcome_node"), "missing_attempt": ("unknown", "none"), "missing": ("unknown", "none"),
    "detailed_attempt_aws": ("aws_lambda", "transport_attempt"), "detailed_retry": ("local", "outcome_node"),
    "detailed_attempt_local": ("local", "transport_attempt"), "node_local": ("local", "outcome_node"), None: ("unknown", "none"),
}


@dataclass
class CtxSpec:
    name: str
    cohort: str = "frozen"
    fold: int = 1
    split_role: str = "train_fit"
    family_id: str = "fam-a"
    dataset: str | None = None
    item: str | None = None
    n_native: int = 14
    n_boot_only: int = 6
    boot_overlap: bool = True
    n_tonly: int = 3
    n_long: int = 0
    n_train: int = 48
    n_test: int = 16
    p: int = 5
    leak: str | None = None
    context_schema: str | None = None


@dataclass
class SynthContext:
    spec: CtxSpec
    cid: str
    view_id: str
    dataset_hash: str
    item_id: str | None
    native_rows: list
    outcome_rows: list
    transport_rows: list
    seed_pool_rows: list
    boot_results_rows: list
    prepared: dict
    jtt1_raw: bytes
    jtt1_gz: bytes
    input_ref: dict
    base: tuple
    best0: float
    admitted: int
    expect: dict
    dsl: dict
    sentinels: Sentinels
    history_sha256: str | None = None          # set by World (the epoch history_ref pin)
    native_lines: list = field(default_factory=list)
    outcome_lines: list = field(default_factory=list)
    transport_lines: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.native_lines = [dump(r) for r in self.native_rows]
        self.outcome_lines = [dump(r) for r in self.outcome_rows]
        self.transport_lines = [dump(r) for r in self.transport_rows]

    @property
    def ident(self) -> dict:
        return {"context_sha256": self.cid, "view_id": self.view_id, "dataset_hash": self.dataset_hash,
                "fold": self.spec.fold}

    @property
    def native_bytes(self) -> bytes:
        return b"".join(self.native_lines)

    @property
    def outcome_bytes(self) -> bytes:
        return b"".join(self.outcome_lines)

    @property
    def transport_bytes(self) -> bytes:
        return b"".join(self.transport_lines)

    @property
    def boot_results_bytes(self) -> bytes:
        return b"".join(dump(r) for r in self.boot_results_rows)

    def stats(self) -> dict:
        """pack_fold_history.build_history's summary counts over this context's sources."""
        progs = set()
        for r in self.outcome_rows:
            if r.get("program"):
                progs.add(r["program_sha256"])
        for r in self.native_rows:
            d = (r.get("program") or {}).get("program_sha256")
            if d:
                progs.add(d)
        for r in self.transport_rows:
            if r.get("program"):
                progs.add(r["program_sha256"])
        return {"unique_programs": len(progs), "native_events": len(self.native_lines),
                "native_bytes": len(self.native_bytes), "outcome_bytes": len(self.outcome_bytes),
                "outcome_events": len(self.outcome_lines), "transport_bytes": len(self.transport_bytes),
                "transport_events": len(self.transport_lines)}

    def native_completed(self) -> int:
        return sum(r.get("status") == "completed" for r in self.native_rows)

    def ctx_meta(self, y_train: Any = None) -> dict:
        from dltrain import jtt1
        if y_train is None:
            y_train = jtt1.train_view(self.jtt1_raw)[3]
        return {"context_sha256": self.cid, "context_schema": self.context_schema, "dataset_hash": self.dataset_hash,
                "family_id": self.spec.family_id, "split_role": self.spec.split_role, "y_train": y_train,
                "cohort": self.spec.cohort, "fold": self.spec.fold, "eligibility_source": self.eligibility_source}

    @property
    def context_schema(self) -> str:
        if self.spec.context_schema:
            return self.spec.context_schema
        return "jope.gp-evaluation-context/v5" if self.spec.cohort == "frozen" else "jope.gp-evaluation-context/v8"

    @property
    def eligibility_source(self) -> str:
        if self.spec.cohort == "frozen":
            return FROZEN_FOLD0_TEXT if self.spec.fold == 0 else FROZEN_TEXT
        return INCOMING_TEXT


def high_water(native_rows: Sequence[Mapping], best0: float | None) -> list[bool]:
    """The fixture's own copy of pack_fold_history.build_history lines 103-113."""
    best, flags = best0, []
    for row in native_rows:
        imp = row.get("status") == "completed" and row.get("cv_gate_passed") and (best is None or row["test_rmse"] < best)
        if imp:
            best = row["test_rmse"]
        flags.append(bool(imp))
    return flags


def _native_dsl(cnum: int, k: int) -> str:
    return f"(pipe X (winsor_clip q=0.0{k % 9 + 1}) (histgbm lr=0.{k % 9 + 1}{cnum % 10} seed={cnum}{k:02d}))"


def _boot_dsl(cnum: int, k: int) -> str:
    return f"(pipe X (robust_scale) (ridge lambda_frac=0.{cnum}{k}))"


def _tonly_dsl(cnum: int, k: int) -> str:
    return f"(pipe X (norm_row) (krr alpha=0.{cnum}{k}))"


def _long_dsl(cnum: int, k: int, stages: int = 60) -> str:
    body = " ".join(f"(winsor_clip q=0.0{(j % 9) + 1})" for j in range(stages))
    return f"(pipe X {body} (ridge lambda_frac=0.{cnum}{k}))"


def synth_context(spec: CtxSpec, *, poison: int = 0) -> SynthContext:
    """One synthetic context (spec above). Every program's expected origin, outcome class and backend is recorded in
    `expect` from the construction tables, independently of dltrain.corpus_schema."""
    x = Sentinels(spec.name, poison)
    cnum = h(f"cnum:{spec.name}") % 900 + 100
    dataset = spec.dataset or sha(f"dataset:{spec.name}")[:16]
    cid = sha(f"ctx:{spec.name}")
    if spec.cohort == "frozen":
        view, item_id = f"{dataset}__r0__f{spec.fold}", None
    else:
        view, item_id = f"{dataset}__r0__f{spec.fold}__test_adaptive_{cid[:12]}", sha(f"item:{spec.item or spec.name}")
    ident = {"context_sha256": cid, "view_id": view, "dataset_hash": dataset, "fold": spec.fold}
    scale = 0.15 + 0.1 * u(f"scale:{spec.name}")
    base = (scale * 1.0, scale * 1.2)
    native_rows: list[dict] = []
    evo_rows: list[dict] = []
    boot_rows: list[dict] = []
    transports: list[dict] = []
    expect: dict[str, dict] = {}
    dsl_of: dict[str, str] = {}
    attempt_n = [0]

    def attempt() -> str:
        attempt_n[0] += 1
        return sha(f"attempt:{cid}:{attempt_n[0]}")[:32]

    def local_kind() -> str:
        return "hybrid" if spec.cohort == "frozen" else "incoming"

    natives = [_native_dsl(cnum, k) for k in range(spec.n_native)] + [_long_dsl(cnum, k) for k in range(spec.n_long)]
    for k, dsl in enumerate(natives):
        status, reason, kind = NATIVE_KINDS[k % len(NATIVE_KINDS)] if k < spec.n_native else ("completed", None,
                                                                                               "detailed_local")
        prog = program_obj(dsl, operator="typed_mutation_v4" if k % 2 else "structured_restart",
                           parents=[sha(f"parent:{cid}:{k}")] if k % 3 else [], birth_seq=1500 + k)
        psha = prog["program_sha256"]
        dsl_of[psha] = dsl
        a = 0.8 - 1.6 * u(f"{cid}:a:{k}")
        m = cv_block(base, a) if status == "completed" else {}
        if kind == "detailless":
            m = {key: v for key, v in m.items() if key != "rmse_j"}
        native_rows.append(native_row(ident, prog, status, k=k, x=x, metrics=m, reason=reason,
                                      population_selected=k % 4 == 0))
        gen = 5 + k if kind != "detailless" else k % 5
        wall = int(1e5 + 4e7 * u(f"{cid}:w:{k}"))
        if kind in ("detailed_local", "detailed_retry"):
            att = attempt()
            if kind == "detailed_retry":
                evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status="missing",
                                            transport_status="transport_error", generation=gen, x=x,
                                            attempt_id=attempt(), backend="local_lambda"))
            node = "xbabe3" if kind == "detailed_local" else "xbabe2"
            evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status="completed",
                                        transport_status="completed_transport", generation=gen, x=x, node=node,
                                        backend="local_lambda", attempt_id=att, wall_us=wall, metrics=m, base=base))
            transports.append(transport_row(ident, prog, attempt_id=att, pool=f"{node}-extra", kind=local_kind(), k=k))
            if k == 0:
                transports.append(transport_row(ident, prog, attempt_id=att, pool=f"{node}-extra", kind=local_kind(),
                                                k=k + 100))
        elif kind == "detailless":
            evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status="completed",
                                        transport_status="completed_transport", generation=gen, x=x, wall_us=wall,
                                        metrics=m, base=base, detail=False))
        elif kind in ("failed_local", "timeout_local"):
            att = attempt()
            rs = "failed" if kind == "failed_local" else "timeout"
            wall = 10533 + k if rs == "failed" else 90_009_446
            evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status=rs,
                                        transport_status="completed_transport", generation=gen, x=x, node="xbabe2",
                                        backend="local_lambda", attempt_id=att, wall_us=wall))
            transports.append(transport_row(ident, prog, attempt_id=att, pool="xbabe2-extra", kind=local_kind(), k=k))
        elif kind == "detailed_aws_node":
            att = attempt()
            evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status="completed",
                                        transport_status="completed_transport", generation=gen, x=x,
                                        node=f"aws:ca-central-1:{AWS_FN}", backend="aws_lambda", attempt_id=att,
                                        wall_us=wall, metrics=m, base=base))
            transports.append(transport_row(ident, prog, attempt_id=att, pool="aws:ca-central-1", kind="regional", k=k))
        elif kind in ("detailed_attempt_aws", "detailed_attempt_local"):
            att = attempt()
            evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status="completed",
                                        transport_status="completed_transport", generation=gen, x=x,
                                        backend="local_lambda", attempt_id=att, wall_us=wall, metrics=m, base=base))
            if kind == "detailed_attempt_aws":
                transports.append(transport_row(ident, prog, attempt_id=att, pool="aws:us-east-2", kind="regional", k=k))
            else:
                transports.append(transport_row(ident, prog, attempt_id=att, pool="xbabe1:7431", kind=local_kind(), k=k))
        elif kind in ("missing_attempt", "missing"):
            evo_rows.append(outcome_row(ident, prog, phase="evolution", record_status="missing", transport_status=reason,
                                        generation=gen, x=x, attempt_id=attempt() if kind == "missing_attempt" else None,
                                        backend="local_lambda"))
        cls = {"completed": "completed", "failed": "failed:unclassified", "timeout": "timeout@90000ms"}.get(
            status, f"abandoned:{reason}")
        backend, source = EXPECTED_BACKEND[kind]
        expect[psha] = {"origin": "evolution", "outcome": cls, "backend": backend, "backend_source": source,
                        "kind": kind, "native_index": k, "long": k >= spec.n_native}

    # bootstrap programs: shared seeds, own seeds and (optionally) one native program
    boot_specs: list[tuple[str, tuple]] = [(d, ("completed", "cached", "detailless")) for d in SHARED_SEEDS]
    boot_specs += [(_boot_dsl(cnum, k), BOOT_KINDS[k % len(BOOT_KINDS)]) for k in range(spec.n_boot_only)]
    if spec.boot_overlap and natives:
        boot_specs.append((natives[0], ("completed", "cached", "detailless")))
    for j, (dsl, (rs, ts, kind)) in enumerate(boot_specs):
        prog = program_obj(dsl, operator="historical_seed", birth_seq=1100 + j)
        psha = prog["program_sha256"]
        dsl_of.setdefault(psha, dsl)
        a = 0.6 - 1.2 * u(f"{cid}:boot:{j}")
        m = cv_block(base, a) if rs == "completed" else {}
        wall = int(2e5 + 1e7 * u(f"{cid}:bw:{j}"))
        if j == len(SHARED_SEEDS):           # the first own seed is retried: missing first, then its final row
            boot_rows.append(outcome_row(ident, prog, phase="bootstrap", record_status="missing",
                                         transport_status="transport_error", generation=0, x=x, backend="local_lambda"))
        if rs == "missing":
            boot_rows.append(outcome_row(ident, prog, phase="bootstrap", record_status="missing", transport_status=ts,
                                         generation=0, x=x, backend="local_lambda"))
        elif kind == "detailless":
            boot_rows.append(outcome_row(ident, prog, phase="bootstrap", record_status=rs, transport_status=ts,
                                         generation=0, x=x, wall_us=wall, metrics=m, base=base, detail=False))
        elif kind == "node_local":
            boot_rows.append(outcome_row(ident, prog, phase="bootstrap", record_status=rs, transport_status=ts,
                                         generation=0, x=x, node="xbabe2", backend="local_lambda", attempt_id=attempt(),
                                         wall_us=90_004_000 if rs == "timeout" else wall))
        elif kind == "detailed_attempt_local":
            att = attempt()
            boot_rows.append(outcome_row(ident, prog, phase="bootstrap", record_status=rs, transport_status=ts,
                                         generation=0, x=x, backend="local_lambda", attempt_id=att, wall_us=wall,
                                         metrics=m, base=base))
            transports.append(transport_row(ident, prog, attempt_id=att, pool="xbabe1:7432", kind=local_kind(), k=200 + j))
        if psha in expect:          # the overlap seed: origin stays evolution
            expect[psha]["bootstrap"] = True
            continue
        if rs == "missing":
            cls = f"abandoned:{ts}" if ts in ("transport_error", "capacity_deferred_before_claim", "runtime_error",
                                              "blocked_existing_claim") else "missing"
        else:
            cls = {"completed": "completed", "failed": "failed:unclassified", "timeout": "timeout@90000ms"}[rs]
        backend, source = EXPECTED_BACKEND[kind]
        expect[psha] = {"origin": "bootstrap", "outcome": cls, "backend": backend, "backend_source": source,
                        "kind": kind or "missing", "shared_seed_dsl": dsl in SHARED_SEEDS}

    # transport-only programs and a transport row without a program
    for k in range(spec.n_tonly):
        dsl = _tonly_dsl(cnum, k)
        prog = program_obj(dsl, operator="typed_crossover_v4", birth_seq=1900 + k)
        dsl_of[prog["program_sha256"]] = dsl
        pools = TONLY_POOLS[k % len(TONLY_POOLS)]
        for j, pool in enumerate(pools):
            transports.append(transport_row(ident, prog, attempt_id=attempt(), pool=pool,
                                            kind="regional" if pool.startswith("aws:") else local_kind(), k=300 + j))
        classes = {"aws" if p.startswith("aws:") else "local" for p in pools}
        if len(classes) == 1:
            backend, source = ("aws_lambda" if classes == {"aws"} else "local"), "transport_unanimous"
        else:
            backend, source = "unknown", "none"
        expect[prog["program_sha256"]] = {"origin": "transport_only", "outcome": "missing", "backend": backend,
                                          "backend_source": source, "kind": "transport_only"}
    transports.append({**({"_telemetry_view_context_sha256": cid} if spec.cohort == "frozen" else {}),
                       "attempt_id": "no-program-attempt", "context_sha256": cid, "schema": HYBRID, "wall_ns": 7})

    outcome_rows = boot_rows + evo_rows
    best0 = x.f()
    admitted = sum(high_water(native_rows, best0))
    seed_pool_rows = []
    seen = set()
    for r in boot_rows:
        if r.get("program") and r["program_sha256"] not in seen:
            seen.add(r["program_sha256"])
            seed_pool_rows.append({"family": x.s("family"), "program": dict(r["program"]), "rank": x.i(),
                                   "schema": "jope.recovery-seed/v1", "source_evidence": {"source_test_rmse": x.f()}})
    boot_results_rows = [{"archive_sha256": x.hex(), "context_sha256": cid, "program_sha256": r["program_sha256"],
                          "record_status": r["record_status"], "result_key": x.s("bootstrap-result-key"),
                          "result_path": x.s("result-path"), "status": "completed_transport", "version_id": x.s("v")}
                         for r in boot_rows]
    X, y = jtt1_matrix(spec.n_train, spec.n_test, spec.p, tag=f"jtt1:{spec.name}", leak=spec.leak)
    raw = jtt1_raw(X, y, spec.n_train)
    body = gz(raw)
    prepared = prepared_doc(ident, spec.n_train, spec.n_test, spec.p, raw=raw)
    return SynthContext(spec=spec, cid=cid, view_id=view, dataset_hash=dataset, item_id=item_id,
                        native_rows=native_rows, outcome_rows=outcome_rows, transport_rows=transports,
                        seed_pool_rows=seed_pool_rows, boot_results_rows=boot_results_rows, prepared=prepared,
                        jtt1_raw=raw, jtt1_gz=body, input_ref=input_ref(raw, body), base=base, best0=best0,
                        admitted=admitted, expect=expect, dsl=dsl_of, sentinels=x)


def parse_context(sc: SynthContext):
    """(native, outcome, transport) parsed rows of a synthetic context through dltrain.corpus_schema."""
    from dltrain import corpus_schema as cs
    ident = sc.ident
    native = [cs.parse_native(line, i + 1, ident)[0] for i, line in enumerate(sc.native_lines)]
    outcome = [cs.parse_outcome(line, i + 1, ident)[0] for i, line in enumerate(sc.outcome_lines)]
    transport = [cs.parse_transport(line, i + 1, ident)[0] for i, line in enumerate(sc.transport_lines)]
    return native, outcome, transport


def context_vocab(*contexts: SynthContext):
    return tiny_vocab(d for sc in contexts for d in sc.dsl.values())


# ------------------------------------------------------------------------------------------------- the World

DEFAULT_SPECS = (
    CtxSpec("F0", "frozen", 0, "train_fit", "fam-a"),
    CtxSpec("F1", "frozen", 1, "train_fit", "fam-b", dataset="d0b0b0b0b0b0b0b1"),
    CtxSpec("F3", "frozen", 3, "diag", "fam-b", dataset="d0b0b0b0b0b0b0b1", n_native=9),
    CtxSpec("F2", "frozen", 2, "heldout", "fam-c", n_long=2),
    CtxSpec("I0", "incoming", 0, "train_fit", "fam-d", dataset="d1d1d1d1d1d1d1d1", item="A"),
    CtxSpec("I2", "incoming", 2, "train_fit", "fam-d", dataset="d1d1d1d1d1d1d1d1", item="A", n_native=10),
    CtxSpec("I1", "incoming", 1, "dev", "fam-e", item="B", n_long=1),
)


class World:
    """A file-backed synthetic epoch in the layouts dltrain.programs reads, plus everything P5 adds."""

    def __init__(self, root: Path | str, specs: Sequence[CtxSpec] = DEFAULT_SPECS, *, poison: int = 0,
                 sealed_name: str | None = "S", excluded_name: str | None = "X"):
        from dltrain import programs
        self.root = Path(root)
        self.poison = poison
        self.contexts: dict[str, SynthContext] = {s.name: synth_context(s, poison=poison) for s in specs}
        self.ids = {name: sc.cid for name, sc in self.contexts.items()}
        inc = self.root / "incoming"
        self.sources = programs.Sources(inc / "fold-archive-cursor-v2.json", inc / "transport-index")
        self.sources.incoming_transport_index.mkdir(parents=True, exist_ok=True)
        self.jtt1_cache = self.root / "jtt1-cache"
        self.jtt1_cache.mkdir(parents=True, exist_ok=True)
        self.frozen_gp = self.root / "frozen" / "gp"
        self.frozen_qualified = self.frozen_gp / "qualified-views.json"
        self.paths: dict[str, dict[str, Path]] = {}
        entries, cursor, qualified = [], {}, {}
        items: dict[str, list[SynthContext]] = {}
        for name, sc in self.contexts.items():
            entries.append(self._write_context(sc, qualified))
            if sc.spec.cohort == "incoming":
                items.setdefault(sc.spec.item or name, []).append(sc)
        for item, members in items.items():
            self._write_item(item, members, cursor)
        self.frozen_gp.mkdir(parents=True, exist_ok=True)
        self.frozen_qualified.write_bytes(json.dumps(qualified, sort_keys=True).encode())
        self.sources.incoming_cursor.write_bytes(json.dumps(cursor, sort_keys=True).encode())
        if sealed_name:
            self.ids[sealed_name] = sha(f"ctx:{sealed_name}")
            entries.append({"cohort": "frozen", "status": "included", "exclusion_reason": None,
                            "context_sha256": self.ids[sealed_name], "view_id": "sealed__r0__f1",
                            "dataset_hash": "5ea1ed5ea1ed5ea1", "fold": 1})
        if excluded_name:
            self.ids[excluded_name] = sha(f"ctx:{excluded_name}")
            entries.append({"cohort": "incoming", "status": "excluded", "exclusion_reason": "withdrawn",
                            "context_sha256": self.ids[excluded_name]})
        self.epoch_doc = {"schema": "jope.dl-corpus-epoch/v1", "experiment_id": "p5-synthetic", "epoch": 0,
                          "contexts": entries}
        self.epoch = self.root / "epoch.json"
        self.epoch.write_bytes(json.dumps(self.epoch_doc, sort_keys=True).encode())
        self.epoch_sha256 = sha_bytes(self.epoch.read_bytes())
        pmap = {sc.cid: {"family_id": sc.spec.family_id, "split_role": sc.spec.split_role}
                for sc in self.contexts.values()}
        if sealed_name:
            pmap[self.ids[sealed_name]] = {"family_id": "fam-s", "split_role": "sealed"}
        if excluded_name:
            pmap[self.ids[excluded_name]] = {"family_id": "fam-x", "split_role": "train_fit"}
        pmap[sha("ctx:not-in-epoch")] = {"family_id": "fam-y", "split_role": "dev"}
        self.pmap_doc = pmap
        self.pmap = self.root / "partition-map.json"
        self.pmap.write_bytes(json.dumps(pmap, sort_keys=True).encode())
        boot_seeds = sorted({p for sc in self.contexts.values() for p, e in sc.expect.items()
                             if e.get("shared_seed_dsl")})
        self.seed_set = frozenset(boot_seeds + [sha("(pipe X (never_seen))")])
        self.seeds = self.root / "seeds.json"
        self.seeds.write_bytes(json.dumps(sorted(self.seed_set)).encode())
        inventory = [{"context_sha256": sc.cid, "cohort": sc.spec.cohort,
                      "native": {"completed": sc.native_completed(), "best": sc.sentinels.f()},
                      "gp_rmse": sc.sentinels.f(), "champion": sc.sentinels.s("champion")}
                     for sc in self.contexts.values()]
        self.inventory = self.root / "folds.json.gz"
        self.inventory.write_bytes(gzip.compress(json.dumps(inventory, sort_keys=True).encode(), mtime=0))
        self.expected_completed = {sc.cid: sc.native_completed() for sc in self.contexts.values()}
        train = [d for sc in self.contexts.values() if sc.spec.split_role == "train_fit" for d in sc.dsl.values()]
        self.tok_dir = self.root / "tokenizer"
        self.vocab, self.max_seq = write_tokenizer(self.tok_dir, train)
        self.train_program_set_sha256 = program_set_sha256(
            p for sc in self.contexts.values() if sc.spec.split_role == "train_fit" for p in sc.dsl)

    # -- layout
    def _item_gp(self, item: str) -> Path:
        return self.root / "incoming" / "items" / sha(f"item:{item}") / "run" / "gp"

    def _write_context(self, sc: SynthContext, qualified: dict) -> dict:
        if sc.spec.cohort == "frozen":
            d = self.frozen_gp / "contexts" / sc.view_id
        else:
            d = self._item_gp(sc.spec.item or sc.spec.name) / "contexts" / sc.view_id
        d.mkdir(parents=True, exist_ok=True)
        native = d / f"evaluations-{sc.cid[:16]}.jsonl"
        native.write_bytes(sc.native_bytes)
        (d / "seed-pool.jsonl").write_bytes(b"".join(dump(r) for r in sc.seed_pool_rows))
        prepared = d / "prepared.json"
        prepared.write_bytes(json.dumps(sc.prepared, sort_keys=True).encode())
        (d / "historical-seen.jsonl").write_bytes(dump({"SENTINEL": sc.sentinels.s("historical-seen")}))
        (d / "bootstrap-results.jsonl").write_bytes(sc.boot_results_bytes)
        paths = {"dir": d, "native": native, "prepared": prepared, "seed_pool": d / "seed-pool.jsonl",
                 "bootstrap_results": d / "bootstrap-results.jsonl"}
        history_sha = sha(f"history:{sc.spec.name}:{self.poison}")
        if sc.spec.cohort == "frozen":
            (d / "history-outcomes.jsonl").write_bytes(sc.outcome_bytes)
            (d / "history-transports.jsonl").write_bytes(sc.transport_bytes)
            st = sc.stats()
            ref = {"schema": "jope.per-fold-history-ref/v1", "view_id": sc.view_id, "context_sha256": sc.cid,
                   "outer_fold": sc.spec.fold,
                   "history_ref": {"bucket": BUCKET, "key": f"hist/{sc.cid}.jsonl.gz", "version_id": "hv-" + sc.cid[:8],
                                   "sha256": history_sha, "bytes": 4096},
                   "best_test_rmse": sc.sentinels.f(), "signal_admitted_improvements": sc.admitted,
                   "island_policies": {"test_gated": 1}, "operators": {"typed_mutation_v4": 1},
                   "previous_history_ref": None, "checkpoint_generations": 20, "complete_20_generations": False, **st}
            (d / "history-ref.json").write_bytes(json.dumps(ref, sort_keys=True).encode())
            paths.update(outcome=d / "history-outcomes.jsonl", transport=d / "history-transports.jsonl",
                         summary=d / "history-ref.json")
            qualified[sc.view_id] = {"bootstrap_summary": {"best_signal_test": {"test_rmse": sc.best0,
                                                                                "fitness": sc.sentinels.f()}},
                                     "tabpfn35": {"rmse": sc.sentinels.f()}, "view_id": sc.view_id}
        else:
            tpath = self.sources.incoming_transport_index / f"{sc.cid}.jsonl"
            tpath.write_bytes(sc.transport_bytes)
            paths.update(transport=tpath, outcome=self._item_gp(sc.spec.item or sc.spec.name) / "hwm-events.jsonl")
        self.paths[sc.spec.name] = paths
        (self.jtt1_cache / f"{sc.input_ref['raw_sha256']}.jtt1.gz").write_bytes(sc.jtt1_gz)
        entry = {"cohort": sc.spec.cohort, "status": "included", "exclusion_reason": None, "context_sha256": sc.cid,
                 "view_id": sc.view_id, "dataset_hash": sc.dataset_hash, "fold": sc.spec.fold,
                 "context_schema": sc.context_schema, "eligibility_source": sc.eligibility_source,
                 "history_ref": {"bucket": BUCKET, "key": f"hist/{sc.cid}.jsonl.gz", "version_id": "hv-" + sc.cid[:8],
                                 "sha256": history_sha, "bytes": 4096},
                 "input_ref": dict(sc.input_ref),
                 "native_source": {"path": str(native), "bytes": len(sc.native_bytes),
                                   "sha256": sha_bytes(sc.native_bytes)},
                 "prepared_ref": {"path": str(prepared), "sha256": sha_bytes(prepared.read_bytes()),
                                  "bytes": prepared.stat().st_size}}
        if sc.item_id:
            entry["item_id"] = sc.item_id
        sc.history_sha256 = history_sha
        return entry

    def _write_item(self, item: str, members: Sequence[SynthContext], cursor: dict) -> None:
        gp = self._item_gp(item)
        gp.mkdir(parents=True, exist_ok=True)
        stranger_cid = sha(f"ctx:stranger:{item}")
        strangers = [dump({"schema": HWM_EVENT, "phase": "evolution", "record_status": "completed",
                           "context_sha256": stranger_cid, "program": program_obj(f"(pipe X (ridge lambda_frac=0.{k + 1}))"),
                           "program_sha256": sha(f"(pipe X (ridge lambda_frac=0.{k + 1}))")}) for k in range(3)]
        merged: list[bytes] = []
        longest = max(len(m.outcome_lines) for m in members)
        for k in range(longest):
            for m in members:
                if k < len(m.outcome_lines):
                    merged.append(m.outcome_lines[k])
            if k % 4 == 0 and strangers:
                merged.append(strangers.pop())
        merged.extend(strangers)
        tail = dump({"schema": HWM_EVENT, "phase": "evolution", "context_sha256": members[0].cid, "record_status": "completed"})
        (gp / "hwm-events.jsonl").write_bytes(b"".join(merged) + tail[: len(tail) // 2])     # a live partial tail line
        qualification = {"failures": [], "qualified": [
            {"view_id": m.view_id, "bootstrap_summary": {"best_signal_test": {"test_rmse": m.best0,
                                                                                "fitness": m.sentinels.f()}},
             "tabpfn35": {"rmse": m.sentinels.f()}} for m in members]}
        (gp / "qualification.json").write_bytes(json.dumps(qualification, sort_keys=True).encode())
        for m in members:
            st = m.stats()
            fp = sha_bytes(m.native_bytes + m.outcome_bytes + m.transport_bytes + m.boot_results_bytes)
            cursor[m.cid] = {"schema": "jope.incoming-fold-telemetry-pointer/v2",
                             "stats": {**st, "best_test_rmse": m.sentinels.f(),
                                       "signal_admitted_improvements": m.admitted},
                             "source_ref": {"bucket": BUCKET, "key": f"hist/{m.cid}.jsonl.gz",
                                            "version_id": "hv-" + m.cid[:8],
                                            "sha256": sha(f"history:{m.spec.name}:{self.poison}"), "bytes": 4096},
                             "fingerprint": fp, "generation_zero_source_sha256": sha_bytes(m.boot_results_bytes),
                             "view": {"context_sha256": m.cid, "tabpfn35": {"rmse": m.sentinels.f()},
                                      "view_id": m.view_id}}

    # -- helpers for tests
    def context(self, name: str) -> SynthContext:
        return self.contexts[name]

    def sentinel_values(self) -> dict:
        out = {"floats": set(), "f32": set(), "ints": set(), "strings": set()}
        for sc in self.contexts.values():
            out["floats"].update(sc.sentinels.floats)
            out["f32"].update(sc.sentinels.f32)
            out["ints"].update(sc.sentinels.ints)
            out["strings"].update(sc.sentinels.strings)
        return out


def walk_keys(obj: Any, path: str = ""):
    """(path, key) of every key of a JSON document, recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k
            yield from walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_keys(v, f"{path}[{i}]")


def deep(obj: Any) -> Any:
    return copy.deepcopy(obj)
