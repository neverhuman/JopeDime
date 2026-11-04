"""H317 GP track: live Rust S1 evaluation engine.

* Worker: `bootstrap.s1main --fit-s1 <env.jtt1> <candidate.json> <out.json>` with env
  {PATH=/usr/bin:/bin, RAYON_NUM_THREADS=1, OMP_NUM_THREADS=1, HOME=TMPDIR=<slot dir>}, `taskset -c <core> nice -n 19`.
* Slots: one per core in the given core list; a slot is one thread that owns one core at a time.
* Memo cache keyed (cid, program_sha256, kind, sha256(envelope worker bytes)); kind in {"search", "audit", "noise*"};
  persisted append-only as memo/<cid>.<mode>.jsonl (mode "split" = SEARCH/AUDIT, "full" = Stage V full envelope);
  every record stores its envelope sha and a record whose envelope sha differs from the current one is refused.
* SEARCH failure (non-zero exit, timeout, unparsable output, row-order mismatch) => utility -2.0 (search convention).
* AUDIT: status "completed" whenever the worker exited 0 with finite predictions in the right row order, with the raw,
  continuous AUDIT utility (no -2.0 cliff, no reclassification by value). A crashed / non-finite AUDIT output is
  scored as the SEARCH-target-mean predictor by the callers (utility 0.0 for a single output).
* SEARCH result keeps both fold prediction vectors; AUDIT result keeps the AUDIT-row prediction vector.
* Resource guard: no new eval starts while MemAvailable < 20 GiB or host CPU busy > 85 % (/proc/meminfo, /proc/stat).
* AUDIT isolation: `selection_scope()` marks code that chooses programs / weights; `EvalEngine.audit` raises
  `AuditLeakError` when called inside it, and an audit hook refuses any open of a res-01 pool file inside it (pool
  fitness/gate/logratio are computed on 100% of rows incl. AUDIT rows). Selection code only ever receives
  `SearchResult` objects, which carry no AUDIT field.
* All paths derive from common_h317 (H317_RES01 / H317_DATA / H317_MICRO), so the run works inside the TRAIN-only
  bubblewrap view.

Scratch lives under /home/ubuntu/jope-data/h317-cl/gp/ (never /tmp). Nothing is deleted: slot files are truncated and
overwritten in place.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

import numpy as np  # noqa: E402

from gp_grammar_h317 import Candidate  # noqa: E402

GP_ROOT = C.DATA_ROOT / "gp"
DEFAULT_TIMEOUT_S = 60.0


class AuditLeakError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# AUDIT isolation scope
# ----------------------------------------------------------------------------------------------------------------------
_SCOPE = threading.local()


@contextlib.contextmanager
def selection_scope(name: str = "selection"):
    """Everything inside picks programs or weights from SEARCH data only; AUDIT evaluation is refused here."""
    stack = getattr(_SCOPE, "stack", None)
    if stack is None:
        stack = _SCOPE.stack = []
    stack.append(name)
    try:
        yield
    finally:
        stack.pop()


def in_selection_scope() -> bool:
    return bool(getattr(_SCOPE, "stack", None))


POOL_GUARD = {"refused_pool_open_in_selection": 0, "installed": False}
_POOL_PREFIX = str(Path(os.path.abspath(C.RES01 / "pools"))) + os.sep


def _pool_hook(event: str, args) -> None:
    if event != "open" or not args or not in_selection_scope():
        return
    p = args[0]
    if isinstance(p, int):
        return
    try:
        raw = os.path.abspath(os.fsdecode(p))
    except TypeError:
        return
    if (
        raw.startswith(_POOL_PREFIX)
        or os.path.basename(os.path.dirname(raw)) == "pools"
    ):
        POOL_GUARD["refused_pool_open_in_selection"] += 1
        raise C.GuardViolation(
            f"H317 GP: pool file opened inside selection scope {_SCOPE.stack!r}: {raw}"
        )


def install_pool_guard() -> None:
    if not POOL_GUARD["installed"]:
        sys.addaudithook(_pool_hook)
        POOL_GUARD["installed"] = True


install_pool_guard()


# ----------------------------------------------------------------------------------------------------------------------
# Resource guard
# ----------------------------------------------------------------------------------------------------------------------
def read_mem_available_gib(path: str = "/proc/meminfo") -> float:
    with open(path) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return float("inf")


def read_cpu_times(path: str = "/proc/stat") -> Tuple[int, int]:
    with open(path) as f:
        parts = f.readline().split()
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals[:8]), idle


class ResourceGuard:
    def __init__(
        self,
        min_mem_gib: float = 20.0,
        max_busy: float = 0.85,
        poll_s: float = 5.0,
        log_path: Optional[Path] = None,
        mem_fn=read_mem_available_gib,
        cpu_fn=read_cpu_times,
    ):
        self.min_mem_gib, self.max_busy, self.poll_s = min_mem_gib, max_busy, poll_s
        self.log_path = Path(log_path) if log_path else None
        self.mem_fn, self.cpu_fn = mem_fn, cpu_fn
        self._lock = threading.Lock()
        self._last: Optional[Tuple[float, int, int]] = None
        self._busy = 0.0
        self._t_busy = 0.0
        self.pauses: List[dict] = []

    def cpu_busy(self) -> float:
        with self._lock:
            now = time.monotonic()
            if now - self._t_busy < 1.0 and self._last is not None:
                return self._busy
            if self._last is None:
                t0, i0 = self.cpu_fn()
                time.sleep(0.5)
                self._last = (time.monotonic(), t0, i0)
            tot, idle = self.cpu_fn()
            _, t0, i0 = self._last
            dt, di = tot - t0, idle - i0
            if dt > 0:
                self._busy = max(0.0, min(1.0, 1.0 - di / dt))
            self._last = (time.monotonic(), tot, idle)
            self._t_busy = time.monotonic()
            return self._busy

    def status(self) -> Tuple[bool, float, float]:
        mem = self.mem_fn()
        busy = self.cpu_busy()
        return (mem >= self.min_mem_gib and busy <= self.max_busy), mem, busy

    def wait(self) -> float:
        ok, mem, busy = self.status()
        if ok:
            return 0.0
        t0 = time.time()
        rec = {
            "pause_start_utc": t0,
            "mem_gib": round(mem, 2),
            "cpu_busy": round(busy, 3),
        }
        self._log({"event": "pause", **rec})
        while not ok:
            time.sleep(self.poll_s)
            ok, mem, busy = self.status()
        dur = time.time() - t0
        rec.update(
            {
                "pause_s": round(dur, 2),
                "resume_mem_gib": round(mem, 2),
                "resume_cpu_busy": round(busy, 3),
            }
        )
        with self._lock:
            self.pauses.append(rec)
        self._log({"event": "resume", **rec})
        return dur

    def _log(self, obj: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.log_path, "a") as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------------------------------------------------
# Context envelopes
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ContextEnvs:
    cid: str
    family_id: str
    width: int
    search: C.Jtt1
    audit: C.Jtt1
    audit_mask: np.ndarray
    search_path: Path
    audit_path: Path
    y0: np.ndarray  # SEARCH fold-0 targets, row order
    y1: np.ndarray  # SEARCH fold-1 targets, row order
    ya: np.ndarray  # AUDIT targets, AUDIT-envelope row order (only for final scoring)
    b0: float  # rmse(y0 - mean(y1)) == worker baseline for inner_cv[0]
    b1: float  # rmse(y1 - mean(y0))
    ba: float  # rmse(ya - mean(SEARCH y))
    full_mode: bool = False  # Stage V: SEARCH = all rows, no AUDIT
    alts: Dict[str, "AltEnv"] = field(
        default_factory=dict
    )  # NOISE_PROBE re-split SEARCH envelopes
    env_sha: Dict[str, str] = field(
        default_factory=dict
    )  # kind -> sha256(worker bytes)

    @property
    def n_search(self) -> int:
        return int(self.search.nt)

    @property
    def mode(self) -> str:
        return "full" if self.full_mode else "split"

    @property
    def y_search_mean(self) -> float:
        return float(np.concatenate([self.y0, self.y1]).mean())

    def kind_sha(self, kind: str) -> str:
        if kind in self.env_sha:
            return self.env_sha[kind]
        if kind in self.alts:
            return self.alts[kind].env_sha
        raise KeyError(kind)


@dataclass
class AltEnv:
    """SEARCH rows with a different y-stratified random 2-fold split (NOISE_PROBE diagnostic only)."""

    name: str
    path: Path
    y0: np.ndarray
    y1: np.ndarray
    env_sha: str


def build_alt(ctx: ContextEnvs, name: str, env_dir: Path) -> AltEnv:
    senv = ctx.search
    y = senv.y
    rng = np.random.default_rng(C.stream_seed("H317-noise-split", name, ctx.cid))
    order = np.argsort(y, kind="stable")
    folds = np.zeros(len(y), dtype=np.uint8)
    for s in range(0, len(y), 2):
        blk = order[s : s + 2]
        if len(blk) == 2:
            first = int(rng.integers(0, 2))
            folds[blk[0]], folds[blk[1]] = first, 1 - first
        else:
            folds[blk[0]] = int(rng.integers(0, 2))
    alt = C.Jtt1(senv.magic, senv.nt, 0, senv.width, senv.flag, folds, senv.mat.copy())
    b = alt.to_worker_bytes()
    p = _write_env(env_dir, b)
    return AltEnv(
        name,
        p,
        y[folds == 0].copy(),
        y[folds == 1].copy(),
        hashlib.sha256(b).hexdigest(),
    )


def _write_env(env_dir: Path, b: bytes) -> Path:
    """Content-addressed worker envelope e_<sha256>.w.jtt1: the worker argv never carries a context id (guard v2
    exec-argv scan; Stage V DL-Val paths must not appear in argv)."""
    env_dir.mkdir(parents=True, exist_ok=True)
    p = env_dir / f"e_{hashlib.sha256(b).hexdigest()}.w.jtt1"
    C.assert_allowed(p)
    if not p.exists() or p.stat().st_size != len(b) or p.read_bytes() != b:
        tmp = p.with_name(p.name + f".tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_bytes(b)
        os.replace(tmp, p)
    return p


def build_context(
    cid: str, family_id: str, env_dir: Path, full_mode: bool = False
) -> ContextEnvs:
    env = C.read_jtt1(C.table_path(cid))
    if full_mode:
        search, audit, mask = env, env, np.zeros(env.nt, dtype=bool)
    else:
        search, audit, mask = C.search_and_audit_envelopes(cid, env)
    sb = search.to_worker_bytes()
    sp = _write_env(env_dir, sb)
    ap = sp
    shas = {"search": hashlib.sha256(sb).hexdigest()}
    if not full_mode:
        ab = audit.to_worker_bytes()
        ap = _write_env(env_dir, ab)
        shas["audit"] = hashlib.sha256(ab).hexdigest()
    y, f = search.y, search.folds
    y0, y1 = y[f == 0].copy(), y[f == 1].copy()
    b0 = float(np.sqrt(np.mean((y0 - y1.mean()) ** 2)))
    b1 = float(np.sqrt(np.mean((y1 - y0.mean()) ** 2)))
    if full_mode:
        ya, ba = np.zeros(0), float("nan")
    else:
        ya = audit.y[audit.folds == 1].copy()
        ba = float(np.sqrt(np.mean((ya - y.mean()) ** 2)))
    return ContextEnvs(
        cid,
        family_id,
        int(env.width),
        search,
        audit,
        mask,
        sp,
        ap,
        y0,
        y1,
        ya,
        b0,
        b1,
        ba,
        full_mode,
        env_sha=shas,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchResult:
    cid: str
    sha: str
    dsl: str
    status: str
    utility: float
    fitness: float
    gate: bool
    lr0: float
    lr1: float
    mean_lr: float
    tail_lr: float
    sd_lr: float
    complexity: int
    p0: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    p1: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    wall_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.status == "completed" and self.p0 is not None and self.p1 is not None
        )


@dataclass(frozen=True)
class AuditResult:
    cid: str
    sha: str
    dsl: str
    status: str
    utility: float
    model_rmse: Optional[float]
    base_rmse: Optional[float]
    pa: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    wall_s: float = 0.0
    error: str = ""


def _arr(x) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float64)


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) != len(b):
        return False
    scale = max(1.0, float(np.max(np.abs(b))) if len(b) else 1.0)
    return bool(np.allclose(a, b, rtol=1e-10, atol=1e-9 * scale))


def parse_search(
    ctx: ContextEnvs,
    cand: Candidate,
    doc: Optional[dict],
    wall: float,
    err: str,
    y0: Optional[np.ndarray] = None,
    y1: Optional[np.ndarray] = None,
) -> SearchResult:
    y0 = ctx.y0 if y0 is None else y0
    y1 = ctx.y1 if y1 is None else y1
    fail = dict(
        cid=ctx.cid,
        sha=cand.sha,
        dsl=cand.dsl,
        status="failed",
        utility=C.GATE_FAIL_UTILITY,
        fitness=100.0,
        gate=False,
        lr0=float("nan"),
        lr1=float("nan"),
        mean_lr=float("nan"),
        tail_lr=float("nan"),
        sd_lr=float("nan"),
        complexity=cand.dsl.count("("),
        wall_s=wall,
    )
    if doc is None:
        return SearchResult(**fail, error=err or "no_output")
    inner = doc.get("inner_cv") or []
    prog = (doc.get("program") or {}).get("program_sha256")
    if prog is not None and prog != cand.sha:
        return SearchResult(**fail, error="program_sha_mismatch")
    s1 = C.s1_from_inner_cv(cand.dsl, inner)
    if s1["status"] != "completed":
        return SearchResult(**fail, error="s1_failed")
    p0, p1 = _arr(inner[0].get("predictions")), _arr(inner[1].get("predictions"))
    t0, t1 = _arr(inner[0].get("targets")), _arr(inner[1].get("targets"))
    if (
        p0 is None
        or p1 is None
        or t0 is None
        or t1 is None
        or not _close(t0, y0)
        or not _close(t1, y1)
    ):
        return SearchResult(**fail, error="row_order_mismatch")
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        return SearchResult(**fail, error="nonfinite_predictions")
    return SearchResult(
        cid=ctx.cid,
        sha=cand.sha,
        dsl=cand.dsl,
        status="completed",
        utility=float(s1["utility"]),
        fitness=float(s1["fitness"]),
        gate=bool(s1["gate"]),
        lr0=s1["lr0"],
        lr1=s1["lr1"],
        mean_lr=s1["mean_lr"],
        tail_lr=s1["tail_lr"],
        sd_lr=s1["sd_lr"],
        complexity=int(s1["complexity"]),
        p0=p0,
        p1=p1,
        wall_s=wall,
    )


def parse_audit(
    ctx: ContextEnvs, cand: Candidate, doc: Optional[dict], wall: float, err: str
) -> AuditResult:
    fail = dict(
        cid=ctx.cid,
        sha=cand.sha,
        dsl=cand.dsl,
        status="failed",
        utility=C.GATE_FAIL_UTILITY,
        model_rmse=None,
        base_rmse=None,
        wall_s=wall,
    )
    if doc is None:
        return AuditResult(**fail, error=err or "no_output")
    inner = doc.get("inner_cv") or []
    if len(inner) != 2:
        return AuditResult(**fail, error="inner_cv_len")
    rec = inner[1]
    m = (rec.get("metrics") or {}).get("rmse")
    b = (rec.get("baseline_metrics") or {}).get("rmse")
    pa, ta = _arr(rec.get("predictions")), _arr(rec.get("targets"))
    if pa is None or ta is None or not _close(ta, ctx.ya):
        return AuditResult(**fail, error="row_order_mismatch")
    if not np.all(np.isfinite(pa)):
        return AuditResult(**fail, error="nonfinite_predictions")
    if (
        m is None
        or b is None
        or not (math.isfinite(float(m)) and math.isfinite(float(b)))
    ):
        return AuditResult(**fail, error="nonfinite_metrics")
    u = C.audit_utility_from_rmse(m, b)  # continuous; no reclassification by value
    return AuditResult(
        cid=ctx.cid,
        sha=cand.sha,
        dsl=cand.dsl,
        status="completed",
        utility=u,
        model_rmse=float(m),
        base_rmse=float(b),
        pa=pa,
        wall_s=wall,
    )


def _result_to_json(kind: str, r, env_sha: str) -> dict:
    d = {k: getattr(r, k) for k in r.__dataclass_fields__}
    d["env_sha256"] = env_sha
    for k in ("p0", "p1", "pa"):
        if k in d and d[k] is not None:
            d[k] = [float(x) for x in d[k]]
    for k, v in list(d.items()):
        if isinstance(v, float) and not math.isfinite(v):
            d[k] = None
    d["kind"] = kind
    return d


def _result_from_json(d: dict):
    kind = d.pop("kind")
    env_sha = d.pop("env_sha256", None)
    cls = AuditResult if kind == "audit" else SearchResult
    vals = {}
    for k in cls.__dataclass_fields__:
        v = d.get(k)
        if k in ("p0", "p1", "pa"):
            v = _arr(v)
        elif k in ("lr0", "lr1", "mean_lr", "tail_lr", "sd_lr") and v is None:
            v = float("nan")
        vals[k] = v
    return kind, env_sha, cls(**vals)


# ----------------------------------------------------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------------------------------------------------
Runner = Callable[
    [ContextEnvs, str, Candidate, str, Path], Tuple[Optional[dict], float, str]
]


def eval_timeout(base_s: float, kind: str, dsl: str) -> float:
    """Per-eval timeout: x4 for explicit portfolios (5-fold internal member fits), x2 on the AUDIT envelope (models
    fit on all SEARCH rows = twice a SEARCH fold)."""
    t = base_s * (4.0 if dsl.startswith("(portfolio") else 1.0)
    return t * (2.0 if kind == "audit" else 1.0)


def rust_runner(worker_bin: Path, timeout_s: float, nice: int = 19) -> Runner:
    def run(ctx: ContextEnvs, kind: str, cand: Candidate, core: str, slot: Path):
        env_path = (
            ctx.search_path
            if kind == "search"
            else ctx.audit_path
            if kind == "audit"
            else ctx.alts[kind].path
        )
        C.assert_allowed(env_path)
        cand_path, out_path = slot / "candidate.json", slot / "out.json"
        cand_path.write_text(
            json.dumps(
                {
                    "dsl": cand.dsl,
                    "program_sha256": cand.sha,
                    "birth_seq": 0,
                    "parents": list(cand.parents),
                    "operator": cand.operator,
                },
                sort_keys=True,
            )
        )
        open(out_path, "w").close()  # truncate in place (no deletion)
        cmd = [
            "taskset",
            "-c",
            core,
            "nice",
            "-n",
            str(nice),
            str(worker_bin),
            "--fit-s1",
            str(env_path),
            str(cand_path),
            str(out_path),
        ]
        env = {
            "PATH": "/usr/bin:/bin",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "HOME": str(slot),
            "TMPDIR": str(slot),
        }
        t0 = time.monotonic()
        try:
            p = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=eval_timeout(timeout_s, kind, cand.dsl),
            )
            rc, err = p.returncode, p.stderr.decode(errors="replace")[-300:].strip()
        except subprocess.TimeoutExpired:
            return None, time.monotonic() - t0, "timeout"
        wall = time.monotonic() - t0
        if rc != 0:
            return None, wall, f"rc={rc}:{err}"
        try:
            doc = json.loads(out_path.read_text())
        except Exception as e:  # noqa: BLE001
            return None, wall, f"unparsable:{type(e).__name__}"
        if doc.get("status") != "completed":
            return None, wall, f"status={doc.get('status')}"
        return doc, wall, ""

    return run


class EvalEngine:
    def __init__(
        self,
        cores: Sequence[int],
        root: Path = GP_ROOT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        guard: Optional[ResourceGuard] = None,
        runner: Optional[Runner] = None,
        persist: bool = True,
        worker_bin: Path = C.WORKER_BIN,
    ):
        if not cores:
            raise ValueError("need at least one core")
        self.cores = [str(int(c)) for c in cores]
        self.root = Path(root)
        self.env_dir = self.root / "envs"
        self.memo_dir = self.root / "memo"
        self.slot_dir = self.root / "slots"
        self.timeout_s = timeout_s
        self.guard = guard
        self.runner = runner or rust_runner(worker_bin, timeout_s)
        self.persist = persist
        self._free: "queue.Queue[str]" = queue.Queue()
        for c in self.cores:
            self._free.put(c)
        self._pool = cf.ThreadPoolExecutor(
            max_workers=len(self.cores), thread_name_prefix="h317gp-eval"
        )
        self._memo: Dict[
            Tuple[str, str, str, str], object
        ] = {}  # (cid, program sha, kind, envelope sha)
        self._inflight: Dict[Tuple[str, str, str, str], cf.Future] = {}
        self._lock = (
            threading.RLock()
        )  # re-entrant: add_done_callback may run inline under the lock
        self._file_locks: Dict[str, threading.Lock] = {}
        self._loaded: set = set()  # (cid, mode)
        self._ctx: Dict[Tuple[str, bool], ContextEnvs] = {}
        self._ctx_locks: Dict[
            Tuple[str, bool], threading.Lock
        ] = {}  # single-flight context + memo construction
        self.retry_log: List[dict] = []
        self.counters = {
            "requests_search": 0,
            "requests_audit": 0,
            "memo_hits": 0,
            "physical_runs": 0,
            "failures": 0,
            "timeouts": 0,
            "wall_s_total": 0.0,
            "memo_env_mismatch_refused": 0,
        }
        self.walls: List[
            Tuple[str, str, float, str]
        ] = []  # (kind, operator, wall, error)

    # -- contexts ------------------------------------------------------------------------------------------------------
    def context(self, cid: str, family_id: str, full_mode: bool = False) -> ContextEnvs:
        """One ContextEnvs object per (cid, mode) per engine, built and memo-loaded exactly once even when several
        jobs (e.g. the two search seeds of one context) ask concurrently (GP red team MEDIUM-2). Shards partition by
        context, so one context's memo file has a single writing process."""
        key = (cid, full_mode)
        with self._lock:
            ctx = self._ctx.get(key)
            if ctx is not None:
                return ctx
            lk = self._ctx_locks.setdefault(key, threading.Lock())
        with lk:
            with self._lock:
                ctx = self._ctx.get(key)
            if ctx is not None:
                return ctx
            ctx = build_context(cid, family_id, self.env_dir, full_mode)
            self._load_memo(ctx)
            with self._lock:
                self._ctx[key] = ctx  # published only after its memo is loaded
            return ctx

    # -- memo persistence ----------------------------------------------------------------------------------------------
    def _memo_path(self, cid: str, mode: str) -> Path:
        return self.memo_dir / f"{cid}.{mode}.jsonl"

    def _load_memo(self, ctx: ContextEnvs) -> None:
        """Called under the per-(cid, mode) single-flight lock; marks the memo loaded only after reading it."""
        with self._lock:
            if (ctx.cid, ctx.mode) in self._loaded:
                return
        if not self.persist:
            with self._lock:
                self._loaded.add((ctx.cid, ctx.mode))
            return
        p = self._memo_path(ctx.cid, ctx.mode)
        C.assert_allowed(p)
        if not p.exists():
            with self._lock:
                self._loaded.add((ctx.cid, ctx.mode))
            return
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line after a crash
                kind, env_sha, r = _result_from_json(d)
                known = ctx.env_sha.get(kind)
                if (
                    r.cid != ctx.cid
                    or env_sha is None
                    or (known is not None and env_sha != known)
                ):
                    with self._lock:
                        self.counters["memo_env_mismatch_refused"] += 1
                    continue  # refused: produced on a different envelope
                with self._lock:
                    self._memo[(r.cid, r.sha, kind, env_sha)] = r
        with self._lock:
            self._loaded.add((ctx.cid, ctx.mode))

    def _append_memo(self, ctx: ContextEnvs, kind: str, r) -> None:
        if not self.persist:
            return
        p = self._memo_path(r.cid, ctx.mode)
        with self._lock:
            lk = self._file_locks.setdefault(str(p), threading.Lock())
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_result_to_json(kind, r, ctx.kind_sha(kind)), sort_keys=True)
        with lk, open(p, "a") as f:
            f.write(line + "\n")
            f.flush()

    # -- evaluation ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _transient(err: str) -> bool:
        return err == "timeout" or err.startswith(
            "rc=-"
        )  # timeout or killed by a signal (outcome-independent)

    def _attempt(self, ctx: ContextEnvs, kind: str, cand: Candidate):
        core = self._free.get()
        try:
            if self.guard is not None:
                self.guard.wait()
            slot = self.slot_dir / f"core{core}"
            slot.mkdir(parents=True, exist_ok=True)
            return self.runner(ctx, kind, cand, core, slot)
        finally:
            self._free.put(core)

    def _run_one(self, ctx: ContextEnvs, kind: str, cand: Candidate):
        """Timeouts and signal kills are retried once before being recorded as a failure (GP red team LOW-1);
        both attempts are logged to <root>/retries.jsonl."""
        doc, wall, err = self._attempt(ctx, kind, cand)
        if doc is None and self._transient(err):
            doc2, wall2, err2 = self._attempt(ctx, kind, cand)
            rec = {
                "cid": ctx.cid,
                "sha": cand.sha,
                "kind": kind,
                "attempt1": {"err": err, "wall_s": wall},
                "attempt2": {"err": err2, "wall_s": wall2},
                "utc": time.time(),
            }
            with self._lock:
                self.retry_log.append(rec)
                self.counters["retries"] = self.counters.get("retries", 0) + 1
            if self.persist:
                self.root.mkdir(parents=True, exist_ok=True)
                with self._lock, open(self.root / "retries.jsonl", "a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
            doc, wall, err = doc2, wall + wall2, err2
        if kind == "audit":
            r = parse_audit(ctx, cand, doc, wall, err)
        elif kind == "search":
            r = parse_search(ctx, cand, doc, wall, err)
        else:
            alt = ctx.alts[kind]
            r = parse_search(ctx, cand, doc, wall, err, alt.y0, alt.y1)
        with self._lock:
            self.counters["physical_runs"] += 1
            self.counters["wall_s_total"] += wall
            if r.status != "completed":
                self.counters["failures"] += 1
            if err == "timeout":
                self.counters["timeouts"] += 1
            self.walls.append((kind, cand.operator, wall, r.error))
            self._memo[(ctx.cid, cand.sha, kind, ctx.kind_sha(kind))] = r
        self._append_memo(ctx, kind, r)
        return r

    def _evaluate(
        self, ctx: ContextEnvs, cands: Sequence[Candidate], kind: str
    ) -> list:
        futs: List[cf.Future] = []
        env_sha = ctx.kind_sha(kind)
        for cand in cands:
            key = (ctx.cid, cand.sha, kind, env_sha)
            with self._lock:
                ck = "requests_" + kind
                self.counters[ck] = self.counters.get(ck, 0) + 1
                hit = self._memo.get(key)
                if hit is not None:
                    self.counters["memo_hits"] += 1
                    f: cf.Future = cf.Future()
                    f.set_result(hit)
                elif key in self._inflight:
                    self.counters["memo_hits"] += 1
                    f = self._inflight[key]
                else:
                    f = self._pool.submit(self._run_one, ctx, kind, cand)
                    self._inflight[key] = f
                    f.add_done_callback(lambda _f, k=key: self._drop_inflight(k))
            futs.append(f)
        return [f.result() for f in futs]

    def _drop_inflight(self, key) -> None:
        with self._lock:
            self._inflight.pop(key, None)

    def search(
        self, ctx: ContextEnvs, cands: Sequence[Candidate]
    ) -> List[SearchResult]:
        return self._evaluate(ctx, cands, "search")

    def search_alt(
        self, ctx: ContextEnvs, name: str, cands: Sequence[Candidate]
    ) -> List[SearchResult]:
        """NOISE_PROBE: SEARCH rows under an alternative y-stratified 2-fold split (diagnostic, never selects)."""
        if not name.startswith("noise"):
            raise ValueError("alternative envelopes are named noise*")
        with self._lock:
            have = name in ctx.alts
        if not have:
            alt = build_alt(ctx, name, self.env_dir)
            with self._lock:
                ctx.alts[name] = alt
        return self._evaluate(ctx, cands, name)

    def memo_get(self, ctx: ContextEnvs, sha: str, kind: str = "search"):
        with self._lock:
            return self._memo.get((ctx.cid, sha, kind, ctx.kind_sha(kind)))

    def audit(self, ctx: ContextEnvs, cands: Sequence[Candidate]) -> List[AuditResult]:
        if in_selection_scope():
            raise AuditLeakError(
                f"AUDIT evaluation requested inside selection scope {_SCOPE.stack!r}"
            )
        if ctx.full_mode:
            raise AuditLeakError("full-mode (Stage V) context has no AUDIT envelope")
        return self._evaluate(ctx, cands, "audit")

    def stats(self) -> dict:
        with self._lock:
            d = dict(self.counters)
            walls = [w for k, _, w, _ in self.walls if k == "search"]
        d["mean_wall_s_search"] = float(np.mean(walls)) if walls else None
        d["p95_wall_s_search"] = float(np.quantile(walls, 0.95)) if walls else None
        if self.guard is not None:
            d["pauses"] = len(self.guard.pauses)
            d["pause_s_total"] = float(
                sum(p.get("pause_s", 0.0) for p in self.guard.pauses)
            )
        return d

    def close(self) -> None:
        self._pool.shutdown(wait=True)


# ----------------------------------------------------------------------------------------------------------------------
# AUDIT ledger: exactly one AUDIT request per declared output (coordinator amendment 2)
# ----------------------------------------------------------------------------------------------------------------------
class AuditLedger:
    """Every pipeline output is fixed from SEARCH results first; then `score(pipeline, ctx, seed, outputs)` requests
    exactly one AUDIT evaluation per declared output (1 tree, or the N effective stack members). A second request for
    the same (pipeline, cid, seed) raises, so AUDIT values cannot be re-queried to steer a choice."""

    def __init__(self, engine: EvalEngine):
        self.engine = engine
        self.declared: Dict[Tuple[str, str, object], int] = {}
        self.requested: Dict[Tuple[str, str, object], int] = {}
        self._lock = threading.Lock()

    def score(
        self, pipeline: str, ctx: ContextEnvs, seed, outputs: Sequence[Candidate]
    ) -> List[AuditResult]:
        key = (pipeline, ctx.cid, seed)
        with self._lock:
            if key in self.declared:
                raise AuditLeakError(f"AUDIT already requested for {key}")
            self.declared[key] = len(outputs)
        res = self.engine.audit(ctx, list(outputs))
        with self._lock:
            self.requested[key] = len(res)
        if len(res) != len(outputs):
            raise AuditLeakError(
                f"AUDIT request count {len(res)} != declared {len(outputs)} for {key}"
            )
        return res

    def check(self) -> dict:
        bad = {
            str(k): (self.declared[k], self.requested.get(k))
            for k in self.declared
            if self.requested.get(k) != self.declared[k]
        }
        if bad:
            raise AuditLeakError(f"AUDIT ledger mismatch: {bad}")
        return {
            "outputs": len(self.declared),
            "audit_requests": int(sum(self.requested.values())),
        }
