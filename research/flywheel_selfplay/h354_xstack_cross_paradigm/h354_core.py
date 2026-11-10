"""H354 core: cross-paradigm SEARCH-only super-learner ("XSTACK") over the locked H335 GP stack members.

Everything here is TRAIN-only. Importing this module imports the frozen H317 `common_h317`, which installs the H317
guard (an audit hook refusing every DL-Val / dev91 path) before any data file is opened.

Per context the stacker sees ONLY SEARCH out-of-fold predictions on H335's native SEARCH two-fold split:
  p0 = predictions for SEARCH fold-0 rows of a model fit on SEARCH fold-1 rows; p1 the mirror.
Candidates = the K members of the stored H335 `AGG.g50.CV3@12` stack (unchanged, in stored order) + a fixed library
of non-GP learners whose OOF predictions are produced on exactly the same folds. Weights = the frozen H317 CV3
S1-direct Frank-Wolfe stacker (`stacking_h317.weights_s1_direct`), unchanged. AUDIT rows enter once, after the
weights are fixed: AUDIT prediction = sum_i w_i * (member i fit on all SEARCH rows, predicting AUDIT rows), scored by
the frozen `stack_audit_utility` (u = -ln(RMSE_audit / RMSE of the SEARCH-mean predictor)).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
WAVE = "H354"
SLUG = "h354_xstack_cross_paradigm"

H317_SRC = Path(os.environ.get(
    "H354_H317_SRC",
    "/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum"))
# The four frozen H317 modules H354 imports (sha256 == h335_core.H317_PINS == origin/v1/next 97a1ffce).
H317_PINS: Dict[str, str] = {
    "common_h317.py": "1242bae25379dc1adb7cbaf253fb7066dcf783b579258f98302b4a3c31c1c3d4",
    "gp_engine_h317.py": "43ec25e9c67bcf86d4be83229fee9c37eb738fe521e20edf762a869bcd171b36",
    "gp_grammar_h317.py": "d26fd2385eafc0e5340eb88f7b29f7269298a09531a1240e60e4d83216856a5c",
    "stacking_h317.py": "c2f5c753ea423f125d0dea05cb3972cca3be211ed39f4637228dfc1791f567ee",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_h317_pins() -> None:
    for name, want in H317_PINS.items():
        got = sha256_file(H317_SRC / name)
        if got != want:
            raise RuntimeError(f"H317 pin mismatch: {name} {got} != {want}")


verify_h317_pins()
os.environ.setdefault("H317_DATA", "/home/ubuntu/jope-data/h335-cl")
sys.path.insert(0, str(H317_SRC))
import common_h317 as C  # noqa: E402  (installs the H317 guard at import)
import gp_engine_h317 as E  # noqa: E402
import stacking_h317 as ST  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# Locked design (PREREG §2-§3)
# ----------------------------------------------------------------------------------------------------------------------
STACK_KEY = "AGG.g50.CV3@12"  # the locked H335 output (LOCK 4d58dd37): its members are the GP candidates
SEED_T = 335101  # H335 Phase T search seed (the stored P201b pools)
C_LIB = 0.0  # complexity charged to a library member in the S1 objective (GP members keep their DSL complexity)
CLASSICAL = ("catboost.v1", "lightgbm.v1", "xgboost.v1", "random_forest.v1")  # reference-board model ids/configs
LLM = ("Opus5p5_Extra", "Gemini4_argon_high", "GPT6p1_Sol_Max", "GPT6_Astra_Max")  # H335 frontier binary entrants
LIBRARY = CLASSICAL + LLM
ARMS: Dict[str, Tuple[str, ...]] = {
    "X0": (),  # harness check: must reproduce the stored H335 AGG.g50.CV3@12 weights and AUDIT utility
    "X_CL": CLASSICAL,
    "X_LLM": LLM,
    "X_ALL": LIBRARY,
}
CANDIDATE_ARMS = ("X_CL", "X_LLM", "X_ALL")
W_EPS = ST.W_EPS

# ----------------------------------------------------------------------------------------------------------------------
# Inputs (Phase D: the stored H335 Phase T P201b artifacts; read-only)
# ----------------------------------------------------------------------------------------------------------------------
H335_DATA = Path("/home/ubuntu/jope-data/h335-cl")
H335_RUNS = (H335_DATA / "runs" / "T_xbabe0", H335_DATA / "runs" / "T_xbabe2")
FRONTIER_T = H335_DATA / "frontier" / "T"
FRONTIER_BIN = H335_DATA / "build" / "frontier_audit_h335"
FRONTIER_BIN_SHA256 = "b359807c289a7fbd8304218d8c29b31d758d552e0b1f8a2e82a748686cc22d43"
REFBOARD_P201B = H335_DATA / "refboard" / "p201b-train-audit-v1"
REFBOARD_TOOLS = Path(os.environ.get(
    "H354_REFBOARD_TOOLS", "/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board/tools"))
REFBOARD_PINS = {  # == origin/v1/next 8dbb774428 research/elo/reference-board (harness_sha256 of every refboard record)
    "tools/score_reference.py": "169cb04e4bc1bf122bdcdcbae16eab3ac57b6770f9afa38a7f9d3970a3190efe",
    "models.json": "20d100c406a44421ea7695053e67ac965d565214d078177fd2f3ef6f1658b193",
}
H335_STATS = Path(os.environ.get("H354_H335_STATS", "/home/ubuntu/h335-stage/code/h335_stats.py"))
H335_STATS_SHA256 = "ee3b45439a08bc395d53079570ed8965b61aad8c2d2e4e89fd1b32c490bbccb1"
DATA = Path(os.environ.get("H354_DATA", "/home/ubuntu/jope-data/h354-xs"))


@dataclass
class Envelope:
    folds: np.ndarray  # uint8 [nt]
    X: np.ndarray  # [nt, width]
    y: np.ndarray  # [nt]
    sha256: str
    nq: int


def read_envelope(path: Path) -> Envelope:
    """JTT1 reader for both forms: train-only (nq = 0) and the H317 worker form (nq = 10 all-zero query records)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"JTT1":
        raise ValueError(f"bad JTT1 magic: {path}")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if len(raw) != 17 + nt + 8 * (nt + nq) * (width + 1):
        raise ValueError(f"{path}: unexpected size")
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17).copy()
    mat = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1).copy()
    if nq and np.any(np.frombuffer(raw, dtype="<f8", offset=17 + nt + 8 * nt * (width + 1))):
        raise ValueError(f"{path}: non-zero query records")
    if not np.isin(folds, (0, 1)).all():
        raise ValueError(f"{path}: fold byte outside {{0, 1}}")
    return Envelope(folds, mat[:, :width], mat[:, width], hashlib.sha256(raw).hexdigest(), int(nq))


def train_only_bytes(env: Envelope, folds: np.ndarray, rows: Optional[np.ndarray] = None) -> bytes:
    """A train-only (nq = 0, flag 2) JTT1 with the given fold bytes (for the frontier binary: fit fold 0, predict 1)."""
    rows = np.arange(len(env.y)) if rows is None else rows
    mat = np.column_stack([env.X[rows], env.y[rows]])
    return (b"JTT1" + struct.pack("<III", len(rows), 0, env.X.shape[1]) + bytes([2])
            + np.asarray(folds, np.uint8).tobytes() + np.ascontiguousarray(mat, dtype="<f8").tobytes())


# ----------------------------------------------------------------------------------------------------------------------
# One context
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Context:
    cid: str
    run: Path
    family: str
    search: Envelope
    audit: Envelope
    fd: "ST.FoldData"
    members: List["E.SearchResult"]  # stored stack order
    audit_by_sha: Dict[str, "E.AuditResult"]
    stored: dict  # outputs[STACK_KEY]
    ya: np.ndarray
    ba: float
    y_search_mean: float
    checks: Dict[str, float] = field(default_factory=dict)


def run_of(cid: str, runs: Sequence[Path] = H335_RUNS) -> Path:
    for r in runs:
        if cid in json.loads((r / "config.json").read_text())["cids"]:
            return r
    raise KeyError(cid)


def p201b_cids() -> List[str]:
    out: List[str] = []
    for r in H335_RUNS:
        out += json.loads((r / "config.json").read_text())["cids"]
    if len(out) != 201 or len(set(out)) != 201:
        raise AssertionError("H335 Phase T run dirs do not hold the 201 P201b contexts")
    return sorted(out)


def family_map(board_root: Path = REFBOARD_P201B) -> Dict[str, str]:
    return {c["cid"]: c["family_id"] for c in json.loads((board_root / "BOARD.json").read_text())["contexts"]}


def _arr(x) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float64)


def fold_data(search: Envelope) -> "ST.FoldData":
    y0, y1 = search.y[search.folds == 0], search.y[search.folds == 1]
    b0 = float(np.sqrt(np.mean((y0 - y1.mean()) ** 2)))
    b1 = float(np.sqrt(np.mean((y1 - y0.mean()) ** 2)))
    return ST.FoldData(y0, y1, b0, b1)


def load_context(cid: str, fam_of: Dict[str, str], runs: Sequence[Path] = H335_RUNS, stage: str = "T_AGG",
                 seed: int = SEED_T) -> Context:
    """One context of a stored H335-recipe AGG run: Phase D = the H335 Phase T runs (defaults); Phase C = the P201c run."""
    run = run_of(cid, runs)
    rec = json.loads((run / "stages" / stage / f"{cid}__{seed}.jsonl").read_text().splitlines()[0])
    if rec["cid"] != cid or rec["seed"] != seed:
        raise AssertionError(f"{cid}: stage record identity")
    stored = rec["outputs"][STACK_KEY]
    spec = stored["spec"]
    want = list(spec["members"])
    search_recs: Dict[str, dict] = {}
    audit_recs: Dict[str, dict] = {}
    with open(run / "gproot_AGG" / "memo" / f"{cid}.split.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["cid"] != cid or r["sha"] not in want:
                continue
            (search_recs if r["kind"] == "search" else audit_recs)[r["sha"]] = r
    missing = [s for s in want if s not in search_recs]
    if missing:
        raise AssertionError(f"{cid}: stack members without a SEARCH memo record: {missing}")
    env_s = {search_recs[s]["env_sha256"] for s in want}
    env_a = {r["env_sha256"] for r in audit_recs.values()}
    if len(env_s) != 1 or len(env_a) != 1:
        raise AssertionError(f"{cid}: members disagree on the SEARCH/AUDIT envelope")
    envdir = run / "gproot_AGG" / "envs"
    search = read_envelope(envdir / f"e_{env_s.pop()}.w.jtt1")
    audit = read_envelope(envdir / f"e_{env_a.pop()}.w.jtt1")
    members = []
    for s in want:
        r = search_recs[s]
        members.append(E.SearchResult(cid, s, r["dsl"], r["status"], float(r["utility"]), float(r["fitness"]),
                                      bool(r["gate"]), float(r["lr0"]), float(r["lr1"]), float(r["mean_lr"]),
                                      float(r["tail_lr"]), float(r["sd_lr"]), r["complexity"], _arr(r.get("p0")),
                                      _arr(r.get("p1")), float(r.get("wall_s", 0.0)), r.get("error", "")))
    audit_by_sha = {s: E.AuditResult(cid, s, r["dsl"], r["status"], float(r["utility"]), r.get("model_rmse"),
                                     r.get("base_rmse"), _arr(r.get("pa")), float(r.get("wall_s", 0.0)),
                                     r.get("error", "")) for s, r in audit_recs.items()}
    ya = audit.y[audit.folds == 1]
    ys = audit.y[audit.folds == 0]
    ysm = float(ys.mean())
    ba = float(np.sqrt(np.mean((ya - ysm) ** 2)))
    return Context(cid, run, fam_of[cid], search, audit, fold_data(search), members, audit_by_sha, stored, ya, ba, ysm)


# ----------------------------------------------------------------------------------------------------------------------
# Alignment checks (run before any library fit; no fit, no new score)
# ----------------------------------------------------------------------------------------------------------------------
def alignment_checks(ctx: Context, frontier_env: Path, refboard_root: Path = REFBOARD_P201B) -> Dict[str, float]:
    """Max abs deviations; every one must be <= 1e-9 (relative) for the context to be usable."""
    fd = ctx.fd
    dev = {}
    # 1. member OOF order: recompute lr0/lr1 from p0/p1 with the fold baselines
    dev["lr_member"] = max(max(abs(ST._lr(m.p0, fd.y0, fd.b0) - m.lr0), abs(ST._lr(m.p1, fd.y1, fd.b1) - m.lr1))
                           for m in ctx.members)
    # 2. SEARCH rows of the AUDIT envelope == SEARCH envelope rows (same order)
    s_rows = ctx.audit.folds == 0
    dev["search_rows"] = float(np.max(np.abs(ctx.audit.y[s_rows] - ctx.search.y))) if s_rows.sum() == len(
        ctx.search.y) else math.inf
    # 3. AUDIT baseline and member AUDIT RMSEs
    bases = {float(a.base_rmse) for a in ctx.audit_by_sha.values() if a.base_rmse is not None}
    dev["audit_base"] = max(abs(b - ctx.ba) / max(ctx.ba, 1e-12) for b in bases) if bases else math.inf
    devs = [abs(float(np.sqrt(np.mean((a.pa - ctx.ya) ** 2))) - a.model_rmse) / max(a.model_rmse, 1e-12)
            for a in ctx.audit_by_sha.values() if a.status == "completed" and a.pa is not None]
    dev["audit_member_rmse"] = max(devs) if devs else 0.0
    # 4. the frontier / reference-board AUDIT envelope has exactly the GP AUDIT envelope's rows and folds
    fe = read_envelope(frontier_env)
    same = fe.X.shape == ctx.audit.X.shape and np.array_equal(fe.folds, ctx.audit.folds)
    dev["frontier_env"] = (float(max(np.nanmax(np.abs(np.nan_to_num(fe.X) - np.nan_to_num(ctx.audit.X))),
                                     np.max(np.abs(fe.y - ctx.audit.y)))) if same else math.inf)
    ctx.checks.update(dev)
    return dev


# ----------------------------------------------------------------------------------------------------------------------
# Library members as SearchResult / AuditResult (same fields the frozen stacker reads)
# ----------------------------------------------------------------------------------------------------------------------
def lib_search_result(ctx: Context, name: str, p0: Optional[np.ndarray], p1: Optional[np.ndarray],
                      status: str = "completed", error: str = "") -> "E.SearchResult":
    fd = ctx.fd
    ok = status == "completed" and p0 is not None and p1 is not None and np.all(np.isfinite(p0)) and np.all(
        np.isfinite(p1)) and len(p0) == len(fd.y0) and len(p1) == len(fd.y1)
    if not ok:
        return E.SearchResult(ctx.cid, f"lib:{name}", f"lib:{name}", "failed", C.GATE_FAIL_UTILITY, 100.0, False,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0, None, None, 0.0, error or "invalid OOF predictions")
    s = ST.s1_from_lr(ST._lr(p0, fd.y0, fd.b0), ST._lr(p1, fd.y1, fd.b1), C_LIB)
    return E.SearchResult(ctx.cid, f"lib:{name}", f"lib:{name}", "completed", s["utility"], s["fitness"], s["gate"],
                          s["lr0"], s["lr1"], s["mean_lr"], s["tail_lr"], s["sd_lr"], C_LIB, p0, p1, 0.0, "")


def lib_audit_result(ctx: Context, name: str, pa: Optional[np.ndarray], status: str) -> "E.AuditResult":
    if status != "completed" or pa is None or len(pa) != len(ctx.ya) or not np.all(np.isfinite(pa)):
        return E.AuditResult(ctx.cid, f"lib:{name}", f"lib:{name}", "failed", C.GATE_FAIL_UTILITY, None, ctx.ba, None)
    rmse = float(np.sqrt(np.mean((pa - ctx.ya) ** 2)))
    return E.AuditResult(ctx.cid, f"lib:{name}", f"lib:{name}", "completed", C.audit_utility_from_rmse(rmse, ctx.ba),
                         rmse, ctx.ba, pa)


# ----------------------------------------------------------------------------------------------------------------------
# The stacker (SEARCH only) and the one AUDIT score per arm
# ----------------------------------------------------------------------------------------------------------------------
def stack_arm(ctx: Context, lib_search: Dict[str, "E.SearchResult"], arm: str) -> "ST.StackSpec":
    """Weights from SEARCH OOF predictions only. Library members join only if they pass the S1 gate on SEARCH."""
    cand = list(ctx.members) + [lib_search[n] for n in ARMS[arm] if lib_search[n].gate and lib_search[n].ok]
    P0, P1, _, comps = ST._mats(cand)
    if len(cand) == 1:
        one = np.ones(1)
        w0 = w1 = wa = one
    else:
        w0, w1, wa = ST.weights_s1_direct(cand, ctx.fd)
    return ST.StackSpec(f"{arm}", [m.sha for m in cand], [m.dsl for m in cand], [-1] * len(cand),
                        np.asarray(wa, np.float64), np.asarray(w0, np.float64), np.asarray(w1, np.float64),
                        ST.crossfit_search_s1(P0, P1, ctx.fd, w0, w1, comps), info={"n_library": len(cand) - len(
                            ctx.members)})


def audit_arm(ctx: Context, spec: "ST.StackSpec", audit_by_sha: Dict[str, "E.AuditResult"]) -> dict:
    out = ST.stack_audit_utility(spec, audit_by_sha, ctx.ya, ctx.ba, ctx.y_search_mean)
    lib_mass = float(sum(w for s, w in zip(spec.members, spec.w_audit) if s.startswith("lib:")))
    out.update({"library_weight": lib_mass, "search": spec.search,
                "weights": {s: float(w) for s, w in zip(spec.members, spec.w_audit) if w > W_EPS}})
    return out


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist")
                              else str(o)) + "\n")
    os.replace(tmp, path)
