"""H317 GP track: canonical DSL grammar, islands, featurization and GP operators.

Import `common_h317` first (done below) so the read guard is installed before any file is opened.

Canonical form mirrors the pinned evaluator source (`bootstrap.s1main`, feat-math rev 6bf65b12):
  * `crates/feat-math/src/dsl.rs` tokenizer / `parse_param_value` / `Display for {ParamValue, Stage, DslExpr}`;
  * `crates/feat-math/src/genome.rs` `GenomeIr::parse` + `canonical` (explicit portfolio form);
  * `deploy/lambda/training/teacher-v2/rust/lambda-worker/src/main.rs::genome` refuses any DSL whose canonical
    re-print differs ("noncanonical program") and pipelines with more than 9 stages.
Parameters print sorted by key (BTreeMap); integral floats print as `{:.1}`; other floats print Rust-`Display` style
(shortest round-trip, never an exponent).

Operator sources (copied and adapted, not imported):
  * `mutate_parameter_clause`, `parse_pipe_dsl`, EditZero kinds (Parameter, ReplaceHead, Insert, Delete, Crossover)
    and champion retrieval: `/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py` (`propose_flywheel_candidates`).
  * `micro_sem24`: JopeDimeMicro `phase12_multimodal_model.compute_semantic_struct_sig_24d` (numpy body, verbatim).
  * H257 used exactly those EditZero kinds through `propose_flywheel_candidates`; the H257 file has no separate
    "subtree crossover / point mutation / linear-head refit" functions. Subtree crossover here = stage-list one-point
    crossover (rust_evaluator) + portfolio member exchange; point mutation = Parameter edit; there is no linear-head
    refit operator in the grammar (stacking weights are fitted in `stacking_h317.py`).
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402  (installs nothing by itself; entry points call C.install_guard())

import numpy as np  # noqa: E402

Value = Union[int, float, bool, str]


class GrammarError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# Canonical DSL (dsl.rs / genome.rs at 6bf65b12)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    op: str
    params: Tuple[Tuple[str, Value], ...] = ()

    @staticmethod
    def make(op: str, **params: Value) -> "Stage":
        return Stage(op, tuple(sorted(params.items())))

    def get(self, key: str, default=None):
        for k, v in self.params:
            if k == key:
                return v
        return default

    def with_params(self, **upd: Value) -> "Stage":
        d = dict(self.params)
        for k, v in upd.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v
        return Stage(self.op, tuple(sorted(d.items())))

    def text(self) -> str:
        return (
            "("
            + self.op
            + "".join(f" {k}={fmt_value(v)}" for k, v in self.params)
            + ")"
        )


@dataclass(frozen=True)
class Pipe:
    stages: Tuple[Stage, ...]

    def text(self) -> str:
        return "(pipe X" + "".join(" " + s.text() for s in self.stages) + ")"

    @property
    def head(self) -> Optional[Stage]:
        return (
            self.stages[-1] if self.stages and self.stages[-1].op in HEAD_OPS else None
        )

    @property
    def transforms(self) -> Tuple[Stage, ...]:
        return self.stages[:-1] if self.head is not None else self.stages


@dataclass(frozen=True)
class Portfolio:
    members: Tuple[Pipe, ...]
    folds: int = 5
    repeats_auto: bool = True

    def text(self) -> str:
        out = "(portfolio" + "".join(" (member " + m.text() + ")" for m in self.members)
        return (
            out
            + f" (blend folds={self.folds} guard=clip kind=nnls repeats={'auto' if self.repeats_auto else '1'}))"
        )


Genome = Union[Pipe, Portfolio]

HEAD_OPS = frozenset(
    [
        "ridge",
        "logistic",
        "ridge_gcv",
        "elasticnet",
        "histgbm",
        "gbm",
        "ensemble",
        "ensemble_ridge_gbm",
        "histgbm2",
        "krr",
        "gpr",
        "ksmooth",
        "stack",
        "portfolio",
    ]
    + ["xtrees", "rforest", "oblivious"]  # LEARNERS-V1: superset worker only (bootstrap.s1main.superset-*)
)
# LEARNERS-V1 conditional arm switch (ported from learners/h317_grammar_patch/gp_grammar_h317.learners_v1.diff,
# sha 49d7f3cc..., onto the A2 grammar d26fd238...). Off (default) = the preregistered grammar, identical behaviour: the
# new heads are parseable but never generated, and descriptor widths are unchanged. On = new heads join island I0 and
# the ext_descriptor family one-hot. Programs containing them REQUIRE the superset worker (the pinned one refuses them).
LEARNERS_V1 = os.environ.get("H317_LEARNERS_V1") == "1"
NEW_HEADS = frozenset(["xtrees", "rforest", "oblivious"])
TARGET_OPS = frozenset(["target"])
MAX_PIPE_STAGES = (
    9  # lambda-worker: "pipeline exceeds eight transforms" when stages > 9
)
MAX_PORTFOLIO_MEMBERS = 12


def fmt_float(v: float) -> str:
    if not math.isfinite(v):
        raise GrammarError(f"non-finite parameter {v}")
    if abs(v - round(v)) < 1e-12:
        return f"{v:.1f}"
    return format(Decimal(repr(float(v))), "f")


def _needs_quotes(s: str) -> bool:
    return s == "" or any(c.isspace() or c in '()"=' for c in s)


def fmt_value(v: Value) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return fmt_float(v)
    s = str(v)
    if _needs_quotes(s):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def parse_value(raw: str) -> Value:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if raw.strip() == raw and raw.lstrip("+-").isdigit():
            return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _tokenize(src: str) -> List[str]:
    toks: List[str] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in "()":
            toks.append(ch)
            i += 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            in_q = False
            while j < n and (in_q or not (src[j].isspace() or src[j] in "()")):
                if src[j] == '"':
                    in_q = not in_q
                j += 1
            if in_q:
                raise GrammarError("unterminated quoted parameter")
            toks.append(src[i:j])
            i = j
    if not toks:
        raise GrammarError("empty DSL expression")
    return toks


class _P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, want=None):
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise GrammarError(f"expected {want!r} at {self.i}, got {tok!r}")
        self.i += 1
        return tok

    def stage(self) -> Stage:
        self.take("(")
        op = self.take()
        if op in "()":
            raise GrammarError("stage missing operator")
        params: Dict[str, Value] = {}
        while self.peek() != ")":
            atom = self.take()
            if atom in "()":
                raise GrammarError("stage parameter must be atom")
            if "=" not in atom:
                raise GrammarError(f"parameter {atom!r} must be key=value")
            k, v = atom.split("=", 1)
            params[k] = parse_value(v)
        self.take(")")
        return Stage(op, tuple(sorted(params.items())))

    def pipe(self) -> Pipe:
        self.take("(")
        self.take("pipe")
        self.take("X")
        stages = []
        while self.peek() != ")":
            stages.append(self.stage())
        self.take(")")
        if not stages:
            raise GrammarError("pipe must contain at least one stage")
        return Pipe(tuple(stages))


def parse(dsl: str) -> Genome:
    src = dsl.strip()
    p = _P(_tokenize(src))
    if src.startswith("(pipe"):
        g = p.pipe()
    elif src.startswith("(portfolio"):
        p.take("(")
        p.take("portfolio")
        members: List[Pipe] = []
        blend: Optional[Dict[str, Value]] = None
        while p.peek() != ")":
            p.take("(")
            tag = p.take()
            if tag == "member":
                if blend is not None:
                    raise GrammarError(
                        "portfolio: members must precede the blend stage"
                    )
                members.append(p.pipe())
                p.take(")")
            elif tag == "blend":
                blend = {}
                while p.peek() != ")":
                    k, v = p.take().split("=", 1)
                    blend[k] = parse_value(v)
                p.take(")")
            else:
                raise GrammarError(f"portfolio: unexpected element {tag!r}")
        p.take(")")
        if not 1 <= len(members) <= MAX_PORTFOLIO_MEMBERS:
            raise GrammarError("portfolio: member count must be 1..12")
        blend = blend or {}
        if (
            blend.get("kind", "nnls") != "nnls"
            or blend.get("folds", 5) != 5
            or blend.get("guard", "clip") != "clip"
        ):
            raise GrammarError("portfolio blend must be kind=nnls folds=5 guard=clip")
        rep = blend.get("repeats", "auto")
        if rep not in ("auto", 1):
            raise GrammarError("blend: repeats must be auto or 1")
        for m in members:
            if any(s.op == "portfolio" for s in m.stages):
                raise GrammarError("portfolio: members must not nest portfolios")
        g = Portfolio(tuple(members), 5, rep == "auto")
    else:
        raise GrammarError(f"unrecognized program form: {src[:40]}")
    if p.i != len(p.t):
        raise GrammarError("trailing tokens after DSL expression")
    validate(g)
    return g


def validate(g: Genome) -> None:
    pipes = g.members if isinstance(g, Portfolio) else (g,)
    for pp in pipes:
        if len(pp.stages) > MAX_PIPE_STAGES:
            raise GrammarError("pipeline exceeds eight transforms")
        for k, s in enumerate(pp.stages):
            if s.op in HEAD_OPS and k != len(pp.stages) - 1:
                raise GrammarError(
                    f"{s.op}: model heads are only valid as the final stage"
                )
            if s.op in TARGET_OPS and k != 0:
                raise GrammarError("target: transform must be the first stage")
            if s.op == "portfolio" and any(
                isinstance(v, str) and v.startswith("(") for _, v in s.params
            ):
                raise GrammarError("compact portfolio takes atoms only")


def canonical(g_or_dsl: Union[Genome, str]) -> str:
    g = parse(g_or_dsl) if isinstance(g_or_dsl, str) else g_or_dsl
    return g.text()


def sha256_text(dsl: str) -> str:
    return hashlib.sha256(dsl.encode()).hexdigest()


def complexity(dsl: str) -> int:
    return dsl.count("(")


# ----------------------------------------------------------------------------------------------------------------------
# Islands (plan §3.3 mapped onto the heads that actually exist in the pinned grammar)
# ----------------------------------------------------------------------------------------------------------------------
HEAD_FAMILY = {
    "histgbm": "histgbm",
    "gbm": "histgbm",
    "histgbm2": "histgbm2",
    "ksmooth": "ksmooth",
    "krr": "krr",
    "gpr": "gpr",
    "ridge": "ridge",
    "logistic": "ridge",
    "ridge_gcv": "ridge_gcv",
    "elasticnet": "elasticnet",
    "ensemble": "ensemble",
    "ensemble_ridge_gbm": "ensemble",
    "stack": "stack",
    "portfolio": "portfolio",
    "xtrees": "xtrees",  # LEARNERS-V1
    "rforest": "rforest",
    "oblivious": "oblivious",
}
# Islands (coordinator amendment A, replacing the plan's learner lists, which do not exist in this grammar):
#   I0 tree = histgbm / histgbm2;  I1 linear = ridge / ridge_gcv / elasticnet (+ transform grammar);
#   I2 kernel / local = ksmooth / gpr / krr, plus linear-headed pipelines over kernel feature maps
#      (rff_cos / rff_sin / nystrom / landmark_dist);  I3 meta = portfolio / ensemble / stack.
# Actual head inventory of the pinned worker (spec.rs HEAD_OPS): ridge, logistic(=ridge), ridge_gcv, elasticnet,
# histgbm, gbm(=histgbm), ensemble, ensemble_ridge_gbm(=ensemble), histgbm2, krr, gpr, ksmooth, stack, portfolio.
# No MLP / forest / SVR / PLS / catboost / knn head exists (registry `knn_loo` is `(ksmooth k=32)`).
ISLAND_OF_FAMILY = {
    "histgbm": 0,
    "histgbm2": 0,
    "ridge": 1,
    "ridge_gcv": 1,
    "elasticnet": 1,
    "ksmooth": 2,
    "krr": 2,
    "gpr": 2,
    "ensemble": 3,
    "stack": 3,
    "portfolio": 3,
    "xtrees": 0,  # LEARNERS-V1: tree island
    "rforest": 0,
    "oblivious": 0,
}
ISLAND_NAMES = ("I0_tree", "I1_linear", "I2_kernel_local", "I3_meta")
KERNEL_MAP_OPS = frozenset(["rff_cos", "rff_sin", "nystrom", "landmark_dist"])
FEAT412_KNOWN_OPS = frozenset(
    [
        "bin_onehot",
        "blom",
        "cheb",
        "elasticnet",
        "hinge",
        "histgbm",
        "ksmooth",
        "oblique_hinge",
        "pair_prod",
        "poly_interact",
        "quantile_bins",
        "rand_proj",
        "rank",
        "rff_cos",
        "rff_sin",
        "ridge",
        "ridge_gcv",
        "robust_scale",
        "row_stats",
        "sigmoid",
        "standardize",
        "target_bin_encode",
        "winsor_clip",
    ]
)


def head_family(g: Union[Genome, str]) -> str:
    g = parse(g) if isinstance(g, str) else g
    if isinstance(g, Portfolio):
        return "portfolio"
    h = g.head
    if h is None:
        return "ridge"  # spec.rs: no head => Ridge { lambda_frac: 0.02 }
    if h.op == "portfolio":
        return "portfolio"
    return HEAD_FAMILY.get(h.op, h.op)


def island_of(g: Union[Genome, str]) -> int:
    g = parse(g) if isinstance(g, str) else g
    isl = ISLAND_OF_FAMILY.get(head_family(g), 3)
    if (
        isl == 1
        and isinstance(g, Pipe)
        and any(s.op in KERNEL_MAP_OPS for s in g.transforms)
    ):
        return 2
    return isl


def feat412_blind(g: Union[Genome, str]) -> bool:
    """True when the DSL uses an op the 412-feature featurizer has no column for (screener is blind to it)."""
    g = parse(g) if isinstance(g, str) else g
    pipes = g.members if isinstance(g, Portfolio) else (g,)
    return isinstance(g, Portfolio) or any(
        s.op not in FEAT412_KNOWN_OPS for p in pipes for s in p.stages
    )


# ----------------------------------------------------------------------------------------------------------------------
# Width guard (heuristic output width; keeps proposals from exploding wide tables, e.g. poly_interact on 400 columns)
# ----------------------------------------------------------------------------------------------------------------------
MAX_EXPANDED_WIDTH = 3000
# Declared cost cap (timing on non-pilot contexts: `stack` / portfolio / kernel heads over wide expansions took 10-70 s):
# pipelines ending in a heavy head (and every portfolio member) must expand to <= 256 columns.
HEAVY_HEADS = frozenset(["stack", "gpr", "krr", "ensemble", "ensemble_ridge_gbm"])
MAX_HEAVY_WIDTH = 256
# Declared cost rule for GENERATED proposals (timing, non-pilot: `stack` 5-48 s and portfolios with heavy members
# 10-60 s per eval were ~70 % of eval time). Exempt: CV5 portfolio, migration splices, BAG_FINAL, registry presets,
# bank records (each at most a few evals per run).
EXPENSIVE_MEMBER_HEADS = frozenset(["stack", "gpr", "krr"])
MAX_PORTFOLIO_GEN_MEMBERS = 3
MAX_HISTGBM2_WORK = 1200  # rounds * bags


def cost_ok(g: Union[Genome, str]) -> bool:
    g = parse(g) if isinstance(g, str) else g
    pipes = g.members if isinstance(g, Portfolio) else (g,)
    if isinstance(g, Portfolio) and len(pipes) > MAX_PORTFOLIO_GEN_MEMBERS:
        return False
    for p in pipes:
        h = p.head
        if h is None:
            continue
        if h.op == "stack":
            return False
        if isinstance(g, Portfolio) and h.op in EXPENSIVE_MEMBER_HEADS:
            return False
        if h.op == "histgbm2":
            work = int(h.get("rounds", 500)) * int(h.get("bags", 3))
            if work > (
                MAX_HISTGBM2_WORK // 2
                if isinstance(g, Portfolio)
                else MAX_HISTGBM2_WORK
            ):
                return False
    return True


def _stage_width(s: Stage, w: int) -> int:
    app = s.get("append", False) is True
    op = s.op
    if op in (
        "standardize",
        "norm_features",
        "robust_scale",
        "rank",
        "blom",
        "sigmoid",
        "winsor_clip",
        "quantile_bins",
        "target_bin_encode",
        "symlog",
        "impute",
        "norm_row",
    ):
        new = w
    elif op == "bin_onehot":
        new = w * int(s.get("bins", 6))
    elif op == "hinge":
        new = w * int(s.get("knots", 4))
    elif op == "cheb":
        new = w * int(s.get("degree", 4))
    elif op == "poly_interact":
        new = w * (w + 1) // 2
    elif op in ("rff_cos", "rff_sin", "oblique_hinge"):
        new = int(s.get("n", 32))
    elif op in ("rand_proj", "svd_project", "landmark_dist", "nystrom", "select_corr"):
        new = int(s.get("k", 16))
    elif op == "pair_prod":
        t = min(int(s.get("topk", 8)), w)
        new = t * (t - 1) // 2
    elif op == "row_stats":
        new = 8
    elif op == "spline":
        new = w * int(s.get("knots", 5))
    elif op == "pair_ratio":
        t = min(int(s.get("topk", 8)), w)
        new = t * (t - 1) // 2
    else:
        new = w
    return w + new if app else new


def expanded_width(g: Union[Genome, str], width: int) -> int:
    g = parse(g) if isinstance(g, str) else g
    pipes = g.members if isinstance(g, Portfolio) else (g,)
    worst = 0
    for p in pipes:
        w = width
        for s in p.transforms:
            if s.op in TARGET_OPS:
                continue
            w = _stage_width(s, w)
        worst = max(worst, w)
    return worst


def width_ok(g: Union[Genome, str], width: int) -> bool:
    g = parse(g) if isinstance(g, str) else g
    if expanded_width(g, width) > MAX_EXPANDED_WIDTH:
        return False
    pipes = g.members if isinstance(g, Portfolio) else (g,)
    for p in pipes:
        heavy = isinstance(g, Portfolio) or (
            p.head is not None and p.head.op in HEAVY_HEADS
        )
        if heavy and expanded_width(p, width) > max(MAX_HEAVY_WIDTH, width):
            return False
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Featurization (feat412 + proj16 via JopeDimeMicro rust_evaluator.featurize_dsl_412; sem24 = Micro canonical signature)
# ----------------------------------------------------------------------------------------------------------------------
_RE = None
_FEAT_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _rust_evaluator():
    global _RE
    if _RE is None:
        p = str(C.MICRO / "python")
        if p not in sys.path:
            sys.path.insert(0, p)
        import rust_evaluator as RE  # noqa: E402

        _RE = RE
    return _RE


def micro_sem24(prog_mat: np.ndarray) -> np.ndarray:
    """Copied verbatim (numpy body) from JopeDimeMicro `python/phase12_multimodal_model.py::compute_semantic_struct_sig_24d`
    -- the canonical signature the H58 priors and the H317 neural screener are trained on (see neural_data.micro_sem24).
    Gemini's `common_h252.compute_sem24_numpy` is a different pooled signature and is NOT used. Parity: test_h317_gp.py."""
    x = np.asarray(prog_mat, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    n = x.shape[0]
    out = np.zeros((n, 24), dtype=np.float64)
    pos = np.maximum(x, 0.0)
    out[:, 0:5] = 1.50 * np.clip(pos[:, 24:29], 0.0, 2.0)
    out[:, 5] = np.log1p(pos[:, 22] + pos[:, 1] + pos[:, 18])
    out[:, 6] = np.log1p(pos[:, 15] + pos[:, 13] + pos[:, 20])
    out[:, 7] = np.log1p(pos[:, 2] + pos[:, 12] + pos[:, 6])
    out[:, 8] = np.log1p(pos[:, 11] + pos[:, 10] + pos[:, 16] + pos[:, 17])
    out[:, 9] = np.log1p(pos[:, 19] + pos[:, 14] + pos[:, 0] + pos[:, 3] + pos[:, 4])
    out[:, 10] = np.log1p(pos[:, 52]) / 6.0
    out[:, 11] = np.clip(pos[:, 46] / 6.0, 0.0, 2.5)
    out[:, 12] = np.log1p(100.0 * pos[:, 50]) / 4.0
    out[:, 13] = np.log1p(pos[:, 58]) / 3.0
    out[:, 14] = np.log1p(pos[:, 60]) / 5.0
    out[:, 15] = np.clip(pos[:, 34] / 7.0, 0.0, 2.5)
    out[:, 16] = np.clip(pos[:, 42] / 8.0, 0.0, 2.5)
    out[:, 17] = np.log1p(100.0 * pos[:, 86]) / 5.0
    out[:, 18] = np.clip(pos[:, 411] / 6.0, 0.0, 2.0)
    out[:, 19] = np.log1p(pos[:, 23]) / 6.0
    out[:, 20] = np.log1p(np.sum(pos[:, 0:23], axis=1)) / 3.0
    out[:, 21] = np.clip(np.sum(pos[:, 111:122], axis=1), 0.0, 1.5)
    out[:, 22] = np.clip(np.sum(pos[:, 122:132], axis=1), 0.0, 1.5)
    out[:, 23] = (pos[:, 411] >= 2.5).astype(np.float64) * 0.8
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    out_norm = (out / norms).astype(np.float32)
    return out_norm[0] if single else out_norm


def featurize(dsl: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feat412 float32, proj16 float32, sem24 float32), cached by DSL text."""
    hit = _FEAT_CACHE.get(dsl)
    if hit is None:
        f412, p16 = _rust_evaluator().featurize_dsl_412(dsl)
        s24 = micro_sem24(f412[None, :])[0]
        hit = (f412.astype(np.float32), p16.astype(np.float32), s24)
        _FEAT_CACHE[dsl] = hit
    return hit


def featurize_batch(dsls: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs = [featurize(d) for d in dsls]
    if not fs:
        return (
            np.zeros((0, 412), np.float32),
            np.zeros((0, 16), np.float32),
            np.zeros((0, 24), np.float32),
        )
    return (
        np.stack([f[0] for f in fs]),
        np.stack([f[1] for f in fs]),
        np.stack([f[2] for f in fs]),
    )


ALL_FAMILIES = (
    "histgbm",
    "histgbm2",
    "ksmooth",
    "krr",
    "gpr",
    "ridge",
    "ridge_gcv",
    "elasticnet",
    "ensemble",
    "stack",
    "portfolio",
)
if LEARNERS_V1:  # widens ext_descriptor by 3 columns: correctors fitted on the old width must be refit for this arm
    ALL_FAMILIES = ALL_FAMILIES + ("xtrees", "rforest", "oblivious")


def ext_descriptor(dsl: str) -> np.ndarray:
    """Grammar-complete descriptor (sem24 + head-family one-hot + log op counts) used by the residual corrector so live
    results on ops the feat412 featurizer cannot see (histgbm2/krr/gpr/stack/portfolio) still inform similar programs."""
    s24 = featurize(dsl)[2].astype(np.float64)
    fam = head_family(dsl)
    oh = np.array([1.0 if fam == f else 0.0 for f in ALL_FAMILIES])
    g = parse(dsl)
    pipes = g.members if isinstance(g, Portfolio) else (g,)
    ops = [s.op for p in pipes for s in p.transforms]
    cnt = np.array([math.log1p(ops.count(o)) for o in STAGE_OPS])
    return np.concatenate([s24, 0.8 * oh, 0.3 * cnt, [0.05 * complexity(dsl)]])


# ----------------------------------------------------------------------------------------------------------------------
# Random stage / head generators
# ----------------------------------------------------------------------------------------------------------------------
STAGE_OPS = (
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "winsor_clip",
    "sigmoid",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "cheb",
    "poly_interact",
    "pair_prod",
    "target_bin_encode",
    "rand_proj",
    "oblique_hinge",
    "rff_sin",
    "rff_cos",
    "row_stats",
    "svd_project",
    "spline",
    "symlog",
    "select_corr",
    "nystrom",
    "landmark_dist",
    "pair_ratio",
)
# regime-flavoured stage groups (rust_evaluator STAGE_POOL_BY_REGIME + the worker's extra transforms), used by
# island-specific synthesis. All ops below parsed and ran on the worker (non-pilot probe, gp/probe_grammar/).
STAGE_GROUPS = {
    "scale": ("standardize", "robust_scale", "rank", "blom", "winsor_clip", "symlog"),
    "nonlinear": (
        "cheb",
        "hinge",
        "poly_interact",
        "sigmoid",
        "quantile_bins",
        "bin_onehot",
        "spline",
    ),
    "interaction": ("pair_prod", "poly_interact", "oblique_hinge", "pair_ratio"),
    "projection": (
        "rand_proj",
        "svd_project",
        "row_stats",
        "target_bin_encode",
        "select_corr",
    ),
    "kernel_map": ("rff_sin", "rff_cos", "nystrom", "landmark_dist"),
}
TARGET_KINDS = ("rank_gauss", "log1p", "winsor", "zscore")
_RF_SEEDS = (701, 1337, 2026, 4242, 9001)


def _r6(x: float) -> float:
    return float(f"{x:.6g}")


def rand_stage(op: str, rng: random.Random) -> Stage:
    app = rng.random() < 0.75
    kw: Dict[str, Value] = {}
    if op in (
        "standardize",
        "robust_scale",
        "rank",
        "blom",
        "sigmoid",
        "poly_interact",
        "row_stats",
    ):
        pass
    elif op == "winsor_clip":
        kw["q"] = _r6(rng.uniform(0.01, 0.2))
    elif op == "quantile_bins":
        kw["bins"] = rng.choice([4, 8, 12, 16, 24, 32])
    elif op == "bin_onehot":
        kw["bins"] = rng.choice([3, 4, 6, 8, 12])
    elif op == "hinge":
        kw["knots"] = rng.randint(2, 6)
    elif op == "cheb":
        kw["degree"] = rng.randint(2, 5)
    elif op == "pair_prod":
        kw["topk"] = rng.choice([4, 8, 12, 16, 24])
    elif op == "target_bin_encode":
        kw["bins"] = rng.choice([4, 8, 12, 16])
    elif op == "rand_proj":
        kw["k"] = rng.choice([8, 16, 32, 48])
        kw["seed"] = rng.choice(_RF_SEEDS)
    elif op == "svd_project":
        kw["k"] = rng.choice([2, 4, 6, 8, 12])
    elif op == "spline":
        kw["knots"] = rng.choice([3, 4, 5, 6, 8])
    elif op == "symlog":
        kw["c"] = rng.choice([0.5, 1.0, 2.0])
    elif op == "select_corr":
        kw["k"] = rng.choice([4, 8, 16, 32])
        if rng.random() < 0.3:
            kw["mode"] = "var"
    elif op in ("nystrom", "landmark_dist"):
        kw["gamma"] = _r6(math.exp(rng.uniform(math.log(0.05), math.log(2.0))))
        kw["k"] = rng.choice([16, 32, 64])
    elif op == "pair_ratio":
        kw["topk"] = rng.choice([4, 8, 12])
    elif op == "oblique_hinge":
        kw["gamma"] = _r6(math.exp(rng.uniform(math.log(0.05), math.log(1.0))))
        kw["n"] = rng.choice([16, 32, 48])
        kw["seed"] = rng.choice(_RF_SEEDS)
    elif op in ("rff_sin", "rff_cos"):
        kw["gamma"] = _r6(math.exp(rng.uniform(math.log(0.05), math.log(2.0))))
        kw["n"] = rng.choice([16, 32, 64])
        kw["seed"] = rng.choice(_RF_SEEDS)
    else:
        raise GrammarError(f"unknown stage op {op}")
    if app:
        kw["append"] = True
    return Stage.make(op, **kw)


HEADS_BY_ISLAND = {
    0: ("histgbm", "histgbm", "histgbm2"),
    1: ("ridge_gcv", "ridge", "elasticnet"),
    2: ("ksmooth", "gpr", "krr"),
    3: (
        "ensemble",
    ),  # explicit portfolios are built from members (op_portfolio); `stack` excluded (cost rule)
}
if LEARNERS_V1:
    HEADS_BY_ISLAND[0] = ("histgbm", "histgbm", "histgbm2", "xtrees", "rforest", "oblivious")
GPR_WIDE_M_CAP = 300  # declared cost cap: grammar-generated gpr heads use m <= 300 when the table has > 100 columns


def rand_target(rng: random.Random) -> Stage:
    kind = rng.choice(TARGET_KINDS)
    return (
        Stage.make("target", kind=kind, q=_r6(rng.uniform(0.01, 0.1)))
        if kind == "winsor"
        else Stage.make("target", kind=kind)
    )


def rand_head(op: str, rng: random.Random) -> Stage:
    if op == "histgbm":
        return Stage.make(
            op,
            depth=rng.randint(2, 6),
            lr=_r6(math.exp(rng.uniform(math.log(0.02), math.log(0.2)))),
            rounds=rng.choice([60, 90, 120, 160, 220, 280]),
        )
    if op == "histgbm2":
        kw = dict(
            bags=rng.choice([1, 2, 3]),
            leaves=rng.choice([7, 15, 31]),
            lr=_r6(math.exp(rng.uniform(math.log(0.03), math.log(0.12)))),
            rounds=rng.choice([150, 250, 400]),
            row_subsample=rng.choice([0.65, 0.85, 1.0]),
        )
        if rng.random() < 0.3:
            kw["col_subsample"] = rng.choice([0.6, 0.8])
        if rng.random() < 0.2:
            kw["val_pct"] = rng.choice([10, 15, 25])
        return Stage.make(op, **kw)
    if op == "ksmooth":
        return Stage.make(
            op,
            gamma=rng.choice([0.5, 1.0, 2.0, 4.0, 8.0]),
            k=rng.choice([8, 16, 32, 64]),
        )
    if op == "krr":
        return Stage.make(op, m=rng.choice([200, 400]))
    if op == "gpr":
        return Stage.make(
            op, kernel=rng.choice(["rbf", "rbf+linear"]), steps=rng.choice([20, 40])
        )
    if op == "ridge":
        return Stage.make(
            op, lambda_frac=_r6(math.exp(rng.uniform(math.log(1e-3), math.log(0.3))))
        )
    if op == "ridge_gcv":
        return Stage.make(op)
    if op == "elasticnet":
        return Stage.make(
            op,
            alpha=_r6(math.exp(rng.uniform(math.log(1e-4), math.log(0.05)))),
            l1_ratio=rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
        )
    if op == "ensemble":
        return Stage.make(
            op,
            depth=rng.randint(2, 5),
            lambda_frac=rng.choice([0.002, 0.02, 0.1]),
            lr=_r6(math.exp(rng.uniform(math.log(0.03), math.log(0.15)))),
            rounds=rng.choice([80, 120, 160]),
        )
    if op == "stack":
        return Stage.make(op)
    # LEARNERS-V1 heads (superset worker). Domains are the worker's strict ranges; values outside them fail the eval.
    if op == "xtrees":
        return Stage.make(
            op,
            trees=rng.choice([60, 100, 150]),
            min_leaf=rng.choice([1, 2, 3, 5]),
            max_features=rng.choice([0.5, 0.7, 1.0]),
        )
    if op == "rforest":
        return Stage.make(
            op,
            trees=rng.choice([60, 100, 150]),
            min_leaf=rng.choice([1, 2, 3, 5]),
            max_features=rng.choice([0.33, 0.5, 0.7]),
        )
    if op == "oblivious":
        return Stage.make(
            op,
            depth=rng.choice([4, 5, 6]),
            lr=_r6(math.exp(rng.uniform(math.log(0.03), math.log(0.12)))),
            rounds=rng.choice([200, 300, 500]),
            l2=rng.choice([1.0, 3.0, 10.0]),
        )
    raise GrammarError(f"unknown head op {op}")


def rand_head_for_island(island: int, rng: random.Random) -> Stage:
    return rand_head(rng.choice(HEADS_BY_ISLAND[island]), rng)


# ----------------------------------------------------------------------------------------------------------------------
# Candidate record
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    dsl: str
    sha: str
    operator: str
    parents: Tuple[str, ...] = ()
    island: int = 0
    meta: Tuple[Tuple[str, str], ...] = ()

    @staticmethod
    def from_genome(
        g: Genome, operator: str, parents: Sequence[str] = (), **meta
    ) -> "Candidate":
        dsl = canonical(g)
        return Candidate(
            dsl,
            sha256_text(dsl),
            operator,
            tuple(parents),
            island_of(g),
            tuple(sorted((k, str(v)) for k, v in meta.items())),
        )

    @staticmethod
    def from_dsl(
        dsl: str, operator: str, parents: Sequence[str] = (), **meta
    ) -> "Candidate":
        return Candidate.from_genome(parse(dsl), operator, parents, **meta)

    def genome(self) -> Genome:
        return parse(self.dsl)


# ----------------------------------------------------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------------------------------------------------
_NEW_HEAD_DOMAINS = {  # (op, key) -> (lo, hi, float|int); inside the superset worker's strict domains
    ("xtrees", "max_features"): (0.1, 1.0, "f"),
    ("rforest", "max_features"): (0.1, 1.0, "f"),
    ("xtrees", "trees"): (20, 300, "i"),
    ("rforest", "trees"): (20, 300, "i"),
    ("xtrees", "min_leaf"): (1, 32, "i"),
    ("rforest", "min_leaf"): (1, 32, "i"),
    ("oblivious", "depth"): (2, 8, "i"),
    ("oblivious", "rounds"): (40, 1500, "i"),
    ("oblivious", "lr"): (0.01, 0.3, "f"),
    ("oblivious", "l2"): (0.1, 100.0, "f"),
    ("oblivious", "subsample"): (0.5, 1.0, "f"),
    ("oblivious", "rsm"): (0.3, 1.0, "f"),
}


def _mutate_params(s: Stage, rng: random.Random) -> Stage:
    """EditKind::Parameter, adapted from rust_evaluator.mutate_parameter_clause (same domains, extended keys)."""
    upd: Dict[str, Value] = {}
    for key, val in s.params:
        if key == "seed" or isinstance(val, (bool, str)):
            continue
        if s.op in NEW_HEADS and (s.op, key) in _NEW_HEAD_DOMAINS:  # LEARNERS-V1 only; existing ops untouched
            lo, hi, kind = _NEW_HEAD_DOMAINS[(s.op, key)]
            if kind == "f":
                upd[key] = _r6(min(max(float(val) * math.exp(rng.uniform(-0.30, 0.30)), lo), hi))
            else:
                upd[key] = int(
                    min(max(int(round(int(val) * rng.uniform(0.75, 1.30))) + rng.choice([-1, 0, 1]), lo), hi)
                )
            continue
        if isinstance(val, float) or key in (
            "q",
            "lr",
            "gamma",
            "lambda_frac",
            "alpha",
            "l1_ratio",
            "row_subsample",
        ):
            v = float(val)
            nv = v * math.exp(rng.uniform(-0.30, 0.30))
            lo, hi = {
                "q": (0.01, 0.25),
                "lr": (0.015, 0.30),
                "gamma": (0.05, 8.0),
                "lambda_frac": (1e-4, 1.0),
                "alpha": (1e-5, 1.0),
                "l1_ratio": (0.05, 0.95),
                "row_subsample": (0.5, 1.0),
                "col_subsample": (0.5, 1.0),
                "c": (1e-3, 1e3),
                "delta": (1e-6, 1e6),
            }.get(key, (1e-6, 1e6))
            upd[key] = _r6(min(max(nv, lo), hi))
        else:
            v = int(val)
            if key == "depth":
                nv = min(max(v + rng.choice([-1, 1]), 2), 6)
            elif key == "degree":
                nv = min(max(v + rng.choice([-1, 1]), 2), 7)
            elif key == "knots":
                nv = min(max(v + rng.choice([-1, 1]), 2), 8)
            elif key == "rounds":
                nv = min(max(int(round(v * rng.uniform(0.75, 1.30))), 40), 600)
            elif key in ("bins", "topk", "k", "n"):
                nv = min(max(int(round(v * rng.uniform(0.75, 1.35))), 2), 128)
                if s.op == "pair_prod":
                    nv = min(max(nv, 2), 24)
            elif key == "leaves":
                nv = min(max(int(round(v * rng.choice([0.5, 2.0]))), 3), 127)
            elif key == "bags":
                nv = min(max(v + rng.choice([-1, 1]), 1), 5)
            elif key == "val_pct":
                nv = min(max(v + rng.choice([-5, 5]), 5), 40)
            elif key == "m":
                nv = min(max(int(round(v * rng.uniform(0.75, 1.30))), 50), 1000)
            elif key == "steps":
                nv = min(max(v + rng.choice([-10, 10]), 10), 80)
            else:
                nv = v
            upd[key] = nv
    return s.with_params(**upd) if upd else s


def op_parameter(g: Genome, rng: random.Random) -> Genome:
    if isinstance(g, Portfolio):
        j = rng.randrange(len(g.members))
        ms = list(g.members)
        ms[j] = op_parameter(ms[j], rng)
        return Portfolio(tuple(ms), g.folds, g.repeats_auto)
    return Pipe(tuple(_mutate_params(s, rng) for s in g.stages))


def op_replace_head(
    g: Genome, rng: random.Random, island: Optional[int] = None
) -> Genome:
    if isinstance(g, Portfolio):
        j = rng.randrange(len(g.members))
        ms = list(g.members)
        ms[j] = op_replace_head(ms[j], rng, rng.choice([0, 1, 2]))
        return Portfolio(tuple(ms), g.folds, g.repeats_auto)
    isl = island if island is not None else rng.randrange(4)
    return Pipe(tuple(g.transforms) + (rand_head_for_island(isl, rng),))


def op_insert_stage(
    g: Genome, rng: random.Random, group: Optional[str] = None
) -> Genome:
    if isinstance(g, Portfolio):
        j = rng.randrange(len(g.members))
        ms = list(g.members)
        ms[j] = op_insert_stage(ms[j], rng, group)
        return Portfolio(tuple(ms), g.folds, g.repeats_auto)
    ops = STAGE_GROUPS[group] if group else STAGE_OPS
    tr = list(g.transforms)
    has_target = bool(tr) and tr[0].op in TARGET_OPS
    if not has_target and group is None and rng.random() < 0.1:
        head = g.head
        return Pipe(
            (rand_target(rng),) + tuple(tr) + ((head,) if head is not None else ())
        )
    st = rand_stage(rng.choice(ops), rng)
    lo = 1 if has_target else 0
    pos = rng.randint(lo, len(tr))
    tr.insert(pos, st)
    if len(tr) > MAX_PIPE_STAGES - 1:
        tr = tr[: MAX_PIPE_STAGES - 1]
    head = g.head
    return Pipe(tuple(tr) + ((head,) if head is not None else ()))


def op_delete_stage(g: Genome, rng: random.Random) -> Optional[Genome]:
    if isinstance(g, Portfolio):
        if len(g.members) > 2 and rng.random() < 0.5:
            j = rng.randrange(len(g.members))
            return Portfolio(
                tuple(m for k, m in enumerate(g.members) if k != j),
                g.folds,
                g.repeats_auto,
            )
        j = rng.randrange(len(g.members))
        sub = op_delete_stage(g.members[j], rng)
        if sub is None:
            return None
        ms = list(g.members)
        ms[j] = sub
        return Portfolio(tuple(ms), g.folds, g.repeats_auto)
    tr = list(g.transforms)
    if len(tr) < 1:
        return None
    del tr[rng.randrange(len(tr))]
    head = g.head
    stages = tuple(tr) + ((head,) if head is not None else ())
    return Pipe(stages) if stages else None


def op_crossover(a: Genome, b: Genome, rng: random.Random) -> Optional[Genome]:
    """Stage-list one-point crossover (rust_evaluator EditKind Crossover); portfolio member exchange for portfolios."""
    if isinstance(a, Portfolio) or isinstance(b, Portfolio):
        pa = list(a.members) if isinstance(a, Portfolio) else [a]
        pb = list(b.members) if isinstance(b, Portfolio) else [b]
        k = max(1, len(pa) // 2)
        ms = pa[:k] + pb[len(pb) // 2 :]
        ms = list(dict.fromkeys(ms))[:4]
        return Portfolio(tuple(ms)) if len(ms) >= 2 else None
    sa, sb = list(a.transforms), list(b.transforms)
    cross = (sa[: max(1, len(sa) // 2)] + sb[len(sb) // 2 :])[:5]
    cross = [s for s in cross if s.op not in TARGET_OPS or cross.index(s) == 0]
    head = b.head or a.head
    stages = tuple(cross) + ((head,) if head is not None else ())
    return Pipe(stages) if stages else None


def op_synth(
    island: int,
    rng: random.Random,
    n_stages: Optional[int] = None,
    group: Optional[str] = None,
) -> Genome:
    """Grammar synthesis (rust_evaluator 'landmark_synthesis' adapted per island)."""
    if island == 3 and rng.random() < 0.5:
        members = [op_synth(i, rng) for i in rng.sample([0, 1, 2], 2)]
        return Portfolio(tuple(members))
    k = n_stages if n_stages is not None else rng.randint(1, 3)
    if group is None:
        group = {
            0: "nonlinear",
            1: rng.choice(["nonlinear", "projection", "interaction"]),
            2: "scale",
            3: "interaction",
        }[island]
    stages: List[Stage] = []
    if island == 1 and rng.random() < 0.2:
        stages.append(rand_target(rng))
    if (
        island == 2 and rng.random() < 0.4
    ):  # kernel feature map + linear head (rff_ridge style)
        stages.append(rand_stage("standardize", rng))
        stages.append(rand_stage(rng.choice(STAGE_GROUPS["kernel_map"]), rng))
        return Pipe(tuple(stages) + (rand_head("ridge_gcv", rng),))
    for j in range(k):
        grp = "scale" if j == 0 and island in (1, 2) else group
        stages.append(rand_stage(rng.choice(STAGE_GROUPS[grp]), rng))
    return Pipe(tuple(stages) + (rand_head_for_island(island, rng),))


def op_portfolio(members: Sequence[Genome]) -> Optional[Genome]:
    """Migration splice combinator: explicit `(portfolio (member ...)+ (blend kind=nnls folds=5 ...))`.

    This is the only multi-model combinator in the pinned grammar (plan names blend2/stack2/residual_boost do not exist).
    Portfolio members are flattened (no nesting) and deduplicated; 2..4 members.
    """
    flat: List[Pipe] = []
    for m in members:
        for p in m.members if isinstance(m, Portfolio) else (m,):
            if p not in flat:
                flat.append(p)
    flat = flat[:4]
    return Portfolio(tuple(flat)) if len(flat) >= 2 else None


def head_swap(g: Genome, island: int, rng: random.Random) -> Genome:
    """Ring-migration helper: keep the migrant's transforms, give it a head of the receiving island."""
    if island == 3:
        return op_portfolio([g, op_synth(rng.choice([0, 1, 2]), rng)]) or g
    if isinstance(g, Portfolio):
        g = g.members[0]
    tr = tuple(s for s in g.transforms if not (island == 1 and s.op in KERNEL_MAP_OPS))
    return Pipe(tr + (rand_head_for_island(island, rng),))


def cap_gpr(g: Genome, width: int) -> Genome:
    """Declared gpr cost cap for wide tables (applied to grammar-generated programs only)."""
    if width <= 100:
        return g

    def fix(p: Pipe) -> Pipe:
        h = p.head
        if h is not None and h.op == "gpr" and int(h.get("m", 500)) > GPR_WIDE_M_CAP:
            return Pipe(p.transforms + (h.with_params(m=GPR_WIDE_M_CAP),))
        return p

    if isinstance(g, Portfolio):
        return Portfolio(tuple(fix(m) for m in g.members), g.folds, g.repeats_auto)
    return fix(g)


# Worker registry `core_v1` member pipelines (feat-math members.rs), canonicalized. Scheme E uses REGISTRY_E_ORDER.
REGISTRY_PRESETS = {
    "ridge_gcv_raw": "(pipe X (ridge_gcv))",
    "ridge_quad": "(pipe X (cheb append=true degree=3) (pair_prod append=true topk=24) (ridge_gcv))",
    "gpr_lin": "(pipe X (gpr kernel=rbf+linear steps=40))",
    "gpr_blom": "(pipe X (blom) (gpr kernel=rbf+linear steps=40))",
    "knn_loo": "(pipe X (ksmooth k=32))",
    "rff_ridge": "(pipe X (rff_cos append=false n=160) (rff_sin append=true n=160) (ridge_gcv))",
    "histgbm_d6": "(pipe X (histgbm2 bags=3 leaves=63 lr=0.06 rounds=400))",
    "histgbm_d3": "(pipe X (histgbm2 leaves=15 lr=0.08 rounds=500))",
    "histgbm_slow": "(pipe X (histgbm2 col_subsample=0.7 leaves=31 lr=0.03 rounds=900))",
    "enet_quad": "(pipe X (cheb append=true degree=3) (pair_prod append=true topk=24) (elasticnet))",
    "gpr_iso": "(pipe X (gpr kernel=rbf steps=40))",
    "gam_tbe": "(pipe X (target_bin_encode bins=12) (ridge_gcv))",
}
# 8 of the 12 (fixed, declared): drops ridge_gcv_raw (trivial), histgbm_slow (900 rounds), enet_quad (near-duplicate
# of ridge_quad; registry gate p <= 40), gpr_iso (registry gate n < 150). Interleaving keeps every head class early.
REGISTRY_E_ORDER = (
    "histgbm_d6",
    "ridge_quad",
    "gpr_lin",
    "knn_loo",
    "histgbm_d3",
    "rff_ridge",
    "gpr_blom",
    "gam_tbe",
)


def bag_final(g: Genome) -> Optional[Genome]:
    """BAG_FINAL secondary: histgbm(depth d, lr, rounds) -> histgbm2(leaves 2^d-1, lr, rounds, bags=5, row_subsample=0.85);
    histgbm2 -> bags=5 row_subsample=0.85. Tree-head pipelines only (None otherwise)."""
    if not isinstance(g, Pipe) or g.head is None:
        return None
    h = g.head
    if h.op in ("histgbm", "gbm"):
        d = int(h.get("depth", 3))
        nh = Stage.make(
            "histgbm2",
            bags=5,
            leaves=max(3, min(255, 2**d - 1)),
            lr=h.get("lr", 0.1),
            rounds=int(h.get("rounds", 150)),
            row_subsample=0.85,
        )
    elif h.op == "histgbm2":
        nh = h.with_params(bags=5, row_subsample=0.85)
    else:
        return None
    return Pipe(g.transforms + (nh,))


# ----------------------------------------------------------------------------------------------------------------------
# Multiplicative coordinate line-search (plan §3.3, gens 8-10) over the parameters the grammar actually has
# ----------------------------------------------------------------------------------------------------------------------
def _line_values(op: str, key: str, val: Value) -> List[Value]:
    if isinstance(val, bool) or isinstance(val, str):
        return []
    if key in ("lambda_frac", "alpha"):  # alpha / l2
        return [_r6(float(val) * f) for f in (0.25, 0.5, 2.0, 4.0)]
    if key == "lr":  # learning_rate
        return [_r6(min(max(float(val) * f, 0.005), 0.5)) for f in (0.6, 1.4)]
    if key == "depth":  # max_depth
        return [d for d in (int(val) - 1, int(val) + 1) if 1 <= d <= 8]
    if key == "leaves":  # histgbm2 max_depth analogue: leaves x {0.5, 2}
        return [
            x
            for x in (max(3, int(val) // 2), min(255, int(val) * 2 + 1))
            if x != int(val)
        ]
    if key == "row_subsample":  # subsample
        return [x for x in (0.65, 0.85, 1.0) if abs(x - float(val)) > 1e-9]
    if key == "k" and op in ("svd_project", "rand_proj"):  # n_components
        return [x for x in (int(val) - 2, int(val) + 2) if x >= 1]
    return []


def line_search_neighbors(g: Genome) -> List[Genome]:
    out: List[Genome] = []
    pipes = list(g.members) if isinstance(g, Portfolio) else [g]
    for pi, p in enumerate(pipes):
        for si, s in enumerate(p.stages):
            for key, val in s.params:
                for nv in _line_values(s.op, key, val):
                    ns = list(p.stages)
                    ns[si] = s.with_params(**{key: nv})
                    np_ = Pipe(tuple(ns))
                    if isinstance(g, Portfolio):
                        ms = list(pipes)
                        ms[pi] = np_
                        out.append(Portfolio(tuple(ms), g.folds, g.repeats_auto))
                    else:
                        out.append(np_)
    return out


def safe_candidate(
    g: Optional[Genome],
    operator: str,
    parents: Sequence[str],
    width: int,
    cost_exempt: bool = False,
    **meta,
) -> Optional[Candidate]:
    """Canonicalize + re-parse + width guard (+ the generated-proposal cost rule unless exempt); None when the
    proposal is not a valid, affordable program."""
    if g is None:
        return None
    try:
        g = cap_gpr(g, width)
        dsl = canonical(g)
        g2 = parse(dsl)
        if canonical(g2) != dsl:
            return None
    except GrammarError:
        return None
    if not width_ok(g2, width) or (not cost_exempt and not cost_ok(g2)):
        return None
    return Candidate(
        dsl,
        sha256_text(dsl),
        operator,
        tuple(parents),
        island_of(g2),
        tuple(sorted((k, str(v)) for k, v in meta.items())),
    )
