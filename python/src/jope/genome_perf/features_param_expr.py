"""G4 (parameter-value semantics) + G5 (expression semantics).

Why this module exists
----------------------
The existing featurizer collapses every parameter into a single summed ``log1p_width``, so
parameter VALUES are invisible to it. Measured within-dataset percentile spreads say that is
throwing away real signal: elasticnet ``alpha`` spreads 0.224 and is near-monotone
(1e-6 -> 0.45, 0.01 -> 0.67), ``rff_sin n`` spreads 0.342, ``nystrom k`` spreads 0.247.

Likewise for expressions, semantics beat size: ``row_mean`` +0.051 and ``row_l2`` +0.028 but
``row_max`` -0.045 and ``row_std`` -0.022. A single "uses a row stat" flag would average those to
roughly nothing, so every row aggregate gets its OWN column here.

Both groups vary between genomes on the same dataset, which is the only thing that can carry a main
effect under GroupKFold-by-dataset.

Conventions that callers should know
------------------------------------
``g4_bin_<op>_<param>``
    Normalized ordinal position of the observed value within the grammar's declared bin grid,
    in [0, 1]; **-1.0** when the op is absent from the pipeline, or present but never carrying
    that param. Columns exist only for (op, param) pairs whose grammar bins are non-empty, so the
    column set is a pure function of the grammar.

    Values off the declared grid are common in the real corpus and are NOT snapped to the nearest
    bin -- ``pair_prod topk`` alone occupies a dense 1..17 range against a declared grid of
    (4, 8, 12), and ``elasticnet alpha`` is mostly continuous jitter. Snapping would erase exactly
    the monotone within-grid ordering that F5 measured. Instead numeric values are placed by
    monotone piecewise-linear interpolation between the surrounding bins and clamped to [0, 1], so
    an exact bin value still lands on its exact ordinal and an off-grid value lands between its
    neighbours. Categorical bins use exact match; an unrecognized category maps to 0.5 (the middle)
    rather than 0.0, so it is not silently asserted to be the lowest ordinal.

    When an op appears more than once, the normalized values are averaged.

``g5_ast_depth`` / ``g5_ast_nodes``
    A deliberately small, deterministic estimator over the expression string -- determinism matters
    more than textbook AST exactness, since this must reproduce identically forever.

      depth = maximum parenthesis nesting reached.
      nodes = (function-call tokens) + (binary operators) + (leaf tokens).

    where a function-call token is an identifier immediately followed by ``(``, a leaf token is an
    identifier NOT followed by ``(`` plus every numeric literal, and a binary operator is an
    occurrence of + - * / whose preceding non-space character is ``)``, a digit, a letter or ``_``
    (so a unary/leading minus does not count). Numeric literals are tokenized including any
    scientific-notation exponent, which keeps the ``-`` in ``1e-06`` from being read as a binary
    operator.

    Worked example -- ``row_mean + col(3)/2``: funcs {col} = 1, binops {+, /} = 2,
    leaves {row_mean, 3, 2} = 3, so nodes = 6 and depth = 1.

``g5_const_magnitude_mean``
    Column indices inside ``col(...)`` are references, not constants, so they are excluded from the
    constant-magnitude average (they still count as leaves for ``ast_nodes``). Otherwise every
    ``col(6)`` would inflate this feature with what is really an addressing detail.

Everything here is pure: one genome in, floats out. No IO, no labels, no dataset information, no
randomness. ``extract`` never raises and never returns nan/inf -- unparseable ``(sym ...)`` and
portfolio genomes (``pg.ok is False``) fall through to the documented defaults.
"""
from __future__ import annotations

import functools
import math
import re

from .parse import ParsedGenome, col_refs, expr_spec, param_bins
from ..grammars import grammar_for

# Params that widen the feature matrix. Summed (not averaged) because the engine cost and the
# overfitting surface both scale with the total emitted width, not with any single stage's width.
WIDTH_KEYS = (
    "n", "k", "degree", "bins", "dims", "components", "rank",
    "n_features", "order", "knots", "topk", "n_paths", "m",
)

# Head regularization strength, in fallback order. featmath spells it `alpha`; the other two are
# accepted so a future head spelling does not silently drop the feature.
_ALPHA_KEYS = ("alpha", "l2", "lam")

_ALPHA_LO, _ALPHA_HI = -12.0, 6.0

# Row aggregates. `row_l1` is not in ExprSpec.vars_common but does occur in corpus expressions, and
# feature_names() may not depend on data -- so it is always emitted, just usually 0.0.
_EXTRA_VARS = ("row_l1",)

_FLAG_VALUES = frozenset({"true", "false"})

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
# Numeric literal INCLUDING any exponent, so the '-' of `1e-06` is never seen as a binary operator.
_NUM_RE = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
_BINOP_CHARS = frozenset("+-*/")
# A binop must follow a value-producing character; otherwise the sign is unary.
_VALUE_END = frozenset(")_")


def _as_float(text: str) -> float | None:
    """Parse a param token as a finite float, or None."""
    try:
        val = float(text)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


@functools.lru_cache(maxsize=8)
def _binned_pairs(engine: str) -> tuple[tuple[str, str], ...]:
    """Every (op, param) in the grammar declaring a non-empty bin grid, deterministically ordered.

    `param_bins` returns () -- not None -- for a declared-but-unbinned param such as `map.expr`,
    so emptiness rather than None-ness is the test for "has a grid".
    """
    grammar = grammar_for(engine)
    pairs: list[tuple[str, str]] = []
    for name, spec in {**dict(grammar.transforms), **dict(grammar.heads)}.items():
        for param in getattr(spec, "params", ()) or ():
            if param_bins(engine, name, param.name):
                pairs.append((name, param.name))
    return tuple(sorted(pairs))


@functools.lru_cache(maxsize=4096)
def _numeric_bins(engine: str, op: str, param: str) -> tuple[float, ...] | None:
    """Bins as ascending floats, or None when the grid is categorical."""
    bins = param_bins(engine, op, param)
    if not bins:
        return None
    out = []
    for raw in bins:
        val = _as_float(raw if isinstance(raw, str) else str(raw))
        if val is None:
            return None
        out.append(val)
    return tuple(sorted(out))


def _bin_position(engine: str, op: str, param: str, raw: str) -> float:
    """-> fractional ordinal index within the grid (NOT yet divided by len-1)."""
    numeric = _numeric_bins(engine, op, param)
    if numeric is None:
        bins = param_bins(engine, op, param) or ()
        texts = [b if isinstance(b, str) else str(b) for b in bins]
        if raw in texts:
            return float(texts.index(raw))
        return (len(texts) - 1) / 2.0 if len(texts) > 1 else 0.0

    val = _as_float(raw)
    if val is None:
        return (len(numeric) - 1) / 2.0 if len(numeric) > 1 else 0.0
    if len(numeric) == 1 or val <= numeric[0]:
        return 0.0
    if val >= numeric[-1]:
        return float(len(numeric) - 1)
    for j in range(len(numeric) - 1):
        lo, hi = numeric[j], numeric[j + 1]
        if lo <= val <= hi:
            return j + ((val - lo) / (hi - lo) if hi > lo else 0.0)
    return float(len(numeric) - 1)


def _is_offgrid(engine: str, op: str, param: str, raw: str) -> bool:
    numeric = _numeric_bins(engine, op, param)
    val = _as_float(raw)
    if numeric is None or val is None:
        return False
    return not any(abs(val - b) <= 1e-12 * max(1.0, abs(b)) for b in numeric)


# --------------------------------------------------------------------------------------------
# Expression tokenizer
# --------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def _tokenize(expr: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...], int, int, dict]:
    """-> (func names, leaf identifiers, constant magnitudes, depth, binop count, binop breakdown).

    See the module docstring for the exact node/depth definition this implements.
    """
    funcs: list[str] = []
    leaves: list[str] = []
    consts: list[float] = []
    n_leaf_nums = 0
    binops = {"add": 0, "sub": 0, "mul": 0, "div": 0}
    depth = max_depth = 0
    i, n = 0, len(expr)
    prev_value_char = ""     # last non-space char, for unary-vs-binary disambiguation
    pending_col_arg = False  # the next numeric literal is a col() index, not a constant

    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue

        ident = _IDENT_RE.match(expr, i)
        if ident:
            name = ident.group(0)
            j = ident.end()
            k = j
            while k < n and expr[k].isspace():
                k += 1
            if k < n and expr[k] == "(":
                funcs.append(name)
                if name == "col":
                    pending_col_arg = True
            else:
                leaves.append(name)
            i = j
            prev_value_char = "_"
            continue

        num = _NUM_RE.match(expr, i)
        if num:
            n_leaf_nums += 1
            if pending_col_arg:
                pending_col_arg = False
            else:
                val = _as_float(num.group(0))
                if val is not None:
                    consts.append(abs(val))
            i = num.end()
            prev_value_char = "_"
            continue

        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
            prev_value_char = "("
        elif ch == ")":
            depth = max(0, depth - 1)
            prev_value_char = ")"
        elif ch in _BINOP_CHARS:
            # Binary only if something value-producing precedes it.
            if prev_value_char in _VALUE_END or prev_value_char.isalnum():
                binops["add" if ch == "+" else "sub" if ch == "-" else "mul" if ch == "*" else "div"] += 1
            prev_value_char = ch
        else:
            prev_value_char = ch
        i += 1

    n_binops = sum(binops.values())
    return (
        tuple(funcs),
        tuple(leaves),
        tuple(consts),
        max_depth,
        n_binops,
        {**binops, "n_leaf_nums": n_leaf_nums},
    )


def _expr_len_band(length: int) -> float:
    """Ordinal band matching the measured inverted-U: 40-70 chars is the best band (0.521),
    >=70 collapses to 0.426."""
    if length < 20:
        return 0.0
    if length < 40:
        return 1.0
    if length < 70:
        return 2.0
    return 3.0


# --------------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _names(engine: str) -> tuple[str, ...]:
    spec = expr_spec(engine)
    out: list[str] = []

    # ---- G4 ----
    for op, param in _binned_pairs(engine):
        out.append(f"g4_bin_{op}_{param}")
    out += [
        "g4_offgrid_count",
        "g4_log_alpha_head",
        "g4_has_head_alpha",
        "g4_rff_n_log2",
        "g4_rff_n_is_pow2",
        "g4_expansion_ratio",
        "g4_param_extremeness",
        "g4_n_seeded_stages",
        "g4_seed_entropy",
        "g4_n_numeric_params",
        "g4_n_flag_params",
    ]

    # ---- G5 ----
    for var in tuple(spec.vars_common) + _EXTRA_VARS:
        out.append(f"g5_uses_{var}")
    out += [
        "g5_n_exprs",
        "g5_expr_total_len",
        "g5_expr_max_len",
        "g5_expr_len_band",
        "g5_ast_depth",
        "g5_ast_nodes",
        "g5_depth_over_budget",
        "g5_nodes_over_budget",
        "g5_distinct_col_refs",
        "g5_distinct_col_refs_is_2",
        "g5_col_ref_max_index",
        "g5_col_ref_span",
    ]
    for fn in tuple(spec.funcs1) + tuple(spec.funcs2) + tuple(spec.funcs3):
        out.append(f"g5_func_{fn}")
    out += [
        "g5_binop_add",
        "g5_binop_sub",
        "g5_binop_mul",
        "g5_binop_div",
        "g5_has_division",
        "g5_const_magnitude_mean",
        "g5_is_pure_rowstat",
    ]
    return tuple(out)


def feature_names(engine: str = "featmath") -> list[str]:
    """Fixed, deterministic, grammar-derived column list. Never depends on data."""
    return list(_names(engine))


def _defaults(engine: str) -> dict[str, float]:
    out = {name: 0.0 for name in _names(engine)}
    for op, param in _binned_pairs(engine):
        out[f"g4_bin_{op}_{param}"] = -1.0
    return out


def extract(pg: ParsedGenome, engine: str = "featmath") -> dict[str, float]:
    """-> {feature name: finite float}. Never raises; unparseable genomes get the defaults."""
    out = _defaults(engine)
    if pg is None or not getattr(pg, "ok", False) or not getattr(pg, "stages", None):
        return out

    spec = expr_spec(engine)

    # ---------------------------------------------------------------- G4
    binned_seen: dict[tuple[str, str], list[float]] = {}
    extremeness: list[float] = []
    offgrid = 0
    width_sum = 0.0
    n_numeric = 0
    n_flag = 0
    n_seeded = 0
    seeds: list[str] = []
    rff_n_max: float | None = None

    for stage in pg.stages:
        has_seed = False
        for key, raw in (stage.params or {}).items():
            low = raw.strip().lower() if isinstance(raw, str) else str(raw)
            if low in _FLAG_VALUES:
                n_flag += 1
            val = _as_float(raw)
            if val is not None:
                n_numeric += 1
                if key in WIDTH_KEYS and val > 0:
                    width_sum += val
            if key == "seed":
                has_seed = True
                seeds.append(str(raw))
            if key == "n" and stage.op.startswith("rff_") and val is not None and val > 0:
                rff_n_max = val if rff_n_max is None else max(rff_n_max, val)

            bins = param_bins(engine, stage.op, key)
            if not bins:
                continue
            pos = _bin_position(engine, stage.op, key, raw if isinstance(raw, str) else str(raw))
            span = len(bins) - 1
            norm = pos / span if span > 0 else 0.0
            binned_seen.setdefault((stage.op, key), []).append(min(1.0, max(0.0, norm)))
            if span > 0:
                centre = span / 2.0
                extremeness.append(min(1.0, abs(min(span, max(0.0, pos)) - centre) / centre))
            if _is_offgrid(engine, stage.op, key, raw if isinstance(raw, str) else str(raw)):
                offgrid += 1
        if has_seed:
            n_seeded += 1

    for (op, param), vals in binned_seen.items():
        name = f"g4_bin_{op}_{param}"
        if name in out:
            out[name] = sum(vals) / len(vals)

    out["g4_offgrid_count"] = float(offgrid)
    out["g4_expansion_ratio"] = math.log1p(width_sum) if width_sum > 0 else 0.0
    out["g4_param_extremeness"] = sum(extremeness) / len(extremeness) if extremeness else 0.0
    out["g4_n_seeded_stages"] = float(n_seeded)
    out["g4_n_numeric_params"] = float(n_numeric)
    out["g4_n_flag_params"] = float(n_flag)

    head = pg.head
    if head is not None:
        for key in _ALPHA_KEYS:
            if key in (head.params or {}):
                val = _as_float(head.params[key])
                if val is None:
                    break
                out["g4_has_head_alpha"] = 1.0
                # A non-positive alpha has no log; it is the weakest-possible regularization, so it
                # pins to the clamp floor rather than being dropped.
                lg = math.log10(val) if val > 0 else _ALPHA_LO
                out["g4_log_alpha_head"] = min(_ALPHA_HI, max(_ALPHA_LO, lg))
                break

    if rff_n_max is not None and rff_n_max > 0:
        out["g4_rff_n_log2"] = math.log2(rff_n_max)
        rounded = int(round(rff_n_max))
        out["g4_rff_n_is_pow2"] = 1.0 if (
            abs(rff_n_max - rounded) < 1e-9 and rounded > 0 and rounded & (rounded - 1) == 0
        ) else 0.0

    if len(seeds) >= 2:
        counts: dict[str, int] = {}
        for s in seeds:
            counts[s] = counts.get(s, 0) + 1
        total = float(len(seeds))
        out["g4_seed_entropy"] = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )

    # ---------------------------------------------------------------- G5
    exprs = [e for e in (pg.exprs or []) if e]
    if exprs:
        lengths = [len(e) for e in exprs]
        out["g5_n_exprs"] = float(len(exprs))
        out["g5_expr_total_len"] = float(sum(lengths))
        out["g5_expr_max_len"] = float(max(lengths))
        out["g5_expr_len_band"] = _expr_len_band(max(lengths))

        all_funcs: dict[str, int] = {}
        all_leaves: set[str] = set()
        consts: list[float] = []
        depth = 0
        nodes = 0
        binops = {"add": 0, "sub": 0, "mul": 0, "div": 0}

        for expr in exprs:
            funcs, leaves, cs, d, n_bin, detail = _tokenize(expr)
            for f in funcs:
                all_funcs[f] = all_funcs.get(f, 0) + 1
            all_leaves.update(leaves)
            consts.extend(cs)
            depth = max(depth, d)
            # nodes = function calls + binary operators + leaves (identifiers + numeric literals)
            nodes += len(funcs) + n_bin + len(leaves) + detail["n_leaf_nums"]
            for k in binops:
                binops[k] += detail[k]

        for var in tuple(spec.vars_common) + _EXTRA_VARS:
            out[f"g5_uses_{var}"] = 1.0 if var in all_leaves else 0.0

        out["g5_ast_depth"] = float(depth)
        out["g5_ast_nodes"] = float(nodes)
        out["g5_depth_over_budget"] = depth / spec.max_depth if spec.max_depth else 0.0
        out["g5_nodes_over_budget"] = nodes / spec.max_nodes if spec.max_nodes else 0.0

        for fn in tuple(spec.funcs1) + tuple(spec.funcs2) + tuple(spec.funcs3):
            out[f"g5_func_{fn}"] = float(all_funcs.get(fn, 0))

        out["g5_binop_add"] = float(binops["add"])
        out["g5_binop_sub"] = float(binops["sub"])
        out["g5_binop_mul"] = float(binops["mul"])
        out["g5_binop_div"] = float(binops["div"])
        out["g5_has_division"] = 1.0 if any("/" in e for e in exprs) else 0.0
        out["g5_const_magnitude_mean"] = sum(consts) / len(consts) if consts else 0.0

        refs: list[int] = []
        for expr in exprs:
            refs.extend(col_refs(expr))
        distinct = set(refs)
        out["g5_distinct_col_refs"] = float(len(distinct))
        out["g5_distinct_col_refs_is_2"] = 1.0 if len(distinct) == 2 else 0.0
        if refs:
            out["g5_col_ref_max_index"] = float(max(refs))
            out["g5_col_ref_span"] = float(max(refs) - min(refs))

        uses_rowstat = any(
            out.get(f"g5_uses_{v}", 0.0) > 0.0
            for v in tuple(spec.vars_common) + _EXTRA_VARS
        )
        out["g5_is_pure_rowstat"] = 1.0 if (uses_rowstat and not refs) else 0.0

    # Final safety net: the contract promises finite plain floats, always.
    for key, val in out.items():
        fv = float(val)
        out[key] = fv if math.isfinite(fv) else 0.0
    return out
