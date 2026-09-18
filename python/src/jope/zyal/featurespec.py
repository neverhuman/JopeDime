"""Declarative FeatureSpec: an agent's feature *theory* as data, not as code.

A FeatureSpec is 8-25 named scalar outputs, each an expression tree over an allowlisted set of
primitives that read a `ParsedGenome` (and, optionally, train-only per-dataset diagnostics). It is
the low-risk half of the campaign: most theories are arithmetic over structure ("cost concentrated
late", "rank family feeding a linear head"), and expressing those as JSON removes the entire plugin
attack surface for them. `security.py` exists for the theories that genuinely need Python.

Three decisions worth stating, because each is a trap that is cheap to fall into and expensive to
detect afterwards:

* **No `eval`, ever, not even on "validated" text.** The expression language is a JSON tree
  interpreted by `_compile_node` against a fixed dispatch table. There is no path from agent text
  to the Python parser, so there is nothing to sandbox-escape from. An expression referencing an
  unknown primitive fails `validate_spec` before it is ever compiled.

* **Dedup keys change when a spec reads diagnostics.** The corpus is 37,703 rows over 34,419
  distinct genomes, so the obvious optimisation is to parse each distinct genome once and fan out
  by inverse index. That is only valid while a spec's outputs are a pure function of the genome.
  The moment a spec calls `diag(...)`, the same genome on two datasets has two different answers
  and genome-keyed dedup would silently broadcast one dataset's value across all of them. So
  `execute_spec` keys on (genome, dataset) pairs exactly when the compiled spec reports
  `uses_diagnostics`, and refuses to run without `datasets` in that case.

* **Finiteness is enforced twice, and the second time is not redundant.** The compiled callable
  substitutes 0.0 for nan/inf so it is safe standalone. `execute_spec` then re-checks *after* the
  float32 downcast, because a finite float64 above ~3.4e38 becomes inf in float32 -- `log1p` of a
  product of costs will not get there, but a spec is free to multiply, and a silent inf in the
  feature matrix would poison a screen without failing anything.

Column naming is the controller's, never the agent's: the agent supplies a bare `name` per output
and the controller supplies `family_id`, which this module joins as ``f"{family_id}.{name}"``. The
assembled names are run through `assert_no_forbidden_inputs` before any matrix is returned.
"""
from __future__ import annotations

import ast
import collections
import math
import re
import time
import zlib

import numpy as np

from jope.genome_perf.features_cost import COST_WEIGHT
from jope.genome_perf.parse import ParsedGenome, col_refs, parse
from jope.grammars import FAMILY_LIST_V3, grammar_for

from . import assert_no_forbidden_inputs

__all__ = [
    "ABSENT", "MIN_OUTPUTS", "MAX_OUTPUTS", "MAX_EXPR_DEPTH", "MAX_EXPR_NODES",
    "PRIMITIVES", "spec_schema", "validate_spec", "compile_spec", "execute_spec",
]

# --- Limits --------------------------------------------------------------------------------------
MIN_OUTPUTS = 8
MAX_OUTPUTS = 25
MAX_EXPR_DEPTH = 12
MAX_EXPR_NODES = 200          # per output; a 25-output spec is bounded at 5,000 nodes
MAX_ABS_LITERAL = 1e12

# Sentinel for "this op is not in the pipeline" / "this param is not set or is not numeric".
# Deliberately OUTSIDE the [0,1] range of `position_norm` and outside any real param range, so a
# tree can split it off cleanly. It is a real number, not nan: nan would be substituted to 0.0 and
# become indistinguishable from a genuine zero, which is the more damaging confusion of the two.
ABSENT = -1.0

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
_FAMILY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.]{2,63}$")
_ALLOWED_TOP_KEYS = frozenset({"outputs", "family_id", "engine", "rationale", "schema"})
_REQUIRED_TOP_KEYS = frozenset({"outputs"})

_ALIASES = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
_DIV_EPS = 1e-12

_FAMILIES = frozenset(FAMILY_LIST_V3)


# --- Primitive signatures ------------------------------------------------------------------------
# Argument kinds:
#   "expr"   -- a nested expression (number literal or another primitive call)
#   "op"     -- a literal op name, checked against the engine grammar when the engine is known
#   "family" -- a literal family name, checked against FAMILY_LIST_V3 (always known)
#   "name"   -- a literal parameter name (free-form; grammars disagree on param sets)
#   "key"    -- a literal diagnostics key (checked against the forbidden-input boundary)
# A trailing "*expr" means "one or more further expression arguments".
_SIGNATURES: dict[str, tuple[str, ...]] = {
    # genome structure
    "n_stages": (),
    "n_transforms": (),
    "position_norm": ("op",),
    "count": ("op",),
    "count_family": ("family",),
    "has_op": ("op",),
    "head_op": ("op",),
    "head_family": ("family",),
    "cost_of": ("expr",),
    "param": ("op", "name"),
    "expr_len": (),
    "n_exprs": (),
    "col_refs": (),
    "last_position_norm": ("op",),
    "max_run": ("op",),
    "repetitions": ("op",),
    "mean_gap": ("op",),
    "transition": ("op", "op"),
    "trigram": ("op", "op", "op"),
    "family_switches": (),
    "family_transition": ("family", "family"),
    "head_distance": ("op",),
    "pre_head_count_family": ("family",),
    "head_input_family": ("family",),
    # expression AST and lexical structure
    "ast_node_count": ("name",),
    "ast_function_count": ("name",),
    "ast_variable_count": ("name",),
    "ast_constant_count": (),
    "ast_depth": (),
    "expr_token_ngram": ("name",),
    "expr_char_ngram": ("name",),
    # parameter geometry
    "param_count": (),
    "numeric_param_count": (),
    "categorical_param_count": (),
    "param_missing": ("op", "name"),
    "param_category": ("op", "name", "name"),
    "param_reuse": ("name",),
    "param_dispersion": ("name",),
    "adjacent_param_reuse": ("name",),
    # resolved cost/capacity traces
    "total_cost": (),
    "max_cost": (),
    "cost_centroid": (),
    "cost_switches": (),
    "runtime_cost_estimate": (),
    "train_rows": (),
    "train_width": (),
    "train_missingness": (),
    "target_cardinality": (),
    "rank_diagnostic": (),
    "condition_diagnostic": (),
    "resolved_width_min": (),
    "resolved_width_max": (),
    "resolved_width_mean": (),
    "resolved_width_range": (),
    "compression_steps": (),
    "expansion_steps": (),
    # r4 stage-state trajectories
    "cumulative_log_width": (),
    "information_retention_proxy": (),
    "width_cost_area": (),
    "bottleneck_depth": (),
    "bottleneck_duration": (),
    "bottleneck_location": (),
    "recovery_after_compression": (),
    "expansion_overshoot": (),
    "maximum_width_drawdown": (),
    "destructive_pressure_waves": (),
    "width_change_points": (),
    # deterministic structural counterfactuals
    "leave_one_stage_out_delta": (),
    "adjacent_swap_delta": (),
    "duplicate_stage_delta": (),
    "head_swap_delta": (),
    "parameter_default_delta": (),
    # grammar-relative parameter semantics
    "param_default_distance": (),
    "param_bound_distance": (),
    "param_constraint_ratio": (),
    "repeated_param_drift": (),
    "seed_collision": (),
    "regularization_capacity_balance": (),
    "missing_default_topology": (),
    # expression abstract interpretation
    "expr_constant_folded_size": (),
    "expr_effective_variables": (),
    "expr_degree_proxy": (),
    "expr_frequency_proxy": (),
    "expr_interval_growth": (),
    "expr_singularity_risk": (),
    "expr_saturation_risk": (),
    "expr_guard_coverage": (),
    "expr_lipschitz_proxy": (),
    "expr_monotonicity_proxy": (),
    "expr_symmetry_proxy": (),
    "expr_branch_balance": (),
    "expr_variable_interactions": (),
    # compatibility, program representation, and deployment
    "head_representation_supply": (),
    "head_representation_demand": (),
    "head_supply_demand_gap": (),
    "effective_capacity": (),
    "dataset_effective_rank": (),
    "dataset_intrinsic_dimension": (),
    "dataset_sparsity": (),
    "operator_entropy": (),
    "transition_entropy": (),
    "path_complexity": (),
    "genome_compression_ratio": (),
    "long_range_repeat": (),
    "semantic_cacheability": (),
    "peak_memory_proxy": (),
    "parallelism_proxy": (),
    # Section B5: expression AST path geometry
    "ast_path_count": (),
    "ast_path_max_len": (),
    "ast_path_mean_len": (),
    "ast_path_ngram": ("name",),
    "ast_repeated_path_fraction": (),
    "ast_branch_factor": (),
    # Section B5: cross-stage parameter geometry
    "cross_stage_param_coherence": ("name",),
    "param_sign": ("op", "name"),
    "param_bucket": ("op", "name", "num"),
    # Section B5: width bottleneck geometry
    "width_bottleneck": (),
    "width_bottleneck_position": (),
    # Section B5: higher-order operator n-grams
    "fourgram": ("op", "op", "op", "op"),
    "positional_bigram": ("op", "op"),
    # train-only dataset diagnostics
    "diag": ("key",),
    # arithmetic
    "add": ("expr", "expr"),
    "sub": ("expr", "expr"),
    "mul": ("expr", "expr"),
    "div": ("expr", "expr"),
    "log1p": ("expr",),
    "sqrt": ("expr",),
    "abs": ("expr",),
    "min": ("expr", "expr", "*expr"),
    "max": ("expr", "expr", "*expr"),
    "clip": ("expr", "expr", "expr"),
}
PRIMITIVES = tuple(sorted(_SIGNATURES))

_DOC = {
    "n_stages": "number of pipeline stages",
    "n_transforms": "number of non-head stages",
    "position_norm": "position of the first stage with this op, in [0,1]; -1 if absent",
    "count": "number of stages with this op",
    "count_family": "number of stages carrying this family (ops are multi-label)",
    "has_op": "1 if this op appears, else 0",
    "head_op": "1 if the final stage is this op, else 0",
    "head_family": "1 if the final stage carries this family, else 0",
    "cost_of": "cost weight of the stage at this index; negative indexes from the end (-1 = head)",
    "param": "numeric value of this param on the first stage with this op; -1 if absent/non-numeric",
    "expr_len": "total character length of all stage expressions",
    "n_exprs": "number of stages carrying an expression",
    "col_refs": "number of DISTINCT column indices referenced across all expressions",
    "last_position_norm": "position of the last occurrence of an op; -1 if absent",
    "max_run": "longest consecutive run of an op",
    "repetitions": "occurrences beyond the first",
    "mean_gap": "mean stage gap between repeated occurrences; -1 with fewer than two",
    "transition": "count of an exact adjacent op pair",
    "trigram": "count of an exact adjacent op triple",
    "family_switches": "adjacent stages whose declared family sets do not overlap",
    "family_transition": "adjacent stages carrying the requested family pair",
    "head_distance": "distance from the last matching op to the head; -1 if absent",
    "pre_head_count_family": "number of non-head stages carrying a family",
    "head_input_family": "1 if the immediately pre-head stage carries a family",
    "ast_node_count": "expression AST node count by class name",
    "ast_function_count": "calls to a named expression function",
    "ast_variable_count": "references to a named expression variable",
    "ast_constant_count": "number of expression constants",
    "ast_depth": "maximum expression AST depth",
    "expr_token_ngram": "overlapping token n-gram count, tokens joined with spaces",
    "expr_char_ngram": "overlapping character n-gram count",
    "param_count": "total stage parameters",
    "numeric_param_count": "parameters parseable as finite numbers",
    "categorical_param_count": "non-numeric parameters",
    "param_missing": "1 when an op is absent or lacks a parameter",
    "param_category": "1 when an op parameter equals a categorical literal",
    "param_reuse": "repeated uses of a parameter value across stages",
    "param_dispersion": "population standard deviation of numeric parameter values",
    "adjacent_param_reuse": "adjacent stages sharing an exact parameter value",
    "total_cost": "sum of controller-owned stage cost weights",
    "max_cost": "largest controller-owned stage cost weight",
    "cost_centroid": "cost-weighted normalized stage position",
    "cost_switches": "adjacent changes in stage cost class",
    "runtime_cost_estimate": "controller-owned additive runtime cost proxy",
    "train_rows": "train-only dataset row count diagnostic",
    "train_width": "train-only dataset width diagnostic",
    "train_missingness": "train-only missingness diagnostic",
    "target_cardinality": "train-only target cardinality diagnostic",
    "rank_diagnostic": "train-only rank diagnostic",
    "condition_diagnostic": "train-only condition diagnostic",
    "resolved_width_min": "minimum controller-resolved representation width",
    "resolved_width_max": "maximum controller-resolved representation width",
    "resolved_width_mean": "mean controller-resolved representation width",
    "resolved_width_range": "range of controller-resolved representation widths",
    "compression_steps": "count of width decreases",
    "expansion_steps": "count of width increases",
    "cumulative_log_width": "sum of log1p resolved representation widths",
    "information_retention_proxy": "minimum width divided by initial width",
    "width_cost_area": "stage-cost-weighted area under the resolved-width trace",
    "bottleneck_depth": "relative drop from pre-bottleneck peak to minimum width",
    "bottleneck_duration": "fraction of stages at or below the geometric midrange",
    "bottleneck_location": "normalized location of the minimum resolved width",
    "recovery_after_compression": "relative recovery after the minimum width",
    "expansion_overshoot": "maximum width beyond the initial/terminal reference",
    "maximum_width_drawdown": "maximum peak-to-later-width drawdown",
    "destructive_pressure_waves": "number of distinct runs of width compression",
    "width_change_points": "direction/magnitude change points in the width trace",
    "leave_one_stage_out_delta": "worst deterministic structural-summary delta after stage deletion",
    "adjacent_swap_delta": "worst positional-cost delta after an adjacent stage swap",
    "duplicate_stage_delta": "worst normalized structural delta after duplicating one stage",
    "head_swap_delta": "distance of the observed head demand from other grammar heads",
    "parameter_default_delta": "alias of grammar-relative parameter default distance",
    "param_default_distance": "mean normalized numeric distance from declared grammar defaults",
    "param_bound_distance": "mean normalized distance to the nearest declared grammar bound",
    "param_constraint_ratio": "largest within-stage ratio between coupled numeric parameters",
    "repeated_param_drift": "mean normalized parameter drift across repeated operators",
    "seed_collision": "number of reused numeric seeds across distinct stages",
    "regularization_capacity_balance": "regularization magnitude relative to effective capacity",
    "missing_default_topology": "fraction of declared optional parameters omitted by the genome",
    "expr_constant_folded_size": "AST size remaining after controller constant-fold approximation",
    "expr_effective_variables": "number of distinct expression variables/column references",
    "expr_degree_proxy": "bounded symbolic polynomial-degree proxy",
    "expr_frequency_proxy": "count of oscillatory expression functions",
    "expr_interval_growth": "bounded interval-growth proxy from expression operations",
    "expr_singularity_risk": "unguarded division/log/sqrt operation count",
    "expr_saturation_risk": "sigmoid/tanh/clip saturation-operation count",
    "expr_guard_coverage": "guard operations divided by risky operations",
    "expr_lipschitz_proxy": "bounded structural Lipschitz proxy",
    "expr_monotonicity_proxy": "signed monotone-operation fraction",
    "expr_symmetry_proxy": "even/absolute operation fraction",
    "expr_branch_balance": "balance of binary expression subtree sizes",
    "expr_variable_interactions": "distinct variable pairs co-occurring in expressions",
    "head_representation_supply": "pre-head width/family supply proxy",
    "head_representation_demand": "grammar/head-parameter demand proxy",
    "head_supply_demand_gap": "absolute log supply-demand mismatch",
    "effective_capacity": "representation supply multiplied by head demand",
    "dataset_effective_rank": "train-only matrix rank normalized by width",
    "dataset_intrinsic_dimension": "train-only intrinsic-dimension diagnostic",
    "dataset_sparsity": "train-only sparsity diagnostic",
    "operator_entropy": "entropy of the operator multiset",
    "transition_entropy": "entropy of adjacent operator transitions",
    "path_complexity": "unique multiscale operator paths normalized by pipeline length",
    "genome_compression_ratio": "zlib description length divided by canonical genome length",
    "long_range_repeat": "repeated non-adjacent operator-pair count",
    "semantic_cacheability": "fraction of repeated exact stages suitable for semantic caching",
    "peak_memory_proxy": "maximum resolved width or controller-owned structural fallback",
    "parallelism_proxy": "fraction of adjacent transforms without a repeated family dependency",
    "ast_path_count": "number of root-to-leaf paths across all expression ASTs",
    "ast_path_max_len": "longest root-to-leaf expression AST path; 0 with no expressions",
    "ast_path_mean_len": "mean root-to-leaf expression AST path length; -1 with no paths",
    "ast_path_ngram": "occurrences of a space-joined node-type n-gram along root-to-leaf paths",
    "ast_repeated_path_fraction": "fraction of root-to-leaf paths repeating an earlier path shape",
    "ast_branch_factor": "mean child count over internal expression AST nodes; 0 with none",
    "cross_stage_param_coherence": "1/(1+population std) of a numeric param across stages; -1 below two values",
    "param_sign": "sign (-1/0/1) of a numeric op param; -1 if absent (shares the ABSENT sentinel)",
    "param_bucket": "1 when a numeric op param is >= the literal edge, 0 below it; -1 if absent",
    "width_bottleneck": "minimum/maximum of the resolved width trace, in (0,1]",
    "width_bottleneck_position": "normalized position of the narrowest resolved width",
    "fourgram": "count of an exact adjacent op quadruple",
    "positional_bigram": "mean normalized start position of an adjacent op pair; -1 if absent",
    "diag": "train-only per-dataset diagnostic by key",
    "add": "a + b", "sub": "a - b", "mul": "a * b",
    "div": "a / b; |b| < 1e-12 yields a non-finite value, which becomes 0.0",
    "log1p": "log(1 + x); x <= -1 becomes 0.0",
    "sqrt": "square root; x < 0 becomes 0.0",
    "abs": "absolute value",
    "min": "minimum of 2 or more", "max": "maximum of 2 or more",
    "clip": "clip(x, lo, hi)",
}


def _variadic(sig: tuple[str, ...]) -> bool:
    return bool(sig) and sig[-1].startswith("*")


def _arity(sig: tuple[str, ...]) -> tuple[int, float]:
    """-> (min_args, max_args); max is inf for a variadic signature."""
    if _variadic(sig):
        return len(sig) - 1, math.inf
    return len(sig), float(len(sig))


# --- Evaluation context --------------------------------------------------------------------------
class _Ctx:
    """Per-genome precomputation, so a primitive is a dict lookup rather than a stage scan.

    Rebuilt per evaluated key. Building it is O(n_stages) over at most 7 stages, which is cheaper
    than caching it would be to invalidate correctly.
    """

    __slots__ = (
        "pg", "diag", "stages", "ops", "n_stages", "n_transforms", "op_count", "fam_count",
        "first_pos", "last_pos", "first_stage", "head", "costs", "cost_classes", "expr_len",
        "n_exprs", "n_col_refs", "expr_text", "ast_nodes", "ast_functions", "ast_variables",
        "ast_constants", "ast_depth", "param_count", "numeric_param_count",
        "categorical_param_count", "params_by_name",
        "ast_paths", "ast_internal_nodes", "ast_child_edges",
    )

    def __init__(self, pg: ParsedGenome, diagnostics: dict | None) -> None:
        stages = getattr(pg, "stages", None) or []
        self.pg = pg
        self.diag = diagnostics or {}
        self.stages = stages
        self.ops = [stage.op for stage in stages]
        self.n_stages = float(len(stages))
        self.n_transforms = float(max(len(stages) - 1, 0))
        self.head = stages[-1] if stages else None

        op_count: dict[str, float] = {}
        fam_count: dict[str, float] = {}
        first_pos: dict[str, float] = {}
        last_pos: dict[str, float] = {}
        first_stage: dict[str, object] = {}
        for s in stages:
            op_count[s.op] = op_count.get(s.op, 0.0) + 1.0
            if s.op not in first_pos:
                first_pos[s.op] = float(s.position_norm)
                first_stage[s.op] = s
            last_pos[s.op] = float(s.position_norm)
            for fam in s.families:
                fam_count[fam] = fam_count.get(fam, 0.0) + 1.0
        self.op_count = op_count
        self.fam_count = fam_count
        self.first_pos = first_pos
        self.last_pos = last_pos
        self.first_stage = first_stage
        self.costs = [COST_WEIGHT.get(s.cost_class, COST_WEIGHT["medium"]) for s in stages]
        self.cost_classes = [s.cost_class for s in stages]

        exprs = getattr(pg, "exprs", None) or []
        self.expr_len = float(sum(len(e) for e in exprs))
        self.n_exprs = float(len(exprs))
        refs: set[int] = set()
        for e in exprs:
            refs.update(col_refs(e))
        self.n_col_refs = float(len(refs))
        self.expr_text = " ".join(exprs).casefold()
        ast_nodes: collections.Counter[str] = collections.Counter()
        ast_functions: collections.Counter[str] = collections.Counter()
        ast_variables: collections.Counter[str] = collections.Counter()
        ast_constants = 0
        ast_depth = 0
        ast_paths: list[tuple[str, ...]] = []
        ast_internal_nodes = 0
        ast_child_edges = 0
        for expression in exprs:
            try:
                tree = ast.parse(expression, mode="eval")
            except (SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                ast_nodes[type(node).__name__.casefold()] += 1
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        ast_functions[node.func.id.casefold()] += 1
                    elif isinstance(node.func, ast.Attribute):
                        ast_functions[node.func.attr.casefold()] += 1
                elif isinstance(node, ast.Name):
                    ast_variables[node.id.casefold()] += 1
                elif isinstance(node, ast.Constant):
                    ast_constants += 1
            stack = [(tree, 1)]
            while stack:
                node, depth = stack.pop()
                ast_depth = max(ast_depth, depth)
                stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
            # Section B5: root-to-leaf path enumeration over the same parsed tree. Paths
            # are tuples of casefolded node-class names, rooted at the expression body.
            path_stack: list[tuple[ast.AST, tuple[str, ...]]] = [
                (tree.body, (type(tree.body).__name__.casefold(),))
            ]
            while path_stack:
                node, path = path_stack.pop()
                children = list(ast.iter_child_nodes(node))
                if children:
                    ast_internal_nodes += 1
                    ast_child_edges += len(children)
                    path_stack.extend(
                        (child, path + (type(child).__name__.casefold(),))
                        for child in children
                    )
                else:
                    ast_paths.append(path)
        self.ast_paths = ast_paths
        self.ast_internal_nodes = ast_internal_nodes
        self.ast_child_edges = ast_child_edges
        self.ast_nodes = ast_nodes
        self.ast_functions = ast_functions
        self.ast_variables = ast_variables
        self.ast_constants = float(ast_constants)
        self.ast_depth = float(ast_depth)
        params_by_name: dict[str, list[tuple[int, str, float | None]]] = collections.defaultdict(list)
        numeric = 0
        categorical = 0
        for stage in stages:
            for name, raw in stage.params.items():
                text = str(raw).strip().strip("\"'")
                try:
                    number = float(text)
                    if not math.isfinite(number):
                        raise ValueError
                except (TypeError, ValueError):
                    number = None
                    categorical += 1
                else:
                    numeric += 1
                params_by_name[name].append((stage.index, text, number))
        self.param_count = float(numeric + categorical)
        self.numeric_param_count = float(numeric)
        self.categorical_param_count = float(categorical)
        self.params_by_name = params_by_name


def _param_value(ctx: _Ctx, op: str, name: str) -> float:
    stage = ctx.first_stage.get(op)
    if stage is None:
        return ABSENT
    raw = stage.params.get(name)
    if raw is None:
        return ABSENT
    try:
        return float(str(raw).strip().strip("\"'"))
    except (TypeError, ValueError):
        # Categorical params (mode="sqrt", kernel="rbf", ...) have no meaningful numeric value.
        # Encoding them by hash would hand a tree a meaningless ordering, so they read as ABSENT.
        return ABSENT


def _cost_of(ctx: _Ctx, index: float) -> float:
    if not math.isfinite(index):
        return 0.0
    i = int(index)
    n = len(ctx.costs)
    if i < 0:
        i += n
    return ctx.costs[i] if 0 <= i < n else 0.0


def _diag(ctx: _Ctx, key: str) -> float:
    value = ctx.diag.get(key)
    if value is None or isinstance(value, bool):
        # nan, not 0.0: a missing diagnostic must surface in n_nonfinite_substituted rather than
        # masquerade as a genuine zero measurement.
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _positions(ctx: _Ctx, op: str) -> list[int]:
    return [index for index, value in enumerate(ctx.ops) if value == op]


def _max_run(ctx: _Ctx, op: str) -> float:
    best = current = 0
    for value in ctx.ops:
        current = current + 1 if value == op else 0
        best = max(best, current)
    return float(best)


def _mean_gap(ctx: _Ctx, op: str) -> float:
    positions = _positions(ctx, op)
    if len(positions) < 2:
        return ABSENT
    return float(sum(b - a for a, b in zip(positions, positions[1:])) / (len(positions) - 1))


def _param_category(ctx: _Ctx, op: str, name: str, expected: str) -> float:
    stage = ctx.first_stage.get(op)
    if stage is None or name not in stage.params:
        return 0.0
    return float(str(stage.params[name]).strip().strip("\"'").casefold() == expected.casefold())


def _param_reuse(ctx: _Ctx, name: str) -> float:
    values = [value for _, value, _ in ctx.params_by_name.get(name, [])]
    return float(max(0, len(values) - len(set(values))))


def _param_dispersion(ctx: _Ctx, name: str) -> float:
    values = [number for _, _, number in ctx.params_by_name.get(name, []) if number is not None]
    if not values:
        return ABSENT
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _adjacent_param_reuse(ctx: _Ctx, name: str) -> float:
    entries = ctx.params_by_name.get(name, [])
    return float(sum(1 for left, right in zip(entries, entries[1:]) if right[0] == left[0] + 1 and right[1] == left[1]))


# --- Section B5 helper metrics --------------------------------------------------------------------
def _param_numeric(ctx: _Ctx, op: str, name: str) -> float | None:
    """Finite numeric value of a param on the first stage carrying the op, else None.

    Distinct from `_param_value`, which folds "absent" into the ABSENT sentinel (-1.0):
    sign/bucket primitives must distinguish a genuine -1.0 parameter from absence."""
    stage = ctx.first_stage.get(op)
    if stage is None:
        return None
    raw = stage.params.get(name)
    if raw is None:
        return None
    try:
        value = float(str(raw).strip().strip("\"'"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _param_sign(ctx: _Ctx, op: str, name: str) -> float:
    value = _param_numeric(ctx, op, name)
    if value is None:
        return ABSENT
    if value > 0.0:
        return 1.0
    return 0.0 if value == 0.0 else -1.0


def _param_bucket(ctx: _Ctx, op: str, name: str, edge: float) -> float:
    value = _param_numeric(ctx, op, name)
    if value is None:
        return ABSENT
    return 1.0 if value >= edge else 0.0


def _cross_stage_param_coherence(ctx: _Ctx, name: str) -> float:
    values = [number for _, _, number in ctx.params_by_name.get(name, []) if number is not None]
    if len(values) < 2:
        return ABSENT
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return 1.0 / (1.0 + std)


def _ast_path_ngram(ctx: _Ctx, tokens: tuple[str, ...]) -> float:
    if not tokens:
        return 0.0
    count = 0
    width = len(tokens)
    for path in ctx.ast_paths:
        for index in range(max(0, len(path) - width + 1)):
            if path[index:index + width] == tokens:
                count += 1
    return float(count)


def _positional_bigram(ctx: _Ctx, left: str, right: str) -> float:
    positions = [
        index
        for index, (a, b) in enumerate(zip(ctx.ops, ctx.ops[1:]))
        if a == left and b == right
    ]
    if not positions:
        return ABSENT
    denominator = max(1, len(ctx.ops) - 1)
    return float(sum(index / denominator for index in positions) / len(positions))


def _width_bottleneck(ctx: _Ctx) -> float:
    values = _width_trace(ctx)
    if not values:
        return math.nan
    top = max(values)
    if top <= 0.0:
        return math.nan
    return min(values) / top


def _width_bottleneck_position(ctx: _Ctx) -> float:
    values = _width_trace(ctx)
    if not values:
        return math.nan
    if len(values) == 1:
        return 0.0
    return float(values.index(min(values))) / (len(values) - 1)


def _cost_centroid(ctx: _Ctx) -> float:
    total = sum(ctx.costs)
    if not ctx.costs or total <= 0:
        return 0.0
    denominator = max(1, len(ctx.costs) - 1)
    return sum((index / denominator) * cost for index, cost in enumerate(ctx.costs)) / total


def _width_trace(ctx: _Ctx) -> list[float]:
    raw = ctx.diag.get("resolved_width_trace")
    if not isinstance(raw, (list, tuple)):
        initial = _fixed_diag(ctx, "train_width")
        if not math.isfinite(initial) or initial <= 0:
            return []
        inferred = [initial]
        width = initial
        for stage in ctx.stages[:-1]:
            candidates = []
            for key in ("n", "k", "topk", "bins", "knots", "m", "leaves"):
                try:
                    value = float(str(stage.params.get(key, "")).strip().strip("\"'"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    candidates.append(value)
            if candidates:
                resolved = max(candidates)
                append = str(stage.params.get("append", "false")).casefold() == "true"
                width = min(4096.0, width + resolved if append else resolved)
            inferred.append(width)
        return inferred
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _entropy(values) -> float:
    counts = collections.Counter(values)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _width_metric(ctx: _Ctx, name: str) -> float:
    values = _width_trace(ctx)
    if not values:
        return math.nan
    positive = [max(0.0, value) for value in values]
    maximum = max(positive)
    minimum = min(positive)
    if name == "cumulative_log_width":
        return sum(math.log1p(value) for value in positive)
    if name == "information_retention_proxy":
        return minimum / max(positive[0], 1.0)
    if name == "width_cost_area":
        costs = ctx.costs or [1.0]
        return sum(value * costs[min(index, len(costs) - 1)] for index, value in enumerate(positive))
    location = positive.index(minimum)
    peak_before = max(positive[: location + 1])
    if name == "bottleneck_depth":
        return (peak_before - minimum) / max(peak_before, 1.0)
    if name == "bottleneck_duration":
        threshold = math.sqrt(max(minimum, 1e-12) * max(maximum, 1e-12))
        return sum(value <= threshold for value in positive) / len(positive)
    if name == "bottleneck_location":
        return location / max(1, len(positive) - 1)
    if name == "recovery_after_compression":
        return (max(positive[location:]) - minimum) / max(peak_before, 1.0)
    if name == "expansion_overshoot":
        reference = max(positive[0], positive[-1], 1.0)
        return max(0.0, maximum - reference) / reference
    if name == "maximum_width_drawdown":
        peak = positive[0]
        drawdown = 0.0
        for value in positive[1:]:
            peak = max(peak, value)
            drawdown = max(drawdown, (peak - value) / max(peak, 1.0))
        return drawdown
    deltas = [right - left for left, right in zip(positive, positive[1:])]
    if name == "destructive_pressure_waves":
        waves = 0
        compressing = False
        for delta in deltas:
            if delta < 0 and not compressing:
                waves += 1
            compressing = delta < 0
        return float(waves)
    return float(
        sum(
            left * right < 0 or abs(right - left) > max(1.0, abs(left), abs(right))
            for left, right in zip(deltas, deltas[1:])
        )
    )


def _grammar_ops(ctx: _Ctx):
    try:
        grammar = grammar_for(ctx.pg.engine)
    except Exception:
        return {}
    return {**dict(grammar.transforms), **dict(grammar.heads)}


def _numeric_params(ctx: _Ctx):
    ops = _grammar_ops(ctx)
    for stage in ctx.stages:
        op_spec = ops.get(stage.op)
        for name, raw in stage.params.items():
            try:
                value = float(str(raw).strip().strip("\"'"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            param_spec = op_spec.param(name) if op_spec is not None else None
            bins = []
            if param_spec is not None:
                for item in param_spec.bins:
                    try:
                        number = float(item)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        bins.append(number)
            yield stage, name, value, param_spec, bins


def _param_metric(ctx: _Ctx, name: str) -> float:
    entries = list(_numeric_params(ctx))
    if name in {"param_default_distance", "parameter_default_delta"}:
        distances = []
        for _stage, _key, value, spec, bins in entries:
            if spec is None or spec.default is None:
                continue
            try:
                default = float(spec.default)
            except (TypeError, ValueError):
                continue
            span = max(bins) - min(bins) if len(bins) >= 2 else max(abs(default), 1.0)
            distances.append(abs(value - default) / max(span, 1e-12))
        return sum(distances) / len(distances) if distances else 0.0
    if name == "param_bound_distance":
        distances = []
        for _stage, _key, value, _spec, bins in entries:
            if len(bins) >= 2:
                lo, hi = min(bins), max(bins)
                distances.append(min(abs(value - lo), abs(hi - value)) / max(hi - lo, 1e-12))
        return sum(distances) / len(distances) if distances else 0.0
    if name == "param_constraint_ratio":
        by_stage: dict[int, list[float]] = collections.defaultdict(list)
        for stage, _key, value, _spec, _bins in entries:
            if abs(value) > 1e-12:
                by_stage[stage.index].append(abs(value))
        ratios = [max(values) / min(values) for values in by_stage.values() if len(values) >= 2]
        return max(ratios, default=0.0)
    if name == "repeated_param_drift":
        by_key: dict[tuple[str, str], list[tuple[float, list[float]]]] = collections.defaultdict(list)
        for stage, key, value, _spec, bins in entries:
            by_key[(stage.op, key)].append((value, bins))
        drifts = []
        for values in by_key.values():
            if len(values) < 2:
                continue
            raw = [value for value, _ in values]
            bins = values[0][1]
            scale = max(bins) - min(bins) if len(bins) >= 2 else max(max(map(abs, raw)), 1.0)
            drifts.extend(abs(right - left) / max(scale, 1e-12) for left, right in zip(raw, raw[1:]))
        return sum(drifts) / len(drifts) if drifts else 0.0
    if name == "seed_collision":
        seeds = [str(stage.params["seed"]) for stage in ctx.stages if "seed" in stage.params]
        return float(max(0, len(seeds) - len(set(seeds))))
    if name == "missing_default_topology":
        ops = _grammar_ops(ctx)
        declared = missing = 0
        for stage in ctx.stages:
            spec = ops.get(stage.op)
            for param in getattr(spec, "params", ()) or ():
                if not param.required:
                    declared += 1
                    missing += param.name not in stage.params
        return missing / declared if declared else 0.0
    supply = _head_supply(ctx)
    regularizers = []
    for stage, key, value, _spec, _bins in entries:
        if stage is ctx.head and key.casefold() in {"alpha", "lambda", "l1_ratio", "prior"}:
            regularizers.append(abs(value))
    return math.log1p(sum(regularizers)) / max(math.log1p(supply), 1e-12)


def _structural_summary(ops: list[str], costs: list[float]) -> tuple[float, ...]:
    n = len(ops)
    centroid = (
        sum(index * cost for index, cost in enumerate(costs)) / max(sum(costs) * max(n - 1, 1), 1e-12)
        if costs else 0.0
    )
    return float(n), float(sum(costs)), centroid, _entropy(ops), _entropy(zip(ops, ops[1:]))


def _summary_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(abs(a - b) / max(1.0, abs(a), abs(b)) for a, b in zip(left, right))


def _counterfactual_metric(ctx: _Ctx, name: str) -> float:
    if name == "parameter_default_delta":
        return _param_metric(ctx, name)
    if not ctx.ops:
        return 0.0
    baseline = _structural_summary(ctx.ops, ctx.costs)
    if name == "leave_one_stage_out_delta":
        return max(
            _summary_delta(
                baseline,
                _structural_summary(ctx.ops[:index] + ctx.ops[index + 1:], ctx.costs[:index] + ctx.costs[index + 1:]),
            )
            for index in range(len(ctx.ops))
        )
    if name == "adjacent_swap_delta":
        deltas = []
        for index in range(len(ctx.ops) - 1):
            ops, costs = list(ctx.ops), list(ctx.costs)
            ops[index], ops[index + 1] = ops[index + 1], ops[index]
            costs[index], costs[index + 1] = costs[index + 1], costs[index]
            deltas.append(_summary_delta(baseline, _structural_summary(ops, costs)))
        return max(deltas, default=0.0)
    if name == "duplicate_stage_delta":
        return max(
            _summary_delta(
                baseline,
                _structural_summary(
                    ctx.ops[: index + 1] + [ctx.ops[index]] + ctx.ops[index + 1:],
                    ctx.costs[: index + 1] + [ctx.costs[index]] + ctx.costs[index + 1:],
                ),
            )
            for index in range(len(ctx.ops))
        )
    observed = _head_demand(ctx)
    try:
        grammar = grammar_for(ctx.pg.engine)
        other = [COST_WEIGHT.get(spec.cost_class, 1.0) for spec in grammar.heads.values()]
    except Exception:
        other = []
    return min((abs(observed - value) for value in other if value != observed), default=0.0)


def _expression_trees(ctx: _Ctx):
    for expression in getattr(ctx.pg, "exprs", ()) or ():
        try:
            yield ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError):
            continue


def _node_size(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _constant_folded_size(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return 1
    children = list(ast.iter_child_nodes(node))
    if children and isinstance(node, (ast.BinOp, ast.UnaryOp)) and all(
        isinstance(child, (ast.Constant, ast.operator, ast.unaryop)) for child in children
    ):
        return 1
    return 1 + sum(_constant_folded_size(child) for child in children)


def _variable_names(node: ast.AST) -> set[str]:
    names = {item.id.casefold() for item in ast.walk(node) if isinstance(item, ast.Name)}
    names.update(f"col_{value}" for value in col_refs(ast.unparse(node) if hasattr(ast, "unparse") else ""))
    return names


def _degree(node: ast.AST) -> int:
    if isinstance(node, (ast.Constant, ast.Name, ast.Call)):
        return 0 if isinstance(node, ast.Constant) else 1
    if isinstance(node, ast.BinOp):
        left, right = _degree(node.left), _degree(node.right)
        if isinstance(node.op, ast.Mult):
            return min(16, left + right)
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            try:
                return min(16, int(abs(float(node.right.value))) * left)
            except (TypeError, ValueError):
                return 16
        return max(left, right)
    return max((_degree(child) for child in ast.iter_child_nodes(node)), default=0)


def _expression_metric(ctx: _Ctx, name: str) -> float:
    trees = list(_expression_trees(ctx))
    if not trees:
        return 0.0
    nodes = [node for tree in trees for node in ast.walk(tree)]
    functions = [
        node.func.id.casefold()
        for node in nodes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    if name == "expr_constant_folded_size":
        return float(sum(_constant_folded_size(tree) for tree in trees))
    if name == "expr_effective_variables":
        return float(len(set().union(*(_variable_names(tree) for tree in trees))))
    if name == "expr_degree_proxy":
        return float(max(_degree(tree) for tree in trees))
    if name == "expr_frequency_proxy":
        return float(sum(value in {"sin", "cos", "tan", "fft", "rfft"} for value in functions))
    risky = sum(isinstance(node, ast.Div) for node in nodes) + sum(
        value in {"log", "log1p", "sqrt", "reciprocal"} for value in functions
    )
    guards = sum(value in {"abs", "clip", "min", "max", "where"} for value in functions)
    if name == "expr_interval_growth":
        return float(min(1e6, sum(2 if isinstance(node, (ast.Mult, ast.Pow)) else 1 for node in nodes)))
    if name == "expr_singularity_risk":
        return float(max(0, risky - guards))
    if name == "expr_saturation_risk":
        return float(sum(value in {"sigmoid", "tanh", "clip", "step"} for value in functions))
    if name == "expr_guard_coverage":
        return guards / max(1, risky)
    if name == "expr_lipschitz_proxy":
        return float(min(1e6, math.exp(min(12.0, 0.15 * len(nodes) + 0.5 * risky))))
    if name == "expr_monotonicity_proxy":
        monotone = sum(value in {"log", "log1p", "sqrt", "sigmoid", "tanh"} for value in functions)
        sign_flips = sum(isinstance(node, (ast.Sub, ast.USub)) for node in nodes)
        return (monotone - sign_flips) / max(1, monotone + sign_flips)
    if name == "expr_symmetry_proxy":
        symmetric = sum(value in {"abs", "cos"} for value in functions)
        symmetric += sum(isinstance(node, ast.Pow) for node in nodes)
        return symmetric / max(1, len(nodes))
    if name == "expr_branch_balance":
        balances = []
        for node in nodes:
            if isinstance(node, ast.BinOp):
                left, right = _node_size(node.left), _node_size(node.right)
                balances.append(1.0 - abs(left - right) / max(left, right))
        return sum(balances) / len(balances) if balances else 1.0
    pairs: set[tuple[str, str]] = set()
    for tree in trees:
        variables = sorted(_variable_names(tree))
        pairs.update((left, right) for i, left in enumerate(variables) for right in variables[i + 1:])
    return float(len(pairs))


def _head_supply(ctx: _Ctx) -> float:
    widths = _width_trace(ctx)
    if widths:
        return max(1.0, widths[-2] if len(widths) >= 2 else widths[-1])
    families = sum(len(stage.families) for stage in ctx.stages[:-1])
    return max(1.0, ctx.n_transforms + families)


def _head_demand(ctx: _Ctx) -> float:
    if ctx.head is None:
        return 0.0
    demand = COST_WEIGHT.get(ctx.head.cost_class, COST_WEIGHT["medium"])
    numeric = 0.0
    for raw in ctx.head.params.values():
        try:
            value = abs(float(str(raw).strip().strip("\"'")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            numeric += math.log1p(value)
    return max(1.0, demand * (1.0 + numeric))


def _r4_primitive(ctx: _Ctx, name: str) -> float:
    if name in _WIDTH_PRIMITIVES:
        return _width_metric(ctx, name)
    if name in _PARAM_PRIMITIVES:
        return _param_metric(ctx, name)
    if name in _COUNTERFACTUAL_PRIMITIVES:
        return _counterfactual_metric(ctx, name)
    if name in _EXPRESSION_PRIMITIVES:
        return _expression_metric(ctx, name)
    if name == "head_representation_supply":
        return _head_supply(ctx)
    if name == "head_representation_demand":
        return _head_demand(ctx)
    if name == "head_supply_demand_gap":
        return abs(math.log1p(_head_supply(ctx)) - math.log1p(_head_demand(ctx)))
    if name == "effective_capacity":
        return _head_supply(ctx) * _head_demand(ctx)
    if name == "dataset_effective_rank":
        return _fixed_diag(ctx, "rank_diagnostic") / max(_fixed_diag(ctx, "train_width"), 1.0)
    if name == "dataset_intrinsic_dimension":
        for key in ("intrinsic_dimension", "effective_dimension"):
            value = _diag(ctx, key)
            if math.isfinite(value):
                return value
        return math.nan
    if name == "dataset_sparsity":
        for key in ("sparsity", "zero_fraction"):
            value = _diag(ctx, key)
            if math.isfinite(value):
                return value
        return math.nan
    if name == "operator_entropy":
        return _entropy(ctx.ops)
    if name == "transition_entropy":
        return _entropy(zip(ctx.ops, ctx.ops[1:]))
    if name == "path_complexity":
        paths = set(zip(ctx.ops, ctx.ops[1:])) | set(zip(ctx.ops, ctx.ops[1:], ctx.ops[2:]))
        return len(paths) / max(1, len(ctx.ops))
    if name == "genome_compression_ratio":
        raw = ctx.pg.text.encode()
        return len(zlib.compress(raw, level=9)) / max(1, len(raw))
    if name == "long_range_repeat":
        return float(sum(ctx.ops[i] == ctx.ops[j] for i in range(len(ctx.ops)) for j in range(i + 2, len(ctx.ops))))
    if name == "semantic_cacheability":
        stages = [(stage.op, tuple(sorted(stage.params.items()))) for stage in ctx.stages]
        return (len(stages) - len(set(stages))) / max(1, len(stages))
    if name == "peak_memory_proxy":
        widths = _width_trace(ctx)
        return max(widths) if widths else max(ctx.costs, default=0.0) * max(1.0, ctx.n_stages)
    independent = 0
    for left, right in zip(ctx.stages, ctx.stages[1:]):
        independent += not set(left.families).intersection(right.families)
    return independent / max(1, len(ctx.stages) - 1)


_WIDTH_PRIMITIVES = frozenset(
    {
        "cumulative_log_width", "information_retention_proxy", "width_cost_area",
        "bottleneck_depth", "bottleneck_duration", "bottleneck_location",
        "recovery_after_compression", "expansion_overshoot", "maximum_width_drawdown",
        "destructive_pressure_waves", "width_change_points",
    }
)
_PARAM_PRIMITIVES = frozenset(
    {
        "param_default_distance", "param_bound_distance", "param_constraint_ratio",
        "repeated_param_drift", "seed_collision", "regularization_capacity_balance",
        "missing_default_topology",
    }
)
_COUNTERFACTUAL_PRIMITIVES = frozenset(
    {
        "leave_one_stage_out_delta", "adjacent_swap_delta", "duplicate_stage_delta",
        "head_swap_delta", "parameter_default_delta",
    }
)
_EXPRESSION_PRIMITIVES = frozenset(
    {
        "expr_constant_folded_size", "expr_effective_variables", "expr_degree_proxy",
        "expr_frequency_proxy", "expr_interval_growth", "expr_singularity_risk",
        "expr_saturation_risk", "expr_guard_coverage", "expr_lipschitz_proxy",
        "expr_monotonicity_proxy", "expr_symmetry_proxy", "expr_branch_balance",
        "expr_variable_interactions",
    }
)
_R4_DIAGNOSTIC_PRIMITIVES = _WIDTH_PRIMITIVES | frozenset(
    {
        "regularization_capacity_balance", "head_representation_supply",
        "head_supply_demand_gap", "effective_capacity", "dataset_effective_rank",
        "dataset_intrinsic_dimension", "dataset_sparsity", "peak_memory_proxy",
    }
)
_R4_PRIMITIVES = (
    _WIDTH_PRIMITIVES | _PARAM_PRIMITIVES | _COUNTERFACTUAL_PRIMITIVES |
    _EXPRESSION_PRIMITIVES | frozenset(
        {
            "head_representation_supply", "head_representation_demand", "head_supply_demand_gap",
            "effective_capacity", "dataset_effective_rank", "dataset_intrinsic_dimension",
            "dataset_sparsity", "operator_entropy", "transition_entropy", "path_complexity",
            "genome_compression_ratio", "long_range_repeat", "semantic_cacheability",
            "peak_memory_proxy", "parallelism_proxy",
        }
    )
)


def _fixed_diag(ctx: _Ctx, name: str) -> float:
    aliases = {
        "train_rows": ("train_rows", "rows"),
        "train_width": ("train_width", "columns", "width"),
        "train_missingness": ("train_missingness", "missingness"),
        "target_cardinality": ("target_cardinality",),
        "rank_diagnostic": ("rank", "matrix_rank"),
        "condition_diagnostic": ("condition", "condition_number"),
    }
    for key in aliases[name]:
        value = _diag(ctx, key)
        if math.isfinite(value):
            return value
    return math.nan


def _safe_log1p(x: float) -> float:
    try:
        return math.log1p(x)
    except (ValueError, OverflowError):
        return math.nan


def _safe_sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0.0 else math.nan


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > _DIV_EPS else math.nan


def _safe_mul(a: float, b: float) -> float:
    try:
        return a * b
    except OverflowError:
        return math.nan


# --- Validation ----------------------------------------------------------------------------------
def _known_ops(engine: str | None) -> frozenset[str] | None:
    """Op names the grammar declares, or None when the engine is unknown/unavailable."""
    if not engine:
        return None
    try:
        g = grammar_for(engine)
    except Exception:
        return None
    return frozenset(dict(g.transforms)) | frozenset(dict(g.heads))


def _validate_node(node, path: str, engine_ops, depth: int, budget: list[int],
                   diag_keys: set[str], errors: list[str]) -> None:
    if depth > MAX_EXPR_DEPTH:
        errors.append(f"{path}: expression nests deeper than {MAX_EXPR_DEPTH}")
        return
    budget[0] += 1
    if budget[0] > MAX_EXPR_NODES:
        errors.append(f"{path}: expression exceeds {MAX_EXPR_NODES} nodes")
        return

    if isinstance(node, bool):
        errors.append(f"{path}: booleans are not expressions; use 1 or 0")
        return
    if isinstance(node, (int, float)):
        value = float(node)
        if not math.isfinite(value):
            errors.append(f"{path}: literal {node!r} is not finite")
        elif abs(value) > MAX_ABS_LITERAL:
            errors.append(f"{path}: literal {node!r} exceeds {MAX_ABS_LITERAL:g}")
        return
    if not isinstance(node, list) or not node:
        errors.append(
            f"{path}: expected a number or a [primitive, ...args] list, got {type(node).__name__}"
        )
        return

    head = node[0]
    if not isinstance(head, str):
        errors.append(f"{path}: first element must be a primitive name, got {head!r}")
        return
    prim = _ALIASES.get(head, head)
    sig = _SIGNATURES.get(prim)
    if sig is None:
        errors.append(f"{path}: unknown primitive {head!r}; allowed: {', '.join(PRIMITIVES)}")
        return

    args = node[1:]
    lo, hi = _arity(sig)
    if not (lo <= len(args) <= hi):
        want = f"{lo}+" if hi == math.inf else str(lo)
        errors.append(f"{path}: {prim!r} takes {want} argument(s), got {len(args)}")
        return

    if _variadic(sig):
        fixed = list(sig[:-1])
        kinds = fixed + [sig[-1][1:]] * (len(args) - len(fixed))
    else:
        kinds = list(sig)
    for i, (kind, arg) in enumerate(zip(kinds, args)):
        sub = f"{path}.{prim}[{i}]"
        if kind == "expr":
            _validate_node(arg, sub, engine_ops, depth + 1, budget, diag_keys, errors)
            continue
        if not isinstance(arg, str) or not arg:
            errors.append(f"{sub}: {prim!r} argument {i} must be a non-empty string ({kind})")
            continue
        if kind == "op" and engine_ops is not None and arg not in engine_ops:
            errors.append(f"{sub}: unknown op {arg!r} for this engine")
        elif kind == "family" and arg not in _FAMILIES:
            errors.append(f"{sub}: unknown family {arg!r}; allowed: {', '.join(FAMILY_LIST_V3)}")
        elif kind == "num":
            try:
                edge = float(arg)
            except (TypeError, ValueError):
                errors.append(f"{sub}: {prim!r} argument {i} must be a numeric literal string")
            else:
                if not math.isfinite(edge) or abs(edge) > MAX_ABS_LITERAL:
                    errors.append(
                        f"{sub}: numeric literal {arg!r} must be finite and within {MAX_ABS_LITERAL:g}"
                    )
        elif kind == "key":
            diag_keys.add(arg)


def validate_spec(spec, engine: str | None = None) -> list[str]:
    """-> a list of human-readable problems; empty means the spec is safe to compile.

    Returns rather than raises because the controller feeds these back to a role as a repair
    prompt (`MAX_ATTEMPTS_PER_ROLE = 2`), and one message per problem is far more useful than the
    first exception. The strings are structural only -- they never contain a score, so they are
    safe to show an agent alongside its categorical feedback.

    `engine` is optional: when supplied, op-name arguments are checked against that grammar, which
    turns a permanently-zero column into a repairable error. Family names are always checked.
    """
    errors: list[str] = []
    if not isinstance(spec, dict):
        return [f"spec must be an object, got {type(spec).__name__}"]

    unknown = sorted(set(spec) - _ALLOWED_TOP_KEYS)
    if unknown:
        errors.append(f"unknown top-level key(s) {unknown}; allowed: {sorted(_ALLOWED_TOP_KEYS)}")
    missing = sorted(_REQUIRED_TOP_KEYS - set(spec))
    if missing:
        errors.append(f"missing required key(s) {missing}")

    family_id = spec.get("family_id")
    if family_id is not None and not (
        isinstance(family_id, str) and _FAMILY_ID_RE.match(family_id)
    ):
        errors.append(f"family_id {family_id!r} is not a valid controller-minted id")

    engine = engine or (spec.get("engine") if isinstance(spec.get("engine"), str) else None)
    engine_ops = _known_ops(engine)

    outputs = spec.get("outputs")
    if not isinstance(outputs, list):
        errors.append("outputs must be a list")
        return errors
    if not (MIN_OUTPUTS <= len(outputs) <= MAX_OUTPUTS):
        errors.append(
            f"outputs must hold {MIN_OUTPUTS}-{MAX_OUTPUTS} entries, got {len(outputs)}"
        )

    seen: set[str] = set()
    diag_keys: set[str] = set()
    for i, out in enumerate(outputs):
        path = f"outputs[{i}]"
        if not isinstance(out, dict):
            errors.append(f"{path}: must be an object with 'name' and 'expr'")
            continue
        extra = sorted(set(out) - {"name", "expr", "rationale"})
        if extra:
            errors.append(f"{path}: unknown key(s) {extra}")
        name = out.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            errors.append(f"{path}: name {name!r} must match {_NAME_RE.pattern}")
        elif name in seen:
            errors.append(f"{path}: duplicate output name {name!r}")
        else:
            seen.add(name)
        if "expr" not in out:
            errors.append(f"{path}: missing 'expr'")
        else:
            _validate_node(out["expr"], path, engine_ops, 0, [0], diag_keys, errors)

    # Leakage boundary. Output names become column names, and diagnostics keys name real inputs;
    # both must clear the forbidden-substring list before anything is compiled.
    for label, values in (("output name", sorted(seen)), ("diagnostics key", sorted(diag_keys))):
        try:
            assert_no_forbidden_inputs(values)
        except Exception as exc:
            errors.append(f"{label} rejected: {exc}")
    return errors


# --- Compilation ---------------------------------------------------------------------------------
def _compile_node(node, prim_cache: dict):
    """Turn one validated expression node into a closure of `_Ctx -> float`.

    Compiling to closures rather than re-walking the JSON per genome matters at corpus scale:
    ~34k genomes x up to 25 outputs x up to 200 nodes is the difference between seconds and
    minutes. There is still no `eval` anywhere -- every closure comes from this fixed table.
    """
    if isinstance(node, (int, float)):
        value = float(node)
        return lambda ctx: value

    prim = _ALIASES.get(node[0], node[0])
    args = node[1:]

    if prim == "n_stages":
        return lambda ctx: ctx.n_stages
    if prim == "n_transforms":
        return lambda ctx: ctx.n_transforms
    if prim == "expr_len":
        return lambda ctx: ctx.expr_len
    if prim == "n_exprs":
        return lambda ctx: ctx.n_exprs
    if prim == "col_refs":
        return lambda ctx: ctx.n_col_refs
    if prim == "family_switches":
        return lambda ctx: float(
            sum(
                1
                for left, right in zip(ctx.stages, ctx.stages[1:])
                if not set(left.families).intersection(right.families)
            )
        )
    if prim == "ast_constant_count":
        return lambda ctx: ctx.ast_constants
    if prim == "ast_depth":
        return lambda ctx: ctx.ast_depth
    if prim == "param_count":
        return lambda ctx: ctx.param_count
    if prim == "numeric_param_count":
        return lambda ctx: ctx.numeric_param_count
    if prim == "categorical_param_count":
        return lambda ctx: ctx.categorical_param_count
    if prim == "total_cost":
        return lambda ctx: float(sum(ctx.costs))
    if prim == "max_cost":
        return lambda ctx: float(max(ctx.costs, default=0.0))
    if prim == "cost_centroid":
        return _cost_centroid
    if prim == "cost_switches":
        return lambda ctx: float(sum(left != right for left, right in zip(ctx.cost_classes, ctx.cost_classes[1:])))
    if prim == "runtime_cost_estimate":
        return lambda ctx: float(sum(ctx.costs) * max(1.0, ctx.n_stages))
    if prim in {
        "train_rows", "train_width", "train_missingness", "target_cardinality",
        "rank_diagnostic", "condition_diagnostic",
    }:
        prim_cache["uses_diagnostics"] = True
        return lambda ctx, name=prim: _fixed_diag(ctx, name)
    if prim in {
        "resolved_width_min", "resolved_width_max", "resolved_width_mean",
        "resolved_width_range", "compression_steps", "expansion_steps",
    }:
        prim_cache["uses_diagnostics"] = True

        def width_value(ctx, name=prim):
            values = _width_trace(ctx)
            if not values:
                return math.nan
            if name == "resolved_width_min":
                return min(values)
            if name == "resolved_width_max":
                return max(values)
            if name == "resolved_width_mean":
                return sum(values) / len(values)
            if name == "resolved_width_range":
                return max(values) - min(values)
            if name == "compression_steps":
                return float(sum(right < left for left, right in zip(values, values[1:])))
            return float(sum(right > left for left, right in zip(values, values[1:])))

        return width_value
    if prim in _R4_PRIMITIVES:
        if prim in _R4_DIAGNOSTIC_PRIMITIVES:
            prim_cache["uses_diagnostics"] = True
        return lambda ctx, name=prim: _r4_primitive(ctx, name)

    # Section B5: AST path geometry, cross-stage parameter geometry, width bottleneck,
    # and higher-order operator n-grams.
    if prim == "ast_path_count":
        return lambda ctx: float(len(ctx.ast_paths))
    if prim == "ast_path_max_len":
        return lambda ctx: float(max((len(path) for path in ctx.ast_paths), default=0))
    if prim == "ast_path_mean_len":
        return lambda ctx: (
            float(sum(len(path) for path in ctx.ast_paths) / len(ctx.ast_paths))
            if ctx.ast_paths else ABSENT
        )
    if prim == "ast_repeated_path_fraction":
        return lambda ctx: (
            float((len(ctx.ast_paths) - len(set(ctx.ast_paths))) / len(ctx.ast_paths))
            if ctx.ast_paths else 0.0
        )
    if prim == "ast_branch_factor":
        return lambda ctx: (
            float(ctx.ast_child_edges / ctx.ast_internal_nodes)
            if ctx.ast_internal_nodes else 0.0
        )
    if prim == "ast_path_ngram":
        tokens = tuple(re.findall(r"[a-z_][a-z0-9_]*", args[0].casefold()))
        return lambda ctx: _ast_path_ngram(ctx, tokens)
    if prim == "cross_stage_param_coherence":
        name = args[0]
        return lambda ctx: _cross_stage_param_coherence(ctx, name)
    if prim == "param_sign":
        op, name = args
        return lambda ctx: _param_sign(ctx, op, name)
    if prim == "param_bucket":
        op, name = args[0], args[1]
        edge = float(args[2])
        return lambda ctx: _param_bucket(ctx, op, name, edge)
    if prim in {"width_bottleneck", "width_bottleneck_position"}:
        prim_cache["uses_diagnostics"] = True
        return _width_bottleneck if prim == "width_bottleneck" else _width_bottleneck_position
    if prim == "fourgram":
        first, second, third, fourth = args
        return lambda ctx: float(
            sum(
                a == first and b == second and c == third and d == fourth
                for a, b, c, d in zip(ctx.ops, ctx.ops[1:], ctx.ops[2:], ctx.ops[3:])
            )
        )
    if prim == "positional_bigram":
        left, right = args
        return lambda ctx: _positional_bigram(ctx, left, right)


    if prim == "position_norm":
        op = args[0]
        return lambda ctx: ctx.first_pos.get(op, ABSENT)
    if prim == "last_position_norm":
        op = args[0]
        return lambda ctx: ctx.last_pos.get(op, ABSENT)
    if prim == "max_run":
        op = args[0]
        return lambda ctx: _max_run(ctx, op)
    if prim == "repetitions":
        op = args[0]
        return lambda ctx: max(0.0, ctx.op_count.get(op, 0.0) - 1.0)
    if prim == "mean_gap":
        op = args[0]
        return lambda ctx: _mean_gap(ctx, op)
    if prim == "transition":
        left, right = args
        return lambda ctx: float(sum(a == left and b == right for a, b in zip(ctx.ops, ctx.ops[1:])))
    if prim == "trigram":
        first, second, third = args
        return lambda ctx: float(
            sum(
                a == first and b == second and c == third
                for a, b, c in zip(ctx.ops, ctx.ops[1:], ctx.ops[2:])
            )
        )
    if prim == "family_transition":
        left, right = args
        return lambda ctx: float(
            sum(
                left in a.families and right in b.families
                for a, b in zip(ctx.stages, ctx.stages[1:])
            )
        )
    if prim == "head_distance":
        op = args[0]
        return lambda ctx: float(len(ctx.ops) - 1 - _positions(ctx, op)[-1]) if op in ctx.op_count else ABSENT
    if prim == "pre_head_count_family":
        family = args[0]
        return lambda ctx: float(sum(family in stage.families for stage in ctx.stages[:-1]))
    if prim == "head_input_family":
        family = args[0]
        return lambda ctx: float(len(ctx.stages) >= 2 and family in ctx.stages[-2].families)
    if prim == "count":
        op = args[0]
        return lambda ctx: ctx.op_count.get(op, 0.0)
    if prim == "has_op":
        op = args[0]
        return lambda ctx: 1.0 if op in ctx.op_count else 0.0
    if prim == "count_family":
        fam = args[0]
        return lambda ctx: ctx.fam_count.get(fam, 0.0)
    if prim == "head_op":
        op = args[0]
        return lambda ctx: 1.0 if ctx.head is not None and ctx.head.op == op else 0.0
    if prim == "head_family":
        fam = args[0]
        return lambda ctx: 1.0 if ctx.head is not None and fam in ctx.head.families else 0.0
    if prim == "param":
        op, name = args[0], args[1]
        return lambda ctx: _param_value(ctx, op, name)
    if prim == "param_missing":
        op, name = args
        return lambda ctx: float(ctx.first_stage.get(op) is None or name not in ctx.first_stage[op].params)
    if prim == "param_category":
        op, name, expected = args
        return lambda ctx: _param_category(ctx, op, name, expected)
    if prim == "param_reuse":
        name = args[0]
        return lambda ctx: _param_reuse(ctx, name)
    if prim == "param_dispersion":
        name = args[0]
        return lambda ctx: _param_dispersion(ctx, name)
    if prim == "adjacent_param_reuse":
        name = args[0]
        return lambda ctx: _adjacent_param_reuse(ctx, name)
    if prim == "ast_node_count":
        name = args[0].casefold()
        return lambda ctx: float(ctx.ast_nodes.get(name, 0))
    if prim == "ast_function_count":
        name = args[0].casefold()
        return lambda ctx: float(ctx.ast_functions.get(name, 0))
    if prim == "ast_variable_count":
        name = args[0].casefold()
        return lambda ctx: float(ctx.ast_variables.get(name, 0))
    if prim == "expr_token_ngram":
        ngram = tuple(re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", args[0].casefold()))
        if not ngram:
            return lambda ctx: 0.0
        return lambda ctx: float(
            sum(
                tuple(tokens[index:index + len(ngram)]) == ngram
                for tokens in [re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", ctx.expr_text)]
                for index in range(max(0, len(tokens) - len(ngram) + 1))
            )
        )
    if prim == "expr_char_ngram":
        ngram = args[0].casefold()
        return lambda ctx: float(sum(ctx.expr_text.startswith(ngram, index) for index in range(len(ctx.expr_text))))
    if prim == "diag":
        key = args[0]
        prim_cache["uses_diagnostics"] = True
        return lambda ctx: _diag(ctx, key)

    if prim == "cost_of":
        inner = _compile_node(args[0], prim_cache)
        return lambda ctx: _cost_of(ctx, inner(ctx))

    sub = [_compile_node(a, prim_cache) for a in args]
    if prim == "add":
        a, b = sub
        return lambda ctx: a(ctx) + b(ctx)
    if prim == "sub":
        a, b = sub
        return lambda ctx: a(ctx) - b(ctx)
    if prim == "mul":
        a, b = sub
        return lambda ctx: _safe_mul(a(ctx), b(ctx))
    if prim == "div":
        a, b = sub
        return lambda ctx: _safe_div(a(ctx), b(ctx))
    if prim == "log1p":
        a, = sub
        return lambda ctx: _safe_log1p(a(ctx))
    if prim == "sqrt":
        a, = sub
        return lambda ctx: _safe_sqrt(a(ctx))
    if prim == "abs":
        a, = sub
        return lambda ctx: abs(a(ctx))
    if prim == "min":
        return lambda ctx: min(f(ctx) for f in sub)
    if prim == "max":
        return lambda ctx: max(f(ctx) for f in sub)
    if prim == "clip":
        x, lo, hi = sub
        return lambda ctx: _clip(x(ctx), lo(ctx), hi(ctx))
    raise ValueError(f"unhandled primitive {prim!r}")   # unreachable after validate_spec


def _clip(x: float, lo: float, hi: float) -> float:
    if math.isnan(x) or math.isnan(lo) or math.isnan(hi):
        return math.nan
    if lo > hi:
        lo, hi = hi, lo
    return lo if x < lo else hi if x > hi else x


class CompiledSpec:
    """A callable `(pg, diagnostics) -> {column: finite float}` plus the metadata around it.

    A class rather than a bare function so the substitution counter and the controller-owned
    column names travel with the callable instead of being recomputed by every caller.
    """

    __slots__ = ("family_id", "output_names", "names", "uses_diagnostics", "n_substituted", "_fns")

    def __init__(self, family_id: str, output_names: list[str], fns: list,
                 uses_diagnostics: bool) -> None:
        self.family_id = family_id
        self.output_names = output_names
        self.names = [f"{family_id}.{n}" for n in output_names]
        self.uses_diagnostics = uses_diagnostics
        self.n_substituted = 0
        self._fns = fns

    def reset(self) -> None:
        self.n_substituted = 0

    def __call__(self, pg: ParsedGenome, diagnostics: dict | None = None) -> dict[str, float]:
        ctx = _Ctx(pg, diagnostics)
        out: dict[str, float] = {}
        subs = 0
        for name, fn in zip(self.names, self._fns):
            try:
                value = float(fn(ctx))
            except (ArithmeticError, ValueError, TypeError):
                # A spec is data, not code, but arithmetic still overflows. Never let one output
                # take down a corpus-wide extraction.
                value = math.nan
            if not math.isfinite(value):
                value = 0.0
                subs += 1
            out[name] = value
        self.n_substituted += subs
        return out


def compile_spec(spec, *, family_id: str | None = None) -> CompiledSpec:
    """Compile a validated spec into a callable. Raises ValueError if the spec does not validate.

    `family_id` is the controller's. It may arrive as an explicit argument or as a controller-set
    `spec["family_id"]`; the argument wins. There is no default, because unprefixed column names
    collide across families and a silent collision inside a stacked matrix is unrecoverable.
    """
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid FeatureSpec: " + "; ".join(errors))
    fid = family_id or spec.get("family_id")
    if not fid:
        raise ValueError(
            "compile_spec needs a controller-minted family_id (argument or spec['family_id']); "
            "agents never name their own columns"
        )
    if not _FAMILY_ID_RE.match(str(fid)):
        raise ValueError(f"family_id {fid!r} is not a valid controller-minted id")

    prim_cache: dict = {"uses_diagnostics": False}
    names = [out["name"] for out in spec["outputs"]]
    fns = [_compile_node(out["expr"], prim_cache) for out in spec["outputs"]]
    compiled = CompiledSpec(str(fid), names, fns, bool(prim_cache["uses_diagnostics"]))
    assert_no_forbidden_inputs(compiled.names)
    return compiled


# --- Execution -----------------------------------------------------------------------------------
def execute_spec(fn, genomes, engine: str, diagnostics_by_dataset: dict | None = None,
                 datasets=None) -> tuple[np.ndarray, list[str], dict]:
    """Run a compiled spec over the corpus. -> (float32 (n_rows, n_outputs), names, stats).

    Distinct work is done once and fanned out by inverse index. The dedup KEY depends on the spec:
    genome alone for a pure structural spec (34,419 distinct over 37,703 rows), but (genome,
    dataset) whenever the spec calls `diag(...)`, because the output is then a function of both and
    genome-keyed dedup would broadcast one dataset's diagnostics over every row sharing that
    genome. Getting this wrong produces a plausible matrix with a fabricated dataset signal, which
    is exactly the failure a within-dataset target is least able to reveal.
    """
    started = time.perf_counter()
    genomes = np.asarray(genomes, dtype=object)
    n_rows = len(genomes)

    uses_diag = bool(getattr(fn, "uses_diagnostics", False))
    if uses_diag and datasets is None:
        raise ValueError(
            "this spec reads dataset diagnostics; `datasets` is required so rows can be keyed by "
            "(genome, dataset) rather than genome alone"
        )
    if datasets is not None:
        datasets = np.asarray(datasets, dtype=object)
        if len(datasets) != n_rows:
            raise ValueError(f"datasets has {len(datasets)} rows, genomes has {n_rows}")

    if uses_diag:
        keys = np.array([f"{g}\x00{d}" for g, d in zip(genomes, datasets)], dtype=object)
        uniq, inverse = np.unique(keys, return_inverse=True)
        pairs = [tuple(str(k).split("\x00", 1)) for k in uniq]
    else:
        uniq, inverse = np.unique(genomes, return_inverse=True)
        pairs = [(str(g), None) for g in uniq]
    inverse = np.asarray(inverse).reshape(-1)

    diag_map = diagnostics_by_dataset or {}
    names = list(getattr(fn, "names", ()) or ())
    compact: np.ndarray | None = None
    per_key_subs = np.zeros(len(pairs), dtype=np.int64)
    n_parse_failures = 0

    if hasattr(fn, "reset"):
        fn.reset()
    before = int(getattr(fn, "n_substituted", 0))

    for r, (text, dataset) in enumerate(pairs):
        pg = parse(text, engine)
        if not pg.ok:
            n_parse_failures += 1
        values = fn(pg, diag_map.get(dataset) if dataset is not None else None)
        if compact is None:
            if not names:
                names = list(values)
            compact = np.zeros((len(pairs), len(names)), dtype=np.float64)
        elif len(values) != len(names):
            raise ValueError(f"spec returned {len(values)} outputs, expected {len(names)}")
        row = compact[r]
        for j, name in enumerate(names):
            row[j] = values[name]
        after = int(getattr(fn, "n_substituted", 0))
        per_key_subs[r] = after - before
        before = after

    if compact is None:      # empty corpus
        compact = np.zeros((0, len(names)), dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        # Overflow to inf here is an expected, handled outcome, not a warning worth printing
        # 34,419 times; the pass below counts and substitutes it.
        out = (compact[inverse].astype(np.float32) if n_rows
               else np.zeros((0, len(names)), np.float32))

    # Second finiteness pass. Not redundant: float64 values above the float32 range become inf on
    # downcast, and an inf here would reach the screen as a real feature value.
    n_substituted = int(per_key_subs[inverse].sum()) if n_rows else 0
    bad = ~np.isfinite(out)
    if bad.any():
        n_substituted += int(bad.sum())
        out[bad] = 0.0

    constant = ([] if len(out) == 0
                else [names[j] for j in range(out.shape[1])
                      if out[:, j].max() == out[:, j].min()])

    assert_no_forbidden_inputs(names)
    stats = {
        "n_rows": int(n_rows),
        "n_outputs": len(names),
        "n_unique_keys": len(pairs),
        "keyed_by": "genome_dataset" if uses_diag else "genome",
        "n_parse_failures": int(n_parse_failures),
        "n_nonfinite_substituted": int(n_substituted),
        "n_constant_columns": len(constant),
        "constant_columns": constant[:25],
        "seconds": round(time.perf_counter() - started, 3),
    }
    return out, list(names), stats


# --- Schema --------------------------------------------------------------------------------------
def spec_schema(engine: str | None = None) -> dict:
    """JSON schema for a FeatureSpec, embedded verbatim in the proposal prompt.

    Carries an `x_primitives` annotation because the schema alone cannot express the expression
    language, and a role that has to guess the primitive signatures burns its two attempts on
    syntax instead of on the theory. Note what is NOT here: `family_id` is documented as
    controller-owned and is absent from `required`, so a role is never asked to invent an id.
    """
    ops = _known_ops(engine)
    prims = {}
    for prim in PRIMITIVES:
        sig = _SIGNATURES[prim]
        prims[prim] = {"args": list(sig), "doc": _DOC.get(prim, "")}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "zyal.FeatureSpec",
        "type": "object",
        "additionalProperties": False,
        "required": ["outputs"],
        "properties": {
            "outputs": {
                "type": "array",
                "minItems": MIN_OUTPUTS,
                "maxItems": MAX_OUTPUTS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "expr"],
                    "properties": {
                        "name": {"type": "string", "pattern": _NAME_RE.pattern},
                        "expr": {"$ref": "#/$defs/expr"},
                        "rationale": {"type": "string", "maxLength": 240},
                    },
                },
            },
            "rationale": {"type": "string", "maxLength": 1000},
            "engine": {"type": "string"},
            "family_id": {"type": "string", "description": "CONTROLLER-OWNED; do not set."},
            "schema": {"type": "string"},
        },
        "$defs": {
            "expr": {
                "description": (
                    "A finite number, or a list [primitive, ...args]. String arguments are literal "
                    "op/family/param/diagnostic names, never nested expressions."
                ),
                "oneOf": [
                    {"type": "number"},
                    {"type": "array", "minItems": 1,
                     "items": {"oneOf": [{"type": "string"}, {"type": "number"},
                                         {"$ref": "#/$defs/expr"}]}},
                ],
            },
        },
        "x_primitives": prims,
        "x_families": list(FAMILY_LIST_V3),
        "x_limits": {
            "max_depth": MAX_EXPR_DEPTH,
            "max_nodes_per_output": MAX_EXPR_NODES,
            "absent_sentinel": ABSENT,
        },
    }
    if ops:
        schema["x_ops"] = sorted(ops)
    return schema
