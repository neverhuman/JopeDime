"""newmath (jmath) grammar — the proven op subset + symbolic exprs + bridge heads.

Matches rust/jope-newmath-validate's bridge registry: newmath core_v1 ops minus lag/roll
minus quarantined zyal/recon ops, plus bridge-implemented heads (knn, elasticnet,
huber_ridge). newmath never uses append. Param grids follow newmath's gp.rs jitter grids.
residual_boost (identity) and concat (alias) are parse-only (generate=False).
"""

from __future__ import annotations

from .spec import ExprSpec, GrammarSpec, OpSpec, ParamSpec

ENGINE = "newmath"

_EXPR = ExprSpec()

# Real, emittable, elementwise MathExpr for the `map` expr bins. The former opaque
# shorthands ("sigmoid_col0" …) were newmath-only tokens battle-transpile can't transpile
# AND referenced col(0) — exactly the runtime-width crash risk (RF4). These parse and
# emit in both engines, keeping 100% gene→code coverage.
_EXPR_MACROS = ("sigmoid(x)", "log1p(x)", "tanh(x)", "x*x", "abs(x)")


def _t(name: str, families: tuple[str, ...], params: tuple[ParamSpec, ...] = (), **kw) -> OpSpec:
    return OpSpec(name=name, role="transform", params=params, families=families, allow_append=False, **kw)


def _h(name: str, families: tuple[str, ...], cost_class: str, params: tuple[ParamSpec, ...] = (), **kw) -> OpSpec:
    return OpSpec(name=name, role="head", params=params, families=families, cost_class=cost_class, **kw)


_TRANSFORMS = [
    _t("robust_scale", ("scaling",)),
    _t("rank", ("rank",)),
    _t("norm_features", ("scaling",)),
    _t("sigmoid", ("basis",)),
    _t("polar", ("geometry", "rowwise")),
    _t("radial", ("geometry", "rowwise")),
    _t("poincare", ("geometry", "rowwise")),
    _t("entropy", ("spectral",)),
    _t("spectral_entropy", ("spectral",)),
    _t("fft_mag", ("spectral",), (ParamSpec("axis", "str", ("features",)),)),
    _t("rfft_mag", ("spectral",), (ParamSpec("axis", "str", ("features",)),)),
    _t("rfft_phase", ("spectral",), (ParamSpec("axis", "str", ("features",)),)),
    _t("spectral_moment", ("spectral",), (ParamSpec("order", "int", (1, 2, 3)),)),
    _t("peak_freq", ("spectral",)),
    _t(
        "rff_cos",
        ("random_features", "kernel"),
        (ParamSpec("n", "int", (16, 32, 64, 128, 256)), ParamSpec("seed", "int", (1, 7, 9, 13, 29, 347))),
    ),
    _t(
        "rff_sin",
        ("random_features", "kernel"),
        (ParamSpec("n", "int", (16, 32, 64, 128)), ParamSpec("seed", "int", (3, 5, 13, 29))),
    ),
    _t(
        "phase_interference",
        ("random_features", "kernel"),
        (
            ParamSpec("n", "int", (16, 32, 64, 128)),
            ParamSpec("n_paths", "int", (2, 3, 4, 5, 7)),
            ParamSpec("seed", "int", (1, 7)),
        ),
    ),
    _t("cheb", ("basis",), (ParamSpec("degree", "int", (2, 3, 4, 6)),)),
    _t("poly_interact", ("interaction",), (ParamSpec("degree", "int", (2, 3)),)),
    _t(
        "proj",
        ("projection",),
        (ParamSpec("k", "int", (8, 16, 32, 64)), ParamSpec("seed", "int", (1, 3, 7, 17))),
    ),
    _t("proj_eigen", ("projection",), (ParamSpec("iters", "int", (8, 16, 24)),), cost_class="medium"),
    _t(
        "svd_project",
        ("projection",),
        (ParamSpec("k", "int", (4, 8, 16, 24, 32)), ParamSpec("iters", "int", (8, 16, 24))),
        cost_class="medium",
    ),
    _t(
        "nystrom",
        ("random_features", "kernel", "projection"),
        (ParamSpec("k", "int", (4, 8, 16, 32)), ParamSpec("gamma", "float", (0.05, 0.1, 0.2, 0.5))),
        cost_class="medium",
    ),
    _t("quantile_bins", ("rank",), (ParamSpec("bins", "int", (3, 4, 6, 8, 12)),)),
    _t(
        "wavelet",
        ("wavelet", "spectral"),
        (ParamSpec("family", "str", ("dyadic", "morlet", "mexican")), ParamSpec("scale", "int", (1, 2, 3, 4))),
    ),
    _t(
        "iterate",
        ("chaos",),
        (
            ParamSpec("steps", "int", (2, 4, 6, 8, 10)),
            ParamSpec("dt", "float", (0.05, 0.1, 0.2)),
            ParamSpec("seed", "int", (1, 7)),
        ),
    ),
    _t("reduce", ("reduction",), (ParamSpec("op", "str", ("mean", "var", "max", "sum", "l2", "energy")),)),
    _t("quasicrystal", ("chaos",), (ParamSpec("phi", "float", (1.6180339887,)),)),
    _t("golden_scale", ("chaos",), (ParamSpec("phi", "float", (1.6180339887,)),)),
    _t("logistic_map", ("chaos",), (ParamSpec("r", "float", (3.6, 3.8, 3.9)),)),
    _t("eigen_ground_state", ("projection",), (ParamSpec("iters", "int", (8, 16)),), cost_class="medium"),
    _t("persistence1d", ("topology",), (ParamSpec("topk", "int", (2, 4, 8)),)),
    _t("local_density", ("topology", "neighbors"), (ParamSpec("k", "int", (4, 8, 16)),), cost_class="medium"),
    _t("gate_mix", ("reduction",)),
    _t("cross_ratio", ("geometry", "rowwise")),
    _t(
        "branch",
        ("chaos",),
        (ParamSpec("n", "int", (2, 3)), ParamSpec("seed", "int", (1, 7))),
        generate=False,
    ),
    _t("residual_boost", ("basis",), (ParamSpec("rounds", "int", (2,)),), generate=False),
    _t("concat", ("basis",), (ParamSpec("banks", "int", (2,)),), generate=False),
    _t(
        "map",
        ("symbolic", "basis"),
        (ParamSpec("expr", "expr", _EXPR_MACROS, default="tanh(x)", required=True),),
        takes_expr=True,
    ),
    _t(
        "derive",
        ("symbolic", "basis"),
        (ParamSpec("expr", "expr", (), default="row_mean", required=True),),
        takes_expr=True,
    ),
]

_HEADS = [
    _h("ridge", ("linear",), "cheap", (ParamSpec("alpha", "float", (0.1, 0.5, 1.0, 3.0)),)),
    _h("logistic", ("linear",), "cheap", (ParamSpec("alpha", "float", (1.0,)),), generate=False),
    _h(
        "gbm",
        ("tree",),
        "medium",
        (
            ParamSpec("rounds", "int", (80, 150, 240)),
            ParamSpec("depth", "int", (2, 3, 4)),
            ParamSpec("lr", "float", (0.05, 0.1)),
        ),
    ),
    _h("random_forest", ("tree",), "medium", (ParamSpec("trees", "int", (100,)),), generate=False),
    _h("ensemble_ridge_gbm", ("ensemble",), "medium", (ParamSpec("alpha", "float", (1.0,)),)),
    _h(
        "knn",
        ("neighbors",),
        "medium",
        (ParamSpec("k", "int", (5, 15, 35)), ParamSpec("weights", "str", ("distance", "uniform"))),
    ),
    _h(
        "elasticnet",
        ("linear",),
        "cheap",
        (
            ParamSpec("alpha", "float", (0.0001, 0.001, 0.01, 0.05)),
            ParamSpec("l1_ratio", "float", (0.2, 0.5, 0.8)),
        ),
    ),
    _h(
        "huber_ridge",
        ("linear", "robust"),
        "cheap",
        (
            ParamSpec("alpha", "float", (0.1, 1.0, 3.0)),
            ParamSpec("delta", "float", (1.0, 1.35, 2.0)),
        ),
    ),
]

GRAMMAR = GrammarSpec(
    engine=ENGINE,
    version="newmath.v2",
    evaluator_version="jope-newmath-validate.eval-regression.v1",
    max_transforms=4,
    transforms={op.name: op for op in _TRANSFORMS},
    heads={op.name: op for op in _HEADS},
    expr=_EXPR,
    fallback_genome="(pipe X (robust_scale) (ridge alpha=1.0))",
)
