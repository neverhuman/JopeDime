"""featmath (feat-math / jain_regressor) grammar — the widened engine-supported set.

31 transforms + 22 heads, matching rust/jope-validate's ACTIVE_TRANSFORMS/ACTIVE_HEADS.
The last four heads are the multiclass decode family; they are multiclass-only in
both directions (see the comment above them).
Static bins come from feat-math's evolve.rs generation grids plus observed report values;
learned bins from corpora are unioned in at corpus-build time (vocab.merge_learned_bins).
lag/roll are deliberately absent (they break iid CV); the (target ...) pseudo-stage and
(sym ...) genomes stay engine-rejected.
"""

from __future__ import annotations

from .spec import ExprSpec, GrammarSpec, OpSpec, ParamSpec

ENGINE = "featmath"

_EXPR = ExprSpec()


def _t(name: str, families: tuple[str, ...], params: tuple[ParamSpec, ...] = (), **kw) -> OpSpec:
    return OpSpec(name=name, role="transform", params=params, families=families, allow_append=True, **kw)


def _h(name: str, families: tuple[str, ...], cost_class: str, params: tuple[ParamSpec, ...] = (), **kw) -> OpSpec:
    return OpSpec(name=name, role="head", params=params, families=families, cost_class=cost_class, **kw)


_TRANSFORMS = [
    _t("standardize", ("scaling",)),
    _t("robust_scale", ("scaling",)),
    _t("norm_features", ("scaling",)),
    _t("rank", ("rank",)),
    _t("blom", ("rank",)),
    _t("quantile_bins", ("rank",), (ParamSpec("bins", "int", (4, 8, 16, 32)),)),
    _t("bin_onehot", ("rank",), (ParamSpec("bins", "int", (4, 8, 12, 16)),)),
    _t("hinge", ("basis",), (ParamSpec("knots", "int", (2, 4, 6, 8)),)),
    _t("winsor_clip", ("scaling", "robust"), (ParamSpec("q", "float", (0.01, 0.02, 0.05, 0.1)),)),
    _t("winsor_moments", ("scaling", "robust"), (ParamSpec("q", "float", (0.02, 0.05, 0.1)),)),
    _t("cheb", ("basis",), (ParamSpec("degree", "int", (2, 3, 4, 6)),)),
    _t("poly_interact", ("interaction",), (ParamSpec("topk", "int", (4, 8, 12)),)),
    _t(
        "rff_cos",
        ("random_features", "kernel"),
        (
            ParamSpec("n", "int", (8, 16, 32, 64)),
            ParamSpec("gamma", "float", (0.5, 1.0, 2.0)),
            ParamSpec("seed", "int", (2, 7, 11, 23)),
        ),
    ),
    _t(
        "rff_sin",
        ("random_features", "kernel"),
        (
            ParamSpec("n", "int", (8, 16, 32, 64)),
            ParamSpec("gamma", "float", (0.5, 1.0, 2.0)),
            ParamSpec("seed", "int", (3, 5, 13, 29)),
        ),
    ),
    _t(
        "rand_proj",
        ("random_features", "kernel"),
        (ParamSpec("k", "int", (8, 16, 32, 64)), ParamSpec("seed", "int", (7, 17, 31))),
    ),
    _t(
        "oblique_hinge",
        ("random_features", "kernel"),
        (ParamSpec("n", "int", (8, 16, 32)), ParamSpec("seed", "int", (7, 19))),
    ),
    _t("row_stats", ("rowwise",)),
    _t("norm_row", ("rowwise",)),
    _t("row_diff_stats", ("rowwise",)),
    _t("pair_prod", ("interaction",), (ParamSpec("topk", "int", (4, 8, 12)),)),
    _t(
        "target_bin_encode",
        ("target_aware",),
        (ParamSpec("bins", "int", (4, 8, 16)), ParamSpec("prior", "int", (1, 5, 10))),
    ),
    _t("sigmoid", ("basis",)),
    _t("step", ("basis",), (ParamSpec("t", "float", (0.25, 0.5, 0.75)),)),
    _t("polar", ("geometry", "rowwise")),
    _t("fft_mag", ("spectral",), (ParamSpec("k", "int", (4, 8, 16)),)),
    _t("row_spectral", ("spectral",)),
    _t(
        "svd_project",
        ("projection", "spectral"),
        (ParamSpec("k", "int", (4, 8, 16, 32)), ParamSpec("iters", "int", (8, 16, 24)), ParamSpec("seed", "int", (7,))),
        cost_class="medium",
    ),
    _t(
        "nystrom",
        ("random_features", "kernel", "projection"),
        (
            ParamSpec("k", "int", (8, 16, 32)),
            ParamSpec("gamma", "float", (0.25, 0.5, 1.0)),
            ParamSpec("seed", "int", (7,)),
        ),
        cost_class="medium",
    ),
    _t(
        "landmark_dist",
        ("random_features", "kernel", "projection"),
        (ParamSpec("k", "int", (8, 16, 32)),),
        cost_class="medium",
    ),
    _t(
        "map",
        ("symbolic", "basis"),
        (ParamSpec("expr", "expr", (), default="tanh(x)", required=True),),
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
    _h("ridge", ("linear",), "cheap"),
    # Compatibility spellings retained in the grammar handshake. They remain
    # parse/encode-only until their exact emitter contracts are certified.
    _h("ridge_gcv", ("linear",), "medium", generate=False),
    _h("logistic", ("linear",), "cheap"),
    # Binary-flag heads stay binary-only: they cannot emit K-class genomes, so
    # granting them "multiclass" would let decode produce engine-rejected
    # pipelines. Multiclass decode now goes through the four multiclass_* heads
    # declared at the end of this list -- they are the only heads feat-math
    # implements `multiclass_probabilities` for (heads/mod.rs), so every other
    # head fails closed with "bound model does not carry a multiclass head".
    _h(
        "logit_ridge",
        ("linear",),
        "cheap",
        (
            ParamSpec("class_weight", "str", ("balanced",)),
            ParamSpec("l2", "float", (0.005, 0.02, 0.1, 0.5)),
        ),
        tasks=("binary",),
    ),
    _h(
        "tab_boost",
        ("tree",),
        "medium",
        (
            ParamSpec("bins", "int", (16, 32, 64, 128)),
            ParamSpec("lambda", "float", (0.5, 1.0, 5.0)),
            ParamSpec("lr", "float", (0.05, 0.1, 0.2)),
            ParamSpec("min_leaf", "int", (3, 5, 10)),
            ParamSpec("rounds", "int", (40, 60, 80, 100, 120)),
        ),
        tasks=("binary",),
    ),
    _h(
        "gam_boost",
        ("tree",),
        "medium",
        (
            ParamSpec("bins", "int", (8, 16, 24, 32)),
            ParamSpec("lr", "float", (0.05, 0.1, 0.2)),
            ParamSpec("rounds", "int", (40, 60, 80, 100)),
        ),
        tasks=("binary",),
    ),
    _h(
        "extra_trees",
        ("tree",),
        "medium",
        (
            ParamSpec("depth", "int", (6, 8, 10)),
            ParamSpec("min_leaf", "int", (3, 5, 10)),
            ParamSpec("trees", "int", (32, 64, 96)),
        ),
        tasks=("binary",),
    ),
    _h(
        "elasticnet",
        ("linear",),
        "cheap",
        (
            ParamSpec("alpha", "float", (0.0005, 0.001, 0.01)),
            ParamSpec("l1_ratio", "float", (0.2, 0.5, 0.8)),
        ),
    ),
    _h(
        "histgbm",
        ("tree",),
        "medium",
        (
            ParamSpec("depth", "int", (2, 3, 4)),
            ParamSpec("lr", "float", (0.05, 0.1, 0.15)),
            ParamSpec("rounds", "int", (80, 120, 180)),
        ),
    ),
    _h(
        "gbm",
        ("tree",),
        "medium",
        (
            ParamSpec("depth", "int", (2, 3, 4)),
            ParamSpec("lr", "float", (0.05, 0.1)),
            ParamSpec("rounds", "int", (80, 150, 240)),
        ),
    ),
    _h(
        "histgbm2",
        ("tree",),
        "expensive",
        (
            ParamSpec("leaves", "int", (15, 31, 63)),
            ParamSpec("lr", "float", (0.03, 0.05, 0.08)),
            ParamSpec("rounds", "int", (300, 500)),
            ParamSpec("bags", "int", (3, 5)),
        ),
    ),
    _h("krr", ("kernel",), "expensive", (ParamSpec("m", "int", (200, 600, 1000)),)),
    _h(
        "gpr",
        ("kernel",),
        "extreme",
        (
            ParamSpec(
                "kernel",
                "str",
                ("rbf", "rq", "linear", "periodic", "rbf+linear", "rq+linear", "rbf*periodic", "periodic+linear"),
            ),
            ParamSpec("m", "int", (300, 400)),
            ParamSpec("steps", "int", (30, 40)),
        ),
        # GP-likelihood regressor; extreme cost, poor $/AUC on classification.
        # Timeseries rides the regression lane (reduction-v2 tables are plain
        # regression problems), so regression-only heads serve timeseries too.
        tasks=("regression", "timeseries"),
    ),
    _h(
        "ksmooth",
        ("kernel",),
        "medium",
        (ParamSpec("k", "int", (16, 32, 64)), ParamSpec("gamma", "float", (0.5, 1.0, 2.0))),
    ),
    _h("ensemble", ("ensemble",), "medium"),
    _h("ensemble_ridge_gbm", ("ensemble",), "medium"),
    _h("stack", ("ensemble",), "expensive"),
    _h("portfolio", ("ensemble",), "portfolio", generate=False, tasks=("regression", "timeseries")),
    # ---------------------------------------------------------------------
    # Multiclass decode heads. These four are EXACTLY the heads feat-math
    # implements `multiclass_probabilities` for (advanced_v3/heads/mod.rs);
    # `predict_multiclass_table` errors with "bound model does not carry a
    # multiclass head" for anything else, and `fit_v3_head` refuses them on
    # non-multiclass tasks. They are therefore multiclass-ONLY in both
    # directions, which is why `tasks` is pinned rather than left at the
    # all-four default.
    #
    # Param grids mirror what the fitters actually read, not the typed v3
    # grammar (which registers only `multiclass_ridge`, and without
    # `feature_budget` -- while the fitter validates it and production
    # receipts carry it). `feature_budget` must be one of 32|64|128|256 or the
    # fit bails; `class_weight` must be "balanced" for multiclass_ridge.
    # `l2` uses feat-math's LOGIT_L2 set.
    _h(
        "multiclass_ridge",
        ("linear",),
        "cheap",
        (
            ParamSpec("feature_budget", "int", (32, 64, 128, 256)),
            ParamSpec("l2", "float", (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)),
            ParamSpec("class_weight", "str", ("balanced",)),
        ),
        tasks=("multiclass",),
    ),
    _h(
        "multiclass_ovr_logistic",
        ("linear",),
        "medium",
        (
            ParamSpec("feature_budget", "int", (32, 64, 128, 256)),
            ParamSpec("l2", "float", (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)),
        ),
        tasks=("multiclass",),
    ),
    _h(
        "multiclass_diagonal_lda",
        ("linear",),
        "cheap",
        (
            ParamSpec("feature_budget", "int", (32, 64, 128, 256)),
            ParamSpec("shrinkage", "float", (0.001, 0.01, 0.1, 0.5)),
        ),
        tasks=("multiclass",),
    ),
    _h(
        "multiclass_nearest_centroid",
        ("linear",),
        "cheap",
        (ParamSpec("feature_budget", "int", (32, 64, 128, 256)),),
        tasks=("multiclass",),
    ),
]

GRAMMAR = GrammarSpec(
    engine=ENGINE,
    version="featmath.v3",
    evaluator_version="jope-validate.eval-regression.v2",
    max_transforms=6,
    transforms={op.name: op for op in _TRANSFORMS},
    heads={op.name: op for op in _HEADS},
    expr=_EXPR,
    fallback_genome="(pipe X (robust_scale) (ridge))",
)
