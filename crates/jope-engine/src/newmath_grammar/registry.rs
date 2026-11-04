use super::{strings, ExprSpec, GrammarSpec, OpSpec, ParamSpec, DEFAULT_TASKS, ENGINE};
use serde_json::{json, Value};

fn param(name: &str, kind: &str, bins: Vec<Value>) -> ParamSpec {
    ParamSpec::new(name, kind, bins)
}

macro_rules! bins {
    ($($value:expr),* $(,)?) => { vec![$(json!($value)),*] };
}

fn op(
    name: &str,
    role: &str,
    families: &[&str],
    params: Vec<ParamSpec>,
    cost_class: &str,
    generate: bool,
    takes_expr: bool,
) -> OpSpec {
    OpSpec {
        name: name.into(),
        role: role.into(),
        params,
        families: strings(families),
        cost_class: cost_class.into(),
        allow_append: false,
        generate,
        takes_expr,
        tasks: strings(&DEFAULT_TASKS),
    }
}

fn t(name: &str, families: &[&str], params: Vec<ParamSpec>) -> OpSpec {
    op(name, "transform", families, params, "cheap", true, false)
}

fn t_medium(name: &str, families: &[&str], params: Vec<ParamSpec>) -> OpSpec {
    op(name, "transform", families, params, "medium", true, false)
}

fn t_parse_only(name: &str, families: &[&str], params: Vec<ParamSpec>) -> OpSpec {
    op(name, "transform", families, params, "cheap", false, false)
}

fn h(name: &str, families: &[&str], cost: &str, params: Vec<ParamSpec>) -> OpSpec {
    op(name, "head", families, params, cost, true, false)
}

pub fn grammar() -> GrammarSpec {
    let transforms = vec![
        t("robust_scale", &["scaling"], vec![]),
        t("rank", &["rank"], vec![]),
        t("norm_features", &["scaling"], vec![]),
        t("sigmoid", &["basis"], vec![]),
        t("polar", &["geometry", "rowwise"], vec![]),
        t("radial", &["geometry", "rowwise"], vec![]),
        t("poincare", &["geometry", "rowwise"], vec![]),
        t("entropy", &["spectral"], vec![]),
        t("spectral_entropy", &["spectral"], vec![]),
        t(
            "fft_mag",
            &["spectral"],
            vec![param("axis", "str", bins!["features"])],
        ),
        t(
            "rfft_mag",
            &["spectral"],
            vec![param("axis", "str", bins!["features"])],
        ),
        t(
            "rfft_phase",
            &["spectral"],
            vec![param("axis", "str", bins!["features"])],
        ),
        t(
            "spectral_moment",
            &["spectral"],
            vec![param("order", "int", bins![1, 2, 3])],
        ),
        t("peak_freq", &["spectral"], vec![]),
        t(
            "rff_cos",
            &["random_features", "kernel"],
            vec![
                param("n", "int", bins![16, 32, 64, 128, 256]),
                param("seed", "int", bins![1, 7, 9, 13, 29, 347]),
            ],
        ),
        t(
            "rff_sin",
            &["random_features", "kernel"],
            vec![
                param("n", "int", bins![16, 32, 64, 128]),
                param("seed", "int", bins![3, 5, 13, 29]),
            ],
        ),
        t(
            "phase_interference",
            &["random_features", "kernel"],
            vec![
                param("n", "int", bins![16, 32, 64, 128]),
                param("n_paths", "int", bins![2, 3, 4, 5, 7]),
                param("seed", "int", bins![1, 7]),
            ],
        ),
        t(
            "cheb",
            &["basis"],
            vec![param("degree", "int", bins![2, 3, 4, 6])],
        ),
        t(
            "poly_interact",
            &["interaction"],
            vec![param("degree", "int", bins![2, 3])],
        ),
        t(
            "proj",
            &["projection"],
            vec![
                param("k", "int", bins![8, 16, 32, 64]),
                param("seed", "int", bins![1, 3, 7, 17]),
            ],
        ),
        t_medium(
            "proj_eigen",
            &["projection"],
            vec![param("iters", "int", bins![8, 16, 24])],
        ),
        t_medium(
            "svd_project",
            &["projection"],
            vec![
                param("k", "int", bins![4, 8, 16, 24, 32]),
                param("iters", "int", bins![8, 16, 24]),
            ],
        ),
        t_medium(
            "nystrom",
            &["random_features", "kernel", "projection"],
            vec![
                param("k", "int", bins![4, 8, 16, 32]),
                param("gamma", "float", bins![0.05, 0.1, 0.2, 0.5]),
            ],
        ),
        t(
            "quantile_bins",
            &["rank"],
            vec![param("bins", "int", bins![3, 4, 6, 8, 12])],
        ),
        t(
            "wavelet",
            &["wavelet", "spectral"],
            vec![
                param("family", "str", bins!["dyadic", "morlet", "mexican"]),
                param("scale", "int", bins![1, 2, 3, 4]),
            ],
        ),
        t(
            "iterate",
            &["chaos"],
            vec![
                param("steps", "int", bins![2, 4, 6, 8, 10]),
                param("dt", "float", bins![0.05, 0.1, 0.2]),
                param("seed", "int", bins![1, 7]),
            ],
        ),
        t(
            "reduce",
            &["reduction"],
            vec![param(
                "op",
                "str",
                bins!["mean", "var", "max", "sum", "l2", "energy"],
            )],
        ),
        t(
            "quasicrystal",
            &["chaos"],
            vec![param("phi", "float", bins![1.6180339887])],
        ),
        t(
            "golden_scale",
            &["chaos"],
            vec![param("phi", "float", bins![1.6180339887])],
        ),
        t(
            "logistic_map",
            &["chaos"],
            vec![param("r", "float", bins![3.6, 3.8, 3.9])],
        ),
        t_medium(
            "eigen_ground_state",
            &["projection"],
            vec![param("iters", "int", bins![8, 16])],
        ),
        t(
            "persistence1d",
            &["topology"],
            vec![param("topk", "int", bins![2, 4, 8])],
        ),
        t_medium(
            "local_density",
            &["topology", "neighbors"],
            vec![param("k", "int", bins![4, 8, 16])],
        ),
        t("gate_mix", &["reduction"], vec![]),
        t("cross_ratio", &["geometry", "rowwise"], vec![]),
        t_parse_only(
            "branch",
            &["chaos"],
            vec![
                param("n", "int", bins![2, 3]),
                param("seed", "int", bins![1, 7]),
            ],
        ),
        t_parse_only(
            "residual_boost",
            &["basis"],
            vec![param("rounds", "int", bins![2])],
        ),
        t_parse_only("concat", &["basis"], vec![param("banks", "int", bins![2])]),
        op(
            "map",
            "transform",
            &["symbolic", "basis"],
            vec![
                ParamSpec::required_default("expr", "tanh(x)").tap_bins(bins![
                    "sigmoid(x)",
                    "log1p(x)",
                    "tanh(x)",
                    "x*x",
                    "abs(x)"
                ]),
            ],
            "cheap",
            true,
            true,
        ),
        op(
            "derive",
            "transform",
            &["symbolic", "basis"],
            vec![ParamSpec::required_default("expr", "row_mean")],
            "cheap",
            true,
            true,
        ),
    ];
    let heads = vec![
        h(
            "ridge",
            &["linear"],
            "cheap",
            vec![param("alpha", "float", bins![0.1, 0.5, 1.0, 3.0])],
        ),
        op(
            "logistic",
            "head",
            &["linear"],
            vec![param("alpha", "float", bins![1.0])],
            "cheap",
            false,
            false,
        ),
        h(
            "gbm",
            &["tree"],
            "medium",
            vec![
                param("rounds", "int", bins![80, 150, 240]),
                param("depth", "int", bins![2, 3, 4]),
                param("lr", "float", bins![0.05, 0.1]),
            ],
        ),
        op(
            "random_forest",
            "head",
            &["tree"],
            vec![param("trees", "int", bins![100])],
            "medium",
            false,
            false,
        ),
        h(
            "ensemble_ridge_gbm",
            &["ensemble"],
            "medium",
            vec![param("alpha", "float", bins![1.0])],
        ),
        h(
            "knn",
            &["neighbors"],
            "medium",
            vec![
                param("k", "int", bins![5, 15, 35]),
                param("weights", "str", bins!["distance", "uniform"]),
            ],
        ),
        h(
            "elasticnet",
            &["linear"],
            "cheap",
            vec![
                param("alpha", "float", bins![0.0001, 0.001, 0.01, 0.05]),
                param("l1_ratio", "float", bins![0.2, 0.5, 0.8]),
            ],
        ),
        h(
            "huber_ridge",
            &["linear", "robust"],
            "cheap",
            vec![
                param("alpha", "float", bins![0.1, 1.0, 3.0]),
                param("delta", "float", bins![1.0, 1.35, 2.0]),
            ],
        ),
    ];
    GrammarSpec {
        engine: ENGINE.into(),
        version: "newmath.v2".into(),
        evaluator_version: "jope-newmath-validate.eval-regression.v1".into(),
        max_transforms: 4,
        transforms,
        heads,
        expr: ExprSpec::default(),
        fallback_genome: "(pipe X (robust_scale) (ridge alpha=1.0))".into(),
        max_expr_stages: 3,
    }
}
