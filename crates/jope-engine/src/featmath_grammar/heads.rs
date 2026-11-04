use serde_json::json;

use super::{head, param, tasks, OpSpec};

macro_rules! bins {
    ($($value:expr),* $(,)?) => { vec![$(json!($value)),*] };
}

pub(super) fn definitions() -> Vec<OpSpec> {
    let mut ridge_gcv = head("ridge_gcv", &["linear"], "medium", vec![]);
    ridge_gcv.generate = false;
    let mut logit_ridge = head(
        "logit_ridge",
        &["linear"],
        "cheap",
        vec![
            param("class_weight", "str", bins!["balanced"]),
            param("l2", "float", bins![0.005, 0.02, 0.1, 0.5]),
        ],
    );
    logit_ridge.tasks = tasks(&["binary"]);
    let mut tab_boost = head(
        "tab_boost",
        &["tree"],
        "medium",
        vec![
            param("bins", "int", bins![16, 32, 64, 128]),
            param("lambda", "float", bins![0.5, 1.0, 5.0]),
            param("lr", "float", bins![0.05, 0.1, 0.2]),
            param("min_leaf", "int", bins![3, 5, 10]),
            param("rounds", "int", bins![40, 60, 80, 100, 120]),
        ],
    );
    tab_boost.tasks = tasks(&["binary"]);
    let mut gam_boost = head(
        "gam_boost",
        &["tree"],
        "medium",
        vec![
            param("bins", "int", bins![8, 16, 24, 32]),
            param("lr", "float", bins![0.05, 0.1, 0.2]),
            param("rounds", "int", bins![40, 60, 80, 100]),
        ],
    );
    gam_boost.tasks = tasks(&["binary"]);
    let mut extra_trees = head(
        "extra_trees",
        &["tree"],
        "medium",
        vec![
            param("depth", "int", bins![6, 8, 10]),
            param("min_leaf", "int", bins![3, 5, 10]),
            param("trees", "int", bins![32, 64, 96]),
        ],
    );
    extra_trees.tasks = tasks(&["binary"]);
    let mut gpr = head(
        "gpr",
        &["kernel"],
        "extreme",
        vec![
            param(
                "kernel",
                "str",
                bins![
                    "rbf",
                    "rq",
                    "linear",
                    "periodic",
                    "rbf+linear",
                    "rq+linear",
                    "rbf*periodic",
                    "periodic+linear"
                ],
            ),
            param("m", "int", bins![300, 400]),
            param("steps", "int", bins![30, 40]),
        ],
    );
    gpr.tasks = tasks(&["regression", "timeseries"]);
    let mut portfolio = head("portfolio", &["ensemble"], "portfolio", vec![]);
    portfolio.generate = false;
    portfolio.tasks = tasks(&["regression", "timeseries"]);
    let mut multiclass_ridge = head(
        "multiclass_ridge",
        &["linear"],
        "cheap",
        vec![
            param("feature_budget", "int", bins![32, 64, 128, 256]),
            param("l2", "float", bins![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
            param("class_weight", "str", bins!["balanced"]),
        ],
    );
    multiclass_ridge.tasks = tasks(&["multiclass"]);
    let mut multiclass_ovr = head(
        "multiclass_ovr_logistic",
        &["linear"],
        "medium",
        vec![
            param("feature_budget", "int", bins![32, 64, 128, 256]),
            param("l2", "float", bins![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
        ],
    );
    multiclass_ovr.tasks = tasks(&["multiclass"]);
    let mut multiclass_lda = head(
        "multiclass_diagonal_lda",
        &["linear"],
        "cheap",
        vec![
            param("feature_budget", "int", bins![32, 64, 128, 256]),
            param("shrinkage", "float", bins![0.001, 0.01, 0.1, 0.5]),
        ],
    );
    multiclass_lda.tasks = tasks(&["multiclass"]);
    let mut multiclass_centroid = head(
        "multiclass_nearest_centroid",
        &["linear"],
        "cheap",
        vec![param("feature_budget", "int", bins![32, 64, 128, 256])],
    );
    multiclass_centroid.tasks = tasks(&["multiclass"]);

    vec![
        head("ridge", &["linear"], "cheap", vec![]),
        ridge_gcv,
        head("logistic", &["linear"], "cheap", vec![]),
        logit_ridge,
        tab_boost,
        gam_boost,
        extra_trees,
        head(
            "elasticnet",
            &["linear"],
            "cheap",
            vec![
                param("alpha", "float", bins![0.0005, 0.001, 0.01]),
                param("l1_ratio", "float", bins![0.2, 0.5, 0.8]),
            ],
        ),
        head(
            "histgbm",
            &["tree"],
            "medium",
            vec![
                param("depth", "int", bins![2, 3, 4]),
                param("lr", "float", bins![0.05, 0.1, 0.15]),
                param("rounds", "int", bins![80, 120, 180]),
            ],
        ),
        head(
            "gbm",
            &["tree"],
            "medium",
            vec![
                param("depth", "int", bins![2, 3, 4]),
                param("lr", "float", bins![0.05, 0.1]),
                param("rounds", "int", bins![80, 150, 240]),
            ],
        ),
        head(
            "histgbm2",
            &["tree"],
            "expensive",
            vec![
                param("leaves", "int", bins![15, 31, 63]),
                param("lr", "float", bins![0.03, 0.05, 0.08]),
                param("rounds", "int", bins![300, 500]),
                param("bags", "int", bins![3, 5]),
            ],
        ),
        head(
            "krr",
            &["kernel"],
            "expensive",
            vec![param("m", "int", bins![200, 600, 1000])],
        ),
        gpr,
        head(
            "ksmooth",
            &["kernel"],
            "medium",
            vec![
                param("k", "int", bins![16, 32, 64]),
                param("gamma", "float", bins![0.5, 1.0, 2.0]),
            ],
        ),
        head("ensemble", &["ensemble"], "medium", vec![]),
        head("ensemble_ridge_gbm", &["ensemble"], "medium", vec![]),
        head("stack", &["ensemble"], "expensive", vec![]),
        portfolio,
        multiclass_ridge,
        multiclass_ovr,
        multiclass_lda,
        multiclass_centroid,
    ]
}
