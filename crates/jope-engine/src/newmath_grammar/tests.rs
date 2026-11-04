use super::*;
use serde_json::json;

fn names(ops: &[OpSpec]) -> Vec<&str> {
    ops.iter().map(|op| op.name.as_str()).collect()
}

#[test]
fn registry_identity_and_order_match_python_newmath() {
    let grammar = grammar();
    assert_eq!(grammar.engine, "newmath");
    assert_eq!(grammar.version, "newmath.v2");
    assert_eq!(
        grammar.evaluator_version,
        "jope-newmath-validate.eval-regression.v1"
    );
    assert_eq!(grammar.max_transforms, 4);
    assert_eq!(grammar.max_expr_stages, 3);
    assert_eq!(
        grammar.fallback_genome,
        "(pipe X (robust_scale) (ridge alpha=1.0))"
    );
    assert_eq!(
        names(&grammar.transforms),
        vec![
            "robust_scale",
            "rank",
            "norm_features",
            "sigmoid",
            "polar",
            "radial",
            "poincare",
            "entropy",
            "spectral_entropy",
            "fft_mag",
            "rfft_mag",
            "rfft_phase",
            "spectral_moment",
            "peak_freq",
            "rff_cos",
            "rff_sin",
            "phase_interference",
            "cheb",
            "poly_interact",
            "proj",
            "proj_eigen",
            "svd_project",
            "nystrom",
            "quantile_bins",
            "wavelet",
            "iterate",
            "reduce",
            "quasicrystal",
            "golden_scale",
            "logistic_map",
            "eigen_ground_state",
            "persistence1d",
            "local_density",
            "gate_mix",
            "cross_ratio",
            "branch",
            "residual_boost",
            "concat",
            "map",
            "derive",
        ]
    );
    assert_eq!(
        names(&grammar.heads),
        vec![
            "ridge",
            "logistic",
            "gbm",
            "random_forest",
            "ensemble_ridge_gbm",
            "knn",
            "elasticnet",
            "huber_ridge",
        ]
    );
}

#[test]
fn generation_append_tasks_and_expression_flags_match_python_specs() {
    let grammar = grammar();
    let transforms = &grammar.transforms;
    assert!(transforms.iter().all(|op| !op.allow_append));
    assert!(transforms
        .iter()
        .all(|op| op.tasks == strings(&DEFAULT_TASKS)));
    for name in ["branch", "residual_boost", "concat"] {
        assert!(!grammar.op(name).unwrap().generate);
    }
    for name in ["map", "derive"] {
        assert!(grammar.op(name).unwrap().takes_expr);
    }
    assert!(!grammar.op("logistic").unwrap().generate);
    assert!(!grammar.op("random_forest").unwrap().generate);
    assert!(grammar.op("ridge").unwrap().generate);
    assert_eq!(grammar.op("proj_eigen").unwrap().cost_class, "medium");
    assert_eq!(
        grammar.op("nystrom").unwrap().families,
        strings(&["random_features", "kernel", "projection"])
    );
    assert_eq!(grammar.expr.vars_map, strings(&["x"]));
    assert_eq!(grammar.expr.max_col, 8);
    assert_eq!(grammar.expr.max_depth, 4);
    assert_eq!(grammar.expr.max_nodes, 15);
}

#[test]
fn parameter_bins_defaults_and_json_fields_match_python_specs() {
    let grammar = grammar();
    let rff = grammar.op("rff_cos").unwrap();
    assert_eq!(rff.params[0].name, "n");
    assert_eq!(
        rff.params[0].bins,
        vec![json!(16), json!(32), json!(64), json!(128), json!(256)]
    );
    assert_eq!(
        rff.params[1].bins,
        vec![
            json!(1),
            json!(7),
            json!(9),
            json!(13),
            json!(29),
            json!(347)
        ]
    );
    let map = &grammar.op("map").unwrap().params[0];
    assert_eq!(map.default, Some(json!("tanh(x)")));
    assert!(map.required);
    assert_eq!(
        map.bins,
        vec![
            json!("sigmoid(x)"),
            json!("log1p(x)"),
            json!("tanh(x)"),
            json!("x*x"),
            json!("abs(x)")
        ]
    );
    let derive = &grammar.op("derive").unwrap().params[0];
    assert_eq!(derive.default, Some(json!("row_mean")));
    assert!(derive.required);
    let serialized = grammar.to_json();
    assert_eq!(serialized["engine"], "newmath");
    assert_eq!(serialized["transforms"]["branch"]["generate"], false);
    assert_eq!(
        serialized["heads"]["ridge"]["params"][0]["bins"],
        json!([0.1, 0.5, 1.0, 3.0])
    );
    assert_eq!(serialized["expr"]["consts"][0], "-2");
}
