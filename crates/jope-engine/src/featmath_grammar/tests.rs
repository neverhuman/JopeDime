use super::*;
use serde_json::json;

macro_rules! bins {
    ($($value:expr),* $(,)?) => { vec![$(json!($value)),*] };
}

#[test]
fn static_counts_defaults_and_expression_limits_match_featmath_spec() {
    let grammar = grammar();
    assert_eq!(
        (grammar.engine.as_str(), grammar.version.as_str()),
        ("featmath", "featmath.v3")
    );
    assert_eq!(
        grammar.evaluator_version,
        "jope-validate.eval-regression.v2"
    );
    assert_eq!((grammar.max_transforms, grammar.max_expr_stages), (6, 3));
    assert_eq!(grammar.transforms.len(), 31);
    assert_eq!(grammar.heads.len(), 22);
    assert_eq!(
        grammar
            .transforms
            .iter()
            .map(|op| op.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "standardize",
            "robust_scale",
            "norm_features",
            "rank",
            "blom",
            "quantile_bins",
            "bin_onehot",
            "hinge",
            "winsor_clip",
            "winsor_moments",
            "cheb",
            "poly_interact",
            "rff_cos",
            "rff_sin",
            "rand_proj",
            "oblique_hinge",
            "row_stats",
            "norm_row",
            "row_diff_stats",
            "pair_prod",
            "target_bin_encode",
            "sigmoid",
            "step",
            "polar",
            "fft_mag",
            "row_spectral",
            "svd_project",
            "nystrom",
            "landmark_dist",
            "map",
            "derive",
        ]
    );
    assert_eq!(
        grammar
            .heads
            .iter()
            .map(|op| op.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "ridge",
            "ridge_gcv",
            "logistic",
            "logit_ridge",
            "tab_boost",
            "gam_boost",
            "extra_trees",
            "elasticnet",
            "histgbm",
            "gbm",
            "histgbm2",
            "krr",
            "gpr",
            "ksmooth",
            "ensemble",
            "ensemble_ridge_gbm",
            "stack",
            "portfolio",
            "multiclass_ridge",
            "multiclass_ovr_logistic",
            "multiclass_diagonal_lda",
            "multiclass_nearest_centroid",
        ]
    );
    assert_eq!(grammar.fallback_genome, "(pipe X (robust_scale) (ridge))");
    let expr = grammar.expr.as_ref().unwrap();
    assert_eq!(expr.max_col, 8);
    assert_eq!(expr.max_depth, 4);
    assert_eq!(expr.max_nodes, 15);
    assert_eq!(expr.consts.first().map(String::as_str), Some("-2"));
    assert_eq!(expr.consts.last().map(String::as_str), Some("20"));
    let value = serde_json::to_value(&grammar).unwrap();
    assert!(value["transforms"].is_object());
    assert!(value["heads"].is_object());
    assert_eq!(
        value["transforms"]["quantile_bins"]["params"][0]["bins"],
        json!([4, 8, 16, 32])
    );
}

#[test]
fn transform_bins_and_expression_defaults_match_python_spec() {
    let grammar = grammar();
    let quantile = grammar.op("quantile_bins").unwrap();
    assert_eq!(quantile.params[0].bins, bins![4, 8, 16, 32]);
    assert!(quantile.allow_append);
    let map = grammar.op("map").unwrap();
    assert!(map.takes_expr);
    assert_eq!(map.param("expr").unwrap().default, Some(json!("tanh(x)")));
    assert!(map.param("expr").unwrap().required);
    let derive = grammar.op("derive").unwrap();
    assert_eq!(
        derive.param("expr").unwrap().default,
        Some(json!("row_mean"))
    );
    assert!(grammar.op("lag").is_none());
    assert!(grammar.op("roll").is_none());
    assert!(grammar.op("target").is_none());
    assert!(grammar.op("sym").is_none());
}

#[test]
fn sampling_and_task_restrictions_match_featmath_heads() {
    let grammar = grammar();
    let gpr = grammar.op("gpr").unwrap();
    assert!(gpr.supports_task("regression"));
    assert!(gpr.supports_task("timeseries"));
    assert!(!gpr.supports_task("binary"));
    let multiclass = [
        "multiclass_ridge",
        "multiclass_ovr_logistic",
        "multiclass_diagonal_lda",
        "multiclass_nearest_centroid",
    ];
    for name in multiclass {
        let head = grammar.op(name).unwrap();
        assert!(head.supports_task("multiclass"));
        assert!(!head.supports_task("regression"));
    }
    assert!(!grammar.op("ridge_gcv").unwrap().generate);
    assert!(!grammar.op("portfolio").unwrap().generate);
    assert!(!grammar.op("lag").is_some_and(|op| op.generate));
}
