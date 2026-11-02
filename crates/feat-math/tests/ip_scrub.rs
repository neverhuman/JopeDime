//! Nothing that ships may reveal how a program was found. Every emitter, on a fixture set that
//! covers every head family and both program forms, is linted with `lint_shipped_text`; the
//! manifest's keys and non-program values are linted too. A leak anywhere here fails CI.
//!
//! The manifest's `program` (and legacy `dsl`) values are the program itself and are exempt: the
//! program text legitimately names registry members that the lint bans elsewhere.
//!
//! The schema-bound emitters (`emit_bound`: Python, Rust and a manifest for a fitted v3 model;
//! `emit_fitted_cpp17`: C++17 source and codec) are covered too, on fitted regression and binary
//! models.
//!
//! Every program whose artifact bytes `tests/compat_freeze.rs` freezes is in the fixture set (plan
//! N6: new emitted-artifact fixtures are linted here).

use feat_math::advanced_v3::{
    emit_bound, emit_fitted_cpp17, fit_table, ColumnSpec, DataSchema, Objective, RawTable,
    RowSemantics, TaskSpec,
};
use feat_math::transpile::{
    emit, emit_cpp17, emit_javascript, lint_shipped_phrases, lint_shipped_text, supports_cpp17,
    supports_javascript, EmitOptions, EMITTER_VERSION, SHIPPED_TEXT_BANNED,
};
use feat_math::GenomeIr;

const PIPELINES: &[&str] = &[
    "(pipe X (portfolio preset=core))",
    "(pipe X (standardize) (rank) (blom) (quantile_bins) (bin_onehot bins=3) (ridge))",
    "(pipe X (hinge) (winsor_clip) (cheb) (poly_interact append=true) (elasticnet))",
    "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (histgbm rounds=5))",
    "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (ensemble rounds=5))",
    "(pipe X (sigmoid) (norm_row) (polar) (lag k=1) (roll k=2) (fft_mag k=2) (ridge))",
    "(pipe X (row_spectral) (row_diff_stats) (svd_project k=2 iters=5) (nystrom k=4) (ridge))",
    "(pipe X (map expr=\"clip(x, -1, 1)\") (derive expr=\"row_l1 + row_l2 + row_min + row_max\") (ridge))",
    "(pipe X (histgbm2 leaves=15 lr=0.08 bags=2))",
    "(pipe X (standardize) (krr m=200))",
    "(pipe X (gpr m=200 steps=10))",
    "(pipe X (gpr kernel=rq+linear m=200 steps=10))",
    "(pipe X (gpr kernel=rbf*periodic m=200 steps=10))",
    "(pipe X (ksmooth k=16 gamma=2.0))",
    "(pipe X (standardize) (cheb degree=3 append=true) (ridge_gcv))",
    "(pipe X (stack))",
];

const SYMBOLIC: &[&str] = &[
    "(sym affine \"tanh(c0) * sqrt(abs(c1)) + min(c2, 1) / max(c3, 1) + sin(c4) + cos(c0)\")",
    "(sym raw \"sigmoid(c0) + 0.5*c1\")",
];

/// The other programs whose artifact bytes `tests/compat_freeze.rs` freezes (its
/// `NAMED_PROGRAMS`, verbatim; compat_freeze fails when one of them is missing here).
const COMPAT_FREEZE_PROGRAMS: &[&str] = &[
    "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))",
    "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))",
    "(pipe X (map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\") (ridge))",
    "(pipe X (derive append=true expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\") (ridge))",
    "(sym affine \"sigmoid(col(0)) + 0.5*col(1) - log1p(abs(col(2))) + max(c3, c4^2)\")",
    "(sym raw \"clip(row_mean + 0.3*row_std - c0/(c1 + 5), -2, 2)\")",
    "(sym affine \"step(c0, 0.0) + step(c1, c2) - 0.5\")",
    "(sym affine \"erf(c0) + asinh(c1) - exp(clip(c2, -1, 1)) + sign(c3) * round(c4)\")",
    "(pipe X (map expr=\"abs(x) + sqrt(abs(c0)) - log1p(abs(x))\" append=true) (derive expr=\"sin(c0) + cos(c1) * tanh(c2)\" append=true) (ridge))",
    "(pipe X (map expr=\"sigmoid(x) - min(x, c1) + max(x, c2)\") (derive expr=\"clip(c0, -1, 1) + step(c1, 0.5)\" append=true) (ridge))",
    "(pipe X (map expr=\"exp(clip(x, -3, 3)) + log(abs(x) + 1)\" append=true) (derive expr=\"asinh(c0) + atan(c1) - erf(c2)\" append=true) (ridge))",
    "(pipe X (map expr=\"sign(x) * floor(x) + round(x)\") (derive expr=\"row_sum - row_l1 + row_l2\" append=true) (ridge))",
    "(pipe X (map expr=\"x - row_mean + row_std\" append=true) (derive expr=\"row_min * row_max + c0 / (c1 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - (-x) * 0.5 + x / (c0 + 2)\") (derive expr=\"c3^3 - c4*c4*c4 + c2^0.5\" append=true) (elasticnet alpha=0.005 l1_ratio=0.7))",
    "(pipe X (symlog c=0.7) (ridge))",
    "(pipe X (select_corr k=3 by=corr) (ridge))",
    "(pipe X (impute strategy=median add_indicator=true) (standardize) (ridge))",
    "(pipe X (pair_ratio topk=4 append=true) (ridge))",
    "(pipe X (spline knots=4) (ridge))",
    "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (ridge))",
    "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (sigmoid) (ridge))",
    "(pipe X (step t=0.5 append=true) (norm_row) (row_diff_stats) (elasticnet))",
    "(pipe X (standardize) (logistic))",
    "(pipe X (target kind=rank_gauss) (ridge))",
    "(pipe X (target kind=log1p) (ridge lambda_frac=0.02))",
    "(pipe X (target kind=winsor q=0.05) (standardize) (ridge))",
    "(pipe X (target kind=zscore) (standardize) (ridge))",
    "(pipe X (step t=0.0) (impute strategy=mean add_indicator=true) (ridge))",
    "(pipe X (step t=0.0 append=true) (impute strategy=const value=0.5) (ridge))",
    "(pipe X (select_corr k=3 by=var) (ridge))",
    "(pipe X (rff_cos n=96 seed=3) (ridge_gcv))",
    "(pipe X (select_corr k=1) (pair_ratio topk=4) (ridge))",
    "(pipe X (derive expr=\"x\" append=true) (ridge))",
    "(sym raw \"x\")",
];

fn every_fixture() -> Vec<String> {
    let mut all: Vec<String> = PIPELINES
        .iter()
        .chain(SYMBOLIC.iter())
        .chain(COMPAT_FREEZE_PROGRAMS.iter())
        .map(|s| (*s).to_owned())
        .collect();
    all.push(feat_math::CHAMPION_GENOME.to_owned());
    all.push(
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/emitted/portable-4lang.dsl"
        ))
        .expect("portable-4lang fixture")
        .trim()
        .to_owned(),
    );
    all
}

fn option_sets() -> Vec<EmitOptions> {
    vec![
        EmitOptions::default(),
        EmitOptions {
            title: Some("Tabular model".to_owned()),
            context: Some(
                "Fitted on the caller's training rows; predicts the target column.".to_owned(),
            ),
            feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            ..Default::default()
        },
    ]
}

fn assert_clean(what: &str, text: &str) {
    let hits = lint_shipped_text(text);
    assert!(hits.is_empty(), "{what}: banned vocabulary {hits:?}");
}

fn numeric_table(x: &[Vec<f64>]) -> (DataSchema, RawTable) {
    let features = x[0].len();
    let schema = DataSchema {
        columns: (0..features)
            .map(|j| ColumnSpec::numeric(format!("c{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let headers = (0..features).map(|j| format!("c{j}")).collect();
    let cells = x
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    (schema, RawTable::new(headers, cells).unwrap())
}

#[path = "ip_scrub/compatibility.rs"]
mod compatibility;
#[path = "ip_scrub/emitters.rs"]
mod emitters;
#[path = "ip_scrub/names.rs"]
mod names;
#[path = "ip_scrub/phrases.rs"]
mod phrases;
