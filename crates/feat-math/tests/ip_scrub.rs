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

#[test]
fn every_emitter_is_clean_on_every_fixture() {
    let mut emitted = 0;
    let (mut emitted_cpp, mut emitted_js) = (0, 0);
    for src in every_fixture() {
        let genome = GenomeIr::parse(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
        for opts in option_sets() {
            let artifact = emit(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_clean(&format!("python of {src}"), &artifact.python);
            assert_clean(&format!("rust of {src}"), &artifact.rust);
            emitted += 2;

            // Manifest: every key, and every value except the program text itself.
            let manifest: serde_json::Value = serde_json::from_str(&artifact.manifest).unwrap();
            let object = manifest.as_object().expect("manifest is an object");
            for (key, value) in object {
                if key != "dsl" && key != "genome_sha256" {
                    assert_clean(&format!("manifest key {key:?} of {src}"), key);
                }
                if key != "program" && key != "dsl" {
                    assert_clean(
                        &format!("manifest value {key:?} of {src}"),
                        &value.to_string(),
                    );
                }
            }
            assert_eq!(
                object["program"], object["dsl"],
                "{src}: program mirrors dsl"
            );
            assert_eq!(
                object["program_sha256"], object["genome_sha256"],
                "{src}: program_sha256 mirrors genome_sha256"
            );
            assert_eq!(object["program"].as_str().unwrap(), genome.canonical());

            if supports_cpp17(&genome).is_ok() {
                let cpp = emit_cpp17(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
                assert_clean(&format!("c++17 of {src}"), &cpp);
                emitted_cpp += 1;
            }
            if supports_javascript(&genome).is_ok() {
                let js = emit_javascript(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
                assert_clean(&format!("javascript of {src}"), &js);
                emitted_js += 1;
            }
        }
    }
    // Python + Rust on every fixture under both option sets, plus every C++ and JavaScript
    // artifact the gates admit: the four-language fixture alone yields two of each.
    assert!(
        emitted >= 2 * 2 * 20,
        "only {emitted} Python/Rust artifacts were checked"
    );
    assert!(
        emitted_cpp >= 2,
        "only {emitted_cpp} C++17 artifacts were checked"
    );
    assert!(
        emitted_js >= 2,
        "only {emitted_js} JavaScript artifacts were checked"
    );
    assert_eq!(EMITTER_VERSION, "2.3.0");
}

/// Numeric schema `c0..` and a shortest-round-trip string table, as the campaign lanes build them.
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

#[test]
fn bound_and_fitted_cpp_emitters_are_clean() {
    let mut rng = feat_math::Pcg32::new(17, 3);
    let mut x = Vec::new();
    let mut y_regression = Vec::new();
    let mut y_binary = Vec::new();
    for _ in 0..160 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        x.push(vec![a, b, c]);
        y_regression.push(2.0 * a - b + 0.3 * c);
        y_binary.push(if 3.0 * (a - b) + (c - 0.5) > 0.0 {
            1.0
        } else {
            0.0
        });
    }
    let (schema, table) = numeric_table(&x);
    let regression = TaskSpec::regression(Objective::R2).unwrap();
    let binary = TaskSpec::binary(Objective::RocAuc).unwrap();
    let cases = [
        ("(pipe X (ridge))", &regression, &y_regression),
        (
            "(pipe X (standardize) (tab_boost))",
            &regression,
            &y_regression,
        ),
        ("(pipe X (logit_ridge))", &binary, &y_binary),
        ("(pipe X (tab_boost))", &binary, &y_binary),
        ("(pipe X (extra_trees))", &binary, &y_binary),
        ("(pipe X (gam_boost))", &binary, &y_binary),
        ("(pipe X (portfolio))", &binary, &y_binary),
    ];
    let (mut bound, mut fitted) = (0, 0);
    for (src, task, y) in cases {
        let genome = GenomeIr::parse(src).unwrap();
        let model =
            fit_table(&genome, task, &schema, &table, y).unwrap_or_else(|e| panic!("{src}: {e:#}"));
        let artifact = emit_bound(&model, &EmitOptions::default())
            .unwrap_or_else(|e| panic!("bound {src}: {e:#}"));
        assert_clean(&format!("bound python of {src}"), &artifact.python);
        assert_clean(&format!("bound rust of {src}"), &artifact.rust);
        assert_clean(&format!("bound manifest of {src}"), &artifact.manifest);
        let manifest: serde_json::Value = serde_json::from_str(&artifact.manifest).unwrap();
        assert_eq!(manifest["emitter_version"], EMITTER_VERSION);
        assert!(
            manifest.get("program_sha256").is_some(),
            "{src}: manifest names the program hash"
        );
        bound += 1;
        match emit_fitted_cpp17(&model) {
            Ok(cpp) => {
                assert_clean(&format!("fitted c++17 of {src}"), &cpp.source);
                assert_clean(&format!("fitted c++17 codec of {src}"), &cpp.codec);
                fitted += 1;
            }
            Err(error) => eprintln!("fitted C++17 not supported for {src}: {error:#}"),
        }
    }
    assert_eq!(bound, 7);
    assert_eq!(
        fitted, 7,
        "every one of the seven heads emits fitted C++17 today; a head losing it is a change to review"
    );
}

/// The lint checks what the emitter writes, not what the caller's data is called: columns named
/// `island` and `candidate` and categories named `population`, `tournament`, `elites` and
/// `generation` (every one a banned word) still emit through both schema-bound emitters, and the
/// artifacts carry those names verbatim.
#[test]
fn user_supplied_names_do_not_trip_the_bound_and_fitted_lints() {
    // `gp` is two bytes and innocent on its own: it must not be masked, so that it can never hide
    // an emitter-written `gp ` (the test only needs emission to succeed and the name to ship).
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("island"),
            ColumnSpec::numeric("candidate"),
            ColumnSpec::numeric("gp"),
            ColumnSpec::categorical("region"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    // Every category contains a banned word on its own; three also carry a quote, a backslash or
    // a control character, so the Python artifact embeds them JSON-quoted inside a JSON literal,
    // the Rust artifact Debug-escaped and the C++ codec JSON-escaped: every spelling the mask must
    // cover, on a name that would refuse without it.
    let categories = ["population\"", "elites\\", "generation\u{8}", "tournament"];
    let mut rng = feat_math::Pcg32::new(23, 5);
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..120 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        let region = categories[i % categories.len()];
        rows.push(vec![
            format!("{a:?}"),
            format!("{b:?}"),
            format!("{c:?}"),
            region.to_owned(),
        ]);
        y.push(if 3.0 * (a - b) + 0.2 * (i % 4) as f64 > 0.2 {
            1.0
        } else {
            0.0
        });
    }
    let table = RawTable::new(
        vec![
            "island".into(),
            "candidate".into(),
            "gp".into(),
            "region".into(),
        ],
        rows,
    )
    .unwrap();
    let binary = TaskSpec::binary(Objective::RocAuc).unwrap();
    for src in [
        "(pipe X (input numeric=median missing=indicator categorical=frequency) (logit_ridge))",
        "(pipe X (input numeric=median missing=indicator categorical=frequency) (tab_boost rounds=8 bins=8 min_leaf=1))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        let model = fit_table(&genome, &binary, &schema, &table, &y)
            .unwrap_or_else(|e| panic!("{src}: {e:#}"));
        let artifact = emit_bound(&model, &EmitOptions::default())
            .unwrap_or_else(|e| panic!("bound {src}: {e:#}"));
        for (what, text) in [("python", &artifact.python), ("rust", &artifact.rust)] {
            assert!(
                text.contains("island"),
                "{what} of {src}: the column name ships verbatim"
            );
            assert!(
                text.contains("population"),
                "{what} of {src}: the vocabulary ships verbatim"
            );
        }
        let cpp = emit_fitted_cpp17(&model).unwrap_or_else(|e| panic!("fitted {src}: {e:#}"));
        assert!(
            cpp.codec.contains("population"),
            "{src}: codec vocabulary ships verbatim"
        );
        assert!(
            cpp.codec.contains("population\\\""),
            "{src}: the quoted category ships JSON-escaped in the codec"
        );
        assert!(
            artifact.rust.contains("elites\\\\"),
            "{src}: the backslash category ships Debug-escaped in the Rust artifact"
        );
        assert!(
            artifact.python.contains("generation"),
            "{src}: the control-character category ships in the Python artifact"
        );
        // The unmasked lint would have refused every one of these artifacts, in every language.
        assert!(!lint_shipped_text(&artifact.python).is_empty());
        assert!(!lint_shipped_text(&artifact.rust).is_empty());
        assert!(!lint_shipped_text(&cpp.codec).is_empty());
    }
}

#[test]
fn the_phrase_lint_trips_on_what_it_must_and_nothing_ordinary() {
    for phrase in [
        "a linear calibration in the style of Keijzer (2003)",
        "found by GP search over programs",
        "the best of generation 0",
        "an expression tree evaluated bottom-up",
        "symbolic regression on the training rows",
        "evolved with tournament selection",
        "one island per worker",
        "Koza-style crossover",
    ] {
        let hits = lint_shipped_phrases(phrase);
        assert!(!hits.is_empty(), "{phrase:?} must trip the phrase lint");
        assert!(!lint_shipped_text(phrase).is_empty());
    }
    for ordinary in [
        "gradient boosted regression trees with depth 5",
        "the generated model reads train.csv",
        "a degenerate covariance is regularised",
        "Gaussian process regression with an RBF kernel",
        "ridge regression on standardized features",
        "grows each tree greedily on histogram bins",
        "the model was generated from your data",
    ] {
        let hits = lint_shipped_phrases(ordinary);
        assert!(
            hits.is_empty(),
            "{ordinary:?} tripped the phrase lint: {hits:?}"
        );
    }
    assert!(SHIPPED_TEXT_BANNED.contains(&"keijzer"));
    // emit() itself refuses a poisoned title through the same lint (the C++ and JavaScript
    // emitters do not embed the title, so the title cannot poison them).
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let opts = EmitOptions {
        title: Some("Generation 3 champion".to_owned()),
        ..Default::default()
    };
    let error = emit(&genome, &opts).unwrap_err().to_string();
    assert!(error.contains("banned vocabulary"), "{error}");
    assert!(error.contains("generation"), "{error}");
}

#[test]
fn the_closed_form_docstring_no_longer_cites_the_method_literature() {
    let genome = GenomeIr::parse(SYMBOLIC[0]).unwrap();
    let artifact = emit(&genome, &EmitOptions::default()).unwrap();
    for text in [&artifact.python, &artifact.rust] {
        assert!(!text.to_lowercase().contains("keijzer"));
        // Docstrings are re-wrapped and comment-prefixed by the emitter, so look for the one
        // token that survives any wrapping.
        assert!(
            text.contains("least-squares"),
            "the calibration is still explained"
        );
    }
}

#[test]
fn fitted_portable_and_public_scalar_cpp_are_clean_on_every_supported_head() {
    use feat_math::advanced_v3::{
        emit_fitted_portable, emit_fitted_public_cpp17, lint_fitted_public_text,
    };
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["amount".into(), "kind".into()],
        (0..18)
            .map(|i| vec![i.to_string(), format!("group-{}", i % 3)])
            .collect(),
    )
    .unwrap();
    for (head, task, classes) in [
        (
            "ridge",
            TaskSpec::regression(Objective::Rmse).unwrap(),
            0usize,
        ),
        (
            "logit_ridge",
            TaskSpec::binary(Objective::RocAuc).unwrap(),
            2,
        ),
        (
            "multiclass_ridge",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
        (
            "multiclass_ovr_logistic",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
        (
            "multiclass_diagonal_lda",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
        (
            "multiclass_nearest_centroid",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
    ] {
        let targets = (0..18)
            .map(|i| {
                if classes == 0 {
                    i as f64
                } else {
                    (i % classes) as f64
                }
            })
            .collect::<Vec<_>>();
        let model = fit_table(
            &GenomeIr::parse(&format!(
                "(pipe X (input numeric=median categorical=onehot) ({head}))"
            ))
            .unwrap(),
            &task,
            &schema,
            &table,
            &targets,
        )
        .unwrap();
        let output = emit_fitted_portable(&model).unwrap();
        for text in [&output.python, &output.rust]
            .into_iter()
            .chain(output.javascript.iter())
        {
            assert!(
                lint_fitted_public_text(&model, &[], text).is_empty(),
                "{head}"
            );
        }
        if classes < 3 {
            let cpp = emit_fitted_public_cpp17(&model).unwrap();
            for text in [&cpp.source, &cpp.codec] {
                assert!(
                    lint_fitted_public_text(&model, &[], text).is_empty(),
                    "{head}"
                );
            }
        }
    }
}
