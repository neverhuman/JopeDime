//! Nothing that ships may reveal how a program was found. Every emitter, on a fixture set that
//! covers every head family and both program forms, is linted with `lint_shipped_text`; the
//! manifest's keys and non-program values are linted too. A leak anywhere here fails CI.
//!
//! The manifest's `program` (and legacy `dsl`) values are the program itself and are exempt: the
//! program text legitimately names registry members that the lint bans elsewhere.

use feat_math::transpile::{
    emit, emit_cpp17, emit_javascript, lint_shipped_phrases, lint_shipped_text, supports_cpp17,
    supports_javascript, EmitOptions, EMITTER_VERSION, SHIPPED_TEXT_BANNED,
};
use feat_math::GenomeIr;

const PIPELINES: &[&str] = &[
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

fn every_fixture() -> Vec<String> {
    let mut all: Vec<String> = PIPELINES
        .iter()
        .chain(SYMBOLIC.iter())
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
                emitted += 1;
            }
            if supports_javascript(&genome).is_ok() {
                let js = emit_javascript(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
                assert_clean(&format!("javascript of {src}"), &js);
                emitted += 1;
            }
        }
    }
    assert!(emitted >= 60, "only {emitted} artifacts were checked");
    assert_eq!(EMITTER_VERSION, "2.2.0");
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
