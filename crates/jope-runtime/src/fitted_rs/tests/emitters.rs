use super::*;

#[test]
fn fitted_module_docs_compile_and_run_as_a_plain_rust_module() {
    let (genome, model) = fitted("(pipe X (standardize) (ridge))");
    let source = emit(&genome, &model, 12).unwrap();
    assert!(source.starts_with("//! Fitted, std-only Rust prediction module"));
    assert!(source.contains("`mod predictor;`"));
    assert!(source.contains("predictor::predict_numeric(&rows)"));
    assert!(source.contains("centers each column by its training mean"));
    assert!(!source.contains("train.csv"));
    assert!(!source.contains("test_features.csv"));
    assert!(!source.contains("Usage: ./model"));
    assert!(!source.contains("fn main()"));

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-rs-proof/module-usage");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("predictor.rs"), source).unwrap();
    let consumer = dir.join("consumer.rs");
    std::fs::write(
            &consumer,
            "mod predictor;\nfn main() {\n    let rows = vec![vec![0.5; 12]];\n    let predictions = predictor::predict_numeric(&rows).expect(\"fitted prediction\");\n    assert_eq!(predictions.len(), 1);\n    assert!(predictions[0].is_finite());\n}\n",
        )
        .unwrap();
    let executable = dir.join("consumer");
    let compile = Command::new("rustc")
        .args(["--edition=2021"])
        .arg(&consumer)
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "plain rustc consumer compile failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let output = Command::new(executable).output().unwrap();
    assert!(
        output.status.success(),
        "plain rustc consumer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn numeric_case(index: usize, dsl: &str, test: &[Vec<f64>]) -> String {
    let (genome, model) = fitted(dsl);
    let expected = model.predict(&test.to_vec()).unwrap();
    let source = emit(&genome, &model, 12).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
    assert_eq!(
        source,
        emit(&genome, &model, 12).unwrap(),
        "deterministic {dsl}"
    );
    assert!(!source.contains("fn main()"));
    let rows = format!(
        "vec![{}]",
        test.iter()
            .map(|row| format!("vec!{row:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let expected = format!("vec!{expected:?}");
    format!("mod c{index} {{\n{source}\n}}\nfn check{index}() {{ let rows = {rows}; let expected: Vec<f64> = {expected}; let got = c{index}::predict_numeric(&rows).unwrap(); assert_eq!(got.len(), expected.len()); for (i,(a,b)) in got.iter().zip(&expected).enumerate() {{ assert!((a-b).abs() <= 1e-12+1e-10*b.abs(), \"case {index} row {{i}}: {{a}} vs {{b}}\"); }} assert!(c{index}::predict_numeric(&[]).unwrap().is_empty()); assert!(c{index}::predict_numeric(&[vec![0.0;11]]).is_err()); assert!(c{index}::predict_numeric(&[vec![f64::NAN;12]]).is_err()); }}\n")
}

#[test]
fn rustc_replays_every_legacy_recipe_operation_head_and_target() {
    // Inventory is cross-checked against EXP-01 below. Every legacy recipe branch occurs.
    let operations = [
        "standardize",
        "robust_scale",
        "rank",
        "blom",
        "quantile_bins bins=4",
        "bin_onehot bins=3",
        "hinge knots=3",
        "winsor_clip",
        "cheb degree=3",
        "poly_interact",
        "rff_cos n=5",
        "rff_sin n=5",
        "rand_proj k=4",
        "landmark_dist k=4 gamma=0.8",
        "oblique_hinge n=4",
        "row_stats",
        "pair_prod topk=4",
        "target_bin_encode bins=4",
        "sigmoid",
        "step t=0.5",
        "norm_row",
        "polar",
        "lag k=2",
        "roll k=2",
        "fft_mag k=4",
        "row_spectral",
        "row_diff_stats",
        "svd_project k=4 iters=3",
        "nystrom k=4 gamma=0.8",
        "symlog c=0.4",
        "select_corr k=4",
        "impute strategy=mean add_indicator=true",
        "pair_ratio topk=4",
        "spline knots=3",
        "map expr=\"sin(x)\"",
        "derive expr=\"c0*c1\"",
    ];
    let heads = [
        "ridge",
        "ridge_gcv",
        "elasticnet sweeps=4",
        "gbm rounds=4 depth=2 min_leaf=2",
        "histgbm2 rounds=4 leaves=4 min_leaf=2",
        "ensemble rounds=4 min_leaf=2",
        "krr m=50",
        "gpr m=50 steps=1",
        "gpr m=50 steps=1 kernel=\"rbf+linear\"",
        "ksmooth k=4 gamma=0.8",
        "stack",
        "portfolio preset=core",
    ];
    let mut dsls = vec!["(pipe X (ridge))".to_owned()];
    // Test both replacement and appending, because the recipe carries that flag, not state.
    for op in operations {
        for append in [false, true] {
            dsls.push(format!("(pipe X ({op} append={append}) (ridge))"));
        }
    }
    dsls.extend(heads.into_iter().map(|h| format!("(pipe X ({h}))")));
    dsls.extend(
        ["rank_gauss", "log1p", "winsor", "zscore"]
            .map(|kind| format!("(pipe X (target kind={kind}) (standardize) (ridge))")),
    );
    dsls.push(
        "(pipe X (robust_scale append=true) (hinge knots=3 append=true) (elasticnet sweeps=4))"
            .into(),
    );
    dsls.extend(
        [
            "(pipe X (norm_features) (ridge))",
            "(pipe X (winsor_moments) (ridge))",
            "(pipe X (ensemble_ridge_gbm rounds=4 min_leaf=2))",
            "(pipe X (histgbm rounds=4 min_leaf=2))",
            "(pipe X (logistic))",
        ]
        .map(str::to_owned),
    );
    let mut covered = std::collections::BTreeSet::new();
    for dsl in &dsls {
        if let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(dsl).unwrap() {
            covered.extend(pipeline.stages.into_iter().map(|s| s.op));
        }
    }
    let registry = feat_math::evolve::api::capability::Registry::current();
    use feat_math::evolve::api::capability::{CapabilityKind, Language};
    let admitted: Vec<_> = registry
        .entries
        .iter()
        .filter(|row| {
            matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head)
                && row.supported[&Language::Rust]
        })
        .map(|row| row.name)
        .collect();
    eprintln!("Rust registry: {admitted:?}");
    let missing: Vec<_> = admitted
        .iter()
        .filter(|name| !covered.contains(**name))
        .collect();
    assert!(missing.is_empty(), "registry cases missing: {missing:?}");
    let (mut test, _) = training();
    test.extend([
        vec![-0.0; 12],
        vec![0.0; 12],
        vec![1.5; 12],
        vec![-0.5; 12],
        vec![0.5; 12],
    ]);
    let mut program = String::new();
    for (i, dsl) in dsls.iter().enumerate() {
        program.push_str(&numeric_case(i, dsl, &test));
    }
    program.push_str("fn main() {\n");
    for i in 0..dsls.len() {
        program.push_str(&format!("check{i}();\n"));
    }
    program.push_str(&format!(
        "println!(\"{} recipes, {} prediction rows, abs1e-12+rel1e-10\"); }}",
        dsls.len(),
        dsls.len() * test.len()
    ));
    run_rust("all-recipes", &program);
}
