//! Opt-in fitted inference export; recipe emission and legacy defaults stay separate.
use crate::{
    emit::{self, Language},
    evidence, run, RunOptions,
};
use feat_math::{emit_fitted_cpp17, fit, fit_legacy_numeric_bound, GenomeIr};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn rows() -> Vec<Vec<f64>> {
    (0..12)
        .map(|i| {
            let a = i as f64 * 0.25;
            vec![a, i as f64 - 2.0, a * a + 0.3 * a + 0.1]
        })
        .collect()
}
fn xy(rows: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    rows.iter()
        .map(|row| (row[..row.len() - 1].to_vec(), *row.last().unwrap()))
        .unzip()
}
fn fitted(directory: &Path, genome: &str) -> emit::Emitted {
    emit::emit_with_training(genome, &[Language::Cpp17Fitted], directory, &rows()).unwrap()
}
fn build(directory: &Path) -> PathBuf {
    let binary = directory.join("predictor");
    let result = Command::new("g++")
        .current_dir(directory)
        .args([
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-DJOPE_STANDALONE",
            "cpp17-fitted/standalone.cpp",
            "-o",
        ])
        .arg(&binary)
        .output()
        .expect("strict C++17 compiler is required");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    binary
}
fn predict(binary: &Path, directory: &Path) -> Vec<u64> {
    predict_on(
        binary,
        directory,
        &[vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]],
    )
}
fn predict_on(binary: &Path, directory: &Path, x: &[Vec<f64>]) -> Vec<u64> {
    let input = directory.join("features.csv");
    let output = directory.join("predictions.csv");
    let mut csv = (0..x[0].len())
        .map(|i| format!("c{i}"))
        .collect::<Vec<_>>()
        .join(",")
        + "\n";
    for row in x {
        csv.push_str(&row.iter().map(f64::to_string).collect::<Vec<_>>().join(","));
        csv.push('\n');
    }
    fs::write(&input, csv).unwrap();
    let result = Command::new(binary)
        .args([&input, &output])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(output).unwrap();
    assert_eq!(text.lines().next(), Some("prediction"));
    text.lines()
        .skip(1)
        .map(|line| line.parse::<f64>().unwrap().to_bits())
        .collect()
}
fn expected(genome: &str) -> Vec<u64> {
    let (x, y) = xy(&rows());
    fit(&GenomeIr::parse(genome).unwrap(), &x, &y)
        .unwrap()
        .predict(&vec![vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]])
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect()
}

#[test]
fn fitted_target_is_opt_in_and_defaults_stay_legacy() {
    assert_eq!(Language::parse_list("all").unwrap(), Language::ALL);
    assert_eq!(RunOptions::new("unused", "unused").languages, Language::ALL);
    assert_eq!(
        Language::parse_list("cpp17-fitted,cpp17-fitted").unwrap(),
        vec![Language::Cpp17Fitted]
    );
    let both = Language::parse_list("cpp17-fitted,all").unwrap();
    assert_eq!(both[..5], Language::ALL);
    assert_eq!(both[5], Language::Cpp17Fitted);
}

#[test]
fn fitted_target_requires_training_before_publication() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path().join("missing-training");
    let error = emit::emit("(pipe X (ridge))", &[Language::Cpp17Fitted], &directory)
        .err()
        .unwrap();
    assert!(error.to_string().contains("training"), "{error:#}");
    assert!(!directory.exists());
}

#[test]
fn malformed_training_is_refused_before_any_artifact() {
    let scratch = tempfile::tempdir().unwrap();
    for (i, bad) in [
        vec![],
        vec![vec![]],
        vec![vec![1.0]],
        vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]],
        vec![vec![f64::NAN, 1.0]],
        vec![vec![1.0, f64::INFINITY]],
    ]
    .iter()
    .enumerate()
    {
        let directory = scratch.path().join(i.to_string());
        assert!(emit::emit_with_training(
            "(pipe X (ridge))",
            &[Language::Cpp17Fitted],
            &directory,
            bad
        )
        .is_err());
        assert!(!directory.exists(), "invalid training published artifacts");
    }
}

#[test]
fn legacy_recipe_bytes_stay_identical_when_fitted_is_requested() {
    let scratch = tempfile::tempdir().unwrap();
    let legacy = scratch.path().join("legacy");
    let mixed = scratch.path().join("mixed");
    let passthrough = scratch.path().join("passthrough");
    let genome = "(pipe X (ridge))";
    let old = emit::emit(genome, &Language::ALL, &legacy).unwrap();
    let languages = Language::parse_list("all,cpp17-fitted").unwrap();
    let new = emit::emit_with_training(genome, &languages, &mixed, &rows()).unwrap();
    emit::emit_with_training(genome, &Language::ALL, &passthrough, &[]).unwrap();
    for artifact in &old.artifacts {
        assert_eq!(
            fs::read(legacy.join(&artifact.path)).unwrap(),
            fs::read(mixed.join(&artifact.path)).unwrap()
        );
        assert_eq!(
            fs::read(legacy.join(&artifact.path)).unwrap(),
            fs::read(passthrough.join(&artifact.path)).unwrap()
        );
    }
    assert_eq!(
        fs::read(legacy.join("manifest.json")).unwrap(),
        fs::read(mixed.join("manifest.json")).unwrap()
    );
    assert_eq!(
        fs::read(legacy.join("targets.txt")).unwrap(),
        fs::read(passthrough.join("targets.txt")).unwrap()
    );
    assert_eq!(new.artifacts.len(), old.artifacts.len() + 3);
    assert!(new.exclusions.is_empty());
    assert!(mixed.join("model.cpp").is_file());
    assert!(mixed.join("cpp17-fitted/model.cpp").is_file());
    assert_eq!(
        fs::read_to_string(mixed.join("targets.txt")).unwrap(),
        "py rs cpp js cpp17-fitted\n"
    );
}

#[test]
fn fitted_exports_execute_expression_and_head_families_bit_exactly() {
    let scratch = tempfile::tempdir().unwrap();
    for (i, genome) in [
        "(sym raw \"c0+c1\")",
        "(pipe X (map expr=\"x+c0\") (ridge))",
        "(pipe X (derive expr=\"c0+c1\") (ridge))",
        "(pipe X (histgbm rounds=3 depth=2 bins=8 min_leaf=1))",
        "(pipe X (histgbm2 rounds=8 leaves=4 min_leaf=1 bags=2))",
        "(pipe X (gpr m=12 steps=1))",
        "(pipe X (ksmooth k=3 gamma=1))",
        "(pipe X (target_bin_encode bins=4 prior=3) (ridge))",
    ]
    .iter()
    .enumerate()
    {
        let directory = scratch.path().join(i.to_string());
        let emitted = fitted(&directory, genome);
        assert_eq!(
            emitted.canonical_genome,
            GenomeIr::parse(genome).unwrap().canonical()
        );
        assert!(emitted.exclusions.is_empty());
        let (x, y) = xy(&rows());
        let reference = emit_fitted_cpp17(
            &fit_legacy_numeric_bound(&GenomeIr::parse(genome).unwrap(), &x, &y).unwrap(),
        )
        .unwrap();
        assert_eq!(
            fs::read(directory.join("cpp17-fitted/model.cpp")).unwrap(),
            reference.source.as_bytes()
        );
        assert_eq!(
            fs::read(directory.join("cpp17-fitted/model.codec.json")).unwrap(),
            reference.codec.as_bytes()
        );
        assert_eq!(
            predict(&build(&directory), &directory),
            expected(genome),
            "{genome}"
        );
    }
}

#[test]
fn generated_expression_child_exports_match_fitted_evaluator() {
    use feat_math::kernel::{
        profile::{shipped_kernel2_profiles, PipelineArm},
        search::{admit_child, mutate_program},
    };
    let profile = shipped_kernel2_profiles()
        .into_iter()
        .find(|p| p.id == "kernel2-baseline")
        .unwrap();
    let child = (0..512)
        .map(|seed| mutate_program("(pipe X (ridge))", &profile, 2, seed).unwrap())
        .find(|child| child.arm == PipelineArm::ExprGrow && child.changed)
        .expect("expression-growth witness");
    let admitted = admit_child(&child.program, &profile, 2).unwrap();
    assert_eq!(admitted.canonical(), child.program);
    let scratch = tempfile::tempdir().unwrap();
    fitted(scratch.path(), &child.program);
    assert_eq!(
        predict(&build(scratch.path()), scratch.path()),
        expected(&child.program)
    );
}

#[test]
fn standalone_refusals_preserve_input_and_existing_predictions() {
    let scratch = tempfile::tempdir().unwrap();
    fitted(scratch.path(), "(pipe X (ridge))");
    let binary = build(scratch.path());
    predict(&binary, scratch.path());
    let output = scratch.path().join("predictions.csv");
    let saved = fs::read(&output).unwrap();
    let bad = scratch.path().join("bad.csv");
    for input in [
        "a,b\n1,2,\n",
        "a,b\n1,2\n\n",
        "a\n1\n",
        "a,b\nNaN,1\n",
        "a,b\nInf,1\n",
    ] {
        fs::write(&bad, input).unwrap();
        assert!(!Command::new(&binary)
            .args([&bad, &output])
            .status()
            .unwrap()
            .success());
        assert_eq!(fs::read(&output).unwrap(), saved);
    }
    fs::write(&bad, "a,b\n1,2\n").unwrap();
    let before = fs::read(&bad).unwrap();
    assert_eq!(
        Command::new(&binary)
            .args([&bad, &bad])
            .status()
            .unwrap()
            .code(),
        Some(10)
    );
    assert_eq!(fs::read(&bad).unwrap(), before);
}

#[test]
fn pipeline_export_refits_selected_genome_on_training_partition_and_verifies_receipt() {
    let scratch = tempfile::tempdir().unwrap();
    let dataset =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test/pmlb-229-pwlinear");
    let directory = scratch.path().join("run");
    let mut options = RunOptions::new(dataset.to_str().unwrap(), &directory);
    options.languages = vec![Language::Cpp17Fitted];
    options.policy = crate::Policy::Skip;
    options.budget = crate::search::Budget::parse("evaluations=1,seconds=60").unwrap();
    options.source_commit = Some("0".repeat(40));
    let receipt = run(&options).unwrap();
    assert!(receipt.ok);
    assert_eq!(receipt.artifacts.len(), 3);
    let verified = evidence::verify_champion_run(&directory.join("receipt.json")).unwrap();
    let resolved = crate::dataset::resolve(dataset.to_str().unwrap()).unwrap();
    let training = crate::dataset::training_rows(&resolved).unwrap();
    let (x, y) = xy(&training);
    let genome = GenomeIr::parse(&verified.receipt.search.selected_genome).unwrap();
    let expected = emit_fitted_cpp17(&fit_legacy_numeric_bound(&genome, &x, &y).unwrap()).unwrap();
    assert_eq!(
        fs::read(directory.join("emit/cpp17-fitted/model.cpp")).unwrap(),
        expected.source.as_bytes()
    );
    assert_eq!(
        fs::read(directory.join("emit/cpp17-fitted/model.codec.json")).unwrap(),
        expected.codec.as_bytes()
    );
    let test_x = x[..3].to_vec();
    let predictions = fit(&genome, &x, &y)
        .unwrap()
        .predict(&test_x)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        predict_on(
            &build(&directory.join("emit")),
            &directory.join("emit"),
            &test_x
        ),
        predictions
    );
    fs::write(directory.join("emit/cpp17-fitted/model.codec.json"), "{}").unwrap();
    assert!(evidence::verify_champion_run(&directory.join("receipt.json")).is_err());
}
