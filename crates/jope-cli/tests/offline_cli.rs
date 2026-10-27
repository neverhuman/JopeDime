use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn jope(cwd: &Path, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope"))
        .current_dir(cwd)
        .env_remove("JOPE_HOME")
        .env_remove("JOPE_INPUTS_ROOT")
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn cli_generates_checks_and_reports_tampering_from_unrelated_directory() {
    let temporary = tempfile::tempdir().unwrap();
    let output = jope(
        temporary.path(),
        &["generate-fixtures", "--output", "portable-inputs", "--json"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let generated: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(generated["dataset_count"], 40);
    for arguments in [
        vec![
            "verify-inputs",
            "--offline",
            "--json",
            "--root",
            "portable-inputs",
        ],
        vec![
            "generate-fixtures",
            "--check",
            "--output",
            "portable-inputs",
            "--json",
        ],
    ] {
        let output = jope(temporary.path(), &arguments);
        assert!(output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap(),
            generated
        );
    }
    fs::write(
        temporary.path().join("portable-inputs/binary/linear.json"),
        b"corrupt",
    )
    .unwrap();
    let failed = jope(
        temporary.path(),
        &[
            "verify-inputs",
            "--offline",
            "--json",
            "--root",
            "portable-inputs",
        ],
    );
    assert!(!failed.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&failed.stdout).unwrap()["ok"],
        false
    );
}

#[test]
fn runtime_environment_and_ancestor_resolution_work_without_build_paths() {
    let temporary = tempfile::tempdir().unwrap();
    let checkout = temporary.path().join("independent-checkout");
    fs::create_dir_all(checkout.join("datasets")).unwrap();
    jope_cli::generate_fixtures(&checkout.join("datasets/synthetic")).unwrap();
    fs::create_dir_all(checkout.join("deep/nested")).unwrap();
    let output = jope(
        &checkout.join("deep/nested"),
        &["verify-inputs", "--offline", "--json"],
    );
    assert!(output.status.success());
    for (key, value) in [
        ("JOPE_HOME", checkout.clone()),
        ("JOPE_INPUTS_ROOT", checkout.join("datasets/synthetic")),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_jope"))
            .current_dir(temporary.path())
            .env_remove("JOPE_HOME")
            .env_remove("JOPE_INPUTS_ROOT")
            .env(key, value)
            .args(["verify-inputs", "--offline", "--json"])
            .output()
            .unwrap();
        assert!(output.status.success());
    }
}

#[test]
fn explicit_missing_root_and_missing_offline_flag_fail() {
    let temporary = tempfile::tempdir().unwrap();
    for arguments in [
        vec!["verify-inputs", "--offline", "--json", "--root", "missing"],
        vec!["verify-inputs", "--json"],
    ] {
        let output = jope(temporary.path(), &arguments);
        assert!(!output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap()["ok"],
            false
        );
    }
}

#[test]
fn committed_corpus_matches_regeneration() {
    // Test-only source location; the installed CLI resolves every path at runtime.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic");
    let report = jope_cli::verify_inputs(&root).unwrap();
    assert_eq!(report.dataset_count, 40);
}

#[test]
fn encode_without_weights_fails_closed() {
    let input =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic/binary/linear.json");
    let output = Command::new(env!("CARGO_BIN_EXE_jope"))
        .env_remove("JOPE_HOME")
        .env_remove("JOPE_INPUTS_ROOT")
        .env_remove("HYPERION_V7_WEIGHTS")
        .env_remove("JAIN_HYPERION_WEIGHTS")
        .env_remove("JAIN_HYPERION_MANIFEST")
        .args([
            "encode",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.encode.v1");
}

fn regression_linear() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic/regression/linear.json")
}

#[test]
fn fit_ridge_on_regression_linear_reports_finite_r2() {
    let input = regression_linear();
    let output = jope(
        Path::new("."),
        &[
            "fit",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["schema"], "jope.fit.v1");
    assert!(report["train_r2"].as_f64().unwrap().is_finite());
    assert!(report["hold_r2"].as_f64().unwrap().is_finite());
}

#[test]
fn emit_all_languages_writes_four_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let out = temporary.path().join("emitted");
    let output = jope(
        temporary.path(),
        &[
            "emit",
            "--json",
            "--lang",
            "all",
            "--output",
            out.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["artifacts"].as_array().unwrap().len(), 4);
    for name in ["model.py", "model.rs", "model.cpp", "model.js"] {
        assert!(out.join(name).is_file(), "missing {name}");
    }
}

#[test]
fn emit_cpp20_writes_the_pipeline_cpp_bytes() {
    let temporary = tempfile::tempdir().unwrap();
    let out = temporary.path().join("emitted");
    let output = jope(
        temporary.path(),
        &[
            "emit",
            "--json",
            "--lang",
            "cpp20",
            "--output",
            out.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["artifacts"].as_array().unwrap().len(), 1);
    assert_eq!(report["artifacts"][0]["lang"], "cpp20");

    // The same default program through the pipeline's emit stage, C++20 only.
    let pipeline = temporary.path().join("pipeline");
    let emitted = jope_pipeline::emit::emit(
        jope_cli::DEFAULT_GENOME,
        &[jope_pipeline::emit::Language::Cpp20],
        &pipeline,
    )
    .unwrap();
    assert!(emitted.exclusions.is_empty(), "{:?}", emitted.exclusions);
    assert_eq!(emitted.artifacts.len(), 1);
    assert_eq!(emitted.artifacts[0].language, "cpp20");
    assert_eq!(
        fs::read(out.join("model.cpp")).unwrap(),
        fs::read(pipeline.join(&emitted.artifacts[0].path)).unwrap(),
        "jope emit --lang cpp20 and the pipeline wrote different C++ bytes"
    );
    assert_eq!(
        report["artifacts"][0]["sha256"],
        emitted.artifacts[0].sha256.as_str()
    );
}

#[test]
fn search_picks_a_ridge_family_genome() {
    let input = regression_linear();
    let output = jope(
        Path::new("."),
        &[
            "search",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["schema"], "jope.search.v1");
    assert_eq!(report["considered"], 3);
    assert!(report["genome"].as_str().unwrap().contains("ridge"));
}

fn synthetic_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic")
}

#[test]
fn champions_pull_and_push_fail_closed() {
    for flag in ["--pull", "--push"] {
        let output = jope(Path::new("."), &["champions", "--json", flag]);
        assert!(!output.status.success());
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["ok"], false);
        assert_eq!(report["schema"], "jope.champions.v1");
        let diagnostic = report.to_string();
        assert!(diagnostic.contains("not implemented for this command"));
        assert!(!diagnostic.contains("no AWS credentials"));
    }
}

#[test]
fn train_step_without_local_fails_closed() {
    let output = jope(Path::new("."), &["train-step", "--json", "--env", "dev"]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.train-step.v1");
}

#[test]
fn local_champions_and_train_step_report_supported_and_unsupported_fixtures() {
    let root = synthetic_root();
    let root = root.to_str().expect("utf8 path");
    let champions = jope(
        Path::new("."),
        &["champions", "--local", "--json", "--root", root],
    );
    assert!(
        champions.status.success(),
        "{}",
        String::from_utf8_lossy(&champions.stderr)
    );
    let bank: Value = serde_json::from_slice(&champions.stdout).unwrap();
    assert_eq!(bank["ok"], true);
    assert_eq!(bank["env"], "local");
    assert_eq!(bank["dataset_count"], 40);
    assert_eq!(bank["supported_count"], 8);
    assert_eq!(bank["unsupported_count"], 32);
    let rows = bank["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 8);
    assert!(rows
        .iter()
        .all(|row| row["task"] == "regression" && row["hold_r2"].as_f64().unwrap().is_finite()));
    let mut searched: Vec<&str> = rows
        .iter()
        .map(|row| row["scenario"].as_str().unwrap())
        .collect();
    searched.sort_unstable();
    assert_eq!(
        searched,
        [
            "collinearity",
            "constant_target",
            "extrapolation",
            "heavy_tailed",
            "heteroscedastic",
            "linear",
            "nonlinear",
            "sparse_signal"
        ]
    );
    let unsupported = bank["unsupported"].as_array().unwrap();
    assert_eq!(unsupported.len(), 32);
    for task in ["binary", "time_series", "multiclass"] {
        let excluded: Vec<_> = unsupported
            .iter()
            .filter(|row| row["task"] == task)
            .collect();
        assert_eq!(excluded.len(), 10, "{task}");
        let reason = format!("unsupported task {task}");
        assert!(excluded
            .iter()
            .all(|row| row["reason"].as_str().unwrap().contains(&reason)));
    }
    let mut regression: Vec<(&str, &str)> = unsupported
        .iter()
        .filter(|row| row["task"] == "regression")
        .map(|row| {
            (
                row["scenario"].as_str().unwrap(),
                row["reason"].as_str().unwrap(),
            )
        })
        .collect();
    regression.sort_unstable();
    assert_eq!(regression.len(), 2);
    for ((scenario, reason), expected) in regression.iter().zip(["categorical", "missing_values"]) {
        assert_eq!(*scenario, expected);
        assert!(reason.contains("integer, non-null"), "{scenario}: {reason}");
    }

    let step = jope(
        Path::new("."),
        &["train-step", "--local", "--json", "--root", root],
    );
    assert!(
        step.status.success(),
        "{}",
        String::from_utf8_lossy(&step.stderr)
    );
    let report: Value = serde_json::from_slice(&step.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["dataset_count"], 40);
    assert_eq!(report["supported_count"], 8);
    assert_eq!(report["unsupported_count"], 32);
    assert_eq!(report["labels"], bank["rows"]);
    assert_eq!(report["unsupported"], bank["unsupported"]);
    assert_eq!(report["router"]["dim"], 128);
    assert_eq!(
        report["router"]["classes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|class| class["n"].as_u64().unwrap())
            .sum::<u64>(),
        8
    );
    assert!(report["note"]
        .as_str()
        .unwrap()
        .contains("does not update jope-policy"));
}

fn test_router(directory: &Path) -> std::path::PathBuf {
    let router = directory.join("router.json");
    fs::write(
        &router,
        serde_json::to_vec(&serde_json::json!({
            "schema": "jope.router.centroid.v1",
            "dim": 128,
            "classes": [{"genome": "(pipe X (ridge))", "n": 1, "centroid": vec![0.0; 128]}]
        }))
        .unwrap(),
    )
    .unwrap();
    router
}

fn assert_command_rejects(command: &str, input: &Path, router: &Path, reason: &str) {
    let mut args = vec![command, "--json", "--input", input.to_str().unwrap()];
    if command == "route" {
        args.extend(["--router", router.to_str().unwrap()]);
    }
    let output = jope(Path::new("."), &args);
    assert!(
        !output.status.success(),
        "{command} {} unexpectedly succeeded",
        input.display()
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert!(
        report["error"].as_str().unwrap().contains(reason),
        "{command} {}: {report}",
        input.display()
    );
}

#[test]
fn regression_commands_refuse_other_tasks_and_unsupported_features() {
    let temporary = tempfile::tempdir().unwrap();
    let router = test_router(temporary.path());
    // A valid regression fixture whose second feature is declared nullable (values unchanged).
    let nullable = temporary.path().join("nullable.json");
    let mut data: Value = serde_json::from_slice(&fs::read(regression_linear()).unwrap()).unwrap();
    data["columns"][1]["nullable"] = Value::Bool(true);
    fs::write(&nullable, serde_json::to_vec(&data).unwrap()).unwrap();
    let mut cases = vec![(nullable, "integer, non-null")];
    for (fixture, reason) in [
        ("binary/linear.json", "unsupported task binary"),
        (
            "multiclass/three_classes.json",
            "unsupported task multiclass",
        ),
        ("time_series/trend.json", "unsupported task time_series"),
        ("regression/categorical.json", "integer, non-null"),
        ("regression/missing_values.json", "integer, non-null"),
    ] {
        cases.push((synthetic_root().join(fixture), reason));
    }
    for (input, reason) in &cases {
        for command in ["fit", "search", "route"] {
            assert_command_rejects(command, input, &router, reason);
        }
    }
}

#[test]
fn dataset_commands_validate_features_and_never_fall_back_to_test() {
    let temporary = tempfile::tempdir().unwrap();
    let router = test_router(temporary.path());
    let input = temporary.path().join("invalid.json");
    let original: Value = serde_json::from_slice(&fs::read(regression_linear()).unwrap()).unwrap();
    for (alteration, reason) in [
        (0, "feature width differs from schema"),
        (1, "expected integer feature"),
        (2, "empty or duplicate column name"),
        (3, "missing train/validation/test partition"),
    ] {
        let mut data = original.clone();
        match alteration {
            0 => data["rows"][0]["features"] = serde_json::json!([]),
            1 => data["rows"][0]["features"][0] = serde_json::json!(1.5),
            2 => data["columns"][1]["name"] = data["columns"][0]["name"].clone(),
            _ => {
                // Without validation rows there is no hold-out; test rows never stand in.
                for row in data["rows"].as_array_mut().unwrap() {
                    if row["split"] == "validation" {
                        row["split"] = serde_json::json!("test");
                    }
                }
            }
        }
        fs::write(&input, serde_json::to_vec(&data).unwrap()).unwrap();
        for command in ["encode", "fit", "search", "route"] {
            assert_command_rejects(command, &input, &router, reason);
        }
    }
}

#[test]
fn search_selection_and_validation_score_do_not_use_test_rows() {
    let temporary = tempfile::tempdir().unwrap();
    let input = temporary.path().join("changed-test.json");
    let original = jope_cli::search_dataset(&regression_linear()).unwrap();
    let fitted = jope_cli::fit_dataset(&regression_linear(), &original.genome).unwrap();
    let mut data: Value = serde_json::from_slice(&fs::read(regression_linear()).unwrap()).unwrap();
    for row in data["rows"].as_array_mut().unwrap() {
        if row["split"] == "test" {
            row["target"] = serde_json::json!(-999_999_999);
            for feature in row["features"].as_array_mut().unwrap() {
                *feature = serde_json::json!(999_999_999);
            }
        }
    }
    fs::write(&input, serde_json::to_vec(&data).unwrap()).unwrap();
    let changed = jope_cli::search_dataset(&input).unwrap();
    assert_eq!(changed.genome, original.genome);
    assert_eq!(changed.hold_r2, original.hold_r2);
    let refitted = jope_cli::fit_dataset(&input, &original.genome).unwrap();
    assert_eq!(refitted.hold_rows, 16);
    assert_eq!(refitted.hold_r2, fitted.hold_r2);
    assert_eq!(refitted.train_r2, fitted.train_r2);
}

#[test]
fn route_without_router_fails_closed() {
    let input = regression_linear();
    let output = jope(
        Path::new("."),
        &[
            "route",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
            "--router",
            "missing-router.json",
        ],
    );
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.route.v1");
}

#[test]
fn train_step_writes_router_and_route_fits_a_ridge_genome() {
    let temporary = tempfile::tempdir().unwrap();
    let root = synthetic_root();
    let trained = jope(
        Path::new("."),
        &[
            "train-step",
            "--local",
            "--json",
            "--root",
            root.to_str().expect("utf8 path"),
            "--output",
            temporary.path().to_str().expect("utf8 path"),
        ],
    );
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    assert!(temporary.path().join("router.v1.json").is_file());
    let input = regression_linear();
    let routed = jope(
        Path::new("."),
        &[
            "route",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
            "--router",
            temporary.path().to_str().expect("utf8 path"),
        ],
    );
    assert!(
        routed.status.success(),
        "{}",
        String::from_utf8_lossy(&routed.stderr)
    );
    let report: Value = serde_json::from_slice(&routed.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["schema"], "jope.route.v1");
    assert!(report["genome"].as_str().unwrap().contains("ridge"));
    assert!(report["hold_r2"].as_f64().unwrap().is_finite());
}

#[cfg(feature = "dope-encoding")]
#[path = "support/landmark_product.rs"]
mod landmark_product;

#[cfg(all(feature = "dope-encoding", target_os = "linux"))]
#[path = "support/landmark_one_shot.rs"]
mod landmark_one_shot;
