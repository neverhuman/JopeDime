use super::*;

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
