// Generate the Go qualification clone from the frozen native suite.
// Only checked insertions and documentation/path changes are permitted.
use std::path::PathBuf;
pub fn generate() {
    println!("cargo:rerun-if-changed=tests/golden.rs");
    println!("cargo:rerun-if-changed=tests/go_support/generate.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GO_PARITY_TESTS");
    if std::env::var_os("CARGO_FEATURE_GO_PARITY_TESTS").is_none() {
        return;
    }
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let original = std::fs::read_to_string(root.join("tests/golden.rs")).unwrap();
    let mut text = original.clone();
    insert(
        &mut text,
        r#####"use feat_math::transpile::{emit, lint_vocabulary, supports, EmitOptions, Emitted};
use feat_math::{fit_predict, GenomeIr, Pcg32};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
"#####,
        r#####"
mod go_support { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/go_support/mod.rs")); }
"#####,
    );
    insert(
        &mut text,
        r#####"    let t_py = std::time::Instant::now();
    let py = run_python(&dir, &artifact);
    let py_secs = t_py.elapsed().as_secs_f64();
    let t_rs = std::time::Instant::now();
    let rs = run_rust(&dir, &artifact);
    let rs_secs = t_rs.elapsed().as_secs_f64();

"#####,
        r#####"    if go_support::enabled() {
        let artifact = feat_math::transpile::emit_go(&genome, &opts).unwrap();
        go_support::run(
            &dir.join("go"),
            &artifact.source,
            &artifact.kernel,
            &serde_json::json!({"train":x_train,"targets":y_train,"test":x_test,"stress":name=="robust_ridge"}),
            &rs,
            false,
        );
    }
"#####,
    );
    insert(
        &mut text,
        r#####"    let x_test: Vec<Vec<f64>> = x[64..].to_vec();
    let reference = fit_predict(&genome, &x_train, &y_train, &x_test).unwrap();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden_portfolio_dups");
    fs::create_dir_all(&dir).unwrap();
    write_csvs(&dir, &x_train, &y_train, &x_test);
    let py = run_python(&dir, &artifact);
    let rs = run_rust(&dir, &artifact);
"#####,
        r#####"    if go_support::enabled() {
        let bundle = feat_math::transpile::emit_go(&genome, &EmitOptions::default()).unwrap();
        go_support::run(
            &dir.join("go"),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"train":x_train,"targets":y_train,"test":x_test}),
            &rs,
            false,
        );
    }
"#####,
    );
    insert(
        &mut text,
        r#####"    let py = run_python(&dir_fast, &fast);
    let rs_fast = run_rust(&dir_fast, &fast);

    let dir_ref = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fastp_{name}_ref"));
    fs::create_dir_all(&dir_ref).unwrap();
    write_csvs(&dir_ref, &x_train, &y_train, &x_test);
    let rs_default = run_rust(&dir_ref, &default);
"#####,
        r#####"    if go_support::enabled() {
        let bundle = feat_math::transpile::emit_go(&genome, &fast_opts()).unwrap();
        go_support::run(
            &dir_fast.join("go"),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"train":x_train,"targets":y_train,"test":x_test}),
            &rs_default,
            false,
        );
    }
"#####,
    );
    // The upstream native suite now includes optional TypeScript qualification.
    // Preserve that module using an absolute include from this generated file.
    assert_eq!(text.matches("mod typescript_support;").count(), 1);
    text = text.replace("mod typescript_support;", "mod typescript_support { include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/tests/typescript_support/mod.rs\")); }");
    // C17 support is also retained verbatim. The generated suite lives in OUT_DIR,
    // so its path attributes must resolve back to the original native helpers.
    for (attribute, relative) in [
        ("#[path = \"support/c17.rs\"]", "tests/support/c17.rs"),
        (
            "#[path = \"support/c17_export.rs\"]",
            "tests/support/c17_export.rs",
        ),
    ] {
        assert_eq!(
            text.matches(attribute).count(),
            1,
            "native C17 module drift"
        );
        text = text.replace(attribute, &format!("#[path = {:?}]", root.join(relative)));
    }
    // An include expansion cannot carry the original crate-level doc comments.
    text = text
        .lines()
        .map(|line| {
            line.strip_prefix("//!")
                .map_or_else(|| line.to_string(), |s| format!("//{s}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    text = text
        .replace(".join(format!(\"golden_", ".join(format!(\"go_golden_")
        .replace(".join(format!(\"fastp_", ".join(format!(\"go_fastp_")
        .replace(".join(\"golden_", ".join(\"go_golden_");
    text.push_str("\n#[test] fn go_runtime_is_required(){assert!(go_support::enabled(),\"set JOPE_GO_PARITY=1 for Go qualification\");}\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("go_golden.rs");
    std::fs::write(out, text).unwrap();
}
fn insert(text: &mut String, anchor: &str, addition: &str) {
    assert_eq!(
        text.matches(anchor).count(),
        1,
        "native golden hook anchor drift"
    );
    *text = text.replacen(anchor, &format!("{anchor}{addition}"), 1);
}
