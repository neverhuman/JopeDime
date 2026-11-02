use super::*;

#[test]
fn calibrated_object_model_mcp_accepts_auxiliary_columns_like_cli_prediction() {
    use std::io::Write;
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    let mut train = String::from("x,margin,y\n");
    for i in 0..90 {
        train.push_str(&format!(
            "{},{},{}\n",
            i,
            2 + i % 4,
            20 + 3 * i + (i * 37) % 11
        ));
    }
    std::fs::write(dir.path().join("train.csv"), train).unwrap();
    let input = "margin,x\n3,1\n5,4\n";
    std::fs::write(dir.path().join("input.csv"), input).unwrap();
    let fitness = r#"{"metric":"weighted","weights":{"op":"column","name":"margin"},"fitness":{"metric":"pinball","quantile":0.9}}"#;
    let fitted: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--task",
                "regression",
                "--candidates",
                "ridge",
                "--folds",
                "3",
                "--fitness",
                fitness,
                "--threads",
                "1",
                "--json",
            ],
        ))
        .stdout,
    )
    .unwrap();
    let path = fitted["outputs"][0]["path"].as_str().unwrap();
    let object =
        jope_runtime::model_object::ModelObject::read(&dir.path().join(path), 1 << 24).unwrap();
    assert!(!object
        .fitted()
        .unwrap()
        .header
        .output_transform
        .unwrap()
        .is_identity());
    ok(run(
        &exe,
        dir.path(),
        &["model", "verify", "--model", path, "--require-executable"],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            path,
            "--input",
            "input.csv",
            "--output",
            "cli.csv",
            "--threads",
            "1",
        ],
    ));
    for format in ["bound-json", "python", "rust", "javascript"] {
        let output = format!("lossy-{format}");
        let refused = run(
            &exe,
            dir.path(),
            &[
                "model", "export", "--model", path, "--format", format, "--output", &output,
            ],
        );
        assert_eq!(
            code(&refused),
            4,
            "{format}: {}",
            String::from_utf8_lossy(&refused.stderr)
        );
        assert!(!dir.path().join(&output).exists());
    }
    let mut child = Command::new(&exe)
        .args(["model", "mcp", "--root", "models"])
        .current_dir(dir.path())
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"model_predict","arguments":{"id":object.id(),"csv":input}}}),
        serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"model_verify","arguments":{"id":object.id()}}}),
    ] {
        writeln!(stdin, "{message}").unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses[1]["result"]["isError"], false, "{}", responses[1]);
    assert_eq!(
        responses[1]["result"]["structuredContent"]["predictions_csv"],
        std::fs::read_to_string(dir.path().join("cli.csv")).unwrap()
    );
    assert_eq!(responses[2]["result"]["isError"], false, "{}", responses[2]);
}

#[test]
fn landmark_flags_preserve_input_when_companion_receipt_aliases_training_csv() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let before = std::fs::read(dir.path().join("train.csv")).unwrap();
    let output = run(
        Path::new(BIN),
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--method",
            "ridge",
            "--output",
            "fitted.jope",
            "--landmark-mode",
            "none",
            "--landmark-receipt",
            "train.csv",
            "--json",
        ],
    );
    assert_eq!(
        code(&output),
        2,
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(std::fs::read(dir.path().join("train.csv")).unwrap(), before);
    assert!(!dir.path().join("fitted.jope").exists());
}

#[cfg(not(feature = "dope-encoding"))]
#[test]
fn explicit_landmark_options_require_a_dope_build_and_keep_native_default() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let output = run(
        Path::new(BIN),
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "fitted.jope",
            "--landmark-mode",
            "none",
            "--json",
        ],
    );
    assert_eq!(
        code(&output),
        4,
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(!dir.path().join("fitted.jope").exists());
}

#[test]
fn landmark_weights_cannot_be_replaced_by_main_outputs() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let weights = dir.path().join("model.safetensors");
    let before = b"protected checkpoint bytes";
    std::fs::write(&weights, before).unwrap();
    for arguments in [
        vec!["fit", "--output", "model.safetensors"],
        vec![
            "fit",
            "--output",
            "fitted.jope",
            "--receipt",
            "model.safetensors",
        ],
        vec![
            "fit-predict",
            "--input",
            "rows.csv",
            "--output",
            "model.safetensors",
        ],
    ] {
        let mut arguments = arguments;
        arguments.extend([
            "--train",
            "train.csv",
            "--target",
            "y",
            "--landmark-mode",
            "none",
            "--dope-weights",
            "model.safetensors",
            "--json",
        ]);
        let output = run(Path::new(BIN), dir.path(), &arguments);
        assert_eq!(
            code(&output),
            2,
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_eq!(std::fs::read(&weights).unwrap(), before);
    }
    assert!(!dir.path().join("fitted.jope").exists());
}
