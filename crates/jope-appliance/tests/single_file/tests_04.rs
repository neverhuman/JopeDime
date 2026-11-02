use super::*;

#[test]
fn standard_object_fit_reload_export_and_mcp_predict_agree() {
    use std::io::Write;
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    let receipt: serde_json::Value = serde_json::from_slice(
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
                "--method",
                "ridge",
                "--threads",
                "1",
                "--receipt",
                "fit-receipt.json",
                "--json",
            ],
        ))
        .stdout,
    )
    .unwrap();
    let saved_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("fit-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt, saved_receipt);
    let path = receipt["outputs"][0]["path"].as_str().unwrap();
    assert!(path.starts_with("models/sha256-"));
    let object_path = dir.path().join(path);
    let manifest_bytes = std::fs::read(object_path.join("manifest.json")).unwrap();
    assert_eq!(
        receipt["outputs"][0]["sha256"],
        jope_runtime::sha256_hex(&manifest_bytes)
    );
    let manifest_schema: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../schemas/jope.model-object.v1.schema.json"
    ))
    .unwrap();
    let manifest_value: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert!(jope_datasets::schema::validate(&manifest_schema, &manifest_value).is_empty());
    let info: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &["model", "verify", "--model", path, "--require-executable"],
        ))
        .stdout,
    )
    .unwrap();
    assert_eq!(info["executable"], true);
    assert_eq!(info["schema"], "jope.model-object-inspection/v1");
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            path,
            "--input",
            "rows.csv",
            "--output",
            "object.csv",
            "--threads",
            "1",
        ],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "model",
            "export",
            "--model",
            path,
            "--format",
            "fitted",
            "--output",
            "export.jope",
        ],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            "export.jope",
            "--input",
            "rows.csv",
            "--output",
            "legacy.csv",
            "--threads",
            "1",
        ],
    ));
    assert_eq!(
        std::fs::read(dir.path().join("object.csv")).unwrap(),
        std::fs::read(dir.path().join("legacy.csv")).unwrap()
    );
    assert_eq!(
        std::fs::read(dir.path().join("export.jope")).unwrap(),
        std::fs::read(object_path.join("model.jope")).unwrap()
    );
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "model",
                "export",
                "--model",
                path,
                "--format",
                "fitted",
                "--output",
                "export.jope"
            ]
        )),
        7
    );
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "model",
                "export",
                "--model",
                path,
                "--format",
                "onnx",
                "--output",
                "model.onnx"
            ]
        )),
        4
    );
    assert!(!dir.path().join("model.onnx").exists());
    // A prediction output or receipt must never overwrite any member of its source object.
    let protected = format!("{path}/model.jope");
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                path,
                "--input",
                "rows.csv",
                "--output",
                &protected,
                "--threads",
                "1"
            ]
        )),
        2
    );
    let protected_manifest = format!("{path}/manifest.json");
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                path,
                "--input",
                "rows.csv",
                "--output",
                "unused.csv",
                "--receipt",
                &protected_manifest,
                "--threads",
                "1"
            ]
        )),
        2
    );
    assert_eq!(
        std::fs::read(object_path.join("manifest.json")).unwrap(),
        manifest_bytes
    );
    let id = info["id"].as_str().unwrap();
    // A directory relative to cwd must still derive its identity from the resolved name.
    let dot: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            &object_path,
            &["model", "inspect", "--model", "."],
        ))
        .stdout,
    )
    .unwrap();
    assert_eq!(dot["id"], info["id"]);
    let first: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "model",
                "session",
                "record",
                "--name",
                "test",
                "--id",
                id,
                "--expected-head",
                "empty",
            ],
        ))
        .stdout,
    )
    .unwrap();
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "model",
                "session",
                "record",
                "--name",
                "test",
                "--id",
                id,
                "--expected-head",
                "empty"
            ]
        )),
        7
    );
    let rollback: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "model",
                "session",
                "rollback",
                "--name",
                "test",
                "--version",
                "1",
                "--expected-head",
                first["head"].as_str().unwrap(),
            ],
        ))
        .stdout,
    )
    .unwrap();
    let event_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jope.model-session.v1.schema.json"
    ))
    .unwrap();
    for event in rollback["versions"].as_array().unwrap() {
        assert!(jope_datasets::schema::validate(&event_schema, event).is_empty());
    }
    assert_eq!(rollback["versions"][1]["model_id"], id);
    assert_eq!(rollback["model_path"], path);
    let requests = [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"model_inspect","arguments":{"id":id}}}),
        serde_json::json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"model_predict","arguments":{"id":id,"csv":std::fs::read_to_string(dir.path().join("rows.csv")).unwrap()}}}),
        serde_json::json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"model_inspect","arguments":{"id":"../escape"}}}),
        serde_json::json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"model_export","arguments":{"id":id}}}),
        serde_json::json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"model_session_show","arguments":{"name":"test"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"tools/call","params":{"name":"model_session_record","arguments":{"name":"unwanted","id":id,"expected_head":"empty"}}}),
    ];
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
    for request in requests {
        writeln!(stdin, "{request}").unwrap();
    }
    drop(stdin);
    let output = ok(child.wait_with_output().unwrap());
    let replies: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(replies.len(), 7); // no response to notifications
    assert_eq!(
        replies[6]["result"]["structuredContent"]["head"],
        rollback["head"]
    );
    assert!(!dir.path().join("models/sessions/unwanted").exists());
    assert_eq!(replies[0]["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(replies[1]["result"]["tools"].as_array().unwrap().len(), 9);
    assert_eq!(replies[2]["result"]["structuredContent"]["id"], info["id"]);
    assert_eq!(
        replies[3]["result"]["structuredContent"]["predictions_csv"],
        std::fs::read_to_string(dir.path().join("object.csv")).unwrap()
    );
    assert_eq!(replies[4]["result"]["isError"], true);
    assert_eq!(
        replies[5]["result"]["structuredContent"]["manifest"],
        String::from_utf8(manifest_bytes).unwrap()
    );
    let exported = &replies[5]["result"]["structuredContent"];
    let import_request = serde_json::json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"model_import","arguments":{"manifest":exported["manifest"],"parts_hex":exported["parts_hex"]}}});
    let mut child = Command::new(&exe)
        .args(["model", "mcp", "--root", "mcp-imported"])
        .current_dir(dir.path())
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{import_request}").unwrap();
    let imported: serde_json::Value =
        serde_json::from_slice(&ok(child.wait_with_output().unwrap()).stdout).unwrap();
    assert_eq!(imported["result"]["isError"], false);
    assert_eq!(
        imported["result"]["structuredContent"]["inspection"]["id"],
        info["id"]
    );
    let imported_path = imported["result"]["structuredContent"]["path"]
        .as_str()
        .unwrap();
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            imported_path,
            "--input",
            "rows.csv",
            "--output",
            "mcp-import.csv",
            "--threads",
            "1",
        ],
    ));
    assert_eq!(
        std::fs::read(dir.path().join("object.csv")).unwrap(),
        std::fs::read(dir.path().join("mcp-import.csv")).unwrap()
    );
}

#[test]
fn foreign_extensions_import_without_claiming_execution() {
    use jope_runtime::model_object::{Manifest, Part, Producer, SCHEMA};
    let dir = tempfile::tempdir().unwrap();
    let state = b"arbitrary future GP bytecode and fitted parameters\0\xff";
    std::fs::write(dir.path().join("state.bin"), state).unwrap();
    let manifest = Manifest {
        schema: SCHEMA.into(),
        minimum_reader_version: 1,
        format: "org.example/future-gp/v9".into(),
        producer: Producer {
            name: "external".into(),
            version: "1".into(),
            language: "typescript".into(),
        },
        primary: "state.bin".into(),
        required_capabilities: vec!["org.example/solver/v9".into()],
        files: vec![Part::new("state.bin", "application/octet-stream", state)],
        metadata: serde_json::json!({"note":"retained, not executable in this build"}),
    };
    std::fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let receipt: serde_json::Value = serde_json::from_slice(
        &ok(run(
            Path::new(BIN),
            dir.path(),
            &["model", "import", "--manifest", "manifest.json"],
        ))
        .stdout,
    )
    .unwrap();
    let path = receipt["path"].as_str().unwrap();
    assert_eq!(receipt["inspection"]["integrity_verified"], true);
    assert_eq!(receipt["inspection"]["executable"], false);
    assert_eq!(
        std::fs::read(dir.path().join(path).join("state.bin")).unwrap(),
        state
    );
    ok(run(
        Path::new(BIN),
        dir.path(),
        &["model", "verify", "--model", path],
    ));
    assert_eq!(
        code(&run(
            Path::new(BIN),
            dir.path(),
            &["model", "verify", "--model", path, "--require-executable"]
        )),
        4
    );
    std::fs::write(dir.path().join(path).join("state.bin"), b"corrupt").unwrap();
    assert_eq!(
        code(&run(
            Path::new(BIN),
            dir.path(),
            &["model", "verify", "--model", path]
        )),
        3
    );
}
