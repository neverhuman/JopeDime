use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discovery_with_early_roots_change_keeps_modern_bootstrap_open() {
    let mut wire = Wire::connect("2026-07-28").await;
    let discovery_id = json!("gemini-discovery");
    wire.send("server/discover", json!({}), Some(discovery_id.clone()))
        .await;
    // Actual agy 1.4.12 sends this legacy hint without per-request metadata
    // immediately after its modern discovery probe, before requesting tools.
    wire.writer
        .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/roots/list_changed\",\"params\":{}}\n")
        .await
        .unwrap();
    wire.writer.flush().await.unwrap();
    let discovered = wire.read_for(discovery_id).await;
    assert_eq!(discovered["result"]["resultType"], "complete");
    assert!(wire.backend.requests.lock().unwrap().is_empty());

    let tools = wire.call("tools/list", json!({})).await;
    assert!(tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "run_start"));
    assert!(wire.backend.requests.lock().unwrap().is_empty());
    let info = wire.tool("system_info", json!({})).await;
    assert_ne!(info["result"]["isError"], true);
    assert_eq!(info["result"]["structuredContent"]["api_version"], "1.0");
    assert_eq!(wire.backend.requests.lock().unwrap().len(), 1);
    wire.finish("early-roots-discovery").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roots_change_without_discovery_does_not_open_a_session() {
    let mut wire = Wire::connect("2026-07-28").await;
    wire.writer
        .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/roots/list_changed\",\"params\":{}}\n")
        .await
        .unwrap();
    wire.writer.flush().await.unwrap();
    wire.writer.shutdown().await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(3), wire.server)
        .await
        .expect("invalid bootstrap must finish")
        .unwrap();
    assert!(result.unwrap_err().to_string().contains("RootsListChanged"));
    assert!(wire.backend.requests.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_discovery_metadata_does_not_allow_a_roots_hint() {
    let mut wire = Wire::connect("2026-07-28").await;
    wire.writer
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"server/discover\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/roots/list_changed\",\"params\":{}}\n")
        .await
        .unwrap();
    wire.writer.flush().await.unwrap();
    wire.writer.shutdown().await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(3), wire.server)
        .await
        .expect("invalid discovery must finish")
        .unwrap();
    assert!(result.unwrap_err().to_string().contains("RootsListChanged"));
    assert!(wire.backend.requests.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discovery_does_not_allow_other_premature_notifications() {
    let mut wire = Wire::connect("2026-07-28").await;
    wire.send("server/discover", json!({}), Some(json!(1)))
        .await;
    wire.writer
        .write_all(
            b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n",
        )
        .await
        .unwrap();
    wire.writer.flush().await.unwrap();
    wire.writer.shutdown().await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(3), wire.server)
        .await
        .expect("unexpected notification must finish")
        .unwrap();
    assert!(result.unwrap_err().to_string().contains("Initialized"));
    assert!(wire.backend.requests.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_aliases_are_listed_and_dispatched_through_existing_handlers() {
    let mut wire = Wire::connect("2025-11-25").await;
    wire.ready().await;

    let catalog = wire.call("tools/list", json!({})).await;
    let tools = catalog["result"]["tools"].as_array().unwrap();
    for name in [
        "fitness_list",
        "fitness_validate",
        "fitness_evaluate",
        "regression_fit_predict",
        "model_session_show",
        "model_session_record",
        "model_session_rollback",
        "narration_read",
        "narration_inspect",
    ] {
        let tool = tools.iter().find(|tool| tool["name"] == name).unwrap();
        assert!(tool["description"].as_str().unwrap().contains("Deprecated"));
        assert_eq!(tool["_meta"]["deprecated"], true);
        assert!(tool["_meta"]["replacement"].is_string());
    }

    let fitness = wire.tool("fitness_list", json!({})).await;
    assert_eq!(fitness["result"]["isError"], false);
    assert!(fitness["result"]["structuredContent"]["fitness"].is_array());

    let removed = wire.tool("narration_read", json!({})).await;
    assert_eq!(removed["result"]["isError"], true);
    assert_eq!(
        removed["result"]["structuredContent"]["code"],
        "tool_removed"
    );

    wire.finish("legacy-aliases").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn four_versions_lifecycle_twelve_tools_resources_and_stdout_purity() {
    for version in VERSIONS {
        let mut wire = Wire::connect(version).await;
        wire.ready().await;
        let catalog = wire.call("tools/list", json!({})).await;
        let tools = catalog["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 21);
        let names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        let expected: Vec<_> = OPERATIONS.iter().filter_map(|op| op.mcp_tool).collect();
        assert_eq!(names[..expected.len()], expected);
        for tool in &tools[expected.len()..] {
            assert_eq!(tool["_meta"]["deprecated"], true);
            assert!(tool["_meta"]["replacement"].is_string());
        }
        assert!(tools[expected.len()..]
            .iter()
            .any(|tool| tool["name"] == "narration_read"));
        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object");
            if tool["_meta"]["jope/schemaQualified"] == true {
                assert!(tool.get("outputSchema").is_some());
            } else {
                assert!(tool.get("outputSchema").is_none());
            }
        }
        let cases = vec![
            ("system_info", json!({})),
            (
                "dataset_register",
                json!({"csv_text":format!("{}\n{}",(0..12).map(|i|format!("x{i}")).chain(["target".into()]).collect::<Vec<_>>().join(","),TRAIN),"idempotency_key":"dataset-key"}),
            ),
            ("dataset_inspect", json!({"dataset_id":DATASET})),
            (
                "objective_validate",
                json!({"spec":"rmse","task":"regression"}),
            ),
            ("run_start", run_spec()),
            ("run_status", json!({"run_id":RUN})),
            (
                "run_control",
                json!({"run_id":RUN,"action":"cancel","idempotency_key":"cancel-key","expected_revision":"1"}),
            ),
            (
                "run_fork",
                json!({"run_id":RUN,"seed":"same","idempotency_key":"fork-key"}),
            ),
            ("run_events", json!({"run_id":RUN,"after":"0","limit":256})),
            (
                "algorithm_get",
                json!({"run_id":RUN,"algorithm_id":ALGORITHM,"language":"rust","max_lines":2}),
            ),
            (
                "model_export",
                json!({"run_id":RUN,"algorithm_id":ALGORITHM,"kind":"bundle","languages":["rust"],"idempotency_key":"export-key"}),
            ),
            (
                "model_predict",
                json!({"model":{"artifact_id":ARTIFACT},"input":{"dataset_id":DATASET},"idempotency_key":"predict-key"}),
            ),
        ];
        let mut prediction_uri = None;
        for (name, args) in cases {
            let result = wire.tool(name, args).await;
            assert!(result.get("error").is_none(), "{version} {name}: {result}");
            assert_ne!(result["result"]["isError"], true, "{name}: {result}");
            assert!(result["result"]["structuredContent"].is_object());
            // Text-only clients need the same real identifiers and plan fields.
            let text = result["result"]["content"][0]["text"].as_str().unwrap();
            let text_value: Value = serde_json::from_str(text)
                .unwrap_or_else(|_| panic!("{version} {name}: missing JSON text fallback"));
            assert_eq!(text_value, result["result"]["structuredContent"]);
            if version == "2026-07-28" {
                assert_eq!(result["result"]["resultType"], "complete");
            }
            if matches!(name, "algorithm_get" | "model_predict") {
                let content = result["result"]["content"].as_array().unwrap();
                assert_eq!(content.len(), 2);
                if version == "2024-11-05" {
                    assert!(content[1]["text"].as_str().unwrap().contains("jope://"));
                } else {
                    assert_eq!(content[1]["type"], "resource_link");
                    assert!(content[1]["uri"].as_str().unwrap().starts_with("jope://"));
                }
            }
            if name == "model_predict" {
                assert_eq!(result["result"]["structuredContent"]["rows"], "2");
                prediction_uri = result["result"]["structuredContent"]["resource"]
                    .as_str()
                    .map(str::to_owned);
            }
        }
        let templates = wire.call("resources/templates/list", json!({})).await;
        assert_eq!(
            templates["result"]["resourceTemplates"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        for uri in [
            format!("jope://runs/{RUN}"),
            format!("jope://datasets/{DATASET}/profile"),
            format!("jope://runs/{RUN}/algorithms/{ALGORITHM}/source/rust"),
            format!("jope://artifacts/{ARTIFACT}/manifest"),
            prediction_uri.unwrap(),
        ] {
            let resource = wire.call("resources/read", json!({"uri":uri})).await;
            assert!(resource.get("error").is_none(), "{resource}");
            assert!(resource["result"]["contents"][0]["text"].is_string());
        }
        let requests = wire.backend.requests.lock().unwrap().clone();
        let control = requests
            .iter()
            .find(|r| r["target"].as_str().unwrap().ends_with("/control"))
            .unwrap();
        assert_eq!(control["if_match"], "\"1\"");
        assert!(control["body"].get("expected_revision").is_none());
        assert_eq!(control["key"], "cancel-key");
        wire.finish("all-tools").await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_predict_uploads_local_csv_and_atomically_writes_requested_output() {
    let mut wire = Wire::connect("2025-11-25").await;
    wire.ready().await;
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("local-input.csv");
    let output = dir.path().join("predictions.csv");
    std::fs::write(&input, b"x,target\n1,0\n2,1\n").unwrap();

    let result = wire
        .tool(
            "model_predict",
            json!({"model":{"artifact_id":ARTIFACT},"input":{"path":input},"out_path":output,"idempotency_key":"local-predict-key"}),
        )
        .await;
    assert_ne!(result["result"]["isError"], true, "{result}");
    let value = &result["result"]["structuredContent"];
    assert_eq!(value["rows"], "2");
    assert_eq!(value["out_path"], output.to_string_lossy().as_ref());
    assert_eq!(std::fs::read(&output).unwrap(), b"prediction\n0.25\n0.5\n");

    {
        let uploads = wire.backend.uploads.lock().unwrap();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].0, "local-input.csv");
        assert_eq!(uploads[0].1, "local-predict-key");
        assert_eq!(uploads[0].2, b"x,target\n1,0\n2,1\n");
    }
    {
        let requests = wire.backend.requests.lock().unwrap();
        let prediction = requests
            .iter()
            .find(|request| request["target"] == "/v1/predictions")
            .unwrap();
        assert_eq!(prediction["key"], "local-predict-key");
        assert_eq!(prediction["body"]["dataset"], DATASET);
    }
    wire.finish("local-file-predict").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prediction_file_retry_verifies_bytes_and_preserves_changed_output() {
    for version in VERSIONS {
        let mut wire = Wire::connect(version).await;
        wire.ready().await;
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("predictions.csv");
        let args = json!({"model":{"artifact_id":ARTIFACT},"input":{"dataset_id":DATASET},"out_path":output,"idempotency_key":"prediction-file-retry"});
        let first = wire.tool("model_predict", args.clone()).await;
        assert_eq!(first["result"]["isError"], false, "{first}");
        let replay = wire.tool("model_predict", args.clone()).await;
        assert_eq!(replay["result"]["isError"], false, "{replay}");
        for key in ["rows", "sha256", "out_path"] {
            assert_eq!(
                first["result"]["structuredContent"][key],
                replay["result"]["structuredContent"][key]
            );
        }
        assert_eq!(std::fs::read(&output).unwrap(), b"prediction\n0.25\n0.5\n");
        std::fs::write(&output, b"prediction\n0.75\n0.5\n").unwrap();
        let changed = wire.tool("model_predict", args).await;
        assert_eq!(changed["result"]["isError"], true);
        assert_eq!(
            changed["result"]["structuredContent"]["detail"],
            "OUTPUT_EXISTS"
        );
        assert_eq!(std::fs::read(&output).unwrap(), b"prediction\n0.75\n0.5\n");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        wire.finish("prediction-file-retry").await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_model_path_is_rejected_before_dataset_upload() {
    let mut wire = Wire::connect("2025-11-25").await;
    wire.ready().await;
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("local-input.csv");
    std::fs::write(&input, b"x,target\n1,0\n").unwrap();
    let result = wire
        .tool(
            "model_predict",
            json!({"model":{"path":"model.bundle"},"input":{"path":input},"idempotency_key":"local-model-key"}),
        )
        .await;
    assert_eq!(result["result"]["isError"], true);
    assert!(wire.backend.uploads.lock().unwrap().is_empty());
    assert!(wire.backend.requests.lock().unwrap().is_empty());
    wire.finish("local-model-refused").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_path_accepts_only_an_explicit_artifact_id() {
    let mut wire = Wire::connect("2025-11-25").await;
    wire.ready().await;
    let result = wire
        .tool(
            "model_predict",
            json!({"model":{"path":ARTIFACT},"input":{"dataset_id":DATASET},"idempotency_key":"artifact-path-key"}),
        )
        .await;
    assert_ne!(result["result"]["isError"], true, "{result}");
    {
        let requests = wire.backend.requests.lock().unwrap();
        let prediction = requests
            .iter()
            .find(|request| request["target"] == "/v1/predictions")
            .unwrap();
        assert_eq!(prediction["body"]["model_artifact"], ARTIFACT);
    }
    assert!(wire.backend.uploads.lock().unwrap().is_empty());
    wire.finish("artifact-id-model-path").await;
}

#[test]
fn uncached_manifest_is_reported_as_explicitly_unavailable() {
    let client = Client(Arc::new(Fixture::default()));
    let result = dispatch::read_resource(
        &client,
        &dispatch::Resources::default(),
        &format!("jope://artifacts/{ARTIFACT}/manifest"),
    );
    assert_eq!(
        result.unwrap_err(),
        "manifest unavailable: artifact manifest is not cached in this process"
    );
    assert!(client.0.requests.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_arguments_side_effect_guards_notifications_and_version_exclusion() {
    for version in VERSIONS {
        let mut wire = Wire::connect(version).await;
        wire.ready().await;
        let count = wire.backend.requests.lock().unwrap().len();
        for args in [
            json!({"run_id":RUN,"wait_seconds":61}),
            json!({"run_id":RUN,"wait_seconds":-1}),
            json!({"wait_seconds":1}),
            json!({"run_id":"bad-id"}),
            json!({"run_id":RUN,"after_event":1}),
            json!({"run_id":RUN,"after_event":"01"}),
            json!({"surprise":true}),
        ] {
            assert_eq!(wire.tool("run_status", args).await["error"]["code"], -32602);
        }
        assert_eq!(
            wire.tool("run_control", json!({"run_id":RUN,"action":"cancel"}))
                .await["error"]["code"],
            -32602
        );
        assert_eq!(
            wire.tool(
                "run_control",
                json!({"run_id":RUN,"action":"cancel","idempotency_key":"bad\nkey"})
            )
            .await["error"]["code"],
            -32602
        );
        assert_eq!(wire.backend.requests.lock().unwrap().len(), count);
        wire.send("tools/call",json!({"name":"run_control","arguments":{"run_id":RUN,"action":"cancel","idempotency_key":"notification"}}),None).await;
        let ping = wire
            .call(
                if version == "2026-07-28" {
                    "tools/list"
                } else {
                    "ping"
                },
                json!({}),
            )
            .await;
        assert!(ping.get("result").is_some(), "{version}: {ping}");
        assert_eq!(wire.backend.requests.lock().unwrap().len(), count);
        assert_eq!(
            wire.tool("unknown_tool", json!({})).await["error"]["code"],
            -32602
        );
        let result=wire.tool("model_export",json!({"run_id":RUN,"kind":"bundle","languages":["rust"],"out_dir":"/nonexistent-jope-parent/must-not-be-written","idempotency_key":"blocked"})).await;
        assert_eq!(result["result"]["isError"], true);
        assert_eq!(wire.backend.requests.lock().unwrap().len(), count);
        wire.finish("invalid").await;
    }
    let mut excluded = Wire::connect("2025-03-26").await;
    excluded.ready().await;
    assert!(excluded
        .call("ping", json!({}))
        .await
        .get("result")
        .is_some());
    excluded.finish("excluded-version").await;
}
