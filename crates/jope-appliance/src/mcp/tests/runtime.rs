use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_frame_recovers_and_notification_cancellation_suppresses_reply() {
    for version in VERSIONS {
        let mut wire = Wire::connect(version).await;
        wire.ready().await;
        wire.writer.write_all(b"{not JSON\n").await.unwrap();
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
        wire.send("tools/call",json!({"name":"run_status","arguments":{"run_id":RUN,"wait_seconds":1,"after_event":"1"}}),Some(json!("cancelled-request"))).await;
        tokio::time::timeout(Duration::from_secs(1), async {
            while wire.backend.waits.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        wire.send(
            "notifications/cancelled",
            json!({"requestId":"cancelled-request","reason":"caller stopped waiting"}),
            None,
        )
        .await;
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
        tokio::time::sleep(Duration::from_millis(300)).await;
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
        assert!(!wire
            .transcript
            .iter()
            .any(|v| v["direction"] == "out" && v["message"]["id"] == "cancelled-request"));
        wire.finish("cancelled").await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn long_poll_uses_bounded_http_slices_and_returns_within_the_requested_deadline() {
    let mut wire = Wire::connect("2025-11-25").await;
    wire.ready().await;
    let started = WallInstant::now();
    let result = wire
        .tool(
            "run_status",
            json!({"run_id":RUN,"wait_seconds":1,"after_event":"1"}),
        )
        .await;
    assert_ne!(result["result"]["isError"], true, "{result}");
    assert!(started.elapsed() < Duration::from_millis(1300));
    let requests = wire.backend.requests.lock().unwrap().clone();
    assert!(!requests.is_empty());
    for request in requests {
        let target = request["target"].as_str().unwrap();
        assert!(target.contains("wait_seconds=1"));
        assert!(target.contains("after_event=1"));
    }
    wire.finish("long-poll").await;
}

#[test]
fn canonical_schema_gaps_are_explicit_and_cached_resources_are_bounded() {
    let inventory = catalog::schema_inventory();
    assert_eq!(inventory.len(), 12);
    // API-07-F02 (#677) materialized every exchange schema, so no tool reports a gap.
    assert_eq!(catalog::pending_exchanges(), Vec::<&str>::new());
    for (tool, gaps) in &inventory {
        assert_eq!(gaps, &json!([]), "{tool} still reports schema gaps");
    }
    let resource = dispatch::Resources::default();
    assert!(resource
        .put(
            "jope://oversized".into(),
            "text/plain",
            "x".repeat(dispatch::MAX_CONTENT + 1)
        )
        .is_err());
    for i in 0..12 {
        resource
            .put(
                format!("jope://prediction/{i}"),
                "text/plain",
                "x".repeat(dispatch::MAX_CONTENT),
            )
            .unwrap();
    }
    assert!(resource.list().unwrap().len() <= 4);
    assert!(resource.get("jope://prediction/0").unwrap().is_none());
    let client = Client(Arc::new(Fixture::default()));
    for uri in [
        "file:///etc/passwd",
        "https://example.com",
        "jope://runs/../../etc/passwd",
        "jope://runs/bad-id",
    ] {
        assert!(dispatch::read_resource(&client, &resource, uri).is_err());
    }
    assert!(client.0.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn stdout_relay_refuses_non_rpc_bytes_without_forwarding_them() {
    let (mut sender, input) = tokio::io::duplex(128);
    let (output, mut reader) = tokio::io::duplex(128);
    let relay = tokio::spawn(transport::relay(input, output));
    sender.write_all(b"debug log\n").await.unwrap();
    sender.shutdown().await.unwrap();
    assert!(relay.await.unwrap().is_err());
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await.unwrap();
    assert!(bytes.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_fitting_and_slow_requests_do_not_block_status_or_cancel_p95() {
    use jope_runtime::{
        api::{fit, FitOptions},
        budget::Budget,
        fitted::CreatedBy,
        limits::Limits,
        table::{parse_csv, training},
        task::{TaskChoice, TaskKind},
    };
    assert_eq!(jope_runtime::sha256_hex(TRAIN.as_bytes()), TRAIN_SHA);
    let csv = format!(
        "{}\n{}",
        (0..12)
            .map(|i| format!("x{i}"))
            .chain(["target".into()])
            .collect::<Vec<_>>()
            .join(","),
        TRAIN
    );
    let table = parse_csv(csv.as_bytes(), &Limits::default()).unwrap();
    let training = training(
        &table,
        "target",
        TaskChoice::Fixed(TaskKind::Regression),
        &Limits::default(),
    )
    .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let fits = Arc::new(AtomicUsize::new(0));
    let stopped = stop.clone();
    let fitted = fits.clone();
    let worker = std::thread::spawn(move || {
        let budget = Budget::new(None, 1);
        let options = FitOptions {
            method: "ridge".into(),
            ..Default::default()
        };
        while !stopped.load(Ordering::SeqCst) {
            let outcome = fit(
                &training,
                &options,
                &budget,
                TRAIN_SHA,
                CreatedBy::default(),
            )
            .unwrap();
            std::hint::black_box(outcome);
            fitted.fetch_add(1, Ordering::SeqCst);
        }
    });
    let mut guard = FittingGuard {
        stop: stop.clone(),
        worker: Some(worker),
    };
    let result=async{
        tokio::time::timeout(Duration::from_secs(3),async{while fits.load(Ordering::SeqCst)==0{tokio::task::yield_now().await;}}).await.unwrap();
        let mut reports=Vec::new();
        for version in VERSIONS {
            let mut wire=Wire::connect(version).await;wire.ready().await;
            // Eight intentionally delayed API polls fill only the slow lane; native fitting
            // runs continuously in a separate real runtime worker, with original training labels.
            for i in 0..8 {wire.send("tools/call",json!({"name":"run_status","arguments":{"run_id":RUN,"wait_seconds":1,"after_event":"1"}}),Some(json!(format!("slow-{i}")))).await;}
            tokio::time::timeout(Duration::from_secs(1),async{while wire.backend.waits.load(Ordering::SeqCst)<8{tokio::task::yield_now().await;}}).await.unwrap();
            let mut status=Vec::new();let mut cancel=Vec::new();
            for i in 0..20 {
                let start=WallInstant::now();let reply=wire.tool("run_status",json!({"run_id":RUN})).await;status.push(start.elapsed().as_secs_f64()*1000.0);assert_ne!(reply["result"]["isError"],true);
                let start=WallInstant::now();let reply=wire.tool("run_control",json!({"run_id":RUN,"action":"cancel","idempotency_key":format!("latency-{i}")})).await;cancel.push(start.elapsed().as_secs_f64()*1000.0);assert_eq!(reply["result"]["structuredContent"]["state"],"cancelled");
            }
            let p95=|samples:&[f64]|{let mut sorted=samples.to_vec();sorted.sort_by(f64::total_cmp);sorted[(sorted.len()*95).div_ceil(100)-1]};
            let status_p95=p95(&status);let cancel_p95=p95(&cancel);assert!(status_p95<100.0,"{version} status p95 {status_p95}ms");assert!(cancel_p95<100.0,"{version} cancel p95 {cancel_p95}ms");
            reports.push(json!({"version":version,"status_ms":status,"cancel_ms":cancel,"status_p95_ms":status_p95,"cancel_p95_ms":cancel_p95}));wire.finish("fitting-latency").await;
        }
        reports
    }.await;
    stop.store(true, Ordering::SeqCst);
    guard.worker.take().unwrap().join().unwrap();
    assert!(fits.load(Ordering::SeqCst) > 0);
    let report = json!({"dataset":"b4c8807a75fc190f","train_sha256":TRAIN_SHA,"native_method":"ridge","actual_completed_fits":fits.load(Ordering::SeqCst),"api_backend":"controlled Transport fixture, not live discover engine","samples_per_operation_per_version":20,"reports":result});
    let dir = proof_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("fitting-latency.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("{report}");
}

pub(super) fn proof_dir() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("JOPE_MCP02_PROOF_DIR") {
        return path.into();
    }
    let path = std::path::Path::new(file!());
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/mcp/tests.rs")
    };
    path.parent()
        .unwrap()
        .join("../../../../target/mcp02-proof")
}

struct FittingGuard {
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Drop for FittingGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generated_input_and_success_error_output_schemas_validate_actual_wire_values() {
    let mut wire = Wire::connect("2025-11-25").await;
    wire.ready().await;
    let definitions = catalog::tools();
    for (name, args) in [
        ("system_info", json!({})),
        ("run_start", run_spec()),
        (
            "algorithm_get",
            json!({"run_id":RUN,"algorithm_id":ALGORITHM}),
        ),
        (
            "model_export",
            json!({"run_id":RUN,"algorithm_id":ALGORITHM,"kind":"bundle","languages":["rust"],"idempotency_key":"schema-export"}),
        ),
    ] {
        let schema = definitions["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == name)
            .unwrap();
        if name == "run_start" {
            // The generated DecimalU64 schemas use ECMA-262's end-of-input
            // look-ahead. The Rust regex crate cannot parse that syntax directly;
            // the shared schema validator must translate it without weakening the
            // full-string constraint.
            let mut pending = vec![&schema["inputSchema"]];
            let mut checked_end_lookaheads = 0;
            while let Some(node) = pending.pop() {
                match node {
                    Value::Object(fields) => {
                        if let Some(Value::String(pattern)) = fields.get("pattern") {
                            if pattern.contains("(?![\\s\\S])") {
                                let pattern_schema = json!({"type": "string", "pattern": pattern});
                                assert!(
                                    jope_datasets::schema::validate(&pattern_schema, &json!("0"))
                                        .is_empty(),
                                    "generated DecimalU64 pattern rejected a valid value: {pattern}"
                                );
                                assert!(
                                    !jope_datasets::schema::validate(
                                        &pattern_schema,
                                        &json!("0\n")
                                    )
                                    .is_empty(),
                                    "generated DecimalU64 pattern accepted a trailing newline: {pattern}"
                                );
                                checked_end_lookaheads += 1;
                            }
                        }
                        pending.extend(fields.values());
                    }
                    Value::Array(items) => pending.extend(items),
                    _ => {}
                }
            }
            assert!(
                checked_end_lookaheads > 0,
                "run_start input schema did not expose a DecimalU64 look-ahead pattern"
            );
        }
        let errors = jope_datasets::schema::validate(&schema["inputSchema"], &args);
        assert!(errors.is_empty(), "{name} input: {errors:?}");
        let result = wire.tool(name, args).await;
        if let Some(output_schema) = schema.get("outputSchema") {
            let errors = jope_datasets::schema::validate(
                output_schema,
                &result["result"]["structuredContent"],
            );
            assert!(errors.is_empty(), "{name} success: {errors:?}");
        } else {
            assert_eq!(name, "algorithm_get");
            assert_eq!(schema["_meta"]["jope/schemaQualified"], false);
            assert!(schema["_meta"]["jope/missingCanonicalSchemas"]
                .as_array()
                .unwrap()
                .contains(&json!("AlgorithmPage")));
        }
    }
    wire.backend.fail.store(true, Ordering::SeqCst);
    let result = wire.tool("system_info", json!({})).await;
    assert_eq!(result["result"]["isError"], true);
    let problem: jope_api::error::Problem =
        serde_json::from_value(result["result"]["structuredContent"].clone()).unwrap();
    assert_eq!(problem.code, jope_api::error::ErrorCode::IoFailure);
    let schema = &definitions["tools"][0]["outputSchema"];
    assert!(
        jope_datasets::schema::validate(schema, &result["result"]["structuredContent"]).is_empty()
    );
    wire.finish("schema").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn modern_request_metadata_and_string_ids_are_enforced() {
    let mut wire = Wire::connect("2026-07-28").await;
    wire.ready().await;
    wire.send("tools/list", json!({}), Some(json!("unicode-id-☃")))
        .await;
    assert_eq!(
        wire.read_for(json!("unicode-id-☃")).await["id"],
        "unicode-id-☃"
    );
    wire.writer
        .write_all(
            b"{\"jsonrpc\":\"2.0\",\"id\":\"no-meta\",\"method\":\"tools/list\",\"params\":{}}\n",
        )
        .await
        .unwrap();
    assert_eq!(
        wire.read_for(json!("no-meta")).await["error"]["code"],
        -32602
    );
    let ping = wire.call("ping", json!({})).await;
    assert_eq!(
        ping["error"]["code"], -32601,
        "SDK stateless protocol does not expose legacy ping"
    );
    assert!(wire
        .call("tools/list", json!({}))
        .await
        .get("result")
        .is_some());
    wire.finish("modern-metadata").await;
}
