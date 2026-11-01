use super::*;
use jope_client::{
    transport::{HeaderMap, HeaderValue},
    Request, Response,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant as WallInstant,
};
use tokio::io::{
    AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, DuplexStream, ReadHalf, WriteHalf,
};
const RUN: &str = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
const DATASET: &str = "ds_000000000000000000000001";
const ARTIFACT: &str = "art_000000000000000000000001";
const ALGORITHM: &str = "alg_0001";
const VERSIONS: [&str; 4] = ["2024-11-05", "2025-06-18", "2025-11-25", "2026-07-28"];
// Catalog v246 b4c8807a75fc190f, TRAIN blob only; original 25 rows/labels unchanged.
const TRAIN: &str = include_str!("fixtures/train.csv");
const TRAIN_SHA: &str = "782acd17eed2bfc7acfa4438be076fb63026763e6578661cfc284c2f3f9a21e1";

#[derive(Default)]
struct Fixture {
    requests: Mutex<Vec<Value>>,
    uploads: Mutex<Vec<(String, String, Vec<u8>)>>,
    cancelled: AtomicBool,
    waits: AtomicUsize,
    delay_immediate: AtomicBool,
    fail: AtomicBool,
}
#[derive(Clone)]
struct Client(Arc<Fixture>);
fn snapshot(cancelled: bool) -> Value {
    json!({"run_id":RUN,"revision":if cancelled{"2"}else{"1"},"state":if cancelled{"cancelled"}else{"running"},"phase":"searching","pending":null,"effort":"fast","budget_used":{"evaluations":"1","wall_seconds":"0.1"},"budget_total":{"evaluations":"100","wall_seconds":"60"},"best":{"algorithm_id":ALGORITHM,"metric":"rmse","cv_value":0.2,"model_artifact":ARTIFACT},"event_watermark":"1","warnings":[],"completion_reason":null})
}
fn system() -> Value {
    json!({"version":"0.4.1","api_version":"1.0","health":"ok","local_paths":false,"capabilities":{"tasks":["regression"],"languages":["rust"],"devices":["cpu"],"learned_guidance":false},"warnings":[],"telemetry":[]})
}
fn profile() -> Value {
    json!({"dataset_id":DATASET,"name":"catalog training fixture","bytes":"3683","rows":"25","columns":[{"name":"target","data_type":"numeric","missing":"0","distinct":"25"}],"target_suggestions":[],"warnings":[]})
}
fn algorithm() -> Value {
    json!({"algorithm_id":ALGORITHM,"summary":"Public fitted predictor","metric":"rmse","cv_value":0.2,"model_artifact":ARTIFACT,"model_digest":"sha256:fixture-model","conditioning_digest":"sha256:fixture-conditioning","bundle_digest":"sha256:fixture-bundle","revision":"1","source":[{"language":"rust","source":"line1\nline2\nline3\nline4\n"}]})
}
fn export() -> Value {
    json!({"artifact_id":ARTIFACT,"algorithm_id":ALGORITHM,"kind":"bundle","model_digest":"sha256:fixture-model","emitter_version":"fixture1","qualified_set_id":"fixture-only","refit":"fit","retains_training_data":false,"languages":[{"status":"exported","language":"rust","path":"src/rust/predictor.rs"}],"parity_status":"unavailable","parity_receipt":null})
}
fn response(value: Value) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
    Response::new(
        200,
        headers,
        Cursor::new(serde_json::to_vec(&value).unwrap()),
    )
}
impl Transport for Client {
    fn execute(&self, request: Request) -> jope_client::Result<Response> {
        let target = request.target().to_owned();
        let body = if request.body().is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(request.body()).unwrap()
        };
        self.0.requests.lock().unwrap().push(json!({"target":target,"method":request.method().as_str(),"body":body,"key":request.headers.get("idempotency-key").map(|v|v.to_str().unwrap()),"if_match":request.headers.get("if-match").map(|v|v.to_str().unwrap())}));
        if self.0.fail.load(Ordering::SeqCst) {
            return Err(jope_client::Error::Status(503));
        }
        let path = target.split('?').next().unwrap();
        if path == "/v1/system" {
            return Ok(response(system()));
        }
        if path == "/v1/objectives/validate" {
            return Ok(response(
                json!({"direction":"minimize","digest":"fixture","errors":[]}),
            ));
        }
        if path == "/v1/datasets" {
            return Ok(response(
                if request.method() == jope_api::registry::HttpMethod::Get {
                    json!({"datasets":[profile()],"next_after":null})
                } else {
                    profile()
                },
            ));
        }
        if path.starts_with("/v1/datasets/") {
            return Ok(response(profile()));
        }
        if path == "/v1/predictions" {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("text/csv"));
            headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
            return Ok(Response::new(
                200,
                headers,
                Cursor::new(b"prediction\n0.25\n0.5\n".to_vec()),
            ));
        }
        if path.ends_with("/exports") {
            return Ok(response(export()));
        }
        if path.ends_with("/events") {
            return Ok(response(json!({"events":[],"next_after":"1"})));
        }
        if path.contains("/algorithms/") {
            return Ok(response(algorithm()));
        }
        if path.ends_with("/control") {
            assert!(request.headers.get("idempotency-key").is_some());
            if body["action"] == "cancel" {
                self.0.cancelled.store(true, Ordering::SeqCst);
            }
            return Ok(response(snapshot(self.0.cancelled.load(Ordering::SeqCst))));
        }
        if path.ends_with("/fork") {
            return Ok(response(snapshot(false)));
        }
        if path == "/v1/runs" {
            return Ok(response(
                if request.method() == jope_api::registry::HttpMethod::Get {
                    json!({"runs":[snapshot(false)],"next_after":null})
                } else {
                    snapshot(false)
                },
            ));
        }
        if path.starts_with("/v1/runs/") {
            if target.contains("wait_seconds=") {
                self.0.waits.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(250));
            } else if self.0.delay_immediate.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(250));
            }
            return Ok(response(snapshot(self.0.cancelled.load(Ordering::SeqCst))));
        }
        panic!("unexpected fixture request {target}")
    }

    fn dataset_register_stream(
        &self,
        name: &str,
        mut body: Box<dyn Read + Send>,
        idempotency_key: &str,
    ) -> jope_client::Result<Response> {
        let mut bytes = Vec::new();
        body.read_to_end(&mut bytes)?;
        self.0
            .uploads
            .lock()
            .unwrap()
            .push((name.to_owned(), idempotency_key.to_owned(), bytes));
        Ok(response(profile()))
    }
}

struct Wire {
    version: &'static str,
    reader: BufReader<ReadHalf<DuplexStream>>,
    writer: WriteHalf<DuplexStream>,
    server: tokio::task::JoinHandle<anyhow::Result<()>>,
    pending: BTreeMap<String, Value>,
    transcript: Vec<Value>,
    counter: u64,
    backend: Arc<Fixture>,
}
impl Wire {
    async fn connect(version: &'static str) -> Self {
        let backend = Arc::new(Fixture::default());
        let (client, server) = tokio::io::duplex(1024 * 1024);
        let (input, output) = tokio::io::split(server);
        let adapter = Adapter::new(Client(backend.clone()));
        let task = tokio::spawn(adapter.serve_io(input, output));
        let (input, output) = tokio::io::split(client);
        Self {
            version,
            reader: BufReader::new(input),
            writer: output,
            server: task,
            pending: BTreeMap::new(),
            transcript: Vec::new(),
            counter: 0,
            backend,
        }
    }
    async fn ready(&mut self) {
        if self.version == "2026-07-28" {
            let info = self.call("server/discover", json!({})).await;
            assert!(info["result"]["supportedVersions"]
                .as_array()
                .unwrap()
                .contains(&json!(self.version)));
            assert_eq!(info["result"]["resultType"], "complete");
            assert!(info["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("idempotency_key"));
        } else {
            let result=self.call("initialize",json!({"protocolVersion":self.version,"capabilities":{},"clientInfo":{"name":"mcp02-wire","version":"1"}})).await;
            assert_eq!(
                result["result"]["protocolVersion"],
                if self.version == "2025-03-26" {
                    "2025-11-25"
                } else {
                    self.version
                }
            );
            assert!(result["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("dataset_register"));
            self.send("notifications/initialized", json!({}), None)
                .await;
        }
    }
    async fn send(&mut self, method: &str, mut params: Value, id: Option<Value>) {
        if self.version == "2026-07-28" {
            params["_meta"] = json!({"io.modelcontextprotocol/protocolVersion":self.version,"io.modelcontextprotocol/clientInfo":{"name":"mcp02-wire","version":"1"},"io.modelcontextprotocol/clientCapabilities":{}});
        }
        let mut request = json!({"jsonrpc":"2.0","method":method,"params":params});
        if let Some(id) = id {
            request["id"] = id;
        }
        self.transcript
            .push(json!({"direction":"in","message":request}));
        let mut bytes = serde_json::to_vec(&request).unwrap();
        bytes.push(b'\n');
        self.writer.write_all(&bytes).await.unwrap();
        self.writer.flush().await.unwrap();
    }
    async fn read_for(&mut self, id: Value) -> Value {
        let key = id.to_string();
        if let Some(value) = self.pending.remove(&key) {
            return value;
        }
        loop {
            let mut bytes = Vec::new();
            let n = tokio::time::timeout(
                Duration::from_secs(3),
                self.reader.read_until(b'\n', &mut bytes),
            )
            .await
            .expect("RPC timeout")
            .unwrap();
            assert!(n > 0, "unexpected MCP EOF");
            assert!(bytes.len() <= MAX_FRAME && bytes.ends_with(b"\n"));
            let value: Value =
                serde_json::from_slice(&bytes).expect("stdout contains only JSON-RPC");
            assert_eq!(value["jsonrpc"], "2.0");
            self.transcript
                .push(json!({"direction":"out","message":value}));
            if value["id"] == id {
                return value;
            }
            self.pending.insert(value["id"].to_string(), value);
        }
    }
    async fn call(&mut self, method: &str, params: Value) -> Value {
        self.counter += 1;
        let id = json!(self.counter);
        self.send(method, params, Some(id.clone())).await;
        self.read_for(id).await
    }
    async fn tool(&mut self, name: &str, args: Value) -> Value {
        self.call("tools/call", json!({"name":name,"arguments":args}))
            .await
    }
    async fn finish(mut self, name: &str) {
        self.writer.shutdown().await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), self.server)
            .await
            .expect("server EOF shutdown")
            .unwrap()
            .unwrap();
        let mut rest = Vec::new();
        self.reader.read_to_end(&mut rest).await.unwrap();
        for line in rest.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
            let value: Value = serde_json::from_slice(line).expect("stdout tail is JSON-RPC only");
            assert_eq!(value["jsonrpc"], "2.0");
            self.transcript
                .push(json!({"direction":"out","message":value}));
        }
        // Resolve the single lane target from the product source, independently of a proof crate.
        let dir = proof_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!("{}-{}", name, self.version);
        let transcript = dir.join(format!("{name}.json"));
        std::fs::write(transcript,serde_json::to_vec_pretty(&json!({"version":self.version,"frames":self.transcript,"requests":self.backend.requests.lock().unwrap().clone()})).unwrap()).unwrap();
    }
}
fn run_spec() -> Value {
    json!({"dataset":DATASET,"target":"target","task":"regression","ignore_columns":[],"test_dataset":null,"split":{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},"objective":"rmse","effort":"fast","budget":{"evaluations":"10","wall_seconds":"10"},"landmarks":"none","guidance":"heuristic","languages":["rust"],"require_all_languages":true,"device":"cpu","seed":"1","dry_run":false,"plan_digest":null,"idempotency_key":"start-key"})
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
            assert!(result["result"]["content"][0]["text"].is_string());
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

fn proof_dir() -> std::path::PathBuf {
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
