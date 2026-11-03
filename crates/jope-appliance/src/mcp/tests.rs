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

#[test]
fn stdio_boundary_accepts_synchronous_callers() {
    assert!(ensure_stdio_runtime_context().is_ok());
}

#[tokio::test]
async fn stdio_boundary_rejects_async_runtime_callers() {
    let adapter = Adapter::new(Client(Arc::new(Fixture::default())));
    let error = adapter.serve_stdio().unwrap_err();
    assert!(error
        .to_string()
        .contains("cannot run inside an async runtime"));
}

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
        let dir = runtime::proof_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!("{}-{}", name, self.version);
        let transcript = dir.join(format!("{name}.json"));
        std::fs::write(transcript,serde_json::to_vec_pretty(&json!({"version":self.version,"frames":self.transcript,"requests":self.backend.requests.lock().unwrap().clone()})).unwrap()).unwrap();
    }
}
fn run_spec() -> Value {
    json!({"dataset":DATASET,"target":"target","task":"regression","ignore_columns":[],"test_dataset":null,"split":{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},"objective":"rmse","effort":"fast","budget":{"evaluations":"10","wall_seconds":"10"},"landmarks":"none","guidance":"heuristic","languages":["rust"],"require_all_languages":true,"device":"cpu","seed":"1","dry_run":false,"plan_digest":null,"idempotency_key":"start-key"})
}

#[path = "tests/bootstrap.rs"]
mod bootstrap;
#[path = "tests/runtime.rs"]
mod runtime;
