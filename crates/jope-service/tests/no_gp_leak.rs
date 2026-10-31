//! T2 pre-stage sentinel harness.
//!
//! These fixtures exercise the existing public projection and wire shapes with synthetic
//! internal fields. They are not a full discovery run. The PTY case remains gated on TUI-09.

use axum::{
    body::{to_bytes, Body},
    http::{header, Method, Request, StatusCode},
};
use jope_api::{decimal::DecimalU64, ids::RunId, PublicEvent};
use jope_service::http::{
    auth::{BearerToken, LocalAuth},
    router,
};
use jope_service::public::{
    phrase::{CatalogEntry, EventDetails, Template},
    project::{AdmittedProposal, EvaluationOrdinal, ProblemDigest, Projector},
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
use tower::ServiceExt;

const SENTINELS: [&str; 4] = [
    "T2_DSL_SENTINEL_8f27",
    "T2_OPERATOR_SENTINEL_19ac",
    "T2_LINEAGE_SENTINEL_43de",
    "T2_ROUND_SENTINEL_71b0",
];

#[derive(Debug)]
struct InternalFixture {
    dsl: String,
    operator: String,
    lineage: String,
    round: String,
}

impl InternalFixture {
    fn synthetic() -> Self {
        Self {
            dsl: SENTINELS[0].into(),
            operator: SENTINELS[1].into(),
            lineage: SENTINELS[2].into(),
            round: SENTINELS[3].into(),
        }
    }

    fn opaque_round_ordinal(&self) -> EvaluationOrdinal {
        // The public projection accepts only an ordinal and emits a digest-derived ID.
        EvaluationOrdinal(self.round.len() as u64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Http,
    Sse,
    Mcp,
    CliStdout,
    CliStderr,
    Terminal,
    Error,
    ExportManifest,
    Log,
}

#[derive(Debug)]
struct Leak {
    surface: Surface,
    sentinel: &'static str,
}

fn scan_surfaces(outputs: &[(Surface, String)]) -> Result<(), Leak> {
    for (surface, output) in outputs {
        for sentinel in SENTINELS {
            if output.contains(sentinel) {
                return Err(Leak {
                    surface: *surface,
                    sentinel,
                });
            }
        }
    }
    Ok(())
}

fn projected_event(internal: &InternalFixture) -> PublicEvent {
    // Internal DSL/operator/lineage values are deliberately not inputs to this public shape.
    let run_id = RunId::new();
    let projector = Projector::new(run_id.clone(), ProblemDigest([3; 32]));
    let proposal = AdmittedProposal {
        ordinal: internal.opaque_round_ordinal(),
        template: Template::StageInsert,
        event_digest: [9; 32],
        entry: CatalogEntry {
            public_name: "log scaling",
            purpose: "reduce skew in numeric features",
        },
        details: EventDetails::default(),
    };
    PublicEvent {
        run_id,
        sequence: DecimalU64(1),
        time: "2026-10-30T00:00:00Z".into(),
        event: projector.experiment_started(&proposal).unwrap(),
    }
}

fn synthetic_surface_outputs(event: &PublicEvent) -> Vec<(Surface, String)> {
    let wire = serde_json::to_string(event).unwrap();
    let mcp = serde_json::json!({
        "content": [{"type": "text", "text": wire}],
        "isError": false,
    });
    let manifest = serde_json::json!({
        "schema_version": 1,
        "model_object_digest": "0".repeat(64),
        "retains_training_data": false,
        "entries": {"predictor.py": "1".repeat(64)},
    });

    vec![
        (Surface::Http, wire.clone()),
        (Surface::Sse, format!("data: {wire}\n\n")),
        (Surface::Mcp, mcp.to_string()),
        (Surface::CliStdout, format!("{wire}\n")),
        (Surface::CliStderr, String::new()),
        (Surface::Terminal, format!("terminal transcript: {wire}")),
        (Surface::Error, "The request could not be completed.".into()),
        (Surface::ExportManifest, manifest.to_string()),
        (Surface::Log, format!("public event: {wire}")),
    ]
}

#[test]
fn synthetic_internal_fields_stay_out_of_current_public_wire_surfaces() {
    let internal = InternalFixture::synthetic();
    assert_eq!(internal.dsl, SENTINELS[0]);
    assert_eq!(internal.operator, SENTINELS[1]);
    assert_eq!(internal.lineage, SENTINELS[2]);

    let outputs = synthetic_surface_outputs(&projected_event(&internal));
    scan_surfaces(&outputs).unwrap();
}

#[test]
fn scanner_reports_a_deliberately_leaking_surface() {
    let outputs = vec![(Surface::Mcp, format!("tool output: {}", SENTINELS[1]))];
    let leak = scan_surfaces(&outputs).unwrap_err();
    assert_eq!(leak.surface, Surface::Mcp);
    assert_eq!(leak.sentinel, SENTINELS[1]);
}

#[test]
fn scanner_rejects_each_sentinel_on_every_public_surface() {
    let surfaces = [
        Surface::Http,
        Surface::Sse,
        Surface::Mcp,
        Surface::CliStdout,
        Surface::CliStderr,
        Surface::Terminal,
        Surface::Error,
        Surface::ExportManifest,
        Surface::Log,
    ];

    for surface in surfaces {
        for sentinel in SENTINELS {
            let output = format!("leaking {surface:?} capture: {sentinel}");
            let leak = scan_surfaces(&[(surface, output)]).unwrap_err();
            assert_eq!(leak.surface, surface);
            assert_eq!(leak.sentinel, sentinel);
        }
    }
}

#[test]
#[ignore = "terminal transcript scan requires the TUI-09 PTY surface"]
fn terminal_transcript_contains_no_internal_sentinels() {
    // TUI-09 will replace this pre-stage placeholder with captured PTY transcript bytes.
    let transcript = String::new();
    scan_surfaces(&[(Surface::CliStdout, transcript)]).unwrap();
}

fn local_router() -> axum::Router {
    let token = BearerToken::from_hex(&"ab".repeat(32)).unwrap();
    router::router(LocalAuth::new(token, 31234).unwrap())
}

fn authenticated_request(method: Method, uri: &str, body: String) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn live_system_info_http_response_does_not_echo_private_request_metadata() {
    let query = format!(
        "dsl={}&operator={}&lineage={}&round={}",
        SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
    );
    let request = Request::builder()
        .method(Method::GET)
        .uri(format!("/v1/system?{query}"))
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header("x-internal-dsl", SENTINELS[0])
        .header("x-internal-operator", SENTINELS[1])
        .header("x-internal-lineage", SENTINELS[2])
        .header("x-internal-round", SENTINELS[3])
        .body(Body::empty())
        .unwrap();
    let response = local_router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let info: jope_api::SystemInfo = serde_json::from_slice(&body).unwrap();
    assert_eq!(info.health, "ok");
    scan_surfaces(&[(Surface::Http, String::from_utf8(body.to_vec()).unwrap())]).unwrap();
}

#[tokio::test]
async fn current_dataset_route_refusals_do_not_echo_private_payloads() {
    for operation in jope_api::registry::OPERATIONS
        .iter()
        .filter(|operation| operation.name.starts_with("dataset_"))
    {
        for path in operation.paths {
            let path = path.replace("{id}", "ds_00112233445566778899aabb");
            let query = format!(
                "?dsl={}&operator={}&lineage={}&round={}",
                SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
            );
            let body = serde_json::json!({
                "internal": {
                    "dsl": SENTINELS[0],
                    "operator": SENTINELS[1],
                    "lineage": SENTINELS[2],
                    "round": SENTINELS[3]
                }
            })
            .to_string();
            let method: Method = operation.method.as_str().parse().unwrap();
            let request = authenticated_request(method, &format!("{path}{query}"), body);
            let response = local_router().oneshot(request).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNPROCESSABLE_ENTITY,
                "{} {path}",
                operation.name
            );
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let problem: jope_api::error::Problem = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                problem.code,
                jope_api::error::ErrorCode::CapabilityUnavailable
            );
            scan_surfaces(&[(Surface::Error, String::from_utf8(bytes.to_vec()).unwrap())]).unwrap();
        }
    }
}

#[tokio::test]
async fn current_run_and_model_responses_do_not_echo_private_payloads() {
    let operations = jope_api::registry::OPERATIONS.iter().filter(|operation| {
        operation.name.starts_with("run_")
            || operation.name.starts_with("model_")
            || matches!(
                operation.name,
                "algorithm_get" | "artifact_download" | "feedback_export"
            )
    });
    let mut checked_paths = 0;

    for operation in operations {
        for path in operation.paths {
            let path = path.replace("{id}", "run_01J00000000000000000000000");
            let query = format!(
                "?dsl={}&operator={}&lineage={}&round={}",
                SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
            );
            let body = serde_json::json!({
                "internal": {
                    "dsl": SENTINELS[0],
                    "operator": SENTINELS[1],
                    "lineage": SENTINELS[2],
                    "round": SENTINELS[3]
                }
            })
            .to_string();
            let method: Method = operation.method.as_str().parse().unwrap();
            let mut request = authenticated_request(method, &format!("{path}{query}"), body);
            if operation.name == "run_events" {
                request.headers_mut().insert(
                    header::ACCEPT,
                    "text/event-stream".parse().expect("static Accept header"),
                );
            }
            let response = local_router().oneshot(request).await.unwrap();
            let status = response.status();
            assert_eq!(
                status,
                StatusCode::UNPROCESSABLE_ENTITY,
                "{} {path}",
                operation.name
            );
            let is_sse = response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"));
            let response_headers = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    format!("{}: {}", name, String::from_utf8_lossy(value.as_bytes()))
                })
                .collect::<Vec<_>>()
                .join("\n");
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let problem: jope_api::error::Problem = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                problem.code,
                jope_api::error::ErrorCode::CapabilityUnavailable,
                "{} {path}",
                operation.name
            );
            let output = format!("{response_headers}\n{}", String::from_utf8_lossy(&bytes));
            let surface = if is_sse {
                Surface::Sse
            } else if status.is_success() {
                Surface::Http
            } else {
                Surface::Error
            };
            scan_surfaces(&[(surface, output)])
                .unwrap_or_else(|leak| panic!("{} {path} leaked {leak:?}", operation.name));
            checked_paths += 1;
        }
    }

    assert!(checked_paths >= 10, "run/model operation inventory changed");
}

// Build the same tiny real native worker used by the service's supervised-worker
// tests. Keeping it here lets this integration test exercise a real bounded run
// without depending on a separately built appliance binary.
fn native_worker(directory: &Path) -> PathBuf {
    let deps = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let mut libraries: Vec<_> = fs::read_dir(&deps)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("libjope_service-")
                && path
                    .extension()
                    .is_some_and(|extension| extension == "rlib")
        })
        .collect();
    libraries
        .sort_by_key(|path| std::cmp::Reverse(fs::metadata(path).unwrap().modified().unwrap()));
    let source = directory.join("native-worker.rs");
    let binary = directory.join("native-worker");
    fs::write(
        &source,
        "fn main() { jope_service::worker::serve(1, |request, matrix| jope_service::discover::r#loop::evaluate_native(request, matrix)).unwrap(); }",
    )
    .unwrap();
    let mut last_error = String::new();
    for library in libraries {
        let mut command = Command::new("rustc");
        command
            .arg("--edition=2021")
            .arg("--crate-name=pub06_native_worker")
            .arg(&source)
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("--extern")
            .arg(format!("jope_service={}", library.display()))
            .arg("-o")
            .arg(&binary);
        #[cfg(target_arch = "aarch64")]
        command.args(["-C", "target-feature=+fullfp16"]);
        let output = command.output().unwrap();
        if output.status.success() {
            return binary;
        }
        last_error = String::from_utf8_lossy(&output.stderr).into_owned();
    }
    panic!("could not link bounded native worker: {last_error}");
}

#[tokio::test]
async fn bounded_real_run_http_status_and_sse_do_not_leak_internal_sentinels() {
    use jope_service::{
        dataset::register,
        engine::InProcessEngine,
        http::auth::{BearerToken, LocalAuth},
    };
    use tokio::time::sleep;

    let target_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("int01");
    fs::create_dir_all(&target_dir).unwrap();
    let home = tempfile::tempdir_in(&target_dir).unwrap();
    let worker = native_worker(home.path());
    let engine = InProcessEngine::open_with_worker(home.path(), worker).unwrap();
    let raw = include_bytes!("../../../test-support/datasets/v1/d2-regression/train.csv");
    let mut headers: Vec<_> = (0..21).map(|index| format!("x{index}")).collect();
    headers.push("target".into());
    let mut csv = headers.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let dataset = register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
    let spec: jope_api::RunSpec = serde_json::from_value(serde_json::json!({
        "dataset": dataset.id,
        "target": "target",
        "task": "regression",
        "ignore_columns": [],
        "test_dataset": null,
        "split": {"kind":"random", "holdout_fraction":0.2, "group_column":null, "time_column":null},
        "objective": "rmse",
        "effort": "fast",
        "budget": {"evaluations":"4", "wall_seconds":"30"},
        "landmarks": "none",
        "guidance": "heuristic",
        "languages": ["rust"],
        "require_all_languages": true,
        "device": "cpu",
        "seed": "73",
        "dry_run": false,
        "plan_digest": null
    }))
    .unwrap();
    let app = engine
        .router(LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap())
        .unwrap();
    let query = format!(
        "?dsl={}&operator={}&lineage={}&round={}",
        SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/v1/runs{query}"))
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, "application/json")
        .header("idempotency-key", "pub06-bounded-run")
        .header("x-internal-dsl", SENTINELS[0])
        .header("x-internal-operator", SENTINELS[1])
        .header("x-internal-lineage", SENTINELS[2])
        .header("x-internal-round", SENTINELS[3])
        .body(Body::from(serde_json::to_vec(&spec).unwrap()))
        .unwrap();
    let started = app.clone().oneshot(request).await.unwrap();
    assert_eq!(started.status(), StatusCode::CREATED);
    let start_headers = started
        .headers()
        .iter()
        .map(|(name, value)| format!("{name}: {}", String::from_utf8_lossy(value.as_bytes())))
        .collect::<Vec<_>>()
        .join("\n");
    let start_body = to_bytes(started.into_body(), 1 << 20).await.unwrap();
    let start_output = format!("{start_headers}\n{}", String::from_utf8_lossy(&start_body));
    scan_surfaces(&[(Surface::Http, start_output)]).unwrap();
    let mut snapshot: jope_api::RunSnapshot = serde_json::from_slice(&start_body).unwrap();
    let deadline = Instant::now() + Duration::from_secs(35);
    while snapshot.state == jope_api::RunState::Running && Instant::now() < deadline {
        sleep(Duration::from_millis(25)).await;
        let response = app
            .clone()
            .oneshot(authenticated_request(
                Method::GET,
                &format!("/v1/runs/{}", snapshot.run_id),
                String::new(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        scan_surfaces(&[(Surface::Http, String::from_utf8_lossy(&bytes).into_owned())]).unwrap();
        snapshot = serde_json::from_slice(&bytes).unwrap();
    }
    assert_eq!(snapshot.state, jope_api::RunState::Completed);
    assert!(snapshot.event_watermark.0 > 0);

    let events = Request::builder()
        .method(Method::GET)
        .uri(format!(
            "/v1/runs/{}/events?after=0&limit=256",
            snapshot.run_id
        ))
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::ACCEPT, "text/event-stream")
        .header("x-internal-dsl", SENTINELS[0])
        .header("x-internal-operator", SENTINELS[1])
        .header("x-internal-lineage", SENTINELS[2])
        .header("x-internal-round", SENTINELS[3])
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(events).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let headers = response
        .headers()
        .iter()
        .map(|(name, value)| format!("{name}: {}", String::from_utf8_lossy(value.as_bytes())))
        .collect::<Vec<_>>()
        .join("\n");
    let mut stream = response.into_body().into_data_stream();
    use futures_util::StreamExt;
    let mut frames = Vec::new();
    for _ in 0..snapshot.event_watermark.0 {
        let Some(frame) = stream.next().await else {
            break;
        };
        frames.push(frame.unwrap());
    }
    assert!(!frames.is_empty());
    let output = format!(
        "{headers}\n{}",
        frames
            .iter()
            .map(|frame| String::from_utf8_lossy(frame.as_ref()))
            .collect::<String>()
    );
    scan_surfaces(&[(Surface::Sse, output)]).unwrap();
}
