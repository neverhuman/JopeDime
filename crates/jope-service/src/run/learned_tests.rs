use super::{learned::AdmittedRanker, service::BackendPlan, surrogate};
use crate::{
    dataset::register,
    engine::InProcessEngine,
    http::auth::{BearerToken, LocalAuth},
    ranker::cuda::release::load_r782_release,
};
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use jope_api::{error::ErrorCode, RunSpec};
use serde_json::{json, Value};
use std::{
    env,
    path::PathBuf,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use tower::ServiceExt;

#[test]
fn frozen_surrogate_gate_and_fitness_ranges_match_reference() {
    let (value, gate) = surrogate::fitness("(pipe X (ridge))", -0.4, -0.2, 2).unwrap();
    assert!(gate);
    let expected = 0.8 * -0.3 + 0.1 * -0.2 + 0.1 * 0.1 + 0.0005 * 2.0;
    assert!((value - expected).abs() < 1e-15);
    for (a, b, wins) in [(0.1, -0.3, 1), (0.1, 0.3, 0), (-0.01, -0.01, 2)] {
        let (f, g) = surrogate::fitness("(pipe X (ridge))", a, b, wins).unwrap();
        assert!(!g);
        assert!(f >= 100.0);
    }
    assert!(surrogate::fitness("x", f64::NAN, 0.0, 0).is_err());
    assert!(surrogate::fitness("x", 0.0, 0.0, 3).is_err());
}

fn spec(dataset: &str, device: &str) -> RunSpec {
    serde_json::from_value(json!({"dataset":dataset,"target":"target","task":"regression",
        "ignore_columns":[],"test_dataset":null,"split":{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},
        "objective":"rmse","effort":"fast","budget":{"evaluations":"12","wall_seconds":"60"},
        "landmarks":"full","guidance":"learned","languages":["rust"],"require_all_languages":true,
        "device":device,"seed":"73","dry_run":true,"plan_digest":null})).unwrap()
}

#[test]
fn learned_requires_explicit_admission_and_invalid_device_is_not_fallback() {
    let mut request = spec("ds_0123456789abcdef01234567", "cpu");
    let err = BackendPlan::resolve(&request, None, 1).err().unwrap();
    assert_eq!(
        super::service::error_code(&err),
        ErrorCode::CapabilityUnavailable
    );
    request.device = "cuda:garbage".into();
    let err = BackendPlan::resolve(&request, None, 1).err().unwrap();
    assert_eq!(super::service::error_code(&err), ErrorCode::InvalidInput);
}

async fn post(app: axum::Router, request: &RunSpec, key: &str) -> (u16, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/runs")
                .header("host", "127.0.0.1:31234")
                .header("authorization", format!("Bearer {}", "ab".repeat(32)))
                .header("content-type", "application/json")
                .header("idempotency-key", key)
                .body(Body::from(serde_json::to_vec(request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
    (status, value)
}

#[tokio::test]
#[ignore = "requires exact release-signed r782 extraction and source-bound real worker binary; nonsealed real TRAIN only"]
async fn signed_r782_live_plan_and_admission_use_the_instantiated_backend() {
    let root =
        PathBuf::from(env::var_os("JOPE_AUTHENTICATED_BUNDLE_DIR").expect("signed r782 required"));
    let worker =
        PathBuf::from(env::var_os("JOPE_RNK12_WORKER_BIN").expect("source-bound worker required"));
    let ranker = AdmittedRanker::r782(load_r782_release(&root).unwrap()).unwrap();
    let forwards = ranker.execution_counter();
    let dir = tempfile::tempdir().unwrap();
    let engine = InProcessEngine::open_with_ranker(dir.path(), &worker, ranker).unwrap();
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(
        jope_runtime::sha256_hex(raw),
        "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
    );
    let mut csv = (0..21)
        .map(|i| format!("x{i}"))
        .chain(std::iter::once("target".into()))
        .collect::<Vec<_>>()
        .join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let data = register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
    let app = engine
        .router(LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap())
        .unwrap();
    let device = env::var("JOPE_RNK12_LIVE_DEVICE").unwrap_or_else(|_| "cpu".into());
    // CUDA qualification retains the validated inherited exclusive lease across
    // both async HTTP requests and the worker's completed finalization.
    #[cfg(unix)]
    let _gpu_lease = if device.starts_with("cuda:") {
        use std::os::fd::AsFd;
        assert_eq!(env::var("JOPE_GPU_LEASE_STDIN").as_deref(), Ok("1"));
        let path = PathBuf::from(env::var_os("JOPE_QUALIFICATION_LOCK").unwrap());
        let lease = std::fs::File::from(std::io::stdin().as_fd().try_clone_to_owned().unwrap());
        let retained = lease.try_clone().unwrap();
        Some(crate::ranker::cuda::inherited_lease(lease, &path, || Ok(retained)).unwrap())
    } else {
        None
    };
    #[cfg(not(unix))]
    assert!(
        !device.starts_with("cuda:"),
        "CUDA qualification needs a Unix lease"
    );
    let mut request = spec(&data.id.to_string(), &device);
    let (status, plan) = post(app.clone(), &request, "rnk12-live-plan").await;
    assert_eq!(status, 200, "{plan}");
    let selected = if device == "auto" {
        "cpu"
    } else {
        device.as_str()
    };
    assert_eq!(
        plan["placement"],
        json!({"dope":selected,"ranker":selected,"evaluation":"cpu","workers":1})
    );
    assert_eq!(plan["guidance"]["value"], "learned");
    assert_eq!(plan["landmarks"]["mode"], "full");
    assert_eq!(plan["landmarks"]["sample_rows"], 76);
    request.dry_run = false;
    request.plan_digest = Some(plan["plan_digest"].as_str().unwrap().into());
    let (status, run) = post(app.clone(), &request, "rnk12-live-start").await;
    assert_eq!(status, 201, "{run}");
    let id = run["run_id"].as_str().unwrap();
    let start = Instant::now();
    let finished = loop {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/runs/{id}"))
                    .header("host", "127.0.0.1:31234")
                    .header("authorization", format!("Bearer {}", "ab".repeat(32)))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap())
                .unwrap();
        if value["state"] == "completed" || value["state"] == "failed" {
            break value;
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "run failed to drain: {value}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    println!(
        "{}",
        json!({"consumer":"RNK-12/live-RunService","plan":plan,"run":finished,
        "real_learned_forward_calls":forwards.load(Ordering::Relaxed),"source":env::var("JOPE_SOURCE_SHA").unwrap(),
        "worker":worker,"original_bundle":"94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c","default_or_Q1_promotion":false})
    );
    assert_eq!(finished["state"], "completed", "{finished}");
    assert!(
        forwards.load(Ordering::Relaxed) > 0,
        "live engine never used its learned backend"
    );
    assert_eq!(finished["budget_used"]["evaluations"], "12");
}
