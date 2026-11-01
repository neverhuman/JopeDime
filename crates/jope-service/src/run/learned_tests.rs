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
fn user_objectives_keep_runtime_identity_and_refuse_unavailable_inputs() {
    use super::service::run_objective;
    use jope_api::{Direction, ObjectiveSpec};
    let named = ObjectiveSpec::Name("rmse".into());
    let equivalent = serde_json::from_value(json!({"metric":"rmse"})).unwrap();
    let (_, rmse) = run_objective(&named).unwrap();
    assert_eq!(run_objective(&equivalent).unwrap().1, rmse);
    assert_eq!(rmse.fitting, "squared_error");
    let custom = serde_json::from_value(json!({"metric":"custom","name":"rnk10_mean_prediction",
        "direction":"maximize","expression":{"op":"pred"},"aggregation":{"kind":"mean"}}))
    .unwrap();
    let (text, resolved) = run_objective(&custom).unwrap();
    assert_eq!(resolved.direction, Direction::Maximize);
    assert_eq!(resolved.fitting, "selection_only");
    assert_eq!(resolved.name, "custom objective");
    assert_eq!(
        resolved.digest,
        jope_runtime::fitness::Fitness::parse(&text)
            .unwrap()
            .identity()
            .unwrap()
    );
    for bad in [
        json!("unknown_metric"),
        json!({"metric":"custom","name":"column_needed",
        "direction":"minimize","expression":{"op":"column","name":"cost"},"aggregation":{"kind":"mean"}}),
        json!("mase"),
    ] {
        assert!(run_objective(&serde_json::from_value(bad).unwrap()).is_err());
    }
}

#[test]
fn trained_surrogate_stays_frozen_while_actual_custom_loss_drives_selection() {
    use crate::{
        dataset::jmat,
        worker::{Attempt, Request},
    };
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(
        jope_runtime::sha256_hex(raw),
        "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
    );
    let mut table = jope_runtime::table::parse_csv(raw, &Default::default()).unwrap();
    table.rows.truncate(32);
    let target = table.headers.last().unwrap().clone();
    let features = table.headers[..table.headers.len() - 1].to_vec();
    let bytes = jmat::encode(&table).unwrap();
    let matrix = jmat::Matrix::open(&bytes).unwrap();
    let rows: Vec<usize> = (0..32).collect();
    let mut request = Request {
        attempt: Attempt {
            run: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            round: 0,
            slot: 0,
            attempt: 0,
        },
        matrix_path: "unused-mapped-input.jmat".into(),
        fold_spec: json!({"version":2,"matrix_sha256":jope_runtime::sha256_hex(&bytes),"target":target,
            "features":features,"folds":[{"repeat":0,"train":rows[..16],"validation":rows[16..]},
            {"repeat":0,"train":rows[16..],"validation":rows[..16]}]}),
        dsl: "(pipe X (standardize) (ridge))".into(),
        objective: json!("rmse"),
        seed: 73,
        limits: crate::worker::Limits {
            wall_ms: 30_000,
            rss_bytes: 1_073_741_824,
        },
    };
    let rmse = surrogate::native(&request, &matrix).unwrap();
    let custom = json!({"metric":"custom","name":"rnk10_mean_prediction","direction":"maximize",
        "expression":{"op":"pred"},"aggregation":{"kind":"mean"}})
    .to_string();
    request.objective = json!(custom);
    let scored = surrogate::native(&request, &matrix).unwrap();
    assert_eq!(scored.values[0].to_bits(), rmse.values[0].to_bits());
    assert_eq!(scored.fold_scores, rmse.fold_scores);
    assert_eq!(scored.predictions_digest, rmse.predictions_digest);
    assert!(scored.values[2].is_finite() && scored.values[2] != rmse.values[2]);
    assert_eq!(scored.values[1], -scored.values[2]);
}

#[tokio::test]
#[ignore = "R1: needs release-signed exact r882, source-bound worker, and an unused external receipt directory"]
async fn signed_r882_in_loop_ab_and_replay() {
    use crate::ranker::cuda::release::{load_r882_release, R882};
    use jope_api::{decimal::DecimalU64, Guidance};
    use std::fs;
    let protocol: Value = serde_json::from_str(include_str!(
        "../../../../research/flywheel_selfplay/rnk10_r882_release/PREREG.json"
    ))
    .unwrap();
    let root = PathBuf::from(env::var_os("JOPE_RNK10_SIGNED_BUNDLE_DIR").unwrap());
    // Refuse before opening any study table when the release proof is missing.
    let authenticated = load_r882_release(&root).unwrap();
    assert_eq!(authenticated.digest().to_string(), R882);
    let report = PathBuf::from(env::var_os("JOPE_RNK10_REPORT_DIR").unwrap());
    assert!(report.is_absolute() && !report.exists());
    fs::create_dir(&report).unwrap();
    let worker = PathBuf::from(env::var_os("JOPE_RNK10_WORKER_BIN").unwrap());
    assert_eq!(
        jope_runtime::sha256_hex(&fs::read(&worker).unwrap()),
        env::var("JOPE_RNK10_WORKER_SHA256").unwrap()
    );
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(
        jope_runtime::sha256_hex(raw),
        protocol["train_sha256"].as_str().unwrap()
    );
    let mut csv = (0..21)
        .map(|i| format!("x{i}"))
        .chain(std::iter::once("target".into()))
        .collect::<Vec<_>>()
        .join(",")
        + "\n";
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let mut results = vec![];
    let mut replay = None;
    for (seed, learned, repeated) in [
        (73, true, false),
        (73, false, false),
        (157101, true, false),
        (157101, false, false),
        (157102, true, false),
        (157102, false, false),
        (73, true, true),
    ] {
        let home = tempfile::Builder::new()
            .prefix("engine-")
            .tempdir_in(&report)
            .unwrap();
        let (engine, forwards) = if learned {
            let admitted = AdmittedRanker::r882(load_r882_release(&root).unwrap()).unwrap();
            let count = admitted.execution_counter();
            (
                InProcessEngine::open_with_ranker(home.path(), &worker, admitted).unwrap(),
                Some(count),
            )
        } else {
            (
                InProcessEngine::open_with_worker(home.path(), &worker).unwrap(),
                None,
            )
        };
        let data = register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
        let app = engine
            .router(
                LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap(),
            )
            .unwrap();
        let mut request = spec(&data.id.to_string(), "cpu");
        request.seed = DecimalU64(seed);
        request.guidance = if learned {
            Guidance::Learned
        } else {
            Guidance::Heuristic
        };
        let (status, plan) = post(app.clone(), &request, "rnk10-plan").await;
        assert_eq!(status, 200, "{plan}");
        assert_eq!(
            plan["guidance"]["value"],
            if learned { "learned" } else { "heuristic" }
        );
        assert_eq!(
            plan["placement"]["ranker"],
            if learned { "cpu" } else { "heuristic" }
        );
        request.dry_run = false;
        request.plan_digest = Some(plan["plan_digest"].as_str().unwrap().into());
        let (status, started) = post(app.clone(), &request, "rnk10-start").await;
        assert_eq!(status, 201, "{started}");
        let id = started["run_id"].as_str().unwrap();
        let began = Instant::now();
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
            assert!(began.elapsed() < Duration::from_secs(120), "{value}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(finished["state"], "completed", "{finished}");
        assert_eq!(finished["budget_used"]["evaluations"], "12");
        let cost = finished["best"]["cv_value"].as_f64().unwrap();
        assert!(cost.is_finite());
        let count = forwards.map_or(0, |v| v.load(Ordering::Relaxed));
        assert_eq!(
            count > 0,
            learned,
            "actual loop must invoke its selected backend"
        );
        let trace = json!({"algorithm":finished["best"]["algorithm_id"],"value":cost});
        if seed == 73 && learned {
            if repeated {
                assert_eq!(replay.as_ref().unwrap(), &trace, "learned replay differs");
            } else {
                replay = Some(trace);
            }
        }
        let row = json!({"seed":seed,"policy":if learned {"r882"} else {"heuristic"},"repeat":repeated,
            "plan":plan,"run":finished,"real_learned_forwards":count});
        fs::write(
            report.join(format!("cell-{seed}-{learned}-{repeated}.json")),
            serde_json::to_vec_pretty(&row).unwrap(),
        )
        .unwrap();
        results.push(row);
    }
    let receipt = json!({"schema":"jope.rnk10.in-loop-ab/v1","source_commit":env::var("JOPE_SOURCE_SHA").unwrap(),
        "worker_sha256":env::var("JOPE_RNK10_WORKER_SHA256").unwrap(),"protocol":protocol,
        "bundle_digest":R882,"signed_loader":true,"results":results,"default_enabled":false,
        "quality_qualified":false,"scope":"one TRAIN family; operational comparison, no fresh validation claim"});
    fs::write(
        report.join("ab-receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    println!("{receipt}");
}

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
