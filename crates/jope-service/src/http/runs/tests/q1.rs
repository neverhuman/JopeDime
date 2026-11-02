use super::*;

#[test]
fn q1_cut_hash_matches_canonical_cross_language_control() {
    let binding = Q1DatasetBinding {
        id: "0000000000000001".into(),
        task: "regression".into(),
        dataset_sha256: "a".repeat(64),
        family: "f".repeat(40),
        family_authoritative: true,
        rows: 4,
        features: 2,
        cut_sha256: String::new(),
        holdout_group: "development_fit".into(),
        catalog_version: "v1".into(),
        catalog_record_sha256: "c".repeat(64),
        dataset_binding_sha256: String::new(),
        target: "y".into(),
        feature_columns: vec!["x1".into(), "x2".into()],
        train_rows: 4,
        train_blob_sha256: "b".repeat(64),
        train_blob_bytes: 1,
        folds: vec![
            Q1Fold {
                repeat: 0,
                train: vec![0, 1],
                validation: vec![2, 3],
            },
            Q1Fold {
                repeat: 0,
                train: vec![2, 3],
                validation: vec![0, 1],
            },
        ],
    };
    assert_eq!(
        q1_cut_sha256(&binding).unwrap(),
        "0c87e715ccb1a01fe05d4bc4e8b81f62f978d8aed0126248d97b866686acf249"
    );
    let identity = Q1DatasetIdentity {
        id: binding.id.clone(),
        task: binding.task.clone(),
        dataset_sha256: binding.dataset_sha256.clone(),
        family: binding.family.clone(),
        family_authoritative: binding.family_authoritative,
        rows: binding.rows,
        features: binding.features,
        cut_sha256: binding.cut_sha256.clone(),
        holdout_group: binding.holdout_group.clone(),
    };
    let mut mismatched = binding;
    mismatched.cut_sha256 = "d".repeat(64);
    let mut binding_value = serde_json::to_value(&mismatched).unwrap();
    binding_value
        .as_object_mut()
        .unwrap()
        .remove("dataset_binding_sha256");
    mismatched.dataset_binding_sha256 = hash(&serde_json::to_vec(&binding_value).unwrap());
    assert!(validate_q1_binding(&mismatched, &identity).is_err());
}

const Q1_FAIL_ALL_TEST_SEED: u64 = 2;

#[test]
#[ignore = "Q1 worker subprocess target; invoked by the worker process supervisor"]
fn q1_worker_helper() {
    use std::io::Write as _;

    assert_eq!(std::env::var("JOPE_WORKER_FIXTURE").unwrap(), "worker");
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "WORKER_FIXTURE_READY").unwrap();
    stdout.flush().unwrap();
    drop(stdout);
    crate::worker::serve(1, |request, matrix| {
        let request_seed = request
            .seed
            .wrapping_sub(request.attempt.attempt.saturating_sub(1));
        if request_seed == Q1_FAIL_ALL_TEST_SEED {
            Err(crate::worker::WorkerFailure::InvalidRequest)
        } else {
            crate::discover::r#loop::evaluate_native(request, matrix)
        }
    })
    .unwrap();
}

#[test]
fn learned_seed_schedule_cycles_until_successful_measurements_exist() {
    let seeds = vec![
        "seed-a".to_owned(),
        "seed-b".to_owned(),
        "seed-c".to_owned(),
    ];
    let scheduled = (0..24)
        .map(|attempt| q1_learned_seed(&seeds, attempt, &[]))
        .collect::<Vec<_>>();
    assert_eq!(scheduled.len(), 24);
    assert!(scheduled
        .iter()
        .enumerate()
        .all(|(attempt, seed)| *seed == seeds[attempt % seeds.len()]));
    let measured = vec![("seed-b".to_owned(), 1.0)];
    assert_eq!(q1_learned_seed(&seeds, 1, &measured), "seed-c");
}

#[test]
fn q1_attempt_timeout_is_bounded_by_one_total_wall_budget() {
    assert_eq!(
        q1_remaining_wall_ms(std::time::Duration::ZERO),
        Some(600_000)
    );
    assert_eq!(
        q1_remaining_wall_ms(std::time::Duration::from_millis(1250)),
        Some(598_750)
    );
    assert_eq!(
        q1_remaining_wall_ms(Q1_WALL_BUDGET - std::time::Duration::from_millis(1)),
        Some(1)
    );
    assert_eq!(q1_remaining_wall_ms(Q1_WALL_BUDGET), None);
    assert_eq!(
        q1_remaining_wall_ms(Q1_WALL_BUDGET + std::time::Duration::from_secs(1)),
        None
    );
}
#[test]
fn q1_expired_policy_never_invokes_the_evaluator() {
    use crate::discover::r#loop::Executor;
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let executor = Q1PinnedExecutor {
        inner: InlineQ1Native {
            calls: Arc::clone(&calls),
        },
        folds: json!({}),
        started_at: std::time::Instant::now() - Q1_WALL_BUDGET,
        attempts: Mutex::new(vec![]),
    };
    let request = crate::worker::Request {
        attempt: crate::worker::Attempt {
            run: jope_api::ids::RunId::new(),
            round: 0,
            slot: 0,
            attempt: 1,
        },
        matrix_path: PathBuf::new(),
        fold_spec: json!({}),
        dsl: "(pipe X (ridge))".into(),
        objective: json!("rmse"),
        seed: 0,
        limits: crate::worker::Limits {
            wall_ms: 600_000,
            rss_bytes: WORKER_RAM,
        },
    };
    let response = executor.evaluate(&request);
    assert_eq!(
        response.failure,
        Some(crate::worker::WorkerFailure::Timeout)
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    let attempts = executor.attempts.into_inner().unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].1.outcome, EvaluationAttemptOutcome::Failed);
    assert!(attempts[0].1.loss.is_none() && attempts[0].1.fold_scores.is_empty());
}

struct Q1AllFailed;
impl crate::discover::r#loop::Executor for Q1AllFailed {
    fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
        crate::worker::Response::failed(
            request.attempt.clone(),
            crate::worker::WorkerFailure::NonFinite,
            0,
        )
    }
    fn fitted_model(&self, _: &crate::worker::Request) -> anyhow::Result<Vec<u8>> {
        panic!("failed candidates cannot be refitted")
    }
}

#[test]
fn q1_all_failed_attempts_commit_terminal_replayable_actual_counts() {
    let (directory, service) = q1_service_fixture();
    let request = q1_request(directory.path(), &service.store);
    let bytes = service
        .q1_evaluate_using(request.clone(), || Ok(Q1AllFailed))
        .unwrap();
    let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(receipt.charged, 24);
    assert_eq!(receipt.attempts.len(), 24);
    assert!(receipt.champion_program.is_none());
    assert!(receipt
        .attempts
        .iter()
        .enumerate()
        .all(|(number, a)| a.attempt == number as u64
            && a.outcome == EvaluationAttemptOutcome::Failed
            && a.failure.is_some()
            && a.loss.is_none()
            && a.fold_scores.is_empty()));
    let command = service
        .store
        .get(&q1_command_key(&receipt.request_sha256))
        .unwrap()
        .unwrap();
    assert_eq!(command.data["status"], "completed");
    let run: jope_api::ids::RunId = command.data["run_id"].as_str().unwrap().parse().unwrap();
    let snapshot = service.journal.snapshot(&run).unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Failed);
    assert!(snapshot.pending.is_none());
    assert_eq!(snapshot.budget_used.evaluations.0, 24);
    let replay = service
        .q1_evaluate_using(request, || -> anyhow::Result<Q1AllFailed> {
            panic!("terminal replay must never invoke an executor factory")
        })
        .unwrap();
    assert_eq!(bytes, replay);
}

#[test]
fn q1_deterministic_search_stop_commits_only_actual_attempts() {
    let (directory, service) = q1_service_fixture();
    let request_value = q1_request(directory.path(), &service.store);
    let request: Q1Request = serde_json::from_value(request_value).unwrap();
    let manifest = load_q1_manifest(directory.path(), &request.pins["protocol_sha256"]).unwrap();
    let dataset_binding_sha256 = manifest.datasets[&request.dataset.id]
        .dataset_binding_sha256
        .clone();
    let run_id = jope_api::ids::RunId::new();
    assert!(service
        .reserve_q1_request(&request, &run_id)
        .unwrap()
        .is_none());
    let attempt = jope_api::EvaluationAttempt {
        attempt: 0,
        outcome: EvaluationAttemptOutcome::Failed,
        fold_scores: vec![],
        loss: None,
        program_sha256: None,
        failure: Some(jope_api::EvaluationFailure {
            code: "non_finite".into(),
            reason: "actual evaluator attempt failed".into(),
        }),
    };
    let receipt = EvaluationReceipt {
        schema: "jope.q1-evaluation-receipt/v1".into(),
        request_sha256: request.request_sha256.clone(),
        identity: EvaluationIdentity {
            source_commit: request.source_commit.clone(),
            plan_sha256: request.plan_sha256.clone(),
            pins: request.pins.clone(),
            dataset_id: request.dataset.id.clone(),
            dataset_sha256: request.dataset.dataset_sha256.clone(),
            cut_sha256: request.dataset.cut_sha256.clone(),
            guidance: request.guidance.clone(),
            metric: request.metric.clone(),
            seed: request.seed,
            requested_budget: request.budget,
            dataset_binding_sha256,
        },
        charged: 1,
        attempts: vec![attempt],
        champion_program: None,
        adapter: "test".into(),
    };
    let bytes = serde_json::to_vec(&receipt).unwrap();
    service
        .commit_q1_receipt(
            &request,
            &run_id,
            receipt.clone(),
            bytes.clone(),
            Q1ReceiptCompletion {
                attempt_events: vec![],
                elapsed: std::time::Duration::from_millis(5),
                terminal_failure: Some("synthetic deterministic discovery stop".into()),
            },
        )
        .unwrap();

    let snapshot = service.journal.snapshot(&run_id).unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Failed);
    assert_eq!(snapshot.pending, None);
    assert_eq!(snapshot.budget_used.evaluations.0, 1);
    assert_eq!(snapshot.evaluation_receipt, Some(receipt));
    let command = service
        .store
        .get(&q1_command_key(&request.request_sha256))
        .unwrap()
        .unwrap();
    assert_eq!(command.data["status"], "completed");
    assert_eq!(
        command.data["terminal_failure"],
        "synthetic deterministic discovery stop"
    );
    assert_eq!(
        command.data["receipt_bytes"],
        String::from_utf8(bytes.clone()).unwrap()
    );
    assert_eq!(q1_replay_or_unknown(command.data).unwrap(), Some(bytes));
}

fn q1_service_fixture() -> (tempfile::TempDir, RunService) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
    std::fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let lease = Arc::new(EngineLock::acquire(directory.path()).unwrap());
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let config = Config::load(
        &directory.path().join("config.toml"),
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    let service = RunService::new(
        store,
        config,
        directory.path().to_path_buf(),
        std::env::current_exe().unwrap(),
        Arc::new(EngineActivity::default()),
        lease,
    )
    .unwrap();
    (directory, service)
}

fn q1_request(directory: &std::path::Path, store: &Store) -> Value {
    let train = b"x,target\n0,0\n1,1\n2,2\n3,3\n4,4\n5,5\n";
    let blob = store.publish_blob(train.as_slice()).unwrap();
    let protocol_sha = "a".repeat(64);
    let plan_sha = "b".repeat(64);
    let source_commit = "c".repeat(40);
    let pins = [
        "bundle_sha256",
        "dope_sha256",
        "evaluator_sha256",
        "vocabulary_sha256",
        "protocol_sha256",
        "training_family_manifest_sha256",
    ]
    .into_iter()
    .map(|key| {
        (
            key.to_owned(),
            if key == "protocol_sha256" {
                protocol_sha.clone()
            } else {
                "d".repeat(64)
            },
        )
    })
    .collect::<std::collections::BTreeMap<_, _>>();
    let mut binding = Q1DatasetBinding {
        id: "0123456789abcdef".into(),
        task: "regression".into(),
        dataset_sha256: "0123456789abcdef".to_owned() + &"e".repeat(48),
        family: "f".repeat(40),
        family_authoritative: true,
        rows: 10,
        features: 1,
        cut_sha256: String::new(),
        holdout_group: "sealed_meta_test".into(),
        catalog_version: "v246".into(),
        catalog_record_sha256: "2".repeat(64),
        dataset_binding_sha256: String::new(),
        target: "target".into(),
        feature_columns: vec!["x".into()],
        train_rows: 6,
        train_blob_sha256: hash(train),
        train_blob_bytes: blob.bytes,
        folds: (0..3)
            .map(|fold| Q1Fold {
                repeat: 0,
                train: (0..6).filter(|row| row / 2 != fold).collect(),
                validation: (fold * 2..(fold + 1) * 2).collect(),
            })
            .collect(),
    };
    binding.cut_sha256 = q1_cut_sha256(&binding).unwrap();
    let mut binding_value = serde_json::to_value(&binding).unwrap();
    binding_value
        .as_object_mut()
        .unwrap()
        .remove("dataset_binding_sha256");
    binding.dataset_binding_sha256 = hash(&serde_json::to_vec(&binding_value).unwrap());
    let manifest = json!({
        "schema":"jope.q1-protocol-manifest/v1",
        "protocol_sha256":protocol_sha,
        "source_commit":source_commit,
        "plan_sha256":plan_sha,
        "pins":pins,
        "datasets":{"0123456789abcdef":binding}
    });
    let protocols = directory.join("q1/protocols");
    std::fs::create_dir_all(&protocols).unwrap();
    std::fs::write(
        protocols.join(format!("{}.json", "a".repeat(64))),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let mut request = json!({
        "schema":"jope.q1-request/v1",
        "mode":"qualification",
        "source_commit":source_commit,
        "plan_sha256":plan_sha,
        "pins":pins,
        "metric":"rmse",
        "dataset":{
            "id":"0123456789abcdef",
            "task":"regression",
            "dataset_sha256":"0123456789abcdefe".to_owned() + &"e".repeat(47),
            "family":"f".repeat(40),
            "family_authoritative":true,
            "rows":10,
            "features":1,
            "cut_sha256":binding.cut_sha256,
            "holdout_group":"sealed_meta_test"
        },
        "guidance":"heuristic",
        "seed":0,
        "budget":24
    });
    // Match the catalog's full dataset identity; the test partition is never stored here.
    request["dataset"]["dataset_sha256"] = json!("0123456789abcdef".to_owned() + &"e".repeat(48));
    let sha = canonical_sha256(&request).unwrap();
    request["request_sha256"] = json!(sha);
    request
}

fn q1_http_request(body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/q1/evaluations")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

#[tokio::test]
#[ignore = "requires the authenticated signed r782 fixture and a source-bound native worker"]
async fn q1_admitted_r782_learned_request_executes_real_guidance() {
    use crate::{ranker::cuda::release::load_r782_release, run::learned::AdmittedRanker};
    use std::sync::atomic::Ordering;
    let assets = PathBuf::from(std::env::var_os("JOPE_AUTHENTICATED_BUNDLE_DIR").unwrap());
    let worker = PathBuf::from(
        std::env::var_os("JOPE_Q1_WORKER_BIN").expect("source-bound native worker required"),
    );
    let ranker = AdmittedRanker::r782(load_r782_release(&assets).unwrap()).unwrap();
    let forwards = ranker.execution_counter();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
    std::fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let engine = InProcessEngine::open_with_ranker(directory.path(), &worker, ranker).unwrap();
    let mut request = q1_request(directory.path(), engine.store());
    request["pins"]["bundle_sha256"] =
        json!("94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c");
    request["pins"]["dope_sha256"] =
        json!("c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814");
    let path = directory
        .path()
        .join("q1/protocols")
        .join(format!("{}.json", "a".repeat(64)));
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["pins"] = request["pins"].clone();
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    for guidance in ["heuristic", "learned"] {
        request["guidance"] = json!(guidance);
        request.as_object_mut().unwrap().remove("request_sha256");
        request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
        let before = forwards.load(Ordering::Relaxed);
        let response = app(&engine)
            .oneshot(q1_http_request(&request))
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
        if let Some(output) = std::env::var_os("JOPE_Q1_TEST_RECEIPTS") {
            let output = PathBuf::from(output);
            std::fs::create_dir_all(&output).unwrap();
            std::fs::write(
                output.join(format!("request-{guidance}.json")),
                serde_json::to_vec_pretty(&request).unwrap(),
            )
            .unwrap();
            std::fs::write(output.join(format!("receipt-{guidance}.json")), &bytes).unwrap();
        }
        assert_eq!(receipt.identity.guidance, guidance);
        assert_eq!(receipt.charged, 24);
        assert_eq!(receipt.attempts.len(), 24);
        let completed = receipt
            .attempts
            .iter()
            .filter(|a| a.outcome == EvaluationAttemptOutcome::Completed)
            .count();
        assert!(completed > 0);
        let after = forwards.load(Ordering::Relaxed);
        if guidance == "learned" {
            assert!(after > before, "no actual neural guidance");
        } else {
            assert_eq!(after, before, "heuristic arm invoked neural guidance");
        }
        let replay = app(&engine)
            .oneshot(q1_http_request(&request))
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        assert_eq!(bytes, to_bytes(replay.into_body(), 1 << 20).await.unwrap());
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            after,
            "replay evaluated again"
        );
        println!(
            "{}",
            json!({"guidance":guidance,"charged":receipt.charged,
            "completed":completed,"actual_neural_forwards":after-before,
            "fixture_only":true,"sealed_payloads_read":0})
        );
    }
    request["pins"]["bundle_sha256"] = json!("e".repeat(64));
    request.as_object_mut().unwrap().remove("request_sha256");
    request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
    let response = app(&engine)
        .oneshot(q1_http_request(&request))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(body(response).await["code"], "INVALID_INPUT");
    assert!(engine
        .store()
        .get(&q1_command_key(request["request_sha256"].as_str().unwrap()))
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn q1_learned_without_admitted_model_refuses_before_reservation() {
    let (directory, engine, _) = fixture();
    let mut request = q1_request(directory.path(), engine.store());
    request["guidance"] = json!("learned");
    request.as_object_mut().unwrap().remove("request_sha256");
    request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
    let response = app(&engine)
        .oneshot(q1_http_request(&request))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body(response).await["code"], "CAPABILITY_UNAVAILABLE");
    assert!(engine
        .store()
        .get(&q1_command_key(request["request_sha256"].as_str().unwrap()))
        .unwrap()
        .is_none());
    assert!(!directory.path().join("q1/search").exists());
}

#[test]
fn q1_development_fit_scope_keeps_request_identity_validation() {
    let (directory, service) = q1_service_fixture();
    let mut request = q1_request(directory.path(), &service.store);
    request["dataset"]["holdout_group"] = json!("development_fit");
    request.as_object_mut().unwrap().remove("request_sha256");
    request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
    let parsed: Q1Request = serde_json::from_value(request.clone()).unwrap();
    validate_q1_request(&parsed, &request).unwrap();
    request["dataset"]["holdout_group"] = json!("unapproved_group");
    request.as_object_mut().unwrap().remove("request_sha256");
    request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
    let parsed: Q1Request = serde_json::from_value(request.clone()).unwrap();
    assert!(validate_q1_request(&parsed, &request).is_err());
}

#[tokio::test]
async fn q1_receipt_uses_real_fold_values_and_replays_identical_durable_bytes() {
    let (directory, service) = q1_service_fixture();
    let request_body = q1_request(directory.path(), &service.store);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let first_bytes = service
        .q1_evaluate_using(request_body.clone(), || {
            Ok(InlineQ1Native {
                calls: Arc::clone(&calls),
            })
        })
        .unwrap();
    let receipt: EvaluationReceipt = serde_json::from_slice(&first_bytes).unwrap();
    assert_eq!(receipt.charged, 24);
    assert_eq!(receipt.attempts.len(), 24);
    let completed: Vec<_> = receipt
        .attempts
        .iter()
        .filter(|attempt| attempt.outcome == EvaluationAttemptOutcome::Completed)
        .collect();
    assert!(!completed.is_empty());
    assert!(completed
        .iter()
        .all(|attempt| attempt.fold_scores.len() == 3
            && attempt.loss.is_some()
            && attempt.program_sha256.is_some()
            && attempt.failure.is_none()));
    let programs: std::collections::BTreeSet<_> = completed
        .iter()
        .map(|attempt| attempt.program_sha256.as_ref().unwrap())
        .collect();
    assert!(
        programs.len() > 1,
        "Q1 must search rather than repeat one ridge DSL"
    );
    assert!(receipt
        .attempts
        .iter()
        .enumerate()
        .all(|(number, attempt)| attempt.attempt == number as u64
            && (attempt.outcome == EvaluationAttemptOutcome::Completed
                || (attempt.failure.is_some()
                    && attempt.fold_scores.is_empty()
                    && attempt.loss.is_none()))));

    let before = calls.load(std::sync::atomic::Ordering::Relaxed);
    let replay_bytes = service
        .q1_evaluate_using(request_body, || {
            Ok(InlineQ1Native {
                calls: Arc::clone(&calls),
            })
        })
        .unwrap();
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        before,
        "idempotent replay must not evaluate again"
    );
    assert_eq!(first_bytes, replay_bytes);

    let stored = service
        .store
        .get(&q1_command_key(&receipt.request_sha256))
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.data["receipt_bytes"],
        String::from_utf8(first_bytes.to_vec()).unwrap()
    );
    let run: jope_api::ids::RunId = stored.data["run_id"].as_str().unwrap().parse().unwrap();
    let receipt_row = service
        .store
        .get(&RecordKey::Evaluation {
            run: run.clone(),
            number: 0,
        })
        .unwrap()
        .unwrap();
    assert_eq!(receipt_row.data["request_sha256"], receipt.request_sha256);
    for attempt in 0..24 {
        assert!(service
            .store
            .get(&RecordKey::Evaluation {
                run: run.clone(),
                number: attempt + 1
            })
            .unwrap()
            .is_some());
    }
}

#[test]
fn q1_threefold_binding_is_refused_before_executor_or_reservation() {
    for case in 0..9 {
        let (directory, service) = q1_service_fixture();
        let mut raw = q1_request(directory.path(), &service.store);
        let path = directory
            .path()
            .join(format!("q1/protocols/{}.json", "a".repeat(64)));
        let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut binding: Q1DatasetBinding =
            serde_json::from_value(manifest["datasets"]["0123456789abcdef"].clone()).unwrap();
        match case {
            0 => {
                binding.folds.pop();
            }
            1 => binding.folds[2].repeat = 1,
            2 => binding.folds[2] = binding.folds[0].clone(),
            3 => binding.folds[2].validation = vec![4, 6],
            4 => binding.folds[2].train = vec![0, 0, 2, 3],
            5 => binding.folds[2].validation = vec![4, 4],
            6 => {
                binding.folds[2].train.pop();
            }
            7 => {
                binding.folds[2].validation.pop();
            }
            _ => {
                // Every local split is complete, but rows 6 and 7 never
                // occur in validation anywhere in the three-fold recipe.
                binding.train_rows = 8;
                for fold in &mut binding.folds {
                    fold.train.extend([6, 7]);
                }
            }
        }
        assert!(validate_q1_folds(&binding).is_err(), "case {case}");
        binding.cut_sha256 = q1_cut_sha256(&binding).unwrap();
        let mut value = serde_json::to_value(&binding).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("dataset_binding_sha256");
        binding.dataset_binding_sha256 = hash(&serde_json::to_vec(&value).unwrap());
        raw["dataset"]["cut_sha256"] = json!(binding.cut_sha256);
        raw.as_object_mut().unwrap().remove("request_sha256");
        let sha = canonical_sha256(&raw).unwrap();
        raw["request_sha256"] = json!(sha);
        manifest["datasets"]["0123456789abcdef"] = serde_json::to_value(&binding).unwrap();
        std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(
            service
                .q1_evaluate_using(raw, || -> anyhow::Result<InlineQ1Native> {
                    panic!("invalid fold recipe must not prepare an executor")
                })
                .is_err(),
            "case {case}"
        );
        assert!(
            service.store.get(&q1_command_key(&sha)).unwrap().is_none(),
            "case {case}"
        );
    }
}

#[test]
fn learned_q1_plan_and_folds_are_derived_from_the_pinned_manifest() {
    let (directory, engine, _) = fixture();
    let raw = q1_request(directory.path(), engine.store());
    let request: Q1Request = serde_json::from_value(raw.clone()).unwrap();
    let manifest = load_q1_manifest(
        directory.path(),
        request.pins.get("protocol_sha256").unwrap(),
    )
    .unwrap();
    let binding = manifest.datasets.get(&request.dataset.id).unwrap();

    let run_spec =
        q1_run_spec(&request_with_guidance(request.clone(), "learned"), binding).unwrap();
    assert_eq!(run_spec.guidance, Guidance::Learned);
    assert_eq!(run_spec.landmarks, LandmarkMode::Full);
    assert_eq!(run_spec.device, "cpu");
    validate_q1_folds(binding).unwrap();

    let fold_spec = q1_fold_spec(binding, &"a".repeat(64), true);
    assert_eq!(fold_spec["version"], 2);
    assert_eq!(fold_spec["target"], binding.target);
    assert_eq!(fold_spec["features"], json!(binding.feature_columns));
    assert_eq!(fold_spec["folds"], json!(binding.folds));
    assert!(fold_spec.get("refit_output").is_none());

    let mut caller_controlled = raw;
    caller_controlled["target"] = json!("attacker_target");
    let mut unsigned = caller_controlled.clone();
    unsigned.as_object_mut().unwrap().remove("request_sha256");
    caller_controlled["request_sha256"] = json!(canonical_sha256(&unsigned).unwrap());
    assert!(serde_json::from_value::<Q1Request>(caller_controlled).is_err());
}

fn request_with_guidance(mut request: Q1Request, guidance: &str) -> Q1Request {
    request.guidance = guidance.into();
    request
}

#[tokio::test]
async fn learned_q1_requires_a_verified_admitted_ranker() {
    let (directory, engine, _) = fixture();
    let mut request_body = q1_request(directory.path(), engine.store());
    request_body["guidance"] = json!("learned");
    request_body
        .as_object_mut()
        .unwrap()
        .remove("request_sha256");
    let request_sha = canonical_sha256(&request_body).unwrap();
    request_body["request_sha256"] = json!(request_sha);

    let denied = app(&engine)
        .oneshot(q1_http_request(&request_body))
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
    assert!(engine
        .store()
        .get(&q1_command_key(
            request_body["request_sha256"].as_str().unwrap()
        ))
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn actual_store_bound_plan_retries_and_conflicts_without_starting_a_run() {
    let (_directory, engine, spec) = fixture();
    let app = app(&engine);
    let first = app
        .clone()
        .oneshot(request(&spec, "plan-key"))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let first = body(first).await;
    let plan: Plan = serde_json::from_value(first.clone()).unwrap();
    assert_eq!(plan.rows.total.0, 96);
    assert_eq!(plan.rows.fit.0, 76);
    assert_eq!(plan.rows.holdout.0, 20);
    assert_eq!(plan.placement.evaluation, "cpu");
    assert_eq!(plan.placement.workers, 1);
    assert_eq!(
        body(
            app.clone()
                .oneshot(request(&spec, "plan-key"))
                .await
                .unwrap()
        )
        .await,
        first
    );
    let mut changed = spec.clone();
    changed.seed.0 += 1;
    let conflict = app
        .clone()
        .oneshot(request(&changed, "plan-key"))
        .await
        .unwrap();
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    assert_eq!(body(conflict).await["code"], "IDEMPOTENCY_CONFLICT");
    let mut stale = spec.clone();
    stale.plan_digest = Some("0".repeat(64));
    assert_eq!(
        body(app.oneshot(request(&stale, "stale-plan")).await.unwrap()).await["code"],
        "PLAN_CHANGED"
    );
    let list = Request::builder()
        .uri("/v1/runs?limit=1")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .body(Body::empty())
        .unwrap();
    let listed = app(&engine).oneshot(list).await.unwrap();
    assert_eq!(listed.status(), StatusCode::OK);
    assert_eq!(body(listed).await, json!({"runs":[],"next_after":null}));
    for raw in [
        r#"{"action":"continue","action":"continue"}"#,
        r#"{"action":"continue","unexpected":true}"#,
        r#"{"action":"continue","expected_revision":"-1"}"#,
    ] {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/runs/run_01ARZ3NDEKTSV4RRFFQ69G5FAV/control")
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .header(header::CONTENT_TYPE, "application/json")
            .header("idempotency-key", "bad-continue")
            .body(Body::from(raw))
            .unwrap();
        let reply = app(&engine).oneshot(request).await.unwrap();
        assert_eq!(reply.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body(reply).await["code"], "INVALID_INPUT");
    }
    assert!(!_directory.path().join("worker-scratch").exists());
}
