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
fn q1_preparation_failure_does_not_poison_admission_or_reserve() {
    let (directory, service) = q1_service_fixture();
    let request = q1_request(directory.path(), &service.store);
    let request_sha = request["request_sha256"].as_str().unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        service.q1_evaluate_using(request.clone(), || -> anyhow::Result<InlineQ1Native> {
            panic!("synthetic preparation panic")
        })
    }));
    assert!(panic.is_err());
    assert!(!service.admission.is_poisoned());
    assert!(service
        .store
        .get(&q1_command_key(request_sha))
        .unwrap()
        .is_none());

    let error = service.q1_evaluate_using(request.clone(), || -> anyhow::Result<InlineQ1Native> {
        assert!(
            service.admission.try_lock().is_ok(),
            "preparation held admission"
        );
        anyhow::bail!("synthetic preparation error")
    });
    assert!(error.is_err());
    assert!(!service.admission.is_poisoned());
    assert!(service
        .store
        .get(&q1_command_key(request_sha))
        .unwrap()
        .is_none());

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let bytes = service
        .q1_evaluate_using(request, || {
            Ok(InlineQ1Native {
                calls: Arc::clone(&calls),
            })
        })
        .unwrap();
    let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(receipt.charged, 24);
    assert!(!service.admission.is_poisoned());
}

#[test]
fn q1_admission_rechecks_busy_state_after_unlocked_preparation() {
    let (directory, service) = q1_service_fixture();
    let request = q1_request(directory.path(), &service.store);
    let request_sha = request["request_sha256"].as_str().unwrap();
    let busy = std::cell::RefCell::new(None);
    let error = service
        .q1_evaluate_using(request.clone(), || {
            assert!(service.admission.try_lock().is_ok());
            busy.replace(Some(service.activity.begin_run()));
            Ok(Q1AllFailed)
        })
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::CapabilityUnavailable);
    assert!(service
        .store
        .get(&q1_command_key(request_sha))
        .unwrap()
        .is_none());
    busy.replace(None);
    assert_eq!(service.activity.active_runs(), 0);
    let bytes = service
        .q1_evaluate_using(request, || Ok(Q1AllFailed))
        .unwrap();
    let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(receipt.charged, 24);
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
    let router = app(&engine);
    let first = router
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
            router
                .clone()
                .oneshot(request(&spec, "plan-key"))
                .await
                .unwrap()
        )
        .await,
        first
    );
    let mut changed = spec.clone();
    changed.seed.0 += 1;
    let conflict = router
        .clone()
        .oneshot(request(&changed, "plan-key"))
        .await
        .unwrap();
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    assert_eq!(body(conflict).await["code"], "IDEMPOTENCY_CONFLICT");
    let mut stale = spec.clone();
    stale.plan_digest = Some("0".repeat(64));
    assert_eq!(
        body(
            router
                .clone()
                .oneshot(request(&stale, "stale-plan"))
                .await
                .unwrap()
        )
        .await["code"],
        "PLAN_CHANGED"
    );
    let list = Request::builder()
        .uri("/v1/runs?limit=1")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .body(Body::empty())
        .unwrap();
    let listed = router.clone().oneshot(list).await.unwrap();
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
        let reply = router.clone().oneshot(request).await.unwrap();
        assert_eq!(reply.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body(reply).await["code"], "INVALID_INPUT");
    }
    assert!(!_directory.path().join("worker-scratch").exists());
}
