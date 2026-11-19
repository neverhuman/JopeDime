use super::*;

#[tokio::test]
async fn run_device_errors_are_typed_before_dataset_lookup_or_worker_start() {
    let (directory, engine, mut spec) = fixture();
    spec.dataset = "ds_00112233445566778899aabb".parse().unwrap();
    let app = app(&engine);
    for dry_run in [false, true] {
        spec.dry_run = dry_run;
        for (device, status, code) in [
            (
                "cuda:0",
                StatusCode::SERVICE_UNAVAILABLE,
                "BACKEND_UNAVAILABLE",
            ),
            (
                "cuda:7",
                StatusCode::SERVICE_UNAVAILABLE,
                "BACKEND_UNAVAILABLE",
            ),
            ("cuda:", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
            ("cuda:-1", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
            ("cuda:4294967296", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
            ("gpu", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
        ] {
            spec.device = device.into();
            let denied = app
                .clone()
                .oneshot(request(&spec, "device-refusal"))
                .await
                .unwrap();
            assert_eq!(denied.status(), status, "{device} dry_run={dry_run}");
            let problem = body(denied).await;
            assert_eq!(problem["code"], code);
            assert!(problem.get("run_id").is_none());
        }
    }
    assert!(!directory.path().join("worker-scratch").exists());
    assert!(!directory.path().join("models").exists());
}

#[tokio::test]
async fn actual_run_route_preserves_auth_host_origin_and_explicit_capability_refusals() {
    let (_directory, engine, spec) = fixture();
    let app = app(&engine);
    for (header_name, value, status) in [
        ("authorization", "Bearer wrong", StatusCode::UNAUTHORIZED),
        ("host", "attacker.invalid:31234", StatusCode::BAD_REQUEST),
        ("origin", "https://example.invalid", StatusCode::BAD_REQUEST),
    ] {
        let mut req = request(&spec, "denied");
        req.headers_mut()
            .insert(header_name, axum::http::HeaderValue::from_static(value));
        let denied = app.clone().oneshot(req).await.unwrap();
        assert_eq!(denied.status(), status);
        assert_eq!(denied.headers()["jope-api-version"], "1.0");
    }
    let mut unsupported = spec.clone();
    unsupported.guidance = Guidance::Learned;
    let denied = app
        .clone()
        .oneshot(request(&unsupported, "learned"))
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
    let missing = Request::builder()
        .uri("/v1/runs/run_01JBD8Q6N4V2T5Y8K3M7P9R1S0/events")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .body(Body::empty())
        .unwrap();
    let missing = app.oneshot(missing).await.unwrap();
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    assert_eq!(body(missing).await["code"], "NOT_FOUND");
}
#[test]
fn requested_export_languages_filter_the_actual_native_seed_and_mutations() {
    let allowed =
        run_language_filter(&[Language::Rust, Language::Python, Language::Javascript]).unwrap();
    let space = CpuRunSpace {
        allowed: allowed.clone(),
        binary_roster: None,
        #[cfg(all(feature = "failpoints", debug_assertions))]
        pub06_values: None,
    };
    let seeds = space.seeds().unwrap();
    assert_eq!(seeds.len(), 1);
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(73);
    let proposals = space.propose(&seeds, 16, &mut rng).unwrap();
    assert!(!proposals.is_empty());
    for dsl in seeds.iter().chain(&proposals) {
        let genome = feat_math::GenomeIr::parse(dsl).unwrap();
        assert!(allowed.accepts_seed(&genome));
        assert!(feat_math::transpile::emit_javascript(
            &genome,
            &feat_math::transpile::EmitOptions::default()
        )
        .is_ok());
    }
    assert!(space.canonical("(sym affine \"c0\")").is_err());
    for languages in [vec![], vec![Language::Typescript]] {
        assert!(run_language_filter(&languages).is_err());
    }
}

#[tokio::test]
async fn javascript_python_and_rust_plan_uses_the_declared_capability_intersection() {
    let (directory, engine, mut spec) = fixture();
    spec.languages = vec![Language::Rust, Language::Python, Language::Javascript];
    let reply = app(&engine)
        .oneshot(request(&spec, "portable-plan"))
        .await
        .unwrap();
    assert_eq!(reply.status(), StatusCode::OK);
    let plan: Plan = serde_json::from_value(body(reply).await).unwrap();
    assert_eq!(plan.languages.requested, spec.languages);
    assert_eq!(
        plan.languages.excluded_operations,
        run_language_filter(&spec.languages).unwrap().excluded_ops()
    );
    assert!(!plan.languages.intersection_empty);
    assert!(!directory.path().join("worker-scratch").exists());
}
#[tokio::test]
async fn full_landmarks_plan_measures_real_fit_and_progressive_stays_unavailable() {
    let (directory, engine, mut spec) = fixture();
    spec.landmarks = LandmarkMode::Full;
    spec.guidance = Guidance::Auto;
    spec.languages = vec![Language::Rust, Language::Python];
    let app = app(&engine);
    let reply = app
        .clone()
        .oneshot(request(&spec, "full-plan"))
        .await
        .unwrap();
    assert_eq!(reply.status(), StatusCode::OK);
    let value = body(reply).await;
    let plan: Plan = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(plan.landmarks.mode, LandmarkMode::Full);
    assert_eq!(plan.landmarks.sample_rows, 76);
    assert_eq!(plan.guidance.value, Guidance::Heuristic);
    assert_eq!(plan.placement.dope, "none");
    assert_eq!(plan.languages.requested, spec.languages);
    assert_eq!(
        body(
            app.clone()
                .oneshot(request(&spec, "full-plan"))
                .await
                .unwrap()
        )
        .await,
        value
    );
    spec.landmarks = LandmarkMode::Progressive;
    let denied = app
        .oneshot(request(&spec, "progressive-plan"))
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
    assert!(!directory.path().join("worker-scratch").exists());
}

fn binary_csv_bytes() -> Vec<u8> {
    let mut csv = String::from("x_num,x_cat,label\n");
    for i in 0..48 {
        let positive = i % 2 == 1;
        let x_num = if i % 11 == 0 {
            String::new()
        } else if positive {
            format!("{:.3}", 2.0 + (i as f64) * 0.05)
        } else {
            format!("{:.3}", -2.0 + (i as f64) * 0.03)
        };
        let x_cat = match (positive, i % 3) {
            (true, 0 | 1) => "pro",
            (true, _) => "starter",
            (false, 0 | 1) => "free",
            (false, _) => "starter",
        };
        let label = if positive { "yes" } else { "no" };
        csv.push_str(&format!("{x_num},{x_cat},{label}\n"));
    }
    csv.into_bytes()
}

#[test]
fn binary_run_space_filters_methods_by_export_language() {
    let allowed = run_language_filter(&[Language::Rust, Language::Python]).unwrap();
    let py_rs_roster = binary_run_roster(&[Language::Rust, Language::Python]).unwrap();
    assert_eq!(py_rs_roster.len(), 5);
    let space = CpuRunSpace {
        allowed: allowed.clone(),
        binary_roster: Some(py_rs_roster),
        #[cfg(all(feature = "failpoints", debug_assertions))]
        pub06_values: None,
    };
    let seeds = space.seeds().unwrap();
    assert_eq!(seeds.len(), 1);
    assert!(seeds[0].contains("logit_ridge"));
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(73);
    let proposals = space.propose(&seeds, 5, &mut rng).unwrap();
    assert_eq!(proposals.len(), 5);
    for dsl in &proposals {
        assert!(space.canonical(dsl).is_ok());
    }

    let js_roster =
        binary_run_roster(&[Language::Rust, Language::Python, Language::Javascript]).unwrap();
    assert_eq!(js_roster.len(), 1);
    assert_eq!(js_roster[0].family, "logit_ridge");
}

#[tokio::test]
async fn live_system_info_and_binary_dry_run_plan_support_explicit_and_auto_tasks() {
    let (_directory, engine, mut spec) = fixture();
    let app = app(&engine);

    let sys_req = Request::builder()
        .method("GET")
        .uri("/v1/system")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .body(Body::empty())
        .unwrap();
    let sys_reply = app.clone().oneshot(sys_req).await.unwrap();
    assert_eq!(sys_reply.status(), StatusCode::OK);
    let sys_info: SystemInfo = serde_json::from_value(body(sys_reply).await).unwrap();
    assert_eq!(
        sys_info.capabilities.tasks,
        vec![Task::Regression, Task::Binary]
    );

    let dataset = register::register(
        engine.store(),
        binary_csv_bytes().as_slice(),
        &Default::default(),
    )
    .unwrap();
    spec.dataset = dataset.id;
    spec.target = "label".into();
    spec.task = Task::Binary;
    spec.objective = ObjectiveSpec::Name("auroc".into());
    spec.languages = vec![Language::Rust, Language::Python];
    spec.landmarks = LandmarkMode::Full;
    spec.guidance = Guidance::Auto;

    let explicit_reply = app
        .clone()
        .oneshot(request(&spec, "binary-explicit-plan"))
        .await
        .unwrap();
    assert_eq!(explicit_reply.status(), StatusCode::OK);
    let explicit_plan: Plan = serde_json::from_value(body(explicit_reply).await).unwrap();
    assert_eq!(explicit_plan.task.value, Task::Binary);
    assert!(!explicit_plan.task.inferred);
    assert_eq!(explicit_plan.objective.name, "roc_auc");
    assert_eq!(explicit_plan.objective.direction, Direction::Maximize);
    assert_eq!(explicit_plan.landmarks.sample_rows, 0);
    assert!(explicit_plan
        .warnings
        .iter()
        .any(|w| w.code == "BASELINE_MEAN"));

    spec.task = Task::Auto;
    spec.objective = ObjectiveSpec::Name("logloss".into());
    let auto_reply = app
        .clone()
        .oneshot(request(&spec, "binary-auto-plan"))
        .await
        .unwrap();
    assert_eq!(auto_reply.status(), StatusCode::OK);
    let auto_plan: Plan = serde_json::from_value(body(auto_reply).await).unwrap();
    assert_eq!(auto_plan.task.value, Task::Binary);
    assert!(auto_plan.task.inferred);
    assert_eq!(auto_plan.objective.name, "logloss");
    assert_eq!(auto_plan.objective.direction, Direction::Minimize);

    spec.task = Task::Multiclass;
    let denied = app
        .oneshot(request(&spec, "multiclass-denied"))
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
}

#[tokio::test]
async fn binary_discovery_finalization_prediction_and_export_roundtrip_end_to_end() {
    use crate::discover::r#loop::HeuristicAdmission;

    for objective_name in ["auroc", "logloss"] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let home = directory.path().canonicalize().unwrap();
        let lease = Arc::new(EngineLock::acquire(&home).unwrap());
        let store = Arc::new(Store::open(&home).unwrap());
        let config = Config::load(
            &home.join("config.toml"),
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
        let executable = std::env::current_exe().unwrap();
        let binary = format!(
            "{}:{}",
            env!("CARGO_PKG_VERSION"),
            hash(&std::fs::read(&executable).unwrap())
        );
        let runs = Arc::new(RunService {
            journal: Arc::new(EventLog::new(Arc::clone(&store))),
            store: Arc::clone(&store),
            config: config.clone(),
            home: home.clone(),
            executable,
            binary,
            activity: Arc::new(EngineActivity::default()),
            admission: Mutex::new(()),
            _lease: lease,
        });
        let csv_bytes = binary_csv_bytes();
        let dataset =
            register::register(&store, csv_bytes.as_slice(), &Default::default()).unwrap();

        let spec: RunSpec = serde_json::from_value(json!({
            "dataset": dataset.id,
            "target": "label",
            "task": "binary",
            "ignore_columns": [],
            "test_dataset": null,
            "split": {"kind": "random", "holdout_fraction": 0.25, "group_column": null, "time_column": null},
            "objective": objective_name,
            "effort": "fast",
            "budget": {"evaluations": "3", "wall_seconds": "60"},
            "landmarks": "none",
            "guidance": "heuristic",
            "languages": ["rust", "python"],
            "require_all_languages": true,
            "device": "cpu",
            "seed": "73",
            "dry_run": false,
            "plan_digest": null
        }))
        .unwrap();

        let mut prepared = runs.prepare(&spec).unwrap();
        let fit_blob = store.publish_blob(prepared.fit.as_slice()).unwrap();
        let holdout_blob = store.publish_blob(prepared.holdout.as_slice()).unwrap();
        prepared.search.matrix_path = store.verified_blob_path(&fit_blob).unwrap();
        let holdout_path = store.verified_blob_path(&holdout_blob).unwrap();

        let finalization = FinalizationRequest {
            run_id: prepared.search.run_id.clone(),
            fit: Partition {
                path: prepared.search.matrix_path.clone(),
                sha256: prepared.search.matrix_sha256.clone(),
                rows: prepared.fit_rows,
            },
            holdout: Partition {
                path: holdout_path,
                sha256: hash(&prepared.holdout),
                rows: prepared.holdout_rows,
            },
            test: None,
            refit: Refit::FitHoldout,
            model_root: home.join("models"),
            scratch_root: home.join("run-scratch"),
            max_input_bytes: 16 * 1024 * 1024,
            max_model_bytes: MODEL_LIMIT,
            comparators: None,
        };

        let identity = RequestIdentity::new(
            "local-engine-owner",
            &format!("bin-start-{objective_name}"),
            4,
            "/v1/runs",
            None,
            &serde_json::to_value(&spec).unwrap(),
        )
        .unwrap();
        let mut discovery = Discovery::start_command(
            Arc::clone(&store),
            prepared.search.clone(),
            Arc::clone(&runs.journal),
            identity,
            vec![fit_blob, holdout_blob],
            prepared.public.warnings,
        )
        .unwrap();

        let space = CpuRunSpace {
            allowed: run_language_filter(&spec.languages).unwrap(),
            binary_roster: Some(binary_run_roster(&spec.languages).unwrap()),
            #[cfg(all(feature = "failpoints", debug_assertions))]
            pub06_values: None,
        };
        let executor = InlineQ1Native {
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };
        while discovery.completed().is_none() {
            discovery
                .advance(&space, &HeuristicAdmission, &executor, 1, usize::MAX)
                .unwrap();
        }
        let finalizer = Finalizer::with_journal(Arc::clone(&store), Arc::clone(&runs.journal));
        let receipt = finalizer.finalize(&finalization, &executor).unwrap();
        assert!(receipt.scores.cv.unwrap().is_finite());
        assert!(receipt.scores.holdout.unwrap().is_finite());
        if objective_name == "auroc" {
            assert!(
                receipt.scores.holdout.unwrap() >= 0.8,
                "expected separable binary holdout AUROC >= 0.8, got {}",
                receipt.scores.holdout.unwrap()
            );
        }

        let auth = LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap();
        let router = crate::http::router::router_with_state(
            auth,
            crate::engine::DatasetState {
                store: Arc::clone(&store),
                config,
                home,
            },
            Some(Arc::clone(&runs)),
        );
        let pred_req = Request::builder()
            .method("POST")
            .uri("/v1/predictions")
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .header(header::CONTENT_TYPE, "application/json")
            .header("idempotency-key", format!("bin-predict-{objective_name}"))
            .body(Body::from(
                serde_json::to_vec(&PredictRequest {
                    model_artifact: receipt.model_artifact.clone(),
                    dataset: dataset.id.clone(),
                })
                .unwrap(),
            ))
            .unwrap();
        let pred_reply = router.clone().oneshot(pred_req).await.unwrap();
        assert_eq!(pred_reply.status(), StatusCode::OK);
        let csv_out = String::from_utf8(
            to_bytes(pred_reply.into_body(), 1 << 20)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        let mut lines = csv_out.lines();
        assert_eq!(lines.next(), Some("prediction,p_no,p_yes"));
        let pred_rows: Vec<&str> = lines.collect();
        assert_eq!(pred_rows.len(), 48);
        for line in pred_rows {
            let parts: Vec<&str> = line.split(',').collect();
            assert_eq!(parts.len(), 3);
            assert!(matches!(parts[0], "no" | "yes"));
            let p_no: f64 = parts[1].parse().unwrap();
            let p_yes: f64 = parts[2].parse().unwrap();
            assert!((0.0..=1.0).contains(&p_no) && (0.0..=1.0).contains(&p_yes));
            assert!((p_no + p_yes - 1.0).abs() <= 1e-12);
        }

        let export_req = Request::builder()
            .method("POST")
            .uri(format!("/v1/runs/{}/exports", receipt.run_id))
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .header(header::CONTENT_TYPE, "application/json")
            .header("idempotency-key", format!("bin-export-{objective_name}"))
            .body(Body::from(
                serde_json::to_vec(&ExportRequest {
                    algorithm_id: receipt.algorithm_id.clone(),
                    kind: ExportKind::Bundle,
                    languages: spec.languages.clone(),
                    require_all_languages: spec.require_all_languages,
                    refit: Refit::FitHoldout,
                })
                .unwrap(),
            ))
            .unwrap();
        let export_reply = router.clone().oneshot(export_req).await.unwrap();
        assert_eq!(export_reply.status(), StatusCode::OK, "{objective_name}");
        let manifest: ExportManifest = serde_json::from_value(body(export_reply).await).unwrap();
        assert_eq!(manifest.model_digest, receipt.model_digest);

        for lang in ["python", "rust"] {
            let alg_req = Request::builder()
                .method("GET")
                .uri(format!(
                    "/v1/runs/{}/algorithms/{}?language={lang}",
                    receipt.run_id, receipt.algorithm_id
                ))
                .header(header::HOST, "127.0.0.1:31234")
                .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
                .body(Body::empty())
                .unwrap();
            let alg_reply = router.clone().oneshot(alg_req).await.unwrap();
            assert_eq!(
                alg_reply.status(),
                StatusCode::OK,
                "{objective_name} {lang}"
            );
            let alg: Algorithm = serde_json::from_value(body(alg_reply).await).unwrap();
            assert_eq!(
                alg.summary,
                "Frozen completed CPU binary classification model"
            );
            assert!(!alg.source[0].source.is_empty());
        }
    }
}
