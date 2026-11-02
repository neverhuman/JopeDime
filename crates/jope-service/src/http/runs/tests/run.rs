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
