use super::*;

#[test]
fn continuation_restores_native_backend_from_persisted_request_without_replanning() {
    let (_directory, engine, context, plan, fit) = continuation_backend_fixture();
    // Legacy native contexts predate the optional landmark evidence.
    let mut legacy = serde_json::to_value(context).unwrap();
    legacy.as_object_mut().unwrap().remove("landmarks");
    let context: ContinuationBackend = serde_json::from_value(legacy).unwrap();
    let original = plan.clone();
    let restored = context
        .restore(engine.store(), &plan, &[fit], None)
        .unwrap();
    assert!(!restored.learned());
    assert_eq!(restored.placement, context.placement);
    assert_eq!(restored.bundle_digest, original.bundle_digest);
    assert_eq!(plan, original);
}

#[test]
fn learned_continuation_without_admitted_capability_never_falls_back_to_native() {
    let (_directory, engine, mut context, plan, fit) = continuation_backend_fixture();
    context.spec.guidance = Guidance::Learned;
    context.spec.landmarks = LandmarkMode::Full;
    let error = context
        .restore(engine.store(), &plan, &[fit], None)
        .err()
        .unwrap();
    assert_eq!(error.code, ErrorCode::CapabilityUnavailable);
}

#[test]
fn continuation_refuses_changed_backend_identity_and_tampered_fit() {
    let (_directory, engine, mut context, mut plan, fit) = continuation_backend_fixture();
    context.placement.workers = 2;
    assert_eq!(
        context
            .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
            .err()
            .unwrap()
            .code,
        ErrorCode::CapabilityUnavailable
    );
    context.placement.workers = 1;
    let original = plan.clone();
    plan.bundle_digest = "b".repeat(64);
    assert_eq!(
        context
            .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
            .err()
            .unwrap()
            .code,
        ErrorCode::CapabilityUnavailable
    );
    plan = original;
    plan.matrix_path = plan.matrix_path.with_extension("changed");
    assert_eq!(
        context
            .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
            .err()
            .unwrap()
            .code,
        ErrorCode::CapabilityUnavailable
    );
    plan.matrix_path = engine.store().verified_blob_path(&fit).unwrap();
    std::fs::write(&plan.matrix_path, b"corrupted owned TRAIN fixture").unwrap();
    assert!(context
        .restore(engine.store(), &plan, &[fit], None)
        .is_err());
}

#[test]
fn full_landmark_continuation_rebuilds_only_original_fit_and_refuses_receipt_drift() {
    let (_directory, engine, mut context, plan, fit) = continuation_backend_fixture();
    context.spec.landmarks = LandmarkMode::Full;
    let bytes = engine.store().read_blob(&fit).unwrap();
    let mut backend = BackendPlan::resolve(&context.spec, None, 1).unwrap();
    backend.bind_fit(&bytes, &plan).unwrap();
    let measured = backend.full_fit_landmarks.clone().unwrap();
    assert_eq!(measured.fit_sha256, plan.matrix_sha256);
    assert_eq!(measured.fit_rows, 96);
    assert_eq!(measured.sampled_fit_rows.len(), 96);
    assert!(measured.sampled_fit_rows.iter().all(|&r| r < 96));
    assert!(!measured.features.contains(&measured.target));
    assert_eq!(measured.vector.len(), 256);
    assert_eq!(measured.columns.len(), 64 * 17);
    assert!(measured
        .vector
        .iter()
        .chain(&measured.columns)
        .all(|v| v.is_finite()));
    assert!(measured.vector.iter().any(|v| *v != 0.));
    context.landmarks = Some(measured);
    let mut context: ContinuationBackend =
        serde_json::from_value(serde_json::to_value(context).unwrap()).unwrap();
    let before = plan.clone();
    let restored = context
        .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
        .unwrap();
    assert_eq!(restored.full_fit_landmarks, context.landmarks);
    assert_eq!(plan, before);
    context.landmarks.as_mut().unwrap().vector[0] += 1.;
    assert_eq!(
        context
            .restore(engine.store(), &plan, &[fit], None)
            .err()
            .unwrap()
            .code,
        ErrorCode::CapabilityUnavailable
    );
    let mut wrong = plan.clone();
    wrong.matrix_sha256 = "0".repeat(64);
    assert!(crate::run::landmarks::FullFitLandmarks::extract(&bytes, &wrong).is_err());
    wrong = plan;
    wrong.features.push(wrong.target.clone());
    assert!(crate::run::landmarks::FullFitLandmarks::extract(&bytes, &wrong).is_err());
}
