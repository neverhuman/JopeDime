use super::*;
use crate::{eval_registry::FitIdentity, prepare::Column};

/// This is an in-memory, uncommitted receipt fixture. It tests the private
/// scorer's contracts, never demonstrates authority custody or a real fit.
fn fixture() -> (Vec<u8>, Vec<u8>, QueryInputs) {
    let preprocessing = serde_json::to_vec(&Preprocessing {
        schema: "parallel.preprocessing.v1".into(),
        columns: vec![Column {
            kind: "numeric".into(),
            impute: 10.,
            mean: 10.,
            scale: 2.,
            categories: vec![],
        }],
        labels: vec![],
        target_mean: 100.,
        target_scale: 5.,
    })
    .unwrap();
    let feature_view = serde_json::to_vec(&json!({"protocol":"raw_training_fold_v1",
        "training_sha256":hash(b"synthetic training CSV"),"preprocessing_sha256":hash(&preprocessing)})).unwrap();
    let model = serde_json::to_vec(&feat_math::Model::Symbolic {
        expr: feat_math::math_expr::MathExpr::parse("col(0)").unwrap(),
        a: 1.,
        b: 0.,
    })
    .unwrap();
    let identity = FitIdentity {
        schema: "parallel.fit_identity.v2".into(),
        canonical_program: "(sym raw \"col(0)\")".into(),
        program_sha256: hash(b"synthetic program"),
        training_sha256: hash(b"synthetic ordered training"),
        preprocessing_sha256: hash(&preprocessing),
        feature_view_sha256: hash(&feature_view),
        executable_sha256: hash(b"synthetic original executable"),
        runtime_sha256: hash(b"synthetic original runtime"),
        fit_seed: 0,
        numerical_contract_sha256: hash(b"synthetic numerical contract"),
    };
    let mut receipt = CompletionReceipt {
        schema: "parallel.fit_completion.v2".into(),
        authority_instance: hash(b"synthetic authority"),
        fit_key: hash(&serde_json::to_vec(&identity).unwrap()),
        attempt_token: hash(b"synthetic attempt"),
        process: ProcessIdentity::current().unwrap(),
        task_id: "uncommitted-synthetic-scorer-fixture".into(),
        executable_sha256: identity.executable_sha256.clone(),
        program_sha256: identity.program_sha256.clone(),
        training_sha256: identity.training_sha256.clone(),
        fit_identity: identity,
        outcome: ExecutionOutcome::Success,
        observed_exit_code: Some(0),
        observed_signal: None,
        elapsed_seconds: Some(0.001),
        deadline_seconds: 1.,
        output_sha256: hash(b"synthetic output"),
        model_sha256: Some(hash(&model)),
        predictions_sha256: None,
        metrics_sha256: None,
        clean_completion_sha256: String::new(),
    };
    let mut value = serde_json::to_value(&receipt).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("clean_completion_sha256");
    receipt.clean_completion_sha256 = hash(&serde_json::to_vec(&value).unwrap());
    let completion = serde_json::to_vec(&receipt).unwrap();
    let inputs = QueryInputs {
        schema: "parallel.query_inputs.v1".into(),
        completion_sha256: hash(&completion),
        scope: "synthetic".into(),
        task_id: "query-synthetic".into(),
        rows: vec![vec!["12".into()], vec!["14".into()]],
        targets: vec![105., 110.],
        preprocessing,
        feature_view,
        contract: QueryContract::regression_rmse(),
        access_receipt_sha256: hash(b"private test-only query access"),
        deadline_monotonic_ns: monotonic_ns().unwrap() + 120_000_000_000,
    };
    (completion, model, inputs)
}
fn score(completion: &[u8], model: &[u8], inputs: &QueryInputs) -> Result<QueryArtifacts> {
    static RUNTIME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let runtime = if let Some(runtime) = RUNTIME.get() {
        runtime.clone()
    } else {
        let runtime = crate::eval_registry::current_runtime_digest(
            &crate::eval_registry::NumericalContract {
                protocol: "synthetic_query_f64_single_thread_v1".into(),
                precision: "f64".into(),
                omp_threads: 1,
                rayon_threads: 1,
                blas_threads: 1,
                mkl_threads: 1,
            },
        )?;
        let _ = RUNTIME.set(runtime.clone());
        runtime
    };
    score_verified(
        completion,
        model,
        inputs,
        &synthetic_access_for_test(inputs)?,
        &runtime,
        |model, rows, _| {
            Ok(Prediction {
                values: load_retained_model(model)?.predict(&rows.to_vec())?,
                process: ProcessIdentity::current()?,
                exit_code: Some(0),
            })
        },
    )
}
fn rebind_completion(
    completion: &[u8],
    inputs: &mut QueryInputs,
    model: &[u8],
    change: impl FnOnce(&mut CompletionReceipt),
) -> Vec<u8> {
    let mut receipt: CompletionReceipt = serde_json::from_slice(completion).unwrap();
    receipt.fit_identity.preprocessing_sha256 = hash(&inputs.preprocessing);
    receipt.fit_identity.feature_view_sha256 = hash(&inputs.feature_view);
    receipt.fit_key = hash(&serde_json::to_vec(&receipt.fit_identity).unwrap());
    receipt.model_sha256 = Some(hash(model));
    change(&mut receipt);
    let mut value = serde_json::to_value(&receipt).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("clean_completion_sha256");
    receipt.clean_completion_sha256 = hash(&serde_json::to_vec(&value).unwrap());
    let bytes = serde_json::to_vec(&receipt).unwrap();
    inputs.completion_sha256 = hash(&bytes);
    bytes
}

#[test]
fn retained_symbolic_model_scores_original_units_with_zero_rmse_without_fit() {
    let (completion, model, inputs) = fixture();
    let artifacts = score(&completion, &model, &inputs).unwrap();
    assert_eq!(
        serde_json::from_slice::<Vec<f64>>(artifacts.predictions_bytes()).unwrap(),
        vec![105., 110.]
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(artifacts.metrics_bytes()).unwrap()["rmse"],
        0.
    );
    let receipt = verify_artifacts(
        &completion,
        &model,
        &inputs,
        artifacts.receipt_bytes(),
        artifacts.predictions_bytes(),
        artifacts.metrics_bytes(),
    )
    .unwrap();
    assert_eq!(receipt.original_completion_sha256, hash(&completion));
    assert_eq!(receipt.request_sha256, hash(artifacts.request_bytes()));
    assert_eq!(
        receipt.producer_executable_sha256,
        hash(&std::fs::read("/proc/self/exe").unwrap())
    );
    assert_eq!(receipt.completion_kind, "service_observed_predictor_exit");
    assert_eq!(artifacts.receipt().query_key, receipt.query_key);
}

#[test]
fn fitting_scope_or_receipt_hash_does_not_grant_query_access() {
    let (_, _, inputs) = fixture();
    assert!(authorize(&inputs).is_err());
    let mut changed = inputs.clone();
    changed.scope = "regression_collection".into();
    assert!(authorize(&changed).is_err());
    assert!(synthetic_access_for_test(&changed).is_err());
}

#[test]
fn access_cannot_be_reused_for_changed_query_target_scope_or_deadline() {
    let (completion, model, inputs) = fixture();
    let access = synthetic_access_for_test(&inputs).unwrap();
    for mutate in [0, 1, 2, 3] {
        let mut changed = inputs.clone();
        match mutate {
            0 => changed.rows[0][0] = "13".into(),
            1 => changed.targets[0] = 106.,
            2 => changed.scope = "qualification".into(),
            _ => changed.deadline_monotonic_ns += 1,
        };
        assert!(score_verified(
            &completion,
            &model,
            &changed,
            &access,
            &hash(b"runtime"),
            |_, _, _| panic!("changed query must fail before numerical prediction")
        )
        .unwrap_err_text()
        .contains("access does not bind"));
    }
}

trait ErrorText {
    fn unwrap_err_text(self) -> String;
}
impl<T> ErrorText for Result<T> {
    fn unwrap_err_text(self) -> String {
        match self {
            Ok(_) => panic!("expected denial"),
            Err(error) => format!("{error:#}"),
        }
    }
}

#[test]
fn query_identity_separates_math_from_access_and_preserves_exact_targets() {
    let (completion, model, inputs) = fixture();
    let key = prepare_query(&completion, &model, &inputs)
        .unwrap()
        .query_key;
    let mut access_only = inputs.clone();
    access_only.task_id = "other-task".into();
    access_only.scope = "other-scope".into();
    assert_eq!(
        key,
        prepare_query(&completion, &model, &access_only)
            .unwrap()
            .query_key
    );
    assert_ne!(
        inputs.descriptor_sha256().unwrap(),
        access_only.descriptor_sha256().unwrap()
    );
    let mut row = inputs.clone();
    row.rows[0][0] = "12.0".into();
    assert_ne!(
        key,
        prepare_query(&completion, &model, &row).unwrap().query_key
    );
    let mut target = inputs.clone();
    target.targets[0] = 0.;
    let plus = prepare_query(&completion, &model, &target)
        .unwrap()
        .query_key;
    target.targets[0] = -0.;
    assert_ne!(
        plus,
        prepare_query(&completion, &model, &target)
            .unwrap()
            .query_key
    );
}

#[test]
fn model_completion_preprocessing_and_view_tampering_deny() {
    let (completion, model, inputs) = fixture();
    assert!(score(&completion, b"{}", &inputs).is_err());
    assert!(score(b"{}", &model, &inputs).is_err());
    let mut changed = inputs.clone();
    changed.preprocessing.push(b' ');
    assert!(score(&completion, &model, &changed).is_err());
    let mut changed = inputs.clone();
    changed.feature_view.push(b' ');
    assert!(score(&completion, &model, &changed).is_err());
    let mut changed = inputs.clone();
    changed.rows[0].push("extra".into());
    assert!(score(&completion, &model, &changed).is_err());
}

#[test]
fn unsupported_metric_units_and_compact_view_deny_before_predicting() {
    let (completion, model, inputs) = fixture();
    for mutate in [0, 1, 2, 3] {
        let mut changed = inputs.clone();
        match mutate {
            0 => changed.contract.units = "normalized_target".into(),
            1 => changed.contract.task = "binary".into(),
            2 => changed.contract.metric = "r2".into(),
            _ => changed.contract.outcome_policy = "ignore_missing".into(),
        };
        assert!(score(&completion, &model, &changed).is_err());
    }
    let mut changed = inputs.clone();
    let mut view: serde_json::Value = serde_json::from_slice(&changed.feature_view).unwrap();
    view["protocol"] = json!("compact_training_fold_v1");
    changed.feature_view = serde_json::to_vec(&view).unwrap();
    let changed_completion = rebind_completion(&completion, &mut changed, &model, |_| {});
    assert!(score(&changed_completion, &model, &changed)
        .unwrap_err_text()
        .contains("unsupported compact"));
}

#[test]
fn failed_original_fit_missing_reader_and_expired_deadline_create_no_quality() {
    let (completion, model, mut inputs) = fixture();
    let failed = rebind_completion(&completion, &mut inputs, &model, |r| {
        r.outcome = ExecutionOutcome::InfrastructureFailure;
        r.observed_exit_code = Some(9);
    });
    assert!(score(&failed, &model, &inputs)
        .unwrap_err_text()
        .contains("unsuccessful original"));
    let (completion, _, mut inputs) = fixture();
    let missing_reader = rebind_completion(
        &completion,
        &mut inputs,
        b"unsupported historical model",
        |_| {},
    );
    assert!(
        score(&missing_reader, b"unsupported historical model", &inputs)
            .unwrap_err_text()
            .contains("refitting is forbidden")
    );
    let (completion, model, mut inputs) = fixture();
    inputs.deadline_monotonic_ns = 1;
    assert!(score(&completion, &model, &inputs)
        .unwrap_err_text()
        .contains("deadline expired"));
}

#[test]
fn nonfinite_target_prediction_or_loss_never_produces_artifacts() {
    let (completion, model, mut inputs) = fixture();
    inputs.targets[0] = f64::NAN;
    assert!(score(&completion, &model, &inputs).is_err());
    assert!(metrics(&[1.], &[f64::INFINITY]).is_err());
    assert!(metrics(&[f64::MAX], &[-f64::MAX]).is_err());
    let (completion, model, mut inputs) = fixture();
    let mut pp: Preprocessing = serde_json::from_slice(&inputs.preprocessing).unwrap();
    pp.target_scale = f64::MAX;
    inputs.preprocessing = serde_json::to_vec(&pp).unwrap();
    let mut view: serde_json::Value = serde_json::from_slice(&inputs.feature_view).unwrap();
    view["preprocessing_sha256"] = json!(hash(&inputs.preprocessing));
    inputs.feature_view = serde_json::to_vec(&view).unwrap();
    let completion = rebind_completion(&completion, &mut inputs, &model, |_| {});
    assert!(score(&completion, &model, &inputs).is_err());
}

#[test]
fn output_or_receipt_tampering_fails_independent_verification_without_predicting() {
    let (completion, model, inputs) = fixture();
    let a = score(&completion, &model, &inputs).unwrap();
    assert!(verify_artifacts(
        &completion,
        &model,
        &inputs,
        a.receipt_bytes(),
        b"[1.0,2.0]",
        a.metrics_bytes()
    )
    .is_err());
    assert!(verify_artifacts(
        &completion,
        &model,
        &inputs,
        a.receipt_bytes(),
        a.predictions_bytes(),
        b"{}"
    )
    .is_err());
    let mut receipt: serde_json::Value = serde_json::from_slice(a.receipt_bytes()).unwrap();
    receipt["task_id"] = json!("restamped-task");
    assert!(verify_artifacts(
        &completion,
        &model,
        &inputs,
        &serde_json::to_vec(&receipt).unwrap(),
        a.predictions_bytes(),
        a.metrics_bytes()
    )
    .is_err());
    // A caller can calculate an inner digest. It still cannot change the
    // request identity or reviewed producer source accepted by the verifier.
    let mut forged: QueryReceipt = serde_json::from_slice(a.receipt_bytes()).unwrap();
    forged.scope = "qualification".into();
    forged.receipt_sha256 = receipt_digest(&forged).unwrap();
    assert!(verify_artifacts(
        &completion,
        &model,
        &inputs,
        &serde_json::to_vec(&forged).unwrap(),
        a.predictions_bytes(),
        a.metrics_bytes(),
    )
    .is_err());
    let mut forged: QueryReceipt = serde_json::from_slice(a.receipt_bytes()).unwrap();
    forged.producer_source_sha256 = hash(b"unreviewed scorer");
    forged.receipt_sha256 = receipt_digest(&forged).unwrap();
    assert!(verify_artifacts(
        &completion,
        &model,
        &inputs,
        &serde_json::to_vec(&forged).unwrap(),
        a.predictions_bytes(),
        a.metrics_bytes(),
    )
    .is_err());
}
