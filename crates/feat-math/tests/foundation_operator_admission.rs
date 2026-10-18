use feat_math::fm_oof::{
    fit_oof, Admission, AdmissionFailure, DistillationReceipt, FoundationError, FoundationOp,
    LicenseAdmission, OofFold,
};

struct FakeOp {
    learned_mean: f64,
}

impl FoundationOp for FakeOp {
    fn transform(&self, rows: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FoundationError> {
        Ok(rows
            .iter()
            .map(|row| vec![row[0] + self.learned_mean])
            .collect())
    }
}

#[test]
fn sealed_and_context_rows_cannot_enter_oof_fit_or_validation() {
    let rows = vec![vec![1.], vec![2.], vec![3.], vec![900.], vec![800.]];
    let labels = vec![10., 20., 30., 9000., 8000.];
    let fold = OofFold {
        train_rows: vec![0, 1],
        validation_rows: vec![2],
        context_rows: vec![3],
        sealed_rows: vec![4],
    };
    let result = fit_oof(&rows, &labels, &fold, 3, 1, |x, y| {
        assert_eq!(x, &[vec![1.], vec![2.]]);
        assert_eq!(y, &[10., 20.]);
        Ok(FakeOp {
            learned_mean: y.iter().sum::<f64>() / y.len() as f64,
        })
    })
    .unwrap();
    assert_eq!(result.validation_rows, vec![2]);
    assert_eq!(result.features, vec![vec![18.]]);
    assert_eq!(result.inference_units, 1);
    assert_eq!(result.model_load_units, 1);
}

#[test]
fn unavailable_license_or_runtime_stays_research_disabled() {
    let admission = LicenseAdmission {
        license_id: String::new(),
        license_verified: false,
        expected_model_sha256: "abc".into(),
        observed_model_sha256: "abc".into(),
        product_eligible: true,
        runtime_available: false,
        required_compute_units: 1,
        available_compute_units: 1,
    };
    assert_eq!(
        admission.evaluate(),
        Admission::ResearchDisabled(vec![
            AdmissionFailure::MissingLicense,
            AdmissionFailure::RuntimeUnavailable,
        ])
    );
}

#[test]
fn teacher_and_distilled_predictions_and_artifacts_remain_distinct() {
    let valid = DistillationReceipt {
        teacher_artifact_sha256: "teacher-artifact".into(),
        distilled_artifact_sha256: "distilled-artifact".into(),
        teacher_prediction_sha256: "teacher-predictions".into(),
        distilled_prediction_sha256: "distilled-predictions".into(),
        max_abs_error: 0.01,
        max_abs_error_limit: 0.02,
        parity_verified: true,
        quality_verified: true,
    };
    assert!(valid.portable_candidate_eligible());
    let mismatch = DistillationReceipt {
        max_abs_error: 0.2,
        ..valid
    };
    assert!(!mismatch.portable_candidate_eligible());
}

#[test]
fn sealed_row_overlap_is_rejected() {
    let rows = vec![vec![1.], vec![2.]];
    let labels = vec![1., 2.];
    let fold = OofFold {
        train_rows: vec![0],
        validation_rows: vec![1],
        context_rows: vec![],
        sealed_rows: vec![1],
    };
    let error = fit_oof(&rows, &labels, &fold, 3, 1, |_, _| {
        Ok(FakeOp { learned_mean: 0. })
    })
    .unwrap_err();
    assert_eq!(error, FoundationError::OverlappingRows(1));
}

#[test]
fn inference_and_model_load_are_both_charged_to_the_budget() {
    let rows = vec![vec![1.], vec![2.]];
    let labels = vec![1., 2.];
    let fold = OofFold {
        train_rows: vec![0],
        validation_rows: vec![1],
        context_rows: vec![],
        sealed_rows: vec![],
    };
    let error = fit_oof(&rows, &labels, &fold, 1, 1, |_, _| {
        Ok(FakeOp { learned_mean: 0. })
    })
    .unwrap_err();
    assert_eq!(error, FoundationError::InferenceBudgetExceeded);
}

#[test]
fn equal_prediction_identities_do_not_prove_distillation_parity() {
    let receipt = DistillationReceipt {
        teacher_artifact_sha256: "teacher-artifact".into(),
        distilled_artifact_sha256: "distilled-artifact".into(),
        teacher_prediction_sha256: "same-predictions".into(),
        distilled_prediction_sha256: "same-predictions".into(),
        max_abs_error: 0.0,
        max_abs_error_limit: 0.02,
        parity_verified: true,
        quality_verified: true,
    };
    assert!(!receipt.portable_candidate_eligible());
}
