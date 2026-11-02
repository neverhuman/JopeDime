use super::*;

fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

#[test]
fn hold_encoder_input_masks_targets_but_fit_input_does_not() {
    let first = RegressionTable {
        features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        targets: vec![10.0, 20.0],
    };
    let mut changed = first.clone();
    changed.targets = vec![-1000.0, 5000.0];
    assert_eq!(
        csv_for_rows(&first, &[0, 1], true),
        csv_for_rows(&changed, &[0, 1], true)
    );
    assert_ne!(
        csv_for_rows(&first, &[0, 1], false),
        csv_for_rows(&changed, &[0, 1], false)
    );
}

#[test]
fn excluded_rows_cannot_change_fit_encoder_input() {
    let first = RegressionTable {
        features: vec![vec![1.0], vec![2.0], vec![3.0]],
        targets: vec![10.0, 20.0, 30.0],
    };
    let mut changed = first.clone();
    changed.features[2][0] = 99_999.0;
    changed.targets[2] = -99_999.0;
    assert_eq!(
        csv_for_rows(&first, &[0, 1], false),
        csv_for_rows(&changed, &[0, 1], false)
    );
    assert_ne!(
        csv_for_rows(&first, &[2], true),
        csv_for_rows(&changed, &[2], true)
    );
}

#[test]
fn unavailable_runtime_materializes_a_valid_raw_decision() {
    let started = Instant::now();
    let decision = raw_fallback_decision(
        "opaque",
        &digest("training"),
        &digest("predictor"),
        &digest("encoder"),
        &digest("cv5"),
        &digest("cv7"),
        "pinned CUDA GPU unavailable",
        started,
    );
    decision.validate().unwrap();
    assert_eq!(decision.selected_view, FeatureViewKindV1::Raw);
    assert!(decision.wall_clock_milliseconds <= WALL_CLOCK_BUDGET.as_millis() as u64);
}
