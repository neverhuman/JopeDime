//! Dedicated CI qualification suite for 2-class binary categorical/boolean target conditioning,
//! class-stratified `FIT`/`HOLDOUT` splitting, `JLMB` landmark extraction, and `JMAT v1` encoding.

use jope_prep::{
    compute_binary_holdout_metrics, jmat_lite::split_and_encode_numeric_jmat_v1, BinaryTargetError,
    BinaryTargetMapping, FitLandmarkBundle, SplitPlan, FIT_LANDMARK_WIRE_BYTES,
};

#[test]
fn binary_target_mapping_handles_boolean_semantic_and_arbitrary_string_labels() {
    let pairs: &[(&[&str], &str, &str)] = &[
        (&["yes", "no", "yes", "no"], "no", "yes"),
        (&["Yes", "yes", "NO", "no", "No"], "no", "yes"),
        (&["true", "false", "true"], "false", "true"),
        (&["True", "FALSE", "true", "False"], "False", "True"),
        (&["y", "n", "Y", "N"], "n", "y"),
        (&["t", "f", "T", "F"], "f", "t"),
        (&["yes", "yes", "true", "no", "no", "false"], "no", "yes"),
        (
            &["positive", "negative", "positive"],
            "negative",
            "positive",
        ),
        (&["+", "-", "+"], "-", "+"),
        (&["pass", "fail", "pass"], "fail", "pass"),
        (&["malignant", "benign", "benign"], "benign", "malignant"),
        (&["<=50K", ">50K", "<=50K"], "<=50K", ">50K"),
        (&["churn", "retain", "retain"], "retain", "churn"),
        (
            &["approved", "declined", "declined"],
            "declined",
            "approved",
        ),
        (&["dog", "cat", "cat"], "cat", "dog"),
        (
            &["Tier B, Secondary", "Tier A, Primary"],
            "Tier A, Primary",
            "Tier B, Secondary",
        ),
        (&["Alpha", "alpha", "Beta", "BETA"], "Alpha", "Beta"),
    ];

    for &(cells, expected_neg, expected_pos) in pairs {
        let mapping = BinaryTargetMapping::infer(cells)
            .unwrap_or_else(|e| panic!("failed to infer binary mapping for {cells:?}: {e}"));
        assert!(
            mapping.negative_label.eq_ignore_ascii_case(expected_neg),
            "unexpected negative label '{}' vs '{expected_neg}' for {cells:?}",
            mapping.negative_label
        );
        assert!(
            mapping.positive_label.eq_ignore_ascii_case(expected_pos),
            "unexpected positive label '{}' vs '{expected_pos}' for {cells:?}",
            mapping.positive_label
        );
        for &cell in cells {
            let encoded = mapping
                .encode_label(cell)
                .unwrap_or_else(|| panic!("failed to encode '{cell}' under {mapping:?}"));
            assert!(encoded == 0.0 || encoded == 1.0);
        }
        assert_eq!(mapping.encode_label(expected_neg), Some(0.0));
        assert_eq!(mapping.encode_label(expected_pos), Some(1.0));
        let (p_hi, lbl_hi) = mapping.decode_prediction(1.35, 0.5);
        assert_eq!(p_hi, 1.0);
        assert!(lbl_hi.eq_ignore_ascii_case(expected_pos));
        let (p_lo, lbl_lo) = mapping.decode_prediction(-0.20, 0.5);
        assert_eq!(p_lo, 0.0);
        assert!(lbl_lo.eq_ignore_ascii_case(expected_neg));
    }
}

#[test]
fn binary_target_refuses_empty_constant_and_multiclass_inputs_with_clear_errors() {
    assert_eq!(
        BinaryTargetMapping::infer(&["", "   "]),
        Err(BinaryTargetError::Empty)
    );
    let const_err = BinaryTargetMapping::infer(&["yes", "YES", "true", "y"])
        .expect_err("all-positive boolean synonyms must fail closed as constant");
    assert!(const_err.to_string().contains("constant"));

    let single_str = BinaryTargetMapping::infer(&["retain", "retain", "retain"])
        .expect_err("1-class string target must fail closed");
    assert!(single_str.to_string().contains("only 1 distinct class"));

    let multi_err = BinaryTargetMapping::infer(&["setosa", "versicolor", "virginica", "setosa"])
        .expect_err("3-class target must fail closed cleanly");
    let msg = multi_err.to_string();
    assert!(msg.contains("3 distinct categorical classes"), "got: {msg}");
    assert!(msg.contains("multiclass"), "got: {msg}");
}

#[test]
fn stratified_binary_split_preserves_rare_minority_class_and_builds_jlmb_and_jmat() {
    // 60 rows with 5% minority positive class (57 "no" = 0, 3 "yes" = 1).
    let n = 60usize;
    let mut raw_labels = vec!["no"; n];
    raw_labels[5] = "yes";
    raw_labels[29] = "yes";
    raw_labels[53] = "yes";

    let mapping = BinaryTargetMapping::infer(&raw_labels).expect("infer 95:5 yes/no");
    let y_f64: Vec<f64> = raw_labels
        .iter()
        .map(|s| mapping.encode_label(s).expect("encode"))
        .collect();
    let y_u8: Vec<u8> = y_f64.iter().map(|&v| u8::from(v >= 0.5)).collect();

    for seed in 1..=50_u64 {
        let split = SplitPlan::stratified_binary(&y_u8, seed, 0.20)
            .expect("stratified_binary split must succeed");
        split.validate_for_table(n).expect("valid SplitPlan");
        assert_eq!(split.holdout_rows().len(), 12);
        assert_eq!(split.fit_rows().len(), 48);

        let fit_pos = split
            .fit_rows()
            .iter()
            .filter(|&&idx| y_u8[idx] == 1)
            .count();
        let hold_pos = split
            .holdout_rows()
            .iter()
            .filter(|&&idx| y_u8[idx] == 1)
            .count();
        assert!(
            fit_pos >= 1 && hold_pos >= 1,
            "seed {seed}: expected >=1 minority positive in both FIT ({fit_pos}) and HOLDOUT ({hold_pos})"
        );
        assert_eq!(fit_pos + hold_pos, 3);
    }

    // Verify full `FitLandmarkBundle` (`JLMB`) + `JMAT v1` encoding on binary `{0.0, 1.0}` target!
    let split = SplitPlan::stratified_binary(&y_u8, 20261117, 0.20).expect("split");
    let n_feats = 4usize;
    let mut x_f32 = Vec::with_capacity(n * n_feats);
    let mut y_f32 = Vec::with_capacity(n);
    let mut numeric_rows = Vec::with_capacity(n);
    for (r, &yv) in y_f64.iter().enumerate() {
        let signal = if yv >= 0.5 { 2.5 } else { -1.5 };
        let row = vec![
            signal + (r as f64) * 0.01,
            ((r % 7) as f64) * 0.3,
            if r % 2 == 0 { 1.0 } else { 0.0 },
            (r as f64).sin(),
            yv,
        ];
        for &fv in &row[..n_feats] {
            x_f32.push(fv as f32);
        }
        y_f32.push(yv as f32);
        numeric_rows.push(row);
    }

    let bundle = FitLandmarkBundle::extract_fit_only(&x_f32, n, n_feats, &y_f32, &split)
        .expect("extract JLMB on binary target");
    assert_eq!(bundle.to_wire_bytes().len(), FIT_LANDMARK_WIRE_BYTES);

    let headers = vec![
        "f1".to_owned(),
        "f2".to_owned(),
        "f3".to_owned(),
        "f4".to_owned(),
        "churn".to_owned(),
    ];
    let jmat = split_and_encode_numeric_jmat_v1(&headers, &numeric_rows, &split)
        .expect("encode JMAT v1 on binary target");
    assert!(!jmat.fit_jmat.is_empty());
    assert!(!jmat.holdout_jmat.is_empty());
}

#[test]
fn binary_holdout_metrics_compute_exact_brier_logloss_roc_auc_and_ties() {
    let y_true = [0.0, 0.0, 1.0, 1.0];
    let perfect = [0.05, 0.15, 0.85, 0.95];
    let m_perf = compute_binary_holdout_metrics(&y_true, &perfect).expect("metrics");
    assert_eq!(m_perf.roc_auc, Some(1.0));
    assert_eq!(m_perf.accuracy, 1.0);
    assert!(m_perf.brier < 0.02);
    assert!(m_perf.logloss < 0.15);

    let tied = [0.5, 0.5, 0.5, 0.5];
    let m_tied = compute_binary_holdout_metrics(&y_true, &tied).expect("tied metrics");
    assert_eq!(m_tied.roc_auc, Some(0.5));
    assert!((m_tied.brier - 0.25).abs() < 1e-12);
}
