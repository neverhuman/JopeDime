//! W3-T04: admission metrics are recomputed from the exact retained predictions and
//! targets. The test evaluator below is written independently of `score_predictions`,
//! `baseline` and `feat_math`; fixtures are synthetic, not a live fit.
use super::provenance::{
    assert_positive_control, assert_refused, assert_refused_record, authority,
};
use super::*;

fn stored<T: for<'de> Deserialize<'de>>(f: &Fixture, sha: &str) -> T {
    blob(&f.state, sha).unwrap()
}

/// Re-store one fold's evidence chain after `mutate`, keeping every digest link coherent.
fn rebind(
    f: &Fixture,
    evidence: &OofEvidence,
    fold: usize,
    mutate: impl FnOnce(&mut FoldPredictions, &mut EvaluationResult, &mut WorkerOutcome),
) -> OofEvidence {
    let mut prediction: FoldPredictions = stored(f, &evidence.fold_predictions_sha256[fold]);
    let mut result: EvaluationResult = stored(f, &prediction.execution_receipt_sha256);
    let mut worker: WorkerOutcome = stored(f, &result.worker_receipt_sha256);
    mutate(&mut prediction, &mut result, &mut worker);
    result.worker_receipt_sha256 = f.state.put_blob(&canonical_json(&worker).unwrap()).unwrap();
    prediction.execution_receipt_sha256 =
        f.state.put_blob(&canonical_json(&result).unwrap()).unwrap();
    let mut evidence = evidence.clone();
    evidence.fold_predictions_sha256[fold] = f
        .state
        .put_blob(&canonical_json(&prediction).unwrap())
        .unwrap();
    evidence
}

fn values(f: &Fixture, fold: usize) -> Vec<f64> {
    let prediction: FoldPredictions = stored(f, &f.evidence.fold_predictions_sha256[fold]);
    match stored(f, &prediction.predictions_sha256) {
        Predictions::Values(values) => values,
        _ => panic!("regression fixture"),
    }
}

fn independent_rmse(pairs: &[(f64, f64)]) -> f64 {
    let mut squared = 0.0;
    for (target, prediction) in pairs {
        squared += (target - prediction) * (target - prediction);
    }
    (squared / pairs.len() as f64).sqrt()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

fn uneven() -> Fixture {
    let folds = [
        vec!["-1.5", "2.0"],
        vec!["0.25", "3.5", "-2.75", "1.0"],
        vec!["4.0", "-0.5", "1.75"],
        vec!["-3.0", "2.5", "0.0", "1.25", "-1.0"],
        vec!["2.0", "-2.0"],
    ];
    // Four folds with heterogeneous small residuals; fold 4 is worse than its baseline.
    let predict = |fold: u8, hold: &[&TargetRow], baseline: Predictions| {
        let Predictions::Values(baseline) = baseline else {
            unreachable!()
        };
        Predictions::Values(if fold < 4 {
            hold.iter()
                .enumerate()
                .map(|(index, row)| row.target.parse::<f64>().unwrap() + 0.03 * (index + 1) as f64)
                .collect()
        } else {
            baseline.iter().map(|value| value + 0.75).collect()
        })
    };
    fixture_spec(
        Task::Regression,
        Some(context_plan()),
        Some(
            folds
                .iter()
                .map(|fold| fold.iter().map(|t| t.to_string()).collect())
                .collect(),
        ),
        &predict,
    )
}

#[test]
fn independent_evaluator_agrees_with_mean_fold_admission_on_uneven_folds() {
    let f = uneven();
    let q = qualify(&f.state, &f.record, &f.evidence).unwrap();
    let mut folds = Vec::new();
    for fold in 0..5u8 {
        let targets = |held: bool| {
            f.rows
                .iter()
                .filter(|row| (row.fold == fold) == held)
                .map(|row| row.target.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let (held, training) = (targets(true), targets(false));
        let mean = training.iter().sum::<f64>() / training.len() as f64;
        let candidate = values(&f, fold as usize);
        assert_eq!(candidate.len(), held.len());
        let c = independent_rmse(&held.iter().copied().zip(candidate).collect::<Vec<_>>());
        let b = independent_rmse(&held.iter().map(|t| (*t, mean)).collect::<Vec<_>>());
        folds.push((c, b));
    }
    assert_eq!(q.fold_losses.len(), 5);
    for ((c, b), (qc, qb)) in folds.iter().zip(&q.fold_losses) {
        assert!(close(*c, *qc) && close(*b, *qb), "{c} {b} vs {qc} {qb}");
    }
    let improved = folds
        .iter()
        .filter(|(c, b)| b - c > 1e-12 + 1e-6 * b.abs())
        .count();
    assert_eq!((improved, q.improved_folds), (4, 4));
    let mean_c = folds.iter().map(|(c, _)| c).sum::<f64>() / 5.0;
    let mean_b = folds.iter().map(|(_, b)| b).sum::<f64>() / 5.0;
    assert!(close(mean_c, q.candidate_loss) && close(mean_b, q.baseline_loss));
    assert!(close(mean_c, f.record.score.value));
    // Four wins and one severe loss satisfy the unchanged current policy.
    assert!(q.guard.admits(&q) && folds[4].0 > folds[4].1);
}

#[test]
fn pooled_rmse_r2_and_tail_are_separately_named_diagnostics() {
    let f = uneven();
    let q = qualify(&f.state, &f.record, &f.evidence).unwrap();
    let mut pooled = Vec::new();
    let mut baseline = Vec::new();
    for fold in 0..5u8 {
        let held = f.rows.iter().filter(|row| row.fold == fold);
        let training = f
            .rows
            .iter()
            .filter(|row| row.fold != fold)
            .collect::<Vec<_>>();
        let mean = training
            .iter()
            .map(|row| row.target.parse::<f64>().unwrap())
            .sum::<f64>()
            / training.len() as f64;
        for (row, value) in held.zip(values(&f, fold as usize)) {
            let target = row.target.parse::<f64>().unwrap();
            pooled.push((target, value));
            baseline.push((target, mean));
        }
    }
    let target_mean = pooled.iter().map(|(t, _)| t).sum::<f64>() / pooled.len() as f64;
    let sst = pooled
        .iter()
        .map(|(t, _)| (t - target_mean) * (t - target_mean))
        .sum::<f64>();
    let sse = pooled.iter().map(|(t, p)| (t - p) * (t - p)).sum::<f64>();
    let d = serde_json::to_value(&q).unwrap()["diagnostics"].clone();
    let number = |key: &str| d[key].as_f64().unwrap_or_else(|| panic!("{key}: {d}"));
    assert_eq!(d["schema"], "jope.teaching-diagnostics/v1");
    assert_eq!(d["fold_rows"], json!([2, 4, 3, 5, 2]));
    assert!(close(number("pooled_rmse"), independent_rmse(&pooled)));
    assert!(close(
        number("pooled_baseline_rmse"),
        independent_rmse(&baseline)
    ));
    assert!(close(number("pooled_r2"), 1.0 - sse / sst));
    // Pooled RMSE is not the admission aggregate and differs from it here.
    assert!((number("pooled_rmse") - q.candidate_loss).abs() > 1e-3);
    let deltas: Vec<f64> = q.fold_losses.iter().map(|(c, b)| c - b).collect();
    assert_eq!(d["adverse_folds"], 1);
    assert!(close(
        number("worst_fold_loss_delta"),
        deltas.iter().copied().fold(f64::MIN, f64::max)
    ));
    assert_eq!(d["fold_loss_deltas"], json!(deltas));
    // The stored qualification revalidates with its diagnostics unchanged.
    assert_eq!(revalidate_qualification(&f.state, &f.record).unwrap(), q);
    let classification = fixture(Task::Binary, 5);
    let q = qualify(
        &classification.state,
        &classification.record,
        &classification.evidence,
    )
    .unwrap();
    let d = serde_json::to_value(&q).unwrap()["diagnostics"].clone();
    assert_eq!(d["adverse_folds"], 0);
    assert!(d["pooled_rmse"].is_null() && d["pooled_r2"].is_null());
}

#[test]
fn forged_units_cannot_relabel_a_measured_score() {
    let f = fixture(Task::Regression, 5);
    for units in ["normalized_rmse", "dimensionless"] {
        let mut record = f.record.clone();
        record.score.units = units.into();
        assert_refused_record(&f, &record, &f.evidence, "score units", units);
    }
    assert_positive_control(&f);
    // The retained qualification cannot carry a relabelled record into the HWM.
    let mut relabelled = f.record.clone();
    relabelled.score.units = "normalized_rmse".into();
    let before = authority(&f.state);
    let report = hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: relabelled,
            blob: b"synthetic candidate".to_vec(),
            chain_from_source: false,
        }],
        json!({"kind":"w3-t04"}),
    )
    .unwrap();
    assert_eq!(report.rejects[0].reason, "teaching-ineligible");
    assert_eq!(before, authority(&f.state));
}

#[test]
fn prediction_evidence_substitutions_fail_before_authority() {
    let f = fixture(Task::Regression, 5);
    let e = &f.evidence;
    let provenance = "prediction provenance differs from pre-optimization freeze";
    let mut swapped = e.clone();
    swapped.fold_predictions_sha256.swap(0, 1);
    assert_refused(&f, &swapped, provenance, "fold order swapped");
    let mut duplicated = e.clone();
    duplicated.fold_predictions_sha256[1] = duplicated.fold_predictions_sha256[0].clone();
    assert_refused(&f, &duplicated, provenance, "duplicated fold");
    let mut missing = e.clone();
    missing.fold_predictions_sha256.pop();
    assert_refused(&f, &missing, "complete five-fold", "missing fold");
    let mut extra = e.clone();
    extra
        .fold_predictions_sha256
        .push(extra.fold_predictions_sha256[0].clone());
    assert_refused(&f, &extra, "complete five-fold", "sixth fold");
    let rows = rebind(&f, e, 0, |p, _, _| p.row_ids[1] = p.row_ids[0].clone());
    assert_refused(
        &f,
        &rows,
        "OOF row identity or order changed",
        "duplicated row",
    );
    let other: FoldPredictions = stored(&f, &e.fold_predictions_sha256[0]);
    let fitted = other.fitted_artifact_sha256.clone();
    let swapped = rebind(&f, e, 1, |p, _, _| {
        p.fitted_artifact_sha256 = fitted.clone()
    });
    assert_refused(
        &f,
        &swapped,
        "names different artifacts",
        "fitted state of another fold",
    );
    let swapped = rebind(&f, e, 1, |p, r, _| {
        p.fitted_artifact_sha256 = fitted.clone();
        r.fitted_state_sha256 = Some(fitted.clone());
    });
    assert_refused(
        &f,
        &swapped,
        "absent from the bound worker outputs",
        "coherently rebound fitted state of another fold",
    );
    let absent = sha256_hex(b"never retained predictions");
    let missing = rebind(&f, e, 2, |p, r, _| {
        p.predictions_sha256 = absent.clone();
        r.predictions_sha256 = Some(absent.clone());
    });
    assert_refused(
        &f,
        &missing,
        "prediction/target artifact is missing",
        "missing predictions",
    );
    let forged = rebind(&f, e, 3, |_, r, _| {
        r.primary = r.primary.map(|value| value + 0.1)
    });
    assert_refused(
        &f,
        &forged,
        "execution score differs from stored predictions",
        "forged fold score",
    );
    let mut claimed = f.record.clone();
    claimed.score.value -= 1e-9;
    assert_refused_record(
        &f,
        &claimed,
        e,
        "claimed HWM score differs",
        "forged aggregate score",
    );
    // JSON has no NaN: a nonfinite prediction reaches the receiver only as `null`.
    let path = {
        let result: EvaluationResult = stored(&f, &other.execution_receipt_sha256);
        let worker: WorkerOutcome = stored(&f, &result.worker_receipt_sha256);
        worker.outputs[0].path.clone()
    };
    let original = fs::read(&path).unwrap();
    let nonfinite = br#"{"kind":"values","data":[null,1.0]}"#;
    fs::write(&path, nonfinite).unwrap();
    let nulls = f.state.put_blob(nonfinite).unwrap();
    let invalid = rebind(&f, e, 0, |p, r, w| {
        p.predictions_sha256 = nulls.clone();
        r.predictions_sha256 = Some(nulls.clone());
        w.outputs[0].sha256 = nulls.clone();
    });
    assert_refused(&f, &invalid, "invalid type: null", "nonfinite prediction");
    fs::write(&path, original).unwrap();
    assert_positive_control(&f);
}
