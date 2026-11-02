use super::*;

#[test]
fn shared_ordered_boundary_fixture_replays_every_order_after_each_snapshot() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/finite_difference.json"
    ))
    .unwrap();
    let decode = |v: &serde_json::Value| {
        f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
    };
    let rows = fixture["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| OrderedRow {
            group: GroupKey::categorical(r["group"].as_str().unwrap()).unwrap(),
            timestamp: decode(&r["time_bits"]),
            values: r["values_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(decode)
                .collect(),
        })
        .collect::<Vec<_>>();
    for order in 1..=4 {
        let config = DifferenceConfig::new(2, 59, order, decode(&fixture["step_bits"])).unwrap();
        let fold = FoldBoundary::new("boundary-fixture", 4, rows.len()).unwrap();
        let mut original =
            DifferenceState::new(config, 2, TaskMode::Forecasting, &input(), fold.clone()).unwrap();
        for r in &rows {
            let bytes = original.to_bytes().unwrap();
            let mut restored =
                DifferenceState::from_bytes(&bytes, TaskMode::Forecasting, &input(), fold.clone())
                    .unwrap();
            let a = original
                .push(r.group.clone(), r.timestamp, &r.values)
                .unwrap();
            let b = restored
                .push(r.group.clone(), r.timestamp, &r.values)
                .unwrap();
            assert_eq!(a.output, b.output);
            assert_eq!(original.to_bytes().unwrap(), restored.to_bytes().unwrap());
        }
        original.finish().unwrap();
    }
}

#[test]
fn malformed_fitted_ridge_words_and_dimensions_are_refused_before_prediction() {
    let model = program()
        .fit_training(
            &[row("a", 0.0, 0.0), row("a", 1.0, 1.0)],
            &[0.0, 1.0],
            TaskMode::Ordered,
            FoldBoundary::new("cut", 2, 4).unwrap(),
        )
        .unwrap();
    let json = model.to_json().unwrap();
    let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    let ridge = &wire["predictor"]["predictor"]["Legacy"]["Pipeline"]["head"]["Ridge"];
    let unknown = json.replace("\"Legacy\":", "\"Unknown\":");
    assert_ne!(unknown, json);
    assert!(DifferenceModel::from_json(&unknown)
        .unwrap_err()
        .to_string()
        .contains("unknown variant"));
    for (field, value) in [
        ("w", serde_json::json!([])),
        ("means", serde_json::json!([])),
        ("scales", serde_json::json!([])),
        ("scales", serde_json::json!([{"f64_bits":0}])),
    ] {
        let old = format!(
            "\"{field}\":{}",
            serde_json::to_string(&ridge[field]).unwrap()
        );
        let replacement = format!("\"{field}\":{value}");
        assert_eq!(json.matches(&old).count(), 1);
        let bad = json.replace(&old, &replacement);
        let error = DifferenceModel::from_json(&bad).unwrap_err().to_string();
        assert!(error.contains("ridge state"), "{field}: {error}");
    }
}

#[test]
fn finite_labels_that_overflow_legacy_fit_refuse_instead_of_creating_null_words() {
    let fit = program().fit_training(
        &[row("a", 0.0, 0.0), row("a", 1.0, 1.0)],
        &[f64::MAX, f64::MAX],
        TaskMode::Ordered,
        FoldBoundary::new("cut", 2, 4).unwrap(),
    );
    assert!(fit.is_err());
}

#[test]
fn public_cursor_admission_is_closed_under_lossless_snapshots() {
    use feat_math::kernel::ordered_axis::{Axis, CausalCursor, Causality, FitScope, StageAxis};
    let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
    let fold = FoldBoundary::new("cursor", 1, 3).unwrap();
    let mut cursor = CausalCursor::new(stage, TaskMode::Ordered, &input(), fold.clone()).unwrap();
    let before = cursor.to_bytes().unwrap();
    for group in [
        GroupKey::NumericBits(f64::NAN.to_bits()),
        GroupKey::Categorical(" ".into()),
    ] {
        assert!(cursor.push(group, 0.0).is_err());
        assert_eq!(before, cursor.to_bytes().unwrap());
    }
    cursor
        .push(GroupKey::NumericBits((-0.0f64).to_bits()), 0.0)
        .unwrap();
    let bytes = cursor.to_bytes().unwrap();
    let mut restored =
        CausalCursor::from_bytes(&bytes, stage, TaskMode::Ordered, &input(), fold).unwrap();
    let admission = restored.push(GroupKey::numeric(0.0).unwrap(), 1.0).unwrap();
    assert_eq!(admission.prior_in_group, 1);
}

#[test]
fn opaque_downstream_codecs_replay_but_unproved_fit_shapes_refuse() {
    for source in [
        "(pipe X (standardize) (ridge lambda_frac=0.01))",
        "(pipe X (elasticnet alpha=0.001 l1_ratio=0.5 sweeps=10))",
    ] {
        let opaque =
            DifferenceProgram::new(config(), 1, input(), GenomeIr::parse(source).unwrap()).unwrap();
        let bytes = opaque.to_bytes().unwrap();
        assert_eq!(
            DifferenceProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let result = opaque.fit_training(
            &[row("a", 0.0, 0.0), row("a", 1.0, 1.0)],
            &[0.0, 1.0],
            TaskMode::Ordered,
            FoldBoundary::new("cut", 2, 4).unwrap(),
        );
        assert!(matches!(
            result,
            Err(feat_math::kernel::Error::Unsupported(_))
        ));
    }
    let model = program()
        .fit_training(
            &[row("a", 0.0, 0.0), row("a", 1.0, 1.0)],
            &[0.0, 1.0],
            TaskMode::Ordered,
            FoldBoundary::new("cut", 2, 4).unwrap(),
        )
        .unwrap();
    let json = model.to_json().unwrap();
    let bad = json.replace("\"indicator\":false", "\"indicator\":true");
    assert_ne!(bad, json);
    assert!(DifferenceModel::from_json(&bad)
        .unwrap_err()
        .to_string()
        .contains("compatibility input layout"));
}
