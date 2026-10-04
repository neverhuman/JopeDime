use feat_math::kernel::finite_difference::{
    DifferenceConfig, DifferenceModel, DifferenceProgram, DifferenceState, OrderedRow,
};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::{kernel::Program, GenomeIr};

fn input() -> OrderedInputIdentity {
    OrderedInputIdentity::new("group", "time")
}
fn config() -> DifferenceConfig {
    DifferenceConfig::new(2, 59, 2, 0.5).unwrap()
}
fn row(group: &str, time: f64, value: f64) -> OrderedRow {
    OrderedRow {
        group: GroupKey::categorical(group).unwrap(),
        timestamp: time,
        values: vec![value],
    }
}
fn program() -> DifferenceProgram {
    DifferenceProgram::new(
        config(),
        1,
        input(),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    )
    .unwrap()
}

#[test]
fn state_roundtrip_preserves_interleaved_groups_bits_faults_and_cut() {
    let fold = FoldBoundary::new("cut-a", 4, 8).unwrap();
    let rows = [
        row("a", 0.0, -0.0),
        row("b", 0.0, f64::from_bits(0x7ff8000000000042)),
        row("a", 0.5, 1.0),
        row("b", 0.5, 5.0),
        row("a", 1.0, 4.0),
        row("b", 0.5, 9.0),
        row("a", 4.0, 8.0),
        row("b", 9.0, 10.0),
    ];
    let mut original =
        DifferenceState::new(config(), 1, TaskMode::Forecasting, &input(), fold.clone()).unwrap();
    for r in &rows[..4] {
        original
            .push(r.group.clone(), r.timestamp, &r.values)
            .unwrap();
    }
    let bytes = original.to_bytes().unwrap();
    let mut restored =
        DifferenceState::from_bytes(&bytes, TaskMode::Forecasting, &input(), fold.clone()).unwrap();
    assert_eq!(bytes, restored.to_bytes().unwrap());
    let wrong = FoldBoundary::new("cut-b", 4, 8).unwrap();
    assert!(DifferenceState::from_bytes(&bytes, TaskMode::Forecasting, &input(), wrong).is_err());
    assert!(DifferenceState::from_bytes(&bytes, TaskMode::IID, &input(), fold.clone()).is_err());
    assert!(DifferenceState::from_bytes(
        &bytes,
        TaskMode::Forecasting,
        &OrderedInputIdentity::new("other", "time"),
        fold.clone()
    )
    .is_err());
    for r in &rows[4..] {
        let a = original
            .push(r.group.clone(), r.timestamp, &r.values)
            .unwrap();
        let b = restored
            .push(r.group.clone(), r.timestamp, &r.values)
            .unwrap();
        assert_eq!(a.admission, b.admission);
        assert_eq!(
            a.output
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            b.output
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(a.output.faults, b.output.faults);
        assert_eq!(original.to_bytes().unwrap(), restored.to_bytes().unwrap());
    }
    original.finish().unwrap();
    restored.finish().unwrap();
}

#[test]
fn truncated_corrupt_or_mismatched_state_refuses_and_bad_push_is_transactional() {
    let fold = FoldBoundary::new("cut", 2, 4).unwrap();
    let mut state =
        DifferenceState::new(config(), 1, TaskMode::Ordered, &input(), fold.clone()).unwrap();
    state
        .push(GroupKey::numeric(-0.0).unwrap(), 0.0, &[1.0])
        .unwrap();
    let bytes = state.to_bytes().unwrap();
    for n in 0..bytes.len() {
        assert!(
            DifferenceState::from_bytes(&bytes[..n], TaskMode::Ordered, &input(), fold.clone())
                .is_err(),
            "{n}"
        );
    }
    for (group, time, values) in [
        (GroupKey::NumericBits(f64::NAN.to_bits()), 1.0, vec![1.0]),
        (GroupKey::numeric(0.0).unwrap(), -1.0, vec![1.0]),
        (GroupKey::numeric(0.0).unwrap(), 1.0, vec![]),
    ] {
        assert!(state.push(group, time, &values).is_err());
        assert_eq!(bytes, state.to_bytes().unwrap());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(DifferenceState::from_bytes(&bad, TaskMode::Ordered, &input(), fold).is_err());
}

#[test]
fn versioned_ordered_program_codec_and_text_are_lossless_and_v1_refuses() {
    let p = program();
    let bytes = p.to_bytes().unwrap();
    assert!(Program::from_bytes(&bytes).is_err());
    let routed = DifferenceProgram::from_bytes(&bytes).unwrap();
    assert_eq!(bytes, routed.to_bytes().unwrap());
    for n in 0..bytes.len() {
        assert!(DifferenceProgram::from_bytes(&bytes[..n]).is_err());
    }
    for i in [7, 8, 9, 10] {
        let mut bad = bytes.clone();
        bad[i] = 255;
        assert!(DifferenceProgram::from_bytes(&bad).is_err());
    }
}

#[test]
fn ordered_fit_uses_only_train_prefix_and_predict_carries_transactionally() {
    let p = program();
    let train = (0..8)
        .map(|i| row("a", f64::from(i) * 0.5, f64::from(i * i)))
        .collect::<Vec<_>>();
    let labels = (0..8)
        .map(|i| f64::from(i) + 0.12345678901234567)
        .collect::<Vec<_>>();
    let fold = FoldBoundary::new("cut", 8, 11).unwrap();
    assert!(p
        .fit_training(&train, &labels, TaskMode::IID, fold.clone())
        .is_err());
    assert!(p
        .fit_training(&train, &[0.0; 11], TaskMode::Forecasting, fold.clone())
        .is_err());
    let mut model = p
        .fit_training(&train, &labels, TaskMode::Forecasting, fold)
        .unwrap();
    let json = model.to_json().unwrap();
    let mut restored = DifferenceModel::from_json(&json).unwrap();
    assert_eq!(json, restored.to_json().unwrap());
    assert!(serde_json::from_str::<feat_math::Model>(&json).is_err());
    assert!(model.predict(&[]).unwrap().is_empty());
    assert_eq!(json, model.to_json().unwrap());
    let wrong = [row("a", 4.0, 64.0), row("a", 3.0, 81.0)];
    assert!(model.predict(&wrong).is_err());
    assert_eq!(json, model.to_json().unwrap());
    let held = [
        row("a", 4.0, 64.0),
        row("a", 4.5, 81.0),
        row("a", 5.0, 100.0),
    ];
    let a = model.predict(&held).unwrap();
    let b = restored.predict(&held).unwrap();
    // Independent downstream fit on the mathematical second differences.
    let transformed = (0..8)
        .map(|i| vec![if i < 2 { 0.0 } else { 8.0 }])
        .collect::<Vec<_>>();
    let independent =
        feat_math::fit_legacy_numeric_bound(p.downstream(), &transformed, &labels).unwrap();
    let expected = independent
        .predict_table(&feat_math::advanced_v3::RawTable {
            headers: vec!["c0".into()],
            rows: vec![vec!["8".into()]; 3],
        })
        .unwrap();
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(model.to_json().unwrap(), restored.to_json().unwrap());
    model.finish().unwrap();
    restored.finish().unwrap();
    let mut poisoned: serde_json::Value = serde_json::from_str(&json).unwrap();
    poisoned["format_version"] = 3.into();
    assert!(DifferenceModel::from_json(&poisoned.to_string()).is_err());
}

#[test]
fn prediction_faults_and_future_poison_preserve_causal_prefix() {
    let p = program();
    let train = (0..4)
        .map(|i| row("a", f64::from(i), f64::from(i * i)))
        .collect::<Vec<_>>();
    let fold = FoldBoundary::new("cut", 4, 6).unwrap();
    let model = p
        .fit_training(&train, &[0.0, 1.0, 2.0, 3.0], TaskMode::Forecasting, fold)
        .unwrap();
    let mut a = model.clone();
    let mut b = model.clone();
    let good = [row("a", 4.0, 16.0), row("a", 5.0, 25.0)];
    let poison = [row("a", 4.0, 16.0), row("a", 5.0, f64::INFINITY)];
    let x = a.predict_with_faults(&good).unwrap();
    let y = b.predict_with_faults(&poison).unwrap();
    assert_eq!(x.values[0].to_bits(), y.values[0].to_bits());
    assert_eq!(x.faults[0], y.faults[0]);
    assert_ne!(
        y.faults[1][0] & feat_math::math_expr::FaultSet::NONFINITE_INPUT.bits(),
        0
    );
    let mut new = model.clone();
    let out = new.predict_with_faults(&[row("b", 0.0, 7.0)]).unwrap();
    assert_eq!(
        out.faults,
        vec![vec![feat_math::kernel::finite_difference::BOUNDARY_HISTORY]]
    );
}

#[test]
fn decoder_rejects_progress_counts_parameter_width_and_model_binding_mutations() {
    let p = program();
    let rows = [row("a", 0.0, 1.0), row("a", 1.0, 2.0)];
    let model = p
        .fit_training(
            &rows,
            &[0.0, 1.0],
            TaskMode::Ordered,
            FoldBoundary::new("cut", 2, 4).unwrap(),
        )
        .unwrap();
    let json = model.to_json().unwrap();
    let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    for (field, value) in [
        ("min_reader_version", 3.into()),
        ("required_features", 0.into()),
        ("numerical_policy", "v1".into()),
        ("grammar_sha256", "0".repeat(64).into()),
        ("task", 0.into()),
    ] {
        let mut bad = wire.clone();
        bad[field] = value;
        assert!(
            DifferenceModel::from_json(&bad.to_string()).is_err(),
            "{field}"
        );
    }
    let bytes = wire["state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect::<Vec<_>>();
    let cursor_length = u64::from_le_bytes(bytes[25..33].try_into().unwrap()) as usize;
    let history_count = 33 + cursor_length;
    for (offset, value) in [(8, 0), (17, 0), (history_count, 2)] {
        let mut bad = bytes.clone();
        bad[offset] = value;
        assert!(
            DifferenceState::from_bytes(
                &bad,
                TaskMode::Ordered,
                &input(),
                FoldBoundary::new("cut", 2, 4).unwrap()
            )
            .is_err(),
            "{offset}"
        );
    }
    let mut invalid_step = bytes.clone();
    invalid_step[9..17].fill(0xff);
    assert!(DifferenceState::from_bytes(
        &invalid_step,
        TaskMode::Ordered,
        &input(),
        FoldBoundary::new("cut", 2, 4).unwrap()
    )
    .is_err());
    for width in [0, feat_math::MAX_WIDTH + 1] {
        assert!(DifferenceProgram::new(config(), width, input(), p.downstream().clone()).is_err());
    }
}

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
