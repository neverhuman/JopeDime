use feat_math::kernel::finite_difference::OrderedRow;
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::{
    Config, TimeGradientModel, TimeGradientProgram, TimeGradientState, BOUNDARY_HISTORY,
    TIME_GAP_REJECTED,
};
use feat_math::kernel::wire_v2::VersionError;
use feat_math::GenomeIr;
fn input() -> OrderedInputIdentity {
    OrderedInputIdentity::new("g", "t")
}
fn state(width: usize, gap: f64, train: usize, total: usize) -> TimeGradientState {
    TimeGradientState::new(
        Config::new(2, 75, width, gap).unwrap(),
        TaskMode::Forecasting,
        &input(),
        FoldBoundary::new("c", train, total).unwrap(),
    )
    .unwrap()
}
fn rows() -> Vec<OrderedRow> {
    (0..8)
        .map(|n| {
            let t = (n / 2) as f64 * 0.25;
            OrderedRow {
                group: GroupKey::categorical(if n % 2 == 0 { "a" } else { "b" }).unwrap(),
                timestamp: t,
                values: vec![3. * t + 2., t * t],
            }
        })
        .collect()
}
fn program() -> TimeGradientProgram {
    TimeGradientProgram::new(
        Config::new(2, 75, 2, 0.125).unwrap(),
        input(),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.001))").unwrap(),
    )
    .unwrap()
}
#[test]
fn irregular_linear_quadratic_secants_group_carry() {
    let mut s = state(2, 0.125, 2, 5);
    let a = GroupKey::categorical("a").unwrap();
    let b = GroupKey::categorical("b").unwrap();
    let first = s.push(a.clone(), 0., &[2., 0.]).unwrap();
    assert_eq!(first.values, vec![0., 0.]);
    assert_eq!(first.faults, vec![BOUNDARY_HISTORY; 2]);
    s.push(b.clone(), 0., &[1., 0.]).unwrap();
    let output = s.push(a.clone(), 0.5, &[3.5, 0.25]).unwrap();
    assert_eq!(output.values, vec![3., 0.5]);
    let bytes = s.to_bytes().unwrap();
    let mut resumed = TimeGradientState::from_bytes(
        &bytes,
        TaskMode::Forecasting,
        &input(),
        FoldBoundary::new("c", 2, 5).unwrap(),
    )
    .unwrap();
    assert_eq!(resumed.to_bytes().unwrap(), bytes);
    let left = s.push(a.clone(), 1.5, &[6.5, 2.25]).unwrap();
    let right = resumed.push(a, 1.5, &[6.5, 2.25]).unwrap();
    assert_eq!(left.values, vec![3., 2.]);
    assert_eq!(left.values, right.values);
    assert_eq!(left.faults, vec![0, 0]);
    let left = s.push(b.clone(), 0.25, &[1.75, 0.0625]).unwrap();
    let right = resumed.push(b, 0.25, &[1.75, 0.0625]).unwrap();
    assert_eq!(left.values, vec![3., 0.25]);
    assert_eq!(left.values, right.values);
    assert_eq!(s.to_bytes().unwrap(), resumed.to_bytes().unwrap());
    s.finish().unwrap();
    resumed.finish().unwrap();
}
#[test]
fn gap_refusal_is_atomic_and_preserves_typed_fault() {
    let mut s = state(1, 0.125, 1, 2);
    let g = GroupKey::numeric(-0.).unwrap();
    s.push(g, 1., &[1.]).unwrap();
    let before = s.to_bytes().unwrap();
    for time in [1., 0., 1.0625, f64::NAN, f64::INFINITY] {
        let e = s
            .push(GroupKey::numeric(0.).unwrap(), time, &[2.])
            .unwrap_err();
        assert_eq!(e.faults, TIME_GAP_REJECTED);
        let routed = VersionError::from(e);
        assert!(matches!(
            routed,
            VersionError::StageRejected {
                faults: TIME_GAP_REJECTED,
                ..
            }
        ));
        assert_eq!(s.to_bytes().unwrap(), before);
    }
    let output = s
        .push(GroupKey::numeric(0.).unwrap(), 1.125, &[2.])
        .unwrap();
    assert_eq!(output.values, vec![8.]);
    s.finish().unwrap();
}
#[test]
fn exact_minimum_gap_codec_raw_nonfinite_and_versions() {
    let mut s = state(2, f64::from_bits(1), 1, 2);
    s.push(
        GroupKey::categorical("a").unwrap(),
        0.,
        &[f64::from_bits(0x7ff8000000000123), -0.],
    )
    .unwrap();
    let bytes = s.to_bytes().unwrap();
    let mut resumed = TimeGradientState::from_bytes(
        &bytes,
        TaskMode::Forecasting,
        &input(),
        FoldBoundary::new("c", 1, 2).unwrap(),
    )
    .unwrap();
    assert_eq!(resumed.to_bytes().unwrap(), bytes);
    let output = resumed
        .push(
            GroupKey::categorical("a").unwrap(),
            f64::from_bits(1),
            &[0., -0.],
        )
        .unwrap();
    assert_eq!(
        output
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![0, 0]
    );
    assert_ne!(
        output.faults[0] & feat_math::math_expr::FaultSet::NONFINITE_INPUT.bits(),
        0
    );
    resumed.finish().unwrap();
    for n in 0..bytes.len() {
        assert!(TimeGradientState::from_bytes(
            &bytes[..n],
            TaskMode::Forecasting,
            &input(),
            FoldBoundary::new("c", 1, 2).unwrap()
        )
        .is_err());
    }
    let mut future = bytes;
    future[7] = 3;
    assert!(TimeGradientState::from_bytes(
        &future,
        TaskMode::Forecasting,
        &input(),
        FoldBoundary::new("c", 1, 2).unwrap()
    )
    .is_err());
}
#[test]
fn fit_checkpoint_predict_and_prefix_isolation() {
    let p = program();
    let rows = rows();
    let labels = [1., 2., 3., 4.];
    let fold = FoldBoundary::new("cut", 4, 8).unwrap();
    let mut a = p
        .fit_training(&rows[..4], &labels, TaskMode::Forecasting, fold.clone())
        .unwrap();
    let json = a.to_json().unwrap();
    let mut b = TimeGradientModel::from_json(&json).unwrap();
    assert_eq!(b.to_json().unwrap(), json);
    assert!(p
        .fit_training(&rows, &[1.; 8], TaskMode::Forecasting, fold.clone())
        .is_err());
    assert!(p
        .fit_training(&rows[..4], &labels, TaskMode::IID, fold.clone())
        .is_err());
    assert!(p
        .fit_training(
            &rows[..4],
            &[1., 2., f64::NAN, 4.],
            TaskMode::Forecasting,
            fold
        )
        .is_err());
    let left = a.predict_with_faults(&rows[4..]).unwrap();
    let right = b.predict_with_faults(&rows[4..]).unwrap();
    assert_eq!(
        left.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        right.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(left.faults, right.faults);
    assert_eq!(a.to_json().unwrap(), b.to_json().unwrap());
    a.finish().unwrap();
    b.finish().unwrap();
    let mut forged: serde_json::Value = serde_json::from_str(&json).unwrap();
    forged["format_version"] = 3.into();
    assert!(TimeGradientModel::from_json(&forged.to_string()).is_err());
}
#[test]
fn late_tiny_gap_model_batch_rolls_back() {
    let p = program();
    let rows = rows();
    let mut model = p
        .fit_training(
            &rows[..4],
            &[1., 2., 3., 4.],
            TaskMode::Forecasting,
            FoldBoundary::new("c", 4, 8).unwrap(),
        )
        .unwrap();
    let before = model.to_json().unwrap();
    let mut late = rows[4].clone();
    late.timestamp += 0.0625;
    let e = model.predict(&[rows[4].clone(), late]).unwrap_err();
    assert_eq!(e.faults, TIME_GAP_REJECTED);
    assert_eq!(model.to_json().unwrap(), before);
    model.predict(&rows[4..]).unwrap();
    model.finish().unwrap();
}
#[test]
fn configuration_and_program_envelopes_are_strict() {
    for width in [0, 513] {
        assert!(Config::new(2, 75, width, 0.125).is_err());
    }
    for gap in [-0., 0., -1., f64::NAN, f64::INFINITY] {
        assert!(Config::new(2, 75, 1, gap).is_err());
    }
    assert!(Config::new(1, 75, 1, 0.125).is_err());
    assert!(Config::new(2, 59, 1, 0.125).is_err());
    let p = program();
    let bytes = p.to_bytes().unwrap();
    assert_eq!(
        TimeGradientProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    for n in 0..bytes.len() {
        assert!(TimeGradientProgram::from_bytes(&bytes[..n]).is_err());
    }
    let mut future = bytes;
    future[9] = 3;
    assert!(TimeGradientProgram::from_bytes(&future).is_err());
}
