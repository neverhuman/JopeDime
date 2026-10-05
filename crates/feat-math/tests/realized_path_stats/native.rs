use feat_math::kernel::finite_difference::OrderedRow;
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::realized_path_stats::{
    summarize, Config, PathState, PathStatsModel, PathStatsProgram, BOUNDARY_HISTORY,
};
use feat_math::math_expr::FaultSet;
use feat_math::GenomeIr;
fn input() -> OrderedInputIdentity {
    OrderedInputIdentity::new("g", "t")
}
fn program() -> PathStatsProgram {
    PathStatsProgram::new(
        Config::new(2, 64, 4).unwrap(),
        input(),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.001))").unwrap(),
    )
    .unwrap()
}
fn rows() -> Vec<OrderedRow> {
    (0..12)
        .map(|n| OrderedRow {
            group: GroupKey::categorical(if n % 2 == 0 { "a" } else { "b" }).unwrap(),
            timestamp: (n / 2) as f64,
            values: vec![((n * 7) % 13) as f64 - 6.],
        })
        .collect()
}
#[test]
fn smooth_jump_signed_semivariances_and_row_energy_oracle() {
    let config = Config::new(2, 64, 4).unwrap();
    let bv = f64::from_bits(0x3ff921fb54442d18);
    for (path, positive, negative) in [
        (vec![0., 1., 2., 3.], 3., 0.),
        (vec![0., -1., -2., -3.], 0., 3.),
        (vec![0., 1., 0., 1.], 2., 1.),
    ] {
        let out = summarize(&path, config);
        assert_eq!(out.values[0], 3.);
        assert_eq!(out.values[1], 2. * bv);
        assert_eq!(out.values[2], 0.);
        assert_eq!(out.values[3], positive);
        assert_eq!(out.values[4], negative);
        assert_eq!(out.faults, [0; 5]);
        let oracle = feat_math::ops::apply_op(
            &feat_math::ops::FittedOp::RowDiffStats { append: false },
            &vec![path],
        )
        .unwrap();
        assert_eq!(out.values[0], oracle[0][7] * 3.);
    }
    let jump = summarize(&[0., 0., 5., 5.], config);
    assert_eq!(jump.values, [25., 0., 25., 25., 0.]);
    assert_eq!(jump.faults, [0; 5]);
}
#[test]
fn zero_one_partial_and_expired_nonfinite_history() {
    let c = Config::new(2, 64, 2).unwrap();
    for path in [vec![], vec![0.], vec![-0.]] {
        let out = summarize(&path, c);
        assert_eq!(out.values, [0.; 5]);
        assert_eq!(out.faults, [BOUNDARY_HISTORY; 5]);
    }
    let out = summarize(&[f64::NAN, 0., 1.], c);
    assert_eq!(out.values, [1., 0., 1., 1., 0.]);
    assert_eq!(out.faults, [0; 5]);
    let out = summarize(&[f64::NAN, 1.], c);
    assert_eq!(out.values, [1., 0., 1., 1., 0.]);
    assert_eq!(out.faults, [FaultSet::NONFINITE_INPUT.bits(); 5]);
}
#[test]
fn clamp_before_square_and_five_outputs_bound() {
    let c = Config::new(2, 64, 129).unwrap();
    let path = (0..129)
        .map(|i| if i % 2 == 0 { f64::MAX } else { -f64::MAX })
        .collect::<Vec<_>>();
    let out = summarize(&path, c);
    assert_eq!(out.values[0], 128e12);
    assert_eq!(out.values[3], 64e12);
    assert_eq!(out.values[4], 64e12);
    assert_eq!(out.values[2], 0.);
    assert!(out
        .values
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 2.1e14));
    assert_eq!(out.faults, [FaultSet::MAGNITUDE_CLAMPED.bits(); 5]);
}
#[test]
fn group_ties_checkpoint_raw_words_and_atomic_backwards() {
    let mut s = PathState::new(
        Config::new(2, 64, 3).unwrap(),
        TaskMode::Forecasting,
        &input(),
        FoldBoundary::new("c", 2, 5).unwrap(),
    )
    .unwrap();
    let a = GroupKey::numeric(-0.).unwrap();
    let b = GroupKey::categorical("b").unwrap();
    let first = s
        .push(a, 0., &[f64::from_bits(0x7ff8000000000123)])
        .unwrap();
    assert_eq!(first.values.len(), 5);
    s.push(b.clone(), 0., &[-0.]).unwrap();
    let bytes = s.to_bytes().unwrap();
    let mut t = PathState::from_bytes(
        &bytes,
        TaskMode::Forecasting,
        &input(),
        FoldBoundary::new("c", 2, 5).unwrap(),
    )
    .unwrap();
    assert_eq!(t.to_bytes().unwrap(), bytes);
    s.push(GroupKey::numeric(0.).unwrap(), 0., &[1.]).unwrap();
    t.push(GroupKey::numeric(0.).unwrap(), 0., &[1.]).unwrap();
    let before = s.to_bytes().unwrap();
    assert!(s.push(GroupKey::numeric(0.).unwrap(), -1., &[2.]).is_err());
    assert_eq!(s.to_bytes().unwrap(), before);
    assert!(s.push(b.clone(), 1., &[]).is_err());
    assert_eq!(s.to_bytes().unwrap(), before);
    s.push(b.clone(), 1., &[2.]).unwrap();
    t.push(b, 1., &[2.]).unwrap();
    let left = s.push(GroupKey::numeric(0.).unwrap(), 1., &[2.]).unwrap();
    let right = t.push(GroupKey::numeric(0.).unwrap(), 1., &[2.]).unwrap();
    assert_eq!(left.values, right.values);
    assert_eq!(left.faults, right.faults);
    assert_eq!(s.to_bytes().unwrap(), t.to_bytes().unwrap());
    s.finish().unwrap();
    t.finish().unwrap();
    for n in 0..bytes.len() {
        assert!(PathState::from_bytes(
            &bytes[..n],
            TaskMode::Forecasting,
            &input(),
            FoldBoundary::new("c", 2, 5).unwrap()
        )
        .is_err());
    }
}
#[test]
fn model_fixed_width_prefix_fit_and_checkpoint_continuation() {
    let p = program();
    assert_eq!(p.width(), 5);
    assert_eq!(p.input_width(), 1);
    let rows = rows();
    let labels = [1., 2., 3., 4., 5., 6.];
    let fold = FoldBoundary::new("cut", 6, 12).unwrap();
    let mut a = p
        .fit_training(&rows[..6], &labels, TaskMode::Forecasting, fold.clone())
        .unwrap();
    let json = a.to_json().unwrap();
    let mut b = PathStatsModel::from_json(&json).unwrap();
    assert_eq!(json, b.to_json().unwrap());
    assert!(p
        .fit_training(&rows, &[1.; 12], TaskMode::Forecasting, fold.clone())
        .is_err());
    assert!(p
        .fit_training(&rows[..6], &labels, TaskMode::IID, fold.clone())
        .is_err());
    let av = a.predict_with_faults(&rows[6..]).unwrap();
    let bv = b.predict_with_faults(&rows[6..]).unwrap();
    assert_eq!(
        av.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        bv.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(av.faults, bv.faults);
    assert!(av.faults.iter().all(|v| v.len() == 5));
    assert_eq!(a.to_json().unwrap(), b.to_json().unwrap());
    a.finish().unwrap();
    b.finish().unwrap();
}
#[test]
fn model_late_invalid_batch_and_strict_envelopes() {
    use feat_math::kernel::wire_v2::VersionedProgram;
    for window in 2..=129 {
        let config = Config::new(2, 64, window).unwrap();
        let p = PathStatsProgram::new(
            config,
            input(),
            GenomeIr::parse("(pipe X (ridge lambda_frac=0.001))").unwrap(),
        )
        .unwrap();
        let bytes = p.to_bytes().unwrap();
        assert_eq!(
            PathStatsProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let public = VersionedProgram::from_bytes(&bytes).unwrap();
        assert_eq!(
            VersionedProgram::parse_lossless_text(&public.to_lossless_text().unwrap())
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let fold = FoldBoundary::new("empty-window", 1, 2).unwrap();
        let bytes = PathState::new(config, TaskMode::Forecasting, &input(), fold.clone())
            .unwrap()
            .to_bytes()
            .unwrap();
        assert_eq!(
            PathState::from_bytes(&bytes, TaskMode::Forecasting, &input(), fold)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    let p = program();
    let rows = rows();
    let mut model = p
        .fit_training(
            &rows[..6],
            &[1., 2., 3., 4., 5., 6.],
            TaskMode::Forecasting,
            FoldBoundary::new("c", 6, 12).unwrap(),
        )
        .unwrap();
    let before = model.to_json().unwrap();
    let mut bad = rows[7].clone();
    bad.values = vec![1., 2.];
    assert!(model.predict(&[rows[6].clone(), bad]).is_err());
    assert_eq!(model.to_json().unwrap(), before);
    let bytes = p.to_bytes().unwrap();
    assert_eq!(
        PathStatsProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    for n in 0..bytes.len() {
        assert!(PathStatsProgram::from_bytes(&bytes[..n]).is_err());
    }
    let mut future = bytes;
    future[9] = 3;
    assert!(PathStatsProgram::from_bytes(&future).is_err());
    let mut future: serde_json::Value = serde_json::from_str(&before).unwrap();
    future["min_reader_version"] = 3.into();
    assert!(PathStatsModel::from_json(&future.to_string()).is_err());
    for window in [0, 1, 130, usize::MAX] {
        assert!(Config::new(2, 64, window).is_err());
    }
    model.predict(&rows[6..]).unwrap();
    model.finish().unwrap();
}
