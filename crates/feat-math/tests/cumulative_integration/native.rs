use feat_math::kernel::cumulative_integration::{
    Config, IntegrationModel, IntegrationProgram, IntegrationState, Mode,
};
use feat_math::kernel::finite_difference::OrderedRow;
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::GenomeIr;
fn rows() -> Vec<OrderedRow> {
    (0..8)
        .map(|i| OrderedRow {
            group: GroupKey::categorical(if i % 2 == 0 { "a" } else { "b" }).unwrap(),
            timestamp: (i / 2) as f64 * 0.25,
            values: vec![i as f64 - 3., (i * i) as f64 / 5.],
        })
        .collect()
}
fn program(mode: Mode) -> IntegrationProgram {
    IntegrationProgram::new(
        Config::new(2, 60, mode, vec![-0., 3.]).unwrap(),
        OrderedInputIdentity::new("series", "time"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.001))").unwrap(),
    )
    .unwrap()
}
#[test]
fn model_fit_checkpoint_continuation_and_poison() {
    for mode in [Mode::Rect, Mode::Trap] {
        let p = program(mode);
        let rows = rows();
        let labels = vec![1., 2., 3., 4.];
        let fold = FoldBoundary::new("c", 4, 8).unwrap();
        let mut a = p
            .fit_training(&rows[..4], &labels, TaskMode::Forecasting, fold.clone())
            .unwrap();
        let json = a.to_json().unwrap();
        let mut b = IntegrationModel::from_json(&json).unwrap();
        assert_eq!(b.to_json().unwrap(), json);
        let av = a.predict_with_faults(&rows[4..]).unwrap();
        let bv = b.predict_with_faults(&rows[4..]).unwrap();
        assert_eq!(
            av.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bv.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(av.faults, bv.faults);
        assert_eq!(a.to_json().unwrap(), b.to_json().unwrap());
        a.finish().unwrap();
        b.finish().unwrap();
        assert!(p
            .fit_training(&rows, &[1.; 8], TaskMode::Forecasting, fold.clone())
            .is_err());
        assert!(p
            .fit_training(
                &rows[..4],
                &[1., 2., f64::NAN, 4.],
                TaskMode::Forecasting,
                fold.clone()
            )
            .is_err());
        assert!(p
            .fit_training(&rows[..4], &labels, TaskMode::IID, fold)
            .is_err());
        let unknown = json.replacen('{', "{\"unknown\":1,", 1);
        assert!(IntegrationModel::from_json(&unknown).is_err());
    }
}
#[test]
fn late_bad_row_rolls_back_whole_batch() {
    let p = program(Mode::Trap);
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
    let batch = vec![
        rows[4].clone(),
        OrderedRow {
            group: rows[4].group.clone(),
            timestamp: rows[4].timestamp,
            values: rows[5].values.clone(),
        },
    ];
    assert!(model.predict(&batch).is_err());
    assert_eq!(model.to_json().unwrap(), before);
    model.predict(&rows[4..]).unwrap();
    model.finish().unwrap();
}
#[test]
fn program_modes_initial_bits_and_version_refusal() {
    for mode in [Mode::Rect, Mode::Trap] {
        let p = program(mode);
        let bytes = p.to_bytes().unwrap();
        let q = IntegrationProgram::from_bytes(&bytes).unwrap();
        assert_eq!(q.to_bytes().unwrap(), bytes);
        assert_eq!(q.config().initial()[0].to_bits(), (-0f64).to_bits());
        assert_eq!(q.identity().unwrap(), p.identity().unwrap());
        let mut future = bytes;
        future[9] = 3;
        assert!(IntegrationProgram::from_bytes(&future).is_err());
    }
    for initial in [
        vec![],
        vec![0.; 513],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1e12 + 1.],
    ] {
        assert!(Config::new(2, 60, Mode::Rect, initial).is_err());
    }
}
#[test]
fn checkpoint_raw_nonfinite_timestamp_binding_and_atomic_ties() {
    let input = OrderedInputIdentity::new("g", "t");
    let fold = FoldBoundary::new("c", 1, 3).unwrap();
    let mut state = IntegrationState::new(
        Config::new(2, 60, Mode::Rect, vec![-0.]).unwrap(),
        TaskMode::Ordered,
        &input,
        fold.clone(),
    )
    .unwrap();
    let group = GroupKey::categorical("a").unwrap();
    let first = state
        .push(group.clone(), 0., &[f64::from_bits(0x7ff8000000000123)])
        .unwrap();
    assert_eq!(first.values[0].to_bits(), (-0f64).to_bits());
    assert_ne!(first.faults[0], 0);
    let bytes = state.to_bytes().unwrap();
    let mut resumed =
        IntegrationState::from_bytes(&bytes, TaskMode::Ordered, &input, fold.clone()).unwrap();
    assert_eq!(resumed.to_bytes().unwrap(), bytes);
    assert!(resumed.push(group.clone(), 0., &[1.]).is_err());
    assert_eq!(resumed.to_bytes().unwrap(), bytes);
    resumed.push(group.clone(), 0.25, &[1.]).unwrap();
    resumed.push(group, 0.5, &[2.]).unwrap();
    resumed.finish().unwrap();
    let mut bad = bytes.clone();
    bad[7] = 3;
    assert!(IntegrationState::from_bytes(&bad, TaskMode::Ordered, &input, fold.clone()).is_err());
    let mut offset = 8 + 1 + 8 + 8;
    let cursor_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8 + cursor_len + 8;
    assert_eq!(bytes[offset], 1);
    offset += 1;
    let text_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8 + text_len;
    bad = bytes;
    bad[offset..offset + 8].copy_from_slice(&1f64.to_bits().to_le_bytes());
    assert!(IntegrationState::from_bytes(&bad, TaskMode::Ordered, &input, fold).is_err());
}
