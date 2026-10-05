use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::{Config, TimeGradientState, TIME_GAP_REJECTED};
#[test]
fn maximum_width_checkpoint_and_gap_rejections_preserve_state() {
    let input = OrderedInputIdentity::new("group", "time");
    let fold = FoldBoundary::new("maximum-width", 1, 2).unwrap();
    let mut state = TimeGradientState::new(
        Config::new(2, 75, 512, 0.125).unwrap(),
        TaskMode::Forecasting,
        &input,
        fold.clone(),
    )
    .unwrap();
    let group = GroupKey::categorical("a").unwrap();
    assert!(state.push(group.clone(), 0., &[]).is_err());
    let first = state.push(group.clone(), 0., &vec![2.; 512]).unwrap();
    assert_eq!(first.values.len(), 512);
    assert_eq!(first.values, vec![0.; 512]);
    let bytes = state.to_bytes().unwrap();
    let mut restored =
        TimeGradientState::from_bytes(&bytes, TaskMode::Forecasting, &input, fold).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    for t in [0., -1., 0.0625, f64::NAN, f64::INFINITY] {
        let rejected = restored.push(group.clone(), t, &vec![5.; 512]).unwrap_err();
        assert_eq!(rejected.faults, TIME_GAP_REJECTED);
        assert_eq!(restored.to_bytes().unwrap(), bytes);
    }
    let output = restored.push(group, 0.5, &vec![5.; 512]).unwrap();
    assert_eq!(output.values, vec![6.; 512]);
    assert_eq!(output.faults, vec![0; 512]);
    restored.finish().unwrap();
    assert!(Config::new(2, 75, 513, 0.125).is_err());
}
