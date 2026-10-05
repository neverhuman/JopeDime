use feat_math::kernel::finite_difference::{DifferenceConfig, DifferenceState};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::{Config, TimeGradientState};
#[test]
fn irregular_secant_cannot_be_replaced_by_a_fixed_step_stencil() {
    let input = OrderedInputIdentity::new("g", "t");
    let mut secants = Vec::new();
    let mut fixed = Vec::new();
    for time in [1.0, 2.0] {
        let fold = FoldBoundary::new("same-cut", 1, 2).unwrap();
        let mut current = TimeGradientState::new(
            Config::new(2, 75, 1, 0.125).unwrap(),
            TaskMode::Forecasting,
            &input,
            fold.clone(),
        )
        .unwrap();
        let mut old = DifferenceState::new(
            DifferenceConfig::new(2, 59, 1, 1.0).unwrap(),
            1,
            TaskMode::Forecasting,
            &input,
            fold,
        )
        .unwrap();
        let group = GroupKey::categorical("a").unwrap();
        current.push(group.clone(), 0.0, &[1.0]).unwrap();
        old.push(group.clone(), 0.0, &[1.0]).unwrap();
        secants.push(current.push(group.clone(), time, &[3.0]).unwrap().values);
        fixed.push(old.push(group, time, &[3.0]).unwrap().output.values);
        current.finish().unwrap();
        old.finish().unwrap();
    }
    assert_eq!(secants, vec![vec![2.0], vec![1.0]]);
    assert_eq!(fixed, vec![vec![2.0], vec![2.0]]);
}
