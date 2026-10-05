//! A row-only tree cannot distinguish these two causal histories.
use feat_math::kernel::capabilities::legacy_recipe_axis;
use feat_math::kernel::finite_difference::{DifferenceConfig, DifferenceState};
use feat_math::kernel::ordered_axis::{
    Axis, FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode,
};
use feat_math::ops::{apply_op, FittedOp};

#[test]
fn shortest_existing_lag_roll_have_a_different_axis_and_cannot_replace_wire59() {
    let last = [vec![3.]];
    for (name, op) in [
        (
            "lag",
            FittedOp::Lag {
                k: 1,
                append: false,
            },
        ),
        (
            "roll",
            FittedOp::Roll {
                k: 1,
                append: false,
            },
        ),
    ] {
        assert_eq!(
            legacy_recipe_axis(name).unwrap().axis,
            Axis::FeatureWithinRow
        );
        let left = apply_op(&op, &vec![vec![0.], vec![1.], vec![3.]]).unwrap();
        let right = apply_op(&op, &vec![vec![0.], vec![2.], vec![3.]]).unwrap();
        assert_eq!(left[2], right[2]);
    }
    let mut results = Vec::new();
    for prior in [1., 2.] {
        let mut state = DifferenceState::new(
            DifferenceConfig::new(2, 59, 1, 1.).unwrap(),
            1,
            TaskMode::Forecasting,
            &OrderedInputIdentity::new("group", "time"),
            FoldBoundary::new("same-cut", 1, 3).unwrap(),
        )
        .unwrap();
        state
            .push(GroupKey::categorical("same-group").unwrap(), 0., &[0.])
            .unwrap();
        state
            .push(GroupKey::categorical("same-group").unwrap(), 1., &[prior])
            .unwrap();
        results.push(
            state
                .push(GroupKey::categorical("same-group").unwrap(), 2., &last[0])
                .unwrap()
                .output
                .values,
        );
        state.finish().unwrap();
    }
    assert_eq!(results, vec![vec![2.], vec![1.]]);
    // Both streams share their training prefix and current feature row. A within-row
    // tree of any size reads that same row and cannot return both results.
}
