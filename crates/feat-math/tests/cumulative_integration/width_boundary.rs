use feat_math::kernel::cumulative_integration::{Config, IntegrationState, Mode};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
#[test]
fn maximum_width_checkpoint_and_empty_row_refusal_are_transactional() {
    let input = OrderedInputIdentity::new("group", "time");
    let fold = FoldBoundary::new("width-boundary", 1, 2).unwrap();
    assert!(FoldBoundary::new("empty", 0, 0).is_err());
    for mode in [Mode::Rect, Mode::Trap] {
        let initial = (0..512)
            .map(|i| if i % 2 == 0 { -0.0 } else { 3.0 })
            .collect::<Vec<_>>();
        let config = Config::new(2, 60, mode, initial.clone()).unwrap();
        let mut state =
            IntegrationState::new(config, TaskMode::Forecasting, &input, fold.clone()).unwrap();
        let empty = state.to_bytes().unwrap();
        assert!(state
            .push(GroupKey::categorical("a").unwrap(), 0., &[])
            .is_err());
        assert_eq!(state.to_bytes().unwrap(), empty);
        let first = state
            .push(GroupKey::categorical("a").unwrap(), 0., &vec![2.; 512])
            .unwrap();
        assert_eq!(first.values.len(), 512);
        assert_eq!(
            first.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            initial.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let bytes = state.to_bytes().unwrap();
        let mut restored =
            IntegrationState::from_bytes(&bytes, TaskMode::Forecasting, &input, fold.clone())
                .unwrap();
        assert_eq!(restored.to_bytes().unwrap(), bytes);
        let out = restored
            .push(GroupKey::categorical("a").unwrap(), 0.5, &vec![2.; 512])
            .unwrap();
        assert_eq!(out.values.len(), 512);
        assert_eq!(out.faults, vec![0; 512]);
        for (i, v) in out.values.iter().enumerate() {
            assert_eq!(*v, if i % 2 == 0 { 1. } else { 4. });
        }
        restored.finish().unwrap();
    }
}
