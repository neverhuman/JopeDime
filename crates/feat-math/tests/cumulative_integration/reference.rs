use feat_math::kernel::cumulative_integration::{Config, IntegrationState, Mode};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
#[test]
fn reference_boundary_areas_and_saturation() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/cumulative_integration.json"
    ))
    .unwrap();
    let cases = fixture["reference_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 70);
    for row in cases {
        let mode = if row["mode"] == 0 {
            Mode::Rect
        } else {
            Mode::Trap
        };
        let value = |name: &str| f64::from_bits(row[name].as_u64().unwrap());
        let mut state = IntegrationState::new(
            Config::new(2, 60, mode, vec![0.]).unwrap(),
            TaskMode::Forecasting,
            &OrderedInputIdentity::new("g", "t"),
            FoldBoundary::new("ref", 1, 2).unwrap(),
        )
        .unwrap();
        let group = GroupKey::numeric(0.).unwrap();
        state
            .push(group.clone(), 0., &[value("left_bits")])
            .unwrap();
        let output = state
            .push(group, value("dt_bits"), &[value("right_bits")])
            .unwrap();
        assert_eq!(
            output.values[0].to_bits(),
            row["actual_bits"].as_u64().unwrap()
        );
        assert_eq!(output.faults[0], row["faults"].as_u64().unwrap() as u32);
        assert_eq!(
            output.faults[0] & feat_math::math_expr::FaultSet::MAGNITUDE_CLAMPED.bits() != 0,
            row["saturated"].as_bool().unwrap()
        );
        let actual = output.values[0];
        let reference = value("reference_bits");
        let positive_order = |x: f64| {
            if x.to_bits() >> 63 == 0 {
                x.to_bits() | (1 << 63)
            } else {
                !x.to_bits()
            }
        };
        assert!(
            positive_order(actual).abs_diff(positive_order(reference)) <= 2
                || (actual - reference).abs() <= f64::from_bits(1)
        );
        state.finish().unwrap();
    }
}

#[test]
fn constant_ramp_irregular_gap_groups_and_refusals() {
    for mode in [Mode::Rect, Mode::Trap] {
        let mut state = IntegrationState::new(
            Config::new(2, 60, mode, vec![-0., 3.]).unwrap(),
            TaskMode::Forecasting,
            &OrderedInputIdentity::new("g", "t"),
            FoldBoundary::new("irregular", 2, 5).unwrap(),
        )
        .unwrap();
        let a = GroupKey::categorical("A").unwrap();
        let b = GroupKey::categorical("B").unwrap();
        let before = state.to_bytes().unwrap();
        assert!(state.push(a.clone(), 0., &[]).is_err());
        assert_eq!(state.to_bytes().unwrap(), before);
        let output = state.push(a.clone(), 0., &[2., 0.]).unwrap();
        assert_eq!(
            output
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![(-0f64).to_bits(), 3f64.to_bits()]
        );
        assert_eq!(output.faults, vec![0, 0]);
        let output = state.push(b.clone(), 0., &[4., 1.]).unwrap();
        assert_eq!(
            output
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![(-0f64).to_bits(), 3f64.to_bits()]
        );
        let output = state.push(a.clone(), 0.5, &[2., 1.]).unwrap();
        assert_eq!(
            output.values,
            if mode == Mode::Rect {
                vec![1., 3.]
            } else {
                vec![1., 3.25]
            }
        );
        let before = state.to_bytes().unwrap();
        for timestamp in [0.5, 0.25, f64::NAN, f64::INFINITY] {
            assert!(state.push(a.clone(), timestamp, &[2., 1.]).is_err());
            assert_eq!(state.to_bytes().unwrap(), before);
        }
        let output = state.push(b, 0.25, &[4., 2.]).unwrap();
        assert_eq!(
            output.values,
            if mode == Mode::Rect {
                vec![1., 3.25]
            } else {
                vec![1., 3.375]
            }
        );
        let output = state.push(a, 1.5, &[2., 3.]).unwrap();
        assert_eq!(
            output.values,
            if mode == Mode::Rect {
                vec![3., 4.]
            } else {
                vec![3., 5.25]
            }
        );
        state.finish().unwrap();
    }
}
