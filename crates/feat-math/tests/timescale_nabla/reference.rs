use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::{Config, TimeGradientState, BOUNDARY_HISTORY};
use feat_math::math_expr::FaultSet;
#[test]
fn independent_rational_secants_cover_tiny_and_overflowing_differences() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/timescale_nabla.json"
    ))
    .unwrap();
    let cases = data["reference_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 62);
    for row in cases {
        let f = |key: &str| f64::from_bits(row[key].as_u64().unwrap());
        let mut state = TimeGradientState::new(
            Config::new(2, 75, 1, f64::from_bits(1)).unwrap(),
            TaskMode::Forecasting,
            &OrderedInputIdentity::new("g", "t"),
            FoldBoundary::new("rational", 1, 2).unwrap(),
        )
        .unwrap();
        let group = GroupKey::numeric(-0.).unwrap();
        let initial = state.push(group.clone(), 0., &[f("left_bits")]).unwrap();
        assert_eq!(initial.values, vec![0.]);
        assert_eq!(initial.faults, vec![BOUNDARY_HISTORY]);
        let output = state.push(group, f("dt_bits"), &[f("right_bits")]).unwrap();
        let saturated = row["saturated"].as_bool().unwrap();
        assert_eq!(
            output.faults[0],
            if saturated {
                FaultSet::MAGNITUDE_CLAMPED.bits()
            } else {
                0
            }
        );
        let reference = f("reference_bits");
        let order = |x: f64| {
            if x.to_bits() >> 63 == 0 {
                x.to_bits() | (1 << 63)
            } else {
                !x.to_bits()
            }
        };
        assert!(
            order(output.values[0]).abs_diff(order(reference)) <= 2
                || (output.values[0] - reference).abs() <= f64::from_bits(1),
            "{row:?}: {} vs {reference}",
            output.values[0]
        );
        state.finish().unwrap();
    }
}
