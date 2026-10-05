use feat_math::kernel::realized_path_stats::{summarize, Config, BOUNDARY_HISTORY};
use feat_math::math_expr::FaultSet;
#[test]
fn independent_energy_and_high_precision_bipower_reference() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/realized_path_stats.json"
    ))
    .unwrap();
    let cases = data["reference_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 45);
    for row in cases {
        let values: Vec<f64> = row["observations_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| f64::from_bits(x.as_u64().unwrap()))
            .collect();
        let output = summarize(
            &values,
            Config::new(2, 64, row["window"].as_u64().unwrap() as usize).unwrap(),
        );
        let flags = if row["boundary"].as_bool().unwrap() {
            BOUNDARY_HISTORY
        } else {
            0
        } | if row["input_fault"].as_bool().unwrap() {
            FaultSet::NONFINITE_INPUT.bits()
        } else {
            0
        } | if row["saturated"].as_bool().unwrap() {
            FaultSet::MAGNITUDE_CLAMPED.bits()
        } else {
            0
        };
        assert_eq!(output.faults, [flags; 5]);
        let refs = row["reference_bits"].as_array().unwrap();
        for (column, word) in refs.iter().enumerate() {
            let reference = f64::from_bits(word.as_u64().unwrap());
            assert!(output.values[column].is_finite());
            assert!(
                output.values[column]
                    .to_bits()
                    .abs_diff(reference.to_bits())
                    <= 4,
                "window={} column={column} {} vs {reference}",
                row["window"],
                output.values[column]
            );
        }
        assert_eq!(output.values[0], output.values[3] + output.values[4]);
    }
}
