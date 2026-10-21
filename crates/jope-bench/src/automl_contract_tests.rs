//! Post-freeze API conformance only: no accuracy threshold or model selection feedback.
use super::submission::{Regressor, RegressorApi};

fn fixture(rows: usize, features: usize) -> (Vec<f64>, Vec<f64>) {
    let mut train = Vec::new();
    let mut predict = Vec::new();
    for i in 0..rows {
        for j in 0..features {
            let value = if (i + j) % 7 == 0 {
                f64::NAN
            } else {
                ((i + 3 * j) % 11) as f64
            };
            train.push(value);
            predict.push(value);
        }
        train.push((i % 13) as f64 - 6.0);
    }
    (train, predict)
}

#[test]
fn unfitted_and_invalid_shapes_fail_explicitly() {
    let mut model = Regressor::new(9);
    assert!(model.predict(&[], 0, 0).is_err());
    for (values, rows, columns) in [
        (&[][..], 0, 1),
        (&[][..], 1, 0),
        (&[][..], 1, 1),
        (&[][..], usize::MAX, 2),
    ] {
        assert!(model.fit(values, rows, columns).is_err());
    }
    model.fit(&[7.0], 1, 1).unwrap();
    assert!(model.predict(&[], usize::MAX, 2).is_err());
    assert!(model.predict(&[0.0], 1, 0).is_err());
    assert!(model.predict(&[0.0], 1, 1).is_err());
}

#[test]
fn final_column_is_target_and_zero_features_work() {
    let mut model = Regressor::new(7);
    model.fit(&[2.0, 4.0, 6.0], 3, 1).unwrap();
    assert_eq!(model.predict(&[], 3, 0).unwrap(), vec![4.0; 3]);
    assert!(model.predict(&[], 0, 0).unwrap().is_empty());
    model.fit(&[123.0, -999.0, 7.0], 1, 3).unwrap();
    assert_eq!(model.predict(&[123.0, -999.0], 1, 2).unwrap(), vec![7.0]);
    assert!(model.predict(&[123.0], 1, 1).is_err());
    assert!(model.predict(&[123.0, -999.0, 7.0], 1, 3).is_err());
}

#[test]
fn invalid_values_invalidate_old_state() {
    let mut model = Regressor::new(1);
    for invalid in [
        vec![f64::INFINITY, 2.0],
        vec![0.0, f64::NAN],
        vec![0.0, f64::NEG_INFINITY],
    ] {
        model.fit(&[1.0, 2.0], 1, 2).unwrap();
        assert!(model.fit(&invalid, 1, 2).is_err());
        assert!(model.predict(&[1.0], 1, 1).is_err());
    }
    model.fit(&[1.0, 2.0], 1, 2).unwrap();
    assert!(model.predict(&[f64::INFINITY], 1, 1).is_err());
    assert_eq!(model.predict(&[1.0], 1, 1).unwrap(), vec![2.0]);
}

#[test]
fn missing_constants_and_extreme_scales_have_finite_outputs() {
    for scale in [1.0, 1.0e-300, 1.0e300, f64::MAX] {
        let mut model = Regressor::new(3);
        let table = [
            f64::NAN,
            -scale,
            scale,
            f64::NAN,
            scale,
            -scale,
            f64::NAN,
            0.0,
            0.0,
        ];
        model.fit(&table, 3, 3).unwrap();
        let result = model
            .predict(&[f64::NAN, scale, f64::NAN, -scale], 2, 2)
            .unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.is_finite()));
    }
    let mut model = Regressor::new(3);
    model.fit(&[f64::NAN, 5.0, f64::NAN, 5.0], 2, 2).unwrap();
    assert_eq!(model.predict(&[f64::NAN, 0.0], 2, 1).unwrap(), vec![5.0; 2]);
}

#[test]
fn validation_path_is_deterministic_and_batch_independent() {
    let (train, features) = fixture(64, 3);
    let mut model = Regressor::new(17);
    model.fit(&train, 64, 4).unwrap();
    let all = model.predict(&features, 64, 3).unwrap();
    assert!(all.iter().all(|v| v.is_finite()));
    let mut batched = model.predict(&features[..21], 7, 3).unwrap();
    batched.extend(model.predict(&features[21..], 57, 3).unwrap());
    assert_eq!(all, batched);
    for &i in &[63, 0, 15, 7] {
        assert_eq!(
            model.predict(&features[3 * i..3 * i + 3], 1, 3).unwrap()[0],
            all[i]
        );
    }
    assert_eq!(model.predict(&features, 64, 3).unwrap(), all);
    model.fit(&[42.0], 1, 1).unwrap();
    model.fit(&train, 64, 4).unwrap();
    assert_eq!(model.predict(&features, 64, 3).unwrap(), all);
    let mut fresh = Regressor::new(17);
    fresh.fit(&train, 64, 4).unwrap();
    assert_eq!(fresh.predict(&features, 64, 3).unwrap(), all);
}

#[test]
fn wide_feature_screening_retains_valid_shapes() {
    let (train, features) = fixture(48, 140);
    let mut model = Regressor::new(11);
    model.fit(&train, 48, 141).unwrap();
    let out = model.predict(&features[..280], 2, 140).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn bounded_sampling_and_refit_are_reproducible() {
    let targets: Vec<f64> = (0..33_000).map(|i| (i % 17) as f64).collect();
    let mut a = Regressor::new(29);
    let mut b = Regressor::new(29);
    a.fit(&targets, targets.len(), 1).unwrap();
    b.fit(&targets, targets.len(), 1).unwrap();
    assert_eq!(a.predict(&[], 5, 0).unwrap(), b.predict(&[], 5, 0).unwrap());
}
