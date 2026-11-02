use super::*;

#[test]
fn final_target_column_is_dropped() {
    let a = parse_feature_csv_text("1,2,0\n3,4,1\n", "a").unwrap();
    let b = parse_feature_csv_text("1,2,99\n3,4,-7\n", "b").unwrap();
    assert_eq!((a.n, a.c), (2, 2));
    assert_eq!(a.data.as_ref(), b.data.as_ref());
}

#[test]
fn test_percentiles_use_train_cdf_with_midpoint_ties() {
    let train = Matrix::new(vec![10.0, 20.0, 20.0, 40.0], 4, 1);
    let test = Matrix::new(vec![5.0, 20.0, 30.0, 50.0], 4, 1);
    let train_pct = train_column_percentiles(&train).unwrap();
    assert_eq!(train_pct.data.as_ref(), &[0.125, 0.375, 0.625, 0.875]);

    let test_pct = train_fitted_test_percentiles(&train, &test).unwrap();
    assert_eq!(test_pct.data.as_ref(), &[0.125, 0.5, 0.75, 0.875]);
}
