use super::*;

#[test]
fn adapter_validation_preserves_existing_and_gemini_contracts() {
    for id in [GPT6P1_SOL_MAX, GEMINI_ARGON_4] {
        for (table, rows, columns) in [
            (&[][..], 0, 1),
            (&[][..], 1, 0),
            (&[][..], usize::MAX, 2),
            (&[1.0][..], 1, 2),
            (&[0.0, f64::NAN][..], 1, 2),
            (&[0.0, f64::INFINITY][..], 1, 2),
            (&[f64::INFINITY, 2.0][..], 1, 2),
        ] {
            assert!(fit_flat(id, table, rows, columns, 41).is_err());
        }
        let first = fit_flat(id, &[f64::NAN, 7.0, 2.0, 7.0], 2, 2, 41).unwrap();
        assert!(first.predict_flat(&[], 1, 1).is_err());
        assert!(first.predict_flat(&[], usize::MAX, 2).is_err());
        assert!(first.predict_flat(&[f64::INFINITY], 1, 1).is_err());
        assert!(first.predict_flat(&[1.0, 7.0], 1, 2).is_err());
        assert!(first.predict_flat(&[], 0, 1).unwrap().is_empty());
        assert!(fit_rows(id, &[vec![1.0], vec![]], &[2.0, 3.0], 41).is_err());
        let second = fit_flat(id, &[1.0, -3.0, 2.0, -3.0], 2, 2, 41).unwrap();
        assert_eq!(second.predict_flat(&[1.0], 1, 1).unwrap(), [-3.0]);
        assert_eq!(first.predict_flat(&[f64::NAN], 1, 1).unwrap(), [7.0]);
    }
}

#[test]
fn gemini_registration_and_feature_only_transport() {
    assert_eq!(
        definition(GEMINI_ARGON_4).unwrap().source_sha256,
        GEMINI_ARGON_4_SHA256
    );
    let first = fit_rows(GEMINI_ARGON_4, &[vec![1.0], vec![2.0]], &[7.0, 7.0], 41).unwrap();
    let second = fit_rows(GEMINI_ARGON_4, &[vec![1.0], vec![2.0]], &[-3.0, -3.0], 41).unwrap();
    assert_eq!(
        first.predict_rows(&[vec![100.0], vec![f64::NAN]]).unwrap(),
        [7.0, 7.0]
    );
    assert_eq!(second.predict_rows(&[vec![100.0]]).unwrap(), [-3.0]);
    assert_eq!(first.predict_rows(&[vec![100.0]]).unwrap(), [7.0]);
    assert!(first.predict_rows(&[vec![100.0, 7.0]]).is_err());
    assert!(fit_rows(GEMINI_ARGON_4, &[vec![1.0], vec![]], &[1.0, 2.0], 41).is_err());
    assert!(fit_rows(GEMINI_ARGON_4, &[vec![1.0]], &[], 41).is_err());
    assert!(fit_flat("unknown", &[1.0], 1, 1, 41).is_err());
}

#[test]
fn gemini_rejects_malformed_inputs_and_invalidates_failed_refit() {
    let mut model = GeminiArgon4::new(41);
    assert!(model.predict(&[], 0, 0).is_err());
    for (table, rows, columns) in [
        (&[][..], 0, 1),
        (&[][..], 1, 0),
        (&[][..], 1, 1),
        (&[][..], usize::MAX, 2),
        (&[1.0][..], 1, 2),
        (&[f64::INFINITY, 1.0][..], 1, 2),
        (&[0.0, f64::NAN][..], 1, 2),
        (&[0.0, f64::NEG_INFINITY][..], 1, 2),
    ] {
        model.fit(&[1.0, 2.0], 1, 2).unwrap();
        assert!(model.fit(table, rows, columns).is_err());
        assert!(model.predict(&[1.0], 1, 1).is_err());
    }
    model.fit(&[1.0, 2.0], 1, 2).unwrap();
    assert!(model.predict(&[f64::INFINITY], 1, 1).is_err());
    assert!(model.predict(&[], usize::MAX, 2).is_err());
    assert!(model.predict(&[], 1, 1).is_err());
}

#[test]
fn gemini_missing_values_determinism_refit_and_batch_independence() {
    let table: Vec<f64> = (0..80)
        .flat_map(|i| {
            [
                if i % 7 == 0 { f64::NAN } else { i as f64 },
                (i % 5) as f64,
                (i % 13) as f64,
            ]
        })
        .collect();
    let rows = [f64::NAN, 2.0, 4.0, 1.0, 33.0, 0.0];
    let mut a = GeminiArgon4::new(41);
    a.fit(&table, 80, 3).unwrap();
    let expected = a.predict(&rows, 3, 2).unwrap();
    assert!(expected.iter().all(|v| v.is_finite()));
    for i in 0..3 {
        assert_eq!(
            a.predict(&rows[2 * i..2 * i + 2], 1, 2).unwrap()[0],
            expected[i]
        );
    }
    assert_eq!(
        a.predict(&[33.0, 0.0, f64::NAN, 2.0, 4.0, 1.0], 3, 2)
            .unwrap(),
        [expected[2], expected[0], expected[1]]
    );
    a.fit(&[9.0], 1, 1).unwrap();
    a.fit(&table, 80, 3).unwrap();
    let mut fresh = GeminiArgon4::new(41);
    fresh.fit(&table, 80, 3).unwrap();
    assert_eq!(a.predict(&rows, 3, 2).unwrap(), expected);
    assert_eq!(fresh.predict(&rows, 3, 2).unwrap(), expected);
}

#[test]
fn gemini_zero_features_and_constant_missing_columns() {
    let model = fit_flat(GEMINI_ARGON_4, &[2.0, 4.0, 6.0], 3, 1, 0).unwrap();
    assert_eq!(model.predict_flat(&[], 3, 0).unwrap(), [4.0; 3]);
    assert!(model.predict_flat(&[], 0, 0).unwrap().is_empty());
    assert!(model.predict_flat(&[], usize::MAX, 0).is_err());
    let model = fit_rows(
        GEMINI_ARGON_4,
        &[vec![f64::NAN], vec![f64::NAN]],
        &[5.0, 5.0],
        0,
    )
    .unwrap();
    assert_eq!(model.predict_rows(&[vec![f64::NAN]]).unwrap(), [5.0]);
}
