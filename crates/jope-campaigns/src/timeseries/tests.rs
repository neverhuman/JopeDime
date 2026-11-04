#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> TimeSeriesManifestSpec {
        TimeSeriesManifestSpec {
            horizon: 3,
            seasonality: 2,
            frequency: "D".to_string(),
            target_is_final_column: true,
            known_future_covariates: vec!["promo".to_string()],
            past_only_covariates: Vec::new(),
        }
    }

    fn meta() -> Value {
        json!({
            "task_type": "timeseries",
            "horizon": 3,
            "seasonality": 2,
            "frequency": "D",
            "target_is_final_column": true,
            "n_features": 1,
            "n_train_rows": 5,
            "n_test_rows": 3,
            "feature_columns": ["promo"],
            "known_dynamic_columns": ["promo"],
            "past_dynamic_columns_excluded": ["customers"],
            "cleaning": {"past_only_covariates": "excluded"},
            "target_summary": {"mase_denominator": 2.0}
        })
    }

    #[test]
    fn mase_matches_hand_calculated_fixture() {
        let denominator = mase_denominator(&[10.0, 12.0, 14.0, 18.0], 2).unwrap();
        assert_eq!(denominator, 5.0);
        let train = TimeSeriesTable {
            x: vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
            y: vec![10.0, 12.0, 12.0, 14.0, 14.0],
            features: 1,
        };
        let test = TimeSeriesTable {
            x: vec![vec![1.0], vec![0.0], vec![1.0]],
            y: vec![16.0, 18.0, 20.0],
            features: 1,
        };
        let data = validate_timeseries_data(train, test, &meta(), &spec()).unwrap();
        let metrics = score_timeseries(&data, &[15.0, 18.0, 23.0]).unwrap();
        assert_eq!(metrics.mae, 4.0 / 3.0);
        assert_eq!(metrics.mase, (4.0 / 3.0) / 2.0);
    }

    #[test]
    fn loader_allows_univariate_target_last_csv() {
        let table = read_timeseries_csv_bytes(b"1\n2\n3\n", "fixture").unwrap();
        assert_eq!(table.features, 0);
        assert_eq!(table.y, vec![1.0, 2.0, 3.0]);
        assert_eq!(table.x, vec![Vec::<f64>::new(), Vec::new(), Vec::new()]);
    }

    #[test]
    fn bad_width_nonfinite_target_and_zero_denominator_are_rejected() {
        assert!(read_timeseries_csv_bytes(b"1,2\n1,2,3\n", "fixture").is_err());
        assert!(read_timeseries_csv_bytes(b"1,NaN\n", "fixture").is_err());
        assert!(mase_denominator(&[1.0, 1.0, 1.0], 1).is_err());
    }

    #[test]
    fn manifest_validation_rejects_past_only_and_target_not_last() {
        let mut bad = spec();
        bad.past_only_covariates = vec!["customers".to_string()];
        assert!(validate_timeseries_data(
            TimeSeriesTable {
                x: vec![vec![0.0]; 5],
                y: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                features: 1,
            },
            TimeSeriesTable {
                x: vec![vec![0.0]; 3],
                y: vec![6.0, 7.0, 8.0],
                features: 1,
            },
            &meta(),
            &bad,
        )
        .is_err());
        bad = spec();
        bad.target_is_final_column = false;
        assert!(validate_manifest_spec(&bad).is_err());
    }

    fn fixture_data() -> TimeSeriesData {
        let train = TimeSeriesTable {
            x: vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
            y: vec![10.0, 12.0, 12.0, 14.0, 14.0],
            features: 1,
        };
        let test = TimeSeriesTable {
            x: vec![vec![1.0], vec![0.0], vec![1.0]],
            y: vec![16.0, 18.0, 20.0],
            features: 1,
        };
        validate_timeseries_data(train, test, &meta(), &spec()).unwrap()
    }

    fn feature_index(names: &[String], name: &str) -> usize {
        names.iter().position(|n| n == name).unwrap()
    }

    #[test]
    fn direct_reduction_covers_all_leads_and_scales_targets() {
        let data = fixture_data();
        let reduction = make_direct_reduction(&data).unwrap();
        // origins 1..=4 with leads min(horizon=3, 5-origin):
        // (1,1)(1,2)(1,3)(2,1)(2,2)(2,3)(3,1)(3,2)(4,1)
        assert_eq!(reduction.x.len(), 9);
        assert_eq!(reduction.target_scale, 2.0);
        let lead_frac = feature_index(&reduction.feature_names, "lead_frac");
        let observed: BTreeSet<i64> = reduction
            .x
            .iter()
            .map(|row| (row[lead_frac] * 3.0).round() as i64)
            .collect();
        assert_eq!(
            observed,
            BTreeSet::from([1, 2, 3]),
            "train must see every lead 1..=H"
        );
        // first sorted pair is (origin=1, lead=1): target y[1]=12.0 scaled by 2.0
        assert_eq!(reduction.y[0], 6.0);
        // last sorted pair is (origin=4, lead=1): target y[4]=14.0 scaled by 2.0
        assert_eq!(reduction.y[8], 7.0);
        // scaling round-trips exactly on this fixture
        for (scaled, raw) in reduction
            .y
            .iter()
            .zip([12.0, 12.0, 14.0, 12.0, 14.0, 14.0, 14.0, 14.0, 14.0])
        {
            assert_eq!(scaled * reduction.target_scale, raw);
        }
    }

    #[test]
    fn direct_reduction_marks_unavailable_lags_and_checks_observed_values() {
        let data = fixture_data();
        let reduction = make_direct_reduction(&data).unwrap();
        let names = &reduction.feature_names;
        let first = &reduction.x[0]; // origin=1: single-point history
        for name in [
            "lag_2",
            "lag_3",
            "seasonal_lag_1",
            "seasonal_lag_2",
            "diff_1",
            "diff_seasonal",
            "rolling_absdiff_3",
        ] {
            assert!(
                first[feature_index(names, name)].is_nan(),
                "{name} must be NaN on a 1-point history"
            );
        }
        assert_eq!(first[feature_index(names, "lag_1")], 5.0); // y[0]=10.0 / 2.0
        let last = &reduction.x[8]; // origin=4: 4-point history, seasonality 2
        assert_eq!(last[feature_index(names, "lag_2")], 6.0); // y[2]=12.0 / 2.0
        assert_eq!(last[feature_index(names, "seasonal_lag_2")], 5.0); // lag 4 = y[0]
        assert_eq!(last[feature_index(names, "diff_1")], 1.0); // (14-12)/2
        assert_eq!(last[feature_index(names, "diff_seasonal")], 1.0); // (14-12)/2
    }

    #[test]
    fn direct_reduction_is_prefix_honest_on_monotone_series() {
        // strictly increasing series: every target strictly exceeds every
        // prefix value, so any leaked future value would show up in hist_max
        // or a lag exceeding the target.
        let n = 40;
        let y: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let train = TimeSeriesTable {
            x: vec![Vec::new(); n],
            y: y.clone(),
            features: 0,
        };
        let test = TimeSeriesTable {
            x: vec![Vec::new(); 3],
            y: vec![41.0, 42.0, 43.0],
            features: 0,
        };
        let denominator = mase_denominator(&y, 2).unwrap();
        let data = TimeSeriesData {
            train,
            test,
            horizon: 3,
            seasonality: 2,
            frequency: "D".to_string(),
            known_future_covariates: Vec::new(),
            mase_denominator: denominator,
        };
        let reduction = make_direct_reduction(&data).unwrap();
        let names = &reduction.feature_names;
        let hist_max = feature_index(names, "hist_max");
        let lag_1 = feature_index(names, "lag_1");
        for (row, target) in reduction.x.iter().zip(&reduction.y) {
            assert!(row[hist_max] < *target, "hist_max must predate the target");
            assert!(row[lag_1] < *target, "lag_1 must predate the target");
        }
    }

    #[test]
    fn direct_forecast_rows_use_full_history_without_feedback() {
        let data = fixture_data();
        let reduction = make_direct_reduction(&data).unwrap();
        let rows = make_direct_forecast_rows(&data).unwrap();
        assert_eq!(rows.len(), data.horizon);
        let names = &reduction.feature_names;
        let lag_1 = feature_index(names, "lag_1");
        let lead_frac = feature_index(names, "lead_frac");
        let promo = feature_index(names, "promo");
        for (h, row) in rows.iter().enumerate() {
            // every forecast row sees the same, real train history: no
            // prediction feedback anywhere.
            assert_eq!(row[lag_1], 7.0); // y[4]=14.0 / 2.0
            assert_eq!(row[lead_frac], (h + 1) as f64 / 3.0);
            assert_eq!(row[promo], data.test.x[h][0]);
            assert_eq!(row.len(), names.len());
        }
    }

    #[test]
    fn direct_reduction_caps_rows_newest_first() {
        let n = 9000;
        let y: Vec<f64> = (0..n)
            .map(|i| ((i % 7) + 1) as f64 + i as f64 * 0.01)
            .collect();
        let denominator = mase_denominator(&y, 7).unwrap();
        let data = TimeSeriesData {
            train: TimeSeriesTable {
                x: vec![Vec::new(); n],
                y: y.clone(),
                features: 0,
            },
            test: TimeSeriesTable {
                x: vec![Vec::new(); 4],
                y: vec![1.0, 2.0, 3.0, 4.0],
                features: 0,
            },
            horizon: 4,
            seasonality: 7,
            frequency: "D".to_string(),
            known_future_covariates: Vec::new(),
            mase_denominator: denominator,
        };
        let reduction = make_direct_reduction(&data).unwrap();
        assert_eq!(reduction.x.len(), DIRECT_REDUCTION_MAX_ROWS);
        // newest-first budget: only the most recent 2048 origins (8192/4)
        // survive, so every retained row has a fully observed seasonal lag.
        let names = &reduction.feature_names;
        let seasonal_lag_2 = feature_index(names, "seasonal_lag_2");
        assert!(reduction
            .x
            .iter()
            .all(|row| row[seasonal_lag_2].is_finite()));
    }

    #[test]
    fn supervised_reduction_uses_only_prior_targets() {
        let train = TimeSeriesTable {
            x: vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
            y: vec![10.0, 12.0, 12.0, 14.0, 14.0],
            features: 1,
        };
        let test = TimeSeriesTable {
            x: vec![vec![1.0], vec![0.0], vec![1.0]],
            y: vec![16.0, 18.0, 20.0],
            features: 1,
        };
        let data = validate_timeseries_data(train, test, &meta(), &spec()).unwrap();
        let reduction = make_supervised_reduction(&data).unwrap();
        assert_eq!(reduction.x.len(), 4);
        assert_eq!(reduction.y, vec![12.0, 12.0, 14.0, 14.0]);
        let lag_1 = reduction
            .feature_names
            .iter()
            .position(|name| name == "lag_1")
            .unwrap();
        assert_eq!(reduction.x[0][lag_1], 10.0);
        assert_eq!(reduction.x[1][lag_1], 12.0);
        let forecast = make_forecast_features(&data, &data.train.y, 0).unwrap();
        assert_eq!(forecast[lag_1], 14.0);
    }
}
