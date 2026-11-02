#[cfg(test)]
mod tests {
    use super::super::schema::ColumnSpec;
    use super::super::types::Objective;
    use super::*;

    fn mixed_schema() -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSpec::numeric("number"),
                ColumnSpec::categorical("category"),
                ColumnSpec::numeric("constant"),
            ],
            row_semantics: RowSemantics::Iid,
        }
    }

    fn mixed_table() -> RawTable {
        RawTable::new(
            vec![
                "number".to_string(),
                "category".to_string(),
                "constant".to_string(),
            ],
            vec![
                vec!["0".to_string(), "a".to_string(), "7".to_string()],
                vec!["1".to_string(), "a".to_string(), "7".to_string()],
                vec!["2".to_string(), "b".to_string(), "7".to_string()],
                vec!["NA".to_string(), "nan".to_string(), "7".to_string()],
                vec!["4".to_string(), "null".to_string(), "7".to_string()],
                vec!["5".to_string(), "c".to_string(), "7".to_string()],
            ],
        )
        .unwrap()
    }

    #[test]
    fn profile_has_exactly_24_named_finite_statistics() {
        assert_eq!(PROFILE_STAT_NAMES.len(), PROFILE_STAT_COUNT);
        assert_eq!(PROFILE_STAT_NAMES.iter().collect::<BTreeSet<_>>().len(), 24);
        let profile = build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(profile.statistics().len(), 24);
        assert!(profile.statistics().iter().all(|value| value.is_finite()));
        assert_eq!(
            profile.statistic("numeric_column_fraction"),
            Some(2.0 / 3.0)
        );
        // Numeric NA and categorical null are missing; categorical "nan" is
        // deliberately a real category.
        assert_eq!(profile.statistic("missing_cell_fraction"), Some(2.0 / 18.0));
        assert!(profile.validate().is_ok());
    }

    #[test]
    fn repeated_profiles_are_byte_hash_deterministic() {
        let table = mixed_table();
        let schema = mixed_schema();
        let task = TaskSpec::regression(Objective::R2).unwrap();
        let targets = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let first = build_dataset_profile(&table, &schema, &task, &targets).unwrap();
        let second = build_dataset_profile(&table, &schema, &task, &targets).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
        assert_eq!(
            first.sketch().selection_sha256,
            second.sketch().selection_sha256
        );
    }

    #[test]
    fn deterministic_selection_respects_every_bound() {
        for total in [0, 1, 2, 255, 256, 257, 4_095, 4_096, 4_097, 100_000] {
            let selected = deterministic_indices(total, MAX_PROFILE_ROWS);
            assert_eq!(selected.len(), total.min(MAX_PROFILE_ROWS));
            assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(selected.iter().all(|index| *index < total));
        }
        let columns: Vec<usize> = (0..MAX_PROFILE_COLUMNS).collect();
        let pairs = deterministic_numeric_pairs(&columns, MAX_PROFILE_NUMERIC_PAIRS);
        assert_eq!(pairs.len(), MAX_PROFILE_NUMERIC_PAIRS);
        assert!(pairs.iter().all(|(left, right)| left < right));
        assert_eq!(
            pairs,
            deterministic_numeric_pairs(&columns, MAX_PROFILE_NUMERIC_PAIRS)
        );
    }

    #[test]
    fn duplicate_hashing_scans_rows_outside_the_profile_sketch() {
        let mut rows: Vec<Vec<String>> = (0..(MAX_PROFILE_ROWS + 1))
            .map(|row| vec![row.to_string()])
            .collect();
        rows[MAX_PROFILE_ROWS][0] = "0".to_string();
        let table = RawTable::new(vec!["x".to_string()], rows).unwrap();
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x")],
            row_semantics: RowSemantics::Iid,
        };
        let targets: Vec<f64> = (0..table.rows.len()).map(|row| row as f64).collect();
        let profile = build_dataset_profile(
            &table,
            &schema,
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &targets,
        )
        .unwrap();
        assert_eq!(
            profile.statistic("duplicate_raw_row_fraction"),
            Some(1.0 / (MAX_PROFILE_ROWS + 1) as f64)
        );
        assert_eq!(profile.sketch().selected_rows, MAX_PROFILE_ROWS);
        assert_eq!(profile.sketch().source_rows, MAX_PROFILE_ROWS + 1);
    }

    #[test]
    fn regression_and_binary_landmark_catalogs_are_fixed_and_finite() {
        let regression = build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(regression.landmarks().scores.len(), 4);
        assert!(regression
            .landmarks()
            .scores
            .iter()
            .all(|entry| entry.score.is_finite()));

        let binary = build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::binary(Objective::LogLoss).unwrap(),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            binary
                .landmarks()
                .scores
                .iter()
                .map(|entry| entry.kind)
                .collect::<Vec<_>>(),
            vec![
                CheapLandmarkKind::BinaryPrior,
                CheapLandmarkKind::BinaryLogitRidge,
                CheapLandmarkKind::BinaryShallowTree,
                CheapLandmarkKind::BinaryShallowGam,
            ]
        );
        assert!(binary
            .landmarks()
            .scores
            .iter()
            .all(|entry| entry.score.is_finite()));
    }

    #[test]
    fn pls_landmark_is_included_only_when_encoded_width_exceeds_rows() {
        let columns: Vec<ColumnSpec> = (0..4)
            .map(|column| ColumnSpec::numeric(format!("x{column}")))
            .collect();
        let headers: Vec<String> = columns.iter().map(|column| column.name.clone()).collect();
        let table = RawTable::new(
            headers,
            vec![
                vec!["0".to_string(); 4],
                vec!["1".to_string(); 4],
                vec!["2".to_string(); 4],
                vec!["3".to_string(); 4],
                vec!["4".to_string(); 4],
                vec!["5".to_string(); 4],
            ],
        )
        .unwrap();
        let schema = DataSchema {
            columns,
            row_semantics: RowSemantics::Iid,
        };
        let profile = build_dataset_profile(
            &table,
            &schema,
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert!(profile
            .landmarks()
            .scores
            .iter()
            .any(|entry| entry.kind == CheapLandmarkKind::RegressionPls));
    }

    #[test]
    fn extreme_finite_values_still_produce_finite_profile_values() {
        let table = RawTable::new(
            vec!["x".to_string(), "y".to_string()],
            vec![
                vec![f64::MAX.to_string(), (-f64::MAX).to_string()],
                vec![(-f64::MAX).to_string(), f64::MAX.to_string()],
                vec!["0".to_string(), "0".to_string()],
            ],
        )
        .unwrap();
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x"), ColumnSpec::numeric("y")],
            row_semantics: RowSemantics::Iid,
        };
        let profile = build_dataset_profile(
            &table,
            &schema,
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[f64::MAX, -f64::MAX, 0.0],
        )
        .unwrap();
        assert!(profile.statistics().iter().all(|value| value.is_finite()));
        assert!(profile
            .landmarks()
            .scores
            .iter()
            .all(|entry| entry.score.is_finite()));
    }

    #[test]
    fn invalid_target_contracts_are_rejected() {
        assert!(build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::binary(Objective::LogLoss).unwrap(),
            &[0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        )
        .is_err());
        assert!(build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0],
        )
        .is_err());
    }
}
