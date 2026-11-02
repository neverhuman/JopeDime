use super::*;
use crate::GenomeIr;
use std::process::Command;

fn schema() -> DataSchema {
    DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("country"),
        ],
        row_semantics: RowSemantics::Iid,
    }
}

fn training_table() -> RawTable {
    RawTable::new(
        vec!["amount".into(), "country".into()],
        vec![
            vec!["0".into(), "US".into()],
            vec!["0.1".into(), "US".into()],
            vec!["0.2".into(), "CA".into()],
            vec!["0.3".into(), "CA".into()],
            vec!["1.0".into(), "FR".into()],
            vec!["1.1".into(), "FR".into()],
            vec!["1.2".into(), "DE".into()],
            vec!["".into(), "DE".into()],
        ],
    )
    .unwrap()
}

#[test]
fn quoted_csv_parser_handles_commas_quotes_and_newlines() {
    let table =
        RawTable::from_csv("a,b\r\n1,\"x,y\"\r\n2,\"say \"\"hi\"\"\"\n3,\"two\nlines\"\n").unwrap();
    assert_eq!(table.headers, ["a", "b"]);
    assert_eq!(table.rows[0][1], "x,y");
    assert_eq!(table.rows[1][1], "say \"hi\"");
    assert_eq!(table.rows[2][1], "two\nlines");
}

#[test]
fn schema_mismatch_and_iid_row_ops_are_rejected() {
    let bad = RawTable::new(vec!["country".into(), "amount".into()], vec![]).unwrap();
    assert!(bad.validate_against(&schema()).is_err());
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let genome = GenomeIr::parse("(pipe X (lag k=1) (ridge))").unwrap();
    assert!(validate_v3_genome(&genome, &task, &schema()).is_err());
}

#[test]
fn ordered_training_values_do_not_reveal_unique_targets() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::categorical("id")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["id".into()],
        (0..10).map(|row| vec![format!("id-{row}")]).collect(),
    )
    .unwrap();
    let y: Vec<f64> = (0..10).map(|row| row as f64).collect();
    let stage =
        GenomeIr::parse("(pipe X (input categorical=ordered folds=5 prior=10) (ridge))").unwrap();
    let GenomeIr::Pipeline(dsl) = stage else {
        panic!()
    };
    let config = InputConfig::from_stage(Some(&dsl.stages[0])).unwrap();
    let (_, block) =
        FittedInput::fit_transform(config, &schema, &table, &y, TaskKind::Regression).unwrap();
    let expected: [f64; 10] = [5.0, 4.75, 4.5, 4.25, 4.0, 5.0, 4.75, 4.5, 4.25, 4.0];
    for (row, expected) in block.values.iter().zip(expected) {
        assert_eq!(row[0].to_bits(), expected.to_bits());
    }

    for validation_fold in 0..5 {
        let mut perturbed_y = y.clone();
        perturbed_y[validation_fold] += 1_000.0;
        perturbed_y[validation_fold + 5] -= 500.0;
        let config = InputConfig::from_stage(Some(&dsl.stages[0])).unwrap();
        let (_, perturbed) =
            FittedInput::fit_transform(config, &schema, &table, &perturbed_y, TaskKind::Regression)
                .unwrap();
        for row in [validation_fold, validation_fold + 5] {
            assert_eq!(
                block.values[row][0].to_bits(),
                perturbed.values[row][0].to_bits()
            );
        }
    }
}

#[test]
fn wide_raw_input_is_not_truncated() {
    let columns: Vec<ColumnSpec> = (0..1024)
        .map(|index| ColumnSpec::numeric(format!("c{index}")))
        .collect();
    let headers = columns.iter().map(|column| column.name.clone()).collect();
    let table = RawTable::new(headers, vec![vec!["1".into(); 1024]; 4]).unwrap();
    let schema = DataSchema {
        columns,
        row_semantics: RowSemantics::Iid,
    };
    let y = [0.0, 1.0, 0.0, 1.0];
    let (input, train) = FittedInput::fit_transform(
        InputConfig::compatibility(),
        &schema,
        &table,
        &y,
        TaskKind::Binary,
    )
    .unwrap();
    assert_eq!(train.values[0].len(), 1024);
    assert_eq!(input.transform(&table).unwrap().values[0].len(), 1024);
}

#[test]
fn logit_table_model_emits_finite_probabilities_and_handles_unknowns() {
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let genome = GenomeIr::parse(
            "(pipe X (input numeric=median missing=indicator categorical=hybrid max_onehot=16 min_count=2 prior=10 folds=5) (logit_ridge class_weight=balanced l2=0.1))",
        )
        .unwrap();
    let y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
    let test =
        RawTable::from_csv("amount,country\n0.05,\"US\"\n1.3,\"unknown, place\"\n,DE\n").unwrap();
    let predictions = model.predict_table(&test).unwrap();
    assert_eq!(predictions.len(), 3);
    assert!(predictions.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(predictions[1] > predictions[0]);
}

#[test]
fn bound_model_recomputes_vocabulary_identity_before_use() {
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let y = [0.0, 0.5, 0.25, 1.0, 0.75, 1.5, 1.25, 2.0];
    let mut model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
    model.vocabulary_sha256 = "00".repeat(32);
    let error = model.predict_table(&training_table()).unwrap_err();
    assert!(error.to_string().contains("vocabulary hash mismatch"));
    let error = emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap_err();
    assert!(error.to_string().contains("vocabulary hash mismatch"));
}

#[test]
fn every_new_head_fits_its_declared_task() {
    let binary = TaskSpec::binary(Objective::LogLoss).unwrap();
    let binary_y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for source in [
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (logit_ridge))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (tab_boost rounds=8 bins=8 min_leaf=1))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (extra_trees trees=8 depth=3 min_leaf=1))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (gam_boost rounds=8 bins=4))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (portfolio))",
        ] {
            let genome = GenomeIr::parse(source).unwrap();
            let model = fit_table(&genome, &binary, &schema(), &training_table(), &binary_y)
                .unwrap_or_else(|error| panic!("{source}: {error}"));
            assert!(model
                .predict_table(&training_table())
                .unwrap()
                .iter()
                .all(|value| (0.0..=1.0).contains(value)));
        }

    let regression = TaskSpec::regression(Objective::R2).unwrap();
    let regression_y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    for source in [
            "(pipe X (input numeric=median categorical=frequency) (tab_boost rounds=8 bins=8 min_leaf=1))",
            "(pipe X (input numeric=median categorical=frequency) (extra_trees trees=8 depth=3 min_leaf=1))",
            "(pipe X (input numeric=median categorical=frequency) (gam_boost rounds=8 bins=4))",
            "(pipe X (input numeric=median categorical=frequency) (pls k=2))",
        ] {
            let genome = GenomeIr::parse(source).unwrap();
            let model = fit_table(
                &genome,
                &regression,
                &schema(),
                &training_table(),
                &regression_y,
            )
            .unwrap_or_else(|error| panic!("{source}: {error}"));
            assert!(model
                .predict_table(&training_table())
                .unwrap()
                .iter()
                .all(|value| value.is_finite()));
        }
}

#[test]
fn imbalanced_binary_calibration_and_threshold_selection_are_safe() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["x".into()],
        (0..20).map(|row| vec![row.to_string()]).collect(),
    )
    .unwrap();
    let mut y = vec![0.0; 20];
    y[18] = 1.0;
    y[19] = 1.0;
    let genome = GenomeIr::parse(
            "(pipe X (input numeric=median missing=indicator) (logit_ridge class_weight=balanced calibration=platt threshold=f1))",
        )
        .unwrap();
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let model = fit_table(&genome, &task, &schema, &table, &y).unwrap();
    assert!(model.threshold.is_finite());
    assert!(model
        .predict_table(&table)
        .unwrap()
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
}

#[test]
fn all_new_transforms_are_reachable() {
    let schema = DataSchema {
        columns: (0..4)
            .map(|index| ColumnSpec::numeric(format!("x{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        (0..4).map(|index| format!("x{index}")).collect(),
        (0..40)
            .map(|row| {
                (0..4)
                    .map(|column| ((row * (column + 1)) as f64 / 17.0 - 2.0).to_string())
                    .collect()
            })
            .collect(),
    )
    .unwrap();
    let y: Vec<f64> = (0..40).map(|row| (row as f64 / 7.0).sin()).collect();
    let task = TaskSpec::regression(Objective::R2).unwrap();
    for source in [
        "(pipe X (input numeric=median) (maxabs_scale) (ridge))",
        "(pipe X (input numeric=median) (power_transform kind=yeo_johnson) (ridge))",
        "(pipe X (input numeric=median) (piecewise_linear bins=8) (ridge))",
        "(pipe X (input numeric=median) (corr_screen k=32) (ridge))",
        "(pipe X (input numeric=median) (pls_project k=2) (ridge))",
        "(pipe X (input numeric=median) (target_bin_encode bins=8 prior=10 oof=5) (ridge))",
    ] {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task, &schema, &table, &y).unwrap();
        let predictions = model.predict_table(&table).unwrap();
        assert_eq!(predictions.len(), y.len(), "{source}");
        assert!(
            predictions.iter().all(|value| value.is_finite()),
            "{source}"
        );
    }
}

#[test]
fn emitted_python_and_rust_match_runtime_on_quoted_csv() {
    require_toolchains();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-v3-")
        .tempdir()
        .unwrap();
    let temp = scratch.path();
    let csv = temp.join("test.csv");
    std::fs::write(
        &csv,
        "amount,country\n0.05,US\n1.3,\"unknown, place\"\n,DE\n",
    )
    .unwrap();
    let test = RawTable::from_csv(&std::fs::read_to_string(&csv).unwrap()).unwrap();

    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let genome = GenomeIr::parse(
            "(pipe X (input numeric=median missing=indicator categorical=hybrid max_onehot=16 min_count=2 prior=10 folds=5) (logit_ridge class_weight=balanced l2=0.1))",
        )
        .unwrap();
    let y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
    let emitted = emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
    assert!(emitted.manifest.contains(ARTIFACT_SCHEMA));
    assert_emitted_matches(&model, &test, &csv, temp, 0);

    let regression = TaskSpec::regression(Objective::R2).unwrap();
    let regression_genome = GenomeIr::parse(
            "(pipe X (target kind=zscore) (input numeric=median categorical=frequency) (standardize) (target_bin_encode bins=8 prior=10 oof=5) (maxabs_scale) (ridge))",
        )
        .unwrap();
    let regression_y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    let regression_model = fit_table(
        &regression_genome,
        &regression,
        &schema(),
        &training_table(),
        &regression_y,
    )
    .unwrap();
    assert_emitted_matches(&regression_model, &test, &csv, temp, 1);
}

#[test]
fn emitted_new_head_families_compile_and_match_runtime() {
    require_toolchains();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-v3-heads-")
        .tempdir()
        .unwrap();
    let temp = scratch.path();
    let csv = temp.join("test.csv");
    std::fs::write(&csv, "amount,country\n0.05,US\n1.3,FR\n,DE\n").unwrap();
    let test = RawTable::from_csv(&std::fs::read_to_string(&csv).unwrap()).unwrap();
    let binary = TaskSpec::binary(Objective::LogLoss).unwrap();
    let binary_y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let sources = [
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (tab_boost rounds=4 bins=8 min_leaf=1))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (extra_trees trees=3 depth=2 min_leaf=1))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (gam_boost rounds=4 bins=4))",
            "(pipe X (input numeric=median missing=indicator categorical=frequency) (portfolio))",
        ];
    for (index, source) in sources.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &binary, &schema(), &training_table(), &binary_y).unwrap();
        assert_emitted_matches(&model, &test, &csv, temp, index);
    }
    let regression = TaskSpec::regression(Objective::R2).unwrap();
    let regression_y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=frequency) (pls k=2))",
    )
    .unwrap();
    let model = fit_table(
        &genome,
        &regression,
        &schema(),
        &training_table(),
        &regression_y,
    )
    .unwrap();
    assert_emitted_matches(&model, &test, &csv, temp, sources.len());
}

#[test]
fn emitted_new_transform_boundaries_compile_and_match_runtime() {
    require_toolchains();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-v3-transforms-")
        .tempdir()
        .unwrap();
    let temp = scratch.path();
    let csv = temp.join("test.csv");
    std::fs::write(&csv, "amount,country\n0.05,US\n1.3,FR\n,DE\n").unwrap();
    let test = RawTable::from_csv(&std::fs::read_to_string(&csv).unwrap()).unwrap();
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    for (index, transform) in [
        "(maxabs_scale)",
        "(power_transform kind=yeo_johnson)",
        "(piecewise_linear bins=8)",
        "(corr_screen k=32)",
        "(pls_project k=2)",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(&format!(
                "(pipe X (input numeric=median missing=indicator categorical=frequency) {transform} (ridge))"
            ))
            .unwrap();
        let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
        assert_emitted_matches(&model, &test, &csv, temp, index);
    }
}

/// A fresh scratch directory, removed when the returned handle drops (also on panic).
fn scratch_dir(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("feat-math-v3-{tag}-"))
        .tempdir()
        .unwrap()
}

fn write_test_csv(temp: &std::path::Path, content: &str) -> (std::path::PathBuf, RawTable) {
    let csv = temp.join("test.csv");
    std::fs::write(&csv, content).unwrap();
    let table = RawTable::from_csv(content).unwrap();
    (csv, table)
}

#[test]
fn emitted_legacy_transform_chain_matches_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("legacy-chain");
    let temp = scratch.path();
    // Unknown quoted category + a 500-magnitude amount: the latter drives
    // the standardized feature far past ±20, so the artifact clamp must
    // match the reference exactly.
    let (csv, test) = write_test_csv(
        temp,
        "amount,country\n0.05,US\n1.3,\"unknown, place\"\n,DE\n500,\"say \"\"hi\"\"\"\n",
    );
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=onehot max_onehot=4) \
             (robust_scale) (winsor_clip q=0.1) (sigmoid) (step t=0.4) (norm_row) \
             (select_corr k=4) (impute strategy=median) (target_bin_encode bins=4 prior=5) \
             (standardize) (ridge))",
    )
    .unwrap();
    let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
    assert_emitted_matches(&model, &test, &csv, temp, 0);
}

#[test]
fn emitted_ordered_lag_roll_matches_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("ordered");
    let temp = scratch.path();
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("t"), ColumnSpec::numeric("v")],
        row_semantics: RowSemantics::Ordered {
            group_by: vec![],
            order_by: vec!["t".into()],
        },
    };
    let table = RawTable::new(
        vec!["t".into(), "v".into()],
        (0..8)
            .map(|row| {
                vec![
                    row.to_string(),
                    format!("{:?}", ((row * 3) as f64 * 0.37).sin()),
                ]
            })
            .collect(),
    )
    .unwrap();
    let y = [0.4, 0.1, 0.7, 0.3, 0.9, 0.2, 0.8, 0.5];
    let (csv, test) = write_test_csv(temp, "t,v\n8,0.1\n9,-0.4\n10,2.5\n");
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let genome =
        GenomeIr::parse("(pipe X (input numeric=median) (lag k=1) (roll k=2) (ridge))").unwrap();
    let model = fit_table(&genome, &task, &schema, &table, &y).unwrap();
    assert_emitted_matches(&model, &test, &csv, temp, 0);
}

#[test]
fn emitted_target_transforms_match_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("targets");
    let temp = scratch.path();
    let (csv, test) = write_test_csv(
        temp,
        "amount,country\n0.05,US\n1.3,\"unknown, place\"\n,DE\n",
    );
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let y = [0.1, 0.3, 0.5, 0.7, 2.1, 2.3, 2.5, 2.7];
    for (index, target) in [
        "(target kind=log1p)",
        "(target kind=winsor q=0.1)",
        "(target kind=rank_gauss)",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(&format!(
            "(pipe X {target} (input numeric=median categorical=frequency) (ridge))"
        ))
        .unwrap();
        let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
        assert_emitted_matches(&model, &test, &csv, temp, index);
    }
}

#[test]
fn emitted_calibration_matches_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("calibration");
    let temp = scratch.path();
    let (csv, test) = write_test_csv(
        temp,
        "amount,country\n0.05,US\n1.3,\"unknown, place\"\n,DE\n",
    );
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (index, source) in [
        "(pipe X (input numeric=median missing=indicator categorical=hybrid) \
             (logit_ridge calibration=platt threshold=f1))",
        "(pipe X (input numeric=median categorical=frequency) \
             (gam_boost rounds=6 bins=4 calibration=isotonic threshold=youden))",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
        assert_emitted_matches(&model, &test, &csv, temp, index);
    }
}

#[test]
fn emitted_regression_boost_heads_match_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("regression-heads");
    let temp = scratch.path();
    // The ",DE" row has a missing amount: with tab_boost/extra_trees and no
    // transforms the pipeline is missing-aware, so the NaN routing and the
    // regression NaN->0.0 final mapping are exercised end to end.
    let (csv, test) = write_test_csv(
        temp,
        "amount,country\n0.05,US\n1.3,\"unknown, place\"\n,DE\n",
    );
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    for (index, source) in [
        "(pipe X (input numeric=median missing=indicator categorical=frequency) \
             (tab_boost rounds=6 bins=8 min_leaf=1))",
        "(pipe X (input numeric=median missing=indicator categorical=frequency) \
             (extra_trees trees=4 depth=3 min_leaf=1))",
        "(pipe X (input numeric=median categorical=frequency) (gam_boost rounds=6 bins=4))",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
        assert_emitted_matches(&model, &test, &csv, temp, index);
    }
}

#[test]
fn emitted_encoders_and_special_categories_match_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("encoders");
    let temp = scratch.path();
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x"), ColumnSpec::categorical("label")],
        row_semantics: RowSemantics::Iid,
    };
    // Vocabulary stress: comma, doubled-quote, unicode, and empty (the
    // missing sentinel) categories must all round-trip through both
    // emitted artifacts' embedded literals.
    let table = RawTable::new(
        vec!["x".into(), "label".into()],
        vec![
            vec!["0.1".into(), "a,b".into()],
            vec!["0.2".into(), "say \"hi\"".into()],
            vec!["0.3".into(), "émojiØ".into()],
            vec!["0.4".into(), "".into()],
            vec!["0.5".into(), "plain".into()],
            vec!["0.6".into(), "a,b".into()],
            vec!["0.7".into(), "plain".into()],
            vec!["0.8".into(), "émojiØ".into()],
        ],
    )
    .unwrap();
    let y = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
    let (csv, test) = write_test_csv(
        temp,
        "x,label\n0.5,\"a,b\"\n0.7,\"say \"\"hi\"\"\"\n0.9,émojiØ\n1.1,\"\"\n1.3,unknown\n",
    );
    let task = TaskSpec::regression(Objective::R2).unwrap();
    for (index, source) in [
        "(pipe X (input numeric=median categorical=native) (ridge))",
        "(pipe X (input numeric=median categorical=ordered prior=5 folds=5) (ridge))",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task, &schema, &table, &y).unwrap();
        assert_emitted_matches(&model, &test, &csv, temp, index);
    }
}

#[test]
fn emitted_extreme_values_match_runtime_bitwise() {
    require_toolchains();
    let scratch = scratch_dir("extremes");
    let temp = scratch.path();
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x"), ColumnSpec::numeric("z")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["x".into(), "z".into()],
        (0..8)
            .map(|row| vec![format!("{}.25", row), format!("{}", 7 - row)])
            .collect(),
    )
    .unwrap();
    let y = [0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6];
    // 1e154 overflows (v+1)^lambda for lambda near 2: the Python artifact
    // must survive via the overflow guard and stay bit-identical to Rust
    // and the reference (which see powf -> inf -> sanitize).
    let (csv, test) = write_test_csv(temp, "x,z\n1e154,2\n-1e154,3\n500,-1\n0.2,0.4\n");
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median) (power_transform kind=yeo_johnson) (standardize) (ridge))",
    )
    .unwrap();
    let model = fit_table(&genome, &task, &schema, &table, &y).unwrap();
    assert_emitted_matches(&model, &test, &csv, temp, 0);

    // Well-separated binary data + far-out test points drive sigmoid
    // outputs toward 0/1: tiny probabilities must survive the artifact
    // output formatting bit-for-bit.
    let binary_schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x")],
        row_semantics: RowSemantics::Iid,
    };
    let binary_table = RawTable::new(
        vec!["x".into()],
        (0..8).map(|row| vec![row.to_string()]).collect(),
    )
    .unwrap();
    let binary_y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let (binary_csv, binary_test) = write_test_csv(temp, "x\n-1000\n1000\n3.5\n");
    let binary_task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let binary_genome =
        GenomeIr::parse("(pipe X (input numeric=median) (logit_ridge l2=0.0001))").unwrap();
    let binary_model = fit_table(
        &binary_genome,
        &binary_task,
        &binary_schema,
        &binary_table,
        &binary_y,
    )
    .unwrap();
    assert_emitted_matches(&binary_model, &binary_test, &binary_csv, temp, 1);
}

/// Parity is a hard invariant: a missing toolchain must FAIL the suite,
/// never silently skip it (tests/golden.rs precedent).
fn require_toolchains() {
    Command::new("python3")
        .arg("--version")
        .output()
        .expect("python3 must be on PATH");
    Command::new("rustc")
        .arg("--version")
        .output()
        .expect("rustc must be on PATH");
}

/// Emit, compile, run both artifacts on `csv`, and require BIT-EXACT
/// agreement (to_bits) with `predict_table` on every row.
fn assert_emitted_matches(
    model: &BoundModel,
    test: &RawTable,
    csv: &std::path::Path,
    temp: &std::path::Path,
    index: usize,
) {
    let emitted = emit_bound(model, &crate::transpile::EmitOptions::default()).unwrap();
    let py = temp.join(format!("model-{index}.py"));
    let rs = temp.join(format!("model-{index}.rs"));
    let bin = temp.join(format!("model-{index}"));
    std::fs::write(&py, emitted.python).unwrap();
    std::fs::write(&rs, emitted.rust).unwrap();
    let compile = Command::new("rustc")
        .args(["--edition=2021", "-O"])
        .arg(&rs)
        .arg("-o")
        .arg(&bin)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "artifact {index} rustc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let python = Command::new("python3").arg(&py).arg(csv).output().unwrap();
    let rust = Command::new(&bin).arg(csv).output().unwrap();
    assert!(
        python.status.success(),
        "artifact {index} python failed: {}",
        String::from_utf8_lossy(&python.stderr)
    );
    assert!(
        rust.status.success(),
        "artifact {index} rust failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let python: Vec<f64> = String::from_utf8(python.stdout)
        .unwrap()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    let rust: Vec<f64> = String::from_utf8(rust.stdout)
        .unwrap()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    let runtime = model.predict_table(test).unwrap();
    assert_eq!(python.len(), runtime.len(), "artifact {index}: python rows");
    assert_eq!(rust.len(), runtime.len(), "artifact {index}: rust rows");
    for (row, ((python, rust), runtime)) in python.iter().zip(&rust).zip(&runtime).enumerate() {
        assert!(
            python.to_bits() == runtime.to_bits(),
            "artifact {index} row {row}: python={python:?} runtime={runtime:?}"
        );
        assert!(
            rust.to_bits() == runtime.to_bits(),
            "artifact {index} row {row}: rust={rust:?} runtime={runtime:?}"
        );
    }
}

#[test]
fn legacy_transform_wide_bound_model_predicts_identically_after_reload() {
    let width = 300;
    let schema = DataSchema {
        columns: (0..width)
            .map(|j| ColumnSpec::numeric(format!("x{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let headers: Vec<_> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let rows: Vec<Vec<String>> = (0..12)
        .map(|i| (0..width).map(|j| ((i + j) % 7).to_string()).collect())
        .collect();
    let table = RawTable::new(headers, rows).unwrap();
    let target: Vec<_> = (0..12)
        .map(|i| if i % 7 >= 3 { 1.0 } else { 0.0 })
        .collect();
    let genome = GenomeIr::parse("(pipe X (input numeric=median missing=none) (robust_scale append=true) (histgbm depth=1 rounds=1 bins=8 min_leaf=1 lr=1 lambda=0))").unwrap();
    let model = fit_table(
        &genome,
        &TaskSpec::regression(Objective::R2).unwrap(),
        &schema,
        &table,
        &target,
    )
    .unwrap();
    let expected = model.predict_table(&table).unwrap();
    assert!(expected.iter().all(|v| v.is_finite()));
    assert!(expected.windows(2).any(|pair| pair[0] != pair[1]));
    let reloaded: BoundModel =
        serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
    reloaded.validate_binding().unwrap();
    let actual = reloaded.predict_table(&table).unwrap();
    assert_eq!(
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn synthesis_width_overflow_is_explicit_not_a_mask_dimension_failure() {
    let schema = DataSchema {
        columns: (0..6)
            .map(|j| ColumnSpec::numeric(format!("x{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        schema.columns.iter().map(|c| c.name.clone()).collect(),
        (0..20)
            .map(|i| (0..6).map(|j| ((i * (j + 1)) % 13).to_string()).collect())
            .collect(),
    )
    .unwrap();
    let target: Vec<_> = (0..20).map(|i| i as f64).collect();
    // Same transform programs as the two retained failed native worker recipes;
    // the numerical input here is entirely synthetic and carries no panel rows.
    for (program, width) in [
        ("(pipe X (input categorical=onehot max_onehot=256 min_count=1 missing=indicator numeric=median) (polar append=true) (target_bin_encode bins=8) (robust_scale append=true) (poly_interact) (standardize append=true) (histgbm depth=2 rounds=180))", 1024),
        ("(pipe X (input categorical=onehot max_onehot=256 min_count=1 missing=indicator numeric=median) (svd_project append=false iters=16 k=16) (bin_onehot bins=12) (norm_features append=true) (robust_scale append=true) (ridge))", 576),
    ] {
        let genome = GenomeIr::parse(program).unwrap();
        let error = fit_table(&genome, &TaskSpec::regression(Objective::R2).unwrap(), &schema, &table, &target).unwrap_err().to_string();
        assert!(error.contains(&format!("generated dense width {width} exceeds 512")), "{error}");
        assert!(!error.contains("row/mask"));
    }
}

#[test]
fn v3_admission_matches_manifest() {
    use crate::kernel::capabilities::{Manifest, OpKind, V3Admission};
    use grammar::{ProductionAdmission, RowRequirement, StageClass};
    let manifest = Manifest::current();
    let mut found = std::collections::BTreeSet::new();
    let mut descriptors = 0;
    for descriptor in grammar::v3_operation_registry() {
        descriptors += 1;
        for &name in std::iter::once(&descriptor.name).chain(descriptor.aliases.iter()) {
            let row = manifest
                .op(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            found.insert(name);
            let admission = match descriptor.admission {
                ProductionAdmission::Baseline => V3Admission::Baseline,
                ProductionAdmission::RequiresReceipt => V3Admission::RequiresReceipt,
                ProductionAdmission::Disabled => V3Admission::Disabled,
            };
            assert_eq!(row.v3_admission, admission, "{name}");
            let kind = match descriptor.stage_class {
                StageClass::Target => OpKind::Target,
                StageClass::Input => OpKind::Input,
                StageClass::Transform => OpKind::Transform,
                StageClass::Head => OpKind::Head,
            };
            assert_eq!(row.kind, kind, "{name}");
            assert_eq!(row.requires_labels, descriptor.target_aware, "{name}");
            assert_eq!(
                row.requires_row_order,
                descriptor.row_requirement == RowRequirement::Ordered,
                "{name}"
            );
            if crate::registry::operator(name).is_none() {
                assert_eq!(row.width_effect, descriptor.width_effect, "{name}");
            }
        }
    }
    assert_eq!(descriptors, 26);
    for row in &manifest.ops {
        if !found.contains(row.name) {
            assert_eq!(row.v3_admission, V3Admission::Absent, "{}", row.name);
        }
    }
}
