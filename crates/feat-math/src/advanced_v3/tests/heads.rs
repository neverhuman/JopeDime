use super::*;

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
