use super::*;

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
