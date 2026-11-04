use super::*;

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
