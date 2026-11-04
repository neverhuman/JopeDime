use super::*;

#[test]
fn fitted_numeric_pipeline_is_deterministic_strict_and_bit_exact() {
    assert!(
        feat_math::cxx::version().is_ok(),
        "a C++ compiler (CXX, default g++) is required"
    );
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=hybrid) (standardize) (ridge lambda_frac=0.02))",
    )
    .unwrap();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let model = fit_table(&genome, &task(), &numeric_schema(), &training_table(), &y).unwrap();
    supports_fitted_cpp17(&model).unwrap();
    let first = emit_fitted_cpp17(&model).unwrap();
    let second = emit_fitted_cpp17(&model).unwrap();
    assert_eq!(first.source, second.source);
    assert_eq!(first.source_sha256, second.source_sha256);
    assert_eq!(first.codec, second.codec);
    assert_eq!(first.codec_sha256, second.codec_sha256);
    assert!(first.codec.contains("jope-dime.fitted-cpp-typed-table.v1"));
    for banned in [
        "fstream", "ifstream", "ofstream", "fopen", "getenv", "system(", "socket(", "chrono",
    ] {
        assert!(
            !first.source.contains(banned),
            "generated source contains {banned}"
        );
    }

    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let expected = model
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(compile_and_run(&first.source, directory, false), expected);
    assert_eq!(compile_and_run(&first.source, directory, true), expected);
    scratch.close().unwrap();
}

#[test]
fn ordered_numeric_row_semantics_matches_bits() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let ordered_schema = DataSchema {
        columns: vec![ColumnSpec::numeric("a"), ColumnSpec::numeric("b")],
        row_semantics: RowSemantics::Ordered {
            group_by: Vec::new(),
            order_by: vec!["a".to_string()],
        },
    };
    let plain = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let ordered = fit_table(&plain, &task(), &ordered_schema, &training_table(), &y).unwrap();
    supports_fitted_cpp17(&ordered).unwrap();
    let artifact = emit_fitted_cpp17(&ordered).unwrap();
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let expected = ordered
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-ordered-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(
        compile_and_run(&artifact.source, directory, false),
        expected
    );
    scratch.close().unwrap();
}

#[test]
fn iid_live_transform_and_regression_head_corpus_matches_bits() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let cases = [
        "(pipe X (input numeric=median missing=indicator) (robust_scale) (winsor_clip q=0.1) (sigmoid) (step t=0.4) (norm_row) (select_corr k=4) (impute strategy=median) (target_bin_encode bins=4 prior=5) (ridge))",
        "(pipe X (input numeric=median) (maxabs_scale) (ridge))",
        "(pipe X (input numeric=median) (power_transform kind=yeo_johnson) (ridge))",
        "(pipe X (input numeric=median) (piecewise_linear bins=8) (ridge))",
        "(pipe X (input numeric=median) (corr_screen k=32) (ridge))",
        "(pipe X (input numeric=median) (pls_project k=2) (ridge))",
        "(pipe X (target kind=log1p) (input numeric=median) (piecewise_linear append=true bins=8) (ridge))",
        "(pipe X (input numeric=median) (tab_boost rounds=8 bins=8 min_leaf=1))",
        "(pipe X (input numeric=median) (extra_trees trees=8 depth=3 min_leaf=1))",
        "(pipe X (input numeric=median) (gam_boost rounds=8 bins=4))",
        "(pipe X (input numeric=median) (pls k=2))",
        "(pipe X (target kind=rank_gauss) (input numeric=median) (standardize) (ridge))",
        "(pipe X (target kind=log1p) (input numeric=median) (standardize) (ridge))",
        "(pipe X (target kind=winsor q=0.05) (input numeric=median) (standardize) (ridge))",
        "(pipe X (target kind=zscore) (input numeric=median) (standardize) (ridge))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-corpus-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task(), &numeric_schema(), &training_table(), &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_and_run(&artifact.source, directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn legacy_row_and_expression_transform_corpus_matches_bits() {
    let train = training_table()
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = [vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let cases = [
        "(pipe X (row_stats) (ridge))",
        "(pipe X (fft_mag k=1) (ridge))",
        "(pipe X (row_spectral) (ridge))",
        "(pipe X (row_diff_stats) (ridge))",
        "(pipe X (map expr=\"sigmoid(x)+row_mean\") (ridge))",
        "(pipe X (derive expr=\"row_l2+col(0)\") (ridge))",
        "(sym affine \"sigmoid(col(0)) + 0.5*col(1)\")",
        "(portfolio (member (pipe X (ridge))) (blend folds=5 guard=clip kind=nnls repeats=1))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-row-corpus-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_legacy_numeric_bound(&genome, &train, &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(
                &RawTable::new(
                    vec!["c0".to_string(), "c1".to_string()],
                    test.iter()
                        .map(|row| row.iter().map(f64::to_string).collect())
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &values,
                &missing,
                test.len(),
                2,
            ),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn complete_legacy_transform_registry_matches_bits() {
    let train = (0..40)
        .map(|index| {
            let x = index as f64 / 11.0 - 1.5;
            vec![
                x,
                (0.7 * x).sin(),
                (1.3 * x).cos(),
                (index % 5) as f64 - 2.0,
            ]
        })
        .collect::<Vec<_>>();
    let y = train
        .iter()
        .map(|row| 0.3 * row[0] - 0.4 * row[1] + row[2] * row[3] * 0.1)
        .collect::<Vec<_>>();
    let test = [
        vec![-1.7, -0.9, 0.4, -2.0],
        vec![0.2, 0.15, 0.95, 0.0],
        vec![2.3, 1.1, -0.8, 2.0],
    ];
    let cases = [
        "(pipe X (standardize append=true) (ridge))",
        "(pipe X (robust_scale append=true) (ridge))",
        "(pipe X (rank append=true) (ridge))",
        "(pipe X (blom append=true) (ridge))",
        "(pipe X (quantile_bins bins=5 append=true) (ridge))",
        "(pipe X (bin_onehot bins=4 append=true) (ridge))",
        "(pipe X (hinge knots=3 append=true) (ridge))",
        "(pipe X (winsor_clip q=0.1 append=true) (ridge))",
        "(pipe X (cheb degree=4 append=true) (ridge))",
        "(pipe X (poly_interact append=true) (ridge))",
        "(pipe X (rff_cos n=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (rff_sin n=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (rand_proj k=5 seed=11 append=true) (ridge))",
        "(pipe X (landmark_dist k=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (oblique_hinge n=5 seed=11 append=true) (ridge))",
        "(pipe X (pair_prod topk=4 append=true) (ridge))",
        "(pipe X (target_bin_encode bins=5 prior=3 append=true) (ridge))",
        "(pipe X (sigmoid append=true) (ridge))",
        "(pipe X (step t=0.2 append=true) (ridge))",
        "(pipe X (norm_row append=true) (ridge))",
        "(pipe X (polar append=true) (ridge))",
        "(pipe X (lag k=2 append=true) (ridge))",
        "(pipe X (roll k=3 append=true) (ridge))",
        "(pipe X (fft_mag k=2 append=true) (ridge))",
        "(pipe X (row_spectral append=true) (ridge))",
        "(pipe X (row_diff_stats append=true) (ridge))",
        "(pipe X (svd_project k=3 iters=8 seed=11 append=true) (ridge))",
        "(pipe X (nystrom k=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (symlog c=0.7 append=true) (ridge))",
        "(pipe X (select_corr k=3 append=true) (ridge))",
        "(pipe X (impute strategy=median append=true) (ridge))",
        "(pipe X (pair_ratio topk=5 append=true) (ridge))",
        "(pipe X (spline knots=4 append=true) (ridge))",
        "(pipe X (map expr=\"asinh(x)+row_mean\" append=true) (ridge))",
        "(pipe X (derive expr=\"row_l2+col(0)\" append=true) (ridge))",
    ];
    let raw = RawTable::new(
        (0..4).map(|column| format!("c{column}")).collect(),
        test.iter()
            .map(|row| row.iter().map(f64::to_string).collect())
            .collect(),
    )
    .unwrap();
    let train_schema = DataSchema {
        columns: (0..4)
            .map(|column| ColumnSpec::numeric(format!("c{column}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train_raw = RawTable::new(
        (0..4).map(|column| format!("c{column}")).collect(),
        train
            .iter()
            .map(|row| row.iter().map(f64::to_string).collect())
            .collect(),
    )
    .unwrap();
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-all-legacy-ops-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let legacy = fit_legacy_numeric_bound(&genome, &train, &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let model = if source.contains("(lag ") || source.contains("(roll ") {
            legacy
        } else {
            let v3 = fit_table(&genome, &task(), &train_schema, &train_raw, &y)
                .unwrap_or_else(|error| panic!("case {index} V3 fit: {error}"));
            assert_eq!(
                v3.predict_table(&raw)
                    .unwrap()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                legacy
                    .predict_table(&raw)
                    .unwrap()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                "case {index} V3 bridge: {source}"
            );
            v3
        };
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&raw)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &values,
                &missing,
                test.len(),
                4,
            ),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}
