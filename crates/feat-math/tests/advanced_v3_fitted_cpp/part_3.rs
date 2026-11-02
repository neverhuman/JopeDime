use super::*;

#[test]
fn admitted_serialized_legacy_head_corpus_matches_bits() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let cases = [
        "(pipe X (ridge_gcv))",
        "(pipe X (elasticnet alpha=0.001 l1_ratio=0.5 sweeps=5))",
        "(pipe X (histgbm rounds=3 depth=2 bins=8 min_leaf=1))",
        "(pipe X (ensemble rounds=3 depth=2 bins=8 min_leaf=1))",
        "(pipe X (histgbm2 rounds=8 leaves=4 min_leaf=1 bags=2))",
        "(pipe X (krr m=50))",
        "(pipe X (gpr m=50 steps=1))",
        "(pipe X (gpr kernel=rbf+linear m=50 steps=1))",
        "(pipe X (ksmooth k=3 gamma=1))",
        "(pipe X (hurdle2))",
        "(pipe X (stack))",
        "(pipe X (portfolio folds=5 repeats=1))",
        "(pipe X (portfolio preset=none add1=group_mean folds=5 repeats=1))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-legacy-heads-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let model = bound_with_serialized_legacy_head(source, &y);
        model.validate_binding().unwrap();
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
fn nontrivial_fitted_legacy_head_states_match_bits() {
    let train = (0..96)
        .map(|index| {
            let x = index as f64 / 17.0 - 2.0;
            vec![x, x.sin(), (x * 0.7).cos(), (index % 7) as f64 / 3.0]
        })
        .collect::<Vec<_>>();
    let y = train
        .iter()
        .enumerate()
        .map(|(index, row)| {
            if index % 9 == 0 {
                0.0
            } else {
                0.4 * row[0] - 0.7 * row[1] + 0.2 * row[2] * row[3]
            }
        })
        .collect::<Vec<_>>();
    let test = [
        vec![-1.7, (-1.7_f64).sin(), (-1.19_f64).cos(), 0.0],
        vec![-0.2, (-0.2_f64).sin(), (-0.14_f64).cos(), 1.0],
        vec![1.3, 1.3_f64.sin(), 0.91_f64.cos(), 2.0],
    ];
    let cases = [
        "(pipe X (histgbm2 rounds=12 leaves=6 min_leaf=2 bags=2))",
        "(pipe X (gpr kernel=rbf+linear m=24 steps=2))",
        "(pipe X (gpr kernel=periodic m=24 steps=2))",
        "(pipe X (hurdle2))",
        "(pipe X (stack))",
        "(pipe X (portfolio folds=5 repeats=1))",
    ];
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-nontrivial-heads-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let model = fit_legacy_numeric_bound(&GenomeIr::parse(source).unwrap(), &train, &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let raw = RawTable::new(
            (0..4).map(|column| format!("c{column}")).collect(),
            test.iter()
                .map(|row| row.iter().map(f64::to_string).collect())
                .collect(),
        )
        .unwrap();
        let expected = model
            .predict_table(&raw)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        if source.contains("(portfolio ") {
            let legacy = fit(&GenomeIr::parse(source).unwrap(), &train, &y).unwrap();
            let Model::Pipeline { head, .. } = legacy else {
                unreachable!()
            };
            let feat_math::heads::FittedHead::Portfolio(portfolio) = *head else {
                unreachable!()
            };
            for member in portfolio.members {
                let feat_math::portfolio::FittedMemberState::Fitted(member_model) = member.state
                else {
                    continue;
                };
                let mut member_json = serde_json::to_value(&model).unwrap();
                member_json["predictor"] = serde_json::json!({"Legacy": *member_model});
                let member_bound: BoundModel = serde_json::from_value(member_json).unwrap();
                let member_artifact = emit_fitted_cpp17(&member_bound).unwrap();
                let member_expected = member_bound
                    .predict_table(&raw)
                    .unwrap()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>();
                assert_eq!(
                    compile_typed(
                        &member_artifact.source,
                        directory,
                        false,
                        &values,
                        &missing,
                        test.len(),
                        4,
                    ),
                    member_expected,
                    "portfolio member {}",
                    member.name
                );
            }
        }
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

#[test]
fn frozen_legacy_pipeline_can_cross_the_bound_cpp_bridge() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let train = training_table()
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.parse::<f64>().unwrap())
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();
    let test_rows = vec![vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    let genome = GenomeIr::parse(
        "(pipe X (target kind=log1p) (quantile_bins append=true bins=8) (blom append=true) (pair_prod append=true topk=4) (elasticnet alpha=0.01 l1_ratio=0.1 sweeps=12))",
    )
    .unwrap();
    let model = fit_legacy_numeric_bound(&genome, &train, &y).unwrap();
    let artifact = emit_fitted_cpp17(&model).unwrap();
    let expected = fit(&genome, &train, &y)
        .unwrap()
        .predict(&test_rows)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let values = test_rows.into_iter().flatten().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-legacy-bridge-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(
        compile_typed(&artifact.source, directory, false, &values, &missing, 3, 2),
        expected
    );
    scratch.close().unwrap();
}

// ---------------------------------------------------------------------------------------------
// Emitter revision 3 (kernel2 X3): fitted index checks, empty rows and tied intervals
// ---------------------------------------------------------------------------------------------

/// The status `jope_fitted_predict` returns when a stage's fitted state does not fit the row it
/// reads (emitter revision 3).
const FITTED_WIDTH_STATUS: i32 = 8;
