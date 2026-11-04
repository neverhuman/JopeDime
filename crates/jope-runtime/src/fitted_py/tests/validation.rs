use super::*;

#[test]
fn fitted_python_refuses_a_model_from_another_plan() {
    let (x, y) = train();
    let fitted_on = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let model = fit(&fitted_on, &x, &y).unwrap();
    for (other, message) in [
        (
            "(pipe X (robust_scale) (ridge))",
            "fitted standardize where the plan has robust_scale",
        ),
        (
            "(pipe X (standardize) (rank) (ridge))",
            "1 fitted stages for 2 planned stages",
        ),
        (
            "(pipe X (standardize) (elasticnet))",
            "fitted head ridge does not match the planned head elasticnet",
        ),
        (
            "(pipe X (target kind=zscore) (standardize) (ridge))",
            "no fitted target transform where the plan has one",
        ),
    ] {
        let error = emit(&GenomeIr::parse(other).unwrap(), &model, WIDTH).unwrap_err();
        assert!(format!("{error:#}").contains(message), "{other}: {error:#}");
    }
    let target = fit(
        &GenomeIr::parse("(pipe X (target kind=zscore) (ridge))").unwrap(),
        &x,
        &y,
    )
    .unwrap();
    let error = emit(
        &GenomeIr::parse("(pipe X (target kind=winsor) (ridge))").unwrap(),
        &target,
        WIDTH,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("target transform kind differs"),
        "{error:#}"
    );
}

#[test]
fn fitted_python_refuses_contradictory_prediction_parameters() {
    let (x, y) = train();
    let cases = [
        (
            "(pipe X (standardize append=true) (ridge))",
            "(pipe X (standardize) (ridge))",
            "append",
        ),
        (
            "(pipe X (step t=0.1) (ridge))",
            "(pipe X (step t=0.9) (ridge))",
            "t",
        ),
        (
            "(pipe X (cheb degree=3) (ridge))",
            "(pipe X (cheb degree=4) (ridge))",
            "degree",
        ),
        (
            "(pipe X (quantile_bins bins=4) (ridge))",
            "(pipe X (quantile_bins bins=8) (ridge))",
            "bins",
        ),
        (
            "(pipe X (bin_onehot bins=3) (ridge))",
            "(pipe X (bin_onehot bins=4) (ridge))",
            "bins",
        ),
        (
            "(pipe X (landmark_dist k=4 gamma=0.25) (ridge))",
            "(pipe X (landmark_dist k=4 gamma=1) (ridge))",
            "gamma",
        ),
        (
            "(pipe X (nystrom k=4 gamma=0.25) (ridge))",
            "(pipe X (nystrom k=4 gamma=1) (ridge))",
            "gamma",
        ),
        (
            "(pipe X (symlog c=0.5) (ridge))",
            "(pipe X (symlog c=2) (ridge))",
            "c",
        ),
        (
            "(pipe X (lag k=1) (ridge))",
            "(pipe X (lag k=2) (ridge))",
            "k",
        ),
        (
            "(pipe X (roll k=1) (ridge))",
            "(pipe X (roll k=2) (ridge))",
            "k",
        ),
        (
            "(pipe X (map expr=\"col(0)\") (ridge))",
            "(pipe X (map expr=\"sigmoid(col(0))\") (ridge))",
            "expr",
        ),
        (
            "(pipe X (ksmooth k=8))",
            "(pipe X (ksmooth k=16))",
            "head k",
        ),
        (
            "(pipe X (ksmooth gamma=0.5))",
            "(pipe X (ksmooth gamma=2))",
            "head gamma",
        ),
        (
            "(pipe X (gpr kernel=rq+linear m=50 steps=2))",
            "(pipe X (gpr kernel=rbf+linear m=50 steps=2))",
            "head kernel",
        ),
    ];
    for (fitted, declared, parameter) in cases {
        let model = fit(&GenomeIr::parse(fitted).unwrap(), &x, &y).unwrap();
        let error = emit(&GenomeIr::parse(declared).unwrap(), &model, WIDTH).unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("{parameter} parameter differs")),
            "{fitted} / {declared}: {error:#}"
        );
    }
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let model = fit(&genome, &x, &y).unwrap();
    assert!(
        emit(&genome, &model, WIDTH - 1).is_err(),
        "wrong input width accepted"
    );
    assert!(
        emit(&genome, &model, WIDTH).is_ok(),
        "zero-stage matching model refused"
    );
}

#[test]
fn fitted_python_refuses_non_finite_state_symbolic_models_and_f32_contracts() {
    let (x, y) = train();
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let mut model = fit(&genome, &x, &y).unwrap();
    if let Model::Pipeline { ops, .. } = &mut model {
        if let FittedOp::Standardize { means, .. } = &mut ops[0] {
            means[1] = f64::NAN;
        }
    }
    let error = emit(&genome, &model, WIDTH).unwrap_err();
    assert!(
        format!("{error:#}").contains("non-finite fitted state"),
        "{error:#}"
    );

    let symbolic = Model::Symbolic {
        expr: feat_math::math_expr::MathExpr::parse("col(0)").unwrap(),
        a: 0.0,
        b: 1.0,
    };
    let error = emit(&genome, &symbolic, WIDTH).unwrap_err();
    assert!(
        format!("{error:#}").contains("needs a pipeline model"),
        "{error:#}"
    );

    let input = include_bytes!("../../../tests/fixtures/math-pipeline/input.json");
    let state = include_bytes!("../../../tests/fixtures/math-pipeline/pipeline.json");
    let parts = std::collections::BTreeMap::from([
        (crate::math_pipeline::INPUT_PART.into(), input.to_vec()),
        (crate::math_pipeline::PIPELINE_PART.into(), state.to_vec()),
    ]);
    let legacy = MathPipeline::decode(&parts, crate::math_pipeline::PIPELINE_PART).unwrap();
    let error = emit_math_pipeline(&legacy).unwrap_err();
    assert!(
        format!("{error:#}").contains("prediction_version 1"),
        "{error:#}"
    );
}

/// Spec §10.5 determinism: the same model and emitter produce byte-identical source.
#[test]
fn fitted_python_is_byte_identical_for_the_same_model() {
    let (x, y) = train();
    for dsl in [
        "(pipe X (rank) (pair_prod topk=4 append=true) (gbm rounds=5))",
        "(pipe X (portfolio))",
    ] {
        let genome = GenomeIr::parse(dsl).unwrap();
        let first = emit(&genome, &fit(&genome, &x, &y).unwrap(), WIDTH).unwrap();
        let again = emit(&genome, &fit(&genome, &x, &y).unwrap(), WIDTH).unwrap();
        assert_eq!(
            crate::sha256_hex(first.as_bytes()),
            crate::sha256_hex(again.as_bytes()),
            "{dsl}"
        );
    }
}
