#[test]
fn scalar_v2_functions_policies_and_carriers_and_row_vector_stages_execute() {
    use feat_math::kernel::numerical_policy::NumericalPolicy;
    use feat_math::kernel::ordered_axis::Axis;
    use feat_math::kernel::row_vector::{Program, RowStage};
    use feat_math::kernel::scalar_program::{
        ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarNode,
    };
    use feat_math::math_expr::Var;
    use feat_math::math_expr::{BinOp, UnOp};
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/cuda-census/v2");
    let (x, y, probe) = data();
    let mut count = 0;
    let functions = (0..=u16::MAX)
        .filter_map(|id| {
            ScalarFunction::from_versioned_id(2, id)
                .ok()
                .map(|f| (id, f))
        })
        .collect::<Vec<_>>();
    assert!(!functions.is_empty());
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            for &(id, function) in &functions {
                let mut args = vec![ScalarNode::Var(if carrier == ScalarCarrier::Row {
                    Var::Col(0)
                } else {
                    Var::X
                })];
                if matches!(id, 7..=10) {
                    args.push(ScalarNode::Const(-0.5));
                }
                if id == 9 {
                    args.push(ScalarNode::Const(0.5));
                }
                let program = ScalarExpressionProgram::new(ScalarNode::Call(function, args))
                    .unwrap()
                    .with_numerical_policy(policy);
                let model = program.fit_training(&x, &y, carrier, 0.02).unwrap();
                let expected = model.predict(&probe).unwrap().0;
                let bundle = emit_scalar(&program, carrier, 5, 0.02).unwrap();
                support::run(
                    &root.join(format!("scalar-{policy:?}-{carrier:?}-{id}")),
                    &bundle,
                    &serde_json::json!({"train":x,"targets":y,"test":probe}),
                    &expected,
                    false,
                );
                count += 1;
            }
            // Function-name enumeration alone misses composite AST nodes and
            // row-statistic variables. Exercise all binary operations, unary
            // negation, literal bits and domain-fault behavior in both policies.
            let mut aggregate = ScalarNode::Const(-0.0);
            for var in [
                Var::RowMean,
                Var::RowSum,
                Var::RowL1,
                Var::RowL2,
                Var::RowMin,
                Var::RowMax,
                Var::RowStd,
                Var::Col(4),
            ] {
                aggregate = ScalarNode::Binary(
                    BinOp::Add,
                    Box::new(aggregate),
                    Box::new(ScalarNode::Var(var)),
                );
            }
            let input = ScalarNode::Var(if carrier == ScalarCarrier::Row {
                Var::Col(0)
            } else {
                Var::X
            });
            let expression = ScalarNode::Binary(
                BinOp::Sub,
                Box::new(ScalarNode::Unary(
                    UnOp::Neg,
                    Box::new(ScalarNode::Binary(
                        BinOp::Div,
                        Box::new(ScalarNode::Binary(
                            BinOp::Mul,
                            Box::new(input),
                            Box::new(aggregate),
                        )),
                        Box::new(ScalarNode::Const(0.0)),
                    )),
                )),
                Box::new(ScalarNode::Binary(
                    BinOp::Pow,
                    Box::new(ScalarNode::Const(-2.0)),
                    Box::new(ScalarNode::Const(0.5)),
                )),
            );
            let program = ScalarExpressionProgram::new(expression)
                .unwrap()
                .with_numerical_policy(policy);
            let mut edge_probe = probe.clone();
            edge_probe.push(vec![
                0.0,
                -0.0,
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                -f64::from_bits(1),
            ]);
            let expected = program
                .fit_training(&x, &y, carrier, 0.02)
                .unwrap()
                .predict(&edge_probe)
                .unwrap()
                .0;
            support::run(
                &root.join(format!("scalar-composition-{policy:?}-{carrier:?}")),
                &emit_scalar(&program, carrier, 5, 0.02).unwrap(),
                &serde_json::json!({"train":x,"targets":y,"test":edge_probe}),
                &expected,
                false,
            );
            count += 1;
        }
    }
    let smoothing = RowStage::Savgol(
        feat_math::kernel::savgol::Config::new(2, 58, Axis::FeatureWithinRow, 5, 2, 1, "shrink")
            .unwrap(),
    );
    let frequency = RowStage::Goertzel(
        feat_math::kernel::goertzel::Config::new(2, 61, Axis::FeatureWithinRow, &[0.0, 0.1, 0.5])
            .unwrap(),
    );
    for (index, stages) in [
        vec![],
        vec![RowStage::Savgol(
            feat_math::kernel::savgol::Config::new(
                2,
                58,
                Axis::FeatureWithinRow,
                5,
                2,
                1,
                "shrink",
            )
            .unwrap(),
        )],
        vec![RowStage::Goertzel(
            feat_math::kernel::goertzel::Config::new(
                2,
                61,
                Axis::FeatureWithinRow,
                &[0.0, 0.1, 0.5],
            )
            .unwrap(),
        )],
        vec![smoothing.clone(), frequency.clone()],
        vec![frequency, smoothing],
    ]
    .into_iter()
    .enumerate()
    {
        let program = Program::new(2, 5, stages, 0.02).unwrap();
        let expected = program.fit(&x, &y).unwrap().predict(&probe).unwrap().0;
        let bundle = emit_row_vector(&program).unwrap();
        support::run(
            &root.join(format!("row-{index}")),
            &bundle,
            &serde_json::json!({"train":x,"targets":y,"test":probe}),
            &expected,
            false,
        );
        count += 1;
    }
    assert_eq!(count, (functions.len() + 1) * 2 * 2 + 5);
    eprintln!(
        "CUDA C++ census: {}",
        serde_json::json!({"kind":"v2","total":count,"executed":count,"unaccounted":0})
    );
}
