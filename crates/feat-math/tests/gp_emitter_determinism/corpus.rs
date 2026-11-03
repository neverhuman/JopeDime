fn function(id: u16, argument: ScalarNode) -> ScalarNode {
    let mut args = vec![argument];
    match id {
        7 | 8 | 10 => args.push(ScalarNode::Var(Var::Col(1))),
        9 => args.extend([ScalarNode::Const(-1.0), ScalarNode::Const(1.0)]),
        _ => {}
    }
    ScalarNode::Call(ScalarFunction::from_versioned_id(2, id).unwrap(), args)
}

fn random_tree(rng: &mut Pcg32, depth: usize) -> ScalarNode {
    let choice = rng.next_u32();
    if depth == 0 || choice.is_multiple_of(5) {
        return if choice.is_multiple_of(2) {
            ScalarNode::Var(Var::Col((choice as usize / 5) % WIDTH))
        } else {
            ScalarNode::Const([-0.0, 0.5, -2.0, 1e-300, 1e200][(choice as usize / 5) % 5])
        };
    }
    if choice.is_multiple_of(3) {
        let op =
            [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow][(choice as usize / 3) % 5];
        ScalarNode::Binary(
            op,
            Box::new(random_tree(rng, depth - 1)),
            Box::new(random_tree(rng, depth - 1)),
        )
    } else {
        function((choice % 23) as u16, random_tree(rng, depth - 1))
    }
}

fn corpus() -> Vec<ScalarExpressionProgram> {
    let mut nodes = Vec::new();
    for value in [
        0.0,
        -0.0,
        f64::from_bits(1),
        1e-300,
        1e200,
        f64::MAX,
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        nodes.push(ScalarNode::Const(value));
    }
    for id in 0..23 {
        nodes.push(function(id, ScalarNode::Var(Var::Col(0))));
    }
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow] {
        nodes.push(ScalarNode::Binary(
            op,
            Box::new(ScalarNode::Var(Var::Col(0))),
            Box::new(ScalarNode::Var(Var::Col(1))),
        ));
    }
    for var in [
        Var::RowSum,
        Var::RowMean,
        Var::RowL1,
        Var::RowL2,
        Var::RowMin,
        Var::RowMax,
        Var::RowStd,
    ] {
        nodes.push(ScalarNode::Var(var));
    }
    let mut deep = ScalarNode::Var(Var::Col(0));
    for _ in 0..15 {
        deep = ScalarNode::Unary(UnOp::Neg, Box::new(deep));
    }
    nodes.push(deep);
    nodes.push(function(21, ScalarNode::Var(Var::X)));
    let mut rng = Pcg32::new(SEED, 54);
    for _ in 0..32 {
        nodes.push(random_tree(&mut rng, 4));
    }
    // Retain constant-folding regressions alongside seeded case 58: Min/Max
    // must select the native second operand for both orders of signed-zero ties.
    for id in [7, 8] {
        for zero in [0.0, -0.0] {
            nodes.push(function(id, ScalarNode::Const(zero)));
        }
    }
    // Clip's bound sorting calls the same kernels; retain both zero orders.
    for (a, b) in [(0.0, -0.0), (-0.0, 0.0)] {
        nodes.push(ScalarNode::Call(
            ScalarFunction::from_versioned_id(2, 9).unwrap(),
            vec![
                ScalarNode::Var(Var::Col(0)),
                ScalarNode::Const(a),
                ScalarNode::Const(b),
            ],
        ));
    }
    nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| {
            ScalarExpressionProgram::new(node)
                .unwrap()
                .with_numerical_policy(if i.is_multiple_of(2) {
                    NumericalPolicy::LegacyV1
                } else {
                    NumericalPolicy::StableV2
                })
        })
        .collect()
}

fn carrier(case: usize) -> ScalarCarrier {
    // The explicit x case and alternating random trees exercise elementwise binding.
    if case.is_multiple_of(3) || case == 45 {
        ScalarCarrier::Elementwise
    } else {
        ScalarCarrier::Row
    }
}

fn emit(case: usize, program: &ScalarExpressionProgram, language: ScalarLanguage) -> String {
    program
        .emit_recipe(language, carrier(case), WIDTH, LAMBDA)
        .unwrap()
}

fn run(command: &mut Command) -> String {
    // The shard tests run concurrently, and each C++ case starts both g++ and
    // clang++. Serialize subprocesses so the integration test stays within the
    // runner's process budget without changing the emitted corpus.
    static PROCESS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _process_guard = PROCESS_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("{command:?}: {e}"));
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}
