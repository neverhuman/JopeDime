#[test]
fn each_rule_simplifies_its_pattern_exactly() {
    let cases = [
        ("(c0*1.0)", "c0"),               // R1
        ("(1.0*c0)", "c0"),               // R1
        ("(c0/1.0)", "c0"),               // R2
        ("(c0+(-0.0))", "c0"),            // R3: x + (-0) = x, including x = ±0
        ("((-0.0)+c0)", "c0"),            // R3
        ("(c0-0.0)", "c0"),               // R3: x - (+0) = x
        ("(-(-(c0)))", "c0"),             // R4: sign-bit flips
        ("abs((-(c0)))", "abs(c0)"),      // R5
        ("abs(abs(c0))", "abs(c0)"),      // R5
        ("(c0-(-(c1)))", "(c0+c1)"),      // R7: a - (-b) = a + b
        ("(c0+(-(c1)))", "(c0-c1)"),      // R7: a + (-b) = a - b
        ("((-(c0))*(-(c1)))", "(c0*c1)"), // R8: RNE is sign-symmetric
        ("((-(c0))*c1)", "(-(c0*c1))"),   // R8: the same node count, shallower? no: kept
        ("(c0*(-1.0))", "(-c0)"),         // R10
        ("(c0/(-1.0))", "(-c0)"),         // R10
        ("((2.0*3.0)+c0)", "(6.0+c0)"),   // folding
        ("(c0+(1.5-1.5))", "(c0+0.0)"),   // folding to +0.0; x + 0.0 itself stays
        ("abs((-2.5))*c0", "(2.5*c0)"),   // folding abs of a literal
        ("((c0*1.0)/1.0)+(-0.0)", "c0"),  // chains
    ];
    for (input, expected) in cases {
        let original = MathExpr::parse(input).unwrap();
        let result = simplify(&original);
        let expected_source = if expected == "(-(c0*c1))" {
            // (-a)*b and -(a*b) both take five nodes at depth 3; the original stays.
            printed(input)
        } else {
            printed(expected)
        };
        assert_eq!(
            result.expression.to_source().unwrap(),
            expected_source,
            "{input}"
        );
        assert_exact(&original, &result.expression);
    }
}

#[test]
fn power_of_two_division_joins_the_reciprocal_product() {
    // R11 keeps the node count, so extraction keeps the original, but both are one class.
    for (c, reciprocal) in [(0.25, 4.0), (-8.0, -0.125), (2f64.powi(-39), 2f64.powi(39))] {
        let mut eg = EGraph::default();
        let x = eg.add(Node::Var(Var::Col(0)));
        let c_id = eg.insert(&Term::Literal(c));
        let division = eg.add(Node::Bin(DIV, x, c_id));
        let r_id = eg.insert(&Term::Literal(reciprocal));
        let product = eg.add(Node::Bin(MUL, x, r_id));
        for rhs in matches(&eg, Rule::DivPowerOfTwo, &Node::Bin(DIV, x, c_id)) {
            let new = eg.insert(&rhs);
            eg.union(division, new);
        }
        eg.rebuild();
        assert_eq!(eg.find(division), eg.find(product), "{c}");
        let original = MathExpr::from_ast(Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Var(Var::Col(0))),
            Box::new(literal(c)),
        ))
        .unwrap();
        let rewritten = MathExpr::from_ast(Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Var(Var::Col(0))),
            Box::new(literal(reciprocal)),
        ))
        .unwrap();
        assert_exact(&original, &rewritten);
    }
    // 2^-40 is under the protected threshold, 3 is not a power of two, and 2^-1074 has no
    // finite reciprocal: none is rewritten.
    for c in [2f64.powi(-40), 3.0, 5e-324, 1.0, -1.0] {
        assert!(exact_reciprocal(c).is_none_or(|r| r.abs() == 1.0), "{c}");
    }
}

#[test]
fn excluded_rewrites_are_wrong_and_never_applied() {
    let differs = |a: &str, b: &str, row: &[f64]| {
        let (va, fa) = MathExpr::parse(a).unwrap().eval_row_traced(row);
        let (vb, fb) = MathExpr::parse(b).unwrap().eval_row_traced(row);
        (va.to_bits(), fa) != (vb.to_bits(), fb)
    };
    // x + 0.0 and x - (-0.0) turn -0.0 into +0.0.
    assert!(differs("(c0+0.0)", "c0", &[-0.0]));
    assert!(differs("(c0-(-0.0))", "c0", &[-0.0]));
    // x*0 keeps the sign (and inf*0 is NaN, sanitized with a fault).
    assert!(differs("(c0*0.0)", "0.0", &[-1.0]));
    assert!(differs("(c0*0.0)", "0.0", &[f64::INFINITY]));
    // x - x and x / x at infinity and zero.
    assert!(differs("(c0-c0)", "0.0", &[f64::INFINITY]));
    assert!(differs("(c0/c0)", "1.0", &[0.0]));
    // Reassociation: 2^53 + 1 rounds to 2^53.
    assert!(differs(
        "((c0+c1)+c2)",
        "(c0+(c1+c2))",
        &[2f64.powi(53), 1.0, -(2f64.powi(53))]
    ));
    // Distribution: 3 (1 + 2^-53) rounds differently from 3 + 3·2^-53.
    assert!(differs(
        "(c0*(c1+c2))",
        "((c0*c1)+(c0*c2))",
        &[3.0, 1.0, 2f64.powi(-53)]
    ));
    // a - b versus -(b - a) at a = b.
    assert!(differs("(c0-c1)", "(-(c1-c0))", &[1.0, 1.0]));
    // The protected branch returns +0.0 for (-a)/b, but -(a/b) is -0.0.
    assert!(differs("((-(c0))/c1)", "(-(c0/c1))", &[1.0, 1e-13]));
    // Commutation changes which operand's faults precede a Hard fault (column out of range).
    assert!(differs("(log(c0)+c9)", "(c9+log(c0))", &[-1.0]));
    // No exact rule reaches the wrong target. x - (-0.0) may become x + 0.0, which IEEE
    // defines as the same operation, but never x.
    for (text, forbidden) in [
        ("(c0+0.0)", "c0"),
        ("(c0-(-0.0))", "c0"),
        ("(c0*0.0)", "0.0"),
        ("(c0-c0)", "0.0"),
        ("(c0/c0)", "1.0"),
    ] {
        let original = MathExpr::parse(text).unwrap();
        let result = simplify(&original);
        assert_ne!(
            result.expression.to_source().unwrap(),
            printed(forbidden),
            "{text}"
        );
        assert_exact(&original, &result.expression);
    }
    assert_eq!(source("(c0-(-0.0))"), printed("(c0+0.0)"));
    // Where no exact rule applies at all, the tree is unchanged.
    for text in [
        "(c0+0.0)",
        "(c0*0.0)",
        "(c0-c0)",
        "(c0/c0)",
        "((c0+c1)+c2)",
        "(c0*(c1+c2))",
        "((-(c0))/c1)",
        "(c0^2.0)",
        "min(c0,c1)",
        "exp(log(c0))",
        "(c0/(2.0^(-40.0)))",
    ] {
        assert_eq!(source(text), printed(text), "{text}");
    }
}

/// Lane D pre-review of B20: three exclusions had no counterexample, and the 2^-40 divisor
/// above is a power, so no rule could fold it into a division by a literal anyway.
#[test]
fn the_remaining_exclusions_have_counterexamples() {
    let differs = |a: &str, b: &str, row: &[f64]| {
        let (va, fa) = MathExpr::parse(a).unwrap().eval_row_traced(row);
        let (vb, fb) = MathExpr::parse(b).unwrap().eval_row_traced(row);
        (va.to_bits(), fa) != (vb.to_bits(), fb)
    };
    // x^2 versus x*x: at overflow pow raises POW_NONFINITE, which * does not.
    assert!(differs("(c0^2.0)", "(c0*c0)", &[1e200]));
    // exp(log(x)) versus x: the protected log of a negative raises LOG_DOMAIN.
    assert!(differs("exp(log(c0))", "c0", &[-1.0]));
    // x / 2^-40 versus x * 2^40: the divisor is under the protected division's threshold.
    assert_eq!(format!("{:?}", 2f64.powi(-40)), "9.094947017729282e-13");
    let literal = "(c0/9.094947017729282e-13)";
    assert!(differs(literal, "(c0*1099511627776.0)", &[3.0]));
    assert_eq!(source(literal), printed(literal));
}

/// Random trees biased toward the rules' patterns, every one simplified exactly.
fn random_tree(rng: &mut Pcg32, depth: usize) -> Expr {
    let leaf = |rng: &mut Pcg32| -> Expr {
        match rng.next_u32() % 5 {
            0 | 1 => Expr::Var(Var::Col(rng.next_u32() as usize % 3)),
            _ => literal(
                [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.25, 3.0, 2f64.powi(-40)]
                    [rng.next_u32() as usize % 9],
            ),
        }
    };
    if depth == 0 || rng.next_u32().is_multiple_of(5) {
        return leaf(rng);
    }
    match rng.next_u32() % 6 {
        0 => Expr::Unary(UnOp::Neg, Box::new(random_tree(rng, depth - 1))),
        1 => Expr::Call(Func::Abs, vec![random_tree(rng, depth - 1)]),
        2 => Expr::Call(
            [Func::Sin, Func::Sqrt, Func::Log, Func::Tanh][rng.next_u32() as usize % 4],
            vec![random_tree(rng, depth - 1)],
        ),
        _ => Expr::Binary(
            [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div][rng.next_u32() as usize % 4],
            Box::new(random_tree(rng, depth - 1)),
            Box::new(random_tree(rng, depth - 1)),
        ),
    }
}

#[test]
fn random_trees_simplify_exactly_and_never_grow() {
    let mut rng = Pcg32::new(0xB20, 2);
    let (mut changed, mut checked) = (0usize, 0usize);
    for _ in 0..300 {
        let Ok(original) = MathExpr::from_ast(random_tree(&mut rng, 4)) else {
            continue;
        };
        let result = simplify(&original);
        assert!(count_nodes(&result.expression.root) <= count_nodes(&original.root));
        assert!(result.enodes <= NODE_CAP_MAX);
        assert_exact(&original, &result.expression);
        changed += usize::from(result.changed);
        checked += 1;
    }
    assert!(
        checked > 250 && changed > 50,
        "{checked} checked, {changed} changed"
    );
}
