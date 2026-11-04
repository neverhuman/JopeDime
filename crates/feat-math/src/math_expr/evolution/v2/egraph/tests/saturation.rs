#[test]
fn saturation_is_bounded_deterministic_and_independent_of_rule_order() {
    let text = "((((-(c0))*(-(c1)))*1.0)-(-((c2/(-1.0)))))+((-(-(c0)))/0.5)";
    let original = MathExpr::parse(text).unwrap();
    let first = simplify(&original);
    let again = simplify(&original);
    assert_eq!(
        first.expression.to_source().unwrap(),
        again.expression.to_source().unwrap()
    );
    assert_eq!(
        (first.enodes, first.applications, first.iterations),
        (again.enodes, again.applications, again.iterations)
    );
    // Rule-order independence is claimed for a completed saturation, which this is.
    let limits = Limits::for_input(count_nodes(&original.root));
    assert_eq!(first.stop, Saturation::Saturated);
    let mut reversed = RULES.to_vec();
    reversed.reverse();
    let permuted = simplify_with(&original, &reversed, limits);
    assert_eq!(permuted.stop, Saturation::Saturated);
    assert_eq!(
        permuted.expression.to_source().unwrap(),
        first.expression.to_source().unwrap()
    );
    assert!(first.changed);
    assert_exact(&original, &first.expression);
    assert!(
        first.enodes <= limits.enodes,
        "{} > {}",
        first.enodes,
        limits.enodes
    );
    // A long chain of x - y rewrites saturates well inside its cap.
    let wide = "((((c0-c1)-c2)-(c0-c2))-((c1-c0)-(c2-c1)))-(((c0-c1)-(c1-c2))-((c2-c0)-(c0-c1)))";
    let wide = MathExpr::parse(wide).unwrap();
    let result = simplify(&wide);
    assert_eq!(result.stop, Saturation::Saturated);
    assert!(result.enodes <= Limits::for_input(count_nodes(&wide.root)).enodes);
    assert_exact(&wide, &result.expression);
}

#[test]
fn hard_limits_stop_saturation_with_an_exact_rebuilt_result() {
    let original =
        MathExpr::parse("((((-(c0))*(-(c1)))*1.0)-(-((c2/(-1.0)))))+((-(-(c0)))/0.5)").unwrap();
    let full = simplify(&original);
    assert_eq!(full.stop, Saturation::Saturated);
    // A tight e-node cap stops the second round part way; the rebuilt graph still extracts
    // an exact tree, never larger than the input.
    let capped = simplify_with(
        &original,
        &RULES,
        Limits {
            iterations: MAX_ITERATIONS,
            enodes: 29,
        },
    );
    assert_eq!(capped.stop, Saturation::NodeLimit, "{capped:?}");
    assert!(capped.enodes <= 29, "{capped:?}");
    assert!(capped.applications < full.applications, "{capped:?}");
    assert!(count_nodes(&capped.expression.root) <= count_nodes(&original.root));
    assert_exact(&original, &capped.expression);
    // One round of a longer saturation.
    let one = simplify_with(
        &original,
        &RULES,
        Limits {
            iterations: 1,
            enodes: NODE_CAP_MAX,
        },
    );
    assert_eq!((one.stop, one.iterations), (Saturation::IterationLimit, 1));
    assert_exact(&original, &one.expression);
    // Both limited runs are deterministic.
    let again = simplify_with(
        &original,
        &RULES,
        Limits {
            iterations: MAX_ITERATIONS,
            enodes: 29,
        },
    );
    assert_eq!(
        (again.enodes, again.applications, again.iterations),
        (capped.enodes, capped.applications, capped.iterations)
    );
}

#[test]
fn signaling_nan_rows_are_outside_the_exact_domain() {
    let snan = f64::from_bits(0x7ff0_0000_0000_0001);
    assert!(is_signaling(snan));
    for input in [
        "((c0*1.0)^0.0)",
        "((c0/1.0)^0.0)",
        "((c0-0.0)^0.0)",
        "((c0+(-0.0))^0.0)",
        "((c0*(-1.0))^0.0)",
        "((c0/(-1.0))^0.0)",
        "(1.0^(c0*1.0))",
    ] {
        let original = MathExpr::parse(input).unwrap();
        let simplified = simplify(&original).expression;
        assert_ne!(simplified, original, "{input}: the rule fires");
        let (a, b) = (original.compile().unwrap(), simplified.compile().unwrap());
        // A quiet NaN stays exact.
        let (qa, qb) = (
            a.eval_row_traced(&[f64::NAN]),
            b.eval_row_traced(&[f64::NAN]),
        );
        assert_eq!((qa.0.to_bits(), qa.1), (qb.0.to_bits(), qb.1), "{input}");
        // A signaling NaN: the deleted step quieted it, and pow tells the two apart.
        let (sa, sb) = (a.eval_row_traced(&[snan]), b.eval_row_traced(&[snan]));
        assert_eq!(
            (sa.0.to_bits(), sa.1),
            (1.0f64.to_bits(), FaultSet::NONFINITE_INPUT),
            "{input}"
        );
        assert_eq!(
            (sb.0.to_bits(), sb.1),
            (
                0.0f64.to_bits(),
                FaultSet::NONFINITE_INPUT.union(FaultSet::NONFINITE_SANITIZED)
            ),
            "{input}"
        );
    }
}

/// Random trees over every operator and the selector, clamp and exponential functions.
fn random_tree_wide(rng: &mut Pcg32, depth: usize) -> Expr {
    let leaf = |rng: &mut Pcg32| -> Expr {
        match rng.next_u32() % 5 {
            0 | 1 => Expr::Var(Var::Col(rng.next_u32() as usize % 3)),
            _ => literal(
                [
                    0.0,
                    -0.0,
                    1.0,
                    -1.0,
                    2.0,
                    0.5,
                    -0.25,
                    3.0,
                    9.0,
                    2f64.powi(-40),
                ][rng.next_u32() as usize % 10],
            ),
        }
    };
    if depth == 0 || rng.next_u32().is_multiple_of(5) {
        return leaf(rng);
    }
    let child = |rng: &mut Pcg32| random_tree_wide(rng, depth - 1);
    match rng.next_u32() % 9 {
        0 => Expr::Unary(UnOp::Neg, Box::new(child(rng))),
        1 => Expr::Call(Func::Abs, vec![child(rng)]),
        2 => {
            let func = [Func::Exp, Func::Sigmoid, Func::Step, Func::Sin, Func::Log]
                [rng.next_u32() as usize % 5];
            let arity = (1..=3)
                .find(|n| validate_arity(func, *n).is_ok())
                .expect("an arity");
            Expr::Call(func, (0..arity).map(|_| child(rng)).collect())
        }
        3 => Expr::Call(
            [Func::Min, Func::Max][rng.next_u32() as usize % 2],
            vec![child(rng), child(rng)],
        ),
        4 => Expr::Call(Func::Clip, vec![child(rng), child(rng), child(rng)]),
        5 => Expr::Binary(BinOp::Pow, Box::new(child(rng)), Box::new(child(rng))),
        _ => Expr::Binary(
            [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div][rng.next_u32() as usize % 4],
            Box::new(child(rng)),
            Box::new(child(rng)),
        ),
    }
}

#[test]
fn random_trees_over_every_operation_simplify_exactly_on_the_domain() {
    let mut rng = Pcg32::new(0xB20, 3);
    let (mut changed, mut checked) = (0usize, 0usize);
    for _ in 0..300 {
        let Ok(original) = MathExpr::from_ast(random_tree_wide(&mut rng, 4)) else {
            continue;
        };
        let result = simplify(&original);
        assert!(count_nodes(&result.expression.root) <= count_nodes(&original.root));
        assert_exact_quiet(&original, &result.expression);
        changed += usize::from(result.changed);
        checked += 1;
    }
    assert!(
        checked > 200 && changed > 30,
        "{checked} checked, {changed} changed"
    );
}

#[test]
fn malformed_calls_are_returned_unchanged() {
    // serde decoding builds these: every row is a compile failure, which no rule may turn into
    // a value, and no rule may index a missing argument.
    let col = |i| Expr::Var(Var::Col(i));
    let neg = |e| Expr::Unary(UnOp::Neg, Box::new(e));
    for root in [
        Expr::Call(Func::Abs, vec![]),
        Expr::Call(Func::Abs, vec![neg(col(0)), col(1)]),
        Expr::Call(Func::Abs, vec![Expr::Call(Func::Abs, vec![col(0)]), col(1)]),
        Expr::Call(Func::Abs, vec![literal(-2.5), col(1)]),
        Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Call(Func::Abs, vec![])),
            Box::new(Expr::Const(1.0)),
        ),
    ] {
        let original = MathExpr { root };
        assert!(original.compile().is_err());
        let result = simplify(&original);
        assert_eq!(
            (result.changed, result.stop, result.iterations),
            (false, Saturation::Malformed, 0)
        );
        assert_eq!(result.expression, original);
    }
    // The abs arms refuse a wrong arity on their own, under every rule.
    let mut eg = EGraph::default();
    let c0 = eg.add(Node::Var(Var::Col(0)));
    let c1 = eg.add(Node::Var(Var::Col(1)));
    let negated = eg.add(Node::Neg(c0));
    let absolute = eg.add(Node::Call(Func::Abs, vec![c0]));
    let constant = eg.insert(&Term::Literal(-2.5));
    for args in [
        vec![],
        vec![negated, c1],
        vec![absolute, c1],
        vec![constant, c1],
    ] {
        let node = Node::Call(Func::Abs, args);
        eg.add(node.clone());
        for rule in RULES {
            assert!(matches(&eg, rule, &node).is_empty(), "{rule:?} on {node:?}");
        }
    }
}
