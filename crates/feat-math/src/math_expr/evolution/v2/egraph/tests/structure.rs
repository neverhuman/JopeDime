#[test]
fn rebuild_merges_congruent_classes() {
    // c0*1.0 joins c0, so abs((c0*1.0)) and abs(c0) become congruent and share one class.
    let original = MathExpr::parse("(abs((c0*1.0))-abs(c0))").unwrap();
    let limits = Limits::for_input(count_nodes(&original.root));
    let run = saturate(&original, &RULES, limits);
    assert_eq!(run.stop, Saturation::Saturated);
    let holding = run
        .eg
        .roots()
        .into_iter()
        .filter(|&id| run.eg.holds_abs(id))
        .count();
    assert_eq!(holding, 1);
}

#[test]
fn saturated_means_the_last_round_changed_nothing() {
    // Its second round adds two e-nodes and merges two classes, and the rebuild collapses two
    // nodes, so the e-node and class counts come back as they were: comparing counts called it
    // saturated after round 2. Round 3 still changes the graph; round 4 changes nothing.
    let pinned = MathExpr::parse("(-(-(((-2.0)-(-1.0))*1.0)))").unwrap();
    let mut inputs = vec![pinned];
    let mut rng = Pcg32::new(0xB20, 5);
    while inputs.len() < 200 {
        if let Ok(input) = MathExpr::from_ast(random_tree_wide(&mut rng, 4)) {
            inputs.push(input);
        }
    }
    for (i, original) in inputs.iter().enumerate() {
        let limits = Limits::for_input(count_nodes(&original.root));
        let run = saturate(original, &RULES, limits);
        if i == 0 {
            assert_eq!((run.stop, run.iterations), (Saturation::Saturated, 4));
        }
        if run.stop != Saturation::Saturated {
            continue;
        }
        // One round fewer already reached the same graph.
        let fewer = Limits {
            iterations: run.iterations - 1,
            ..limits
        };
        let before = saturate(original, &RULES, fewer);
        assert_eq!(
            (&before.eg.parent, &before.eg.memo),
            (&run.eg.parent, &run.eg.memo),
            "{original:?}"
        );
    }
}

/// `root` as serde decodes it: derived `Deserialize` checks no node or depth limit and no
/// call's arity.
fn decoded(root: Expr) -> MathExpr {
    serde_json::from_value(serde_json::json!({ "root": root })).unwrap()
}

fn column(i: usize) -> Expr {
    Expr::Var(Var::Col(i))
}

/// `n` negations over `tree`.
fn negated(n: usize, tree: Expr) -> Expr {
    (0..n).fold(tree, |tree, _| Expr::Unary(UnOp::Neg, Box::new(tree)))
}

/// The complete tree of `-` over c0 with `levels` levels under its root: 2^(levels+1) - 1
/// nodes at depth levels + 1.
fn complete(levels: usize) -> Expr {
    if levels == 0 {
        return column(0);
    }
    Expr::Binary(
        BinOp::Sub,
        Box::new(complete(levels - 1)),
        Box::new(complete(levels - 1)),
    )
}

/// `input` fails the structure check `from_ast` runs, and both entry points refuse it before
/// saturation and hand back the input itself: the refusal holds no copy of it.
fn assert_refused(input: &MathExpr) {
    assert!(MathExpr::from_ast(input.root.clone()).is_err());
    let limits = Limits {
        iterations: MAX_ITERATIONS,
        enodes: NODE_CAP_MAX,
    };
    for result in [simplify(input), simplify_with(input, &RULES, limits)] {
        assert_eq!(
            (result.changed, result.stop, result.iterations),
            (false, Saturation::Malformed, 0)
        );
        assert_eq!((result.enodes, result.applications, result.peak), (0, 0, 0));
        let output: &MathExpr = &result.expression;
        assert!(std::ptr::eq(output, input), "the refusal copied its input");
    }
}

/// Independent review 04: serde decoding checks no arity, so a wrong one is refused wherever
/// it is, before anything counts, clones or recurses into the input.
#[test]
fn decoded_wrong_arities_are_refused_uncopied() {
    // The reviewer's shallow 65-node tree (independent-review-04/malformed-65-nodes.json).
    let leaves = vec![serde_json::json!({ "Const": 1.0 }); 64];
    let reviewer: MathExpr =
        serde_json::from_value(serde_json::json!({ "root": { "Call": ["Abs", leaves] } })).unwrap();
    assert_eq!(count_nodes(&reviewer.root), 65);
    assert!(reviewer.check_structure(MAX_NODES, MAX_DEPTH).is_err());
    assert_refused(&reviewer);
    // At the root.
    for (func, arity) in [
        (Func::Abs, 2),
        (Func::Sin, 0),
        (Func::Max, 1),
        (Func::Step, 3),
        (Func::Clip, 2),
    ] {
        assert_refused(&decoded(Expr::Call(func, (0..arity).map(column).collect())));
    }
    // Deep inside: at depth 15 under fourteen negations (with the right arity it is
    // admitted), and in a sum's right branch.
    assert_refused(&decoded(negated(
        14,
        Expr::Call(Func::Max, vec![column(0)]),
    )));
    let two_arguments = negated(14, Expr::Call(Func::Max, vec![column(0), column(1)]));
    assert!(MathExpr::from_ast(two_arguments).is_ok());
    assert_refused(&decoded(Expr::Binary(
        BinOp::Mul,
        Box::new(column(0)),
        Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(column(1)),
            Box::new(Expr::Call(Func::Clip, vec![column(0), column(1)])),
        )),
    )));
}

/// Independent review 04: one node or one level over the limits is refused.
#[test]
fn decoded_inputs_over_the_limits_are_refused_uncopied() {
    // 65 nodes, every arity right, at depth 8.
    let nodes = decoded(negated(2, complete(5)));
    assert_eq!((count_nodes(&nodes.root), depth(&nodes.root)), (65, 8));
    assert_refused(&nodes);
    // Depth 17: sixteen negations over c0.
    let deep = decoded(negated(16, column(0)));
    assert_eq!((count_nodes(&deep.root), depth(&deep.root)), (17, 17));
    assert_refused(&deep);
}

/// Independent review 04: the structure check stops at the first violation, so refusing a
/// wide input costs no more than checking one at the limits. These are ordinary sizes.
#[test]
fn wide_decoded_inputs_are_refused_uncopied() {
    // 10,000 leaves under one call: at the root, and under a sum.
    let leaves = || (0..10_000).map(|i| column(i % 8)).collect::<Vec<_>>();
    assert_refused(&decoded(Expr::Call(Func::Min, leaves())));
    assert_refused(&decoded(Expr::Binary(
        BinOp::Add,
        Box::new(column(0)),
        Box::new(Expr::Call(Func::Abs, leaves())),
    )));
    // Every arity right, but 16,383 nodes at depth 14: the check stops at the 65th node.
    let wide = decoded(complete(13));
    assert_eq!((count_nodes(&wide.root), depth(&wide.root)), (16_383, 14));
    assert_refused(&wide);
}

/// Independent review 04: inputs exactly at the limits are admitted, and simplified as they
/// were before the structure check moved first.
#[test]
fn decoded_inputs_at_the_limits_are_admitted_as_before() {
    let sub = |a, b| Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b));
    // 64 nodes at depth 16: c0 under fourteen negations, plus the negated difference of two
    // complete trees. The negations cancel and the sum becomes a difference.
    let both = Expr::Binary(
        BinOp::Add,
        Box::new(negated(14, column(0))),
        Box::new(negated(1, sub(complete(4), complete(3)))),
    );
    let both_simplified = sub(column(0), sub(complete(4), complete(3)));
    // 64 nodes at depth 7, which nothing shrinks, and 16 nodes at depth 16.
    let nodes = negated(1, complete(5));
    let deep = negated(15, column(0));
    for (input, size, expected, changed) in [
        (both, (64, 16), both_simplified, true),
        (nodes.clone(), (64, 7), nodes, false),
        (deep, (16, 16), negated(1, column(0)), true),
    ] {
        let input = decoded(input);
        assert_eq!((count_nodes(&input.root), depth(&input.root)), size);
        assert!(MathExpr::from_ast(input.root.clone()).is_ok());
        let result = simplify(&input);
        assert_eq!(result.expression, MathExpr::from_ast(expected).unwrap());
        assert_eq!(
            (result.changed, result.stop),
            (changed, Saturation::Saturated)
        );
        assert!(
            result.peak <= Limits::for_input(size.0).enodes,
            "{result:?}"
        );
        assert_exact(&input, &result.expression);
    }
}

/// Independent review 04: folding a libm-backed function is excluded, because libm results
/// are not portable, and no test pinned it. A literal argument, given or folded, leaves the
/// call in place, and the call's class never holds a literal.
#[test]
fn libm_calls_are_never_folded() {
    for (input, expected) in [
        ("sin(0.5)", "sin(0.5)"),
        ("log1p(0.5)", "log1p(0.5)"),
        ("cos((2.0*3.0))", "cos(6.0)"),
    ] {
        let original = MathExpr::parse(input).unwrap();
        let limits = Limits::for_input(count_nodes(&original.root));
        let run = saturate(&original, &RULES, limits);
        assert_eq!(run.stop, Saturation::Saturated, "{input}");
        assert!(run.eg.literal(run.root).is_none(), "{input}");
        assert_eq!(source(input), printed(expected), "{input}");
    }
}

/// Portable expression shape excludes every libm-backed literal call, including Tanh.
/// Constant arithmetic may fold the argument; it must never fold the enclosing call.
#[test]
fn all_libm_calls_preserve_literal_shape_including_tanh() {
    let functions = [
        Func::Sqrt,
        Func::Log1p,
        Func::Sin,
        Func::Cos,
        Func::Tanh,
        Func::Sigmoid,
        Func::Exp,
        Func::Log,
        Func::Asinh,
        Func::Atan,
        Func::Erf,
    ];
    for function in functions {
        for argument in [0.0, -0.0, 0.5, -0.5, 6.0] {
            for arithmetic in [false, true] {
                let child = if arithmetic {
                    Expr::Binary(
                        BinOp::Mul,
                        Box::new(Expr::Const(argument)),
                        Box::new(Expr::Const(1.0)),
                    )
                } else {
                    Expr::Const(argument)
                };
                let input = MathExpr::from_ast(Expr::Call(function, vec![child])).unwrap();
                let expected =
                    MathExpr::from_ast(Expr::Call(function, vec![Expr::Const(argument)])).unwrap();
                let result = simplify(&input);
                assert_eq!(result.stop, Saturation::Saturated, "{input:?}");
                assert_eq!(result.expression.as_ref(), &expected, "{input:?}");
                assert_exact(&input, &result.expression);
                let run = saturate(&input, &RULES, Limits::for_input(count_nodes(&input.root)));
                assert!(run.eg.literal(run.root).is_none(), "{input:?}");
            }
        }
    }
}
