use crate::math_expr::evaluator::compute_expr_with;

/// The raw evaluation of `tree` on `row`, before the root's `safe`: the value's bits with
/// every NaN collapsed to one (`None` for an error), and the faults.
fn raw(tree: &Expr, row: &[f64]) -> (Option<u64>, FaultSet) {
    let mut sink = crate::math_expr::fault::Collect::default();
    let value = compute_expr_with(tree, &EvalContext { x: None, row }, &mut sink);
    let bits = value.ok().map(|v| {
        if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        }
    });
    (bits, sink.0)
}

/// Every node of every class, built over its children's picks, evaluates on `rows` exactly
/// like the class's own pick, so every union is exact, not only the one extraction shows. The
/// pick also ranks no higher than any of them.
fn assert_every_union_exact(run: &Run, rows: &[Vec<f64>]) {
    let best = extract(&run.eg);
    for id in run.eg.roots() {
        let pick = &best[&id];
        let expected: Vec<_> = rows.iter().map(|row| raw(pick, row)).collect();
        for node in &run.eg.classes[id] {
            let tree = build(&run.eg, &best, node).expect("every class has a pick");
            assert!(rank(pick) <= rank(&tree), "{pick:?} over {tree:?}");
            for (row, expected) in rows.iter().zip(&expected) {
                assert_eq!(
                    &raw(&tree, row),
                    expected,
                    "{tree:?} vs {pick:?} on {row:?}"
                );
            }
        }
    }
}

#[test]
fn every_union_is_exact_not_only_the_extracted_root() {
    // The two random-tree tests' own trees: the wide ones on the documented domain.
    let quiet: Vec<Vec<f64>> = rows()
        .into_iter()
        .filter(|row| !row.iter().copied().any(is_signaling))
        .collect();
    let all = rows();
    for (stream, wide) in [(2, false), (3, true)] {
        let mut rng = Pcg32::new(0xB20, stream);
        let mut checked = 0usize;
        for _ in 0..300 {
            let tree = if wide {
                random_tree_wide(&mut rng, 4)
            } else {
                random_tree(&mut rng, 4)
            };
            let Ok(original) = MathExpr::from_ast(tree) else {
                continue;
            };
            let limits = Limits::for_input(count_nodes(&original.root));
            let run = saturate(&original, &RULES, limits);
            assert_every_union_exact(&run, if wide { &quiet } else { &all });
            checked += 1;
        }
        assert!(checked > 200, "{checked} checked");
    }
}

#[test]
fn the_enode_cap_holds_at_every_application() {
    let original =
        MathExpr::parse("((((-(c0))*(-(c1)))*1.0)-(-((c2/(-1.0)))))+((-(-(c0)))/0.5)").unwrap();
    let capped = simplify_with(
        &original,
        &RULES,
        Limits {
            iterations: MAX_ITERATIONS,
            enodes: 29,
        },
    );
    assert_eq!(capped.stop, Saturation::NodeLimit, "{capped:?}");
    assert!(capped.peak <= 29, "{capped:?}");
    // Every cap from the input's own 16 e-nodes up.
    for cap in 16..=48 {
        let limits = Limits {
            iterations: MAX_ITERATIONS,
            enodes: cap,
        };
        let result = simplify_with(&original, &RULES, limits);
        assert!(result.peak <= cap, "{result:?}");
        assert_exact(&original, &result.expression);
    }
    // Random trees, under the production limits and under caps a few e-nodes over the input.
    let mut rng = Pcg32::new(0xB20, 2);
    let mut stopped = 0usize;
    for _ in 0..300 {
        let Ok(original) = MathExpr::from_ast(random_tree(&mut rng, 4)) else {
            continue;
        };
        let nodes = count_nodes(&original.root);
        let tight = |extra| Limits {
            iterations: MAX_ITERATIONS,
            enodes: nodes + extra,
        };
        for limits in [Limits::for_input(nodes), tight(3), tight(6)] {
            let result = simplify_with(&original, &RULES, limits);
            assert!(result.peak <= limits.enodes, "{result:?}");
            assert_exact(&original, &result.expression);
            stopped += usize::from(result.stop == Saturation::NodeLimit);
        }
    }
    assert!(stopped > 50, "{stopped} stopped at the cap");
}

/// `eg` with its live classes renumbered: the `i`-th live class becomes `ids[i]`.
fn relabel(eg: &EGraph, ids: &[Id]) -> (EGraph, BTreeMap<Id, Id>) {
    let roots = eg.roots();
    let map: BTreeMap<Id, Id> = roots.iter().copied().zip(ids.iter().copied()).collect();
    let to = |id: &Id| map[&eg.find(*id)];
    let mut out = EGraph {
        parent: (0..roots.len()).collect(),
        classes: vec![BTreeSet::new(); roots.len()],
        memo: BTreeMap::new(),
    };
    for id in &roots {
        for node in &eg.classes[*id] {
            let node = match node {
                Node::Neg(a) => Node::Neg(to(a)),
                Node::Bin(t, a, b) => Node::Bin(*t, to(a), to(b)),
                Node::Call(f, args) => Node::Call(*f, args.iter().map(to).collect()),
                leaf => leaf.clone(),
            };
            out.memo.insert(node.clone(), to(id));
            out.classes[to(id)].insert(node);
        }
    }
    (out, map)
}

/// Over `relabellings` random renumberings of `original`'s saturated graph, extraction picks
/// the same tree for every class as on the real numbering.
fn assert_numbering_free(original: &MathExpr, relabellings: usize, rng: &mut Pcg32) {
    let limits = Limits::for_input(count_nodes(&original.root));
    let run = saturate(original, &RULES, limits);
    let best = extract(&run.eg);
    let mut ids: Vec<Id> = (0..run.eg.roots().len()).collect();
    for _ in 0..relabellings {
        for i in (1..ids.len()).rev() {
            ids.swap(i, rng.next_u32() as usize % (i + 1));
        }
        let (relabelled, map) = relabel(&run.eg, &ids);
        let other = extract(&relabelled);
        for (id, new) in &map {
            assert_eq!(rank(&best[id]), rank(&other[new]), "{original:?}");
        }
    }
}

#[test]
fn extraction_does_not_depend_on_class_numbering() {
    // In this input's root class, ((0.75*(-(c2+8.0)))+...) has the smaller Debug form, but
    // the product's class picks its shallower (-0.75)*(c2+8.0), whatever the numbering. An
    // in-place fixed point kept the root's stale pick on about 1 renumbering in 150.
    let pinned = "((-(-0.75))*(-(c2+8.0)))+sin(sin(sin(sin(c0))))";
    assert_eq!(
        source(pinned),
        printed("(((-0.75)*(c2+8.0))+sin(sin(sin(sin(c0)))))")
    );
    let mut rng = Pcg32::new(0xB20, 4);
    assert_numbering_free(&MathExpr::parse(pinned).unwrap(), 1024, &mut rng);
    let mut checked = 0usize;
    while checked < 100 {
        if let Ok(input) = MathExpr::from_ast(random_tree_wide(&mut rng, 4)) {
            assert_numbering_free(&input, 32, &mut rng);
            checked += 1;
        }
    }
}

#[test]
fn replacement_and_depth_tie_break_are_pinned() {
    for (input, expected) in [
        // x*2 and x + x tie on nodes and depth, so the input stays.
        ("(c0*2.0)", "(c0*2.0)"),
        // As small and shallower: 13 nodes at depth 6 become 13 nodes at depth 5.
        (
            "max((-(c0*sin(sin(c1)))),((-c0)*sin(sin(c1))))",
            "max(((-c0)*sin(sin(c1))),((-c0)*sin(sin(c1))))",
        ),
        // Depth, not the Debug form, settles (-0.75)*(c2+8.0) against 0.75*(-(c2+8.0)).
        ("((-(-0.75))*(-(c2+8.0)))", "((-0.75)*(c2+8.0))"),
    ] {
        let original = MathExpr::parse(input).unwrap();
        assert_eq!(source(input), printed(expected), "{input}");
        assert_exact(&original, &simplify(&original).expression);
    }
}
