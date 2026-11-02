use super::*;

// ---------------------------------------------------------------------------------------------
// The search profile
// ---------------------------------------------------------------------------------------------

#[test]
fn the_search_profile_is_fifteen_nodes_and_depth_four() {
    for (columns, elementwise) in [(0, false), (3, true), (512, false)] {
        let scope = ExpressionScope::search_profile(columns, elementwise);
        assert_eq!(
            (
                scope.columns,
                scope.elementwise,
                scope.max_nodes,
                scope.max_depth
            ),
            (columns, elementwise, 15, 4)
        );
    }
    let scope = ExpressionScope::search_profile(3, true);
    let parse = |source: &str| MathExpr::parse(source).expect("the expression parses");
    // Depth 4 fits and depth 5 does not.
    scope.validate(&parse("sin(sin(sin(c0)))")).unwrap();
    assert!(scope.validate(&parse("sin(sin(sin(sin(c0))))")).is_err());
    // Fifteen nodes fit and sixteen do not, both at depth 4.
    let fifteen = parse("((c0 + c1) + (c2 + c0)) + ((c1 + c2) + (x + c0))");
    let found = shape(fifteen.ast());
    assert_eq!((found.nodes, found.depth), (15, 4));
    scope.validate(&fifteen).unwrap();
    let sixteen = parse("clip(c0 + c1, (c2 + c0) + c1, (c1 + c2) + (x + c0))");
    let found = shape(sixteen.ast());
    assert_eq!((found.nodes, found.depth), (16, 4));
    assert!(scope.validate(&sixteen).is_err());
    // The kernel pipeline operators' caps, 64 nodes and depth 16, still admit both.
    let kernel = ExpressionScope {
        columns: 3,
        elementwise: true,
        max_nodes: 64,
        max_depth: 16,
    };
    kernel.validate(&parse("sin(sin(sin(sin(c0))))")).unwrap();
    kernel.validate(&sixteen).unwrap();
}

/// Chained edits: each seed edits the parent with `mutate_v1` in a scope that changes every 50
/// seeds (map or row context) and every 250 seeds (0, 1, 2, 3 or 8 columns); a parent outside the
/// new scope restarts from `0.1`.
#[test]
fn five_thousand_chained_edits_under_the_search_profile_stay_in_scope_and_within_the_caps() {
    let restart = MathExpr::parse("0.1").expect("the restart parses");
    let mut parent = restart.clone();
    let (mut changed, mut restarts, mut max_nodes, mut max_depth) = (0, 0, 0, 0);
    let mut distinct = BTreeSet::new();
    for seed in 0..5_000u64 {
        let columns = [0, 1, 2, 3, 8][(seed / 250 % 5) as usize];
        let elementwise = seed / 50 % 2 == 0;
        let scope = ExpressionScope::search_profile(columns, elementwise);
        if scope.validate(&parent).is_err() {
            parent = restart.clone();
            restarts += 1;
        }
        let edit = parent
            .mutate_v1(seed, scope)
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        let child = &edit.expression;
        if let Err(error) = scope.validate(child) {
            panic!("seed {seed}: {error}");
        }
        // Counted again through the tree, independently of the scope's own check.
        let found = shape(child.ast());
        assert!(
            found.nodes <= 15 && found.depth <= 4,
            "seed {seed}: {} nodes, depth {}",
            found.nodes,
            found.depth
        );
        assert!(
            elementwise || !found.reads_x,
            "seed {seed}: x outside a map"
        );
        assert!(
            found.columns.iter().all(|column| *column < columns),
            "seed {seed}: columns {:?} of {columns}",
            found.columns
        );
        // The static check admits the same tree in a stage at that width.
        if columns > 0 {
            let op = if elementwise { "map" } else { "derive" };
            let source = child
                .to_source()
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
            if let Err(error) = validate_scope(&stage_program(op, &source), columns) {
                panic!("seed {seed}: {op} {source}: {error}");
            }
        }
        max_nodes = max_nodes.max(found.nodes);
        max_depth = max_depth.max(found.depth);
        changed += usize::from(edit.changed);
        distinct.insert(format!("{:?}", child.ast()));
        parent = edit.expression;
    }
    eprintln!(
        "PROPERTY edits=5000 changed={changed} distinct={} restarts={restarts} \
         max_nodes={max_nodes} max_depth={max_depth}",
        distinct.len()
    );
    // The caps are reached, so they bound the edits rather than hold vacuously.
    assert_eq!((max_nodes, max_depth), (15, 4));
    assert!(changed > 2_500, "changed {changed} of 5000");
}
