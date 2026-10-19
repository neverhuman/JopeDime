use feat_math::combinators::residual::FitRole;
use feat_math::combinators::union::{
    Action, ColumnRef, DenseInput, ExactScalar, FitEffect, GraphError, Limits, Node, Program,
};
use feat_math::kernel::ir::graph::{
    Constraints, Expression, InputSchema, Node as GraphNode, ProgramGraph, RowSemantics,
    PURE_FEATURE_DAG_SCHEMA,
};

fn exact(value: f64) -> ExactScalar {
    ExactScalar::new(value).unwrap()
}

/// A NaN payload written directly, since `ExactScalar::new` refuses non-finite values.
fn nonfinite_bits() -> ExactScalar {
    ExactScalar::from_bits(f64::NAN.to_bits())
}

fn limits() -> Limits {
    Limits {
        max_nodes: 64,
        max_width: 64,
        max_depth: 64,
        max_rows: 1000,
        max_fuel: 1_000_000,
    }
}

fn program(nodes: Vec<Node>, output: usize) -> Program {
    Program {
        abi: Program::ABI,
        nodes,
        output,
        limits: limits(),
    }
}

/// Node 2 (`2a+1`) is bound by a Let and also consumed directly by the output. Node 4 is
/// consumed by a SelectFeature and by the output. Each shared node must be computed once.
#[test]
fn w04_shared_subexpression_evaluates_once() {
    let input = DenseInput::new(
        vec!["a".into(), "b".into()],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    let a = ColumnRef::bind(&input.schema, "a").unwrap();
    let b = ColumnRef::bind(&input.schema, "b").unwrap();
    let p = program(
        vec![
            Node::new(Action::Column(a)), // 0: a
            Node::new(Action::Column(b)), // 1: b
            Node::new(Action::Affine {
                input: 0,
                scale: exact(2.0),
                offset: exact(1.0),
            }), // 2: 2a+1 (shared)
            Node::new(Action::Let(2)),    // 3: bind 2
            Node::new(Action::ConcatFeatures(vec![3, 1])), // 4: [2a+1, b]
            Node::new(Action::SelectFeature {
                input: 4,
                column_index: 0,
            }), // 5: 2a+1
            Node::new(Action::ConcatFeatures(vec![4, 5, 2])), // 6: output
        ],
        6,
    );
    let (output, validation, counts) = p.evaluate_counted(&input).unwrap();
    assert_eq!(
        counts,
        vec![1; 7],
        "every node must be computed exactly once"
    );
    assert_eq!(counts[2], 1, "shared Affine node computed more than once");
    assert_eq!(counts[4], 1, "shared Concat node computed more than once");
    assert_eq!(validation.node_widths[6], 4);
    assert_eq!(output.rows, 2);
    assert_eq!(output.width, 4);
    // Row order and column order: [2a+1, b, 2a+1, 2a+1] per row.
    assert_eq!(output.values, vec![3.0, 2.0, 3.0, 3.0, 7.0, 4.0, 7.0, 7.0]);
}

/// Two inputs whose columns are permuted share one semantic schema id, so one bound program
/// runs on both. Each call gets fresh counts, and outputs match across the permutation.
#[test]
fn invocation_local_cache_resets_across_inputs_and_permuted_schemas() {
    let ab = InputSchema::new(vec!["a".into(), "b".into()], RowSemantics::Iid).unwrap();
    let a = ab.bind_semantic_column("a").unwrap();
    let p = program(
        vec![
            Node::new(Action::Column(a)),
            Node::new(Action::Affine {
                input: 0,
                scale: exact(3.0),
                offset: exact(0.0),
            }),
            Node::new(Action::ConcatFeatures(vec![1, 1])),
        ],
        2,
    );
    let in_ab = DenseInput::new(
        vec!["a".into(), "b".into()],
        vec![vec![1.0, 9.0], vec![2.0, 9.0]],
    )
    .unwrap();
    let in_ba = DenseInput::new(
        vec!["b".into(), "a".into()],
        vec![vec![9.0, 1.0], vec![9.0, 2.0]],
    )
    .unwrap();
    let (out1, _, counts1) = p.evaluate_counted(&in_ab).unwrap();
    let (out2, _, counts2) = p.evaluate_counted(&in_ba).unwrap();
    let (out3, _, counts3) = p.evaluate_counted(&in_ab).unwrap();
    assert_eq!(
        out1.values, out2.values,
        "permuted schema changed the output"
    );
    assert_eq!(out1.values, out3.values);
    assert_eq!(counts1, vec![1, 1, 1]);
    assert_eq!(counts2, counts1, "counts leaked across invocations");
    assert_eq!(counts3, counts1);
    assert_eq!(out1.values, vec![3.0, 3.0, 6.0, 6.0]);

    // Same program, changed `a` values and a different row count: no cached rows leak in.
    let in_changed = DenseInput::new(
        vec!["a".into(), "b".into()],
        vec![vec![10.0, 9.0], vec![20.0, 9.0], vec![-5.0, 9.0]],
    )
    .unwrap();
    let (out4, _, counts4) = p.evaluate_counted(&in_changed).unwrap();
    assert_eq!(out4.rows, 3);
    assert_eq!(out4.values, vec![30.0, 30.0, 60.0, 60.0, -15.0, -15.0]);
    assert_ne!(out4.values, out1.values);
    assert_eq!(counts4, vec![1, 1, 1], "counts leaked across invocations");
}

/// Refusals, each returned as a typed error.
#[test]
fn rejects_cycles_invalid_refs_oob_select_overflow_and_fit_effects() {
    let input = DenseInput::new(vec!["a".into()], vec![vec![1.0]]).unwrap();
    let a = || Node::new(Action::Column(ColumnRef::bind(&input.schema, "a").unwrap()));

    // Self cycle.
    let err = program(vec![Node::new(Action::Let(0))], 0)
        .evaluate_counted(&input)
        .unwrap_err();
    assert!(matches!(err, GraphError::Cycle(_)), "self cycle: {err:?}");

    // Two-node cycle.
    let err = program(
        vec![Node::new(Action::Union(vec![1])), Node::new(Action::Let(0))],
        0,
    )
    .evaluate_counted(&input)
    .unwrap_err();
    assert!(
        matches!(err, GraphError::Cycle(_)),
        "two-node cycle: {err:?}"
    );

    // Invalid node reference.
    let err = program(vec![a(), Node::new(Action::Let(9))], 1)
        .evaluate_counted(&input)
        .unwrap_err();
    assert!(
        matches!(err, GraphError::InvalidGraph(_)),
        "invalid ref: {err:?}"
    );

    // Out-of-bounds SelectFeature on a width-2 input.
    let err = program(
        vec![
            a(),
            Node::new(Action::ConcatFeatures(vec![0, 0])),
            Node::new(Action::SelectFeature {
                input: 1,
                column_index: 2,
            }),
        ],
        2,
    )
    .evaluate_counted(&input)
    .unwrap_err();
    assert!(
        matches!(err, GraphError::InvalidGraph(_)),
        "oob select: {err:?}"
    );

    // Fuel overflow against a declared limit.
    let mut tight = program(vec![a()], 0);
    tight.limits.max_fuel = 1;
    let err = tight
        .evaluate_counted(&DenseInput::new(vec!["a".into()], vec![vec![1.0], vec![2.0]]).unwrap())
        .unwrap_err();
    assert!(matches!(err, GraphError::Limit(_)), "fuel limit: {err:?}");

    // Non-finite affine parameter.
    let err = program(
        vec![
            a(),
            Node::new(Action::Affine {
                input: 0,
                scale: nonfinite_bits(),
                offset: exact(0.0),
            }),
        ],
        1,
    )
    .evaluate_counted(&input)
    .unwrap_err();
    assert!(
        matches!(err, GraphError::InvalidGraph(_)),
        "nan affine: {err:?}"
    );

    // Value overflow during evaluation is refused, not returned as infinity.
    let big = DenseInput::new(vec!["a".into()], vec![vec![1e308]]).unwrap();
    let err = program(
        vec![
            a(),
            Node::new(Action::Affine {
                input: 0,
                scale: exact(10.0),
                offset: exact(0.0),
            }),
        ],
        1,
    )
    .evaluate_counted(&big)
    .unwrap_err();
    assert!(
        matches!(err, GraphError::Evaluation(_)),
        "overflow: {err:?}"
    );

    // A node with a fit effect is refused by the pure-feature evaluator.
    let fit = a().with_fit_effect(FitEffect {
        fit_id: "fit-1".into(),
        target_id: "target".into(),
        role: FitRole::FullTraining,
        train_row_ids: vec![0],
        fuel: 1,
    });
    let err = program(vec![fit], 0).evaluate_counted(&input).unwrap_err();
    assert!(
        matches!(err, GraphError::InvalidGraph(_)),
        "fit effect: {err:?}"
    );
}

/// Legacy v1 identity keeps physical column order. The opt-in semantic schema is
/// permutation-invariant. These identity values were recorded from the unchanged v1 code.
#[test]
fn legacy_v1_input_schema_identity_and_graph_lower_unchanged() {
    let ab = InputSchema::new(vec!["a".into(), "b".into()], RowSemantics::Iid).unwrap();
    let ba = InputSchema::new(vec!["b".into(), "a".into()], RowSemantics::Iid).unwrap();
    assert_eq!(
        ab.identity().unwrap(),
        "6ec5b83eca17da46b199f7a6c8d73a9720ebf9cb2fb016703048564cdabb45f3"
    );
    assert_eq!(
        ba.identity().unwrap(),
        "d61ec5ee882742388bdd86255158cc203fb6440429f27be8c6e4f3b44dbe6328"
    );
    assert_ne!(
        ab.identity().unwrap(),
        ba.identity().unwrap(),
        "v1 identity must stay order-sensitive"
    );
    assert_eq!(
        ab.semantic_key_schema().unwrap().id(),
        ba.semantic_key_schema().unwrap().id(),
        "semantic schema must be permutation-invariant"
    );
}

/// The pure feature DAG through public APIs only: admitted construction, canonical bytes and
/// identity, legacy rejection, repeated evaluation, and bound refusals.
#[test]
fn program_graph_pure_feature_dag_public_roundtrip_and_bounds() {
    let schema = InputSchema::new(vec!["a".into(), "b".into()], RowSemantics::Iid).unwrap();
    let a = schema.column(0).unwrap();
    let b = schema.column(1).unwrap();
    let constraints = Constraints::default();
    // Node 1 = Columns(a, b), node 2 = Scalar(a), node 3 = Stack[1, 2], node 4 = Stack[1, 3].
    let graph = ProgramGraph::new_pure_feature_dag(
        schema.clone(),
        vec![
            GraphNode::Input,
            GraphNode::Columns {
                input: 0,
                columns: vec![a.clone(), b],
            },
            GraphNode::Scalar {
                input: 0,
                expression: Expression::Column { column: a },
                affine: false,
            },
            GraphNode::Stack {
                members: vec![1, 2],
            },
            GraphNode::Stack {
                members: vec![1, 3],
            },
        ],
        4,
        &constraints,
        limits(),
    )
    .unwrap();
    let program = graph
        .lower_pure_feature_dag(&constraints, limits())
        .unwrap();

    // Column "a" is one shared union node, and both Columns(a, b) and column "a" have several
    // consumers in the lowered program.
    let a_nodes: Vec<usize> = program
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(&node.action, Action::Column(c) if c.key() == "a"))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(a_nodes.len(), 1, "column 'a' must be a single union node");
    let concat_ab = program
        .nodes
        .iter()
        .position(|node| matches!(&node.action, Action::ConcatFeatures(_)))
        .unwrap();
    let consumers = |target: usize| {
        program
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.action, Action::ConcatFeatures(members) if members.contains(&target))
            })
            .count()
    };
    assert!(consumers(a_nodes[0]) > 1, "column 'a' has one consumer");
    assert!(consumers(concat_ab) > 1, "Columns(a, b) has one consumer");

    // Canonical roundtrip and domain-separated identity.
    let bytes = graph
        .to_pure_feature_dag_bytes(&constraints, limits())
        .unwrap();
    let restored =
        ProgramGraph::from_pure_feature_dag_bytes(&bytes, &constraints, limits()).unwrap();
    assert_eq!(restored, graph);
    let id = graph
        .pure_feature_dag_identity(&constraints, limits())
        .unwrap();
    assert_eq!(id.schema_id, PURE_FEATURE_DAG_SCHEMA);
    assert_eq!(
        restored
            .pure_feature_dag_identity(&constraints, limits())
            .unwrap(),
        id
    );
    assert!(
        ProgramGraph::from_bytes(&bytes, &constraints).is_err(),
        "legacy v1 parser accepted pure-DAG bytes"
    );

    // Two evaluations with different column orders and row counts, each computing every node once.
    let first = DenseInput::new(
        vec!["a".into(), "b".into()],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    let (out1, _, counts1) = graph
        .evaluate_pure_feature_dag_counted(&first, &constraints, limits())
        .unwrap();
    assert_eq!(out1.rows, 2);
    assert_eq!(
        out1.values,
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0]
    );
    assert_eq!(counts1, vec![1; program.nodes.len()]);
    let second = DenseInput::new(vec!["b".into(), "a".into()], vec![vec![9.0, 5.0]]).unwrap();
    let (out2, _, counts2) = graph
        .evaluate_pure_feature_dag_counted(&second, &constraints, limits())
        .unwrap();
    assert_eq!(out2.rows, 1);
    assert_eq!(out2.values, vec![5.0, 9.0, 5.0, 9.0, 5.0]);
    assert_eq!(counts2, vec![1; program.nodes.len()]);

    // Invalid graph Constraints: edges above the ABI bound, zero depth, and bytes one above
    // canon::MAX_BYTES. The exact MAX_BYTES value is valid, so it is not a refusal witness.
    let invalid = [
        Constraints {
            max_edges: 256,
            ..constraints
        },
        Constraints {
            max_depth: 0,
            ..constraints
        },
        Constraints {
            max_bytes: 1_048_577,
            ..constraints
        },
    ];
    for bad in invalid {
        assert!(
            graph.lower_pure_feature_dag(&bad, limits()).is_err(),
            "{bad:?}"
        );
        assert!(
            graph.to_pure_feature_dag_bytes(&bad, limits()).is_err(),
            "{bad:?}"
        );
    }

    // Stricter graph Constraints than the dag limits: depth 3 and width 3 nodes, and a graph
    // larger than 64 bytes, each refused against loose dag limits.
    let stricter = [
        Constraints {
            max_depth: 2,
            ..constraints
        },
        Constraints {
            max_width: 2,
            ..constraints
        },
        Constraints {
            max_bytes: 64,
            ..constraints
        },
    ];
    for tight in stricter {
        assert!(
            graph.lower_pure_feature_dag(&tight, limits()).is_err(),
            "{tight:?}"
        );
    }
}
