use feat_math::combinators::union::{Action, ColumnRef, DenseInput, Limits, Node, Program};

fn limits() -> Limits {
    Limits {
        max_nodes: 64,
        max_width: 64,
        max_depth: 64,
        max_rows: 1000,
        max_fuel: 1_000_000,
    }
}

/// A shared column (node 0) feeds two consumers, so it is a fan-out subexpression.
#[test]
fn w04_shared_subexpression_evaluates_once() {
    let input = DenseInput::new(
        vec!["a".into(), "b".into()],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    let a = ColumnRef::bind(&input.schema, "a").unwrap();
    let b = ColumnRef::bind(&input.schema, "b").unwrap();
    let program = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(a)),
            Node::new(Action::Column(b)),
            Node::new(Action::Union(vec![0, 1])),
            Node::new(Action::Union(vec![0, 2])),
        ],
        output: 3,
        limits: limits(),
    };
    let (output, _validation, counts) = program.evaluate_counted(&input).unwrap();
    assert_eq!(counts[0], 1, "shared column node computed more than once");
    assert_eq!(counts, vec![1, 1, 1, 1]);
    assert_eq!(output.values, vec![1.0, 1.0, 2.0, 3.0, 3.0, 4.0]);
}
