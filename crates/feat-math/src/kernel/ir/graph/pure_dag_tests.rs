use super::*;
use crate::combinators::union::{DenseInput, Limits};

fn dag_limits() -> Limits {
    Limits {
        max_nodes: 64,
        max_width: 64,
        max_depth: 64,
        max_rows: 1000,
        max_fuel: 1_000_000,
    }
}

fn column(schema: &InputSchema, index: usize) -> ColumnRef {
    ColumnRef {
        schema_id: schema.identity().unwrap(),
        index,
        key: schema.columns[index].clone(),
    }
}

fn graph(schema: InputSchema, nodes: Vec<Node>, output: usize) -> ProgramGraph {
    ProgramGraph {
        schema: PURE_FEATURE_DAG_SCHEMA.into(),
        version: 1,
        input_schema: schema,
        profile_id: registry_v2::PROFILE_ID.into(),
        profile_revision: registry_v2::REVISION,
        registry_sha256: registry_v2::source_sha256(),
        nodes,
        output,
    }
}

/// Node 1 is `Columns(a,b)`, node 2 is a bare-column scalar over `a`, node 3 stacks 1 and 2,
/// and node 4 stacks 1 and 3. Column `a` and nodes 1 and 3 each have several consumers.
#[test]
fn lowered_stack_evaluates_each_shared_column_once() {
    let schema = InputSchema::new(vec!["a".into(), "b".into()], RowSemantics::Iid).unwrap();
    let g = ProgramGraph::new_pure_feature_dag(
        schema.clone(),
        vec![
            Node::Input,
            Node::Columns {
                input: 0,
                columns: vec![column(&schema, 0), column(&schema, 1)],
            },
            Node::Scalar {
                input: 0,
                expression: Expression::Column {
                    column: column(&schema, 0),
                },
                affine: false,
            },
            Node::Stack {
                members: vec![1, 2],
            },
            Node::Stack {
                members: vec![1, 3],
            },
        ],
        4,
        &Constraints::default(),
        dag_limits(),
    )
    .unwrap();
    let limits = Constraints::default();
    let program = g.lower_pure_feature_dag(&limits, dag_limits()).unwrap();
    let columns = program
        .nodes
        .iter()
        .filter(|node| matches!(node.action, crate::combinators::union::Action::Column(_)))
        .count();
    assert_eq!(columns, 2, "shared column 'a' must be one union node");
    let input = DenseInput::new(
        vec!["a".into(), "b".into()],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    let (output, _validation, counts) = g
        .evaluate_pure_feature_dag_counted(&input, &limits, dag_limits())
        .unwrap();
    assert_eq!(
        counts,
        vec![1; program.nodes.len()],
        "every lowered node computed once"
    );
    assert_eq!(output.width, 5);
    assert_eq!(
        output.values,
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0]
    );
}

#[test]
fn operator_node_is_refused_at_the_pure_entry() {
    let schema = InputSchema::new(vec!["a".into()], RowSemantics::Iid).unwrap();
    let g = graph(
        schema.clone(),
        vec![
            Node::Input,
            Node::Operator {
                input: 0,
                operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
                parameters: BTreeMap::new(),
            },
        ],
        1,
    );
    let err = g
        .lower_pure_feature_dag(&Constraints::default(), dag_limits())
        .unwrap_err();
    assert!(err.contains("refuses node 1"), "{err}");
}

#[test]
fn ordered_row_semantics_are_refused() {
    let schema = InputSchema::new(vec!["a".into()], RowSemantics::Ordered).unwrap();
    let g = graph(
        schema.clone(),
        vec![
            Node::Input,
            Node::Columns {
                input: 0,
                columns: vec![column(&schema, 0)],
            },
        ],
        1,
    );
    let err = g
        .lower_pure_feature_dag(&Constraints::default(), dag_limits())
        .unwrap_err();
    assert!(err.contains("Iid"), "{err}");
}

#[test]
fn unreachable_node_is_refused() {
    let schema = InputSchema::new(vec!["a".into()], RowSemantics::Iid).unwrap();
    let g = graph(
        schema.clone(),
        vec![
            Node::Input,
            Node::Columns {
                input: 0,
                columns: vec![column(&schema, 0)],
            },
            Node::Columns {
                input: 0,
                columns: vec![column(&schema, 0)],
            },
        ],
        1,
    );
    let err = g
        .lower_pure_feature_dag(&Constraints::default(), dag_limits())
        .unwrap_err();
    assert!(err.contains("reachable"), "{err}");
}
