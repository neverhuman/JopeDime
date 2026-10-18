#[path = "../src/combinators/residual.rs"]
#[allow(dead_code)]
mod residual;
#[path = "../src/combinators/route.rs"]
mod route;
#[path = "../src/combinators/union.rs"]
mod union;

use residual::FitRole;
use union::{Action, ColumnRef, DenseInput, FitEffect, GraphError, Limits, Node, Program, Schema};

fn limits(width: usize, fuel: u64) -> Limits {
    Limits {
        max_nodes: 32,
        max_width: width,
        max_depth: 12,
        max_rows: 16,
        max_fuel: fuel,
    }
}

#[test]
fn columns_17_and_63_keep_semantic_identity_under_schema_shuffle() {
    let columns: Vec<_> = (0..70).map(|index| format!("feature_{index}")).collect();
    let schema = Schema::new(columns.clone()).unwrap();
    let col17 = ColumnRef::bind(&schema, "feature_17").unwrap();
    let col63 = ColumnRef::bind(&schema, "feature_63").unwrap();
    assert_eq!(col17.key(), "feature_17");
    let graph = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(col17)),
            Node::new(Action::Column(col63)),
            Node::new(Action::Union(vec![0, 1])),
        ],
        output: 2,
        limits: limits(2, 128),
    };
    let rows: Vec<_> = (0..3)
        .map(|row| (0..70).map(|column| (row * 100 + column) as f64).collect())
        .collect();
    let original = DenseInput::new(columns.clone(), rows.clone()).unwrap();
    let (first, _) = graph.evaluate(&original).unwrap();
    assert_eq!(first.width, 2);
    assert_eq!(first.values, vec![17.0, 63.0, 117.0, 163.0, 217.0, 263.0]);

    let mut permutation: Vec<_> = (0..70).collect();
    permutation.rotate_left(31);
    let shuffled_columns: Vec<_> = permutation.iter().map(|&i| columns[i].clone()).collect();
    let shuffled_rows: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| permutation.iter().map(|&i| row[i]).collect())
        .collect();
    let shuffled = DenseInput::new(shuffled_columns, shuffled_rows).unwrap();
    assert_eq!(schema.id(), shuffled.schema.id());
    let (second, _) = graph.evaluate(&shuffled).unwrap();
    assert_eq!(second.values, first.values);

    let foreign_schema = Schema::new(columns[..69].to_vec()).unwrap();
    assert!(matches!(
        graph.validate(&foreign_schema, 3),
        Err(GraphError::InvalidSchema(_))
    ));
}

#[test]
fn cycles_and_incompatible_shared_target_fits_are_refused() {
    let schema = Schema::new(vec!["x".into(), "y".into()]).unwrap();
    let x = ColumnRef::bind(&schema, "x").unwrap();
    let y = ColumnRef::bind(&schema, "y").unwrap();
    let cycle = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Union(vec![1])),
            Node::new(Action::Union(vec![0])),
        ],
        output: 0,
        limits: limits(4, 100),
    };
    assert!(matches!(
        cycle.validate(&schema, 2),
        Err(GraphError::Cycle(0))
    ));

    let fit_a = FitEffect {
        fit_id: "shared-estimator".into(),
        target_id: "label".into(),
        role: FitRole::InnerOof {
            outer_fold: Some(0),
            inner_fold: 0,
        },
        train_row_ids: vec![1, 2],
        fuel: 3,
    };
    let fit_b = FitEffect {
        train_row_ids: vec![1, 3],
        ..fit_a.clone()
    };
    let incompatible = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(x)).with_fit_effect(fit_a.clone()),
            Node::new(Action::Column(y)).with_fit_effect(fit_b),
            Node::new(Action::Union(vec![0, 1])),
        ],
        output: 2,
        limits: limits(4, 100),
    };
    assert!(matches!(
        incompatible.validate(&schema, 2),
        Err(GraphError::InvalidGraph(message)) if message.contains("shared fit identity")
    ));

    let shared = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(ColumnRef::bind(&schema, "x").unwrap()))
                .with_fit_effect(fit_a.clone()),
            Node::new(Action::Column(ColumnRef::bind(&schema, "y").unwrap()))
                .with_fit_effect(fit_a),
            Node::new(Action::Union(vec![0, 1])),
        ],
        output: 2,
        limits: limits(4, 20),
    };
    let report = shared.validate(&schema, 2).unwrap();
    assert_eq!(report.shared_fit_ids, vec!["shared-estimator"]);
    assert_eq!(report.worst_case_fuel, 11); // shared target fit charged once
}

#[test]
fn width_and_eager_route_fuel_are_bounded_before_evaluation() {
    let schema = Schema::new(vec!["predicate".into(), "yes".into(), "no".into()]).unwrap();
    let predicate = ColumnRef::bind(&schema, "predicate").unwrap();
    let yes = ColumnRef::bind(&schema, "yes").unwrap();
    let no = ColumnRef::bind(&schema, "no").unwrap();
    let wide = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(yes.clone())),
            Node::new(Action::Column(no.clone())),
            Node::new(Action::Union(vec![0, 1])),
        ],
        output: 2,
        limits: limits(1, 100),
    };
    assert!(matches!(
        wide.validate(&schema, 3),
        Err(GraphError::Limit(message)) if message.contains("width")
    ));

    let routed = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(predicate)),
            Node::new(Action::Column(yes)),
            Node::new(Action::Column(no)),
            Node::new(route::route_action(0, 1, 2)),
        ],
        output: 3,
        limits: limits(1, 11),
    };
    assert!(matches!(
        routed.validate(&schema, 3),
        Err(GraphError::Limit(message)) if message.contains("fuel")
    ));
    let bounded = Program {
        limits: limits(1, 12),
        ..routed.clone()
    };
    let input = DenseInput::new(
        schema.columns().to_vec(),
        vec![
            vec![1.0, 10.0, 20.0],
            vec![0.0, 11.0, 21.0],
            vec![1.0, 12.0, 22.0],
        ],
    )
    .unwrap();
    let (output, report) = bounded.evaluate(&input).unwrap();
    assert_eq!(output.values, vec![10.0, 21.0, 12.0]);
    assert_eq!(report.worst_case_fuel, 12); // predicate, both branches, and route
    assert_eq!(report.eager_route_branches, 1);
}

#[test]
fn invalid_predicate_and_unavailable_emitter_fail_explicitly() {
    let schema = Schema::new(vec!["predicate".into(), "yes".into(), "no".into()]).unwrap();
    let graph = Program {
        abi: Program::ABI,
        nodes: vec![
            Node::new(Action::Column(
                ColumnRef::bind(&schema, "predicate").unwrap(),
            )),
            Node::new(Action::Column(ColumnRef::bind(&schema, "yes").unwrap())),
            Node::new(Action::Column(ColumnRef::bind(&schema, "no").unwrap())),
            Node::new(route::route_action(0, 1, 2)),
        ],
        output: 3,
        limits: limits(1, 20),
    };
    let input = DenseInput::new(schema.columns().to_vec(), vec![vec![0.5, 1.0, 2.0]]).unwrap();
    assert!(matches!(
        graph.evaluate(&input),
        Err(GraphError::Evaluation(message)) if message.contains("exactly zero or one")
    ));
    assert!(matches!(
        graph.emit_for_target("wasm"),
        Err(GraphError::UnsupportedEmitter(message)) if message.contains("wasm")
    ));
}
