use feat_math::kernel::ordered_axis::OrderedInputIdentity;
use feat_math::kernel::strategy::residual::{
    propose, DiagnosticBudget, EditKind, FitClosure, RowSemantics,
};

fn closure<'a>(
    y: &'a [f64],
    predictions: &'a [f64],
    rows: &'a [Vec<f64>],
    features: &'a [usize],
    row_ids: &'a [u64],
) -> FitClosure<'a> {
    FitClosure {
        producer: "fit_residual_v1",
        fold_id: "fold-a",
        row_semantics: RowSemantics::Unordered,
        selected_features: features,
        train_row_ids: row_ids,
        train_x: rows,
        train_y: y,
        train_predictions: predictions,
        model_state_sha256: "model-digest",
        role_ancestry_sha256: "role-digest",
    }
}

#[test]
fn held_labels_are_outside_the_proposal_input_and_cannot_change_it() {
    let train_y = [1.0, 3.0, 5.0, 7.0];
    let predictions = [1.0, 2.0, 3.0, 4.0];
    let rows = vec![
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
    ];
    let features = [0, 1];
    let row_ids = [10, 11, 12, 13];
    let mut first_budget = DiagnosticBudget::new(100);
    let first = propose(
        &closure(&train_y, &predictions, &rows, &features, &row_ids),
        &mut first_budget,
    )
    .unwrap();
    let perturbed_held_y = [1.0e30, -1.0e30];
    let _held_labels_are_not_an_argument_to_the_fit_closure = perturbed_held_y;
    let mut second_budget = DiagnosticBudget::new(100);
    let second = propose(
        &closure(&train_y, &predictions, &rows, &features, &row_ids),
        &mut second_budget,
    )
    .unwrap();
    assert_eq!(first, second);
}

#[test]
fn ordered_rows_need_a_nonempty_declared_semantics() {
    let train_y = [1.0, 3.0, 5.0, 7.0];
    let predictions = [1.0, 2.0, 3.0, 4.0];
    let rows = vec![
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
    ];
    let features = [0, 1];
    let row_ids = [10, 11, 12, 13];
    let mut undeclared = closure(&train_y, &predictions, &rows, &features, &row_ids);
    undeclared.row_semantics = RowSemantics::Ordered {
        identity: OrderedInputIdentity::new(" ", " "),
    };
    assert!(propose(&undeclared, &mut DiagnosticBudget::new(100)).is_none());
    let mut declared = closure(&train_y, &predictions, &rows, &features, &row_ids);
    declared.row_semantics = RowSemantics::Ordered {
        identity: OrderedInputIdentity::new("group", "timestamp"),
    };
    assert!(propose(&declared, &mut DiagnosticBudget::new(100)).is_some());
}

#[test]
fn heavy_diagnostics_are_charged_before_work_and_overrun_refuses() {
    let train_y = [1.0, 3.0, 5.0, 7.0];
    let predictions = [1.0, 2.0, 3.0, 4.0];
    let rows = vec![
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
    ];
    let features = [0, 1];
    let row_ids = [10, 11, 12, 13];
    let input = closure(&train_y, &predictions, &rows, &features, &row_ids);
    let mut short = DiagnosticBudget::new(15);
    assert!(propose(&input, &mut short).is_none());
    assert_eq!(short.charged(), 0);
    let mut sufficient = DiagnosticBudget::new(16);
    let proposal = propose(&input, &mut sufficient).unwrap();
    assert_eq!(proposal.charged_work, 16);
    assert_eq!(sufficient.charged(), 16);
    assert_eq!(proposal.source, "fit_residual_v1");
    assert!(proposal.edits.iter().all(|edit| matches!(
        edit.kind,
        EditKind::AddResidualFeature | EditKind::ScaleFeatureByResidual
    )));
}
