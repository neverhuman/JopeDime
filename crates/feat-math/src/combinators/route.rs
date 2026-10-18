//! Explicit eager row routing actions for bounded DAG programs.

use super::union::{Action, GraphError, Matrix};

/// Construct a route action. Both branches are evaluated by `Program`; fuel and
/// memory budgets therefore include both branches even when a row selects only one.
pub fn route_action(predicate: usize, when_true: usize, when_false: usize) -> Action {
    Action::Route {
        predicate,
        when_true,
        when_false,
    }
}

pub(crate) fn route_rows(
    predicate: &Matrix,
    when_true: &Matrix,
    when_false: &Matrix,
) -> Result<Matrix, GraphError> {
    if predicate.width != 1
        || predicate.rows != when_true.rows
        || predicate.rows != when_false.rows
        || when_true.width != when_false.width
    {
        return Err(GraphError::Evaluation(
            "route predicate and branch shapes do not align".into(),
        ));
    }
    let mut values = Vec::with_capacity(predicate.rows * when_true.width);
    for row in 0..predicate.rows {
        let selector = predicate.get(row, 0);
        let branch = if selector == 0.0 {
            when_false
        } else if selector == 1.0 {
            when_true
        } else {
            return Err(GraphError::Evaluation(
                "route predicate values must be exactly zero or one".into(),
            ));
        };
        let start = row * branch.width;
        values.extend_from_slice(&branch.values[start..start + branch.width]);
    }
    Ok(Matrix {
        rows: predicate.rows,
        width: when_true.width,
        values,
    })
}
