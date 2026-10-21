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

/// Eager row routing: every row takes `when_true` where `predicate` is nonzero, else
/// `when_false`. Public so the lazy partition route can be checked against it bit for bit.
pub fn route_rows(
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

/// Region-based lazy scalar conditional (`LazyIf`, W03 / G05).
///
/// Evaluates `predicate` first (`1.0` -> `when_true`, `0.0` -> `when_false`) and
/// invokes ONLY the selected branch region closure. The unselected branch is never
/// evaluated and receives zero invocations even if it contains a poison trap.
#[allow(dead_code)]
pub struct LazyIf;

#[allow(dead_code)]
impl LazyIf {
    pub fn eval_scalar<FTrue, FFalse>(
        predicate: f64,
        when_true: FTrue,
        when_false: FFalse,
    ) -> Result<f64, GraphError>
    where
        FTrue: FnOnce() -> Result<f64, GraphError>,
        FFalse: FnOnce() -> Result<f64, GraphError>,
    {
        let out = if predicate == 1.0 {
            when_true()?
        } else if predicate == 0.0 {
            when_false()?
        } else {
            return Err(GraphError::Evaluation(
                "LazyIf predicate must be exactly zero or one".into(),
            ));
        };
        if !out.is_finite() {
            return Err(GraphError::Evaluation(
                "LazyIf selected branch returned non-finite scalar".into(),
            ));
        }
        Ok(out)
    }

    pub fn eval_rows<FTrue, FFalse>(
        predicates: &[f64],
        mut when_true: FTrue,
        mut when_false: FFalse,
    ) -> Result<Vec<f64>, GraphError>
    where
        FTrue: FnMut(usize) -> Result<f64, GraphError>,
        FFalse: FnMut(usize) -> Result<f64, GraphError>,
    {
        let mut out = Vec::with_capacity(predicates.len());
        for (row, &pred) in predicates.iter().enumerate() {
            let val = if pred == 1.0 {
                when_true(row)?
            } else if pred == 0.0 {
                when_false(row)?
            } else {
                return Err(GraphError::Evaluation(
                    "LazyIf predicate must be exactly zero or one".into(),
                ));
            };
            if !val.is_finite() {
                return Err(GraphError::Evaluation(
                    "LazyIf selected branch returned non-finite value".into(),
                ));
            }
            out.push(val);
        }
        Ok(out)
    }
}

/// Validated row-partition routing mask (`RouteMask`, W03 / G05).
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteMask {
    pub total_rows: usize,
    pub true_rows: Vec<usize>,
    pub false_rows: Vec<usize>,
    pub select_true: Vec<bool>,
}

#[allow(dead_code)]
impl RouteMask {
    pub fn from_predicate(predicate: &Matrix) -> Result<Self, GraphError> {
        if predicate.width != 1 || predicate.rows == 0 || predicate.values.len() != predicate.rows {
            return Err(GraphError::Evaluation(
                "RouteMask predicate must be a non-empty single-column matrix".into(),
            ));
        }
        let mut true_rows = Vec::new();
        let mut false_rows = Vec::new();
        let mut select_true = Vec::with_capacity(predicate.rows);
        for row in 0..predicate.rows {
            let selector = predicate.get(row, 0);
            if selector == 1.0 {
                true_rows.push(row);
                select_true.push(true);
            } else if selector == 0.0 {
                false_rows.push(row);
                select_true.push(false);
            } else {
                return Err(GraphError::Evaluation(
                    "RouteMask predicate values must be exactly zero or one".into(),
                ));
            }
        }
        Ok(Self {
            total_rows: predicate.rows,
            true_rows,
            false_rows,
            select_true,
        })
    }
}

/// Batch row partitioner (`Partition`, W03 / G05).
#[allow(dead_code)]
pub struct Partition;

#[allow(dead_code)]
impl Partition {
    pub fn split(input: &Matrix, mask: &RouteMask) -> Result<(Matrix, Matrix), GraphError> {
        if input.rows != mask.total_rows
            || input.width == 0
            || input.values.len() != input.rows.saturating_mul(input.width)
        {
            return Err(GraphError::Evaluation(
                "Partition input matrix shape does not match RouteMask".into(),
            ));
        }
        let extract = |indices: &[usize]| -> Matrix {
            let mut values = Vec::with_capacity(indices.len() * input.width);
            for &row in indices {
                let start = row * input.width;
                values.extend_from_slice(&input.values[start..start + input.width]);
            }
            Matrix {
                rows: indices.len(),
                width: input.width,
                values,
            }
        };
        Ok((extract(&mask.true_rows), extract(&mask.false_rows)))
    }
}

/// Partitioned batch merger (`Merge`, W03 / G05).
#[allow(dead_code)]
pub struct Merge;

#[allow(dead_code)]
impl Merge {
    pub fn reconstruct(
        mask: &RouteMask,
        true_out: &Matrix,
        false_out: &Matrix,
    ) -> Result<Matrix, GraphError> {
        if true_out.rows != mask.true_rows.len()
            || false_out.rows != mask.false_rows.len()
            || true_out.width == 0
            || true_out.width != false_out.width
            || true_out.values.len() != true_out.rows.saturating_mul(true_out.width)
            || false_out.values.len() != false_out.rows.saturating_mul(false_out.width)
        {
            return Err(GraphError::Evaluation(
                "Merge branch matrix shapes do not align with RouteMask".into(),
            ));
        }
        if true_out.values.iter().any(|v| !v.is_finite())
            || false_out.values.iter().any(|v| !v.is_finite())
        {
            return Err(GraphError::Evaluation(
                "Merge branch output contains non-finite values".into(),
            ));
        }
        let width = true_out.width;
        let mut values = Vec::with_capacity(mask.total_rows * width);
        let mut t_idx = 0usize;
        let mut f_idx = 0usize;
        for &is_true in &mask.select_true {
            if is_true {
                let start = t_idx * width;
                values.extend_from_slice(&true_out.values[start..start + width]);
                t_idx += 1;
            } else {
                let start = f_idx * width;
                values.extend_from_slice(&false_out.values[start..start + width]);
                f_idx += 1;
            }
        }
        Ok(Matrix {
            rows: mask.total_rows,
            width,
            values,
        })
    }
}

/// Partitioned lazy batch router: splits `input` by `mask`, evaluates ONLY non-empty
/// branch partitions, and merges the outputs back into exact row order.
#[allow(dead_code)]
pub fn partition_route_lazy<FTrue, FFalse>(
    input: &Matrix,
    mask: &RouteMask,
    out_width: usize,
    eval_true: FTrue,
    eval_false: FFalse,
) -> Result<Matrix, GraphError>
where
    FTrue: FnOnce(&Matrix) -> Result<Matrix, GraphError>,
    FFalse: FnOnce(&Matrix) -> Result<Matrix, GraphError>,
{
    if out_width == 0 {
        return Err(GraphError::Evaluation(
            "partition_route_lazy requires non-zero output width".into(),
        ));
    }
    let (true_in, false_in) = Partition::split(input, mask)?;
    let true_out = if true_in.rows > 0 {
        eval_true(&true_in)?
    } else {
        Matrix {
            rows: 0,
            width: out_width,
            values: Vec::new(),
        }
    };
    let false_out = if false_in.rows > 0 {
        eval_false(&false_in)?
    } else {
        Matrix {
            rows: 0,
            width: out_width,
            values: Vec::new(),
        }
    };
    Merge::reconstruct(mask, &true_out, &false_out)
}
