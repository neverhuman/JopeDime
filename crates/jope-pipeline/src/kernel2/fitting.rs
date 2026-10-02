//! Fit with completed-stage counts retained even when a later stage/head fails.
//! Existing fitted operations and heads own every numerical calculation.

use super::*;
use feat_math::{
    dsl::Stage,
    heads::{fit_head, is_head_op},
    kernel::telemetry::{apply_op_traced, fit_traced},
    model::Model,
    ops::fit_op,
    target_transform::{fit_target, is_target_op},
};

pub(super) fn fit_observed(
    genome: &GenomeIr,
    rows: &Rows,
    targets: &[f64],
    record: &mut Kernel2Observation,
) -> Result<Model> {
    let GenomeIr::Pipeline(pipe) = genome else {
        let (model, pass) = fit_traced(genome, rows, targets)?;
        record.faults.add_pass(&pass);
        return Ok(model);
    };
    let target_stage = pipe.stages.first().filter(|s| is_target_op(&s.op));
    let stages = &pipe.stages[usize::from(target_stage.is_some())..];
    ensure!(
        !stages.iter().any(|s| is_target_op(&s.op)),
        "target transform must be the first stage"
    );
    let target = target_stage.map(|s| fit_target(s, targets)).transpose()?;
    let effective_targets = target.as_ref().map(|t| t.forward_all(targets));
    let targets = effective_targets.as_deref().unwrap_or(targets);
    let (transforms, head_stage) = match stages.split_last() {
        Some((last, prefix)) if is_head_op(&last.op) => (prefix, last.clone()),
        _ => (stages, Stage::new("ridge")),
    };
    let mut current = rows.clone();
    feat_math::sanitize(&mut current);
    let mut ops = Vec::with_capacity(transforms.len());
    for stage in transforms {
        let op = fit_op(stage, &current, targets)?;
        let (next, counts) = apply_op_traced(&op, &current);
        record.faults.add_stage(&counts);
        super::measurement::reject_hard(record)?;
        current = next;
        ops.push(op);
    }
    let head = Box::new(fit_head(&head_stage, &current, targets)?);
    Ok(Model::Pipeline { target, ops, head })
}
