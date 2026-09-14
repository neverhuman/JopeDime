use super::PreprocessPlan;
use crate::data::{parse_field, MultiTargets};
use crate::ingest::{Frame, RawColumn};
use crate::task::{TaskKind, TaskSpec};

fn parse_target_col(c: &RawColumn) -> Vec<f32> {
    c.tokens.iter().map(|s| parse_field(s)).collect()
}

/// Build the primary label vector (task-mapped/scaled) + optional multi-target matrix.
pub fn build_targets(
    frame: &Frame,
    t: &TaskSpec,
    plan: &PreprocessPlan,
) -> (Vec<f32>, Option<MultiTargets>) {
    let n = frame.n;
    let prim_raw = parse_target_col(&frame.targets[frame.primary]);
    let y: Vec<f32> = prim_raw
        .iter()
        .map(|&v| match t.kind {
            TaskKind::Regression => plan.target_scaler.transform(v),
            _ => t.class_index(v),
        })
        .collect();

    let y_multi = if frame.targets.len() > 1 && t.kind != TaskKind::Regression {
        let cols: Vec<Vec<f32>> = frame
            .targets
            .iter()
            .map(|c| {
                parse_target_col(c)
                    .iter()
                    .map(|&v| if v > 0.5 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let non_constant = |c: &Vec<f32>| c.first().is_some_and(|&f| c.iter().any(|&v| v != f));
        let keep: Vec<usize> = (0..cols.len())
            .filter(|&t| t == frame.primary || non_constant(&cols[t]))
            .collect();
        if keep.len() > 1 {
            let new_primary = keep.iter().position(|&t| t == frame.primary).unwrap();
            let mut md = Vec::with_capacity(n * keep.len());
            for &t in &keep {
                md.extend_from_slice(&cols[t]);
            }
            Some(MultiTargets {
                data: md,
                k: keep.len(),
                primary: new_primary,
                names: keep
                    .iter()
                    .map(|&t| frame.targets[t].name.clone())
                    .collect(),
            })
        } else {
            None
        }
    } else {
        None
    };
    (y, y_multi)
}
