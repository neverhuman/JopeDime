use super::*;

mod classical;
mod neural;
mod statistics;

pub(super) use classical::classical_summary;
pub(super) use neural::neural_explicit_stats;
use statistics::*;

pub(super) fn torch_unit_point(index: usize, steps: usize) -> f32 {
    let step = 1.0f32 / (steps - 1).max(1) as f32;
    if index < steps.div_ceil(2) {
        step * index as f32
    } else {
        1.0 - step * (steps - index - 1) as f32
    }
}
