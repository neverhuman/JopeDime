use super::*;
use feat_math::kernel::{
    profile::PipelineArm,
    search::{crossover_programs, mutate_program, random_program},
};

pub(super) fn draw(
    parents: &[String],
    state: &mut State,
    cut: &Cut,
    options: &Options,
    slot: usize,
) -> Result<(String, &'static str)> {
    let profile = options.evaluator.profile();
    let width = cut.train_x[0].len();
    for attempt in 0..8 {
        let seed = crate::search::stream(
            options.search.rng_seed,
            cut.index,
            state.generation,
            slot,
            attempt,
        );
        if parents.is_empty() {
            return Ok((random_program(profile, width, seed)?, "random"));
        }
        let left = &parents[(seed as usize) % parents.len()];
        let (child, origin) = if parents.len() >= 2 && slot.is_multiple_of(2) {
            let right = &parents[((seed >> 17) as usize) % parents.len()];
            (
                crossover_programs(left, right, profile, width, seed)?,
                "crossover",
            )
        } else {
            (mutate_program(left, profile, width, seed)?, "mutate")
        };
        *state.arms.entry(arm_name(child.arm)).or_default() += 1;
        if child.changed && !state.seen.contains(&child.program) {
            return Ok((child.program, origin));
        }
    }
    let seed = crate::search::stream(
        options.search.rng_seed,
        cut.index,
        state.generation,
        slot,
        0xFFFF,
    );
    Ok((random_program(profile, width, seed)?, "random"))
}

fn arm_name(arm: PipelineArm) -> &'static str {
    match arm {
        PipelineArm::ParameterJitter => "parameter_jitter",
        PipelineArm::Insert => "insert",
        PipelineArm::Delete => "delete",
        PipelineArm::AdjacentSwap => "adjacent_swap",
        PipelineArm::HeadReplacement => "head_replacement",
        PipelineArm::Simplify => "simplify",
        PipelineArm::ExprGrow => "expr_grow",
        PipelineArm::ExprEdit => "expr_edit",
        PipelineArm::ExprDrop => "expr_drop",
        PipelineArm::StageCross => "stage_cross",
        PipelineArm::ExprCross => "expr_cross",
    }
}
