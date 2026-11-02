//! Fitted Python for a feat-math pipeline: the program's Python 3 recipe from
//! `feat_math::transpile::emit`, followed by the fitted state and `fitted_model()`, which returns
//! the recipe's own `Model` with every stage and the head already set, so no caller retrains.
//!
//! The recipe is generated from the program's plan (`feat_math::spec::build_plan`), so the plan is
//! what the fitted model must match: each fitted stage must carry the planned kind in the same
//! position, every parameter the recipe bakes into its prediction calls must equal the fitted one,
//! and each state is written in the shape that kind's `*_apply`/`*_predict` function reads.
//! Anything outside the recipe's slice is refused by name. [`emit_math_pipeline`] adds the
//! `math-pipeline/v1` input contract (`prediction_version: 2`, f64) and `predict_rows`.
//!
//! Floats are written with Rust's shortest round-trip form, which CPython parses back to the same
//! bits; a non-finite value anywhere in a state is refused.

mod emit;
mod head;
mod render;
mod stage;
#[cfg(test)]
mod tests;

pub use emit::{
    emit, emit_calibrated, emit_calibrated_with_checks, emit_math_pipeline, emit_with_checks,
    emit_with_checks_and_platt, emit_with_platt, CheckRow,
};
