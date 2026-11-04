//! Rust mirror of `jope.grammars.newmath`.
//!
//! Operator vectors retain the Python registry's insertion order because grammar
//! consumers derive stable token ids from first-seen order.

mod model;
mod registry;

pub use model::{ExprSpec, GrammarSpec, OpSpec, ParamSpec};
pub use registry::grammar;

pub const ENGINE: &str = "newmath";
const DEFAULT_TASKS: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}

#[cfg(test)]
mod tests;
