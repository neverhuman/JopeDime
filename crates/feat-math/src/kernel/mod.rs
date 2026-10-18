//! Versioned kernel contracts. Historical APIs and identities remain separate.
//!
//! Decoding and pipeline slot checks do not establish algorithm execution or
//! backend admission. The bounded algorithm executor is a separate release gate.
pub mod admission;
pub mod capabilities;
pub mod codec;
pub mod export;
pub mod expression_recipe;
pub mod expression_search;
pub mod finite_difference;
pub mod inverse_links;
pub mod ir;
pub mod lexicase;
pub mod linear_scaling;
pub mod numerical_policy;
pub mod ordered_axis;
pub mod pipeline;
pub mod profile;
mod program;
pub mod registry_v2;
pub mod ridge_fast;
pub mod scalar_execution;
pub mod scalar_program;
pub mod scope;
pub mod search;
pub mod stable_exponential;
pub mod telemetry;
pub mod width;
pub mod wire_v2;

// B12: explicit FeatureWithinRow programs; legacy admission is unchanged.
pub mod row_vector;
pub mod savgol;
// B18 Pareto archive: the kernel2 profile route publishes it, opt-in, as evidence only (Lane D
// kickoff, wiring after B17).
pub mod pareto;

// B13: sparse Goertzel bank kernel (v2 wire 61).
pub mod goertzel;

pub use program::Program;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Syntax(String),
    Domain(String),
    Bounds(String),
    Shape(String),
    Unsupported(String),
    Resource(String),
    NonFinite(String),
    IterationLimit,
    TimeLimit,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

// B14: causal cumulative integration (v2 wire 60).
pub mod cumulative_integration;

// B15: opt-in causal irregular-grid backward difference.
pub mod timescale_nabla;

pub mod realized_path_stats;
