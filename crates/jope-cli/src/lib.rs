//! Offline fixtures, CPU Hyperion encode, and feat-math fit/emit/search.

mod admission;
mod encode;
mod generator;
mod kernel;
mod schema;
mod storage;

pub use admission::{verify_inputs, Verification};
pub use encode::{encode_dataset, EncodeReport};
pub use generator::generate_fixtures;
pub use kernel::{
    emit_genome, fit_dataset, search_dataset, EmitReport, FitReport, SearchReport, DEFAULT_GENOME,
};
pub use storage::resolve_root;

#[cfg(test)]
mod tests;
