//! Deterministic, offline input qualification. No model or production activation is implied.

mod admission;
mod generator;
mod schema;
mod storage;

pub use admission::{verify_inputs, Verification};
pub use generator::generate_fixtures;
pub use storage::resolve_root;

#[cfg(test)]
mod tests;
