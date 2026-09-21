//! Deterministic, offline input qualification plus CPU Hyperion encode. Encode needs weights.

mod admission;
mod encode;
mod generator;
mod schema;
mod storage;

pub use admission::{verify_inputs, Verification};
pub use encode::{encode_dataset, EncodeReport};
pub use generator::generate_fixtures;
pub use storage::resolve_root;

#[cfg(test)]
mod tests;
