//! Composite randomized sweep (candle-free): tiling, dual scatter-back, chunked streaming driver.
//! Vendored/adapted from the veox-bootstrap sweep module; extended with the per-cell confidence
//! surface and per-chunk streaming callbacks.

mod driver;
mod flush;
mod patch;
mod types;
mod validate;

#[cfg(test)]
mod tests;

pub use driver::{run_composite_sweep, ChunkReport, SweepAccum, SweepOutput};
pub use types::{PatchModel, PatchOutputs};
