//! Composite randomized sweep (candle-free): tiling, dual scatter-back, chunked streaming driver.
//! Vendored/adapted from the veox-bootstrap sweep module; extended with the per-cell confidence
//! surface and per-chunk streaming callbacks.

mod driver;
mod flush;
mod orig_v2;
mod patch;
mod types;
mod validate;

#[cfg(test)]
mod tests;

pub use driver::{
    run_composite_sweep, run_raw_dataset_vector_sweep, ChunkReport, SweepAccum, SweepOutput,
};
pub use orig_v2::{
    run_orig_v2_cell_sweep, OrigV2SweepOutput, OrigV2SweepPlan, ORIG_V2_BASE_SEED, ORIG_V2_BATCH,
    ORIG_V2_SEED_STRIDE, ORIG_V2_SHIFTS, ORIG_V2_SWEEPS, ORIG_V2_WINDOWS,
};
pub use types::{PatchModel, PatchOutputs};
