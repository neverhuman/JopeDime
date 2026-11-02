use anyhow::{Context, Result};
use rand::seq::SliceRandom;

use crate::hashing::mix64;
use crate::progress::Sink;
use crate::rng::seeded;

use super::super::config::SweepPlan;
use super::super::grid::U8Grid;
use super::super::EMB_KEEP;
use super::flush::flush_patch_batch;
use super::patch::fill_one_patch;
use super::types::{PatchCoord, PatchModel};
use super::validate::{validate_permutation, validate_plan};

mod execution;
mod output;
mod raw_vector;

pub use execution::{run_composite_embedding_sweep, run_composite_sweep};
pub(super) use output::sigmoid;
pub use output::{ChunkReport, EmbeddingSweepOutput, PatchInput, SweepAccum, SweepOutput};
pub use raw_vector::run_raw_dataset_vector_sweep;
