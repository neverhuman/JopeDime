//! Publish and re-derive the coordinator's complete measurement/trial evidence.

use anyhow::{ensure, Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    path::{Component, Path},
};

use super::{
    receipt::{self, Receipt, Search},
    records::Measurement,
    search::CutSearch,
    Kernel2Observation, TrainingRefinement,
};
use crate::{receipt::Seeding, sha256_hex};
use feat_math::kernel::pareto::Archive;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementRecord {
    pub parent_sha256: String,
    pub refinement: TrainingRefinement,
    /// Index in observations.json, not a second unaccounted measurement.
    pub measurement: Option<usize>,
    pub measurement_not_started_reason: Option<String>,
}

mod artifact_checks;
mod evidence;
mod publish_impl;
mod refinement_checks;
mod selection_checks;
mod verification;

use artifact_checks::*;
use evidence::*;
use refinement_checks::*;
use selection_checks::*;

pub use publish_impl::publish;
pub use verification::{verify_run, verify_run_v3};
