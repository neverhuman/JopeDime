use crate::{
    manifest::*,
    telemetry::{now, Hardware, StageHeartbeat},
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::os::unix::process::CommandExt;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

pub mod coverage;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpJob {
    pub prepared: PathBuf,
    pub prepared_sha256: String,
    pub cut: usize,
    pub output: PathBuf,
    #[serde(default)]
    pub seeds: Option<PathBuf>,
    #[serde(default)]
    pub seeds_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_view: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpJobs {
    pub schema: String,
    pub qualification: bool,
    pub jobs: Vec<GpJob>,
    #[serde(default)]
    pub vocabulary: Option<PathBuf>,
}

mod process;
pub(super) use process::{stop, OwnedSlots, STOP};

mod recovery;
pub(super) use recovery::restore_completed;

mod runner;
pub use runner::{run, run_with_wait};

mod validation;
pub(super) use validation::allowed_cpus;
pub use validation::validate_generation;

#[cfg(test)]
mod allowed_cpu_tests;
