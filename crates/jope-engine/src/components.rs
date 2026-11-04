//! Frozen component workers. Every cut is reconstructed in memory, encoded once,
//! and committed with a complete input/model receipt before entering DL training.
use crate::{
    manifest::*,
    prepare::{self, Prepared},
    service::{GpJob, GpJobs},
};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use tch::{CModule, Device, Kind, Tensor};
#[derive(Serialize, Deserialize)]
pub struct Spec {
    pub module: PathBuf,
    pub module_sha256: String,
    pub sampling: PathBuf,
    pub sampling_sha256: String,
    pub parity: PathBuf,
    pub parity_sha256: String,
}
impl Spec {
    pub fn verify(&self, root: &Path) -> Result<()> {
        for (p, sha) in [
            (&self.module, &self.module_sha256),
            (&self.sampling, &self.sampling_sha256),
            (&self.parity, &self.parity_sha256),
        ] {
            ensure!(
                file_hash(&resolve(root, p)?)? == *sha,
                "frozen target dependency changed"
            );
        }
        let parity: serde_json::Value = read_json(&resolve(root, &self.parity)?)?;
        ensure!(
            parity["passed"] == true
                && parity["module_sha256"] == self.module_sha256
                && parity["sampling_sha256"] == self.sampling_sha256,
            "target parity identity mismatch"
        );
        Ok(())
    }
}

mod foundation;
mod target;

pub use foundation::foundation_rank;
pub use target::{service, target_rank};
