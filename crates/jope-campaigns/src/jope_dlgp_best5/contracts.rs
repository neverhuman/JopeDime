use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use serde::{Deserialize, Serialize};

use crate::jope_generalized_gp::{
    raw_sha256, task_head_valid, GenomeV1, GpTaskV1, ProposalFreezeV1,
};
use crate::search_v3::{canonical_sha256, validate_sha256, PredictionObjectV1};

pub const CAMPAIGN_ID: &str = "jope-dlgp-best5";
pub const PROJECT_ID: &str = "jope-dlgp-best5";
pub const FROZEN_CHECKPOINT_SHA256: &str =
    "e0aef7d74afd75d3003bdb049fa0bac06653f66408c7bfc458b1aba5ae1b98cb";
pub const FROZEN_INPUT_MANIFEST_SHA256: &str =
    "9e47eb9a10f6d92cab839093f3817eb1e0a17c1ac83f63a2ed3633af585bc6c0";
pub const FROZEN_PANEL_SHA256: &str =
    "b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75";
pub const BOOTSTRAP_SEED: u64 = 20_260_807;
pub const SEARCH_SEEDS: [u64; 2] = [101, 202];
pub const POPULATION: usize = 80;
pub const GENERATIONS: usize = 8;
pub const SEARCH_SPLITS: usize = 8;
pub const LOGICAL_OPERATION_CAP: u32 = 1_100;
pub const PREDICTED_CPU_SECOND_CAP: u64 = 7_200;
pub const PILOT_DATASETS: usize = 20;
pub const CONFIRMATION_DATASETS: usize = 22;
pub const ALL_DATASETS: usize = 42;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenFileV1 {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl FrozenFileV1 {
    pub fn validate(&self, name: &str) -> Result<(), String> {
        nonempty(&format!("{name} path"), &self.path)?;
        validate_sha256(name, &self.sha256)?;
        if self.bytes == 0 {
            return Err(format!("{name} byte length must be positive"));
        }
        Ok(())
    }

    pub fn verify(&self, name: &str) -> Result<(), String> {
        self.validate(name)?;
        let bytes =
            fs::read(&self.path).map_err(|error| format!("read {name} {}: {error}", self.path))?;
        if bytes.len() as u64 != self.bytes || raw_sha256(&bytes) != self.sha256 {
            return Err(format!("{name} bytes differ from their freeze"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSourceV1 {
    pub schema: String,
    pub sotarena_root: String,
    pub input_manifest: FrozenFileV1,
    pub panel: FrozenFileV1,
    pub observation_snapshot: FrozenFileV1,
    pub six_win_manifest: FrozenFileV1,
    pub registry: FrozenFileV1,
    pub evaluator: FrozenFileV1,
    pub math_policy: FrozenFileV1,
    pub backend_policy: FrozenFileV1,
    pub proposal_generator: FrozenFileV1,
    pub proposal_runtime: FrozenFileV1,
    pub checkpoint: FrozenCheckpointV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SixWinManifestV1 {
    pub schema: String,
    pub dataset_ids: Vec<String>,
}

impl SixWinManifestV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SixWinManifest.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self.dataset_ids.len() != 6 || !self.dataset_ids.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err("six-win manifest must contain six sorted unique dataset IDs".into());
        }
        for id in &self.dataset_ids {
            short_dataset_id(id)?;
        }
        Ok(())
    }
}

impl CampaignSourceV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5CampaignSource.v1";

    pub fn verify(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        nonempty("SOTArena root", &self.sotarena_root)?;
        let root = std::path::Path::new(&self.sotarena_root);
        if !root.is_dir() {
            return Err("frozen SOTArena root is not a directory".into());
        }
        for (name, artifact) in [
            ("input manifest", &self.input_manifest),
            ("panel", &self.panel),
            ("observation snapshot", &self.observation_snapshot),
            ("six-win manifest", &self.six_win_manifest),
            ("registry", &self.registry),
            ("evaluator", &self.evaluator),
            ("math policy", &self.math_policy),
            ("backend policy", &self.backend_policy),
            ("proposal generator", &self.proposal_generator),
            ("proposal runtime", &self.proposal_runtime),
        ] {
            artifact.verify(name)?;
        }
        if self.input_manifest.sha256 != FROZEN_INPUT_MANIFEST_SHA256
            || self.panel.sha256 != FROZEN_PANEL_SHA256
        {
            return Err("source substituted the frozen input manifest or panel".into());
        }
        self.checkpoint.verify_artifact()
    }
}

pub const PILOT_DATASET_IDS: [&str; PILOT_DATASETS] = [
    "1cb97b1d7314d43d",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "43b4cc3ab0ebbd97",
    "55b144de19659467",
    "5f91a00bd3b66742",
    "79e600bc5fcd5da7",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b78d323566930bd7",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "fe3cc716a05d036f",
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "1a803b27c2400dc4",
];

const FROZEN_INPUT_MANIFEST_JSON: &str = include_str!("../../manifests/input-manifest.v1.json");

#[derive(Deserialize)]
struct EmbeddedInputManifestV1 {
    datasets: Vec<EmbeddedInputDatasetV1>,
}

#[derive(Deserialize)]
struct EmbeddedInputDatasetV1 {
    dataset_hash: String,
    features: u32,
    meta_sha256: String,
    train_rows: u32,
    train_sha256: String,
    test_sha256: String,
}

fn embedded_input_manifest() -> Result<EmbeddedInputManifestV1, String> {
    if raw_sha256(FROZEN_INPUT_MANIFEST_JSON.as_bytes()) != FROZEN_INPUT_MANIFEST_SHA256 {
        return Err("compiled frozen input manifest bytes drifted".into());
    }
    serde_json::from_str(FROZEN_INPUT_MANIFEST_JSON)
        .map_err(|error| format!("decode compiled frozen input manifest: {error}"))
}

fn schema(actual: &str, expected: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("schema must be {expected}, found {actual}"))
    }
}

fn finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn nonempty(name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must not be empty"))
    } else {
        Ok(())
    }
}

fn short_dataset_id(value: &str) -> Result<(), String> {
    if value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "dataset ID must be 16 lowercase hex characters: {value}"
        ))
    }
}

mod evidence;
mod freeze;
mod search;
mod selection;

pub use evidence::*;
pub use freeze::*;
pub use search::*;
pub use selection::*;
