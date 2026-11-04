//! Materialize measured, family-disjoint real training inputs. No encoder/model is run.
//! All private execution proofs are re-admitted before any corpus is published.
use super::{artifact, directory, immutable};
use anyhow::{bail, ensure, Context, Result};
use clap::Args;
use feat_math::GenomeIr;
use jope_datasets::family_contracts::build_training_family_partition;
use jope_learning::{
    canon::{canonical_json, canonical_sha256},
    eligibility::{freeze_teaching, Qualification},
    state::State,
};
use jope_pipeline::{
    cpu_context_bridge::{load_qualified, AdmissionExpectation, CpuContextFiles, ExecutionProof},
    execution::Artifact,
    experiment::PanelDataset,
    neural_search::FrozenRecipe,
    prepared_data::parse_development,
    search_context::TrainingIdentity,
    sha256_hex,
};
use jope_policy::context_producer::{
    self, CpuContextArtifact, EncoderIdentity, TrainingInput, TrainingRow,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

#[derive(Args)]
pub(super) struct PrepareArgs {
    #[arg(long)]
    learning_state: PathBuf,
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    request_sha256: String,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Args)]
pub(super) struct RequestArgs {
    #[arg(long)]
    learning_state: PathBuf,
    #[arg(long)]
    family_registry: PathBuf,
    #[arg(long)]
    vocabulary: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    expect_source: String,
    #[arg(long, required = true)]
    context_files: Vec<PathBuf>,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
}
include!("corpus/admission.rs");

include!("corpus/prepare.rs");

include!("corpus/prepare_request.rs");

include!("corpus/tests.rs");
