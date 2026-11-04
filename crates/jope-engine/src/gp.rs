use crate::{
    manifest::*,
    prepare::{csv_bytes, Prepared, Preprocessing, Table},
};
use anyhow::{ensure, Context, Result};
use feat_math::{GenomeIr, Model};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

include!("gp/predictor.rs");
#[cfg(test)]
mod fit_contract_tests;
pub mod handoff;
pub mod interop;
pub mod lineage;
pub mod outer_release;
pub mod outer_release_cli;
pub mod paired;
pub mod selection_replay;
include!("gp/csv_identity.rs");
include!("gp/artifact_identity.rs");
include!("gp/fitting.rs");
include!("gp/fit_memory.rs");
include!("gp/evaluation.rs");
include!("gp/search_types.rs");
include!("gp/parent_selection.rs");
include!("gp/search_evaluation.rs");
include!("gp/search_api.rs");
include!("gp/search_inner.rs");
