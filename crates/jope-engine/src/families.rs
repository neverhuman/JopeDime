//! Immutable qualification family assignments. This never rewrites an audit or a checkpoint.
use crate::manifest::{
    file_hash, hash, read_json, write_json, Admission, AuditReport, Decision, Evidence, SEED,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
};

const SCHEMA: &str = "parallel.family_registry.v1";

mod freeze;
mod growth;
mod model;
mod validation;

pub(super) use freeze::field;
pub use freeze::freeze;
pub use growth::{binding, freeze_growth};
pub use model::*;
pub(super) use validation::verified_growth;
