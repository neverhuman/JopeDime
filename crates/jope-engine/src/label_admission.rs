//! Current native evaluation identity, checked before any program deduplication.
use crate::{
    corpus::ContextRow,
    manifest::*,
    prepare::{Prepared, Preprocessing},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub const PROTOCOL: &str = "featmath_training_fold_normalized_v1";
pub const RESOURCE: &str = "single_candidate_30s_8gib_v1";
pub const POLICY_SCHEMA: &str = "parallel.label_admission.v2";
pub const LABEL_POLICY: &str = "strict_midrank_v4";
pub const COMPLETION_SCHEMA: &str = "parallel.clean_completion.v3";

mod artifacts;
mod authority;
mod census;
mod policy;

pub use artifacts::*;
pub use authority::*;
pub use census::*;
pub(crate) use policy::validate_preverified;
pub use policy::*;

#[cfg(test)]
mod tests;
