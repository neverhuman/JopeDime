//! The private fragment table: exact types, identities, the integrity digest and the V0 checks.
//!
//! Every struct refuses unknown fields, maps are ordered and nothing is a float, so the bytes
//! of a table body are deterministic. The digest is integrity, not authenticity: anyone can
//! recompute it.

use crate::fragment::{self, HostBody};
use crate::source::{is_family_label, LabelOrigin, TaskLabel};
use crate::{is_hex64, sha256_hex, Code, NAME_PREFIX};
use anyhow::{bail, ensure, Result};
use feat_math::advanced_v3::ARTIFACT_SCHEMA;
use feat_math::heads::is_head_op;
use feat_math::kernel::pipeline::{stage_class, PipelineSlots};
use feat_math::target_transform::is_target_op;
use feat_math::{members, registry, transpile, GenomeIr};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Private label of the table format. There is no schema file for it.
pub const TABLE_SCHEMA: &str = "jope.motif-table/v1";
const EXPANSION_DOMAIN: &[u8] = b"jope-motif-expansion/v1\0";
const TABLE_DOMAIN: &[u8] = b"jope-motif-table/v1\0";

mod model;
pub use model::*;

mod operations;
pub use operations::{dataset_set, motif_order, occurrence_order, support_of};

#[cfg(test)]
mod tests;
