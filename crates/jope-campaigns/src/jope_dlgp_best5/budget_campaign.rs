//! Contracts for the budget-first campaign layered on the existing Best5
//! implementation.  These types do not add or rename public Best5 commands.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use super::{FrozenPredictorComponentV1, LicenseTrackV1, FROZEN_CHECKPOINT_SHA256, SEARCH_SEEDS};
use crate::hyperion_gate::{FeatureViewKindV1, GateThresholdsV1};
use crate::jope_generalized_gp::raw_sha256;
use crate::search_v3::{canonical_sha256, validate_sha256};

mod budget;
mod policy;

pub use budget::*;
pub use policy::*;
use policy::{finite_positive, sorted_unique};
