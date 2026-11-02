//! G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
//!
//! [`decide`] reads two documents:
//!
//! - an **arena manifest** (`jope.qualify-arena-manifest/v1`), fixed before any result exists:
//!   the incumbent and challenger, the independent families, the metric and its direction, the
//!   practical margin, the alpha budget and its predeclared looks, the number of contenders
//!   sharing each look, the bootstrap, the failure-rate limit and the L6 safety panel;
//! - the **results** of one look (`jope.qualify-arena-results/v1`), bound to the exact manifest
//!   bytes by SHA-256: one row per context with the selected program's outer metric for both
//!   contenders, attempts and failures, holds, cost, and the safety panel values.
//!
//! The independent unit is the family, and every family weighs the same (`family_weighting:
//! "equal"`, the only weighting accepted). Context effects are averaged within their family, and
//! a family with any unknown context is left out whole rather than imputed. Leaving families out
//! can favour either side, so the bounds decide nothing while more families are incomplete than
//! the manifest's `max_incomplete_families` (default 0) allows: the look is inconclusive. The
//! challenger is promote-eligible only when the one-sided lower bound of the mean family effect
//! clears the practical margin, not merely zero. That bound comes from a percentile bootstrap
//! over families at this look's alpha divided by the contenders. A percentile bootstrap
//! under-covers with few families; `min_families` is the guard, and its value belongs to the
//! G36 ruling.
//!
//! Some inputs are refused as invalid, not given a verdict: an outer-oracle row (a diagnostic,
//! never evidence), a development or meta-training roster (not independent confirmation), a
//! sealed-final roster (no roster is adopted in this build, so no seal look exists), a look out
//! of schedule, and a look after the study stopped. A negative or inconclusive arena is a
//! verdict, not an error. The report never promotes and never claims unseen-family
//! generalization.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::canon::{is_hex64, sha256_hex};

/// The arena manifest schema.
pub const MANIFEST_SCHEMA: &str = "jope.qualify-arena-manifest/v1";
/// The look results schema.
pub const RESULTS_SCHEMA: &str = "jope.qualify-arena-results/v1";
/// The report schema [`decide`] writes.
pub const REPORT_SCHEMA: &str = "jope.qualify-champion-report/v1";
/// Fewest bootstrap resamples a manifest may declare.
pub const MIN_RESAMPLES: u32 = 1000;
/// What every report says about unseen-family generalization.
pub const GENERALIZATION_CLAIM: &str = "none: no adopted sealed-family boundary in this build";

/// Tolerance for the alpha-spending sum, so that a schedule like 0.01 + 0.04 = 0.05 is exact.
const ALPHA_SLACK: f64 = 1e-12;

mod decision;
mod types;
mod validation;

pub use decision::decide;
pub use types::*;
use validation::*;
