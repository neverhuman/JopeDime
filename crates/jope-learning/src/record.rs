//! The two committed records: `jope.hwm/v1` (one high-water mark per exact cut + evaluator
//! identity) and `jope.score/v1` (one row per evaluation, never edited).

use crate::canon::{canonical_sha256, is_hex64};
use crate::metric::{Direction, Metric};
use anyhow::{ensure, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod hwm;
mod identity;
mod schema;
mod score;

#[cfg(test)]
mod tests;

use hwm::nullable_hex64;
pub use hwm::{Cost, Evaluator, Genome, HwmRecord, Provenance, Score};
pub use identity::{ComparisonContext, HwmIdentity, HWM_SCHEMA, SCORE_SCHEMA, UNBOUND};
pub use schema::{derived_hwm_schema, derived_score_schema};
pub use score::{ScoreRecord, ScoreSource};
