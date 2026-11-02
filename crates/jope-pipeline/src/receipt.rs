//! `jope.pipeline-receipt/v3`: what the run did, stage by stage.
//!
//! The committed JSON Schema under `schemas/` is the public contract; the Rust
//! type derives `JsonSchema` so `tests/schema_agreement.rs` can assert the two
//! agree on every object's required set and property list, exactly as
//! `jope-datasets` does for its catalog. Validation reuses
//! `jope_datasets::schema`, the workspace's JSON Schema 2020-12 subset validator.
//!
//! v3 adds `search.seeding` (the genomes generation 0 was seeded with, their digests and the
//! parent receipt they came from). v1 and v2 receipts keep validating against their own
//! committed schemas; nothing about them is reinterpreted.
//!
//! A search ranked by a regression objective writes `jope.pipeline-objective-receipt/v1`
//! instead: v3 plus the objective (`search.objective`) and the best program's held-out
//! objective loss (`search.best.validation_objective`, per cut `best_validation_objective`).
//! A search ranked by RMSE still writes v3, byte for byte. The v3 file is frozen
//! (`docs/contracts/algorithmic-selfplay-identity.md`) and `jope.pipeline-receipt/v5` is
//! reserved for the self-play plan, hence a name of its own.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA_ID: &str = "jope.pipeline-receipt/v3";

/// The committed receipt schema (JSON Schema 2020-12).
pub const RECEIPT_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v3.schema.json");

/// The receipt of a search ranked by a regression objective: v3 plus the objective.
pub const OBJECTIVE_SCHEMA_ID: &str = "jope.pipeline-objective-receipt/v1";
pub const OBJECTIVE_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-objective-receipt.v1.schema.json");

/// What the objective receipt adds to v3, as JSON pointers into the derived schema.
const OBJECTIVE_ONLY: [&str; 4] = [
    "/$defs/Best/properties/validation_objective",
    "/$defs/CutSearch/properties/best_validation_objective",
    "/$defs/Search/properties/objective",
    "/$defs/Objective",
];

/// The schema a new receipt is written under: the objective receipt exactly when the search
/// ranked programs by an objective.
pub fn schema_id(objective_ranked: bool) -> &'static str {
    if objective_ranked {
        OBJECTIVE_SCHEMA_ID
    } else {
        RECEIPT_SCHEMA_ID
    }
}

/// The previous current schema: total-budget accounting and complete-coverage selection,
/// without seeding.
pub const V2_SCHEMA_ID: &str = "jope.pipeline-receipt/v2";
pub const V2_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v2.schema.json");

pub const LEGACY_SCHEMA_ID: &str = "jope.pipeline-receipt/v1";
pub const LEGACY_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v1.schema.json");

mod schema;
mod types;
mod validation;

pub use schema::*;
pub use types::*;
pub use validation::*;
