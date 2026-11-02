#[path = "test_support.rs"]
mod test_support;
use test_support::*;

#[allow(unused_imports)]
use jope_learning::canon::{canonical_sha256, sha256_hex};
#[allow(unused_imports)]
use jope_learning::clover::round_qual::{
    parse_parent_map, qualify_rounds, qualify_rounds_with_prereg, Endpoint, ParityRecord,
    QualTestScore, RoundQualInputs, RoundQualPlan, SelfPlayPrereg, SourceGrade, UnpinnedOr,
    MULTIPLICITY_ADJUSTMENT_STATUS, PREREG_CALLER_ASSERTED_REASON, PREREG_PINNED_BINDING_STATUS,
    PREREG_SCHEMA, PREREG_SECONDARY_STATUS, PREREG_UNPINNED_BINDING_STATUS,
    PROVENANCE_BINDING_STATUS, RATCHET_SCHEMA, ROUND_QUAL_SCHEMA,
};
#[allow(unused_imports)]
use jope_learning::clover::{Budget, ExperimentIdentity, InitialState, SourceIdentity, SCHEMA};
#[allow(unused_imports)]
use jope_learning::pools::bootstrap::Verdict;
#[allow(unused_imports)]
use jope_pipeline::test_score::{ScoredDataset, TEST_SCORE_SCHEMA_ID};
#[allow(unused_imports)]
use serde_json::{json, Value};
#[allow(unused_imports)]
use std::collections::BTreeMap;
#[allow(unused_imports)]
use std::process::Command;

#[path = "basic_a.rs"]
mod basic_a;
#[path = "basic_b.rs"]
mod basic_b;
#[path = "cli.rs"]
mod cli;
#[path = "prereg_duplicate_keys.rs"]
mod prereg_duplicate_keys;
#[path = "prereg_endpoint.rs"]
mod prereg_endpoint;
#[path = "prereg_legacy_cli.rs"]
mod prereg_legacy_cli;
#[path = "prereg_mutations.rs"]
mod prereg_mutations;
#[path = "prereg_ratchets.rs"]
mod prereg_ratchets;
#[path = "producer.rs"]
mod producer;
