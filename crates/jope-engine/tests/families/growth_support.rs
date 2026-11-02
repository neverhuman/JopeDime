use jope_parallel::{
    family_growth::{self, GrowthError, Request},
    manifest::{hash, SEED},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[path = "growth_support/assignment.rs"]
pub(crate) mod assignment;
#[path = "growth_support/cases.rs"]
pub(crate) mod cases;
#[path = "growth_support/fixture.rs"]
pub(crate) mod fixture;
#[path = "growth_support/scenarios.rs"]
pub(crate) mod scenarios;
#[path = "growth_support/serialization.rs"]
pub(crate) mod serialization;

pub(crate) use assignment::{assign, existing_case, Tamper};
pub(crate) use cases::random_case;
pub(crate) use fixture::GrowthFixture;
pub(crate) use scenarios::{agrees, build_scenario, scenario_digest, scenario_names};
pub(crate) use serialization::{py_compact, World};

const TEST_KEY: &[u8] = b"test-only-key-never-production!!!";
const PARITY_FIXTURE: &str = "tests/fixtures/family-growth/python-verify-outputs.jsonl";
const RANDOM_SEED: u64 = 20_260_926;
const RANDOM_SCENARIOS: usize = 128;
const EXISTING: [&str; 14] = [
    "live-and-sealed-aliases",
    "alias-invariance-canonical",
    "alias-invariance-alias",
    "forged-signature",
    "duplicate-decision",
    "unresolved-component",
    "stale-parent",
    "missing-units",
    "empty-prior-evidence",
    "empty-source-objects",
    "graph-merge-quarantine",
    "forged-partition",
    "tampered-source",
    "registry-parent",
];
