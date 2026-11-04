use anyhow::bail;
use jope_learning::{
    canon::{canonical_json, canonical_sha256, sha256_hex},
    clover::*,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

fn digest(s: &str) -> String {
    sha256_hex(s.as_bytes())
}
fn scratch() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn identity() -> ExperimentIdentity {
    ExperimentIdentity {
        schema: SCHEMA.into(),
        source: SourceIdentity {
            commit: "a".repeat(40),
            closure_sha256: digest("full source closure"),
        },
        initial_state: InitialState::ModelFree {
            search_state_sha256: digest("initial GP population and controller state"),
            rng_state_sha256: digest("initial rng"),
        },
        evaluator_sha256: digest("evaluator"),
        parity_contract_sha256: digest("parity"),
        support_sha256: digest("program set"),
        objective_sha256: digest("objective"),
        data_roster_sha256: digest("roster"),
        seed: 0,
        budget: Budget {
            attempts: 2,
            wall_seconds: 60,
            cpu_seconds: 60,
            gpu_seconds: 0,
            memory_bytes: 1024,
            daily_bytes: DAILY_BYTES_CAP,
            daily_puts: DAILY_PUTS_CAP,
        },
        guards: ["quantile", "label_visibility", "hold_recovery"]
            .into_iter()
            .map(|s| (s.into(), digest(s)))
            .collect(),
    }
}

#[path = "clover_phase0/identity_tests.rs"]
mod identity_tests;
#[path = "clover_phase0/projection_tests.rs"]
mod projection_tests;
#[path = "clover_phase0/recovery_tests.rs"]
mod recovery_tests;
