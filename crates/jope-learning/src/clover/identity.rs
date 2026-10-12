use super::{DAILY_BYTES_CAP, DAILY_PUTS_CAP, SCHEMA};
use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub commit: String,
    /// Content digest of the complete source/build dependency closure, not a short Git label.
    pub closure_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InitialState {
    ModelFree {
        /// Initial GP population, ordering and controller state, without neural weights.
        search_state_sha256: String,
        rng_state_sha256: String,
    },
    Checkpoint {
        parameters_sha256: String,
        optimizer_sha256: String,
        rng_state_sha256: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub attempts: u64,
    pub wall_seconds: u64,
    pub cpu_seconds: u64,
    /// Zero is explicitly permitted for model-free discovery.
    pub gpu_seconds: u64,
    pub memory_bytes: u64,
    pub daily_bytes: u64,
    pub daily_puts: u64,
}

/// Hashes name immutable, independently retained contracts, not caller PASS flags.
/// Lane labels are aliases stored separately; they never change scientific identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentIdentity {
    pub schema: String,
    pub source: SourceIdentity,
    pub initial_state: InitialState,
    pub evaluator_sha256: String,
    pub parity_contract_sha256: String,
    /// Binds program set, grammar, tokenizer, canonicalizer, and admission evidence.
    pub support_sha256: String,
    pub objective_sha256: String,
    /// Binds dataset/family, role, split, fold, row order, normalizer and time boundaries.
    pub data_roster_sha256: String,
    pub seed: u64,
    pub budget: Budget,
    pub guards: BTreeMap<String, String>,
}

impl ExperimentIdentity {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unknown experiment schema");
        ensure!(
            self.source.commit.len() == 40
                && self
                    .source
                    .commit
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "source requires a full lowercase Git SHA"
        );
        let mut hashes = vec![
            &self.source.closure_sha256,
            &self.evaluator_sha256,
            &self.parity_contract_sha256,
            &self.support_sha256,
            &self.objective_sha256,
            &self.data_roster_sha256,
        ];
        match &self.initial_state {
            InitialState::ModelFree {
                search_state_sha256,
                rng_state_sha256,
            } => {
                hashes.extend([search_state_sha256, rng_state_sha256]);
            }
            InitialState::Checkpoint {
                parameters_sha256,
                optimizer_sha256,
                rng_state_sha256,
            } => {
                hashes.extend([parameters_sha256, optimizer_sha256, rng_state_sha256]);
            }
        }
        ensure!(
            hashes.into_iter().all(|h| is_hex64(h)),
            "incomplete experiment identity"
        );
        for name in ["quantile", "label_visibility", "hold_recovery"] {
            ensure!(
                self.guards.get(name).is_some_and(|h| is_hex64(h)),
                "missing guard identity: {name}"
            );
        }
        ensure!(
            self.guards
                .iter()
                .all(|(name, h)| !name.trim().is_empty() && is_hex64(h)),
            "invalid guard identity"
        );
        let b = &self.budget;
        ensure!(
            b.attempts > 0 && b.wall_seconds > 0 && b.cpu_seconds > 0 && b.memory_bytes > 0,
            "incomplete resource budget"
        );
        ensure!(
            b.daily_bytes > 0
                && b.daily_bytes <= DAILY_BYTES_CAP
                && b.daily_puts > 0
                && b.daily_puts <= DAILY_PUTS_CAP,
            "storage budget exceeds Phase 0 planning caps"
        );
        Ok(())
    }

    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
