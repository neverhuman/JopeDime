//! Durable best-found teachers. Scores never cross a cut or evaluator identity.
use crate::{manifest::*, prepare::Prepared};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identity {
    pub dataset: String,
    pub family: String,
    pub partition: String,
    pub cut: usize,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub validation_sha256: String,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
}
impl Identity {
    pub fn key(&self) -> Result<String> {
        Ok(hash(&serde_json::to_vec(self)?))
    }
    pub fn observation(value: &Value) -> Result<Option<Self>> {
        if value["vocabulary_sha256"].is_null() {
            return Ok(None);
        }
        let string = |name: &str| -> Result<String> {
            Ok(value[name]
                .as_str()
                .with_context(|| format!("teacher {name} absent"))?
                .to_string())
        };
        let result = Self {
            dataset: string("dataset")?,
            family: string("family")?,
            partition: string("partition")?,
            cut: value["cut"].as_u64().context("teacher cut absent")? as usize,
            prepared_sha256: string("prepared_sha256")?,
            training_sha256: string("training_sha256")?,
            validation_sha256: string("validation_sha256")?,
            evaluator_sha256: string("evaluator_sha256")?,
            vocabulary_sha256: string("vocabulary_sha256")?,
        };
        for digest in [
            &result.prepared_sha256,
            &result.training_sha256,
            &result.validation_sha256,
            &result.evaluator_sha256,
            &result.vocabulary_sha256,
        ] {
            ensure!(
                digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid teacher identity hash"
            );
        }
        Ok(Some(result))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Champion {
    pub schema: String,
    pub identity: Identity,
    pub observation: Value,
    pub observation_sha256: String,
    pub source: PathBuf,
    pub rmse_original_units: f64,
    pub previous_event_sha256: Option<String>,
    pub previous_rmse_original_units: Option<f64>,
    pub improvement_number: usize,
    pub timestamp: u64,
}
impl Champion {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "parallel.cut_champion.v1"
                && Identity::observation(&self.observation)?.as_ref() == Some(&self.identity)
                && hash(&serde_json::to_vec(&self.observation)?) == self.observation_sha256
                && score(&self.observation)? == Some(self.rmse_original_units),
            "champion evidence changed"
        );
        Ok(())
    }
    pub fn genome(&self) -> &str {
        self.observation["genome"].as_str().unwrap()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub schema: String,
    pub request: Value,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
    pub entries: Vec<Champion>,
}

mod outbox;
pub use outbox::{record, recover_outbox};
mod snapshot;
pub use snapshot::{freeze_native_snapshot, freeze_snapshot, import, verify_cycle_snapshot};

#[cfg(test)]
mod tests;
