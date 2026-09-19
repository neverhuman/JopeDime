//! A byte-exact mirror of `jope-engine`'s `champions::{Identity, Champion}` (`crates/jope-engine/
//! src/champions.rs`), kept here so the learning store does not link the whole engine.
//!
//! The engine's canonicalisation is `sha256(serde_json::to_vec(&value))` with the struct field
//! order below and `serde_json::Value` maps sorted (no `preserve_order`), so re-parsing a
//! `current.json`/`event-*.json` and re-serialising compactly reproduces the event sha256 that
//! names the file. The integration tests prove this on real restore files, and
//! `hwm ingest --source champions --events <dir>` proves it on every ingested identity.

use crate::canon::{is_hex64, sha256_hex};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

pub const SCHEMA: &str = "parallel.cut_champion.v1";

/// `champions.rs:15-26`, same field order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identity {
    pub dataset: String,
    pub family: String,
    pub partition: String,
    pub cut: u64,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub validation_sha256: String,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
}

impl Identity {
    /// `Identity::key()`: the champion directory name.
    pub fn key(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// `Identity::observation()`: the identity an observation declares, if it names a vocabulary.
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
            cut: value["cut"].as_u64().context("teacher cut absent")?,
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
            ensure!(is_hex64(digest), "invalid teacher identity hash");
        }
        Ok(Some(result))
    }

    pub fn is_sealed(&self) -> bool {
        self.partition == "meta_test"
    }
}

/// `champions.rs:66-77`, same field order.
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
    pub improvement_number: u64,
    pub timestamp: u64,
}

/// `champions.rs:93-121`: the strict teacher score in original target units.
pub fn score(o: &Value) -> Result<Option<f64>> {
    if !o["failure"].is_null() || o["evaluation_bias"] != "adaptive_cut_heldout_teacher" {
        return Ok(None);
    }
    let rmse = o["metrics"]["rmse"]
        .as_f64()
        .context("teacher RMSE absent")?;
    let scale = o["target_scale"]
        .as_f64()
        .context("teacher target scale absent")?;
    let original = rmse * scale;
    ensure!(
        rmse.is_finite() && rmse >= 0.0 && scale.is_finite() && scale > 0.0 && original.is_finite(),
        "invalid teacher score"
    );
    let genome = o["genome"].as_str().context("teacher program absent")?;
    ensure!(
        sha256_hex(genome.as_bytes()) == o["program_sha256"],
        "teacher program hash changed"
    );
    ensure!(
        o["independent_evaluation"] == false,
        "teacher bias identity absent"
    );
    Ok(Some(original))
}

impl Champion {
    /// Parse the bytes of a `current.json` / `event-*.json`.
    pub fn parse(bytes: &[u8]) -> Result<Champion> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// The engine's event identity: sha256 of the compact serialisation.
    pub fn event_sha256(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// `Champion::validate()`.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && Identity::observation(&self.observation)?.as_ref() == Some(&self.identity)
                && sha256_hex(&serde_json::to_vec(&self.observation)?) == self.observation_sha256
                && score(&self.observation)? == Some(self.rmse_original_units),
            "champion evidence changed"
        );
        ensure!(
            self.previous_event_sha256.is_none() == (self.improvement_number == 0),
            "champion chain fields disagree"
        );
        Ok(())
    }

    pub fn genome(&self) -> Result<&str> {
        self.observation["genome"]
            .as_str()
            .context("champion genome absent")
    }

    pub fn program_sha256(&self) -> Result<&str> {
        self.observation["program_sha256"]
            .as_str()
            .context("champion program_sha256 absent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn observation(identity: &Identity, genome: &str) -> Value {
        json!({
            "cut": identity.cut, "dataset": identity.dataset,
            "evaluation_bias": "adaptive_cut_heldout_teacher", "evaluator_sha256": identity.evaluator_sha256,
            "failure": null, "family": identity.family, "genome": genome,
            "independent_evaluation": false, "metrics": {"rmse": 0.25},
            "partition": identity.partition, "prepared_sha256": identity.prepared_sha256,
            "program_sha256": sha256_hex(genome.as_bytes()), "target_scale": 2.0,
            "training_sha256": identity.training_sha256, "validation_sha256": identity.validation_sha256,
            "vocabulary_sha256": identity.vocabulary_sha256,
        })
    }

    #[test]
    fn key_and_event_hash_follow_the_engine_canonicalisation() {
        let identity = Identity {
            dataset: "d".into(),
            family: "f".into(),
            partition: "meta_train".into(),
            cut: 3,
            prepared_sha256: "1".repeat(64),
            training_sha256: "2".repeat(64),
            validation_sha256: "3".repeat(64),
            evaluator_sha256: "4".repeat(64),
            vocabulary_sha256: "5".repeat(64),
        };
        let expected = format!(
            r#"{{"dataset":"d","family":"f","partition":"meta_train","cut":3,"prepared_sha256":"{}","training_sha256":"{}","validation_sha256":"{}","evaluator_sha256":"{}","vocabulary_sha256":"{}"}}"#,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64)
        );
        assert_eq!(identity.key().unwrap(), sha256_hex(expected.as_bytes()));
        let obs = observation(&identity, "(pipe X (ridge alpha=1.0))");
        let champion = Champion {
            schema: SCHEMA.into(),
            identity: identity.clone(),
            observation: obs.clone(),
            observation_sha256: sha256_hex(&serde_json::to_vec(&obs).unwrap()),
            source: PathBuf::from("runs/x/observations.json"),
            rmse_original_units: 0.5,
            previous_event_sha256: None,
            previous_rmse_original_units: None,
            improvement_number: 0,
            timestamp: 1_788_856_284,
        };
        champion.validate().unwrap();
        // The pretty file form re-parses to the same compact hash.
        let pretty = serde_json::to_vec_pretty(&champion).unwrap();
        let reparsed = Champion::parse(&pretty).unwrap();
        assert_eq!(
            reparsed.event_sha256().unwrap(),
            champion.event_sha256().unwrap()
        );
        assert_eq!(
            champion.event_sha256().unwrap(),
            sha256_hex(&serde_json::to_vec(&champion).unwrap())
        );
        assert!(Identity::observation(&json!({})).unwrap().is_none());
    }
}
