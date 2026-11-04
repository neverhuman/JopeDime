use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::contracts::*;
use crate::search_v3::{canonical_sha256, PredictionObjectV1};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRowV1 {
    pub split: SplitRowV1,
    /// One headerless, last-column-target legacy corpus_eval row.
    pub csv_line: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetTableV1 {
    pub task: GpTaskV1,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub rows: Vec<DatasetRowV1>,
}

impl DatasetTableV1 {
    pub fn csv_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in &self.rows {
            bytes.extend_from_slice(row.csv_line.as_bytes());
            bytes.push(b'\n');
        }
        bytes
    }

    pub fn split_rows(&self) -> Vec<SplitRowV1> {
        self.rows.iter().map(|row| row.split.clone()).collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        crate::search_v3::validate_sha256("dataset table", &self.dataset_sha256)?;
        crate::search_v3::validate_sha256("training table", &self.training_sha256)?;
        if self.rows.is_empty() || raw_sha256(&self.csv_bytes()) != self.training_sha256 {
            return Err("dataset training bytes do not match their immutable hash".into());
        }
        let mut ids = BTreeSet::new();
        for row in &self.rows {
            row.split.validate(self.task)?;
            if !ids.insert(row.split.stable_row_id.as_str()) || row.csv_line.trim().is_empty() {
                return Err("dataset rows require unique IDs and nonempty CSV lines".into());
            }
        }
        Ok(())
    }

    fn csv_for(&self, ids: &[String]) -> Result<Vec<u8>, String> {
        let positions: BTreeMap<_, _> = self
            .rows
            .iter()
            .map(|row| (row.split.stable_row_id.as_str(), row.csv_line.as_str()))
            .collect();
        let mut bytes = Vec::new();
        for id in ids {
            let line = positions
                .get(id.as_str())
                .ok_or_else(|| format!("split names unknown row {id}"))?;
            bytes.extend_from_slice(line.as_bytes());
            bytes.push(b'\n');
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestTableV1 {
    pub dataset_sha256: String,
    pub test_csv_path: String,
    pub test_sha256: String,
    pub test_row_ids: Vec<String>,
}

impl SealedTestTableV1 {
    /// Metadata-only validation. The sealed path is not opened here.
    pub fn validate_metadata(&self) -> Result<(), String> {
        crate::search_v3::validate_sha256("sealed dataset", &self.dataset_sha256)?;
        crate::search_v3::validate_sha256("sealed test", &self.test_sha256)?;
        if self.test_csv_path.trim().is_empty()
            || self.test_row_ids.is_empty()
            || self.test_row_ids.iter().any(String::is_empty)
            || self.test_row_ids.iter().collect::<BTreeSet<_>>().len() != self.test_row_ids.len()
        {
            return Err("sealed-test metadata is incomplete".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorOutcomeV1 {
    pub key: PredictionCacheKeyV1,
    pub prediction: PredictionObjectV1,
    pub runtime: RuntimeObservationV1,
    pub legacy_receipt_sha256: String,
}

impl EvaluatorOutcomeV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_prediction_binding(&self.key, &self.prediction)?;
        self.runtime.validate()?;
        crate::search_v3::validate_sha256("legacy evaluator receipt", &self.legacy_receipt_sha256)
    }
}

mod adapter;

pub use adapter::CorpusEvalAdapter;
