use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnDescriptor {
    pub key: String,
    pub kind: String,
}

/// I25 closure identity carried as draft metadata until the owning residual adapter is accepted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualClosure {
    pub producer_sha256: String,
    pub fold_sha256: String,
    pub row_order_sha256: String,
    pub model_state_sha256: String,
    pub feature_selection_sha256: String,
    pub role_ancestry_sha256: String,
    pub diagnostics_charged_micros: u64,
}

/// Complete candidate context identity. Schema order is significant in its digest; pointer
/// remapping uses stable column keys and checks equivalent schemas before changing the index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextAbiV2 {
    pub schema: String,
    pub backend: String,
    pub schema_sha256: String,
    pub columns: Vec<ColumnDescriptor>,
    pub selected_column_keys: Vec<String>,
    pub feature_selection_sha256: String,
    pub fold_sha256: String,
    pub row_order_sha256: String,
    /// Required by time-lag edits; absent means row order has no declared temporal meaning.
    pub ordered_time_semantics: Option<String>,
    pub model_state_sha256: String,
    pub program_tokens: Vec<u32>,
    pub vocabulary_size: u32,
    pub max_tokens: usize,
    pub context_features: Vec<f32>,
    pub residual: ResidualClosure,
}

impl ContextAbiV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        columns: Vec<ColumnDescriptor>,
        selected_column_keys: Vec<String>,
        fold_sha256: String,
        row_order_sha256: String,
        ordered_time_semantics: Option<String>,
        model_state_sha256: String,
        program_tokens: Vec<u32>,
        vocabulary_size: u32,
        residual_producer_sha256: String,
        role_ancestry_sha256: String,
        diagnostics_charged_micros: u64,
    ) -> Result<Self> {
        let selected_column_keys = sorted_unique(selected_column_keys)?;
        let schema_sha256 = schema_digest(&columns)?;
        let feature_selection_sha256 = digest_json(&selected_column_keys)?;
        let residual = ResidualClosure {
            producer_sha256: residual_producer_sha256,
            fold_sha256: fold_sha256.clone(),
            row_order_sha256: row_order_sha256.clone(),
            model_state_sha256: model_state_sha256.clone(),
            feature_selection_sha256: feature_selection_sha256.clone(),
            role_ancestry_sha256,
            diagnostics_charged_micros,
        };
        let value = Self {
            schema: CONTEXT_ABI_SCHEMA.into(),
            backend: REFERENCE_BACKEND.into(),
            schema_sha256,
            columns,
            selected_column_keys,
            feature_selection_sha256,
            fold_sha256,
            row_order_sha256,
            ordered_time_semantics,
            model_state_sha256,
            program_tokens,
            vocabulary_size,
            max_tokens: MAX_PROGRAM_TOKENS,
            context_features: vec![0.0; CONTEXT_FEATURES],
            residual,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CONTEXT_ABI_SCHEMA,
            "unsupported context ABI schema"
        );
        ensure!(
            self.backend == REFERENCE_BACKEND,
            "unsupported edit policy backend"
        );
        ensure!(
            self.columns.len() <= MAX_SCHEMA_COLUMNS && !self.columns.is_empty(),
            "unsupported schema column count"
        );
        ensure!(
            schema_digest(&self.columns)? == self.schema_sha256,
            "context schema digest mismatch"
        );
        let mut columns = BTreeMap::new();
        for column in &self.columns {
            ensure!(
                !column.key.is_empty() && !column.kind.is_empty(),
                "empty schema column identity"
            );
            ensure!(
                columns.insert(&column.key, &column.kind).is_none(),
                "duplicate schema column key"
            );
        }
        ensure!(
            self.selected_column_keys
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "selected columns must be unique and sorted"
        );
        ensure!(
            self.selected_column_keys
                .iter()
                .all(|key| columns.contains_key(key)),
            "feature selection names a column outside the schema"
        );
        ensure!(
            digest_json(&self.selected_column_keys)? == self.feature_selection_sha256,
            "feature selection digest mismatch"
        );
        for digest in [
            &self.fold_sha256,
            &self.row_order_sha256,
            &self.model_state_sha256,
            &self.residual.producer_sha256,
            &self.residual.role_ancestry_sha256,
        ] {
            ensure!(is_digest(digest), "invalid context identity digest");
        }
        ensure!(
            self.max_tokens > 0 && self.max_tokens <= MAX_PROGRAM_TOKENS,
            "unsupported token limit"
        );
        ensure!(
            !self.program_tokens.is_empty()
                && self.program_tokens.len() <= self.max_tokens
                && self
                    .program_tokens
                    .iter()
                    .all(|token| *token < self.vocabulary_size),
            "unsupported program length or token"
        );
        ensure!(self.vocabulary_size > 0, "empty edit vocabulary");
        ensure!(
            self.context_features.len() == CONTEXT_FEATURES
                && self.context_features.iter().all(|value| value.is_finite()),
            "unsupported context feature length or value"
        );
        ensure!(
            self.residual.fold_sha256 == self.fold_sha256
                && self.residual.row_order_sha256 == self.row_order_sha256
                && self.residual.model_state_sha256 == self.model_state_sha256
                && self.residual.feature_selection_sha256 == self.feature_selection_sha256,
            "residual closure differs from the current fold/context"
        );
        if let Some(semantics) = &self.ordered_time_semantics {
            ensure!(!semantics.trim().is_empty(), "empty ordered-time semantics");
        }
        Ok(())
    }

    pub fn column_pointer(&self, index: usize) -> Result<ColumnPointer> {
        self.validate()?;
        let column = self
            .columns
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("column index outside schema"))?;
        Ok(ColumnPointer {
            schema_sha256: self.schema_sha256.clone(),
            index,
            key: column.key.clone(),
        })
    }

    pub fn remap_pointer(
        &self,
        pointer: &ColumnPointer,
        target: &ContextAbiV2,
    ) -> Result<ColumnPointer> {
        self.validate()?;
        target.validate()?;
        pointer.validate_against(self)?;
        let source: BTreeMap<_, _> = self.columns.iter().map(|c| (&c.key, &c.kind)).collect();
        let destination: BTreeMap<_, _> =
            target.columns.iter().map(|c| (&c.key, &c.kind)).collect();
        ensure!(
            source == destination,
            "column remap requires the same complete schema"
        );
        let index = target
            .columns
            .iter()
            .position(|column| column.key == pointer.key)
            .ok_or_else(|| anyhow::anyhow!("column absent from target schema"))?;
        ensure!(
            self.fold_sha256 == target.fold_sha256
                && self.row_order_sha256 == target.row_order_sha256
                && self.model_state_sha256 == target.model_state_sha256,
            "column remap changed fold, row order, or model state"
        );
        ensure!(
            self.feature_selection_sha256 == target.feature_selection_sha256
                && self.residual == target.residual
                && target
                    .selected_column_keys
                    .binary_search(&pointer.key)
                    .is_ok(),
            "column remap changed feature selection or residual closure"
        );
        Ok(ColumnPointer {
            schema_sha256: target.schema_sha256.clone(),
            index,
            key: pointer.key.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnPointer {
    pub schema_sha256: String,
    pub index: usize,
    pub key: String,
}

impl ColumnPointer {
    pub fn validate_against(&self, context: &ContextAbiV2) -> Result<()> {
        ensure!(
            self.schema_sha256 == context.schema_sha256,
            "column pointer schema mismatch"
        );
        let column = context
            .columns
            .get(self.index)
            .ok_or_else(|| anyhow::anyhow!("column pointer outside schema"))?;
        ensure!(column.key == self.key, "column pointer index/key mismatch");
        ensure!(
            context
                .selected_column_keys
                .binary_search(&self.key)
                .is_ok(),
            "column not selected by context"
        );
        Ok(())
    }
}
