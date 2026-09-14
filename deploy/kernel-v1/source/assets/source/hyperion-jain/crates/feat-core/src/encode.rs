//! Fold-safe preprocessing: categorical encoding, feature scaling, and task-aware target building.
//!
//! The persisted `PreprocessPlan` is fit on TRAIN only and replayed for holdout/scoring. Every emitted
//! feature is finite in `[0,1]`; regression targets are min-max scaled with inverse metadata while
//! classification labels keep their task-correct representation.

mod fit;
mod targets;
mod transform;

use crate::ingest::ColKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use fit::{fit_encoders, fit_preprocess_plan, fit_preprocess_plan_with_cap};
pub use targets::build_targets;
pub use transform::transform;

pub const PREPROCESS_VERSION: u32 = 2;
pub const LOW_CARDINALITY_MAX: usize = 32;
pub const DEFAULT_HASH_BUCKETS: usize = 64;
pub const MIN_HASH_BUCKETS: usize = 8;
pub const DEFAULT_WIDTH_CAP: usize = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatEncoding {
    /// Legacy single-column ordinal codes. Kept for compatibility tests; not used by the CLI default.
    Ordinal,
    /// Legacy scalar train-frequency encoding.
    Frequency,
    /// Default capped hybrid: one-hot for low-cardinality columns, hash buckets for high-cardinality,
    /// deterministic bucket shrink, then scalar-frequency backup if the global cap is still exceeded.
    Hybrid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dictionary {
    pub map: HashMap<String, u32>, // token -> code (sorted-distinct order, stable)
    pub freq: Vec<f32>,            // code -> train frequency
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericScaler {
    pub min: f32,
    pub max: f32,
    pub fill: f32,
    pub constant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetScaler {
    pub min: f32,
    pub max: f32,
    pub scale: f32,
}

impl TargetScaler {
    pub fn identity() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            scale: 1.0,
        }
    }

    pub fn fit(values: &[f32]) -> Self {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in values {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
        if !min.is_finite() || !max.is_finite() || (max - min).abs() <= 1e-12 {
            return Self {
                min: if min.is_finite() { min } else { 0.0 },
                max: if max.is_finite() { max } else { 0.0 },
                scale: 1.0,
            };
        }
        Self {
            min,
            max,
            scale: max - min,
        }
    }

    pub fn transform(&self, v: f32) -> f32 {
        if !v.is_finite() || self.scale <= 1e-12 {
            0.0
        } else {
            ((v - self.min) / self.scale).clamp(0.0, 1.0)
        }
    }

    pub fn inverse(&self, v: f32) -> f32 {
        if self.scale <= 1e-12 {
            self.min
        } else {
            self.min + v * self.scale
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ColumnEncoding {
    Numeric { scaler: NumericScaler },
    Ordinal { dict: Dictionary },
    Frequency { dict: Dictionary },
    OneHot { levels: Vec<String> },
    Hash { buckets: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnPlan {
    pub input_name: String,
    pub kind: ColKind,
    pub encoding: ColumnEncoding,
    pub output_names: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissingPolicy {
    pub numeric_fill_scaled: f32,
    pub categorical_missing_token: String,
    pub categorical_other_token: String,
}

impl Default for MissingPolicy {
    fn default() -> Self {
        Self {
            numeric_fill_scaled: 0.0,
            categorical_missing_token: "__MISSING__".into(),
            categorical_other_token: "__OTHER__".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreprocessPlan {
    pub version: u32,
    pub policy: CatEncoding,
    pub low_cardinality_max: usize,
    pub default_hash_buckets: usize,
    pub width_cap: usize,
    pub missing: MissingPolicy,
    pub columns: Vec<ColumnPlan>,
    pub target_scaler: TargetScaler,
    pub issues: Vec<String>,
}

/// Backwards-compatible public name used by existing manifest fields.
pub type Encoders = PreprocessPlan;

impl PreprocessPlan {
    pub fn output_width(&self) -> usize {
        self.columns.iter().map(|c| c.output_names.len()).sum()
    }

    pub fn output_names(&self) -> Vec<String> {
        self.columns
            .iter()
            .flat_map(|c| c.output_names.iter().cloned())
            .collect()
    }

    pub fn raw_feature_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.input_name.clone()).collect()
    }
}
