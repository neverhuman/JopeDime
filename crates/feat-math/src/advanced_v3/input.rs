use super::csv::RawTable;
use super::feature_block::{FeatureBlock, FeatureLineage, FeatureRole};
use super::schema::{ColumnKind, DataSchema, RowSemantics};
use super::types::{stratified_fold_assignments, TaskKind};
use crate::dsl::{ParamValue, Stage};
use crate::ops::quantile_sorted;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum NumericMode {
    Median,
    Mean,
    Zero,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum CategoricalMode {
    Native,
    Frequency,
    Onehot,
    Ordered,
    Hybrid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct InputConfig {
    numeric: NumericMode,
    missing_indicator: bool,
    categorical: CategoricalMode,
    max_onehot: usize,
    min_count: usize,
    pub(super) prior: f64,
    folds: usize,
}

impl InputConfig {
    pub(super) fn compatibility() -> Self {
        Self {
            numeric: NumericMode::Zero,
            missing_indicator: false,
            categorical: CategoricalMode::Native,
            max_onehot: 16,
            min_count: 2,
            prior: 10.0,
            folds: 5,
        }
    }

    pub(super) fn from_stage(stage: Option<&Stage>) -> Result<Self> {
        let Some(stage) = stage else {
            return Ok(Self::compatibility());
        };
        let allowed = [
            "numeric",
            "missing",
            "categorical",
            "max_onehot",
            "min_count",
            "prior",
            "folds",
        ];
        for key in stage.params.keys() {
            if !allowed.contains(&key.as_str()) {
                bail!("input: unknown parameter '{key}'");
            }
        }
        let numeric = match str_param(stage, "numeric", "median")? {
            "median" => NumericMode::Median,
            "mean" => NumericMode::Mean,
            "zero" => NumericMode::Zero,
            other => bail!("input: numeric must be median|mean|zero, got '{other}'"),
        };
        let missing_indicator = match str_param(stage, "missing", "indicator")? {
            "indicator" => true,
            "none" => false,
            other => bail!("input: missing must be indicator|none, got '{other}'"),
        };
        let categorical = match str_param(stage, "categorical", "hybrid")? {
            "native" => CategoricalMode::Native,
            "frequency" => CategoricalMode::Frequency,
            "onehot" => CategoricalMode::Onehot,
            "ordered" => CategoricalMode::Ordered,
            "hybrid" => CategoricalMode::Hybrid,
            other => bail!(
                "input: categorical must be native|frequency|onehot|ordered|hybrid, got '{other}'"
            ),
        };
        let prior = stage.get_f64("prior", 10.0);
        if !prior.is_finite() || prior < 0.0 {
            bail!("input: prior must be finite and nonnegative");
        }
        Ok(Self {
            numeric,
            missing_indicator,
            categorical,
            max_onehot: stage.get_usize("max_onehot", 16).clamp(1, 256),
            min_count: stage.get_usize("min_count", 2).max(1),
            prior,
            folds: stage.get_usize("folds", 5).clamp(2, 20),
        })
    }
}

pub(super) fn str_param<'a>(stage: &'a Stage, key: &str, default: &'a str) -> Result<&'a str> {
    match stage.params.get(key) {
        None => Ok(default),
        Some(ParamValue::Str(value)) => Ok(value),
        Some(_) => bail!("{}: {key} must be a string", stage.op),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum FittedColumn {
    Numeric {
        fill: f64,
        indicator: bool,
    },
    Categorical {
        mode: CategoricalMode,
        vocabulary: Vec<String>,
        counts: Vec<usize>,
        sums: Vec<f64>,
        total: usize,
        global: f64,
        onehot: Vec<usize>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct FiniteKey(pub(super) u64);

impl FiniteKey {
    fn new(value: f64) -> Self {
        Self(if value == 0.0 { 0 } else { value.to_bits() })
    }

    fn value(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl PartialOrd for FiniteKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FiniteKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value().total_cmp(&other.value())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SemanticKey {
    Missing,
    Numeric(FiniteKey),
    Categorical(String),
}

pub(super) type GroupKey = Vec<SemanticKey>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum FittedRowOrder {
    Iid,
    Ordered {
        group_columns: Vec<(usize, ColumnKind)>,
        order_columns: Vec<(usize, ColumnKind)>,
    },
}

impl FittedRowOrder {
    fn from_schema(schema: &DataSchema) -> Self {
        match &schema.row_semantics {
            RowSemantics::Iid => Self::Iid,
            RowSemantics::Ordered { group_by, order_by } => {
                let resolve = |name: &str| {
                    schema
                        .columns
                        .iter()
                        .enumerate()
                        .find_map(|(index, column)| {
                            (column.name == name).then_some((index, column.kind))
                        })
                        .expect("validated row-semantics column")
                };
                Self::Ordered {
                    group_columns: group_by.iter().map(|name| resolve(name)).collect(),
                    order_columns: order_by.iter().map(|name| resolve(name)).collect(),
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct HistoricalCategoryState {
    pub(super) category: usize,
    pub(super) count: usize,
    pub(super) sum: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct HistoricalGroupState {
    pub(super) key: GroupKey,
    pub(super) count: usize,
    pub(super) sum: f64,
    pub(super) categories: Vec<HistoricalCategoryState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct HistoricalTargetState {
    pub(super) count: usize,
    pub(super) sum: f64,
    pub(super) groups: Vec<HistoricalGroupState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct FittedInput {
    pub(super) config: InputConfig,
    pub(super) columns: Vec<FittedColumn>,
    pub(super) row_order: FittedRowOrder,
    pub(super) historical: Vec<Option<HistoricalTargetState>>,
}

mod fit;
mod historical;
#[cfg(test)]
mod tests;
mod transform;

use historical::*;
