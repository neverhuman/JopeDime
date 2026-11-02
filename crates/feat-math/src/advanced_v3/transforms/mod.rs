use super::csv::RawTable;
use super::feature_block::{
    FeatureBlock, FeatureLineage, FeatureOrigin, FeatureRole, MAX_GENERATED_DENSE_WIDTH,
};
use super::input::str_param;
use super::schema::{ColumnKind, DataSchema, RowSemantics};
use super::types::{stable_hash, stratified_fold_assignments};
use crate::dsl::{ParamValue, Stage};
use crate::ops::{fit_op, quantile_sorted, FittedOp};
use crate::Rows;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

// -------------------------- fitted v3 operations --------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum V3Op {
    Legacy(FittedOp),
    MaxAbs {
        scales: Vec<f64>,
        append: bool,
    },
    YeoJohnson {
        lambdas: Vec<f64>,
        append: bool,
    },
    PiecewiseLinear {
        edges: Vec<Vec<f64>>,
        append: bool,
    },
    CorrScreen {
        indices: Vec<usize>,
        append: bool,
    },
    PlsProject {
        means: Vec<f64>,
        weights: Vec<Vec<f64>>,
        append: bool,
    },
    Lag {
        k: usize,
        append: bool,
        order: FittedTableOrder,
        tails: Vec<GroupTail>,
    },
    Roll {
        k: usize,
        append: bool,
        order: FittedTableOrder,
        tails: Vec<GroupTail>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum RowOperation {
    Lag,
    Roll,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct FiniteKey(pub(super) u64);

impl FiniteKey {
    fn new(value: f64) -> Self {
        Self(if value == 0.0 { 0 } else { value.to_bits() })
    }

    pub(super) fn value(self) -> f64 {
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct FittedTableOrder {
    pub(super) group_columns: Vec<(usize, ColumnKind)>,
    pub(super) order_columns: Vec<(usize, ColumnKind)>,
}

impl FittedTableOrder {
    fn from_schema(schema: &DataSchema) -> Result<Self> {
        let RowSemantics::Ordered { group_by, order_by } = &schema.row_semantics else {
            bail!("lag/roll require explicit ordered row semantics");
        };
        let resolve = |name: &str| {
            schema
                .columns
                .iter()
                .enumerate()
                .find_map(|(index, column)| (column.name == name).then_some((index, column.kind)))
                .with_context(|| format!("row-semantics column '{name}' is absent from schema"))
        };
        Ok(Self {
            group_columns: group_by
                .iter()
                .map(|name| resolve(name))
                .collect::<Result<Vec<_>>>()?,
            order_columns: order_by
                .iter()
                .map(|name| resolve(name))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TailRow {
    pub(super) values: Vec<f64>,
    pub(super) missing: Vec<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GroupTail {
    pub(super) key: GroupKey,
    pub(super) last_order: Vec<SemanticKey>,
    pub(super) rows: Vec<TailRow>,
    pub(super) state_sha256: String,
}

mod generated;
mod legacy;
mod operations;
mod row_ops;
mod scalar;

use generated::*;
use legacy::*;
pub(super) use operations::fit_v3_op;
use row_ops::*;
pub(super) use scalar::fit_pls_weights;
use scalar::{piecewise_row, yeo_johnson};

impl V3Op {
    pub(super) fn validate_fitted_state(&self, schema: &DataSchema) -> Result<()> {
        let (k, order, tails) = match self {
            V3Op::Lag {
                k, order, tails, ..
            }
            | V3Op::Roll {
                k, order, tails, ..
            } => (k, order, tails),
            _ => return Ok(()),
        };
        if *k == 0 {
            bail!("fitted lag/roll window must be positive");
        }
        if order != &FittedTableOrder::from_schema(schema)? {
            bail!("fitted row-operation order does not match schema");
        }
        let mut seen = BTreeSet::new();
        for tail in tails {
            validate_group_tail_keys(tail, order)?;
            if !seen.insert(&tail.key) {
                bail!("duplicate fitted row-operation group tail");
            }
        }
        Ok(())
    }

    pub(super) fn apply(&self, x: &Rows) -> Rows {
        let (mut generated, append) = match self {
            V3Op::Legacy(op) => (crate::ops::apply_op_generated(op, x), legacy_append(op)),
            V3Op::MaxAbs { scales, append } => (
                x.iter()
                    .map(|row| {
                        row.iter()
                            .zip(scales)
                            .map(|(value, scale)| value / scale)
                            .collect()
                    })
                    .collect(),
                *append,
            ),
            V3Op::YeoJohnson { lambdas, append } => (
                x.iter()
                    .map(|row| {
                        row.iter()
                            .zip(lambdas)
                            .map(|(value, lambda)| yeo_johnson(*value, *lambda))
                            .collect()
                    })
                    .collect(),
                *append,
            ),
            V3Op::PiecewiseLinear { edges, append } => (
                x.iter().map(|row| piecewise_row(row, edges)).collect(),
                *append,
            ),
            V3Op::CorrScreen { indices, append } => (
                x.iter()
                    .map(|row| indices.iter().map(|&index| row[index]).collect())
                    .collect(),
                *append,
            ),
            V3Op::PlsProject {
                means,
                weights,
                append,
            } => (
                x.iter()
                    .map(|row| {
                        weights
                            .iter()
                            .map(|weight| {
                                weight
                                    .iter()
                                    .enumerate()
                                    .map(|(index, value)| value * (row[index] - means[index]))
                                    .sum()
                            })
                            .collect()
                    })
                    .collect(),
                *append,
            ),
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                panic!("advanced-v3 row operations require a raw table")
            }
        };
        if append {
            for (row, original) in generated.iter_mut().zip(x) {
                let mut joined = original.clone();
                joined.append(row);
                *row = joined;
            }
        }
        for row in &mut generated {
            for value in row {
                if !value.is_finite() {
                    *value = 0.0;
                }
            }
        }
        generated
    }

    pub(super) fn apply_block(
        &self,
        block: &FeatureBlock,
        table: &RawTable,
    ) -> Result<FeatureBlock> {
        match self {
            V3Op::Legacy(op) => legacy_to_block(op, block),
            V3Op::MaxAbs { scales, append } => {
                let generated: Rows = block
                    .values
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(scales)
                            .map(|(value, scale)| value / scale)
                            .collect()
                    })
                    .collect();
                finish_one_to_one(block, generated, *append, "maxabs_scale")
            }
            V3Op::YeoJohnson { lambdas, append } => {
                let generated: Rows = block
                    .values
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(lambdas)
                            .map(|(value, lambda)| yeo_johnson(*value, *lambda))
                            .collect()
                    })
                    .collect();
                finish_one_to_one(block, generated, *append, "power_transform")
            }
            V3Op::PiecewiseLinear { edges, append } => {
                check_piecewise_linear_width(edges, block)?;
                let generated: Rows = block
                    .values
                    .iter()
                    .map(|row| piecewise_row(row, edges))
                    .collect();
                let sources: Vec<usize> = edges
                    .iter()
                    .enumerate()
                    .flat_map(|(source, column_edges)| {
                        std::iter::repeat_n(source, column_edges.len().saturating_sub(1))
                    })
                    .collect();
                finish_generated_dense(block, generated, sources, *append, "piecewise_linear")
            }
            V3Op::CorrScreen { indices, append } => {
                check_corr_screen_width(indices, block)?;
                let generated: Rows = block
                    .values
                    .iter()
                    .map(|row| indices.iter().map(|&j| row[j]).collect())
                    .collect();
                finish_selected(block, generated, indices, *append, "corr_screen")
            }
            V3Op::PlsProject {
                means,
                weights,
                append,
            } => {
                check_pls_project_width(means, weights, block)?;
                let generated: Rows = block
                    .values
                    .iter()
                    .map(|row| {
                        weights
                            .iter()
                            .map(|weight| {
                                weight
                                    .iter()
                                    .enumerate()
                                    .map(|(j, w)| w * (row[j] - means[j]))
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                let all_sources = (0..block.roles.len()).collect::<Vec<_>>();
                finish_generated_from_sets(
                    block,
                    generated,
                    vec![all_sources; weights.len()],
                    vec![FeatureRole::Generated; weights.len()],
                    vec![vec![false; weights.len()]; block.values.len()],
                    *append,
                    "pls_project",
                )
            }
            V3Op::Lag {
                k,
                append,
                order,
                tails,
            } => apply_row_operation(block, table, order, *k, *append, RowOperation::Lag, tails)
                .map(|(block, _)| block),
            V3Op::Roll {
                k,
                append,
                order,
                tails,
            } => apply_row_operation(block, table, order, *k, *append, RowOperation::Roll, tails)
                .map(|(block, _)| block),
        }
    }
}

#[cfg(test)]
mod tests;
