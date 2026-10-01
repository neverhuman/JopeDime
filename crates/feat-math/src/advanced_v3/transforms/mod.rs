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

fn stage_append(stage: &Stage) -> bool {
    matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1
}

pub(super) fn fit_v3_op(
    stage: &Stage,
    block: &FeatureBlock,
    y: &[f64],
    table: &RawTable,
    schema: &DataSchema,
) -> Result<(V3Op, FeatureBlock)> {
    let append = stage_append(stage);
    match stage.op.as_str() {
        "maxabs_scale" => {
            let cols = block.values.first().map_or(0, Vec::len);
            let mut scales = vec![1.0f64; cols];
            for row in &block.values {
                for (j, value) in row.iter().enumerate() {
                    scales[j] = scales[j].max(value.abs());
                }
            }
            let op = V3Op::MaxAbs { scales, append };
            let train = op.apply_block(block, table)?;
            Ok((op, train))
        }
        "power_transform" => {
            if str_param(stage, "kind", "yeo_johnson")? != "yeo_johnson" {
                bail!("power_transform: kind must be yeo_johnson");
            }
            let cols = block.values.first().map_or(0, Vec::len);
            let lambdas = (0..cols)
                .map(|j| fit_yeo_lambda(block.values.iter().map(|row| row[j])))
                .collect();
            let op = V3Op::YeoJohnson { lambdas, append };
            let train = op.apply_block(block, table)?;
            Ok((op, train))
        }
        "piecewise_linear" => {
            let bins = stage.get_usize("bins", 8);
            if ![8, 16, 32].contains(&bins) {
                bail!("piecewise_linear: bins must be 8|16|32");
            }
            let cols = block.values.first().map_or(0, Vec::len);
            if cols.saturating_mul(bins) > MAX_GENERATED_DENSE_WIDTH {
                bail!(
                    "piecewise_linear derived width {} exceeds {}",
                    cols.saturating_mul(bins),
                    MAX_GENERATED_DENSE_WIDTH
                );
            }
            let mut edges = Vec::with_capacity(cols);
            for j in 0..cols {
                let mut sorted: Vec<f64> = block.values.iter().map(|row| row[j]).collect();
                sorted.sort_by(f64::total_cmp);
                edges.push(
                    (0..=bins)
                        .map(|b| quantile_sorted(&sorted, b as f64 / bins as f64))
                        .collect(),
                );
            }
            let op = V3Op::PiecewiseLinear { edges, append };
            let train = op.apply_block(block, table)?;
            Ok((op, train))
        }
        "corr_screen" => {
            let requested = stage.get_usize("k", 32);
            if ![32, 64, 128, 256, 512].contains(&requested) {
                bail!("corr_screen: k must be 32|64|128|256|512");
            }
            let cols = block.values.first().map_or(0, Vec::len);
            let mut ranked: Vec<(usize, f64)> = (0..cols)
                .map(|j| {
                    let column: Vec<f64> = block.values.iter().map(|row| row[j]).collect();
                    (j, crate::metrics::pearson(&column, y).abs())
                })
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut indices: Vec<usize> = ranked
                .into_iter()
                .take(requested.min(cols))
                .map(|entry| entry.0)
                .collect();
            indices.sort_unstable();
            let op = V3Op::CorrScreen { indices, append };
            let train = op.apply_block(block, table)?;
            Ok((op, train))
        }
        "pls_project" => {
            let k = stage.get_usize("k", 8);
            if ![2, 4, 8, 16, 32].contains(&k) {
                bail!("pls_project: k must be 2|4|8|16|32");
            }
            let (means, weights) = fit_pls_weights(&block.values, y, k);
            let op = V3Op::PlsProject {
                means,
                weights,
                append,
            };
            let train = op.apply_block(block, table)?;
            Ok((op, train))
        }
        "lag" | "roll" => {
            let k = stage.get_usize("k", 1);
            if k == 0 {
                bail!("{}: k must be positive", stage.op);
            }
            let order = FittedTableOrder::from_schema(schema)?;
            let kind = if stage.op == "lag" {
                RowOperation::Lag
            } else {
                RowOperation::Roll
            };
            let (train, tails) = apply_row_operation(block, table, &order, k, append, kind, &[])?;
            let op = match kind {
                RowOperation::Lag => V3Op::Lag {
                    k,
                    append,
                    order,
                    tails,
                },
                RowOperation::Roll => V3Op::Roll {
                    k,
                    append,
                    order,
                    tails,
                },
            };
            Ok((op, train))
        }
        "target_bin_encode" if stage.params.contains_key("oof") => {
            let folds = stage.get_usize("oof", 5);
            if folds != 5 {
                bail!("target_bin_encode: oof currently must be 5");
            }
            let full = fit_op(stage, &block.values, y)?;
            let mut train = vec![Vec::<f64>::new(); block.values.len()];
            let fold_ids = stratified_fold_assignments(y, folds);
            for fold in 0..folds {
                let train_idx: Vec<usize> = (0..block.values.len())
                    .filter(|&r| fold_ids[r] != fold)
                    .collect();
                let valid_idx: Vec<usize> = (0..block.values.len())
                    .filter(|&r| fold_ids[r] == fold)
                    .collect();
                if train_idx.is_empty() || valid_idx.is_empty() {
                    continue;
                }
                let x_fit: Rows = train_idx.iter().map(|&r| block.values[r].clone()).collect();
                let y_fit: Vec<f64> = train_idx.iter().map(|&r| y[r]).collect();
                let x_valid: Rows = valid_idx.iter().map(|&r| block.values[r].clone()).collect();
                let fold_op = fit_op(stage, &x_fit, &y_fit)?;
                let transformed = crate::ops::apply_op_generated(&fold_op, &x_valid);
                for (&row, value) in valid_idx.iter().zip(transformed) {
                    train[row] = value;
                }
            }
            let train = legacy_generated_to_block(&full, block, train)?;
            Ok((V3Op::Legacy(full), train))
        }
        _ => {
            let op = fit_op(stage, &block.values, y)?;
            let train = legacy_to_block(&op, block)?;
            Ok((V3Op::Legacy(op), train))
        }
    }
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

const GROUP_TAIL_STATE_VERSION: &str = "advanced-v3-group-tail-v1";

fn group_tail_hash(tail: &GroupTail) -> Result<String> {
    stable_hash(&(
        GROUP_TAIL_STATE_VERSION,
        &tail.key,
        &tail.last_order,
        &tail.rows,
    ))
}

fn validate_stored_key(key: &SemanticKey, kind: ColumnKind) -> bool {
    match key {
        SemanticKey::Missing => true,
        SemanticKey::Numeric(value) => {
            let value_f64 = value.value();
            kind == ColumnKind::Numeric
                && value_f64.is_finite()
                && (value_f64 != 0.0 || value.0 == 0)
        }
        SemanticKey::Categorical(value) => {
            kind == ColumnKind::Categorical && !category_is_missing(value)
        }
    }
}

fn validate_group_tail_keys(tail: &GroupTail, order: &FittedTableOrder) -> Result<()> {
    if tail.key.len() != order.group_columns.len()
        || tail.last_order.len() != order.order_columns.len()
    {
        bail!("fitted row-operation tail key width mismatch");
    }
    if tail
        .key
        .iter()
        .zip(&order.group_columns)
        .any(|(key, (_, kind))| !validate_stored_key(key, *kind))
        || tail
            .last_order
            .iter()
            .zip(&order.order_columns)
            .any(|(key, (_, kind))| !validate_stored_key(key, *kind))
    {
        bail!("fitted row-operation tail key is invalid for declared column kind");
    }
    if tail.state_sha256 != group_tail_hash(tail)? {
        bail!("fitted row-operation tail state hash mismatch");
    }
    Ok(())
}

fn sanitize_with_masks(values: &mut Rows, masks: &[Vec<bool>]) -> Result<()> {
    if values.len() != masks.len() {
        bail!("transformed feature row/mask count mismatch");
    }
    for (row, mask) in values.iter_mut().zip(masks) {
        if row.len() != mask.len() {
            bail!("transformed feature row/mask width mismatch");
        }
        for (value, missing) in row.iter_mut().zip(mask) {
            if *missing || !value.is_finite() {
                *value = 0.0;
            }
        }
    }
    Ok(())
}

fn lineage_with_operation(lineage: &FeatureLineage, operation: &str) -> FeatureLineage {
    let mut lineage = lineage.clone();
    lineage.operations.push(operation.to_string());
    lineage
}

fn source_lineage(
    block: &FeatureBlock,
    feature_indices: &[usize],
    role: FeatureRole,
    operation: &str,
) -> Result<FeatureLineage> {
    let mut raw_sources = BTreeSet::new();
    let mut operations = Vec::new();
    for &index in feature_indices {
        let lineage = block
            .lineage
            .get(index)
            .with_context(|| format!("generated feature source {index} is out of bounds"))?;
        raw_sources.extend(lineage.source_columns.iter().copied());
        for prior in &lineage.operations {
            if !operations.contains(prior) {
                operations.push(prior.clone());
            }
        }
    }
    operations.push(operation.to_string());
    let sparse = matches!(
        role,
        FeatureRole::MissingIndicator | FeatureRole::CategoricalOneHot
    );
    Ok(FeatureLineage {
        source_columns: raw_sources.into_iter().collect(),
        operations,
        origin: if sparse {
            FeatureOrigin::GeneratedSparse
        } else {
            FeatureOrigin::GeneratedDense
        },
    })
}

fn finish_generated_from_sets(
    block: &FeatureBlock,
    mut generated: Rows,
    feature_sources: Vec<Vec<usize>>,
    roles: Vec<FeatureRole>,
    masks: Vec<Vec<bool>>,
    append: bool,
    operation: &str,
) -> Result<FeatureBlock> {
    if feature_sources.len() != roles.len() {
        bail!("generated feature source/role width mismatch");
    }
    sanitize_with_masks(&mut generated, &masks)?;
    let lineage = feature_sources
        .iter()
        .zip(&roles)
        .map(|(sources, role)| source_lineage(block, sources, *role, operation))
        .collect::<Result<Vec<_>>>()?;

    let generated_width = roles.len();
    if generated.iter().any(|row| row.len() != generated_width)
        || masks.iter().any(|row| row.len() != generated_width)
    {
        bail!("generated feature rows are ragged");
    }
    if generated.len() != block.values.len() || masks.len() != block.values.len() {
        bail!("generated feature row count mismatch");
    }

    let base_width = if append { block.roles.len() } else { 0 };
    let mut values = if append {
        block.values.clone()
    } else {
        vec![Vec::new(); generated.len()]
    };
    for (row, added) in values.iter_mut().zip(generated) {
        row.extend(added);
    }
    let mut output_roles = if append {
        block.roles.clone()
    } else {
        Vec::new()
    };
    output_roles.extend(roles);
    let mut output_masks = if append {
        block.missing_masks.clone()
    } else {
        vec![Vec::new(); masks.len()]
    };
    for (mask, added) in output_masks.iter_mut().zip(masks) {
        mask.extend(added);
    }
    let mut output_lineage = if append {
        block.lineage.clone()
    } else {
        Vec::new()
    };
    output_lineage.extend(lineage.clone());
    let mut sparse_columns = if append {
        block.sparse_columns.clone()
    } else {
        Vec::new()
    };
    sparse_columns.extend(lineage.iter().enumerate().filter_map(|(index, item)| {
        (item.origin == FeatureOrigin::GeneratedSparse).then_some(base_width + index)
    }));
    FeatureBlock::new(
        values,
        output_roles,
        output_masks,
        sparse_columns,
        output_lineage,
        block.raw_source_width,
    )
}

fn finish_one_to_one(
    block: &FeatureBlock,
    mut generated: Rows,
    append: bool,
    operation: &str,
) -> Result<FeatureBlock> {
    if append {
        let sources = (0..block.roles.len()).map(|index| vec![index]).collect();
        return finish_generated_from_sets(
            block,
            generated,
            sources,
            block.roles.clone(),
            block.missing_masks.clone(),
            true,
            operation,
        );
    }
    sanitize_with_masks(&mut generated, &block.missing_masks)?;
    FeatureBlock::new(
        generated,
        block.roles.clone(),
        block.missing_masks.clone(),
        block.sparse_columns.clone(),
        block
            .lineage
            .iter()
            .map(|lineage| lineage_with_operation(lineage, operation))
            .collect(),
        block.raw_source_width,
    )
}

fn finish_selected(
    block: &FeatureBlock,
    mut generated: Rows,
    indices: &[usize],
    append: bool,
    operation: &str,
) -> Result<FeatureBlock> {
    let masks: Vec<Vec<bool>> = block
        .missing_masks
        .iter()
        .map(|mask| indices.iter().map(|&index| mask[index]).collect())
        .collect();
    if append {
        return finish_generated_from_sets(
            block,
            generated,
            indices.iter().map(|&index| vec![index]).collect(),
            indices.iter().map(|&index| block.roles[index]).collect(),
            masks,
            true,
            operation,
        );
    }
    sanitize_with_masks(&mut generated, &masks)?;
    let roles = indices.iter().map(|&index| block.roles[index]).collect();
    let lineage: Vec<FeatureLineage> = indices
        .iter()
        .map(|&index| lineage_with_operation(&block.lineage[index], operation))
        .collect();
    let sparse_columns = lineage
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            (item.origin == FeatureOrigin::GeneratedSparse).then_some(index)
        })
        .collect();
    FeatureBlock::new(
        generated,
        roles,
        masks,
        sparse_columns,
        lineage,
        block.raw_source_width,
    )
}

fn finish_generated_dense(
    block: &FeatureBlock,
    generated: Rows,
    source_indices: Vec<usize>,
    append: bool,
    operation: &str,
) -> Result<FeatureBlock> {
    let generated_width = source_indices.len();
    let masks: Vec<Vec<bool>> = block
        .missing_masks
        .iter()
        .map(|mask| source_indices.iter().map(|&source| mask[source]).collect())
        .collect();
    finish_generated_from_sets(
        block,
        generated,
        source_indices
            .into_iter()
            .map(|source| vec![source])
            .collect(),
        vec![FeatureRole::Generated; generated_width],
        masks,
        append,
        operation,
    )
}

fn legacy_append(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::Polar { append }
        | FittedOp::Lag { append, .. }
        | FittedOp::Roll { append, .. }
        | FittedOp::FftMag { append, .. }
        | FittedOp::RowSpectral { append }
        | FittedOp::RowDiffStats { append }
        | FittedOp::SvdProject { append, .. }
        | FittedOp::Nystrom { append, .. }
        | FittedOp::MapExpr { append, .. }
        | FittedOp::DeriveExpr { append, .. }
        | FittedOp::Symlog { append, .. }
        | FittedOp::SelectCorr { append, .. }
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

fn legacy_generated_to_block(
    op: &FittedOp,
    block: &FeatureBlock,
    generated: Rows,
) -> Result<FeatureBlock> {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::WinsorClip { append, .. } => {
            finish_one_to_one(block, generated, *append, "legacy_v3_transform")
        }
        FittedOp::SelectCorr {
            indices, append, ..
        } => finish_selected(block, generated, indices, *append, "select_corr"),
        FittedOp::Impute {
            indicator_cols,
            append,
            ..
        } => finish_impute(block, generated, indicator_cols, *append),
        FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
            bail!("legacy row operations are not valid inside advanced-v3")
        }
        _ => {
            let generated_width = generated.first().map_or(0, Vec::len);
            let sources = (0..block.roles.len()).collect::<Vec<_>>();
            finish_generated_from_sets(
                block,
                generated,
                vec![sources; generated_width],
                vec![FeatureRole::Generated; generated_width],
                vec![vec![false; generated_width]; block.values.len()],
                legacy_append(op),
                "legacy_v3_transform",
            )
        }
    }
}

fn legacy_to_block(op: &FittedOp, block: &FeatureBlock) -> Result<FeatureBlock> {
    check_legacy_width(op, block)?;
    let generated = crate::ops::apply_op_generated(op, &block.values);
    legacy_generated_to_block(op, block, generated)
}

/// The strict width boundary of a legacy stage in a schema-bound pipeline
/// (kernel2 WP-1.7): its fitted state must agree with itself, accept the
/// block's width and give output rows that a `Vec` can hold, or the stage is
/// refused with an error where `apply_op_generated` would panic or read state
/// fitted for another column. State fitted on the block always passes. Every
/// row of a block is `roles.len()` wide (`FeatureBlock::validate`).
///
/// A block with no rows has no width to compare. Nothing is read, and the
/// generated-width arm of `legacy_generated_to_block` takes its width from the
/// first row, so after such a stage the block is narrower than at fit time (no
/// column, or only the appended input). Without rows, the fitted state is
/// checked, and beyond it only the fitted column indices at which finishing the
/// stage looks the block up: `select_corr`'s indices, and `impute`'s indicators
/// when it does not append. Finishing panicked there. No output row is
/// allocated either, so there is no overflow to refuse. For state that
/// `fit_op` produced, every other outcome without rows is what it was before
/// these checks.
fn check_legacy_width(op: &FittedOp, block: &FeatureBlock) -> Result<()> {
    use crate::kernel::width;
    let name = width::stage_name(op);
    if let Err(reason) = width::check_fitted_state(op) {
        bail!("legacy stage {name} has inconsistent fitted state: {reason}");
    }
    let got = block.roles.len();
    if block.values.is_empty() {
        let looked_up: &[usize] = match op {
            FittedOp::SelectCorr { indices, .. } => indices,
            FittedOp::Impute {
                indicator_cols,
                append: false,
                ..
            } => indicator_cols,
            _ => &[],
        };
        if let Some(index) = looked_up.iter().find(|&&index| index >= got) {
            bail!(
                "legacy stage {name} reads column {index}, but the feature block has {got} columns"
            );
        }
        return Ok(());
    }
    let required = width::required_input_width(op);
    if !required.admits(got) {
        bail!(
            "legacy stage {name} needs {required} input columns, but the feature block has {got}"
        );
    }
    // The state and the width are admitted above, so only an output row too
    // wide for any `Vec` is left for `output_width` to refuse.
    if width::output_width(op, got).is_err() {
        bail!("legacy stage {name} output width overflows at {got} input columns");
    }
    Ok(())
}

/// `corr_screen` reads every row at its fitted column indices, and finishing
/// the selection looks the block up at them, with rows or without.
fn check_corr_screen_width(indices: &[usize], block: &FeatureBlock) -> Result<()> {
    let width = block.roles.len();
    if let Some(index) = indices.iter().find(|&&index| index >= width) {
        bail!("corr_screen reads column {index}, but the feature block has {width} columns");
    }
    Ok(())
}

/// `pls_project` reads the row and the fitted means at every index of a weight.
/// Without rows it reads nothing, and the block may be narrower than at fit
/// time (see `check_legacy_width`), so there is nothing to check.
fn check_pls_project_width(
    means: &[f64],
    weights: &[Vec<f64>],
    block: &FeatureBlock,
) -> Result<()> {
    if block.values.is_empty() {
        return Ok(());
    }
    let width = block.roles.len();
    if let Some(weight) = weights
        .iter()
        .find(|weight| weight.len() > width.min(means.len()))
    {
        bail!(
            "pls_project weight spans {} columns, but the feature block has {width} and the fitted means {}",
            weight.len(),
            means.len()
        );
    }
    Ok(())
}

/// `piecewise_linear` finishes by gathering the block's missing mask at the column of every fitted
/// edge list that has an interval (two edges or more), so each such column must be in the block;
/// `piecewise_row` itself zips the row with the edge lists and reads nothing past either (kernel2
/// X3, closing the gap WP-1.7 recorded). Without rows nothing is read or gathered, and the block
/// may be narrower than at fit time (see `check_legacy_width`), so there is nothing to check.
fn check_piecewise_linear_width(edges: &[Vec<f64>], block: &FeatureBlock) -> Result<()> {
    if block.values.is_empty() {
        return Ok(());
    }
    let width = block.roles.len();
    if let Some((column, _)) = edges
        .iter()
        .enumerate()
        .skip(width)
        .find(|(_, column_edges)| column_edges.len() > 1)
    {
        bail!("piecewise_linear reads column {column}, but the feature block has {width} columns");
    }
    Ok(())
}

fn finish_impute(
    block: &FeatureBlock,
    mut generated: Rows,
    indicator_cols: &[usize],
    append: bool,
) -> Result<FeatureBlock> {
    let width = block.roles.len();
    let generated_masks = vec![vec![false; width + indicator_cols.len()]; block.values.len()];
    sanitize_with_masks(&mut generated, &generated_masks)?;
    if append {
        let mut sources: Vec<Vec<usize>> = (0..width).map(|index| vec![index]).collect();
        sources.extend(indicator_cols.iter().map(|&index| vec![index]));
        let mut roles = block.roles.clone();
        roles.extend(vec![FeatureRole::MissingIndicator; indicator_cols.len()]);
        return finish_generated_from_sets(
            block,
            generated,
            sources,
            roles,
            generated_masks,
            true,
            "impute",
        );
    }

    let mut roles = block.roles.clone();
    roles.extend(vec![FeatureRole::MissingIndicator; indicator_cols.len()]);
    let mut lineage: Vec<FeatureLineage> = block
        .lineage
        .iter()
        .map(|item| lineage_with_operation(item, "impute"))
        .collect();
    lineage.extend(indicator_cols.iter().map(|&index| {
        let mut item = source_lineage(
            block,
            &[index],
            FeatureRole::MissingIndicator,
            "impute:indicator",
        )
        .expect("validated impute source");
        item.origin = FeatureOrigin::GeneratedSparse;
        item
    }));
    let sparse_columns = lineage
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            (item.origin == FeatureOrigin::GeneratedSparse).then_some(index)
        })
        .collect();
    FeatureBlock::new(
        generated,
        roles,
        generated_masks,
        sparse_columns,
        lineage,
        block.raw_source_width,
    )
}

fn parse_numeric(raw: &str) -> Result<Option<f64>> {
    let text = raw.trim();
    if text.is_empty()
        || text.eq_ignore_ascii_case("na")
        || text.eq_ignore_ascii_case("nan")
        || text.eq_ignore_ascii_case("null")
    {
        return Ok(None);
    }
    let value = text
        .parse::<f64>()
        .with_context(|| format!("invalid numeric value '{raw}'"))?;
    Ok(value.is_finite().then_some(value))
}

fn category_is_missing(raw: &str) -> bool {
    let text = raw.trim();
    text.is_empty() || text.eq_ignore_ascii_case("na") || text.eq_ignore_ascii_case("null")
}

fn semantic_key(raw: &str, kind: ColumnKind) -> Result<SemanticKey> {
    match kind {
        ColumnKind::Numeric => Ok(match parse_numeric(raw)? {
            Some(value) => SemanticKey::Numeric(FiniteKey::new(value)),
            None => SemanticKey::Missing,
        }),
        ColumnKind::Categorical => Ok(if category_is_missing(raw) {
            SemanticKey::Missing
        } else {
            SemanticKey::Categorical(raw.to_string())
        }),
    }
}

fn table_keys(
    table: &RawTable,
    order: &FittedTableOrder,
) -> Result<(Vec<GroupKey>, Vec<Vec<SemanticKey>>)> {
    let mut groups = Vec::with_capacity(table.rows.len());
    let mut orders = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        groups.push(
            order
                .group_columns
                .iter()
                .map(|(index, kind)| {
                    row.get(*index)
                        .with_context(|| format!("row-operation group column {index} is absent"))
                        .and_then(|value| semantic_key(value, *kind))
                })
                .collect::<Result<Vec<_>>>()?,
        );
        orders.push(
            order
                .order_columns
                .iter()
                .map(|(index, kind)| {
                    row.get(*index)
                        .with_context(|| format!("row-operation order column {index} is absent"))
                        .and_then(|value| semantic_key(value, *kind))
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok((groups, orders))
}

fn finish_row_operation(
    block: &FeatureBlock,
    mut generated: Rows,
    masks: Vec<Vec<bool>>,
    append: bool,
    operation: &str,
) -> Result<FeatureBlock> {
    sanitize_with_masks(&mut generated, &masks)?;
    if !append {
        return FeatureBlock::new(
            generated,
            block.roles.clone(),
            masks,
            block.sparse_columns.clone(),
            block
                .lineage
                .iter()
                .map(|lineage| lineage_with_operation(lineage, operation))
                .collect(),
            block.raw_source_width,
        );
    }

    let base_width = block.roles.len();
    let mut values = block.values.clone();
    for (row, added) in values.iter_mut().zip(generated) {
        row.extend(added);
    }
    let mut output_masks = block.missing_masks.clone();
    for (mask, added) in output_masks.iter_mut().zip(masks) {
        mask.extend(added);
    }
    let mut roles = block.roles.clone();
    roles.extend(block.roles.iter().copied());
    let generated_lineage: Vec<FeatureLineage> = block
        .lineage
        .iter()
        .map(|source| {
            let mut derived = lineage_with_operation(source, operation);
            derived.origin = if source.origin == FeatureOrigin::GeneratedSparse {
                FeatureOrigin::GeneratedSparse
            } else {
                FeatureOrigin::GeneratedDense
            };
            derived
        })
        .collect();
    let mut lineage = block.lineage.clone();
    lineage.extend(generated_lineage.iter().cloned());
    let mut sparse_columns = block.sparse_columns.clone();
    sparse_columns.extend(
        generated_lineage
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                (item.origin == FeatureOrigin::GeneratedSparse).then_some(base_width + index)
            }),
    );
    FeatureBlock::new(
        values,
        roles,
        output_masks,
        sparse_columns,
        lineage,
        block.raw_source_width,
    )
}

fn apply_row_operation(
    block: &FeatureBlock,
    table: &RawTable,
    order: &FittedTableOrder,
    k: usize,
    append: bool,
    operation: RowOperation,
    initial_tails: &[GroupTail],
) -> Result<(FeatureBlock, Vec<GroupTail>)> {
    if k == 0 {
        bail!("lag/roll window must be positive");
    }
    if block.values.len() != table.rows.len() {
        bail!("row-operation table/feature row count mismatch");
    }
    let width = block.roles.len();
    let (group_keys, order_keys) = table_keys(table, order)?;
    let mut chronological: Vec<usize> = (0..table.rows.len()).collect();
    chronological.sort_by(|left, right| {
        group_keys[*left]
            .cmp(&group_keys[*right])
            .then(order_keys[*left].cmp(&order_keys[*right]))
            .then(left.cmp(right))
    });

    let mut histories = BTreeMap::<GroupKey, Vec<TailRow>>::new();
    let mut fitted_cutoffs = BTreeMap::<GroupKey, Vec<SemanticKey>>::new();
    for tail in initial_tails {
        validate_group_tail_keys(tail, order)?;
        if tail
            .rows
            .iter()
            .any(|row| row.values.len() != width || row.missing.len() != width)
        {
            bail!("fitted row-operation tail width mismatch");
        }
        if histories
            .insert(tail.key.clone(), tail.rows.clone())
            .is_some()
        {
            bail!("duplicate fitted row-operation group tail");
        }
        fitted_cutoffs.insert(tail.key.clone(), tail.last_order.clone());
    }
    for row_index in 0..table.rows.len() {
        if let Some(cutoff) = fitted_cutoffs.get(&group_keys[row_index]) {
            if order_keys[row_index] <= *cutoff {
                bail!("prediction row order must be strictly after fitted order");
            }
        }
    }
    let mut latest_orders = fitted_cutoffs;

    let mut generated = vec![vec![0.0; width]; block.values.len()];
    let mut masks = vec![vec![true; width]; block.values.len()];
    let tail_limit = match operation {
        RowOperation::Lag => k,
        RowOperation::Roll => k.saturating_sub(1),
    };
    for row_index in chronological {
        latest_orders.insert(group_keys[row_index].clone(), order_keys[row_index].clone());
        let history = histories.entry(group_keys[row_index].clone()).or_default();
        match operation {
            RowOperation::Lag => {
                if history.len() >= k {
                    let prior = &history[history.len() - k];
                    generated[row_index].clone_from(&prior.values);
                    masks[row_index].clone_from(&prior.missing);
                }
            }
            RowOperation::Roll => {
                for feature in 0..width {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for prior in history.iter().rev().take(k.saturating_sub(1)).rev() {
                        if !prior.missing[feature] && prior.values[feature].is_finite() {
                            sum += prior.values[feature];
                            count += 1;
                        }
                    }
                    if !block.missing_masks[row_index][feature]
                        && block.values[row_index][feature].is_finite()
                    {
                        sum += block.values[row_index][feature];
                        count += 1;
                    }
                    if count > 0 {
                        generated[row_index][feature] = sum / count as f64;
                        masks[row_index][feature] = false;
                    }
                }
            }
        }
        history.push(TailRow {
            values: block.values[row_index].clone(),
            missing: block.missing_masks[row_index].clone(),
        });
        if history.len() > tail_limit {
            history.drain(..history.len() - tail_limit);
        }
    }
    let mut tails = Vec::with_capacity(histories.len());
    for (key, rows) in histories {
        let mut tail = GroupTail {
            last_order: latest_orders
                .remove(&key)
                .expect("every row-operation group has a latest order"),
            key,
            rows,
            state_sha256: String::new(),
        };
        tail.state_sha256 = group_tail_hash(&tail)?;
        tails.push(tail);
    }
    let name = match operation {
        RowOperation::Lag => "lag",
        RowOperation::Roll => "roll",
    };
    Ok((
        finish_row_operation(block, generated, masks, append, name)?,
        tails,
    ))
}

fn yeo_johnson(value: f64, lambda: f64) -> f64 {
    if value >= 0.0 {
        if lambda.abs() < 1e-12 {
            value.ln_1p()
        } else {
            ((value + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        -(-value).ln_1p()
    } else {
        -((1.0 - value).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
    }
}

fn fit_yeo_lambda(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let data: Vec<f64> = values.collect();
    let n = data.len().max(1) as f64;
    let mut best = 1.0;
    let mut best_score = f64::NEG_INFINITY;
    for step in -4..=4 {
        let lambda = step as f64 * 0.5;
        let transformed: Vec<f64> = data.iter().map(|v| yeo_johnson(*v, lambda)).collect();
        let mean = transformed.iter().sum::<f64>() / n;
        let variance = transformed
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;
        let jacobian: f64 = data
            .iter()
            .map(|v| {
                if *v >= 0.0 {
                    (lambda - 1.0) * v.ln_1p()
                } else {
                    (1.0 - lambda) * (-v).ln_1p()
                }
            })
            .sum();
        let score = -0.5 * n * variance.max(1e-24).ln() + jacobian;
        if score > best_score {
            best_score = score;
            best = lambda;
        }
    }
    best
}

fn piecewise_row(row: &[f64], edges: &[Vec<f64>]) -> Vec<f64> {
    let mut out = Vec::new();
    for (value, column_edges) in row.iter().zip(edges) {
        for interval in column_edges.windows(2) {
            let width = interval[1] - interval[0];
            let encoded = if width.abs() <= 1e-15 {
                f64::from(*value > interval[0])
            } else {
                ((*value - interval[0]) / width).clamp(0.0, 1.0)
            };
            out.push(encoded);
        }
    }
    out
}

pub(super) fn fit_pls_weights(x: &Rows, y: &[f64], k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = x.len();
    let cols = x.first().map_or(0, Vec::len);
    let mut means = vec![0.0; cols];
    for row in x {
        for (j, value) in row.iter().enumerate() {
            means[j] += value / n.max(1) as f64;
        }
    }
    let y_mean = y.iter().take(n).sum::<f64>() / n.max(1) as f64;
    let mut residual_x: Rows = x
        .iter()
        .map(|row| row.iter().enumerate().map(|(j, v)| v - means[j]).collect())
        .collect();
    let mut residual_y: Vec<f64> = y.iter().take(n).map(|v| v - y_mean).collect();
    let mut weights = Vec::new();
    for _ in 0..k.min(cols).min(n.saturating_sub(1)) {
        let mut weight = vec![0.0; cols];
        for (row, target) in residual_x.iter().zip(&residual_y) {
            for j in 0..cols {
                weight[j] += row[j] * target;
            }
        }
        let norm = weight.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 1e-15 {
            break;
        }
        for value in &mut weight {
            *value /= norm;
        }
        let scores: Vec<f64> = residual_x
            .iter()
            .map(|row| row.iter().zip(&weight).map(|(a, b)| a * b).sum())
            .collect();
        let denom = scores.iter().map(|v| v * v).sum::<f64>().max(1e-24);
        let y_loading = scores
            .iter()
            .zip(&residual_y)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / denom;
        let mut loadings = vec![0.0; cols];
        for j in 0..cols {
            loadings[j] = scores
                .iter()
                .zip(&residual_x)
                .map(|(score, row)| score * row[j])
                .sum::<f64>()
                / denom;
        }
        for r in 0..n {
            for j in 0..cols {
                residual_x[r][j] -= scores[r] * loadings[j];
            }
            residual_y[r] -= scores[r] * y_loading;
        }
        weights.push(weight);
    }
    (means, weights)
}

#[cfg(test)]
mod tests {
    use super::super::schema::ColumnSpec;
    use super::*;

    fn ordered_schema() -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSpec::categorical("group"),
                ColumnSpec::numeric("time"),
                ColumnSpec::numeric("value"),
            ],
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".to_string()],
                order_by: vec!["time".to_string()],
            },
        }
    }

    fn table(rows: &[(&str, &str, &str)]) -> RawTable {
        RawTable::new(
            vec!["group".to_string(), "time".to_string(), "value".to_string()],
            rows.iter()
                .map(|(group, time, value)| {
                    vec![group.to_string(), time.to_string(), value.to_string()]
                })
                .collect(),
        )
        .unwrap()
    }

    fn one_feature(values: &[f64], missing: &[bool]) -> FeatureBlock {
        FeatureBlock::new(
            values.iter().map(|value| vec![*value]).collect(),
            vec![FeatureRole::Numeric],
            missing.iter().map(|value| vec![*value]).collect(),
            Vec::new(),
            vec![FeatureLineage::raw(2, "input:numeric")],
            3,
        )
        .unwrap()
    }

    fn column(block: &FeatureBlock, index: usize) -> Vec<f64> {
        block.values.iter().map(|row| row[index]).collect()
    }

    fn mask_column(block: &FeatureBlock, index: usize) -> Vec<bool> {
        block.missing_masks.iter().map(|row| row[index]).collect()
    }

    #[test]
    fn lag_and_roll_are_group_order_aware_and_restore_caller_order() {
        // Caller order is deliberately neither group nor time order. The two
        // A/time=2 rows also prove original ordinal is the stable final key.
        let raw = table(&[
            ("B", "2", "200"),
            ("A", "2", "20"),
            ("A", "1", "10"),
            ("B", "1", "100"),
            ("A", "2", "21"),
        ]);
        let block = one_feature(&[200.0, 20.0, 10.0, 100.0, 21.0], &[false; 5]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();

        let (lag, _) =
            apply_row_operation(&block, &raw, &order, 1, false, RowOperation::Lag, &[]).unwrap();
        assert_eq!(column(&lag, 0), [100.0, 10.0, 0.0, 0.0, 20.0]);
        assert_eq!(mask_column(&lag, 0), [false, false, true, true, false]);

        let (roll, _) =
            apply_row_operation(&block, &raw, &order, 2, false, RowOperation::Roll, &[]).unwrap();
        assert_eq!(column(&roll, 0), [150.0, 15.0, 10.0, 100.0, 20.5]);
        assert_eq!(mask_column(&roll, 0), [false; 5]);
    }

    #[test]
    fn rolling_all_missing_window_stays_missing() {
        let raw = table(&[("A", "1", ""), ("A", "2", ""), ("A", "3", "4")]);
        let block = one_feature(&[0.0, 0.0, 4.0], &[true, true, false]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
        let (roll, _) =
            apply_row_operation(&block, &raw, &order, 2, false, RowOperation::Roll, &[]).unwrap();
        assert_eq!(column(&roll, 0), [0.0, 0.0, 4.0]);
        assert_eq!(mask_column(&roll, 0), [true, true, false]);
    }

    #[test]
    fn rolling_mean_uses_a_sequential_f64_fold() {
        let raw = table(&[("A", "1", "1e16"), ("A", "2", "1"), ("A", "3", "-1e16")]);
        let block = one_feature(&[1e16, 1.0, -1e16], &[false; 3]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
        let (roll, _) =
            apply_row_operation(&block, &raw, &order, 3, false, RowOperation::Roll, &[]).unwrap();
        assert_eq!(roll.values[2][0].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn typed_multicolumn_keys_cover_missing_nan_unicode_and_signed_zero() {
        let schema = DataSchema {
            columns: vec![
                ColumnSpec::numeric("group_number"),
                ColumnSpec::categorical("group_name"),
                ColumnSpec::numeric("order_number"),
                ColumnSpec::categorical("order_name"),
                ColumnSpec::numeric("value"),
            ],
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group_number".to_string(), "group_name".to_string()],
                order_by: vec!["order_number".to_string(), "order_name".to_string()],
            },
        };
        let raw = RawTable::new(
            schema
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect(),
            vec![
                vec![
                    "-0".into(),
                    "雪".into(),
                    "nan".into(),
                    "nan".into(),
                    "1".into(),
                ],
                vec![
                    "0".into(),
                    "雪".into(),
                    "NA".into(),
                    "nan".into(),
                    "2".into(),
                ],
                vec![
                    "nan".into(),
                    "雪".into(),
                    "0".into(),
                    "é".into(),
                    "3".into(),
                ],
                vec![
                    "NA".into(),
                    "雪".into(),
                    "-0".into(),
                    "é".into(),
                    "4".into(),
                ],
            ],
        )
        .unwrap();
        let order = FittedTableOrder::from_schema(&schema).unwrap();
        let (groups, orders) = table_keys(&raw, &order).unwrap();
        assert_eq!(groups[0], groups[1], "signed zero is one numeric key");
        assert_eq!(groups[2], groups[3], "numeric nan/NA are missing keys");
        assert_ne!(groups[0], groups[2]);
        assert_eq!(
            orders[0], orders[1],
            "missing numeric order ties use ordinal"
        );
        assert_eq!(orders[2], orders[3], "signed-zero numeric order ties");
        assert_eq!(groups[0].len(), 2);
        assert_eq!(orders[0].len(), 2);
        assert_eq!(groups[0][1], SemanticKey::Categorical("雪".to_string()));
        assert_eq!(orders[0][0], SemanticKey::Missing);
        assert_eq!(orders[0][1], SemanticKey::Categorical("nan".to_string()));

        let block = FeatureBlock::new(
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec![FeatureRole::Numeric],
            vec![vec![false]; 4],
            Vec::new(),
            vec![FeatureLineage::raw(4, "input:numeric")],
            5,
        )
        .unwrap();
        let (lag, _) =
            apply_row_operation(&block, &raw, &order, 1, false, RowOperation::Lag, &[]).unwrap();
        assert_eq!(column(&lag, 0), [0.0, 1.0, 0.0, 3.0]);
        assert_eq!(mask_column(&lag, 0), [true, false, true, false]);
    }

    #[test]
    fn append_preserves_original_metadata_and_generated_missing_mask() {
        let raw = table(&[("A", "1", "5"), ("A", "2", "6")]);
        let block = one_feature(&[5.0, 6.0], &[false, false]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
        let (lag, _) =
            apply_row_operation(&block, &raw, &order, 1, true, RowOperation::Lag, &[]).unwrap();

        assert_eq!(lag.values, [vec![5.0, 0.0], vec![6.0, 5.0]]);
        assert_eq!(lag.missing_masks, [vec![false, true], vec![false, false]]);
        assert_eq!(lag.roles, [FeatureRole::Numeric, FeatureRole::Numeric]);
        assert_eq!(lag.lineage[0].origin, FeatureOrigin::RawDerived);
        assert_eq!(lag.lineage[1].origin, FeatureOrigin::GeneratedDense);
        assert_eq!(lag.lineage[1].source_columns, [2]);
        assert_eq!(lag.lineage[1].operations, ["input:numeric", "lag"]);
        assert_eq!(lag.widths().raw_derived, 1);
        assert_eq!(lag.widths().generated_dense, 1);
    }

    #[test]
    fn prediction_uses_fitted_group_tails_and_unseen_groups_have_no_history() {
        let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
        let training = one_feature(&[1.0, 2.0], &[false, false]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
        let (_, tails) = apply_row_operation(
            &training,
            &training_raw,
            &order,
            2,
            false,
            RowOperation::Lag,
            &[],
        )
        .unwrap();
        let op = V3Op::Lag {
            k: 2,
            append: false,
            order,
            tails,
        };

        // A rows arrive in reverse chronological caller order. B is unseen.
        let prediction_raw = table(&[("A", "4", "4"), ("A", "3", "3"), ("B", "1", "10")]);
        let prediction = one_feature(&[4.0, 3.0, 10.0], &[false; 3]);
        let transformed = op.apply_block(&prediction, &prediction_raw).unwrap();
        assert_eq!(column(&transformed, 0), [2.0, 1.0, 0.0]);
        assert_eq!(mask_column(&transformed, 0), [false, false, true]);
    }

    #[test]
    fn rolling_prediction_starts_with_fitted_tail() {
        let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
        let training = one_feature(&[1.0, 2.0], &[false, false]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
        let (_, tails) = apply_row_operation(
            &training,
            &training_raw,
            &order,
            3,
            false,
            RowOperation::Roll,
            &[],
        )
        .unwrap();
        let op = V3Op::Roll {
            k: 3,
            append: false,
            order,
            tails,
        };
        let raw = table(&[("A", "3", "3"), ("B", "1", "10")]);
        let prediction = one_feature(&[3.0, 10.0], &[false, false]);
        let transformed = op.apply_block(&prediction, &raw).unwrap();
        assert_eq!(column(&transformed, 0), [2.0, 10.0]);
        assert_eq!(mask_column(&transformed, 0), [false, false]);
    }

    #[test]
    fn fitted_row_tails_reject_earlier_and_equal_prediction_orders() {
        let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
        let training = one_feature(&[1.0, 2.0], &[false, false]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();

        for operation in [RowOperation::Lag, RowOperation::Roll] {
            let (_, tails) =
                apply_row_operation(&training, &training_raw, &order, 2, false, operation, &[])
                    .unwrap();

            for prediction_order in ["1", "2"] {
                let prediction_raw = table(&[("A", prediction_order, "99")]);
                let prediction = one_feature(&[99.0], &[false]);
                let error = apply_row_operation(
                    &prediction,
                    &prediction_raw,
                    &order,
                    2,
                    false,
                    operation,
                    &tails,
                )
                .unwrap_err();
                assert!(
                    error.to_string().contains("strictly after fitted order"),
                    "unexpected {operation:?} error: {error}"
                );
            }

            let future_raw = table(&[("A", "3", "3"), ("B", "1", "10")]);
            let future = one_feature(&[3.0, 10.0], &[false, false]);
            apply_row_operation(&future, &future_raw, &order, 2, false, operation, &tails).unwrap();
        }
    }

    #[test]
    fn fitted_row_tails_reject_corrupt_serialized_key_state() {
        let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
        let training = one_feature(&[1.0, 2.0], &[false, false]);
        let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
        let (_, tails) = apply_row_operation(
            &training,
            &training_raw,
            &order,
            2,
            false,
            RowOperation::Lag,
            &[],
        )
        .unwrap();
        let encoded = serde_json::to_value(&tails).unwrap();

        let mut missing_cutoff = encoded.clone();
        missing_cutoff[0]["last_order"][0] = serde_json::json!("Missing");
        let mut nonfinite_cutoff = encoded.clone();
        nonfinite_cutoff[0]["last_order"][0] = serde_json::json!({"Numeric": f64::NAN.to_bits()});
        let mut wrong_order_kind = encoded.clone();
        wrong_order_kind[0]["last_order"][0] = serde_json::json!({"Categorical": "2"});
        let mut wrong_group_kind = encoded.clone();
        wrong_group_kind[0]["key"][0] = serde_json::json!({"Numeric": 0});

        let prediction_raw = table(&[("A", "1", "99")]);
        let prediction = one_feature(&[99.0], &[false]);
        for corrupt in [
            missing_cutoff,
            nonfinite_cutoff,
            wrong_order_kind,
            wrong_group_kind,
        ] {
            let corrupt_tails: Vec<GroupTail> = serde_json::from_value(corrupt).unwrap();
            let error = apply_row_operation(
                &prediction,
                &prediction_raw,
                &order,
                2,
                false,
                RowOperation::Lag,
                &corrupt_tails,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("fitted row-operation tail"),
                "unexpected corrupt-tail error: {error}"
            );
        }

        let mut missing_field = encoded;
        missing_field[0]
            .as_object_mut()
            .unwrap()
            .remove("last_order");
        assert!(serde_json::from_value::<Vec<GroupTail>>(missing_field).is_err());

        let mut wrong_order = order.clone();
        wrong_order.order_columns[0].0 = 2;
        let op = V3Op::Lag {
            k: 2,
            append: false,
            order: wrong_order,
            tails,
        };
        assert!(op
            .validate_fitted_state(&ordered_schema())
            .unwrap_err()
            .to_string()
            .contains("does not match schema"));
    }

    #[test]
    fn one_to_one_v3_transform_never_truncates_wide_raw_block() {
        let width = 1_024;
        let block = FeatureBlock::new(
            vec![vec![0.5; width]],
            vec![FeatureRole::Numeric; width],
            vec![vec![false; width]],
            Vec::new(),
            (0..width)
                .map(|source| FeatureLineage::raw(source, "input:numeric"))
                .collect(),
            width,
        )
        .unwrap();
        let raw = RawTable::new(vec!["x".to_string()], vec![vec!["0.5".to_string()]]).unwrap();
        let transformed = V3Op::MaxAbs {
            scales: vec![0.5; width],
            append: false,
        }
        .apply_block(&block, &raw)
        .unwrap();
        assert_eq!(transformed.values[0].len(), width);
        assert_eq!(transformed.lineage.len(), width);
        assert_eq!(transformed.values[0][width - 1], 1.0);
    }
    fn raw_transform_fixture(width: usize) -> FeatureBlock {
        FeatureBlock::new(
            vec![vec![2.0; width]],
            vec![FeatureRole::Numeric; width],
            vec![vec![false; width]],
            vec![],
            (0..width)
                .map(|source| FeatureLineage::raw(source, "input:numeric"))
                .collect(),
            width,
        )
        .unwrap()
    }

    #[test]
    fn legacy_one_to_one_v3_preserves_wide_raw_columns_and_masks() {
        let width = 1024;
        let mut block = raw_transform_fixture(width);
        block.missing_masks[0][width - 1] = true;
        block.values[0][width - 1] = 0.0;
        let op = FittedOp::Standardize {
            means: vec![1.0; width],
            scales: vec![2.0; width],
            append: false,
        };
        // Legacy public behavior remains capped; v3 retains all original columns.
        assert_eq!(
            crate::ops::apply_op(&op, &block.values).unwrap()[0].len(),
            crate::MAX_WIDTH
        );
        let output = legacy_to_block(&op, &block).unwrap();
        assert_eq!(output.values[0].len(), width);
        assert_eq!(output.values[0][width - 2], 0.5);
        assert_eq!(output.values[0][width - 1], 0.0);
        assert_eq!(output.missing_masks, block.missing_masks);
        assert_eq!(output.widths().raw_derived, width);
        assert_eq!(output.widths().generated_dense, 0);
    }

    #[test]
    fn legacy_append_v3_keeps_raw_plus_allowed_generated_width_on_reload() {
        let width = 300;
        let block = raw_transform_fixture(width);
        let op = V3Op::Legacy(FittedOp::RobustScale {
            medians: vec![0.0; width],
            scales: vec![2.0; width],
            append: true,
        });
        let raw = RawTable::new(vec!["x".into()], vec![vec!["2".into()]]).unwrap();
        let output = op.apply_block(&block, &raw).unwrap();
        assert_eq!(output.values[0].len(), 600);
        assert_eq!(output.values[0][299], 2.0);
        assert_eq!(output.values[0][599], 1.0);
        assert_eq!(output.widths().raw_derived, 300);
        assert_eq!(output.widths().generated_dense, 300);
        let reloaded: V3Op = serde_json::from_slice(&serde_json::to_vec(&op).unwrap()).unwrap();
        let replayed = reloaded.apply_block(&block, &raw).unwrap();
        assert_eq!(
            serde_json::to_value(&replayed).unwrap(),
            serde_json::to_value(&output).unwrap()
        );
        assert_eq!(reloaded.apply(&block.values), output.values);
    }

    #[test]
    fn legacy_append_v3_rejects_dense_overflow_before_losing_feature_columns() {
        let block = raw_transform_fixture(512);
        let op = FittedOp::Standardize {
            means: vec![0.0; 512],
            scales: vec![1.0; 512],
            append: true,
        };
        let mut dense = block.clone();
        dense.roles = vec![FeatureRole::Generated; 512];
        dense.lineage = (0..512)
            .map(|source| FeatureLineage::generated_dense(vec![source], "synthetic:generated"))
            .collect();
        dense.validate().unwrap();
        let error = legacy_to_block(&op, &dense).unwrap_err().to_string();
        assert!(
            error.contains("generated dense width 1024 exceeds 512"),
            "{error}"
        );
        assert!(!error.contains("row/mask"));
    }
}
