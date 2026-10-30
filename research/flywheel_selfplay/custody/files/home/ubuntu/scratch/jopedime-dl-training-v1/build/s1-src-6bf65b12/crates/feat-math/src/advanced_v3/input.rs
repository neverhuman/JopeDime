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

impl FittedInput {
    pub(super) fn fit_transform(
        config: InputConfig,
        schema: &DataSchema,
        table: &RawTable,
        y: &[f64],
        task: TaskKind,
    ) -> Result<(Self, FeatureBlock)> {
        table.validate_against(schema)?;
        if table.rows.is_empty() || table.rows.len() != y.len() {
            bail!("training table is empty or target length differs from row count");
        }
        let global = y.iter().sum::<f64>() / y.len() as f64;
        let row_order = FittedRowOrder::from_schema(schema);
        let mut columns = Vec::with_capacity(schema.columns.len());
        let mut historical = Vec::with_capacity(schema.columns.len());
        for (j, spec) in schema.columns.iter().enumerate() {
            match spec.kind {
                ColumnKind::Numeric => {
                    let mut finite = Vec::new();
                    for row in &table.rows {
                        if let Some(value) = parse_numeric(&row[j])? {
                            finite.push(value);
                        }
                    }
                    let fill = match config.numeric {
                        NumericMode::Zero => 0.0,
                        NumericMode::Mean => {
                            finite.iter().sum::<f64>() / finite.len().max(1) as f64
                        }
                        NumericMode::Median => {
                            finite.sort_by(f64::total_cmp);
                            quantile_sorted(&finite, 0.5)
                        }
                    };
                    columns.push(FittedColumn::Numeric {
                        fill,
                        indicator: config.missing_indicator,
                    });
                    historical.push(None);
                }
                ColumnKind::Categorical => {
                    let mut by_value = BTreeMap::<String, (usize, f64)>::new();
                    for (row, &target) in table.rows.iter().zip(y) {
                        let key = category_value(&row[j]);
                        let entry = by_value.entry(key).or_insert((0, 0.0));
                        entry.0 += 1;
                        entry.1 += target;
                    }
                    let mut vocabulary = Vec::new();
                    let mut counts = Vec::new();
                    let mut sums = Vec::new();
                    for (value, (count, sum)) in by_value {
                        vocabulary.push(value);
                        counts.push(count);
                        sums.push(sum);
                    }
                    let mut eligible: Vec<usize> = counts
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &count)| (count >= config.min_count).then_some(i))
                        .collect();
                    eligible.sort_by(|&a, &b| {
                        counts[b]
                            .cmp(&counts[a])
                            .then(vocabulary[a].cmp(&vocabulary[b]))
                    });
                    eligible.truncate(config.max_onehot);
                    eligible.sort_unstable();
                    let onehot = eligible;
                    let mode = if config.categorical == CategoricalMode::Hybrid
                        && vocabulary.len() <= config.max_onehot
                    {
                        CategoricalMode::Onehot
                    } else {
                        config.categorical
                    };
                    let fitted_historical = if matches!(
                        (&row_order, mode),
                        (
                            FittedRowOrder::Ordered { .. },
                            CategoricalMode::Ordered | CategoricalMode::Hybrid
                        )
                    ) {
                        let indices: Vec<Option<usize>> = table
                            .rows
                            .iter()
                            .map(|row| vocabulary.binary_search(&category_value(&row[j])).ok())
                            .collect();
                        Some(
                            honest_historical_values(&indices, y, table, &row_order, config.prior)?
                                .1,
                        )
                    } else {
                        None
                    };
                    if matches!(
                        task,
                        TaskKind::Regression
                            | TaskKind::Binary
                            | TaskKind::Timeseries
                            | TaskKind::Multiclass
                    ) {
                        columns.push(FittedColumn::Categorical {
                            mode,
                            vocabulary,
                            counts,
                            sums,
                            total: table.rows.len(),
                            global,
                            onehot,
                        });
                        historical.push(fitted_historical);
                    }
                }
            }
        }
        let fitted = Self {
            config,
            columns,
            row_order,
            historical,
        };
        let block = fitted.transform_impl(table, Some(y))?;
        Ok((fitted, block))
    }

    pub(super) fn transform(&self, table: &RawTable) -> Result<FeatureBlock> {
        self.transform_impl(table, None)
    }

    fn transform_impl(&self, table: &RawTable, train_y: Option<&[f64]>) -> Result<FeatureBlock> {
        let mut values = vec![Vec::<f64>::new(); table.rows.len()];
        let mut masks = vec![Vec::<bool>::new(); table.rows.len()];
        let mut roles = Vec::new();
        let mut sparse = Vec::new();
        let mut lineage = Vec::new();
        for (j, fitted) in self.columns.iter().enumerate() {
            match fitted {
                FittedColumn::Numeric { fill, indicator } => {
                    roles.push(FeatureRole::Numeric);
                    lineage.push(FeatureLineage::raw(j, "input:numeric"));
                    for (r, raw) in table.rows.iter().map(|row| &row[j]).enumerate() {
                        let parsed = parse_numeric(raw)?;
                        values[r].push(parsed.unwrap_or(*fill));
                        masks[r].push(parsed.is_none());
                    }
                    if *indicator {
                        let out_j = roles.len();
                        roles.push(FeatureRole::MissingIndicator);
                        lineage.push(FeatureLineage::generated_sparse(
                            vec![j],
                            "input:numeric_missing_indicator",
                        ));
                        sparse.push(out_j);
                        for (r, raw) in table.rows.iter().map(|row| &row[j]).enumerate() {
                            let missing = parse_numeric(raw)?.is_none();
                            values[r].push(if missing { 1.0 } else { 0.0 });
                            masks[r].push(false);
                        }
                    }
                }
                FittedColumn::Categorical {
                    mode,
                    vocabulary,
                    counts,
                    sums,
                    total,
                    global,
                    onehot,
                } => {
                    let indices: Vec<Option<usize>> = table
                        .rows
                        .iter()
                        .map(|row| vocabulary.binary_search(&category_value(&row[j])).ok())
                        .collect();
                    let ordered =
                        if matches!(mode, CategoricalMode::Ordered | CategoricalMode::Hybrid) {
                            Some(match train_y {
                                Some(y) => match &self.row_order {
                                    FittedRowOrder::Iid => honest_ordered_values(
                                        &indices,
                                        y,
                                        self.config.prior,
                                        self.config.folds,
                                    ),
                                    FittedRowOrder::Ordered { .. } => {
                                        honest_historical_values(
                                            &indices,
                                            y,
                                            table,
                                            &self.row_order,
                                            self.config.prior,
                                        )?
                                        .0
                                    }
                                },
                                None => match self.historical.get(j).and_then(Option::as_ref) {
                                    Some(state) => historical_prediction_values(
                                        &indices,
                                        table,
                                        &self.row_order,
                                        state,
                                        self.config.prior,
                                    )?,
                                    None => indices
                                        .iter()
                                        .map(|index| match index {
                                            Some(i) => smoothed_statistic(
                                                sums[*i],
                                                counts[*i],
                                                self.config.prior,
                                                *global,
                                            ),
                                            None => *global,
                                        })
                                        .collect(),
                                },
                            })
                        } else {
                            None
                        };
                    match mode {
                        CategoricalMode::Native => {
                            roles.push(FeatureRole::CategoricalNative);
                            lineage.push(FeatureLineage::raw(j, "input:categorical_native"));
                            for (r, index) in indices.iter().enumerate() {
                                values[r].push(index.map_or(-1.0, |i| i as f64));
                                masks[r].push(category_is_missing(&table.rows[r][j]));
                            }
                        }
                        CategoricalMode::Frequency => {
                            roles.push(FeatureRole::CategoricalFrequency);
                            lineage.push(FeatureLineage::raw(j, "input:categorical_frequency"));
                            for (r, index) in indices.iter().enumerate() {
                                values[r]
                                    .push(index.map_or(0.0, |i| counts[i] as f64 / *total as f64));
                                masks[r].push(category_is_missing(&table.rows[r][j]));
                            }
                        }
                        CategoricalMode::Onehot => {
                            for &category in onehot {
                                let out_j = roles.len();
                                roles.push(FeatureRole::CategoricalOneHot);
                                lineage.push(FeatureLineage::generated_sparse(
                                    vec![j],
                                    format!("input:categorical_one_hot:{category}"),
                                ));
                                sparse.push(out_j);
                                for (r, index) in indices.iter().enumerate() {
                                    values[r].push(if *index == Some(category) {
                                        1.0
                                    } else {
                                        0.0
                                    });
                                    masks[r].push(false);
                                }
                            }
                        }
                        CategoricalMode::Ordered => {
                            roles.push(FeatureRole::CategoricalOrdered);
                            lineage.push(FeatureLineage::raw(j, "input:categorical_ordered"));
                            for (r, value) in
                                ordered.expect("ordered values").into_iter().enumerate()
                            {
                                values[r].push(value);
                                masks[r].push(category_is_missing(&table.rows[r][j]));
                            }
                        }
                        CategoricalMode::Hybrid => {
                            roles.push(FeatureRole::CategoricalFrequency);
                            roles.push(FeatureRole::CategoricalOrdered);
                            lineage.push(FeatureLineage::raw(j, "input:categorical_frequency"));
                            lineage.push(FeatureLineage::generated_dense(
                                vec![j],
                                "input:categorical_ordered",
                            ));
                            let ordered = ordered.expect("hybrid ordered values");
                            for (r, index) in indices.iter().enumerate() {
                                values[r]
                                    .push(index.map_or(0.0, |i| counts[i] as f64 / *total as f64));
                                masks[r].push(category_is_missing(&table.rows[r][j]));
                                values[r].push(ordered[r]);
                                masks[r].push(false);
                            }
                        }
                    }
                }
            }
        }
        FeatureBlock::new(values, roles, masks, sparse, lineage, self.columns.len())
    }
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
    text.is_empty() || text.eq_ignore_ascii_case("null") || text.eq_ignore_ascii_case("na")
}

fn category_value(raw: &str) -> String {
    if category_is_missing(raw) {
        "\0MISSING".to_string()
    } else {
        raw.to_string()
    }
}

fn honest_ordered_values(
    indices: &[Option<usize>],
    y: &[f64],
    prior: f64,
    folds: usize,
) -> Vec<f64> {
    let fold_ids = stratified_fold_assignments(y, folds);
    let category_count = indices
        .iter()
        .filter_map(|index| *index)
        .max()
        .map_or(0, |index| index + 1);
    let mut training_category_counts = vec![vec![0usize; category_count]; folds];
    let mut training_category_sums = vec![vec![0.0; category_count]; folds];
    let mut training_counts = vec![0usize; folds];
    let mut training_sums = vec![0.0; folds];
    for (r, index) in indices.iter().enumerate() {
        for validation_fold in 0..folds {
            if fold_ids[r] == validation_fold {
                continue;
            }
            training_counts[validation_fold] += 1;
            training_sums[validation_fold] += y[r];
            if let Some(i) = index {
                training_category_counts[validation_fold][*i] += 1;
                training_category_sums[validation_fold][*i] += y[r];
            }
        }
    }
    indices
        .iter()
        .enumerate()
        .map(|(r, index)| {
            let fold = fold_ids[r];
            let fold_global = if training_counts[fold] == 0 {
                0.0
            } else {
                training_sums[fold] / training_counts[fold] as f64
            };
            match index {
                Some(i) => smoothed_statistic(
                    training_category_sums[fold][*i],
                    training_category_counts[fold][*i],
                    prior,
                    fold_global,
                ),
                None => fold_global,
            }
        })
        .collect()
}

#[derive(Default)]
struct RunningTarget {
    count: usize,
    sum: f64,
}

#[derive(Default)]
struct WorkingHistoricalGroup {
    count: usize,
    sum: f64,
    categories: BTreeMap<usize, RunningTarget>,
}

fn smoothed_statistic(sum: f64, count: usize, prior: f64, fallback: f64) -> f64 {
    if count == 0 && prior == 0.0 {
        fallback
    } else {
        (sum + prior * fallback) / (count as f64 + prior)
    }
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

fn keys_for_rows(
    table: &RawTable,
    row_order: &FittedRowOrder,
) -> Result<(Vec<GroupKey>, Vec<Vec<SemanticKey>>)> {
    let FittedRowOrder::Ordered {
        group_columns,
        order_columns,
    } = row_order
    else {
        bail!("historical encoding requires ordered row semantics");
    };
    let mut groups = Vec::with_capacity(table.rows.len());
    let mut orders = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        groups.push(
            group_columns
                .iter()
                .map(|(index, kind)| semantic_key(&row[*index], *kind))
                .collect::<Result<Vec<_>>>()?,
        );
        orders.push(
            order_columns
                .iter()
                .map(|(index, kind)| semantic_key(&row[*index], *kind))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok((groups, orders))
}

fn honest_historical_values(
    indices: &[Option<usize>],
    y: &[f64],
    table: &RawTable,
    row_order: &FittedRowOrder,
    prior: f64,
) -> Result<(Vec<f64>, HistoricalTargetState)> {
    let (group_keys, order_keys) = keys_for_rows(table, row_order)?;
    let mut order: Vec<usize> = (0..table.rows.len()).collect();
    order.sort_by(|left, right| {
        order_keys[*left]
            .cmp(&order_keys[*right])
            .then(group_keys[*left].cmp(&group_keys[*right]))
            .then(left.cmp(right))
    });

    let mut values = vec![0.0; table.rows.len()];
    let mut global = RunningTarget::default();
    let mut groups = BTreeMap::<GroupKey, WorkingHistoricalGroup>::new();
    for row in order {
        let group = groups.entry(group_keys[row].clone()).or_default();
        let fallback = if group.count > 0 {
            group.sum / group.count as f64
        } else if global.count > 0 {
            global.sum / global.count as f64
        } else {
            0.0
        };
        values[row] = match indices[row] {
            Some(category) => match group.categories.get(&category) {
                Some(statistic) => {
                    smoothed_statistic(statistic.sum, statistic.count, prior, fallback)
                }
                None => fallback,
            },
            None => fallback,
        };

        group.count += 1;
        group.sum += y[row];
        if let Some(category) = indices[row] {
            let statistic = group.categories.entry(category).or_default();
            statistic.count += 1;
            statistic.sum += y[row];
        }
        global.count += 1;
        global.sum += y[row];
    }

    let groups = groups
        .into_iter()
        .map(|(key, group)| HistoricalGroupState {
            key,
            count: group.count,
            sum: group.sum,
            categories: group
                .categories
                .into_iter()
                .map(|(category, statistic)| HistoricalCategoryState {
                    category,
                    count: statistic.count,
                    sum: statistic.sum,
                })
                .collect(),
        })
        .collect();
    Ok((
        values,
        HistoricalTargetState {
            count: global.count,
            sum: global.sum,
            groups,
        },
    ))
}

fn historical_prediction_values(
    indices: &[Option<usize>],
    table: &RawTable,
    row_order: &FittedRowOrder,
    state: &HistoricalTargetState,
    prior: f64,
) -> Result<Vec<f64>> {
    let (group_keys, _) = keys_for_rows(table, row_order)?;
    let global = if state.count == 0 {
        0.0
    } else {
        state.sum / state.count as f64
    };
    Ok(group_keys
        .iter()
        .zip(indices)
        .map(|(key, index)| {
            let Ok(group_index) = state.groups.binary_search_by(|group| group.key.cmp(key)) else {
                return global;
            };
            let group = &state.groups[group_index];
            let fallback = if group.count == 0 {
                global
            } else {
                group.sum / group.count as f64
            };
            let Some(category) = index else {
                return fallback;
            };
            let Ok(category_index) = group
                .categories
                .binary_search_by_key(category, |item| item.category)
            else {
                return fallback;
            };
            let statistic = &group.categories[category_index];
            smoothed_statistic(statistic.sum, statistic.count, prior, fallback)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::schema::ColumnSpec;
    use super::*;

    fn config(numeric: NumericMode, categorical: CategoricalMode, prior: f64) -> InputConfig {
        InputConfig {
            numeric,
            missing_indicator: false,
            categorical,
            max_onehot: 16,
            min_count: 1,
            prior,
            folds: 3,
        }
    }

    fn one_column_schema(kind: ColumnKind) -> DataSchema {
        DataSchema {
            columns: vec![ColumnSpec {
                name: "x".to_string(),
                kind,
            }],
            row_semantics: RowSemantics::Iid,
        }
    }

    #[test]
    fn numeric_modes_treat_tokens_and_nonfinite_values_as_missing() {
        let table = RawTable::new(
            vec!["x".into()],
            ["", " na ", "NaN", "null", "inf", "-inf", "1", "3"]
                .into_iter()
                .map(|value| vec![value.to_string()])
                .collect(),
        )
        .unwrap();
        let schema = one_column_schema(ColumnKind::Numeric);
        let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for (mode, fill) in [
            (NumericMode::Median, 2.0_f64),
            (NumericMode::Mean, 2.0_f64),
            (NumericMode::Zero, 0.0_f64),
        ] {
            let (_, block) = FittedInput::fit_transform(
                config(mode, CategoricalMode::Native, 0.0),
                &schema,
                &table,
                &y,
                TaskKind::Regression,
            )
            .unwrap();
            assert!(block.values[..6]
                .iter()
                .all(|row| row[0].to_bits() == fill.to_bits()));
            assert!(block.missing_masks[..6].iter().all(|row| row[0]));
            assert_eq!(block.values[6][0], 1.0);
            assert_eq!(block.values[7][0], 3.0);
        }
    }

    #[test]
    fn categorical_vocabulary_is_canonical_and_missing_nan_unseen_are_distinct() {
        let table = RawTable::new(
            vec!["x".into()],
            ["z", "", "nan", "a", "NA", "null"]
                .into_iter()
                .map(|value| vec![value.to_string()])
                .collect(),
        )
        .unwrap();
        let schema = one_column_schema(ColumnKind::Categorical);
        let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let (fitted, block) = FittedInput::fit_transform(
            config(NumericMode::Zero, CategoricalMode::Native, 0.0),
            &schema,
            &table,
            &y,
            TaskKind::Regression,
        )
        .unwrap();
        let FittedColumn::Categorical { vocabulary, .. } = &fitted.columns[0] else {
            panic!("categorical column expected")
        };
        assert_eq!(vocabulary, &["\0MISSING", "a", "nan", "z"]);
        assert_eq!(block.roles, [FeatureRole::CategoricalNative]);

        let prediction = RawTable::new(
            vec!["x".into()],
            vec![vec!["".into()], vec!["nan".into()], vec!["unseen".into()]],
        )
        .unwrap();
        let encoded = fitted.transform(&prediction).unwrap();
        assert_ne!(encoded.values[0][0], encoded.values[1][0]);
        assert_eq!(encoded.values[2][0], -1.0);
        assert_eq!(encoded.missing_masks, [[true], [false], [false]]);
    }

    #[test]
    fn every_categorical_mode_keeps_its_declared_roles() {
        let table = RawTable::new(
            vec!["x".into()],
            ["b", "a", "nan", ""]
                .into_iter()
                .map(|value| vec![value.to_string()])
                .collect(),
        )
        .unwrap();
        let schema = one_column_schema(ColumnKind::Categorical);
        let y = [0.1, 0.4, 0.7, 1.2];
        for (mode, expected) in [
            (
                CategoricalMode::Native,
                vec![FeatureRole::CategoricalNative],
            ),
            (
                CategoricalMode::Frequency,
                vec![FeatureRole::CategoricalFrequency],
            ),
            (
                CategoricalMode::Ordered,
                vec![FeatureRole::CategoricalOrdered],
            ),
        ] {
            let (_, block) = FittedInput::fit_transform(
                config(NumericMode::Zero, mode, 0.0),
                &schema,
                &table,
                &y,
                TaskKind::Regression,
            )
            .unwrap();
            assert_eq!(block.roles, expected);
        }

        let (_, onehot) = FittedInput::fit_transform(
            config(NumericMode::Zero, CategoricalMode::Onehot, 0.0),
            &schema,
            &table,
            &y,
            TaskKind::Regression,
        )
        .unwrap();
        assert_eq!(onehot.roles, vec![FeatureRole::CategoricalOneHot; 4]);

        let mut hybrid_config = config(NumericMode::Zero, CategoricalMode::Hybrid, 0.0);
        hybrid_config.max_onehot = 1;
        let (_, hybrid) =
            FittedInput::fit_transform(hybrid_config, &schema, &table, &y, TaskKind::Regression)
                .unwrap();
        assert_eq!(
            hybrid.roles,
            [
                FeatureRole::CategoricalFrequency,
                FeatureRole::CategoricalOrdered
            ]
        );
    }

    #[test]
    fn iid_ordered_encoding_uses_only_fold_training_statistics() {
        let indices = vec![Some(0); 6];
        let y = [0.2, 10.0, 1.2, 20.0, 2.2, 30.0];
        let original = honest_ordered_values(&indices, &y, 0.0, 3);
        let mut permuted = y;
        permuted.swap(0, 3);
        let changed = honest_ordered_values(&indices, &permuted, 0.0, 3);
        assert_eq!(original[0].to_bits(), changed[0].to_bits());
        assert_eq!(original[3].to_bits(), changed[3].to_bits());

        let unique: Vec<Option<usize>> = (0..6).map(Some).collect();
        let encoded = honest_ordered_values(&unique, &y, 0.0, 3);
        for (row, &actual) in encoded.iter().enumerate().take(y.len()) {
            let fold = row % 3;
            let expected: Vec<f64> = y
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (index % 3 != fold).then_some(*value))
                .collect();
            let expected = expected.iter().sum::<f64>() / expected.len() as f64;
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn ordered_history_is_stable_for_ties_and_has_no_future_access() {
        let schema = DataSchema {
            columns: vec![
                ColumnSpec::categorical("group"),
                ColumnSpec::numeric("time"),
                ColumnSpec::categorical("category"),
            ],
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".into()],
                order_by: vec!["time".into()],
            },
        };
        let table = RawTable::new(
            vec!["group".into(), "time".into(), "category".into()],
            vec![
                vec!["A".into(), "10".into(), "x".into()],
                vec!["A".into(), "2".into(), "x".into()],
                vec!["A".into(), "10".into(), "x".into()],
            ],
        )
        .unwrap();
        let mut ordered_config = config(NumericMode::Zero, CategoricalMode::Ordered, 0.0);
        ordered_config.folds = 2;
        let y = [20.0, 10.0, 30.0];
        let (fitted, block) = FittedInput::fit_transform(
            ordered_config.clone(),
            &schema,
            &table,
            &y,
            TaskKind::Regression,
        )
        .unwrap();
        assert_eq!(block.values[0][2], 10.0);
        assert_eq!(block.values[1][2], 0.0);
        assert_eq!(block.values[2][2], 15.0);

        let mut future_changed = y;
        future_changed[2] = 999.0;
        let (_, changed) = FittedInput::fit_transform(
            ordered_config,
            &schema,
            &table,
            &future_changed,
            TaskKind::Regression,
        )
        .unwrap();
        for row in 0..3 {
            assert_eq!(
                block.values[row][2].to_bits(),
                changed.values[row][2].to_bits()
            );
        }

        let prediction = RawTable::new(
            vec!["group".into(), "time".into(), "category".into()],
            vec![
                vec!["A".into(), "11".into(), "x".into()],
                vec!["A".into(), "12".into(), "unseen".into()],
                vec!["B".into(), "1".into(), "x".into()],
            ],
        )
        .unwrap();
        let predicted = fitted.transform(&prediction).unwrap();
        assert_eq!(predicted.values[0][2], 20.0);
        assert_eq!(predicted.values[1][2], 20.0);
        assert_eq!(predicted.values[2][2], 20.0);
    }

    #[test]
    fn ordered_history_global_fallback_never_uses_later_group_targets() {
        let schema = DataSchema {
            columns: vec![
                ColumnSpec::categorical("group"),
                ColumnSpec::numeric("time"),
                ColumnSpec::categorical("category"),
            ],
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".into()],
                order_by: vec!["time".into()],
            },
        };
        let table = RawTable::new(
            vec!["group".into(), "time".into(), "category".into()],
            vec![
                vec!["A".into(), "100".into(), "x".into()],
                vec!["B".into(), "1".into(), "x".into()],
                vec!["B".into(), "2".into(), "x".into()],
            ],
        )
        .unwrap();
        let ordered_config = config(NumericMode::Zero, CategoricalMode::Ordered, 0.0);
        let (_, baseline) = FittedInput::fit_transform(
            ordered_config.clone(),
            &schema,
            &table,
            &[999.0, 10.0, 20.0],
            TaskKind::Regression,
        )
        .unwrap();
        let (_, perturbed) = FittedInput::fit_transform(
            ordered_config,
            &schema,
            &table,
            &[-999.0, 10.0, 20.0],
            TaskKind::Regression,
        )
        .unwrap();

        assert_eq!(baseline.values[1][2].to_bits(), 0.0_f64.to_bits());
        assert_eq!(baseline.values[2][2].to_bits(), 10.0_f64.to_bits());
        for row in [1, 2] {
            assert_eq!(
                baseline.values[row][2].to_bits(),
                perturbed.values[row][2].to_bits()
            );
        }
    }
}
