use super::*;

pub(crate) fn parse_numeric(raw: &str) -> Result<Option<f64>> {
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

pub(crate) fn category_is_missing(raw: &str) -> bool {
    let text = raw.trim();
    text.is_empty() || text.eq_ignore_ascii_case("null") || text.eq_ignore_ascii_case("na")
}

pub(crate) fn category_value(raw: &str) -> String {
    if category_is_missing(raw) {
        "\0MISSING".to_string()
    } else {
        raw.to_string()
    }
}

pub(crate) fn honest_ordered_values(
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

pub(crate) fn smoothed_statistic(sum: f64, count: usize, prior: f64, fallback: f64) -> f64 {
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

pub(crate) fn honest_historical_values(
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

pub(crate) fn historical_prediction_values(
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
