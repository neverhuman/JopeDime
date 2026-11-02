use super::*;

fn compute_statistics<'a>(
    table: &'a RawTable,
    schema: &DataSchema,
    task: &TaskSpec,
    targets: &[f64],
    rows: &[usize],
    columns: &[usize],
    numeric_pairs: &[(usize, usize)],
) -> Result<([f64; PROFILE_STAT_COUNT], Vec<ColumnSketch<'a>>)> {
    let mut sketches = Vec::with_capacity(columns.len());
    let mut row_has_missing = vec![false; rows.len()];
    let mut column_missingness = Vec::with_capacity(columns.len());
    let mut numeric_unique_fractions = Vec::new();
    let mut categorical_log_cardinalities = Vec::new();
    let mut absolute_skews = Vec::new();
    let mut absolute_kurtoses = Vec::new();
    let mut missing_cells = 0usize;
    let mut observed_numeric_cells = 0usize;
    let mut numeric_zero_cells = 0usize;
    let mut constant_columns = 0usize;
    let mut dominant_columns = 0usize;
    let mut categorical_cells = 0usize;
    let mut rare_categorical_cells = 0usize;
    let mut outlier_numeric_cells = 0usize;

    for &column in columns {
        match schema.columns[column].kind {
            ColumnKind::Numeric => {
                let mut values = Vec::with_capacity(rows.len());
                for (sample_row, &source_row) in rows.iter().enumerate() {
                    let value = parse_numeric_profile(&table.rows[source_row][column])
                        .with_context(|| {
                            format!(
                                "invalid numeric profile value at row {}, column '{}'",
                                source_row + 1,
                                schema.columns[column].name
                            )
                        })?;
                    if value.is_none() {
                        missing_cells += 1;
                        row_has_missing[sample_row] = true;
                    }
                    values.push(value);
                }
                let observed: Vec<f64> = values.iter().flatten().copied().collect();
                observed_numeric_cells += observed.len();
                numeric_zero_cells += observed.iter().filter(|value| **value == 0.0).count();
                let unique = numeric_unique_count(&observed);
                numeric_unique_fractions.push(if observed.is_empty() {
                    0.0
                } else {
                    unique as f64 / observed.len() as f64
                });
                let (skew, kurtosis) = skew_and_excess_kurtosis(&observed);
                absolute_skews.push(skew.abs());
                absolute_kurtoses.push(kurtosis.abs());
                outlier_numeric_cells += three_iqr_outlier_count(&observed);
                constant_columns += usize::from(unique <= 1);
                dominant_columns += usize::from(dominant_numeric(&observed) >= 0.99);
                column_missingness
                    .push(missing_fraction(values.len() - observed.len(), rows.len()));
                sketches.push(ColumnSketch::Numeric(values));
            }
            ColumnKind::Categorical => {
                let mut values = Vec::with_capacity(rows.len());
                let mut counts = BTreeMap::<&str, usize>::new();
                for (sample_row, &source_row) in rows.iter().enumerate() {
                    let value = categorical_profile_value(&table.rows[source_row][column]);
                    if let Some(value) = value {
                        *counts.entry(value).or_default() += 1;
                    } else {
                        missing_cells += 1;
                        row_has_missing[sample_row] = true;
                    }
                    values.push(value);
                }
                let observed = counts.values().sum::<usize>();
                categorical_cells += observed;
                let rare_limit = (observed / 100).max(1);
                rare_categorical_cells += counts
                    .values()
                    .filter(|count| **count <= rare_limit)
                    .sum::<usize>();
                categorical_log_cardinalities.push((counts.len() as f64).ln_1p());
                constant_columns += usize::from(counts.len() <= 1);
                let dominant = counts.values().copied().max().unwrap_or(0);
                let dominant_fraction = if observed == 0 {
                    1.0
                } else {
                    dominant as f64 / observed as f64
                };
                dominant_columns += usize::from(dominant_fraction >= 0.99);
                column_missingness.push(missing_fraction(rows.len() - observed, rows.len()));
                sketches.push(ColumnSketch::Categorical(values));
            }
        }
    }

    let selected_targets: Vec<f64> = rows.iter().map(|row| targets[*row]).collect();
    let column_to_sketch: BTreeMap<usize, usize> = columns
        .iter()
        .copied()
        .enumerate()
        .map(|(sketch, column)| (column, sketch))
        .collect();
    let pair_correlations: Vec<f64> = numeric_pairs
        .iter()
        .map(|(left, right)| {
            let ColumnSketch::Numeric(left_values) = &sketches[column_to_sketch[left]] else {
                unreachable!("pair selection contains only numeric columns")
            };
            let ColumnSketch::Numeric(right_values) = &sketches[column_to_sketch[right]] else {
                unreachable!("pair selection contains only numeric columns")
            };
            optional_correlation(left_values, right_values).abs()
        })
        .collect();

    let mut maximum_numeric_target = 0.0f64;
    let mut maximum_categorical_target = 0.0f64;
    for sketch in &sketches {
        match sketch {
            ColumnSketch::Numeric(values) => {
                maximum_numeric_target = maximum_numeric_target
                    .max(optional_target_correlation(values, &selected_targets).abs());
            }
            ColumnSketch::Categorical(values) => {
                maximum_categorical_target = maximum_categorical_target
                    .max(categorical_correlation_ratio(values, &selected_targets));
            }
        }
    }

    let selected_column_count = columns.len().max(1);
    let selected_cell_count = rows.len().saturating_mul(columns.len()).max(1);
    let numeric_column_count = schema
        .columns
        .iter()
        .filter(|column| column.kind == ColumnKind::Numeric)
        .count();
    let target_stat = match task.kind {
        TaskKind::Regression | TaskKind::Timeseries => {
            skew_and_excess_kurtosis(&selected_targets).0
        }
        TaskKind::Binary => {
            let positive = selected_targets
                .iter()
                .filter(|target| **target == 1.0)
                .count();
            positive.min(selected_targets.len() - positive) as f64
                / selected_targets.len().max(1) as f64
        }
        TaskKind::Multiclass => {
            let mut counts = std::collections::BTreeMap::<i64, usize>::new();
            for target in &selected_targets {
                *counts.entry(*target as i64).or_default() += 1;
            }
            counts.values().copied().min().unwrap_or(0) as f64
                / selected_targets.len().max(1) as f64
        }
    };
    let statistics = [
        (table.rows.len() as f64).ln_1p(),
        (schema.columns.len() as f64).ln_1p(),
        finite_stat((table.rows.len() as f64 / schema.columns.len() as f64).ln()),
        numeric_column_count as f64 / schema.columns.len() as f64,
        missing_cells as f64 / selected_cell_count as f64,
        row_has_missing.iter().filter(|missing| **missing).count() as f64
            / rows.len().max(1) as f64,
        quantile(&column_missingness, 0.5),
        column_missingness.iter().copied().fold(0.0, f64::max),
        numeric_zero_cells as f64 / observed_numeric_cells.max(1) as f64,
        constant_columns as f64 / selected_column_count as f64,
        dominant_columns as f64 / selected_column_count as f64,
        quantile(&numeric_unique_fractions, 0.5),
        quantile(&numeric_unique_fractions, 0.9),
        quantile(&categorical_log_cardinalities, 0.5),
        categorical_log_cardinalities
            .iter()
            .copied()
            .fold(0.0, f64::max),
        rare_categorical_cells as f64 / categorical_cells.max(1) as f64,
        quantile(&absolute_skews, 0.5),
        quantile(&absolute_kurtoses, 0.5),
        outlier_numeric_cells as f64 / observed_numeric_cells.max(1) as f64,
        duplicate_raw_row_fraction(table),
        quantile(&pair_correlations, 0.5),
        maximum_numeric_target,
        maximum_categorical_target,
        target_stat,
    ]
    .map(finite_stat);
    Ok((statistics, sketches))
}

fn deterministic_indices(total: usize, limit: usize) -> Vec<usize> {
    let selected = total.min(limit);
    if selected == total {
        return (0..total).collect();
    }
    (0..selected)
        .map(|index| ((index as u128 * total as u128) / selected as u128) as usize)
        .collect()
}

fn deterministic_numeric_pairs(columns: &[usize], limit: usize) -> Vec<(usize, usize)> {
    let mut all = Vec::new();
    for left in 0..columns.len() {
        for right in (left + 1)..columns.len() {
            all.push((columns[left], columns[right]));
        }
    }
    deterministic_indices(all.len(), limit)
        .into_iter()
        .map(|index| all[index])
        .collect()
}

fn subset_table(table: &RawTable, rows: &[usize], columns: &[usize]) -> Result<RawTable> {
    RawTable::new(
        columns
            .iter()
            .map(|column| table.headers[*column].clone())
            .collect(),
        rows.iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|column| table.rows[*row][*column].clone())
                    .collect()
            })
            .collect(),
    )
}

fn parse_numeric_profile(raw: &str) -> Result<Option<f64>> {
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

fn categorical_profile_value(raw: &str) -> Option<&str> {
    let text = raw.trim();
    (!text.is_empty() && !text.eq_ignore_ascii_case("na") && !text.eq_ignore_ascii_case("null"))
        .then_some(raw)
}

fn numeric_unique_count(values: &[f64]) -> usize {
    values
        .iter()
        .map(|value| normalized_bits(*value))
        .collect::<BTreeSet<_>>()
        .len()
}

fn normalized_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn dominant_numeric(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let mut counts = BTreeMap::<u64, usize>::new();
    for value in values {
        *counts.entry(normalized_bits(*value)).or_default() += 1;
    }
    counts.values().copied().max().unwrap_or(0) as f64 / values.len() as f64
}

fn missing_fraction(missing: usize, total: usize) -> f64 {
    missing as f64 / total.max(1) as f64
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    finite_stat(quantile_sorted(&sorted, probability))
}

fn skew_and_excess_kurtosis(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (0.0, 0.0);
    }
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return (0.0, 0.0);
    }
    let scaled: Vec<f64> = values.iter().map(|value| value / scale).collect();
    let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
    let mut second = 0.0;
    let mut third = 0.0;
    let mut fourth = 0.0;
    for value in scaled {
        let centered = value - mean;
        let square = centered * centered;
        second += square;
        third += square * centered;
        fourth += square * square;
    }
    second /= values.len() as f64;
    third /= values.len() as f64;
    fourth /= values.len() as f64;
    if second <= 1e-30 {
        return (0.0, 0.0);
    }
    (
        finite_stat(third / second.powf(1.5)),
        finite_stat(fourth / (second * second) - 3.0),
    )
}

fn three_iqr_outlier_count(values: &[f64]) -> usize {
    if values.len() < 2 {
        return 0;
    }
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return 0;
    }
    let mut scaled: Vec<f64> = values.iter().map(|value| value / scale).collect();
    scaled.sort_by(f64::total_cmp);
    let q1 = quantile_sorted(&scaled, 0.25);
    let q3 = quantile_sorted(&scaled, 0.75);
    let iqr = q3 - q1;
    let lower = q1 - 3.0 * iqr;
    let upper = q3 + 3.0 * iqr;
    scaled
        .iter()
        .filter(|value| **value < lower || **value > upper)
        .count()
}

fn optional_correlation(left: &[Option<f64>], right: &[Option<f64>]) -> f64 {
    let pairs: Vec<(f64, f64)> = left
        .iter()
        .zip(right)
        .filter_map(|(left, right)| Some((left.as_ref()?, right.as_ref()?)))
        .map(|(left, right)| (*left, *right))
        .collect();
    stable_correlation(&pairs)
}

fn optional_target_correlation(values: &[Option<f64>], targets: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = values
        .iter()
        .zip(targets)
        .filter_map(|(value, target)| value.map(|value| (value, *target)))
        .collect();
    stable_correlation(&pairs)
}

fn stable_correlation(pairs: &[(f64, f64)]) -> f64 {
    if pairs.len() < 2 {
        return 0.0;
    }
    let left_scale = pairs.iter().map(|(left, _)| left.abs()).fold(0.0, f64::max);
    let right_scale = pairs
        .iter()
        .map(|(_, right)| right.abs())
        .fold(0.0, f64::max);
    if left_scale == 0.0 || right_scale == 0.0 {
        return 0.0;
    }
    let left_mean =
        pairs.iter().map(|(left, _)| left / left_scale).sum::<f64>() / pairs.len() as f64;
    let right_mean = pairs
        .iter()
        .map(|(_, right)| right / right_scale)
        .sum::<f64>()
        / pairs.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in pairs {
        let left = left / left_scale - left_mean;
        let right = right / right_scale - right_mean;
        covariance += left * right;
        left_variance += left * left;
        right_variance += right * right;
    }
    if left_variance <= 1e-30 || right_variance <= 1e-30 {
        0.0
    } else {
        finite_stat(covariance / (left_variance * right_variance).sqrt()).clamp(-1.0, 1.0)
    }
}

fn categorical_correlation_ratio(values: &[Option<&str>], targets: &[f64]) -> f64 {
    let observed: Vec<(&str, f64)> = values
        .iter()
        .zip(targets)
        .filter_map(|(value, target)| value.map(|value| (value, *target)))
        .collect();
    if observed.len() < 2 {
        return 0.0;
    }
    let scale = observed
        .iter()
        .map(|(_, target)| target.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let mean = observed
        .iter()
        .map(|(_, target)| target / scale)
        .sum::<f64>()
        / observed.len() as f64;
    let total = observed
        .iter()
        .map(|(_, target)| {
            let centered = target / scale - mean;
            centered * centered
        })
        .sum::<f64>();
    if total <= 1e-30 {
        return 0.0;
    }
    let mut groups = BTreeMap::<&str, (usize, f64)>::new();
    for (value, target) in observed {
        let entry = groups.entry(value).or_default();
        entry.0 += 1;
        entry.1 += target / scale;
    }
    let between = groups
        .values()
        .map(|(count, sum)| {
            let delta = sum / *count as f64 - mean;
            *count as f64 * delta * delta
        })
        .sum::<f64>();
    finite_stat((between / total).clamp(0.0, 1.0).sqrt())
}

fn duplicate_raw_row_fraction(table: &RawTable) -> f64 {
    let mut unique = BTreeSet::<[u8; 32]>::new();
    for row in &table.rows {
        let mut hasher = Sha256::new();
        hasher.update((row.len() as u64).to_le_bytes());
        for field in row {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        unique.insert(hasher.finalize().into());
    }
    (table.rows.len() - unique.len()) as f64 / table.rows.len().max(1) as f64
}

fn finite_stat(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}
