use super::*;

impl FittedInput {
    pub(in crate::advanced_v3) fn compatibility_numeric(width: usize) -> Self {
        Self {
            config: InputConfig::compatibility(),
            columns: vec![
                FittedColumn::Numeric {
                    fill: 0.0,
                    indicator: false,
                };
                width
            ],
            row_order: FittedRowOrder::Iid,
            historical: vec![None; width],
        }
    }

    pub(in crate::advanced_v3) fn fit_transform(
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

    pub(in crate::advanced_v3) fn transform(&self, table: &RawTable) -> Result<FeatureBlock> {
        self.transform_impl(table, None)
    }

    pub(in crate::advanced_v3) fn transform_numeric_rows(
        &self,
        x: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        let mut out_width = 0usize;
        for fitted in &self.columns {
            match fitted {
                FittedColumn::Numeric { indicator, .. } => {
                    out_width += 1 + usize::from(*indicator);
                }
                FittedColumn::Categorical { .. } => {
                    bail!("transform_numeric_rows requires all-numeric fitted columns");
                }
            }
        }
        let mut values: Vec<Vec<f64>> = (0..x.len())
            .map(|_| Vec::with_capacity(out_width))
            .collect();
        for (j, fitted) in self.columns.iter().enumerate() {
            let FittedColumn::Numeric { fill, indicator } = fitted else {
                unreachable!();
            };
            for (r, row) in x.iter().enumerate() {
                let v = row[j];
                values[r].push(if v.is_finite() { v } else { *fill });
            }
            if *indicator {
                for (r, row) in x.iter().enumerate() {
                    let v = row[j];
                    values[r].push(if v.is_finite() { 0.0 } else { 1.0 });
                }
            }
        }
        Ok(values)
    }
}
