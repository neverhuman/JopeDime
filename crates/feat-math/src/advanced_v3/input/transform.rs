use super::*;

impl FittedInput {
    pub(super) fn transform_impl(
        &self,
        table: &RawTable,
        train_y: Option<&[f64]>,
    ) -> Result<FeatureBlock> {
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
