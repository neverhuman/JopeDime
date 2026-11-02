use super::*;

fn stage_append(stage: &Stage) -> bool {
    matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1
}

pub(crate) fn fit_v3_op(
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
