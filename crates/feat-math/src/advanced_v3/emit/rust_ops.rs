use super::rust_input::*;
use super::*;

pub(super) fn rust_numeric_ops_body(ops: &[V3Op]) -> Result<String> {
    let mut source = String::new();
    for (index, op) in ops.iter().enumerate() {
        let append = match op {
            V3Op::Legacy(op) => match op {
                FittedOp::Standardize { append, .. }
                | FittedOp::RobustScale { append, .. }
                | FittedOp::TargetBinEncode { append, .. }
                | FittedOp::Sigmoid { append }
                | FittedOp::Step { append, .. }
                | FittedOp::NormRow { append }
                | FittedOp::Lag { append, .. }
                | FittedOp::Roll { append, .. }
                | FittedOp::SelectCorr { append, .. }
                | FittedOp::WinsorClip { append, .. }
                | FittedOp::Impute { append, .. } => *append,
                _ => bail!("this legacy transform is not supported by bound emission"),
            },
            V3Op::MaxAbs { append, .. }
            | V3Op::YeoJohnson { append, .. }
            | V3Op::PiecewiseLinear { append, .. }
            | V3Op::CorrScreen { append, .. }
            | V3Op::PlsProject { append, .. } => *append,
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                bail!("table-aware row operation requires batch emission")
            }
        };
        source.push_str(&format!("    let generated_{index}: Vec<f64> = "));
        match op {
            V3Op::MaxAbs { scales, .. } => source.push_str(&format!(
                "x.iter().zip({scales}.iter()).map(|(value, scale)| value / scale).collect();\n",
                scales = rust_f64_slice(scales)
            )),
            V3Op::YeoJohnson { lambdas, .. } => source.push_str(&format!(
                "x.iter().zip({lambdas}.iter()).map(|(value, lambda)| yj(*value, *lambda)).collect();\n",
                lambdas = rust_f64_slice(lambdas)
            )),
            V3Op::CorrScreen { indices, .. } => source.push_str(&format!(
                "{indices}.iter().map(|index| x[*index]).collect();\n",
                indices = rust_usize_slice(indices)
            )),
            V3Op::PiecewiseLinear { edges, .. } => {
                source.push_str("{ let mut generated = Vec::new();\n");
                for (j, column) in edges.iter().enumerate() {
                    for interval in column.windows(2) {
                        source.push_str(&format!(
                            "        generated.push(piece(x[{j}], {lo}, {hi}));\n",
                            lo = rust_f64(interval[0]),
                            hi = rust_f64(interval[1])
                        ));
                    }
                }
                source.push_str("        generated };\n");
            }
            V3Op::PlsProject { means, weights, .. } => {
                source.push_str("{ let mut generated = Vec::new();\n");
                for weight in weights {
                    source.push_str(&format!(
                        "        generated.push(dot_centered(&x, &{means}, &{weight}));\n",
                        means = rust_f64_slice(means),
                        weight = rust_f64_slice(weight)
                    ));
                }
                source.push_str("        generated };\n");
            }
            V3Op::Legacy(op) => match op {
                FittedOp::Standardize { means, scales, .. } => source.push_str(&format!(
                    "x.iter().zip({means}.iter()).zip({scales}.iter()).map(|((value, mean), scale)| ((value-mean)/scale).clamp(-20.0, 20.0)).collect();\n",
                    means = rust_f64_slice(means),
                    scales = rust_f64_slice(scales),
                )),
                FittedOp::RobustScale {
                    medians, scales, ..
                } => source.push_str(&format!(
                    "x.iter().zip({medians}.iter()).zip({scales}.iter()).map(|((value, median), scale)| (value-median)/scale).collect();\n",
                    medians = rust_f64_slice(medians),
                    scales = rust_f64_slice(scales),
                )),
                FittedOp::TargetBinEncode { edges, enc, .. } => {
                    source.push_str("{ let mut generated = Vec::new();\n");
                    for (j, (column_edges, column_enc)) in edges.iter().zip(enc).enumerate() {
                        source.push_str(&format!(
                            "        let edges: &[f64] = &{edges}; let encoded: &[f64] = &{encoded}; let bin = edges.partition_point(|edge| *edge < x[{j}]).min(encoded.len()-1); generated.push(encoded[bin]);\n",
                            edges = rust_f64_slice(column_edges),
                            encoded = rust_f64_slice(column_enc),
                        ));
                    }
                    source.push_str("        generated };\n");
                }
                // Mirrors ops.rs apply_op exactly (single-branch form) — the
                // artifact's two-branch `sigmoid` is the HEAD formula and
                // differs from the transform by up to 1 ULP for negatives.
                FittedOp::Sigmoid { .. } => source.push_str(
                    "x.iter().map(|value| 1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())).collect();\n",
                ),
                FittedOp::Step { t, .. } => source.push_str(&format!(
                    "x.iter().map(|value| if *value > {} {{ 1.0 }} else {{ 0.0 }}).collect();\n",
                    rust_f64(*t)
                )),
                FittedOp::NormRow { .. } => source.push_str(
                    "{ let inv = 1.0 / x.iter().map(|value| value*value).sum::<f64>().sqrt().max(1e-9); x.iter().map(|value| value*inv).collect() };\n",
                ),
                FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
                    bail!("legacy row operation is invalid in advanced-v3 emission")
                }
                FittedOp::SelectCorr { indices, .. } => source.push_str(&format!(
                    "{indices}.iter().map(|j| x[*j]).collect();\n",
                    indices = rust_usize_slice(indices)
                )),
                FittedOp::WinsorClip { lo, hi, .. } => source.push_str(&format!(
                    "x.iter().zip({lo}.iter()).zip({hi}.iter()).map(|((value, lo), hi)| value.clamp(*lo, *hi)).collect();\n",
                    lo = rust_f64_slice(lo),
                    hi = rust_f64_slice(hi),
                )),
                FittedOp::Impute {
                    fill,
                    indicator_cols,
                    ..
                } => {
                    source.push_str(&format!(
                        "{{ let fill: &[f64] = &{}; let mut generated: Vec<f64> = x.iter().zip(fill).map(|(value, fill)| if *value == 0.0 {{ *fill }} else {{ *value }}).collect();",
                        rust_f64_slice(fill)
                    ));
                    for column in indicator_cols {
                        source.push_str(&format!(
                            " generated.push(if x[{column}] == 0.0 {{ 1.0 }} else {{ 0.0 }});"
                        ));
                    }
                    source.push_str(" generated };\n");
                }
                _ => unreachable!(),
            },
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                bail!("table-aware row operation requires batch emission")
            }
        }
        if append {
            source.push_str(&format!("    x.extend(generated_{index});\n"));
        } else {
            source.push_str(&format!("    x = generated_{index};\n"));
        }
        source.push_str("    for value in &mut x { if !value.is_finite() { *value = 0.0; } }\n");
    }
    source.push_str("    x\n");
    Ok(source)
}

pub(super) fn rust_bool_slice(values: &[bool]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(bool::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn v3_append(op: &V3Op) -> Result<bool> {
    Ok(match op {
        V3Op::Legacy(op) => match op {
            FittedOp::Standardize { append, .. }
            | FittedOp::RobustScale { append, .. }
            | FittedOp::TargetBinEncode { append, .. }
            | FittedOp::Sigmoid { append }
            | FittedOp::Step { append, .. }
            | FittedOp::NormRow { append }
            | FittedOp::SelectCorr { append, .. }
            | FittedOp::WinsorClip { append, .. }
            | FittedOp::Impute { append, .. } => *append,
            FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
                bail!("legacy row operation is invalid in advanced-v3 emission")
            }
            _ => bail!("legacy transform is not supported by bound emission"),
        },
        V3Op::MaxAbs { append, .. }
        | V3Op::YeoJohnson { append, .. }
        | V3Op::PiecewiseLinear { append, .. }
        | V3Op::CorrScreen { append, .. }
        | V3Op::PlsProject { append, .. }
        | V3Op::Lag { append, .. }
        | V3Op::Roll { append, .. } => *append,
    })
}

pub(super) fn rust_generated_mask_expr(op: &V3Op) -> Result<String> {
    let expression = match op {
        V3Op::MaxAbs { .. } | V3Op::YeoJohnson { .. } => "missing.clone()".to_string(),
        V3Op::CorrScreen { indices, .. } => format!(
            "{}.iter().map(|index| missing[*index]).collect()",
            rust_usize_slice(indices)
        ),
        V3Op::PiecewiseLinear { edges, .. } => {
            let items = edges
                .iter()
                .enumerate()
                .flat_map(|(index, column)| {
                    std::iter::repeat_n(format!("missing[{index}]"), column.len().saturating_sub(1))
                })
                .collect::<Vec<_>>();
            format!("vec![{}]", items.join(","))
        }
        V3Op::PlsProject { weights, .. } => format!("vec![false; {}]", weights.len()),
        V3Op::Legacy(op) => match op {
            FittedOp::Standardize { .. }
            | FittedOp::RobustScale { .. }
            | FittedOp::TargetBinEncode { .. }
            | FittedOp::Sigmoid { .. }
            | FittedOp::Step { .. }
            | FittedOp::NormRow { .. }
            | FittedOp::WinsorClip { .. } => "missing.clone()".to_string(),
            FittedOp::SelectCorr { indices, .. } => format!(
                "{}.iter().map(|index| missing[*index]).collect()",
                rust_usize_slice(indices)
            ),
            FittedOp::Impute { indicator_cols, .. } => {
                format!("vec![false; missing.len() + {}]", indicator_cols.len())
            }
            FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
                bail!("legacy row operation is invalid in advanced-v3 emission")
            }
            _ => bail!("legacy transform is not supported by bound emission"),
        },
        V3Op::Lag { .. } | V3Op::Roll { .. } => {
            bail!("row-operation masks require batch emission")
        }
    };
    Ok(expression)
}

pub(super) fn rust_semantic_key(key: &SemanticKey) -> String {
    match key {
        SemanticKey::Missing => "SemanticKey::Missing".to_string(),
        SemanticKey::Numeric(value) => format!("SemanticKey::Numeric(FiniteKey({}))", value.0),
        SemanticKey::Categorical(value) => {
            format!(
                "SemanticKey::Categorical({}.to_string())",
                rust_string(value)
            )
        }
    }
}

pub(super) fn rust_group_tail(tail: &GroupTail) -> String {
    let key = tail
        .key
        .iter()
        .map(rust_semantic_key)
        .collect::<Vec<_>>()
        .join(",");
    let last_order = tail
        .last_order
        .iter()
        .map(rust_semantic_key)
        .collect::<Vec<_>>()
        .join(",");
    let rows = tail
        .rows
        .iter()
        .map(|row| {
            format!(
                "({}.to_vec(),{}.to_vec())",
                rust_f64_slice(&row.values),
                rust_bool_slice(&row.missing)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("histories.insert(vec![{key}], vec![{rows}]);\n    fitted_cutoffs.insert(vec![{key}], vec![{last_order}]);\n")
}

pub(super) fn rust_key_vector(columns: &[(usize, super::schema::ColumnKind)]) -> String {
    columns
        .iter()
        .map(|(index, kind)| {
            format!(
                "semantic_key(&row[{index}], {})?",
                matches!(kind, super::schema::ColumnKind::Numeric)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn rust_row_operation_body(index: usize, op: &V3Op) -> Result<String> {
    let (name, k, append, order, tails) = match op {
        V3Op::Lag {
            k,
            append,
            order,
            tails,
        } => ("lag", *k, *append, order, tails),
        V3Op::Roll {
            k,
            append,
            order,
            tails,
        } => ("roll", *k, *append, order, tails),
        _ => bail!("expected a row operation"),
    };
    let group_keys = rust_key_vector(&order.group_columns);
    let order_keys = rust_key_vector(&order.order_columns);
    let mut source = format!(
        "    let group_keys_{index}: Vec<Vec<SemanticKey>> = raw.iter().map(|row| -> Result<Vec<SemanticKey>, String> {{ Ok(vec![{group_keys}]) }}).collect::<Result<Vec<_>, _>>()?;\n    let order_keys_{index}: Vec<Vec<SemanticKey>> = raw.iter().map(|row| -> Result<Vec<SemanticKey>, String> {{ Ok(vec![{order_keys}]) }}).collect::<Result<Vec<_>, _>>()?;\n    let mut chronological_{index}: Vec<usize> = (0..rows.len()).collect();\n    chronological_{index}.sort_by(|left, right| group_keys_{index}[*left].cmp(&group_keys_{index}[*right]).then(order_keys_{index}[*left].cmp(&order_keys_{index}[*right])).then(left.cmp(right)));\n    let width_{index} = rows.first().map_or(0, Vec::len);\n    let mut generated_{index} = vec![vec![0.0; width_{index}]; rows.len()];\n    let mut generated_missing_{index} = vec![vec![true; width_{index}]; rows.len()];\n    let mut histories: std::collections::BTreeMap<Vec<SemanticKey>, Vec<(Vec<f64>, Vec<bool>)>> = std::collections::BTreeMap::new();\n    let mut fitted_cutoffs: std::collections::BTreeMap<Vec<SemanticKey>, Vec<SemanticKey>> = std::collections::BTreeMap::new();\n"
    );
    for tail in tails {
        source.push_str("    ");
        source.push_str(&rust_group_tail(tail));
    }
    source.push_str(&format!(
        "    for row_index in 0..rows.len() {{ if let Some(cutoff) = fitted_cutoffs.get(&group_keys_{index}[row_index]) {{ if order_keys_{index}[row_index].as_slice() <= cutoff.as_slice() {{ return Err(\"prediction row order must be strictly after fitted order\".to_string()); }} }} }}\n"
    ));
    source.push_str(&format!(
        "    for row_index in chronological_{index} {{\n        let history = histories.entry(group_keys_{index}[row_index].clone()).or_default();\n"
    ));
    if name == "lag" {
        source.push_str(&format!(
            "        if history.len() >= {k} {{ generated_{index}[row_index].clone_from(&history[history.len()-{k}].0); generated_missing_{index}[row_index].clone_from(&history[history.len()-{k}].1); }}\n"
        ));
    } else {
        source.push_str(&format!(
            "        for feature in 0..width_{index} {{ let mut sum = 0.0; let mut count = 0usize; let start = history.len().saturating_sub({previous}); for prior in &history[start..] {{ if !prior.1[feature] && prior.0[feature].is_finite() {{ sum += prior.0[feature]; count += 1; }} }} if !masks[row_index][feature] && rows[row_index][feature].is_finite() {{ sum += rows[row_index][feature]; count += 1; }} if count > 0 {{ generated_{index}[row_index][feature] = sum / count as f64; generated_missing_{index}[row_index][feature] = false; }} }}\n",
            previous = k.saturating_sub(1)
        ));
    }
    let keep = if name == "lag" {
        k
    } else {
        k.saturating_sub(1)
    };
    source.push_str(&format!(
        "        history.push((rows[row_index].clone(), masks[row_index].clone()));\n        if history.len() > {keep} {{ history.drain(..history.len()-{keep}); }}\n    }}\n"
    ));
    if append {
        source.push_str(&format!(
            "    for row_index in 0..rows.len() {{ rows[row_index].extend(std::mem::take(&mut generated_{index}[row_index])); masks[row_index].extend(std::mem::take(&mut generated_missing_{index}[row_index])); }}\n"
        ));
    } else {
        source.push_str(&format!(
            "    rows = generated_{index};\n    masks = generated_missing_{index};\n"
        ));
    }
    source.push_str(
        "    for (row, mask) in rows.iter_mut().zip(&masks) { for (value, absent) in row.iter_mut().zip(mask) { if *absent || !value.is_finite() { *value = 0.0; } } }\n",
    );
    Ok(source)
}

pub(super) fn rust_batch_ops_source(ops: &[V3Op]) -> Result<(String, String)> {
    let mut helpers = String::new();
    let mut body = String::new();
    for (index, op) in ops.iter().enumerate() {
        if matches!(op, V3Op::Lag { .. } | V3Op::Roll { .. }) {
            body.push_str(&rust_row_operation_body(index, op)?);
            continue;
        }
        let numeric_body = rust_numeric_ops_body(std::slice::from_ref(op))?;
        helpers.push_str(&format!(
            "fn numeric_op_{index}(mut x: Vec<f64>) -> Vec<f64> {{\n{numeric_body}}}\n\n"
        ));
        let mask_expr = rust_generated_mask_expr(op)?;
        let append = v3_append(op)?;
        body.push_str(&format!(
            "    for row_index in 0..rows.len() {{\n        let missing = masks[row_index].clone();\n        let generated_missing: Vec<bool> = {mask_expr};\n        rows[row_index] = numeric_op_{index}(std::mem::take(&mut rows[row_index]));\n"
        ));
        if append {
            body.push_str("        masks[row_index].extend(generated_missing);\n");
        } else {
            body.push_str("        masks[row_index] = generated_missing;\n");
        }
        body.push_str(
            "        for (value, absent) in rows[row_index].iter_mut().zip(&masks[row_index]) { if *absent || !value.is_finite() { *value = 0.0; } }\n    }\n",
        );
    }
    body.push_str("    Ok(rows)\n");
    Ok((helpers, body))
}
