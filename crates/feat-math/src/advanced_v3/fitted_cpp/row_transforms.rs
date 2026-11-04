fn emit_input(model: &BoundModel, missing_aware: bool) -> Result<String> {
    let mut body = String::from(
        "static bool encode_row(const double* raw,const std::uint8_t* missing,V& x,V& mask){\n  x.clear(); mask.clear();\n",
    );
    for (column, fitted) in model.input.columns.iter().enumerate() {
        match fitted {
            FittedColumn::Numeric { fill, indicator } => {
                body.push_str(&format!(
                    "  const bool absent_{column}=missing[{column}]!=0||!std::isfinite(raw[{column}]); const double value_{column}=absent_{column}?{fill}:raw[{column}]; x.push_back(value_{column}); mask.push_back(absent_{column}?1.0:0.0);\n",
                    fill = f64_lit(*fill),
                ));
                if *indicator {
                    body.push_str(&format!(
                        "  x.push_back(absent_{column}?1.0:0.0); mask.push_back(0.0);\n"
                    ));
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
                body.push_str(&format!(
                    "  const bool absent_{column}=missing[{column}]!=0; (void)absent_{column}; const double encoded_{column}=raw[{column}]; if(!std::isfinite(encoded_{column})||std::floor(encoded_{column})!=encoded_{column}||encoded_{column} < -1.0||encoded_{column}>={vocab}.0)return false; const std::ptrdiff_t index_{column}=static_cast<std::ptrdiff_t>(encoded_{column});\n",
                    vocab = vocabulary.len(),
                ));
                match mode {
                    CategoricalMode::Native => body.push_str(&format!(
                        "  x.push_back(static_cast<double>(index_{column})); mask.push_back(absent_{column}?1.0:0.0);\n"
                    )),
                    CategoricalMode::Frequency => body.push_str(&format!(
                        "  {{const V counts={counts}; x.push_back(index_{column}<0?0.0:counts[static_cast<std::size_t>(index_{column})]/{total}.0); mask.push_back(absent_{column}?1.0:0.0);}}\n",
                        counts = f64_vec(&counts.iter().map(|v| *v as f64).collect::<Vec<_>>()),
                    )),
                    CategoricalMode::Onehot => {
                        for category in onehot {
                            body.push_str(&format!(
                                "  x.push_back(index_{column}=={category}?1.0:0.0); mask.push_back(0.0);\n"
                            ));
                        }
                    }
                    CategoricalMode::Ordered | CategoricalMode::Hybrid => {
                        let ordered = match model.input.historical.get(column).and_then(Option::as_ref) {
                            Some(state) => emit_historical_ordered(model, state, column)?,
                            None => format!(
                                "const V counts={counts};const V sums={sums};double ordered=index_{column}<0?{global}:(sums[static_cast<std::size_t>(index_{column})]+{prior}*{global})/(counts[static_cast<std::size_t>(index_{column})]+{prior});",
                                counts = f64_vec(&counts.iter().map(|v| *v as f64).collect::<Vec<_>>()),
                                sums = f64_vec(sums),
                                global = f64_lit(*global),
                                prior = f64_lit(model.input.config.prior),
                            ),
                        };
                        body.push_str(&format!(
                            "  {{{ordered}",
                        ));
                        if *mode == CategoricalMode::Hybrid {
                            body.push_str(&format!(
                                "x.push_back(index_{column}<0?0.0:counts[static_cast<std::size_t>(index_{column})]/{total}.0);mask.push_back(absent_{column}?1.0:0.0);"
                            ));
                        }
                        body.push_str(&format!(
                            "x.push_back(ordered);mask.push_back({});}}\n",
                            if *mode == CategoricalMode::Ordered {
                                format!("absent_{column}?1.0:0.0")
                            } else {
                                "0.0".to_string()
                            }
                        ));
                    }
                }
            }
        }
    }
    if missing_aware {
        body.push_str(
            "  for(std::size_t i=0;i<x.size();++i)if(mask[i]!=0.0)x[i]=std::numeric_limits<double>::quiet_NaN();\n",
        );
    }
    body.push_str("  return true;\n}\n");
    Ok(body)
}

fn emit_historical_ordered(
    model: &BoundModel,
    state: &HistoricalTargetState,
    target_column: usize,
) -> Result<String> {
    let FittedRowOrder::Ordered { group_columns, .. } = &model.input.row_order else {
        bail!("historical target state is attached to IID row semantics");
    };
    let global = if state.count == 0 {
        0.0
    } else {
        state.sum / state.count as f64
    };
    let mut body = format!("double ordered={};", f64_lit(global));
    for group in &state.groups {
        if group.key.len() != group_columns.len() {
            bail!("historical group key width differs from fitted row semantics");
        }
        let mut conditions = Vec::new();
        for ((column, kind), key) in group_columns.iter().zip(&group.key) {
            let condition = match (kind, key) {
                (_, SemanticKey::Missing) => format!("missing[{column}]!=0"),
                (ColumnKind::Numeric, SemanticKey::Numeric(value)) => format!(
                    "missing[{column}]==0&&raw[{column}]=={}",
                    f64_lit(f64::from_bits(value.0))
                ),
                (ColumnKind::Categorical, SemanticKey::Categorical(value)) => {
                    let FittedColumn::Categorical { vocabulary, .. } =
                        &model.input.columns[*column]
                    else {
                        bail!("historical categorical key references a numeric fitted column");
                    };
                    let index = vocabulary.binary_search(value).map_err(|_| {
                        anyhow::anyhow!(
                            "historical categorical key is absent from fitted vocabulary"
                        )
                    })?;
                    format!("missing[{column}]==0&&raw[{column}]=={index}.0")
                }
                _ => bail!("historical group key kind differs from fitted schema"),
            };
            conditions.push(condition);
        }
        let condition = if conditions.is_empty() {
            "true".to_string()
        } else {
            conditions.join("&&")
        };
        let fallback = if group.count == 0 {
            global
        } else {
            group.sum / group.count as f64
        };
        body.push_str(&format!(
            "if({condition}){{const double fallback={};ordered=fallback;",
            f64_lit(fallback)
        ));
        for category in &group.categories {
            let statistic = if category.count == 0 && model.input.config.prior == 0.0 {
                fallback
            } else {
                (category.sum + model.input.config.prior * fallback)
                    / (category.count as f64 + model.input.config.prior)
            };
            body.push_str(&format!(
                "if(index_{target_column}=={}.0)ordered={};",
                category.category,
                f64_lit(statistic)
            ));
        }
        body.push('}');
    }
    Ok(body)
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
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
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
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

fn emit_transforms(ops: &[V3Op]) -> Result<String> {
    emit_transforms_named(ops, "transform", false)
}

fn emit_transforms_named(ops: &[V3Op], name: &str, block: bool) -> Result<String> {
    let signature = if block { "Block" } else { "V" };
    let mut body = format!("static {signature} {name}(V x,V mask){{\n  (void)mask;\n");
    for op in ops {
        match op {
            V3Op::Legacy(op) => emit_legacy_op(&mut body, op)?,
            V3Op::MaxAbs { scales, append } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g;const V scales={};for(std::size_t j=0;j<std::min(x.size(),scales.size());++j)g.push_back(x[j]/scales[j]);V gm=mask;finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                f64_vec(scales), append
            )),
            V3Op::YeoJohnson { lambdas, append } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g;const V lambdas={};for(std::size_t j=0;j<std::min(x.size(),lambdas.size());++j)g.push_back(yj(x[j],lambdas[j]));V gm=mask;finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                f64_vec(lambdas), append
            )),
            V3Op::PiecewiseLinear { edges, append } => {
                body.push_str(&format!("  {{V input=x,input_mask=mask,g,gm;const M edges={};for(std::size_t j=0;j<std::min(x.size(),edges.size());++j)for(std::size_t k=1;k<edges[j].size();++k){{g.push_back(piece(x[j],edges[j][k-1],edges[j][k]));gm.push_back(mask[j]);}}finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n", f64_matrix(edges), append));
            }
            V3Op::CorrScreen { indices, append } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g,gm;const U indices={};for(std::size_t j:indices){{g.push_back(x[fitted_index(j,x.size())]);gm.push_back(mask[fitted_index(j,mask.size())]);}}finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                usize_vec(indices), append
            )),
            V3Op::PlsProject {
                means,
                weights,
                append,
            } => body.push_str(&format!(
                "  {{V input=x,input_mask=mask,g;const V means={};const M weights={};for(const V& w:weights){{double s=0.0;for(std::size_t j=0;j<std::min(x.size(),std::min(means.size(),w.size()));++j)s+=w[j]*(x[j]-means[j]);g.push_back(s);}}V gm(g.size(),0.0);finish_block(input,input_mask,std::move(g),std::move(gm),{},x,mask);}}\n",
                f64_vec(means), f64_matrix(weights), append
            )),
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                unreachable!("table-aware stages are emitted at the batch boundary")
            }
        }
    }
    body.push_str(if block {
        "  return Block{std::move(x),std::move(mask)};\n}\n"
    } else {
        "  return x;\n}\n"
    });
    Ok(body)
}

fn emit_legacy_transforms(ops: &[FittedOp]) -> Result<String> {
    emit_legacy_transforms_named(ops, "transform", false)
}

fn emit_legacy_transforms_named(ops: &[FittedOp], name: &str, block: bool) -> Result<String> {
    let signature = if block { "Block" } else { "V" };
    let mut body = format!("static {signature} {name}(V x,V mask){{\n  (void)mask;\n");
    for op in ops {
        emit_legacy_op_padded(&mut body, op)?;
    }
    body.push_str(if block {
        "  return Block{std::move(x),std::move(mask)};\n}\n"
    } else {
        "  return x;\n}\n"
    });
    Ok(body)
}

fn encoded_row_key_value(
    model: &BoundModel,
    column: usize,
    kind: ColumnKind,
    key: &RowSemanticKey,
) -> Result<String> {
    match (kind, key) {
        (_, RowSemanticKey::Missing) => Ok("KCell{false,0.0}".to_string()),
        (ColumnKind::Numeric, RowSemanticKey::Numeric(value)) => {
            Ok(format!("KCell{{true,{}}}", f64_lit(value.value())))
        }
        (ColumnKind::Categorical, RowSemanticKey::Categorical(value)) => {
            let FittedColumn::Categorical { vocabulary, .. } = &model.input.columns[column] else {
                bail!("row-operation categorical key references a numeric fitted column");
            };
            let index = vocabulary.binary_search(value).map_err(|_| {
                anyhow::anyhow!("row-operation categorical key is absent from fitted vocabulary")
            })?;
            Ok(format!("KCell{{true,{index}.0}}"))
        }
        _ => bail!("row-operation key kind differs from fitted schema"),
    }
}

fn row_key_literal(
    model: &BoundModel,
    columns: &[(usize, ColumnKind)],
    key: &[RowSemanticKey],
) -> Result<String> {
    if columns.len() != key.len() {
        bail!("row-operation stored key width mismatch");
    }
    Ok(format!(
        "Key{{{}}}",
        columns
            .iter()
            .zip(key)
            .map(|((column, kind), key)| encoded_row_key_value(model, *column, *kind, key))
            .collect::<Result<Vec<_>>>()?
            .join(",")
    ))
}

fn tail_rows_literal(tail: &GroupTail) -> String {
    format!(
        "std::vector<Block>{{{}}}",
        tail.rows
            .iter()
            .map(|row| format!(
                "Block{{{},{}}}",
                f64_vec(&row.values),
                f64_vec(
                    &row.missing
                        .iter()
                        .map(|missing| if *missing { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>()
                )
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}
