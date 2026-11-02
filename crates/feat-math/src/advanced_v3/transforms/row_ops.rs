use super::*;

const GROUP_TAIL_STATE_VERSION: &str = "advanced-v3-group-tail-v1";

pub(super) fn group_tail_hash(tail: &GroupTail) -> Result<String> {
    stable_hash(&(
        GROUP_TAIL_STATE_VERSION,
        &tail.key,
        &tail.last_order,
        &tail.rows,
    ))
}

pub(super) fn validate_stored_key(key: &SemanticKey, kind: ColumnKind) -> bool {
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

pub(super) fn validate_group_tail_keys(tail: &GroupTail, order: &FittedTableOrder) -> Result<()> {
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

pub(super) fn parse_numeric(raw: &str) -> Result<Option<f64>> {
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

pub(super) fn category_is_missing(raw: &str) -> bool {
    let text = raw.trim();
    text.is_empty() || text.eq_ignore_ascii_case("na") || text.eq_ignore_ascii_case("null")
}

pub(super) fn semantic_key(raw: &str, kind: ColumnKind) -> Result<SemanticKey> {
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

pub(super) fn table_keys(
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

pub(super) fn finish_row_operation(
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

pub(super) fn apply_row_operation(
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
