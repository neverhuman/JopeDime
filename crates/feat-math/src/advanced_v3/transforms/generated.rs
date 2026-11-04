use super::*;

pub(crate) fn sanitize_with_masks(values: &mut Rows, masks: &[Vec<bool>]) -> Result<()> {
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

pub(crate) fn lineage_with_operation(lineage: &FeatureLineage, operation: &str) -> FeatureLineage {
    let mut lineage = lineage.clone();
    lineage.operations.push(operation.to_string());
    lineage
}

pub(crate) fn source_lineage(
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

pub(crate) fn finish_generated_from_sets(
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

pub(crate) fn finish_one_to_one(
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

pub(crate) fn finish_selected(
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

pub(crate) fn finish_generated_dense(
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

pub(crate) fn finish_impute(
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
