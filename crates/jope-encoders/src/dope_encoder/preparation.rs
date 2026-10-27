//! Separate product preparation from the historical Cap32 input contract.
use super::{Head, TrainingInput, LOSSLESS_PREPARATION_RECIPE};
use anyhow::{ensure, Context, Result};
use dope_core::prepare::{
    prepare_table_mixed, prepare_table_mixed_with_policy, prepare_table_with_ids,
    prepare_table_with_policy, ClassVocabPolicy, PrepTask, PreparedTable,
};
use sha2::{Digest, Sha256};

/// Historical train-only preparation. Retains the legacy Cap32V1/full-report inputs.
pub fn prepare_training_input(input: &TrainingInput) -> Result<PreparedTable> {
    prepare_with_policy(input, ClassVocabPolicy::Cap32V1)
}

/// Explicit Direct/JopeDL preparation with uncapped classes and exact FIT-label checks.
/// LosslessV1 still rounds upstream labels; any changed labels or codes are refused here.
/// This prepares FIT observations only and grants no query-label or training admission.
pub fn prepare_product_training_input(input: &TrainingInput) -> Result<PreparedTable> {
    let mut prepared = prepare_with_policy(input, ClassVocabPolicy::LosslessV1)?;
    require_product_classes(input, &prepared)?;
    // Upstream binary identity binds codes and the positive label, not every raw FIT target.
    // Bind the complete original target rows without changing descriptors or legacy digests.
    let mut source = Sha256::new();
    source.update(LOSSLESS_PREPARATION_RECIPE.as_bytes());
    source.update(b"\0");
    source.update(prepared.source_rows_digest().as_bytes());
    source.update((input.y.len() as u64).to_le_bytes());
    for label in &input.y {
        source.update(label.to_bits().to_le_bytes());
    }
    prepared.set_source_digest(format!("{:x}", source.finalize()));
    Ok(prepared)
}

fn prepare_with_policy(input: &TrainingInput, policy: ClassVocabPolicy) -> Result<PreparedTable> {
    input.validate()?;
    let task = match input.head {
        super::Head::Regression => PrepTask::Regression,
        super::Head::Binary => PrepTask::Classification,
        super::Head::Multiclass(_) => PrepTask::Multiclass,
    };
    let mut prepared = if input.categories.iter().any(Option::is_some) {
        let cells = input.raw_cells.clone().unwrap_or_else(|| {
            input
                .x
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, v)| {
                            if !v.is_finite() {
                                String::new()
                            } else if input.categories[j].is_some() {
                                format!("category:{:020}", *v as usize)
                            } else {
                                v.to_string()
                            }
                        })
                        .collect()
                })
                .collect::<Vec<Vec<String>>>()
        });
        let kinds = input
            .categories
            .iter()
            .map(|c| {
                if c.is_some() {
                    dope_core::prepare::scaling::ColumnKind::Categorical
                } else {
                    dope_core::prepare::scaling::ColumnKind::Numeric
                }
            })
            .collect::<Vec<_>>();
        match policy {
            ClassVocabPolicy::Cap32V1 => prepare_table_mixed(
                &cells,
                &input.y,
                Some(&kinds),
                task,
                Some(input.ids.clone()),
            )?,
            ClassVocabPolicy::LosslessV1 => prepare_table_mixed_with_policy(
                &cells,
                &input.y,
                Some(&kinds),
                task,
                Some(input.ids.clone()),
                policy,
            )?,
        }
    } else {
        match policy {
            ClassVocabPolicy::Cap32V1 => {
                prepare_table_with_ids(&input.x, &input.y, task, input.ids.clone())?
            }
            ClassVocabPolicy::LosslessV1 => {
                prepare_table_with_policy(&input.x, &input.y, task, input.ids.clone(), policy)?
            }
        }
    };
    if let Some(digest) = &input.source_digest {
        // External file identity supplements the fitting task/rows; it must never replace them.
        let bound = jope_runtime::sha256_hex(&serde_json::to_vec(&(
            "jope.dope-fitting-source/v1",
            prepared.source_rows_digest(),
            digest,
        ))?);
        prepared = prepared.with_source_digest(bound);
    }
    Ok(prepared)
}

fn require_product_classes(input: &TrainingInput, prepared: &PreparedTable) -> Result<()> {
    ensure!(
        prepared.transform.class_vocab_policy == ClassVocabPolicy::LosslessV1
            && prepared.row_ids == input.ids
            && prepared.rows == input.y.len(),
        "product preparation policy or physical FIT row ids differ"
    );
    if matches!(input.head, Head::Regression) {
        return Ok(());
    }
    ensure!(
        input.y.iter().all(|v| v.is_finite()),
        "product class labels must be finite"
    );
    let mut classes = input.y.clone();
    classes.sort_by(f64::total_cmp);
    classes.dedup_by(|a, b| *a == *b);
    match input.head {
        Head::Binary => ensure!(
            classes.len() == 2 && prepared.classes == 2,
            "binary product requires two FIT classes"
        ),
        Head::Multiclass(k) => ensure!(
            k >= 3 && classes.len() == k && prepared.classes == k
                && prepared.transform.target.class_labels == classes,
            "product requires an exact uncapped FIT class vocabulary; rounded or merged labels are unsupported"
        ),
        Head::Regression => unreachable!(),
    }
    ensure!(
        prepared.y_scaled.len() == input.y.len() && prepared.frame_y.len() == input.y.len(),
        "product target rows differ"
    );
    for (row, label) in input.y.iter().enumerate() {
        // Equality canonicalizes signed zero without tolerances or nearest-class assignment.
        let code = classes
            .binary_search_by(|class| {
                if class == label {
                    std::cmp::Ordering::Equal
                } else {
                    class.total_cmp(label)
                }
            })
            .ok()
            .context("product target is outside the exact FIT vocabulary")?
            as f64;
        ensure!(
            prepared.y_scaled[row] == code && prepared.frame_y[row] == code,
            "product FIT class row code differs; binary threshold remapping and nearest-class assignment are unsupported"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
