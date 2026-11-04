use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryTargetSummaryV1 {
    pub split_view_manifest_hash: String,
    pub view_index: usize,
    pub training_rows_hash: String,
    pub training_row_count: usize,
    pub positive_row_count: usize,
    pub values: Vec<f64>,
    pub task: Task,
    pub schema: String,
}

impl BinaryTargetSummaryV1 {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "schema": self.schema,
            "task": self.task,
            "split_view_manifest_hash": self.split_view_manifest_hash,
            "view_index": self.view_index,
            "training_rows_hash": self.training_rows_hash,
            "training_row_count": self.training_row_count,
            "positive_row_count": self.positive_row_count,
            "values": self.values,
        })
    }
}

/// Summarize one selected view's training targets and bind their exact row hash.
pub fn binary_target_summary(
    training_targets: &[f64],
    training_row_ids: &[String],
    split_view_manifest: &SplitViewManifestV1,
    view_index: usize,
) -> Result<BinaryTargetSummaryV1> {
    ensure!(
        split_view_manifest.task == Task::Binary,
        "binary target summary requires a binary split view manifest"
    );
    ensure!(
        training_row_ids.len() == training_targets.len() && !training_targets.is_empty(),
        "binary targets and row identities must be non-empty and aligned"
    );
    ensure!(
        training_row_ids.iter().all(|id| !id.is_empty())
            && training_row_ids.iter().collect::<BTreeSet<_>>().len() == training_row_ids.len(),
        "training row identities must be non-empty and unique"
    );
    ensure!(
        training_targets
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0),
        "binary targets must be a strict 0/1 sequence"
    );
    let view = split_view_manifest
        .views
        .get(view_index)
        .ok_or_else(|| anyhow::anyhow!("binary target summary view_index must be in 0..5"))?;
    ensure!(
        training_row_ids == view.train_row_ids(),
        "binary target rows do not match the selected split view"
    );
    let actual_train_hash = row_set_hash(
        &training_row_ids
            .iter()
            .map(|row| sha256_hex(format!("{}|{row}", split_view_manifest.base_dataset).as_bytes()))
            .collect::<Vec<_>>(),
    )?;
    ensure!(
        actual_train_hash == view.train_hash,
        "binary target rows do not match the selected training hash"
    );

    let n = training_targets.len();
    ensure!(
        n >= 2,
        "binary target summary requires at least two training rows"
    );
    let positive = training_targets
        .iter()
        .filter(|value| **value == 1.0)
        .count();
    ensure!(
        positive > 0 && positive < n,
        "binary target summary requires both classes"
    );
    let rate = positive as f64 / n as f64;
    let defined = [
        n as f64 / (n as f64 + 1.0),
        rate,
        rate.min(1.0 - rate),
        -(rate * rate.ln() + (1.0 - rate) * (1.0 - rate).ln()),
        (rate / (1.0 - rate)).ln().clamp(-12.0, 12.0),
        0.5 / rate,
        0.5 / (1.0 - rate),
    ];
    let values = defined
        .into_iter()
        .map(|value| (value as f32) as f64)
        .chain(std::iter::repeat_n(0.0, 512 - 7))
        .collect::<Vec<_>>();
    Ok(BinaryTargetSummaryV1 {
        split_view_manifest_hash: split_view_manifest.sha256()?,
        view_index,
        training_rows_hash: view.train_hash.clone(),
        training_row_count: n,
        positive_row_count: positive,
        values,
        task: Task::Binary,
        schema: "BinaryTargetSummary.v1".to_owned(),
    })
}

/// Build all five train-only summaries for one binary base dataset.
pub fn build_binary_target_summaries(
    training_targets: &[f64],
    row_ids: &[String],
    split_view_manifest: &SplitViewManifestV1,
) -> Result<Vec<BinaryTargetSummaryV1>> {
    ensure!(
        training_targets.len() == row_ids.len()
            && row_ids.iter().collect::<BTreeSet<_>>().len() == row_ids.len(),
        "binary targets and unique row identities must be aligned"
    );
    let target_by_row = row_ids
        .iter()
        .cloned()
        .zip(training_targets.iter().copied())
        .collect::<BTreeMap<_, _>>();
    split_view_manifest
        .views
        .iter()
        .map(|view| {
            let targets =
                view.train_row_ids()
                    .iter()
                    .map(|row| {
                        target_by_row.get(row).copied().ok_or_else(|| {
                            anyhow::anyhow!("training row is absent from target table")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
            binary_target_summary(
                &targets,
                view.train_row_ids(),
                split_view_manifest,
                view.view_index,
            )
        })
        .collect()
}
