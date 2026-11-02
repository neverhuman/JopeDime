use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OuterSplit {
    Train,
    Holdout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitViewV1 {
    pub view_index: usize,
    pub train_rows: usize,
    pub test_rows: usize,
    pub row_assignment_hash: String,
    pub fold_hash: String,
    pub train_hash: String,
    pub test_hash: String,
    pub encoder_hash: String,
    #[serde(skip)]
    train_row_ids: Vec<String>,
    #[serde(skip)]
    test_row_ids: Vec<String>,
}

impl SplitViewV1 {
    pub fn train_row_ids(&self) -> &[String] {
        &self.train_row_ids
    }

    pub fn test_row_ids(&self) -> &[String] {
        &self.test_row_ids
    }

    fn to_json(&self) -> Value {
        serde_json::json!({
            "view_index": self.view_index,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "row_assignment_hash": self.row_assignment_hash,
            "fold_hash": self.fold_hash,
            "train_hash": self.train_hash,
            "test_hash": self.test_hash,
            "encoder_hash": self.encoder_hash,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitViewManifestV1 {
    pub base_dataset: String,
    pub source_family: String,
    pub task: Task,
    pub outer_split: OuterSplit,
    pub outer_split_hash: String,
    pub fold_plan_hash: String,
    pub views: Vec<SplitViewV1>,
    pub schema: String,
}

impl SplitViewManifestV1 {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "schema": self.schema,
            "base_dataset": self.base_dataset,
            "source_family": self.source_family,
            "task": self.task,
            "outer_split": self.outer_split,
            "outer_split_hash": self.outer_split_hash,
            "fold_plan_hash": self.fold_plan_hash,
            "views": self.views.iter().map(SplitViewV1::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn sha256(&self) -> Result<String> {
        canonical_hash(&self.to_json())
    }
}

/// Build five complementary, balanced views and train-only encoder bindings.
pub fn build_split_view_manifest(
    dataset: &str,
    source_family: &str,
    task: Task,
    outer_split: OuterSplit,
    row_ids: &[String],
    train_sha256: &str,
    test_sha256: Option<&str>,
) -> Result<SplitViewManifestV1> {
    ensure!(
        !dataset.is_empty() && !source_family.is_empty() && row_ids.len() >= 5,
        "dataset/family and at least five row identities are required"
    );
    let unique = row_ids.iter().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == row_ids.len() && row_ids.iter().all(|id| !id.is_empty()),
        "row identities must be non-empty and unique"
    );
    ensure!(
        is_sha256(train_sha256),
        "train_sha256 must be a lowercase SHA-256 hex digest"
    );
    if let Some(test_sha256) = test_sha256 {
        ensure!(
            is_sha256(test_sha256),
            "test_sha256 must be a lowercase SHA-256 hex digest"
        );
    }

    let row_hashes = row_ids
        .iter()
        .map(|row| {
            (
                row.clone(),
                sha256_hex(format!("{dataset}|{row}").as_bytes()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut ordered = row_ids.to_vec();
    ordered.sort_by_key(|row| row_hashes[row].clone());
    let mut buckets = vec![Vec::new(); 5];
    for (index, row) in ordered.iter().enumerate() {
        buckets[index % 5].push(row.clone());
    }

    let mut views = Vec::with_capacity(5);
    for (view_index, test_row_ids) in buckets.into_iter().enumerate() {
        let test_set = test_row_ids.iter().collect::<BTreeSet<_>>();
        let train_row_ids = ordered
            .iter()
            .filter(|row| !test_set.contains(*row))
            .cloned()
            .collect::<Vec<_>>();
        let train_row_hashes = train_row_ids
            .iter()
            .map(|row| row_hashes[row].clone())
            .collect::<Vec<_>>();
        let test_row_hashes = test_row_ids
            .iter()
            .map(|row| row_hashes[row].clone())
            .collect::<Vec<_>>();
        let train_hash = row_set_hash(&train_row_hashes)?;
        let test_hash = row_set_hash(&test_row_hashes)?;

        let mut assignment = Map::new();
        for row in &ordered {
            assignment.insert(
                row_hashes[row].clone(),
                Value::String(
                    if test_set.contains(row) {
                        "test"
                    } else {
                        "train"
                    }
                    .to_owned(),
                ),
            );
        }
        let row_assignment_hash = canonical_hash(&serde_json::json!({
            "schema": "SplitViewRowAssignment.v1",
            "view_index": view_index,
            "assignment": assignment,
        }))?;
        let fold_hash = canonical_hash(&serde_json::json!({
            "schema": "SplitViewFold.v1",
            "view_index": view_index,
            "row_assignment_hash": row_assignment_hash,
        }))?;
        let mut encoder_kinds = vec!["dataset", "target"];
        if task == Task::Binary {
            encoder_kinds.push("class_stats");
        }
        let encoder_hash = canonical_hash(&serde_json::json!({
            "schema": "TrainOnlyEncoderFits.v1",
            "task": task,
            "train_hash": train_hash,
            "encoders": encoder_kinds,
        }))?;
        views.push(SplitViewV1 {
            view_index,
            train_rows: train_row_ids.len(),
            test_rows: test_row_ids.len(),
            row_assignment_hash,
            fold_hash,
            train_hash,
            test_hash,
            encoder_hash,
            train_row_ids,
            test_row_ids,
        });
    }

    let outer_split_hash = canonical_hash(&serde_json::json!({
        "schema": "OuterSplitBinding.v1",
        "base_dataset": dataset,
        "source_family": source_family,
        "task": task,
        "outer_split": outer_split,
        "train_sha256": train_sha256,
        "test_sha256": test_sha256,
    }))?;
    let fold_plan_hash = canonical_hash(&serde_json::json!({
        "schema": "FiveComplementaryViews.v1",
        "outer_split_hash": outer_split_hash,
        "views": views.iter().map(SplitViewV1::to_json).collect::<Vec<_>>(),
    }))?;
    Ok(SplitViewManifestV1 {
        base_dataset: dataset.to_owned(),
        source_family: source_family.to_owned(),
        task,
        outer_split,
        outer_split_hash,
        fold_plan_hash,
        views,
        schema: "SplitViewManifest.v1".to_owned(),
    })
}
