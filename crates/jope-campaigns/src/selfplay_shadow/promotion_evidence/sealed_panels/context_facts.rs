use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelContextFactsV1 {
    pub schema: String,
    pub task: Task,
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub train_sha256: Sha256Digest,
    pub test_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub origin: DatasetOriginV2,
    pub rows: usize,
    pub columns: usize,
    pub missing_fraction: String,
    pub size_slice: String,
    pub missingness_slice: String,
    pub imbalance_slice: String,
    pub target_shape_slice: String,
    pub created_at_ms: u128,
}

impl SealedPanelContextFactsV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA,
            "sealed panel context facts schema is not v1"
        );
        ensure!(
            !self.context_identity.trim().is_empty()
                && self.rows > 0
                && self.columns >= 2
                && [
                    self.missing_fraction.as_str(),
                    self.size_slice.as_str(),
                    self.missingness_slice.as_str(),
                    self.imbalance_slice.as_str(),
                    self.target_shape_slice.as_str(),
                ]
                .into_iter()
                .all(|value| !value.trim().is_empty()),
            "sealed panel context facts are incomplete"
        );
        let missing_fraction = self
            .missing_fraction
            .parse::<f64>()
            .context("sealed panel missing fraction is not numeric")?;
        ensure!(
            missing_fraction.is_finite() && (0.0..=1.0).contains(&missing_fraction),
            "sealed panel missing fraction is outside 0..1"
        );
        Ok(())
    }

    pub fn slices(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "origin".to_owned(),
                format!("{:?}", self.origin).to_ascii_lowercase(),
            ),
            ("size".to_owned(), self.size_slice.clone()),
            ("missingness".to_owned(), self.missingness_slice.clone()),
            ("imbalance".to_owned(), self.imbalance_slice.clone()),
            ("target_shape".to_owned(), self.target_shape_slice.clone()),
        ])
    }
}
