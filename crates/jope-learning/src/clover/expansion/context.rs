use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `DiscoveryContext` is a record for the discovery context contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct DiscoveryContext {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// Parent family recorded for this discovery context.
    pub parent_family: String,
    /// Must be issued by the frozen role manifest, never inferred from a dataset name.
    pub role: String,
    /// SHA-256 digest identifying the associated dataset.
    pub dataset_sha256: String,
    /// Cut recorded for this discovery context.
    pub cut: usize,
    /// SHA-256 digest identifying the associated training.
    pub training_sha256: String,
    /// SHA-256 digest identifying the associated inner holdout.
    pub inner_holdout_sha256: String,
    /// SHA-256 digest identifying the associated preprocessing.
    pub preprocessing_sha256: String,
    /// SHA-256 digest identifying the associated encoder input.
    pub encoder_input_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `TrainingRoster` is a record for the training roster contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct TrainingRoster {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated role manifest.
    pub role_manifest_sha256: String,
    /// Collection of contexts values carried by this training roster.
    pub contexts: Vec<DiscoveryContext>,
}

impl TrainingRoster {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "clover.discovery-roster/v1",
            "unknown roster schema"
        );
        ensure!(
            is_hex64(&self.role_manifest_sha256),
            "role manifest pin missing"
        );
        ensure!(!self.contexts.is_empty(), "empty discovery roster");
        let mut seen = BTreeSet::new();
        for c in &self.contexts {
            ensure!(
                c.role == "train",
                "discovery forbids validation or sealed families"
            );
            ensure!(!c.parent_family.is_empty(), "parent family missing");
            ensure!(
                seen.insert(&c.context_sha256),
                "duplicate discovery context"
            );
            ensure!(
                [
                    &c.context_sha256,
                    &c.dataset_sha256,
                    &c.training_sha256,
                    &c.inner_holdout_sha256,
                    &c.preprocessing_sha256,
                    &c.encoder_input_sha256,
                ]
                .into_iter()
                .all(|h| is_hex64(h)),
                "incomplete discovery input pins"
            );
            ensure!(
                c.training_sha256 != c.inner_holdout_sha256,
                "fit/holdout identity collision"
            );
        }
        Ok(())
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }

    /// Verifies the bound evidence and returns an error when verification fails.
    pub fn verify_roles(&self, role_manifest: &[u8]) -> Result<()> {
        self.validate()?;
        ensure!(
            jope_pipeline::sha256_hex(role_manifest) == self.role_manifest_sha256,
            "frozen family role manifest changed"
        );
        let roles: Value = serde_json::from_slice(role_manifest)?;
        ensure!(
            roles["schema"] == "clover.discovery-family-roles/v1",
            "unknown family role schema"
        );
        for context in &self.contexts {
            ensure!(
                roles["families"][&context.parent_family] == "train",
                "family is not authorized for discovery"
            );
        }
        Ok(())
    }

    fn bound_context(&self, context: &str, cut: &Cut) -> Result<&DiscoveryContext> {
        self.validate()?;
        let c = self
            .contexts
            .iter()
            .find(|c| c.context_sha256 == context)
            .context("context outside frozen training roster")?;
        ensure!(c.cut == cut.index, "discovery cut mismatch");
        ensure!(
            c.training_sha256 == rows_digest(&cut.train_x, &cut.train_y)?,
            "training rows changed"
        );
        ensure!(
            c.inner_holdout_sha256 == rows_digest(&cut.hold_x, &cut.hold_y)?,
            "inner holdout rows changed"
        );
        Ok(c)
    }
}

/// Ordered, transformed rows with labels intact. The preprocessing pin lives in the roster.
pub fn rows_digest(x: &[Vec<f64>], y: &[f64]) -> Result<String> {
    ensure!(!x.is_empty() && x.len() == y.len(), "row/label alignment");
    let width = x[0].len();
    ensure!(
        width > 0
            && x.iter()
                .all(|row| row.len() == width && row.iter().all(|v| v.is_finite()))
            && y.iter().all(|v| v.is_finite()),
        "finite rectangular rows required"
    );
    canonical_sha256(&(x, y))
}
