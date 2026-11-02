#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuContextArtifact {
    pub schema: String,
    pub backend: String,
    pub qualification: String,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub input_sha256: String,
    pub training: TrainingIdentity,
    pub selected_training_rows_sha256: String,
    pub original_labels_sha256: String,
    pub preprocessing_sha256: String,
    pub prepared_rows_sha256: String,
    pub preprocessing: TrainingPreprocessing,
    pub encoder: EncoderIdentity,
    /// Serialized encoder products (claims); their layout is checked when parsed.
    pub components: ComponentVectors,
    /// Serialized final context (a claim); see [`CpuContextArtifact::replay_assembly`].
    pub vector: Vec<f32>,
    pub final_test_targets_used: bool,
}
impl CpuContextArtifact {
    /// Receiver check of the serialized products against the accepted frozen transforms: the
    /// final context must be the frozen assembly of the serialized components. Returns the
    /// typed context under this artifact's input scope (`prepared_rows_sha256`).
    pub fn replay_assembly(&self, transforms: &FrozenTransforms) -> Result<FinalContext> {
        self.validate()?;
        transforms.replay(&self.components, &self.vector, &self.prepared_rows_sha256)
    }

    /// Envelope, identities, pinned encoder release and product layout. Whether the context is
    /// the assembly of the components needs the transforms: [`Self::replay_assembly`].
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && self.backend == BACKEND
                && self.qualification == "unqualified"
                && !self.final_test_targets_used,
            "invalid CPU context envelope"
        );
        validate_hex(&self.source_sha256, 40)?;
        for d in [
            &self.producer_sha256,
            &self.input_sha256,
            &self.selected_training_rows_sha256,
            &self.original_labels_sha256,
            &self.preprocessing_sha256,
            &self.prepared_rows_sha256,
        ] {
            validate_hex(d, 64)?;
        }
        self.training.validate()?;
        ensure!(
            self.selected_training_rows_sha256 == self.training.fitting_rows_sha256,
            "context does not match exact fitting-row assignment"
        );
        ensure!(
            self.preprocessing_sha256 == sha(&serde_json::to_vec(&self.preprocessing)?),
            "preprocessing hash changed"
        );
        // All eight encoder fields are held to the release decision, not four.
        self.encoder.require_accepted()?;
        self.components.validate()?;
        vector(&self.vector, FinalContext::WIDTH).context("final context layout")?;
        ensure!(
            self.vector.iter().all(|v| v.abs() <= 32.0),
            "context exceeds frozen clip"
        );
        Ok(())
    }
}

/// Foundation context labels: the native component builder casts to f32, standardizes in f64 with
/// the population deviation plus 1e-8, then casts back.
fn standardize_context_labels(y: &[f32]) -> Vec<f32> {
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
    let sd =
        (y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt() + 1e-8;
    y.iter().map(|&v| ((v as f64 - mean) / sd) as f32).collect()
}

/// Target-encoder labels: min-max over the training rows, with a unit span for constant targets.
fn minmax_targets(targets: &[f64]) -> Vec<f32> {
    let low = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let high = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if high > low { high - low } else { 1.0 };
    targets.iter().map(|v| ((v - low) / span) as f32).collect()
}

pub(super) fn sha(bytes: &[u8]) -> String {
    crate::sha256_hex(bytes)
}
pub(super) fn validate_hex(s: &str, length: usize) -> Result<()> {
    ensure!(
        s.len() == length
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid lowercase hex identity"
    );
    Ok(())
}
fn verified_hyperion_settings(manifest: &AssetFile) -> Result<HyperionSweepSettings> {
    ensure!(
        manifest.sha256 == HYPERION_MANIFEST,
        "Hyperion manifest pin changed"
    );
    // Parse exactly the authenticated bytes; the manifest path is not reopened. The legacy
    // 64-bit identifier parse_settings also returns is not a SHA-256 digest and is ignored.
    let bytes = manifest.read(1 << 20)?;
    let text = std::str::from_utf8(&bytes).context("Hyperion manifest is not UTF-8")?;
    let (settings, _) = parse_settings(text, HYPERION_SEED, "accepted Hyperion manifest")?;
    ensure!(settings.emb_dim == 128, "Hyperion embedding width changed");
    Ok(settings)
}

pub(super) fn vector(v: &[f32], length: usize) -> Result<()> {
    ensure!(
        v.len() == length && v.iter().all(|v| v.is_finite()),
        "invalid finite vector of length {length}"
    );
    Ok(())
}
pub(super) fn validate_rows(rows: &[Vec<f64>]) -> Result<usize> {
    let width = rows.first().context("empty prepared rows")?.len();
    ensure!(
        rows.len() >= 8
            && width >= 2
            && rows
                .iter()
                .all(|r| r.len() == width && r.iter().all(|v| v.is_finite())),
        "prepared rows must contain at least8 finite training rows, features and target"
    );
    Ok(width)
}
