use super::*;

/// Measurements from an explicit, trusted host qualification run. There is no public
/// endpoint that accepts a caller's claimed parity. Exporting never launches a compiler.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredModelParity {
    pub schema: String,
    pub model_digest: String,
    pub emitter_version: String,
    pub refit: jope_api::Refit,
    pub source_sha256: BTreeMap<String, String>,
    pub toolchains: BTreeMap<String, String>,
    /// Raw feature cells, in the frozen input-contract order; no target values.
    pub rows: Vec<Vec<Option<String>>>,
    pub engine: Vec<f64>,
    pub observed: BTreeMap<String, Vec<f64>>,
}

impl MeasuredModelParity {
    fn validate(&self, object: &ModelObject, refit: &serde_json::Value) -> Result<()> {
        ensure!(
            self.schema == "jope.measured-model-parity/v1",
            "unknown parity protocol"
        );
        ensure!(self.model_digest == object.id(), "parity model mismatch");
        ensure!(
            self.emitter_version == env!("CARGO_PKG_VERSION"),
            "parity emitter mismatch"
        );
        ensure!(
            *refit == serde_json::to_value(self.refit)?,
            "parity refit mismatch"
        );
        ensure!(
            self.rows.len() == 64 && self.engine.len() == 64,
            "parity requires 64 check rows"
        );
        let expected_keys = ["python".to_string(), "rust".to_string()];
        for keys in [
            self.source_sha256.keys().cloned().collect::<Vec<_>>(),
            self.observed.keys().cloned().collect(),
            self.toolchains.keys().cloned().collect(),
        ] {
            ensure!(
                keys == expected_keys,
                "parity requires Python and Rust measurements"
            );
        }
        let pipeline = object.math_pipeline()?;
        let borrowed = self
            .rows
            .iter()
            .map(|r| r.iter().map(Option::as_deref).collect())
            .collect::<Vec<_>>();
        let engine = pipeline.predict_cells(&borrowed)?;
        ensure!(
            engine == self.engine && engine.iter().all(|v| v.is_finite()),
            "parity engine values differ"
        );
        for language in &expected_keys {
            ensure!(
                self.source_sha256[language] == sha256_hex(&object.export_source(language)?),
                "parity source mismatch"
            );
            ensure!(
                !self.toolchains[language].trim().is_empty(),
                "missing parity toolchain identity"
            );
            let observed = &self.observed[language];
            ensure!(observed.len() == 64, "parity output count differs");
            ensure!(
                observed
                    .iter()
                    .zip(&engine)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-12 + 1e-10 * b.abs()),
                "exported predictor parity failed"
            );
        }
        Ok(())
    }
}

/// Bind actual toolchain measurements to a completed model in the trusted local store.
/// Callers retain the toolchain execution receipts. The service rechecks all bindings
/// and the engine values on every new export; absence of evidence stays Unavailable.
pub fn record_model_parity(
    store: &crate::store::Store,
    model_artifact: &jope_api::ids::ArtifactId,
    object: &ModelObject,
    receipt: &MeasuredModelParity,
) -> Result<jope_api::ids::ArtifactId> {
    use crate::store::{Record, RecordKey};
    let key = RecordKey::Artifact(model_artifact.clone());
    let original = store.get(&key)?.context("parity model artifact missing")?;
    ensure!(
        original.data["kind"] == "finalized_model" && original.data["model_digest"] == object.id(),
        "parity needs the completed model"
    );
    let refit = original
        .data
        .get("refit")
        .cloned()
        .unwrap_or(serde_json::json!("fit"));
    receipt.validate(object, &refit)?;
    let bytes = serde_json::to_vec(receipt)?;
    ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "parity receipt exceeds bound"
    );
    let blob = store.publish_blob(std::io::Cursor::new(bytes))?;
    let digest = hex_blob(&blob.sha256);
    let id: jope_api::ids::ArtifactId = format!("art_{}", &digest[..24]).parse()?;
    store.transaction(|tx| {
        let mut current = tx.get(&key)?.ok_or(crate::store::StoreError::Invalid(
            "parity model disappeared",
        ))?;
        if current != original {
            return Err(crate::store::StoreError::Invalid("parity model changed"));
        }
        let parity_key = RecordKey::Artifact(id.clone());
        let parity = Record::metadata(
            serde_json::json!({"kind":"model_parity","model_digest":object.id(),"archive":blob}),
            vec![blob.clone()],
        );
        if let Some(old) = tx.get(&parity_key)? {
            if old != parity {
                return Err(crate::store::StoreError::Integrity(
                    "parity artifact collision",
                ));
            }
        } else {
            tx.put(&parity_key, &parity)?;
        }
        current.data["parity_receipt"] = serde_json::to_value(&id)?;
        tx.put(&key, &current)
    })?;
    Ok(id)
}

fn hex_blob(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}

/// Return only measured evidence for the exact frozen model, mode and source bytes.
pub(crate) fn verified_model_parity(
    store: &crate::store::Store,
    original: &crate::store::Record,
    object: &ModelObject,
    languages: &[jope_api::Language],
) -> Result<Option<(jope_api::ids::ArtifactId, String, Vec<u8>)>> {
    use crate::store::RecordKey;
    let Some(value) = original.data.get("parity_receipt") else {
        return Ok(None);
    };
    let id: jope_api::ids::ArtifactId = serde_json::from_value(value.clone())?;
    let record = store
        .get(&RecordKey::Artifact(id.clone()))?
        .context("parity receipt missing")?;
    ensure!(
        record.data["kind"] == "model_parity" && record.data["model_digest"] == object.id(),
        "parity record mismatch"
    );
    let blob = record
        .blobs
        .first()
        .filter(|b| b.bytes <= 16 * 1024 * 1024)
        .context("parity blob missing or too large")?;
    let bytes = store.read_blob(blob)?;
    let receipt: MeasuredModelParity = jope_runtime::strict_json::decode(&bytes)?;
    let refit = original
        .data
        .get("refit")
        .cloned()
        .unwrap_or(serde_json::json!("fit"));
    receipt.validate(object, &refit)?;
    if languages.is_empty()
        || languages
            .iter()
            .any(|l| !matches!(l, jope_api::Language::Python | jope_api::Language::Rust))
    {
        return Ok(None);
    }
    ensure!(
        id.to_string() == format!("art_{}", &sha256_hex(&bytes)[..24]),
        "parity content identity differs"
    );
    Ok(Some((
        id,
        format!("measured-model:{}", sha256_hex(&bytes)),
        bytes,
    )))
}
