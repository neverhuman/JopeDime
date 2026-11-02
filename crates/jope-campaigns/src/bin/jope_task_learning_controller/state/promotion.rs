use super::*;

pub(super) fn corpus_pointer(root: &Path) -> Result<CorpusPointer> {
    let value = read_json(&root.join("training/current-corpus.json"))?;
    let manifest = PathBuf::from(
        value
            .get("manifest")
            .and_then(Value::as_str)
            .context("corpus pointer lacks manifest")?,
    );
    let packed_manifest = PathBuf::from(
        value
            .get("packed_manifest")
            .and_then(Value::as_str)
            .context("corpus pointer lacks packed manifest")?,
    );
    let pointer = CorpusPointer {
        watermark: value
            .get("watermark")
            .and_then(Value::as_u64)
            .context("corpus pointer lacks watermark")?,
        manifest,
        manifest_sha256: value
            .get("manifest_sha256")
            .and_then(Value::as_str)
            .context("corpus pointer lacks manifest digest")?
            .to_owned(),
        packed_manifest,
        packed_manifest_sha256: value
            .get("packed_manifest_sha256")
            .and_then(Value::as_str)
            .context("corpus pointer lacks packed manifest digest")?
            .to_owned(),
    };
    ensure!(
        pointer.manifest.is_file()
            && sha256_file(&pointer.manifest)? == pointer.manifest_sha256
            && pointer.packed_manifest.is_file()
            && sha256_file(&pointer.packed_manifest)? == pointer.packed_manifest_sha256,
        "task trainer corpus pointer content drift"
    );
    Ok(pointer)
}

pub(super) fn load_ledger(root: &Path) -> Result<Option<CausalEvidenceLedgerV3>> {
    let path = root.join("training/causal-evidence-ledger.v3.json");
    if !path.is_file() {
        return Ok(None);
    }
    let ledger: CausalEvidenceLedgerV3 = serde_json::from_value(read_json(&path)?)?;
    ledger.validate()?;
    Ok(Some(ledger))
}
