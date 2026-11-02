use super::*;

/// Read one regular file, refusing symlinks at every relative component.
pub fn read_artifact(root: &Path, relative: &str) -> Result<Vec<u8>> {
    safe_relative(relative)?;
    ensure!(
        !std::fs::symlink_metadata(root)?.file_type().is_symlink(),
        "symlink artifact root"
    );
    let mut path = root.to_path_buf();
    for c in Path::new(relative).components() {
        path.push(c.as_os_str());
        ensure!(
            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "symlink artifact path"
        );
    }
    ensure!(
        std::fs::metadata(&path)?.is_file(),
        "artifact is not a regular file"
    );
    std::fs::read(&path).with_context(|| format!("read artifact {}", path.display()))
}

/// An external trust anchor. Never derive this from loose files during verification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedArtifactV3 {
    pub manifest_sha256: String,
    pub program_sha256: String,
    pub requested: Vec<Target>,
}
impl ExpectedArtifactV3 {
    pub fn from_emission(bytes: &[u8], manifest: &ArtifactManifestV3) -> Self {
        Self {
            manifest_sha256: sha256(bytes),
            program_sha256: manifest.genome_sha256.clone(),
            requested: manifest.requested.clone(),
        }
    }
}
pub fn read_manifest(bytes: &[u8], expected: &ExpectedArtifactV3) -> Result<ArtifactManifestV3> {
    check_digest(&expected.manifest_sha256)?;
    check_digest(&expected.program_sha256)?;
    validate_requests(&expected.requested)?;
    ensure!(
        sha256(bytes) == expected.manifest_sha256,
        "pinned manifest digest mismatch"
    );
    let envelope: ArtifactEnvelope = serde_json::from_slice(bytes)?;
    ensure!(
        envelope.envelope_schema == ENVELOPE && envelope.schema_id == SCHEMA,
        "legacy or unknown manifest cannot claim v3 binding"
    );
    ensure!(
        canonical(&envelope)? == bytes,
        "noncanonical artifact envelope"
    );
    let payload = canonical(&envelope.payload)?;
    ensure!(
        envelope.own_id
            == OwnId {
                formula: FORMULA.into(),
                schema_id: SCHEMA.into(),
                payload_byte_len: payload.len() as u64,
                sha256: domain_hash(&payload)
            },
        "artifact payload identity mismatch"
    );
    let manifest = envelope.payload;
    manifest.validate()?;
    ensure!(
        manifest.genome_sha256 == expected.program_sha256,
        "selected program mismatch"
    );
    let mut requested = expected.requested.clone();
    requested.sort();
    let mut actual = manifest.requested.clone();
    actual.sort();
    ensure!(actual == requested, "expected request set mismatch");
    Ok(manifest)
}
pub fn verify_directory(
    root: &Path,
    manifest_path: &str,
    expected: &ExpectedArtifactV3,
) -> Result<ArtifactManifestV3> {
    let manifest = read_manifest(&read_artifact(root, manifest_path)?, expected)?;
    for target in &manifest.targets {
        if let TargetState::Supported { files, .. } = &target.state {
            for file in files {
                let bytes = read_artifact(root, &file.path)?;
                ensure!(
                    bytes.len() as u64 == file.bytes && sha256(&bytes) == file.sha256,
                    "{}: artifact bytes disagree with selected manifest",
                    file.path
                );
            }
        }
    }
    Ok(manifest)
}

/// Sources are assembled before publication; the manifest binds intended bytes,
/// and the writer must verify actual readback against it before returning success.
pub struct TargetEmission {
    pub manifest: ArtifactManifestV3,
    pub sources: BTreeMap<String, String>,
}
