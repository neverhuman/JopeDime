//! Reviewed policy-only projection. Verification grants no redistribution approval.
use super::*;
use std::sync::Arc;

const SELECTED: [&str; 4] = [
    JOPE_POLICY_ARTIFACT,
    JOPE_POLICY_METADATA,
    JOPE_PREPROCESSING_ARTIFACT,
    OPERATOR_REGISTRY,
];
const REVIEWED_MANIFEST: &[u8] = include_bytes!("../../artifacts/model-bundle.v1.json");

/// Immutable identity of the selected, verified assets; distinct from the full bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopePolicyAssetsIdentity {
    pub schema: String,
    pub parent_manifest_sha256: String,
    pub projection_digest: String,
    pub policy_sha256: String,
}

/// Concrete layout for owner review. This record is not permission to distribute assets.
#[derive(Clone, Debug, Serialize)]
pub struct JopePolicyProjectionLayout {
    pub schema: String,
    pub identity: JopePolicyAssetsIdentity,
    pub files: Vec<ModelFileV1>,
    pub notices: BTreeMap<String, String>,
    pub distribution: String,
}

/// Only the pinned policy closure is opened. No public mutable or unchecked byte access.
#[derive(Clone)]
pub struct VerifiedJopePolicyAssets {
    policy: Arc<[u8]>,
    metadata: Arc<[u8]>,
    layout: JopePolicyProjectionLayout,
}

impl VerifiedJopePolicyAssets {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let (_, directory) = loading::open_bundle_root(root.as_ref())?;
        let manifest = loading::read_manifest(&directory)?;
        Self::verify(&manifest, |path, bound| {
            if let Some(bytes) = loading::read_regular_bytes_optional(&directory, path, bound)? {
                return Ok(bytes);
            }
            // Fetched v2 bundles retain these same pinned v1 entries without the artifacts/ prefix.
            loading::read_regular_bytes(
                &directory,
                path.strip_prefix("artifacts/").unwrap_or(path),
                bound,
            )
        })
    }

    /// Package adapters supply authenticated named entries; content is independently pinned.
    /// Extra entries are rejected, rather than broadening the projection's read authority.
    pub fn from_entries(manifest: &[u8], entries: &[(&str, &[u8])]) -> Result<Self> {
        let layout = Self::reviewed_layout()?;
        let expected = layout
            .files
            .iter()
            .map(|f| f.path.as_str())
            .chain(layout.notices.keys().map(String::as_str))
            .collect::<BTreeSet<_>>();
        let supplied = entries
            .iter()
            .map(|(name, _)| *name)
            .collect::<BTreeSet<_>>();
        if supplied != expected || supplied.len() != entries.len() {
            bail!("policy projection entries must equal the reviewed selected closure");
        }
        Self::verify(manifest, |path, bound| {
            let bytes = entries
                .iter()
                .find(|(name, _)| *name == path)
                .with_context(|| format!("missing policy projection entry {path}"))?
                .1;
            if bytes.len() as u64 > bound {
                bail!("policy projection entry exceeds its read bound: {path}");
            }
            Ok(bytes.to_vec())
        })
    }

    pub fn identity(&self) -> &JopePolicyAssetsIdentity {
        &self.layout.identity
    }

    pub fn layout(&self) -> &JopePolicyProjectionLayout {
        &self.layout
    }

    pub fn reviewed_layout() -> Result<JopePolicyProjectionLayout> {
        projection_layout(REVIEWED_MANIFEST)
    }

    pub(crate) fn policy_bytes(&self) -> &[u8] {
        &self.policy
    }

    pub(crate) fn metadata_bytes(&self) -> &[u8] {
        &self.metadata
    }

    fn verify(manifest: &[u8], mut read: impl FnMut(&str, u64) -> Result<Vec<u8>>) -> Result<Self> {
        let layout = projection_layout(manifest)?;
        let manifest: ModelBundleV1 = serde_json::from_slice(manifest)?;
        let mut bytes = BTreeMap::new();
        let mut inventories = BTreeMap::new();
        for file in &layout.files {
            let value = read(&file.path, file.bytes)?;
            if let Some(inventory) = verify_file_bytes(file, &value)? {
                inventories.insert(file.path.clone(), inventory);
            }
            bytes.insert(file.path.clone(), value);
        }
        for (path, digest) in &layout.notices {
            if sha256_bytes(&read(path, MAX_NOTICE_BYTES)?) != *digest {
                bail!("policy projection notice sha256 mismatch: {path}");
            }
        }
        verify_operator_registry(&bytes[OPERATOR_REGISTRY])?;
        verify_jope_metadata(
            &bytes[JOPE_POLICY_METADATA],
            &manifest,
            &inventories[JOPE_POLICY_ARTIFACT],
            &inventories[JOPE_PREPROCESSING_ARTIFACT],
        )?;
        Ok(Self {
            policy: bytes.remove(JOPE_POLICY_ARTIFACT).unwrap().into(),
            metadata: bytes.remove(JOPE_POLICY_METADATA).unwrap().into(),
            layout,
        })
    }
}

fn projection_layout(bytes: &[u8]) -> Result<JopePolicyProjectionLayout> {
    if bytes.len() as u64 > MAX_MANIFEST_BYTES
        || sha256_bytes(bytes) != MODEL_BUNDLE_MANIFEST_SHA256
    {
        bail!("policy projection parent manifest identity changed");
    }
    let manifest: ModelBundleV1 = serde_json::from_slice(bytes)?;
    manifest.validate_metadata()?;
    let files = SELECTED
        .iter()
        .map(|path| {
            let file = manifest
                .files
                .iter()
                .find(|f| f.path == *path)
                .with_context(|| format!("policy projection file is not listed: {path}"))?;
            let license = &file.license;
            if !license.policy_eligibility.eligible
                || license.policy_eligibility.open_only
                || license.license_class != "eligible_non_open"
                || license.redistribution != "permitted"
            {
                bail!("policy projection file is not eligible: {path}");
            }
            Ok(file.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    let mut notices = BTreeMap::new();
    for path in files.iter().flat_map(|f| &f.license.notice_paths) {
        notices.insert(path.clone(), manifest.notices[path].clone());
    }
    let projection_digest = sha256_bytes(&serde_json::to_vec(&(
        "jope.policy-assets-projection/v1",
        MODEL_BUNDLE_MANIFEST_SHA256,
        &files,
        &notices,
    ))?);
    Ok(JopePolicyProjectionLayout {
        schema: "jope.policy-assets-layout/v1".into(),
        identity: JopePolicyAssetsIdentity {
            schema: "jope.policy-assets-projection/v1".into(),
            parent_manifest_sha256: MODEL_BUNDLE_MANIFEST_SHA256.into(),
            projection_digest,
            policy_sha256: files[0].sha256.clone(),
        },
        files,
        notices,
        distribution: "owner-review-required; verification is not redistribution approval".into(),
    })
}

#[cfg(test)]
mod tests;
