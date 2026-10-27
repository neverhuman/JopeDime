//! Read-only adapter to controller-owned state. Requests cannot select these authority files.
use super::sha256;
use crate::bundle::loading::{open_bundle_root, read_regular_bytes, read_regular_bytes_optional};
use anyhow::{bail, ensure, Context, Result};
use jope_records::replay_admission::{
    ArtifactStore, ComponentAuthoritySnapshot, RevocationSnapshot, ADOPTION_PATH,
    COMPONENT_AUTHORITY_PATH, COMPONENT_AUTHORITY_SCHEMA,
};
use jope_records::wire_contract::CanonicalPayload;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
};

const MAX_CONTROL: u64 = 2 * 1024 * 1024;
const MAX_BLOB: u64 = 512 * 1024 * 1024;

pub(super) fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub(super) fn now_ms() -> Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

pub(super) fn read_input(path: &Path, bound: u64) -> Result<Vec<u8>> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (_, directory) = open_bundle_root(parent)?;
    let name = path
        .file_name()
        .and_then(|p| p.to_str())
        .context("invalid component input path")?;
    read_regular_bytes(&directory, name, bound)
}

pub(super) struct ControllerStore {
    root: PathBuf,
    directory: File,
    config: Vec<u8>,
    adoption: Vec<u8>,
    snapshot: ComponentAuthoritySnapshot,
    digest: String,
    prohibited_models: BTreeSet<String>,
}

impl ControllerStore {
    pub(super) fn load(root: &Path) -> Result<Self> {
        let (root, directory) = open_bundle_root(root)?;
        let config = read_regular_bytes(&directory, COMPONENT_AUTHORITY_PATH, MAX_CONTROL)
            .context("controller component authority is not installed; request metadata cannot grant admission")?;
        let canonical = CanonicalPayload::from_canonical_bytes(&config)?;
        ensure!(
            canonical.schema_id() == COMPONENT_AUTHORITY_SCHEMA,
            "unknown component authority schema"
        );
        let snapshot: ComponentAuthoritySnapshot = serde_json::from_slice(&config)?;
        ensure!(
            is_digest(&snapshot.adoption_sha256)
                && snapshot.revocation_files.len() <= 4096
                && snapshot.revocation_files.values().all(|d| is_digest(d))
                && snapshot.revoked_digests.iter().all(|d| is_digest(d)),
            "invalid component authority snapshot"
        );
        let adoption = read_regular_bytes(&directory, ADOPTION_PATH, MAX_CONTROL)?;
        let digest = sha256(&config);
        // Data admission cannot override the frozen model licence/eligibility boundary.
        // This reads embedded metadata only, never unrelated model payloads.
        let manifest: crate::bundle::ModelBundleV1 =
            serde_json::from_slice(include_bytes!("../../../artifacts/model-bundle.v1.json"))?;
        let prohibited_models = manifest
            .files
            .into_iter()
            .filter(|file| {
                file.license.redistribution != "permitted"
                    || !file.license.policy_eligibility.eligible
            })
            .map(|file| file.sha256)
            .collect();
        let out = Self {
            root,
            directory,
            config,
            adoption,
            snapshot,
            digest,
            prohibited_models,
        };
        out.require_active()?;
        Ok(out)
    }
    pub(super) fn root(&self) -> &Path {
        &self.root
    }
    pub(super) fn digest(&self) -> &str {
        &self.digest
    }
    pub(super) fn subject(&self, digest: &str) -> Result<Vec<u8>> {
        self.read_blob(digest)?
            .context("admitted component subject is missing")
    }
    pub(super) fn require_active(&self) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let opened = self.directory.metadata()?;
            let named = std::fs::symlink_metadata(&self.root)?;
            ensure!(
                named.is_dir()
                    && !named.file_type().is_symlink()
                    && opened.dev() == named.dev()
                    && opened.ino() == named.ino(),
                "controller state root changed"
            );
        }
        ensure!(
            read_regular_bytes(&self.directory, COMPONENT_AUTHORITY_PATH, MAX_CONTROL)?
                == self.config,
            "component controller authority changed; reconstruct the session"
        );
        let adoption = read_regular_bytes(&self.directory, ADOPTION_PATH, MAX_CONTROL)?;
        ensure!(
            adoption == self.adoption && sha256(&adoption) == self.snapshot.adoption_sha256,
            "controller replay adoption changed"
        );
        ensure!(
            revocation_inventory(&self.root, &self.directory)? == self.snapshot.revocation_files,
            "component revocation state changed; a fresh controller snapshot is required"
        );
        Ok(())
    }
}

pub(super) struct Revoked(BTreeSet<String>);
impl RevocationSnapshot for Revoked {
    fn refuse(&self, what: &str, digests: &[&str]) -> Result<()> {
        ensure!(
            digests.iter().all(|d| !self.0.contains(*d)),
            "RevokedEvidence: {what}"
        );
        Ok(())
    }
}
impl ArtifactStore for ControllerStore {
    type StoreId = Path;
    type Revocations = Revoked;
    fn store_id(&self) -> &Path {
        &self.root
    }
    fn adopted_policy_bytes(&self) -> Result<Vec<u8>> {
        self.require_active()?;
        Ok(self.adoption.clone())
    }
    fn read_blob(&self, digest: &str) -> Result<Option<Vec<u8>>> {
        ensure!(is_digest(digest), "invalid component proof digest");
        ensure!(
            !self.prohibited_models.contains(digest),
            "prohibited model weights cannot enter component inputs"
        );
        let bytes = read_regular_bytes_optional(
            &self.directory,
            &format!("blobs/sha256/{digest}"),
            MAX_BLOB,
        )?;
        if let Some(bytes) = &bytes {
            ensure!(
                sha256(bytes) == digest,
                "altered component proof or subject"
            );
        }
        Ok(bytes)
    }
    fn revocations(&self) -> Result<Revoked> {
        self.require_active()?;
        Ok(Revoked(self.snapshot.revoked_digests.clone()))
    }
}

// Match the existing ledger's HEAD and depth-three jsonl batch roster, without recovery/writes.
fn revocation_inventory(root: &Path, directory: &File) -> Result<BTreeMap<String, String>> {
    let relative = "revocations/ledger";
    let path = root.join(relative);
    let mut result = BTreeMap::new();
    match std::fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(result),
        Err(e) => return Err(e.into()),
        Ok(meta) => ensure!(
            meta.is_dir() && !meta.file_type().is_symlink(),
            "invalid revocation ledger root"
        ),
    }
    let mut pending = vec![(path, 0usize)];
    let mut visited = 0;
    let mut total = 0usize;
    while let Some((path, depth)) = pending.pop() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            visited += 1;
            ensure!(
                visited <= 8192,
                "revocation ledger roster exceeds its bound"
            );
            let kind = entry.file_type()?;
            ensure!(!kind.is_symlink(), "revocation ledger contains a symlink");
            if kind.is_dir() && depth < 2 {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file()
                && ((depth == 0 && entry.file_name() == "HEAD.json")
                    || (depth == 2 && entry.path().extension().is_some_and(|p| p == "jsonl")))
            {
                let relative = entry
                    .path()
                    .strip_prefix(root)?
                    .to_str()
                    .context("non-UTF8 ledger path")?
                    .to_owned();
                let bytes = read_regular_bytes(directory, &relative, 8 * 1024 * 1024)?;
                total = total
                    .checked_add(bytes.len())
                    .context("revocation inventory overflow")?;
                ensure!(
                    total <= 64 * 1024 * 1024 && result.len() < 4096,
                    "revocation inventory exceeds its bound"
                );
                result.insert(relative, sha256(&bytes));
            } else if !kind.is_file() {
                bail!("revocation ledger contains an unsupported entry");
            }
        }
    }
    Ok(result)
}
