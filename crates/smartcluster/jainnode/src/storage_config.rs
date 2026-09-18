use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path, PathBuf},
};

use jain_smartcluster_fabric::{sha256, valid_identifier, Digest32};
use nix::unistd::Uid;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zeroize::{Zeroize, Zeroizing};

const STORAGE_CONFIG_SCHEMA: &str = "jain.node.storage-config/v1";
const MAX_STORAGE_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_VOLUMES: usize = 64;
const MAX_PROJECTS: usize = 128;
const MAX_KEKS_PER_PROJECT: usize = 16;
const KEY_BYTES: u64 = 32;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageVolumeConfig {
    root: PathBuf,
    filesystem_uuid: String,
    failure_domain: String,
}

impl StorageVolumeConfig {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn filesystem_uuid(&self) -> &str {
        &self.filesystem_uuid
    }

    pub fn failure_domain(&self) -> &str {
        &self.failure_domain
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedKeyPath {
    version: u32,
    path: PathBuf,
}

impl VersionedKeyPath {
    pub const fn version(&self) -> u32 {
        self.version
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectKeyConfig {
    project_id: String,
    dedup_key_path: PathBuf,
    active_kek_version: u32,
    kek_files: Vec<VersionedKeyPath>,
}

impl ProjectKeyConfig {
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub fn dedup_key_path(&self) -> &Path {
        &self.dedup_key_path
    }

    pub const fn active_kek_version(&self) -> u32 {
        self.active_kek_version
    }

    pub fn kek_files(&self) -> &[VersionedKeyPath] {
        &self.kek_files
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StorageConfigDocument {
    schema: String,
    deployment_id: String,
    node_id: String,
    staging_parent: PathBuf,
    volumes: Vec<StorageVolumeConfig>,
    projects: Vec<ProjectKeyConfig>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageConfig {
    source_path: PathBuf,
    source_sha256: Digest32,
    document: StorageConfigDocument,
}

impl StorageConfig {
    pub fn open(path: &Path) -> Result<Self, StorageConfigError> {
        let bytes =
            read_private_regular_file(path, MAX_STORAGE_CONFIG_BYTES, FilePurpose::StorageConfig)?;
        let document: StorageConfigDocument = serde_json::from_slice(&bytes)?;
        validate_document(&document)?;
        validate_key_files(&document.projects)?;
        Ok(Self {
            source_path: path.to_path_buf(),
            source_sha256: sha256(&bytes),
            document,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub const fn source_sha256(&self) -> Digest32 {
        self.source_sha256
    }

    pub fn deployment_id(&self) -> &str {
        &self.document.deployment_id
    }

    pub fn node_id(&self) -> &str {
        &self.document.node_id
    }

    pub fn staging_parent(&self) -> &Path {
        &self.document.staging_parent
    }

    pub fn volumes(&self) -> &[StorageVolumeConfig] {
        &self.document.volumes
    }

    pub fn projects(&self) -> &[ProjectKeyConfig] {
        &self.document.projects
    }

    pub fn load_project_keys(
        &self,
        project_id: &str,
    ) -> Result<ProjectKeyMaterial, StorageConfigError> {
        let project = self
            .document
            .projects
            .iter()
            .find(|candidate| candidate.project_id == project_id)
            .ok_or(StorageConfigError::UnknownProject)?;
        let dedup_key = Zeroizing::new(read_key(&project.dedup_key_path)?);
        let mut keks = BTreeMap::new();
        for key in &project.kek_files {
            keks.insert(key.version, Zeroizing::new(read_key(&key.path)?));
        }
        Ok(ProjectKeyMaterial {
            project_id: project.project_id.clone(),
            active_kek_version: project.active_kek_version,
            dedup_key,
            keks,
        })
    }
}

pub struct ProjectKeyMaterial {
    project_id: String,
    active_kek_version: u32,
    dedup_key: Zeroizing<[u8; 32]>,
    keks: BTreeMap<u32, Zeroizing<[u8; 32]>>,
}

impl ProjectKeyMaterial {
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub const fn active_kek_version(&self) -> u32 {
        self.active_kek_version
    }

    pub fn dedup_key(&self) -> &[u8; 32] {
        &self.dedup_key
    }

    pub fn keks(&self) -> impl Iterator<Item = (u32, &[u8; 32])> {
        self.keks.iter().map(|(version, key)| (*version, &**key))
    }
}

impl Drop for ProjectKeyMaterial {
    fn drop(&mut self) {
        self.project_id.zeroize();
        self.active_kek_version.zeroize();
    }
}

#[derive(Clone, Copy)]
enum FilePurpose {
    StorageConfig,
    Key,
}

impl FilePurpose {
    const fn label(self) -> &'static str {
        match self {
            Self::StorageConfig => "storage config",
            Self::Key => "project key",
        }
    }
}

fn validate_document(document: &StorageConfigDocument) -> Result<(), StorageConfigError> {
    if document.schema != STORAGE_CONFIG_SCHEMA
        || !valid_identifier(&document.deployment_id)
        || !valid_identifier(&document.node_id)
        || document.volumes.len() < 2
        || document.volumes.len() > MAX_VOLUMES
        || document.projects.is_empty()
        || document.projects.len() > MAX_PROJECTS
        || !valid_absolute_path(&document.staging_parent)
    {
        return Err(StorageConfigError::InvalidConfig);
    }
    let mut roots = BTreeSet::new();
    let mut uuids = BTreeSet::new();
    let mut domains = BTreeSet::new();
    for volume in &document.volumes {
        if !valid_absolute_path(&volume.root)
            || !valid_filesystem_uuid(&volume.filesystem_uuid)
            || !valid_identifier(&volume.failure_domain)
            || !roots.insert(&volume.root)
            || !uuids.insert(&volume.filesystem_uuid)
            || !domains.insert(&volume.failure_domain)
        {
            return Err(StorageConfigError::InvalidConfig);
        }
    }
    let mut project_ids = BTreeSet::new();
    for project in &document.projects {
        if !valid_identifier(&project.project_id)
            || !project_ids.insert(&project.project_id)
            || !valid_absolute_path(&project.dedup_key_path)
            || project.active_kek_version == 0
            || project.kek_files.is_empty()
            || project.kek_files.len() > MAX_KEKS_PER_PROJECT
        {
            return Err(StorageConfigError::InvalidConfig);
        }
        let mut versions = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for key in &project.kek_files {
            if key.version == 0
                || !valid_absolute_path(&key.path)
                || !versions.insert(key.version)
                || !paths.insert(&key.path)
                || key.path == project.dedup_key_path
            {
                return Err(StorageConfigError::InvalidConfig);
            }
        }
        if !versions.contains(&project.active_kek_version) {
            return Err(StorageConfigError::InvalidConfig);
        }
    }
    Ok(())
}

fn validate_key_files(projects: &[ProjectKeyConfig]) -> Result<(), StorageConfigError> {
    let mut paths = BTreeSet::new();
    for project in projects {
        if !paths.insert(&project.dedup_key_path) {
            return Err(StorageConfigError::InvalidConfig);
        }
        validate_private_regular_file(&project.dedup_key_path, KEY_BYTES, FilePurpose::Key)?;
        for key in &project.kek_files {
            if !paths.insert(&key.path) {
                return Err(StorageConfigError::InvalidConfig);
            }
            validate_private_regular_file(&key.path, KEY_BYTES, FilePurpose::Key)?;
        }
    }
    Ok(())
}

fn read_key(path: &Path) -> Result<[u8; 32], StorageConfigError> {
    let bytes = Zeroizing::new(read_private_regular_file(
        path,
        KEY_BYTES,
        FilePurpose::Key,
    )?);
    if bytes.len() != KEY_BYTES as usize {
        return Err(StorageConfigError::InvalidKeyFile);
    }
    let mut key = [0_u8; 32];
    key.copy_from_slice(&bytes);
    Ok(key)
}

fn validate_private_regular_file(
    path: &Path,
    exact_bytes: u64,
    purpose: FilePurpose,
) -> Result<(), StorageConfigError> {
    let mut file = open_private_regular_file(path, purpose)?;
    let metadata = file.metadata()?;
    if metadata.len() != exact_bytes {
        return Err(StorageConfigError::InvalidKeyFile);
    }
    let mut probe = [0_u8; 1];
    file.read_exact(&mut probe)?;
    probe.zeroize();
    Ok(())
}

fn read_private_regular_file(
    path: &Path,
    maximum_bytes: u64,
    purpose: FilePurpose,
) -> Result<Vec<u8>, StorageConfigError> {
    let mut file = open_private_regular_file(path, purpose)?;
    let before = file.metadata()?;
    if before.len() == 0 || before.len() > maximum_bytes {
        return Err(StorageConfigError::InvalidFile(purpose.label()));
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    Read::by_ref(&mut file)
        .take(maximum_bytes + 1)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    if bytes.len() as u64 > maximum_bytes
        || before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
    {
        bytes.zeroize();
        return Err(StorageConfigError::InvalidFile(purpose.label()));
    }
    Ok(bytes)
}

fn open_private_regular_file(
    path: &Path,
    purpose: FilePurpose,
) -> Result<File, StorageConfigError> {
    if !valid_absolute_path(path) {
        return Err(StorageConfigError::InvalidFile(purpose.label()));
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC);
    let file = options.open(path)?;
    let opened = file.metadata()?;
    let named = fs::symlink_metadata(path)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || fs::canonicalize(path)? != path
    {
        return Err(StorageConfigError::InvalidFile(purpose.label()));
    }
    validate_physical_components(path)?;
    Ok(file)
}

pub(crate) fn validate_private_directory(
    path: &Path,
    label: &'static str,
) -> Result<(), StorageConfigError> {
    if !valid_absolute_path(path) {
        return Err(StorageConfigError::InvalidDirectory(label));
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != Uid::effective().as_raw()
        || metadata.mode() & 0o777 != 0o700
        || fs::canonicalize(path)? != path
    {
        return Err(StorageConfigError::InvalidDirectory(label));
    }
    validate_physical_components(path)
}

fn validate_physical_components(path: &Path) -> Result<(), StorageConfigError> {
    let mut current = PathBuf::from("/");
    for component in path.components().skip(1) {
        current.push(component.as_os_str());
        if fs::symlink_metadata(&current)?.file_type().is_symlink() {
            return Err(StorageConfigError::AliasedPath);
        }
    }
    Ok(())
}

fn valid_absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && path != Path::new("/")
        && path
            .components()
            .all(|component| !matches!(component, Component::CurDir | Component::ParentDir))
}

fn valid_filesystem_uuid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if [8, 13, 18, 23].contains(&index) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')
            }
        })
}

#[derive(Debug, Error)]
pub enum StorageConfigError {
    #[error("invalid JainNode storage configuration")]
    InvalidConfig,
    #[error("invalid {0} file custody")]
    InvalidFile(&'static str),
    #[error("invalid {0} directory custody")]
    InvalidDirectory(&'static str),
    #[error("storage path contains an alias component")]
    AliasedPath,
    #[error("project key file must contain exactly 32 bytes")]
    InvalidKeyFile,
    #[error("project is absent from the closed key registry")]
    UnknownProject,
    #[error("storage volume filesystem UUID does not identify its mounted device")]
    FilesystemUuidMismatch,
    #[error("two storage volume roots resolve to the same mounted device")]
    SharedDevice,
    #[error("two storage failure domains share a physical block-device ancestor")]
    SharedDeviceAncestry,
    #[error("storage block-device ancestry is unavailable")]
    DeviceAncestryUnavailable,
    #[error("storage configuration JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("storage configuration I/O failed: {0}")]
    Io(#[from] std::io::Error),
}
