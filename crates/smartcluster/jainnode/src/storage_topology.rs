use std::{
    collections::BTreeSet,
    fs,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Path, PathBuf},
};

use nix::sys::stat;

use crate::{storage_config::validate_private_directory, StorageConfig, StorageConfigError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostVolumeIdentity {
    pub root: PathBuf,
    pub filesystem_uuid: String,
    pub failure_domain: String,
    pub device_path: PathBuf,
    pub leaf_devices: BTreeSet<PathBuf>,
}

impl StorageConfig {
    pub fn validate_host_topology(&self) -> Result<Vec<HostVolumeIdentity>, StorageConfigError> {
        validate_private_directory(self.staging_parent(), "staging parent")?;
        let mut devices = BTreeSet::new();
        let mut claimed_leaves = BTreeSet::new();
        let mut identities = Vec::with_capacity(self.volumes().len());
        for volume in self.volumes() {
            validate_private_directory(volume.root(), "storage volume")?;
            let root_metadata = fs::metadata(volume.root())?;
            if !devices.insert(root_metadata.dev()) {
                return Err(StorageConfigError::SharedDevice);
            }
            let uuid_link = Path::new("/dev/disk/by-uuid").join(volume.filesystem_uuid());
            let block_path = fs::canonicalize(&uuid_link)
                .map_err(|_| StorageConfigError::FilesystemUuidMismatch)?;
            let block_metadata = fs::metadata(&block_path)
                .map_err(|_| StorageConfigError::FilesystemUuidMismatch)?;
            if !block_metadata.file_type().is_block_device()
                || block_metadata.rdev() != root_metadata.dev()
            {
                return Err(StorageConfigError::FilesystemUuidMismatch);
            }
            let device_path = sys_device_path(root_metadata.dev())?;
            let leaves = leaf_block_devices(&device_path)?;
            if leaves.is_empty()
                || leaves
                    .iter()
                    .any(|leaf| !claimed_leaves.insert(leaf.clone()))
            {
                return Err(StorageConfigError::SharedDeviceAncestry);
            }
            identities.push(HostVolumeIdentity {
                root: volume.root().to_path_buf(),
                filesystem_uuid: volume.filesystem_uuid().to_owned(),
                failure_domain: volume.failure_domain().to_owned(),
                device_path: block_path,
                leaf_devices: leaves,
            });
        }
        Ok(identities)
    }
}

fn sys_device_path(device: u64) -> Result<PathBuf, StorageConfigError> {
    let link = Path::new("/sys/dev/block").join(format!(
        "{}:{}",
        stat::major(device),
        stat::minor(device)
    ));
    fs::canonicalize(link).map_err(|_| StorageConfigError::DeviceAncestryUnavailable)
}

fn leaf_block_devices(path: &Path) -> Result<BTreeSet<PathBuf>, StorageConfigError> {
    let slaves = path.join("slaves");
    let mut leaves = BTreeSet::new();
    if slaves.is_dir() {
        for entry in fs::read_dir(slaves)? {
            let child = fs::canonicalize(entry?.path())?;
            leaves.extend(leaf_block_devices(&child)?);
        }
    }
    if leaves.is_empty() {
        let leaf = if path.join("partition").is_file() {
            path.parent()
                .ok_or(StorageConfigError::DeviceAncestryUnavailable)?
                .to_path_buf()
        } else {
            path.to_path_buf()
        };
        leaves.insert(fs::canonicalize(leaf)?);
    }
    Ok(leaves)
}
