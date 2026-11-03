use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

mod policy;
pub use policy::*;

const FRAME_LIMIT: usize = 1024 * 1024;

pub(crate) fn write_immutable_record(path: &Path, value: &Value) -> Result<()> {
    create_immutable(path, &serde_json::to_vec(value)?)
}

fn create_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() <= FRAME_LIMIT,
        "durable control record too large"
    );
    let parent = path.parent().context("record parent absent")?;
    fs::create_dir_all(parent)?;
    let temporary = tempfile::Builder::new()
        .prefix(".immutable-pending-")
        .tempfile_in(parent)?;
    let (mut file, temporary_path) = temporary.keep()?;
    file.write_all(bytes)?;
    file.sync_all()?;
    match crate::native_persistence::publish_staged(&temporary_path, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(fs::read(path)? == bytes, "conflicting immutable record");
        }
        Err(error) => return Err(error.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&temporary_path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

pub(crate) fn hash_directory(path: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let mut entries = BTreeMap::new();
    for entry in walkdir::WalkDir::new(path).follow_links(false) {
        let entry = entry?;
        ensure!(!entry.file_type().is_symlink(), "output symlink refused");
        if entry.file_type().is_file() {
            entries.insert(
                entry.path().strip_prefix(path)?.to_owned(),
                crate::manifest::file_hash(entry.path())?,
            );
        }
    }
    ensure!(!entries.is_empty(), "numerical artifact directory is empty");
    Ok(entries)
}
