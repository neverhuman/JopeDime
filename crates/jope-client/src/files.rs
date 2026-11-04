//! Client side file transfer helpers for the v1 dataset and artifact routes.
use crate::transport::{Error as TransportError, Response, Result as TransportResult, Transport};
use jope_api::ids::ArtifactId;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use zip::ZipArchive;

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum FileError {
    Transport(TransportError),
    Io(io::Error),
    InvalidDigest,
    DigestMismatch,
    UnsafeArchive(&'static str),
    Archive(zip::result::ZipError),
    OutputExists,
}

impl std::fmt::Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
            Self::InvalidDigest => f.write_str("artifact digest must be sha256:<64 lowercase hex>"),
            Self::DigestMismatch => f.write_str("artifact SHA-256 digest mismatch"),
            Self::UnsafeArchive(reason) => write!(f, "unsafe artifact archive: {reason}"),
            Self::Archive(error) => error.fmt(f),
            Self::OutputExists => f.write_str("OUTPUT_EXISTS"),
        }
    }
}
impl std::error::Error for FileError {}
impl From<io::Error> for FileError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<TransportError> for FileError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}
impl From<zip::result::ZipError> for FileError {
    fn from(error: zip::result::ZipError) -> Self {
        Self::Archive(error)
    }
}
pub type Result<T> = std::result::Result<T, FileError>;

/// Stream a CSV file to the existing remote dataset registration endpoint.
/// `idempotency_key` is required by the registry's mutating-operation contract.
pub fn upload_csv<T: Transport>(
    transport: &T,
    name: &str,
    reader: impl Read + Send + 'static,
    idempotency_key: &str,
) -> TransportResult<Response> {
    transport.dataset_register_stream(name, Box::new(reader), idempotency_key)
}

/// Atomically publish a CSV response, or verify a byte-identical retry.
/// An existing path is never replaced; only a regular, non-symlink file qualifies.
pub fn write_csv_atomic(out: &Path, mut reader: impl Read) -> Result<u64> {
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = out
        .file_name()
        .ok_or(FileError::UnsafeArchive("output must name a file"))?;
    if !parent.is_dir() {
        return Err(FileError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            "output parent does not exist",
        )));
    }
    match fs::symlink_metadata(out) {
        Ok(metadata) if !metadata.is_file() => return Err(FileError::OutputExists),
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let (temporary, mut file) = loop {
        let temporary = unique_sibling(parent, name, "csv");
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    let _cleanup = RemoveOnDrop(Some(temporary.clone()));
    let bytes = io::copy(&mut reader, &mut file)?;
    file.sync_all()?;
    drop(file);
    // Unlike check-then-rename, hard_link cannot replace a concurrent creator.
    // The sibling is on the same filesystem and is removed on every exit path.
    match fs::hard_link(&temporary, out) {
        Ok(()) => Ok(bytes),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            if same_csv_contents(&temporary, out)? {
                Ok(bytes)
            } else {
                Err(FileError::OutputExists)
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn same_csv_contents(expected: &Path, actual: &Path) -> Result<bool> {
    let metadata = fs::symlink_metadata(actual)?;
    if !metadata.is_file() || metadata.len() != fs::metadata(expected)?.len() {
        return Ok(false);
    }
    let mut expected = BufReader::new(File::open(expected)?);
    let mut actual = BufReader::new(File::open(actual)?);
    loop {
        let left = expected.fill_buf()?;
        let right = actual.fill_buf()?;
        if left.is_empty() || right.is_empty() {
            return Ok(left.is_empty() && right.is_empty());
        }
        let count = left.len().min(right.len());
        if left[..count] != right[..count] {
            return Ok(false);
        }
        expected.consume(count);
        actual.consume(count);
    }
}

/// Materialize a model export without confusing its model identity with archive bytes.
/// The authenticated immutable artifact endpoint supplies the archive SHA-256 header;
/// the existing downloader still verifies every byte before opening the ZIP.
pub fn download_export_artifact<T: Transport>(
    transport: &T,
    artifact: &ArtifactId,
    out: &Path,
    force: bool,
) -> Result<()> {
    validate_export_destination(out)?;
    let response = transport.artifact_download(artifact)?;
    let digest = response
        .headers
        .get("x-jope-artifact-sha256")
        .and_then(|v| v.to_str().ok())
        .ok_or(FileError::InvalidDigest)?;
    let digest = format!("sha256:{digest}");
    parse_digest(&digest)?;
    drop(response);
    let empty = out.is_dir() && fs::read_dir(out)?.next().is_none();
    if force || !out.exists() || empty {
        return download_artifact(transport, artifact, &digest, out, force);
    }
    // Verify a retry against a fresh, authenticated archive. Never overwrite the
    // existing destination, and reject extras, changed files and symlinks.
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = out.file_name().ok_or(FileError::OutputExists)?;
    let verify = unique_sibling(parent, name, "verify");
    fs::create_dir(&verify)?;
    let _cleanup = RemoveOnDrop(Some(verify.clone()));
    let expected = verify.join("expected");
    download_artifact(transport, artifact, &digest, &expected, false)?;
    if same_export_tree(&expected, out)? {
        Ok(())
    } else {
        Err(FileError::OutputExists)
    }
}

/// Validate local destination syntax and type before a mutating engine request.
/// Existing directories may be verified retries; files and symlinks never qualify.
pub fn validate_export_destination(out: &Path) -> Result<()> {
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if out.as_os_str().is_empty() || out.file_name().is_none() || !parent.is_dir() {
        return Err(FileError::UnsafeArchive(
            "output must name a directory with an existing parent",
        ));
    }
    match fs::symlink_metadata(out) {
        Ok(m) if !m.is_dir() || m.file_type().is_symlink() => Err(FileError::OutputExists),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}
fn same_export_tree(expected: &Path, actual: &Path) -> Result<bool> {
    let a = fs::symlink_metadata(expected)?;
    let b = fs::symlink_metadata(actual)?;
    if a.file_type().is_symlink() || b.file_type().is_symlink() {
        return Ok(false);
    }
    if a.is_file() && b.is_file() {
        if a.len() != b.len() {
            return Ok(false);
        }
        let mut left = Vec::new();
        let mut right = Vec::new();
        File::open(expected)?
            .take(a.len().saturating_add(1))
            .read_to_end(&mut left)?;
        File::open(actual)?
            .take(a.len().saturating_add(1))
            .read_to_end(&mut right)?;
        return Ok(left == right);
    }
    if !a.is_dir() || !b.is_dir() {
        return Ok(false);
    }
    let names = |p: &Path| -> io::Result<std::collections::BTreeSet<std::ffi::OsString>> {
        fs::read_dir(p)?.map(|v| v.map(|e| e.file_name())).collect()
    };
    let expected_names = names(expected)?;
    if expected_names != names(actual)? {
        return Ok(false);
    }
    for name in expected_names {
        if !same_export_tree(&expected.join(&name), &actual.join(&name))? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Download an artifact, verify the service-provided digest, and publish its ZIP contents.
/// The archive is never opened until the digest over the raw HTTP bytes has matched.
pub fn download_artifact<T: Transport>(
    transport: &T,
    artifact: &ArtifactId,
    service_digest: &str,
    out: &Path,
    force: bool,
) -> Result<()> {
    let expected = parse_digest(service_digest)?;
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = out
        .file_name()
        .ok_or(FileError::UnsafeArchive("output must name a directory"))?;
    if !parent.is_dir() {
        return Err(FileError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            "output parent does not exist",
        )));
    }
    let mut response = transport.artifact_download(artifact)?;
    let archive_path = unique_sibling(parent, name, "download");
    // Arm only after exclusive creation; the later file binding closes first.
    let mut archive_guard = RemoveOnDrop(None);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&archive_path)?;
    archive_guard.0 = Some(archive_path.clone());
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = response.body.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        file.write_all(&buffer[..count])?;
    }
    file.sync_all()?;
    if digest_hex(hash.finalize().as_slice()) != expected {
        return Err(FileError::DigestMismatch);
    }

    let stage_path = unique_sibling(parent, name, "extract");
    fs::create_dir(&stage_path)?;
    let mut stage_guard = RemoveOnDrop(Some(stage_path.clone()));
    let archive_file = File::open(&archive_path)?;
    extract_zip(archive_file, &stage_path)?;
    publish(&stage_path, out, force)?;
    stage_guard.take();
    Ok(())
}

fn parse_digest(value: &str) -> Result<String> {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return Err(FileError::InvalidDigest);
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(FileError::InvalidDigest);
    }
    Ok(digest.to_owned())
}
fn digest_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 0x0f) as usize] as char);
    }
    text
}

fn extract_zip(reader: impl Read + io::Seek, destination: &Path) -> Result<()> {
    let mut archive = ZipArchive::new(reader)?;
    let mut seen = std::collections::HashSet::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let raw_name = entry.name()?.into_owned();
        let windows_drive = raw_name.as_bytes().get(1) == Some(&b':')
            && raw_name.as_bytes()[0].is_ascii_alphabetic();
        if raw_name.contains('\\') || raw_name.starts_with('/') || windows_drive {
            return Err(FileError::UnsafeArchive(
                "absolute or platform-ambiguous entry path",
            ));
        }
        let path = Path::new(&raw_name);
        if path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err(FileError::UnsafeArchive(
                "absolute or parent path component",
            ));
        }
        let relative = entry
            .enclosed_name()
            .ok_or(FileError::UnsafeArchive("entry escapes output directory"))?;
        if entry.is_symlink() {
            return Err(FileError::UnsafeArchive("symlink entry"));
        }
        if !seen.insert(relative.clone()) {
            return Err(FileError::UnsafeArchive("duplicate entry path"));
        }
        let target = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&target)?;
        } else {
            let parent = target
                .parent()
                .ok_or(FileError::UnsafeArchive("entry has no parent"))?;
            fs::create_dir_all(parent)?;
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)?;
            io::copy(&mut entry, &mut output)?;
            output.sync_all()?;
        }
    }
    Ok(())
}

fn publish(stage: &Path, out: &Path, force: bool) -> Result<()> {
    let metadata = match fs::symlink_metadata(out) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    if let Some(metadata) = &metadata {
        if metadata.file_type().is_symlink() {
            return Err(FileError::UnsafeArchive("output path is a symlink"));
        }
        let nonempty = !metadata.is_dir() || fs::read_dir(out)?.next().is_some();
        if nonempty && !force {
            return Err(FileError::OutputExists);
        }
    }
    let Some(metadata) = metadata else {
        fs::rename(stage, out)?;
        return Ok(());
    };
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = out
        .file_name()
        .ok_or(FileError::UnsafeArchive("output must name a directory"))?;
    let is_nonempty = !metadata.is_dir() || fs::read_dir(out)?.next().is_some();
    let old_path = if is_nonempty {
        let backup_name = format!("{}.bak-{}Z", name.to_string_lossy(), unix_seconds());
        let backup = parent.join(backup_name);
        if backup.exists() {
            return Err(FileError::OutputExists);
        }
        backup
    } else {
        unique_sibling(parent, name, "empty-old")
    };
    fs::rename(out, &old_path)?;
    if let Err(error) = fs::rename(stage, out) {
        let _ = fs::rename(&old_path, out);
        return Err(error.into());
    }
    if !is_nonempty {
        fs::remove_dir(old_path)?;
    }
    Ok(())
}

fn unique_sibling(parent: &Path, name: &std::ffi::OsStr, suffix: &str) -> PathBuf {
    let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".{}.{}-{}-{id}",
        name.to_string_lossy(),
        suffix,
        std::process::id()
    ))
}
fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
struct RemoveOnDrop(Option<PathBuf>);
impl RemoveOnDrop {
    fn take(&mut self) -> Option<PathBuf> {
        self.0.take()
    }
}
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            if path.is_dir() {
                let _ = fs::remove_dir_all(path);
            } else {
                let _ = fs::remove_file(path);
            }
        }
    }
}

#[cfg(test)]
mod tests;
