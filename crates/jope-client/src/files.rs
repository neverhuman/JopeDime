//! Client side file transfer helpers for the v1 dataset and artifact routes.
use crate::{
    http::Http,
    transport::{Error as TransportError, Request, Response, Result as TransportResult, Transport},
};
use jope_api::ids::ArtifactId;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
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
pub fn upload_csv(
    http: &Http,
    name: &str,
    reader: impl Read + Send + 'static,
    idempotency_key: &str,
) -> TransportResult<Response> {
    let request = Request::new(2, 0, &[])?
        .query(&[("name", name)])
        .key(Some(idempotency_key))?;
    http.execute_streaming(request, reader, "text/csv")
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
    let mut archive_guard = RemoveOnDrop(Some(archive_path.clone()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&archive_path)?;
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
    drop(archive_guard.take());
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
mod tests {
    use super::*;
    use crate::transport::{HeaderMap, Response};
    use std::{
        io::{BufRead, BufReader, Cursor, Read},
        net::TcpListener,
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };
    use zip::{write::SimpleFileOptions, ZipWriter};

    struct Fake {
        bytes: Vec<u8>,
    }
    impl Transport for Fake {
        fn execute(&self, _: Request) -> TransportResult<Response> {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", "application/octet-stream".parse().unwrap());
            headers.insert("jope-api-version", "1.0".parse().unwrap());
            Ok(Response::new(200, headers, Cursor::new(self.bytes.clone())))
        }
    }
    fn temp_dir() -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let target = std::env::current_exe()
            .unwrap()
            .ancestors()
            .find(|path| path.file_name().is_some_and(|name| name == "target"))
            .unwrap()
            .to_path_buf();
        let path = target.join(format!("jope-client-files-{}-{n}", std::process::id()));
        fs::create_dir(&path).unwrap();
        path
    }
    fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);
        for (name, bytes) in entries {
            writer
                .start_file(*name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }
    fn digest(bytes: &[u8]) -> String {
        format!("sha256:{}", digest_hex(&Sha256::digest(bytes)))
    }
    fn artifact_id() -> ArtifactId {
        "art_0123456789abcdef01234567".parse().unwrap()
    }

    #[test]
    fn csv_upload_streams_to_dataset_route_with_auth_and_idempotency() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut input = BufReader::new(socket.try_clone().unwrap());
            let mut headers = String::new();
            loop {
                let mut line = String::new();
                input.read_line(&mut line).unwrap();
                headers.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = Vec::new();
            loop {
                let mut size = String::new();
                input.read_line(&mut size).unwrap();
                let size = usize::from_str_radix(size.trim(), 16).unwrap();
                if size == 0 {
                    let mut trailer = String::new();
                    input.read_line(&mut trailer).unwrap();
                    break;
                }
                let start = body.len();
                body.resize(start + size, 0);
                input.read_exact(&mut body[start..]).unwrap();
                let mut crlf = [0; 2];
                input.read_exact(&mut crlf).unwrap();
                assert_eq!(&crlf, b"\r\n");
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
            (headers.to_ascii_lowercase(), body)
        });
        let http = Http::new(&endpoint, &"ab".repeat(32)).unwrap();
        upload_csv(
            &http,
            "train data.csv",
            Cursor::new(b"x,y\n1,2\n".to_vec()),
            "upload-1",
        )
        .unwrap();
        let (headers, body) = server.join().unwrap();
        assert!(headers.starts_with("post /v1/datasets?name=train%20data.csv http/1.1\r\n"));
        assert!(headers.contains("content-type: text/csv\r\n"));
        assert!(headers.contains("idempotency-key: upload-1\r\n"));
        assert!(headers.contains(&format!("authorization: bearer {}\r\n", "ab".repeat(32))));
        assert_eq!(body, b"x,y\n1,2\n");
    }

    #[test]
    fn digest_format_must_be_exact_lowercase_sha256() {
        assert!(matches!(
            parse_digest("sha256:ABC"),
            Err(FileError::InvalidDigest)
        ));
        assert!(matches!(
            parse_digest(&format!("sha256:{}", "A".repeat(64))),
            Err(FileError::InvalidDigest)
        ));
    }
    #[test]
    fn verified_zip_is_extracted_and_force_backs_up_existing_output() {
        let root = temp_dir();
        let out = root.join("out");
        fs::create_dir(&out).unwrap();
        fs::write(out.join("old"), "old").unwrap();
        let bytes = zip(&[("nested/a.txt", b"new")]);
        download_artifact(
            &Fake {
                bytes: bytes.clone(),
            },
            &artifact_id(),
            &digest(&bytes),
            &out,
            true,
        )
        .unwrap();
        assert_eq!(fs::read(out.join("nested/a.txt")).unwrap(), b"new");
        let backup = fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .find(|n| n.to_string_lossy().contains(".bak-"))
            .unwrap();
        assert_eq!(fs::read(root.join(backup).join("old")).unwrap(), b"old");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn mismatch_existing_output_and_interrupted_archive_leave_output_untouched() {
        let root = temp_dir();
        let out = root.join("out");
        fs::create_dir(&out).unwrap();
        fs::write(out.join("keep"), "same").unwrap();
        let bytes = zip(&[("new", b"data")]);
        assert!(matches!(
            download_artifact(
                &Fake {
                    bytes: bytes.clone()
                },
                &artifact_id(),
                &digest(b"wrong"),
                &out,
                true
            ),
            Err(FileError::DigestMismatch)
        ));
        assert_eq!(fs::read(out.join("keep")).unwrap(), b"same");
        assert!(matches!(
            download_artifact(
                &Fake {
                    bytes: bytes.clone()
                },
                &artifact_id(),
                &digest(&bytes),
                &out,
                false
            ),
            Err(FileError::OutputExists)
        ));
        assert_eq!(fs::read(out.join("keep")).unwrap(), b"same");
        let broken = b"not a zip".to_vec();
        assert!(download_artifact(
            &Fake {
                bytes: broken.clone()
            },
            &artifact_id(),
            &digest(&broken),
            &out,
            true
        )
        .is_err());
        assert_eq!(fs::read(out.join("keep")).unwrap(), b"same");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn refuses_parent_traversal_and_symlink_entries() {
        let root = temp_dir();
        let dest = root.join("stage");
        fs::create_dir(&dest).unwrap();
        assert!(matches!(
            extract_zip(Cursor::new(zip(&[("../escape", b"x")])), &dest),
            Err(FileError::UnsafeArchive(_))
        ));
        assert!(matches!(
            extract_zip(Cursor::new(zip(&[("C:/escape", b"x")])), &dest),
            Err(FileError::UnsafeArchive(_))
        ));
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);
        writer
            .start_file("link", SimpleFileOptions::default().unix_permissions(0o777))
            .unwrap();
        writer.write_all(b"target").unwrap();
        let mut symlink = writer.finish().unwrap().into_inner();
        let central = symlink
            .windows(4)
            .position(|bytes| bytes == b"PK\x01\x02")
            .unwrap();
        symlink[central + 38..central + 42].copy_from_slice(&(0o120777u32 << 16).to_le_bytes());
        assert!(matches!(
            extract_zip(Cursor::new(symlink), &dest),
            Err(FileError::UnsafeArchive("symlink entry"))
        ));
        fs::remove_dir_all(root).unwrap();
    }
}
