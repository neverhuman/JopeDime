use super::*;
use crate::{
    http::Http,
    transport::{HeaderMap, Request, Response},
};
use std::{
    io::{BufRead, BufReader, Cursor, Read},
    net::TcpListener,
    sync::Mutex,
    thread,
};
use zip::{write::SimpleFileOptions, ZipWriter};

struct Fake {
    bytes: Vec<u8>,
}
impl Transport for Fake {
    fn execute(&self, _: Request) -> TransportResult<Response> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-jope-artifact-sha256",
            digest_hex(&Sha256::digest(&self.bytes)).parse().unwrap(),
        );
        headers.insert("content-type", "application/octet-stream".parse().unwrap());
        headers.insert("jope-api-version", "1.0".parse().unwrap());
        Ok(Response::new(200, headers, Cursor::new(self.bytes.clone())))
    }
}
/// A fresh test directory, removed on drop by `tempfile`.
struct TestTempDir(tempfile::TempDir);

impl std::ops::Deref for TestTempDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.path()
    }
}

impl AsRef<Path> for TestTempDir {
    fn as_ref(&self) -> &Path {
        self.0.path()
    }
}

fn temp_dir() -> TestTempDir {
    TestTempDir(
        tempfile::Builder::new()
            .prefix("jope-client-files-")
            .tempdir()
            .unwrap(),
    )
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

#[derive(Default)]
struct UploadCapture(Mutex<Option<(String, String, Vec<u8>)>>);
impl Transport for UploadCapture {
    fn execute(&self, _: Request) -> TransportResult<Response> {
        unreachable!("upload test uses the streaming method")
    }

    fn dataset_register_stream(
        &self,
        name: &str,
        mut body: Box<dyn Read + Send>,
        idempotency_key: &str,
    ) -> TransportResult<Response> {
        let mut bytes = Vec::new();
        body.read_to_end(&mut bytes)?;
        *self.0.lock().unwrap() = Some((name.to_owned(), idempotency_key.to_owned(), bytes));
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("jope-api-version", "1.0".parse().unwrap());
        Ok(Response::new(200, headers, Cursor::new(b"{}".to_vec())))
    }
}

#[test]
fn generic_csv_upload_preserves_name_key_and_bytes() {
    let transport = UploadCapture::default();
    upload_csv(
        &transport,
        "input.csv",
        Cursor::new(b"x,y\n1,2\n"),
        "retry-key",
    )
    .unwrap();
    assert_eq!(
        transport.0.lock().unwrap().as_ref().unwrap(),
        &(
            "input.csv".into(),
            "retry-key".into(),
            b"x,y\n1,2\n".to_vec()
        )
    );
}

#[test]
fn atomic_csv_write_publishes_complete_file_and_refuses_overwrite() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    assert_eq!(
        write_csv_atomic(&out, Cursor::new(b"prediction\n0.25\n".to_vec())).unwrap(),
        16
    );
    assert_eq!(fs::read(&out).unwrap(), b"prediction\n0.25\n");
    assert!(matches!(
        write_csv_atomic(&out, Cursor::new(b"replacement".to_vec())),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(&out).unwrap(), b"prediction\n0.25\n");
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
}

#[test]
fn atomic_csv_retry_verifies_existing_bytes_without_replacing_the_file() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    let csv = b"prediction\n0.25\n0.5\n";
    write_csv_atomic(&out, Cursor::new(csv)).unwrap();
    // Keeping this handle open also proves that a successful replay does not
    // replace the original file with a newly staged one.
    let mut original = File::open(&out).unwrap();
    let before = original.metadata().unwrap();
    assert_eq!(
        write_csv_atomic(&out, Cursor::new(csv)).unwrap(),
        csv.len() as u64
    );
    assert_eq!(
        fs::metadata(&out).unwrap().modified().unwrap(),
        before.modified().unwrap()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(fs::metadata(&out).unwrap().ino(), before.ino());
    }
    let mut retained = Vec::new();
    original.read_to_end(&mut retained).unwrap();
    assert_eq!(retained, csv);
    for different in [
        &b"prediction\n0.25\n0.6\n"[..],
        &csv[..csv.len() - 1],
        b"prediction\n0.25\n0.5\n0\n",
    ] {
        assert!(matches!(
            write_csv_atomic(&out, Cursor::new(different)),
            Err(FileError::OutputExists)
        ));
        assert_eq!(fs::read(&out).unwrap(), csv);
    }
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
}

#[test]
fn atomic_csv_retry_refuses_directories_and_symlinks_even_to_identical_bytes() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    fs::create_dir(&out).unwrap();
    assert!(matches!(
        write_csv_atomic(&out, Cursor::new(b"prediction\n0.25\n")),
        Err(FileError::OutputExists)
    ));
    assert!(out.is_dir());
    #[cfg(unix)]
    {
        let original = temp.join("original.csv");
        fs::write(&original, b"prediction\n0.25\n").unwrap();
        let link = temp.join("linked.csv");
        std::os::unix::fs::symlink(&original, &link).unwrap();
        assert!(matches!(
            write_csv_atomic(&link, Cursor::new(b"prediction\n0.25\n")),
            Err(FileError::OutputExists)
        ));
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&original).unwrap(), b"prediction\n0.25\n");
    }
}

struct FailsAfterPrefix(bool);
impl Read for FailsAfterPrefix {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if !self.0 {
            self.0 = true;
            output[..7].copy_from_slice(b"partial");
            return Ok(7);
        }
        Err(io::Error::other("fixture read failure"))
    }
}

#[test]
fn atomic_csv_write_removes_partial_temporary_after_read_error() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    assert!(write_csv_atomic(&out, FailsAfterPrefix(false)).is_err());
    assert!(!out.exists());
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 0);
}

#[test]
fn atomic_csv_failed_retry_preserves_existing_file_and_removes_staging() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    fs::write(&out, b"prediction\n0.25\n").unwrap();
    assert!(matches!(
        write_csv_atomic(&out, FailsAfterPrefix(false)),
        Err(FileError::Io(_))
    ));
    assert_eq!(fs::read(&out).unwrap(), b"prediction\n0.25\n");
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
}

#[test]
fn atomic_csv_publication_verifies_a_concurrent_creator_without_overwriting() {
    struct PublishDuringRead<'a> {
        out: &'a Path,
        concurrent: Option<&'a [u8]>,
        response: Cursor<&'a [u8]>,
    }
    impl Read for PublishDuringRead<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if let Some(concurrent) = self.concurrent.take() {
                fs::write(self.out, concurrent)?;
            }
            self.response.read(bytes)
        }
    }
    let csv = b"prediction\n0.25\n";
    for concurrent in [&csv[..], &b"prediction\n0.75\n"[..]] {
        let temp = temp_dir();
        let out = temp.join("prediction.csv");
        let result = write_csv_atomic(
            &out,
            PublishDuringRead {
                out: &out,
                concurrent: Some(concurrent),
                response: Cursor::new(csv),
            },
        );
        if concurrent == csv {
            assert_eq!(result.unwrap(), csv.len() as u64);
        } else {
            assert!(matches!(result, Err(FileError::OutputExists)));
        }
        assert_eq!(fs::read(&out).unwrap(), concurrent);
        assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
    }
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
fn export_retry_checks_every_existing_byte_and_refuses_tampering() {
    let temp = temp_dir();
    let out = temp.join("export");
    fs::create_dir(&out).unwrap(); // Existing empty directories are valid destinations.
    let transport = Fake {
        bytes: zip(&[("model/data", b"frozen"), ("manifest.json", b"manifest")]),
    };
    download_export_artifact(&transport, &artifact_id(), &out, false).unwrap();
    download_export_artifact(&transport, &artifact_id(), &out, false).unwrap();
    fs::write(out.join("model/data"), b"edited").unwrap();
    assert!(matches!(
        download_export_artifact(&transport, &artifact_id(), &out, false),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(out.join("model/data")).unwrap(), b"edited");
    fs::write(out.join("model/data"), b"frozen").unwrap();
    fs::write(out.join("extra"), b"keep").unwrap();
    assert!(matches!(
        download_export_artifact(&transport, &artifact_id(), &out, false),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(out.join("extra")).unwrap(), b"keep");
    #[cfg(unix)]
    {
        fs::remove_file(out.join("extra")).unwrap();
        fs::remove_file(out.join("model/data")).unwrap();
        std::os::unix::fs::symlink("../manifest.json", out.join("model/data")).unwrap();
        assert!(matches!(
            download_export_artifact(&transport, &artifact_id(), &out, false),
            Err(FileError::OutputExists)
        ));
        assert!(fs::symlink_metadata(out.join("model/data"))
            .unwrap()
            .file_type()
            .is_symlink());
    }
}

#[test]
fn download_collision_preserves_existing_directory() {
    const CHILD: &str = "JOPE_CLIENT_DOWNLOAD_COLLISION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // A fresh process isolates the staging counter from parallel tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "files::tests::download_collision_preserves_existing_directory",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "collision child failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let root = temp_dir();
    let existing = root.join(format!(".out.download-{}-0", std::process::id()));
    fs::create_dir(&existing).unwrap();
    fs::write(existing.join("keep"), b"existing data").unwrap();
    let bytes = zip(&[("new", b"downloaded data")]);
    let result = download_artifact(
        &Fake {
            bytes: bytes.clone(),
        },
        &artifact_id(),
        &digest(&bytes),
        &root.join("out"),
        false,
    );
    assert!(
        matches!(result, Err(FileError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists)
    );
    assert_eq!(
        fs::read(existing.join("keep")).ok().as_deref(),
        Some(b"existing data".as_slice()),
        "exclusive-create failure removed a pre-existing directory"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn download_archive_is_cleaned_after_success_and_publish_refusal() {
    let root = temp_dir();
    let out = root.join("out");
    let bytes = zip(&[("new", b"downloaded data")]);
    let transport = Fake {
        bytes: bytes.clone(),
    };
    let expected = digest(&bytes);
    download_artifact(&transport, &artifact_id(), &expected, &out, false).unwrap();
    let download_archives = || {
        fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains(".download-"))
            .count()
    };
    assert_eq!(download_archives(), 0, "successful download leaked its ZIP");
    assert!(matches!(
        download_artifact(&transport, &artifact_id(), &expected, &out, false),
        Err(FileError::OutputExists)
    ));
    assert_eq!(download_archives(), 0, "refused publication leaked its ZIP");
    assert_eq!(fs::read(out.join("new")).unwrap(), b"downloaded data");
    fs::remove_dir_all(root).unwrap();
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
