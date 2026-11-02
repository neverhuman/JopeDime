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

#[path = "tests/download.rs"]
mod download;
#[path = "tests/upload.rs"]
mod upload;

#[test]
fn generic_csv_upload_preserves_name_key_and_bytes() {
    upload::generic_csv_upload_preserves_name_key_and_bytes();
}

#[test]
fn atomic_csv_write_publishes_complete_file_and_refuses_overwrite() {
    upload::atomic_csv_write_publishes_complete_file_and_refuses_overwrite();
}

#[test]
fn atomic_csv_retry_verifies_existing_bytes_without_replacing_the_file() {
    upload::atomic_csv_retry_verifies_existing_bytes_without_replacing_the_file();
}

#[test]
fn atomic_csv_retry_refuses_directories_and_symlinks_even_to_identical_bytes() {
    upload::atomic_csv_retry_refuses_directories_and_symlinks_even_to_identical_bytes();
}

#[test]
fn atomic_csv_write_removes_partial_temporary_after_read_error() {
    upload::atomic_csv_write_removes_partial_temporary_after_read_error();
}

#[test]
fn atomic_csv_failed_retry_preserves_existing_file_and_removes_staging() {
    upload::atomic_csv_failed_retry_preserves_existing_file_and_removes_staging();
}

#[test]
fn atomic_csv_publication_verifies_a_concurrent_creator_without_overwriting() {
    upload::atomic_csv_publication_verifies_a_concurrent_creator_without_overwriting();
}

#[test]
fn csv_upload_streams_to_dataset_route_with_auth_and_idempotency() {
    upload::csv_upload_streams_to_dataset_route_with_auth_and_idempotency();
}

#[test]
fn digest_format_must_be_exact_lowercase_sha256() {
    download::digest_format_must_be_exact_lowercase_sha256();
}

#[test]
fn export_retry_checks_every_existing_byte_and_refuses_tampering() {
    download::export_retry_checks_every_existing_byte_and_refuses_tampering();
}

#[test]
fn download_collision_preserves_existing_directory() {
    download::download_collision_preserves_existing_directory();
}

#[test]
fn download_archive_is_cleaned_after_success_and_publish_refusal() {
    download::download_archive_is_cleaned_after_success_and_publish_refusal();
}

#[test]
fn verified_zip_is_extracted_and_force_backs_up_existing_output() {
    download::verified_zip_is_extracted_and_force_backs_up_existing_output();
}

#[test]
fn mismatch_existing_output_and_interrupted_archive_leave_output_untouched() {
    download::mismatch_existing_output_and_interrupted_archive_leave_output_untouched();
}

#[test]
fn refuses_parent_traversal_and_symlink_entries() {
    download::refuses_parent_traversal_and_symlink_entries();
}
