//! Small deterministic transport fault fixtures; no remote credentials, files or model weights.
use super::*;
use anyhow::bail;
use futures_util::{stream, StreamExt};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct FixtureBackend {
    version: Option<String>,
    bytes: Option<Vec<u8>>,
    fail: bool,
    calls: AtomicUsize,
}
impl InjectedBackend for FixtureBackend {
    fn get_versioned(
        &self,
        _: &ObjPath,
        requested: Option<&str>,
    ) -> Result<Option<(Vec<u8>, UpdateVersion)>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ensure!(
            requested == Some("version-a"),
            "fixture was not read at the exact version"
        );
        ensure!(!self.fail, "fixture transport failure");
        Ok(self.bytes.clone().map(|bytes| {
            (
                bytes,
                UpdateVersion {
                    e_tag: Some("version-a".into()),
                    version: self.version.clone(),
                },
            )
        }))
    }
    fn put_create(&self, _: &ObjPath, _: Vec<u8>) -> Result<(PutOutcome, Option<UpdateVersion>)> {
        bail!("read-only fixture")
    }
    fn put_overwrite(&self, _: &ObjPath, _: Vec<u8>) -> Result<UpdateVersion> {
        bail!("read-only fixture")
    }
    fn list_keys(&self, _: &ObjPath) -> Result<Vec<ObjPath>> {
        bail!("read-only fixture")
    }
}
fn fixture(
    version: Option<&str>,
    bytes: Option<Vec<u8>>,
    fail: bool,
) -> (Store, Arc<FixtureBackend>) {
    let backend = Arc::new(FixtureBackend {
        version: version.map(str::to_owned),
        bytes,
        fail,
        calls: AtomicUsize::new(0),
    });
    let store = Store::from_injected_backend(
        "s3://fixture.invalid",
        ObjPath::from("release"),
        ObjectStoreScheme::AmazonS3,
        backend.clone(),
    );
    (store, backend)
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn exact_version_bytes_and_key_are_returned_from_the_same_read() {
    let (store, backend) = fixture(Some("version-a"), Some(vec![1, 2]), false);
    let object = runtime()
        .block_on(store.get_observed_version_bounded("blob", "version-a", 3, Some(2)))
        .unwrap()
        .unwrap();
    assert_eq!(object.bytes, [1, 2]);
    assert_eq!(object.version(), Some("version-a"));
    assert_eq!(object.key_str(), "release/blob");
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
}
#[test]
fn matching_etag_cannot_substitute_for_missing_or_wrong_physical_version() {
    for version in [None, Some("version-b")] {
        let (store, _) = fixture(version, Some(vec![1, 2]), false);
        let error = runtime()
            .block_on(store.get_observed_version_bounded("blob", "version-a", 3, Some(2)))
            .unwrap_err();
        assert!(error.to_string().contains("VersionId mismatch"));
    }
}
#[test]
fn invalid_version_or_budget_refuses_before_the_backend_is_called() {
    let (store, backend) = fixture(Some("version-a"), Some(vec![1, 2]), false);
    for version in ["", " ", "null", "UNPINNED", "bad\nversion"] {
        assert!(runtime()
            .block_on(store.get_observed_version_bounded("blob", version, 3, None))
            .is_err());
    }
    for (maximum, expected) in [(0, None), (3, Some(4)), (u64::MAX, None)] {
        assert!(runtime()
            .block_on(store.get_observed_version_bounded("blob", "version-a", maximum, expected))
            .is_err());
    }
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}
#[test]
fn short_long_and_over_budget_fixture_bodies_refuse() {
    for (maximum, expected) in [(3, Some(1)), (3, Some(3)), (1, None)] {
        let (store, _) = fixture(Some("version-a"), Some(vec![1, 2]), false);
        assert!(runtime()
            .block_on(store.get_observed_version_bounded("blob", "version-a", maximum, expected))
            .is_err());
    }
}
#[test]
fn missing_objects_and_transport_errors_are_distinguished() {
    let (missing, _) = fixture(Some("version-a"), None, false);
    assert!(runtime()
        .block_on(missing.get_observed_version_bounded("blob", "version-a", 3, None))
        .unwrap()
        .is_none());
    let (broken, _) = fixture(Some("version-a"), Some(vec![1]), true);
    let error = runtime()
        .block_on(broken.get_observed_version_bounded("blob", "version-a", 3, None))
        .unwrap_err();
    assert!(error.to_string().contains("fixture transport failure"));
}
fn ok(bytes: Vec<u8>) -> std::io::Result<Vec<u8>> {
    Ok(bytes)
}
#[test]
fn streaming_chunks_preserve_the_exact_body() {
    let chunks = stream::iter([ok(vec![1, 2]), ok(vec![]), ok(vec![3, 4])]);
    assert_eq!(
        runtime()
            .block_on(collect_bounded_response(chunks, 4, 4))
            .unwrap(),
        [1, 2, 3, 4]
    );
}
#[test]
fn oversized_metadata_refuses_before_polling_the_body() {
    let polls = Arc::new(AtomicUsize::new(0));
    let count = polls.clone();
    let chunks = stream::iter([ok(vec![1])]).inspect(move |_| {
        count.fetch_add(1, Ordering::SeqCst);
    });
    assert!(runtime()
        .block_on(collect_bounded_response(chunks, 5, 4))
        .is_err());
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}
#[test]
fn a_long_stream_refuses_without_polling_later_chunks() {
    let polls = Arc::new(AtomicUsize::new(0));
    let count = polls.clone();
    let chunks = stream::iter([ok(vec![1, 2]), ok(vec![3, 4]), ok(vec![5])]).inspect(move |_| {
        count.fetch_add(1, Ordering::SeqCst);
    });
    let error = runtime()
        .block_on(collect_bounded_response(chunks, 3, 4))
        .unwrap_err();
    assert!(error.to_string().contains("exceeds declared size"));
    assert_eq!(polls.load(Ordering::SeqCst), 2);
}
#[test]
fn a_short_stream_and_stream_error_never_return_partial_success() {
    let error = runtime()
        .block_on(collect_bounded_response(stream::iter([ok(vec![1])]), 2, 2))
        .unwrap_err();
    assert!(error.to_string().contains("ended before declared size"));
    let chunks = stream::iter([ok(vec![1]), Err(std::io::Error::other("broken stream"))]);
    assert!(runtime()
        .block_on(collect_bounded_response(chunks, 2, 2))
        .unwrap_err()
        .to_string()
        .contains("broken stream"));
}
#[test]
fn a_declared_empty_body_is_supported_with_a_positive_bound() {
    let chunks = stream::empty::<std::io::Result<Vec<u8>>>();
    assert!(runtime()
        .block_on(collect_bounded_response(chunks, 0, 1))
        .unwrap()
        .is_empty());
}

// Exercise the actual AmazonS3 -> get_opts -> GetResult path against a local,
// unsigned HTTP fixture. No credentials, remote service or model data is used.
struct HttpFixture {
    stop: Arc<std::sync::atomic::AtomicBool>,
    requests: Arc<std::sync::Mutex<Vec<String>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl HttpFixture {
    fn assert_exact_request(&self) {
        let requests = self.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "no retries or latest fallback");
        assert!(
            requests[0].starts_with("GET /fixture/release/blob?versionId=version-a HTTP/1.1\r\n"),
            "{}",
            requests[0]
        );
        assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));
    }
}
impl Drop for HttpFixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn http_fixture(
    status: &str,
    version: Option<&str>,
    declared: usize,
    body: &[u8],
) -> (Store, HttpFixture) {
    use std::io::{Read as _, Write as _};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let thread_stop = stop.clone();
    let thread_requests = requests.clone();
    let version_header = version
        .map(|v| format!("x-amz-version-id: {v}\r\n"))
        .unwrap_or_default();
    let mut response = format!("HTTP/1.1 {status}\r\nContent-Length: {declared}\r\nETag: \"version-a\"\r\nLast-Modified: Mon, 19 Oct 2026 00:00:00 GMT\r\n{version_header}Connection: close\r\n\r\n").into_bytes();
    response.extend_from_slice(body);
    let worker = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !thread_stop.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            match listener.accept() {
                Ok((mut socket, _)) => {
                    socket
                        .set_read_timeout(Some(std::time::Duration::from_millis(200)))
                        .unwrap();
                    socket
                        .set_write_timeout(Some(std::time::Duration::from_millis(200)))
                        .unwrap();
                    let mut request = Vec::new();
                    let mut chunk = [0; 1024];
                    while request.len() < 16 * 1024 && !request.ends_with(b"\r\n\r\n") {
                        match socket.read(&mut chunk) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => request.extend_from_slice(&chunk[..n]),
                        }
                    }
                    thread_requests
                        .lock()
                        .unwrap()
                        .push(String::from_utf8(request).unwrap());
                    let _ = socket.write_all(&response);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(2))
                }
                Err(_) => break,
            }
        }
    });
    let backend = object_store::aws::AmazonS3Builder::new()
        .with_bucket_name("fixture")
        .with_region("us-east-2")
        .with_endpoint(format!("http://{address}"))
        .with_allow_http(true)
        .with_skip_signature(true)
        .with_retry(object_store::RetryConfig {
            max_retries: 0,
            ..Default::default()
        })
        .with_client_options(
            object_store::ClientOptions::new()
                .with_allow_http(true)
                .with_timeout(std::time::Duration::from_secs(2)),
        )
        .build()
        .unwrap();
    let store = Store {
        inner: Box::new(backend),
        prefix: ObjPath::from("release"),
        url: format!("http://{address}/fixture"),
        local_root: None,
    };
    (
        store,
        HttpFixture {
            stop,
            requests,
            worker: Some(worker),
        },
    )
}

#[test]
fn real_get_opts_sends_the_pinned_query_and_returns_the_same_metadata_and_bytes() {
    let (store, server) = http_fixture("200 OK", Some("version-a"), 3, b"abc");
    let object = runtime()
        .block_on(store.get_observed_version_bounded("blob", "version-a", 4, Some(3)))
        .unwrap()
        .unwrap();
    assert_eq!(object.bytes, b"abc");
    assert_eq!(object.version(), Some("version-a"));
    assert_eq!(object.key_str(), "release/blob");
    server.assert_exact_request();
}

#[test]
fn real_get_opts_refuses_missing_and_wrong_version_even_with_the_matching_etag() {
    for version in [None, Some("version-b")] {
        let (store, server) = http_fixture("200 OK", version, 3, b"abc");
        let error = runtime()
            .block_on(store.get_observed_version_bounded("blob", "version-a", 4, Some(3)))
            .unwrap_err();
        assert!(format!("{error:#}").contains("VersionId mismatch"));
        server.assert_exact_request();
    }
}

#[test]
fn real_get_opts_refuses_bad_declared_or_expected_lengths() {
    for (maximum, expected, message) in [
        (4, Some(2), "differs from expected length"),
        (2, None, "exceeds"),
    ] {
        let (store, server) = http_fixture("200 OK", Some("version-a"), 3, b"abc");
        let error = runtime()
            .block_on(store.get_observed_version_bounded("blob", "version-a", maximum, expected))
            .unwrap_err();
        assert!(format!("{error:#}").contains(message), "{error:#}");
        server.assert_exact_request();
    }
}

#[test]
fn real_get_opts_distinguishes_missing_from_access_denied_without_retry() {
    let missing = b"<Error><Code>NoSuchKey</Code><Message>missing</Message></Error>";
    let (store, server) = http_fixture("404 Not Found", None, missing.len(), missing);
    assert!(runtime()
        .block_on(store.get_observed_version_bounded("blob", "version-a", 4, None))
        .unwrap()
        .is_none());
    server.assert_exact_request();
    let denied = b"<Error><Code>AccessDenied</Code><Message>denied</Message></Error>";
    let (store, server) = http_fixture("403 Forbidden", None, denied.len(), denied);
    assert!(runtime()
        .block_on(store.get_observed_version_bounded("blob", "version-a", 4, None))
        .is_err());
    server.assert_exact_request();
}

#[test]
fn real_get_opts_stream_truncation_never_returns_partial_bytes() {
    let (store, server) = http_fixture("200 OK", Some("version-a"), 4, b"ab");
    assert!(runtime()
        .block_on(store.get_observed_version_bounded("blob", "version-a", 4, Some(4)))
        .is_err());
    server.assert_exact_request();
}

#[test]
fn excessive_version_and_nonempty_expected_zero_refuse() {
    let (store, backend) = fixture(Some("version-a"), Some(vec![1]), false);
    assert!(runtime()
        .block_on(store.get_observed_version_bounded("blob", &"v".repeat(1025), 1, None))
        .is_err());
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    assert!(runtime()
        .block_on(store.get_observed_version_bounded("blob", "version-a", 1, Some(0)))
        .is_err());
}

#[test]
fn dropping_an_injected_store_releases_its_registered_backend() {
    let (store, backend) = fixture(Some("version-a"), Some(vec![1]), false);
    let weak = Arc::downgrade(&backend);
    drop(backend);
    assert!(weak.upgrade().is_some());
    drop(store);
    assert!(
        weak.upgrade().is_none(),
        "registry retained a backend after its Store was dropped"
    );
}
#[test]
fn dropping_one_store_does_not_remove_another_stores_backend() {
    let (first, _) = fixture(Some("version-a"), Some(vec![1]), false);
    let (second, backend) = fixture(Some("version-a"), Some(vec![2]), false);
    drop(first);
    let object = runtime()
        .block_on(second.get_observed_version_bounded("blob", "version-a", 1, Some(1)))
        .unwrap()
        .unwrap();
    assert_eq!(object.bytes, [2]);
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn unknown_pinned_versions_are_absence_or_errors_without_fallback() {
    for (status, code, absent) in [
        ("404 Not Found", "NoSuchVersion", true),
        ("400 Bad Request", "InvalidArgument", false),
    ] {
        let body =
            format!("<Error><Code>{code}</Code><Message>unknown pinned version</Message></Error>");
        let (store, server) = http_fixture(status, None, body.len(), body.as_bytes());
        let result =
            runtime().block_on(store.get_observed_version_bounded("blob", "version-a", 4, None));
        if absent {
            // None is missing pinned content, never a successful or substituted release.
            assert!(result.unwrap().is_none());
        } else {
            assert!(result.is_err());
        }
        server.assert_exact_request();
    }
}
