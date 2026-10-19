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
