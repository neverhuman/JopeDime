//! Integration tests for `jope_datasets::verified_object` (`I46`).
//!
//! Covers:
//! 1. Fresh correct publication on `file://`, `memory://`, and injected versioned backends.
//! 2. Existing identical publication (`ExistingVerified`).
//! 3. Existing same-size wrong content refusal (no overwrite, no repair).
//! 4. Concurrent create race (`AlreadyExists`) with same-size wrong content refusal.
//! 5. Source snapshot verification (`snapshot_and_verify_source_file`) and key/hash validation.
//! 6. Successful `PUT` followed by corrupted or absent readback refusal.
//! 7. Lost/ambiguous `PUT` response reconciliation (succeeds only on exact content + version;
//!    fails when absent or wrong bytes, preserving both errors).
//! 8. Explicit access denial (`AccessDenied` / `PermissionDenied`) refusal without reconciliation.
//! 9. Physical `VersionId` vs opaque `ETag` separation and exact-version readback under adversarial
//!    key replacement.

use anyhow::{anyhow, Result};
use jope_datasets::blobs::blob_key;
use jope_datasets::hash::sha256_hex;
use jope_datasets::store::{
    InjectedBackend, ObjPath, ObjectStoreScheme, PutOutcome, Store, UpdateVersion,
};
use jope_datasets::verified_object::{
    publish_verified_object, publish_verified_object_with_options, snapshot_and_verify_source_file,
    verify_existing_object_at_version, PhysicalVersionStatus, PublicationStatus,
    VerificationOptions, VERIFIED_OBJECT_CONTRACT_ID,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[derive(Clone, Debug)]
struct VersionRecord {
    version_id: Option<String>,
    e_tag: Option<String>,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
enum PutFault {
    /// Simulate a concurrent writer winning the create race with `bytes`.
    CreateRaceWinner(Vec<u8>),
    /// Write the object to storage, then fail with an ambiguous lost-response transport error.
    LostResponseAfterWrite(Vec<u8>),
    /// Fail before writing with a transport error (object stays absent or at prior state).
    LostResponseBeforeWrite,
    /// Fail with an explicit permission denial (`AccessDenied`).
    AccessDenied,
    /// Return `Created` from `put_create`, but store corrupted bytes in the backend.
    StoreCorruptBytesOnCreate(Vec<u8>),
    /// Return `Created` from `put_create`, but drop the stored object before readback.
    DropOnCreate,
    /// Return `Created` with one `VersionId`, but store a different `VersionId` on readback.
    MismatchedReadbackVersion { promised: String, stored: String },
}

struct MockVersionedBackend {
    emit_version_ids: bool,
    versions: Mutex<BTreeMap<String, Vec<VersionRecord>>>,
    next_fault: Mutex<Option<PutFault>>,
    next_seq: Mutex<u64>,
    put_create_calls: Mutex<u64>,
}

impl MockVersionedBackend {
    fn new(emit_version_ids: bool) -> Arc<Self> {
        Arc::new(Self {
            emit_version_ids,
            versions: Mutex::new(BTreeMap::new()),
            next_fault: Mutex::new(None),
            next_seq: Mutex::new(1000),
            put_create_calls: Mutex::new(0),
        })
    }

    fn set_next_fault(&self, fault: PutFault) {
        *self.next_fault.lock().unwrap() = Some(fault);
    }

    fn allocate_version(&self, bytes: &[u8]) -> (Option<String>, Option<String>) {
        let mut seq = self.next_seq.lock().unwrap();
        *seq += 1;
        let ver = self.emit_version_ids.then(|| format!("v-{}", *seq));
        let etag = Some(format!("\"etag-{}-{}\"", *seq, bytes.len()));
        (ver, etag)
    }

    fn current_bytes(&self, rel: &str) -> Option<Vec<u8>> {
        self.versions
            .lock()
            .unwrap()
            .get(rel)
            .and_then(|v| v.last())
            .map(|r| r.bytes.clone())
    }
}

impl InjectedBackend for MockVersionedBackend {
    fn get_versioned(
        &self,
        key: &ObjPath,
        required_version: Option<&str>,
    ) -> Result<Option<(Vec<u8>, UpdateVersion)>> {
        let map = self.versions.lock().unwrap();
        let Some(history) = map.get(key.as_ref()) else {
            return Ok(None);
        };
        let record = match required_version {
            Some(req) => history
                .iter()
                .rev()
                .find(|r| r.version_id.as_deref() == Some(req)),
            None => history.last(),
        };
        Ok(record.map(|r| {
            (
                r.bytes.clone(),
                UpdateVersion {
                    e_tag: r.e_tag.clone(),
                    version: r.version_id.clone(),
                },
            )
        }))
    }

    fn put_create(
        &self,
        key: &ObjPath,
        bytes: Vec<u8>,
    ) -> Result<(PutOutcome, Option<UpdateVersion>)> {
        *self.put_create_calls.lock().unwrap() += 1;
        if let Some(fault) = self.next_fault.lock().unwrap().take() {
            match fault {
                PutFault::CreateRaceWinner(winner_bytes) => {
                    let (version_id, e_tag) = self.allocate_version(&winner_bytes);
                    self.versions
                        .lock()
                        .unwrap()
                        .entry(key.as_ref().to_owned())
                        .or_default()
                        .push(VersionRecord {
                            version_id,
                            e_tag,
                            bytes: winner_bytes,
                        });
                    return Ok((PutOutcome::Exists, None));
                }
                PutFault::LostResponseAfterWrite(stored_bytes) => {
                    let (version_id, e_tag) = self.allocate_version(&stored_bytes);
                    self.versions
                        .lock()
                        .unwrap()
                        .entry(key.as_ref().to_owned())
                        .or_default()
                        .push(VersionRecord {
                            version_id,
                            e_tag,
                            bytes: stored_bytes,
                        });
                    return Err(anyhow!(
                        "put {}: simulated connection reset after write (lost response)",
                        key.as_ref()
                    ));
                }
                PutFault::LostResponseBeforeWrite => {
                    return Err(anyhow!(
                        "put {}: simulated transport timeout before write",
                        key.as_ref()
                    ));
                }
                PutFault::AccessDenied => {
                    return Err(anyhow!(
                        "put {}: AccessDenied: User is not authorized to perform s3:PutObject",
                        key.as_ref()
                    ));
                }
                PutFault::StoreCorruptBytesOnCreate(corrupt) => {
                    let (version_id, e_tag) = self.allocate_version(&corrupt);
                    self.versions
                        .lock()
                        .unwrap()
                        .entry(key.as_ref().to_owned())
                        .or_default()
                        .push(VersionRecord {
                            version_id: version_id.clone(),
                            e_tag: e_tag.clone(),
                            bytes: corrupt,
                        });
                    return Ok((
                        PutOutcome::Created,
                        Some(UpdateVersion {
                            e_tag,
                            version: version_id,
                        }),
                    ));
                }
                PutFault::DropOnCreate => {
                    let (version_id, e_tag) = self.allocate_version(&bytes);
                    return Ok((
                        PutOutcome::Created,
                        Some(UpdateVersion {
                            e_tag,
                            version: version_id,
                        }),
                    ));
                }
                PutFault::MismatchedReadbackVersion { promised, stored } => {
                    let (_, e_tag) = self.allocate_version(&bytes);
                    self.versions
                        .lock()
                        .unwrap()
                        .entry(key.as_ref().to_owned())
                        .or_default()
                        .push(VersionRecord {
                            version_id: Some(stored),
                            e_tag: e_tag.clone(),
                            bytes,
                        });
                    return Ok((
                        PutOutcome::Created,
                        Some(UpdateVersion {
                            e_tag,
                            version: Some(promised),
                        }),
                    ));
                }
            }
        }

        let mut map = self.versions.lock().unwrap();
        if let Some(history) = map.get(key.as_ref()) {
            if !history.is_empty() {
                return Ok((PutOutcome::Exists, None));
            }
        }
        let (version_id, e_tag) = self.allocate_version(&bytes);
        map.entry(key.as_ref().to_owned())
            .or_default()
            .push(VersionRecord {
                version_id: version_id.clone(),
                e_tag: e_tag.clone(),
                bytes,
            });
        Ok((
            PutOutcome::Created,
            Some(UpdateVersion {
                e_tag,
                version: version_id,
            }),
        ))
    }

    fn put_overwrite(&self, key: &ObjPath, bytes: Vec<u8>) -> Result<UpdateVersion> {
        let (version_id, e_tag) = self.allocate_version(&bytes);
        self.versions
            .lock()
            .unwrap()
            .entry(key.as_ref().to_owned())
            .or_default()
            .push(VersionRecord {
                version_id: version_id.clone(),
                e_tag: e_tag.clone(),
                bytes,
            });
        Ok(UpdateVersion {
            e_tag,
            version: version_id,
        })
    }

    fn list_keys(&self, prefix: &ObjPath) -> Result<Vec<ObjPath>> {
        let map = self.versions.lock().unwrap();
        let p = prefix.as_ref();
        Ok(map
            .keys()
            .filter(|k| p.is_empty() || k.starts_with(p))
            .map(|k| ObjPath::from(k.as_str()))
            .collect())
    }
}

#[test]
fn fresh_and_existing_verified_objects_report_truthful_backend_and_version_status() {
    let tmp = tempfile::tempdir().unwrap();
    let file_store = Store::open(&format!("file://{}", tmp.path().display())).unwrap();
    let mem_store = Store::open("memory:///").unwrap();

    let body = b"immutable payload v1";
    let sha = sha256_hex(body);
    let key = blob_key(&sha);

    // 1. Local file:// fixture store: Created then ExistingVerified, UnversionedLocalFixture.
    let v1 = rt()
        .block_on(publish_verified_object(
            &file_store,
            &key,
            body,
            &sha,
            body.len() as u64,
        ))
        .unwrap();
    assert_eq!(v1.put_outcome(), PutOutcome::Created);
    assert_eq!(v1.reference().schema, VERIFIED_OBJECT_CONTRACT_ID);
    assert_eq!(v1.reference().status, PublicationStatus::Created);
    assert_eq!(
        v1.reference().version_status,
        PhysicalVersionStatus::UnversionedLocalFixture
    );
    assert_eq!(v1.reference().physical_version_id, None);
    assert!(!v1.reference().live_s3_qualified);

    let v2 = rt()
        .block_on(publish_verified_object(
            &file_store,
            &key,
            body,
            &sha,
            body.len() as u64,
        ))
        .unwrap();
    assert_eq!(v2.put_outcome(), PutOutcome::Exists);
    assert_eq!(v2.reference().status, PublicationStatus::ExistingVerified);

    // 2. Memory fixture store: UnversionedMemoryFixture, and refuses RequirePhysicalVersionId.
    let vm = rt()
        .block_on(publish_verified_object(
            &mem_store,
            &key,
            body,
            &sha,
            body.len() as u64,
        ))
        .unwrap();
    assert_eq!(
        vm.reference().version_status,
        PhysicalVersionStatus::UnversionedMemoryFixture
    );
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &mem_store,
            &key,
            body,
            &sha,
            body.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no physical VersionId"),
        "unversioned backend must refuse RequirePhysicalVersionId: {err}"
    );
}

#[test]
fn existing_same_size_wrong_content_and_create_race_refuse_without_overwriting() {
    let backend = MockVersionedBackend::new(true);
    let store = Store::from_injected_backend(
        "s3://test-versioned-bucket/prefix",
        ObjPath::from("prefix"),
        ObjectStoreScheme::AmazonS3,
        backend.clone(),
    );

    let good = b"one: permitted weights";
    let wrong_same_size = b"one: corrupted weights";
    assert_eq!(good.len(), wrong_same_size.len());
    let good_sha = sha256_hex(good);
    let key = blob_key(&good_sha);

    // Pre-populate same-size wrong bytes under the immutable key.
    rt().block_on(store.put(&key, wrong_same_size.to_vec()))
        .unwrap();

    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key,
            good,
            &good_sha,
            good.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("different bytes") && err.contains("immutable"),
        "expected content-mismatch refusal, got: {err}"
    );
    assert_eq!(
        backend.current_bytes(&format!("prefix/{key}")).as_deref(),
        Some(wrong_same_size.as_slice()),
        "existing wrong object must never be overwritten"
    );

    // Concurrent create race: another writer wins with same-size wrong bytes during put_if_absent.
    let race_body = b"two: expected payload!";
    let race_wrong = b"two: tampered payload!";
    assert_eq!(race_body.len(), race_wrong.len());
    let race_sha = sha256_hex(race_body);
    let race_key = blob_key(&race_sha);
    backend.set_next_fault(PutFault::CreateRaceWinner(race_wrong.to_vec()));

    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &race_key,
            race_body,
            &race_sha,
            race_body.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("different bytes"),
        "create race with wrong bytes must fail verification: {err}"
    );
    assert_eq!(
        backend
            .current_bytes(&format!("prefix/{race_key}"))
            .as_deref(),
        Some(race_wrong.as_slice()),
        "race winner bytes must remain untouched"
    );
}

#[test]
fn readback_corruption_missing_object_and_version_mismatch_refuse() {
    let backend = MockVersionedBackend::new(true);
    let store = Store::from_injected_backend(
        "s3://test-versioned-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        backend.clone(),
    );

    let body = b"verified payload bytes";
    let corrupt = b"corrupt! payload bytes";
    let sha = sha256_hex(body);
    let key = blob_key(&sha);

    // 1. PUT reports Created, but readback returns corrupted bytes.
    backend.set_next_fault(PutFault::StoreCorruptBytesOnCreate(corrupt.to_vec()));
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key,
            body,
            &sha,
            body.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("different bytes"),
        "corrupted readback after Created must fail: {err}"
    );

    // 2. PUT reports Created, but object is absent on readback.
    let body2 = b"second payload bytes";
    let sha2 = sha256_hex(body2);
    let key2 = blob_key(&sha2);
    backend.set_next_fault(PutFault::DropOnCreate);
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key2,
            body2,
            &sha2,
            body2.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("absent on verification readback"),
        "missing object after Created must fail: {err}"
    );

    // 3. PUT promises version v-promised, but readback returns v-stored.
    let body3 = b"third payload bytes";
    let sha3 = sha256_hex(body3);
    let key3 = blob_key(&sha3);
    backend.set_next_fault(PutFault::MismatchedReadbackVersion {
        promised: "v-promised".into(),
        stored: "v-stored".into(),
    });
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key3,
            body3,
            &sha3,
            body3.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("absent on verification readback") || err.contains("VersionId mismatch"),
        "mismatched readback version must fail: {err}"
    );
}

#[test]
fn lost_put_response_reconciles_only_on_exact_content_and_version_while_denial_never_reconciles() {
    let backend = MockVersionedBackend::new(true);
    let store = Store::from_injected_backend(
        "s3://test-versioned-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        backend.clone(),
    );

    let body = b"reconcilable payload after lost response";
    let sha = sha256_hex(body);
    let key = blob_key(&sha);

    // 1. Lost response after write with exact bytes + version -> ReconciledAfterAmbiguousPut.
    backend.set_next_fault(PutFault::LostResponseAfterWrite(body.to_vec()));
    let verified = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key,
            body,
            &sha,
            body.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap();
    assert_eq!(
        verified.reference().status,
        PublicationStatus::ReconciledAfterAmbiguousPut
    );
    assert_eq!(
        verified.reference().version_status,
        PhysicalVersionStatus::VerifiedVersionId
    );
    assert!(verified.reference().physical_version_id.is_some());

    // 2. Lost response after write with WRONG bytes -> fails and preserves both errors.
    let body_bad = b"expected bytes for bad lost response";
    let wrong_bad = b"corrupt! bytes for bad lost response";
    let sha_bad = sha256_hex(body_bad);
    let key_bad = blob_key(&sha_bad);
    backend.set_next_fault(PutFault::LostResponseAfterWrite(wrong_bad.to_vec()));
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key_bad,
            body_bad,
            &sha_bad,
            body_bad.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("lost response")
            && err.contains("reconciliation readback failed")
            && err.contains("different bytes"),
        "expected both PUT error and reconciliation failure in message: {err}"
    );

    // 3. Lost response before write (object absent) -> fails and preserves both errors.
    let body_abs = b"absent after transport failure";
    let sha_abs = sha256_hex(body_abs);
    let key_abs = blob_key(&sha_abs);
    backend.set_next_fault(PutFault::LostResponseBeforeWrite);
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key_abs,
            body_abs,
            &sha_abs,
            body_abs.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("before write") && err.contains("absent on verification readback"),
        "expected transport failure + absent readback: {err}"
    );

    // 4. Explicit AccessDenied never reconciles even if the object already exists in the bucket!
    backend.set_next_fault(PutFault::AccessDenied);
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key,
            body,
            &sha,
            body.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("AccessDenied") && !err.contains("reconciliation"),
        "AccessDenied must fail immediately without relabeling existing object as published: {err}"
    );
}

#[test]
fn physical_version_id_binds_exact_version_across_replacements_and_rejects_etag_only_s3() {
    let backend = MockVersionedBackend::new(true);
    let store = Store::from_injected_backend(
        "s3://test-versioned-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        backend.clone(),
    );

    let v1_bytes = b"version 1 immutable bytes";
    let v2_bytes = b"version 2 replacement!!!!";
    assert_eq!(v1_bytes.len(), v2_bytes.len());
    let v1_sha = sha256_hex(v1_bytes);
    let key = blob_key(&v1_sha);

    let published = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key,
            v1_bytes,
            &v1_sha,
            v1_bytes.len() as u64,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap();
    let v1_id = published
        .reference()
        .physical_version_id
        .clone()
        .expect("VersionId must be recorded");
    assert!(published.reference().e_tag.is_some());
    assert_ne!(
        published.reference().physical_version_id,
        published.reference().e_tag
    );

    // Adversarial overwrite replaces the current tip of `key` with v2_bytes under a new VersionId.
    let v2_update = backend
        .put_overwrite(&ObjPath::from(key.as_str()), v2_bytes.to_vec())
        .unwrap();
    let v2_id = v2_update.version.unwrap();
    assert_ne!(v1_id, v2_id);

    // Unpinned verification (latest version) now sees v2_bytes and refuses v1_sha.
    let err = rt()
        .block_on(verify_existing_object_at_version(
            &store,
            &key,
            &v1_sha,
            v1_bytes.len() as u64,
            None,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("different bytes"), "{err}");

    // Version-pinned verification against v2_id also refuses v1_sha.
    let err = rt()
        .block_on(verify_existing_object_at_version(
            &store,
            &key,
            &v1_sha,
            v1_bytes.len() as u64,
            Some(&v2_id),
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("different bytes"), "{err}");

    // Version-pinned verification against the original v1_id still reads and verifies v1_bytes!
    let verified_v1 = rt()
        .block_on(verify_existing_object_at_version(
            &store,
            &key,
            &v1_sha,
            v1_bytes.len() as u64,
            Some(&v1_id),
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap();
    assert_eq!(
        verified_v1.reference().physical_version_id.as_deref(),
        Some(v1_id.as_str())
    );
    assert_eq!(verified_v1.reference().observed_sha256, v1_sha);

    // An S3-classified backend that emits only an ETag and no VersionId is refused even under
    // default options: an ETag is never fabricated into a physical VersionId.
    let etag_only_backend = MockVersionedBackend::new(false);
    let etag_only_store = Store::from_injected_backend(
        "s3://unversioned-s3-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        etag_only_backend,
    );
    let err = rt()
        .block_on(publish_verified_object(
            &etag_only_store,
            &key,
            v1_bytes,
            &v1_sha,
            v1_bytes.len() as u64,
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no physical VersionId") && err.contains("opaque concurrency token"),
        "ETag-only S3 backend must refuse: {err}"
    );
}

#[test]
fn malformed_keys_hash_mismatches_and_readback_size_bounds_refuse() {
    let store = Store::open("memory:///").unwrap();
    let body = b"small payload";
    let sha = sha256_hex(body);
    let key = blob_key(&sha);

    // Content-addressed blob key mismatching expected_sha256 refuses before PUT.
    let other_sha = sha256_hex(b"other payload");
    let err = rt()
        .block_on(publish_verified_object(
            &store,
            &blob_key(&other_sha),
            body,
            &sha,
            body.len() as u64,
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not match expected sha256"), "{err}");

    // Path traversal segment in key refuses.
    let err = rt()
        .block_on(publish_verified_object(
            &store,
            "blobs/../evil",
            body,
            &sha,
            body.len() as u64,
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("traversal"), "{err}");

    // Readback bound exceeded refuses.
    let tight_opts = VerificationOptions::default().with_max_readback_bytes(4);
    let err = rt()
        .block_on(publish_verified_object_with_options(
            &store,
            &key,
            body,
            &sha,
            body.len() as u64,
            tight_opts,
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds max verified readback bound"), "{err}");

    // snapshot_and_verify_source_file refuses when file on disk differs from expected hash.
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("source.bin");
    std::fs::write(&file_path, b"mutated bytes").unwrap();
    let err = snapshot_and_verify_source_file(&file_path, &sha, body.len() as u64)
        .unwrap_err()
        .to_string();
    assert!(err.contains("differs from the manifest"), "{err}");
}
