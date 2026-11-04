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

#[path = "verified_object/fresh_and_existing.rs"]
mod fresh_and_existing;
#[path = "verified_object/lost_put.rs"]
mod lost_put;
#[path = "verified_object/malformed.rs"]
mod malformed;
#[path = "verified_object/readback.rs"]
mod readback;
#[path = "verified_object/versions.rs"]
mod versions;
#[path = "verified_object/wrong_content.rs"]
mod wrong_content;
