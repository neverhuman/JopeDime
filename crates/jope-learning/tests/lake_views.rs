use anyhow::{bail, Result};
use jope_learning::lake::{
    reader::{
        read_view, require_complete, ExclusionReason, Outcome, SourceManifest, VersionedBytes,
        VersionedReader,
    },
    roles::{role_matches, Purpose, RoleAuthorizer, RoleFilter, RowRole},
    writer::{
        publish, read_committed, reject_listing_as_authority, HeadObservation, ManifestCommit,
        ProjectionStore, VIEW_SCHEMA,
    },
};
use sha2::{Digest, Sha256};
use std::{cell::Cell, collections::BTreeMap};

const FIXTURE: &[u8] = include_bytes!("fixtures/lake-views-v1/row.bin");
const MANIFEST: &str = include_str!("fixtures/lake-views-v1/manifest.json");

struct Auth {
    purposes: Vec<Purpose>,
}
impl RoleAuthorizer for Auth {
    fn authorize(&self, purpose: Purpose, _row_role: RowRole) -> Result<()> {
        if self.purposes.contains(&purpose) {
            Ok(())
        } else {
            bail!("denied")
        }
    }
}

#[derive(Default)]
struct Reader {
    objects: BTreeMap<(String, String), Vec<u8>>,
    reads: Cell<usize>,
    wrong_version: bool,
}
impl VersionedReader for Reader {
    fn get_versioned(&self, key: &str, version_id: &str) -> Result<Option<VersionedBytes>> {
        self.reads.set(self.reads.get() + 1);
        Ok(self
            .objects
            .get(&(key.into(), version_id.into()))
            .map(|bytes| VersionedBytes {
                version_id: if self.wrong_version {
                    "other-version".into()
                } else {
                    version_id.into()
                },
                bytes: bytes.clone(),
            }))
    }
}

fn fixture_manifest() -> SourceManifest {
    serde_json::from_str(MANIFEST).unwrap()
}
fn populated_reader() -> Reader {
    Reader {
        objects: BTreeMap::from([(
            ("receipts/row.bin".into(), "version-1".into()),
            FIXTURE.to_vec(),
        )]),
        ..Reader::default()
    }
}
fn train_auth() -> Auth {
    Auth {
        purposes: vec![Purpose::Train],
    }
}

#[test]
fn verified_source_manifest_reads_the_exact_version_and_preserves_locators() {
    let manifest = fixture_manifest();
    let expected = manifest.digest().unwrap();
    let view = read_view(
        &manifest,
        &expected,
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &populated_reader(),
    )
    .unwrap();
    assert!(view.complete);
    assert_eq!(
        (view.requested, view.observed, view.excluded.len()),
        (1, 1, 0)
    );
    assert_eq!(view.rows[0].source_version_id, "version-1");
    assert_eq!(view.rows[0].outcome, Outcome::Ok { value: 0.75 });
}

#[test]
fn denied_role_fails_before_materializing_source_bytes() {
    let manifest = fixture_manifest();
    let expected = manifest.digest().unwrap();
    let reader = populated_reader();
    let view = read_view(
        &manifest,
        &expected,
        &Auth { purposes: vec![] },
        RoleFilter {
            purpose: Purpose::Train,
        },
        &reader,
    )
    .unwrap();
    assert_eq!(reader.reads.get(), 0);
    assert_eq!(view.excluded[0].reason, ExclusionReason::Denied);
    assert!(view.rows.is_empty());
}

#[test]
fn role_purpose_matrix_allows_only_matching_rows() {
    let pairs = [
        (Purpose::Train, RowRole::Train),
        (Purpose::Dev, RowRole::Dev),
        (Purpose::Confirm, RowRole::Confirm),
        (Purpose::Sealed, RowRole::Sealed),
    ];
    for purpose in [
        Purpose::Train,
        Purpose::Dev,
        Purpose::Confirm,
        Purpose::Sealed,
    ] {
        for role in [
            RowRole::Train,
            RowRole::Dev,
            RowRole::Confirm,
            RowRole::Sealed,
        ] {
            assert_eq!(
                role_matches(purpose, role),
                pairs.contains(&(purpose, role))
            );
        }
    }
}

#[test]
fn unsafe_keys_and_duplicate_object_versions_are_rejected() {
    let mut manifest = fixture_manifest();
    manifest.source_objects[0].key = "../outside.bin".into();
    assert!(manifest.validate().is_err());

    let mut manifest = fixture_manifest();
    manifest
        .source_objects
        .push(manifest.source_objects[0].clone());
    assert!(manifest.validate().is_err());
}

#[test]
fn failed_and_unknown_observations_are_preserved() {
    let failed = br#"{"failed":{"reason":"runner_exit"}}"#;
    let unknown = br#""unknown""#;
    let mut manifest = fixture_manifest();
    manifest.source_objects[0].sha256 = hex::encode(Sha256::digest(failed));
    let mut second = manifest.source_objects[0].clone();
    second.key = "receipts/unknown.bin".into();
    second.version_id = "version-unknown".into();
    second.sha256 = hex::encode(Sha256::digest(unknown));
    manifest.source_objects.push(second);
    let reader = Reader {
        objects: BTreeMap::from([
            (
                ("receipts/row.bin".into(), "version-1".into()),
                failed.to_vec(),
            ),
            (
                ("receipts/unknown.bin".into(), "version-unknown".into()),
                unknown.to_vec(),
            ),
        ]),
        ..Reader::default()
    };
    let expected = manifest.digest().unwrap();
    let view = read_view(
        &manifest,
        &expected,
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &reader,
    )
    .unwrap();
    assert_eq!(view.rows.len(), 2);
    assert_eq!(
        view.rows[0].outcome,
        Outcome::Failed {
            reason: "runner_exit".into()
        }
    );
    assert_eq!(view.rows[1].outcome, Outcome::Unknown);
}

#[test]
fn forged_manifest_role_and_unpinned_manifest_fail_closed() {
    let forged = MANIFEST.replace("\"train\"", "\"judge\"");
    assert!(serde_json::from_str::<SourceManifest>(&forged).is_err());
    let manifest = fixture_manifest();
    assert!(read_view(
        &manifest,
        &"0".repeat(64),
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train
        },
        &populated_reader()
    )
    .is_err());
    let forged_valid_role: SourceManifest =
        serde_json::from_str(&MANIFEST.replace("\"train\"", "\"sealed\"")).unwrap();
    let reader = populated_reader();
    assert!(read_view(
        &forged_valid_role,
        &manifest.digest().unwrap(),
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &reader,
    )
    .is_err());
    assert_eq!(reader.reads.get(), 0);
}

#[test]
fn missing_corrupt_and_wrong_version_objects_are_excluded_with_denominators() {
    let manifest = fixture_manifest();
    let expected = manifest.digest().unwrap();
    let missing = read_view(
        &manifest,
        &expected,
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &Reader::default(),
    )
    .unwrap();
    assert_eq!(missing.excluded[0].reason, ExclusionReason::Missing);
    assert_eq!(
        (missing.requested, missing.observed, missing.excluded.len()),
        (1, 0, 1)
    );
    assert!(!missing.complete);
    assert!(require_complete(&missing).is_err());

    let bad = Reader {
        objects: BTreeMap::from([(
            ("receipts/row.bin".into(), "version-1".into()),
            b"tampered".to_vec(),
        )]),
        ..Reader::default()
    };
    let corrupt = read_view(
        &manifest,
        &expected,
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &bad,
    )
    .unwrap();
    assert_eq!(corrupt.excluded[0].reason, ExclusionReason::Corrupt);

    let mut wrong = populated_reader();
    wrong.wrong_version = true;
    let mismatch = read_view(
        &manifest,
        &expected,
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &wrong,
    )
    .unwrap();
    assert_eq!(
        mismatch.excluded[0].reason,
        ExclusionReason::VersionMismatch
    );

    let mut malformed_manifest = fixture_manifest();
    let malformed_bytes = b"not-json";
    malformed_manifest.source_objects[0].sha256 = hex::encode(Sha256::digest(malformed_bytes));
    let malformed_reader = Reader {
        objects: BTreeMap::from([(
            ("receipts/row.bin".into(), "version-1".into()),
            malformed_bytes.to_vec(),
        )]),
        ..Reader::default()
    };
    let malformed = read_view(
        &malformed_manifest,
        &malformed_manifest.digest().unwrap(),
        &train_auth(),
        RoleFilter {
            purpose: Purpose::Train,
        },
        &malformed_reader,
    )
    .unwrap();
    assert_eq!(malformed.excluded[0].reason, ExclusionReason::Corrupt);
}

#[derive(Default)]
struct FakeStore {
    head: std::cell::RefCell<Option<TestHead>>,
    objects: std::cell::RefCell<BTreeMap<String, Vec<u8>>>,
    race: bool,
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct TestHead {
    version_id: String,
    bytes: Vec<u8>,
}
impl HeadObservation for TestHead {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
impl ProjectionStore for FakeStore {
    type Head = TestHead;
    fn get_head(&self) -> Result<Option<Self::Head>> {
        Ok(self.head.borrow().clone())
    }
    fn put_immutable_verified(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let existing = self.objects.borrow().get(key).cloned();
        if let Some(existing) = existing {
            if existing != bytes {
                bail!("immutable object conflict");
            }
        } else {
            self.objects.borrow_mut().insert(key.into(), bytes.into());
        }
        Ok(())
    }
    fn get_immutable(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.objects.borrow().get(key).cloned())
    }
    fn put_head_if_unchanged(&self, observed: Option<&Self::Head>, bytes: &[u8]) -> Result<()> {
        if self.race {
            *self.head.borrow_mut() = Some(TestHead {
                version_id: "racer".into(),
                bytes: b"raced".to_vec(),
            });
        }
        if self.head.borrow().as_ref() != observed {
            bail!("CAS lost");
        }
        *self.head.borrow_mut() = Some(TestHead {
            version_id: "next".into(),
            bytes: bytes.into(),
        });
        Ok(())
    }
}

fn commit(view_bytes: &[u8]) -> ManifestCommit {
    ManifestCommit {
        schema: VIEW_SCHEMA.into(),
        generation: 1,
        parent_sha256: None,
        source_manifest_sha256: "a".repeat(64),
        view_sha256: hex::encode(Sha256::digest(view_bytes)),
        requested: 1,
        observed: 1,
        excluded: 0,
    }
}

#[test]
fn publication_writes_immutable_objects_before_observed_head_cas() {
    let store = FakeStore::default();
    let view = b"{}";
    let result = publish(&store, &commit(view), view).unwrap();
    assert!(store.objects.borrow().contains_key(&result.manifest_key));
    assert!(store.head.borrow().is_some());
    let (stored_commit, stored_view) = read_committed(&store).unwrap().unwrap();
    assert_eq!(stored_commit.digest().unwrap(), result.commit_sha256);
    assert_eq!(stored_view, view);
    let next = ManifestCommit {
        generation: 2,
        parent_sha256: Some(stored_commit.digest().unwrap()),
        view_sha256: hex::encode(Sha256::digest(b"next-view")),
        requested: 1,
        observed: 1,
        excluded: 0,
        ..stored_commit
    };
    publish(&store, &next, b"next-view").unwrap();
    assert_eq!(read_committed(&store).unwrap().unwrap().1, b"next-view");
}

#[test]
fn stale_listing_is_not_an_authority_and_head_race_does_not_commit() {
    assert!(reject_listing_as_authority().is_err());
    let store = FakeStore {
        race: true,
        ..FakeStore::default()
    };
    assert!(publish(&store, &commit(b"{}"), b"{}").is_err());
    assert_eq!(store.head.borrow().as_ref().unwrap().version_id, "racer");
}

#[test]
fn commit_rejects_unaccounted_rows() {
    let mut value = commit(b"{}");
    value.excluded = 1;
    assert!(value.validate().is_err());
}
