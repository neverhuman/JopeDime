//! Model bundle publication integrity tests (`I46`).
//!
//! Proves:
//! 1. Pre-existing same-size wrong blob at `blobs/sha256/<sha>` is refused by `publish` and by the
//!    `jope-models publish` CLI (fixing the size-only check bug), leaving the corrupt blob untouched
//!    and writing no bundle manifest.
//! 2. Concurrent create race (`AlreadyExists`) with wrong content during `publish` is refused and
//!    blocks manifest publication.
//! 3. Source file mutation between initial prehash and upload snapshot (`on_before_snapshot`) is
//!    refused before upload or manifest publication; source file mutation after the verified
//!    snapshot is taken (`on_after_snapshot`) does not alter the uploaded bytes.
//! 4. Multi-file bundle dependency failure after blob 1 succeeds blocks `manifest.json` from being
//!    published (`manifest-last` atomic boundary).
//! 5. Versioned backend publication records `VerifiedVersionId` on every blob and manifest in
//!    `PublishReport`, and `--receipt-out` / `--require-version-id` CLI flags enforce and emit the
//!    verified receipt create-only.

use anyhow::{anyhow, Result};
use jope_datasets::blobs::blob_key;
use jope_datasets::hash::sha256_hex;
use jope_datasets::store::{
    InjectedBackend, ObjPath, ObjectStoreScheme, PutOutcome, Store, UpdateVersion,
};
use jope_datasets::verified_object::{
    PhysicalVersionStatus, PublicationStatus, VerificationOptions, VERIFIED_OBJECT_CONTRACT_ID,
};
use jope_models::publish::{
    publish, publish_with_options, publish_with_options_and_hooks, PublishHooks, PublishReport,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

const FIXTURE_JSON: &str = include_str!("fixtures/publication-integrity-v1/fixture_manifest.json");
const ONE_PATH: &str = "weights/one.safetensors";
const TWO_PATH: &str = "two.json";
const NOTICE_A: &str = "docs/notices/a.txt";

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn fixture_values() -> (String, Vec<u8>, String, Vec<u8>, String, Vec<u8>, String) {
    let v: Value = serde_json::from_str(FIXTURE_JSON).unwrap();
    let bundle_id = v["bundle_id"].as_str().unwrap().to_owned();
    let one_bytes = v["weights_one_utf8"].as_str().unwrap().as_bytes().to_vec();
    let one_sha = v["weights_one_sha256"].as_str().unwrap().to_owned();
    let corrupt_bytes = v["corrupt_same_size_utf8"]
        .as_str()
        .unwrap()
        .as_bytes()
        .to_vec();
    let corrupt_sha = v["corrupt_same_size_sha256"].as_str().unwrap().to_owned();
    let two_bytes = v["metadata_two_utf8"].as_str().unwrap().as_bytes().to_vec();
    let two_sha = sha256_hex(&two_bytes);
    assert_eq!(sha256_hex(&one_bytes), one_sha);
    assert_eq!(sha256_hex(&corrupt_bytes), corrupt_sha);
    assert_eq!(one_bytes.len(), corrupt_bytes.len());
    (
        bundle_id,
        one_bytes,
        one_sha,
        corrupt_bytes,
        corrupt_sha,
        two_bytes,
        two_sha,
    )
}

fn write_bundle_fixture(from: &Path, manifest_path: &Path) -> Vec<u8> {
    let (bundle_id, one_bytes, one_sha, _, _, two_bytes, two_sha) = fixture_values();
    for (rel, content) in [(ONE_PATH, &one_bytes), (TWO_PATH, &two_bytes)] {
        let p = from.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, content).unwrap();
    }
    let manifest = json!({
        "schema": "jope.model-bundle/v2",
        "bundle_id": bundle_id,
        "bucket": "test",
        "published_at": "2026-10-18T00:00:00Z",
        "files": [
            {
                "path": ONE_PATH,
                "sha256": one_sha,
                "bytes": one_bytes.len(),
                "blob": blob_key(&one_sha),
                "kind": "model",
                "license": {
                    "schema": "jain.model-license/v1",
                    "spdx_expression": "LicenseRef-Test-Permitted",
                    "license_class": "internal",
                    "redistribution": "permitted",
                    "notice_paths": [NOTICE_A]
                }
            },
            {
                "path": TWO_PATH,
                "sha256": two_sha,
                "bytes": two_bytes.len(),
                "blob": blob_key(&two_sha),
                "kind": "metadata",
                "license": null
            }
        ],
        "notices": [
            {
                "path": NOTICE_A,
                "text": "Notice A: permitted internal model\n"
            }
        ]
    });
    let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    std::fs::write(manifest_path, &bytes).unwrap();
    bytes
}

type StoredVersionedObject = (Vec<u8>, String, String);

struct VersionedPublishBackend {
    objects: Mutex<BTreeMap<String, StoredVersionedObject>>,
    race_corrupt_key: Mutex<Option<(String, Vec<u8>)>>,
    deny_key: Mutex<Option<String>>,
    seq: Mutex<u64>,
}

impl VersionedPublishBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            objects: Mutex::new(BTreeMap::new()),
            race_corrupt_key: Mutex::new(None),
            deny_key: Mutex::new(None),
            seq: Mutex::new(500),
        })
    }

    fn next_tokens(&self, len: usize) -> (String, String) {
        let mut s = self.seq.lock().unwrap();
        *s += 1;
        (format!("ver-{}", *s), format!("\"etag-{}-{len}\"", *s))
    }
}

impl InjectedBackend for VersionedPublishBackend {
    fn get_versioned(
        &self,
        key: &ObjPath,
        required_version: Option<&str>,
    ) -> Result<Option<(Vec<u8>, UpdateVersion)>> {
        let map = self.objects.lock().unwrap();
        let Some((bytes, ver, etag)) = map.get(key.as_ref()) else {
            return Ok(None);
        };
        if let Some(req) = required_version {
            if ver != req {
                return Ok(None);
            }
        }
        Ok(Some((
            bytes.clone(),
            UpdateVersion {
                e_tag: Some(etag.clone()),
                version: Some(ver.clone()),
            },
        )))
    }

    fn put_create(
        &self,
        key: &ObjPath,
        bytes: Vec<u8>,
    ) -> Result<(PutOutcome, Option<UpdateVersion>)> {
        if let Some(denied) = self.deny_key.lock().unwrap().as_deref() {
            if key.as_ref() == denied {
                return Err(anyhow!(
                    "put {}: AccessDenied: simulated permission refusal",
                    key.as_ref()
                ));
            }
        }
        let matched_race = {
            let mut guard = self.race_corrupt_key.lock().unwrap();
            if guard
                .as_ref()
                .is_some_and(|(race_key, _)| key.as_ref() == race_key)
            {
                guard.take()
            } else {
                None
            }
        };
        if let Some((race_key, race_bytes)) = matched_race {
            let (ver, etag) = self.next_tokens(race_bytes.len());
            self.objects
                .lock()
                .unwrap()
                .insert(race_key, (race_bytes, ver, etag));
            return Ok((PutOutcome::Exists, None));
        }
        let mut map = self.objects.lock().unwrap();
        if map.contains_key(key.as_ref()) {
            return Ok((PutOutcome::Exists, None));
        }
        let (ver, etag) = self.next_tokens(bytes.len());
        map.insert(key.as_ref().to_owned(), (bytes, ver.clone(), etag.clone()));
        Ok((
            PutOutcome::Created,
            Some(UpdateVersion {
                e_tag: Some(etag),
                version: Some(ver),
            }),
        ))
    }

    fn put_overwrite(&self, key: &ObjPath, bytes: Vec<u8>) -> Result<UpdateVersion> {
        let (ver, etag) = self.next_tokens(bytes.len());
        self.objects
            .lock()
            .unwrap()
            .insert(key.as_ref().to_owned(), (bytes, ver.clone(), etag.clone()));
        Ok(UpdateVersion {
            e_tag: Some(etag),
            version: Some(ver),
        })
    }

    fn list_keys(&self, prefix: &ObjPath) -> Result<Vec<ObjPath>> {
        let map = self.objects.lock().unwrap();
        let p = prefix.as_ref();
        Ok(map
            .keys()
            .filter(|k| p.is_empty() || k.starts_with(p))
            .map(|k| ObjPath::from(k.as_str()))
            .collect())
    }
}

#[test]
fn preexisting_same_size_wrong_blob_refuses_publish_and_leaves_corrupt_object_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_path = tmp.path().join("bundle.json");
    write_bundle_fixture(&from, &manifest_path);

    let (bundle_id, _, one_sha, corrupt_bytes, _, _, _) = fixture_values();
    let store_dir = tmp.path().join("store");
    let pre_blob_path = store_dir.join(blob_key(&one_sha));
    std::fs::create_dir_all(pre_blob_path.parent().unwrap()).unwrap();
    std::fs::write(&pre_blob_path, &corrupt_bytes).unwrap();

    let store_url = format!("file://{}", store_dir.display());
    let store = Store::open(&store_url).unwrap();

    // 1. Library publish refuses same-size wrong existing blob before writing manifest.
    let err = rt()
        .block_on(publish(&store, &manifest_path, &from))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("different bytes") && err.contains(&one_sha),
        "publish must refuse same-size wrong blob: {err}"
    );
    assert_eq!(
        std::fs::read(&pre_blob_path).unwrap(),
        corrupt_bytes,
        "corrupt preexisting blob must never be overwritten"
    );
    assert!(
        !store_dir
            .join(format!("bundles/{bundle_id}/manifest.json"))
            .exists(),
        "manifest must not be published when a blob verification fails"
    );

    // 2. CLI `jope-models publish` also exits non-zero (code 2) and writes no manifest.
    let out = Command::new(env!("CARGO_BIN_EXE_jope-models"))
        .args([
            "publish",
            manifest_path.to_str().unwrap(),
            "--from",
            from.to_str().unwrap(),
            "--store",
            &store_url,
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("different bytes"), "{stderr}");
    assert!(!store_dir
        .join(format!("bundles/{bundle_id}/manifest.json"))
        .exists());
}

#[test]
fn create_race_and_second_blob_failure_block_manifest_publication() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_path = tmp.path().join("bundle.json");
    write_bundle_fixture(&from, &manifest_path);

    let (bundle_id, _, one_sha, _, _, two_bytes, two_sha) = fixture_values();
    let backend = VersionedPublishBackend::new();
    let store = Store::from_injected_backend(
        "s3://test-models-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        backend.clone(),
    );

    // Case A: Blob 1 succeeds (Created), Blob 2 loses a create race to wrong bytes of the same
    // length (19 bytes). Manifest must NOT be written!
    let wrong_two = b"{\"kind\":\"corrupt!\"}".to_vec();
    assert_eq!(wrong_two.len(), two_bytes.len());
    *backend.race_corrupt_key.lock().unwrap() = Some((blob_key(&two_sha), wrong_two.clone()));

    let err = rt()
        .block_on(publish_with_options(
            &store,
            &manifest_path,
            &from,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("different bytes"), "{err}");

    let keys = rt().block_on(store.list("")).unwrap();
    assert!(keys.contains(&blob_key(&one_sha)));
    assert!(
        !keys.contains(&format!("bundles/{bundle_id}/manifest.json")),
        "manifest must never be published when a dependency blob fails verification"
    );

    // Case B: Denied write on manifest also blocks completion and surfaces AccessDenied.
    let backend2 = VersionedPublishBackend::new();
    *backend2.deny_key.lock().unwrap() = Some(format!("bundles/{bundle_id}/manifest.json"));
    let store2 = Store::from_injected_backend(
        "s3://test-models-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        backend2,
    );
    let err = rt()
        .block_on(publish_with_options(
            &store2,
            &manifest_path,
            &from,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap_err()
        .to_string();
    assert!(err.contains("AccessDenied"), "{err}");
}

#[test]
fn source_mutation_before_snapshot_refuses_and_mutation_after_snapshot_preserves_verified_upload() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_path = tmp.path().join("bundle.json");
    write_bundle_fixture(&from, &manifest_path);

    let (bundle_id, one_bytes, one_sha, corrupt_bytes, _, _, _) = fixture_values();
    let store_dir = tmp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();

    // 1. Mutating the file AFTER initial prehash but BEFORE snapshot read (`on_before_snapshot`)
    //    is caught by `snapshot_and_verify_source_file` and aborts before uploading or publishing
    //    the manifest.
    let corrupt_for_hook = corrupt_bytes.clone();
    let mut mutate_before_snapshot = |rel: &str, src: &Path| {
        if rel == ONE_PATH {
            std::fs::write(src, &corrupt_for_hook).unwrap();
        }
    };
    let mut hooks = PublishHooks {
        on_before_snapshot: Some(&mut mutate_before_snapshot),
        on_after_snapshot: None,
    };
    let err = rt()
        .block_on(publish_with_options_and_hooks(
            &store,
            &manifest_path,
            &from,
            VerificationOptions::default(),
            &mut hooks,
        ))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("differs from the manifest"),
        "mutation between prehash and snapshot read must fail: {err}"
    );
    assert!(
        !store_dir.join(blob_key(&one_sha)).exists(),
        "mutated pre-snapshot file must not be uploaded"
    );
    assert!(
        !store_dir
            .join(format!("bundles/{bundle_id}/manifest.json"))
            .exists(),
        "manifest must not be written when source mutates before snapshot"
    );

    // Restore clean fixture and test mutation AFTER snapshot (`on_after_snapshot`).
    write_bundle_fixture(&from, &manifest_path);
    let corrupt_after = corrupt_bytes.clone();
    let mut mutate_after_snapshot = |rel: &str, src: &Path| {
        if rel == ONE_PATH {
            std::fs::write(src, &corrupt_after).unwrap();
        }
    };
    let mut hooks2 = PublishHooks {
        on_before_snapshot: None,
        on_after_snapshot: Some(&mut mutate_after_snapshot),
    };
    let report = rt()
        .block_on(publish_with_options_and_hooks(
            &store,
            &manifest_path,
            &from,
            VerificationOptions::default(),
            &mut hooks2,
        ))
        .unwrap();
    assert_eq!(report.blobs_created, 2);
    assert_eq!(
        std::fs::read(store_dir.join(blob_key(&one_sha))).unwrap(),
        one_bytes,
        "uploaded blob must match the frozen verified snapshot, not the post-snapshot disk mutation"
    );
}

#[test]
fn versioned_publish_report_and_cli_receipt_out_bind_content_and_versions() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_path = tmp.path().join("bundle.json");
    write_bundle_fixture(&from, &manifest_path);

    // 1. Injected versioned S3 backend records VerifiedVersionId for all blobs and manifest.
    let backend = VersionedPublishBackend::new();
    let versioned_store = Store::from_injected_backend(
        "s3://test-models-bucket",
        ObjPath::default(),
        ObjectStoreScheme::AmazonS3,
        backend,
    );
    let rep = rt()
        .block_on(publish_with_options(
            &versioned_store,
            &manifest_path,
            &from,
            VerificationOptions::require_physical_version_id(),
        ))
        .unwrap();
    assert_eq!(rep.blobs_created, 2);
    assert_eq!(rep.verified_blobs.len(), 2);
    for vb in &rep.verified_blobs {
        assert_eq!(vb.schema, VERIFIED_OBJECT_CONTRACT_ID);
        assert_eq!(vb.version_status, PhysicalVersionStatus::VerifiedVersionId);
        assert!(vb.physical_version_id.is_some());
        assert_eq!(vb.expected_sha256, vb.observed_sha256);
        assert!(!vb.live_s3_qualified);
    }
    let vm = rep.verified_manifest.as_ref().unwrap();
    assert_eq!(vm.version_status, PhysicalVersionStatus::VerifiedVersionId);
    assert_eq!(vm.status, PublicationStatus::Created);

    // 2. CLI `--receipt-out` writes create-only JSON receipt and refuses to overwrite an existing
    //    receipt file; `--require-version-id` refuses unversioned `file://` stores.
    let store_dir = tmp.path().join("file-store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let file_url = format!("file://{}", store_dir.display());
    let receipt_path = tmp.path().join("receipt.json");

    let out = Command::new(env!("CARGO_BIN_EXE_jope-models"))
        .args([
            "publish",
            manifest_path.to_str().unwrap(),
            "--from",
            from.to_str().unwrap(),
            "--store",
            &file_url,
            "--receipt-out",
            receipt_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let written: PublishReport =
        serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
    assert_eq!(written.verified_blobs.len(), 2);
    assert_eq!(
        written.verified_blobs[0].version_status,
        PhysicalVersionStatus::UnversionedLocalFixture
    );

    // Re-running with the same `--receipt-out` path refuses because `--receipt-out` is create-only.
    let out_dup = Command::new(env!("CARGO_BIN_EXE_jope-models"))
        .args([
            "publish",
            manifest_path.to_str().unwrap(),
            "--from",
            from.to_str().unwrap(),
            "--store",
            &file_url,
            "--receipt-out",
            receipt_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out_dup.status.code(), Some(2));

    // `--require-version-id` against an unversioned `file://` store refuses truthfully.
    let out_req = Command::new(env!("CARGO_BIN_EXE_jope-models"))
        .args([
            "publish",
            manifest_path.to_str().unwrap(),
            "--from",
            from.to_str().unwrap(),
            "--store",
            &file_url,
            "--require-version-id",
        ])
        .output()
        .unwrap();
    assert_eq!(out_req.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out_req.stderr).contains("no physical VersionId"),
        "{}",
        String::from_utf8_lossy(&out_req.stderr)
    );
}
