use super::*;

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

    // Default options reject an S3 backend with only an ETag; see
    // physical_version_id_binds_exact_version_across_replacements_and_rejects_etag_only_s3.
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
