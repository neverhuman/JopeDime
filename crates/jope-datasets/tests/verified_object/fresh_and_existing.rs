use super::*;

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
