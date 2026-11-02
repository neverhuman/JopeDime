use super::*;

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
