use super::*;

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
