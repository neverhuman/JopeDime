use super::*;

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
