use super::*;

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
