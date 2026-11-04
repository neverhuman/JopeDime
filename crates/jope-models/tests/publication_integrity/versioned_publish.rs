use super::*;

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
