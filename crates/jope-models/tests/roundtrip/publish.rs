use super::*;

#[test]
fn publish_fetch_verify_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_bytes = synthetic_bundle(&from);
    let manifest_path = tmp.path().join("bundle.json");
    std::fs::write(&manifest_path, &manifest_bytes).unwrap();
    let url = store_url(&tmp.path().join("store"));
    let store = Store::open(&url).unwrap();

    let r = rt()
        .block_on(publish(&store, &manifest_path, &from))
        .unwrap();
    assert_eq!(
        (r.blobs_created, r.blobs_existing, r.manifest.as_str()),
        (3, 0, "created")
    );
    assert_eq!(r.verified_blobs.len(), 3);
    assert!(r.verified_manifest.is_some());
    let keys = rt().block_on(store.list("")).unwrap();
    assert_eq!(
        keys.iter()
            .filter(|k| k.starts_with("blobs/sha256/"))
            .count(),
        3
    );
    assert!(keys.contains(&format!("bundles/{ID}/manifest.json")));
    // Publishing again rewrites nothing and verifies all existing blobs by content.
    let r = rt()
        .block_on(publish(&store, &manifest_path, &from))
        .unwrap();
    assert_eq!(
        (r.blobs_created, r.blobs_existing, r.manifest.as_str()),
        (0, 3, "identical")
    );
    assert_eq!(r.verified_blobs.len(), 3);
    assert!(r.verified_manifest.is_some());
    // A manifest that differs under the same id is refused.
    let mut altered: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    altered["published_at"] = Value::String("2026-09-19T00:00:00Z".into());
    let altered_path = tmp.path().join("altered.json");
    std::fs::write(&altered_path, serde_json::to_vec_pretty(&altered).unwrap()).unwrap();
    let err = rt()
        .block_on(publish(&store, &altered_path, &from))
        .unwrap_err()
        .to_string();
    assert!(err.contains("immutable"), "{err}");
    // A file that does not match its manifest entry stops the publish before any upload.
    let bad_from = tmp.path().join("bad-from");
    synthetic_bundle(&bad_from);
    std::fs::write(bad_from.join(TWO), b"{}").unwrap();
    let err = rt()
        .block_on(publish(&store, &manifest_path, &bad_from))
        .unwrap_err()
        .to_string();
    assert!(err.contains("differs from the manifest"), "{err}");

    let cache = ModelsCache::at(tmp.path().join("cache"));
    let fetcher = Fetcher {
        cache: &cache,
        store: Some(&store),
        repo: None,
    };
    // Without --accept-restricted the prohibited file refuses the whole fetch before downloading.
    let mut out = Vec::new();
    let err = rt()
        .block_on(fetcher.fetch(ID, &FetchOptions::default(), &mut out))
        .unwrap_err()
        .to_string();
    assert!(err.contains("prohibited") && err.contains(THREE), "{err}");
    assert!(!cache.bundle_dir(ID).exists());
    assert!(cache.blobs().list().is_empty());
    let printed = String::from_utf8(out).unwrap();
    assert!(
        printed.contains("Notice A: permitted") && printed.contains("Notice B: fail-closed"),
        "{printed}"
    );
    assert!(
        printed.contains("LicenseRef-Test-Restricted / prohibited  [restricted]"),
        "{printed}"
    );
    assert!(
        printed.contains("LicenseRef-Test-Permitted / permitted"),
        "{printed}"
    );
    assert!(
        printed.contains("two.json  (no licence block: metadata)"),
        "{printed}"
    );

    // Selecting the permitted components needs no acceptance.
    let opts = FetchOptions {
        components: vec![ONE.into(), TWO.into()],
        accept_restricted: false,
    };
    let r = rt()
        .block_on(fetcher.fetch(ID, &opts, &mut Vec::new()))
        .unwrap();
    assert_eq!(
        (
            r.files_selected,
            r.files_fetched,
            r.files_cached,
            r.errors.len()
        ),
        (2, 2, 0, 0),
        "{r:?}"
    );
    assert!(cache.file_path(ID, ONE).is_file());
    assert!(!cache.file_path(ID, THREE).exists());
    assert!(cache.manifest_path(ID).is_file());
    let notice = std::fs::read_to_string(cache.file_path(ID, NOTICE_A)).unwrap();
    assert_eq!(notice, "Notice A: permitted internal model\n");
    let v = cache.verify(ID, false).unwrap();
    assert_eq!(
        (v.files_checked, v.files_absent, v.errors.len()),
        (2, 1, 0),
        "{v:?}"
    );

    // Accepting the restriction fetches the rest; cached files are not re-downloaded.
    let opts = FetchOptions {
        components: vec![],
        accept_restricted: true,
    };
    let r = rt()
        .block_on(fetcher.fetch(ID, &opts, &mut Vec::new()))
        .unwrap();
    assert_eq!(
        (r.files_fetched, r.files_cached, r.errors.len()),
        (1, 2, 0),
        "{r:?}"
    );
    assert_eq!(r.restricted_accepted, vec![THREE.to_string()]);
    let linked = cache.file_path(ID, THREE);
    let blob = cache
        .blobs()
        .path(&sha256_hex(b"three: provenance pending"));
    assert_eq!(
        std::fs::metadata(&linked).unwrap().ino(),
        std::fs::metadata(&blob).unwrap().ino()
    );
    let v = cache.verify(ID, false).unwrap();
    assert_eq!(
        (v.files_checked, v.files_absent, v.errors.len()),
        (3, 0, 0),
        "{v:?}"
    );
    assert_eq!(cache.list(), vec![ID.to_string()]);

    // Tampering with a materialised file is caught by verify (the link is replaced so the
    // blob itself stays intact and is checked separately).
    std::fs::remove_file(&linked).unwrap();
    std::fs::write(&linked, b"tampered").unwrap();
    let v = cache.verify(ID, false).unwrap();
    assert_eq!(v.errors.len(), 1, "{v:?}");
    assert!(v.errors[0].starts_with(THREE), "{v:?}");
    let v = cache.verify(ID, true).unwrap();
    assert_eq!(
        v.errors.len(),
        1,
        "quick verify sees the size change: {v:?}"
    );
}

#[test]
fn corrupted_blob_is_deleted_and_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_bytes = synthetic_bundle(&from);
    let manifest_path = tmp.path().join("bundle.json");
    std::fs::write(&manifest_path, &manifest_bytes).unwrap();
    let store_dir = tmp.path().join("store");
    let url = store_url(&store_dir);
    let store = Store::open(&url).unwrap();
    rt().block_on(publish(&store, &manifest_path, &from))
        .unwrap();
    // Corrupt the permitted blob in the store (same length, different bytes).
    let sha = sha256_hex(b"one: permitted weights");
    std::fs::write(store_dir.join(blob_key(&sha)), b"one: corrupted weights").unwrap();

    let cache = ModelsCache::at(tmp.path().join("cache"));
    let fetcher = Fetcher {
        cache: &cache,
        store: Some(&store),
        repo: None,
    };
    let opts = FetchOptions {
        components: vec![],
        accept_restricted: true,
    };
    let r = rt()
        .block_on(fetcher.fetch(ID, &opts, &mut Vec::new()))
        .unwrap();
    assert_eq!(r.errors.len(), 1, "{r:?}");
    assert!(
        r.errors[0].starts_with(ONE) && r.errors[0].contains("sha256 mismatch"),
        "{r:?}"
    );
    assert!(!cache.blobs().path(&sha).exists());
    assert!(!cache.blobs().partial_path(&sha).exists());
    let partials: Vec<_> = std::fs::read_dir(cache.blobs().dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".partial"))
        .collect();
    assert!(partials.is_empty());
    // Nothing is exposed: no bundle directory, no manifest.
    assert!(!cache.bundle_dir(ID).exists());
    assert!(!cache.manifest_path(ID).exists());
}

#[test]
fn offline_serves_only_the_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_bytes = synthetic_bundle(&from);
    let manifest_path = tmp.path().join("bundle.json");
    std::fs::write(&manifest_path, &manifest_bytes).unwrap();
    let url = store_url(&tmp.path().join("store"));
    let store = Store::open(&url).unwrap();
    rt().block_on(publish(&store, &manifest_path, &from))
        .unwrap();
    let cache = ModelsCache::at(tmp.path().join("cache"));
    let opts = FetchOptions {
        components: vec![],
        accept_restricted: true,
    };

    let offline = Fetcher {
        cache: &cache,
        store: None,
        repo: None,
    };
    let err = rt()
        .block_on(offline.fetch(ID, &opts, &mut Vec::new()))
        .unwrap_err()
        .to_string();
    assert!(err.contains("offline"), "{err}");

    let online = Fetcher {
        cache: &cache,
        store: Some(&store),
        repo: None,
    };
    rt().block_on(online.fetch(ID, &opts, &mut Vec::new()))
        .unwrap();
    let r = rt()
        .block_on(offline.fetch(ID, &opts, &mut Vec::new()))
        .unwrap();
    assert_eq!(
        (r.files_fetched, r.files_cached, r.errors.len()),
        (0, 3, 0),
        "{r:?}"
    );
    // A committed manifest is enough to name the files, but blobs still need the store.
    let repo_root = tmp.path().join("repo");
    std::fs::create_dir_all(repo_root.join("models/bundles")).unwrap();
    std::fs::write(
        repo_root.join("models/manifest.json"),
        b"{\"schema\":\"jope.model-index/v1\",\"bucket\":\"test\",\"bundles\":[]}",
    )
    .unwrap();
    std::fs::write(
        repo_root.join(format!("models/bundles/{ID}.json")),
        &manifest_bytes,
    )
    .unwrap();
    let repo = Repo::at(&repo_root);
    let fresh = ModelsCache::at(tmp.path().join("fresh"));
    let offline = Fetcher {
        cache: &fresh,
        store: None,
        repo: Some(&repo),
    };
    let r = rt()
        .block_on(offline.fetch(ID, &opts, &mut Vec::new()))
        .unwrap();
    assert_eq!(r.errors.len(), 3, "{r:?}");
    assert!(r.errors.iter().all(|e| e.contains("offline")), "{r:?}");
    assert!(!fresh.bundle_dir(ID).exists());
}
