use super::*;

fn jope_models() -> Command {
    Command::new(env!("CARGO_BIN_EXE_jope-models"))
}

#[test]
fn cli_pinned_path_verify_and_offline() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from");
    let manifest_bytes = synthetic_bundle(&from);
    let manifest: BundleManifest = BundleManifest::parse(&manifest_bytes).unwrap();
    let repo_root = tmp.path().join("repo");
    std::fs::create_dir_all(repo_root.join("models/bundles")).unwrap();
    std::fs::write(
        repo_root.join(format!("models/bundles/{ID}.json")),
        &manifest_bytes,
    )
    .unwrap();
    std::fs::write(
        repo_root.join("models/manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "schema": "jope.model-index/v1",
            "bucket": "test",
            "bundles": [{
                "bundle_id": ID,
                "manifest": format!("models/bundles/{ID}.json"),
                "manifest_sha256": sha256_hex(&manifest_bytes),
                "s3_key": format!("bundles/{ID}/manifest.json"),
                "files": 3,
                "bytes": manifest.total_bytes(),
                "licenses": ["LicenseRef-Test-Permitted", "LicenseRef-Test-Restricted", "none"]
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        repo_root.join("models/PINNED"),
        format!("# pinned\n{ID}\n\n"),
    )
    .unwrap();
    let store_dir = tmp.path().join("store");
    let url = store_url(&store_dir);
    let home = tmp.path().join("home");
    let cache = ModelsCache::at(home.join("models"));

    // verify --manifests-only touches neither the store nor the cache.
    let o = jope_models()
        .args([
            "--repo",
            repo_root.to_str().unwrap(),
            "verify",
            "--manifests-only",
        ])
        .env("JOPE_HOME", &home)
        .env("JOPE_MODELS_URL", "file:///nonexistent-store")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("1 bundles, 1 pinned, 0 errors"));
    assert!(!home.exists());

    let o = jope_models()
        .args([
            "publish",
            tmp.path().join("../nonexistent.json").to_str().unwrap(),
            "--from",
            from.to_str().unwrap(),
        ])
        .env("JOPE_MODELS_URL", &url)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let o = jope_models()
        .args([
            "publish",
            repo_root
                .join(format!("models/bundles/{ID}.json"))
                .to_str()
                .unwrap(),
            "--from",
            from.to_str().unwrap(),
        ])
        .env("JOPE_MODELS_URL", &url)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));

    // fetch --pinned refuses because of the restricted file (exit 2, nothing cached) …
    let o = jope_models()
        .args(["--repo", repo_root.to_str().unwrap(), "fetch", "--pinned"])
        .env("JOPE_HOME", &home)
        .env("JOPE_MODELS_URL", &url)
        .output()
        .unwrap();
    assert_eq!(
        o.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(String::from_utf8_lossy(&o.stderr).contains("prohibited"));
    assert!(!cache.bundle_dir(ID).exists());
    // … and succeeds with --accept-restricted; stdout ends with the bundle directory.
    let o = jope_models()
        .args([
            "--repo",
            repo_root.to_str().unwrap(),
            "fetch",
            "--pinned",
            "--accept-restricted",
        ])
        .env("JOPE_HOME", &home)
        .env("JOPE_MODELS_URL", &url)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(
        stdout.contains("--- notice docs/notices/b.txt ---"),
        "{stdout}"
    );
    assert_eq!(
        stdout.lines().last().unwrap(),
        cache.bundle_dir(ID).to_str().unwrap()
    );

    let o = jope_models()
        .args(["path", ID, ONE])
        .env("JOPE_HOME", &home)
        .output()
        .unwrap();
    assert!(o.status.success());
    let printed = String::from_utf8_lossy(&o.stdout).trim().to_string();
    assert_eq!(Path::new(&printed), cache.file_path(ID, ONE));
    assert!(Path::new(&printed).is_file());
    let o = jope_models()
        .args(["path", "never-fetched"])
        .env("JOPE_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));

    let o = jope_models()
        .args(["verify", "--all"])
        .env("JOPE_HOME", &home)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(
        stdout.contains(&format!("{ID}: 3 files verified, 0 not fetched, 0 errors")),
        "{stdout}"
    );
    assert!(stdout.contains("3 blobs, 0 errors"), "{stdout}");

    // JOPE_OFFLINE=1 with an unreachable store is served from the cache.
    let o = jope_models()
        .args(["fetch", ID, "--accept-restricted"])
        .env("JOPE_HOME", &home)
        .env("JOPE_OFFLINE", "1")
        .env("JOPE_MODELS_URL", "s3://no-such-bucket-for-this-test")
        .current_dir(tmp.path())
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stderr)
        .contains("0 files downloaded (0 bytes), 3 already cached"));
}
