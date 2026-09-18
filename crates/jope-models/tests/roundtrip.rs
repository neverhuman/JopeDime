//! publish → fetch → verify on a `file://` store with a synthetic three-file bundle (one file
//! `redistribution: prohibited`), a corrupted blob, `--offline` and `--pinned`.

use jope_datasets::blobs::blob_key;
use jope_datasets::hash::sha256_hex;
use jope_datasets::store::Store;
use jope_models::cache::ModelsCache;
use jope_models::fetch::{FetchOptions, Fetcher};
use jope_models::index::Repo;
use jope_models::manifest::BundleManifest;
use jope_models::publish::publish;
use serde_json::{json, Value};
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::process::Command;

const ID: &str = "synthetic-test-v1";
const ONE: &str = "weights/one.safetensors";
const TWO: &str = "two.json";
const THREE: &str = "restricted/three.safetensors";
const NOTICE_A: &str = "docs/notices/a.txt";
const NOTICE_B: &str = "docs/notices/b.txt";

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn license(spdx: &str, class: &str, redistribution: &str, notice: &str) -> Value {
    json!({
        "schema": "jain.model-license/v1",
        "spdx_expression": spdx,
        "license_class": class,
        "redistribution": redistribution,
        "notice_paths": [notice]
    })
}

fn file_entry(path: &str, bytes: &[u8], kind: &str, license: Value) -> Value {
    let sha = sha256_hex(bytes);
    json!({
        "path": path,
        "sha256": sha,
        "bytes": bytes.len(),
        "blob": blob_key(&sha),
        "kind": kind,
        "license": license
    })
}

/// Write the three files under `from` and return the manifest bytes.
fn synthetic_bundle(from: &Path) -> Vec<u8> {
    let one = b"one: permitted weights".to_vec();
    let two = br#"{"kind":"metadata"}"#.to_vec();
    let three = b"three: provenance pending".to_vec();
    for (p, b) in [(ONE, &one), (TWO, &two), (THREE, &three)] {
        let dst = from.join(p);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::write(dst, b).unwrap();
    }
    let manifest = json!({
        "schema": "jope.model-bundle/v2",
        "bundle_id": ID,
        "bucket": "test",
        "published_at": "2026-09-18T00:00:00Z",
        "files": [
            file_entry(ONE, &one, "model", license("LicenseRef-Test-Permitted", "internal", "permitted", NOTICE_A)),
            file_entry(TWO, &two, "metadata", Value::Null),
            file_entry(THREE, &three, "model", license("LicenseRef-Test-Restricted", "restricted", "prohibited", NOTICE_B)),
        ],
        "notices": [
            {"path": NOTICE_A, "text": "Notice A: permitted internal model\n"},
            {"path": NOTICE_B, "text": "Notice B: fail-closed, redistribution prohibited\n"},
        ]
    });
    serde_json::to_vec_pretty(&manifest).unwrap()
}

fn store_url(dir: &Path) -> String {
    std::fs::create_dir_all(dir).unwrap();
    format!("file://{}", dir.display())
}

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
    let keys = rt().block_on(store.list("")).unwrap();
    assert_eq!(
        keys.iter()
            .filter(|k| k.starts_with("blobs/sha256/"))
            .count(),
        3
    );
    assert!(keys.contains(&format!("bundles/{ID}/manifest.json")));
    // Publishing again rewrites nothing.
    let r = rt()
        .block_on(publish(&store, &manifest_path, &from))
        .unwrap();
    assert_eq!(
        (r.blobs_created, r.blobs_existing, r.manifest.as_str()),
        (0, 3, "identical")
    );
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
