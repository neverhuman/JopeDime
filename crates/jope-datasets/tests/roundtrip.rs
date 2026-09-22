mod common;

use common::*;
use jope_datasets::cache::{link_from, verify_cache, Fetcher};
use jope_datasets::catalog::{Collection, CollectionTask, Policy, Visibility};
use jope_datasets::ingest::{self, Options, Source};
use jope_datasets::paths::Cache;
use jope_datasets::store::Store;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::process::Command;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn ingest_into(root: &Path, out: &Path, store_url: &str, resume: bool) -> ingest::Summary {
    let opts = Options {
        source: Source::Quant,
        root: root.to_path_buf(),
        out: out.to_path_buf(),
        dry_run: false,
        resume,
        jobs: 4,
        limit: None,
        store_url: Some(store_url.to_string()),
    };
    ingest::run(&opts).unwrap()
}

#[test]
fn ingest_upload_fetch_verify_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("quant");
    build_tree(&root);
    let store_dir = tmp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let url = format!("file://{}", store_dir.display());
    let out = tmp.path().join("out");

    let s = ingest_into(&root, &out, &url, false);
    assert_eq!(
        (s.rows, s.uploaded, s.already_present, s.skipped_by_ledger),
        (3, 9, 0, 0),
        "{s:?}"
    );
    let store = Store::open(&url).unwrap();
    let keys = rt().block_on(store.list("")).unwrap();
    assert_eq!(
        keys.iter()
            .filter(|k| k.starts_with("blobs/sha256/"))
            .count(),
        9
    );
    assert!(keys.contains(&format!("datasets/{A}/manifest.json")));
    assert!(keys.contains(&"catalog/v1/catalog.jsonl".to_string()));
    assert!(keys
        .iter()
        .any(|k| k.starts_with("ingest/reports/") && k.ends_with("/ledger.jsonl")));
    let ledger = std::fs::read_to_string(out.join("ledger.jsonl")).unwrap();
    assert_eq!(ledger.lines().count(), 9);

    // --resume: every blob is ledger-verified, nothing is re-put; the catalog gets a new version.
    let s = ingest_into(&root, &out, &url, true);
    assert_eq!(
        (s.uploaded, s.already_present, s.skipped_by_ledger),
        (0, 0, 9),
        "{s:?}"
    );
    assert_eq!(
        rt().block_on(store.get_string("catalog/LATEST"))
            .unwrap()
            .unwrap()
            .trim(),
        "v2"
    );
    // Without --resume the create-only put reports the existing keys and rewrites nothing.
    let s = ingest_into(&root, &out, &url, false);
    assert_eq!((s.uploaded, s.already_present), (0, 9), "{s:?}");

    // Publish a collection the way the maintainer side would.
    let members = vec![A.to_string(), format!("synth_{B}"), D.to_string()];
    let coll = Collection {
        schema: "jope.dataset-collection/v1".into(),
        name: "test-3".into(),
        version: 1,
        visibility: Visibility::Private,
        task: CollectionTask::Mixed,
        description: None,
        created_at: None,
        members: members.clone(),
        policy: Policy {
            license_required: false,
            allowed_spdx: vec![],
        },
        members_sha256: Collection::members_digest(&members),
    };
    rt().block_on(store.put(
        "collections/test-3/v1.json",
        serde_json::to_vec(&coll).unwrap(),
    ))
    .unwrap();
    rt().block_on(store.put("collections/test-3/LATEST", b"v1\n".to_vec()))
        .unwrap();

    // Fetch into a fresh cache: by id, by task, by collection.
    let cache = Cache::at(tmp.path().join("home").join("datasets"));
    let f = Fetcher {
        cache: &cache,
        store: Some(&store),
    };
    let r = rt().block_on(f.fetch(A)).unwrap();
    assert_eq!(
        (r.datasets, r.files_fetched, r.files_cached, r.errors.len()),
        (1, 3, 0, 0),
        "{r:?}"
    );
    let by_id = cache.by_id_dir(A);
    for name in ["train.csv", "test.csv", "meta.json", "manifest.json"] {
        assert!(by_id.join(name).is_file(), "{name}");
    }
    let row = jope_datasets::cache::read_manifest(&by_id.join("manifest.json")).unwrap();
    let blob = cache.blob_path(&row.files.train.sha256);
    assert_eq!(
        std::fs::metadata(&blob).unwrap().ino(),
        std::fs::metadata(by_id.join("train.csv")).unwrap().ino(),
        "hardlink"
    );
    let r = rt().block_on(f.fetch("regression")).unwrap();
    assert_eq!(
        (r.datasets, r.files_fetched, r.files_cached),
        (2, 3, 3),
        "{r:?}"
    );
    let r = rt().block_on(f.fetch("test-3")).unwrap();
    assert_eq!(
        (r.datasets, r.files_fetched, r.files_cached),
        (3, 3, 6),
        "{r:?}"
    );
    assert!(cache
        .collections_dir()
        .join("test-3")
        .join("v1.json")
        .is_file());
    let latest = std::fs::read_to_string(cache.catalog_dir().join("LATEST")).unwrap();
    assert_eq!(
        latest.trim(),
        "v3",
        "three ingests published three catalog versions"
    );
    assert!(cache
        .catalog_dir()
        .join("v3")
        .join("catalog.jsonl")
        .is_file());

    // Offline serves the cache and refuses what is not there.
    let off = Fetcher {
        cache: &cache,
        store: None,
    };
    let r = rt().block_on(off.fetch("test-3")).unwrap();
    assert_eq!((r.datasets, r.files_fetched, r.errors.len()), (3, 0, 0));
    let empty = Cache::at(tmp.path().join("empty"));
    let off = Fetcher {
        cache: &empty,
        store: None,
    };
    assert!(rt().block_on(off.fetch("test-3")).is_err());
    assert!(rt().block_on(off.fetch(A)).is_err());

    // verify: clean, then a corrupted blob is reported (quick mode only checks sizes).
    let v = verify_cache(&cache, false);
    assert_eq!((v.blobs, v.datasets, v.errors.len()), (9, 3, 0), "{v:?}");
    std::fs::write(&blob, "corrupted\n").unwrap();
    let v = verify_cache(&cache, false);
    assert_eq!(
        v.errors.len(),
        1,
        "the by-id file is the same inode, so one finding: {:?}",
        v.errors
    );
    assert!(v.errors[0].starts_with("blob "));
    assert!(verify_cache(&cache, true).errors.is_empty());

    // A corrupted store object: the download is rejected and no partial survives.
    let key = store_dir
        .join("blobs")
        .join("sha256")
        .join(&row.files.test.sha256);
    std::fs::write(&key, "tampered\n").unwrap();
    let fresh = Cache::at(tmp.path().join("fresh"));
    let f = Fetcher {
        cache: &fresh,
        store: Some(&store),
    };
    let r = rt().block_on(f.fetch(A)).unwrap();
    assert_eq!(r.errors.len(), 2, "{:?}", r.errors);
    assert!(r.errors[0].contains("sha256 mismatch") || r.errors[1].contains("sha256 mismatch"));
    assert!(!fresh.blob_path(&row.files.test.sha256).exists());
    let partials: Vec<_> = std::fs::read_dir(fresh.blobs_dir())
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".partial")
        })
        .collect();
    assert!(partials.is_empty());
}

#[test]
fn ranged_resume_completes_a_partial() {
    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();
    let body: Vec<u8> = (0..100_000u32).flat_map(|i| i.to_le_bytes()).collect();
    let sha = jope_datasets::hash::sha256_hex(&body);
    rt().block_on(store.put(&format!("blobs/sha256/{sha}"), body.clone()))
        .unwrap();
    let dest = tmp.path().join("blob");
    std::fs::write(tmp.path().join("blob.partial"), &body[..123_456]).unwrap();
    let n = rt()
        .block_on(store.download_verified(&format!("blobs/sha256/{sha}"), &dest, &sha, 1024))
        .unwrap();
    assert_eq!(n, body.len() as u64);
    assert_eq!(std::fs::read(&dest).unwrap(), body);
}

#[test]
fn link_seeds_cache_from_registry_layout_and_repo_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("quant");
    build_tree(&root);
    let cache = Cache::at(tmp.path().join("home").join("datasets"));
    let r = link_from(
        &cache,
        &root.join("regression"),
        &[A.to_string(), format!("synth_{B}")],
        None,
    )
    .unwrap();
    assert_eq!((r.datasets, r.errors.len()), (2, 0), "{r:?}");
    assert_eq!(
        r.skipped, 3,
        "C, F and _dupes_quarantine are skipped by --only"
    );
    let off = Fetcher {
        cache: &cache,
        store: None,
    };
    let f = rt().block_on(off.fetch("regression")).unwrap();
    assert_eq!((f.datasets, f.errors.len()), (2, 0));
    // The mismatching directory C is an error, not silently skipped.
    let r = link_from(&cache, &root.join("regression"), &[C.to_string()], None).unwrap();
    assert_eq!((r.datasets, r.errors.len()), (0, 1), "{r:?}");

    // The committed repository seeds the test-10 collection for offline use.
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test");
    let r = link_from(&cache, &repo, &[], None).unwrap();
    assert_eq!((r.datasets, r.errors.len()), (10, 0), "{r:?}");
    assert_eq!(r.collections, vec!["test-10".to_string()]);
    let f = rt().block_on(off.fetch("test-10")).unwrap();
    assert_eq!(
        (f.datasets, f.files_fetched, f.files_cached, f.errors.len()),
        (10, 0, 20, 0),
        "{f:?}"
    );
    assert!(cache
        .by_id_dir("1cb97b1d7314d43d")
        .join("train.csv")
        .is_file());
    let v = verify_cache(&cache, false);
    assert_eq!((v.blobs, v.datasets, v.errors.len()), (26, 12, 0), "{v:?}");
}

#[test]
fn cli_offline_fetch_prints_registry() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test");
    let bin = env!("CARGO_BIN_EXE_jope-datasets");
    let st = std::process::Command::new(bin)
        .env("JOPE_HOME", &home)
        .args(["link", "--from"])
        .arg(&repo)
        .status()
        .unwrap();
    assert!(st.success());
    let out = std::process::Command::new(bin)
        .env("JOPE_HOME", &home)
        .env("JOPE_OFFLINE", "1")
        .args(["fetch", "test-10", "--print-registry"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        home.join("datasets").join("by-id").display().to_string()
    );
    let st = std::process::Command::new(bin)
        .env("JOPE_HOME", &home)
        .args(["verify", "--catalog"])
        .status()
        .unwrap();
    assert!(st.success());
    let st = std::process::Command::new(bin)
        .args(["verify", "--repo"])
        .arg(&repo)
        .status()
        .unwrap();
    assert!(st.success());
}

#[test]
fn native_providers_remain_available() {
    if std::env::var_os("JOPE_TEST_PROVIDER_CHILD").is_some() {
        assert!(jope_datasets::credentials::resolve().unwrap().is_none());
        return;
    }
    for environment in [
        vec![
            ("AWS_WEB_IDENTITY_TOKEN_FILE", "/unused-token"),
            ("AWS_ROLE_ARN", "role"),
        ],
        vec![("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/credentials")],
        vec![
            (
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "http://localhost/credentials",
            ),
            ("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", "/unused-token"),
        ],
    ] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native_providers_remain_available",
                "--nocapture",
            ])
            .env_clear()
            .env("JOPE_TEST_PROVIDER_CHILD", "1")
            .envs(environment)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

#[cfg(unix)]
#[test]
fn failing_helper_cannot_disclose_stderr() {
    if std::env::var_os("JOPE_TEST_HELPER_CHILD").is_some() {
        let error = match jope_datasets::credentials::from_profile("fixture") {
            Ok(_) => panic!("failing helper unexpectedly succeeded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("failed for profile fixture"));
        assert!(
            !error.contains("SECRET_CANARY"),
            "helper stderr leaked: {error}"
        );
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let aws = temp.path().join("aws");
    std::fs::write(&aws, "#!/bin/sh\nprintf SECRET_CANARY >&2\nexit 7\n").unwrap();
    std::fs::set_permissions(&aws, std::fs::Permissions::from_mode(0o700)).unwrap();
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "failing_helper_cannot_disclose_stderr",
            "--nocapture",
        ])
        .env_clear()
        .env("JOPE_TEST_HELPER_CHILD", "1")
        .env("PATH", temp.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
