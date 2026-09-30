//! W1-T05 (fixture scope): the staging loader serves only verified exact
//! objects. Cache hit, cache miss and resume are exercised through the real
//! `Fetcher`, not only a cold start. Live worker isolation (mounts, credentials,
//! network denial on managed hosts) is W7 and is not claimed here.

mod common;

use common::*;
use jope_datasets::cache::Fetcher;
use jope_datasets::catalog::CatalogRow;
use jope_datasets::ingest::{self, Options, Source};
use jope_datasets::paths::Cache;
use jope_datasets::store::Store;
use std::path::PathBuf;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

struct Staged {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    url: String,
    cache: Cache,
}

/// Publish the synthetic tree to a local store, then stage dataset `A` into a
/// fresh cache through an online fetch (cache miss path).
fn staged() -> Staged {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("quant");
    build_tree(&root);
    let store_dir = tmp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let url = format!("file://{}", store_dir.display());
    ingest::run(&Options {
        source: Source::Quant,
        root: root.clone(),
        out: tmp.path().join("out"),
        dry_run: false,
        resume: false,
        real_only: false,
        jobs: 2,
        limit: None,
        store_url: Some(url.clone()),
    })
    .unwrap();
    let cache = Cache::at(tmp.path().join("home").join("datasets"));
    let store = Store::open(&url).unwrap();
    let online = Fetcher {
        cache: &cache,
        store: Some(&store),
    };
    let report = rt().block_on(online.fetch(A)).unwrap();
    assert_eq!(
        (
            report.files_fetched,
            report.files_cached,
            report.errors.len()
        ),
        (3, 0, 0),
        "{report:?}"
    );
    Staged {
        _tmp: tmp,
        root,
        url,
        cache,
    }
}

fn manifest(s: &Staged) -> CatalogRow {
    jope_datasets::cache::read_manifest(&s.cache.by_id_dir(A).join("manifest.json")).unwrap()
}

fn original_train(s: &Staged) -> Vec<u8> {
    std::fs::read(s.root.join("regression").join(A).join("train.csv")).unwrap()
}

/// Overwrite the cached train blob in place with same-length different bytes.
/// `by-id/<A>/train.csv` is a hard link to it, so the consumer copy changes too.
fn corrupt_train_blob(s: &Staged) -> PathBuf {
    let row = manifest(s);
    let blob = s.cache.blob_path(&row.files.train.sha256);
    let mut bytes = std::fs::read(&blob).unwrap();
    bytes[0] = if bytes[0] == b'9' { b'8' } else { b'9' };
    std::fs::write(&blob, &bytes).unwrap();
    assert_eq!(bytes.len() as u64, row.files.train.bytes);
    blob
}

fn consumer_copy(s: &Staged) -> PathBuf {
    s.cache.by_id_dir(A).join("train.csv")
}

#[test]
fn verified_cache_hit_is_served_offline_without_network() {
    let s = staged();
    let offline = Fetcher {
        cache: &s.cache,
        store: None,
    };
    let report = rt().block_on(offline.fetch(A)).unwrap();
    assert_eq!(
        (
            report.files_fetched,
            report.files_cached,
            report.errors.len()
        ),
        (0, 3, 0),
        "{report:?}"
    );
    assert_eq!(
        std::fs::read(consumer_copy(&s)).unwrap(),
        original_train(&s)
    );
}

#[test]
fn stale_same_size_cache_entry_is_replaced_online_and_never_served() {
    let s = staged();
    corrupt_train_blob(&s);
    let store = Store::open(&s.url).unwrap();
    let online = Fetcher {
        cache: &s.cache,
        store: Some(&store),
    };
    let report = rt().block_on(online.fetch(A)).unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        std::fs::read(consumer_copy(&s)).unwrap(),
        original_train(&s),
        "a same-size stale cache entry was served as the exact object ({report:?})"
    );
    assert_eq!(report.files_fetched, 1, "{report:?}");
}

#[test]
fn stale_cache_entry_offline_stops_the_request_and_is_not_left_servable() {
    let s = staged();
    let blob = corrupt_train_blob(&s);
    let offline = Fetcher {
        cache: &s.cache,
        store: None,
    };
    let report = rt().block_on(offline.fetch(A)).unwrap();
    assert!(
        !report.errors.is_empty(),
        "offline fetch served a stale cache entry: {report:?}"
    );
    assert!(!blob.exists(), "stale blob left in the verified store");
    assert!(
        !consumer_copy(&s).exists(),
        "stale consumer copy left servable"
    );
    // The untouched members of the same dataset remain exact.
    let row = manifest(&s);
    assert_eq!(
        jope_datasets::hash::sha256_hex(
            &std::fs::read(s.cache.by_id_dir(A).join("test.csv")).unwrap()
        ),
        row.files.test.sha256
    );
}

#[test]
fn missing_required_object_offline_stops_the_request() {
    let s = staged();
    let row = manifest(&s);
    std::fs::remove_file(s.cache.blob_path(&row.files.train.sha256)).unwrap();
    let offline = Fetcher {
        cache: &s.cache,
        store: None,
    };
    let report = rt().block_on(offline.fetch(A)).unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|e| e.contains("not cached and offline")),
        "{report:?}"
    );
    assert!(!s.cache.blob_path(&row.files.train.sha256).exists());
}

#[test]
fn corrupt_resume_partial_is_never_promoted() {
    let s = staged();
    let row = manifest(&s);
    let store = Store::open(&s.url).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("blob");
    let partial = dir.path().join("blob.partial");
    let original = original_train(&s);
    let mut prefix = original[..original.len() / 2].to_vec();
    prefix[0] ^= 0x55;
    std::fs::write(&partial, &prefix).unwrap();
    let key = format!("blobs/sha256/{}", row.files.train.sha256);
    let err = rt()
        .block_on(store.download_verified(&key, &dest, &row.files.train.sha256, 1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("sha256 mismatch"), "{err}");
    assert!(!dest.exists() && !partial.exists());
    // A retry from scratch then produces the exact object.
    rt().block_on(store.download_verified(&key, &dest, &row.files.train.sha256, 1))
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), original);
}
