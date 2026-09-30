//! W1 consumer-map route R4: catalog and manifest resolution.
//!
//! A dataset id's catalog row (which exact blobs are its files) must come from
//! an authority: the store's manifest when the store is reachable, or an
//! explicitly pinned catalog version with its digest. A stale or altered
//! locally cached manifest is never served in its place.

mod common;

use common::*;
use jope_datasets::cache::{CatalogPin, Fetcher};
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

struct Published {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    url: String,
    cache: Cache,
}

/// Publish the synthetic tree (catalog v1) and stage dataset `A` online.
fn published() -> Published {
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
    let report = rt()
        .block_on(
            Fetcher {
                cache: &cache,
                store: Some(&store),
            }
            .fetch(A),
        )
        .unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    Published {
        _tmp: tmp,
        root,
        url,
        cache,
    }
}

fn original_train(p: &Published) -> Vec<u8> {
    std::fs::read(p.root.join("regression").join(A).join("train.csv")).unwrap()
}

/// Alter the cached `by-id/<A>/manifest.json` so that `train` names the test
/// blob (still a schema-valid row; both blobs are verified in the cache).
fn alter_cached_manifest(p: &Published) -> PathBuf {
    let path = p.cache.by_id_dir(A).join("manifest.json");
    let mut row: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let train = row["files"]["train"].clone();
    let test = row["files"]["test"].clone();
    for (to, from) in [("train", &test), ("test", &train)] {
        row["files"][to]["sha256"] = from["sha256"].clone();
        row["files"][to]["bytes"] = from["bytes"].clone();
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&row).unwrap()).unwrap();
    path
}

fn pin_v1(p: &Published) -> CatalogPin {
    let store = Store::open(&p.url).unwrap();
    let bytes = rt()
        .block_on(store.get_bytes("catalog/v1/catalog.jsonl"))
        .unwrap()
        .expect("ingest published catalog v1");
    CatalogPin::new("v1", &jope_datasets::hash::sha256_hex(&bytes)).unwrap()
}

/// Publish catalog v2 without dataset `A` and advance `LATEST` to it.
fn advance_latest_without_a(p: &Published) {
    let store = Store::open(&p.url).unwrap();
    let v1 = rt()
        .block_on(store.get_string("catalog/v1/catalog.jsonl"))
        .unwrap()
        .unwrap();
    let v2: String = v1
        .lines()
        .filter(|line| !line.contains(&format!("\"dataset_hash\":\"{A}\"")))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_ne!(v1, v2, "fixture: A must be in catalog v1");
    rt().block_on(store.put("catalog/v2/catalog.jsonl", v2.into_bytes()))
        .unwrap();
    rt().block_on(store.put("catalog/LATEST", b"v2\n".to_vec()))
        .unwrap();
}

fn names_a(report: &jope_datasets::cache::FetchReport, p: &Published) -> bool {
    report.errors.is_empty() && p.cache.by_id_dir(A).join("manifest.json").is_file()
}

#[test]
fn a_pinned_fetch_resolves_exactly_the_pinned_version_and_never_follows_latest() {
    let p = published();
    let pin = pin_v1(&p);
    advance_latest_without_a(&p);
    let store = Store::open(&p.url).unwrap();
    let online = Fetcher {
        cache: &p.cache,
        store: Some(&store),
    };
    // Unpinned resolution follows LATEST (documented acquisition behavior).
    let latest = rt().block_on(online.fetch("regression")).unwrap();
    // The pinned resolution is v1, whatever LATEST says, online and offline.
    let pinned = rt()
        .block_on(online.fetch_pinned("regression", &pin))
        .unwrap();
    assert_eq!(pinned.catalog.as_ref(), Some(&pin));
    assert_eq!(
        pinned.datasets,
        latest.datasets + 1,
        "{pinned:?} vs {latest:?}"
    );
    assert!(names_a(&pinned, &p));
    let offline = Fetcher {
        cache: &p.cache,
        store: None,
    };
    let again = rt().block_on(offline.fetch_pinned(A, &pin)).unwrap();
    assert!(
        again.errors.is_empty() && again.files_fetched == 0,
        "{again:?}"
    );
    assert_eq!(
        std::fs::read(p.cache.by_id_dir(A).join("train.csv")).unwrap(),
        original_train(&p)
    );
}

#[test]
fn a_pinned_catalog_whose_bytes_differ_is_refused_offline_and_reverified_online() {
    let p = published();
    let pin = pin_v1(&p);
    let store = Store::open(&p.url).unwrap();
    let online = Fetcher {
        cache: &p.cache,
        store: Some(&store),
    };
    rt().block_on(online.pinned_catalog(&pin)).unwrap();
    let cached = p.cache.catalog_dir().join("v1").join("catalog.jsonl");
    let mut bytes = std::fs::read(&cached).unwrap();
    bytes.extend_from_slice(b"\n");
    std::fs::write(&cached, &bytes).unwrap();
    let offline = Fetcher {
        cache: &p.cache,
        store: None,
    };
    let refused = format!(
        "{:#}",
        rt().block_on(offline.pinned_catalog(&pin)).unwrap_err()
    );
    assert!(refused.contains("pinned digest"), "{refused}");
    // Online, the store's copy is verified against the pin and reinstalled.
    rt().block_on(online.pinned_catalog(&pin)).unwrap();
    assert_eq!(
        jope_datasets::hash::sha256_hex(&std::fs::read(&cached).unwrap()),
        pin.sha256
    );
    // A pin the store cannot satisfy is refused and nothing is materialized.
    let wrong = CatalogPin::new("v1", &"0".repeat(64)).unwrap();
    let fresh = Cache::at(p._tmp.path().join("fresh"));
    let clean = Fetcher {
        cache: &fresh,
        store: Some(&store),
    };
    assert!(rt().block_on(clean.fetch_pinned(A, &wrong)).is_err());
    assert!(!fresh.by_id_dir(A).exists());
}

#[test]
fn a_pinned_fetch_replaces_a_stale_cached_manifest_even_offline() {
    let p = published();
    let pin = pin_v1(&p);
    let store = Store::open(&p.url).unwrap();
    rt().block_on(
        Fetcher {
            cache: &p.cache,
            store: Some(&store),
        }
        .pinned_catalog(&pin),
    )
    .unwrap();
    alter_cached_manifest(&p);
    let offline = Fetcher {
        cache: &p.cache,
        store: None,
    };
    let report = rt().block_on(offline.fetch_pinned(A, &pin)).unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        std::fs::read(p.cache.by_id_dir(A).join("train.csv")).unwrap(),
        original_train(&p)
    );
    // Offline without a pin, only the cached manifest exists: it is served.
    // This is the documented unpinned offline limit, not an authority.
    alter_cached_manifest(&p);
    rt().block_on(offline.fetch(A)).unwrap();
    assert_ne!(
        std::fs::read(p.cache.by_id_dir(A).join("train.csv")).unwrap(),
        original_train(&p)
    );
}

#[test]
fn pinned_selectors_are_ids_and_tasks_present_in_the_pinned_catalog() {
    let p = published();
    let pin = pin_v1(&p);
    let store = Store::open(&p.url).unwrap();
    let online = Fetcher {
        cache: &p.cache,
        store: Some(&store),
    };
    let collection = format!(
        "{:#}",
        rt().block_on(online.fetch_pinned("test-3", &pin))
            .unwrap_err()
    );
    assert!(collection.contains("own versions"), "{collection}");
    let absent = format!(
        "{:#}",
        rt().block_on(online.fetch_pinned("ffffffffffffffff", &pin))
            .unwrap_err()
    );
    assert!(absent.contains("not in pinned catalog"), "{absent}");
    assert!(CatalogPin::new("latest", &pin.sha256).is_err());
    assert!(CatalogPin::new("v1", "abc").is_err());
}

#[test]
fn a_stale_cached_manifest_is_not_served_while_the_store_is_reachable() {
    let p = published();
    alter_cached_manifest(&p);
    let store = Store::open(&p.url).unwrap();
    let online = Fetcher {
        cache: &p.cache,
        store: Some(&store),
    };
    let report = rt().block_on(online.fetch(A)).unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        std::fs::read(p.cache.by_id_dir(A).join("train.csv")).unwrap(),
        original_train(&p),
        "the altered cached manifest was served in place of the store's row"
    );
}
