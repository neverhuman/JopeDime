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

mod cli;
mod publish;
