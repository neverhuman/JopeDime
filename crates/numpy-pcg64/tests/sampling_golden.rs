//! Regenerate the frozen jope-dime `sampling.json` recipes and compare them
//! exactly. The numpy 1.26.4 output recorded in those files is the oracle.
//!
//! The default lane checks a vendored subset (`tests/golden/`) and rebuilds
//! each whole source document from its ordered key list, matching the
//! recorded sha256 and git blob id byte for byte. Provenance of the subset
//! against the real files runs when `NUMPY_PCG64_JOPE_DIME_GIT` (a jope-dime
//! clone) and `NUMPY_PCG64_ARCHIVE_ROOT` (the directory holding the named
//! archive, e.g. `jain-datasets/.runtime`) are set; missing sources print a
//! `SKIP provenance` line, and `NUMPY_PCG64_REQUIRE_SOURCES=1` makes them fail.
//! The complete check runs over every file named in
//! `NUMPY_PCG64_SAMPLING_FILES` (colon-separated paths) with
//! `cargo test -p numpy-pcg64 -- --ignored`.

use numpy_pcg64::document::{foundation_document, target_document};
use numpy_pcg64::recipes::{
    foundation_recipe, target_key, target_recipe, FOUNDATION_SCHEMA, FOUNDATION_SEED,
    NUMPY_VERSION, TARGET_SCHEMA, TARGET_SEED, TARGET_VIEWS, TARGET_WINDOW,
};
use serde_json::Value;

const FILES_ENV: &str = "NUMPY_PCG64_SAMPLING_FILES";
const GIT_ENV: &str = "NUMPY_PCG64_JOPE_DIME_GIT";
const ARCHIVE_ENV: &str = "NUMPY_PCG64_ARCHIVE_ROOT";
const REQUIRE_ENV: &str = "NUMPY_PCG64_REQUIRE_SOURCES";

fn indices(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("index list")
        .iter()
        .map(|v| v.as_u64().expect("non-negative index"))
        .collect()
}

fn windows(value: &Value) -> Vec<Vec<u64>> {
    value
        .as_array()
        .expect("window list")
        .iter()
        .map(indices)
        .collect()
}

/// Check every recipe of one sampling document; returns the recipe count.
fn check_document(label: &str, doc: &Value) -> usize {
    assert_eq!(doc["numpy_version"], NUMPY_VERSION, "{label}");
    let recipes = doc["recipes"].as_object().expect("recipes object");
    assert!(!recipes.is_empty(), "{label}: no recipes");
    match doc["schema"].as_str().expect("schema") {
        FOUNDATION_SCHEMA => {
            assert_eq!(doc["seed"], FOUNDATION_SEED, "{label}");
            for (key, recipe) in recipes {
                let n: usize = key.parse().expect("foundation key is a length");
                let actual = foundation_recipe(n);
                assert_eq!(
                    actual.context,
                    indices(&recipe["context"]),
                    "{label}: context {key}"
                );
                assert_eq!(
                    actual.query,
                    indices(&recipe["query"]),
                    "{label}: query {key}"
                );
            }
        }
        TARGET_SCHEMA => {
            assert_eq!(doc["seed"], TARGET_SEED, "{label}");
            assert_eq!(doc["window"], TARGET_WINDOW as u64, "{label}");
            assert_eq!(doc["views"], TARGET_VIEWS as u64, "{label}");
            for (key, recipe) in recipes {
                let (task, n) = key.split_once(':').expect("task:length key");
                let (task, n): (u32, usize) = (task.parse().unwrap(), n.parse().unwrap());
                assert_eq!(target_key(task, n), *key);
                let actual = target_recipe(task, n).expect("valid target length");
                assert_eq!(
                    actual.classical,
                    windows(&recipe["classical"]),
                    "{label}: classical {key}"
                );
                assert_eq!(
                    actual.neural,
                    windows(&recipe["neural"]),
                    "{label}: neural {key}"
                );
            }
        }
        other => panic!("{label}: unknown sampling schema {other}"),
    }
    recipes.len()
}

fn sha256_hex(data: &[u8]) -> String {
    // Minimal SHA-256 so the vendored provenance digests are checked without a
    // runtime dependency.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = s1
                .wrapping_add(w[i - 7])
                .wrapping_add(s0)
                .wrapping_add(w[i - 16]);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let t1 = hh
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

/// Git blob id (`git hash-object`) so a source's identity is checkable
/// without git.
fn git_blob_hex(data: &[u8]) -> String {
    let mut object = format!("blob {}\0", data.len()).into_bytes();
    object.extend_from_slice(data);
    sha1_hex(&object)
}

fn sha1_hex(data: &[u8]) -> String {
    let mut h: [u32; 5] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5a827999),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

fn subset() -> Value {
    serde_json::from_str(include_str!("golden/sampling-subset.json")).expect("subset json")
}

fn source_label(source: &Value) -> String {
    let path = source["path"].as_str().expect("source path");
    assert!(
        !path.starts_with('/'),
        "source path must be relative: {path}"
    );
    match source["store"].as_str().expect("source store") {
        "git" => format!(
            "{}@{}:{path}",
            source["repo"].as_str().expect("repo"),
            source["commit"].as_str().expect("commit")
        ),
        "archive" => format!("{}:{path}", source["archive"].as_str().expect("archive")),
        other => panic!("unknown source store {other}"),
    }
}

/// Read a vendored source through `NUMPY_PCG64_JOPE_DIME_GIT` (a jope-dime
/// clone, for `git` sources) or `NUMPY_PCG64_ARCHIVE_ROOT` (the directory
/// holding the named archive, for `archive` sources).
fn read_source(source: &Value) -> Result<Vec<u8>, String> {
    let path = source["path"].as_str().unwrap();
    match source["store"].as_str().unwrap() {
        "git" => {
            let repo = std::env::var(GIT_ENV).map_err(|_| format!("{GIT_ENV} is unset"))?;
            let spec = format!("{}:{path}", source["commit"].as_str().unwrap());
            let output = std::process::Command::new("git")
                .args(["-C", &repo, "cat-file", "blob", &spec])
                .output()
                .map_err(|e| format!("git: {e}"))?;
            if output.status.success() {
                Ok(output.stdout)
            } else {
                Err(format!("{spec} is not readable in {repo}"))
            }
        }
        _ => {
            let root = std::env::var(ARCHIVE_ENV).map_err(|_| format!("{ARCHIVE_ENV} is unset"))?;
            let file = std::path::Path::new(&root)
                .join(source["archive"].as_str().unwrap())
                .join(path);
            std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))
        }
    }
}

#[test]
fn vendored_subset_regenerates_exactly() {
    let mut total = 0;
    for doc in subset()["documents"].as_array().expect("documents") {
        let source = &doc["source"];
        let label = source_label(source);
        assert_eq!(source["sha256"].as_str().unwrap().len(), 64, "{label}");
        assert_eq!(source["git_blob"].as_str().unwrap().len(), 40, "{label}");
        total += check_document(&label, doc);
    }
    assert!(total >= 6, "vendored subset shrank to {total} recipes");
}

#[test]
fn documents_rebuild_their_sources_byte_for_byte() {
    // The writer must reproduce each whole source file from its ordered key
    // list alone: the recorded sha256 and git blob id are the oracle.
    for doc in subset()["documents"].as_array().unwrap() {
        let source = &doc["source"];
        let label = source_label(source);
        let keys = source["recipe_keys"].as_array().expect("recipe_keys");
        let rebuilt = match doc["schema"].as_str().unwrap() {
            FOUNDATION_SCHEMA => {
                let lengths: Vec<usize> = keys
                    .iter()
                    .map(|k| k.as_u64().expect("foundation key") as usize)
                    .collect();
                foundation_document(&lengths)
            }
            TARGET_SCHEMA => {
                let pairs: Vec<(u32, usize)> = keys
                    .iter()
                    .map(|k| {
                        let (task, n) = k.as_str().unwrap().split_once(':').unwrap();
                        (task.parse().unwrap(), n.parse().unwrap())
                    })
                    .collect();
                target_document(&pairs).expect("valid target lengths")
            }
            other => panic!("{label}: unknown schema {other}"),
        };
        assert_eq!(sha256_hex(rebuilt.as_bytes()), source["sha256"], "{label}");
        assert_eq!(
            git_blob_hex(rebuilt.as_bytes()),
            source["git_blob"],
            "{label}"
        );
        // The excerpt is part of the rebuilt document.
        let full: Value = serde_json::from_str(&rebuilt).unwrap();
        for (key, recipe) in doc["recipes"].as_object().unwrap() {
            assert_eq!(&full["recipes"][key], recipe, "{label}: {key}");
        }
    }
}

#[test]
fn vendored_subset_matches_its_sources_when_available() {
    // Provenance: when a vendored source is readable, the subset must be a
    // verbatim excerpt of the file with the recorded digests. Missing sources
    // are reported; NUMPY_PCG64_REQUIRE_SOURCES=1 turns them into failures.
    let require = std::env::var(REQUIRE_ENV).is_ok_and(|v| v == "1");
    let mut checked = 0;
    for doc in subset()["documents"].as_array().unwrap() {
        let source = &doc["source"];
        let label = source_label(source);
        let bytes = match read_source(source) {
            Ok(bytes) => bytes,
            Err(reason) if require && source["store"].as_str() == Some("git") => {
                panic!("{label}: git source required but missing: {reason}")
            }
            Err(reason) => {
                eprintln!(
                    "SKIP provenance {label}: {reason} (set {REQUIRE_ENV}=1 to fail git sources)"
                );
                continue;
            }
        };
        assert_eq!(sha256_hex(&bytes), source["sha256"], "{label}");
        assert_eq!(git_blob_hex(&bytes), source["git_blob"], "{label}");
        // Key order is pinned by the sha256 above together with
        // `documents_rebuild_their_sources_byte_for_byte`.
        let full: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            full["recipes"].as_object().unwrap().len(),
            source["recipe_keys"].as_array().unwrap().len(),
            "{label}"
        );
        for (key, recipe) in doc["recipes"].as_object().unwrap() {
            assert_eq!(&full["recipes"][key], recipe, "{label}: {key}");
        }
        eprintln!("provenance {label}: verified");
        checked += 1;
    }
    if require {
        let git_docs = subset()["documents"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|doc| doc["source"]["store"].as_str() == Some("git"))
            .count();
        assert!(
            checked >= git_docs && git_docs > 0,
            "{REQUIRE_ENV}=1 must read every git-store source (checked {checked}, git sources {git_docs})"
        );
    }
}

#[test]
fn digest_helpers_are_correct() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(
        git_blob_hex(b""),
        "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    );
}

#[test]
#[ignore = "needs NUMPY_PCG64_SAMPLING_FILES pointing at committed sampling.json files"]
fn committed_sampling_files_regenerate_exactly() {
    let files = std::env::var(FILES_ENV).expect("set NUMPY_PCG64_SAMPLING_FILES");
    let mut checked = 0;
    for path in files.split(':').filter(|p| !p.is_empty()) {
        let doc: Value = serde_json::from_slice(&std::fs::read(path).expect(path)).expect(path);
        let count = check_document(path, &doc);
        println!("{path}: {count} recipes regenerated exactly");
        checked += 1;
    }
    assert!(checked > 0, "no sampling files named");
}
