//! Weight-bound tests read the governed `jope-prime-gen31-v2` bundle from `JOPE_MODELS_ROOT`
//! (the directory `jope models fetch jope-prime-gen31-v2 --accept-restricted` materialises, i.e.
//! `jope models path jope-prime-gen31-v2`, or a checkout laid out like jain-starforge with
//! `artifacts/model-bundle.v1.json`). Without it a test prints a visible `SKIPPED:` line and
//! returns without asserting; CI lanes that must not skip grep for that marker.
#![allow(dead_code)]

use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub const ENV: &str = "JOPE_MODELS_ROOT";
pub const BUNDLE_ID: &str = "jope-prime-gen31-v2";

/// Print `SKIPPED: <test>: <reason>` straight to stderr so `cargo test`'s output capture does
/// not hide it.
pub fn skipped(test: &str, reason: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "SKIPPED: {test}: {reason}");
}

/// The bundle root in the layout `starforge::bundle::LoadedModelBundle::load` reads
/// (`model-bundle.v1.json` with the files at `artifacts/…` and notices at `docs/notices/…`),
/// or `None` after printing `SKIPPED:`.
pub fn bundle_root(test: &str) -> Option<PathBuf> {
    match try_bundle_root() {
        Ok(root) => Some(root),
        Err(reason) => {
            skipped(test, &reason);
            None
        }
    }
}

/// [`bundle_root`] without the `SKIPPED:` line: the root, or the reason it is unavailable.
pub fn try_bundle_root() -> Result<PathBuf, String> {
    let root = std::env::var_os(ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "{ENV} is not set (run `jope models fetch {BUNDLE_ID} --accept-restricted` and \
                 export {ENV}=$(jope models path {BUNDLE_ID}))"
            )
        })?;
    let root =
        std::fs::canonicalize(&root).map_err(|e| format!("{ENV}={}: {e}", root.display()))?;
    if !root.is_dir() {
        return Err(format!("{ENV}={} is not a directory", root.display()));
    }
    if root.join("artifacts/model-bundle.v1.json").is_file()
        || (root.join("model-bundle.v1.json").is_file() && root.join("artifacts").is_dir())
    {
        return Ok(root);
    }
    if root.join("model-bundle.v1.json").is_file() {
        return starforge_view(&root);
    }
    Err(format!(
        "{ENV}={} holds no model-bundle.v1.json (expected the fetched {BUNDLE_ID} bundle)",
        root.display()
    ))
}

/// A fetched `jope.model-bundle/v2` directory stores its files at `<root>/<path>` while the v1
/// manifest inside it names them `artifacts/<path>`. Build (idempotently, by hardlink) the
/// sibling `<root>.starforge/` holding the manifest, every listed file at `artifacts/<path>`
/// and the notices at `docs/notices/…`, so the crate's symlink-refusing loader can open it.
fn starforge_view(root: &Path) -> Result<PathBuf, String> {
    // One builder per process; other processes are safe because every link lands under a
    // private name and is renamed into place.
    static BUILD: Mutex<()> = Mutex::new(());
    let _guard = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    let manifest_path = root.join("model-bundle.v1.json");
    let bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let name = root
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| BUNDLE_ID.to_string());
    let view = root.with_file_name(format!("{name}.starforge"));
    link(&manifest_path, &view.join("model-bundle.v1.json"))?;
    let files = manifest["files"]
        .as_array()
        .ok_or("model-bundle.v1.json: `files` is not an array")?;
    for f in files {
        let rel = f["path"]
            .as_str()
            .ok_or("model-bundle.v1.json: a file has no `path`")?;
        let src = root.join(rel.strip_prefix("artifacts/").unwrap_or(rel));
        link(&src, &view.join(rel))?;
    }
    if let Some(notices) = manifest["notices"].as_object() {
        for rel in notices.keys() {
            link(&root.join(rel), &view.join(rel))?;
        }
    }
    Ok(view)
}

/// Hardlink (else copy) `src` at `dst` through a private temporary name renamed into place:
/// never write into an existing `dst`, which may already be a hardlink of `src` (a plain copy
/// onto it would truncate the shared inode — the weights in the cache).
fn link(src: &Path, dst: &Path) -> Result<(), String> {
    let s = std::fs::metadata(src).map_err(|e| {
        format!(
            "{}: {e} (re-run `jope models fetch {BUNDLE_ID} --accept-restricted`)",
            src.display()
        )
    })?;
    if let Ok(d) = std::fs::metadata(dst) {
        if d.ino() == s.ino() && d.dev() == s.dev() {
            return Ok(());
        }
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let tmp = dst.with_file_name(format!(
        ".{}.{}.{}.linktmp",
        dst.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    if std::fs::hard_link(src, &tmp).is_err() {
        // `create_new`: only ever write into a file this call created.
        let copy = (|| -> std::io::Result<u64> {
            let mut out = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?;
            std::io::copy(&mut std::fs::File::open(src)?, &mut out)
        })();
        if let Err(e) = copy {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("copy {} -> {}: {e}", src.display(), tmp.display()));
        }
    }
    let renamed = std::fs::rename(&tmp, dst);
    // A rename between two names of the same inode is a successful no-op: drop the temp entry.
    let _ = std::fs::remove_file(&tmp);
    renamed.map_err(|e| format!("rename {}: {e}", dst.display()))?;
    Ok(())
}
