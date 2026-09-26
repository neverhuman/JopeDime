//! A content-addressed blob directory (`<dir>/<sha256>`) shared by the dataset and model caches:
//! atomic local stores (`.partial` + rename), hardlink-or-copy materialisation, verified downloads
//! from a [`Store`] (`blobs/sha256/<sha256>`) and re-hash verification.

use crate::hash::digest_file;
use crate::store::Store;
use anyhow::{anyhow, Context, Result};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Partials above this size are resumed with a ranged GET.
pub const RESUME_THRESHOLD: u64 = 8 * 1024 * 1024;
/// Concurrent blob downloads.
pub const PARALLEL_GETS: usize = 16;

/// The object key of a blob in every JOPE bucket.
pub fn blob_key(sha256: &str) -> String {
    format!("blobs/sha256/{sha256}")
}

pub fn is_sha256_name(name: &str) -> bool {
    name.len() == 64 && name.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Clone, Debug, Default)]
pub struct BlobVerify {
    pub blobs: usize,
    pub errors: Vec<String>,
}

/// `<dir>/<sha256>` plus `<dir>/<sha256>.partial` while a download or copy is in flight.
#[derive(Clone, Debug)]
pub struct BlobDir {
    pub dir: PathBuf,
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(x), Ok(y)) => x.ino() == y.ino() && x.dev() == y.dev(),
        _ => false,
    }
}

/// Hardlink `blob` at `dst` (copy when linking fails, e.g. across devices).
///
/// The link or copy is made under a private temporary name and then renamed over `dst`, so a
/// concurrent materialiser of the same path can never make this write into — and truncate —
/// an inode it shares with `blob` (a plain `copy` onto an existing hardlink would).
pub fn link_or_copy(blob: &Path, dst: &Path) -> Result<()> {
    if same_file(blob, dst) {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let name = dst
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let tmp = dst.with_file_name(format!(
        ".{name}.{}.{}.linktmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    if std::fs::hard_link(blob, &tmp).is_err() {
        // `create_new`: the copy only ever writes into a file this call created.
        let copy = (|| -> std::io::Result<()> {
            let mut out = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?;
            std::io::copy(&mut std::fs::File::open(blob)?, &mut out)?;
            out.sync_all()
        })();
        if let Err(e) = copy {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("copy {} -> {}", blob.display(), dst.display()));
        }
    }
    let renamed = std::fs::rename(&tmp, dst);
    // POSIX rename is a successful no-op when `tmp` and `dst` already name the same inode (a
    // concurrent caller linked `dst` first), so the temporary entry is dropped either way.
    let _ = std::fs::remove_file(&tmp);
    renamed.with_context(|| format!("rename into {}", dst.display()))?;
    Ok(())
}

impl BlobDir {
    pub fn new(dir: impl Into<PathBuf>) -> BlobDir {
        BlobDir { dir: dir.into() }
    }

    pub fn path(&self, sha256: &str) -> PathBuf {
        self.dir.join(sha256)
    }

    pub fn partial_path(&self, sha256: &str) -> PathBuf {
        self.dir.join(format!("{sha256}.partial"))
    }

    /// Copy a local file into the store under `sha256` (atomic via `.partial`). The caller has
    /// hashed `src`; pass `verify` to re-hash the copy before it is exposed.
    pub fn store_file(&self, src: &Path, sha256: &str, verify: bool) -> Result<PathBuf> {
        let dest = self.path(sha256);
        if dest.is_file() {
            return Ok(dest);
        }
        std::fs::create_dir_all(&self.dir)?;
        let partial = self.partial_path(sha256);
        std::fs::copy(src, &partial).with_context(|| format!("copy {}", src.display()))?;
        if verify {
            let got = digest_file(&partial)?.sha256;
            if got != sha256 {
                let _ = std::fs::remove_file(&partial);
                return Err(anyhow!(
                    "{}: sha256 {got} != expected {sha256}",
                    src.display()
                ));
            }
        }
        std::fs::rename(&partial, &dest)?;
        Ok(dest)
    }

    /// Make sure `<dir>/<sha256>` exists with exactly the named content, downloading
    /// `blobs/sha256/<sha256>` from `store` when it is absent or stale. A cached entry is served
    /// only after its bytes re-hash to `sha256`; a same-size stale or corrupted entry is removed
    /// and, offline, the request fails rather than serving any substitute. `Ok(true)` when it was
    /// downloaded.
    pub async fn ensure(&self, store: Option<&Store>, sha256: &str, bytes: u64) -> Result<bool> {
        let dest = self.path(sha256);
        if let Ok(m) = std::fs::metadata(&dest) {
            let found = if m.len() == bytes {
                let got = digest_file(&dest)?.sha256;
                if got == sha256 {
                    return Ok(false);
                }
                format!("content hashes to {got}")
            } else {
                format!("size {} != {bytes}", m.len())
            };
            std::fs::remove_file(&dest)?;
            if store.is_none() {
                return Err(anyhow!(
                    "cached blob {sha256} is stale ({found}); removed, and offline: no substitute is served"
                ));
            }
        }
        let store = store.ok_or_else(|| anyhow!("blob {sha256} is not cached and offline"))?;
        store
            .download_verified(&blob_key(sha256), &dest, sha256, RESUME_THRESHOLD)
            .await?;
        Ok(true)
    }

    /// Every complete blob name (sorted); in-flight `.partial` files and the `.partial.lock`
    /// files that serialise writers (left in place on purpose) are skipped.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| !n.ends_with(".partial") && !n.ends_with(".partial.lock"))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    /// Re-hash every blob (or check sizes with `quick`).
    pub fn verify(&self, quick: bool) -> BlobVerify {
        let mut report = BlobVerify::default();
        for name in self.list() {
            report.blobs += 1;
            if !is_sha256_name(&name) {
                report
                    .errors
                    .push(format!("blob {name}: not a sha256 name"));
                continue;
            }
            let p = self.path(&name);
            if quick {
                if std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0) == 0 {
                    report.errors.push(format!("blob {name}: empty"));
                }
            } else {
                match digest_file(&p) {
                    Ok(d) if d.sha256 == name => {}
                    Ok(d) => report
                        .errors
                        .push(format!("blob {name}: content hashes to {}", d.sha256)),
                    Err(e) => report.errors.push(format!("blob {name}: {e}")),
                }
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_hex;

    #[test]
    fn store_verify_and_link() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = BlobDir::new(tmp.path().join("blobs"));
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hello").unwrap();
        let sha = sha256_hex(b"hello");
        assert!(blobs
            .store_file(&src, "0".repeat(64).as_str(), true)
            .is_err());
        assert!(blobs.list().is_empty(), "a failed store leaves no partial");
        let dest = blobs.store_file(&src, &sha, true).unwrap();
        assert_eq!(dest, blobs.path(&sha));
        let linked = tmp.path().join("out").join("x.bin");
        link_or_copy(&dest, &linked).unwrap();
        assert!(same_file(&dest, &linked));
        let v = blobs.verify(false);
        assert_eq!((v.blobs, v.errors.len()), (1, 0));
        std::fs::write(&dest, b"tampered").unwrap();
        let v = blobs.verify(false);
        assert_eq!(v.errors.len(), 1, "{:?}", v.errors);
    }

    #[test]
    fn concurrent_links_never_truncate_the_shared_inode() {
        let tmp = tempfile::tempdir().unwrap();
        let blob = tmp.path().join("blob");
        let payload = vec![7u8; 1 << 16];
        std::fs::write(&blob, &payload).unwrap();
        let dst = tmp.path().join("out").join("linked");
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::write(&dst, b"stale").unwrap();
        std::thread::scope(|s| {
            for _ in 0..16 {
                s.spawn(|| link_or_copy(&blob, &dst).unwrap());
            }
        });
        assert!(same_file(&blob, &dst));
        assert_eq!(std::fs::read(&blob).unwrap(), payload);
        assert!(std::fs::read_dir(dst.parent().unwrap())
            .unwrap()
            .all(|e| e.unwrap().file_name() == "linked"));
    }

    #[test]
    fn ensure_downloads_and_rejects_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        std::fs::create_dir_all(store_dir.join("blobs/sha256")).unwrap();
        let sha = sha256_hex(b"payload");
        std::fs::write(store_dir.join(blob_key(&sha)), b"payload").unwrap();
        let bad = "1".repeat(64);
        std::fs::write(store_dir.join(blob_key(&bad)), b"corrupt").unwrap();
        let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();
        let blobs = BlobDir::new(tmp.path().join("cache"));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        assert!(rt.block_on(blobs.ensure(Some(&store), &sha, 7)).unwrap());
        assert!(!rt.block_on(blobs.ensure(Some(&store), &sha, 7)).unwrap());
        assert!(rt.block_on(blobs.ensure(None, &sha, 7)).is_ok());
        let err = rt
            .block_on(blobs.ensure(Some(&store), &bad, 7))
            .unwrap_err()
            .to_string();
        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(!blobs.path(&bad).exists());
        assert!(!blobs.partial_path(&bad).exists());
        assert!(rt.block_on(blobs.ensure(None, &bad, 7)).is_err());
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;

    /// Eight concurrent `ensure` calls for one blob (threads and a second process would race the
    /// same way) must produce exactly one correct blob and no doubled content or leftover partial.
    #[test]
    fn concurrent_ensure_of_one_blob_is_serialised() {
        let tmp = tempfile::tempdir().unwrap();
        let src_root = tmp.path().join("store");
        std::fs::create_dir_all(src_root.join("blobs/sha256")).unwrap();
        let payload: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let sha = crate::hash::sha256_hex(&payload);
        std::fs::write(src_root.join("blobs/sha256").join(&sha), &payload).unwrap();
        let store = Store::open(&format!("file://{}", src_root.display())).unwrap();
        let blobs = BlobDir::new(tmp.path().join("cache"));
        std::fs::create_dir_all(tmp.path().join("cache")).unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8)
            .enable_all()
            .build()
            .unwrap();
        let results = rt.block_on(async {
            let tasks: Vec<_> = (0..8)
                .map(|_| {
                    let store = &store;
                    let blobs = &blobs;
                    let sha = sha.clone();
                    let len = payload.len() as u64;
                    async move { blobs.ensure(Some(store), &sha, len).await }
                })
                .collect();
            futures_util::future::join_all(tasks).await
        });
        for r in &results {
            r.as_ref().unwrap();
        }
        let out = std::fs::read(blobs.path(&sha)).unwrap();
        assert_eq!(
            out.len(),
            payload.len(),
            "blob must not be doubled or truncated"
        );
        assert_eq!(crate::hash::sha256_hex(&out), sha);
        assert!(!blobs.partial_path(&sha).exists());
        assert_eq!(blobs.list(), vec![sha]);
    }
}
