//! Descriptor-relative release cache; publication never replaces an existing entry.
use super::*;
use crate::cache::ModelsCache;
use std::path::PathBuf;

impl VerifiedReleasePayloads {
    /// Install complete verified bytes at `models/releases/<release_id>`.
    /// Existing versions are reverified, never repaired or overwritten. Failed staging
    /// is retained for explicit owner recovery. This path is not runtime admission.
    pub fn install_cache(&self, cache: &ModelsCache) -> Result<PathBuf> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            disk::install(self, cache, |_| Ok(()))
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = cache;
            bail!("atomic release cache unsupported on this platform")
        }
    }

    #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
    pub(super) fn install_cache_with_hook<F>(&self, cache: &ModelsCache, hook: F) -> Result<PathBuf>
    where
        F: FnOnce(&std::path::Path) -> Result<()>,
    {
        disk::install(self, cache, hook)
    }
}

impl VerifiedReleaseManifest {
    /// Re-open and hash the declared local closure into private immutable buffers.
    /// This verifies local bytes, not fresh remote versions, rights or a runtime.
    pub fn load_cache(
        &self,
        cache: &ModelsCache,
        limits: PayloadLimits,
    ) -> Result<VerifiedReleasePayloads> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            disk::load(self, cache, limits)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = (cache, limits);
            bail!("atomic release cache unsupported on this platform")
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) mod disk {
    use super::*;
    use rustix::fs::{mkdirat, open, openat, renameat_with, Mode, OFlags, RenameFlags};
    use rustix::io::Errno;
    use std::ffi::OsStr;
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::unix::fs::MetadataExt;
    use std::path::{Component, Path};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

    fn directory_flags() -> OFlags {
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
    }

    fn open_directory(parent: &File, name: &OsStr) -> rustix::io::Result<File> {
        openat(parent, name, directory_flags(), Mode::empty()).map(File::from)
    }

    fn directory(parent: &File, name: &OsStr, create: bool) -> Result<File> {
        if create {
            match mkdirat(parent, name, Mode::from_raw_mode(0o700)) {
                Ok(()) | Err(Errno::EXIST) => {}
                Err(e) => return Err(e.into()),
            }
        }
        open_directory(parent, name).with_context(|| format!("open non-symlink directory {name:?}"))
    }

    // Every component is opened relative to a held directory descriptor. No ancestor
    // is subsequently resolved again through a mutable string path.
    fn root(path: &Path, create: bool) -> Result<File> {
        ensure!(path.is_absolute(), "release cache root must be absolute");
        ensure!(
            path.components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
            "release cache root contains a parent or platform prefix"
        );
        let mut current = File::from(open("/", directory_flags(), Mode::empty())?);
        for part in path.components() {
            if let Component::Normal(name) = part {
                current = directory(&current, name, create)?;
            }
        }
        Ok(current)
    }

    fn open_regular(
        parent: &File,
        name: &str,
        maximum: u64,
        expected: Option<u64>,
    ) -> Result<File> {
        ensure!(
            maximum > 0 && maximum <= isize::MAX as u64,
            "invalid cache read bound"
        );
        // NONBLOCK ensures a substituted FIFO cannot block before fstat rejects it.
        let file = File::from(openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )?);
        let stat = file.metadata()?;
        ensure!(stat.is_file(), "cache entry {name} is not a regular file");
        ensure!(
            stat.len() <= maximum && expected.is_none_or(|n| stat.len() == n),
            "cache entry {name} length differs from bound"
        );
        Ok(file)
    }

    fn read_regular(
        parent: &File,
        name: &str,
        maximum: u64,
        expected: Option<u64>,
    ) -> Result<Vec<u8>> {
        let mut file = open_regular(parent, name, maximum, expected)?;
        let mut bytes = Vec::new();
        (&mut file).take(maximum + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= maximum && expected.is_none_or(|n| bytes.len() as u64 == n),
            "cache entry {name} changed length during read"
        );
        Ok(bytes)
    }

    fn write_new(parent: &File, name: &str, bytes: &[u8]) -> Result<()> {
        let mut file = File::from(openat(
            parent,
            name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )?);
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    fn verify_directory(
        manifest: &VerifiedReleaseManifest,
        dir: &File,
        limits: PayloadLimits,
    ) -> Result<VerifiedReleasePayloads> {
        let bytes = read_regular(dir, "manifest.json", MAX_MANIFEST_BYTES as u64, None)?;
        ensure!(
            sha256_hex(&bytes) == manifest.manifest_sha256,
            "cached release manifest differs from compiled pin"
        );
        let bundles_dir = directory(dir, OsStr::new("bundles"), false)?;
        let mut bundles = BTreeMap::new();
        for c in &manifest.manifest.components {
            if !bundles.contains_key(&c.bundle_id) {
                bundles.insert(
                    c.bundle_id.clone(),
                    read_regular(
                        &bundles_dir,
                        &format!("{}.json", c.bundle_id),
                        MAX_MANIFEST_BYTES as u64,
                        None,
                    )?,
                );
            }
        }
        let payload_dir = directory(dir, OsStr::new("payloads"), false)?;
        manifest.verify_payloads(&bundles, limits, |c| {
            open_regular(&payload_dir, &c.id, c.bytes, Some(c.bytes))
        })
    }

    pub(super) fn load(
        manifest: &VerifiedReleaseManifest,
        cache: &ModelsCache,
        limits: PayloadLimits,
    ) -> Result<VerifiedReleasePayloads> {
        let root = root(&cache.root, false)?;
        let releases = directory(&root, OsStr::new("releases"), false)?;
        let dir = directory(&releases, OsStr::new(&manifest.manifest.release_id), false)?;
        verify_directory(manifest, &dir, limits)
    }

    pub(super) fn install<F>(
        release: &VerifiedReleasePayloads,
        cache: &ModelsCache,
        before_publish: F,
    ) -> Result<PathBuf>
    where
        F: FnOnce(&Path) -> Result<()>,
    {
        let root = root(&cache.root, true)?;
        let releases = directory(&root, OsStr::new("releases"), true)?;
        let id = &release.metadata.manifest.release_id;
        let destination = cache.root.join("releases").join(id);
        let limits = PayloadLimits {
            max_component_bytes: release
                .payloads
                .values()
                .map(|b| b.len() as u64)
                .max()
                .unwrap_or(1),
            max_release_bytes: release.verified_bytes,
        };
        match open_directory(&releases, OsStr::new(id)) {
            Ok(existing) => {
                verify_directory(&release.metadata, &existing, limits)?;
                return Ok(destination);
            }
            Err(Errno::NOENT) => {}
            Err(e) => return Err(e).context("existing release is not a safe directory"),
        }
        let mut stage = None;
        for _ in 0..8 {
            let name = format!(
                ".partial-{id}-{}-{}",
                std::process::id(),
                NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
            );
            match mkdirat(&releases, name.as_str(), Mode::from_raw_mode(0o700)) {
                Ok(()) => {
                    stage = Some(name);
                    break;
                }
                Err(Errno::EXIST) => {}
                Err(e) => return Err(e.into()),
            }
        }
        let stage_name = stage.context("could not reserve a unique release staging directory")?;
        let stage_path = cache.root.join("releases").join(&stage_name);
        // Deliberately no Drop cleanup: failed owned staging remains available for inspection.
        let dir = directory(&releases, OsStr::new(&stage_name), false)?;
        let bundles = directory(&dir, OsStr::new("bundles"), true)?;
        for (id, bytes) in &release.bundle_manifests {
            write_new(&bundles, &format!("{id}.json"), bytes)?;
        }
        bundles.sync_all()?;
        let payloads = directory(&dir, OsStr::new("payloads"), true)?;
        for (id, bytes) in &release.payloads {
            write_new(&payloads, id, bytes)?;
        }
        payloads.sync_all()?;
        write_new(&dir, "manifest.json", &release.metadata.manifest_bytes)?;
        dir.sync_all()?;
        before_publish(&stage_path)?;
        // Verify actual staged bytes before publication, not just the source buffers.
        verify_directory(&release.metadata, &dir, limits)?;
        // The verified descriptor may outlive a renamed staging entry. Refuse a
        // substituted name (including a symlink) before publishing that name.
        let named_stage = open_directory(&releases, OsStr::new(&stage_name))
            .context("staging name no longer names a non-symlink directory")?;
        let verified_identity = dir.metadata()?;
        let named_identity = named_stage.metadata()?;
        ensure!(
            verified_identity.dev() == named_identity.dev()
                && verified_identity.ino() == named_identity.ino(),
            "staging name no longer names the verified directory"
        );
        renameat_with(
            &releases,
            stage_name.as_str(),
            &releases,
            id.as_str(),
            RenameFlags::NOREPLACE,
        )
        .with_context(|| {
            format!(
                "publish release without replacement; staging retained at {}",
                stage_path.display()
            )
        })?;
        // Failure here means a complete version may already be visible. Never roll back
        // by deleting it; a retry reopens and verifies the existing complete closure.
        releases
            .sync_all()
            .context("complete release published but parent sync failed; reverify before use")?;
        Ok(destination)
    }
}
