//! Bounded descriptor-relative files for the host-selected local snapshot store.
use anyhow::{ensure, Context, Result};
use rustix::fs::{linkat, mkdirat, open, openat, renameat, unlinkat, AtFlags, Dir, Mode, OFlags};
use rustix::io::Errno;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
const MAX_BYTES: u64 = 16 * 1024 * 1024;
const DIR_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

pub(super) struct LocalFiles {
    path: PathBuf,
    directory: File,
}
impl LocalFiles {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let directory = File::from(
            open(path, DIR_FLAGS, Mode::empty())
                .context("local snapshot root must exist as a directory without symlinks")?,
        );
        let out = Self {
            path: fs::canonicalize(path)?,
            directory,
        };
        out.check_root()?;
        for relative in [
            "selfplay",
            "selfplay/authority",
            "selfplay/snapshots",
            "selfplay/candidate-states",
        ] {
            let mut parent = out.directory.try_clone()?;
            for name in Path::new(relative).components() {
                let Component::Normal(name) = name else {
                    anyhow::bail!("invalid local directory");
                };
                let fd = match openat(&parent, name, DIR_FLAGS, Mode::empty()) {
                    Ok(fd) => fd,
                    Err(Errno::NOENT) => {
                        match mkdirat(&parent, name, Mode::from_raw_mode(0o700)) {
                            Ok(()) | Err(Errno::EXIST) => {}
                            Err(error) => return Err(error.into()),
                        }
                        parent.sync_all()?;
                        openat(&parent, name, DIR_FLAGS, Mode::empty())?
                    }
                    Err(error) => {
                        return Err(error).context("local snapshot directory must not be a symlink")
                    }
                };
                parent = File::from(fd);
            }
        }
        Ok(out)
    }
    fn check_root(&self) -> Result<()> {
        let named = fs::symlink_metadata(&self.path)?;
        ensure!(
            named.is_dir() && !named.file_type().is_symlink(),
            "local snapshot root changed"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let held = self.directory.metadata()?;
            ensure!(
                held.dev() == named.dev() && held.ino() == named.ino(),
                "local snapshot root changed"
            );
        }
        Ok(())
    }
    fn parent(&self, relative: &str) -> Result<(File, String)> {
        self.check_root()?;
        let parts = Path::new(relative).components().collect::<Vec<_>>();
        ensure!(!parts.is_empty(), "empty local snapshot path");
        let mut directory = self.directory.try_clone()?;
        for (index, part) in parts.iter().enumerate() {
            let Component::Normal(name) = part else {
                anyhow::bail!("invalid local snapshot path");
            };
            if index + 1 == parts.len() {
                return Ok((
                    directory,
                    name.to_str()
                        .context("non-UTF8 local snapshot path")?
                        .into(),
                ));
            }
            directory = File::from(
                openat(&directory, *name, DIR_FLAGS, Mode::empty())
                    .context("local snapshot parent must not be a symlink")?,
            );
        }
        unreachable!("nonempty local path has a final component")
    }
    pub(super) fn has_results(&self) -> Result<bool> {
        let (directory, _) = self.parent("selfplay/candidate-states/probe")?;
        for entry in Dir::read_from(&directory)? {
            let entry = entry?;
            if !matches!(entry.file_name().to_bytes(), b"." | b"..") {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(super) fn lock(&self) -> Result<File> {
        let (directory, name) = self.parent("selfplay/authority/.lock")?;
        let file = File::from(
            openat(
                &directory,
                name.as_str(),
                OFlags::RDWR
                    | OFlags::CREATE
                    | OFlags::NOFOLLOW
                    | OFlags::CLOEXEC
                    | OFlags::NONBLOCK,
                Mode::from_raw_mode(0o600),
            )
            .context("open regular local snapshot process lock")?,
        );
        ensure!(
            file.metadata()?.is_file(),
            "local snapshot lock is not regular"
        );
        file.lock()?;
        self.check_root()?;
        Ok(file)
    }
    pub(super) fn read(&self, relative: &str, max_bytes: u64) -> Result<Option<Vec<u8>>> {
        let (parent, name) = self.parent(relative)?;
        let fd = match openat(
            &parent,
            name.as_str(),
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(Errno::NOENT) => return Ok(None),
            Err(error) => {
                return Err(error).context("read local snapshot without following symlinks")
            }
        };
        let file = File::from(fd);
        let meta = file.metadata()?;
        ensure!(
            meta.is_file() && meta.len() <= max_bytes,
            "local snapshot input type/size differs"
        );
        let mut bytes = Vec::with_capacity(usize::try_from(meta.len())?);
        file.take(max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= max_bytes,
            "local snapshot input grew beyond bound"
        );
        Ok(Some(bytes))
    }
    pub(super) fn write(&self, relative: &str, bytes: &[u8], immutable: bool) -> Result<()> {
        ensure!(
            bytes.len() as u64 <= MAX_BYTES,
            "local snapshot output exceeds bound"
        );
        if let Some(existing) = self.read(relative, MAX_BYTES)? {
            if immutable {
                ensure!(existing == bytes, "immutable local snapshot differs");
                return Ok(());
            }
        }
        let (parent, name) = self.parent(relative)?;
        let temp = format!(
            ".{name}.tmp-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        );
        let mut file = File::from(openat(
            &parent,
            temp.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )?);
        file.write_all(bytes)?;
        file.sync_all()?;
        if immutable {
            // A hard link never replaces an existing immutable target, including a symlink.
            match linkat(
                &parent,
                temp.as_str(),
                &parent,
                name.as_str(),
                AtFlags::empty(),
            ) {
                Ok(()) => {}
                Err(Errno::EXIST) => ensure!(
                    self.read(relative, MAX_BYTES)?.as_deref() == Some(bytes),
                    "immutable local snapshot appeared with different bytes"
                ),
                Err(error) => return Err(error.into()),
            }
            unlinkat(&parent, temp.as_str(), AtFlags::empty())?;
        } else {
            renameat(&parent, temp.as_str(), &parent, name.as_str())?;
        }
        parent.sync_all()?;
        self.check_root()
    }
}
