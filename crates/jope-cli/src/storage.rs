use anyhow::{bail, ensure, Result};
use std::path::{Path, PathBuf};

/// Runtime resolution follows the explicit-root/ancestor approach reviewed in PR #13.
/// There is no embedded build directory or fallback to a developer's home directory.
pub fn resolve_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(root) = explicit {
        return Ok(root);
    }
    if let Some(root) = std::env::var_os("JOPE_INPUTS_ROOT").filter(|value| !value.is_empty()) {
        return Ok(root.into());
    }
    if let Some(root) = std::env::var_os("JOPE_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(root).join("datasets/synthetic"));
    }
    let current = std::env::current_dir()?;
    let executable = std::env::current_exe()?;
    for start in [Some(current.as_path()), executable.parent()]
        .into_iter()
        .flatten()
    {
        for ancestor in start.ancestors() {
            let candidate = ancestor.join("datasets/synthetic");
            if candidate.join("catalog.json").is_file() {
                return Ok(candidate);
            }
        }
    }
    bail!("fixture catalog not found; set --root or JOPE_INPUTS_ROOT to its directory")
}

pub(crate) fn components(relative: &str) -> Result<Vec<&str>> {
    let parts: Vec<_> = relative.split('/').collect();
    ensure!(
        parts.iter().all(|part| !part.is_empty()
            && *part != "."
            && *part != ".."
            && !part.contains(['\\', ':', '\0'])),
        "invalid relative fixture path: {relative}"
    );
    Ok(parts)
}

#[cfg(unix)]
mod platform {
    use super::*;
    use anyhow::Context;
    use rustix::fs::{mkdirat, open, openat, Dir, Mode, OFlags};
    use std::{
        collections::BTreeSet,
        fs::File,
        io::{Read, Write},
    };

    pub(crate) struct Directory(File);

    impl Directory {
        pub fn open(path: &Path) -> Result<Self> {
            // Remove trailing '/' and '/.' before O_NOFOLLOW: otherwise the kernel can resolve
            // a final directory symlink while interpreting the trailing directory separator.
            let path: PathBuf = path.components().collect();
            let fd = open(
                &path,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .with_context(|| format!("open fixture root {} without symlinks", path.display()))?;
            Ok(Self(File::from(fd)))
        }

        fn child(&self, name: &str) -> Result<Self> {
            let fd = openat(
                &self.0,
                name,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .with_context(|| format!("open fixture directory {name} without symlinks"))?;
            Ok(Self(File::from(fd)))
        }

        pub fn read(&self, relative: &str, limit: u64) -> Result<Vec<u8>> {
            let parts = components(relative)?;
            let (name, parents) = parts.split_last().context("empty fixture path")?;
            let mut directory = Self(self.0.try_clone()?);
            for parent in parents {
                directory = directory.child(parent)?;
            }
            let file = File::from(
                openat(
                    &directory.0,
                    *name,
                    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .with_context(|| format!("open fixture {relative} without symlinks"))?,
            );
            let metadata = file.metadata()?;
            ensure!(
                metadata.is_file(),
                "fixture must be a regular file: {relative}"
            );
            ensure!(
                metadata.len() <= limit,
                "fixture exceeds byte limit: {relative}"
            );
            let mut bytes = Vec::new();
            file.take(limit + 1).read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() as u64 <= limit,
                "fixture grew beyond byte limit: {relative}"
            );
            // This one bounded buffer is both hashed and deserialized by admission.
            Ok(bytes)
        }

        pub fn names(&self) -> Result<BTreeSet<String>> {
            let mut names = BTreeSet::new();
            for entry in Dir::read_from(&self.0)? {
                let entry = entry?;
                let name = entry
                    .file_name()
                    .to_str()
                    .context("non-UTF8 fixture filename")?;
                if !matches!(name, "." | "..") {
                    names.insert(name.into());
                }
                ensure!(names.len() <= 64, "too many fixture directory entries");
            }
            Ok(names)
        }

        pub fn child_names(&self, name: &str) -> Result<BTreeSet<String>> {
            self.child(name)?.names()
        }

        pub fn create_child(&self, name: &str) -> Result<()> {
            ensure!(
                components(name)?.len() == 1,
                "expected one directory component"
            );
            mkdirat(&self.0, name, Mode::from_raw_mode(0o755))?;
            Ok(())
        }

        pub fn write_new(&self, relative: &str, bytes: &[u8]) -> Result<()> {
            let parts = components(relative)?;
            let (name, parents) = parts.split_last().context("empty fixture path")?;
            let mut directory = Self(self.0.try_clone()?);
            for parent in parents {
                directory = directory.child(parent)?;
            }
            let mut file = File::from(openat(
                &directory.0,
                *name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o644),
            )?);
            file.write_all(bytes)?;
            Ok(())
        }
    }
}

#[cfg(not(unix))]
mod platform {
    use super::*;
    use std::collections::BTreeSet;

    pub(crate) struct Directory;

    impl Directory {
        pub fn open(_: &Path) -> Result<Self> {
            bail!("offline fixture qualification requires a supported Unix host")
        }
        pub fn read(&self, _: &str, _: u64) -> Result<Vec<u8>> {
            bail!("unsupported host")
        }
        pub fn names(&self) -> Result<BTreeSet<String>> {
            bail!("unsupported host")
        }
        pub fn child_names(&self, _: &str) -> Result<BTreeSet<String>> {
            bail!("unsupported host")
        }
        pub fn create_child(&self, _: &str) -> Result<()> {
            bail!("unsupported host")
        }
        pub fn write_new(&self, _: &str, _: &[u8]) -> Result<()> {
            bail!("unsupported host")
        }
    }
}

pub(crate) use platform::Directory;
