enum AssetReader<'a> {
    Embedded(&'a [EmbeddedAsset<'a>]),
    Directory(DirectoryReader),
}

impl<'a> AssetReader<'a> {
    fn new(input: BundleInput<'a>, expected: &BTreeMap<String, ReleaseAsset>) -> Result<Self> {
        match input {
            BundleInput::Embedded(assets) => {
                let paths: BTreeSet<_> = assets.iter().map(|a| a.path).collect();
                if paths.len() != assets.len()
                    || paths.len() != expected.len()
                    || paths.iter().any(|p| !expected.contains_key(*p))
                {
                    return Err(invalid(
                        "embedded asset inventory differs from release catalog",
                    ));
                }
                Ok(Self::Embedded(assets))
            }
            BundleInput::DevelopmentDirectory(root) => {
                Ok(Self::Directory(DirectoryReader::open(root)?))
            }
        }
    }

    fn read(&self, path: &str, identity: AssetIdentity, limits: BundleLimits) -> Result<Vec<u8>> {
        match self {
            Self::Embedded(assets) => {
                let asset = assets
                    .iter()
                    .find(|asset| asset.path == path)
                    .ok_or_else(|| invalid("missing embedded asset"))?;
                if asset.bytes.len() as u64 != identity.bytes {
                    return Err(invalid("embedded asset size mismatch"));
                }
                read_hashed_asset(asset.bytes, identity, limits)
            }
            Self::Directory(directory) => {
                let file = directory.open_asset(path)?;
                let metadata = file
                    .metadata()
                    .map_err(|_| invalid("cannot stat opened asset"))?;
                if !metadata.is_file() || metadata.len() != identity.bytes {
                    return Err(invalid("asset is nonregular or has an unexpected size"));
                }
                read_hashed_asset(file, identity, limits)
            }
        }
    }
}

#[cfg(unix)]
struct DirectoryReader {
    root: File,
}

#[cfg(unix)]
impl DirectoryReader {
    fn open(path: &Path) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let root = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)
            .map_err(|_| invalid("cannot open bundle directory"))?;
        Ok(Self { root })
    }

    #[allow(unsafe_code)] // nix 0.29 returns RawFd; only the owned conversion below is unsafe.
    fn open_asset(&self, path: &str) -> Result<File> {
        use nix::{
            fcntl::{openat, OFlag},
            sys::stat::Mode,
        };
        use std::os::fd::{AsRawFd, FromRawFd};
        let mut parent = self
            .root
            .try_clone()
            .map_err(|_| invalid("cannot retain bundle directory"))?;
        let mut parts = path.split('/').peekable();
        while let Some(part) = parts.next() {
            let mut flags =
                OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_NONBLOCK | OFlag::O_CLOEXEC;
            if parts.peek().is_some() {
                flags |= OFlag::O_DIRECTORY;
            }
            let fd = openat(Some(parent.as_raw_fd()), part, flags, Mode::empty())
                .map_err(|_| invalid("cannot open regular bundle asset without links"))?;
            // SAFETY: openat returned a new owned descriptor on success. It is
            // transferred exactly once to File; parent stays live through openat.
            parent = unsafe { File::from_raw_fd(fd) };
        }
        Ok(parent)
    }
}

#[cfg(windows)]
struct DirectoryReader {
    root: std::path::PathBuf,
    pins: Vec<File>,
}

#[cfg(windows)]
impl DirectoryReader {
    fn open(path: &Path) -> Result<Self> {
        // Canonical root is caller-selected; pin every ancestor without delete
        // sharing before walking asset paths, so directories cannot be swapped.
        let selected = Self::open_path(path)?;
        if !selected
            .metadata()
            .map_err(|_| invalid("cannot stat bundle directory"))?
            .is_dir()
        {
            return Err(invalid("bundle root is not a directory"));
        }
        let root =
            std::fs::canonicalize(path).map_err(|_| invalid("cannot resolve bundle directory"))?;
        let mut current = std::path::PathBuf::new();
        let mut pins = vec![selected];
        for part in root.components() {
            current.push(part);
            if matches!(part, std::path::Component::Prefix(_)) {
                continue;
            }
            let file = Self::open_path(&current)?;
            if !file
                .metadata()
                .map_err(|_| invalid("cannot stat directory"))?
                .is_dir()
            {
                return Err(invalid("bundle ancestor is not a directory"));
            }
            pins.push(file);
        }
        Ok(Self { root, pins })
    }

    fn open_path(path: &Path) -> Result<File> {
        use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
        // BACKUP_SEMANTICS | OPEN_REPARSE_POINT; FILE_SHARE_READ | WRITE only.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(3)
            .custom_flags(0x0200_0000 | 0x0020_0000)
            .open(path)
            .map_err(|_| invalid("cannot open bundle asset"))?;
        if file
            .metadata()
            .map_err(|_| invalid("cannot stat opened asset"))?
            .file_attributes()
            & 0x400
            != 0
        {
            return Err(invalid("bundle reparse points are refused"));
        }
        Ok(file)
    }

    fn open_asset(&self, path: &str) -> Result<File> {
        let _root_pins = &self.pins;
        let mut current = self.root.clone();
        let mut pins = Vec::new();
        let mut parts = path.split('/').peekable();
        while let Some(part) = parts.next() {
            current.push(part);
            let file = Self::open_path(&current)?;
            if parts.peek().is_none() {
                return Ok(file);
            }
            if !file
                .metadata()
                .map_err(|_| invalid("cannot stat asset parent"))?
                .is_dir()
            {
                return Err(invalid("asset parent is not a directory"));
            }
            pins.push(file);
        }
        Err(invalid("empty bundle asset path"))
    }
}

#[cfg(not(any(unix, windows)))]
struct DirectoryReader;
#[cfg(not(any(unix, windows)))]
impl DirectoryReader {
    fn open(_: &Path) -> Result<Self> {
        Err(invalid(
            "development directories unsupported on this platform",
        ))
    }
    fn open_asset(&self, _: &str) -> Result<File> {
        Err(invalid(
            "development directories unsupported on this platform",
        ))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integrity_tests;
