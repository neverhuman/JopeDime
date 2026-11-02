use super::*;

pub(super) struct SingleInstanceLock {
    _file: File,
}

impl SingleInstanceLock {
    pub(super) fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        if flock::exclusive(&file, false).is_err() {
            bail!(
                "another micro-wave supply supervisor owns {}",
                path.display()
            );
        }
        Ok(Self { _file: file })
    }
}
