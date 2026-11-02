use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]

pub(crate) struct RatchetArtifactPointerV1 {
    pub(crate) schema: String,
    pub(crate) artifact_sha256: String,
    pub(crate) updated_at_unix_seconds: u64,
    pub(crate) serving_authority: bool,
}

impl RatchetArtifactPointerV1 {
    pub(crate) fn validate(&self, serving_authority: bool) -> Result<()> {
        if self.schema != POINTER_SCHEMA || self.serving_authority != serving_authority {
            bail!("ratchet pointer has the wrong authority class");
        }
        validate_sha256("ratchet pointer artifact", &self.artifact_sha256)
    }
}

pub(crate) struct RatchetStore {
    root: PathBuf,
}

impl RatchetStore {
    pub(crate) fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

mod serving_state;
mod training_head;

pub(crate) struct WriterGuard {
    path: PathBuf,
}

impl WriterGuard {
    fn acquire(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)?;
        let path = root.join(".ratchet-writer.lock");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .context("another Rust ratchet writer is active or a fault lock requires recovery")?;
        writeln!(file, "{}", std::process::id())?;
        file.sync_all()?;
        Ok(Self { path })
    }
}

impl Drop for WriterGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(crate) fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(path)? == bytes {
                Ok(())
            } else {
                bail!("immutable artifact already exists with different bytes");
            }
        }
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn atomic_pointer_write(path: &Path, pointer: &RatchetArtifactPointerV1) -> Result<()> {
    atomic_bytes_write(path, &canonical_json(pointer)?)
}

pub(crate) fn atomic_bytes_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic pointer path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .context("atomic pointer filename is not UTF-8")?,
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create atomic temporary {}", temporary.display()))?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
