use super::*;

/// JSONL strings retain multiline entries. The client supplies the canonical
/// home path; this component does not guess a second home-resolution policy.
pub struct HistoryFile {
    path: PathBuf,
    entries: Vec<String>,
}
impl HistoryFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut history = Self {
            path: path.into(),
            entries: Vec::new(),
        };
        history.push_inner(None)?;
        Ok(history)
    }
    pub fn entries(&self) -> &[String] {
        &self.entries
    }
    pub fn push(&mut self, text: &str) -> io::Result<()> {
        if !remember(text) {
            return Ok(());
        }
        self.push_inner(Some(text))
    }
    fn push_inner(&mut self, text: Option<&str>) -> io::Result<()> {
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let mut lock_name = self.path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let mut options = regular_options();
        options.write(true).create(true);
        let lock = options.open(PathBuf::from(lock_name))?;
        regular(&lock)?;
        acquire_history_lock(&lock)?;
        let mut entries = match regular_options().open(&self.path) {
            Ok(file) => {
                regular(&file)?;
                if file.metadata()?.len() > MAX_HISTORY as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "history exceeds 16 MiB",
                    ));
                }
                let mut bytes = Vec::new();
                file.take(MAX_HISTORY as u64 + 1).read_to_end(&mut bytes)?;
                if bytes.len() > MAX_HISTORY {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "history exceeds 16 MiB",
                    ));
                }
                let content = std::str::from_utf8(&bytes).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "history is not UTF-8")
                })?;
                let mut entries = Vec::new();
                for line in content.lines() {
                    let entry: String = serde_json::from_str(line).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid history record")
                    })?;
                    if remember(&entry) {
                        entries.push(clean(&entry));
                    }
                }
                entries
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        if let Some(text) = text {
            let text = clean(text);
            if entries.last() != Some(&text) {
                entries.push(text);
            }
        }
        if entries.len() > HISTORY_ENTRIES {
            entries.drain(..entries.len() - HISTORY_ENTRIES);
        }
        let mut records: Vec<_> = entries
            .iter()
            .map(|s| serde_json::to_string(s).expect("string serialization") + "\n")
            .collect();
        let mut size: usize = records.iter().map(String::len).sum();
        let mut drop_count = 0;
        while size > MAX_HISTORY {
            size -= records[drop_count].len();
            drop_count += 1;
        }
        entries.drain(..drop_count);
        records.drain(..drop_count);
        let id = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = format!(".history-{}-{id}.tmp", std::process::id());
        let temporary = parent.join(name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        owner_mode(&mut options);
        let mut file = options.open(&temporary)?;
        let published: io::Result<()> = (|| {
            for line in records {
                file.write_all(line.as_bytes())?;
            }
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &self.path)?;
            #[cfg(unix)]
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if published.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        published?;
        self.entries = entries;
        Ok(())
    }
}
pub(super) fn remember(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() || text.len() > MAX_DRAFT {
        return false;
    }
    let command = text.trim_start_matches(['\\', '/']);
    let name = command.split_whitespace().next().unwrap_or("");
    if name.eq_ignore_ascii_case("login") {
        return false;
    }
    if name.eq_ignore_ascii_case("objective")
        || name.eq_ignore_ascii_case("fitness")
        || text.starts_with('{')
    {
        if let Some(start) = text.find('{') {
            if text[start..].len() > 1024 {
                return false;
            }
        }
    }
    true
}
pub(super) fn owner_mode(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(not(unix))]
    let _ = options;
}
pub(super) fn regular_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true);
    owner_mode(&mut options);
    // OS ABI flags: refuse symlinks/reparse points and avoid blocking on FIFOs.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x8000 | 0x800);
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    options
}
fn regular(file: &File) -> io::Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "history must be a regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "history must have mode 0600",
            ));
        }
    }
    Ok(())
}
