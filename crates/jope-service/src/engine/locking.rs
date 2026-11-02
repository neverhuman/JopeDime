use super::*;

/// Failure to own the engine or to start/stop its local listener.
#[derive(Debug)]
pub enum EngineError {
    Io(io::Error),
    Store(crate::store::StoreError),
    Busy { pid: Option<u32>, command: String },
    Daemon(String),
    InvalidIdleExit,
    Runtime(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "engine IO: {error}"),
            Self::Store(error) => write!(f, "engine store: {error}"),
            Self::Busy { pid, command } => write!(
                f,
                "ENGINE_BUSY: pid={} command={}",
                pid.map_or_else(|| "unknown".to_owned(), |pid| pid.to_string()),
                if command.is_empty() {
                    "unknown"
                } else {
                    command
                }
            ),
            Self::Daemon(message) => write!(f, "engine daemon: {message}"),
            Self::InvalidIdleExit => {
                f.write_str("idle exit must be a positive duration (for example 30m or 2s)")
            }
            Self::Runtime(message) => write!(f, "engine runtime: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<io::Error> for EngineError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<crate::store::StoreError> for EngineError {
    fn from(error: crate::store::StoreError) -> Self {
        Self::Store(error)
    }
}

/// A live process's OS advisory lock. Dropping the file releases ownership.
pub struct EngineLock {
    file: File,
}

impl std::fmt::Debug for EngineLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineLock").finish_non_exhaustive()
    }
}

impl EngineLock {
    pub fn acquire(home: impl AsRef<Path>) -> Result<Self, EngineError> {
        let home = home.as_ref();
        if !home.exists() {
            fs::create_dir_all(home)?;
        }
        if !fs::symlink_metadata(home)?.file_type().is_dir() {
            return Err(EngineError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "engine home is not a real directory",
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(home, fs::Permissions::from_mode(0o700))?;
        }
        let path = home.join("lock");
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if !metadata.file_type().is_file() {
                return Err(EngineError::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "engine lock is not a regular file",
                )));
            }
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(&path)?;
        if let Err(error) = file.try_lock() {
            let mut holder = read_holder(&path);
            for _ in 0..20 {
                if holder.0.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
                holder = read_holder(&path);
            }
            return match error {
                std::fs::TryLockError::WouldBlock => Err(EngineError::Busy {
                    pid: holder.0,
                    command: holder.1,
                }),
                std::fs::TryLockError::Error(error) => Err(EngineError::Io(error)),
            };
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        writeln!(file, "{}", std::process::id())?;
        writeln!(file, "{}", process_command())?;
        file.sync_all()?;
        Ok(Self { file })
    }

    pub fn is_held(&self) -> bool {
        self.file.metadata().is_ok()
    }
}

fn process_command() -> String {
    let command = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    command
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(1024)
        .collect()
}

fn read_holder(path: &Path) -> (Option<u32>, String) {
    let mut text = String::new();
    if File::open(path)
        .and_then(|file| file.take(4096).read_to_string(&mut text))
        .is_err()
    {
        return (None, String::new());
    }
    let mut lines = text.lines();
    (
        lines.next().and_then(|line| line.parse().ok()),
        lines.next().unwrap_or_default().to_owned(),
    )
}
