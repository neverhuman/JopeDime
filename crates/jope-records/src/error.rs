//! The one error type every verifier returns.

use std::fmt;
use std::path::{Path, PathBuf};

/// A verifier failure.
///
/// [`Error::Invalid`] is the port of the Python verifiers' `ValueError` raised by `require`; its
/// message is the rule's message, byte for byte. The other variants cover what Python raised as
/// other exception types.
#[derive(Debug)]
pub enum Error {
    /// A record failed a validation rule (Python: `ValueError` from `require`).
    Invalid(String),
    /// Bytes are not valid UTF-8 or JSON (Python: `UnicodeDecodeError` / `JSONDecodeError`, both
    /// `ValueError` subclasses).
    Parse { path: PathBuf, message: String },
    /// A file could not be read, written or inspected (Python: `OSError`).
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A record lacks the shape a verifier dereferences (Python: `KeyError`, `TypeError` or
    /// `IndexError`).
    Shape(String),
    /// An external command failed (Python: `subprocess.CalledProcessError`).
    Command(String),
    /// A retained input exists but cannot be decoded on this host; it is never counted as
    /// verified (Python: `elo_history.Unavailable`).
    Unavailable(String),
}

impl Error {
    /// Whether Python would have raised a `ValueError` (the class the Python tests asserted).
    pub fn is_value_error(&self) -> bool {
        matches!(self, Error::Invalid(_) | Error::Parse { .. })
    }

    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    pub(crate) fn parse(path: &Path, message: impl Into<String>) -> Self {
        Error::Parse {
            path: path.to_path_buf(),
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(message)
            | Error::Shape(message)
            | Error::Command(message)
            | Error::Unavailable(message) => f.write_str(message),
            Error::Parse { path, message } => write!(f, "{}: {message}", path.display()),
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// `require(condition, message)`: fail with [`Error::Invalid`] carrying the formatted message.
/// The message is formatted only on failure.
#[macro_export]
macro_rules! require {
    ($condition:expr, $($message:tt)+) => {
        if !$condition {
            return Err($crate::Error::Invalid(format!($($message)+)));
        }
    };
}
