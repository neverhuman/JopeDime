//! `JOPE_HOME`, the dataset cache layout, and a dependency-free RFC 3339 clock.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// `$JOPE_HOME`, else `${XDG_CACHE_HOME:-~/.cache}/jope`.
pub fn jope_home() -> PathBuf {
    if let Some(h) = std::env::var_os("JOPE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(h);
    }
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    cache.join("jope")
}

/// `JOPE_OFFLINE=1` (or any non-empty value other than `0`).
pub fn offline_env() -> bool {
    std::env::var("JOPE_OFFLINE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// The dataset cache under `$JOPE_HOME/datasets`.
#[derive(Clone, Debug)]
pub struct Cache {
    pub root: PathBuf,
}

impl Cache {
    pub fn from_env() -> Cache {
        Cache {
            root: jope_home().join("datasets"),
        }
    }

    pub fn at(root: impl Into<PathBuf>) -> Cache {
        Cache { root: root.into() }
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs").join("sha256")
    }

    pub fn blob_path(&self, sha256: &str) -> PathBuf {
        self.blobs_dir().join(sha256)
    }

    /// `JOPE_REGISTRY` if set, else `<cache>/by-id`.
    pub fn registry_dir(&self) -> PathBuf {
        std::env::var_os("JOPE_REGISTRY")
            .filter(|v| !v.is_empty())
            .map_or_else(|| self.root.join("by-id"), PathBuf::from)
    }

    pub fn by_id_dir(&self, dataset_hash: &str) -> PathBuf {
        self.registry_dir().join(dataset_hash)
    }

    pub fn catalog_dir(&self) -> PathBuf {
        self.root.join("catalog")
    }

    pub fn collections_dir(&self) -> PathBuf {
        self.root.join("collections")
    }

    pub fn ingest_dir(&self) -> PathBuf {
        self.root.join("ingest")
    }
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rfc3339_from_unix(secs)
}

/// A compact timestamp for directory names: `YYYYMMDDTHHMMSSZ`.
pub fn now_compact() -> String {
    now_rfc3339()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// Civil-from-days (Howard Hinnant) without a calendar dependency.
pub fn rfc3339_from_unix(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Relative path of `path` below `root`, with `/` separators.
pub fn relpath(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339() {
        assert_eq!(rfc3339_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_unix(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_789_066_935), "2026-09-10T19:02:15Z");
    }

    #[test]
    fn relpaths() {
        assert_eq!(relpath(Path::new("/a/b"), Path::new("/a/b/c/d")), "c/d");
    }
}
