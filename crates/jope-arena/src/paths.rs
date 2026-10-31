//! The crate's SOTArena-specific roots, alongside the shared `JOPE_HOME` and `JOPE_STATE`
//! resolution rules from `jope_paths`. The benchmark and reporter overrides remain distinct.

use std::env;
use std::path::{Path, PathBuf};

/// Marker file that identifies a `JOPE_HOME` checkout (committed empty at the repository root).
pub const HOME_MARKER: &str = ".jope-home";

/// Environment variable names.
pub mod var {
    /// The source checkout: assets, manifests, `benchmarks/`.
    pub const JOPE_HOME: &str = "JOPE_HOME";
    /// Mutable state (runs, pointers, the reporter data dir).
    pub const JOPE_STATE: &str = "JOPE_STATE";
    /// The benchmark clone (registry, tracks, results).
    pub const SOTARENA_ROOT: &str = "SOTARENA_ROOT";
    /// The Elo reporter's data dir (config.json, data/).
    pub const SOTARENA_REPORTER_ROOT: &str = "SOTARENA_REPORTER_ROOT";
    /// Where fetched benchmark datasets are cached.
    pub const SOTARENA_BENCHMARK_CACHE: &str = "SOTARENA_BENCHMARK_CACHE";
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn marker_ancestor(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(HOME_MARKER).is_file())
        .map(Path::to_path_buf)
}

/// `JOPE_HOME` when set; otherwise the nearest ancestor of the current directory, then of the
/// executable, holding a [`HOME_MARKER`] file; else the current directory.
pub fn jope_home() -> PathBuf {
    if let Some(home) = env_path(var::JOPE_HOME) {
        return home;
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if let Some(found) = marker_ancestor(&cwd) {
        return found;
    }
    if let Some(found) = env::current_exe()
        .ok()
        .and_then(|exe| marker_ancestor(&exe))
    {
        return found;
    }
    cwd
}

/// `JOPE_STATE` when set; otherwise `$XDG_STATE_HOME/jope`, else `$HOME/.local/state/jope`, else
/// `<jope_home>/state`.
pub fn jope_state() -> PathBuf {
    if let Some(state) = env_path(var::JOPE_STATE) {
        return state;
    }
    if let Some(xdg) = env_path("XDG_STATE_HOME") {
        return xdg.join("jope");
    }
    if let Some(home) = env_path("HOME") {
        return home.join(".local/state/jope");
    }
    jope_home().join("state")
}

/// `SOTARENA_ROOT` when set, otherwise `<jope_home>/benchmarks/sotarena` (the benchmark clone).
pub fn sotarena_root() -> PathBuf {
    env_path(var::SOTARENA_ROOT).unwrap_or_else(|| jope_home().join("benchmarks/sotarena"))
}

/// `SOTARENA_REPORTER_ROOT` when set, otherwise `<jope_state>/arena` (the Elo reporter's data).
pub fn sotarena_reporter_root() -> PathBuf {
    env_path(var::SOTARENA_REPORTER_ROOT).unwrap_or_else(|| jope_state().join("arena"))
}

/// `SOTARENA_BENCHMARK_CACHE` when set, otherwise `<jope_home>/benchmarks/cache`.
pub fn benchmark_cache() -> PathBuf {
    env_path(var::SOTARENA_BENCHMARK_CACHE).unwrap_or_else(|| jope_home().join("benchmarks/cache"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV: Mutex<()> = Mutex::new(());

    fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
        let _guard = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let saved: Vec<_> = pairs
            .iter()
            .map(|(name, _)| (*name, env::var_os(name)))
            .collect();
        for (name, value) in pairs {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
        let result = body();
        for (name, value) in saved {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
        result
    }

    #[test]
    fn roots_follow_home_and_state() {
        with_env(
            &[
                (var::JOPE_HOME, Some("/opt/jope/checkout")),
                (var::JOPE_STATE, Some("/var/lib/jope")),
                (var::SOTARENA_ROOT, None),
                (var::SOTARENA_REPORTER_ROOT, None),
                (var::SOTARENA_BENCHMARK_CACHE, None),
            ],
            || {
                assert_eq!(
                    sotarena_root(),
                    PathBuf::from("/opt/jope/checkout/benchmarks/sotarena")
                );
                assert_eq!(
                    sotarena_reporter_root(),
                    PathBuf::from("/var/lib/jope/arena")
                );
                assert_eq!(
                    benchmark_cache(),
                    PathBuf::from("/opt/jope/checkout/benchmarks/cache")
                );
            },
        );
    }

    #[test]
    fn explicit_variables_win_and_empty_values_are_unset() {
        with_env(
            &[
                (var::JOPE_HOME, Some("/opt/jope/checkout")),
                (var::JOPE_STATE, Some("")),
                ("XDG_STATE_HOME", Some("/xdg")),
                (var::SOTARENA_ROOT, Some("/data/sotarena")),
                (var::SOTARENA_REPORTER_ROOT, Some("/data/arena")),
                (var::SOTARENA_BENCHMARK_CACHE, Some("/data/cache")),
            ],
            || {
                assert_eq!(jope_state(), PathBuf::from("/xdg/jope"));
                assert_eq!(sotarena_root(), PathBuf::from("/data/sotarena"));
                assert_eq!(sotarena_reporter_root(), PathBuf::from("/data/arena"));
                assert_eq!(benchmark_cache(), PathBuf::from("/data/cache"));
            },
        );
    }

    #[test]
    fn home_marker_is_found_from_the_working_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(HOME_MARKER), "").unwrap();
        let nested = dir.path().join("a/b");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(marker_ancestor(&nested), Some(dir.path().to_path_buf()));
    }
}
