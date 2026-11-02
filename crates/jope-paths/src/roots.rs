//! Runtime path root resolution helpers.

use std::env;
use std::path::{Path, PathBuf};

use super::{var, HOME_MARKER};

pub(crate) fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub(crate) fn absolute_env_path(name: &str) -> Option<PathBuf> {
    env_path(name).filter(|path| path.is_absolute())
}

pub(crate) fn find_marker(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(HOME_MARKER).is_file())
        .map(Path::to_path_buf)
}

/// The source checkout (assets, manifests, seeds, `python/`).
///
/// `JOPE_HOME` when set; otherwise the nearest ancestor of the current
/// directory, then of the running executable's directory, that contains a
/// [`HOME_MARKER`]; otherwise the current directory. Replaces
/// `~/jope-dime`.
pub fn jope_home() -> PathBuf {
    select_home(
        env_path(var::JOPE_HOME).as_deref(),
        env::current_dir().ok().as_deref(),
        env::current_exe().ok().as_deref(),
    )
}

/// JopeDime's service data directory.
///
/// `JOPEDIME_HOME` when set; otherwise `$XDG_DATA_HOME/jopedime` or
/// `$HOME/.local/share/jopedime` on Linux, `~/Library/Application Support/jopedime`
/// on macOS, and `%LOCALAPPDATA%\\jopedime` on Windows. The environment is
/// read on every call so callers and tests can change it at runtime.
pub fn jopedime_home() -> PathBuf {
    let override_path = env::var_os("JOPEDIME_HOME");
    let xdg_data_home = env::var_os("XDG_DATA_HOME");
    let home = env::var_os("HOME");
    let local_app_data = env::var_os("LOCALAPPDATA");
    default_jopedime_home(
        std::env::consts::OS,
        override_path.as_deref(),
        xdg_data_home.as_deref(),
        home.as_deref(),
        local_app_data.as_deref(),
    )
}

pub(crate) fn default_jopedime_home(
    os: &str,
    override_path: Option<&std::ffi::OsStr>,
    xdg_data_home: Option<&std::ffi::OsStr>,
    home: Option<&std::ffi::OsStr>,
    local_app_data: Option<&std::ffi::OsStr>,
) -> PathBuf {
    if let Some(path) = override_path.filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }

    let home = home
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    match os {
        "linux" => xdg_data_home
            .filter(|path| !path.is_empty() && Path::new(path).is_absolute())
            .map(|path| PathBuf::from(path).join("jopedime"))
            .unwrap_or_else(|| home.join(".local/share/jopedime")),
        "macos" => home.join("Library/Application Support/jopedime"),
        "windows" => local_app_data
            .filter(|path| !path.is_empty())
            .map(|path| PathBuf::from(path).join("jopedime"))
            .unwrap_or_else(|| home.join("AppData/Local/jopedime")),
        _ => home.join(".local/share/jopedime"),
    }
}

pub(crate) fn select_home(
    home: Option<&Path>,
    cwd: Option<&Path>,
    executable: Option<&Path>,
) -> PathBuf {
    if let Some(home) = home.filter(|path| !path.as_os_str().is_empty()) {
        return home.to_path_buf();
    }
    if let Some(found) = cwd.and_then(find_marker) {
        return found;
    }
    if let Some(found) = executable.and_then(Path::parent).and_then(find_marker) {
        return found;
    }
    cwd.unwrap_or_else(|| Path::new(".")).to_path_buf()
}

/// Mutable run state (runs, queues, sqlite stores, pointers, logs).
///
/// `JOPE_STATE` when set; otherwise `$XDG_STATE_HOME/jope` when that variable
/// is an absolute path (the XDG rule), otherwise `$HOME/.local/state/jope`,
/// otherwise `<jope_home>/state`. Replaces `~/jope-flywheel/test_exact_gen0`
/// (now [`legacy_root`]`/test_exact_gen0`) and every other mutable tree.
pub fn jope_state() -> PathBuf {
    if let Some(state) = env_path(var::JOPE_STATE) {
        return state;
    }
    if let Some(xdg) = absolute_env_path("XDG_STATE_HOME") {
        return xdg.join("jope");
    }
    if let Some(home) = absolute_env_path("HOME") {
        return home.join(".local/state/jope");
    }
    jope_home().join("state")
}

/// The dataset registry: `<registry>/<dataset id>/…`.
///
/// `JOPE_REGISTRY` when set, otherwise `<jope_home>/datasets/by-id`. Replaces
/// `~/remote_super/quant/regression`.
pub fn jope_registry() -> PathBuf {
    env_path(var::JOPE_REGISTRY).unwrap_or_else(|| jope_home().join("datasets/by-id"))
}

/// The benchmark submodule.
///
/// `SOTARENA_ROOT` when set, otherwise `<jope_home>/benchmarks/sotarena`.
/// Replaces the nested `sotarena` clone of the historical checkout.
pub fn sotarena_root() -> PathBuf {
    env_path(var::SOTARENA_ROOT).unwrap_or_else(|| jope_home().join("benchmarks/sotarena"))
}

/// The Elo reporter's data root (leaderboards, scans, reports).
///
/// `SOTARENA_REPORTER_ROOT` when set, otherwise `<jope_state>/arena`. Replaces
/// `~/sotarena` — the Rust Elo-reporter repo, which is not the
/// benchmark clone above (B20).
pub fn sotarena_reporter_root() -> PathBuf {
    env_path(var::SOTARENA_REPORTER_ROOT).unwrap_or_else(|| jope_state().join("arena"))
}

/// The scheduler checkout the self-play control plane drives (`scq`).
///
/// `JOPE_SCQ_ROOT` when set, otherwise `<jope_state>/scq`. Replaces
/// `~/smart_cluster`; the relative layout beneath it
/// (`target/release/scq`, `var/runs`, `var/scqd.sock`, `var/events.jsonl`,
/// `etc/projects.toml`) is unchanged so an operator can point the variable at
/// an existing checkout.
pub fn scq_root() -> PathBuf {
    env_path(var::JOPE_SCQ_ROOT).unwrap_or_else(|| jope_state().join("scq"))
}

/// The dataset tree that contains the registry: the parent of
/// [`jope_registry`] (the registry itself when it has no parent). Replaces
/// `~/remote_super/quant`, whose `regression/` child was the
/// registry and whose siblings (`timeseries/`, `multi_label/`, `releases/`,
/// `campaigns/`) the campaign CLIs still address relative to it.
pub fn quant_root() -> PathBuf {
    let registry = jope_registry();
    registry
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or(registry.clone(), Path::to_path_buf)
}

/// Retired per-host trees, kept by name under `<jope_state>/legacy`
/// (`test_exact_gen0/`, `training/`, `encodings/`, `scripts/`, …). Replaces
/// `~/jope-flywheel` and the other single-host roots.
pub fn legacy_root() -> PathBuf {
    jope_state().join("legacy")
}

/// Lossless-when-possible string form of a resolved path, for call sites that
/// carry paths as strings (clap defaults, JSON receipts, shell fragments).
pub fn to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
