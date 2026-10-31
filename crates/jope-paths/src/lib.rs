//! Runtime-resolved roots (ADR-0003).
//!
//! Every absolute home-directory path the historical tree hard-coded is
//! replaced by runtime roots, each resolved **at call time, environment
//! first**. Historical values below are written `~/…`, where `~` is the
//! `ubuntu` account's home on the xbabe hosts (the README spells them out). Nothing here is cached, so a test may override a variable, call a
//! resolver, and restore it (serialise such tests with a mutex: the process
//! environment is shared). Call sites that want a `&'static str` once per
//! process use [`once_str!`].
//!
//! | Resolver | Variable | Default | Historical value it replaces |
//! |---|---|---|---|
//! | [`jope_home`] | `JOPE_HOME` | nearest ancestor of the current directory, then of the executable, holding a [`HOME_MARKER`] file; else the current directory | `~/jope-dime` (the source checkout: assets, manifests, seeds, `python/`) |
//! | [`jopedime_home`] | `JOPEDIME_HOME` | OS data directory for the JopeDime service | — |
//! | [`jope_state`] | `JOPE_STATE` | `$XDG_STATE_HOME/jope`, else `$HOME/.local/state/jope`, else `<jope_home>/state` | `~/jope-flywheel/test_exact_gen0` and every other mutable tree (runs, queues, sqlite, pointers, logs); the restored `parallel/` tree lives here, lineage-A material under `legacy/` |
//! | [`jope_registry`] | `JOPE_REGISTRY` | `<jope_home>/datasets/by-id` | `~/remote_super/quant/regression` (dataset ids resolve to directories under it) |
//! | [`sotarena_root`] | `SOTARENA_ROOT` | `<jope_home>/benchmarks/sotarena` | the nested `sotarena` benchmark clone |
//! | [`sotarena_reporter_root`] | `SOTARENA_REPORTER_ROOT` | `<jope_state>/arena` | `~/sotarena`, the Rust Elo-reporter repo the campaign CLIs read leaderboards from (a different tree from the benchmark clone) |
//!
//! Two derived locations exist because the historical tree used them as roots
//! of their own:
//!
//! | Resolver | Variable | Default | Historical value it replaces |
//! |---|---|---|---|
//! | [`scq_root`] | `JOPE_SCQ_ROOT` | `<jope_state>/scq` | `~/smart_cluster` (the scheduler checkout: `target/release/scq`, `var/runs`, `etc/projects.toml` keep their relative layout under it) |
//! | [`quant_root`] | — | parent of [`jope_registry`] | `~/remote_super/quant` (the dataset tree whose `regression/` child is the registry) |
//! | [`legacy_root`] | — | `<jope_state>/legacy` | `~/jope-flywheel` and the other retired per-host trees (`test_exact_gen0/`, `training/`, `encodings/` keep their names under it) |
//!
//! Environment values are taken verbatim when non-empty; relative values are
//! not resolved against anything (systemd and shells hand over absolute paths).

use std::env;
use std::path::{Path, PathBuf};

pub mod systemd;

/// Marker file that identifies a `JOPE_HOME` checkout; committed empty at the
/// repository root.
pub const HOME_MARKER: &str = ".jope-home";

/// Environment variable names, so call sites and clap `env = …` attributes
/// spell them once.
pub mod var {
    /// The source checkout.
    pub const JOPE_HOME: &str = "JOPE_HOME";
    /// Mutable run state.
    pub const JOPE_STATE: &str = "JOPE_STATE";
    /// Dataset registry (id → directory).
    pub const JOPE_REGISTRY: &str = "JOPE_REGISTRY";
    /// The benchmark submodule.
    pub const SOTARENA_ROOT: &str = "SOTARENA_ROOT";
    /// The Elo reporter's data root.
    pub const SOTARENA_REPORTER_ROOT: &str = "SOTARENA_REPORTER_ROOT";
    /// The scheduler (`scq`) checkout.
    pub const JOPE_SCQ_ROOT: &str = "JOPE_SCQ_ROOT";
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn absolute_env_path(name: &str) -> Option<PathBuf> {
    env_path(name).filter(|path| path.is_absolute())
}

fn find_marker(start: &Path) -> Option<PathBuf> {
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

fn default_jopedime_home(
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

fn select_home(home: Option<&Path>, cwd: Option<&Path>, executable: Option<&Path>) -> PathBuf {
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

/// Defines a function returning a `&'static str` resolved once per process.
///
/// ```
/// jope_paths::once_str!(fn corpus_db => jope_paths::jope_state().join("corpus/strict.sqlite"));
/// assert!(corpus_db().ends_with("corpus/strict.sqlite"));
/// ```
///
/// The expression is evaluated on first use and cached, so a call site that
/// used to name a `const &str` keeps its type; prefer the plain resolvers when
/// the environment may change during the process (tests).
#[macro_export]
macro_rules! once_str {
    ($(#[$meta:meta])* $vis:vis fn $name:ident => $expr:expr) => {
        $(#[$meta])*
        $vis fn $name() -> &'static str {
            static CELL: ::std::sync::OnceLock<::std::string::String> = ::std::sync::OnceLock::new();
            CELL.get_or_init(|| {
                let path: ::std::path::PathBuf = $expr;
                $crate::to_string(&path)
            })
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, MutexGuard};

    /// The process environment is shared: every test that mutates it holds
    /// this lock and restores what it touched.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const VARS: [&str; 11] = [
        var::JOPE_HOME,
        var::JOPE_STATE,
        var::JOPE_REGISTRY,
        var::SOTARENA_ROOT,
        var::SOTARENA_REPORTER_ROOT,
        var::JOPE_SCQ_ROOT,
        "JOPEDIME_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
        "HOME",
        "LOCALAPPDATA",
    ];

    struct EnvScope {
        _guard: MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
        cwd: PathBuf,
    }

    impl EnvScope {
        fn new() -> Self {
            let guard = ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let saved = VARS.iter().map(|name| (*name, env::var_os(name))).collect();
            let cwd = env::current_dir().expect("cwd");
            for name in VARS {
                env::remove_var(name);
            }
            Self {
                _guard: guard,
                saved,
                cwd,
            }
        }
    }

    impl Drop for EnvScope {
        fn drop(&mut self) {
            env::set_current_dir(&self.cwd).expect("restore cwd");
            for (name, value) in self.saved.drain(..) {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let unique = format!(
            "jope-paths-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let dir = env::temp_dir().join(unique);
        fs::create_dir_all(&dir).expect("temp dir");
        dir.canonicalize().expect("canonical temp dir")
    }

    #[test]
    fn home_prefers_the_environment() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_HOME, "/opt/jope/checkout");
        assert_eq!(jope_home(), PathBuf::from("/opt/jope/checkout"));
        env::set_var(var::JOPE_HOME, "");
        assert_ne!(jope_home(), PathBuf::from(""), "an empty value is unset");
    }

    #[test]
    fn jopedime_home_uses_environment_and_os_data_directories() {
        let _scope = EnvScope::new();

        env::set_var("JOPEDIME_HOME", "/configured/jopedime");
        assert_eq!(jopedime_home(), PathBuf::from("/configured/jopedime"));
        env::remove_var("JOPEDIME_HOME");

        assert_eq!(
            default_jopedime_home(
                "linux",
                None,
                Some(std::ffi::OsStr::new("/xdg/data")),
                Some(std::ffi::OsStr::new("/users/alice")),
                None,
            ),
            PathBuf::from("/xdg/data/jopedime")
        );
        assert_eq!(
            default_jopedime_home(
                "linux",
                None,
                None,
                Some(std::ffi::OsStr::new("/users/alice")),
                None,
            ),
            PathBuf::from("/users/alice/.local/share/jopedime")
        );
        assert_eq!(
            default_jopedime_home(
                "macos",
                None,
                None,
                Some(std::ffi::OsStr::new("/Users/alice")),
                None,
            ),
            PathBuf::from("/Users/alice/Library/Application Support/jopedime")
        );
        assert_eq!(
            default_jopedime_home(
                "windows",
                None,
                None,
                None,
                Some(std::ffi::OsStr::new(r"C:\Users\Alice\AppData\Local")),
            ),
            PathBuf::from(r"C:\Users\Alice\AppData\Local").join("jopedime")
        );
    }

    #[test]
    fn home_walks_up_from_the_current_directory_to_the_marker() {
        let _scope = EnvScope::new();
        let root = temp_dir("marker");
        fs::write(root.join(HOME_MARKER), "").unwrap();
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        env::set_current_dir(&nested).unwrap();
        assert_eq!(jope_home().canonicalize().unwrap(), root);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn home_resolution_is_independent_of_the_build_directory() {
        let root = temp_dir("resolution");
        let cwd_root = root.join("checkout");
        let exe_root = root.join("installed");
        let nested = cwd_root.join("nested");
        let exe = exe_root.join("bin/jope");
        let stray = root.join("no-marker");
        for dir in [&nested, exe.parent().unwrap(), &stray] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(cwd_root.join(HOME_MARKER), "").unwrap();
        fs::write(exe_root.join(HOME_MARKER), "").unwrap();
        assert!(
            find_marker(&stray).is_none(),
            "fixture must have no ancestor marker"
        );

        // Environment overrides remain verbatim, including relative paths.
        assert_eq!(
            select_home(Some(Path::new("relative")), Some(&nested), Some(&exe)),
            PathBuf::from("relative")
        );
        assert_eq!(
            select_home(Some(Path::new("")), Some(&nested), Some(&exe)),
            cwd_root
        );
        assert_eq!(select_home(None, Some(&nested), Some(&exe)), cwd_root);
        assert_eq!(select_home(None, Some(&stray), Some(&exe)), exe_root);
        assert_eq!(select_home(None, None, Some(&exe)), exe_root);
        assert_eq!(
            select_home(None, Some(&stray), Some(&stray.join("bin"))),
            stray
        );
        assert_eq!(select_home(None, Some(&stray), None), stray);
        assert_eq!(select_home(None, None, None), PathBuf::from("."));
        assert_eq!(
            select_home(None, None, Some(&stray.join("bin"))),
            PathBuf::from(".")
        );

        // The nearest marker wins within each ancestry chain.
        fs::write(nested.join(HOME_MARKER), "").unwrap();
        assert_eq!(select_home(None, Some(&nested), Some(&exe)), nested);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn state_resolution_order() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_STATE, "/var/lib/jope");
        assert_eq!(jope_state(), PathBuf::from("/var/lib/jope"));
        env::remove_var(var::JOPE_STATE);

        env::set_var("XDG_STATE_HOME", "/xdg/state");
        env::set_var("HOME", "/users/someone");
        assert_eq!(jope_state(), PathBuf::from("/xdg/state/jope"));

        env::set_var("XDG_STATE_HOME", "relative/ignored");
        assert_eq!(
            jope_state(),
            PathBuf::from("/users/someone/.local/state/jope")
        );

        env::remove_var("XDG_STATE_HOME");
        env::remove_var("HOME");
        env::set_var(var::JOPE_HOME, "/opt/jope/checkout");
        assert_eq!(jope_state(), PathBuf::from("/opt/jope/checkout/state"));
    }

    #[test]
    fn registry_and_arena_roots_follow_home_and_state() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_HOME, "/opt/jope/checkout");
        env::set_var(var::JOPE_STATE, "/var/lib/jope");
        assert_eq!(
            jope_registry(),
            PathBuf::from("/opt/jope/checkout/datasets/by-id")
        );
        assert_eq!(quant_root(), PathBuf::from("/opt/jope/checkout/datasets"));
        assert_eq!(
            sotarena_root(),
            PathBuf::from("/opt/jope/checkout/benchmarks/sotarena")
        );
        assert_eq!(
            sotarena_reporter_root(),
            PathBuf::from("/var/lib/jope/arena")
        );
        assert_eq!(scq_root(), PathBuf::from("/var/lib/jope/scq"));
        assert_eq!(legacy_root(), PathBuf::from("/var/lib/jope/legacy"));

        env::set_var(var::JOPE_REGISTRY, "/data/quant/regression");
        env::set_var(var::SOTARENA_ROOT, "/data/sotarena");
        env::set_var(var::SOTARENA_REPORTER_ROOT, "/data/arena");
        env::set_var(var::JOPE_SCQ_ROOT, "/data/smart_cluster");
        assert_eq!(jope_registry(), PathBuf::from("/data/quant/regression"));
        assert_eq!(quant_root(), PathBuf::from("/data/quant"));
        assert_eq!(sotarena_root(), PathBuf::from("/data/sotarena"));
        assert_eq!(sotarena_reporter_root(), PathBuf::from("/data/arena"));
        assert_eq!(scq_root(), PathBuf::from("/data/smart_cluster"));
    }

    #[test]
    fn quant_root_of_a_bare_registry_is_itself() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_REGISTRY, "registry");
        assert_eq!(quant_root(), PathBuf::from("registry"));
    }

    #[test]
    fn string_form() {
        assert_eq!(to_string(Path::new("/x/y")), "/x/y");
    }

    once_str!(fn cached_home_child => jope_home().join("child"));

    #[test]
    fn once_str_resolves_once() {
        let first = cached_home_child();
        let second = cached_home_child();
        assert!(first.ends_with("/child"), "{first}");
        assert!(std::ptr::eq(first, second), "cached per process");
    }
}
