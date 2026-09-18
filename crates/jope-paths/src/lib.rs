//! Runtime-resolved roots (ADR-0003).
//!
//! Every absolute `/home/<user>/…` path the historical tree hard-coded is
//! replaced by one of five roots, each resolved **at call time, environment
//! first**. Nothing here is cached, so a test may override a variable, call a
//! resolver, and restore it (serialise such tests with a mutex: the process
//! environment is shared). Call sites that want a `&'static str` once per
//! process use [`once_str!`].
//!
//! | Resolver | Variable | Default | Historical value it replaces |
//! |---|---|---|---|
//! | [`jope_home`] | `JOPE_HOME` | nearest ancestor of the current directory, then of the executable, holding a [`HOME_MARKER`] file; else the current directory | `/home/ubuntu/jope-dime` (the source checkout: assets, manifests, seeds, `python/`) |
//! | [`jope_state`] | `JOPE_STATE` | `$XDG_STATE_HOME/jope`, else `$HOME/.local/state/jope`, else `<jope_home>/state` | `/home/ubuntu/jope-flywheel/test_exact_gen0` and every other mutable tree (runs, queues, sqlite, pointers, logs); the restored `parallel/` tree lives here, lineage-A material under `legacy/` |
//! | [`jope_registry`] | `JOPE_REGISTRY` | `<jope_home>/datasets/by-id` | `/home/ubuntu/remote_super/quant/regression` (dataset ids resolve to directories under it) |
//! | [`sotarena_root`] | `SOTARENA_ROOT` | `<jope_home>/benchmarks/sotarena` | the nested `sotarena` benchmark clone |
//! | [`sotarena_reporter_root`] | `SOTARENA_REPORTER_ROOT` | `<jope_state>/arena` | `/home/ubuntu/sotarena`, the Rust Elo-reporter repo the campaign CLIs read leaderboards from (a different tree from the benchmark clone) |
//!
//! Two derived locations exist because the historical tree used them as roots
//! of their own:
//!
//! | Resolver | Variable | Default | Historical value it replaces |
//! |---|---|---|---|
//! | [`scq_root`] | `JOPE_SCQ_ROOT` | `<jope_state>/scq` | `/home/ubuntu/smart_cluster` (the scheduler checkout: `target/release/scq`, `var/runs`, `etc/projects.toml` keep their relative layout under it) |
//! | [`quant_root`] | — | parent of [`jope_registry`] | `/home/ubuntu/remote_super/quant` (the dataset tree whose `regression/` child is the registry) |
//! | [`legacy_root`] | — | `<jope_state>/legacy` | `/home/ubuntu/jope-flywheel` and the other retired per-host trees (`test_exact_gen0/`, `training/`, `encodings/` keep their names under it) |
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
/// `/home/ubuntu/jope-dime`.
pub fn jope_home() -> PathBuf {
    if let Some(home) = env_path(var::JOPE_HOME) {
        return home;
    }
    let cwd = env::current_dir().ok();
    if let Some(found) = cwd.as_deref().and_then(find_marker) {
        return found;
    }
    if let Some(found) = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().and_then(find_marker))
    {
        return found;
    }
    cwd.unwrap_or_else(|| PathBuf::from("."))
}

/// Mutable run state (runs, queues, sqlite stores, pointers, logs).
///
/// `JOPE_STATE` when set; otherwise `$XDG_STATE_HOME/jope` when that variable
/// is an absolute path (the XDG rule), otherwise `$HOME/.local/state/jope`,
/// otherwise `<jope_home>/state`. Replaces `/home/ubuntu/jope-flywheel/test_exact_gen0`
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
/// `/home/ubuntu/remote_super/quant/regression`.
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
/// `/home/ubuntu/sotarena` — the Rust Elo-reporter repo, which is not the
/// benchmark clone above (B20).
pub fn sotarena_reporter_root() -> PathBuf {
    env_path(var::SOTARENA_REPORTER_ROOT).unwrap_or_else(|| jope_state().join("arena"))
}

/// The scheduler checkout the self-play control plane drives (`scq`).
///
/// `JOPE_SCQ_ROOT` when set, otherwise `<jope_state>/scq`. Replaces
/// `/home/ubuntu/smart_cluster`; the relative layout beneath it
/// (`target/release/scq`, `var/runs`, `var/scqd.sock`, `var/events.jsonl`,
/// `etc/projects.toml`) is unchanged so an operator can point the variable at
/// an existing checkout.
pub fn scq_root() -> PathBuf {
    env_path(var::JOPE_SCQ_ROOT).unwrap_or_else(|| jope_state().join("scq"))
}

/// The dataset tree that contains the registry: the parent of
/// [`jope_registry`] (the registry itself when it has no parent). Replaces
/// `/home/ubuntu/remote_super/quant`, whose `regression/` child was the
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
/// `/home/ubuntu/jope-flywheel` and the other single-host roots.
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

    const VARS: [&str; 8] = [
        var::JOPE_HOME,
        var::JOPE_STATE,
        var::JOPE_REGISTRY,
        var::SOTARENA_ROOT,
        var::SOTARENA_REPORTER_ROOT,
        var::JOPE_SCQ_ROOT,
        "XDG_STATE_HOME",
        "HOME",
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

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root")
    }

    #[test]
    fn home_prefers_the_environment() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_HOME, "/opt/jope/home");
        assert_eq!(jope_home(), PathBuf::from("/opt/jope/home"));
        env::set_var(var::JOPE_HOME, "");
        assert_ne!(jope_home(), PathBuf::from(""), "an empty value is unset");
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
    fn home_falls_back_to_the_executable_ancestor_with_the_marker() {
        let _scope = EnvScope::new();
        let stray = temp_dir("no-marker");
        env::set_current_dir(&stray).unwrap();
        // No marker above the temp dir; the test binary lives under the repo's
        // target/, and the repo root commits the marker.
        assert!(
            find_marker(&stray).is_none(),
            "temp tree must not carry a marker"
        );
        assert_eq!(jope_home().canonicalize().unwrap(), repo_root());
        assert!(repo_root().join(HOME_MARKER).is_file());
        fs::remove_dir_all(&stray).ok();
    }

    #[test]
    fn state_resolution_order() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_STATE, "/var/lib/jope");
        assert_eq!(jope_state(), PathBuf::from("/var/lib/jope"));
        env::remove_var(var::JOPE_STATE);

        env::set_var("XDG_STATE_HOME", "/xdg/state");
        env::set_var("HOME", "/home/someone");
        assert_eq!(jope_state(), PathBuf::from("/xdg/state/jope"));

        env::set_var("XDG_STATE_HOME", "relative/ignored");
        assert_eq!(
            jope_state(),
            PathBuf::from("/home/someone/.local/state/jope")
        );

        env::remove_var("XDG_STATE_HOME");
        env::remove_var("HOME");
        env::set_var(var::JOPE_HOME, "/opt/jope/home");
        assert_eq!(jope_state(), PathBuf::from("/opt/jope/home/state"));
    }

    #[test]
    fn registry_and_arena_roots_follow_home_and_state() {
        let _scope = EnvScope::new();
        env::set_var(var::JOPE_HOME, "/opt/jope/home");
        env::set_var(var::JOPE_STATE, "/var/lib/jope");
        assert_eq!(
            jope_registry(),
            PathBuf::from("/opt/jope/home/datasets/by-id")
        );
        assert_eq!(quant_root(), PathBuf::from("/opt/jope/home/datasets"));
        assert_eq!(
            sotarena_root(),
            PathBuf::from("/opt/jope/home/benchmarks/sotarena")
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
