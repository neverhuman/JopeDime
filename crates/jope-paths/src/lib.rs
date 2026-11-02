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

#[cfg(test)]
use std::env;
#[cfg(test)]
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

#[cfg(test)]
use roots::{absolute_env_path, default_jopedime_home, env_path, find_marker, select_home};
mod roots;
pub use roots::{
    jope_home, jope_registry, jope_state, jopedime_home, legacy_root, quant_root, scq_root,
    sotarena_reporter_root, sotarena_root, to_string,
};

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
mod tests;
