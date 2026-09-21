//! `State::resolve` must implement ADR-0003 exactly once — by calling `jope_paths::jope_state()`.
//!
//! Before this test existed, the crate carried its own copy of the rule that skipped the
//! `XDG_STATE_HOME` step and hard-errored when `HOME` was unset, so a host with `XDG_STATE_HOME`
//! set resolved a *different* state tree here than every other jope tool did. Each case below is
//! one step of the documented precedence; the last two are the ones the old copy got wrong.
//!
//! These tests mutate the process environment, so they serialise on `ENV_LOCK` and restore what
//! they touched. They live in their own integration-test binary, which is a separate process from
//! every other test target in the crate.

use jope_learning::state::State;
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

const VARS: [&str; 4] = ["JOPE_STATE", "XDG_STATE_HOME", "HOME", "JOPE_HOME"];

struct EnvScope {
    _guard: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn new() -> Self {
        let guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let saved = VARS.iter().map(|n| (*n, env::var_os(n))).collect();
        for name in VARS {
            env::remove_var(name);
        }
        Self {
            _guard: guard,
            saved,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(v) => env::set_var(name, v),
                None => env::remove_var(name),
            }
        }
    }
}

fn resolved(arg: Option<&str>) -> PathBuf {
    State::resolve(arg.map(PathBuf::from))
        .expect("resolve must not fail")
        .root()
        .to_path_buf()
}

#[test]
fn state_flag_wins_over_every_variable() {
    let _scope = EnvScope::new();
    env::set_var("JOPE_STATE", "/var/lib/jope");
    env::set_var("XDG_STATE_HOME", "/xdg/state");
    env::set_var("HOME", "/users/someone");
    assert_eq!(
        resolved(Some("/tmp/explicit")),
        PathBuf::from("/tmp/explicit")
    );
}

#[test]
fn jope_state_wins_over_xdg_and_home() {
    let _scope = EnvScope::new();
    env::set_var("JOPE_STATE", "/var/lib/jope");
    env::set_var("XDG_STATE_HOME", "/xdg/state");
    env::set_var("HOME", "/users/someone");
    assert_eq!(resolved(None), PathBuf::from("/var/lib/jope"));
}

#[test]
fn xdg_state_home_wins_over_home() {
    let _scope = EnvScope::new();
    env::set_var("XDG_STATE_HOME", "/xdg/state");
    env::set_var("HOME", "/users/someone");
    // The copy this crate used to carry ignored XDG_STATE_HOME and returned
    // /users/someone/.local/state/jope here — a different tree from every jope-paths caller.
    assert_eq!(resolved(None), PathBuf::from("/xdg/state/jope"));
    assert_eq!(resolved(None), jope_paths::jope_state());
}

#[test]
fn home_is_used_when_nothing_else_is_set() {
    let _scope = EnvScope::new();
    env::set_var("HOME", "/users/someone");
    assert_eq!(
        resolved(None),
        PathBuf::from("/users/someone/.local/state/jope")
    );
}

#[test]
fn a_relative_xdg_state_home_is_ignored_in_favour_of_home() {
    let _scope = EnvScope::new();
    env::set_var("XDG_STATE_HOME", "relative/ignored");
    env::set_var("HOME", "/users/someone");
    assert_eq!(
        resolved(None),
        PathBuf::from("/users/someone/.local/state/jope")
    );
}

#[test]
fn home_unset_falls_back_to_jope_home_rather_than_erroring() {
    let _scope = EnvScope::new();
    env::set_var("JOPE_HOME", "/opt/jope/checkout");
    // The old copy returned Err("HOME unset and no --state") for exactly this environment.
    assert_eq!(resolved(None), PathBuf::from("/opt/jope/checkout/state"));
    assert_eq!(resolved(None), jope_paths::jope_state());
}

#[test]
fn every_step_agrees_with_jope_paths() {
    let _scope = EnvScope::new();
    for (jope_state, xdg, home) in [
        (Some("/var/lib/jope"), Some("/xdg/state"), Some("/users/a")),
        (None, Some("/xdg/state"), Some("/users/a")),
        (None, None, Some("/users/a")),
        (None, None, None),
    ] {
        for (name, value) in [
            ("JOPE_STATE", jope_state),
            ("XDG_STATE_HOME", xdg),
            ("HOME", home),
        ] {
            match value {
                Some(v) => env::set_var(name, v),
                None => env::remove_var(name),
            }
        }
        assert_eq!(
            resolved(None),
            jope_paths::jope_state(),
            "divergence at JOPE_STATE={jope_state:?} XDG_STATE_HOME={xdg:?} HOME={home:?}"
        );
    }
}
