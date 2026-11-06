//! jAIn CLI first run: VEOX terms of service, the telemetry choice, and login or demos.
//!
//! The decision is stored in `<home>/state.json` (0600) with the accepted [`TERMS_VERSION`];
//! a new terms version asks again. Agents and scripts accept with `JAIN_ACCEPT_TERMS=1` after
//! showing the user [`TERMS_URL`]; nothing ever blocks on a prompt without a terminal. The anonymous-use policy lives in [`local_file_gate`].

use anyhow::{Context, Result};
use ratatui::crossterm::event::KeyCode;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The version of https://veox.ai/terms this build asks users to accept.
pub const TERMS_VERSION: &str = "2026-11-03";
pub const TERMS_URL: &str = super::links::TERMS_URL;
pub const PRIVACY_URL: &str = "https://veox.ai/privacy";
const STATE_SCHEMA: &str = "jain.cli-state/v1";
const MAX_STATE: u64 = 64 * 1024;

/// Persisted onboarding decisions. Unknown fields written by newer clients are preserved.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms_accepted_at: Option<String>,
    /// How the terms were accepted: `tui`, `line`, `prompt`, `flag` or `env`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_decided_at: Option<String>,
    /// Whether a finished local run's model and code go to the workspace (`/sync on|off`).
    /// `None` until the user is first asked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub results_sync: Option<bool>,
    /// Upload consents, keyed by `<origin> <account> <file sha256>`: one question per dataset.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub upload_consents: std::collections::BTreeMap<String, UploadConsent>,
    /// The account's plan as last read from the workspace, for offline routing (MCP).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_tier: Option<String>,
    /// The last update check (at most one per 24 hours).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_check: Option<UpdateCheck>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// What the user agreed to upload for one dataset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadConsent {
    pub rows: u64,
    pub columns: u64,
    pub bytes: u64,
    pub at: String,
}

/// The cached result of the last update check.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateCheck {
    /// Unix seconds of the last attempt (successful or not).
    pub checked_at: u64,
    /// The newer stable version found, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available: Option<String>,
}

/// Read-modify-write the state under one process-wide lock, so background threads (the update
/// check, a sync) never lose a concurrent answer.
pub fn update(home: &Path, change: impl FnOnce(&mut State)) -> Result<State> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut state = load(home);
    change(&mut state);
    if state.schema.is_empty() {
        state.schema = STATE_SCHEMA.into();
    }
    save(home, &state)?;
    Ok(state)
}

/// The terms version to require: this build's [`TERMS_VERSION`], or a newer one reported by a
/// workspace. Versions are ISO dates, so they order as text.
pub fn required_terms_version(server: Option<&str>) -> String {
    match server {
        Some(server) if server > TERMS_VERSION => server.to_owned(),
        _ => TERMS_VERSION.to_owned(),
    }
}

/// The required version now. The workspace does not report a terms version yet.
pub fn required_terms_version_now() -> String {
    required_terms_version(None)
}

impl State {
    pub fn terms_current(&self) -> bool {
        self.terms_accepted(&required_terms_version_now())
    }

    /// Whether the accepted version is `required` or later.
    pub fn terms_accepted(&self, required: &str) -> bool {
        self.terms_version
            .as_deref()
            .is_some_and(|accepted| accepted >= required)
    }

    pub fn accept_terms(&mut self, source: &str, now: String) {
        self.accept_terms_version(&required_terms_version_now(), source, now);
    }

    pub fn accept_terms_version(&mut self, version: &str, source: &str, now: String) {
        self.schema = STATE_SCHEMA.into();
        self.terms_version = Some(version.into());
        self.terms_accepted_at = Some(now);
        self.terms_source = Some(source.into());
    }
}

pub fn state_path(home: &Path) -> PathBuf {
    home.join("state.json")
}

/// The stored state; a missing, unreadable or malformed file reads as "nothing decided", so
/// the user is asked again rather than the program failing.
pub fn load(home: &Path) -> State {
    let path = state_path(home);
    let Ok(meta) = std::fs::symlink_metadata(&path) else {
        return State::default();
    };
    if !meta.is_file() || meta.len() > MAX_STATE {
        return State::default();
    }
    std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// Atomically write the state as an owner-only (0600) file.
pub fn save(home: &Path, state: &State) -> Result<()> {
    if !home.exists() {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(home)
            .with_context(|| format!("create {}", home.display()))?;
    }
    let path = state_path(home);
    if std::fs::symlink_metadata(&path).is_ok_and(|meta| !meta.is_file()) {
        anyhow::bail!("{} must be a regular file", path.display());
    }
    let mut bytes = serde_json::to_vec_pretty(state)?;
    bytes.push(b'\n');
    let temp = home.join(format!(".state-{}.tmp", ulid::Ulid::new()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| -> Result<()> {
        use std::io::Write;
        let mut file = options.open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, &path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result.with_context(|| format!("write {}", path.display()))
}

fn truthy(value: Option<String>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// `DO_NOT_TRACK=1` (any value but empty/`0`) or `JAIN_TELEMETRY=0|off|false` force telemetry
/// off and skip the question.
pub fn telemetry_forced_off(var: impl Fn(&str) -> Option<String>) -> bool {
    var("DO_NOT_TRACK").is_some_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    }) || var("JAIN_TELEMETRY").is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    })
}

pub fn telemetry_forced_off_from_env() -> bool {
    telemetry_forced_off(|name| std::env::var(name).ok()) || flags().telemetry == Some(false)
}

/// The global `--accept-terms` and `--telemetry on|off` flags, taken off the command line before
/// anything else parses it (see [`take_flags`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    pub accept_terms: bool,
    pub telemetry: Option<bool>,
}

static FLAGS: std::sync::OnceLock<Flags> = std::sync::OnceLock::new();

pub fn flags() -> Flags {
    FLAGS.get().copied().unwrap_or_default()
}

/// Remove `--accept-terms`, `--telemetry on|off` and `--telemetry=on|off` from `args`, wherever
/// they appear, and return them. An invalid telemetry value is an error.
pub fn parse_flags(args: &mut Vec<String>) -> Result<Flags, String> {
    let mut flags = Flags::default();
    let mut out = Vec::with_capacity(args.len());
    let mut iter = std::mem::take(args).into_iter();
    while let Some(arg) = iter.next() {
        if arg == "--" {
            out.push(arg);
            out.extend(iter.by_ref());
            break;
        }
        let value = if arg == "--accept-terms" {
            flags.accept_terms = true;
            continue;
        } else if arg == "--telemetry" {
            iter.next()
                .ok_or("--telemetry needs on or off (show the user https://veox.ai/terms first)")?
        } else if let Some(value) = arg.strip_prefix("--telemetry=") {
            value.to_owned()
        } else {
            out.push(arg);
            continue;
        };
        flags.telemetry = Some(match value.to_ascii_lowercase().as_str() {
            "on" | "1" | "true" | "yes" => true,
            "off" | "0" | "false" | "no" => false,
            _ => return Err(format!("--telemetry takes on or off, not {value:?}")),
        });
    }
    *args = out;
    Ok(flags)
}

/// [`parse_flags`] for this process, applying the decisions to the stored state the way the
/// environment variables do.
pub fn take_flags(args: &mut Vec<String>) -> Result<Flags, String> {
    let flags = parse_flags(args)?;
    let _ = FLAGS.set(flags);
    let home = jope_paths::jopedime_home();
    if flags.accept_terms {
        let _ = accept_noninteractive(&home, "flag");
    }
    if let Some(choice) = noninteractive_telemetry(flags, |name| std::env::var(name).ok()) {
        let _ = record_telemetry(&home, choice);
    }
    Ok(flags)
}

/// The telemetry choice a flag or `JAIN_TELEMETRY` makes without a prompt. `DO_NOT_TRACK`
/// still forces it off.
pub fn noninteractive_telemetry(
    flags: Flags,
    var: impl Fn(&str) -> Option<String>,
) -> Option<bool> {
    let env = var("JAIN_TELEMETRY").and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    });
    let choice = flags.telemetry.or(env)?;
    let dnt = var("DO_NOT_TRACK").is_some_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    });
    Some(choice && !dnt)
}

fn record_telemetry(home: &Path, enabled: bool) -> Result<()> {
    let current = load(home);
    if current.telemetry == Some(enabled) {
        return Ok(());
    }
    update(home, |state| {
        state.telemetry = Some(enabled);
        state.telemetry_decided_at = Some(crate::cloud::chrono_like_now());
    })?;
    crate::cloud::set_telemetry(enabled)
}

/// `JAIN_ACCEPT_TERMS=1` or `--accept-terms` accepts the terms non-interactively.
pub fn env_accepts_terms() -> bool {
    truthy(std::env::var("JAIN_ACCEPT_TERMS").ok()) || flags().accept_terms
}

static ACCEPTED_NONINTERACTIVELY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether this process accepted the terms from `JAIN_ACCEPT_TERMS`.
pub fn accepted_noninteractively() -> bool {
    ACCEPTED_NONINTERACTIVELY.load(std::sync::atomic::Ordering::SeqCst)
}

/// Record a non-interactive acceptance (`JAIN_ACCEPT_TERMS=1`) unless the current terms are
/// already accepted.
pub fn accept_noninteractive(home: &Path, source: &str) -> Result<()> {
    ACCEPTED_NONINTERACTIVELY.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut state = load(home);
    if state.terms_current() {
        return Ok(());
    }
    state.accept_terms(source, crate::cloud::chrono_like_now());
    save(home, &state)
}

// ---- anonymous-use policy: the single place to change it --------------------------------

/// Whether a signed-out interactive session may open the user's own (non-example) files.
/// Bundled demos are always allowed. `JAIN_ALLOW_LOCAL_FILES=1` lifts the gate for offline
/// and CI use; command-line subcommands (`jopedime run`, `fit`, ...) are not gated.
pub const ANONYMOUS_LOCAL_FILES: bool = false;

pub const LOGIN_FOR_FILES: &str = "Opening your own files needs a free veox Cloud Workspace account. Type /login to log in or create one (free; runs on your own data happen in the veox Cloud, and jAIn asks before uploading), or try a bundled demo: /demo";

pub fn local_file_gate(logged_in: bool, override_env: bool) -> Result<(), &'static str> {
    if logged_in || ANONYMOUS_LOCAL_FILES || override_env {
        Ok(())
    } else {
        Err(LOGIN_FOR_FILES)
    }
}

pub fn local_files_override_from_env() -> bool {
    truthy(std::env::var("JAIN_ALLOW_LOCAL_FILES").ok())
}

// ---- the first-run dialog ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Terms,
    Telemetry,
    Account,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Login,
    Demos,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The user declined the terms; exit without changing anything.
    Quit,
    Finished {
        terms_accepted: bool,
        /// `None` when the question was not asked.
        telemetry: Option<bool>,
        choice: Option<Choice>,
    },
}

/// The dialog's steps, decided once from the stored state.
#[derive(Clone, Debug)]
pub struct FirstRun {
    /// The terms version this dialog asks for.
    pub terms_version: String,
    steps: Vec<Step>,
    index: usize,
    terms_accepted: bool,
    telemetry: Option<bool>,
}

impl FirstRun {
    /// `None` when nothing needs asking. Terms are asked when the stored version is not the
    /// current one; telemetry when it was never decided and is not forced off; the account
    /// choice on the very first run while signed out.
    pub fn new(
        state: &State,
        required_terms: &str,
        telemetry_forced_off: bool,
        logged_in: bool,
    ) -> Option<Self> {
        let mut steps = Vec::new();
        if !state.terms_accepted(required_terms) {
            steps.push(Step::Terms);
        }
        if state.telemetry.is_none() && !telemetry_forced_off {
            steps.push(Step::Telemetry);
        }
        if state.terms_version.is_none() && !logged_in {
            steps.push(Step::Account);
        }
        // A telemetry question alone (terms already current) is still worth asking once.
        (!steps.is_empty()).then_some(Self {
            terms_version: required_terms.to_owned(),
            steps,
            index: 0,
            terms_accepted: false,
            telemetry: None,
        })
    }

    pub fn step(&self) -> Option<Step> {
        self.steps.get(self.index).copied()
    }

    fn advance(&mut self) -> Option<Outcome> {
        self.index += 1;
        if self.index < self.steps.len() {
            None
        } else {
            Some(self.finished(None))
        }
    }

    fn finished(&self, choice: Option<Choice>) -> Outcome {
        Outcome::Finished {
            terms_accepted: self.terms_accepted,
            telemetry: self.telemetry,
            choice,
        }
    }

    /// Handle one key in the full-screen dialog.
    pub fn key(&mut self, code: KeyCode) -> Option<Outcome> {
        let pressed = match code {
            KeyCode::Enter => Some('\n'),
            KeyCode::Esc => Some('q'),
            KeyCode::Char(c) => Some(c.to_ascii_lowercase()),
            _ => None,
        }?;
        match self.step()? {
            Step::Terms => match pressed {
                '\n' | 'y' | 'a' => {
                    self.terms_accepted = true;
                    self.advance()
                }
                'q' | 'n' => Some(Outcome::Quit),
                _ => None,
            },
            Step::Telemetry => match pressed {
                '\n' | 'y' => {
                    self.telemetry = Some(true);
                    self.advance()
                }
                'n' | 'o' => {
                    self.telemetry = Some(false);
                    self.advance()
                }
                _ => None,
            },
            Step::Account => match pressed {
                '\n' | '1' | 'l' => Some(self.finished(Some(Choice::Login))),
                '2' | 'd' | 'q' => Some(self.finished(Some(Choice::Demos))),
                _ => None,
            },
        }
    }

    /// Handle one answered line in line mode: an empty line is Enter.
    pub fn line(&mut self, answer: &str) -> Option<Outcome> {
        let answer = answer.trim().to_ascii_lowercase();
        let code = match answer.as_str() {
            "" => KeyCode::Enter,
            "yes" | "accept" => KeyCode::Char('y'),
            "no" | "off" => KeyCode::Char('n'),
            "quit" | "exit" => KeyCode::Char('q'),
            "login" => KeyCode::Char('l'),
            "demo" | "demos" => KeyCode::Char('d'),
            other => KeyCode::Char(other.chars().next()?),
        };
        if answer.chars().count() > 1
            && !matches!(
                answer.as_str(),
                "yes" | "accept" | "no" | "off" | "quit" | "exit" | "login" | "demo" | "demos"
            )
        {
            return None;
        }
        self.key(code)
    }

    /// The text of the current step. URLs are kept whole on their own lines.
    pub fn lines(&self) -> Vec<String> {
        let count = self.steps.len();
        let header = |title: &str| {
            format!(
                "Welcome to the jAIn CLI by VEOX ({}/{count}) · {title}",
                self.index + 1
            )
        };
        match self.step() {
            Some(Step::Terms) => vec![
                header("Terms of Service"),
                String::new(),
                format!(
                    "Please read the VEOX Terms of Service (version {}):",
                    self.terms_version
                ),
                TERMS_URL.into(),
                String::new(),
                "Press Enter (or y) to accept and continue · q to quit".into(),
            ],
            Some(Step::Telemetry) => vec![
                header("Usage statistics"),
                String::new(),
                "Help improve jAIn by sending anonymous usage statistics: counts, bucketed sizes"
                    .into(),
                "and scores rounded to tenths; never dataset rows, column names or file paths. Details:".into(),
                PRIVACY_URL.into(),
                String::new(),
                "Press Enter (or y) to share statistics · n to turn them off".into(),
                "Change it any time: jain cloud telemetry on|off · DO_NOT_TRACK=1 always wins"
                    .into(),
            ],
            Some(Step::Account) => vec![
                header("Get started"),
                String::new(),
                "1  Log in to your free veox Cloud Workspace / create an account (opens your browser)".into(),
                "2  Continue with the bundled demos only".into(),
                String::new(),
                "Press Enter or 1 to log in · 2 for demos · you can /login at any time".into(),
            ],
            None => Vec::new(),
        }
    }
}

/// Persist a finished dialog and apply the telemetry choice to the same setting that
/// `jain cloud telemetry on|off` changes.
pub fn record(home: &Path, outcome: Outcome, terms_version: &str, source: &str) -> Result<()> {
    let Outcome::Finished {
        terms_accepted,
        telemetry,
        ..
    } = outcome
    else {
        return Ok(());
    };
    let mut state = load(home);
    let now = crate::cloud::chrono_like_now();
    if terms_accepted {
        state.accept_terms_version(terms_version, source, now.clone());
    }
    if let Some(enabled) = telemetry {
        state.schema = STATE_SCHEMA.into();
        state.telemetry = Some(enabled);
        state.telemetry_decided_at = Some(now);
        crate::cloud::set_telemetry(enabled)?;
    }
    save(home, &state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn state_round_trips_privately_and_reprompts_on_new_terms() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        assert_eq!(load(&home), State::default());
        let mut state = State::default();
        state.accept_terms_version(TERMS_VERSION, "tui", "2026-11-03T00:00:00Z".into());
        state.telemetry = Some(false);
        state
            .extra
            .insert("future_field".into(), serde_json::json!(1));
        save(&home, &state).unwrap();
        assert_eq!(load(&home), state);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(state_path(&home))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert!(FirstRun::new(&load(&home), TERMS_VERSION, false, false).is_none());

        // An older accepted version asks for the terms again, and only for them.
        state.terms_version = Some("2020-01-01".into());
        save(&home, &state).unwrap();
        let dialog = FirstRun::new(&load(&home), TERMS_VERSION, false, false).unwrap();
        assert_eq!(dialog.steps, [Step::Terms]);

        std::fs::write(state_path(&home), b"{not json").unwrap();
        assert_eq!(load(&home), State::default());
    }

    #[test]
    fn a_newer_workspace_terms_version_reprompts_and_an_older_one_does_not() {
        assert_eq!(required_terms_version(None), TERMS_VERSION);
        assert_eq!(required_terms_version(Some("2020-01-01")), TERMS_VERSION);
        assert_eq!(required_terms_version(Some("2027-01-01")), "2027-01-01");
        let mut state = State::default();
        state.accept_terms_version(TERMS_VERSION, "tui", "t".into());
        state.telemetry = Some(true);
        assert!(FirstRun::new(&state, TERMS_VERSION, false, true).is_none());
        let dialog = FirstRun::new(&state, "2027-01-01", false, true).unwrap();
        assert_eq!(dialog.steps, [Step::Terms]);
        assert!(dialog
            .lines()
            .iter()
            .any(|line| line.contains("2027-01-01")));
    }

    #[cfg(unix)]
    #[test]
    fn state_is_never_written_through_a_symlink() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("elsewhere");
        std::fs::write(&target, b"{}").unwrap();
        std::os::unix::fs::symlink(&target, state_path(root.path())).unwrap();
        assert_eq!(load(root.path()), State::default());
        assert!(save(root.path(), &State::default()).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"{}");
    }

    #[test]
    fn do_not_track_and_jain_telemetry_force_telemetry_off() {
        assert!(telemetry_forced_off(env(&[("DO_NOT_TRACK", "1")])));
        assert!(telemetry_forced_off(env(&[("DO_NOT_TRACK", "yes")])));
        assert!(!telemetry_forced_off(env(&[("DO_NOT_TRACK", "0")])));
        assert!(!telemetry_forced_off(env(&[("DO_NOT_TRACK", "")])));
        assert!(telemetry_forced_off(env(&[("JAIN_TELEMETRY", "0")])));
        assert!(telemetry_forced_off(env(&[("JAIN_TELEMETRY", "off")])));
        assert!(!telemetry_forced_off(env(&[("JAIN_TELEMETRY", "1")])));
        assert!(!telemetry_forced_off(env(&[])));
        // Forced off: the telemetry question is skipped entirely.
        let dialog = FirstRun::new(&State::default(), TERMS_VERSION, true, false).unwrap();
        assert_eq!(dialog.steps, [Step::Terms, Step::Account]);
    }

    #[test]
    fn first_run_defaults_accept_terms_enable_telemetry_and_log_in() {
        let mut dialog = FirstRun::new(&State::default(), TERMS_VERSION, false, false).unwrap();
        assert_eq!(dialog.steps, [Step::Terms, Step::Telemetry, Step::Account]);
        assert!(dialog.lines().iter().any(|line| line == TERMS_URL));
        assert_eq!(dialog.key(KeyCode::Char('x')), None);
        assert_eq!(dialog.key(KeyCode::Enter), None);
        assert_eq!(dialog.step(), Some(Step::Telemetry));
        assert_eq!(dialog.key(KeyCode::Enter), None);
        assert_eq!(
            dialog.key(KeyCode::Enter),
            Some(Outcome::Finished {
                terms_accepted: true,
                telemetry: Some(true),
                choice: Some(Choice::Login)
            })
        );
    }

    #[test]
    fn first_run_opt_out_demos_and_quit_paths() {
        let mut dialog = FirstRun::new(&State::default(), TERMS_VERSION, false, false).unwrap();
        assert_eq!(dialog.line("y"), None);
        assert_eq!(dialog.line("n"), None);
        assert_eq!(
            dialog.line("2"),
            Some(Outcome::Finished {
                terms_accepted: true,
                telemetry: Some(false),
                choice: Some(Choice::Demos)
            })
        );
        let mut dialog = FirstRun::new(&State::default(), TERMS_VERSION, false, true).unwrap();
        assert_eq!(dialog.steps, [Step::Terms, Step::Telemetry]);
        assert_eq!(dialog.key(KeyCode::Char('q')), Some(Outcome::Quit));
        let mut dialog = FirstRun::new(&State::default(), TERMS_VERSION, false, false).unwrap();
        assert_eq!(dialog.line("nonsense"), None);
        assert_eq!(dialog.step(), Some(Step::Terms));
        assert_eq!(dialog.line("n"), Some(Outcome::Quit));
    }

    #[test]
    fn anonymous_policy_allows_signed_in_and_overridden_sessions_only() {
        assert!(local_file_gate(true, false).is_ok());
        assert!(local_file_gate(false, true).is_ok());
        let message = local_file_gate(false, false).unwrap_err();
        assert!(message.contains("/login") && message.contains("/demo"));
    }
}
