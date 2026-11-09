//! The steps of a cloud run that the terminal and `jopedime run --cloud` share: just-in-time
//! login, the account's plan, the remembered per-dataset upload consent, and submit + follow.
use super::cloud_run::{Refusal, Update, Upload};
use super::route::Tier;
use super::*;
use crate::terminal::onboarding;

/// A local table ready to upload: the exact bytes and what the consent states about them.
#[derive(Clone)]
pub struct Prepared {
    pub file_name: String,
    pub bytes: Arc<Vec<u8>>,
    pub sha256: String,
    pub rows: u64,
    pub columns: u64,
}

impl Prepared {
    /// Read the file as it is on disk now; `rows` and `columns` come from its local profile.
    pub fn read(path: &Path, rows: u64, columns: u64) -> Result<Self> {
        const MAX_UPLOAD: u64 = 80 * 1024 * 1024; // Pro's ceiling (tier::PRO_MAX_UPLOAD_BYTES)
        let mut bytes = Vec::new();
        fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?
            .take(MAX_UPLOAD + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 <= MAX_UPLOAD,
            "{} is larger than any plan's 80 MB cloud upload",
            path.display()
        );
        Ok(Self {
            file_name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("dataset.csv")
                .to_owned(),
            sha256: jope_runtime::sha256_hex(&bytes),
            bytes: Arc::new(bytes),
            rows,
            columns,
        })
    }

    /// The upload question, asked once per table (and account). Enter means no.
    pub fn question(&self) -> String {
        format!(
            "Upload {} ({} rows, {} columns, {}) to train in the veox Cloud? [y/N]",
            self.file_name,
            super::route::grouped(self.rows),
            super::route::grouped(self.columns),
            super::route::size(self.bytes.len() as u64)
        )
    }

    pub fn upload(&self) -> Upload<'_> {
        Upload {
            file_name: &self.file_name,
            bytes: &self.bytes,
            rows: self.rows,
            columns: self.columns,
        }
    }
}

/// The key a consent is remembered under: the workspace, the account and the file's hash.
pub fn consent_key(origin: &str, account: &str, sha256: &str) -> String {
    format!("{origin} {account} {sha256}")
}

/// Whether this exact dataset was already approved for this account.
pub fn consented(state: &onboarding::State, key: &str, p: &Prepared) -> bool {
    state.upload_consents.get(key).is_some_and(|c| {
        c.rows == p.rows && c.columns == p.columns && c.bytes == p.bytes.len() as u64
    })
}

pub fn remember_consent(home: &Path, key: &str, p: &Prepared) -> Result<()> {
    onboarding::update(home, |state| {
        state.upload_consents.insert(
            key.to_owned(),
            onboarding::UploadConsent {
                rows: p.rows,
                columns: p.columns,
                bytes: p.bytes.len() as u64,
                at: chrono_like_now(),
            },
        );
    })
    .map(|_| ())
}

/// The account's plan and whether it may start runs (`mayRun`: false while it waits on the
/// waitlist), read from the workspace and remembered for offline checks (MCP).
pub fn admission(c: &Connection) -> Result<(Option<Tier>, bool)> {
    let (tier, may_run) = Client::connected(c)?.account_tier()?;
    let _ = onboarding::update(&jope_paths::jopedime_home(), |state| {
        state.account_tier = tier.map(|t| t.as_str().to_owned());
    });
    Ok((tier, may_run))
}

/// The own-data gate (owner rule 2026-11-06) for the stored login: signed out, or an account
/// still on the waitlist, stops with [`super::route::Gate::message`]. The workspace is asked
/// for the plan; when it cannot be reached the check passes and the workspace decides at upload.
/// On success, the connection and the plan, when known.
pub fn admitted() -> std::result::Result<(Connection, Option<Tier>), super::route::Gate> {
    let Ok(c) = connection() else {
        return Err(super::route::Gate::SignedOut);
    };
    let account = admission(&c).ok();
    super::route::gate(Some(&c.email), account)?;
    Ok((c, account.and_then(|(tier, _)| tier)))
}

pub(crate) fn browser_allowed() -> bool {
    browser_allowed_with(|name| std::env::var(name).ok())
}

/// Whether to try opening a browser: not when `JAIN_NO_BROWSER` is set, and not where none can
/// show (#2147): an SSH session without a forwarded display, or a Linux/BSD machine with no
/// graphical session. `BROWSER` names one explicitly and is always tried.
pub(crate) fn browser_allowed_with(var: impl Fn(&str) -> Option<String>) -> bool {
    let set = |name: &str| var(name).is_some_and(|value| !value.trim().is_empty());
    if var("JAIN_NO_BROWSER").is_some_and(|v| !v.is_empty() && v != "0") {
        return false;
    }
    if set("BROWSER") {
        return true;
    }
    let display = set("DISPLAY") || set("WAYLAND_DISPLAY");
    if (set("SSH_CONNECTION") || set("SSH_TTY")) && !display {
        return false;
    }
    if cfg!(any(target_os = "macos", windows)) {
        return true;
    }
    display
}

/// The device login, blocking until it finishes; `say` receives the pairing lines.
pub fn login_blocking(
    workspace: &str,
    cancel: &AtomicBool,
    say: &mut dyn FnMut(String),
) -> Result<Connection> {
    let client = Client::new(workspace, None, gateway(None, workspace)?)?;
    let device = client.start_login()?;
    say(format!("Open {}", device.verification_uri_complete));
    say(format!(
        "Pairing code: {} · sign in and approve this CLI in your browser.",
        device.user_code
    ));
    if browser_allowed() && open_browser(&device.verification_uri_complete).is_ok() {
        say(
            "Opened your browser at that link; if it did not appear, open the link yourself."
                .into(),
        );
    } else {
        say("Open the link in a browser on any device (no browser was opened here).".into());
    }
    client.finish_login(&device, cancel)
}

/// Submit and follow one cloud run. `Err` from the outer result is a transport failure.
#[allow(clippy::too_many_arguments)]
pub fn submit_and_follow(
    c: &Connection,
    prepared: &Prepared,
    profile: &str,
    fallback: &str,
    target: Option<&str>,
    interval: Duration,
    cancel: &AtomicBool,
    emit: &mut dyn FnMut(Update),
) -> Result<std::result::Result<(), Refusal>> {
    let client = Client::connected(c)?;
    let id = match client.start_cloud_run(&prepared.upload(), profile, fallback)? {
        Ok(id) => id,
        Err(refusal) => return Ok(Err(refusal)),
    };
    emit(Update::Started {
        url: format!("{}/runs/{id}", c.origin),
        id: id.clone(),
    });
    super::cloud_run::follow(&client, &id, target, interval, cancel, &mut *emit);
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(bytes: &[u8], rows: u64) -> Prepared {
        Prepared {
            file_name: "t.csv".into(),
            sha256: jope_runtime::sha256_hex(bytes),
            bytes: Arc::new(bytes.to_vec()),
            rows,
            columns: 3,
        }
    }

    #[test]
    fn consent_is_remembered_per_file_hash_and_account() {
        let home = tempfile::tempdir().unwrap();
        let p = prepared(b"a,b,c\n1,2,3\n", 1);
        let key = consent_key("https://app.veox.ai", "acct-1", &p.sha256);
        assert!(!consented(&onboarding::load(home.path()), &key, &p));
        remember_consent(home.path(), &key, &p).unwrap();
        let state = onboarding::load(home.path());
        assert!(consented(&state, &key, &p), "asked once per dataset");
        // Another account, or changed bytes, asks again.
        let other = consent_key("https://app.veox.ai", "acct-2", &p.sha256);
        assert!(!consented(&state, &other, &p));
        let changed = prepared(b"a,b,c\n1,2,3\n4,5,6\n", 2);
        let changed_key = consent_key("https://app.veox.ai", "acct-1", &changed.sha256);
        assert!(!consented(&state, &changed_key, &changed));
        // Stored with what was shown, never the data.
        let raw = std::fs::read_to_string(onboarding::state_path(home.path())).unwrap();
        assert!(
            raw.contains("\"rows\": 1") && !raw.contains("1,2,3"),
            "{raw}"
        );
    }

    #[test]
    fn the_question_names_file_rows_columns_and_size_and_defaults_to_no() {
        let p = prepared(&vec![b'x'; 2048], 123_456);
        assert_eq!(
            p.question(),
            "Upload t.csv (123,456 rows, 3 columns, 2 KB) to train in the veox Cloud? [y/N]"
        );
    }
}
