//! AWS credential resolution for the S3 backend.
//!
//! `object_store`'s [`AmazonS3Builder::from_env`] reads `AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` from the environment and then falls back to the
//! EC2 instance metadata service. It never executes `credential_process`, which is how every
//! profile in `~/.aws/config` supplies credentials on the authoring hosts. So a host with working
//! credentials waits ~13 s retrying an address it cannot reach and then fails with an HTTP error
//! naming the metadata endpoint, which reads as a network fault rather than "no credentials":
//!
//! ```text
//! error: get datasets/0014b9dad5725a1e/manifest.json: Generic S3 error: Error performing
//! PUT http://169.254.169.254/latest/api/token in 12.942946932s, after 10 retries,
//! max_retries: 10, retry_timeout: 180s - HTTP error: error sending request
//! ```
//!
//! The same fetch succeeds once the profile's credentials are in the environment, so this module
//! resolves them and hands them to the builder explicitly.
//!
//! Two deliberate choices. The process environment is **not** mutated: `std::env::set_var` races
//! every other thread in the process, and on this host `credential_process` returns a long-lived
//! access key and secret with no session token, so leaving those in a shared place is worse than
//! passing them once. And when nothing resolves this fails **fast and by name** rather than
//! letting the IMDS fallback produce that error above — with `JOPE_AWS_ALLOW_IMDS=1` for the case
//! where the fallback is genuinely right, an EC2 instance carrying an instance role.
//!
//! [`AmazonS3Builder::from_env`]: object_store::aws::AmazonS3Builder::from_env

use anyhow::{bail, Context, Result};
use std::process::Command;

/// One resolved credential set. `Debug` is deliberately not derived.
#[derive(Clone)]
pub struct Credentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    /// Present for STS/SSO credentials; absent for a static IAM user key.
    pub session_token: Option<String>,
}

/// Whether the environment already carries what `from_env` needs, in which case object_store
/// reads it directly and this module stays out of the way.
pub fn environment_has_credentials() -> bool {
    std::env::var_os("AWS_ACCESS_KEY_ID").is_some_and(|v| !v.is_empty())
        && std::env::var_os("AWS_SECRET_ACCESS_KEY").is_some_and(|v| !v.is_empty())
}

/// The profile to evaluate: `JOPE_AWS_PROFILE` wins so this can be pointed somewhere specific
/// without disturbing whatever else on the host reads `AWS_PROFILE`.
pub fn configured_profile() -> Option<String> {
    for name in ["JOPE_AWS_PROFILE", "AWS_PROFILE"] {
        if let Some(value) = std::env::var_os(name) {
            let value = value.to_string_lossy().into_owned();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

/// Evaluate a profile's credentials through the AWS CLI, which does run `credential_process`.
///
/// Shelling out is not elegant, but the alternative is reimplementing the credential-provider
/// chain, and the CLI is already the thing that defines what a profile means on this host.
pub fn from_profile(profile: &str) -> Result<Credentials> {
    let output = Command::new("aws")
        .args(["configure", "export-credentials", "--format", "env"])
        .env("AWS_PROFILE", profile)
        .output()
        .context(
            "run `aws configure export-credentials`; the AWS CLI is required to evaluate a \
             profile's credential_process",
        )?;
    if !output.status.success() {
        // stderr can name the profile and the failing helper, neither of which is secret.
        bail!(
            "aws configure export-credentials failed for profile {profile}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_export_env(&stdout).ok_or_else(|| {
        // Never echo the output: it is the credential.
        anyhow::anyhow!(
            "aws configure export-credentials produced no key pair for profile {profile}; \
             it printed {} line(s)",
            stdout.lines().count()
        )
    })
}

/// Parse `aws configure export-credentials --format env`, which emits `export KEY=value` lines.
///
/// Split out from [`from_profile`] because this is the part that can be wrong in a way no
/// integration test on this host would catch — the host's own profile happens to return a static
/// key pair with no session token, so the token path would never be exercised.
fn parse_export_env(stdout: &str) -> Option<Credentials> {
    let mut access_key_id = None;
    let mut secret_access_key = None;
    let mut session_token = None;
    for line in stdout.lines() {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "AWS_ACCESS_KEY_ID" => access_key_id = Some(value),
            "AWS_SECRET_ACCESS_KEY" => secret_access_key = Some(value),
            "AWS_SESSION_TOKEN" => session_token = Some(value),
            _ => {}
        }
    }
    Some(Credentials {
        access_key_id: access_key_id?,
        secret_access_key: secret_access_key?,
        session_token,
    })
}

/// Resolve credentials for a signed S3 store, or explain precisely why none are available.
///
/// `Ok(None)` means "the environment already has them, let object_store read it".
pub fn resolve() -> Result<Option<Credentials>> {
    if environment_has_credentials() {
        return Ok(None);
    }
    if let Some(profile) = configured_profile() {
        return from_profile(&profile).map(Some);
    }
    if std::env::var_os("JOPE_AWS_ALLOW_IMDS").is_some_and(|v| v == "1") {
        return Ok(None);
    }
    bail!(
        "no AWS credentials.\n\
         The environment carries no AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and no profile is \
         selected, so object_store would fall back to the EC2 metadata service and fail after \
         ~13 s with an error naming 169.254.169.254.\n\
         Pick one:\n  \
         AWS_PROFILE=<profile> (or JOPE_AWS_PROFILE) — evaluated through the AWS CLI, which does \
         run credential_process\n  \
         eval \"$(aws configure export-credentials --format env)\" — put them in the environment \
         yourself\n  \
         JOPE_DATASETS_ANON=1 — unsigned requests, for a public bucket\n  \
         JOPE_AWS_ALLOW_IMDS=1 — really do use instance metadata (an EC2 instance role)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape this host actually produces: a static IAM key pair, no session token.
    #[test]
    fn parses_a_static_key_pair() {
        let creds = parse_export_env(
            "export AWS_ACCESS_KEY_ID=AKIAEXAMPLE\nexport AWS_SECRET_ACCESS_KEY=s3cr3t\n",
        )
        .expect("a key pair is enough");
        assert_eq!(creds.access_key_id, "AKIAEXAMPLE");
        assert_eq!(creds.secret_access_key, "s3cr3t");
        assert!(creds.session_token.is_none());
    }

    /// The shape an SSO or assumed-role profile produces. Not exercised by this host's own
    /// profile, which is exactly why it is tested here rather than end to end.
    #[test]
    fn carries_a_session_token_when_one_is_present() {
        let creds = parse_export_env(
            "export AWS_ACCESS_KEY_ID=ASIAEXAMPLE\n\
             export AWS_SECRET_ACCESS_KEY=s3cr3t\n\
             export AWS_SESSION_TOKEN=tok\n",
        )
        .expect("a key pair is enough");
        assert_eq!(creds.session_token.as_deref(), Some("tok"));
    }

    #[test]
    fn tolerates_quoting_and_unrelated_lines() {
        let creds = parse_export_env(
            "AWS_REGION=us-east-2\n\
             export AWS_ACCESS_KEY_ID=\"AKIAEXAMPLE\"\n\
             export AWS_SECRET_ACCESS_KEY='s3cr3t'\n\
             \n",
        )
        .expect("quoting and noise must not defeat it");
        assert_eq!(creds.access_key_id, "AKIAEXAMPLE");
        assert_eq!(creds.secret_access_key, "s3cr3t");
    }

    /// A half-resolved profile must be an error, not blank credentials: signing with an empty
    /// secret gets a SignatureDoesNotMatch from S3, which is far harder to trace back than this.
    #[test]
    fn a_partial_or_empty_result_is_not_a_credential() {
        assert!(parse_export_env("export AWS_ACCESS_KEY_ID=AKIAEXAMPLE\n").is_none());
        assert!(parse_export_env(
            "export AWS_ACCESS_KEY_ID=AKIAEXAMPLE\nexport AWS_SECRET_ACCESS_KEY=\n"
        )
        .is_none());
        assert!(parse_export_env("").is_none());
    }

    /// `resolve` must never silently pick the IMDS path: that is the 13-second failure this
    /// module exists to remove.
    #[test]
    fn resolve_explains_itself_when_nothing_is_available() {
        if environment_has_credentials() || configured_profile().is_some() {
            return; // this host can resolve; the no-credential path is covered by the parser tests
        }
        let text = match resolve() {
            Ok(_) => return, // JOPE_AWS_ALLOW_IMDS is set; that is a deliberate opt-in
            Err(err) => err.to_string(),
        };
        assert!(
            text.contains("AWS_PROFILE"),
            "must say how to fix it: {text}"
        );
        assert!(
            text.contains("169.254.169.254"),
            "must name the symptom: {text}"
        );
    }
}
