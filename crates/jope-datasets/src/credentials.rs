//! AWS credential resolution for the S3 backend.
//!
//! `object_store`'s [`AmazonS3Builder::from_env`] reads `AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`, plus web-identity and container-provider
//! configuration, from the environment. With no provider it falls back to EC2 instance metadata. It never executes `credential_process`, which is how every
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
use std::{ffi::OsString, process::Command};

/// One resolved credential set. `Debug` is deliberately not derived.
#[derive(Clone)]
pub struct Credentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    /// Present for STS/SSO credentials; absent for a static IAM user key.
    pub session_token: Option<String>,
}

impl Credentials {
    /// Bind exactly this profile's credentials, replacing any ambient session token.
    pub(crate) fn apply_to(
        self,
        builder: object_store::aws::AmazonS3Builder,
    ) -> object_store::aws::AmazonS3Builder {
        builder.with_credentials(std::sync::Arc::new(
            object_store::StaticCredentialProvider::new(object_store::aws::AwsCredential {
                key_id: self.access_key_id,
                secret_key: self.secret_access_key,
                token: self.session_token,
            }),
        ))
    }
}

/// Choose a profile or defer to object_store's native provider chain. Injecting the
/// lookup keeps provider tests independent of the host's credentials.
fn selected_profile(get: impl Fn(&str) -> Option<OsString>) -> Result<Option<String>> {
    let present = |name| get(name).is_some_and(|v| !v.is_empty());
    if present("AWS_ACCESS_KEY_ID") && present("AWS_SECRET_ACCESS_KEY") {
        return Ok(None);
    }
    for name in ["JOPE_AWS_PROFILE", "AWS_PROFILE"] {
        if let Some(value) = get(name).filter(|v| !v.is_empty()) {
            return Ok(Some(value.to_string_lossy().into_owned()));
        }
    }
    // object_store 0.14's web-identity, ECS and EKS providers precede IMDS.
    // Preserve them; an explicitly selected profile still takes precedence.
    if (present("AWS_WEB_IDENTITY_TOKEN_FILE") && present("AWS_ROLE_ARN"))
        || present("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        || (present("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            && present("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"))
        || get("JOPE_AWS_ALLOW_IMDS").is_some_and(|v| v == "1")
    {
        return Ok(None);
    }
    bail!(
        "no complete AWS credential source: a complete static key pair, selected profile, \
         web-identity or container provider is required. Select AWS_PROFILE=<profile> \
         (or JOPE_AWS_PROFILE), configure a native provider, use JOPE_DATASETS_ANON=1 \
         for unsigned requests, or explicitly allow EC2 instance metadata with \
         JOPE_AWS_ALLOW_IMDS=1. Without a source the metadata fallback at 169.254.169.254 \
         can stall before reporting a misleading network error."
    )
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
        // A credential_process controls stderr too; never disclose either captured stream.
        bail!(
            "aws configure export-credentials failed for profile {profile} ({})",
            output.status
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

/// Resolve a selected profile or let object_store use a configured native provider.
/// No process environment variables are changed.
pub fn resolve() -> Result<Option<Credentials>> {
    selected_profile(|name| std::env::var_os(name))?
        .map(|profile| from_profile(&profile))
        .transpose()
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

    fn select(env: &[(&str, &str)]) -> Result<Option<String>> {
        selected_profile(|key| env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.into()))
    }

    #[test]
    fn preserves_native_oidc_and_container_providers() {
        for env in [
            vec![
                ("AWS_WEB_IDENTITY_TOKEN_FILE", "/token"),
                ("AWS_ROLE_ARN", "role"),
            ],
            vec![("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/credentials")],
            vec![
                (
                    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                    "http://localhost/credentials",
                ),
                ("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", "/token"),
            ],
            vec![("JOPE_AWS_ALLOW_IMDS", "1")],
        ] {
            assert_eq!(select(&env).unwrap(), None);
        }
    }

    #[test]
    fn rejects_missing_partial_and_empty_sources() {
        for env in [
            vec![],
            vec![("AWS_ACCESS_KEY_ID", "key")],
            vec![("AWS_SECRET_ACCESS_KEY", "secret")],
            vec![("AWS_ACCESS_KEY_ID", "key"), ("AWS_SECRET_ACCESS_KEY", "")],
            vec![("AWS_WEB_IDENTITY_TOKEN_FILE", "/token")],
            vec![("AWS_ROLE_ARN", "role")],
            vec![(
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "http://localhost/credentials",
            )],
            vec![("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "")],
            vec![("JOPE_AWS_ALLOW_IMDS", "0")],
        ] {
            let error = select(&env).unwrap_err().to_string();
            assert!(error.contains("no complete AWS credential source"));
            assert!(error.contains("AWS_PROFILE"));
            assert!(error.contains("169.254.169.254"));
        }
    }

    #[test]
    fn explicit_precedence_is_static_keys_then_jope_profile_then_aws_profile() {
        assert_eq!(
            select(&[
                ("AWS_ACCESS_KEY_ID", "key"),
                ("AWS_SECRET_ACCESS_KEY", "secret"),
                ("JOPE_AWS_PROFILE", "jope")
            ])
            .unwrap(),
            None
        );
        assert_eq!(
            select(&[
                ("JOPE_AWS_PROFILE", "jope"),
                ("AWS_PROFILE", "aws"),
                ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/credentials")
            ])
            .unwrap()
            .as_deref(),
            Some("jope")
        );
        assert_eq!(
            select(&[
                ("JOPE_AWS_PROFILE", ""),
                ("AWS_PROFILE", "aws"),
                ("JOPE_AWS_ALLOW_IMDS", "1")
            ])
            .unwrap()
            .as_deref(),
            Some("aws")
        );
    }
    #[test]
    fn profile_replaces_ambient_session_token() {
        for token in [None, Some("profile-token")] {
            let builder = object_store::aws::AmazonS3Builder::new()
                .with_bucket_name("offline-fixture")
                .with_region("us-east-2")
                .with_token("stale-token");
            let profile = Credentials {
                access_key_id: "profile-key".into(),
                secret_access_key: "profile-secret".into(),
                session_token: token.map(str::to_owned),
            };
            let store = profile.apply_to(builder).build().unwrap();
            let rt = tokio::runtime::Runtime::new().unwrap();
            let actual = rt.block_on(store.credentials().get_credential()).unwrap();
            assert_eq!(actual.key_id, "profile-key");
            assert_eq!(actual.secret_key, "profile-secret");
            assert_eq!(actual.token.as_deref(), token);
        }
    }
}
