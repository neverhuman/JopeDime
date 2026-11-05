//! "Is a newer jAIn out?" — read the installer's signed stable channel pointer, never install.
//!
//! At most once every 24 hours (cached in the CLI state file), on an interactive start and on
//! `--version` / `doctor`, a background thread GETs [`REGISTRY`]`/channels/stable.channel` with a
//! two-second budget, verifies it with the installer's pinned release key (ai-veox-www
//! `src/cli/release-signing.pub`, ECDSA P-256 / SHA-256), and compares versions. A plain GET with
//! no identifier of any kind. `JAIN_NO_UPDATE_CHECK=1`, `DO_NOT_TRACK=1` or no network turn it
//! off. An expired, malformed or badly signed pointer is ignored (logged only with
//! `JAIN_DEBUG=1`).
use std::time::Duration;

/// The installer's public registry (overridable with `JAIN_REGISTRY`, an HTTPS mirror).
pub const REGISTRY: &str = "https://veox-cli-releases-714044927654.s3.us-west-2.amazonaws.com";
#[cfg_attr(not(feature = "cloud-tls"), allow(dead_code))]
/// The installer's pinned release key, byte for byte (sha256 `de84945e…26e4e`).
pub const RELEASE_KEY_PEM: &str = "-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEMx44DTDE4ObFoh7otiKzAJA9fd8U
eEunkkDgvRCL74VyxIW9Q0wt49FMtTqv7aH7sj//B2uqMwPdm/MlBu45Nw==
-----END PUBLIC KEY-----
";
pub const INSTALL_HINT: &str = "curl -fsSL https://veox.ai/install.sh | sh";
const INTERVAL_SECONDS: u64 = 24 * 60 * 60;
const TIMEOUT: Duration = Duration::from_secs(2);
const MAX_POINTER: u64 = 65_536;

fn debug(message: &str) {
    if std::env::var("JAIN_DEBUG").is_ok_and(|v| v == "1") {
        eprintln!("jain update check: {message}");
    }
}

pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let bytes: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (index, chunk) in bytes.chunks(4).enumerate() {
        let last = index == bytes.len() / 4 - 1;
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        if pad > 2 || (pad > 0 && !last) {
            return None;
        }
        let mut n = 0u32;
        for &c in &chunk[..4 - pad] {
            n = (n << 6) | value(c)?;
        }
        n <<= 6 * pad as u32;
        let group = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&group[..3 - pad]);
    }
    Some(out)
}

#[cfg_attr(not(feature = "cloud-tls"), allow(dead_code))]
/// The uncompressed P-256 point inside the pinned SubjectPublicKeyInfo.
pub fn release_key_point() -> Option<Vec<u8>> {
    const SPKI_P256_PREFIX: [u8; 26] = [
        0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08,
        0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03, 0x42, 0x00,
    ];
    let body: String = RELEASE_KEY_PEM
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();
    let der = base64_decode(&body)?;
    (der.len() == 91 && der[..26] == SPKI_P256_PREFIX && der[26] == 0x04)
        .then(|| der[26..].to_vec())
}

/// A parsed `X.Y.Z` or `X.Y.Z-pre.release` version, ordered as SemVer orders them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Version {
    core: [u64; 3],
    pre: Vec<String>,
}

impl Version {
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.split('+').next()?;
        let (core, pre) = match text.split_once('-') {
            Some((core, pre)) => (core, Some(pre)),
            None => (text, None),
        };
        let mut parts = core.split('.');
        let mut numbers = [0u64; 3];
        for slot in &mut numbers {
            let part = parts.next()?;
            if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            *slot = part.parse().ok()?;
        }
        if parts.next().is_some() {
            return None;
        }
        let pre = match pre {
            Some(pre) => {
                let ids: Vec<String> = pre.split('.').map(str::to_owned).collect();
                if ids.iter().any(|id| {
                    id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                }) {
                    return None;
                }
                ids
            }
            None => Vec::new(),
        };
        Some(Self { core: numbers, pre })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        self.core.cmp(&other.core).then_with(|| {
            match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                // A release outranks its own pre-releases: 1.0.0 > 1.0.0-rc.9.
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => {
                    for (a, b) in self.pre.iter().zip(&other.pre) {
                        let order = match (a.parse::<u64>(), b.parse::<u64>()) {
                            (Ok(a), Ok(b)) => a.cmp(&b),
                            (Ok(_), Err(_)) => Ordering::Less,
                            (Err(_), Ok(_)) => Ordering::Greater,
                            (Err(_), Err(_)) => a.cmp(b),
                        };
                        if order != Ordering::Equal {
                            return order;
                        }
                    }
                    self.pre.len().cmp(&other.pre.len())
                }
            }
        })
    }
}

/// Whether `available` is newer than the running `current`.
pub fn newer(current: &str, available: &str) -> bool {
    match (Version::parse(current), Version::parse(available)) {
        (Some(current), Some(available)) => available > current,
        _ => false,
    }
}

/// Verify a `veox-cli-channel-v1` envelope (body lines, then `signature <base64 DER>`) and
/// return its version. `verify` checks the ECDSA signature over the body bytes.
pub fn parse_pointer(
    envelope: &[u8],
    now: u64,
    verify: impl Fn(&[u8], &[u8]) -> bool,
) -> Result<String, &'static str> {
    if envelope.len() as u64 > MAX_POINTER {
        return Err("pointer too large");
    }
    let text = std::str::from_utf8(envelope).map_err(|_| "pointer is not UTF-8")?;
    let trimmed = text.strip_suffix('\n').unwrap_or(text);
    let (body, last) = trimmed.rsplit_once('\n').ok_or("pointer has no body")?;
    let signature = last.strip_prefix("signature ").ok_or("missing signature")?;
    let der = base64_decode(signature).ok_or("invalid signature encoding")?;
    let body = format!("{body}\n");
    if !verify(body.as_bytes(), &der) {
        return Err("signature verification failed");
    }
    let field = |key: &str| -> Result<&str, &'static str> {
        let mut found = None;
        for line in body.lines() {
            let mut words = line.split_whitespace();
            if words.next() == Some(key) {
                let value = words.next().ok_or("malformed field")?;
                if words.next().is_some() || found.replace(value).is_some() {
                    return Err("malformed field");
                }
            }
        }
        found.ok_or("missing field")
    };
    if field("schema")? != "veox-cli-channel-v1" || field("channel")? != "stable" {
        return Err("unsupported channel pointer");
    }
    let expires: u64 = field("expires")?.parse().map_err(|_| "invalid expiry")?;
    if expires <= now {
        return Err("channel pointer expired");
    }
    let version = field("version")?;
    Version::parse(version).ok_or("invalid version")?;
    Ok(version.to_owned())
}

#[cfg(feature = "cloud-tls")]
fn verify_release_signature(body: &[u8], der: &[u8]) -> bool {
    use rustls::SignatureScheme;
    let Some(point) = release_key_point() else {
        return false;
    };
    let provider = rustls::crypto::aws_lc_rs::default_provider();
    provider
        .signature_verification_algorithms
        .mapping
        .iter()
        .find(|(scheme, _)| *scheme == SignatureScheme::ECDSA_NISTP256_SHA256)
        .and_then(|(_, algs)| algs.first())
        .is_some_and(|alg| alg.verify_signature(&point, body, der).is_ok())
}

#[cfg(not(feature = "cloud-tls"))]
fn verify_release_signature(_body: &[u8], _der: &[u8]) -> bool {
    false
}

/// The pointer URL: `JAIN_REGISTRY` if it is a plain HTTPS origin/path, else [`REGISTRY`].
pub fn pointer_url(var: impl Fn(&str) -> Option<String>) -> String {
    let base = var("JAIN_REGISTRY")
        .filter(|r| {
            r.starts_with("https://")
                && !r.contains(['@', '?', '#'])
                && !r.chars().any(|c| c.is_control() || c.is_whitespace())
        })
        .unwrap_or_else(|| REGISTRY.to_owned());
    format!("{}/channels/stable.channel", base.trim_end_matches('/'))
}

pub fn disabled(var: impl Fn(&str) -> Option<String>) -> bool {
    let on = |name: &str| {
        var(name).is_some_and(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        })
    };
    on("JAIN_NO_UPDATE_CHECK") || on("DO_NOT_TRACK")
}

/// One plain GET with the two-second budget. No headers beyond the agent's own.
#[cfg(feature = "client")]
fn fetch(url: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let builder = ureq::Agent::config_builder();
    #[cfg(feature = "cloud-tls")]
    let builder = builder.tls_config(
        ureq::tls::TlsConfig::builder()
            .unversioned_rustls_crypto_provider(std::sync::Arc::new(
                rustls::crypto::aws_lc_rs::default_provider(),
            ))
            .build(),
    );
    let agent = ureq::Agent::new_with_config(
        builder
            .proxy(None)
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_global(Some(TIMEOUT))
            .build(),
    );
    let mut response = agent.get(url).call().map_err(|e| e.to_string())?;
    if response.status().as_u16() != 200 {
        return Err(format!("HTTP {}", response.status().as_u16()));
    }
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(MAX_POINTER + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The status line for an available update.
pub fn line(version: &str) -> String {
    format!("jAIn {version} is available — update: {INSTALL_HINT}")
}

/// Check now if due (or return the cached answer). Blocking, bounded by the two-second budget;
/// callers run it on a background thread.
#[cfg(feature = "client")]
pub fn check(home: &std::path::Path, current: &str) -> Option<String> {
    use crate::terminal::onboarding;
    let var = |name: &str| std::env::var(name).ok();
    if disabled(var) {
        return None;
    }
    let state = onboarding::load(home);
    let at = now();
    if let Some(cached) = &state.update_check {
        if at.saturating_sub(cached.checked_at) < INTERVAL_SECONDS {
            return cached.available.clone().filter(|v| newer(current, v));
        }
    }
    let available = match fetch(&pointer_url(var))
        .and_then(|bytes| parse_pointer(&bytes, at, verify_release_signature).map_err(Into::into))
    {
        Ok(version) => Some(version).filter(|v| newer(current, v)),
        Err(error) => {
            debug(&error);
            None
        }
    };
    let _ = onboarding::update(home, |state| {
        state.update_check = Some(onboarding::UpdateCheck {
            checked_at: at,
            available: available.clone(),
        });
    });
    available
}

/// Start [`check`] on a background thread; the receiver gets at most one status line.
#[cfg(feature = "client")]
pub fn spawn(home: std::path::PathBuf, current: String) -> std::sync::mpsc::Receiver<String> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let _ = std::thread::Builder::new()
        .name("jain-update-check".into())
        .spawn(move || {
            if let Some(version) = check(&home, &current) {
                let _ = sender.send(line(&version));
            }
        });
    receiver
}

/// For `--version` and `doctor`: only for a person at a terminal (scripts are never slowed),
/// wait at most the budget, then give up silently. The line goes to stderr.
#[cfg(feature = "client")]
pub fn notice_blocking(current: &str) {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() {
        return;
    }
    if let Ok(line) = spawn(jope_paths::jopedime_home(), current.to_owned())
        .recv_timeout(TIMEOUT + Duration::from_millis(250))
    {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &[u8] = b"schema veox-cli-channel-v1\nchannel stable\nversion 1.0.0-rc.1\nmanifest cc4cb1bb16d69a32bf2c7d6d5e59a716b17aefc685600ac76f3cc55d1986e06d\nexpires 1794444176\nsignature MEYCIQDdeQ/EpyzvxTHubuOKzSLW8V0N6f1Dd2ouatoZbE2fTQIhAL/vprqFgEIt4NzzDGvr0VIZD4RZIT2z1+oDk+QBSwud\n";

    #[test]
    fn the_pinned_key_is_the_installers() {
        assert_eq!(
            jope_runtime::sha256_hex(RELEASE_KEY_PEM.as_bytes()),
            "de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e"
        );
        assert_eq!(release_key_point().unwrap().len(), 65);
    }

    #[test]
    fn versions_order_like_semver_including_release_candidates() {
        assert!(newer("1.0.0", "1.0.1"));
        assert!(newer("1.0.0", "1.1.0-rc.1"));
        assert!(!newer("1.0.0", "1.0.0"));
        // A binary that reports 1.0.0 is not behind its own rc pointer.
        assert!(!newer("1.0.0", "1.0.0-rc.1"));
        assert!(newer("1.0.0-rc.1", "1.0.0-rc.2"));
        assert!(newer("1.0.0-rc.9", "1.0.0-rc.10"));
        assert!(newer("1.0.0-rc.2", "1.0.0"));
        assert!(!newer("1.0.1", "1.0.0"));
        assert!(!newer("1.0.0", "not-a-version"));
        assert!(Version::parse("1.0").is_none());
    }

    #[test]
    fn a_signed_pointer_is_read_and_a_bad_signature_or_expiry_is_ignored() {
        let ok = |_: &[u8], _: &[u8]| true;
        assert_eq!(
            parse_pointer(LIVE, 1_700_000_000, ok).unwrap(),
            "1.0.0-rc.1"
        );
        assert_eq!(
            parse_pointer(LIVE, 1_800_000_000, ok).unwrap_err(),
            "channel pointer expired"
        );
        let bad = |_: &[u8], _: &[u8]| false;
        assert_eq!(
            parse_pointer(LIVE, 1_700_000_000, bad).unwrap_err(),
            "signature verification failed"
        );
        let unsigned =
            b"schema veox-cli-channel-v1\nchannel stable\nversion 9.0.0\nexpires 1794444176\n";
        assert!(parse_pointer(unsigned, 1_700_000_000, ok).is_err());
        let preview = String::from_utf8_lossy(LIVE).replace("channel stable", "channel preview");
        assert!(parse_pointer(preview.as_bytes(), 1_700_000_000, ok).is_err());
    }

    #[cfg(feature = "cloud-tls")]
    #[test]
    fn the_live_stable_pointer_verifies_with_the_pinned_key_and_a_tampered_one_does_not() {
        assert_eq!(
            parse_pointer(LIVE, 1_700_000_000, verify_release_signature).unwrap(),
            "1.0.0-rc.1"
        );
        let tampered = String::from_utf8_lossy(LIVE).replace("1.0.0-rc.1", "9.9.9");
        assert_eq!(
            parse_pointer(tampered.as_bytes(), 1_700_000_000, verify_release_signature)
                .unwrap_err(),
            "signature verification failed"
        );
    }

    #[test]
    fn opt_outs_and_registry_override() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        assert!(disabled(env(&[("JAIN_NO_UPDATE_CHECK", "1")])));
        assert!(disabled(env(&[("DO_NOT_TRACK", "1")])));
        assert!(!disabled(env(&[("DO_NOT_TRACK", "0")])));
        assert!(!disabled(env(&[])));
        assert_eq!(
            pointer_url(env(&[])),
            format!("{REGISTRY}/channels/stable.channel")
        );
        assert_eq!(
            pointer_url(env(&[("JAIN_REGISTRY", "https://mirror.example/jain/")])),
            "https://mirror.example/jain/channels/stable.channel"
        );
        // Not HTTPS, or carrying credentials: ignored.
        for bad in ["http://mirror.example", "https://u:p@mirror.example"] {
            let pairs: &'static [(&'static str, &'static str)] =
                Box::leak(vec![("JAIN_REGISTRY", bad)].into_boxed_slice());
            assert!(pointer_url(env(pairs)).starts_with(REGISTRY));
        }
    }

    #[cfg(feature = "client")]
    #[test]
    fn a_registry_that_never_answers_times_out_within_the_budget() {
        // Accept the connection, never respond.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let hold = std::thread::spawn(move || {
            let held = listener.accept();
            std::thread::sleep(Duration::from_secs(4));
            drop(held);
        });
        let started = std::time::Instant::now();
        let result = fetch(&format!("http://{address}/channels/stable.channel"));
        assert!(result.is_err());
        assert!(
            started.elapsed() < Duration::from_millis(3500),
            "{:?}",
            started.elapsed()
        );
        drop(hold);
    }

    #[test]
    fn the_notice_names_the_version_and_the_installer() {
        assert_eq!(
            line("1.0.1"),
            "jAIn 1.0.1 is available — update: curl -fsSL https://veox.ai/install.sh | sh"
        );
    }
}
