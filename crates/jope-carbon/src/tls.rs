//! Shared HTTPS `ureq` agents configured explicitly with the workspace's sole `rustls` crypto
//! provider (`aws_lc_rs`), strict zero-redirect credential safety (`max_redirects = 0`), bounded
//! socket/read/write timeouts, and merged Enterprise PEM root CA support (`CARBON_CA_BUNDLE`).

#[cfg(feature = "edge")]
use std::fs;
#[cfg(feature = "edge")]
use std::path::Path;
#[cfg(feature = "edge")]
use std::sync::{Arc, OnceLock};
#[cfg(feature = "edge")]
use std::time::Duration;

#[cfg(feature = "edge")]
use ureq::tls::{parse_pem, Certificate, PemItem, RootCerts, TlsConfig, TlsProvider};

#[cfg(feature = "edge")]
const SYSTEM_PEM_BUNDLE_CANDIDATES: &[&str] = &[
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/ssl/cert.pem",
];

/// Load custom PEM root certificates from `path` for Enterprise VPC / private-PKI deployments.
#[cfg(feature = "edge")]
pub fn load_custom_pem_root_certs(path: &Path) -> Result<Vec<Certificate<'static>>, String> {
    let pem_bytes = fs::read(path)
        .map_err(|e| format!("failed to read custom CA bundle '{}': {e}", path.display()))?;
    let mut certs = Vec::new();
    for item in parse_pem(&pem_bytes) {
        match item {
            Ok(PemItem::Certificate(cert)) => certs.push(cert.to_owned()),
            Ok(_) => {}
            Err(e) => {
                return Err(format!(
                    "invalid PEM certificate in '{}': {e}",
                    path.display()
                ));
            }
        }
    }
    if certs.is_empty() {
        return Err(format!(
            "custom CA bundle '{}' contained zero X.509 certificates",
            path.display()
        ));
    }
    Ok(certs)
}

/// Fail-closed preflight check when `CARBON_CA_BUNDLE` is explicitly set by the user/enterprise admin.
/// Prevents silent fallback to default roots when a configured corporate CA path has a typo or bad PEM.
#[cfg(feature = "edge")]
pub fn validate_tls_env_config() -> Result<(), String> {
    if let Ok(explicit) = std::env::var("CARBON_CA_BUNDLE") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            let _ = load_custom_pem_root_certs(Path::new(trimmed))
                .map_err(|e| format!("CARBON_CA_BUNDLE validation failed: {e}"))?;
        }
    }
    Ok(())
}

#[cfg(not(feature = "edge"))]
pub fn validate_tls_env_config() -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "edge")]
fn resolve_root_certs() -> RootCerts {
    let custom_certs = if let Ok(explicit) = std::env::var("CARBON_CA_BUNDLE") {
        let trimmed = explicit.trim();
        if trimmed.is_empty() {
            None
        } else {
            match load_custom_pem_root_certs(Path::new(trimmed)) {
                Ok(certs) => Some(certs),
                Err(err) => {
                    eprintln!("error: CARBON_CA_BUNDLE invalid ({err}); refusing silent fallback");
                    return RootCerts::Specific(Arc::new(Vec::new()));
                }
            }
        }
    } else if std::env::var("CARBON_TRUST_SSL_CERT_FILE")
        .ok()
        .is_some_and(|v| v.trim() == "1" || v.eq_ignore_ascii_case("true"))
    {
        std::env::var("SSL_CERT_FILE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .and_then(|p| load_custom_pem_root_certs(Path::new(p.trim())).ok())
    } else {
        None
    };

    let Some(mut merged) = custom_certs else {
        return RootCerts::WebPki;
    };

    // Merge host system PEM bundle so adding a corporate proxy CA via CARBON_CA_BUNDLE
    // does not break TLS to public AWS S3 or GitHub Releases (`INS-05`).
    for sys_path in SYSTEM_PEM_BUNDLE_CANDIDATES {
        let p = Path::new(sys_path);
        if p.is_file() {
            if let Ok(sys_certs) = load_custom_pem_root_certs(p) {
                merged.extend(sys_certs);
                break;
            }
        }
    }

    RootCerts::new_with_certs(&merged)
}

#[cfg(feature = "edge")]
fn build_tls_config() -> TlsConfig {
    TlsConfig::builder()
        .provider(TlsProvider::Rustls)
        .root_certs(resolve_root_certs())
        .unversioned_rustls_crypto_provider(Arc::new(rustls::crypto::aws_lc_rs::default_provider()))
        .build()
}

/// Return a process-shared `ureq::Agent` bound to `rustls::crypto::aws_lc_rs::default_provider()`,
/// `max_redirects = 0` (preventing capability tokens or SigV4 headers from following redirects),
/// bounded connect/request/response timeouts, and provenance `User-Agent`.
#[cfg(feature = "edge")]
pub fn carbon_https_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .tls_config(build_tls_config())
            .max_redirects(0)
            .user_agent(crate::carbon_user_agent())
            .timeout_resolve(Some(Duration::from_secs(5)))
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_send_request(Some(Duration::from_secs(15)))
            .timeout_send_body(Some(Duration::from_secs(180)))
            .timeout_recv_response(Some(Duration::from_secs(150)))
            .timeout_recv_body(Some(Duration::from_secs(180)))
            .build()
            .new_agent()
    })
}

/// Return a process-shared `ureq::Agent` dedicated to internal AWS Lambda `/invocations` RPC calls
/// (`edge -> engine` and `engine -> eval`), with `315 s` response/body read ceilings strictly above
/// `EvalFunction` (`300 s`) and `EdgeFunction` (`60 s`) so long-running shards return cleanly
/// without premature socket disconnects or retry amplification.
#[cfg(feature = "edge")]
pub fn carbon_lambda_rpc_https_agent() -> &'static ureq::Agent {
    static RPC_AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    RPC_AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .tls_config(build_tls_config())
            .max_redirects(0)
            .user_agent(crate::carbon_user_agent())
            .timeout_resolve(Some(Duration::from_secs(5)))
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_send_request(Some(Duration::from_secs(15)))
            .timeout_send_body(Some(Duration::from_secs(60)))
            .timeout_recv_response(Some(Duration::from_secs(315)))
            .timeout_recv_body(Some(Duration::from_secs(315)))
            .build()
            .new_agent()
    })
}

/// Return a bounded `ureq::Agent` (`3.0 s` connect, `8.0 s` cold-start response, `max_redirects = 0`)
/// used by parallel regional `/v1/carbon/ping` RTT discovery and `/v1/carbon/version` probes so a
/// blackholed region never stalls startup while Lambda cold starts in distant regions succeed reliably.
#[cfg(feature = "edge")]
pub fn carbon_probe_https_agent() -> &'static ureq::Agent {
    static PROBE_AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    PROBE_AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .tls_config(build_tls_config())
            .max_redirects(0)
            .user_agent(crate::carbon_user_agent())
            .timeout_resolve(Some(Duration::from_millis(2_500)))
            .timeout_connect(Some(Duration::from_millis(3_000)))
            .timeout_send_request(Some(Duration::from_millis(3_000)))
            .timeout_recv_response(Some(Duration::from_millis(8_000)))
            .timeout_recv_body(Some(Duration::from_millis(8_000)))
            .build()
            .new_agent()
    })
}

#[cfg(all(test, feature = "edge"))]
mod tests {
    use super::*;

    #[test]
    fn custom_pem_loader_rejects_empty_and_missing_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("missing.pem");
        assert!(load_custom_pem_root_certs(&missing).is_err());

        let empty = dir.path().join("empty.pem");
        std::fs::write(&empty, b"# comment only\n").expect("write empty");
        let err = load_custom_pem_root_certs(&empty).expect_err("must reject empty cert bundle");
        assert!(err.contains("zero X.509 certificates"), "{err}");
    }
}
