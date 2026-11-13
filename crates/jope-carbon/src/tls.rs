//! Shared HTTPS `ureq` agents configured explicitly with the workspace's sole `rustls` crypto
//! provider (`aws_lc_rs`), strict zero-redirect credential safety (`max_redirects = 0`), bounded
//! socket/read/write timeouts, and optional Enterprise PEM root CA support (`CARBON_CA_BUNDLE`).

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

#[cfg(feature = "edge")]
fn resolve_root_certs() -> RootCerts {
    let custom_ca_path = std::env::var("CARBON_CA_BUNDLE")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| {
            if std::env::var("CARBON_TRUST_SSL_CERT_FILE")
                .ok()
                .is_some_and(|v| v.trim() == "1" || v.eq_ignore_ascii_case("true"))
            {
                std::env::var("SSL_CERT_FILE")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            } else {
                None
            }
        });
    if let Some(raw_path) = custom_ca_path {
        if let Ok(certs) = load_custom_pem_root_certs(Path::new(raw_path.trim())) {
            return RootCerts::new_with_certs(&certs);
        }
    }
    RootCerts::WebPki
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
            .timeout_recv_response(Some(Duration::from_secs(90)))
            .timeout_recv_body(Some(Duration::from_secs(180)))
            .build()
            .new_agent()
    })
}

/// Return a fast-fail `ureq::Agent` (`2.5 s` connect, `4.0 s` response, `max_redirects = 0`)
/// used by parallel regional `/v1/carbon/ping` RTT discovery so a blackholed region never stalls startup.
#[cfg(feature = "edge")]
pub fn carbon_probe_https_agent() -> &'static ureq::Agent {
    static PROBE_AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    PROBE_AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .tls_config(build_tls_config())
            .max_redirects(0)
            .user_agent(crate::carbon_user_agent())
            .timeout_resolve(Some(Duration::from_millis(2_000)))
            .timeout_connect(Some(Duration::from_millis(2_500)))
            .timeout_send_request(Some(Duration::from_millis(2_500)))
            .timeout_recv_response(Some(Duration::from_millis(4_000)))
            .timeout_recv_body(Some(Duration::from_millis(4_000)))
            .build()
            .new_agent()
    })
}
