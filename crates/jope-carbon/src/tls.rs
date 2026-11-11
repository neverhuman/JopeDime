//! Shared HTTPS `ureq` agent configured explicitly with the workspace's sole `rustls` crypto
//! provider (`aws_lc_rs`) so `ureq/rustls-no-provider` never introduces a second provider.

#[cfg(feature = "edge")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "edge")]
use ureq::tls::{TlsConfig, TlsProvider};

/// Return a process-shared `ureq::Agent` bound to `rustls::crypto::aws_lc_rs::default_provider()`
/// and Mozilla WebPKI root certificates.
#[cfg(feature = "edge")]
pub fn carbon_https_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .tls_config(
                TlsConfig::builder()
                    .provider(TlsProvider::Rustls)
                    .unversioned_rustls_crypto_provider(Arc::new(
                        rustls::crypto::aws_lc_rs::default_provider(),
                    ))
                    .build(),
            )
            .build()
            .new_agent()
    })
}
