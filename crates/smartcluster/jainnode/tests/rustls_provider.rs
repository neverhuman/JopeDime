//! Guard: the build must resolve exactly one rustls crypto provider.
//!
//! `rustls::ClientConfig::builder()` (and every other feature-inferred entry point) resolves its
//! provider through the private `CryptoProvider::from_crate_features()`, which returns `None` when
//! the rustls crate features name **two** providers — e.g. `aws_lc_rs` from this crate and `ring`
//! pulled in by some HTTP client elsewhere in the workspace — or none at all. `rustls` then
//! panics, and the panic only reaches a *whole-workspace* build, where cargo unifies features; a
//! `-p jainnode` build looks healthy. Nothing in this workspace installs a process-wide provider,
//! so the panic used to surface at the first outbound mTLS control connection, with no test
//! covering it.
//!
//! This file is deliberately a single test in its own binary: it must observe the process before
//! anything has called `CryptoProvider::install_default()`, which would mask the ambiguity.
//!
//! `deploy/ci/check.sh` guards the same property statically (`cargo tree -e features -i rustls`),
//! so the hygiene lane catches a re-introduced provider without building anything.

use rustls::{ClientConfig, RootCertStore};

#[test]
fn rustls_crate_features_name_exactly_one_crypto_provider() {
    assert!(
        rustls::crypto::CryptoProvider::get_default().is_none(),
        "a process-wide CryptoProvider was already installed, which would mask an ambiguous \
         feature set; this guard must be the only test in its binary"
    );

    // Panics inside rustls when the resolved feature set is ambiguous (ring + aws_lc_rs) or empty.
    let _config = ClientConfig::builder()
        .with_root_certificates(RootCertStore::empty())
        .with_no_client_auth();

    assert!(
        rustls::crypto::CryptoProvider::get_default().is_some(),
        "rustls resolved a provider from the crate features but did not install it"
    );
}
