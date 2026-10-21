//! Compiles the release signing key into the crate.
//!
//! `docs/release/signing.pub` is the public half of the root-held cosign key (ECDSA P-256, SPKI
//! PEM) that already signs every release's `SHA256SUMS` (`deploy/release/release.sh`). The
//! appliance trusts the same authority for its embedded package signature, so no second release
//! key exists to provision or rotate (deploy/appliance/AMENDMENTS.md A6).
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by Cargo"));
    let key = manifest_dir.join("../../docs/release/signing.pub");
    println!("cargo:rerun-if-changed={}", key.display());
    let pem = std::fs::read_to_string(&key)
        .unwrap_or_else(|error| panic!("read {}: {error}", key.display()));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("set by Cargo"));
    std::fs::write(
        out.join("release_keys.rs"),
        format!(
            "/// Release signing keys (SPKI PEM): `docs/release/signing.pub`.\n\
             pub const RELEASE_KEYS_PEM: &[&str] = &[{pem:?}];\n"
        ),
    )
    .expect("write release_keys.rs");
}
