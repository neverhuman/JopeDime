//! Versioned bounded-run contract for repository-managed multi-host execution (W7).
//!
//! This lives alongside the fixed `jope.native_loop.v1` run contract; it never reinterprets
//! an old manifest. Every record here is source- and fixture-level: nothing in this module
//! contacts another host, probes a network, reads live GPU state or starts numerical work.
//! A same-host fixture exercising these types is a fixture, not the three-host proof (W7-T07).
//!
//! * [`census`] — typed host census and admission preflight refusals (W7-T01, source half).
//! * [`contract`] — the versioned bounded-run contract and its plan validation (W7-T02).
//! * [`controller`] — Plan, read-only Status/Verify, idempotent Cancel, the durable journal,
//!   whole-group drain and coordinator recovery (W7-T02, W7-T05).
//! * [`rendezvous`] — authenticated rendezvous, group admission, bounded framing and
//!   authenticated results (W7-T04).
pub mod census;
pub mod contract;
pub mod controller;
pub mod rendezvous;

use anyhow::{ensure, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::Serialize;
use std::path::{Component, Path};

/// Lowercase 64-hex SHA-256 digest.
pub(crate) fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// Lowercase 40-hex git commit.
pub(crate) fn is_commit(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// Bounded printable identifier used for runs, hosts and work ids.
pub(crate) fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}
/// Relative, normal-component-only path.
pub(crate) fn is_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path.as_os_str().len() <= 1024
        && path.components().all(|c| matches!(c, Component::Normal(_)))
}
pub(crate) fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(crate::manifest::hash(&serde_json::to_vec(value)?))
}
pub(crate) fn verifying_key(hex_key: &str) -> Result<VerifyingKey> {
    let raw: [u8; 32] = hex::decode(hex_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("ed25519 public key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak ed25519 public key");
    Ok(key)
}
fn message<T: Serialize>(domain: &[u8], body: &T) -> Result<Vec<u8>> {
    let mut m = domain.to_vec();
    m.extend(serde_json::to_vec(body)?);
    Ok(m)
}
/// Domain-separated signature over the canonical serde bytes of `body`.
pub fn sign<T: Serialize>(key: &SigningKey, domain: &[u8], body: &T) -> Result<String> {
    Ok(hex::encode(key.sign(&message(domain, body)?).to_bytes()))
}
pub(crate) fn verify<T: Serialize>(
    public_key: &str,
    domain: &[u8],
    body: &T,
    signature: &str,
) -> Result<()> {
    let key = verifying_key(public_key)?;
    let signature = Signature::from_slice(&hex::decode(signature)?)?;
    key.verify_strict(&message(domain, body)?, &signature)?;
    Ok(())
}
