//! Versioned bounded-run contract for repository-managed multi-host execution (W7).
//!
//! This lives alongside the fixed `jope.native_loop.v1` run contract; it never reinterprets
//! an old manifest. Nothing here contacts another host, reads live GPU state or starts
//! numerical work on its own: the census collector reads only the local host, and the
//! transport connects only where an operator-run coordinator, participant or agent directs it.
//! A same-host fixture exercising these types is a fixture, not the three-host proof (W7-T07).
//!
//! * [`census`] — typed host census and admission preflight refusals (W7-T01, source half).
//! * [`collect`] — local-host census collection and pinned per-host census merge (W7-T01).
//!   The collector reads only the host it runs on and never allocates or reserves anything.
//! * [`contract`] — the versioned bounded-run contract and its plan validation (W7-T02).
//! * [`controller`] — Plan, read-only Status/Verify, idempotent Cancel, the durable journal,
//!   whole-group drain and coordinator recovery (W7-T02, W7-T05).
//! * [`launch`] — the frozen per-rank launch record, the launcher's exact worker environment
//!   and the worker's own visibility check (W7-T03 delivery half).
//! * [`rendezvous`] — authenticated rendezvous, group admission, bounded framing and
//!   authenticated results (W7-T04).
//! * [`transport`] — the cross-host TCP transport: TLS 1.3 with mutual certificate identity
//!   pinned by the plan, deadlines and delivery ids (W7-T04/T07).
//! * [`supervisor`] — the SmartCluster-backed [`controller::Supervisor`]: ranks are trusted
//!   service jobs of the host's own daemon, through its authenticated exchange (W7-T05/T07).
//! * [`participant`] — the rank worker of the CPU phase: frozen launch, own host identity,
//!   authenticated rendezvous over TLS, one update round per heartbeat, signed result.
//! * [`driver`] — the coordinator process: preflight, dispatch, gather, admission, heartbeat
//!   and poll to a terminal outcome, or reconcile after a restart.
//! * [`agent`] — cross-host supervision: a per-host agent serves its host's supervisor to the
//!   coordinator over the pinned transport, and a remote supervisor routes each rank to it.
//! * `assignment` (`neural` only) — the pure plan → trainer assignment join (W7-T03).
pub mod agent;
/// Plan → trainer assignment join (W7-T02 × W7-T03); the trainer is `neural`-gated.
#[cfg(feature = "neural")]
pub mod assignment;
pub mod census;
pub mod collect;
pub mod contract;
pub mod controller;
pub mod driver;
pub mod launch;
pub mod participant;
pub mod rendezvous;
pub mod supervisor;
pub mod transport;

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
