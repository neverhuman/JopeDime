//! Minimal frozen comparison registry. Adaptive league scheduling is deliberately deferred.
//!
//! This is a worker-local v1 proposal, not a canonical agy schema or a live arena adapter. All
//! measurements, acceptance receipts and identities are caller-supplied evidence references.
use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const LEAGUE_SCHEMA: &str = "jope.selfplay.frozen_league.v1";
pub const REGISTRY_SCHEMA: &str = "jope.selfplay.league_registry.v1";

fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

mod payoffs;
mod readiness;
mod registry;

pub use payoffs::{ContextIdentity, MemberSnapshot, PayoffCell, PayoffManifest, PayoffPurpose};
pub use readiness::{LeagueGate, LeagueGateReason, MatchmakingMemory, Pfsp, TrainingMatchRecord};
pub use registry::{HistoricalCost, LeagueMember, LeagueRegistry, MemberKind};
