//! Deterministic executable cache identities and immutable packed evidence metadata.
//!
//! This module is deliberately local and side-effect free. It does not publish objects, change
//! retention tags, or make custody decisions on behalf of a caller. Cache entries are admitted
//! only after two independent emitter runs produce identical bytes.

use crate::canon::sha256_hex;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableCacheKey {
    pub program_sha256: String,
    pub emitter: String,
    pub emitter_version: String,
    pub toolchain: String,
    pub target: String,
    pub numeric_profile: String,
    pub fitted_identity: String,
}

impl ExecutableCacheKey {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            is_sha256(&self.program_sha256),
            "program_sha256 must be 64 lowercase hex characters"
        );
        for (name, value) in [
            ("emitter", &self.emitter),
            ("emitter_version", &self.emitter_version),
            ("toolchain", &self.toolchain),
            ("target", &self.target),
            ("numeric_profile", &self.numeric_profile),
            ("fitted_identity", &self.fitted_identity),
        ] {
            ensure!(!value.trim().is_empty(), "{name} is required");
        }
        Ok(())
    }

    /// Hash the canonical struct representation; every field participates in the key.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackMemberRange {
    pub key: String,
    pub version_id: String,
    pub sha256: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackManifest {
    pub schema: String,
    pub pack_key: String,
    pub pack_version_id: String,
    pub pack_sha256: String,
    pub pack_bytes: u64,
    pub members: Vec<PackMemberRange>,
}

impl PackManifest {
    /// Validate immutable member identity and exact, non-overlapping in-bounds ranges.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.executable-pack/v1",
            "unsupported executable pack schema"
        );
        ensure!(
            !self.pack_key.is_empty() && !self.pack_version_id.is_empty(),
            "pack key and version are required"
        );
        ensure!(
            is_sha256(&self.pack_sha256),
            "pack_sha256 must be 64 lowercase hex characters"
        );
        ensure!(self.pack_bytes > 0, "empty packs are not valid");
        ensure!(
            !self.members.is_empty(),
            "pack must contain at least one member"
        );
        let mut keys = BTreeSet::new();
        let mut ranges = Vec::with_capacity(self.members.len());
        for member in &self.members {
            ensure!(!member.key.is_empty(), "member key is required");
            ensure!(
                !member.version_id.is_empty(),
                "member {} has no immutable version",
                member.key
            );
            ensure!(
                is_sha256(&member.sha256),
                "member {} has invalid sha256",
                member.key
            );
            ensure!(
                member.length > 0,
                "member {} has an empty range",
                member.key
            );
            ensure!(
                keys.insert(&member.key),
                "duplicate member key {}",
                member.key
            );
            let end = member
                .offset
                .checked_add(member.length)
                .ok_or_else(|| anyhow::anyhow!("range overflow for {}", member.key))?;
            ensure!(
                end <= self.pack_bytes,
                "member {} range ends beyond pack",
                member.key
            );
            ranges.push((member.offset, end, member.key.as_str()));
        }
        ranges.sort_by_key(|range| (range.0, range.1));
        for pair in ranges.windows(2) {
            ensure!(
                pair[0].1 <= pair[1].0,
                "member ranges overlap: {} and {}",
                pair[0].2,
                pair[1].2
            );
        }
        Ok(())
    }

    /// Verify bytes, pack digest, length, and every member slice against the declared hashes.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.validate()?;
        ensure!(
            bytes.len() as u64 == self.pack_bytes,
            "pack byte length disagrees with manifest"
        );
        ensure!(
            sha256_hex(bytes) == self.pack_sha256,
            "pack bytes disagree with manifest sha256"
        );
        for member in &self.members {
            let start = usize::try_from(member.offset)?;
            let end = usize::try_from(
                member
                    .offset
                    .checked_add(member.length)
                    .ok_or_else(|| anyhow::anyhow!("range overflow for {}", member.key))?,
            )?;
            ensure!(
                sha256_hex(&bytes[start..end]) == member.sha256,
                "member {} bytes disagree with manifest sha256",
                member.key
            );
        }
        Ok(())
    }
}

/// Tracks admitted immutable pack keys and source object versions. Exact retries are idempotent;
/// reusing a pack or member key with a changed version or digest is rejected.
#[derive(Default)]
pub struct PackCatalog {
    packs: BTreeMap<String, PackManifest>,
    members: BTreeMap<String, (String, String)>,
}

impl PackCatalog {
    pub fn admit(&mut self, manifest: &PackManifest, bytes: &[u8]) -> Result<()> {
        manifest.verify_bytes(bytes)?;
        if let Some(existing) = self.packs.get(&manifest.pack_key) {
            ensure!(
                existing == manifest,
                "pack key already names a different immutable manifest"
            );
        }
        for member in &manifest.members {
            let identity = (member.version_id.clone(), member.sha256.clone());
            if let Some(existing) = self.members.get(&member.key) {
                ensure!(
                    existing == &identity,
                    "member key {} already names a different immutable version",
                    member.key
                );
            }
        }
        self.packs
            .insert(manifest.pack_key.clone(), manifest.clone());
        for member in &manifest.members {
            self.members.insert(
                member.key.clone(),
                (member.version_id.clone(), member.sha256.clone()),
            );
        }
        Ok(())
    }
}

/// Whether the caller has declared every obligation needed to omit nonkeeper evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeeperPolicy {
    pub is_keeper: bool,
    pub proof_obligation_allows_omission: bool,
    pub retention_obligation_allows_omission: bool,
}

impl KeeperPolicy {
    pub fn may_omit(self) -> bool {
        !self.is_keeper
            && self.proof_obligation_allows_omission
            && self.retention_obligation_allows_omission
    }

    pub fn require_keep_or_permitted_omission(self) -> Result<()> {
        ensure!(
            self.is_keeper || self.may_omit(),
            "evidence cannot be omitted without both proof and retention permission"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedExecutable {
    pub key_digest: String,
    pub executable_sha256: String,
    pub bytes: Vec<u8>,
}

/// In-memory immutable candidate cache. Persistence/publication belongs to the integrating layer.
#[derive(Default)]
pub struct ExecutableCache {
    entries: BTreeMap<String, CachedExecutable>,
}

impl ExecutableCache {
    /// Admit output only if two builds under the same full key are byte-identical.
    pub fn insert_verified(
        &mut self,
        key: &ExecutableCacheKey,
        first_build: &[u8],
        second_build: &[u8],
    ) -> Result<&CachedExecutable> {
        let digest = key.digest()?;
        ensure!(!first_build.is_empty(), "empty executable output");
        ensure!(
            first_build == second_build,
            "independent executable builds differ"
        );
        let executable_sha256 = sha256_hex(first_build);
        if let Some(existing) = self.entries.get(&digest) {
            ensure!(
                existing.executable_sha256 == executable_sha256 && existing.bytes == first_build,
                "cache key already names a different immutable executable"
            );
            return Ok(self.entries.get(&digest).expect("entry checked above"));
        }
        self.entries.insert(
            digest.clone(),
            CachedExecutable {
                key_digest: digest.clone(),
                executable_sha256,
                bytes: first_build.to_vec(),
            },
        );
        Ok(self.entries.get(&digest).expect("entry inserted above"))
    }

    pub fn get(&self, key: &ExecutableCacheKey) -> Result<Option<&CachedExecutable>> {
        Ok(self.entries.get(&key.digest()?))
    }
}
