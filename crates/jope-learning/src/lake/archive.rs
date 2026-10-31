//! Bounded quality-diversity proposal memory.
//!
//! Entries are retrieval proposals only. Nothing in this module is a qualification record or
//! a final-champion selector; final measurement and the accepted bank remain independent.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for archive values.
pub const ARCHIVE_SCHEMA: &str = "jope.qd-archive/v1";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `ArchiveScope` is an enum for the archive scope contract in this module: Bounded quality-diversity proposal memory.
pub enum ArchiveScope {
    /// The `PerContext` case of `ArchiveScope`.
    PerContext,
    /// The `PerFamily` case of `ArchiveScope`.
    PerFamily,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `Fidelity` is an enum for the fidelity contract in this module: Bounded quality-diversity proposal memory.
pub enum Fidelity {
    /// The `Low` case of `Fidelity`.
    Low,
    /// The `Final` case of `Fidelity`.
    Final,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `CandidateRole` is an enum for the candidate role contract in this module: Bounded quality-diversity proposal memory.
pub enum CandidateRole {
    /// The `General` case of `CandidateRole`.
    General,
    /// The `Specialist` case of `CandidateRole`.
    Specialist,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `ArchiveConfig` is a record for the archive config contract in this module: Bounded quality-diversity proposal memory.
pub struct ArchiveConfig {
    /// SHA-256 identity emitted by the frozen descriptor-space definition.
    pub descriptor_version: String,
    /// SHA-256 digest identifying the associated descriptor identity.
    pub descriptor_identity_sha256: String,
    /// Scope label describing the population covered by this result.
    pub scope: ArchiveScope,
    /// Upper bound on cells.
    pub max_cells: usize,
    /// Upper bound on bytes.
    pub max_bytes: usize,
    /// Upper bound on events.
    pub max_events: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `Candidate` is a record for the candidate contract in this module: Bounded quality-diversity proposal memory.
pub struct Candidate {
    /// SHA-256 digest identifying the associated evaluator identity.
    pub evaluator_identity_sha256: String,
    /// Fidelity recorded for this candidate.
    pub fidelity: Fidelity,
    /// Data or authority role assigned to this value.
    pub role: CandidateRole,
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated family.
    pub family_id: String,
    /// Identifier for the associated lineage.
    pub lineage_id: String,
    /// Exposure ancestry recorded for this candidate.
    pub exposure_ancestry: BTreeSet<String>,
    /// SHA-256 digest identifying the associated observation.
    pub observation_sha256: String,
    /// Quality recorded for this candidate.
    pub quality: f64,
    /// Collection of descriptor bins values carried by this candidate.
    pub descriptor_bins: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `Elite` is a record for the elite contract in this module: Bounded quality-diversity proposal memory.
pub struct Elite {
    /// Candidate recorded for this elite.
    pub candidate: Candidate,
    /// SHA-256 digest identifying the associated descriptor identity.
    pub descriptor_identity_sha256: String,
    /// Collection of bins values carried by this elite.
    pub bins: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `Decision` is an enum for the decision contract in this module: Bounded quality-diversity proposal memory.
pub enum Decision {
    /// The `Inserted` case of `Decision`.
    Inserted,
    /// The `Replaced` case of `Decision`.
    Replaced {
        /// SHA-256 digest identifying the associated old observation.
        old_observation_sha256: String,
    },
    /// The `RejectedLowerQuality` case of `Decision`.
    RejectedLowerQuality,
    /// The `RejectedCapacity` case of `Decision`.
    RejectedCapacity,
    /// The `RejectedByteLimit` case of `Decision`.
    RejectedByteLimit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `ArchiveEvent` is a record for the archive event contract in this module: Bounded quality-diversity proposal memory.
pub struct ArchiveEvent {
    /// SHA-256 digest identifying the associated observation.
    pub observation_sha256: String,
    /// Decision recorded for this archive event.
    pub decision: Decision,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `Archive` is a record for the archive contract in this module: Bounded quality-diversity proposal memory.
pub struct Archive {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated identity.
    pub identity_sha256: String,
    /// Config recorded for this archive.
    pub config: ArchiveConfig,
    cells: BTreeMap<String, Elite>,
    events: Vec<ArchiveEvent>,
}

impl Archive {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(config: ArchiveConfig) -> Result<Self> {
        validate_config(&config)?;
        let identity_sha256 = archive_identity(&config)?;
        let archive = Self {
            schema: ARCHIVE_SCHEMA.into(),
            identity_sha256,
            config,
            cells: BTreeMap::new(),
            events: Vec::new(),
        };
        ensure!(
            archive.encoded_bytes()? <= archive.config.max_bytes,
            "archive byte cap is below its header size"
        );
        Ok(archive)
    }

    /// Performs the from bytes operation and returns an error if it cannot be completed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let archive: Self = serde_json::from_slice(bytes).context("decode QD archive")?;
        archive.validate()?;
        ensure!(
            archive.to_bytes()?.as_slice() == bytes,
            "archive bytes are not canonical"
        );
        Ok(archive)
    }

    /// Performs the to bytes operation and returns an error if it cannot be completed.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("serialize QD archive")?;
        ensure!(
            bytes.len() <= self.config.max_bytes,
            "archive exceeds encoded byte cap"
        );
        Ok(bytes)
    }

    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        validate_config(&self.config)?;
        ensure!(self.schema == ARCHIVE_SCHEMA, "unknown archive schema");
        ensure!(
            self.identity_sha256 == archive_identity(&self.config)?,
            "archive identity does not match its configuration"
        );
        ensure!(
            self.cells.len() <= self.config.max_cells,
            "archive exceeds cell cap"
        );
        ensure!(
            self.events.len() <= self.config.max_events,
            "archive exceeds event cap"
        );
        for (key, elite) in &self.cells {
            validate_candidate(&elite.candidate)?;
            ensure!(
                elite.descriptor_identity_sha256 == self.config.descriptor_identity_sha256,
                "elite descriptor identity mismatch"
            );
            ensure!(
                elite.bins == elite.candidate.descriptor_bins,
                "elite bin mismatch"
            );
            ensure!(
                (2..=4).contains(&elite.bins.len()),
                "elite needs 2 to 4 descriptor bins"
            );
            ensure!(
                *key == cell_key(
                    &self.config,
                    &elite.candidate,
                    &elite.bins,
                    &elite.descriptor_identity_sha256
                )?,
                "elite cell identity mismatch"
            );
        }
        for event in &self.events {
            ensure!(
                is_hex_digest(&event.observation_sha256),
                "event observation digest is invalid"
            );
        }
        ensure!(
            self.encoded_bytes()? <= self.config.max_bytes,
            "archive exceeds encoded byte cap"
        );
        Ok(())
    }

    /// Performs the identity SHA-256 operation.
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    /// Performs the cells operation.
    pub fn cells(&self) -> &BTreeMap<String, Elite> {
        &self.cells
    }

    /// Performs the events operation.
    pub fn events(&self) -> &[ArchiveEvent] {
        &self.events
    }

    /// Performs the encoded bytes operation and returns an error if it cannot be completed.
    pub fn encoded_bytes(&self) -> Result<usize> {
        Ok(serde_json::to_vec(self)
            .context("serialize QD archive")?
            .len())
    }

    /// Insert or reject one observation. Equal-quality ties go to the lower observation digest.
    /// The evaluator, fidelity, role, exposure ancestry, scope and bins all participate in the
    /// cell key, so scores from unlike identities are never compared.
    pub fn insert(&mut self, candidate: Candidate) -> Result<ArchiveEvent> {
        self.validate()?;
        validate_candidate(&candidate)?;
        ensure!(
            (2..=4).contains(&candidate.descriptor_bins.len()),
            "candidate needs 2 to 4 descriptor bins"
        );
        let descriptor_identity_sha256 = self.config.descriptor_identity_sha256.clone();
        let bins = &candidate.descriptor_bins;
        let key = cell_key(&self.config, &candidate, bins, &descriptor_identity_sha256)?;
        let elite = Elite {
            candidate: candidate.clone(),
            descriptor_identity_sha256,
            bins: bins.clone(),
        };
        let decision = match self.cells.get(&key) {
            Some(old) if is_better(&candidate, &old.candidate) => Decision::Replaced {
                old_observation_sha256: old.candidate.observation_sha256.clone(),
            },
            Some(_) => Decision::RejectedLowerQuality,
            None if self.cells.len() >= self.config.max_cells => Decision::RejectedCapacity,
            None => Decision::Inserted,
        };
        let event = ArchiveEvent {
            observation_sha256: candidate.observation_sha256.clone(),
            decision,
        };

        if matches!(
            &event.decision,
            Decision::Inserted | Decision::Replaced { .. }
        ) {
            let mut trial = self.clone();
            trial.cells.insert(key, elite);
            trial.push_event(event.clone());
            if trial.encoded_bytes()? <= trial.config.max_bytes {
                *self = trial;
                return Ok(event);
            }
            let rejected = ArchiveEvent {
                observation_sha256: candidate.observation_sha256,
                decision: Decision::RejectedByteLimit,
            };
            self.push_event(rejected.clone());
            return Ok(rejected);
        }
        self.push_event(event.clone());
        Ok(event)
    }

    /// Return proposal elites for a specialist request. Exposure and policy checks run before
    /// ranking; the supplied policy is the adapter seam for I04 authenticated role authority.
    pub fn retrieve_specialists<P: RetrievalPolicy>(
        &self,
        request: &RetrievalRequest,
        policy: &P,
        limit: usize,
    ) -> Result<Vec<&Elite>> {
        self.validate()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut eligible: Vec<&Elite> = self
            .cells
            .values()
            .filter(|elite| {
                let candidate = &elite.candidate;
                candidate.role == CandidateRole::Specialist
                    && candidate.context_id != request.context_id
                    && candidate.family_id != request.family_id
                    && candidate.lineage_id != request.lineage_id
                    && !request
                        .forbidden_exposure_ids
                        .contains(&candidate.context_id)
                    && !request
                        .forbidden_exposure_ids
                        .contains(&candidate.family_id)
                    && !request
                        .forbidden_exposure_ids
                        .contains(&candidate.lineage_id)
                    && candidate
                        .exposure_ancestry
                        .is_disjoint(&request.forbidden_exposure_ids)
                    && policy.allows(request, candidate)
            })
            .collect();
        eligible.sort_by(|left, right| {
            right
                .candidate
                .quality
                .total_cmp(&left.candidate.quality)
                .then_with(|| {
                    left.candidate
                        .observation_sha256
                        .cmp(&right.candidate.observation_sha256)
                })
        });
        eligible.truncate(limit);
        Ok(eligible)
    }

    fn push_event(&mut self, event: ArchiveEvent) {
        self.events.push(event);
        while self.events.len() > self.config.max_events {
            self.events.remove(0);
        }
        while self
            .encoded_bytes()
            .is_ok_and(|n| n > self.config.max_bytes)
            && !self.events.is_empty()
        {
            self.events.remove(0);
        }
    }
}

fn validate_config(config: &ArchiveConfig) -> Result<()> {
    ensure!(
        !config.descriptor_version.trim().is_empty(),
        "descriptor version is empty"
    );
    ensure!(
        is_hex_digest(&config.descriptor_identity_sha256),
        "descriptor identity must be SHA-256"
    );
    ensure!(config.max_cells > 0, "archive max_cells must be positive");
    ensure!(config.max_bytes > 0, "archive max_bytes must be positive");
    ensure!(config.max_events > 0, "archive max_events must be positive");
    Ok(())
}

fn archive_identity(config: &ArchiveConfig) -> Result<String> {
    digest(&(
        ARCHIVE_SCHEMA,
        &config.descriptor_version,
        &config.descriptor_identity_sha256,
        config.scope,
        config.max_cells,
        config.max_bytes,
        config.max_events,
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// `RetrievalRequest` is a record for the retrieval request contract in this module: Bounded quality-diversity proposal memory.
pub struct RetrievalRequest {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated family.
    pub family_id: String,
    /// Identifier for the associated lineage.
    pub lineage_id: String,
    /// Forbidden exposure ids recorded for this retrieval request.
    pub forbidden_exposure_ids: BTreeSet<String>,
}

/// I04 adapter must bind this decision to the authenticated requester and its current capabilities.
pub trait RetrievalPolicy {
    /// Performs the allows operation.
    fn allows(&self, request: &RetrievalRequest, candidate: &Candidate) -> bool;
}

fn validate_candidate(candidate: &Candidate) -> Result<()> {
    ensure!(
        is_hex_digest(&candidate.evaluator_identity_sha256),
        "evaluator identity must be SHA-256"
    );
    ensure!(
        !candidate.context_id.trim().is_empty(),
        "context identity is empty"
    );
    ensure!(
        !candidate.family_id.trim().is_empty(),
        "family identity is empty"
    );
    ensure!(
        !candidate.lineage_id.trim().is_empty(),
        "lineage identity is empty"
    );
    ensure!(candidate.quality.is_finite(), "quality must be finite");
    ensure!(
        is_hex_digest(&candidate.observation_sha256),
        "observation digest must be SHA-256"
    );
    ensure!(
        candidate
            .exposure_ancestry
            .iter()
            .all(|id| !id.trim().is_empty()),
        "exposure ancestry contains an empty identity"
    );
    Ok(())
}

fn is_better(candidate: &Candidate, current: &Candidate) -> bool {
    candidate.quality > current.quality
        || (candidate.quality == current.quality
            && candidate.observation_sha256 < current.observation_sha256)
}

fn cell_key(
    config: &ArchiveConfig,
    candidate: &Candidate,
    bins: &[u32],
    descriptor_identity_sha256: &str,
) -> Result<String> {
    let context_scope = match config.scope {
        ArchiveScope::PerContext => Some(candidate.context_id.as_str()),
        ArchiveScope::PerFamily => None,
    };
    digest(&(
        descriptor_identity_sha256,
        context_scope,
        &candidate.family_id,
        &candidate.evaluator_identity_sha256,
        candidate.fidelity,
        candidate.role,
        &candidate.exposure_ancestry,
        bins,
    ))
}

fn digest<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("serialize archive identity")?;
    Ok(hex_digest(Sha256::digest(bytes).as_slice()))
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
