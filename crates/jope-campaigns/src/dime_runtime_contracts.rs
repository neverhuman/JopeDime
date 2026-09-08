//! DIME-owned recovery and campaign status contracts.
//!
//! Infrastructure attempts are immutable terminal receipts.  A retry appends
//! one receipt for the attempt that just ended, then launches a fresh job
//! identity carrying the same `scientific_payload`.  Intentional cancellation
//! is a separate, short-lived, hash-bound authority; scheduler state alone is
//! never cancellation authority.

use crate::selfplay_runtime::{atomic_json, sha256_bytes};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const DIME_INFRASTRUCTURE_ATTEMPT_SCHEMA: &str = "DimeInfrastructureAttempt.v1";
pub const DIME_DRAIN_INTENT_SCHEMA: &str = "DimeDrainIntent.v1";
pub const DIME_RECOVERY_PROJECTION_SCHEMA: &str = "DimeRecoveryProjection.v1";
pub const JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA: &str = "JopeDlgpBudgetCampaignStatus.v1";

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
    Ok(())
}

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DimeRoleV1 {
    CohortProposer,
    AlternatingTrainer,
    BreadthEncoder,
    BudgetCampaign,
}

impl DimeRoleV1 {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CohortProposer => "cohort_proposer",
            Self::AlternatingTrainer => "alternating_trainer",
            Self::BreadthEncoder => "breadth_encoder",
            Self::BudgetCampaign => "budget_campaign",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeNamedDigestV1 {
    pub name: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeScientificPayloadV1 {
    pub parent_intent_sha256: String,
    pub runtime_bundle_sha256: String,
    pub binary_sha256: String,
    pub configuration_sha256: String,
    pub train_data_sha256: String,
    pub protocol_sha256: String,
    pub metric_sha256: String,
    pub seed: u64,
    pub inputs: Vec<DimeNamedDigestV1>,
}

impl DimeScientificPayloadV1 {
    pub fn validate(&self) -> Result<()> {
        for (field, digest) in [
            ("parent intent", &self.parent_intent_sha256),
            ("runtime bundle", &self.runtime_bundle_sha256),
            ("binary", &self.binary_sha256),
            ("configuration", &self.configuration_sha256),
            ("train data", &self.train_data_sha256),
            ("protocol", &self.protocol_sha256),
            ("metric", &self.metric_sha256),
        ] {
            require_sha256(digest, field)?;
        }
        ensure!(!self.inputs.is_empty(), "scientific payload has no inputs");
        let mut prior: Option<&str> = None;
        for input in &self.inputs {
            ensure!(
                !input.name.trim().is_empty(),
                "scientific input name is empty"
            );
            require_sha256(&input.sha256, "scientific input")?;
            if let Some(previous) = prior {
                ensure!(
                    previous < input.name.as_str(),
                    "scientific inputs are not sorted and unique"
                );
            }
            prior = Some(&input.name);
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DimeTerminalClassV1 {
    InfrastructureFailure,
    ScientificFailure,
    IntentionalDrain,
    Completed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeTerminalEvidenceV1 {
    pub terminal_state: String,
    pub class: DimeTerminalClassV1,
    pub evidence_sha256: String,
    pub observed_at_ms: u128,
    pub foreign_cancellation: bool,
}

impl DimeTerminalEvidenceV1 {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.terminal_state.trim().is_empty() && self.observed_at_ms > 0,
            "terminal evidence is incomplete"
        );
        require_sha256(&self.evidence_sha256, "terminal evidence")?;
        ensure!(
            !self.foreign_cancellation
                || (self.class == DimeTerminalClassV1::InfrastructureFailure
                    && self.terminal_state == "cancelled"),
            "foreign cancellation classification is inconsistent"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeInfrastructureAttemptV1 {
    pub schema: String,
    pub role: DimeRoleV1,
    pub scientific_payload: DimeScientificPayloadV1,
    pub scientific_payload_sha256: String,
    pub attempt_number: u64,
    pub job_identity: String,
    pub next_job_identity: Option<String>,
    pub prior_attempt_receipt_sha256: Option<String>,
    pub created_at_ms: u128,
    pub retry_not_before_ms: u128,
    pub terminal: DimeTerminalEvidenceV1,
    pub content_sha256: String,
}

impl DimeInfrastructureAttemptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new_terminal(
        role: DimeRoleV1,
        scientific_payload: DimeScientificPayloadV1,
        attempt_number: u64,
        job_identity: String,
        next_job_identity: Option<String>,
        prior_attempt_receipt_sha256: Option<String>,
        created_at_ms: u128,
        retry_not_before_ms: u128,
        terminal: DimeTerminalEvidenceV1,
    ) -> Result<Self> {
        let scientific_payload_sha256 = scientific_payload.identity_sha256()?;
        let mut value = Self {
            schema: DIME_INFRASTRUCTURE_ATTEMPT_SCHEMA.to_owned(),
            role,
            scientific_payload,
            scientific_payload_sha256,
            attempt_number,
            job_identity,
            next_job_identity,
            prior_attempt_receipt_sha256,
            created_at_ms,
            retry_not_before_ms,
            terminal,
            content_sha256: String::new(),
        };
        value.content_sha256 = value.computed_content_sha256()?;
        value.validate()?;
        Ok(value)
    }

    fn computed_content_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        canonical_sha256(&copy)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DIME_INFRASTRUCTURE_ATTEMPT_SCHEMA,
            "infrastructure attempt schema is not v1"
        );
        self.scientific_payload.validate()?;
        ensure!(
            self.scientific_payload.identity_sha256()? == self.scientific_payload_sha256,
            "infrastructure attempt scientific payload hash drift"
        );
        ensure!(
            self.attempt_number > 0
                && !self.job_identity.trim().is_empty()
                && self.created_at_ms > 0
                && self.retry_not_before_ms >= self.created_at_ms,
            "infrastructure attempt identity or time is invalid"
        );
        if let Some(next) = &self.next_job_identity {
            ensure!(
                !next.trim().is_empty() && next != &self.job_identity,
                "recovery did not allocate a fresh job identity"
            );
        }
        if let Some(prior) = &self.prior_attempt_receipt_sha256 {
            require_sha256(prior, "prior attempt receipt")?;
        }
        self.terminal.validate()?;
        require_sha256(&self.content_sha256, "attempt content")?;
        ensure!(
            self.computed_content_sha256()? == self.content_sha256,
            "infrastructure attempt content hash drift"
        );
        Ok(())
    }
}

fn attempt_directory(root: &Path, receipt: &DimeInfrastructureAttemptV1) -> PathBuf {
    root.join("ops/dime-infrastructure-attempts")
        .join(receipt.role.as_str())
        .join(&receipt.scientific_payload_sha256)
}

/// Atomically append one terminal attempt receipt. Replaying an identical
/// receipt is idempotent; reusing an attempt number with different bytes or
/// skipping a number is rejected.
pub fn append_infrastructure_attempt(
    root: &Path,
    receipt: &DimeInfrastructureAttemptV1,
) -> Result<PathBuf> {
    receipt.validate()?;
    let directory = attempt_directory(root, receipt);
    fs::create_dir_all(&directory)?;
    let prior =
        read_attempt_directory(&directory, receipt.role, &receipt.scientific_payload_sha256)?;
    if let Some((existing, path)) = prior
        .iter()
        .find(|(value, _)| value.attempt_number == receipt.attempt_number)
    {
        ensure!(
            existing == receipt,
            "attempt replay differs from its receipt"
        );
        return Ok(path.clone());
    }
    let expected = prior
        .last()
        .map_or(1, |(value, _)| value.attempt_number + 1);
    ensure!(
        receipt.attempt_number == expected,
        "infrastructure attempt number is not monotonic"
    );
    let expected_prior = prior.last().map(|(value, _)| value.content_sha256.as_str());
    ensure!(
        receipt.prior_attempt_receipt_sha256.as_deref() == expected_prior,
        "infrastructure attempt prior-receipt chain drift"
    );
    let path = directory.join(format!(
        "{:08}-{}.json",
        receipt.attempt_number, receipt.content_sha256
    ));
    atomic_json(&path, &serde_json::to_value(receipt)?)?;
    Ok(path)
}

fn read_attempt_directory(
    directory: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Vec<(DimeInfrastructureAttemptV1, PathBuf)>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(directory)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    let mut values = Vec::with_capacity(paths.len());
    for path in paths {
        let value: DimeInfrastructureAttemptV1 = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read attempt receipt {}", path.display()))?,
        )
        .with_context(|| format!("decode attempt receipt {}", path.display()))?;
        value.validate()?;
        ensure!(
            value.role == role && value.scientific_payload_sha256 == scientific_payload_sha256,
            "attempt directory contains a foreign receipt"
        );
        ensure!(
            path.file_name().and_then(|name| name.to_str())
                == Some(
                    format!("{:08}-{}.json", value.attempt_number, value.content_sha256).as_str()
                ),
            "attempt receipt filename does not bind its content"
        );
        values.push((value, path));
    }
    values.sort_by_key(|(value, _)| value.attempt_number);
    let mut prior = None;
    for (index, (value, _)) in values.iter().enumerate() {
        ensure!(
            value.attempt_number == index as u64 + 1,
            "attempt ledger is not contiguous"
        );
        ensure!(
            value.prior_attempt_receipt_sha256.as_deref() == prior,
            "attempt ledger prior-receipt chain drift"
        );
        prior = Some(value.content_sha256.as_str());
    }
    Ok(values)
}

pub fn latest_attempt_receipt_sha256(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Option<String>> {
    require_sha256(scientific_payload_sha256, "scientific payload")?;
    Ok(
        infrastructure_attempts(root, role, scientific_payload_sha256)?
            .last()
            .map(|value| value.content_sha256.clone()),
    )
}

pub fn latest_infrastructure_attempt(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Option<DimeInfrastructureAttemptV1>> {
    Ok(infrastructure_attempts(root, role, scientific_payload_sha256)?.pop())
}

pub fn infrastructure_attempts(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Vec<DimeInfrastructureAttemptV1>> {
    require_sha256(scientific_payload_sha256, "scientific payload")?;
    let directory = root
        .join("ops/dime-infrastructure-attempts")
        .join(role.as_str())
        .join(scientific_payload_sha256);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    Ok(
        read_attempt_directory(&directory, role, scientific_payload_sha256)?
            .into_iter()
            .map(|(value, _)| value)
            .collect(),
    )
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeRecoveryProjectionV1 {
    pub schema: String,
    pub role: DimeRoleV1,
    pub scientific_payload: DimeScientificPayloadV1,
    pub scientific_payload_sha256: String,
    pub current_job_identity: String,
    pub current_attempt_number: u64,
    pub last_attempt_receipt_sha256: Option<String>,
    pub retry_not_before_ms: u128,
    pub retry_pending_submission: bool,
    pub submitted_at_ms: u128,
    pub scientific_failures: u32,
    pub foreign_cancellation_timestamps_ms: Vec<u128>,
    pub contention_alarm_open: bool,
    pub content_sha256: String,
}

impl DimeRecoveryProjectionV1 {
    pub fn initial(
        role: DimeRoleV1,
        scientific_payload: DimeScientificPayloadV1,
        current_job_identity: String,
    ) -> Result<Self> {
        let scientific_payload_sha256 = scientific_payload.identity_sha256()?;
        ensure!(
            !current_job_identity.trim().is_empty(),
            "recovery projection job identity is empty"
        );
        let mut value = Self {
            schema: DIME_RECOVERY_PROJECTION_SCHEMA.to_owned(),
            role,
            scientific_payload,
            scientific_payload_sha256,
            current_job_identity,
            current_attempt_number: 1,
            last_attempt_receipt_sha256: None,
            retry_not_before_ms: 0,
            retry_pending_submission: false,
            submitted_at_ms: 0,
            scientific_failures: 0,
            foreign_cancellation_timestamps_ms: Vec::new(),
            contention_alarm_open: false,
            content_sha256: String::new(),
        };
        value.refresh_content_sha256()?;
        Ok(value)
    }

    fn computed_content_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        canonical_sha256(&copy)
    }

    fn refresh_content_sha256(&mut self) -> Result<()> {
        self.content_sha256 = self.computed_content_sha256()?;
        self.validate()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DIME_RECOVERY_PROJECTION_SCHEMA,
            "recovery projection schema is not v1"
        );
        self.scientific_payload.validate()?;
        ensure!(
            self.scientific_payload.identity_sha256()? == self.scientific_payload_sha256,
            "recovery projection payload binding drift"
        );
        ensure!(
            !self.current_job_identity.trim().is_empty() && self.current_attempt_number > 0,
            "recovery projection job/attempt identity is invalid"
        );
        if let Some(receipt) = &self.last_attempt_receipt_sha256 {
            require_sha256(receipt, "recovery projection attempt receipt")?;
        }
        ensure!(
            self.foreign_cancellation_timestamps_ms
                .windows(2)
                .all(|pair| pair[0] <= pair[1]),
            "recovery projection cancellation timestamps are unsorted"
        );
        require_sha256(&self.content_sha256, "recovery projection content")?;
        ensure!(
            self.computed_content_sha256()? == self.content_sha256,
            "recovery projection content hash drift"
        );
        Ok(())
    }

    pub fn apply_receipt(&mut self, receipt: &DimeInfrastructureAttemptV1) -> Result<()> {
        receipt.validate()?;
        ensure!(
            receipt.role == self.role
                && receipt.scientific_payload_sha256 == self.scientific_payload_sha256
                && receipt.scientific_payload == self.scientific_payload,
            "attempt receipt does not belong to the recovery projection"
        );
        if self.last_attempt_receipt_sha256.as_deref() == Some(&receipt.content_sha256) {
            return self.validate();
        }
        ensure!(
            receipt.attempt_number == self.current_attempt_number
                && receipt.job_identity == self.current_job_identity
                && receipt.prior_attempt_receipt_sha256 == self.last_attempt_receipt_sha256,
            "attempt receipt cannot advance the recovery projection"
        );
        if receipt.terminal.class == DimeTerminalClassV1::ScientificFailure {
            self.scientific_failures = self.scientific_failures.saturating_add(1);
        }
        if receipt.terminal.foreign_cancellation {
            self.foreign_cancellation_timestamps_ms
                .push(receipt.terminal.observed_at_ms);
        }
        self.contention_alarm_open = self
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| {
                receipt.terminal.observed_at_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000
            })
            .count()
            >= 3;
        self.last_attempt_receipt_sha256 = Some(receipt.content_sha256.clone());
        self.retry_not_before_ms = receipt.retry_not_before_ms;
        self.retry_pending_submission = receipt.next_job_identity.is_some();
        self.submitted_at_ms = 0;
        if let Some(next_job_identity) = &receipt.next_job_identity {
            self.current_job_identity = next_job_identity.clone();
            self.current_attempt_number = receipt.attempt_number.saturating_add(1);
        }
        self.refresh_content_sha256()
    }

    pub fn mark_submitted(&mut self, job_identity: &str, now_ms: u128) -> Result<()> {
        ensure!(
            self.retry_pending_submission
                && self.current_job_identity == job_identity
                && now_ms >= self.retry_not_before_ms,
            "recovery projection submission is not due or has the wrong identity"
        );
        self.retry_pending_submission = false;
        self.submitted_at_ms = now_ms;
        self.retry_not_before_ms = now_ms.saturating_add(self.next_backoff_ms(now_ms));
        self.refresh_content_sha256()
    }

    pub fn record_initial_submission(&mut self, job_identity: &str, now_ms: u128) -> Result<()> {
        ensure!(
            self.last_attempt_receipt_sha256.is_none()
                && self.current_attempt_number == 1
                && self.current_job_identity == job_identity
                && !self.retry_pending_submission
                && self.submitted_at_ms == 0,
            "initial recovery projection submission is duplicated or mismatched"
        );
        self.submitted_at_ms = now_ms;
        self.retry_not_before_ms = now_ms.saturating_add(retry_backoff_ms(1));
        self.refresh_content_sha256()
    }

    pub fn next_backoff_ms(&self, now_ms: u128) -> u128 {
        if self
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| now_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
            .count()
            >= 3
        {
            15 * 60 * 1_000
        } else {
            retry_backoff_ms(self.current_attempt_number)
        }
    }

    pub fn refresh_contention_alarm(&mut self, now_ms: u128) -> Result<bool> {
        self.contention_alarm_open = self
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| now_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
            .count()
            >= 3;
        self.refresh_content_sha256()?;
        Ok(self.contention_alarm_open)
    }
}

pub fn recovery_projection_path(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<PathBuf> {
    require_sha256(scientific_payload_sha256, "recovery projection payload")?;
    Ok(root
        .join("ops/dime-recovery-projections")
        .join(role.as_str())
        .join(format!("{scientific_payload_sha256}.v1.json")))
}

pub fn rebuild_recovery_projection(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload: &DimeScientificPayloadV1,
    initial_job_identity: &str,
) -> Result<DimeRecoveryProjectionV1> {
    let mut projection = DimeRecoveryProjectionV1::initial(
        role,
        scientific_payload.clone(),
        initial_job_identity.to_owned(),
    )?;
    for receipt in infrastructure_attempts(root, role, &scientific_payload.identity_sha256()?)? {
        projection.apply_receipt(&receipt)?;
    }
    Ok(projection)
}

pub fn load_or_rebuild_recovery_projection(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload: &DimeScientificPayloadV1,
    initial_job_identity: &str,
) -> Result<DimeRecoveryProjectionV1> {
    let rebuilt =
        rebuild_recovery_projection(root, role, scientific_payload, initial_job_identity)?;
    let path = recovery_projection_path(root, role, &rebuilt.scientific_payload_sha256)?;
    if path.is_file() {
        let stored: DimeRecoveryProjectionV1 = serde_json::from_slice(&fs::read(&path)?)?;
        stored.validate()?;
        // The ledger is authoritative. A stale projection is repaired from it;
        // it is never used to manufacture another terminal receipt.
        if stored.role == rebuilt.role
            && stored.scientific_payload == rebuilt.scientific_payload
            && stored.scientific_payload_sha256 == rebuilt.scientific_payload_sha256
            && stored.current_job_identity == rebuilt.current_job_identity
            && stored.current_attempt_number == rebuilt.current_attempt_number
            && stored.last_attempt_receipt_sha256 == rebuilt.last_attempt_receipt_sha256
            && stored.scientific_failures == rebuilt.scientific_failures
            && stored.foreign_cancellation_timestamps_ms
                == rebuilt.foreign_cancellation_timestamps_ms
        {
            return Ok(stored);
        }
    }
    save_recovery_projection(root, &rebuilt)?;
    Ok(rebuilt)
}

pub fn save_recovery_projection(
    root: &Path,
    projection: &DimeRecoveryProjectionV1,
) -> Result<PathBuf> {
    projection.validate()?;
    let path =
        recovery_projection_path(root, projection.role, &projection.scientific_payload_sha256)?;
    atomic_json(&path, &serde_json::to_value(projection)?)?;
    Ok(path)
}

/// Cancellation never consumes a scientific failure artifact. Only the
/// scheduler's exact `failed` state may be paired with deterministic evidence.
pub fn terminal_classification(
    scheduler_state: &str,
    deterministic_scientific_evidence_sha256: Option<&str>,
) -> Result<DimeTerminalClassV1> {
    if scheduler_state == "failed" {
        if let Some(evidence) = deterministic_scientific_evidence_sha256 {
            require_sha256(evidence, "deterministic scientific failure evidence")?;
            return Ok(DimeTerminalClassV1::ScientificFailure);
        }
    }
    Ok(DimeTerminalClassV1::InfrastructureFailure)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeDrainIntentV1 {
    pub schema: String,
    pub role: DimeRoleV1,
    pub scientific_payload_sha256: String,
    pub parent_intent_sha256: String,
    pub issued_at_ms: u128,
    pub expires_at_ms: u128,
    pub issuer: String,
    pub reason: String,
    pub content_sha256: String,
}

impl DimeDrainIntentV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        role: DimeRoleV1,
        scientific_payload_sha256: String,
        parent_intent_sha256: String,
        issued_at_ms: u128,
        expires_at_ms: u128,
        issuer: String,
        reason: String,
    ) -> Result<Self> {
        let mut value = Self {
            schema: DIME_DRAIN_INTENT_SCHEMA.to_owned(),
            role,
            scientific_payload_sha256,
            parent_intent_sha256,
            issued_at_ms,
            expires_at_ms,
            issuer,
            reason,
            content_sha256: String::new(),
        };
        value.content_sha256 = value.computed_content_sha256()?;
        value.validate_at(issued_at_ms)?;
        Ok(value)
    }

    fn computed_content_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        canonical_sha256(&copy)
    }

    pub fn validate_at(&self, now_ms: u128) -> Result<()> {
        ensure!(
            self.schema == DIME_DRAIN_INTENT_SCHEMA,
            "drain schema is not v1"
        );
        require_sha256(&self.scientific_payload_sha256, "drain scientific payload")?;
        require_sha256(&self.parent_intent_sha256, "drain parent intent")?;
        ensure!(
            self.issued_at_ms > 0
                && self.issued_at_ms <= now_ms
                && now_ms < self.expires_at_ms
                && self.expires_at_ms.saturating_sub(self.issued_at_ms) <= 24 * 60 * 60 * 1_000,
            "drain receipt is future-dated, expired, or exceeds 24 hours"
        );
        ensure!(
            !self.issuer.trim().is_empty() && !self.reason.trim().is_empty(),
            "drain issuer or reason is empty"
        );
        require_sha256(&self.content_sha256, "drain content")?;
        ensure!(
            self.computed_content_sha256()? == self.content_sha256,
            "drain content hash drift"
        );
        Ok(())
    }
}

pub fn write_drain_intent(root: &Path, intent: &DimeDrainIntentV1) -> Result<PathBuf> {
    intent.validate_at(intent.issued_at_ms)?;
    let path = root
        .join("ops/dime-drain-intents")
        .join(format!("{}.json", intent.content_sha256));
    if path.is_file() {
        let existing: DimeDrainIntentV1 = serde_json::from_slice(&fs::read(&path)?)?;
        ensure!(existing == *intent, "drain replay differs from its receipt");
        return Ok(path);
    }
    atomic_json(&path, &serde_json::to_value(intent)?)?;
    Ok(path)
}

pub fn matching_drain_intent(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload: &DimeScientificPayloadV1,
    now_ms: u128,
) -> Result<Option<DimeDrainIntentV1>> {
    let directory = root.join("ops/dime-drain-intents");
    if !directory.is_dir() {
        return Ok(None);
    }
    let payload_sha256 = scientific_payload.identity_sha256()?;
    let mut matches = Vec::new();
    for entry in fs::read_dir(directory)?.filter_map(|entry| entry.ok()) {
        let Ok(value) = serde_json::from_slice::<DimeDrainIntentV1>(&fs::read(entry.path())?)
        else {
            continue;
        };
        if value.validate_at(now_ms).is_ok()
            && value.role == role
            && value.scientific_payload_sha256 == payload_sha256
            && value.parent_intent_sha256 == scientific_payload.parent_intent_sha256
        {
            matches.push(value);
        }
    }
    matches.sort_by_key(|value| value.issued_at_ms);
    Ok(matches.pop())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DlgpCampaignGateStateV1 {
    ReproductionRequired,
    LaunchGatesBlocked,
    Runnable,
    Complete,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DlgpActiveJobV1 {
    pub job_identity: String,
    pub trajectory_identity: String,
    pub attempt_number: u64,
    pub state: String,
    pub progress_evaluations: u64,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeDlgpBudgetCampaignStatusV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub last_advanced_at_ms: u128,
    pub protocol_sha256: String,
    pub stage: String,
    pub gate_state: DlgpCampaignGateStateV1,
    pub trajectory_count: u64,
    pub completed_trajectory_count: u64,
    pub rung_counts: BTreeMap<String, u64>,
    pub evaluations_used: u64,
    pub evaluations_cap: u64,
    pub active_jobs: Vec<DlgpActiveJobV1>,
    pub robust_wins: u64,
    pub robust_wins_target: u64,
    pub robust_wins_denominator: u64,
    pub seed_agreement: String,
    pub license_track: String,
    pub production_compatible_state: String,
    pub research_only_state: String,
    pub test_opening_state: String,
    pub promotion_decisions: Vec<String>,
    pub blockers: Vec<String>,
}

impl JopeDlgpBudgetCampaignStatusV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA,
            "budget campaign status schema is not v1"
        );
        require_sha256(&self.protocol_sha256, "campaign protocol")?;
        ensure!(
            self.generated_at_ms > 0
                && self.last_advanced_at_ms <= self.generated_at_ms
                && self.completed_trajectory_count <= self.trajectory_count
                && self.evaluations_used <= self.evaluations_cap
                && self.evaluations_cap == 200_000
                && self.robust_wins <= self.robust_wins_denominator
                && self.robust_wins_target == 7
                && self.robust_wins_denominator == 22,
            "budget campaign status accounting drift"
        );
        ensure!(
            matches!(
                self.license_track.as_str(),
                "production_compatible" | "research_only" | "separate"
            ),
            "budget campaign license track is invalid"
        );
        let identities = self
            .active_jobs
            .iter()
            .map(|job| job.job_identity.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            identities.len() == self.active_jobs.len()
                && self.active_jobs.iter().all(|job| {
                    !job.job_identity.trim().is_empty()
                        && !job.trajectory_identity.trim().is_empty()
                        && job.attempt_number > 0
                }),
            "budget campaign active job identity is invalid or duplicated"
        );
        Ok(())
    }

    pub fn to_compact_json(&self) -> Result<Value> {
        self.validate()?;
        Ok(serde_json::json!({
            "schema": self.schema,
            "generated_at_ms": self.generated_at_ms,
            "last_advanced_at_ms": self.last_advanced_at_ms,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "gate_state": self.gate_state,
            "trajectories": [self.completed_trajectory_count, self.trajectory_count],
            "rungs": self.rung_counts,
            "evaluations": [self.evaluations_used, self.evaluations_cap],
            "active_jobs": self.active_jobs,
            "robust_wins": [self.robust_wins, self.robust_wins_target, self.robust_wins_denominator],
            "seed_agreement": self.seed_agreement,
            "license_track": self.license_track,
            "test_opening_state": self.test_opening_state,
            "blockers": self.blockers,
        }))
    }
}

pub fn retry_backoff_ms(attempt_number: u64) -> u128 {
    match attempt_number {
        0 | 1 => 30_000,
        2 => 60_000,
        3 => 120_000,
        4 => 240_000,
        _ => 15 * 60 * 1_000,
    }
}

pub fn digest_label(label: &str) -> String {
    sha256_bytes(label.as_bytes())
}

pub fn read_validated_attempt(path: &Path) -> Result<DimeInfrastructureAttemptV1> {
    let value: DimeInfrastructureAttemptV1 = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    value.validate()?;
    Ok(value)
}

pub fn ensure_same_scientific_payload(
    expected: &DimeScientificPayloadV1,
    observed: &DimeScientificPayloadV1,
) -> Result<()> {
    expected.validate()?;
    observed.validate()?;
    if expected != observed {
        bail!("infrastructure retry changed its scientific payload");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(seed: u64) -> DimeScientificPayloadV1 {
        DimeScientificPayloadV1 {
            parent_intent_sha256: digest_label("parent"),
            runtime_bundle_sha256: digest_label("runtime"),
            binary_sha256: digest_label("binary"),
            configuration_sha256: digest_label("config"),
            train_data_sha256: digest_label("train"),
            protocol_sha256: digest_label("protocol"),
            metric_sha256: digest_label("metric"),
            seed,
            inputs: vec![DimeNamedDigestV1 {
                name: "frozen".to_owned(),
                sha256: digest_label("frozen"),
            }],
        }
    }

    fn receipt(attempt_number: u64, prior: Option<String>) -> DimeInfrastructureAttemptV1 {
        DimeInfrastructureAttemptV1::new_terminal(
            DimeRoleV1::CohortProposer,
            payload(101),
            attempt_number,
            format!("proposer-attempt-{attempt_number}"),
            Some(format!("proposer-attempt-{}", attempt_number + 1)),
            prior,
            attempt_number as u128,
            attempt_number as u128 + retry_backoff_ms(attempt_number),
            DimeTerminalEvidenceV1 {
                terminal_state: "cancelled".to_owned(),
                class: DimeTerminalClassV1::InfrastructureFailure,
                evidence_sha256: digest_label(&format!("terminal-{attempt_number}")),
                observed_at_ms: attempt_number as u128,
                foreign_cancellation: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn retry_five_can_append_attempt_six_without_changing_science() {
        let directory = tempfile::tempdir().unwrap();
        let mut prior = None;
        for attempt in 1..=6 {
            let value = receipt(attempt, prior.clone());
            append_infrastructure_attempt(directory.path(), &value).unwrap();
            prior = Some(value.content_sha256.clone());
        }
        let last = latest_attempt_receipt_sha256(
            directory.path(),
            DimeRoleV1::CohortProposer,
            &payload(101).identity_sha256().unwrap(),
        )
        .unwrap();
        assert_eq!(last, prior);
    }

    #[test]
    fn append_is_idempotent_but_number_reuse_and_science_drift_fail() {
        let directory = tempfile::tempdir().unwrap();
        let one = receipt(1, None);
        let path = append_infrastructure_attempt(directory.path(), &one).unwrap();
        assert_eq!(
            append_infrastructure_attempt(directory.path(), &one).unwrap(),
            path
        );
        let mut drift = one.clone();
        drift.terminal.evidence_sha256 = digest_label("different-terminal");
        drift.content_sha256 = drift.computed_content_sha256().unwrap();
        assert!(append_infrastructure_attempt(directory.path(), &drift).is_err());
        let mut changed_science = payload(101);
        changed_science.seed = 202;
        assert!(ensure_same_scientific_payload(&payload(101), &changed_science).is_err());
    }

    #[test]
    fn only_an_unexpired_exact_drain_suppresses_recovery() {
        let directory = tempfile::tempdir().unwrap();
        let payload = payload(101);
        let intent = DimeDrainIntentV1::new(
            DimeRoleV1::CohortProposer,
            payload.identity_sha256().unwrap(),
            payload.parent_intent_sha256.clone(),
            1_000,
            2_000,
            "dime-supervisor".to_owned(),
            "planned maintenance".to_owned(),
        )
        .unwrap();
        write_drain_intent(directory.path(), &intent).unwrap();
        assert!(matching_drain_intent(
            directory.path(),
            DimeRoleV1::CohortProposer,
            &payload,
            1_500
        )
        .unwrap()
        .is_some());
        assert!(matching_drain_intent(
            directory.path(),
            DimeRoleV1::CohortProposer,
            &payload,
            2_000
        )
        .unwrap()
        .is_none());
        assert!(matching_drain_intent(
            directory.path(),
            DimeRoleV1::AlternatingTrainer,
            &payload,
            1_500
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn infrastructure_backoff_is_exact_and_then_caps_at_fifteen_minutes() {
        assert_eq!(
            (1..=7).map(retry_backoff_ms).collect::<Vec<_>>(),
            vec![30_000, 60_000, 120_000, 240_000, 900_000, 900_000, 900_000]
        );
    }

    #[test]
    fn cancellation_ignores_even_a_stray_scientific_artifact() {
        assert_eq!(
            terminal_classification("cancelled", Some(&digest_label("stray"))).unwrap(),
            DimeTerminalClassV1::InfrastructureFailure
        );
        assert_eq!(
            terminal_classification("failed", Some(&digest_label("bound"))).unwrap(),
            DimeTerminalClassV1::ScientificFailure
        );
    }

    #[test]
    fn receipt_append_crash_rebuilds_the_recorded_next_identity_without_duplication() {
        let directory = tempfile::tempdir().unwrap();
        let payload = payload(101);
        let receipt = receipt(1, None);
        append_infrastructure_attempt(directory.path(), &receipt).unwrap();

        let rebuilt = load_or_rebuild_recovery_projection(
            directory.path(),
            DimeRoleV1::CohortProposer,
            &payload,
            "proposer-attempt-1",
        )
        .unwrap();
        assert_eq!(rebuilt.current_job_identity, "proposer-attempt-2");
        assert_eq!(rebuilt.current_attempt_number, 2);
        assert_eq!(rebuilt.retry_not_before_ms, receipt.retry_not_before_ms);
        assert_eq!(
            rebuilt.last_attempt_receipt_sha256,
            Some(receipt.content_sha256.clone())
        );

        let replayed = load_or_rebuild_recovery_projection(
            directory.path(),
            DimeRoleV1::CohortProposer,
            &payload,
            "proposer-attempt-1",
        )
        .unwrap();
        assert_eq!(replayed, rebuilt);
        assert_eq!(
            infrastructure_attempts(
                directory.path(),
                DimeRoleV1::CohortProposer,
                &payload.identity_sha256().unwrap(),
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn three_foreign_cancellations_open_contention_and_force_half_open_cap() {
        let mut projection = DimeRecoveryProjectionV1::initial(
            DimeRoleV1::CohortProposer,
            payload(101),
            "proposer-attempt-1".to_owned(),
        )
        .unwrap();
        let mut prior = None;
        for attempt in 1..=3 {
            let observed = attempt as u128 * 60_000;
            let value = DimeInfrastructureAttemptV1::new_terminal(
                DimeRoleV1::CohortProposer,
                payload(101),
                attempt,
                format!("proposer-attempt-{attempt}"),
                Some(format!("proposer-attempt-{}", attempt + 1)),
                prior.clone(),
                observed,
                observed + retry_backoff_ms(attempt),
                DimeTerminalEvidenceV1 {
                    terminal_state: "cancelled".to_owned(),
                    class: DimeTerminalClassV1::InfrastructureFailure,
                    evidence_sha256: digest_label(&format!("cancelled-{attempt}")),
                    observed_at_ms: observed,
                    foreign_cancellation: true,
                },
            )
            .unwrap();
            projection.apply_receipt(&value).unwrap();
            prior = Some(value.content_sha256);
        }
        assert!(projection.contention_alarm_open);
        assert_eq!(projection.next_backoff_ms(180_000), 15 * 60 * 1_000);
    }
}
