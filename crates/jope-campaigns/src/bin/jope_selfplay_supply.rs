//! Proactive Rust-only supply supervisor for immutable self-play micro-waves.
//!
//! This process never submits, holds, releases, or cancels SCQ jobs. It keeps a
//! bounded reserve of current-checkpoint manifests ahead of the queue keeper by
//! invoking only the Rust `jope_selfplay_wave` planner/sanitizer/builder.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::dime_runtime_contracts::{
    append_infrastructure_attempt, digest_label, load_or_rebuild_recovery_projection,
    matching_drain_intent, save_recovery_projection, terminal_classification,
    DimeInfrastructureAttemptV1, DimeNamedDigestV1, DimeRecoveryProjectionV1, DimeRoleV1,
    DimeScientificPayloadV1, DimeTerminalClassV1, DimeTerminalEvidenceV1,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, checkpoint_epoch, default_root_str, default_scq_str, now_ms, sha256_bytes,
    sha256_file,
};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

#[path = "common/flock.rs"]
mod flock;

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-supply",
    about = "Maintain a bounded reserve of immutable current-checkpoint micro-waves"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    /// Build waves. Without this flag the supervisor is read-only.
    #[arg(long)]
    build: bool,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 5)]
    interval_seconds: u64,
    #[arg(long, default_value_t = 2_048)]
    queue_floor: usize,
    #[arg(long, default_value_t = 2_048)]
    queued_target: usize,
    #[arg(long, default_value_t = 4_096)]
    queued_hard_cap: usize,
    /// Maintain this many current-checkpoint manifests beyond scheduler demand.
    #[arg(long, default_value_t = 2_048)]
    ready_reserve_min: usize,
    /// Minimum queued/starting/dispatching plus checkpoint-valid ready work.
    #[arg(long, default_value_t = MIN_USEFUL_WORK)]
    useful_work_min: usize,
    /// Canonical runway floor measured in candidate evaluations, not jobs.
    #[arg(long, default_value_t = MIN_CANDIDATE_EVALUATION_WORK)]
    candidate_work_min: usize,
    /// Protected ready reserve measured in candidate evaluations.
    #[arg(long, default_value_t = MIN_CANDIDATE_RESERVE)]
    candidate_reserve_min: usize,
    /// Newest complete waves protected from scheduler admission.
    #[arg(long, default_value_t = MIN_COMPLETE_RESERVE_WAVES)]
    min_complete_reserve_waves: usize,
    /// Maximum current-checkpoint waves with unclaimed manifests.
    #[arg(long, default_value_t = 8)]
    max_overlap_waves: usize,
    /// Exactly one wave is built before waiting for keeper acknowledgment.
    #[arg(long, default_value_t = 1)]
    max_builds_per_cycle: usize,
    /// Production default is 50. A matched 25-vs-50 experiment remains gated.
    #[arg(long, default_value_t = 50)]
    max_candidates_per_manifest: usize,
    #[arg(long, default_value_t = 30)]
    queue_status_max_age_seconds: u64,
    #[arg(long, default_value_t = 300)]
    performance_max_age_seconds: u64,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::release_binary("jope_selfplay_wave")
    )]
    wave_builder: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::research_script("test_exact_context_proposals.py")
    )]
    proposal_generator: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::research_script("test_exact_dl_generate.py")
    )]
    proposal_legacy_generator: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::research_script("test_exact_common.py")
    )]
    proposal_common: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::runtime_bundle_pytorch()
    )]
    proposal_pythonpath: PathBuf,
    #[arg(long)]
    catalog_pointer: Option<PathBuf>,
    #[arg(long)]
    checkpoint_pointer: Option<PathBuf>,
    #[arg(long)]
    proposals_pointer: Option<PathBuf>,
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long)]
    sealed_cohort: Option<PathBuf>,
    #[arg(long)]
    robust_seed_bank: Option<PathBuf>,
    #[arg(long)]
    pair_registry: Option<PathBuf>,
    #[arg(long)]
    queue_status: Option<PathBuf>,
}

impl Args {
    fn catalog_pointer(&self) -> PathBuf {
        match &self.catalog_pointer {
            Some(path) => path.clone(),
            None => self.root.join("catalog/current.json"),
        }
    }

    fn proposals_pointer(&self) -> PathBuf {
        match &self.proposals_pointer {
            Some(path) => path.clone(),
            None => self.root.join("proposals/current.json"),
        }
    }

    fn corpus(&self) -> PathBuf {
        match &self.corpus {
            Some(path) => path.clone(),
            None => self.root.join("corpus/strict-test-exact.sqlite"),
        }
    }

    fn sealed_cohort(&self) -> PathBuf {
        match &self.sealed_cohort {
            Some(path) => path.clone(),
            None => self.root.join("catalog/private/sealed-current-v1.json"),
        }
    }

    fn robust_seed_bank(&self) -> PathBuf {
        match &self.robust_seed_bank {
            Some(path) => path.clone(),
            None => self.root.join("corpus/robust-gp-seed-bank.v1.json"),
        }
    }

    fn pair_registry(&self) -> PathBuf {
        match &self.pair_registry {
            Some(path) => path.clone(),
            None => self.root.join("ops/pair-registry.sqlite"),
        }
    }

    fn queue_status(&self) -> PathBuf {
        match &self.queue_status {
            Some(path) => path.clone(),
            None => self.root.join("ops/queue-keeper-v3.json"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct QueueStatus {
    schema: String,
    generated_at_ms: u128,
    admission_allowed: bool,
    #[serde(default)]
    degraded: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    fail_closed_reason: Option<String>,
    active_checkpoint_sha256: Option<String>,
    checkpoint_epoch: Option<String>,
    queued_target: i64,
    queued_hard_cap: i64,
    queued: usize,
    starting: usize,
    queued_starting: usize,
    dispatching: usize,
    admission_waves: Vec<KeeperAdmissionWave>,
    protected_reserve_waves: Vec<KeeperAdmissionWave>,
    complete_current_checkpoint_waves: usize,
    partial_current_checkpoint_waves: usize,
    complete_reserve_waves: usize,
    reserve_manifests: usize,
    admission_manifests: usize,
    validated_ready: usize,
    useful_work: usize,
    candidate_evaluation_work: u64,
    unclaimed_current_checkpoint_manifests: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct KeeperAdmissionWave {
    wave: String,
    summary_sha256: String,
    manifest_count: usize,
    unclaimed_manifests: usize,
    candidate_evaluation_count: u64,
    complete: bool,
}

#[derive(Debug)]
struct CheckpointInputs {
    checkpoint_sha256: String,
    checkpoint_pointer: PathBuf,
    checkpoint_artifact: PathBuf,
    proposals_pointer: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
struct ReadyWave {
    wave: String,
    summary_sha256: String,
    manifests: usize,
    unclaimed_manifests: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SupplyDecision {
    below_floor: bool,
    overloaded: bool,
    candidate_evaluation_work: usize,
    candidate_evaluation_work_target: usize,
    estimated_manifests_per_wave: usize,
    target_feasible: bool,
    builds_planned: usize,
}

#[derive(Clone, Copy, Debug)]
struct PerformanceMeasurements {
    generated_at_ms: u128,
    p95_jobs_per_minute: f64,
    p95_candidate_evaluations_per_minute: f64,
    proposer_to_ready_minutes: f64,
    candidate_evaluation_work_target: usize,
}

#[derive(Debug, Serialize)]
struct BuiltWave {
    wave: String,
    cohort_sha256: String,
    proposal_pointer_sha256: String,
    summary_sha256: String,
    manifests: usize,
}

#[derive(Debug, Serialize)]
struct CycleReport {
    schema: &'static str,
    generated_at_ms: u128,
    mode: &'static str,
    active_checkpoint_sha256: String,
    checkpoint_epoch: String,
    queue_floor: usize,
    queued_target: usize,
    queued_hard_cap: usize,
    ready_reserve_min: usize,
    useful_work_min: usize,
    useful_work_target: usize,
    useful_work_target_formula: &'static str,
    p95_jobs_per_minute: Option<f64>,
    p95_candidate_evaluations_per_minute: Option<f64>,
    proposer_to_ready_minutes: Option<f64>,
    performance_generated_at_ms: Option<u128>,
    performance_fresh: bool,
    keeper_queued_target: i64,
    keeper_queued_hard_cap: i64,
    queued_starting: usize,
    queued: usize,
    starting: usize,
    dispatching: usize,
    queue_depth: usize,
    below_floor: bool,
    overloaded: bool,
    current_ready_waves: Vec<ReadyWave>,
    ready_manifests: usize,
    validated_ready: usize,
    reserve_manifests: usize,
    admission_manifests: usize,
    complete_current_checkpoint_waves: usize,
    partial_current_checkpoint_waves: usize,
    complete_reserve_waves: usize,
    minimum_complete_reserve_waves: usize,
    protected_reserve_waves: Vec<KeeperAdmissionWave>,
    useful_work: usize,
    candidate_work_min: usize,
    candidate_reserve_min: usize,
    candidate_evaluation_work: usize,
    candidate_evaluation_work_target: usize,
    protected_candidate_evaluations: usize,
    target_feasible: bool,
    critical_infeasibility_reason: Option<String>,
    projected_jobs_before_build: usize,
    projected_jobs_after_build: usize,
    max_overlap_waves: usize,
    builds_planned: usize,
    builds_completed: Vec<BuiltWave>,
    awaiting_keeper_ack: Option<String>,
    awaiting_keeper_ack_since_ms: Option<u128>,
    keeper_status_generated_at_ms: u128,
    admission_allowed: bool,
    hold_reason: Option<String>,
}

struct PendingAck {
    wave: String,
    since_ms: u128,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PersistedSupplyStage {
    FreezeCohort,
    SubmitCohortProposer,
    FetchValidateProposals,
    CacheFilterResample,
    BuildWave,
    KeeperAcknowledgment,
}

impl PersistedSupplyStage {
    fn ordinal(self) -> u8 {
        match self {
            Self::FreezeCohort => 0,
            Self::SubmitCohortProposer => 1,
            Self::FetchValidateProposals => 2,
            Self::CacheFilterResample => 3,
            Self::BuildWave => 4,
            Self::KeeperAcknowledgment => 5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedSupplyState {
    schema: String,
    wave: String,
    checkpoint_sha256: String,
    cohort: PathBuf,
    cohort_sha256: String,
    catalog_sha256: String,
    decoder_configuration_sha256: String,
    seed: u64,
    #[serde(default = "default_proposal_reservoir")]
    proposal_reservoir_per_context: usize,
    stage: PersistedSupplyStage,
    revision: u64,
    recovery_accounting_only: bool,
    #[serde(default)]
    proposal_attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proposal_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_proposal_pointer: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_proposal_pointer_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sanitized_proposal_pointer: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sanitized_proposal_pointer_sha256: Option<String>,
    #[serde(default)]
    runtime_pinned_attempts: Vec<RuntimePinnedIntentV1>,
}

fn default_proposal_reservoir() -> usize {
    64
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedProposerInput {
    name: String,
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimePinnedIntentV1 {
    schema: String,
    runtime_bundle: PathBuf,
    runtime_bundle_sha256: String,
    catalog_snapshot: PathBuf,
    catalog_snapshot_sha256: String,
    proposer_inputs: Vec<PinnedProposerInput>,
    immutable_spec_path: PathBuf,
    immutable_spec_sha256: String,
    job_identity: String,
    attempt: u32,
}

impl RuntimePinnedIntentV1 {
    fn validate(&self) -> Result<()> {
        if self.schema != RUNTIME_PINNED_INTENT_SCHEMA {
            bail!("runtime-pinned proposer intent schema is not v1");
        }
        validate_sha256(&self.runtime_bundle_sha256)?;
        validate_sha256(&self.catalog_snapshot_sha256)?;
        validate_sha256(&self.immutable_spec_sha256)?;
        if self.job_identity.trim().is_empty() {
            bail!("runtime-pinned proposer job identity is empty");
        }
        if !self.runtime_bundle.is_dir()
            || self
                .runtime_bundle
                .file_name()
                .and_then(|value| value.to_str())
                != Some(self.runtime_bundle_sha256.as_str())
        {
            bail!("runtime-pinned proposer bundle identity drift");
        }
        let bundle: Value =
            serde_json::from_slice(&fs::read(self.runtime_bundle.join("bundle.json"))?)?;
        if bundle.get("bundle_sha256").and_then(Value::as_str)
            != Some(self.runtime_bundle_sha256.as_str())
        {
            bail!("runtime-pinned proposer bundle manifest drift");
        }
        if !self.immutable_spec_path.is_file()
            || sha256_file(&self.immutable_spec_path)? != self.immutable_spec_sha256
        {
            bail!("runtime-pinned proposer immutable spec drift");
        }
        if self.proposer_inputs.is_empty() {
            bail!("runtime-pinned proposer has no immutable inputs");
        }
        let mut names = BTreeSet::new();
        for input in &self.proposer_inputs {
            validate_sha256(&input.sha256)?;
            if input.name.trim().is_empty()
                || !names.insert(input.name.as_str())
                || !input.path.is_file()
                || sha256_file(&input.path)? != input.sha256
            {
                bail!("runtime-pinned proposer input identity drift");
            }
        }
        if !names.contains("generator")
            || !names.contains("common")
            || !names.contains("pointer")
            || !names.contains("catalog")
            || !names.contains("data")
            || !names.contains("checkpoint")
            || !names.contains("cohort")
        {
            bail!("runtime-pinned proposer input set is incomplete");
        }
        let pointer = self
            .proposer_inputs
            .iter()
            .find(|input| input.name == "pointer")
            .context("runtime-pinned proposer has no catalog snapshot input")?;
        if pointer.path != self.catalog_snapshot || pointer.sha256 != self.catalog_snapshot_sha256 {
            bail!("runtime-pinned proposer catalog snapshot drift");
        }
        Ok(())
    }
}

impl PersistedSupplyState {
    fn validate(&self, expected_wave: &str, expected_checkpoint: &str) -> Result<()> {
        if !matches!(
            self.schema.as_str(),
            SUPPLY_STATE_V1_SCHEMA | SUPPLY_STATE_V2_SCHEMA
        ) || self.wave != expected_wave
            || self.checkpoint_sha256 != expected_checkpoint
        {
            bail!("persisted cohort supply identity drift");
        }
        validate_sha256(&self.cohort_sha256)?;
        validate_sha256(&self.catalog_sha256)?;
        validate_sha256(&self.decoder_configuration_sha256)?;
        if !matches!(self.proposal_reservoir_per_context, 64 | 128 | 256) {
            bail!("cohort proposal reservoir must be 64, 128, or 256");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal()
            && self.proposal_job_id.is_none()
        {
            bail!("submitted cohort supply state lacks a proposer job identity");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::FetchValidateProposals.ordinal()
            && (self.cohort_proposal_pointer.is_none()
                || self.cohort_proposal_pointer_sha256.is_none())
        {
            bail!("fetched cohort supply state lacks an immutable proposal pointer");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::CacheFilterResample.ordinal()
            && (self.sanitized_proposal_pointer.is_none()
                || self.sanitized_proposal_pointer_sha256.is_none())
        {
            bail!("cache-filtered supply state lacks its proposal pointer");
        }
        for (index, intent) in self.runtime_pinned_attempts.iter().enumerate() {
            intent.validate()?;
            if intent.attempt as usize != index {
                bail!("runtime-pinned proposer attempts are not contiguous");
            }
        }
        if self.schema == SUPPLY_STATE_V2_SCHEMA
            && self.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal()
        {
            let intent = self
                .runtime_pinned_attempts
                .last()
                .context("submitted v2 supply state lacks a runtime-pinned intent")?;
            if Some(intent.job_identity.as_str()) != self.proposal_job_id.as_deref()
                || intent.attempt != self.proposal_attempt
            {
                bail!("submitted v2 supply state proposer attempt identity drift");
            }
        }
        Ok(())
    }

    fn advance(&mut self, next: PersistedSupplyStage) -> Result<()> {
        if next.ordinal() != self.stage.ordinal() + 1 {
            bail!("cohort supply stage must advance exactly once");
        }
        self.stage = next;
        self.revision += 1;
        self.validate(&self.wave, &self.checkpoint_sha256)
    }

    fn retry_proposer(
        &mut self,
        root: &Path,
        terminal_state: &str,
        observed_at_ms: u128,
    ) -> Result<ProposerRetryDisposition> {
        if self.stage != PersistedSupplyStage::SubmitCohortProposer {
            bail!("cohort proposer retry is allowed only in its submitted stage");
        }
        let prior = self
            .runtime_pinned_attempts
            .last()
            .cloned()
            .context("cohort proposer retry lacks its pinned original spec")?;
        prior.validate()?;
        let scientific_payload = proposer_scientific_payload(self, &prior)?;
        let mut projection = proposer_recovery_projection(root, self, &scientific_payload)?;
        let replayed_receipt =
            self.proposal_job_id.as_deref() != Some(projection.current_job_identity.as_str());
        reconcile_proposer_projection(root, self, &projection)?;
        if replayed_receipt {
            self.revision = self.revision.saturating_add(1);
            self.validate(&self.wave, &self.checkpoint_sha256)?;
            return Ok(ProposerRetryDisposition::Retried);
        }
        if projection.scientific_failures >= 5 {
            return Ok(ProposerRetryDisposition::ScientificCircuitOpen);
        }
        if projection.retry_pending_submission {
            return Ok(ProposerRetryDisposition::Backoff);
        }
        if projection.submitted_at_ms > 0 && observed_at_ms < projection.retry_not_before_ms {
            return Ok(ProposerRetryDisposition::Backoff);
        }
        if let Some(drain) = matching_drain_intent(
            root,
            DimeRoleV1::CohortProposer,
            &scientific_payload,
            observed_at_ms,
        )? {
            append_supply_event(
                root,
                "intentional_drain_honored",
                &json!({
                    "job_id": self.proposal_job_id,
                    "attempt": self.proposal_attempt,
                    "drain_receipt_sha256": drain.content_sha256,
                    "scientific_payload_sha256": scientific_payload.identity_sha256()?,
                }),
            )?;
            return Ok(ProposerRetryDisposition::Drained);
        }
        let current = projection.current_job_identity.clone();
        ensure!(
            self.proposal_job_id.as_deref() == Some(current.as_str()),
            "primary proposer state is not reconciled with its recovery projection"
        );
        let scientific = if terminal_state == "failed" {
            scientific_failure_receipt(root, &current, &scientific_payload)?
        } else {
            None
        };
        let terminal_class = terminal_classification(terminal_state, scientific.as_deref())?;
        let current_attempt_number = projection.current_attempt_number;
        let base = current
            .rsplit_once("-infra")
            .filter(|(_, suffix)| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            .map(|(base, _)| base)
            .or_else(|| {
                current
                    .rsplit_once("-retry")
                    .filter(|(_, suffix)| suffix.bytes().all(|byte| byte.is_ascii_digit()))
                    .map(|(base, _)| base)
            })
            .unwrap_or(&current)
            .to_owned();
        let next_attempt_number = current_attempt_number
            .checked_add(1)
            .context("cohort proposer infrastructure attempt overflow")?;
        let next_job_identity = format!("{base}-infra{next_attempt_number:06}");
        let scientific_failures = projection.scientific_failures.saturating_add(u32::from(
            terminal_class == DimeTerminalClassV1::ScientificFailure,
        ));
        let retry_allowed = scientific_failures < 5;
        let evidence_sha256 = scientific.unwrap_or_else(|| {
            sha256_bytes(
                format!(
                    "dime-proposer-terminal-v1\0{}\0{}\0{}",
                    current, terminal_state, prior.immutable_spec_sha256
                )
                .as_bytes(),
            )
        });
        let recent_cancellations = projection
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| observed_at_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
            .count()
            + usize::from(terminal_state == "cancelled");
        let backoff_ms = if recent_cancellations >= 3 {
            15 * 60 * 1_000
        } else {
            jope_dime_regressor::dime_runtime_contracts::retry_backoff_ms(current_attempt_number)
        };
        let retry_not_before_ms = observed_at_ms.saturating_add(backoff_ms);
        let receipt = DimeInfrastructureAttemptV1::new_terminal(
            DimeRoleV1::CohortProposer,
            scientific_payload,
            current_attempt_number,
            current.clone(),
            retry_allowed.then_some(next_job_identity.clone()),
            projection.last_attempt_receipt_sha256.clone(),
            prior
                .immutable_spec_path
                .metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(1, |duration| duration.as_millis()),
            retry_not_before_ms,
            DimeTerminalEvidenceV1 {
                terminal_state: terminal_state.to_owned(),
                class: terminal_class,
                evidence_sha256,
                observed_at_ms,
                foreign_cancellation: terminal_state == "cancelled",
            },
        )?;
        append_infrastructure_attempt(root, &receipt)?;
        projection.apply_receipt(&receipt)?;
        let contention_alarm_open = projection.refresh_contention_alarm(observed_at_ms)?;
        save_recovery_projection(root, &projection)?;
        if contention_alarm_open {
            append_supply_event(
                root,
                "foreign_cancellation_contention_alarm",
                &json!({
                    "severity": "prominent",
                    "job_id": current,
                    "scientific_payload_sha256": projection.scientific_payload_sha256,
                    "retry_mode": "capped_half_open",
                    "retry_not_before_ms": projection.retry_not_before_ms,
                }),
            )?;
        }
        if !retry_allowed {
            append_supply_event(
                root,
                "scientific_failure_circuit_open",
                &json!({
                    "job_id": current,
                    "scientific_failures": projection.scientific_failures,
                    "attempt_receipt_sha256": receipt.content_sha256,
                }),
            )?;
            return Ok(ProposerRetryDisposition::ScientificCircuitOpen);
        }
        reconcile_proposer_projection(root, self, &projection)?;
        self.revision += 1;
        self.validate(&self.wave, &self.checkpoint_sha256)?;
        append_supply_event(
            root,
            if terminal_state == "cancelled" {
                "foreign_cancellation_recovered"
            } else {
                "infrastructure_attempt_recovered"
            },
            &json!({
                "prior_job_id": current,
                "next_job_id": self.proposal_job_id,
                "infrastructure_attempt": projection.current_attempt_number,
                "scientific_failures": projection.scientific_failures,
                "foreign_cancellations": projection.foreign_cancellation_timestamps_ms.len(),
                "attempt_receipt_sha256": receipt.content_sha256,
                "retry_not_before_ms": retry_not_before_ms,
                "contention_alarm_open": contention_alarm_open,
            }),
        )?;
        Ok(ProposerRetryDisposition::Retried)
    }
}

fn reconcile_proposer_projection(
    root: &Path,
    state: &mut PersistedSupplyState,
    projection: &DimeRecoveryProjectionV1,
) -> Result<()> {
    let desired_attempt = u32::try_from(projection.current_attempt_number.saturating_sub(1))
        .context("proposer recovery attempt overflows primary schema")?;
    if state.proposal_attempt == desired_attempt
        && state.proposal_job_id.as_deref() == Some(projection.current_job_identity.as_str())
    {
        return Ok(());
    }
    ensure!(
        desired_attempt == state.proposal_attempt.saturating_add(1),
        "proposer recovery projection is not the next rollback-compatible attempt"
    );
    let prior = state
        .runtime_pinned_attempts
        .last()
        .cloned()
        .context("proposer recovery lacks a pinned prior intent")?;
    let job_identity = projection.current_job_identity.clone();
    let prior_body = fs::read_to_string(&prior.immutable_spec_path)?;
    let replacement = format!("id = {}", toml_quote(&job_identity)?);
    let mut replaced = false;
    let body = prior_body
        .lines()
        .map(|line| {
            if !replaced && line.starts_with("id = ") {
                replaced = true;
                replacement.as_str()
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !replaced {
        bail!("pinned cohort proposer spec lacks a job identity");
    }
    let directory = root.join("specs/cohort-propose");
    fs::create_dir_all(&directory)?;
    let spec = directory.join(format!("{job_identity}.toml"));
    if spec.is_file() {
        if fs::read_to_string(&spec)? != body {
            bail!("immutable cohort proposer retry spec drift");
        }
    } else {
        fs::write(&spec, &body)?;
    }
    let pinned = runtime_pinned_intent_from_spec(&spec, desired_attempt)?;
    if pinned.runtime_bundle_sha256 != prior.runtime_bundle_sha256
        || pinned.catalog_snapshot_sha256 != prior.catalog_snapshot_sha256
        || pinned.proposer_inputs != prior.proposer_inputs
    {
        bail!("cohort proposer retry changed its pinned bundle or inputs");
    }
    state.runtime_pinned_attempts.push(pinned);
    state.proposal_attempt = desired_attempt;
    state.proposal_job_id = Some(job_identity);
    Ok(())
}

fn proposer_recovery_projection(
    root: &Path,
    state: &PersistedSupplyState,
    payload: &DimeScientificPayloadV1,
) -> Result<DimeRecoveryProjectionV1> {
    let initial_job = state
        .runtime_pinned_attempts
        .first()
        .map(|intent| intent.job_identity.as_str())
        .context("proposer recovery has no initial pinned identity")?;
    load_or_rebuild_recovery_projection(root, DimeRoleV1::CohortProposer, payload, initial_job)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProposerRetryDisposition {
    Retried,
    Backoff,
    Drained,
    ScientificCircuitOpen,
}

fn proposer_scientific_payload(
    state: &PersistedSupplyState,
    pinned: &RuntimePinnedIntentV1,
) -> Result<DimeScientificPayloadV1> {
    pinned.validate()?;
    let parent_intent_sha256 = sha256_bytes(&serde_json::to_vec(&json!({
        "domain": "dime-cohort-proposer-parent-intent-v1",
        "wave": state.wave,
        "checkpoint_sha256": state.checkpoint_sha256,
        "cohort_sha256": state.cohort_sha256,
        "catalog_sha256": state.catalog_sha256,
        "decoder_configuration_sha256": state.decoder_configuration_sha256,
        "seed": state.seed,
        "proposal_reservoir_per_context": state.proposal_reservoir_per_context,
    }))?);
    let binary_sha256 = pinned
        .proposer_inputs
        .iter()
        .find(|input| input.name == "generator")
        .map(|input| input.sha256.clone())
        .context("pinned proposer has no generator digest")?;
    let mut inputs = pinned
        .proposer_inputs
        .iter()
        .map(|input| DimeNamedDigestV1 {
            name: input.name.clone(),
            sha256: input.sha256.clone(),
        })
        .collect::<Vec<_>>();
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    let payload = DimeScientificPayloadV1 {
        parent_intent_sha256,
        runtime_bundle_sha256: pinned.runtime_bundle_sha256.clone(),
        binary_sha256,
        configuration_sha256: state.decoder_configuration_sha256.clone(),
        train_data_sha256: state.cohort_sha256.clone(),
        protocol_sha256: digest_label("jope-cohort-proposer-protocol-v1"),
        metric_sha256: digest_label("jope-proposal-validity-metric-v1"),
        seed: state.seed,
        inputs,
    };
    payload.validate()?;
    Ok(payload)
}

fn scientific_failure_receipt(
    root: &Path,
    job_identity: &str,
    payload: &DimeScientificPayloadV1,
) -> Result<Option<String>> {
    let path = root
        .join("ops/dime-scientific-failures")
        .join(format!("{job_identity}.json"));
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let evidence_sha256 = value
        .get("evidence_sha256")
        .and_then(Value::as_str)
        .context("scientific failure receipt lacks evidence")?;
    validate_sha256(evidence_sha256)?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("DimeScientificFailure.v1")
            && value.get("job_identity").and_then(Value::as_str) == Some(job_identity)
            && value
                .get("scientific_payload_sha256")
                .and_then(Value::as_str)
                == Some(payload.identity_sha256()?.as_str())
            && value.get("deterministic").and_then(Value::as_bool) == Some(true),
        "scientific failure receipt is not hash-bound to this proposer intent"
    );
    Ok(Some(evidence_sha256.to_owned()))
}

fn append_supply_event(root: &Path, kind: &str, detail: &Value) -> Result<()> {
    append_event(
        &root.join("ops/micro-wave-supply-events.v1.jsonl"),
        &json!({
            "schema": "JopeMicroWaveSupplyEvent.v1",
            "generated_at_ms": now_ms(),
            "kind": kind,
            "detail": detail,
        }),
    )
}

mod constants;
use constants::*;
mod lock;
use lock::*;
mod proposer_context;
mod proposer_input;
mod runtime;
mod supply;
mod wave_builder;
use proposer_context::*;
use proposer_input::*;
use runtime::*;
use supply::*;
use wave_builder::*;

#[cfg(test)]
mod tests;

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let _lock = SingleInstanceLock::acquire(&args.root.join("ops/micro-wave-supply-builder.lock"))?;
    let circuit_path = args.root.join("ops/cohort-supply-circuit-breaker.v1.json");
    let (mut previous_error, mut identical_failures) =
        load_supply_failure_streak(&circuit_path).unwrap_or_default();
    loop {
        let mut retry_after_seconds = args.interval_seconds;
        match cycle(&args) {
            Ok(report) => {
                let value = serde_json::to_value(&report)?;
                atomic_json(&args.root.join("ops/micro-wave-supply-v2.json"), &value)?;
                previous_error = None;
                identical_failures = 0;
                atomic_json(
                    &circuit_path,
                    &json!({
                        "schema": "SupplyCircuitBreaker.v1",
                        "generated_at_ms": now_ms(),
                        "open": false,
                        "identical_failures": 0,
                        "critical": false,
                        "retry_after_seconds": args.interval_seconds,
                        "recoverable_state": args.root.join("ops/cohort-supply-state-v2"),
                    }),
                )?;
                println!("{}", serde_json::to_string(&report)?);
            }
            Err(error) => {
                let error_text = format!("{error:#}");
                if previous_error.as_deref() == Some(error_text.as_str()) {
                    identical_failures = identical_failures.saturating_add(1);
                } else {
                    previous_error = Some(error_text.clone());
                    identical_failures = 1;
                }
                let exponent = identical_failures.saturating_sub(1).min(31);
                retry_after_seconds = args
                    .interval_seconds
                    .saturating_mul(1_u64 << exponent)
                    .min(ERROR_BACKOFF_MAX_SECONDS);
                let failure = json!({
                    "schema": "JopeMicroWaveSupplyError.v1",
                    "generated_at_ms": now_ms(),
                    "mode": if args.build { "build" } else { "audit" },
                    "error": error_text,
                    "builds_completed": 0,
                    "identical_failures": identical_failures,
                    "critical": identical_failures >= 3,
                    "retry_after_seconds": retry_after_seconds,
                });
                atomic_json(&args.root.join("ops/micro-wave-supply-v2.json"), &failure)?;
                atomic_json(
                    &circuit_path,
                    &json!({
                        "schema": "SupplyCircuitBreaker.v1",
                        "generated_at_ms": now_ms(),
                        "open": true,
                        "error": previous_error,
                        "identical_failures": identical_failures,
                        "critical": identical_failures >= 3,
                        "retry_after_seconds": retry_after_seconds,
                        "recoverable_state": args.root.join("ops/cohort-supply-state-v2"),
                    }),
                )?;
                eprintln!("{}", serde_json::to_string(&failure)?);
                if args.once {
                    return Err(error);
                }
            }
        }
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(retry_after_seconds));
    }
}
