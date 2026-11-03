//! Persistent, fail-closed supervisor for the budget-first DLGP campaign.
//!
//! The supervisor never invents campaign work. It reconciles only hash-bound
//! trajectory plans, counts partial receipts against the 200k ceiling, keeps
//! license tracks separate, quarantines foreign strategy-screen jobs, and
//! submits a frozen spec only after Stage 0 and every launch gate pass.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::dime_runtime_contracts::{
    append_infrastructure_attempt, infrastructure_attempts, matching_drain_intent,
    retry_backoff_ms, DimeInfrastructureAttemptV1, DimeNamedDigestV1, DimeRoleV1,
    DimeScientificPayloadV1, DimeTerminalClassV1, DimeTerminalEvidenceV1, DlgpActiveJobV1,
    DlgpCampaignGateStateV1, JopeDlgpBudgetCampaignStatusV1,
    JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA,
};
use jope_dime_regressor::jope_dlgp_best5::{
    FROZEN_CHECKPOINT_SHA256, STAGE1_MAX_SEARCH_EVALUATIONS,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, scq_json, sha256_bytes,
    sha256_file,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

#[path = "jope_dlgp_budget_supervisor/campaign_cycle.rs"]
mod campaign_cycle;
#[path = "jope_dlgp_budget_supervisor/scheduler.rs"]
mod scheduler;
#[cfg(test)]
#[path = "jope_dlgp_budget_supervisor/tests.rs"]
mod tests;

use campaign_cycle::cycle;
use scheduler::{
    launch_receipt_path, load_plans, load_progress, materialize_attempt_spec,
    next_attempt_identity, read_launch_receipt, scheduler_jobs, submit_attempt, terminal_class,
};

jope_paths::once_str!(
    fn default_protocol_str => paths::campaign_manifest("dlgp-budget-first/protocol.v1.json")
);
const PROJECT: &str = "jope-dlgp-budget-first";

#[derive(Parser, Debug)]
#[command(
    name = "jope-dlgp-budget-supervisor",
    about = "Reconcile and resume only frozen budget-first DLGP trajectories"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_protocol_str())]
    protocol: PathBuf,
    #[arg(long)]
    campaign_root: Option<PathBuf>,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long)]
    submit: bool,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 30)]
    interval_seconds: u64,
}

impl Args {
    fn campaign_root(&self) -> PathBuf {
        self.campaign_root
            .clone()
            .unwrap_or_else(|| self.root.join("campaigns/dlgp-budget-first"))
    }

    fn status_path(&self) -> PathBuf {
        self.root.join("ops/dlgp-budget-campaign-status.v1.json")
    }

    fn compact_path(&self) -> PathBuf {
        self.root
            .join("ops/dlgp-budget-campaign-status.compact.v1.json")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Stage0Receipt {
    schema: String,
    protocol_sha256: String,
    round2_payload_sha256: String,
    control_prefix_count: u64,
    test_csv_read: bool,
    reproduced: bool,
    generated_at_ms: u128,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchGates {
    schema: String,
    protocol_sha256: String,
    proposal_checkpoint_sha256: String,
    fresh_evaluator_evidence: bool,
    fresh_supply_evidence: bool,
    reviewed_acknowledgement: bool,
    real_scq: bool,
    observed_project_weight: u64,
    expected_project_weight: u64,
    disk_free_fraction_micros: u64,
    generated_at_ms: u128,
}

impl LaunchGates {
    fn blockers(&self, protocol_sha256: &str, now: u128) -> Vec<String> {
        let mut blockers = Vec::new();
        if self.schema != "JopeDlgpBudgetLaunchGates.v1" || self.protocol_sha256 != protocol_sha256
        {
            blockers.push("launch gate receipt is off protocol".to_owned());
        }
        if self.proposal_checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256 {
            blockers.push("exact proposal checkpoint gate failed".to_owned());
        }
        if !self.fresh_evaluator_evidence {
            blockers.push("evaluator evidence is absent or stale".to_owned());
        }
        if !self.fresh_supply_evidence {
            blockers.push("supply evidence is absent or stale".to_owned());
        }
        if !self.reviewed_acknowledgement {
            blockers.push("reviewed acknowledgement is absent".to_owned());
        }
        if !self.real_scq {
            blockers.push("real scq executable gate failed".to_owned());
        }
        if self.expected_project_weight == 0
            || self.observed_project_weight != self.expected_project_weight
        {
            blockers.push("project priority weight drift".to_owned());
        }
        if self.disk_free_fraction_micros < 20_000 {
            blockers.push("worker disk is below the exact 2% launch floor".to_owned());
        }
        if self.generated_at_ms == 0 || now.saturating_sub(self.generated_at_ms) > 2 * 60 * 1_000 {
            blockers.push("launch gate receipt is stale".to_owned());
        }
        blockers
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryPlan {
    schema: String,
    protocol_sha256: String,
    trajectory_identity: String,
    dataset_sha256: String,
    seed: u64,
    license_track: String,
    runtime_bundle_sha256: String,
    binary_sha256: String,
    configuration_sha256: String,
    train_data_sha256: String,
    metric_sha256: String,
    job_identity: String,
    spec_path: PathBuf,
    spec_sha256: String,
}

impl TrajectoryPlan {
    fn validate(&self, protocol_sha256: &str) -> Result<()> {
        ensure!(
            self.schema == "JopeDlgpBudgetTrajectoryPlan.v1"
                && self.protocol_sha256 == protocol_sha256
                && matches!(self.seed, 101 | 202)
                && matches!(
                    self.license_track.as_str(),
                    "production_compatible" | "research_only"
                )
                && !self.trajectory_identity.trim().is_empty()
                && !self.job_identity.trim().is_empty(),
            "trajectory plan identity drift"
        );
        for digest in [
            &self.dataset_sha256,
            &self.runtime_bundle_sha256,
            &self.binary_sha256,
            &self.configuration_sha256,
            &self.train_data_sha256,
            &self.metric_sha256,
            &self.spec_sha256,
        ] {
            ensure!(
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "trajectory plan has an invalid digest"
            );
        }
        ensure!(
            self.spec_path.is_file() && sha256_file(&self.spec_path)? == self.spec_sha256,
            "trajectory plan immutable spec drift"
        );
        Ok(())
    }
}

impl TrajectoryPlan {
    fn scientific_payload(&self) -> Result<DimeScientificPayloadV1> {
        let payload = DimeScientificPayloadV1 {
            parent_intent_sha256: sha256_bytes(&serde_json::to_vec(self)?),
            runtime_bundle_sha256: self.runtime_bundle_sha256.clone(),
            binary_sha256: self.binary_sha256.clone(),
            configuration_sha256: self.configuration_sha256.clone(),
            train_data_sha256: self.train_data_sha256.clone(),
            protocol_sha256: self.protocol_sha256.clone(),
            metric_sha256: self.metric_sha256.clone(),
            seed: self.seed,
            inputs: vec![
                DimeNamedDigestV1 {
                    name: "dataset".to_owned(),
                    sha256: self.dataset_sha256.clone(),
                },
                DimeNamedDigestV1 {
                    name: "frozen_spec".to_owned(),
                    sha256: self.spec_sha256.clone(),
                },
            ],
        };
        payload.validate()?;
        Ok(payload)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryProgress {
    schema: String,
    protocol_sha256: String,
    trajectory_identity: String,
    dataset_sha256: String,
    seed: u64,
    license_track: String,
    completed_evaluations: u64,
    completed_rungs: Vec<u64>,
    current_rung: Option<u64>,
    promotion_decisions: Vec<String>,
    test_csv_read: bool,
    updated_at_ms: u128,
}

impl TrajectoryProgress {
    fn validate(&self, plan: &TrajectoryPlan, protocol_sha256: &str) -> Result<()> {
        ensure!(
            self.schema == "JopeDlgpBudgetTrajectoryProgress.v1"
                && self.protocol_sha256 == protocol_sha256
                && self.trajectory_identity == plan.trajectory_identity
                && self.dataset_sha256 == plan.dataset_sha256
                && self.seed == plan.seed
                && self.license_track == plan.license_track
                && !self.test_csv_read
                && self.completed_evaluations <= 10_000
                && self
                    .completed_rungs
                    .iter()
                    .all(|rung| matches!(rung, 1_000 | 2_000 | 5_000 | 10_000))
                && self
                    .current_rung
                    .is_none_or(|rung| matches!(rung, 1_000 | 2_000 | 5_000 | 10_000)),
            "trajectory progress differs from its frozen plan"
        );
        ensure!(
            self.completed_rungs
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "trajectory rungs are not a unique nested prefix"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SchedulerJob {
    id: String,
    project: String,
    state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignLaunchReceipt {
    schema: String,
    scientific_payload_sha256: String,
    trajectory_identity: String,
    attempt_number: u64,
    job_identity: String,
    spec_sha256: String,
    submitted_at_ms: u128,
    content_sha256: String,
}

impl CampaignLaunchReceipt {
    fn new(
        payload_sha256: String,
        trajectory_identity: String,
        attempt_number: u64,
        job_identity: String,
        spec_sha256: String,
        submitted_at_ms: u128,
    ) -> Result<Self> {
        let mut value = Self {
            schema: "JopeDlgpBudgetCampaignLaunch.v1".to_owned(),
            scientific_payload_sha256: payload_sha256,
            trajectory_identity,
            attempt_number,
            job_identity,
            spec_sha256,
            submitted_at_ms,
            content_sha256: String::new(),
        };
        value.content_sha256 = value.computed_sha256()?;
        value.validate()?;
        Ok(value)
    }

    fn computed_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        Ok(sha256_bytes(&serde_json::to_vec(&copy)?))
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "JopeDlgpBudgetCampaignLaunch.v1"
                && self.scientific_payload_sha256.len() == 64
                && self.attempt_number > 0
                && !self.job_identity.trim().is_empty()
                && self.spec_sha256.len() == 64
                && self.submitted_at_ms > 0
                && self.computed_sha256()? == self.content_sha256,
            "campaign launch receipt drift"
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    loop {
        let status = match cycle(&args) {
            Ok(status) => status,
            Err(error) => JopeDlgpBudgetCampaignStatusV1 {
                schema: JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA.to_owned(),
                generated_at_ms: now_ms(),
                last_advanced_at_ms: 0,
                protocol_sha256: sha256_file(&args.protocol).unwrap_or_else(|_| "0".repeat(64)),
                stage: "supervisor_error".to_owned(),
                gate_state: DlgpCampaignGateStateV1::LaunchGatesBlocked,
                trajectory_count: 0,
                completed_trajectory_count: 0,
                rung_counts: BTreeMap::new(),
                evaluations_used: 0,
                evaluations_cap: STAGE1_MAX_SEARCH_EVALUATIONS,
                active_jobs: Vec::new(),
                robust_wins: 0,
                robust_wins_target: 7,
                robust_wins_denominator: 22,
                seed_agreement: "unavailable".to_owned(),
                license_track: "separate".to_owned(),
                production_compatible_state: "blocked".to_owned(),
                research_only_state: "blocked".to_owned(),
                test_opening_state: "sealed".to_owned(),
                promotion_decisions: Vec::new(),
                blockers: vec![format!("supervisor error: {error:#}")],
            },
        };
        atomic_json(&args.status_path(), &serde_json::to_value(&status)?)?;
        atomic_json(&args.compact_path(), &status.to_compact_json()?)?;
        println!("{}", serde_json::to_string(&status)?);
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}
