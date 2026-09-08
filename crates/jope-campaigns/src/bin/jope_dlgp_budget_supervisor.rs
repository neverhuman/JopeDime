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
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, now_ms, read_json, scq_json, sha256_bytes, sha256_file, DEFAULT_ROOT, DEFAULT_SCQ,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const DEFAULT_PROTOCOL: &str =
    "/home/ubuntu/jope-dime/manifests/dlgp-budget-first/protocol.v1.json";
const PROJECT: &str = "jope-dlgp-budget-first";

#[derive(Parser, Debug)]
#[command(
    name = "jope-dlgp-budget-supervisor",
    about = "Reconcile and resume only frozen budget-first DLGP trajectories"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_PROTOCOL)]
    protocol: PathBuf,
    #[arg(long)]
    campaign_root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_SCQ)]
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

fn load_plans(root: &Path, protocol_sha256: &str) -> Result<BTreeMap<String, TrajectoryPlan>> {
    let directory = root.join("plans");
    if !directory.is_dir() {
        return Ok(BTreeMap::new());
    }
    let mut plans = BTreeMap::new();
    for entry in fs::read_dir(directory)?.filter_map(|entry| entry.ok()) {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let plan: TrajectoryPlan = serde_json::from_value(read_json(&entry.path())?)?;
        plan.validate(protocol_sha256)?;
        ensure!(
            plans
                .insert(plan.trajectory_identity.clone(), plan)
                .is_none(),
            "duplicate trajectory plan identity"
        );
    }
    Ok(plans)
}

fn load_progress(
    root: &Path,
    plans: &BTreeMap<String, TrajectoryPlan>,
    protocol_sha256: &str,
) -> Result<BTreeMap<String, TrajectoryProgress>> {
    let directory = root.join("receipts");
    if !directory.is_dir() {
        return Ok(BTreeMap::new());
    }
    let mut progress: BTreeMap<String, TrajectoryProgress> = BTreeMap::new();
    for entry in fs::read_dir(directory)?.filter_map(|entry| entry.ok()) {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let value: TrajectoryProgress = serde_json::from_value(read_json(&entry.path())?)?;
        let plan = plans
            .get(&value.trajectory_identity)
            .context("trajectory receipt has no frozen plan")?;
        value.validate(plan, protocol_sha256)?;
        match progress.get(&value.trajectory_identity) {
            Some(existing) if existing.updated_at_ms == value.updated_at_ms => {
                ensure!(
                    serde_json::to_vec(existing)? == serde_json::to_vec(&value)?,
                    "duplicate trajectory completion conflicts"
                );
            }
            Some(existing) if existing.updated_at_ms > value.updated_at_ms => {}
            _ => {
                progress.insert(value.trajectory_identity.clone(), value);
            }
        }
    }
    Ok(progress)
}

fn scheduler_jobs(value: &Value) -> Vec<SchedulerJob> {
    value
        .pointer("/Jobs/jobs")
        .or_else(|| value.get("jobs"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|job| serde_json::from_value(job.clone()).ok())
        .collect()
}

fn launch_receipt_path(
    campaign_root: &Path,
    trajectory_identity: &str,
    attempt_number: u64,
) -> PathBuf {
    campaign_root
        .join("launches")
        .join(trajectory_identity)
        .join(format!("{attempt_number:08}.json"))
}

fn read_launch_receipt(
    campaign_root: &Path,
    trajectory_identity: &str,
    attempt_number: u64,
) -> Result<Option<CampaignLaunchReceipt>> {
    let path = launch_receipt_path(campaign_root, trajectory_identity, attempt_number);
    if !path.is_file() {
        return Ok(None);
    }
    let receipt: CampaignLaunchReceipt = serde_json::from_value(read_json(&path)?)?;
    receipt.validate()?;
    ensure!(
        receipt.trajectory_identity == trajectory_identity
            && receipt.attempt_number == attempt_number,
        "campaign launch receipt path/identity drift"
    );
    Ok(Some(receipt))
}

fn materialize_attempt_spec(
    campaign_root: &Path,
    plan: &TrajectoryPlan,
    attempt_number: u64,
    job_identity: &str,
) -> Result<PathBuf> {
    if attempt_number == 1 {
        ensure!(
            job_identity == plan.job_identity,
            "initial campaign attempt identity drift"
        );
        return Ok(plan.spec_path.clone());
    }
    let source = fs::read_to_string(&plan.spec_path)?;
    let mut in_job = false;
    let mut replacements = 0_u64;
    let mut rendered = String::with_capacity(source.len() + 32);
    for line in source.split_inclusive('\n') {
        let newline = if line.ends_with('\n') { "\n" } else { "" };
        let body = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = body.trim();
        if trimmed.starts_with('[') {
            in_job = trimmed == "[job]";
        }
        if in_job {
            if let Some((key, value)) = trimmed.split_once('=') {
                if key.trim() == "id" {
                    ensure!(
                        value.trim() == serde_json::to_string(&plan.job_identity)?,
                        "frozen campaign spec original job identity drift"
                    );
                    let indentation = &body[..body.len().saturating_sub(body.trim_start().len())];
                    rendered.push_str(indentation);
                    rendered.push_str("id = ");
                    rendered.push_str(&serde_json::to_string(job_identity)?);
                    rendered.push_str(newline);
                    replacements += 1;
                    continue;
                }
            }
        }
        rendered.push_str(line);
    }
    ensure!(
        replacements == 1,
        "campaign spec must contain exactly one [job].id"
    );
    let path = campaign_root
        .join("scq-specs/recovery")
        .join(format!("{job_identity}.toml"));
    if path.is_file() {
        ensure!(
            fs::read_to_string(&path)? == rendered,
            "campaign retry spec replay differs"
        );
        return Ok(path);
    }
    fs::create_dir_all(path.parent().context("campaign retry spec parent")?)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, rendered.as_bytes())?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

fn submit_attempt(
    args: &Args,
    campaign_root: &Path,
    plan: &TrajectoryPlan,
    payload_sha256: &str,
    attempt_number: u64,
    job_identity: &str,
) -> Result<CampaignLaunchReceipt> {
    if let Some(existing) =
        read_launch_receipt(campaign_root, &plan.trajectory_identity, attempt_number)?
    {
        ensure!(
            existing.scientific_payload_sha256 == payload_sha256
                && existing.job_identity == job_identity,
            "campaign launch replay changed payload or job identity"
        );
        return Ok(existing);
    }
    let spec = materialize_attempt_spec(campaign_root, plan, attempt_number, job_identity)?;
    let check = Command::new(&args.scq).arg("check").arg(&spec).output()?;
    ensure!(
        check.status.success(),
        "scq check failed for frozen trajectory {}: {}",
        plan.trajectory_identity,
        String::from_utf8_lossy(&check.stderr).trim()
    );
    let submit = Command::new(&args.scq).arg("submit").arg(&spec).output()?;
    ensure!(
        submit.status.success()
            || String::from_utf8_lossy(&submit.stderr).contains("already exists"),
        "submit failed for frozen trajectory {}: {}",
        plan.trajectory_identity,
        String::from_utf8_lossy(&submit.stderr).trim()
    );
    let receipt = CampaignLaunchReceipt::new(
        payload_sha256.to_owned(),
        plan.trajectory_identity.clone(),
        attempt_number,
        job_identity.to_owned(),
        sha256_file(&spec)?,
        now_ms(),
    )?;
    atomic_json(
        &launch_receipt_path(campaign_root, &plan.trajectory_identity, attempt_number),
        &serde_json::to_value(&receipt)?,
    )?;
    Ok(receipt)
}

fn terminal_class(state: &str) -> Option<DimeTerminalClassV1> {
    match state {
        "failed" | "cancelled" | "preempted" | "preempting" | "lost" | "vanished" => {
            Some(DimeTerminalClassV1::InfrastructureFailure)
        }
        "finished" | "completed" => Some(DimeTerminalClassV1::Completed),
        _ => None,
    }
}

fn next_attempt_identity(plan: &TrajectoryPlan, attempt_number: u64) -> String {
    format!("{}-infra{attempt_number:06}", plan.job_identity)
}

fn cycle(args: &Args) -> Result<JopeDlgpBudgetCampaignStatusV1> {
    let now = now_ms();
    let protocol: Value = read_json(&args.protocol)?;
    ensure!(
        protocol.get("schema").and_then(Value::as_str) == Some("JopeDlgpBudgetFirstProtocol.v1"),
        "budget-first protocol schema drift"
    );
    let protocol_sha256 = sha256_file(&args.protocol)?;
    let campaign_root = args.campaign_root();
    let plans = load_plans(&campaign_root, &protocol_sha256)?;
    let progress = load_progress(&campaign_root, &plans, &protocol_sha256)?;
    let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?);

    let mut blockers = Vec::new();
    let stage0_path = campaign_root.join("gates/stage0-reproduction.v1.json");
    let stage0 = if stage0_path.is_file() {
        Some(serde_json::from_value::<Stage0Receipt>(read_json(
            &stage0_path,
        )?)?)
    } else {
        None
    };
    let reproduction_passed = stage0.as_ref().is_some_and(|receipt| {
        receipt.schema == "JopeDlgpBudgetStage0Reproduction.v1"
            && receipt.protocol_sha256 == protocol_sha256
            && receipt.round2_payload_sha256
                == protocol
                    .pointer("/round2_evidence/payload_sha256")
                    .and_then(Value::as_str)
                    .unwrap_or("")
            && receipt.control_prefix_count == 6
            && !receipt.test_csv_read
            && receipt.reproduced
            && receipt.generated_at_ms > 0
    });
    if !reproduction_passed {
        blockers.push("Stage 0 reproduction and six 1k control prefixes are required".to_owned());
    }
    let gate_path = campaign_root.join("gates/launch-gates.v1.json");
    let gates = if gate_path.is_file() {
        Some(serde_json::from_value::<LaunchGates>(read_json(
            &gate_path,
        )?)?)
    } else {
        None
    };
    if reproduction_passed {
        if let Some(gates) = &gates {
            blockers.extend(gates.blockers(&protocol_sha256, now));
        } else {
            blockers.push("launch gate receipt is absent".to_owned());
        }
    }

    let mut plan_attempts = BTreeMap::new();
    let mut known_jobs = BTreeSet::new();
    for plan in plans.values() {
        let payload_sha256 = plan.scientific_payload()?.identity_sha256()?;
        let attempts =
            infrastructure_attempts(&args.root, DimeRoleV1::BudgetCampaign, &payload_sha256)?;
        known_jobs.insert(plan.job_identity.clone());
        for attempt in &attempts {
            known_jobs.insert(attempt.job_identity.clone());
            if let Some(next) = &attempt.next_job_identity {
                known_jobs.insert(next.clone());
            }
        }
        plan_attempts.insert(plan.trajectory_identity.clone(), attempts);
    }
    let foreign = jobs
        .iter()
        .filter(|job| {
            (job.project.starts_with(PROJECT) || job.project.contains("strategy-screen"))
                && !known_jobs.contains(&job.id)
        })
        .collect::<Vec<_>>();
    if !foreign.is_empty() {
        let quarantine = json!({
            "schema": "JopeDlgpForeignJobQuarantine.v1",
            "generated_at_ms": now,
            "protocol_sha256": protocol_sha256,
            "jobs": foreign.iter().map(|job| json!({
                "id": job.id,
                "project": job.project,
                "state": job.state,
                "disposition": "quarantined_not_adopted_not_cancelled",
            })).collect::<Vec<_>>(),
        });
        atomic_json(
            &campaign_root.join("quarantine/foreign-jobs-current.v1.json"),
            &quarantine,
        )?;
        blockers.push(format!(
            "{} strategy-screen jobs are foreign to the frozen campaign",
            foreign.len()
        ));
    }

    let evaluations_used = progress
        .values()
        .map(|value| value.completed_evaluations)
        .sum::<u64>();
    ensure!(
        evaluations_used <= STAGE1_MAX_SEARCH_EVALUATIONS,
        "partial and completed receipts exceed the 200,000-evaluation ceiling"
    );
    let mut rung_counts = BTreeMap::from([
        ("1000".to_owned(), 0),
        ("2000".to_owned(), 0),
        ("5000".to_owned(), 0),
        ("10000".to_owned(), 0),
    ]);
    let mut promotion_decisions = BTreeSet::new();
    for value in progress.values() {
        for rung in &value.completed_rungs {
            *rung_counts.entry(rung.to_string()).or_default() += 1;
        }
        promotion_decisions.extend(value.promotion_decisions.iter().cloned());
    }
    let launch_gates_clear = blockers.is_empty();
    let mut drain_suppressed = false;
    let mut active_jobs = Vec::new();
    let mut launch_candidates = Vec::new();
    let mut terminal_candidates = Vec::new();
    for plan in plans.values() {
        let completed = progress
            .get(&plan.trajectory_identity)
            .map_or(0, |value| value.completed_evaluations);
        if completed >= 10_000 {
            continue;
        }
        let payload = plan.scientific_payload()?;
        let payload_sha256 = payload.identity_sha256()?;
        let attempts = plan_attempts
            .get(&plan.trajectory_identity)
            .context("campaign attempt index missing")?;
        let (attempt_number, job_identity, retry_not_before_ms) =
            if let Some(last) = attempts.last() {
                let Some(next) = last.next_job_identity.clone() else {
                    continue;
                };
                (last.attempt_number + 1, next, last.retry_not_before_ms)
            } else {
                (1, plan.job_identity.clone(), 0)
            };
        let launch =
            read_launch_receipt(&campaign_root, &plan.trajectory_identity, attempt_number)?;
        let scheduler = jobs.iter().find(|job| job.id == job_identity);
        if let Some(job) = scheduler.filter(|job| {
            matches!(
                job.state.as_str(),
                "queued" | "starting" | "dispatching" | "running" | "held"
            )
        }) {
            active_jobs.push(DlgpActiveJobV1 {
                job_identity: job_identity.clone(),
                trajectory_identity: plan.trajectory_identity.clone(),
                attempt_number,
                state: job.state.clone(),
                progress_evaluations: completed,
                reason: "frozen trajectory; infrastructure identity is mutable only".to_owned(),
            });
            continue;
        }
        if let Some(job) = scheduler {
            if let Some(class) = terminal_class(&job.state) {
                if job.state == "cancelled"
                    && matching_drain_intent(&args.root, DimeRoleV1::BudgetCampaign, &payload, now)?
                        .is_some()
                {
                    drain_suppressed = true;
                    blockers.push(format!(
                        "trajectory {} intentionally drained by an exact DIME receipt",
                        plan.trajectory_identity
                    ));
                    continue;
                }
                terminal_candidates.push((
                    plan.trajectory_identity.clone(),
                    attempt_number,
                    job_identity,
                    job.state.clone(),
                    class,
                    job.state == "cancelled",
                    sha256_bytes(&serde_json::to_vec(job)?),
                    payload,
                ));
                continue;
            }
        }
        if let Some(launch) = launch {
            if now.saturating_sub(launch.submitted_at_ms) >= 60_000 {
                terminal_candidates.push((
                    plan.trajectory_identity.clone(),
                    attempt_number,
                    job_identity,
                    "vanished".to_owned(),
                    DimeTerminalClassV1::InfrastructureFailure,
                    false,
                    sha256_bytes(&serde_json::to_vec(&launch)?),
                    payload,
                ));
            } else {
                active_jobs.push(DlgpActiveJobV1 {
                    job_identity,
                    trajectory_identity: plan.trajectory_identity.clone(),
                    attempt_number,
                    state: "launching".to_owned(),
                    progress_evaluations: completed,
                    reason: "submitted; awaiting scheduler visibility".to_owned(),
                });
            }
            continue;
        }
        if now < retry_not_before_ms {
            active_jobs.push(DlgpActiveJobV1 {
                job_identity,
                trajectory_identity: plan.trajectory_identity.clone(),
                attempt_number,
                state: "backoff".to_owned(),
                progress_evaluations: completed,
                reason: format!("retry not before {retry_not_before_ms}"),
            });
        } else {
            launch_candidates.push((
                plan.trajectory_identity.clone(),
                attempt_number,
                job_identity,
                payload_sha256,
            ));
        }
    }

    if args.submit && launch_gates_clear {
        if let Some((
            trajectory,
            attempt_number,
            job_identity,
            state,
            class,
            foreign,
            evidence,
            payload,
        )) = terminal_candidates.into_iter().next()
        {
            let attempts = plan_attempts
                .get(&trajectory)
                .context("terminal campaign attempt index missing")?;
            let next_job_identity = plans
                .get(&trajectory)
                .map(|plan| next_attempt_identity(plan, attempt_number + 1))
                .context("terminal campaign plan missing")?;
            let receipt = DimeInfrastructureAttemptV1::new_terminal(
                DimeRoleV1::BudgetCampaign,
                payload,
                attempt_number,
                job_identity,
                Some(next_job_identity),
                attempts.last().map(|value| value.content_sha256.clone()),
                now,
                now.saturating_add(retry_backoff_ms(attempt_number)),
                DimeTerminalEvidenceV1 {
                    terminal_state: state.clone(),
                    class,
                    evidence_sha256: evidence,
                    observed_at_ms: now,
                    foreign_cancellation: foreign,
                },
            )?;
            append_infrastructure_attempt(&args.root, &receipt)?;
            if foreign {
                blockers.push(format!(
                    "foreign cancellation recovered for {trajectory}; scientific payload unchanged"
                ));
            }
        } else if let Some((trajectory, attempt_number, job_identity, payload_sha256)) =
            launch_candidates.into_iter().next()
        {
            ensure!(
                evaluations_used < STAGE1_MAX_SEARCH_EVALUATIONS,
                "evaluation ceiling leaves no launch budget"
            );
            let plan = plans
                .get(&trajectory)
                .context("campaign launch plan missing")?;
            submit_attempt(
                args,
                &campaign_root,
                plan,
                &payload_sha256,
                attempt_number,
                &job_identity,
            )?;
        }
    }
    let prior_foreign_cancellations = plan_attempts
        .values()
        .flatten()
        .filter(|attempt| attempt.terminal.foreign_cancellation)
        .count();
    if prior_foreign_cancellations >= 2 {
        blockers.push(format!(
            "warning: {prior_foreign_cancellations} repeated foreign campaign cancellations; recovery preserves scientific payloads"
        ));
    }

    let wins_path = campaign_root.join("receipts/robust-wins.v1.json");
    let wins = read_json(&wins_path).unwrap_or_else(|_| json!({}));
    let robust_wins = wins.get("robust_wins").and_then(Value::as_u64).unwrap_or(0);
    let seed_agreement = wins
        .get("seed_agreement")
        .and_then(Value::as_str)
        .unwrap_or("unavailable")
        .to_owned();
    let last_advanced_at_ms = progress
        .values()
        .map(|value| value.updated_at_ms)
        .max()
        .or_else(|| stage0.as_ref().map(|receipt| receipt.generated_at_ms))
        .unwrap_or(0);
    let complete = plans.len() == 20
        && progress.len() == plans.len()
        && progress
            .values()
            .all(|value| value.completed_evaluations >= 5_000);
    let gate_state = if complete {
        DlgpCampaignGateStateV1::Complete
    } else if !reproduction_passed {
        DlgpCampaignGateStateV1::ReproductionRequired
    } else if launch_gates_clear && !drain_suppressed {
        DlgpCampaignGateStateV1::Runnable
    } else {
        DlgpCampaignGateStateV1::LaunchGatesBlocked
    };
    let track_state = |track: &str| {
        if plans.values().any(|plan| plan.license_track == track) {
            "active"
        } else {
            "not_started"
        }
    };
    let status = JopeDlgpBudgetCampaignStatusV1 {
        schema: JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA.to_owned(),
        generated_at_ms: now,
        last_advanced_at_ms,
        protocol_sha256,
        stage: if complete {
            "stage1_complete"
        } else if reproduction_passed {
            "stage1"
        } else {
            "stage0_reproduction"
        }
        .to_owned(),
        gate_state,
        trajectory_count: plans.len() as u64,
        completed_trajectory_count: progress
            .values()
            .filter(|value| value.completed_evaluations >= 5_000)
            .count() as u64,
        rung_counts,
        evaluations_used,
        evaluations_cap: STAGE1_MAX_SEARCH_EVALUATIONS,
        active_jobs,
        robust_wins,
        robust_wins_target: 7,
        robust_wins_denominator: 22,
        seed_agreement,
        license_track: "separate".to_owned(),
        production_compatible_state: track_state("production_compatible").to_owned(),
        research_only_state: track_state("research_only").to_owned(),
        test_opening_state: "sealed_until_selection_freeze".to_owned(),
        promotion_decisions: promotion_decisions.into_iter().collect(),
        blockers,
    };
    status.validate()?;
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_bytes(label.as_bytes())
    }

    #[test]
    fn exact_disk_boundary_blocks_below_two_percent_and_only_warns_below_five() {
        let gate = LaunchGates {
            schema: "JopeDlgpBudgetLaunchGates.v1".to_owned(),
            protocol_sha256: "a".repeat(64),
            proposal_checkpoint_sha256: FROZEN_CHECKPOINT_SHA256.to_owned(),
            fresh_evaluator_evidence: true,
            fresh_supply_evidence: true,
            reviewed_acknowledgement: true,
            real_scq: true,
            observed_project_weight: 10,
            expected_project_weight: 10,
            disk_free_fraction_micros: 20_000,
            generated_at_ms: 100,
        };
        assert!(gate.blockers(&"a".repeat(64), 100).is_empty());
        let mut below = gate;
        below.disk_free_fraction_micros = 19_999;
        assert!(below
            .blockers(&"a".repeat(64), 100)
            .iter()
            .any(|value| value.contains("2%")));
    }

    #[test]
    fn status_counts_partial_evaluations_and_keeps_tracks_separate() {
        let status = JopeDlgpBudgetCampaignStatusV1 {
            schema: JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA.to_owned(),
            generated_at_ms: 10,
            last_advanced_at_ms: 9,
            protocol_sha256: "a".repeat(64),
            stage: "stage1".to_owned(),
            gate_state: DlgpCampaignGateStateV1::Runnable,
            trajectory_count: 20,
            completed_trajectory_count: 2,
            rung_counts: BTreeMap::from([("1000".to_owned(), 3)]),
            evaluations_used: 3_500,
            evaluations_cap: 200_000,
            active_jobs: Vec::new(),
            robust_wins: 3,
            robust_wins_target: 7,
            robust_wins_denominator: 22,
            seed_agreement: "3/22".to_owned(),
            license_track: "separate".to_owned(),
            production_compatible_state: "active".to_owned(),
            research_only_state: "not_started".to_owned(),
            test_opening_state: "sealed".to_owned(),
            promotion_decisions: Vec::new(),
            blockers: Vec::new(),
        };
        status.validate().unwrap();
        assert_eq!(status.evaluations_used, 3_500);
        assert_ne!(
            status.production_compatible_state,
            status.research_only_state
        );
    }

    #[test]
    fn recovery_spec_changes_only_the_infrastructure_identity() {
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("original.toml");
        fs::write(
            &original,
            "[job]\nid = \"budget-traj-a\"\nproject = \"jope-dlgp-budget-first\"\n\n[env]\nSEED = \"101\"\nDATA = \"frozen\"\n",
        )
        .unwrap();
        let plan = TrajectoryPlan {
            schema: "JopeDlgpBudgetTrajectoryPlan.v1".to_owned(),
            protocol_sha256: digest("protocol"),
            trajectory_identity: "trajectory-a".to_owned(),
            dataset_sha256: digest("dataset"),
            seed: 101,
            license_track: "production_compatible".to_owned(),
            runtime_bundle_sha256: digest("runtime"),
            binary_sha256: digest("binary"),
            configuration_sha256: digest("configuration"),
            train_data_sha256: digest("train"),
            metric_sha256: digest("metric"),
            job_identity: "budget-traj-a".to_owned(),
            spec_path: original.clone(),
            spec_sha256: sha256_file(&original).unwrap(),
        };
        plan.validate(&plan.protocol_sha256).unwrap();
        let retry =
            materialize_attempt_spec(root.path(), &plan, 6, "budget-traj-a-infra000006").unwrap();
        let retry_text = fs::read_to_string(retry).unwrap();
        assert!(retry_text.contains("id = \"budget-traj-a-infra000006\""));
        assert!(retry_text.contains("SEED = \"101\""));
        assert!(retry_text.contains("DATA = \"frozen\""));
        assert!(fs::read_to_string(original)
            .unwrap()
            .contains("id = \"budget-traj-a\""));
    }
}
