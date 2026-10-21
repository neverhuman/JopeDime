//! DIME-only runtime activation and rollback transaction.
//!
//! The transaction snapshots and mutates only the runtime pointer, active
//! document, DIME user units, their drop-ins, and their service state. Science
//! pointers are observed before every step and are never restoration inputs.

use crate::recovery_contracts::{JsonEvidenceRefV2, RawEvidenceRefV2, REQUIRED_NODES};
use crate::selfplay_runtime::sha256_bytes;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{symlink, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

pub const ACTIVATION_TRANSACTION_SCHEMA: &str = "JopeDimeActivationTransaction.v1";
pub const EXTERNAL_ACTIVATION_GATE_SCHEMA: &str = "JopeDimeExternalActivationGate.v1";
pub const SCQ_CAPABILITY_RECEIPT_SCHEMA: &str = "JopeScqCapabilityReceipt.v1";
pub const CANCELLATION_OWNERSHIP_SCHEMA: &str = "JopeCancellationOwnershipEvidence.v1";
pub const CANDIDATE_REVIEW_GATE_SCHEMA: &str = "JopeDimeCandidateReviewGate.v1";
pub const CANDIDATE_REVIEW_ACK_SCHEMA: &str = "JopeDimeCandidateReviewAcknowledgement.v1";
pub const CANDIDATE_TEST_RUN_SCHEMA: &str = "JopeDimeCandidateTestRun.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationOperationV1 {
    Activate,
    Rollback,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationStageV1 {
    Prepared,
    CandidateValidated,
    UnitsStaged,
    UnitsInstalled,
    RuntimeSwitched,
    ActivePublished,
    DaemonReloaded,
    TimersEnabled,
    ServicesRestarted,
    Complete,
    Aborted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScqCapabilityObservationV1 {
    pub schema: String,
    pub node: String,
    pub observed_hostname: String,
    pub scheduler_binary_sha256: String,
    pub eligible_for_dime_gpu: bool,
    pub gpu_companion_core_reserve: Option<u64>,
    pub observed_at_unix_ms: u64,
    pub raw_output: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationOwnershipEvidenceV1 {
    pub schema: String,
    pub owner: String,
    pub cancellation_message: String,
    pub source: String,
    pub observed_at_unix_ms: u64,
    pub raw_events: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeExternalActivationGateV1 {
    pub schema: String,
    pub candidate_bundle_sha256: String,
    pub generated_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub scq_capability_receipts: Vec<JsonEvidenceRefV2>,
    pub cancellation_ownership: JsonEvidenceRefV2,
}

impl DimeExternalActivationGateV1 {
    pub fn validate_at(&self, evidence_root: &Path, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == EXTERNAL_ACTIVATION_GATE_SCHEMA,
            "invalid DIME external activation-gate schema"
        );
        require_sha256(&self.candidate_bundle_sha256, "external gate candidate")?;
        ensure!(
            self.generated_at_unix_ms > 0
                && self.generated_at_unix_ms <= now_unix_ms
                && now_unix_ms < self.expires_at_unix_ms
                && self.expires_at_unix_ms - self.generated_at_unix_ms <= 60 * 60 * 1_000,
            "external activation gate is future-dated, stale, or overlong"
        );
        ensure!(
            self.scq_capability_receipts.len() == REQUIRED_NODES.len(),
            "external activation gate lacks five-node SCQ evidence"
        );
        let mut nodes = BTreeSet::new();
        for reference in &self.scq_capability_receipts {
            let receipt: ScqCapabilityObservationV1 = reference.load(evidence_root)?;
            ensure!(
                receipt.schema == SCQ_CAPABILITY_RECEIPT_SCHEMA
                    && REQUIRED_NODES.contains(&receipt.node.as_str())
                    && receipt.observed_hostname.split('.').next() == Some(receipt.node.as_str())
                    && receipt.observed_at_unix_ms >= self.generated_at_unix_ms
                    && receipt.observed_at_unix_ms < self.expires_at_unix_ms,
                "SCQ capability receipt has invalid node, host, or time scope"
            );
            require_sha256(&receipt.scheduler_binary_sha256, "SCQ scheduler binary")?;
            if receipt.eligible_for_dime_gpu {
                ensure!(
                    receipt
                        .gpu_companion_core_reserve
                        .is_some_and(|value| value >= 4),
                    "eligible node does not expose gpu_companion_core_reserve >= 4"
                );
            }
            receipt.raw_output.load(evidence_root)?;
            ensure!(
                nodes.insert(receipt.node),
                "duplicate SCQ capability node receipt"
            );
        }
        ensure!(
            REQUIRED_NODES.iter().all(|node| nodes.contains(*node)),
            "SCQ capability node set is incomplete"
        );
        let ownership: CancellationOwnershipEvidenceV1 =
            self.cancellation_ownership.load(evidence_root)?;
        ensure!(
            ownership.schema == CANCELLATION_OWNERSHIP_SCHEMA
                && !ownership.owner.trim().is_empty()
                && !ownership.source.trim().is_empty()
                && ownership.cancellation_message == "cancelled by operator"
                && ownership.observed_at_unix_ms >= self.generated_at_unix_ms
                && ownership.observed_at_unix_ms < self.expires_at_unix_ms,
            "cancellation ownership is absent, stale, or not tied to the operator message"
        );
        ownership.raw_events.load(evidence_root)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeCandidateTestRunV1 {
    pub schema: String,
    pub candidate_bundle_sha256: String,
    pub integration_commit: String,
    pub command: Vec<String>,
    pub exit_status: i32,
    pub passed: u64,
    pub failed: u64,
    pub ignored: u64,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
    pub stdout: RawEvidenceRefV2,
    pub stderr: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeCandidateReviewAcknowledgementV1 {
    pub schema: String,
    pub reviewer: String,
    pub candidate_bundle_sha256: String,
    pub integration_commit: String,
    pub bundle_manifest_file_sha256: String,
    pub bundle_manifest_canonical_json_sha256: String,
    pub test_runs: Vec<JsonEvidenceRefV2>,
    pub reviewed_at_unix_ms: u64,
    pub raw_review: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeCandidateReviewGateV1 {
    pub schema: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub candidate_bundle_sha256: String,
    pub integration_commit: String,
    pub bundle_manifest_file_sha256: String,
    pub bundle_manifest_canonical_json_sha256: String,
    pub acknowledgements: Vec<JsonEvidenceRefV2>,
}

impl DimeCandidateReviewGateV1 {
    pub fn validate_at(&self, evidence_root: &Path, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == CANDIDATE_REVIEW_GATE_SCHEMA,
            "invalid candidate review gate schema"
        );
        require_sha256(&self.candidate_bundle_sha256, "reviewed candidate")?;
        require_sha256(&self.bundle_manifest_file_sha256, "reviewed manifest file")?;
        require_sha256(
            &self.bundle_manifest_canonical_json_sha256,
            "reviewed manifest canonical JSON",
        )?;
        require_commit(&self.integration_commit)?;
        ensure!(
            self.issued_at_unix_ms > 0
                && self.issued_at_unix_ms <= now_unix_ms
                && now_unix_ms < self.expires_at_unix_ms
                && self.expires_at_unix_ms - self.issued_at_unix_ms <= 24 * 60 * 60 * 1_000,
            "candidate review gate is future-dated, stale, or overlong"
        );
        ensure!(
            self.acknowledgements.len() == 2,
            "candidate review gate requires exactly two acknowledgements"
        );
        let mut reviewers = BTreeSet::new();
        let mut evidence_paths = BTreeSet::new();
        let mut test_run_paths_by_reviewer = Vec::new();
        for reference in &self.acknowledgements {
            ensure!(
                evidence_paths.insert(reference.path.clone()),
                "candidate review acknowledgement is duplicated"
            );
            let acknowledgement: DimeCandidateReviewAcknowledgementV1 =
                reference.load(evidence_root)?;
            ensure!(
                acknowledgement.schema == CANDIDATE_REVIEW_ACK_SCHEMA
                    && matches!(acknowledgement.reviewer.as_str(), "codex" | "claude")
                    && acknowledgement.candidate_bundle_sha256 == self.candidate_bundle_sha256
                    && acknowledgement.integration_commit == self.integration_commit
                    && acknowledgement.bundle_manifest_file_sha256
                        == self.bundle_manifest_file_sha256
                    && acknowledgement.bundle_manifest_canonical_json_sha256
                        == self.bundle_manifest_canonical_json_sha256
                    && acknowledgement.reviewed_at_unix_ms >= self.issued_at_unix_ms
                    && acknowledgement.reviewed_at_unix_ms < self.expires_at_unix_ms,
                "candidate review acknowledgement scope mismatch"
            );
            ensure!(
                reviewers.insert(acknowledgement.reviewer.clone()),
                "candidate reviewer is duplicated"
            );
            acknowledgement.raw_review.load(evidence_root)?;
            ensure!(
                !acknowledgement.test_runs.is_empty(),
                "candidate reviewer supplied no reproduced test runs"
            );
            let mut reviewer_test_paths = BTreeSet::new();
            for test_reference in &acknowledgement.test_runs {
                ensure!(
                    reviewer_test_paths.insert(test_reference.path.clone()),
                    "candidate reviewer duplicated a test-run receipt"
                );
                let run: DimeCandidateTestRunV1 = test_reference.load(evidence_root)?;
                ensure!(
                    run.schema == CANDIDATE_TEST_RUN_SCHEMA
                        && run.candidate_bundle_sha256 == self.candidate_bundle_sha256
                        && run.integration_commit == self.integration_commit
                        && !run.command.is_empty()
                        && run
                            .command
                            .iter()
                            .all(|argument| !argument.trim().is_empty())
                        && run.exit_status == 0
                        && run.passed > 0
                        && run.failed == 0
                        && run.started_at_unix_ms >= self.issued_at_unix_ms
                        && run.finished_at_unix_ms >= run.started_at_unix_ms
                        && run.finished_at_unix_ms <= acknowledgement.reviewed_at_unix_ms,
                    "candidate test run is unbound, unsuccessful, or lacks exact counts"
                );
                run.stdout.load(evidence_root)?;
                run.stderr.load(evidence_root)?;
            }
            test_run_paths_by_reviewer.push(reviewer_test_paths);
        }
        ensure!(
            reviewers == BTreeSet::from(["claude".to_owned(), "codex".to_owned()]),
            "candidate review gate does not contain both independent reviewers"
        );
        ensure!(
            test_run_paths_by_reviewer[0].is_disjoint(&test_run_paths_by_reviewer[1]),
            "candidate reviewers did not independently reproduce test evidence"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationUnitCandidateV1 {
    pub unit: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationDropInPreimageV1 {
    pub relative_path: String,
    pub file: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationUnitPreimageV1 {
    pub unit: String,
    pub base_file: Option<RawEvidenceRefV2>,
    pub drop_ins: Vec<ActivationDropInPreimageV1>,
    pub enabled: bool,
    pub active: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationPreimageV1 {
    pub active_json: Option<RawEvidenceRefV2>,
    pub current_symlink_target: Option<String>,
    pub units: Vec<ActivationUnitPreimageV1>,
    pub recovery_tool: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeActivationTransactionV1 {
    pub schema: String,
    pub transaction_id: String,
    pub operation: ActivationOperationV1,
    pub stage: ActivationStageV1,
    pub candidate_bundle_sha256: String,
    pub rollback_capsule_sha256: String,
    pub candidate_review_gate_sha256: Option<String>,
    pub external_gate_sha256: Option<String>,
    pub active_document: Value,
    pub preimage: ActivationPreimageV1,
    pub candidate_units: Vec<ActivationUnitCandidateV1>,
    pub candidate_timers: Vec<String>,
    pub affected_services: Vec<String>,
    pub protected_science_sha256: BTreeMap<String, String>,
}

pub trait ActivationHost {
    fn is_enabled(&mut self, unit: &str) -> Result<bool>;
    fn is_active(&mut self, unit: &str) -> Result<bool>;
    fn validate_candidate(&mut self, staged_root: &Path, units: &[String]) -> Result<()>;
    fn daemon_reload(&mut self) -> Result<()>;
    fn enable_timers(&mut self, timers: &[String]) -> Result<()>;
    fn try_restart_services(&mut self, services: &[String]) -> Result<()>;
    fn restore_states(&mut self, units: &[ActivationUnitPreimageV1]) -> Result<()>;
}

pub struct SystemdUserHost;

impl SystemdUserHost {
    fn systemctl(args: &[&str]) -> Result<std::process::Output> {
        Command::new("systemctl")
            .arg("--user")
            .args(args)
            .output()
            .with_context(|| format!("run systemctl --user {}", args.join(" ")))
    }
}

impl ActivationHost for SystemdUserHost {
    fn is_enabled(&mut self, unit: &str) -> Result<bool> {
        Ok(Self::systemctl(&["is-enabled", "--quiet", unit])?
            .status
            .success())
    }

    fn is_active(&mut self, unit: &str) -> Result<bool> {
        Ok(Self::systemctl(&["is-active", "--quiet", unit])?
            .status
            .success())
    }

    fn validate_candidate(&mut self, staged_root: &Path, units: &[String]) -> Result<()> {
        let version = Command::new("systemd-analyze").arg("--version").output()?;
        ensure!(version.status.success(), "systemd-analyze --version failed");
        let output = Command::new("systemd-analyze")
            .arg("--user")
            .arg(format!("--root={}", staged_root.display()))
            .arg("verify")
            .args(units)
            .output()?;
        ensure!(
            output.status.success(),
            "candidate user-unit validation failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        Ok(())
    }

    fn daemon_reload(&mut self) -> Result<()> {
        let output = Self::systemctl(&["daemon-reload"])?;
        ensure!(
            output.status.success(),
            "systemctl --user daemon-reload failed"
        );
        Ok(())
    }

    fn enable_timers(&mut self, timers: &[String]) -> Result<()> {
        if timers.is_empty() {
            return Ok(());
        }
        let mut args = vec!["enable", "--now"];
        args.extend(timers.iter().map(String::as_str));
        let output = Self::systemctl(&args)?;
        ensure!(output.status.success(), "enable DIME timers failed");
        Ok(())
    }

    fn try_restart_services(&mut self, services: &[String]) -> Result<()> {
        if services.is_empty() {
            return Ok(());
        }
        let mut args = vec!["try-restart"];
        args.extend(services.iter().map(String::as_str));
        let output = Self::systemctl(&args)?;
        ensure!(
            output.status.success(),
            "restart affected DIME services failed"
        );
        Ok(())
    }

    fn restore_states(&mut self, units: &[ActivationUnitPreimageV1]) -> Result<()> {
        for unit in units {
            if unit.enabled {
                let output = Self::systemctl(&["enable", &unit.unit])?;
                ensure!(output.status.success(), "restore enabled state failed");
            } else {
                let output = Self::systemctl(&["disable", &unit.unit])?;
                ensure!(output.status.success(), "restore disabled state failed");
            }
            if unit.active {
                let output = Self::systemctl(&["start", &unit.unit])?;
                ensure!(output.status.success(), "restore active state failed");
            } else {
                let output = Self::systemctl(&["stop", &unit.unit])?;
                ensure!(output.status.success(), "restore inactive state failed");
            }
        }
        Ok(())
    }
}

pub struct ActivationLock {
    _file: File,
}

impl ActivationLock {
    pub fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        // Every failure, including EWOULDBLOCK or other errors, is refused below and
        // drops/closes `file`.
        ensure!(
            file.try_lock().is_ok(),
            "another DIME activation transaction owns the lock"
        );
        Ok(Self { _file: file })
    }
}

pub struct ActivationRequest<'a> {
    pub output_root: &'a Path,
    pub user_unit_root: &'a Path,
    pub transaction_root: &'a Path,
    pub candidate_bundle_sha256: &'a str,
    pub rollback_capsule_sha256: &'a str,
    pub candidate_review_gate_sha256: Option<&'a str>,
    pub external_gate_sha256: Option<&'a str>,
    pub active_document: Value,
    pub operation: ActivationOperationV1,
}

pub fn execute_activation_transaction(
    request: &ActivationRequest<'_>,
    host: &mut dyn ActivationHost,
    stop_after: Option<ActivationStageV1>,
) -> Result<DimeActivationTransactionV1> {
    require_sha256(request.candidate_bundle_sha256, "activation candidate")?;
    require_sha256(
        request.rollback_capsule_sha256,
        "activation rollback capsule",
    )?;
    if let Some(digest) = request.external_gate_sha256 {
        require_sha256(digest, "activation external gate")?;
    }
    if let Some(digest) = request.candidate_review_gate_sha256 {
        require_sha256(digest, "activation candidate review gate")?;
    }
    let candidate_root = request.output_root.join(request.candidate_bundle_sha256);
    ensure!(
        candidate_root.join("bundle.json").is_file(),
        "activation candidate is absent"
    );
    fs::create_dir_all(request.transaction_root)?;
    let transaction_path = request.transaction_root.join("transaction.v1.json");
    let mut transaction = if transaction_path.is_file() {
        let value: DimeActivationTransactionV1 =
            serde_json::from_slice(&fs::read(&transaction_path)?)?;
        ensure!(
            value.stage != ActivationStageV1::Aborted
                && value.candidate_bundle_sha256 == request.candidate_bundle_sha256
                && value.rollback_capsule_sha256 == request.rollback_capsule_sha256
                && value.candidate_review_gate_sha256
                    == request.candidate_review_gate_sha256.map(str::to_owned)
                && value.external_gate_sha256 == request.external_gate_sha256.map(str::to_owned)
                && value.active_document == request.active_document
                && value.operation == request.operation,
            "an incompatible or aborted activation transaction already exists"
        );
        if value.stage == ActivationStageV1::Complete {
            return Ok(value);
        }
        value
    } else {
        prepare_transaction(request, host, &candidate_root)?
    };
    write_transaction(&transaction_path, &transaction)?;
    if stop_after == Some(transaction.stage) {
        return Ok(transaction);
    }

    loop {
        match transaction.stage {
            ActivationStageV1::Prepared => {
                stage_units(
                    request.transaction_root,
                    &candidate_root,
                    &transaction.candidate_units,
                )?;
                transaction.stage = ActivationStageV1::UnitsStaged;
            }
            ActivationStageV1::UnitsStaged => {
                validate_candidate(&candidate_root, &transaction.candidate_units)?;
                let names = transaction
                    .candidate_units
                    .iter()
                    .map(|row| row.unit.clone())
                    .collect::<Vec<_>>();
                host.validate_candidate(&request.transaction_root.join("staged"), &names)?;
                transaction.stage = ActivationStageV1::CandidateValidated;
            }
            ActivationStageV1::CandidateValidated => {
                install_units(
                    request.user_unit_root,
                    request.transaction_root,
                    &transaction.preimage.units,
                    &transaction.candidate_units,
                )?;
                transaction.stage = ActivationStageV1::UnitsInstalled;
            }
            ActivationStageV1::UnitsInstalled => {
                switch_current(request.output_root, request.candidate_bundle_sha256)?;
                transaction.stage = ActivationStageV1::RuntimeSwitched;
            }
            ActivationStageV1::RuntimeSwitched => {
                atomic_json(
                    &request.output_root.join("active.json"),
                    &transaction.active_document,
                )?;
                transaction.stage = ActivationStageV1::ActivePublished;
            }
            ActivationStageV1::ActivePublished => {
                host.daemon_reload()?;
                transaction.stage = ActivationStageV1::DaemonReloaded;
            }
            ActivationStageV1::DaemonReloaded => {
                host.enable_timers(&transaction.candidate_timers)?;
                transaction.stage = ActivationStageV1::TimersEnabled;
            }
            ActivationStageV1::TimersEnabled => {
                host.try_restart_services(&transaction.affected_services)?;
                transaction.stage = ActivationStageV1::ServicesRestarted;
            }
            ActivationStageV1::ServicesRestarted => {
                transaction.stage = ActivationStageV1::Complete;
            }
            ActivationStageV1::Complete | ActivationStageV1::Aborted => return Ok(transaction),
        }
        write_transaction(&transaction_path, &transaction)?;
        if stop_after == Some(transaction.stage) {
            return Ok(transaction);
        }
    }
}

pub fn abort_activation_transaction(
    output_root: &Path,
    user_unit_root: &Path,
    transaction_root: &Path,
    host: &mut dyn ActivationHost,
) -> Result<DimeActivationTransactionV1> {
    let path = transaction_root.join("transaction.v1.json");
    let mut transaction: DimeActivationTransactionV1 = serde_json::from_slice(&fs::read(&path)?)?;
    ensure!(
        transaction.stage != ActivationStageV1::Complete
            && transaction.stage != ActivationStageV1::Aborted,
        "activation transaction is already terminal"
    );
    restore_units(
        user_unit_root,
        transaction_root,
        &transaction.preimage.units,
    )?;
    restore_raw_optional(
        &output_root.join("active.json"),
        transaction_root,
        transaction.preimage.active_json.as_ref(),
    )?;
    restore_current(
        output_root,
        transaction.preimage.current_symlink_target.as_deref(),
    )?;
    host.daemon_reload()?;
    host.restore_states(&transaction.preimage.units)?;
    transaction.stage = ActivationStageV1::Aborted;
    write_transaction(&path, &transaction)?;
    Ok(transaction)
}

fn prepare_transaction(
    request: &ActivationRequest<'_>,
    host: &mut dyn ActivationHost,
    candidate_root: &Path,
) -> Result<DimeActivationTransactionV1> {
    let candidate_units = candidate_units(candidate_root)?;
    let current_units = current_unit_names(request.output_root)?;
    let all_units = candidate_units
        .iter()
        .map(|row| row.unit.clone())
        .chain(current_units)
        .collect::<BTreeSet<_>>();
    let mut preimages = Vec::new();
    for unit in all_units {
        validate_unit_name(&unit)?;
        let base_path = request.user_unit_root.join(&unit);
        let base_file = if base_path.is_file() {
            Some(store_raw(
                request.transaction_root,
                &format!("preimages/units/{unit}"),
                &fs::read(&base_path)?,
            )?)
        } else {
            None
        };
        let drop_in_root = request.user_unit_root.join(format!("{unit}.d"));
        let mut drop_ins = Vec::new();
        if drop_in_root.is_dir() {
            let mut paths = collect_files(&drop_in_root)?;
            paths.sort();
            for path in paths {
                let relative = path
                    .strip_prefix(&drop_in_root)?
                    .to_string_lossy()
                    .into_owned();
                validate_relative(&relative, "drop-in preimage")?;
                drop_ins.push(ActivationDropInPreimageV1 {
                    relative_path: relative.clone(),
                    file: store_raw(
                        request.transaction_root,
                        &format!("preimages/drop-ins/{unit}/{relative}"),
                        &fs::read(path)?,
                    )?,
                });
            }
        }
        preimages.push(ActivationUnitPreimageV1 {
            unit: unit.clone(),
            base_file,
            drop_ins,
            enabled: host.is_enabled(&unit)?,
            active: host.is_active(&unit)?,
        });
    }
    let active_json = if request.output_root.join("active.json").is_file() {
        Some(store_raw(
            request.transaction_root,
            "preimages/active.json",
            &fs::read(request.output_root.join("active.json"))?,
        )?)
    } else {
        None
    };
    let current = request.output_root.join("current");
    let current_symlink_target = if current.is_symlink() {
        Some(fs::read_link(&current)?.to_string_lossy().into_owned())
    } else {
        ensure!(
            !current.exists(),
            "runtime current is not an atomic symlink"
        );
        None
    };
    let recovery_tool_path = current.join("runtime/bin/jope_selfplay_bundle");
    ensure!(
        recovery_tool_path.is_file(),
        "current runtime lacks its recovery tool"
    );
    let recovery_tool = store_raw(
        request.transaction_root,
        "recovery-tool/jope_selfplay_bundle",
        &fs::read(recovery_tool_path)?,
    )?;
    fs::set_permissions(
        request.transaction_root.join(&recovery_tool.path),
        fs::Permissions::from_mode(0o700),
    )?;
    let affected_services = preimages
        .iter()
        .filter(|preimage| preimage.unit.ends_with(".service"))
        .filter(|preimage| {
            let candidate = candidate_units.iter().find(|row| row.unit == preimage.unit);
            candidate.map(|row| row.sha256.as_str())
                != preimage
                    .base_file
                    .as_ref()
                    .map(|file| file.raw_file_sha256.as_str())
                || !preimage.drop_ins.is_empty()
        })
        .map(|preimage| preimage.unit.clone())
        .collect::<Vec<_>>();
    let candidate_timers = active_timer_units(candidate_root, &candidate_units)?;
    let transaction_id = sha256_bytes(
        format!(
            "{}\0{}\0{:?}",
            request.candidate_bundle_sha256, request.rollback_capsule_sha256, request.operation
        )
        .as_bytes(),
    );
    Ok(DimeActivationTransactionV1 {
        schema: ACTIVATION_TRANSACTION_SCHEMA.to_owned(),
        transaction_id,
        operation: request.operation,
        stage: ActivationStageV1::Prepared,
        candidate_bundle_sha256: request.candidate_bundle_sha256.to_owned(),
        rollback_capsule_sha256: request.rollback_capsule_sha256.to_owned(),
        candidate_review_gate_sha256: request.candidate_review_gate_sha256.map(str::to_owned),
        external_gate_sha256: request.external_gate_sha256.map(str::to_owned),
        active_document: request.active_document.clone(),
        preimage: ActivationPreimageV1 {
            active_json,
            current_symlink_target,
            units: preimages,
            recovery_tool,
        },
        candidate_units,
        candidate_timers,
        affected_services,
        protected_science_sha256: snapshot_protected_science(
            request
                .output_root
                .parent()
                .context("runtime root has no state parent")?,
        )?,
    })
}

fn candidate_units(candidate_root: &Path) -> Result<Vec<ActivationUnitCandidateV1>> {
    let root = candidate_root.join("runtime/systemd");
    ensure!(root.is_dir(), "candidate has no systemd directory");
    let mut units = fs::read_dir(root)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|extension| matches!(extension, "service" | "timer"))
        })
        .map(|entry| {
            let unit = entry.file_name().to_string_lossy().into_owned();
            validate_unit_name(&unit)?;
            Ok(ActivationUnitCandidateV1 {
                unit,
                sha256: sha256_file(&entry.path())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    units.sort_by(|left, right| left.unit.cmp(&right.unit));
    ensure!(
        !units.is_empty() && units.windows(2).all(|pair| pair[0].unit < pair[1].unit),
        "candidate systemd inventory is empty or duplicated"
    );
    Ok(units)
}

fn active_timer_units(
    candidate_root: &Path,
    candidates: &[ActivationUnitCandidateV1],
) -> Result<Vec<String>> {
    let manifest: Value = serde_json::from_slice(&fs::read(candidate_root.join("bundle.json"))?)?;
    let active_services = manifest
        .pointer("/service_manifest/services")
        .and_then(Value::as_array)
        .context("candidate bundle lacks its embedded service policy")?
        .iter()
        .filter(|row| row.get("active").and_then(Value::as_bool) == Some(true))
        .filter_map(|row| row.get("unit").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let mut timers = Vec::new();
    for candidate in candidates.iter().filter(|row| row.unit.ends_with(".timer")) {
        let contents =
            fs::read_to_string(candidate_root.join("runtime/systemd").join(&candidate.unit))?;
        let target = contents
            .lines()
            .map(str::trim)
            .find_map(|line| line.strip_prefix("Unit="))
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}.service", candidate.unit.trim_end_matches(".timer")));
        if active_services.contains(target.as_str()) {
            timers.push(candidate.unit.clone());
        }
    }
    Ok(timers)
}

fn current_unit_names(output_root: &Path) -> Result<Vec<String>> {
    let root = output_root.join("current/runtime/systemd");
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    Ok(fs::read_dir(root)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (name.ends_with(".service") || name.ends_with(".timer")).then_some(name)
        })
        .collect())
}

fn validate_candidate(candidate_root: &Path, units: &[ActivationUnitCandidateV1]) -> Result<()> {
    for unit in units {
        ensure!(
            sha256_file(&candidate_root.join("runtime/systemd").join(&unit.unit))? == unit.sha256,
            "candidate unit bytes drifted: {}",
            unit.unit
        );
    }
    Ok(())
}

fn stage_units(
    transaction_root: &Path,
    candidate_root: &Path,
    units: &[ActivationUnitCandidateV1],
) -> Result<()> {
    let staged = transaction_root.join("staged/etc/systemd/user");
    fs::create_dir_all(&staged)?;
    for unit in units {
        let destination = staged.join(&unit.unit);
        fs::copy(
            candidate_root.join("runtime/systemd").join(&unit.unit),
            &destination,
        )?;
        ensure!(
            sha256_file(&destination)? == unit.sha256,
            "staged unit copy drift"
        );
    }
    Ok(())
}

fn install_units(
    user_unit_root: &Path,
    transaction_root: &Path,
    preimages: &[ActivationUnitPreimageV1],
    candidates: &[ActivationUnitCandidateV1],
) -> Result<()> {
    fs::create_dir_all(user_unit_root)?;
    for preimage in preimages {
        validate_unit_name(&preimage.unit)?;
        let base = user_unit_root.join(&preimage.unit);
        if base.exists() {
            fs::remove_file(&base)?;
        }
        let drop_ins = user_unit_root.join(format!("{}.d", preimage.unit));
        if drop_ins.exists() {
            ensure!(
                drop_ins.starts_with(user_unit_root),
                "drop-in removal escaped user-unit root"
            );
            fs::remove_dir_all(&drop_ins)?;
        }
    }
    for candidate in candidates {
        let source = transaction_root
            .join("staged/etc/systemd/user")
            .join(&candidate.unit);
        let destination = user_unit_root.join(&candidate.unit);
        fs::copy(&source, &destination)?;
        ensure!(
            sha256_file(&destination)? == candidate.sha256,
            "installed unit copy drift"
        );
    }
    Ok(())
}

fn restore_units(
    user_unit_root: &Path,
    transaction_root: &Path,
    preimages: &[ActivationUnitPreimageV1],
) -> Result<()> {
    for preimage in preimages {
        let base = user_unit_root.join(&preimage.unit);
        restore_raw_optional(&base, transaction_root, preimage.base_file.as_ref())?;
        let drop_in_root = user_unit_root.join(format!("{}.d", preimage.unit));
        if drop_in_root.exists() {
            fs::remove_dir_all(&drop_in_root)?;
        }
        for drop_in in &preimage.drop_ins {
            let destination = drop_in_root.join(&drop_in.relative_path);
            restore_raw_optional(&destination, transaction_root, Some(&drop_in.file))?;
        }
    }
    Ok(())
}

fn switch_current(output_root: &Path, bundle_sha256: &str) -> Result<()> {
    ensure!(
        output_root
            .join(bundle_sha256)
            .join("bundle.json")
            .is_file(),
        "runtime target is absent"
    );
    let current = output_root.join("current");
    ensure!(
        !current.exists() || current.is_symlink(),
        "runtime current is not a symlink"
    );
    let temporary = output_root.join(format!(".current.activation-{}", std::process::id()));
    if temporary.is_symlink() {
        fs::remove_file(&temporary)?;
    }
    symlink(bundle_sha256, &temporary)?;
    fs::rename(temporary, current)?;
    Ok(())
}

fn restore_current(output_root: &Path, target: Option<&str>) -> Result<()> {
    let current = output_root.join("current");
    if current.is_symlink() {
        fs::remove_file(&current)?;
    }
    if let Some(target) = target {
        let temporary = output_root.join(format!(".current.restore-{}", std::process::id()));
        symlink(target, &temporary)?;
        fs::rename(temporary, current)?;
    }
    Ok(())
}

fn snapshot_protected_science(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    for relative in [
        "training/accepted.json",
        "training/accepted-by-task.json",
        "training/learning.json",
        "training/current-corpus.json",
        "training/task-learning-controller.v2.json",
        "ops/dlgp-budget-campaign-status.v1.json",
        "ops/micro-wave-supply-v2.json",
    ] {
        let path = root.join(relative);
        if path.is_file() {
            values.insert(relative.to_owned(), sha256_file(&path)?);
        }
    }
    Ok(values)
}

fn write_transaction(path: &Path, value: &DimeActivationTransactionV1) -> Result<()> {
    ensure!(
        value.schema == ACTIVATION_TRANSACTION_SCHEMA,
        "invalid activation transaction schema"
    );
    atomic_json(path, &serde_json::to_value(value)?)
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("JSON filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut output = File::create(&temporary)?;
    serde_json::to_writer(&mut output, value)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn store_raw(root: &Path, relative: &str, bytes: &[u8]) -> Result<RawEvidenceRefV2> {
    validate_relative(relative, "activation preimage")?;
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.is_file() {
        ensure!(
            fs::read(&path)? == bytes,
            "activation preimage replay drift"
        );
    } else {
        fs::write(&path, bytes)?;
    }
    Ok(RawEvidenceRefV2 {
        path: relative.to_owned(),
        raw_file_sha256: sha256_bytes(bytes),
    })
}

fn restore_raw_optional(
    destination: &Path,
    evidence_root: &Path,
    reference: Option<&RawEvidenceRefV2>,
) -> Result<()> {
    if let Some(reference) = reference {
        let bytes = reference.load(evidence_root)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(destination, bytes)?;
    } else if destination.exists() {
        fs::remove_file(destination)?;
    }
    Ok(())
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut output = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            output.extend(collect_files(&entry.path())?);
        } else if metadata.is_file() {
            output.push(entry.path());
        } else {
            bail!("activation preimage contains a non-file drop-in")
        }
    }
    Ok(output)
}

fn validate_unit_name(unit: &str) -> Result<()> {
    ensure!(
        !unit.contains('/') && (unit.ends_with(".service") || unit.ends_with(".timer")),
        "invalid DIME unit name"
    );
    Ok(())
}

fn validate_relative(value: &str, field: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(
        !path.is_absolute()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "{field} is not a normalized relative path"
    );
    Ok(())
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} is not a lowercase SHA-256"
    );
    Ok(())
}

fn require_commit(value: &str) -> Result<()> {
    ensure!(
        value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "integration commit is not a full Git object identity"
    );
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_reference(root: &Path, name: &str, value: &Value) -> JsonEvidenceRefV2 {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut bytes = serde_json::to_vec(value).unwrap();
        bytes.push(b'\n');
        fs::write(path, &bytes).unwrap();
        JsonEvidenceRefV2 {
            path: name.to_owned(),
            raw_file_sha256: sha256_bytes(&bytes),
            canonical_json_sha256: crate::selfplay_data::canonical_json_sha256(value).unwrap(),
        }
    }

    fn raw_reference(root: &Path, name: &str, bytes: &[u8]) -> RawEvidenceRefV2 {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, bytes).unwrap();
        RawEvidenceRefV2 {
            path: name.to_owned(),
            raw_file_sha256: sha256_bytes(bytes),
        }
    }

    #[derive(Default)]
    struct FakeHost {
        enabled: BTreeSet<String>,
        active: BTreeSet<String>,
        reloads: usize,
        enabled_batches: Vec<Vec<String>>,
        restarted: Vec<Vec<String>>,
    }

    impl ActivationHost for FakeHost {
        fn is_enabled(&mut self, unit: &str) -> Result<bool> {
            Ok(self.enabled.contains(unit))
        }
        fn is_active(&mut self, unit: &str) -> Result<bool> {
            Ok(self.active.contains(unit))
        }
        fn validate_candidate(&mut self, _root: &Path, _units: &[String]) -> Result<()> {
            Ok(())
        }
        fn daemon_reload(&mut self) -> Result<()> {
            self.reloads += 1;
            Ok(())
        }
        fn enable_timers(&mut self, timers: &[String]) -> Result<()> {
            self.enabled_batches.push(timers.to_vec());
            Ok(())
        }
        fn try_restart_services(&mut self, services: &[String]) -> Result<()> {
            self.restarted.push(services.to_vec());
            Ok(())
        }
        fn restore_states(&mut self, _units: &[ActivationUnitPreimageV1]) -> Result<()> {
            Ok(())
        }
    }

    fn fixture(root: &Path) -> (PathBuf, PathBuf, String, String) {
        let output = root.join("state/runtime-bundles");
        let units = root.join("user-units");
        let old = "a".repeat(64);
        let candidate = "b".repeat(64);
        for (identity, service) in [
            (&old, b"old-service".as_slice()),
            (&candidate, b"new-service".as_slice()),
        ] {
            let bundle = output.join(identity);
            fs::create_dir_all(bundle.join("runtime/systemd")).unwrap();
            fs::create_dir_all(bundle.join("runtime/bin")).unwrap();
            fs::write(
                bundle.join("bundle.json"),
                br#"{"service_manifest":{"services":[{"unit":"dime.service","active":true}]}}"#,
            )
            .unwrap();
            fs::write(bundle.join("runtime/systemd/dime.service"), service).unwrap();
            fs::write(bundle.join("runtime/systemd/dime.timer"), b"timer").unwrap();
            fs::write(bundle.join("runtime/bin/jope_selfplay_bundle"), b"recovery").unwrap();
        }
        fs::create_dir_all(&units).unwrap();
        fs::write(units.join("dime.service"), b"old-service").unwrap();
        symlink(&old, output.join("current")).unwrap();
        fs::write(output.join("active.json"), b"{\"old\":true}\n").unwrap();
        fs::create_dir_all(root.join("state/training")).unwrap();
        fs::write(root.join("state/training/accepted.json"), b"science").unwrap();
        (output, units, old, candidate)
    }

    #[test]
    fn interruption_at_every_step_resumes_without_touching_science() {
        let stages = [
            ActivationStageV1::Prepared,
            ActivationStageV1::CandidateValidated,
            ActivationStageV1::UnitsStaged,
            ActivationStageV1::UnitsInstalled,
            ActivationStageV1::RuntimeSwitched,
            ActivationStageV1::ActivePublished,
            ActivationStageV1::DaemonReloaded,
            ActivationStageV1::TimersEnabled,
            ActivationStageV1::ServicesRestarted,
        ];
        for stage in stages {
            let temporary = tempfile::tempdir().unwrap();
            let (output, units, _old, candidate) = fixture(temporary.path());
            let transaction = temporary.path().join("transaction");
            let capsule = "c".repeat(64);
            let gate = "d".repeat(64);
            let request = ActivationRequest {
                output_root: &output,
                user_unit_root: &units,
                transaction_root: &transaction,
                candidate_bundle_sha256: &candidate,
                rollback_capsule_sha256: &capsule,
                candidate_review_gate_sha256: Some(&gate),
                external_gate_sha256: Some(&gate),
                active_document: serde_json::json!({"bundle_sha256": candidate}),
                operation: ActivationOperationV1::Activate,
            };
            let mut host = FakeHost::default();
            let stopped = execute_activation_transaction(&request, &mut host, Some(stage)).unwrap();
            assert_eq!(stopped.stage, stage);
            fs::write(
                temporary.path().join("state/training/accepted.json"),
                b"science-advanced",
            )
            .unwrap();
            let complete = execute_activation_transaction(&request, &mut host, None).unwrap();
            assert_eq!(complete.stage, ActivationStageV1::Complete);
            assert_eq!(
                fs::read(temporary.path().join("state/training/accepted.json")).unwrap(),
                b"science-advanced"
            );
            assert_ne!(
                fs::metadata(transaction.join("recovery-tool/jope_selfplay_bundle"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o111,
                0
            );
            assert_eq!(
                fs::read_link(output.join("current")).unwrap(),
                PathBuf::from(&candidate)
            );
            assert_eq!(
                fs::read(units.join("dime.service")).unwrap(),
                b"new-service"
            );
        }
    }

    #[test]
    fn abort_restores_runtime_units_and_service_preimages_only() {
        let temporary = tempfile::tempdir().unwrap();
        let (output, units, old, candidate) = fixture(temporary.path());
        let transaction = temporary.path().join("transaction");
        let capsule = "c".repeat(64);
        let request = ActivationRequest {
            output_root: &output,
            user_unit_root: &units,
            transaction_root: &transaction,
            candidate_bundle_sha256: &candidate,
            rollback_capsule_sha256: &capsule,
            candidate_review_gate_sha256: None,
            external_gate_sha256: None,
            active_document: serde_json::json!({"bundle_sha256": candidate}),
            operation: ActivationOperationV1::Rollback,
        };
        let mut host = FakeHost::default();
        execute_activation_transaction(
            &request,
            &mut host,
            Some(ActivationStageV1::ActivePublished),
        )
        .unwrap();
        fs::write(
            temporary.path().join("state/training/accepted.json"),
            b"science-advanced",
        )
        .unwrap();
        let aborted =
            abort_activation_transaction(&output, &units, &transaction, &mut host).unwrap();
        assert_eq!(aborted.stage, ActivationStageV1::Aborted);
        assert_eq!(
            fs::read_link(output.join("current")).unwrap(),
            PathBuf::from(old)
        );
        assert_eq!(
            fs::read(units.join("dime.service")).unwrap(),
            b"old-service"
        );
        assert_eq!(
            fs::read(temporary.path().join("state/training/accepted.json")).unwrap(),
            b"science-advanced"
        );
    }

    #[test]
    fn external_gate_requires_real_companion_capacity_and_named_cancellation_owner() {
        let temporary = tempfile::tempdir().unwrap();
        let raw = raw_reference(temporary.path(), "raw.txt", b"observed");
        let mut receipts = Vec::new();
        for node in REQUIRED_NODES {
            let eligible = node != "xbabe4";
            receipts.push(json_reference(
                temporary.path(),
                &format!("{node}.json"),
                &serde_json::json!(ScqCapabilityObservationV1 {
                    schema: SCQ_CAPABILITY_RECEIPT_SCHEMA.to_owned(),
                    node: node.to_owned(),
                    observed_hostname: node.to_owned(),
                    scheduler_binary_sha256: "a".repeat(64),
                    eligible_for_dime_gpu: eligible,
                    gpu_companion_core_reserve: Some(if eligible { 4 } else { 0 }),
                    observed_at_unix_ms: 1_100,
                    raw_output: raw.clone(),
                }),
            ));
        }
        let ownership = json_reference(
            temporary.path(),
            "ownership.json",
            &serde_json::json!(CancellationOwnershipEvidenceV1 {
                schema: CANCELLATION_OWNERSHIP_SCHEMA.to_owned(),
                owner: "DLGP prison-hold campaign".to_owned(),
                cancellation_message: "cancelled by operator".to_owned(),
                source: "events.jsonl".to_owned(),
                observed_at_unix_ms: 1_100,
                raw_events: raw,
            }),
        );
        let mut gate = DimeExternalActivationGateV1 {
            schema: EXTERNAL_ACTIVATION_GATE_SCHEMA.to_owned(),
            candidate_bundle_sha256: "b".repeat(64),
            generated_at_unix_ms: 1_000,
            expires_at_unix_ms: 2_000,
            scq_capability_receipts: receipts,
            cancellation_ownership: ownership,
        };
        gate.validate_at(temporary.path(), 1_200).unwrap();

        let mut broken: ScqCapabilityObservationV1 = gate.scq_capability_receipts[0]
            .load(temporary.path())
            .unwrap();
        broken.gpu_companion_core_reserve = None;
        gate.scq_capability_receipts[0] = json_reference(
            temporary.path(),
            "broken-capability.json",
            &serde_json::json!(broken),
        );
        assert!(gate.validate_at(temporary.path(), 1_200).is_err());
    }

    #[test]
    fn candidate_review_gate_requires_independent_hash_bound_test_reproduction() {
        let temporary = tempfile::tempdir().unwrap();
        let bundle = "a".repeat(64);
        let integration_commit = "b".repeat(40);
        let manifest_file = "c".repeat(64);
        let manifest_canonical = "d".repeat(64);
        let mut acknowledgements = Vec::new();
        let mut test_references = Vec::new();
        for reviewer in ["codex", "claude"] {
            let run = DimeCandidateTestRunV1 {
                schema: CANDIDATE_TEST_RUN_SCHEMA.to_owned(),
                candidate_bundle_sha256: bundle.clone(),
                integration_commit: integration_commit.clone(),
                command: vec!["cargo".to_owned(), "test".to_owned()],
                exit_status: 0,
                passed: 14,
                failed: 0,
                ignored: 0,
                started_at_unix_ms: 1_100,
                finished_at_unix_ms: 1_500,
                stdout: raw_reference(
                    temporary.path(),
                    &format!("{reviewer}-stdout.txt"),
                    b"14 passed",
                ),
                stderr: raw_reference(temporary.path(), &format!("{reviewer}-stderr.txt"), b""),
            };
            let test_reference = json_reference(
                temporary.path(),
                &format!("{reviewer}-test.json"),
                &serde_json::to_value(run).unwrap(),
            );
            test_references.push(test_reference.clone());
            let acknowledgement = DimeCandidateReviewAcknowledgementV1 {
                schema: CANDIDATE_REVIEW_ACK_SCHEMA.to_owned(),
                reviewer: reviewer.to_owned(),
                candidate_bundle_sha256: bundle.clone(),
                integration_commit: integration_commit.clone(),
                bundle_manifest_file_sha256: manifest_file.clone(),
                bundle_manifest_canonical_json_sha256: manifest_canonical.clone(),
                test_runs: vec![test_reference],
                reviewed_at_unix_ms: 1_600,
                raw_review: raw_reference(
                    temporary.path(),
                    &format!("{reviewer}-review.txt"),
                    reviewer.as_bytes(),
                ),
            };
            acknowledgements.push(json_reference(
                temporary.path(),
                &format!("{reviewer}-ack.json"),
                &serde_json::to_value(acknowledgement).unwrap(),
            ));
        }
        let mut gate = DimeCandidateReviewGateV1 {
            schema: CANDIDATE_REVIEW_GATE_SCHEMA.to_owned(),
            issued_at_unix_ms: 1_000,
            expires_at_unix_ms: 3_000,
            candidate_bundle_sha256: bundle.clone(),
            integration_commit: integration_commit.clone(),
            bundle_manifest_file_sha256: manifest_file.clone(),
            bundle_manifest_canonical_json_sha256: manifest_canonical.clone(),
            acknowledgements,
        };
        gate.validate_at(temporary.path(), 2_000).unwrap();

        let duplicate_ack = DimeCandidateReviewAcknowledgementV1 {
            schema: CANDIDATE_REVIEW_ACK_SCHEMA.to_owned(),
            reviewer: "claude".to_owned(),
            candidate_bundle_sha256: bundle,
            integration_commit,
            bundle_manifest_file_sha256: manifest_file,
            bundle_manifest_canonical_json_sha256: manifest_canonical,
            test_runs: vec![test_references[0].clone()],
            reviewed_at_unix_ms: 1_600,
            raw_review: raw_reference(temporary.path(), "duplicate-review.txt", b"claude"),
        };
        gate.acknowledgements[1] = json_reference(
            temporary.path(),
            "duplicate-ack.json",
            &serde_json::to_value(duplicate_ack).unwrap(),
        );
        assert!(gate.validate_at(temporary.path(), 2_000).is_err());
    }
}
