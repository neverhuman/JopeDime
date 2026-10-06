//! Fail-closed recovery authorities for the Phase 5.1 cutover.
//!
//! The types in this module are deliberately independent of the active
//! runtime pointer.  In particular, a rollback capsule binds an unchanged
//! runtime bundle and a separately content-addressed unit overlay.

use crate::encoder_v2_identity::ArtifactIdentity;
use crate::paths;
use crate::selfplay_data::canonical_json_sha256;
use crate::selfplay_runtime::sha256_bytes;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path};

pub const V1_QUARANTINE_SCHEMA: &str = "JopeV1EncodingQuarantine.v1";
pub const ROLLBACK_CAPSULE_SCHEMA: &str = "JopeRollbackCapsule.v1";
pub const ROLLBACK_CAPSULE_V2_SCHEMA: &str = "JopeRollbackCapsule.v2";
pub const ROLLBACK_BASELINE_DERIVATION_SCHEMA: &str = "JopeRollbackBaselineDerivation.v1";
pub const ROLLBACK_CHALLENGE_SCHEMA: &str = "JopeRollbackChallenge.v1";
pub const ROLLBACK_NODE_PROBE_SCHEMA: &str = "JopeRollbackNodeProbe.v2";
pub const ROLLBACK_PATH_AUDIT_SCHEMA: &str = "JopeRollbackPathAudit.v2";
pub const ROLLBACK_INSTALL_SIMULATION_SCHEMA: &str = "JopeRollbackInstallSimulation.v2";
pub const ROLLBACK_EVALUATOR_RETENTION_SCHEMA: &str = "JopeRollbackEvaluatorRetention.v2";
pub const ROLLBACK_STATE_COMPATIBILITY_SCHEMA: &str = "JopeRollbackStateCompatibility.v2";
pub const RECOVERY_CONTINUITY_OBSERVATION_SCHEMA: &str = "JopeRecoveryContinuityObservation.v1";
pub const ROLLBACK_RUNTIME_BUNDLE_SHA256: &str =
    "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4";
jope_paths::once_str!(
    pub fn immutable_flywheel_root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    pub fn runtime_current_root_str => paths::runtime_bundle_current()
);
pub const REQUIRED_NODES: [&str; 5] = ["xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4"];
pub const CONTINUITY_SERVICES: [&str; 7] = [
    "freezer",
    "freshness",
    "ingestion",
    "liveness",
    "queue",
    "reporting",
    "supply",
];

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

fn require_absolute_normalized(value: &str, field: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(path.is_absolute(), "{field} is not absolute");
    ensure!(
        path.components()
            .all(|part| !matches!(part, Component::ParentDir | Component::CurDir)),
        "{field} is not normalized"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuityNodeBundleReceipt {
    pub node: String,
    pub receipt_sha256: String,
    pub runtime_bundle_sha256: String,
    pub active_process_count: u64,
    pub verified: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuityServiceReceipt {
    pub service: String,
    pub status_sha256: String,
    pub fresh: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryContinuityObservationV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub incident_containment_sha256: String,
    pub quarantine_index_sha256: String,
    pub runtime_bundle_sha256: String,
    pub nodes: Vec<ContinuityNodeBundleReceipt>,
    pub active_process_bundle_sha256: Vec<String>,
    pub causal_evidence_sha256: String,
    pub causal_success: bool,
    pub telemetry_sha256: String,
    pub telemetry_rows_exact_and_fresh: bool,
    pub useful_work: u64,
    pub useful_work_target: u64,
    pub protected_complete_waves: u64,
    pub services: Vec<ContinuityServiceReceipt>,
    pub accepted_model_pointer_sha256: String,
    pub model_pointer_sha256: String,
    pub held_job_inventory_sha256: String,
    pub held_job_count: u64,
    pub ownerless_hold_count: u64,
    pub evaluation_cancellation_count_since_epoch: u64,
    pub v1_output_inventory_sha256: String,
    pub new_v1_output_count_since_epoch: u64,
    pub v1_quarantine_consumption_count_since_epoch: u64,
    pub expansion_active: bool,
    pub blockers: Vec<String>,
}

impl RecoveryContinuityObservationV1 {
    pub fn validate_at(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == RECOVERY_CONTINUITY_OBSERVATION_SCHEMA,
            "invalid recovery-continuity observation schema"
        );
        ensure!(
            self.generated_at_unix_ms > 0
                && self.generated_at_unix_ms <= now_unix_ms
                && self.freshness_budget_seconds > 0
                && self.freshness_budget_seconds <= 120
                && now_unix_ms - self.generated_at_unix_ms
                    <= self.freshness_budget_seconds.saturating_mul(1_000),
            "recovery-continuity observation is absent, future-dated, or stale"
        );
        ensure!(
            self.nodes.len() == REQUIRED_NODES.len(),
            "continuity node receipts are incomplete"
        );
        let mut expected_inputs = BTreeSet::new();
        for digest in [
            &self.incident_containment_sha256,
            &self.quarantine_index_sha256,
            &self.causal_evidence_sha256,
            &self.telemetry_sha256,
            &self.accepted_model_pointer_sha256,
            &self.model_pointer_sha256,
            &self.held_job_inventory_sha256,
            &self.v1_output_inventory_sha256,
        ] {
            require_sha256(digest, "continuity evidence digest")?;
            expected_inputs.insert(digest.clone());
        }
        require_sha256(&self.runtime_bundle_sha256, "continuity runtime bundle")?;
        expected_inputs.insert(self.runtime_bundle_sha256.clone());
        let mut nodes = BTreeSet::new();
        for receipt in &self.nodes {
            ensure!(
                REQUIRED_NODES.contains(&receipt.node.as_str())
                    && nodes.insert(receipt.node.as_str()),
                "continuity node receipt set is invalid"
            );
            require_sha256(&receipt.receipt_sha256, "continuity node receipt")?;
            ensure!(
                receipt.runtime_bundle_sha256 == self.runtime_bundle_sha256
                    && receipt.active_process_count > 0
                    && receipt.verified,
                "continuity node does not prove the common runtime bundle"
            );
            expected_inputs.insert(receipt.receipt_sha256.clone());
        }
        ensure!(
            self.active_process_bundle_sha256.len() >= REQUIRED_NODES.len()
                && self
                    .active_process_bundle_sha256
                    .iter()
                    .all(|digest| digest == &self.runtime_bundle_sha256),
            "active processes do not resolve to one runtime bundle"
        );
        ensure!(
            self.services.len() == CONTINUITY_SERVICES.len(),
            "continuity service receipts are incomplete"
        );
        for (receipt, expected) in self.services.iter().zip(CONTINUITY_SERVICES) {
            ensure!(
                receipt.service == expected && receipt.fresh,
                "continuity service is absent, out of order, or stale: {expected}"
            );
            require_sha256(&receipt.status_sha256, "continuity service status")?;
            expected_inputs.insert(receipt.status_sha256.clone());
        }
        ensure!(
            self.causal_success && self.telemetry_rows_exact_and_fresh,
            "causal evidence or exact four-task telemetry is not continuously green"
        );
        ensure!(
            self.useful_work >= self.useful_work_target
                && self.useful_work_target >= 6_144
                && self.protected_complete_waves >= 2,
            "useful-work target or protected reserves are not continuously satisfied"
        );
        ensure!(
            self.held_job_count == 3
                && self.ownerless_hold_count == 0
                && self.evaluation_cancellation_count_since_epoch == 0,
            "held-job ownership or evaluation-cancellation invariant failed"
        );
        ensure!(
            self.new_v1_output_count_since_epoch == 0
                && self.v1_quarantine_consumption_count_since_epoch == 0
                && !self.expansion_active,
            "V1 output, quarantine consumption, or expansion activation observed"
        );
        ensure!(
            self.blockers.is_empty(),
            "continuity observation has blockers"
        );
        ensure!(
            self.input_digests.windows(2).all(|pair| pair[0] < pair[1])
                && self.input_digests.iter().all(|digest| is_sha256(digest))
                && self.input_digests.iter().cloned().collect::<BTreeSet<_>>() == expected_inputs,
            "continuity input digest set is incomplete or noncanonical"
        );
        Ok(())
    }

    pub fn epoch_identity_sha256(&self, now_unix_ms: u64) -> Result<String> {
        self.validate_at(now_unix_ms)?;
        canonical_json_sha256(&json!({
            "domain": "jope-recovery-continuity-epoch-v1",
            "incident_containment_sha256": self.incident_containment_sha256,
            "quarantine_index_sha256": self.quarantine_index_sha256,
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "accepted_model_pointer_sha256": self.accepted_model_pointer_sha256,
            "model_pointer_sha256": self.model_pointer_sha256,
            "held_job_inventory_sha256": self.held_job_inventory_sha256,
            "v1_output_inventory_sha256": self.v1_output_inventory_sha256,
        }))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantineFileIdentity {
    pub path: String,
    pub file_type: String,
    pub size_bytes: u64,
    pub mode: u32,
    pub sha256: String,
}

impl QuarantineFileIdentity {
    fn validate(&self) -> Result<()> {
        let path = Path::new(&self.path);
        ensure!(!path.is_absolute(), "quarantine file path is absolute");
        ensure!(
            path.components()
                .all(|part| matches!(part, Component::Normal(_))),
            "quarantine file path is not canonical: {}",
            self.path
        );
        ensure!(
            matches!(self.file_type.as_str(), "regular" | "symlink"),
            "unsupported quarantine file type: {}",
            self.file_type
        );
        ensure!(self.mode <= 0o7777, "quarantine file mode is invalid");
        require_sha256(&self.sha256, "quarantine file SHA-256")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantinedEncodingTree {
    pub identity: String,
    pub state: String,
    pub original_path: String,
    pub quarantine_path: String,
    pub files: Vec<QuarantineFileIdentity>,
    pub aggregate_directory_sha256: String,
}

impl QuarantinedEncodingTree {
    pub fn computed_directory_sha256(&self) -> Result<String> {
        canonical_json_sha256(&json!({
            "domain": "jope-v1-quarantine-directory-v1",
            "files": self.files,
        }))
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.identity.is_empty(), "quarantined identity is absent");
        ensure!(
            matches!(self.state.as_str(), "complete" | "partial"),
            "quarantined tree state is invalid"
        );
        require_absolute_normalized(&self.original_path, "original quarantine path")?;
        require_absolute_normalized(&self.quarantine_path, "destination quarantine path")?;
        ensure!(
            !self.original_path.contains("/incident-quarantine/"),
            "original path is already in incident quarantine"
        );
        ensure!(
            self.quarantine_path
                .contains(&format!("/incident-quarantine/{}/", self.state)),
            "quarantine destination does not match tree state"
        );
        ensure!(
            !self.files.is_empty(),
            "quarantined tree has no file inventory"
        );
        let mut paths = BTreeSet::new();
        for file in &self.files {
            file.validate()?;
            ensure!(
                paths.insert(file.path.as_str()),
                "quarantined tree repeats file path: {}",
                file.path
            );
        }
        ensure!(
            self.files
                .windows(2)
                .all(|pair| pair[0].path < pair[1].path),
            "quarantine file inventory is not strictly sorted"
        );
        require_sha256(
            &self.aggregate_directory_sha256,
            "aggregate quarantine directory SHA-256",
        )?;
        ensure!(
            self.computed_directory_sha256()? == self.aggregate_directory_sha256,
            "aggregate quarantine directory digest does not match its inventory"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantinePolicyFlags {
    pub catalog_allowed: bool,
    pub training_allowed: bool,
    pub benchmark_allowed: bool,
    pub v2_import_allowed: bool,
    pub gc_allowed: bool,
}

impl QuarantinePolicyFlags {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.catalog_allowed
                && !self.training_allowed
                && !self.benchmark_allowed
                && !self.v2_import_allowed
                && !self.gc_allowed,
            "V1 quarantine policy permits consumption or collection"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MutableScriptObservation {
    pub script_name: String,
    pub sha256: String,
    pub observed_paths: Vec<String>,
}

impl MutableScriptObservation {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.script_name.as_str(),
                "multiclass_drain.py" | "test_exact_expansion.py"
            ),
            "unexpected mutable incident script"
        );
        require_sha256(&self.sha256, "mutable script SHA-256")?;
        ensure!(
            self.observed_paths.len() == 2,
            "mutable script node paths are incomplete"
        );
        for node in ["xbabe1", "xbabe3"] {
            ensure!(
                self.observed_paths
                    .iter()
                    .any(|path| path.contains(node) && path.ends_with(&self.script_name)),
                "mutable script observation is absent for {node}"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceExclusionProof {
    pub authority: String,
    pub authority_sha256: String,
    pub quarantined_reference_count: u64,
}

impl ReferenceExclusionProof {
    fn validate(&self) -> Result<()> {
        ensure!(!self.authority.is_empty(), "reference authority is absent");
        require_sha256(&self.authority_sha256, "reference authority SHA-256")?;
        ensure!(
            self.quarantined_reference_count == 0,
            "authority {} references quarantined identities",
            self.authority
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct V1EncodingQuarantineV1 {
    pub schema: String,
    pub incident_id: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub discovered_at_unix_ms: u64,
    pub contained_at_unix_ms: u64,
    pub governing_freeze_entry_sha256: String,
    pub responsible_execution_paths: Vec<ArtifactIdentity>,
    pub trees: Vec<QuarantinedEncodingTree>,
    pub policy: QuarantinePolicyFlags,
    pub mutable_scripts: Vec<MutableScriptObservation>,
    pub non_consumption: Vec<ReferenceExclusionProof>,
}

impl V1EncodingQuarantineV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == V1_QUARANTINE_SCHEMA,
            "invalid V1 quarantine schema"
        );
        ensure!(!self.incident_id.is_empty(), "incident ID is absent");
        ensure!(
            self.freshness_budget_seconds > 0,
            "freshness budget is zero"
        );
        ensure!(
            self.discovered_at_unix_ms <= self.contained_at_unix_ms
                && self.contained_at_unix_ms <= self.generated_at_unix_ms,
            "quarantine timestamps are inconsistent"
        );
        require_sha256(
            &self.governing_freeze_entry_sha256,
            "governing freeze entry SHA-256",
        )?;
        ensure!(
            !self.responsible_execution_paths.is_empty(),
            "responsible execution paths are absent"
        );
        for artifact in &self.responsible_execution_paths {
            artifact.validate()?;
        }
        self.policy.validate()?;
        ensure!(
            self.trees
                .iter()
                .filter(|tree| tree.state == "complete")
                .count()
                == 11,
            "V1 quarantine does not index exactly 11 complete caches"
        );
        ensure!(
            self.trees
                .iter()
                .filter(|tree| tree.state == "partial")
                .count()
                == 7,
            "V1 quarantine does not index exactly seven partial trees"
        );
        let mut identities = BTreeSet::new();
        let mut original_paths = BTreeSet::new();
        let mut quarantine_paths = BTreeSet::new();
        for tree in &self.trees {
            tree.validate()?;
            ensure!(
                identities.insert(tree.identity.as_str()),
                "duplicate V1 identity"
            );
            ensure!(
                original_paths.insert(tree.original_path.as_str()),
                "duplicate V1 original path"
            );
            ensure!(
                quarantine_paths.insert(tree.quarantine_path.as_str()),
                "duplicate V1 quarantine path"
            );
        }
        ensure!(
            self.trees.windows(2).all(|pair| {
                (&pair[0].state, &pair[0].identity) < (&pair[1].state, &pair[1].identity)
            }),
            "V1 quarantine trees are not in complete-then-partial identity order"
        );
        ensure!(
            self.mutable_scripts.len() == 2,
            "V1 incident does not name exactly two mutable scripts"
        );
        for script in &self.mutable_scripts {
            script.validate()?;
        }
        let required_authorities = [
            "catalog",
            "corpus",
            "packed_corpus",
            "observations",
            "accepted_model",
            "runtime_pointer",
            "model_pointer",
        ];
        ensure!(
            self.non_consumption.len() == required_authorities.len(),
            "non-consumption proof is incomplete"
        );
        for authority in required_authorities {
            let proof = self
                .non_consumption
                .iter()
                .find(|proof| proof.authority == authority)
                .with_context(|| format!("non-consumption proof lacks {authority}"))?;
            proof.validate()?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }

    pub fn denies_identity(&self, identity: &str) -> bool {
        self.trees.iter().any(|tree| tree.identity == identity)
    }

    pub fn require_operation_allowed(&self, identity: &str, operation: &str) -> Result<()> {
        ensure!(
            !self.denies_identity(identity),
            "{operation} denied for quarantined V1 identity {identity}"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantineRelocationReceipt {
    pub identity: String,
    pub original_path: String,
    pub quarantine_path: String,
    pub aggregate_directory_sha256: String,
    pub atomic_rename: bool,
}

/// Relocate exactly the frozen V1 inventory with same-filesystem atomic
/// renames. All sources are checked before the first rename; each destination
/// is checked again before the next tree moves.
pub fn relocate_v1_quarantine(
    index: &V1EncodingQuarantineV1,
) -> Result<Vec<QuarantineRelocationReceipt>> {
    index.validate()?;
    for tree in &index.trees {
        verify_quarantined_tree_at(tree, Path::new(&tree.original_path))?;
        let destination = Path::new(&tree.quarantine_path);
        let parent = destination
            .parent()
            .context("quarantine destination has no parent")?;
        ensure!(parent.is_dir(), "quarantine destination parent is absent");
        ensure!(
            !destination.exists(),
            "quarantine destination already exists"
        );
        ensure!(
            fs::metadata(&tree.original_path)?.dev() == fs::metadata(parent)?.dev(),
            "quarantine relocation crosses filesystems"
        );
    }
    let mut receipts = Vec::with_capacity(index.trees.len());
    for tree in &index.trees {
        fs::rename(&tree.original_path, &tree.quarantine_path)?;
        verify_quarantined_tree_at(tree, Path::new(&tree.quarantine_path))?;
        receipts.push(QuarantineRelocationReceipt {
            identity: tree.identity.clone(),
            original_path: tree.original_path.clone(),
            quarantine_path: tree.quarantine_path.clone(),
            aggregate_directory_sha256: tree.aggregate_directory_sha256.clone(),
            atomic_rename: true,
        });
    }
    Ok(receipts)
}

pub fn verify_quarantined_tree_at(tree: &QuarantinedEncodingTree, root: &Path) -> Result<()> {
    ensure!(root.is_dir(), "quarantine tree root is absent");
    let mut actual = BTreeSet::new();
    collect_relative_files(root, root, &mut actual)?;
    let expected = tree
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        actual.iter().map(String::as_str).collect::<BTreeSet<_>>() == expected,
        "quarantine tree file inventory drift"
    );
    for expected in &tree.files {
        let path = root.join(&expected.path);
        let metadata = fs::symlink_metadata(&path)?;
        let observed_type = if metadata.file_type().is_symlink() {
            "symlink"
        } else if metadata.is_file() {
            "regular"
        } else {
            "unsupported"
        };
        ensure!(
            observed_type == expected.file_type,
            "quarantine file-type drift"
        );
        ensure!(
            metadata.len() == expected.size_bytes,
            "quarantine file-size drift"
        );
        ensure!(
            metadata.permissions().mode() & 0o7777 == expected.mode,
            "quarantine file-mode drift"
        );
        let digest = if metadata.file_type().is_symlink() {
            hex::encode(Sha256::digest(fs::read_link(&path)?.as_os_str().as_bytes()))
        } else {
            sha256_file(&path)?
        };
        ensure!(digest == expected.sha256, "quarantine file-content drift");
    }
    ensure!(
        tree.computed_directory_sha256()? == tree.aggregate_directory_sha256,
        "quarantine aggregate digest drift"
    );
    Ok(())
}

fn collect_relative_files(
    root: &Path,
    directory: &Path,
    output: &mut BTreeSet<String>,
) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            collect_relative_files(root, &entry.path(), output)?;
        } else {
            output.insert(
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hex::encode(hash.finalize()))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackUnitIdentity {
    pub unit: String,
    pub unit_file_sha256: String,
    pub resolved_effective_unit_sha256: String,
    pub exec_start_argv: Vec<String>,
    pub working_directory: String,
    pub drop_ins: Vec<ArtifactIdentity>,
    pub evaluator: bool,
}

impl RollbackUnitIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.unit.ends_with(".service"),
            "rollback unit name is invalid"
        );
        require_sha256(&self.unit_file_sha256, "rollback unit-file SHA-256")?;
        require_sha256(
            &self.resolved_effective_unit_sha256,
            "resolved effective-unit SHA-256",
        )?;
        ensure!(
            !self.exec_start_argv.is_empty(),
            "rollback unit has no ExecStart"
        );
        let executable = &self.exec_start_argv[0];
        require_absolute_normalized(executable, "rollback ExecStart")?;
        ensure!(
            executable.starts_with(&format!(
                "{runtime_current_root_str}/runtime/bin/",
                runtime_current_root_str = runtime_current_root_str()
            )),
            "rollback ExecStart is not bundle-pinned: {executable}"
        );
        ensure!(
            !executable.ends_with("python")
                && !executable.ends_with("python3")
                && !executable.ends_with(".py"),
            "rollback ExecStart directly invokes Python"
        );
        for argument in self.exec_start_argv.iter().skip(1) {
            if argument.ends_with(".py") {
                require_absolute_normalized(argument, "rollback Python argument")?;
                ensure!(
                    argument.starts_with(&format!(
                        "{runtime_current_root_str}/pytorch/",
                        runtime_current_root_str = runtime_current_root_str()
                    )),
                    "rollback unit references Python outside the bundled PyTorch boundary"
                );
            }
            ensure!(
                !argument.contains("/jope-dime/")
                    && !argument.contains("/jope-prime/")
                    && !argument.contains("/jope-slime/"),
                "rollback unit references a mutable checkout"
            );
        }
        ensure!(
            self.working_directory == immutable_flywheel_root_str(),
            "rollback WorkingDirectory is mutable or noncanonical"
        );
        let mut drop_in_paths = BTreeSet::new();
        for drop_in in &self.drop_ins {
            drop_in.validate()?;
            ensure!(
                drop_in_paths.insert(drop_in.path.as_str()),
                "rollback unit repeats a drop-in"
            );
        }
        ensure!(
            self.drop_ins
                .windows(2)
                .all(|pair| pair[0].path < pair[1].path),
            "rollback drop-ins are not sorted"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackNodeReceipt {
    pub node: String,
    pub runtime_bundle_sha256: String,
    pub unit_overlay_sha256: String,
    pub runtime_file_count: u64,
    pub unit_count: u64,
    pub verified_at_unix_ms: u64,
}

impl RollbackNodeReceipt {
    fn validate(&self, runtime: &str, overlay: &str, unit_count: usize) -> Result<()> {
        ensure!(
            REQUIRED_NODES.contains(&self.node.as_str()),
            "unknown rollback node"
        );
        ensure!(
            self.runtime_bundle_sha256 == runtime && self.unit_overlay_sha256 == overlay,
            "rollback node runtime/unit identity mismatch"
        );
        ensure!(
            self.runtime_file_count > 0,
            "rollback runtime inventory is empty"
        );
        ensure!(
            self.unit_count == unit_count as u64,
            "rollback node unit inventory is incomplete"
        );
        ensure!(
            self.verified_at_unix_ms > 0,
            "rollback node receipt has no timestamp"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackCapsuleV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub runtime_bundle_sha256: String,
    pub runtime_bundle_manifest_sha256: String,
    pub unit_overlay_sha256: String,
    pub units: Vec<RollbackUnitIdentity>,
    pub service_manifest_sha256: String,
    pub evaluator_retention_index_sha256: String,
    pub accepted_model_pointer_snapshot: Value,
    pub accepted_model_pointer_snapshot_sha256: String,
    pub model_pointer_sha256: String,
    pub database_cursor_snapshot: Value,
    pub database_cursor_snapshot_sha256: String,
    pub node_receipts: Vec<RollbackNodeReceipt>,
    pub absolute_path_audit_receipt_sha256: String,
    pub simulated_install_receipt_sha256: String,
    pub simulated_install_passed: bool,
}

impl RollbackCapsuleV1 {
    pub fn computed_unit_overlay_sha256(&self) -> Result<String> {
        canonical_json_sha256(&json!({
            "domain": "jope-rollback-unit-overlay-v1",
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "units": self.units,
        }))
    }

    pub fn validate_at(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == ROLLBACK_CAPSULE_SCHEMA,
            "invalid rollback capsule schema"
        );
        ensure!(
            self.freshness_budget_seconds > 0,
            "rollback freshness budget is zero"
        );
        ensure!(
            now_unix_ms >= self.generated_at_unix_ms,
            "rollback capsule is future-dated"
        );
        ensure!(
            now_unix_ms - self.generated_at_unix_ms
                <= self.freshness_budget_seconds.saturating_mul(1_000),
            "rollback unit overlay evidence is stale"
        );
        ensure!(
            self.runtime_bundle_sha256 == ROLLBACK_RUNTIME_BUNDLE_SHA256,
            "rollback runtime bundle identity drift"
        );
        for (value, field) in [
            (
                &self.runtime_bundle_manifest_sha256,
                "runtime bundle manifest SHA-256",
            ),
            (&self.unit_overlay_sha256, "unit overlay SHA-256"),
            (&self.service_manifest_sha256, "service manifest SHA-256"),
            (
                &self.evaluator_retention_index_sha256,
                "evaluator-retention index SHA-256",
            ),
            (
                &self.accepted_model_pointer_snapshot_sha256,
                "accepted-model snapshot SHA-256",
            ),
            (&self.model_pointer_sha256, "model pointer SHA-256"),
            (
                &self.database_cursor_snapshot_sha256,
                "database cursor snapshot SHA-256",
            ),
            (
                &self.absolute_path_audit_receipt_sha256,
                "absolute-path audit receipt SHA-256",
            ),
            (
                &self.simulated_install_receipt_sha256,
                "simulated-install receipt SHA-256",
            ),
        ] {
            require_sha256(value, field)?;
        }
        ensure!(!self.units.is_empty(), "rollback unit inventory is absent");
        let mut unit_names = BTreeSet::new();
        for unit in &self.units {
            unit.validate()?;
            ensure!(
                unit_names.insert(unit.unit.as_str()),
                "duplicate rollback unit"
            );
        }
        ensure!(
            self.units
                .windows(2)
                .all(|pair| pair[0].unit < pair[1].unit),
            "rollback unit inventory is not sorted"
        );
        ensure!(
            self.units.iter().any(|unit| unit.evaluator),
            "rollback unit inventory has no retained evaluator"
        );
        ensure!(
            self.computed_unit_overlay_sha256()? == self.unit_overlay_sha256,
            "rollback unit overlay digest mismatch"
        );
        ensure!(
            canonical_json_sha256(&self.accepted_model_pointer_snapshot)?
                == self.accepted_model_pointer_snapshot_sha256,
            "accepted-model pointer snapshot drift"
        );
        ensure!(
            canonical_json_sha256(&self.database_cursor_snapshot)?
                == self.database_cursor_snapshot_sha256,
            "database cursor snapshot drift"
        );
        ensure!(
            self.node_receipts.len() == REQUIRED_NODES.len(),
            "rollback five-node receipts are incomplete"
        );
        let mut nodes = BTreeSet::new();
        for receipt in &self.node_receipts {
            receipt.validate(
                &self.runtime_bundle_sha256,
                &self.unit_overlay_sha256,
                self.units.len(),
            )?;
            ensure!(
                nodes.insert(receipt.node.as_str()),
                "duplicate rollback node receipt"
            );
        }
        ensure!(
            REQUIRED_NODES.iter().all(|node| nodes.contains(node)),
            "rollback receipt node set is incomplete"
        );
        ensure!(
            self.simulated_install_passed,
            "simulated rollback install failed"
        );
        Ok(())
    }

    pub fn digest_at(&self, now_unix_ms: u64) -> Result<String> {
        self.validate_at(now_unix_ms)?;
        canonical_json_sha256(&json!({
            "domain": "jope-composite-rollback-capsule-v1",
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "unit_overlay_sha256": self.unit_overlay_sha256,
            "capsule": self,
        }))
    }

    pub fn verify_effective_units(&self, observed: &[RollbackUnitIdentity]) -> Result<()> {
        ensure!(
            observed == self.units,
            "installed rollback unit inventory or drop-ins differ from the capsule"
        );
        Ok(())
    }
}

/// A v1 capsule can still be decoded for historical labelling, but this is
/// deliberately the only authority it retains.
pub fn classify_rollback_capsule_schema(value: &Value) -> Result<&'static str> {
    match value.get("schema").and_then(Value::as_str) {
        Some(ROLLBACK_CAPSULE_SCHEMA) => Ok("historical-non-authorizing-v1"),
        Some(ROLLBACK_CAPSULE_V2_SCHEMA) => Ok("authorizing-v2"),
        _ => anyhow::bail!("unknown rollback capsule schema"),
    }
}

fn require_relative_normalized(value: &str, field: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(!path.is_absolute(), "{field} is absolute");
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_))),
        "{field} is not a normalized relative path"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JsonEvidenceRefV2 {
    pub path: String,
    pub raw_file_sha256: String,
    pub canonical_json_sha256: String,
}

impl JsonEvidenceRefV2 {
    pub fn validate(&self) -> Result<()> {
        require_relative_normalized(&self.path, "JSON evidence path")?;
        require_sha256(&self.raw_file_sha256, "JSON evidence raw-file SHA-256")?;
        require_sha256(
            &self.canonical_json_sha256,
            "JSON evidence canonical SHA-256",
        )
    }

    pub fn load_value(&self, root: &Path) -> Result<Value> {
        self.validate()?;
        let path = root.join(&self.path);
        let bytes =
            fs::read(&path).with_context(|| format!("read JSON evidence {}", path.display()))?;
        ensure!(
            sha256_bytes(&bytes) == self.raw_file_sha256,
            "JSON evidence raw-file digest mismatch: {}",
            self.path
        );
        let value: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode JSON evidence {}", path.display()))?;
        ensure!(
            canonical_json_sha256(&value)? == self.canonical_json_sha256,
            "JSON evidence canonical digest mismatch: {}",
            self.path
        );
        Ok(value)
    }

    pub fn load<T: for<'de> Deserialize<'de>>(&self, root: &Path) -> Result<T> {
        serde_json::from_value(self.load_value(root)?)
            .with_context(|| format!("decode typed JSON evidence {}", self.path))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawEvidenceRefV2 {
    pub path: String,
    pub raw_file_sha256: String,
}

impl RawEvidenceRefV2 {
    pub fn load(&self, root: &Path) -> Result<Vec<u8>> {
        require_relative_normalized(&self.path, "raw evidence path")?;
        require_sha256(&self.raw_file_sha256, "raw evidence file SHA-256")?;
        let path = root.join(&self.path);
        let bytes =
            fs::read(&path).with_context(|| format!("read raw evidence {}", path.display()))?;
        ensure!(
            sha256_bytes(&bytes) == self.raw_file_sha256,
            "raw evidence digest mismatch: {}",
            self.path
        );
        Ok(bytes)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackBundleFileV2 {
    pub target: String,
    pub bytes: u64,
    pub sha256: String,
}

impl RollbackBundleFileV2 {
    fn validate(&self) -> Result<()> {
        require_relative_normalized(&self.target, "rollback bundle target")?;
        require_sha256(&self.sha256, "rollback bundle file SHA-256")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackSystemdArtifactV2 {
    pub unit: String,
    pub kind: String,
    pub bundled_target: String,
    pub unit_file_sha256: String,
    pub resolved_effective_unit_sha256: String,
    pub exec_start_argv: Vec<String>,
    pub working_directory: Option<String>,
    pub drop_ins: Vec<ArtifactIdentity>,
}

impl RollbackSystemdArtifactV2 {
    pub fn computed_effective_unit_sha256(&self) -> Result<String> {
        canonical_json_sha256(&json!({
            "domain": "jope-rollback-effective-unit-v2",
            "unit": self.unit,
            "unit_file_sha256": self.unit_file_sha256,
            "drop_ins": self.drop_ins,
        }))
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.kind.as_str(), "service" | "timer")
                && self.unit.ends_with(&format!(".{}", self.kind))
                && !self.unit.contains('/'),
            "rollback systemd artifact kind/name is invalid"
        );
        ensure!(
            self.bundled_target == format!("runtime/systemd/{}", self.unit),
            "rollback systemd bundled target is not canonical"
        );
        require_sha256(&self.unit_file_sha256, "rollback unit-file SHA-256")?;
        require_sha256(
            &self.resolved_effective_unit_sha256,
            "rollback effective-unit SHA-256",
        )?;
        if self.kind == "service" {
            ensure!(
                !self.exec_start_argv.is_empty(),
                "rollback service has no ExecStart"
            );
            require_absolute_normalized(&self.exec_start_argv[0], "rollback ExecStart")?;
            ensure!(
                !self.exec_start_argv.iter().any(|argument| {
                    argument.contains("/jope-dime/")
                        || argument.contains("/jope-prime/")
                        || argument.contains("/jope-slime/")
                }),
                "rollback service references a mutable checkout"
            );
            if let Some(working_directory) = &self.working_directory {
                ensure!(
                    working_directory == immutable_flywheel_root_str(),
                    "rollback WorkingDirectory is mutable or noncanonical"
                );
            }
        } else {
            ensure!(
                self.exec_start_argv.is_empty() && self.working_directory.is_none(),
                "rollback timer contains service-only fields"
            );
        }
        let mut paths = BTreeSet::new();
        for drop_in in &self.drop_ins {
            drop_in.validate()?;
            ensure!(
                paths.insert(&drop_in.path),
                "rollback drop-in is duplicated"
            );
        }
        ensure!(
            self.drop_ins
                .windows(2)
                .all(|pair| pair[0].path < pair[1].path),
            "rollback drop-ins are not sorted"
        );
        ensure!(
            self.computed_effective_unit_sha256()? == self.resolved_effective_unit_sha256,
            "rollback effective-unit composition digest mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackChallengeV1 {
    pub schema: String,
    pub nonce_sha256: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub verifier_tool_sha256: String,
}

impl RollbackChallengeV1 {
    fn validate_at(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == ROLLBACK_CHALLENGE_SCHEMA,
            "invalid rollback challenge schema"
        );
        require_sha256(&self.nonce_sha256, "rollback challenge nonce")?;
        require_sha256(&self.verifier_tool_sha256, "rollback verifier tool")?;
        ensure!(
            self.issued_at_unix_ms > 0
                && self.issued_at_unix_ms <= now_unix_ms
                && now_unix_ms < self.expires_at_unix_ms
                && self.expires_at_unix_ms - self.issued_at_unix_ms <= 15 * 60 * 1_000,
            "rollback challenge is future-dated, stale, or overlong"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackProbeToolV2 {
    pub executable_sha256: String,
    pub version: String,
    pub argv: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackNodeProbeV2 {
    pub schema: String,
    pub challenge_nonce_sha256: String,
    pub node: String,
    pub observed_hostname: String,
    pub runtime_bundle_sha256: String,
    pub bundle_manifest_file_sha256: String,
    pub bundle_manifest_canonical_json_sha256: String,
    pub files: Vec<RollbackBundleFileV2>,
    pub systemd_artifacts: Vec<RollbackSystemdArtifactV2>,
    pub tool: RollbackProbeToolV2,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
    pub raw_output: RawEvidenceRefV2,
}

struct RollbackBundleEvidence<'a> {
    bundle_sha256: &'a str,
    manifest_file_sha256: &'a str,
    manifest_canonical_sha256: &'a str,
    files: &'a [RollbackBundleFileV2],
    artifacts: &'a [RollbackSystemdArtifactV2],
}

impl RollbackNodeProbeV2 {
    fn validate(
        &self,
        root: &Path,
        challenge: &RollbackChallengeV1,
        bundle: RollbackBundleEvidence<'_>,
    ) -> Result<()> {
        let RollbackBundleEvidence {
            bundle_sha256,
            manifest_file_sha256,
            manifest_canonical_sha256,
            files,
            artifacts,
        } = bundle;
        ensure!(
            self.schema == ROLLBACK_NODE_PROBE_SCHEMA,
            "invalid rollback node-probe schema"
        );
        ensure!(
            self.challenge_nonce_sha256 == challenge.nonce_sha256,
            "rollback node probe challenge mismatch"
        );
        ensure!(
            REQUIRED_NODES.contains(&self.node.as_str())
                && self.observed_hostname.split('.').next() == Some(self.node.as_str()),
            "rollback node probe hostname mismatch"
        );
        ensure!(
            self.runtime_bundle_sha256 == bundle_sha256
                && self.bundle_manifest_file_sha256 == manifest_file_sha256
                && self.bundle_manifest_canonical_json_sha256 == manifest_canonical_sha256,
            "rollback node probe bundle/manifest mismatch"
        );
        ensure!(
            self.files == files,
            "rollback node probe file inventory is incomplete or altered"
        );
        ensure!(
            self.systemd_artifacts == artifacts,
            "rollback node probe systemd inventory is incomplete or altered"
        );
        require_sha256(
            &self.tool.executable_sha256,
            "rollback probe tool executable",
        )?;
        ensure!(
            !self.tool.version.trim().is_empty() && !self.tool.argv.is_empty(),
            "rollback probe tool identity is incomplete"
        );
        ensure!(
            self.started_at_unix_ms >= challenge.issued_at_unix_ms
                && self.finished_at_unix_ms >= self.started_at_unix_ms
                && self.finished_at_unix_ms < challenge.expires_at_unix_ms,
            "rollback node probe timestamps do not fit the challenge"
        );
        self.raw_output.load(root)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackPathAuditV2 {
    pub schema: String,
    pub challenge_nonce_sha256: String,
    pub runtime_bundle_sha256: String,
    pub audited_systemd_artifacts: Vec<String>,
    pub findings: Vec<Value>,
    pub raw_output: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackInstallSimulationV2 {
    pub schema: String,
    pub challenge_nonce_sha256: String,
    pub runtime_bundle_sha256: String,
    pub installer_argv: Vec<String>,
    pub installer_executable_sha256: String,
    pub installer_version: String,
    pub installer_exit_status: i32,
    pub systemd_analyze_argv: Vec<String>,
    pub systemd_analyze_version: String,
    pub systemd_analyze_exit_status: i32,
    pub staged_tree_sha256: String,
    pub expected_staged_tree_sha256: String,
    pub stdout: RawEvidenceRefV2,
    pub stderr: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackEvaluatorRetentionV2 {
    pub schema: String,
    pub challenge_nonce_sha256: String,
    pub runtime_bundle_sha256: String,
    pub retention_index: JsonEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackStateDocumentV2 {
    pub role: String,
    pub schema: String,
    pub document: JsonEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackStateBinaryV2 {
    pub role: String,
    pub bundled_target: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackStateCompatibilityV2 {
    pub schema: String,
    pub challenge_nonce_sha256: String,
    pub runtime_bundle_sha256: String,
    pub rollback_binaries: Vec<RollbackStateBinaryV2>,
    pub command: Vec<String>,
    pub exit_status: i32,
    pub stdout: RawEvidenceRefV2,
    pub stderr: RawEvidenceRefV2,
    pub primary_documents: Vec<RollbackStateDocumentV2>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackCapsuleV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub runtime_bundle_sha256: String,
    pub runtime_bundle_manifest_file_sha256: String,
    pub runtime_bundle_manifest_canonical_json_sha256: String,
    pub embedded_service_manifest_file_sha256: String,
    pub embedded_service_manifest_canonical_json_sha256: String,
    pub files: Vec<RollbackBundleFileV2>,
    pub systemd_closure: Vec<RollbackSystemdArtifactV2>,
    pub systemd_closure_canonical_json_sha256: String,
    pub challenge: JsonEvidenceRefV2,
    pub node_probe_receipts: Vec<JsonEvidenceRefV2>,
    pub complete_path_audit: JsonEvidenceRefV2,
    pub isolated_install_simulation: JsonEvidenceRefV2,
    pub evaluator_retention: JsonEvidenceRefV2,
    pub rollback_state_compatibility: JsonEvidenceRefV2,
}

pub fn rollback_isolated_staged_tree_sha256(
    runtime_bundle_sha256: &str,
    systemd_closure: &[RollbackSystemdArtifactV2],
) -> Result<String> {
    canonical_json_sha256(&json!({
        "domain": "jope-rollback-isolated-unit-tree-v2",
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "systemd_artifacts": systemd_closure.iter().map(|artifact| json!({
            "bundled_target": artifact.bundled_target,
            "unit_file_sha256": artifact.unit_file_sha256,
            "resolved_effective_unit_sha256": artifact.resolved_effective_unit_sha256,
            "drop_ins": artifact.drop_ins,
        })).collect::<Vec<_>>(),
    }))
}

impl RollbackCapsuleV2 {
    pub fn computed_systemd_closure_sha256(&self) -> Result<String> {
        canonical_json_sha256(&json!({
            "domain": "jope-rollback-systemd-closure-v2",
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "artifacts": self.systemd_closure,
        }))
    }

    pub fn computed_isolated_staged_tree_sha256(&self) -> Result<String> {
        rollback_isolated_staged_tree_sha256(&self.runtime_bundle_sha256, &self.systemd_closure)
    }

    pub fn validate_evidence_at(&self, evidence_root: &Path, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == ROLLBACK_CAPSULE_V2_SCHEMA,
            "rollback capsule is not authorizing v2"
        );
        ensure!(
            self.generated_at_unix_ms > 0
                && self.generated_at_unix_ms <= now_unix_ms
                && self.freshness_budget_seconds > 0
                && self.freshness_budget_seconds <= 15 * 60
                && now_unix_ms - self.generated_at_unix_ms
                    <= self.freshness_budget_seconds.saturating_mul(1_000),
            "rollback capsule is future-dated or stale"
        );
        ensure!(
            self.runtime_bundle_sha256 == ROLLBACK_RUNTIME_BUNDLE_SHA256,
            "rollback runtime bundle identity drift"
        );
        for (digest, field) in [
            (
                &self.runtime_bundle_manifest_file_sha256,
                "rollback manifest file",
            ),
            (
                &self.runtime_bundle_manifest_canonical_json_sha256,
                "rollback manifest canonical JSON",
            ),
            (
                &self.embedded_service_manifest_file_sha256,
                "service manifest file",
            ),
            (
                &self.embedded_service_manifest_canonical_json_sha256,
                "service manifest canonical JSON",
            ),
            (
                &self.systemd_closure_canonical_json_sha256,
                "systemd closure canonical JSON",
            ),
        ] {
            require_sha256(digest, field)?;
        }
        ensure!(!self.files.is_empty(), "rollback file inventory is empty");
        for row in &self.files {
            row.validate()?;
        }
        ensure!(
            self.files
                .windows(2)
                .all(|pair| pair[0].target < pair[1].target),
            "rollback file inventory is not sorted and unique"
        );
        ensure!(
            self.systemd_closure.len() == 30
                && self
                    .systemd_closure
                    .iter()
                    .filter(|row| row.kind == "service")
                    .count()
                    == 23
                && self
                    .systemd_closure
                    .iter()
                    .filter(|row| row.kind == "timer")
                    .count()
                    == 7,
            "rollback systemd closure is not exactly 23 services and seven timers"
        );
        for artifact in &self.systemd_closure {
            artifact.validate()?;
            ensure!(
                self.files.iter().any(|row| {
                    row.target == artifact.bundled_target && row.sha256 == artifact.unit_file_sha256
                }),
                "rollback systemd artifact is not bound to bundled bytes: {}",
                artifact.unit
            );
        }
        ensure!(
            self.systemd_closure
                .windows(2)
                .all(|pair| pair[0].unit < pair[1].unit),
            "rollback systemd closure is not sorted and unique"
        );
        ensure!(
            self.computed_systemd_closure_sha256()? == self.systemd_closure_canonical_json_sha256,
            "rollback systemd closure digest mismatch"
        );

        let challenge: RollbackChallengeV1 = self.challenge.load(evidence_root)?;
        challenge.validate_at(now_unix_ms)?;
        ensure!(
            self.generated_at_unix_ms >= challenge.issued_at_unix_ms
                && self.generated_at_unix_ms < challenge.expires_at_unix_ms,
            "rollback capsule was not generated under its challenge"
        );
        ensure!(
            self.node_probe_receipts.len() == REQUIRED_NODES.len(),
            "rollback five-node evidence is incomplete"
        );
        let mut nodes = BTreeSet::new();
        for reference in &self.node_probe_receipts {
            let receipt: RollbackNodeProbeV2 = reference.load(evidence_root)?;
            receipt.validate(
                evidence_root,
                &challenge,
                RollbackBundleEvidence {
                    bundle_sha256: &self.runtime_bundle_sha256,
                    manifest_file_sha256: &self.runtime_bundle_manifest_file_sha256,
                    manifest_canonical_sha256: &self.runtime_bundle_manifest_canonical_json_sha256,
                    files: &self.files,
                    artifacts: &self.systemd_closure,
                },
            )?;
            ensure!(
                nodes.insert(receipt.node),
                "duplicate rollback node receipt"
            );
        }
        ensure!(
            REQUIRED_NODES.iter().all(|node| nodes.contains(*node)),
            "rollback receipt node set is incomplete"
        );

        let path_audit: RollbackPathAuditV2 = self.complete_path_audit.load(evidence_root)?;
        let artifact_names = self
            .systemd_closure
            .iter()
            .map(|row| row.unit.clone())
            .collect::<Vec<_>>();
        ensure!(
            path_audit.schema == ROLLBACK_PATH_AUDIT_SCHEMA
                && path_audit.challenge_nonce_sha256 == challenge.nonce_sha256
                && path_audit.runtime_bundle_sha256 == self.runtime_bundle_sha256
                && path_audit.audited_systemd_artifacts == artifact_names
                && path_audit.findings.is_empty(),
            "rollback path audit is incomplete, mismatched, or has findings"
        );
        path_audit.raw_output.load(evidence_root)?;

        let install: RollbackInstallSimulationV2 =
            self.isolated_install_simulation.load(evidence_root)?;
        let installer_path = install
            .installer_argv
            .first()
            .context("isolated install command has no executable")?;
        require_absolute_normalized(installer_path, "isolated installer path")?;
        let installer_relative = Path::new(installer_path)
            .strip_prefix(Path::new(immutable_flywheel_root_str()).join("runtime-bundles"))
            .context("isolated installer is outside the content-addressed runtime root")?;
        let installer_parts = installer_relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        ensure!(
            installer_parts.len() == 4
                && installer_parts[1..] == ["runtime", "bin", "jope_selfplay_bundle"],
            "isolated installer path is not an exact bundled installer"
        );
        require_sha256(&installer_parts[0], "isolated installer bundle identity")?;
        ensure!(
            install.schema == ROLLBACK_INSTALL_SIMULATION_SCHEMA
                && install.challenge_nonce_sha256 == challenge.nonce_sha256
                && install.runtime_bundle_sha256 == self.runtime_bundle_sha256
                && !install.installer_version.trim().is_empty()
                && install.installer_exit_status == 0
                && install.systemd_analyze_exit_status == 0
                && install
                    .systemd_analyze_argv
                    .iter()
                    .any(|arg| arg == "--user")
                && install
                    .systemd_analyze_argv
                    .iter()
                    .any(|arg| arg == "--root" || arg.starts_with("--root="))
                && !install.systemd_analyze_version.trim().is_empty()
                && install.staged_tree_sha256 == install.expected_staged_tree_sha256
                && install.expected_staged_tree_sha256
                    == self.computed_isolated_staged_tree_sha256()?,
            "isolated rollback installation was not genuinely executed or did not validate"
        );
        require_sha256(
            &install.installer_executable_sha256,
            "isolated installer executable",
        )?;
        require_sha256(&install.staged_tree_sha256, "isolated staged tree")?;
        install.stdout.load(evidence_root)?;
        install.stderr.load(evidence_root)?;

        let retention: RollbackEvaluatorRetentionV2 =
            self.evaluator_retention.load(evidence_root)?;
        ensure!(
            retention.schema == ROLLBACK_EVALUATOR_RETENTION_SCHEMA
                && retention.challenge_nonce_sha256 == challenge.nonce_sha256
                && retention.runtime_bundle_sha256 == self.runtime_bundle_sha256,
            "evaluator-retention receipt scope mismatch"
        );
        let retention_index = retention.retention_index.load_value(evidence_root)?;
        verify_retention_index(&retention_index, &self.runtime_bundle_sha256)?;

        let compatibility: RollbackStateCompatibilityV2 =
            self.rollback_state_compatibility.load(evidence_root)?;
        ensure!(
            compatibility.schema == ROLLBACK_STATE_COMPATIBILITY_SCHEMA
                && compatibility.challenge_nonce_sha256 == challenge.nonce_sha256
                && compatibility.runtime_bundle_sha256 == self.runtime_bundle_sha256
                && compatibility.exit_status == 0
                && !compatibility.command.is_empty()
                && compatibility.primary_documents.len() >= 2,
            "rollback-state compatibility receipt failed or is incomplete"
        );
        ensure!(
            compatibility.rollback_binaries.len() == 2,
            "rollback compatibility did not execute both primary-state readers"
        );
        let required_binaries = BTreeMap::from([
            ("cohort_proposer", "runtime/bin/jope_selfplay_supply"),
            (
                "alternating_trainer",
                "runtime/bin/jope_task_learning_controller",
            ),
        ]);
        let file_digests = self
            .files
            .iter()
            .map(|row| (row.target.as_str(), row.sha256.as_str()))
            .collect::<BTreeMap<_, _>>();
        let mut compatibility_roles = BTreeSet::new();
        for binary in &compatibility.rollback_binaries {
            let required_target = required_binaries
                .get(binary.role.as_str())
                .context("rollback compatibility binary has an unknown role")?;
            ensure!(
                compatibility_roles.insert(binary.role.as_str())
                    && binary.bundled_target == *required_target
                    && file_digests.get(binary.bundled_target.as_str())
                        == Some(&binary.sha256.as_str())
                    && compatibility
                        .command
                        .iter()
                        .any(|argument| argument == &binary.bundled_target),
                "rollback compatibility binary is not the exact bundled state reader"
            );
        }
        compatibility.stdout.load(evidence_root)?;
        compatibility.stderr.load(evidence_root)?;
        let forbidden = [
            "scientific_payload",
            "infrastructure_attempt",
            "scientific_failures",
            "foreign_cancellations",
            "retry_pending_submission",
            "proposal_submitted_at_ms",
            "next_retry_not_before_ms",
        ];
        let mut roles = BTreeSet::new();
        for row in &compatibility.primary_documents {
            ensure!(
                roles.insert(row.role.as_str()),
                "duplicate rollback state role"
            );
            let document = row.document.load_value(evidence_root)?;
            ensure!(
                document.get("schema").and_then(Value::as_str) == Some(row.schema.as_str()),
                "rollback state document schema mismatch"
            );
            let encoded = serde_json::to_string(&document)?;
            ensure!(
                forbidden
                    .iter()
                    .all(|field| !encoded.contains(&format!("\"{field}\""))),
                "rollback primary state contains successor-only recovery fields"
            );
        }
        ensure!(
            roles.contains("cohort_proposer") && roles.contains("alternating_trainer"),
            "rollback compatibility receipt omits a primary role"
        );
        Ok(())
    }

    pub fn digest_at(&self, evidence_root: &Path, now_unix_ms: u64) -> Result<String> {
        self.validate_evidence_at(evidence_root, now_unix_ms)?;
        canonical_json_sha256(&json!({
            "domain": "jope-composite-rollback-capsule-v2",
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "capsule": self,
        }))
    }
}

fn verify_retention_index(value: &Value, prepared_bundle_sha256: &str) -> Result<()> {
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("JopeEvaluatorRetentionIndex.v1")
            && value
                .get("prepared_for_bundle_sha256")
                .and_then(Value::as_str)
                == Some(prepared_bundle_sha256),
        "evaluator retention index scope mismatch"
    );
    let retained = value
        .get("retained")
        .and_then(Value::as_array)
        .context("evaluator retention index has no retained inventory")?;
    let retained_digests = retained
        .iter()
        .map(|row| {
            row.get("evaluator_sha256")
                .and_then(Value::as_str)
                .context("retained evaluator lacks digest")
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        !retained_digests.is_empty(),
        "evaluator retention inventory is empty"
    );
    let referenced = value
        .get("referenced_evaluator_sha256")
        .and_then(Value::as_array)
        .context("evaluator retention index has no reference inventory")?;
    ensure!(
        referenced.iter().all(|digest| {
            digest
                .as_str()
                .is_some_and(|digest| retained_digests.contains(digest))
        }),
        "referenced evaluator is absent from the retained inventory"
    );
    ensure!(
        value
            .get("unavailable_referenced_evaluator_sha256")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty),
        "evaluator retention index reports unavailable references"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackUnitTransformationV1 {
    pub target: String,
    pub before_sha256: String,
    pub after_sha256: String,
    pub reviewed_source_sha256: String,
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackBaselineDerivationV1 {
    pub schema: String,
    pub parent_bundle_sha256: String,
    pub derived_bundle_sha256: String,
    pub parent_manifest_file_sha256: String,
    pub parent_manifest_canonical_json_sha256: String,
    pub reviewed_source_commit: String,
    pub transformations: Vec<RollbackUnitTransformationV1>,
    pub unchanged_inventory_root_sha256: String,
}

impl RollbackBaselineDerivationV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ROLLBACK_BASELINE_DERIVATION_SCHEMA,
            "invalid rollback baseline derivation schema"
        );
        for (digest, field) in [
            (&self.parent_bundle_sha256, "derivation parent bundle"),
            (&self.derived_bundle_sha256, "derivation output bundle"),
            (
                &self.parent_manifest_file_sha256,
                "derivation parent manifest file",
            ),
            (
                &self.parent_manifest_canonical_json_sha256,
                "derivation parent manifest canonical JSON",
            ),
            (
                &self.unchanged_inventory_root_sha256,
                "derivation unchanged inventory root",
            ),
        ] {
            require_sha256(digest, field)?;
        }
        ensure!(
            self.parent_bundle_sha256
                == "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4"
                && self.reviewed_source_commit == "7e1d0429021ab11e8ec63141669f352d32eebe38",
            "rollback derivation parent or reviewed source is not exact"
        );
        ensure!(
            self.transformations.len() == 7,
            "rollback derivation must change exactly seven units"
        );
        let mut targets = BTreeSet::new();
        for row in &self.transformations {
            require_relative_normalized(&row.target, "rollback derivation target")?;
            require_sha256(&row.before_sha256, "rollback derivation before")?;
            require_sha256(&row.after_sha256, "rollback derivation after")?;
            require_sha256(
                &row.reviewed_source_sha256,
                "rollback derivation reviewed source",
            )?;
            ensure!(
                targets.insert(row.target.as_str())
                    && row.from == jope_paths::to_string(&crate::paths::jope_home())
                    && row.to == immutable_flywheel_root_str()
                    && row.before_sha256 != row.after_sha256,
                "rollback derivation transformation is duplicated or noncanonical"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod rollback_v2_tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_bytes(label.as_bytes())
    }

    fn write_json(root: &Path, name: &str, value: &Value) -> JsonEvidenceRefV2 {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut bytes = serde_json::to_vec(value).unwrap();
        bytes.push(b'\n');
        fs::write(&path, &bytes).unwrap();
        JsonEvidenceRefV2 {
            path: name.to_owned(),
            raw_file_sha256: sha256_bytes(&bytes),
            canonical_json_sha256: canonical_json_sha256(value).unwrap(),
        }
    }

    fn write_raw(root: &Path, name: &str, bytes: &[u8]) -> RawEvidenceRefV2 {
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

    fn fixture(root: &Path) -> RollbackCapsuleV2 {
        let mut closure = Vec::new();
        for index in 0..23 {
            let unit = format!("unit-{index:02}.service");
            let mut row = RollbackSystemdArtifactV2 {
                bundled_target: format!("runtime/systemd/{unit}"),
                unit,
                kind: "service".to_owned(),
                unit_file_sha256: digest(&format!("service-{index}")),
                resolved_effective_unit_sha256: String::new(),
                exec_start_argv: vec![format!(
                    "{runtime_current_root_str}/runtime/bin/service-{index}",
                    runtime_current_root_str = runtime_current_root_str()
                )],
                working_directory: Some(immutable_flywheel_root_str().to_owned()),
                drop_ins: Vec::new(),
            };
            row.resolved_effective_unit_sha256 = row.computed_effective_unit_sha256().unwrap();
            closure.push(row);
        }
        for index in 0..7 {
            let unit = format!("unit-{index:02}.timer");
            let mut row = RollbackSystemdArtifactV2 {
                bundled_target: format!("runtime/systemd/{unit}"),
                unit,
                kind: "timer".to_owned(),
                unit_file_sha256: digest(&format!("timer-{index}")),
                resolved_effective_unit_sha256: String::new(),
                exec_start_argv: Vec::new(),
                working_directory: None,
                drop_ins: Vec::new(),
            };
            row.resolved_effective_unit_sha256 = row.computed_effective_unit_sha256().unwrap();
            closure.push(row);
        }
        closure.sort_by(|left, right| left.unit.cmp(&right.unit));
        let mut files = closure
            .iter()
            .map(|row| RollbackBundleFileV2 {
                target: row.bundled_target.clone(),
                bytes: 1,
                sha256: row.unit_file_sha256.clone(),
            })
            .collect::<Vec<_>>();
        files.extend([
            RollbackBundleFileV2 {
                target: "runtime/bin/jope_selfplay_supply".to_owned(),
                bytes: 1,
                sha256: digest("rollback-supply"),
            },
            RollbackBundleFileV2 {
                target: "runtime/bin/jope_task_learning_controller".to_owned(),
                bytes: 1,
                sha256: digest("rollback-controller"),
            },
        ]);
        files.sort_by(|left, right| left.target.cmp(&right.target));
        let challenge = RollbackChallengeV1 {
            schema: ROLLBACK_CHALLENGE_SCHEMA.to_owned(),
            nonce_sha256: digest("challenge"),
            issued_at_unix_ms: 900,
            expires_at_unix_ms: 2_000,
            verifier_tool_sha256: digest("verifier"),
        };
        let challenge_ref = write_json(root, "challenge.json", &json!(challenge));
        let raw = write_raw(root, "raw/probe.txt", b"raw-probe-output");
        let manifest_file = digest("manifest-file");
        let manifest_canonical = digest("manifest-canonical");
        let mut node_probe_receipts = Vec::new();
        for node in REQUIRED_NODES {
            let receipt = RollbackNodeProbeV2 {
                schema: ROLLBACK_NODE_PROBE_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                node: node.to_owned(),
                observed_hostname: format!("{node}.cluster"),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                bundle_manifest_file_sha256: manifest_file.clone(),
                bundle_manifest_canonical_json_sha256: manifest_canonical.clone(),
                files: files.clone(),
                systemd_artifacts: closure.clone(),
                tool: RollbackProbeToolV2 {
                    executable_sha256: digest("probe-tool"),
                    version: "probe 1".to_owned(),
                    argv: vec!["probe".to_owned(), challenge.nonce_sha256.clone()],
                },
                started_at_unix_ms: 950,
                finished_at_unix_ms: 960,
                raw_output: raw.clone(),
            };
            node_probe_receipts.push(write_json(
                root,
                &format!("nodes/{node}.json"),
                &json!(receipt),
            ));
        }
        let artifact_names = closure.iter().map(|row| row.unit.clone()).collect();
        let path_audit = write_json(
            root,
            "path-audit.json",
            &json!(RollbackPathAuditV2 {
                schema: ROLLBACK_PATH_AUDIT_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                audited_systemd_artifacts: artifact_names,
                findings: Vec::new(),
                raw_output: raw.clone(),
            }),
        );
        let staged_tree_sha256 =
            rollback_isolated_staged_tree_sha256(ROLLBACK_RUNTIME_BUNDLE_SHA256, &closure).unwrap();
        let install = write_json(
            root,
            "install.json",
            &json!(RollbackInstallSimulationV2 {
                schema: ROLLBACK_INSTALL_SIMULATION_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                installer_argv: vec![
                    format!(
                        "{immutable_flywheel_root_str}/runtime-bundles/{ROLLBACK_RUNTIME_BUNDLE_SHA256}/runtime/bin/jope_selfplay_bundle", immutable_flywheel_root_str = immutable_flywheel_root_str()
                    ),
                    "--install-units".to_owned()
                ],
                installer_executable_sha256: digest("installer"),
                installer_version: "jope-selfplay-bundle 2".to_owned(),
                installer_exit_status: 0,
                systemd_analyze_argv: vec![
                    "systemd-analyze".to_owned(),
                    "--user".to_owned(),
                    "--root".to_owned(),
                    "/isolated".to_owned(),
                ],
                systemd_analyze_version: "systemd 257".to_owned(),
                systemd_analyze_exit_status: 0,
                staged_tree_sha256: staged_tree_sha256.clone(),
                expected_staged_tree_sha256: staged_tree_sha256,
                stdout: raw.clone(),
                stderr: raw.clone(),
            }),
        );
        let retention_index = write_json(
            root,
            "retention-index.json",
            &json!({
                "schema": "JopeEvaluatorRetentionIndex.v1",
                "active_bundle_sha256": ROLLBACK_RUNTIME_BUNDLE_SHA256,
                "prepared_for_bundle_sha256": ROLLBACK_RUNTIME_BUNDLE_SHA256,
                "retained": [{"evaluator_sha256": digest("eval")}],
                "referenced_evaluator_sha256": [digest("eval")],
                "unavailable_referenced_evaluator_sha256": [],
            }),
        );
        let retention = write_json(
            root,
            "retention.json",
            &json!(RollbackEvaluatorRetentionV2 {
                schema: ROLLBACK_EVALUATOR_RETENTION_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                retention_index,
            }),
        );
        let proposer = write_json(
            root,
            "states/proposer.json",
            &json!({"schema": "CohortSupplyState.v2", "proposal_job_id": "p"}),
        );
        let trainer = write_json(
            root,
            "states/trainer.json",
            &json!({"schema": "TaskLearningControllerState.v2", "intent": null}),
        );
        let compatibility = write_json(
            root,
            "compatibility.json",
            &json!(RollbackStateCompatibilityV2 {
                schema: ROLLBACK_STATE_COMPATIBILITY_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256,
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                rollback_binaries: vec![
                    RollbackStateBinaryV2 {
                        role: "cohort_proposer".to_owned(),
                        bundled_target: "runtime/bin/jope_selfplay_supply".to_owned(),
                        sha256: digest("rollback-supply"),
                    },
                    RollbackStateBinaryV2 {
                        role: "alternating_trainer".to_owned(),
                        bundled_target: "runtime/bin/jope_task_learning_controller".to_owned(),
                        sha256: digest("rollback-controller"),
                    },
                ],
                command: vec![
                    "runtime/bin/jope_selfplay_supply".to_owned(),
                    "runtime/bin/jope_task_learning_controller".to_owned(),
                    "--read-state".to_owned(),
                ],
                exit_status: 0,
                stdout: raw.clone(),
                stderr: raw,
                primary_documents: vec![
                    RollbackStateDocumentV2 {
                        role: "cohort_proposer".to_owned(),
                        schema: "CohortSupplyState.v2".to_owned(),
                        document: proposer,
                    },
                    RollbackStateDocumentV2 {
                        role: "alternating_trainer".to_owned(),
                        schema: "TaskLearningControllerState.v2".to_owned(),
                        document: trainer,
                    },
                ],
            }),
        );
        let mut capsule = RollbackCapsuleV2 {
            schema: ROLLBACK_CAPSULE_V2_SCHEMA.to_owned(),
            generated_at_unix_ms: 1_000,
            freshness_budget_seconds: 1,
            runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
            runtime_bundle_manifest_file_sha256: manifest_file,
            runtime_bundle_manifest_canonical_json_sha256: manifest_canonical,
            embedded_service_manifest_file_sha256: digest("service-file"),
            embedded_service_manifest_canonical_json_sha256: digest("service-canonical"),
            files,
            systemd_closure: closure,
            systemd_closure_canonical_json_sha256: String::new(),
            challenge: challenge_ref,
            node_probe_receipts,
            complete_path_audit: path_audit,
            isolated_install_simulation: install,
            evaluator_retention: retention,
            rollback_state_compatibility: compatibility,
        };
        capsule.systemd_closure_canonical_json_sha256 =
            capsule.computed_systemd_closure_sha256().unwrap();
        capsule
    }

    #[test]
    fn v2_derives_outcomes_from_complete_content_addressed_evidence() {
        let directory = tempfile::tempdir().unwrap();
        fixture(directory.path())
            .validate_evidence_at(directory.path(), 1_000)
            .unwrap();
    }

    #[test]
    fn v1_and_fake_pass_boolean_cannot_authorize() {
        let historical = json!({"schema": ROLLBACK_CAPSULE_SCHEMA});
        assert_eq!(
            classify_rollback_capsule_schema(&historical).unwrap(),
            "historical-non-authorizing-v1"
        );
        assert!(serde_json::from_value::<RollbackCapsuleV2>(historical).is_err());
        let directory = tempfile::tempdir().unwrap();
        let mut value = serde_json::to_value(fixture(directory.path())).unwrap();
        value["simulated_install_passed"] = Value::Bool(true);
        assert!(serde_json::from_value::<RollbackCapsuleV2>(value).is_err());
    }

    #[test]
    fn missing_altered_stale_duplicate_and_incomplete_evidence_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let capsule = fixture(directory.path());
        fs::remove_file(directory.path().join(&capsule.node_probe_receipts[0].path)).unwrap();
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let capsule = fixture(directory.path());
        fs::write(
            directory.path().join(&capsule.node_probe_receipts[0].path),
            b"{}\n",
        )
        .unwrap();
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let capsule = fixture(directory.path());
        assert!(capsule
            .validate_evidence_at(directory.path(), 2_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        let first: RollbackNodeProbeV2 = capsule.node_probe_receipts[0]
            .load(directory.path())
            .unwrap();
        let mut duplicate: RollbackNodeProbeV2 = capsule.node_probe_receipts[1]
            .load(directory.path())
            .unwrap();
        duplicate.node = first.node;
        duplicate.observed_hostname = first.observed_hostname;
        capsule.node_probe_receipts[1] =
            write_json(directory.path(), "nodes/duplicate.json", &json!(duplicate));
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        let mut incomplete: RollbackNodeProbeV2 = capsule.node_probe_receipts[0]
            .load(directory.path())
            .unwrap();
        incomplete.files.pop();
        capsule.node_probe_receipts[0] = write_json(
            directory.path(),
            "nodes/incomplete.json",
            &json!(incomplete),
        );
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());
    }

    #[test]
    fn overlay_and_raw_vs_canonical_digest_substitution_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        capsule.systemd_closure[0].unit_file_sha256 = digest("altered-unit");
        capsule.systemd_closure_canonical_json_sha256 =
            capsule.computed_systemd_closure_sha256().unwrap();
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        std::mem::swap(
            &mut capsule.challenge.raw_file_sha256,
            &mut capsule.challenge.canonical_json_sha256,
        );
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());
    }
}
