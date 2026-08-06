//! Fail-closed recovery authorities for the Phase 5.1 cutover.
//!
//! The types in this module are deliberately independent of the active
//! runtime pointer.  In particular, a rollback capsule binds an unchanged
//! runtime bundle and a separately content-addressed unit overlay.

use crate::encoder_v2_identity::ArtifactIdentity;
use crate::selfplay_data::canonical_json_sha256;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path};

pub const V1_QUARANTINE_SCHEMA: &str = "JopeV1EncodingQuarantine.v1";
pub const ROLLBACK_CAPSULE_SCHEMA: &str = "JopeRollbackCapsule.v1";
pub const RECOVERY_CONTINUITY_OBSERVATION_SCHEMA: &str = "JopeRecoveryContinuityObservation.v1";
pub const ROLLBACK_RUNTIME_BUNDLE_SHA256: &str =
    "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4";
pub const IMMUTABLE_FLYWHEEL_ROOT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0";
pub const RUNTIME_CURRENT_ROOT: &str =
    "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current";
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
            executable.starts_with(&format!("{RUNTIME_CURRENT_ROOT}/runtime/bin/")),
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
                    argument.starts_with(&format!("{RUNTIME_CURRENT_ROOT}/pytorch/")),
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
            self.working_directory == IMMUTABLE_FLYWHEEL_ROOT,
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
