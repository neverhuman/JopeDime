use super::evidence::RollbackBundleEvidence;
use super::*;

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
