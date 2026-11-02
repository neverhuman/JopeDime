use super::*;

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
