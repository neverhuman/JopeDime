use super::*;

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

pub(super) struct RollbackBundleEvidence<'a> {
    pub(super) bundle_sha256: &'a str,
    pub(super) manifest_file_sha256: &'a str,
    pub(super) manifest_canonical_sha256: &'a str,
    pub(super) files: &'a [RollbackBundleFileV2],
    pub(super) artifacts: &'a [RollbackSystemdArtifactV2],
}

impl RollbackNodeProbeV2 {
    pub(super) fn validate(
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
