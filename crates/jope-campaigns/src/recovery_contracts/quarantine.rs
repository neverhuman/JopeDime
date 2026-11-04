use super::*;

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
