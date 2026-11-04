use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ParentLink {
    pub(crate) run_id: String,
    pub(crate) manifest_path: PathBuf,
    pub(crate) manifest_sha256: String,
    pub(crate) validation_path: PathBuf,
    pub(crate) validation_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ProposalBundleLink {
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
    pub(crate) exact_rows_per_dataset: usize,
    pub(crate) jittered_rows_per_dataset: usize,
    pub(crate) scaling_statistic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DatasetInput {
    pub(crate) dataset_hash: String,
    pub(crate) train_path: PathBuf,
    pub(crate) train_sha256: String,
    pub(crate) rows: usize,
    pub(crate) features: usize,
    pub(crate) context_sha256: String,
    pub(crate) transfer_ranking_path: PathBuf,
    pub(crate) transfer_ranking_sha256: String,
    pub(crate) parent_generation_paths: Vec<PathBuf>,
    pub(crate) parent_generation_sha256: Vec<String>,
    pub(crate) fold_bank_path: PathBuf,
    pub(crate) fold_bank_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TrainOnlyRedteamManifest {
    pub(crate) schema: String,
    pub(crate) run_id: String,
    pub(crate) config: TrainOnlyRedteamConfig,
    pub(crate) config_sha256: String,
    pub(crate) development_only: bool,
    pub(crate) confirmatory_valid: bool,
    pub(crate) test_reuse_disclosure: String,
    pub(crate) parent: ParentLink,
    pub(crate) proposal_bundle: ProposalBundleLink,
    pub(crate) scaling_statistic_path: PathBuf,
    pub(crate) scaling_statistic_sha256: String,
    pub(crate) datasets: Vec<DatasetInput>,
    pub(crate) firewall: SelectionFirewall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SelectionFirewall {
    pub(crate) selection_workers_receive_training_only: bool,
    pub(crate) contains_test_paths: bool,
    pub(crate) contains_baseline_paths: bool,
    pub(crate) test_paths_constructed: bool,
    pub(crate) baseline_paths_constructed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ScqExecutionProvenance {
    pub expected_node: String,
    pub scheduler_reported_node: String,
    pub scq_job_id: String,
    pub scq_attempt: usize,
    pub release_binary_sha256: String,
    pub source_bundle_sha256: String,
    pub manifest_sha256: String,
    pub proposal_bundle_sha256: String,
    pub fold_bank_sha256: String,
    pub parent_input_sha256: BTreeMap<String, String>,
    pub output_package_sha256: String,
    pub node_marker_probe: String,
    pub workers: usize,
    pub cpus_reserved: usize,
    pub ram_gb: usize,
    pub non_preemptible: bool,
    pub max_runtime_hours: usize,
    pub imported_without_overwrite: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ScalingStatistic {
    pub(crate) schema: String,
    pub(crate) source_catalog_sha256: String,
    pub(crate) contexts: usize,
    pub(crate) dimensions: usize,
    pub(crate) population_sd: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FoldBankReceipt {
    pub(crate) schema: String,
    pub(crate) run_id: String,
    pub(crate) dataset_hash: String,
    pub(crate) train_sha256: String,
    pub(crate) rows: usize,
    pub(crate) folds: usize,
    pub(crate) search: Vec<FoldRepeat>,
    pub(crate) selector: Vec<FoldRepeat>,
    pub(crate) audit: Vec<FoldRepeat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FoldRepeat {
    pub(crate) purpose: String,
    pub(crate) repeat: usize,
    pub(crate) seed: u64,
    pub(crate) validation_indices: Vec<Vec<usize>>,
    pub(crate) repeat_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ProposalRequest {
    pub(crate) schema: String,
    pub(crate) run_id: String,
    pub(crate) parent_manifest_sha256: String,
    pub(crate) checkpoint_path: PathBuf,
    pub(crate) checkpoint_sha256: String,
    pub(crate) catalog_pointer_path: PathBuf,
    pub(crate) proposal_dataset_manifest_path: PathBuf,
    pub(crate) scaling_statistic_path: PathBuf,
    pub(crate) scaling_statistic_sha256: String,
    pub(crate) exact_decoder_seeds: Vec<u64>,
    pub(crate) jitter_decoder_seeds: Vec<u64>,
    pub(crate) perturbation_seeds: Vec<u64>,
    pub(crate) proposals_per_invocation: usize,
    pub(crate) expected_combined_path: PathBuf,
    pub(crate) development_only: bool,
    pub(crate) confirmatory_valid: bool,
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(crate) fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        other => other,
    }
}

pub(crate) fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&canonicalize_json(value.clone()))?)
}

pub(crate) fn receipt_digest<T: Serialize>(receipt: &T) -> Result<String> {
    Ok(sha256_bytes(&canonical_json_bytes(&serde_json::to_value(
        receipt,
    )?)?))
}

pub(crate) fn scq_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

pub(crate) fn scq_execution_provenance(
    manifest: &TrainOnlyRedteamManifest,
    dataset: Option<&DatasetInput>,
    workers: usize,
) -> Option<ScqExecutionProvenance> {
    let job_id = scq_env("REDTEAM_SCQ_JOB_ID")?;
    let parse = |name: &str, default: usize| {
        scq_env(name)
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let mut parent_input_sha256 = BTreeMap::from([
        (
            "parent_manifest".to_string(),
            manifest.parent.manifest_sha256.clone(),
        ),
        (
            "parent_validation".to_string(),
            manifest.parent.validation_sha256.clone(),
        ),
    ]);
    let fold_bank_sha256 = dataset
        .map(|dataset| {
            for (generation, digest) in dataset.parent_generation_sha256.iter().enumerate() {
                parent_input_sha256
                    .insert(format!("parent_generation_{generation:03}"), digest.clone());
            }
            scq_env("REDTEAM_FOLD_BANK_SHA256").unwrap_or_else(|| dataset.fold_bank_sha256.clone())
        })
        .unwrap_or_default();
    if let Some(encoded) = scq_env("REDTEAM_PARENT_INPUT_SHA256") {
        if let Ok(captured) = serde_json::from_str::<BTreeMap<String, String>>(&encoded) {
            parent_input_sha256 = captured;
        }
    }
    Some(ScqExecutionProvenance {
        expected_node: scq_env("REDTEAM_EXPECTED_NODE").unwrap_or_default(),
        scheduler_reported_node: scq_env("REDTEAM_SCHEDULER_NODE")
            .unwrap_or_else(|| "unknown".to_string()),
        scq_job_id: job_id,
        scq_attempt: parse("REDTEAM_SCQ_ATTEMPT", 0),
        release_binary_sha256: scq_env("REDTEAM_RELEASE_BINARY_SHA256").unwrap_or_default(),
        source_bundle_sha256: scq_env("REDTEAM_SOURCE_BUNDLE_SHA256").unwrap_or_default(),
        manifest_sha256: scq_env("REDTEAM_MANIFEST_SHA256")
            .unwrap_or_else(|| receipt_digest(manifest).unwrap_or_default()),
        proposal_bundle_sha256: scq_env("REDTEAM_PROPOSAL_BUNDLE_SHA256")
            .unwrap_or_else(|| manifest.proposal_bundle.sha256.clone()),
        fold_bank_sha256,
        parent_input_sha256,
        output_package_sha256: scq_env("REDTEAM_OUTPUT_PACKAGE_SHA256")
            .unwrap_or_else(|| "not-captured".to_string()),
        node_marker_probe: scq_env("REDTEAM_NODE_MARKER_PROBE").unwrap_or_default(),
        workers: parse("REDTEAM_WORKERS", workers),
        cpus_reserved: parse("REDTEAM_CPUS_RESERVED", 8),
        ram_gb: parse("REDTEAM_RAM_GB", 40),
        non_preemptible: scq_env("REDTEAM_NON_PREEMPTIBLE").is_none_or(|value| value == "true"),
        max_runtime_hours: parse("REDTEAM_MAX_RUNTIME_HOURS", 24),
        imported_without_overwrite: scq_env("REDTEAM_IMPORTED_WITHOUT_OVERWRITE")
            .is_some_and(|value| value == "true"),
    })
}

pub(crate) fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("immutable artifact has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(());
        }
        bail!(
            "immutable artifact already exists with different bytes: {}",
            path.display()
        );
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(crate) fn write_receipt<T: Serialize>(path: &Path, receipt: &T) -> Result<String> {
    let mut value = serde_json::to_value(receipt)?;
    let digest = sha256_bytes(&canonical_json_bytes(&value)?);
    value
        .as_object_mut()
        .context("receipt must be a JSON object")?
        .insert("receipt_sha256".to_string(), Value::String(digest.clone()));
    let mut bytes = canonical_json_bytes(&value)?;
    bytes.push(b'\n');
    immutable_write(path, &bytes)?;
    Ok(digest)
}

pub(crate) fn read_receipt<T: DeserializeOwned>(path: &Path) -> Result<(T, String)> {
    let mut value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let claimed = value
        .as_object_mut()
        .and_then(|object| object.remove("receipt_sha256"))
        .and_then(|value| value.as_str().map(str::to_string))
        .context("receipt lacks receipt_sha256")?;
    let actual = sha256_bytes(&canonical_json_bytes(&value)?);
    if claimed != actual {
        bail!("receipt digest mismatch: {}", path.display());
    }
    Ok((serde_json::from_value(value)?, actual))
}

pub(crate) fn absolute(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    if let Some(root) = std::env::var_os("REDTEAM_PATH_ROOT").map(PathBuf::from) {
        if let Ok(relative) = absolute.strip_prefix(root) {
            return Ok(Path::new("/").join(relative));
        }
    }
    Ok(absolute)
}

pub(crate) fn run_root(study_root: &Path, run_id: &str) -> Result<PathBuf> {
    if run_id.is_empty()
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("run id contains unsupported characters");
    }
    Ok(study_root.join(run_id))
}

pub(crate) fn value_string(value: &Value, pointer: &str) -> Result<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("missing string at {pointer}"))
}

pub(crate) fn value_usize(value: &Value, pointer: &str) -> Result<usize> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("missing integer at {pointer}"))
}
