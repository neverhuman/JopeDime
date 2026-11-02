use super::*;

pub(super) fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("runs")
}

pub(super) fn original_root() -> PathBuf {
    manifest_dir().join(ORIGINAL_PANEL_ID)
}

pub(super) fn follow_up_root() -> PathBuf {
    manifest_dir().join(FOLLOW_UP_PANEL_ID)
}

pub(super) fn replacement_root() -> PathBuf {
    follow_up_root().join("replacement")
}

pub(super) fn original_ledger_path() -> PathBuf {
    original_root().join(ORIGINAL_LEDGER_FILE)
}

pub(super) fn public_target(registry: &Path, entry: &TargetEntry) -> Result<PathBuf> {
    match (entry.kind.as_str(), entry.dataset_hash.as_deref()) {
        ("prediction", Some(hash)) if LOSS_DATASETS.contains(&hash) => Ok(registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE)),
        ("report", Some(hash)) if LOSS_DATASETS.contains(&hash) => {
            Ok(registry.join(hash).join("reports").join(REPORT_FILE))
        }
        ("panel_ledger", None) => Ok(original_ledger_path()),
        _ => bail!("replacement manifest contains a target outside the allowlist"),
    }
}

pub(super) fn untouched_target(registry: &Path, entry: &UntouchedEntry) -> Result<PathBuf> {
    if LOSS_DATASETS.contains(&entry.dataset_hash.as_str())
        || !DATASETS.contains(&entry.dataset_hash.as_str())
    {
        bail!("untouched-winner manifest contains an invalid dataset");
    }
    match entry.kind.as_str() {
        "prediction" => Ok(registry
            .join(&entry.dataset_hash)
            .join("predictions")
            .join(PREDICTION_FILE)),
        "report" => Ok(registry
            .join(&entry.dataset_hash)
            .join("reports")
            .join(REPORT_FILE)),
        _ => bail!("untouched-winner manifest contains an invalid target kind"),
    }
}

pub(super) fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} is not a lowercase SHA-256 digest");
    }
    if value != value.to_ascii_lowercase() {
        bail!("{label} is not a lowercase SHA-256 digest");
    }
    Ok(())
}

pub(super) fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("replacement path has no parent")?;
    fs::create_dir_all(parent)?;
    let staged = parent.join(format!(
        ".jope-dime-replace-{}-{}",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    if staged.exists() {
        bail!(
            "stale replacement staging file exists: {}",
            staged.display()
        );
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(&staged)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&staged, permissions)?;
    fs::rename(&staged, path).with_context(|| format!("replace {}", path.display()))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn replacement_complete_path() -> PathBuf {
    replacement_root().join("OFFICIAL_REPLACEMENT_COMPLETE")
}

pub(super) fn authorized_source_ledger_bytes() -> Result<Vec<u8>> {
    let current = fs::read(original_ledger_path()).context("read current panel ledger")?;
    if sha256_bytes(&current) == ORIGINAL_LEDGER_SHA256 {
        return Ok(current);
    }
    let pointer_path = replacement_root().join("ARCHIVE_COMPLETE");
    let pointer: ArchivePointer = serde_json::from_slice(
        &fs::read(&pointer_path)
            .context("current panel ledger drifted before a complete archive")?,
    )?;
    validate_archive_pointer(&pointer)?;
    let archive = replacement_root()
        .join("archives")
        .join(&pointer.archive_manifest_sha256);
    let bytes = fs::read(archive.join("payload/panel-ledger.json"))?;
    if sha256_bytes(&bytes) != ORIGINAL_LEDGER_SHA256 {
        bail!("archived source panel ledger differs from the pinned SHA-256");
    }
    let stage = load_stage()?;
    let ledger = stage
        .targets
        .iter()
        .find(|entry| entry.kind == "panel_ledger")
        .context("stage manifest has no panel ledger")?;
    if sha256_bytes(&current) != ledger.new_sha256 {
        bail!("current panel ledger is neither the archived old hash nor staged new hash");
    }
    Ok(bytes)
}

pub(super) fn validate_source_hash_only() -> Result<()> {
    if sha256_bytes(&authorized_source_ledger_bytes()?) != ORIGINAL_LEDGER_SHA256 {
        bail!("source panel ledger SHA-256 drift");
    }
    Ok(())
}

pub(super) fn validate_compiled_cohort(source_ledger: &[u8]) -> Result<Value> {
    let value: Value = serde_json::from_slice(source_ledger)?;
    if value["schema"] != "JopeDimePanelResults.v1"
        || value["dataset_count"].as_u64() != Some(42)
        || value["outcomes"].as_array().map(Vec::len) != Some(42)
    {
        bail!("pinned source ledger is not the original complete panel");
    }
    let losses = value["outcomes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["strict_overall_outcome"] == "loss")
        .map(|row| {
            row["dataset_hash"]
                .as_str()
                .context("source outcome has no dataset hash")
        })
        .collect::<Result<Vec<_>>>()?;
    if losses != LOSS_DATASETS {
        bail!("compiled 35-dataset cohort differs from the pinned source ledger");
    }
    Ok(value)
}

pub(super) fn directory_digest(root: &Path) -> Result<String> {
    fn visit(root: &Path, current: &Path, rows: &mut Vec<String>) -> Result<()> {
        let mut entries = fs::read_dir(current)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                visit(root, &path, rows)?;
            } else if path.is_file() {
                rows.push(format!(
                    "{} {}",
                    path.strip_prefix(root)?.display(),
                    hash_file(&path)?
                ));
            }
        }
        Ok(())
    }
    let mut rows = Vec::new();
    visit(root, root, &mut rows)?;
    Ok(sha256_bytes(format!("{}\n", rows.join("\n")).as_bytes()))
}

pub(super) fn recover_incomplete_dataset(root: &Path, hash: &str) -> Result<()> {
    if !root.exists() || fs::read_dir(root)?.next().transpose()?.is_none() {
        return Ok(());
    }
    let digest = directory_digest(root)?;
    let recovery = follow_up_root().join("recovery");
    fs::create_dir_all(&recovery)?;
    let destination = recovery.join(format!("{hash}-{}", &digest[..16]));
    if destination.exists() {
        bail!(
            "incomplete recovery destination already exists: {}",
            destination.display()
        );
    }
    fs::rename(root, &destination)?;
    File::open(&recovery)?.sync_all()?;
    eprintln!(
        "[jope-dime] preserved incomplete {hash} attempt at {}",
        destination.display()
    );
    Ok(())
}

pub(super) fn load_validated_follow_up(
    registry: &Path,
) -> Result<BTreeMap<String, ValidatedFreeze>> {
    validate_static_inputs()?;
    validate_source_hash_only()?;
    let mut freezes = BTreeMap::new();
    let mut semantics = BTreeSet::new();
    let mut slots = 0usize;
    let mut finalists = 0usize;
    for hash in LOSS_DATASETS {
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let artifact = follow_up_root().join(hash);
        let validated = exact_cv::validate_freeze(
            &artifact,
            hash,
            &train_sha256,
            train.y.len(),
            dataset_seed(hash)?,
            FOLLOW_UP_GENERATIONS,
        )
        .with_context(|| format!("30-generation freeze validation failed for {hash}"))?;
        exact_cv::validate_original_prefix(&original_root().join(hash), &artifact)
            .with_context(|| format!("10-generation replay prefix mismatch for {hash}"))?;
        if !semantics.insert(validated.semantic_sha256.clone()) {
            bail!("follow-up selected semantic genome is reused across datasets");
        }
        slots += validated.evolutionary_slots;
        finalists += validated.finalist_receipts;
        freezes.insert((*hash).to_string(), validated);
    }
    if freezes.len() != LOSS_DATASETS.len()
        || semantics.len() != LOSS_DATASETS.len()
        || slots != FOLLOW_UP_SELECTION_SLOTS
        || finalists != FOLLOW_UP_FINALISTS
    {
        bail!(
            "follow-up accounting failed: freezes={} distinct={} slots={slots} finalists={finalists}",
            freezes.len(),
            semantics.len()
        );
    }
    Ok(freezes)
}
