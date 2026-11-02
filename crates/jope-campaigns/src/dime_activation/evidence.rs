use super::*;

pub(super) fn snapshot_protected_science(root: &Path) -> Result<BTreeMap<String, String>> {
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

pub(super) fn write_transaction(path: &Path, value: &DimeActivationTransactionV1) -> Result<()> {
    ensure!(
        value.schema == ACTIVATION_TRANSACTION_SCHEMA,
        "invalid activation transaction schema"
    );
    atomic_json(path, &serde_json::to_value(value)?)
}

pub(super) fn atomic_json(path: &Path, value: &Value) -> Result<()> {
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

pub(super) fn store_raw(root: &Path, relative: &str, bytes: &[u8]) -> Result<RawEvidenceRefV2> {
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

pub(super) fn restore_raw_optional(
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

pub(super) fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
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

pub(super) fn validate_unit_name(unit: &str) -> Result<()> {
    ensure!(
        !unit.contains('/') && (unit.ends_with(".service") || unit.ends_with(".timer")),
        "invalid DIME unit name"
    );
    Ok(())
}

pub(super) fn validate_relative(value: &str, field: &str) -> Result<()> {
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

pub(super) fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} is not a lowercase SHA-256"
    );
    Ok(())
}

pub(super) fn require_commit(value: &str) -> Result<()> {
    ensure!(
        value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "integration commit is not a full Git object identity"
    );
    Ok(())
}
