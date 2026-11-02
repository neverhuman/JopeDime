use super::prelude::*;
use super::{
    cache::copy_reflink_or_copy,
    inventory::valid_dataset_id,
    live::SIDECAR_NAMES,
    model::{corpus_dir_name, load_v1_quarantine, v1_identity_denied, ExpansionPaths, TASKS},
};

pub(super) fn validate_sidecar(path: &Path, name: &str) -> Result<String> {
    let rows = match name {
        "hyperion3train128dim.vs" => read_vs_matrix(path, Some(128))?,
        _ => read_vs_matrix(path, None)?,
    };
    if name == "train_target.vs" {
        ensure!(
            rows.len() == 1 && rows[0].len() == TARGET_DIM,
            "{}: train_target.vs must be exactly 1x{TARGET_DIM}",
            path.display()
        );
    }
    sha256_file(path)
}

/// Additively install encoding sidecars into the authoritative registry.
///
/// Existing registry files are never overwritten: identical content is skipped,
/// differing content fails closed. Every action appends a sha receipt line.
pub fn install_sidecars(
    paths: &ExpansionPaths,
    from: &Path,
    task: &str,
    receipts: &Path,
) -> Result<Value> {
    ensure!(TASKS.contains(&task), "unsupported task: {task}");
    let quarantine = load_v1_quarantine(paths)?;
    if let Some(parent) = receipts.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(receipts)?;
    fn record(
        ledger: &mut File,
        task: &str,
        dataset: &str,
        file: &str,
        digest: &str,
        action: &str,
    ) -> Result<()> {
        serde_json::to_writer(
            &mut *ledger,
            &json!({
                "schema": "JopeSidecarInstallReceipt.v1",
                "task": task,
                "dataset_id": dataset,
                "file": file,
                "sha256": digest,
                "action": action,
                "at_ms": now_ms(),
            }),
        )?;
        ledger.write_all(b"\n")?;
        Ok(())
    }
    let mut directories = fs::read_dir(from)
        .with_context(|| format!("read sidecar source {}", from.display()))?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter(|entry| valid_dataset_id(&entry.file_name().to_string_lossy()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    directories.sort();
    let mut installed = 0_u64;
    let mut skipped = 0_u64;
    let mut missing_registry = 0_u64;
    for directory in &directories {
        let dataset_id = directory
            .file_name()
            .and_then(|value| value.to_str())
            .context("sidecar source has no UTF-8 identity")?
            .to_owned();
        if v1_identity_denied(quarantine.as_ref(), &dataset_id) {
            record(
                &mut ledger,
                task,
                &dataset_id,
                "-",
                "-",
                "quarantine_denied",
            )?;
            continue;
        }
        let registry = paths
            .source_root
            .join(corpus_dir_name(task))
            .join(&dataset_id);
        if !registry.is_dir() {
            missing_registry += 1;
            record(
                &mut ledger,
                task,
                &dataset_id,
                "-",
                "-",
                "missing_registry_dir",
            )?;
            continue;
        }
        for name in SIDECAR_NAMES {
            let source = directory.join(name);
            if !source.is_file() {
                continue;
            }
            let source_sha256 = validate_sidecar(&source, name)?;
            let destination = registry.join(name);
            if destination.is_file() {
                let existing = sha256_file(&destination)?;
                if existing == source_sha256 {
                    skipped += 1;
                    record(
                        &mut ledger,
                        task,
                        &dataset_id,
                        name,
                        &existing,
                        "skipped_identical",
                    )?;
                    continue;
                }
                record(&mut ledger, task, &dataset_id, name, &existing, "conflict")?;
                ledger.sync_all()?;
                bail!(
                    "{task}:{dataset_id}: refusing to overwrite {name} (registry {existing} != source {source_sha256})"
                );
            }
            copy_reflink_or_copy(&source, &destination)?;
            ensure!(
                sha256_file(&destination)? == source_sha256,
                "{task}:{dataset_id}: installed sidecar hash drift for {name}"
            );
            installed += 1;
            record(
                &mut ledger,
                task,
                &dataset_id,
                name,
                &source_sha256,
                "installed",
            )?;
        }
    }
    ledger.sync_all()?;
    Ok(json!({
        "schema": "JopeSidecarInstallResult.v1",
        "task": task,
        "datasets": directories.len(),
        "installed": installed,
        "skipped_identical": skipped,
        "missing_registry_dir": missing_registry,
        "receipts": receipts,
    }))
}
