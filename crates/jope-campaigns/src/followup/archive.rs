use super::*;

pub(super) fn archive_manifest(stage: &StageManifest) -> ArchiveManifest {
    ArchiveManifest {
        schema: "JopeDimeFollowUpArchive.v1".to_string(),
        source_panel_ledger_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        entries: stage
            .targets
            .iter()
            .map(|entry| ArchiveEntry {
                kind: entry.kind.clone(),
                dataset_hash: entry.dataset_hash.clone(),
                archive_file: entry.archive_file.clone(),
                sha256: entry.old_sha256.clone(),
            })
            .collect(),
    }
}

pub(super) fn validate_archive_pointer(pointer: &ArchivePointer) -> Result<()> {
    if pointer.schema != "JopeDimeFollowUpArchivePointer.v1" || pointer.status != "COMPLETE" {
        bail!("follow-up archive pointer is invalid");
    }
    validate_sha256("archive manifest SHA-256", &pointer.archive_manifest_sha256)
}

pub(super) fn validate_archive(stage: &StageManifest, pointer: &ArchivePointer) -> Result<PathBuf> {
    validate_archive_pointer(pointer)?;
    let root = replacement_root()
        .join("archives")
        .join(&pointer.archive_manifest_sha256);
    let marker: CompletionMarker =
        serde_json::from_slice(&fs::read(root.join("ARCHIVE_COMPLETE"))?)?;
    if marker.schema != "JopeDimeFollowUpArchiveComplete.v1"
        || marker.status != "COMPLETE"
        || marker.manifest_sha256 != pointer.archive_manifest_sha256
    {
        bail!("content-addressed archive completion marker is invalid");
    }
    let manifest_bytes = fs::read(root.join("manifest.v1.json"))?;
    if sha256_bytes(&manifest_bytes) != pointer.archive_manifest_sha256 {
        bail!("content-addressed archive manifest hash mismatch");
    }
    let manifest: ArchiveManifest = serde_json::from_slice(&manifest_bytes)?;
    if canonical_json(&serde_json::to_value(&manifest)?)?
        != canonical_json(&serde_json::to_value(archive_manifest(stage))?)?
    {
        bail!("content-addressed archive does not match the staged replacement");
    }
    for entry in &manifest.entries {
        if hash_file(&root.join(&entry.archive_file))? != entry.sha256 {
            bail!("archived payload hash mismatch for {}", entry.archive_file);
        }
    }
    Ok(root)
}

pub(super) fn ensure_archive(
    registry: &Path,
    stage: &StageManifest,
) -> Result<(ArchivePointer, PathBuf)> {
    let pointer_path = replacement_root().join("ARCHIVE_COMPLETE");
    if pointer_path.exists() {
        let pointer: ArchivePointer = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        let root = validate_archive(stage, &pointer)?;
        return Ok((pointer, root));
    }
    for entry in &stage.targets {
        let target = public_target(registry, entry)?;
        if hash_file(&target)? != entry.old_sha256 {
            bail!("cannot archive {} after it changed", target.display());
        }
    }
    let manifest = archive_manifest(stage);
    let manifest_bytes = canonical_json(&serde_json::to_value(&manifest)?)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let archive_parent = replacement_root().join("archives");
    fs::create_dir_all(&archive_parent)?;
    let final_root = archive_parent.join(&manifest_sha256);
    if !final_root.exists() {
        let incoming = archive_parent.join(format!(".incoming-{}", std::process::id()));
        if incoming.exists() {
            bail!(
                "stale incoming archive directory exists: {}",
                incoming.display()
            );
        }
        fs::create_dir_all(&incoming)?;
        for entry in &stage.targets {
            let bytes = fs::read(public_target(registry, entry)?)?;
            if sha256_bytes(&bytes) != entry.old_sha256 {
                bail!("archive source drifted during copy");
            }
            immutable_write(&incoming.join(&entry.archive_file), &bytes)?;
        }
        immutable_write(&incoming.join("manifest.v1.json"), &manifest_bytes)?;
        let marker = CompletionMarker {
            schema: "JopeDimeFollowUpArchiveComplete.v1".to_string(),
            status: "COMPLETE".to_string(),
            manifest_sha256: manifest_sha256.clone(),
        };
        // Marker-last: no archive member is created after this receipt.
        immutable_write(
            &incoming.join("ARCHIVE_COMPLETE"),
            &canonical_json(&serde_json::to_value(marker)?)?,
        )?;
        fs::rename(&incoming, &final_root)?;
        File::open(&archive_parent)?.sync_all()?;
    }
    let pointer = ArchivePointer {
        schema: "JopeDimeFollowUpArchivePointer.v1".to_string(),
        status: "COMPLETE".to_string(),
        archive_manifest_sha256: manifest_sha256,
    };
    validate_archive(stage, &pointer)?;
    immutable_write(
        &pointer_path,
        &canonical_json(&serde_json::to_value(&pointer)?)?,
    )?;
    Ok((pointer, final_root))
}
