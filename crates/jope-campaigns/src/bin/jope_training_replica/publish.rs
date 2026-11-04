use super::*;

// manifest
// ---------------------------------------------------------------------------

pub(super) fn build_manifest(
    archive: &Path,
    entries: &[Entry],
    db: &DbSnapshot,
    lock_held: bool,
) -> Value {
    let files: Vec<Value> = entries
        .iter()
        .map(|e| {
            json!({
                "path": e.path,
                "bytes": e.bytes,
                "sha256": e.sha256,
                "mtime_ms": e.mtime_ms,
            })
        })
        .collect();
    json!({
        "schema": MANIFEST_SCHEMA,
        "generated_at_ms": now_ms(),
        "source_node": SOURCE_NODE,
        "source_archive": archive.to_string_lossy(),
        "capture_lock_held_during_scan": lock_held,
        "file_count": entries.len(),
        "total_bytes": entries.iter().map(|e| e.bytes).sum::<u64>(),
        "db": {
            "path": DB_RELATIVE,
            "sha256": db.sha256,
            "bytes": db.bytes,
            "snapshot_mechanism": "VACUUM INTO",
            "integrity_check": db.integrity,
            "run_rows": db.run_rows,
            "epoch_rows": db.epoch_rows,
            "epoch_field_rows": db.epoch_field_rows,
        },
        "files": files,
    })
}

/// Every object a manifest depends on: the payload files plus the database
/// snapshot. Used both to decide what to upload and to decide what remote GC
/// must not delete.
pub(super) fn manifest_objects(manifest: &Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if let Some(sha) = manifest.pointer("/db/sha256").and_then(Value::as_str) {
        if is_sha256(sha) {
            out.insert(sha.to_string());
        }
    }
    for file in manifest
        .get("files")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        if let Some(sha) = file.get("sha256").and_then(Value::as_str) {
            if is_sha256(sha) {
                out.insert(sha.to_string());
            }
        }
    }
    out
}

pub(super) fn manifest_name(manifest: &Value, sha: &str) -> String {
    let ts = manifest
        .get("generated_at_ms")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    format!("{ts:013}-{}.v1.json", &sha[..12])
}

// ---------------------------------------------------------------------------
// per-target publish
// ---------------------------------------------------------------------------

pub(super) struct PublishOutcome {
    pub(super) objects_uploaded: usize,
    pub(super) bytes_uploaded: u64,
    pub(super) objects_present: usize,
    pub(super) published_sha256: String,
    pub(super) gc_deleted: usize,
    /// Objects found corrupt by the scrub and re-uploaded on this cycle.
    pub(super) objects_repaired: usize,
}

/// Upload counters that survive a mid-cycle failure, so the receipt can say how
/// far a broken target got instead of reporting a flat zero.
#[derive(Default)]
pub(super) struct Progress {
    pub(super) uploaded: usize,
    pub(super) bytes: u64,
}

// Keep replica content, manifest identity and progress state explicit at publication.
#[allow(clippy::too_many_arguments)]
pub(super) fn publish(
    target: &Target,
    rroot: &str,
    stage: &Path,
    entries: &[Entry],
    db: &DbSnapshot,
    manifest: &Value,
    manifest_bytes: &[u8],
    manifest_sha: &str,
    progress: &mut Progress,
) -> Result<PublishOutcome> {
    let survey = remote_objects(target, rroot)?;
    let present = &survey.verified;
    let needed = manifest_objects(manifest);
    // Anything the scrub rejected is about to be re-uploaded, because it is
    // absent from `verified` and `mv -f` overwrites in place.
    let repaired = needed.intersection(&survey.corrupt).count();

    // Upload every referenced object the target lacks, BEFORE the manifest that
    // names them. This ordering is the whole atomicity argument: a manifest is
    // only ever published on top of objects already verified at rest.
    let mut done: BTreeSet<String> = BTreeSet::new();
    if !present.contains(&db.sha256) {
        upload_object(target, rroot, &db.sha256, &db.path)?;
        progress.uploaded += 1;
        progress.bytes += db.bytes;
        done.insert(db.sha256.clone());
    }
    for entry in entries {
        if present.contains(&entry.sha256) || done.contains(&entry.sha256) {
            continue;
        }
        upload_object(
            target,
            rroot,
            &entry.sha256,
            &stage_path(stage, &entry.sha256),
        )?;
        progress.uploaded += 1;
        progress.bytes += entry.bytes;
        done.insert(entry.sha256.clone());
    }

    // Publish: land the manifest under its content-addressed name, then flip
    // the pointer with a rename on the same filesystem.
    let tmp = upload_bytes(target, rroot, manifest_bytes)?;
    let name = manifest_name(manifest, manifest_sha);
    let cur_tmp = format!("{rroot}/.{CURRENT_MANIFEST}.{}.tmp", std::process::id());
    ssh_run(
        target,
        &format!(
            "set -e; mkdir -p {}; mv -f {} {}; cp -f {} {}; mv -f {} {}",
            quote(&format!("{rroot}/manifests")),
            quote(&format!("{rroot}/incoming/{tmp}")),
            quote(&format!("{rroot}/manifests/{name}")),
            quote(&format!("{rroot}/manifests/{name}")),
            quote(&cur_tmp),
            quote(&cur_tmp),
            quote(&format!("{rroot}/{CURRENT_MANIFEST}"))
        ),
    )?;

    // Read the published state back off the target. "I sent it" is not evidence.
    let echoed = ssh_run(
        target,
        &format!(
            "sha256sum {}",
            quote(&format!("{rroot}/{CURRENT_MANIFEST}"))
        ),
    )?;
    let echoed_sha = echoed
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if echoed_sha != manifest_sha {
        bail!("published manifest reads back as {echoed_sha:?}, expected {manifest_sha}");
    }
    // Re-survey by content: every object the manifest names must now be present
    // AND hash correctly on the target. This is the statement "the replica is
    // restorable right now", checked rather than assumed.
    let after = remote_objects(target, rroot)?;
    let missing: Vec<&String> = needed
        .iter()
        .filter(|sha| !after.verified.contains(*sha))
        .collect();
    if !missing.is_empty() {
        bail!(
            "{} referenced objects missing or corrupt after publish, first {}",
            missing.len(),
            missing[0]
        );
    }

    let gc_deleted = gc(target, rroot, &after.all()).unwrap_or(0);
    Ok(PublishOutcome {
        objects_uploaded: progress.uploaded,
        bytes_uploaded: progress.bytes,
        objects_present: after.verified.len(),
        published_sha256: echoed_sha,
        gc_deleted,
        objects_repaired: repaired,
    })
}

/// Drop manifests beyond the retention window and any object no retained
/// manifest references.
///
/// Fail-safe: if *any* retained manifest cannot be read or parsed, no object is
/// deleted. Deleting on an incomplete reference set is exactly how a backup
/// system eats the thing it was protecting.
pub(super) fn gc(target: &Target, rroot: &str, present: &BTreeSet<String>) -> Result<usize> {
    let listing = ssh_run(
        target,
        &format!(
            "ls -1 {} 2>/dev/null || true",
            quote(&format!("{rroot}/manifests"))
        ),
    )?;
    let mut names: Vec<String> = listing
        .lines()
        .map(str::trim)
        .filter(|n| n.ends_with(".v1.json"))
        .map(str::to_string)
        .collect();
    // Names are `<zero-padded ms>-<sha12>.v1.json`, so lexical order is time order.
    names.sort();
    let keep_from = names.len().saturating_sub(RETAIN_MANIFESTS);
    let (drop_names, keep_names) = names.split_at(keep_from);

    let mut referenced: BTreeSet<String> = BTreeSet::new();
    for name in keep_names {
        let text = ssh_run(
            target,
            &format!("cat {}", quote(&format!("{rroot}/manifests/{name}"))),
        )?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("parse retained manifest {name}"))?;
        referenced.extend(manifest_objects(&value));
    }
    if referenced.is_empty() {
        return Ok(0);
    }

    let dead: Vec<String> = present
        .iter()
        .filter(|sha| !referenced.contains(*sha))
        .cloned()
        .collect();
    let mut script = format!(
        "cd {} && find . -type f -mmin +{INCOMING_STALE_MINUTES} -delete 2>/dev/null; true",
        quote(&format!("{rroot}/incoming"))
    );
    for name in drop_names {
        script.push_str(&format!(
            "; rm -f {}",
            quote(&format!("{rroot}/manifests/{name}"))
        ));
    }
    for sha in &dead {
        script.push_str(&format!(
            "; rm -f {}",
            quote(&format!("{rroot}/objects/{}", object_rel(sha)))
        ));
    }
    ssh_run(target, &script)?;
    Ok(dead.len())
}

// ---------------------------------------------------------------------------
