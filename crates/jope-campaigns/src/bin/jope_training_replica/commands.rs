use super::*;

// ---------------------------------------------------------------------------
// replicate
// ---------------------------------------------------------------------------

pub(super) fn cmd_replicate(root: &Path) -> Result<Value> {
    let archive = th::archive_root(root);
    if !archive.join(th::INDEX_FILE).exists() {
        bail!("no archive at {} — nothing to replicate", archive.display());
    }
    let replica = replica_dir(root);
    fs::create_dir_all(&replica)?;
    let targets = targets_from_env()?;
    let rroot = remote_root();
    let previous = load_status(root)?;
    let started = now_ms();

    // Hold the capture lock only while the payload is frozen into the stage, so
    // segments and index.v1.json are captured as one coherent set. It is
    // released before any network work: this must never sit in the capture
    // path's way while an ssh hangs.
    let stage = replica.join("stage");
    let lock = th::CaptureLock::acquire(&archive, 20, 250).unwrap_or(None);
    let lock_held = lock.is_some();
    let entries = stage_payload(&archive, &stage)?;
    drop(lock);

    let db = snapshot_db(&archive, &stage)?;
    let manifest = build_manifest(&archive, &entries, &db, lock_held);
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let manifest_sha = sha256_hex(&manifest_bytes);
    let name = manifest_name(&manifest, &manifest_sha);
    th::write_atomic(&replica.join("manifests").join(&name), &manifest_bytes)?;
    th::write_atomic(&replica.join("manifest-latest.v1.json"), &manifest_bytes)?;

    let mut rows = Vec::new();
    let mut healthy = 0usize;
    for target in &targets {
        let label = target.label();
        let t0 = now_ms();
        let prior = previous_success(&previous, &label);
        let mut progress = Progress::default();
        let row = match publish(
            target,
            &rroot,
            &stage,
            &entries,
            &db,
            &manifest,
            &manifest_bytes,
            &manifest_sha,
            &mut progress,
        ) {
            Ok(outcome) => {
                healthy += 1;
                json!({
                    "target": label,
                    "node": target.node,
                    "port": target.port,
                    "remote_root": rroot,
                    "ok": true,
                    "error": Value::Null,
                    "published_manifest_sha256": outcome.published_sha256,
                    "matches_source": true,
                    "objects_uploaded": outcome.objects_uploaded,
                    "bytes_uploaded": outcome.bytes_uploaded,
                    "objects_present": outcome.objects_present,
                    "objects_repaired": outcome.objects_repaired,
                    "gc_objects_deleted": outcome.gc_deleted,
                    "duration_ms": now_ms().saturating_sub(t0),
                    "last_success_ms": now_ms(),
                })
            }
            Err(err) => json!({
                "target": label,
                "node": target.node,
                "port": target.port,
                "remote_root": rroot,
                "ok": false,
                "error": format!("{err:#}"),
                "published_manifest_sha256": Value::Null,
                "matches_source": false,
                "objects_uploaded": progress.uploaded,
                "bytes_uploaded": progress.bytes,
                "duration_ms": now_ms().saturating_sub(t0),
                "last_success_ms": prior,
            }),
        };
        rows.push(row);
    }

    let stage_swept = sweep_stage(&stage, &manifest_objects(&manifest));
    let last_success = rows
        .iter()
        .filter_map(|r| r.get("last_success_ms").and_then(Value::as_u64))
        .max()
        .unwrap_or(0);
    let report = json!({
        "schema": STATUS_SCHEMA,
        "generated_at_ms": now_ms(),
        "started_at_ms": started,
        "source_node": SOURCE_NODE,
        "source_archive": archive.to_string_lossy(),
        "capture_lock_held_during_scan": lock_held,
        "manifest_sha256": manifest_sha,
        "manifest_name": name,
        "manifest_generated_at_ms": manifest.get("generated_at_ms").cloned().unwrap_or(Value::Null),
        "payload_files": entries.len(),
        "payload_bytes": entries.iter().map(|e| e.bytes).sum::<u64>(),
        "stage_objects_swept": stage_swept,
        "db_bytes": db.bytes,
        "db_sha256": db.sha256,
        "db_integrity_check": db.integrity,
        "db_run_rows": db.run_rows,
        "db_epoch_rows": db.epoch_rows,
        "db_epoch_field_rows": db.epoch_field_rows,
        "target_count": targets.len(),
        "healthy_targets": healthy,
        "ok": healthy == targets.len(),
        "last_success_ms": last_success,
        "targets": rows,
    });
    write_status(root, &report)?;
    append_event(root, &report);
    Ok(report)
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

pub(super) fn cmd_verify(root: &Path) -> Result<Value> {
    let targets = targets_from_env()?;
    let rroot = remote_root();
    let local = fs::read(replica_dir(root).join("manifest-latest.v1.json")).ok();
    let local_sha = local.as_deref().map(sha256_hex);
    let mut rows = Vec::new();
    let mut healthy = 0usize;
    for target in &targets {
        let row = (|| -> Result<Value> {
            let text = ssh_run(
                target,
                &format!("cat {}", quote(&format!("{rroot}/{CURRENT_MANIFEST}"))),
            )?;
            let sha = sha256_hex(text.as_bytes());
            let manifest: Value = serde_json::from_str(&text).context("parse remote manifest")?;
            let needed = manifest_objects(&manifest);
            let survey = remote_objects(target, &rroot)?;
            let missing: Vec<&String> = needed
                .iter()
                .filter(|s| !survey.verified.contains(*s))
                .collect();
            Ok(json!({
                "target": target.label(),
                "ok": missing.is_empty(),
                "remote_manifest_sha256": sha,
                "matches_local_latest": Some(&sha) == local_sha.as_ref(),
                "manifest_generated_at_ms": manifest.get("generated_at_ms").cloned().unwrap_or(Value::Null),
                "referenced_objects": needed.len(),
                "objects_present": survey.verified.len(),
                "objects_corrupt": survey.corrupt.len(),
                "missing_objects": missing.len(),
                "db_epoch_rows": manifest.pointer("/db/epoch_rows").cloned().unwrap_or(Value::Null),
                "error": Value::Null,
            }))
        })();
        match row {
            Ok(value) => {
                if value.get("ok").and_then(Value::as_bool) == Some(true) {
                    healthy += 1;
                }
                rows.push(value);
            }
            Err(err) => rows.push(json!({
                "target": target.label(),
                "ok": false,
                "error": format!("{err:#}"),
            })),
        }
    }
    Ok(json!({
        "schema": "JopeTrainingReplicaVerify.v1",
        "generated_at_ms": now_ms(),
        "local_manifest_sha256": local_sha,
        "healthy_targets": healthy,
        "target_count": targets.len(),
        "targets": rows,
    }))
}

// ---------------------------------------------------------------------------
// restore
// ---------------------------------------------------------------------------

/// Canonical serialization of one `epoch` row, used to prove a restored row is
/// byte-identical to the source. `raw_json` is the trainer's original line, so
/// comparing it compares the actual archived bytes rather than a re-derivation.
type EpochProbeRow = (String, i64, String, String, String);

fn epoch_probe(conn: &Connection, limit: usize) -> Result<Vec<EpochProbeRow>> {
    let mut stmt = conn.prepare(
        "SELECT run_key, epoch, phase, raw_sha256, raw_json \
         FROM epoch ORDER BY run_key, epoch, phase LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

pub(super) fn cmd_restore(
    root: &Path,
    target_name: &str,
    into: &Path,
    compare: bool,
) -> Result<Value> {
    let targets = targets_from_env()?;
    let target = targets
        .iter()
        .find(|t| t.node == target_name || t.label() == target_name)
        .cloned()
        .or_else(|| {
            parse_targets(target_name)
                .ok()
                .and_then(|v| v.into_iter().next())
        })
        .context("unknown restore target")?;
    let rroot = remote_root();

    let text = ssh_run(
        &target,
        &format!("cat {}", quote(&format!("{rroot}/{CURRENT_MANIFEST}"))),
    )?;
    let manifest_sha = sha256_hex(text.as_bytes());
    let manifest: Value = serde_json::from_str(&text).context("parse remote manifest")?;

    fs::create_dir_all(into)?;
    let objdir = into.join(".objects");
    fs::create_dir_all(&objdir)?;

    // One connection, exactly the referenced objects: the object names are fed
    // to remote tar on stdin, so this neither hits ARG_MAX nor drags down
    // objects that belong to retained older manifests.
    let needed = manifest_objects(&manifest);
    let list: String = needed
        .iter()
        .map(|sha| format!("{}\n", object_rel(sha)))
        .collect();
    let tarball = into.join(".restore.tar");
    {
        let mut child = ssh_base(&target)
            .arg(format!(
                "tar -C {} -cf - -T -",
                quote(&format!("{rroot}/objects"))
            ))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        child
            .stdin
            .as_mut()
            .context("stdin")?
            .write_all(list.as_bytes())?;
        drop(child.stdin.take());
        let out = child.wait_with_output()?;
        if !out.status.success() {
            bail!(
                "remote tar exit {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        fs::write(&tarball, &out.stdout)?;
    }
    let status = Command::new("tar")
        .arg("-xf")
        .arg(&tarball)
        .arg("-C")
        .arg(&objdir)
        .status()?;
    if !status.success() {
        bail!("local tar extract failed");
    }
    let _ = fs::remove_file(&tarball);

    // Materialize the archive tree, verifying every object against the hash the
    // manifest claims. A replica that hands back different bytes fails here.
    let mut restored = 0usize;
    let mut restored_bytes = 0u64;
    let mut bad: Vec<String> = Vec::new();
    let place = |rel: &str, sha: &str, bytes: u64, bad: &mut Vec<String>| -> Result<()> {
        let object = objdir.join(object_rel(sha));
        if !object.exists() {
            bad.push(format!("{rel}: object {sha} absent from replica"));
            return Ok(());
        }
        let (observed, size) = sha256_file(&object)?;
        if observed != sha || size != bytes {
            bad.push(format!(
                "{rel}: sha {observed} size {size}, manifest said {sha} size {bytes}"
            ));
            return Ok(());
        }
        let dest = into.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&object, &dest)?;
        Ok(())
    };
    for file in manifest
        .get("files")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let (Some(rel), Some(sha), Some(bytes)) = (
            file.get("path").and_then(Value::as_str),
            file.get("sha256").and_then(Value::as_str),
            file.get("bytes").and_then(Value::as_u64),
        ) else {
            continue;
        };
        place(rel, sha, bytes, &mut bad)?;
        restored += 1;
        restored_bytes += bytes;
    }
    let db_sha = manifest
        .pointer("/db/sha256")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let db_bytes = manifest
        .pointer("/db/bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    place(DB_RELATIVE, db_sha, db_bytes, &mut bad)?;
    fs::remove_dir_all(&objdir).ok();

    // The restored database must open, pass integrity_check, and hold the rows
    // the manifest promised.
    let db_path = into.join(DB_RELATIVE);
    let conn = Connection::open(&db_path).with_context(|| format!("open {}", db_path.display()))?;
    let integrity: String = conn
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .unwrap_or_else(|e| format!("error: {e}"));
    let run_rows = table_count(&conn, "run");
    let epoch_rows = table_count(&conn, "epoch");
    let epoch_field_rows = table_count(&conn, "epoch_field");
    let probe = epoch_probe(&conn, 3)?;

    let mut comparison = Value::Null;
    if compare {
        let archive = th::archive_root(root);
        let live = Connection::open(archive.join(DB_RELATIVE))?;
        let live_probe = epoch_probe(&live, 3)?;
        let identical =
            probe.len() == live_probe.len() && probe.iter().zip(&live_probe).all(|(a, b)| a == b);
        let sample: Vec<Value> = probe
            .iter()
            .map(|(run_key, epoch, phase, sha, raw)| {
                json!({
                    "run_key": run_key,
                    "epoch": epoch,
                    "phase": phase,
                    "raw_sha256": sha,
                    "raw_json_bytes": raw.len(),
                    "raw_json_sha256": sha256_hex(raw.as_bytes()),
                })
            })
            .collect();
        // Payload files whose live copy has not grown since the manifest must
        // still hash the same; ones that grew are the archive appending, not a
        // replication defect, and are reported separately rather than as errors.
        let mut identical_files = 0usize;
        let mut grown = 0usize;
        let mut differing = Vec::new();
        for file in manifest
            .get("files")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let (Some(rel), Some(sha), Some(bytes)) = (
                file.get("path").and_then(Value::as_str),
                file.get("sha256").and_then(Value::as_str),
                file.get("bytes").and_then(Value::as_u64),
            ) else {
                continue;
            };
            let live_path = archive.join(rel);
            let Ok((live_sha, live_bytes)) = sha256_file(&live_path) else {
                differing.push(format!("{rel}: absent from live archive"));
                continue;
            };
            if live_sha == sha {
                identical_files += 1;
            } else if live_bytes > bytes {
                grown += 1;
            } else {
                differing.push(format!("{rel}: live {live_sha} vs replica {sha}"));
            }
        }
        comparison = json!({
            "epoch_rows_sampled": probe.len(),
            "sampled_rows_byte_identical": identical,
            "sample": sample,
            "payload_files_identical_to_live": identical_files,
            "payload_files_grown_since_manifest": grown,
            "payload_files_differing": differing,
        });
    }

    Ok(json!({
        "schema": RESTORE_SCHEMA,
        "generated_at_ms": now_ms(),
        "restored_from": target.label(),
        "remote_root": rroot,
        "restored_into": into.to_string_lossy(),
        "manifest_sha256": manifest_sha,
        "manifest_generated_at_ms": manifest.get("generated_at_ms").cloned().unwrap_or(Value::Null),
        "files_restored": restored,
        "files_bytes": restored_bytes,
        "db_bytes": db_bytes,
        "hash_mismatches": bad.len(),
        "hash_mismatch_detail": bad,
        "db_integrity_check": integrity,
        "db_run_rows": run_rows,
        "db_epoch_rows": epoch_rows,
        "db_epoch_field_rows": epoch_field_rows,
        "manifest_run_rows": manifest.pointer("/db/run_rows").cloned().unwrap_or(Value::Null),
        "manifest_epoch_rows": manifest.pointer("/db/epoch_rows").cloned().unwrap_or(Value::Null),
        "manifest_epoch_field_rows": manifest.pointer("/db/epoch_field_rows").cloned().unwrap_or(Value::Null),
        "rows_match_manifest": run_rows == manifest.pointer("/db/run_rows").and_then(Value::as_i64).unwrap_or(-1)
            && epoch_rows == manifest.pointer("/db/epoch_rows").and_then(Value::as_i64).unwrap_or(-1)
            && epoch_field_rows == manifest.pointer("/db/epoch_field_rows").and_then(Value::as_i64).unwrap_or(-1),
        "compared_with_live": comparison,
    }))
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

mod status;
use status::previous_success;
pub(super) use status::{append_event, load_status, write_status};
#[cfg(test)]
mod tests;
