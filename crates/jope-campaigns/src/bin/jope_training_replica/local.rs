use super::*;

// local scan
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(super) struct Entry {
    pub(super) path: String,
    pub(super) bytes: u64,
    pub(super) sha256: String,
    pub(super) mtime_ms: u64,
}

/// Files excluded from the payload, each for a reason:
/// the live database and its WAL sidecars (replaced by the `VACUUM INTO`
/// snapshot), the capture flock, and any `.<name>.<pid>.tmp` a concurrent
/// `write_atomic` happens to be holding mid-rename.
pub(super) fn skipped(name: &str) -> bool {
    name.starts_with('.')
        || name == "history.sqlite3"
        || name == "history.sqlite3-wal"
        || name == "history.sqlite3-shm"
}

pub(super) fn walk(_archive: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = match fs::read_dir(dir) {
        Ok(iter) => iter.filter_map(std::result::Result::ok).collect(),
        Err(_) => return Ok(()),
    };
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if skipped(&name) {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            walk(_archive, &path, out)?;
        } else if meta.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

pub(super) fn stage_path(stage: &Path, sha: &str) -> PathBuf {
    stage.join(&sha[..2]).join(sha)
}

/// Copy the payload into a local content-addressed staging store and hash the
/// **copies**, then upload from there.
///
/// Hashing the archive in place and uploading from it later is wrong, and it
/// failed on the very first production run: `training/history` is live. The
/// capture timer appends to `train_metrics.jsonl` and rewrites `index.v1.json`
/// every three minutes, and the analysis timer rewrites `db/export/*` every
/// fifteen, while a full cycle spends minutes pushing ~80 MB. Any file that
/// moved in between arrived at the target hashing to something other than what
/// the manifest promised, and the transfer was correctly rejected as corrupt.
///
/// Staging removes the race rather than tolerating it: the bytes are frozen
/// under the capture lock, the manifest describes the frozen bytes, and the
/// upload reads the frozen bytes. Cost is one ~11 MB copy per cycle.
pub(super) fn stage_payload(archive: &Path, stage: &Path) -> Result<Vec<Entry>> {
    let mut paths = Vec::new();
    walk(archive, archive, &mut paths)?;
    fs::create_dir_all(stage)?;
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let rel = path
            .strip_prefix(archive)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let mtime_ms = fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let temporary = stage.join(format!(".staging.{}.tmp", std::process::id()));
        // A file can vanish mid-scan (a rewrite rotating into `rewrites/`);
        // that is not a replication failure, the next cycle picks it up.
        let Ok(bytes) = fs::copy(&path, &temporary) else {
            let _ = fs::remove_file(&temporary);
            continue;
        };
        let (sha, _) = sha256_file(&temporary)?;
        let dest = stage_path(stage, &sha);
        fs::create_dir_all(dest.parent().context("stage parent")?)?;
        fs::rename(&temporary, &dest)?;
        out.push(Entry {
            path: rel,
            bytes,
            sha256: sha,
            mtime_ms,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Drop staged objects the current manifest no longer references, so the stage
/// stays about one archive in size instead of growing without bound.
pub(super) fn sweep_stage(stage: &Path, keep: &BTreeSet<String>) -> usize {
    let mut removed = 0;
    let Ok(prefixes) = fs::read_dir(stage) else {
        return 0;
    };
    for prefix in prefixes.filter_map(std::result::Result::ok) {
        let Ok(objects) = fs::read_dir(prefix.path()) else {
            continue;
        };
        for object in objects.filter_map(std::result::Result::ok) {
            let name = object.file_name().to_string_lossy().to_string();
            if is_sha256(&name) && !keep.contains(&name) && fs::remove_file(object.path()).is_ok() {
                removed += 1;
            }
        }
    }
    removed
}

// ---------------------------------------------------------------------------
// SQLite snapshot
// ---------------------------------------------------------------------------

pub(super) struct DbSnapshot {
    pub(super) path: PathBuf,
    pub(super) sha256: String,
    pub(super) bytes: u64,
    pub(super) integrity: String,
    pub(super) run_rows: i64,
    pub(super) epoch_rows: i64,
    pub(super) epoch_field_rows: i64,
}

pub(super) fn table_count(conn: &Connection, table: &str) -> i64 {
    let query = match table {
        "run" => "SELECT count(*) FROM run",
        "epoch" => "SELECT count(*) FROM epoch",
        "epoch_field" => "SELECT count(*) FROM epoch_field",
        _ => return -1,
    };
    conn.query_row(query, [], |row| row.get(0)).unwrap_or(-1)
}

/// Take a consistent snapshot of the live WAL database.
///
/// `VACUUM INTO` executes inside a read transaction: it never blocks the
/// ingester's writer, it never observes a half-committed transaction, and its
/// output is a fully checkpointed standalone database with no `-wal` sidecar to
/// forget. Byte-copying `history.sqlite3` would do none of those three things.
pub(super) fn snapshot_db(archive: &Path, stage: &Path) -> Result<DbSnapshot> {
    let source = archive.join(DB_RELATIVE);
    fs::create_dir_all(stage)?;
    let staged = stage.join(format!(".history.sqlite3.{}.tmp", std::process::id()));
    let _ = fs::remove_file(&staged);

    let conn = Connection::open(&source).with_context(|| format!("open {}", source.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(60))?;
    let literal = staged.to_string_lossy().replace('\'', "''");
    conn.execute_batch(&format!("VACUUM INTO '{literal}'"))
        .context("VACUUM INTO snapshot")?;
    drop(conn);

    // Prove the snapshot before anyone depends on it: an unreadable replica
    // discovered at restore time is worse than no replica at all.
    let snap = Connection::open(&staged)?;
    let integrity: String = snap
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .unwrap_or_else(|e| format!("error: {e}"));
    let run_rows = table_count(&snap, "run");
    let epoch_rows = table_count(&snap, "epoch");
    let epoch_field_rows = table_count(&snap, "epoch_field");
    drop(snap);
    if integrity != "ok" {
        bail!("snapshot integrity_check said {integrity:?}");
    }

    // The snapshot joins the same content-addressed stage as the payload, so
    // every byte the manifest names is uploaded from frozen local storage.
    let (sha, bytes) = sha256_file(&staged)?;
    let final_path = stage_path(stage, &sha);
    fs::create_dir_all(final_path.parent().context("stage parent")?)?;
    fs::rename(&staged, &final_path)?;
    Ok(DbSnapshot {
        path: final_path,
        sha256: sha,
        bytes,
        integrity,
        run_rows,
        epoch_rows,
        epoch_field_rows,
    })
}

// ---------------------------------------------------------------------------
