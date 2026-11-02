//! Off-node replication of the durable DL training-metrics archive.
//!
//! `training/history/` is the only place every per-epoch train/validation loss
//! the policy model ever produced still exists. Until this binary it lived on
//! exactly one filesystem — `/dev/nvme2n1p2` on xbabe2 — the *same* filesystem
//! as the `var/runs` tree it is supposed to outlive. That survives garbage
//! collection and it survives a crash; it does not survive losing the disk or
//! the node. This ships the archive to other machines.
//!
//! Design, in the order the failure modes forced it:
//!
//! * **Content-addressed.** Every payload file is stored on the target under
//!   `objects/<aa>/<sha256>`. Re-replicating an unchanged archive uploads
//!   nothing, so the cycle is incremental and idempotent by construction rather
//!   than by a timestamp comparison that can lie.
//! * **Verified at rest, not just in flight.** An object is written to
//!   `incoming/<sha>.part`, hashed *by the receiving node*, and only renamed
//!   into `objects/` when the remote hash equals the local one. A truncated or
//!   corrupted transfer can never occupy a final object name.
//! * **Atomic publish.** A replica's visible state is one file,
//!   `manifest-current.v1.json`, swapped by `mv` (same filesystem, so the
//!   rename is atomic). Objects are always uploaded *before* the manifest that
//!   references them, so a reader never sees a manifest pointing at bytes that
//!   have not landed. There is no window in which a replica is half-published.
//! * **The SQLite is snapshotted, never copied.** `db/history.sqlite3` is a
//!   live WAL database with `jope-training-archive.timer` writing to it every
//!   15 minutes. Copying the file byte-wise while a writer is active yields a
//!   torn database — and copying it *without* the `-wal` yields a silently
//!   stale one. `VACUUM INTO` runs inside a read transaction and emits a
//!   self-contained, already-checkpointed database file. It also happens to be
//!   deterministic: an unchanged database produces a byte-identical snapshot,
//!   so content-addressing dedupes it to a zero-byte upload.
//! * **Loud on failure.** Every cycle rewrites `ops/training-replica.v1.json`
//!   (schema `JopeTrainingReplicaStatus.v1`) whether it succeeded or not, and
//!   `jope_selfplay_alarms` turns a stale or unreachable replica into a
//!   critical alarm. `last_success_ms` is carried forward per target, so a
//!   target that silently stops accepting data ages visibly instead of
//!   disappearing from the report.
//!
//! Subcommands: `replicate` (the timer's job), `verify` (read-only check of
//! what each target actually holds), `restore` (reconstruct the archive from a
//! replica and prove it), `status` (print the receipt).

use anyhow::{bail, Context, Result};
use jope_dime_regressor::paths;
use jope_dime_regressor::training_history as th;
use rusqlite::Connection;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

jope_paths::once_str!(
    fn root_str => paths::flywheel_root()
);
mod commands;

const SOURCE_NODE: &str = "xbabe2";
/// `host:port` pairs. xbabe1 is deliberately absent: it sits at 91% disk with a
/// load average near 190, and a replica has no business making that worse when
/// xbabe3 (72%, 257G free) and xbabe4 (4%, 1.9T free) are both available and in
/// different failure domains — xbabe4 is a separate VM behind its own port.
const DEFAULT_TARGETS: &str = "xbabe3:22,xbabe4:2224";
jope_paths::once_str!(
    fn default_remote_root_str => paths::legacy_tree("jope-replica").join("test_exact_gen0/training-history")
);

const MANIFEST_SCHEMA: &str = "JopeTrainingReplicaManifest.v1";
const STATUS_SCHEMA: &str = "JopeTrainingReplicaStatus.v1";
const RESTORE_SCHEMA: &str = "JopeTrainingReplicaRestore.v1";
const CURRENT_MANIFEST: &str = "manifest-current.v1.json";
const STATUS_RELATIVE: &str = "ops/training-replica.v1.json";
/// Archive-relative location of the database inside a manifest. The bytes under
/// this name are the `VACUUM INTO` snapshot, not the live file.
const DB_RELATIVE: &str = "db/history.sqlite3";
/// Older manifests are kept so that corruption of the live archive does not
/// immediately propagate to the only remote copy: three cycles of history is
/// three chances to notice before the last clean snapshot ages out.
const RETAIN_MANIFESTS: usize = 3;
/// A `.part` older than this was orphaned by a killed cycle.
const INCOMING_STALE_MINUTES: u64 = 120;
const SSH_CONNECT_TIMEOUT: &str = "12";

#[derive(Clone, Debug)]
struct Target {
    node: String,
    port: u16,
}

impl Target {
    fn label(&self) -> String {
        if self.port == 22 {
            self.node.clone()
        } else {
            format!("{}:{}", self.node, self.port)
        }
    }
}

fn parse_targets(spec: &str) -> Result<Vec<Target>> {
    let mut out = Vec::new();
    for piece in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (node, port) = match piece.split_once(':') {
            Some((n, p)) => (
                n,
                p.parse::<u16>()
                    .with_context(|| format!("bad port in {piece:?}"))?,
            ),
            None => (piece, 22u16),
        };
        if node.is_empty()
            || !node
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
        {
            bail!("bad target host {node:?}");
        }
        // xbabe2 is this host. The `xbabe2` ssh alias resolves to a different
        // machine entirely, so replicating "to xbabe2" would silently write the
        // archive somewhere it is not wanted and prove nothing about durability.
        if node == SOURCE_NODE {
            bail!("{SOURCE_NODE} is the source node; replicating to it is a no-op");
        }
        out.push(Target {
            node: node.to_string(),
            port,
        });
    }
    if out.is_empty() {
        bail!("no replica targets configured");
    }
    Ok(out)
}

fn targets_from_env() -> Result<Vec<Target>> {
    parse_targets(&std::env::var("JOPE_REPLICA_TARGETS").unwrap_or_else(|_| DEFAULT_TARGETS.into()))
}

fn remote_root() -> String {
    std::env::var("JOPE_REPLICA_REMOTE_ROOT").unwrap_or_else(|_| default_remote_root_str().into())
}

fn root_from_env() -> PathBuf {
    PathBuf::from(std::env::var("JOPE_ROOT").unwrap_or_else(|_| root_str().into()))
}

/// `<root>/training/replica` — replication working state. Deliberately a
/// *sibling* of `training/history`, never a child: a staging area inside the
/// tree being replicated would replicate itself.
fn replica_dir(root: &Path) -> PathBuf {
    root.join("training").join("replica")
}

fn now_ms() -> u64 {
    th::now_ms()
}

// ---------------------------------------------------------------------------
// hashing
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Stream a file through SHA-256. The database snapshot is ~70 MB, so this
/// never reads a whole payload into memory.
fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex::encode(hasher.finalize()), total))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// `objects/<aa>/<sha>` — the two-character fan-out keeps any single directory
/// from collecting every object as the archive grows.
fn object_rel(sha: &str) -> String {
    format!("{}/{}", &sha[..2], sha)
}

// ---------------------------------------------------------------------------
// local scan
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Entry {
    path: String,
    bytes: u64,
    sha256: String,
    mtime_ms: u64,
}

/// Files excluded from the payload, each for a reason:
/// the live database and its WAL sidecars (replaced by the `VACUUM INTO`
/// snapshot), the capture flock, and any `.<name>.<pid>.tmp` a concurrent
/// `write_atomic` happens to be holding mid-rename.
fn skipped(name: &str) -> bool {
    name.starts_with('.')
        || name == "history.sqlite3"
        || name == "history.sqlite3-wal"
        || name == "history.sqlite3-shm"
}

fn walk(_archive: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
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

fn stage_path(stage: &Path, sha: &str) -> PathBuf {
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
fn stage_payload(archive: &Path, stage: &Path) -> Result<Vec<Entry>> {
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
fn sweep_stage(stage: &Path, keep: &BTreeSet<String>) -> usize {
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

struct DbSnapshot {
    path: PathBuf,
    sha256: String,
    bytes: u64,
    integrity: String,
    run_rows: i64,
    epoch_rows: i64,
    epoch_field_rows: i64,
}

fn table_count(conn: &Connection, table: &str) -> i64 {
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
fn snapshot_db(archive: &Path, stage: &Path) -> Result<DbSnapshot> {
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
// ssh transport
// ---------------------------------------------------------------------------

fn ssh_base(target: &Target) -> Command {
    let mut cmd = Command::new("ssh");
    cmd.arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg(format!("ConnectTimeout={SSH_CONNECT_TIMEOUT}"))
        .arg("-o")
        .arg("ServerAliveInterval=15")
        .arg("-o")
        .arg("ServerAliveCountMax=4")
        .arg("-C")
        .arg("-p")
        .arg(target.port.to_string())
        .arg(&target.node);
    cmd
}

fn ssh_run(target: &Target, script: &str) -> Result<String> {
    let out = ssh_base(target)
        .arg(script)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("ssh {}", target.label()))?;
    if !out.status.success() {
        bail!(
            "ssh {} exit {:?}: {}",
            target.label(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Push bytes to `incoming/<tmp>` and return the hash the *receiving node*
/// computed over what actually landed. The caller compares it to the local hash
/// before promoting the object; nothing is ever renamed on trust.
fn push_stream(target: &Target, rroot: &str, tmp: &str, source: Stdio) -> Result<String> {
    let dest = format!("{rroot}/incoming/{tmp}");
    let script = format!(
        "set -e; mkdir -p {} ; cat > {} ; sha256sum {}",
        quote(&format!("{rroot}/incoming")),
        quote(&dest),
        quote(&dest)
    );
    let child = ssh_base(target)
        .arg(&script)
        .stdin(source)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("ssh {}", target.label()))?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "upload to {} exit {:?}: {}",
            target.label(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let sha = text
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if !is_sha256(&sha) {
        bail!("remote sha256sum returned {text:?}");
    }
    Ok(sha)
}

fn upload_object(target: &Target, rroot: &str, sha: &str, path: &Path) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let tmp = format!("{sha}.{}.part", std::process::id());
    let observed = push_stream(target, rroot, &tmp, Stdio::from(file))?;
    if observed != sha {
        let _ = ssh_run(
            target,
            &format!("rm -f {}", quote(&format!("{rroot}/incoming/{tmp}"))),
        );
        bail!("corrupt transfer of {sha}: target computed {observed}");
    }
    let rel = object_rel(sha);
    ssh_run(
        target,
        &format!(
            "set -e; mkdir -p {}; mv -f {} {}",
            quote(&format!("{rroot}/objects/{}", &sha[..2])),
            quote(&format!("{rroot}/incoming/{tmp}")),
            quote(&format!("{rroot}/objects/{rel}"))
        ),
    )?;
    Ok(())
}

fn upload_bytes(target: &Target, rroot: &str, bytes: &[u8]) -> Result<String> {
    let sha = sha256_hex(bytes);
    let tmp = format!("{sha}.{}.part", std::process::id());
    let dest = format!("{rroot}/incoming/{tmp}");
    let script = format!(
        "set -e; mkdir -p {} ; cat > {} ; sha256sum {}",
        quote(&format!("{rroot}/incoming")),
        quote(&dest),
        quote(&dest)
    );
    let mut child = ssh_base(target)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.as_mut().context("stdin")?.write_all(bytes)?;
    drop(child.stdin.take());
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "manifest upload to {} exit {:?}: {}",
            target.label(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let observed = text
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if observed != sha {
        bail!("corrupt manifest transfer: target computed {observed:?}");
    }
    Ok(tmp)
}

/// What a target actually holds, by content rather than by filename.
struct RemoteObjects {
    /// Objects whose bytes still hash to their own name.
    verified: BTreeSet<String>,
    /// Object names present but hashing to something else — bit rot, a truncated
    /// filesystem, or tampering.
    corrupt: BTreeSet<String>,
}

impl RemoteObjects {
    fn all(&self) -> BTreeSet<String> {
        self.verified.union(&self.corrupt).cloned().collect()
    }
}

/// Re-hash every object on the target and keep only the ones that still match
/// their own name.
///
/// Listing filenames is not enough, and this was a real hole rather than a
/// theoretical one: a corrupted object keeps its correct name, so a
/// name-only survey reports it "present", never re-uploads it, and the replica
/// stays quietly unusable until someone attempts a restore. Anything that fails
/// to verify is excluded from `verified`, which makes the ordinary upload path
/// repair it on this very cycle.
///
/// `nice` keeps the scrub off whatever the target is really doing.
fn remote_objects(target: &Target, rroot: &str) -> Result<RemoteObjects> {
    let text = ssh_run(
        target,
        &format!(
            "set -e; mkdir -p {} {} {}; cd {}; \
             find . -type f -print0 | xargs -0 -r nice -n 10 sha256sum 2>/dev/null || true",
            quote(&format!("{rroot}/objects")),
            quote(&format!("{rroot}/incoming")),
            quote(&format!("{rroot}/manifests")),
            quote(&format!("{rroot}/objects"))
        ),
    )?;
    let mut verified = BTreeSet::new();
    let mut corrupt = BTreeSet::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(hash), Some(path)) = (parts.next(), parts.next()) else {
            continue;
        };
        let name = path.rsplit('/').next().unwrap_or_default();
        if !is_sha256(name) {
            continue;
        }
        if hash == name {
            verified.insert(name.to_string());
        } else {
            corrupt.insert(name.to_string());
        }
    }
    Ok(RemoteObjects { verified, corrupt })
}

// ---------------------------------------------------------------------------
// manifest
// ---------------------------------------------------------------------------

fn build_manifest(archive: &Path, entries: &[Entry], db: &DbSnapshot, lock_held: bool) -> Value {
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
fn manifest_objects(manifest: &Value) -> BTreeSet<String> {
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

fn manifest_name(manifest: &Value, sha: &str) -> String {
    let ts = manifest
        .get("generated_at_ms")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    format!("{ts:013}-{}.v1.json", &sha[..12])
}

// ---------------------------------------------------------------------------
// per-target publish
// ---------------------------------------------------------------------------

struct PublishOutcome {
    objects_uploaded: usize,
    bytes_uploaded: u64,
    objects_present: usize,
    published_sha256: String,
    gc_deleted: usize,
    /// Objects found corrupt by the scrub and re-uploaded on this cycle.
    objects_repaired: usize,
}

/// Upload counters that survive a mid-cycle failure, so the receipt can say how
/// far a broken target got instead of reporting a flat zero.
#[derive(Default)]
struct Progress {
    uploaded: usize,
    bytes: u64,
}

// Keep replica content, manifest identity and progress state explicit at publication.
#[allow(clippy::too_many_arguments)]
fn publish(
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
fn gc(target: &Target, rroot: &str, present: &BTreeSet<String>) -> Result<usize> {
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
// receipt
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         jope_training_replica replicate\n  \
         jope_training_replica verify\n  \
         jope_training_replica restore --target <node> --into <dir> [--compare]\n  \
         jope_training_replica status\n\n\
         env: JOPE_ROOT, JOPE_REPLICA_TARGETS (host[:port],...), JOPE_REPLICA_REMOTE_ROOT"
    );
    std::process::exit(2);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = root_from_env();
    match args.first().map(String::as_str) {
        Some("replicate") | None => {
            // The unit must never go red: a failed cycle is reported through
            // ops/training-replica.v1.json and the alarm scanner, which is what
            // anyone actually watches.
            match commands::cmd_replicate(&root) {
                Ok(report) => println!("{}", serde_json::to_string_pretty(&report)?),
                Err(err) => {
                    let mut last_success_ms = Value::Null;
                    let error = match commands::load_status(&root) {
                        Ok(status) => {
                            last_success_ms = match status.get("last_success_ms") {
                                Some(value) => value.clone(),
                                None => Value::Null,
                            };
                            format!("{err:#}")
                        }
                        Err(status_err) => {
                            format!("{err:#}; status receipt unavailable: {status_err:#}")
                        }
                    };
                    let report = json!({
                        "schema": STATUS_SCHEMA,
                        "generated_at_ms": now_ms(),
                        "source_node": SOURCE_NODE,
                        "ok": false,
                        "healthy_targets": 0,
                        "target_count": 0,
                        "error": error,
                        "last_success_ms": last_success_ms,
                        "targets": [],
                    });
                    let _ = commands::write_status(&root, &report);
                    commands::append_event(&root, &report);
                    eprintln!("replicate failed: {err:#}");
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
            }
            Ok(())
        }
        Some("verify") => {
            let report = commands::cmd_verify(&root)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let healthy = report
                .get("healthy_targets")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if healthy == 0 {
                std::process::exit(1);
            }
            Ok(())
        }
        Some("restore") => {
            let mut target = None;
            let mut into = None;
            let mut compare = false;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--target" => {
                        target = args.get(i + 1).cloned();
                        i += 2;
                    }
                    "--into" => {
                        into = args.get(i + 1).cloned();
                        i += 2;
                    }
                    "--compare" => {
                        compare = true;
                        i += 1;
                    }
                    _ => usage(),
                }
            }
            let (Some(target), Some(into)) = (target, into) else {
                usage()
            };
            let report = commands::cmd_restore(&root, &target, Path::new(&into), compare)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let clean = report.get("hash_mismatches").and_then(Value::as_u64) == Some(0)
                && report.get("db_integrity_check").and_then(Value::as_str) == Some("ok")
                && report.get("rows_match_manifest").and_then(Value::as_bool) == Some(true);
            if !clean {
                std::process::exit(1);
            }
            Ok(())
        }
        Some("status") => {
            println!(
                "{}",
                serde_json::to_string_pretty(&commands::load_status(&root)?)?
            );
            Ok(())
        }
        _ => usage(),
    }
}
