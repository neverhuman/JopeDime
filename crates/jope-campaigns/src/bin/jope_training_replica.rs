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
    conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
        row.get(0)
    })
    .unwrap_or(-1)
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

fn load_status(root: &Path) -> Value {
    fs::read(root.join(STATUS_RELATIVE))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}))
}

/// A target that fails today must still report when it last worked, otherwise
/// "no successes ever" and "no successes for six hours" look identical.
fn previous_success(previous: &Value, label: &str) -> u64 {
    previous
        .get("targets")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .find(|t| t.get("target").and_then(Value::as_str) == Some(label))
        .and_then(|t| t.get("last_success_ms").and_then(Value::as_u64))
        .unwrap_or(0)
}

fn write_status(root: &Path, report: &Value) -> Result<()> {
    let path = root.join(STATUS_RELATIVE);
    th::write_atomic(&path, &serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn append_event(root: &Path, record: &Value) {
    let dir = replica_dir(root);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("replica-events.v1.jsonl"))
    {
        let _ = writeln!(file, "{record}");
    }
}

// ---------------------------------------------------------------------------
// replicate
// ---------------------------------------------------------------------------

fn cmd_replicate(root: &Path) -> Result<Value> {
    let archive = th::archive_root(root);
    if !archive.join(th::INDEX_FILE).exists() {
        bail!("no archive at {} — nothing to replicate", archive.display());
    }
    let replica = replica_dir(root);
    fs::create_dir_all(&replica)?;
    let targets = targets_from_env()?;
    let rroot = remote_root();
    let previous = load_status(root);
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

fn cmd_verify(root: &Path) -> Result<Value> {
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

fn cmd_restore(root: &Path, target_name: &str, into: &Path, compare: bool) -> Result<Value> {
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
            match cmd_replicate(&root) {
                Ok(report) => println!("{}", serde_json::to_string_pretty(&report)?),
                Err(err) => {
                    let report = json!({
                        "schema": STATUS_SCHEMA,
                        "generated_at_ms": now_ms(),
                        "source_node": SOURCE_NODE,
                        "ok": false,
                        "healthy_targets": 0,
                        "target_count": 0,
                        "error": format!("{err:#}"),
                        "last_success_ms": load_status(&root).get("last_success_ms").and_then(Value::as_u64).unwrap_or(0),
                        "targets": [],
                    });
                    let _ = write_status(&root, &report);
                    append_event(&root, &report);
                    eprintln!("replicate failed: {err:#}");
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
            }
            Ok(())
        }
        Some("verify") => {
            let report = cmd_verify(&root)?;
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
            let report = cmd_restore(&root, &target, Path::new(&into), compare)?;
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
            println!("{}", serde_json::to_string_pretty(&load_status(&root))?);
            Ok(())
        }
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_parse_with_and_without_ports() {
        let parsed = parse_targets("xbabe3,xbabe4:2224").unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].node, "xbabe3");
        assert_eq!(parsed[0].port, 22);
        assert_eq!(parsed[1].port, 2224);
        assert_eq!(parsed[1].label(), "xbabe4:2224");
    }

    /// Replicating to the source node would produce a "replica" on the very
    /// disk this exists to escape, and the `xbabe2` ssh alias does not even
    /// reach this machine.
    #[test]
    fn source_node_is_rejected_as_a_target() {
        assert!(parse_targets("xbabe2").is_err());
        assert!(parse_targets("xbabe3,xbabe2").is_err());
    }

    #[test]
    fn live_database_and_sidecars_are_never_part_of_the_payload() {
        assert!(skipped("history.sqlite3"));
        assert!(skipped("history.sqlite3-wal"));
        assert!(skipped("history.sqlite3-shm"));
        assert!(skipped(".capture.lock"));
        assert!(skipped(".index.v1.json.1234.tmp"));
        assert!(!skipped("index.v1.json"));
        assert!(!skipped("train_metrics.jsonl"));
    }

    #[test]
    fn manifest_objects_covers_payload_and_database() {
        let db_sha = "a".repeat(64);
        let file_sha = "b".repeat(64);
        let manifest = json!({
            "db": {"sha256": db_sha},
            "files": [{"path": "x", "sha256": file_sha}],
        });
        let objects = manifest_objects(&manifest);
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&db_sha));
        assert!(objects.contains(&file_sha));
    }

    /// Manifest names must sort chronologically, because remote GC decides what
    /// to retain by taking the lexical tail of `ls -1`.
    #[test]
    fn manifest_names_sort_in_time_order() {
        let sha = "c".repeat(64);
        let early = manifest_name(&json!({"generated_at_ms": 1785213126163u64}), &sha);
        let later = manifest_name(&json!({"generated_at_ms": 1785299526163u64}), &sha);
        assert!(early < later, "{early} !< {later}");
    }

    #[test]
    fn object_paths_fan_out_by_hash_prefix() {
        let sha = format!("ab{}", "0".repeat(62));
        assert_eq!(object_rel(&sha), format!("ab/{sha}"));
    }

    #[test]
    fn only_lowercase_hex_digests_are_accepted() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256(&"A".repeat(64)));
        assert!(!is_sha256(&"a".repeat(63)));
        assert!(!is_sha256("not-a-hash"));
    }

    #[test]
    fn shell_quoting_survives_a_single_quote() {
        assert_eq!(quote("plain"), "'plain'");
        assert_eq!(quote("it's"), "'it'\\''s'");
    }
}
