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
#[path = "jope_training_replica/commands.rs"]
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
#[path = "jope_training_replica/local.rs"]
mod local;
use local::*;
#[path = "jope_training_replica/remote.rs"]
mod remote;
use remote::*;
#[path = "jope_training_replica/publish.rs"]
mod publish;
use publish::*;

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
