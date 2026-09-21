//! Append-only, hash-chained JSONL ledgers.
//!
//! A batch is `<ledger>/<yyyy>/<mm>/<ulid>.jsonl`: one compact JSON event per line, then one
//! seal line (`jope.ledger-seal/v1`) carrying the sha256 of the event lines, the previous
//! batch's ULID and the sha256 of the previous batch *file*. `HEAD.json` names the tip. A batch is
//! written to a temporary file, fsynced and renamed; nothing is ever rewritten. `verify` walks the
//! batches in ULID order and recomputes every hash, so an *edited*, reordered, duplicated or
//! mid-chain-missing batch shows up as a break, as does a `HEAD.json` that disagrees with the tip
//! it names in batch, file hash, path or counts.
//!
//! What it does **not** detect is a tail that was removed *and* `HEAD.json` rewritten to match:
//! delete the newest batches, point `HEAD.json` at the new last one with its real sha and
//! corrected counts, and the remaining chain is internally consistent, because `HEAD.json` is the
//! only anchor and it lives in the directory it anchors. No hash inside the tree can notice
//! history that is no longer in the tree. Detecting it needs an anchor from outside: today that is
//! the mirror — [`Ledger::mirror_position`] refuses a mirror whose tip this host's chain no longer
//! contains, so a truncated host cannot push and is told why — or any external record of the tip.
//! `verify` reporting zero breaks means "everything here is consistent with everything else here",
//! which is the strongest thing a self-contained chain can say, and is stated that way rather than
//! as an integrity guarantee it cannot make.

use crate::canon::sha256_hex;
use crate::civil::year_month;
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const SEAL_SCHEMA: &str = "jope.ledger-seal/v1";
pub const HEAD_SCHEMA: &str = "jope.ledger-head/v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Seal {
    pub schema: String,
    pub ledger: String,
    pub batch: String,
    pub events: u64,
    pub events_sha256: String,
    pub prev_batch: Option<String>,
    pub prev_batch_sha256: Option<String>,
    pub source: Value,
    pub tool: String,
    pub ts_unix: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub schema: String,
    pub ledger: String,
    pub batch: String,
    /// Batch file path relative to the ledger directory.
    pub path: String,
    pub sha256: String,
    pub batches: u64,
    pub events_total: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct BatchReceipt {
    pub batch: String,
    pub path: String,
    pub sha256: String,
    pub events: u64,
    pub prev_batch_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct LedgerReport {
    pub ledger: String,
    pub batches: u64,
    pub events: u64,
    pub head: Option<Head>,
    pub breaks: Vec<Break>,
}

impl LedgerReport {
    pub fn ok(&self) -> bool {
        self.breaks.is_empty()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Break {
    pub batch: String,
    pub kind: String,
    pub detail: String,
}

/// The result of comparing a mirror's `HEAD.json` against the local chain
/// ([`Ledger::mirror_position`]).
#[derive(Clone, Debug)]
pub enum MirrorPosition {
    /// The mirror's tip is this chain's tip (or both are empty): nothing to backfill.
    UpToDate,
    /// The mirror's tip is a batch of this chain, or the mirror has no `HEAD.json` at all.
    /// `missing` is every local batch after it, oldest first — the gap to upload before the
    /// pointer moves.
    Behind { missing: Vec<(String, PathBuf)> },
    /// The mirror names a chain this host cannot continue. Refuse; do not overwrite.
    Diverged { detail: String },
}

pub struct Ledger {
    root: PathBuf,
    name: String,
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Write `bytes` to `path` through a sibling temporary file, fsync, rename.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic write of a root path")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        std::process::id()
    ));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    if let Ok(dir) = fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

impl Ledger {
    /// `<state>/<name>/ledger`.
    pub fn open(state: &Path, name: &str) -> Ledger {
        Ledger {
            root: state.join(name).join("ledger"),
            name: name.to_string(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn head(&self) -> Result<Option<Head>> {
        let p = self.root.join("HEAD.json");
        if !p.exists() {
            return Ok(None);
        }
        let head: Head = serde_json::from_slice(&fs::read(&p)?)
            .with_context(|| format!("parse {}", p.display()))?;
        ensure!(head.schema == HEAD_SCHEMA, "HEAD.json schema mismatch");
        Ok(Some(head))
    }

    /// Append one sealed batch. Serialises against a ledger-wide lock.
    pub fn append(&self, events: &[Value], source: Value) -> Result<BatchReceipt> {
        ensure!(!events.is_empty(), "refusing to seal an empty batch");
        fs::create_dir_all(&self.root)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join(".lock"))?;
        lock.lock()?;
        let head_path = self.root.join("HEAD.json");
        let head = self.head()?;
        let ulid = ulid::Ulid::new().to_string();
        let ts = now_unix();
        let (yyyy, mm) = year_month(ts);
        let rel = format!("{yyyy}/{mm}/{ulid}.jsonl");
        let mut body = Vec::with_capacity(events.len() * 1024);
        for e in events {
            serde_json::to_writer(&mut body, e)?;
            body.push(b'\n');
        }
        let seal = Seal {
            schema: SEAL_SCHEMA.into(),
            ledger: self.name.clone(),
            batch: ulid.clone(),
            events: events.len() as u64,
            events_sha256: sha256_hex(&body),
            prev_batch: head.as_ref().map(|h| h.batch.clone()),
            prev_batch_sha256: head.as_ref().map(|h| h.sha256.clone()),
            source,
            tool: crate::TOOL_VERSION.into(),
            ts_unix: ts,
        };
        serde_json::to_writer(&mut body, &seal)?;
        body.push(b'\n');
        let path = self.root.join(&rel);
        ensure!(!path.exists(), "batch {rel} already exists");
        atomic_write(&path, &body)?;
        let sha = sha256_hex(&body);
        let new_head = Head {
            schema: HEAD_SCHEMA.into(),
            ledger: self.name.clone(),
            batch: ulid.clone(),
            path: rel.clone(),
            sha256: sha.clone(),
            batches: head.as_ref().map_or(0, |h| h.batches) + 1,
            events_total: head.as_ref().map_or(0, |h| h.events_total) + events.len() as u64,
        };
        atomic_write(&head_path, &serde_json::to_vec_pretty(&new_head)?)?;
        lock.unlock()?;
        Ok(BatchReceipt {
            batch: ulid,
            path: rel,
            sha256: sha,
            events: events.len() as u64,
            prev_batch_sha256: seal.prev_batch_sha256,
        })
    }

    /// Every batch file, relative path and absolute path, in ULID (= time) order.
    pub fn batches(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        for entry in walkdir::WalkDir::new(&self.root).min_depth(3).max_depth(3) {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let rel = p
                .strip_prefix(&self.root)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, p.to_path_buf()));
        }
        // ULIDs are lexicographically time-ordered; the yyyy/mm prefix agrees with them.
        out.sort_by(|a, b| {
            let stem = |r: &str| r.rsplit('/').next().unwrap_or(r).to_string();
            stem(&a.0).cmp(&stem(&b.0))
        });
        Ok(out)
    }

    /// Parse one batch: (events, seal, file sha256, recomputed events sha256).
    pub fn read_batch(path: &Path) -> Result<(Vec<Value>, Seal, String, String)> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let file_sha = sha256_hex(&bytes);
        let text = std::str::from_utf8(&bytes).context("batch is not UTF-8")?;
        ensure!(text.ends_with('\n'), "batch does not end with a newline");
        let last_start = text[..text.len() - 1].rfind('\n').map_or(0, |i| i + 1);
        let seal: Seal = serde_json::from_str(text[last_start..].trim_end())
            .with_context(|| format!("seal of {}", path.display()))?;
        ensure!(seal.schema == SEAL_SCHEMA, "seal schema mismatch");
        let body = &bytes[..last_start];
        let events_sha = sha256_hex(body);
        // `seal.events` is a `u64` read out of the file being checked for tampering, so it is not
        // an allocation size until it has been reconciled with what the file can actually hold.
        // Every event is at least one byte and its newline, so the body bounds the count outright;
        // a seal claiming `u64::MAX` used to reach `Vec::with_capacity` and abort the process with
        // `capacity overflow`, which walks straight past `verify`'s careful conversion of a
        // `read_batch` failure into a `Break` — the one command whose job is to report corruption
        // died on it, and so did `require_intact` on the write path. Refusing here makes it an
        // ordinary `unreadable` break, like every other malformed batch.
        let max_events = (body.len() / 2) as u64;
        ensure!(
            seal.events <= max_events,
            "seal of {} declares {} events, which {} bytes of batch body cannot hold (at most {})",
            path.display(),
            seal.events,
            body.len(),
            max_events
        );
        let mut events = Vec::with_capacity(seal.events as usize);
        for (i, line) in text[..last_start].lines().enumerate() {
            events.push(
                serde_json::from_str(line)
                    .with_context(|| format!("{} line {}", path.display(), i + 1))?,
            );
        }
        Ok((events, seal, file_sha, events_sha))
    }

    /// Every event of every batch that *can* be read, skipping the batches that cannot.
    ///
    /// Only for the verifiers. [`Ledger::verify`] already records an `unreadable` break for each
    /// batch it could not parse, and the tool whose job is to diagnose a broken ledger is the one
    /// tool that must not stop at the first break — a forged seal in batch 1 should not hide what
    /// is wrong with batches 2 to 40. Every other caller uses [`Ledger::events`] and fails closed,
    /// because a partial read is not a ledger and must never be mistaken for one.
    pub fn events_readable(&self) -> Result<Vec<Value>> {
        let mut all = Vec::new();
        for (_, p) in self.batches()? {
            if let Ok((events, _, _, _)) = Ledger::read_batch(&p) {
                all.extend(events);
            }
        }
        Ok(all)
    }

    /// All events across all batches in chain order (does not verify).
    pub fn events(&self) -> Result<Vec<Value>> {
        let mut all = Vec::new();
        for (_, p) in self.batches()? {
            let (events, _, _, _) = Ledger::read_batch(&p)?;
            all.extend(events);
        }
        Ok(all)
    }

    /// Walk the chain, recomputing every hash.
    ///
    /// Self-contained: it compares the tree against itself. See the module docs for the one thing
    /// that cannot detect — a removed tail with `HEAD.json` rewritten to match — and what does.
    pub fn verify(&self) -> Result<LedgerReport> {
        let mut report = LedgerReport {
            ledger: self.name.clone(),
            ..Default::default()
        };
        let batches = self.batches()?;
        let head = match self.head() {
            Ok(h) => h,
            Err(e) => {
                report.breaks.push(Break {
                    batch: "HEAD".into(),
                    kind: "head-unreadable".into(),
                    detail: e.to_string(),
                });
                None
            }
        };
        let mut prev: Option<(String, String)> = None;
        let mut last_rel = None;
        for (rel, path) in &batches {
            let (events, seal, file_sha, events_sha) = match Ledger::read_batch(path) {
                Ok(x) => x,
                Err(e) => {
                    report.breaks.push(Break {
                        batch: rel.clone(),
                        kind: "unreadable".into(),
                        detail: e.to_string(),
                    });
                    continue;
                }
            };
            let stem = rel
                .rsplit('/')
                .next()
                .unwrap_or(rel)
                .trim_end_matches(".jsonl");
            let mut brk = |kind: &str, detail: String| {
                report.breaks.push(Break {
                    batch: rel.clone(),
                    kind: kind.into(),
                    detail,
                })
            };
            if seal.batch != stem {
                brk(
                    "batch-name",
                    format!("seal names {} in file {stem}", seal.batch),
                );
            }
            if seal.ledger != self.name {
                brk("ledger-name", format!("seal ledger {}", seal.ledger));
            }
            if seal.events != events.len() as u64 {
                brk(
                    "event-count",
                    format!(
                        "seal says {} events, file has {}",
                        seal.events,
                        events.len()
                    ),
                );
            }
            if seal.events_sha256 != events_sha {
                brk(
                    "events-sha256",
                    format!("seal {} != recomputed {events_sha}", seal.events_sha256),
                );
            }
            match (&prev, &seal.prev_batch, &seal.prev_batch_sha256) {
                (None, None, None) => {}
                (None, pb, ps) => brk(
                    "prev-link",
                    format!("first batch names a previous batch {pb:?} {ps:?}"),
                ),
                (Some((pb, ps)), Some(sb), Some(ss)) => {
                    if pb != sb || ps != ss {
                        brk(
                            "prev-link",
                            format!("seal prev {sb}/{ss} != previous batch {pb}/{ps}"),
                        );
                    }
                }
                (Some((pb, _)), _, _) => {
                    brk(
                        "prev-link",
                        format!("batch after {pb} has no previous link"),
                    );
                }
            }
            report.batches += 1;
            report.events += events.len() as u64;
            prev = Some((seal.batch.clone(), file_sha));
            last_rel = Some(rel.clone());
        }
        match (&head, &prev) {
            (None, None) => {}
            (None, Some((b, _))) => report.breaks.push(Break {
                batch: b.clone(),
                kind: "head-missing".into(),
                detail: "batches exist but HEAD.json is absent".into(),
            }),
            (Some(h), None) => report.breaks.push(Break {
                batch: h.batch.clone(),
                kind: "head-dangling".into(),
                detail: "HEAD.json names a batch but none exist".into(),
            }),
            (Some(h), Some((b, s))) => {
                if &h.batch != b || &h.sha256 != s {
                    report.breaks.push(Break {
                        batch: h.batch.clone(),
                        kind: "head-mismatch".into(),
                        detail: format!("HEAD {}/{} != tip {b}/{s}", h.batch, h.sha256),
                    });
                }
                if Some(&h.path) != last_rel.as_ref() {
                    report.breaks.push(Break {
                        batch: h.batch.clone(),
                        kind: "head-path".into(),
                        detail: format!("HEAD path {} != tip path {last_rel:?}", h.path),
                    });
                }
                if h.batches != report.batches || h.events_total != report.events {
                    report.breaks.push(Break {
                        batch: h.batch.clone(),
                        kind: "head-counts".into(),
                        detail: format!(
                            "HEAD counts {}/{} != walked {}/{}",
                            h.batches, h.events_total, report.batches, report.events
                        ),
                    });
                }
            }
        }
        report.head = head;
        Ok(report)
    }

    /// Where a mirror's `HEAD.json` sits relative to *this host's* chain.
    ///
    /// This is the whole of the push path's safety argument, so it is deliberately narrow: the
    /// only thing that may be caught up is a mirror whose tip is a batch **this host already
    /// has**, at the position and with the file hash the mirror's own `HEAD.json` claims.
    /// Anything else — a tip this host cannot produce, a tip whose bytes differ, a tip that
    /// claims a different depth — is a real divergence and is refused. `head_bytes` is a fresh
    /// read of the key that is about to be written; using it as the expected-previous value is
    /// only sound *because* this function proves it is an ancestor of the local chain first.
    pub fn mirror_position(&self, head_bytes: Option<&[u8]>) -> Result<MirrorPosition> {
        let batches = self.batches()?;
        let diverged = |detail: String| Ok(MirrorPosition::Diverged { detail });
        let Some(bytes) = head_bytes else {
            // The mirror has no HEAD for this ledger: everything local is missing.
            return Ok(if batches.is_empty() {
                MirrorPosition::UpToDate
            } else {
                MirrorPosition::Behind { missing: batches }
            });
        };
        let head: Head = match serde_json::from_slice(bytes) {
            Ok(h) => h,
            Err(e) => {
                return diverged(format!(
                    "its HEAD.json does not parse as {HEAD_SCHEMA}: {e}"
                ))
            }
        };
        if head.schema != HEAD_SCHEMA {
            return diverged(format!("its HEAD.json has schema {}", head.schema));
        }
        if head.ledger != self.name {
            return diverged(format!(
                "its HEAD.json is for ledger {}, not {}",
                head.ledger, self.name
            ));
        }
        if head.path.rsplit('/').next().unwrap_or(&head.path) != format!("{}.jsonl", head.batch) {
            return diverged(format!(
                "its HEAD.json names batch {} at path {}",
                head.batch, head.path
            ));
        }
        let Some(i) = batches.iter().position(|(rel, _)| rel == &head.path) else {
            return diverged(format!(
                "its tip {} is not a batch of this host's {} chain (local tip {})",
                head.batch,
                self.name,
                batches
                    .last()
                    .map_or("(empty)".to_string(), |(rel, _)| rel.clone())
            ));
        };
        let sha = sha256_hex(&fs::read(&batches[i].1)?);
        if sha != head.sha256 {
            return diverged(format!(
                "its tip {} hashes to {} there and {sha} here",
                head.batch, head.sha256
            ));
        }
        if head.batches != (i + 1) as u64 {
            return diverged(format!(
                "its tip {} is batch {} of this host's chain but its HEAD.json counts {}",
                head.batch,
                i + 1,
                head.batches
            ));
        }
        Ok(if i + 1 == batches.len() {
            MirrorPosition::UpToDate
        } else {
            MirrorPosition::Behind {
                missing: batches[i + 1..].to_vec(),
            }
        })
    }

    /// Fail unless the chain verifies (used before any append in strict mode).
    pub fn require_intact(&self) -> Result<LedgerReport> {
        let r = self.verify()?;
        if !r.ok() {
            bail!(
                "{} ledger chain broken: {}",
                self.name,
                r.breaks
                    .iter()
                    .map(|b| format!("{}:{}:{}", b.batch, b.kind, b.detail))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        Ok(r)
    }
}

pub fn map_err_walk(e: walkdir::Error) -> anyhow::Error {
    anyhow!("{e}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chain_appends_and_verifies_and_detects_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path(), "hwm");
        assert!(ledger.head().unwrap().is_none());
        assert!(ledger.verify().unwrap().ok());
        let r1 = ledger
            .append(&[json!({"n": 1}), json!({"n": 2})], json!({"t": "a"}))
            .unwrap();
        let r2 = ledger
            .append(&[json!({"n": 3})], json!({"t": "b"}))
            .unwrap();
        let r3 = ledger
            .append(&[json!({"n": 4})], json!({"t": "c"}))
            .unwrap();
        assert_eq!(r2.prev_batch_sha256.as_deref(), Some(r1.sha256.as_str()));
        assert_eq!(r3.prev_batch_sha256.as_deref(), Some(r2.sha256.as_str()));
        let report = ledger.verify().unwrap();
        assert!(report.ok(), "{:?}", report.breaks);
        assert_eq!((report.batches, report.events), (3, 4));
        assert_eq!(
            ledger
                .events()
                .unwrap()
                .iter()
                .map(|e| e["n"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(ledger.append(&[], json!({})).is_err());

        // Tamper with one byte of an event in the middle batch.
        let p2 = ledger.root().join(&r2.path);
        let mut bytes = fs::read(&p2).unwrap();
        let i = bytes.iter().position(|&b| b == b'3').unwrap();
        bytes[i] = b'9';
        fs::write(&p2, &bytes).unwrap();
        let report = ledger.verify().unwrap();
        let kinds: Vec<&str> = report.breaks.iter().map(|b| b.kind.as_str()).collect();
        assert!(kinds.contains(&"events-sha256"), "{kinds:?}");
        assert!(kinds.contains(&"prev-link"), "{kinds:?}");
        assert!(ledger.require_intact().is_err());

        // Restore, then drop the tip: HEAD dangles.
        fs::write(&p2, {
            bytes[i] = b'3';
            &bytes
        })
        .unwrap();
        assert!(ledger.verify().unwrap().ok());
        fs::remove_file(ledger.root().join(&r3.path)).unwrap();
        let kinds: Vec<String> = ledger
            .verify()
            .unwrap()
            .breaks
            .into_iter()
            .map(|b| b.kind)
            .collect();
        assert!(kinds.contains(&"head-mismatch".to_string()), "{kinds:?}");
    }
}
