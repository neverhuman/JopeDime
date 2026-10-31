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
//! An append is two renames: the batch's, then `HEAD.json`'s. A crash between them leaves a
//! sealed batch that the head does not name. [`Ledger::recover`] rolls such a head forward, and
//! so do [`Ledger::append`] and [`Ledger::require_intact`]. It does so only onto an exact,
//! sealed continuation of the chain the head names ([`Ledger::lagging_head`]). Anything else
//! remains a break.
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

/// Schema identifier for seal values.
pub const SEAL_SCHEMA: &str = "jope.ledger-seal/v1";
/// Schema identifier for head values.
pub const HEAD_SCHEMA: &str = "jope.ledger-head/v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Seal` is a record for the seal contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Seal {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Ledger recorded for this seal.
    pub ledger: String,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Events recorded for this seal.
    pub events: u64,
    /// SHA-256 digest identifying the associated events.
    pub events_sha256: String,
    /// Optional prev batch value; `None` means it is not available.
    pub prev_batch: Option<String>,
    /// SHA-256 digest identifying the associated prev batch.
    pub prev_batch_sha256: Option<String>,
    /// Provenance source for this value.
    pub source: Value,
    /// Tool recorded for this seal.
    pub tool: String,
    /// Unix timestamp when this event was recorded.
    pub ts_unix: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Head` is a record for the head contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Head {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Ledger recorded for this head.
    pub ledger: String,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Batch file path relative to the ledger directory.
    pub path: String,
    /// Sha-256 recorded for this head.
    pub sha256: String,
    /// Batches recorded for this head.
    pub batches: u64,
    /// Events total recorded for this head.
    pub events_total: u64,
}

#[derive(Clone, Debug, Serialize)]
/// `BatchReceipt` is a record for the batch receipt contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct BatchReceipt {
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Filesystem path associated with this input or output.
    pub path: String,
    /// Sha-256 recorded for this batch receipt.
    pub sha256: String,
    /// Events recorded for this batch receipt.
    pub events: u64,
    /// SHA-256 digest identifying the associated prev batch.
    pub prev_batch_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
/// `LedgerReport` is a record for the ledger report contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct LedgerReport {
    /// Ledger recorded for this ledger report.
    pub ledger: String,
    /// Batches recorded for this ledger report.
    pub batches: u64,
    /// Events recorded for this ledger report.
    pub events: u64,
    /// Optional head value; `None` means it is not available.
    pub head: Option<Head>,
    /// Collection of breaks values carried by this ledger report.
    pub breaks: Vec<Break>,
}

impl LedgerReport {
    /// Performs the ok operation and returns an error if it cannot be completed.
    pub fn ok(&self) -> bool {
        self.breaks.is_empty()
    }
}

#[derive(Clone, Debug, Serialize)]
/// `Break` is a record for the break contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Break {
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: String,
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Detail recorded for this break.
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
    /// Field value recorded by this public API type.
    Behind {
        /// Missing recorded for this outcome.
        missing: Vec<(String, PathBuf)>,
    },
    /// The mirror names a chain this host cannot continue. Refuse; do not overwrite.
    /// Field value recorded by this public API type.
    Diverged {
        /// Explanation accompanying this outcome.
        detail: String,
    },
}

/// `Ledger` is a record for the ledger contract in this module: Append-only, hash-chained JSONL ledgers.
pub struct Ledger {
    root: PathBuf,
    name: String,
}

/// Performs the now unix operation and returns an error if it cannot be completed.
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

    /// Performs the root operation and returns an error if it cannot be completed.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Performs the name operation and returns an error if it cannot be completed.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Performs the head operation and returns an error if it cannot be completed.
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

    fn lock(&self) -> Result<fs::File> {
        fs::create_dir_all(&self.root)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join(".lock"))?;
        lock.lock()?;
        Ok(lock)
    }

    /// The `HEAD.json` this ledger should have, when the one on disk lags a sealed continuation
    /// of the chain it names. That is the only state an interrupted [`Ledger::append`] can leave:
    /// the batch's atomic rename completed and `HEAD.json`'s did not. Otherwise `None`. Read-only.
    ///
    /// A head is rolled forward only when all of these hold:
    /// - `HEAD.json` names a batch of this chain at the path, position, file hash and batch count
    ///   it claims, or it is absent;
    /// - at least one batch follows it;
    /// - every following batch is readable, sealed for this ledger under its own name, carries
    ///   the event count and hash its seal claims, and links to its predecessor's id and file
    ///   hash.
    ///
    /// Anything else stays a break for [`Ledger::verify`] to report. A batch file exists only
    /// once its rename has completed, and every reader ([`Ledger::events`]) already counts it as
    /// part of the ledger. Rolling the head forward therefore makes the anchor agree with what is
    /// already read; it admits nothing new. It cannot see a removed tail any better than before
    /// (see the module docs).
    pub fn lagging_head(&self) -> Result<Option<Head>> {
        let batches = self.batches()?;
        let Ok(head) = self.head() else {
            return Ok(None);
        };
        let (start, mut prev, mut batch_count, mut event_count) = match &head {
            None => (0, None, 0, 0),
            Some(h) => {
                let Some(i) = batches.iter().position(|(rel, _)| *rel == h.path) else {
                    return Ok(None);
                };
                let sha = sha256_hex(&fs::read(&batches[i].1)?);
                if sha != h.sha256
                    || h.batches != (i + 1) as u64
                    || h.path.rsplit('/').next() != Some(format!("{}.jsonl", h.batch).as_str())
                {
                    return Ok(None);
                }
                (
                    i + 1,
                    Some((h.batch.clone(), sha)),
                    h.batches,
                    h.events_total,
                )
            }
        };
        let mut tip = None;
        for (rel, path) in batches.iter().skip(start) {
            let Ok((events, seal, file_sha, events_sha)) = Ledger::read_batch(path) else {
                return Ok(None);
            };
            let stem = rel
                .rsplit('/')
                .next()
                .unwrap_or(rel)
                .trim_end_matches(".jsonl");
            let linked = match &prev {
                None => seal.prev_batch.is_none() && seal.prev_batch_sha256.is_none(),
                Some((b, s)) => {
                    seal.prev_batch.as_ref() == Some(b)
                        && seal.prev_batch_sha256.as_ref() == Some(s)
                }
            };
            if !linked
                || seal.batch != stem
                || seal.ledger != self.name
                || seal.events != events.len() as u64
                || seal.events_sha256 != events_sha
            {
                return Ok(None);
            }
            batch_count += 1;
            event_count += events.len() as u64;
            tip = Some(Head {
                schema: HEAD_SCHEMA.into(),
                ledger: self.name.clone(),
                batch: seal.batch.clone(),
                path: rel.clone(),
                sha256: file_sha.clone(),
                batches: batch_count,
                events_total: event_count,
            });
            prev = Some((seal.batch, file_sha));
        }
        Ok(tip)
    }

    /// Roll a lagging `HEAD.json` forward ([`Ledger::lagging_head`]). The caller holds the
    /// ledger lock. Returns the head it wrote, if any.
    fn roll_forward_locked(&self) -> Result<Option<Head>> {
        let Some(head) = self.lagging_head()? else {
            return Ok(None);
        };
        atomic_write(
            &self.root.join("HEAD.json"),
            &serde_json::to_vec_pretty(&head)?,
        )?;
        Ok(Some(head))
    }

    /// Recover from an interrupted append: roll a lagging `HEAD.json` forward under the ledger
    /// lock. Writes nothing (and takes no lock) when the head does not lag.
    pub fn recover(&self) -> Result<Option<Head>> {
        if self.lagging_head()?.is_none() {
            return Ok(None);
        }
        let _lock = self.lock()?;
        self.roll_forward_locked()
    }

    /// Append one sealed batch. Serialises against a ledger-wide lock.
    ///
    /// Two durable steps: the batch's rename, then `HEAD.json`'s. A crash between them leaves a
    /// sealed batch the head does not name; the next append (or [`Ledger::require_intact`])
    /// rolls the head forward before chaining onto it, instead of forking the chain.
    pub fn append(&self, events: &[Value], source: Value) -> Result<BatchReceipt> {
        ensure!(!events.is_empty(), "refusing to seal an empty batch");
        let lock = self.lock()?;
        self.roll_forward_locked()?;
        let head_path = self.root.join("HEAD.json");
        let head = self.head()?;
        // ULIDs are lexicographically ordered only ACROSS milliseconds. Within one,
        // `Ulid::new()` fills the low 80 bits with random entropy, so two batches sealed in the
        // same millisecond sort randomly against each other -- measured on this machine, 9923 of
        // 19995 consecutive same-millisecond pairs came out lexicographically reversed. Since
        // `batches()` orders by exactly this name and `verify()` walks the prev-links in that
        // order, an intact ledger was reported `prev-link` BROKEN whenever two appends landed in
        // one millisecond: a false alarm from the one structure whose job is telling the truth
        // about tampering, and one that only appears on hardware fast enough to do two appends
        // inside a millisecond.
        //
        // Appends are serialised by the lock taken above, and `head` is the batch being chained
        // onto, so clamping against it makes each id strictly greater than its predecessor by
        // construction -- across processes, not merely within one -- which is what `batches()`
        // needs in order to be sorting by anything meaningful.
        let ulid = next_batch_id(head.as_ref().map(|h| h.batch.as_str()))?.to_string();
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
        // Batch ids are strictly increasing by construction (see `next_batch_id`), which is
        // what makes this sort the append order; the yyyy/mm prefix agrees with them.
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
    ///
    /// An interrupted append is recovered first ([`Ledger::recover`]): a head that lags a
    /// sealed continuation of its own chain is crash residue, not tampering. Refusing it forever
    /// wedged every later write of this ledger.
    pub fn require_intact(&self) -> Result<LedgerReport> {
        self.recover()?;
        let r = self.verify()?;
        if !r.ok() {
            let mut kinds: Vec<&str> = r.breaks.iter().map(|b| b.kind.as_str()).collect();
            kinds.sort_unstable();
            kinds.dedup();
            bail!(
                "{} ledger chain broken: {} break(s) [{}] across {} batch(es): {}",
                self.name,
                r.breaks.len(),
                kinds.join(", "),
                r.batches,
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

/// Performs the map err walk operation and returns an error if it cannot be completed.
pub fn map_err_walk(e: walkdir::Error) -> anyhow::Error {
    anyhow!("{e}")
}

/// The id for the next batch: a fresh ULID, or the previous id incremented when the clock has not
/// advanced past it.
///
/// Strictly greater than `prev` whenever `prev` parses, which is what `Ledger::batches()`'s
/// ordering depends on. Callers must hold the ledger lock, so `prev` really is the batch being
/// chained onto and no other writer can mint in between.
fn next_batch_id(prev: Option<&str>) -> Result<ulid::Ulid> {
    let fresh = ulid::Ulid::new();
    let Some(prev) = prev else { return Ok(fresh) };
    let Ok(prev) = ulid::Ulid::from_string(prev) else {
        // An id that does not parse cannot be ordered against. Returning the fresh one leaves
        // `verify()` to report the malformed chain rather than inventing an order for it here.
        return Ok(fresh);
    };
    if fresh > prev {
        return Ok(fresh);
    }
    prev.increment()
        .ok_or_else(|| anyhow!("batch id space exhausted after {prev}"))
}

#[cfg(test)]
mod tests {
    /// The ordering `Ledger::batches()` sorts by, pinned without depending on machine speed.
    ///
    /// A `prev` stamped in the future is one a fresh `Ulid::new()` cannot beat, so the clamp in
    /// `next_batch_id` is the only thing that can satisfy this -- no timing, no retries, no
    /// "usually passes". Reverting the clamp fails this test every run on every machine.
    #[test]
    fn a_batch_id_is_always_greater_than_the_one_it_chains_onto() {
        let now = ulid::Ulid::new();

        // Unreachable by the wall clock for the next minute.
        let future = ulid::Ulid::from_parts(now.timestamp_ms() + 60_000, 0);
        let next = next_batch_id(Some(&future.to_string())).unwrap();
        assert!(
            next > future,
            "{next} does not follow {future}; batches() would sort them backwards"
        );
        assert!(
            next.to_string() > future.to_string(),
            "ids order differently as strings than as ULIDs, and batches() sorts strings"
        );

        // Repeated against its own output: a run of appends inside one tick must keep climbing.
        let mut prev = future;
        for _ in 0..1000 {
            let cur = next_batch_id(Some(&prev.to_string())).unwrap();
            assert!(cur.to_string() > prev.to_string(), "{cur} <= {prev}");
            prev = cur;
        }

        // An ordinary prev, and the two degenerate inputs, must not panic.
        assert!(next_batch_id(Some(&now.to_string())).unwrap() > now);
        assert!(next_batch_id(None).is_ok());
        assert!(
            next_batch_id(Some("not-a-ulid")).is_ok(),
            "an unparseable id must leave verify() to report the malformed chain, not panic here"
        );
    }

    /// Batches sealed inside the same millisecond must still verify.
    ///
    /// `Ulid::new()` is not monotonic within a millisecond -- it randomises the low 80 bits -- so
    /// batch names sorted lexicographically came out in a different order than they were appended
    /// whenever the host was fast enough to seal two in one tick. `verify()` walks prev-links in
    /// that order, so it reported `prev-link` BREAK on a ledger nobody had touched. It surfaced as
    /// a CI failure on the NVMe runner while passing on slower disks, which is the worst shape for
    /// a bug in a tamper-evident structure: it accuses when it is fastest.
    #[test]
    fn batches_sealed_within_one_millisecond_still_verify() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path(), "hwm");

        // NOTE: whether this actually seals two batches inside one millisecond depends on how
        // fast the disk is -- on a slow one every append takes longer than a tick and the bug
        // cannot appear. So this is an end-to-end guard, not the regression test; the property
        // itself is pinned deterministically by
        // `a_batch_id_is_always_greater_than_the_one_it_chains_onto` below.
        let mut ids = Vec::new();
        for n in 0..64u64 {
            ids.push(
                ledger
                    .append(&[json!({"n": n})], json!({"t": n}))
                    .unwrap()
                    .batch,
            );
        }

        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(
            ids, sorted,
            "batch ids are not in append order, so batches() cannot sort by them"
        );
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            ids.len(),
            "two batches were given the same id"
        );

        let report = ledger.verify().unwrap();
        assert!(
            report.ok(),
            "an untouched ledger reported breaks: {:?}",
            report.breaks
        );
        assert_eq!((report.batches, report.events), (64, 64));

        // And the events come back in the order they were appended, which is the property the
        // name ordering exists to provide.
        assert_eq!(
            ledger
                .events()
                .unwrap()
                .iter()
                .map(|e| e["n"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            (0..64).collect::<Vec<_>>()
        );
    }

    use super::*;
    use serde_json::json;

    fn numbers(ledger: &Ledger) -> Vec<u64> {
        ledger
            .events()
            .unwrap()
            .iter()
            .map(|e| e["n"].as_u64().unwrap())
            .collect()
    }

    /// Append batches 1 and 2, then put back batch 1's `HEAD.json`. This is the state a process
    /// leaves when it dies after batch 2's atomic rename but before `HEAD.json`'s: a complete,
    /// sealed batch that the head does not name. Returns (ledger dir, head 1, head 2).
    fn crashed_between_batch_and_head() -> (tempfile::TempDir, Vec<u8>, Vec<u8>) {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path(), "hwm");
        ledger.append(&[json!({"n": 1})], json!({"t": 1})).unwrap();
        let head_1 = fs::read(ledger.root().join("HEAD.json")).unwrap();
        ledger.append(&[json!({"n": 2})], json!({"t": 2})).unwrap();
        let head_2 = fs::read(ledger.root().join("HEAD.json")).unwrap();
        fs::write(ledger.root().join("HEAD.json"), &head_1).unwrap();
        (dir, head_1, head_2)
    }

    /// W8-T06: a crash between a batch's rename and its `HEAD.json`'s must not wedge the ledger.
    ///
    /// Every reader already treats the batch as part of the ledger (`events()` walks batch
    /// files). `require_intact`, the gate in front of every write, reported `head-mismatch` and
    /// refused for ever, so each later `hwm ingest`, `scores append` and eligibility decision
    /// failed until someone repaired the tree by hand.
    #[test]
    fn a_crash_between_a_batch_and_its_head_is_rolled_forward_not_a_wedge() {
        let (dir, _, head_2) = crashed_between_batch_and_head();
        let ledger = Ledger::open(dir.path(), "hwm");
        assert_eq!(numbers(&ledger), [1, 2], "readers already see batch 2");
        let report = ledger
            .require_intact()
            .expect("a crash-lagged HEAD.json wedged the ledger");
        assert_eq!((report.batches, report.events), (2, 2));
        assert_eq!(
            fs::read(ledger.root().join("HEAD.json")).unwrap(),
            head_2,
            "the head rolled forward to exactly what the interrupted append would have written"
        );
        ledger.append(&[json!({"n": 3})], json!({"t": 3})).unwrap();
        let report = ledger.verify().unwrap();
        assert!(report.ok(), "{:?}", report.breaks);
        assert_eq!(numbers(&ledger), [1, 2, 3]);
    }

    /// The same crash, followed directly by another append, must not fork the chain: the next
    /// batch has to chain onto batch 2, not onto the stale head's batch 1.
    #[test]
    fn an_append_after_a_lagging_head_never_forks_the_chain() {
        let (dir, _, _) = crashed_between_batch_and_head();
        let ledger = Ledger::open(dir.path(), "hwm");
        ledger.append(&[json!({"n": 3})], json!({"t": 3})).unwrap();
        let report = ledger.verify().unwrap();
        assert!(
            report.ok(),
            "an append chained onto a stale head and forked the ledger: {:?}",
            report.breaks
        );
        assert_eq!((report.batches, report.events), (3, 3));
        assert_eq!(numbers(&ledger), [1, 2, 3]);
    }

    /// Only a complete, sealed continuation of the head is rolled forward. Anything else is still
    /// a break that refuses, and nothing is written.
    #[test]
    fn only_a_sealed_continuation_of_the_head_is_rolled_forward() {
        // The unnamed batch was altered after its rename.
        let (dir, head_1, _) = crashed_between_batch_and_head();
        let ledger = Ledger::open(dir.path(), "hwm");
        let (_, tip) = ledger.batches().unwrap().pop().unwrap();
        let mut bytes = fs::read(&tip).unwrap();
        let i = bytes.iter().position(|&b| b == b'2').unwrap();
        bytes[i] = b'7';
        fs::write(&tip, &bytes).unwrap();
        assert!(
            ledger.require_intact().is_err(),
            "an altered batch was adopted"
        );
        assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), head_1);

        // A batch that does not chain onto the head: its predecessor is missing.
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path(), "hwm");
        ledger.append(&[json!({"n": 1})], json!({})).unwrap();
        let head_1 = fs::read(ledger.root().join("HEAD.json")).unwrap();
        let second = ledger.append(&[json!({"n": 2})], json!({})).unwrap();
        ledger.append(&[json!({"n": 3})], json!({})).unwrap();
        fs::remove_file(ledger.root().join(&second.path)).unwrap();
        fs::write(ledger.root().join("HEAD.json"), &head_1).unwrap();
        assert!(
            ledger.require_intact().is_err(),
            "a gapped suffix was adopted"
        );
        assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), head_1);

        // A head that does not match the batch it names.
        let (dir, head_1, _) = crashed_between_batch_and_head();
        let ledger = Ledger::open(dir.path(), "hwm");
        let mut forged: Head = serde_json::from_slice(&head_1).unwrap();
        forged.sha256 = "0".repeat(64);
        let forged = serde_json::to_vec_pretty(&forged).unwrap();
        fs::write(ledger.root().join("HEAD.json"), &forged).unwrap();
        assert!(
            ledger.require_intact().is_err(),
            "a forged head was rolled forward"
        );
        assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), forged);

        // Positive control: the very first append died before any HEAD.json was written.
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path(), "hwm");
        ledger.append(&[json!({"n": 1})], json!({})).unwrap();
        let head = fs::read(ledger.root().join("HEAD.json")).unwrap();
        fs::remove_file(ledger.root().join("HEAD.json")).unwrap();
        assert!(ledger.require_intact().is_ok());
        assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), head);
    }

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
