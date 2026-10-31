//! W8 product integration: the engine's durable events as W8 progress objects.
//!
//! As with the champion and genebank sources, the learning store maps the engine's *published
//! bytes* at its own receiving path: the engine (`jope-parallel`) keeps no dependency on the
//! learning store, and nothing here trusts an in-process producer's claim. Every mapping reads
//! each file exactly once, checks those bytes against the invariants the engine itself publishes
//! them under, and builds the W8 objects from exactly the bytes it checked. The same bytes then
//! go to the outbox as blobs, so what is published is what was verified.
//!
//! | Engine event | Engine source | W8 objects |
//! |---|---|---|
//! | a no-replace publication | `native_persistence::publish_staged` (loop records, worker results, champions): a `{path, sha256}` artifact | the verified result blob of a `completed` event ([`result`]) |
//! | a published teaching corpus | `corpus_publication::build` (`parallel.teaching_corpus.v1`) | a [`CorpusManifest`] whose members are the corpus payload files ([`corpus`]) |
//! | a published checkpoint | `training::checkpoint_commit::commit`, named by the run's `latest.json` (`parallel.checkpoint.v1`) | an [`Exposure`] and a [`Checkpoint`] at the next consecutive update index ([`checkpoint`]) |
//! | a checkpoint recovery | `checkpoint_recovery::reconcile` (`parallel.checkpoint_recovery.v1` receipt) | a retryable `worker-lost` failure of the interrupted attempt ([`recovery`]) |
//!
//! What is refused, with nothing appended anywhere:
//! - a corpus that is still staging (`.<name>.staging`), retired (`<name>.retired-…`),
//!   diagnostic (`qualification: false`), or whose payload, receipt or publication request does
//!   not bind together exactly as `corpus_publication::verify` requires;
//! - a checkpoint that `latest.json` does not name (an unpublished or quarantined one), one whose
//!   manifest or objects differ from their digests, one bound to another corpus, and one whose
//!   step does not advance past the last mapped checkpoint;
//! - a recovery receipt that is incomplete or moved nothing, and an event for work the run
//!   manifest did not plan;
//! - a symlink or non-regular file anywhere on a path that is read.
//!
//! **Update index.** One W8 update is the window of optimizer steps between two published W8
//! checkpoints: `update_index` counts those windows 0, 1, 2, … and `consumed_rows` is the
//! examples the window consumed, `(step − previous step) × global_batch`. The W8 graph rules
//! need exactly one checkpoint per update (an exposure starts from the checkpoint of the update
//! before it), so an engine checkpoint that the cadence does not select is not a W8 update; its
//! optimizer steps are counted in the next selected one.

use crate::canon::sha256_hex;
use crate::progress::inventory::{
    BlobRef, Checkpoint, CorpusManifest, Decoded, Encoded, Event, EventBody, Exposure,
    FailureClass, Kind, RunManifest,
};
use crate::state::plain_segment;
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Schema identifier for corpus values.
pub const CORPUS_SCHEMA: &str = "parallel.teaching_corpus.v1";
/// Schema identifier for corpus publication values.
pub const CORPUS_PUBLICATION_SCHEMA: &str = "parallel.corpus_publication.v1";
/// Schema identifier for checkpoint values.
pub const CHECKPOINT_SCHEMA: &str = "parallel.checkpoint.v1";
/// Schema identifier for recovery values.
pub const RECOVERY_SCHEMA: &str = "parallel.checkpoint_recovery.v1";

/// The files of a published corpus directory, every one of them a corpus member.
pub const CORPUS_FILES: [&str; 6] = [
    "index.json",
    "labels.jsonl",
    "provenance.json",
    "publication.json",
    "records.bin",
    "vocabulary.json",
];

/// The objects of a checkpoint directory with the manifest field holding each digest
/// (`training::checkpoint_commit::REQUIRED_OBJECTS`).
pub const CHECKPOINT_OBJECTS: [(&str, &str); 4] = [
    ("model.ot", "model_sha256"),
    ("optimizer.ot", "optimizer_sha256"),
    ("optimizer.json", "optimizer_metadata_sha256"),
    ("config.json", "config_sha256"),
];

/// What one engine event becomes: blobs first, then the objects that name them, in order.
#[derive(Clone, Debug, Default)]
pub struct Mapped {
    /// Collection of blobs values carried by this mapped.
    pub blobs: Vec<Vec<u8>>,
    /// Number of objects represented by this value.
    pub objects: Vec<Encoded>,
}

/// Read a regular file once, refusing a symlink, a non-regular file and anything larger than
/// `max_bytes` (a blob the outbox could never hold).
fn read_plain(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let meta =
        fs::symlink_metadata(path).with_context(|| format!("{} is missing", path.display()))?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "{} is not a regular file; nothing was mapped",
        path.display()
    );
    ensure!(
        meta.len() <= max_bytes,
        "{} is {} bytes, more than the {max_bytes}-byte limit a single outbox item may have",
        path.display(),
        meta.len()
    );
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() as u64 <= max_bytes,
        "{} grew past {max_bytes} bytes while it was read",
        path.display()
    );
    Ok(bytes)
}

/// Every component of `rel` below `root` must be a plain, real directory (the last one too when
/// `last_is_dir`): no `..`, no absolute path, no symlink.
fn real_path(root: &Path, rel: &Path, last_is_dir: bool) -> Result<PathBuf> {
    ensure!(
        !rel.as_os_str().is_empty() && rel.components().all(|c| matches!(c, Component::Normal(_))),
        "{} is not a plain relative path under {}",
        rel.display(),
        root.display()
    );
    let parts: Vec<_> = rel.components().collect();
    let mut at = root.to_path_buf();
    for (i, part) in parts.iter().enumerate() {
        at.push(part);
        let is_last = i + 1 == parts.len();
        if is_last && !last_is_dir {
            break;
        }
        let meta =
            fs::symlink_metadata(&at).with_context(|| format!("{} is missing", at.display()))?;
        ensure!(
            meta.is_dir() && !meta.file_type().is_symlink(),
            "{} is not a real directory",
            at.display()
        );
    }
    Ok(at)
}

fn json(bytes: &[u8], what: &str) -> Result<Value> {
    serde_json::from_slice(bytes).with_context(|| format!("{what} does not parse"))
}

fn text<'a>(v: &'a Value, field: &str, what: &str) -> Result<&'a str> {
    v[field]
        .as_str()
        .with_context(|| format!("{what} has no string {field}"))
}

fn unsigned(v: &Value, field: &str, what: &str) -> Result<u64> {
    v[field]
        .as_u64()
        .with_context(|| format!("{what} has no unsigned {field}"))
}

fn planned(manifest: &Decoded<RunManifest>, work_id: &str) -> Result<()> {
    ensure!(
        manifest.value().units.iter().any(|u| u.work_id == work_id),
        "work {work_id} is not planned in run {}'s manifest; nothing was mapped",
        manifest.value().run_id
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Corpus publication
// ------------------------------------------------------------------------------------------

/// A published engine corpus as a W8 corpus manifest.
#[derive(Clone, Debug)]
pub struct EngineCorpus {
    /// Manifest recorded for this engine corpus.
    pub manifest: Decoded<CorpusManifest>,
    /// sha256 of the corpus `index.json`: the identity every checkpoint trained on it binds
    /// (`corpus_sha256` in its `state.json`).
    pub index_sha256: String,
    /// Mapped recorded for this engine corpus.
    pub mapped: Mapped,
}

/// Map the published corpus directory `dir`. Checks, on the bytes read once, everything
/// `corpus_publication::verify` checks on a published corpus, plus that the directory is the
/// published one (not its staging or a retired copy), that it is a qualification corpus, and that
/// its recorded publication request names the same inputs.
pub fn corpus(
    run_id: &str,
    dir: &Path,
    parent: Option<&str>,
    max_bytes: u64,
) -> Result<EngineCorpus> {
    let meta =
        fs::symlink_metadata(dir).with_context(|| format!("{} is missing", dir.display()))?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "{} is not a real corpus directory",
        dir.display()
    );
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{} has no corpus name", dir.display()))?;
    ensure!(
        plain_segment(name) && !name.starts_with('.') && !name.contains(".retired-"),
        "{} is not a published corpus: staging (`.<name>.staging`) and retired \
         (`<name>.retired-*`) builds are never progress; nothing was mapped",
        dir.display()
    );
    let mut files = Vec::with_capacity(CORPUS_FILES.len());
    for file in CORPUS_FILES {
        files.push(read_plain(&dir.join(file), max_bytes)?);
    }
    let [index_b, labels_b, provenance_b, publication_b, records_b, vocabulary_b]: [Vec<u8>; 6] =
        files.try_into().expect("six corpus files");
    let what = |f: &str| format!("{}/{f}", dir.display());
    let index = json(&index_b, &what("index.json"))?;
    let provenance = json(&provenance_b, &what("provenance.json"))?;
    let publication = json(&publication_b, &what("publication.json"))?;
    ensure!(
        index["schema"] == CORPUS_SCHEMA,
        "{} is not a {CORPUS_SCHEMA} index",
        what("index.json")
    );
    ensure!(
        index["qualification"] == true,
        "{} is a diagnostic corpus (qualification: false): it is never optimizer progress",
        dir.display()
    );
    let index_sha256 = sha256_hex(&index_b);
    ensure!(
        provenance["index_sha256"] == index_sha256.as_str()
            && provenance["labels_sha256"] == sha256_hex(&labels_b).as_str()
            && index["records_sha256"] == sha256_hex(&records_b).as_str()
            && index["vocabulary_sha256"] == sha256_hex(&vocabulary_b).as_str(),
        "{}: corpus payload or receipt hash changed",
        dir.display()
    );
    let records = unsigned(&index, "records", &what("index.json"))?;
    let record_bytes = unsigned(&index, "record_bytes", &what("index.json"))?;
    ensure!(
        records.checked_mul(record_bytes) == Some(records_b.len() as u64),
        "{}: truncated corpus records ({} bytes for {records} records of {record_bytes})",
        dir.display(),
        records_b.len()
    );
    ensure!(
        publication["schema"] == CORPUS_PUBLICATION_SCHEMA
            && publication["contexts_sha256"] == index["context_index_sha256"]
            && publication["vocabulary_sha256"] == index["vocabulary_sha256"]
            && publication["qualification"] == index["qualification"],
        "{}: corpus publication inputs changed",
        dir.display()
    );
    let bytes = [
        index_b,
        labels_b,
        provenance_b,
        publication_b,
        records_b,
        vocabulary_b,
    ];
    let mut members: Vec<BlobRef> = bytes.iter().map(|b| BlobRef::of(b)).collect();
    members.sort();
    members.dedup();
    let manifest = Decoded::new(CorpusManifest {
        schema: Kind::CorpusManifest.schema().into(),
        run_id: run_id.into(),
        corpus_id: name.into(),
        parent: parent.map(str::to_string),
        members,
        decisions: Vec::new(),
    })?;
    Ok(EngineCorpus {
        mapped: Mapped {
            blobs: bytes.into(),
            objects: vec![manifest.encoded()?],
        },
        manifest,
        index_sha256,
    })
}

// ------------------------------------------------------------------------------------------
// Checkpoints
// ------------------------------------------------------------------------------------------

/// What a run's `latest.json` names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Published {
    /// Step recorded for this published.
    pub step: u64,
    /// `<run>/checkpoints/step-<step>`, relative to the engine root.
    pub checkpoint: PathBuf,
    /// SHA-256 digest identifying the associated state.
    pub state_sha256: String,
    /// The `latest.json` bytes that said so.
    pub latest: Vec<u8>,
}

/// The checkpoint `run`'s `latest.json` names, or `None` before the first publication. `run` is
/// relative to the engine `root`. The named directory must be this run's `checkpoints/step-<n>`.
pub fn published(root: &Path, run: &Path) -> Result<Option<Published>> {
    let run_dir = real_path(root, run, true)?;
    let path = run_dir.join("latest.json");
    match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
        Ok(_) => {}
    }
    let latest = read_plain(&path, 1 << 20)?;
    let v = json(&latest, &path.display().to_string())?;
    let step = unsigned(&v, "step", "latest.json")?;
    let checkpoint = PathBuf::from(text(&v, "checkpoint", "latest.json")?);
    let state_sha256 = text(&v, "state_sha256", "latest.json")?.to_string();
    ensure!(
        checkpoint == run.join("checkpoints").join(format!("step-{step}")),
        "{} names {}, which is not this run's checkpoints/step-{step}",
        path.display(),
        checkpoint.display()
    );
    Ok(Some(Published {
        step,
        checkpoint,
        state_sha256,
        latest,
    }))
}

/// The last W8 checkpoint mapped for the run and the engine step it was taken at.
#[derive(Clone, Debug)]
pub struct Previous {
    /// Checkpoint recorded for this previous.
    pub checkpoint: Decoded<Checkpoint>,
    /// Step recorded for this previous.
    pub step: u64,
}

/// A published engine checkpoint as the next W8 update.
#[derive(Clone, Debug)]
pub struct EngineCheckpoint {
    /// Step recorded for this engine checkpoint.
    pub step: u64,
    /// Exposure recorded for this engine checkpoint.
    pub exposure: Decoded<Exposure>,
    /// Checkpoint recorded for this engine checkpoint.
    pub checkpoint: Decoded<Checkpoint>,
    /// Mapped recorded for this engine checkpoint.
    pub mapped: Mapped,
}

/// Map the checkpoint `run`'s `latest.json` publishes as the update after `previous` (or update
/// 0, starting at optimizer step `start_step`). The checkpoint must be bound to `corpus` and its
/// step must advance past the previous one.
pub fn checkpoint(
    root: &Path,
    run: &Path,
    corpus: &EngineCorpus,
    previous: Option<&Previous>,
    start_step: u64,
    max_bytes: u64,
) -> Result<EngineCheckpoint> {
    let published = published(root, run)?.with_context(|| {
        format!(
            "{} has no latest.json: no checkpoint is published",
            root.join(run).display()
        )
    })?;
    let dir = real_path(root, &published.checkpoint, true)?;
    let state_b = read_plain(&dir.join("state.json"), max_bytes)?;
    ensure!(
        sha256_hex(&state_b) == published.state_sha256,
        "{}/state.json is not the manifest latest.json published",
        dir.display()
    );
    let state = json(&state_b, "checkpoint state.json")?;
    ensure!(
        state["schema"] == CHECKPOINT_SCHEMA && state["step"].as_u64() == Some(published.step),
        "{}: checkpoint manifest schema or step differs",
        dir.display()
    );
    ensure!(
        state["corpus_sha256"] == corpus.index_sha256.as_str(),
        "{}: the checkpoint was trained on corpus {}, not {} ({})",
        dir.display(),
        state["corpus_sha256"],
        corpus.manifest.value().corpus_id,
        corpus.index_sha256
    );
    let global_batch = unsigned(&state, "global_batch", "checkpoint state.json")?;
    ensure!(global_batch > 0, "{}: empty global batch", dir.display());
    let mut blobs = vec![state_b];
    for (name, field) in CHECKPOINT_OBJECTS {
        let bytes = read_plain(&dir.join(name), max_bytes)?;
        ensure!(
            state[field] == sha256_hex(&bytes).as_str(),
            "checkpoint object {}/{name} differs from its manifest digest",
            dir.display()
        );
        blobs.push(bytes);
    }
    let (update_index, before, base_step) = match previous {
        Some(p) => (
            p.checkpoint
                .value()
                .update_index
                .checked_add(1)
                .context("update index overflow")?,
            Some(p.checkpoint.digest().to_string()),
            p.step,
        ),
        None => (0, None, start_step),
    };
    ensure!(
        published.step > base_step,
        "the published checkpoint at step {} does not advance past step {base_step}, where the \
         last W8 update ended; nothing was mapped",
        published.step
    );
    let consumed_rows = (published.step - base_step)
        .checked_mul(global_batch)
        .context("consumed rows overflow")?;
    let corpus_digest = corpus.manifest.digest().to_string();
    let exposure = Decoded::new(Exposure {
        schema: Kind::Exposure.schema().into(),
        run_id: corpus.manifest.value().run_id.clone(),
        update_index,
        corpus: corpus_digest.clone(),
        checkpoint_before: before.clone(),
        consumed_rows,
    })?;
    let mut state_refs: Vec<BlobRef> = blobs.iter().map(|b| BlobRef::of(b)).collect();
    state_refs.sort();
    state_refs.dedup();
    let checkpoint = Decoded::new(Checkpoint {
        schema: Kind::Checkpoint.schema().into(),
        run_id: corpus.manifest.value().run_id.clone(),
        update_index,
        parent: before,
        exposure: exposure.digest().into(),
        corpus: corpus_digest,
        state: state_refs,
    })?;
    Ok(EngineCheckpoint {
        step: published.step,
        mapped: Mapped {
            blobs,
            objects: vec![exposure.encoded()?, checkpoint.encoded()?],
        },
        exposure,
        checkpoint,
    })
}

/// The engine step a mapped W8 checkpoint was taken at, from its own `state.json` blob.
pub fn checkpoint_step(state_json: &[u8]) -> Result<u64> {
    let v = json(state_json, "checkpoint state.json")?;
    ensure!(
        v["schema"] == CHECKPOINT_SCHEMA,
        "not a {CHECKPOINT_SCHEMA} manifest"
    );
    unsigned(&v, "step", "checkpoint state.json")
}

// ------------------------------------------------------------------------------------------
// Checkpoint recovery and no-replace publications
// ------------------------------------------------------------------------------------------

/// Map a completed checkpoint-recovery receipt as the retryable `worker-lost` failure of
/// `(work_id, attempt)`: the ranks of that attempt were reaped and its unpublished checkpoints
/// quarantined, so the attempt ended without a result. Returns the receipt bytes (retained by the
/// caller for custody) and the event.
pub fn recovery(
    manifest: &Decoded<RunManifest>,
    work_id: &str,
    attempt: u32,
    receipt: &Path,
) -> Result<(Vec<u8>, Decoded<Event>)> {
    planned(manifest, work_id)?;
    let bytes = read_plain(receipt, 16 << 20)?;
    let v = json(&bytes, &receipt.display().to_string())?;
    ensure!(
        v["schema"] == RECOVERY_SCHEMA,
        "{} is not a {RECOVERY_SCHEMA} receipt",
        receipt.display()
    );
    ensure!(
        v["completed"] == true,
        "{} is an incomplete recovery; its moves may still be in progress, so it is not mapped",
        receipt.display()
    );
    let moves = v["moves"]
        .as_array()
        .filter(|m| !m.is_empty())
        .with_context(|| format!("{} moved nothing", receipt.display()))?;
    ensure!(
        moves.iter().all(|m| m["moved"] == true),
        "{} records a move that did not happen",
        receipt.display()
    );
    let published = match &v["published_step"] {
        Value::Null => "no published step".to_string(),
        step => format!(
            "published step {}",
            step.as_u64().context("published_step is not a step")?
        ),
    };
    let event = Decoded::new(Event::new(
        manifest,
        work_id,
        attempt,
        EventBody::Failed {
            class: FailureClass::WorkerLost,
            retryable: true,
            detail: format!(
                "checkpoint recovery quarantined {} unpublished checkpoint(s) after {published}; \
                 receipt sha256 {}",
                moves.len(),
                sha256_hex(&bytes)
            ),
        },
    ))?;
    Ok((bytes, event))
}

/// Map a no-replace engine publication (`{path, sha256}` relative to `root`) as the result of
/// `(work_id, attempt)`: its bytes must hash to the published digest. Returns the result bytes
/// and the `completed` event naming them.
pub fn result(
    manifest: &Decoded<RunManifest>,
    work_id: &str,
    attempt: u32,
    root: &Path,
    artifact: &Path,
    sha256: &str,
    max_bytes: u64,
) -> Result<(Vec<u8>, Decoded<Event>)> {
    planned(manifest, work_id)?;
    let path = real_path(root, artifact, false)?;
    let bytes = read_plain(&path, max_bytes)?;
    let got = sha256_hex(&bytes);
    if got != sha256 {
        bail!(
            "{} hashes to {got}, not the published {sha256}; nothing was mapped",
            path.display()
        );
    }
    let event = Decoded::new(Event::new(
        manifest,
        work_id,
        attempt,
        EventBody::Completed {
            result: BlobRef::of(&bytes),
        },
    ))?;
    Ok((bytes, event))
}

/// Explicit join to the already planned W8 unit; never reconstruct it from a winner or change
/// the original run/attempt identity. This is observation, not I04 replay eligibility.
#[derive(Clone, Debug)]
pub struct SearchWorkBinding {
    /// Identifier for the associated request.
    pub request_id: String,
    /// Identifier for the associated work.
    pub work_id: String,
}

#[derive(Clone, Debug)]
/// `SearchMapped` is a record for the search mapped contract in this module: W8 product integration: the engine's durable events as W8 progress objects.
pub struct SearchMapped {
    /// Mapped recorded for this search mapped.
    pub mapped: Mapped,
    /// Collection of events values carried by this search mapped.
    pub events: Vec<Decoded<Event>>,
    /// Reconciliation recorded for this search mapped.
    pub reconciliation: jope_pipeline::search_events::Reconciliation,
}

/// Map an authenticated event sidecar and original result blobs through the actual W8 event
/// types. The caller retains `mapped.blobs` (including the full sidecar/costs) before its objects.
/// Missing/changed evidence refuses the entire mapping. Duplicate deliveries keep their IDs.
pub fn search_events(
    manifest: &Decoded<RunManifest>,
    bytes: &[u8],
    expected_sha256: &str,
    bindings: &[SearchWorkBinding],
    observations: &[Vec<u8>],
) -> Result<SearchMapped> {
    use jope_pipeline::search_events::{SearchEvent, SearchEventLog};
    use std::collections::{BTreeMap, BTreeSet};
    ensure!(
        sha256_hex(bytes) == expected_sha256,
        "altered search-event artifact"
    );
    let log = SearchEventLog::from_bytes(bytes)?;
    ensure!(
        log.binding.run_id == manifest.value().run_id,
        "search event belongs to another original run"
    );
    let reconciliation = log.require_complete()?;
    let mut units = BTreeMap::new();
    let mut used_units = BTreeSet::new();
    for binding in bindings {
        planned(manifest, &binding.work_id)?;
        ensure!(
            manifest
                .value()
                .units
                .iter()
                .any(|unit| unit.work_id == binding.work_id
                    && unit.spec.inputs.contains(&binding.request_id)),
            "request absent from original planned work inputs"
        );
        ensure!(
            units
                .insert(&binding.request_id, &binding.work_id)
                .is_none()
                && used_units.insert(&binding.work_id),
            "conflicting request/work join"
        );
    }
    let mut results = BTreeMap::new();
    for observation in observations {
        results.insert(sha256_hex(observation), observation);
    }
    let mut starts = BTreeMap::new();
    let mut terminals = BTreeMap::new();
    for record in &log.events {
        match &record.event {
            SearchEvent::Started {
                request_id,
                ordinal,
                ..
            } => {
                starts.insert((request_id, ordinal), &record.event);
            }
            SearchEvent::Finished { attempt_id, .. } => {
                terminals.insert(attempt_id, (&record.event_id, &record.event));
            }
            _ => {}
        }
    }
    ensure!(
        units
            .keys()
            .all(|id| starts.keys().any(|(request, _)| request == id)),
        "work join without physical request"
    );
    let mut mapped = Mapped {
        blobs: vec![bytes.to_vec()],
        objects: Vec::new(),
    };
    let mut events = Vec::new();
    let mut failed: BTreeMap<&String, String> = BTreeMap::new();
    for ((request, ordinal), start) in starts {
        let SearchEvent::Started {
            attempt_id,
            retry_of,
            ..
        } = start
        else {
            unreachable!()
        };
        let work = units
            .get(request)
            .context("physical attempt missing original work binding")?;
        let opening = if *ordinal == 1 {
            EventBody::Scheduled
        } else {
            EventBody::Retry {
                failed_event: failed
                    .get(retry_of.as_ref().context("missing retry predecessor")?)
                    .context("retry missing original failure")?
                    .clone(),
            }
        };
        events.push(Decoded::new(Event::new(manifest, work, *ordinal, opening))?);
        events.push(Decoded::new(Event::new(
            manifest,
            work,
            *ordinal,
            EventBody::Started {
                worker: "pipeline-in-process".into(),
            },
        ))?);
        let (terminal_id, terminal_event) = terminals
            .get(attempt_id)
            .context("missing terminal/cost event")?;
        let SearchEvent::Finished {
            status,
            reason,
            observation_sha256,
            cost,
            ..
        } = terminal_event
        else {
            unreachable!()
        };
        ensure!(
            *ordinal <= manifest.value().max_attempts,
            "attempt exceeds original run budget"
        );
        let original = observation_sha256
            .as_ref()
            .map(|digest| {
                results
                    .get(digest)
                    .context("missing original observation bytes")
            })
            .transpose()?;
        if let Some(original) = original {
            let observation: jope_pipeline::search::ObservationRecord =
                serde_json::from_slice(original)?;
            let observation = observation.into_observation()?;
            ensure!(
                observation.seconds.is_finite() && observation.seconds >= 0.,
                "invalid original observation clock"
            );
            ensure!(
                observation.status == *status && observation.failure == *reason,
                "terminal status differs from original observation"
            );
            if let Some(wall) = &cost.wall_seconds {
                ensure!(
                    wall.value()? >= observation.seconds,
                    "attempt wall cost is smaller than its original observation span"
                );
            }
            let SearchEvent::Started { draw_id, .. } = start else {
                unreachable!()
            };
            let program = log
                .events
                .iter()
                .find_map(|r| match &r.event {
                    SearchEvent::Draw { decision } if &r.event_id == draw_id => {
                        decision.program_sha256.as_ref()
                    }
                    _ => None,
                })
                .context("missing original draw subject")?;
            ensure!(
                *program == observation.program_sha256,
                "result belongs to another drawn program"
            );
        }
        let body = if status == "measured" {
            let digest = observation_sha256
                .as_ref()
                .context("missing result digest")?;
            let original = results
                .get(digest)
                .context("missing original observation bytes")?;
            mapped.blobs.push((*original).clone());
            EventBody::Completed {
                result: BlobRef::of(original),
            }
        } else {
            // Retain original failure bytes too whenever the producer names them.
            if let Some(digest) = observation_sha256 {
                let original = results
                    .get(digest)
                    .context("missing original failed observation bytes")?;
                mapped.blobs.push((*original).clone());
            }
            let class = match status.as_str() {
                "parse_failed" => FailureClass::ParseRejection,
                "timed_out" => FailureClass::Timeout,
                "interrupted" => FailureClass::Cancelled,
                _ => FailureClass::WorkerError,
            };
            EventBody::Failed {
                class,
                retryable: *ordinal < manifest.value().max_attempts,
                detail: format!(
                    "{}; attempt {}; cost basis {}; terminal event {}",
                    reason.as_deref().context("failure reason missing")?,
                    attempt_id,
                    cost.reason,
                    terminal_id
                ),
            }
        };
        let terminal = Decoded::new(Event::new(manifest, work, *ordinal, body))?;
        if status != "measured" {
            failed.insert(attempt_id, terminal.digest().to_owned());
        }
        events.push(terminal);
    }
    for event in &events {
        mapped.objects.push(event.encoded()?);
    }
    Ok(SearchMapped {
        mapped,
        events,
        reconciliation,
    })
}
