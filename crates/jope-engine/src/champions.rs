//! Durable best-found teachers. Scores never cross a cut or evaluator identity.
use crate::{manifest::*, prepare::Prepared};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identity {
    pub dataset: String,
    pub family: String,
    pub partition: String,
    pub cut: usize,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub validation_sha256: String,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
}
impl Identity {
    pub fn key(&self) -> Result<String> {
        Ok(hash(&serde_json::to_vec(self)?))
    }
    pub fn observation(value: &Value) -> Result<Option<Self>> {
        if value["vocabulary_sha256"].is_null() {
            return Ok(None);
        }
        let string = |name: &str| -> Result<String> {
            Ok(value[name]
                .as_str()
                .with_context(|| format!("teacher {name} absent"))?
                .to_string())
        };
        let result = Self {
            dataset: string("dataset")?,
            family: string("family")?,
            partition: string("partition")?,
            cut: value["cut"].as_u64().context("teacher cut absent")? as usize,
            prepared_sha256: string("prepared_sha256")?,
            training_sha256: string("training_sha256")?,
            validation_sha256: string("validation_sha256")?,
            evaluator_sha256: string("evaluator_sha256")?,
            vocabulary_sha256: string("vocabulary_sha256")?,
        };
        for digest in [
            &result.prepared_sha256,
            &result.training_sha256,
            &result.validation_sha256,
            &result.evaluator_sha256,
            &result.vocabulary_sha256,
        ] {
            ensure!(
                digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid teacher identity hash"
            );
        }
        Ok(Some(result))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Champion {
    pub schema: String,
    pub identity: Identity,
    pub observation: Value,
    pub observation_sha256: String,
    pub source: PathBuf,
    pub rmse_original_units: f64,
    pub previous_event_sha256: Option<String>,
    pub previous_rmse_original_units: Option<f64>,
    pub improvement_number: usize,
    pub timestamp: u64,
}
impl Champion {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "parallel.cut_champion.v1"
                && Identity::observation(&self.observation)?.as_ref() == Some(&self.identity)
                && hash(&serde_json::to_vec(&self.observation)?) == self.observation_sha256
                && score(&self.observation)? == Some(self.rmse_original_units),
            "champion evidence changed"
        );
        Ok(())
    }
    pub fn genome(&self) -> &str {
        self.observation["genome"].as_str().unwrap()
    }
}

fn score(o: &Value) -> Result<Option<f64>> {
    if !o["failure"].is_null() || o["evaluation_bias"] != "adaptive_cut_heldout_teacher" {
        return Ok(None);
    }
    let rmse = o["metrics"]["rmse"]
        .as_f64()
        .context("teacher RMSE absent")?;
    let scale = o["target_scale"]
        .as_f64()
        .context("teacher target scale absent")?;
    let original = rmse * scale;
    ensure!(
        rmse.is_finite() && rmse >= 0.0 && scale.is_finite() && scale > 0.0 && original.is_finite(),
        "invalid teacher score"
    );
    let genome = o["genome"].as_str().context("teacher program absent")?;
    ensure!(
        hash(genome.as_bytes()) == o["program_sha256"],
        "teacher program hash changed"
    );
    ensure!(
        o["independent_evaluation"] == false,
        "teacher bias identity absent"
    );
    Ok(Some(original))
}

/// Persist the complete observation and strict winner before the caller
/// acknowledges its journal row. Rotation cannot erase this evidence.
pub fn record(root: &Path, observation: &Value, source: &Path) -> Result<bool> {
    record_inner(root, observation, source, true)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PendingChampion {
    champion: Champion,
    publish: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PendingPublication {
    sequence: u64,
    champion: Champion,
}

fn immutable_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("immutable champion parent absent")?;
    durable_directory(parent)?;
    let temporary = tempfile::Builder::new()
        .prefix(".pending-")
        .tempfile_in(parent)?;
    let (mut file, staging) = temporary.keep()?;
    file.write_all(bytes)?;
    file.sync_all()?;
    match crate::native_persistence::publish_staged(&staging, path) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(
                fs::read(path)? == bytes,
                "conflicting immutable champion bytes"
            );
        }
        Err(error) => return Err(error.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&staging)?;
    remove_recovered_staging_links(path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

// Called under the original identity/publication writer lock. A crash after the
// durable no-replace link can leave only our named staging alias behind. Remove
// aliases solely when both regular-file identities match the committed inode;
// unrelated partial attempts and conflicting versions remain preserved.
fn remove_recovered_staging_links(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let committed = fs::symlink_metadata(path)?;
    ensure!(
        committed.is_file(),
        "champion immutable target is not regular"
    );
    if committed.nlink() <= 1 {
        return Ok(());
    }
    let parent = path.parent().context("champion staging parent")?;
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with(".pending-") {
            continue;
        }
        let candidate = fs::symlink_metadata(entry.path())?;
        if candidate.is_file()
            && candidate.dev() == committed.dev()
            && candidate.ino() == committed.ino()
        {
            fs::remove_file(entry.path())?;
        }
    }
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn durable_directory(path: &Path) -> Result<()> {
    if !path.exists() {
        let parent = path.parent().context("champion directory parent absent")?;
        durable_directory(parent)?;
        match fs::create_dir(path) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error.into()),
        }
        fs::File::open(parent)?.sync_all()?;
    }
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "champion directory is not physical"
    );
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

fn finish_publication(outbox: &Path, pending: &PendingPublication) -> Result<()> {
    pending.champion.validate()?;
    ensure!(
        matches!(
            pending.champion.identity.partition.as_str(),
            "meta_train" | "meta_validation"
        ),
        "sealed or unknown pending champion must remain opaque"
    );
    let bytes = serde_json::to_vec(&pending.champion)?;
    let digest = hash(&bytes);
    let next_path = outbox.join("next.json");
    if next_path.exists() {
        let next: u64 = read_json(&next_path)?;
        ensure!(
            next == pending.sequence
                || next
                    == pending
                        .sequence
                        .checked_add(1)
                        .context("winner sequence exhausted")?,
            "winner counter conflicts with pending sequence"
        );
    }
    immutable_bytes(
        &outbox
            .join("by-sequence")
            .join(format!("{:020}.json", pending.sequence)),
        &serde_json::to_vec(&digest)?,
    )?;
    immutable_bytes(
        &outbox.join(format!("{:020}-{digest}.json", pending.sequence)),
        &bytes,
    )?;
    immutable_bytes(
        &outbox.join("by-event").join(format!("{digest}.json")),
        &serde_json::to_vec(&pending.sequence)?,
    )?;
    let next = pending
        .sequence
        .checked_add(1)
        .context("winner sequence exhausted")?;
    write_json(&outbox.join("next.json"), &next)?;
    fs::remove_file(outbox.join("pending.json"))?;
    fs::File::open(outbox)?.sync_all()?;
    Ok(())
}

fn publish_winner(root: &Path, champion: &Champion) -> Result<()> {
    ensure!(
        matches!(
            champion.identity.partition.as_str(),
            "meta_train" | "meta_validation"
        ),
        "sealed or unknown partition cannot enter winner publication"
    );
    let outbox = root.join("runs/champions-outbox");
    durable_directory(&outbox)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(outbox.join("lock"))?;
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
        "winner outbox lock failed"
    );
    // A crashed publisher must complete the original reserved sequence before
    // another winner can acquire one. This also repairs missing derived indexes.
    let pending_path = outbox.join("pending.json");
    if pending_path.exists() {
        finish_publication(&outbox, &read_json(&pending_path)?)?;
    }
    let bytes = serde_json::to_vec(champion)?;
    let digest = hash(&bytes);
    let index = outbox.join("by-event").join(format!("{digest}.json"));
    if index.exists() {
        let sequence: u64 = read_json(&index)?;
        ensure!(
            fs::read(outbox.join(format!("{sequence:020}-{digest}.json")))? == bytes,
            "published winner bytes missing or changed"
        );
        return Ok(());
    }
    let next = outbox.join("next.json");
    let sequence = if next.exists() {
        let sequence: u64 = read_json(&next)?;
        if sequence > 0 {
            let previous: String = read_json(
                &outbox
                    .join("by-sequence")
                    .join(format!("{:020}.json", sequence - 1)),
            )?;
            ensure!(
                hash(&fs::read(
                    outbox.join(format!("{:020}-{previous}.json", sequence - 1))
                )?) == previous,
                "winner counter lacks its preceding durable event"
            );
        }
        sequence
    } else {
        for entry in fs::read_dir(&outbox)? {
            let name = entry?.file_name();
            ensure!(
                !name
                    .to_string_lossy()
                    .starts_with(|c: char| c.is_ascii_digit()),
                "winner sequence state missing from populated outbox"
            );
        }
        0u64
    };
    let pending = PendingPublication {
        sequence,
        champion: champion.clone(),
    };
    immutable_bytes(&pending_path, &serde_json::to_vec(&pending)?)?;
    finish_publication(&outbox, &pending)
}

fn finish_champion(root: &Path, directory: &Path, pending: &PendingChampion) -> Result<()> {
    pending.champion.validate()?;
    ensure!(
        matches!(
            pending.champion.identity.partition.as_str(),
            "meta_train" | "meta_validation"
        ),
        "sealed or unknown pending champion must remain opaque"
    );
    let event = hash(&serde_json::to_vec(&pending.champion)?);
    ensure!(
        directory.file_name().and_then(|x| x.to_str())
            == Some(pending.champion.identity.key()?.as_str()),
        "pending champion namespace changed"
    );
    immutable_bytes(
        &directory.join(format!("event-{event}.json")),
        &serde_json::to_vec_pretty(&pending.champion)?,
    )?;
    immutable_bytes(
        &directory.join("origins").join(format!("{event}.json")),
        &serde_json::to_vec(&json!({
            "schema":"parallel.champion_origin.v1", "event_sha256":event,
            "origin":if pending.publish {"live_capture"} else {"historical_import"},
            "original_observation_sha256":pending.champion.observation_sha256,
            "source":pending.champion.source
        }))?,
    )?;
    if pending.publish {
        publish_winner(root, &pending.champion)?;
    }
    write_json(&directory.join("current.json"), &pending.champion)?;
    fs::remove_file(directory.join("pending.json"))?;
    fs::File::open(directory)?.sync_all()?;
    Ok(())
}

/// Reconcile already-fsynced winners before publication or proposal snapshots.
/// Takes the same per-identity locks as writers and preserves original events.
pub fn recover_outbox(root: &Path) -> Result<usize> {
    let champions = root.join("runs/champions");
    if !champions.exists() {
        return Ok(0);
    }
    let mut recovered = 0;
    for entry in fs::read_dir(champions)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let directory = entry.path();
        let pending = directory.join("pending.json");
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(directory.join("lock"))?;
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
            "champion recovery lock failed"
        );
        if pending.exists() {
            finish_champion(root, &directory, &read_json(&pending)?)?;
            recovered += 1;
        }
        // Only after pending admission passed its sealed-partition checks.
        for artifact in fs::read_dir(&directory)? {
            let artifact = artifact?;
            if artifact.file_name().to_string_lossy().starts_with("event-")
                && artifact.file_type()?.is_file()
            {
                remove_recovered_staging_links(&artifact.path())?;
            }
        }
    }
    Ok(recovered)
}

fn record_inner(root: &Path, observation: &Value, source: &Path, publish: bool) -> Result<bool> {
    let Some(identity) = Identity::observation(observation)? else {
        return Ok(false);
    };
    ensure!(
        matches!(
            identity.partition.as_str(),
            "meta_train" | "meta_validation"
        ),
        "sealed or unknown partition cannot enter champion admission"
    );
    let Some(rmse) = score(observation)? else {
        return Ok(false);
    };
    let directory = root.join("runs/champions").join(identity.key()?);
    durable_directory(&directory)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join("lock"))?;
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
        "champion lock failed"
    );
    let pending_path = directory.join("pending.json");
    if pending_path.exists() {
        finish_champion(root, &directory, &read_json(&pending_path)?)?;
    }
    let current = directory.join("current.json");
    let previous = if current.exists() {
        let old: Champion = read_json(&current)?;
        old.validate()?;
        ensure!(old.identity == identity, "champion namespace collision");
        if old.rmse_original_units <= rmse {
            return Ok(false);
        }
        Some(old)
    } else {
        None
    };
    let champion = Champion {
        schema: "parallel.cut_champion.v1".into(),
        identity,
        observation: observation.clone(),
        observation_sha256: hash(&serde_json::to_vec(observation)?),
        source: source.strip_prefix(root).unwrap_or(source).to_path_buf(),
        rmse_original_units: rmse,
        previous_event_sha256: previous
            .as_ref()
            .map(|p| hash(&serde_json::to_vec(p).unwrap())),
        previous_rmse_original_units: previous.as_ref().map(|p| p.rmse_original_units),
        improvement_number: previous.as_ref().map_or(0, |p| p.improvement_number + 1),
        timestamp: crate::telemetry::now(),
    };
    champion.validate()?;
    // Event identities use canonical compact serialization independently of
    // the human-readable file representation.
    let pending = PendingChampion { champion, publish };
    immutable_bytes(&pending_path, &serde_json::to_vec(&pending)?)?;
    finish_champion(root, &directory, &pending)?;
    Ok(true)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub schema: String,
    pub request: Value,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
    pub entries: Vec<Champion>,
}

/// Both proposal arms share this immutable cycle-level snapshot. The request
/// binds their common corpus/checkpoint/cut range; changes fail on resume.
pub fn freeze_snapshot(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
) -> Result<PathBuf> {
    freeze_snapshot_inner(root, corpus, checkpoint, output, cuts, cut_start, None)
}

/// Native input freezing uses the actual qualified evaluator identity from installed
/// root-controlled signed policy. It does not authorize or execute numerical work.
pub fn freeze_native_snapshot(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    policy_path: &Path,
) -> Result<PathBuf> {
    let policy = crate::eval_registry::native_scq::policy::InstalledPolicy::load(policy_path)?;
    ensure!(
        policy.signed.body.root == root,
        "native snapshot policy root differs"
    );
    freeze_snapshot_inner(
        root,
        corpus,
        checkpoint,
        output,
        cuts,
        cut_start,
        Some(&policy.signed.body.evaluator_executable_sha256),
    )
}
fn freeze_snapshot_inner(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    native_evaluator: Option<&str>,
) -> Result<PathBuf> {
    recover_outbox(root)?;
    let destination = output
        .parent()
        .context("proposal parent absent")?
        .join("champion-snapshot.json");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(destination.with_extension("lock"))?;
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
        "snapshot lock failed"
    );
    let mut request = json!({"corpus_sha256":file_hash(&corpus.join("index.json"))?,
        "checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"cuts":cuts,"cut_start":cut_start,
        "vocabulary_sha256":file_hash(&corpus.join("vocabulary.json"))?});
    if let Some(evaluator) = native_evaluator {
        request["native_qualified_evaluator_sha256"] = json!(evaluator);
    }
    let selection_path = destination.with_file_name("coverage-selection.json");
    if selection_path.exists() {
        request["coverage_selection_sha256"] = file_hash(&selection_path)?.into();
        request["coverage_selection_protocol"] = if native_evaluator.is_some() {
            read_json::<Value>(&selection_path)?["request"]["protocol"].clone()
        } else {
            crate::service::coverage::PROTOCOL.into()
        };
    }
    if destination.exists() {
        let snapshot: Snapshot = read_json(&destination)?;
        ensure!(
            snapshot.schema == "parallel.champion_snapshot.v1" && snapshot.request == request,
            "cycle champion snapshot request changed"
        );
        for entry in &snapshot.entries {
            entry.validate()?;
        }
        bind_snapshot(&destination)?;
        return Ok(destination);
    }
    let vocabulary = request["vocabulary_sha256"].as_str().unwrap().to_string();
    let evaluator = if let Some(evaluator) = native_evaluator {
        evaluator.to_owned()
    } else {
        let service: Value = read_json(&root.join("runs/gp-service.json"))?;
        let evaluator = service["executable_sha256"]
            .as_str()
            .context("GP service evaluator identity absent")?;
        let executable = PathBuf::from(
            service["executable"]
                .as_str()
                .context("GP service executable absent")?,
        );
        let executable = if executable.is_absolute() {
            executable
        } else {
            root.join(executable)
        };
        ensure!(
            executable.starts_with(root)
                && file_hash(&executable)? == evaluator
                && crate::telemetry::now()
                    .saturating_sub(service["timestamp"].as_u64().unwrap_or(0))
                    <= 10,
            "GP evaluator receipt is stale or changed"
        );
        ensure!(
            service["vocabulary_sha256"] == vocabulary,
            "GP service vocabulary differs from corpus"
        );
        evaluator.to_owned()
    };
    let evaluator = evaluator.as_str();
    let index = crate::teaching_store::read_index(root, corpus)?;
    let mut available = Vec::<Champion>::new();
    let store = root.join("runs/champions");
    if store.exists() {
        for directory in fs::read_dir(store)? {
            let path = directory?.path().join("current.json");
            if !path.is_file() {
                continue;
            }
            let entry: Champion = read_json(&path)?;
            entry.validate()?;
            if entry.identity.vocabulary_sha256 == vocabulary {
                available.push(entry);
            }
        }
    }
    // Current evaluator first, then a deterministic version order. Scores from
    // different evaluator versions are never compared; every seed is refitted.
    available.sort_by(|a, b| {
        (
            a.identity.evaluator_sha256 != evaluator,
            &a.identity.evaluator_sha256,
        )
            .cmp(&(
                b.identity.evaluator_sha256 != evaluator,
                &b.identity.evaluator_sha256,
            ))
    });
    let mut prepared = BTreeMap::<PathBuf, Prepared>::new();
    let mut entries = Vec::new();
    let selected = if selection_path.exists() {
        crate::service::coverage::selected_indices(&selection_path, corpus, &index.contexts)?
    } else {
        index
            .contexts
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.partition != "meta_test" && c.cut >= cut_start && c.cut - cut_start < cuts
            })
            .map(|(i, _)| i)
            .collect()
    };
    for i in selected {
        let c = &index.contexts[i];
        if !prepared.contains_key(&c.prepared) {
            ensure!(
                file_hash(&resolve(root, &c.prepared)?.join("prepared.json"))? == c.prepared_sha256,
                "snapshot preparation changed"
            );
            prepared.insert(
                c.prepared.clone(),
                read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?,
            );
        }
        let cut = prepared[&c.prepared]
            .cuts
            .iter()
            .find(|p| p.index == c.cut)
            .context("snapshot cut absent")?;
        let identity = Identity {
            dataset: c.dataset.clone(),
            family: c.family.clone(),
            partition: c.partition.clone(),
            cut: c.cut,
            prepared_sha256: c.prepared_sha256.clone(),
            training_sha256: cut.training_sha256.clone(),
            validation_sha256: cut.validation_sha256.clone(),
            evaluator_sha256: evaluator.into(),
            vocabulary_sha256: vocabulary.clone(),
        };
        for entry in &available {
            let mut compatible = entry.identity.clone();
            compatible.evaluator_sha256 = evaluator.into();
            if compatible == identity {
                entries.push(entry.clone());
            }
        }
    }
    write_json(
        &destination,
        &Snapshot {
            schema: "parallel.champion_snapshot.v1".into(),
            request,
            evaluator_sha256: evaluator.into(),
            vocabulary_sha256: vocabulary,
            entries,
        },
    )?;
    bind_snapshot(&destination)?;
    Ok(destination)
}

fn bind_snapshot(path: &Path) -> Result<()> {
    let receipt = path.with_file_name("champion-snapshot-binding.json");
    let expected =
        json!({"schema":"parallel.champion_snapshot_binding.v1","sha256":file_hash(path)?});
    if receipt.exists() {
        ensure!(
            read_json::<Value>(&receipt)? == expected,
            "champion snapshot binding changed"
        );
    } else {
        write_json(&receipt, &expected)?;
    }
    Ok(())
}

pub fn verify_cycle_snapshot(
    output: &Path,
    corpus_sha: &str,
    cuts: usize,
    cut_start: usize,
) -> Result<()> {
    let path = output.join("champion-snapshot.json");
    let snapshot: Snapshot = read_json(&path)?;
    let binding: Value = read_json(&output.join("champion-snapshot-binding.json"))?;
    ensure!(
        binding["sha256"] == file_hash(&path)?
            && snapshot.request["corpus_sha256"] == corpus_sha
            && snapshot.request["cuts"] == cuts
            && snapshot.request["cut_start"] == cut_start,
        "cycle champion snapshot identity changed"
    );
    for entry in &snapshot.entries {
        entry.validate()?;
    }
    Ok(())
}

/// One bounded pass over explicitly declared completed ledgers. Only the best
/// observation for each exact identity survives in RAM; no test CSV is opened.
pub fn import(root: &Path, sources: &[PathBuf]) -> Result<Value> {
    let mut best = BTreeMap::<Identity, (f64, Value, PathBuf)>::new();
    let mut ledgers = 0usize;
    let mut observations = 0usize;
    for source in sources {
        for entry in walkdir::WalkDir::new(resolve(root, source)?).follow_links(false) {
            let entry = entry?;
            if !entry.file_type().is_file() || entry.file_name() != "observations.json" {
                continue;
            }
            let summary = entry.path().parent().unwrap().join("search.json");
            if !summary.is_file() {
                continue;
            }
            ensure!(
                read_json::<Value>(&summary)?["schema"] == "parallel.search.v1",
                "invalid completed teacher search"
            );
            let rows: Vec<Value> = read_json(entry.path())?;
            ledgers += 1;
            for row in rows {
                observations += 1;
                let Some(identity) = Identity::observation(&row)? else {
                    continue;
                };
                let Some(rmse) = score(&row)? else {
                    continue;
                };
                if best.get(&identity).is_none_or(|(old, _, _)| rmse < *old) {
                    best.insert(identity, (rmse, row, entry.path().into()));
                }
            }
        }
    }
    let mut changes = 0;
    for (_, (_, row, path)) in &best {
        // Historical import preserves its evidence without manufacturing a new
        // live producer event or entering the live winner publication sequence.
        changes += record_inner(root, row, path, false)? as usize;
    }
    Ok(
        json!({"schema":"parallel.champion_import.v1","ledgers":ledgers,"observations":observations,
        "identities":best.len(),"new_or_improved_champions":changes,"source_roots":sources,
        "qualification_only":true,"final_tests_opened":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation(rmse: f64) -> Value {
        let genome = "(pipe X (ridge))";
        json!({"dataset":"d","family":"f","partition":"meta_train","cut":0,
            "prepared_sha256":"1".repeat(64),"training_sha256":"2".repeat(64),
            "validation_sha256":"3".repeat(64),"evaluator_sha256":"4".repeat(64),
            "vocabulary_sha256":"5".repeat(64),"genome":genome,"program_sha256":hash(genome.as_bytes()),
            "metrics":{"rmse":rmse},"target_scale":2.0,"failure":null,
            "evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false})
    }
    #[test]
    fn recovered_link_before_unlink_restores_single_link_winner_and_keeps_partial() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempfile::tempdir().unwrap();
        record(tmp.path(), &observation(3.0), Path::new("first")).unwrap();
        let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
        let directory = tmp
            .path()
            .join("runs/champions")
            .join(identity.key().unwrap());
        let event = fs::read_dir(&directory)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("event-")
            })
            .unwrap();
        let committed = fs::read(&event).unwrap();
        let alias = directory.join(".pending-crashed-link");
        fs::hard_link(&event, &alias).unwrap();
        let partial = directory.join(".pending-unrelated-partial");
        fs::write(&partial, b"partial").unwrap();
        assert_eq!(fs::metadata(&event).unwrap().nlink(), 2);
        recover_outbox(tmp.path()).unwrap();
        assert_eq!(fs::metadata(&event).unwrap().nlink(), 1);
        assert!(!alias.exists());
        assert_eq!(fs::read(&partial).unwrap(), b"partial");
        assert_eq!(fs::read(&event).unwrap(), committed);
    }
    #[test]
    fn only_improvements_replace_exact_identity_and_preserve_evidence() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let first = observation(3.0);
        assert!(record(root, &first, Path::new("run1/observations.json")).unwrap());
        let identity = Identity::observation(&first).unwrap().unwrap();
        let dir = root.join("runs/champions").join(identity.key().unwrap());
        let original = fs::read(dir.join("current.json")).unwrap();
        assert!(!record(
            root,
            &observation(4.0),
            Path::new("worse/observations.json")
        )
        .unwrap());
        assert_eq!(original, fs::read(dir.join("current.json")).unwrap());
        assert!(record(
            root,
            &observation(1.0),
            Path::new("better/observations.json")
        )
        .unwrap());
        let best: Champion = read_json(&dir.join("current.json")).unwrap();
        best.validate().unwrap();
        assert_eq!(best.rmse_original_units, 2.0);
        assert_eq!(best.previous_rmse_original_units, Some(6.0));
        let previous: Champion = read_json(&dir.join(format!(
            "event-{}.json",
            best.previous_event_sha256.unwrap()
        )))
        .unwrap();
        assert_eq!(previous.observation, first);
    }
    #[test]
    fn interrupted_winner_is_published_before_a_later_improvement() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        record(root, &observation(3.0), Path::new("first")).unwrap();
        let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
        let dir = root.join("runs/champions").join(identity.key().unwrap());
        let first: Champion = read_json(&dir.join("current.json")).unwrap();
        let mut intermediate = first.clone();
        intermediate.observation = observation(2.0);
        intermediate.observation_sha256 =
            hash(&serde_json::to_vec(&intermediate.observation).unwrap());
        intermediate.rmse_original_units = 4.0;
        intermediate.previous_event_sha256 = Some(hash(&serde_json::to_vec(&first).unwrap()));
        intermediate.previous_rmse_original_units = Some(6.0);
        intermediate.improvement_number = 1;
        let pending = PendingChampion {
            champion: intermediate,
            publish: true,
        };
        immutable_bytes(
            &dir.join("pending.json"),
            &serde_json::to_vec(&pending).unwrap(),
        )
        .unwrap();
        // This is the persisted state after intent fsync and before event/current.
        record(root, &observation(1.0), Path::new("latest")).unwrap();
        let mut winners = fs::read_dir(root.join("runs/champions-outbox"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(|c: char| c.is_ascii_digit())
            })
            .collect::<Vec<_>>();
        winners.sort();
        assert_eq!(winners.len(), 3);
        for (sequence, (path, rmse)) in winners.iter().zip([6.0, 4.0, 2.0]).enumerate() {
            let bytes = fs::read(path).unwrap();
            assert_eq!(
                path.file_name().unwrap().to_string_lossy(),
                format!("{sequence:020}-{}.json", hash(&bytes))
            );
            assert_eq!(
                serde_json::from_slice::<Champion>(&bytes)
                    .unwrap()
                    .rmse_original_units,
                rmse
            );
        }
    }

    #[test]
    fn outbox_replays_published_bytes_without_reallocating_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        record(tmp.path(), &observation(3.0), Path::new("first")).unwrap();
        let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
        let champion: Champion = read_json(
            &tmp.path()
                .join("runs/champions")
                .join(identity.key().unwrap())
                .join("current.json"),
        )
        .unwrap();
        let outbox = tmp.path().join("runs/champions-outbox");
        let digest = hash(&serde_json::to_vec(&champion).unwrap());
        fs::remove_file(outbox.join("by-event").join(format!("{digest}.json"))).unwrap();
        fs::remove_file(outbox.join("next.json")).unwrap();
        immutable_bytes(
            &outbox.join("pending.json"),
            &serde_json::to_vec(&PendingPublication {
                sequence: 0,
                champion: champion.clone(),
            })
            .unwrap(),
        )
        .unwrap();
        // Crash after publishing the event, before derived index/counter commit.
        publish_winner(tmp.path(), &champion).unwrap();
        assert_eq!(read_json::<u64>(&outbox.join("next.json")).unwrap(), 1);
        assert!(!outbox.join("pending.json").exists());
        assert_eq!(
            read_json::<u64>(&outbox.join("by-event").join(format!("{digest}.json"))).unwrap(),
            0
        );
        fs::write(outbox.join(format!("{:020}-{digest}.json", 0)), b"corrupt").unwrap();
        assert!(publish_winner(tmp.path(), &champion).is_err());
    }
    #[test]
    fn corrupt_counter_cannot_assign_a_second_event_to_an_existing_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        record(tmp.path(), &observation(3.0), Path::new("first")).unwrap();
        let outbox = tmp.path().join("runs/champions-outbox");
        let original = fs::read(outbox.join("by-sequence/00000000000000000000.json")).unwrap();
        write_json(&outbox.join("next.json"), &0u64).unwrap();
        assert!(record(tmp.path(), &observation(2.0), Path::new("second")).is_err());
        assert_eq!(
            fs::read(outbox.join("by-sequence/00000000000000000000.json")).unwrap(),
            original
        );
        assert_eq!(
            fs::read_dir(&outbox)
                .unwrap()
                .filter(|entry| entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(|c: char| c.is_ascii_digit()))
                .count(),
            1
        );
    }

    #[test]
    fn historical_origin_persists_and_sealed_rows_leave_no_live_pending_work() {
        let tmp = tempfile::tempdir().unwrap();
        let row = observation(3.0);
        record_inner(tmp.path(), &row, Path::new("historical"), false).unwrap();
        let identity = Identity::observation(&row).unwrap().unwrap();
        let directory = tmp
            .path()
            .join("runs/champions")
            .join(identity.key().unwrap());
        let champion: Champion = read_json(&directory.join("current.json")).unwrap();
        let event = hash(&serde_json::to_vec(&champion).unwrap());
        let origin: Value =
            read_json(&directory.join("origins").join(format!("{event}.json"))).unwrap();
        assert_eq!(origin["origin"], "historical_import");
        assert_eq!(champion.observation, row);
        assert!(!tmp.path().join("runs/champions-outbox").exists());
        let mut sealed = row;
        sealed["partition"] = json!("meta_test");
        assert!(record(tmp.path(), &sealed, Path::new("sealed")).is_err());
        assert!(record_inner(tmp.path(), &sealed, Path::new("sealed"), false).is_err());
        assert_eq!(recover_outbox(tmp.path()).unwrap(), 0);
        assert!(!tmp.path().join("runs/champions-outbox").exists());
    }
    #[test]
    fn different_heldout_or_evaluator_cannot_replace_an_incumbent() {
        let tmp = tempfile::tempdir().unwrap();
        let first = observation(3.0);
        record(tmp.path(), &first, Path::new("original")).unwrap();
        for field in [
            "validation_sha256",
            "training_sha256",
            "prepared_sha256",
            "evaluator_sha256",
            "vocabulary_sha256",
        ] {
            let mut other = observation(0.0);
            other[field] = json!("a".repeat(64));
            assert!(record(tmp.path(), &other, Path::new("other")).unwrap());
        }
        let id = Identity::observation(&first).unwrap().unwrap();
        let saved: Champion = read_json(
            &tmp.path()
                .join("runs/champions")
                .join(id.key().unwrap())
                .join("current.json"),
        )
        .unwrap();
        assert_eq!(saved.rmse_original_units, 6.0);
        assert_eq!(saved.improvement_number, 0);
    }
    #[test]
    fn changed_program_or_score_bias_is_never_admitted() {
        let tmp = tempfile::tempdir().unwrap();
        let mut o = observation(1.0);
        o["program_sha256"] = json!("0".repeat(64));
        assert!(record(tmp.path(), &o, Path::new("x")).is_err());
        o = observation(1.0);
        o["evaluation_bias"] = json!("independent_test");
        assert!(!record(tmp.path(), &o, Path::new("x")).unwrap());
    }
    #[test]
    fn partial_ledgers_are_not_imported() {
        let tmp = tempfile::tempdir().unwrap();
        let ledger = tmp.path().join("runs/incomplete/observations.json");
        write_json(&ledger, &vec![observation(1.0)]).unwrap();
        let report = import(tmp.path(), &[PathBuf::from("runs/incomplete")]).unwrap();
        assert_eq!(report["identities"], 0);
        assert!(!tmp.path().join("runs/champions").exists());
    }
    #[test]
    fn both_arms_freeze_same_programs_and_versions_never_share_scores() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for path in [
            "runs",
            "corpus",
            "checkpoint",
            "cycle/neural",
            "cycle/shuffled",
            "prepared",
        ] {
            fs::create_dir_all(root.join(path)).unwrap();
        }
        fs::write(root.join("evaluator"), b"new evaluator").unwrap();
        fs::write(root.join("corpus/vocabulary.json"), b"{}").unwrap();
        fs::write(root.join("checkpoint/state.json"), b"checkpoint").unwrap();
        let vocab = file_hash(&root.join("corpus/vocabulary.json")).unwrap();
        let evaluator = file_hash(&root.join("evaluator")).unwrap();
        let prepared = json!({"schema":"parallel.prepared.v1","dataset":"d","family":"f","task":"regression",
            "meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,
            "clean_status":"raw_reconstructed","original_train_sha256":"0","original_test_sha256":"1",
            "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
            "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"2".repeat(64),
                "validation_sha256":"3".repeat(64),"preprocessing_sha256":"6".repeat(64)}]});
        write_json(&root.join("prepared/prepared.json"), &prepared).unwrap();
        let prepared_sha = file_hash(&root.join("prepared/prepared.json")).unwrap();
        let mut old = observation(1.0);
        old["prepared_sha256"] = json!(prepared_sha);
        old["vocabulary_sha256"] = json!(vocab);
        record(root, &old, Path::new("old/observations.json")).unwrap();
        let mut current = old.clone();
        current["evaluator_sha256"] = json!(evaluator);
        current["metrics"]["rmse"] = json!(100.0);
        record(root, &current, Path::new("new/observations.json")).unwrap();
        write_json(
            &root.join("runs/gp-service.json"),
            &json!({"executable":"evaluator","executable_sha256":evaluator,
            "vocabulary_sha256":vocab,"timestamp":crate::telemetry::now()}),
        )
        .unwrap();
        write_json(&root.join("corpus/index.json"),&json!({"schema":"parallel.teaching_corpus.v1","qualification":true,
            "vocabulary_sha256":vocab,"context_index_sha256":"x","records_sha256":"x","record_bytes":0,"records":0,
            "legal_sets":[],"groups":[],"contexts":[{"dataset":"d","family":"f","cut":0,"partition":"meta_train",
                "path":"x","sha256":"x","prepared":"prepared","prepared_sha256":prepared_sha}]})).unwrap();
        let a = freeze_snapshot(
            root,
            &root.join("corpus"),
            &root.join("checkpoint"),
            &root.join("cycle/neural"),
            2,
            0,
        )
        .unwrap();
        let frozen = fs::read(&a).unwrap();
        let s: Snapshot = read_json(&a).unwrap();
        assert_eq!(s.entries.len(), 2);
        assert_eq!(s.entries[0].rmse_original_units, 200.0);
        assert_eq!(s.entries[1].rmse_original_units, 2.0);
        old["metrics"]["rmse"] = json!(0.1);
        record(root, &old, Path::new("improved/observations.json")).unwrap();
        let b = freeze_snapshot(
            root,
            &root.join("corpus"),
            &root.join("checkpoint"),
            &root.join("cycle/shuffled"),
            2,
            0,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(frozen, fs::read(b).unwrap());
        assert!(freeze_snapshot(
            root,
            &root.join("corpus"),
            &root.join("checkpoint"),
            &root.join("cycle/shuffled"),
            3,
            0
        )
        .is_err());
    }
    #[test]
    fn recovered_sealed_and_unknown_pending_never_publish_or_materialize() {
        let donor = tempfile::tempdir().unwrap();
        record(donor.path(), &observation(3.0), Path::new("donor")).unwrap();
        let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
        let original: Champion = read_json(
            &donor
                .path()
                .join("runs/champions")
                .join(identity.key().unwrap())
                .join("current.json"),
        )
        .unwrap();
        for partition in ["meta_test", "unknown"] {
            let mut champion = original.clone();
            champion.observation["partition"] = json!(partition);
            champion.identity.partition = partition.into();
            champion.observation_sha256 = hash(&serde_json::to_vec(&champion.observation).unwrap());
            champion.validate().unwrap();
            let root = tempfile::tempdir().unwrap();
            let directory = root
                .path()
                .join("runs/champions")
                .join(champion.identity.key().unwrap());
            fs::create_dir_all(&directory).unwrap();
            let pending = PendingChampion {
                champion: champion.clone(),
                publish: true,
            };
            let bytes = serde_json::to_vec(&pending).unwrap();
            fs::write(directory.join("pending.json"), &bytes).unwrap();
            assert!(recover_outbox(root.path()).is_err());
            assert_eq!(fs::read(directory.join("pending.json")).unwrap(), bytes);
            assert!(!directory.join("origins").exists());
            assert!(!directory.join("current.json").exists());
            assert!(!root.path().join("runs/champions-outbox").exists());
            assert!(!fs::read_dir(&directory).unwrap().any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("event-")));
            let outbox = root.path().join("standalone-outbox");
            fs::create_dir(&outbox).unwrap();
            let pending = PendingPublication {
                sequence: 0,
                champion,
            };
            let bytes = serde_json::to_vec(&pending).unwrap();
            fs::write(outbox.join("pending.json"), &bytes).unwrap();
            assert!(finish_publication(&outbox, &pending).is_err());
            assert_eq!(fs::read(outbox.join("pending.json")).unwrap(), bytes);
            assert_eq!(fs::read_dir(&outbox).unwrap().count(), 1);
        }
    }
}
