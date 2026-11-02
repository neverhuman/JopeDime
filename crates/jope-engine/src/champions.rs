//! Durable best-found teachers. Scores never cross a cut or evaluator identity.
use crate::{manifest::*, prepare::Prepared};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
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
    ensure!(lock.lock().is_ok(), "winner outbox lock failed");
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
        ensure!(lock.lock().is_ok(), "champion recovery lock failed");
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
    ensure!(lock.lock().is_ok(), "champion lock failed");
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
    ensure!(lock.lock().is_ok(), "snapshot lock failed");
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
    for (_, row, path) in best.values() {
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
mod tests;
