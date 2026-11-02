use super::*;

pub(super) fn score(o: &Value) -> Result<Option<f64>> {
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

pub(super) fn record_inner(
    root: &Path,
    observation: &Value,
    source: &Path,
    publish: bool,
) -> Result<bool> {
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
