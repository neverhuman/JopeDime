use super::*;

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
