use super::*;

fn plain_file(path: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(path)
        .with_context(|| format!("checkpoint object missing: {}", path.display()))?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "checkpoint object is not a regular file: {}",
        path.display()
    );
    Ok(())
}

/// Load the run's published checkpoint and verify it is complete.
pub fn load_latest(root: &Path, run: &Path, expect: &Expectation) -> Result<Committed> {
    let latest_path = run.join("latest.json");
    plain_file(&latest_path)?;
    let latest: Value = read_json(&latest_path)?;
    let step = latest["step"]
        .as_u64()
        .context("published checkpoint step absent")?;
    let dir = resolve(
        root,
        Path::new(
            latest["checkpoint"]
                .as_str()
                .context("published checkpoint path absent")?,
        ),
    )?;
    ensure!(
        dir == run.canonicalize()?.join(format!("checkpoints/step-{step}")),
        "published checkpoint is outside this run or names another step"
    );
    let committed = verify_dir(root, &dir, step, expect)?;
    ensure!(
        latest["state_sha256"] == committed.state_sha256.as_str(),
        "published checkpoint manifest changed"
    );
    Ok(committed)
}

fn field_present(state: &Value, field: &str) -> bool {
    state.get(field).is_some_and(|v| !v.is_null())
}

/// Verify a checkpoint directory as a loader requires it: every object
/// present and bound, the manifest complete and internally consistent, and
/// every caller expectation met. Nothing is loaded into a model.
pub fn verify_dir(root: &Path, dir: &Path, step: u64, expect: &Expectation) -> Result<Committed> {
    let meta = fs::symlink_metadata(dir)
        .with_context(|| format!("checkpoint directory missing: {}", dir.display()))?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "checkpoint directory is not a real directory"
    );
    let state_path = dir.join("state.json");
    plain_file(&state_path)?;
    let state_bytes = fs::read(&state_path)?;
    let state: Value =
        serde_json::from_slice(&state_bytes).context("torn or malformed checkpoint manifest")?;
    ensure!(
        state["schema"] == SCHEMA && state["step"].as_u64() == Some(step),
        "checkpoint manifest schema or step differs"
    );
    for field in RESUME_FIELDS {
        ensure!(
            field_present(&state, field),
            "checkpoint manifest lacks {field}"
        );
    }
    for (name, field) in REQUIRED_OBJECTS {
        let path = dir.join(name);
        plain_file(&path)?;
        ensure!(
            state[field].is_string() && file_hash(&path)? == state[field],
            "checkpoint object {name} differs from its manifest digest"
        );
    }
    let optimizer: Value = read_json(&dir.join("optimizer.json"))?;
    ensure!(
        optimizer["step"].as_u64() == Some(step)
            && optimizer["sha256"] == state["optimizer_sha256"]
            && optimizer["lr"]
                .as_f64()
                .is_some_and(|lr| lr.is_finite() && lr > 0.0)
            && optimizer["decay"]
                .as_f64()
                .is_some_and(|decay| decay.is_finite() && decay >= 0.0)
            && optimizer["backend"].is_string(),
        "optimizer metadata is inconsistent with the manifest"
    );
    let _: crate::neural::Config =
        read_json(&dir.join("config.json")).context("checkpoint configuration invalid")?;
    ensure!(
        state["seed"].is_u64() && state["sampler_id"].as_str().is_some_and(|s| !s.is_empty()),
        "checkpoint RNG or sampler identity absent"
    );
    let world = usize::try_from(state["world"].as_u64().context("checkpoint world")?)?;
    let global = usize::try_from(
        state["global_batch"]
            .as_u64()
            .context("checkpoint global batch")?,
    )?;
    let micro = usize::try_from(
        state["execution_partition"]["microbatch"]
            .as_u64()
            .context("checkpoint microbatch")?,
    )?;
    // W7-T06: the record's own schema selects v1 or v2; v1 is recomputed
    // exactly as before.
    let partition_schema = state["execution_partition"]["schema"]
        .as_str()
        .context("checkpoint execution partition schema")?;
    ensure!(
        state["execution_partition"]
            == crate::task_contract::training_partition_of(partition_schema, world, micro, global)?,
        "checkpoint execution partition is inconsistent"
    );
    let v2 = partition_schema == crate::task_contract::PARTITION_V2_SCHEMA;
    let rows_per_rank =
        crate::task_contract::partition_rows_per_rank(&state["execution_partition"])?;
    for (expected, field) in [
        (expect.corpus_sha256, "corpus_sha256"),
        (expect.vocabulary_sha256, "vocabulary_sha256"),
    ] {
        if let Some(expected) = expected {
            ensure!(
                state[field] == expected,
                "checkpoint {field} differs from the expected input"
            );
        }
    }
    ensure!(
        expect.world.is_none_or(|w| w == world) && expect.global_batch.is_none_or(|g| g == global),
        "checkpoint topology differs from the expected run"
    );
    if field_present(&state, "objective_sha256") {
        ensure!(
            state["objective_sha256"].as_str()
                == Some(super::super::objective::identity_digest(&state["objective"]).as_str()),
            "checkpoint objective identity does not match its digest"
        );
    }
    if state.get("committed_exposure_by_rank").is_some() {
        for rank in 0..world {
            crate::training_exposure::checkpoint_baseline_rows(&state, rank, &rows_per_rank)?;
        }
    }
    let plan = if v2 {
        let plan = super::super::partition::Plan::new(world, global, micro)?;
        verify_partition_state(&state, step, &plan)?;
        Some(plan)
    } else {
        None
    };
    let format = if field_present(&state, "checkpoint_format") {
        ensure!(
            state["checkpoint_format"] == FORMAT || state["checkpoint_format"] == SUCCESSOR_FORMAT,
            "unknown checkpoint format"
        );
        verify_complete(root, &state, step, world, global, plan.as_ref())?;
        if state["checkpoint_format"] == SUCCESSOR_FORMAT {
            verify_successor(root, &state, step, world)?;
            ensure!(
                expect
                    .lineage_sha256
                    .is_none_or(|sha| state["lineage"]["sha256"] == sha),
                "checkpoint lineage differs from the expected identity"
            );
            ensure!(
                expect
                    .intent_sha256
                    .is_none_or(|sha| state["intent"]["sha256"] == sha),
                "checkpoint intent differs from the expected identity"
            );
            ensure!(
                expect.attempt_id.is_none_or(|id| state["attempt_id"] == id),
                "checkpoint belongs to another writer attempt"
            );
            ensure!(
                expect.window_id.is_none_or(|id| state["window_id"] == id),
                "checkpoint belongs to another training window"
            );
            Format::Successor
        } else {
            Format::Complete
        }
    } else {
        ensure!(
            !v2,
            "a partition-v2 checkpoint must use the complete commit format"
        );
        ensure!(
            !expect.require_complete,
            "checkpoint predates the complete commit format"
        );
        Format::Legacy
    };
    ensure!(
        (expect.lineage_sha256.is_none()
            && expect.intent_sha256.is_none()
            && expect.attempt_id.is_none()
            && expect.window_id.is_none())
            || format == Format::Successor,
        "checkpoint identity expectations require successor commit format"
    );
    Ok(Committed {
        dir: dir.to_path_buf(),
        step,
        state_sha256: crate::manifest::hash(&state_bytes),
        state,
        format,
    })
}

/// Successor-only immutable identity and per-rank closure checks. The shapes
/// are intentionally generic JSON bindings so I11/I12 can supply their
/// versioned payloads without changing legacy checkpoint bytes.
fn verify_successor(root: &Path, state: &Value, step: u64, world: usize) -> Result<()> {
    for field in ["lineage", "intent"] {
        let binding = state
            .get(field)
            .context(format!("successor lacks {field}"))?;
        let schema = binding["schema"]
            .as_str()
            .context(format!("{field} schema absent"))?;
        let digest = binding["sha256"]
            .as_str()
            .context(format!("{field} digest absent"))?;
        let value = binding
            .get("value")
            .context(format!("{field} value absent"))?;
        ensure!(
            schema.starts_with("jope.")
                && digest == crate::manifest::hash(&serde_json::to_vec(value)?),
            "successor {field} binding malformed"
        );
        ensure!(
            value["schema"] == schema,
            "successor {field} payload schema mismatch"
        );
    }
    let attempt = state["attempt_id"]
        .as_str()
        .context("successor attempt identity absent")?;
    let window = state["window_id"]
        .as_str()
        .context("successor window identity absent")?;
    ensure!(
        !attempt.is_empty() && !window.is_empty(),
        "successor attempt/window identity empty"
    );
    ensure!(
        state["applied_boundary"].as_u64() == Some(step),
        "successor applied boundary differs from checkpoint step"
    );
    let prefixes = state["journal_prefixes"]
        .as_array()
        .context("successor journal prefixes absent")?;
    ensure!(
        prefixes.len() == world,
        "successor must bind every rank journal prefix"
    );
    for (rank, entry) in prefixes.iter().enumerate() {
        ensure!(
            entry["rank"].as_u64() == Some(rank as u64),
            "successor journal prefix rank order differs"
        );
        ensure!(
            entry["attempt_id"] == attempt && entry["window_id"] == window,
            "rank journal prefix belongs to another attempt/window"
        );
        let path = resolve(
            root,
            Path::new(
                entry["path"]
                    .as_str()
                    .context("journal prefix path absent")?,
            ),
        )?;
        plain_file(&path)?;
        let bytes = fs::read(path)?;
        let binding: super::super::consumption::Binding =
            serde_json::from_value(entry["binding"].clone())?;
        super::super::consumption::verify_binding(&bytes, &binding)?;
        let reconciled =
            super::super::consumption::reconcile(&bytes[..binding.bytes as usize], Some(step))?;
        ensure!(
            reconciled.header.rank == rank
                && reconciled.header.world == world
                && reconciled.committed_through == step
                && reconciled.applied_through == step,
            "successor rank {rank} prefix does not prove the common applied boundary"
        );
        ensure!(
            entry["applied_through"].as_u64() == Some(reconciled.applied_through),
            "successor rank applied boundary differs from journal"
        );
    }
    let objects = state["closure_objects"]
        .as_array()
        .context("successor closure objects absent")?;
    ensure!(
        !objects.is_empty(),
        "successor closure object list is empty"
    );
    for object in objects {
        let path = resolve(
            root,
            Path::new(
                object["path"]
                    .as_str()
                    .context("closure object path absent")?,
            ),
        )?;
        plain_file(&path)?;
        ensure!(
            file_hash(&path)? == object["sha256"],
            "successor closure object changed: {}",
            path.display()
        );
    }
    Ok(())
}

/// Make external objects referenced by the successor manifest durable before
/// publishing latest. The journal writer normally synced these already; doing
/// it here closes the manifest's durability contract at the commit boundary.
fn sync_successor_closure(root: &Path, staging: &Path) -> Result<()> {
    let state: Value = read_json(&staging.join("state.json"))?;
    for entry in state["journal_prefixes"]
        .as_array()
        .context("successor journal prefixes absent")?
    {
        let path = resolve(
            root,
            Path::new(
                entry["path"]
                    .as_str()
                    .context("journal prefix path absent")?,
            ),
        )?;
        fs::File::open(&path)?.sync_all()?;
        if let Some(parent) = path.parent() {
            sync_dir(parent)?;
        }
    }
    for entry in state["closure_objects"]
        .as_array()
        .context("successor closure objects absent")?
    {
        let path = resolve(
            root,
            Path::new(
                entry["path"]
                    .as_str()
                    .context("closure object path absent")?,
            ),
        )?;
        fs::File::open(&path)?.sync_all()?;
        if let Some(parent) = path.parent() {
            sync_dir(parent)?;
        }
    }
    Ok(())
}

/// A partition-v2 checkpoint binds the v2 objective and its persisted
/// partition state: every rank's RNG and sampler state and the global
/// committed update, re-derived from the plan.
fn verify_partition_state(
    state: &Value,
    step: u64,
    plan: &super::super::partition::Plan,
) -> Result<()> {
    ensure!(
        state["objective"]["normalization"] == super::super::partition::NORMALIZATION,
        "a partition-v2 checkpoint must bind the globally normalized objective"
    );
    let persisted: super::super::partition::State = serde_json::from_value(
        state
            .get("partition_state")
            .filter(|v| !v.is_null())
            .context("partition-v2 checkpoint lacks its partition state")?
            .clone(),
    )
    .context("partition state malformed")?;
    persisted.verify(plan)?;
    ensure!(
        persisted.committed_step == step
            && Some(persisted.seed) == state["seed"].as_u64()
            && Some(persisted.sampling_offset)
                == Some(
                    state["initialization"]["optimizer_parent_step"]
                        .as_u64()
                        .unwrap_or(0)
                )
            && state["parameter_sha256"] == persisted.committed_update.parameter_sha256.as_str(),
        "partition state differs from the checkpoint it is bound to"
    );
    Ok(())
}

fn verify_complete(
    root: &Path,
    state: &Value,
    step: u64,
    world: usize,
    global: usize,
    plan: Option<&super::super::partition::Plan>,
) -> Result<()> {
    for field in COMPLETE_FIELDS {
        ensure!(
            field_present(state, field),
            "complete checkpoint manifest lacks {field}"
        );
    }
    ensure!(
        state["required_objects"] == json!(required_object_names()),
        "checkpoint required-object list differs"
    );
    ensure!(
        state["rng"].as_str().is_some_and(|s| !s.is_empty())
            && state["sampler"].as_str().is_some_and(|s| !s.is_empty()),
        "checkpoint RNG or sampler identity absent"
    );
    let receipts = state["rank_receipts"]
        .as_array()
        .context("rank receipts must be a list")?;
    let exposure = state["committed_exposure_by_rank"]
        .as_array()
        .context("committed exposure must be a list")?;
    ensure!(
        receipts.len() == world && exposure.len() == world,
        "checkpoint does not bind every rank"
    );
    for rank in 0..world {
        let entry = &receipts[rank];
        ensure!(
            entry["rank"].as_u64() == Some(rank as u64),
            "rank receipt order differs"
        );
        let path = resolve(
            root,
            Path::new(entry["path"].as_str().context("rank receipt path")?),
        )?;
        plain_file(&path)?;
        ensure!(
            file_hash(&path)? == entry["sha256"],
            "rank {rank} receipt changed"
        );
        let receipt: Value = read_json(&path)?;
        ensure!(
            receipt["rank"].as_u64() == Some(rank as u64)
                && receipt["step"].as_u64() == Some(step)
                && receipt["parameter_sha256"] == state["parameter_sha256"],
            "rank {rank} receipt does not match the committed update"
        );
        let journal = &exposure[rank]["consumption_journal"];
        ensure!(
            journal == &receipt["consumption_journal"],
            "rank {rank} consumption binding differs from its receipt"
        );
        let binding: super::super::consumption::Binding =
            serde_json::from_value(journal["binding"].clone())
                .context("rank consumption binding absent")?;
        let bytes = fs::read(resolve(
            root,
            Path::new(
                journal["path"]
                    .as_str()
                    .context("consumption journal path")?,
            ),
        )?)?;
        super::super::consumption::verify_binding(&bytes, &binding)?;
        let consumed =
            super::super::consumption::reconcile(&bytes[..binding.bytes as usize], Some(step))?;
        ensure!(
            consumed.committed_through == step
                && consumed.header.rank == rank
                && consumed.header.world == world
                && consumed.header.global_batch == global
                && Some(consumed.header.seed) == state["seed"].as_u64()
                && state["corpus_sha256"] == consumed.header.corpus_sha256.as_str(),
            "rank {rank} consumption journal does not reach the committed update"
        );
        // Partition v2: the journal ran exactly this rank's shard of the
        // checkpoint's plan; a v1 journal never binds a v2 checkpoint.
        ensure!(
            consumed.shard.as_ref() == plan.map(|p| &p.shards()[rank])
                && plan.is_none_or(|p| consumed.header.microbatch == p.microbatch_rows()),
            "rank {rank} consumption journal ran another partition"
        );
    }
    Ok(())
}
