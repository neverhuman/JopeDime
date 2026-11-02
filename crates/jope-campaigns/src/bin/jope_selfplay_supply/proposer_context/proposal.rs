use super::*;

pub(crate) fn context_proposal_request(
    args: &Args,
    state: &PersistedSupplyState,
    pointer_snapshot: &Path,
    catalog_pointer: &Value,
    runtime_bundle_sha256: &str,
) -> Result<PathBuf> {
    let cohort: Value = serde_json::from_slice(&fs::read(&state.cohort)?)?;
    if sha256_file(&state.cohort)? != state.cohort_sha256 {
        bail!("frozen cohort changed before proposal request creation");
    }
    let source_schema = cohort
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("JopeMicroCohort.v2");
    let entries = cohort
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen cohort lacks entries")?;
    if entries.is_empty() {
        bail!("context proposal request cohort is empty");
    }

    if catalog_pointer
        .get("catalog_sha256")
        .and_then(Value::as_str)
        != Some(state.catalog_sha256.as_str())
    {
        bail!("frozen catalog pointer does not match the cohort binding");
    }
    let catalog_target = catalog_pointer
        .get("catalog")
        .and_then(Value::as_str)
        .context("frozen catalog pointer lacks catalog")?;
    let catalog_path = pointer_target(pointer_snapshot, catalog_target);
    if !catalog_path.is_file() || sha256_file(&catalog_path)? != state.catalog_sha256 {
        bail!("frozen catalog artifact content drift");
    }
    let catalog: Value = serde_json::from_slice(&fs::read(&catalog_path)?)?;
    let catalog_schema = catalog
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("legacy")
        .to_owned();
    let catalog_entries = catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen catalog lacks entries")?;
    let data_sha256 = catalog_pointer
        .get("data_sha256")
        .and_then(Value::as_str)
        .context("frozen catalog pointer lacks context data identity")?;
    validate_sha256(data_sha256)?;
    if catalog.get("data_sha256").and_then(Value::as_str) != Some(data_sha256) {
        bail!("frozen catalog and pointer disagree on context data");
    }
    let data_target = catalog_pointer
        .get("data")
        .and_then(Value::as_str)
        .context("frozen catalog pointer lacks context data")?;
    let data_path = pointer_target(pointer_snapshot, data_target);
    if !data_path.is_file() || sha256_file(&data_path)? != data_sha256 {
        bail!("frozen catalog context data drift");
    }
    let stride = catalog
        .get("record_stride_bytes")
        .and_then(Value::as_u64)
        .context("frozen catalog lacks context stride")?;
    if stride == 0 || data_path.metadata()?.len() != stride * catalog_entries.len() as u64 {
        bail!("frozen catalog context topology drift");
    }
    let mut by_dataset = BTreeMap::<(String, String), Vec<String>>::new();
    let mut data = File::open(&data_path)?;
    let mut context_bytes = vec![0_u8; stride as usize];
    for entry in catalog_entries {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen catalog entry lacks task")?;
        let dataset_hash = entry
            .get("dataset_hash")
            .and_then(Value::as_str)
            .context("frozen catalog entry lacks dataset hash")?;
        let context_sha256 = entry
            .get("context_sha256")
            .and_then(Value::as_str)
            .context("frozen catalog entry lacks context hash")?;
        validate_sha256(context_sha256)?;
        let index = entry
            .get("catalog_index")
            .and_then(Value::as_u64)
            .context("frozen catalog entry lacks context index")?;
        if index >= catalog_entries.len() as u64 {
            bail!("frozen catalog context index is out of range");
        }
        data.seek(SeekFrom::Start(index * stride))?;
        data.read_exact(&mut context_bytes)?;
        if sha256_bytes(&context_bytes) != context_sha256 {
            bail!("frozen catalog entry context bytes drift");
        }
        by_dataset
            .entry((task.to_owned(), dataset_hash.to_owned()))
            .or_default()
            .push(context_sha256.to_owned());
    }

    let mut identities = BTreeSet::new();
    let mut contexts = Vec::with_capacity(entries.len());
    let mut compatibility = Vec::with_capacity(entries.len());
    let mut exclusions = Vec::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks task")?;
        let dataset_hash = entry
            .get("dataset_hash")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks dataset hash")?;
        if !matches!(task, "binary" | "regression" | "timeseries" | "multiclass") {
            bail!("frozen cohort has an invalid task");
        }
        let matches = by_dataset
            .get(&(task.to_owned(), dataset_hash.to_owned()))
            .cloned()
            .unwrap_or_default();
        let source_context = entry.get("context_sha256").and_then(Value::as_str);
        let (context_sha256, resolution) = match (source_context, matches.as_slice()) {
            (Some(source), [catalog_context]) if source == catalog_context => {
                validate_sha256(source)?;
                (Some(source.to_owned()), "preserved")
            }
            (None, [catalog_context]) => (Some(catalog_context.clone()), "resolved"),
            (Some(_), [..]) => (None, "context_catalog_mismatch"),
            (_, []) => (None, "absent_context"),
            (_, [_, ..]) => (None, "ambiguous_context"),
        };
        let Some(context_sha256) = context_sha256 else {
            compatibility.push(json!({
                "entry_index": entry_index,
                "task": task,
                "dataset_hash": dataset_hash,
                "status": "excluded",
                "reason": resolution,
                "context_sha256": Value::Null,
            }));
            exclusions.push(json!({
                "schema": COHORT_CONTEXT_EXCLUSION_SCHEMA,
                "source_cohort_sha256": state.cohort_sha256,
                "source_entry_index": entry_index,
                "task": task,
                "dataset_hash": dataset_hash,
                "reason": resolution,
                "frozen_catalog_sha256": state.catalog_sha256,
            }));
            continue;
        };
        if !identities.insert((
            task.to_owned(),
            dataset_hash.to_owned(),
            context_sha256.clone(),
        )) {
            bail!("frozen cohort has a duplicate resolved context identity");
        }
        compatibility.push(json!({
            "entry_index": entry_index,
            "task": task,
            "dataset_hash": dataset_hash,
            "status": resolution,
            "reason": Value::Null,
            "context_sha256": context_sha256,
        }));
        contexts.push(json!({
            "task": task,
            "dataset_hash": dataset_hash,
            "context_sha256": context_sha256,
            "micro_lane": entry.get("micro_lane")
                .or_else(|| entry.get("lane"))
                .cloned()
                .unwrap_or(Value::Null),
        }));
    }
    let directory = args.root.join("proposals/cohort-inputs");
    fs::create_dir_all(&directory)?;
    if contexts.is_empty() {
        let receipt = json!({
            "schema": COHORT_CONTEXT_MIGRATION_SCHEMA,
            "source_cohort_sha256": state.cohort_sha256,
            "source_schema": source_schema,
            "frozen_catalog_sha256": state.catalog_sha256,
            "frozen_catalog_data_identity": data_sha256,
            "request_sha256": Value::Null,
            "resolved_entries": 0,
            "excluded_entries": exclusions.len(),
            "exclusions": exclusions,
            "compatibility_resolution": compatibility,
        });
        let receipt_path = directory.join(format!(
            "{}.{}.context-migration.v1.json",
            state.cohort_sha256, state.catalog_sha256
        ));
        immutable_json(&receipt_path, &receipt)?;
        bail!("no frozen cohort entries resolve against the frozen catalog");
    }
    let resolved_entries = contexts.len();
    let request = json!({
        "schema": "ContextProposalRequest.v4",
        "task": "mixed",
        "source_cohort_schema": source_schema,
        "source_cohort_hash": state.cohort_sha256,
        "catalog_schema": catalog_schema,
        "catalog_sha256": state.catalog_sha256,
        "catalog_data_identity": data_sha256,
        "accepted_parent_hash": state.checkpoint_sha256,
        "accepted_checkpoint_sha256": state.checkpoint_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "seed": state.seed,
        "cohort_sha256": state.cohort_sha256,
        "requested_contexts": contexts.clone(),
        "contexts": contexts.clone(),
        "compatibility_resolution": compatibility.clone(),
    });
    let path = directory.join(format!(
        "{}.{}.request.v4.json",
        state.cohort_sha256, state.checkpoint_sha256
    ));
    immutable_json(&path, &request)?;
    let request_sha256 = sha256_file(&path)?;
    let receipt = json!({
        "schema": COHORT_CONTEXT_MIGRATION_SCHEMA,
        "source_cohort_sha256": state.cohort_sha256,
        "source_schema": source_schema,
        "frozen_catalog_sha256": state.catalog_sha256,
        "frozen_catalog_data_identity": data_sha256,
        "request_sha256": request_sha256.clone(),
        "resolved_entries": resolved_entries,
        "excluded_entries": exclusions.len(),
        "exclusions": exclusions,
        "compatibility_resolution": compatibility,
    });
    let receipt_path = directory.join(format!(
        "{}.{}.{}.context-migration.v1.json",
        state.cohort_sha256, state.catalog_sha256, request_sha256
    ));
    immutable_json(&receipt_path, &receipt)?;
    Ok(path)
}

pub(crate) fn frozen_catalog_pointer(
    live_pointer: &Path,
    pointer_snapshot: &Path,
    expected_catalog_sha256: &str,
) -> Result<Value> {
    let source = if pointer_snapshot.is_file() {
        pointer_snapshot
    } else {
        live_pointer
    };
    let pointer: Value = serde_json::from_slice(&fs::read(source)?)?;
    if pointer.get("catalog_sha256").and_then(Value::as_str) != Some(expected_catalog_sha256) {
        bail!(
            "{} catalog identity does not match the frozen cohort",
            if source == pointer_snapshot {
                "snapshot"
            } else {
                "live"
            }
        );
    }
    if source == live_pointer {
        atomic_json(pointer_snapshot, &pointer)?;
    }
    Ok(pointer)
}
