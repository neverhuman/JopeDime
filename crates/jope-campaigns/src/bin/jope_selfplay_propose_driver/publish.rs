use super::*;

pub(super) fn sh(program: &Path, args: &[&str]) -> Result<(i32, String, String)> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("spawn {} {:?}", program.display(), args))?;
    Ok((
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

pub(super) fn append_event(path: &Path, event: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut line = serde_json::to_vec(event)?;
    line.push(b'\n');
    file.write_all(&line)?;
    Ok(())
}

/// Non-blocking exclusive lock; a held lock means a previous cycle is still
/// fetching a multi-hundred-megabyte proposal file and this tick must be a
/// no-op rather than a duplicate.
pub(super) fn try_lock(path: &Path) -> Result<Option<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    if flock::exclusive(&file, false).is_err() {
        return Ok(None);
    }
    Ok(Some(file))
}

// ---------------------------------------------------------------------------
// scq
// ---------------------------------------------------------------------------

pub(super) fn job_states(scq: &Path, prefix: &str) -> Result<Vec<(String, String)>> {
    let (code, stdout, stderr) = sh(scq, &["ls", "--json"])?;
    if code != 0 {
        bail!("scq ls failed ({code}): {}", stderr.trim());
    }
    let value: Value = serde_json::from_str(&stdout).context("parse scq ls --json")?;
    let jobs = value
        .get("Jobs")
        .and_then(|jobs| jobs.get("jobs"))
        .and_then(Value::as_array)
        .context("scq ls payload lacks Jobs.jobs")?;
    let mut found = Vec::new();
    for job in jobs {
        let Some(id) = job.get("id").and_then(Value::as_str) else {
            continue;
        };
        if id.starts_with(prefix) {
            let state = job
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            found.push((id.to_owned(), state));
        }
    }
    found.sort();
    Ok(found)
}

pub(super) fn is_active(state: &str) -> bool {
    matches!(state, "queued" | "running" | "dispatching" | "held")
}

/// Completed generations worth installing, best first: a run aimed at the live
/// catalog beats one aimed at an intermediate catalog, and both beat nothing.
/// `install` still refuses anything that would not move the binding forward.
pub(super) fn order_install_candidates<'a>(jobs: &'a [(String, String)], prefix: &str) -> Vec<&'a String> {
    let completed = || jobs.iter().filter(|(_, state)| state == "completed");
    let mut ordered: Vec<&String> = completed()
        .filter(|(id, _)| id.starts_with(prefix))
        .map(|(id, _)| id)
        .collect();
    ordered.extend(
        completed()
            .filter(|(id, _)| !id.starts_with(prefix))
            .map(|(id, _)| id),
    );
    ordered
}

pub(super) fn check_and_submit(scq: &Path, spec: &Path) -> Result<()> {
    let spec_text = spec.to_string_lossy().into_owned();
    let (code, _, stderr) = sh(scq, &["check", &spec_text])?;
    if code != 0 {
        bail!("scq check rejected {}: {}", spec.display(), stderr.trim());
    }
    let (code, stdout, stderr) = sh(scq, &["submit", &spec_text])?;
    if code != 0 {
        // A duplicate id is the benign race with a concurrent cycle.
        let blob = format!("{stdout}{stderr}");
        if blob.contains("already") || blob.contains("duplicate") {
            return Ok(());
        }
        bail!("scq submit failed ({code}): {}", stderr.trim());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

pub(super) struct Installed {
    pub(super) catalog_sha256: String,
    pub(super) proposals: PathBuf,
    pub(super) proposals_sha256: String,
    pub(super) summary: PathBuf,
    pub(super) summary_sha256: String,
    pub(super) counts: Value,
    pub(super) validity: Value,
    pub(super) datasets: u64,
}

/// Publish a finished generation. `bound` is the catalog the *currently
/// installed* proposal set was built for; anything newer than that is an
/// improvement and gets installed even if the live catalog has moved on again
/// since the job was submitted. Generations take hours and the expansion driver
/// flips the catalog hourly, so demanding an exact match with the live pointer
/// would throw away every completed run and never improve coverage at all.
/// Returns `Ok(None)` when the artifact would not change the binding.
pub(super) fn install(
    scq: &Path,
    root: &Path,
    job_id: &str,
    checkpoint_sha256: &str,
    bound: Option<&str>,
    task: Option<&str>,
) -> Result<Option<Installed>> {
    let staging = root.join("proposals/_incoming").join(job_id);
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    let staging_text = staging.to_string_lossy().into_owned();
    let (code, _, stderr) = sh(scq, &["fetch", job_id, "--output", &staging_text])?;
    if code != 0 {
        bail!("scq fetch {job_id} failed ({code}): {}", stderr.trim());
    }
    let mut proposals = Vec::new();
    let mut summaries = Vec::new();
    rglob(&staging, "dl-proposals.jsonl", &mut proposals);
    rglob(&staging, "dl-proposals.summary.json", &mut summaries);
    let proposals_source = proposals
        .into_iter()
        .next()
        .context("fetched outputs lack dl-proposals.jsonl")?;
    let summary_source = summaries
        .into_iter()
        .next()
        .context("fetched outputs lack dl-proposals.summary.json")?;
    let summary = read_json(&summary_source)?;

    // Provenance is verified before anything is published: a proposal file
    // built for a different checkpoint would silently invalidate every wave,
    // and one built for a different catalog would not fix coverage at all.
    let summary_checkpoint = get_str(&summary, "model_checkpoint_sha256")?;
    if summary_checkpoint != checkpoint_sha256 {
        bail!(
            "fetched proposals carry checkpoint {summary_checkpoint}, expected {checkpoint_sha256}"
        );
    }
    let summary_catalog = get_str(&summary, "catalog_sha256")?.to_owned();
    if !is_sha256(&summary_catalog) {
        bail!("fetched proposals declare a malformed catalog hash {summary_catalog}");
    }
    // Branch identity is provenance too. A per-task lane that published a
    // whole-catalog file (or the other branch's file) would put rows for the
    // wrong checkpoint epoch under a per-task name, which is precisely the
    // mixed-epoch failure `jope_selfplay_wave.rs:1072` exists to catch.
    let summary_task = summary.get("task").and_then(Value::as_str);
    if summary_task != task {
        bail!(
            "fetched proposals carry task {:?}, expected {:?}",
            summary_task,
            task
        );
    }
    if Some(summary_catalog.as_str()) == bound {
        let _ = fs::remove_dir_all(&staging);
        return Ok(None);
    }
    let proposals_sha256 = sha256_file(&proposals_source)?;
    let declared = get_str(&summary, "output_sha256")?;
    if declared != proposals_sha256 {
        bail!("fetched proposal file hash {proposals_sha256} differs from summary {declared}");
    }

    // Publish under a catalog-scoped name so the file the *old* pointer names
    // is never mutated: readers mid-sanitize keep a consistent view and the
    // pointer swap is a single atomic rename.
    let stem = format!("{checkpoint_sha256}.catalog-{}", &summary_catalog[..12]);
    let destination_dir = match task {
        Some(task) => root.join("proposals/by-task").join(task),
        None => root.join("proposals/by-checkpoint"),
    };
    fs::create_dir_all(&destination_dir)?;
    let proposals_destination = destination_dir.join(format!("{stem}.jsonl"));
    let summary_destination = destination_dir.join(format!("{stem}.summary.json"));
    move_or_copy(&proposals_source, &proposals_destination)?;
    move_or_copy(&summary_source, &summary_destination)?;
    if sha256_file(&proposals_destination)? != proposals_sha256 {
        bail!("published proposal file failed post-copy verification");
    }
    let summary_sha256 = sha256_file(&summary_destination)?;
    let _ = fs::remove_dir_all(&staging);

    Ok(Some(Installed {
        catalog_sha256: summary_catalog,
        proposals: proposals_destination,
        proposals_sha256,
        summary: summary_destination,
        summary_sha256,
        counts: summary.get("counts").cloned().unwrap_or(Value::Null),
        validity: json!({
            "proposal_validity": summary.get("proposal_validity").cloned().unwrap_or(Value::Null),
            "proposal_validity_by_task": summary
                .get("proposal_validity_by_task")
                .cloned()
                .unwrap_or(Value::Null),
            "deficient_contexts": summary
                .get("deficient_contexts")
                .and_then(Value::as_object)
                .map(|m| m.len())
                .unwrap_or(0),
        }),
        datasets: summary.get("datasets").and_then(Value::as_u64).unwrap_or(0),
    }))
}

pub(super) fn move_or_copy(source: &Path, destination: &Path) -> Result<()> {
    if fs::rename(source, destination).is_ok() {
        return Ok(());
    }
    fs::copy(source, destination)
        .with_context(|| format!("copy {} -> {}", source.display(), destination.display()))?;
    Ok(())
}

/// Rewrite proposals/current.json in place, preserving every field the other
/// consumers rely on (`legacy_adopted` resolves the checkpoint artifact in
/// selfplay_runtime and the freshness authority) and swapping only the catalog
/// binding.
pub(super) fn rebind(binding: &ProposalBinding, installed: &Installed) -> Result<Value> {
    let mut pointer: Map<String, Value> = binding
        .pointer
        .as_object()
        .cloned()
        .context("proposal pointer is not an object")?;
    pointer.insert(
        "proposals".into(),
        Value::String(installed.proposals.to_string_lossy().into_owned()),
    );
    pointer.insert(
        "proposals_sha256".into(),
        Value::String(installed.proposals_sha256.clone()),
    );
    pointer.insert(
        "summary".into(),
        Value::String(installed.summary.to_string_lossy().into_owned()),
    );
    pointer.insert(
        "summary_sha256".into(),
        Value::String(installed.summary_sha256.clone()),
    );
    pointer.insert(
        "catalog_sha256".into(),
        Value::String(installed.catalog_sha256.clone()),
    );
    pointer.insert(
        "checkpoint_sha256".into(),
        Value::String(binding.checkpoint_sha256.clone()),
    );
    // `proposals/current.json` has never carried a `schema` key, so no consumer
    // — jope_selfplay_supply.rs:96, jope_selfplay_freshness.rs:409,
    // jope_selfplay_trainer_watchdog.rs:1016, selfplay_runtime.rs:107 — can
    // version-gate it. Stamp one here (this driver is the only production
    // writer) so a future shape change is rejectable instead of silently
    // half-read. Purely additive: every consumer reads named fields off a
    // serde_json::Value.
    pointer.insert(
        "schema".into(),
        Value::String("JopeSelfplayProposalPointer.v1".into()),
    );
    pointer.entry("dl_slots_by_task").or_insert_with(
        || json!({"binary": PROPOSALS_PER_DATASET, "regression": PROPOSALS_PER_DATASET}),
    );
    pointer.insert("cross_context_backfill_rows".into(), json!(0));
    let value = Value::Object(pointer);
    atomic_json(&binding.pointer_path, &value)?;
    Ok(value)
}

/// Publish one branch into `proposals/current-by-task.json`.
///
/// ADDITIVE by construction. `proposals/current.json` is NOT written here and
/// is not written at all while the branches disagree: `jope_selfplay_wave.rs`
/// (`:1072`, `:1325`) rejects any proposal file whose rows do not all carry the
/// cohort plan's single `checkpoint_sha256`, and `jope_selfplay_queue.rs`
/// filters micro-waves by a single `checkpoint_epoch`. Handing either of them a
/// two-checkpoint pointer would fail the wave build, not improve it.
pub(super) fn rebind_task(root: &Path, lane: &Lane, installed: &Installed) -> Result<Value> {
    let pointer_path = root.join("proposals/current-by-task.json");
    let mut document = read_task_proposal_pointer(root)?;
    if document.get("tasks").and_then(Value::as_object).is_none() {
        document["tasks"] = json!({});
    }
    let task = lane
        .task
        .as_deref()
        .ok_or_else(|| anyhow!("rebind_task called for the combined lane"))?;
    let dl_slots = document
        .pointer(&format!("/tasks/{task}/dl_slots"))
        .and_then(Value::as_i64)
        .unwrap_or(PROPOSALS_PER_DATASET);
    document["tasks"][task] = json!({
        "task": task,
        "checkpoint_sha256": lane.checkpoint_sha256,
        "checkpoint": lane.checkpoint.to_string_lossy(),
        "catalog_sha256": installed.catalog_sha256,
        "proposals": installed.proposals.to_string_lossy(),
        "proposals_sha256": installed.proposals_sha256,
        "summary": installed.summary.to_string_lossy(),
        "summary_sha256": installed.summary_sha256,
        "datasets": installed.datasets,
        "counts": installed.counts.clone(),
        "validity": installed.validity.clone(),
        "dl_slots": dl_slots,
        "cross_context_backfill_rows": 0,
        "updated_at_ms": now_ms() as u64,
    });
    document["schema"] = json!(TASK_POINTER_SCHEMA);
    document["combined_pointer"] = json!(root.join("proposals/current.json").to_string_lossy());
    document["publishes_into_combined_pointer"] = json!(false);
    document["publish_note"] = json!(
        "per-task proposal sets are not published into proposals/current.json: \
         jope_selfplay_wave.rs:1072 rejects a proposal file that mixes checkpoint epochs"
    );
    document["updated_at_ms"] = json!(now_ms() as u64);
    atomic_json(&pointer_path, &document)?;
    Ok(document)
}
