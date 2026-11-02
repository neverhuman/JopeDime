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

pub(crate) fn render_cohort_proposer_spec(
    args: &Args,
    inputs: &CheckpointInputs,
    state: &PersistedSupplyState,
) -> Result<PathBuf> {
    let pointer_directory = args.root.join("proposals/cohort-inputs");
    fs::create_dir_all(&pointer_directory)?;
    let pointer_snapshot = pointer_directory.join(format!("{}.pointer.json", state.catalog_sha256));
    // Once a cohort is frozen, resume exclusively from its immutable catalog
    // snapshot. Breadth encoding may advance the live catalog independently
    // while a proposer is queued or running.
    let catalog_pointer = frozen_catalog_pointer(
        &args.catalog_pointer(),
        &pointer_snapshot,
        &state.catalog_sha256,
    )?;
    let catalog = PathBuf::from(
        catalog_pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    );
    let data = PathBuf::from(
        catalog_pointer
            .get("data")
            .and_then(Value::as_str)
            .context("catalog pointer lacks data")?,
    );
    let job_id = state
        .proposal_job_id
        .as_deref()
        .context("cohort proposer job identity is absent")?;
    let pythonpath = fs::canonicalize(&args.proposal_pythonpath)
        .unwrap_or_else(|_| args.proposal_pythonpath.clone());
    let operator_registry = pythonpath.join("registry/operator_registry.json");
    let runtime_bundle_sha256 = pythonpath
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .context("cohort proposer runtime bundle has no content identity")?
        .to_owned();
    validate_sha256(&runtime_bundle_sha256)?;
    let proposal_generator = fs::canonicalize(&args.proposal_generator)
        .unwrap_or_else(|_| args.proposal_generator.clone());
    let proposal_legacy_generator = fs::canonicalize(&args.proposal_legacy_generator)
        .unwrap_or_else(|_| args.proposal_legacy_generator.clone());
    let proposal_common =
        fs::canonicalize(&args.proposal_common).unwrap_or_else(|_| args.proposal_common.clone());
    let request = context_proposal_request(
        args,
        state,
        &pointer_snapshot,
        &catalog_pointer,
        &runtime_bundle_sha256,
    )?;
    let cache_root = args.root.join("proposals/context-cache.v3");
    let args_line = format!(
        "args = [\"{{input:generator}}\", \"--catalog-pointer\", \"{{input:pointer}}\", \
         \"--catalog\", \"{{input:catalog}}\", \"--data\", \"{{input:data}}\", \
         \"--checkpoint\", \"{{input:checkpoint}}\", \"--request\", \"{{input:request}}\", \
         \"--cache-root\", {}, \"--output\", \"{{output:proposals}}/dl-proposals.jsonl\", \
         \"--device\", \"cuda\"]",
        toml_quote(&cache_root.to_string_lossy())?
    );
    let mut lines = vec![
        "[job]".into(),
        format!("id = {}", toml_quote(job_id)?),
        "project = \"jope-selfplay-train\"".into(),
        "tier = \"normal\"".into(),
        "command = \"/usr/bin/python3\"".into(),
        args_line,
        String::new(),
        "[resource]".into(),
        "gpus_per_task = 1".into(),
        "cpus_reserved = 4".into(),
        "omp_threads = 4".into(),
        "ram_mb = 12288".into(),
        "gang_size = 1".into(),
        "placement = \"single_node\"".into(),
        String::new(),
        "[[dependencies.smoke]]".into(),
        "name = \"proposal GPU capacity\"".into(),
        "command = \"/usr/bin/nvidia-smi\"".into(),
        "args = [\"--query-gpu=uuid,memory.total\", \"--format=csv,noheader\"]".into(),
        "timeout_secs = 5".into(),
        String::new(),
        "[[dependencies.smoke]]".into(),
        "name = \"RuntimeBundle JOPE Python package\"".into(),
        "command = \"/usr/bin/env\"".into(),
        format!(
            "args = [{}, \"/usr/bin/python3\", \"-c\", \"import jope\"]",
            toml_quote(&format!("PYTHONPATH={}", pythonpath.to_string_lossy()))?
        ),
        "timeout_secs = 30".into(),
        String::new(),
        // The registry is part of the content-addressed RuntimeBundle and is a
        // hash-pinned job input. Probe that exact file rather than a hostname
        // or mutable checkout so placement and execution share one dependency.
        "[[dependencies.smoke]]".into(),
        "name = \"RuntimeBundle operator registry\"".into(),
        "command = \"/usr/bin/test\"".into(),
        format!(
            "args = [\"-f\", {}]",
            toml_quote(&operator_registry.to_string_lossy())?
        ),
        "timeout_secs = 5".into(),
        String::new(),
        "[env]".into(),
        format!(
            "PYTHONPATH = {}",
            toml_quote(&pythonpath.to_string_lossy())?
        ),
        format!(
            "JOPE_OPERATOR_REGISTRY = {}",
            toml_quote(&operator_registry.to_string_lossy())?
        ),
        "OMP_NUM_THREADS = \"4\"".into(),
        String::new(),
        "[preempt]".into(),
        "preemptible = true".into(),
        "checkpoint = \"none\"".into(),
        "resume = \"rerun\"".into(),
        String::new(),
        "[limits]".into(),
        "max_runtime_hours = 24".into(),
        "max_restarts = 1".into(),
        String::new(),
    ];
    append_spec_input(
        &mut lines,
        "generator",
        &proposal_generator,
        "test_exact_context_proposals.py",
    )?;
    append_spec_input(
        &mut lines,
        "legacy_generator",
        &proposal_legacy_generator,
        "test_exact_dl_generate.py",
    )?;
    append_spec_input(
        &mut lines,
        "selection_policy",
        &pythonpath.join("proposals/proposal-selection-policy.v1.json"),
        "proposal-selection-policy.v1.json",
    )?;
    append_spec_input(
        &mut lines,
        "common",
        &proposal_common,
        "test_exact_common.py",
    )?;
    append_spec_input(
        &mut lines,
        "operator_registry",
        &operator_registry,
        "registry/operator_registry.json",
    )?;
    append_spec_input(
        &mut lines,
        "pointer",
        &pointer_snapshot,
        "catalog/pointer.json",
    )?;
    append_spec_input(&mut lines, "catalog", &catalog, "catalog/catalog.json")?;
    append_spec_input(&mut lines, "data", &data, "catalog/contexts.f32.bin")?;
    append_spec_input(
        &mut lines,
        "checkpoint",
        &inputs.checkpoint_artifact,
        &format!("checkpoint/{}.model.pt", state.checkpoint_sha256),
    )?;
    append_spec_input(&mut lines, "cohort", &state.cohort, "cohort/cohort.json")?;
    append_spec_input(
        &mut lines,
        "request",
        &request,
        "cohort/context-proposal-request.v4.json",
    )?;
    lines.extend([
        "[[output]]".into(),
        "name = \"proposals\"".into(),
        "source = \"proposals\"".into(),
        "directory = true".into(),
        String::new(),
    ]);
    let directory = args.root.join("specs/cohort-propose");
    fs::create_dir_all(&directory)?;
    let spec = directory.join(format!("{job_id}.toml"));
    let body = lines.join("\n");
    if spec.is_file() {
        bail!(
            "refusing to re-render existing immutable cohort proposer spec {}",
            spec.display()
        );
    }
    fs::write(&spec, body)?;
    Ok(spec)
}

/// The cohort proposer is NOT placement-pinned. Kept as a function so the reason
/// survives, exactly as `spec_policy::PROBE_FLYWHEEL_ROOT_XBABE2_ONLY` does.
///
/// It used to emit `[dependencies] binaries = ["<root>/catalog/current.json"]`,
/// which scq resolves as a node-local file requirement. That path exists only on
/// the state-root node, so placement collapsed to xbabe2 alone. On 2026-08-04
/// that deadlocked the whole flywheel: the mobile shadow trainer took xbabe2's
/// single GPU under a 168-hour cap, the immobile proposer queued behind it
/// forever on "need 1 any GPU(s) without external use", no waves were produced,
/// and the eval queue drained to zero while three other GPUs sat idle.
///
/// The pin is redundant on both of its grounds:
///
/// 1. Its stated rationale was the ~46 s launch deadline being shorter than
///    cross-node staging. That deadline was `VANISH_GRACE_SECS = 45` in the scq
///    controller and is now **180** (measured launch results return at 65-86 s),
///    so cross-node dispatch no longer dies "vanished from its node".
/// 2. The proposer reads NOTHING from a state-root host path at runtime. The
///    catalog, the context blob, the pointer, the checkpoint, the cohort and the
///    request are all staged as sha256-pinned `[[input]]`s into the job's own
///    directory -- which is precisely the rule `PROBE_PRIME_POLICY_SRC` states:
///    staged inputs "must NOT be probed".
///
/// Node capability is still enforced, by the two smoke probes the spec already
/// carries: `proposal GPU capacity` (nvidia-smi) and `RuntimeBundle JOPE Python
/// package` (`import jope`). Those exclude a node that genuinely cannot run the
/// job -- e.g. xbabe0, which has a free GPU but no runtime bundle -- on real
/// capability rather than on an accident of file placement.
#[cfg(test)]
pub(crate) fn proposer_placement_pin_lines(_args: &Args) -> Result<Vec<String>> {
    Ok(Vec::new())
}

pub(crate) fn resolve_pinned_proposer_spec(
    args: &Args,
    inputs: &CheckpointInputs,
    state: &mut PersistedSupplyState,
) -> Result<PathBuf> {
    if let Some(intent) = state.runtime_pinned_attempts.last() {
        intent.validate()?;
        if Some(intent.job_identity.as_str()) != state.proposal_job_id.as_deref()
            || intent.attempt != state.proposal_attempt
        {
            bail!("current runtime-pinned proposer attempt identity drift");
        }
        return Ok(intent.immutable_spec_path.clone());
    }
    let job_identity = state
        .proposal_job_id
        .as_deref()
        .context("cohort proposer job identity is absent")?;
    let candidate = args
        .root
        .join("specs/cohort-propose")
        .join(format!("{job_identity}.toml"));
    let spec = if candidate.is_file() {
        candidate
    } else {
        render_cohort_proposer_spec(args, inputs, state)?
    };
    let pinned = runtime_pinned_intent_from_spec(&spec, state.proposal_attempt)?;
    if pinned.job_identity != job_identity {
        bail!("cohort proposer spec job identity drift");
    }
    state.runtime_pinned_attempts.push(pinned);
    state.revision = state.revision.saturating_add(1);
    state.validate(&state.wave, &state.checkpoint_sha256)?;
    Ok(spec)
}

pub(crate) fn run_scq(args: &Args, arguments: &[&str]) -> Result<String> {
    let output = Command::new(&args.scq)
        .args(arguments)
        .output()
        .with_context(|| format!("spawn {} {:?}", args.scq.display(), arguments))?;
    if !output.status.success() {
        bail!(
            "scq {:?} failed: {}",
            arguments,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Is this scq rejection "come back later", or a real configuration error?
///
/// scq probes dependencies only on nodes that ALREADY have the requested
/// GPU/cores/RAM free, so a perfectly healthy cluster rejects a submit whenever
/// every capable node happens to be busy. That must queue, not `bail!` -- a bail
/// fails the supply cycle and opens the circuit breaker.
///
/// This used to name ONE probe (`smoke proposal GPU capacity`) literally. That
/// made it fragile in exactly the way `spec_policy::PROBE_FLYWHEEL_ROOT_XBABE2_ONLY`
/// documents: when the proposer's probe changed, a busy-node rejection stopped
/// matching and produced **44 consecutive `cycle_failed` records**. Now that the
/// proposer probes `jope-prime policy source present` instead, a busy-node
/// rejection reads `... smoke jope-prime policy source present: smoke test exited
/// with exit 1`, which the old literal would also have missed.
///
/// So match the SHAPE of a probe failure rather than any probe's name. scq renders
/// one as `<node>: smoke <probe name>: smoke test exited with exit <n>`, so the
/// discriminator is the delimiter **`": smoke test exited"`** -- a named probe
/// that ran and failed. Bare prose that merely happens to contain the words is
/// not a probe result and stays fatal, which is what
/// `proposer_capacity_pressure_is_a_normal_backoff` has always asserted.
///
/// A genuinely mis-provisioned node (e.g. xbabe0, which has a free GPU but no
/// runtime bundle) lands here too -- correctly, because the right response is to
/// keep queueing for a node that can run the job, not to fail the supply loop.
pub(crate) fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            || (message.contains("smoke ") && message.contains(": smoke test exited")))
}

pub(crate) fn register_proposer_admission(args: &Args, spec: &Path) -> Result<()> {
    let job_id = fs::read_to_string(spec)?
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("id = ")
                .and_then(|value| serde_json::from_str::<String>(value).ok())
        })
        .context("cohort proposer spec lacks job identity")?;
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let controller = executable
        .parent()
        .context("supply executable has no bin directory")?
        .join("jope_gpu_lane_admission");
    let output = Command::new(&controller)
        .args([
            "--root",
            args.root.to_string_lossy().as_ref(),
            "--register-role",
            "cohort-proposer",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            &job_id,
        ])
        .output()
        .with_context(|| {
            format!(
                "register proposer admission through {}",
                controller.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "cohort proposer GPU admission request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Capacity-only failures are handed to the crash-safe GPU admission owner.
/// The supply state remains at the submit stage until that owner has placed
/// the immutable spec.
pub(crate) fn scq_check_submit(args: &Args, spec: &Path) -> Result<bool> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec_text.as_ref()])
        .output()
        .with_context(|| format!("spawn {} check {}", args.scq.display(), spec.display()))?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && temporarily_lacks_launch_capacity(&check_message) {
        register_proposer_admission(args, spec)?;
        return Ok(false);
    }
    if !check.status.success() {
        bail!(
            "SCQ check rejected cohort proposer: {}",
            check_message.trim()
        );
    }

    let submit = Command::new(&args.scq)
        .args(["submit", spec_text.as_ref()])
        .output()
        .with_context(|| format!("spawn {} submit {}", args.scq.display(), spec.display()))?;
    let submit_message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if submit.status.success()
        || submit_message
            .to_ascii_lowercase()
            .contains("already exists")
    {
        return Ok(true);
    }
    if temporarily_lacks_launch_capacity(&submit_message) {
        register_proposer_admission(args, spec)?;
        return Ok(false);
    }
    bail!(
        "SCQ submit rejected cohort proposer: {}",
        submit_message.trim()
    )
}

pub(crate) fn proposer_job_state(args: &Args, job_id: &str) -> Result<Option<String>> {
    let value: Value = serde_json::from_str(&run_scq(args, &["--json", "ls"])?)?;
    Ok(value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| {
            jobs.iter().find_map(|job| {
                (job.get("id").and_then(Value::as_str) == Some(job_id)).then(|| {
                    job.get("state")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_owned()
                })
            })
        }))
}

pub(crate) fn recursive_find(directory: &Path, name: &str, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            recursive_find(&path, name, found);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            found.push(path);
        }
    }
}

pub(crate) fn fetch_cohort_proposals(
    args: &Args,
    state: &PersistedSupplyState,
) -> Result<(PathBuf, String)> {
    let job_id = state
        .proposal_job_id
        .as_deref()
        .context("proposal job identity is absent")?;
    let incoming = args.root.join("proposals/_incoming").join(job_id);
    if incoming.exists() {
        fs::remove_dir_all(&incoming)?;
    }
    fs::create_dir_all(&incoming)?;
    run_scq(
        args,
        &["fetch", job_id, "--output", &incoming.to_string_lossy()],
    )?;
    let mut proposals = Vec::new();
    let mut summaries = Vec::new();
    recursive_find(&incoming, "dl-proposals.jsonl", &mut proposals);
    recursive_find(&incoming, "dl-proposals.summary.json", &mut summaries);
    let proposal_source = proposals
        .into_iter()
        .next()
        .context("cohort proposer output lacks proposals")?;
    let summary_source = summaries
        .into_iter()
        .next()
        .context("cohort proposer output lacks summary")?;
    let summary: Value = serde_json::from_slice(&fs::read(&summary_source)?)?;
    let proposal_sha256 = sha256_file(&proposal_source)?;
    if summary
        .get("model_checkpoint_sha256")
        .and_then(Value::as_str)
        != Some(state.checkpoint_sha256.as_str())
        || summary.get("catalog_sha256").and_then(Value::as_str)
            != Some(state.catalog_sha256.as_str())
        || summary.get("cohort_sha256").and_then(Value::as_str)
            != Some(state.cohort_sha256.as_str())
        || summary.get("output_sha256").and_then(Value::as_str) != Some(proposal_sha256.as_str())
        // Any legal micro-wave shape (tasks.len() * 64). The cohort identity is
        // already pinned by cohort_sha256 above, so this is a sanity check on
        // shape, not the provenance authority. NOTE for activation: a 192-wave
        // at max_candidates_per_manifest=25 emits 2,112 manifests, which breaks
        // `queued_target + manifests_per_wave <= queued_hard_cap` -- raising the
        // manifest cap (validated 25..=100) is part of flipping activation on.
        || !matches!(
            summary.get("datasets").and_then(Value::as_u64),
            Some(128 | 192 | 256)
        )
        || summary
            .get("zero_cross_context_backfill")
            .and_then(Value::as_bool)
            != Some(true)
    {
        bail!("cohort proposer output provenance drift");
    }
    let validity = summary
        .get("proposal_validity")
        .and_then(Value::as_f64)
        .context("cohort proposer summary lacks validity")?;
    if validity < 0.99 {
        bail!("cohort proposal validity {validity:.6} is below 0.99");
    }
    let cohort_value: Value = serde_json::from_slice(&fs::read(&state.cohort)?)?;
    let mut task_counts = BTreeMap::<String, usize>::new();
    for entry in cohort_value
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen cohort lacks entries")?
    {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks task")?;
        *task_counts.entry(task.to_owned()).or_default() += 1;
    }
    if task_counts.values().any(|count| *count != 64) {
        bail!("cohort proposal task scope is not 64 contexts per task");
    }
    let destination = args
        .root
        .join("proposals/by-cohort")
        .join(&state.cohort_sha256);
    fs::create_dir_all(&destination)?;
    let proposals_path = destination.join(format!("{proposal_sha256}.jsonl"));
    if !proposals_path.is_file() {
        fs::copy(&proposal_source, &proposals_path)?;
    }
    if sha256_file(&proposals_path)? != proposal_sha256 {
        bail!("published cohort proposal content drift");
    }
    let summary_sha256 = sha256_file(&summary_source)?;
    let summary_path = destination.join(format!("{summary_sha256}.summary.json"));
    if !summary_path.is_file() {
        fs::copy(&summary_source, &summary_path)?;
    }
    // Every task the cohort carries gets a DL allocation at the historical
    // 8-slot rung. The literal {"binary": 8, "regression": 8} that used to sit
    // in the pointer stranded wave 552 for hours: the wave builder asks
    // `normalized_dl_slots` for each DECLARED task, and a three-task cohort
    // met a two-key map -- "proposal pointer lacks multiclass DL allocation",
    // retried forever against the same immutable pointer.
    let dl_slots_by_task = task_counts
        .keys()
        .map(|task| (task.clone(), json!(8)))
        .collect::<serde_json::Map<_, _>>();
    let task_scopes = task_counts
        .into_iter()
        .map(|(task, context_count)| {
            json!({
                "task": task,
                "accepted_checkpoint_sha256": state.checkpoint_sha256,
                "context_count": context_count,
            })
        })
        .collect::<Vec<_>>();
    let pointer = json!({
        "schema": "CohortProposalPointer.v1",
        "immutable": true,
        "task_scopes": task_scopes,
        "cohort": state.cohort,
        "cohort_sha256": state.cohort_sha256,
        "catalog_sha256": state.catalog_sha256,
        "checkpoint_sha256": state.checkpoint_sha256,
        "decoder_configuration_sha256": state.decoder_configuration_sha256,
        "seed": state.seed,
        "proposal_job_id": job_id,
        "proposals": proposals_path,
        "proposals_sha256": proposal_sha256,
        "summary": summary_path,
        "summary_sha256": summary_sha256,
        "dl_slots_by_task": dl_slots_by_task,
        "cross_context_backfill_rows": 0,
    });
    let mut bytes = serde_json::to_vec(&pointer)?;
    bytes.push(b'\n');
    let pointer_sha256 = sha256_bytes(&bytes);
    let pointer_path = destination.join(format!("{pointer_sha256}.pointer.json"));
    if pointer_path.is_file() {
        if fs::read(&pointer_path)? != bytes {
            bail!("immutable cohort proposal pointer drift");
        }
    } else {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pointer_path)?;
        file.write_all(&bytes)?;
        file.sync_data()?;
    }
    Ok((pointer_path, pointer_sha256))
}
