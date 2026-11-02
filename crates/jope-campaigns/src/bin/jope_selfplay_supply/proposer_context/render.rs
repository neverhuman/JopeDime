use super::*;

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
