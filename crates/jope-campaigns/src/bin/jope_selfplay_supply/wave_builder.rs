use super::*;

pub(super) fn built_wave_from_disk(args: &Args, state: &PersistedSupplyState) -> Result<BuiltWave> {
    let directory = args.root.join("waves").join(&state.wave);
    let summary: Value = serde_json::from_slice(&fs::read(directory.join("summary.json"))?)?;
    Ok(BuiltWave {
        wave: state.wave.clone(),
        cohort_sha256: state.cohort_sha256.clone(),
        proposal_pointer_sha256: state
            .sanitized_proposal_pointer_sha256
            .clone()
            .context("built supply state lacks sanitized proposal identity")?,
        summary_sha256: sha256_file(&directory.join("summary.json"))?,
        manifests: summary
            .get("manifest_count")
            .and_then(Value::as_u64)
            .context("built wave summary lacks manifest count")? as usize,
    })
}

pub(super) fn create_supply_state(
    args: &Args,
    inputs: &CheckpointInputs,
    waves_root: &Path,
    wave: &str,
) -> Result<PersistedSupplyState> {
    let cohort = waves_root
        .join("pending")
        .join(format!("{wave}.cohort.json"));
    let plan = run_builder(
        &args.wave_builder,
        vec![
            "plan".into(),
            "--catalog-pointer".into(),
            args.catalog_pointer().into_os_string(),
            "--checkpoint-pointer".into(),
            inputs.checkpoint_pointer.clone().into_os_string(),
            "--corpus".into(),
            args.corpus().into_os_string(),
            "--sealed-cohort".into(),
            args.sealed_cohort().into_os_string(),
            "--output-root".into(),
            waves_root.to_path_buf().into_os_string(),
            "--wave-id".into(),
            wave.into(),
            "--allow-pending-lookahead".into(),
        ],
    )?;
    require_output_identity(&plan, wave, &inputs.checkpoint_sha256)?;
    if plan.get("cohort").and_then(Value::as_str) != Some(cohort.to_string_lossy().as_ref()) {
        bail!("{wave}: planner returned an unexpected cohort path");
    }
    let cohort_sha256 = require_sha(&plan, "cohort_sha256")?;
    let catalog_sha256 = require_sha(&plan, "catalog_sha256")?;
    let proposal_reservoir_per_context = adaptive_proposal_reservoir(&args.root);
    let decoder_configuration_sha256 =
        proposal_decoder_configuration_sha256(args, proposal_reservoir_per_context)?;
    let recovery_accounting_only = wave == "strict-wave-000189";
    let proposal_job_id = if recovery_accounting_only {
        format!(
            "recovery-source-{}",
            &sha256_file(&inputs.proposals_pointer)?[..16]
        )
    } else {
        format!(
            "jope-selfplay-cohort-propose-{}-{}-{}-s{}",
            &inputs.checkpoint_sha256[..12],
            &cohort_sha256[..12],
            &decoder_configuration_sha256[..12],
            COHORT_PROPOSAL_SEED
        )
    };
    let state = PersistedSupplyState {
        schema: SUPPLY_STATE_V2_SCHEMA.into(),
        wave: wave.to_owned(),
        checkpoint_sha256: inputs.checkpoint_sha256.clone(),
        cohort,
        cohort_sha256,
        catalog_sha256,
        decoder_configuration_sha256,
        seed: COHORT_PROPOSAL_SEED,
        proposal_reservoir_per_context,
        stage: PersistedSupplyStage::FreezeCohort,
        revision: 0,
        recovery_accounting_only,
        proposal_attempt: 0,
        proposal_job_id: Some(proposal_job_id),
        cohort_proposal_pointer: None,
        cohort_proposal_pointer_sha256: None,
        sanitized_proposal_pointer: None,
        sanitized_proposal_pointer_sha256: None,
        runtime_pinned_attempts: Vec::new(),
    };
    let mut state = state;
    // Snapshot the catalog pointer NOW, while the cohort is being frozen and the
    // live pointer still names the binding this state carries. The proposer-spec
    // renderer also snapshots lazily, but by the time it runs the expansion
    // driver may already have rotated `catalog/current.json` -- and a rotation
    // with no snapshot on disk is unrecoverable, because the pointer for a
    // superseded catalog cannot be reconstructed from anything that remains.
    // Taking it here is what makes a frozen cohort genuinely self-contained:
    // whatever the live catalog does afterwards, this wave can still resolve the
    // catalog it was planned against. Snapshots are content-addressed by catalog
    // digest and are never garbage collected, so this is write-once per catalog.
    snapshot_frozen_catalog_pointer(args, &state.catalog_sha256)?;
    if !state.recovery_accounting_only {
        resolve_pinned_proposer_spec(args, inputs, &mut state)?;
    }
    persist_supply_state(&args.root, &state)?;
    Ok(state)
}

/// Persist the live catalog pointer under its own digest so a cohort frozen
/// against it stays resolvable after `catalog/current.json` rotates. Idempotent:
/// an existing snapshot is verified rather than rewritten.
pub(super) fn snapshot_frozen_catalog_pointer(args: &Args, catalog_sha256: &str) -> Result<()> {
    let pointer_directory = args.root.join("proposals/cohort-inputs");
    fs::create_dir_all(&pointer_directory)?;
    let pointer_snapshot = pointer_directory.join(format!("{catalog_sha256}.pointer.json"));
    frozen_catalog_pointer(&args.catalog_pointer(), &pointer_snapshot, catalog_sha256)
        .with_context(|| format!("snapshot catalog pointer {catalog_sha256} at cohort freeze"))?;
    Ok(())
}

pub(super) fn ensure_cohort_lookahead(
    args: &Args,
    inputs: &CheckpointInputs,
    waves_root: &Path,
    desired_pending: usize,
) -> Result<()> {
    loop {
        let pending = waves_root.join("pending");
        // Count only cohorts that can actually still become waves. A QUARANTINED
        // cohort never will -- `wave_to_build` skips it forever via
        // `pending_wave_quarantined` -- so counting it here fills the lookahead
        // budget with corpses and stops any new cohort being pre-planned.
        //
        // Measured 2026-08-04: waves 224 and 226 have been quarantined since
        // 07-30 and still sit in `pending/`. With `desired_pending = 3` they plus
        // the one real pending wave satisfied the budget permanently, so the
        // effective lookahead was 1 and the 9-28 minute cohort-plan step ran on
        // the critical path of every wave instead of being hidden behind the
        // previous one. That is roughly two thirds of the ~21 min wave period.
        //
        // This is the same predicate `wave_to_build` already uses; the two
        // disagreeing is the whole bug. Written as a loop rather than a
        // `filter_map` because `pending_wave_quarantined` returns `Result` and
        // legitimately bails on cohort/state drift -- swallowing that with `.ok()`
        // would hide real corruption behind a silently short lookahead.
        let mut unresolved = 0_usize;
        for entry in fs::read_dir(&pending).into_iter().flatten().flatten() {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(wave) = stem.strip_suffix(".cohort") else {
                continue;
            };
            if !valid_wave_id(wave) || waves_root.join(wave).is_dir() {
                continue;
            }
            if pending_wave_quarantined(waves_root, wave, &path)? {
                continue;
            }
            unresolved += 1;
        }
        if unresolved >= desired_pending {
            return Ok(());
        }
        let wave = next_wave_id(waves_root)?;
        create_supply_state(args, inputs, waves_root, &wave)?;
    }
}

pub(super) fn build_next_wave(args: &Args, inputs: &CheckpointInputs) -> Result<Option<BuiltWave>> {
    let waves_root = args.root.join("waves");
    let wave = wave_to_build(&waves_root)?;
    let cohort = waves_root
        .join("pending")
        .join(format!("{wave}.cohort.json"));
    let mut state = if let Some(state) =
        load_supply_state(&args.root, &wave, &inputs.checkpoint_sha256)?
    {
        let expected_decoder_configuration_sha256 =
            proposal_decoder_configuration_sha256(args, state.proposal_reservoir_per_context)?;
        if state.runtime_pinned_attempts.is_empty()
            && state.decoder_configuration_sha256 != expected_decoder_configuration_sha256
        {
            bail!(
                "{wave}: persisted proposer decoder configuration differs from the active immutable runtime"
            );
        }
        state
    } else {
        create_supply_state(args, inputs, &waves_root, &wave)?
    };
    if !state.cohort.is_file() || sha256_file(&state.cohort)? != state.cohort_sha256 {
        bail!("{wave}: persisted frozen cohort content drift");
    }

    if state.stage == PersistedSupplyStage::FreezeCohort {
        if state.recovery_accounting_only {
            state.advance(PersistedSupplyStage::SubmitCohortProposer)?;
            persist_supply_state(&args.root, &state)?;
        } else {
            let spec = resolve_pinned_proposer_spec(args, inputs, &mut state)?;
            persist_supply_state(&args.root, &state)?;
            let job_id = state
                .proposal_job_id
                .as_deref()
                .context("cohort proposer job id is absent")?;
            match proposer_job_state(args, job_id)? {
                None => {
                    if !scq_check_submit(args, &spec)? {
                        return Ok(None);
                    }
                }
                Some(existing) if existing == "completed" => {}
                Some(existing)
                    if matches!(
                        existing.as_str(),
                        "queued" | "starting" | "dispatching" | "running" | "held"
                    ) => {}
                Some(existing) => bail!("{job_id}: proposer is terminal in state {existing}"),
            }
            let payload = proposer_scientific_payload(
                &state,
                state
                    .runtime_pinned_attempts
                    .last()
                    .context("initial proposer lacks a pinned intent")?,
            )?;
            let mut projection = proposer_recovery_projection(&args.root, &state, &payload)?;
            if projection.submitted_at_ms == 0 {
                projection.record_initial_submission(job_id, now_ms())?;
                save_recovery_projection(&args.root, &projection)?;
            }
            state.advance(PersistedSupplyStage::SubmitCohortProposer)?;
            persist_supply_state(&args.root, &state)?;
            ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
            return Ok(None);
        }
    }

    if state.stage == PersistedSupplyStage::SubmitCohortProposer {
        let (pointer, pointer_sha256) = if state.recovery_accounting_only {
            if wave != "strict-wave-000189" {
                bail!("whole-catalog proposal sources are recovery-only");
            }
            (
                inputs.proposals_pointer.clone(),
                sha256_file(&inputs.proposals_pointer)?,
            )
        } else {
            let payload = proposer_scientific_payload(
                &state,
                state
                    .runtime_pinned_attempts
                    .last()
                    .context("submitted proposer lacks a pinned intent")?,
            )?;
            let mut projection = proposer_recovery_projection(&args.root, &state, &payload)?;
            if state.proposal_job_id.as_deref() != Some(projection.current_job_identity.as_str()) {
                reconcile_proposer_projection(&args.root, &mut state, &projection)?;
                state.revision = state.revision.saturating_add(1);
                persist_supply_state(&args.root, &state)?;
                ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                return Ok(None);
            }
            if projection.retry_pending_submission {
                if now_ms() < projection.retry_not_before_ms {
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
                let spec = resolve_pinned_proposer_spec(args, inputs, &mut state)?;
                let job_id = state
                    .proposal_job_id
                    .clone()
                    .context("pending proposer retry lacks a job identity")?;
                match proposer_job_state(args, &job_id)?.as_deref() {
                    None => {
                        if !scq_check_submit(args, &spec)? {
                            return Ok(None);
                        }
                    }
                    Some(
                        "completed" | "queued" | "starting" | "dispatching" | "running" | "held",
                    ) => {}
                    Some(other) => {
                        bail!("{job_id}: pending proposer retry is terminal in state {other}")
                    }
                }
                projection.mark_submitted(&job_id, now_ms())?;
                save_recovery_projection(&args.root, &projection)?;
                ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                return Ok(None);
            }
            let job_id = state
                .proposal_job_id
                .clone()
                .context("cohort proposer job id is absent")?;
            match proposer_job_state(args, &job_id)?.as_deref() {
                Some("completed") => {
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    fetch_cohort_proposals(args, &state)?
                }
                Some("queued" | "starting" | "dispatching" | "running" | "held") => {
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
                Some(terminal_state) => {
                    match state.retry_proposer(&args.root, terminal_state, now_ms())? {
                        ProposerRetryDisposition::Retried | ProposerRetryDisposition::Backoff => {}
                        ProposerRetryDisposition::Drained
                        | ProposerRetryDisposition::ScientificCircuitOpen => {
                            persist_supply_state(&args.root, &state)?;
                            return Ok(None);
                        }
                    }
                    persist_supply_state(&args.root, &state)?;
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
                None => {
                    match state.retry_proposer(&args.root, "vanished", now_ms())? {
                        ProposerRetryDisposition::Retried | ProposerRetryDisposition::Backoff => {}
                        ProposerRetryDisposition::Drained
                        | ProposerRetryDisposition::ScientificCircuitOpen => {
                            persist_supply_state(&args.root, &state)?;
                            return Ok(None);
                        }
                    }
                    persist_supply_state(&args.root, &state)?;
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
            }
        };
        state.cohort_proposal_pointer = Some(pointer);
        state.cohort_proposal_pointer_sha256 = Some(pointer_sha256);
        state.advance(PersistedSupplyStage::FetchValidateProposals)?;
        persist_supply_state(&args.root, &state)?;
    }

    if state.stage == PersistedSupplyStage::FetchValidateProposals {
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        let source_pointer = state
            .cohort_proposal_pointer
            .clone()
            .context("validated state lacks proposal pointer")?;
        if sha256_file(&source_pointer)?
            != state
                .cohort_proposal_pointer_sha256
                .clone()
                .context("validated state lacks proposal pointer hash")?
        {
            bail!("{wave}: immutable cohort proposal pointer drift");
        }
        let sanitized = run_builder(
            &args.wave_builder,
            vec![
                "sanitize-proposals".into(),
                "--cohort".into(),
                cohort.clone().into_os_string(),
                "--proposals-pointer".into(),
                source_pointer.into_os_string(),
                "--output-root".into(),
                args.root.join("proposals/sanitized").into_os_string(),
            ],
        )?;
        require_output_identity(&sanitized, &wave, &inputs.checkpoint_sha256)?;
        state.sanitized_proposal_pointer_sha256 = Some(require_sha(&sanitized, "pointer_sha256")?);
        state.sanitized_proposal_pointer = Some(PathBuf::from(
            sanitized
                .get("pointer")
                .and_then(Value::as_str)
                .context("sanitizer output lacks proposal pointer")?,
        ));
        state.advance(PersistedSupplyStage::CacheFilterResample)?;
        persist_supply_state(&args.root, &state)?;
    }

    if state.stage == PersistedSupplyStage::CacheFilterResample {
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        let proposal_pointer = state
            .sanitized_proposal_pointer
            .clone()
            .context("cache-filtered state lacks proposal pointer")?;
        let built = run_builder(
            &args.wave_builder,
            vec![
                "build".into(),
                "--cohort".into(),
                cohort.into_os_string(),
                "--catalog-pointer".into(),
                args.catalog_pointer().into_os_string(),
                "--proposals-pointer".into(),
                proposal_pointer.into_os_string(),
                "--corpus".into(),
                args.corpus().into_os_string(),
                "--robust-seed-bank".into(),
                args.robust_seed_bank().into_os_string(),
                "--sealed-cohort".into(),
                args.sealed_cohort().into_os_string(),
                "--pair-registry".into(),
                args.pair_registry().into_os_string(),
                "--output-root".into(),
                waves_root.into_os_string(),
                "--max-candidates-per-manifest".into(),
                args.max_candidates_per_manifest.to_string().into(),
            ],
        )?;
        if built.get("wave").and_then(Value::as_str) != Some(wave.as_str()) {
            bail!("{wave}: Rust wave builder returned a mismatched wave");
        }
        let manifests = built
            .get("manifests")
            .and_then(Value::as_u64)
            .context("builder output lacks manifest count")? as usize;
        if manifests == 0 || manifests > manifests_per_wave(args.max_candidates_per_manifest) {
            bail!("{wave}: builder manifest count is outside the bounded micro-wave size");
        }
        state.advance(PersistedSupplyStage::BuildWave)?;
        persist_supply_state(&args.root, &state)?;
    }

    if state.stage == PersistedSupplyStage::BuildWave {
        return Ok(Some(built_wave_from_disk(args, &state)?));
    }
    Ok(None)
}

pub(super) fn require_output_identity(output: &Value, wave: &str, checkpoint: &str) -> Result<()> {
    if output.get("wave").and_then(Value::as_str) != Some(wave)
        || output.get("checkpoint_sha256").and_then(Value::as_str) != Some(checkpoint)
    {
        bail!("{wave}: Rust wave stage returned stale or mismatched provenance");
    }
    Ok(())
}

pub(super) fn require_sha(output: &Value, field: &str) -> Result<String> {
    let digest = output
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("Rust wave stage lacks {field}"))?;
    validate_sha256(digest)?;
    Ok(digest.to_owned())
}

pub(super) fn ensure_native_wave_builder(binary: &Path) -> Result<()> {
    let mut magic = [0_u8; 4];
    File::open(binary)
        .with_context(|| format!("open Rust wave builder {}", binary.display()))?
        .read_exact(&mut magic)?;
    if magic != [0x7f, b'E', b'L', b'F'] {
        bail!("wave builder must be a native ELF binary; scripts and Python are forbidden");
    }
    Ok(())
}

pub(super) fn run_builder(binary: &Path, arguments: Vec<OsString>) -> Result<Value> {
    let output = Command::new(binary)
        .args(arguments)
        .env_remove("PYTHONPATH")
        .output()
        .with_context(|| format!("execute Rust wave builder {}", binary.display()))?;
    if !output.status.success() {
        bail!(
            "Rust wave builder failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("parse Rust wave builder JSON output")
}

pub(super) fn append_event(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)?;
    // Serialize record and newline before appending; write_all may retry short writes.
    // An I/O failure can still leave an incomplete record.
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    file.write_all(&line)?;
    file.sync_data()?;
    Ok(())
}
