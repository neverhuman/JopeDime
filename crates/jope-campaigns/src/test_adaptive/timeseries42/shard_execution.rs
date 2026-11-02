fn copy_remote_output(node: &str, remote: &str, local: &Path) -> Result<()> {
    if local.exists() {
        fs::remove_dir_all(local)?;
    }
    fs::create_dir_all(local)?;
    if is_controller_node(node)? {
        scq_batch::command_output(
            {
                let mut command = Command::new("cp");
                command.arg("-a").arg(format!("{remote}/.")).arg(local);
                command
            },
            "copy local TimeSeries42 SCQ output",
        )?;
    } else {
        scq_batch::command_output(
            {
                let mut command = Command::new("scp");
                command
                    .arg("-r")
                    .arg("--")
                    .arg(format!("{node}:{remote}/."))
                    .arg(local);
                command
            },
            "pull TimeSeries42 SCQ output",
        )?;
    }
    Ok(())
}

fn read_receipts(path: &Path) -> Result<Vec<EvalReceipt>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse corpus_eval receipt"))
        .collect()
}

// Keep staged shard identities and output paths explicit.
#[allow(clippy::too_many_arguments)]
fn prepare_shard(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    shard: usize,
    node_index: usize,
    attempt: usize,
    members: &[Member],
) -> Result<PreparedShard> {
    let node_name = NODES[node_index];
    let node = staged_node(staging, node_name)?;
    let staged = staged_dataset(staging, node_name, &spec.dataset_hash)?;
    let authority = label_authority(generation, seed_sha256, release_sha256, campaign_sha256);
    let protocol_sha256 = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    let execution_binding =
        is_forecast3_gen20(run_id).then_some((run_id, campaign_sha256, release_sha256));
    let remote_manifest_value = corpus_manifest(
        spec,
        members,
        (&staged.train_path, &staged.test_path, &staged.meta_path),
        authority.clone(),
        protocol_sha256.as_deref(),
        execution_binding,
    )?;
    let local_manifest_value = corpus_manifest(
        spec,
        members,
        (&spec.train_path, &spec.test_path, &spec.meta_path),
        authority,
        protocol_sha256.as_deref(),
        execution_binding,
    )?;
    let local_manifest_bytes = canonical_json(&local_manifest_value)?;
    let local_manifest_sha256 = sha256_bytes(&local_manifest_bytes);
    let dataset_dir = dataset_root(root, &spec.dataset_hash);
    let local_manifest_path = generation_root(&dataset_dir, generation)
        .join("corpus-manifests")
        .join(format!(
            "shard-{shard:02}-attempt-{attempt}-{short}.v1.json",
            short = &local_manifest_sha256[..16]
        ));
    immutable_or_equal(&local_manifest_path, &local_manifest_bytes)?;
    validate_with_corpus_eval(&local_manifest_path)?;

    let remote_manifest_bytes = canonical_json(&remote_manifest_value)?;
    let remote_manifest_sha256 = sha256_bytes(&remote_manifest_bytes);
    let remote_manifest_local = root
        .join("jobs")
        .join("manifests")
        .join(format!("{remote_manifest_sha256}.json"));
    immutable_or_equal(&remote_manifest_local, &remote_manifest_bytes)?;
    let job_id = expected_job_id(
        run_id,
        &spec.dataset_hash,
        generation,
        shard,
        node_name,
        attempt,
        &remote_manifest_sha256,
    );
    let remote_manifest = format!(
        "{}/jobs/manifests/{remote_manifest_sha256}.json",
        node.bundle_root
    );
    let remote_output = format!("{}/jobs/{job_id}/output", node.bundle_root);
    let project = if is_forecast3_gen20(run_id) {
        FORECAST3_PROJECT
    } else {
        LEGACY_PROJECT
    };
    let spec_text = scq_spec(node, &job_id, &remote_manifest, &remote_output, project)?;
    let spec_path = root
        .join("jobs")
        .join("specs")
        .join(format!("{job_id}.toml"));
    immutable_or_equal(&spec_path, spec_text.as_bytes())?;

    let receipt_path = root
        .join("jobs")
        .join("receipts")
        .join(format!("{job_id}.v1.json"));
    if !receipt_path.is_file() {
        if is_controller_node(node_name)? {
            copy_local_checked(&remote_manifest_local, Path::new(&remote_manifest), false)?;
            fs::create_dir_all(&remote_output)?;
        } else {
            copy_remote_checked(node_name, &remote_manifest_local, &remote_manifest, false)?;
            scq_batch::ssh_output(
                node_name,
                &["mkdir", "-p", &remote_output],
                "create TimeSeries42 remote output",
            )?;
        }
    }

    Ok(PreparedShard {
        generation,
        shard,
        node: node_name.to_string(),
        attempt,
        job_id,
        remote_manifest_sha256,
        local_manifest_path,
        spec_path,
        spec_text,
        remote_output,
        receipt_path,
    })
}

fn finalize_prepared_shard(
    root: &Path,
    spec: &DatasetSpec,
    prepared: &PreparedShard,
) -> Result<ShardRecord> {
    if !prepared.receipt_path.is_file() {
        let show = wait_for_job(&prepared.job_id)?;
        let show_bytes = canonical_json(&show)?;
        let job = scq_batch::scq_job(&show, &prepared.job_id)?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .context("SCQ job has no state")?
            .to_string();
        let exit_code = job
            .get("exit_code")
            .and_then(Value::as_i64)
            .context("SCQ job has no exit_code")?;
        if state != "completed" || exit_code != 0 {
            bail!(
                "TimeSeries42 shard {} failed: state={state}, exit={exit_code}",
                prepared.job_id
            );
        }
        let temp = std::env::temp_dir().join(format!(
            "jope-ts42-{}-{}",
            std::process::id(),
            prepared.job_id
        ));
        copy_remote_output(&prepared.node, &prepared.remote_output, &temp)?;
        let raw = root.join("jobs").join("raw").join(&prepared.job_id);
        if raw.exists() {
            fs::remove_dir_all(&raw)?;
        }
        fs::create_dir_all(raw.parent().context("raw output has no parent")?)?;
        fs::rename(&temp, &raw)?;
        let worker_bytes = fs::read(raw.join("batch-worker-receipt.v1.json"))?;
        let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
        let receipt_file = raw.join(format!("{}.receipts.jsonl", spec.dataset_hash));
        let receipt_bytes = fs::read(&receipt_file)?;
        if worker.schema != WORKER_SCHEMA
            || !worker.test_adaptive
            || worker.benchmark_valid
            || worker.dataset_hash != spec.dataset_hash
            || worker.manifest_sha256 != prepared.remote_manifest_sha256
            || worker.receipt_sha256 != sha256_bytes(&receipt_bytes)
        {
            bail!(
                "TimeSeries42 worker receipt mismatch for {}",
                prepared.job_id
            );
        }
        let job_receipt = JobReceipt {
            schema: JOB_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            job_id: prepared.job_id.clone(),
            node: prepared.node.clone(),
            attempt: prepared.attempt,
            state,
            exit_code,
            corpus_manifest_sha256: prepared.remote_manifest_sha256.clone(),
            scq_spec_sha256: sha256_bytes(prepared.spec_text.as_bytes()),
            scq_show_sha256: sha256_bytes(&show_bytes),
            worker_receipt_sha256: sha256_bytes(&worker_bytes),
            receipt_sha256: sha256_bytes(&receipt_bytes),
            completed_at_unix_ms: unix_millis(),
        };
        write_json(&prepared.receipt_path, &job_receipt)?;
    }
    let raw_receipt = root
        .join("jobs")
        .join("raw")
        .join(&prepared.job_id)
        .join(format!("{}.receipts.jsonl", spec.dataset_hash));
    let receipt_bytes = fs::read(&raw_receipt)?;
    let copied_receipt =
        generation_root(&dataset_root(root, &spec.dataset_hash), prepared.generation)
            .join("receipts")
            .join(format!(
                "shard-{shard:02}-attempt-{attempt}-{short}.receipts.jsonl",
                shard = prepared.shard,
                attempt = prepared.attempt,
                short = &prepared.remote_manifest_sha256[..16]
            ));
    immutable_or_equal(&copied_receipt, &receipt_bytes)?;
    let receipts = read_receipts(&raw_receipt)?;
    Ok(ShardRecord {
        generation: prepared.generation,
        shard: prepared.shard,
        node: prepared.node.clone(),
        attempt: prepared.attempt,
        job_id: prepared.job_id.clone(),
        corpus_manifest_file: prepared.local_manifest_path.to_string_lossy().into_owned(),
        corpus_manifest_sha256: sha256_file(&prepared.local_manifest_path)?,
        scq_spec_file: prepared.spec_path.to_string_lossy().into_owned(),
        scq_spec_sha256: sha256_bytes(prepared.spec_text.as_bytes()),
        receipt_file: copied_receipt.to_string_lossy().into_owned(),
        receipt_sha256: sha256_bytes(&receipt_bytes),
        valid_candidates: receipts.iter().filter(|receipt| receipt.valid).count(),
        total_candidates: receipts.len(),
    })
}

// Keep transaction identity, staging state and wave budget explicit.
#[allow(clippy::too_many_arguments)]
fn execute_transactional_wave(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    chunks: &[Vec<Member>],
) -> Result<Vec<ShardRecord>> {
    loop {
        let submitted_at = unix_millis();
        let mut prepared = Vec::with_capacity(SHARDS);
        for (shard, members) in chunks.iter().enumerate() {
            prepared.push(prepare_shard(
                root,
                run_id,
                campaign_sha256,
                spec,
                seed_sha256,
                release_sha256,
                staging,
                generation,
                shard,
                shard % NODES.len(),
                0,
                members,
            )?);
        }
        let mut capacity_wait = None;
        for shard in &prepared {
            if shard.receipt_path.is_file() {
                continue;
            }
            match scq_check(&shard.spec_path, &shard.node) {
                Ok(()) => {}
                Err(error) if is_scq_capacity_error_for_node(&error, &shard.node) => {
                    capacity_wait = Some(format!("{error:#}"));
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(error) = capacity_wait {
            eprintln!(
                "TimeSeries42 generation {generation} dataset {} waiting for SCQ capacity before transactional wave: {error}",
                spec.dataset_hash
            );
            thread::sleep(std::time::Duration::from_secs(SCQ_CAPACITY_RETRY_SECONDS));
            continue;
        }
        let missing_specs = prepared
            .iter()
            .filter(|shard| !shard.receipt_path.is_file() && !scq_job_exists(&shard.job_id))
            .map(|shard| shard.spec_path.clone())
            .collect::<Vec<_>>();
        match submit_batch(&missing_specs) {
            Ok(()) => {}
            Err(error) if is_scq_capacity_error(&error) => {
                eprintln!(
                    "TimeSeries42 generation {generation} dataset {} waiting for SCQ capacity before submit-batch: {error:#}",
                    spec.dataset_hash
                );
                thread::sleep(std::time::Duration::from_secs(SCQ_CAPACITY_RETRY_SECONDS));
                continue;
            }
            Err(error) => return Err(error),
        }

        let mut records = Vec::with_capacity(SHARDS);
        for shard in &prepared {
            records.push(finalize_prepared_shard(root, spec, shard)?);
        }
        records.sort_by_key(|record| record.shard);
        let wave = WaveReceipt {
            schema: WAVE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            campaign_sha256: campaign_sha256.to_string(),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            nodes: prepared.iter().map(|shard| shard.node.clone()).collect(),
            job_ids: prepared.iter().map(|shard| shard.job_id.clone()).collect(),
            corpus_manifest_sha256: prepared
                .iter()
                .map(|shard| shard.remote_manifest_sha256.clone())
                .collect(),
            scq_spec_sha256: prepared
                .iter()
                .map(|shard| sha256_bytes(shard.spec_text.as_bytes()))
                .collect(),
            terminal_rows: records.iter().map(|record| record.total_candidates).sum(),
            submitted_at_unix_ms: submitted_at,
            completed_at_unix_ms: unix_millis(),
        };
        write_json(
            &generation_root(&dataset_root(root, &spec.dataset_hash), generation)
                .join("wave-receipt.v1.json"),
            &wave,
        )?;
        return Ok(records);
    }
}
