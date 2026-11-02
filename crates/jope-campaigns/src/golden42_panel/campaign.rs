use super::*;

pub(crate) fn run(
    registry: &Path,
    run_id: &str,
    dataset: &str,
    node: &str,
    workers: usize,
    population: usize,
    resume: bool,
) -> Result<()> {
    if dataset != DATASET || node != "xbabe2" || workers != WORKERS || population != WORKERS {
        bail!("Golden42 V3 requires dataset {DATASET}, node xbabe2, and 80 workers/population");
    }
    let hostname = command_text("hostname", &["-s"])?;
    if hostname.trim() != "xbabe2" {
        bail!("Golden42 V3 must run on xbabe2, found {}", hostname.trim());
    }
    validate_dataset_hashes(registry)?;
    let workspace = workspace_root();
    let root = workspace.join("runs").join(run_id);
    let admission_path = workspace.join(ADMISSION_RELATIVE);
    let admission_bytes = fs::read(&admission_path)
        .with_context(|| format!("read admission receipt {}", admission_path.display()))?;
    let admission: AdmissionReceipt = serde_json::from_slice(&admission_bytes)?;
    if !admission.admitted || admission.benchmark_valid || !admission.test_adaptive {
        bail!("fitted C++ admission receipt is not production-admitted for this campaign");
    }
    if admission.source_sha256 != source_tree_hash()? {
        bail!("fitted C++ admission receipt does not bind the current source tree");
    }
    let release = std::env::current_exe()?;
    let release_sha256 = sha256_file(&release)?;
    if admission.release_sha256 != release_sha256 {
        bail!("admission receipt does not bind the running release binary");
    }
    if root.exists() && !resume {
        bail!("run root {} already exists; use --resume", root.display());
    }
    fs::create_dir_all(&root)?;
    let _node_lease = Lease::acquire(Path::new(node_lease_str()))?;
    let _run_lease = Lease::acquire(&root.join("run.lock"))?;
    wait_for_resources(&root, run_id, 0, 0, 0, 0, 0)?;
    let cpu_ids = choose_cpus(WORKERS)?;
    let dataset_root = registry.join(DATASET);
    let test = read_regression_csv(&dataset_root.join("test.csv"))?;
    let test_features = root.join("test-features.v1.json");
    atomic_immutable_write(&test_features, &json_bytes(&test.x)?)?;
    let config = V3EvolutionConfig::default();
    let supplied = v3_seed_sources();
    let initial = initial_population(&config, &supplied);
    let manifest_path = root.join("manifest.v3.json");
    let admission_sha256 = sha256_bytes(&admission_bytes);
    let manifest = if manifest_path.exists() {
        let existing: CampaignManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        validate_manifest_contract(
            &existing,
            run_id,
            node,
            workers,
            population,
            &release_sha256,
            &admission_sha256,
            &initial,
        )?;
        existing
    } else {
        let created = CampaignManifest {
            schema: "jope-dime.golden42-v3-max-manifest.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: DATASET.to_string(),
            train_sha256: TRAIN_SHA256.to_string(),
            test_sha256: TEST_SHA256.to_string(),
            node: node.to_string(),
            workers,
            population,
            islands: 4,
            island_size: 20,
            strict_threshold: WINNER_RMSE,
            test_adaptive: true,
            benchmark_valid: false,
            prediction_backend: "fitted-cpp17-only".to_string(),
            release_sha256: release_sha256.clone(),
            admission_sha256,
            initial_population: initial.clone(),
            created_at_unix_ms: unix_ms(),
        };
        atomic_immutable_write(&manifest_path, &json_bytes(&created)?)?;
        created
    };
    if root.join("final-result.v1.json").exists() {
        validate(run_id)?;
        return Ok(());
    }
    let (mut generation, mut population, mut archive, mut previous_checkpoint) =
        resume_state(&root, &manifest)?;
    let mut retry = 0usize;
    loop {
        let mut workers = spawn_workers(
            &release,
            registry,
            &root,
            run_id,
            &dataset_root.join("train.csv"),
            &test_features,
            &cpu_ids,
        )?;
        write_worker_registry(&root, &workers)?;
        let result = campaign_loop(
            &root,
            run_id,
            &release_sha256,
            &test.y,
            &config,
            &mut generation,
            &mut population,
            &mut archive,
            &mut previous_checkpoint,
            &mut workers,
            &cpu_ids,
            &release,
            &dataset_root.join("train.csv"),
            &test_features,
        );
        terminate_workers(&mut workers);
        match result {
            Ok(()) => return Ok(()),
            Err(error)
                if error
                    .to_string()
                    .starts_with("retryable V3 worker failure:")
                    && retry < 3 =>
            {
                retry += 1;
                let record = serde_json::json!({
                    "schema": "jope-dime.golden42-v3-worker-retry.v1",
                    "run_id": run_id,
                    "generation": generation,
                    "retry": retry,
                    "error": error.to_string(),
                    "unix_ms": unix_ms(),
                });
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(root.join("worker-retries.v1.jsonl"))?;
                // Serialize record and newline before appending; write_all may retry short writes.
                // An I/O failure can still leave an incomplete record.
                let mut line = serde_json::to_vec(&record)?;
                line.push(b'\n');
                file.write_all(&line)?;
                file.flush()?;
            }
            Err(error) => return Err(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn campaign_loop(
    root: &Path,
    run_id: &str,
    release_sha256: &str,
    test_targets: &[f64],
    config: &V3EvolutionConfig,
    generation: &mut usize,
    population: &mut Vec<V3Candidate>,
    archive: &mut V3Archive,
    previous_checkpoint: &mut Option<String>,
    workers: &mut [WorkerProcess],
    cpu_ids: &[usize],
    release: &Path,
    train_path: &Path,
    test_features_path: &Path,
) -> Result<()> {
    let (_sender, receiver) = worker_channel(workers)?;
    let mut best_by_island = [f64::INFINITY; 4];
    let mut stagnation = [0usize; 4];
    loop {
        wait_for_resources(root, run_id, *generation, 0, population.len(), 0, 0)?;
        let generation_started = Instant::now();
        let generation_root = root
            .join("receipts")
            .join(format!("generation-{generation:06}"));
        fs::create_dir_all(&generation_root)?;
        let mut receipts = BTreeMap::<usize, CandidateReceipt>::new();
        let mut cache_hits = 0usize;
        for (slot, member) in population.iter().enumerate() {
            let path = generation_root.join(format!("slot-{slot:03}.v1.json"));
            if path.exists() {
                let receipt: CandidateReceipt = serde_json::from_slice(&fs::read(&path)?)?;
                validate_candidate_receipt(
                    &receipt,
                    root,
                    run_id,
                    *generation,
                    slot,
                    test_targets,
                    member,
                    release_sha256,
                )?;
                receipts.insert(slot, receipt);
                cache_hits += 1;
            }
        }
        if let Some(winner) = receipts
            .values()
            .filter(|receipt| receipt.rmse < WINNER_RMSE)
            .min_by(|left, right| {
                left.finished_at_unix_ms
                    .cmp(&right.finished_at_unix_ms)
                    .then_with(|| left.slot.cmp(&right.slot))
            })
            .cloned()
        {
            return finalize_winner(
                root,
                run_id,
                *generation,
                winner,
                workers,
                cpu_ids,
                release,
                train_path,
                test_features_path,
                test_targets,
            );
        }
        let missing = population
            .iter()
            .enumerate()
            .filter(|(slot, _)| !receipts.contains_key(slot))
            .map(|(slot, candidate)| (slot, candidate.clone()))
            .collect::<Vec<_>>();
        let worker_count = workers.len();
        for (assignment, (slot, candidate)) in missing.iter().enumerate() {
            let worker = &mut workers[assignment % worker_count];
            let context_sha256 = context_hash(release_sha256, candidate);
            let request = WorkerRequest {
                schema: "jope-dime.golden42-v3-worker-request.v1".to_string(),
                generation: *generation,
                slot: *slot,
                candidate: candidate.clone(),
                context_sha256,
            };
            serde_json::to_writer(&mut worker.stdin, &request)?;
            worker.stdin.write_all(b"\n")?;
            worker.stdin.flush()?;
        }
        let mut remaining = missing.len();
        let mut winner: Option<CandidateReceipt> = None;
        let mut last_telemetry = Instant::now() - Duration::from_secs(2);
        while remaining > 0 {
            match receiver.recv_timeout(Duration::from_millis(250)) {
                Ok((worker_id, response)) => {
                    let response = response?;
                    if response.worker_id != worker_id
                        || response.generation != *generation
                        || response.slot >= population.len()
                    {
                        bail!("worker returned a response for the wrong assignment");
                    }
                    let candidate = &population[response.slot];
                    if response.semantic_hash != candidate.semantic_hash
                        || response.context_sha256 != context_hash(release_sha256, candidate)
                        || response.predictions.len() != test_targets.len()
                        || response.prediction_sha256 != prediction_hash(&response.predictions)
                    {
                        bail!("worker response identity or prediction tape is invalid");
                    }
                    validate_codec_artifact(root, &response.codec_file, &response.codec_sha256)?;
                    let rmse = raw_rmse(test_targets, &response.predictions)?;
                    let receipt = CandidateReceipt {
                        schema: "jope-dime.golden42-v3-candidate-receipt.v1".to_string(),
                        run_id: run_id.to_string(),
                        dataset_hash: DATASET.to_string(),
                        generation: *generation,
                        slot: response.slot,
                        worker_id,
                        cpu: workers[worker_id].cpu,
                        candidate: candidate.clone(),
                        context_sha256: response.context_sha256,
                        rmse,
                        predictions: response.predictions,
                        prediction_sha256: response.prediction_sha256,
                        codec_sha256: response.codec_sha256,
                        codec_file: response.codec_file,
                        fit_ns: response.fit_ns,
                        codec_ns: response.codec_ns,
                        cpp_load_ns: response.cpp_load_ns,
                        cpp_predict_ns: response.cpp_predict_ns,
                        started_at_unix_ms: response.started_at_unix_ms,
                        finished_at_unix_ms: response.finished_at_unix_ms,
                    };
                    let receipt_path =
                        generation_root.join(format!("slot-{:03}.v1.json", receipt.slot));
                    atomic_immutable_write(&receipt_path, &json_bytes(&receipt)?)?;
                    remaining -= 1;
                    if rmse < WINNER_RMSE {
                        winner = Some(receipt);
                        break;
                    }
                    receipts.insert(receipt.slot, receipt);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("retryable V3 worker failure: all response channels disconnected")
                }
            }
            for worker in workers.iter_mut() {
                if let Some(status) = worker.child.try_wait()? {
                    bail!(
                        "retryable V3 worker failure: worker {} exited with {status}",
                        worker.id
                    );
                }
            }
            if last_telemetry.elapsed() >= Duration::from_secs(1) {
                append_telemetry(
                    root,
                    run_id,
                    *generation,
                    population.len() - remaining,
                    remaining,
                    workers,
                    cache_hits,
                    0,
                    population
                        .iter()
                        .map(|candidate| &candidate.semantic_hash)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    &receipts,
                    generation_started,
                )?;
                last_telemetry = Instant::now();
            }
        }
        if let Some(winner) = winner {
            return finalize_winner(
                root,
                run_id,
                *generation,
                winner,
                workers,
                cpu_ids,
                release,
                train_path,
                test_features_path,
                test_targets,
            );
        }
        if receipts.len() != population.len() {
            bail!("generation barrier closed without all candidate receipts");
        }
        let mut scored = receipts
            .into_values()
            .map(|receipt| V3ScoredCandidate {
                novelty: novelty(&receipt.candidate, archive),
                candidate: receipt.candidate,
                rmse: receipt.rmse,
            })
            .collect::<Vec<_>>();
        scored.sort_by(scored_order);
        for island in 0..4 {
            let best = scored
                .iter()
                .filter(|item| item.candidate.island as usize == island)
                .map(|item| item.rmse)
                .min_by(f64::total_cmp)
                .context("empty scored island")?;
            if best < best_by_island[island] {
                best_by_island[island] = best;
                stagnation[island] = 0;
            } else {
                stagnation[island] += 1;
            }
        }
        archive.update(scored.clone());
        let next = next_generation(config, *generation + 1, &scored, &stagnation)?;
        let checkpoint = Checkpoint {
            schema: "jope-dime.golden42-v3-checkpoint.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: DATASET.to_string(),
            generation: *generation,
            previous_sha256: previous_checkpoint.clone(),
            ranked_semantic_hashes: scored
                .iter()
                .map(|item| item.candidate.semantic_hash.clone())
                .collect(),
            scored,
            archive: archive.clone(),
            next_population: next.clone(),
            created_at_unix_ms: unix_ms(),
        };
        let bytes = json_bytes(&checkpoint)?;
        atomic_immutable_write(
            &root
                .join("checkpoints")
                .join(format!("generation-{:06}.v1.json", *generation)),
            &bytes,
        )?;
        *previous_checkpoint = Some(sha256_bytes(&bytes));
        *population = next;
        *generation += 1;
    }
}
