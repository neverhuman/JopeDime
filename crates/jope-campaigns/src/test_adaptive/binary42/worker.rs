use super::*;

pub(crate) fn batch_worker_impl(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: ShardManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest_bytes != canonical_json(&manifest)? {
        bail!("Binary42 shard manifest is not canonical JSON");
    }
    validate_shard_manifest(&manifest)?;
    if sha256_file(Path::new(&manifest.train_path))? != manifest.train_sha256
        || sha256_file(Path::new(&manifest.test_path))? != manifest.test_sha256
    {
        bail!("Binary42 worker data hash mismatch");
    }
    let train = read_binary_csv(Path::new(&manifest.train_path))?;
    let test = read_binary_csv(Path::new(&manifest.test_path))?;
    if train.x.len() != manifest.train_rows
        || test.x.len() != manifest.test_rows
        || train.features != manifest.features
        || test.features != manifest.features
    {
        bail!("Binary42 worker data shape mismatch");
    }
    let hostname = manifest.node.clone();
    let fresh = manifest
        .slots
        .iter()
        .filter(|slot| slot.fresh)
        .cloned()
        .collect::<Vec<_>>();
    let allowed = scq_batch::job_allowed_cpus()?;
    if allowed.len() < manifest.cpus_reserved || fresh.len() > manifest.cpus_reserved {
        bail!(
            "Binary42 SCQ allocation exposes {} CPUs for {} reserved and {} fresh candidates",
            allowed.len(),
            manifest.cpus_reserved,
            fresh.len()
        );
    }
    fs::create_dir_all(output_dir)?;
    let cpp17_toolchain = manifest
        .cohort_mode
        .as_deref()
        .is_some_and(|mode| mode == "all")
        .then(cpp17_toolchain)
        .transpose()?;
    let started_at_unix_ms = unix_millis();
    let outputs = Arc::new(Mutex::new(Vec::<WorkerCandidate>::new()));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for (index, slot) in fresh.iter().enumerate() {
            let outputs = Arc::clone(&outputs);
            let thread_failures = Arc::clone(&failures);
            let spawn_failures = Arc::clone(&failures);
            let train = &train;
            let test = &test;
            let manifest = &manifest;
            let cpp17_toolchain = cpp17_toolchain.as_ref();
            let cpu = allowed[index];
            let spawn = thread::Builder::new()
                .name(format!(
                    "binary42-g{:02}-s{:02}",
                    slot.generation, slot.slot
                ))
                .stack_size(CANDIDATE_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    let result = (|| -> Result<WorkerCandidate> {
                        scq_batch::pin_current_thread(cpu)?;
                        let runtime_root = output_dir.join(format!(
                            ".cpp17-runtime-g{:02}-s{:02}",
                            slot.generation, slot.slot
                        ));
                        let (receipt, prediction_bytes) = evaluate_worker_candidate(
                            manifest,
                            slot,
                            train,
                            test,
                            cpu,
                            cpp17_toolchain,
                            &runtime_root,
                        );
                        let _ = fs::remove_dir_all(&runtime_root);
                        validate_candidate_backend(
                            &receipt,
                            manifest.cohort_mode.as_deref() == Some("all"),
                        )?;
                        let receipt_bytes = canonical_json(&receipt)?;
                        let receipt_file = candidate_file(slot.generation, slot.slot);
                        let predictions_file = prediction_file(slot.generation, slot.slot);
                        immutable_write(&output_dir.join(&receipt_file), &receipt_bytes)?;
                        immutable_write(&output_dir.join(&predictions_file), &prediction_bytes)?;
                        Ok(WorkerCandidate {
                            generation: slot.generation,
                            slot: slot.slot,
                            semantic_sha256: slot.member.semantic_sha256.clone(),
                            receipt_file,
                            receipt_sha256: sha256_bytes(&receipt_bytes),
                            predictions_file,
                            predictions_sha256: sha256_bytes(&prediction_bytes),
                        })
                    })();
                    match result {
                        Ok(output) => outputs.lock().unwrap().push(output),
                        Err(error) => thread_failures.lock().unwrap().push(format!(
                            "generation {} slot {}: {error:#}",
                            slot.generation, slot.slot
                        )),
                    }
                });
            if let Err(error) = spawn {
                spawn_failures.lock().unwrap().push(format!(
                    "generation {} slot {}: spawn candidate thread: {error}",
                    slot.generation, slot.slot
                ));
            }
        }
    });
    let failures = Arc::try_unwrap(failures)
        .expect("Binary42 worker failure references remain")
        .into_inner()
        .unwrap();
    if !failures.is_empty() {
        bail!(
            "Binary42 worker infrastructure failures: {}",
            failures.join("; ")
        );
    }
    let mut candidates = Arc::try_unwrap(outputs)
        .expect("Binary42 worker output references remain")
        .into_inner()
        .unwrap();
    candidates.sort_by_key(|candidate| candidate.slot);
    if candidates.len() != fresh.len() {
        bail!("Binary42 worker produced a short fresh-candidate set");
    }
    let receipt = WorkerReceipt {
        schema: WORKER_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: manifest.kind.clone(),
        run_id: manifest.run_id.clone(),
        dataset: manifest.dataset.clone(),
        generation: manifest.generation,
        shard: manifest.shard,
        node: manifest.node.clone(),
        hostname,
        job_id: manifest.job_id.clone(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        logical_slots: manifest.slots.len(),
        fresh_candidates: fresh.len(),
        cpus_reserved: manifest.cpus_reserved,
        candidate_threads: 1,
        candidates,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(
        &output_dir.join("batch-worker-receipt.v1.json"),
        &canonical_json(&receipt)?,
    )
}
