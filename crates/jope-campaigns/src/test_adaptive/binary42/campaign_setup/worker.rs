use super::*;

pub(super) fn expected_job_id(manifest: &ShardManifest) -> Result<String> {
    let mut binding = manifest.clone();
    binding.job_id.clear();
    Ok(format!(
        "jope-b42-{}",
        &sha256_bytes(&canonical_json(&binding)?)[..24]
    ))
}

pub(super) fn validate_shard_manifest(manifest: &ShardManifest) -> Result<()> {
    let expected_slots = if manifest.kind == "generation" {
        if manifest.cohort_mode.as_deref() == Some("all") {
            *ALL_SHARD_SIZES
                .get(manifest.shard)
                .context("Binary42 all-dataset shard index is invalid")?
        } else {
            SHARD_SIZE
        }
    } else if manifest.kind == "confirmation" {
        1
    } else {
        bail!("unknown Binary42 shard kind");
    };
    let expected_cpus = expected_slots;
    let hostname = String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    let expected_hostname = physical_hostname(&manifest.node)?;
    let (dataset, _) = cohort_dataset(&manifest.dataset)?;
    if manifest.schema != SHARD_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || (manifest.cohort_mode.is_none() && manifest.dataset != ILPD)
        || (manifest.cohort_mode.is_some() && manifest.cohort_mode.as_deref() != Some("all"))
        || !NODES.contains(&manifest.node.as_str())
        || hostname != expected_hostname
        || manifest.slots.len() != expected_slots
        || manifest.cpus_reserved != expected_cpus
        || manifest.gpus_per_task != 0
        || manifest.candidate_threads != 1
        || manifest.job_id != expected_job_id(manifest)?
        || current_binary_sha256()? != manifest.release_binary_sha256
    {
        bail!("Binary42 shard identity or resource contract mismatch");
    }
    if manifest.train_sha256 != dataset.train_sha256
        || manifest.test_sha256 != dataset.test_sha256
        || manifest.train_rows != dataset.train_rows
        || manifest.test_rows != dataset.test_rows
        || manifest.features != dataset.features
        || (manifest.cohort_mode.as_deref() == Some("all")
            && manifest.scq_binary_path.as_deref() != Some(all_scq_binary_str()))
        || (manifest.cohort_mode.is_none() && manifest.scq_binary_path.is_some())
    {
        bail!("Binary42 shard dataset binding mismatch");
    }
    let mut slot_ids = BTreeSet::new();
    let mut semantics = BTreeSet::new();
    for slot in &manifest.slots {
        if slot.generation != manifest.generation
            || slot.member.semantic_sha256 != sha256_bytes(slot.member.canonical_genome.as_bytes())
            || !slot_ids.insert(slot.slot)
            || !semantics.insert(slot.member.semantic_sha256.clone())
        {
            bail!("Binary42 shard contains an invalid or duplicate slot");
        }
    }
    if manifest.kind == "confirmation" && manifest.slots.iter().any(|slot| !slot.fresh) {
        bail!("Binary42 confirmations must be independently fresh");
    }
    Ok(())
}

pub(super) fn failed_candidate_receipt(
    manifest: &ShardManifest,
    slot: &ShardSlot,
    hostname: &str,
    cpu: usize,
    runtime_seconds: f64,
    backend: Option<CandidateBackendReceipt>,
    failure: String,
) -> CandidateReceipt {
    let empty = binary_prediction_bytes(&[]);
    CandidateReceipt {
        schema: CANDIDATE_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: manifest.kind.clone(),
        run_id: manifest.run_id.clone(),
        dataset: manifest.dataset.clone(),
        generation: manifest.generation,
        slot: slot.slot,
        attempt: manifest.attempt,
        semantic_sha256: slot.member.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: slot.member.complexity,
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        train_sha256: manifest.train_sha256.clone(),
        test_sha256: manifest.test_sha256.clone(),
        train_rows: manifest.train_rows,
        test_rows: manifest.test_rows,
        features: manifest.features,
        node: manifest.node.clone(),
        hostname: hostname.to_string(),
        scq_job_id: manifest.job_id.clone(),
        candidate_threads: 1,
        cpu,
        runtime_seconds,
        fit_seconds: None,
        predict_seconds: None,
        backend,
        predictions_sha256: sha256_bytes(&empty),
        score: None,
        failure: Some(failure),
    }
}

pub(super) fn cpp17_toolchain() -> Result<CandidateBackendReceipt> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new("g++");
            command.arg("--version");
            command
        },
        "Binary42 fitted C++17 compiler identity",
    )?;
    let compiler = String::from_utf8(output.stdout)?
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    if !compiler.contains("13.3") {
        bail!("Binary42 fitted C++17 requires g++ 13.3, found {compiler:?}");
    }
    Ok(CandidateBackendReceipt {
        name: CPP17_BACKEND.to_string(),
        language_standard: CPP17_STANDARD.to_string(),
        compiler,
        compiler_flags: CPP17_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        source_sha256: None,
        codec_sha256: None,
        compile_seconds: None,
    })
}

pub(super) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn validate_candidate_backend(
    receipt: &CandidateReceipt,
    all_dataset_cohort: bool,
) -> Result<()> {
    match (&receipt.backend, all_dataset_cohort) {
        (None, false) => Ok(()),
        (Some(backend), true) => {
            if backend.name != CPP17_BACKEND
                || backend.language_standard != CPP17_STANDARD
                || !backend.compiler.contains("13.3")
                || backend.compiler_flags
                    != CPP17_FLAGS
                        .iter()
                        .map(|flag| (*flag).to_string())
                        .collect::<Vec<_>>()
            {
                bail!("Binary42 fitted C++17 backend identity mismatch");
            }
            match (&receipt.score, &receipt.failure) {
                (Some(_), None)
                    if backend.source_sha256.as_deref().is_some_and(is_sha256_hex)
                        && backend.codec_sha256.as_deref().is_some_and(is_sha256_hex)
                        && backend
                            .compile_seconds
                            .is_some_and(|seconds| seconds >= 0.0) =>
                {
                    Ok(())
                }
                (None, Some(_))
                    if backend.source_sha256.is_none()
                        && backend.codec_sha256.is_none()
                        && backend.compile_seconds.is_none() =>
                {
                    Ok(())
                }
                _ => bail!("Binary42 fitted C++17 build provenance is incomplete"),
            }
        }
        _ => bail!("Binary42 candidate used the wrong execution backend"),
    }
}

pub(super) fn evaluate_worker_candidate(
    manifest: &ShardManifest,
    slot: &ShardSlot,
    train: &RegressionData,
    test: &RegressionData,
    cpu: usize,
    cpp17_toolchain: Option<&CandidateBackendReceipt>,
    runtime_root: &Path,
) -> (CandidateReceipt, Vec<u8>) {
    let started = Instant::now();
    let all_dataset_cohort = manifest.cohort_mode.as_deref() == Some("all");
    let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if all_dataset_cohort {
            let built = evaluate_test_adaptive_binary_cpp17(
                &slot.member.canonical_genome,
                train,
                test,
                runtime_root,
            )?;
            let mut backend = cpp17_toolchain
                .context("Binary42 fitted C++17 toolchain was not initialized")?
                .clone();
            backend.source_sha256 = Some(built.source_sha256);
            backend.codec_sha256 = Some(built.codec_sha256);
            backend.compile_seconds = Some(built.compile_seconds);
            Ok((built.evaluation, Some(backend)))
        } else {
            evaluate_test_adaptive_binary(&slot.member.canonical_genome, train, test)
                .map(|evaluation| (evaluation, None))
        }
    }));
    match evaluated {
        Ok(Ok((result, backend))) => {
            let predictions = if all_dataset_cohort {
                binary_prediction_roundtrip_bytes(&result.predictions)
            } else {
                binary_prediction_bytes(&result.predictions)
            };
            (
                CandidateReceipt {
                    schema: CANDIDATE_SCHEMA.to_string(),
                    test_adaptive: true,
                    benchmark_valid: false,
                    kind: manifest.kind.clone(),
                    run_id: manifest.run_id.clone(),
                    dataset: manifest.dataset.clone(),
                    generation: manifest.generation,
                    slot: slot.slot,
                    attempt: manifest.attempt,
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    canonical_genome: slot.member.canonical_genome.clone(),
                    complexity: result.complexity,
                    release_binary_sha256: manifest.release_binary_sha256.clone(),
                    train_sha256: manifest.train_sha256.clone(),
                    test_sha256: manifest.test_sha256.clone(),
                    train_rows: manifest.train_rows,
                    test_rows: manifest.test_rows,
                    features: manifest.features,
                    node: manifest.node.clone(),
                    hostname: manifest.node.clone(),
                    scq_job_id: manifest.job_id.clone(),
                    candidate_threads: 1,
                    cpu,
                    runtime_seconds: started.elapsed().as_secs_f64(),
                    fit_seconds: Some(result.fit_seconds),
                    predict_seconds: Some(result.predict_seconds),
                    backend,
                    predictions_sha256: sha256_bytes(&predictions),
                    score: Some(CandidateScore {
                        auc: result.auc,
                        logloss: result.logloss,
                    }),
                    failure: None,
                },
                predictions,
            )
        }
        Ok(Err(error)) => {
            let predictions = binary_prediction_bytes(&[]);
            (
                failed_candidate_receipt(
                    manifest,
                    slot,
                    &manifest.node,
                    cpu,
                    started.elapsed().as_secs_f64(),
                    cpp17_toolchain.cloned(),
                    format!("{error:#}"),
                ),
                predictions,
            )
        }
        Err(_) => {
            let predictions = binary_prediction_bytes(&[]);
            (
                failed_candidate_receipt(
                    manifest,
                    slot,
                    &manifest.node,
                    cpu,
                    started.elapsed().as_secs_f64(),
                    cpp17_toolchain.cloned(),
                    "candidate panicked".to_string(),
                ),
                predictions,
            )
        }
    }
}
