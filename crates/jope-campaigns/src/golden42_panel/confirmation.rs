use super::*;

pub(crate) fn confirm(request_path: &Path) -> Result<()> {
    let request: ConfirmationRequest = serde_json::from_slice(&fs::read(request_path)?)?;
    if request.schema != "jope-dime.golden42-v3-confirmation-request.v1" {
        bail!("invalid V3 confirmation request");
    }
    let train = read_regression_csv(&request.train_path)?;
    let test_x: Vec<Vec<f64>> = serde_json::from_slice(&fs::read(&request.test_features_path)?)?;
    let encoded = EncodedTable::from_numeric_rows(&test_x)?;
    let training = campaign_training(&train.x)?;
    let model = fit_campaign_candidate(
        &GenomeIr::parse(&request.candidate.syntax)?,
        &training,
        &train.y,
    )?;
    let codec = encode_fitted_cpp17(&model)?;
    let runtime = FittedCppRuntime::new(&request.runtime_root)?;
    let predictions = runtime.load(&codec.bytes)?.predict(&encoded)?;
    let prediction_sha256 = prediction_hash(&predictions);
    if codec.sha256 != request.expected_codec_sha256
        || prediction_sha256 != request.expected_prediction_sha256
    {
        bail!("fresh fitted C++ confirmation changed codec or predictions");
    }
    let receipt = ConfirmationReceipt {
        schema: "jope-dime.golden42-v3-confirmation.v1".to_string(),
        cpu: request.cpu,
        pid: std::process::id(),
        codec_sha256: codec.sha256,
        prediction_sha256,
        predictions,
        confirmed_at_unix_ms: unix_ms(),
    };
    atomic_immutable_write(&request.output, &json_bytes(&receipt)?)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_winner(
    root: &Path,
    run_id: &str,
    generation: usize,
    winner: CandidateReceipt,
    workers: &mut [WorkerProcess],
    cpus: &[usize],
    release: &Path,
    train_path: &Path,
    test_features_path: &Path,
    targets: &[f64],
) -> Result<()> {
    if winner.rmse >= WINNER_RMSE {
        bail!("attempted to finalize a nonwinning V3 candidate");
    }
    let expected_claim = WinnerClaim {
        schema: "jope-dime.golden42-v3-winner-claim.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: DATASET.to_string(),
        generation,
        slot: winner.slot,
        semantic_hash: winner.candidate.semantic_hash.clone(),
        rmse: winner.rmse,
        threshold: WINNER_RMSE,
        codec_sha256: winner.codec_sha256.clone(),
        prediction_sha256: winner.prediction_sha256.clone(),
        claimed_at_unix_ms: unix_ms(),
    };
    let claim_path = root.join("winner-claim.v1.json");
    let claim_bytes = if claim_path.exists() {
        let bytes = fs::read(&claim_path)?;
        let present: WinnerClaim = serde_json::from_slice(&bytes)?;
        if present.schema != expected_claim.schema
            || present.run_id != expected_claim.run_id
            || present.dataset_hash != expected_claim.dataset_hash
            || present.generation != expected_claim.generation
            || present.slot != expected_claim.slot
            || present.semantic_hash != expected_claim.semantic_hash
            || present.rmse.to_bits() != expected_claim.rmse.to_bits()
            || present.threshold.to_bits() != expected_claim.threshold.to_bits()
            || present.codec_sha256 != expected_claim.codec_sha256
            || present.prediction_sha256 != expected_claim.prediction_sha256
        {
            bail!("existing V3 winner claim conflicts with the qualifying receipt");
        }
        bytes
    } else {
        let bytes = json_bytes(&expected_claim)?;
        atomic_immutable_write(&claim_path, &bytes)?;
        bytes
    };
    terminate_workers(workers);
    let confirmations = confirm_winner(
        root,
        &winner,
        cpus,
        release,
        train_path,
        test_features_path,
        targets,
    )?;
    let confirmations_complete = confirmations.len() == 3;
    let train = read_regression_csv(train_path)?;
    let training = campaign_training(&train.x)?;
    let packaged_model = fit_campaign_candidate(
        &GenomeIr::parse(&winner.candidate.syntax)?,
        &training,
        &train.y,
    )?;
    let packaged_artifact = emit_fitted_cpp17(&packaged_model)?;
    let packaged_codec = encode_fitted_cpp17(&packaged_model)?;
    if packaged_codec.sha256 != winner.codec_sha256 {
        bail!("standalone winner emission changed the admitted fitted codec");
    }
    let package_root = root.join("winner-package");
    let frozen_codec_path = package_root.join("winner.fitted.v2.codec");
    let standalone_source_path = package_root.join("winner.cpp");
    atomic_immutable_write(&frozen_codec_path, &packaged_codec.bytes)?;
    atomic_immutable_write(&standalone_source_path, packaged_artifact.source.as_bytes())?;
    let result = FinalResult {
        schema: "jope-dime.golden42-v3-max-result.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: DATASET.to_string(),
        outcome: "confirmed_strict_winner".to_string(),
        generation,
        slot: winner.slot,
        rmse: winner.rmse,
        strict_threshold: WINNER_RMSE,
        candidate: winner.candidate,
        winner_claim_sha256: sha256_bytes(&claim_bytes),
        confirmation_sha256: confirmations,
        codec_sha256: winner.codec_sha256,
        prediction_sha256: winner.prediction_sha256,
        standalone_source_sha256: packaged_artifact.source_sha256,
        frozen_codec_path,
        standalone_source_path,
        test_adaptive: true,
        benchmark_valid: false,
        finished_at_unix_ms: unix_ms(),
    };
    let summary_path = root.join("terminal-summary.v1.json");
    if !summary_path.exists() {
        let health = resource_health()?;
        let workers_terminated = workers
            .iter_mut()
            .all(|worker| worker.child.try_wait().ok().flatten().is_some());
        let summary = serde_json::json!({
            "schema": "jope-dime.golden42-v3-terminal-summary.v1",
            "run_id": run_id,
            "outcome": "confirmed_strict_winner",
            "generation": generation,
            "slot": winner.slot,
            "rmse": winner.rmse,
            "strict_threshold": WINNER_RMSE,
            "worker_processes": workers.len(),
            "workers_terminated": workers_terminated,
            "candidate_latency_ms": (winner.fit_ns + winner.codec_ns + winner.cpp_load_ns + winner.cpp_predict_ns) as f64 / 1_000_000.0,
            "fit_ms": winner.fit_ns as f64 / 1_000_000.0,
            "codec_ms": winner.codec_ns as f64 / 1_000_000.0,
            "cpp_load_ms": winner.cpp_load_ns as f64 / 1_000_000.0,
            "cpp_predict_ms": winner.cpp_predict_ns as f64 / 1_000_000.0,
            "cpu_load_1m": health.load,
            "available_ram_bytes": health.ram,
            "swap_used_bytes": health.swap,
            "free_disk_bytes": health.disk,
            "cache_hits": 0,
            "invalids": 0,
            "diversity": WORKERS,
            "operator_success": workers_terminated && confirmations_complete,
            "candidate_finished_at_unix_ms": winner.finished_at_unix_ms,
        });
        atomic_immutable_write(&summary_path, &json_bytes(&summary)?)?;
    }
    atomic_immutable_write(&root.join("final-result.v1.json"), &json_bytes(&result)?)
}

pub(super) fn confirm_winner(
    root: &Path,
    winner: &CandidateReceipt,
    cpus: &[usize],
    release: &Path,
    train_path: &Path,
    test_features_path: &Path,
    targets: &[f64],
) -> Result<Vec<String>> {
    let confirmation_root = root.join("confirmation");
    fs::create_dir_all(&confirmation_root)?;
    let mut hashes = Vec::new();
    for (index, cpu) in cpus.iter().copied().take(3).enumerate() {
        let output = confirmation_root.join(format!("confirmation-{index}.v1.json"));
        let request = ConfirmationRequest {
            schema: "jope-dime.golden42-v3-confirmation-request.v1".to_string(),
            cpu,
            candidate: winner.candidate.clone(),
            train_path: train_path.to_path_buf(),
            test_features_path: test_features_path.to_path_buf(),
            runtime_root: confirmation_root.join(format!("runtime-{index}")),
            expected_codec_sha256: winner.codec_sha256.clone(),
            expected_prediction_sha256: winner.prediction_sha256.clone(),
            output: output.clone(),
        };
        let request_path = confirmation_root.join(format!("request-{index}.json"));
        atomic_immutable_write(&request_path, &json_bytes(&request)?)?;
        if output.exists() {
            let bytes = fs::read(&output)?;
            let receipt: ConfirmationReceipt = serde_json::from_slice(&bytes)?;
            if receipt.schema != "jope-dime.golden42-v3-confirmation.v1"
                || receipt.cpu != cpu
                || receipt.codec_sha256 != winner.codec_sha256
                || receipt.prediction_sha256 != winner.prediction_sha256
                || receipt.prediction_sha256 != prediction_hash(&receipt.predictions)
                || raw_rmse(targets, &receipt.predictions)? >= WINNER_RMSE
            {
                bail!("existing V3 confirmation {index} is invalid");
            }
            hashes.push(sha256_bytes(&bytes));
            continue;
        }
        let status = Command::new("taskset")
            .args(["-c", &cpu.to_string()])
            .arg(release)
            .arg("golden42-v3-confirm")
            .args(["--request", &request_path.display().to_string()])
            .status()?;
        if !status.success() {
            bail!("fresh fitted C++ confirmation {index} failed");
        }
        let bytes = fs::read(&output)?;
        let receipt: ConfirmationReceipt = serde_json::from_slice(&bytes)?;
        if receipt.schema != "jope-dime.golden42-v3-confirmation.v1"
            || receipt.cpu != cpu
            || receipt.codec_sha256 != winner.codec_sha256
            || receipt.prediction_sha256 != winner.prediction_sha256
            || receipt.prediction_sha256 != prediction_hash(&receipt.predictions)
            || raw_rmse(targets, &receipt.predictions)? >= WINNER_RMSE
        {
            bail!("fresh fitted C++ confirmation {index} did not remain a strict winner");
        }
        hashes.push(sha256_bytes(&bytes));
    }
    Ok(hashes)
}
