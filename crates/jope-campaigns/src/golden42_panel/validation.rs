use super::*;

pub(crate) fn validate(run_id: &str) -> Result<()> {
    let root = workspace_root().join("runs").join(run_id);
    let manifest: CampaignManifest =
        serde_json::from_slice(&fs::read(root.join("manifest.v3.json"))?)?;
    let admission_bytes = fs::read(workspace_root().join(ADMISSION_RELATIVE))?;
    let admission: AdmissionReceipt = serde_json::from_slice(&admission_bytes)?;
    let release_sha256 = sha256_file(&std::env::current_exe()?)?;
    if manifest.schema != "jope-dime.golden42-v3-max-manifest.v1"
        || manifest.run_id != run_id
        || manifest.dataset_hash != DATASET
        || manifest.workers != WORKERS
        || manifest.population != WORKERS
        || manifest.strict_threshold.to_bits() != WINNER_RMSE.to_bits()
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.prediction_backend != "fitted-cpp17-only"
        || manifest.release_sha256 != release_sha256
        || manifest.admission_sha256 != sha256_bytes(&admission_bytes)
        || !admission.admitted
        || admission.release_sha256 != release_sha256
        || admission.source_sha256 != source_tree_hash()?
    {
        bail!("V3 campaign manifest contract mismatch");
    }
    validate_checkpoint_chain(&root, run_id)?;
    let result_path = root.join("final-result.v1.json");
    if result_path.exists() {
        let result: FinalResult = serde_json::from_slice(&fs::read(&result_path)?)?;
        if result.schema != "jope-dime.golden42-v3-max-result.v1"
            || result.run_id != run_id
            || result.dataset_hash != DATASET
            || result.outcome != "confirmed_strict_winner"
            || result.rmse >= WINNER_RMSE
            || result.strict_threshold.to_bits() != WINNER_RMSE.to_bits()
            || result.confirmation_sha256.len() != 3
            || !result.test_adaptive
            || result.benchmark_valid
        {
            bail!("V3 final result is not a confirmed strict winner");
        }
        let claim_bytes = fs::read(root.join("winner-claim.v1.json"))?;
        let claim: WinnerClaim = serde_json::from_slice(&claim_bytes)?;
        if sha256_bytes(&claim_bytes) != result.winner_claim_sha256
            || claim.schema != "jope-dime.golden42-v3-winner-claim.v1"
            || claim.run_id != run_id
            || claim.dataset_hash != DATASET
            || claim.generation != result.generation
            || claim.slot != result.slot
            || claim.semantic_hash != result.candidate.semantic_hash
            || claim.rmse.to_bits() != result.rmse.to_bits()
            || claim.threshold.to_bits() != WINNER_RMSE.to_bits()
            || claim.codec_sha256 != result.codec_sha256
            || claim.prediction_sha256 != result.prediction_sha256
        {
            bail!("V3 winner claim does not bind the final result");
        }
        let expected_candidate = if result.generation == 0 {
            manifest
                .initial_population
                .get(result.slot)
                .context("winner slot exceeds initial population")?
                .clone()
        } else {
            let prior_path = root
                .join("checkpoints")
                .join(format!("generation-{:06}.v1.json", result.generation - 1));
            let prior: Checkpoint = serde_json::from_slice(&fs::read(prior_path)?)?;
            prior
                .next_population
                .get(result.slot)
                .context("winner slot exceeds checkpoint population")?
                .clone()
        };
        let test = read_regression_csv(&paths::jope_registry().join(DATASET).join("test.csv"))?;
        let receipt_path = root.join("receipts").join(format!(
            "generation-{:06}/slot-{:03}.v1.json",
            result.generation, result.slot
        ));
        let receipt: CandidateReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
        validate_candidate_receipt(
            &receipt,
            &root,
            run_id,
            result.generation,
            result.slot,
            &test.y,
            &expected_candidate,
            &release_sha256,
        )?;
        if receipt.rmse.to_bits() != result.rmse.to_bits()
            || receipt.codec_sha256 != result.codec_sha256
            || receipt.prediction_sha256 != result.prediction_sha256
            || receipt.candidate != result.candidate
        {
            bail!("V3 final result differs from its candidate receipt");
        }
        if !result
            .frozen_codec_path
            .starts_with(root.join("winner-package"))
            || !result
                .standalone_source_path
                .starts_with(root.join("winner-package"))
            || sha256_file(&result.frozen_codec_path)? != result.codec_sha256
            || sha256_file(&result.standalone_source_path)? != result.standalone_source_sha256
        {
            bail!("V3 winner package hashes or locations are invalid");
        }
        validate_fitted_cpp_codec(&fs::read(&result.frozen_codec_path)?)?;
        let mut confirmation_cpus = BTreeSet::new();
        let mut confirmation_pids = BTreeSet::new();
        for (index, expected) in result.confirmation_sha256.iter().enumerate() {
            let bytes = fs::read(
                root.join("confirmation")
                    .join(format!("confirmation-{index}.v1.json")),
            )?;
            if sha256_bytes(&bytes) != *expected {
                bail!("V3 confirmation {index} hash mismatch");
            }
            let confirmation: ConfirmationReceipt = serde_json::from_slice(&bytes)?;
            if confirmation.schema != "jope-dime.golden42-v3-confirmation.v1"
                || confirmation.codec_sha256 != result.codec_sha256
                || confirmation.prediction_sha256 != result.prediction_sha256
                || confirmation.prediction_sha256 != prediction_hash(&confirmation.predictions)
                || raw_rmse(&test.y, &confirmation.predictions)? >= WINNER_RMSE
            {
                bail!("V3 confirmation {index} payload mismatch");
            }
            confirmation_cpus.insert(confirmation.cpu);
            confirmation_pids.insert(confirmation.pid);
        }
        if confirmation_cpus.len() != 3 || confirmation_pids.len() != 3 {
            bail!("V3 confirmations were not fresh processes on distinct CPUs");
        }
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("terminal-summary.v1.json"))?)?;
        if summary["schema"] != "jope-dime.golden42-v3-terminal-summary.v1"
            || summary["run_id"] != run_id
            || summary["operator_success"] != true
        {
            bail!("V3 terminal utilization summary is invalid");
        }
    }
    Ok(())
}

pub(super) fn resume_state(
    root: &Path,
    manifest: &CampaignManifest,
) -> Result<(usize, Vec<V3Candidate>, V3Archive, Option<String>)> {
    let Some((checkpoint, sha256)) = validate_checkpoint_chain(root, &manifest.run_id)? else {
        return Ok((
            0,
            manifest.initial_population.clone(),
            V3Archive::default(),
            None,
        ));
    };
    Ok((
        checkpoint.generation + 1,
        checkpoint.next_population,
        checkpoint.archive,
        Some(sha256),
    ))
}

pub(super) fn validate_checkpoint_chain(
    root: &Path,
    run_id: &str,
) -> Result<Option<(Checkpoint, String)>> {
    let checkpoint_root = root.join("checkpoints");
    if !checkpoint_root.exists() {
        return Ok(None);
    }
    let mut paths = fs::read_dir(&checkpoint_root)?
        .filter_map(|entry| match entry {
            Ok(entry)
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("generation-")
                    && entry.file_name().to_string_lossy().ends_with(".v1.json") =>
            {
                Some(Ok(entry.path()))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    let mut previous = None;
    let mut last = None;
    for (generation, path) in paths.iter().enumerate() {
        let expected_name = format!("generation-{generation:06}.v1.json");
        if path.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str()) {
            bail!("V3 checkpoint filenames are not contiguous");
        }
        let bytes = fs::read(path)?;
        let checkpoint: Checkpoint = serde_json::from_slice(&bytes)?;
        if checkpoint.schema != "jope-dime.golden42-v3-checkpoint.v1"
            || checkpoint.run_id != run_id
            || checkpoint.dataset_hash != DATASET
            || checkpoint.generation != generation
            || checkpoint.previous_sha256 != previous
            || checkpoint.scored.len() != WORKERS
            || checkpoint.ranked_semantic_hashes.len() != WORKERS
            || checkpoint.next_population.len() != WORKERS
            || checkpoint.archive.entries.len() > 128
            || checkpoint.scored.iter().any(|item| !item.rmse.is_finite())
            || !checkpoint
                .scored
                .windows(2)
                .all(|pair| scored_order(&pair[0], &pair[1]) != std::cmp::Ordering::Greater)
            || checkpoint.ranked_semantic_hashes
                != checkpoint
                    .scored
                    .iter()
                    .map(|item| item.candidate.semantic_hash.clone())
                    .collect::<Vec<_>>()
        {
            bail!("V3 checkpoint chain or payload is invalid");
        }
        let unique = checkpoint
            .next_population
            .iter()
            .map(|candidate| &candidate.semantic_hash)
            .collect::<BTreeSet<_>>();
        if unique.len() != WORKERS {
            bail!("V3 checkpoint next population contains duplicate semantics");
        }
        for island in 0..4 {
            if checkpoint
                .next_population
                .iter()
                .filter(|candidate| candidate.island as usize == island)
                .count()
                != 20
            {
                bail!("V3 checkpoint next population has an invalid island size");
            }
        }
        for candidate in checkpoint
            .scored
            .iter()
            .map(|item| &item.candidate)
            .chain(checkpoint.next_population.iter())
        {
            if GenomeIr::parse(&candidate.syntax)?.canonical() != candidate.syntax {
                bail!("V3 checkpoint contains a noncanonical candidate");
            }
        }
        let sha256 = sha256_bytes(&bytes);
        previous = Some(sha256.clone());
        last = Some((checkpoint, sha256));
    }
    Ok(last)
}

// Keep each receipt identity and worker contract input explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_candidate_receipt(
    receipt: &CandidateReceipt,
    root: &Path,
    run_id: &str,
    generation: usize,
    slot: usize,
    targets: &[f64],
    expected_candidate: &V3Candidate,
    release_sha256: &str,
) -> Result<()> {
    if receipt.schema != "jope-dime.golden42-v3-candidate-receipt.v1"
        || receipt.run_id != run_id
        || receipt.dataset_hash != DATASET
        || receipt.generation != generation
        || receipt.slot != slot
        || &receipt.candidate != expected_candidate
        || receipt.context_sha256 != context_hash(release_sha256, expected_candidate)
        || receipt.predictions.len() != targets.len()
        || receipt.prediction_sha256 != prediction_hash(&receipt.predictions)
        || !receipt.rmse.is_finite()
    {
        bail!("invalid cached V3 candidate receipt");
    }
    let recomputed = raw_rmse(targets, &receipt.predictions)?;
    if recomputed.to_bits() != receipt.rmse.to_bits() {
        bail!("cached V3 candidate RMSE does not match its prediction tape");
    }
    validate_codec_artifact(root, &receipt.codec_file, &receipt.codec_sha256)?;
    Ok(())
}

pub(super) fn validate_codec_artifact(
    root: &Path,
    path: &Path,
    expected_sha256: &str,
) -> Result<()> {
    if !path.starts_with(root.join("worker-artifacts")) || !path.is_file() {
        bail!("fitted codec artifact is outside the worker artifact root");
    }
    let bytes = fs::read(path)?;
    if sha256_bytes(&bytes) != expected_sha256 {
        bail!("fitted codec artifact hash mismatch");
    }
    validate_fitted_cpp_codec(&bytes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_manifest_contract(
    manifest: &CampaignManifest,
    run_id: &str,
    node: &str,
    workers: usize,
    population: usize,
    release_sha256: &str,
    admission_sha256: &str,
    initial_population: &[V3Candidate],
) -> Result<()> {
    if manifest.schema != "jope-dime.golden42-v3-max-manifest.v1"
        || manifest.run_id != run_id
        || manifest.dataset_hash != DATASET
        || manifest.train_sha256 != TRAIN_SHA256
        || manifest.test_sha256 != TEST_SHA256
        || manifest.node != node
        || manifest.workers != workers
        || manifest.population != population
        || manifest.islands != 4
        || manifest.island_size != 20
        || manifest.strict_threshold.to_bits() != WINNER_RMSE.to_bits()
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.prediction_backend != "fitted-cpp17-only"
        || manifest.release_sha256 != release_sha256
        || manifest.admission_sha256 != admission_sha256
        || manifest.initial_population != initial_population
    {
        bail!("existing V3 campaign manifest does not match this release and run contract");
    }
    Ok(())
}
