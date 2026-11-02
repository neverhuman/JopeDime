use super::*;

pub(crate) fn validate_artifact_file(path: &Path) -> Result<()> {
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .context("ratchet artifact has no schema")?;
    match schema {
        DATASET_STATE_SCHEMA => serde_json::from_value::<DatasetLearningStateV2>(value)?.validate(),
        WAVE_SCHEMA => serde_json::from_value::<LearningWaveManifestV1>(value)?
            .validate(None, &BTreeSet::new()),
        SEED_REQUEST_V2_SCHEMA => {
            serde_json::from_value::<SeedInferenceRequestV2>(value)?.validate()
        }
        SEED_RESPONSE_V2_SCHEMA => match read_seed_inference(&bytes)? {
            ReadableSeedInference::V2(response) => {
                if response.proposals.len() != MODEL_PROPOSAL_COUNT {
                    bail!("v2 seed response does not contain 96 proposals");
                }
                for proposal in &response.proposals {
                    proposal.validate()?;
                }
                require_unique(
                    "v2 seed response",
                    response
                        .proposals
                        .iter()
                        .map(|proposal| proposal.candidate.semantic_sha256.as_str()),
                )
            }
            ReadableSeedInference::V1(_) => unreachable!(),
        },
        GENERATION_ZERO_V3_SCHEMA => {
            serde_json::from_value::<Generation0ReceiptV3>(value)?.validate()
        }
        REPLAY_V2_SCHEMA => {
            serde_json::from_value::<ReplayManifestV2>(value)?.validate(&BTreeSet::new())
        }
        TRAINING_HEAD_SCHEMA => serde_json::from_value::<TrainingHeadManifestV1>(value)?.validate(),
        "LearningSchedulerState.v4" => {
            serde_json::from_value::<SchedulerStateV4>(value)?.validate()
        }
        "RatchetLiveTelemetry.v2" => {
            serde_json::from_value::<RatchetLiveTelemetryV2>(value)?.validate()
        }
        "BroadShadowPilotManifest.v1" => {
            serde_json::from_value::<BroadShadowPilotManifestV1>(value)?.validate()
        }
        "FrozenRealDatasetSplits.v1" => {
            serde_json::from_value::<FrozenRealSplitsV1>(value)?.validate()
        }
        "ProductionModelExperimentManifest.v1" => {
            serde_json::from_value::<TrainingExperimentManifestV1>(value)?.validate()
        }
        schema if schema.starts_with("SeedInference") => {
            let _ = read_seed_inference(&bytes)?;
            Ok(())
        }
        schema if schema.starts_with("Generation0Receipt") => {
            let _ = read_generation_zero(&bytes)?;
            Ok(())
        }
        schema if schema.starts_with("AttemptAdmissionManifest") => {
            let _ = read_admission(&bytes)?;
            Ok(())
        }
        schema if schema.starts_with("RatchetLiveTelemetry") => {
            let _ = read_telemetry(&bytes)?;
            Ok(())
        }
        _ => bail!("unsupported ratchet artifact schema {schema}"),
    }
}

pub(crate) fn retain_model_seeds_from_files(
    request_path: &Path,
    response_path: &Path,
    excluded_semantics_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let request: SeedInferenceRequestV2 = serde_json::from_slice(&fs::read(request_path)?)?;
    let response: SeedInferenceResponseV2 = serde_json::from_slice(&fs::read(response_path)?)?;
    let excluded: BTreeSet<String> = serde_json::from_slice(&fs::read(excluded_semantics_path)?)?;
    let receipt = ModelRetentionReceiptV2::build(&request, &response, &excluded)?;
    immutable_write(output_path, &canonical_json(&receipt)?)
}

pub(crate) fn commit_dataset_state_from_file(
    root: &Path,
    state_path: &Path,
    at: u64,
) -> Result<()> {
    let state: DatasetLearningStateV2 = serde_json::from_slice(&fs::read(state_path)?)?;
    let digest = RatchetStore::new(root).commit_dataset_state(&state, at)?;
    println!("{digest}");
    Ok(())
}

pub(crate) fn publish_training_head_from_files(
    root: &Path,
    head_path: &Path,
    micro_batch_path: &Path,
    replay_path: &Path,
) -> Result<()> {
    let head: TrainingHeadManifestV1 = serde_json::from_slice(&fs::read(head_path)?)?;
    let batch: LearningMicroBatchManifestV1 = serde_json::from_slice(&fs::read(micro_batch_path)?)?;
    let replay: ReplayManifestV2 = serde_json::from_slice(&fs::read(replay_path)?)?;
    let digest = RatchetStore::new(root).publish_training_head(&head, &batch, &replay)?;
    println!("{digest}");
    Ok(())
}

pub(crate) fn publish_serving_wave_from_files(
    root: &Path,
    wave_path: &Path,
    switches_path: &Path,
    splits_path: &Path,
) -> Result<()> {
    let wave: LearningWaveManifestV1 = serde_json::from_slice(&fs::read(wave_path)?)?;
    let switches: RolloutSwitchesV1 = serde_json::from_slice(&fs::read(switches_path)?)?;
    let splits: FrozenRealSplitsV1 = serde_json::from_slice(&fs::read(splits_path)?)?;
    splits.validate()?;
    let digest = RatchetStore::new(root).publish_serving_wave(
        &wave,
        &switches,
        &splits.sealed_holdout_hashes(),
    )?;
    println!("{digest}");
    Ok(())
}

pub(crate) fn admit_attempt_from_files(
    root: &Path,
    generation_path: &Path,
    admission_path: &Path,
) -> Result<()> {
    let generation: Generation0ReceiptV3 = serde_json::from_slice(&fs::read(generation_path)?)?;
    let admission: AttemptAdmissionManifestV3 = serde_json::from_slice(&fs::read(admission_path)?)?;
    let digest = RatchetStore::new(root).admit_attempt(&generation, &admission)?;
    println!("{digest}");
    Ok(())
}
