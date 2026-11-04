use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetSearch {
    pub dataset: String,
    pub task: Task,
    pub pair: PairInputPlan,
    pub receipt: neural_search::PairedReceipt,
    pub neural: Option<neural_search::FrozenRecipe>,
    pub generic: Option<neural_search::FrozenRecipe>,
    pub neural_unavailable: Option<String>,
    pub neural_retryable_inventory: bool,
    pub final_test_targets_used: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teaching_admission: Option<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teaching_error: Option<String>,
}

/// Pair sampling seed. An absent round and round 0 keep the historical seed, so
/// accepted plans and K0 draw exactly the proposals they always drew.
pub(crate) fn pair_seed(root: u64, dataset_sha256: &str, round: Option<u32>) -> Result<u64> {
    let seed = jope_pipeline::search_plan::candidate_seed(root, dataset_sha256)?;
    match round {
        None | Some(0) => Ok(seed),
        Some(round) => jope_pipeline::search_plan::candidate_seed(
            seed,
            &digest(&("asm.search-round/v1", round))?,
        ),
    }
}

/// Read-only: verifies the pinned blob in the teaching state without creating it.
pub(crate) fn incumbent_seed(
    state_root: &Path,
    pin: &Artifact,
    selection: &jope_pipeline::search_plan::SelectionPlan,
    neural: bool,
) -> Result<neural_search::IncumbentSeed> {
    let state = jope_learning::state::State::at(state_root);
    ensure!(
        jope_pipeline::experiment::sha256(&pin.sha256) && pin.path == state.blob_path(&pin.sha256),
        "incumbent must pin a frozen-recipe blob in the teaching state"
    );
    // read_blob re-hashes the stored bytes and refuses a corrupt blob.
    let bytes = state
        .read_blob(&pin.sha256)?
        .context("incumbent frozen recipe is absent from the teaching state")?;
    neural_search::IncumbentSeed::from_frozen_recipe(&bytes, &pin.sha256, selection, neural)
}

fn resolve_incumbents(
    configuration: &Configuration,
    selection: &jope_pipeline::search_plan::SelectionPlan,
) -> Result<Option<neural_search::ArmIncumbents>> {
    let Some(pins) = &configuration.incumbents else {
        return Ok(None);
    };
    let settings = configuration
        .teaching
        .as_ref()
        .context("incumbent seeding requires a teaching state")?;
    let resolve = |pins: &[Artifact], neural: bool| {
        pins.iter()
            .map(|pin| incumbent_seed(&settings.state_root, pin, selection, neural))
            .collect::<Result<Vec<_>>>()
    };
    Ok(Some(neural_search::ArmIncumbents {
        neural: resolve(&pins.neural, true)?,
        generic: resolve(&pins.generic, false)?,
    }))
}

/// Receipt and weights the policy worker loads: the native challenger's, else the accepted pins.
pub(crate) fn proposal_files(configuration: &Configuration) -> (&Artifact, &Artifact) {
    configuration.native_challenger.as_ref().map_or(
        (
            &configuration.checkpoint_receipt,
            &configuration.policy_weights,
        ),
        |c| (&c.checkpoint_receipt, &c.policy_weights),
    )
}
/// Read-only: the checkpoint the policy worker will bind, from the configuration's pins. The
/// native challenger's receipt is held to the challenger rules; otherwise the pinned receipt must
/// be the accepted `tools/ot-export` receipt naming the pinned weights. A re-export of the
/// accepted checkpoint by this repository's exporter carries no acceptance authority and is
/// refused here, before any worker starts (W2 G-08, OWNER-DECISIONS.md D13).
pub(crate) fn pinned_checkpoint(
    config: &Configuration,
) -> Result<jope_policy::config::CheckpointBinding> {
    if let Some(challenger) = &config.native_challenger {
        return challenger_checkpoint(challenger);
    }
    config.checkpoint_receipt.verify()?;
    let checkpoint = jope_policy::config::CheckpointBinding::for_proposal(
        &fs::read(&config.checkpoint_receipt.path)?,
        None,
    )?;
    ensure!(
        checkpoint.policy_safetensors_sha256() == config.policy_weights.sha256,
        "policy checksum differs from accepted receipt"
    );
    Ok(checkpoint)
}
/// Read-only: the challenger's export receipt must bind its pinned identity and weights.
pub(crate) fn challenger_checkpoint(
    challenger: &NativeChallenger,
) -> Result<jope_policy::config::CheckpointBinding> {
    challenger.checkpoint_receipt.verify()?;
    let binding = jope_policy::config::CheckpointBinding::for_proposal(
        &fs::read(&challenger.checkpoint_receipt.path)?,
        Some(&challenger.identity),
    )?;
    ensure!(
        binding.policy_safetensors_sha256() == challenger.policy_weights.sha256,
        "native challenger weights differ from its export receipt"
    );
    Ok(binding)
}

/// Reconstruct every pair identity from exact authenticated train values and configuration.
/// This is shared by execution and read-only report admission; no artifact is written.
pub fn build_pair_input(
    campaign: &CampaignPlan,
    configuration: &Configuration,
    dataset: &DevelopmentData,
) -> Result<(
    super::super::experimental_proposal::ContextInput,
    PairInputPlan,
)> {
    let cut = dataset.cut(0)?;
    let mut outer = super::super::campaign::request(
        dataset,
        &cut,
        &campaign.methods[dataset.dataset.task.as_str()][0],
        &campaign.experiment,
        EvaluationStage::Screen,
    )?;
    // This is the pre-output pair slot, never a claimed executable recipe.
    outer.recipe_sha256 = digest(&(
        "asm.pair-slot/v1",
        &configuration.schema,
        &configuration.source_manifest.sha256,
    ))?;
    let selection = optimize_fit::selection(&cut, outer)?;
    let rows = cut
        .train
        .rows
        .iter()
        .map(|r| TrainingRow {
            id: r.id.clone(),
            values: r.values.clone(),
            label: r.target.clone(),
        })
        .collect::<Vec<_>>();
    let training_sha256 = context_producer::training_rows_sha256(&rows)?;
    // W1-T01: a member of the committed regression release takes its source
    // and prepared identities from the release admission its loader verified.
    let (dataset_sha256, prepared_sha256) = match &dataset.release {
        Some(release) => (
            release.dataset_sha256.clone(),
            release.development_sha256.clone(),
        ),
        None => (
            dataset.dataset.source.sha256.clone(),
            dataset.dataset.split.development_sha256.clone(),
        ),
    };
    let input = TrainingInput {
        schema: if dataset.dataset.task == Task::Regression {
            "jope.cpu-context-training/regression-v1"
        } else {
            "asm.experimental-classification-input/v1"
        }
        .into(),
        identity: context_producer::TrainingIdentity {
            dataset_sha256,
            prepared_sha256,
            training_sha256: training_sha256.clone(),
            fitting_rows_sha256: cut.train.rows_sha256.clone(),
            cut: 0,
        },
        features: dataset
            .dataset
            .features
            .iter()
            .map(|f| context_producer::Feature {
                name: f.name.clone(),
                kind: match f.kind {
                    ColumnKind::Numeric => context_producer::FeatureKind::Numeric,
                    ColumnKind::Categorical => context_producer::FeatureKind::Categorical,
                },
            })
            .collect(),
        rows,
    };
    let prepared = if dataset.dataset.task == Task::Regression {
        Some(input.prepare()?)
    } else {
        None
    };
    let preprocessing_sha256 = match prepared {
        Some(ref p) => digest(&p.preprocessing)?,
        None => digest(&("asm.classification-context-unavailable/v1", &input.features))?,
    };
    // Classification keeps original labels in its content identity. It is never
    // sent through the regression encoder or ordinal target context.
    let input_sha256 = digest(&input)?;
    let seed = pair_seed(
        campaign.experiment.seed,
        &dataset.dataset.source.sha256,
        configuration.search_round,
    )?;
    let incumbents = resolve_incumbents(configuration, &selection)?;
    // A native challenger changes only the receipt, weights, state and identity the policy
    // worker is pinned to; selection, encoding, seeds and variation are unchanged.
    let (proposal_receipt, proposal_weights) = proposal_files(configuration);
    let plan = PairInputPlan {
        selection,
        preprocessing_sha256,
        raw_input_policy: RAW_INPUT_POLICY.into(),
        fit_randomness_policy: FIT_RANDOMNESS_POLICY.into(),
        installation: InstallationPins {
            release_manifest_sha256: configuration.release_manifest.sha256.clone(),
            policy_program_sha256: configuration.policy.program.sha256.clone(),
            source_manifest_sha256: configuration.source_manifest.sha256.clone(),
            checkpoint_receipt_sha256: proposal_receipt.sha256.clone(),
        },
        proposal: RequestPlan {
            worker_source_sha256: configuration.source_manifest.sha256.clone(),
            prepared_sha256: input.identity.prepared_sha256.clone(),
            training_sha256: training_sha256.clone(),
            cut: 0,
            checkpoint_sha256: configuration
                .native_challenger
                .as_ref()
                .map_or(jope_policy::ACCEPTED_STATE_SHA256, |c| {
                    c.identity.state_sha256.as_str()
                })
                .into(),
            policy_safetensors_sha256: proposal_weights.sha256.clone(),
            vocabulary_sha256: configuration.vocabulary.sha256.clone(),
            sampling_seed: seed,
            fallback_seed: seed.wrapping_add(1),
            exploration_seed: seed.wrapping_add(2),
            sampling_rounds: configuration.sampling_rounds,
            sampling_batch: configuration.sampling_batch,
            sampling_temperature: configuration.sampling_temperature,
            challenger: configuration
                .native_challenger
                .as_ref()
                .map(|c| c.identity.clone()),
        }
        .into(),
        encoding: EncodingPlan {
            input_sha256,
            source_sha40: configuration.source_sha40.clone(),
            program_sha256: configuration.encoder.program.sha256.clone(),
            source_manifest_sha256: configuration.source_manifest.sha256.clone(),
            encoder_assets_manifest_sha256: configuration
                .encoder_assets
                .configuration
                .sha256
                .clone(),
            qualification_sha256: configuration
                .qualification
                .as_ref()
                .map(|q| q.qualification.sha256.clone()),
            reference_manifest_sha256: configuration
                .qualification
                .as_ref()
                .map(|q| q.reference_manifest.sha256.clone()),
            backend: if dataset.dataset.task == Task::Regression {
                context_producer::BACKEND
            } else {
                "asm.experimental-classification-context/unavailable"
            }
            .into(),
        },
        variation_identity: variation_policy(dataset.dataset.task).into(),
        generic_sampling_policy: generic_policy(dataset.dataset.task).into(),
        checkpoint_training_exposure: "unknown".into(),
        exposure_evidence_sha256: None,
        prepared_sha256: input.identity.prepared_sha256.clone(),
        training_sha256,
        cut: 0,
        vocabulary_sha256: configuration.vocabulary.sha256.clone(),
        seconds: configuration.seconds,
        max_evaluations: configuration.max_evaluations,
        random_seed: seed,
        incumbents,
    };
    if let Some(experimental) = configuration
        .experimental
        .get(dataset.dataset.task.as_str())
    {
        let vocabulary = jope_policy::grammar::bind_vocabulary(
            &fs::read(&configuration.vocabulary.path)?,
            jope_policy::CHECKPOINT_VOCABULARY_SHA256,
        )?;
        let mut variation = TaskVariation {
            vocabulary: &vocabulary,
            task: dataset.dataset.task,
        };
        return super::super::experimental_proposal::bind_pair(
            configuration,
            experimental,
            dataset,
            input,
            plan,
            &mut variation,
        );
    }
    Ok((input.into(), plan))
}
