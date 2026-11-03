use super::*;

mod train_only_dispatch;

pub(super) fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.kernel_version == KernelVersion::AdvancedV3 && command_starts_campaign(&cli.command) {
        bail!(
            "advanced-v3 campaign launch is admission-gated; use a fresh run only after the v3 benchmark gate has been recorded (existing manifests remain legacy-v2)"
        );
    }
    if train_only_dispatch::accepts(&cli.command) {
        return train_only_dispatch::run(
            cli.command,
            &cli.registry,
            cli.dataset.as_deref(),
            cli.kernel_version,
        );
    }
    match cli.command {
        CampaignCommand::VerifyInputs => verify_inputs(&cli.registry),
        CampaignCommand::SelectPanel => {
            select_panel(&cli.registry, &cli.panel_id, cli.dataset.as_deref())
        }
        CampaignCommand::ValidatePanel => validate_panel(&cli.registry, &cli.panel_id),
        CampaignCommand::ScorePanel => score_panel(&cli.registry, &cli.panel_id),
        CampaignCommand::VerifyEvaluationV2 { manifest } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("verify-evaluation-v2 is pinned to the legacy-v2 evaluator");
            }
            commands::single_fold_proof::verify_manifest(
                &manifest,
                &cli.registry,
                cli.dataset.as_deref(),
            )
        }
        CampaignCommand::VerifyInputsBinary => verify_inputs_binary(&cli.registry),
        CampaignCommand::SelectPanelBinary => {
            select_panel_binary(&cli.registry, &cli.panel_id, cli.dataset.as_deref())
        }
        CampaignCommand::ValidatePanelBinary => validate_panel_binary(&cli.registry, &cli.panel_id),
        CampaignCommand::ScorePanelBinary => score_panel_binary(&cli.registry, &cli.panel_id),
        CampaignCommand::RunBinaryTestAdaptive { run_id, resume } => {
            let dataset = cli
                .dataset
                .as_deref()
                .context("run-binary-test-adaptive requires --dataset")?;
            test_adaptive::run_binary42(&run_id, dataset, resume)
        }
        CampaignCommand::ValidateBinaryTestAdaptive { run_id } => {
            let dataset = cli
                .dataset
                .as_deref()
                .context("validate-binary-test-adaptive requires --dataset")?;
            test_adaptive::validate_binary42(&run_id, dataset)
        }
        CampaignCommand::RunBinaryTestAdaptiveAll { run_id, resume } => {
            if cli.dataset.is_some() {
                bail!("run-binary-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_binary42_all(&run_id, resume)
        }
        CampaignCommand::ValidateBinaryTestAdaptiveAll { run_id } => {
            if cli.dataset.is_some() {
                bail!("validate-binary-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_binary42_all(&run_id)
        }
        CampaignCommand::PlanMulticlassTestAdaptive {
            run_id,
            campaign_root,
            corpus_manifest,
            comparator_root,
            seed_catalog,
            population,
            generations,
        } => {
            if cli.dataset.is_some() {
                bail!("plan-multiclass-test-adaptive does not accept --dataset");
            }
            if population != 60 || generations != 20 {
                bail!(
                    "multiclass all-dataset campaign is frozen at population=60 and generations=20"
                );
            }
            test_adaptive::plan_multiclass60(
                &run_id,
                &campaign_root,
                &corpus_manifest,
                &comparator_root,
                &seed_catalog,
            )
        }
        CampaignCommand::RunMulticlassTestAdaptiveAll {
            run_id,
            campaign_root,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-multiclass-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_multiclass60_all(&run_id, &campaign_root, resume)
        }
        CampaignCommand::ValidateMulticlassTestAdaptiveAll {
            run_id,
            campaign_root,
        } => {
            if cli.dataset.is_some() {
                bail!("validate-multiclass-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_multiclass60_all(&run_id, &campaign_root)
        }
        CampaignCommand::RunTimeseriesTestAdaptiveAll { run_id, resume } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_timeseries42_all(&run_id, resume)
        }
        CampaignCommand::ValidateTimeseriesTestAdaptiveAll { run_id } => {
            if cli.dataset.is_some() {
                bail!("validate-timeseries-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_timeseries42_all(&run_id)
        }
        CampaignCommand::RunTimeseriesTestAdaptiveGen30 {
            base_run_id,
            run_id,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-gen30 does not accept --dataset");
            }
            test_adaptive::run_timeseries42_gen30(&base_run_id, &run_id, resume)
        }
        CampaignCommand::ValidateTimeseriesTestAdaptiveGen30 {
            base_run_id,
            run_id,
        } => {
            if cli.dataset.is_some() {
                bail!("validate-timeseries-test-adaptive-gen30 does not accept --dataset");
            }
            test_adaptive::validate_timeseries42_gen30(&base_run_id, &run_id)
        }
        CampaignCommand::RunTimeseriesTestAdaptiveWinall {
            source_run_id,
            run_id,
            max_generations,
            lineage_salt,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-winall does not accept --dataset");
            }
            test_adaptive::run_timeseries42_winall(
                &source_run_id,
                &run_id,
                resume,
                max_generations,
                lineage_salt.as_deref(),
            )
        }
        CampaignCommand::BinaryTestAdaptiveBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::binary42_batch_worker(&manifest, &output_dir),
        CampaignCommand::TimeseriesTestAdaptiveBatchWorker {
            manifest,
            output_dir,
            corpus_eval,
            timeout_seconds,
        } => test_adaptive::timeseries42_batch_worker(
            &manifest,
            &output_dir,
            &corpus_eval,
            timeout_seconds,
        ),
        CampaignCommand::RunPanel => {
            select_panel(&cli.registry, &cli.panel_id, cli.dataset.as_deref())?;
            score_panel(&cli.registry, &cli.panel_id)
        }
        CampaignCommand::Summarize => summarize(&cli.registry, &cli.panel_id),
        CampaignCommand::RunLosses3x => followup::run(&cli.registry),
        CampaignCommand::ValidateLosses3x => followup::validate(&cli.registry),
        CampaignCommand::RunTestAdaptive { run_id, resume } => {
            test_adaptive::run(&cli.registry, &run_id, resume)
        }
        CampaignCommand::RunTestAdaptivePanel { run_id, resume } => {
            test_adaptive::run_panel(&cli.registry, &run_id, resume)
        }
        CampaignCommand::RunTestAdaptiveExtension { run_id, resume } => {
            test_adaptive::run_extension(&cli.registry, &run_id, resume)
        }
        CampaignCommand::RunTestAdaptiveIslandExtension { run_id, resume } => {
            test_adaptive::run_island_extension(&cli.registry, &run_id, resume)
        }
        CampaignCommand::RunGolden42Champions { run_id, resume } => {
            test_adaptive::run_champion_continuation(&cli.registry, &run_id, resume)
        }
        CampaignCommand::RunTabarenaLiteIslands { run_id, resume } => {
            test_adaptive::run_tabarena_lite_islands(&run_id, resume, cli.dataset.as_deref())
        }
        CampaignCommand::RecoverTabarenaLiteResult { run_id, dataset } => {
            test_adaptive::recover_tabarena_lite_result(&run_id, &dataset)
        }
        CampaignCommand::TestAdaptiveWorker { request } => test_adaptive::worker(&request),
        CampaignCommand::TestAdaptiveIslandBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::island_batch_worker(&manifest, &output_dir),
        CampaignCommand::ValidateTestAdaptive { run_id } => {
            test_adaptive::validate(&cli.registry, &run_id)
        }
        CampaignCommand::ValidateTestAdaptivePanel { run_id } => {
            test_adaptive::validate_panel(&cli.registry, &run_id)
        }
        CampaignCommand::ValidateTestAdaptiveExtension { run_id } => {
            test_adaptive::validate_extension(&cli.registry, &run_id)
        }
        CampaignCommand::ValidateTestAdaptiveIslandExtension { run_id } => {
            test_adaptive::validate_island_extension(&cli.registry, &run_id)
        }
        CampaignCommand::ValidateGolden42Champions { run_id } => {
            test_adaptive::validate_champion_continuation(&cli.registry, &run_id)
        }
        CampaignCommand::ValidateTabarenaLiteIslands { run_id } => {
            test_adaptive::validate_tabarena_lite_islands(&run_id)
        }
        CampaignCommand::BenchLanguagesV3 { output } => aggressive::bench_languages(&output),
        CampaignCommand::AdmitFittedBackendV1 { output } => {
            aggressive::write_fitted_backend_admission(&output)
        }
        CampaignCommand::AdmitFittedCppV3 { output } => golden42_panel::admit(&output),
        CampaignCommand::RunGolden42V3Max {
            run_id,
            dataset,
            node,
            workers,
            population,
            resume,
        } => {
            if cli.kernel_version != KernelVersion::AdvancedV3 {
                bail!("run-golden42-v3-max requires --kernel-version advanced-v3");
            }
            golden42_panel::run(
                &cli.registry,
                &run_id,
                &dataset,
                &node,
                workers,
                population,
                resume,
            )
        }
        CampaignCommand::ValidateGolden42V3Max { run_id } => golden42_panel::validate(&run_id),
        CampaignCommand::Golden42V3PersistentWorker { bootstrap } => {
            golden42_panel::persistent_worker(&bootstrap)
        }
        CampaignCommand::Golden42V3Confirm { request } => golden42_panel::confirm(&request),
        CampaignCommand::PlanGolden42Aggressive { output } => {
            aggressive::preflight_contract(&output)
        }
        CampaignCommand::RunGolden42Aggressive {
            run_id,
            dataset,
            start_index,
            resume,
        } => test_adaptive::run_aggressive(
            &cli.registry,
            &run_id,
            dataset.as_deref(),
            start_index,
            resume,
        ),
        CampaignCommand::ValidateGolden42Aggressive { run_id } => {
            test_adaptive::validate_aggressive(&cli.registry, &run_id)
        }
        CampaignCommand::RunGolden46 {
            run_id,
            dataset,
            transfer_arm,
            resume,
        } => test_adaptive::run_golden46(
            &cli.registry,
            &run_id,
            dataset.as_deref(),
            transfer_arm,
            resume,
        ),
        CampaignCommand::ValidateGolden46 {
            run_id,
            transfer_arm,
        } => test_adaptive::validate_golden46(&cli.registry, &run_id, transfer_arm),
        CampaignCommand::Golden42AggressiveBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::aggressive_batch_worker(&manifest, &output_dir),
        CampaignCommand::MaterializeScoreCorpus {
            archive_root,
            output_root,
            cohort,
        } => score_corpus::materialize(&archive_root, &output_root, cohort.as_deref()),
        CampaignCommand::ValidateRatchetArtifact { input } => {
            ratchet::validate_artifact_file(&input)
        }
        CampaignCommand::RatchetRetainModelSeeds {
            request,
            response,
            excluded_semantics,
            output,
        } => ratchet::retain_model_seeds_from_files(
            &request,
            &response,
            &excluded_semantics,
            &output,
        ),
        CampaignCommand::RatchetCommitDatasetState {
            root,
            state,
            committed_at_unix_seconds,
        } => ratchet::commit_dataset_state_from_file(&root, &state, committed_at_unix_seconds),
        CampaignCommand::RatchetPublishTrainingHead {
            root,
            manifest,
            micro_batch,
            replay,
        } => ratchet::publish_training_head_from_files(&root, &manifest, &micro_batch, &replay),
        CampaignCommand::RatchetPublishServingWave {
            root,
            manifest,
            rollout_switches,
            frozen_splits,
        } => ratchet::publish_serving_wave_from_files(
            &root,
            &manifest,
            &rollout_switches,
            &frozen_splits,
        ),
        CampaignCommand::RatchetAdmitAttempt {
            root,
            generation_zero,
            admission,
        } => ratchet::admit_attempt_from_files(&root, &generation_zero, &admission),
        CampaignCommand::GenomeBoostFeatures {
            manifest,
            output_dir,
        } => genome_boost::run_features(&manifest, &output_dir),
        CampaignCommand::ValidateGenomeBoostFeatures { output_dir } => {
            genome_boost::validate_features(&output_dir)
        }
        command if train_only_dispatch::accepts(&command) => train_only_dispatch::run(
            command,
            &cli.registry,
            cli.dataset.as_deref(),
            cli.kernel_version,
        ),
    }
}
