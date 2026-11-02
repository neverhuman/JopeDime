use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::commands::exact_cv::{self, ValidatedFreeze, ORIGINAL_GENERATIONS};
use crate::core::require_core_plan;
use crate::data::{read_binary_csv, read_regression_csv};
use jope_dime_regressor::paths;

mod aggressive;
mod args;
mod commands;
mod core;
mod data;
// frozen, hash-pinned module; bytes immutable until unfrozen
mod binary_panel;
#[allow(dead_code, clippy::too_many_arguments, clippy::let_and_return)]
mod campaign_evaluator;
mod cli;
mod followup;
mod genome_boost;
mod golden42_panel;
mod ratchet;
mod regression_panel;
mod runtime;
mod score_corpus;
mod test_adaptive;

use binary_panel::*;
use cli::{Cli, Command, KernelVersion};
use regression_panel::*;

jope_paths::once_str!(
    fn registry_str => paths::jope_registry()
);
const PANEL_SHA256: &str = "b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75";
const PANEL_ANALYSIS_SHA256: &str =
    "0117abb32d547b134706e210376f2e6f437b706b2a12edb6d9302d74d4749282";
const SEED_SHA256: &str = "5f097a0017ddc7e92ad5ea8dfadaf783b548e82287d92626dc9d2fa9429b2105";
const BASE_SEED: u64 = 20260616;
// --- binary-classification lane (v2-binary; fixture-backed until real datasets land) ---
const PANEL_BINARY_SHA256: &str =
    "be3e8ff9654e31bbb1015a1e783bb9ba58f86295706dbdbaf137509e0b69d575";
const PANEL_ANALYSIS_BINARY_SHA256: &str =
    "34f476c21e055d58033eb9278247d5ddfdf9c7c65e4660f4e7372f8345b26da6";
const MODEL_BINARY: &str = "jope-dime-binary";
const BINARY_REPORT_FILE: &str = "jope-dime-binary-report.v1.json";
const BINARY_PREDICTION_FILE: &str = "jope-dime-binary-predictions.csv";
const BINARY_RESULTS_FILE: &str = "jope-dime-binary-panel-results.v1.json";
/// Lane tag ("jpbin_v1" as bytes) XORed into the seed derivation so binary
/// campaigns never share offspring streams with the regression lane.
const BINARY_LANE_TAG: u64 = 0x6a70_6269_6e5f_7631;
const MODEL: &str = "jope-dime";
const REPORT_FILE: &str = "jope-dime-regressor.json";
const PREDICTION_FILE: &str = "jope-dime-regressor.csv";
const BASELINES: &[&str] = &[
    "TabFM ensemble",
    "TabPFNv3",
    "Fable5",
    "Grok4p5",
    "OpenAI GPT-5.6-sol",
    "Kimik3",
];
const DATASETS: &[&str] = &[
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1a803b27c2400dc4",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
];

#[derive(Debug)]
struct ScoredDataset {
    dataset_hash: String,
    predictions: Vec<f64>,
    prediction_bytes: Vec<u8>,
    prediction_sha256: String,
    report_bytes: Vec<u8>,
    report_sha256: String,
    scores: BTreeMap<String, f64>,
    outcome: Value,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.kernel_version == KernelVersion::AdvancedV3 && command_starts_campaign(&cli.command) {
        bail!(
            "advanced-v3 campaign launch is admission-gated; use a fresh run only after the v3 benchmark gate has been recorded (existing manifests remain legacy-v2)"
        );
    }
    match cli.command {
        Command::VerifyInputs => verify_inputs(&cli.registry),
        Command::SelectPanel => select_panel(&cli.registry, &cli.panel_id, cli.dataset.as_deref()),
        Command::ValidatePanel => validate_panel(&cli.registry, &cli.panel_id),
        Command::ScorePanel => score_panel(&cli.registry, &cli.panel_id),
        Command::PrepareTrainOnlyStudy {
            run_id,
            study_root,
            checkpoint_pointer,
            checkpoint_provenance,
            catalog_pointer,
            observation_db,
            dl_proposals,
            proposal_generator,
            python,
            proposal_device,
            sotarena_root,
            workers,
            smoke,
            proposal_manifest_only,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("prepare-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let mut args =
                commands::train_only_study::PrepareArgs::production(run_id, Some(study_root));
            args.checkpoint_pointer = checkpoint_pointer;
            args.checkpoint_provenance = checkpoint_provenance;
            args.catalog_pointer = catalog_pointer;
            args.observation_db = observation_db;
            args.dl_proposals = dl_proposals;
            args.proposal_generator = proposal_generator;
            args.python = python;
            args.proposal_device = proposal_device;
            args.sotarena_root = sotarena_root;
            args.workers = workers;
            args.smoke = smoke;
            args.proposal_manifest_only = proposal_manifest_only;
            let path = commands::train_only_study::prepare(&cli.registry, args)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::RunTrainOnlyStudy {
            manifest,
            resume,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("run-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_study::run(
                &manifest,
                resume,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ValidateTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("validate-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_study::validate(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ScoreTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("score-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_study::score(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ReportTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("report-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_study::report(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::PrepareTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            parent_run,
            proposals,
            workers,
            smoke,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("prepare-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let args = commands::train_only_redteam::PrepareArgs::production(
                run_id,
                study_root,
                sotarena_root,
                parent_run,
                proposals,
                workers,
                smoke,
            );
            let path = commands::train_only_redteam::prepare(args)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::RunTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root: _,
            arm,
            selection_mode,
            resume,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("run-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let selection_mode = match selection_mode {
                CliTrainOnlyRedteamSelectionMode::Combined => {
                    commands::train_only_redteam::SelectionMode::Combined
                }
                CliTrainOnlyRedteamSelectionMode::SmartOnly => {
                    commands::train_only_redteam::SelectionMode::SmartOnly
                }
            };
            let arm = match (selection_mode, arm) {
                (
                    commands::train_only_redteam::SelectionMode::Combined,
                    Some(CliTrainOnlyRedteamArm::G20Extension),
                ) => commands::train_only_redteam::Arm::G20Extension,
                (
                    commands::train_only_redteam::SelectionMode::Combined,
                    Some(CliTrainOnlyRedteamArm::SmartV3),
                )
                | (
                    commands::train_only_redteam::SelectionMode::SmartOnly,
                    None | Some(CliTrainOnlyRedteamArm::SmartV3),
                ) => commands::train_only_redteam::Arm::SmartV3,
                (commands::train_only_redteam::SelectionMode::Combined, None) => {
                    bail!("combined selection mode requires --arm")
                }
                (
                    commands::train_only_redteam::SelectionMode::SmartOnly,
                    Some(CliTrainOnlyRedteamArm::G20Extension),
                ) => bail!("smart-only selection mode runs only the Smart-v3 arm"),
            };
            let manifest = study_root.join(run_id).join("manifest.redteam.v1.json");
            let path = commands::train_only_redteam::run(
                &manifest,
                arm,
                selection_mode,
                resume,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ValidateTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("validate-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::validate(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ScoreTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("score-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_redteam::score(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ScoreSmartV3Redteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("score-smart-v3-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::score_smart_v3(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ReportTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("report-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::report(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::VerifyEvaluationV2 { manifest } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("verify-evaluation-v2 is pinned to the legacy-v2 evaluator");
            }
            commands::single_fold_proof::verify_manifest(
                &manifest,
                &cli.registry,
                cli.dataset.as_deref(),
            )
        }
        Command::VerifyInputsBinary => verify_inputs_binary(&cli.registry),
        Command::SelectPanelBinary => {
            select_panel_binary(&cli.registry, &cli.panel_id, cli.dataset.as_deref())
        }
        Command::ValidatePanelBinary => validate_panel_binary(&cli.registry, &cli.panel_id),
        Command::ScorePanelBinary => score_panel_binary(&cli.registry, &cli.panel_id),
        Command::RunBinaryTestAdaptive { run_id, resume } => {
            let dataset = cli
                .dataset
                .as_deref()
                .context("run-binary-test-adaptive requires --dataset")?;
            test_adaptive::run_binary42(&run_id, dataset, resume)
        }
        Command::ValidateBinaryTestAdaptive { run_id } => {
            let dataset = cli
                .dataset
                .as_deref()
                .context("validate-binary-test-adaptive requires --dataset")?;
            test_adaptive::validate_binary42(&run_id, dataset)
        }
        Command::RunBinaryTestAdaptiveAll { run_id, resume } => {
            if cli.dataset.is_some() {
                bail!("run-binary-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_binary42_all(&run_id, resume)
        }
        Command::ValidateBinaryTestAdaptiveAll { run_id } => {
            if cli.dataset.is_some() {
                bail!("validate-binary-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_binary42_all(&run_id)
        }
        Command::PlanMulticlassTestAdaptive {
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
        Command::RunMulticlassTestAdaptiveAll {
            run_id,
            campaign_root,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-multiclass-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_multiclass60_all(&run_id, &campaign_root, resume)
        }
        Command::ValidateMulticlassTestAdaptiveAll {
            run_id,
            campaign_root,
        } => {
            if cli.dataset.is_some() {
                bail!("validate-multiclass-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_multiclass60_all(&run_id, &campaign_root)
        }
        Command::RunTimeseriesTestAdaptiveAll { run_id, resume } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_timeseries42_all(&run_id, resume)
        }
        Command::ValidateTimeseriesTestAdaptiveAll { run_id } => {
            if cli.dataset.is_some() {
                bail!("validate-timeseries-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_timeseries42_all(&run_id)
        }
        Command::RunTimeseriesTestAdaptiveGen30 {
            base_run_id,
            run_id,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-gen30 does not accept --dataset");
            }
            test_adaptive::run_timeseries42_gen30(&base_run_id, &run_id, resume)
        }
        Command::ValidateTimeseriesTestAdaptiveGen30 {
            base_run_id,
            run_id,
        } => {
            if cli.dataset.is_some() {
                bail!("validate-timeseries-test-adaptive-gen30 does not accept --dataset");
            }
            test_adaptive::validate_timeseries42_gen30(&base_run_id, &run_id)
        }
        Command::RunTimeseriesTestAdaptiveWinall {
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
        Command::BinaryTestAdaptiveBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::binary42_batch_worker(&manifest, &output_dir),
        Command::TimeseriesTestAdaptiveBatchWorker {
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
        Command::RunPanel => {
            select_panel(&cli.registry, &cli.panel_id, cli.dataset.as_deref())?;
            score_panel(&cli.registry, &cli.panel_id)
        }
        Command::Summarize => summarize(&cli.registry, &cli.panel_id),
        Command::RunLosses3x => followup::run(&cli.registry),
        Command::ValidateLosses3x => followup::validate(&cli.registry),
        Command::RunTestAdaptive { run_id, resume } => {
            test_adaptive::run(&cli.registry, &run_id, resume)
        }
        Command::RunTestAdaptivePanel { run_id, resume } => {
            test_adaptive::run_panel(&cli.registry, &run_id, resume)
        }
        Command::RunTestAdaptiveExtension { run_id, resume } => {
            test_adaptive::run_extension(&cli.registry, &run_id, resume)
        }
        Command::RunTestAdaptiveIslandExtension { run_id, resume } => {
            test_adaptive::run_island_extension(&cli.registry, &run_id, resume)
        }
        Command::RunGolden42Champions { run_id, resume } => {
            test_adaptive::run_champion_continuation(&cli.registry, &run_id, resume)
        }
        Command::RunTabarenaLiteIslands { run_id, resume } => {
            test_adaptive::run_tabarena_lite_islands(&run_id, resume, cli.dataset.as_deref())
        }
        Command::RecoverTabarenaLiteResult { run_id, dataset } => {
            test_adaptive::recover_tabarena_lite_result(&run_id, &dataset)
        }
        Command::TestAdaptiveWorker { request } => test_adaptive::worker(&request),
        Command::TestAdaptiveIslandBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::island_batch_worker(&manifest, &output_dir),
        Command::ValidateTestAdaptive { run_id } => test_adaptive::validate(&cli.registry, &run_id),
        Command::ValidateTestAdaptivePanel { run_id } => {
            test_adaptive::validate_panel(&cli.registry, &run_id)
        }
        Command::ValidateTestAdaptiveExtension { run_id } => {
            test_adaptive::validate_extension(&cli.registry, &run_id)
        }
        Command::ValidateTestAdaptiveIslandExtension { run_id } => {
            test_adaptive::validate_island_extension(&cli.registry, &run_id)
        }
        Command::ValidateGolden42Champions { run_id } => {
            test_adaptive::validate_champion_continuation(&cli.registry, &run_id)
        }
        Command::ValidateTabarenaLiteIslands { run_id } => {
            test_adaptive::validate_tabarena_lite_islands(&run_id)
        }
        Command::BenchLanguagesV3 { output } => aggressive::bench_languages(&output),
        Command::AdmitFittedBackendV1 { output } => {
            aggressive::write_fitted_backend_admission(&output)
        }
        Command::AdmitFittedCppV3 { output } => golden42_panel::admit(&output),
        Command::RunGolden42V3Max {
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
        Command::ValidateGolden42V3Max { run_id } => golden42_panel::validate(&run_id),
        Command::Golden42V3PersistentWorker { bootstrap } => {
            golden42_panel::persistent_worker(&bootstrap)
        }
        Command::Golden42V3Confirm { request } => golden42_panel::confirm(&request),
        Command::PlanGolden42Aggressive { output } => aggressive::preflight_contract(&output),
        Command::RunGolden42Aggressive {
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
        Command::ValidateGolden42Aggressive { run_id } => {
            test_adaptive::validate_aggressive(&cli.registry, &run_id)
        }
        Command::RunGolden46 {
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
        Command::ValidateGolden46 {
            run_id,
            transfer_arm,
        } => test_adaptive::validate_golden46(&cli.registry, &run_id, transfer_arm),
        Command::Golden42AggressiveBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::aggressive_batch_worker(&manifest, &output_dir),
        Command::MaterializeScoreCorpus {
            archive_root,
            output_root,
            cohort,
        } => score_corpus::materialize(&archive_root, &output_root, cohort.as_deref()),
        Command::ValidateRatchetArtifact { input } => ratchet::validate_artifact_file(&input),
        Command::RatchetRetainModelSeeds {
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
        Command::RatchetCommitDatasetState {
            root,
            state,
            committed_at_unix_seconds,
        } => ratchet::commit_dataset_state_from_file(&root, &state, committed_at_unix_seconds),
        Command::RatchetPublishTrainingHead {
            root,
            manifest,
            micro_batch,
            replay,
        } => ratchet::publish_training_head_from_files(&root, &manifest, &micro_batch, &replay),
        Command::RatchetPublishServingWave {
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
        Command::RatchetAdmitAttempt {
            root,
            generation_zero,
            admission,
        } => ratchet::admit_attempt_from_files(&root, &generation_zero, &admission),
        Command::GenomeBoostFeatures {
            manifest,
            output_dir,
        } => genome_boost::run_features(&manifest, &output_dir),
        Command::ValidateGenomeBoostFeatures { output_dir } => {
            genome_boost::validate_features(&output_dir)
        }
    }
}

fn command_starts_campaign(command: &Command) -> bool {
    matches!(
        command,
        Command::SelectPanel
            | Command::RunPanel
            | Command::PrepareTrainOnlyStudy { .. }
            | Command::RunTrainOnlyStudy { .. }
            | Command::PrepareTrainOnlyRedteam { .. }
            | Command::RunTrainOnlyRedteam { .. }
            | Command::RunLosses3x
            | Command::RunTimeseriesTestAdaptiveAll { .. }
            | Command::RunTimeseriesTestAdaptiveGen30 { .. }
            | Command::TimeseriesTestAdaptiveBatchWorker { .. }
            | Command::RunTestAdaptive { .. }
            | Command::RunTestAdaptivePanel { .. }
            | Command::RunTestAdaptiveExtension { .. }
            | Command::RunTestAdaptiveIslandExtension { .. }
            | Command::RunGolden42Champions { .. }
            | Command::RunTabarenaLiteIslands { .. }
            | Command::TestAdaptiveWorker { .. }
            | Command::TestAdaptiveIslandBatchWorker { .. }
            | Command::RunGolden42Aggressive { .. }
            | Command::RunGolden46 { .. }
            | Command::Golden42AggressiveBatchWorker { .. }
    )
}

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("manifests")
}

fn run_root(panel_id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(panel_id)
}

fn seed_bytes() -> &'static [u8] {
    include_bytes!("../seeds/generation-zero.txt")
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn dataset_seed(hash: &str) -> Result<u64> {
    Ok(BASE_SEED ^ u64::from_str_radix(hash, 16).context("dataset hash is not hexadecimal")?)
}

/// Binary-lane seed derivation: BASE_SEED ^ lane tag ^ dataset hash, so the
/// binary offspring streams never coincide with the regression lane's.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn binary_elo_is_deterministic_and_rewards_strictly_greater_auc() {
        let mut dataset = BTreeMap::new();
        dataset.insert("A".to_string(), 0.9);
        dataset.insert("B".to_string(), 0.7);
        let ratings = binary_elo(&[dataset.clone()]);
        // One game: A expected 0.5, wins -> +8; B symmetric -> -8.
        assert_eq!(ratings["A"], 1508.0);
        assert_eq!(ratings["B"], 1492.0);
        // Exact tie moves nothing at equal ratings.
        let mut tie = BTreeMap::new();
        tie.insert("A".to_string(), 0.8);
        tie.insert("B".to_string(), 0.8);
        let tie_ratings = binary_elo(&[tie]);
        assert_eq!(tie_ratings["A"], 1500.0);
        assert_eq!(tie_ratings["B"], 1500.0);
        // Determinism across repeated evaluation.
        assert_eq!(binary_elo(&[dataset.clone()]), binary_elo(&[dataset]));
    }

    #[test]
    fn binary_static_inputs_are_pinned() {
        assert_eq!(
            sha256_bytes(binary_seed_bytes()),
            commands::binary_cv::SEED_BINARY_SHA256
        );
        let panel = binary_panel().unwrap();
        assert_eq!(panel, vec!["b1a0f1c500000001", "b1a0f1c500000002"]);
        let baselines = binary_baselines().unwrap();
        assert_eq!(baselines.len(), 2);
        assert!(baselines["b1a0f1c500000001"]["BaselineBoosted"] > 0.8);
    }

    #[test]
    fn strict_relation_has_no_tolerance() {
        assert_eq!(relation(0.9, 1.0), "win");
        assert_eq!(relation(1.0, 1.0), "tie");
        assert_eq!(relation(1.0 + f64::EPSILON, 1.0), "loss");
    }

    #[test]
    fn baseline_comparison_covers_all_six_models() {
        let baselines = BASELINES
            .iter()
            .enumerate()
            .map(|(index, model)| ((*model).to_string(), 1.0 + index as f64))
            .collect::<BTreeMap<_, _>>();
        let (comparisons, best, models, overall, tabfm) =
            baseline_comparisons(0.5, &baselines).unwrap();
        assert_eq!(comparisons.as_object().unwrap().len(), 6);
        assert_eq!(best, 1.0);
        assert_eq!(models, vec!["TabFM ensemble"]);
        assert_eq!(overall, "win");
        assert_eq!(tabfm, "win");
    }

    #[test]
    fn immutable_publication_refuses_overwrite() {
        let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "jope-dime-publication-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("report.json");
        publish_new(&path, b"first\n").unwrap();
        assert!(publish_new(&path, b"second\n").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"first\n");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn timeseries_gen30_cli_parses_base_and_run_ids() {
        let cli = Cli::try_parse_from([
            "jope-dime-regressor",
            "run-timeseries-test-adaptive-gen30",
            "--base-run-id",
            "base",
            "--run-id",
            "next",
            "--resume",
        ])
        .unwrap();
        match cli.command {
            Command::RunTimeseriesTestAdaptiveGen30 {
                base_run_id,
                run_id,
                resume,
            } => {
                assert_eq!(base_run_id, "base");
                assert_eq!(run_id, "next");
                assert!(resume);
            }
            _ => panic!("wrong command variant"),
        }
    }

    #[test]
    fn timeseries_winall_cli_parses_source_run_id_and_generation_limit() {
        let cli = Cli::try_parse_from([
            "jope-dime-regressor",
            "run-timeseries-test-adaptive-winall",
            "--source-run-id",
            "gen30",
            "--run-id",
            "winall",
            "--max-generations",
            "200",
            "--lineage-salt",
            "salt-a",
            "--resume",
        ])
        .unwrap();
        match cli.command {
            Command::RunTimeseriesTestAdaptiveWinall {
                source_run_id,
                run_id,
                max_generations,
                lineage_salt,
                resume,
            } => {
                assert_eq!(source_run_id, "gen30");
                assert_eq!(run_id, "winall");
                assert_eq!(max_generations, Some(200));
                assert_eq!(lineage_salt.as_deref(), Some("salt-a"));
                assert!(resume);
            }
            _ => panic!("wrong command variant"),
        }
    }
}
