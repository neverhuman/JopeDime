use super::*;
use crate::cli::{CliTrainOnlyRedteamArm, CliTrainOnlyRedteamSelectionMode};

pub(super) fn accepts(command: &CampaignCommand) -> bool {
    matches!(
        command,
        CampaignCommand::PrepareTrainOnlyStudy { .. }
            | CampaignCommand::RunTrainOnlyStudy { .. }
            | CampaignCommand::ValidateTrainOnlyStudy { .. }
            | CampaignCommand::ScoreTrainOnlyStudy { .. }
            | CampaignCommand::ReportTrainOnlyStudy { .. }
            | CampaignCommand::PrepareTrainOnlyRedteam { .. }
            | CampaignCommand::RunTrainOnlyRedteam { .. }
            | CampaignCommand::ValidateTrainOnlyRedteam { .. }
            | CampaignCommand::ScoreTrainOnlyRedteam { .. }
            | CampaignCommand::ScoreSmartV3Redteam { .. }
            | CampaignCommand::ReportTrainOnlyRedteam { .. }
    )
}

pub(super) fn run(
    command: CampaignCommand,
    registry: &Path,
    dataset: Option<&str>,
    kernel_version: KernelVersion,
) -> Result<()> {
    match command {
        CampaignCommand::PrepareTrainOnlyStudy {
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
            if kernel_version != KernelVersion::LegacyV2 {
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
            let path = commands::train_only_study::prepare(registry, args)?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::RunTrainOnlyStudy {
            manifest,
            resume,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
                bail!("run-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_study::run(&manifest, resume, workers, dataset)?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::ValidateTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
                bail!("validate-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_study::validate(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
                dataset,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::ScoreTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
                bail!("score-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_study::score(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::ReportTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
                bail!("report-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_study::report(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::PrepareTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            parent_run,
            proposals,
            workers,
            smoke,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
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
        CampaignCommand::RunTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root: _,
            arm,
            selection_mode,
            resume,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
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
                dataset,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::ValidateTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
                bail!("validate-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::validate(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
                dataset,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::ScoreTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
                bail!("score-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_redteam::score(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        CampaignCommand::ScoreSmartV3Redteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
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
        CampaignCommand::ReportTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if kernel_version != KernelVersion::LegacyV2 {
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
        _ => unreachable!("only train-only commands are dispatched here"),
    }
}
