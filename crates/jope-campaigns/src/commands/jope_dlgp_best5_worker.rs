use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use jope_dime_regressor::jope_dlgp_best5::*;
use jope_dime_regressor::jope_generalized_gp::raw_sha256;
use jope_dime_regressor::paths;
use serde::de::DeserializeOwned;
use serde::Serialize;

trait ProtocolResultExt<T> {
    fn protocol(self) -> Result<T>;
}

impl<T> ProtocolResultExt<T> for std::result::Result<T, String> {
    fn protocol(self) -> Result<T> {
        self.map_err(anyhow::Error::msg)
    }
}

#[derive(Parser)]
#[command(name = "jope_dlgp_best5_worker")]
#[command(about = "SCQ-only executable worker for the isolated Best5 campaign")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    ProposeAll {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        artifact_root: PathBuf,
        #[arg(long)]
        work_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    SearchOne {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        proposal_batch: PathBuf,
        #[arg(long)]
        dataset_id: String,
        #[arg(long)]
        arm: ArmArg,
        #[arg(long)]
        master_seed: u64,
        #[arg(long)]
        artifact_root: PathBuf,
        #[arg(long)]
        work_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    SealedOne {
        #[arg(long)]
        campaign: PathBuf,
        /// JSON array containing all six pilot selections, or confirmation
        /// winner plus control, for exactly one dataset.
        #[arg(long)]
        selection_set: PathBuf,
        #[arg(long)]
        artifact_root: PathBuf,
        #[arg(long)]
        work_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    EmitPlan {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        launch_evidence: PathBuf,
        #[arg(long)]
        worker_binary: PathBuf,
        #[arg(long)]
        proposal_batch: Option<PathBuf>,
        #[arg(long)]
        selection_set_root: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        all_pilot_selection_freezes_exist: bool,
        #[arg(long)]
        pilot_winner: Option<ArmArg>,
        #[arg(long)]
        spec_root: PathBuf,
        #[arg(long, default_value_t = jope_paths::to_string(&paths::jope_home()))]
        workdir: String,
        #[arg(long, default_value = "scq")]
        scq_binary: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ArmArg {
    Lineage,
    Residual,
    Portfolio,
    Map,
    Surrogate,
    Control,
}

impl From<ArmArg> for Best5ArmV1 {
    fn from(value: ArmArg) -> Self {
        match value {
            ArmArg::Lineage => Self::EvolvabilityUcbLineages,
            ArmArg::Residual => Self::CrossFittedResidualDlgp,
            ArmArg::Portfolio => Self::SparseOofPortfolioGp,
            ArmArg::Map => Self::HeadFamilyMapElites,
            ArmArg::Surrogate => Self::SurrogateUcbOffspringSearch,
            ArmArg::Control => Self::ConfidenceGpControl,
        }
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("decode {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes =
        jope_dime_regressor::search_v3::canonical_json_bytes(value).map_err(anyhow::Error::msg)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(&bytes)?;
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let present = fs::read(path)?;
            if present == bytes {
                Ok(())
            } else {
                anyhow::bail!("immutable worker output collision at {}", path.display())
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn scq_input(name: &str, source: &Path, target: &str, executable: bool) -> Result<ScqInputV1> {
    let bytes = fs::read(source).with_context(|| format!("read SCQ input {}", source.display()))?;
    Ok(ScqInputV1 {
        name: name.into(),
        source: source.to_string_lossy().into_owned(),
        target: target.into(),
        sha256: raw_sha256(&bytes),
        executable,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::ProposeAll {
            campaign,
            artifact_root,
            work_root,
            output,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let batch = propose_all(&campaign, &artifact_root, &work_root).protocol()?;
            write_json(&output, &batch)?;
            println!("{}", batch.identity(&campaign).protocol()?);
        }
        Command::SearchOne {
            campaign,
            proposal_batch,
            dataset_id,
            arm,
            master_seed,
            artifact_root,
            work_root,
            output,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let proposal_batch: ProposalBatchFreezeV1 = read_json(&proposal_batch)?;
            let bundle = run_search_one(
                &campaign,
                &proposal_batch,
                &dataset_id,
                arm.into(),
                master_seed,
                &artifact_root,
                &work_root,
            )
            .protocol()?;
            write_json(&output, &bundle)?;
            println!("{}", bundle.identity().protocol()?);
        }
        Command::SealedOne {
            campaign,
            selection_set,
            artifact_root,
            work_root,
            output,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let selections: Vec<SelectionFreezeV1> = read_json(&selection_set)?;
            let bundle =
                run_sealed_one(&campaign, selections, &artifact_root, &work_root).protocol()?;
            write_json(&output, &bundle)?;
            println!("{}", bundle.identity().protocol()?);
        }
        Command::EmitPlan {
            campaign,
            launch_evidence,
            worker_binary,
            proposal_batch,
            selection_set_root,
            all_pilot_selection_freezes_exist,
            pilot_winner,
            spec_root,
            workdir,
            scq_binary,
            output,
        } => {
            let campaign_value: CampaignFreezeV1 = read_json(&campaign)?;
            let evidence: LaunchEvidenceV1 = read_json(&launch_evidence)?;
            verify_campaign_artifacts(&campaign_value).protocol()?;
            if all_pilot_selection_freezes_exist {
                let root = selection_set_root.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--all-pilot-selection-freezes-exist requires --selection-set-root"
                    )
                })?;
                let mut selections = Vec::new();
                for dataset in campaign_value
                    .datasets
                    .iter()
                    .filter(|dataset| dataset.stage == DatasetStageV1::Pilot)
                {
                    selections.extend(read_json::<Vec<SelectionFreezeV1>>(
                        &root.join(format!("{}.json", dataset.dataset_id)),
                    )?);
                }
                validate_pilot_selection_coverage(&campaign_value, &selections).protocol()?;
            }
            let plan = build_scq_plan(
                &campaign_value,
                &evidence,
                all_pilot_selection_freezes_exist,
                pilot_winner.map(Into::into),
            )
            .protocol()?;
            let worker_input = scq_input("worker", &worker_binary, "worker", true)?;
            let campaign_input = scq_input("campaign", &campaign, "campaign.json", false)?;
            let proposal_input = proposal_batch
                .as_deref()
                .map(|path| scq_input("proposal_batch", path, "proposal-batch.json", false))
                .transpose()?;
            emit_and_check_plan(
                &plan,
                worker_input,
                campaign_input,
                &spec_root,
                &workdir,
                &scq_binary,
                |job| {
                    let mut values = Vec::new();
                    let frozen = |name: &str,
                                  artifact: &FrozenFileV1,
                                  target: &str,
                                  executable: bool|
                     -> std::result::Result<ScqInputV1, String> {
                        let input = scq_input(name, Path::new(&artifact.path), target, executable)
                            .map_err(|error| error.to_string())?;
                        if input.sha256 != artifact.sha256 {
                            return Err(format!("SCQ input {name} changed after freeze"));
                        }
                        Ok(input)
                    };
                    if job.stage == ScqStageV1::ProposalAll42 {
                        values.push(frozen(
                            "proposal_runtime",
                            &campaign_value.proposal_runtime_artifact,
                            "proposal-runtime",
                            true,
                        )?);
                        values.push(frozen(
                            "proposal_generator",
                            &campaign_value.proposal_generator_artifact,
                            "proposal-generator",
                            true,
                        )?);
                        values.push(
                            scq_input(
                                "checkpoint",
                                Path::new(&campaign_value.checkpoint.artifact_path),
                                "checkpoint.bin",
                                false,
                            )
                            .map_err(|error| error.to_string())?,
                        );
                        for dataset in &campaign_value.datasets {
                            values.push(
                                scq_input(
                                    &format!("train_{}", dataset.dataset_id),
                                    &Path::new(&campaign_value.sotarena_root)
                                        .join(&dataset.train_relative_path),
                                    &format!("catalog/{}/train.csv", dataset.dataset_id),
                                    false,
                                )
                                .map_err(|error| error.to_string())?,
                            );
                            values.push(
                                scq_input(
                                    &format!("metadata_{}", dataset.dataset_id),
                                    &Path::new(&campaign_value.sotarena_root)
                                        .join(&dataset.metadata_relative_path),
                                    &format!("catalog/{}/metadata.json", dataset.dataset_id),
                                    false,
                                )
                                .map_err(|error| error.to_string())?,
                            );
                        }
                    }
                    if matches!(
                        job.stage,
                        ScqStageV1::ControlProof
                            | ScqStageV1::ProtectedPilot
                            | ScqStageV1::RemainingPilot
                            | ScqStageV1::ConfirmationSearch
                    ) {
                        values.push(
                            proposal_input.clone().ok_or_else(|| {
                                "search specs require --proposal-batch".to_string()
                            })?,
                        );
                        values.push(frozen(
                            "evaluator",
                            &campaign_value.evaluator_artifact,
                            "evaluator",
                            true,
                        )?);
                        values.push(frozen(
                            "math_policy",
                            &campaign_value.math_policy_artifact,
                            "math-policy.json",
                            false,
                        )?);
                        values.push(frozen(
                            "backend_policy",
                            &campaign_value.backend_policy_artifact,
                            "backend-policy.json",
                            false,
                        )?);
                        values.push(frozen(
                            "observation_snapshot",
                            &campaign_value.observation_snapshot_artifact,
                            "observation-snapshot.json",
                            false,
                        )?);
                        let id = job
                            .dataset_id
                            .as_deref()
                            .ok_or_else(|| "search job lacks dataset ID".to_string())?;
                        let dataset = campaign_value
                            .datasets
                            .iter()
                            .find(|dataset| dataset.dataset_id == id)
                            .ok_or_else(|| "search job dataset is not frozen".to_string())?;
                        values.push(
                            scq_input(
                                "training",
                                &Path::new(&campaign_value.sotarena_root)
                                    .join(&dataset.train_relative_path),
                                "dataset/train.csv",
                                false,
                            )
                            .map_err(|error| error.to_string())?,
                        );
                        values.push(
                            scq_input(
                                "metadata",
                                &Path::new(&campaign_value.sotarena_root)
                                    .join(&dataset.metadata_relative_path),
                                "dataset/metadata.json",
                                false,
                            )
                            .map_err(|error| error.to_string())?,
                        );
                    }
                    if matches!(
                        job.stage,
                        ScqStageV1::PilotSealedScore | ScqStageV1::ConfirmationSealedScore
                    ) {
                        let root = selection_set_root.as_deref().ok_or_else(|| {
                            "sealed specs require --selection-set-root".to_string()
                        })?;
                        let dataset = job
                            .dataset_id
                            .as_deref()
                            .ok_or_else(|| "sealed job lacks dataset ID".to_string())?;
                        values.push(
                            scq_input(
                                "selection_set",
                                &root.join(format!("{dataset}.json")),
                                "selection-set.json",
                                false,
                            )
                            .map_err(|error| error.to_string())?,
                        );
                        values.push(frozen(
                            "evaluator",
                            &campaign_value.evaluator_artifact,
                            "evaluator",
                            true,
                        )?);
                        values.push(frozen(
                            "math_policy",
                            &campaign_value.math_policy_artifact,
                            "math-policy.json",
                            false,
                        )?);
                        values.push(frozen(
                            "backend_policy",
                            &campaign_value.backend_policy_artifact,
                            "backend-policy.json",
                            false,
                        )?);
                        let frozen_dataset = campaign_value
                            .datasets
                            .iter()
                            .find(|frozen| frozen.dataset_id == dataset)
                            .ok_or_else(|| "sealed job dataset is not frozen".to_string())?;
                        values.push(
                            scq_input(
                                "training",
                                &Path::new(&campaign_value.sotarena_root)
                                    .join(&frozen_dataset.train_relative_path),
                                "dataset/train.csv",
                                false,
                            )
                            .map_err(|error| error.to_string())?,
                        );
                        values.push(
                            scq_input(
                                "test",
                                &Path::new(&campaign_value.sotarena_root)
                                    .join(&frozen_dataset.test_relative_path),
                                "dataset/test.csv",
                                false,
                            )
                            .map_err(|error| error.to_string())?,
                        );
                    }
                    Ok(values)
                },
            )
            .protocol()?;
            write_json(&output, &plan)?;
            println!("{}", plan.identity().protocol()?);
        }
    }
    Ok(())
}
