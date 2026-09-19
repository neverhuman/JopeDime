use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use crate::jope_dlgp_best5::*;
use crate::jope_generalized_gp::{raw_sha256, ArtifactStoreV1};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;

trait ProtocolResultExt<T> {
    fn protocol(self) -> Result<T>;
}

impl<T> ProtocolResultExt<T> for std::result::Result<T, String> {
    fn protocol(self) -> Result<T> {
        self.map_err(anyhow::Error::msg)
    }
}

#[derive(Parser)]
#[command(name = "jope_dlgp_best5")]
#[command(about = "Isolated best-five DL-seeded GP regression campaign")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify and content-address the immutable campaign freeze.
    Freeze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output_root: PathBuf,
    },
    /// Validate the freeze, exact checkpoint bytes, and regression schema bind.
    ValidateFreeze {
        #[arg(long)]
        campaign: PathBuf,
    },
    /// Validate and publish one evaluator-produced dataset/arm/seed search result.
    RunSearch {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        worker_result: PathBuf,
        #[arg(long)]
        output_root: PathBuf,
    },
    /// Select between the immutable seed-101 and seed-202 search results.
    SelectSeeds {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long, required = true, num_args = 2)]
        search_result: Vec<PathBuf>,
        #[arg(long)]
        output_root: PathBuf,
    },
    /// Publish one sealed score only after binding it to a selection freeze.
    ScoreSealed {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        score: PathBuf,
        #[arg(long)]
        output_root: PathBuf,
    },
    /// Apply the frozen paired bootstrap, gates, and tie-breaks.
    Compare {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        stage: StageArg,
        #[arg(long)]
        scores: PathBuf,
        #[arg(long)]
        output_root: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum StageArg {
    Pilot,
    Confirmation,
}

impl From<StageArg> for DatasetStageV1 {
    fn from(value: StageArg) -> Self {
        match value {
            StageArg::Pilot => Self::Pilot,
            StageArg::Confirmation => Self::Confirmation,
        }
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("decode {}", path.display()))
}

fn publish<T: Serialize>(
    output_root: &Path,
    identity: &str,
    kind: &str,
    value: &T,
) -> Result<String> {
    let store = ArtifactStoreV1::new(output_root);
    let object_sha256 = store.publish(identity, value).protocol()?;
    let receipt = json!({
        "schema": "JopeDlgpBest5ArtifactReceipt.v1",
        "kind": kind,
        "logical_identity_sha256": identity,
        "object_sha256": object_sha256,
    });
    let receipt_identity = raw_sha256(format!("{kind}|{identity}").as_bytes());
    store
        .append_receipt(&receipt_identity, &receipt)
        .protocol()?;
    Ok(object_sha256)
}

fn fold_sd(values: &[f64]) -> f64 {
    let center = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    match args.command {
        Command::Freeze { input, output_root } => {
            let source: CampaignSourceV1 = read_json(&input)?;
            let campaign = freeze_campaign(&source).protocol()?;
            let identity = campaign.identity().protocol()?;
            publish(&output_root, &identity, "campaign_freeze", &campaign)?;
            println!("{identity}");
        }
        Command::ValidateFreeze { campaign } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            verify_campaign_artifacts(&campaign).protocol()?;
            println!("{}", campaign.identity().protocol()?);
        }
        Command::RunSearch {
            campaign,
            worker_result,
            output_root,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let mut bundle: SearchBundleV1 = read_json(&worker_result)?;
            campaign.validate().protocol()?;
            bundle.normalize_completion_order();
            bundle.validate().protocol()?;
            let store = ArtifactStoreV1::new(&output_root);
            if !store
                .contains(&bundle.staging.proposal_batch_sha256)
                .protocol()?
            {
                bail!("search bundle references an unpublished proposal batch");
            }
            let proposal_batch: ProposalBatchFreezeV1 = store
                .load(&bundle.staging.proposal_batch_sha256)
                .protocol()?;
            verify_search_bundle(&campaign, &proposal_batch, &bundle, &output_root).protocol()?;
            let result = &bundle.result;
            if result.campaign_freeze_sha256 != campaign.identity().protocol()?
                || !campaign.budgets.contains_key(&result.arm)
                || !campaign
                    .datasets
                    .iter()
                    .any(|dataset| dataset.dataset_sha256 == result.dataset_sha256)
            {
                bail!("worker search result does not bind the supplied campaign");
            }
            let identity = result.identity().protocol()?;
            publish(&output_root, &identity, "search_result", result)?;
            println!("{identity}");
        }
        Command::SelectSeeds {
            campaign,
            search_result,
            output_root,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            campaign.validate().protocol()?;
            let results = search_result
                .iter()
                .map(|path| read_json::<SearchResultV1>(path))
                .collect::<Result<Vec<_>>>()?;
            for result in &results {
                result.validate().protocol()?;
            }
            if results[0].campaign_freeze_sha256 != campaign.identity().protocol()?
                || results[1].campaign_freeze_sha256 != results[0].campaign_freeze_sha256
                || results[1].arm != results[0].arm
                || results[1].dataset_sha256 != results[0].dataset_sha256
            {
                bail!("seed results do not describe one matched campaign/arm/dataset");
            }
            let mut choices = results
                .iter()
                .map(|result| {
                    let winner = result
                        .finalists
                        .iter()
                        .find(|candidate| {
                            candidate.semantic_sha256 == result.winner_semantic_sha256
                        })
                        .unwrap();
                    Ok(SeedChoiceV1 {
                        master_seed: result.master_seed,
                        search_result_sha256: result.identity().protocol()?,
                        winner_semantic_sha256: result.winner_semantic_sha256.clone(),
                        winner_predictor: winner.predictor.clone(),
                        pooled_oof_r2: winner.pooled_oof_r2,
                        fold_r2_sd: fold_sd(&winner.fold_r2),
                        rmse: winner.rmse,
                        runtime_seconds: winner.runtime_seconds,
                        complexity: winner.complexity,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            choices.sort_by_key(|choice| choice.master_seed);
            let best = choices
                .iter()
                .min_by(|left, right| {
                    right
                        .robust_score()
                        .total_cmp(&left.robust_score())
                        .then_with(|| left.rmse.total_cmp(&right.rmse))
                        .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
                        .then_with(|| left.complexity.cmp(&right.complexity))
                        .then_with(|| {
                            left.winner_semantic_sha256
                                .cmp(&right.winner_semantic_sha256)
                        })
                })
                .unwrap();
            let selection = SelectionFreezeV1 {
                schema: SelectionFreezeV1::SCHEMA.into(),
                campaign_freeze_sha256: results[0].campaign_freeze_sha256.clone(),
                arm: results[0].arm,
                dataset_sha256: results[0].dataset_sha256.clone(),
                choices: choices.clone(),
                selected_seed: best.master_seed,
                winner_semantic_sha256: best.winner_semantic_sha256.clone(),
                winner_predictor: best.winner_predictor.clone(),
            };
            selection.validate().protocol()?;
            let identity = selection.identity().protocol()?;
            publish(&output_root, &identity, "selection_freeze", &selection)?;
            println!("{identity}");
        }
        Command::ScoreSealed {
            campaign,
            score,
            output_root,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let bundle: SealedBundleV1 = read_json(&score)?;
            verify_sealed_bundle(&campaign, &bundle, &output_root).protocol()?;
            let mut score_sha256s = Vec::new();
            for score in &bundle.scores {
                let selection = bundle
                    .selections
                    .iter()
                    .find(|selection| selection.arm == score.arm)
                    .unwrap();
                let identity = score.identity(selection).protocol()?;
                publish(&output_root, &identity, "sealed_score", score)?;
                score_sha256s.push(identity);
            }
            score_sha256s.sort();
            let bundle_sha256 = bundle.identity().protocol()?;
            let receipt = SealedDatasetReceiptV1 {
                schema: SealedDatasetReceiptV1::SCHEMA.into(),
                sealed_bundle_sha256: bundle_sha256,
                claim_sha256: bundle.claim.identity().protocol()?,
                score_sha256s,
            };
            receipt.validate().protocol()?;
            let identity =
                raw_sha256(format!("sealed-dataset-receipt|{}", receipt.claim_sha256).as_bytes());
            publish(&output_root, &identity, "sealed_dataset_receipt", &receipt)?;
            println!("{identity}");
        }
        Command::Compare {
            campaign,
            stage,
            scores,
            output_root,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let scores: Vec<ComparisonScoreV1> = read_json(&scores)?;
            campaign.validate().protocol()?;
            let stage = stage.into();
            let expected = campaign
                .datasets
                .iter()
                .filter(|dataset| dataset.stage == stage)
                .map(|dataset| dataset.dataset_sha256.clone())
                .collect::<Vec<_>>();
            let comparison =
                compare_campaign(&campaign.identity().protocol()?, stage, &expected, &scores)
                    .protocol()?;
            let identity = comparison.identity().protocol()?;
            publish(&output_root, &identity, "comparison", &comparison)?;
            println!("{identity}");
        }
    }
    Ok(())
}
