use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use crate::jope_generalized_gp::*;
use crate::search_v3::canonical_sha256;
use serde::de::DeserializeOwned;
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
#[command(name = "jope_generalized_gp")]
#[command(about = "Additive split-robust regression/binary GP campaign runner")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a complete immutable production freeze without launching it.
    ValidateFreeze {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        checkpoints: PathBuf,
        #[arg(long)]
        evaluator: PathBuf,
        #[arg(long, default_value_t = false)]
        verify_artifact_bytes: bool,
    },
    /// Evaluate the explicit operational launch gate.
    LaunchGate {
        #[arg(long)]
        evidence: PathBuf,
    },
    /// Freeze development or post-development validation tranches.
    PlanTranches {
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        campaign_sha256: String,
        #[arg(long)]
        pilot_comparison_sha256: String,
        #[arg(long)]
        arm: ArmArg,
        #[arg(long)]
        partition: PartitionArg,
        #[arg(long)]
        development_completion_sha256: Option<String>,
        #[arg(long, value_delimiter = ',')]
        reused_pilot_dataset_sha256s: Vec<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Run the deterministic two-dataset protocol smoke test with simulated
    /// evaluator evidence. It is never benchmark-valid.
    DryRun {
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ArmArg {
    RotatingMean,
    ConfidenceGp,
    QualityDiversityGp,
}

impl From<ArmArg> for CampaignArmV1 {
    fn from(value: ArmArg) -> Self {
        match value {
            ArmArg::RotatingMean => Self::RotatingMean,
            ArmArg::ConfidenceGp => Self::ConfidenceGp,
            ArmArg::QualityDiversityGp => Self::QualityDiversityGp,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum PartitionArg {
    Development,
    Validation,
}

impl From<PartitionArg> for CatalogPartitionV1 {
    fn from(value: PartitionArg) -> Self {
        match value {
            PartitionArg::Development => Self::Development,
            PartitionArg::Validation => Self::Validation,
        }
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("decode {}", path.display()))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("refusing to replace {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn fixture_genome(task: GpTaskV1, index: usize) -> Result<GenomeV1> {
    let head = match task {
        GpTaskV1::Regression => "ridge",
        GpTaskV1::Binary => "logit_ridge",
    };
    let source = format!(
        "(pipe X (standardize) ({head} lambda_frac={}))",
        (index + 1) as f64 / 10_000.0
    );
    let canonical = feat_math::GenomeIr::parse(&source)?.canonical();
    GenomeV1::from_canonical(task, &canonical).map_err(anyhow::Error::msg)
}

fn fixture_rows(task: GpTaskV1) -> Vec<SplitRowV1> {
    (0..70)
        .map(|index| SplitRowV1 {
            stable_row_id: format!("row-{index:03}"),
            atomic_group: format!("group-{index:03}"),
            regression_target: (task == GpTaskV1::Regression)
                .then_some((index as f64 / 7.0).sin() + index as f64 / 100.0),
            binary_target: (task == GpTaskV1::Binary).then_some((index % 2) as u8),
        })
        .collect()
}

fn dry_dataset(task: GpTaskV1, root: &Path) -> Result<serde_json::Value> {
    let dataset_sha256 = raw_sha256(format!("dry-run-{}", task.as_str()).as_bytes());
    let training_sha256 = raw_sha256(format!("dry-training-{}", task.as_str()).as_bytes());
    let seed_sha256 = raw_sha256(format!("dry-seed-{}", task.as_str()).as_bytes());
    let rows = fixture_rows(task);
    let rotating =
        build_rotating_split_plan(task, &dataset_sha256, &training_sha256, &seed_sha256, &rows)
            .protocol()?;
    let cv7 = build_cv7_plan(
        task,
        &dataset_sha256,
        &training_sha256,
        &raw_sha256(format!("{seed_sha256}|cv7").as_bytes()),
        &rows,
    )
    .protocol()?;

    let genomes = (0..80)
        .map(|index| fixture_genome(task, index))
        .collect::<Result<Vec<_>>>()?;
    let proposal = ProposalFreezeV1 {
        schema: ProposalFreezeV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.clone(),
        checkpoint_sha256: raw_sha256(format!("dry-checkpoint-{}", task.as_str()).as_bytes()),
        inference_request_sha256: raw_sha256(
            format!("dry-inference-request-{}", task.as_str()).as_bytes(),
        ),
        inference_receipt_sha256: raw_sha256(
            format!("dry-inference-receipt-{}", task.as_str()).as_bytes(),
        ),
        inference_ordinal: 1,
        proposals: genomes[..64].to_vec(),
    };
    let mut ledger = InferenceLedgerV1::empty();
    ledger.record(&proposal).protocol()?;
    let global = genomes[64..74]
        .iter()
        .enumerate()
        .map(|(index, genome)| GlobalGeneRankV1 {
            genome: genome.clone(),
            support_shrunk_lower_bound: 0.9 - index as f64 / 100.0,
            worst_percentile: 0.8 - index as f64 / 100.0,
            dataset_support: 3,
            source_family_support: 2,
            strict_wins: 1,
        })
        .collect::<Vec<_>>();
    let generation_zero = build_generation_zero(&proposal, &global, &genomes[74..]).protocol()?;
    let split_plan_sha256 = rotating.identity().protocol()?;
    let arm = CampaignArmV1::ConfidenceGp;
    let mut evaluations = Vec::new();
    for split in &rotating.splits {
        let split_sha256 = canonical_sha256(split).protocol()?;
        for (index, member) in generation_zero.population.iter().enumerate() {
            let uplift = 0.2 - index as f64 / 500.0 + split.split_index as f64 / 1000.0;
            let anchor = 0.1;
            evaluations.push(CandidateSplitEvaluationV1 {
                schema: CandidateSplitEvaluationV1::SCHEMA.into(),
                arm,
                dataset_sha256: dataset_sha256.clone(),
                generation: split.split_index,
                split_index: split.split_index,
                split_sha256: split_sha256.clone(),
                candidate: member.genome.clone(),
                prediction_object_sha256: raw_sha256(
                    format!("dry-prediction|{index}|{}", split.split_index).as_bytes(),
                ),
                candidate_skill: anchor + uplift,
                anchor_median_skill: anchor,
                paired_uplift: uplift,
                secondary_metric: 1.0 - uplift,
                valid: true,
                invalid_reason: None,
                runtime: RuntimeObservationV1 {
                    cpu_seconds: 0.001,
                    winner_latency_seconds: 0.001,
                    timed_out: false,
                },
            });
        }
    }
    let ranked = rank_population(arm, &evaluations).protocol()?;
    let next = build_next_generation(arm, 1, &seed_sha256, &ranked).protocol()?;
    let ranked_generations = vec![ranked.clone(); 5];
    let archive = build_provisional_archive(
        arm,
        &dataset_sha256,
        &split_plan_sha256,
        &ranked_generations,
        &evaluations,
        |_, _| Err("dry-run archive unexpectedly needed backfill".into()),
    )
    .protocol()?;
    let archive_sha256 = archive.identity().protocol()?;
    let finalists = archive
        .members
        .iter()
        .take(10)
        .enumerate()
        .map(|(index, member)| Cv7CandidateEvidenceV1 {
            candidate: member.candidate.clone(),
            fold_primary_scores: vec![0.8 - index as f64 / 100.0; 7],
            pooled_oof_primary: 0.8 - index as f64 / 100.0,
            secondary_metric: 0.2 + index as f64 / 100.0,
            runtime_seconds: 0.01 + index as f64 / 1000.0,
            valid: true,
        })
        .collect();
    let cv7_sha256 = cv7.identity().protocol()?;
    let selection = select_cv7_winner(
        arm,
        &dataset_sha256,
        &archive_sha256,
        &cv7_sha256,
        finalists,
    )
    .protocol()?;
    let store = ArtifactStoreV1::new(root.join("cas"));
    let selection_identity = selection.identity().protocol()?;
    let selection_object = store.publish(&selection_identity, &selection).protocol()?;
    Ok(json!({
        "task": task,
        "dataset_sha256": dataset_sha256,
        "split_plan_sha256": split_plan_sha256,
        "cv7_plan_sha256": cv7_sha256,
        "generation_zero_sha256": generation_zero.identity().protocol()?,
        "inference_ledger_sha256": ledger.identity().protocol()?,
        "next_generation_members": next.len(),
        "archive_sha256": archive_sha256,
        "selection_freeze_sha256": selection_identity,
        "selection_object_sha256": selection_object,
        "winner_semantic_sha256": selection.winner_semantic_sha256,
    }))
}

fn dry_run(output: &Path) -> Result<()> {
    fs::create_dir(output).with_context(|| {
        format!(
            "dry-run output must be a new directory: {}",
            output.display()
        )
    })?;
    let datasets = [
        dry_dataset(GpTaskV1::Regression, output)?,
        dry_dataset(GpTaskV1::Binary, output)?,
    ];
    let report = json!({
        "schema": "JopeGeneralizedGpDryRun.v1",
        "benchmark_valid": false,
        "simulated_evaluator": true,
        "dataset_count": 2,
        "datasets": datasets,
    });
    let bytes = serde_json::to_vec(&report)?;
    let root = raw_sha256(&bytes);
    write_new(&output.join("dry-run.v1.json"), &bytes)?;
    write_new(&output.join("dry-run.root"), format!("{root}\n").as_bytes())?;
    println!("{root}");
    Ok(())
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    match args.command {
        Command::ValidateFreeze {
            campaign,
            catalog,
            checkpoints,
            evaluator,
            verify_artifact_bytes,
        } => {
            let campaign: CampaignFreezeV1 = read_json(&campaign)?;
            let catalog: CatalogFreezeV1 = read_json(&catalog)?;
            let checkpoints: CheckpointFreezeV1 = read_json(&checkpoints)?;
            let evaluator: EvaluatorFreezeV1 = read_json(&evaluator)?;
            campaign.validate().protocol()?;
            catalog.validate_production_cardinality().protocol()?;
            checkpoints.validate().protocol()?;
            evaluator.validate().protocol()?;
            if campaign.catalog_freeze_sha256 != catalog.identity().protocol()?
                || campaign.checkpoint_freeze_sha256 != checkpoints.identity().protocol()?
                || campaign.evaluator_freeze_sha256 != evaluator.identity().protocol()?
                || campaign.observation_snapshot_sha256 != catalog.observation_snapshot_sha256
            {
                bail!("campaign references do not bind the supplied freezes");
            }
            if verify_artifact_bytes {
                checkpoints.verify_artifacts().protocol()?;
                evaluator.verify_artifact().protocol()?;
            }
            println!("{}", campaign.identity().protocol()?);
            Ok(())
        }
        Command::LaunchGate { evidence } => {
            let gate: LaunchGateV1 = read_json(&evidence)?;
            if !gate.launch_allowed().protocol()? {
                bail!("generalized GP campaign launch is blocked by operational evidence");
            }
            println!("launch_allowed");
            Ok(())
        }
        Command::PlanTranches {
            catalog,
            campaign_sha256,
            pilot_comparison_sha256,
            arm,
            partition,
            development_completion_sha256,
            reused_pilot_dataset_sha256s,
            output,
        } => {
            let catalog: CatalogFreezeV1 = read_json(&catalog)?;
            let plan = build_tranche_plan(
                &catalog,
                &campaign_sha256,
                arm.into(),
                partition.into(),
                &pilot_comparison_sha256,
                development_completion_sha256,
                reused_pilot_dataset_sha256s,
            )
            .protocol()?;
            let mut bytes = serde_json::to_vec_pretty(&plan)?;
            bytes.push(b'\n');
            write_new(&output, &bytes)?;
            println!("{}", plan.identity().protocol()?);
            Ok(())
        }
        Command::DryRun { output } => dry_run(&output),
    }
}
