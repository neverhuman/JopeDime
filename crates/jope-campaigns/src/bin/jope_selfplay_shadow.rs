//! Audit-first command surface for the causal self-play shadow namespace.

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::selfplay_shadow::{
    advance_corpus_pointer, append_evaluation_result, append_lineage_transition, atomic_json,
    build_context_lane, compare_and_swap_accepted_file, initialize_shadow, load_evaluation_index,
    plan_breadth_cohort, read_json, stage_corpus_shard, AcceptedByTaskV2, AcceptedFront,
    BreadthDataset, ContextProposalPool, EvaluationKeyV2, EvaluationResultV2, LineageStage,
    SelfPlayWaveV2, Sha256Digest, ShadowPaths, Task, DEFAULT_SHADOW_ROOT, SHADOW_WAVE_SCHEMA,
};
use serde_json::json;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-shadow",
    about = "Manage the isolated causal/cache-exact self-play v2 namespace"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_SHADOW_ROOT, global = true)]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create immutable generation-zero task fronts in the shadow namespace.
    Init {
        #[arg(long)]
        binary_checkpoint_sha256: String,
        #[arg(long)]
        regression_checkpoint_sha256: String,
        #[arg(long)]
        binary_evidence_sha256: String,
        #[arg(long)]
        regression_evidence_sha256: String,
    },
    /// Calculate one exact EvaluationKey.v2 identity.
    EvaluationKey {
        #[arg(long)]
        task: Task,
        #[arg(long)]
        dataset_content_sha256: String,
        #[arg(long)]
        semantic_sha256: String,
        #[arg(long)]
        evaluator_sha256: String,
        #[arg(long)]
        protocol_sha256: String,
    },
    /// Validate a frozen wave against all persisted exact evaluations.
    ValidateWave {
        #[arg(long)]
        wave: PathBuf,
        #[arg(long)]
        require_full_cohort: bool,
    },
    /// Plan one breadth-first 64-dataset task cohort.
    PlanCohort {
        #[arg(long)]
        task: Task,
        #[arg(long)]
        wave_index: u64,
        #[arg(long)]
        datasets: PathBuf,
    },
    /// Cache-filter, rank, and freeze one isolated task wave.
    BuildWave {
        #[arg(long)]
        wave_id: String,
        #[arg(long)]
        task: Task,
        #[arg(long)]
        parent_checkpoint_sha256: String,
        #[arg(long)]
        evaluator_sha256: String,
        #[arg(long)]
        protocol_sha256: String,
        #[arg(long)]
        direct_budget: usize,
        #[arg(long)]
        pools: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        require_full_cohort: bool,
    },
    /// Append one authoritative EvaluationResult.v2, idempotently.
    AppendResult {
        #[arg(long)]
        result: PathBuf,
    },
    /// Compare-and-swap only the selected accepted task front.
    Accept {
        #[arg(long)]
        task: Task,
        #[arg(long)]
        expected_parent_sha256: String,
        #[arg(long)]
        child_checkpoint_sha256: String,
        #[arg(long)]
        promotion_evidence_sha256: String,
    },
    /// Append one immutable transition to the causal lineage chain.
    Lineage {
        #[arg(long)]
        task: Task,
        #[arg(long)]
        generation: u64,
        #[arg(long)]
        parent_checkpoint_sha256: String,
        #[arg(long)]
        stage: LineageStage,
        #[arg(long)]
        artifact_sha256: String,
    },
    /// Stage an immutable corpus shard and optionally advance its CAS pointer.
    Corpus {
        #[arg(long)]
        shard: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        publish: bool,
        #[arg(long)]
        expected_previous_sha256: Option<String>,
    },
    /// Report the isolated fronts and exact evaluation count.
    Status,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis() as u64
}

fn digest(value: String) -> Result<Sha256Digest> {
    Sha256Digest::parse(value)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let paths = ShadowPaths::new(args.root)?;
    let output = match args.command {
        Command::Init {
            binary_checkpoint_sha256,
            regression_checkpoint_sha256,
            binary_evidence_sha256,
            regression_evidence_sha256,
        } => {
            let timestamp = now_ms();
            let accepted = AcceptedByTaskV2::new(
                AcceptedFront {
                    task: Task::Binary,
                    generation: 0,
                    checkpoint_sha256: digest(binary_checkpoint_sha256)?,
                    parent_checkpoint_sha256: None,
                    promotion_evidence_sha256: digest(binary_evidence_sha256)?,
                    promoted_at_ms: timestamp,
                },
                AcceptedFront {
                    task: Task::Regression,
                    generation: 0,
                    checkpoint_sha256: digest(regression_checkpoint_sha256)?,
                    parent_checkpoint_sha256: None,
                    promotion_evidence_sha256: digest(regression_evidence_sha256)?,
                    promoted_at_ms: timestamp,
                },
            )?;
            initialize_shadow(&paths, &accepted, timestamp)?;
            json!({
                "schema": "SelfPlayShadowInit.v2",
                "root": paths.root,
                "accepted": accepted,
                "production_anchor_mutated": false,
            })
        }
        Command::EvaluationKey {
            task,
            dataset_content_sha256,
            semantic_sha256,
            evaluator_sha256,
            protocol_sha256,
        } => {
            let key = EvaluationKeyV2::new(
                task,
                digest(dataset_content_sha256)?,
                digest(semantic_sha256)?,
                digest(evaluator_sha256)?,
                digest(protocol_sha256)?,
            );
            json!({"evaluation_key": key, "evaluation_key_sha256": key.id()})
        }
        Command::ValidateWave {
            wave,
            require_full_cohort,
        } => {
            let wave: SelfPlayWaveV2 = read_json(&wave)?;
            let index = load_evaluation_index(&paths)?;
            wave.validate(&index, require_full_cohort)?;
            json!({
                "schema": "SelfPlayWaveValidation.v2",
                "wave_id": wave.wave_id,
                "task": wave.task,
                "contexts": wave.contexts.len(),
                "actions": wave.contexts.iter().map(|context| context.actions.len()).sum::<usize>(),
                "zero_backfill": wave.contexts.iter().all(|context| context.zero_backfill),
                "valid": true,
            })
        }
        Command::PlanCohort {
            task,
            wave_index,
            datasets,
        } => {
            let datasets: Vec<BreadthDataset> = read_json(&datasets)?;
            serde_json::to_value(plan_breadth_cohort(task, wave_index, &datasets)?)?
        }
        Command::BuildWave {
            wave_id,
            task,
            parent_checkpoint_sha256,
            evaluator_sha256,
            protocol_sha256,
            direct_budget,
            pools,
            output,
            require_full_cohort,
        } => {
            ensure!(!wave_id.trim().is_empty(), "wave id is empty");
            let parent = digest(parent_checkpoint_sha256)?;
            let evaluator = digest(evaluator_sha256)?;
            let protocol = digest(protocol_sha256)?;
            let pools: Vec<ContextProposalPool> = read_json(&pools)?;
            ensure!(
                pools.iter().all(|pool| pool.context.task == task),
                "proposal pools contain another task"
            );
            let index = load_evaluation_index(&paths)?;
            let contexts = pools
                .into_iter()
                .map(|pool| {
                    build_context_lane(pool, direct_budget, &parent, &evaluator, &protocol, &index)
                })
                .collect::<Result<Vec<_>>>()?;
            let wave = SelfPlayWaveV2 {
                schema: SHADOW_WAVE_SCHEMA.to_owned(),
                wave_id,
                task,
                parent_checkpoint_sha256: parent,
                evaluator_sha256: evaluator,
                protocol_sha256: protocol,
                contexts,
            };
            wave.validate(&index, require_full_cohort)?;
            atomic_json(&output, &wave)?;
            json!({
                "schema": "SelfPlayWaveBuildReceipt.v2",
                "wave_id": wave.wave_id,
                "task": wave.task,
                "contexts": wave.contexts.len(),
                "actions": wave.contexts.iter().map(|context| context.actions.len()).sum::<usize>(),
                "output": output,
                "zero_backfill": true,
            })
        }
        Command::AppendResult { result } => {
            let result: EvaluationResultV2 = read_json(&result)?;
            let path = append_evaluation_result(&paths, &result)?;
            json!({
                "schema": "EvaluationAppendReceipt.v2",
                "evaluation_key_sha256": result.evaluation_key.id(),
                "path": path,
                "append_only": true,
            })
        }
        Command::Accept {
            task,
            expected_parent_sha256,
            child_checkpoint_sha256,
            promotion_evidence_sha256,
        } => {
            ensure!(
                paths.marker().is_file(),
                "shadow namespace must be initialized before acceptance"
            );
            let accepted = compare_and_swap_accepted_file(
                &paths,
                task,
                &digest(expected_parent_sha256)?,
                digest(child_checkpoint_sha256)?,
                digest(promotion_evidence_sha256)?,
                now_ms(),
            )?;
            json!({
                "schema": "TaskAcceptanceReceipt.v2",
                "task": task,
                "accepted": accepted,
                "supply_wave_created": false,
            })
        }
        Command::Lineage {
            task,
            generation,
            parent_checkpoint_sha256,
            stage,
            artifact_sha256,
        } => {
            let event_sha256 = append_lineage_transition(
                &paths,
                task,
                generation,
                digest(parent_checkpoint_sha256)?,
                stage,
                digest(artifact_sha256)?,
                now_ms(),
            )?;
            json!({
                "schema": "CausalLineageAppendReceipt.v2",
                "task": task,
                "generation": generation,
                "stage": stage,
                "event_sha256": event_sha256,
            })
        }
        Command::Corpus {
            shard,
            rows,
            publish,
            expected_previous_sha256,
        } => {
            let staged = stage_corpus_shard(&paths, &std::fs::read(&shard)?)?;
            let pointer = if publish {
                let expected = expected_previous_sha256.map(digest).transpose()?;
                Some(advance_corpus_pointer(
                    &paths,
                    expected.as_ref(),
                    &staged,
                    rows,
                    now_ms(),
                )?)
            } else {
                None
            };
            json!({
                "schema": "CorpusPublicationReceipt.v2",
                "staged_shard": staged,
                "published": publish,
                "pointer": pointer,
            })
        }
        Command::Status => {
            let accepted: AcceptedByTaskV2 = read_json(&paths.accepted())
                .with_context(|| "shadow namespace is not initialized")?;
            let index = load_evaluation_index(&paths)?;
            json!({
                "schema": "SelfPlayShadowStatus.v2",
                "root": paths.root,
                "binary_front": accepted.binary,
                "regression_front": accepted.regression,
                "authoritative_evaluation_keys": index.completed.len(),
                "active_or_reserved_evaluation_keys": index.active_or_reserved.len(),
                "production_anchor_mutated": false,
            })
        }
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
