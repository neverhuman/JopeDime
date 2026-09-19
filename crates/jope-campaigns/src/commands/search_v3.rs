use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::search_v3::{
    build_generation_state, canonical_json_bytes, CandidateEvidenceV1, CheckpointStoreV2,
    ConfirmedArchiveV1, SearchCandidatePlanV1, SearchRunContractV3, VerificationMode,
};
use serde::{de::DeserializeOwned, Serialize};

#[derive(Debug, Parser)]
#[command(
    name = "search-v3",
    about = "Deterministic Search V3 scientific evidence CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate and freeze the identities and work count for one generation.
    Plan {
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Build and durably publish one complete generation from captured evidence.
    Run {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long)]
        evidence: PathBuf,
    },
    /// Recover the latest complete checkpoint, optionally then publish a generation.
    Resume {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, requires_all = ["candidates", "evidence"])]
        contract: Option<PathBuf>,
        #[arg(long, requires_all = ["contract", "evidence"])]
        candidates: Option<PathBuf>,
        #[arg(long, requires_all = ["contract", "candidates"])]
        evidence: Option<PathBuf>,
    },
    /// Verify checkpoint identities and optionally recompute registered scores.
    Verify {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, default_value = "structural")]
        mode: String,
    },
}

#[derive(Serialize)]
struct PlanReceipt<'a> {
    schema: &'static str,
    contract_sha256: String,
    candidate_plan_sha256: String,
    generation: u32,
    task: &'a jope_dime_regressor::search_v3::SearchTaskV1,
    production_qualifiable: bool,
    fold_evaluation_count: u64,
}

#[derive(Serialize)]
struct RunReceipt {
    schema: &'static str,
    generation: u32,
    generation_state_sha256: String,
    archive_sha256: String,
    winner_semantic_sha256: String,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Plan {
            contract,
            candidates,
            output,
        } => {
            let contract: SearchRunContractV3 = read_json(&contract)?;
            let candidates: SearchCandidatePlanV1 = read_json(&candidates)?;
            contract.validate().map_err(anyhow::Error::msg)?;
            candidates.validate().map_err(anyhow::Error::msg)?;
            let candidate_plan_sha256 = candidates.identity().map_err(anyhow::Error::msg)?;
            if contract.candidate_plan_root_sha256 != candidate_plan_sha256 {
                return Err(anyhow!(
                    "contract does not bind the supplied candidate plan"
                ));
            }
            if candidates.generation >= contract.budget.generations {
                return Err(anyhow!(
                    "candidate generation lies outside the declared search budget"
                ));
            }
            let receipt = PlanReceipt {
                schema: "SearchPlanReceipt.v1",
                contract_sha256: contract.identity().map_err(anyhow::Error::msg)?,
                candidate_plan_sha256,
                generation: candidates.generation,
                task: &contract.task,
                production_qualifiable: contract.task.production_qualifiable(),
                fold_evaluation_count: contract.budget.fold_evaluation_count(),
            };
            emit_json(&receipt, output.as_deref())?;
        }
        Command::Run {
            run_dir,
            contract,
            candidates,
            evidence,
        } => publish_generation(&run_dir, &contract, &candidates, &evidence)?,
        Command::Resume {
            run_dir,
            contract,
            candidates,
            evidence,
        } => {
            let store = CheckpointStoreV2::new(&run_dir);
            let digest = store
                .repair_pointer_to_latest_complete()
                .map_err(anyhow::Error::msg)?;
            if let (Some(contract), Some(candidates), Some(evidence)) =
                (contract, candidates, evidence)
            {
                publish_generation(&run_dir, &contract, &candidates, &evidence)?;
            } else {
                println!("{{\"recovered_state_sha256\":\"{digest}\"}}");
            }
        }
        Command::Verify { run_dir, mode } => {
            let mode: VerificationMode = mode.parse().map_err(anyhow::Error::msg)?;
            let report = CheckpointStoreV2::new(run_dir)
                .verify(mode)
                .map_err(anyhow::Error::msg)?;
            emit_json(&report, None)?;
        }
    }
    Ok(())
}

fn publish_generation(
    run_dir: &Path,
    contract_path: &Path,
    candidates_path: &Path,
    evidence_path: &Path,
) -> Result<()> {
    let contract: SearchRunContractV3 = read_json(contract_path)?;
    let candidates: SearchCandidatePlanV1 = read_json(candidates_path)?;
    let evidence: Vec<CandidateEvidenceV1> = read_json(evidence_path)?;
    let store = CheckpointStoreV2::new(run_dir);
    let (previous, archive) = if candidates.generation == 0 {
        (None, ConfirmedArchiveV1::empty())
    } else {
        let (digest, state) = store.load_latest().map_err(anyhow::Error::msg)?;
        if state.generation + 1 != candidates.generation {
            return Err(anyhow!(
                "candidate generation does not follow the current checkpoint"
            ));
        }
        (Some(digest), state.archive)
    };
    let state = build_generation_state(contract, candidates, evidence, previous, &archive)
        .map_err(anyhow::Error::msg)?;
    let generation_state_sha256 = store.publish(&state).map_err(anyhow::Error::msg)?;
    emit_json(
        &RunReceipt {
            schema: "SearchRunReceipt.v1",
            generation: state.generation,
            generation_state_sha256,
            archive_sha256: state.selection.archive_sha256,
            winner_semantic_sha256: state.selection.winner_semantic_sha256,
        },
        None,
    )
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("decode {}", path.display()))
}

fn emit_json<T: Serialize>(value: &T, output: Option<&Path>) -> Result<()> {
    let mut bytes = canonical_json_bytes(value).map_err(anyhow::Error::msg)?;
    bytes.push(b'\n');
    if let Some(path) = output {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create {}", path.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("write {}", path.display()))?;
        file.sync_all()
            .with_context(|| format!("fsync {}", path.display()))?;
        if let Some(parent) = path.parent() {
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .with_context(|| format!("fsync {}", parent.display()))?;
        }
    } else {
        std::io::stdout().write_all(&bytes)?;
    }
    Ok(())
}
