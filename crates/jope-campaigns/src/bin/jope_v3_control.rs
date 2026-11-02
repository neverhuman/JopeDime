//! Fail-closed activation and validation for the v3 flywheel proof path.

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, now_ms, read_json, sha256_file,
};
#[cfg(test)]
use jope_dime_regressor::selfplay_shadow::PromotionObservationV3;
use jope_dime_regressor::selfplay_shadow::{
    build_sealed_panel_pool_v2, ensure_evidence_epoch_immutable, evaluate_task_promotion_v3,
    sealed_zero_prior_use_proof_v2, validate_panel_consumption, CausalEvidenceEpochV1,
    SealedPanelCandidateSetV2, SealedPanelCandidateV2, SealedPanelConsumptionV1,
    SealedPanelContextFactsV1, SealedPanelPoolV2, Sha256Digest, Task,
    TaskPromotionObservationSetV3, SEALED_PANEL_CANDIDATE_SET_V2_SCHEMA,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "jope-v3-control",
    about = "Open immutable evidence epochs and validate v3 promotion authorities"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    OpenEvidenceEpoch {
        #[arg(long)]
        epoch: PathBuf,
    },
    ValidateSealedPanelPool {
        #[arg(long)]
        pool: PathBuf,
    },
    BuildSealedPanelPool {
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ProduceSealedPanelCandidates {
        #[arg(long)]
        base_seal: PathBuf,
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        facts_directory: PathBuf,
        #[arg(long)]
        prior_use_index: PathBuf,
        #[arg(long)]
        admission_watermark: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    EvaluateTaskPromotion {
        #[arg(long)]
        pool: PathBuf,
        #[arg(long)]
        consumption: PathBuf,
        #[arg(long)]
        observations: PathBuf,
        #[arg(long)]
        accepted_parent: String,
        #[arg(long)]
        bootstrap_seed: u64,
    },
    FreezeLegacyPromotion,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedPanelPriorUseIndexV1 {
    schema: String,
    watermark: u64,
    used_context_sha256: Vec<Sha256Digest>,
}

fn produce_sealed_panel_candidates(
    root: &Path,
    base_seal: &Path,
    catalog: &Path,
    facts_directory: &Path,
    prior_use_index: &Path,
    admission_watermark: u64,
    output: Option<PathBuf>,
) -> Result<Value> {
    ensure!(
        base_seal.is_file()
            && catalog.is_file()
            && prior_use_index.is_file()
            && facts_directory.is_dir(),
        "sealed candidate producer input is absent"
    );
    let base_seal_sha256 = Sha256Digest::parse(sha256_file(base_seal)?)?;
    let catalog_sha256 = Sha256Digest::parse(sha256_file(catalog)?)?;
    let prior_use_index_sha256 = Sha256Digest::parse(sha256_file(prior_use_index)?)?;
    let prior: SealedPanelPriorUseIndexV1 = serde_json::from_value(read_json(prior_use_index)?)?;
    ensure!(
        prior.schema == "SealedPanelPriorUseIndex.v1" && prior.watermark == admission_watermark,
        "sealed prior-use index scope drift"
    );
    let used = prior
        .used_context_sha256
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let mut facts_paths = fs::read_dir(facts_directory)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    facts_paths.sort();
    ensure!(
        !facts_paths.is_empty(),
        "sealed context-facts directory is empty"
    );
    let mut entries = Vec::with_capacity(facts_paths.len());
    for path in facts_paths {
        let facts_sha256 = Sha256Digest::parse(sha256_file(&path)?)?;
        let facts: SealedPanelContextFactsV1 = serde_json::from_value(read_json(&path)?)?;
        facts.validate()?;
        ensure!(
            facts.catalog_sha256 == catalog_sha256,
            "sealed context facts catalog drift"
        );
        ensure!(
            !used.contains(&facts.context_sha256),
            "sealed context was already used at the admission watermark"
        );
        let proof = sealed_zero_prior_use_proof_v2(
            &facts.context_sha256,
            &base_seal_sha256,
            &catalog_sha256,
            &prior_use_index_sha256,
            admission_watermark,
        );
        entries.push(SealedPanelCandidateV2 {
            task: facts.task,
            context_identity: facts.context_identity,
            context_sha256: facts.context_sha256,
            source_family_sha256: facts.source_family_sha256,
            near_duplicate_cluster_sha256: facts.near_duplicate_cluster_sha256,
            origin: facts.origin,
            base_seal_sha256: base_seal_sha256.clone(),
            catalog_sha256: catalog_sha256.clone(),
            train_sha256: facts.train_sha256,
            test_sha256: facts.test_sha256,
            context_facts_sha256: facts_sha256,
            prior_use_index_sha256: prior_use_index_sha256.clone(),
            zero_prior_use_proof_sha256: proof,
            admission_watermark,
        });
    }
    entries.sort_by(|left, right| left.context_sha256.cmp(&right.context_sha256));
    let output = output.unwrap_or_else(|| root.join("training/sealed-panel-candidates.v2.json"));
    let produced_at_ms = if output.is_file() {
        serde_json::from_value::<SealedPanelCandidateSetV2>(read_json(&output)?)?.produced_at_ms
    } else {
        now_ms()
    };
    let candidate_set = SealedPanelCandidateSetV2 {
        schema: SEALED_PANEL_CANDIDATE_SET_V2_SCHEMA.to_owned(),
        base_seal_sha256,
        catalog_sha256,
        prior_use_index_sha256,
        admission_watermark,
        entries,
        produced_at_ms,
    };
    candidate_set.validate()?;
    if output.is_file() {
        let existing: SealedPanelCandidateSetV2 = serde_json::from_value(read_json(&output)?)?;
        existing.validate()?;
        ensure!(
            existing == candidate_set,
            "sealed candidate set is immutable"
        );
    } else {
        atomic_json(&output, &serde_json::to_value(&candidate_set)?)?;
    }
    Ok(json!({
        "schema": "SealedPanelCandidateSetPublication.v2",
        "candidate_set": output,
        "candidate_set_sha256": sha256_file(&output)?,
        "entries": candidate_set.entries.len(),
        "admission_watermark": admission_watermark,
        "published_at_ms": now_ms(),
    }))
}

fn accepted_parent(root: &Path, task: Task) -> Result<String> {
    read_json(&root.join("training/accepted-by-task.json"))?
        .get("tasks")
        .and_then(|tasks| tasks.get(task.as_str()))
        .and_then(|row| row.get("checkpoint_sha256"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .with_context(|| format!("accepted pointer lacks {task}"))
}

fn open_evidence_epoch(root: &Path, source: &Path) -> Result<Value> {
    let epoch: CausalEvidenceEpochV1 = serde_json::from_value(read_json(source)?)?;
    epoch.validate()?;
    ensure!(
        accepted_parent(root, epoch.task)? == epoch.accepted_parent_checkpoint_sha256.as_str(),
        "evidence epoch is not pinned to the current accepted task parent"
    );
    let target = root
        .join("training/causal-evidence-epochs")
        .join(format!("{}.v1.json", epoch.task.as_str()));
    let status = if target.is_file() {
        let existing: CausalEvidenceEpochV1 = serde_json::from_value(read_json(&target)?)?;
        ensure_evidence_epoch_immutable(&existing, &epoch)?;
        "already_open_identical"
    } else {
        atomic_json(&target, &serde_json::to_value(&epoch)?)?;
        "opened"
    };
    Ok(json!({
        "schema": "CausalEvidenceEpochActivation.v1",
        "task": epoch.task,
        "epoch": target,
        "epoch_sha256": epoch.identity()?,
        "status": status,
        "activated_at_ms": now_ms(),
    }))
}

fn freeze_legacy_promotion(root: &Path) -> Result<Value> {
    let existing_sealed = root.join("catalog/private/sealed-expanded-v1.json");
    let marker = json!({
        "schema": "FlywheelPromotionAuthority.v3",
        "created_at_ms": now_ms(),
        "production_frozen": true,
        "legacy_cumulative_causal_ledger_authoritative": false,
        "legacy_recycled_canary_authoritative": false,
        "legacy_sealed_cohort": {
            "path": existing_sealed,
            "authoritative": false,
            "reason": "pre-v2 provenance families and empty scored audit registry",
        },
        "required_evidence": "CausalEvidenceLedger.v3 promotion_epoch",
        "required_external_evaluation": "three TaskPromotionEvaluation.v3 passes on disjoint single-use panels",
    });
    let path = root.join("ops/promotion-authority.v3.json");
    if path.is_file() {
        let existing = read_json(&path)?;
        ensure!(
            existing.get("schema").and_then(Value::as_str) == Some("FlywheelPromotionAuthority.v3")
                && existing.get("production_frozen").and_then(Value::as_bool) == Some(true),
            "existing promotion authority marker is unsafe"
        );
        return Ok(existing);
    }
    atomic_json(&path, &marker)?;
    Ok(marker)
}

fn evaluate_task_promotion(
    root: &Path,
    pool_path: &Path,
    consumption_path: &Path,
    observations_path: &Path,
    accepted_parent_sha256: &str,
    bootstrap_seed: u64,
) -> Result<Value> {
    let pool: SealedPanelPoolV2 = serde_json::from_value(read_json(pool_path)?)?;
    pool.validate()?;
    let consumption: SealedPanelConsumptionV1 =
        serde_json::from_value(read_json(consumption_path)?)?;
    consumption.validate()?;
    let consumption_sha256 = Sha256Digest::parse(sha256_file(consumption_path)?)?;
    let observation_set: TaskPromotionObservationSetV3 =
        serde_json::from_value(read_json(observations_path)?)?;
    observation_set.validate()?;
    ensure!(
        observation_set.schema == "TaskPromotionObservationSet.v3"
            && observation_set.panel_consumption_sha256 == consumption_sha256,
        "promotion observations are not pinned to their panel consumption"
    );
    let parent_digest = Sha256Digest::parse(accepted_parent_sha256.to_owned())?;
    ensure!(
        parent_digest.as_str() == accepted_parent(root, consumption.task)?
            && observation_set
                .observations
                .iter()
                .all(|row| consumption.context_sha256.contains(&row.context_sha256)),
        "promotion observations do not match the current parent or consumed contexts"
    );
    let family_by_context = pool
        .promotion_panel(consumption.task, &consumption.panel)
        .into_iter()
        .map(|entry| (&entry.context_sha256, &entry.source_family_sha256))
        .collect::<std::collections::BTreeMap<_, _>>();
    ensure!(
        observation_set.observations.iter().all(|row| {
            family_by_context.get(&row.context_sha256).copied() == Some(&row.source_family_sha256)
        }),
        "promotion observation source-family provenance drift"
    );
    let consumed_directory = root.join("training/sealed-panels/consumed");
    let prior = if consumed_directory.is_dir() {
        fs::read_dir(&consumed_directory)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != consumption_path)
            .filter_map(|entry| {
                serde_json::from_value::<SealedPanelConsumptionV1>(read_json(&entry.path()).ok()?)
                    .ok()
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    validate_panel_consumption(&pool, &consumption, &prior)?;
    let evaluation = evaluate_task_promotion_v3(
        consumption.task,
        parent_digest,
        consumption.candidate_checkpoint_sha256.clone(),
        consumption_sha256,
        consumption.run_index,
        &observation_set.observations,
        bootstrap_seed,
        now_ms(),
    )?;
    let tier = if evaluation.accepted {
        "accepted"
    } else {
        "rejected"
    };
    let directory = root.join("training/task-promotion-evaluations").join(tier);
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "{}-{}-run{}-n{}.v3.json",
        evaluation.candidate_checkpoint_sha256,
        evaluation.task.as_str(),
        evaluation.run_index,
        evaluation.metrics.contexts,
    ));
    if path.is_file() {
        let existing: jope_dime_regressor::selfplay_shadow::TaskPromotionEvaluationV3 =
            serde_json::from_value(read_json(&path)?)?;
        ensure!(
            existing == evaluation,
            "promotion evaluation identity drift"
        );
    } else {
        atomic_json(&path, &serde_json::to_value(&evaluation)?)?;
    }
    Ok(json!({
        "schema": "TaskPromotionEvaluationPublication.v3",
        "evaluation": path,
        "evaluation_sha256": sha256_file(&path)?,
        "accepted": evaluation.accepted,
        "rejection_reasons": evaluation.rejection_reasons,
        "published_at_ms": now_ms(),
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let result = match args.command {
        Command::OpenEvidenceEpoch { epoch } => open_evidence_epoch(&args.root, &epoch)?,
        Command::ValidateSealedPanelPool { pool } => {
            let value: SealedPanelPoolV2 = serde_json::from_value(read_json(&pool)?)?;
            value.validate()?;
            json!({
                "schema": "SealedPanelPoolValidation.v2",
                "pool": pool,
                "pool_sha256": sha256_file(&pool)?,
                "entries": value.entries.len(),
                "valid": true,
                "validated_at_ms": now_ms(),
            })
        }
        Command::BuildSealedPanelPool { candidates, output } => {
            let candidates_sha256 = sha256_file(&candidates)?;
            let source: SealedPanelCandidateSetV2 =
                serde_json::from_value(read_json(&candidates)?)?;
            source.validate()?;
            let output =
                output.unwrap_or_else(|| args.root.join("training/sealed-panel-pool.v2.json"));
            let existing = if output.is_file() {
                Some(serde_json::from_value::<SealedPanelPoolV2>(read_json(
                    &output,
                )?)?)
            } else {
                None
            };
            let pool = build_sealed_panel_pool_v2(
                source.entries,
                existing
                    .as_ref()
                    .map_or_else(now_ms, |pool| pool.created_at_ms),
            )?;
            if let Some(existing) = existing {
                existing.validate()?;
                ensure!(existing == pool, "sealed panel pool is immutable");
            } else {
                atomic_json(&output, &serde_json::to_value(&pool)?)?;
            }
            json!({
                "schema": "SealedPanelPoolBuild.v2",
                "candidates": candidates,
                "candidates_sha256": candidates_sha256,
                "pool": output,
                "pool_sha256": sha256_file(&output)?,
                "entries": pool.entries.len(),
                "built_at_ms": now_ms(),
            })
        }
        Command::ProduceSealedPanelCandidates {
            base_seal,
            catalog,
            facts_directory,
            prior_use_index,
            admission_watermark,
            output,
        } => produce_sealed_panel_candidates(
            &args.root,
            &base_seal,
            &catalog,
            &facts_directory,
            &prior_use_index,
            admission_watermark,
            output,
        )?,
        Command::EvaluateTaskPromotion {
            pool,
            consumption,
            observations,
            accepted_parent,
            bootstrap_seed,
        } => evaluate_task_promotion(
            &args.root,
            &pool,
            &consumption,
            &observations,
            &accepted_parent,
            bootstrap_seed,
        )?,
        Command::FreezeLegacyPromotion => freeze_legacy_promotion(&args.root)?,
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests;
