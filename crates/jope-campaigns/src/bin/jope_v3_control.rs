//! Fail-closed activation and validation for the v3 flywheel proof path.

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, now_ms, read_json, sha256_file, DEFAULT_ROOT,
};
use jope_dime_regressor::selfplay_shadow::{
    build_sealed_panel_pool_v2, ensure_evidence_epoch_immutable, evaluate_task_promotion_v3,
    validate_panel_consumption, CausalEvidenceEpochV1, PromotionObservationV3,
    SealedPanelCandidateV2, SealedPanelConsumptionV1, SealedPanelPoolV2, Sha256Digest, Task,
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
    #[arg(long, default_value = DEFAULT_ROOT)]
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
struct PromotionObservationSetV3 {
    schema: String,
    panel_consumption_sha256: Sha256Digest,
    observations: Vec<PromotionObservationV3>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedPanelCandidateSetV2 {
    schema: String,
    entries: Vec<SealedPanelCandidateV2>,
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
    let observation_set: PromotionObservationSetV3 =
        serde_json::from_value(read_json(observations_path)?)?;
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
            ensure!(
                source.schema == "SealedPanelCandidateSet.v2",
                "sealed panel candidate set schema is not v2"
            );
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
mod tests {
    use super::*;
    use jope_dime_regressor::selfplay_shadow::{
        EvidenceEpochContextV1, EvidencePreflightCohortV1, CAUSAL_EVIDENCE_EPOCH_SCHEMA,
        EVIDENCE_SLOTS_PER_CONTEXT, PREFLIGHT_COHORTS_REQUIRED, PREFLIGHT_CONTEXTS_PER_COHORT,
    };
    use tempfile::tempdir;

    fn digest(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    #[test]
    fn opening_an_epoch_is_idempotent_but_not_resettable() {
        let temporary = tempdir().unwrap();
        let root = temporary.path();
        let parent = digest("parent");
        atomic_json(
            &root.join("training/accepted-by-task.json"),
            &json!({
                "tasks": {
                    "binary": {"checkpoint_sha256": parent},
                    "regression": {"checkpoint_sha256": digest("regression")},
                }
            }),
        )
        .unwrap();
        let epoch = CausalEvidenceEpochV1 {
            schema: CAUSAL_EVIDENCE_EPOCH_SCHEMA.to_owned(),
            task: Task::Binary,
            accepted_parent_checkpoint_sha256: parent.clone(),
            start_wave: 300,
            runtime_policy_sha256: digest("runtime"),
            proposer_policy_sha256: digest("proposer"),
            evaluator_policy_sha256: digest("evaluator"),
            catalog_eligibility_sha256: digest("catalog"),
            requested_contexts: vec![EvidenceEpochContextV1 {
                context_identity: "context".to_owned(),
                context_sha256: digest("context"),
                requested_wave: 300,
            }],
            slots_per_context: EVIDENCE_SLOTS_PER_CONTEXT,
            preflight_cohorts: (0..PREFLIGHT_COHORTS_REQUIRED)
                .map(|sequence| EvidencePreflightCohortV1 {
                    sequence,
                    task: Task::Binary,
                    accepted_parent_checkpoint_sha256: parent.clone(),
                    contexts: PREFLIGHT_CONTEXTS_PER_COHORT,
                    requested_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
                    valid_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
                    completed: true,
                })
                .collect(),
            opened_at_ms: 1,
        };
        let source = root.join("epoch.json");
        atomic_json(&source, &serde_json::to_value(&epoch).unwrap()).unwrap();
        assert_eq!(
            open_evidence_epoch(root, &source).unwrap()["status"],
            "opened"
        );
        assert_eq!(
            open_evidence_epoch(root, &source).unwrap()["status"],
            "already_open_identical"
        );
        let mut changed = epoch;
        changed.start_wave += 1;
        atomic_json(&source, &serde_json::to_value(&changed).unwrap()).unwrap();
        assert!(open_evidence_epoch(root, &source).is_err());
    }

    #[test]
    fn external_evaluation_publishes_only_after_consumption_and_family_checks() {
        use jope_dime_regressor::selfplay_shadow::{
            DatasetOriginV2, PanelPoolKindV2, SealedPanelEntryV2,
            SEALED_PANEL_CONSUMPTION_V1_SCHEMA, SEALED_PANEL_POOL_V2_SCHEMA,
        };

        let temporary = tempdir().unwrap();
        let root = temporary.path();
        let parent = digest("parent");
        let candidate = digest("candidate");
        atomic_json(
            &root.join("training/accepted-by-task.json"),
            &json!({
                "tasks": {
                    "binary": {"checkpoint_sha256": parent},
                    "regression": {"checkpoint_sha256": digest("regression")},
                }
            }),
        )
        .unwrap();
        let pool = SealedPanelPoolV2 {
            schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
            entries: (0..1_000)
                .map(|index| {
                    let kind = match index {
                        0..=699 => PanelPoolKindV2::Training,
                        700..=799 => PanelPoolKindV2::Development,
                        800..=949 => PanelPoolKindV2::Promotion,
                        _ => PanelPoolKindV2::Audit,
                    };
                    SealedPanelEntryV2 {
                        task: Task::Binary,
                        context_identity: format!("context-{index}"),
                        context_sha256: digest(&format!("context-{index}")),
                        source_family_sha256: digest(&format!("family-{index}")),
                        near_duplicate_cluster_sha256: digest(&format!("cluster-{index}")),
                        origin: DatasetOriginV2::Real,
                        pool: kind,
                        promotion_panel: (kind == PanelPoolKindV2::Promotion)
                            .then(|| "panel-a".to_owned()),
                    }
                })
                .collect(),
            created_at_ms: 1,
        };
        let pool_path = root.join("pool.json");
        atomic_json(&pool_path, &serde_json::to_value(&pool).unwrap()).unwrap();
        let panel = pool.promotion_panel(Task::Binary, "panel-a");
        let contexts = panel
            .iter()
            .take(128)
            .map(|entry| entry.context_sha256.clone())
            .collect::<Vec<_>>();
        let consumption = SealedPanelConsumptionV1 {
            schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
            task: Task::Binary,
            panel: "panel-a".to_owned(),
            candidate_checkpoint_sha256: candidate,
            run_index: 0,
            seed: 9,
            context_sha256: contexts.clone(),
            consumed_at_ms: 1,
        };
        let pending = root.join("consumption.json");
        atomic_json(&pending, &serde_json::to_value(&consumption).unwrap()).unwrap();
        let consumption_sha = sha256_file(&pending).unwrap();
        let consumption_path = root
            .join("training/sealed-panels/consumed")
            .join(format!("{consumption_sha}.v1.json"));
        fs::create_dir_all(consumption_path.parent().unwrap()).unwrap();
        fs::rename(pending, &consumption_path).unwrap();
        let observations = contexts
            .into_iter()
            .enumerate()
            .map(|(index, context_sha256)| PromotionObservationV3 {
                context_sha256,
                source_family_sha256: panel[index].source_family_sha256.clone(),
                candidate_score: 1.02,
                accepted_score: 1.0,
                gp_score: 1.0,
                candidate_conditional_advantage: 0.02,
                accepted_conditional_advantage: 0.01,
                valid: true,
                replay_verified: true,
                candidate_failed: false,
                accepted_failed: false,
                candidate_latency_ms: 100,
                accepted_latency_ms: 100,
                strict_win: index == 0,
                equal_budget_win: index == 1,
                slices: std::collections::BTreeMap::from([
                    ("origin".to_owned(), "real".to_owned()),
                    ("size".to_owned(), "small".to_owned()),
                    ("missingness".to_owned(), "none".to_owned()),
                    ("imbalance".to_owned(), "balanced".to_owned()),
                    ("target_shape".to_owned(), "balanced".to_owned()),
                ]),
            })
            .collect::<Vec<_>>();
        let observation_path = root.join("observations.json");
        atomic_json(
            &observation_path,
            &json!({
                "schema": "TaskPromotionObservationSet.v3",
                "panel_consumption_sha256": consumption_sha,
                "observations": observations,
            }),
        )
        .unwrap();
        let result = evaluate_task_promotion(
            root,
            &pool_path,
            &consumption_path,
            &observation_path,
            parent.as_str(),
            7,
        )
        .unwrap();
        assert_eq!(result["accepted"], true);
        assert!(Path::new(result["evaluation"].as_str().unwrap()).is_file());
    }
}
