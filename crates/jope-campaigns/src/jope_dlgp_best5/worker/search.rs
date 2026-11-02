use super::*;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::jope_generalized_gp::{
    raw_sha256, ArtifactStoreV1, CorpusEvalAdapter, DatasetTableV1, EvaluatorFreezeV1,
    FrozenArtifactV1, GenomeOriginV1, GenomeV1, PopulationMemberV1, RowSplitV1, SealedTestTableV1,
};
use crate::search_v3::{canonical_sha256, PredictionObjectV1};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ProposalCatalogV1<'a> {
    schema: &'static str,
    campaign_freeze_sha256: &'a str,
    task: &'static str,
    datasets: Vec<ProposalCatalogDatasetV1<'a>>,
}

#[derive(Serialize)]
struct ProposalCatalogDatasetV1<'a> {
    dataset_sha256: &'a str,
    training_path: String,
    training_sha256: &'a str,
    metadata_path: String,
    metadata_sha256: &'a str,
    feature_count: u32,
}

pub fn proposal_job_identity(campaign: &CampaignFreezeV1) -> Result<String, String> {
    Ok(raw_sha256(
        format!("best5-propose-all-v1|{}", campaign.identity()?).as_bytes(),
    ))
}

fn write_once(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            file.sync_all()
                .map_err(|error| format!("sync {}: {error}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let present = fs::read(path)
                .map_err(|read| format!("read existing {}: {read}", path.display()))?;
            if present == bytes {
                Ok(())
            } else {
                Err(format!(
                    "immutable worker input collision at {}",
                    path.display()
                ))
            }
        }
        Err(error) => Err(format!("create {}: {error}", path.display())),
    }
}

/// Executes the single all-dataset inference or loads the already-published
/// batch on retry. Once the job identity exists, the runtime is never invoked.
pub fn propose_all(
    campaign: &CampaignFreezeV1,
    artifact_root: &Path,
    work_root: &Path,
) -> Result<ProposalBatchFreezeV1, String> {
    campaign.validate()?;
    campaign
        .proposal_generator_artifact
        .verify("proposal generator")?;
    campaign
        .proposal_runtime_artifact
        .verify("proposal runtime")?;
    campaign.checkpoint.verify_artifact()?;
    let store = ArtifactStoreV1::new(artifact_root);
    let job_identity = proposal_job_identity(campaign)?;
    if store.contains(&job_identity)? {
        let batch: ProposalBatchFreezeV1 = store.load(&job_identity)?;
        batch.validate(campaign)?;
        let canonical_identity = batch.identity(campaign)?;
        if !store.contains(&canonical_identity)? {
            // A retry may observe the publish-once job alias after a crash
            // immediately before canonical indexing. Repair that index from
            // the validated published bytes; never rerun inference.
            store.publish(&canonical_identity, &batch)?;
        }
        return Ok(batch);
    }
    let campaign_sha256 = campaign.identity()?;
    let directory = work_root.join(&job_identity[..16]);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create proposal work directory: {error}"))?;
    let catalog = ProposalCatalogV1 {
        schema: "JopeDlgpBest5ProposalCatalog.v1",
        campaign_freeze_sha256: &campaign_sha256,
        task: "regression",
        datasets: campaign
            .datasets
            .iter()
            .map(|dataset| ProposalCatalogDatasetV1 {
                dataset_sha256: &dataset.dataset_sha256,
                training_path: Path::new(&campaign.sotarena_root)
                    .join(&dataset.train_relative_path)
                    .to_string_lossy()
                    .into_owned(),
                training_sha256: &dataset.train_sha256,
                metadata_path: Path::new(&campaign.sotarena_root)
                    .join(&dataset.metadata_relative_path)
                    .to_string_lossy()
                    .into_owned(),
                metadata_sha256: &dataset.metadata_sha256,
                feature_count: dataset.feature_count,
            })
            .collect(),
    };
    let catalog_path = directory.join("train-only-catalog.json");
    write_once(
        &catalog_path,
        &crate::search_v3::canonical_json_bytes(&catalog)?,
    )?;
    let output_path = directory.join("proposal-batch.json");
    if output_path.exists() {
        return Err("unpublished proposal output collision requires operator review".into());
    }
    let output = Command::new(&campaign.proposal_runtime_artifact.path)
        .args([
            "--generator",
            &campaign.proposal_generator_artifact.path,
            "--checkpoint",
            &campaign.checkpoint.artifact_path,
            "--catalog",
            catalog_path
                .to_str()
                .ok_or_else(|| "proposal catalog path is not UTF-8".to_string())?,
            "--output",
            output_path
                .to_str()
                .ok_or_else(|| "proposal output path is not UTF-8".to_string())?,
            "--proposals-per-dataset",
            "64",
        ])
        .output()
        .map_err(|error| format!("launch frozen proposal runtime: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "frozen proposal runtime failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut batch: ProposalBatchFreezeV1 = serde_json::from_slice(
        &fs::read(&output_path).map_err(|error| format!("read proposal output: {error}"))?,
    )
    .map_err(|error| format!("decode proposal output: {error}"))?;
    batch.normalize_completion_order(campaign);
    batch.validate(campaign)?;
    let canonical_identity = batch.identity(campaign)?;
    // Publish the deterministic job alias first. Its existence is the
    // durable no-rerun marker; canonical indexing is replay-safe afterward.
    store.publish(&job_identity, &batch)?;
    store.publish(&canonical_identity, &batch)?;
    Ok(batch)
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ObservationSnapshotV1 {
    Direct(Vec<AnchorObservationV1>),
    Wrapped {
        snapshot_sha256: String,
        observations: Vec<AnchorObservationV1>,
        #[serde(default)]
        diversity_seeds: Vec<GenomeV1>,
    },
}

fn observation_inputs(
    campaign: &CampaignFreezeV1,
    dataset: &FrozenDatasetV1,
) -> Result<(Vec<RankedAnchorV1>, Vec<GenomeV1>), String> {
    campaign
        .observation_snapshot_artifact
        .verify("observation snapshot")?;
    let snapshot: ObservationSnapshotV1 = serde_json::from_slice(
        &fs::read(&campaign.observation_snapshot_artifact.path)
            .map_err(|error| format!("read observation snapshot: {error}"))?,
    )
    .map_err(|error| format!("decode observation snapshot: {error}"))?;
    let (observations, explicit_seeds) = match snapshot {
        ObservationSnapshotV1::Direct(observations) => (observations, Vec::new()),
        ObservationSnapshotV1::Wrapped {
            snapshot_sha256,
            observations,
            diversity_seeds,
        } => {
            if snapshot_sha256 != campaign.observation_snapshot_sha256 {
                return Err(
                    "observation snapshot protocol ID differs from the campaign binding".into(),
                );
            }
            (observations, diversity_seeds)
        }
    };
    let ranked = rank_anchor_observations(
        &dataset.dataset_sha256,
        &dataset.source_family,
        &campaign.observation_snapshot_sha256,
        &observations,
    )?;
    let mut seeds = explicit_seeds;
    seeds.extend(
        observations
            .into_iter()
            .map(|observation| observation.genome),
    );
    let mut seen = BTreeSet::new();
    seeds.retain(|genome| seen.insert(genome.semantic_sha256.clone()));
    seeds.sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
    Ok((ranked, seeds))
}

fn best5_ancestry(member: &PopulationMemberV1, predictor: &FrozenPredictorV1) -> AncestryV1 {
    AncestryV1 {
        predictor: predictor.clone(),
        founder_semantic_sha256: member
            .ancestry
            .parents
            .first()
            .cloned()
            .unwrap_or_else(|| member.genome.semantic_sha256.clone()),
        parent_semantic_sha256s: member.ancestry.parents.clone(),
        model_seeded: matches!(member.ancestry.origin, GenomeOriginV1::ModelProposal),
    }
}

fn evidence(
    evaluated: &EvaluatedPredictorV1,
    ancestry: AncestryV1,
    split_index: u32,
    anchor_median: f64,
    targets: &[TargetRowV1],
) -> Result<CandidateEvidenceV1, String> {
    let (r2, rmse) = prediction_metrics(targets, &evaluated.prediction)?;
    Ok(CandidateEvidenceV1 {
        predictor: evaluated.predictor.clone(),
        ancestry,
        prediction_receipt_sha256: evaluated.receipt.identity()?,
        split_index,
        r2_clipped: r2,
        anchor_median_r2_clipped: anchor_median,
        paired_uplift: r2 - anchor_median,
        rmse,
        runtime_seconds: evaluated.runtime_seconds,
        valid: evaluated.prediction.valid,
        failure: (!evaluated.prediction.valid).then_some(FailureCodeV1::CandidateTimeout),
    })
}

fn rank_evidence(values: &[CandidateEvidenceV1]) -> Result<Vec<usize>, String> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    let mut scores = Vec::with_capacity(values.len());
    for value in values {
        value.validate()?;
        scores.push(confidence_score(&[value.paired_uplift])?);
    }
    indices.sort_by(|left, right| {
        values[*right]
            .valid
            .cmp(&values[*left].valid)
            .then_with(|| scores[*right].total_cmp(&scores[*left]))
            .then_with(|| values[*left].rmse.total_cmp(&values[*right].rmse))
            .then_with(|| {
                values[*left]
                    .predictor
                    .complexity()
                    .cmp(&values[*right].predictor.complexity())
            })
            .then_with(|| {
                values[*left]
                    .predictor
                    .semantic_sha256()
                    .ok()
                    .cmp(&values[*right].predictor.semantic_sha256().ok())
            })
    });
    Ok(indices)
}

fn portfolio_finalist_selection(
    archive: &[ArchiveMemberV1],
) -> Result<(Vec<FrozenPredictorV1>, BTreeMap<String, PortfolioLaneV1>), String> {
    let mut scored = archive
        .iter()
        .map(|member| {
            let semantic = member.predictor.semantic_sha256()?;
            let uplifts = member
                .split_evidence
                .iter()
                .map(|evidence| evidence.paired_uplift)
                .collect::<Vec<_>>();
            Ok((
                semantic,
                member.predictor.clone(),
                confidence_score(&uplifts)?,
                uplifts.iter().sum::<f64>() / uplifts.len() as f64,
                uplifts.iter().copied().min_by(f64::total_cmp).unwrap(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut selected = BTreeSet::new();
    let mut lanes = BTreeMap::new();
    let mut predictors = Vec::with_capacity(10);
    for lane in [
        PortfolioLaneV1::ConfidenceLcb,
        PortfolioLaneV1::CumulativeMean,
        PortfolioLaneV1::WorstSplit,
    ] {
        scored.sort_by(|left, right| {
            let (left_score, right_score) = match lane {
                PortfolioLaneV1::ConfidenceLcb => (left.2, right.2),
                PortfolioLaneV1::CumulativeMean => (left.3, right.3),
                PortfolioLaneV1::WorstSplit => (left.4, right.4),
                PortfolioLaneV1::SemanticNovelty => unreachable!(),
            };
            right_score
                .total_cmp(&left_score)
                .then_with(|| left.0.cmp(&right.0))
        });
        let leaders = scored
            .iter()
            .filter(|value| !selected.contains(&value.0))
            .take(2)
            .map(|value| value.1.clone())
            .collect::<Vec<_>>();
        for predictor in leaders {
            let semantic = predictor.semantic_sha256()?;
            selected.insert(semantic.clone());
            lanes.insert(semantic, lane);
            predictors.push(predictor);
        }
    }
    for _ in 0..2 {
        let candidate = scored
            .iter()
            .filter(|value| !selected.contains(&value.0))
            .max_by(|left, right| {
                let novelty = |semantic: &str| {
                    let bytes = hex::decode(semantic).unwrap();
                    selected
                        .iter()
                        .map(|chosen| {
                            bytes
                                .iter()
                                .zip(hex::decode(chosen).unwrap())
                                .map(|(left, right)| (left ^ right).count_ones())
                                .sum::<u32>()
                        })
                        .min()
                        .unwrap_or(256)
                };
                novelty(&left.0)
                    .cmp(&novelty(&right.0))
                    .then_with(|| right.0.cmp(&left.0))
            })
            .ok_or_else(|| "portfolio semantic-novelty lane exhausted".to_string())?;
        selected.insert(candidate.0.clone());
        lanes.insert(candidate.0.clone(), PortfolioLaneV1::SemanticNovelty);
        predictors.push(candidate.1.clone());
    }
    scored.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    predictors.extend(
        scored
            .iter()
            .filter(|value| !selected.contains(&value.0))
            .take(2)
            .map(|value| value.1.clone()),
    );
    if predictors.len() != 10 || lanes.len() != 8 {
        return Err("portfolio lanes did not produce eight leaders plus two common ranks".into());
    }
    Ok((predictors, lanes))
}

fn next_population(
    arm: Best5ArmV1,
    ranked: &[PopulationMemberV1],
    generation: u32,
    seed: u64,
) -> Result<Vec<PopulationMemberV1>, String> {
    let genomes = ranked
        .iter()
        .map(|member| member.genome.clone())
        .collect::<Vec<_>>();
    match arm {
        Best5ArmV1::HeadFamilyMapElites => {
            return build_map_elites_offspring(&genomes, generation, seed);
        }
        Best5ArmV1::SurrogateUcbOffspringSearch => {
            return build_surrogate_offspring(&genomes, generation, seed);
        }
        _ => {}
    }
    build_control_offspring(&genomes, generation, seed)
}

fn cv7_evidence(
    predictor: FrozenPredictorV1,
    evaluated: Vec<EvaluatedPredictorV1>,
    targets: &[TargetRowV1],
) -> Result<Cv7EvidenceV1, String> {
    let semantic_sha256 = predictor.semantic_sha256()?;
    let mut fold_r2 = Vec::with_capacity(7);
    let mut receipts = Vec::with_capacity(7);
    let mut actual = Vec::new();
    let mut predicted = Vec::new();
    let target_map = targets
        .iter()
        .map(|row| (row.stable_row_id.as_str(), row.target))
        .collect::<BTreeMap<_, _>>();
    let valid = evaluated.iter().all(|value| value.prediction.valid);
    for value in evaluated {
        fold_r2.push(prediction_metrics(targets, &value.prediction)?.0);
        receipts.push(value.receipt.identity()?);
        for (row, bits) in value
            .prediction
            .validation_row_ids
            .iter()
            .zip(&value.prediction.prediction_f64_bits)
        {
            actual.push(target_map[row.as_str()]);
            predicted.push(f64::from_bits(*bits));
        }
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let sse = actual
        .iter()
        .zip(&predicted)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>();
    let sst = actual
        .iter()
        .map(|actual| (actual - mean).powi(2))
        .sum::<f64>();
    Ok(Cv7EvidenceV1 {
        semantic_sha256,
        predictor: predictor.clone(),
        fold_prediction_receipt_sha256s: receipts,
        fold_r2,
        pooled_oof_r2: if sst == 0.0 { -1.0 } else { 1.0 - sse / sst }.clamp(-1.0, 1.0),
        rmse: (sse / actual.len() as f64).sqrt(),
        runtime_seconds: 0.0,
        complexity: predictor.complexity(),
        valid,
    })
}

/// Runs one isolated dataset × arm × seed search. Every generation evaluates
/// exactly 80 members on its matching split; archive and CV7 backfills are
/// separate logical operations even when their prediction object is cached.
pub fn run_search_one(
    campaign: &CampaignFreezeV1,
    proposal_batch: &ProposalBatchFreezeV1,
    dataset_id: &str,
    arm: Best5ArmV1,
    master_seed: u64,
    artifact_root: &Path,
    work_root: &Path,
) -> Result<SearchBundleV1, String> {
    campaign.validate()?;
    proposal_batch.validate(campaign)?;
    if !SEARCH_SEEDS.contains(&master_seed) {
        return Err("search-one seed must be 101 or 202".into());
    }
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| format!("dataset {dataset_id} is not in the campaign"))?;
    let proposal = proposal_batch
        .proposals
        .iter()
        .find(|proposal| proposal.dataset_sha256 == dataset.dataset_sha256)
        .ok_or_else(|| "proposal batch omitted the search dataset".to_string())?;
    let training = build_training_table(campaign, dataset_id)?;
    let split_plan = build_search_split_plan(
        &dataset.dataset_sha256,
        &dataset.train_sha256,
        master_seed,
        &training.target_rows,
    )?;
    let cv7_plan = build_cv7_plan(
        &dataset.dataset_sha256,
        &dataset.train_sha256,
        master_seed,
        &training.target_rows,
    )?;
    let (ranked_anchors, diversity_seeds) = observation_inputs(campaign, dataset)?;
    let generation_zero = build_generation_zero(proposal, &ranked_anchors, &diversity_seeds)?;
    let anchors = generation_zero
        .population
        .iter()
        .filter(|member| member.ancestry.origin == GenomeOriginV1::GlobalGene)
        .map(|member| FrozenPredictorV1::single(member.genome.clone()))
        .collect::<Vec<_>>();
    if anchors.len() != 10 {
        return Err("search-one lacks ten target-excluded anchors".into());
    }
    let mut context = Best5EvaluationContext::new(campaign, artifact_root, work_root)?;
    let mut population = generation_zero.population;
    let mut populations = Vec::with_capacity(GENERATIONS);
    let mut anchor_receipt_sets = Vec::with_capacity(SEARCH_SPLITS);
    let mut last_ranked = Vec::new();
    let mut frozen_residual_base = None;
    let mut lineage_founders = Vec::new();
    let mut lineage_credits = Vec::new();
    for generation in 0..GENERATIONS {
        let split = &split_plan.splits[generation];
        let anchor_evaluated = context.evaluate_search_split(&training.table, split, &anchors)?;
        let mut anchor_r2 = anchor_evaluated
            .iter()
            .map(|value| prediction_metrics(&training.target_rows, &value.prediction).map(|v| v.0))
            .collect::<Result<Vec<_>, _>>()?;
        anchor_r2.sort_by(f64::total_cmp);
        let anchor_median = 0.5 * (anchor_r2[4] + anchor_r2[5]);
        anchor_receipt_sets.push(AnchorReceiptSetV1 {
            split_index: generation as u32,
            prediction_receipt_sha256s: anchor_evaluated
                .iter()
                .map(|value| value.receipt.identity())
                .collect::<Result<Vec<_>, _>>()?,
            median_r2_clipped: anchor_median,
        });
        let lineage_baselines = if arm == Best5ArmV1::EvolvabilityUcbLineages && generation > 0 {
            let evaluated =
                context.evaluate_search_split(&training.table, split, &lineage_founders)?;
            evaluated
                .iter()
                .map(|value| {
                    Ok((
                        value.predictor.semantic_sha256()?,
                        prediction_metrics(&training.target_rows, &value.prediction)?.0,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, String>>()?
        } else {
            BTreeMap::new()
        };
        if arm == Best5ArmV1::CrossFittedResidualDlgp && generation == 4 {
            frozen_residual_base = last_ranked
                .first()
                .map(|member: &PopulationMemberV1| member.genome.clone());
            let base_semantic = frozen_residual_base
                .as_ref()
                .unwrap()
                .semantic_sha256
                .clone();
            if let Some(position) = population
                .iter()
                .position(|member| member.genome.semantic_sha256 == base_semantic)
            {
                let existing = population
                    .iter()
                    .map(|member| member.genome.semantic_sha256.clone())
                    .collect::<BTreeSet<_>>();
                let replacement = build_control_offspring(
                    &last_ranked
                        .iter()
                        .map(|member| member.genome.clone())
                        .collect::<Vec<_>>(),
                    generation as u32 + 100,
                    master_seed,
                )?
                .into_iter()
                .find(|member| {
                    member.genome.semantic_sha256 != base_semantic
                        && !existing.contains(&member.genome.semantic_sha256)
                })
                .ok_or_else(|| {
                    "could not remove the residual base from its residual population".to_string()
                })?;
                population[position] = replacement;
            }
        }
        let predictors = population
            .iter()
            .map(|member| {
                if arm == Best5ArmV1::CrossFittedResidualDlgp && generation >= 4 {
                    Ok(FrozenPredictorV1::ResidualComposite {
                        base: frozen_residual_base.clone().ok_or_else(|| {
                            "residual base was not frozen after generation 3".to_string()
                        })?,
                        residual_pipeline: member.genome.clone(),
                        eta: 1.0,
                    })
                } else {
                    Ok(FrozenPredictorV1::single(member.genome.clone()))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        let evaluated = context.evaluate_search_split(&training.table, split, &predictors)?;
        let evidence_values = evaluated
            .iter()
            .zip(&population)
            .map(|(value, member)| {
                evidence(
                    value,
                    best5_ancestry(member, &value.predictor),
                    generation as u32,
                    anchor_median,
                    &training.target_rows,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ranking = rank_evidence(&evidence_values)?;
        populations.push(GenerationPopulationV1 {
            generation: generation as u32,
            semantic_sha256s: predictors
                .iter()
                .map(FrozenPredictorV1::semantic_sha256)
                .collect::<Result<Vec<_>, _>>()?,
        });
        populations.last_mut().unwrap().semantic_sha256s.sort();
        last_ranked = ranking
            .iter()
            .map(|index| population[*index].clone())
            .collect();
        if arm == Best5ArmV1::EvolvabilityUcbLineages && generation == 0 {
            lineage_founders = select_head_diverse_founders(
                &ranking
                    .iter()
                    .map(|index| FounderScoreV1 {
                        genome: population[*index].genome.clone(),
                        confidence_score: confidence_score(
                            &[evidence_values[*index].paired_uplift],
                        )
                        .unwrap(),
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_iter()
            .map(FrozenPredictorV1::single)
            .collect();
            lineage_credits = lineage_founders
                .iter()
                .map(|predictor| {
                    Ok(LineageCreditV1 {
                        founder_semantic_sha256: predictor.semantic_sha256()?,
                        paired_improvements: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
        } else if arm == Best5ArmV1::EvolvabilityUcbLineages {
            let founder_ids = lineage_credits
                .iter()
                .map(|credit| credit.founder_semantic_sha256.clone())
                .collect::<BTreeSet<_>>();
            for (member, candidate) in population.iter().zip(&evidence_values) {
                let founder = if founder_ids.contains(&member.genome.semantic_sha256) {
                    member.genome.semantic_sha256.clone()
                } else {
                    member
                        .ancestry
                        .parents
                        .first()
                        .filter(|semantic| founder_ids.contains(*semantic))
                        .cloned()
                        .ok_or_else(|| {
                            "lineage population member lost its frozen founder binding".to_string()
                        })?
                };
                let baseline = lineage_baselines
                    .get(&founder)
                    .ok_or_else(|| "lineage founder lacks its paired split baseline".to_string())?;
                lineage_credits
                    .iter_mut()
                    .find(|credit| credit.founder_semantic_sha256 == founder)
                    .unwrap()
                    .paired_improvements
                    .push(candidate.r2_clipped - baseline);
            }
        }
        if generation + 1 < GENERATIONS {
            population = if arm == Best5ArmV1::EvolvabilityUcbLineages {
                let founders = lineage_founders
                    .iter()
                    .map(|predictor| {
                        predictor
                            .pipelines()
                            .first()
                            .copied()
                            .cloned()
                            .ok_or_else(|| "lineage founder is not a single pipeline".to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                build_lineage_offspring(
                    &last_ranked,
                    &founders,
                    &lineage_credits,
                    generation as u32 + 1,
                    master_seed,
                )?
            } else {
                next_population(arm, &last_ranked, generation as u32 + 1, master_seed)?
            };
        }
        if arm == Best5ArmV1::CrossFittedResidualDlgp && generation >= 4 {
            let base = FrozenPredictorV1::single(frozen_residual_base.clone().unwrap());
            for _ in 0..6 {
                context.evaluate_search_split(
                    &training.table,
                    split,
                    std::slice::from_ref(&base),
                )?;
            }
        }
    }
    if arm == Best5ArmV1::EvolvabilityUcbLineages && lineage_founders.len() != 20 {
        return Err("lineage arm did not freeze 20 split-0 founders".into());
    }
    let archive_predictors = last_ranked
        .iter()
        .take(20)
        .map(|member| {
            if arm == Best5ArmV1::CrossFittedResidualDlgp {
                FrozenPredictorV1::ResidualComposite {
                    base: frozen_residual_base.clone().unwrap(),
                    residual_pipeline: member.genome.clone(),
                    eta: 1.0,
                }
            } else {
                FrozenPredictorV1::single(member.genome.clone())
            }
        })
        .collect::<Vec<_>>();
    let archive_ancestries = archive_predictors
        .iter()
        .zip(&last_ranked)
        .map(|(predictor, member)| best5_ancestry(member, predictor))
        .collect::<Vec<_>>();
    let mut archive_evidence = (0..20)
        .map(|_| Vec::with_capacity(SEARCH_SPLITS))
        .collect::<Vec<_>>();
    for split in &split_plan.splits {
        let evaluated =
            context.evaluate_search_split(&training.table, split, &archive_predictors)?;
        for (index, value) in evaluated.iter().enumerate() {
            archive_evidence[index].push(evidence(
                value,
                archive_ancestries[index].clone(),
                split.split_index,
                anchor_receipt_sets[split.split_index as usize].median_r2_clipped,
                &training.target_rows,
            )?);
        }
    }
    let mut archive = archive_predictors
        .into_iter()
        .zip(archive_ancestries)
        .zip(archive_evidence)
        .map(|((predictor, ancestry), split_evidence)| ArchiveMemberV1 {
            predictor,
            ancestry,
            split_evidence,
        })
        .collect::<Vec<_>>();
    archive.sort_by(|left, right| {
        left.predictor
            .semantic_sha256()
            .ok()
            .cmp(&right.predictor.semantic_sha256().ok())
    });
    let mut ranked_archive = archive
        .iter()
        .map(|member| {
            Ok((
                confidence_score(
                    &member
                        .split_evidence
                        .iter()
                        .map(|evidence| evidence.paired_uplift)
                        .collect::<Vec<_>>(),
                )?,
                member.predictor.clone(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    ranked_archive.sort_by(|left, right| {
        right.0.total_cmp(&left.0).then_with(|| {
            left.1
                .semantic_sha256()
                .ok()
                .cmp(&right.1.semantic_sha256().ok())
        })
    });
    let (mut finalist_predictors, portfolio_lanes) = if arm == Best5ArmV1::SparseOofPortfolioGp {
        portfolio_finalist_selection(&archive)?
    } else {
        (
            ranked_archive
                .into_iter()
                .take(10)
                .map(|value| value.1)
                .collect::<Vec<_>>(),
            BTreeMap::new(),
        )
    };
    let mut by_finalist = finalist_predictors
        .iter()
        .map(|predictor| (predictor.semantic_sha256().unwrap(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for split in &cv7_plan.0.splits {
        let evaluated =
            context.evaluate_search_split(&training.table, split, &finalist_predictors)?;
        for value in evaluated {
            by_finalist
                .get_mut(&value.predictor.semantic_sha256()?)
                .unwrap()
                .push(value);
        }
    }
    if arm == Best5ArmV1::SparseOofPortfolioGp {
        let lane_predictors = finalist_predictors[..8].to_vec();
        let target_map = training
            .target_rows
            .iter()
            .map(|row| (row.stable_row_id.as_str(), row.target))
            .collect::<BTreeMap<_, _>>();
        let reference = by_finalist
            .get(&lane_predictors[0].semantic_sha256()?)
            .unwrap();
        let actual = reference
            .iter()
            .flat_map(|value| value.prediction.validation_row_ids.iter())
            .map(|row| target_map[row.as_str()])
            .collect::<Vec<_>>();
        let portfolio_members = lane_predictors
            .iter()
            .map(|predictor| {
                let semantic = predictor.semantic_sha256()?;
                Ok(PortfolioMemberV1 {
                    semantic_sha256: semantic.clone(),
                    lane: portfolio_lanes[&semantic],
                    oof_predictions: by_finalist[&semantic]
                        .iter()
                        .flat_map(|value| value.prediction.prediction_f64_bits.iter())
                        .map(|bits| f64::from_bits(*bits))
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let fit = fit_sparse_portfolio(&actual, &portfolio_members)?;
        let pipelines = fit
            .semantic_sha256s
            .iter()
            .map(|semantic| {
                lane_predictors
                    .iter()
                    .find(|predictor| predictor.semantic_sha256().as_ref() == Ok(semantic))
                    .and_then(|predictor| predictor.pipelines().first().copied().cloned())
                    .ok_or_else(|| "portfolio fit selected an unknown lane finalist".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let portfolio = FrozenPredictorV1::SparsePortfolio {
            pipelines,
            weights: fit.weights,
        };
        let replaced_semantic = lane_predictors[0].semantic_sha256()?;
        let portfolio_folds = (0..7)
            .map(|fold| {
                let components = lane_predictors
                    .iter()
                    .map(|predictor| Ok(by_finalist[&predictor.semantic_sha256()?][fold].clone()))
                    .collect::<Result<Vec<_>, String>>()?;
                let replaced_receipt = components[0].receipt.clone();
                context.replace_with_derived_predictor(
                    &training.table,
                    portfolio.clone(),
                    &components,
                    &replaced_receipt,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        by_finalist.remove(&replaced_semantic);
        by_finalist.insert(portfolio.semantic_sha256()?, portfolio_folds);
        finalist_predictors[0] = portfolio;
    }
    let mut finalists = finalist_predictors
        .into_iter()
        .map(|predictor| {
            let evaluated = by_finalist.remove(&predictor.semantic_sha256()?).unwrap();
            cv7_evidence(predictor, evaluated, &training.target_rows)
        })
        .collect::<Result<Vec<_>, String>>()?;
    finalists.sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
    let winner = finalists
        .iter()
        .filter(|candidate| candidate.valid)
        .min_by(|left, right| {
            right
                .robust_score()
                .total_cmp(&left.robust_score())
                .then_with(|| left.rmse.total_cmp(&right.rmse))
                .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
                .then_with(|| left.complexity.cmp(&right.complexity))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        })
        .ok_or_else(|| "all CV7 finalists are invalid".to_string())?;
    let winner_semantic_sha256 = winner.semantic_sha256.clone();
    let arm_specific_operations = if arm == Best5ArmV1::CrossFittedResidualDlgp {
        24
    } else if arm == Best5ArmV1::EvolvabilityUcbLineages {
        140
    } else {
        0
    };
    let ledger = BudgetLedgerV1 {
        logical_operations: 950 + arm_specific_operations,
        predicted_cpu_seconds: 0,
        mandatory_anchor_evaluations: 80,
        completed_anchor_evaluations: 80,
        mandatory_search_evaluations: 640,
        completed_search_evaluations: 640,
        mandatory_archive_evaluations: 160,
        completed_archive_evaluations: 160,
        mandatory_cv7_evaluations: 70,
        completed_cv7_evaluations: 70,
        arm_specific_operations,
    };
    let result = SearchResultV1 {
        schema: SearchResultV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity()?,
        proposal_freeze_sha256: proposal_batch.identity(campaign)?,
        arm,
        dataset_sha256: dataset.dataset_sha256.clone(),
        master_seed,
        split_plan_sha256: split_plan.identity()?,
        populations,
        archive,
        cv7_plan_sha256: cv7_plan.identity()?,
        finalists,
        winner_semantic_sha256,
        ledger,
        test_data_accessed: false,
        ranking_state_shared: false,
    };
    let mut bundle = SearchBundleV1 {
        schema: SearchBundleV1::SCHEMA.into(),
        staging: SearchStagingAttestationV1 {
            schema: SearchStagingAttestationV1::SCHEMA.into(),
            campaign_freeze_sha256: campaign.identity()?,
            proposal_batch_sha256: proposal_batch.identity(campaign)?,
            dataset_sha256: dataset.dataset_sha256.clone(),
            training: training.training_artifact,
            metadata: training.metadata_artifact,
            test_artifacts_staged: false,
        },
        target_rows: training.target_rows,
        split_plan,
        cv7_plan,
        anchor_receipt_sets,
        prediction_receipts: context.into_receipts(),
        result,
    };
    bundle.normalize_completion_order();
    bundle.validate()?;
    Ok(bundle)
}
