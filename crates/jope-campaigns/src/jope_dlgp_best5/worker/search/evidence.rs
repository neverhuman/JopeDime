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

pub(super) fn observation_inputs(
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

pub(super) fn best5_ancestry(
    member: &PopulationMemberV1,
    predictor: &FrozenPredictorV1,
) -> AncestryV1 {
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

pub(super) fn evidence(
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

pub(super) fn rank_evidence(values: &[CandidateEvidenceV1]) -> Result<Vec<usize>, String> {
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

pub(super) fn portfolio_finalist_selection(
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

pub(super) fn next_population(
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

pub(super) fn cv7_evidence(
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
