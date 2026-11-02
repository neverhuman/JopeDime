use super::*;

pub(super) struct PopulationEvaluation {
    pub(super) cache_miss: bool,
    pub(super) cached: CachedEvaluation,
}

pub(super) fn evaluate_population(
    generation: usize,
    population: &[PopulationMember],
    train: &RegressionData,
    config: &CampaignConfig,
    artifact_root: &Path,
    cache: &HashMap<String, CachedEvaluation>,
    live_logger: &LiveLogger,
) -> Result<Vec<PopulationEvaluation>> {
    if config.rayon_threads < population.len() {
        bail!(
            "resolved runtime capacity {} cannot provide one thread for each of {} population members",
            config.rayon_threads,
            population.len()
        );
    }
    thread::scope(|scope| -> Result<Vec<PopulationEvaluation>> {
        let mut handles = Vec::with_capacity(population.len());
        for (slot, member) in population.iter().enumerate() {
            let thread_name = format!("jain-member-g{generation:02}-s{slot:02}");
            handles.push(
                thread::Builder::new()
                    .name(thread_name)
                    .spawn_scoped(scope, move || -> Result<PopulationEvaluation> {
                        if let Some(cached) = cache.get(&member.semantic_sha256) {
                            let evaluation = PopulationEvaluation {
                                cache_miss: false,
                                cached: cached.clone(),
                            };
                            live_logger.record_slot(
                                generation,
                                slot,
                                member,
                                &evaluation.cached,
                                false,
                            )?;
                            return Ok(evaluation);
                        }
                        let cached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            evaluate_unique(generation, slot, member, train, config, artifact_root)
                        }))
                        .unwrap_or_else(|payload| CachedEvaluation {
                            source_generation: generation,
                            source_slot: slot,
                            complexity: genome_complexity_text(&member.canonical_genome),
                            head_family: head_family_text(&member.canonical_genome),
                            artifacts: None,
                            result: None,
                            rejection_reason: Some(format!(
                                "panic: {}",
                                panic_message(payload.as_ref())
                            )),
                        });
                        let evaluation = PopulationEvaluation {
                            cache_miss: true,
                            cached,
                        };
                        live_logger.record_slot(
                            generation,
                            slot,
                            member,
                            &evaluation.cached,
                            true,
                        )?;
                        Ok(evaluation)
                    })
                    .with_context(|| {
                        format!(
                            "spawn dedicated member thread for generation {generation} slot {slot}"
                        )
                    })?,
            );
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|payload| {
                    anyhow::anyhow!(
                        "dedicated population member thread panicked: {}",
                        panic_message(payload.as_ref())
                    )
                })?
            })
            .collect()
    })
}

pub(super) fn evaluate_unique(
    generation: usize,
    slot: usize,
    member: &PopulationMember,
    train: &RegressionData,
    config: &CampaignConfig,
    artifact_root: &Path,
) -> CachedEvaluation {
    let rejected = |complexity, head_family, artifacts, reason: String| CachedEvaluation {
        source_generation: generation,
        source_slot: slot,
        complexity,
        head_family,
        artifacts,
        result: None,
        rejection_reason: Some(reason),
    };
    let genome = match GenomeIr::parse(&member.canonical_genome) {
        Ok(genome) if matches!(genome, GenomeIr::Pipeline(_)) => genome,
        Ok(_) => {
            return rejected(
                0,
                "non_pipeline".to_string(),
                None,
                "not a pipeline genome".to_string(),
            )
        }
        Err(error) => {
            return rejected(
                0,
                "parse_error".to_string(),
                None,
                format!("parse: {error:#}"),
            )
        }
    };
    let canonical = genome.canonical();
    let complexity = genome_complexity(&genome);
    let head_family = head_family(&genome);
    if canonical != member.canonical_genome
        || sha256_bytes(canonical.as_bytes()) != member.semantic_sha256
    {
        return rejected(
            complexity,
            head_family,
            None,
            "canonical or semantic identity mismatch".to_string(),
        );
    }
    let emitted = match emit(&genome, &Default::default()) {
        Ok(emitted) => emitted,
        Err(error) => return rejected(complexity, head_family, None, format!("emit: {error:#}")),
    };
    let artifacts = match publish_emitted(artifact_root, &member.semantic_sha256, &emitted) {
        Ok(artifacts) => artifacts,
        Err(error) => {
            return rejected(
                complexity,
                head_family,
                None,
                format!("publish emitted artifacts: {error:#}"),
            )
        }
    };
    match exact_cv_on_folds(
        &genome,
        train,
        config.cv_folds,
        &[QUICK_CV5_HOLDOUT_FOLD],
        &config.dataset,
    ) {
        Ok(result) => CachedEvaluation {
            source_generation: generation,
            source_slot: slot,
            complexity,
            head_family,
            artifacts: Some(artifacts),
            result: Some(result),
            rejection_reason: None,
        },
        Err(error) => rejected(
            complexity,
            head_family,
            Some(artifacts),
            format!("cv: {error:#}"),
        ),
    }
}

pub(super) fn exact_cv(
    genome: &GenomeIr,
    train: &RegressionData,
    folds: usize,
    dataset: &str,
) -> Result<ExactCvResult> {
    let evaluated_folds = (0..folds).collect::<Vec<_>>();
    exact_cv_on_folds(genome, train, folds, &evaluated_folds, dataset)
}

pub(super) fn exact_cv_on_folds(
    genome: &GenomeIr,
    train: &RegressionData,
    fold_plan_folds: usize,
    evaluated_folds: &[usize],
    dataset: &str,
) -> Result<ExactCvResult> {
    let n = train.x.len().min(train.y.len());
    let mut oof = vec![f64::NAN; n];
    let mut fold_r2 = Vec::with_capacity(evaluated_folds.len());
    let mut fit_seconds = Vec::with_capacity(evaluated_folds.len());
    let mut predict_seconds = Vec::with_capacity(evaluated_folds.len());
    if evaluated_folds.is_empty() || evaluated_folds.iter().any(|fold| *fold >= fold_plan_folds) {
        bail!("evaluated folds are empty or outside the fold plan");
    }
    for &fold in evaluated_folds {
        let mut train_x: Rows = Vec::new();
        let mut train_y = Vec::new();
        let mut valid_x: Rows = Vec::new();
        let mut valid_y = Vec::new();
        let mut valid_indices = Vec::new();
        for row in 0..n {
            if fold_of(dataset, row, fold_plan_folds) == fold {
                valid_x.push(train.x[row].clone());
                valid_y.push(train.y[row]);
                valid_indices.push(row);
            } else {
                train_x.push(train.x[row].clone());
                train_y.push(train.y[row]);
            }
        }
        if valid_x.is_empty() || train_x.len() < 8 {
            bail!("fold {fold} is empty or has fewer than eight training rows");
        }
        let fit_started = Instant::now();
        let model = fit(genome, &train_x, &train_y).with_context(|| format!("fit fold {fold}"))?;
        fit_seconds.push(fit_started.elapsed().as_secs_f64());
        let predict_started = Instant::now();
        let predictions = model.predict(&valid_x);
        predict_seconds.push(predict_started.elapsed().as_secs_f64());
        if predictions.len() != valid_y.len() || predictions.iter().any(|v| !v.is_finite()) {
            bail!("fold {fold} produced non-finite or wrong-length predictions");
        }
        let score = independent_metrics(&valid_y, &predictions)?.1;
        if !score.is_finite() {
            bail!("fold {fold} R2 is non-finite");
        }
        fold_r2.push(score);
        for (index, prediction) in valid_indices.into_iter().zip(predictions) {
            oof[index] = prediction;
        }
    }
    if fold_r2.len() != evaluated_folds.len() {
        bail!("CV did not evaluate every requested holdout fold");
    }
    let mut oof_truth = Vec::new();
    let mut oof_predictions = Vec::new();
    for (row, prediction) in oof.into_iter().enumerate() {
        if prediction.is_finite() {
            oof_truth.push(train.y[row]);
            oof_predictions.push(prediction);
        }
    }
    if oof_truth.is_empty() {
        bail!("CV produced no held-out predictions");
    }
    let (pooled_oof_rmse, pooled_oof_r2, _) = independent_metrics(&oof_truth, &oof_predictions)?;
    let robust_fitness = robust_fitness(pooled_oof_r2, &fold_r2);
    if !pooled_oof_r2.is_finite() || !robust_fitness.is_finite() {
        bail!("pooled OOF or robust fitness is non-finite");
    }
    Ok(ExactCvResult {
        fold_r2,
        pooled_oof_r2,
        pooled_oof_rmse,
        robust_fitness,
        fit_seconds,
        predict_seconds,
    })
}

pub(super) fn refill_population(
    config: &CampaignConfig,
    generation: usize,
    ranked: &[SlotLedger],
) -> Result<Vec<PopulationMember>> {
    if ranked.len() < config.parent_pool || config.elites >= config.population {
        bail!("cannot refill exact population from configured parent pool/elites");
    }
    let mut next = Vec::with_capacity(config.population);
    let mut seen = HashSet::with_capacity(config.population);
    for elite in ranked.iter().take(config.elites) {
        seen.insert(elite.semantic_sha256.clone());
        next.push(PopulationMember {
            canonical_genome: elite.canonical_genome.clone(),
            semantic_sha256: elite.semantic_sha256.clone(),
            ancestry: Ancestry {
                operation: "elite".to_string(),
                parents: vec![elite.semantic_sha256.clone()],
                retry_count: 0,
            },
        });
    }
    for slot in config.elites..config.population {
        let mut retry = 0usize;
        loop {
            if retry >= 1_000_000 {
                bail!("could not generate a canonical-unique child for generation {generation} slot {slot}");
            }
            let mut rng =
                DeterministicRng::new(offspring_seed(config.seed, generation, slot, retry));
            let crossover = rng.next_u64() % 4 == 0;
            let parent_a =
                tournament_winner(&mut rng, ranked, config.parent_pool, config.tournament);
            let (operation, parents, child) = if crossover {
                let parent_b =
                    tournament_winner(&mut rng, ranked, config.parent_pool, config.tournament);
                let result = crossover_canonical_pipelines(
                    &parent_a.canonical_genome,
                    &parent_b.canonical_genome,
                    rng.next_u64(),
                    config.max_transforms,
                )
                .map_err(anyhow::Error::msg)?;
                (
                    "crossover".to_string(),
                    vec![
                        parent_a.semantic_sha256.clone(),
                        parent_b.semantic_sha256.clone(),
                    ],
                    result,
                )
            } else {
                let result = mutate_canonical_pipeline(
                    &parent_a.canonical_genome,
                    rng.next_u64(),
                    config.max_transforms,
                )
                .map_err(anyhow::Error::msg)?;
                (
                    "mutation".to_string(),
                    vec![parent_a.semantic_sha256.clone()],
                    result,
                )
            };
            let parsed = GenomeIr::parse(&child).context("parse generated exact-CV child")?;
            if !matches!(parsed, GenomeIr::Pipeline(_)) {
                retry += 1;
                continue;
            }
            let canonical = parsed.canonical();
            let semantic_sha256 = sha256_bytes(canonical.as_bytes());
            if !seen.insert(semantic_sha256.clone()) {
                retry += 1;
                continue;
            }
            next.push(PopulationMember {
                canonical_genome: canonical,
                semantic_sha256,
                ancestry: Ancestry {
                    operation,
                    parents,
                    retry_count: retry,
                },
            });
            break;
        }
    }
    validate_unique_population(&next, config.population)?;
    Ok(next)
}

pub(super) fn tournament_winner<'a>(
    rng: &mut DeterministicRng,
    ranked: &'a [SlotLedger],
    parent_pool: usize,
    tournament: usize,
) -> &'a SlotLedger {
    let mut winner = (rng.next_u64() as usize) % parent_pool;
    for _ in 1..tournament {
        winner = winner.min((rng.next_u64() as usize) % parent_pool);
    }
    &ranked[winner]
}

#[derive(Clone, Debug)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

pub(super) fn offspring_seed(base: u64, generation: usize, slot: usize, retry: usize) -> u64 {
    base ^ (generation as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ (slot as u64).wrapping_mul(0xa076_1d64_78bd_642f)
        ^ (retry as u64).wrapping_mul(0xe703_7ed1_a0b4_28db)
}

pub(super) fn rank_slots(left: &SlotLedger, right: &SlotLedger) -> Ordering {
    right
        .robust_fitness
        .unwrap()
        .total_cmp(&left.robust_fitness.unwrap())
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(super) fn rank_finalists(left: &&FinalistCv7, right: &&FinalistCv7) -> Ordering {
    let left_score = left.cv7.as_ref().unwrap().robust_fitness;
    let right_score = right.cv7.as_ref().unwrap().robust_fitness;
    right_score
        .total_cmp(&left_score)
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(super) fn selection_evidence_from_slot(
    slot: &SlotLedger,
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: &str,
) -> Result<SelectionEvidence> {
    if !slot.rankable() || slot.fold_r2.as_ref().unwrap().len() != evaluated_fold_ids.len() {
        bail!("quick-score finalist lacks complete held-out evidence");
    }
    Ok(SelectionEvidence {
        fold_plan_folds,
        evaluated_fold_ids,
        fold_manifest_sha256: fold_manifest_sha256.to_string(),
        fold_r2: slot.fold_r2.clone().unwrap(),
        pooled_oof_r2: slot.pooled_oof_r2.unwrap(),
        pooled_oof_rmse: slot.pooled_oof_rmse.unwrap(),
        robust_fitness: slot.robust_fitness.unwrap(),
        fit_seconds: slot.fold_fit_seconds.clone(),
        predict_seconds: slot.fold_predict_seconds.clone(),
    })
}

pub(super) fn selection_evidence_from_result(
    result: ExactCvResult,
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: &str,
) -> SelectionEvidence {
    SelectionEvidence {
        fold_plan_folds,
        evaluated_fold_ids,
        fold_manifest_sha256: fold_manifest_sha256.to_string(),
        fold_r2: result.fold_r2,
        pooled_oof_r2: result.pooled_oof_r2,
        pooled_oof_rmse: result.pooled_oof_rmse,
        robust_fitness: result.robust_fitness,
        fit_seconds: result.fit_seconds,
        predict_seconds: result.predict_seconds,
    }
}

pub(super) fn generation_summary<'a>(
    generation: usize,
    slots: &[SlotLedger],
    generation_zero_best: f64,
    incumbent_history: &'a [Incumbent],
    next_population: &'a [PopulationMember],
    terminal_slots: usize,
    unique_evaluations: usize,
    ledger_sha256: &'a str,
) -> Result<GenerationSummary<'a>> {
    let mut fitness = slots
        .iter()
        .filter_map(|slot| slot.rankable().then_some(slot.robust_fitness.unwrap()))
        .collect::<Vec<_>>();
    fitness.sort_by(f64::total_cmp);
    let best = fitness
        .last()
        .copied()
        .context("generation has no valid fitness")?;
    let incumbent = slots
        .iter()
        .filter(|slot| slot.rankable())
        .min_by(|left, right| rank_slots(left, right))
        .context("generation has no ranked incumbent")?;
    let incumbent_pooled_oof_rmse = incumbent.pooled_oof_rmse.unwrap();
    let median = quantile(&fitness, 0.5);
    let mut fitness_quantiles = BTreeMap::new();
    for (name, q) in [
        ("p00", 0.0),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p100", 1.0),
    ] {
        fitness_quantiles.insert(name.to_string(), quantile(&fitness, q));
    }
    let mut head_mix = BTreeMap::new();
    for slot in slots {
        *head_mix.entry(slot.head_family.clone()).or_insert(0) += 1;
    }
    let diversity = slots
        .iter()
        .map(|slot| slot.semantic_sha256.as_str())
        .collect::<HashSet<_>>()
        .len() as f64
        / slots.len() as f64;
    Ok(GenerationSummary {
        schema: SUMMARY_SCHEMA,
        generation,
        population: slots.len(),
        valid: fitness.len(),
        invalid: slots.len() - fitness.len(),
        cache_hits: slots
            .iter()
            .filter(|slot| slot.cache_disposition == "hit")
            .count(),
        cache_misses: slots
            .iter()
            .filter(|slot| slot.cache_disposition == "miss")
            .count(),
        fitness_quantiles,
        best_fitness: best,
        incumbent_pooled_oof_rmse,
        cv_rmse_signed_margin_vs_tabfm: TABFM_RMSE - incumbent_pooled_oof_rmse,
        cv_proxy_track_status: if incumbent_pooled_oof_rmse < TABFM_RMSE {
            "CV_PROXY_AHEAD"
        } else {
            "CV_PROXY_BEHIND"
        }
        .to_string(),
        median_fitness: median,
        improvement_from_generation_zero: best - generation_zero_best,
        canonical_diversity: diversity,
        head_mix,
        incumbent_history,
        next_population,
        cumulative_terminal_slots: terminal_slots,
        cumulative_unique_evaluations: unique_evaluations,
        cumulative_ledger_sha256: ledger_sha256,
    })
}
