use super::*;

pub(crate) fn slice_data(
    data: &RegressionData,
    indices: &[usize],
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut x = Vec::with_capacity(indices.len());
    let mut y = Vec::with_capacity(indices.len());
    for &index in indices {
        x.push(
            data.x
                .get(index)
                .with_context(|| format!("row {index} outside training data"))?
                .clone(),
        );
        y.push(data.y[index]);
    }
    Ok((x, y))
}

struct StudySplitSlices {
    train_x: Vec<Vec<f64>>,
    train_y: Vec<f64>,
    validation_x: Vec<Vec<f64>>,
    validation_y: Vec<f64>,
}

pub(super) struct StudyEvalContext {
    context_tag: String,
    folds: usize,
    slices: BTreeMap<String, StudySplitSlices>,
    stage_cache: StageCache,
}

impl StudyEvalContext {
    pub(super) fn new(
        data: &RegressionData,
        dataset: &DatasetStudyEntry,
        splits: &TrainOnlySplitManifest,
    ) -> Result<Self> {
        let mut cached = BTreeMap::new();
        for split in &splits.splits {
            let (train_x, train_y) = slice_data(data, &split.train_indices)?;
            let (validation_x, validation_y) = slice_data(data, &split.validation_indices)?;
            cached.insert(
                split.split_sha256.clone(),
                StudySplitSlices {
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                },
            );
        }
        Ok(Self {
            context_tag: format!(
                "{}|{}|rows={}|features={}",
                dataset.dataset_hash,
                dataset.train_sha256,
                data.y.len(),
                data.features
            ),
            folds: splits.splits.len(),
            slices: cached,
            stage_cache: StageCache::new(DEFAULT_BYTE_CAP),
        })
    }

    pub(super) fn begin_generation(&self, generation: usize) {
        self.stage_cache.begin_generation(generation as u64);
    }

    pub(super) fn sweep(&self) {
        self.stage_cache.sweep();
    }

    pub(super) fn evaluate(
        &self,
        genome: &GenomeIr,
        split: &SplitReceipt,
    ) -> Result<(f64, f64, f64, f64, f64)> {
        let slices = self
            .slices
            .get(&split.split_sha256)
            .context("split slice cache miss")?;
        let context = StageCacheCtx {
            context_tag: self.context_tag.clone(),
            folds: self.folds,
            fold: split.generation,
        };
        let fit_started = Instant::now();
        let model = fit_with_stage_cache(
            genome,
            &slices.train_x,
            &slices.train_y,
            &self.stage_cache,
            &context,
        )?;
        let fit_seconds = fit_started.elapsed().as_secs_f64();
        let predict_started = Instant::now();
        let predictions = model.predict(&slices.validation_x)?;
        let predict_seconds = predict_started.elapsed().as_secs_f64();
        let (rmse, r2, mae) = metrics(&slices.validation_y, &predictions)?;
        Ok((rmse, r2, mae, fit_seconds, predict_seconds))
    }
}

pub(super) fn evaluate_member_split(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    evaluator: &StudyEvalContext,
    member: &GenomeMember,
    split: &SplitReceipt,
) -> Result<(TrainOnlyFitnessObservation, ObservationLink)> {
    let key = cache_key(
        &dataset.train_sha256,
        &member.semantic_sha256,
        &split.split_sha256,
    )?;
    let path = root
        .join("evaluations")
        .join(&dataset.dataset_hash)
        .join(format!("{key}.v1.json"));
    if path.exists() {
        let (observation, digest): (TrainOnlyFitnessObservation, _) = read_receipt(&path)?;
        if observation.schema != "TrainOnlyFitnessObservation.v2"
            || observation.run_id != manifest.run_id
            || observation.dataset_hash != dataset.dataset_hash
            || observation.train_sha256 != dataset.train_sha256
            || observation.encoding_sha256 != dataset.encoding.context_sha256
            || observation.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
            || observation.semantic_sha256 != member.semantic_sha256
            || observation.split_generation != split.generation
            || observation.split_sha256 != split.split_sha256
            || observation.kernel_version != KERNEL_VERSION
            || observation.cache_key != key
        {
            bail!("evaluation cache identity drift: {}", path.display());
        }
        return Ok((
            observation,
            ObservationLink {
                split_generation: split.generation,
                split_sha256: split.split_sha256.clone(),
                receipt_path: absolute_path(&path)?,
                receipt_sha256: digest,
                cache_key: key,
                cache_hit: true,
            },
        ));
    }

    let mut observation = TrainOnlyFitnessObservation {
        schema: "TrainOnlyFitnessObservation.v2".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        encoding_sha256: dataset.encoding.context_sha256.clone(),
        checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        semantic_sha256: member.semantic_sha256.clone(),
        split_generation: split.generation,
        split_sha256: split.split_sha256.clone(),
        kernel_version: KERNEL_VERSION.to_string(),
        cache_key: key.clone(),
        r2: None,
        rmse: None,
        mae: None,
        fit_seconds: 0.0,
        predict_seconds: 0.0,
        failure_reason: None,
    };
    let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        if semantic_identity(&member.canonical_genome) != member.semantic_sha256
            || !genome_compatible(&member.canonical_genome, dataset.features)
        {
            bail!("canonical, semantic, or feature compatibility failure");
        }
        let genome = GenomeIr::parse(&member.canonical_genome)?;
        evaluator.evaluate(&genome, split)
    }));
    match evaluated {
        Ok(Ok((rmse, r2, mae, fit_seconds, predict_seconds))) => {
            observation.rmse = Some(rmse);
            observation.r2 = Some(r2);
            observation.mae = Some(mae);
            observation.fit_seconds = fit_seconds;
            observation.predict_seconds = predict_seconds;
        }
        Ok(Err(error)) => observation.failure_reason = Some(format!("{error:#}")),
        Err(payload) => {
            observation.failure_reason = Some(
                payload
                    .downcast_ref::<&str>()
                    .map(|value| format!("panic: {value}"))
                    .or_else(|| {
                        payload
                            .downcast_ref::<String>()
                            .map(|value| format!("panic: {value}"))
                    })
                    .unwrap_or_else(|| "panic: non-string payload".to_string()),
            )
        }
    }
    let digest = write_receipt(&path, &observation)?;
    Ok((
        observation,
        ObservationLink {
            split_generation: split.generation,
            split_sha256: split.split_sha256.clone(),
            receipt_path: absolute_path(&path)?,
            receipt_sha256: digest,
            cache_key: key,
            cache_hit: false,
        },
    ))
}

// Keep population identity, folds and search budget explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn score_population(
    root: &Path,
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
    evaluator: &StudyEvalContext,
    splits: &TrainOnlySplitManifest,
    generation: usize,
    population: &[GenomeMember],
    pool: &ThreadPool,
) -> Result<Vec<ScoredMember>> {
    if population.len() != POPULATION {
        bail!("generation {generation} population is not exactly {POPULATION}");
    }
    let mut semantics = HashSet::new();
    if population
        .iter()
        .any(|member| !semantics.insert(member.semantic_sha256.as_str()))
    {
        bail!("generation {generation} population repeats canonical semantics");
    }
    let mut scored = pool.install(|| {
        population
            .par_iter()
            .map(|member| -> Result<ScoredMember> {
                let mut observations = Vec::with_capacity(generation + 1);
                let mut r2 = Vec::with_capacity(generation + 1);
                let mut failure = None;
                for split in &splits.splits[..=generation] {
                    let (observation, mut link) =
                        evaluate_member_split(root, manifest, dataset, evaluator, member, split)?;
                    // Ledger cache accounting is logical rather than dependent on
                    // whether a prior interrupted attempt happened to materialize the
                    // same immutable observation before the generation barrier.
                    link.cache_hit =
                        member.ancestry.operation == "elite" && split.generation < generation;
                    observations.push(link);
                    if let Some(score) = observation.r2 {
                        r2.push(score);
                    } else {
                        failure = observation.failure_reason.clone();
                    }
                }
                let robust_fitness = if failure.is_none() && r2.len() == generation + 1 {
                    Some(paired_robust_fitness(&r2)?)
                } else {
                    None
                };
                let worst_split_r2 = robust_fitness.map(|_| {
                    r2.iter()
                        .copied()
                        .min_by(f64::total_cmp)
                        .expect("nonempty paired history")
                });
                Ok(ScoredMember {
                    member: member.clone(),
                    observations,
                    split_r2: r2,
                    robust_fitness,
                    worst_split_r2,
                    failure_reason: failure,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    scored.sort_by(rank_scored);
    Ok(scored)
}

#[derive(Clone, Debug)]
pub(super) struct DeterministicRng(u64);

impl DeterministicRng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

pub(crate) fn offspring_seed(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let material =
        format!("train-only-offspring-v1\0{run_id}\0{dataset_hash}\0{generation}\0{slot}\0{retry}");
    let digest = Sha256::digest(material.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

pub(crate) fn tournament<'a>(
    rng: &mut DeterministicRng,
    ranked: &'a [ScoredMember],
) -> &'a ScoredMember {
    let mut winner = (rng.next_u64() as usize) % PARENT_POOL;
    for _ in 1..TOURNAMENT {
        winner = winner.min((rng.next_u64() as usize) % PARENT_POOL);
    }
    &ranked[winner]
}

pub(super) fn refill_population(
    run_id: &str,
    dataset_hash: &str,
    features: usize,
    generation: usize,
    ranked: &[ScoredMember],
    campaign_history: &mut HashSet<String>,
) -> Result<Vec<GenomeMember>> {
    if ranked
        .iter()
        .filter(|member| member.robust_fitness.is_some())
        .count()
        < PARENT_POOL
    {
        bail!("cannot refill from fewer than {PARENT_POOL} rankable parents");
    }
    let mut next = Vec::with_capacity(POPULATION);
    let mut generation_semantics = HashSet::new();
    for elite in ranked.iter().take(ELITES) {
        generation_semantics.insert(elite.member.semantic_sha256.clone());
        let mut member = elite.member.clone();
        member.slot = next.len();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![member.semantic_sha256.clone()],
            roots: member.ancestry.roots.clone(),
            retry_count: 0,
        };
        next.push(member);
    }
    for slot in ELITES..POPULATION {
        let mut accepted = None;
        for retry in 0..100_000 {
            let mut rng = DeterministicRng::new(offspring_seed(
                run_id,
                dataset_hash,
                generation,
                slot,
                retry,
            ));
            let crossover = rng.next_u64().is_multiple_of(4);
            let parent_a = tournament(&mut rng, ranked);
            let (operation, parents, roots, child) = if crossover {
                let parent_b = tournament(&mut rng, ranked);
                let Ok(child) = crossover_canonical_pipelines(
                    &parent_a.member.canonical_genome,
                    &parent_b.member.canonical_genome,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                ) else {
                    continue;
                };
                let mut roots = parent_a.member.ancestry.roots.clone();
                roots.extend(parent_b.member.ancestry.roots.iter().cloned());
                (
                    "crossover".to_string(),
                    vec![
                        parent_a.member.semantic_sha256.clone(),
                        parent_b.member.semantic_sha256.clone(),
                    ],
                    roots,
                    child,
                )
            } else {
                let Ok(child) = mutate_canonical_pipeline_v2(
                    &parent_a.member.canonical_genome,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                    true,
                ) else {
                    continue;
                };
                (
                    "mutation".to_string(),
                    vec![parent_a.member.semantic_sha256.clone()],
                    parent_a.member.ancestry.roots.clone(),
                    child,
                )
            };
            let Ok(genome) = GenomeIr::parse(&child) else {
                continue;
            };
            let canonical = genome.canonical();
            let semantic = semantic_identity(&canonical);
            if !matches!(genome, GenomeIr::Pipeline(_))
                || !genome_compatible(&canonical, features)
                || !generation_semantics.insert(semantic.clone())
            {
                continue;
            }
            if !campaign_history.insert(semantic.clone()) {
                generation_semantics.remove(&semantic);
                continue;
            }
            accepted = Some(GenomeMember {
                slot,
                canonical_genome: canonical,
                semantic_sha256: semantic,
                complexity: genome_complexity(&genome),
                source: format!("evolution_{operation}"),
                source_evidence: json!({
                    "generation": generation,
                    "slot": slot,
                    "operator_version": "canonical-v2",
                }),
                ancestry: Ancestry {
                    operation,
                    parents,
                    roots,
                    retry_count: retry,
                },
            });
            break;
        }
        next.push(accepted.with_context(|| {
            format!(
                "could not generate a campaign-unique child for generation {generation} slot {slot}"
            )
        })?);
    }
    Ok(next)
}
