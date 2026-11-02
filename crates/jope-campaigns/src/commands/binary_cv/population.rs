use super::*;

pub(super) fn parse_binary_population(
    bytes: &[u8],
    expected: usize,
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<Vec<BinaryMember>> {
    let text = std::str::from_utf8(bytes).context("binary generation-zero file is not UTF-8")?;
    let mut population = Vec::new();
    let mut seen = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let genome = GenomeIr::parse(body).with_context(|| {
            format!(
                "binary generation-zero line {} does not parse",
                line_number + 1
            )
        })?;
        if !matches!(genome, GenomeIr::Pipeline(_)) {
            bail!(
                "binary generation-zero line {} is not a pipeline",
                line_number + 1
            );
        }
        let canonical = genome.canonical();
        if canonical != body {
            bail!(
                "binary generation-zero line {} is not canonical",
                line_number + 1
            );
        }
        validate_v3_genome(&genome, task, schema).with_context(|| {
            format!(
                "binary generation-zero line {} is not a valid binary genome",
                line_number + 1
            )
        })?;
        let semantic_sha256 = sha256_bytes(canonical.as_bytes());
        if !seen.insert(semantic_sha256.clone()) {
            bail!(
                "binary generation-zero line {} duplicates an earlier genome",
                line_number + 1
            );
        }
        population.push(BinaryMember {
            canonical_genome: canonical,
            semantic_sha256,
            ancestry: BinaryAncestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        });
    }
    if population.len() != expected {
        bail!(
            "binary population requires {expected} genomes, found {}",
            population.len()
        );
    }
    Ok(population)
}

/// Deterministic preset head stage for offspring reattachment. Consumes a
/// deterministic number of draws per head family, so the child is a pure
/// function of (parents, offspring_seed).
pub(super) fn binary_head_stage(rng: &mut DeterministicRng) -> String {
    let head = HEADS_BINARY[(rng.next_u64() as usize) % HEADS_BINARY.len()];
    match head {
        "logit_ridge" => {
            let l2 = ["0.005", "0.02", "0.1", "0.5"][(rng.next_u64() as usize) % 4];
            if rng.next_u64().is_multiple_of(2) {
                format!("(logit_ridge class_weight=balanced l2={l2})")
            } else {
                format!("(logit_ridge l2={l2})")
            }
        }
        "tab_boost" => {
            let lr = ["0.05", "0.1", "0.2"][(rng.next_u64() as usize) % 3];
            let rounds = ["40", "60", "80", "120"][(rng.next_u64() as usize) % 4];
            format!("(tab_boost lr={lr} rounds={rounds})")
        }
        "gam_boost" => {
            let lr = ["0.05", "0.1", "0.2"][(rng.next_u64() as usize) % 3];
            let rounds = ["40", "60", "100"][(rng.next_u64() as usize) % 3];
            format!("(gam_boost lr={lr} rounds={rounds})")
        }
        "extra_trees" => {
            let trees = ["32", "64", "96"][(rng.next_u64() as usize) % 3];
            let depth = ["6", "8", "10"][(rng.next_u64() as usize) % 3];
            format!("(extra_trees depth={depth} trees={trees})")
        }
        _ => "(logistic)".to_string(),
    }
}

/// Detach the binary head so the frozen legacy operators can work on the
/// transform body; a placeholder ridge head keeps the body a valid legacy
/// pipeline for mutation/crossover.
pub(super) fn legacy_body(canonical: &str) -> Result<String> {
    let genome = GenomeIr::parse(canonical).context("parse binary parent")?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("binary parent is not a pipeline");
    };
    let mut stages = dsl.stages;
    if stages
        .last()
        .is_some_and(|stage| is_binary_head(&stage.op) || is_head_op(&stage.op))
    {
        stages.pop();
    }
    let mut body = String::from("(pipe X");
    for stage in &stages {
        body.push(' ');
        body.push_str(&stage.to_string());
    }
    body.push_str(" (ridge))");
    Ok(body)
}

/// Reattach a deterministically drawn binary head to a mutated legacy body.
pub(super) fn with_binary_head(child_body: &str, head_stage: &str) -> Result<String> {
    let genome = GenomeIr::parse(child_body).context("parse mutated binary body")?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("mutated binary body is not a pipeline");
    };
    let mut stages = dsl.stages;
    while stages
        .last()
        .is_some_and(|stage| is_head_op(&stage.op) || is_binary_head(&stage.op))
    {
        stages.pop();
    }
    // Binary genomes may not carry target stages; drop any defensively.
    stages.retain(|stage| !is_target_op(&stage.op));
    let mut text = String::from("(pipe X");
    for stage in &stages {
        text.push(' ');
        text.push_str(&stage.to_string());
    }
    text.push(' ');
    text.push_str(head_stage);
    text.push(')');
    let rebuilt = GenomeIr::parse(&text).context("parse reheaded binary child")?;
    Ok(rebuilt.canonical())
}

pub(super) fn refill_binary_population(
    seed: u64,
    generation: usize,
    ranked: &[BinarySlotLedger],
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<Vec<BinaryMember>> {
    if ranked.len() < PARENT_POOL || ELITES >= POPULATION {
        bail!("cannot refill binary population from configured parent pool/elites");
    }
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = HashSet::with_capacity(POPULATION);
    for elite in ranked.iter().take(ELITES) {
        seen.insert(elite.semantic_sha256.clone());
        next.push(BinaryMember {
            canonical_genome: elite.canonical_genome.clone(),
            semantic_sha256: elite.semantic_sha256.clone(),
            ancestry: BinaryAncestry {
                operation: "elite".to_string(),
                parents: vec![elite.semantic_sha256.clone()],
                retry_count: 0,
            },
        });
    }
    for slot in ELITES..POPULATION {
        let mut retry = 0usize;
        loop {
            if retry >= 1_000_000 {
                bail!(
                    "could not generate an admissible binary child for generation {generation} slot {slot}"
                );
            }
            let mut rng = DeterministicRng::new(offspring_seed(seed, generation, slot, retry));
            let crossover = rng.next_u64().is_multiple_of(4);
            let winner = |rng: &mut DeterministicRng| {
                let mut best = (rng.next_u64() as usize) % PARENT_POOL;
                for _ in 1..TOURNAMENT {
                    best = best.min((rng.next_u64() as usize) % PARENT_POOL);
                }
                best
            };
            let parent_a = &ranked[winner(&mut rng)];
            let produced = if crossover {
                let parent_b = &ranked[winner(&mut rng)];
                let body = crossover_canonical_pipelines(
                    &legacy_body(&parent_a.canonical_genome)?,
                    &legacy_body(&parent_b.canonical_genome)?,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                );
                body.map(|child| {
                    (
                        "crossover".to_string(),
                        vec![
                            parent_a.semantic_sha256.clone(),
                            parent_b.semantic_sha256.clone(),
                        ],
                        child,
                    )
                })
            } else {
                mutate_canonical_pipeline(
                    &legacy_body(&parent_a.canonical_genome)?,
                    rng.next_u64(),
                    MAX_TRANSFORMS,
                )
                .map(|child| {
                    (
                        "mutation".to_string(),
                        vec![parent_a.semantic_sha256.clone()],
                        child,
                    )
                })
            };
            let (operation, parents, child_body) = match produced {
                Ok(product) => product,
                Err(_) => {
                    retry += 1;
                    continue;
                }
            };
            let head_stage = binary_head_stage(&mut rng);
            let canonical = match with_binary_head(&child_body, &head_stage) {
                Ok(canonical) => canonical,
                Err(_) => {
                    retry += 1;
                    continue;
                }
            };
            let parsed = match GenomeIr::parse(&canonical) {
                Ok(parsed) => parsed,
                Err(_) => {
                    retry += 1;
                    continue;
                }
            };
            if validate_v3_genome(&parsed, task, schema).is_err() {
                retry += 1;
                continue;
            }
            let semantic_sha256 = sha256_bytes(canonical.as_bytes());
            if !seen.insert(semantic_sha256.clone()) {
                retry += 1;
                continue;
            }
            next.push(BinaryMember {
                canonical_genome: canonical,
                semantic_sha256,
                ancestry: BinaryAncestry {
                    operation,
                    parents,
                    retry_count: retry,
                },
            });
            break;
        }
    }
    if next.len() != POPULATION {
        bail!("binary refill produced a short population");
    }
    Ok(next)
}

/// Test-adaptive ranking input for the binary lane. This deliberately exposes
/// only the fields needed by the frozen offspring operator; held-out AUC is
/// supplied as the fitness value by the caller.
#[derive(Clone, Debug)]
pub(crate) struct BinaryAdaptiveRanked {
    pub(crate) canonical_genome: String,
    pub(crate) semantic_sha256: String,
    pub(crate) complexity: usize,
    pub(crate) auc: f64,
    pub(crate) logloss: f64,
}

/// Result of one full-train, held-out-test advanced-v3 binary evaluation.
#[derive(Clone, Debug)]
pub(crate) struct BinaryAdaptiveEvaluation {
    pub(crate) complexity: usize,
    pub(crate) auc: f64,
    pub(crate) logloss: f64,
    pub(crate) predictions: Vec<f64>,
    pub(crate) fit_seconds: f64,
    pub(crate) predict_seconds: f64,
}

/// C++17 build identities accompanying one fitted binary evaluation.
#[derive(Clone, Debug)]
pub(crate) struct BinaryAdaptiveCpp17Evaluation {
    pub(crate) evaluation: BinaryAdaptiveEvaluation,
    pub(crate) source_sha256: String,
    pub(crate) codec_sha256: String,
    pub(crate) compile_seconds: f64,
}

/// Parse the exact canonical 60-member binary seed file through the same
/// admission checks used by the historical train-only binary campaign.
pub(crate) fn test_adaptive_seed_population(
    bytes: &[u8],
    features: usize,
) -> Result<Vec<BinaryMember>> {
    parse_binary_population(bytes, POPULATION, &binary_task(), &binary_schema(features))
}

pub(crate) fn binary_genome_complexity(canonical_genome: &str) -> Result<usize> {
    Ok(genome_complexity(&GenomeIr::parse(canonical_genome)?))
}

/// Refill a deterministic 60-member generation using the historical binary
/// mutation/crossover/head operators and the caller's held-out-test ranking.
pub(crate) fn test_adaptive_next_population(
    seed: u64,
    generation: usize,
    ranked: &[BinaryAdaptiveRanked],
    features: usize,
) -> Result<Vec<BinaryMember>> {
    let ranked = ranked
        .iter()
        .enumerate()
        .map(|(slot, candidate)| BinarySlotLedger {
            schema: LEDGER_SCHEMA.to_string(),
            generation: generation.saturating_sub(1),
            slot,
            canonical_genome: candidate.canonical_genome.clone(),
            semantic_sha256: candidate.semantic_sha256.clone(),
            ancestry: BinaryAncestry {
                operation: "ranked-parent".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
            complexity: candidate.complexity,
            valid: true,
            rejection_reason: None,
            cache_disposition: "test-adaptive-parent".to_string(),
            fold_auc: None,
            fold_defined: None,
            pooled_oof_auc: Some(candidate.auc),
            pooled_oof_logloss: Some(candidate.logloss),
            pooled_oof_brier: None,
            robust_fitness: Some(candidate.auc),
        })
        .collect::<Vec<_>>();
    refill_binary_population(
        seed,
        generation,
        &ranked,
        &binary_task(),
        &binary_schema(features),
    )
}

/// Fit one canonical advanced-v3 binary genome on all train rows and evaluate
/// probabilities directly on the held-out test target.
pub(crate) fn evaluate_test_adaptive_binary(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<BinaryAdaptiveEvaluation> {
    let (genome, model, fit_seconds) = fit_test_adaptive_binary(canonical_genome, train, test)?;
    let test_table = raw_table(test.features, &test.x)?;
    let predict_started = Instant::now();
    let predictions = model.predict_table(&test_table)?;
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    finish_test_adaptive_binary(
        genome_complexity(&genome),
        predictions,
        test,
        fit_seconds,
        predict_seconds,
    )
}

/// Fit on the train split, lower the fitted model to a compiled C++17 image,
/// and use only that image to produce held-out probabilities.
pub(crate) fn evaluate_test_adaptive_binary_cpp17(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
    runtime_root: &Path,
) -> Result<BinaryAdaptiveCpp17Evaluation> {
    let (genome, model, fit_seconds) = fit_test_adaptive_binary(canonical_genome, train, test)?;
    let emitted = emit_fitted_cpp17(&model).context("emit fitted binary C++17 model")?;
    let compile_started = Instant::now();
    let codec = encode_fitted_cpp17(&model).context("compile fitted binary C++17 model")?;
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    validate_fitted_cpp_codec(&codec.bytes)?;
    let runtime = FittedCppRuntime::new(runtime_root)?;
    let compiled = runtime.load(&codec.bytes)?;
    let encoded_test = EncodedTable::from_numeric_rows(&test.x)?;
    let predict_started = Instant::now();
    let predictions = compiled.predict(&encoded_test)?;
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    let evaluation = finish_test_adaptive_binary(
        genome_complexity(&genome),
        predictions,
        test,
        fit_seconds,
        predict_seconds,
    )?;
    Ok(BinaryAdaptiveCpp17Evaluation {
        evaluation,
        source_sha256: emitted.source_sha256,
        codec_sha256: codec.sha256,
        compile_seconds,
    })
}

pub(super) fn fit_test_adaptive_binary(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<(GenomeIr, BoundModel, f64)> {
    if train.features != test.features {
        bail!("binary test-adaptive train/test feature widths differ");
    }
    let genome = GenomeIr::parse(canonical_genome).context("parse binary adaptive genome")?;
    if genome.canonical() != canonical_genome {
        bail!("binary test-adaptive genome is not canonical");
    }
    let task = binary_task();
    let schema = binary_schema(train.features);
    validate_v3_genome(&genome, &task, &schema)?;
    let train_rows = train.x.len().min(train.y.len());
    if train_rows != train.x.len() || train_rows == 0 || test.y.len() != test.x.len() {
        bail!("binary test-adaptive data has inconsistent row counts");
    }
    let train_table = raw_table(train.features, &train.x)?;
    let fit_started = Instant::now();
    let model = fit_table(
        &genome,
        &task,
        &schema,
        &train_table,
        &train.y[..train_rows],
    )?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    Ok((genome, model, fit_seconds))
}

pub(super) fn finish_test_adaptive_binary(
    complexity: usize,
    predictions: Vec<f64>,
    test: &RegressionData,
    fit_seconds: f64,
    predict_seconds: f64,
) -> Result<BinaryAdaptiveEvaluation> {
    validate_probabilities(&predictions, test.y.len())?;
    let auc = auc(&test.y, &predictions);
    let logloss = logloss(&test.y, &predictions);
    if !auc.is_finite() || !logloss.is_finite() {
        bail!("binary test-adaptive metrics are non-finite");
    }
    Ok(BinaryAdaptiveEvaluation {
        complexity,
        auc,
        logloss,
        predictions,
        fit_seconds,
        predict_seconds,
    })
}

// ---------------------------------------------------------------------------
// campaign
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(super) struct BinaryEvaluation {
    complexity: usize,
    result: Option<BinaryCvResult>,
    rejection_reason: Option<String>,
}

pub(super) fn evaluate_binary_unique(
    member: &BinaryMember,
    fold_cache: &BinaryFoldCache<'_>,
    task: &TaskSpec,
    schema: &DataSchema,
) -> BinaryEvaluation {
    let rejected = |complexity, reason: String| BinaryEvaluation {
        complexity,
        result: None,
        rejection_reason: Some(reason),
    };
    let genome = match GenomeIr::parse(&member.canonical_genome) {
        Ok(genome) if matches!(genome, GenomeIr::Pipeline(_)) => genome,
        Ok(_) => return rejected(0, "not a pipeline genome".to_string()),
        Err(error) => return rejected(0, format!("parse: {error:#}")),
    };
    let canonical = genome.canonical();
    let complexity = genome_complexity(&genome);
    if canonical != member.canonical_genome
        || sha256_bytes(canonical.as_bytes()) != member.semantic_sha256
    {
        return rejected(
            complexity,
            "canonical or semantic identity mismatch".to_string(),
        );
    }
    if let Err(error) = validate_v3_genome(&genome, task, schema) {
        return rejected(complexity, format!("admission: {error:#}"));
    }
    match binary_cv_on_slices(
        &genome,
        fold_cache,
        CV5_FOLDS,
        &[QUICK_HOLDOUT_FOLD],
        task,
        schema,
    ) {
        Ok(result) => BinaryEvaluation {
            complexity,
            result: Some(result),
            rejection_reason: None,
        },
        Err(error) => rejected(complexity, format!("cv: {error:#}")),
    }
}
