use super::*;

pub(crate) fn make_member(source: &str, origin: &str) -> Option<Member> {
    let canonical = canonicalize(source).ok()?;
    Some(Member {
        canonical: canonical.canonical,
        semantic: canonical.semantic_sha256,
        complexity: canonical.complexity,
        origin: origin.to_string(),
        scores: Vec::new(),
        fitness_complete: false,
    })
}

pub(crate) fn tournament_pick(rng: &mut Pcg32, pool: usize) -> usize {
    let mut best = usize::MAX;
    for _ in 0..TOURNAMENT {
        let c = (rng.next_u32() as usize) % pool;
        best = best.min(c);
    }
    best
}

/// Breed one child from the ranked pool (golden42 operators: tournament-3
/// parents, 0.25 crossover, else mutate; 100 deterministic retries).
pub(crate) fn breed_child(
    ranked: &[Member],
    pool: usize,
    features: usize,
    tag: &str,
    seen: &HashSet<String>,
    expanded_vocab: bool,
) -> Option<Member> {
    for retry in 0..BREED_RETRIES {
        let seed = seed_from(&format!("screen:{tag}:{retry}"));
        let mut rng = Pcg32::new(seed, 0x5343_5245_454e);
        let first = tournament_pick(&mut rng, pool);
        let evolved = if rng.next_f64() < CROSSOVER_PROBABILITY {
            let second = tournament_pick(&mut rng, pool);
            crossover_pipelines(
                &ranked[first].canonical,
                &ranked[second].canonical,
                rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            )
        } else {
            let mutate = if expanded_vocab {
                mutate_pipeline_expanded
            } else {
                mutate_pipeline
            };
            mutate(
                &ranked[first].canonical,
                rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                features,
            )
        };
        if let Ok(g) = evolved {
            if !seen.contains(&g.semantic_sha256) {
                return Some(Member {
                    canonical: g.canonical,
                    semantic: g.semantic_sha256,
                    complexity: g.complexity,
                    origin: "evolved".to_string(),
                    scores: Vec::new(),
                    fitness_complete: false,
                });
            }
        }
    }
    None
}

pub(crate) fn fill_population(
    base: Vec<Member>,
    target: usize,
    features: usize,
    tag: &str,
    expanded_vocab: bool,
) -> Vec<Member> {
    let mut seen: HashSet<String> = base.iter().map(|m| m.semantic.clone()).collect();
    let mut population = base;
    let mut slot = 0usize;
    while population.len() < target && slot < target * 40 {
        if let Some(child) = breed_child(
            &population,
            population.len(),
            features,
            &format!("{tag}:fill:{slot}"),
            &seen,
            expanded_vocab,
        ) {
            seen.insert(child.semantic.clone());
            population.push(child);
        }
        slot += 1;
    }
    population
}

pub(crate) fn run_strategy(
    config: &Config,
    train_path: &str,
    x: &Rows,
    y: &[f64],
) -> Result<(Vec<Member>, Vec<GenerationStat>, usize), String> {
    let features = x[0].len();
    let ds = &config.dataset_hash;
    let ms = config.master_seed;
    let mut evaluations = 0usize;
    let mut stats = Vec::new();

    // ---- generation 0 composition ----
    let mut gen0: Vec<Member> = Vec::new();
    let mut seen = HashSet::new();
    let push_unique = |pool: &mut Vec<Member>, seen: &mut HashSet<String>, m: Member| {
        if seen.insert(m.semantic.clone()) {
            pool.push(m);
        }
    };
    for g in config.dl_genes.iter().take(10) {
        if let Some(m) = make_member(g, "dl_seed") {
            push_unique(&mut gen0, &mut seen, m);
        }
    }
    let prior: Vec<(&str, &Vec<String>)> = match config.strategy.as_str() {
        "s2" => vec![("neighbor_prior", &config.neighbor_prior)],
        "s5" => vec![
            ("global_prior", &config.global_prior),
            ("neighbor_prior", &config.neighbor_prior),
        ],
        _ => vec![("global_prior", &config.global_prior)],
    };
    for (origin, list) in prior {
        for g in list
            .iter()
            .take(if config.strategy == "s5" { 32 } else { 10 })
        {
            if let Some(m) = make_member(g, origin) {
                push_unique(&mut gen0, &mut seen, m);
            }
        }
    }
    // A fixed, train-only baseline set. D1 measured that a plain fit of one of
    // these beats a full 1,000-eval search on 7 of 22 datasets (e5a3a7b3 by 37%,
    // b78d3235 by 12%), and that none of them ever entered a pool: DL proposals
    // and priors are all elaborate pipelines, and mutation only elaborates
    // further, so the search could never fall back on a textbook model. Seeding
    // them costs |BASELINE_SEEDS| evaluations and floors the result.
    if config.baseline_seeds {
        for g in BASELINE_SEEDS {
            if let Some(m) = make_member(g, "baseline") {
                push_unique(&mut gen0, &mut seen, m);
            }
        }
    }

    let schedule: Vec<(usize, usize)> = if config.strategy == "s5" {
        config.budget.halving.clone()
    } else {
        let splits = if config.budget.cv_fitness > 0 {
            0
        } else {
            config.budget.splits_per_gen
        };
        (0..config.budget.generations)
            .map(|_| (config.budget.population, splits))
            .collect()
    };
    if config.strict_pooled_search() && schedule.is_empty() {
        return Err("strict pooled search requires a non-empty schedule".to_string());
    }
    let expanded_vocab = config.vocab == "expanded";
    let gen0_target = schedule.first().map_or(config.budget.population, |s| s.0);
    if config.strict_pooled_search() && gen0.is_empty() {
        return Err("strict pooled search requires at least one valid seed genome".to_string());
    }
    if config.strict_pooled_search() && gen0.len() > gen0_target {
        return Err(format!(
            "strict pooled search has {} unique seeds for {gen0_target} generation-0 slots",
            gen0.len()
        ));
    }
    let mut population = fill_population(
        gen0,
        gen0_target,
        features,
        &format!("{ds}:{ms}:g0"),
        expanded_vocab,
    );

    // ---- DL draw bank for s3 (continual injection) ----
    let mut bank: Vec<Member> = config
        .dl_draw_bank
        .iter()
        .filter_map(|g| make_member(g, "dl_injected"))
        .filter(|m| !seen.contains(&m.semantic))
        .collect();

    // ---- generational loop ----
    let mut split_hashes: HashSet<String> = HashSet::new();
    for (generation, &(width, splits_per_gen)) in schedule.iter().enumerate() {
        // shrink (successive halving) or plain carry
        if population.len() > width {
            rank(&mut population);
            population.truncate(width);
        }
        if config.strict_pooled_search() && population.len() != width {
            return Err(format!(
                "generation {generation} contains {} members; exact budget requires {width}",
                population.len()
            ));
        }
        let mut invalid_members = 0usize;
        // V2: every member receives exactly one complete pooled OOF score from
        // the current generation's common two-fold bank. No member carries a
        // longer fitness history merely because it survived another generation.
        if config.strict_pooled_search() {
            let folds = kfold(
                y,
                config.budget.cv_fitness,
                seed_from(&format!("screen:{ds}:{ms}:cvgen:{generation}")),
            );
            validate_strict_fold_plan(y.len(), &folds)?;
            let fold_train: Vec<Vec<usize>> = (0..folds.len())
                .map(|f| {
                    folds
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != f)
                        .flat_map(|(_, ids)| ids.iter().copied())
                        .collect()
                })
                .collect();
            let outcomes: Vec<(Option<f64>, EvalFailureCounts)> = population
                .par_iter()
                .map(|member| {
                    let predictions: Vec<Result<Vec<f64>, EvalFailure>> = (0..folds.len())
                        .map(|f| {
                            eval_predictions(
                                train_path,
                                &member.canonical,
                                &fold_train[f],
                                &folds[f],
                            )
                        })
                        .collect();
                    let mut failures = EvalFailureCounts::default();
                    for failure in predictions
                        .iter()
                        .filter_map(|result| result.as_ref().err())
                    {
                        failures.record(*failure);
                    }
                    let prediction_views: Vec<Option<&[f64]>> = predictions
                        .iter()
                        .map(|result| result.as_ref().ok().map(Vec::as_slice))
                        .collect();
                    let fitness = complete_pooled_oof_r2(y, &folds, &prediction_views);
                    if fitness.is_none() && failures.total() == 0 {
                        failures.invalid_oof = 1;
                    }
                    (fitness, failures)
                })
                .collect();
            let mut failures = EvalFailureCounts::default();
            for (_, member_failures) in &outcomes {
                failures.merge(*member_failures);
            }
            invalid_members = replace_generation_outcomes(
                &mut population,
                outcomes.into_iter().map(|(fitness, _)| fitness).collect(),
            );
            strict_generation_completion(generation, invalid_members, failures)?;
            evaluations += population.len() * folds.len();
        // V1 replay: fresh split(s) or CV folds append to member history.
        } else if config.budget.cv_fitness > 0 {
            let folds = kfold(
                y,
                usable_folds(y.len(), config.budget.cv_fitness),
                seed_from(&format!("screen:{ds}:{ms}:cvgen:{generation}")),
            );
            for f in 0..folds.len() {
                let valid_ids = &folds[f];
                let train_ids: Vec<usize> = folds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != f)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect();
                let scores: Vec<f64> = population
                    .par_iter()
                    .map(|m| eval_with_timeout(train_path, &m.canonical, &train_ids, valid_ids))
                    .collect();
                for (m, s) in population.iter_mut().zip(scores) {
                    m.scores.push(s);
                    evaluations += 1;
                }
            }
        } else {
            for s in 0..splits_per_gen.max(1) {
                let seed = seed_from(&format!("screen:{ds}:{ms}:split:{generation}:{s}"));
                let (train_ids, valid_ids) = split_8020(y, seed);
                let (eval_path, train_ids) = if config.perturb_fitness {
                    match perturbed_generation_csv(
                        x,
                        y,
                        &train_ids,
                        generation,
                        &format!("{ds}:{ms}"),
                    ) {
                        Ok(v) => v,
                        Err(_) => (train_path.to_string(), train_ids),
                    }
                } else {
                    (train_path.to_string(), train_ids)
                };
                // train_only_study discipline: refuse silently-identical splits
                let mut h = Sha256::new();
                for i in &valid_ids {
                    h.update(i.to_le_bytes());
                }
                let hh = format!("{:x}", h.finalize());
                if !split_hashes.insert(hh) {
                    continue;
                }
                let scores: Vec<f64> = population
                    .par_iter()
                    .map(|m| eval_with_timeout(&eval_path, &m.canonical, &train_ids, &valid_ids))
                    .collect();
                for (m, sc) in population.iter_mut().zip(scores) {
                    m.scores.push(sc);
                    evaluations += 1;
                }
                if eval_path.starts_with(".perturbed-") {
                    let _ = fs::remove_file(&eval_path);
                }
            }
        }
        rank(&mut population);
        stats.push(GenerationStat {
            generation,
            evaluations,
            best_fitness: population
                .first()
                .map_or(-1.0, |m| robust_fitness(&m.scores)),
            invalid_members,
        });
        // breed the next generation (skip after the final evaluation round)
        if generation + 1 == schedule.len() {
            break;
        }
        let next_width = schedule[generation + 1].0;
        let elites = config.budget.elites.min(population.len());
        let mut next: Vec<Member> = population.iter().take(elites).cloned().collect();
        let mut next_seen: HashSet<String> = next.iter().map(|m| m.semantic.clone()).collect();
        // s3: fresh DL draws take offspring slots each generation
        let mut injected = 0usize;
        while injected < config.budget.dl_inject_per_gen && !bank.is_empty() {
            let m = bank.remove(0);
            if next_seen.insert(m.semantic.clone()) {
                next.push(m);
                injected += 1;
            }
        }
        let mut slot = 0usize;
        while next.len() < next_width && slot < next_width * 40 {
            if let Some(child) = breed_child(
                &population,
                population.len().min(width),
                features,
                &format!("{ds}:{ms}:g{}:{slot}", generation + 1),
                &next_seen,
                expanded_vocab,
            ) {
                next_seen.insert(child.semantic.clone());
                next.push(child);
            }
            slot += 1;
        }
        if config.strict_pooled_search() && next.len() != next_width {
            return Err(format!(
                "generation {} could produce only {} of {next_width} required members",
                generation + 1,
                next.len()
            ));
        }
        population = next;
    }
    rank(&mut population);
    if config.strict_pooled_search() {
        let expected = expected_search_evaluations(&schedule, config.budget.cv_fitness);
        if evaluations != expected {
            return Err(format!(
                "search evaluation count {evaluations} does not match exact budget {expected}"
            ));
        }
    }
    Ok((population, stats, evaluations))
}
