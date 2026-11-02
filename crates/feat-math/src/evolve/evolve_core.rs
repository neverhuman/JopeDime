use super::*;

/// Evolve pipeline genomes on (x, y). Returns the deduped population best-first (callers take the
/// top-N finalists for proper CV downstream). Dispatches on `cfg.fitness`: the v1 surrogate path
/// is untouched, bit-identical code; the v2 lite-race is a separate loop (callers that need the
/// per-rung score dump should call `evolve_lite_race` directly).

pub fn evolve(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
) -> Vec<Scored> {
    match cfg.fitness {
        FitnessMode::Surrogate => evolve_surrogate(x, y, metric, cfg, obs),
        FitnessMode::LiteRace => evolve_lite_race(x, y, metric, cfg, obs).ranked,
    }
}

/// v1 fitness loop: ridge-surrogate scoring of every candidate (see `search_surrogate`).
pub(super) fn evolve_surrogate(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
) -> Vec<Scored> {
    // `None` under `FaultPolicy::Legacy`: every step below then runs the legacy code.
    let mut admission = Admission::start(cfg.fault_policy);
    let t0 = Instant::now();
    let mut rng = Pcg32::new(cfg.seed, 0x5EED);
    let assign = fold_assign(y.len(), cfg.folds.max(2), "invent");
    let k = cfg.folds.max(2);
    let gen_vocab = generation_vocab(cfg);
    // B10 lexicase: begin
    if let Err(reason) = cfg.selection.validate(cfg.fault_policy) {
        panic!("{reason}");
    }
    let lexicase = cfg.selection.lexicase();
    let case_rows = match lexicase {
        Some(_) => lexicase_case_rows(y.len(), cfg.seed),
        None => Vec::new(),
    };
    let mut cases: HashMap<String, Vec<f64>> = HashMap::new();
    // B10 lexicase: end
    // B11 ridge scoring: begin
    if let Err(reason) = cfg.ridge_scoring.validate(cfg.fitness, cfg.fault_policy) {
        panic!("{reason}");
    }
    let mut certified = (cfg.ridge_scoring == RidgeScoring::Certified).then(CertifiedScoring::new);
    // B11 ridge scoring: end

    // Seeds: champion + external seeds + starters + random fill.
    let target_pop = cfg.pop.max(1);
    let mut population: Vec<DslExpr> = Vec::with_capacity(target_pop);
    let mut seed_seen = std::collections::HashSet::new();
    for src in std::iter::once(CHAMPION_GENOME)
        .chain(cfg.seed_genomes.iter().map(String::as_str))
        .chain(
            cfg.hard_ten_priors
                .then_some(HARD_TEN_STARTERS.as_slice())
                .into_iter()
                .flatten()
                .copied(),
        )
        .chain(STARTERS.iter().copied())
        .chain(
            cfg.expansion_vocab
                .then_some(EXPANSION_STARTERS.as_slice())
                .into_iter()
                .flatten()
                .copied(),
        )
    {
        if population.len() >= target_pop {
            break;
        }
        if let Ok(GenomeIr::Pipeline(d)) = GenomeIr::parse(src) {
            let canonical = d.to_string();
            if seed_seen.insert(canonical) {
                population.push(d);
            }
        }
    }
    while population.len() < target_pop {
        population.push(rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab));
    }

    let mut memo: HashMap<String, Option<Vec<f64>>> = HashMap::new();
    let mut ledger = TokenLedger::new();
    let mut evals: u64 = 0;
    let mut best: Option<Scored> = None;
    let mut ranked: Vec<Scored> = Vec::new();
    // Wall-clock throttles for the live streams (decoupled from generations). A fast
    // score-only tick (~16/sec) drives the ticker + score scatter. The code panel is fed
    // ONLY by the per-generation leader (`ranked[0]`), throttled and emitted just when the
    // leader changes — arbitrary candidates are never transpiled. Both are best-effort.
    let mut last_eval_tick = Instant::now();
    let mut last_eval_code = Instant::now();
    let mut last_leader: Option<String> = None;

    for gen in 0..cfg.gens.max(1) {
        if obs.is_cancelled() {
            break;
        }
        if cfg.time_budget_secs > 0.0 && t0.elapsed().as_secs_f64() > cfg.time_budget_secs {
            break;
        }

        for (slot, genome) in population.iter().enumerate() {
            crate::capture_gp_source!(
                "surrogate",
                &format!("generation/{gen}/slot/{slot}"),
                &genome.to_string()
            );
        }

        // Evaluate all unseen genomes in parallel (the Rust interpreter executes each candidate).
        let unseen: Vec<(String, DslExpr)> = population
            .iter()
            .map(|g| (g.to_string(), g.clone()))
            .filter(|(key, _)| !memo.contains_key(key))
            .collect();
        let fresh: Vec<(String, DslExpr, Option<Vec<f64>>)> = match admission.as_mut() {
            // B11 ridge scoring: the same scores (and cases) from the memo and the stage cache.
            None if certified.is_some() => {
                let Some(scoring) = certified.as_mut() else {
                    unreachable!("certified scoring is set")
                };
                scoring.cache.begin_generation(gen as u64);
                let rows = lexicase.map(|_| case_rows.as_slice());
                let fresh =
                    scoring.score_unseen(unseen, x, y, &assign, k, metric, rows, &mut cases);
                scoring.cache.sweep();
                obs.on_scoring(&scoring.receipt(gen));
                fresh
            }
            // B10 lexicase: the same scores, keeping each candidate's residual cases.
            None if lexicase.is_some() => {
                score_unseen_with_cases(unseen, x, y, &assign, k, metric, &case_rows, &mut cases)
            }
            None => unseen
                .into_par_iter()
                .map(|(key, g)| {
                    let scores = score_genome(&g, x, y, &assign, k, metric);
                    (key, g, scores)
                })
                .collect(),
            Some(admission) => {
                let policy = admission.policy;
                let admitted: Vec<(String, DslExpr, Admitted)> = unseen
                    .into_par_iter()
                    .map(|(key, g)| {
                        let admitted = score_genome_admitted(
                            &g,
                            x,
                            y,
                            &assign,
                            k,
                            metric,
                            policy,
                            &fit_traced,
                        );
                        (key, g, admitted)
                    })
                    .collect();
                admitted
                    .into_iter()
                    .map(|(key, g, admitted)| {
                        let scores = admission.record(&key, 0, admitted, obs);
                        (key, g, scores)
                    })
                    .collect()
            }
        };
        // Per-generation token accounting uses the STATIC estimator (X6, no
        // emission): every evaluated genome is a shippable dual-language
        // algorithm, so we credit its exact predicted Python+Rust token
        // count without paying to emit it. tokens/sec is therefore live even
        // when no improvement fires. Genomes the emitter cannot support
        // (never generated here) simply credit nothing. The estimates are pure,
        // so they are computed in parallel and credited in candidate order.
        let estimates = token_estimates(fresh.iter().map(|(_, g, _)| g));
        for ((key, _, scores), est) in fresh.iter().zip(estimates) {
            evals += 1;
            ledger.add(est);
            memo.insert(key.clone(), scores.clone());

            // Live candidate stream (throttled), SCORE ONLY. Arbitrary candidates are never
            // transpiled — only the per-generation leader is (see the leader-code emit after
            // ranking). This keeps the ticker + score scatter live without spraying code.
            if let Some(fold) = scores {
                if !fold.is_empty() {
                    let cv = fold.iter().sum::<f64>() / fold.len() as f64;
                    if cv.is_finite() {
                        let now = Instant::now();
                        if now.duration_since(last_eval_tick).as_millis() >= EVAL_SCORE_SAMPLE_MS {
                            last_eval_tick = now;
                            obs.on_eval(key, cv, None);
                        }
                    }
                }
            }
        }

        // Rank the current population.
        ranked = {
            let mut seen = std::collections::HashSet::new();
            let mut v: Vec<Scored> = population
                .iter()
                .filter_map(|g| {
                    let key = g.to_string();
                    if !seen.insert(key.clone()) {
                        return None;
                    }
                    let fold_scores = memo.get(&key)?.clone()?;
                    let cv_mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                    Some(Scored {
                        dsl: key,
                        cv_mean,
                        fold_scores,
                    })
                })
                .collect();
            v.sort_by(|a, b| {
                b.cv_mean
                    .partial_cmp(&a.cv_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            v
        };
        if let Some(admission) = &admission {
            if admission.policy.breaks_ties() {
                prefer_fewer_faults(
                    &mut ranked,
                    |s| s.cv_mean,
                    |s| admission.totals(&s.dsl, 0).faulted_cells(),
                );
            }
        }
        if ranked.is_empty() {
            // population collapsed to invalid genomes — reseed randoms and continue
            population = (0..cfg.pop)
                .map(|_| rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab))
                .collect();
            continue;
        }

        // Stream ONLY the generation leader's code (ephemeral): whenever ranked[0] — the best
        // of this generation — changes, transpile just that one genome and push it to the code
        // panel, throttled so ultra-fast generations can't spray. No other candidate is ever
        // converted to code, and this rides the ephemeral eval channel so it is never persisted.
        {
            // Emit a NEW leader's code immediately (the first emittable one always goes); only
            // rate-limit subsequent leader changes so ultra-fast generations can't stream faster
            // than the panel can read. State advances only on a successful emit, so a leader the
            // transpiler can't render is retried, not silently swallowed.
            let leader_changed = last_leader.as_deref() != Some(ranked[0].dsl.as_str());
            let now = Instant::now();
            let first = last_leader.is_none();
            if leader_changed
                && (first || now.duration_since(last_eval_code).as_millis() >= EVAL_CODE_SAMPLE_MS)
            {
                if let Some(py) = GenomeIr::parse(&ranked[0].dsl)
                    .ok()
                    .and_then(|gm| emit(&gm, &EmitOptions::default()).ok())
                    .map(|e| e.python)
                {
                    last_eval_code = now;
                    last_leader = Some(ranked[0].dsl.clone());
                    obs.on_eval(&ranked[0].dsl, ranked[0].cv_mean, Some(&py));
                }
            }
        }

        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let counters = admission
            .as_ref()
            .map_or_else(FaultCounters::default, |admission| {
                admission.counters(&population, &ranked, (cfg.pop / 8).clamp(1, 8), |_| 0)
            });
        let mut stats = GenStats {
            gen,
            gens: cfg.gens,
            evals,
            evals_per_sec: evals as f64 / elapsed,
            tokens_total: ledger.total(),
            tokens_per_sec: ledger.rate(),
            best_cv: ranked[0].cv_mean,
            pop_fitness: ranked.iter().map(|s| s.cv_mean).collect(),
            hard_rejected_share: counters.hard_rejected_share,
            hard_rejected: counters.hard_rejected,
            pop_domain_rate: counters.pop_domain_rate,
            elite_domain_rate: counters.elite_domain_rate,
        };

        // New best? Emit its FULL dual-language source and stream it.
        let improved = best
            .as_ref()
            .map(|b| ranked[0].cv_mean > b.cv_mean + 1e-9)
            .unwrap_or(true);
        if improved {
            best = Some(ranked[0].clone());
            if let Ok(genome) = GenomeIr::parse(&ranked[0].dsl) {
                if let Ok(emitted) = emit(&genome, &EmitOptions::default()) {
                    // real token credit for the artifact we actually produced
                    ledger.add(
                        (emitted.py_tokens.code_tokens
                            + emitted.py_tokens.doc_tokens
                            + emitted.rs_tokens.code_tokens
                            + emitted.rs_tokens.doc_tokens) as f64,
                    );
                    stats.tokens_total = ledger.total();
                    stats.tokens_per_sec = ledger.rate();
                    obs.on_improvement(&ranked[0], &emitted, &stats);
                }
            }
        }
        obs.on_generation(&stats);

        // Next generation: elites survive, the rest are tournament offspring.
        let elite = (cfg.pop / 8).clamp(1, 8);
        let mut next: Vec<DslExpr> = ranked
            .iter()
            .take(elite)
            .filter_map(|s| match GenomeIr::parse(&s.dsl) {
                Ok(GenomeIr::Pipeline(d)) => Some(d),
                _ => None,
            })
            .collect();
        let parents = ranked.len().min(cfg.pop / 2).max(1);
        let tie_cells = admission
            .as_ref()
            .and_then(|admission| admission.tie_break_cells(&ranked, |_| 0));
        // B10 lexicase: begin
        let matrix =
            lexicase.map(|_| lexicase_matrix(&ranked, |dsl| cases.get(dsl).map(Vec::as_slice)));
        let mut draws = start_lexicase(lexicase, matrix.as_ref(), cfg.seed, gen);
        // B10 lexicase: end
        while next.len() < cfg.pop {
            let pa = pick_parent(&mut rng, &ranked, parents, tie_cells.as_deref(), &mut draws);
            let child = if rng.next_f64() < 0.25 && ranked.len() >= 2 {
                let pb = pick_parent(&mut rng, &ranked, parents, tie_cells.as_deref(), &mut draws);
                let (ga, gb) = (parse_pipe(&ranked[pa].dsl), parse_pipe(&ranked[pb].dsl));
                match (ga, gb) {
                    (Some(a), Some(b)) => {
                        crossover_with_source_capture(&mut rng, &a, &b, cfg.max_transforms)
                    }
                    _ => rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab),
                }
            } else {
                match parse_pipe(&ranked[pa].dsl) {
                    Some(a) => mutate_with_source_capture(&mut rng, &a, cfg.max_transforms),
                    None => rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab),
                }
            };
            next.push(child);
        }
        report_selection(obs, gen, ranked.len(), draws);
        population = next;
    }

    ranked
}

// ---------------------------------------------------------------------------
