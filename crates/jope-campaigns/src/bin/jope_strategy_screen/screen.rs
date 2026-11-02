use super::*;

pub(super) fn run(config_path: &str, train_path: &str, output_path: &str) -> Result<(), String> {
    let config: Config = serde_json::from_str(
        &fs::read_to_string(config_path).map_err(|e| format!("read {config_path}: {e}"))?,
    )
    .map_err(|e| format!("parse config: {e}"))?;
    config.validate()?;
    let (x, y) = load_csv(train_path)?;
    let started = std::time::Instant::now();
    let (population, stats, search_evals) = run_strategy(&config, train_path, &x, &y)?;

    // Seeding baselines into generation 0 is not enough: the in-generation
    // fitness culls every one of them before the finale (measured — all 8
    // finalists came back origin=evolved on a seeded run). Union them into the
    // finale pool so CV always gets to compare against the floor.
    let mut finalists: Vec<Member> = population
        .iter()
        .take(config.final_top_k)
        .cloned()
        .collect();
    if config.baseline_seeds {
        let chosen: HashSet<String> = finalists.iter().map(|m| m.semantic.clone()).collect();
        // Rebuilt from the constant, not filtered out of `population`: evolution
        // has already dropped them by this point, which is the whole problem.
        for g in BASELINE_SEEDS {
            if let Some(m) = make_member(g, "baseline") {
                if !chosen.contains(&m.semantic) {
                    finalists.push(m);
                }
            }
        }
    }

    let mut top: Vec<TopGene> = finalists
        .iter()
        .map(|m| TopGene {
            canonical: m.canonical.clone(),
            semantic: m.semantic.clone(),
            origin: m.origin.clone(),
            complexity: m.complexity,
            robust_fitness: robust_fitness(&m.scores),
            worst_split: worst_split(&m.scores),
            splits_seen: m.scores.len(),
            search_fitness_complete: m.fitness_complete,
            cv7_mean_r2: None,
            rcv_mean_r2: None,
            rcv_se: None,
            boot_mean_r2: None,
            boot_sd_r2: None,
            cv_failed_folds: 0,
            rcv_failed_repeats: 0,
        })
        .collect();
    let cv_evals = final_cv(&config, train_path, &x, &y, &mut top)?;
    if config.strict_pooled_search() && top.iter().all(|gene| gene.cv7_mean_r2.is_none()) {
        return Err("strict pooled final CV produced no complete finalist".to_string());
    }

    // ---- T4: repeated stratified CV (5 x k folds), 1-SE + parsimony ----
    let repeats = 5usize;
    let mut rcv_evals = 0usize;
    let mut oof_matrix: Vec<Vec<(usize, f64)>> = vec![Vec::new(); top.len()];
    for rep in 0..repeats {
        let fold_count = if config.strict_pooled_search() {
            config.final_cv_folds
        } else {
            usable_folds(y.len(), config.final_cv_folds)
        };
        let folds = kfold(
            &y,
            fold_count,
            seed_from(&format!(
                "screen:{}:{}:rcv:{rep}",
                config.dataset_hash, config.master_seed
            )),
        );
        if config.strict_pooled_search() {
            validate_strict_fold_plan(y.len(), &folds)?;
        }
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
        let pairs: Vec<(usize, usize)> = (0..top.len())
            .flat_map(|g| (0..folds.len()).map(move |f| (g, f)))
            .collect();
        rcv_evals += pairs.len();
        // One pooled OOF R^2 per repeat, not a mean of per-fold R^2. Per-fold
        // R^2 is undefined under leave-one-out -- a single held-out row has
        // ss_tot = 0, so every fold returns None and the whole estimator
        // collapses to the clip floor with zero variance, which then hands
        // selection to the complexity/hash fallback. Pooling the repeat's
        // predictions keeps the statistic defined at every n we support.
        type FoldPrediction = ((usize, usize), Result<Vec<f64>, EvalFailure>);
        let outs: Vec<FoldPrediction> = pairs
            .par_iter()
            .map(|&(g, f)| {
                (
                    (g, f),
                    eval_predictions(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
                )
            })
            .collect();
        for g in 0..top.len() {
            let mut truth: Vec<f64> = Vec::new();
            let mut pred: Vec<f64> = Vec::new();
            let mut prediction_views: Vec<Option<&[f64]>> = Vec::with_capacity(folds.len());
            for (f, fold) in folds.iter().enumerate() {
                if let Some((_, Ok(p))) = outs.iter().find(|((gg, ff), _)| *gg == g && *ff == f) {
                    if p.len() == fold.len() {
                        prediction_views.push(Some(p.as_slice()));
                        for (k, &row) in fold.iter().enumerate() {
                            truth.push(y[row]);
                            pred.push(p[k]);
                        }
                    } else {
                        prediction_views.push(None);
                    }
                } else {
                    prediction_views.push(None);
                }
            }
            let score = if config.strict_pooled_search() {
                complete_pooled_oof_r2(&y, &folds, &prediction_views)
            } else {
                r2(&truth, &pred).map(|value| value.max(-1.0))
            };
            if let Some(v) = score {
                oof_matrix[g].push((rep, v.max(-1.0)));
            } else {
                top[g].rcv_failed_repeats += 1;
            }
        }
    }
    for (g, gene) in top.iter_mut().enumerate() {
        let vals: Vec<f64> = oof_matrix[g].iter().map(|(_, v)| *v).collect();
        let complete = !config.strict_pooled_search()
            || (gene.rcv_failed_repeats == 0 && vals.len() == repeats);
        if complete && !vals.is_empty() {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sd =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
            gene.rcv_mean_r2 = Some(mean);
            gene.rcv_se = Some(sd / (vals.len() as f64).sqrt());
        }
    }
    let t4_choice = {
        let best = top
            .iter()
            .filter(|g| !config.strict_pooled_search() || g.cv_failed_folds == 0)
            .filter_map(|g| g.rcv_mean_r2.map(|m| (m, g.rcv_se.unwrap_or(0.0))))
            .fold(
                (f64::NEG_INFINITY, 0.0),
                |a, b| if b.0 > a.0 { b } else { a },
            );
        // Band width is configurable because 1.0 SE admitted 10-14 of 16
        // candidates on measured runs, and a band containing most of the pool
        // is not a selection rule -- combined with the complexity tie-break it
        // degenerates to "pick whichever injected baseline is simplest".
        let threshold = best.0 - config.se_band * best.1;
        top.iter()
            .filter(|g| !config.strict_pooled_search() || g.cv_failed_folds == 0)
            .filter(|g| g.rcv_mean_r2.unwrap_or(f64::NEG_INFINITY) >= threshold)
            // Ties at equal complexity break by SCORE, then by hash. Breaking
            // straight to the hash made this an OLS bias, not a 1-SE rule: all
            // six baselines are complexity 1 and (pipe X (ridge lambda_frac=0.0))
            // hashes smallest, so plain OLS won every tie it entered -- on one
            // measured run it was the worst of five tied candidates.
            .min_by(|a, b| {
                a.complexity
                    .cmp(&b.complexity)
                    .then(
                        b.rcv_mean_r2
                            .unwrap_or(f64::NEG_INFINITY)
                            .total_cmp(&a.rcv_mean_r2.unwrap_or(f64::NEG_INFINITY)),
                    )
                    .then(a.semantic.cmp(&b.semantic))
            })
            .map(|g| g.semantic.clone())
    };
    if config.strict_pooled_search() && config.winner_rule == "rcv_1se" && t4_choice.is_none() {
        return Err("strict repeated CV produced no complete finalist".to_string());
    }

    // ---- T3: bootstrap-refit stability over the top pool ----
    let bags = 20usize;
    let n = y.len();
    let bag_sets: Vec<(Vec<usize>, Vec<usize>)> = (0..bags)
        .map(|b| {
            let seed = seed_from(&format!(
                "screen:{}:{}:boot:{b}",
                config.dataset_hash, config.master_seed
            ));
            let mut rng = Pcg32::new(seed, 0x424f_4f54);
            let mut inb = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            for _ in 0..n {
                let i = (rng.next_u32() as usize) % n;
                inb.push(i);
                seen[i] = true;
            }
            let oob: Vec<usize> = (0..n).filter(|&i| !seen[i]).collect();
            (inb, oob)
        })
        .collect();
    let boot_pairs: Vec<(usize, usize)> = (0..top.len())
        .flat_map(|g| (0..bags).map(move |b| (g, b)))
        .collect();
    let bootstrap_evals = boot_pairs.len();
    let boot_outs: Vec<((usize, usize), f64)> = boot_pairs
        .par_iter()
        .map(|&(g, b)| {
            (
                (g, b),
                eval_with_timeout(
                    train_path,
                    &top[g].canonical,
                    &bag_sets[b].0,
                    &bag_sets[b].1,
                ),
            )
        })
        .collect();
    let mut boot_scores: Vec<Vec<f64>> = vec![Vec::new(); top.len()];
    for ((g, _), v) in boot_outs {
        boot_scores[g].push(v);
    }
    for (g, gene) in top.iter_mut().enumerate() {
        let vals = &boot_scores[g];
        if !vals.is_empty() {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sd =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
            gene.boot_mean_r2 = Some(mean);
            gene.boot_sd_r2 = Some(sd);
        }
    }
    let t3_choice = top
        .iter()
        .filter(|g| g.boot_mean_r2.is_some())
        .max_by(|a, b| {
            let fa = a.boot_mean_r2.unwrap() - 0.5 * a.boot_sd_r2.unwrap_or(0.0);
            let fb = b.boot_mean_r2.unwrap() - 0.5 * b.boot_sd_r2.unwrap_or(0.0);
            fa.total_cmp(&fb).then(b.semantic.cmp(&a.semantic))
        })
        .map(|g| g.semantic.clone());

    // ---- T2: ensembles from a common OOF partition (first rcv repeat folds) ----
    let ensemble_fold_count = if config.strict_pooled_search() {
        config.final_cv_folds
    } else {
        usable_folds(y.len(), config.final_cv_folds)
    };
    let ens_folds = kfold(
        &y,
        ensemble_fold_count,
        seed_from(&format!(
            "screen:{}:{}:rcv:0",
            config.dataset_hash, config.master_seed
        )),
    );
    if config.strict_pooled_search() {
        validate_strict_fold_plan(y.len(), &ens_folds)?;
    }
    let ensemble_evals = top.len() * ens_folds.len();
    let ens_train: Vec<Vec<usize>> = (0..ens_folds.len())
        .map(|f| {
            ens_folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != f)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect()
        })
        .collect();
    // per-gene OOF prediction vector via child predict? cheap approximation:
    // per-gene per-fold R2 already captured; greedy selection needs prediction
    // vectors -- run one more pass collecting predictions in-process (no
    // timeout risk dominates: reuse eval budget bound by top_k * folds).
    let mut oof_preds: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; top.len()];
    for (g, gene) in top.iter().enumerate() {
        if let Ok(genome) = GenomeIr::parse(&gene.canonical) {
            for f in 0..ens_folds.len() {
                let (tx, ty) = subset(&x, &y, &ens_train[f]);
                let (vx, _) = subset(&x, &y, &ens_folds[f]);
                if let Ok(pred) = model::fit_predict(&genome, &tx, &ty, &vx) {
                    for (j, &row) in ens_folds[f].iter().enumerate() {
                        if let Some(v) = pred.get(j) {
                            oof_preds[g][row] = *v;
                        }
                    }
                }
            }
        }
    }
    let oof_rmse = |mix: &[f64]| -> f64 {
        let mut se = 0.0;
        let mut c = 0usize;
        for i in 0..n {
            if mix[i].is_finite() {
                se += (mix[i] - y[i]).powi(2);
                c += 1;
            }
        }
        if c == 0 {
            f64::INFINITY
        } else {
            (se / c as f64).sqrt()
        }
    };
    // uniform top-5
    let ens_uniform: Vec<String> = top.iter().take(5).map(|g| g.semantic.clone()).collect();
    // greedy Caruana on OOF predictions, with replacement, up to 8 picks
    let mut picks: Vec<usize> = Vec::new();
    let mut mix = vec![f64::NAN; n];
    for _ in 0..8 {
        let mut best: Option<(usize, f64)> = None;
        for (g, predictions) in oof_preds[..top.len()].iter().enumerate() {
            if predictions.iter().all(|v| !v.is_finite()) {
                continue;
            }
            let k = picks.len() as f64;
            let cand: Vec<f64> = (0..n)
                .map(|i| {
                    let a = if mix[i].is_finite() { mix[i] * k } else { 0.0 };
                    let b = if predictions[i].is_finite() {
                        predictions[i]
                    } else {
                        return f64::NAN;
                    };
                    (a + b) / (k + 1.0)
                })
                .collect();
            let r = oof_rmse(&cand);
            if best.is_none() || r < best.unwrap().1 {
                best = Some((g, r));
            }
        }
        match best {
            Some((g, _)) => {
                let k = picks.len() as f64;
                for i in 0..n {
                    let b = oof_preds[g][i];
                    if b.is_finite() {
                        let a = if mix[i].is_finite() { mix[i] * k } else { 0.0 };
                        mix[i] = (a + b) / (k + 1.0);
                    }
                }
                picks.push(g);
            }
            None => break,
        }
    }
    let ens_greedy: Vec<String> = picks.iter().map(|&g| top[g].semantic.clone()).collect();
    // winner: best CV mean, complexity/semantic tie-break via stable sort order
    let mut order: Vec<usize> = (0..top.len()).collect();
    order.sort_by(|&a, &b| {
        top[b]
            .cv7_mean_r2
            .unwrap_or(-2.0)
            .total_cmp(&top[a].cv7_mean_r2.unwrap_or(-2.0))
            .then_with(|| top[a].semantic.cmp(&top[b].semantic))
    });
    let mut winner = order.first().copied().ok_or("empty final population")?;
    // Captured BEFORE any winner_rule reassignment. Emitting top[winner] after
    // the rule ran made control and treatment the same value, so the A/B this
    // field exists to record was silently never recorded.
    let control_cv7_semantic = top[winner].semantic.clone();
    if config.winner_rule == "rcv_1se" {
        // The argmax of a CV score computed on the same rows the search already
        // optimised against is a winner's-curse estimator. Step back to the
        // simplest gene statistically indistinguishable from the best under
        // repeated CV. t4_choice is exactly that rule, already computed.
        if let Some(pick) = t4_choice
            .as_ref()
            .and_then(|sem| top.iter().position(|g| &g.semantic == sem))
        {
            winner = pick;
        }
    }

    let result_schema = if config.strict_pooled_search() {
        SCHEMA_RESULT_V2
    } else {
        SCHEMA_RESULT_V1
    };
    let total_model_fits = search_evals + cv_evals + rcv_evals + bootstrap_evals + ensemble_evals;
    let result = serde_json::json!({
        "schema": result_schema,
        "strategy": config.strategy,
        "dataset_hash": config.dataset_hash,
        "master_seed": config.master_seed,
        "search_fitness": if config.strict_pooled_search() {
            "complete_pooled_oof_r2"
        } else {
            "legacy_history"
        },
        "strict_fold_contract": if config.strict_pooled_search() {
            Some(serde_json::json!({
                "minimum_fit_rows": STRICT_MIN_FIT_ROWS,
                "minimum_hold_rows": STRICT_MIN_HOLD_ROWS,
                "required_search_folds": config.budget.cv_fitness,
                "required_final_folds": config.final_cv_folds,
                "complete_predictions_required": true,
                "fail_on_incomplete_candidate": true,
                "candidate_timeout_seconds": eval_timeout_seconds(),
            }))
        } else {
            None
        },
        "winner_genome": top[winner].canonical,
        "winner_semantic": top[winner].semantic,
        "winner_origin": top[winner].origin,
        "winner_cv_mean_r2": top[winner].cv7_mean_r2,
        "finales": {
            "control_cv7": control_cv7_semantic,
            "t3_bootstrap": t3_choice,
            "t4_rcv_1se": t4_choice,
            "t2_uniform5": ens_uniform,
            "t2_greedy": ens_greedy,
        },
        "search_evaluations": search_evals,
        "final_cv_evaluations": cv_evals,
        "model_fit_accounting": {
            "search": search_evals,
            "final_cv": cv_evals,
            "repeated_cv": rcv_evals,
            "bootstrap": bootstrap_evals,
            "ensemble_oof": ensemble_evals,
            "total": total_model_fits,
        },
        "runtime_seconds": started.elapsed().as_secs_f64(),
        "top": top,
        "generations": stats.iter().map(|s| serde_json::json!({
            "generation": s.generation,
            "evaluations": s.evaluations,
            "best_fitness": s.best_fitness,
            "invalid_members": s.invalid_members,
        })).collect::<Vec<_>>(),
        "test_csv_read": false,
    });
    fs::write(output_path, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| format!("write {output_path}: {e}"))?;
    Ok(())
}
