use super::*;

pub(super) const TRANSFORMS: &[&str] = &[
    "standardize",
    "robust_scale",
    "quantile_bins",
    "winsor_clip",
    "sigmoid",
    "symlog",
];
// Rejection sampling of the existing engine's grammar, not an unbounded evaluator.
// Heads are ridge-family only; actual bound task heads are fitted, never surrogate scored.
pub(super) fn safe_genome(source: &str) -> Result<String, Error> {
    let GenomeIr::Pipeline(mut dsl) = GenomeIr::parse(source)? else {
        return Err("pipeline required".into());
    };
    if dsl.stages.is_empty() || dsl.stages.len() > 3 {
        return Err("stage bound".into());
    }
    let mut head = dsl.stages.pop().unwrap();
    if head.op != "ridge" {
        return Err("head outside bounded ridge-family grammar".into());
    }
    for stage in &dsl.stages {
        if !TRANSFORMS.contains(&stage.op.as_str())
            || matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        {
            return Err("transform outside bounded width-preserving grammar".into());
        }
    }
    head.params.retain(|k, _| k == "lambda_frac");
    let lambda = head.get_f64("lambda_frac", 0.02).clamp(0.00001, 1.0);
    head.params
        .insert("lambda_frac".into(), ParamValue::Float(lambda));
    dsl.stages.push(head);
    Ok(GenomeIr::Pipeline(dsl).canonical())
}
pub(super) fn bound_genome(source: &str, task: TaskKind) -> Result<GenomeIr, Error> {
    let GenomeIr::Pipeline(mut dsl) = GenomeIr::parse(source)? else {
        return Err("pipeline required".into());
    };
    let mut head = dsl.stages.pop().ok_or("no head")?;
    let lambda = head.get_f64("lambda_frac", 0.02);
    if task == TaskKind::Binary {
        head = Stage::new("logit_ridge");
        head.params.insert("l2".into(), ParamValue::Float(lambda));
    }
    if task == TaskKind::Multiclass {
        head = Stage::new("multiclass_ridge");
        head.params.insert("l2".into(), ParamValue::Float(lambda));
    }
    dsl.stages.push(head);
    let GenomeIr::Pipeline(input) =
        GenomeIr::parse("(pipe X (input numeric=median missing=none categorical=native) (ridge))")?
    else {
        unreachable!()
    };
    dsl.stages.insert(0, input.stages[0].clone());
    Ok(GenomeIr::Pipeline(dsl))
}
pub(super) fn metric(task: TaskKind, y: &[f64], p: &[f64]) -> Result<Value, Error> {
    if y.len() != p.len() || y.is_empty() || p.iter().any(|v| !v.is_finite()) {
        return Err("invalid prediction vector".into());
    }
    match task {
        TaskKind::Binary => {
            let eps = 1e-15;
            let loss = y
                .iter()
                .zip(p)
                .map(|(&y, &p)| {
                    if y == 0.0 {
                        -(1.0 - p.clamp(eps, 1.0 - eps)).ln()
                    } else if y == 1.0 {
                        -p.clamp(eps, 1.0 - eps).ln()
                    } else {
                        -eps.ln()
                    }
                })
                .sum::<f64>()
                / y.len() as f64;
            let accuracy = y
                .iter()
                .zip(p)
                .filter(|(y, p)| **y == if **p >= 0.5 { 1.0 } else { 0.0 })
                .count() as f64
                / y.len() as f64;
            Ok(
                json!({"selection_loss":loss,"logloss":loss,"accuracy":accuracy,"unknown_target_count":y.iter().filter(|&&v|v<0.0).count()}),
            )
        }
        TaskKind::Multiclass => {
            let f1 = feat_math::macro_f1(y, p);
            let accuracy = y.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / y.len() as f64;
            Ok(
                json!({"selection_loss":1.0-f1,"macro_f1":f1,"accuracy":accuracy,"unknown_target_count":y.iter().filter(|&&v|v<0.0).count()}),
            )
        }
        _ => {
            let rmse = feat_math::rmse(y, p);
            let r2 = feat_math::r2(y, p);
            if !rmse.is_finite() {
                return Err("nonfinite score".into());
            }
            Ok(
                json!({"selection_loss":rmse,"rmse":rmse,"r2":if r2.is_finite() && y.iter().any(|v|*v!=y[0]){Some(r2)}else{None},"r2_defined":y.iter().any(|v|*v!=y[0])}),
            )
        }
    }
}
pub(super) fn derive_seed(seed: u64, slot: usize, draw: usize) -> u64 {
    let b = Sha256::digest(format!("{PROTOCOL}:candidate:{seed}:{slot}:{draw}"));
    u64::from_le_bytes(b[..8].try_into().unwrap())
}
pub(super) struct Measured {
    dsl: String,
    loss: f64,
    trace: usize,
}
#[cfg(test)]
pub(super) fn evaluate(
    p: &Prepared,
    job: &Job,
    seed: u64,
    max_seconds: u64,
) -> Result<Value, Error> {
    evaluate_with_log(p, job, seed, max_seconds, None)
}
pub(super) fn evaluate_with_log(
    p: &Prepared,
    job: &Job,
    seed: u64,
    max_seconds: u64,
    log: Option<&LogContext>,
) -> Result<Value, Error> {
    let start = Instant::now();
    let mut trace: Vec<Value> = Vec::new();
    let mut measured: Vec<Measured> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut champion: Option<(BoundModel, String, f64, usize)> = None;
    let mut stop = "candidate_budget";
    let mut rejected = 0usize;
    for slot in 0..CANDIDATES {
        if slot > 0 && start.elapsed().as_secs() >= max_seconds {
            stop = "seconds_before_next_fit";
            break;
        }
        let mut picked = None;
        for draw in 0..2048 {
            let childseed = derive_seed(seed, slot, draw);
            let (raw, origin, parents) = if slot == 0 {
                (
                    String::from("(pipe X (ridge lambda_frac=0.02))"),
                    "bootstrap",
                    Vec::new(),
                )
            } else if slot < 4 || measured.is_empty() {
                (
                    evolve::random_canonical_pipeline(childseed, 2),
                    "random",
                    Vec::new(),
                )
            } else {
                let mut ranking: Vec<&Measured> = measured.iter().collect();
                ranking.sort_by(|a, b| a.loss.total_cmp(&b.loss).then(a.dsl.cmp(&b.dsl)));
                let count = ranking.len().min(4);
                let a = ranking[(childseed as usize) % count];
                if slot % 3 == 0 && count > 1 {
                    let b = ranking[((childseed >> 16) as usize) % count];
                    (
                        evolve::crossover_canonical_pipelines(&a.dsl, &b.dsl, childseed, 2)
                            .map_err(|e| e.to_string())?,
                        "crossover",
                        vec![a.trace, b.trace],
                    )
                } else {
                    (
                        evolve::mutate_canonical_pipeline(&a.dsl, childseed, 2)
                            .map_err(|e| e.to_string())?,
                        "mutate",
                        vec![a.trace],
                    )
                }
            };
            if let Ok(dsl) = safe_genome(&raw) {
                if seen.insert(dsl.clone()) {
                    picked = Some((dsl, origin, parents, childseed, draw));
                    break;
                }
            }
            rejected += 1;
        }
        let Some((dsl, origin, parents, childseed, draw)) = picked else {
            stop = "bounded_grammar_exhausted";
            break;
        };
        let genome = bound_genome(&dsl, p.task.kind)?;
        let actual = genome.canonical();
        let fit_start = Instant::now();
        let fit_start_utc_us = utc_us();
        let candidate_start_relative_ns = start.elapsed().as_nanos().to_string();
        let before = memory();
        let mut row = json!({"candidate_index":slot,"generation":if slot<4 {0}else{1+(slot-4)/4},"origin":origin,"parent_candidate_indices":parents,
            "seed":childseed,"draw":draw,"search_dsl":dsl,"executed_dsl":actual,"program_sha256":hash(actual.as_bytes()),"memory_before":before});
        log_event(
            log,
            "candidate_start",
            json!({"candidate_index":slot,"origin":origin,"program_sha256":hash(actual.as_bytes())}),
        );
        // The restricted operators avoid the unsafe/unbounded kernels; no panic recovery with panic=abort.
        let fitted = feat_math::fit_table(&genome, &p.task, &p.schema, &p.fit, &p.fit_y);
        row["fit_start_utc_us"] = json!(fit_start_utc_us);
        row["fit_end_utc_us"] = json!(utc_us());
        row["candidate_start_relative_ns"] = json!(candidate_start_relative_ns);
        match fitted {
            Err(e) => {
                row["status"] = json!("fit_failed");
                row["error"] = json!(e.to_string());
            }
            Ok(model) => {
                row["selection_predict_start_utc_us"] = json!(utc_us());
                let prediction = model.predict_table(&p.selection);
                row["selection_predict_end_utc_us"] = json!(utc_us());
                match prediction
                    .map_err(|e| -> Error { e.into() })
                    .and_then(|pred| metric(p.task.kind, &p.selection_y, &pred))
                {
                    Err(e) => {
                        row["status"] = json!("selection_failed");
                        row["error"] = json!(e.to_string());
                    }
                    Ok(metrics) => {
                        let loss = metrics["selection_loss"].as_f64().ok_or("invalid loss")?;
                        row["status"] = json!("measured");
                        row["selection_metrics"] = metrics;
                        row["model_sha256"] = json!(hash(&serde_json::to_vec(&model)?));
                        measured.push(Measured {
                            dsl: dsl.clone(),
                            loss,
                            trace: slot,
                        });
                        if champion
                            .as_ref()
                            .is_none_or(|(_, d, l, _)| loss < *l || (loss == *l && actual < *d))
                        {
                            champion = Some((model, actual, loss, slot));
                        }
                    }
                }
            }
        }
        row["candidate_s"] = json!(fit_start.elapsed().as_secs_f64());
        row["candidate_end_utc_us"] = json!(utc_us());
        row["candidate_end_relative_ns"] = json!(start.elapsed().as_nanos().to_string());
        row["memory_after"] = memory();
        log_event(
            log,
            "candidate_end",
            json!({"candidate_index":slot,"status":row["status"],"model_sha256":row["model_sha256"],"candidate_s":row["candidate_s"]}),
        );
        trace.push(row);
    }
    let (model, dsl, _, index) = champion.ok_or("no GP candidate measured successfully")?;
    // Champion is frozen before audit predictions/labels are consulted. No refit or selection on audit.
    let model_bytes = serde_json::to_vec(&model)?;
    if model_bytes.len() > 3_500_000 {
        return Err("fitted artifact exceeds synchronous result bound".into());
    }
    let audit_predictions = model.predict_table(&p.audit)?;
    let audit_metrics = metric(p.task.kind, &p.audit_y, &audit_predictions)?;
    let selection_predictions = model.predict_table(&p.selection)?;
    let selection_metrics = metric(p.task.kind, &p.selection_y, &selection_predictions)?;
    let replay: BoundModel = serde_json::from_slice(&model_bytes)?;
    replay.validate_binding()?;
    let replay_predictions = replay.predict_table(&p.audit)?;
    let prediction_bytes: Vec<u8> = audit_predictions
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let replay_bytes: Vec<u8> = replay_predictions
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    if prediction_bytes != replay_bytes {
        return Err("serialized fitted-model replay mismatch".into());
    }
    Ok(
        json!({"schema":"jope.lambda-gp-job.v1","protocol":PROTOCOL,"dataset":job,
        "benchmark_valid":false,"canonical_learning_admitted":false,"forecast_valid":false,"scope":"isolated numeric corpus GP screening; not canonical evaluator qualification",
        "task_original":job.metadata.get("catalog_task").cloned().unwrap_or_else(||json!(job.task)),"task_executed":p.task,"task_alias_note":if job.metadata.get("catalog_task").is_some_and(|v|v=="multi_label") || job.task=="multi_label"{"legacy single target class code, mapped to multiclass"}else if job.task=="timeseries"{"CSV-order blocked screening; original chronological provenance unverified"}else{"none"},
        "seed":seed,"fold_policy":if job.task=="timeseries"{"csv_order_60_20_20"}else{"sha256_row_60_20_20"},"fold":p.fold,"fold_sha256":hash(&serde_json::to_vec(&p.fold)?),
        "fit_rows":p.fit.rows.len(),"selection_rows":p.selection.rows.len(),"audit_rows":p.audit.rows.len(),
        "feature_projection":{"policy":"fit_only_top64_population_variance_tie_original_index_v1","original_features":p.original_features,"selected_original_column_indices":p.columns,"schema":p.schema},
        "target_vocabulary_original_values":p.labels,"unknown_class_policy":"unseen labels encoded -1 and count as errors; fit-only vocabulary",
        "search":{"engine":"feat_math::evolve canonical mutation/crossover + actual task-bound fit_table","grammar":"max2 nonexpanding transforms; ridge-family actual task heads","candidate_limit":CANDIDATES,"candidate_attempts":trace.len(),"measured_candidates":measured.len(),"evolved_measured_candidates":trace.iter().filter(|v| v["status"]=="measured" && (v["origin"]=="mutate" || v["origin"]=="crossover")).count(),"rejected_grammar_draws":rejected,"stop_reason":stop,"soft_seconds":max_seconds,"trace":trace},
        "champion":{"candidate_index":index,"dsl":dsl,"program_sha256":hash(dsl.as_bytes()),"model":model,"model_sha256":hash(&model_bytes),"model_json_bytes":model_bytes.len(),"selection_metrics":selection_metrics,"audit_metrics":audit_metrics,
            "audit_predictions":audit_predictions,"audit_targets_encoded":p.audit_y,"audit_predictions_sha256":hash(&prediction_bytes),"serialized_model_replay_exact":true},
        "training_s":start.elapsed().as_secs_f64(),"memory_after":memory()}),
    )
}
