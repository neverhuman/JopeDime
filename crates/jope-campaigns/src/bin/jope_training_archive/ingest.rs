use super::*;

#[path = "ingest/reports.rs"]
mod reports;
pub(crate) use reports::{derive_annotations, ingest_truncations, open_db, recompute_flags};

#[allow(clippy::too_many_lines)]
pub(crate) fn ingest(root: &Path, dry_run: bool) -> Result<()> {
    let db_dir = root.join("db");
    fs::create_dir_all(&db_dir)?;
    let db_path = db_dir.join("history.sqlite3");

    let mut specs = discover_scq(root)?;
    let (imported, manifest) = discover_imported(root)?;
    specs.extend(imported);

    let timings = scq_timings()?;
    let started = now_ms();

    if dry_run {
        println!("{:<58} {:>6}  path", "run_key", "rows");
        for sp in &specs {
            let n = fs::read_to_string(&sp.path)
                .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
                .unwrap_or(0);
            println!("{:<58} {:>6}  {}", sp.run_key(), n, sp.path.display());
        }
        println!("\n{} sources, dry-run (nothing written)", specs.len());
        return Ok(());
    }

    let mut conn = Connection::open(&db_path)?;
    conn.execute_batch(SCHEMA_SQL)?;

    let mut total_rows = 0usize;
    let mut total_runs = 0usize;
    let mut all_problems: Vec<String> = vec![];
    #[allow(unused_assignments)]
    let mut truncated = 0usize;

    {
        let tx = conn.transaction()?;
        // flags/field_catalog are fully recomputed each pass
        tx.execute("DELETE FROM flag", [])?;
        tx.execute("DELETE FROM field_catalog", [])?;

        type FieldStats = (String, BTreeSet<String>, i64, i64, i64, String, String);
        let mut field_stats: BTreeMap<String, FieldStats> = BTreeMap::new();

        for sp in &specs {
            let bytes = fs::read(&sp.path)?;
            let text = String::from_utf8_lossy(&bytes).to_string();
            let src_sha = sha256_hex(&bytes);
            let src_mtime = mtime_ms(&sp.path)?;
            let (rows, problems) = parse_rows(&text, &sp.lane_kind);
            for p in &problems {
                all_problems.push(format!("{}: {p}", sp.run_key()));
            }
            if rows.is_empty() {
                continue;
            }

            let run_key = sp.run_key();

            // ---- source table
            tx.execute(
                "INSERT INTO source(sha256,origin_node,origin_path,archive_path,bytes,mtime_ms,rows,first_seen_ms,last_ingest_ms)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?8)
                 ON CONFLICT(sha256) DO UPDATE SET last_ingest_ms=?8, archive_path=?4, rows=?7",
                params![
                    src_sha,
                    sp.node.clone().unwrap_or_else(|| "xbabe2".into()),
                    sp.origin,
                    sp.path.display().to_string(),
                    bytes.len() as i64,
                    src_mtime,
                    rows.len() as i64,
                    started
                ],
            )?;
            for (node, path) in &sp.aliases {
                tx.execute(
                    "INSERT OR IGNORE INTO source_alias(sha256,origin_node,origin_path) VALUES(?1,?2,?3)",
                    params![src_sha, node, path],
                )?;
            }

            // ---- per-run provenance
            let timing = timings.get(&sp.run_id).cloned().unwrap_or_default();
            let spec = if sp.lane == "scq" {
                read_spec(&sp.run_id)
            } else {
                SpecInfo::default()
            };
            let warm = if sp.lane == "scq" {
                read_warm_start(&sp.run_id, sp.attempt)
            } else {
                None
            };

            let last = rows.last().unwrap();
            let last_elapsed = f(last.obj.get("elapsed_sec")).unwrap_or(0.0);
            let (started_ms, ended_ms, basis) = match (timing.running_ms, timing.end_ms) {
                (Some(a), Some(b2)) => (Some(a), Some(b2), "scq_events"),
                (Some(a), None) => (Some(a), Some(src_mtime), "scq_running+file_mtime"),
                _ => {
                    let st = src_mtime - (last_elapsed * 1000.0) as i64;
                    (Some(st), Some(src_mtime), "file_mtime-elapsed_sec")
                }
            };

            let result = find_train_result(sp).and_then(|p| {
                fs::read_to_string(&p)
                    .ok()
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            });
            let rank0 = if sp.sidecar_kind.as_deref() == Some("rank0_log") {
                sp.sidecar_result
                    .as_ref()
                    .and_then(|p| fs::read_to_string(p).ok())
                    .and_then(|t| {
                        t.lines()
                            .rev()
                            .find(|l| l.trim_start().starts_with('{'))
                            .and_then(|l| serde_json::from_str::<Value>(l).ok())
                    })
            } else {
                None
            };
            let manifest_side = if sp.sidecar_kind.as_deref() == Some("checkpoint_manifest") {
                sp.sidecar_result
                    .as_ref()
                    .and_then(|p| fs::read_to_string(p).ok())
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            } else {
                None
            };

            let model_version = last
                .obj
                .get("model_version")
                .and_then(|x| x.as_str())
                .map(|x| x.to_string())
                .or_else(|| sp.model_version_hint.clone());

            let anti_overfit = rows
                .iter()
                .any(|r| b(r.obj.get("anti_overfit_experiment")) == Some(1));
            let early_stopped = rows
                .iter()
                .any(|r| b(r.obj.get("macro_validation_early_stopped")) == Some(1));

            let (ck_sha, sel_epoch, sel_metric_name, sel_metric, epochs_planned) =
                match (&result, &rank0, &manifest_side) {
                    (Some(r), _, _) => (
                        s(r.get("checkpoint_sha256"))
                            .or_else(|| s(r.pointer("/driver/checkpoint_sha256"))),
                        i(r.get("selected_checkpoint_epoch")),
                        s(r.get("selected_checkpoint_metric_name")),
                        f(r.get("selected_checkpoint_metric")).or_else(|| f(r.get("best_metric"))),
                        i(r.get("epochs")),
                    ),
                    (None, Some(r), _) => (
                        s(r.get("checkpoint")),
                        None,
                        Some("val_pairwise_win_accuracy".to_string()),
                        f(r.get("best_metric")),
                        i(r.get("epochs")),
                    ),
                    (None, None, Some(m)) => (
                        s(m.get("model_hash")),
                        i(m.pointer("/metrics/historical_best_epoch")),
                        Some("val_pairwise_win_accuracy".to_string()),
                        f(m.pointer("/metrics/historical_best_metric_value")),
                        i(m.get("epoch")),
                    ),
                    _ => (None, None, None, None, i(last.obj.get("epochs"))),
                };

            let metric_definition = if !rows
                .iter()
                .any(|r| r.obj.contains_key("val_binary_pairwise_win_accuracy"))
            {
                "no_split_heads"
            } else if started_ms.unwrap_or(0) < METRIC_FIX_MS {
                "pre_task_types_fix"
            } else {
                "post_task_types_fix"
            };

            tx.execute(
                "INSERT INTO run(run_key,lane,lane_kind,in_policy_curve,run_id,attempt,watermark,node,generation,
                                 model_version,recipe,seed,sampler,corpus_snapshot,anti_overfit,
                                 warm_start_sha256,warm_start_path,warm_start_tensors_loaded,warm_start_tensors_total,
                                 result_checkpoint_sha256,selected_checkpoint_epoch,selected_checkpoint_metric_name,
                                 selected_checkpoint_metric,epochs_planned,rows_recorded,first_epoch,last_epoch,
                                 early_stopped,completed,started_at_ms,ended_at_ms,time_basis,scq_state,
                                 metric_definition,trainer_source_sha256,train_script_sha256,
                                 source_sha256,source_origin,spec_path,notes,source_live)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,
                        ?26,?27,?28,?29,?30,?31,?32,?33,?34,?35,?36,?37,?38,?39,?40,?41)
                 ON CONFLICT(run_key) DO UPDATE SET
                   node=excluded.node, model_version=excluded.model_version, recipe=excluded.recipe,
                   seed=excluded.seed, sampler=excluded.sampler, corpus_snapshot=excluded.corpus_snapshot,
                   anti_overfit=excluded.anti_overfit, warm_start_sha256=excluded.warm_start_sha256,
                   warm_start_path=excluded.warm_start_path,
                   warm_start_tensors_loaded=excluded.warm_start_tensors_loaded,
                   warm_start_tensors_total=excluded.warm_start_tensors_total,
                   result_checkpoint_sha256=excluded.result_checkpoint_sha256,
                   selected_checkpoint_epoch=excluded.selected_checkpoint_epoch,
                   selected_checkpoint_metric_name=excluded.selected_checkpoint_metric_name,
                   selected_checkpoint_metric=excluded.selected_checkpoint_metric,
                   epochs_planned=excluded.epochs_planned, rows_recorded=excluded.rows_recorded,
                   first_epoch=excluded.first_epoch, last_epoch=excluded.last_epoch,
                   early_stopped=excluded.early_stopped, completed=excluded.completed,
                   started_at_ms=excluded.started_at_ms, ended_at_ms=excluded.ended_at_ms,
                   time_basis=excluded.time_basis, scq_state=excluded.scq_state,
                   metric_definition=excluded.metric_definition,
                   trainer_source_sha256=excluded.trainer_source_sha256,
                   train_script_sha256=excluded.train_script_sha256,
                   source_sha256=excluded.source_sha256, source_origin=excluded.source_origin,
                   spec_path=excluded.spec_path, source_live=excluded.source_live",
                params![
                    run_key, sp.lane, sp.lane_kind, sp.in_policy_curve as i64, sp.run_id, sp.attempt,
                    sp.watermark,
                    sp.node.clone().or(timing.node.clone()),
                    sp.generation,
                    model_version, spec.recipe, spec.seed, spec.sampler, spec.corpus_snapshot,
                    anti_overfit as i64,
                    warm.as_ref().and_then(|(_, _, p)| {
                        Path::new(p).file_stem().map(|x| {
                            x.to_string_lossy().trim_end_matches(".model").to_string()
                        })
                    }).or(spec.warm_start_sha256.clone()),
                    warm.as_ref().map(|(_, _, p)| p.clone()),
                    warm.as_ref().map(|(a, _, _)| *a),
                    warm.as_ref().map(|(_, t, _)| *t),
                    ck_sha, sel_epoch, sel_metric_name, sel_metric, epochs_planned,
                    rows.len() as i64,
                    rows.first().map(|r| r.epoch),
                    rows.last().map(|r| r.epoch),
                    early_stopped as i64,
                    result.is_some() as i64,
                    started_ms, ended_ms, basis, timing.state,
                    metric_definition, spec.trainer_source_sha256, spec.train_script_sha256,
                    src_sha, sp.origin, spec.path, sp.notes,
                    {
                        // Does the ORIGINAL location still exist? For scq that is the
                        // run directory (which the GC prunes); for imported lanes it is
                        // the path the bytes were mirrored from.
                        let live = if sp.lane == "scq" {
                            Path::new(scq_runs_str()).join(&sp.run_id).is_dir()
                        } else {
                            sp.origin
                                .split_once(':')
                                .map(|(node, p)| node != "xbabe2" || Path::new(p).is_file())
                                .unwrap_or(false)
                        };
                        live as i64
                    }
                ],
            )?;
            total_runs += 1;

            // ---- per-epoch rows
            let mut stmt = tx.prepare(
                "INSERT INTO epoch(run_key,epoch,phase,row_ord,train_loss,val_loss,overfit_gap,lr,batch_size,epochs,
                                   elapsed_sec,epoch_sec,train_rows,validation_rows,model_version,device,world_size,
                                   val_fallback,macro_validation_best,macro_validation_early_stopped,
                                   val_conditional_advantage,val_conditional_score,val_pairwise_win_accuracy,
                                   val_binary_pairwise_win_accuracy,val_regression_pairwise_win_accuracy,
                                   val_ablation_pairwise_win,val_reward_cv_correlation,val_top1_historical_regret,
                                   val_seq_nll,val_head_accuracy_featmath,val_uncertainty_error_spearman,
                                   anti_overfit,wall_ms,raw_json,raw_sha256)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,
                        ?26,?27,?28,?29,?30,?31,?32,?33,?34,?35)
                 ON CONFLICT(run_key,epoch,phase) DO UPDATE SET
                   row_ord=excluded.row_ord, train_loss=excluded.train_loss, val_loss=excluded.val_loss,
                   overfit_gap=excluded.overfit_gap, lr=excluded.lr, batch_size=excluded.batch_size,
                   epochs=excluded.epochs, elapsed_sec=excluded.elapsed_sec, epoch_sec=excluded.epoch_sec,
                   train_rows=excluded.train_rows, validation_rows=excluded.validation_rows,
                   model_version=excluded.model_version, device=excluded.device, world_size=excluded.world_size,
                   val_fallback=excluded.val_fallback, macro_validation_best=excluded.macro_validation_best,
                   macro_validation_early_stopped=excluded.macro_validation_early_stopped,
                   val_conditional_advantage=excluded.val_conditional_advantage,
                   val_conditional_score=excluded.val_conditional_score,
                   val_pairwise_win_accuracy=excluded.val_pairwise_win_accuracy,
                   val_binary_pairwise_win_accuracy=excluded.val_binary_pairwise_win_accuracy,
                   val_regression_pairwise_win_accuracy=excluded.val_regression_pairwise_win_accuracy,
                   val_ablation_pairwise_win=excluded.val_ablation_pairwise_win,
                   val_reward_cv_correlation=excluded.val_reward_cv_correlation,
                   val_top1_historical_regret=excluded.val_top1_historical_regret,
                   val_seq_nll=excluded.val_seq_nll,
                   val_head_accuracy_featmath=excluded.val_head_accuracy_featmath,
                   val_uncertainty_error_spearman=excluded.val_uncertainty_error_spearman,
                   anti_overfit=excluded.anti_overfit, wall_ms=excluded.wall_ms,
                   raw_json=excluded.raw_json, raw_sha256=excluded.raw_sha256",
            )?;
            let mut fstmt = tx.prepare(
                "INSERT INTO epoch_field(run_key,epoch,phase,key,num,txt,json) VALUES(?1,?2,?3,?4,?5,?6,?7)
                 ON CONFLICT(run_key,epoch,phase,key) DO UPDATE SET num=excluded.num, txt=excluded.txt, json=excluded.json",
            )?;

            for r in &rows {
                let o = &r.obj;
                let wall = started_ms
                    .map(|st| st + (f(o.get("elapsed_sec")).unwrap_or(0.0) * 1000.0) as i64);
                stmt.execute(params![
                    run_key,
                    r.epoch,
                    r.phase,
                    r.row_ord,
                    f(o.get("train_loss")).or_else(|| f(o.get("loss"))),
                    f(o.get("val_loss")),
                    f(o.get("overfit_gap")),
                    f(o.get("lr")),
                    i(o.get("batch_size")),
                    i(o.get("epochs")),
                    f(o.get("elapsed_sec")),
                    f(o.get("epoch_sec")),
                    i(o.get("train_rows")),
                    i(o.get("validation_rows")),
                    s(o.get("model_version")),
                    s(o.get("device")),
                    i(o.get("world_size")),
                    b(o.get("val_fallback")),
                    f(o.get("macro_validation_best")),
                    b(o.get("macro_validation_early_stopped")),
                    f(o.get("val_conditional_advantage")),
                    f(o.get("val_conditional_score")),
                    f(o.get("val_pairwise_win_accuracy")),
                    f(o.get("val_binary_pairwise_win_accuracy")),
                    f(o.get("val_regression_pairwise_win_accuracy")),
                    f(o.get("val_ablation_pairwise_win")),
                    f(o.get("val_reward_cv_correlation")),
                    f(o.get("val_top1_historical_regret")),
                    f(o.get("val_seq_nll")),
                    f(o.get("val_head_accuracy_featmath")),
                    f(o.get("val_uncertainty_error_spearman")),
                    b(o.get("anti_overfit_experiment")),
                    wall,
                    r.raw,
                    sha256_hex(r.raw.as_bytes()),
                ])?;

                for (k, v) in o.iter() {
                    let (num, txt, json) = match v {
                        Value::Number(n) => (n.as_f64(), None, None),
                        Value::Bool(x) => (Some(*x as i64 as f64), None, None),
                        Value::String(x) => (None, Some(x.clone()), None),
                        Value::Null => (None, None, None),
                        other => (None, None, Some(other.to_string())),
                    };
                    fstmt.execute(params![run_key, r.epoch, r.phase, k, num, txt, json])?;
                    let kind = match v {
                        Value::Number(_) => "number",
                        Value::Bool(_) => "bool",
                        Value::String(_) => "string",
                        Value::Null => "null",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                    };
                    let e = field_stats.entry(k.clone()).or_insert_with(|| {
                        (
                            kind.to_string(),
                            BTreeSet::new(),
                            0,
                            i64::MAX,
                            i64::MIN,
                            String::new(),
                            String::new(),
                        )
                    });
                    e.1.insert(run_key.clone());
                    e.2 += 1;
                    let w = wall.unwrap_or(started);
                    if w < e.3 {
                        e.3 = w;
                        e.5 = run_key.clone();
                    }
                    if w > e.4 {
                        e.4 = w;
                        e.6 = run_key.clone();
                    }
                }
                total_rows += 1;
            }
        }

        for (k, (kind, runs, rows_n, first, last, frun, lrun)) in field_stats {
            tx.execute(
                "INSERT INTO field_catalog(key,kind,runs_present,rows_present,first_seen_ms,last_seen_ms,first_run_key,last_run_key)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
                params![k, kind, runs.len() as i64, rows_n, first, last, frun, lrun],
            )?;
        }

        // ---- annotations (curve-honesty markers)
        let anns: Vec<(&str, i64, &str, &str, &str, &str)> = vec![
            (
                "metric-fix-task-types",
                METRIC_FIX_MS,
                "metric_definition_change",
                "validation metrics: binary rows were scored by the REGRESSION heads before this point",
                "policy_v3_validation_metrics called model.reward_stats(z, token_ids, pad_mask) with no task_types, so _binary_task_mask defaulted every row to regression. Training was correct; MEASUREMENT was wrong. val_conditional_advantage, val_conditional_score and val_binary_pairwise_win_accuracy are NOT comparable across this line.",
                "jope-prime/src/jope/policy/train_v3.py mtime 2026-07-27T21:27:52Z. Runs a-g3 and a-g7 have byte-identical specs (diff = the job id line only: same train_v3.py provenance sha 358acf21..., same test_exact_train.py bc2652da..., same corpus snapshot-001805437-full_consolidation, same warm-start ff95a336...); a-g3 started 19:58:24Z and reports val_conditional_advantage -0.0602..-0.0869, a-g7 started 21:43:31Z and reports +0.0314..+0.0390. Every spec sets PYTHONPATH=~/jope-prime/src so the executed validation code is the LIVE file, not the staged provenance copy.",
            ),
            (
                "equal-quota-sampler",
                1_785_204_089_000,
                "data_distribution_change",
                "equal-quota sampler eq32v32c128 replaces full_consolidation / recent_plus_replay",
                "Validation loss magnitudes are not comparable across this line: the validation SET changed, not just the model.",
                "spec [[input]] name=manifest source path moves from training/snapshot-001805437-full_consolidation/ to training/snapshot-001805437-full_consolidation-eq32v32c128/ at a-g8.",
            ),
            (
                "anti-overfit-experiment",
                1_785_204_089_000,
                "regularisation_change",
                "PrimeAntiOverfitExperiment.v1 enabled (token dropout, embedding jitter, weight decay, dataset row cap)",
                "Runs with anti_overfit=1 carry different regularisation than the rest of the curve.",
                "anti_overfit_experiment key present only in a-g8, a-g9 and c-r000002 rows.",
            ),
        ];
        for (id, at, kind, label, detail, evidence) in anns {
            tx.execute(
                "INSERT INTO annotation(id,at_ms,kind,label,detail,evidence) VALUES(?1,?2,?3,?4,?5,?6)
                 ON CONFLICT(id) DO UPDATE SET at_ms=excluded.at_ms, kind=excluded.kind, label=excluded.label,
                   detail=excluded.detail, evidence=excluded.evidence",
                params![id, at, kind, label, detail, evidence],
            )?;
        }

        derive_annotations(&tx)?;

        // ---- proven-lost ledger, from the imported manifest
        if let Some(list) = manifest.get("proven_lost").and_then(|x| x.as_array()) {
            for (n, g) in list.iter().enumerate() {
                tx.execute(
                    "INSERT INTO gap(id,lane,what,evidence,approx_window) VALUES(?1,?2,?3,?4,?5)
                     ON CONFLICT(id) DO UPDATE SET lane=excluded.lane, what=excluded.what,
                       evidence=excluded.evidence, approx_window=excluded.approx_window",
                    params![
                        format!("gap-{n:02}"),
                        g["lane"].as_str().unwrap_or("?"),
                        g["what"].as_str().unwrap_or("?"),
                        g["evidence"].as_str().unwrap_or("?"),
                        g["approx_window"].as_str()
                    ],
                )?;
            }
        }

        // ---- sanity flags
        recompute_flags(&tx)?;
        truncated = ingest_truncations(&tx, root)?;
        for p in &all_problems {
            let (rk, det) = p.split_once(": ").unwrap_or(("*", p.as_str()));
            tx.execute(
                "INSERT OR REPLACE INTO flag(run_key,epoch,phase,code,severity,detail)
                 VALUES(?1,-1,'*','row_parse_anomaly','warn',?2)",
                params![rk, det],
            )?;
        }

        tx.execute(
            "INSERT INTO meta(key,value) VALUES('schema','JopeTrainingHistoryDb.v1')
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [],
        )?;
        tx.execute(
            "INSERT INTO meta(key,value) VALUES('last_ingest_ms',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![started.to_string()],
        )?;
        tx.execute(
            "INSERT INTO meta(key,value) VALUES('metric_fix_ms',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![METRIC_FIX_MS.to_string()],
        )?;
        tx.commit()?;
    }

    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    println!(
        "ingested {total_runs} runs, {total_rows} epoch rows -> {}",
        db_path.display()
    );
    if !all_problems.is_empty() {
        println!("{} row anomalies recorded as flags", all_problems.len());
    }
    if truncated > 0 {
        println!("{truncated} truncated tail(s) recorded as flags + gaps");
    }
    Ok(())
}
