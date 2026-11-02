use super::*;

pub(super) fn cmd_stats(root: &Path) -> Result<()> {
    let conn = open_db(root)?;
    let (runs, rows): (i64, i64) = conn.query_row(
        "SELECT (SELECT COUNT(*) FROM run),(SELECT COUNT(*) FROM epoch)",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    println!("archive: {}", root.join("db/history.sqlite3").display());
    println!("runs={runs}  epoch_rows={rows}");
    println!("\nby lane:");
    let mut st = conn.prepare(
        "SELECT lane, lane_kind, in_policy_curve, COUNT(*), SUM(rows_recorded),
                MIN(started_at_ms), MAX(ended_at_ms)
         FROM run GROUP BY lane, lane_kind, in_policy_curve ORDER BY MIN(started_at_ms)",
    )?;
    println!(
        "{:<22} {:<6} {:<6} {:>5} {:>7}  window (UTC)",
        "lane", "kind", "curve", "runs", "rows"
    );
    for r in st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, i64>(4)?,
            r.get::<_, Option<i64>>(5)?,
            r.get::<_, Option<i64>>(6)?,
        ))
    })? {
        let (lane, kind, curve, n, rows, a, b2) = r?;
        println!(
            "{:<22} {:<6} {:<6} {:>5} {:>7}  {} .. {}",
            lane,
            kind,
            if curve == 1 { "yes" } else { "no" },
            n,
            rows,
            fmt_ms(a),
            fmt_ms(b2)
        );
    }
    println!("\npolicy curve (in_policy_curve=1, lane_kind='epoch'):");
    let (pr, pe): (i64, i64) = conn.query_row(
        "SELECT COUNT(DISTINCT r.run_key), COUNT(*) FROM epoch e JOIN run r USING(run_key)
         WHERE r.in_policy_curve=1 AND r.lane_kind='epoch'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    println!("  {pr} runs, {pe} epochs");
    println!(
        "\ndistinct metric keys tracked: {}",
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM field_catalog", [], |r| r.get(0))?
    );
    println!(
        "epoch_field cells:            {}",
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM epoch_field", [], |r| r.get(0))?
    );
    println!("\nflags:");
    let mut fs2 = conn.prepare("SELECT severity,code,COUNT(*) FROM flag GROUP BY severity,code ORDER BY severity DESC, COUNT(*) DESC")?;
    for r in fs2.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
        ))
    })? {
        let (sev, code, n) = r?;
        println!("  {:<6} {:<34} {}", sev, code, n);
    }
    println!("\nknown-lost history:");
    let mut gs = conn.prepare("SELECT lane,what,approx_window FROM gap ORDER BY id")?;
    for r in gs.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<String>>(2)?,
        ))
    })? {
        let (lane, what, w) = r?;
        println!(
            "  [{}] {} ({})",
            lane,
            what,
            w.unwrap_or_else(|| "?".into())
        );
    }
    Ok(())
}

pub(super) fn fmt_ms(ms: Option<i64>) -> String {
    let Some(ms) = ms else { return "-".into() };
    // civil_from_days
    let secs = ms / 1000;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub(super) fn cmd_runs(root: &Path, lane: Option<String>) -> Result<()> {
    let conn = open_db(root)?;
    let sql = "SELECT run_key,lane,node,model_version,recipe,sampler,corpus_snapshot,anti_overfit,
                      warm_start_sha256,warm_start_tensors_loaded,warm_start_tensors_total,
                      result_checkpoint_sha256,selected_checkpoint_epoch,rows_recorded,
                      first_epoch,last_epoch,started_at_ms,ended_at_ms,metric_definition,
                      early_stopped,completed,scq_state,time_basis
               FROM run WHERE (?1 IS NULL OR lane=?1) ORDER BY started_at_ms";
    let mut st = conn.prepare(sql)?;
    let mut n = 0;
    for row in st.query_map(params![lane], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, Option<String>>(3)?,
            r.get::<_, Option<String>>(4)?,
            r.get::<_, Option<String>>(5)?,
            r.get::<_, Option<String>>(6)?,
            r.get::<_, Option<i64>>(7)?,
            r.get::<_, Option<String>>(8)?,
            r.get::<_, Option<i64>>(9)?,
            r.get::<_, Option<i64>>(10)?,
            r.get::<_, Option<String>>(11)?,
            r.get::<_, Option<i64>>(12)?,
            r.get::<_, i64>(13)?,
            r.get::<_, Option<i64>>(14)?,
            r.get::<_, Option<i64>>(15)?,
            r.get::<_, Option<i64>>(16)?,
            r.get::<_, Option<i64>>(17)?,
            r.get::<_, Option<String>>(18)?,
            r.get::<_, Option<i64>>(19)?,
            r.get::<_, Option<i64>>(20)?,
            r.get::<_, Option<String>>(21)?,
            r.get::<_, Option<String>>(22)?,
        ))
    })? {
        let t = row?;
        n += 1;
        println!("── {}", t.0);
        println!(
            "   started {}  ended {}  ({})",
            fmt_ms(t.16),
            fmt_ms(t.17),
            t.22.unwrap_or_default()
        );
        println!(
            "   node={:<8} model={:<26} rows={} epochs {}..{}",
            t.2.unwrap_or_else(|| "?".into()),
            t.3.unwrap_or_else(|| "?".into()),
            t.13,
            t.14.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            t.15.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        );
        println!(
            "   recipe={:<4} sampler={:<12} corpus={}",
            t.4.unwrap_or_else(|| "-".into()),
            t.5.unwrap_or_else(|| "-".into()),
            t.6.unwrap_or_else(|| "-".into())
        );
        println!(
            "   warm_start={} ({}/{} tensors)",
            t.8.as_deref().map(|x| &x[..x.len().min(16)]).unwrap_or("-"),
            t.9.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            t.10.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        );
        println!("   result_ckpt={} selected_epoch={} anti_overfit={} early_stopped={} complete={} scq={}",
                 t.11.as_deref().map(|x| &x[..x.len().min(16)]).unwrap_or("-"),
                 t.12.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
                 t.7.unwrap_or(0), t.19.unwrap_or(0), t.20.unwrap_or(0),
                 t.21.unwrap_or_else(|| "-".into()));
        println!(
            "   metric_definition={}",
            t.18.unwrap_or_else(|| "?".into())
        );
    }
    println!("\n{n} runs");
    Ok(())
}

pub(super) fn cmd_flags(root: &Path, severity: Option<String>) -> Result<()> {
    let conn = open_db(root)?;
    let mut st = conn.prepare(
        "SELECT severity,code,run_key,epoch,phase,detail FROM flag
         WHERE (?1 IS NULL OR severity=?1)
         ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END, code, run_key, epoch",
    )?;
    let mut n = 0;
    for r in st.query_map(params![severity], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, String>(4)?,
            r.get::<_, Option<String>>(5)?,
        ))
    })? {
        let (sev, code, rk, ep, ph, det) = r?;
        n += 1;
        let loc = if ep < 0 {
            "run".to_string()
        } else {
            format!("e{ep}/{ph}")
        };
        println!(
            "{:<6} {:<32} {:<52} {:<10} {}",
            sev,
            code,
            rk,
            loc,
            det.unwrap_or_default()
        );
    }
    println!("\n{n} flags");
    Ok(())
}

pub(super) fn cmd_verify(root: &Path) -> Result<()> {
    let conn = open_db(root)?;
    let mut st = conn.prepare("SELECT run_key,epoch,phase,raw_json,raw_sha256 FROM epoch")?;
    let mut bad = 0;
    let mut n = 0;
    for r in st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, String>(4)?,
        ))
    })? {
        let (rk, ep, ph, raw, sha) = r?;
        n += 1;
        if sha256_hex(raw.as_bytes()) != sha {
            bad += 1;
            println!("HASH MISMATCH {rk} e{ep}/{ph}");
        }
    }
    println!("verified {n} rows, {bad} hash mismatches");

    // reconcile against sources that are still readable
    let mut ss =
        conn.prepare("SELECT run_key,source_sha256,source_origin,rows_recorded FROM run")?;
    let mut drift = 0;
    for r in ss.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, Option<String>>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, i64>(3)?,
        ))
    })? {
        let (rk, sha, origin, rows) = r?;
        let Some(origin) = origin else { continue };
        let Some((node, path)) = origin.split_once(':') else {
            continue;
        };
        if node != "xbabe2" {
            continue;
        }
        let p = Path::new(path);
        if !p.is_file() {
            println!("source gone (archive is the only copy): {rk} -> {path}");
            continue;
        }
        let bytes = fs::read(p)?;
        let live_rows = String::from_utf8_lossy(&bytes)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count() as i64;
        let live_sha = sha256_hex(&bytes);
        if Some(&live_sha) != sha.as_ref() {
            if live_rows > rows {
                println!("source ADVANCED (re-ingest): {rk} {rows} -> {live_rows} rows");
            } else {
                println!("source CHANGED: {rk} archived {rows} rows, live {live_rows}");
            }
            drift += 1;
        }
    }
    println!("{drift} sources drifted from the archived copy");
    if bad > 0 {
        bail!("{bad} rows failed hash verification");
    }
    Ok(())
}

// ------------------------------------------------------------------ export

#[derive(Debug)]
struct Point {
    gidx: i64,
    run_key: String,
    lane: String,
    epoch: i64,
    phase: String,
    train_loss: Option<f64>,
    val_loss: Option<f64>,
    overfit_gap: Option<f64>,
    val_pairwise: Option<f64>,
    val_cond_adv: Option<f64>,
    wall_ms: Option<i64>,
    metric_definition: String,
    val_fallback: i64,
    early_stopped: i64,
}

pub(super) fn load_points(conn: &Connection, policy_only: bool) -> Result<Vec<Point>> {
    let sql = format!(
        "SELECT e.run_key, r.lane, e.epoch, e.phase, e.train_loss, e.val_loss, e.overfit_gap,
                e.val_pairwise_win_accuracy, e.val_conditional_advantage, e.wall_ms,
                r.metric_definition, COALESCE(e.val_fallback,0), COALESCE(e.macro_validation_early_stopped,0)
         FROM epoch e JOIN run r USING(run_key)
         {}
         ORDER BY r.started_at_ms, r.run_key, e.epoch, e.row_ord",
        if policy_only { "WHERE r.in_policy_curve=1 AND r.lane_kind='epoch'" } else { "" }
    );
    let mut st = conn.prepare(&sql)?;
    let mut out = vec![];
    for (gidx, r) in (1i64..).zip(st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, Option<f64>>(4)?,
            r.get::<_, Option<f64>>(5)?,
            r.get::<_, Option<f64>>(6)?,
            r.get::<_, Option<f64>>(7)?,
            r.get::<_, Option<f64>>(8)?,
            r.get::<_, Option<i64>>(9)?,
            r.get::<_, Option<String>>(10)?,
            r.get::<_, i64>(11)?,
            r.get::<_, i64>(12)?,
        ))
    })?) {
        let t = r?;
        out.push(Point {
            gidx,
            run_key: t.0,
            lane: t.1,
            epoch: t.2,
            phase: t.3,
            train_loss: t.4,
            val_loss: t.5,
            overfit_gap: t.6,
            val_pairwise: t.7,
            val_cond_adv: t.8,
            wall_ms: t.9,
            metric_definition: t.10.unwrap_or_else(|| "unknown".into()),
            val_fallback: t.11,
            early_stopped: t.12,
        });
    }
    Ok(out)
}

pub(super) fn cmd_export(
    root: &Path,
    format: &str,
    out: Option<PathBuf>,
    policy_only: bool,
) -> Result<()> {
    let conn = open_db(root)?;
    let outdir = out.unwrap_or_else(|| root.join("db/export"));
    fs::create_dir_all(&outdir)?;
    let pts = load_points(&conn, policy_only)?;

    // run boundaries in global-index space
    let mut boundaries: Vec<(i64, String, String, String)> = vec![]; // gidx, run_key, recipe/sampler label, metric_definition
    let mut last_run = String::new();
    for p in &pts {
        if p.run_key != last_run {
            let label: (Option<String>, Option<String>, Option<String>, Option<i64>) = conn
                .query_row(
                "SELECT recipe, sampler, corpus_snapshot, anti_overfit FROM run WHERE run_key=?1",
                params![p.run_key],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )?;
            let short = p
                .run_key
                .rsplit('/')
                .nth(2)
                .unwrap_or(&p.run_key)
                .to_string();
            let ann = format!(
                "{}{}{}",
                label.0.map(|x| format!("recipe={x} ")).unwrap_or_default(),
                label.1.map(|x| format!("sampler={x} ")).unwrap_or_default(),
                if label.3 == Some(1) {
                    "anti_overfit "
                } else {
                    ""
                }
            );
            boundaries.push((
                p.gidx,
                short,
                ann.trim().to_string(),
                p.metric_definition.clone(),
            ));
            last_run = p.run_key.clone();
        }
    }

    if format == "json" || format == "both" {
        let mut runs_json = vec![];
        let mut st = conn.prepare(
            "SELECT run_key,lane,run_id,attempt,watermark,node,model_version,recipe,sampler,corpus_snapshot,
                    anti_overfit,warm_start_sha256,warm_start_tensors_loaded,warm_start_tensors_total,
                    result_checkpoint_sha256,selected_checkpoint_epoch,selected_checkpoint_metric_name,
                    selected_checkpoint_metric,rows_recorded,first_epoch,last_epoch,started_at_ms,ended_at_ms,
                    time_basis,scq_state,metric_definition,early_stopped,completed,in_policy_curve,lane_kind,
                    source_origin,notes
             FROM run ORDER BY started_at_ms",
        )?;
        for r in st.query_map([], |r| {
            let mut m = Map::new();
            for (idx, name) in [
                "run_key",
                "lane",
                "run_id",
                "attempt",
                "watermark",
                "node",
                "model_version",
                "recipe",
                "sampler",
                "corpus_snapshot",
                "anti_overfit",
                "warm_start_sha256",
                "warm_start_tensors_loaded",
                "warm_start_tensors_total",
                "result_checkpoint_sha256",
                "selected_checkpoint_epoch",
                "selected_checkpoint_metric_name",
                "selected_checkpoint_metric",
                "rows_recorded",
                "first_epoch",
                "last_epoch",
                "started_at_ms",
                "ended_at_ms",
                "time_basis",
                "scq_state",
                "metric_definition",
                "early_stopped",
                "completed",
                "in_policy_curve",
                "lane_kind",
                "source_origin",
                "notes",
            ]
            .iter()
            .enumerate()
            {
                let v = r.get_ref(idx)?;
                m.insert(
                    (*name).to_string(),
                    match v {
                        rusqlite::types::ValueRef::Null => Value::Null,
                        rusqlite::types::ValueRef::Integer(x) => Value::from(x),
                        rusqlite::types::ValueRef::Real(x) => Value::from(x),
                        rusqlite::types::ValueRef::Text(x) => {
                            Value::from(String::from_utf8_lossy(x).to_string())
                        }
                        rusqlite::types::ValueRef::Blob(_) => Value::Null,
                    },
                );
            }
            m.insert(
                "started_at".into(),
                Value::from(fmt_ms(m["started_at_ms"].as_i64())),
            );
            m.insert(
                "ended_at".into(),
                Value::from(fmt_ms(m["ended_at_ms"].as_i64())),
            );
            Ok(Value::Object(m))
        })? {
            runs_json.push(r?);
        }

        let series: Vec<Value> = pts
            .iter()
            .map(|p| {
                let mut m = Map::new();
                m.insert("gidx".into(), Value::from(p.gidx));
                m.insert("run_key".into(), Value::from(p.run_key.clone()));
                m.insert("lane".into(), Value::from(p.lane.clone()));
                m.insert("epoch".into(), Value::from(p.epoch));
                m.insert("phase".into(), Value::from(p.phase.clone()));
                m.insert(
                    "train_loss".into(),
                    p.train_loss.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "val_loss".into(),
                    p.val_loss.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "overfit_gap".into(),
                    p.overfit_gap.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "val_pairwise_win_accuracy".into(),
                    p.val_pairwise.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "val_conditional_advantage".into(),
                    p.val_cond_adv.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "wall_ms".into(),
                    p.wall_ms.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "metric_definition".into(),
                    Value::from(p.metric_definition.clone()),
                );
                m.insert("val_fallback".into(), Value::from(p.val_fallback));
                m.insert("early_stopped".into(), Value::from(p.early_stopped));
                Value::Object(m)
            })
            .collect();

        let mut anns = vec![];
        let mut ast = conn
            .prepare("SELECT id,at_ms,kind,label,detail,evidence FROM annotation ORDER BY at_ms")?;
        for r in ast.query_map([], |r| {
            let mut m = Map::new();
            m.insert("id".into(), Value::from(r.get::<_, String>(0)?));
            let at: i64 = r.get(1)?;
            m.insert("at_ms".into(), Value::from(at));
            m.insert("at".into(), Value::from(fmt_ms(Some(at))));
            m.insert("kind".into(), Value::from(r.get::<_, String>(2)?));
            m.insert("label".into(), Value::from(r.get::<_, String>(3)?));
            m.insert("detail".into(), Value::from(r.get::<_, String>(4)?));
            m.insert(
                "evidence".into(),
                Value::from(r.get::<_, Option<String>>(5)?.unwrap_or_default()),
            );
            Ok(Value::Object(m))
        })? {
            anns.push(r?);
        }

        let mut gaps = vec![];
        let mut gst =
            conn.prepare("SELECT id,lane,what,evidence,approx_window FROM gap ORDER BY id")?;
        for r in gst.query_map([], |r| {
            let mut m = Map::new();
            m.insert("id".into(), Value::from(r.get::<_, String>(0)?));
            m.insert("lane".into(), Value::from(r.get::<_, String>(1)?));
            m.insert("what".into(), Value::from(r.get::<_, String>(2)?));
            m.insert("evidence".into(), Value::from(r.get::<_, String>(3)?));
            m.insert(
                "approx_window".into(),
                Value::from(r.get::<_, Option<String>>(4)?.unwrap_or_default()),
            );
            Ok(Value::Object(m))
        })? {
            gaps.push(r?);
        }

        let bounds: Vec<Value> = boundaries
            .iter()
            .map(|(g, run, ann, md)| {
                let mut m = Map::new();
                m.insert("gidx".into(), Value::from(*g));
                m.insert("run".into(), Value::from(run.clone()));
                m.insert("annotation".into(), Value::from(ann.clone()));
                m.insert("metric_definition".into(), Value::from(md.clone()));
                Value::Object(m)
            })
            .collect();

        let doc = serde_json::json!({
            "schema": "JopeTrainingHistorySeries.v1",
            "generated_at_ms": now_ms(),
            "archive": root.join("db/history.sqlite3").display().to_string(),
            "policy_only": policy_only,
            "counts": {"runs": runs_json.len(), "points": series.len()},
            "runs": runs_json,
            "run_boundaries": bounds,
            "annotations": anns,
            "known_lost": gaps,
            "series": series,
        });
        let p = outdir.join("train-val-series.v1.json");
        fs::write(&p, serde_json::to_vec_pretty(&doc)?)?;
        println!("wrote {}", p.display());
    }

    if format == "pgfplots" || format == "both" {
        // main table
        let p = outdir.join("loss.dat");
        let mut w = fs::File::create(&p)?;
        writeln!(w, "# gidx epoch train_loss val_loss overfit_gap val_pairwise val_cond_adv run metric_definition")?;
        for pt in &pts {
            writeln!(
                w,
                "{} {} {} {} {} {} {} {} {}",
                pt.gidx,
                pt.epoch,
                pt.train_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.overfit_gap
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_pairwise
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_cond_adv
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.run_key.replace(' ', "_"),
                pt.metric_definition,
            )?;
        }
        println!("wrote {}", p.display());

        // per-run files so each generation can be drawn as its own line
        let seriesdir = outdir.join("series");
        fs::create_dir_all(&seriesdir)?;
        let mut cur: Option<(String, fs::File)> = None;
        for pt in &pts {
            let fname = pt.run_key.replace('/', "__") + ".dat";
            let need_new = cur.as_ref().map(|(n, _)| n != &fname).unwrap_or(true);
            if need_new {
                let mut fh = fs::File::create(seriesdir.join(&fname))?;
                writeln!(fh, "# gidx epoch train_loss val_loss")?;
                cur = Some((fname, fh));
            }
            let (_, fh) = cur.as_mut().unwrap();
            writeln!(
                fh,
                "{} {} {} {}",
                pt.gidx,
                pt.epoch,
                pt.train_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
            )?;
        }
        println!(
            "wrote {}/*.dat ({} runs)",
            seriesdir.display(),
            boundaries.len()
        );

        // boundary + annotation tables
        let bp = outdir.join("boundaries.dat");
        let mut bw = fs::File::create(&bp)?;
        writeln!(bw, "# gidx run annotation metric_definition")?;
        for (g, run, ann, md) in &boundaries {
            writeln!(
                bw,
                "{} {} {} {}",
                g,
                run,
                if ann.is_empty() { "-" } else { ann },
                md
            )?;
        }
        println!("wrote {}", bp.display());

        // the .tex
        let tex = build_tex(&pts, &boundaries);
        let tp = outdir.join("train-val-loss.tex");
        fs::write(&tp, tex)?;
        println!("wrote {}", tp.display());
    }
    Ok(())
}

pub(super) fn build_tex(pts: &[Point], boundaries: &[(i64, String, String, String)]) -> String {
    let maxg = pts.last().map(|p| p.gidx).unwrap_or(1);
    let mut marks = String::new();
    // 91 run boundaries would overplot the axis. Draw a rule at every boundary,
    // but only LABEL the ones that carry information: a lane change, or a
    // recipe/sampler/regularisation change.
    let lane_of = |gidx: i64| pts.iter().find(|p| p.gidx == gidx).map(|p| p.lane.clone());
    let mut prev_lane: Option<String> = None;
    for (g, run, ann, md) in boundaries {
        let style = if md == "pre_task_types_fix" {
            "dashed, red!55"
        } else {
            "dotted, black!30"
        };
        marks.push_str(&format!(
            "    \\draw[{style}] (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymax}});\n"
        ));
        let lane = lane_of(*g);
        let lane_changed = lane != prev_lane;
        prev_lane = lane.clone();
        if lane_changed || !ann.is_empty() {
            marks.push_str(&format!(
                "    \\node[rotate=90, anchor=south west, font=\\tiny, black!65] at (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymin}}) {{{}{}{}}};\n",
                lane.map(|l| format!("{}: ", tex_escape(&l))).unwrap_or_default(),
                tex_escape(run),
                if ann.is_empty() { String::new() } else { format!(" ({})", tex_escape(ann)) }
            ));
        }
    }
    // the metric-definition fix boundary: first post-fix point
    let fix_g = pts
        .iter()
        .find(|p| p.metric_definition == "post_task_types_fix")
        .map(|p| p.gidx);
    let fix_mark = match fix_g {
        Some(g) => format!(
            "    \\draw[very thick, blue!70] (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymax}});\n    \\node[anchor=south east, font=\\scriptsize, blue!70] at (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymax}}) {{validation-metric fix (task\\_types)}};\n"
        ),
        None => String::new(),
    };
    format!(
        r#"% Generated by jope_training_archive export --format pgfplots.
% Build:  pdflatex train-val-loss.tex   (needs loss.dat in the same directory)
\documentclass[tikz,border=6pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepgfplotslibrary{{groupplots}}
\begin{{document}}
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=22cm, height=10cm,
    xlabel={{cumulative epoch across all archived runs}},
    ylabel={{loss}},
    xmin=1, xmax={maxg},
    ymode=log,
    legend pos=north east,
    legend cell align=left,
    grid=both, grid style={{black!12}},
    title={{jope policy DL training --- train vs validation loss, complete archived history}},
  ]
    \addplot[blue, thick, mark=none, unbounded coords=jump]
      table[x index=0, y index=2, col sep=space, header=false] {{loss.dat}};
    \addlegendentry{{train\_loss}}
    \addplot[orange, thick, mark=none, unbounded coords=jump]
      table[x index=0, y index=3, col sep=space, header=false] {{loss.dat}};
    \addlegendentry{{val\_loss}}
{marks}{fix_mark}  \end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"#
    )
}

pub(super) fn tex_escape(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('_', "\\_")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('#', "\\#")
}
