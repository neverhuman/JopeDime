use super::*;

/// Annotations that must come from the DATA, not from a constant: model-version
/// transitions and the first appearance of each validation metric. Re-derived on
/// every ingest so a future schema change annotates itself.
pub(super) fn derive_annotations(tx: &rusqlite::Transaction) -> Result<()> {
    // model_version transitions along the wall-clock run order
    let mut st = tx.prepare(
        "SELECT model_version, MIN(started_at_ms), MAX(COALESCE(ended_at_ms, started_at_ms))
         FROM run WHERE model_version IS NOT NULL AND lane_kind='epoch'
         GROUP BY model_version ORDER BY MIN(started_at_ms)",
    )?;
    let versions: Vec<(String, i64, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(st);
    for w in versions.windows(2) {
        let (from, _, from_end) = &w[0];
        let (to, to_start, _) = &w[1];
        let at = (*from_end + *to_start) / 2;
        tx.execute(
            "INSERT INTO annotation(id,at_ms,kind,label,detail,evidence) VALUES(?1,?2,?3,?4,?5,?6)
             ON CONFLICT(id) DO UPDATE SET at_ms=excluded.at_ms, label=excluded.label,
               detail=excluded.detail, evidence=excluded.evidence",
            params![
                format!("model-version-{from}-to-{to}"),
                at,
                "model_version_change",
                format!("{from} -> {to}"),
                "Different architecture generation: loss magnitudes and head sets are not comparable across this line.",
                format!(
                    "last {from} run ends {}, first {to} run starts {} (run.model_version, derived at ingest)",
                    fmt_ms(Some(*from_end)),
                    fmt_ms(Some(*to_start))
                )
            ],
        )?;
    }

    // first appearance of each validation metric that is not universal
    let mut ks = tx.prepare(
        "SELECT key, first_seen_ms, first_run_key, runs_present FROM field_catalog
         WHERE key LIKE 'val\\_%' ESCAPE '\\' ORDER BY first_seen_ms",
    )?;
    let total_runs: i64 = tx.query_row("SELECT COUNT(*) FROM run", [], |r| r.get(0))?;
    let keys: Vec<(String, i64, String, i64)> = ks
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(ks);
    for (key, first_ms, first_run, runs_present) in keys {
        if runs_present >= total_runs {
            continue; // present everywhere: nothing changed
        }
        tx.execute(
            "INSERT INTO annotation(id,at_ms,kind,label,detail,evidence) VALUES(?1,?2,?3,?4,?5,?6)
             ON CONFLICT(id) DO UPDATE SET at_ms=excluded.at_ms, label=excluded.label,
               detail=excluded.detail, evidence=excluded.evidence",
            params![
                format!("metric-added-{key}"),
                first_ms,
                "metric_added",
                format!("{key} first emitted"),
                format!("Present in {runs_present} of {total_runs} archived runs. Earlier epochs have no value for it; a series that plots it must start here, not at the origin of the curve."),
                format!("field_catalog: first row carrying {key} is in {first_run} at {}", fmt_ms(Some(first_ms)))
            ],
        )?;
    }
    // pin the sampler / regularisation annotations to the first run that used them
    for (id, sql) in [
        (
            "equal-quota-sampler",
            "SELECT MIN(started_at_ms) FROM run WHERE sampler IS NOT NULL",
        ),
        (
            "anti-overfit-experiment",
            "SELECT MIN(started_at_ms) FROM run WHERE anti_overfit=1",
        ),
    ] {
        if let Ok(Some(at)) = tx.query_row(sql, [], |r| r.get::<_, Option<i64>>(0)) {
            tx.execute(
                "UPDATE annotation SET at_ms=?2 WHERE id=?1",
                params![id, at],
            )?;
        }
    }
    Ok(())
}

/// Project the capture layer's truncation ledger into `flag` + `gap`.
///
/// `training/history/truncations.v1.jsonl` is written by jope_training_history
/// whenever a trainer died mid-write and left a trailing INCOMPLETE JSONL row.
/// Under the operator policy that fragment is not data — the epoch never
/// finished being written — so every complete row is archived, the fragment is
/// kept verbatim next to the segment, and the run directory becomes prunable.
/// That decision has to be visible to anyone querying this database, or the
/// curve would silently read as if the run simply ended one epoch earlier.
pub(super) fn ingest_truncations(tx: &rusqlite::Transaction, root: &Path) -> Result<usize> {
    let Ok(text) = fs::read_to_string(root.join("truncations.v1.jsonl")) else {
        return Ok(0);
    };
    let mut count = 0usize;
    for line in text.lines() {
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(run_key) = record["run_key"].as_str() else {
            continue;
        };
        let residual = record["residual_bytes"].as_u64().unwrap_or(0);
        let after = record["last_archived_epoch"]
            .as_i64()
            .map(|epoch| epoch.to_string())
            .unwrap_or_else(|| "?".into());
        let detail = format!(
            "writer stopped mid-row: {residual}B trailing INCOMPLETE line after epoch {after}. \
             Every complete row is archived; the fragment is kept verbatim as {} and the \
             source directory was released for pruning. Node {}, source idle {}s at record time.",
            record["evidence_file"].as_str().unwrap_or("?"),
            record["node"].as_str().unwrap_or("?"),
            record["source_idle_secs"].as_u64().unwrap_or(0),
        );
        tx.execute(
            "INSERT OR REPLACE INTO flag(run_key,epoch,phase,code,severity,detail)
             VALUES(?1,-1,'*','truncated_tail','warn',?2)",
            params![run_key, detail],
        )?;
        // Also a gap row: this is history that is proven never to have existed
        // in complete form, and the ledger is the evidence.
        tx.execute(
            "INSERT INTO gap(id,lane,what,evidence,approx_window) VALUES(?1,'scq',?2,?3,?4)
             ON CONFLICT(id) DO UPDATE SET what=excluded.what, evidence=excluded.evidence,
               approx_window=excluded.approx_window",
            params![
                format!("gap-trunc-{}", record["ts_ms"].as_u64().unwrap_or(0)),
                format!("{run_key}: final epoch after {after} was never written completely"),
                format!(
                    "truncations.v1.jsonl fragment sha256 {}",
                    record["fragment_sha256"].as_str().unwrap_or("?")
                ),
                record["ts_ms"].as_u64().map(|ms| ms.to_string()),
            ],
        )?;
        count += 1;
    }
    Ok(count)
}

pub(super) fn recompute_flags(tx: &rusqlite::Transaction) -> Result<()> {
    let stmts = [
        // epoch-level
        ("INSERT OR REPLACE INTO flag SELECT e.run_key,e.epoch,e.phase,'val_loss_missing','error',
            'no val_loss on this epoch' FROM epoch e JOIN run r USING(run_key)
            WHERE e.val_loss IS NULL AND r.lane_kind='epoch'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,epoch,phase,'val_fallback','warn',
            'trainer fell back to a substitute validation split' FROM epoch WHERE val_fallback=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,epoch,phase,'early_stopped','info',
            'macro_validation_early_stopped set on this epoch' FROM epoch WHERE macro_validation_early_stopped=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,epoch,phase,'val_loss_nonfinite','error',
            'val_loss is not finite' FROM epoch WHERE val_loss IS NOT NULL AND (val_loss!=val_loss OR val_loss>1e300)", ()),
        ("INSERT OR REPLACE INTO flag SELECT e.run_key,e.epoch,e.phase,'train_loss_missing','error',
            'no train_loss on this epoch' FROM epoch e JOIN run r USING(run_key)
            WHERE e.train_loss IS NULL AND r.lane_kind='epoch'", ()),
        // run-level
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','metric_definition_pre_fix','warn',
            'validation metrics computed with binary rows scored by the regression heads; val_conditional_advantage / val_binary_* not comparable to post-fix runs'
            FROM run WHERE metric_definition='pre_task_types_fix'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','no_split_heads','info',
            'run predates the val_binary_*/val_regression_* split; only pooled pairwise accuracy exists'
            FROM run WHERE metric_definition='no_split_heads'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','run_incomplete','warn',
            'no train-result.json: run was cancelled, preempted or is still live; no checkpoint was selected'
            FROM run WHERE completed=0 AND lane='scq'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','early_stopped_run','info',
            'run early-stopped on macro validation' FROM run WHERE early_stopped=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','anti_overfit_experiment','info',
            'PrimeAntiOverfitExperiment.v1 active: token dropout, embedding jitter, weight decay, dataset row cap'
            FROM run WHERE anti_overfit=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','not_on_policy_curve','info',
            'step-granularity or different-objective lane; archived but excluded from the train/val epoch curve'
            FROM run WHERE in_policy_curve=0", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','wall_clock_derived','info',
            'start/end wall time derived from file mtime minus elapsed_sec, not from an scq event'
            FROM run WHERE time_basis='file_mtime-elapsed_sec'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','epoch_axis_not_from_one','warn',
            'first archived epoch is not 1: earlier epochs of this stream were never captured'
            FROM run WHERE lane_kind='epoch' AND first_epoch IS NOT NULL AND first_epoch>1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','source_pruned','warn',
            'originating run directory / file no longer exists; this archive is the only copy'
            FROM run WHERE source_live=0", ()),
    ];
    for (sql, _) in stmts {
        tx.execute(sql, [])?;
    }

    // step lanes: one run-level note instead of one flag per step row
    tx.execute(
        "INSERT OR REPLACE INTO flag
         SELECT r.run_key,-1,'*','step_lane_val_sparse','info',
                'step-granularity lane: ' || SUM(CASE WHEN e.val_loss IS NULL THEN 1 ELSE 0 END)
                || ' of ' || COUNT(*) || ' rows carry no val_loss (validation runs on a slower cadence)'
         FROM run r JOIN epoch e USING(run_key)
         WHERE r.lane_kind='step' GROUP BY r.run_key",
        [],
    )?;

    // epoch-count gaps inside a run
    tx.execute(
        "INSERT OR REPLACE INTO flag
         SELECT r.run_key,-1,'*','epoch_gap','warn',
                'epochs recorded (' || r.rows_recorded || ') < span ' || r.first_epoch || '..' || r.last_epoch
         FROM run r
         WHERE r.lane_kind='epoch' AND r.first_epoch IS NOT NULL
           AND r.rows_recorded < (r.last_epoch - r.first_epoch + 1)",
        [],
    )?;
    Ok(())
}

// ------------------------------------------------------------------ reports

pub(super) fn open_db(root: &Path) -> Result<Connection> {
    let p = root.join("db/history.sqlite3");
    if !p.is_file() {
        bail!("no archive at {} — run `ingest` first", p.display());
    }
    Ok(Connection::open(p)?)
}
