use super::*;

pub(super) fn load_terminal_seen(connection: &Connection) -> Result<HashSet<String>> {
    let mut statement = connection.prepare("SELECT job_id FROM terminal_jobs")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<HashSet<_>>>()?)
}

pub(super) fn record_burn_rate(
    connection: &mut Connection,
    jobs: &[SchedulerJob],
    terminal_seen: &mut HashSet<String>,
) -> Result<Vec<f64>> {
    let timestamp = now_ms() as i64;
    let terminal = jobs
        .iter()
        .filter(|job| job.project == PROJECT && TERMINAL_STATES.contains(&job.state.as_str()))
        .filter(|job| !terminal_seen.contains(&job.id))
        .map(|job| job.id.clone())
        .collect::<Vec<_>>();
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let mut new = 0_i64;
    for job_id in &terminal {
        new += tx.execute(
            "INSERT OR IGNORE INTO terminal_jobs(job_id,observed_at_ms) VALUES(?,?)",
            params![job_id, timestamp],
        )? as i64;
    }
    let previous: Option<i64> =
        tx.query_row("SELECT MAX(observed_at_ms) FROM burn_samples", [], |row| {
            row.get(0)
        })?;
    let elapsed_minutes = previous
        .map(|prior| ((timestamp - prior).max(5_000) as f64) / 60_000.0)
        .unwrap_or(1.0);
    let rate = if previous.is_some() {
        new as f64 / elapsed_minutes
    } else {
        0.0
    };
    tx.execute(
        "INSERT OR REPLACE INTO burn_samples(observed_at_ms,jobs_per_minute) VALUES(?,?)",
        params![timestamp, rate],
    )?;
    tx.execute(
        "DELETE FROM burn_samples WHERE observed_at_ms<?",
        [timestamp - 60 * 60_000],
    )?;
    let values = {
        let mut statement = tx.prepare(
            "SELECT jobs_per_minute FROM burn_samples \
             WHERE observed_at_ms>=? ORDER BY observed_at_ms",
        )?;
        let rows = statement.query_map([timestamp - 60 * 60_000], |row| row.get::<_, f64>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    tx.commit()?;
    terminal_seen.extend(terminal);
    Ok(values)
}

pub(super) fn append_keeper_attempts(
    tx: &Transaction<'_>,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
    state: &str,
    initial_only: i64,
) -> Result<()> {
    // New self-play identities overwhelmingly begin at attempt one. Trying
    // that insert first avoids two indexed reads for every pair in a wave.
    // `OR IGNORE` is safe here because both uniqueness constraints remain the
    // authorities; only the uncommon retry/idempotency path needs the reads.
    let mut first_attempt_statement = tx.prepare_cached(
        "INSERT OR IGNORE INTO keeper_attempts(task,dataset_hash,semantic_sha256,\
         protocol_sha256,attempt_number,state,owner,manifest_path,manifest_sha256,\
         job_id,created_at_ms) VALUES(?,?,?,?,1,?,?,?,?,?,?)",
    )?;
    let mut active_statement = tx.prepare_cached(
        "SELECT manifest_sha256,job_id FROM keeper_attempts WHERE task=? AND dataset_hash=? \
         AND semantic_sha256=? AND protocol_sha256=? \
         AND state IN ('checked','submitting','submitted','deferred_horizon')",
    )?;
    let mut attempt_statement = tx.prepare_cached(include_str!(
        "../../sql/jope_selfplay_queue_attempt_number.sql"
    ))?;
    let mut insert_statement = tx.prepare_cached(
        "INSERT INTO keeper_attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
         attempt_number,state,owner,manifest_path,manifest_sha256,job_id,created_at_ms) \
         VALUES(?,?,?,?,?,?,?,?,?,?,?)",
    )?;
    for pair in &manifest.pairs {
        let inserted = first_attempt_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            state,
            owner,
            manifest.path.to_string_lossy(),
            manifest.sha256,
            job_id,
            now_ms() as i64
        ])?;
        if inserted == 1 {
            continue;
        }
        let active: Option<(String, String)> = active_statement
            .query_row(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((active_manifest, active_job)) = active {
            if initial_only != 0 {
                if active_manifest != manifest.sha256 || active_job != job_id {
                    bail!("protocol pair has a conflicting active queue attempt");
                }
                continue;
            }
            bail!("protocol pair already has an active queue attempt");
        }
        let attempt: i64 = attempt_statement.query_row(
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ],
            |row| row.get(0),
        )?;
        insert_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            attempt,
            state,
            owner,
            manifest.path.to_string_lossy(),
            manifest.sha256,
            job_id,
            now_ms() as i64
        ])?;
    }
    Ok(())
}

pub(super) fn update_native_attempts_submitted(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
) -> Result<()> {
    let Some(attempts) = &schema.attempts else {
        return Ok(());
    };
    let (select, update) = match (attempts.number_column, attempts.owner_column) {
        ("attempt_no", "owner") => (
            "SELECT attempt_no,state,owner,manifest_sha256,job_id FROM attempts \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND state IN ('reserved','submitted') ORDER BY attempt_no DESC",
            "UPDATE attempts SET state='submitted',job_id=? \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND attempt_no=? AND owner=? AND manifest_sha256=? AND state='reserved'",
        ),
        ("attempt_no", "reservation_owner") => (
            "SELECT attempt_no,state,reservation_owner,manifest_sha256,job_id FROM attempts \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND state IN ('reserved','submitted') ORDER BY attempt_no DESC",
            "UPDATE attempts SET state='submitted',job_id=? \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND attempt_no=? AND reservation_owner=? AND manifest_sha256=? \
               AND state='reserved'",
        ),
        ("attempt_number", "owner") => (
            "SELECT attempt_number,state,owner,manifest_sha256,job_id FROM attempts \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND state IN ('reserved','submitted') ORDER BY attempt_number DESC",
            "UPDATE attempts SET state='submitted',job_id=? \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND attempt_number=? AND owner=? AND manifest_sha256=? AND state='reserved'",
        ),
        ("attempt_number", "reservation_owner") => (
            "SELECT attempt_number,state,reservation_owner,manifest_sha256,job_id \
             FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
               AND protocol_sha256=? AND state IN ('reserved','submitted') \
             ORDER BY attempt_number DESC",
            "UPDATE attempts SET state='submitted',job_id=? \
             WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
               AND attempt_number=? AND reservation_owner=? AND manifest_sha256=? \
               AND state='reserved'",
        ),
        _ => bail!("attempt ledger schema is not supported"),
    };
    let mut select_statement = tx.prepare_cached(select)?;
    let mut update_statement = tx.prepare_cached(update)?;
    for pair in &manifest.pairs {
        let rows = select_statement
            .query_map(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if rows.len() != 1 {
            bail!(
                "native attempt conflict for {}:{}:{}: expected one active attempt, found {}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                rows.len()
            );
        }
        let (attempt_number, state, attempt_owner, attempt_manifest, attempt_job) = &rows[0];
        if attempt_owner != owner || attempt_manifest.as_deref() != Some(manifest.sha256.as_str()) {
            bail!(
                "native attempt reservation conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if state == "submitted" {
            if attempt_job.as_deref() == Some(job_id) {
                continue;
            }
            bail!(
                "native attempt job conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        let changed = update_statement.execute(params![
            job_id,
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            attempt_number,
            owner,
            manifest.sha256,
        ])?;
        if changed != 1 {
            bail!(
                "native attempt changed during exact submission transition for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
    }
    Ok(())
}
