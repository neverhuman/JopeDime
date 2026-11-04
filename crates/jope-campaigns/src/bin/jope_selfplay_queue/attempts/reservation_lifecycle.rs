use super::*;

pub(crate) fn ensure_native_reserved_attempts(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
) -> Result<()> {
    let Some(attempts) = &schema.attempts else {
        return Ok(());
    };
    let (active_query, next_query, insert) = match (attempts.number_column, attempts.owner_column) {
        ("attempt_no", "owner") => (
            "SELECT attempt_no,state,owner,manifest_sha256,job_id FROM attempts \
                 WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
                   AND state IN ('reserved','submitted') ORDER BY attempt_no DESC",
            include_str!("../../../../sql/native_attempts_next_no.sql"),
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                 attempt_no,state,owner,manifest_path,manifest_sha256,job_id,created_at_ms) \
                 VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,?)",
        ),
        ("attempt_no", "reservation_owner") => (
            "SELECT attempt_no,state,reservation_owner,manifest_sha256,job_id \
                 FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
                   AND protocol_sha256=? AND state IN ('reserved','submitted') \
                 ORDER BY attempt_no DESC",
            include_str!("../../../../sql/native_attempts_next_no.sql"),
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                 attempt_no,state,reservation_owner,manifest_path,manifest_sha256,job_id,\
                 created_at_ms) VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,?)",
        ),
        ("attempt_number", "owner") => (
            "SELECT attempt_number,state,owner,manifest_sha256,job_id FROM attempts \
                 WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
                   AND state IN ('reserved','submitted') ORDER BY attempt_number DESC",
            include_str!("../../../../sql/native_attempts_next_number.sql"),
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                 attempt_number,state,owner,manifest_path,manifest_sha256,job_id,created_at_ms) \
                 VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,?)",
        ),
        ("attempt_number", "reservation_owner") => (
            "SELECT attempt_number,state,reservation_owner,manifest_sha256,job_id \
                 FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
                   AND protocol_sha256=? AND state IN ('reserved','submitted') \
                 ORDER BY attempt_number DESC",
            include_str!("../../../../sql/native_attempts_next_number.sql"),
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                 attempt_number,state,reservation_owner,manifest_path,manifest_sha256,job_id,\
                 created_at_ms) VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,?)",
        ),
        _ => bail!("attempt ledger schema is not supported"),
    };
    let mut active_statement = tx.prepare_cached(active_query)?;
    let mut next_statement = tx.prepare_cached(next_query)?;
    let mut insert_statement = tx.prepare_cached(insert)?;
    for pair in &manifest.pairs {
        let rows = active_statement
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
        if rows.len() > 1 {
            bail!(
                "native attempt conflict for {}:{}:{}: found multiple active attempts",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if let Some((_, state, attempt_owner, attempt_manifest, attempt_job)) = rows.first() {
            ensure!(
                matches!(state.as_str(), "reserved" | "submitted")
                    && attempt_owner == owner
                    && attempt_manifest.as_deref() == Some(manifest.sha256.as_str())
                    && (state == "reserved" || attempt_job.as_deref() == Some(job_id)),
                "native attempt reservation conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
            continue;
        }
        // Migration repair for reservations created before the native attempt
        // ledger was wired into queue admission. The pair row already pins the
        // exact owner and manifest; this transaction supplies only the missing
        // initial attempt before submission.
        let attempt_number: i64 = next_statement.query_row(
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
            attempt_number,
            owner,
            manifest.path.to_string_lossy(),
            manifest.sha256,
            now_ms() as i64
        ])?;
    }
    Ok(())
}

pub(crate) fn update_pairs_submitted(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
) -> Result<()> {
    let (select, update) = match schema.owner_column {
        "owner" => (
            include_str!("../../../../sql/pairs_submitted_select_owner.sql"),
            include_str!("../../../../sql/pairs_submitted_update_owner.sql"),
        ),
        "reservation_owner" => (
            include_str!("../../../../sql/pairs_submitted_select_reservation_owner.sql"),
            include_str!("../../../../sql/pairs_submitted_update_reservation_owner.sql"),
        ),
        _ => bail!("pair registry schema is not supported"),
    };
    let mut select_statement = tx.prepare_cached(select)?;
    let mut update_statement = tx.prepare_cached(update)?;
    for pair in &manifest.pairs {
        let row: Option<(String, String, Option<String>, Option<String>)> = select_statement
            .query_row(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let Some((state, pair_owner, pair_manifest, pair_job)) = row else {
            bail!(
                "submission pair is absent for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        };
        if pair_owner != owner || pair_manifest.as_deref() != Some(manifest.sha256.as_str()) {
            bail!(
                "submission pair reservation conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if state == "submitted" {
            if pair_job.as_deref() == Some(job_id) {
                continue;
            }
            bail!(
                "submission pair job conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if !matches!(state.as_str(), "reserved" | "infrastructure_retry") {
            bail!(
                "submission pair is not launch-ready for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        let changed = update_statement.execute(params![
            job_id,
            now_ms() as i64,
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            owner,
            manifest.sha256,
        ])?;
        if changed != 1 {
            bail!(
                "submission pair changed during exact transition for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
    }
    Ok(())
}

pub(crate) fn finalize_native_attempts_for_retry(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    failure: &str,
) -> Result<()> {
    let Some(_) = &schema.attempts else {
        return Ok(());
    };
    let columns = table_columns(tx, "attempts")?;
    let finalized = if columns.contains("finalized_at_ms") {
        ",finalized_at_ms=?"
    } else {
        ""
    };
    let query = format!(
        "UPDATE attempts SET state='infrastructure_retry',failure_class=?{finalized} \
         WHERE manifest_sha256=? AND state='submitted'"
    );
    let changed = if finalized.is_empty() {
        tx.execute(&query, params![failure, manifest.sha256])?
    } else {
        tx.execute(&query, params![failure, now_ms() as i64, manifest.sha256])?
    };
    if changed != manifest.pairs.len() {
        let prior: i64 = tx.query_row(
            "SELECT COUNT(*) FROM attempts WHERE manifest_sha256=? \
             AND state='infrastructure_retry'",
            [manifest.sha256.as_str()],
            |row| row.get(0),
        )?;
        if changed != 0 || prior < manifest.pairs.len() as i64 {
            bail!("native attempt infrastructure retry transition count mismatch");
        }
    }
    Ok(())
}

pub(crate) fn append_native_retry_attempts(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    new_job_id: &str,
) -> Result<()> {
    let Some(attempts) = &schema.attempts else {
        return Ok(());
    };
    for pair in &manifest.pairs {
        let active_job: Option<String> = tx
            .query_row(
                "SELECT job_id FROM attempts WHERE task=? AND dataset_hash=? \
                 AND semantic_sha256=? AND protocol_sha256=? AND state='reserved'",
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(job_id) = active_job {
            if job_id != new_job_id {
                bail!("native retry attempt is bound to a different job");
            }
            continue;
        }
        let next_attempt_query = match attempts.number_column {
            "attempt_no" => include_str!("../../../../sql/native_attempts_next_no.sql"),
            "attempt_number" => include_str!("../../../../sql/native_attempts_next_number.sql"),
            _ => bail!("attempt ledger schema is not supported"),
        };
        let attempt_number: i64 = tx.query_row(
            next_attempt_query,
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ],
            |row| row.get(0),
        )?;
        let insert_query = match (attempts.number_column, attempts.owner_column) {
            ("attempt_no", "owner") => {
                include_str!("../../../../sql/native_attempt_no_owner_retry_insert.sql")
            }
            ("attempt_no", "reservation_owner") => {
                include_str!("../../../../sql/native_attempt_no_reservation_owner_retry_insert.sql")
            }
            ("attempt_number", "owner") => {
                include_str!("../../../../sql/native_attempt_number_owner_retry_insert.sql")
            }
            ("attempt_number", "reservation_owner") => {
                include_str!(
                    "../../../../sql/native_attempt_number_reservation_owner_retry_insert.sql"
                )
            }
            _ => bail!("attempt ledger schema is not supported"),
        };
        tx.execute(
            insert_query,
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256,
                attempt_number,
                owner,
                manifest.path.to_string_lossy(),
                manifest.sha256,
                new_job_id,
                now_ms() as i64
            ],
        )?;
    }
    Ok(())
}
