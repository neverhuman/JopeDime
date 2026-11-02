use super::*;

pub(super) fn reserve_manifest_pairs_in_tx(
    tx: &Transaction<'_>,
    owner_column: &str,
    manifest: &ValidManifest,
) -> Result<Option<String>> {
    let owner = format!("queue-keeper:{}", &manifest.sha256[..24]);
    let manifest_path = manifest.path.to_string_lossy().into_owned();
    let candidates = manifest
        .value
        .get("candidates")
        .and_then(Value::as_array)
        .context("validated manifest lost its candidates")?;
    ensure!(
        candidates.len() == manifest.pairs.len(),
        "validated manifest pair/candidate count drift"
    );
    let genomes = candidates
        .iter()
        .map(|candidate| {
            let semantic = candidate
                .get("semantic_sha256")
                .and_then(Value::as_str)
                .context("validated candidate lost its semantic identity")?;
            let genome = candidate
                .get("canonical_genome")
                .and_then(Value::as_str)
                .context("validated candidate lost its canonical genome")?;
            Ok((semantic.to_owned(), genome.to_owned()))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(
        genomes.len() == manifest.pairs.len(),
        "validated manifest repeats a semantic identity"
    );

    let lookup = format!(
        "SELECT state,{owner_column},manifest_path,manifest_sha256,job_id,\
         receipt_sha256,canonical_genome FROM pairs WHERE task=? AND dataset_hash=? \
         AND semantic_sha256=? AND protocol_sha256=?"
    );
    let mut lookup_statement = tx.prepare_cached(&lookup)?;
    for pair in &manifest.pairs {
        let existing: Option<PairReservationRow> = lookup_statement
            .query_row(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, pair_owner, path, digest, job_id, receipt, canonical)) = existing else {
            continue;
        };
        let expected_genome = genomes
            .get(&pair.semantic_sha256)
            .context("validated pair has no canonical genome")?;
        if canonical
            .as_deref()
            .is_some_and(|value| value != expected_genome)
        {
            bail!("pair registry semantic hash collision");
        }
        let reusable = state == "available" && receipt.is_none() && job_id.is_none();
        let already_reserved = state == "reserved"
            && pair_owner == owner
            && path.as_deref() == Some(manifest_path.as_str())
            && digest.as_deref() == Some(manifest.sha256.as_str())
            && job_id.is_none()
            && receipt.is_none();
        if !reusable && !already_reserved {
            return Ok(None);
        }
    }

    let timestamp = now_ms() as i64;
    let insert = format!(
        "INSERT OR IGNORE INTO pairs(task,dataset_hash,semantic_sha256,protocol_sha256,\
         canonical_genome,state,{owner_column},manifest_path,manifest_sha256,job_id,\
         receipt_sha256,failure_class,created_at_ms,updated_at_ms) \
         VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,NULL,NULL,?,?)"
    );
    let update = format!(
        "UPDATE pairs SET canonical_genome=COALESCE(canonical_genome,?),state='reserved',\
         {owner_column}=?,manifest_path=?,manifest_sha256=?,job_id=NULL,\
         receipt_sha256=NULL,failure_class=NULL,updated_at_ms=? \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
         AND state='available' AND receipt_sha256 IS NULL AND job_id IS NULL"
    );
    let mut insert_statement = tx.prepare_cached(&insert)?;
    let mut update_statement = tx.prepare_cached(&update)?;
    let mut event_statement = tx.prepare_cached(
        "INSERT INTO events(kind,task,dataset_hash,semantic_sha256,protocol_sha256,\
         detail_json,created_at_ms) VALUES('reserved',?,?,?,?,?,?)",
    )?;
    for pair in &manifest.pairs {
        let genome = genomes
            .get(&pair.semantic_sha256)
            .context("validated pair has no canonical genome")?;
        let inserted = insert_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            genome,
            owner,
            manifest_path,
            manifest.sha256,
            timestamp,
            timestamp
        ])?;
        let changed = if inserted == 1 {
            1
        } else {
            update_statement.execute(params![
                genome,
                owner,
                manifest_path,
                manifest.sha256,
                timestamp,
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ])?
        };
        if changed == 1 {
            event_statement.execute(params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256,
                serde_json::to_string(&json!({
                    "owner": owner,
                    "manifest_path": manifest_path,
                    "manifest_sha256": manifest.sha256,
                }))?,
                timestamp
            ])?;
        }
    }
    Ok(Some(owner))
}

pub(super) fn empty_report(
    submit: bool,
    targets: QueueTargets,
    nodes: Vec<NodeSlots>,
    reason: &str,
) -> CycleReport {
    CycleReport {
        schema: "JopeSelfplayQueueKeeper.v3",
        generated_at_ms: now_ms(),
        mode: if submit { "submit" } else { "audit" },
        admission_allowed: false,
        degraded: false,
        error: None,
        fail_closed_reason: Some(reason.to_owned()),
        active_checkpoint_sha256: None,
        checkpoint_epoch: None,
        newest_micro_wave: None,
        admission_waves: Vec::new(),
        protected_reserve_waves: Vec::new(),
        complete_current_checkpoint_waves: 0,
        partial_current_checkpoint_waves: 0,
        complete_reserve_waves: 0,
        reserve_manifests: 0,
        admission_manifests: 0,
        unclaimed_current_checkpoint_manifests: 0,
        queued_target: targets.target,
        queued_hard_cap: targets.hard_cap,
        p95_60m_jobs_per_minute: targets.p95_jobs_per_minute,
        p95_60m_candidate_evaluations_per_minute: 0.0,
        candidate_throughput_estimator:
            "p95_job_completions_times_current_candidate_evaluations_per_useful_manifest",
        queued: 0,
        starting: 0,
        queued_starting: 0,
        running: 0,
        dispatching: 0,
        held: 0,
        released_held: 0,
        queue_maximum_age_seconds: 0.0,
        stale_or_unproven_active: 0,
        launch_ready: 0,
        validated_ready: 0,
        useful_work: 0,
        queued_candidate_evaluations: 0,
        starting_candidate_evaluations: 0,
        dispatching_candidate_evaluations: 0,
        validated_ready_candidate_evaluations: 0,
        candidate_evaluation_work: 0,
        launch_ready_deferred_by_horizon: 0,
        manifests_unclaimed: 0,
        manifests_validated: 0,
        manifest_validation_failures: 0,
        preflight_accepted: 0,
        preflight_permanent_rejections: 0,
        preflight_bounded_screenings: 0,
        candidate_evaluations_preflighted: 0,
        candidate_evaluations_avoided: 0,
        predicted_cpu_seconds_preflighted: 0,
        predicted_cpu_seconds_avoided: 0,
        pair_reservation_conflicts: 0,
        prepared: 0,
        infrastructure_retries_prepared: 0,
        adopted_after_restart: 0,
        submitted: 0,
        submission_failures: 0,
        effective_cluster_slots: nodes.iter().map(|row| row.slot_target).sum(),
        nodes,
        state_counts: BTreeMap::new(),
    }
}

pub(super) fn failure_report(submit: bool, error: &str) -> CycleReport {
    let mut report = empty_report(
        submit,
        QueueTargets {
            target: 0,
            hard_cap: 0,
            p95_jobs_per_minute: 0.0,
        },
        Vec::new(),
        "queue cycle failed closed",
    );
    report.degraded = true;
    report.error = Some(error.to_owned());
    report
}

pub(super) fn initialize_backlog(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        r#"
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS claims(
          manifest_sha256 TEXT PRIMARY KEY,
          manifest TEXT NOT NULL UNIQUE,
          job_id TEXT NOT NULL UNIQUE,
          spec TEXT NOT NULL,
          state TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          reservation_owner TEXT,
          updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS terminal_jobs(
          job_id TEXT PRIMARY KEY,
          observed_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS burn_samples(
          observed_at_ms INTEGER PRIMARY KEY,
          jobs_per_minute REAL NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS rejected_manifests(
          manifest TEXT PRIMARY KEY,
          manifest_sha256 TEXT,
          reason TEXT NOT NULL,
          updated_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS task_preflight_receipts(
          manifest TEXT PRIMARY KEY,
          manifest_sha256 TEXT NOT NULL,
          outcome_sha256 TEXT NOT NULL UNIQUE,
          disposition TEXT NOT NULL,
          task TEXT NOT NULL,
          candidate_evaluations INTEGER NOT NULL,
          predicted_cpu_seconds INTEGER NOT NULL,
          candidate_evaluations_avoided INTEGER NOT NULL,
          predicted_cpu_seconds_avoided INTEGER NOT NULL,
          outcome_json TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS keeper_attempts(
          id INTEGER PRIMARY KEY,
          task TEXT NOT NULL,
          dataset_hash TEXT NOT NULL,
          semantic_sha256 TEXT NOT NULL,
          protocol_sha256 TEXT NOT NULL,
          attempt_number INTEGER NOT NULL,
          state TEXT NOT NULL,
          owner TEXT NOT NULL,
          manifest_path TEXT NOT NULL,
          manifest_sha256 TEXT NOT NULL,
          job_id TEXT NOT NULL,
          failure_class TEXT,
          created_at_ms INTEGER NOT NULL,
          finalized_at_ms INTEGER,
          UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number)
        );
        CREATE INDEX IF NOT EXISTS keeper_attempt_manifest_state
          ON keeper_attempts(manifest_sha256,state);
        "#,
    )?;
    // This index covers millions of attempt rows. Rebuilding it on every
    // daemon restart turns a five-second keeper recovery into minutes of disk
    // saturation. Migrate only when the declared predicate actually changes.
    const ACTIVE_INDEX_SQL: &str = "CREATE UNIQUE INDEX keeper_one_active_attempt \
        ON keeper_attempts(task,dataset_hash,semantic_sha256,protocol_sha256) \
        WHERE state IN ('checked','submitting','submitted','deferred_horizon')";
    let existing: Option<String> = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='keeper_one_active_attempt'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let normalized = |sql: &str| {
        sql.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase()
    };
    if existing.as_deref().map(normalized) != Some(normalized(ACTIVE_INDEX_SQL)) {
        connection.execute_batch(
            "DROP INDEX IF EXISTS keeper_one_active_attempt;
             CREATE UNIQUE INDEX keeper_one_active_attempt
               ON keeper_attempts(task,dataset_hash,semantic_sha256,protocol_sha256)
               WHERE state IN ('checked','submitting','submitted','deferred_horizon');",
        )?;
    }
    Ok(())
}

pub(super) fn inspect_pair_schema(connection: &Connection) -> Result<PairSchema> {
    let columns = table_columns(connection, "pairs")?;
    if columns.is_empty() {
        bail!("pair registry lacks the pairs table");
    }
    let owner_column = if columns.contains("reservation_owner") {
        "reservation_owner"
    } else if columns.contains("owner") {
        "owner"
    } else {
        bail!("pair registry lacks an owner column");
    };
    let attempt_columns = table_columns(connection, "attempts")?;
    let attempts = if attempt_columns.is_empty() {
        None
    } else {
        let number_column = if attempt_columns.contains("attempt_no") {
            "attempt_no"
        } else if attempt_columns.contains("attempt_number") {
            "attempt_number"
        } else {
            bail!("attempt ledger lacks an attempt number");
        };
        let attempt_owner = if attempt_columns.contains("reservation_owner") {
            "reservation_owner"
        } else if attempt_columns.contains("owner") {
            "owner"
        } else {
            bail!("attempt ledger lacks an owner column");
        };
        Some(AttemptSchema {
            number_column,
            owner_column: attempt_owner,
        })
    };
    Ok(PairSchema {
        owner_column,
        has_infrastructure_retries: columns.contains("infrastructure_retries"),
        attempts,
    })
}

pub(super) fn table_columns(connection: &Connection, table: &str) -> Result<BTreeSet<String>> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub(super) fn load_claims(connection: &Connection) -> Result<HashMap<String, Claim>> {
    let mut statement = connection.prepare(
        "SELECT manifest_sha256,manifest,job_id,spec,state,attempts,reservation_owner \
         FROM claims",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Claim {
            manifest_sha256: row.get(0)?,
            manifest: PathBuf::from(row.get::<_, String>(1)?),
            job_id: row.get(2)?,
            spec: PathBuf::from(row.get::<_, String>(3)?),
            state: row.get(4)?,
            attempts: row.get(5)?,
            owner: row.get(6)?,
        })
    })?;
    let mut result = HashMap::new();
    for row in rows {
        let claim = row?;
        if result.insert(claim.job_id.clone(), claim).is_some() {
            bail!("backlog contains duplicate job identities");
        }
    }
    Ok(result)
}

pub(super) fn load_rejected_manifests(connection: &Connection) -> Result<BTreeSet<PathBuf>> {
    let mut statement = connection.prepare("SELECT manifest FROM rejected_manifests")?;
    let rows = statement.query_map([], |row| Ok(PathBuf::from(row.get::<_, String>(0)?)))?;
    let mut rejected = BTreeSet::new();
    for row in rows {
        rejected.insert(row?);
    }
    Ok(rejected)
}

pub(super) fn reject_manifest(
    connection: &Connection,
    path: &Path,
    digest: Option<&str>,
    reason: &str,
) -> Result<()> {
    connection.execute(
        "INSERT INTO rejected_manifests(manifest,manifest_sha256,reason,updated_at_ms) \
         VALUES(?,?,?,?) ON CONFLICT(manifest) DO UPDATE SET \
         manifest_sha256=excluded.manifest_sha256,reason=excluded.reason,\
         updated_at_ms=excluded.updated_at_ms",
        params![
            path.to_string_lossy(),
            digest,
            reason.chars().take(500).collect::<String>(),
            now_ms() as i64
        ],
    )?;
    Ok(())
}

pub(super) fn record_preflight_receipt(
    connection: &Connection,
    path: &Path,
    outcome: &TaskPreflightOutcomeV1,
) -> Result<()> {
    outcome.validate()?;
    let outcome_sha256 = outcome.digest()?;
    let manifest_sha256 = sha256_file(path)?;
    let disposition = serde_json::to_value(outcome.disposition)?
        .as_str()
        .context("task-preflight disposition is not a string")?
        .to_owned();
    let outcome_json = serde_json::to_string(outcome)?;
    let inserted = connection.execute(
        "INSERT OR IGNORE INTO task_preflight_receipts(
           manifest,manifest_sha256,outcome_sha256,disposition,task,
           candidate_evaluations,predicted_cpu_seconds,
           candidate_evaluations_avoided,predicted_cpu_seconds_avoided,
           outcome_json,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        params![
            path.to_string_lossy(),
            manifest_sha256,
            outcome_sha256,
            disposition,
            outcome.task.as_str(),
            i64::try_from(outcome.requested_work.candidate_evaluation_count)?,
            i64::try_from(outcome.requested_work.predicted_cpu_seconds)?,
            i64::try_from(outcome.estimated_compute_avoided.candidate_evaluation_count)?,
            i64::try_from(outcome.estimated_compute_avoided.predicted_cpu_seconds)?,
            outcome_json,
            now_ms() as i64,
        ],
    )?;
    if inserted == 0 {
        let existing: (String, String) = connection.query_row(
            "SELECT manifest_sha256,outcome_sha256 FROM task_preflight_receipts WHERE manifest=?",
            [path.to_string_lossy().as_ref()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        ensure!(
            existing == (manifest_sha256, outcome_sha256),
            "append-only task-preflight receipt identity drift"
        );
    }
    Ok(())
}
