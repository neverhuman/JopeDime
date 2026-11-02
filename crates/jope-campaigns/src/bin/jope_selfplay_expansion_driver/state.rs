use super::*;

pub(super) fn activations(root: &Path) -> Result<Vec<Value>> {
    let path = root.join("catalog/v2/activations.jsonl");
    let mut rows = Vec::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines() {
            if !line.trim().is_empty() {
                rows.push(serde_json::from_str(line)?);
            }
        }
    }
    Ok(rows)
}

pub(super) fn next_tranche_id(existing: &[Value], prefix: char) -> String {
    let max = existing
        .iter()
        .filter_map(|row| row.get("tranche_id").and_then(Value::as_str))
        .filter_map(|id| id.strip_prefix(&format!("expanded-{prefix}")))
        .filter_map(|n| n.parse::<u32>().ok())
        .max()
        .unwrap_or(0);
    format!("expanded-{prefix}{:04}", max + 1)
}

/// Newest activation FOR ONE TASK. The valve must pace each task against its
/// own last tranche: keying it off the globally-last activation coupled the two
/// lanes, so a fresh regression tranche (coverage ~0) shut the binary lane down
/// for hours even when the last binary tranche was fully absorbed — the model
/// has two task branches and both must keep widening.
pub(super) fn latest_tranche_for(existing: &[Value], prefix: char) -> Option<(String, PathBuf)> {
    let want = format!("expanded-{prefix}");
    existing
        .iter()
        .rev()
        .find(|row| {
            row.get("tranche_id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with(&want))
        })
        .map(|row| {
            (
                row.get("tranche_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                PathBuf::from(row.get("catalog").and_then(Value::as_str).unwrap_or("")),
            )
        })
}

// ---------------------------------------------------------------------------
// Dataset-state index
// ---------------------------------------------------------------------------

pub(super) fn open_state_db(root: &Path) -> Result<Connection> {
    let conn = Connection::open(root.join("ops/dataset-state.sqlite"))?;
    conn.busy_timeout(Duration::from_secs(60))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS datasets(
           task TEXT NOT NULL, dataset_hash TEXT NOT NULL,
           has_train_test_meta INTEGER NOT NULL DEFAULT 0,
           sidecars_complete INTEGER NOT NULL DEFAULT 0,
           foundation_npy INTEGER NOT NULL DEFAULT 0,
           cache_materialized INTEGER NOT NULL DEFAULT 0,
           tranche_id TEXT, catalog_active INTEGER NOT NULL DEFAULT 0,
           sealed INTEGER NOT NULL DEFAULT 0,
           encode_job_state TEXT,
           obs_count INTEGER NOT NULL DEFAULT 0,
           best_score REAL, last_observed_ms INTEGER,
           updated_at_ms INTEGER NOT NULL,
           PRIMARY KEY(task, dataset_hash)) WITHOUT ROWID;
         CREATE INDEX IF NOT EXISTS ds_stage ON datasets(task, sidecars_complete, catalog_active);
         CREATE INDEX IF NOT EXISTS ds_active_obs ON datasets(task, catalog_active, obs_count);
         CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;
    // Added after the first deployment: existing prod DBs predate the column,
    // and ALTER TABLE ADD COLUMN has no IF NOT EXISTS, so a duplicate-column
    // error here is the expected steady state and must not fail the cycle.
    let _ = conn.execute(
        "ALTER TABLE datasets ADD COLUMN encode_attempts INTEGER NOT NULL DEFAULT 0",
        [],
    );
    Ok(conn)
}

pub(super) fn refresh_filesystem_state(conn: &Connection, root: &Path) -> Result<(u64, u64)> {
    let ptr = read_json(&root.join("catalog/current.json"))?;
    let cat = read_json(Path::new(
        ptr.get("catalog").and_then(Value::as_str).unwrap_or(""),
    ))?;
    let mut active: HashSet<(String, String)> = HashSet::new();
    let mut binding = Vec::new();
    for e in cat
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        active.insert((
            e.get("task")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            e.get("dataset_hash")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        ));
    }
    binding.clear();
    let sealed =
        read_json(&root.join("catalog/private/sealed-expanded-v1.json")).unwrap_or(json!({}));
    let mut sealed_ids: HashSet<String> = HashSet::new();
    for e in sealed
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        if let Some(id) = e
            .get("dataset_id")
            .or_else(|| e.get("dataset_hash"))
            .and_then(Value::as_str)
        {
            sealed_ids.insert(id.to_owned());
        }
    }
    let now = now_ms();
    let mut scanned = 0_u64;
    let tx = conn.unchecked_transaction()?;
    for task in TASKS {
        let reg = Path::new(registry_str()).join(corpus_dir_name(task));
        let cache_root = Path::new(encodings_str()).join(format!("{task}-expanded"));
        // The legacy corpus roots are mandatory; a new task's root may not
        // exist on every host yet and simply contributes no rows.
        if !reg.is_dir() && !matches!(task, "regression" | "binary") {
            continue;
        }
        for entry in std::fs::read_dir(&reg)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let bare = name.strip_prefix("synth_").unwrap_or(&name);
            if name.starts_with('.') || bare.len() != 16 {
                continue;
            }
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            scanned += 1;
            let has_data = ["train.csv", "test.csv", "meta.json"]
                .iter()
                .all(|f| dir.join(f).is_file());
            let sidecars = SIDECARS
                .iter()
                .all(|s| dir.join(s).metadata().map(|m| m.len() > 0).unwrap_or(false));
            let npy = Path::new(foundation_str())
                .join(format!("{name}.npy"))
                .is_file();
            let cached = cache_root
                .join(&name)
                .join("cache-manifest.v1.json")
                .is_file();
            let is_active = active.contains(&(task.to_owned(), name.clone()));
            let is_sealed = sealed_ids.contains(&name);
            tx.execute(
                "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
                   foundation_npy,cache_materialized,catalog_active,sealed,updated_at_ms)
                 VALUES(?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(task,dataset_hash) DO UPDATE SET
                   has_train_test_meta=excluded.has_train_test_meta,
                   sidecars_complete=excluded.sidecars_complete,
                   foundation_npy=excluded.foundation_npy,
                   cache_materialized=excluded.cache_materialized,
                   catalog_active=excluded.catalog_active,
                   sealed=excluded.sealed,
                   updated_at_ms=excluded.updated_at_ms",
                rusqlite::params![
                    task,
                    name,
                    has_data as i64,
                    sidecars as i64,
                    npy as i64,
                    cached as i64,
                    is_active as i64,
                    is_sealed as i64,
                    now as i64
                ],
            )?;
        }
    }
    tx.commit()?;
    Ok((scanned, now))
}

pub(super) fn refresh_observation_state(conn: &Connection, root: &Path) -> Result<u64> {
    let last: i64 = conn
        .query_row(
            "SELECT value FROM meta WHERE key='obs_watermark_ms'",
            [],
            |r| r.get::<_, String>(0),
        )
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    corpus.busy_timeout(Duration::from_secs(60))?;
    // Count the WHOLE history of every dataset touched since the watermark, not
    // just the rows inside the window. Ingest re-evaluates with REPLACE, which
    // bumps created_at_ms, so a windowed count re-counts rows that were already
    // counted in an earlier cycle — accumulating those deltas inflated obs_count
    // to 5.94M against 3.09M real rows.
    let mut statement = corpus.prepare(
        "WITH changed(task,dataset_hash) AS (
           SELECT DISTINCT task,dataset_hash
             FROM observations INDEXED BY observations_created_v1
            WHERE created_at_ms > ?
         )
         SELECT observations.task, observations.dataset_hash, COUNT(*),
                MAX(observations.score), MAX(observations.created_at_ms)
           FROM changed
           JOIN observations
             ON observations.task=changed.task
            AND observations.dataset_hash=changed.dataset_hash
          GROUP BY observations.task, observations.dataset_hash",
    )?;
    let rows: Vec<(String, String, i64, Option<f64>, i64)> = statement
        .query_map([last], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?
        .collect::<std::result::Result<_, _>>()?;
    drop(statement);
    drop(corpus);
    let mut watermark = last;
    let count = rows.len() as u64;
    let tx = conn.unchecked_transaction()?;
    for (task, dataset, obs, best, newest) in rows {
        watermark = watermark.max(newest);
        tx.execute(
            "INSERT INTO datasets(task,dataset_hash,obs_count,best_score,last_observed_ms,updated_at_ms)
             VALUES(?,?,?,?,?,?)
             ON CONFLICT(task,dataset_hash) DO UPDATE SET
               obs_count=excluded.obs_count,
               best_score=MAX(COALESCE(datasets.best_score,-1e308),COALESCE(excluded.best_score,-1e308)),
               last_observed_ms=excluded.last_observed_ms,
               updated_at_ms=excluded.updated_at_ms",
            rusqlite::params![task, dataset, obs, best, newest, now_ms() as i64],
        )?;
    }
    tx.execute(
        "INSERT INTO meta VALUES('obs_watermark_ms',?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [watermark.to_string()],
    )?;
    tx.commit()?;
    Ok(count)
}

pub(super) fn stage_summary(conn: &Connection) -> Result<Value> {
    let mut summary = BTreeMap::new();
    for task in TASKS {
        let row = conn.query_row(
            "SELECT COUNT(*),
                    SUM(has_train_test_meta),
                    SUM(sidecars_complete),
                    SUM(CASE WHEN sidecars_complete=1 AND foundation_npy=1 THEN 1 ELSE 0 END),
                    SUM(cache_materialized),
                    SUM(catalog_active),
                    SUM(CASE WHEN obs_count>0 THEN 1 ELSE 0 END),
                    SUM(sealed)
             FROM datasets WHERE task=?",
            [task],
            |r| {
                Ok(json!({
                    "registry": r.get::<_, i64>(0)?,
                    "with_data": r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    "sidecars_complete": r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    "fully_encoded": r.get::<_, Option<i64>>(3)?.unwrap_or(0),
                    "cache_materialized": r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                    "catalog_active": r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                    "observed": r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                    "sealed": r.get::<_, Option<i64>>(7)?.unwrap_or(0),
                }))
            },
        )?;
        summary.insert(task.to_owned(), row);
    }
    Ok(json!(summary))
}

pub(super) fn emit_encoding_exclusion_receipts(conn: &Connection, root: &Path) -> Result<usize> {
    let directory = root.join("ops/encoding-exclusions-v1");
    std::fs::create_dir_all(&directory)?;
    let mut statement = conn.prepare(
        "SELECT task,dataset_hash,sealed,has_train_test_meta FROM datasets
          WHERE sealed=1 OR has_train_test_meta=0 ORDER BY task,dataset_hash",
    )?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut created = 0;
    for (task, dataset_hash, sealed, has_data) in rows {
        // Keep this maintenance pass bounded so a large legacy catalog cannot
        // delay encoding admission.  Subsequent two-minute cycles resume
        // monotonically because receipts are immutable and skipped above.
        if created >= 512 {
            break;
        }
        let path = directory.join(format!("{task}-{dataset_hash}.json"));
        if path.is_file() {
            continue;
        }
        let reason = if sealed != 0 {
            "sealed_audit_dataset"
        } else if has_data == 0 {
            "malformed_or_missing_train_test_meta"
        } else {
            "ineligible"
        };
        let receipt = json!({
            "schema": "EncodingEligibilityReceipt.v1",
            "task": task,
            "dataset_hash": dataset_hash,
            "eligible": false,
            "disposition": "excluded",
            "reason": reason,
            "recorded_at_ms": now_ms(),
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        use std::io::Write;
        // Serialize the complete receipt before writing to this newly created file.
        // An I/O failure can still leave partial contents.
        let mut line = serde_json::to_vec(&receipt)?;
        line.push(b'\n');
        file.write_all(&line)?;
        file.sync_data()?;
        created += 1;
    }
    Ok(created)
}

pub(super) fn tranche_coverage(conn: &Connection, tranche_catalog: &Path) -> Result<f64> {
    let tranche = read_json(tranche_catalog)?;
    let binding = Vec::new();
    let entries = tranche
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding);
    if entries.is_empty() {
        return Ok(1.0);
    }
    let mut observed = 0_usize;
    let mut statement =
        conn.prepare("SELECT obs_count FROM datasets WHERE task=? AND dataset_hash=?")?;
    for e in entries {
        let task = e
            .get("task")
            .and_then(Value::as_str)
            .or_else(|| e.pointer("/dataset/task").and_then(Value::as_str))
            .unwrap_or("");
        let hash = e
            .get("dataset_hash")
            .and_then(Value::as_str)
            .or_else(|| e.pointer("/dataset/dataset_hash").and_then(Value::as_str))
            .unwrap_or("");
        let count: i64 = statement.query_row([task, hash], |r| r.get(0)).unwrap_or(0);
        if count > 0 {
            observed += 1;
        }
    }
    Ok(observed as f64 / entries.len() as f64)
}

// ---------------------------------------------------------------------------
// Work steps
// ---------------------------------------------------------------------------
