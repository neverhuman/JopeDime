use super::*;

pub(crate) fn sqlite_rows(database: &Path, sql: &str) -> Result<Vec<Value>> {
    let connection = open_sqlite(database)?;
    let mut statement = connection
        .prepare(sql)
        .with_context(|| format!("prepare SQLite audit for {}", database.display()))?;
    let names = statement
        .column_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let mut rows = statement.query([])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        let mut object = Map::new();
        for (index, name) in names.iter().enumerate() {
            let value = match row.get_ref(index)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(value) => json!(value),
                ValueRef::Real(value) => json!(value),
                ValueRef::Text(value) => Value::String(String::from_utf8_lossy(value).into_owned()),
                ValueRef::Blob(value) => Value::String(hex::encode(value)),
            };
            object.insert(name.clone(), value);
        }
        result.push(Value::Object(object));
    }
    Ok(result)
}

pub(crate) fn sqlite_exec(database: &Path, sql: &str) -> Result<()> {
    open_sqlite(database)?
        .execute_batch(sql)
        .with_context(|| format!("update {}", database.display()))
}

pub(crate) fn open_sqlite(database: &Path) -> Result<Connection> {
    let connection =
        Connection::open(database).with_context(|| format!("open {}", database.display()))?;
    connection.busy_timeout(Duration::from_secs(60))?;
    Ok(connection)
}

pub(crate) fn sqlite_column_exists(
    connection: &Connection,
    table: &str,
    column: &str,
) -> Result<bool> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(columns.iter().any(|candidate| candidate == column))
}

pub(crate) fn import_legacy_state(path: &Path) -> Result<DurableState> {
    if !path.is_file() {
        return Ok(DurableState::new());
    }
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open legacy freshness state {}", path.display()))?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let has_controls: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='controls')",
        [],
        |row| row.get(0),
    )?;
    if !has_controls {
        return Ok(DurableState::new());
    }
    let control = |key: &str| -> Result<String> {
        Ok(connection
            .query_row("SELECT value FROM controls WHERE key=?1", [key], |row| {
                row.get(0)
            })
            .optional()?
            .unwrap_or_default())
    };
    let mut state = DurableState {
        schema: "JopeFreshnessState.v1".to_owned(),
        active_epoch: control("active_epoch")?,
        active_checkpoint: control("active_checkpoint")?,
        admission_paused: control("admission_paused")? == "1",
        ..DurableState::default()
    };
    let mut checkpoints = connection.prepare(
        "SELECT epoch,activated_at_ms,superseded_at_ms FROM checkpoints ORDER BY activated_at_ms",
    )?;
    for row in checkpoints.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, Option<i64>>(2)?,
        ))
    })? {
        let (epoch, activated, superseded) = row?;
        state
            .checkpoint_activated_at_ms
            .insert(epoch.clone(), activated);
        if let Some(superseded) = superseded {
            state.checkpoint_superseded_at_ms.insert(epoch, superseded);
        }
    }
    let mut previews = connection
        .prepare("SELECT receipt_sha256,payload_json FROM previews ORDER BY created_at_ms")?;
    for row in previews.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })? {
        let (digest, payload) = row?;
        let preview: Preview =
            serde_json::from_str(&payload).context("parse legacy freshness preview")?;
        state.previews.insert(digest, preview);
    }
    let mut mutations = connection.prepare(
        "SELECT idempotency_key,tool,preview_sha256,result_json,created_at_ms
         FROM mutations ORDER BY created_at_ms",
    )?;
    for row in mutations.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })? {
        let (key, tool, preview_sha256, result, created_at_ms) = row?;
        state.mutations.insert(
            key,
            Mutation {
                tool,
                preview_sha256,
                result: serde_json::from_str(&result).context("parse legacy freshness mutation")?,
                created_at_ms,
            },
        );
    }
    Ok(state)
}
