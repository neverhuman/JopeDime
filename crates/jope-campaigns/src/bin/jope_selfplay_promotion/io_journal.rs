use super::*;

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub(super) fn sha256_bytes(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

pub(super) fn sha256_text(text: &str) -> String {
    sha256_bytes(text.as_bytes())
}

pub(super) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(super) fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(super) fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// `test_exact_common.atomic_json`: pretty CPython dump + newline, temp file
/// named `.{name}.tmp-{pid}` in place, then rename.
pub(super) fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("atomic_json target has no parent"))?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("atomic_json target has no name"))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let text = format!("{}\n", py_dumps(value, PyStyle::Pretty)?);
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub(super) fn copy_tree(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &destination)?;
        } else if kind.is_file() {
            fs::copy(entry.path(), destination)?;
        } else {
            bail!("unsupported filesystem entry in fetched outputs");
        }
    }
    Ok(())
}

pub(super) fn rglob(root: &Path, name: &str, results: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut children: Vec<_> = entries.flatten().collect();
    children.sort_by_key(|entry| entry.file_name());
    for entry in children {
        let path = entry.path();
        if path.is_dir() {
            rglob(&path, name, results);
        } else if entry.file_name().to_string_lossy() == name
            || (name.starts_with("*.") && entry.file_name().to_string_lossy().ends_with(&name[1..]))
        {
            results.push(path);
        }
    }
}

pub(super) fn get_str<'v>(value: &'v Value, key: &str) -> Result<&'v str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string field {key}"))
}

pub(super) fn get_i64(value: &Value, key: &str) -> Result<i64> {
    let field = value
        .get(key)
        .ok_or_else(|| anyhow!("missing integer field {key}"))?;
    match field {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|f| f as i64))
            .ok_or_else(|| anyhow!("field {key} is not an integer")),
        Value::String(text) => text
            .parse()
            .map_err(|_| anyhow!("field {key} is not an integer")),
        _ => bail!("field {key} is not an integer"),
    }
}

pub(super) fn get_f64(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("missing float field {key}"))
}

// ---------------------------------------------------------------------------
// Pending / failure semantics
// ---------------------------------------------------------------------------

/// The action is healthy but its external result is not complete yet.
#[derive(Debug)]
pub(super) struct PendingAction(String);

impl std::fmt::Display for PendingAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for PendingAction {}

pub(super) fn pending(detail: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(PendingAction(detail.into()))
}

pub(super) fn as_pending(error: &anyhow::Error) -> Option<&PendingAction> {
    error.downcast_ref::<PendingAction>()
}

// ---------------------------------------------------------------------------
// Action journal
// ---------------------------------------------------------------------------

pub(super) fn action_key(
    action: &str,
    corpus_watermark: i64,
    accepted_ancestor: &str,
    recipe: &str,
    seed: i64,
    catalog_sha256: &str,
    evaluator_sha256: &str,
) -> Result<String> {
    let material = json!({
        "action": action,
        "corpus_watermark": corpus_watermark,
        "accepted_ancestor": accepted_ancestor,
        "recipe": recipe,
        "seed": seed,
        "catalog_sha256": catalog_sha256,
        "evaluator_sha256": evaluator_sha256,
    });
    Ok(sha256_text(&py_dumps(&material, PyStyle::Compact)?))
}

pub(super) struct ActionJournal {
    pub(super) db: Connection,
    events: PathBuf,
}

impl ActionJournal {
    pub(super) fn open(root: &Path) -> Result<Self> {
        let ops = root.join("ops");
        fs::create_dir_all(&ops)?;
        let db = Connection::open(ops.join("controller.sqlite"))?;
        db.busy_timeout(std::time::Duration::from_secs(60))?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS actions(
              idempotency_key TEXT PRIMARY KEY,
              action TEXT NOT NULL,
              status TEXT NOT NULL,
              request_json TEXT NOT NULL,
              result_json TEXT,
              attempts INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            ) WITHOUT ROWID",
        )?;
        Ok(Self {
            db,
            events: ops.join("controller-events.jsonl"),
        })
    }

    pub(super) fn event(
        &self,
        key: &str,
        action: &str,
        status: &str,
        payload: &Value,
    ) -> Result<()> {
        let line = py_dumps(
            &json!({
                "schema": "JopeSelfplayControllerEvent.v1",
                "idempotency_key": key,
                "action": action,
                "status": status,
                "at_ms": now_ms(),
                "payload": payload,
            }),
            PyStyle::Compact,
        )?;
        let mut handle = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&self.events)?;
        handle.write_all(line.as_bytes())?;
        handle.write_all(b"\n")?;
        handle.sync_all()?;
        Ok(())
    }

    pub(super) fn perform(
        &self,
        key: &str,
        action: &str,
        request: &Value,
        callback: impl FnOnce(&Value) -> Result<Value>,
    ) -> Result<Value> {
        let existing: Option<(String, Option<String>, i64)> = self
            .db
            .query_row(
                "SELECT status, result_json, attempts FROM actions WHERE idempotency_key=?",
                [key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((status, result_json, _)) = &existing {
            if status == "succeeded" {
                let stored = result_json
                    .as_ref()
                    .ok_or_else(|| anyhow!("succeeded action has no result"))?;
                return Ok(serde_json::from_str(stored)?);
            }
        }
        let attempts = existing.map(|(_, _, count)| count + 1).unwrap_or(1);
        self.db.execute(
            "INSERT INTO actions VALUES(?,?,?,?,?,?,?)
             ON CONFLICT(idempotency_key) DO UPDATE SET
               status=excluded.status,request_json=excluded.request_json,
               attempts=excluded.attempts,updated_at_ms=excluded.updated_at_ms",
            params![
                key,
                action,
                "running",
                py_dumps(request, PyStyle::Default)?,
                Option::<String>::None,
                attempts,
                now_ms(),
            ],
        )?;
        self.event(key, action, "started", &json!({"attempt": attempts}))?;
        match callback(request) {
            Ok(result) => {
                self.db.execute(
                    "UPDATE actions SET status='succeeded',result_json=?,updated_at_ms=?
                     WHERE idempotency_key=?",
                    params![py_dumps(&result, PyStyle::Default)?, now_ms(), key],
                )?;
                self.event(key, action, "succeeded", &result)?;
                Ok(result)
            }
            Err(error) => {
                if let Some(detail) = as_pending(&error) {
                    self.db.execute(
                        "UPDATE actions SET status='pending',updated_at_ms=? WHERE idempotency_key=?",
                        params![now_ms(), key],
                    )?;
                    self.event(
                        key,
                        action,
                        "pending",
                        &json!({"detail": detail.to_string()}),
                    )?;
                } else {
                    self.db.execute(
                        "UPDATE actions SET status='failed',updated_at_ms=? WHERE idempotency_key=?",
                        params![now_ms(), key],
                    )?;
                    self.event(key, action, "failed", &json!({"error": error.to_string()}))?;
                }
                Err(error)
            }
        }
    }
}
