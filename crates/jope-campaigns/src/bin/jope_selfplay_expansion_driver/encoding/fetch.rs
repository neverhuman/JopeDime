use super::*;

/// Exhaustion must be a BACKOFF, never a death sentence. Datasets get parked
/// after MAX_ENCODE_ATTEMPTS, but placement failures can burn the whole budget
/// without a dataset ever reaching a node that has the encode environment
/// (scq picks the node and the spec's hostname dependency does not constrain
/// placement — b2c17 attempt 1 landed on xbabe1, which has no prime config).
/// Once nothing submittable is left, un-park everything and start over: broken
/// datasets are only ever retried after the healthy backlog is exhausted, so
/// this cannot starve real work, and the binary lane can never permanently die.
pub(crate) fn unpark_exhausted_if_starved(conn: &Connection, log: &mut Vec<Value>) -> Result<()> {
    let submittable: i64 = conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE task='binary' AND has_train_test_meta=1
           AND sidecars_complete=0 AND catalog_active=0 AND encode_job_state IS NULL",
        [],
        |r| r.get(0),
    )?;
    if submittable > 0 {
        return Ok(());
    }
    let unparked = conn.execute(
        "UPDATE datasets SET encode_job_state=NULL, encode_attempts=0
          WHERE task='binary' AND encode_job_state='exhausted'",
        [],
    )?;
    if unparked > 0 {
        log.push(json!({"step": "encode_exhausted_unparked", "datasets": unparked}));
    }
    Ok(())
}

pub(crate) fn fetch_root(task: &str) -> PathBuf {
    if task == "binary" {
        PathBuf::from(fetch_binary_str())
    } else {
        paths::flywheel_encodings_root().join("_fetch/regression-expansion")
    }
}

pub(crate) fn hash_directory(path: &Path) -> Result<String> {
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(directory)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut identity = Vec::new();
    for file in files {
        identity.extend_from_slice(
            file.strip_prefix(path)
                .unwrap_or(&file)
                .to_string_lossy()
                .as_bytes(),
        );
        identity.push(0);
        identity.extend_from_slice(sha256_file(&file)?.as_bytes());
        identity.push(b'\n');
    }
    Ok(sha256_bytes(&identity))
}

pub(crate) fn encoding_chunk_listing(job_id: &str) -> Result<Option<Value>> {
    let directory = paths::flywheel_encodings_root().join("_batch_manifests");
    for entry in std::fs::read_dir(directory)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(value) = read_json(&path) else {
            continue;
        };
        if value.get("schema").and_then(Value::as_str) == Some("EncodingChunk.v2")
            && value.get("job_identity").and_then(Value::as_str) == Some(job_id)
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}
