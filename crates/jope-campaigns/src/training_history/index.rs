use super::*;

/// One metrics file found under a run directory.
#[derive(Clone, Debug)]
pub struct FoundMetrics {
    pub key: SegmentKey,
    pub path: PathBuf,
    pub len: u64,
}

/// Split `<runs_root>/<run_id>/attempts/<attempt>/outputs/.../<watermark>/train_metrics.jsonl`
/// into a segment key. Returns None when the path is not shaped like a trainer
/// output (so junk can never create archive directories).
pub fn key_from_path(run_id: &str, path: &Path) -> Option<SegmentKey> {
    let watermark = path.parent()?.file_name()?.to_str()?.to_string();
    let mut attempt = None;
    let components: Vec<String> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
        .collect();
    for (index, part) in components.iter().enumerate() {
        if part == "attempts" {
            attempt = components.get(index + 1).cloned();
        }
    }
    let key = SegmentKey {
        run_id: run_id.to_string(),
        attempt: attempt.unwrap_or_else(|| "0".to_string()),
        watermark,
    };
    key.is_sane().then_some(key)
}

fn walk_metrics(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > WALK_DEPTH || out.len() > 512 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten().take(2048) {
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => walk_metrics(&path, depth + 1, out),
            Ok(kind)
                if kind.is_file()
                    && path.file_name().and_then(|n| n.to_str()) == Some(METRICS_FILE) =>
            {
                out.push(path);
            }
            _ => {}
        }
    }
}

/// Every metrics file under one run directory. Bounded walk, `attempts/` only.
pub fn scan_run_dir(runs_root: &Path, run_id: &str) -> Vec<FoundMetrics> {
    let mut paths = Vec::new();
    walk_metrics(&runs_root.join(run_id).join("attempts"), 0, &mut paths);
    let mut found = Vec::new();
    for path in paths {
        let Some(key) = key_from_path(run_id, &path) else {
            continue;
        };
        let len = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        found.push(FoundMetrics { key, path, len });
    }
    found.sort_by(|a, b| a.key.cmp(&b.key));
    found
}

/// Trainer run directory names under `runs_root`. One `getdents` sweep plus a
/// substring test — no `stat` per entry, because this root holds ~70k dirs.
pub fn discover_trainer_runs(runs_root: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(runs_root)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains(TRAIN_MARKER) && sane_component(&name) {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Load every segment currently in the archive, keyed for lookup.
fn read_dir_if_exists(path: &Path) -> Result<Option<fs::ReadDir>> {
    match fs::read_dir(path) {
        Ok(entries) => Ok(Some(entries)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("read directory {}", path.display())),
    }
}

pub fn load_segments(archive: &Path) -> Result<BTreeMap<SegmentKey, SegmentState>> {
    let mut out = BTreeMap::new();
    let runs = archive.join("runs");
    let Some(run_entries) = read_dir_if_exists(&runs)? else {
        return Ok(out);
    };
    for run in run_entries {
        let run = run.with_context(|| format!("read run entry under {}", runs.display()))?;
        let run_id = run.file_name().to_string_lossy().into_owned();
        let run_path = run.path();
        let Some(attempts) = read_dir_if_exists(&run_path)? else {
            continue;
        };
        for attempt in attempts {
            let attempt = attempt
                .with_context(|| format!("read attempt entry under {}", run_path.display()))?;
            let attempt_name = attempt.file_name().to_string_lossy().into_owned();
            let attempt_path = attempt.path();
            let Some(watermarks) = read_dir_if_exists(&attempt_path)? else {
                continue;
            };
            for watermark in watermarks {
                let watermark = watermark.with_context(|| {
                    format!("read watermark entry under {}", attempt_path.display())
                })?;
                let key = SegmentKey {
                    run_id: run_id.clone(),
                    attempt: attempt_name.clone(),
                    watermark: watermark.file_name().to_string_lossy().into_owned(),
                };
                let watermark_path = watermark.path();
                if let Some(state) = SegmentState::load(&watermark_path)? {
                    out.insert(key, state);
                }
            }
        }
    }
    Ok(out)
}

/// How many bytes of a given segment the archive already holds. Read straight
/// off the archive file so a lost or stale state file cannot over-report.
pub fn archived_bytes(archive: &Path, key: &SegmentKey) -> u64 {
    fs::metadata(SegmentState::metrics_path(&segment_dir(archive, key)))
        .map(|m| m.len())
        .unwrap_or(0)
}

/// `name` -> (`sha256`, `source`) for every `[[input]]` block in a job spec.
fn spec_inputs(text: &str) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    for block in text.split("[[input]]").skip(1) {
        // Stop at the next table header so keys cannot leak across blocks.
        let body = block.split("\n[").next().unwrap_or(block);
        let field = |key: &str| -> Option<String> {
            body.lines()
                .find_map(|line| {
                    let (left, right) = line.split_once('=')?;
                    (left.trim() == key).then(|| right.trim().trim_matches('"').to_string())
                })
                .filter(|value| !value.is_empty())
        };
        if let Some(name) = field("name") {
            out.insert(
                name,
                (
                    field("sha256").unwrap_or_default(),
                    field("source").unwrap_or_default(),
                ),
            );
        }
    }
    out
}

/// Turn a `--warm` argument into a checkpoint sha256. The argument is normally
/// the placeholder `{input:checkpoint}`; a literal path is also accepted.
fn resolve_checkpoint(value: &str, inputs: &BTreeMap<String, (String, String)>) -> Option<String> {
    if let Some(name) = value
        .strip_prefix("{input:")
        .and_then(|v| v.strip_suffix('}'))
    {
        let (sha, source) = inputs.get(name)?;
        if !sha.is_empty() {
            return Some(sha.clone());
        }
        return Path::new(source)
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.trim_end_matches(".model.pt").to_string());
    }
    Path::new(value)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".model.pt").to_string())
        .filter(|n| !n.is_empty())
}

/// Lineage annotations pulled out of `specs/<run_id>.toml`: which recipe the run
/// used, which checkpoint it warm-started from, and which corpus snapshot (hence
/// which sampler) fed it. Without these a single curve across runs is a lie.
pub fn spec_annotations(root: &Path, run_id: &str) -> Map<String, Value> {
    let mut out = Map::new();
    let Ok(text) = fs::read_to_string(root.join("specs").join(format!("{run_id}.toml"))) else {
        return out;
    };
    // Command args reference inputs by placeholder (`--warm {input:checkpoint}`),
    // so the real checkpoint identity lives in the matching [[input]] block.
    let inputs = spec_inputs(&text);
    let tokens: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || matches!(c, '"' | ',' | '[' | ']'))
        .filter(|t| !t.is_empty())
        .collect();
    for (index, token) in tokens.iter().enumerate() {
        let next = tokens.get(index + 1).copied();
        match *token {
            "--recipe" => {
                if let Some(value) = next {
                    out.insert("recipe".into(), json!(value));
                }
            }
            "--seed" => {
                if let Some(value) = next {
                    out.insert("seed".into(), json!(value));
                }
            }
            "--warm" => {
                if let Some(value) = next {
                    if let Some(sha) = resolve_checkpoint(value, &inputs) {
                        out.insert("warm_start".into(), json!(sha));
                    }
                }
            }
            _ => {}
        }
    }
    // The snapshot directory name encodes mode and sampler, e.g.
    // snapshot-001805437-full_consolidation-eq32v32c128.
    if let Some(start) = text.find("/training/snapshot-") {
        let rest = &text[start + "/training/".len()..];
        if let Some(end) = rest.find('/') {
            out.insert("corpus_snapshot".into(), json!(&rest[..end]));
        }
    }
    out
}

/// Final-selection facts a run publishes when it finishes.
fn result_annotations(seg_dir: &Path) -> Map<String, Value> {
    let mut out = Map::new();
    let Ok(text) = fs::read_to_string(seg_dir.join(RESULT_FILE)) else {
        return out;
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return out;
    };
    for key in [
        "checkpoint_sha256",
        "selected_checkpoint_epoch",
        "recipe",
        "epochs",
    ] {
        if let Some(found) = value.get(key) {
            if !found.is_null() {
                out.insert(format!("result_{key}"), found.clone());
            }
        }
    }
    out
}

/// Rebuild `index.v1.json` from the segments on disk. This is the file every
/// consumer (TUI, plotters, the alarm) reads; nothing downstream should ever
/// touch a run directory again.
pub fn rebuild_index(root: &Path, archive: &Path) -> Result<Value> {
    let segments = load_segments(archive)?;
    let mut rows = Vec::new();
    let mut total_rows = 0u64;
    let mut total_bytes = 0u64;
    let mut newest_advance = 0u64;
    let mut truncated_segments = 0u64;
    let mut truncated_bytes = 0u64;
    for (key, state) in &segments {
        let seg_dir = segment_dir(archive, key);
        let bytes = archived_bytes(archive, key);
        total_rows += state.archived_rows;
        total_bytes += bytes;
        newest_advance = newest_advance.max(state.last_advance_ms);
        if state.truncation_count > 0 {
            truncated_segments += 1;
            truncated_bytes += state.truncated_tail_bytes;
        }
        let mut row = json!({
            "run_id": key.run_id,
            "attempt": key.attempt,
            "watermark": key.watermark,
            "path": key.relative().join(METRICS_FILE).to_string_lossy(),
            "rows": state.archived_rows,
            "bytes": bytes,
            "first_epoch": state.first_epoch,
            "last_epoch": state.last_epoch,
            "first_seen_ms": state.first_seen_ms,
            "last_advance_ms": state.last_advance_ms,
            "last_check_ms": state.last_check_ms,
            "source_present": state.source_present,
            "source_path": state.source_path,
            "node": state.node,
            "rewrite_count": state.rewrite_count,
            "truncated_tail_bytes": state.truncated_tail_bytes,
            "truncation_count": state.truncation_count,
            "last_truncation_ms": state.last_truncation_ms,
            "head": state.head,
            "tail": state.tail,
        });
        let annotations = row.as_object_mut().expect("object");
        for (name, value) in spec_annotations(root, &key.run_id) {
            annotations.insert(name, value);
        }
        for (name, value) in result_annotations(&seg_dir) {
            annotations.insert(name, value);
        }
        rows.push(row);
    }
    // Oldest first: the index reads as the training history in order.
    rows.sort_by_key(|row| {
        (
            row["first_seen_ms"].as_u64().unwrap_or(0),
            row["run_id"].as_str().unwrap_or("").to_string(),
        )
    });
    let run_count = segments
        .keys()
        .map(|k| k.run_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let index = json!({
        "schema": INDEX_SCHEMA,
        "generated_at_ms": now_ms(),
        "archive_root": archive.to_string_lossy(),
        "run_count": run_count,
        "segment_count": rows.len(),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "newest_advance_ms": newest_advance,
        // Holes, stated at the top of the record rather than buried: a
        // consumer that never opens a segment still sees them.
        "truncated_segments": truncated_segments,
        "truncated_tail_bytes": truncated_bytes,
        "nodes": load_nodes(archive).values().collect::<Vec<_>>(),
        "segments": rows,
    });
    write_atomic(
        &archive.join(INDEX_FILE),
        &serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(index)
}

/// Advisory whole-archive lock. The capture timer takes it non-blocking and
/// skips; the collector takes it with retries and refuses to prune if it cannot
/// get it, because pruning without the ability to archive is the failure mode
/// this module exists to prevent.
pub struct CaptureLock {
    file: fs::File,
}

impl CaptureLock {
    pub fn acquire(archive: &Path, attempts: u32, backoff_ms: u64) -> Result<Option<Self>> {
        fs::create_dir_all(archive)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(archive.join(LOCK_FILE))?;
        for attempt in 0..attempts.max(1) {
            if file.try_lock().is_ok() {
                return Ok(Some(Self { file }));
            }
            if attempt + 1 < attempts.max(1) {
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            }
        }
        Ok(None)
    }
}

impl Drop for CaptureLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}
