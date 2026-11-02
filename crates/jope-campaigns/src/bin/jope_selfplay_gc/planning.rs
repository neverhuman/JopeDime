use super::*;

pub(super) fn evaluate_gates(
    name: &str,
    prefix: &str,
    age_secs: u64,
    min_age_secs: u64,
    protected_newest: bool,
    scq_state: Option<&str>,
    ingested: bool,
    receipts_dir_exists: bool,
) -> Option<Vec<String>> {
    if !name.starts_with(prefix) || !sane_name(name) {
        return None;
    }
    if age_secs < min_age_secs || protected_newest {
        return None;
    }
    let mut basis = vec![
        format!("prefix:{prefix}"),
        format!("age_secs:{age_secs}"),
        "not_in_keep_newest".to_string(),
    ];
    match scq_state {
        None => basis.push("scheduler:pruned".to_string()),
        Some(state) if TERMINAL_STATES.contains(&state) => {
            basis.push(format!("scheduler:{state}"));
        }
        Some(_) => return None,
    }
    if ingested {
        basis.push("evidence:ingested_jobs".to_string());
    } else if receipts_dir_exists {
        basis.push("evidence:receipts_fetched".to_string());
    } else {
        return None;
    }
    Some(basis)
}

pub(super) fn ssh_target(node: &str) -> Result<Vec<String>> {
    match node {
        "xbabe0" | "xbabe1" | "xbabe3" => Ok(vec![
            "ssh".into(),
            "-o".into(),
            "ConnectTimeout=10".into(),
            node.into(),
        ]),
        "xbabe4" => Ok(vec![
            "ssh".into(),
            "-o".into(),
            "ConnectTimeout=10".into(),
            "-p".into(),
            "2224".into(),
            "xbabe4".into(),
        ]),
        LOCAL_NODE => bail!("xbabe2 is the local host; never ssh to it"),
        other => bail!("unknown node {other}"),
    }
}

pub(super) fn run_capture(program: &str, args: &[String], stdin: Option<&str>) -> Result<String> {
    let mut command = Command::new(program);
    command.args(args);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {program}"))?;
    if let Some(input) = stdin {
        child
            .stdin
            .as_mut()
            .context("stdin unavailable")?
            .write_all(input.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Byte-exact variant of `run_capture`. Archived metrics must be identical to
/// what the trainer wrote, so a lossy UTF-8 round trip is not acceptable here.
pub(super) fn run_capture_bytes(
    program: &str,
    args: &[String],
    stdin: Option<&str>,
) -> Result<Vec<u8>> {
    let mut command = Command::new(program);
    command.args(args);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {program}"))?;
    if let Some(input) = stdin {
        child
            .stdin
            .as_mut()
            .context("stdin unavailable")?
            .write_all(input.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

/// Enumerate immediate subdirectories with mtimes, locally or over ssh.
pub(super) fn enumerate_dirs(node: &str, runs_root: &str) -> Result<Vec<DirEntryInfo>> {
    if node == LOCAL_NODE {
        let mut rows = Vec::new();
        for entry in std::fs::read_dir(runs_root)? {
            let entry = entry.context("read local run-directory entry")?;
            let file_type = entry.file_type().context("read local run-directory type")?;
            if !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let mtime_ms = entry
                .metadata()
                .context("read local run-directory metadata")?
                .modified()
                .context("read local run-directory mtime")?
                .duration_since(UNIX_EPOCH)
                .context("local run-directory mtime predates Unix epoch")?
                .as_millis() as u64;
            rows.push(DirEntryInfo { name, mtime_ms });
        }
        return Ok(rows);
    }
    let target = ssh_target(node)?;
    let script = format!(
        "cd {runs_root} 2>/dev/null && find . -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\\n'"
    );
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(script);
    let stdout = run_capture(&target[0], &args, None)?;
    let mut rows = Vec::new();
    for (line_number, line) in stdout.lines().enumerate() {
        let (ts, name) = line.split_once(' ').with_context(|| {
            format!("malformed remote directory row at line {}", line_number + 1)
        })?;
        let secs: f64 = ts.parse().with_context(|| {
            format!("invalid remote directory mtime at line {}", line_number + 1)
        })?;
        if !secs.is_finite() || secs < 0.0 || !sane_name(name) {
            bail!("unsafe remote directory row at line {}", line_number + 1);
        }
        rows.push(DirEntryInfo {
            name: name.to_string(),
            mtime_ms: (secs * 1000.0) as u64,
        });
    }
    Ok(rows)
}

pub(super) fn parse_scq_job_states(stdout: &[u8]) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    let value = serde_json::from_slice::<Value>(stdout).context("decode SCQ job-state JSON")?;
    let jobs = value
        .pointer("/Jobs/jobs")
        .or_else(|| value.get("jobs"))
        .and_then(Value::as_array)
        .context("SCQ job-state JSON has no jobs array")?;
    for job in jobs {
        let id = job
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .context("SCQ job is missing a non-empty id")?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .context("SCQ job is missing a non-empty state")?;
        if map.insert(id.to_string(), state.to_string()).is_some() {
            bail!("SCQ job-state JSON repeats job id {id}");
        }
    }
    Ok(map)
}

pub(super) fn scq_job_states() -> Result<HashMap<String, String>> {
    let output = Command::new(scq_str())
        .args(["--json", "ls"])
        .output()
        .context("execute SCQ job-state query")?;
    if !output.status.success() {
        bail!(
            "SCQ job-state query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    parse_scq_job_states(&output.stdout)
}

/// Look up which of `names` are durably ingested. Falls back to the empty set
/// on persistent lock/errors (the receipts-dir gate still applies then).
pub(super) fn ingested_set(root: &Path, names: &[String]) -> HashSet<String> {
    let mut found = HashSet::new();
    let corpus = root.join("corpus/strict-test-exact.sqlite");
    let Ok(connection) = Connection::open_with_flags(
        &corpus,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return found;
    };
    if connection.busy_timeout(Duration::from_secs(60)).is_err() {
        return found;
    }
    let Ok(mut statement) = connection.prepare("SELECT 1 FROM ingested_jobs WHERE job_id = ?")
    else {
        return found;
    };
    for name in names {
        if let Ok(mut rows) = statement.query([name]) {
            if let Ok(Some(_)) = rows.next() {
                found.insert(name.clone());
            }
        }
    }
    found
}

pub(super) struct Plan {
    pub(super) schema: &'static str,
    pub(super) node: String,
    pub(super) entries: Vec<Value>,
    pub(super) candidate_names: Vec<String>,
    pub(super) estimated_bytes: u64,
    pub(super) foreign_prefix_counts: HashMap<String, usize>,
    pub(super) total_dirs: usize,
}

pub(super) fn build_runs_plan(
    node: &str,
    runs_root: &str,
    root: &Path,
    prefix: &str,
    min_age_hours: u64,
    keep_newest: usize,
) -> Result<Plan> {
    let mut dirs = enumerate_dirs(node, runs_root)?;
    let total_dirs = dirs.len();
    let mut foreign: HashMap<String, usize> = HashMap::new();
    for entry in &dirs {
        if !entry.name.starts_with(prefix) {
            let key = entry
                .name
                .split(['-', '_'])
                .next()
                .unwrap_or("other")
                .to_string();
            *foreign.entry(key).or_default() += 1;
        }
    }
    dirs.retain(|d| d.name.starts_with(prefix));
    dirs.sort_by_key(|d| std::cmp::Reverse(d.mtime_ms));
    let protected: HashSet<String> = dirs
        .iter()
        .take(keep_newest)
        .map(|d| d.name.clone())
        .collect();
    let states = scq_job_states()?;
    let names: Vec<String> = dirs.iter().map(|d| d.name.clone()).collect();
    let ingested = ingested_set(root, &names);
    let receipts_root = root.join("receipts");
    let now = now_ms();
    let min_age_secs = min_age_hours * 3600;
    let mut entries = Vec::new();
    let mut candidate_names = Vec::new();
    for dir in &dirs {
        let age_secs = now.saturating_sub(dir.mtime_ms) / 1000;
        let basis = evaluate_gates(
            &dir.name,
            prefix,
            age_secs,
            min_age_secs,
            protected.contains(&dir.name),
            states.get(&dir.name).map(String::as_str),
            ingested.contains(&dir.name),
            receipts_root.join(&dir.name).is_dir(),
        );
        if let Some(basis) = basis {
            entries.push(json!({
                "dir": format!("{runs_root}/{}", dir.name),
                "mtime_ms": dir.mtime_ms,
                "basis": basis,
            }));
            candidate_names.push(dir.name.clone());
        }
    }
    let estimated_bytes = estimate_bytes(node, runs_root, &candidate_names)?;
    Ok(Plan {
        schema: "JopeSelfplayGcPlan.v1",
        node: node.to_string(),
        entries,
        candidate_names,
        estimated_bytes,
        foreign_prefix_counts: foreign,
        total_dirs,
    })
}

/// Sample-based size estimate: du -sk over up to SAMPLE_FOR_SIZE candidates,
/// mean extrapolated over the full candidate list.
pub(super) fn estimate_bytes(node: &str, runs_root: &str, names: &[String]) -> Result<u64> {
    if names.is_empty() {
        return Ok(0);
    }
    let step = (names.len() / SAMPLE_FOR_SIZE).max(1);
    let sample: Vec<&String> = names.iter().step_by(step).take(SAMPLE_FOR_SIZE).collect();
    let listing = sample
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let stdout = if node == LOCAL_NODE {
        let args = vec![
            "-c".to_string(),
            format!("cd {runs_root} && xargs -d '\\n' du -sk -- 2>/dev/null | awk '{{s+=$1}} END {{print s}}'"),
        ];
        run_capture("sh", &args, Some(&listing))?
    } else {
        let target = ssh_target(node)?;
        let mut args: Vec<String> = target[1..].to_vec();
        args.push(format!(
            "cd {runs_root} && xargs -d '\\n' du -sk -- 2>/dev/null | awk '{{s+=$1}} END {{print s}}'"
        ));
        run_capture(&target[0], &args, Some(&listing))?
    };
    let sampled_kb: u64 = stdout.trim().parse().unwrap_or(0);
    let mean = sampled_kb as f64 / sample.len().max(1) as f64;
    Ok((mean * names.len() as f64 * 1024.0) as u64)
}

pub(super) fn write_plan(root: &Path, file_name: &str, plan: &Plan) -> Result<PathBuf> {
    let report = json!({
        "schema": plan.schema,
        "node": plan.node,
        "generated_at_ms": now_ms(),
        "total_dirs_seen": plan.total_dirs,
        "candidate_count": plan.entries.len(),
        "estimated_bytes": plan.estimated_bytes,
        "estimated_bytes_note": "sampled du, extrapolated",
        "foreign_prefix_counts": plan.foreign_prefix_counts,
        "entries": plan.entries,
    });
    let path = root.join("ops").join(file_name);
    let temporary = root
        .join("ops")
        .join(format!(".{file_name}.{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(&report)?)?;
    std::fs::rename(&temporary, &path)?;
    Ok(path)
}

pub(super) fn append_receipt(root: &Path, record: Value) -> Result<()> {
    let path = root.join("ops/gc-events.v1.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

pub(super) fn delete_batch(node: &str, runs_root: &str, batch: &[String]) -> Result<()> {
    for name in batch {
        if !sane_name(name) {
            bail!("refusing unsafe dir name {name:?}");
        }
    }
    if node == LOCAL_NODE {
        for name in batch {
            let path = Path::new(runs_root).join(name);
            if path.is_dir() {
                std::fs::remove_dir_all(&path)
                    .with_context(|| format!("remove {}", path.display()))?;
            }
        }
        return Ok(());
    }
    let listing = batch.join("\n");
    let target = ssh_target(node)?;
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(format!("cd {runs_root} && xargs -d '\\n' rm -rf --"));
    run_capture(&target[0], &args, Some(&listing))?;
    Ok(())
}
