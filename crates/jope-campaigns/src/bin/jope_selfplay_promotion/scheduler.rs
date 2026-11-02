use super::*;

// ---------------------------------------------------------------------------
// Scheduler helpers
// ---------------------------------------------------------------------------

pub(super) fn run_command(program: &str, args: &[&str]) -> Result<(i32, String, String)> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("spawn {program}"))?;
    Ok((
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

pub(super) fn scq_json(args: &[&str]) -> Result<Value> {
    let (code, stdout, stderr) = run_command(scq_str(), args)?;
    if code != 0 {
        bail!(
            "{}",
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    if stdout.trim().is_empty() {
        return Ok(json!({}));
    }
    Ok(serde_json::from_str(&stdout)?)
}

pub(super) fn scheduler_jobs() -> Result<BTreeMap<String, Value>> {
    let value = scq_json(&["--json", "ls"])?;
    let mut jobs = BTreeMap::new();
    if let Some(rows) = value
        .get("Jobs")
        .and_then(|j| j.get("jobs"))
        .and_then(Value::as_array)
    {
        for row in rows {
            if let Some(id) = row.get("id").and_then(Value::as_str) {
                if !id.is_empty() {
                    jobs.insert(id.to_owned(), row.clone());
                }
            }
        }
    }
    Ok(jobs)
}

pub(super) fn job_state<'j>(jobs: &'j BTreeMap<String, Value>, job_id: &str) -> Option<&'j str> {
    jobs.get(job_id)
        .and_then(|row| row.get("state"))
        .and_then(Value::as_str)
}

/// `scq check` then `scq submit`, tolerating an already-submitted job.
pub(super) fn check_and_submit(spec: &str) -> Result<()> {
    let (code, stdout, stderr) = run_command(scq_str(), &["check", spec])?;
    if code != 0 {
        bail!(
            "{}",
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    let (code, stdout, stderr) = run_command(scq_str(), &["submit", spec])?;
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    if code != 0 && !detail.to_lowercase().contains("already exists") {
        bail!("{detail}");
    }
    Ok(())
}

/// Locate outputs for a job the scheduler has already pruned from its records.
pub(super) fn pruned_job_outputs(job_id: &str) -> Option<PathBuf> {
    let attempts = Path::new(runs_root_str()).join(job_id).join("attempts");
    if !attempts.is_dir() {
        return None;
    }
    let mut outputs: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&attempts).ok()?.flatten() {
        let candidate = entry.path().join("outputs");
        if candidate.is_dir() {
            let modified = fs::metadata(&candidate)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            outputs.push((modified, candidate));
        }
    }
    outputs.sort();
    for (_, candidate) in outputs.into_iter().rev() {
        let mut results = Vec::new();
        rglob(&candidate, "train-result.json", &mut results);
        if !results.is_empty() {
            return Some(candidate);
        }
    }
    None
}

pub(super) fn fetch_job_output(job_id: &str, destination: &Path) -> Result<PathBuf> {
    if destination.is_dir() {
        return Ok(destination.to_path_buf());
    }
    let name = destination
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("fetch destination has no name"))?;
    let temporary = destination.with_file_name(format!(".{name}.fetch-{}", std::process::id()));
    fs::create_dir_all(&temporary)?;
    let (code, stdout, stderr) = run_command(
        scq_str(),
        &["fetch", job_id, "--output", &temporary.to_string_lossy()],
    )?;
    if code != 0 {
        let _ = fs::remove_dir_all(&temporary);
        bail!(
            "{}",
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    fs::rename(&temporary, destination)?;
    Ok(destination.to_path_buf())
}
