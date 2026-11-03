use super::*;

/// How many guard-held jobs one cycle may release. Deliberately bounded so a
/// 5-second cycle cannot flood a recovering host: draining 996 held jobs (the
/// 2026-08-01 pileup) at 64/cycle takes ~80 s, while the overload guard gets a
/// fresh look between batches. Release only fills the gap left under the
/// queue target — a full queue releases nothing.
pub(crate) const HELD_RELEASE_BATCH_MAX: usize = 64;

pub(crate) fn held_release_budget(
    queued_starting: usize,
    queued_target: i64,
    held: usize,
) -> usize {
    let target = queued_target.max(0) as usize;
    target
        .saturating_sub(queued_starting)
        .min(held)
        .min(HELD_RELEASE_BATCH_MAX)
}

pub(crate) fn scq_release(scq: &Path, job_id: &str) -> Result<()> {
    let output = Command::new(scq)
        .arg("release")
        .arg(job_id)
        .output()
        .with_context(|| format!("run SCQ release {job_id}"))?;
    if !output.status.success() {
        bail!(
            "SCQ refused release of {job_id}: {}",
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

pub(crate) fn scq_check(scq: &Path, spec: &Path) -> Result<()> {
    let output = Command::new(scq)
        .arg("check")
        .arg(spec)
        .output()
        .with_context(|| format!("run SCQ check {}", spec.display()))?;
    if !output.status.success() {
        bail!(
            "SCQ rejected {}: {}",
            spec.display(),
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

pub(crate) fn rewrite_spec_job_id(
    old: &Path,
    new: &Path,
    old_id: &str,
    new_id: &str,
) -> Result<()> {
    let text = fs::read_to_string(old)?;
    let old_line = format!("id = {}", serde_json::to_string(old_id)?);
    let new_line = format!("id = {}", serde_json::to_string(new_id)?);
    if !text.contains(&old_line) {
        bail!("retry source spec does not bind its claimed job id");
    }
    atomic_write(new, text.replacen(&old_line, &new_line, 1).as_bytes())
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("write target has no filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub(crate) fn authoritative_receipt_exists(root: &Path, job_id: &str) -> bool {
    root.join("receipts")
        .join(job_id)
        .join("SHARD_OK")
        .is_file()
}

pub(crate) fn reject_python_runtime(runner: &Path) -> Result<()> {
    let content = fs::read_to_string(runner)
        .with_context(|| format!("read evaluator runner {}", runner.display()))?;
    let lowered = content.to_ascii_lowercase();
    if ["python", "python3", ".py "]
        .iter()
        .any(|marker| lowered.contains(marker))
    {
        bail!("evaluator runner invokes Python");
    }
    Ok(())
}

pub(crate) fn path_field(value: &Value, key: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(string_field(value, key)?))
}

pub(crate) fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("manifest lacks {key}"))
}

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn is_infrastructure_failure_class(value: &str) -> bool {
    INFRA_FAILURES.contains(&value.to_ascii_lowercase().as_str())
}

pub(crate) fn is_transient_cli_error(detail: &str) -> bool {
    let detail = detail.to_ascii_lowercase();
    [
        "controller dropped request",
        "connection reset",
        "connection refused",
        "broken pipe",
        "timed out",
        "socket",
    ]
    .iter()
    .any(|marker| detail.contains(marker))
}

pub(crate) fn submit_batch(scq: &Path, claims: &[ReadyClaim]) -> Result<Vec<(bool, bool, String)>> {
    ensure!(
        !claims.is_empty() && claims.len() <= 1_024,
        "SCQ bulk submission requires 1..=1024 claims"
    );
    let output = Command::new(scq)
        .arg("--json")
        .arg("submit-batch")
        .args(claims.iter().map(|row| row.claim.spec.as_path()))
        .output()
        .with_context(|| format!("run SCQ bulk submit with {} specs", claims.len()))?;
    let value: Value = match serde_json::from_slice(&output.stdout) {
        Ok(value) => value,
        Err(error) => {
            let detail = format!(
                "SCQ bulk response parse failed: {error}; stdout={}; stderr={}",
                String::from_utf8_lossy(&output.stdout).trim(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
            return Ok((0..claims.len())
                .map(|_| (false, true, detail.clone()))
                .collect());
        }
    };
    let results = value
        .get("BatchSubmitted")
        .and_then(|value| value.get("results"))
        .and_then(Value::as_array);
    let Some(results) = results else {
        let detail = format!("SCQ rejected bulk request: {value}");
        return Ok((0..claims.len())
            .map(|_| (false, true, detail.clone()))
            .collect());
    };
    ensure!(
        results.len() == claims.len(),
        "SCQ bulk response count mismatch"
    );
    let mut outcomes = Vec::with_capacity(results.len());
    for (index, (result, claim)) in results.iter().zip(claims).enumerate() {
        ensure!(
            result.get("index").and_then(Value::as_u64) == Some(index as u64),
            "SCQ bulk response ordering mismatch"
        );
        let submitted = result
            .get("submitted")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if submitted {
            ensure!(
                result.get("id").and_then(Value::as_str) == Some(claim.claim.job_id.as_str()),
                "SCQ bulk response job identity mismatch"
            );
        }
        let detail = result
            .get("error")
            .and_then(Value::as_str)
            .or_else(|| result.get("code").and_then(Value::as_str))
            .unwrap_or(if submitted { "submitted" } else { "rejected" })
            .to_owned();
        let lowered = detail.to_ascii_lowercase();
        let accepted =
            submitted || lowered.contains("already exists") || lowered.contains("already active");
        outcomes.push((
            accepted,
            !accepted && is_transient_cli_error(&detail),
            detail,
        ));
    }
    Ok(outcomes)
}

pub(crate) fn parse_rfc3339_seconds(value: &str) -> Option<i64> {
    // SCQ emits UTC RFC3339 timestamps.  Parsing them locally avoids pulling
    // a high-level time runtime into the always-on keeper.
    let year = value.get(0..4)?.parse::<i64>().ok()?;
    let month = value.get(5..7)?.parse::<i64>().ok()?;
    let day = value.get(8..10)?.parse::<i64>().ok()?;
    let hour = value.get(11..13)?.parse::<i64>().ok()?;
    let minute = value.get(14..16)?.parse::<i64>().ok()?;
    let second = value.get(17..19)?.parse::<i64>().ok()?;
    let timezone = value.find('Z').map(|_| 0).or_else(|| {
        let index = value.get(19..)?.find(['+', '-'])? + 19;
        let sign = if value.as_bytes()[index] == b'+' {
            1
        } else {
            -1
        };
        let hours = value.get(index + 1..index + 3)?.parse::<i64>().ok()?;
        let minutes = value.get(index + 4..index + 6)?.parse::<i64>().ok()?;
        Some(sign * (hours * 3_600 + minutes * 60))
    })?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let days = days_from_civil(year, month, day)?;
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second - timezone)
}

pub(crate) fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}
