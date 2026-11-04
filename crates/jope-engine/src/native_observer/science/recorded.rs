use super::*;

pub(super) fn regular(path: &Path, limit: u64) -> Result<Vec<u8>> {
    physical(path)?;
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let before = f.metadata()?;
    ensure!(
        before.is_file() && before.len() <= limit,
        "bounded regular science evidence required"
    );
    let mut b = Vec::new();
    (&f).take(limit + 1).read_to_end(&mut b)?;
    let after = f.metadata()?;
    let named = fs::symlink_metadata(path)?;
    ensure!(
        b.len() as u64 <= limit
            && before.len() == after.len()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.dev() == named.dev()
            && before.ino() == named.ino()
            && !named.file_type().is_symlink(),
        "scientific source changed during read"
    );
    Ok(b)
}
pub(super) fn checked(root: Option<&Path>, a: &Artifact, limit: u64) -> Result<Value> {
    ensure!(
        a.sha256.len() == 64
            && a.sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid science digest"
    );
    let path = match root {
        Some(root) => crate::manifest::resolve(root, &a.path)?,
        None => a.path.clone(),
    };
    let bytes = regular(&path, limit)?;
    ensure!(hash(&bytes) == a.sha256, "scientific artifact hash changed");
    Ok(serde_json::from_slice(&bytes)?)
}
pub(super) fn artifact(v: &Value) -> Result<Artifact> {
    let a = Artifact {
        path: v["path"].as_str().context("artifact path absent")?.into(),
        sha256: v["sha256"]
            .as_str()
            .context("artifact digest absent")?
            .into(),
    };
    ensure!(
        a.path.as_os_str().len() <= 4096
            && a.sha256.len() == 64
            && a.sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "malformed scientific artifact identity"
    );
    Ok(a)
}
pub(super) fn original_time(value: Option<u64>, at: u64) -> Result<Option<u64>> {
    ensure!(
        value.is_none_or(|t| t > 0 && t <= at.saturating_add(5000)),
        "original scientific timestamp is invalid/future"
    );
    Ok(value)
}
pub(super) fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|v| v.is_finite() && *v >= 0.0)
}
pub(super) fn row_evidence(
    id: &str,
    reference: &Artifact,
    kind: &str,
    original: Option<u64>,
    at: u64,
) -> Value {
    json!({"id":id,"artifact":reference,"verification_kind":kind,"original_at_unix_ms":original,
        "verified_at_unix_ms":at,"stale_after_ms":STALE_MS,"production_model_improvement_claim":false})
}
pub(super) fn qualification_rows(
    id: &str,
    a: &Artifact,
    v: &Value,
    original: Option<u64>,
    at: u64,
) -> Result<Rows> {
    let mut rows = Rows::default();
    original_time(original, at)?;
    rows.evidence.push(row_evidence(
        id,
        a,
        "hash_checked_root_catalog_recorded_qualification",
        original,
        at,
    ));
    rows.notes.push(text(format!("Historical qualification {id}; original time={original:?}; receipt {}. Recorded review, not fresh numerical execution or production learning.",a.sha256)));
    match v["schema"]
        .as_str()
        .context("qualification schema absent")?
    {
        "jope.hardware_training_benchmark_verified.v1" => {
            ensure!(
                v["production_learning"] == false && v["numerical_result_verified"] == true,
                "qualification training flags disagree"
            );
            let initial = v["initial_optimizer_step"]
                .as_u64()
                .context("initial optimizer step absent")?;
            let target = v["target_optimizer_step"]
                .as_u64()
                .context("target optimizer step absent")?;
            ensure!(
                initial.checked_add(v["exact_updates"].as_u64().context("update count absent")?)
                    == Some(target),
                "qualification optimizer accounting differs"
            );
            let state = artifact(&v["checkpoint"])?;
            rows.checkpoints.push(json!({"identity":state.sha256,"status":"historical qualification checkpoint","step":target,
                "reason":text(format!("{id}; {} updates; microbatch {}; lifecycle passed={}; original receipt {}",v["exact_updates"],v["microbatch"],v["full_scheduler_lifecycle_passed"],a.sha256)),
                "qualification":true,"original_at_unix_ms":original,"source":a,"checkpoint":state}));
        }
        "jope.actual_neural_eof_recovery_qualification.v1" => {
            ensure!(
                v["production_learning"] == false
                    && v["passed"] == true
                    && v["unknown_final_tail"] == false,
                "EOF qualification incomplete"
            );
            let initial = v["original_initial_step"]
                .as_u64()
                .context("original initial absent")?;
            let target = v["original_target_step"]
                .as_u64()
                .context("original target absent")?;
            let count = v["first_attempt_committed_updates"]
                .as_u64()
                .context("first updates absent")?
                .checked_add(
                    v["retry_executed_updates"]
                        .as_u64()
                        .context("retry updates absent")?,
                )
                .context("update overflow")?;
            ensure!(
                initial.checked_add(count) == Some(target) && v["total_updates"] == count,
                "EOF original window differs"
            );
            let state = artifact(&v["final_checkpoint"])?;
            rows.checkpoints.push(json!({"identity":state.sha256,"status":"historical qualification EOF recovery","step":target,
                "reason":text(format!("{id}; original {initial} + {count} updates; job {}; not production learning",v["job"]["id"])),
                "qualification":true,"original_at_unix_ms":original,"source":a,"checkpoint":state}));
        }
        "jope.native_qualification_receipt_check.v1" => {
            ensure!(
                v["qualification_only"] == true
                    && v["output"]["schema"] == "jope.native_gp_completion.v1"
                    && v["durable_result"]["schema"] == "jope.smartcluster_completion.v1"
                    && v["durable_result"]["manifest_sha256"] == v["manifest_sha256"],
                "GP qualification binding differs"
            );
            rows.notes.push(text(format!("Historical population GP {id}: logical budget {}, native receipt {}; this is not a seeded-continuation result.",v["output"]["evaluations_budget"],v["durable_result"]["receipt_id"])));
        }
        "jope.actual_gp_archive_independent_restoration.v1" => {
            ensure!(
                v["qualification_only"] == true
                    && v["passed"] == true
                    && v["source_node_access"] == false,
                "restoration qualification flags differ"
            );
            rows.notes.push(text(format!("Historical GP archive restoration {id}: {} original files / {} bytes; {} authorization witness rows; no fresh fits inferred.",v["regular_files_verified"],v["bytes_verified"],v["table_counts"]["authorization_witness"])));
        }
        "jope.actual_native_scope_recovery_qualification.v1" => {
            ensure!(
                v["fixture_only"] == true
                    && v["fits"] == 0
                    && v["activations"] == 0
                    && v["acknowledged_scope_replayed_after_stale_db"] == true
                    && v["recovery_hold_retained"] == true,
                "scope fixture is not retained zero-fit recovery evidence"
            );
            rows.notes.push(text(format!("Historical scope-recovery fixture {id}: {} native identities restored, zero fits/activations; metadata qualification only",v["native_training_rows"])));
        }
        schema => anyhow::bail!("unsupported recorded qualification schema {schema}"),
    }
    stamp_rows(&mut rows, at);
    Ok(rows)
}
pub(super) fn stamp_rows(rows: &mut Rows, at: u64) {
    for row in rows
        .checkpoints
        .iter_mut()
        .chain(&mut rows.continuations)
        .chain(&mut rows.comparisons)
        .chain(&mut rows.winners)
    {
        if row.get("scientific_verified_at_unix_ms").is_none() {
            row["scientific_verified_at_unix_ms"] = json!(at);
        }
    }
    if let Some(c) = rows.corpus.as_mut() {
        if c.get("scientific_verified_at_unix_ms").is_none() {
            c["scientific_verified_at_unix_ms"] = json!(at);
        }
    }
}
pub(super) fn merge(to: &mut Rows, mut from: Rows) {
    to.checkpoints.append(&mut from.checkpoints);
    to.continuations.append(&mut from.continuations);
    to.comparisons.append(&mut from.comparisons);
    to.winners.append(&mut from.winners);
    if from.corpus.is_some() {
        to.corpus = from.corpus;
    }
    to.notes.append(&mut from.notes);
    to.evidence.append(&mut from.evidence);
    to.errors.append(&mut from.errors);
    for (event, status) in from.publications {
        if let Some(previous) = to.publications.get(&event) {
            if previous["local_commit"] != status["local_commit"] {
                to.errors.push(text(format!(
                    "Conflicting recorded publication for winner {event}"
                )));
                continue;
            }
        }
        to.publications.insert(event, status);
    }
}
pub(super) fn bounded_files(path: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    physical(path)?;
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        ensure!(
            files.len() < limit,
            "scientific directory census exceeds bound"
        );
        files.push(entry?.path());
    }
    files.sort();
    Ok(files)
}
