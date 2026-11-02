fn naive_mase_denominator(train: &[Vec<f64>]) -> Result<f64> {
    ensure!(
        train.len() >= 2,
        "timeseries MASE needs at least two training rows"
    );
    let mut abs = 0.0;
    for pair in train.windows(2) {
        abs += (pair[1].last().context("empty row")? - pair[0].last().context("empty row")?).abs();
    }
    let denom = abs / (train.len() - 1) as f64;
    ensure!(
        denom.is_finite() && denom > 0.0,
        "invalid naive MASE denominator"
    );
    Ok(denom)
}

fn evaluate_tables(
    tr: &[Vec<f64>],
    va: &[Vec<f64>],
    genome: &Path,
    output: &Path,
    task: &str,
    mase_denominator: Option<f64>,
    preprocessing: &[u8],
) -> Result<()> {
    crate::task_contract::require_task(task)?;
    tchless_threads();
    let (x, y) = xy(tr);
    let (vx, vy) = xy(va);
    let g = GenomeIr::parse(&fs::read_to_string(genome)?)?;
    let start = Instant::now();
    // The ordered training and scoring row digests are bound into the feature view, the
    // baseline evidence and the score identity. Each one formats the whole table as CSV and
    // hashes it, so compute each once (training here, where it was first needed and timed;
    // scoring after the timed fit/predict, as before) and reuse the strings.
    let training_rows_sha256 = csv_sha256(tr)?;
    let feature_view = raw_feature_view(&training_rows_sha256, preprocessing)?;
    let fitted = crate::fit_boundary::fit(
        &g.canonical(),
        x,
        y,
        preprocessing.to_vec(),
        feature_view,
        output,
    )?;
    let pred = fitted.model.predict(&vx)?;
    let mut m = if task == "regression" {
        let mut row = metrics(&vy, &pred)?;
        row["units"] = json!("cut_normalized_target");
        row
    } else {
        crate::task_contract::score_predictions(task, &vy, &pred, mase_denominator)?
    };
    m["task"] = json!(task);
    m["primary_metric"] = json!(crate::task_contract::primary_metric(task)?);
    m["seconds"] = json!(start.elapsed().as_secs_f64());
    m["semantic_sha256"] = json!(hash(&serde_json::to_vec(&pred)?));
    let scoring_rows_sha256 = csv_sha256(va)?;
    // Produce the baseline from the very same authorized cut arrays. This is
    // a new diagnostic; historical results without it stay unavailable.
    let training_mean = tr.iter().map(|row| row[row.len() - 1]).sum::<f64>() / tr.len() as f64;
    let baseline = metrics(&vy, &vec![training_mean; vy.len()]);
    if let Some(baseline) = baseline
        .ok()
        .filter(|row| row["rmse"].as_f64().is_some_and(f64::is_finite))
    {
        m["baseline_evidence"] = json!({"schema":"parallel.cut_baseline.v1",
            "rule":"training_target_mean_on_scoring_rows","units":"cut_normalized_target",
            "rmse":baseline["rmse"],"training_target_mean":training_mean,
            "training_rows":tr.len(),"scoring_rows":va.len(),
            "training_sha256":training_rows_sha256,"scoring_sha256":scoring_rows_sha256,
            "preprocessing_sha256":hash(preprocessing)});
    } else {
        m["baseline_status"] = json!("nonfinite baseline; ratios unavailable");
    }
    // W5-T03: fit and score identities are distinct. The fit key is the
    // authority's fitted identity (program, ordered training rows,
    // preprocessing, feature view, numerical contract, seed) and may be reused.
    // The score identity additionally binds the exact fitted subject and the
    // scored rows and protocol, so a reused fit is always rescored on its own
    // rows and a score never transfers to another subject or membership.
    let receipt = crate::eval_registry::verify_completion_receipt(&fs::read(
        fitted.completion["path"]
            .as_str()
            .context("fit completion path absent")?,
    )?)?;
    let score_identity = json!({"schema":"parallel.gp_score_identity.v1",
        "fit_key":receipt.fit_key,"model_sha256":receipt.model_sha256,
        "training_rows_sha256":training_rows_sha256,"scoring_rows_sha256":scoring_rows_sha256,
        "preprocessing_sha256":hash(preprocessing),"task":task,
        "primary_metric":m["primary_metric"],"metric_units":m["units"],
        "evaluator_protocol":crate::label_admission::PROTOCOL,
        "predictions_sha256":m["semantic_sha256"]});
    m["score_identity_sha256"] = json!(hash(&serde_json::to_vec(&score_identity)?));
    m["score_identity"] = score_identity;
    m["fit_completion"] = fitted.completion;
    m["fitted_model"] = fitted.model_reference;
    m["evaluation_origin"] = json!(if fitted.reused { "reuse" } else { "execution" });
    m["physical_fit_starts"] = json!(usize::from(!fitted.reused));
    write_json(output, &m)
}
fn tchless_threads() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
        )
        .build_global();
}

/// Read complete records without changing original evidence. A torn tail is an
/// unavailable outcome requiring reconciliation, never permission to retry.
pub fn read_journal(path: &Path) -> Result<Vec<serde_json::Value>> {
    let bytes = fs::read(path)?;
    let end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    ensure!(
        end == bytes.len(),
        "incomplete GP journal; original bytes retained for reconciliation"
    );
    let records = bytes[..end]
        .split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(records)
}

/// A journaled success is resumable measured evidence only while its original
/// completion attests exactly that output. Model eviction is not examined here;
/// `selection_replay` authenticates the complete fitted evidence independently.
fn journaled_measurement_attested(root: &Path, row: &serde_json::Value) -> Result<()> {
    let completion = crate::label_admission::artifact(root, &row["clean_completion"])
        .context("journaled measurement lacks its original completion")?;
    ensure!(
        completion["schema"] == crate::label_admission::COMPLETION_SCHEMA
            && completion["outcome"] == "success"
            && completion["failure"].is_null()
            && completion["evaluation_identity_sha256"] == row["evaluation_identity_sha256"]
            && completion["program_sha256"] == row["program_sha256"]
            && crate::label_admission::artifact(root, &completion["output"])? == row["metrics"],
        "journaled measurement differs from its original completion"
    );
    Ok(())
}

/// W5-T05: one live searcher per output directory. The exclusive advisory
/// lease is held until the searcher returns and is released by the kernel if
/// it dies, so crash recovery needs no manual cleanup. The descriptor is
/// close-on-exec: evaluator children never hold it.
fn search_lease(output: &Path) -> Result<fs::File> {
    let lease = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("search.lease"))?;
    match lease.try_lock() {
        Ok(()) => Ok(lease),
        Err(fs::TryLockError::WouldBlock) => {
            anyhow::bail!(
                "search output is held by another live searcher; concurrent dispatch refused"
            )
        }
        Err(fs::TryLockError::Error(error)) => Err(error.into()),
    }
}

/// Whether the evaluator has exited, without reaping it: an unreaped leader
/// keeps its process-group id reserved while the group is drained.
fn evaluator_exited(pid: i32) -> Result<bool> {
    // SAFETY: siginfo_t contains C integer/pointer fields and unions whose all-zero
    // representation is valid; initializing it first also gives WNOHANG's no-result case
    // a defined si_pid value without relying on unspecified output bytes.
    let mut info = unsafe { std::mem::zeroed::<libc::siginfo_t>() };
    // SAFETY: info is a fully initialized writable siginfo_t and waitid accepts its address;
    // WNOWAIT leaves a ready child unreaped so its pid stays reserved.
    let status = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    ensure!(
        status == 0,
        "evaluator wait failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: info is initialized before waitid, and successful waitid either writes a child
    // result or leaves the initialized WNOHANG no-result value intact.
    Ok(unsafe { info.si_pid() } != 0)
}

/// Live (non-zombie) members of a process group other than its leader.
fn process_group_members(group: i32) -> usize {
    let Ok(entries) = fs::read_dir("/proc") else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<i32>().ok())
        .filter(|&pid| pid != group)
        .filter(|pid| {
            let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
                return false;
            };
            // Fields after the parenthesised command: state, ppid, pgrp.
            let Some((_, rest)) = stat.rsplit_once(')') else {
                return false;
            };
            let mut fields = rest.split_whitespace();
            let state = fields.next();
            let pgrp = fields.nth(1).and_then(|g| g.parse::<i32>().ok());
            pgrp == Some(group) && state != Some("Z")
        })
        .count()
}

/// Kill the evaluator's whole process group and census what remains. Called
/// while the leader is unreaped (running on expiry, a zombie after exit).
fn drain_process_group(group: i32) -> serde_json::Value {
    let members_at_exit = process_group_members(group);
    // SAFETY: kill takes only integers and has no memory-safety preconditions; the caller holds
    // the unreaped leader, so `group` still names the evaluator's own process group.
    unsafe {
        libc::kill(-group, libc::SIGKILL);
    }
    let settle = Instant::now();
    let mut remaining = process_group_members(group);
    while remaining > 0 && settle.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(5));
        remaining = process_group_members(group);
    }
    json!({"isolated":true,"members_at_exit":members_at_exit,"members_after_drain":remaining})
}

fn experiment_failpoint(name: &str, isolated: bool) {
    // Only explicitly isolated experiments can exercise crash recovery gates.
    if isolated && std::env::var("JOPE_EXPERIMENT_CRASH_AFTER").ok().as_deref() == Some(name) {
        // SAFETY: raise has no memory-safety preconditions; SIGKILL ends this process at once,
        // which is the failpoint's purpose.
        unsafe {
            libc::raise(libc::SIGKILL);
        }
    }
}
