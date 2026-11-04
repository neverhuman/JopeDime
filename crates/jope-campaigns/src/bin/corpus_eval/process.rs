use super::*;

/// Put a child in its own process group.  `eval-one` is intentionally a
/// private protocol: it reads the pinned manifest, evaluates exactly one
/// candidate, and writes one small JSON outcome.  The parent owns receipt
/// publication and therefore never ingests a child-owned partial artifact.
pub(crate) fn spawn_eval_one(
    manifest: &Path,
    candidate_index: usize,
    result: &Path,
) -> Result<std::process::Child> {
    let executable = env::current_exe().context("resolve corpus_eval executable")?;
    let mut command = Command::new(executable);
    command
        .arg("eval-one")
        .arg("--manifest")
        .arg(manifest)
        .arg("--candidate-index")
        .arg(candidate_index.to_string())
        .arg("--result")
        .arg(result)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls (setsid,
    // prctl, getppid); its errors are errno codes, so nothing allocates in the child.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let parent = libc::getppid();
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                // The parent exited: refuse with an errno, which does not allocate
                // (an io::Error carrying a message would box it in the forked child).
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
    command
        .spawn()
        .with_context(|| format!("spawn eval-one candidate {candidate_index}"))
}

pub(crate) fn process_group_exists(pid: libc::pid_t) -> Result<bool> {
    #[cfg(unix)]
    // SAFETY: kill with signal 0 sends nothing and reads no memory; every caller passes a
    // spawned child's PID (never 0 or 1), so -pid names one process group.
    unsafe {
        if libc::kill(-pid, 0) == 0 {
            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(false);
        }
        Err(error).context("probe eval-one process group")
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        Ok(false)
    }
}

pub(crate) fn wait_for_process_group_exit(pid: libc::pid_t, grace: Duration) -> Result<bool> {
    let deadline = Instant::now() + grace;
    loop {
        if !process_group_exists(pid)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

pub(crate) fn terminate_process_group(
    child: &mut std::process::Child,
    grace: Duration,
) -> Result<()> {
    let pid = child.id() as libc::pid_t;
    #[cfg(unix)]
    // SAFETY: kill reads no memory; pid is the spawned child's PID (never 0 or 1), so -pid names
    // one process group. __errno_location returns this thread's valid, aligned errno pointer.
    unsafe {
        // The child is a session leader created by setsid, so its PID is its
        // process-group ID. A negative kill reaches descendants as well.
        if libc::kill(-pid, libc::SIGTERM) != 0 && *libc::__errno_location() != libc::ESRCH {
            return Err(std::io::Error::last_os_error()).context("terminate eval-one group");
        }
    }
    let deadline = Instant::now() + grace;
    let mut leader_reaped = false;
    while Instant::now() < deadline {
        if !leader_reaped && child.try_wait()?.is_some() {
            leader_reaped = true;
        }
        if leader_reaped && !process_group_exists(pid)? {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    #[cfg(unix)]
    // SAFETY: kill reads no memory; pid is the spawned child's PID (never 0 or 1), so -pid names
    // one process group. __errno_location returns this thread's valid, aligned errno pointer.
    unsafe {
        if libc::kill(-pid, libc::SIGKILL) != 0 && *libc::__errno_location() != libc::ESRCH {
            return Err(std::io::Error::last_os_error()).context("kill eval-one group");
        }
    }
    if !leader_reaped {
        child.wait().context("reap eval-one process group")?;
    }
    if !wait_for_process_group_exit(pid, grace)? {
        bail!("eval-one process group survived SIGKILL grace period");
    }
    Ok(())
}

pub(crate) fn run_with_timeout(
    manifest: &Path,
    candidate_index: usize,
    result: &Path,
    timeout: Duration,
) -> Result<Outcome> {
    let started = Instant::now();
    let mut child = match spawn_eval_one(manifest, candidate_index, result) {
        Ok(child) => child,
        Err(error) => {
            return Ok(Outcome::Fail {
                class: "panic",
                message: format!("{error:#}"),
            });
        }
    };
    let pid = child.id() as libc::pid_t;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !wait_for_process_group_exit(pid, Duration::from_millis(250))? {
                    return Err(anyhow::anyhow!(
                        "eval-one process group survived after leader exit"
                    ));
                }
                if !status.success() {
                    return Ok(Outcome::Fail {
                        class: "panic",
                        message: format!("eval-one exited with {status}"),
                    });
                }
                let bytes = match fs::read(result) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        return Ok(Outcome::Fail {
                            class: "panic",
                            message: format!("eval-one result is missing: {error}"),
                        });
                    }
                };
                return Ok(
                    match serde_json::from_slice::<EvalOneWire>(&bytes)
                        .map_err(|error| error.to_string())
                        .and_then(Outcome::try_from)
                    {
                        Ok(outcome) => outcome,
                        Err(error) => Outcome::Fail {
                            class: "panic",
                            message: format!("invalid eval-one result: {error}"),
                        },
                    },
                );
            }
            Ok(None) if started.elapsed() >= timeout => {
                let message = format!("exceeded {} second wall-clock budget", timeout.as_secs());
                terminate_process_group(&mut child, Duration::from_millis(250)).with_context(
                    || format!("cleanup timed-out eval-one candidate {candidate_index}"),
                )?;
                return Ok(Outcome::Fail {
                    class: "timeout",
                    message,
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(error) => {
                terminate_process_group(&mut child, Duration::from_millis(250)).with_context(
                    || format!("cleanup after polling eval-one candidate {candidate_index}"),
                )?;
                return Ok(Outcome::Fail {
                    class: "panic",
                    message: format!("poll eval-one: {error}"),
                });
            }
        }
    }
}

pub(crate) fn evaluate_one_manifest(shard: &Shard, candidate_index: usize) -> Result<Outcome> {
    let candidate = shard
        .candidates
        .get(candidate_index)
        .context("candidate index is outside the shard")?;
    let reduction_version = resolve_reduction(shard)?;
    if shard.task == "timeseries" {
        let data = load_timeseries_shard_data(shard)?;
        let result = catch_unwind(AssertUnwindSafe(|| match reduction_version {
            ReductionVersion::V1 => evaluate_timeseries(candidate, &data),
            ReductionVersion::V2 => evaluate_timeseries_v2(candidate, &data),
        }));
        return Ok(match result {
            Ok(outcome) => outcome,
            Err(panic) => Outcome::Fail {
                class: "panic",
                message: panic_message(panic),
            },
        });
    }

    let mut train = load_pinned_csv(&shard.train_csv, &shard.train_sha256, "train")?;
    let mut test = load_pinned_csv(&shard.test_csv, &shard.test_sha256, "test")?;
    if train.features != test.features {
        bail!(
            "train/test feature widths differ ({} vs {})",
            train.features,
            test.features
        );
    }
    let mut multiclass_context = None;
    let dataset_failure = if shard.task == "binary" {
        validate_binary_targets(&train.y, "train")
            .or_else(|| validate_binary_targets(&test.y, "test"))
    } else if shard.task == "multiclass" {
        let meta_path = shard
            .meta_json
            .as_deref()
            .context("multiclass shard manifest must pin meta_json")?;
        let meta_sha256 = shard
            .meta_sha256
            .as_deref()
            .context("multiclass shard manifest must pin meta_sha256")?;
        let meta = load_pinned_json(meta_path, meta_sha256, "meta")?;
        match decode_multiclass_splits(&train.y, &test.y, &meta) {
            Ok(decoded) => {
                train.y = decoded.train;
                test.y = decoded.test;
                multiclass_context = Some(decoded.context);
                None
            }
            Err(error) => Some(error),
        }
    } else {
        None
    };
    if let Some(message) = dataset_failure {
        return Ok(Outcome::Fail {
            class: "fit",
            message,
        });
    }
    let tables = if matches!(shard.task.as_str(), "binary" | "multiclass") {
        Some((
            raw_table(train.features, &train.x)?,
            raw_table(test.features, &test.x)?,
        ))
    } else {
        None
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| match tables {
        None => evaluate_regression(candidate, &train, &test),
        Some((train_table, test_table)) if shard.task == "multiclass" => evaluate_multiclass(
            candidate,
            &train_table,
            &train.y,
            &test_table,
            &test.y,
            train.features,
            multiclass_context
                .as_ref()
                .expect("validated multiclass context"),
        ),
        Some((train_table, test_table)) => evaluate_binary(
            candidate,
            &train_table,
            &train.y,
            &test_table,
            &test.y,
            train.features,
        ),
    }));
    Ok(match outcome {
        Ok(outcome) => outcome,
        Err(panic) => Outcome::Fail {
            class: "panic",
            message: panic_message(panic),
        },
    })
}

pub(crate) fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|value| (*value).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "opaque panic payload".into())
}

pub(crate) fn run_eval_one(manifest: &Path, candidate_index: usize, result: &Path) -> Result<()> {
    let bytes = fs::read(manifest).with_context(|| format!("read {}", manifest.display()))?;
    let shard: Shard = serde_json::from_slice(&bytes).context("parse shard manifest")?;
    if task_metric(&shard.task).is_none() {
        bail!("unsupported task '{}'", shard.task);
    }
    let outcome = evaluate_one_manifest(&shard, candidate_index)?;
    let wire: EvalOneWire = outcome.into();
    let encoded = serde_json::to_vec(&wire).context("encode eval-one result")?;
    if result.exists() {
        bail!("refusing to replace eval-one result {}", result.display());
    }
    fs::write(result, encoded)
        .with_context(|| format!("write eval-one result {}", result.display()))
}

// ---------------------------------------------------------------------------
// eval mode
// ---------------------------------------------------------------------------
