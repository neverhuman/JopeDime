        let result_path = output.join(format!("candidate-{i}.json"));
        experiment_failpoint("candidate_intent", experiment.is_some());
        let completion_path = output.join(format!("candidate-{i}.completion.json"));
        let saved_completion: Option<serde_json::Value> =
            if saved_attempt.is_some() && completion_path.exists() {
                Some(read_json(&completion_path)?)
            } else {
                None
            };
        let recovered_metric = saved_completion.is_some();
        let begun = Instant::now();
        let deadline_seconds =
            (seconds as f64 - start.elapsed().as_secs_f64() - previous_seconds).clamp(0.001, 30.);
        let deadline_monotonic_ns =
            crate::fit_boundary::deadline_after(Duration::from_secs_f64(deadline_seconds))?;
        let mut failure = None;
        let mut exit_code = None;
        let mut exit_signal = None;
        let mut exit_observed = false;
        let mut deadline_exceeded = false;
        let mut process_group = serde_json::Value::Null;
        if let Some(completion) = &saved_completion {
            failure = completion["failure"].as_str().map(String::from);
        } else if saved_attempt.is_some() {
            // An intent/result is not an observed successful exit. It might
            // already have fitted; never dispatch this candidate again.
            failure = Some("unknown_original_execution_without_completion".into());
        } else {
            let log = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(output.join(format!("candidate-{i}.log")))?;
            let mut command = evaluator_command(&executable);
            use std::os::fd::AsRawFd;
            command
                .args(["evaluate-memory", "--fd"])
                .arg(memory.as_raw_fd().to_string())
                .arg("--prepared")
                .arg(prepared)
                .arg("--cut")
                .arg(cut.to_string())
                .arg("--genome")
                .arg(&genome_path)
                .arg("--output")
                .arg(&result_path)
                .env(
                    "JOPE_FIT_DEADLINE_MONOTONIC_NS",
                    deadline_monotonic_ns.to_string(),
                )
                .env(
                    "OMP_NUM_THREADS",
                    std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".into()),
                )
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env(
                    "RAYON_NUM_THREADS",
                    std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into()),
                )
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            use std::os::unix::process::CommandExt;
            let controller_pid = std::process::id();
            // W5-T05: the evaluator's guard kills its whole process group when this
            // controller dies, even by SIGKILL (`guard_evaluator_group`).
            command.env(EVALUATOR_GROUP_GUARD, controller_pid.to_string());
            // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls
            // (setpgid, prctl, getppid, setrlimit); its errors are errno codes, so nothing
            // allocates in the child.
            unsafe {
                command.pre_exec(move || {
                    // SIGTERM, not SIGKILL: the guard turns it into SIGKILL for the group;
                    // an evaluator without the guard still ends by its default action.
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The controller exited: refuse with an errno, which does not allocate
                    // (an io::Error carrying a message would box it in the forked child).
                    if libc::getppid() as u32 != controller_pid {
                        return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                    }
                    let limit = libc::rlimit {
                        rlim_cur: 8 * 1024 * 1024 * 1024,
                        rlim_max: 8 * 1024 * 1024 * 1024,
                    };
                    // W5-T05: the evaluator leads its own process group so that
                    // every process it starts is drained with it.
                    if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 || libc::setpgid(0, 0) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let mut child = command.spawn()?;
            let group = i32::try_from(child.id()).context("evaluator pid out of range")?;
            loop {
                let exited = evaluator_exited(group)?;
                let expired = !exited && begun.elapsed().as_secs_f64() >= deadline_seconds;
                if exited || expired {
                    // The leader is still unreaped (a zombie when it exited), so
                    // its group id cannot be reused while the group is drained.
                    process_group = drain_process_group(group);
                    let status = child.wait()?;
                    use std::os::unix::process::ExitStatusExt;
                    exit_observed = true;
                    exit_code = status.code();
                    exit_signal = status.signal();
                    deadline_exceeded = expired || begun.elapsed().as_secs_f64() > deadline_seconds;
                    if deadline_exceeded {
                        failure = Some("candidate_or_view_timeout".into());
                    } else if !status.success() {
                        failure = Some(format!("worker_exit:{status}"));
                    }
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let result: Option<serde_json::Value> =
            if take_clean_result(&mut failure, result_path.exists()) {
                Some(read_json(&result_path)?)
            } else {
                None
            };
        let measured_runtime = if let Some(completion) = &saved_completion {
            completion["runtime_seconds"].as_f64()
        } else if saved_attempt.is_some() {
            None
        } else {
            Some(begun.elapsed().as_secs_f64())
        };
        let completion_reference = if completion_path.exists() {
            Some(relative_artifact(&completion_root, &completion_path)?)
        } else if saved_attempt.is_none() {
            let mut completion = candidate_identity.clone();
            completion["schema"] = json!(crate::label_admission::COMPLETION_SCHEMA);
            completion["attempt"] = relative_artifact(&completion_root, &attempt_path)?;
            completion["exit_observed"] = json!(exit_observed);
            completion["exit_code"] = json!(exit_code);
            completion["signal"] = json!(exit_signal);
            completion["deadline_seconds"] = json!(deadline_seconds);
            completion["deadline_exceeded"] = json!(deadline_exceeded);
            completion["runtime_seconds"] = json!(measured_runtime);
            completion["failure"] = json!(failure);
            completion["process_group"] = process_group;
            let log_path = output.join(format!("candidate-{i}.log"));
            if log_path.exists() {
                fs::File::open(&log_path)?.sync_all()?;
                completion["worker_log"] = relative_artifact(&completion_root, &log_path)?;
            }
            completion["outcome"] = json!(if failure.is_none() {
                "success"
            } else if deadline_exceeded {
                "timeout"
            } else {
                "infrastructure_failure"
            });
            completion["output"] = if result.is_some() {
                relative_artifact(&completion_root, &result_path)?
            } else {
                serde_json::Value::Null
            };
            if let Some(r) = &result {
                completion["fit_completion"] =
                    relative_existing_reference(&completion_root, &r["fit_completion"])?;
                completion["model"] =
                    relative_existing_reference(&completion_root, &r["fitted_model"])?;
            } else {
                let terminal_path = result_path.with_extension("fit-outcome.json");
                if terminal_path.exists() {
                    let terminal: serde_json::Value = read_json(&terminal_path)?;
                    completion["fit_completion"] =
                        relative_existing_reference(&completion_root, &terminal["fit_completion"])?;
                    if !deadline_exceeded && terminal["outcome"] == "deterministic_failure" {
                        completion["outcome"] = json!("fit_failure");
                    }
                    completion["original_fit_outcome"] = terminal["outcome"].clone();
                }
            }
            write_json(&completion_path, &completion)?;
            experiment_failpoint("candidate_completion", experiment.is_some());
            Some(relative_artifact(&completion_root, &completion_path)?)
        } else {
            None
        };
        if recovered_metric {
            let mut recovered = candidate_identity.clone();
            recovered["clean_completion"] = json!(completion_reference);
            recovered["failure"] = json!(failure);
            recovered["metrics"] = json!(result);
            recovered["runtime_seconds"] = json!(measured_runtime);
            let completion = saved_completion
                .as_ref()
                .context("original completion absent")?;
            ensure!(
                completion["schema"] == crate::label_admission::COMPLETION_SCHEMA
                    && completion["attempt"] == relative_artifact(&completion_root, &attempt_path)?,
                "recovered completion attempt changed"
            );
            for key in [
                "evaluation_identity_sha256",
                "program_sha256",
                "evaluator_sha256",
            ] {
                ensure!(
                    completion[key] == candidate_identity[key],
                    "recovered completion identity changed: {key}"
                );
            }
            if completion["fit_completion"].is_object() {
                let native = crate::label_admission::artifact(
                    &completion_root,
                    &completion["fit_completion"],
                )?;
                recovered["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
                crate::label_admission::validate_completion(&completion_root, &recovered)?;
            } else {
                // A denied dispatch has no numerical receipt and cannot pass
                // quality admission. It can still recover its original failure.
                ensure!(
                    result.is_none()
                        && failure.as_ref().is_some_and(|s| !s.is_empty())
                        && (completion["outcome"] == "infrastructure_failure"
                            || completion["outcome"] == "timeout")
                        && completion["exit_observed"] == true
                        && (completion["exit_code"]
                            .as_i64()
                            .is_some_and(|code| code != 0)
                            || completion["signal"]
                                .as_i64()
                                .is_some_and(|signal| signal > 0)
                            || completion["deadline_exceeded"] == true),
                    "recovered failure lacks original non-success process evidence"
                );
                crate::label_admission::artifact_bytes(
                    &completion_root,
                    &completion["worker_log"],
                )?;
            }
        }
