use super::proposal::training_executable;
use super::training_rank::{training_step_window, verify_training_rank};
use super::*;

pub(crate) fn train_controlled(
    root: &Path,
    corpus: &Path,
    output: &Path,
    options: TrainingOptions<'_>,
    stop: Option<&AtomicBool>,
    policy: Option<&Path>,
) -> Result<()> {
    let TrainingOptions {
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        fail_rank,
        initialize,
    } = options;

    // Input validation below is read-only: it creates no GPU context, lease,
    // lock, output directory or receipt. It precedes the dispatch hold so an
    // invalid request is refused for its own reason, and the hold still
    // refuses every valid request before any resource admission.
    ensure!(
        [1, 2, 4, 8].contains(&world)
            && [64, 128, 256].contains(&micro)
            && 2048 % (world * micro) == 0
            && steps > 0
            && checkpoint_steps > 0,
        "unsupported training topology"
    );
    let policy = policy
        .map(|p| -> Result<PathBuf> {
            resolve(
                root,
                if p.is_absolute() {
                    p.strip_prefix(root)?
                } else {
                    p
                },
            )
        })
        .transpose()?;
    let policy_sha = policy.as_deref().map(file_hash).transpose()?;
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    if let Some(path) = &policy {
        let p: serde_json::Value = read_json(path)?;
        let index: serde_json::Value = read_json(&corpus.join("index.json"))?;
        if p["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA {
            crate::checkpoint_recovery::verify_continuation(
                root,
                &p,
                &file_hash(&corpus.join("index.json"))?,
                index["vocabulary_sha256"]
                    .as_str()
                    .context("continuation vocabulary absent")?,
                crate::checkpoint_recovery::TrainingTopology {
                    world,
                    micro,
                    global: 2048,
                },
                steps,
            )?;
            if !resume {
                let parent =
                    initialize.context("continuation requires its original parent checkpoint")?;
                let parent = resolve(
                    root,
                    if parent.is_absolute() {
                        parent.strip_prefix(root)?
                    } else {
                        parent
                    },
                )?;
                ensure!(
                    file_hash(&parent.join("state.json"))? == p["parent_checkpoint_sha256"],
                    "continuation initialization differs from recovered parent"
                );
            }
        }
        if let Some(p) = crate::checkpoint_recovery::continuation_replay(root, &p)? {
            ensure!(
                p["schema"] == "parallel.replay_policy.v2"
                    && p["eligible"] == true
                    && p["final_tests_opened"] == false
                    && p["corpus_sha256"] == file_hash(&corpus.join("index.json"))?
                    && index["label_policy"] == "strict_midrank_v4"
                    && index["admission_policy"].is_object(),
                "training policy requires a matching strictly admitted corpus"
            );
        }
    }
    ensure!(
        resume || !output.exists() || fs::read_dir(output)?.next().is_none(),
        "run exists; use explicit resume"
    );
    let resume = resume && output.join("run.json").is_file();
    ensure!(
        resume || !output.exists() || fs::read_dir(output)?.next().is_none(),
        "nonempty run has no launch identity"
    );
    let step_window = if resume {
        let previous: serde_json::Value = read_json(&output.join("run.json"))?;
        let window = previous["step_window"].clone();
        let initial = window["initial_optimizer_step"]
            .as_u64()
            .context("legacy run has no increment identity; explicit reconciliation required")?;
        ensure!(
            window == training_step_window(initial, steps)?,
            "resume requested increment changed"
        );
        ensure!(!output.join("complete.json").exists(),
            "training run already completed; retain original completion, use a new funded task for new steps");
        window
    } else {
        let carry = policy
            .as_ref()
            .map(|p| read_json::<serde_json::Value>(p))
            .transpose()?
            .is_some_and(|p| p["optimizer"] == "carry");
        let initial = if carry {
            let checkpoint =
                initialize.context("optimizer carry requires an explicit parent checkpoint")?;
            let checkpoint = if checkpoint.is_absolute() {
                checkpoint.to_path_buf()
            } else {
                root.join(checkpoint)
            };
            read_json::<serde_json::Value>(&checkpoint.join("state.json"))?["step"]
                .as_u64()
                .context("parent optimizer step absent")?
        } else {
            0
        };
        training_step_window(initial, steps)?
    };
    crate::task_contract::require_original_gpu_grant("training-controller")?;
    check_cancel(stop)?;
    let _lease = gpu_lease_controlled(root, "neural policy training", stop)?;
    let h: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
    ensure!(
        crate::telemetry::now().saturating_sub(h.timestamp) <= 5
            && h.available_gb >= 40.0
            && h.disk_free_gb >= 50.0,
        "training admission requires fresh telemetry, 40 GiB RAM and 50 GiB disk headroom"
    );
    let executable = training_executable(root)?;
    fs::create_dir_all(output)?;
    let corpus_hash = file_hash(&corpus.join("index.json"))?;
    let spec = serde_json::json!({"schema":"parallel.training_run.v1","corpus_sha256":corpus_hash,"world":world,"microbatch":micro,"global_batch":2048,"qualification":true,"executable_sha256":file_hash(&executable)?,"source_families_sampled_evenly":true,"evaluation_bias":"adaptive_cut_heldout_teacher"});
    let initialization_sha = initialize
        .map(|p| file_hash(&p.join("state.json")))
        .transpose()?;
    let mut spec = spec;
    let inherited_deadline = crate::task_contract::gpu_task_deadline_ms()?;
    let task_deadline = if resume {
        read_json::<serde_json::Value>(&output.join("run.json"))?["deadline_unix_ms"]
            .as_u64()
            .context("legacy training run lacks its original absolute deadline")?
    } else {
        inherited_deadline
    };
    ensure!(
        task_deadline == inherited_deadline,
        "resume task deadline differs from its original run"
    );
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    ensure!(
        task_deadline as u128 > now_ms,
        "original training task deadline elapsed"
    );
    spec["deadline_unix_ms"] = serde_json::json!(task_deadline);
    spec["execution_partition"] = crate::task_contract::training_partition(world, micro, 2048)?;
    #[cfg(feature = "neural")]
    {
        spec["sampler_id"] = crate::training::SAMPLER_ID.into();
    }
    spec["initialization_sha256"] = serde_json::json!(initialization_sha);
    spec["replay_policy_sha256"] = serde_json::json!(policy_sha);
    spec["training_policy_schema"] = policy
        .as_ref()
        .map(|path| read_json::<serde_json::Value>(path).map(|value| value["schema"].clone()))
        .transpose()?
        .unwrap_or(serde_json::Value::Null);
    spec["step_window"] = step_window.clone();
    // W6 follow-up: the controller declares the run's lineage mode in
    // run.json and on every rank request; each rank validates it. A resumed
    // run whose run.json predates the field keeps its implicit meaning.
    let lineage_mode = if resume {
        let old: serde_json::Value = read_json(&output.join("run.json"))?;
        ensure!(
            old["corpus_sha256"] == corpus_hash
                && old["world"] == world
                && old["microbatch"] == micro
                && old["execution_partition"] == spec["execution_partition"]
                && old["sampler_id"] == spec["sampler_id"]
                && old["executable_sha256"] == spec["executable_sha256"]
                && old["replay_policy_sha256"] == spec["replay_policy_sha256"]
                && (initialize.is_none()
                    || old["initialization_sha256"] == serde_json::json!(initialization_sha)),
            "resume run identity mismatch"
        );
        old.get("continuation_mode")
            .filter(|mode| !mode.is_null())
            .map(|mode| {
                crate::task_contract::ContinuationMode::parse(
                    mode.as_str().context("recorded continuation mode")?,
                )
            })
            .transpose()?
    } else {
        let mode = crate::task_contract::classify_continuation(
            false,
            initialize.is_some(),
            policy
                .as_ref()
                .map(|path| read_json::<serde_json::Value>(path))
                .transpose()?
                .as_ref(),
        )?;
        spec["continuation_mode"] = serde_json::json!(mode.as_str());
        spec["continuation_mode_declaration"] = serde_json::json!("controller");
        write_json(&output.join("run.json"), &spec)?;
        Some(mode)
    };
    let mut injected = false;
    let mut last_error = String::new();
    for attempt in 0..3 {
        check_cancel(stop)?;
        if std::env::var_os("JOPE_GPU_POOL").is_some() {
            ensure!(
                !output
                    .join("resident-reconciliation-required.json")
                    .try_exists()?,
                "resident run requires explicit receipt/process reconciliation"
            );
            // A controller crash can occur before it writes the failure marker.
            // Changing the launch directory on resume must not bypass custody.
            for entry in fs::read_dir(output)? {
                let entry = entry?;
                if entry.file_type()?.is_dir()
                    && entry.file_name().to_string_lossy().starts_with("launch-")
                {
                    let prior:serde_json::Value=read_json(&entry.path().join("controller.json"))
                        .context("resident launch has no committed controller result; reconciliation required")?;
                    ensure!(
                        prior["passed"] == true,
                        "resident launch failed; reconciliation required"
                    );
                }
            }
        }
        ensure!(
            policy.as_deref().map(file_hash).transpose()? == policy_sha,
            "training policy changed during recovery"
        );
        crate::checkpoint_recovery::reconcile(root, output)?;
        let launch = output.join(format!(
            "launch-{}-{}-{attempt}",
            crate::telemetry::now(),
            std::process::id()
        ));
        fs::create_dir_all(&launch)?;
        let begun = Instant::now();
        let mut children = Ranks(Vec::new());
        let should_resume = output.join("latest.json").exists();
        if should_resume {
            // Every rank consumes this launch's frozen selection. Checkpoint
            // publication may advance latest.json without changing its input.
            let selected: serde_json::Value = read_json(&output.join("latest.json"))?;
            write_json(&launch.join("resume-input.json"), &selected)?;
        }
        for rank in 0..world {
            check_cancel(stop)?;
            let cpu = if rank < 4 {
                rank * 2
            } else {
                62 + (rank - 4) * 2
            };
            let log = fs::File::create(launch.join(format!("rank-{rank}.log")))?;
            let mut cmd = Command::new("taskset");
            cmd.args(["-c", &format!("{cpu},{}", cpu + 1)])
                .arg(&executable)
                .arg("--root")
                .arg(root)
                .args(["train-rank", "--corpus"])
                .arg(corpus)
                .arg("--output")
                .arg(output)
                .arg("--launch")
                .arg(&launch)
                .arg("--rank")
                .arg(rank.to_string())
                .arg("--world")
                .arg(world.to_string())
                .arg("--microbatch")
                .arg(micro.to_string())
                .arg("--steps")
                .arg(steps.to_string())
                .arg("--checkpoint-steps")
                .arg(checkpoint_steps.to_string())
                .env("OMP_NUM_THREADS", "2")
                .env("RAYON_NUM_THREADS", "2")
                .env("MKL_NUM_THREADS", "1")
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("JOPE_GPU_TASK_DEADLINE_MS", task_deadline.to_string())
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            if should_resume {
                cmd.arg("--resume");
            }
            if let Some(policy) = &policy {
                cmd.arg("--replay-policy").arg(policy);
            }
            if let Some(mode) = lineage_mode {
                cmd.arg("--continuation-mode").arg(mode.as_str());
            }
            #[cfg(feature = "neural")]
            crate::gpu_worker::inherit_lease(&mut cmd, &_lease)?;
            if !should_resume {
                if let Some(p) = initialize {
                    let relative = if p.is_absolute() {
                        p.strip_prefix(root)?
                    } else {
                        p
                    };
                    cmd.arg("--initialize").arg(relative);
                }
            }
            bind_parent_death(&mut cmd);
            // SAFETY: the hook runs between fork and exec and only calls async-signal-safe
            // setpgid; io::Error::last_os_error reads errno without allocating.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setpgid(0, 0) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            children.0.push(cmd.spawn()?);
        }
        let result = (|| -> Result<()> {
            loop {
                check_cancel(stop)?;
                ensure!(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_millis()
                        < task_deadline as u128,
                    "training task absolute deadline elapsed"
                );
                let mut complete = true;
                for (rank, child) in children.0.iter_mut().enumerate() {
                    let receipt_path = launch.join(format!("rank-{rank}.json"));
                    let mut receipt = read_training_rank_progress(&receipt_path)?;
                    if !injected
                        && fail_rank == Some(rank)
                        && output.join("latest.json").exists()
                        && receipt
                            .as_ref()
                            .is_some_and(|progress| progress.step >= checkpoint_steps as u64)
                    {
                        child.kill()?;
                        injected = true;
                    }
                    match child.try_wait()? {
                        Some(status) => {
                            ensure!(status.success(), "rank {rank} failed: {status}");
                            if receipt.is_none() {
                                // The final receipt may have appeared between our read and wait.
                                receipt = read_training_rank_progress(&receipt_path)?;
                            }
                            ensure!(
                                receipt.is_some(),
                                "rank {rank} exited successfully without a receipt: {}",
                                receipt_path.display()
                            );
                        }
                        None => {
                            complete = false;
                            let age = match receipt.as_ref() {
                                Some(progress) => {
                                    crate::telemetry::now().saturating_sub(progress.timestamp)
                                }
                                None => begun.elapsed().as_secs(),
                            };
                            ensure!(age < 600, "rank {rank} heartbeat timed out");
                        }
                    }
                }
                if complete {
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            let mut hashes = std::collections::BTreeSet::new();
            for rank in 0..world {
                let r: serde_json::Value =
                    read_json(&launch.join(format!("complete-{rank}.json")))?;
                verify_training_rank(&r, &step_window, rank, world, &corpus_hash)?;
                hashes.insert(r["parameter_sha256"].as_str().unwrap_or("").to_string());
            }
            ensure!(
                hashes.len() == 1 && !hashes.contains(""),
                "trained parameters differ across ranks"
            );
            Ok(())
        })();
        drop(children);
        write_json(
            &launch.join("controller.json"),
            &serde_json::json!({"attempt":attempt,"passed":result.is_ok(),"failure":result.as_ref().err().map(|e|format!("{e:#}")),"rank_failure_injected":injected,"resumed":should_resume,"seconds":begun.elapsed().as_secs_f64()}),
        )?;
        match result {
            Ok(()) => {
                write_json(
                    &output.join("complete.json"),
                    &serde_json::json!({"schema":"parallel.training_complete.v2","steps":step_window["target_optimizer_step"],"step_window":step_window,"committed_new_steps":steps,"world":world,"microbatch":micro,"attempts":attempt+1,"rank_failure_injected":injected,"corpus_sha256":corpus_hash,"latest":read_json::<serde_json::Value>(&output.join("latest.json"))?,"qualification":true,"final_tests_opened":false}),
                )?;
                let mut state: serde_json::Value =
                    read_json(&root.join("runs/neural-service.json"))?;
                state["phase"] = "completed real-corpus training block".into();
                state["timestamp"] = crate::telemetry::now().into();
                write_json(&root.join("runs/neural-service.json"), &state)?;
                return Ok(());
            }
            Err(e) => {
                // A new launch path must not turn uncertain resident acceptance
                // into a fresh task. Preserve the original launch for explicit
                // receipt/process reconciliation, including injected failures.
                if std::env::var_os("JOPE_GPU_POOL").is_some() {
                    write_json(
                        &output.join("resident-reconciliation-required.json"),
                        &serde_json::json!({"schema":"parallel.resident_reconciliation.v1",
                            "launch":launch,"attempt":attempt,"failure":format!("{e:#}"),
                            "automatic_retry_permitted":false}),
                    )?;
                    return Err(e.context(
                        "resident task acceptance requires reconciliation; automatic retry refused",
                    ));
                }
                last_error = format!("{e:#}");
            }
        }
    }
    anyhow::bail!("training exhausted three coordinated attempts: {last_error}")
}
