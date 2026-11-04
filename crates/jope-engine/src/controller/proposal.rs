use super::*;

pub fn propose(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    shuffled: bool,
) -> Result<()> {
    propose_with_generation(
        root,
        corpus,
        checkpoint,
        output,
        crate::controller::ProposalOptions {
            cuts,
            cut_start,
            shuffled,
        },
        None,
    )
}

/// Cut selection and context-shuffle mode for a proposal run.
#[derive(Clone, Copy)]
pub struct ProposalOptions {
    pub cuts: usize,
    pub cut_start: usize,
    pub shuffled: bool,
}

pub fn propose_with_generation(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    options: ProposalOptions,
    generation: Option<&Path>,
) -> Result<()> {
    let ProposalOptions {
        cuts,
        cut_start,
        shuffled,
    } = options;

    crate::task_contract::require_original_gpu_grant("proposal-controller")?;
    crate::campaign::check_deadline()?;
    let task_deadline = crate::task_contract::gpu_task_deadline_ms()?;
    verify_proposal_protocol(output, false)?;
    fs::create_dir_all(output)?;
    if let Some(path) = generation {
        let manifest: serde_json::Value = read_json(path)?;
        if let Some(protocol) = manifest["coverage_selection_protocol"].as_str() {
            ensure!(
                protocol == crate::service::coverage::PROTOCOL,
                "unsupported coverage selection protocol"
            );
            let owner = output
                .parent()
                .context("coverage generation parent absent")?;
            let selection_path = owner.join("coverage-selection.json");
            let index = crate::teaching_store::read_index(root, corpus)?;
            let selection = crate::service::coverage::Ledger::open(root)?.freeze_population(
                root,
                corpus,
                owner,
                &index.contexts,
                cuts,
                crate::telemetry::now(),
            )?;
            if selection_path.exists() {
                ensure!(
                    read_json::<serde_json::Value>(&selection_path)? == selection,
                    "frozen coverage selection changed"
                );
            } else {
                write_json(&selection_path, &selection)?;
            }
            crate::service::coverage::selected_indices(&selection_path, corpus, &index.contexts)?;
        }
    }
    let champions =
        crate::champions::freeze_snapshot(root, corpus, checkpoint, output, cuts, cut_start)?;
    let mut spec = serde_json::json!({"schema":"parallel.proposal_run.v1","corpus_sha256":file_hash(&corpus.join("index.json"))?,"checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"cuts":cuts,"shuffled":shuffled,"world":8,"champion_snapshot_sha256":file_hash(&champions)?});
    spec["deadline_unix_ms"] = serde_json::json!(task_deadline);
    spec["executable_sha256"] = serde_json::json!(file_hash(&std::env::current_exe()?)?);
    let selection_path = output
        .parent()
        .context("proposal parent absent")?
        .join("coverage-selection.json");
    if selection_path.exists() {
        spec["coverage_selection_sha256"] = file_hash(&selection_path)?.into();
        spec["coverage_selection_protocol"] = crate::service::coverage::PROTOCOL.into();
    }
    if let Some(path) = generation {
        let manifest: serde_json::Value = read_json(path)?;
        ensure!(
            manifest["checkpoint_sha256"] == spec["checkpoint_sha256"],
            "generation parent checkpoint changed"
        );
        let digest = file_hash(path)?;
        let bytes = hex::decode(&digest)?;
        spec["proposal_seed"] = u64::from_le_bytes(bytes[..8].try_into().unwrap()).into();
        spec["generation_manifest_sha256"] = digest.into();
    }
    bind_proposal_protocol(&mut spec);
    if cut_start != 0 {
        spec["cut_start"] = cut_start.into();
    }
    #[cfg(feature = "neural")]
    {
        spec["proposal_selection_id"] = crate::proposal_ranking::SELECTION_ID.into();
    }
    if output.join("run.json").exists() {
        ensure!(
            read_json::<serde_json::Value>(&output.join("run.json"))? == spec,
            "proposal resume specification changed"
        );
    } else {
        ensure!(
            fs::read_dir(output)?.next().is_none(),
            "unidentified proposal output exists"
        );
        write_json(&output.join("run.json"), &spec)?;
    }
    let _lease = gpu_lease(root, "neural proposal generation")?;
    let h: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
    ensure!(
        crate::telemetry::now().saturating_sub(h.timestamp) <= 5
            && h.available_gb >= 40.0
            && h.disk_free_gb >= 50.0,
        "proposal admission headroom unavailable"
    );
    let executable = std::env::current_exe()?;
    let mut children = Ranks(Vec::new());
    let begun = Instant::now();
    for rank in 0..8 {
        crate::campaign::check_deadline()?;
        let cpu = if rank < 4 {
            rank * 2
        } else {
            62 + (rank - 4) * 2
        };
        let log = fs::File::create(output.join(format!("rank-{rank}.log")))?;
        let mut cmd = Command::new("taskset");
        cmd.args(["-c", &format!("{cpu},{}", cpu + 1)])
            .env("JOPE_CHAMPION_SNAPSHOT", &champions)
            .arg(&executable)
            .arg("--root")
            .arg(root)
            .args(["propose-rank", "--corpus"])
            .arg(corpus)
            .arg("--checkpoint")
            .arg(checkpoint)
            .arg("--output")
            .arg(output)
            .arg("--rank")
            .arg(rank.to_string())
            .args(["--world", "8", "--cuts"])
            .arg(cuts.to_string())
            .arg("--cut-start")
            .arg(cut_start.to_string())
            .env("OMP_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        if shuffled {
            cmd.arg("--shuffled");
        }
        #[cfg(feature = "neural")]
        crate::gpu_worker::inherit_lease(&mut cmd, &_lease)?;
        cmd.env("JOPE_GPU_TASK_DEADLINE_MS", task_deadline.to_string());
        bind_parent_death(&mut cmd);
        // SAFETY: the hook runs between fork and exec and only calls async-signal-safe setpgid;
        // io::Error::last_os_error reads errno without allocating.
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
    loop {
        crate::campaign::check_deadline()?;
        let mut complete = true;
        let mut ranks = Vec::new();
        for (rank, child) in children.0.iter_mut().enumerate() {
            let receipt_path = output.join(format!("rank-{rank}.json"));
            let mut receipt = read_rank_receipt(&receipt_path)?;
            match child.try_wait()? {
                Some(s) => {
                    ensure!(s.success(), "proposal rank {rank} failed: {s}");
                    if receipt.is_none() {
                        // The child can publish its final receipt between our file read and
                        // wait check; re-read before treating successful exit as corruption.
                        receipt = read_rank_receipt(&receipt_path)?;
                    }
                    ensure!(
                        receipt.is_some(),
                        "proposal rank {rank} exited successfully without a receipt: {}",
                        receipt_path.display()
                    );
                }
                None => {
                    complete = false;
                    let age = match receipt.as_ref().and_then(|r| r["timestamp"].as_u64()) {
                        Some(timestamp) => crate::telemetry::now().saturating_sub(timestamp),
                        None => begun.elapsed().as_secs(),
                    };
                    ensure!(age < 600, "proposal rank {rank} heartbeat timeout");
                }
            }
            ranks.push(receipt.unwrap_or(serde_json::Value::Null));
        }
        write_json(
            &root.join("runs/encoding-service.json"),
            &serde_json::json!({"phase":if complete{"completed"}else{"neural program proposals"},"timestamp":crate::telemetry::now(),"persistent":false,"world":8,"ranks":ranks,"cpu_budget":16,"output":output,"seconds":begun.elapsed().as_secs_f64()}),
        )?;
        if complete {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    let mut combined = crate::service::GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: Vec::new(),
        vocabulary: Some(
            corpus
                .join("vocabulary.json")
                .strip_prefix(root)?
                .to_path_buf(),
        ),
    };
    for rank in 0..8 {
        let jobs: crate::service::GpJobs = read_json(&output.join(format!("jobs-{rank}.json")))?;
        combined.jobs.extend(jobs.jobs);
    }
    combined
        .jobs
        .sort_by(|a, b| a.prepared.cmp(&b.prepared).then(a.cut.cmp(&b.cut)));
    write_json(&output.join("jobs.json"), &combined)?;
    let mut complete = serde_json::json!({"jobs":combined.jobs.len(),"seconds":begun.elapsed().as_secs_f64(),"shuffled_context":shuffled,"checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"world":8,"final_tests_opened":false});
    bind_proposal_protocol(&mut complete);
    write_json(&output.join("complete.json"), &complete)?;
    Ok(())
}

/// Qualification wrappers may select the same immutable rank runtime as production.
pub(super) fn training_executable(root: &Path) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("JOPE_TRAIN_EXECUTABLE") {
        let path = PathBuf::from(path);
        ensure!(path.is_absolute(), "JOPE_TRAIN_EXECUTABLE must be absolute");
        return crate::manifest::resolve(
            root,
            path.strip_prefix(root)
                .context("training executable must be inside workspace")?,
        );
    }
    Ok(std::env::current_exe()?)
}
