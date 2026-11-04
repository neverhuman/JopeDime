use super::*;

pub(super) fn spawn_workers(
    release: &Path,
    registry: &Path,
    root: &Path,
    run_id: &str,
    train_path: &Path,
    test_features_path: &Path,
    cpu_ids: &[usize],
) -> Result<Vec<WorkerProcess>> {
    let mut workers = Vec::with_capacity(cpu_ids.len());
    let release_sha256 = sha256_file(release)?;
    for (id, &cpu) in cpu_ids.iter().enumerate() {
        let bootstrap = WorkerBootstrap {
            schema: "jope-dime.golden42-v3-worker-bootstrap.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: DATASET.to_string(),
            worker_id: id,
            cpu,
            release_sha256: release_sha256.clone(),
            train_path: train_path.to_path_buf(),
            test_features_path: test_features_path.to_path_buf(),
            output_root: root
                .join("worker-artifacts")
                .join(format!("worker-{id:03}")),
            runtime_root: root.join("worker-runtime").join(format!("worker-{id:03}")),
        };
        let bootstrap_path = root
            .join("workers")
            .join(format!("worker-{id:03}.bootstrap.json"));
        atomic_immutable_write(&bootstrap_path, &json_bytes(&bootstrap)?)?;
        let mut command = Command::new("setsid");
        command
            .arg("taskset")
            .args(["-c", &cpu.to_string()])
            .arg(release)
            .args(["--registry", &registry.display().to_string()])
            .arg("golden42-v3-persistent-worker")
            .args(["--bootstrap", &bootstrap_path.display().to_string()])
            .env("RAYON_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("VECLIB_MAXIMUM_THREADS", "1")
            .env("NUMEXPR_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(
                OpenOptions::new().create(true).append(true).open(
                    root.join("workers")
                        .join(format!("worker-{id:03}.stderr.log")),
                )?,
            ));
        let parent_pid = std::process::id() as libc::pid_t;
        // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls
        // (prctl, getppid); its errors are errno codes, so nothing allocates in the child.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent_pid {
                    // The parent exited: refuse with an errno, which does not allocate
                    // (an io::Error carrying a message would box it in the forked child).
                    return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                }
                Ok(())
            });
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn V3 worker {id} on CPU {cpu}"))?;
        let stdin = BufWriter::new(child.stdin.take().context("worker stdin unavailable")?);
        workers.push(WorkerProcess {
            id,
            cpu,
            child,
            stdin,
        });
    }
    thread::sleep(Duration::from_millis(250));
    for worker in &mut workers {
        if let Some(status) = worker.child.try_wait()? {
            bail!(
                "V3 worker {} exited during startup with {status}",
                worker.id
            );
        }
        let allowed = fs::read_to_string(format!("/proc/{}/status", worker.child.id()))?;
        if !allowed
            .lines()
            .find(|line| line.starts_with("Cpus_allowed_list:"))
            .is_some_and(|line| {
                line.split(':')
                    .nth(1)
                    .is_some_and(|value| value.trim() == worker.cpu.to_string())
            })
        {
            bail!(
                "V3 worker {} is not pinned exclusively to CPU {}",
                worker.id,
                worker.cpu
            );
        }
    }
    Ok(workers)
}

pub(super) type WorkerChannel = (
    Sender<(usize, Result<WorkerResponse>)>,
    Receiver<(usize, Result<WorkerResponse>)>,
);

pub(super) fn worker_channel(workers: &mut [WorkerProcess]) -> Result<WorkerChannel> {
    let (sender, receiver) = mpsc::channel();
    for worker in workers {
        let stdout = worker
            .child
            .stdout
            .take()
            .context("worker stdout unavailable")?;
        let sender = sender.clone();
        let id = worker.id;
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let response = line.context("read V3 worker response").and_then(|line| {
                    serde_json::from_str(&line).context("decode V3 worker response")
                });
                if sender.send((id, response)).is_err() {
                    break;
                }
            }
        });
    }
    Ok((sender, receiver))
}

pub(crate) fn persistent_worker(bootstrap_path: &Path) -> Result<()> {
    let bootstrap: WorkerBootstrap = serde_json::from_slice(&fs::read(bootstrap_path)?)?;
    if bootstrap.schema != "jope-dime.golden42-v3-worker-bootstrap.v1"
        || bootstrap.dataset_hash != DATASET
        || bootstrap.release_sha256 != sha256_file(&std::env::current_exe()?)?
    {
        bail!("invalid V3 worker bootstrap");
    }
    fs::create_dir_all(&bootstrap.output_root)?;
    fs::create_dir_all(&bootstrap.runtime_root)?;
    let train = read_regression_csv(&bootstrap.train_path)?;
    let training = campaign_training(&train.x)?;
    let test_x: Vec<Vec<f64>> = serde_json::from_slice(&fs::read(&bootstrap.test_features_path)?)?;
    let encoded = EncodedTable::from_numeric_rows(&test_x)?;
    let runtime = FittedCppRuntime::new(&bootstrap.runtime_root)?;
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: WorkerRequest = serde_json::from_str(&line)?;
        let response =
            evaluate_worker_request(&bootstrap, &training, &train.y, &encoded, &runtime, request)?;
        // Serialize first and write each response line in one call, then flush it.
        let mut line = serde_json::to_vec(&response)?;
        line.push(b'\n');
        stdout.write_all(&line)?;
        stdout.flush()?;
    }
    Ok(())
}

pub(super) fn evaluate_worker_request(
    bootstrap: &WorkerBootstrap,
    training: &CampaignTraining,
    train_y: &[f64],
    encoded: &EncodedTable,
    runtime: &FittedCppRuntime,
    request: WorkerRequest,
) -> Result<WorkerResponse> {
    if request.schema != "jope-dime.golden42-v3-worker-request.v1"
        || request.context_sha256 != context_hash(&bootstrap.release_sha256, &request.candidate)
    {
        bail!("invalid V3 worker request identity");
    }
    let started_at_unix_ms = unix_ms();
    let started = Instant::now();
    let genome = GenomeIr::parse(&request.candidate.syntax)?;
    let model = fit_campaign_candidate(&genome, training, train_y)?;
    let fit_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let codec = encode_fitted_cpp17(&model)?;
    let codec_ns = started.elapsed().as_nanos();
    let codec_file = bootstrap.output_root.join(format!(
        "generation-{:06}-slot-{:03}-{}.codec",
        request.generation,
        request.slot,
        &request.candidate.semantic_hash[..16]
    ));
    atomic_immutable_write(&codec_file, &codec.bytes)?;
    let started = Instant::now();
    let cpp_model = runtime.load(&codec.bytes)?;
    let cpp_load_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let predictions = cpp_model.predict(encoded)?;
    let cpp_predict_ns = started.elapsed().as_nanos();
    drop(cpp_model);
    let prediction_sha256 = prediction_hash(&predictions);
    Ok(WorkerResponse {
        schema: "jope-dime.golden42-v3-worker-response.v1".to_string(),
        worker_id: bootstrap.worker_id,
        generation: request.generation,
        slot: request.slot,
        semantic_hash: request.candidate.semantic_hash,
        context_sha256: request.context_sha256,
        predictions,
        prediction_sha256,
        codec_sha256: codec.sha256,
        codec_file,
        fit_ns,
        codec_ns,
        cpp_load_ns,
        cpp_predict_ns,
        started_at_unix_ms,
        finished_at_unix_ms: unix_ms(),
    })
}
