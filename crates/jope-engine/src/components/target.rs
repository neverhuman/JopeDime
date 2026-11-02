use super::*;

#[derive(Deserialize)]
struct Recipe {
    classical: Vec<Vec<usize>>,
    neural: Vec<Vec<usize>>,
}
#[derive(Deserialize)]
struct Sampling {
    window: usize,
    views: usize,
    recipes: BTreeMap<String, Recipe>,
}
struct View {
    job: GpJob,
    p: Prepared,
    dest: PathBuf,
    classical: Vec<f32>,
    neural: Vec<f32>,
    length: f32,
    task: i64,
    min: f64,
    max: f64,
}
fn read_view(
    root: &Path,
    job: &GpJob,
    output: &Path,
    sampling: &Sampling,
    spec_sha: &str,
) -> Result<Option<View>> {
    let path = resolve(root, &job.prepared)?;
    ensure!(
        file_hash(&path.join("prepared.json"))? == job.prepared_sha256,
        "prepared input changed"
    );
    let p: Prepared = read_json(&path.join("prepared.json"))?;
    let cut = p
        .cuts
        .iter()
        .find(|c| c.index == job.cut)
        .context("cut absent")?;
    let dest = output.join(&p.dataset).join(format!("cut-{}", job.cut));
    if dest.join("target.json").exists() {
        let old: serde_json::Value = read_json(&dest.join("target.json"))?;
        ensure!(
            old["training_sha256"] == cut.training_sha256
                && old["spec_sha256"] == spec_sha
                && old["signature_sha256"] == file_hash(&dest.join("target.f32"))?,
            "target cache identity mismatch"
        );
        return Ok(None);
    }
    crate::task_contract::require_task(&p.task)?;
    let task = if p.task == "binary" { 0 } else { 1 };
    let rows = prepare::load_view(&path, job.cut, true)?;
    let y = rows.iter().map(|r| *r.last().unwrap()).collect::<Vec<_>>();
    let min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // The frozen target bundle was trained on unit-interval targets. Its scaler
    // is fitted on this cut's training targets only, before window selection.
    let values = y
        .iter()
        .map(|&v| {
            if task == 0 {
                v as f32
            } else if max > min {
                ((v - min) / (max - min)) as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let recipe = sampling
        .recipes
        .get(&format!("{task}:{}", y.len()))
        .context("frozen sampler lacks this input length; export its index recipe first")?;
    let gather = |indices: &Vec<Vec<usize>>| -> Result<Vec<f32>> {
        ensure!(
            indices.len() == 2 && indices.iter().all(|r| r.len() == 32),
            "invalid window recipe"
        );
        indices
            .iter()
            .flatten()
            .map(|&i| {
                values
                    .get(i)
                    .copied()
                    .context("sample index outside training cut")
            })
            .collect()
    };
    Ok(Some(View {
        job: job.clone(),
        p,
        dest,
        classical: gather(&recipe.classical)?,
        neural: gather(&recipe.neural)?,
        length: y.len() as f32,
        task,
        min,
        max,
    }))
}
pub fn target_rank(
    root: &Path,
    jobs: &Path,
    spec_path: &Path,
    output: &Path,
    rank: usize,
    world: usize,
    batch: usize,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("target-rank")?;
    ensure!(
        rank < world && world <= 8 && batch > 0 && batch <= 512,
        "invalid target worker shape"
    );
    ensure!(tch::Cuda::is_available(), "CUDA required");
    tch::jit::set_graph_executor_optimize(false);
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    let spec: Spec = read_json(spec_path)?;
    spec.verify(root)?;
    let spec_sha = file_hash(spec_path)?;
    let sampling: Sampling = read_json(&resolve(root, &spec.sampling)?)?;
    ensure!(
        sampling.window == 32 && sampling.views == 2,
        "frozen target sampler schema mismatch"
    );
    let mut module = CModule::load_on_device(resolve(root, &spec.module)?, Device::Cuda(0))?;
    module.set_eval();
    let jobs: GpJobs = read_json(jobs)?;
    let jobs = jobs
        .jobs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % world == rank)
        .map(|(_, j)| j)
        .collect::<Vec<_>>();
    let total = jobs.len();
    fs::create_dir_all(output)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let start = Instant::now();
    let mut completed = 0;
    let mut gpu_seconds = 0.0;
    let mut batches = 0;
    for chunk in jobs.chunks(batch) {
        let views = pool.install(|| {
            chunk
                .par_iter()
                .map(|j| read_view(root, j, output, &sampling, &spec_sha))
                .collect::<Result<Vec<_>>>()
        })?;
        let skipped = views.iter().filter(|v| v.is_none()).count();
        let views = views.into_iter().flatten().collect::<Vec<_>>();
        completed += skipped;
        if !views.is_empty() {
            let n = views.len() as i64;
            let device = Device::Cuda(0);
            let classical = views
                .iter()
                .flat_map(|v| v.classical.iter().copied())
                .collect::<Vec<_>>();
            let neural = views
                .iter()
                .flat_map(|v| v.neural.iter().copied())
                .collect::<Vec<_>>();
            let lengths = views.iter().map(|v| v.length).collect::<Vec<_>>();
            let tasks = views.iter().map(|v| v.task).collect::<Vec<_>>();
            let begun = Instant::now();
            let result = tch::no_grad(|| {
                module.forward_ts(&[
                    Tensor::from_slice(&classical)
                        .view([n, 2, 32])
                        .to_device(device),
                    Tensor::from_slice(&neural)
                        .view([n, 2, 32])
                        .to_device(device),
                    Tensor::from_slice(&lengths).to_device(device),
                    Tensor::from_slice(&tasks).to_device(device),
                ])
            })?
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
            ensure!(
                result.size() == [n, 512] && result.isfinite().all().int64_value(&[]) != 0,
                "invalid target encoder output"
            );
            let values: Vec<f32> = Vec::try_from(result.flatten(0, -1))?;
            gpu_seconds += begun.elapsed().as_secs_f64();
            batches += 1;
            for (v, encoded) in views.iter().zip(values.chunks_exact(512)) {
                fs::create_dir_all(&v.dest)?;
                let bytes = encoded
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                let cut = v.p.cuts.iter().find(|c| c.index == v.job.cut).unwrap();
                atomic_write(&v.dest.join("target.f32"), &bytes)?;
                write_json(
                    &v.dest.join("target.json"),
                    &json!({"schema":"parallel.target_cut.v1","dataset":v.p.dataset,"family":v.p.family,"cut":v.job.cut,"prepared_sha256":v.job.prepared_sha256,"training_sha256":cut.training_sha256,"preprocessing_sha256":cut.preprocessing_sha256,"spec_sha256":spec_sha,"shape":[512],"signature_sha256":hash(&bytes),"target_min":v.min,"target_max":v.max,"target_transform":"training-cut minmax; constant targets map to zero","backend":"Rust/LibTorch 2.7 CUDA FP32","final_test_targets_used":false,"copied_dataset_bytes":0}),
                )?;
                completed += 1;
            }
        }
        write_json(
            &output.join(format!("rank-{rank}.json")),
            &json!({"rank":rank,"world":world,"phase":"frozen target encoding","completed":completed,"total":total,"gpu_batches":batches,"gpu_seconds":gpu_seconds,"last_batch":views.len(),"cuts_per_second":completed as f64/start.elapsed().as_secs_f64(),"timestamp":crate::telemetry::now(),"elapsed_seconds":start.elapsed().as_secs_f64(),"failures":[]}),
        )?;
    }
    let receipt = json!({"rank":rank,"world":world,"phase":"completed","completed":completed,"total":total,"gpu_batches":batches,"gpu_seconds":gpu_seconds,"timestamp":crate::telemetry::now(),"seconds":start.elapsed().as_secs_f64(),"failures":[]});
    write_json(&output.join(format!("rank-{rank}.json")), &receipt)?;
    write_json(&output.join(format!("complete-{rank}.json")), &receipt)
}
pub fn service(
    root: &Path,
    jobs: &Path,
    spec: &Path,
    output: &Path,
    batch: usize,
    component: &str,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("component-service")?;
    ensure!(
        matches!(component, "target" | "foundation"),
        "unknown encoder component"
    );
    fs::create_dir_all(output)?;
    let _lease = crate::controller::gpu_lease(root, &format!("frozen {component} encoding"))?;
    let hardware: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
    ensure!(
        crate::telemetry::now().saturating_sub(hardware.timestamp) <= 5
            && hardware.available_gb > 128.0
            && hardware.disk_free_gb > 50.0,
        "component admission paused: stale telemetry or insufficient RAM/disk"
    );
    let started = crate::telemetry::now();
    let total = read_json::<GpJobs>(jobs)?.jobs.len();
    let mut children = Vec::new();
    let exe = std::env::current_exe()?;
    let result = (|| -> Result<()> {
        for rank in 0..8 {
            let cpu = if rank < 4 {
                rank * 2
            } else {
                62 + (rank - 4) * 2
            };
            let log = fs::File::create(output.join(format!("worker-{rank}.log")))?;
            let mut cmd = Command::new("taskset");
            cmd.args(["-c", &format!("{cpu}-{}", cpu + 1)])
                .arg(&exe)
                .arg("--root")
                .arg(root)
                .arg(format!("{component}-rank"))
                .arg("--jobs")
                .arg(jobs)
                .arg("--spec")
                .arg(spec)
                .arg("--output")
                .arg(output)
                .arg("--rank")
                .arg(rank.to_string())
                .args(["--world", "8", "--batch"])
                .arg(batch.to_string())
                .env("CUDA_VISIBLE_DEVICES", rank.to_string())
                .env("OMP_NUM_THREADS", "2")
                .env("OPENBLAS_NUM_THREADS", "1")
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            use std::os::unix::process::CommandExt;
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
            children.push(cmd.spawn()?);
        }
        loop {
            let mut complete = true;
            let mut ranks = Vec::new();
            for (rank, child) in children.iter_mut().enumerate() {
                let status = child.try_wait()?;
                if let Some(status) = status {
                    ensure!(status.success(), "{component} rank {rank} failed");
                } else {
                    complete = false;
                }
                let r = read_json::<serde_json::Value>(&output.join(format!("rank-{rank}.json")))
                    .unwrap_or_default();
                let timestamp = r["timestamp"].as_u64().filter(|t| *t >= started);
                if status.is_none() {
                    ensure!(
                        crate::telemetry::now().saturating_sub(timestamp.unwrap_or(started)) < 600,
                        "encoder rank {rank} stalled for 600 seconds"
                    );
                } else {
                    ensure!(
                        timestamp.is_some(),
                        "{component} rank {rank} exited without a current completion receipt"
                    );
                }
                if timestamp.is_some() {
                    ranks.push(r);
                } else {
                    ranks.push(json!({"rank":rank,"phase":"initializing frozen CUDA model","completed":0,"total":(total+7-rank)/8,"timestamp":started}));
                }
            }
            write_json(
                &root.join("runs/encoding-service.json"),
                &json!({"phase":if complete{"completed"}else{"frozen component CUDA encoding"},"component":component,"world":8,"persistent":false,"cpu_budget":16,"completed":ranks.iter().map(|v|v["completed"].as_u64().unwrap_or(0)).sum::<u64>(),"total":total,"ranks":ranks,"timestamp":crate::telemetry::now()}),
            )?;
            if complete {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        Ok(())
    })();
    if result.is_err() {
        for child in &mut children {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; it targets
            // the process group each rank created with setpgid(0, 0); the result is ignored.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
    result
}
