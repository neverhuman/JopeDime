//! Frozen component workers. Every cut is reconstructed in memory, encoded once,
//! and committed with a complete input/model receipt before entering DL training.
use crate::{
    manifest::*,
    prepare::{self, Prepared},
    service::{GpJob, GpJobs},
};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use tch::{CModule, Device, Kind, Tensor};
#[derive(Serialize, Deserialize)]
pub struct Spec {
    pub module: PathBuf,
    pub module_sha256: String,
    pub sampling: PathBuf,
    pub sampling_sha256: String,
    pub parity: PathBuf,
    pub parity_sha256: String,
}
impl Spec {
    pub fn verify(&self, root: &Path) -> Result<()> {
        for (p, sha) in [
            (&self.module, &self.module_sha256),
            (&self.sampling, &self.sampling_sha256),
            (&self.parity, &self.parity_sha256),
        ] {
            ensure!(
                file_hash(&resolve(root, p)?)? == *sha,
                "frozen target dependency changed"
            );
        }
        let parity: serde_json::Value = read_json(&resolve(root, &self.parity)?)?;
        ensure!(
            parity["passed"] == true
                && parity["module_sha256"] == self.module_sha256
                && parity["sampling_sha256"] == self.sampling_sha256,
            "target parity identity mismatch"
        );
        Ok(())
    }
}
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
    ensure!(
        matches!(p.task.as_str(), "regression" | "binary"),
        "frozen target encoder supports regression/binary only"
    );
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
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
    result
}

#[derive(Deserialize)]
struct FoundationRecipe {
    context: Vec<usize>,
    query: Vec<usize>,
}
#[derive(Deserialize)]
struct FoundationSampling {
    recipes: BTreeMap<String, FoundationRecipe>,
}
struct FoundationView {
    job: GpJob,
    p: Prepared,
    dest: PathBuf,
    x: Vec<f32>,
    y: Vec<f32>,
    shape: (usize, usize, usize),
}
fn foundation_view(
    root: &Path,
    job: &GpJob,
    output: &Path,
    sampling: &FoundationSampling,
    spec_sha: &str,
) -> Result<Option<FoundationView>> {
    let path = resolve(root, &job.prepared)?;
    ensure!(
        file_hash(&path.join("prepared.json"))? == job.prepared_sha256,
        "prepared identity changed"
    );
    let p: Prepared = read_json(&path.join("prepared.json"))?;
    let cut = p
        .cuts
        .iter()
        .find(|c| c.index == job.cut)
        .context("unknown cut")?;
    let dest = output.join(&p.dataset).join(format!("cut-{}", job.cut));
    if dest.join("foundation.json").exists() {
        let old: serde_json::Value = read_json(&dest.join("foundation.json"))?;
        ensure!(
            old["training_sha256"] == cut.training_sha256
                && old["spec_sha256"] == spec_sha
                && old["signature_sha256"] == file_hash(&dest.join("foundation_raw.f32"))?,
            "foundation cache identity mismatch"
        );
        return Ok(None);
    }
    let rows = prepare::load_view(&path, job.cut, true)?;
    let width = rows[0].len() - 1;
    let recipe = sampling
        .recipes
        .get(&rows.len().to_string())
        .context("foundation sampler lacks cut length")?;
    let mut low = vec![f64::INFINITY; width];
    let mut high = vec![f64::NEG_INFINITY; width];
    for row in &rows {
        for j in 0..width {
            low[j] = low[j].min(row[j]);
            high[j] = high[j].max(row[j]);
        }
    }
    let indices = recipe
        .context
        .iter()
        .chain(&recipe.query)
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        indices.iter().all(|&i| i < rows.len()),
        "foundation sample index outside training cut"
    );
    let x = indices
        .iter()
        .flat_map(|&i| {
            (0..width)
                .map(|j| {
                    let span = high[j] - low[j];
                    ((rows[i][j] - low[j]) / if span > 1e-12 { span } else { 1.0 }) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let y = recipe
        .context
        .iter()
        .map(|&i| rows[i][width] as f32)
        .collect::<Vec<_>>();
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
    let sd =
        (y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt() + 1e-8;
    let y = y.iter().map(|&v| ((v as f64 - mean) / sd) as f32).collect();
    Ok(Some(FoundationView {
        job: job.clone(),
        p,
        dest,
        x,
        y,
        shape: (recipe.context.len(), recipe.query.len(), width),
    }))
}
pub fn foundation_rank(
    root: &Path,
    jobs_path: &Path,
    spec_path: &Path,
    output: &Path,
    rank: usize,
    world: usize,
    batch: usize,
) -> Result<()> {
    ensure!(
        rank < world && world <= 8 && batch > 0 && batch <= 2,
        "foundation requires qualified batch 1 or 2"
    );
    ensure!(tch::Cuda::is_available(), "CUDA required");
    tch::jit::set_graph_executor_optimize(false);
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    let spec: Spec = read_json(spec_path)?;
    spec.verify(root)?;
    let spec_sha = file_hash(spec_path)?;
    let sampling: FoundationSampling = read_json(&resolve(root, &spec.sampling)?)?;
    let parity: serde_json::Value = read_json(&resolve(root, &spec.parity)?)?;
    let mut module = CModule::load_on_device(resolve(root, &spec.module)?, Device::Cuda(0))?;
    module.set_eval();
    let jobs: GpJobs = read_json(jobs_path)?;
    let jobs = jobs
        .jobs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % world == rank)
        .map(|(_, j)| j)
        .collect::<Vec<_>>();
    let total = jobs.len();
    fs::create_dir_all(output)?;
    let start = Instant::now();
    let mut completed = 0;
    let mut batches = 0;
    let mut gpu_seconds = 0.0;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    for window in jobs.chunks(32) {
        let loaded = pool.install(|| {
            window
                .par_iter()
                .map(|j| foundation_view(root, j, output, &sampling, &spec_sha))
                .collect::<Result<Vec<_>>>()
        })?;
        completed += loaded.iter().filter(|v| v.is_none()).count();
        let mut shapes: BTreeMap<(usize, usize, usize), Vec<FoundationView>> = BTreeMap::new();
        for v in loaded.into_iter().flatten() {
            shapes.entry(v.shape).or_default().push(v);
        }
        for (shape, views) in shapes {
            let (nc, nq, width) = shape;
            ensure!(
                parity["tests"]
                    .as_array()
                    .context("foundation parity cases missing")?
                    .iter()
                    .any(|r| r["context"] == nc
                        && r["query"] == nq
                        && r["width"] == width
                        && r["batch"] == batch
                        && r["passed"] == true),
                "foundation input shape lacks numerical qualification"
            );
            for group in views.chunks(batch) {
                let n = group.len();
                let mut x = Vec::with_capacity((nc + nq) * n * width);
                let mut y = Vec::with_capacity(nc * n);
                for row in 0..nc + nq {
                    for v in group {
                        x.extend_from_slice(&v.x[row * width..(row + 1) * width]);
                    }
                }
                for row in 0..nc {
                    for v in group {
                        y.push(v.y[row]);
                    }
                }
                let begun = Instant::now();
                let device = Device::Cuda(0);
                let result = tch::no_grad(|| {
                    module.forward_ts(&[
                        Tensor::from_slice(&x)
                            .view([(nc + nq) as i64, n as i64, width as i64])
                            .to_device(device),
                        Tensor::from_slice(&y)
                            .view([nc as i64, n as i64])
                            .to_device(device),
                    ])
                })?
                .to_kind(Kind::Float)
                .to_device(Device::Cpu);
                ensure!(
                    result.size() == [n as i64, 2048]
                        && result.isfinite().all().int64_value(&[]) != 0,
                    "invalid foundation output"
                );
                let values: Vec<f32> = Vec::try_from(result.flatten(0, -1))?;
                gpu_seconds += begun.elapsed().as_secs_f64();
                batches += 1;
                for (v, encoded) in group.iter().zip(values.chunks_exact(2048)) {
                    fs::create_dir_all(&v.dest)?;
                    let bytes = encoded
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    let cut = v.p.cuts.iter().find(|c| c.index == v.job.cut).unwrap();
                    atomic_write(&v.dest.join("foundation_raw.f32"), &bytes)?;
                    write_json(
                        &v.dest.join("foundation.json"),
                        &json!({"schema":"parallel.foundation_cut.v1","dataset":v.p.dataset,"family":v.p.family,"cut":v.job.cut,"prepared_sha256":v.job.prepared_sha256,"training_sha256":cut.training_sha256,"preprocessing_sha256":cut.preprocessing_sha256,"spec_sha256":spec_sha,"signature_sha256":hash(&bytes),"shape":[2048],"projected":false,"context_rows":nc,"query_rows":nq,"feature_width":width,"backend":"Rust/LibTorch 2.7 CUDA FP32","final_test_targets_used":false,"copied_dataset_bytes":0}),
                    )?;
                    completed += 1;
                }
                write_json(
                    &output.join(format!("rank-{rank}.json")),
                    &json!({"rank":rank,"world":world,"phase":"frozen foundation encoding","completed":completed,"total":total,"gpu_batches":batches,"gpu_seconds":gpu_seconds,"last_batch":n,"cuts_per_second":completed as f64/start.elapsed().as_secs_f64(),"timestamp":crate::telemetry::now(),"elapsed_seconds":start.elapsed().as_secs_f64(),"failures":[]}),
                )?;
            }
        }
    }
    let receipt = json!({"rank":rank,"world":world,"phase":"completed","completed":completed,"total":total,"gpu_batches":batches,"gpu_seconds":gpu_seconds,"timestamp":crate::telemetry::now(),"seconds":start.elapsed().as_secs_f64(),"failures":[]});
    write_json(&output.join(format!("rank-{rank}.json")), &receipt)?;
    write_json(&output.join(format!("complete-{rank}.json")), &receipt)
}
