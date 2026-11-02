use super::*;

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
    crate::task_contract::require_original_gpu_grant("foundation-rank")?;
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
