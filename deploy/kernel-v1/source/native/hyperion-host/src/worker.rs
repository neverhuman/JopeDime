use anyhow::{ensure, Context, Result};
use feat_core::{
    data::Matrix,
    hyperion::{fit_quantile_bounds, load_settings},
    hyperion_v7::{
        backend::V7Model,
        config::SweepPlan,
        grid::quantize_u8_with_bounds,
        sweep::{run_composite_sweep, PatchModel, PatchOutputs},
    },
    progress::NullSink,
};
use jope_parallel::{
    hyperion::pooled,
    manifest::*,
    prepare::{load_view, Prepared},
    service::{GpJob, GpJobs},
};
use rayon::prelude::*;
use serde_json::json;
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};
struct Request {
    x: Vec<f32>,
    m: Vec<f32>,
    batch: usize,
    p: usize,
    reply: mpsc::SyncSender<std::result::Result<PatchOutputs, String>>,
}
#[derive(Clone)]
struct Proxy {
    tx: mpsc::SyncSender<Request>,
    dim: usize,
}
impl PatchModel for Proxy {
    fn forward_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.tx.send(Request {
            x: x.into(),
            m: m.into(),
            batch,
            p,
            reply: tx,
        })?;
        rx.recv()?.map_err(anyhow::Error::msg)
    }
    fn emb_dim(&self) -> usize {
        self.dim
    }
}
struct Budget {
    used: Mutex<usize>,
    ready: Condvar,
}
struct Permit<'a> {
    budget: &'a Budget,
    bytes: usize,
}
impl Budget {
    fn acquire(&self, bytes: usize) -> Result<Permit<'_>> {
        ensure!(
            bytes <= (64usize << 30),
            "one encoding view exceeds the 64 GiB host memory budget"
        );
        let mut used = self.used.lock().unwrap();
        while *used + bytes > (64usize << 30) {
            used = self.ready.wait(used).unwrap();
        }
        *used += bytes;
        Ok(Permit {
            budget: self,
            bytes,
        })
    }
}
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.budget.used.lock().unwrap() -= self.bytes;
        self.budget.ready.notify_all();
    }
}
fn encode(
    root: &Path,
    j: &GpJob,
    spec_sha: &str,
    output: &Path,
    model: &Proxy,
    settings: &feat_core::hyperion::HyperionSweepSettings,
    budget: &Budget,
) -> Result<()> {
    let prepared = resolve(root, &j.prepared)?;
    ensure!(
        file_hash(&prepared.join("prepared.json"))? == j.prepared_sha256,
        "prepared identity changed"
    );
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    let cut = p
        .cuts
        .iter()
        .find(|c| c.index == j.cut)
        .context("unknown cut")?;
    let dest = output.join(&p.dataset).join(format!("cut-{}", j.cut));
    fs::create_dir_all(&dest)?;
    if dest.join("hyperion.json").exists() {
        let old: serde_json::Value = read_json(&dest.join("hyperion.json"))?;
        ensure!(
            old["prepared_sha256"] == j.prepared_sha256
                && old["training_sha256"] == cut.training_sha256
                && old["preprocessing_sha256"] == cut.preprocessing_sha256
                && old["spec_sha256"] == spec_sha
                && old["signature_sha256"] == file_hash(&dest.join("hyperion.f32"))?,
            "cached encoding identity mismatch"
        );
        return Ok(());
    }
    let pp: jope_parallel::prepare::Preprocessing =
        read_json(&prepared.join(format!("cut-{}/preprocessing.json", j.cut)))?;
    let bytes = cut
        .train_indices
        .len()
        .checked_mul(pp.columns.len())
        .and_then(|n| n.checked_mul(128 * 4 * 3 + 32))
        .context("encoding shape overflow")?
        + 16 * 1024 * 1024;
    let _permit = budget.acquire(bytes)?;
    let rows = load_view(&prepared, j.cut, true)?;
    let n = rows.len();
    let width = rows[0].len() - 1;
    let mut columns = vec![0f32; n * width];
    for (i, row) in rows.iter().enumerate() {
        for k in 0..width {
            columns[k * n + i] = row[k] as f32;
        }
    }
    drop(rows);
    let x = Matrix::new(columns, n, width);
    let bounds = fit_quantile_bounds(&x, settings.q_low, settings.q_high);
    let grid = quantize_u8_with_bounds(&x, &bounds, settings.pad_u8);
    let plan = SweepPlan {
        sweeps: 3,
        scales: vec![4, 7, 11, 15],
        batch: 1024,
        seed: settings.seed,
        target_chunks_per_sweep: 1,
        fixed_perms: None,
    };
    let start = Instant::now();
    let encoded = run_composite_sweep(
        model,
        &grid,
        settings.pad_u8,
        settings.quant_scale,
        settings.max_patch,
        &plan,
        &NullSink,
        |_| {},
    )?;
    ensure!(!encoded.cancelled, "encoding cancelled");
    let rows = encoded
        .row_emb
        .chunks_exact(128)
        .map(|r| r.iter().map(|v| *v as f64).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let signature = pooled(&rows)?;
    let values = signature
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    atomic_write(&dest.join("hyperion.f32"), &values)?;
    write_json(
        &dest.join("hyperion.json"),
        &json!({"schema":"parallel.hyperion_cut.v1","dataset":p.dataset,"family":p.family,"cut":j.cut,"prepared_sha256":j.prepared_sha256,"training_sha256":cut.training_sha256,"preprocessing_sha256":cut.preprocessing_sha256,"spec_sha256":spec_sha,"signature_path":"hyperion.f32","signature_sha256":hash(&values),"shape":[512],"seconds":start.elapsed().as_secs_f64(),"backend":"persistent Rust/Candle CUDA; independent patches; 24000-cell forward cap","max_forward_cells":24000,"full_context_complete":false,"copied_dataset_bytes":0,"final_test_targets_used":false}),
    )
}
fn arg(name: &str) -> Result<String> {
    let a = std::env::args().collect::<Vec<_>>();
    let i = a
        .iter()
        .position(|v| v == name)
        .with_context(|| format!("missing {name}"))?;
    a.get(i + 1).cloned().context("missing argument value")
}
fn main() -> Result<()> {
    let root = PathBuf::from(arg("--root")?);
    let jobs_path = PathBuf::from(arg("--jobs")?);
    let output = PathBuf::from(arg("--output")?);
    let spec_path = PathBuf::from(arg("--spec")?);
    let rank: usize = arg("--rank")?.parse()?;
    let world: usize = arg("--world")?.parse()?;
    ensure!(rank < world && world <= 8, "invalid rank");
    fs::create_dir_all(&output)?;
    let typed: jope_parallel::hyperion::Spec = read_json(&spec_path)?;
    typed.verify(&root)?;
    let spec: serde_json::Value = read_json(&spec_path)?;
    let spec_sha = file_hash(&spec_path)?;
    let weights = resolve(
        &root,
        Path::new(spec["weights"].as_str().context("weights missing")?),
    )?;
    let config = resolve(
        &root,
        Path::new(spec["config"].as_str().context("configuration missing")?),
    )?;
    ensure!(
        file_hash(&weights)? == spec["weights_sha256"]
            && file_hash(&config)? == spec["config_sha256"],
        "encoder identity changed"
    );
    let (settings, _) = load_settings(&config, 20260623)?;
    let model = V7Model::load(&weights, &settings)?;
    ensure!(
        model.device_label() == "cuda:0",
        "persistent encoder requires CUDA; CPU fallback refused"
    );
    let jobs: GpJobs = read_json(&jobs_path)?;
    let jobs = jobs
        .jobs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % world == rank)
        .map(|(_, j)| j)
        .collect::<Vec<_>>();
    let total = jobs.len();
    let start = Instant::now();
    let completed = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let finished = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::sync_channel::<Request>(28);
    let proxy = Proxy {
        tx,
        dim: model.emb_dim(),
    };
    let budget = Budget {
        used: Mutex::new(0),
        ready: Condvar::new(),
    };
    let (done_count, errors, done) = (completed.clone(), failures.clone(), finished.clone());
    let worker_output = output.clone();
    let worker_root = root.clone();
    let worker = std::thread::spawn(move || {
        rayon::ThreadPoolBuilder::new()
            .num_threads(14)
            .build()
            .unwrap()
            .install(|| {
                jobs.par_iter().for_each(|job| {
                    match encode(
                        &worker_root,
                        job,
                        &spec_sha,
                        &worker_output,
                        &proxy,
                        &settings,
                        &budget,
                    ) {
                        Ok(()) => {
                            done_count.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => errors.lock().unwrap().push(format!(
                            "{} cut {}: {e:#}",
                            job.prepared.display(),
                            job.cut
                        )),
                    }
                    done.fetch_add(1, Ordering::Relaxed);
                })
            });
    });
    let mut pending = VecDeque::new();
    let mut batches = 0usize;
    let mut patches = 0usize;
    let mut gpu_seconds = 0.0;
    let mut last = Instant::now() - Duration::from_secs(1);
    // A CUDA call can remain busy without making progress. Terminate this rank
    // after a bounded stalled forward; the controller restarts from saved cuts.
    let forward_started_ms = Arc::new(AtomicU64::new(0));
    let watchdog_started = forward_started_ms.clone();
    let watchdog_output = output.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(1));
        let begun = watchdog_started.load(Ordering::Acquire);
        let elapsed = start.elapsed().as_millis() as u64 + 1;
        if begun != 0 && elapsed.saturating_sub(begun) > 60_000 {
            let _ = write_json(&watchdog_output.join(format!("failed-rank-{rank}.json")),
                &json!({"rank":rank,"phase":"failed","reason":"CUDA forward exceeded 60-second useful-progress deadline","timestamp":jope_parallel::telemetry::now()}));
            eprintln!("rank {rank}: CUDA forward exceeded 60 seconds; durable cut encodings preserved");
            std::process::exit(124);
        }
    });
    loop {
        if pending.is_empty() {
            match rx.recv_timeout(Duration::from_millis(2)) {
                Ok(r) => pending.push_back(r),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(_) => {
                    if finished.load(Ordering::Relaxed) == total {
                        break;
                    }
                    continue;
                }
            }
        }
        // A short coalescing window merges independent datasets only along the
        // batch dimension. Attention and cut preprocessing remain independent.
        let until = Instant::now() + Duration::from_millis(2);
        while Instant::now() < until {
            match rx.recv_timeout(until.saturating_duration_since(Instant::now())) {
                Ok(r) => pending.push_back(r),
                Err(_) => break,
            }
        }
        let first = pending.pop_front().unwrap();
        let scale = first.p;
        let mut count = first.batch;
        let mut group = vec![first];
        let mut retained = VecDeque::new();
        while let Some(request) = pending.pop_front() {
            if request.p == scale && (count + request.batch) * scale * scale <= 24_000 {
                count += request.batch;
                group.push(request);
            } else {
                retained.push_back(request);
            }
        }
        pending = retained;
        let x = group
            .iter()
            .flat_map(|r| r.x.iter().copied())
            .collect::<Vec<_>>();
        let m = group
            .iter()
            .flat_map(|r| r.m.iter().copied())
            .collect::<Vec<_>>();
        let begun = Instant::now();
        if last.elapsed() >= Duration::from_secs(1) {
            write_json(&output.join(format!("inflight-rank-{rank}.json")),
                &json!({"rank":rank,"phase":"CUDA forward","batch":count,"scale":scale,"cells":count*scale*scale,"max_forward_cells":24000,"completed":completed.load(Ordering::Relaxed),"gpu_batches":batches,"timestamp":jope_parallel::telemetry::now()}))?;
        }
        forward_started_ms.store(start.elapsed().as_millis() as u64 + 1, Ordering::Release);
        let result = model.forward_batch(&x, &m, count, scale);
        forward_started_ms.store(0, Ordering::Release);
        gpu_seconds += begun.elapsed().as_secs_f64();
        batches += 1;
        patches += count;
        match result {
            Ok(result) => {
                let mut offset = 0;
                for request in group {
                    let d = model.emb_dim();
                    let result = PatchOutputs {
                        emb: result.emb[offset * d..(offset + request.batch) * d].to_vec(),
                        logits: result.logits
                            [offset * scale * scale..(offset + request.batch) * scale * scale]
                            .to_vec(),
                    };
                    offset += request.batch;
                    let _ = request.reply.send(Ok(result));
                }
            }
            Err(error) => {
                let msg = format!("{error:#}");
                for r in group {
                    let _ = r.reply.send(Err(msg.clone()));
                }
            }
        }
        if last.elapsed() >= Duration::from_secs(1) {
            write_json(
                &output.join(format!("rank-{rank}.json")),
                &json!({"rank":rank,"world":world,"phase":"persistent batched GPU encoding","completed":completed.load(Ordering::Relaxed),"total":total,"failures":*failures.lock().unwrap(),"gpu_batches":batches,"patches":patches,"patches_per_second":patches as f64/start.elapsed().as_secs_f64(),"gpu_seconds":gpu_seconds,"last_batch":count,"timestamp":jope_parallel::telemetry::now(),"elapsed_seconds":start.elapsed().as_secs_f64()}),
            )?;
            last = Instant::now();
        }
    }
    worker
        .join()
        .map_err(|_| anyhow::anyhow!("encoding CPU worker panicked"))?;
    let receipt = json!({"rank":rank,"phase":"completed","completed":completed.load(Ordering::Relaxed),"total":total,"failures":*failures.lock().unwrap(),"seconds":start.elapsed().as_secs_f64(),"patches":patches,"gpu_batches":batches,"gpu_seconds":gpu_seconds,"patches_per_second":patches as f64/start.elapsed().as_secs_f64(),"timestamp":jope_parallel::telemetry::now()});
    write_json(&output.join(format!("complete-{rank}.json")), &receipt)?;
    write_json(&output.join(format!("rank-{rank}.json")), &receipt)?;
    ensure!(failures.lock().unwrap().is_empty(), "cut encoding failures");
    Ok(())
}
