//! Pinned deployed Rust/CUDA encoder. This produces the 512-dimensional
//! Hyperion component only; a full policy context also needs foundation/target.
use crate::{manifest::*, prepare::Prepared, service::GpJobs};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
#[derive(Serialize, Deserialize)]
pub struct Spec {
    /// Keep the bounded 104-worker GP pool active while each persistent GPU
    /// encoder shares two reserved CPU cores among its queued input streams.
    /// Absent/false preserves the original exclusive 120-CPU encoding block.
    #[serde(default)]
    pub concurrent_gp: bool,
    #[serde(default)]
    pub persistent_binary: Option<PathBuf>,
    #[serde(default)]
    pub persistent_binary_sha256: Option<String>,
    pub loader: PathBuf,
    pub loader_sha256: String,
    pub runtime_manifest: PathBuf,
    pub runtime_manifest_sha256: String,
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub weights: PathBuf,
    pub weights_sha256: String,
    pub config: PathBuf,
    pub config_sha256: String,
    pub source_manifest: PathBuf,
    pub source_manifest_sha256: String,
}

#[derive(Debug)]
struct Schedule {
    cpu_width: usize,
    exclusive_cpu: bool,
    minimum_available_gib: f64,
}
impl Schedule {
    fn new(persistent: bool, concurrent_gp: bool) -> Result<Self> {
        ensure!(
            !concurrent_gp || persistent,
            "concurrent GP encoding requires the persistent batch worker"
        );
        Ok(Self {
            cpu_width: if persistent && !concurrent_gp { 15 } else { 2 },
            exclusive_cpu: persistent && !concurrent_gp,
            // Native workers each enforce a 64-GiB input budget. Leave 128 GiB
            // beyond that combined 512-GiB budget for queues and host headroom.
            minimum_available_gib: if concurrent_gp { 640.0 } else { 128.0 },
        })
    }
    fn cpu_range(&self, rank: usize) -> std::ops::Range<usize> {
        let first = if rank < 4 {
            rank * self.cpu_width
        } else {
            62 + (rank - 4) * self.cpu_width
        };
        first..first + self.cpu_width
    }
}
impl Spec {
    pub fn verify(&self, root: &Path) -> Result<()> {
        for (p, sha) in [
            (&self.loader, &self.loader_sha256),
            (&self.runtime_manifest, &self.runtime_manifest_sha256),
            (&self.binary, &self.binary_sha256),
            (&self.weights, &self.weights_sha256),
            (&self.config, &self.config_sha256),
            (&self.source_manifest, &self.source_manifest_sha256),
        ] {
            ensure!(
                file_hash(&resolve(root, p)?)? == *sha,
                "frozen Hyperion dependency changed"
            );
        }
        if let Some(binary) = &self.persistent_binary {
            ensure!(
                Some(file_hash(&resolve(root, binary)?)?) == self.persistent_binary_sha256,
                "persistent encoder binary changed"
            );
        }
        let runtime: TransferManifest = read_json(&resolve(root, &self.runtime_manifest)?)?;
        for f in &runtime.files {
            ensure!(
                file_hash(&resolve(root, &f.destination)?)? == f.sha256,
                "frozen runtime file changed"
            );
        }
        Ok(())
    }
}
pub fn pooled(rows: &[Vec<f64>]) -> Result<Vec<f32>> {
    ensure!(
        !rows.is_empty()
            && rows
                .iter()
                .all(|r| r.len() == 128 && r.iter().all(|v| v.is_finite())),
        "invalid Hyperion row shape or values"
    );
    let mut out = vec![0f32; 512];
    for j in 0..128 {
        let mut values = rows.iter().map(|r| r[j]).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        let quantile = |q: f64| {
            let x = q * (values.len() - 1) as f64;
            let lo = x.floor() as usize;
            let hi = x.ceil() as usize;
            values[lo] * (1.0 - x.fract()) + values[hi] * x.fract()
        };
        out[j] = mean as f32;
        out[128 + j] = std as f32;
        out[256 + j] = quantile(0.1) as f32;
        out[384 + j] = quantile(0.9) as f32;
    }
    Ok(out)
}
pub fn rank(
    root: &Path,
    jobs_path: &Path,
    spec_path: &Path,
    output: &Path,
    rank: usize,
    world: usize,
) -> Result<()> {
    ensure!(
        world > 0 && world <= 8 && rank < world,
        "invalid encoder rank"
    );
    let spec: Spec = read_json(spec_path)?;
    spec.verify(root)?;
    let spec_sha = file_hash(spec_path)?;
    let jobs: GpJobs = read_json(jobs_path)?;
    let mut completed = 0;
    let mut failures = Vec::new();
    let start = Instant::now();
    for (index, j) in jobs
        .jobs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % world == rank)
    {
        let prep = resolve(root, &j.prepared)?;
        ensure!(
            file_hash(&prep.join("prepared.json"))? == j.prepared_sha256,
            "prepared input changed"
        );
        let p: Prepared = read_json(&prep.join("prepared.json"))?;
        let c = p
            .cuts
            .iter()
            .find(|c| c.index == j.cut)
            .context("unknown cut")?;
        let rows = crate::prepare::load_view(&prep, j.cut, true)?;
        // Anonymous, inherited memory file: the existing CLI sees a file handle,
        // while no cut CSV is ever stored on disk.
        use std::io::Write;
        use std::os::fd::{AsRawFd, FromRawFd};
        let name = std::ffi::CString::new("jope-training-view")?;
        let fd = unsafe { libc::memfd_create(name.as_ptr(), 0) };
        ensure!(fd >= 0, "memory view allocation failed");
        let mut memory = unsafe { fs::File::from_raw_fd(fd) };
        memory.write_all(&crate::prepare::csv_bytes(&rows)?)?;
        let train = PathBuf::from(format!("/proc/self/fd/{}", memory.as_raw_fd()));
        let dest = output.join(&p.dataset).join(format!("cut-{}", j.cut));
        fs::create_dir_all(&dest)?;
        if dest.join("hyperion.json").exists() {
            let receipt: serde_json::Value = read_json(&dest.join("hyperion.json"))?;
            ensure!(
                receipt["training_sha256"] == c.training_sha256
                    && receipt["spec_sha256"] == spec_sha,
                "existing Hyperion context identity mismatch"
            );
            ensure!(
                receipt["signature_sha256"] == file_hash(&dest.join("hyperion.f32"))?,
                "existing Hyperion rows changed"
            );
            completed += 1;
            continue;
        }
        let log = fs::File::create(dest.join("encoder.log"))?;
        let mut cmd = Command::new("timeout");
        cmd.args(["--kill-after=5", "600"])
            .arg(resolve(root, &spec.loader)?)
            .arg("--library-path")
            .arg(
                resolve(root, &spec.loader)?
                    .parent()
                    .context("runtime library directory")?,
            )
            .arg(resolve(root, &spec.binary)?)
            .arg("--train")
            .arg(&train)
            .arg("--test")
            .arg(&train)
            .arg("--out-cell-train")
            .arg(dest.join("cells.vs"))
            .arg("--out-cell-test")
            .arg(dest.join("replay-cells.vs"))
            .arg("--out-row-train")
            .arg(dest.join("rows.vs"))
            .arg("--out-row-test")
            .arg(dest.join("replay-rows.vs"))
            .arg("--weights")
            .arg(resolve(root, &spec.weights)?)
            .arg("--manifest")
            .arg(resolve(root, &spec.config)?)
            .args([
                "--sweeps",
                "3",
                "--scales",
                "4,7,11,15",
                "--seed",
                "20260623",
                "--batch",
                "256",
            ])
            .env("CUDA_VISIBLE_DEVICES", rank.to_string())
            .env("RAYON_NUM_THREADS", "2")
            .env("OMP_NUM_THREADS", "2")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        let result = (|| -> Result<()> {
            ensure!(
                cmd.status()?.success(),
                "encoder process failed or exceeded 600 seconds"
            );
            ensure!(
                fs::read_to_string(dest.join("encoder.log"))?.contains("device=cuda:0"),
                "encoder fell back to CPU; refusing a CUDA receipt"
            );
            let rows = crate::prepare::Table::numeric(&dest.join("rows.vs"))?;
            ensure!(
                rows.len() == c.train_indices.len(),
                "encoder row count mismatch"
            );
            let signature = pooled(&rows)?;
            atomic_write(
                &dest.join("hyperion.f32"),
                &signature
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            write_json(
                &dest.join("hyperion.json"),
                &serde_json::json!({"schema":"parallel.hyperion_cut.v1","dataset":p.dataset,"cut":j.cut,"prepared_sha256":j.prepared_sha256,"training_sha256":c.training_sha256,"preprocessing_sha256":c.preprocessing_sha256,"spec_sha256":spec_sha,"row_sha256":file_hash(&dest.join("rows.vs"))?,"signature_path":"hyperion.f32","signature_sha256":hash(&signature.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),"shape":[512],"target_access":"training partition only; final test never opened","backend":"pinned Rust/Candle CUDA binary","full_context_complete":false}),
            )?;
            for file in ["cells.vs", "replay-cells.vs", "rows.vs", "replay-rows.vs"] {
                fs::remove_file(dest.join(file))?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            failures.push(serde_json::json!({"job":index,"dataset":p.dataset,"cut":j.cut,"error":format!("{e:#}")}));
        } else {
            completed += 1;
        }
        ensure!(
            failures.len() < 3,
            "three encoder failures on rank {rank}; stopping for diagnosis"
        );
        write_json(
            &output.join(format!("rank-{rank}.json")),
            &serde_json::json!({"rank":rank,"world":world,"phase":"frozen Hyperion cut encoding","completed":completed,"failures":failures,"timestamp":crate::telemetry::now(),"elapsed_seconds":start.elapsed().as_secs_f64()}),
        )?;
    }
    write_json(
        &output.join(format!("complete-{rank}.json")),
        &serde_json::json!({"rank":rank,"completed":completed,"failures":failures,"seconds":start.elapsed().as_secs_f64()}),
    )?;
    ensure!(failures.is_empty(), "Hyperion cut jobs failed");
    Ok(())
}
pub fn service(root: &Path, jobs: &Path, spec: &Path, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    use std::os::fd::AsRawFd;
    let lease = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join("runs/encoding-service.lock"))?;
    ensure!(
        unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "encoding service already running"
    );
    let specification: Spec = read_json(spec)?;
    specification.verify(root)?;
    let persistent = specification
        .persistent_binary
        .as_ref()
        .map(|p| resolve(root, p))
        .transpose()?;
    let schedule = Schedule::new(persistent.is_some(), specification.concurrent_gp)?;
    let hardware: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
    ensure!(
        crate::telemetry::now().saturating_sub(hardware.timestamp) <= 5
            && hardware.available_gb >= schedule.minimum_available_gib
            && hardware.disk_free_gb > 50.0,
        "encoding admission paused: stale telemetry or insufficient RAM/disk headroom"
    );
    ensure!(!specification.concurrent_gp || hardware.ram_total_gb >= 1472.0,
        "concurrent budgets require 832 GiB for GP workers, 512 GiB for encoder inputs, and 128 GiB headroom");
    if specification.concurrent_gp {
        let gp: serde_json::Value = read_json(&root.join("runs/gp-service.json"))?;
        ensure!(gp["schema"] == "parallel.gp_service.v1"
            && (gp["phase"] == "running" || gp["phase"] == "completed")
            && gp["cpu_budget"].as_u64().is_some_and(|n| n <= 104)
            && gp["worker_memory_limit_gib"].as_u64().is_some_and(|n| n <= 8)
            && (gp["phase"] == "completed" || crate::telemetry::now().saturating_sub(gp["timestamp"].as_u64().unwrap_or(0)) <= 5),
            "concurrent GP service must have fresh telemetry and stay within 104 CPUs / 8 GiB per worker");
    }
    let mut children = Vec::new();
    let exe = persistent.clone().unwrap_or(std::env::current_exe()?);
    let total = read_json::<GpJobs>(jobs)?.jobs.len();
    if schedule.exclusive_cpu {
        let gp: serde_json::Value =
            read_json(&root.join("runs/gp-service.json")).unwrap_or_default();
        ensure!(
            gp["active"].as_u64().unwrap_or(0) == 0,
            "exclusive encoding requires the GP CPU block to drain first"
        );
    }
    let started_at = crate::telemetry::now();
    let started = Instant::now();
    let result = (|| -> Result<()> {
        for rank in 0..8 {
            let cpus = schedule.cpu_range(rank);
            let log = fs::File::create(output.join(format!("worker-{rank}.log")))?;
            let mut command = Command::new("taskset");
            command
                .args(["-c", &format!("{}-{}", cpus.start, cpus.end - 1)])
                .arg(&exe);
            if persistent.is_none() {
                command.arg("--root").arg(root).arg("hyperion-rank");
            }
            if persistent.is_some() {
                command.arg("--root").arg(root);
            }
            command
                .arg("--jobs")
                .arg(jobs)
                .arg("--spec")
                .arg(spec)
                .arg("--output")
                .arg(output)
                .arg("--rank")
                .arg(rank.to_string())
                .args(["--world", "8"])
                .env("CUDA_VISIBLE_DEVICES", rank.to_string())
                .env("JAIN_HYPERION_CUDA_MAX_LOAD_RATIO", "1000")
                // The native worker retains its bounded 14-stream pool and
                // depth-28 queue. Affinity, rather than input-queue starvation,
                // limits those waiting/working streams to two CPUs per GPU.
                .env(
                    "RAYON_NUM_THREADS",
                    if specification.concurrent_gp {
                        "1"
                    } else {
                        "14"
                    },
                )
                .env("OMP_NUM_THREADS", "1")
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            use std::os::unix::process::CommandExt;
            unsafe {
                command.pre_exec(|| {
                    if libc::setpgid(0, 0) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            children.push(command.spawn()?);
        }
        loop {
            let hardware: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
            ensure!(
                crate::telemetry::now().saturating_sub(hardware.timestamp) <= 5
                    && hardware.available_gb >= 128.0
                    && hardware.disk_free_gb > 50.0,
                "encoding stopped to preserve host headroom or recover unavailable telemetry"
            );
            let mut complete = true;
            let mut ranks = Vec::new();
            for (rank, child) in children.iter_mut().enumerate() {
                if let Some(status) = child.try_wait()? {
                    ensure!(status.success(), "Hyperion rank {rank} failed");
                } else {
                    complete = false;
                }
                if let Ok(receipt) =
                    read_json::<serde_json::Value>(&output.join(format!("rank-{rank}.json")))
                {
                    if receipt["timestamp"].as_u64().unwrap_or(0) >= started_at {
                        ensure!(
                            crate::telemetry::now()
                                .saturating_sub(receipt["timestamp"].as_u64().unwrap())
                                < 600,
                            "Hyperion rank {rank} heartbeat timeout"
                        );
                        ranks.push(receipt);
                    } else {
                        ensure!(
                            started.elapsed() < Duration::from_secs(600),
                            "Hyperion rank {rank} startup timeout"
                        );
                    }
                } else {
                    ensure!(
                        started.elapsed() < Duration::from_secs(600),
                        "Hyperion rank {rank} startup timeout"
                    );
                }
            }
            write_json(
                &root.join("runs/encoding-service.json"),
                // `persistent` is the legacy v23 GP admission flag. Keep it
                // false in concurrent mode so the deployed GP controller does
                // not drain; record the actual encoder backend separately.
                &serde_json::json!({"phase":if complete{"completed"}else{"persistent batched Hyperion encoding"},"total":total,"world":8,"persistent":schedule.exclusive_cpu,"persistent_encoder":persistent.is_some(),"exclusive_cpu":schedule.exclusive_cpu,"concurrent_gp":specification.concurrent_gp,"cpu_budget":schedule.cpu_width*8,"cpu_threads_each":schedule.cpu_width,"preprocessing_streams_per_rank":if persistent.is_some(){14}else{1},"input_queue_depth_per_rank":if persistent.is_some(){28}else{1},"encoder_input_memory_budget_gib":if persistent.is_some(){512}else{0},"completed":ranks.iter().map(|r|r["completed"].as_u64().unwrap_or(0)).sum::<u64>(),"ranks":ranks,"timestamp":crate::telemetry::now()}),
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
            if child.try_wait()?.is_none() {
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
                let _ = child.wait();
            }
        }
        write_json(
            &root.join("runs/encoding-service.json"),
            &serde_json::json!({
                "phase":"failed", "persistent":false, "persistent_encoder":persistent.is_some(),
                "exclusive_cpu":false, "concurrent_gp":specification.concurrent_gp,
                "failure":result.as_ref().err().map(|e|format!("{e:#}")),
                "timestamp":crate::telemetry::now(), "saved_encodings_preserved":true
            }),
        )?;
    }
    result
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn concurrent_encoders_and_gp_have_disjoint_complete_cpu_budgets() {
        let schedule = Schedule::new(true, true).unwrap();
        let encoding = (0..8)
            .flat_map(|r| schedule.cpu_range(r))
            .collect::<BTreeSet<_>>();
        let gp = (8..60).chain(70..122).collect::<BTreeSet<_>>();
        let reserved = [60, 61, 122, 123].into_iter().collect::<BTreeSet<_>>();
        assert_eq!(encoding.len(), 16);
        assert_eq!(gp.len(), 104);
        assert!(encoding.is_disjoint(&gp));
        assert!(encoding.is_disjoint(&reserved));
        assert_eq!(
            encoding
                .union(&gp)
                .copied()
                .collect::<BTreeSet<_>>()
                .union(&reserved)
                .count(),
            124
        );
        assert!(!schedule.exclusive_cpu);
        assert_eq!(schedule.minimum_available_gib, 640.0);
    }
    #[test]
    fn default_exclusive_budget_remains_available() {
        let schedule = Schedule::new(true, false).unwrap();
        assert!(schedule.exclusive_cpu);
        assert_eq!(
            (0..8)
                .flat_map(|r| schedule.cpu_range(r))
                .collect::<BTreeSet<_>>()
                .len(),
            120
        );
        assert!(Schedule::new(false, true).is_err());
    }
}
