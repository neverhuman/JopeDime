use crate::manifest::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    process::Command,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Gpu {
    pub index: usize,
    pub utilization: f64,
    pub memory_mb: f64,
    pub memory_total_mb: f64,
    pub watts: f64,
    pub temperature: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Hardware {
    pub timestamp: u64,
    pub gpus: Vec<Gpu>,
    pub cpu: Vec<f64>,
    pub cpu_total: f64,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub available_gb: f64,
    pub disk_free_gb: f64,
    pub load: String,
    pub errors: Vec<String>,
    #[serde(default)]
    pub numa: Vec<Numa>,
    #[serde(default)]
    pub disk_read_mib_s: Option<f64>,
    #[serde(default)]
    pub disk_write_mib_s: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Numa {
    pub node: usize,
    pub cpus: String,
    pub activity: f64,
}
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Stage liveness is separate from useful work. A live controller can still be
/// waiting on a fit or filesystem operation; only `advance` moves work counters.
pub struct StageHeartbeat {
    state: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
    stop: std::sync::mpsc::Sender<()>,
    worker: Option<std::thread::JoinHandle<Result<()>>>,
}
impl StageHeartbeat {
    pub fn start(path: &Path, state: serde_json::Value) -> Result<Self> {
        Self::with_interval(path, state, Duration::from_secs(2))
    }
    fn with_interval(path: &Path, mut value: serde_json::Value, interval: Duration) -> Result<Self> {
        anyhow::ensure!(value.is_object() && !interval.is_zero(), "invalid stage heartbeat");
        value["heartbeat_role"] = "controller_liveness".into();
        value["timestamp"] = now().into();
        value["last_progress_at"] = now().into();
        value["heartbeat_sequence"] = 0.into();
        write_json(path, &value)?;
        let state = std::sync::Arc::new(std::sync::Mutex::new(value));
        let shared = state.clone();
        let path = path.to_owned();
        let (stop, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            loop {
                let stopping = !matches!(receiver.recv_timeout(interval), Err(std::sync::mpsc::RecvTimeoutError::Timeout));
                let mut value = shared.lock().map_err(|_| anyhow::anyhow!("stage progress poisoned"))?;
                value["timestamp"] = now().into();
                value["heartbeat_sequence"] = (value["heartbeat_sequence"].as_u64().unwrap_or(0) + 1).into();
                write_json(&path, &*value)?;
                if stopping { return Ok(()); }
            }
        });
        Ok(Self { state, stop, worker: Some(worker) })
    }
    pub fn advance(&self, operation: &str, completed: usize, total: Option<usize>) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| anyhow::anyhow!("stage progress poisoned"))?;
        let work = serde_json::json!({"operation":operation,"completed":completed,"total":total});
        if state["work"] == work { return Ok(()); }
        state["work"] = work;
        state["last_progress_at"] = now().into();
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        self.join()
    }
    fn join(&mut self) -> Result<()> {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| anyhow::anyhow!("stage heartbeat panicked"))??;
        }
        Ok(())
    }
}
impl Drop for StageHeartbeat {
    fn drop(&mut self) { let _ = self.join(); }
}

#[cfg(test)]
mod heartbeat_tests {
    use super::*;
    #[test]
    fn slow_stages_keep_liveness_separate_and_join_before_next_phase() {
        for phase in ["Fit and save predictors", "Pack newly scored programs"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("progress.json");
            let payload = dir.path().join("payload");
            fs::write(&payload, b"frozen fitted predictor or corpus").unwrap();
            let original = file_hash(&payload).unwrap();
            let stage = StageHeartbeat::with_interval(&path, serde_json::json!({"phase":phase}), Duration::from_millis(10)).unwrap();
            let initial: serde_json::Value = read_json(&path).unwrap();
            let started = std::time::Instant::now();
            loop {
                let value: serde_json::Value = read_json(&path).unwrap();
                if value["heartbeat_sequence"].as_u64().unwrap() >= 2 {
                    assert_eq!(value["phase"], phase);
                    assert_eq!(value["last_progress_at"], initial["last_progress_at"]);
                    assert!(value["work"].is_null());
                    break;
                }
                assert!(started.elapsed() < Duration::from_secs(2));
                std::thread::sleep(Duration::from_millis(5));
            }
            stage.advance("published artifacts", 1, Some(2)).unwrap();
            stage.finish().unwrap();
            let value: serde_json::Value = read_json(&path).unwrap();
            assert_eq!(value["work"]["completed"], 1);
            assert_eq!(value["heartbeat_role"], "controller_liveness");
            write_json(&path, &serde_json::json!({"phase":"next"})).unwrap();
            std::thread::sleep(Duration::from_millis(40));
            assert_eq!(read_json::<serde_json::Value>(&path).unwrap()["phase"], "next");
            assert_eq!(file_hash(&payload).unwrap(), original);
        }
    }
    #[test]
    fn cancellation_joins_publisher_and_write_errors_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progress.json");
        let stage = StageHeartbeat::start(&path, serde_json::json!({"phase":"packing"})).unwrap();
        drop(stage); // Same cleanup as early return or unwinding in a stage.
        let saved = fs::read(&path).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(fs::read(&path).unwrap(), saved);
        let stage = StageHeartbeat::start(&path, serde_json::json!({})).unwrap();
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(stage.finish().is_err());
    }
}
fn cpu_counters() -> Vec<(u64, u64)> {
    fs::read_to_string("/proc/stat")
        .unwrap_or_default()
        .lines()
        .filter(|s| s.starts_with("cpu"))
        .filter_map(|line| {
            let ns = line
                .split_whitespace()
                .skip(1)
                .take(8)
                .map(|v| v.parse::<u64>().ok())
                .collect::<Option<Vec<_>>>()?;
            Some((
                ns.iter().sum(),
                ns.get(3).unwrap_or(&0) + ns.get(4).unwrap_or(&0),
            ))
        })
        .collect()
}
pub struct Sampler {
    previous: Vec<(u64, u64)>,
    previous_disk: Option<(u64, u64)>,
    sampled: std::time::Instant,
    nodes: Vec<(usize, String, Vec<usize>)>,
}
fn disk_counters() -> Option<(u64, u64)> {
    let mut read = 0;
    let mut write = 0;
    let mut found = false;
    for disk in fs::read_dir("/sys/block").ok()?.flatten() {
        let name = disk.file_name().to_string_lossy().to_string();
        if name.starts_with("loop")
            || name.starts_with("ram")
            || name.starts_with("dm-")
            || name.starts_with("md")
        {
            continue;
        }
        let stats = fs::read_to_string(disk.path().join("stat")).ok()?;
        let fields: Vec<u64> = stats
            .split_whitespace()
            .filter_map(|v| v.parse().ok())
            .collect();
        if fields.len() >= 7 {
            read += fields[2];
            write += fields[6];
            found = true;
        }
    }
    found.then_some((read, write))
}
fn numa_nodes() -> Vec<(usize, String, Vec<usize>)> {
    let mut nodes: Vec<_> = fs::read_dir("/sys/devices/system/node")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let node: usize = entry
                .file_name()
                .to_str()?
                .strip_prefix("node")?
                .parse()
                .ok()?;
            let list = fs::read_to_string(entry.path().join("cpulist"))
                .ok()?
                .trim()
                .to_string();
            let mut cpus = vec![];
            for item in list.split(',') {
                if let Some((start, end)) = item.split_once('-') {
                    cpus.extend(start.parse::<usize>().ok()?..=end.parse::<usize>().ok()?);
                } else {
                    cpus.push(item.parse().ok()?);
                }
            }
            Some((node, list, cpus))
        })
        .collect();
    nodes.sort_by_key(|v| v.0);
    nodes
}
impl Default for Sampler {
    fn default() -> Self {
        Self {
            previous: cpu_counters(),
            previous_disk: disk_counters(),
            sampled: std::time::Instant::now(),
            nodes: numa_nodes(),
        }
    }
}
impl Sampler {
    pub fn sample(&mut self, root: &Path) -> Hardware {
        let mut h = Hardware {
            timestamp: now(),
            ..Default::default()
        };
        match Command::new("timeout").args(["3","nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu","--format=csv,noheader,nounits"]).output(){Ok(o)if o.status.success()=>{for line in String::from_utf8_lossy(&o.stdout).lines(){let v=line.split(',').map(|s|s.trim().parse::<f64>()).collect::<std::result::Result<Vec<_>,_>>();match v{Ok(v)if v.len()==6=>h.gpus.push(Gpu{index:v[0]as usize,utilization:v[1],memory_mb:v[2],memory_total_mb:v[3],watts:v[4],temperature:v[5]}),_=>h.errors.push(format!("unavailable GPU sample: {line}"))}}},Ok(o)=>h.errors.push(format!("GPU telemetry unavailable: {}",String::from_utf8_lossy(&o.stderr))),Err(e)=>h.errors.push(format!("GPU telemetry unavailable: {e}"))}
        let counters = cpu_counters();
        let usage = counters
            .iter()
            .zip(&self.previous)
            .map(|((total, idle), (old, old_idle))| {
                let total = total.saturating_sub(*old);
                let idle = idle.saturating_sub(*old_idle);
                if total == 0 {
                    0.0
                } else {
                    100.0 * (total.saturating_sub(idle)) as f64 / total as f64
                }
            })
            .collect::<Vec<_>>();
        h.cpu_total = usage.first().copied().unwrap_or(0.0);
        h.cpu = usage.into_iter().skip(1).collect();
        h.numa = self
            .nodes
            .iter()
            .map(|(node, cpus, ids)| Numa {
                node: *node,
                cpus: cpus.clone(),
                activity: ids.iter().filter_map(|&i| h.cpu.get(i)).sum::<f64>()
                    / ids.len().max(1) as f64,
            })
            .collect();
        self.previous = counters;
        let disk = disk_counters();
        if let (Some((read, write)), Some((old_read, old_write))) = (disk, self.previous_disk) {
            let seconds = self.sampled.elapsed().as_secs_f64().max(0.001);
            h.disk_read_mib_s = Some(read.saturating_sub(old_read) as f64 / 2048. / seconds);
            h.disk_write_mib_s = Some(write.saturating_sub(old_write) as f64 / 2048. / seconds);
        }
        self.previous_disk = disk;
        self.sampled = std::time::Instant::now();
        let mem = fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let field = |key: &str| {
            mem.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0)
                / 1024.0
                / 1024.0
        };
        h.ram_total_gb = field("MemTotal:");
        h.available_gb = field("MemAvailable:");
        h.ram_used_gb = h.ram_total_gb - h.available_gb;
        if let Ok(path) = std::ffi::CString::new(root.as_os_str().as_encoded_bytes()) {
            let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
            if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } == 0 {
                let stat = unsafe { stat.assume_init() };
                h.disk_free_gb = stat.f_bavail as f64 * stat.f_frsize as f64 / 1024f64.powi(3);
            }
        }
        h.load = fs::read_to_string("/proc/loadavg").unwrap_or_default();
        h
    }
}
pub fn monitor(root: &Path) -> Result<()> {
    let mut sampler = Sampler::default();
    loop {
        let started = std::time::Instant::now();
        let h = sampler.sample(root);
        write_json(&root.join("runs/hardware.json"), &h)?;
        use std::io::Write;
        let archive = root.join(format!(
            "runs/telemetry/hardware-day-{}.jsonl",
            h.timestamp / 86400
        ));
        fs::create_dir_all(archive.parent().unwrap())?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(archive)?;
        serde_json::to_writer(
            &mut file,
            &serde_json::json!({"timestamp":h.timestamp,"cpu":h.cpu_total,"available_gb":h.available_gb,"disk_free_gb":h.disk_free_gb,"disk_read_mib_s":h.disk_read_mib_s,"disk_write_mib_s":h.disk_write_mib_s,"numa":h.numa,"gpus":h.gpus,"errors":h.errors}),
        )?;
        file.write_all(b"\n")?;
        std::thread::sleep(Duration::from_secs(1).saturating_sub(started.elapsed()));
    }
}
