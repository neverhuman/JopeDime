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
    fn with_interval(
        path: &Path,
        mut value: serde_json::Value,
        interval: Duration,
    ) -> Result<Self> {
        anyhow::ensure!(
            value.is_object() && !interval.is_zero(),
            "invalid stage heartbeat"
        );
        value["status_binding"] = status_binding(&value)?;
        value["heartbeat_role"] = "controller_liveness".into();
        let timestamp = now();
        value["timestamp"] = timestamp.into();
        value["expires_at"] = timestamp.saturating_add(10).into();
        value["terminal"] = false.into();
        value["last_progress_at"] = now().into();
        value["heartbeat_sequence"] = 0.into();
        write_json(path, &value)?;
        let state = std::sync::Arc::new(std::sync::Mutex::new(value));
        let shared = state.clone();
        let path = path.to_owned();
        let (stop, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || loop {
            let stopping = !matches!(
                receiver.recv_timeout(interval),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            );
            let mut value = shared
                .lock()
                .map_err(|_| anyhow::anyhow!("stage progress poisoned"))?;
            let timestamp = now();
            value["timestamp"] = timestamp.into();
            value["expires_at"] = timestamp.saturating_add(10).into();
            value["heartbeat_sequence"] =
                (value["heartbeat_sequence"].as_u64().unwrap_or(0) + 1).into();
            write_json(&path, &*value)?;
            if stopping {
                return Ok(());
            }
        });
        Ok(Self {
            state,
            stop,
            worker: Some(worker),
        })
    }
    pub fn advance(&self, operation: &str, completed: usize, total: Option<usize>) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("stage progress poisoned"))?;
        let work = serde_json::json!({"operation":operation,"completed":completed,"total":total});
        if state["work"] == work {
            return Ok(());
        }
        state["work"] = work;
        state["last_progress_at"] = now().into();
        Ok(())
    }
    /// Replace progress without allowing a caller to replace process custody.
    pub fn publish(&self, mut value: serde_json::Value) -> Result<()> {
        anyhow::ensure!(value.is_object(), "invalid heartbeat progress");
        anyhow::ensure!(
            !self
                .worker
                .as_ref()
                .is_some_and(|worker| worker.is_finished()),
            "heartbeat publisher stopped"
        );
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("stage progress poisoned"))?;
        for field in [
            "status_binding",
            "heartbeat_role",
            "heartbeat_sequence",
            "timestamp",
            "expires_at",
            "terminal",
        ] {
            value[field] = state[field].clone();
        }
        value["last_progress_at"] = now().into();
        *state = value;
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        self.terminal("completed")?;
        self.join()
    }
    fn terminal(&self, reason: &str) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("stage progress poisoned"))?;
        state["terminal"] = true.into();
        if state["termination_reason"].is_null() {
            state["termination_reason"] = reason.into();
        }
        Ok(())
    }
    fn join(&mut self) -> Result<()> {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("stage heartbeat panicked"))??;
        }
        Ok(())
    }
}
impl Drop for StageHeartbeat {
    fn drop(&mut self) {
        if self.worker.is_some() {
            let _ = self.terminal("publisher dropped before completion; reconcile activity");
            let _ = self.join();
        }
    }
}

fn executable_observation(pid: u32) -> Result<serde_json::Value> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::metadata(format!("/proc/{pid}/exe"))?;
    Ok(
        serde_json::json!({"device":metadata.dev(),"inode":metadata.ino(),
        "bytes":metadata.len(),"mtime":metadata.mtime(),"mtime_ns":metadata.mtime_nsec(),
        "ctime":metadata.ctime(),"ctime_ns":metadata.ctime_nsec()}),
    )
}

fn status_binding(initial: &serde_json::Value) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    type CachedExecutable = (
        crate::eval_registry::ProcessIdentity,
        serde_json::Value,
        String,
    );
    static CACHE: std::sync::Mutex<Option<CachedExecutable>> = std::sync::Mutex::new(None);
    let process = crate::eval_registry::ProcessIdentity::current()?;
    let executable = executable_observation(process.pid)?;
    // Status publishers in one process may start many stages. Reuse only the
    // same loaded executable, process birth and complete file observation.
    // This cache never grants fit authority or substitutes for a witness ACK.
    let executable_sha256 = {
        let mut cache = CACHE
            .lock()
            .map_err(|_| anyhow::anyhow!("status identity cache poisoned"))?;
        if let Some((_, _, digest)) = cache
            .as_ref()
            .filter(|(p, e, _)| *p == process && *e == executable)
        {
            digest.clone()
        } else {
            let digest = file_hash(Path::new(&format!("/proc/{}/exe", process.pid)))?;
            anyhow::ensure!(
                executable_observation(process.pid)? == executable,
                "loaded status executable changed while hashing"
            );
            *cache = Some((process.clone(), executable.clone(), digest.clone()));
            digest
        }
    };
    let generation = hex::encode(Sha256::digest(serde_json::to_vec(&serde_json::json!({
        "process":process,"executable":executable,"initial":initial,
        "created_monotonic_ns":crate::fit_boundary::deadline_after(Duration::ZERO)?}))?));
    Ok(
        serde_json::json!({"schema":"parallel.status_binding.v1","process":process,
        "executable":executable,"executable_sha256":executable_sha256,
        "controller_generation":generation}),
    )
}

/// Saved counters are history; activity needs an unexpired process-bound lease.
/// File timestamps and reused PIDs alone never establish a current publisher.
pub fn observed_status(value: &serde_json::Value, observed_at: u64) -> serde_json::Value {
    let mut result = if value.is_object() {
        value.clone()
    } else {
        serde_json::json!({})
    };
    let binding = &value["status_binding"];
    let timestamp = value["timestamp"].as_u64();
    let expires = value["expires_at"].as_u64();
    let fresh = matches!((timestamp, expires), (Some(t), Some(e))
        if t <= observed_at && e == t.saturating_add(10) && observed_at < e);
    // A live publisher thread cannot keep old worker counts current while the
    // controller is stalled. The dispatch loop must renew its observation too.
    let activity_fresh = value["last_progress_at"]
        .as_u64()
        .is_some_and(|t| t <= observed_at && observed_at < t.saturating_add(10));
    let process: Option<crate::eval_registry::ProcessIdentity> =
        serde_json::from_value(binding["process"].clone()).ok();
    let alive = process.is_some_and(|p| {
        crate::eval_registry::ProcessIdentity::read(p.pid)
            .ok()
            .as_ref()
            == Some(&p)
            && executable_observation(p.pid).ok().as_ref() == Some(&binding["executable"])
            && fs::read_to_string(format!("/proc/{}/stat", p.pid))
                .ok()
                .and_then(|stat| {
                    stat.rsplit_once(')')
                        .map(|(_, tail)| tail.split_whitespace().next().unwrap_or("").to_owned())
                })
                .is_some_and(|state| state != "Z" && state != "X")
    });
    let bound = binding["schema"] == "parallel.status_binding.v1"
        && binding["controller_generation"]
            .as_str()
            .is_some_and(|v| v.len() == 64)
        && binding["executable_sha256"]
            .as_str()
            .is_some_and(|v| v.len() == 64)
        && value["executable_sha256"]
            .as_str()
            .is_none_or(|v| binding["executable_sha256"] == v);
    let activity = if value["terminal"] == true {
        "terminal"
    } else if fresh && activity_fresh && alive && bound {
        "current"
    } else {
        "stale"
    };
    result["current_activity"] = activity.into();
    if activity != "current" {
        if !result["last_recorded_activity"].is_object() {
            result["last_recorded_activity"] = serde_json::json!({"phase":value["phase"],
                "active":value["active"],"running_evaluations":value["running_evaluations"],
                "evaluations_per_second":value["evaluations_per_second"],"active_rows":value["active_rows"]});
        }
        if activity == "stale" {
            result["phase"] = "stale".into();
        }
        for field in ["active", "running_evaluations", "evaluations_per_second"] {
            result[field] = serde_json::Value::Null;
        }
        result["active_rows"] = serde_json::json!([]);
    }
    result
}

#[cfg(test)]
mod heartbeat_tests {
    use super::*;
    #[test]
    fn expired_or_changed_process_activity_is_unknown_but_history_survives() {
        let mut value = serde_json::json!({"phase":"running","active":104,
            "timestamp":100,"expires_at":110,"last_progress_at":100,"completed_views":50801,
            "completed_evaluations":857000,"evaluations_per_second":158.1});
        value["status_binding"] = status_binding(&value).unwrap();
        assert_eq!(observed_status(&value, 109)["current_activity"], "current");
        for observed in [99, 110, 1000] {
            let stale = observed_status(&value, observed);
            assert_eq!(stale["current_activity"], "stale");
            assert!(stale["active"].is_null());
            assert_eq!(stale["completed_views"], 50801);
            assert_eq!(stale["last_recorded_activity"]["active"], 104);
            assert_eq!(observed_status(&stale, observed), stale);
        }
        for field in ["birth_ticks", "pid"] {
            let mut reused = value.clone();
            reused["status_binding"]["process"][field] = 1.into();
            assert_eq!(observed_status(&reused, 105)["current_activity"], "stale");
        }
        let mut replaced = value.clone();
        replaced["status_binding"]["executable"]["inode"] = 0.into();
        assert_eq!(observed_status(&replaced, 105)["current_activity"], "stale");
        let mut stalled = value.clone();
        stalled["timestamp"] = 200.into();
        stalled["expires_at"] = 210.into();
        assert_eq!(observed_status(&stalled, 205)["current_activity"], "stale");
        value.as_object_mut().unwrap().remove("status_binding");
        assert_eq!(observed_status(&value, 105)["current_activity"], "stale");
    }

    #[test]
    fn publisher_preserves_generation_and_records_terminal_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("status.json");
        let publisher = StageHeartbeat::with_interval(
            &path,
            serde_json::json!({"phase":"running"}),
            Duration::from_millis(10),
        )
        .unwrap();
        let original: serde_json::Value = read_json(&path).unwrap();
        publisher
            .publish(serde_json::json!({"phase":"completed","completed_views":42,
            "status_binding":{"controller_generation":"forged"}}))
            .unwrap();
        publisher.finish().unwrap();
        let saved: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(saved["status_binding"], original["status_binding"]);
        assert_eq!(saved["completed_views"], 42);
        assert_eq!(saved["terminal"], true);
        assert_eq!(saved["termination_reason"], "completed");
        assert_eq!(
            observed_status(&saved, now() + 100)["current_activity"],
            "terminal"
        );
        let publisher =
            StageHeartbeat::start(&path, serde_json::json!({"phase":"running"})).unwrap();
        drop(publisher);
        let saved: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(saved["terminal"], true);
        assert!(saved["termination_reason"]
            .as_str()
            .unwrap()
            .contains("reconcile"));
    }

    #[test]
    fn slow_stages_keep_liveness_separate_and_join_before_next_phase() {
        for phase in ["Fit and save predictors", "Pack newly scored programs"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("progress.json");
            let payload = dir.path().join("payload");
            fs::write(&payload, b"frozen fitted predictor or corpus").unwrap();
            let original = file_hash(&payload).unwrap();
            let stage = StageHeartbeat::with_interval(
                &path,
                serde_json::json!({"phase":phase}),
                Duration::from_millis(10),
            )
            .unwrap();
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
            assert_eq!(
                read_json::<serde_json::Value>(&path).unwrap()["phase"],
                "next"
            );
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
