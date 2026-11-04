use super::*;

/// A fresh, empty store directory removed on drop.
fn test_store() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("jope-telemetry-")
        .tempdir()
        .unwrap()
}

const ENGINE: u32 = 1;

#[cfg(target_os = "macos")]
#[test]
fn macos_system_memory_sensor_reports_available_bytes() {
    let available = platform::mem_available().expect("native Mach VM memory sensor");
    assert!(available > 0, "available memory must be nonzero");
}

#[cfg(target_os = "linux")]
#[test]
fn linux_affinity_and_disk_sensors_read_successful_outputs() {
    let allowed = platform::allowed_cpus().expect("native CPU affinity sensor");
    assert!(allowed > 0.0, "at least one CPU must be available");
    let _free = platform::disk_free(std::path::Path::new(".")).expect("native statvfs disk sensor");
}

#[cfg(target_os = "macos")]
#[test]
fn macos_disk_sensor_reads_successful_statvfs_output() {
    let _free = platform::disk_free(std::path::Path::new(".")).expect("native statvfs disk sensor");
}

#[derive(Clone, Default)]
struct Fake {
    process: BTreeMap<u32, ProcessReading>,
    logical: Option<f64>,
    allowed: Option<f64>,
    mem: Option<u64>,
    disk: Option<u64>,
    gpus: Option<Vec<GpuReading>>,
}
impl Sensors for Fake {
    fn process(&mut self, pid: u32) -> Result<ProcessReading, String> {
        self.process
            .get(&pid)
            .copied()
            .ok_or_else(|| "process_exited".to_owned())
    }
    fn logical_cpus(&mut self) -> Result<f64, String> {
        self.logical.ok_or_else(|| "no_logical".to_owned())
    }
    fn allowed_cpus(&mut self) -> Result<f64, String> {
        self.allowed.ok_or_else(|| "no_allowed".to_owned())
    }
    fn mem_available(&mut self) -> Result<u64, String> {
        self.mem.ok_or_else(|| "meminfo_unavailable".to_owned())
    }
    fn disk_free(&mut self) -> Result<u64, String> {
        self.disk.ok_or_else(|| "statvfs_failed".to_owned())
    }
    fn gpus(&mut self) -> Result<Vec<GpuReading>, String> {
        self.gpus
            .clone()
            .ok_or_else(|| "fake_gpu_absent".to_owned())
    }
}
fn get<'a>(samples: &'a [MetricSample], name: &str) -> &'a MetricSample {
    samples
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("{name} missing"))
}
fn num(sample: &MetricSample) -> f64 {
    match sample.value {
        Some(MetricValue::Number(n)) => n,
        ref other => panic!("{} not a number: {other:?}", sample.name),
    }
}
fn reading(cpu: f64, rss: u64) -> ProcessReading {
    ProcessReading {
        cpu_seconds: cpu,
        rss_bytes: rss,
    }
}

#[test]
fn cpu_math_follows_f06_without_clamping() {
    let t0 = Instant::now();
    let counters = Counters::default();
    for (cpu_per_s, cores, host_pct) in [(1.0, 1.00, 12.5), (4.0, 4.00, 50.0), (16.0, 16.0, 200.0)]
    {
        let mut fake = Fake {
            logical: Some(8.0),
            allowed: Some(4.0),
            ..Fake::default()
        };
        let mut telemetry = Telemetry::new(ENGINE, t0);
        fake.process.insert(ENGINE, reading(10.0, 1));
        let first = telemetry.tick(t0, &mut fake, &counters);
        assert_eq!(
            get(&first, "cpu.cores").availability,
            "unavailable:no_previous_sample"
        );
        fake.process
            .insert(ENGINE, reading(10.0 + 2.0 * cpu_per_s, 1));
        let second = telemetry.tick(t0 + Duration::from_secs(2), &mut fake, &counters);
        assert!((num(get(&second, "cpu.cores")) - cores).abs() < 1e-9);
        assert!((num(get(&second, "cpu.host_pct")) - host_pct).abs() < 1e-9);
        assert!((num(get(&second, "cpu.allowed_pct")) - 100.0 * cores / 4.0).abs() < 1e-9);
        assert_eq!(get(&second, "cpu.cores").window_ms, 2000);
    }
}

#[test]
fn rss_sums_engine_and_live_workers_and_new_workers_baseline() {
    let t0 = Instant::now();
    let counters = Counters::default();
    let mut fake = Fake::default();
    let mut telemetry = Telemetry::new(ENGINE, t0);
    fake.process.insert(ENGINE, reading(0.0, 100));
    fake.process.insert(2, reading(0.0, 10));
    counters.set_workers(&[2]);
    telemetry.tick(t0, &mut fake, &counters);
    // Worker 2 exits; worker 3 appears with 50 s of history that must not count as a spike.
    fake.process.remove(&2);
    fake.process.insert(3, reading(50.0, 20));
    fake.process.insert(ENGINE, reading(1.0, 100));
    counters.set_workers(&[3]);
    let s = telemetry.tick(t0 + Duration::from_secs(1), &mut fake, &counters);
    assert!((num(get(&s, "cpu.cores")) - 1.0).abs() < 1e-9);
    assert_eq!(get(&s, "mem.rss").value, Some(bytes(120)));
    fake.process.insert(3, reading(51.0, 20));
    fake.process.insert(ENGINE, reading(2.0, 100));
    let s = telemetry.tick(t0 + Duration::from_secs(2), &mut fake, &counters);
    assert!((num(get(&s, "cpu.cores")) - 2.0).abs() < 1e-9);
    let summary = telemetry.summary();
    assert_eq!(summary.mem_rss_peak_bytes, Some(120));
    assert_eq!(summary.cpu_cores_peak, Some(2.0));
    assert_eq!(summary.cpu_cores_mean, Some(1.5));
}

#[test]
fn missing_sensors_produce_null_with_a_reason_never_zero() {
    let t0 = Instant::now();
    let counters = Counters::default();
    let mut fake = Fake::default();
    let mut telemetry = Telemetry::new(ENGINE, t0);
    let s = telemetry.tick(t0, &mut fake, &counters);
    for metric in &s {
        if metric.value.is_none() {
            assert!(
                metric.availability.starts_with("unavailable:"),
                "{metric:?}"
            );
            assert!(metric.availability.len() > "unavailable:".len());
        } else {
            assert_eq!(metric.availability, "measured");
        }
    }
    let reason = |name: &str| get(&s, name).availability.clone();
    assert_eq!(reason("cpu.cores"), "unavailable:engine_process_exited");
    assert_eq!(reason("mem.rss"), "unavailable:engine_process_exited");
    assert_eq!(reason("mem.available"), "unavailable:meminfo_unavailable");
    assert_eq!(reason("disk.free"), "unavailable:statvfs_failed");
    assert_eq!(reason("gpu.util_pct"), "unavailable:fake_gpu_absent");
    assert_eq!(reason("ranker.scores_per_s"), "unavailable:not_reported");
    assert_eq!(reason("store.bytes"), "unavailable:not_reported");
    assert_eq!(reason(TOK_S_RUST), "unavailable:no_text_token_path");
    assert_eq!(
        reason(TOK_S_ALL_LANGUAGES),
        "unavailable:no_text_token_path"
    );
    assert_eq!(
        reason(TOK_S_RUST_TIMED_ALONE),
        "unavailable:no_text_token_path"
    );
    assert_eq!(reason("time.first_model_s"), "unavailable:no_model_yet");
    assert_eq!(reason("eval.per_s"), "unavailable:no_previous_sample");
    let summary = telemetry.summary();
    assert_eq!(
        summary.unavailable["gpu.mem_used"],
        "unavailable:fake_gpu_absent"
    );
    assert_eq!(summary.tokens_per_s_rust, None);
}

#[test]
fn a_failed_live_worker_makes_totals_unavailable_not_falsely_complete() {
    let t0 = Instant::now();
    let counters = Counters::default();
    let mut fake = Fake::default();
    let mut telemetry = Telemetry::new(ENGINE, t0);
    fake.process.insert(ENGINE, reading(0.0, 100));
    fake.process.insert(2, reading(0.0, 10));
    counters.set_workers(&[2]);
    telemetry.tick(t0, &mut fake, &counters);
    // Worker 2 is still listed as live but its reading fails.
    fake.process.remove(&2);
    fake.process.insert(ENGINE, reading(1.0, 100));
    let s = telemetry.tick(t0 + Duration::from_secs(1), &mut fake, &counters);
    for name in ["cpu.cores", "cpu.host_pct", "mem.rss"] {
        let metric = get(&s, name);
        assert_eq!(metric.value, None, "{name}");
        assert_eq!(
            metric.availability, "unavailable:worker_process_exited",
            "{name}"
        );
    }
    assert_eq!(telemetry.summary().mem_rss_peak_bytes, Some(110));
}

#[test]
fn store_bytes_are_exact_and_invalid_reports_are_unavailable() {
    let t0 = Instant::now();
    let counters = Counters::default();
    let exact = (1u64 << 53) + 1; // not representable as f64
    counters.report_store_bytes(exact, t0);
    counters.report(Reported::RankerLatencyMs, -3.0, t0);
    counters.report(Reported::IoReadGbS, f64::INFINITY, t0);
    counters.report(Reported::PredictRowsPerS, f64::NAN, t0);
    let mut fake = Fake::default();
    let s = Telemetry::new(ENGINE, t0).tick(t0, &mut fake, &counters);
    assert_eq!(get(&s, "store.bytes").value, Some(bytes(exact)));
    assert_eq!(
        get(&s, "ranker.latency_ms").availability,
        "unavailable:negative_value"
    );
    assert_eq!(
        get(&s, "io.read_gb_s").availability,
        "unavailable:non_finite"
    );
    assert_eq!(
        get(&s, "predict.rows_per_s").availability,
        "unavailable:non_finite"
    );
    for name in ["ranker.latency_ms", "io.read_gb_s", "predict.rows_per_s"] {
        assert_eq!(get(&s, name).value, None);
    }
}

#[test]
fn allowed_cpus_follow_the_tightest_quota_on_the_processes_own_cgroup_path() {
    let files: BTreeMap<&str, &str> = [
        ("", "max 100000\n"),
        ("system.slice", "800000 100000\n"),
        ("system.slice/jope.service", "250000 100000\n"),
        ("system.slice/jope.service/worker", "max 100000\n"),
    ]
    .into();
    let read = |path: &str| files.get(path).map(|s| s.to_string());
    let membership = "0::/system.slice/jope.service/worker\n";
    assert_eq!(cgroup_allowed_cpus(membership, 64.0, read), Ok(2.5));
    // Affinity can be tighter than any quota.
    assert_eq!(cgroup_allowed_cpus(membership, 2.0, read), Ok(2.0));
    // A sibling cgroup does not inherit an unrelated quota.
    assert_eq!(
        cgroup_allowed_cpus("0::/user.slice\n", 64.0, read),
        Ok(64.0)
    );
    // cgroup v1 only (no 0:: line): affinity alone.
    assert_eq!(
        cgroup_allowed_cpus("4:cpu,cpuacct:/x\n", 12.0, read),
        Ok(12.0)
    );
    // Root membership reads the root file only.
    assert_eq!(
        cgroup_allowed_cpus("0::/\n", 16.0, |p| (p.is_empty())
            .then(|| "400000 100000".to_string())),
        Ok(4.0)
    );
}

#[test]
fn invalid_cgroup_membership_never_reads_quota_outside_the_mount() {
    for path in [
        "/../outside",
        "/a/../../outside",
        "/a/../b",
        "/./a",
        "relative",
        "",
    ] {
        let reads = std::cell::Cell::new(0);
        let result = cgroup_allowed_cpus(&format!("0::{path}\n"), 8.0, |_| {
            reads.set(reads.get() + 1);
            Some("max 100000".to_owned())
        });
        assert_eq!(reads.get(), 0, "invalid cgroup membership was read: {path}");
        assert_eq!(result, Err("cgroup_path_unresolved".to_owned()));
    }
}

/// (uuid, utilization, memory used) for one fake device.
type FakeDevice = (String, Result<u32, NvmlError>, Result<u64, NvmlError>);

mod gpu_and_service;
