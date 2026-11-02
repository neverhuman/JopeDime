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

#[derive(Default)]
struct FakeGpu {
    init_error: Option<NvmlError>,
    list_error: Option<NvmlError>,
    devices: Vec<FakeDevice>,
    inits: u32,
    resets: u32,
}
impl GpuQuery for FakeGpu {
    fn initialize(&mut self) -> Result<(), NvmlError> {
        self.inits += 1;
        match self.init_error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    fn reset(&mut self) {
        self.resets += 1;
    }
    fn device_uuids(&self) -> Result<Vec<String>, NvmlError> {
        match &self.list_error {
            Some(_) => Err(NvmlError::GpuLost),
            None => Ok(self.devices.iter().map(|d| d.0.clone()).collect()),
        }
    }
    fn utilization_pct(&self, uuid: &str) -> Result<u32, NvmlError> {
        match &self.devices.iter().find(|d| d.0 == uuid).unwrap().1 {
            Ok(v) => Ok(*v),
            Err(_) => Err(NvmlError::NoPermission),
        }
    }
    fn memory_used_bytes(&self, uuid: &str) -> Result<u64, NvmlError> {
        match &self.devices.iter().find(|d| d.0 == uuid).unwrap().2 {
            Ok(v) => Ok(*v),
            Err(_) => Err(NvmlError::GpuLost),
        }
    }
}

#[test]
fn missing_nvml_is_unavailable_with_bounded_retry_and_never_panics() {
    let t0 = Instant::now();
    let mut sensor = NvmlSensor::new(FakeGpu {
        init_error: Some(NvmlError::LibraryNotFound),
        ..FakeGpu::default()
    });
    assert_eq!(sensor.readings(t0), Err("missing_library".to_owned()));
    // Within the backoff window NVML is not re-initialized.
    assert_eq!(
        sensor.readings(t0 + Duration::from_millis(50)),
        Err("missing_library".to_owned())
    );
    assert_eq!(sensor.query.inits, 1);
    // After the backoff it retries; the driver is now present but has no devices.
    assert_eq!(sensor.readings(t0 + Duration::from_millis(150)), Ok(vec![]));
    assert_eq!(sensor.query.inits, 2);

    for (error, reason) in [
        (NvmlError::DriverNotLoaded, "missing_driver"),
        (NvmlError::NoPermission, "permission_denied"),
        (NvmlError::Uninitialized, "nvml_error"),
    ] {
        let mut sensor = NvmlSensor::new(FakeGpu {
            init_error: Some(error),
            ..FakeGpu::default()
        });
        assert_eq!(sensor.readings(t0), Err(reason.to_owned()));
    }
}

#[test]
fn nvml_readings_are_device_scoped_and_per_device_failures_carry_a_reason() {
    let t0 = Instant::now();
    let mut sensor = NvmlSensor::new(FakeGpu {
        devices: vec![
            ("GPU-a".into(), Ok(37), Ok(1 << 30)),
            (
                "GPU-b".into(),
                Err(NvmlError::NoPermission),
                Err(NvmlError::GpuLost),
            ),
        ],
        ..FakeGpu::default()
    });
    let readings = sensor.readings(t0).unwrap();
    assert_eq!(readings[0].util_pct, Ok(37.0));
    assert_eq!(readings[0].mem_used_bytes, Ok(1 << 30));
    assert_eq!(readings[1].util_pct, Err("permission_denied".to_owned()));
    assert_eq!(readings[1].mem_used_bytes, Err("device_lost".to_owned()));

    let mut fake = Fake {
        gpus: Some(readings),
        ..Fake::default()
    };
    let s = Telemetry::new(ENGINE, t0).tick(t0, &mut fake, &Counters::default());
    let device = |name: &str, scope: &str| {
        s.iter()
            .find(|m| m.name == name && m.scope == scope)
            .unwrap()
            .clone()
    };
    assert_eq!(num(&device("gpu.util_pct", "device:GPU-a")), 37.0);
    assert_eq!(
        device("gpu.util_pct", "device:GPU-b").availability,
        "unavailable:permission_denied"
    );
    assert_eq!(
        device("gpu.mem_used", "device:GPU-b").availability,
        "unavailable:device_lost"
    );
}

#[test]
fn a_lost_device_resets_nvml_and_reinitializes_after_backoff() {
    let t0 = Instant::now();
    let mut sensor = NvmlSensor::new(FakeGpu {
        list_error: Some(NvmlError::GpuLost),
        devices: vec![("GPU-a".into(), Ok(5), Ok(7))],
        ..FakeGpu::default()
    });
    assert_eq!(sensor.readings(t0), Err("device_lost".to_owned()));
    assert_eq!(sensor.query.resets, 1);
    sensor.query.list_error = None;
    assert_eq!(
        sensor.readings(t0 + Duration::from_millis(10)),
        Err("device_lost".to_owned())
    );
    let readings = sensor.readings(t0 + Duration::from_millis(150)).unwrap();
    assert_eq!(sensor.query.inits, 2);
    assert_eq!(readings[0].util_pct, Ok(5.0));
}

#[test]
fn gpu_less_system_sensors_report_a_reason() {
    let store = test_store();
    let mut sensors = SystemSensors::without_gpu(store.path());
    assert_eq!(sensors.gpus(), Err("gpu_sampling_disabled".to_owned()));
}

#[test]
fn tokens_per_second_uses_tel07_counts_and_registry_names() {
    let t0 = Instant::now();
    let counters = Counters::default();
    counters.add_tokens(TokenCounts {
        rust_tokens: 600,
        rust_seconds: 1.0,
        rust_timed_alone: true,
        all_tokens: 2000,
        all_seconds: 3.0,
    });
    counters.add_tokens(TokenCounts {
        rust_tokens: 400,
        rust_seconds: 1.0,
        rust_timed_alone: false,
        all_tokens: 1000,
        all_seconds: 1.0,
    });
    let mut fake = Fake::default();
    let s = Telemetry::new(ENGINE, t0).tick(t0, &mut fake, &counters);
    assert_eq!(num(get(&s, TOK_S_RUST)), 500.0);
    assert_eq!(num(get(&s, TOK_S_ALL_LANGUAGES)), 750.0);
    assert_eq!(
        get(&s, TOK_S_RUST_TIMED_ALONE).value,
        Some(MetricValue::Boolean(false))
    );
    assert_eq!(get(&s, TOK_S_RUST).unit, "tokens/s");
}

#[test]
fn registry_names_are_the_jope_telemetry_constants() {
    assert_eq!(TOK_S_RUST, "throughput.tokens_per_s.rust");
    assert_eq!(TOK_S_ALL_LANGUAGES, "throughput.tokens_per_s.all_languages");
    assert_eq!(
        TOK_S_RUST_TIMED_ALONE,
        "throughput.tokens_per_s.rust_timed_alone"
    );
}

#[test]
fn counters_feed_rates_times_and_reported_rows() {
    let t0 = Instant::now();
    let counters = Counters::default();
    let mut fake = Fake {
        process: [(ENGINE, reading(0.0, 1))].into(),
        gpus: Some(vec![GpuReading {
            uuid: "GPU-abc".into(),
            util_pct: Ok(37.5),
            mem_used_bytes: Ok(1 << 30),
        }]),
        ..Fake::default()
    };
    let mut telemetry = Telemetry::new(ENGINE, t0);
    telemetry.tick(t0, &mut fake, &counters);
    for _ in 0..6 {
        counters.rung0_completed();
    }
    counters.evaluation_failed("timeout");
    counters.evaluation_failed("timeout");
    counters.evaluation_failed("memory_limit");
    counters.champion_committed(t0 + Duration::from_millis(1500));
    counters.champion_committed(t0 + Duration::from_millis(2500));
    counters.report(
        Reported::RankerScoresPerS,
        1234.0,
        t0 + Duration::from_millis(2000),
    );
    counters.report_store_bytes(4096, t0 + Duration::from_millis(2000));
    let s = telemetry.tick(t0 + Duration::from_secs(3), &mut fake, &counters);
    assert_eq!(num(get(&s, "eval.per_s")), 2.0);
    assert_eq!(get(&s, "eval.failed").value, Some(bytes(3)));
    assert_eq!(get(&s, "eval.failed.timeout").value, Some(bytes(2)));
    assert_eq!(num(get(&s, "time.first_model_s")), 1.5);
    assert_eq!(num(get(&s, "time.best_s")), 2.5);
    let ranker = get(&s, "ranker.scores_per_s");
    assert_eq!((num(ranker), ranker.age_ms), (1234.0, 1000));
    assert_eq!(get(&s, "store.bytes").value, Some(bytes(4096)));
    let gpu = get(&s, "gpu.util_pct");
    assert_eq!((gpu.scope.as_str(), num(gpu)), ("device:GPU-abc", 37.5));
    assert_eq!(get(&s, "gpu.mem_used").value, Some(bytes(1 << 30)));
    let json = telemetry.summary().to_json();
    assert_eq!(json["evaluations_rung0"], 6);
    assert_eq!(json["evaluations_failed"]["timeout"], 2);
}

#[test]
fn service_publishes_ephemeral_frames_and_returns_a_summary() {
    let (_dir, log, snapshot) = crate::events::tests::fixture();
    let mut frames = log.subscribe_telemetry();
    let counters = Arc::new(Counters::default());
    let fake = Fake {
        process: [(std::process::id(), reading(0.0, 4096))].into(),
        ..Fake::default()
    };
    let service = TelemetryService::spawn(
        snapshot.run_id.clone(),
        log.clone(),
        Box::new(fake),
        counters,
        Instant::now(),
        Duration::from_millis(5),
    );
    let mut received = 0;
    let deadline = Instant::now() + Duration::from_secs(10);
    while received < 3 && Instant::now() < deadline {
        match frames.try_recv() {
            Ok(frame) => {
                assert_eq!(frame.run_id, snapshot.run_id);
                let TelemetryBody::Telemetry(samples) = frame.event;
                assert!(samples.iter().any(|s| s.name == "cpu.cores"));
                received += 1;
            }
            Err(_) => thread::sleep(Duration::from_millis(2)),
        }
    }
    let summary = service.stop();
    assert!(received >= 3, "received {received} frames");
    assert!(summary.samples >= 3);
    assert_eq!(summary.mem_rss_peak_bytes, Some(4096));
}

#[test]
fn delayed_attachment_measures_times_from_the_canonical_run_start() {
    let (_dir, log, snapshot) = crate::events::tests::fixture();
    let mut frames = log.subscribe_telemetry();
    let counters = Arc::new(Counters::default());
    let run_started = Instant::now();
    // The run started, a champion was committed, and only then telemetry attached.
    let committed = run_started + Duration::from_millis(40);
    counters.champion_committed(committed);
    while Instant::now() < committed + Duration::from_millis(20) {
        thread::sleep(Duration::from_millis(5));
    }
    let fake = Fake {
        process: [(std::process::id(), reading(0.0, 1))].into(),
        ..Fake::default()
    };
    let service = TelemetryService::spawn(
        snapshot.run_id,
        log,
        Box::new(fake),
        counters,
        run_started,
        Duration::from_millis(5),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let first = loop {
        match frames.try_recv() {
            Ok(frame) => break frame,
            Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
            Err(error) => panic!("no telemetry frame: {error:?}"),
        }
    };
    let summary = service.stop();
    let TelemetryBody::Telemetry(samples) = first.event;
    let first_model = num(get(&samples, "time.first_model_s"));
    assert!(
        (first_model - 0.040).abs() < 1e-9,
        "first_model_s {first_model}"
    );
    assert_eq!(summary.time_first_model_s, Some(first_model));

    // A mark before the run start is refused rather than published as a measured 0.
    let before = Counters::default();
    before.champion_committed(run_started);
    let mut fake = Fake::default();
    let s = Telemetry::new(ENGINE, run_started + Duration::from_secs(1)).tick(
        run_started + Duration::from_secs(2),
        &mut fake,
        &before,
    );
    assert_eq!(
        get(&s, "time.first_model_s").availability,
        "unavailable:before_run_start"
    );
}

/// Gate T3: telemetry on or off yields the identical champion and committed history.
#[test]
fn telemetry_on_or_off_leaves_the_search_identical() {
    use crate::run::{Control, Evaluator, FakeEvaluator, RunController};
    use jope_api::{Effort, RunPhase};
    use std::num::NonZeroU32;
    fn search() -> Vec<String> {
        let mut run = RunController::new(
            "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            Effort::Medium,
            NonZeroU32::new(2).unwrap(),
        );
        for phase in [RunPhase::Encoding, RunPhase::Seeding, RunPhase::Searching] {
            run.advance_phase(phase).unwrap();
        }
        let fake = FakeEvaluator::new(Duration::ZERO);
        for round in 0..20u64 {
            let requests: Vec<_> = (0..3u32)
                .map(|slot| {
                    run.dispatch(round, slot, 31 * round + u64::from(slot))
                        .unwrap()
                })
                .collect();
            for request in &requests {
                run.receive(fake.evaluate(request)).unwrap();
            }
            run.round_committed().unwrap();
        }
        run.control(Control::Cancel, None).unwrap();
        run.committed().iter().map(|r| format!("{r:?}")).collect()
    }
    let off = search();
    let (_dir, log, snapshot) = crate::events::tests::fixture();
    let mut frames = log.subscribe_telemetry();
    let store = test_store();
    let service = TelemetryService::spawn(
        snapshot.run_id,
        log,
        Box::new(SystemSensors::without_gpu(store.path())),
        Arc::new(Counters::default()),
        Instant::now(),
        Duration::from_millis(1),
    );
    // The zero-duration fake search can finish before the sampler is scheduled.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match frames.try_recv() {
            Ok(_) => break,
            Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
            Err(error) => panic!("telemetry did not start: {error:?}"),
        }
    }
    let on = search();
    let summary = service.stop();
    assert!(summary.samples > 0);
    assert_eq!(off, on);
}

// Reasoning-mode neutrality is not currently observable in a production run:
// Config::defaults.reasoning is parsed but RunController and the telemetry
// observer do not consume it. Do not count the former config-only test as T3
// evidence; add a real run-level test when reasoning is wired into a consumer.

/// Informational T3 measurement; owner direction forbids a speed verdict.
#[test]
fn real_sensor_tick_cost_measurement() {
    let counters = Counters::default();
    counters.set_workers(&[std::process::id()]);
    let store = test_store();
    let mut sensors = SystemSensors::without_gpu(store.path());
    let mut telemetry = Telemetry::new(std::process::id(), Instant::now());
    let ticks = 200;
    let start = Instant::now();
    for _ in 0..ticks {
        std::hint::black_box(telemetry.tick(Instant::now(), &mut sensors, &counters));
    }
    let per_tick = start.elapsed() / ticks;
    println!("informational mean telemetry tick {per_tick:?}; interval {DEFAULT_INTERVAL:?}");
}

/// Informational fixed-work wall-time measurement for T3; no speed verdict.
/// `cargo test -p jope-service --lib telemetry::tests::fixed_work_overhead_measurement -- --ignored --nocapture`.
#[test]
#[ignore]
fn fixed_work_overhead_measurement() {
    fn work() -> Duration {
        let start = Instant::now();
        let mut state = 0x9e3779b97f4a7c15u64;
        for i in 0..400_000_000u64 {
            state = (state ^ i).wrapping_mul(0xbf58476d1ce4e5b9).rotate_left(17);
        }
        std::hint::black_box(state);
        start.elapsed()
    }
    work();
    let off: Duration = (0..3).map(|_| work()).min().unwrap();
    let (_dir, log, snapshot) = crate::events::tests::fixture();
    let store = test_store();
    let service = TelemetryService::spawn(
        snapshot.run_id,
        log,
        Box::new(SystemSensors::without_gpu(store.path())),
        Arc::new(Counters::default()),
        Instant::now(),
        DEFAULT_INTERVAL,
    );
    let on: Duration = (0..3).map(|_| work()).min().unwrap();
    service.stop();
    let overhead = on.as_secs_f64() / off.as_secs_f64() - 1.0;
    println!(
        "fixed work off {off:?} on {on:?} overhead {:.3}%",
        overhead * 100.0
    );
}
