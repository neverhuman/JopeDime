use super::*;

#[derive(Default)]
struct FakeGpuState {
    list_error: Option<NvmlError>,
    inits: u32,
    resets: u32,
}

#[derive(Default)]
struct FakeGpu {
    init_error: Option<NvmlError>,
    state: Arc<std::sync::Mutex<FakeGpuState>>,
    devices: Vec<FakeDevice>,
}
impl GpuQuery for FakeGpu {
    fn initialize(&mut self) -> Result<(), NvmlError> {
        self.state.lock().unwrap().inits += 1;
        match self.init_error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    fn reset(&mut self) {
        self.state.lock().unwrap().resets += 1;
    }
    fn device_uuids(&self) -> Result<Vec<String>, NvmlError> {
        match &self.state.lock().unwrap().list_error {
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
    let state = Arc::new(std::sync::Mutex::new(FakeGpuState::default()));
    let mut sensor = NvmlSensor::new(FakeGpu {
        init_error: Some(NvmlError::LibraryNotFound),
        state: state.clone(),
        ..FakeGpu::default()
    });
    assert_eq!(sensor.readings(t0), Err("missing_library".to_owned()));
    // Within the backoff window NVML is not re-initialized.
    assert_eq!(
        sensor.readings(t0 + Duration::from_millis(50)),
        Err("missing_library".to_owned())
    );
    assert_eq!(state.lock().unwrap().inits, 1);
    // After the backoff it retries; the driver is now present but has no devices.
    assert_eq!(sensor.readings(t0 + Duration::from_millis(150)), Ok(vec![]));
    assert_eq!(state.lock().unwrap().inits, 2);

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
    let state = Arc::new(std::sync::Mutex::new(FakeGpuState {
        list_error: Some(NvmlError::GpuLost),
        ..FakeGpuState::default()
    }));
    let mut sensor = NvmlSensor::new(FakeGpu {
        state: state.clone(),
        devices: vec![("GPU-a".into(), Ok(5), Ok(7))],
        ..FakeGpu::default()
    });
    assert_eq!(sensor.readings(t0), Err("device_lost".to_owned()));
    assert_eq!(state.lock().unwrap().resets, 1);
    state.lock().unwrap().list_error = None;
    assert_eq!(
        sensor.readings(t0 + Duration::from_millis(10)),
        Err("device_lost".to_owned())
    );
    let readings = sensor.readings(t0 + Duration::from_millis(150)).unwrap();
    assert_eq!(state.lock().unwrap().inits, 2);
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
    let (_dir, log, snapshot) = crate::events::tests::tests::fixture();
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
    let (_dir, log, snapshot) = crate::events::tests::tests::fixture();
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
    let (_dir, log, snapshot) = crate::events::tests::tests::fixture();
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
    let (_dir, log, snapshot) = crate::events::tests::tests::fixture();
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
