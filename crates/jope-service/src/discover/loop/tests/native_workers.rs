use super::*;

// Build a tiny real worker launcher against the just-built library. No mock
// scores, libtest stdout shim, protocol changes or extra Cargo target exist.
pub(super) fn native_worker(directory: &Path) -> PathBuf {
    native_worker_fault(directory, "")
}
pub(super) fn native_worker_fault(directory: &Path, fault: &str) -> PathBuf {
    let deps = std::env::var_os("JOPE_DSC_DEPENDENCY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        });
    let library = std::env::var_os("JOPE_DSC_SERVICE_RLIB")
        .map(PathBuf::from)
        .unwrap_or_else(|| deps.parent().unwrap().join("libjope_service.rlib"));
    assert!(
        library.is_file(),
        "build the current service with cargo build -p jope-service --all-targets --locked"
    );
    let source = directory.join("native-worker.rs");
    let binary = directory.join("native-worker");
    let pids = directory.join("worker-pids");
    fs::create_dir_all(&pids).unwrap();
    let fault=match fault {
        "timeout"=>"if request.fold_spec.get(\"refit_output\").is_some() { loop { std::hint::spin_loop(); } }",
        "memory"=>"if request.fold_spec.get(\"refit_output\").is_some() { let mut bytes=vec![0_u8;128*1024*1024]; for i in (0..bytes.len()).step_by(4096) { bytes[i]=1; } loop { std::hint::black_box(&bytes); std::thread::sleep(std::time::Duration::from_millis(10)); } }",
        "r2-timeout"=>"if request.attempt.slot == 0 { loop { std::hint::spin_loop(); } }",
        "r2-memory"=>"if request.attempt.slot == 0 { let mut bytes=vec![0_u8;128*1024*1024]; for i in (0..bytes.len()).step_by(4096) { bytes[i]=1; } loop { std::hint::black_box(&bytes); std::thread::sleep(std::time::Duration::from_millis(10)); } }",
        ""=>"",_=>panic!("unknown owned fixture fault"),
    };
    fs::write(&source,format!("fn main() {{ std::fs::write(std::path::Path::new({:?}).join(std::process::id().to_string()), b\"native\").unwrap(); jope_service::worker::serve(1, |request,matrix| {{ {} jope_service::discover::r#loop::evaluate_native(request,matrix) }}).unwrap(); }}",pids,fault)).unwrap();
    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&compiler)
        .arg("--edition=2021")
        .arg("--crate-name=dsc08_native_worker")
        .arg(&source)
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("--extern")
        .arg(format!("jope_service={}", library.display()))
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "exact current service artifact could not link native worker: {}\n{}",
        library.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
        let compiler_version = Command::new(&compiler).arg("-vV").output().unwrap();
        ensure_worker_receipt(&PathBuf::from(receipt), &library, &compiler_version.stdout);
    }
    binary
}

pub(super) fn ensure_worker_receipt(receipt: &Path, library: &Path, compiler: &[u8]) {
    fs::create_dir_all(receipt).unwrap();
    fs::write(
        receipt.join("native-worker-build.json"),
        serde_json::to_vec_pretty(&json!({
            "library": library, "sha256": digest(&fs::read(library).unwrap()),
            "compiler": String::from_utf8_lossy(compiler),
            "source": std::env::var("JOPE_BUILD_SOURCE_SHA").unwrap_or_default(),
        }))
        .unwrap(),
    )
    .unwrap();
}

#[cfg(unix)]
pub(super) fn live_native_workers(directory: &Path) -> usize {
    fs::read_dir(directory)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_string_lossy().parse::<u32>().ok())
        .filter(|pid| {
            let status = Command::new("/bin/kill")
                .args(["-0", &pid.to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            status.is_ok_and(|status| status.success())
        })
        .count()
}

pub(super) fn pool(binary: &Path, workers: usize) -> ProcessExecutor {
    ProcessExecutor::new(
        binary,
        workers,
        &binary.parent().unwrap().join("worker-scratch"),
    )
    .unwrap()
}
pub(super) fn run_store(home: &Path, p: &Plan, binary: &Path, workers: usize) -> (Value, Value) {
    let store = Arc::new(Store::open(home).unwrap());
    let mut d = Discovery::open(store, p.clone()).unwrap();
    let pool = pool(binary, workers);
    finish(&mut d, &NativeSpace, &pool, workers);
    #[cfg(unix)]
    assert_eq!(
        live_native_workers(&binary.parent().unwrap().join("worker-pids")),
        workers,
        "native worker process count while the pool is still alive"
    );
    assert_eq!(d.nondeterminism().count.0, 0);
    let projection = d.projection().unwrap();
    assert!(
        projection
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e["kind"] != "experiment.finished" || e["body"]["failure"].is_null()),
        "S2 proof requires zero worker failures"
    );
    (champion_snapshot(d.champion().unwrap()), projection)
}

fn champion_snapshot(champion: &Champion) -> Value {
    json!({
        "candidate": &champion.candidate,
        "loss": champion.loss,
        "value": champion.value,
        "predictions_digest": &champion.predictions_digest,
        "model": &champion.model,
    })
}

pub(super) fn same(label: &str, expected: &(Value, Value), actual: &(Value, Value)) {
    if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
        let receipt = PathBuf::from(receipt);
        fs::create_dir_all(&receipt).unwrap();
        fs::write(
            receipt.join("s2-baseline.json"),
            serde_json::to_vec_pretty(expected).unwrap(),
        )
        .unwrap();
        fs::write(
            receipt.join(format!("s2-{label}.json")),
            serde_json::to_vec_pretty(actual).unwrap(),
        )
        .unwrap();
    }
    assert_eq!(expected.0, actual.0, "{label}: champion/model identity");
    let expected = expected.1.as_array().unwrap();
    let actual = actual.1.as_array().unwrap();
    assert_eq!(expected.len(), actual.len(), "{label}: event count");
    for (index, (a, b)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(a, b, "{label}: first event difference at {index}");
    }
}

#[test]
#[ignore = "dedicated bounded-process gate; build jope-service library first and run explicitly"]
fn promoted_refits_enforce_worker_wall_and_rss_limits_without_engine_fitting() {
    struct RidgeOnly;
    impl SearchSpace for RidgeOnly {
        fn canonical(&self, s: &str) -> Result<String> {
            NativeSpace.canonical(s)
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok(vec![self.canonical("(pipe X (ridge))")?])
        }
        fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }
    let engine = std::process::id();
    for (fault, expected) in [
        ("timeout", WorkerFailure::Timeout),
        ("memory", WorkerFailure::MemoryLimit),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 1);
        p.evaluation_wall_ms = 1000;
        p.evaluation_rss_bytes = 64 * 1024 * 1024;
        let binary = native_worker_fault(directory.path(), fault);
        let workers = pool(&binary, 1);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
        finish(&mut d, &RidgeOnly, &workers, 1);
        assert_eq!(std::process::id(), engine);
        assert!(d.champion().is_none());
        let row = store
            .get(&RecordKey::Evaluation {
                run: p.run_id.clone(),
                number: 1,
            })
            .unwrap()
            .unwrap();
        assert!(
            row.data["cv_failure"].is_null(),
            "CV must succeed before the injected full-fit fault"
        );
        assert_eq!(
            row.data["scores"].as_array().unwrap().len(),
            3,
            "real CV measurements remain recorded"
        );
        assert_eq!(
            row.data["refit_failure"]["failure"],
            serde_json::to_value(expected).unwrap()
        );
        assert_eq!(row.data["refit_failure"]["attempt"]["slot"], 131072);
        assert_eq!(d.nondeterminism().count.0, 1);
        assert_eq!(
            d.nondeterminism().experiment_ids,
            vec![experiment(&p, 0, 0, 1).unwrap()]
        );
        assert!(!d
            .projection()
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "algorithm.accepted"));
        println!("bounded promoted refit {fault}: CV succeeded; isolated worker failed as {expected:?}; engine alive; limit receipt retained");
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "dedicated REL-07 R2 native worker/RSS gate; build the current service library first"]
fn rel07_r2_worker_faults_preserve_engine_memory_and_complete_search() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    struct TwoSeeds;
    impl SearchSpace for TwoSeeds {
        fn canonical(&self, dsl: &str) -> Result<String> {
            NativeSpace.canonical(dsl)
        }
        fn seeds(&self) -> Result<Vec<String>> {
            ["(pipe X (ridge))", "(pipe X (standardize) (ridge))"]
                .into_iter()
                .map(|dsl| self.canonical(dsl))
                .collect()
        }
        fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }
    struct Observed {
        inner: ProcessExecutor,
        failures: Mutex<Vec<(WorkerFailure, u64)>>,
    }
    impl Executor for Observed {
        fn evaluate(&self, request: &Request) -> Response {
            let start = Instant::now();
            let response = self.inner.evaluate(request);
            if let Some(failure) = response.failure {
                self.failures
                    .lock()
                    .unwrap()
                    .push((failure, start.elapsed().as_millis() as u64));
            }
            response
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            self.inner.fitted_model(request)
        }
        fn revoke(&self, attempt: &Attempt) {
            self.inner.revoke(attempt);
        }
    }

    // Warm an ordinary successful native run before establishing the RSS baseline.
    // This charges startup/store/allocator setup neither to the malicious candidate
    // nor to a fabricated extra memory allowance. The 10% limit stays exact.
    let warm = tempfile::tempdir().unwrap();
    let warm_plan = plan(warm.path(), 30, Effort::Fast, 1);
    let warm_worker = native_worker(warm.path());
    run_store(&warm.path().join("store"), &warm_plan, &warm_worker, 1);

    for (fault, expected) in [
        ("timeout", WorkerFailure::Timeout),
        ("memory", WorkerFailure::MemoryLimit),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 2);
        p.planned_workers = 1;
        p.evaluation_wall_ms = 1000;
        p.evaluation_rss_bytes = 96 * 1024 * 1024;
        let binary = native_worker_fault(directory.path(), &format!("r2-{fault}"));
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
        let observed = Observed {
            inner: pool(&binary, 1),
            failures: Mutex::new(vec![]),
        };
        let prime = d
            .request(
                99,
                0,
                1,
                &Candidate {
                    algorithm_id: "alg_00000099".into(),
                    dsl: NativeSpace.canonical("(pipe X (ridge))").unwrap(),
                },
            )
            .unwrap();
        assert_eq!(observed.inner.evaluate(&prime).failure, None);
        let engine = std::process::id();
        let baseline = crate::sampler::sample(engine).unwrap().1;
        assert!(baseline > 0);
        let peak = AtomicU64::new(baseline);
        let samples = AtomicU64::new(0);
        let stop = AtomicBool::new(false);
        let sampling_failed = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let observer = scope.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    match crate::sampler::sample(engine) {
                        Ok((_, rss)) if rss > 0 => {
                            peak.fetch_max(rss, Ordering::Relaxed);
                            samples.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {
                            sampling_failed.store(true, Ordering::Release);
                            break;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            });
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                finish(&mut d, &TwoSeeds, &observed, 1);
            }));
            stop.store(true, Ordering::Release);
            observer.join().unwrap();
            if let Err(panic) = outcome {
                std::panic::resume_unwind(panic);
            }
        });
        assert!(
            !sampling_failed.load(Ordering::Acquire),
            "RSS sensor unavailable"
        );
        let sample_count = samples.load(Ordering::Relaxed);
        let peak_rss = peak.load(Ordering::Relaxed);
        assert!(sample_count > 0);
        assert!(
            peak_rss * 10 <= baseline * 11,
            "engine RSS exceeded 110% of warmed baseline: {peak_rss}/{baseline}"
        );
        assert!(
            d.completed().is_some(),
            "search must complete after the isolated fault"
        );
        let champion = d
            .champion()
            .expect("healthy candidate must still produce a champion");
        assert!(champion.value.is_finite() && champion.loss.is_finite());
        store.verify_blob(&champion.model).unwrap();
        assert_eq!(d.nondeterminism().count.0, 1);
        let failures = observed.failures.lock().unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, expected);
        let elapsed_ms = failures[0].1;
        assert!(
            elapsed_ms <= p.evaluation_wall_ms + 2000,
            "worker fault exceeded its limit plus two seconds"
        );
        drop(failures);
        let pids = directory.path().join("worker-pids");
        let workers_started = fs::read_dir(&pids).unwrap().count();
        assert!(workers_started >= 2, "faulted worker must be replaced");
        drop(observed);
        let workers_remaining = live_native_workers(&pids);
        assert_eq!(workers_remaining, 0, "owned worker cleanup must finish");
        println!(
            "REL07_R2 {}",
            json!({
                "schema":"jope.rel07.worker-isolation/v1", "fault":fault,
                "failure":format!("{expected:?}"), "wall_limit_ms":p.evaluation_wall_ms,
                "elapsed_ms":elapsed_ms, "baseline_rss_bytes":baseline,
                "peak_rss_bytes":peak_rss, "rss_samples":sample_count,
                "completed":true, "champion_present":true,
                "workers_started":workers_started, "workers_remaining":workers_remaining,
                "train_sha256":TRAIN_SHA,
                "source_sha":std::env::var("JOPE_BUILD_SOURCE_SHA").unwrap_or_default()
            })
        );
    }
}
