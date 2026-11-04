use super::super::types::BatchPreflight;
use super::*;

#[test]
fn batch_preflight_deduplicates_only_identical_successful_pins() {
    let scratch = scratch();
    let job = preflight_job(scratch.path());
    let mut preflight = BatchPreflight::default();
    preflight.verify(&job).unwrap();
    preflight.verify(&job).unwrap();
    assert_eq!(preflight.verified.len(), 2);
    let mut changed = job.clone();
    changed.command.inputs[0].sha256 = crate::sha256_hex(b"different expected bytes");
    assert!(preflight.verify(&changed).is_err());
    assert_eq!(preflight.verified.len(), 2);
    assert!(BatchPreflight::default().verify(&changed).is_err());
}
#[test]
fn full_gate_hash_rejects_mutation_despite_preflight_and_restored_mtime() {
    let scratch = scratch();
    let job = preflight_job(scratch.path());
    let input = &job.command.inputs[0].path;
    let modified = fs::metadata(input).unwrap().modified().unwrap();
    let mut preflight = BatchPreflight::default();
    preflight.verify(&job).unwrap();
    fs::write(input, b"altered input").unwrap();
    File::options()
        .write(true)
        .open(input)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    assert_eq!(fs::metadata(input).unwrap().modified().unwrap(), modified);
    // Inventory deduplication is deliberately not the integrity boundary.
    preflight.verify(&job).unwrap();
    assert!(job.verify_inputs().is_err());
    assert!(BatchPreflight::default().verify(&job).is_err());
}
#[test]
fn full_gate_hash_rejects_program_replacement_after_preflight() {
    let scratch = scratch();
    let job = preflight_job(scratch.path());
    let mut preflight = BatchPreflight::default();
    preflight.verify(&job).unwrap();
    let replacement = scratch.path().join("replacement");
    fs::write(&replacement, b"altered program").unwrap();
    fs::rename(replacement, &job.command.program.path).unwrap();
    preflight.verify(&job).unwrap();
    assert!(job.verify_inputs().is_err());
}
#[test]
fn pre_gate_expiry_is_terminal_with_unknown_setup_cost_and_no_fit_claim() {
    let job = fixture_job('a', 'b');
    let started = Started {
        job_sha256: job.identity().unwrap(),
        job,
        started_unix_ms: 100,
        fit_deadline_unix_ms: 120,
        cgroup: PathBuf::from("/synthetic/not-used"),
    };
    let before = pre_gate_failure(&started, 119, "synthetic missing input".into()).unwrap();
    assert_eq!(before.command_status, EvaluationStatus::Unavailable);
    for now in [120, 121] {
        let after = pre_gate_failure(&started, now, "fit expired before launch".into()).unwrap();
        assert_eq!(after.command_status, EvaluationStatus::BudgetExhausted);
        assert!(after.consumption.gate_released_unix_ms.is_none());
        assert!(after.consumption.cpu_seconds.is_none());
        assert!(after.exit_code.is_none());
        assert!(after.outputs.is_empty());
        assert_eq!(after.job_sha256, before.job_sha256);
    }
}

fn setup_record(root: &Path, recipe: char, deadline: u64) -> Started {
    let job = fixture_job('a', recipe);
    let id = job.identity().unwrap();
    Started {
        job,
        job_sha256: id.clone(),
        started_unix_ms: now_ms().unwrap(),
        fit_deadline_unix_ms: deadline,
        cgroup: root.join(format!("fit-{id}")),
    }
}
#[test]
fn bounded_watchers_enter_setup_concurrently_before_either_finishes() {
    use std::sync::mpsc;
    let scratch = scratch();
    let (entered, notices) = mpsc::channel();
    let mut fits = Vec::new();
    let mut release = Vec::new();
    for (cpu, recipe) in [(2, 'b'), (3, 'c')] {
        let started = setup_record(scratch.path(), recipe, now_ms().unwrap() + 5000);
        let path = scratch.path().join(format!("started-{cpu}.json"));
        let entered = entered.clone();
        let (send, wait) = mpsc::channel();
        release.push(send);
        fits.push(
            Running::spawn(started, cpu, path, move |_| {
                entered.send(cpu).unwrap();
                wait.recv_timeout(Duration::from_secs(2)).unwrap();
                anyhow::bail!("synthetic setup failure; no process/cgroup invoked")
            })
            .unwrap(),
        );
    }
    let first = notices.recv_timeout(Duration::from_secs(1)).unwrap();
    let second = notices.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_ne!(first, second);
    assert_eq!(
        fits.iter().map(|f| f.cpu).collect::<BTreeSet<_>>(),
        BTreeSet::from([2, 3])
    );
    assert!(fits.iter().all(|f| !f.is_done()));
    for send in release {
        send.send(()).unwrap();
    }
    for fit in fits {
        let outcome = fit.finish().unwrap();
        assert_eq!(outcome.command_status, EvaluationStatus::Unavailable);
        assert!(outcome.consumption.gate_released_unix_ms.is_none());
        assert!(outcome.consumption.cpu_seconds.is_none());
    }
}
#[test]
fn async_setup_failure_preserves_original_expiry_without_gate() {
    let scratch = scratch();
    let started = setup_record(scratch.path(), 'b', now_ms().unwrap().saturating_sub(1));
    let path = scratch.path().join("not-started.json");
    let running = Running::spawn(started, 2, path.clone(), |_| {
        anyhow::bail!("synthetic setup failed at original cutoff")
    })
    .unwrap();
    let outcome = running.finish().unwrap();
    assert_eq!(outcome.command_status, EvaluationStatus::BudgetExhausted);
    assert!(outcome.consumption.gate_released_unix_ms.is_none());
    assert!(outcome.exit_code.is_none());
    assert!(!path.exists());
}
#[test]
fn dropping_setup_slot_signals_and_joins_its_watcher() {
    use std::sync::mpsc;
    let scratch = scratch();
    let started = setup_record(scratch.path(), 'b', now_ms().unwrap() + 5000);
    let (entered, notice) = mpsc::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let observed = stopped.clone();
    let running = Running::spawn(
        started,
        2,
        scratch.path().join("not-started.json"),
        move |stop| {
            entered.send(()).unwrap();
            let limit = Instant::now() + Duration::from_secs(2);
            while !stop.load(Ordering::Acquire) {
                ensure!(
                    Instant::now() < limit,
                    "setup drop did not signal cancellation"
                );
                thread::sleep(Duration::from_millis(1));
            }
            observed.store(true, Ordering::Release);
            anyhow::bail!("synthetic setup observed stop before any command")
        },
    )
    .unwrap();
    notice.recv_timeout(Duration::from_secs(1)).unwrap();
    drop(running);
    assert!(stopped.load(Ordering::Acquire));
}
#[test]
fn pre_gate_setup_cancellation_does_not_reset_or_override_expired_cutoff() {
    let scratch = scratch();
    let started = setup_record(scratch.path(), 'b', 120);
    let cancelled = pre_gate_cancellation(&started, 119).unwrap();
    assert_eq!(cancelled.command_status, EvaluationStatus::Cancelled);
    assert!(cancelled.exit_code.is_none());
    assert!(cancelled.consumption.gate_released_unix_ms.is_none());
    assert!(cancelled.consumption.cpu_seconds.is_none());
    let expired = pre_gate_cancellation(&started, 120).unwrap();
    assert_eq!(expired.command_status, EvaluationStatus::BudgetExhausted);
    assert_eq!(expired.job_sha256, cancelled.job_sha256);
}
