use super::*;
use std::{fs, io::Write, sync::Arc};

fn matrix(root: &std::path::Path) -> PathBuf {
    let mut csv = (0..21)
        .map(|i| format!("x{i}"))
        .chain(["target".into()])
        .collect::<Vec<_>>()
        .join(",")
        .into_bytes();
    csv.push(b'\n');
    csv.extend(include_bytes!(
        "../../../../../test-support/datasets/v1/d2-regression/train.csv"
    ));
    let mut table =
        jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
    table.rows.truncate(1); // TRAIN-derived protocol fixture; not a native quality measurement.
    let path = root.join("protocol.jmat");
    fs::write(&path, crate::dataset::jmat::encode(&table).unwrap()).unwrap();
    path
}
fn request(matrix: &std::path::Path, slot: u32) -> Request {
    Request {
        attempt: Attempt {
            run: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            round: 0,
            slot,
            attempt: 1,
        },
        matrix_path: matrix.to_path_buf(),
        fold_spec: serde_json::json!({}),
        dsl: "pid fixture".into(),
        objective: serde_json::json!("protocol fixture"),
        seed: 73,
        limits: super::super::Limits {
            wall_ms: 10000,
            rss_bytes: 256 * 1024 * 1024,
        },
    }
}
fn pool(executable: PathBuf, workers: usize) -> Arc<WorkerPool> {
    let mut config = PoolConfig::new(executable, workers);
    config.fixture = true;
    config.fixture_test = Some("worker::pool::resize_tests::resize_child");
    Arc::new(WorkerPool::new(config).unwrap())
}
fn until(mut predicate: impl FnMut() -> bool) {
    let start = Instant::now();
    while !predicate() {
        assert!(
            start.elapsed() < Duration::from_secs(8),
            "owned worker boundary timed out"
        );
        thread::sleep(Duration::from_millis(2));
    }
}
fn pids(pool: &WorkerPool) -> Vec<u32> {
    pool.state
        .lock()
        .unwrap()
        .processes
        .iter()
        .flatten()
        .map(|p| p.child.id())
        .collect()
}
#[cfg(target_os = "linux")]
fn reaped(pid: u32) {
    let mut status = 0;
    #[allow(unsafe_code)]
    // SAFETY: this PID came from our pool's owned Child; WNOHANG cannot block.
    let result = unsafe { libc::waitpid(pid as libc::pid_t, &mut status, libc::WNOHANG) };
    assert_eq!(result, -1, "worker{pid} was not already reaped");
    assert_eq!(
        io::Error::last_os_error().raw_os_error(),
        Some(libc::ECHILD)
    );
}
#[cfg(not(target_os = "linux"))]
fn reaped(_: u32) {}
fn warm(pool: &WorkerPool, matrix: &std::path::Path, root: &std::path::Path) -> Vec<Response> {
    let release = root.join("release");
    thread::scope(|scope| {
        let jobs: Vec<_> = (0..pool.live_workers())
            .map(|slot| {
                let mut request = request(matrix, slot as u32);
                request.fold_spec = serde_json::json!({"release":release});
                scope.spawn(move || pool.evaluate(&request))
            })
            .collect();
        until(|| pool.state.lock().unwrap().available.is_empty());
        fs::write(release, b"go").unwrap();
        jobs.into_iter()
            .map(|job| {
                let value = job.join().unwrap();
                assert!(value.failure.is_none(), "{value:?}");
                value
            })
            .collect()
    })
}

#[test]
#[ignore = "owned resize supervisor child, explicitly launched by pool tests"]
fn resize_child() {
    assert_eq!(std::env::var("JOPE_WORKER_FIXTURE").unwrap(), "worker");
    println!("WORKER_FIXTURE_READY");
    std::io::stdout().flush().unwrap();
    super::super::serve(1, |request, matrix| {
        if let Some(release) = request.fold_spec["release"].as_str() {
            until(|| std::path::Path::new(release).exists());
        }
        Ok(super::super::Scores {
            values: vec![
                std::process::id() as f64,
                matrix.bytes().as_ptr() as usize as f64,
            ],
            fold_scores: vec![],
            predictions_digest: "0".repeat(64),
        })
    })
    .unwrap();
    // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
    std::process::exit(0);
}

#[test]
fn physical_pid_ladder_reaps_one_and_preserves_all_other_launches() {
    let directory = tempfile::tempdir().unwrap();
    let matrix = matrix(directory.path());
    let pool = pool(std::env::current_exe().unwrap(), 8);
    let replies = warm(&pool, &matrix, directory.path());
    let initial = pids(&pool);
    assert_eq!(initial.len(), 8);
    let mappings: BTreeMap<_, _> = replies
        .iter()
        .map(|r| (r.scores[0] as u32, r.scores[1]))
        .collect();
    assert_eq!(mappings.len(), 8);
    let mut inventory = vec![serde_json::json!({"target":8,"pids":initial})];
    for target in (1..8).rev().chain(2..=8) {
        let before = pids(&pool);
        let surviving: Vec<_> = pool
            .state
            .lock()
            .unwrap()
            .processes
            .iter()
            .flatten()
            .map(|p| {
                (
                    p.child.id(),
                    p.pipe_thread.as_ref().unwrap().thread().id(),
                    p.matrix.clone(),
                    p.cap,
                )
            })
            .collect();
        // The resize caller exits immediately. It must not be the creator
        // whose Linux lifetime owns the newly restored process.
        thread::scope(|scope| {
            scope
                .spawn(|| pool.set_live_workers(target))
                .join()
                .unwrap()
        })
        .unwrap();
        assert_eq!(pool.live_workers(), target);
        let after = pids(&pool);
        assert_eq!(after.len(), target);
        assert_eq!(
            before.iter().filter(|pid| !after.contains(pid)).count(),
            usize::from(target < before.len())
        );
        assert_eq!(
            after.iter().filter(|pid| !before.contains(pid)).count(),
            usize::from(target > before.len())
        );
        for pid in before.iter().filter(|pid| !after.contains(pid)) {
            reaped(*pid);
        }
        let state = pool.state.lock().unwrap();
        for process in state.processes.iter().flatten() {
            assert_eq!(process.cap, 256 * 1024 * 1024);
            assert_eq!(process.matrix, matrix);
            if let Some((_, thread, path, cap)) =
                surviving.iter().find(|entry| entry.0 == process.child.id())
            {
                assert_eq!(process.pipe_thread.as_ref().unwrap().thread().id(), *thread);
                assert_eq!(&process.matrix, path);
                assert_eq!(process.cap, *cap);
            }
            #[cfg(target_os = "linux")]
            assert!(std::path::Path::new(&format!("/proc/{}", process.child.id())).exists());
        }
        drop(state);
        if target > before.len() {
            let restored = *after.iter().find(|pid| !before.contains(pid)).unwrap();
            let response = pool.evaluate(&request(&matrix, 200 + target as u32));
            assert!(response.failure.is_none(), "{response:?}");
            assert_eq!(
                response.scores[0] as u32, restored,
                "restored child must answer with its original PID after creator caller exits"
            );
            assert_eq!(
                pids(&pool),
                after,
                "no silent replacement of a restored process"
            );
        }
        assert!(pool.set_live_workers(target).is_ok());
        assert_eq!(pids(&pool), after);
        inventory.push(serde_json::json!({"target":target,"pids":after}));
    }
    // Exercise the surviving original slot: it kept the same mapped matrix.
    // All enabled slots are idle; isolate its next lease deterministically.
    let original = initial[0];
    {
        let mut state = pool.state.lock().unwrap();
        state.available.retain(|&slot| slot != 0);
        state.available.push(0);
    }
    let response = pool.evaluate(&request(&matrix, 100));
    assert!(response.failure.is_none());
    assert_eq!(response.scores[0] as u32, original);
    assert_eq!(response.scores[1], mappings[&original]);
    let final_pids = pids(&pool);
    drop(pool);
    for pid in final_pids {
        reaped(pid);
    }
    if let Some(root) = std::env::var_os("JOPE_WRK04_RECEIPTS") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("pid-ladder.json"),serde_json::to_vec_pretty(&serde_json::json!({"kind":"real PID protocol fixture, not NativeSpace S2","one_process_per_action":true,"inventory":inventory})).unwrap()).unwrap();
    }
}

#[test]
fn running_and_queued_calls_refuse_resize_without_consuming_fences() {
    let directory = tempfile::tempdir().unwrap();
    let matrix = matrix(directory.path());
    let pool = pool(std::env::current_exe().unwrap(), 2);
    pool.set_live_workers(1).unwrap();
    let release = directory.path().join("queue-release");
    let mut first = request(&matrix, 0);
    first.fold_spec = serde_json::json!({"release":release});
    let second = request(&matrix, 1);
    thread::scope(|scope| {
        let running = scope.spawn(|| pool.evaluate(&first));
        until(|| {
            let state = pool.state.lock().unwrap();
            state.in_flight == 1 && state.available.is_empty()
        });
        let queued = scope.spawn(|| pool.evaluate(&second));
        until(|| pool.state.lock().unwrap().in_flight == 2);
        assert_eq!(
            pool.set_live_workers(2).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert_eq!(pool.live_workers(), 1);
        fs::write(release, b"go").unwrap();
        assert!(running.join().unwrap().failure.is_none());
        assert!(queued.join().unwrap().failure.is_none());
    });
    assert_eq!(pool.state.lock().unwrap().in_flight, 0);
    pool.set_live_workers(2).unwrap();
    assert_eq!(
        pool.evaluate(&second).failure,
        Some(WorkerFailure::StaleAttempt)
    );
    pool.revoke(&first.attempt);
    assert_eq!(
        pool.evaluate(&first).failure,
        Some(WorkerFailure::StaleAttempt)
    );
    assert_eq!(pool.state.lock().unwrap().in_flight, 0);
}

#[test]
fn absolute_retry_and_invalid_jumps_never_retire_an_extra_process() {
    let directory = tempfile::tempdir().unwrap();
    let matrix = matrix(directory.path());
    let pool = pool(std::env::current_exe().unwrap(), 2);
    warm(&pool, &matrix, directory.path());
    pool.set_live_workers(1).unwrap();
    // A controller can lose the acknowledgement AFTER this effect. Retrying
    // its unchanged absolute target cannot apply another physical retirement.
    let pids_after_lost_ack = pids(&pool);
    pool.set_live_workers(1).unwrap();
    assert_eq!(pids(&pool), pids_after_lost_ack);
    for invalid in [0, 3, usize::MAX] {
        assert_eq!(
            pool.set_live_workers(invalid).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(pids(&pool), pids_after_lost_ack);
    }
    let cold = WorkerPool::new(PoolConfig::new(std::env::current_exe().unwrap(), 8)).unwrap();
    assert_eq!(
        cold.set_live_workers(6).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(cold.live_workers(), 8);
}

#[cfg(unix)]
#[test]
fn failed_restore_keeps_the_target_retryable_and_joins_its_failed_creator() {
    let directory = tempfile::tempdir().unwrap();
    let matrix = matrix(directory.path());
    let executable = directory.path().join("owned-worker");
    std::os::unix::fs::symlink(std::env::current_exe().unwrap(), &executable).unwrap();
    let pool = pool(executable.clone(), 2);
    warm(&pool, &matrix, directory.path());
    pool.set_live_workers(1).unwrap();
    let before = pids(&pool);
    fs::remove_file(&executable).unwrap();
    assert!(pool.set_live_workers(2).is_err());
    assert_eq!(pool.live_workers(), 1);
    assert_eq!(pids(&pool), before);
    std::os::unix::fs::symlink(std::env::current_exe().unwrap(), &executable).unwrap();
    pool.set_live_workers(2).unwrap();
    assert_eq!(pool.live_workers(), 2);
    let after = pids(&pool);
    assert!(before.iter().all(|pid| after.contains(pid)));
    drop(pool);
    for pid in after {
        reaped(pid);
    }
}
