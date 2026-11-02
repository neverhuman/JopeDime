use super::*;

#[test]
fn concurrent_commits_never_poison_a_healthy_live_index() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    let dir = fixture_dir();
    let path = dir.path().join("live-race.jsonl");
    let (tap, journal) =
        FactJournal::create(&path, "run-a", jope_narrator::Limits::default()).unwrap();
    let mut reader = index(&path);
    let done = Arc::new(AtomicBool::new(false));
    std::thread::scope(|scope| {
        let completed = done.clone();
        scope.spawn(move || {
            for n in 0..512 {
                assert_eq!(
                    tap.emit(LineageFact::ExplorationStopped {
                        cut: 0,
                        reason: "fixture_budget".into(),
                        generations: 1,
                        evaluations: n
                    }),
                    jope_narrator::EmitStatus::Queued
                );
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            assert_eq!(journal.finish().unwrap().status, "complete");
            completed.store(true, Ordering::Release);
        });
        while !done.load(Ordering::Acquire) {
            reader
                .status()
                .expect("ordinary append/commit must not poison the index");
            std::thread::yield_now();
        }
    });
    // A quiet reader must accept any already committed cursor after a burst,
    // including boundaries beyond the former 2,048-record refresh turn.
    let burst_path = dir.path().join("committed-burst.jsonl");
    let (tap, journal) =
        FactJournal::create(&burst_path, "run-a", jope_narrator::Limits::default()).unwrap();
    let mut status_reader = index(&burst_path);
    let mut cursor_reader = index(&burst_path);
    for n in 0..3100 {
        assert_eq!(
            tap.emit(LineageFact::ExplorationStopped {
                cut: 0,
                reason: "synthetic_burst".into(),
                generations: 1,
                evaluations: n,
            }),
            jope_narrator::EmitStatus::Queued
        );
    }
    assert_eq!(journal.finish().unwrap().status, "complete");
    let status = status_reader.status().unwrap();
    assert_eq!(status.last_sequence, "3100");
    assert_eq!(status.last_sequence, status.capture.records);
    let page = cursor_reader
        .page(
            &Cursor {
                run_id: "run-a".into(),
                sequence: "3090".into(),
            },
            20,
            MAX_PAGE_BYTES,
        )
        .expect("committed burst cursor must be valid on the first request");
    assert_eq!(page.events.len(), 10);
    assert_eq!(page.events[0].sequence, "3091");
    assert_eq!(page.next.sequence, "3100");
    assert!(!page.more);
    assert_eq!(reader.status().unwrap().last_sequence, "512");
    assert_eq!(
        reader
            .page(&zero(), 256, MAX_PAGE_BYTES)
            .unwrap()
            .events
            .len(),
        256
    );
}

#[test]
fn cli_distinguishes_integrity_and_completion_and_follow_stops_at_terminal_capture() {
    use std::process::{Command, Stdio};
    fn run(path: &Path, args: &[&str]) -> std::process::Output {
        let mut child = Command::new(env!("CARGO_BIN_EXE_gp-narration-stream"))
            .args(["--journal"])
            .arg(path)
            .args(["--run-id", "run-a"])
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                let _ = child.wait();
                panic!("finite reader command did not stop");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    let dir = fixture_dir();
    let complete = dir.path().join("complete.jsonl");
    let (tap, writer) =
        FactJournal::create(&complete, "run-a", jope_narrator::Limits::default()).unwrap();
    tap.emit(LineageFact::RunStopped {
        status: "failed".into(),
    });
    writer.finish().unwrap();
    assert!(
        run(&complete, &["verify", "--require-complete"])
            .status
            .success(),
        "source failure does not imply capture failure"
    );
    let output = run(&complete, &["follow"]);
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 1);
    let limited = dir.path().join("limited.jsonl");
    let (tap, writer) = FactJournal::create(
        &limited,
        "run-a",
        jope_narrator::Limits {
            max_records: 1,
            ..jope_narrator::Limits::default()
        },
    )
    .unwrap();
    tap.emit(LineageFact::RunStopped {
        status: "completed".into(),
    });
    tap.emit(LineageFact::RunStopped {
        status: "completed".into(),
    });
    assert!(writer.finish().is_err());
    assert!(
        run(&limited, &["verify"]).status.success(),
        "committed prefix integrity remains inspectable"
    );
    assert!(!run(&limited, &["verify", "--require-complete"])
        .status
        .success());
    assert!(!run(&limited, &["follow"]).status.success());
}
