use super::*;
use std::{cell::RefCell, sync::Arc};

struct Shared {
    receiver: Receiver<Vec<u8>>,
    history: RefCell<Vec<serde_json::Value>>,
}

impl Shared {
    fn records(&self) -> Vec<serde_json::Value> {
        let mut history = self.history.borrow_mut();
        for bytes in self.receiver.try_iter() {
            assert!(bytes.len() <= MAX_LINE_BYTES);
            history.push(serde_json::from_slice(&bytes).unwrap());
        }
        history.clone()
    }
    fn kind<T: serde::de::DeserializeOwned>(&self, kind: &str) -> Vec<T> {
        self.records()
            .into_iter()
            .filter(|r| r["kind"] == kind)
            .map(|r| serde_json::from_value(r).unwrap())
            .collect()
    }
    fn lines(&self) -> Vec<Line> {
        self.kind(PROGRESS)
    }
}

fn observation(rmse: Option<f64>) -> Observation {
    Observation {
        genome: crate::select::BOOTSTRAP_GENOME.into(),
        program_sha256: String::new(),
        cut: 0,
        generation: 0,
        origin: "random",
        train_r2: rmse.map(|_| 0.5),
        hold_r2: rmse.map(|r| 1.0 - r),
        rmse,
        objective_loss: None,
        seconds: 0.0,
        status: if rmse.is_some() {
            crate::search::MEASURED
        } else {
            "fit_failed"
        },
        failure: rmse.is_none().then(|| "boom".into()),
        island: None,
        resources: None,
    }
}

fn reporter(t0: Instant) -> (Progress, Shared) {
    let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    let sink = Shared {
        receiver,
        history: RefCell::new(Vec::new()),
    };
    let budget = crate::search::Budget {
        evaluations: 10,
        seconds: 30,
    };
    (Progress::with_sender(Some(sender), 2, budget, t0), sink)
}

#[test]
fn lines_are_rate_limited_except_improvements_and_cut_ends() {
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let (progress, sink) = reporter(t0);
    progress.begin_cut(0);
    // The first boundary always reports.
    progress.observe(0, &observation(None));
    progress.generation_done_at(1, vec![], at(10));
    assert_eq!(sink.lines().len(), 1);
    // Within the second and no improvement: suppressed.
    progress.observe(0, &observation(None));
    progress.generation_done_at(2, vec![], at(400));
    progress.tick_at(at(900));
    assert_eq!(sink.lines().len(), 1);
    // An improvement reports immediately, inside the interval.
    progress.observe(0, &observation(Some(0.5)));
    progress.generation_done_at(3, vec![0.5], at(950));
    assert_eq!(sink.lines().len(), 2);
    // A worse measurement inside the interval is suppressed; after it, it reports.
    progress.observe(0, &observation(Some(0.9)));
    progress.generation_done_at(4, vec![0.5, 0.1], at(1500));
    assert_eq!(sink.lines().len(), 2);
    progress.generation_done_at(5, vec![0.1, f64::NAN, 0.5], at(1950));
    assert_eq!(sink.lines().len(), 3);
    // The end of a cut reports, unless the line just written already is its end state.
    progress.cut_done_at(5, at(1960));
    assert_eq!(sink.lines().len(), 3);
    progress.begin_cut(1);
    progress.observe(1, &observation(Some(0.7)));
    progress.generation_done_at(1, vec![0.3], at(2000));
    assert_eq!(sink.lines().len(), 3);
    progress.cut_done_at(1, at(2010));
    assert_eq!(sink.lines().len(), 4);
    progress.finish();

    let lines = sink.lines();
    assert_eq!(lines.len(), 5);
    assert!(lines.iter().all(|l| l.schema == SCHEMA && l.cuts == 2));
    assert!(lines
        .windows(2)
        .all(|w| w[0].evaluations <= w[1].evaluations
            && w[0].generation <= w[1].generation
            && w[0].sequence < w[1].sequence));
    // One counter numbers every kind.
    let sequences: Vec<u64> = sink
        .records()
        .iter()
        .map(|r| r["sequence"].as_u64().unwrap())
        .collect();
    assert_eq!(sequences, (0..sequences.len() as u64).collect::<Vec<_>>());
    assert!(lines
        .iter()
        .all(|l| l.kind == PROGRESS && l.generations.is_none()));
    assert_eq!(lines[0].population_r2, Some(vec![]));
    assert_eq!(lines[2].population_r2, Some(vec![0.5, 0.1]));
    assert_eq!(lines.last().unwrap().population_r2, Some(vec![0.3]));
    assert_eq!(lines[0].best_hold_rmse, None);
    assert_eq!(lines[1].best_hold_rmse, Some(0.5));
    assert_eq!(lines[1].best_hold_r2, Some(0.5));
    let last = lines.last().unwrap();
    assert_eq!(last.phase, DONE);
    assert_eq!((last.cut, last.generation), (1, 6));
    assert_eq!((last.evaluations, last.measured, last.failed), (5, 3, 2));
    assert_eq!(last.best_hold_rmse, Some(0.5));
    assert_eq!((last.eval_budget, last.seconds_budget), (20, 60.0));
    // Every evaluation, failed or not, is charged its emitted source in all four languages.
    let one = feat_math::source_stream::emitted_tokens(crate::select::BOOTSTRAP_GENOME);
    assert!(one.iter().all(|&n| n > 0), "{one:?}");
    assert_eq!(last.emitted_tokens, 5 * one.iter().sum::<u64>());
    let by = &last.emitted_tokens_by_language;
    assert_eq!(
        by.rust + by.python + by.cpp17 + by.javascript,
        last.emitted_tokens
    );
    assert!(lines
        .windows(2)
        .all(|w| w[0].emitted_tokens <= w[1].emitted_tokens));
}

#[test]
fn full_and_disconnected_queues_do_not_wait_or_lose_counter_state() {
    let budget = crate::search::Budget {
        evaluations: 10,
        seconds: 30,
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    let progress = Progress::with_sender(Some(sender), 2, budget, Instant::now());
    // Counted directly: this test is about progress records, not candidate records.
    Progress::count(&mut progress.lock(), &observation(Some(0.5)), [0; 4]);
    progress.tick(); // fills queue
    Progress::count(&mut progress.lock(), &observation(Some(0.4)), [0; 4]);
    progress.finish(); // full queue: drops done, without changing evaluation counters
    let first: Line = serde_json::from_slice(&receiver.try_recv().unwrap()).unwrap();
    assert_eq!((first.sequence, first.evaluations), (0, 1));
    progress.lock().cut = 1;
    Progress::count(&mut progress.lock(), &observation(None), [0; 4]);
    progress.finish();
    let latest: Line = serde_json::from_slice(&receiver.try_recv().unwrap()).unwrap();
    assert_eq!(
        (
            latest.sequence,
            latest.evaluations,
            latest.measured,
            latest.failed
        ),
        (2, 3, 2, 1)
    );
    assert_eq!(latest.phase, DONE);
    assert_eq!(latest.best_hold_rmse, Some(0.4));
    drop(receiver);
    progress.finish(); // disconnected queue: also returns
    assert_eq!(progress.lock().sequence, 4);
    assert_eq!(progress.lock().evaluations, 3);
}

#[test]
fn worker_drains_records_and_stops_on_write_error() {
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            // Deliberately accept partial writes: drain must finish the record.
            let count = bytes.len().min(3);
            self.0.lock().unwrap().extend_from_slice(&bytes[..count]);
            Ok(count)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let (sender, receiver) = mpsc::sync_channel(2);
    sender.try_send(b"first\n".to_vec()).unwrap();
    sender.try_send(b"second\n".to_vec()).unwrap();
    drop(sender);
    drain(Sink(bytes.clone()), receiver);
    assert_eq!(*bytes.lock().unwrap(), b"first\nsecond\n");

    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            panic!("failed write must not flush")
        }
    }
    let (sender, receiver) = mpsc::sync_channel(1);
    sender.try_send(vec![0]).unwrap();
    drain(Broken, receiver);
    assert!(matches!(
        sender.try_send(vec![1]),
        Err(mpsc::TrySendError::Disconnected(_))
    ));
}

#[test]
fn blocked_worker_cannot_hold_the_search_or_grow_the_queue() {
    struct Blocked {
        entered: mpsc::Sender<()>,
        release: Receiver<()>,
    }
    impl Write for Blocked {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let (entered, wait_entered) = mpsc::channel();
    let (release, wait_release) = mpsc::channel();
    let (sender, receiver) = mpsc::sync_channel(1);
    let worker = std::thread::spawn(move || {
        drain(
            Blocked {
                entered,
                release: wait_release,
            },
            receiver,
        )
    });
    sender.try_send(b"blocked\n".to_vec()).unwrap();
    wait_entered.recv_timeout(Duration::from_secs(2)).unwrap();
    let (complete, wait_complete) = mpsc::channel();
    let search = std::thread::spawn(move || {
        let progress = Progress::with_sender(
            Some(sender),
            2,
            crate::search::Budget {
                evaluations: 1000,
                seconds: 30,
            },
            Instant::now(),
        );
        for _ in 0..1000 {
            Progress::count(&mut progress.lock(), &observation(None), [0; 4]);
            progress.finish();
        }
        let summary = {
            let state = progress.lock();
            (state.evaluations, state.sequence)
        };
        complete.send(summary).unwrap();
    });
    // This synchronizes on actual completion while the writer remains blocked.
    let result = wait_complete.recv_timeout(Duration::from_secs(2));
    // Release first write and queued record even if the no-wait assertion fails.
    release.send(()).unwrap();
    release.send(()).unwrap();
    assert_eq!(result.unwrap(), (1000, 1000));
    search.join().unwrap();
    worker.join().unwrap();
}

#[test]
fn file_worker_writes_regular_output_with_actual_token_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.ndjson");
    let progress = Progress::file(
        std::fs::File::create(&path).unwrap(),
        1,
        crate::search::Budget {
            evaluations: 1,
            seconds: 30,
        },
    );
    progress.observe(0, &observation(Some(0.5)));
    progress.finish();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let text = std::fs::read_to_string(&path).unwrap();
        if let Some(line) = text
            .lines()
            .filter_map(|s| serde_json::from_str::<Line>(s).ok())
            .find(|line| line.phase == DONE)
        {
            let expected =
                feat_math::source_stream::emitted_tokens(crate::select::BOOTSTRAP_GENOME);
            assert_eq!(line.emitted_tokens, expected.iter().sum::<u64>());
            assert!(line.emitted_tokens > 0);
            assert_eq!(line.evaluations, 1);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "regular-file worker produced no done record"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn nonfinite_scores_never_cross_the_progress_json_boundary() {
    let (progress, sink) = reporter(Instant::now());
    progress.observe(0, &observation(Some(f64::NAN)));
    progress.tick();
    let mut measured = observation(Some(0.5));
    measured.hold_r2 = Some(f64::INFINITY);
    progress.observe(0, &measured);
    progress.finish();
    let lines = sink.lines();
    assert_eq!(lines[0].best_hold_rmse, None);
    assert_eq!(lines[1].best_hold_rmse, Some(0.5));
    assert_eq!(lines[1].best_hold_r2, None);
    assert_eq!(lines[1].evaluations, lines[1].measured + lines[1].failed);
}

#[test]
fn candidates_are_rate_limited_newest_wins_and_the_last_is_attempted_at_the_end() {
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let (progress, sink) = reporter(t0);
    progress.begin_cut(0);
    // 50 evaluations 10 ms apart (100 a second); the last one measured.
    for i in 0..50 {
        let rmse = (i == 49).then_some(0.25);
        progress.observe_at(Some(0), &observation(rmse), at(i * 10));
    }
    progress.finish();
    let candidates: Vec<CandidateLine> = sink.kind(CANDIDATE);
    // 0, 200 and 400 ms, then the newest waiting one (490 ms) at the end.
    assert_eq!(candidates.len(), 4);
    assert!(candidates[..3]
        .windows(2)
        .all(|w| w[1].elapsed_s - w[0].elapsed_s >= 0.2 - 1e-9));
    assert!(candidates[..3]
        .iter()
        .all(|c| c.status == "failed" && c.hold_r2.is_none() && c.hold_rmse.is_none()));
    let last = candidates.last().unwrap();
    assert_eq!(
        (last.status.as_str(), last.hold_rmse),
        ("measured", Some(0.25))
    );
    assert_eq!(last.hold_r2, Some(0.75));
    // Fixed labels and numbers only.
    for record in sink.records() {
        let object = record.as_object().unwrap();
        assert!(object.iter().all(|(key, value)| !value.is_string()
            || matches!(key.as_str(), "schema" | "kind" | "phase" | "status")));
    }
}

#[test]
fn worst_case_records_fit_the_cap() {
    let (progress, sink) = reporter(Instant::now());
    {
        let mut state = progress.lock();
        state.evaluations = u64::MAX;
        state.measured = u64::MAX;
        state.failed = u64::MAX;
        state.eval_budget = u64::MAX;
        state.generations_before = u64::MAX / 2;
        state.generations_in_cut = u64::MAX / 2;
        state.tokens = [u64::MAX / 4; 4];
        state.cut = usize::MAX;
        state.cuts = usize::MAX;
        state.seconds_budget = -f64::MAX;
        state.best = Some((-f64::MIN_POSITIVE, Some(-f64::MAX)));
        state.sequence = u64::MAX / 2;
        state.pending = Some(PendingCandidate {
            generation: u64::MAX,
            cut: usize::MAX,
            hold_r2: Some(-f64::MIN_POSITIVE),
            hold_rmse: Some(-f64::MAX),
            measured: false,
        });
        state.phase = COMPARISON;
    }
    // 1000 of the longest finite values: sorted, downsampled to 64, then halved to fit.
    let wide = (0..1000).map(|i| -f64::MAX / (1.0 + i as f64 * 1e-13));
    progress.generation_done(u64::MAX / 2, wide.collect());
    let records = sink.records();
    let line: Line = serde_json::from_value(
        records
            .iter()
            .find(|r| r["kind"] == PROGRESS)
            .unwrap()
            .clone(),
    )
    .unwrap();
    let candidate = records.iter().find(|r| r["kind"] == CANDIDATE).unwrap();
    assert!(serde_json::to_vec(candidate).unwrap().len() < 512);
    let population = line.population_r2.unwrap();
    assert!(!population.is_empty() && population.len() < POPULATION_SAMPLES);
    assert!(population.windows(2).all(|w| w[0] >= w[1]));
    // Typical R^2 values keep all 64.
    let (progress, sink) = reporter(Instant::now());
    progress.generation_done(1, (0..500).map(|i| 1.0 - i as f64 * 0.001234567).collect());
    assert_eq!(sink.lines()[0].population_r2.as_ref().unwrap().len(), 64);
}

#[test]
fn population_is_finite_sorted_descending_and_at_most_64() {
    let values =
        (0..1000)
            .map(|i| i as f64 / 1000.0)
            .chain([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let snapshot = population_snapshot(values, POPULATION_SAMPLES);
    assert_eq!(snapshot.len(), 64);
    assert!(snapshot.iter().all(|v| v.is_finite()));
    assert!(snapshot.windows(2).all(|w| w[0] >= w[1]));
    assert_eq!((snapshot[0], snapshot[63]), (0.999, 0.0));
    assert_eq!(
        population_snapshot([0.2, f64::NAN, 0.9], 64),
        vec![0.9, 0.2]
    );
}
