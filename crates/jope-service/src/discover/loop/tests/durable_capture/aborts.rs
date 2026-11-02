use super::*;

#[test]
fn auxiliary_abort_or_stale_reply_stays_unknown() {
    for mode in 0..3 {
        struct AbortAux {
            inner: Evaluator,
            mode: usize,
        }
        impl Executor for AbortAux {
            fn evaluate(&self, r: &Request) -> Response {
                if r.attempt.slot == 65536 && self.mode < 2 {
                    assert_ne!(self.mode, 0, "owned auxiliary panic control");
                    let mut response = self.inner.evaluate(r);
                    response.attempt.attempt += 1;
                    return response;
                }
                self.inner.evaluate(r)
            }
            fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
                panic!("owned refit panic control")
            }
        }
        let home = tempfile::tempdir().unwrap();
        let p = control_plan(home.path(), 40);
        let run = p.run_id.clone();
        let (c, _) = context(&p, 20, true, false);
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
        let e = AbortAux {
            inner: Evaluator::new(store.clone()),
            mode,
        };
        assert!(d
            .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
            .is_err());
        assert!(d.poisoned);
        assert!(stored(&d)["capture"]["rounds"]["0"]["committed"].is_null());
        assert_eq!(stored(&d)["capture"]["charged"], 4);
        assert!(d.projection().unwrap().as_array().unwrap().is_empty());
        drop(d);
        let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
        assert!(d.checkpoint.capture.as_ref().unwrap().tainted);
        assert!(d
            .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
            .is_err());
        assert!(materialize_controls(&store, &run, &home.path().join("refused")).is_err());
    }
}

#[test]
fn exhausted_wall_stops_new_dispatch_without_refunding_reservations() {
    let (_scope, clock) = ClockScope::frozen();
    let home = tempfile::tempdir().unwrap();
    let mut p = control_plan(home.path(), 40);
    p.wall_seconds = 1;
    let (c, _) = context(&p, 20, true, false);
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
    struct Deadline(Evaluator, Arc<std::sync::atomic::AtomicU64>);
    impl Executor for Deadline {
        fn evaluate(&self, r: &Request) -> Response {
            self.1.fetch_add(1000, Ordering::SeqCst);
            self.0.evaluate(r)
        }
        fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
            unreachable!()
        }
    }
    let e = Deadline(Evaluator::new(store.clone()), clock);
    // Work performed before advance leaves the frozen admission budget intact.
    let before = d.checkpoint.elapsed_at(wall_clock_ms().unwrap()).unwrap();
    assert_eq!(before, 0);
    assert!(d
        .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
        .is_err());
    assert!(d.poisoned);
    assert_eq!(e.0.calls.lock().unwrap().len(), 1);
    assert_eq!(stored(&d)["capture"]["charged"], 4);
    drop(d);
    let d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
    assert_eq!(d.completed(), Some(CompletionReason::BudgetTime));
    assert!(d.checkpoint.capture.as_ref().unwrap().tainted);
    assert!(materialize_controls(&store, &p.run_id, &home.path().join("refused")).is_err());
}

#[test]
fn terminated_pending_suffix_retains_known_prefix_but_refuses_complete_export() {
    for timed_out in [false, true] {
        for workers in [1, 8] {
            let (_scope, clock) = ClockScope::frozen();
            let home = tempfile::tempdir().unwrap();
            let mut p = control_plan(home.path(), 80);
            if timed_out {
                p.wall_seconds = 1;
            }
            let (context, _) = context(&p, 64, true, false);
            let store_home = home.path().join("store");
            let store = Arc::new(Store::open(&store_home).unwrap());
            let mut d =
                Discovery::open_research(store.clone(), p.clone(), context.clone()).unwrap();
            struct FinishAtDeadline {
                inner: Evaluator,
                clock: Arc<std::sync::atomic::AtomicU64>,
                timed_out: bool,
            }
            impl Executor for FinishAtDeadline {
                fn evaluate(&self, request: &Request) -> Response {
                    self.inner.evaluate(request)
                }
                fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
                    let bytes = self.inner.fitted_model(request)?;
                    if self.timed_out && request.attempt.round == 1 {
                        self.clock.fetch_add(1000, Ordering::SeqCst);
                    }
                    Ok(bytes)
                }
            }
            let executor = FinishAtDeadline {
                inner: Evaluator::new(store.clone()),
                clock,
                timed_out,
            };
            for _ in 0..4 {
                if d.completed().is_some() {
                    break;
                }
                d.advance(&SPACE, &HeuristicAdmission, &executor, workers, usize::MAX)
                    .unwrap();
            }
            let reason = if timed_out {
                CompletionReason::BudgetTime
            } else {
                CompletionReason::Converged
            };
            assert_eq!(d.completed(), Some(reason));
            assert!(d.checkpoint.pending.is_empty());
            let captured = d.checkpoint.capture.as_ref().unwrap().clone();
            let prefix = records(&d);
            assert_eq!(prefix.len(), if timed_out { 2 } else { 4 });
            assert!(prefix.iter().all(|round| round.capture_complete));
            let (number, suffix) = captured.rounds.last_key_value().unwrap();
            assert_eq!(*number, prefix.len() as u64);
            assert!(suffix.aborted && suffix.committed.is_none());
            assert_eq!(suffix.attempts.len(), if timed_out { 8 } else { 16 });
            assert!(suffix
                .attempts
                .iter()
                .all(|attempt| attempt.result.is_some()));
            assert!(captured.tainted && captured.unknown());
            assert_eq!(captured.charged, if timed_out { 20 } else { 52 });
            let rung0_calls = executor
                .inner
                .calls
                .lock()
                .unwrap()
                .iter()
                .filter(|r| r.attempt.slot < 65536)
                .count();
            assert_eq!(rung0_calls as u64, captured.charged);
            assert_eq!(d.checkpoint.discarded.len(), 4);
            assert!(d
                .checkpoint
                .discarded
                .iter()
                .all(|r| r.reason == reason && r.response.is_some()));
            let before = store
                .get(&RecordKey::Run(p.run_id.clone()))
                .unwrap()
                .unwrap();
            assert!(captured
                .roots()
                .iter()
                .all(|root| before.blobs.contains(root)));
            let output = home.path().join("refused.jsonl");
            let error = materialize_controls(&store, &p.run_id, &output)
                .unwrap_err()
                .to_string();
            assert!(error.contains("unknown/aborted"), "{error}");
            assert!(
                error.contains("committed prefix is inspection-only"),
                "{error}"
            );
            assert!(!output.exists());
            assert!(!fs::read_dir(home.path()).unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".jope-research-")));
            assert!(store
                .gc(true)
                .unwrap()
                .blobs
                .iter()
                .all(|root| !captured.roots().contains(root)));
            drop(d);
            drop(executor);
            drop(store);
            let store = Arc::new(Store::open(&store_home).unwrap());
            let mut resumed = Discovery::open_research(store.clone(), p.clone(), context).unwrap();
            assert_eq!(resumed.completed(), Some(reason));
            assert_eq!(
                serde_json::to_value(records(&resumed)).unwrap(),
                serde_json::to_value(prefix).unwrap()
            );
            assert_eq!(
                resumed.checkpoint.capture.as_ref().unwrap().charged,
                captured.charged
            );
            let no_work = Evaluator::new(store.clone());
            assert!(resumed
                .advance(&SPACE, &HeuristicAdmission, &no_work, workers, usize::MAX)
                .unwrap()
                .is_empty());
            assert!(no_work.calls.lock().unwrap().is_empty());
            assert!(materialize_controls(&store, &p.run_id, &output).is_err());
            let after = store
                .get(&RecordKey::Run(p.run_id.clone()))
                .unwrap()
                .unwrap();
            assert_eq!(before.blobs, after.blobs);
        }
    }
}
