use super::*;

#[test]
fn round_admission_refusal_preserves_committed_state_and_retry_identity() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Fast, 24);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    let admission = ContextAdmission {
        rounds: Mutex::new(vec![]),
        refuse: false,
    };
    d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
        .unwrap();
    let before = serde_json::to_value(&d.checkpoint).unwrap();
    let snapshot = d.snapshot().clone();
    let refusal = ContextAdmission {
        rounds: Mutex::new(vec![]),
        refuse: true,
    };
    for _ in 0..2 {
        assert!(d
            .advance(&RoundSpace, &refusal, &OpposingScores, 1, 1)
            .is_err());
        assert_eq!(serde_json::to_value(&d.checkpoint).unwrap(), before);
        assert_eq!(d.snapshot(), &snapshot);
    }
    assert_eq!(*refusal.rounds.lock().unwrap(), vec![1, 1]);
    d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
        .unwrap();
    assert_eq!(*admission.rounds.lock().unwrap(), vec![1, 2]);
}

struct TinySpace;
impl SearchSpace for TinySpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        Ok(dsl.into())
    }
    fn seeds(&self) -> Result<Vec<String>> {
        Ok(vec!["lucky-seed".into()])
    }
    fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
        Ok(vec!["later-better".into(), "tie".into()])
    }
}
struct Controlled {
    failure: Option<WorkerFailure>,
}
impl Executor for Controlled {
    fn evaluate(&self, request: &Request) -> Response {
        if let Some(failure) = self.failure {
            return Response::failed(request.attempt.clone(), failure, 0);
        }
        let promotion = request.attempt.slot >= 65_536;
        let loss = if request.dsl == "lucky-seed" {
            if promotion {
                10.0
            } else {
                0.0
            }
        } else {
            if promotion {
                5.0
            } else {
                1.0
            }
        };
        Response {
            attempt: request.attempt.clone(),
            scores: vec![loss, loss, loss],
            fold_scores: vec![],
            predictions_digest: digest(request.dsl.as_bytes()),
            wall_ms: 0,
            failure: None,
        }
    }
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        Ok(request.dsl.as_bytes().to_vec())
    }
}

pub(super) fn finish(
    discovery: &mut Discovery,
    space: &impl SearchSpace,
    executor: &impl Executor,
    workers: usize,
) {
    for _ in 0..300 {
        if discovery.completed().is_some() {
            return;
        }
        discovery
            .advance(space, &HeuristicAdmission, executor, workers, usize::MAX)
            .unwrap();
    }
    panic!("search did not stop within bounded rounds");
}

#[test]
fn strict_rung1_promotion_is_not_blocked_by_a_lucky_seed_rung0() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Fast, 24);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(Arc::clone(&store), p).unwrap();
    let executor = Controlled { failure: None };
    d.advance(&TinySpace, &HeuristicAdmission, &executor, 8, usize::MAX)
        .unwrap();
    assert_eq!(d.champion().unwrap().loss, 10.0);
    d.advance(&TinySpace, &HeuristicAdmission, &executor, 8, usize::MAX)
        .unwrap();
    assert_eq!(d.champion().unwrap().candidate.dsl, "later-better");
    assert_eq!(d.champion().unwrap().loss, 5.0);
    finish(&mut d, &TinySpace, &executor, 8);
    assert_eq!(d.completed(), Some(CompletionReason::Converged));
    let accepted = d
        .projection()
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == "algorithm.accepted")
        .count();
    assert_eq!(accepted, 2);
    assert_eq!(d.admitted(), 3);
    let model_id: ArtifactId = format!("art_{}", &hex_blob(&d.champion().unwrap().model)[..24])
        .parse()
        .unwrap();
    assert!(store.get(&RecordKey::Artifact(model_id)).unwrap().is_some());
}

#[test]
fn limit_hits_count_admissions_and_list_exact_experiments() {
    for failure in [
        WorkerFailure::Timeout,
        WorkerFailure::MemoryLimit,
        WorkerFailure::InvalidRequest,
        WorkerFailure::NonFinite,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 1);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        finish(
            &mut d,
            &TinySpace,
            &Controlled {
                failure: Some(failure),
            },
            1,
        );
        assert_eq!(d.admitted(), 1);
        assert!(d.champion().is_none());
        assert_eq!(d.completed(), Some(CompletionReason::BudgetEvaluations));
        let expected = usize::from(matches!(
            failure,
            WorkerFailure::Timeout | WorkerFailure::MemoryLimit
        ));
        assert_eq!(d.nondeterminism().count.0, expected as u64);
        assert_eq!(
            d.nondeterminism().experiment_ids,
            if expected == 1 {
                vec![experiment(&d.checkpoint.plan, 0, 0, 0).unwrap()]
            } else {
                vec![]
            }
        );
        if failure == WorkerFailure::NonFinite {
            let projection = d.projection().unwrap();
            assert!(projection
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["body"]["failure"]["code"] == "FITNESS_UNDEFINED"));
        }
    }
}

#[test]
fn heuristic_exploration_is_seeded_distinct_and_retains_first_two() {
    let proposals: Vec<_> = (0..16).map(|n| n.to_string()).collect();
    let a = HeuristicAdmission.admit(proposals.clone(), true, &mut round_rng(73, 4));
    let b = HeuristicAdmission.admit(proposals.clone(), true, &mut round_rng(73, 4));
    assert_eq!(a, b);
    assert_eq!(&a[..2], &["0", "1"]);
    assert_eq!(a.iter().collect::<BTreeSet<_>>().len(), 4);
    assert_eq!(
        HeuristicAdmission.admit(proposals, false, &mut round_rng(73, 4)),
        vec!["0", "1", "2", "3"]
    );
}

#[test]
fn invalid_and_changed_resume_plans_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Fast, 5);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
    drop(d);
    p.seed += 1;
    assert!(Discovery::open(Arc::clone(&store), p.clone()).is_err());
    p.seed -= 1;
    p.features.push(p.target.clone());
    assert!(Discovery::open(store, p).is_err());
}

#[test]
fn interrupted_active_span_consumes_remaining_wall_budget_before_replay() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Fast, 24);
    p.wall_seconds = 1;
    let home = directory.path().join("store");
    {
        let store = Arc::new(Store::open(&home).unwrap());
        let _interrupted = Discovery::open(store, p.clone()).unwrap();
        std::thread::sleep(Duration::from_millis(1500));
    }
    let store = Arc::new(Store::open(&home).unwrap());
    let mut recovered = Discovery::open(store, p).unwrap();
    assert!(recovered.checkpoint.elapsed_ms >= 1000);
    assert_eq!(recovered.completed(), Some(CompletionReason::BudgetTime));
    assert_eq!(recovered.budget_bound_by(), Some("time"));
    assert!(recovered
        .advance(
            &TinySpace,
            &HeuristicAdmission,
            &Controlled { failure: None },
            1,
            1
        )
        .unwrap()
        .is_empty());
    assert_eq!(recovered.admitted(), 0);
}

#[test]
fn explicit_pause_excludes_idle_time_and_backwards_active_clock_refuses() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Fast, 24);
    p.wall_seconds = 1;
    let home = directory.path().join("store");
    let paused_elapsed;
    {
        let store = Arc::new(Store::open(&home).unwrap());
        let mut d = Discovery::open(store, p.clone()).unwrap();
        let mut checkpoint = d.checkpoint.clone();
        checkpoint.elapsed_ms = 900;
        checkpoint.active_since_ms = None;
        assert_eq!(
            checkpoint.elapsed_at(u64::MAX).unwrap(),
            900,
            "known paused idle interval is not charged"
        );
        checkpoint.active_since_ms = Some(1000);
        assert!(checkpoint.elapsed_at(999).is_err());
        d.pause().unwrap();
        paused_elapsed = d.checkpoint.elapsed_ms;
    }
    std::thread::sleep(Duration::from_millis(50));
    let store = Arc::new(Store::open(&home).unwrap());
    let recovered = Discovery::open(store, p).unwrap();
    assert_eq!(recovered.checkpoint.elapsed_ms, paused_elapsed);
    assert_eq!(recovered.completed(), None);
    assert!(recovered.checkpoint.active_since_ms.is_some());
}

#[test]
fn aborted_round_exposes_no_checkpoint_events_or_artifact_reference() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Fast, 2);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
    let before = d.snapshot().clone();
    // Force the round's transaction hook to fail *after* EventLog has
    // staged checkpoint/event rows and the candidate CAS object exists.
    store
        .transaction(|tx| {
            tx.put(
                &RecordKey::Evaluation {
                    run: p.run_id.clone(),
                    number: 0,
                },
                &Record::metadata(json!({"fixture":"collision"}), vec![]),
            )
        })
        .unwrap();
    assert!(d
        .advance(
            &TinySpace,
            &HeuristicAdmission,
            &Controlled { failure: None },
            1,
            1
        )
        .is_err());
    assert_eq!(d.snapshot(), &before);
    assert_eq!(d.log.snapshot(&p.run_id).unwrap().unwrap(), before);
    assert!(d.projection().unwrap().as_array().unwrap().is_empty());
    assert!(d.champion().is_none());
    assert_eq!(d.admitted(), 0);
    let artifact: ArtifactId = format!("art_{}", &digest(b"lucky-seed")[..24])
        .parse()
        .unwrap();
    assert!(store.get(&RecordKey::Artifact(artifact)).unwrap().is_none());
    assert!(
        d.advance(
            &TinySpace,
            &HeuristicAdmission,
            &Controlled { failure: None },
            1,
            1
        )
        .is_err(),
        "unknown commit outcome requires reopen"
    );
}
