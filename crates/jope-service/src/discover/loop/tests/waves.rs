use super::*;

#[test]
fn equal_rung1_scores_never_replace_the_champion() {
    struct Unique;
    impl SearchSpace for Unique {
        fn canonical(&self, s: &str) -> Result<String> {
            Ok(s.into())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            TinySpace.seeds()
        }
        fn propose(&self, _: &[String], _: usize, rng: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec![format!("tie-{}", rng.next_u64())])
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Medium, 4);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    finish(&mut d, &Unique, &Controlled { failure: None }, 8);
    let events = d.projection().unwrap();
    assert_eq!(
        events
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "algorithm.accepted")
            .count(),
        2
    );
    assert_eq!(
        events
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "experiment.finished"
                && e["body"]["experiment_id"]
                    .as_str()
                    .unwrap()
                    .starts_with("exp_"))
            .count(),
        8
    );
    assert_eq!(d.champion().unwrap().candidate.algorithm_id, "alg_00000001");
}

struct WaveSpace;
impl SearchSpace for WaveSpace {
    fn canonical(&self, s: &str) -> Result<String> {
        Ok(s.into())
    }
    fn seeds(&self) -> Result<Vec<String>> {
        Ok(vec!["private-seed-program".into()])
    }
    fn propose(&self, _: &[String], _: usize, rng: &mut ChaCha12Rng) -> Result<Vec<String>> {
        Ok(vec![format!("private-proposal-program-{}", rng.next_u64())])
    }
}
struct LateImprovement {
    wait: bool,
    late_promotion: std::sync::atomic::AtomicBool,
    revoked: std::sync::atomic::AtomicUsize,
}
impl LateImprovement {
    fn new(wait: bool) -> Self {
        Self {
            wait,
            late_promotion: std::sync::atomic::AtomicBool::new(false),
            revoked: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}
impl Executor for LateImprovement {
    fn evaluate(&self, request: &Request) -> Response {
        if request.attempt.round == 1 && request.attempt.slot >= 65536 && self.wait {
            std::thread::sleep(Duration::from_millis(1100));
        }
        if request.attempt.round >= 3 && request.attempt.slot >= 65536 {
            self.late_promotion.store(true, Ordering::SeqCst);
        }
        let loss = match request.attempt.round {
            0 => 1.0,
            1 | 2 => 2.0,
            _ => 0.0,
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
    fn revoke(&self, _: &Attempt) {
        self.revoked.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn convergence_stops_and_fences_a_wave_before_a_late_better_result_can_promote() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Fast, 24);
    p.planned_workers = 12;
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    let executor = LateImprovement::new(false);
    d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
        .unwrap();
    d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
        .unwrap();
    assert_eq!(d.completed(), Some(CompletionReason::Converged));
    assert_eq!(d.checkpoint.next_round, 3);
    assert_eq!(d.champion().unwrap().loss, 1.0);
    assert_eq!(d.checkpoint.discarded.len(), 1);
    assert_eq!(d.checkpoint.discarded[0].round, 3);
    assert!(d.checkpoint.pending.is_empty());
    assert!(!executor.late_promotion.load(Ordering::SeqCst));
    assert_eq!(executor.revoked.load(Ordering::SeqCst), 1);
    assert_eq!(
        d.admitted(),
        4,
        "already admitted discarded work remains charged"
    );
}

#[test]
fn wall_completion_is_checked_at_each_round_and_discards_the_remaining_wave() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Medium, 24);
    p.planned_workers = 12;
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    let executor = LateImprovement::new(true);
    d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
        .unwrap();
    // Model time already spent before this call without charging time
    // between the checkpoint setup and `advance`. The executor's deliberate
    // 1.1s delay below is what deterministically carries this round past the
    // 60s budget; scheduler load must not decide whether the round starts.
    d.checkpoint.elapsed_ms = 59_000;
    d.checkpoint.active_since_ms = None;
    d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
        .unwrap();
    assert_eq!(d.completed(), Some(CompletionReason::BudgetTime));
    assert_eq!(d.checkpoint.next_round, 2);
    assert_eq!(d.champion().unwrap().loss, 1.0);
    assert_eq!(d.checkpoint.discarded.len(), 2);
    assert!(!executor.late_promotion.load(Ordering::SeqCst));
    assert_eq!(executor.revoked.load(Ordering::SeqCst), 2);
}

#[test]
fn public_events_use_opaque_ids_and_projected_phrases_without_programs_or_search_counters() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Fast, 24);
    p.planned_workers = 12;
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    finish(&mut d, &WaveSpace, &LateImprovement::new(false), 8);
    let projection = d.projection().unwrap();
    let text = serde_json::to_string(&projection).unwrap();
    for word in [
        "private-seed-program",
        "private-proposal-program",
        "round",
        "slot",
        "rung",
        "admitted",
        "stalled",
        "(pipe",
    ] {
        assert!(!text.contains(word), "public leak: {word}");
    }
    for event in projection.as_array().unwrap() {
        if let Some(id) = event["body"]["experiment_id"].as_str() {
            assert!(id.starts_with("exp_"));
            assert_eq!(id.len(), 20);
        }
        if event["kind"] == "algorithm.accepted" {
            assert_eq!(event["body"]["excerpt"], "");
        }
        for key in ["summary", "hypothesis"] {
            if let Some(value) = event["body"][key].as_str() {
                phrase::lint_public_phrase(value).unwrap();
            }
        }
    }
}

#[test]
fn numerical_artifact_refit_failure_fails_only_the_candidate() {
    struct Unfittable;
    impl Executor for Unfittable {
        fn evaluate(&self, request: &Request) -> Response {
            Controlled { failure: None }.evaluate(request)
        }
        fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
            bail!("unit fixture numerical refit failure")
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Fast, 1);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    finish(&mut d, &TinySpace, &Unfittable, 1);
    assert!(d.champion().is_none());
    assert_eq!(d.completed(), Some(CompletionReason::BudgetEvaluations));
    assert_eq!(d.nondeterminism().count.0, 0);
    let projection = d.projection().unwrap();
    assert!(projection
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["body"]["failure"]["code"] == "FITNESS_UNDEFINED"));
}

#[test]
fn ultra_evaluates_both_cv_repeats_without_treating_repeated_validation_as_leakage() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Ultra, 9);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p.clone()).unwrap();
    let candidate = Candidate {
        algorithm_id: "alg_0000".into(),
        dsl: NativeSpace.canonical("(pipe X (ridge))").unwrap(),
    };
    let request = d.request(0, 0, 1, &candidate).unwrap();
    let spec: NativeFolds = serde_json::from_value(request.fold_spec.clone()).unwrap();
    assert_eq!(spec.folds.len(), 10);
    assert_eq!(spec.folds.iter().filter(|f| f.repeat == 0).count(), 5);
    assert_eq!(spec.folds.iter().filter(|f| f.repeat == 1).count(), 5);
    let matrix = Matrix::open(fs::read(&p.matrix_path).unwrap()).unwrap();
    let score = evaluate_native(&request, &matrix).unwrap();
    assert_eq!(score.values.len(), 3);
    assert!(score.values.iter().all(|s| s.is_finite()));
}
