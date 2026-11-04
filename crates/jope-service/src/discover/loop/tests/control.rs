use super::*;

#[test]
fn custom_objective_controls_every_promotion_despite_opposing_surrogate() {
    struct CustomScores {
        promoted: Mutex<Vec<String>>,
    }
    impl Executor for CustomScores {
        fn evaluate(&self, request: &Request) -> Response {
            let prediction = request
                .dsl
                .strip_prefix("seed-")
                .and_then(|s| s.parse::<u32>().ok())
                .map_or(100.0, f64::from);
            let objective = RegressionObjective::parse(request.objective.as_str().unwrap())
                .unwrap()
                .evaluate(&[0.0], &[prediction])
                .unwrap();
            let surrogate = RegressionObjective::parse("rmse")
                .unwrap()
                .evaluate(&[0.0], &[prediction])
                .unwrap();
            assert_eq!(objective.loss, -prediction);
            assert_eq!(surrogate.loss, prediction);
            if request.attempt.slot >= 65_536 {
                self.promoted.lock().unwrap().push(request.dsl.clone());
            }
            Response {
                attempt: request.attempt.clone(),
                scores: vec![surrogate.loss, objective.loss, objective.value],
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
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 30, Effort::Fast, 24);
    // A newly defined objective, rather than one of the ranker's trained
    // catalog labels. Actual runtime parsing/scoring supplies the loss.
    p.objective = json!({
        "metric": "custom", "name": "rel07_untrained_pred_mean",
        "direction": "maximize", "expression": {"op": "pred"},
        "aggregation": {"kind": "mean"}
    })
    .to_string();
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    let executor = CustomScores {
        promoted: Mutex::new(vec![]),
    };
    let admission = ContextAdmission {
        rounds: Mutex::new(vec![]),
        refuse: false,
    };
    d.advance(&RoundSpace, &admission, &executor, 1, 1).unwrap();
    assert_eq!(d.champion().unwrap().candidate.dsl, "seed-3");
    assert_eq!(d.champion().unwrap().loss, -3.0);
    d.advance(&RoundSpace, &admission, &executor, 1, 1).unwrap();
    assert_eq!(d.champion().unwrap().loss, -100.0);
    assert!(d.champion().unwrap().candidate.dsl.ends_with("-4"));
    let promoted = executor.promoted.lock().unwrap();
    assert_eq!(promoted.len(), 2, "every rung-1 decision is observed");
    assert_eq!(promoted[0], "seed-3", "seed-0 had the best surrogate");
    assert!(promoted[1].ends_with("-4"));
    assert_eq!(*admission.rounds.lock().unwrap(), vec![1, 2]);
}

#[test]
fn round_context_admission_preserves_objective_promotions_and_pending_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Fast, 24);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let mut d = Discovery::open(store.clone(), p.clone()).unwrap();
    let admission = ContextAdmission {
        rounds: Mutex::new(vec![]),
        refuse: false,
    };
    d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
        .unwrap();
    assert!(
        admission.rounds.lock().unwrap().is_empty(),
        "seed round bypasses admission"
    );
    assert_eq!(d.champion().unwrap().candidate.dsl, "seed-3");
    assert_eq!(d.champion().unwrap().loss, 17.0);
    d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
        .unwrap();
    assert_eq!(*admission.rounds.lock().unwrap(), vec![1, 2]);
    assert_eq!(d.checkpoint.pending.len(), 1);
    assert!(d.champion().unwrap().candidate.dsl.ends_with("-4"));
    assert_eq!(
        d.champion().unwrap().loss,
        5.0,
        "objective improves despite worse surrogate"
    );
    let pending = d.checkpoint.pending[0].candidates.clone();
    assert!(pending
        .iter()
        .all(|c| c.dsl.rsplit('-').next().unwrap().parse::<u32>().unwrap() >= 4));
    d.pause().unwrap();
    drop(d);
    let mut d = Discovery::open(store, p).unwrap();
    let refusal = ContextAdmission {
        rounds: Mutex::new(vec![]),
        refuse: true,
    };
    d.advance(&RoundSpace, &refusal, &OpposingScores, 7, 1)
        .unwrap();
    assert!(
        refusal.rounds.lock().unwrap().is_empty(),
        "retained wave is not reranked on recovery/capacity change"
    );
}
