use feat_math::kernel::strategy::{
    execute, execute_with_recorder, Allocator, Candidate, Cost, Evaluation, Evaluator, Event,
    Identity, Proposal, Proposer, Recorder, Reservation, Strategy, StrategyConfig,
};

struct Queue(Vec<Proposal>);
impl Proposer for Queue {
    fn propose(&mut self, _: usize) -> Option<Proposal> {
        (!self.0.is_empty()).then(|| self.0.remove(0))
    }
}

struct UnitAllocator;
impl Allocator for UnitAllocator {
    fn reserve(&mut self, candidate: &Candidate, remaining: u64) -> Option<Reservation> {
        (remaining > 0).then(|| Reservation {
            id: format!("r:{}", candidate.id),
            cost: Cost { units: 1 },
        })
    }
}

struct Score;
impl Evaluator for Score {
    fn evaluate(&mut self, candidate: &Candidate, _: &Reservation) -> Evaluation {
        Evaluation {
            outcome: if candidate.bytes == b"bad" {
                "fit_failed"
            } else {
                "measured"
            }
            .into(),
            score: (candidate.bytes != b"bad").then_some(candidate.bytes.len() as f64),
        }
    }
}

fn candidate(id: &str, bytes: &[u8]) -> Proposal {
    Proposal {
        candidate: Candidate {
            id: id.into(),
            bytes: bytes.into(),
        },
        origin: "fixture".into(),
    }
}

fn strategy() -> Strategy {
    let mut identity = Identity::new(
        "vocabulary-a",
        3,
        "profile-hash",
        "fixture-evaluator",
        "1",
        "seed-v1",
        "unit-v1",
    );
    identity.source_id = "85d7304a11462bb782607cbc780108408a930777".into();
    identity.context_id = "fixture-context".into();
    identity.input_sha256 = "input-hash".into();
    Strategy {
        identity,
        config: StrategyConfig {
            max_attempts: 3,
            cost_budget: 2,
            max_candidate_bytes: 8,
        },
    }
}

fn run() -> feat_math::kernel::strategy::Run {
    let mut proposer = Queue(vec![
        candidate("a", b"one"),
        candidate("dup", b"one"),
        candidate("b", b"two"),
    ]);
    let mut allocator = UnitAllocator;
    let mut evaluator = Score;
    execute(&strategy(), &mut proposer, &mut allocator, &mut evaluator).unwrap()
}

#[test]
fn identical_inputs_replay_the_same_ordered_trace_and_result() {
    let first = run();
    let second = run();
    assert_eq!(first.trace, second.trace);
    assert_eq!(first.outcomes, second.outcomes);
    assert_eq!(first.best, second.best);
    assert_eq!(
        first
            .trace
            .events
            .iter()
            .map(|e| e.sequence)
            .collect::<Vec<_>>(),
        (0..first.trace.events.len() as u64).collect::<Vec<_>>()
    );
}

#[test]
fn duplicate_and_cost_bounds_are_recorded_without_recharging() {
    let mut strategy = strategy();
    strategy.config.max_attempts = 4;
    let mut proposer = Queue(vec![
        candidate("a", b"one"),
        candidate("dup", b"one"),
        candidate("b", b"two"),
        candidate("c", b"three"),
    ]);
    let result = execute(&strategy, &mut proposer, &mut UnitAllocator, &mut Score).unwrap();
    assert_eq!(result.consumed_cost, 2);
    assert_eq!(result.outcomes.len(), 2);
    assert!(result
        .trace
        .events
        .iter()
        .any(|event| event.kind == feat_math::kernel::strategy::EventKind::Duplicate));
    assert!(result
        .trace
        .events
        .iter()
        .any(|event| event.kind == feat_math::kernel::strategy::EventKind::ReservationRefused));
    assert_eq!(result.best, Some(0));
}

#[test]
fn failures_and_panics_are_retained_without_a_rankable_score() {
    let mut proposer = Queue(vec![candidate("bad", b"bad")]);
    let run = execute(&strategy(), &mut proposer, &mut UnitAllocator, &mut Score).unwrap();
    assert_eq!(run.outcomes[0].result.outcome, "fit_failed");
    assert_eq!(run.outcomes[0].result.score, None);
    struct Panic;
    impl Evaluator for Panic {
        fn evaluate(&mut self, _: &Candidate, _: &Reservation) -> Evaluation {
            panic!("fixture evaluation panic");
        }
    }
    let mut proposer = Queue(vec![candidate("panic", b"safe")]);
    let run = execute(&strategy(), &mut proposer, &mut UnitAllocator, &mut Panic).unwrap();
    assert_eq!(run.outcomes[0].result.outcome, "panicked");
    assert_eq!(run.best, None);
}

#[test]
fn injected_recorder_receives_the_same_complete_ordered_trace() {
    #[derive(Default)]
    struct Events(Vec<Event>);
    impl Recorder for Events {
        fn record(&mut self, event: Event) {
            self.0.push(event);
        }
    }
    let mut proposer = Queue(vec![candidate("a", b"one")]);
    let mut recorder = Events::default();
    let run = execute_with_recorder(
        &strategy(),
        &mut proposer,
        &mut UnitAllocator,
        &mut Score,
        &mut recorder,
    )
    .unwrap();
    assert_eq!(recorder.0, run.trace.events);
}

#[test]
fn strategy_identity_is_orthogonal_to_vocabulary_identity() {
    let first = strategy();
    let mut other_profile = first.clone();
    other_profile.identity.profile_id = "vocabulary-b".into();
    other_profile.identity.profile_sha256 = "other-profile-hash".into();
    assert_ne!(first.identity.sha256(), other_profile.identity.sha256());
    assert_eq!(
        first.identity.strategy_id,
        other_profile.identity.strategy_id
    );
    assert_eq!(
        first.identity.strategy_version,
        other_profile.identity.strategy_version
    );
    let mut other_seed = first.clone();
    other_seed.identity.seed = 74;
    assert_ne!(first.identity.sha256(), other_seed.identity.sha256());
    let mut other_strategy = first.clone();
    other_strategy.identity.strategy_id = "other-strategy".into();
    let mut proposer = Queue(vec![candidate("a", b"one")]);
    let mut allocator = UnitAllocator;
    let mut evaluator = Score;
    assert!(execute(
        &other_strategy,
        &mut proposer,
        &mut allocator,
        &mut evaluator
    )
    .is_err());
}

#[test]
fn oversized_candidates_and_nonfinite_scores_are_rejected() {
    let mut s = strategy();
    let mut proposer = Queue(vec![candidate("large", b"123456789")]);
    assert!(execute(&s, &mut proposer, &mut UnitAllocator, &mut Score).is_err());
    s.config.max_candidate_bytes = 16;
    let mut bad = Queue(vec![candidate("nan", b"nan")]);
    struct NonFinite;
    impl Evaluator for NonFinite {
        fn evaluate(&mut self, _: &Candidate, _: &Reservation) -> Evaluation {
            Evaluation {
                outcome: "measured".into(),
                score: Some(f64::NAN),
            }
        }
    }
    assert!(execute(&s, &mut bad, &mut UnitAllocator, &mut NonFinite).is_err());
}
