use super::*;
use crate::{
    clock::Clock,
    search::{Budget, BOOTSTRAP_GENOME},
};
use feat_math::{kernel::search::admit_child, GenomeIr};
use std::{
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

fn data() -> Cut {
    super::super::tests::data()
}
fn seed() -> String {
    GenomeIr::parse("(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge))")
        .unwrap()
        .canonical()
}
fn options(evaluations: usize) -> Options {
    let mut search = SearchOptions::new(
        Budget {
            evaluations,
            seconds: 120,
        },
        7,
    );
    search.clock = Clock::Frozen;
    Options::new(
        search,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        Refinement {
            top_k: ELITES,
            trials_per_program: 3.min(evaluations),
        },
    )
    .unwrap()
}
fn parent_state(cut: &Cut, options: &Options) -> State {
    let mut state = State::new(options.search.budget.evaluations);
    let record = options
        .evaluator
        .measure(
            &seed(),
            cut,
            0,
            "incumbent",
            options.search.clock,
            &mut state.budget,
        )
        .unwrap();
    assert!(record.observation.measured(), "{record:?}");
    state.seen.insert(record.observation.genome.clone());
    state.observations.push(record);
    state
}

#[test]
fn resolves_only_shipped_profiles_and_refuses_unsupported_limits() {
    for p in shipped_kernel2_profiles() {
        let o = Options::new(
            options(8).search,
            p.id,
            FaultPolicy::RejectHard,
            Refinement {
                top_k: 1,
                trials_per_program: 3,
            },
        )
        .unwrap();
        assert_eq!(o.evaluator.profile().sha256(), p.sha256());
    }
    for id in ["legacy", "legacy-v2", "kernel2-unshipped"] {
        assert!(Options::new(
            options(8).search,
            id,
            FaultPolicy::RejectHard,
            Refinement {
                top_k: 1,
                trials_per_program: 3
            }
        )
        .is_err());
    }
    let mut o = options(8);
    o.search.threads = 2;
    assert!(run_cut(&data(), &o)
        .unwrap_err()
        .to_string()
        .contains("one thread"));
    o.search.threads = 1;
    o.search.islands = 2;
    assert!(run_cut(&data(), &o).is_err());
    assert!(Options::new(
        options(8).search,
        "kernel2-baseline",
        FaultPolicy::Legacy,
        Refinement {
            top_k: 1,
            trials_per_program: 3
        }
    )
    .is_err());
    assert!(Options::new(
        options(8).search,
        "kernel2-baseline",
        FaultPolicy::Penalize {
            lambda_domain: f64::NAN,
            lambda_saturation: 0.0,
            tie_break: true
        },
        Refinement {
            top_k: 1,
            trials_per_program: 3
        }
    )
    .is_err());
}

#[test]
fn cut_search_consumes_profile_wrappers_and_records_every_attempt() {
    let mut o = options(16);
    o.refinement.top_k = 0;
    let result = run_cut(&data(), &o).unwrap();
    assert_eq!(result.completed_evaluations, 16);
    assert_eq!(result.candidate_measurements, 16);
    assert_eq!(result.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert!(result.successful_measurements > 0);
    assert!(!result.arm_counts.is_empty());
    assert_eq!(result.observations[0].observation.genome, BOOTSTRAP_GENOME);
    for record in &result.observations {
        assert_eq!(record.observation.cut, data().index);
        let ir = admit_child(&record.observation.genome, o.evaluator.profile(), 1).unwrap();
        assert_eq!(ir.canonical(), record.observation.genome);
        assert_eq!(
            record.observation.program_sha256,
            crate::sha256_hex(record.observation.genome.as_bytes())
        );
        if !record.observation.measured() {
            assert!(record.selection_score.is_none());
        }
    }
    assert_eq!(result.best, *o.evaluator.rank(&result.observations)[0]);
}

#[test]
fn refinement_objectives_and_child_measurements_share_the_total_allowance() {
    let mut o = options(16);
    o.search.seeds = vec![seed()];
    let cut = data();
    let result = run_cut(&cut, &o).unwrap();
    assert!(result.objective_trials > 0);
    assert_eq!(
        result.completed_evaluations,
        result.candidate_measurements + result.objective_trials
    );
    assert!(result.completed_evaluations <= o.search.budget.evaluations);
    assert!(result.refinements.iter().any(|r| r.measurement.is_some()));
    for child in &result.refinements {
        if let Some(measured) = &child.measurement {
            assert_eq!(measured.observation.origin, super::super::REFINE);
            assert!(result.observations.contains(measured));
        }
    }
}

#[test]
fn evaluation_bound_replays_observation_and_profile_bytes() {
    let mut o = options(4);
    o.refinement.top_k = 0;
    o.search.seeds = vec![seed()];
    let first = run_cut(&data(), &o).unwrap();
    let second = run_cut(&data(), &o).unwrap();
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
    assert_eq!(first.profile_sha256, o.evaluator.profile().sha256());
    assert_eq!(first.observations[1].observation.origin, "incumbent");
    assert_eq!(first.observations[1].observation.genome, seed());
}

#[test]
fn expired_deadline_keeps_required_seed_measurements_and_starts_no_trials() {
    let mut o = options(32);
    o.search.clock = Clock::Expired;
    o.search.seeds = vec![seed()];
    let result = run_cut(&data(), &o).unwrap();
    assert_eq!(result.candidate_measurements, 2);
    assert_eq!(result.objective_trials, 0);
    assert_eq!(result.completed_evaluations, 2);
    assert_eq!(result.stopped_by, crate::search::STOPPED_BY_SECONDS);
    assert!(result
        .observations
        .iter()
        .all(|o| matches!(o.observation.origin, "bootstrap" | "incumbent")));
}

#[test]
fn cancellation_refuses_search_and_guarded_generation_work() {
    let mut o = options(12);
    let cut = data();
    let mut state = parent_state(&cut, &o);
    let used = state.budget.used();
    o.search.cancel = Some(Arc::new(AtomicBool::new(true)));
    assert!(crate::parallel::is_cancelled(
        &run_cut(&cut, &o).unwrap_err()
    ));
    generation::refine_generation(
        &mut state,
        &cut,
        &o,
        Clock::Frozen.start(),
        Duration::from_secs(120),
    )
    .unwrap();
    assert_eq!(state.budget.used(), used);
    assert!(state
        .refinements
        .iter()
        .all(|r| r.refinement.evaluations == 0 && r.measurement.is_none()));
}

#[test]
fn seed_scope_duplicates_and_insufficient_allowance_fail_before_measurement() {
    let mut o = options(4);
    o.search.seeds = vec![GenomeIr::parse("(pipe X (map expr=\"exp(x)\") (ridge))")
        .unwrap()
        .canonical()];
    assert!(run_cut(&data(), &o).is_err());
    o.search.seeds = vec![seed(), seed()];
    assert!(run_cut(&data(), &o)
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    o.search.seeds = vec![seed()];
    o.search.budget.evaluations = 1;
    o.refinement.trials_per_program = 1;
    assert!(run_cut(&data(), &o)
        .unwrap_err()
        .to_string()
        .contains("bootstrap and seeds"));
    o.search.seeds.clear();
    let mut cut = data();
    cut.train_x[0].clear();
    assert!(run_cut(&cut, &o).is_err());
}

#[test]
fn hard_refusals_spend_slots_and_never_become_parents() {
    let o = options(5);
    let mut cut = data();
    cut.hold_x[0].clear();
    let mut state = State::new(5);
    generation::measure_generation(
        &mut state,
        &cut,
        &o,
        Clock::Frozen.start(),
        Duration::from_secs(120),
    )
    .unwrap();
    assert_eq!(state.budget.used(), 5);
    assert_eq!(state.observations.len(), 5);
    assert!(state
        .observations
        .iter()
        .all(|r| !r.observation.measured() && r.selection_score.is_none()));
    assert!(generation::elite_genomes(&state, &o).is_empty());
    assert!(state.result(&cut, &o, "test", 0.0).is_err());
}

#[test]
fn elite_pool_uses_fault_aware_score_order_and_unique_genomes() {
    let o = options(12);
    let mut state = parent_state(&data(), &o);
    let mut a = state.observations[0].clone();
    a.selection_score = Some(0.8);
    a.observation.rmse = Some(0.01);
    let mut b = a.clone();
    b.observation.genome = BOOTSTRAP_GENOME.into();
    b.selection_score = Some(0.9);
    b.observation.rmse = Some(9.0);
    state.observations = vec![a.clone(), b.clone(), a, b];
    assert_eq!(
        generation::elite_genomes(&state, &o),
        vec![BOOTSTRAP_GENOME.to_owned(), seed()]
    );
}

#[test]
fn coordinator_draws_all_expression_arms_from_the_shipped_pool() {
    let o = options(12);
    let cut = data();
    let mut state = State::new(12);
    state.generation = 1;
    let parents = ["(pipe X (map expr=\"sin(x)+(0.5*c0)\") (ridge))",
        "(pipe X (map expr=\"cos(x)+row_l2\") (derive expr=\"tanh(c0)+row_std\" append=true) (ridge))"]
        .into_iter().map(|s| GenomeIr::parse(s).unwrap().canonical()).collect::<Vec<_>>();
    for slot in 0..256 {
        let (text, _) = offspring::draw(&parents, &mut state, &cut, &o, slot).unwrap();
        assert_eq!(
            admit_child(&text, o.evaluator.profile(), 1)
                .unwrap()
                .canonical(),
            text
        );
        state.seen.insert(text);
    }
    for arm in ["expr_grow", "expr_edit", "expr_drop", "expr_cross"] {
        assert!(
            state.arms.get(arm).copied().unwrap_or(0) > 0,
            "{arm}: {:?}",
            state.arms
        );
    }
}

#[test]
fn generation_refinement_keeps_poisoned_holdout_outside_training_objectives() {
    let o = options(12);
    let ordinary = data();
    let mut poisoned = ordinary.clone();
    poisoned.hold_y.iter_mut().for_each(|y| *y += 1.0e9);
    let mut reports = Vec::new();
    for cut in [&ordinary, &poisoned] {
        let mut state = parent_state(cut, &o);
        generation::refine_generation(
            &mut state,
            cut,
            &o,
            Clock::Frozen.start(),
            Duration::from_secs(120),
        )
        .unwrap();
        assert!(state
            .refinements
            .iter()
            .any(|r| r.refinement.evaluations > 0));
        reports.push(
            state
                .refinements
                .iter()
                .map(|r| r.refinement.clone())
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(reports[0], reports[1]);
}
