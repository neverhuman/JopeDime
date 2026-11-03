use super::*;

#[test]
fn rewritten_children_score_bit_for_bit_as_their_parents_and_cost_one_each() {
    let cut = data();
    for policy in [FaultPolicy::RejectHard, penalize()] {
        let opt = options(4, 64, policy);
        let mut state = parents(&cut, &opt, &[ONE_TIMES, TIMES_ONE, FOLDED, MAPPED]);
        let before = state.budget.used();
        simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
        assert_eq!(state.moves.len(), 4, "{policy:?}");
        let measured: Vec<&MoveRecord> = state
            .moves
            .iter()
            .filter(|m| m.outcome == MEASURED)
            .collect();
        // `1*c0` and `c0*1.0` reduce to one program: the first ranked is measured, the other is a
        // duplicate and spends nothing.
        assert_eq!(measured.len(), 3, "{:?}", state.moves);
        let duplicate: Vec<&MoveRecord> = state
            .moves
            .iter()
            .filter(|m| m.outcome == DUPLICATE)
            .collect();
        assert_eq!(duplicate.len(), 1);
        assert!(duplicate[0].measurement.is_none());
        // One evaluation per measured child; the shared identity holds; no program twice.
        assert_eq!(state.budget.used() - before, measured.len());
        assert_eq!(state.budget.used(), state.observations.len());
        assert!(all_genomes_distinct(&state));
        let expected: BTreeSet<String> = [
            with_expression(ONE_TIMES, 0, "c0"),
            with_expression(FOLDED, 0, "c0+6.0"),
            with_expression(MAPPED, 0, "abs(x)"),
        ]
        .into_iter()
        .collect();
        let programs: BTreeSet<String> = measured
            .iter()
            .map(|m| m.program.clone().unwrap())
            .collect();
        assert_eq!(programs, expected);
        assert_eq!(
            duplicate[0].program,
            Some(with_expression(ONE_TIMES, 0, "c0"))
        );
        for m in &measured {
            let child = &state.observations[m.measurement.unwrap()];
            let parent = parent_of(&state, m);
            assert_eq!(child.observation.origin, crate::kernel2::REWRITE);
            assert_eq!(child.observation.generation, state.generation);
            assert_eq!(Some(&child.observation.genome), m.program.as_ref());
            assert!(child.observation.measured(), "{:?}", child.observation);
            assert_eq!(
                scores(child),
                scores(parent),
                "{} vs {}",
                child.observation.genome,
                parent.observation.genome
            );
            assert_eq!(m.rewrites.len(), 1);
            let rewrite = &m.rewrites[0];
            assert!(rewrite.expression.is_some() && rewrite.refusal.is_none());
            assert_eq!(rewrite.stop, "saturated");
            assert!(rewrite.iterations > 0 && rewrite.applications > 0);
            assert!(rewrite.enodes > 0 && rewrite.peak >= rewrite.enodes);
        }
    }
}

#[test]
fn refused_outputs_are_discarded_with_their_reason_and_admitted_stages_kept() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &[NEGATED, MIXED, SINE]);
    let before = state.budget.used();
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let record = |genome: &str| {
        let sha = crate::sha256_hex(genome.as_bytes());
        state
            .moves
            .iter()
            .find(|m| m.parent_sha256 == sha)
            .unwrap()
            .clone()
    };
    // The whole proposal refused: nothing assembled, nothing charged, the reason kept.
    let negated = record(NEGATED);
    assert_eq!(negated.outcome, REFUSED);
    assert!(negated.program.is_none() && negated.measurement.is_none());
    assert!(negated.reason.is_none());
    let refusal = negated.rewrites[0].refusal.as_deref().unwrap();
    assert!(refusal.contains("negative constant"), "{refusal}");
    assert!(negated.rewrites[0].expression.is_none());
    // One stage refused, the other admitted: the refused stage keeps its expression.
    let mixed = record(MIXED);
    assert_eq!(mixed.outcome, MEASURED);
    assert!(mixed.rewrites[0].refusal.is_some() && mixed.rewrites[0].expression.is_none());
    assert!(mixed.rewrites[1].refusal.is_none() && mixed.rewrites[1].expression.is_some());
    assert_eq!(mixed.program, Some(with_expression(MIXED, 1, "c0")));
    let child = &state.observations[mixed.measurement.unwrap()];
    assert_eq!(scores(child), scores(parent_of(&state, &mixed)));
    // Nothing smaller: unchanged, with the saturation counters still recorded.
    let sine = record(SINE);
    assert_eq!(sine.outcome, UNCHANGED);
    assert!(sine.program.is_none() && sine.measurement.is_none());
    assert!(sine.rewrites[0].expression.is_none() && sine.rewrites[0].refusal.is_none());
    assert!(sine.rewrites[0].iterations > 0);
    assert_eq!(state.budget.used() - before, 1);
    assert_eq!(state.budget.used(), state.observations.len());
    // The refusal is the profile's own: the same program with `-c0` is refused by admission.
    let refused = with_expression(NEGATED, 0, "-c0");
    assert!(admit_child(&refused, opt.evaluator.profile(), 1).is_err());
}

#[test]
fn the_allowance_and_the_start_guard_bound_the_pass() {
    let cut = data();
    let genomes = [ONE_TIMES, FOLDED, MAPPED];
    // One evaluation left: one child, then the pass stops before the next proposal.
    let opt = options(4, genomes.len() + 1, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 1);
    assert_eq!(state.moves[0].outcome, MEASURED);
    assert_eq!(state.budget.remaining(), 0);
    assert_eq!(state.budget.used(), state.observations.len());
    // None left: no proposal at all.
    let opt = options(4, genomes.len(), FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert!(state.moves.is_empty());
    // The guard refuses the first proposal: nothing is recorded or spent.
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    let used = state.budget.used();
    simplify_round(&mut state, &cut, &opt, &mut || false).unwrap();
    assert!(state.moves.is_empty());
    assert_eq!(state.budget.used(), used);
    // The guard admits the proposal and refuses its measurement: recorded as not started, and
    // the pass ends.
    let mut state = parents(&cut, &opt, &genomes);
    let mut calls = 0;
    simplify_round(&mut state, &cut, &opt, &mut || {
        calls += 1;
        calls == 1
    })
    .unwrap();
    assert_eq!(state.moves.len(), 1);
    assert_eq!(state.moves[0].outcome, NOT_STARTED);
    assert!(state.moves[0].program.is_some() && state.moves[0].measurement.is_none());
    assert_eq!(state.budget.used(), used);
    // Only `top_k` parents are read.
    let opt = options(1, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 1);
}

#[test]
fn proposals_are_identical_with_a_poisoned_holdout() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut poisoned = cut.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    // The parents are fixed: measured on the clean rows.
    let fixed = parents(&cut, &opt, &[ONE_TIMES, FOLDED, NEGATED, MAPPED]);
    let run = |c: &Cut| {
        let mut state = State::new(opt.search.budget.evaluations);
        state.budget = fixed.budget.clone();
        state.observations = fixed.observations.clone();
        state.seen = fixed.seen.clone();
        simplify_round(&mut state, c, &opt, &mut || true).unwrap();
        state
    };
    let (clean, dirty) = (run(&cut), run(&poisoned));
    assert_eq!(clean.moves, dirty.moves);
    assert_eq!(clean.budget.used(), dirty.budget.used());
    // The poison reached the children's measurements, not the proposals.
    let children = |s: &State| -> Vec<bool> {
        s.moves
            .iter()
            .filter_map(|m| m.measurement)
            .map(|i| s.observations[i].observation.measured())
            .collect()
    };
    assert_eq!(children(&clean), vec![true; 3]);
    assert_eq!(children(&dirty), vec![false; 3]);
    // A proposal reads the parent's text and the input width only.
    for genome in [ONE_TIMES, FOLDED, NEGATED, MAPPED] {
        let a = propose(&opt.evaluator, genome, 1).unwrap();
        let b = propose(&opt.evaluator, genome, 1).unwrap();
        assert_eq!(
            (a.rewrites, a.program, a.reason, a.admitted),
            (b.rewrites, b.program, b.reason, b.admitted)
        );
    }
}

// Runs through the published route.

/// The `cli_tests` seed program, which the pass reduces to `c0`.
pub(super) const SEED: &str = ONE_TIMES;

/// The files every run publishes.
pub(super) const SIDECARS: [&str; 7] = [
    "receipt.json",
    "prepared/prepared.json",
    "search/cut-0/search.json",
    "search/cut-0/observations.json",
    "search/cut-0/refinements.json",
    "emit/model.rs",
    "emit/manifest.json",
];
