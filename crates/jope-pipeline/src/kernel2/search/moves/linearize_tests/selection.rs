use super::super::linearize_round;
use super::*;

#[test]
fn a_line_that_misses_the_minimax_bound_is_refused_with_its_reason_and_nothing_is_measured() {
    let hold = [1e6 + 0.25, 1e6 + 0.5, 1e6 + 0.75];
    let cut = cut(1e6, 1e6 + 1.0, 201, &hold, |v| 2.0 * v - 2_000_001.0);
    let opt = options(4, 64, linearize());
    let parent = program(OFF_CENTRE);
    let mut state = parents(&cut, &opt, std::slice::from_ref(&parent));
    let (used, measured) = (state.budget.used(), state.observations.len());
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    // The root, then its two children, each considered and refused.
    assert_eq!(state.moves.len(), 3, "{:?}", state.moves);
    let root = &state.moves[0];
    let record = line(root);
    assert_eq!((record.stage, record.path.as_slice()), (0, &[][..]));
    assert_eq!(record.degree, 1);
    let fit = record.fit.as_ref().unwrap();
    assert!(
        ["exact_fit", "converged", "noise_limited"].contains(&fit.stop.as_str()),
        "{fit:?}"
    );
    let (lowered, bound) = (record.lowered_error.unwrap(), record.bound.unwrap());
    assert!(lowered > bound, "{lowered} <= {bound}");
    assert_eq!(
        bound.to_bits(),
        (fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * fit.max_abs).to_bits()
    );
    assert_eq!(record.refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    assert_eq!(root.outcome, REFUSED);
    assert!(root.program.is_none() && root.reason.is_none() && root.measurement.is_none());
    assert!(record.expression.is_none());
    for m in &state.moves[1..] {
        assert_eq!(m.outcome, REFUSED);
        assert_eq!(line(m).refusal.as_deref(), Some(LINEARIZE_NOT_SMALLER));
    }
    // Nothing measured, nothing charged.
    assert_eq!(
        (state.budget.used(), state.observations.len()),
        (used, measured)
    );
    // The same records re-derive from the parent and the recorded fits.
    let recorded: Vec<LinearizationRecord> = state.moves.iter().map(|m| line(m).clone()).collect();
    let again = linearize::rederive(&opt.evaluator, &parent, 1, &recorded).unwrap();
    assert!(again
        .iter()
        .zip(&state.moves)
        .all(|(p, m)| Some(&p.linearization) == m.linearization.as_ref() && !p.admitted));
}

#[test]
fn near_linear_subtrees_are_linearized_measured_and_ranked_by_their_native_score() {
    let cut = cubic_cut();
    let opt = options(4, 64, linearize());
    let (a, b) = (program(CUBE), program(CUBE_PLUS));
    let mut state = parents(&cut, &opt, &[a.clone(), b.clone()]);
    let before = state.budget.used();
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let (from_a, from_b) = (records_of(&state, &a), records_of(&state, &b));
    assert_eq!(from_a.len(), 1);
    assert_eq!(
        from_b
            .iter()
            .map(|m| line(m).path.clone())
            .collect::<Vec<_>>(),
        vec![vec![], vec![0], vec![1]]
    );
    assert_eq!(from_b[1].outcome, REFUSED);
    assert_eq!(
        line(from_b[1]).refusal.as_deref(),
        Some(LINEARIZE_NOT_SMALLER)
    );
    let offered: Vec<&MoveRecord> = [from_a[0], from_b[0], from_b[2]].into();
    for m in &offered {
        let record = line(m);
        assert_eq!(m.outcome, MEASURED, "{m:?}");
        assert!(record.lowered_error.unwrap() <= record.bound.unwrap());
        assert!(record.refusal.is_none() && m.reason.is_none());
        let o = child(&state, m);
        assert_eq!(o.observation.origin, crate::kernel2::LINEARIZE);
        assert_eq!(Some(&o.observation.genome), m.program.as_ref());
        assert!(o.observation.measured(), "{:?}", o.observation);
        // The child's score is its own native measurement.
        let native = opt.evaluator.evaluate(
            &o.observation.genome,
            &cut,
            state.generation,
            crate::kernel2::LINEARIZE,
            Clock::Frozen,
        );
        assert_eq!(&native, o);
        // The program is the parent with the stage's expression replaced, and that expression is
        // the line in place of the subtree.
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(m.program.as_ref().unwrap()).unwrap() else {
            panic!("pipeline expected")
        };
        let Some(ParamValue::Str(source)) = pipe.stages[0].params.get("expr") else {
            panic!("expression expected")
        };
        assert_eq!(Some(source), record.expression.as_ref());
    }
    // One evaluation per measured child; the shared identity holds.
    assert_eq!(state.budget.used() - before, 3);
    assert_eq!(state.budget.used(), state.observations.len());
    // The cube's line is far closer to its subtree than the quadratic part's line, yet the
    // program keeping the cube (x³ plus a line) ranks above the program that is a line: the
    // measured score decides, never the line's error.
    let (line_a, line_b) = (line(from_a[0]), line(from_b[2]));
    assert!(line_a.lowered_error.unwrap() < line_b.lowered_error.unwrap());
    let (child_a, child_b) = (child(&state, from_a[0]), child(&state, from_b[2]));
    assert!(score(child_b) > score(child_a), "{child_a:?} {child_b:?}");
    assert!(rank_of(&state, &opt, child_b) < rank_of(&state, &opt, child_a));
    // Linearizing the quadratic part also beats its own parent; the exact cube stays first.
    let parent_b = &state.observations[1];
    assert!(score(child_b) > score(parent_b));
    assert_eq!(rank_of(&state, &opt, &state.observations[0]), 0);
}

#[test]
fn a_line_offered_on_the_training_range_can_be_worse_off_it_and_the_native_score_decides() {
    // `x³ - 3x`: on the training range its line falls steeply; on the held-out rows, in [1.2, 2],
    // the curve turns back up through the training targets' own range while the line keeps
    // falling.
    let hold = [1.2, 1.4, 1.6, 1.8, 2.0];
    let cut = cut(-1.0, 1.05, 201, &hold, |v| v * v * v - 3.0 * v);
    let opt = options(4, 64, linearize());
    let parent = program(TURN);
    let mut state = parents(&cut, &opt, std::slice::from_ref(&parent));
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let m = state
        .moves
        .iter()
        .find(|m| line(m).path.is_empty())
        .unwrap();
    let record = line(m);
    // Offered: within the minimax bound on the training range.
    assert_eq!(m.outcome, MEASURED);
    let lowered = record.lowered_error.unwrap();
    assert!(lowered <= record.bound.unwrap());
    // Off that range the line drifts from the subtree far past its error on it.
    let subtree = MathExpr::parse(&record.subtree).unwrap();
    let admitted = MathExpr::parse(record.expression.as_ref().unwrap()).unwrap();
    let (s2, _) = subtree.eval_row_traced(&[2.0]);
    let (l2, _) = admitted.eval_row_traced(&[2.0]);
    assert!((s2 - l2).abs() > 2.0 * lowered, "{s2} vs {l2}, {lowered}");
    // The parent fits the held-out rows; the child does not, and ranks under it.
    let parent_o = &state.observations[0];
    let child_o = child(&state, m);
    assert!(child_o.observation.measured());
    let (parent_hold, child_hold) = (
        parent_o.observation.hold_r2.unwrap(),
        child_o.observation.hold_r2.unwrap(),
    );
    let both = format!("{parent_o:?}\n{child_o:?}");
    assert!(parent_hold > 0.9, "{both}");
    assert!(child_hold < parent_hold - 0.5, "{both}");
    assert!(score(child_o) < score(parent_o), "{both}");
    assert!(rank_of(&state, &opt, parent_o) < rank_of(&state, &opt, child_o));
}

#[test]
fn proposals_are_identical_with_a_poisoned_holdout() {
    let clean = cubic_cut();
    let opt = options(4, 64, linearize());
    let mut poisoned = clean.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    poisoned.hold_indices = vec![0, 1];
    // The parents are fixed: measured on the clean rows.
    let genomes = [program(CUBE), program(CUBE_PLUS)];
    let fixed = parents(&clean, &opt, &genomes);
    let run = |c: &Cut| {
        let mut state = State::new(opt.search.budget.evaluations);
        state.budget = fixed.budget.clone();
        state.observations = fixed.observations.clone();
        state.seen = fixed.seen.clone();
        linearize_round(&mut state, c, &opt, &mut || true).unwrap();
        state
    };
    let (a, b) = (run(&clean), run(&poisoned));
    assert_eq!(a.moves, b.moves);
    assert_eq!(a.budget.used(), b.budget.used());
    // The poison reached the children's measurements, not the proposals.
    let children = |s: &State| -> Vec<bool> {
        s.moves
            .iter()
            .filter_map(|m| m.measurement)
            .map(|i| s.observations[i].observation.measured())
            .collect()
    };
    assert_eq!(children(&a), vec![true; 3]);
    assert_eq!(children(&b), vec![false; 3]);
    // A proposal reads the parent's text and the training rows and labels only.
    for genome in &genomes {
        let x = linearize::propose(&opt.evaluator, genome, &clean, 1).unwrap();
        let y = linearize::propose(&opt.evaluator, genome, &poisoned, 1).unwrap();
        assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(&y) {
            assert_eq!(
                (&x.linearization, &x.program, &x.reason, x.admitted),
                (&y.linearization, &y.program, &y.reason, y.admitted)
            );
        }
    }
}

#[test]
fn the_allowance_the_start_guard_top_k_and_deduplication_bound_the_pass() {
    let cut = cubic_cut();
    let genomes = [program(CUBE), program(CUBE_SWAPPED), program(CUBE_PLUS)];
    // The swapped cube has the same values, so the same line and the same child: measured once.
    let opt = options(4, 64, linearize());
    let mut state = parents(&cut, &opt, &genomes);
    let before = state.budget.used();
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let outcomes: Vec<&str> = state.moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(
        outcomes.iter().filter(|o| **o == DUPLICATE).count(),
        1,
        "{outcomes:?}"
    );
    let duplicate = state.moves.iter().find(|m| m.outcome == DUPLICATE).unwrap();
    assert!(duplicate.measurement.is_none());
    assert!(state
        .moves
        .iter()
        .any(|m| m.outcome == MEASURED && m.program == duplicate.program));
    let measured = outcomes.iter().filter(|o| **o == MEASURED).count();
    assert_eq!(measured, 3);
    assert_eq!(state.budget.used() - before, measured);
    assert_eq!(state.budget.used(), state.observations.len());
    let distinct: BTreeSet<&str> = state
        .observations
        .iter()
        .map(|o| o.observation.genome.as_str())
        .collect();
    assert_eq!(distinct.len(), state.observations.len());
    // One evaluation left for a parent with two offers: the first is measured, the second
    // records the spent allowance, and the pass ends.
    let plus = [program(CUBE_PLUS)];
    let opt = options(4, plus.len() + 1, linearize());
    let mut state = parents(&cut, &opt, &plus);
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let outcomes: Vec<&str> = state.moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(outcomes, [MEASURED, REFUSED, NO_ALLOWANCE]);
    assert!(state.moves[2].program.is_some() && state.moves[2].measurement.is_none());
    assert_eq!(state.budget.remaining(), 0);
    assert_eq!(state.budget.used(), state.observations.len());
    // None left: no proposal at all.
    let opt = options(4, plus.len(), linearize());
    let mut state = parents(&cut, &opt, &plus);
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert!(state.moves.is_empty());
    // The guard refuses the first parent: nothing is recorded or spent.
    let opt = options(4, 64, linearize());
    let mut state = parents(&cut, &opt, &plus);
    let used = state.budget.used();
    linearize_round(&mut state, &cut, &opt, &mut || false).unwrap();
    assert!(state.moves.is_empty());
    assert_eq!(state.budget.used(), used);
    // The guard admits the parent and refuses its first measurement: every offer of that parent
    // is recorded unstarted, and the pass ends.
    let mut state = parents(&cut, &opt, &plus);
    let mut calls = 0;
    linearize_round(&mut state, &cut, &opt, &mut || {
        calls += 1;
        calls == 1
    })
    .unwrap();
    let outcomes: Vec<&str> = state.moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(outcomes, [NOT_STARTED, REFUSED, NOT_STARTED]);
    assert_eq!(calls, 2);
    assert_eq!(state.budget.used(), used);
    // Only `top_k` parents are read: the exact cube ranks first.
    let opt = options(1, 64, linearize());
    let mut state = parents(&cut, &opt, &[program(CUBE_PLUS), program(CUBE)]);
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let read: BTreeSet<&str> = state
        .moves
        .iter()
        .map(|m| m.parent_sha256.as_str())
        .collect();
    assert_eq!(
        read,
        BTreeSet::from([crate::sha256_hex(program(CUBE).as_bytes()).as_str()])
    );
}

// Runs through the published route.

/// The `cli_tests` seed program, which the rewrite pass reduces to `c0`.
pub(super) const ONE_TIMES: &str = "(pipe X (derive expr=\"1*c0\") (ridge))";
/// A seed whose 6-node expression is a curve the linearization pass can replace by its line. It
/// reads `c1`, which takes three values on this dataset; `c0` takes two, too few for a line's
/// minimax fit.
pub(super) const SINE: &str = "(sin((0.5*c1))*3.0)";

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
