use super::super::{pool_count, simplify_round};
use super::*;

// Fix round 01: parents that raise faults, inputs the rewrite cannot read, the pool refusal
// count, and the set and order of the published records.

/// `(pipe X (<op> expr="<source>") (ridge))`, as the search prints it.
pub(super) fn carrier(op: &str, source: &str) -> String {
    let expression = MathExpr::parse(source).unwrap().to_source().unwrap();
    GenomeIr::parse(&format!("(pipe X ({op} expr=\"{expression}\") (ridge))"))
        .unwrap()
        .canonical()
}

/// Three columns with a `0.0` and a `-0.0` in `c0`, values of `c0` under -1, negative values of
/// `c1`, and a row whose columns are equal (a zero row standard deviation).
pub(super) fn faulting_cut() -> Cut {
    let row = |i: usize| -> Vec<f64> {
        let t = i as f64;
        match i {
            3 => vec![-0.0, -0.0, 0.5],
            7 => vec![1.0, 1.0, 1.0],
            _ => vec![(t - 12.0) / 4.0, (i % 5) as f64 - 2.0, (0.7 * t).sin()],
        }
    };
    let target = |r: &Vec<f64>| r[0] + 0.5 * r[1] - r[2];
    let train_x: Vec<Vec<f64>> = (0..24).map(row).collect();
    let hold_x: Vec<Vec<f64>> = (24..32).map(row).collect();
    Cut {
        index: 2,
        train_y: train_x.iter().map(target).collect(),
        hold_y: hold_x.iter().map(target).collect(),
        hold_indices: (0..hold_x.len()).collect(),
        train_x,
        hold_x,
    }
}

#[test]
fn rewritten_children_keep_fault_counts_and_signed_zeros_bit_for_bit() {
    use feat_math::math_expr::FaultClass;
    let cut = faulting_cut();
    assert!(
        cut.train_x
            .iter()
            .any(|r| r[0] == 0.0 && r[0].is_sign_negative())
    );
    // Each parent, the expression its rewrite gives, and the fault class its own expression
    // raises on a training row (none checked for the elementwise carrier, whose cells the
    // row-summary evaluator reads).
    let cases: [(String, &str, Option<FaultClass>); 5] = [
        (
            carrier("derive", "(1.0*c1)/(c0*1.0)"),
            "c1/c0",
            Some(FaultClass::DivNearZero),
        ),
        (
            carrier("derive", "log1p(c0/1.0)"),
            "log1p(c0)",
            Some(FaultClass::Log1pDomain),
        ),
        (
            carrier("derive", "sqrt(c1*1.0)"),
            "sqrt(c1)",
            Some(FaultClass::SqrtProjected),
        ),
        // `c0 + (-0.0)` keeps the sign of every zero, so the division sees `-0.0` as before.
        (
            carrier("derive", "1.0/(c0+(-0.0))"),
            "1.0/c0",
            Some(FaultClass::DivNearZero),
        ),
        // A `map` stage that reads row summaries runs on the compiled evaluator.
        (
            carrier("map", "(x-row_mean)/(row_std*1.0)"),
            "(x-row_mean)/row_std",
            None,
        ),
    ];
    for policy in [FaultPolicy::RejectHard, penalize()] {
        let opt = options(1, 64, policy);
        for (parent, rewritten, class) in &cases {
            let mut state = parents(&cut, &opt, &[parent.as_str()]);
            let original = state.observations[0].clone();
            assert!(
                original.faults.domain_cells > 0,
                "{parent}: {:?}",
                original.faults
            );
            if let Some(class) = class {
                let GenomeIr::Pipeline(pipe) = GenomeIr::parse(parent).unwrap() else {
                    panic!("{parent} is not a pipeline");
                };
                let Some(ParamValue::Str(source)) = pipe.stages[0].params.get("expr") else {
                    panic!("{parent} has no expression");
                };
                let expression = MathExpr::parse(source).unwrap();
                assert!(
                    cut.train_x
                        .iter()
                        .any(|row| expression.eval_row_traced(row).1.contains(class.set())),
                    "{parent} raises no {}",
                    class.name()
                );
            }
            simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
            assert_eq!(state.moves.len(), 1, "{parent}");
            let m = &state.moves[0];
            assert_eq!(m.outcome, MEASURED, "{parent}: {m:?}");
            assert_eq!(m.program, Some(with_expression(parent, 0, rewritten)));
            let child = &state.observations[m.measurement.unwrap()];
            assert_eq!(child.observation.origin, crate::kernel2::REWRITE);
            // Status, R², RMSE, selection score and every fault count, bit for bit.
            assert_eq!(scored(child), scored(&original), "{policy:?} {parent}");
            assert_eq!(scores(child), scores(&original), "{policy:?} {parent}");
        }
    }
}

/// `c0/(-1.0)`, which the pass reduces to `-c0`, a negation of a variable the profile refuses.
pub(super) const DIVIDED: &str = "(pipe X (derive expr=\"c0/(-1.0)\") (ridge))";
/// `x*(-1.0)` in an elementwise carrier, which the pass reduces to `-x`, refused the same way.
pub(super) const MAPPED_NEGATED: &str = "(pipe X (map expr=\"x*(-1.0)\") (ridge))";

#[test]
fn the_pool_refusal_count_of_fixed_parents_is_pinned() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &[NEGATED, MIXED, DIVIDED, MAPPED_NEGATED]);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 4);
    // One smaller expression dropped per parent: the negation of a variable each reduces to.
    assert_eq!(refused_by_pool(&state.moves), 4);
    let dropped: Vec<&StageRewrite> = state
        .moves
        .iter()
        .flat_map(|m| &m.rewrites)
        .filter(|r| r.refused_by_pool())
        .collect();
    assert_eq!(dropped.len(), 4);
    for r in dropped {
        let refusal = r.refusal.as_deref().unwrap();
        assert!(refusal.contains("negative constant"), "{refusal}");
        assert!(r.expression.is_none() && r.stop != NOT_RUN && r.applications > 0);
    }
    // The count is what publication writes and `verify_run` counts again.
    let ids = [REWRITE_SATURATE.to_owned()];
    assert_eq!(pool_count(&ids, &state.moves), Some(4));
    assert_eq!(
        pool_count(&[MINIMAX_LINEARIZE.to_owned()], &state.moves),
        None
    );
    // A stage the rewrite cannot read is refused, not counted.
    let unreadable = propose(
        &opt.evaluator,
        "(pipe X (derive expr=\"exp(c0)\") (ridge))",
        1,
    )
    .unwrap();
    assert!(!unreadable.rewrites[0].refused_by_pool());
}

/// `exp(c0)`: a function outside the kernel2 pool.
pub(super) const OUTSIDE_POOL: &str = "(pipe X (derive expr=\"exp(c0)\") (ridge))";
/// 16 nodes at depth 4: over the 15-node cap.
pub(super) const OVER_CAPS: &str =
    "(pipe X (derive expr=\"clip((c0*1.0),(c0*1.0),(c0*1.0))+(c0*(c0*1.0))\") (ridge))";

#[test]
fn a_parent_the_rewrite_cannot_read_is_refused_and_the_pass_goes_on() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    for (text, why) in [(OUTSIDE_POOL, "function"), (OVER_CAPS, "node")] {
        // `simplify_v2` refuses the expression itself: the stage is refused, not the run.
        let proposal = propose(&opt.evaluator, text, 1).unwrap();
        assert!(!proposal.admitted && proposal.program.is_none() && proposal.reason.is_none());
        assert_eq!(proposal.outcome(), REFUSED);
        let r = &proposal.rewrites[0];
        assert_eq!(r.stop, NOT_RUN);
        assert_eq!(
            (r.iterations, r.enodes, r.applications, r.peak),
            (0, 0, 0, 0)
        );
        assert!(r.expression.is_none() && !r.refused_by_pool());
        let refusal = r.refusal.as_deref().unwrap();
        assert!(
            refusal.starts_with(REWRITE_FAILED) && refusal.len() > REWRITE_FAILED.len() + 2,
            "{refusal}"
        );
        // The profile's own reason: the pool's functions, or the node cap.
        assert!(refusal.contains(why), "{refusal}");
        lint(refusal);
        // The same proposal again: what `check` re-derives.
        assert_eq!(
            propose(&opt.evaluator, text, 1).unwrap().rewrites,
            proposal.rewrites
        );
        // In a round, ranked first: refused, and the pass reads the next parent.
        let mut state = parents(&cut, &opt, &[ONE_TIMES]);
        let mut unreadable = state.observations[0].clone();
        unreadable.observation.genome = text.into();
        unreadable.observation.program_sha256 = crate::sha256_hex(text.as_bytes());
        unreadable.selection_score = unreadable.selection_score.map(|s| s + 1.0);
        state.observations.push(unreadable);
        let used = state.budget.used();
        simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
        assert_eq!(state.moves.len(), 2, "{text}");
        assert_eq!(
            state.moves[0].parent_sha256,
            crate::sha256_hex(text.as_bytes())
        );
        assert_eq!(state.moves[0].outcome, REFUSED);
        assert_eq!(state.moves[0].rewrites, proposal.rewrites);
        assert_eq!(state.moves[1].outcome, MEASURED);
        assert_eq!(state.budget.used(), used + 1);
        assert_eq!(refused_by_pool(&state.moves), 0);
        // The receipt's not-run count sees it.
        assert_eq!(not_run(&state.moves), 1);
        assert_eq!(
            not_run_count(&[REWRITE_SATURATE.to_owned()], &state.moves),
            Some(1)
        );
    }
}
