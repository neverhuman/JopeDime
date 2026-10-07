//! The bounded minimax linearization, degree 1 only (laneD:B19-minimax-linearize).
use super::*;
use crate::kernel::profile::profile;
use anyhow::Context;
use std::collections::BTreeSet;

fn baseline() -> SearchProfile {
    profile("kernel2-baseline", 3).unwrap()
}

/// The kernel2 caps (15 nodes, depth 4) over two columns.
fn scope() -> ExpressionScope {
    ExpressionScope::search_profile(2, false)
}

/// `points` rows with `c0` evenly spaced over `[lo, hi]` and `c1` fixed.
fn rows(lo: f64, hi: f64, points: usize) -> Vec<Vec<f64>> {
    (0..points)
        .map(|i| vec![lo + (hi - lo) * i as f64 / (points - 1) as f64, 0.25])
        .collect()
}

fn linearize(source: &str, rows: &[Vec<f64>]) -> Vec<SubtreeLinearization> {
    MathExpr::parse(source)
        .unwrap()
        .linearize_v2(&baseline(), scope(), rows)
        .unwrap()
}

/// What a record keeps of the live measurements.
fn evidence(l: &SubtreeLinearization) -> LinearizationEvidence {
    LinearizationEvidence {
        fit: l.fit,
        lowered_error: l.lowered_error,
        refusal: l.refusal.clone(),
    }
}

fn recorded(
    source: &str,
    evidence: &[LinearizationEvidence],
) -> anyhow::Result<Vec<SubtreeLinearization>> {
    MathExpr::parse(source)
        .unwrap()
        .linearize_recorded_v2(&baseline(), scope(), &mut |i| {
            evidence.get(i).cloned().context("missing")
        })
}

/// A 6-node expression that is nearly linear on [-0.5, 1.5].
const NEAR_LINEAR: &str = "sin((0.1*c0))*3.0";
/// A 6-node one-variable subtree beside a second variable.
const BESIDE: &str = "(sin(c0)+(c0*c0))+c1";

/// The native line at `path` of an admitted expression stays within the record's own measured
/// error of the subtree on every training row.
fn assert_within_lowered_error(l: &SubtreeLinearization, training: &[Vec<f64>]) {
    let admitted = l.expression.as_ref().unwrap();
    let native = MathExpr::from_ast(at(&admitted.root, &l.path).clone()).unwrap();
    let lowered = l.lowered_error.unwrap();
    for row in training {
        let (a, faults) = native.eval_row_traced(row);
        let (b, _) = l.subtree.eval_row_traced(row);
        assert!(faults.is_empty());
        assert!((a - b).abs() <= lowered, "{row:?}");
    }
}

#[test]
fn a_near_linear_subtree_becomes_its_minimax_line_within_the_bound() {
    let training = rows(-0.5, 1.5, 201);
    let expression = MathExpr::parse(NEAR_LINEAR).unwrap();
    // The root, and `sin(0.1*c0)`; `0.1*c0` is too small.
    assert_eq!(
        expression.linearizable_v2(&baseline(), scope()).unwrap(),
        vec![vec![], vec![0]]
    );
    let out = linearize(NEAR_LINEAR, &training);
    assert_eq!(
        out.iter().map(|l| l.path.clone()).collect::<Vec<_>>(),
        vec![vec![], vec![0]]
    );
    let line = &out[0];
    assert_eq!(line.subtree, expression);
    let fit = line.fit.unwrap();
    assert!(fit.stop.closed(), "{fit:?}");
    assert!((fit.lo, fit.hi) == (-0.5, 1.5), "{fit:?}");
    // Degree 1: the slope of 3 sin(0.1 x) is about 0.3.
    assert!((fit.coefficients[1] - 0.3).abs() < 1e-3, "{fit:?}");
    let (lowered, bound) = (line.lowered_error.unwrap(), fit.bound());
    assert!(lowered <= bound, "{lowered} > {bound}");
    assert_eq!(
        bound.to_bits(),
        (fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * fit.max_abs).to_bits()
    );
    assert!(
        fit.levelled <= fit.max_error && fit.max_error < 1e-3,
        "{fit:?}"
    );
    assert_eq!((line.nodes, line.line_nodes), (6, Some(5)));
    assert!(line.refusal.is_none());
    // The admitted expression is the line itself: `x * a1 + a0`, ordinary nodes.
    let admitted = line.expression.as_ref().unwrap();
    assert_eq!((count_nodes(&admitted.root), depth(&admitted.root)), (5, 3));
    assert_within_lowered_error(line, &training);
    // The 4-node `sin(0.1*c0)` gets a 5-node line: not smaller.
    let small = &out[1];
    assert_eq!(small.refusal.as_deref(), Some(LINEARIZE_NOT_SMALLER));
    assert!(small.fit.unwrap().stop.closed());
    assert_eq!((small.line_nodes, small.lowered_error), (Some(5), None));
    assert!(small.expression.is_none());
    // Deterministic.
    assert_eq!(linearize(NEAR_LINEAR, &training), out);
    // Under a root reading two variables, the line is spliced in place of its subtree.
    let beside = linearize(BESIDE, &training);
    assert_eq!(beside.len(), 1);
    assert_eq!(beside[0].path, vec![0]);
    let admitted = beside[0].expression.as_ref().unwrap();
    assert_eq!((count_nodes(&admitted.root), depth(&admitted.root)), (7, 4));
    assert_eq!(at(&admitted.root, &[1]), &Expr::Var(Var::Col(1)));
    assert_within_lowered_error(&beside[0], &training);
}

/// Exactly linear plus `1e-6 sin(c0)` on [1e6, 1e6 + 1]: the fit's error is about 1e-7, and the
/// monomial line `2 x - 2000001` loses about 1e-10 to rounding at this offset, far over the
/// bound's `1e-6` relative and `64 ε max|f|` (1.4e-14) allowances.
const OFF_CENTRE: &str = "((c0*2.0)-2000001.0)+(sin(c0)*0.000001)";

#[test]
fn a_line_that_misses_the_minimax_bound_on_the_training_range_is_refused() {
    let training = rows(1e6, 1e6 + 1.0, 201);
    let out = linearize(OFF_CENTRE, &training);
    assert_eq!(
        out.iter().map(|l| l.path.clone()).collect::<Vec<_>>(),
        vec![vec![], vec![0], vec![1]]
    );
    let root = &out[0];
    let fit = root.fit.unwrap();
    assert!(fit.stop.closed(), "{fit:?}");
    let lowered = root.lowered_error.unwrap();
    assert!(lowered > fit.bound(), "{lowered} <= {}", fit.bound());
    assert_eq!(root.refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    assert_eq!(root.line_nodes, Some(5));
    assert!(root.expression.is_none());
    // The fit itself is fine: the refusal is the lowered line's own error.
    assert!(fit.max_error < 1e-6, "{fit:?}");
    // The exactly linear part is fitted to a few ulps, but its line is no smaller.
    assert_eq!(out[1].fit.unwrap().stop, MinimaxStop::ExactFit);
    assert_eq!(out[1].refusal.as_deref(), Some(LINEARIZE_NOT_SMALLER));
    assert_eq!(out[2].refusal.as_deref(), Some(LINEARIZE_NOT_SMALLER));
    // The same expression on a centred range offers the line.
    let centred = linearize("((c0*2.0)-1.0)+(sin(c0)*0.000001)", &rows(-1.0, 1.0, 201));
    assert!(centred[0].expression.is_some(), "{:?}", centred[0]);
    assert!(centred[0].lowered_error.unwrap() <= centred[0].fit.unwrap().bound());
}

#[test]
fn a_fit_whose_bracket_did_not_close_is_refused_before_any_line_is_built() {
    assert_eq!(
        MinimaxStop::ALL
            .into_iter()
            .filter(|s| s.closed())
            .collect::<Vec<_>>(),
        [
            MinimaxStop::ExactFit,
            MinimaxStop::Converged,
            MinimaxStop::NoiseLimited
        ]
    );
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    assert!(live[0].expression.is_some());
    for stop in [
        MinimaxStop::Stalled,
        MinimaxStop::Degenerate,
        MinimaxStop::IterationLimit,
    ] {
        let mut kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
        let fit = kept[0].fit.as_mut().unwrap();
        fit.stop = stop;
        fit.steps = MAX_ITERATIONS;
        let out = recorded(NEAR_LINEAR, &kept).unwrap();
        let open = &out[0];
        assert_eq!(
            open.refusal.as_deref(),
            Some(format!("{LINEARIZE_OPEN_BRACKET}: {}", stop.as_str()).as_str())
        );
        assert_eq!(
            (
                open.line_nodes,
                open.lowered_error,
                open.expression.as_ref()
            ),
            (None, None, None)
        );
        assert_eq!(open.fit.unwrap().stop, stop);
    }
}

#[test]
fn the_recorded_path_re_derives_the_live_linearization() {
    let cases = [
        (NEAR_LINEAR, rows(-0.5, 1.5, 201)),
        (BESIDE, rows(-0.5, 1.5, 201)),
        (OFF_CENTRE, rows(1e6, 1e6 + 1.0, 201)),
        // A domain fault on a training row, and a fit with too few distinct values.
        ("(sqrt(c0)*2.0)+c1", rows(-1.0, 1.0, 21)),
        (NEAR_LINEAR, rows(0.0, 1.0, 2)),
        // A negative slope one level under the root: depth 5, refused by the profile.
        ("((0.25-c0)*(c0+2.0))+c1", rows(0.0, 1.0, 101)),
        ("(0.25-c0)*(c0+2.0)", rows(0.0, 1.0, 101)),
    ];
    let mut reasons = BTreeSet::new();
    for (source, training) in &cases {
        let live = linearize(source, training);
        assert!(!live.is_empty(), "{source}");
        let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
        assert_eq!(recorded(source, &kept).unwrap(), live, "{source}");
        for l in &live {
            let reason = l.refusal.as_deref().unwrap_or("offered");
            reasons.insert(reason.split(':').next().unwrap().to_owned());
        }
        // Too few records, or a record without a fit and without a refusal, is refused.
        assert!(recorded(source, &kept[..kept.len() - 1]).is_err());
        let mut bare = kept.clone();
        bare[0] = LinearizationEvidence {
            fit: None,
            lowered_error: None,
            refusal: None,
        };
        assert!(recorded(source, &bare).is_err(), "{source}");
    }
    let expected: BTreeSet<String> = [
        "offered",
        LINEARIZE_UNSAMPLED,
        LINEARIZE_FIT_REFUSED,
        LINEARIZE_NOT_SMALLER,
        LINEARIZE_MISSES_BOUND,
        LINEARIZE_PROFILE_REFUSED,
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(reasons, expected);
    // The fault and the fit's own refusal, by name.
    let faulted = linearize("(sqrt(c0)*2.0)+c1", &rows(-1.0, 1.0, 21));
    assert_eq!(faulted[0].refusal.as_deref(), Some(LINEARIZE_UNSAMPLED));
    assert!(faulted[0].fit.is_none());
    let few = linearize(NEAR_LINEAR, &rows(0.0, 1.0, 2));
    assert_eq!(
        few[0].refusal.as_deref(),
        Some("the fit refused its samples: fewer than degree + 2 distinct abscissae")
    );
    // The profile's own refusal, by its depth limit; at the root the same line is offered.
    let deep = linearize("((0.25-c0)*(c0+2.0))+c1", &rows(0.0, 1.0, 101));
    let refusal = deep[0].refusal.as_deref().unwrap();
    assert!(refusal.starts_with(LINEARIZE_PROFILE_REFUSED), "{refusal}");
    assert!(deep[0].fit.unwrap().coefficients[1] < 0.0);
    assert_eq!(deep[0].line_nodes, Some(6));
    let root = linearize("(0.25-c0)*(c0+2.0)", &rows(0.0, 1.0, 101));
    assert!(root[0].expression.is_some(), "{:?}", root[0]);
}

#[test]
fn recorded_numbers_drive_the_bound_and_must_agree_with_their_stop() {
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    // A recorded error over the bound refuses the line, as the live check would.
    let mut over = kept.clone();
    let fit = over[0].fit.unwrap();
    over[0].lowered_error = Some(fit.bound() * 2.0);
    let out = recorded(NEAR_LINEAR, &over).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    assert!(out[0].expression.is_none());
    // A missing measured error is the out-of-range refusal, for a line that leaves the range at a
    // sampled end. A line in range at both ends is in range at every sample between them, so its
    // missing error is refused (review 04).
    let mut missing = kept.clone();
    missing[0].fit.as_mut().unwrap().coefficients = [0.0, 4.0 * MAX_ABS];
    missing[0].lowered_error = None;
    let out = recorded(NEAR_LINEAR, &missing).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_OUT_OF_RANGE));
    let mut erased = kept.clone();
    erased[0].lowered_error = None;
    let error = recorded(NEAR_LINEAR, &erased).unwrap_err();
    assert!(
        format!("{error:#}").contains("no measured error is in the evaluator's range"),
        "{error:#}"
    );
    // Changed coefficients change the line, and so the admitted expression. The intercept moves
    // by 1e-12 and the recorded error covers the moved line's least residual (reviews 02 to 04),
    // well within the bound's room.
    let mut moved = kept.clone();
    let line = moved[0].fit.as_mut().unwrap();
    line.coefficients[0] += 1e-12;
    let least = least_error(&live[0], line);
    moved[0].lowered_error = Some(least.max(kept[0].lowered_error.unwrap()));
    let out = recorded(NEAR_LINEAR, &moved).unwrap();
    assert!(out[0].expression.is_some());
    assert_ne!(out[0].expression, live[0].expression);
    // Non-finite coefficients are refused by name.
    let mut nan = kept.clone();
    nan[0].fit.as_mut().unwrap().coefficients = [f64::NAN; 2];
    let out = recorded(NEAR_LINEAR, &nan).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_NON_FINITE));
    // A fit that disagrees with its stop is refused outright.
    let edits: [fn(&mut LinearFit); 7] = [
        |f| f.levelled = f.max_error * 2.0,
        |f| f.max_error = f64::INFINITY,
        |f| f.hi = f.lo,
        |f| f.iterations = 0,
        |f| f.steps = MAX_ITERATIONS + 1,
        |f| {
            f.stop = MinimaxStop::ExactFit;
            f.max_error = 1e-3;
        },
        |f| {
            f.stop = MinimaxStop::Converged;
            f.levelled = 0.0;
        },
    ];
    for edit in edits {
        let mut bad = kept.clone();
        edit(bad[0].fit.as_mut().unwrap());
        assert!(recorded(NEAR_LINEAR, &bad).is_err(), "{:?}", bad[0]);
    }
}

#[test]
fn assess_names_the_first_check_that_fails() {
    let x = Expr::Var(Var::Col(0));
    // A 9-node stand-in for the replaced subtree.
    let big = (0..4).fold(x.clone(), |acc, _| {
        Expr::Binary(BinOp::Add, Box::new(acc), Box::new(x.clone()))
    });
    let line = [0.5, 2.0];
    let ok = |_: &[f64]| Some(1e-3);
    let reason = |r: Result<Proposal, Refused>| r.map(|p| p.nodes).map_err(|r| r.reason);
    assert_eq!(reason(assess(&line, 1e-3, 1.0, &x, &big, 15, ok)), Ok(5));
    let square = Expr::Binary(BinOp::Mul, Box::new(x.clone()), Box::new(x.clone()));
    assert_eq!(
        reason(assess(&line, 1e-3, 1.0, &square, &big, 15, ok)),
        Err(NOT_A_LEAF)
    );
    assert_eq!(
        reason(assess(&[0.5, f64::NAN], 1e-3, 1.0, &x, &big, 15, ok)),
        Err(LINEARIZE_NON_FINITE)
    );
    assert_eq!(
        reason(assess(&line, 1e-3, 1.0, &x, &square, 15, ok)),
        Err(LINEARIZE_NOT_SMALLER)
    );
    assert_eq!(
        reason(assess(&line, 1e-3, 1.0, &x, &big, 4, ok)),
        Err(LINEARIZE_OVER_BUDGET)
    );
    // Degree 8 lowers to depth 17, over the expression limits.
    let nine = [0.5; 9];
    let huge = (0..40).fold(x.clone(), |acc, _| {
        Expr::Binary(BinOp::Add, Box::new(acc), Box::new(x.clone()))
    });
    assert_eq!(
        reason(assess(&nine, 1e-3, 1.0, &x, &huge, 64, ok)),
        Err(LINEARIZE_OVER_LIMITS)
    );
    assert_eq!(
        reason(assess(&line, 1e-3, 1.0, &x, &big, 15, |_| None)),
        Err(LINEARIZE_OUT_OF_RANGE)
    );
    // The bound is the allowance on max_error: equal passes, one ulp over fails.
    let bound = allowance(1e-3, 1.0);
    assert_eq!(
        reason(assess(&line, 1e-3, 1.0, &x, &big, 15, |_| Some(bound))),
        Ok(5)
    );
    let refused = assess(&line, 1e-3, 1.0, &x, &big, 15, |_| Some(bound.next_up())).unwrap_err();
    assert_eq!(
        refused,
        Refused {
            reason: LINEARIZE_MISSES_BOUND,
            nodes: Some(5),
            lowered_error: Some(bound.next_up()),
        }
    );
}

#[test]
fn only_one_variable_subtrees_where_a_line_fits_are_considered() {
    let paths = |source: &str| {
        MathExpr::parse(source)
            .unwrap()
            .linearizable_v2(&baseline(), scope())
            .unwrap()
    };
    // Two variables at the root, none considered below a depth a line cannot fit.
    assert!(paths("(c0*c1)+(c0*c1)").is_empty());
    // Three nodes or fewer: never smaller as a line.
    assert!(paths("(c0*2.0)+c1").is_empty());
    // The root and its children, in preorder; a grandchild only with three arguments.
    assert_eq!(paths("((c0*c0)-(c0*0.3))+(c0*c0)"), vec![vec![], vec![0]]);
    assert_eq!(
        paths("(clip(c0,0.5,2.0)*3.0)+c1"),
        vec![vec![0], vec![0, 0]]
    );
    // The input must pass the profile's own admission.
    assert!(MathExpr::parse("-(c0)")
        .unwrap()
        .linearizable_v2(&baseline(), scope())
        .is_err());
}

// Fix round 01: a recorded fit stays in the evaluator's range, and the records of one expression
// agree about the samples they share.

#[test]
fn a_recorded_fit_must_stay_in_the_evaluators_range() {
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(NEAR_LINEAR, &kept).unwrap(), live);
    assert!(kept.iter().filter(|e| e.fit.is_some()).count() > 1);
    let refused = |bad: &[LinearizationEvidence]| {
        let error = recorded(NEAR_LINEAR, bad).unwrap_err();
        assert!(
            format!("{error:#}").contains("a recorded fit disagrees with its stop"),
            "{error:#}"
        );
    };
    // An exact-fit claim needs `max_error <= 4 eps max|f|`; a max|f| inflated past the range met
    // it, and widened the bound with it.
    let mut exact = kept.clone();
    let fit = exact[0].fit.as_mut().unwrap();
    fit.stop = MinimaxStop::ExactFit;
    fit.max_abs = (fit.max_error / (4.0 * f64::EPSILON) * 1.5).max(2.0 * MAX_ABS);
    refused(&exact);
    // max|f|, lo or hi past the range, on every fit of the variable alike (so that the shared
    // training range still agrees).
    let edits: [fn(&mut LinearFit); 3] = [
        |f| f.max_abs = 2.0 * MAX_ABS,
        |f| f.lo = -2.0 * MAX_ABS,
        |f| f.hi = 2.0 * MAX_ABS,
    ];
    for edit in edits {
        let mut bad = kept.clone();
        for e in &mut bad {
            if let Some(fit) = e.fit.as_mut() {
                edit(fit);
            }
        }
        refused(&bad);
    }
    // At the range's edge the range check accepts max|f|. The fit's error is then within the
    // exact-fit test, so only an `ExactFit` stop is consistent, and a target that large forces
    // the line's residual up to `max|f| - M` (review 03): the record is consistent only with both,
    // and its line then misses the bound. A sine's targets stay within 3 (review 06), so this
    // takes a subtree with no such bound.
    let unbounded = linearize(UNBOUNDED, &training);
    let mut edge: Vec<LinearizationEvidence> = unbounded.iter().map(evidence).collect();
    assert_eq!(recorded(UNBOUNDED, &edge).unwrap(), unbounded);
    let fit = edge[0].fit.as_mut().unwrap();
    fit.max_abs = MAX_ABS;
    assert!(fit.max_error <= 4.0 * f64::EPSILON * MAX_ABS);
    let refusal = recorded(UNBOUNDED, &edge).unwrap_err();
    assert!(format!("{refusal:#}").contains("a recorded fit disagrees with its stop"));
    let fit = edge[0].fit.as_mut().unwrap();
    fit.stop = MinimaxStop::ExactFit;
    let least = MAX_ABS - largest_end(fit);
    edge[0].lowered_error = Some(least);
    let out = recorded(UNBOUNDED, &edge).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
}

/// A 9-node expression, nearly linear on [-0.5, 1.5], whose values have no bound short of the
/// evaluator's range: no bounded call holds its variable.
const UNBOUNDED: &str = "((c0*c0)*0.001)+(c0*0.3)";

/// `(sin(c0)+c0)` twice: the root and two equal subtrees are considered.
const REPEATED: &str = "(sin(c0)+c0)*(sin(c0)+c0)";

#[test]
fn records_of_one_expression_must_agree_about_the_samples_they_share() {
    let training = rows(-0.5, 1.5, 201);
    let refused = |source: &str, bad: &[LinearizationEvidence], expected: &str| {
        let error = recorded(source, bad).unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
    };
    let unsampled = LinearizationEvidence {
        fit: None,
        lowered_error: None,
        refusal: Some(LINEARIZE_UNSAMPLED.into()),
    };
    // A repeated subtree has the same samples, so the same record.
    let live = linearize(REPEATED, &training);
    assert_eq!(live.len(), 3);
    assert_eq!(live[1].subtree, live[2].subtree);
    assert!(live[1].fit.is_some());
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(REPEATED, &kept).unwrap(), live);
    let repeated = "a repeated subtree records other samples";
    let mut hidden = kept.clone();
    hidden[2] = unsampled.clone();
    refused(REPEATED, &hidden, repeated);
    let mut other = kept.clone();
    other[2].fit.as_mut().unwrap().coefficients[0] += 0.25;
    refused(REPEATED, &other, repeated);
    // A subtree that occurs once has no such witness: its fit hidden as unsampled is accepted.
    let mut root = kept.clone();
    root[0] = unsampled;
    let out = recorded(REPEATED, &root).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_UNSAMPLED));
    // The subtrees of one variable read its values on the same rows: one training range.
    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert!(kept[0].fit.is_some() && kept[1].fit.is_some());
    let mut moved = kept.clone();
    moved[1].fit.as_mut().unwrap().lo += 1e-3;
    refused(
        NEAR_LINEAR,
        &moved,
        "subtrees of one variable record different training ranges",
    );
    // A fit of the variable, and a refusal of too few distinct values of that same variable.
    for reason in [FEW_ABSCISSAE, EMPTY_INTERVAL] {
        let mut few = kept.clone();
        few[1] = LinearizationEvidence {
            fit: None,
            lowered_error: None,
            refusal: Some(format!("{LINEARIZE_FIT_REFUSED}: {reason}")),
        };
        refused(
            NEAR_LINEAR,
            &few,
            "a fit and a refusal of the same variable's values disagree",
        );
    }
    // Without a fit beside it, that refusal stands, as the live path gives it.
    let few = linearize(NEAR_LINEAR, &rows(0.0, 1.0, 2));
    let kept: Vec<LinearizationEvidence> = few.iter().map(evidence).collect();
    assert_eq!(recorded(NEAR_LINEAR, &kept).unwrap(), few);
}

/// Like `simplify_v2` (B20 wiring review 01, B20-R01-AST-FIRST), the public linearization entries
/// check a decoded tree's structure before they copy or normalize it: a `MathExpr` decoded from
/// JSON has had no structure check, and the copy and the admission's normalization both recurse.
#[test]
fn the_public_entries_check_a_decoded_tree_before_copying_it() {
    // #202 review 04's shallow 65-node tree: one call with 64 literal arguments.
    let leaves = vec![serde_json::json!({ "Const": 1.0 }); 64];
    let wide: MathExpr =
        serde_json::from_value(serde_json::json!({ "root": { "Call": ["Abs", leaves] } })).unwrap();
    let refused = |error: anyhow::Error| {
        assert!(
            format!("{error:#}").contains("before it is copied"),
            "{error:#}"
        );
    };
    refused(wide.linearizable_v2(&baseline(), scope()).unwrap_err());
    refused(
        wide.linearize_v2(&baseline(), scope(), &rows(0.0, 1.0, 8))
            .unwrap_err(),
    );
    refused(
        wide.linearize_recorded_v2(&baseline(), scope(), &mut |_| {
            unreachable!("a refused tree records nothing")
        })
        .unwrap_err(),
    );
    // An admitted tree passes the check.
    assert!(MathExpr::parse(NEAR_LINEAR)
        .unwrap()
        .linearizable_v2(&baseline(), scope())
        .is_ok());
}

/// A line's measured error is its largest absolute residual over the samples, so a recorded one
/// is finite and not negative (B19 wiring review 01, B19-R01-NEGATIVE-LOWERED-ERROR): a negative
/// error would pass any bound, and a NaN one would compare under none.
#[test]
fn a_recorded_line_error_must_be_a_finite_nonnegative_residual() {
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    assert!(live[0].expression.is_some());
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(NEAR_LINEAR, &kept).unwrap(), live);
    for error in [
        -1.0,
        -f64::MIN_POSITIVE,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let mut bad = kept.clone();
        bad[0].lowered_error = Some(error);
        let refusal = recorded(NEAR_LINEAR, &bad).unwrap_err();
        assert!(
            format!("{refusal:#}").contains("not a finite nonnegative residual"),
            "{error}: {refusal:#}"
        );
    }
    // The smallest error the record's own line allows is offered (review 02 adds that floor).
    let mut floor = kept.clone();
    floor[0].lowered_error = Some(least_error(&live[0], kept[0].fit.as_ref().unwrap()));
    assert!(recorded(NEAR_LINEAR, &floor).unwrap()[0]
        .expression
        .is_some());
}

/// The larger size of a recorded line at its sampled ends, `M`: between the samples the computed
/// line lies between its values there.
fn largest_end(fit: &LinearFit) -> f64 {
    [fit.lo, fit.hi]
        .map(|end| horner(&fit.coefficients, end).abs())
        .into_iter()
        .fold(0.0, f64::max)
}

/// The least error a recorded line can carry on its own record, for the subtree of `l`: at either
/// sampled end, its residual against that end's target and its size past `max|f|`; at the largest
/// target, `max|f| - M`; and never under zero.
fn least_error(l: &SubtreeLinearization, fit: &LinearFit) -> f64 {
    let mut found = Vec::new();
    variables(&l.subtree.root, &mut found);
    [fit.lo, fit.hi]
        .map(|end| {
            let value = horner(&fit.coefficients, end);
            let target = target_at(&l.subtree.root, found[0], end).unwrap();
            (value - target).abs().max(value.abs() - fit.max_abs)
        })
        .into_iter()
        .fold(fit.max_abs - largest_end(fit), f64::max)
        .max(0.0)
}

/// The live measurement keeps a line only while it is in the evaluator's range at every sample,
/// and `lo` and `hi` are samples whose targets are within `max|f|` (B19 wiring review 02,
/// B19-R02-IMPOSSIBLE-ENDPOINT-RESIDUAL). So a recorded line error is at most
/// `MAX_ABS + max|f|`, the line is in range at both ends, and the error is at least the line's
/// residual there. The computed line is monotone between its ends, and some sample attains
/// `max|f|` (review 03, B19-R03-LINE-ENVELOPE-RESIDUAL), so the error is also at most
/// `M + max|f|` and at least `max|f| - M`, with `M` the line's larger size at the ends. Each bound
/// is exact, and a record at it is accepted.
#[test]
fn a_recorded_line_error_must_be_one_its_line_and_samples_allow() {
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(NEAR_LINEAR, &kept).unwrap(), live);
    let fit = kept[0].fit.unwrap();
    let error = kept[0].lowered_error.unwrap();
    let refused = |bad: &[LinearizationEvidence], expected: &str| {
        let refusal = recorded(NEAR_LINEAR, bad).unwrap_err();
        assert!(format!("{refusal:#}").contains(expected), "{refusal:#}");
    };
    // Past what the evaluator's range allows.
    for over in [(MAX_ABS + fit.max_abs).next_up(), 1e100] {
        let mut bad = kept.clone();
        bad[0].lowered_error = Some(over);
        refused(&bad, "exceeds what the evaluator's range allows");
    }
    // Past what the line and its samples allow, `M + max|f|`, though within the evaluator's
    // range; at it, the error only misses the bound.
    let ceiling = largest_end(&fit) + fit.max_abs;
    for over in [ceiling.next_up(), MAX_ABS + fit.max_abs] {
        let mut bad = kept.clone();
        bad[0].lowered_error = Some(over);
        refused(&bad, "exceeds what its line and samples allow");
    }
    let mut at = kept.clone();
    at[0].lowered_error = Some(ceiling);
    let out = recorded(NEAR_LINEAR, &at).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    // A largest target far past the line, with the line's small error kept: refused. At
    // `max|f| - M` the record is consistent and misses the bound; one step under it is refused.
    // The subtree's targets stay within 3, its bound (review 06), so the target is 3.
    let mut wide = kept.clone();
    wide[0].fit.as_mut().unwrap().max_abs = 3.0;
    refused(&wide, "smaller than the residual at the largest target");
    let least = 3.0 - largest_end(&fit);
    assert!(least > error, "{least} <= {error}");
    wide[0].lowered_error = Some(least);
    let out = recorded(NEAR_LINEAR, &wide).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    wide[0].lowered_error = Some(least.next_down());
    refused(&wide, "smaller than the residual at the largest target");
    // The line moved far from every target keeps its small recorded error: refused. At the
    // moved line's own floor the record is consistent and misses the bound; one step under it
    // is refused.
    let mut moved = kept.clone();
    let line = moved[0].fit.as_mut().unwrap();
    line.coefficients[0] += 10.0 * (fit.max_abs + error) + 1.0;
    let floor = least_error(&live[0], line);
    assert!(floor > error, "{floor} <= {error}");
    refused(&moved, "smaller than the line's residual at a sampled end");
    moved[0].lowered_error = Some(floor);
    let out = recorded(NEAR_LINEAR, &moved).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    moved[0].lowered_error = Some(floor.next_down());
    refused(&moved, "smaller than the line's residual at a sampled end");
    // A line outside the evaluator's range at a sampled end cannot have been measured (here at
    // both ends, `lo` = -0.5 and `hi` = 1.5).
    let mut steep = kept.clone();
    steep[0].fit.as_mut().unwrap().coefficients = [0.0, 4.0 * MAX_ABS];
    refused(&steep, "outside the evaluator's range at a sampled end");
    // A line that cannot be lowered is refused before it is measured: its recorded error is not
    // read at the ends, as for every refusal before the measurement.
    let mut flat = kept.clone();
    flat[0].fit.as_mut().unwrap().coefficients = [f64::NAN; 2];
    let out = recorded(NEAR_LINEAR, &flat).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_NON_FINITE));
}

/// `lo` and `hi` are sampled values of the subtree's one variable, so the subtree itself gives its
/// targets there (B19 wiring review 04, B19-R04-ENDPOINT-TARGET-RESIDUAL): they are fault-free and
/// within `max|f|`, and a recorded error covers the line's residual against each. The targets are
/// what the sampling records, bit for bit, for column and aggregate variables alike.
#[test]
fn a_recorded_fit_must_agree_with_its_subtrees_targets_at_the_sampled_ends() {
    let training = rows(-0.5, 1.5, 201);
    // The element read gives each sampled target bit for bit, for a column and an aggregate.
    for source in [NEAR_LINEAR, "(sin((0.5*row_mean))*3.0)"] {
        let subtree = MathExpr::parse(source).unwrap();
        let mut found = Vec::new();
        variables(&subtree.root, &mut found);
        let leaf = MathExpr::from_ast(Expr::Var(found[0])).unwrap();
        for row in &training {
            let (v, _) = leaf.eval_row_traced(row);
            let (f, faults) = subtree.eval_row_traced(row);
            assert!(faults.is_empty());
            let target = target_at(&subtree.root, found[0], v).unwrap();
            assert_eq!(target.to_bits(), f.to_bits(), "{source} at {v}");
        }
    }
    // A value where the subtree faults gives no target.
    let quotient = MathExpr::parse("(1.0/c0)").unwrap();
    assert_eq!(target_at(&quotient.root, Var::Col(0), 2.0), Some(0.5));
    assert_eq!(target_at(&quotient.root, Var::Col(0), 0.0), None);

    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    let fit = kept[0].fit.unwrap();
    let error = kept[0].lowered_error.unwrap();
    assert_eq!(fit.stop, MinimaxStop::Converged);
    let refused = |bad: &[LinearizationEvidence], expected: &str| {
        let refusal = recorded(NEAR_LINEAR, bad).unwrap_err();
        assert!(format!("{refusal:#}").contains(expected), "{refusal:#}");
    };
    // `max|f|` under a sampled end's target.
    let mut short = kept.clone();
    let end_target = target_at(&live[0].subtree.root, Var::Col(0), fit.hi).unwrap();
    short[0].fit.as_mut().unwrap().max_abs = end_target.abs() / 2.0;
    refused(&short, "sampled ends give targets its samples cannot have");
    // The line moved far off with `max|f|` set to its larger end: every envelope bound holds for
    // its small error, but the subtree's target at a sampled end is far from the line. At that
    // residual the record is consistent and misses the bound; one step under it is refused. The
    // move keeps the larger end within the subtree's bound of 3 (review 06).
    let mut moved = kept.clone();
    let line = moved[0].fit.as_mut().unwrap();
    line.coefficients[0] += 2.0;
    line.max_abs = largest_end(line);
    assert!(line.max_abs <= 3.0, "{line:?}");
    let least = least_error(&live[0], line);
    assert!(least > error, "{least} <= {error}");
    let target_residual = "smaller than the line's residual at a sampled end's target";
    refused(&moved, target_residual);
    moved[0].lowered_error = Some(least);
    let out = recorded(NEAR_LINEAR, &moved).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    moved[0].lowered_error = Some(least.next_down());
    refused(&moved, target_residual);
}

/// A subtree refused before any fit names `unsampled`, a reason the fit itself gives (B19 wiring
/// review 04's inventory), or the caller's input faults for every subtree of the expression
/// (review 05: the pipeline's stage-prefix refusal); each of those is accepted, and any other
/// refused, the range and interval refusals the sampler cannot give (reviews 05 and 06) and
/// input faults beside another record included.
#[test]
fn a_refusal_before_the_fit_names_a_reason_the_fit_gives() {
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    let unfitted = |refusal: &str| -> Vec<LinearizationEvidence> {
        live.iter()
            .map(|_| LinearizationEvidence {
                fit: None,
                lowered_error: None,
                refusal: Some(refusal.into()),
            })
            .collect()
    };
    let mut accepted: Vec<String> = FIT_REFUSALS
        .iter()
        .map(|reason| format!("{LINEARIZE_FIT_REFUSED}: {reason}"))
        .collect();
    accepted.push(LINEARIZE_UNSAMPLED.into());
    accepted.push(LINEARIZE_INPUT_FAULTS.into());
    for refusal in &accepted {
        let out = recorded(NEAR_LINEAR, &unfitted(refusal)).unwrap();
        assert!(out
            .iter()
            .all(|l| l.refusal.as_deref() == Some(refusal.as_str())));
    }
    for refusal in [
        format!("{LINEARIZE_FIT_REFUSED}: degree above the cap"),
        format!("{LINEARIZE_FIT_REFUSED}: a sample is not finite"),
        format!("{LINEARIZE_FIT_REFUSED}: {OUTSIDE_RANGE}"),
        format!("{LINEARIZE_FIT_REFUSED}: {EMPTY_INTERVAL}"),
        format!("{LINEARIZE_FIT_REFUSED}: no reason"),
        "the line was not wanted".into(),
    ] {
        let error = recorded(NEAR_LINEAR, &unfitted(&refusal)).unwrap_err();
        assert!(
            format!("{error:#}").contains("names a refusal the fit never gives"),
            "{refusal}: {error:#}"
        );
    }
    // The caller's input faults refuse every subtree of the expression or none.
    assert_eq!(live.len(), 2);
    let faulted = unfitted(LINEARIZE_INPUT_FAULTS);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    let unsampled = unfitted(LINEARIZE_UNSAMPLED);
    for mixed in [
        vec![faulted[0].clone(), unsampled[1].clone()],
        vec![unsampled[0].clone(), faulted[1].clone()],
        vec![faulted[0].clone(), kept[1].clone()],
        vec![kept[0].clone(), faulted[1].clone()],
    ] {
        let error = recorded(NEAR_LINEAR, &mixed).unwrap_err();
        assert!(
            format!("{error:#}").contains("a faulting input refuses only some"),
            "{error:#}"
        );
    }
}

/// A subtree that is zero wherever it is finite, like `(c0-c0)*(c0-c0)`, has a zero target on
/// every sample, whatever the rows: a recorded fit of it with a nonzero `max|f|` is refused, so it
/// cannot carry the brackets, stops, bounds or offered lines a nonzero target would allow (B19
/// wiring review 05, B19-R05-KNOWN-ZERO-TARGET-ENVELOPE). The structural test is sufficient, and
/// every expression it marks is zero or not finite at each probe value; the live fit and its
/// replay are unchanged.
#[test]
fn a_subtree_zero_wherever_finite_records_only_zero_targets() {
    let marked = [
        "(c0-c0)*(c0-c0)",
        "c0-c0",
        "c0*0.0",
        "(c0-c0)/c0",
        "sin((c0-c0))*3.0",
        "((c0+1.0)-(c0+1.0))*c0",
        "-(c0-c0)",
        "(c0-c0)^2.0",
        "abs((c0-c0))+sqrt((c0-c0))",
        "erf(log1p((c0-c0)))-sign(tanh(atan((c0-c0))))",
        "floor((c0-c0))*round((c0-c0))",
    ];
    let unmarked = [
        "c0",
        "(c0-c0)+1.0",
        "cos((c0-c0))",
        "(c0-c0)^0.0",
        "c0/(c0-c0)",
        "(c0+1.0)-(c0+2.0)",
        "exp((c0-c0))*c0",
    ];
    let probes = [
        -1e12,
        -3.5,
        -1.0,
        -f64::MIN_POSITIVE,
        -0.0,
        0.0,
        5e-324,
        1e-300,
        0.5,
        2.0,
        1e6,
        1e12,
    ];
    for source in marked {
        let expression = MathExpr::parse(source).unwrap();
        assert!(zero_wherever_finite(&expression.root), "{source}");
        for v in probes {
            let (f, faults) = expression.eval_row_traced(&[v, 0.25]);
            assert!(
                !(faults.is_empty() && f.is_finite()) || f == 0.0,
                "{source} at {v}: {f}"
            );
        }
    }
    for source in unmarked {
        assert!(
            !zero_wherever_finite(&MathExpr::parse(source).unwrap().root),
            "{source}"
        );
    }
    // The literal's bits decide sameness.
    let signed = Expr::Binary(
        BinOp::Sub,
        Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Var(Var::Col(0))),
            Box::new(Expr::Const(0.0)),
        )),
        Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Var(Var::Col(0))),
            Box::new(Expr::Const(-0.0)),
        )),
    );
    assert!(!zero_wherever_finite(&signed));
    // The live fit has zero targets and replays.
    const ZERO: &str = "(c0-c0)*(c0-c0)";
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(ZERO, &training);
    assert_eq!(live.len(), 1);
    let fit = live[0].fit.unwrap();
    assert_eq!((fit.max_abs, fit.stop), (0.0, MinimaxStop::ExactFit));
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(ZERO, &kept).unwrap(), live);
    // The zero line's pair on this finite range is finite (`+0.0`, `+0.0`), so an absent pair is
    // refused here, as for any subtree whose targets are all zero (Lane B's review of #247).
    assert_eq!(fit.coefficients.map(f64::to_bits), [0.0f64.to_bits(); 2]);
    let mut absent = kept.clone();
    absent[0].fit.as_mut().unwrap().coefficients = [f64::NAN; 2];
    let error = recorded(ZERO, &absent).unwrap_err();
    assert!(
        format!("{error:#}").contains("a recorded fit disagrees with its stop"),
        "{error:#}"
    );
    // The review's witness: a constant line `1` over targets of size 1, consistent with its stop,
    // whose error meets every envelope bound and the minimax bound, so it was offered.
    let nonzero = LinearizationEvidence {
        fit: Some(LinearFit {
            stop: MinimaxStop::Converged,
            coefficients: [1.0, 0.0],
            levelled: 1.0,
            max_error: 1.0,
            max_abs: 1.0,
            iterations: 1,
            steps: 1,
            ..fit
        }),
        lowered_error: Some(1.0),
        refusal: None,
    };
    let error = recorded(ZERO, std::slice::from_ref(&nonzero)).unwrap_err();
    assert!(
        format!("{error:#}").contains("zero on every sample a nonzero target"),
        "{error:#}"
    );
    // The same metadata on a subtree that is not zero everywhere is not refused for its zeros.
    let near = recorded(
        NEAR_LINEAR,
        &[
            nonzero.clone(),
            evidence(&linearize(NEAR_LINEAR, &training)[1]),
        ],
    )
    .map_err(|e| format!("{e:#}"));
    assert!(
        near.as_ref()
            .err()
            .is_none_or(|e| !e.contains("zero on every sample")),
        "{near:?}"
    );
    // A nonzero `max|f|` under any stop, with the zero line, is refused too.
    let mut zero_line = nonzero;
    let line = zero_line.fit.as_mut().unwrap();
    line.coefficients = [0.0, 0.0];
    line.max_error = 0.25;
    line.levelled = 0.25;
    zero_line.lowered_error = None;
    let error = recorded(ZERO, &[zero_line]).unwrap_err();
    assert!(
        format!("{error:#}").contains("zero on every sample a nonzero target"),
        "{error:#}"
    );
}

/// The live samples read each row's variable, and the element `x` faults on every row, so a
/// subtree of `x` has no fit: a fit recorded for one is refused (B19 wiring review 05's note
/// B19-R05-X-FIT-APPLICABILITY), while its live refusals replay; and it is refused before any fit
/// only as unsampled, for too few values (no rows at all), or for the caller's input faults, any
/// other reason refused (review 06's note).
#[test]
fn a_subtree_of_the_element_x_records_no_fit() {
    let training = rows(-0.5, 1.5, 201);
    let element = ExpressionScope::search_profile(2, true);
    let x = MathExpr::parse("sin((0.1*x))*3.0").unwrap();
    let live = x.linearize_v2(&baseline(), element, &training).unwrap();
    assert!(live
        .iter()
        .all(|l| l.fit.is_none() && l.refusal.as_deref() == Some(LINEARIZE_UNSAMPLED)));
    let replay = x
        .linearize_recorded_v2(&baseline(), element, &mut |i| Ok(evidence(&live[i])))
        .unwrap();
    assert_eq!(replay, live);
    // The fits of the same expression over `c0`.
    let copied: Vec<LinearizationEvidence> = linearize(NEAR_LINEAR, &training)
        .iter()
        .map(evidence)
        .collect();
    assert!(copied[0].fit.is_some());
    let error = x
        .linearize_recorded_v2(&baseline(), element, &mut |i| Ok(copied[i].clone()))
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("subtree of the element x"),
        "{error:#}"
    );
    // Its refusals before any fit.
    let refused_as = |refusal: String| {
        x.linearize_recorded_v2(&baseline(), element, &mut |_| {
            Ok(LinearizationEvidence {
                fit: None,
                lowered_error: None,
                refusal: Some(refusal.clone()),
            })
        })
    };
    assert!(x
        .linearize_v2(&baseline(), element, &[])
        .unwrap()
        .iter()
        .all(|l| {
            l.refusal.as_deref()
                == Some(format!("{LINEARIZE_FIT_REFUSED}: {FEW_ABSCISSAE}").as_str())
        }));
    for refusal in [
        LINEARIZE_UNSAMPLED.to_string(),
        LINEARIZE_INPUT_FAULTS.to_string(),
        format!("{LINEARIZE_FIT_REFUSED}: {FEW_ABSCISSAE}"),
    ] {
        refused_as(refusal).unwrap();
    }
    for reason in [SINGULAR_FIRST, NONFINITE_ERROR, TWO_TARGETS] {
        let error = refused_as(format!("{LINEARIZE_FIT_REFUSED}: {reason}")).unwrap_err();
        assert!(
            format!("{error:#}").contains("element x names a refusal its samples never give"),
            "{reason}: {error:#}"
        );
    }
}

/// A subtree with one value on every sample, such as `(c0-c0)+1.0` or `step((c0-c0),1.0)`, has
/// targets of that value's size on any rows, and the first reference's solve gives that value at
/// once: a recorded fit of it must carry that size as `max|f|`, the first step's exact fit, that
/// value and a zero slope, and a line error equal to its line's larger end residual (B19 wiring
/// review 06, B19-R06-KNOWN-TARGET-EVIDENCE). The structural test is sufficient, and every finite
/// value of an expression it marks has one size at the probe values; the live fit replays.
#[test]
fn a_subtree_with_one_value_records_that_values_exact_fit() {
    let marked = [
        "(c0-c0)+1.0",
        "step((c0-c0),1.0)",
        "cos((c0-c0))*2.0",
        "sigmoid((c0*0.0))+0.25",
        "((c0-c0)*c0)+1.5",
        "min((c0-c0),2.0)-1.0",
        "exp(sin((c0-c0)))*(c0-c0)",
    ];
    let unmarked = [
        "c0+1.0",
        "step(c0,1.0)",
        "cos(c0)",
        "(c0-c0)+c0",
        "exp(c0)*0.5",
        "(c0/(c0-c0))+2.0",
    ];
    let probes = [-1e12, -3.5, -1.0, -0.0, 0.0, 5e-324, 0.5, 2.0, 1e6, 1e12];
    for source in marked {
        let expression = MathExpr::parse(source).unwrap();
        assert!(one_value_wherever_finite(&expression.root), "{source}");
        let sizes: BTreeSet<u64> = probes
            .iter()
            .map(|&v| expression.eval_row_traced(&[v, 0.25]))
            .filter(|(f, faults)| faults.is_empty() && f.is_finite())
            .map(|(f, _)| f.abs().to_bits())
            .collect();
        assert!(sizes.len() <= 1, "{source}: {sizes:?}");
    }
    for source in unmarked {
        let expression = MathExpr::parse(source).unwrap();
        assert!(!one_value_wherever_finite(&expression.root), "{source}");
    }
    // The live fit of a constant: exact at the first step, that value and a zero slope, no error.
    const ONE: &str = "(c0-c0)+1.0";
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(ONE, &training);
    assert_eq!(live.len(), 1);
    let fit = live[0].fit.unwrap();
    assert_eq!(
        (
            fit.stop,
            fit.iterations,
            fit.steps,
            fit.max_abs,
            fit.max_error
        ),
        (MinimaxStop::ExactFit, 1, 1, 1.0, 0.0)
    );
    assert_eq!(fit.coefficients, [1.0, 0.0]);
    assert!(
        live[0].lowered_error.is_none_or(|e| e == 0.0),
        "{:?}",
        live[0]
    );
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(ONE, &kept).unwrap(), live);
    let refused = |evidence: LinearizationEvidence, source: &str, expected: &str| {
        let error = recorded(source, std::slice::from_ref(&evidence)).unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
    };
    let not_exact = "is not that value's exact fit";
    // The review's witness: the zero line for the constant one, offered before.
    let zero_line = LinearizationEvidence {
        fit: Some(LinearFit {
            stop: MinimaxStop::Converged,
            coefficients: [0.0, 0.0],
            levelled: 1.0,
            max_error: 1.0,
            iterations: 1,
            steps: 1,
            ..fit
        }),
        lowered_error: Some(1.0),
        refusal: None,
    };
    refused(zero_line, ONE, not_exact);
    // A converged bracket at the right size, its line kept.
    let mut bracket = kept[0].clone();
    let edit = bracket.fit.as_mut().unwrap();
    (edit.stop, edit.levelled, edit.max_error) = (MinimaxStop::Converged, 0.25, 0.25);
    bracket.lowered_error = None;
    refused(bracket, ONE, not_exact);
    // A larger size, within the bound's slack of one ulp, is refused for that size.
    let mut size = kept[0].clone();
    size.fit.as_mut().unwrap().max_abs = 1.0f64.next_up();
    refused(size, ONE, not_exact);
    // The line's error edited within every envelope bound: not its end residual, zero.
    let mut error = kept[0].clone();
    error.lowered_error = Some(0.5);
    refused(error, ONE, "not the larger end residual");
    // On this finite range the conversion's pair is finite, so an absent pair (NaN, as a published
    // record reads one back) is refused (Lane B's review of #247), as is any other finite pair.
    assert_eq!(constant_line(1.0, fit.lo, fit.hi), [1.0, 0.0]);
    for pair in [
        [f64::NAN; 2],
        [1.0, f64::NAN],
        [f64::NAN, 0.0],
        [1.0f64.next_up(), 0.0],
    ] {
        let mut absent = kept[0].clone();
        absent.fit.as_mut().unwrap().coefficients = pair;
        absent.lowered_error = None;
        refused(absent, ONE, not_exact);
    }
    // On a subnormal range the conversion scales the zero slope by an infinite `α`: the live pair
    // holds a NaN, and both its own pair and its published read-back (both NaN) replay.
    let tiny = [vec![-1e-310, 0.25], vec![0.0, 0.25], vec![1e-310, 0.25]];
    let live_tiny = linearize(ONE, &tiny);
    let tiny_fit = live_tiny[0].fit.unwrap();
    assert_eq!(tiny_fit.coefficients[0], 1.0);
    assert!(tiny_fit.coefficients[1].is_nan(), "{tiny_fit:?}");
    recorded(ONE, &[evidence(&live_tiny[0])]).unwrap();
    let mut read_back = evidence(&live_tiny[0]);
    read_back.fit.as_mut().unwrap().coefficients = [f64::NAN; 2];
    recorded(ONE, &[read_back]).unwrap();
    // `step((c0-c0),1.0)` is zero on every sample, though not by the zero test.
    const STEP: &str = "step((c0-c0),1.0)";
    let step = MathExpr::parse(STEP).unwrap();
    assert!(!zero_wherever_finite(&step.root));
    let live = linearize(STEP, &training);
    let fit = live[0].fit.unwrap();
    assert_eq!((fit.max_abs, fit.stop), (0.0, MinimaxStop::ExactFit));
    assert_eq!(
        recorded(STEP, &live.iter().map(evidence).collect::<Vec<_>>()).unwrap(),
        live
    );
    let nonzero = LinearizationEvidence {
        fit: Some(LinearFit {
            stop: MinimaxStop::Converged,
            coefficients: [f64::NAN; 2],
            levelled: 0.25,
            max_error: 0.25,
            max_abs: 1.0,
            iterations: 1,
            steps: 1,
            ..fit
        }),
        lowered_error: None,
        refusal: None,
    };
    refused(nonzero, STEP, not_exact);
}

/// No target is larger than every value its subtree can take: `step((c0+c0),0.0)` takes only 0
/// and 1, so a recorded `max|f|` of 2 is refused (B19 wiring review 06), while the live fit
/// replays. The bound is sound at every probe value.
#[test]
fn a_recorded_fit_stays_within_what_its_subtree_can_take() {
    // Each with the bound the analysis gives, up to its rounding up.
    let bounded = [
        ("step((c0+c0),0.0)", 1.0),
        ("sin(c0)*3.0", 3.0001),
        ("tanh(c0)+2.0", 3.0001),
        ("abs(cos(c0))", 1.0),
        ("(atan(c0)*2.0)-sign(c0)", 4.2),
        ("sqrt(abs(sin(c0)))+floor(erf(c0))", 3.0001),
        ("max(sigmoid(c0),clip(c0,-2.0,0.5))", 2.0),
    ];
    let unbounded = ["c0*2.0", "c0/sin(c0)", "log1p(abs(c0))", "c0^2.0"];
    let probes = [-1e12, -3.5, -1.0, -0.0, 0.0, 5e-324, 0.5, 2.0, 1e6, 1e12];
    for (source, at_most) in bounded {
        let expression = MathExpr::parse(source).unwrap();
        let bound = largest_magnitude(&expression.root);
        assert!(bound <= at_most, "{source}: {bound}");
        for v in probes {
            let (f, faults) = expression.eval_row_traced(&[v, 0.25]);
            assert!(
                !(faults.is_empty() && f.is_finite()) || f.abs() <= bound,
                "{source} at {v}: {f}"
            );
        }
    }
    for source in unbounded {
        assert_eq!(
            largest_magnitude(&MathExpr::parse(source).unwrap().root),
            f64::INFINITY,
            "{source}"
        );
    }
    const STEP: &str = "step((c0+c0),0.0)";
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(STEP, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(live[0].fit.unwrap().max_abs, 1.0);
    assert_eq!(recorded(STEP, &kept).unwrap(), live);
    let mut larger = kept.clone();
    larger[0].fit.as_mut().unwrap().max_abs = 2.0;
    let error = recorded(STEP, &larger).unwrap_err();
    assert!(
        format!("{error:#}").contains("max|f| exceeds every value its subtree can take"),
        "{error:#}"
    );
}

/// A row's L2 norm and standard deviation are never negative, so a recorded fit of either over a
/// range with a negative end is refused (B19 wiring review 06, B19-R06-AGGREGATE-INPUT-RANGE),
/// while their live fits replay.
#[test]
fn a_recorded_range_holds_only_values_its_variable_takes() {
    let training = rows(-0.5, 1.5, 201);
    let copied: Vec<LinearizationEvidence> = linearize(NEAR_LINEAR, &training)
        .iter()
        .map(evidence)
        .collect();
    assert!(copied[0].fit.unwrap().lo < 0.0);
    for var in ["row_l2", "row_std"] {
        let source = format!("sin((0.1*{var}))*3.0");
        let live = linearize(&source, &training);
        assert!(live[0].fit.unwrap().lo >= 0.0, "{var}");
        let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
        assert_eq!(recorded(&source, &kept).unwrap(), live, "{var}");
        let error = recorded(&source, &copied).unwrap_err();
        assert!(
            format!("{error:#}").contains("range holds values its variable never takes"),
            "{var}: {error:#}"
        );
    }
}

/// A closing stop returns the fit of the step it stops at, the iteration cap stops after the last
/// step, and the fit's residuals and `max|f|` are maxima of absolute values from `+0.0`, never
/// `-0.0` (B19 wiring review 02's two notes); an exact fit is never recorded under another stop,
/// and a line's fit spans three distinct abscissae (review 03's two notes); a finite zero
/// coefficient is `+0.0`, and a fit of all-zero targets is the zero polynomial at the first step
/// (review 04's notes).
#[test]
fn a_recorded_fit_must_carry_its_stops_step_and_positive_zeros() {
    let disagrees = "a recorded fit disagrees with its stop";
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    let fit = kept[0].fit.unwrap();
    assert!(fit.stop.closed() && fit.iterations == fit.steps && fit.steps < MAX_ITERATIONS);
    let edits: [fn(&mut LinearFit); 3] = [
        |f| f.steps += 1,
        |f| {
            f.stop = MinimaxStop::IterationLimit;
            f.steps = MAX_ITERATIONS - 1;
        },
        |f| f.coefficients[0] = -0.0,
    ];
    for edit in edits {
        let mut bad = kept.clone();
        edit(bad[0].fit.as_mut().unwrap());
        let error = recorded(NEAR_LINEAR, &bad).unwrap_err();
        assert!(format!("{error:#}").contains(disagrees), "{error:#}");
    }
    // `minimax` returns `ExactFit` at the first fit within its test (review 03,
    // B19-R03-EXACT-STOP-PRIORITY): an exact fit of nonzero targets, recorded under any other
    // stop, is refused.
    let linear = "(c0*2.0)+(c0*0.5)";
    let live = linearize(linear, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(linear, &kept).unwrap(), live);
    let exact = kept[0].fit.unwrap();
    assert_eq!(exact.stop, MinimaxStop::ExactFit);
    assert!(exact.max_abs > 0.0);
    for stop in [
        MinimaxStop::Converged,
        MinimaxStop::NoiseLimited,
        MinimaxStop::Stalled,
        MinimaxStop::Degenerate,
        MinimaxStop::IterationLimit,
    ] {
        let mut bad = kept.clone();
        let fit = bad[0].fit.as_mut().unwrap();
        fit.stop = stop;
        if stop == MinimaxStop::IterationLimit {
            fit.steps = MAX_ITERATIONS;
        }
        let error = recorded(linear, &bad).unwrap_err();
        assert!(
            format!("{error:#}").contains(disagrees),
            "{stop:?}: {error:#}"
        );
    }
    // A subtree that is zero on every row: its fit's bracket and max|f| are all `+0.0`.
    let zero = "(c0-c0)*(c0-c0)";
    let live = linearize(zero, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    assert_eq!(recorded(zero, &kept).unwrap(), live);
    let at = kept
        .iter()
        .position(|e| e.fit.is_some_and(|f| f.max_abs == 0.0))
        .expect("a fit of the zero subtree");
    let edits: [fn(&mut LinearFit); 3] = [
        |f| f.levelled = -0.0,
        |f| f.max_error = -0.0,
        |f| f.max_abs = -0.0,
    ];
    for edit in edits {
        let mut bad = kept.clone();
        edit(bad[at].fit.as_mut().unwrap());
        assert_ne!(
            bad[at]
                .fit
                .map(|f| [f.levelled, f.max_error, f.max_abs].map(f64::to_bits)),
            kept[at]
                .fit
                .map(|f| [f.levelled, f.max_error, f.max_abs].map(f64::to_bits))
        );
        let error = recorded(zero, &bad).unwrap_err();
        assert!(format!("{error:#}").contains(disagrees), "{error:#}");
    }
    // `minimax` returns `ExactFit` at the first fit within its test (review 03,
    // B19-R03-EXACT-STOP-PRIORITY): the zero subtree's exact fit under any other stop is refused.
    for stop in [
        MinimaxStop::Converged,
        MinimaxStop::NoiseLimited,
        MinimaxStop::Stalled,
        MinimaxStop::Degenerate,
        MinimaxStop::IterationLimit,
    ] {
        let mut bad = kept.clone();
        let exact = bad[at].fit.as_mut().unwrap();
        assert_eq!(exact.stop, MinimaxStop::ExactFit);
        exact.stop = stop;
        if stop == MinimaxStop::IterationLimit {
            exact.steps = MAX_ITERATIONS;
        }
        let error = recorded(zero, &bad).unwrap_err();
        assert!(
            format!("{error:#}").contains(disagrees),
            "{stop:?}: {error:#}"
        );
    }
    // With every target zero the fit is the zero polynomial, exact at the first step with `+0.0`
    // coefficients (review 04's notes): other steps, a nonzero bracket, or a `-0.0` or nonzero
    // coefficient are refused. A finite zero coefficient of any fit is `+0.0` (above).
    let zero_fit = kept[at].fit.unwrap();
    assert_eq!(
        (zero_fit.iterations, zero_fit.steps, zero_fit.max_error),
        (1, 1, 0.0)
    );
    assert!(zero_fit
        .coefficients
        .iter()
        .all(|c| *c == 0.0 && c.is_sign_positive()));
    let edits: [fn(&mut LinearFit); 5] = [
        |f| {
            f.iterations = MAX_ITERATIONS;
            f.steps = MAX_ITERATIONS;
        },
        |f| {
            f.stop = MinimaxStop::Converged;
            f.levelled = 0.25;
            f.max_error = 0.25;
        },
        |f| f.coefficients[0] = -0.0,
        |f| f.coefficients[1] = -0.0,
        |f| f.coefficients[1] = 1.0,
    ];
    for edit in edits {
        let mut bad = kept.clone();
        edit(bad[at].fit.as_mut().unwrap());
        let error = recorded(zero, &bad).unwrap_err();
        assert!(format!("{error:#}").contains(disagrees), "{error:#}");
    }
    // A line's fit needs three distinct abscissae (review 03, B19-R03-RANGE-CARDINALITY), so its
    // range holds a value strictly between its ends; -0 and +0 are one value.
    let tiny = f64::from_bits(1);
    let holds = |lo: f64, hi: f64| consistent(&LinearFit { lo, hi, ..fit });
    assert!(holds(-tiny, tiny));
    assert!(holds(fit.lo, fit.hi));
    for (lo, hi) in [(0.0, tiny), (-0.0, tiny), (-tiny, 0.0), (-tiny, -0.0)] {
        assert!(!holds(lo, hi), "{lo:e} {hi:e}");
    }
    let mut narrow: Vec<LinearizationEvidence> = linearize(NEAR_LINEAR, &training)
        .iter()
        .map(evidence)
        .collect();
    for e in &mut narrow {
        if let Some(f) = e.fit.as_mut() {
            (f.lo, f.hi) = (0.0, tiny);
        }
    }
    let error = recorded(NEAR_LINEAR, &narrow).unwrap_err();
    assert!(format!("{error:#}").contains(disagrees), "{error:#}");
}

/// A fit refused for a coefficient past the finite range is published without its coefficients
/// and read back as NaN, which never equals itself (B19 wiring review 01,
/// B19-R01-NAN-REFUSAL-ROUNDTRIP). The records of a repeated subtree compare bit for bit, so two
/// such read-backs agree, while any other number in one of them is still refused.
#[test]
fn repeated_subtrees_compare_their_records_bit_for_bit() {
    // Subnormal abscissae: each line's slope leaves the finite range.
    let training = rows(-1e-310, 1e-310, 3);
    let live = linearize(REPEATED, &training);
    assert_eq!(live.len(), 3);
    assert_eq!(live[1].subtree, live[2].subtree);
    for l in &live {
        assert_eq!(l.refusal.as_deref(), Some(LINEARIZE_NON_FINITE), "{l:?}");
    }
    // As the published record reads it back: an absent pair is NaN.
    let read_back: Vec<LinearizationEvidence> = live
        .iter()
        .map(|l| {
            let mut kept = evidence(l);
            let fit = kept.fit.as_mut().unwrap();
            if fit.coefficients.iter().any(|c| !c.is_finite()) {
                fit.coefficients = [f64::NAN; 2];
            }
            kept
        })
        .collect();
    let out = recorded(REPEATED, &read_back).unwrap();
    for (out, live) in out.iter().zip(&live) {
        assert_eq!(
            (&out.refusal, out.expression.is_none()),
            (&live.refusal, true)
        );
    }
    let repeated = "a repeated subtree records other samples";
    let differs = |edit: fn(&mut LinearizationEvidence)| {
        let mut bad = read_back.clone();
        edit(&mut bad[2]);
        let error = recorded(REPEATED, &bad).unwrap_err();
        assert!(format!("{error:#}").contains(repeated), "{error:#}");
    };
    differs(|e| e.fit.as_mut().unwrap().coefficients = [0.0, 1.0]);
    differs(|e| e.fit.as_mut().unwrap().coefficients[0] = -f64::NAN);
    differs(|e| e.fit.as_mut().unwrap().max_abs *= 2.0);
}
