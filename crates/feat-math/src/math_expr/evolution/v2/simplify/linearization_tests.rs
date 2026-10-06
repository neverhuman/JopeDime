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
    // A missing measured error is the out-of-range refusal.
    let mut missing = kept.clone();
    missing[0].lowered_error = None;
    let out = recorded(NEAR_LINEAR, &missing).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_OUT_OF_RANGE));
    // Changed coefficients change the line, and so the admitted expression. The samples' largest
    // |f| is widened past the moved line, so that its recorded error stays one the line allows.
    let mut moved = kept.clone();
    moved[0].fit.as_mut().unwrap().coefficients[0] += 0.5;
    moved[0].fit.as_mut().unwrap().max_abs += 1.0;
    let out = recorded(NEAR_LINEAR, &moved).unwrap();
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
    // At the range's edge a consistent record is accepted: its bound grows with max|f|, which
    // is the limit of checking a record without its rows.
    let mut edge = kept.clone();
    edge[0].fit.as_mut().unwrap().max_abs = MAX_ABS;
    let out = recorded(NEAR_LINEAR, &edge).unwrap();
    assert!(out[0].expression.is_some());
}

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
    floor[0].lowered_error = Some(residual_floor(kept[0].fit.as_ref().unwrap()));
    assert!(recorded(NEAR_LINEAR, &floor).unwrap()[0]
        .expression
        .is_some());
}

/// The least residual a recorded line can have on its own record: at either sampled end, its
/// value's distance from `max|f|`, and never under zero.
fn residual_floor(fit: &LinearFit) -> f64 {
    [fit.lo, fit.hi]
        .map(|end| horner(&fit.coefficients, end).abs() - fit.max_abs)
        .into_iter()
        .fold(0.0, f64::max)
}

/// The live measurement keeps a line only while it is in the evaluator's range at every sample,
/// and `lo` and `hi` are samples whose targets are within `max|f|` (B19 wiring review 02,
/// B19-R02-IMPOSSIBLE-ENDPOINT-RESIDUAL). So a recorded line error is at most
/// `MAX_ABS + max|f|`, the line is in range at both ends, and the error is at least the line's
/// residual there. Each bound is exact, and a record at it is accepted.
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
    // Past what the evaluator's range allows; at it, the error only misses the bound.
    for over in [(MAX_ABS + fit.max_abs).next_up(), 1e100] {
        let mut bad = kept.clone();
        bad[0].lowered_error = Some(over);
        refused(&bad, "exceeds what the evaluator's range allows");
    }
    let mut at = kept.clone();
    at[0].lowered_error = Some(MAX_ABS + fit.max_abs);
    let out = recorded(NEAR_LINEAR, &at).unwrap();
    assert_eq!(out[0].refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    // The line moved far from every target keeps its small recorded error: refused. At the
    // moved line's own floor the record is consistent and misses the bound; one step under it
    // is refused.
    let mut moved = kept.clone();
    let line = moved[0].fit.as_mut().unwrap();
    line.coefficients[0] += 10.0 * (fit.max_abs + error) + 1.0;
    let floor = residual_floor(line);
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

/// A closing stop returns the fit of the step it stops at, the iteration cap stops after the last
/// step, and the fit's residuals and `max|f|` are maxima of absolute values from `+0.0`, never
/// `-0.0` (B19 wiring review 02's two notes).
#[test]
fn a_recorded_fit_must_carry_its_stops_step_and_positive_zeros() {
    let disagrees = "a recorded fit disagrees with its stop";
    let training = rows(-0.5, 1.5, 201);
    let live = linearize(NEAR_LINEAR, &training);
    let kept: Vec<LinearizationEvidence> = live.iter().map(evidence).collect();
    let fit = kept[0].fit.unwrap();
    assert!(fit.stop.closed() && fit.iterations == fit.steps && fit.steps < MAX_ITERATIONS);
    let edits: [fn(&mut LinearFit); 2] = [
        |f| f.steps += 1,
        |f| {
            f.stop = MinimaxStop::IterationLimit;
            f.steps = MAX_ITERATIONS - 1;
        },
    ];
    for edit in edits {
        let mut bad = kept.clone();
        edit(bad[0].fit.as_mut().unwrap());
        let error = recorded(NEAR_LINEAR, &bad).unwrap_err();
        assert!(format!("{error:#}").contains(disagrees), "{error:#}");
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
