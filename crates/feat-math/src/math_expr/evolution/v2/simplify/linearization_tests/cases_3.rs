use super::*;

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
