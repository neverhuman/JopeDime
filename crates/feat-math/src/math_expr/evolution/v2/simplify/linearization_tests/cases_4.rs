use super::*;

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
