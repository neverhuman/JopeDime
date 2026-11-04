use super::*;

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
