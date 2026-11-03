use super::*;

/// `verify_run`'s refusal of `path`, which must name `expected`.
pub(super) fn refused_with(path: &Path, label: &str, expected: &str) -> String {
    let error = format!(
        "{:#}",
        publication::verify_run(path).map(|_| ()).expect_err(label)
    );
    assert!(error.contains(expected), "{label}: {error}");
    format!("{label}: {error}")
}

/// The bound a fit record's numbers give, as `LinearFit::bound` computes it.
pub(super) fn bound_of(fit: &FitRecord) -> f64 {
    fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * fit.max_abs
}

/// The larger size of a recorded line at its sampled ends, as the lowered line computes it: the
/// product by the slope, then the intercept added.
pub(super) fn largest_end(fit: &FitRecord) -> f64 {
    let [a0, a1] = fit.coefficients.unwrap();
    [fit.lo, fit.hi]
        .map(|v| (v * a1 + a0).abs())
        .into_iter()
        .fold(0.0, f64::max)
}

/// The records `edit` leaves, published under a rehashed digest.
pub(super) fn published_edit(
    output: &Path,
    receipt: &Receipt,
    engine: &EngineRecord,
    moves: &[MoveRecord],
    edit: impl FnOnce(&mut Vec<MoveRecord>),
) -> PathBuf {
    let mut edited = moves.to_vec();
    edit(&mut edited);
    assert_ne!(edited, moves, "the edit was not applied");
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    rewrite(
        output,
        receipt,
        Some(bytes.as_slice()),
        Some(digested(engine, &bytes)),
    )
}

/// The cubic `0.5 x³` over `c1`, which takes three values on this dataset: a second seed with a
/// subtree the pass considers, so round 0's pass reads two parents.
pub(super) const CUBE_C1: &str = "(c1*c1)*(c1*0.5)";

#[test]
fn verify_refuses_inflated_and_hidden_fits_and_moved_parent_groups() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(
        temp.path(),
        &[ONE_TIMES.into(), program(SINE), program(CUBE_C1)],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let originals: Vec<(PathBuf, Vec<u8>)> = ["receipt.json", "search/cut-0/search.json", PATH]
        .iter()
        .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
        .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    let measured = moves
        .iter()
        .position(|m| m.id == MINIMAX_LINEARIZE && m.outcome == MEASURED)
        .unwrap();
    let refused = moves
        .iter()
        .position(|m| {
            m.id == MINIMAX_LINEARIZE
                && m.outcome == REFUSED
                && line(m).fit.is_some()
                && line(m).refusal.is_some()
        })
        .unwrap();
    let mut errors = Vec::new();
    fn fit_of(m: &mut MoveRecord) -> &mut LinearizationRecord {
        m.linearization.as_mut().unwrap()
    }

    // A fit's largest |f| past the evaluator's range, the bound recomputed to match: the line,
    // the program and the outcome are unchanged, so only the range check refuses it.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_mut().unwrap();
        fit.max_abs = 2e12;
        record.bound = Some(bound_of(fit));
    });
    errors.push(refused_with(
        &path,
        "max|f| past the range",
        "a recorded fit disagrees with its stop",
    ));
    restore();
    // An exact-fit claim, whose closing test `max_error <= 4 eps max|f|` an inflated max|f| meets.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_mut().unwrap();
        fit.stop = "exact_fit".into();
        fit.max_abs = (fit.max_error / (4.0 * f64::EPSILON) * 1.5).max(2e12);
        record.bound = Some(bound_of(fit));
    });
    errors.push(refused_with(
        &path,
        "exact fit with an inflated max|f|",
        "a recorded fit disagrees with its stop",
    ));
    restore();
    // A measured line's error made negative (B19 wiring review 01,
    // B19-R01-NEGATIVE-LOWERED-ERROR): within any bound, yet no largest absolute residual. JSON
    // carries no NaN or infinity; negative zero is the one other sign it can carry.
    for (label, error) in [
        ("a negative line error", -1.0),
        ("a negative zero line error", -0.0),
    ] {
        let path = published_edit(output, &receipt, &engine, &moves, |m| {
            fit_of(&mut m[measured]).lowered_error = Some(error);
        });
        errors.push(refused_with(
            &path,
            label,
            "a recorded line error is not a finite nonnegative residual",
        ));
        restore();
    }
    // Errors the recorded line and samples cannot give (B19 wiring review 02,
    // B19-R02-IMPOSSIBLE-ENDPOINT-RESIDUAL): one past what the evaluator's range allows, and the
    // line moved far from every target with its small error kept.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        fit_of(&mut m[measured]).lowered_error = Some(1e100);
    });
    errors.push(refused_with(
        &path,
        "a line error past the evaluator's range",
        "a recorded line error exceeds what the evaluator's range allows",
    ));
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let shift = 10.0 * (record.fit.as_ref().unwrap().max_abs + record.lowered_error.unwrap());
        record.fit.as_mut().unwrap().coefficients.as_mut().unwrap()[0] += shift + 1.0;
    });
    errors.push(refused_with(
        &path,
        "a line moved off its samples",
        "a recorded line error is smaller than the line's residual at a sampled end",
    ));
    restore();
    // Errors the line's own size rules out (B19 wiring review 03,
    // B19-R03-LINE-ENVELOPE-RESIDUAL): one past `M + max|f|`, though within the evaluator's
    // range, and the line's small error kept under a largest target far past the line.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_ref().unwrap();
        let ceiling = largest_end(fit) + fit.max_abs;
        record.lowered_error = Some(2.0 * ceiling + 1.0);
    });
    errors.push(refused_with(
        &path,
        "a line error past its line's size",
        "a recorded line error exceeds what its line and samples allow",
    ));
    restore();
    // The measured line is the sine seed's, `3 sin(0.5 c1)`, whose targets stay within 3, its
    // bound (review 06): the far largest target is 3.
    assert!(line(&moves[measured]).subtree.contains("sin("));
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let error = record.lowered_error.unwrap();
        let fit = record.fit.as_mut().unwrap();
        assert!(3.0 - largest_end(fit) > 4.0 * error, "{fit:?}");
        fit.max_abs = 3.0;
    });
    errors.push(refused_with(
        &path,
        "a largest target far past the line",
        "a recorded line error is smaller than the residual at the largest target",
    ));
    restore();
    // Records the subtree's own targets at the sampled ends rule out (B19 wiring review 04): the
    // line moved far off with `max|f|` set to its larger end, so only the residual against a
    // sampled end's target binds; `max|f|` under those targets; and an in-range line's error
    // erased, which the measurement could not have refused.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let error = record.lowered_error.unwrap();
        let fit = record.fit.as_mut().unwrap();
        // Up by the room its larger end leaves under the sine's bound of 3.
        let shift = 3.0 - largest_end(fit);
        assert!(shift > 4.0 * error, "{shift} {error}");
        fit.coefficients.as_mut().unwrap()[0] += shift;
        fit.max_abs = largest_end(fit);
        assert!(fit.max_abs <= 3.0f64.next_up(), "{fit:?}");
    });
    errors.push(refused_with(
        &path,
        "a line moved off its ends' targets",
        "a recorded line error is smaller than the line's residual at a sampled end's target",
    ));
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        fit_of(&mut m[measured]).fit.as_mut().unwrap().max_abs = 1e-300;
    });
    errors.push(refused_with(
        &path,
        "max|f| under the sampled ends' targets",
        "a recorded fit's sampled ends give targets its samples cannot have",
    ));
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        fit_of(&mut m[measured]).lowered_error = None;
    });
    errors.push(refused_with(
        &path,
        "an in-range line's error erased",
        "a recorded line with no measured error is in the evaluator's range at both sampled ends",
    ));
    restore();

    // A refused line's fit hidden behind the faulting-input refusal, while the stage's other
    // subtrees keep their own records.
    let group = |at: usize| -> Vec<usize> {
        let m = &moves[at];
        (0..moves.len())
            .filter(|&i| {
                moves[i].id == m.id
                    && moves[i].round == m.round
                    && moves[i].parent_sha256 == m.parent_sha256
                    && line(&moves[i]).stage == line(m).stage
            })
            .collect()
    };
    assert!(group(refused).len() > 1, "the stage has one subtree");
    fn hide(m: &mut MoveRecord, refusal: &str) {
        let record = fit_of(m);
        record.fit = None;
        record.bound = None;
        record.line_nodes = None;
        record.lowered_error = None;
        record.expression = None;
        record.refusal = Some(refusal.into());
    }
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        hide(&mut m[refused], PREFIX_FAULTS)
    });
    errors.push(refused_with(
        &path,
        "a fit hidden behind a faulting input",
        "a faulting training input refuses only some of its stage's subtrees",
    ));
    restore();

    // Whole parent groups: removed, swapped with the next, or repeated.
    let starts: Vec<usize> = (0..moves.len())
        .filter(|&i| {
            moves[i].id == MINIMAX_LINEARIZE
                && (i == 0
                    || moves[i - 1].parent_sha256 != moves[i].parent_sha256
                    || moves[i - 1].round != moves[i].round
                    || moves[i - 1].id != moves[i].id)
        })
        .collect();
    let next_in_round = starts
        .windows(2)
        .find(|w| moves[w[0]].round == moves[w[1]].round)
        .map(|w| (w[0], w[1]))
        .expect("a round in which the pass read two parents");
    let (first, second) = next_in_round;
    let end = (second..moves.len())
        .find(|&i| {
            moves[i].parent_sha256 != moves[second].parent_sha256
                || moves[i].round != moves[second].round
                || moves[i].id != moves[second].id
        })
        .unwrap_or(moves.len());
    let off = "move records do not follow the ranked parents";
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        m.drain(first..second);
    });
    errors.push(refused_with(&path, "a parent's group removed", off));
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let later: Vec<MoveRecord> = m.drain(second..end).collect();
        for (k, record) in later.into_iter().enumerate() {
            m.insert(first + k, record);
        }
    });
    errors.push(refused_with(&path, "two parents' groups swapped", off));
    restore();
    // Repeated after the next parent's group (next to itself it would read as one group, which
    // the subtree count refuses).
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let copy: Vec<MoveRecord> = m[first..second].to_vec();
        for (k, record) in copy.into_iter().enumerate() {
            m.insert(end + k, record);
        }
    });
    errors.push(refused_with(&path, "a parent's group repeated", off));
    restore();

    // The documented limits: without the training rows, a closed fit scaled consistently (its
    // bracket ten times over, the bound recomputed; `max|f|` stays, since a target ten times
    // larger would force a residual the record does not have) and a refused line's fit hidden as
    // an unsampled subtree both re-derive the same decisions and are accepted.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_mut().unwrap();
        assert_eq!(
            fit.stop, "converged",
            "the scaled bracket keeps a converged stop closed"
        );
        fit.max_error *= 10.0;
        fit.levelled *= 10.0;
        record.bound = Some(bound_of(fit));
    });
    publication::verify_run(&path).unwrap();
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        hide(&mut m[refused], LINEARIZE_UNSAMPLED)
    });
    publication::verify_run(&path).unwrap();
    restore();
    // A third: an offered line's error edited up to its bound, a residual its line and samples
    // allow, keeps the decision and is accepted.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        assert!(record.lowered_error < record.bound);
        record.lowered_error = record.bound;
    });
    publication::verify_run(&path).unwrap();
    restore();
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

/// `check` reads the linearization pass's parents back from evidence, which is untrusted, so
/// `offers` and `assemble` (behind `propose` and `rederive`) parse them within
/// `parse_untrusted_pipeline`'s bounds (B17 review 02's class): the pipeline form only, and each
/// expression's nesting bytes counted before it is parsed. Unary plus never reaches the parsed
/// tree, so a parent at the limit offers what its plain form offers; one byte more offers nothing,
/// and `propose` refuses it unparsed.
#[test]
fn linearization_parents_are_bounded_before_they_are_parsed() {
    use crate::kernel2::refinement::MAX_PARENT_EXPRESSION_NESTING;
    let cut = cubic_cut();
    let opt = options(4, 64, linearize());
    let width = cut.train_x[0].len();
    // Built directly: `program` would print the padding away.
    let parent = |source: &str| format!("(pipe X (derive expr=\"{source}\") (ridge))");
    let held = CUBE
        .bytes()
        .filter(|b| matches!(b, b'(' | b'+' | b'-' | b'*' | b'/' | b'^'))
        .count();
    let room = MAX_PARENT_EXPRESSION_NESTING - held;
    let plus = |n: usize| parent(&format!("{}{CUBE}", "+".repeat(n)));
    assert!(linearize::offers(&opt.evaluator, &parent(CUBE)).unwrap());
    assert!(linearize::offers(&opt.evaluator, &plus(room)).unwrap());
    assert!(!linearize::offers(&opt.evaluator, &plus(room + 1)).unwrap());
    let error = linearize::propose(&opt.evaluator, &plus(room + 1), &cut, width).unwrap_err();
    assert!(
        format!("{error:#}").contains("nests too deeply"),
        "{error:#}"
    );
    let closed_form = format!("(sym raw \"{}c0\")", "+".repeat(room + 1));
    assert!(!linearize::offers(&opt.evaluator, &closed_form).unwrap());
    let error = linearize::propose(&opt.evaluator, &closed_form, &cut, width).unwrap_err();
    // Refused before any parse, by the pre-parse check's own message.
    assert!(
        format!("{error:#}").contains("not in the pipeline form"),
        "{error:#}"
    );
}

// Fix round r01 (B19 wiring review 01): repeated refused subtrees through publication.
