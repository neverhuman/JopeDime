use super::*;

/// Why a public linearization entry refuses a tree outside the structure limits, before any copy
/// of it.
pub(crate) const STRUCTURE_REFUSAL: &str =
    "the linearization refuses a tree outside the structure limits before it is copied";

/// The bounded minimax linearization of `expression`'s one-variable subtrees; see
/// [`MathExpr::linearize_v2`]. `expression` may be any tree, one decoded from JSON included, and
/// the copy and the admission's normalization both recurse, so its borrowed structure check
/// (`from_ast`'s limits, walked with an explicit stack) comes first.
pub(crate) fn linearize(
    expression: &MathExpr,
    profile: &SearchProfile,
    scope: ExpressionScope,
    mut source: Source<'_>,
) -> Result<Vec<SubtreeLinearization>> {
    expression
        .check_structure(MAX_NODES, MAX_DEPTH)
        .map_err(|error| error.context(STRUCTURE_REFUSAL))?;
    let search = Search::new(profile, scope)?;
    let input = search.finish(expression.root.clone())?;
    let total = count_nodes(&input.root);
    let mut out = Vec::new();
    let mut shared = Shared::default();
    for (index, path) in linearizable(&input.root, search.scope.max_depth)
        .into_iter()
        .enumerate()
    {
        let node = at(&input.root, &path);
        let nodes = count_nodes(node);
        let mut found = Vec::new();
        variables(node, &mut found);
        let x = Expr::Var(found[0]);
        let mut record = SubtreeLinearization {
            path: path.clone(),
            subtree: MathExpr::from_ast(node.clone())?,
            nodes,
            fit: None,
            line_nodes: None,
            lowered_error: None,
            expression: None,
            refusal: None,
        };
        // A recorded line in the evaluator's range at both sampled ends, with no measured error.
        let mut unmeasured_in_range = false;
        // The fit, and how the line's own error over the samples is measured.
        let (fit, measured): (LinearFit, Measure) = match &mut source {
            Source::Rows(rows) => {
                let Some((_, points)) = samples(&record.subtree, rows) else {
                    record.refusal = Some(LINEARIZE_UNSAMPLED.into());
                    out.push(record);
                    continue;
                };
                let fit = match minimax(&points, LINEARIZATION_DEGREE) {
                    Ok(fit) => fit,
                    Err(reason) => {
                        record.refusal = Some(format!("{LINEARIZE_FIT_REFUSED}: {reason}"));
                        out.push(record);
                        continue;
                    }
                };
                let unique = prepare(&points).map_err(anyhow::Error::msg)?;
                let line = fit.monomial();
                let public = LinearFit {
                    stop: fit.stop.public(),
                    coefficients: [line[0], line[1]],
                    lo: fit.lo,
                    hi: fit.hi,
                    levelled: fit.levelled,
                    max_error: fit.max_error,
                    max_abs: largest(&unique),
                    iterations: fit.iterations,
                    steps: fit.steps,
                };
                (
                    public,
                    Box::new(move |coefficients: &[f64]| lowered_error(coefficients, &unique)),
                )
            }
            Source::Recorded(recorded) => {
                let evidence = recorded(index)?;
                shared.check(node, found[0], &evidence)?;
                let Some(fit) = evidence.fit else {
                    anyhow::ensure!(
                        evidence.lowered_error.is_none() && evidence.refusal.is_some(),
                        "a linearization without a fit names no refusal"
                    );
                    // The live path refuses a subtree before any fit only as unsampled, for a
                    // reason the fit itself gives, or, from its caller, because the input rows
                    // could not be built (every subtree of the expression or none: `Shared`).
                    let refusal = evidence.refusal.as_deref().unwrap_or_default();
                    let reason = refusal
                        .strip_prefix(LINEARIZE_FIT_REFUSED)
                        .and_then(|r| r.strip_prefix(": "));
                    anyhow::ensure!(
                        refusal == LINEARIZE_UNSAMPLED
                            || refusal == LINEARIZE_INPUT_FAULTS
                            || reason.is_some_and(|r| FIT_REFUSALS.contains(&r)),
                        "a linearization without a fit names a refusal the fit never gives"
                    );
                    // The element `x` faults on every row, so a subtree of it is unsampled, or
                    // has too few values when there are no rows at all.
                    anyhow::ensure!(
                        found[0] != Var::X
                            || refusal == LINEARIZE_UNSAMPLED
                            || refusal == LINEARIZE_INPUT_FAULTS
                            || reason == Some(FEW_ABSCISSAE),
                        "a subtree of the element x names a refusal its samples never give"
                    );
                    record.refusal = evidence.refusal;
                    out.push(record);
                    continue;
                };
                anyhow::ensure!(consistent(&fit), "a recorded fit disagrees with its stop");
                // The live samples read each row's variable, and the element `x` faults on every
                // row, so a subtree of `x` is never sampled and has no fit.
                anyhow::ensure!(
                    found[0] != Var::X,
                    "a recorded fit names a subtree of the element x, which no training row samples"
                );
                // A subtree that is zero wherever it is finite gives every sample a zero target,
                // so its fit's largest `|f|` is zero, and `consistent` then holds the fit to the
                // zero polynomial, exact at the first step.
                anyhow::ensure!(
                    fit.max_abs == 0.0 || !zero_wherever_finite(node),
                    "a recorded fit gives a subtree that is zero on every sample a nonzero target"
                );
                // A row's L1 and L2 norms and its standard deviation are never negative, so a
                // fit of one has a range that is not either.
                anyhow::ensure!(
                    !matches!(found[0], Var::RowL1 | Var::RowL2 | Var::RowStd) || fit.lo >= 0.0,
                    "a recorded fit's range holds values its variable never takes"
                );
                // No target is larger than every value the subtree can take.
                anyhow::ensure!(
                    fit.max_abs <= largest_magnitude(node),
                    "a recorded fit's max|f| exceeds every value its subtree can take"
                );
                // `lo` and `hi` are sampled values of the subtree's one variable, so its targets
                // there follow from the subtree itself: fault-free, finite, and within `max|f|`.
                let mut targets = [0.0f64; 2];
                for (target, end) in targets.iter_mut().zip([fit.lo, fit.hi]) {
                    *target = target_at(node, found[0], end)
                        .filter(|f| f.abs() <= fit.max_abs)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "a recorded fit's sampled ends give targets its samples cannot have"
                            )
                        })?;
                }
                // A subtree with one value on every sample has targets of the size its target at
                // `lo` shows, so `max|f|` is that size. The first reference's solve, whose first
                // column is all ones, then gives that value at once with a zero slope and no
                // error, so the fit is exact at the first step, its coefficients the conversion
                // of that line on the recorded range, exactly ([`constant_line`]); a singular
                // first solve gives no fit.
                let one_value = one_value_wherever_finite(node);
                if one_value {
                    anyhow::ensure!(
                        fit.max_abs == targets[0].abs()
                            && fit.stop == MinimaxStop::ExactFit
                            && fit.iterations == 1
                            && fit.steps == 1
                            && fit.max_error == 0.0
                            && constant_line_recorded(fit.coefficients, targets[0], fit.lo, fit.hi),
                        "a recorded fit of a subtree with one value on every sample is not that value's exact fit"
                    );
                }
                let lowered = evidence.lowered_error;
                // A line that can be lowered and is in the evaluator's range at both sampled ends
                // is in it at every sample between them, so the live measurement gives its error.
                let lowerable = fit.coefficients.iter().all(|c| c.is_finite());
                let ends = [fit.lo, fit.hi].map(|end| horner(&fit.coefficients, end));
                unmeasured_in_range = lowered.is_none()
                    && lowerable
                    && ends.iter().all(|v| v.is_finite() && v.abs() <= MAX_ABS);
                // The line's error is its largest absolute residual over the samples: a recorded
                // one that is negative, negative zero or not finite cannot have been measured.
                anyhow::ensure!(
                    lowered.is_none_or(|error| error.is_finite() && error.is_sign_positive()),
                    "a recorded line error is not a finite nonnegative residual"
                );
                if let Some(error) = lowered {
                    // The live measurement keeps a line only while it is in the evaluator's range
                    // at every sample, `lo` and `hi` are samples, and every target is within
                    // `max|f|`, which one of them attains. So every residual is at most
                    // `MAX_ABS + max|f|`, and the one at either end is `|p - f|` against that
                    // end's own target, which is within `max|f|`, so it is also at least
                    // `|p| - max|f|`. The line's computed value, a rounded product by the slope
                    // and then a rounded sum with the intercept, is monotone in the abscissa, so
                    // between the samples it lies between its values at the ends: with `M` the
                    // larger of their sizes, every residual is at most `M + max|f|`, and the one
                    // at the largest target at least `max|f| - M`. Rounding is monotone, so each
                    // bound holds for the computed values exactly, with no allowance. A line whose
                    // coefficients are not finite is refused before it is measured, as for every
                    // refusal before the measurement, so its ends are not read.
                    anyhow::ensure!(
                        error <= MAX_ABS + fit.max_abs,
                        "a recorded line error exceeds what the evaluator's range allows"
                    );
                    if lowerable {
                        let (mut largest_end, mut largest_residual) = (0.0f64, 0.0f64);
                        for (value, target) in ends.into_iter().zip(targets) {
                            anyhow::ensure!(
                                value.is_finite() && value.abs() <= MAX_ABS,
                                "a recorded line error names a line outside the evaluator's range at a sampled end"
                            );
                            // The residual at the sampled end itself, as the measurement takes it.
                            anyhow::ensure!(
                                error >= (value - target).abs(),
                                "a recorded line error is smaller than the line's residual at a sampled end's target"
                            );
                            largest_end = largest_end.max(value.abs());
                            largest_residual = largest_residual.max((value - target).abs());
                        }
                        // Against one value, the residual of a line monotone between its ends is
                        // largest at an end, and the measurement takes the ends: the error is
                        // exactly the larger end residual.
                        anyhow::ensure!(
                            !one_value || error == largest_residual,
                            "a recorded line error is not the larger end residual of a line against its subtree's one value"
                        );
                        anyhow::ensure!(
                            error <= largest_end + fit.max_abs,
                            "a recorded line error exceeds what its line and samples allow"
                        );
                        anyhow::ensure!(
                            error >= fit.max_abs - largest_end,
                            "a recorded line error is smaller than the residual at the largest target"
                        );
                    }
                }
                (fit, Box::new(move |_: &[f64]| lowered))
            }
        };
        record.fit = Some(fit);
        if !fit.stop.closed() {
            record.refusal = Some(format!("{LINEARIZE_OPEN_BRACKET}: {}", fit.stop.as_str()));
            out.push(record);
            continue;
        }
        // The node limit the rest of the expression leaves the line.
        let budget = search.scope.max_nodes - (total - nodes);
        match assess(
            &fit.coefficients,
            fit.max_error,
            fit.max_abs,
            &x,
            node,
            budget,
            measured,
        ) {
            Err(refused) => {
                // The measurement refuses a line as out of range only when some sample leaves the
                // range, which a line in range at both sampled ends cannot.
                anyhow::ensure!(
                    !(unmeasured_in_range && refused.reason == LINEARIZE_OUT_OF_RANGE),
                    "a recorded line with no measured error is in the evaluator's range at both sampled ends"
                );
                record.line_nodes = refused.nodes;
                record.lowered_error = refused.lowered_error;
                record.refusal = Some(refused.reason.into());
            }
            Ok(proposal) => {
                record.line_nodes = Some(proposal.nodes);
                record.lowered_error = Some(proposal.lowered_error);
                let mut root = input.root.clone();
                replace(&mut root, &path, proposal.tree);
                match search.finish(root) {
                    Ok(expression) => record.expression = Some(expression),
                    Err(error) => {
                        record.refusal = Some(format!("{LINEARIZE_PROFILE_REFUSED}: {error:#}"))
                    }
                }
            }
        }
        out.push(record);
    }
    Ok(out)
}

impl LinearFit {
    /// The minimax bound a line is held to on the training samples: `max_error (1 + 1e-6) + 64 ε
    /// max_abs`. The line's own largest error over those samples, in its own evaluation order,
    /// must not exceed it. Both are floating-point residuals, not certified bounds. The bound is
    /// the fit's own training error with room for rounding, so it catches a lowering that lost
    /// precision, not a poor line: the native score of the whole program judges the line.
    pub fn bound(&self) -> f64 {
        allowance(self.max_error, self.max_abs)
    }
}

impl MathExpr {
    /// The paths of the subtrees [`MathExpr::linearize_v2`] considers, in its order, from the
    /// expression's text alone: each reads exactly one variable, has more than three nodes, and
    /// sits where a line of three nodes and depth two fits the scope's depth limit. A tree outside
    /// the structure limits is refused before anything copies or walks it.
    pub fn linearizable_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
    ) -> Result<Vec<Vec<usize>>> {
        self.check_structure(MAX_NODES, MAX_DEPTH)
            .map_err(|error| error.context(STRUCTURE_REFUSAL))?;
        let search = Search::new(profile, scope)?;
        let input = search.finish(self.root.clone())?;
        Ok(linearizable(&input.root, search.scope.max_depth))
    }

    /// A bounded minimax linearization, degree 1 only, of the expression's one-variable subtrees
    /// on the given training rows. It draws no random number, and reads no row but `rows`.
    ///
    /// `self` must pass the same admission as every search edit (`Search::finish`: the pool, the
    /// caps and the negation rule), or this is an error. For each subtree of
    /// [`MathExpr::linearizable_v2`], in that order:
    /// 1. it is sampled on `rows` as `(x, f(x))` pairs of its variable and its value; a row that
    ///    faults, or gives a value that is not finite, refuses the subtree;
    /// 2. the discrete minimax line on those samples is fitted; a fit that refuses its samples
    ///    (fewer than three distinct values of the variable, say) refuses the subtree;
    /// 3. a fit whose bracket did not close ([`MinimaxStop::closed`]) refuses it: its error need
    ///    not be the minimax error;
    /// 4. the line is lowered to ordinary nodes, `x * a1 + a0`, and refused unless it is smaller
    ///    than the subtree, within the node limit the rest of the expression leaves it, and its
    ///    own largest error over the samples, in its own evaluation order, is within
    ///    [`LinearFit::bound`];
    /// 5. the line replaces the subtree, and the profile must admit the whole expression.
    ///
    /// Every considered subtree is returned, with the expression or the refusal. The minimax
    /// bound holds only on the training samples: off that range the line can be far from the
    /// subtree, so the caller re-scores each expression natively on its own data, and that score
    /// decides, never the line's error.
    pub fn linearize_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        rows: &[Vec<f64>],
    ) -> Result<Vec<SubtreeLinearization>> {
        linearize(self, profile, scope, Source::Rows(rows))
    }

    /// What [`MathExpr::linearize_v2`] returns, re-derived from a record of what it measured on
    /// the training rows: `recorded(i)` gives the `i`-th considered subtree's
    /// [`LinearizationEvidence`]. The subtrees, the bracket rule, the lowered line and its size,
    /// the bound, every refusal after the fit, and the admitted expression are recomputed. The
    /// fit itself and the line's measured error are taken as recorded, checked only against the
    /// fit's stop and the evaluator's range, the subtree's own targets at the sampled ends, what the
    /// subtree's structure fixes without the rows (a `max|f|` within the subtree's bound; for a
    /// subtree that is zero wherever it is finite, a zero `max|f|`; for one with a single value on
    /// every sample, that value's exact first-step fit and a line error equal to the larger end
    /// residual; a range within its variable's values; no fit, and only its own refusals, for a
    /// subtree of the element `x`), the error as a finite nonnegative residual within what the recorded line and
    /// samples allow (at most `1e12 + max|f|`; for a line that can be lowered, at least its
    /// residual at either sampled end and against that end's target, and between `max|f| - M` and
    /// `M + max|f|`, with `M` its larger size there; present whenever such a line in range at both
    /// ends reaches the measurement), a refusal before any fit as unsampled, one the fit gives, or
    /// [`LINEARIZE_INPUT_FAULTS`] for every subtree of the expression, and against the
    /// expression's other records (a repeated subtree records the same samples, bit for bit; the
    /// fits of one variable share its training range); without the rows they cannot be
    /// recomputed. A subtree with no recorded fit keeps its recorded refusal.
    pub fn linearize_recorded_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        recorded: &mut dyn FnMut(usize) -> Result<LinearizationEvidence>,
    ) -> Result<Vec<SubtreeLinearization>> {
        linearize(self, profile, scope, Source::Recorded(recorded))
    }
}
