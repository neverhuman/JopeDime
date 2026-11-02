use super::*;

pub(super) fn finite(what: &str, value: f64) -> Result<f64> {
    ensure!(value.is_finite(), "{what} is not finite: {value}");
    Ok(value)
}

/// The median of `values` (the mean of the middle two for an even count).
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    })
}

/// Sample standard deviation (`n - 1`), or `None` for fewer than two values.
pub fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let squares: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    Some((squares / (values.len() - 1) as f64).sqrt())
}

pub(super) fn mean_of(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), v| (sum + v, count + 1));
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

pub(super) fn mean_of_options(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values: Option<Vec<f64>> = values.collect();
    let values = values?;
    (!values.is_empty()).then(|| mean_of(values.into_iter()))
}

/// M3: the 1-based number of evaluations after which the running best held-out RMSE has closed
/// `fraction` of the gap between the first measured candidate and the final best. A search that
/// never improves on its first measured candidate reports that candidate's position. `rmse` holds
/// one entry per evaluation, in order, `None` for a candidate that did not measure.
pub fn evaluations_to_fraction(rmse: &[Option<f64>], fraction: f64) -> Option<u64> {
    let first = rmse.iter().position(Option::is_some)?;
    let start = rmse[first]?;
    let final_best = rmse.iter().flatten().copied().fold(f64::INFINITY, f64::min);
    let gap = start - final_best;
    if gap <= 0.0 {
        return Some(first as u64 + 1);
    }
    let mut best = f64::INFINITY;
    for (index, value) in rmse.iter().enumerate() {
        if let Some(value) = value {
            if *value < best {
                best = *value;
            }
        }
        if start - best >= fraction * gap {
            return Some(index as u64 + 1);
        }
    }
    Some(rmse.len() as u64)
}

/// A program's stage count and the node count and depth of each of its expressions.
pub(super) fn program_shape(program: &str) -> Result<(usize, Vec<(usize, usize)>)> {
    let genome = GenomeIr::parse(program).with_context(|| format!("parse {program}"))?;
    Ok(match genome {
        GenomeIr::Pipeline(dsl) => {
            let mut expressions = Vec::new();
            for stage in &dsl.stages {
                if !matches!(stage.op.as_str(), "map" | "derive") {
                    continue;
                }
                expressions.push(match stage.params.get("expr") {
                    Some(ParamValue::Str(source)) => expression_size(
                        MathExpr::parse(source)
                            .with_context(|| format!("expression {source:?}"))?
                            .ast(),
                    ),
                    // A constant-only expression prints unquoted and parses back as a number.
                    Some(ParamValue::Int(_) | ParamValue::Float(_)) => (1, 1),
                    other => bail!("{program}: expression parameter {other:?}"),
                });
            }
            (dsl.stages.len(), expressions)
        }
        GenomeIr::Symbolic { expr, .. } => (1, vec![expression_size(expr.ast())]),
        GenomeIr::Portfolio(portfolio) => (
            portfolio.members.iter().map(|m| m.stages.len()).sum(),
            Vec::new(),
        ),
    })
}

/// `(nodes, depth)` of an expression, counted as the expression parser counts them.
pub fn expression_size(expr: &Expr) -> (usize, usize) {
    match expr {
        Expr::Const(_) | Expr::Var(_) => (1, 1),
        Expr::Unary(_, child) => {
            let (nodes, depth) = expression_size(child);
            (nodes + 1, depth + 1)
        }
        Expr::Binary(_, left, right) => {
            let (left_nodes, left_depth) = expression_size(left);
            let (right_nodes, right_depth) = expression_size(right);
            (
                left_nodes + right_nodes + 1,
                left_depth.max(right_depth) + 1,
            )
        }
        Expr::Call(_, args) => {
            let sizes: Vec<(usize, usize)> = args.iter().map(expression_size).collect();
            (
                sizes.iter().map(|(nodes, _)| nodes).sum::<usize>() + 1,
                sizes.iter().map(|(_, depth)| *depth).max().unwrap_or(0) + 1,
            )
        }
    }
}

pub(super) fn cost(program: &str) -> Cost {
    let unsupported = |reason: String| Cost {
        python_bytes: None,
        rust_bytes: None,
        python_tokens: None,
        rust_tokens: None,
        unsupported: Some(reason),
    };
    let genome = match GenomeIr::parse(program) {
        Ok(genome) => genome,
        Err(error) => return unsupported(format!("{error:#}")),
    };
    let options = EmitOptions::default();
    let emitted = match feat_math::transpile::emit(&genome, &options) {
        Ok(emitted) => emitted,
        Err(error) => return unsupported(format!("{error:#}")),
    };
    match feat_math::tokens::try_estimate_tokens(&genome, &options) {
        Ok(tokens) => Cost {
            python_bytes: Some(emitted.python.len()),
            rust_bytes: Some(emitted.rust.len()),
            python_tokens: Some(tokens.py),
            rust_tokens: Some(tokens.rs),
            unsupported: None,
        },
        Err(missing) => unsupported(missing.join("; ")),
    }
}

/// One program of a route's selection ranking, best first.
pub(super) struct Ranked<'a> {
    rmse: f64,
    r2: f64,
    cuts_measured: usize,
    program_sha256: &'a str,
    genome: &'a str,
}

/// Every metric of one run from its cuts' searches.
pub fn metrics(searches: &[CutSearch]) -> Result<Metrics> {
    let per_cut: Vec<(usize, &[Observation])> = searches
        .iter()
        .map(|s| (s.cut, s.observations.as_slice()))
        .collect();
    let selected = select::ranking(&per_cut)?;
    let ranking: Vec<Ranked<'_>> = selected
        .iter()
        .map(|b| Ranked {
            rmse: b.validation_rmse,
            r2: b.validation_r2,
            cuts_measured: b.cuts_measured,
            program_sha256: &b.program_sha256,
            genome: &b.genome,
        })
        .collect();
    // The search's own candidates, in evaluation order; comparison re-measurements excluded.
    let exploration: Vec<Vec<&Observation>> = searches
        .iter()
        .map(|s| {
            s.observations
                .iter()
                .filter(|o| o.origin != search::REMEASURE)
                .collect()
        })
        .collect();
    ranked_metrics(&ranking, &exploration)
}

/// Every metric of one kernel2 run. The ranking is the route's own (the evaluator's
/// fault-aware validation score, programs once each), so M1 is the program the route
/// publishes, measured on its one cut. The route has no comparison phase, so every measurement,
/// refinement children included, counts as exploration.
pub fn kernel2_metrics(
    options: &kernel2::search::Options,
    searched: &kernel2::search::CutSearch,
) -> Result<Metrics> {
    let mut seen = BTreeSet::new();
    let mut ranking = Vec::new();
    for measured in options.evaluator().rank(&searched.observations) {
        let o = &measured.observation;
        if !seen.insert(o.genome.as_str()) {
            continue;
        }
        ranking.push(Ranked {
            rmse: o
                .rmse
                .context("a ranked kernel2 measurement has no held-out RMSE")?,
            r2: o
                .hold_r2
                .context("a ranked kernel2 measurement has no held-out R2")?,
            cuts_measured: 1,
            program_sha256: &o.program_sha256,
            genome: &o.genome,
        });
    }
    ensure!(
        ranking.first().map(|b| b.genome) == Some(searched.best.observation.genome.as_str()),
        "the kernel2 ranking does not start with the program the search published"
    );
    let exploration = vec![searched
        .observations
        .iter()
        .map(|o| &o.observation)
        .collect()];
    ranked_metrics(&ranking, &exploration)
}

/// Every metric from a route's ranking and each cut's exploration observations, in evaluation
/// order.
pub(super) fn ranked_metrics(
    ranking: &[Ranked<'_>],
    exploration: &[Vec<&Observation>],
) -> Result<Metrics> {
    let best = ranking
        .first()
        .context("no program was measured on every cut")?;

    let top: Vec<&Ranked<'_>> = ranking.iter().take(TOP_MEDIAN).collect();
    let top_rmse: Vec<f64> = top.iter().map(|b| b.rmse).collect();
    let top_r2: Vec<f64> = top.iter().map(|b| b.r2).collect();

    let mut per_cut_speed = Vec::with_capacity(exploration.len());
    for observations in exploration {
        let rmse: Vec<Option<f64>> = observations
            .iter()
            .map(|o| if o.measured() { o.rmse } else { None })
            .collect();
        per_cut_speed.push(
            evaluations_to_fraction(&rmse, M3_FRACTION)
                .context("a cut measured no candidate in exploration")?,
        );
    }
    let attempted = exploration.iter().map(Vec::len).sum::<usize>() as u64;
    ensure!(attempted > 0, "the search attempted no candidate");
    let mut by_status: BTreeMap<String, u64> = search::STATUSES
        .iter()
        .filter(|status| **status != search::MEASURED)
        .map(|status| ((*status).to_owned(), 0))
        .collect();
    let mut valid_derived = 0u64;
    for observation in exploration.iter().flatten() {
        if observation.measured() {
            if DERIVED_ORIGINS.contains(&observation.origin) {
                valid_derived += 1;
            }
        } else {
            *by_status.entry(observation.status.to_owned()).or_default() += 1;
        }
    }
    let invalid: u64 = by_status.values().sum();

    let mut stages = Vec::new();
    let mut expressions = Vec::new();
    for program in ranking.iter().take(TOP_K) {
        let (count, sizes) = program_shape(program.genome)?;
        stages.push(count as f64);
        expressions.extend(sizes);
    }
    let nodes: Vec<f64> = expressions.iter().map(|(n, _)| *n as f64).collect();
    let depths: Vec<f64> = expressions.iter().map(|(_, d)| *d as f64).collect();

    Ok(Metrics {
        m1: Best {
            rmse: finite("M1 rmse", best.rmse)?,
            r2: finite("M1 r2", best.r2)?,
            cuts_measured: best.cuts_measured,
            program_sha256: best.program_sha256.to_owned(),
            program: best.genome.to_owned(),
        },
        m2: TopMedian {
            programs: top.len(),
            rmse: finite("M2 rmse", median(&top_rmse).context("empty ranking")?)?,
            r2: finite("M2 r2", median(&top_r2).context("empty ranking")?)?,
        },
        m3: Speed {
            mean: mean_of(per_cut_speed.iter().map(|v| *v as f64)),
            per_cut: per_cut_speed,
        },
        m4: None,
        m5: None,
        m6: Invalid {
            rate: invalid as f64 / attempted as f64,
            invalid,
            attempted,
            by_status,
        },
        m7: Shape {
            programs: stages.len(),
            mean_stages: mean_of(stages.iter().copied()),
            expression_stages: expressions.len(),
            mean_expression_nodes: (!nodes.is_empty()).then(|| mean_of(nodes.iter().copied())),
            mean_expression_depth: (!depths.is_empty()).then(|| mean_of(depths.iter().copied())),
            max_expression_nodes: expressions.iter().map(|(n, _)| *n).max(),
            max_expression_depth: expressions.iter().map(|(_, d)| *d).max(),
        },
        m8: cost(best.genome),
        m9: None,
        m11: Yield {
            per_evaluation: valid_derived as f64 / attempted as f64,
            valid_derived,
            evaluations: attempted,
        },
    })
}

// ------------------------------------------------------------------------------------------------
// Aggregates, gates and the verdict
// ------------------------------------------------------------------------------------------------
