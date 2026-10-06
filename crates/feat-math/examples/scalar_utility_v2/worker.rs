//! Equal-budget scalar utility worker using the existing fit and mutation routes.
//! Its arguments contain training rows only. Reporting rows never enter search.
use feat_math::evolve::FaultPolicy;
use feat_math::folds::fold_of;
use feat_math::kernel::admission::{outranks, FaultTotals};
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCandidate, ScalarExpressionProgram, ScalarLanguage, ScalarModel, ScalarNode,
    ScalarProducerProfile,
};
use feat_math::kernel::telemetry::StageFaultCounts;
use feat_math::math_expr::{FaultClass, FaultSet, FaultSeverity, Var};
use feat_math::rng::Pcg32;
use feat_math::{metrics, Rows};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const LAMBDAS: [f64; 3] = [1e-5, 0.02, 0.2];
const POPULATION: usize = 16;
const SEARCH_STREAM: u64 = 0x7363616c7574696c;

#[path = "shortest_controls.rs"]
mod shortest_controls;

#[derive(Clone)]
struct FitEvaluation<T> {
    candidate: T,
    lambda: f64,
    score: f64,
    faults: FaultTotals,
    fold_scores: Vec<f64>,
}

type Evaluation = FitEvaluation<ScalarCandidate>;

/// Actual fits/passes include the explicit telemetry witnesses below. The
/// witness and native fit/predict both evaluate their rows, so count both.
#[derive(Default)]
pub struct WorkCounts {
    attempted: usize,
    completed: usize,
    refused: usize,
    hard_rejected_attempts: usize,
    unobserved_internal_passes: usize,
    fit_attempts: usize,
    fits: usize,
    predictions: usize,
    evaluated_rows: usize,
    invalid_edits: usize,
    unchanged_edits: usize,
    edit_attempts: usize,
    birth_attempts: usize,
    rejected_births: usize,
    cells: FaultTotals,
}

fn stage(features: &Rows, masks: &[u32]) -> Result<StageFaultCounts, String> {
    if features.len() != masks.len() || features.iter().any(|row| row.len() != 1) {
        return Err("utility telemetry requires the Row carrier".into());
    }
    let mut stage = StageFaultCounts::default();
    stage.evaluated_cells = features.len() as u64;
    stage.constant_output = features.first().is_some_and(|first| {
        features
            .iter()
            .all(|row| row[0].to_bits() == first[0].to_bits())
    });
    for &bits in masks {
        let faults = FaultSet::from_bits(bits).ok_or("unknown utility fault bits")?;
        for class in FaultClass::ALL {
            stage.faulted_cells[class.index()] += u64::from(faults.contains(class.set()));
        }
        for (severity, count) in [
            (FaultSeverity::Domain, &mut stage.domain_cells),
            (FaultSeverity::Saturation, &mut stage.saturation_cells),
            (FaultSeverity::Hard, &mut stage.hard_rows),
        ] {
            *count += u64::from(faults.intersects(severity.classes()));
        }
    }
    Ok(stage)
}

fn add_repeated_passes(
    features: &Rows,
    masks: &[u32],
    evaluation: &mut FaultTotals,
    work: &mut WorkCounts,
) -> Result<(), String> {
    let counts = stage(features, masks)?;
    // One native internal pass and one actual explicit witness transform.
    for _ in 0..2 {
        evaluation.add_stage(&counts);
        work.cells.add_stage(&counts);
        work.evaluated_rows += features.len();
    }
    Ok(())
}

fn evaluate(
    candidate: &ScalarCandidate,
    key: &str,
    training: &Rows,
    labels: &[f64],
    policy: FaultPolicy,
    work: &mut WorkCounts,
) -> Result<Evaluation, String> {
    evaluate_folds(
        candidate,
        key,
        training,
        labels,
        policy,
        work,
        |tx, ty, lambda| {
            candidate
                .profile()
                .fit_training(candidate.program(), tx, ty, lambda)
                .map_err(|error| error.to_string())
        },
    )
}

// Formula references and generated candidates share folds, work accounting,
// objectives and selection. Only producer admission/binding differs.
fn evaluate_folds<T: Clone>(
    candidate: &T,
    key: &str,
    training: &Rows,
    labels: &[f64],
    policy: FaultPolicy,
    work: &mut WorkCounts,
    mut fit: impl FnMut(&Rows, &[f64], f64) -> Result<ScalarModel, String>,
) -> Result<FitEvaluation<T>, String> {
    let mut best: Option<FitEvaluation<T>> = None;
    let mut hard_lambda_refusal = false;
    for lambda in LAMBDAS {
        let mut scores = vec![];
        let mut faults = FaultTotals::default();
        for fold in 0..3 {
            let mut tx = vec![];
            let mut ty = vec![];
            let mut vx = vec![];
            let mut vy = vec![];
            for (index, (row, &label)) in training.iter().zip(labels).enumerate() {
                if fold_of(key, index, 3) == fold {
                    vx.push(row.clone());
                    vy.push(label);
                } else {
                    tx.push(row.clone());
                    ty.push(label);
                }
            }
            if tx.is_empty() || vx.is_empty() {
                return Err("empty utility inner fold".into());
            }
            // The producer-bound writer delegates to the generic fit route.
            // Any failed internal pass has unknown partial cell accounting;
            // keep that gap explicit instead of converting it to zero.
            work.fit_attempts += 1;
            let model = fit(&tx, &ty, lambda).map_err(|error| {
                work.unobserved_internal_passes += 1;
                error.to_string()
            })?;
            work.fits += 1;
            let (features, masks) = model.transform(&tx).map_err(|error| {
                work.unobserved_internal_passes += 1;
                error.to_string()
            })?;
            add_repeated_passes(&features, &masks, &mut faults, work)?;
            let (predictions, actual_masks) = model.predict(&vx).map_err(|error| {
                work.unobserved_internal_passes += 1;
                error.to_string()
            })?;
            work.predictions += 1;
            let (features, masks) = model.transform(&vx).map_err(|error| {
                work.unobserved_internal_passes += 1;
                error.to_string()
            })?;
            if actual_masks != masks {
                return Err("prediction telemetry witness disagrees".into());
            }
            add_repeated_passes(&features, &masks, &mut faults, work)?;
            scores.push(metrics::r2(&vy, &predictions));
        }
        if !matches!(policy, FaultPolicy::Legacy) && faults.is_hard() {
            hard_lambda_refusal = true;
            continue;
        }
        let scores = scores
            .into_iter()
            .map(|score| match policy {
                FaultPolicy::Penalize {
                    lambda_domain,
                    lambda_saturation,
                    ..
                } => faults.penalize(score, lambda_domain, lambda_saturation),
                _ => score.is_finite().then_some(score),
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("nonfinite utility objective")?;
        let score = scores.iter().sum::<f64>() / scores.len() as f64;
        let evaluated = FitEvaluation {
            candidate: candidate.clone(),
            lambda,
            score,
            faults,
            fold_scores: scores,
        };
        if best
            .as_ref()
            .is_none_or(|old| better(&evaluated, old, policy))
        {
            best = Some(evaluated);
        }
    }
    if best.is_none() && hard_lambda_refusal {
        work.hard_rejected_attempts += 1;
    }
    best.ok_or("all utility lambda fits refused".into())
}

fn better<T, U>(left: &FitEvaluation<T>, right: &FitEvaluation<U>, policy: FaultPolicy) -> bool {
    if matches!(
        policy,
        FaultPolicy::Penalize {
            tie_break: true,
            ..
        }
    ) {
        outranks(
            left.score,
            left.faults.faulted_cells(),
            right.score,
            right.faults.faulted_cells(),
        )
    } else {
        left.score > right.score
    }
}

fn shape(node: &ScalarNode) -> (usize, usize) {
    let children: Vec<&ScalarNode> = match node {
        ScalarNode::Unary(_, a) => vec![a],
        ScalarNode::Binary(_, a, b) => vec![a, b],
        ScalarNode::Call(_, args) => args.iter().collect(),
        _ => vec![],
    };
    children
        .into_iter()
        .map(shape)
        .fold((1, 1), |(n, d), (cn, cd)| (n + cn, d.max(cd + 1)))
}

fn counts(value: FaultTotals) -> Value {
    json!({"cells":value.evaluated_cells,"domain":value.domain_cells,
        "saturation":value.saturation_cells,"constant_passes":value.constant_passes,
        "hard_rows":value.hard_rows,"hard_classes":value.hard_classes.bits()})
}

pub struct SearchResult {
    pub model: ScalarModel,
    pub deterministic: Value,
}

fn usage(program: &ScalarExpressionProgram) -> [usize; 3] {
    fn visit(node: &ScalarNode, stable: bool, out: &mut [usize; 3]) {
        match node {
            ScalarNode::Var(Var::RowSum | Var::RowMean | Var::RowL1 | Var::RowL2) if stable => {
                out[2] += 1
            }
            ScalarNode::Unary(_, child) => visit(child, stable, out),
            ScalarNode::Binary(_, a, b) => {
                visit(a, stable, out);
                visit(b, stable, out);
            }
            ScalarNode::Call(function, args) => {
                match function.id() {
                    19 | 20 => out[0] += 1,
                    21 | 22 => out[1] += 1,
                    13 if stable => out[2] += 1,
                    _ => {}
                }
                for child in args {
                    visit(child, stable, out);
                }
            }
            _ => {}
        }
    }
    let mut out = [0; 3];
    // The recorded opt-in selector is itself the B07 capability, even for
    // a preserved body containing no stable-reduction or asinh leaf.
    out[2] = usize::from(program.numerical_policy() == NumericalPolicy::StableV2);
    visit(
        program.ast(),
        program.numerical_policy() == NumericalPolicy::StableV2,
        &mut out,
    );
    out
}

/// This identity contains only the worker training cut and inner-fold contract.
/// Dataset/report metadata and reporting labels cannot alter fold membership.
pub fn training_fold_key(rows: &Rows, labels: &[f64]) -> String {
    let identity = json!({
        "rows_bits":rows.iter().map(|r|r.iter().map(|v|v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "labels_bits":labels.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "cut":"training-only", "folds":3,
    });
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&identity).expect("integer bit identity"))
    )
}

/// Supplementary old-formula comparisons use the same training-only folds,
/// lambda grid, fit route and fault-policy comparator as the GP worker.
/// Their actual work is recorded separately; it never consumes or changes an
/// A/B search attempt, RNG stream, report row or report label.
pub fn compare_existing_formulas(
    card: usize,
    profile: &ScalarProducerProfile,
    training: &Rows,
    labels: &[f64],
    policy: FaultPolicy,
) -> Result<(Vec<(String, ScalarModel)>, Value), String> {
    policy.validate()?;
    if training.is_empty() || training.len() != labels.len() {
        return Err("formula-reference training shape".into());
    }
    let key = training_fold_key(training, labels);
    let mut work = WorkCounts::default();
    let mut models = vec![];
    let mut records = vec![];
    let mut best: Option<(usize, FitEvaluation<ScalarExpressionProgram>)> = None;
    for formula in shortest_controls::references(card, profile.input_width())? {
        work.attempted += 1;
        let before = work.fits;
        let (nodes, depth) = shape(formula.program.ast());
        if nodes > 15 || depth > 4 {
            return Err("formula-reference shape outside equal-budget limits".into());
        }
        // These are explicit old-primitive oracle programs, not producer
        // offspring. Frozen profiles exclude some of their old primitives;
        // record that refusal without expanding a catalogue or dropping a control.
        let producer_admission = profile
            .admit_candidate(formula.program.clone(), 0)
            .map(|_| "admitted".to_owned())
            .unwrap_or_else(|reason| format!("refused: {reason}"));
        let fit = |tx: &Rows, ty: &[f64], lambda| -> Result<ScalarModel, String> {
            let route =
                feat_math::kernel::wire_v2::VersionedProgram::ScalarV2(formula.program.clone());
            match route
                .fit_scalar(
                    tx,
                    ty,
                    feat_math::kernel::scalar_program::ScalarCarrier::Row,
                    lambda,
                )
                .map_err(|error| error.to_string())?
            {
                feat_math::kernel::wire_v2::VersionedModel::ScalarV2(model) => Ok(*model),
                _ => Err("formula-reference scalar fit misroute".into()),
            }
        };
        match evaluate_folds(
            &formula.program,
            &key,
            training,
            labels,
            policy,
            &mut work,
            fit,
        ) {
            Ok(evaluation) => {
                work.completed += 1;
                work.fit_attempts += 1;
                let model =
                    fit(training, labels, evaluation.lambda).map_err(|error| error.to_string())?;
                work.fits += 1;
                let (features, masks) = model.transform(training).map_err(|e| e.to_string())?;
                let mut refit_faults = FaultTotals::default();
                add_repeated_passes(&features, &masks, &mut refit_faults, &mut work)?;
                let (nodes, depth) = shape(formula.program.ast());
                let mut sources = vec![];
                for language in [
                    ScalarLanguage::Rust,
                    ScalarLanguage::Python,
                    ScalarLanguage::Cpp17,
                ] {
                    let (source, tokens) = formula
                        .program
                        .emit_recipe_counted(
                            language,
                            model.carrier(),
                            model.input_width(),
                            evaluation.lambda,
                        )
                        .map_err(|error| error.to_string())?;
                    sources.push(
                        json!({"language":format!("{language:?}"),"bytes":source.len(),
                        "tokens":tokens.code_tokens+tokens.doc_tokens,
                        "source_sha256":format!("{:x}",Sha256::digest(source.as_bytes()))}),
                    );
                }
                let index = records.len();
                records.push(json!({"name":formula.name,"compared_functions":formula.compared_functions,
                    "comparison_limits":formula.note,"producer_admission":producer_admission,"execution_route":"explicit-unbound-legacy-formula; generic scalar fit","nodes":nodes,"depth":depth,
                    "cv_bits":evaluation.score.to_bits(),"lambda_bits":evaluation.lambda.to_bits(),
                    "fold_score_bits":evaluation.fold_scores.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "faults":counts(evaluation.faults),"refit_faults":counts(refit_faults),
                    "fits":work.fits-before,"program":formula.program.to_bytes().map_err(|e|e.to_string())?,
                    "model_sha256":model.identity().map_err(|e|e.to_string())?,"sources":sources}));
                if best
                    .as_ref()
                    .is_none_or(|(_, old)| better(&evaluation, old, policy))
                {
                    best = Some((index, evaluation));
                }
                models.push((formula.name, model));
            }
            Err(reason) => {
                work.refused += 1;
                records.push(json!({"name":formula.name,"compared_functions":formula.compared_functions,
                    "comparison_limits":formula.note,"producer_admission":producer_admission,"execution_route":"explicit-unbound-legacy-formula; generic scalar fit","refusal":reason,"fits":work.fits-before}));
            }
        }
    }
    Ok((
        models,
        json!({"training_only_fold_key":key,
        "comparison_scope":"explicit minimal legacy formula references outside producer offspring; no exhaustive old-tree optimum claim",
        "producer_profile_unchanged":true,"controls_do_not_expand_generation_catalogues":true,
        "probit_reference":"logistic surrogate, no claimed inverse-CDF equivalence",
        "attempted":work.attempted,"completed":work.completed,"refused":work.refused,
        "hard_rejected_attempts":work.hard_rejected_attempts,
        "unobserved_internal_passes":work.unobserved_internal_passes,
        "cell_accounting_complete":work.unobserved_internal_passes == 0,
        "fit_attempts":work.fit_attempts,"fits":work.fits,"predictions":work.predictions,
        "evaluated_rows":work.evaluated_rows,"all_pass_faults":counts(work.cells),
        "best_formula_index":best.map(|(index, _)| index),"formulas":records}),
    ))
}

/// The reporter owns report rows; the worker receives only the training cut.
/// Each attempt consumes its budget even if growth/edit/fit is refused.
pub fn search(
    profile: &ScalarProducerProfile,
    training: &Rows,
    labels: &[f64],
    seed: u64,
    budget: usize,
    policy: FaultPolicy,
) -> Result<SearchResult, String> {
    policy.validate()?;
    if training.len() != labels.len() || training.is_empty() || budget < POPULATION {
        return Err("utility training shape or budget".into());
    }
    let key = training_fold_key(training, labels);
    let mut work = WorkCounts::default();
    let mut rng = Pcg32::new(seed, SEARCH_STREAM);
    let mut population: Vec<Evaluation> = vec![];
    let mut attempts = vec![];
    let mut progression: Vec<(usize, f64)> = vec![];
    for index in 0..budget {
        work.attempted += 1;
        let draw_seed = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
        let (candidate, edit) = if index < POPULATION || population.is_empty() || index % 8 == 0 {
            work.birth_attempts += 1;
            let candidate = profile.grow(draw_seed);
            let description = if let Ok(value) = &candidate {
                work.rejected_births += usize::from(value.proposal_rejected());
                let (drawn, admitted) = value.birth_draw();
                json!({"kind":"grow","opt_in_drawn":drawn,"opt_in_admitted":admitted,"proposal_rejected":value.proposal_rejected()})
            } else {
                work.rejected_births += 1;
                json!({"kind":"grow"})
            };
            (candidate, description)
        } else {
            work.edit_attempts += 1;
            let mut parent = rng.next_u32() as usize % population.len();
            for _ in 1..4 {
                let rival = rng.next_u32() as usize % population.len();
                if better(&population[rival], &population[parent], policy) {
                    parent = rival;
                }
            }
            match population[parent].candidate.mutate(draw_seed) {
                Ok(edit) => {
                    work.unchanged_edits += usize::from(!edit.changed);
                    work.invalid_edits += usize::from(edit.proposal_rejected);
                    let description = json!({"kind":"edit","parent":parent,
                        "arm":format!("{:?}",edit.arm),"changed":edit.changed,
                        "opt_in_drawn":edit.opt_in_drawn,"opt_in_admitted":edit.opt_in_admitted,"proposal_rejected":edit.proposal_rejected});
                    (Ok(edit.candidate), description)
                }
                Err(reason) => {
                    work.invalid_edits += 1;
                    (Err(reason), json!({"kind":"edit","parent":parent}))
                }
            }
        };
        let fits_before = work.fits;
        let candidate = candidate
            .and_then(|candidate| evaluate(&candidate, &key, training, labels, policy, &mut work));
        match candidate {
            Ok(evaluation) => {
                work.completed += 1;
                attempts.push(json!({"attempt":index+1,"seed":draw_seed,"edit":edit,
                    "program":evaluation.candidate.program().to_bytes().map_err(|e|e.to_string())?,
                    "lambda_bits":evaluation.lambda.to_bits(),"score_bits":evaluation.score.to_bits(),
                    "fold_score_bits":evaluation.fold_scores.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "faults":counts(evaluation.faults),"fits":work.fits-fits_before}));
                // Stable insertion uses the existing fault-policy comparator;
                // ties retain arrival order, without a non-transitive sort.
                let position = population
                    .iter()
                    .position(|old| better(&evaluation, old, policy))
                    .unwrap_or(population.len());
                population.insert(position, evaluation);
                population.truncate(POPULATION);
            }
            Err(reason) => {
                work.refused += 1;
                attempts.push(json!({"attempt":index+1,"seed":draw_seed,"edit":edit,
                    "refusal":reason,"fits":work.fits-fits_before}));
            }
        }
        if let Some(best) = population.first() {
            progression.push((index + 1, best.score));
        }
    }
    let winner = population
        .first()
        .ok_or("utility population has no fitted candidate")?;
    work.fit_attempts += 1;
    let model = profile
        .fit_training(winner.candidate.program(), training, labels, winner.lambda)
        .map_err(|error| error.to_string())?;
    work.fits += 1;
    let (features, masks) = model.transform(training).map_err(|e| e.to_string())?;
    let mut refit_faults = FaultTotals::default();
    add_repeated_passes(&features, &masks, &mut refit_faults, &mut work)?;
    let elite: Vec<_> = population.iter().take(10).collect();
    let mut scores: Vec<_> = elite.iter().map(|entry| entry.score).collect();
    scores.sort_by(f64::total_cmp);
    let top_ten_median = if scores.len() % 2 == 0 {
        (scores[scores.len() / 2 - 1] + scores[scores.len() / 2]) / 2.0
    } else {
        scores[scores.len() / 2]
    };
    // Ninety-five percent of improvement over the initial population's best;
    // multiplication of a possibly negative final R2 is not an improvement.
    let initial = progression
        .iter()
        .filter(|(n, _)| *n <= POPULATION)
        .map(|(_, v)| *v)
        .max_by(f64::total_cmp)
        .ok_or("no initial fitted population")?;
    let threshold = initial + 0.95 * (winner.score - initial);
    let evals_to_95 = progression
        .iter()
        .find(|(_, score)| *score >= threshold)
        .map(|(n, _)| *n)
        .ok_or("unreached utility threshold")?;
    // Retain W17's literal final-best threshold separately. It may be
    // unreachable for a negative final score; do not manufacture zero.
    let literal_evals_to_95 = progression
        .iter()
        .find(|(_, score)| *score >= 0.95 * winner.score)
        .map(|(n, _)| *n);
    let mut elite_metrics = vec![];
    for entry in &elite {
        let program: &ScalarExpressionProgram = entry.candidate.program();
        let (nodes, depth) = shape(program.ast());
        let mut sources = vec![];
        for language in [
            ScalarLanguage::Rust,
            ScalarLanguage::Python,
            ScalarLanguage::Cpp17,
        ] {
            let (source, tokens) = program
                .emit_recipe_counted(language, model.carrier(), model.input_width(), entry.lambda)
                .map_err(|e| e.to_string())?;
            sources.push(
                json!({"language":format!("{language:?}"),"bytes":source.len(),
                "tokens":tokens.code_tokens+tokens.doc_tokens,"source_sha256":format!("{:x}",Sha256::digest(source.as_bytes()))}),
            );
        }
        elite_metrics.push(json!({"nodes":nodes,"depth":depth,"required_features":program.required_features().map_err(|e|e.to_string())?,"card_token_occurrences":usage(program),"sources":sources}));
    }
    let deterministic = json!({"training_only_fold_key":key,"fold_assignments":(0..training.len()).map(|index|fold_of(&key,index,3)).collect::<Vec<_>>(),"training_rows":training.len(),"evaluator":"native-scalar-ridge-inner-cv-v1","profile":profile.canonical_json(),"profile_sha256":profile.sha256(),
        "seed":seed,"budget":budget,"attempted":work.attempted,"completed":work.completed,
        "refused":work.refused,"hard_rejected_attempts":work.hard_rejected_attempts,
        "unobserved_internal_passes":work.unobserved_internal_passes,
        "cell_accounting_complete":work.unobserved_internal_passes == 0,
        "invalid_offspring_proposals":work.rejected_births+work.invalid_edits,"fit_attempts":work.fit_attempts,"fits":work.fits,"predictions":work.predictions,
        "evaluated_rows":work.evaluated_rows,"all_pass_faults":counts(work.cells),
        "edit_attempts":work.edit_attempts,"birth_attempts":work.birth_attempts,"rejected_births":work.rejected_births,"invalid_edits":work.invalid_edits,"unchanged_edits":work.unchanged_edits,
        "best_cv_bits":winner.score.to_bits(),"top_ten_median_bits":top_ten_median.to_bits(),
        "evals_to_95_final_best":literal_evals_to_95,"evals_to_95_improvement":evals_to_95,"elite":elite_metrics,"attempts":attempts,
        "winner_json":model.to_json().map_err(|e|e.to_string())?,"winner_refit_faults":counts(refit_faults)});
    Ok(SearchResult {
        model,
        deterministic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::kernel::scalar_program::{ScalarCarrier, ScalarFunction};

    #[test]
    fn fault_penalties_never_replace_raw_cv_fold_scores() {
        let program = ScalarExpressionProgram::new(ScalarNode::Call(
            ScalarFunction::Log1mexp,
            vec![ScalarNode::Var(Var::Col(0))],
        ))
        .unwrap();
        let rows: Rows = (0..64)
            .map(|index| {
                vec![match index % 4 {
                    0 => -0.25,
                    1 => 1.25,
                    _ => (index + 1) as f64 / 65.0,
                }]
            })
            .collect();
        let labels: Vec<_> = (0..64)
            .map(|index| index as f64 * 0.5 + (index % 3) as f64)
            .collect();
        let key = training_fold_key(&rows, &labels);
        for policy in [
            FaultPolicy::Penalize {
                lambda_domain: 0.05,
                lambda_saturation: 0.0,
                tie_break: true,
            },
            FaultPolicy::Legacy,
        ] {
            let evaluation = evaluate_folds(
                &program,
                &key,
                &rows,
                &labels,
                policy,
                &mut WorkCounts::default(),
                |tx, ty, lambda| {
                    program
                        .fit_training(tx, ty, ScalarCarrier::Row, lambda)
                        .map_err(|error| error.to_string())
                },
            )
            .unwrap();
            assert!(evaluation.faults.domain_cells > 0);
            let mut raw = Vec::new();
            for fold in 0..3 {
                let mut tx = Vec::new();
                let mut ty = Vec::new();
                let mut vx = Vec::new();
                let mut vy = Vec::new();
                for (index, (row, &label)) in rows.iter().zip(&labels).enumerate() {
                    if fold_of(&key, index, 3) == fold {
                        vx.push(row.clone());
                        vy.push(label);
                    } else {
                        tx.push(row.clone());
                        ty.push(label);
                    }
                }
                let model = program
                    .fit_training(&tx, &ty, ScalarCarrier::Row, evaluation.lambda)
                    .unwrap();
                raw.push(metrics::r2(&vy, &model.predict(&vx).unwrap().0));
            }
            assert_eq!(
                evaluation
                    .fold_scores
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                raw.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "raw predictive CV scores must survive the separate selection penalty"
            );
            let objectives: Vec<_> = raw
                .iter()
                .map(|&score| match policy {
                    FaultPolicy::Penalize {
                        lambda_domain,
                        lambda_saturation,
                        ..
                    } => evaluation
                        .faults
                        .penalize(score, lambda_domain, lambda_saturation)
                        .unwrap(),
                    _ => score,
                })
                .collect();
            let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
            assert_eq!(evaluation.score.to_bits(), mean(&objectives).to_bits());
            if matches!(policy, FaultPolicy::Penalize { .. }) {
                assert!(evaluation.score < mean(&raw));
            } else {
                assert_eq!(evaluation.score.to_bits(), mean(&raw).to_bits());
            }
        }
    }
}
